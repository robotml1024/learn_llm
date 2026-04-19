from torch.utils.data import Dataset
import torch
import random
import json

from datasets import load_dataset, Features, Sequence, Value

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_ids = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length-2, truncation=True).input_ids
        input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    

def pre_process_chat(conversations, add_system_ratio=0.2):
    # tool use数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_process_chat(prompt, empty_think_ratio):
    if '<think>\n\n</think>\n\n' in prompt and random.random() > empty_think_ratio:
        prompt = prompt.replace('<think>\n\n</think>\n\n', '')
    return prompt

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversation': [
            {'role': Value('string')},
            {'content': Value('string')},
            {'reasoning_content': Value('string')},
            {'tools': Value('string')},
            {'tool_calls', Value('string')}
        ]})
        self.data = load_dataset('json', data_file=data_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.data)
    
    def create_chat_prompt(self, conversations):
        messages = []
        for message in conversations:
            message = dict(message)
            if message.get('role') == 'system' and message.get('tools'):
                tools = json.loads(message['tools']) if isinstance(message['tools'], str) else message['tools']
            if message.get('tool_calls') and isinstance(message['tool_calls'], str):
                message['tool_calls'] = json.loads(message['tool_calls'])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, end + 1):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < self.eos_id else len(input_ids)
            else:
                i += 1
        return labels
    
    def __getitem__(self, idx):
        data = self.data[idx]
        conversations = pre_process_chat(data['conversation']) # 加上system prompt
        prompt = self.create_chat_prompt(conversations) # 将tools相关的内容从str转为json格式
        prompt = post_process_chat(prompt) # 删除空思考
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids) # 生成label
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.data = load_dataset('json', data_file=data_path, split='train')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        chosen = sample['chosen']
        rejected = sample['rejected']
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenizer=False, add_generation_prompt=False
        )
        chosen_prompt = post_process_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenizer=False, add_generation_prompt=False
        )
        rejected_prompt = post_process_chat(rejected_prompt)

        chosen_input_ids = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )['input_ids']
        rejected_input_ids = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )['input_ids']

        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                end = min(end + len(self.eos_id), len(input_ids))
                for j in range(start, end):
                    loss_mask[j] = 1
                i = end
            else:
                i += 1
        return loss_mask
    

class RLAIFDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, think_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.think_ratio = think_ratio
        self.samples = load_dataset('json', data_file=data_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_process_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }