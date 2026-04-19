import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.minimind import MinimindConfig, MinimindForCausalLM
from dataset.datasets import RLAIFDataset
from trainer.train_utils import Logger, is_main_process, load_ckp, init_distributed_mode, set_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, RolloutEngine

warnings.filterwarnings('ignore')

class CriticModel(MinimindForCausalLM):
    def __init__(self, config = None):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.post_norm(outputs[0])
        score = self.value_head(hidden_states).squeeze(-1)
        return score

def rep_penalty(text, n=3, cap=0.5):
    # 将单词和标点符号分开，并转换为小写
    toks = re.findall(r"\w+|[^\w\s]", text.lower())

    # 生成n-gram
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]

    # set去重，相减得到重复的数量，/len()归一化防止长文本
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    '''
    计算response的奖励
    return (len(responses), )的tensor
    奖励计算分为两部分
    1. 规则奖励：对responses的长度进行约束
    2. reward model：由reward model打分
    '''
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_score = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{'role': role, 'content': content.strip()} for role, content in matches]
            answer = response
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)

            score = reward_model.get_score(messages, answer)
            reward_model_score.append(score)

        reward_model_score = torch.tensor(reward_model_score, device=args.device)
        rewards += reward_model_score
    return rewards

def ppo_train_epoch(epoch, dataloader, iters, rollout_engine: RolloutEngine, ref_model, reward_model, actor_scheduler, critic_scheduler, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0
    
    for step, batch in enumerate(dataloader, start=start_step + 1):
        prompts = batch['prompt'] # (bs, )
        prompt_ids = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
            padding_side="left").to(args.device)
        prompt_len = prompt_ids.input_ids.shape[1]
        
        rollout_res = rollout_engine.rollout(
            prompt_ids=prompt_ids.input_ids, 
            attention_mask=prompt_ids.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8
        )

        gen_out = rollout_res.output_ids  # (bs, gen_len)
        responses_text = rollout_res.completions
        rewards = calculate_rewards(prompts, responses_text, reward_model)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_len}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        full_mask = (gen_out != tokenizer.pad_token_id).long() # 生成的有效位置
        labels = gen_out[:, 1:].clone() # 整体labels
        slen, resp_start = labels.shape[1], prompt_len - 1
        resp_mask = torch.arange(slen, device=gen_out.device).unsqueeze(0) >= resp_start # labels中的response部分
        final_mask = (resp_mask & ~(labels.eq(tokenizer.pad_token_id))).float() # 最终用于loss计算的mask，response部分且非pad

        resp_labels = labels[:, resp_start:] # response的lebel
        resp_idx = torch.arange(resp_labels.shape[1], device=gen_out.device).unsqueeze(0) # response的token位置索引
        resp_pad_mask = ~resp_labels.eq(tokenizer.pad_token_id) # 非pad位置
        resp_label_len = resp_pad_mask.sum(dim=1) # response label长度
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask # response中eos位置
        has_eos = eos_mask.any(dim=1) # 是否有eos
        eos_pos = torch.argmax(eos_mask.int(), dim=1) # 每个batch的eos位置
        resp_len = torch.where(has_eos, eos_pos + 1, resp_label_len).long().clamp(min=1) # response长度（包含eos）
        resp_policy_mask = ((resp_idx < resp_len.unsqueeze(1)) & resp_pad_mask).float() # 用于policy loss计算的mask，response部分且非pad且在eos位置前
        resp_value_mask = resp_policy_mask.clone() # 用于value loss计算的mask，response部分且非pad且在eos位置前

        with torch.no_grad():
            # 计算Vold
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq[:, resp_start:-1] * resp_value_mask
            
            # 计算logπ(a_t | s_t)
            actor_for_rollout = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            with autocast_ctx:
                logits = actor_for_rollout(input_ids=gen_out, attention_mask=full_mask).logits
            old_resp_logp = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)[:, resp_start:]

            # 计算logπref(a_t | s_t)
            ref_logp_all = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[full_mask:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            ref_resp_logp = ref_logp_all[:, resp_start:]
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_label_len - 1
            bs = len(prompts)
            token_rewards[torch.arange(bs, device=args.device), last_idx] += rewards

            # 计算GAE优势
            gen_len = old_resp_values.shape[1]
            last_gae = torch.zeros(bs, device=args.device)
            advs_reversed = []
            for t in reversed(range(gen_len)):
                # 循环里是在计算每个batch的t时刻的优势, 而不是一个batch一个batch地计算
                V_old_next = old_resp_values[:, t + 1] if t + 1 < gen_len else 0.0
                delta = token_rewards[:, t] + args.gamma * V_old_next - old_resp_values[:, t]
                last_gae = delta + args.gamma * args.lam * last_gae
                advs_reversed.append(last_gae)
            advantages = torch.stack(advs_reversed[::-1], dim=1) # (B, R)
            returns = advantages + old_resp_values

            # 归一化优势
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) / torch.sqrt(adv_var + 1e-8) * resp_policy_mask

        mb_size = max(1, min(args.mini_batch_size, bs))
        stop_ppo = False # 用于早停ppo，当KL散度过大时早停
        policy_loss_sum = 0.0
        clip_frac_sum = 0.0 # 被clip的数量
        value_loss_sum = 0.0
        aux_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        log_count = 0
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break

            batch_indices = torch.randperm(bs, device=args.device)
            for i in range(0, bs, mb_size):
                indices = batch_indices[i: i + mb_size]
                with autocast_ctx:
                    actor_output = actor_unwrapped(input_ids=gen_out[indices], attnention_mask=full_mask[indices])
                    aux_loss = actor_output.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                
                mb_logp_all = F.log_softmax(actor_output.logits[:, :-1], dim=-1).gather(2, labels[[indices]].unsqueeze(-1)).squeeze(-1)
                mb_resp_logp = mb_logp_all[:, resp_start:]

                log_ratio = mb_resp_logp - old_resp_logp[indices]
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[indices]).sum() / resp_policy_mask[indices].sum().clamp(min=1) # KL散度的二阶近似

                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                
                ratio = torch.exp(log_ratio)
                clip_frac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[indices]).sum()
                            / resp_policy_mask[indices].sum().clamp(min=1))
                kl_ref_penalty = ((torch.exp(ref_resp_logp[indices] - mb_resp_logp) - (ref_resp_logp[indices] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[indices]).sum() / resp_policy_mask[indices].sum().clamp(min=1)
                policy_loss = ((torch.max(
                    -advantages * ratio,
                    -advantages * torch.clamp(ratio, min=1.0-args.clip_epsilon, max=1.0 + args.clip_epsilon)
                ) * resp_policy_mask[indices]).sum / resp_policy_mask[indices].sum().clamp(min=1)
                + args.kl_eof * kl_ref_penalty)

                mb_values_seq = critic_unwrapped(input_ids=gen_out[indices], attention_mask=full_mask[indices])
                mb_resp_values = mb_values_seq[:, resp_start:-1]
                value_loss = 0.5 * (torch.max(
                    (mb_resp_values - returns[indices]) ** 2
                    (torch.clamp(mb_resp_values, min=old_resp_values[indices]-args.clip_epsilon, max=old_resp_values[indices]+args.clip_epsilon) - returns[indices]) ** 2
                ) * resp_value_mask[indices]).sum() / resp_value_mask[indices].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                aux_loss_sum += aux_loss.item()
                clip_frac_sum += clip_frac.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                log_count += 1
                grad_accum_step += 1

                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
        
        # 更新参数
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        
        # 日志输出
        if is_main_process() and step % args.log_interval == 0:
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clip_frac_sum / max(log_count, 1)
            avg_len_val = resp_len.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            rollout_engine.update_policy(actor_model)

            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            load_ckp(lm_config, prefix=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, responses_text, rewards, full_mask, values_seq, advantages
        del logits, labels, final_mask, resp_labels, resp_idx, resp_pad_mask, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_logp_all, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8997", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # 1. 初始化随机种子、分布式环境初始化
    local_rank = init_distributed_mode()
    args.device = f'cuda:{local_rank}' if dist.is_initialized() else args.device
    set_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2. 模型配置
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MinimindConfig(args.hidden_size, args.num_hidden_layers, bool(args.use_moe))
    ckp_data = load_ckp(lm_config=lm_config, prefix=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # 3. 混合精度设置
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    # 4. 日志配置
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 初始化模型
    actor_model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config=lm_config, from_weight=args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    critic_model = CriticModel(lm_config).to(args.device)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    rollout_engine = create_rollout_engine(args.rollout_engine, actor_model, tokenizer, args.device, autocast_ctx)
    train_ds = RLAIFDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    mb_factor = max(1, (args.batch_size + args.mini_batch_size - 1) // args.mini_batch_size)
    total_optimizer_steps = math.ceil(iters * args.epochs * mb_factor * args.ppo_update_iters / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate * 0.1)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate * 0.1)

    # 6. ckp加载
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['actor_model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['actor_optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['actor_scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    # 7. 编译、分布式包装
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    if is_main_process(): rollout_engine.update_policy(actor_model)

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        set_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        dataloader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, dataloader, len(dataloader) + skip, rollout_engine, ref_model, reward_model, actor_scheduler, critic_scheduler, start_step, wandb)
        else:
            ppo_train_epoch(epoch, dataloader, len(dataloader), rollout_engine, ref_model, reward_model, actor_scheduler, critic_scheduler, 0, wandb)

    # 9. 清理分布式环境
    if dist.is_initialized(): dist.destroy_process_group()