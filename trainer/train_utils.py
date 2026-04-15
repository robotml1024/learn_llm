import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import random
import math
from model.minimind import MinimindConfig, MinimindForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def Logger(s):
    print(s)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_ckp(lm_config: MinimindConfig, epoch=0, step=0, model=None, optimizer=None, wandb=None, prefix='full_sft', save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{prefix}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{prefix}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
    return None

def get_model_param(config: MinimindConfig, model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_expert = getattr(config, 'num_experts', 0)
    n_active = getattr(config, 'num_experts_per_tok', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0' in n)
    base = total - (n_expert * expert) # 永远要走的部分
    active = base + (n_active * expert) # moe激活的部分
    Logger(f'total parameters: {total:.2f}M, active parameters: {active:.2f}M')

def init_model(lm_config: MinimindConfig, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MinimindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_path = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_path}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    get_model_param(lm_config, model)
    Logger(f'trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, bs, skip_batches):
        self.sampler = sampler
        self.bs = bs
        self.skip_batches = skip_batches

    def __iter__(self):
        skip_batches = 0
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.bs:
                if self.skip_batches > skip_batches:
                    skip_batches += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skip_batches >= self.skip_batches:
            yield batch
            

    def __len__(self):
        total_batches = (len(self.sampler) + self.bs - 1) // self.bs
        return max(0, total_batches - self.skip_batches)