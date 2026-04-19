from dataclasses import dataclass
from torch import Tensor
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from torch.nn.parallel import DistributedDataParallel

@dataclass
class RolloutResult:
    output_ids: Tensor # 包含 prompt + response 的完整id
    completion_ids: Tensor # 仅包含 response 的id
    per_token_logps: Tensor # 每个生成token的log概率，就是logπ(a_t|s_t)
    completions: List[str] # response文本形式

class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass

def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    
    # (bs, slen)
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids

    # (bs, slen, vocab_size)
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []

    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps) # 将二维list转化为(bs, n_keep)形状的tensor


class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                attn_mask=attention_mask,
                temperature=temperature,
                num_return_sequences=num_generations,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )

        prompt_len = prompt_ids.shape[1]
        completion_ids = output_ids[:, prompt_len:]

        from contextlib import nullcontext
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with ctx:
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.shape[1])
        
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)


    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


def create_rollout_engine(
    engine_type: str = 'torch',
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")