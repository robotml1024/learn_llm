import math
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

def repeat_kv(x, n_rep):
    # x(bs, slen, num_kv, head_dim)
    bs, slen, num_kv, head_dim = x.shape
    return x[:,:,:,None,:].expand(bs, slen, num_kv, n_rep, head_dim).reshape(bs, slen, num_kv * n_rep, head_dim)

def precompute_freq_cis(end, dim, rope_base=1e5, rope_scaling=None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)), 1.0
    if rope_scaling is not None:
        orgi_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 8),
            rope_scaling.get("beta_fast", 32), rope_scaling.get("beta_slow", 1), rope_scaling.get("attention_factor", 1.0))
        if end / orgi_max > 1.0:
            bound_dim = lambda b: (dim * math.log(orgi_max / (2 * math.pi * b))) / (2 * math.log(rope_base))
            low_dim, high_dim = max(bound_dim(beta_fast), 0), min(bound_dim(beta_slow), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2) - low_dim) / max(high_dim - low_dim, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    freq_cos = torch.cat((cos, cos), dim=-1) * attn_factor # (slen, hidden_size)
    freq_sin = torch.cat((sin, sin), dim=-1) * attn_factor # (slen, hidden_size)
    return freq_cos, freq_sin


def apply_rope_embedding(xq, xk, cos, sin, unsqueeze_dim=1):
    # xq (bs, slen, num_attention_heads, head_dim), xk (bs, slen, num_kv, head_dim)
    # cos, sin(slen, head_dim)
    def rotate_half(x): return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    xq_embed = ((xq * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(xq) * sin.unsqueeze(unsqueeze_dim))).to(xq.dtype)
    xk_embed = ((xk * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(xk) * sin.unsqueeze(unsqueeze_dim))).to(xk.dtype)
    return xq_embed, xk_embed

class MinimindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size, num_hidden_layers, use_moe, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash = kwargs.get("flash", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_dim = kwargs.get("intermediate_dim", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_base = kwargs.get("rope_base", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_dim)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

class Attention(nn.Module):
    def __init__(self, config: MinimindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.rms_eps = config.rms_norm_eps
        self.dropout = config.dropout
        self.flash = config.flash and hasattr(F, 'scaled_dot_product_attention')
        self.is_causal = True
        self.wq = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=False)
        self.wo = nn.Linear(self.head_dim * self.num_attention_heads, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=self.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=self.rms_eps)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x, pos_embed, kv_cache=None, use_cache=False, attn_mask=None):
        bs, slen, _ = x.shape
        # 得到qkv
        xq = self.wq(x).view(bs, slen, self.num_attention_heads, self.head_dim) # (bs, slen, num_attention_heads, head_dim)
        xk = self.wk(x).view(bs, slen, self.num_key_value_heads, self.head_dim) # (bs, slen, num_key_value_heads, head_dim)
        xv = self.wv(x).view(bs, slen, self.num_key_value_heads, self.head_dim) # (bs, slen, num_key_value_heads, head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 加入位置编码
        cos, sin = pos_embed
        xq, xk = apply_rope_embedding(xq, xk, cos, sin)
        if kv_cache is not None:
            xk = torch.cat([kv_cache[0], xk], dim=1)
            xv = torch.cat([kv_cache[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 交换维度计算注意力分数(bs, num_attention_heads, slen, head_dim)
        xq, xk, xv = xq.transpose(1,2), repeat_kv(xk, self.n_rep).transpose(1,2), repeat_kv(xv, self.n_rep).transpose(1,2)

        # 计算注意力
        if self.flash and slen > 1 and (not self.is_causal or kv_cache is None) and (attn_mask is None or torch.all(attn_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            attn_scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # (bs, num_attention_heads, slen, slen)
            if self.is_causal: attn_scores[..., -slen:] += torch.full((slen, slen), float('-inf'), device=attn_scores.device).triu(diagonal=1)
            if attn_mask is not None: attn_scores += (1 - attn_mask) * 1e-9
            attn_weights = F.softmax(attn_scores, dim=-1)
            output = self.attn_dropout(torch.matmul(attn_weights, xv)) # (bs, num_attention_heads, slen, head_dim)   
        output = output.transpose(1,2).contiguous().view(bs, slen, self.hidden_size) # (bs, slen, hidden_size)
        output = self.resid_dropout(self.wo(output)) # (bs, slen, hidden_size)
        return output, past_kv
    
class FFN(nn.Module):
    def __init__(self, config: MinimindConfig, intermediate_dim=None):
        super().__init__()
        intermediate_dim = intermediate_dim or config.intermediate_dim
        self.up_proj = nn.Linear(config.hidden_size, intermediate_dim)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, config.hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x) * self.up_proj(x)))
    
class MoEFFN(nn.Module):
    def __init__(self, config: MinimindConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([FFN(config, intermediate_dim=config.moe_intermediate_dim) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.hidden_size, config.num_experts)

    def forward(self, x):
        bs, slen, dim = x.shape
        x_flat = x.reshape(-1, dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)

        top_weight, top_idx = torch.topk(scores, k=self.config.moe_top_k, dim=-1, sorted=False)
        if self.config.norm_moe_topk: top_weight = top_weight / (top_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (top_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = top_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, expert(x_flat[mask]) * weight).to(y.dtype)
            elif self.training:
                y[0,0] += 0 * (p.sum() for p in expert.parameters())
            
        if self.training and self.config.router_aux_loss_coef > 0:
            # 计算aux loss，鼓励router均匀使用expert
            load = F.one_hot(top_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.config.router_aux_loss_coef
        else:
            # 保证推理或者不设定coef的时候aux_loss也是一个标量0, 避免后续loss计算出现错误, 同时保证和score对齐
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(bs, slen, dim)

class MinimindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MinimindConfig):
        super().__init__()
        self.attn = Attention(config)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FFN(config) if not config.use_moe else MoEFFN(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_state, pos_embed, kv_cache=None, use_cache=False, attn_mask=None):
        residual = hidden_state
        hidden_state, kv_cache = self.attn(self.input_norm(hidden_state), pos_embed, kv_cache=kv_cache, use_cache=use_cache, attn_mask=attn_mask)
        hidden_state += residual
        hidden_state = hidden_state + self.mlp(self.post_attn_norm(hidden_state))
        return hidden_state, kv_cache
    
class MinimindModel(nn.Module):
    def __init__(self, config: MinimindConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MinimindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        freq_cos, freq_sin = precompute_freq_cis(end=config.max_position_embeddings, dim=config.head_dim, rope_base=config.rope_base, rope_scaling=config.rope_scaling)
        self.register_buffer('freq_cos', freq_cos, persistent=False)
        self.register_buffer('freq_sin', freq_sin, persistent=False)

    def forward(self, input_ids, kv_cache=None, use_cache=False, attn_mask=None):
        '''
        kv_cache最外层0表示k，1表示v
        再里层就是self.layers的层数
        然后就是(bs, slen, num_attention_heads, head_dim)
        '''
        bs, slen = input_ids.shape
        kv_cache = kv_cache or [None] * len(self.layers)
        start_pos = kv_cache[0][0].shape[1] if kv_cache[0] is not None else 0
        pos_embed = (self.freq_cos[start_pos: start_pos + slen], self.freq_sin[start_pos: start_pos + slen])
        token_embed = self.dropout(self.embed(input_ids))
        present = []
        for layer, kv_cache_item in zip(self.layers, kv_cache):
            token_embed, cur_kv_cache = layer(token_embed, pos_embed, kv_cache=kv_cache_item, use_cache=use_cache, attn_mask=attn_mask)
            present.append(cur_kv_cache)
        
        output = self.post_norm(token_embed)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MoEFFN)], token_embed.new_zeros(1).squeeze())
        return output, present, aux_loss
    
class MinimindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MinimindConfig
    def __init__(self, config: MinimindConfig=None):
        self.config = config or MinimindConfig()
        super().__init__(self.config)
        self.model = MinimindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids, logits_to_keep=0, labels=None, kv_cache=None, use_cache=False, attn_mask=None):
        '''
        logits_to_keep: int, 用于保留最后的几个token的logits
        '''
        hidden_state, kv_cache, aux_loss = self.model(input_ids, kv_cache=kv_cache, use_cache=use_cache, attn_mask=attn_mask)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_state[:, slice_indices, :])
        if labels is not None:
            x, y = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.reshape(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, logits=logits, aux_loss=aux_loss, past_key_values=kv_cache, hidden_states=hidden_state)
    
    @torch.inference_mode()
    def generate(self, input_ids, max_new_token=8192, kv_cache=None, use_cache=False, attn_mask=None, do_sample=True, temperature=1.0, top_p=0.95, top_k=50, eos_token_id=2):
        '''
        input_ids输入，经过模型得到输出，将输出拼接上原始输入再接着输入直到得到eos_id
        模型最后输出是用greedy_search还是通过分布通过do_sample控制
        '''
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_token):
            start_pos = kv_cache[0][0].shape[1] if kv_cache[0] is not None else 0
            outputs = self.forward(input_ids[:, start_pos:], kv_cache=kv_cache, use_cache=use_cache, attn_mask=attn_mask)
            logits = outputs.logits[:, -1, :] / temperature
            if do_sample:
                if top_k > 0:
                    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = float('-inf')
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    mask[..., 1:], mask[..., 0] = mask[..., :-1], 0
                    logits[mask.scatter(1, sorted_indices, mask)] = float('-inf')
                next_token = torch.multinomial(torch.softmax(logits, -1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            kv_cache = outputs.past_key_values if use_cache else None
            if eos_token_id:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        
        return input_ids