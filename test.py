# import torch

# def precompute_freq_cis(dim, end, rope_base=1e5, rope_scaling=None):
#     freq = 1.0 / rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
#     t = torch.arange(end)
#     freq = torch.outer(t, freq)
#     if rope_scaling is not None:
#         orig_max, factor, beta_fast, beta_slow, attn_factor = \
#         rope_scaling.get("original_max_position_embeddings", 2048),
#         rope_scaling.get("factor", 8),
#         rope_scaling.get("beta_fast", 32)
#         rope_scaling.get("beta_slow", 1)
#         rope_scaling.get("attention_factor", 1.0)
#         if end / orig_max > 1.0:
#             pass

#     cos, sin = torch.cos(freq), torch.sin(freq)
#     freq_cos = torch.cat([cos, cos], dim=-1)
#     freq_sin = torch.cat([sin, sin], dim=-1)
#     return freq_cos, freq_sin

# def apply_rope_embedding(q, k, cos, sin, unsqueeze_dim=1):
#     # q (bs, slen, n_heads, dim)
#     # cos (slen, dim)
#     def rotate_half(x): return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
#     q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
#     k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
#     return q_embed, k_embed


import math
print(math.log(math.e))