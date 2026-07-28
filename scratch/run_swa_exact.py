import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.insert(0, "/home/patemotter_google_com/workspace/vllm-torchtpu/src")

from vllm_torchtpu.kernels.deepseek_v4 import mla_swa

path = "/home/patemotter_google_com/workspace/vllm-torchtpu/scratch/nan_dump_model.layers.0.attn_rank2.pt"
print(f"Loading dump from {path}...")
dump = torch.load(path)

# Convert torch tensors to jax arrays
def to_jax(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        orig_dtype = t.dtype
        if orig_dtype in (torch.bfloat16, torch.float16, torch.float32):
            t_f32 = t.detach().to(torch.float32)
            np_arr = t_f32.numpy()
            jax_arr = jnp.array(np_arr)
            if orig_dtype == torch.bfloat16:
                return jax_arr.astype(jnp.bfloat16)
            return jax_arr
        else:
            np_arr = t.detach().cpu().numpy()
            return jnp.array(np_arr)
    return t

q = to_jax(dump["q"])
kv = to_jax(dump["kv"])
sw_cache = to_jax(dump["sw_cache"])
swa_seq_lens = to_jax(dump["swa_seq_lens"])
swa_block_tables = to_jax(dump["swa_block_tables"])
swa_query_start_loc = to_jax(dump["swa_query_start_loc"])
swa_request_distribution = to_jax(dump["swa_request_distribution"])
attn_sink = to_jax(dump["attn_sink"])

if sw_cache is not None and sw_cache.ndim == 5:
    sw_cache = jnp.squeeze(sw_cache, axis=2)

print("Inputs loaded:")
print("q shape:", q.shape, "dtype:", q.dtype)
print("kv shape:", kv.shape, "dtype:", kv.dtype)
print("sw_cache shape:", sw_cache.shape, "dtype:", sw_cache.dtype)
print("swa_seq_lens:", swa_seq_lens[:4].tolist())
print("swa_block_tables:", swa_block_tables[:4].tolist())
print("swa_query_start_loc:", swa_query_start_loc[:4].tolist())
print("swa_request_distribution:", swa_request_distribution.tolist())
print("attn_sink shape:", attn_sink.shape, "dtype:", attn_sink.dtype)

# SWA config from model
sliding_window = 32
logical_page_size = 16

# Run SWA attention kernel
print("Running SWA attention kernel with exact dump inputs...")
out, updated_kv, _, _ = mla_swa.mla_sliding_window_ragged_paged_attention(
    q=q,
    new_kv=kv,
    cache_kv=sw_cache,
    kv_lens=swa_seq_lens,
    page_indices=swa_block_tables,
    cu_q_lens=swa_query_start_loc,
    distribution=swa_request_distribution,
    attention_sinks=attn_sink[:q.shape[1]], # Only keep the local query heads slice
    sm_scale=1.0 / (512 ** 0.5), # standard scale for head_dim=512
    sliding_window=sliding_window,
    logical_page_size=logical_page_size,
    num_kv_pages_per_block=2,
    num_queries_per_block=8,
    unnormalized_output=True
)

nan_count = jnp.isnan(out).sum().item()
print(f"Kernel execution finished. NaN count in output: {nan_count}")
if nan_count > 0:
    print("Output contains NaNs! Successfully reproduced the bug locally.")
else:
    print("Output is clean! No NaNs.")
    out_f32 = out.astype(jnp.float32)
    print(f"Output Stats: min={out_f32.min().item():.6f} | max={out_f32.max().item():.6f} | mean={out_f32.mean().item():.6f}")

