import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.insert(0, "/home/patemotter_google_com/workspace/vllm-torchtpu/src")

from vllm_torchtpu.kernels.deepseek_v4 import mla_swa

# Configuration
batch_size = 12
num_heads = 128
head_dim = 512
lkv_dim = 640
q_dtype = jnp.bfloat16
sliding_window = 32
page_size = 12
pages_per_seq = 27
total_num_pages = batch_size * pages_per_seq  # 324

print("JAX Backend:", jax.default_backend())

# Setup random inputs
np.random.seed(42)

# Step 1: Prefill Step
new_kv_lens_1 = np.array([21, 19, 60, 48, 22, 25, 23, 26, 23, 31, 16, 38], dtype=np.int32)
kv_lens_1 = new_kv_lens_1.copy()
cu_q_lens_1 = np.concatenate([[0], np.cumsum(new_kv_lens_1)]).astype(np.int32)
total_tokens_1 = np.sum(new_kv_lens_1)

# Step 2: Mixed Decode/Prefill Step
new_kv_lens_2 = np.array([1, 1, 1, 1, 1, 1, 17, 26, 22, 26, 31, 31], dtype=np.int32)
kv_lens_2 = kv_lens_1 + new_kv_lens_2
cu_q_lens_2 = np.concatenate([[0], np.cumsum(new_kv_lens_2)]).astype(np.int32)
total_tokens_2 = np.sum(new_kv_lens_2)

# Page indices
page_indices = np.arange(total_num_pages, dtype=np.int32)

# Quantization scale factors
kv_packing = 4
cache_shape = (total_num_pages, page_size // kv_packing, kv_packing, lkv_dim)
cache_kv = jnp.zeros(cache_shape, dtype=jnp.uint8)
attention_sinks = jnp.zeros((num_heads,), dtype=jnp.float32)

# Random inputs to ensure realistic scale
q_1 = jax.random.normal(jax.random.PRNGKey(0), (total_tokens_1, num_heads, head_dim), dtype=q_dtype)
new_kv_1 = jax.random.normal(jax.random.PRNGKey(1), (total_tokens_1, head_dim), dtype=q_dtype)
distribution_1 = jnp.array([0, 0, batch_size], dtype=jnp.int32)

print("Running Step 1...")
out_1, updated_kv_1, _, _ = mla_swa.mla_sliding_window_ragged_paged_attention(
    q_1, new_kv_1, cache_kv, jnp.array(kv_lens_1), jnp.array(page_indices),
    jnp.array(cu_q_lens_1), distribution_1, attention_sinks=attention_sinks,
    sm_scale=1.0, sliding_window=sliding_window, logical_page_size=page_size,
    num_kv_pages_per_block=2, num_queries_per_block=8, unnormalized_output=True
)

nan_count_1 = jnp.isnan(out_1).sum().item()
print(f"Step 1 completed. NaN count in output: {nan_count_1}")

# Step 2
q_2 = jax.random.normal(jax.random.PRNGKey(2), (total_tokens_2, num_heads, head_dim), dtype=q_dtype)
new_kv_2 = jax.random.normal(jax.random.PRNGKey(3), (total_tokens_2, head_dim), dtype=q_dtype)
distribution_2 = jnp.array([6, 6, batch_size], dtype=jnp.int32)

print("Running Step 2 (Decode/Prefill)...")
out_2, updated_kv_2, _, _ = mla_swa.mla_sliding_window_ragged_paged_attention(
    q_2, new_kv_2, updated_kv_1, jnp.array(kv_lens_2), jnp.array(page_indices),
    jnp.array(cu_q_lens_2), distribution_2, attention_sinks=attention_sinks,
    sm_scale=1.0, sliding_window=sliding_window, logical_page_size=page_size,
    num_kv_pages_per_block=2, num_queries_per_block=8, unnormalized_output=True
)

nan_count_2 = jnp.isnan(out_2).sum().item()
print(f"Step 2 completed. NaN count in output: {nan_count_2}")

print("Done!")
