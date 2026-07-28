import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add src to python path so we can import our kernels
sys.path.append("/home/patemotter_google_com/workspace/vllm-torchtpu/src")

from vllm_torchtpu.kernels.deepseek_v4.mla import mla_ragged_paged_attention

def main():
    print("JAX default backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    # Inputs matching Layer 25 prefill sharded shape (for 1 TP rank, so num_heads = 8)
    num_seqs = 64
    num_heads = 8
    head_dim = 512
    q_len = 1
    
    # 1. q: [64 * 1, 8, 512]
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    q = jax.random.normal(k1, (num_seqs * q_len, num_heads, head_dim), dtype=jnp.bfloat16)
    print("First 10 elements of input q[0, 0]:", q[0, 0, :10])

    # 2. cache_kv: [100, 16, 4, 640]
    cache_kv = jnp.zeros((100, 16, 4, 640), dtype=jnp.uint8)
    # Write some non-zero values for the active page index 0, offset 0
    cache_kv = cache_kv.at[0, 0, :, :512].set(100)

    # 3. kv_lens: [64], all values = 1
    kv_lens = jnp.ones((num_seqs,), dtype=jnp.int32)

    # 4. kv_lens_to_attend: [64], all values = 0 (simulating early tokens)
    kv_lens_to_attend = jnp.zeros((num_seqs,), dtype=jnp.int32)

    # 5. page_indices: [64], flattened 1D array
    page_indices = jnp.zeros((num_seqs * 1,), dtype=jnp.int32)

    # 6. cu_q_lens: [65], cumulative query lengths [0, 1, 2, ..., 64]
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)

    # 7. distribution: [3], representing (num_decode, num_prefill, num_total)
    distribution = jnp.array([0, num_seqs, num_seqs], dtype=jnp.int32)

    # 8. attention_sinks: [8], sharded weight
    # Some channels have -inf, some have finite values
    attention_sinks = jnp.array([-jnp.inf, 1.5, -jnp.inf, 2.0, -jnp.inf, -jnp.inf, 0.8, -jnp.inf], dtype=jnp.float32)

    # 9. swa_accumution: [64, 8, 512], zeros
    swa_accumution = jnp.zeros((num_seqs, num_heads, head_dim), dtype=jnp.bfloat16)

    # 10. swa_l & swa_m
    swa_l = jnp.zeros((num_seqs, num_heads), dtype=jnp.float32)
    swa_m = jnp.full((num_seqs, num_heads), -jnp.inf, dtype=jnp.float32)

    print("Running mla_ragged_paged_attention...")
    out = mla_ragged_paged_attention(
        q=q,
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        kv_lens_to_attend=kv_lens_to_attend,
        topk_indices=None,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_accumution,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=0.044194,
        chunk_prefill_size=1, # Fix: pass chunk_prefill_size!
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        swa_enabled=False,
    )
    
    # Force execution and print output stats
    out.block_until_ready()
    print("Output shape:", out.shape)
    print("First 10 elements of out[0, 0]:", out[0, 0, :10])
    print("Output min:", jnp.min(out))
    print("Output max:", jnp.max(out))
    print("Output mean:", jnp.mean(out))
    print("Output has NaN:", jnp.isnan(out).any())

if __name__ == "__main__":
    main()
