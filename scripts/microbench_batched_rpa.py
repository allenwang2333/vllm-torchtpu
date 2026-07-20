"""Micro-bench batched RPA attention (--attention-backend CUSTOM).

Qwen3.5-397B: 60 hidden layers, layer_types pattern = 3 linear + 1 full,
so 15 layers use this attention. head_dim=256, num_attention_heads=32,
num_key_value_heads=2 (GQA 16:1).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB


_DECODE_BLOCKS = KBC.BlockSizes(
    bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8, n_buffer=3)
_PREFILL_BLOCKS = KBC.BlockSizes(
    bq_sz=1792, bq_c_sz=28, bkv_sz=256, batch_size=2, n_buffer=3)


def _make(q_len, kv_len, distribution, num_q_heads=32, num_kv_heads=2,
          head_dim=128, page_size=64, dtype=jnp.bfloat16, seed=0):
    rng = np.random.default_rng(seed)
    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    total_pages = max(16, (kv_len + page_size - 1) // page_size + 8)
    sm_scale = head_dim ** -0.5

    query = r(q_len, num_q_heads, head_dim)
    keys = r(q_len, num_kv_heads, head_dim)
    values = r(q_len, num_kv_heads, head_dim)

    cache_shape = KB.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype)
    cache = np.zeros(cache_shape, np.float32)
    kv_lens = jnp.array([kv_len], jnp.int32)
    page_indices = jnp.arange(min(total_pages, (kv_len // page_size) + 1), dtype=jnp.int32)
    # Pad page_indices out.
    page_indices = jnp.pad(page_indices, (0, total_pages - page_indices.size))
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    dist = jnp.asarray(distribution, jnp.int32)

    return dict(
        queries=jnp.asarray(query, dtype),
        keys=jnp.asarray(keys, dtype),
        values=jnp.asarray(values, dtype),
        kv_cache=jnp.asarray(cache, dtype),
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=dist,
    ), sm_scale


def _time_run(fn, iters=50, warmup=10):
    for _ in range(warmup):
        out, cache = fn()
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out, cache = fn()
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    scenarios = [
        # (name, q_len, kv_len, distribution)
        ("prefill_1024",  1024, 1024, (0, 0, 1)),
        ("decode_64",     64,   8192, (1, 1, 1)),
        ("decode_1_kv1k", 1,    1024, (1, 1, 1)),
        ("decode_1_kv8k", 1,    8192, (1, 1, 1)),
    ]
    print(f"{'shape':<20} {'per-call':>10}")
    print("-" * 34)
    for name, q_len, kv_len, dist in scenarios:
        try:
            args, sm_scale = _make(q_len, kv_len, dist)
            run = lambda: KB.ragged_paged_attention(
                args["queries"], args["keys"], args["values"], args["kv_cache"],
                args["kv_lens"], args["page_indices"], args["cu_q_lens"],
                args["distribution"],
                sm_scale=sm_scale,
                decode_block_sizes=_DECODE_BLOCKS,
                prefill_block_sizes=_PREFILL_BLOCKS,
                skip_kv_update=False,
            )
            us = _time_run(run)
            print(f"{name:<20} {us:>7.2f} µs")
        except Exception as exc:
            print(f"{name:<20} FAILED: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
