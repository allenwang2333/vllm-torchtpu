"""FP8 fused_moe_func micro-benchmark at Qwen3.5-397B production shapes.

Compared to microbench_fused_moe.py (bf16), this exercises the FP8 dequant
path inside gmm_v2 — if the regression is FP8-specific, we'd see it here.
"""

import argparse
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func


def _fake_fp8(rng_key, shape, block_k=128, block_n=128):
    """Return (fp8_weight, scale) with realistic shapes.

    fp8 weight is float8_e4m3fn; scale is float32 in shape
    (num_experts, num_quant_blocks_k, 1, size_n) as expected by gmm_v2.
    """
    k1, k2 = jax.random.split(rng_key)
    # Random values in the fp8 representable range.
    weight_bf16 = (jax.random.normal(k1, shape, dtype=jnp.float32) / 10)
    weight_fp8 = weight_bf16.astype(jnp.float8_e4m3fn)
    num_experts, size_k, size_n = shape
    num_blocks_k = size_k // block_k
    # Scale of shape (num_experts, num_blocks_k, 1, size_n): per-K-block scalar
    # broadcast across the block, one per N-column.
    scale = (jnp.ones((num_experts, num_blocks_k, 1, size_n), jnp.float32)
             * 0.01)
    return weight_fp8, scale


def _make_inputs(num_tokens, hidden, intermediate, num_local, topk, seed):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)
    hs = (jax.random.normal(keys[0], (num_tokens, hidden), dtype=jnp.float32)
          / 10).astype(jnp.bfloat16)
    w1, w1_scale = _fake_fp8(keys[1], (num_local, hidden, intermediate * 2))
    w2, w2_scale = _fake_fp8(keys[2], (num_local, intermediate, hidden))
    global_experts = num_local * 8
    experts_start = 3 * num_local
    ids = jax.random.randint(keys[3], (num_tokens, topk), 0, global_experts,
                             dtype=jnp.int32)
    weights = jax.random.uniform(keys[4], (num_tokens, topk),
                                 dtype=jnp.float32)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return (hs, w1, w1_scale, w2, w2_scale, ids,
            weights.astype(jnp.bfloat16), experts_start)


def _time_run(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        out = fn(*args)
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    hidden = 4096
    intermediate = 1024
    num_local = 64
    topk = 10

    print(f"tag={args.tag or 'unset'}  jax={jax.__version__}")
    print(f"{'shape':<16} {'per-call':>10}")
    print("-" * 30)
    for name, num_tokens in [("prefill_1024", 1024), ("decode_64", 64)]:
        try:
            (hs, w1, w1_s, w2, w2_s, ids, wts, exp_start
             ) = _make_inputs(num_tokens, hidden, intermediate, num_local,
                              topk, seed=hash(name) & 0xFFFF)

            def run(hs, w1, w1_s, w2, w2_s, ids, wts):
                return fused_moe_func(
                    hidden_states=hs, w1=w1, w2=w2,
                    w1_scale=w1_s, w2_scale=w2_s,
                    w1_bias=None, w2_bias=None,
                    topk_weights=wts, topk_ids=ids,
                    experts_start=exp_start,
                    topk=topk, activation="silu",
                    use_ep=True, use_sparse_core=True,
                )
            us = _time_run(run, (hs, w1, w1_s, w2, w2_s, ids, wts),
                           iters=args.iters, warmup=args.warmup)
            print(f"{name:<16} {us:>7.2f} µs")
        except Exception as exc:
            print(f"{name:<16} FAILED: {type(exc).__name__}: {str(exc)[:100]}")


if __name__ == "__main__":
    main()
