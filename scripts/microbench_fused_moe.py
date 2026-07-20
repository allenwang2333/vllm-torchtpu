"""Micro-benchmark fused_moe_func at Qwen3.5-397B-A17B production shapes.

Set TPU_LIBRARY_PATH to swap libtpu versions between runs.

Shapes (per DP rank, EP shard):
  hidden_size = 4096
  moe_intermediate_size = 1024
  num_local_experts = 64  # 512 global / 8 EP shards
  topk = 10

Two batch sizes: prefill_1024 and decode_64.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func


def _make_inputs(num_tokens, hidden, intermediate, num_local, topk, seed):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)
    hs = (jax.random.normal(keys[0], (num_tokens, hidden), dtype=jnp.float32)
          / 10).astype(jnp.bfloat16)
    # gate_up weight has 2× intermediate for silu-gated MLP.
    w1 = (jax.random.normal(keys[1], (num_local, hidden, intermediate * 2),
                            dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (jax.random.normal(keys[2], (num_local, intermediate, hidden),
                            dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    # topk_ids are GLOBAL ids in [0, 8*num_local). We span [experts_start,
    # experts_start+num_local) most of the time so most experts route locally,
    # with occasional out-of-shard ids so the mask is exercised.
    global_experts = num_local * 8
    experts_start = 3 * num_local  # arbitrary shard id
    ids = jax.random.randint(keys[3], (num_tokens, topk), 0, global_experts,
                             dtype=jnp.int32)
    weights = jax.random.uniform(keys[4], (num_tokens, topk),
                                 dtype=jnp.float32)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return hs, w1, w2, ids, weights.astype(jnp.bfloat16), experts_start


def _time_run(fn, args, iters=50, warmup=5):
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
    ap.add_argument("--tag", default="", help="Label for the output row")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--use-sparse-core", default="1",
                    help="Set to 0 to disable SC path")
    args = ap.parse_args()

    hidden = 4096
    intermediate = 1024
    num_local = 64  # 512 global / 8 EP shards
    topk = 10

    shapes = [
        ("prefill_1024", 1024),
        ("decode_64", 64),
    ]

    print(f"tag={args.tag or 'unset'}  jax={jax.__version__}  "
          f"libtpu_from_env={os.environ.get('TPU_LIBRARY_PATH','<default>')}")
    print(f"{'shape':<16} {'per-call':>10}")
    print("-" * 30)
    for name, num_tokens in shapes:
        hs, w1, w2, ids, wts, exp_start = _make_inputs(
            num_tokens, hidden, intermediate, num_local, topk,
            seed=hash(name) & 0xFFFF)

        run = lambda hs, w1, w2, ids, wts: fused_moe_func(  # noqa
            hidden_states=hs, w1=w1, w2=w2,
            w1_scale=None, w2_scale=None,
            w1_bias=None, w2_bias=None,
            topk_weights=wts, topk_ids=ids,
            experts_start=exp_start,
            topk=topk, activation="silu",
            use_ep=True, use_sparse_core=(args.use_sparse_core != "0"),
        )

        us = _time_run(run, (hs, w1, w2, ids, wts),
                       iters=args.iters, warmup=args.warmup)
        print(f"{name:<16} {us:>7.2f} µs")


if __name__ == "__main__":
    main()
