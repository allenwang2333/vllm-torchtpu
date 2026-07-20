"""Micro-benchmark ragged_gated_delta_rule_decode_only (GDN decode kernel).

At concurrency=64 with 1k-token generation, this kernel fires on every step
for every linear-attention layer (~45 layers). If libtpu 0.0.42.1 lowers it
worse than 0.0.40, we'd see it here.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.gdn.v2.gdn_decode_kernel import \
    ragged_gated_delta_rule_decode_only


def _make_inputs(decode_N, H_qk=2, H_v=8, K=128, V=128, dtype=jnp.bfloat16,
                 max_num_req=None):
    rng = np.random.RandomState(42)
    all_seqlens = [1] * decode_N
    N = len(all_seqlens)
    T = sum(all_seqlens)
    cu_seqlens = np.cumsum([0] + all_seqlens).astype(np.int32)
    if max_num_req is not None:
        padded_cu = np.full(max_num_req + 1, T, dtype=np.int32)
        padded_cu[:len(cu_seqlens)] = cu_seqlens
        cu_seqlens = padded_cu
    q = rng.randn(T, H_qk, K).astype(np.float32)
    k = rng.randn(T, H_qk, K).astype(np.float32)
    v = rng.randn(T, H_v, V).astype(np.float32)
    a = rng.randn(T, H_v).astype(np.float32)
    b = rng.randn(T, H_v).astype(np.float32)
    A_log = rng.randn(H_v).astype(np.float32)
    h0_N = max_num_req if max_num_req is not None else N
    h0 = rng.randn(h0_N, H_v, K, V).astype(np.float32)
    state_indices = np.arange(h0_N, dtype=np.int32)
    q, k, v, a, b = (jnp.array(x, dtype=dtype) for x in [q, k, v, a, b])
    return dict(
        mixed_qkv=jnp.concatenate([q.reshape(T, -1), k.reshape(T, -1),
                                   v.reshape(T, -1)], axis=-1),
        b=b, a=a,
        recurrent_state=jnp.array(h0),
        A_log=jnp.array(A_log),
        dt_bias=jnp.zeros(H_v, dtype=jnp.float32),
        query_start_loc=jnp.array(cu_seqlens),
        state_indices=jnp.array(state_indices),
        distribution=jnp.array([decode_N, N, N], dtype=jnp.int32),
        has_initial_state=jnp.ones((state_indices.shape[0],), dtype=jnp.bool_),
    ), dict(n_kq=H_qk, n_v=H_v, d_k=K, d_v=V)


def _time_run(fn, iters=100, warmup=20):
    for _ in range(warmup):
        state, out = fn()
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        state, out = fn()
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    scenarios = [
        ("decode_64",  64, 512),
        ("decode_128", 128, 512),
        ("decode_256", 256, 512),
    ]
    print(f"{'shape':<16} {'per-call':>10}")
    print("-" * 30)
    for name, decode_N, max_req in scenarios:
        args_, kwargs = _make_inputs(decode_N, max_num_req=max_req)
        run = lambda: ragged_gated_delta_rule_decode_only(**args_, **kwargs)
        us = _time_run(run, iters=args.iters, warmup=args.warmup)
        print(f"{name:<16} {us:>7.2f} µs")


if __name__ == "__main__":
    main()
