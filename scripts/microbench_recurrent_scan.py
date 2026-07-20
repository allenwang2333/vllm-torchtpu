"""Micro-benchmark recurrent_scan_v2 (GDN linear attention prefill kernel).

Modeled on tests/kernels/test_recurrent_scan.py. Qwen3.5-397B's linear-attention
layers use this on prefill; if libtpu 0.0.42.1 lowers it worse than 0.0.40,
we'd see it here.

Shape modeled on the test: prefill of a few concatenated requests, decode of 8.
Set TPU_LIBRARY_PATH to swap libtpu versions between runs.
"""

import argparse
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v2.recurrent_scan_v2 import recurrent_scan


def _make_inputs(decode_lengths, prefill_lengths, chunk_size, seed):
    dtype = jnp.bfloat16
    kq_head_dim, v_head_dim = 128, 128
    n_kq, n_v = 2, 8
    MAX_TOKENS = 8192
    MAX_REQS = 512

    actual_num_tokens = decode_lengths + sum(prefill_lengths)
    actual_max_reqs = decode_lengths + len(prefill_lengths)

    q_loc = jnp.cumsum(
        jnp.array([0] + [1] * decode_lengths + prefill_lengths, dtype=jnp.int32))
    q_loc = jnp.pad(q_loc, (0, MAX_REQS + 1 - len(q_loc)), constant_values=1)

    rngs = iter(jax.random.split(jax.random.key(seed), 15))

    available_indices = jax.random.permutation(
        next(rngs), jnp.arange(1, MAX_REQS, dtype=jnp.int32))
    valid_state_indices = available_indices[:actual_max_reqs]
    state_indices = jnp.pad(
        valid_state_indices, (0, MAX_REQS - actual_max_reqs),
        constant_values=0)

    query = jax.random.normal(next(rngs), (MAX_TOKENS, n_kq * kq_head_dim), dtype=dtype)
    key = jax.random.normal(next(rngs), (MAX_TOKENS, n_kq * kq_head_dim), dtype=dtype)
    value = jax.random.normal(next(rngs), (MAX_TOKENS, n_v * v_head_dim), dtype=dtype)
    b = jax.random.normal(next(rngs), (MAX_TOKENS, n_v), dtype=dtype)
    a = jax.random.normal(next(rngs), (MAX_TOKENS, n_v), dtype=dtype)

    query = jnp.pad(query, ((0, chunk_size), (0, 0)))
    key = jnp.pad(key, ((0, chunk_size), (0, 0)))
    value = jnp.pad(value, ((0, chunk_size), (0, 0)))
    b = jnp.pad(b, ((0, chunk_size), (0, 0)))
    a = jnp.pad(a, ((0, chunk_size), (0, 0)))

    recurrent_state = jnp.zeros((MAX_REQS, n_v, kq_head_dim, v_head_dim), dtype=jnp.float32)

    num_decodes = decode_lengths
    decode_state = jax.random.normal(
        next(rngs), (num_decodes, n_v, kq_head_dim, v_head_dim), dtype=jnp.float32)
    prefill_state = jnp.zeros(
        (actual_max_reqs - num_decodes, n_v, kq_head_dim, v_head_dim), dtype=jnp.float32)
    valid_state = jnp.concatenate([decode_state, prefill_state], axis=0)
    recurrent_state = recurrent_state.at[valid_state_indices].set(valid_state)

    A_log = jax.random.normal(next(rngs), (n_v,), dtype=dtype)
    dt_bias = jax.random.normal(jax.random.key(0), (n_v,), dtype=dtype)

    mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
    distribution = jnp.array(
        [decode_lengths, actual_max_reqs, actual_max_reqs], dtype=jnp.int32)
    return dict(
        mixed_qkv=mixed_qkv, b=b, a=a, recurrent_state=recurrent_state,
        A_log=A_log, dt_bias=dt_bias, query_start_loc=q_loc,
        state_indices=state_indices, distribution=distribution,
    ), dict(n_kq=n_kq, n_v=n_v, d_k=kq_head_dim, d_v=v_head_dim,
            chunk_size=chunk_size, BT=chunk_size, use_qk_norm_in_gdn=True)


def _time_run(fn, iters=50, warmup=10):
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
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    scenarios = [
        ("prefill_1024", 0, [1024], 128),
        ("prefill_8k",   0, [8192], 128),
        ("decode_64",    64, [],   128),
        ("mixed_dec8_pf1016", 8, [1016], 128),
    ]

    print(f"{'shape':<24} {'per-call':>10}")
    print("-" * 40)
    for name, dec, pref, cs in scenarios:
        args_, kwargs = _make_inputs(dec, pref, cs, seed=hash(name) & 0xFFFF)
        run = lambda: recurrent_scan(**args_, **kwargs)
        us = _time_run(run, iters=args.iters, warmup=args.warmup)
        print(f"{name:<24} {us:>7.2f} µs")


if __name__ == "__main__":
    main()
