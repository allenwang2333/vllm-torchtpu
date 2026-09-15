# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: RPA v3 decode at the Qwen3-Coder-30B-A3B-FP8 production shape.

Why this exists
---------------
On Qwen3-Coder-30B-A3B-Instruct-FP8 at TP=8, each rank sees ``num_kv_heads=1``
(the checkpoint's 4 KV heads replicated across 8 ranks). The v3 KV cache is laid
out as::

    [num_blocks, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)

With an fp8 KV cache ``kv_packing = 32 // 8 = 4``, so ``align_to(1 * 2, 4) == 4``:
the packing axis holds 4 slots but only 2 (K and V) carry data. Half of every
byte allocated -- and half of every byte the decode kernel DMAs -- is zero
padding.

The arms below are chosen to *discriminate*, not merely to measure. Arms A, B
and C all move the **same number of physical bytes**; they differ only in how
much of that traffic is useful:

  A  fp8,  1 kv head  -> packing axis [K, V, pad, pad]   <- production today
  B  fp8,  2 kv heads -> packing axis [K0, V0, K1, V1]   <- same bytes, 0 padding
  C  bf16, 1 kv head  -> kv_packing=2, axis [K, V]       <- same bytes, 0 padding
  D  fp8,  1 kv head, half the kv_len                    <- half the bytes

Reading the result:

  * A ~= B ~= C  => the kernel is bandwidth-bound on *physical* bytes, so the
    padding is pure waste and removing it is worth up to ~2x on this op. Note
    that A ~= C also means the fp8 KV cache is currently buying nothing over
    bf16 on this model -- identical bytes per token.
  * A >> B       => the cost tracks *useful* work, not bytes moved, and the
    padding is much cheaper than the byte count suggests. The layout change
    would then not be worth its risk.
  * D ~= A/2     => time scales with bytes, corroborating the A~=B~=C reading.

D is the closest thing to a direct estimate of the post-fix cost, since a
depadded fp8 cache moves exactly the bytes D moves.

Run one arm per chip; see --arm.
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ragged_paged_attention)

# Production decode shape, per rank, for Qwen3-Coder-30B-A3B-Instruct-FP8 at TP=8.
# Cross-check: the xprof row for this op is `RPAd-p_256-bq_1_1-bkv_9216_9216`.
PAGE_SIZE = 256
HEAD_DIM = 128
NUM_Q_HEADS = 4  # 32 attention heads / TP 8
BATCH = 64  # --max-num-seqs=64, all decoding
KV_LEN = 8192  # --random-input-len=8192
MAX_MODEL_LEN = 9216  # --max-model-len=9216

ARMS = {
    # name: (kv_dtype, num_kv_heads, kv_len)
    "A_fp8_1kvh": (jnp.float8_e4m3fn, 1, KV_LEN),
    "B_fp8_2kvh": (jnp.float8_e4m3fn, 2, KV_LEN),
    "C_bf16_1kvh": (jnp.bfloat16, 1, KV_LEN),
    "D_fp8_1kvh_halfkv": (jnp.float8_e4m3fn, 1, KV_LEN // 2),
}


def build(kv_dtype, num_kv_heads, kv_len, seed=0):
    """Build a pure-decode workload: BATCH sequences, one query token each."""
    pages_per_seq = (MAX_MODEL_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = BATCH * pages_per_seq

    cache_shape = get_kv_cache_shape(num_blocks, PAGE_SIZE, num_kv_heads,
                                     HEAD_DIM, kv_dtype)

    rng = np.random.default_rng(seed)
    # Fill with small values: fp8_e4m3 saturates easily and NaNs would make the
    # softmax degenerate, which changes nothing about bandwidth but makes the
    # numbers harder to trust.
    kv_cache = jnp.asarray(rng.uniform(-1.0, 1.0, size=cache_shape),
                           dtype=kv_dtype)

    # One new token per sequence, as in steady-state decode.
    q = jnp.asarray(rng.standard_normal((BATCH, NUM_Q_HEADS, HEAD_DIM)),
                    dtype=jnp.bfloat16)
    k = jnp.asarray(rng.uniform(-1.0, 1.0, (BATCH, num_kv_heads, HEAD_DIM)),
                    dtype=kv_dtype)
    v = jnp.asarray(rng.uniform(-1.0, 1.0, (BATCH, num_kv_heads, HEAD_DIM)),
                    dtype=kv_dtype)

    kv_lens = jnp.full((BATCH, ), kv_len, dtype=jnp.int32)
    # Shuffled so the page walk is not artificially sequential -- a contiguous
    # block table would let the DMA engine prefetch in a way production cannot.
    pool = np.arange(num_blocks, dtype=np.int32)
    rng.shuffle(pool)
    page_indices = jnp.asarray(pool[:BATCH * pages_per_seq])
    cu_q_lens = jnp.asarray(np.arange(BATCH + 1, dtype=np.int32))
    distribution = jnp.asarray([BATCH, BATCH, BATCH], dtype=jnp.int32)

    return dict(q=q,
                k=k,
                v=v,
                kv_cache=kv_cache,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution), cache_shape


@functools.partial(jax.jit, static_argnames=("is_fp8", ))
def run(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
        is_fp8):
    scales = dict(k_scale=1.0, v_scale=1.0) if is_fp8 else {}
    return ragged_paged_attention(q,
                                  k,
                                  v,
                                  kv_cache,
                                  kv_lens,
                                  page_indices,
                                  cu_q_lens,
                                  distribution,
                                  sm_scale=HEAD_DIM**-0.5,
                                  **scales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    kv_dtype, num_kv_heads, kv_len = ARMS[args.arm]
    is_fp8 = kv_dtype == jnp.float8_e4m3fn
    wl, cache_shape = build(kv_dtype, num_kv_heads, kv_len)

    itemsize = jnp.dtype(kv_dtype).itemsize
    # Bytes the kernel must stream for the KV it attends to, padding included.
    packed_slots = cache_shape[2] * cache_shape[3]
    bytes_moved = BATCH * kv_len * packed_slots * HEAD_DIM * itemsize
    useful = BATCH * kv_len * (num_kv_heads * 2) * HEAD_DIM * itemsize

    out = run(**wl, is_fp8=is_fp8)
    jax.block_until_ready(out)

    for _ in range(args.warmup):
        out = run(**wl, is_fp8=is_fp8)
    jax.block_until_ready(out)

    # Time each iteration separately rather than averaging one long loop, so a
    # single contended outlier is visible instead of silently inflating a mean.
    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = run(**wl, is_fp8=is_fp8)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1e6)

    samples = np.array(samples)
    med = float(np.median(samples))

    print(f"ARM                 : {args.arm}")
    print(f"kv_dtype            : {jnp.dtype(kv_dtype).name}")
    print(f"num_kv_heads        : {num_kv_heads}")
    print(f"kv_len              : {kv_len}")
    print(f"cache_shape         : {tuple(cache_shape)}")
    print(f"packing slots       : {packed_slots} "
          f"(useful {num_kv_heads * 2}, padding {packed_slots - num_kv_heads * 2})")
    print(f"KV bytes/call       : {bytes_moved / 2**20:.1f} MiB "
          f"(useful {useful / 2**20:.1f} MiB)")
    print(f"iters               : {args.iters}")
    print(f"median us/call      : {med:.2f}")
    print(f"mean   us/call      : {samples.mean():.2f}")
    print(f"min    us/call      : {samples.min():.2f}")
    print(f"p90    us/call      : {np.percentile(samples, 90):.2f}")
    print(f"stdev  us/call      : {samples.std():.2f}")
    print(f"spread_pct (p90/med): {100 * (np.percentile(samples, 90) / med - 1):.2f}")
    print(f"eff GiB/s (moved)   : {bytes_moved / (med * 1e-6) / 2**30:.1f}")
    print(f"RESULT {args.arm} median_us={med:.2f} "
          f"bytes_mib={bytes_moved / 2**20:.1f}")


if __name__ == "__main__":
    main()
