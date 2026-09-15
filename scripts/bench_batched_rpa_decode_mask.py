# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: the collapsed decode mask in the batched RPA kernel.

Why this exists
---------------
``RPAd-p256-b8-q1-k3840`` is **44.59% of decode device time** on
Qwen3-Coder-30B-A3B-Instruct-FP8 at TP=8 -- 95.006 us/call over 480 events per
rank-file, 2.6x the next-largest op, with nothing above it (job
j-0b567614-04cb-479f-a3c1, xprof ``hlo_stats``). That op is the batched RPA
kernel, which is what ``--attention-backend CUSTOM`` selects and what this
checkpoint's nightly config pins.

Inside it, ``flash_attention_qk_softmax`` built its causal mask from two
full-shape index tensors per batch lane. At ``bq_c_sz == 1`` -- every decode
dispatch -- the q-side tensor is a broadcast scalar, so the whole mask collapses
to one shared iota against a per-lane scalar bound. Traced at s=256, batch 8,
the rewrite takes iota material from 16,384 elements to 1,024 and removes the
full-shape integer floor-division entirely (the general path's ``//`` lowers to
``div`` + ``rem`` + ``sign`` + ``ne``, 8 to 16 tile-sized ops each).

What this measures, and what it cannot
--------------------------------------
This is a **paired, interleaved A/B in a single process**. The two arms differ
only in the trace-time constant ``flash_attention._COLLAPSE_DECODE_MASK``;
every array, block size, seed and page table is shared. The expected effect is
low single-digit percent, which is the regime where one-arm-per-process
benchmarking measures process placement rather than the kernel -- so the arms
alternate within a pair, and the pair order flips every other pair to cancel
any monotone drift.

The reported statistic is the mean of the **per-pair** percentage deltas, with
a t-based 95% CI over pairs. A whole-run mean of each arm separately would be
the wrong estimator here: it throws away the pairing that makes the comparison
tight.

Both arms must produce a **bit-identical** checksum. They compute the same
boolean mask by different arithmetic, so anything else means the rewrite
changed behaviour and its timing is not a speedup. The script exits non-zero if
they differ.

This measures one kernel in isolation on an idle chip. It is not an end-to-end
result, and the 44.59% share above is what converts it into one.
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import flash_attention
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB

# Production decode shape, per rank, for Qwen3-Coder-30B-A3B-Instruct-FP8 at
# TP=8. Cross-check: the xprof row for this op is `RPAd-p256-b8-q1-k3840`,
# i.e. page 256, batch 8 lanes, 1 query token, bkv 3840 -- which is what
# `wrapper.calculate_block_sizes` returns for this configuration and what
# DECODE_BLOCKS below pins.
PAGE_SIZE = 256
HEAD_DIM = 128
NUM_Q_HEADS = 4  # 32 attention heads / TP 8
NUM_KV_HEADS = 1  # 4 KV heads replicated across 8 ranks
BATCH = 64  # --max-num-seqs=64, all decoding
KV_LEN = 8192  # --random-input-len=8192
MAX_MODEL_LEN = 9216  # --max-model-len=9216
KV_DTYPE = jnp.float8_e4m3fn  # --kv-cache-dtype=fp8
Q_DTYPE = jnp.bfloat16

DECODE_BLOCKS = KBC.BlockSizes(bq_sz=1,
                               bq_c_sz=1,
                               bkv_sz=3840,
                               batch_size=8,
                               n_buffer=3)

ARMS = {"collapsed": True, "general": False}


def build(seed: int = 0):
    rng = np.random.default_rng(seed)
    pages_per_seq = (MAX_MODEL_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = BATCH * pages_per_seq

    cache_shape = KB.get_kv_cache_shape(num_blocks, PAGE_SIZE, NUM_KV_HEADS,
                                        HEAD_DIM, KV_DTYPE)
    # Kept on the host. Each arm donates its cache, which invalidates the
    # buffer, so the two arms must be handed independent device copies made
    # from this one source rather than two references to the same array.
    kv_cache_np = rng.uniform(-1.0, 1.0, cache_shape)

    q = jnp.asarray(rng.uniform(-1.0, 1.0, (BATCH, NUM_Q_HEADS, HEAD_DIM)),
                    dtype=Q_DTYPE)
    # K/V at the cache dtype, not the query dtype: `validate_inputs` requires
    # them to match the cache. Queries are the exception -- the kernel reads
    # `queries.dtype` as `dtype_q` and quantizes them internally, so bf16 here
    # is what the served model actually passes.
    k = jnp.asarray(rng.uniform(-1.0, 1.0, (BATCH, NUM_KV_HEADS, HEAD_DIM)),
                    dtype=KV_DTYPE)
    v = jnp.asarray(rng.uniform(-1.0, 1.0, (BATCH, NUM_KV_HEADS, HEAD_DIM)),
                    dtype=KV_DTYPE)

    kv_lens = jnp.full((BATCH, ), KV_LEN, dtype=jnp.int32)
    # Shuffled so the page walk is not artificially sequential -- a contiguous
    # block table would let the DMA engine prefetch in a way production cannot.
    pool = np.arange(num_blocks, dtype=np.int32)
    rng.shuffle(pool)
    page_indices = jnp.asarray(pool[:BATCH * pages_per_seq])
    cu_q_lens = jnp.asarray(np.arange(BATCH + 1, dtype=np.int32))
    # Cumulative (decode, prefill, mixed): every sequence is decode.
    distribution = jnp.asarray([BATCH, BATCH, BATCH], dtype=jnp.int32)

    return dict(q=q,
                k=k,
                v=v,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution), kv_cache_np, cache_shape


def make_runner():
    """Build a *fresh* jitted callable.

    One per arm, deliberately. ``_COLLAPSE_DECODE_MASK`` is read at trace time,
    so the arm is baked into the compiled binary -- and `jax.jit` caches on the
    function object, so reusing a single runner for both arms would silently
    hand the second arm the first arm's binary and report a 0.0% delta.
    """

    @functools.partial(jax.jit,
                       static_argnames=("inner", ),
                       donate_argnames=("kv_cache", ))
    def run(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
            inner):
        """Run the kernel ``inner`` times inside a single dispatch.

        A one-call-per-`block_until_ready` loop measures the host round trip
        rather than the kernel. Chaining through ``kv_cache`` inside
        ``fori_loop`` serializes the calls (so XLA cannot overlap or hoist
        iterations) and amortizes the dispatch away. ``kv_cache`` is donated to
        stop XLA inserting a defensive whole-cache copy, which on the v3
        version of this harness cost ~16.8 us/call of pure harness time.
        """

        def body(_, carry):
            acc, cache = carry
            out, cache = KB.ragged_paged_attention(
                q,
                k,
                v,
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=HEAD_DIM**-0.5,
                decode_block_sizes=DECODE_BLOCKS,
            )
            # Keep `out` live so the attention compute cannot be dead-coded.
            return acc + out.astype(jnp.float32).sum(), cache

        return jax.lax.fori_loop(0, inner, body, (jnp.float32(0.0), kv_cache))

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=10,
                    help="interleaved A/B pairs; the CI is over these")
    ap.add_argument("--inner", type=int, default=20,
                    help="kernel calls per dispatch, to amortize host overhead")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    wl, cache_np, cache_shape = build()
    runners = {name: make_runner() for name in ARMS}
    # Independent device copies of identical data: identical so the checksums
    # are comparable, independent so one arm's donation cannot invalidate the
    # other's buffer.
    caches = {
        name: jnp.asarray(cache_np, dtype=KV_DTYPE)
        for name in ARMS
    }
    checksums: dict[str, float] = {}

    def call(name):
        # Set before the call because it is read while tracing; after the first
        # call per arm the binary is cached and this is a no-op.
        flash_attention._COLLAPSE_DECODE_MASK = ARMS[name]
        acc, caches[name] = runners[name](**wl,
                                          kv_cache=caches[name],
                                          inner=args.inner)
        jax.block_until_ready((acc, caches[name]))
        return float(acc)

    # Compile and capture the checksum from the very first call of each arm,
    # before either cache has been mutated, so the two are comparable.
    for name in ARMS:
        checksums[name] = call(name)

    for _ in range(args.warmup):
        for name in ARMS:
            call(name)

    samples: dict[str, list[float]] = {name: [] for name in ARMS}
    for i in range(args.pairs):
        # Flip the within-pair order every other pair so that any monotone
        # drift (thermal, neighbour contention) cancels rather than
        # accumulating into whichever arm always goes first.
        order = list(ARMS) if i % 2 == 0 else list(ARMS)[::-1]
        for name in order:
            t0 = time.perf_counter()
            call(name)
            samples[name].append(
                (time.perf_counter() - t0) * 1e6 / args.inner)

    pages_per_seq = (MAX_MODEL_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"shape               : batch {BATCH}, kv_len {KV_LEN}, "
          f"page {PAGE_SIZE}, q_heads {NUM_Q_HEADS}, kv_heads {NUM_KV_HEADS}")
    print(f"blocks              : {DECODE_BLOCKS}")
    print(f"kv_dtype / q_dtype  : {jnp.dtype(KV_DTYPE).name} / "
          f"{jnp.dtype(Q_DTYPE).name}")
    print(f"cache_shape         : {tuple(cache_shape)}")
    print(f"pages/seq           : {pages_per_seq}")
    print(f"pairs x inner       : {args.pairs} x {args.inner}")
    print()

    for name in ARMS:
        s = np.array(samples[name])
        print(f"{name:10} median {np.median(s):8.2f} us/call  "
              f"mean {s.mean():8.2f}  min {s.min():8.2f}  "
              f"p90 {np.percentile(s, 90):8.2f}  "
              f"spread_pct {100 * (np.percentile(s, 90) / np.median(s) - 1):5.2f}")
    print()

    # Checksums must match *exactly*: the two paths compute the same boolean
    # mask, so any difference is a behaviour change, not a speedup.
    c_col, c_gen = checksums["collapsed"], checksums["general"]
    print(f"checksum collapsed  : {c_col!r}")
    print(f"checksum general    : {c_gen!r}")
    identical = (c_col == c_gen) or (np.isnan(c_col) and np.isnan(c_gen))
    if np.isnan(c_col) or np.isnan(c_gen):
        print("CHECKSUM: NaN -- the comparison is meaningless, not passing")
        identical = False
    print(f"CHECKSUM MATCH      : {identical}")

    # Paired statistic. The pairing is the whole point: it removes the
    # between-sample variance that both arms share.
    deltas = [
        100.0 * (c - g) / g
        for c, g in zip(samples["collapsed"], samples["general"])
    ]
    mean_d = statistics.fmean(deltas)
    print()
    print(f"per-pair deltas (%) : "
          f"{', '.join(f'{d:+.2f}' for d in deltas)}")
    if len(deltas) > 1:
        sd = statistics.stdev(deltas)
        se = sd / len(deltas)**0.5
        # t(0.975) for small n; table rather than a scipy dependency.
        tcrit = {
            2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 15: 2.145,
            20: 2.093, 30: 2.045,
        }.get(len(deltas) - 1, 1.96)
        lo, hi = mean_d - tcrit * se, mean_d + tcrit * se
        print(f"mean delta          : {mean_d:+.3f}%  "
              f"95% CI [{lo:+.3f}%, {hi:+.3f}%]  (n={len(deltas)} pairs)")
        print(f"SIGNIFICANT         : {hi < 0}  "
              f"(negative = collapsed is faster)")
    else:
        print(f"mean delta          : {mean_d:+.3f}%  (n=1, no CI)")

    if not identical:
        raise SystemExit("FAIL: arms disagree; the timing is not a speedup")


if __name__ == "__main__":
    main()
