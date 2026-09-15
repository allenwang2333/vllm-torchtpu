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
every array, block size, seed and page table is shared.

Separating the arms is harder than it looks, and the first version of this
script failed at it: the flag is read inside a *nested* ``@jax.jit``
(``wrapper.ragged_paged_attention``) whose cache key does not include it, so
both arms ran one binary and the resulting -0.662% was the difference between
two HBM allocations. ``compile_arms`` now clears the cache between lowerings and
**refuses to run unless the two arms' compiled HLO differs**. A benchmark whose
control cannot fail is not a benchmark; treat that assertion as the first
result, not as setup.

That fix removed the mechanism but did not, by itself, retire the number: the
retracted A/A reported -0.662% with two provably identical binaries, so the
harness has a demonstrated bias of the same sign and size as the effect being
claimed. ``--mode aa`` measures that null directly, and ``--swap-alloc``
exchanges the arm->buffer binding, which the pair-order flip does not touch
because it is bound to the arm rather than to the call position. **Run both
before believing an A/B number from this script.**

The expected effect is
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
    """Build a *fresh* jitted callable, one per arm.

    A fresh outer object is **necessary but not sufficient**, and an earlier
    version of this script got that wrong and reported a number for it. The flag
    is read inside ``KB.ragged_paged_attention``, which is itself ``@jax.jit`` at
    module scope (``wrapper.py:375``) with ``static_argnames`` that contain
    nothing varying between arms. Both arms therefore hit the *inner* cache on
    the same key, the second arm silently reuses the first arm's kernel, and the
    reported delta is the difference between two copies of one binary. See
    ``compile_arms`` for the part that actually separates them.
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


def compile_arms(runners, wl, caches, inner, arm_flags, require_equal=False):
    """Compile each arm to its own executable, and prove they differ.

    ``jax.clear_caches()`` between the two lowerings is what forces a second
    trace: it drops the inner ``ragged_paged_attention`` cache entry that would
    otherwise be reused. Compiling ahead of time rather than clearing around
    every call is deliberate -- clearing inside the timing loop would recompile
    on each call and time the compiler. An already-built ``Compiled`` holds its
    own executable and survives a later ``clear_caches``.

    The HLO comparison is the assertion this script most needed and did not
    have. It is a *negative control on the harness*: if the two arms are the
    same binary it fails here, loudly, instead of reporting the difference
    between two allocations as a speedup.
    """
    compiled = {}
    original = flash_attention._COLLAPSE_DECODE_MASK
    try:
        for name in ARMS:
            flash_attention._COLLAPSE_DECODE_MASK = arm_flags[name]
            jax.clear_caches()
            compiled[name] = runners[name].lower(**wl,
                                                 kv_cache=caches[name],
                                                 inner=inner).compile()
    finally:
        # Leave the module as found. Harmless while everything downstream is
        # AOT-compiled, but a landmine for anything added after this point.
        flash_attention._COLLAPSE_DECODE_MASK = original

    texts = {name: compiled[name].as_text() for name in ARMS}
    same = texts["collapsed"] == texts["general"]
    if require_equal and not same:
        raise SystemExit(
            "FAIL: the A/A arms compiled to *different* HLO. They were given "
            "the same flag, so this run cannot measure the harness's null.")
    if not require_equal and same:
        raise SystemExit(
            "FAIL: both arms compiled to byte-identical HLO, so the flag never "
            "reached a second trace and any delta below would be noise between "
            "two copies of one binary. Do not report a number from this run.")
    print(f"arms differ (HLO)   : {not same}  "
          f"({len(texts['collapsed'])} vs {len(texts['general'])} chars)"
          f"{'  [A/A: equality required and held]' if require_equal else ''}")
    return compiled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=10,
                    help="interleaved A/B pairs; the CI is over these")
    ap.add_argument("--inner", type=int, default=20,
                    help="kernel calls per dispatch, to amortize host overhead")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--mode", choices=("ab", "aa"), default="ab",
        help="'aa' gives both arms the same flag, so the true delta is zero "
             "and whatever this reports is the harness's own bias. Run it "
             "before believing any 'ab' number: a null of the same sign and "
             "size as the effect means the effect is not established.")
    ap.add_argument(
        "--swap-alloc", action="store_true",
        help="allocate the arms' caches in reverse order. The arm->buffer "
             "binding is otherwise fixed ('collapsed' is always allocated "
             "first), so a bias tied to allocation order rather than to code "
             "would survive the pair-order flip and masquerade as the effect. "
             "If the delta reverses sign under this flag, it is allocation.")
    args = ap.parse_args()

    # In A/A both arms get the *collapsed* flag: the null must be measured on
    # the arm whose number is being claimed.
    arm_flags = dict(ARMS) if args.mode == "ab" else {n: True for n in ARMS}

    wl, cache_np, cache_shape = build()
    runners = {name: make_runner() for name in ARMS}
    # Independent device copies of identical data: identical so the checksums
    # are comparable, independent so one arm's donation cannot invalidate the
    # other's buffer. Allocation order is a real variable, not bookkeeping --
    # two 302 MB buffers land at different addresses against a shuffled page
    # table, and whichever is allocated first keeps that advantage for the
    # whole run regardless of which order the *calls* are made in.
    caches = {}
    for name in (list(ARMS)[::-1] if args.swap_alloc else list(ARMS)):
        caches[name] = jnp.asarray(cache_np, dtype=KV_DTYPE)
    checksums: dict[str, float] = {}

    print(f"mode                : {args.mode}"
          f"{' (swapped allocation order)' if args.swap_alloc else ''}")
    compiled = compile_arms(runners, wl, caches, args.inner, arm_flags,
                            require_equal=(args.mode == "aa"))

    def call(name):
        # `inner` was static at lowering time and is baked into the executable,
        # so it must *not* be passed again here -- an AOT `Compiled` matches the
        # caller's pytree against the lowered one and rejects the extra arg.
        acc, caches[name] = compiled[name](**wl, kv_cache=caches[name])
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
        # t(0.975), keyed by *degrees of freedom*; a table rather than a scipy
        # dependency. An earlier version keyed this table by n but looked it up
        # by n-1, and fell back to the normal 1.96 for any df it did not list --
        # which at n=40 quietly reported a z-interval under a "t-based" label.
        # Missing df now takes the largest tabulated df below it, so the
        # fallback is always conservative rather than anti-conservative.
        df = len(deltas) - 1
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 14: 2.145,
            19: 2.093, 29: 2.045, 39: 2.023, 59: 2.001, 119: 1.980,
        }
        tcrit = table[max(k for k in table if k <= df)] if df >= 1 else 12.706
        lo, hi = mean_d - tcrit * se, mean_d + tcrit * se
        print(f"mean delta          : {mean_d:+.3f}%  "
              f"95% CI [{lo:+.3f}%, {hi:+.3f}%]  (n={len(deltas)} pairs)")
        print(f"SIGNIFICANT         : {hi < 0}  "
              f"(negative = collapsed is faster)")
        if args.mode == "aa":
            # In A/A the true delta is zero by construction, so a significant
            # result is a statement about the harness, not the kernel.
            biased = hi < 0 or lo > 0
            print(f"HARNESS BIAS        : {biased}  "
                  f"(A/A: the true delta is 0, so a CI excluding 0 is bias; "
                  f"any 'ab' effect inside [{lo:+.3f}%, {hi:+.3f}%] is not "
                  f"established)")
    else:
        print(f"mean delta          : {mean_d:+.3f}%  (n=1, no CI)")

    if not identical:
        raise SystemExit("FAIL: arms disagree; the timing is not a speedup")


if __name__ == "__main__":
    main()
