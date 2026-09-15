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

MEASURED (job j-35ccee08, serial, one chip; arm A reproduces the production
xprof figure of 123.64 us/call to within 1.32x, so the harness is credible):

    A 163.75us   B 198.60us   C 159.17us   D 151.71us     (spread_pct < 0.65)

**The depadding hypothesis is falsified.** D halves the bytes and buys only
7.4%, not the ~2x a bandwidth-bound kernel would give. Fitting A and D, ~85%
of this op's time is byte-independent. A ~= C confirms fp8 KV currently costs
the same as bf16 -- but depadding it would not fix that, because bytes are not
what this kernel is spending its time on. B, at identical bytes and identical
FLOPs, is 21% *slower* than A, which no bandwidth story explains.

Where the time actually goes: the fetch is a **static** ``for i in
range(bkv_p)`` loop of guarded per-page DMAs, with ``bkv_p = bkv_sz //
page_size``. At page 256 and max_model_len 9216 that is 36 DMA issues per
sequence x 64 sequences = 2304 per call, whether or not each one moves data --
which is exactly why D, whose DMAs are predicated off, still pays the floor.
This op is DMA-issue-bound, not bandwidth-bound.

The P* arms follow from that: doubling page_size halves the DMA count while
moving identical bytes. The prior experiment log measured page 16 -> 256 as
+31.7% end-to-end and described it as saturating "between p128 and p256", but
that was concluded without testing beyond 256.

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
    # name: (kv_dtype, num_kv_heads, kv_len, page_size, d_block_sizes)
    # d_block_sizes is (bq_sz, bkv_sz, bq_csz, bkv_csz); None = kernel default,
    # which for this shape resolves to (1, 9216, 1, 9216).
    "A_fp8_1kvh": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, None),
    "B_fp8_2kvh": (jnp.float8_e4m3fn, 2, KV_LEN, PAGE_SIZE, None),
    "C_bf16_1kvh": (jnp.bfloat16, 1, KV_LEN, PAGE_SIZE, None),
    "D_fp8_1kvh_halfkv": (jnp.float8_e4m3fn, 1, KV_LEN // 2, PAGE_SIZE, None),
    # Page-size sweep. Arms A/B/C/D showed ~85% of this op's time is
    # byte-independent: halving the bytes (D) bought 7.4%, and fp8 costs the
    # same as bf16 (A vs C). The fetch is a *static* `for i in range(bkv_p)`
    # loop of guarded per-page DMAs, bkv_p = bkv_sz // page_size, so the floor
    # scales with the number of pages, not their contents: at page 256 and
    # max_model_len 9216 that is 36 DMA issues per sequence x 64 sequences.
    # Doubling the page halves the DMA count without changing bytes moved.
    #
    # The prior experiment log measured 16 -> 256 as +31.7% end-to-end and
    # called it saturated "between p128 and p256"; that conclusion was drawn
    # without testing past 256, which is what these arms do.
    "P128": (jnp.float8_e4m3fn, 1, KV_LEN, 128, None),
    "P256": (jnp.float8_e4m3fn, 1, KV_LEN, 256, None),
    "P512": (jnp.float8_e4m3fn, 1, KV_LEN, 512, None),
    "P1024": (jnp.float8_e4m3fn, 1, KV_LEN, 1024, None),
    # bkv compute-size sweep. `/xprof-llo` on arm P256 (job j-e3eade3f) found
    # that 67.8% of the kernel is a single 2173-bundle *straight-line* block:
    # the score matrix is materialized at the full bkv_csz and the kv axis is
    # unrolled into bkv_csz/128 chunks. Two consequences the earlier arms could
    # not separate, because both P* and A-D held bkv_csz fixed at 9216:
    #
    #   1. `num_loops = cdiv(effective_bkv_sz, bkv_csz)` is 1 for every page
    #      size, since bkv_csz = bkv_sz = pages_per_seq * page_size = 9216
    #      regardless of how the 9216 is divided into pages. That is why an 8x
    #      span in DMA count moved the clock 1.7%: every P* arm ran the
    #      *identical* 4x9216 VPU pass.
    #   2. 9216 is max_model_len, not kv_len. At kv_len 8192 the block computes
    #      1024 columns of mask-killed scores -- 11% of the pass -- and the cost
    #      does not fall when the sequence is shorter.
    #
    # Shrinking bkv_csz turns that one unrolled pass into `cdiv(kv_len, bkv_csz)`
    # trips of a rolled `pl.loop`, which both tracks the actual kv_len and cuts
    # the straight-line block (and its 749 spill stores) proportionally. It pays
    # for that with one flash-attention running-max/sum rescale per extra trip,
    # so the curve should be U-shaped and the minimum is an empirical question.
    #
    # bkv_sz is held at 9216 throughout, so the DMA/fetch path is byte-for-byte
    # unchanged and any delta is attributable to the compute block alone.
    # Constraint from _verify_block_sizes: bkv_sz % bkv_csz == 0 and
    # bkv_csz % page_size == 0.
    "C9216": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 9216, 1, 9216)),
    "C4608": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 9216, 1, 4608)),
    "C2304": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 9216, 1, 2304)),
    "C1536": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 9216, 1, 1536)),
    "C1024": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 9216, 1, 1024)),
    "C512": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 9216, 1, 512)),
    # MEASURED (job j-aaf016ab, serial, one chip, spread_pct <= 1.40):
    #   C9216 144.81  C4608 153.96  C2304 181.07
    #   C1536 212.16  C1024 239.83  C512  404.11
    # Monotone increasing, no interior minimum -- the hypothesis above is
    # refuted at every point. The per-trip flash-attention rescale costs far
    # more than the 11% of mask-killed columns it would avoid, and blows up at
    # the small end. The production default bkv_csz=9216 is already optimal.
    # (A_fp8_1kvh landed at 145.25, within spread of C9216, so passing
    # d_block_sizes explicitly is equivalent to the default path.)
    #
    # ---- Separating the two costs C* could not ----
    # C* moved bkv_csz, which changes *both* the padded-column count and the
    # trip count at once, and the trip count dominated. These two arms hold the
    # trip count at 1 and move the other two variables one at a time.
    #
    # bkv_sz = bkv_csz = 8192 = kv_len exactly: still a single trip (so no
    # rescale penalty), but the pass is 8192 columns instead of 9216, dropping
    # the 1024 mask-killed columns that max_model_len forces. Isolates the
    # padded-column cost -- predicted ~11% if the block is issue-bound.
    "M8192_masked": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 8192, 1, 8192),
                     False),
    # Same block sizes, but with the kv-length mask compiled out entirely.
    # `/xprof-llo` attributed 3170 int16 instructions -- ~25% of the hot block --
    # to building a 4x9216 iota and comparing it elementwise, to enforce what is
    # a single scalar bound. At bkv_csz == kv_len there is nothing to mask, so
    # `skip_kv_mask` is legal (its contract is use_causal_mask=False and every
    # dynamic kv_len % bkv_csz == 0, both of which hold here) and the result must
    # be numerically IDENTICAL to M8192_masked. The printed `checksum` is that
    # correctness gate: if the two arms disagree, the delta is the kernel
    # skipping work, not the mask being free, and must not be read as a speedup.
    #
    # NOTE this arm is a measurement, not a shippable config: in production the
    # per-sequence kv_lens differ and grow with each decoded token, so
    # `kv_len % bkv_csz == 0` does not hold and skip_kv_mask cannot be set. What
    # it prices is how much a *cheaper* mask would be worth -- the shippable
    # version of which is computing the bound on the scalar core instead of
    # materializing a 4x9216 iota, which is a kernel change, not a flag.
    "M8192_nomask": (jnp.float8_e4m3fn, 1, KV_LEN, PAGE_SIZE, (1, 8192, 1, 8192),
                     True),
}


def arm_spec(name):
    """Unpack an ARMS entry, defaulting the optional skip_mask field to False."""
    spec = ARMS[name]
    kv_dtype, num_kv_heads, kv_len, page_size, d_block_sizes = spec[:5]
    skip_mask = spec[5] if len(spec) > 5 else False
    return kv_dtype, num_kv_heads, kv_len, page_size, d_block_sizes, skip_mask


def build(kv_dtype, num_kv_heads, kv_len, page_size, seed=0):
    """Build a pure-decode workload: BATCH sequences, one query token each."""
    pages_per_seq = (MAX_MODEL_LEN + page_size - 1) // page_size
    num_blocks = BATCH * pages_per_seq

    cache_shape = get_kv_cache_shape(num_blocks, page_size, num_kv_heads,
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


@functools.partial(jax.jit,
                   static_argnames=("is_fp8", "inner", "d_block_sizes",
                                    "skip_mask"),
                   donate_argnames=("kv_cache", ))
def run(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
        is_fp8, inner, d_block_sizes, skip_mask):
    """Run the kernel ``inner`` times inside a single dispatch.

    A one-call-per-`block_until_ready` loop measures the host round trip, not
    the kernel: the first version of this script clocked every arm at ~700 us
    against the 123.64 us/call the production xprof reports for the same op,
    and the half-bytes arm came out *slower* than the full-bytes one. Chaining
    the calls through ``kv_cache`` inside ``fori_loop`` serializes them (so XLA
    cannot overlap or hoist iterations) and amortizes the dispatch away.

    ``kv_cache`` is donated. Without it XLA inserts a defensive `copy.21` of the
    whole 302 MB cache because the buffer is an undonated input that the kernel
    mutates; `/xprof-llo` measured that copy at 335.9 us, i.e. 16.8 us amortized
    over --inner 20 -- about 10% of the reported per-call figure, and entirely
    harness. The caller must thread the returned cache into the next call.
    """
    scales = dict(k_scale=1.0, v_scale=1.0) if is_fp8 else {}
    # skip_kv_mask's contract requires use_causal_mask=False alongside it.
    mask_kw = dict(use_causal_mask=False,
                   skip_kv_mask=True) if skip_mask else {}

    def body(_, carry):
        acc, cache = carry
        out, cache = ragged_paged_attention(q,
                                            k,
                                            v,
                                            cache,
                                            kv_lens,
                                            page_indices,
                                            cu_q_lens,
                                            distribution,
                                            sm_scale=HEAD_DIM**-0.5,
                                            d_block_sizes=d_block_sizes,
                                            **mask_kw,
                                            **scales)
        # Keep `out` live so the attention compute cannot be dead-coded; the
        # reduction is ~32K elements against a 256 MiB KV read.
        return acc + out.astype(jnp.float32).sum(), cache

    return jax.lax.fori_loop(0, inner, body,
                             (jnp.float32(0.0), kv_cache))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--iters", type=int, default=20,
                    help="timed outer samples; each is --inner kernel calls")
    ap.add_argument("--inner", type=int, default=20,
                    help="kernel calls per dispatch, to amortize host overhead")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    (kv_dtype, num_kv_heads, kv_len, page_size, d_block_sizes,
     skip_mask) = arm_spec(args.arm)
    is_fp8 = kv_dtype == jnp.float8_e4m3fn
    wl, cache_shape = build(kv_dtype, num_kv_heads, kv_len, page_size)
    pages_per_seq = (MAX_MODEL_LEN + page_size - 1) // page_size
    # Donated, so it must be threaded call to call rather than reused from `wl`.
    cache = wl.pop("kv_cache")

    itemsize = jnp.dtype(kv_dtype).itemsize
    # Bytes the kernel must stream for the KV it attends to, padding included.
    packed_slots = cache_shape[2] * cache_shape[3]
    bytes_moved = BATCH * kv_len * packed_slots * HEAD_DIM * itemsize
    useful = BATCH * kv_len * (num_kv_heads * 2) * HEAD_DIM * itemsize

    call = functools.partial(run,
                             **wl,
                             is_fp8=is_fp8,
                             inner=args.inner,
                             d_block_sizes=d_block_sizes,
                             skip_mask=skip_mask)

    acc, cache = call(kv_cache=cache)
    jax.block_until_ready((acc, cache))
    # Captured from the very first call, before the cache has been mutated by
    # any later one, so it is comparable across arms that share a shape.
    checksum = float(acc)

    for _ in range(args.warmup):
        acc, cache = call(kv_cache=cache)
    jax.block_until_ready((acc, cache))

    # Time each sample separately rather than averaging one long loop, so a
    # single contended outlier is visible instead of silently inflating a mean.
    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        acc, cache = call(kv_cache=cache)
        jax.block_until_ready((acc, cache))
        samples.append((time.perf_counter() - t0) * 1e6 / args.inner)

    samples = np.array(samples)
    med = float(np.median(samples))

    print(f"ARM                 : {args.arm}")
    print(f"kv_dtype            : {jnp.dtype(kv_dtype).name}")
    print(f"num_kv_heads        : {num_kv_heads}")
    print(f"kv_len              : {kv_len}")
    print(f"page_size           : {page_size}")
    print(f"pages/seq           : {pages_per_seq}")
    print(f"page DMAs/call      : {pages_per_seq * BATCH}")
    print(f"d_block_sizes       : {d_block_sizes or 'kernel default (1,9216,1,9216)'}")
    if d_block_sizes is not None:
        print(f"bkv compute trips   : {-(-kv_len // d_block_sizes[3])}")
    print(f"skip_kv_mask        : {skip_mask}")
    # Arms sharing a shape and seed must agree here. A masked and an unmasked
    # arm whose bkv_csz equals kv_len have nothing to mask, so a mismatch means
    # the faster one dropped work and its time is not a speedup.
    print(f"checksum            : {checksum:.6f}")
    print(f"cache_shape         : {tuple(cache_shape)}")
    print(f"packing slots       : {packed_slots} "
          f"(useful {num_kv_heads * 2}, padding {packed_slots - num_kv_heads * 2})")
    print(f"KV bytes/call       : {bytes_moved / 2**20:.1f} MiB "
          f"(useful {useful / 2**20:.1f} MiB)")
    print(f"iters x inner       : {args.iters} x {args.inner}")
    print(f"median us/call      : {med:.2f}")
    print(f"mean   us/call      : {samples.mean():.2f}")
    print(f"min    us/call      : {samples.min():.2f}")
    print(f"p90    us/call      : {np.percentile(samples, 90):.2f}")
    print(f"stdev  us/call      : {samples.std():.2f}")
    print(f"spread_pct (p90/med): {100 * (np.percentile(samples, 90) / med - 1):.2f}")
    print(f"eff GiB/s (moved)   : {bytes_moved / (med * 1e-6) / 2**30:.1f}")
    # Sanity gate. xprof on j-bf2efb16 measured this exact op (arm A's shape)
    # at 123.64 us/call in production. If arm A does not land near that, the
    # harness is measuring something other than the kernel and no cross-arm
    # comparison from this run should be believed.
    #
    # NOTE: donating kv_cache removed a ~16.8 us/call harness copy, so numbers
    # from this revision are NOT comparable to the A/B/C/D and P* figures in the
    # module docstring (job j-35ccee08, j-cdf62466), which were taken without
    # donation. Compare within a run only; C9216 is the baseline for the C* arms.
    if args.arm == "A_fp8_1kvh":
        print(f"xprof reference     : 123.64 us/call (j-bf2efb16, decode_only)")
        print(f"ratio to reference  : {med / 123.64:.2f}x")
    print(f"RESULT {args.arm} median_us={med:.2f} "
          f"bytes_mib={bytes_moved / 2**20:.1f}")


if __name__ == "__main__":
    main()
