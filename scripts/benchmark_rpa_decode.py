#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Microbenchmark the ragged-paged-attention decode pass (`RPAd`).

Default shape is one TP=8 shard of Qwen3-Coder-30B-A3B-Instruct-FP8 decoding at
its steady state, as measured in xprof job ``j-bf2efb16`` (commit ``c70336be``):
64 sequences, q_len 1, kv_len 8192, page_size 256, 1 KV head, 4 Q heads,
head_dim 128, fp8 KV cache.  xprof names that kernel
``RPAd-p_256-bq_1_1-bkv_9216_9216`` and attributes 53.1% of decode device time
and ~33.6% of end-to-end device time to it.

Why the knobs below are the ones exposed
----------------------------------------
``get_default_block_sizes`` picks the decode block sizes as::

    bkv_sz = bkv_csz = min(min_bkv_sz_to_peak, pages_per_seq * page_size)

which at ``--max-model-len 9216`` and page 256 saturates at the *padded* max
sequence length, 9216, not at the live ``kv_len``.  Two consequences are
measurable here and neither needs a kernel edit, because ``d_block_sizes`` is
already a parameter of ``ragged_paged_attention``:

* ``bkv_p = bkv_sz // page_size = 36`` per-page DMAs are issued from a *static*
  Python loop (``v3/kernel.py:599``) for every sequence, whatever ``kv_len`` is.
  Pages past the live length still cost a descriptor; only their byte count is
  clipped to zero.  A per-sequence descriptor cost that does not shrink with
  ``kv_len`` is exactly an additive constant, so ``--kv-len-sweep`` fits
  ``t(kv_len) = intercept + slope*kv_len`` to size it.
* ``bkv_csz = 9216`` makes the flash-attention block statically 9216 wide, so
  at kv_len 8192 the QK^T and PV matmuls cover 1024 masked-off columns.

``--sweep`` therefore walks ``bkv_sz``/``bkv_csz`` to test whether 9216 is
actually the best choice at the production point, before anyone edits a kernel.

Two properties this harness is deliberate about:

* **It reports which kernel it timed.** ``d_block_sizes`` is baked into the
  Pallas scope name, so ``--assert-kernel`` greps the compiled HLO for the
  expected ``RPAd-p_<page>-bq_1_1-bkv_<sz>_<csz>`` and fails if the variant that
  compiled is not the one that was asked for.
* **It checks every variant against the XLA reference.** A block size that
  silently drops KV would be fast and wrong.

Usage:
  python3 scripts/benchmark_rpa_decode.py --assert-kernel
  python3 scripts/benchmark_rpa_decode.py --sweep --chain 100
  python3 scripts/benchmark_rpa_decode.py --kv-len-sweep 512,1024,2048,4096,8192
"""

import argparse
import re
import statistics
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase, get_default_block_sizes, get_kv_cache_shape,
    ragged_paged_attention, ref_ragged_paged_attention)

_KV_DTYPES = {
    "fp8": jnp.float8_e4m3fn,
    "bf16": jnp.bfloat16,
}


def _make_workload(num_seqs, kv_len, page_size, pages_per_seq, total_pages,
                   num_q_heads, num_kv_heads, head_dim, q_dtype, kv_dtype,
                   seed):
    """Build one decode step: ``num_seqs`` sequences of q_len 1 at ``kv_len``.

    Page indices are drawn from a shuffled pool spanning the whole cache, the
    way vLLM's block allocator leaves them after a long run -- a sequence's 36
    pages are scattered, so the per-page DMAs are a genuine gather rather than
    a contiguous read that the memory system could coalesce.
    """
    rng = np.random.default_rng(seed)

    q_lens = [1] * num_seqs
    cu_q_lens = np.cumsum([0] + q_lens, dtype=np.int32)
    num_tokens = int(cu_q_lens[-1])
    # Decode-only: seqs[0:n] decode, no prefill, no mixed.
    distribution = np.array([num_seqs, num_seqs, num_seqs], dtype=np.int32)

    kv_lens = np.full(num_seqs, kv_len, dtype=np.int32)

    need = num_seqs * pages_per_seq
    if total_pages < need:
        raise ValueError(f"--total-pages {total_pages} < {need} required for "
                         f"{num_seqs} seqs x {pages_per_seq} pages/seq")
    pool = rng.permutation(total_pages).astype(np.int32)
    page_indices = pool[:need]

    cache_shape = get_kv_cache_shape(total_pages, page_size, num_kv_heads,
                                     head_dim, kv_dtype)

    def _rand(shape):
        return rng.standard_normal(shape, dtype=np.float32)

    # The cache is the thing being streamed; fill it with real values so the
    # softmax is not degenerate.
    kv_cache = jnp.asarray(_rand(cache_shape) * 0.1, dtype=kv_dtype)

    return dict(
        q=jnp.asarray(_rand((num_tokens, num_q_heads, head_dim)),
                      dtype=q_dtype),
        # The kernel requires k/v to match the cache dtype -- it asserts
        # `kv_cache.dtype == k.dtype == v.dtype` (v3/kernel.py:1425), because
        # KV quantization happens outside RPA. These are the new tokens this
        # decode step writes into the cache, so they are already quantized.
        k=jnp.asarray(_rand((num_tokens, num_kv_heads, head_dim)),
                      dtype=kv_dtype),
        v=jnp.asarray(_rand((num_tokens, num_kv_heads, head_dim)),
                      dtype=kv_dtype),
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
    )


def _call(wl, d_block_sizes, sm_scale):
    """One `ragged_paged_attention` decode call. Returns (out, kv_cache)."""
    return ragged_paged_attention(
        wl["q"],
        wl["k"],
        wl["v"],
        wl["kv_cache"],
        wl["kv_lens"],
        wl["page_indices"],
        wl["cu_q_lens"],
        wl["distribution"],
        sm_scale=sm_scale,
        d_block_sizes=d_block_sizes,
    )


class _Bound:
    """Binds `distribution` so a traced-distribution chain still takes only the
    KV cache, preserving `.lower()` for `_compiled_scopes`."""

    def __init__(self, fn, distribution):
        self._fn = fn
        self._distribution = distribution

    def __call__(self, kv_cache):
        return self._fn(kv_cache, self._distribution)

    def lower(self, kv_cache):
        return self._fn.lower(kv_cache, self._distribution)


def _closed_over(wl, trace_distribution=False):
    """Everything but the KV cache, to be captured rather than passed.

    Closing over the workload makes it a compile time constant, and nothing
    here varies across iterations anyway.

    ``distribution`` is the one entry where that choice is not neutral, and it
    is controlled by ``trace_distribution``:

    * Closed over (the default), it is a compile-time constant, so XLA folds
      away every pass whose sequence range is statically empty.  On a
      decode-only workload the compiled HLO then contains ``RPAd`` alone.
    * Traced, it reaches the kernel as a value, which is what production does
      -- the batch composition is not known at compile time.  All the passes
      `ragged_paged_attention` launches survive into the HLO, including the
      MIXED one whose range is empty on a decode-only step.

    So the default is the faster measurement and the traced one is the
    *representative* one; anything about per-pass launch overhead has to be
    measured with ``trace_distribution=True`` or there is no empty pass left
    to observe.

    Note this only applies to the timed path.  The XLA reference still needs a
    concrete ``distribution``: `dynamic_validate_inputs` branches on it in
    Python (``if not (i <= j <= k)``, `v3/kernel.py:1311`) and raises
    `TracerBoolConversionError` on a tracer.  `_reference` runs eagerly for
    that reason and is unaffected by this flag.
    """
    drop = {"kv_cache"} | ({"distribution"} if trace_distribution else set())
    return {k: v for k, v in wl.items() if k not in drop}


def _build_chain(wl, d_block_sizes, sm_scale, chain, trace_distribution=False):
    """A jit of `chain` back-to-back decode calls, threading the KV cache.

    Chaining inside a single jit is what keeps the measurement off the host
    dispatch path: at ~120 us/call a per-call round trip is a large fraction of
    the thing being measured.  The cache is threaded because the kernel donates
    and updates it; q/k/v are loop-invariant and tiny (64 x 4 x 128).
    """
    static = _closed_over(wl, trace_distribution)

    if not trace_distribution:

        @jax.jit
        def run(kv_cache):
            out = None
            for _ in range(chain):
                args = dict(static, kv_cache=kv_cache)
                out, kv_cache = _call(args, d_block_sizes, sm_scale)
            return out, kv_cache

        return run

    @jax.jit
    def run_traced(kv_cache, distribution):
        out = None
        for _ in range(chain):
            args = dict(static, kv_cache=kv_cache, distribution=distribution)
            out, kv_cache = _call(args, d_block_sizes, sm_scale)
        return out, kv_cache

    # Keep the one-argument calling convention the timing loop and the scope
    # dump both expect, rather than threading the flag through each of them.
    return _Bound(run_traced, wl["distribution"])


def _compiled_scopes(run, kv_cache):
    """Return the RPA scope names baked into the compiled HLO."""
    try:
        text = run.lower(kv_cache).compile().as_text()
    except Exception as exc:  # pragma: no cover - diagnostic path
        # Print the stack rather than only the message: a compile failure here
        # is a bug in the harness or the block sizes, and the one-line message
        # rarely says which.
        traceback.print_exc()
        return None, f"{type(exc).__name__}: {exc}"
    return sorted(set(re.findall(r"RPA[dpm]-p_\d+-bq_\d+_\d+-bkv_\d+_\d+",
                                 text))), None


def _time(run, kv_cache, iters, warmup, chain):
    out = jax.block_until_ready(run(kv_cache))
    for _ in range(warmup - 1):
        out = run(kv_cache)
    jax.block_until_ready(out)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(run(kv_cache))
        samples.append((time.perf_counter() - t0) * 1e6 / chain)
    return samples


def _reference(wl, sm_scale):
    """The XLA reference, run eagerly.

    It must not be wrapped in `jax.jit`: `dynamic_validate_inputs` unstacks
    `distribution` and branches on it in Python (`if not (i <= j <= k)`,
    `v3/kernel.py:1311`), and under a trace even an op on a closed-over
    constant yields a tracer, so jitting it raises
    `TracerBoolConversionError` however the argument is passed.  Eagerly the
    entries are concrete and the check does what it was written to do.  This
    costs per-op dispatch, which does not matter: the reference is computed
    once per variant and is never the thing being timed.
    """
    return ref_ragged_paged_attention(wl["q"], wl["k"], wl["v"],
                                      wl["kv_cache"], wl["kv_lens"],
                                      wl["page_indices"], wl["cu_q_lens"],
                                      wl["distribution"], sm_scale=sm_scale)


def _check(wl, d_block_sizes, sm_scale, rtol, atol):
    """Single (unchained) call compared against the XLA reference."""
    s = _closed_over(wl)

    @jax.jit
    def run(kv_cache):
        return _call(dict(s, kv_cache=kv_cache), d_block_sizes, sm_scale)

    ref, _ = _reference(wl, sm_scale)
    got, _ = run(wl["kv_cache"])
    r = np.asarray(ref, dtype=np.float32)
    g = np.asarray(got, dtype=np.float32)
    if r.shape != g.shape:
        return False, f"shape {g.shape} != ref {r.shape}"
    absd = np.abs(g - r)
    rel = absd / np.maximum(np.abs(r), 1e-6)
    close = (absd <= atol) | (rel <= rtol)
    return bool(close.all()), (f"max_abs={absd.max():.4g} "
                               f"max_rel={rel.max():.4g} "
                               f"mismatch={100.0*(~close).mean():.4f}%")


def _fit_affine(xs, ys):
    """Least-squares fit y = a + b*x. Returns (intercept, slope)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx)**2 for x in xs)
    if den == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-seqs", type=int, default=64)
    p.add_argument("--kv-len", type=int, default=8192)
    p.add_argument("--page-size", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=9216,
                   help="sets pages_per_seq = ceil(len/page_size); this is "
                        "what fixes bkv_sz in the default heuristic")
    p.add_argument("--total-pages", type=int, default=14697,
                   help="cache pages; default mirrors the production cache")
    p.add_argument("--num-q-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--kv-dtype", default="fp8", choices=sorted(_KV_DTYPES))
    p.add_argument("--chain", type=int, default=100,
                   help="calls fused into one jit; >=100 keeps host dispatch "
                        "out of the measurement")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true",
                   help="walk bkv_sz/bkv_csz instead of timing the default")
    p.add_argument("--sweep-bkv", default="1024,2048,4096,9216",
                   help="bkv_sz values for --sweep (bkv_csz follows bkv_sz "
                        "unless --sweep-csz is given)")
    p.add_argument("--sweep-csz", default="",
                   help="if set, bkv_csz values crossed with --sweep-bkv")
    p.add_argument("--kv-len-sweep", default="",
                   help="comma-separated kv_lens; fits the affine intercept")
    p.add_argument("--trace-distribution", action="store_true",
                   help="pass `distribution` as a jit argument instead of "
                        "closing over it. Production does not know the batch "
                        "composition at compile time, so this is the "
                        "representative mode: every pass survives into the "
                        "HLO, including ones whose sequence range is empty. "
                        "Closed over (the default), XLA folds those away and "
                        "a decode-only run compiles to RPAd alone.")
    p.add_argument("--assert-kernel", action="store_true",
                   help="exit non-zero if the compiled RPAd scope is not the "
                        "block size that was requested")
    p.add_argument("--skip-check", action="store_true",
                   help="skip the correctness comparison (timing only)")
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    args = p.parse_args()

    kv_dtype = _KV_DTYPES[args.kv_dtype]
    q_dtype = jnp.bfloat16
    page_size = args.page_size
    pages_per_seq = -(-args.max_model_len // page_size)
    sm_scale = 1.0 / (args.head_dim**0.5)

    # Off-TPU this raises NotImplementedError(tpu_version=-1). That is not
    # fatal for an explicit --sweep, and being able to smoke-test the harness
    # on a CPU host is worth more than insisting on the label, so it degrades
    # instead of exiting: only the paths that need the default will complain.
    try:
        default_bs = get_default_block_sizes(
            q_dtype, kv_dtype, args.num_q_heads, args.num_kv_heads,
            args.head_dim, page_size, args.num_seqs, args.num_seqs,
            pages_per_seq, case=RpaCase.DECODE)
    except NotImplementedError as exc:
        print(f"note: default block sizes unavailable here ({exc}); "
              f"--sweep still works with explicit sizes")
        default_bs = None

    print(f"device          : {jax.devices()[0].device_kind} "
          f"x{jax.device_count()}")
    print(f"shape           : num_seqs={args.num_seqs} q_len=1 "
          f"kv_len={args.kv_len} page={page_size} "
          f"pages_per_seq={pages_per_seq}")
    print(f"heads           : q={args.num_q_heads} kv={args.num_kv_heads} "
          f"head_dim={args.head_dim} kv_dtype={args.kv_dtype}")
    print(f"default d_block : {default_bs}")
    print(f"chain/iters     : {args.chain}/{args.iters}")

    cache_shape = get_kv_cache_shape(args.total_pages, page_size,
                                     args.num_kv_heads, args.head_dim,
                                     kv_dtype)
    # Materialized bytes: the cache stores align_to(nkv*2, packing) planes, so
    # at nkv=1 with fp8 packing 4 it holds 4 planes where only 2 are real KV.
    planes = cache_shape[2] * cache_shape[3]
    real_planes = args.num_kv_heads * 2
    bytes_per_token = planes * cache_shape[4] * jnp.dtype(kv_dtype).itemsize
    live_pages = -(-args.kv_len // page_size)
    mat = args.num_seqs * live_pages * page_size * bytes_per_token
    useful = mat * real_planes / planes
    print(f"kv cache        : {cache_shape} "
          f"({planes} planes stored, {real_planes} real)")
    print(f"per decode step : {mat/2**20:.1f} MiB materialized, "
          f"{useful/2**20:.1f} MiB useful "
          f"({100.0*real_planes/planes:.0f}% of bytes are real KV)\n")

    def build(kv_len, d_block_sizes):
        wl = _make_workload(args.num_seqs, kv_len, page_size, pages_per_seq,
                            args.total_pages, args.num_q_heads,
                            args.num_kv_heads, args.head_dim, q_dtype,
                            kv_dtype, args.seed)
        run = _build_chain(wl, d_block_sizes, sm_scale, args.chain,
                           args.trace_distribution)
        return wl, run

    failures = []

    def measure(label, kv_len, d_block_sizes, want_scope=None):
        """Time one variant. Records a failure rather than raising, so that a
        single bad block size costs its own row and not the rest of the sweep.
        """
        try:
            return _measure(label, kv_len, d_block_sizes, want_scope)
        except Exception as exc:  # pragma: no cover - diagnostic path
            traceback.print_exc()
            print(f"[{label}] ERROR: {type(exc).__name__}: {exc}\n")
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
            return None

    def _measure(label, kv_len, d_block_sizes, want_scope=None):
        wl, run = build(kv_len, d_block_sizes)
        scopes, err = _compiled_scopes(run, wl["kv_cache"])
        if err:
            # A variant that would not compile is a failure, not a skip. The
            # timing, the correctness check and --assert-kernel all live below
            # this point, so returning quietly here would let a run in which
            # nothing was measured exit 0 -- which is precisely the outcome
            # --assert-kernel exists to make impossible.
            print(f"[{label}] FAILED TO COMPILE: {err}")
            failures.append(f"{label}: did not compile: {err}")
            return None
        samples = _time(run, wl["kv_cache"], args.iters, args.warmup,
                        args.chain)
        med = statistics.median(samples)

        # Print the timing before running the correctness check. The check is
        # the more fragile of the two -- it traces a second, different program
        # -- and a throw there would otherwise discard a measurement that has
        # already been taken, losing the whole arm instead of one column.
        step_bytes = (args.num_seqs * (-(-kv_len // page_size)) * page_size *
                      bytes_per_token)
        gib_s = step_bytes / (med * 1e-6) / 2**30
        decode = [s for s in scopes if s.startswith("RPAd")]
        print(f"[{label}] median {med:9.2f} us   min {min(samples):9.2f}   "
              f"p90 {np.percentile(samples, 90):9.2f}   "
              f"spread {100.0*(max(samples)-min(samples))/med:5.1f}%")
        print(f"[{label}] {gib_s:7.1f} GiB/s materialized")
        print(f"[{label}] compiled scopes: {decode}")
        # Print every pass, not just RPAd. `ragged_paged_attention` launches
        # DECODE and MIXED unconditionally, so on a decode-only workload the
        # MIXED one is an empty launch -- and filtering it out of this line is
        # what made it invisible here for four jobs.
        print(f"[{label}] all passes    : {scopes}")

        ok, detail = (True, "skipped (--skip-check)")
        if not args.skip_check:
            try:
                ok, detail = _check(wl, d_block_sizes, sm_scale, args.rtol,
                                    args.atol)
            except Exception as exc:  # pragma: no cover - diagnostic path
                traceback.print_exc()
                ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        print(f"[{label}] correctness {'PASS' if ok else 'FAIL'}  {detail}")
        print()

        if not ok:
            failures.append(f"{label}: correctness {detail}")
        if args.assert_kernel and want_scope:
            if not any(want_scope in s for s in decode):
                failures.append(
                    f"{label}: wanted scope containing {want_scope!r}, "
                    f"compiled {decode}")
        return med

    if args.kv_len_sweep:
        kv_lens = [int(x) for x in args.kv_len_sweep.split(",") if x.strip()]
        print("=== kv_len sweep (default block sizes) ===\n")
        pts = []
        for kv_len in kv_lens:
            med = measure(f"kv={kv_len}", kv_len, None)
            if med is not None:
                pts.append((kv_len, med))
        if len(pts) >= 2:
            a, b = _fit_affine([x for x, _ in pts], [y for _, y in pts])
            at_prod = a + b * args.kv_len
            print("--- affine fit  t(kv_len) = intercept + slope*kv_len ---")
            print(f"intercept : {a:9.2f} us")
            print(f"slope     : {b:.6f} us/token")
            if at_prod > 0:
                print(f"at kv_len={args.kv_len}: fitted {at_prod:.2f} us, "
                      f"intercept is {100.0*a/at_prod:.1f}% of it")
        return 1 if failures else 0

    if args.sweep:
        bkvs = [int(x) for x in args.sweep_bkv.split(",") if x.strip()]
        cszs = [int(x) for x in args.sweep_csz.split(",") if x.strip()]
        print("=== bkv_sz / bkv_csz sweep ===\n")
        results = {}
        for bkv in bkvs:
            for csz in (cszs or [bkv]):
                if csz > bkv:
                    continue
                bs = (1, bkv, 1, csz)
                label = f"bkv={bkv},csz={csz}"
                med = measure(label, args.kv_len, bs,
                              want_scope=f"bkv_{bkv}_{csz}")
                if med is not None:
                    results[label] = med
        if results:
            base_label = (f"bkv={default_bs['bkv_sz']},"
                          f"csz={default_bs['bkv_csz']}") if default_bs else ""
            base = results.get(base_label, min(results.values()))
            print("--- relative (lower is better) ---")
            for label, med in sorted(results.items(), key=lambda kv: kv[1]):
                tag = "  (default)" if label == base_label else ""
                print(f"{label:>22}: {med:9.2f} us  "
                      f"{base/med:5.3f}x vs default  "
                      f"({100.0*(med-base)/base:+.1f}% time){tag}")
        return 1 if failures else 0

    if default_bs is None:
        print("error: the default arm needs a TPU to resolve block sizes; "
              "use --sweep with explicit --sweep-bkv off-TPU")
        return 1
    measure("default", args.kv_len, None,
            want_scope=f"bkv_{default_bs['bkv_sz']}_{default_bs['bkv_csz']}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
