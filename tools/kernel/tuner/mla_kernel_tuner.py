# SPDX-License-Identifier: Apache-2.0
"""Tune the MLA v2 kernel's decode block sizes for one model geometry.

A local port of tpu-inference's
``tools/kernel/tuner/v1/mla_kernel_tuner.py``. Input generation and the search
space are carried over unchanged so results stay comparable with upstream's;
what is not carried over is the harness around them -- Buildkite pipeline
generation, the Spanner backend, the multi-process executor and the optuna
Bayesian optimizer, roughly 3,000 lines that a single-chip local sweep does not
need (and optuna is not installed here). ``KernelTunerBase`` asks a subclass for
only three things -- ``generate_cases``, ``generate_inputs``, ``run`` -- and
those are what this file keeps.

Three deliberate differences from upstream, each of which changes what you get:

1. **The key is built from the same fields the call site derives from live
   operand shapes** (see ``mla_attention`` in
   ``vllm_torchtpu/layers/common/attention_interface.py``). Upstream assembles
   a ``TuningKey`` from flags independently of the call site, so an entry can
   be generated that no invocation ever asks for -- which is how all 61
   shipped entries came to be unreachable. Every flag below names one
   ``TuningKey`` field, so a generated entry is reachable as long as the flags
   are read off a running server.

2. **``max_num_tokens`` comes from a flag** rather than upstream's hardcoded
   ``[4, 8, ..., 2048]``. That list has no 2, and Kimi-K3 compiles buckets
   2/16/512, so one of its three shapes could not be tuned at all.

3. **Every configuration is checked against a reference**, not just timed. A
   tiling that computes a different answer is not a speedup. Changing the KV
   block count reassociates the online-softmax accumulation and moves the
   result by ~1 bf16 ulp, which is expected; anything larger is a bug and shows
   up in the ``maxdiff`` column.

Note what upstream's input generation does **not** vary: ``total_q_len`` is
``max_num_seqs``, not ``max_num_tokens``, so every bucket benchmarks the same
problem and picks the same winner. The buckets still each need a table entry --
the runtime key carries ``max_num_tokens`` -- so this measures once and emits an
identical entry per bucket rather than paying for three identical sweeps.
Reproducing each bucket's real padded query buffer would be a fidelity
improvement and is noted in the README as future work; it needs a TPU to
validate the padding convention against a live step.

This tunes ``case="batched_decode"`` only, exactly as upstream does: the search
space pins ``num_queries_per_block`` to 1 and every generated sequence has
``q_len=1``. The prefill/mixed tiling -- which on K3 is the larger prize -- needs
a mixed-case generator and is not addressed here.

  python tools/kernel/tuner/mla_kernel_tuner.py \
      --mla_actual_num_q_heads 3 --mla_actual_lkv_dim 512 --mla_actual_r_dim 64 \
      --mla_kv_dtype bfloat16 --mla_q_dtype bfloat16 \
      --mla_page_size_per_kv_packing 8 --mla_kv_packing 2 \
      --mla_max_num_seqs 8 --mla_pages_per_seq 2048 \
      --mla_total_num_pages 30311 \
      --max_num_tokens 2,16,512 --out results/mla-tune-k3.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention
from vllm_torchtpu.kernels.mla.v2.tuned_params import TunableParams, TuningKey
from vllm_torchtpu.utils import align_to, get_dtype_packing


def cdiv(a, b):
    return (a + b - 1) // b


def _tuning_key(args, max_num_tokens, case="batched_decode"):
    """The TuningKey the serving path builds for this geometry.

    Fields mirror the live-shape derivation in attention_interface.py:
    page_size_per_kv_packing and kv_packing from the cache shape, max_num_seqs
    from the padded metadata, pages_per_seq from the block table.
    """
    return TuningKey(
        case=case,
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=args.mla_actual_num_q_heads,
        actual_lkv_dim=args.mla_actual_lkv_dim,
        actual_r_dim=args.mla_actual_r_dim,
        kv_dtype=args.mla_kv_dtype,
        q_dtype=args.mla_q_dtype,
        page_size_per_kv_packing=args.mla_page_size_per_kv_packing,
        kv_packing=args.mla_kv_packing,
        max_num_seqs=args.mla_max_num_seqs,
        pages_per_seq=args.mla_pages_per_seq,
    )


# --------------------------------------------------------------------------
# Input generation, carried over from upstream unchanged apart from imports.
# --------------------------------------------------------------------------
def _generate_mla_inputs(seq_lens,
                         num_heads,
                         lkv_dim,
                         r_dim,
                         page_size,
                         q_dtype,
                         kv_dtype,
                         num_pages,
                         rng=None):
    """Build one set of kernel inputs. `seq_lens` is a list of (q_len, kv_len)."""
    if rng is None:
        rng = np.random.default_rng(1234)

    def gen_random(shape, dtype):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(dtype)

    padded_r_dim = align_to(r_dim, 128)
    padded_lkv_dim = align_to(lkv_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    packing = get_dtype_packing(kv_dtype)
    q_lens = [s[0] for s in seq_lens]
    kv_lens_list = [s[1] for s in seq_lens]
    total_q_len = sum(q_lens)
    cu_q_lens_list = [0]
    for q_len in q_lens:
        cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

    max_kv_len = max(kv_lens_list) if kv_lens_list else 0
    pages_per_seq = cdiv(max_kv_len, page_size)

    page_indices_list = []
    page_count = 0
    for kv_len in kv_lens_list:
        num_seq_pages = cdiv(kv_len, page_size)
        indices = list(range(page_count, page_count + num_seq_pages))
        page_indices_list.extend(indices + [-1] *
                                 (pages_per_seq - num_seq_pages))
        page_count += num_seq_pages

    total_num_pages = max(num_pages, page_count)

    ql_nope = gen_random((num_heads, total_q_len, lkv_dim), q_dtype)
    q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
    new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
    new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)

    assert page_size % packing == 0, (
        f"page_size ({page_size}) must be a multiple of packing ({packing})")

    cache_kv = gen_random(
        (total_num_pages, page_size // packing, packing, padded_kv_dim),
        kv_dtype)
    # Sequences with q_len == 1 lead the batch and form the decode segment.
    num_decode_seqs = 0
    for s in seq_lens:
        if s[0] == 1:
            num_decode_seqs += 1
        else:
            break
    return {
        'ql_nope':
        ql_nope,
        'q_pe':
        q_pe,
        'new_kv_c':
        new_kv_c,
        'new_k_pe':
        new_k_pe,
        'cache_kv':
        cache_kv,
        'kv_lens':
        jnp.array(kv_lens_list, dtype=jnp.int32),
        'page_indices':
        jnp.array(page_indices_list, dtype=jnp.int32),
        'cu_q_lens':
        jnp.array(cu_q_lens_list, dtype=jnp.int32),
        'distribution':
        jnp.array([num_decode_seqs, num_decode_seqs,
                   len(seq_lens)],
                  dtype=jnp.int32),
    }


def build_mixed_inputs(key, total_num_pages, q_len, cached_len, decode_seqs=0):
    """One scheduler step: `decode_seqs` decodes plus a prefill chunk.

    ``q_len`` is the step's whole token budget, not the prefill sequence's
    share of it. The scheduler spends that budget on the decodes first -- one
    token each -- and gives the remainder to the prefill chunk, so total query
    length stays equal to the compile bucket ``max_num_tokens`` names no matter
    how many decodes ride along. ``decode_seqs=0`` is the idle-server case: a
    single request prefilling a full chunk with nothing else in flight.

    ``mla_attention_op`` stacks the metadata as ``[rd0, rd0, rd2]`` before
    handing it to the kernel, collapsing the prefill segment to empty -- so on
    MLA *every* non-decode sequence runs in the mixed stage, and the leading
    decodes share the call with it. That is the shape reproduced here:
    ``_generate_mla_inputs`` counts leading ``q_len == 1`` sequences, so
    ``decode_seqs`` of them followed by one longer sequence yields
    ``distribution=[k, k, k+1]``.
    """
    if decode_seqs >= q_len:
        raise ValueError(
            f"{decode_seqs=} leaves no tokens for the prefill chunk in a "
            f"{q_len}-token budget")
    seq_lens = [[1, cached_len + 1] for _ in range(decode_seqs)]
    prefill_q = q_len - decode_seqs
    seq_lens.append([prefill_q, cached_len + prefill_q])
    return _generate_mla_inputs(
        seq_lens=seq_lens,
        num_heads=key.actual_num_q_heads,
        lkv_dim=key.actual_lkv_dim,
        r_dim=key.actual_r_dim,
        page_size=key.page_size_per_kv_packing * key.kv_packing,
        q_dtype=jnp.dtype(key.q_dtype),
        kv_dtype=jnp.dtype(key.kv_dtype),
        num_pages=total_num_pages,
        rng=np.random.default_rng(0),
    )


def get_search_space(tuning_key, max_kv_pages_per_block):
    """Upstream's decode search space, with an explicit cap on the KV axis.

    Upstream derives the KV-page ceiling from ``pages_per_seq``, which is fine
    at its default of 9 but explodes for a real deployment: Kimi-K3 at
    max_model_len 32768 and page size 16 has pages_per_seq 2048, so the
    unbounded space is 12 values on that axis alone, most of them certain to
    exhaust VMEM. The cap keeps the sweep finite; what it removed is logged.
    """
    decode_batch_sizes = []
    dbs = 1
    while dbs <= tuning_key.max_num_seqs:
        decode_batch_sizes.append(dbs)
        dbs *= 2

    kv_pages = sorted(
        set([2**j for j in range(tuning_key.pages_per_seq.bit_length())] +
            ([3] if tuning_key.pages_per_seq >= 3 else [])))
    dropped = [v for v in kv_pages if v > max_kv_pages_per_block]
    kv_pages = [v for v in kv_pages if v <= max_kv_pages_per_block]
    return {
        'decode_batch_size': decode_batch_sizes,
        'num_kv_pages_per_block': kv_pages,
        'num_queries_per_block': [1],
        'vmem_limit_bytes': [60 * 1024 * 1024, 64 * 1024 * 1024],
    }, dropped


def build_inputs(key, total_num_pages, bench_kv_len=0):
    """Decode-shaped inputs for a key: every sequence contributes one query.

    ``bench_kv_len`` decouples the benchmarked context from the paging the key
    describes, and it matters more than it looks.

    ``pages_per_seq`` is a key field derived from ``max_model_len``, so the
    default context here is the *longest* the deployment could ever serve --
    32,768 tokens for Kimi-K3. Tuning there picked a 2,048-token KV tile, which
    is larger than the entire context of a 1k-2k serving workload: the kernel
    would run one tile with most of it masked padding. Lowering
    ``pages_per_seq`` to compensate is not an option, because that changes the
    key and the entry stops being found.

    So the key keeps the real paging and this overrides only what is measured.
    The underlying limitation stands and is worth knowing: the TuningKey schema
    describes the paging, not the workload, so two deployments with identical
    paging and very different context lengths share one entry.
    """
    kv_len = bench_kv_len or (key.pages_per_seq *
                              key.page_size_per_kv_packing * key.kv_packing)
    return _generate_mla_inputs(
        seq_lens=[[1, kv_len] for _ in range(key.max_num_seqs)],
        num_heads=key.actual_num_q_heads,
        lkv_dim=key.actual_lkv_dim,
        r_dim=key.actual_r_dim,
        page_size=key.page_size_per_kv_packing * key.kv_packing,
        q_dtype=jnp.dtype(key.q_dtype),
        kv_dtype=jnp.dtype(key.kv_dtype),
        num_pages=total_num_pages,
        rng=np.random.default_rng(0),
    )


def make_call(inputs, key, params):
    """A jitted closure over one (key, params) pair.

    Two departures from upstream, both about measuring the kernel rather than
    the harness:

    * **jit.** Upstream calls the kernel directly in the timing loop, so each
      iteration pays Python tracing and dispatch. That is a fixed cost, but the
      configurations being compared here differ by tens of microseconds.
    * **The cache input is not rebound between iterations.** The kernel returns
      an updated cache and upstream feeds it back in, so successive iterations
      operate on a cache whose contents have moved. Holding the input fixed
      means every iteration times the same problem.
    """

    @jax.jit
    def run(cache_kv):
        return mla_ragged_paged_attention(
            ql_nope=inputs['ql_nope'],
            q_pe=inputs['q_pe'],
            new_kv_c=inputs['new_kv_c'],
            new_k_pe=inputs['new_k_pe'],
            cache_kv=cache_kv,
            kv_lens=inputs['kv_lens'],
            page_indices=inputs['page_indices'],
            cu_q_lens=inputs['cu_q_lens'],
            distribution=inputs['distribution'],
            sliding_window=key.sliding_window,
            soft_cap=key.soft_cap,
            chunk_prefill_size=key.chunk_prefill_size,
            s_dtype=jnp.dtype(key.s_dtype),
            p_same_dtype_as_v=key.p_same_dtype_as_v,
            decode_batch_size=params.decode_batch_size,
            num_kv_pages_per_block=params.num_kv_pages_per_block,
            num_queries_per_block=params.num_queries_per_block,
            vmem_limit_bytes=params.vmem_limit_bytes,
            mixed_q_split=params.q_split,
        )

    return run


def time_one(inputs, key, params, warmup, iters):
    """(status, ms per call, output) -- output is returned for the diff check."""
    try:
        run = make_call(inputs, key, params)
        cache = inputs['cache_kv']
        out, _ = jax.block_until_ready(run(cache))
        for _ in range(warmup - 1):
            jax.block_until_ready(run(cache))
        t0 = time.perf_counter()
        for _ in range(iters):
            res = run(cache)
        jax.block_until_ready(res)
        return "ok", (time.perf_counter() - t0) / iters * 1e3, out
    except Exception as exc:  # noqa: BLE001 - the sweep must survive a bad point
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg:
            return "oom", float("inf"), None
        return f"error: {msg.splitlines()[0][:70]}", float("inf"), None


def run_mixed(args):
    """Sweep the prefill/mixed tiling across cached-prefix lengths.

    Two axes here that the decode case does not have: num_queries_per_block is
    a real knob (decode pins it to 1), and the cached prefix length varies
    independently of the chunk size. decode_batch_size is unused in this stage.

    The shipped default for this stage is (num_queries_per_block=16,
    num_kv_pages_per_block=1), hardcoded at the call site rather than looked
    up, so it is included in the space and reported as the baseline.
    """
    q_lens = [int(x) for x in str(args.mixed_q_len).split(",")]
    decode_seq_counts = [int(x) for x in args.mixed_decode_seqs.split(",")]
    cached_lens = [int(x) for x in args.mixed_cached_lens.split(",")]
    bqs = [int(x) for x in args.mixed_queries_per_block.split(",")]
    kv_pages = [int(x) for x in args.mixed_kv_pages.split(",")]
    q_splits = [int(x) for x in args.mixed_q_splits.split(",")]
    page_size = args.mla_page_size_per_kv_packing * args.mla_kv_packing
    results = []

    print(f"device: {jax.devices()[0].device_kind}")
    print(f"case=mixed  q_len={q_lens}  page_size={page_size}  "
          f"cached={cached_lens}  decode_seqs={decode_seq_counts}")

    for q_len, decode_seqs, cached in itertools.product(
            q_lens, decode_seq_counts, cached_lens):
        total_kv = cached + q_len
        key = _tuning_key(args, max_num_tokens=q_len, case="mixed")
        print(f"\n=== cached={cached} + q={q_len} -> {total_kv} KV tokens "
              f"({cdiv(total_kv, page_size)} pages), "
              f"{decode_seqs} decode + 1 prefill seq")
        if args.dry_run:
            print(f"  key: {key}")
            print(f"  num_queries_per_block: {bqs}")
            print(f"  num_kv_pages_per_block: {kv_pages}")
            print(f"  q_split: {q_splits}")
            print(f"  decode_seqs: {decode_seqs}")
            continue
        inputs = build_mixed_inputs(key, args.mla_total_num_pages, q_len,
                                    cached, decode_seqs)
        print(f"  {'bq':>5} {'kv_pg':>6} {'kv_tok':>7} {'qspl':>5} "
              f"{'ms':>9} {'vs default':>11} {'maxdiff':>9}  status")
        ref = default = best = None
        for bq, kvp, qspl in itertools.product(bqs, kv_pages, q_splits):
            # The kernel splits num_queries_per_block[2] q_split ways and
            # rejects a remainder, so an indivisible pair is not a candidate.
            if bq % qspl:
                continue
            params = TunableParams(num_kv_pages_per_block=kvp,
                                   num_queries_per_block=bq,
                                   vmem_limit_bytes=62914560,
                                   q_split=qspl)
            status, ms, out = time_one(inputs, key, params, args.warmup,
                                       args.iters)
            diff = ""
            if out is not None:
                if ref is None:
                    ref = out
                else:
                    diff = "%.4f" % float(
                        jnp.max(
                            jnp.abs(
                                out.astype(jnp.float32) -
                                ref.astype(jnp.float32))))
                if (bq, kvp, qspl) == (16, 1, 1):
                    default = ms
                if best is None or ms < best[0]:
                    best = (ms, bq, kvp, qspl)
            results.append({
                "cached": cached,
                "q_len": q_len,
                "decode_seqs": decode_seqs,
                "num_queries_per_block": bq,
                "num_kv_pages_per_block": kvp,
                "q_split": qspl,
                "ms": None if ms == float("inf") else ms,
                "maxdiff": diff or None,
                "status": status
            })
            rel = (f"{default / ms:10.2f}x"
                   if default and ms != float("inf") else "         -")
            print(f"  {bq:>5} {kvp:>6} {kvp * page_size:>7} {qspl:>5} "
                  f"{('%9.4f' % ms) if ms != float('inf') else '        -'} "
                  f"{rel} {diff:>9}  {status}")
        if best and default:
            print(f"\n  default (bq=16, kv_pages=1, q_split=1): "
                  f"{default:.4f} ms")
            print(f"  best    (bq={best[1]}, kv_pages={best[2]}, "
                  f"q_split={best[3]}): "
                  f"{best[0]:.4f} ms  -> {default / best[0]:.2f}x")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"geometry": vars(args), "rows": results}, fh, indent=1)
        print(f"\n-> {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Flag names mirror upstream's so a geometry recorded for one tool reads
    # the same in the other.
    ap.add_argument("--mla_actual_num_q_heads", type=int, required=True)
    ap.add_argument("--mla_actual_lkv_dim", type=int, required=True)
    ap.add_argument("--mla_actual_r_dim", type=int, required=True)
    ap.add_argument("--mla_kv_dtype", default="bfloat16")
    ap.add_argument("--mla_q_dtype", default="bfloat16")
    ap.add_argument("--mla_page_size_per_kv_packing", type=int, required=True)
    ap.add_argument("--mla_kv_packing", type=int, required=True)
    ap.add_argument("--mla_max_num_seqs", type=int, required=True)
    ap.add_argument("--mla_pages_per_seq", type=int, required=True)
    ap.add_argument("--mla_total_num_pages", type=int, required=True)
    ap.add_argument("--max_num_tokens",
                    default="2,16,512",
                    help="comma-separated; use the model's compile buckets")
    ap.add_argument("--max_kv_pages_per_block", type=int, default=128)
    ap.add_argument("--bench_kv_len",
                    type=int,
                    default=0,
                    help="context length to benchmark, in tokens. 0 uses "
                    "pages_per_seq * page_size, i.e. max_model_len -- the "
                    "longest context the deployment could serve, which is "
                    "usually not the one it does serve. Set this to the "
                    "real workload's context; the key is unaffected.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", help="write the full sweep here as JSON")
    ap.add_argument("--case",
                    default="batched_decode",
                    choices=["batched_decode", "mixed"],
                    help="mixed tunes the prefill/chunked-prefill stage: "
                    "num_queries_per_block matters and every sequence has "
                    "q_len > 1. batched_decode is upstream's only case.")
    # A list, so one invocation can cover several compile buckets. Each value
    # is a separate max_num_tokens and therefore a separate table entry.
    ap.add_argument("--mixed_q_len",
                    default="4096",
                    help="comma-separated token budgets per step. Each value "
                    "is the step's whole query length, so it is also the "
                    "max_num_tokens the entry is keyed on, and it is only "
                    "reachable if the server compiles that bucket -- i.e. it "
                    "must be <= --max-num-batched-tokens and appear in "
                    "--compilation-config compile_sizes.")
    ap.add_argument("--mixed_cached_lens",
                    default="4096,8192,16384,28672",
                    help="comma-separated cached-prefix lengths to sweep")
    ap.add_argument("--mixed_queries_per_block",
                    default="16,32,64,128,256,512")
    ap.add_argument(
        "--mixed_decode_seqs",
        default="0",
        help="concurrency axis: how many decode sequences share "
        "the step with the prefill chunk. The step's token budget "
        "is fixed at --mixed_q_len, so each decode takes one token "
        "from the chunk. 0 is the idle-server case. Note the "
        "TuningKey cannot tell these apart -- max_num_seqs is the "
        "padded batch dim, constant at the server's "
        "--max-num-seqs -- so one entry has to serve every value "
        "here, and this axis says which one it is tuned for.")
    ap.add_argument("--mixed_q_splits",
                    default="1",
                    help="q_split candidates for the mixed stage. The kernel "
                    "runs num_queries_per_block[2] // q_split queries per "
                    "sub-block and requires the division to be exact, so a "
                    "candidate that does not divide a given bq is skipped "
                    "rather than treated as a failure.")
    ap.add_argument("--mixed_kv_pages",
                    default="1,2,4,8,16,32",
                    help="num_kv_pages_per_block candidates. The useful cap "
                    "scales down as page_size goes up: this axis is in PAGES, "
                    "so at page 1024 the default's top entry is a "
                    "32768-token tile that only OOMs, and an OOM still costs "
                    "most of a compile to discover.")
    ap.add_argument("--dry_run",
                    action="store_true",
                    help="print the keys and the search space and stop. A TPU "
                    "allocation is expensive; a geometry typo is not "
                    "worth one.")
    args = ap.parse_args()

    if not args.dry_run:
        print(f"device: {jax.devices()[0].device_kind}  "
              f"{len(jax.devices())} visible")

    if args.case == "mixed":
        return run_mixed(args)

    buckets = [int(x) for x in args.max_num_tokens.split(",")]
    results = []
    winner = None
    for i, max_num_tokens in enumerate(buckets):
        # Built exactly the way mla_attention builds it, so a winning entry is
        # reachable from a live server rather than merely plausible.
        key = _tuning_key(args, max_num_tokens=max_num_tokens)
        space, dropped = get_search_space(key, args.max_kv_pages_per_block)
        combos = list(itertools.product(*space.values()))
        if i > 0 and not args.dry_run:
            # The inputs are identical to the first bucket's, so re-running the
            # sweep would re-measure the same problem. Say so rather than
            # quietly emitting an entry that was never benchmarked.
            print(f"\n=== max_num_tokens={max_num_tokens}: reusing the "
                  f"measurement from bucket {buckets[0]} (identical inputs)")
            continue
        print(f"\n=== max_num_tokens={max_num_tokens}: {len(combos)} configs" +
              (f"  (num_kv_pages_per_block {dropped} dropped by "
               f"--max_kv_pages_per_block={args.max_kv_pages_per_block})"
               if dropped else ""))
        if args.dry_run:
            kv_tokens = args.bench_kv_len or (key.pages_per_seq *
                                              key.page_size_per_kv_packing *
                                              key.kv_packing)
            print(f"  key: {key}")
            print(f"  inputs would be {key.max_num_seqs} seqs x {kv_tokens} "
                  f"kv tokens, page_size="
                  f"{key.page_size_per_kv_packing * key.kv_packing}")
            for name, values in space.items():
                print(f"    {name}: {values}")
            continue
        inputs = build_inputs(key, args.mla_total_num_pages, args.bench_kv_len)
        page_size = key.page_size_per_kv_packing * key.kv_packing
        kv_len = args.bench_kv_len or key.pages_per_seq * page_size
        print(f"  inputs: {key.max_num_seqs} seqs x {kv_len} kv tokens "
              f"({cdiv(kv_len, page_size)} pages), page_size={page_size}" +
              ("" if not args.bench_kv_len else
               f"  [key says {key.pages_per_seq} pages = "
               f"{key.pages_per_seq * page_size} tokens]"))
        print(f"  {'dbs':>4} {'kv_pg':>6} {'vmem_MiB':>9} {'ms':>9} "
              f"{'vs base':>9} {'maxdiff':>9}  status")

        ref = None
        base = None
        best = None
        for combo in combos:
            params = TunableParams(**dict(zip(space.keys(), combo)))
            status, ms, out = time_one(inputs, key, params, args.warmup,
                                       args.iters)
            diff = ""
            if out is not None:
                if ref is None:
                    ref = out
                else:
                    d = float(
                        jnp.max(
                            jnp.abs(
                                out.astype(jnp.float32) -
                                ref.astype(jnp.float32))))
                    diff = f"{d:.4f}"
                if base is None:
                    base = ms
                if best is None or ms < best[0]:
                    best = (ms, params)
            row = {
                "max_num_tokens": max_num_tokens,
                "decode_batch_size": params.decode_batch_size,
                "num_kv_pages_per_block": params.num_kv_pages_per_block,
                "num_queries_per_block": params.num_queries_per_block,
                "vmem_limit_bytes": params.vmem_limit_bytes,
                "ms": None if ms == float("inf") else ms,
                "maxdiff": diff or None,
                "status": status,
            }
            results.append(row)
            print(
                f"  {params.decode_batch_size:>4} "
                f"{params.num_kv_pages_per_block:>6} "
                f"{params.vmem_limit_bytes // (1024 * 1024):>9} "
                f"{('%9.4f' % ms) if ms != float('inf') else '        -'} "
                f"{('%8.2fx' % (base / ms)) if base and ms != float('inf') else '        -'} "
                f"{diff:>9}  {status}")
        if best:
            ms, params = best
            winner = (key, params, ms)
            print(f"\n  best: decode_batch_size={params.decode_batch_size} "
                  f"num_kv_pages_per_block={params.num_kv_pages_per_block} "
                  f"vmem={params.vmem_limit_bytes} -> {ms:.4f} ms "
                  f"({base / ms:.2f}x over the first working config)")

    if winner:
        key, params, ms = winner
        print("\n" + "=" * 72)
        print(f"tuned_params.py entries ({ms:.4f} ms each, one per bucket):\n")
        for max_num_tokens in buckets:
            print(f"    TuningKey(\n"
                  f"        case=\"batched_decode\",\n"
                  f"        max_num_tokens={max_num_tokens},\n"
                  f"        actual_num_q_heads={key.actual_num_q_heads},\n"
                  f"        actual_lkv_dim={key.actual_lkv_dim},\n"
                  f"        actual_r_dim={key.actual_r_dim},\n"
                  f"        kv_dtype=\"{key.kv_dtype}\",\n"
                  f"        q_dtype=\"{key.q_dtype}\",\n"
                  f"        page_size_per_kv_packing="
                  f"{key.page_size_per_kv_packing},\n"
                  f"        kv_packing={key.kv_packing},\n"
                  f"        max_num_seqs={key.max_num_seqs},\n"
                  f"        pages_per_seq={key.pages_per_seq},\n"
                  f"    ):\n"
                  f"    TunableParams(\n"
                  f"        decode_batch_size={params.decode_batch_size},\n"
                  f"        num_kv_pages_per_block="
                  f"{params.num_kv_pages_per_block},\n"
                  f"        num_queries_per_block="
                  f"{params.num_queries_per_block},\n"
                  f"        vmem_limit_bytes={params.vmem_limit_bytes},\n"
                  f"    ),")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"geometry": vars(args), "rows": results}, fh, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
