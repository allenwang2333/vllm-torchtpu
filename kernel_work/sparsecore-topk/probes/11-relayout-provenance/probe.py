"""Probe 11 -- is the A0 residual's XLA relayout real in the production path?

probes/10 traced its 75.349 us `A0` residual to work that happens INSIDE the
timed `jit` and OUTSIDE the Pallas kernel: a `bitcast-convert` fusion plus a
layout-changing `copy` wrapped in a `sparse-core-data-format-call`, over the
whole 40 MiB score buffer. The residual is flat in `num_waves` (0.009 us/wave)
and linear in buffer bytes (1.8246 us/MiB input, R2 0.977; 1.1180 us/MiB
output, R2 0.99978).

That number is 39.8% of the modelled production prefill kernel, so whether it
is real decides the denominator of the whole exercise. This probe asks three
things and refuses to let any of them rest on prose.

Q1 -- SOURCE + COMPILE. Does the PRODUCTION entry emit the same relayout?
     The real caller is `streamindex_topk` (streamindex_topk.py:521, jitted at
     :509); it calls `sparsecore_topk` at :831 with an f32 score matrix that
     `run_scores_kernel` -- a `pl.pallas_call` -- produced INSIDE the same jit.
     We compile THAT function, at the GLM-5.2 shipping geometry, and read its
     optimized HLO. Nothing here is a wrapper of the probe's own making: the
     only thing the probe supplies is the argument avals. The compiled kernel
     names itself `sc_topk_direct_b<b>_n<n>_k<k>`, so the presence of
     `sc_topk_direct_b1024_n10240_k2048` in the production HLO is what proves
     the avals reproduced the shipping shape rather than something near it.

Q2 -- TIMING, CAUSAL. If the relayout is removed, does the residual fall?
     Eight arms in ONE interleaved pool, every one of them probes/09's `A0` or
     `B` patched source verbatim (md5-asserted against probes/09's
     547031a2... / fd46ef14...), differing only in the jit signature wrapped
     around it. Every arm's own optimized HLO is dumped and summarized BEFORE
     any timing, and the report prints `data-format-call present?` and
     `layout copy present?` beside every us_median. An arm whose time moved
     but whose HLO still contains the copy has not tested the hypothesis and
     is reported as a NULL, not a win.

Q3 -- THE OUTPUT-BYTE TERM. probes/10 found a clean 1.118 us/MiB on the output
     axis that the input relayout cannot explain. `A0_flatout` returns the
     kernel's raw flat `s32[b*k]` instead of `.reshape(b, k)`, which is the
     one thing that can delete an output-side relayout, and the HLO inventory
     prints what is on the output side either way.

ARMS (8 timed + 3 nulls, one pool, 9 rounds):

    B_base     B , f32[b,n] entry             CROSS-JOB control (p08/p09/p10)
    A0_base    A0, f32[b,n] entry             CROSS-JOB control (p09/p10)
    A0_s32in   A0, s32[b,n] entry             bitcast-convert has nothing to do
    A0_flatin  A0, s32[b*n] entry             input already in the call's layout
    A0_flatout A0, f32[b,n] entry, flat out   output relayout deleted
    A0_flatio  A0, flat in AND flat out       both deleted
    A0_fmtin   A0, XLA-chosen (AUTO) in-layout, committed OUTSIDE the timed
               region -- this MOVES work out of the measurement, it does not
               prove the work is unnecessary. Stated again in the report.
    A0_donate  A0, donate_argnums on the score buffer -- tests whether the
               copy is an aliasing/liveness artifact rather than a layout one.

Every arm is smoke-called three times before the pool opens; an arm that
raises is dropped loudly rather than taking the job down. Three nulls are
registered and their spread is the resolution floor for every delta quoted --
probes/10's was 0.770 us.

NOT a correctness experiment. Every `A0_*` arm computes the wrong answer by
construction. The patch-evidence gate is reproduced from probes/10 and carries
probes/09's caveat: it cannot distinguish "not written" from "written
identically" for an output-suppressing arm.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import traceback
import types

import numpy as np

SEED = 20260904  # same stream as probes 05, 07, 08, 09 and 10
REPS_LO = 20
REPS_HI = 60

_LAST_ORDER: list[list[str]] = []

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------
# Timing harness -- byte-identical protocol to probes 07-10. Prefer the FROZEN
# instrument so the numbers compose with probes 03 and 07-10 directly.
# --------------------------------------------------------------------------
def _time(fn, reps: int) -> float:
    import jax
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def _measure_inline(variants, rounds=9, shuffle_seed=20260902):
    import jax
    rng = random.Random(shuffle_seed)
    lo_best, hi_best, slopes, order_log = {}, {}, {}, []
    for name in variants:  # warm: an unwarmed thunk puts its compile in `lo`
        jax.block_until_ready(variants[name]())
    for _ in range(rounds):
        names = list(variants)
        rng.shuffle(names)
        order_log.append(names)
        lo_r = {}
        for name in names:
            lo_r[name] = _time(variants[name], REPS_LO)
            lo_best[name] = min(lo_best.get(name, math.inf), lo_r[name])
        for name in names:
            hi = _time(variants[name], REPS_HI)
            hi_best[name] = min(hi_best.get(name, math.inf), hi)
            slopes.setdefault(name, []).append(hi - lo_r[name])
    out = {}
    span = REPS_HI - REPS_LO
    for name in variants:
        s = sorted(slopes[name])
        n = len(s)
        med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        per_device = s[0] / span
        naive = hi_best[name] / REPS_HI
        host = max(naive - per_device, 0.0)
        out[name] = {
            "us": per_device * 1e6,
            "us_median": med / span * 1e6,
            "us_mean": (sum(s) / n) / span * 1e6,
            "us_unpaired": (hi_best[name] - lo_best[name]) / span * 1e6,
            "hostbound_pct": (host / naive * 100.0) if naive > 0 else 0.0,
            "suspect": bool(min(s) <= 0.0 or per_device <= 0.0),
        }
        if out[name]["suspect"]:
            print(f"  !! SUSPECT {name}: min slope <= 0 -- unwarmed recompile. "
                  f"`us` invalid; read us_median "
                  f"({out[name]['us_median']:.2f}).", file=sys.stderr)
    global _LAST_ORDER
    _LAST_ORDER = order_log
    return out


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _load_harness(extra_dirs):
    for d in extra_dirs:
        cand = os.path.join(d, "glm_dsa_block_bench.py")
        if not os.path.isfile(cand):
            continue
        try:
            if d not in sys.path:
                sys.path.insert(0, d)
            mod = importlib.import_module("glm_dsa_block_bench")
            if (mod.REPS_LO, mod.REPS_HI) != (REPS_LO, REPS_HI):
                print(f"  harness at {cand} has REPS "
                      f"{mod.REPS_LO}/{mod.REPS_HI} != {REPS_LO}/{REPS_HI} "
                      f"-- refusing it, using inline", flush=True)
                continue
            print(f"  timing harness: FROZEN INSTRUMENT {cand} "
                  f"(md5 {_md5(open(cand, 'rb').read())})", flush=True)
            return mod.measure, (lambda: mod.LAST_ORDER), "frozen"
        except Exception as exc:  # pragma: no cover - pod-side path variance
            print(f"  harness import from {d} failed: {exc!r}", flush=True)
    print("  timing harness: INLINE COPY (frozen instrument not importable)",
          flush=True)
    return _measure_inline, (lambda: _LAST_ORDER), "inline"


# --------------------------------------------------------------------------
# Arm construction by textual patch of the installed module.
#
# EVERY anchor and EVERY patch string below is copied verbatim from
# probes/10/probe.py, which copied them verbatim from probes/09's. Anchors are
# quoted with their exact leading whitespace and every `_sub` call asserts the
# occurrence count, so a source drift under this probe is a hard failure and
# never a silently-unpatched arm.
# --------------------------------------------------------------------------
SRC_LEVELS = "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))"
NO_LEVELS = "REFINEMENT_LEVELS = ()"

ANCHOR_WALK = "        walk_chunks = _cdiv_dyn(i_hi)"
INJECT_ZEROS = (
    "        walk_chunks = _cdiv_dyn(i_hi)\n"
    "        z_dyn = jnp.where(w < 0, walk_chunks, 0)"
)

SRC_CVT = "            plsc.parallel_loop(0, walk_chunks, unroll=4)(cvt)"
SRC_HIST = "            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)"
SRC_EMIT = "            jnp.where(active, walk_chunks, 0),"
SRC_ZERO_PRIV = "            zero_priv(NUM_BUCKETS)"
SRC_FOLD = "            fold_priv(NUM_BUCKETS // LANES)"
SRC_SCAN = "        total = scan_glob(NUM_BUCKETS // LANES)"
SRC_FIND = "        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)"
SRC_DMA = "            dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)"
DEL_DMA = "            pass  # probe08: score DMA removed"

SRC_FILL = "        fill_bound = jnp.where(out_gate & (total < k), k // LANES, 0)"
NOFILL = ("        fill_bound = jnp.where(out_gate & (total < k) & (w < 0), "
          "k // LANES, 0)")
SRC_OUTDMA = "        @pl.when(out_gate)"
NOOUTDMA = "        @pl.when(out_gate & (w < 0))"
SRC_WAVES = "    lax.fori_loop(0, num_waves, wave_body, 0)"


def _sub(src: str, old: str, new: str, what: str, expect: int = 1) -> str:
    n = src.count(old)
    assert n == expect, f"patch {what}: expected {expect} occurrence(s), found {n}"
    return src.replace(old, new)


DROPS = {
    "B":  (),
    "A0": ("dma", "cvt", "hist", "sweep", "emit", "fill", "outdma"),
}

WAVES: dict[str, int] = {}

Z_DYN_DROPS = frozenset(
    ("cvt", "hist", "emit", "sweep", "zero", "fold", "scan", "find"))

# probes/09's own patched-source md5s (job j-4e0df30a-d833-4c1c-825d),
# re-asserted by probes/10 (job j-770c6de9-2044-4216-bcdd). Asserted here, not
# printed-and-hoped: if `_patch_src` does not reproduce them bit for bit then
# "the same A0" is false and every cross-job comparison below is void.
P09_SRC_MD5 = {"B": "fd46ef14b38fd7781885e73d0ca33965",
               "A0": "547031a2fd48545550d864b008d817ff"}


def _patch_src(base_src: str, name: str) -> str:
    """The single source of truth for what each patch does. Verbatim probes/10."""
    src = base_src
    drops = DROPS[name]
    zero = "z_dyn"

    src = _sub(src, SRC_LEVELS, NO_LEVELS, f"{name}/levels")

    if any(d in Z_DYN_DROPS for d in drops):
        src = _sub(src, ANCHOR_WALK, INJECT_ZEROS, f"{name}/inject")

    if "dma" in drops:
        src = _sub(src, SRC_DMA, DEL_DMA, f"{name}/dma")
    if "cvt" in drops:
        src = _sub(src, SRC_CVT,
                   SRC_CVT.replace("walk_chunks", zero), f"{name}/cvt")
    if "hist" in drops:
        src = _sub(src, SRC_HIST,
                   SRC_HIST.replace("walk_chunks", zero), f"{name}/hist")
    if "emit" in drops:
        src = _sub(src, SRC_EMIT,
                   SRC_EMIT.replace("walk_chunks", zero), f"{name}/emit")
    if "sweep" in drops or "zero" in drops:
        src = _sub(src, SRC_ZERO_PRIV,
                   f"            zero_priv({zero})", f"{name}/zero_priv")
    if "sweep" in drops or "fold" in drops:
        src = _sub(src, SRC_FOLD,
                   f"            fold_priv({zero})", f"{name}/fold")
    if "sweep" in drops or "scan" in drops:
        src = _sub(src, SRC_SCAN,
                   f"        total = scan_glob({zero})", f"{name}/scan")
    if "sweep" in drops or "find" in drops:
        src = _sub(src, SRC_FIND,
                   f"        b_star = find_bucket({zero}, total - k)",
                   f"{name}/find")
    if "fill" in drops:
        src = _sub(src, SRC_FILL, NOFILL, f"{name}/fill")
    if "outdma" in drops:
        src = _sub(src, SRC_OUTDMA, NOOUTDMA, f"{name}/outdma")
    if name in WAVES:
        src = _sub(src, SRC_WAVES,
                   SRC_WAVES.replace("num_waves", str(WAVES[name])),
                   f"{name}/waves")
    return src


def build_arm(base_src: str, patch: str, label: str) -> types.ModuleType:
    """One module object per LABEL, even when two labels share a patch."""
    src = _patch_src(base_src, patch)
    mod = types.ModuleType(f"sctopk11_{abs(hash((patch, label)))}")
    mod.__file__ = f"<probe11:{label}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


# ---------------------------------------------------------------------------
# The arm table. Every arm runs at the ONE shipping shape b=1024 n=10240 k=2048.
# `patch` is which of the two patched sources runs; the jit SIGNATURE wrapped
# around it is what varies.
# ---------------------------------------------------------------------------
B, N, K = 1024, 10240, 2048

ARMS = [
    # label,       patch, what it isolates
    ("B_base",     "B",  "cross-job control (p08 327.166 / p09 326.782 / p10 326.732)"),
    ("A0_base",    "A0", "cross-job control (p09 92.231 / p10 92.118)"),
    ("A0_s32in",   "A0", "s32[b,n] entry: bitcast-convert has nothing to do"),
    ("A0_flatin",  "A0", "s32[b*n] entry: already in the custom call's layout"),
    ("A0_flatout", "A0", "flat s32[b*k] result: output relayout deleted"),
    ("A0_flatio",  "A0", "flat in AND flat out"),
    ("A0_fmtin",   "A0", "AUTO input layout, committed OUTSIDE the timed region"),
    ("A0_donate",  "A0", "donate_argnums on the score buffer (aliasing test)"),
]
LABELS = [a[0] for a in ARMS]
NULLS = ["#null_a", "#null_b", "#null_c"]

# probes/10, job j-770c6de9-2044-4216-bcdd, us_median.
P10 = {"B_base": 326.732, "A0_base": 92.118, "null_mean": 16.770,
       "null_spread": 0.770, "A0_base_residual": 75.349,
       "in_slope_us_per_MiB": 1.8246, "out_slope_us_per_MiB": 1.1180}
P09_B = 326.782
P08_B = 327.166

# --------------------------------------------------------------------------
# Q1: the production entry, at the GLM-5.2 shipping geometry.
#
# Every value below is a config value with a citation, not a guess:
#   MAX_NUM_BATCHED_TOKENS=1024, MAX_NUM_SEQS=64, MAX_MODEL_LEN=9216
#       -- scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh
#   page_size=1024
#       -- layers/vllm/attention.py get_page_size
#   num_kv_pages_per_block=(2,2,2), num_queries_per_block=(1,128,128), k=topk,
#   compression_ratio=1, enable_early_exit=STREAMIDX_ENABLE_EARLY_EXIT (True)
#       -- layers/vllm/custom_ops/mla_attention_op.py, the GLM-5.2 call site
#   cache_kv uint8[.., page_size//kv_packing, kv_packing, lkv_dim],
#   lkv_dim = align_to(128+1, 128) = 256
#       -- VllmDeepseekV4IndexerCache.get_kv_cache_spec / the in-tree test
#
# pages_per_seq = cdiv(9216, 1024) = 9, and
# num_sublanes_total = max(align_to(9, p) * 1024 // 128 for p in (2,2,2)) = 80,
# so the score matrix is f32[1024, 80*128] = f32[1024, 10240]. The compiled
# Pallas kernel names itself after its own b/n/k, so
# `sc_topk_direct_b1024_n10240_k2048` appearing in the production HLO is the
# check that this arithmetic landed on the shipping shape.
# --------------------------------------------------------------------------
PROD = {
    "num_tokens": 1024, "max_num_seqs": 64, "max_model_len": 9216,
    "page_size": 1024, "kv_packing": 4, "lkv_dim": 256,
    "num_index_heads": 64, "head_dim": 128, "k": 2048,
    "compression_ratio": 1, "num_kv_pages_per_block": (2, 2, 2),
    "num_queries_per_block": (1, 128, 128),
}


def _prod_avals():
    import jax.numpy as jnp
    p = PROD
    pages_per_seq = -(-p["max_model_len"] // p["page_size"])
    npages = p["max_num_seqs"] * pages_per_seq
    nt, hi, d = p["num_tokens"], p["num_index_heads"], p["head_dim"]
    ms = p["max_num_seqs"]
    q = jnp.zeros((nt, hi, d), jnp.float8_e4m3fn)
    w = jnp.zeros((nt, hi), jnp.float32)
    ck = jnp.zeros((npages, p["page_size"] // p["kv_packing"],
                    p["kv_packing"], p["lkv_dim"]), jnp.uint8)
    sl = jnp.full((ms, ), p["max_model_len"], jnp.int32)
    pi = jnp.zeros((npages, ), jnp.int32)
    cq = jnp.zeros((ms + 1, ), jnp.int32)
    dist = jnp.array([0, 0, ms], jnp.int32)
    return (q, w, ck, sl, pi, cq, dist), {
        "pages_per_seq": pages_per_seq, "num_pages": npages,
        "num_sublanes_total": max(
            (-(-pages_per_seq // bp) * bp) * p["page_size"] // 128
            for bp in p["num_kv_pages_per_block"]),
    }


def _print_hlo_report(tag, txt, out, kernel_tag="sc_topk_direct"):
    """Everything Q1/Q3 need, resolved by walking operand NAMES."""
    from hlo_tools import Hlo, summarize
    h = Hlo(txt)
    s = summarize(txt, kernel_tag)
    out[tag] = s
    print(f"\n  === HLO REPORT: {tag} ===", flush=True)
    print(f"  module {s['lines']} lines; kernel custom-calls: "
          f"{s['kernel_calls'] or 'NONE'}", flush=True)
    print(f"  sparse-core-data-format-call present : "
          f"{s['data_format_call']}  ({s['data_format_line_count']} lines)",
          flush=True)
    print(f"  layout-changing copy >=1 MiB present : "
          f"{s['layout_changing_big_copy']}  "
          f"({s['layout_changing_big_copy_count']} of "
          f"{s['big_copy_count']} big copies)", flush=True)
    print(f"  materialized >=1 MiB: copies {s['big_copy_MiB']:.3f} MiB + "
          f"fusions {s['big_fusion_MiB']:.3f} MiB = "
          f"{s['materialized_MiB']:.3f} MiB "
          f"(~{2 * s['materialized_MiB']:.1f} MiB of traffic)", flush=True)
    for ln in s["data_format_lines"]:
        print(f"    [df] {ln[:230]}", flush=True)
    for c in s["big_copy_detail"]:
        print(f"    [copy] {c['name']} in %{c['comp']}  {c['MiB']:.3f} MiB  "
              f"layout {c['layout_in']} -> {c['layout_out']}  "
              f"changes={c['changes']}", flush=True)
        print(f"           operand %{c['src']} = {c['src_shape']}", flush=True)
        print(f"           output    {c['shape']}", flush=True)
    print(f"    inventory (>=1 MiB, async wrappers excluded):", flush=True)
    for i in s["inventory"]:
        print(f"      {i['MiB']:9.3f} MiB  {i['opcode']:16s} "
              f"%{i['name'][:52]:52s} in %{i['comp'][:34]:34s} "
              f"{i['shape'][:70]}", flush=True)
    seeds = [i["name"] for i in h.order
             if kernel_tag in i["name"] and i["opcode"] == "custom-call"]
    if seeds:
        print(f"    operand walk from {seeds[:2]} (names resolved, not "
              f"grepped):", flush=True)
        for nd in h.walk(seeds[:2], 40):
            print(f"      %{nd['name'][:56]:56s} {nd['opcode']:16s} "
                  f"{nd['shape'][:78]}", flush=True)
    else:
        print(f"    !! no `{kernel_tag}` custom-call in this module -- the "
              f"walk cannot be seeded.", file=sys.stderr, flush=True)
    entry = [c for c in h.comps.values() if c["is_entry"]]
    for c in entry:
        print(f"    ENTRY {c['line'][:240]}", flush=True)
    return s


def q1_production(results):
    """Compile the REAL production entry and read its optimized HLO."""
    import inspect
    import jax
    print("\n" + "=" * 74, flush=True)
    print("Q1 -- does the PRODUCTION call path emit the same relayout?",
          flush=True)
    print("=" * 74, flush=True)
    q1 = {"config": PROD}
    results["q1"] = q1
    try:
        import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as sit
    except Exception as exc:
        print(f"  !! cannot import streamindex_topk: {exc!r}",
              file=sys.stderr, flush=True)
        q1["error"] = repr(exc)
        return
    path = sit.__file__
    raw = open(path, "rb").read()
    fn = getattr(sit.streamindex_topk, "__wrapped__", sit.streamindex_topk)
    try:
        lines, start = inspect.getsourcelines(fn)
        deflen = len(lines)
    except Exception:
        start, deflen = -1, 0
    src_lines = raw.decode().splitlines()
    call_lines = [i + 1 for i, ln in enumerate(src_lines)
                  if "sparsecore_topk(" in ln and "def " not in ln
                  and "import" not in ln]
    q1.update({"caller_file": path, "caller_md5": _md5(raw),
               "streamindex_topk_def_line": start,
               "streamindex_topk_def_len": deflen,
               "sparsecore_topk_call_lines": call_lines})
    print(f"  production caller : {path}", flush=True)
    print(f"                      md5 {_md5(raw)}", flush=True)
    print(f"  streamindex_topk  : defined at {os.path.basename(path)}:{start} "
          f"({deflen} lines), jax.jit-decorated", flush=True)
    for cl in call_lines:
        print(f"  sparsecore_topk call site: {os.path.basename(path)}:{cl}"
              f"   |{src_lines[cl - 1].rstrip()}", flush=True)
    for cl in call_lines:
        for j in range(max(0, cl - 14), cl + 6):
            print(f"      {j + 1:4d}| {src_lines[j].rstrip()[:150]}",
                  flush=True)
        break

    avals, derived = _prod_avals()
    q1["derived"] = derived
    print(f"\n  avals (GLM-5.2 shipping geometry, every value cited in the "
          f"PROD comment):", flush=True)
    names = ("q", "indexer_weights", "cache_kv", "seq_lens", "page_indices",
             "cu_q_lens", "distribution")
    for nm, a in zip(names, avals):
        print(f"      {nm:16s} {str(a.dtype):14s} {tuple(a.shape)}",
              flush=True)
    print(f"      derived: {derived}", flush=True)
    print(f"      -> score matrix f32[{PROD['num_tokens']}, "
          f"{derived['num_sublanes_total'] * 128}] "
          f"(reshaped from f32[nt, {derived['num_sublanes_total']}, 128]), "
          f"k={PROD['k']}", flush=True)

    static = dict(k=PROD["k"], compression_ratio=PROD["compression_ratio"],
                  num_kv_pages_per_block=PROD["num_kv_pages_per_block"],
                  num_queries_per_block=PROD["num_queries_per_block"])
    q1["hlo"] = {}
    for tag, eee in (("prod_early_exit_True", True),
                     ("prod_early_exit_False", False)):
        try:
            t0 = time.perf_counter()
            txt = sit.streamindex_topk.lower(
                *avals, enable_early_exit=eee, **static).compile().as_text()
            dt = time.perf_counter() - t0
            print(f"\n  compiled streamindex_topk (enable_early_exit={eee}) "
                  f"in {dt:.1f}s -- THIS IS THE PRODUCTION ENTRY, not a "
                  f"wrapper written by this probe.", flush=True)
            _print_hlo_report(tag, txt, q1["hlo"])
            want = f"sc_topk_direct_b{B}_n{N}_k{K}"
            hit = want in txt
            q1["hlo"][tag]["shipping_shape_kernel_present"] = hit
            print(f"  SHAPE CHECK: `{want}` present in the production HLO: "
                  f"{hit}", flush=True)
            if not hit:
                print(f"  !! the production entry did NOT compile to the "
                      f"shipping topk shape; the avals are wrong and every "
                      f"Q1 statement from this module is about a DIFFERENT "
                      f"shape.", file=sys.stderr, flush=True)
            with open(os.path.join(results["_outdir"], f"hlo-{tag}.txt"),
                      "w") as fh:
                fh.write(txt)
        except Exception as exc:
            print(f"  !! lowering/compiling streamindex_topk "
                  f"(enable_early_exit={eee}) FAILED: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            q1["hlo"][tag] = {"error": f"{type(exc).__name__}: {exc}"}
    del jax


# ---------------------------------------------------------------------------


def _lsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    icept = my - slope * mx
    sstot = sum((y - my) ** 2 for y in ys)
    ssres = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = (1.0 - ssres / sstot) if sstot > 0 else float("nan")
    return slope, icept, r2


def _shape_facts(inc, b, n, k):
    p = inc._pick_partition(b, n, k)
    return {
        "b": b, "n": n, "k": k, "P": p, "n_p": n // p,
        "slice_len": inc._align_to(n, inc.LANES),
        "two_stage": p != 1,
        "waves": inc._cdiv(b, 32),
        "in_mib": b * n * 4 / 2 ** 20,
        "out_mib": b * k * 4 / 2 ** 20,
        "k_ok": (k % inc.LANES == 0) and (0 < k <= 4096),
        "n_ok": (n % inc.LANES == 0),
    }


def _probe_layout_api():
    """Print exactly what this jaxlib exposes. Never assume a spelling."""
    import jax
    found = {"jax": jax.__version__}
    mod = None
    for name in ("jax.experimental.layout", "jax._src.layout"):
        try:
            mod = importlib.import_module(name)
            found["module"] = name
            found["exports"] = sorted(
                x for x in dir(mod) if not x.startswith("_"))
            break
        except Exception as exc:
            found[name] = repr(exc)
    print(f"  layout API probe: {json.dumps(found)}", flush=True)
    if mod is None:
        return None, found
    fmt = getattr(mod, "Format", None)
    lay = getattr(mod, "Layout", None) or getattr(mod, "DeviceLocalLayout",
                                                  None)
    auto = getattr(mod, "AUTO", None)
    if auto is None and lay is not None:
        auto = getattr(lay, "AUTO", None)
    found["has_Format"] = fmt is not None
    found["has_Layout"] = lay is not None
    found["has_AUTO"] = auto is not None
    print(f"    Format={fmt} Layout={lay} AUTO={auto}", flush=True)
    if fmt is None or auto is None:
        return None, found
    return (fmt, lay, auto), found


def build_thunks(mods, bufs, k=K, b=B, n=N):
    """label -> {thunk, hlo_fn, note}. Every wrapper is stated in `note`."""
    import jax
    import jax.numpy as jnp
    out, meta = {}, {}

    ln = bufs["ln"]

    def reg(label, thunk, hlo_fn, note):
        out[label] = thunk
        meta[label] = {"note": note, "hlo_fn": hlo_fn}

    # 1/2 -- the two cross-job controls, called EXACTLY as probes/09 and /10
    #        called them.
    for label in ("B_base", "A0_base"):
        fn = mods[label].sparsecore_topk
        sc = bufs["f32"]
        reg(label,
            (lambda fn=fn, sc=sc: fn(sc, k=k, row_lengths=ln)),
            (lambda fn=fn, sc=sc: fn.lower(sc, k=k,
                                           row_lengths=ln).compile().as_text()),
            "probes/09 arm called verbatim: f32[b,n] jit entry")

    # 3 -- s32 entry. `sparsecore_topk` demands f32, so the entry bitcasts back
    #      immediately; XLA folds bitcast(bitcast(x)) and the fusion has
    #      nothing left to convert.
    m = mods["A0_s32in"]
    g3 = jax.jit(lambda s, l: m.sparsecore_topk(
        jax.lax.bitcast_convert_type(s, jnp.float32), k=k, row_lengths=l))
    reg("A0_s32in", (lambda: g3(bufs["s32"], ln)),
        (lambda: g3.lower(bufs["s32"], ln).compile().as_text()),
        "jit entry takes s32[b,n]; bitcast-convert folds away")

    # 4 -- flat entry, in the layout the custom call constrains its operand to.
    m = mods["A0_flatin"]
    g4 = jax.jit(lambda s, l: m.sparsecore_topk(
        jax.lax.bitcast_convert_type(s.reshape(b, n), jnp.float32),
        k=k, row_lengths=l))
    reg("A0_flatin", (lambda: g4(bufs["flat"], ln)),
        (lambda: g4.lower(bufs["flat"], ln).compile().as_text()),
        "jit entry takes s32[b*n] flat")

    # 5 -- flat result. `sparsecore_topk` ends in `.reshape(b, k)`; reshaping
    #      straight back is what lets XLA drop the pair.
    m = mods["A0_flatout"]
    g5 = jax.jit(lambda s, l: m.sparsecore_topk(
        s, k=k, row_lengths=l).reshape(-1))
    reg("A0_flatout", (lambda: g5(bufs["f32"], ln)),
        (lambda: g5.lower(bufs["f32"], ln).compile().as_text()),
        "jit returns the flat s32[b*k]")

    # 6 -- both.
    m = mods["A0_flatio"]
    g6 = jax.jit(lambda s, l: m.sparsecore_topk(
        jax.lax.bitcast_convert_type(s.reshape(b, n), jnp.float32),
        k=k, row_lengths=l).reshape(-1))
    reg("A0_flatio", (lambda: g6(bufs["flat"], ln)),
        (lambda: g6.lower(bufs["flat"], ln).compile().as_text()),
        "flat s32[b*n] in AND flat s32[b*k] out")

    # 7 -- AUTO input layout. XLA picks the entry layout; the buffer is
    #      committed to it OUTSIDE the timed region. This MOVES the relayout,
    #      it does not prove it unnecessary.
    #
    #      The AUTO-layout jit cannot be CALLED with concrete arrays (jax
    #      refuses: "The layout given to jax.jit is Layout.AUTO but the
    #      corresponding argument passed is a jax.Array with a concrete
    #      layout"), so it is used only to ASK XLA what layout it wants. The
    #      timed thunk is an ordinary `jax.jit` called with arrays already
    #      committed to that layout -- an arg's layout is part of jit's cache
    #      key, so the entry layout follows the buffer. That also keeps this
    #      arm on the same dispatch path as every other arm; calling a
    #      Compiled object directly would not.
    api, layout_info = _probe_layout_api()
    meta["_layout_api"] = layout_info
    if api is not None:
        Format, Layout, AUTO = api
        try:
            m = mods["A0_fmtin"]
            base = lambda s, l: m.sparsecore_topk(s, k=k, row_lengths=l)
            g7a = jax.jit(base, in_shardings=(Format(AUTO), Format(AUTO)))
            sds = [jax.ShapeDtypeStruct(x.shape, x.dtype)
                   for x in (bufs["f32"], ln)]
            c7a = g7a.lower(*sds).compile()
            fmts = getattr(c7a, "input_formats", None)
            if fmts is None:
                fmts = getattr(c7a, "input_layouts", None)
            in_f = fmts[0] if isinstance(fmts, (list, tuple)) else fmts
            f_sc = in_f[0] if isinstance(in_f, (list, tuple)) else in_f
            f_ln = in_f[1] if isinstance(in_f, (list, tuple)) and len(in_f) > 1 \
                else None
            print(f"    AUTO-chosen input formats: {in_f}", flush=True)
            layout_info["auto_input_formats"] = str(in_f)
            layout_info["auto_output_formats"] = str(
                getattr(c7a, "output_formats", None))
            sc7 = jax.device_put(bufs["f32"], f_sc)
            ln7 = jax.device_put(ln, f_ln) if f_ln is not None else ln
            jax.block_until_ready((sc7, ln7))
            layout_info["default_score_format"] = str(
                getattr(bufs["f32"], "format", None))
            layout_info["committed_score_format"] = str(
                getattr(sc7, "format", None))
            same = (layout_info["default_score_format"]
                    == layout_info["committed_score_format"])
            layout_info["auto_equals_default"] = same
            print(f"    default score format  : "
                  f"{layout_info['default_score_format']}", flush=True)
            print(f"    committed score format: "
                  f"{layout_info['committed_score_format']}", flush=True)
            if same:
                print(f"  !! A0_fmtin: XLA-AUTO chose the SAME layout the "
                      f"buffer already had -- this arm is a no-op by "
                      f"construction and its HLO will match A0_base. Report "
                      f"it as such, not as a null result about relayout.",
                      file=sys.stderr, flush=True)
            g7 = jax.jit(base)   # ordinary jit; the ARG carries the layout
            reg("A0_fmtin", (lambda: g7(sc7, ln7)),
                (lambda: g7.lower(sc7, ln7).compile().as_text()),
                "AUTO in-layout, committed OUTSIDE the timed region "
                "(moves the relayout, does not delete it)")
        except Exception as exc:
            print(f"  !! A0_fmtin unavailable: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            layout_info["error"] = f"{type(exc).__name__}: {exc}"
    else:
        print(f"  !! A0_fmtin unavailable: no usable Format/AUTO in this "
              f"jaxlib.", file=sys.stderr, flush=True)

    # 8 -- donation. The score buffer is donated AND returned, so the alias
    #      hands the same buffer back and the timing loop can keep going.
    try:
        m = mods["A0_donate"]
        g8 = jax.jit(lambda s, l: (m.sparsecore_topk(s, k=k, row_lengths=l), s),
                     donate_argnums=(0,))
        st = {"s": bufs["donate"]}

        def _t8():
            o, s2 = g8(st["s"], ln)
            st["s"] = s2
            return o

        reg("A0_donate", _t8,
            (lambda: g8.lower(bufs["donate"], ln).compile().as_text()),
            "donate_argnums=(0,); the donated buffer is returned and rethreaded")
    except Exception as exc:
        print(f"  !! A0_donate unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
    return out, meta


def report(m, order, nulls_present, arm_hlo, meta, ev, results, facts):
    """The whole reporting/fit path. Pure -- called by --self-test too."""
    nulls = [m[nm] for nm in nulls_present]
    null = sum(nulls) / len(nulls)
    null_spread = max(nulls) - min(nulls)
    live = [lb for (lb, _p, _w) in ARMS if lb in m]

    print("\n--- results ---", flush=True)
    print(f"  null floor (mean of {len(nulls)}): {null:10.3f}   "
          f"samples {' '.join(f'{v:.3f}' for v in nulls)}   "
          f"spread {null_spread:.3f}   "
          f"(probes/10 spread {P10['null_spread']:.3f})", flush=True)
    print(f"  EVERY delta below is quoted against that spread as the "
          f"resolution floor.", flush=True)

    print(f"\n  {'arm':11s} {'us_median':>10s} {'us':>10s} {'host%':>6s} "
          f"{'residual':>9s} {'d(A0_base)':>10s} {'dfmt?':>6s} "
          f"{'lcopy?':>7s} {'matMiB':>8s}  note", flush=True)
    base = m.get("A0_base")
    rows = {}
    for lb in live:
        s = arm_hlo.get(lb, {})
        r = m[lb] - null
        d = (m[lb] - base) if base is not None else float("nan")
        rows[lb] = {
            "us_median": m[lb], "residual": r, "delta_vs_A0_base": d,
            "data_format_call": s.get("data_format_call"),
            "layout_changing_big_copy": s.get("layout_changing_big_copy"),
            "materialized_MiB": s.get("materialized_MiB"),
            "big_copy_MiB": s.get("big_copy_MiB"),
            "big_fusion_MiB": s.get("big_fusion_MiB"),
            "note": meta.get(lb, {}).get("note", ""),
        }
        print(f"  {lb:11s} {m[lb]:10.3f} "
              f"{results['raw'][lb]['us']:10.3f} "
              f"{results['raw'][lb]['hostbound_pct']:6.1f} "
              f"{r:9.3f} {d:10.3f} "
              f"{str(s.get('data_format_call')):>6s} "
              f"{str(s.get('layout_changing_big_copy')):>7s} "
              f"{(s.get('materialized_MiB') if s.get('materialized_MiB') is not None else float('nan')):8.2f}"
              f"  {meta.get(lb, {}).get('note', '')[:58]}", flush=True)

    for nm in nulls_present:
        print(f"  null {nm} hostbound {results['raw'][nm]['hostbound_pct']:.1f}% -- a null is host-dominated "
              f"by construction; not flagged.", flush=True)
    for lb in live:
        hp = results["raw"][lb]["hostbound_pct"]
        if hp > 5.0:
            print(f"  !! HOSTBOUND {lb}: {hp:.1f}% > 5% -- the device slope is "
                  f"a smaller part of this arm's wall time than the rule "
                  f"allows; read us_median, and treat its delta with the "
                  f"spread, not the mean.", file=sys.stderr, flush=True)
        if results["raw"][lb]["suspect"]:
            print(f"  !! SUSPECT {lb}", file=sys.stderr, flush=True)

    # ---- the arm-activation rule, applied mechanically ---------------------
    print("\n--- arm activation (an arm that did not delete what its name "
          "says is a NULL, not a win) ---", flush=True)
    verdicts = {}
    base_s = arm_hlo.get("A0_base", {})
    base_mat = base_s.get("materialized_MiB")
    for lb in live:
        if lb in ("A0_base", "B_base"):
            continue
        s = arm_hlo.get(lb, {})
        mat = s.get("materialized_MiB")
        deleted = (mat is not None and base_mat is not None
                   and mat < base_mat - 0.5)
        v = "DELETED-WORK" if deleted else "NO-DELETION"
        verdicts[lb] = {
            "verdict": v, "materialized_MiB": mat,
            "base_materialized_MiB": base_mat,
            "MiB_removed": (base_mat - mat) if deleted else 0.0,
            "still_has_data_format_call": s.get("data_format_call"),
            "still_has_layout_copy": s.get("layout_changing_big_copy"),
            "delta_us": rows[lb]["delta_vs_A0_base"],
        }
        print(f"  {lb:11s} {v:13s} materialized {mat} vs base {base_mat} MiB; "
              f"df_call still present={s.get('data_format_call')}, layout copy "
              f"still present={s.get('layout_changing_big_copy')}; "
              f"delta {rows[lb]['delta_vs_A0_base']:+.3f} us", flush=True)
        if not deleted and abs(rows[lb]["delta_vs_A0_base"]) > null_spread:
            print(f"  !! {lb} moved {rows[lb]['delta_vs_A0_base']:+.3f} us but "
                  f"its HLO still materializes {mat} MiB -- report as a NULL, "
                  f"not a win.", file=sys.stderr, flush=True)

    # ---- Q2 ---------------------------------------------------------------
    print("\n--- Q2: does deleting the relayout remove the residual? ---",
          flush=True)
    q2 = {}
    def sav(lb):
        return (base - m[lb]) if (base is not None and lb in m) else None
    for lb, what in (("A0_flatin", "input relayout"),
                     ("A0_flatout", "output relayout"),
                     ("A0_flatio", "both"),
                     ("A0_s32in", "bitcast-convert only"),
                     ("A0_fmtin", "AUTO in-layout (moved, not deleted)"),
                     ("A0_donate", "donation")):
        s = sav(lb)
        if s is None:
            continue
        mib = (base_mat - arm_hlo.get(lb, {}).get("materialized_MiB", base_mat)
               ) if base_mat is not None else None
        q2[lb] = {"saving_us": s, "MiB_removed": mib,
                  "us_per_MiB_removed": (s / mib) if mib else None,
                  "what": what}
        pm = f"{s / mib:8.4f}" if mib else "     n/a"
        print(f"  {lb:11s} {what:36s} saving {s:9.3f} us   "
              f"MiB removed {('%8.2f' % mib) if mib is not None else '     n/a'}"
              f"   {pm} us/MiB", flush=True)
    if ("A0_flatin" in q2 and "A0_flatout" in q2 and "A0_flatio" in q2
            and q2["A0_flatio"]["saving_us"]):
        add = (q2["A0_flatin"]["saving_us"] + q2["A0_flatout"]["saving_us"])
        ratio = add / q2["A0_flatio"]["saving_us"]
        q2["additivity_ratio"] = ratio
        print(f"  additivity: (flatin + flatout) / flatio = "
              f"{add:.3f} / {q2['A0_flatio']['saving_us']:.3f} = {ratio:.3f}"
              f"   -- probes/08 learned that single-region deletion arms can "
              f"OVERLAP; a ratio far from 1 means they do.", flush=True)
    if base is not None and "A0_flatio" in m:
        resid_left = m["A0_flatio"] - null
        q2["residual_after_both"] = resid_left
        print(f"  residual left after deleting both: {resid_left:.3f} us "
              f"(was {base - null:.3f}). probes/10's byte model predicted "
              f"{P10['in_slope_us_per_MiB'] * 40 + P10['out_slope_us_per_MiB'] * 8:.3f} us "
              f"for the two buffers together.", flush=True)

    # ---- Q3 ---------------------------------------------------------------
    print("\n--- Q3: the output-byte term ---", flush=True)
    q3 = {"p10_out_slope_us_per_MiB": P10["out_slope_us_per_MiB"],
          "p10_predicted_at_8MiB": P10["out_slope_us_per_MiB"] * 8}
    if "A0_flatout" in q2:
        q3["measured_saving_us"] = q2["A0_flatout"]["saving_us"]
        q3["MiB_removed"] = q2["A0_flatout"]["MiB_removed"]
    for lb in ("A0_base", "A0_flatout"):
        s = arm_hlo.get(lb, {})
        outs = [c for c in s.get("big_copy_detail", [])
                if c["MiB"] <= 12.0]
        q3.setdefault("output_side_copies", {})[lb] = outs
        print(f"  {lb:11s} output-side big copies (<=12 MiB): "
              f"{[ (c['name'], c['MiB'], c['layout_in'] + ' -> ' + c['layout_out']) for c in outs ] or 'none'}",
              flush=True)
    print(f"  probes/10 read 1.118 us/MiB on the output axis, i.e. "
          f"{q3['p10_predicted_at_8MiB']:.3f} us at the base shape.",
          flush=True)

    # ---- cross-job --------------------------------------------------------
    print("\n--- cross-job controls ---", flush=True)
    cj = {}
    if "B_base" in m:
        cj["B_base"] = m["B_base"]
        for tag, ref in (("p10", P10["B_base"]), ("p09", P09_B),
                         ("p08", P08_B)):
            d = 100.0 * (m["B_base"] - ref) / ref
            cj[f"B_vs_{tag}_pct"] = d
            print(f"  B_base  {m['B_base']:10.3f}  vs {tag} {ref:8.3f} "
                  f"-> {d:+.2f}%", flush=True)
    if "A0_base" in m:
        cj["A0_base"] = m["A0_base"]
        d = 100.0 * (m["A0_base"] - P10["A0_base"]) / P10["A0_base"]
        cj["A0_vs_p10_pct"] = d
        print(f"  A0_base {m['A0_base']:10.3f}  vs p10 "
              f"{P10['A0_base']:8.3f} -> {d:+.2f}%", flush=True)
        r = m["A0_base"] - null
        cj["A0_base_residual"] = r
        cj["A0_base_residual_vs_p10_pct"] = (
            100.0 * (r - P10["A0_base_residual"]) / P10["A0_base_residual"])
        print(f"  A0_base residual {r:9.3f}  vs p10 "
              f"{P10['A0_base_residual']:8.3f} -> "
              f"{cj['A0_base_residual_vs_p10_pct']:+.2f}%", flush=True)

    results.update({
        "null_mean": null, "null_spread": null_spread,
        "nulls": {nm: m[nm] for nm in nulls_present},
        "us_median": dict(m), "rows": rows, "activation": verdicts,
        "q2": q2, "q3": q3, "cross_job": cj, "order": order,
        "patch_evidence": ev, "facts": facts,
        "arm_notes": {kk: vv.get("note") for kk, vv in meta.items()
                      if isinstance(vv, dict) and "note" in vv},
        "layout_api": meta.get("_layout_api"),
    })
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--harness-dir", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()

    if args.dry_run:
        return _dry_run(args)
    if args.self_test:
        return _self_test(args)

    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    import vllm_torchtpu  # noqa: F401  (registers the TPU platform)

    os.makedirs(args.outdir, exist_ok=True)
    info = pltpu.get_tpu_info()
    print(f"host={os.uname().nodename} devices={jax.device_count()} "
          f"kind={jax.devices()[0].device_kind} jax={jax.__version__}",
          flush=True)
    print(f"sparse_core={info.sparse_core}", flush=True)

    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as inc
    base_path = inc.__file__
    raw = open(base_path, "rb").read()
    print(f"subject: {base_path}  md5={_md5(raw)}", flush=True)
    base_src = raw.decode()

    measure, last_order, harness_kind = _load_harness(
        list(args.harness_dir) + [_HERE, os.getcwd()])

    results = {"probe": "11-relayout-provenance", "subject_path": base_path,
               "subject_md5": _md5(raw), "harness": harness_kind,
               "seed": SEED, "rounds": args.rounds, "jax": jax.__version__,
               "p09_src_md5": P09_SRC_MD5, "_outdir": args.outdir,
               "arms": [{"label": a[0], "patch": a[1], "isolates": a[2]}
                        for a in ARMS],
               "shape": {"b": B, "n": N, "k": K}}

    # --- the verbatim-reuse gate (probes/10's P09_SRC_GATE, verbatim) -------
    for patch, want in sorted(P09_SRC_MD5.items()):
        got = _md5(_patch_src(base_src, patch).encode())
        ok = "OK" if got == want else "MISMATCH"
        print(f"  patch {patch:3s} src_md5={got} vs probes/09 {want} {ok}",
              flush=True)
        assert got == want, (
            f"patch {patch} does not reproduce probes/09's source; every "
            f"cross-job comparison in this probe would be void")
    print("  P09_SRC_GATE_OK", flush=True)

    mods = {}
    for (label, patch, why) in ARMS:
        mods[label] = build_arm(base_src, patch, label)
        with open(os.path.join(args.outdir, f"arm-{label}.py"), "w") as fh:
            fh.write(mods[label]._probe_src)
        print(f"  arm {label:11s} patch={patch:3s} "
              f"drops={','.join(DROPS[patch]) or '-':45s} "
              f"src_md5={_md5(mods[label]._probe_src.encode())}", flush=True)
    ref_mod = build_arm(base_src, "B", "#ref")

    # --- shape gate ---------------------------------------------------------
    f = _shape_facts(inc, B, N, K)
    facts = {lb: f for lb in LABELS}
    print(f"\n--- derived shape facts (checked BEFORE timing) ---", flush=True)
    print(f"  b={f['b']} n={f['n']} k={f['k']} P={f['P']} n_p={f['n_p']} "
          f"slice={f['slice_len']} waves={f['waves']} "
          f"two_stage={f['two_stage']} in_MiB={f['in_mib']:.3f} "
          f"out_MiB={f['out_mib']:.3f}", flush=True)
    assert f["P"] == 1 and not f["two_stage"] and f["k_ok"] and f["n_ok"], (
        "the shipping shape is not P=1 single-stage under this subject")
    print("  SHAPE_GATE_OK -- every arm runs the same P=1 single-stage "
          "program; only the jit signature differs.", flush=True)

    # --- inputs -------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    sc_np = rng.standard_normal((B, N), dtype=np.float32)
    bufs = {
        "f32": jnp.asarray(sc_np),
        "donate": jnp.asarray(sc_np),
        "s32": jnp.asarray(sc_np.view(np.int32)),
        "flat": jnp.asarray(sc_np.view(np.int32).reshape(-1)),
        "ln": jnp.full((B,), N, jnp.int32),
    }
    for kk, v in bufs.items():
        jax.block_until_ready(v)
        print(f"  buffer {kk:7s} {str(v.dtype):6s} {tuple(v.shape)}",
              flush=True)
    print(f"  the s32/flat buffers are the SAME BITS as f32 "
          f"(numpy .view, no conversion), so no arm sees different data.",
          flush=True)

    thunks, meta = build_thunks(mods, bufs)
    dropped = [lb for (lb, _p, _w) in ARMS if lb not in thunks]
    if dropped:
        print(f"  !! arms not built: {dropped}", file=sys.stderr, flush=True)
    results["dropped_arms"] = dropped

    # --- per-arm HLO, BEFORE any timing ------------------------------------
    print("\n" + "=" * 74, flush=True)
    print("PER-ARM OPTIMIZED HLO (printed before any timing)", flush=True)
    print("=" * 74, flush=True)
    arm_hlo = {}
    for lb in list(thunks):
        try:
            txt = meta[lb]["hlo_fn"]()
            _print_hlo_report(lb, txt, arm_hlo)
            with open(os.path.join(args.outdir, f"hlo-{lb}.txt"), "w") as fh:
                fh.write(txt)
        except Exception as exc:
            print(f"  !! HLO for {lb} unavailable: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            arm_hlo[lb] = {"error": f"{type(exc).__name__}: {exc}"}
    results["arm_hlo"] = arm_hlo

    # --- smoke gate: three calls each, before the pool opens ----------------
    print("\n--- arm smoke gate (3 calls each) ---", flush=True)
    bad = []
    for lb in list(thunks):
        try:
            for _ in range(3):
                jax.block_until_ready(thunks[lb]())
            print(f"  {lb:11s} ok", flush=True)
        except Exception as exc:
            print(f"  !! {lb} raises {type(exc).__name__}: {exc} -- DROPPED",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            bad.append(lb)
    for lb in bad:
        thunks.pop(lb, None)
    results["dropped_arms"] = dropped + bad
    assert "A0_base" in thunks and "B_base" in thunks, (
        "both cross-job controls must survive or the job proves nothing")
    print("  SMOKE_GATE_OK", flush=True)

    # --- patch evidence -----------------------------------------------------
    print("", flush=True)
    ev = {}
    ref = np.asarray(ref_mod.sparsecore_topk(bufs["f32"], k=K,
                                             row_lengths=bufs["ln"]))
    for lb in list(thunks):
        got = np.asarray(thunks[lb]()).reshape(B, K)
        same = int((np.sort(got, axis=1) == np.sort(ref, axis=1))
                   .all(axis=1).sum())
        ev[lb] = {"rows_matching_B": same, "rows": B}
        print(f"  patch evidence {lb:11s}: {same:5d}/{B} rows match B@shape",
              flush=True)
    for lb in list(thunks):
        if lb == "B_base":
            continue
        if ev[lb]["rows_matching_B"] == B:
            print(f"  !! {lb} reproduces B exactly -- the patch was NOT "
                  f"effective. Its delta is not a cost.",
                  file=sys.stderr, flush=True)
    print(f"  (probes/09's caveat stands: for an output-suppressing arm this "
          f"gate cannot tell -not written- from -written identically-. No "
          f"out_hbm sentinel exists in this subject.)", flush=True)

    # --- nulls and the one interleaved pool ---------------------------------
    _nullf = jax.jit(lambda a: a + 1.0)
    _nullx = jnp.zeros((8, 128), jnp.float32)
    for nm in NULLS:
        thunks[nm] = lambda: _nullf(_nullx)

    print(f"\n  measuring {len(thunks)} variants in ONE interleaved pool, "
          f"{args.rounds} rounds", flush=True)
    out = measure(thunks, rounds=args.rounds)
    results["raw"] = out
    print("", flush=True)
    for name in list(thunks):
        r = out[name]
        print(f"  {name:11s} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  hostbound={r['hostbound_pct']:5.1f}%  "
              f"suspect={r['suspect']}", flush=True)

    m = {kk: out[kk]["us_median"] for kk in out}
    report(m, last_order(), NULLS, arm_hlo, meta, ev, results, facts)

    path = os.path.join(args.outdir, "relayout_provenance.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {path} (timing complete; Q1 follows)", flush=True)

    # --- Q1 last: it is the heaviest compile and must not risk the timing ---
    try:
        q1_production(results)
    except Exception as exc:
        print(f"  !! Q1 section FAILED: {type(exc).__name__}: {exc} -- the "
              f"timing above is unaffected and already on disk.",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        results.setdefault("q1", {})["fatal"] = f"{type(exc).__name__}: {exc}"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"\nrewrote {path} with Q1", flush=True)
    return 0


# ---------------------------------------------------------------------------


def _dry_run(args) -> int:
    """Host-side gate: anchor counts, then build+compile every patched arm."""
    path = args.subject
    raw = open(path, "rb").read()
    src = raw.decode()
    print(f"subject: {path}  md5={_md5(raw)}")
    anchors = {
        "SRC_LEVELS": SRC_LEVELS, "ANCHOR_WALK": ANCHOR_WALK,
        "SRC_CVT": SRC_CVT, "SRC_HIST": SRC_HIST, "SRC_EMIT": SRC_EMIT,
        "SRC_ZERO_PRIV": SRC_ZERO_PRIV, "SRC_FOLD": SRC_FOLD,
        "SRC_SCAN": SRC_SCAN, "SRC_FIND": SRC_FIND, "SRC_DMA": SRC_DMA,
        "SRC_FILL": SRC_FILL, "SRC_OUTDMA": SRC_OUTDMA,
        "SRC_WAVES": SRC_WAVES,
    }
    bad = 0
    for nm, a in anchors.items():
        c = src.count(a)
        ok = "OK" if c == 1 else "WRONG"
        if c != 1:
            bad += 1
        print(f"  anchor {nm:14s} count={c} {ok}")
    if bad:
        print(f"ANCHOR_GATE_FAIL: {bad} anchor(s) not unique")
        return 1
    print("ANCHOR_GATE_OK")

    bad = 0
    for patch, want in sorted(P09_SRC_MD5.items()):
        got = _md5(_patch_src(src, patch).encode())
        ok = "OK" if got == want else "MISMATCH"
        if got != want:
            bad += 1
        print(f"  patch {patch:3s} src_md5={got} vs probes/09 {want} {ok}")
    if bad:
        print(f"P09_SRC_GATE_FAIL: {bad} patched source(s) differ from "
              f"probes/09 -- the arms are NOT the same arms")
        return 1
    print("P09_SRC_GATE_OK")

    import difflib
    base = _patch_src(src, "B")
    for (label, patch, why) in ARMS:
        s = _patch_src(src, patch)
        compile(s, f"<probe11:{label}>", "exec")
        d = list(difflib.unified_diff(base.splitlines(), s.splitlines(),
                                      n=0, lineterm=""))
        changed = sum(1 for ln in d
                      if (ln.startswith("+") or ln.startswith("-"))
                      and not ln.startswith(("+++", "---")))
        print(f"  arm {label:11s} patch={patch:3s} compiles OK  "
              f"changed_lines_vs_B={changed:2d}  src_md5={_md5(s.encode())}")
    print("COMPILE_GATE_OK")

    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as inc
    f = _shape_facts(inc, B, N, K)
    print(f"\n--- derived shape facts ---")
    print(f"  b={f['b']} n={f['n']} k={f['k']} P={f['P']} n_p={f['n_p']} "
          f"slice={f['slice_len']} waves={f['waves']} "
          f"two_stage={f['two_stage']} in_MiB={f['in_mib']:.3f} "
          f"out_MiB={f['out_mib']:.3f}")
    ok = f["P"] == 1 and not f["two_stage"] and f["k_ok"] and f["n_ok"]
    print(f"SHAPE_GATE_{'OK' if ok else 'FAIL'}")

    # The Q1 aval arithmetic, checked on the host where it is free.
    try:
        import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as sit
        pps = -(-PROD["max_model_len"] // PROD["page_size"])
        nst = max(sit.align_to(pps, bp) * PROD["page_size"] // 128
                  for bp in PROD["num_kv_pages_per_block"])
        print(f"\n--- Q1 aval arithmetic (host-side) ---")
        print(f"  pages_per_seq={pps}  num_sublanes_total={nst}  "
              f"n = {nst * 128}")
        print(f"Q1_SHAPE_GATE_{'OK' if nst * 128 == N else 'FAIL'}")
        if nst * 128 != N:
            return 1
    except Exception as exc:
        print(f"  Q1 aval arithmetic unavailable: {exc!r}")
    return 0 if ok else 1


def _self_test(args) -> int:
    """CPU validation of the WHOLE reporting/fit/JSON path, stubbed kernels."""
    os.makedirs(args.outdir, exist_ok=True)
    from hlo_tools import summarize
    fix = os.path.join(_HERE, "fixture_hlo.txt")
    txt = open(fix).read()
    print(f"fixture: {fix} md5={_md5(open(fix, 'rb').read())}")

    arm_hlo = {}
    s_full = _print_hlo_report("FIXTURE", txt, arm_hlo)
    assert s_full["data_format_call"] is True
    assert s_full["layout_changing_big_copy"] is True
    assert abs(s_full["materialized_MiB"] - 88.0) < 1e-6, s_full
    assert s_full["kernel_calls"] == ["sc_topk_direct_b1024_n10240_k2048.6"]
    print("HLO_PARSER_GATE_OK  (fixture: df=True, layout copy=True, "
          "88.000 MiB materialized, kernel named at the shipping shape)")

    # a stripped fixture: the same module with the relayout removed
    thin = "\n".join(
        ln for ln in txt.splitlines()
        if "copy" not in ln and "fusion" not in ln
        and "sparse-core-data-format" not in ln)
    s_thin = summarize(thin)
    assert s_thin["data_format_call"] is False and s_thin["big_copy"] is False
    print(f"HLO_PARSER_GATE_OK  (stripped fixture: df=False, copies=False, "
          f"{s_thin['materialized_MiB']:.3f} MiB)")

    # ---- build_thunks + measure, on CPU, with STUBBED kernels -------------
    # This is the part that can only be checked by running it: every jit
    # wrapper, the layout API path, the donation rethreading, the HLO hook and
    # the pool itself. The kernel is replaced by a shape-correct stub, so this
    # says nothing about TPU behaviour -- only that no arm raises.
    print("\n--- build_thunks on CPU with stubbed kernels ---")
    import jax
    import jax.numpy as jnp
    sb, sn, sk = 8, 256, 16

    @functools.partial(jax.jit, static_argnames=("k",))
    def _stub_topk(scores, k, row_lengths, **_kw):
        # jitted with a static k, exactly like the real `sparsecore_topk`, so
        # the `.lower(...).compile()` hook in build_thunks is exercised too.
        v = jnp.sum(scores, axis=1, keepdims=True) + row_lengths[:, None]
        return (jnp.broadcast_to(v, (scores.shape[0], k))
                .astype(jnp.int32) + jnp.arange(k, dtype=jnp.int32))

    smods = {}
    for (lb, _p, _w) in ARMS:
        mm = types.ModuleType(f"stub_{lb}")
        mm.sparsecore_topk = _stub_topk
        smods[lb] = mm
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((sb, sn), dtype=np.float32)
    sbufs = {"f32": jnp.asarray(a), "donate": jnp.asarray(a),
             "s32": jnp.asarray(a.view(np.int32)),
             "flat": jnp.asarray(a.view(np.int32).reshape(-1)),
             "ln": jnp.full((sb,), sn, jnp.int32)}
    th, mt = build_thunks(smods, sbufs, k=sk, b=sb, n=sn)
    missing = [lb for (lb, _p, _w) in ARMS if lb not in th]
    print(f"  built {len(th)}/{len(ARMS)} arms; missing={missing or 'none'}")
    ref = None
    for lb in list(th):
        for _ in range(3):
            o = np.asarray(jax.block_until_ready(th[lb]())).reshape(sb, sk)
        if ref is None:
            ref = o
        print(f"  {lb:11s} ok  out{o.shape}  matches_first={bool((o == ref).all())}")
        try:
            txt = mt[lb]["hlo_fn"]()
            from hlo_tools import summarize
            ss = summarize(txt, "stub")
            print(f"              hlo {ss['lines']:4d} lines  "
                  f"df={ss['data_format_call']} "
                  f"bigcopy={ss['big_copy']} mat={ss['materialized_MiB']}")
        except Exception as exc:
            print(f"  !! hlo_fn for {lb} raised {type(exc).__name__}: {exc}")
            raise
    assert not missing, f"arms failed to build on CPU: {missing}"
    _nf = jax.jit(lambda x: x + 1.0)
    _nx = jnp.zeros((8, 128), jnp.float32)
    for nm in NULLS:
        th[nm] = lambda: _nf(_nx)
    mres = _measure_inline(th, rounds=2)
    print(f"  pool of {len(th)} ran; us_median sample: "
          f"{ {kk: round(v['us_median'], 2) for kk, v in list(mres.items())[:3]} }")
    print("BUILD_THUNKS_GATE_OK")

    # ---- the whole reporting path on stub timings -------------------------
    hlos = {"B_base": s_full, "A0_base": s_full, "A0_s32in": s_full,
            "A0_flatin": s_thin, "A0_flatout": s_full, "A0_flatio": s_thin,
            "A0_fmtin": s_thin, "A0_donate": s_full}
    meta = {lb: {"note": w} for (lb, _p, w) in ARMS}
    meta["_layout_api"] = {"module": "stub"}
    m = {"B_base": 326.5, "A0_base": 92.0, "A0_s32in": 55.0,
         "A0_flatin": 25.0, "A0_flatout": 83.0, "A0_flatio": 17.5,
         "A0_fmtin": 26.0, "A0_donate": 92.4,
         "#null_a": 17.2, "#null_b": 16.4, "#null_c": 16.6}
    raw = {kk: {"us": v - 0.4, "us_median": v, "us_mean": v,
                "us_unpaired": v, "hostbound_pct": 3.0 if kk[0] != "#" else 27.,
                "suspect": False} for kk, v in m.items()}
    ev = {lb: {"rows_matching_B": (1024 if lb == "B_base" else 0),
               "rows": 1024} for (lb, _p, _w) in ARMS}
    facts = {lb: {"b": B, "n": N, "k": K, "P": 1, "in_mib": 40.0,
                  "out_mib": 8.0} for lb in LABELS}
    results = {"probe": "11-selftest", "raw": raw, "arm_hlo": hlos,
               "arms": [{"label": a[0], "patch": a[1], "isolates": a[2]}
                        for a in ARMS], "_outdir": args.outdir}
    report(m, [["A0_base"]], NULLS, hlos, meta, ev, results, facts)
    p = os.path.join(args.outdir, "selftest.json")
    with open(p, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    back = json.load(open(p))
    assert back["q2"]["A0_flatin"]["saving_us"] == 67.0
    assert abs(back["q2"]["A0_flatout"]["saving_us"] - 9.0) < 1e-9
    assert back["cross_job"]["A0_base"] == 92.0
    assert back["activation"]["A0_flatin"]["verdict"] == "DELETED-WORK"
    assert back["activation"]["A0_flatout"]["verdict"] == "NO-DELETION"
    assert back["activation"]["A0_s32in"]["verdict"] == "NO-DELETION"
    print(f"\nREPORT_GATE_OK  json round-trips at {p} "
          f"({os.path.getsize(p)} bytes)")
    print("SELF_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
