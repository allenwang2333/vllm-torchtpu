"""Probe 10 -- what does the A0 residual scale with?

A shape sweep, not a patch sweep. Every timed arm except `B_base` is probes/09's
`A0` patched source, byte-for-byte; only the (b, n, k) it is called at varies.

probes/09 left `A0` -- the subject with `dma,cvt,hist,sweep,emit,fill,outdma`
all removed -- reading 92.231 us at b=1024/n=10240/k=2048 against a 16.548 us
null floor. That is a **75.683 us residual with no known mechanism**, and 9.949
us of the same thing at b=64. Three mechanisms are already refuted and are NOT
re-tested here:

  * a per-wave Mosaic barrier -- the Mosaic MLIR has exactly one, outside every
    loop;
  * score-DMA arrival latency -- `A0` deletes the score DMA;
  * `plsc.parallel_loop` entry cost -- probes/08's `-alld` vs `-alls`.

probes/09 also measured the residual's slope against `num_waves` directly
(`A0_W8`, `A0_W16`): **0.009 us/wave, i.e. zero**. So the residual is not
per-wave -- yet it grows 7.6x from b=64 to b=1024, a 16x change in b. Three
candidates remain, and the shape axes separate them:

  * the INPUT buffer `scores_hbm`, `b*n` int32  (41.9 MB at b=1024, n=10240);
  * the OUTPUT buffer `out_hbm`, `b*k` int32     (8.39 MB at b=1024, k=2048);
  * something PER ROW (b), independent of both byte counts.

Arms (8 timed + 3 nulls). Every `A0_*` arm is the same patched source in its own
module object, so no jax.jit cache can alias, and every shape point is asserted
`P == 1` single-stage before it is fitted.

    B_base    B  @ b=1024 n=10240 k=2048   CROSS-JOB control (probes/08, /09)
    A0_base   A0 @ b=1024 n=10240 k=2048   CROSS-JOB control (probes/09)
    A0_n/4    A0 @ b=1024 n= 2560 k=2048   input bytes 1/4
    A0_n/2    A0 @ b=1024 n= 5120 k=2048   input bytes 1/2
    A0_k/4    A0 @ b=1024 n=10240 k= 512   output bytes 1/4
    A0_k/2    A0 @ b=1024 n=10240 k=1024   output bytes 1/2
    A0_b/4    A0 @ b= 256 n=10240 k=2048   b 1/4  (both byte counts also 1/4)
    A0_b/16   A0 @ b=  64 n=10240 k=2048   b 1/16 (reproduces probes/09's 9.949)
    #null_a #null_b #null_c                the harness fence. Three of them.

`B_base` is probes/09's `B` arm -- the subject with `REFINEMENT_LEVELS = ()` and
nothing else changed. That is the arm probes/08 and probes/09 both read at
327.166 / 326.782, so it is the arm that can be compared to them. Its patched
source md5 is asserted equal to probes/09's `fd46ef14b38fd7781885e73d0ca33965`,
and `A0`'s to `547031a2fd48545550d864b008d817ff`, before anything is measured;
that assertion is what makes "reused verbatim" a fact rather than an intention.

TWO DELIBERATE DEVIATIONS from probes/09's structure, both stated in the log:

 1. **One `measure()` call over all shapes**, not one per b. probes/09 looped
    over b and called the harness once per b, so its b=1024 and b=64 numbers sat
    in different interleaved-rounds pools and could only be compared across
    pools. Here the whole question IS the cross-shape comparison, so every arm
    is interleaved against every other arm in a single pool and pod drift lands
    on all of them equally.

 2. **Three nulls, not "two nulls per b".** The harness null is
    `jax.jit(lambda a: a + 1.0)` on an `(8, 128)` array; it does not depend on
    b, so a "null at b=256" is not a distinct quantity -- it is a repeat sample
    of the one fence. Three samples are registered so the fence has a spread
    rather than a value. A constant floor shifts every fit's INTERCEPT and no
    fit's SLOPE, so the choice cannot manufacture a driver; probes/09's own
    b=1024 vs b=64 floors (16.548 vs 15.747, 0.80 apart) are printed against
    this job's intra-pool spread so the b-independence is checked, not assumed.

NOT a correctness experiment. Every `A0_*` arm computes the wrong answer by
construction -- it never writes `out_hbm` -- and the printed PATCH EVIDENCE
against a reference `B` at the same shape is what proves each patch took.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

import vllm_torchtpu  # noqa: F401  (registers the TPU platform)
from jax.experimental.pallas import tpu as pltpu

SEED = 20260904  # same stream as probes 05, 07, 08 and 09
REPS_LO = 20
REPS_HI = 60

_LAST_ORDER: list[list[str]] = []


# --------------------------------------------------------------------------
# Timing harness -- byte-identical protocol to probes 07, 08 and 09. Prefer the
# FROZEN instrument so the numbers compose with probes 03, 07, 08 and 09
# directly; the inline copy is the same estimator and prints which one ran.
# --------------------------------------------------------------------------
def _time(fn, reps: int) -> float:
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def _measure_inline(variants, rounds=9, shuffle_seed=20260902):
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
# probes/09/probe.py. Anchors are quoted with their exact leading whitespace and
# every `_sub` call asserts the occurrence count, so a source drift under this
# probe is a hard failure and never a silently-unpatched arm.
# --------------------------------------------------------------------------
SRC_LEVELS = "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))"
NO_LEVELS = "REFINEMENT_LEVELS = ()"

# The dynamic zero. `Z_DYN` keys off the fori_loop induction variable `w`,
# NOT off `i_hi`: `i_hi` is provably non-negative by interval analysis through
# `jnp.where(eff_len > 0, jnp.minimum(...), 0)`, so `i_hi < 0` is exactly the
# kind of predicate XLA folds. `w` arrives as a parameter of the loop body
# computation and carries no range with it.
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

# `& (w < 0)` and NOT a literal `False`, for the same reason probes/08's
# PIN_TRUE uses `| (w >= 0)`: XLA constant-folds a literal, which turns a
# dynamic-zero bound into a STATIC one and eliminates the loop at trace time,
# measuring something the incumbent never does. `w` is the `fori_loop`
# induction variable, arrives as a parameter of the body computation, and
# carries no range with it, so the predicate survives to run time.
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


# Which passes each patch removes. Verbatim from probes/09; only the two entries
# this probe actually uses are kept, and `sweep` stays a single group name so
# `A0` is byte-for-byte probes/09's `A0`.
DROPS = {
    "B":  (),
    "A0": ("dma", "cvt", "hist", "sweep", "emit", "fill", "outdma"),
}

# Wave-loop bound override. Empty here -- probes/09 already measured the wave
# slope at 0.009 us/wave and this probe does not re-ask that question. The hook
# is kept so `_patch_src` stays byte-comparable with probes/09's.
WAVES: dict[str, int] = {}

# Drops that need the `z_dyn` binding injected. `fill` and `outdma` splice
# `& (w < 0)` in directly and do not.
Z_DYN_DROPS = frozenset(
    ("cvt", "hist", "emit", "sweep", "zero", "fold", "scan", "find"))

# probes/09's own patched-source md5s, job j-4e0df30a-d833-4c1c-825d. Asserted,
# not printed-and-hoped: if `_patch_src` here does not reproduce them bit for
# bit then "the same A0" is false and every cross-job comparison below is void.
P09_SRC_MD5 = {"B": "fd46ef14b38fd7781885e73d0ca33965",
               "A0": "547031a2fd48545550d864b008d817ff"}


def _patch_src(base_src: str, name: str) -> str:
    """The single source of truth for what each patch does. Verbatim probes/09."""
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
    """One module object per LABEL, even when two labels share a patch.

    `A0_base` and `A0_k/2` run identical source at different shapes. Giving them
    one module each costs nothing and removes the question of whether a jax.jit
    cache entry could be shared between two rows of the table.
    """
    src = _patch_src(base_src, patch)
    mod = types.ModuleType(f"sctopk10_{abs(hash((patch, label)))}")
    mod.__file__ = f"<probe10:{label}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


# ---------------------------------------------------------------------------
# The arm table. `patch` is which of the two patched sources runs; (b, n, k) is
# what varies. Order here is the registration order, NOT the visiting order --
# the harness reshuffles every round and logs what it used.
# ---------------------------------------------------------------------------
ARMS = [
    # label,      patch,  b,     n,      k,     isolates
    ("B_base",    "B",    1024, 10240, 2048, "cross-job control (p08/p09)"),
    ("A0_base",   "A0",   1024, 10240, 2048, "cross-job control (p09)"),
    ("A0_n/4",    "A0",   1024,  2560, 2048, "input bytes 1/4"),
    ("A0_n/2",    "A0",   1024,  5120, 2048, "input bytes 1/2"),
    ("A0_k/4",    "A0",   1024, 10240,  512, "output bytes 1/4"),
    ("A0_k/2",    "A0",   1024, 10240, 1024, "output bytes 1/2"),
    ("A0_b/4",    "A0",    256, 10240, 2048, "b 1/4 (bytes also 1/4)"),
    ("A0_b/16",   "A0",     64, 10240, 2048, "b 1/16 (p09 residual 9.949)"),
]
LABELS = [a[0] for a in ARMS]
NULLS = ["#null_a", "#null_b", "#null_c"]

IN_FIT = ["A0_base", "A0_n/2", "A0_n/4"]
OUT_FIT = ["A0_base", "A0_k/2", "A0_k/4"]
B_FIT = ["A0_base", "A0_b/4", "A0_b/16"]

# probes/09, job j-4e0df30a-d833-4c1c-825d, us_median. probes/08,
# job j-300d4690-d53e-441e-9d39, us_median.
P09 = {"B_base": 326.782, "A0_base": 92.231,
       "A0_residual_b1024": 75.683, "A0_residual_b64": 9.949,
       "null_b1024": 16.548, "null_b64": 15.747}
P08_B = 327.166


def _lsq(xs, ys):
    """Ordinary least squares through 3 points. slope, intercept, R^2."""
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


# 1 MiB / (slope us) expressed in GB/s with GB = 1e9 bytes.
MIB_PER_US_TO_GBS = (2 ** 20) / 1e9 / 1e-6  # = 1048.576


def _shape_facts(inc, b, n, k):
    p = inc._pick_partition(b, n, k)
    return {
        "b": b, "n": n, "k": k, "P": p, "n_p": n // p,
        "slice_len": inc._align_to(n, inc.LANES),
        "two_stage": p != 1,
        "waves": inc._cdiv(b, 32),
        "in_mb": b * n * 4 / 2 ** 20,
        "out_mb": b * k * 4 / 2 ** 20,
        "resident_vmem_b": inc._align_to(n, inc.LANES) * 4,
        "rowbuf_vmem_b": (k + inc.LANES) * 4,
        "k_ok": (k % inc.LANES == 0) and (0 < k <= 4096),
        "n_ok": (n % inc.LANES == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--harness-dir", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true",
                    help="build and compile every arm, print line deltas, exit")
    ap.add_argument("--subject", default=None,
                    help="dry-run only: path to the subject source")
    args = ap.parse_args()

    if args.dry_run:
        return _dry_run(args)

    os.makedirs(args.outdir, exist_ok=True)

    info = pltpu.get_tpu_info()
    print(f"host={os.uname().nodename} devices={jax.device_count()} "
          f"kind={jax.devices()[0].device_kind}", flush=True)
    print(f"sparse_core={info.sparse_core}", flush=True)

    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as inc
    base_path = inc.__file__
    raw = open(base_path, "rb").read()
    print(f"subject: {base_path}  md5={_md5(raw)}", flush=True)
    base_src = raw.decode()

    measure, last_order, harness_kind = _load_harness(
        list(args.harness_dir) + [
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/microbench", "/app/microbench", os.getcwd(),
        ])

    # --- the verbatim-reuse gate ------------------------------------------
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
    for (label, patch, b, n, k, why) in ARMS:
        mods[label] = build_arm(base_src, patch, label)
        safe = label.replace("/", "_over_").replace("-", "no")
        with open(os.path.join(args.outdir, f"arm-{safe}.py"), "w") as fh:
            fh.write(mods[label]._probe_src)
        print(f"  arm {label:9s} patch={patch:3s} "
              f"drops={','.join(DROPS[patch]) or '-':45s} "
              f"src_md5={_md5(mods[label]._probe_src.encode())}", flush=True)
    # Never timed. Only used to produce the reference top-k at each shape so
    # every A0 arm's patch has evidence at ITS OWN shape.
    ref_mod = build_arm(base_src, "B", "#ref")

    # --- what the harness derives for each (b, n, k) ------------------------
    print("\n--- derived shape facts (checked BEFORE timing) ---", flush=True)
    print(f"  {'label':9s} {'b':>5s} {'n':>6s} {'k':>5s} {'P':>2s} "
          f"{'n_p':>6s} {'slice':>6s} {'waves':>5s} {'two_stage':>9s} "
          f"{'in_MB':>8s} {'out_MB':>8s} {'residVMEM':>9s} {'rowbufVMEM':>10s}",
          flush=True)
    facts, excluded = {}, []
    for (label, patch, b, n, k, why) in ARMS:
        f = _shape_facts(inc, b, n, k)
        facts[label] = f
        print(f"  {label:9s} {b:5d} {n:6d} {k:5d} {f['P']:2d} "
              f"{f['n_p']:6d} {f['slice_len']:6d} {f['waves']:5d} "
              f"{str(f['two_stage']):>9s} {f['in_mb']:8.3f} {f['out_mb']:8.3f} "
              f"{f['resident_vmem_b']:9d} {f['rowbuf_vmem_b']:10d}", flush=True)
        if f["two_stage"] or f["P"] != 1 or not f["k_ok"] or not f["n_ok"]:
            excluded.append(label)
    if excluded:
        for label in excluded:
            print(f"  !! {label} is NOT COMPARABLE (P={facts[label]['P']}, "
                  f"two_stage={facts[label]['two_stage']}, "
                  f"k_ok={facts[label]['k_ok']}, n_ok={facts[label]['n_ok']}) "
                  f"-- a P>1 or two-stage point runs a different program, so it "
                  f"is excluded from every fit below.", file=sys.stderr,
                  flush=True)
    else:
        print("  all shape points are P=1 single-stage -- no regime change "
              "across the sweep, every point is fittable.", flush=True)
    print(f"  n changes BOTH scores_hbm bytes AND resident_vmem "
          f"({facts['A0_base']['resident_vmem_b']} -> "
          f"{facts['A0_n/4']['resident_vmem_b']} B); k changes BOTH out_hbm "
          f"bytes AND rowbuf_vmem ({facts['A0_base']['rowbuf_vmem_b']} -> "
          f"{facts['A0_k/4']['rowbuf_vmem_b']} B). Both VMEM buffers are "
          f"scratch allocations, not traffic, and A0 touches neither.",
          flush=True)

    # --- inputs, shared wherever two arms take the same (b, n) --------------
    rng = np.random.default_rng(SEED)
    scores_cache, len_cache = {}, {}
    for (label, patch, b, n, k, why) in ARMS:
        if (b, n) not in scores_cache:
            scores_cache[(b, n)] = jnp.asarray(
                rng.standard_normal((b, n), dtype=np.float32))
            len_cache[(b, n)] = jnp.full((b,), n, jnp.int32)
    print(f"\n  score buffers: {len(scores_cache)} distinct (b, n) "
          f"{sorted(scores_cache)}", flush=True)

    thunks = {}
    for (label, patch, b, n, k, why) in ARMS:
        fn = mods[label].sparsecore_topk
        sc, ln = scores_cache[(b, n)], len_cache[(b, n)]
        thunks[label] = (lambda fn=fn, sc=sc, ln=ln, k=k:
                         fn(sc, k=k, row_lengths=ln))
    _nullf = jax.jit(lambda a: a + 1.0)
    _nullx = jnp.zeros((8, 128), jnp.float32)
    for nm in NULLS:
        thunks[nm] = lambda: _nullf(_nullx)

    # --- PATCH EVIDENCE, before timing. -------------------------------------
    # Each A0 arm is compared to a reference `B` AT ITS OWN SHAPE, not to
    # `B_base`; a b=64 arm and a b=1024 control have no row correspondence.
    print("", flush=True)
    ev = {}
    for (label, patch, b, n, k, why) in ARMS:
        got = np.asarray(thunks[label]())
        ref = np.asarray(ref_mod.sparsecore_topk(
            scores_cache[(b, n)], k=k, row_lengths=len_cache[(b, n)]))
        same = int((np.sort(got, axis=1) == np.sort(ref, axis=1))
                   .all(axis=1).sum())
        ev[label] = {"rows_matching_B": same, "rows": int(b)}
        print(f"  patch evidence {label:9s}: {same:5d}/{b} rows match B@shape",
              flush=True)
    for (label, patch, b, n, k, why) in ARMS:
        if patch == "B":
            continue
        if ev[label]["rows_matching_B"] == b:
            print(f"  !! {label} reproduces B exactly -- the patch was NOT "
                  f"effective. Its delta is not a cost.",
                  file=sys.stderr, flush=True)

    # --- ONE interleaved pool over every shape ------------------------------
    print(f"\n  measuring {len(thunks)} variants in ONE interleaved pool, "
          f"{args.rounds} rounds (see module docstring, deviation 1)",
          flush=True)
    out = measure(thunks, rounds=args.rounds)

    print("", flush=True)
    for name in list(thunks):
        r = out[name]
        print(f"  {name:9s} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  hostbound={r['hostbound_pct']:5.1f}%  "
              f"suspect={r['suspect']}", flush=True)

    m = {kk: out[kk]["us_median"] for kk in out}
    nulls = [m[nm] for nm in NULLS]
    null = sum(nulls) / len(nulls)
    null_spread = max(nulls) - min(nulls)

    # ---------------------------------------------------------------- (2) --
    print("\n--- residual driver ---", flush=True)
    print(f"  null floor (mean of {len(NULLS)}): {null:10.3f}   "
          f"samples {' '.join(f'{v:.3f}' for v in nulls)}   "
          f"spread {null_spread:.3f}", flush=True)
    print(f"  the harness null is a b-INDEPENDENT (8,128) jit, so one floor is "
          f"subtracted at every b (deviation 2). probes/09 read it at "
          f"{P09['null_b1024']:.3f} @b=1024 and {P09['null_b64']:.3f} @b=64, "
          f"{abs(P09['null_b1024'] - P09['null_b64']):.3f} apart, against this "
          f"job's intra-pool spread of {null_spread:.3f}.", flush=True)
    print(f"  a constant floor moves every INTERCEPT below and no SLOPE.",
          flush=True)

    resid = {}
    print(f"\n  {'arm':9s} {'b':>5s} {'n':>6s} {'k':>5s} {'in_MB':>8s} "
          f"{'out_MB':>8s} {'us_median':>10s} {'null':>7s} {'residual':>9s}",
          flush=True)
    for (label, patch, b, n, k, why) in ARMS:
        if patch != "A0":
            continue
        r = m[label] - null
        resid[label] = r
        f = facts[label]
        print(f"  {label:9s} {b:5d} {n:6d} {k:5d} {f['in_mb']:8.3f} "
              f"{f['out_mb']:8.3f} {m[label]:10.3f} {null:7.3f} {r:9.3f}",
              flush=True)

    fits = {}

    def _fit(name, members, xkey, xunit, bytes_axis, note):
        miss = [x for x in members if x in excluded]
        if miss:
            print(f"\n  fit {name}: SKIPPED -- {miss} not comparable",
                  flush=True)
            fits[name] = None
            return
        xs = [facts[x][xkey] for x in members]
        ys = [resid[x] for x in members]
        s, ic, r2 = _lsq(xs, ys)
        print(f"\n  fit: residual vs {name}   over {members}", flush=True)
        for x, y, lab in zip(xs, ys, members):
            print(f"      {lab:9s} x={x:10.3f} {xunit:5s} "
                  f"residual={y:9.3f} us   "
                  f"fitted={ic + s * x:9.3f}", flush=True)
        print(f"      slope     = {s:10.4f} us/{xunit}", flush=True)
        print(f"      intercept = {ic:10.3f} us", flush=True)
        print(f"      R2        = {r2:10.5f}", flush=True)
        bw = None
        if bytes_axis and s > 0:
            bw = MIB_PER_US_TO_GBS / s
            print(f"      implied BW= {bw:10.1f} GB/s  (one pass over the "
                  f"buffer; halve it if the traffic is read+write)", flush=True)
        elif bytes_axis:
            print(f"      implied BW= n/a (slope <= 0)", flush=True)
        else:
            print(f"      per row   = {s * 1000.0:10.3f} ns/row", flush=True)
        print(f"      {note}", flush=True)
        fits[name] = {"members": members, "xs": xs, "ys": ys, "slope": s,
                      "intercept": ic, "r2": r2, "bw_gbs": bw}

    _fit("input MB", IN_FIT, "in_mb", "MB", True,
         "clean: b and out_MB are constant across these three.")
    _fit("output MB", OUT_FIT, "out_mb", "MB", True,
         "clean: b and in_MB are constant across these three.")
    _fit("b (rows)", B_FIT, "b", "row", False,
         "CONFOUNDED: b/4 and b/16 scale in_MB and out_MB by the same factor, "
         "so this slope absorbs both byte axes. See the bytes-only prediction "
         "below, which removes them.")

    # The confound removal. Predict the two b arms from the two clean byte
    # slopes alone; whatever is left over is the genuine per-row term.
    fi, fo = fits.get("input MB"), fits.get("output MB")
    print("\n  bytes-only prediction for the b arms "
          "(base residual + s_in*d_in + s_out*d_out):", flush=True)
    if fi and fo:
        base = resid["A0_base"]
        fb = facts["A0_base"]
        print(f"      {'arm':9s} {'d_in_MB':>9s} {'d_out_MB':>9s} "
              f"{'predicted':>10s} {'measured':>10s} {'leftover':>10s} "
              f"{'per row':>10s}", flush=True)
        for lab in ("A0_b/4", "A0_b/16"):
            if lab in excluded:
                continue
            f = facts[lab]
            din = f["in_mb"] - fb["in_mb"]
            dout = f["out_mb"] - fb["out_mb"]
            pred = base + fi["slope"] * din + fo["slope"] * dout
            meas = resid[lab]
            left = meas - pred
            db = fb["b"] - f["b"]
            print(f"      {lab:9s} {din:9.3f} {dout:9.3f} {pred:10.3f} "
                  f"{meas:10.3f} {left:10.3f} "
                  f"{(-left / db) * 1000.0:10.3f} ns", flush=True)
        print(f"      leftover is what b costs BEYOND its byte counts. A "
              f"leftover inside the null spread ({null_spread:.3f} us) is no "
              f"per-row term at all.", flush=True)
    else:
        print("      SKIPPED -- a byte fit was not available.", flush=True)

    # ---------------------------------------------------------------- (3) --
    print("\n--- consistency ---", flush=True)
    d09b = 100.0 * (m["B_base"] - P09["B_base"]) / P09["B_base"]
    d08b = 100.0 * (m["B_base"] - P08_B) / P08_B
    print(f"  B_base   {m['B_base']:10.3f}  vs probes/09 {P09['B_base']:8.3f} "
          f"-> {d09b:+.2f}%   vs probes/08 {P08_B:8.3f} -> {d08b:+.2f}%",
          flush=True)
    d09a = 100.0 * (m["A0_base"] - P09["A0_base"]) / P09["A0_base"]
    print(f"  A0_base  {m['A0_base']:10.3f}  vs probes/09 "
          f"{P09['A0_base']:8.3f} -> {d09a:+.2f}%", flush=True)
    r_base = resid["A0_base"]
    d09rb = 100.0 * (r_base - P09["A0_residual_b1024"]) / \
        P09["A0_residual_b1024"]
    print(f"  A0_base residual  {r_base:9.3f}  vs probes/09 "
          f"{P09['A0_residual_b1024']:8.3f} -> {d09rb:+.2f}%", flush=True)
    r_b16 = resid["A0_b/16"]
    d09r = 100.0 * (r_b16 - P09["A0_residual_b64"]) / P09["A0_residual_b64"]
    print(f"  A0_b/16 residual  {r_b16:9.3f}  vs probes/09 "
          f"{P09['A0_residual_b64']:8.3f} -> {d09r:+.2f}%   "
          f"(probes/09 subtracted its own b=64 floor {P09['null_b64']:.3f}; "
          f"this job subtracts {null:.3f})", flush=True)
    print(f"  A0_b/16 residual against probes/09's floor instead: "
          f"{m['A0_b/16'] - P09['null_b64']:9.3f}", flush=True)

    # ------------------------------------------------- HLO, best effort -----
    # Where the bytes would have to move if the driver is a buffer copy. Bounded
    # and guarded: a diagnostic, never a measurement.
    print("\n--- optimized HLO of A0_base (diagnostic, best-effort) ---",
          flush=True)
    hlo_lines = []
    try:
        sc = scores_cache[(1024, 10240)]
        ln = len_cache[(1024, 10240)]
        txt = mods["A0_base"].sparsecore_topk.lower(
            sc, k=2048, row_lengths=ln).compile().as_text()
        keep = [ln_ for ln_ in txt.splitlines()
                if (" copy(" in ln_ or " bitcast(" in ln_
                    or "custom-call" in ln_ or " fusion(" in ln_
                    or ln_.strip().startswith("ENTRY"))]
        print(f"  optimized module: {len(txt.splitlines())} lines; "
              f"{len(keep)} copy/bitcast/fusion/custom-call/ENTRY lines",
              flush=True)
        for ln_ in keep[:30]:
            s = ln_.strip()
            hlo_lines.append(s)
            print(f"    {s[:220]}", flush=True)
        if len(keep) > 30:
            print(f"    ... {len(keep) - 30} more suppressed", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"  unavailable: {type(exc).__name__}: {exc}", flush=True)

    results = {
        "probe": "10-residual-driver",
        "subject_md5": _md5(raw), "harness": harness_kind,
        "seed": SEED, "rounds": args.rounds,
        "p09_src_md5": P09_SRC_MD5,
        "arms": [{"label": a[0], "patch": a[1], "b": a[2], "n": a[3],
                  "k": a[4], "isolates": a[5]} for a in ARMS],
        "facts": facts, "excluded": excluded,
        "patch_evidence": ev, "raw": out, "order": last_order(),
        "nulls": {nm: m[nm] for nm in NULLS},
        "null_mean": null, "null_spread": null_spread,
        "us_median": {kk: m[kk] for kk in m},
        "residual": resid, "fits": fits,
        "cross_job": {
            "B_base": m["B_base"], "p09_B": P09["B_base"], "p08_B": P08_B,
            "B_vs_p09_pct": d09b, "B_vs_p08_pct": d08b,
            "A0_base": m["A0_base"], "p09_A0": P09["A0_base"],
            "A0_vs_p09_pct": d09a,
            "A0_base_residual": r_base,
            "p09_A0_residual_b1024": P09["A0_residual_b1024"],
            "A0_base_residual_vs_p09_pct": d09rb,
            "A0_b16_residual": r_b16,
            "p09_A0_residual_b64": P09["A0_residual_b64"],
            "A0_b16_residual_vs_p09_pct": d09r,
        },
        "hlo_lines": hlo_lines,
    }
    path = os.path.join(args.outdir, "residual_driver.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {path}", flush=True)
    return 0


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
    for (label, patch, b, n, k, why) in ARMS:
        s = _patch_src(src, patch)
        compile(s, f"<probe10:{label}>", "exec")
        d = list(difflib.unified_diff(base.splitlines(), s.splitlines(),
                                      n=0, lineterm=""))
        changed = sum(1 for ln in d
                      if (ln.startswith("+") or ln.startswith("-"))
                      and not ln.startswith(("+++", "---")))
        print(f"  arm {label:9s} patch={patch:3s} compiles OK  "
              f"changed_lines_vs_B={changed:2d}  src_md5={_md5(s.encode())}")
    print("COMPILE_GATE_OK")

    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as inc
    print("\n--- derived shape facts ---")
    print(f"  {'label':9s} {'b':>5s} {'n':>6s} {'k':>5s} {'P':>2s} "
          f"{'n_p':>6s} {'slice':>6s} {'waves':>5s} {'two_stage':>9s} "
          f"{'in_MB':>8s} {'out_MB':>8s} {'residVMEM':>9s} {'rowbufVMEM':>10s}")
    bad = 0
    for (label, patch, b, n, k, why) in ARMS:
        f = _shape_facts(inc, b, n, k)
        flag = ""
        if f["two_stage"] or f["P"] != 1 or not f["k_ok"] or not f["n_ok"]:
            flag = "  NOT COMPARABLE"
            bad += 1
        print(f"  {label:9s} {b:5d} {n:6d} {k:5d} {f['P']:2d} "
              f"{f['n_p']:6d} {f['slice_len']:6d} {f['waves']:5d} "
              f"{str(f['two_stage']):>9s} {f['in_mb']:8.3f} {f['out_mb']:8.3f} "
              f"{f['resident_vmem_b']:9d} {f['rowbuf_vmem_b']:10d}{flag}")
    print(f"SHAPE_GATE_{'FAIL' if bad else 'OK'}"
          f"{'' if not bad else f': {bad} point(s) not comparable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
