"""Probe 08 -- what is the 81 us of per-wave cost that belongs to no op?

hardware_map.md stage C prices the base (REFINEMENT_LEVELS=()) arm at
9.531 us/wave and can account for only 6.991 of it:

    score DMA 1.045 | cvt 0.643 | hist0 3.636 | L0 sweep 0.600
    emit 0.858 | out DMA 0.209                 = 6.991 us/wave
    measured                                     9.531 us/wave
    UNATTRIBUTED                                 2.540 us/wave = 81.28 us
                                                 = 17.9% of the prefill kernel

That residual is larger than the score DMA and is the third-biggest row in the
block. It is `measured by subtraction`, which is the weakest tag in the file --
every error in every other row lands in it, which is exactly the mistake
report.md §5d made with the refinement term.

Stage C also rules out one whole class of explanation before this probe runs.
At the calibrated 574 ps/bundle (probes/06 Q4's 7 bundles per 64 elements for
`cvt`, against probes/07's measured 20.589 us), 2.540 us/wave is 4,425 bundles.
The ENTIRE program is 1,168 bundles. So the residual is 3.8x the whole kernel's
static instruction count, per wave: it is not instruction issue, and no amount
of scheduling, unrolling or hoisting can be its cause.

Three mechanisms survive, and this probe separates them by removing exactly one
thing at a time from the base arm:

    M1  the `sc_tpu.sfence` + `sc_tpu.barrier` Mosaic emits per wave, which
        makes the wave cost `max` over 32 subcores
    M2  score-DMA *latency* that the 33.43 us bandwidth figure does not
        contain -- with one buffer the subcore stalls a full HBM round trip
        every wave
    M3  `plsc.parallel_loop` entry/exit (about 10 per wave) and `pl.when`
        predicate evaluation

M2 is the one that matters most, and not because it is the likeliest. If the
residual is DMA latency then the caller's double buffer recovers D1 AND D3
together and its prize is up to -25.2%, not the -7.18% report.md §6 models. A
factor of 3.5 on the headline candidate is decided by the `-dma` arm alone.

Four other open questions ride along, because they are the same job:

    D4    the 22.042 us launch intercept is 43.2% of the DECODE kernel, and
          this instrument's own null reads 27.19-27.36 us -- ABOVE it. The two
          are not subtractable (the intercept already contains dispatch), so
          the separation needs launch-count slopes on the kernel and on a
          trivial op measured side by side.
    D5    deleting V4's 16x lane-private histogram replication is worth 13.6%
          of the kernel and is gated on ONE question: does
          `plsc.addupdate_scatter` accumulate correctly when several lanes
          target the same address? The `R1` arm answers it by CORRECTNESS,
          not by timing -- and times it too, so if the answer is yes the whole
          `mutate` edge is priced in the same job.
    probe 05 gap: `implied active levels` reads 2.231 at prefill against a
          predicted 2.03. A skipped level is not free -- it still enters four
          `parallel_loop`s at bound 0 and runs the scalar epilogue. `P3T` (3
          levels, `done` pinned TRUE) measures that overhead directly.
    M3's static/dynamic half: `-alls` gives every removed loop a STATIC 0
          bound, so the loop vanishes at trace time; `-alld` gives it a
          DYNAMIC 0, so the loop is entered and exits immediately. The
          difference between them IS the `parallel_loop` entry cost.

Arms (17). All patched textually from the installed module, every substitution
asserted to hit an exact count, each arm exec'd into its own module object so
no jax.jit cache can alias. Every arm's patched source is written to arms/.

    B      REFINEMENT_LEVELS = ()                the base being decomposed
    -dma   score DMA line deleted                M2
    -cvt   cvt loop bound -> dynamic 0
    -hist  hist0 loop bound -> dynamic 0
    -sweep zero_priv/fold_priv/scan/find -> 0
    -emit  emit loop bound -> dynamic 0
    -alld  all five, dynamic 0                   M1 + M3, loops entered
    -alls  all five, static 0                    M1 only, loops gone
    P3T    3 levels, `done` pinned True          skipped-level overhead
    R1     addr = bucket, fold over 1 replica    D5's gate, by correctness
    X2 X4  the base called 2x at b/2, 4x at b/4  D4: kernel launch slope
    N1 N2 N4   a trivial jitted add, 1/2/4x      D4: dispatch-only slope
    #null_a #null_b                              the floor. Two of them.

NOT a correctness experiment, with one exception. Every arm except `B`, `X*`
and `R1` computes the wrong answer by construction, and the printed PATCH
EVIDENCE is what proves each patch took. Two of those checks are EQUALITIES
that must hold rather than inequalities, which is the stronger form:

    P3T  MUST equal B exactly    -- pinning `done` True skips all refinement,
                                   so P3T's answer is the no-levels answer
    R1   MUST equal B exactly    -- iff the scatter resolves lane conflicts.
                                   If it does not, R1 undercounts and this is
                                   the whole of D5's answer, at zero cost.
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

SEED = 20260904  # same stream as probes 05 and 07
REPS_LO = 20
REPS_HI = 60

_LAST_ORDER: list[list[str]] = []


# --------------------------------------------------------------------------
# Timing harness -- byte-identical protocol to probe 07. Prefer the FROZEN
# instrument so the numbers compose with probes 03 and 07 directly; the inline
# copy is the same estimator and prints which one ran.
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
# Anchors are quoted with their exact leading whitespace and every `_sub` call
# asserts the occurrence count, so a source drift under this probe is a hard
# failure and never a silently-unpatched arm.
# --------------------------------------------------------------------------
SRC_LEVELS = "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))"
NO_LEVELS = "REFINEMENT_LEVELS = ()"

# The two zero bounds. `Z_DYN` keys off the fori_loop induction variable `w`,
# NOT off `i_hi`: `i_hi` is provably non-negative by interval analysis through
# `jnp.where(eff_len > 0, jnp.minimum(...), 0)`, so `i_hi < 0` is exactly the
# kind of predicate XLA folds. `w` arrives as a parameter of the loop body
# computation and carries no range with it. `-alls` uses a literal 0 so the
# loop is eliminated at trace time; the DIFFERENCE between the two arms is the
# `parallel_loop` entry cost, which is why both exist.
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

SRC_DONE = "        done = keep_all | (quota == cnt)"
# Pin TRUE: every refinement level is entered and every one of its bounds is
# DYNAMICALLY zero, which is precisely the "a skipped level is not free"
# question -- the incumbent's own skipped levels have dynamic-zero bounds too.
# `| jnp.ones((), jnp.bool_)` would be wrong here: XLA folds `x | True` to a
# constant, the bounds become STATIC zero, the loops are eliminated at trace
# time, and the arm would measure a skip the incumbent never performs. `w` is
# a parameter of the fori_loop body and carries no range with it.
PIN_TRUE = "        done = (keep_all | (quota == cnt)) | (w >= 0)"
SRC_NEWDONE = "            new_done = done | (quota2 == cnt2) | (quota2 == 0)"
PIN_NEWDONE = "            new_done = done"

# R1: replication factor 1. `addr` appears twice (hist0_body and ref_body) with
# identical text; both are replaced, and with REFINEMENT_LEVELS=() the second
# never traces. `fold_priv` then folds a single replica, and `zero_priv` only
# has 256 words to clear instead of 4096.
SRC_ADDR = "                addr = lane_iota * NUM_BUCKETS + bucket"
R1_ADDR = "                addr = bucket"
SRC_FOLDLOOP = "            for m in range(LANES):"
R1_FOLDLOOP = "            for m in range(1):"
R1_ZERO_PRIV = "            zero_priv(NUM_BUCKETS // LANES)"


def _sub(src: str, old: str, new: str, what: str, expect: int = 1) -> str:
    n = src.count(old)
    assert n == expect, f"patch {what}: expected {expect} occurrence(s), found {n}"
    return src.replace(old, new)


# Which passes each arm removes, and with which flavour of zero.
DROPS = {
    "B":      (),
    "-dma":   ("dma",),
    "-cvt":   ("cvt",),
    "-hist":  ("hist",),
    "-sweep": ("sweep",),
    "-emit":  ("emit",),
    "-alld":  ("dma", "cvt", "hist", "sweep", "emit"),
    "-alls":  ("dma", "cvt", "hist", "sweep", "emit"),
    "P3T":    (),
    "R1":     (),
}


def build_arm(base_src: str, name: str) -> types.ModuleType:
    src = base_src
    drops = DROPS[name]
    static = name == "-alls"
    zero = "0" if static else "z_dyn"

    if name in ("P3T",):
        # 3 levels, all skipped.
        src = _sub(src, SRC_DONE, PIN_TRUE, f"{name}/done")
        src = _sub(src, SRC_NEWDONE, PIN_NEWDONE, f"{name}/new_done")
    else:
        src = _sub(src, SRC_LEVELS, NO_LEVELS, f"{name}/levels")

    if name == "R1":
        src = _sub(src, SRC_ADDR, R1_ADDR, f"{name}/addr", expect=2)
        src = _sub(src, SRC_FOLDLOOP, R1_FOLDLOOP, f"{name}/foldloop")
        src = _sub(src, SRC_ZERO_PRIV, R1_ZERO_PRIV, f"{name}/zero_priv")

    if drops and not static:
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
    if "sweep" in drops:
        src = _sub(src, SRC_ZERO_PRIV,
                   f"            zero_priv({zero})", f"{name}/zero_priv")
        src = _sub(src, SRC_FOLD,
                   f"            fold_priv({zero})", f"{name}/fold")
        src = _sub(src, SRC_SCAN,
                   f"        total = scan_glob({zero})", f"{name}/scan")
        src = _sub(src, SRC_FIND,
                   f"        b_star = find_bucket({zero}, total - k)",
                   f"{name}/find")

    mod = types.ModuleType(f"sctopk08_{abs(hash(name))}")
    mod.__file__ = f"<probe08:{name}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


PATCH_ARMS = ["B", "-dma", "-cvt", "-hist", "-sweep", "-emit",
              "-alld", "-alls", "P3T", "R1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--ctx", type=int, default=9216)
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 64])
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--harness-dir", nargs="*", default=[])
    args = ap.parse_args()
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

    mods = {}
    for name in PATCH_ARMS:
        mods[name] = build_arm(base_src, name)
        safe = name.replace("-", "no")
        with open(os.path.join(args.outdir, f"arm-{safe}.py"), "w") as fh:
            fh.write(mods[name]._probe_src)
        print(f"  arm {name:6s} drops={','.join(DROPS[name]) or '-':30s} "
              f"src_md5={_md5(mods[name]._probe_src.encode())}", flush=True)

    n = ((args.ctx + 1024) + 15) // 16 * 16
    results = {
        "n": n, "k": args.k, "rounds": args.rounds, "subject_md5": _md5(raw),
        "harness": harness_kind, "seed": SEED, "shapes": {},
    }

    rng = np.random.default_rng(SEED)
    for b in args.batches:
        p = inc._pick_partition(b, n, args.k)
        waves = (b + 31) // 32
        print(f"\n=== b={b} n={n} k={args.k} P={p} waves={waves} "
              f"{'SINGLE-STAGE' if p == 1 else 'TWO-STAGE'} ===", flush=True)
        if p != 1:
            print("  skipped: single-stage path only.", flush=True)
            continue

        scores = jnp.asarray(rng.standard_normal((b, n), dtype=np.float32))
        lengths = jnp.full((b,), n, jnp.int32)

        thunks = {}
        for name in PATCH_ARMS:
            fn = mods[name].sparsecore_topk
            thunks[name] = (lambda fn=fn: fn(scores, k=args.k,
                                             row_lengths=lengths))

        # --- D4: launch-count slope, at CONSTANT total work. -----------------
        # j calls of b/j rows each. Total rows and total waves are identical
        # across X1/X2/X4, so the slope in j is the per-launch cost and nothing
        # else. N1/N2/N4 are the same slope on a trivial op that never enters
        # the SparseCore, so kernel-slope minus null-slope is the part of the
        # launch that is the SC program rather than host dispatch.
        base_fn = mods["B"].sparsecore_topk
        splits = {}
        for j in (2, 4):
            if b % (32 * j):
                continue
            bj = b // j
            sub_s = [scores[i * bj:(i + 1) * bj] for i in range(j)]
            sub_l = jnp.full((bj,), n, jnp.int32)
            splits[j] = (sub_s, sub_l)
            thunks[f"X{j}"] = (
                lambda ss=sub_s, ll=sub_l: [base_fn(x, k=args.k, row_lengths=ll)
                                            for x in ss])
        _nullf = jax.jit(lambda a: a + 1.0)
        _nullx = jnp.zeros((8, 128), jnp.float32)
        for j in (1, 2, 4):
            thunks[f"N{j}"] = (lambda j=j: [_nullf(_nullx) for _ in range(j)])
        thunks["#null_a"] = lambda: _nullf(_nullx)
        thunks["#null_b"] = lambda: _nullf(_nullx)

        # --- PATCH EVIDENCE, before timing. ---------------------------------
        # Two EQUALITIES that must hold and seven inequalities that must not.
        ref_b = np.asarray(thunks["B"]())
        ev = {}
        for name in PATCH_ARMS:
            got = np.asarray(thunks[name]())
            same = int((np.sort(got, axis=1) == np.sort(ref_b, axis=1))
                       .all(axis=1).sum())
            ev[name] = {"rows_matching_B": same, "rows": int(b)}
            print(f"  patch evidence {name:6s}: {same:5d}/{b} rows match B",
                  flush=True)

        # P3T pins `done` True from the start, so it skips all refinement and
        # must reproduce the no-levels answer exactly. Anything else means the
        # pin did not take and its "skip overhead" would contain real work.
        if ev["P3T"]["rows_matching_B"] != b:
            print("  !! P3T does not reproduce B -- `done` pin did not take. "
                  "Its number is NOT a skipped-level overhead.",
                  file=sys.stderr, flush=True)

        # R1 is D5's entire gate, and it is answered here rather than by any
        # timing below.
        r1_exact = ev["R1"]["rows_matching_B"] == b
        print(f"\n  *** D5 GATE: plsc.addupdate_scatter with 16 lanes "
              f"colliding on one address is "
              f"{'EXACT -- the 16x replication is REMOVABLE' if r1_exact else 'LOSSY -- the 16x replication is MANDATORY'}"
              f" ({ev['R1']['rows_matching_B']}/{b} rows) ***\n", flush=True)

        for name in ("-dma", "-cvt", "-hist", "-sweep", "-emit",
                     "-alld", "-alls"):
            if ev[name]["rows_matching_B"] == b:
                print(f"  !! {name} reproduces B exactly -- the pass was NOT "
                      f"removed. Its delta is not a cost.",
                      file=sys.stderr, flush=True)

        out = measure(thunks, rounds=args.rounds)
        for name in list(thunks):
            r = out[name]
            print(f"  {name:8s} us_median={r['us_median']:10.3f}  "
                  f"us={r['us']:10.3f}  hostbound={r['hostbound_pct']:5.1f}%  "
                  f"suspect={r['suspect']}", flush=True)

        m = {k2: out[k2]["us_median"] for k2 in out}
        null = 0.5 * (m["#null_a"] + m["#null_b"])
        null_spread = abs(m["#null_a"] - m["#null_b"])

        # Per-pass costs, each as B minus the arm that lacks it.
        passes = {p2: m["B"] - m[f"-{p2}"]
                  for p2 in ("dma", "cvt", "hist", "sweep", "emit")}
        sum_passes = sum(passes.values())
        residual_d = m["-alld"]      # everything removed, loops still entered
        residual_s = m["-alls"]      # everything removed, loops gone
        loop_entry = residual_d - residual_s
        additivity = (m["B"] - sum_passes) - residual_d

        # The launch decomposition.
        kslope = ((m["X2"] - m["B"]) if "X2" in m else float("nan"))
        kslope4 = (((m["X4"] - m["B"]) / 3.0) if "X4" in m else float("nan"))
        nslope = m["N2"] - m["N1"]
        nslope4 = (m["N4"] - m["N1"]) / 3.0

        print(f"\n  --- per-wave residual, b={b} ({waves} waves) ---",
              flush=True)
        print(f"  B (base, no levels)           {m['B']:10.3f}", flush=True)
        for p2, v in passes.items():
            print(f"    - {p2:22s}    {v:10.3f}   "
                  f"({v / waves:7.4f} us/wave)", flush=True)
        print(f"    sum of the five             {sum_passes:10.3f}",
              flush=True)
        print(f"  -alld (loops entered, bound 0){residual_d:10.3f}   "
              f"({residual_d / waves:7.4f} us/wave)", flush=True)
        print(f"  -alls (loops gone)            {residual_s:10.3f}   "
              f"({residual_s / waves:7.4f} us/wave)", flush=True)
        print(f"  parallel_loop entry (d - s)   {loop_entry:10.3f}   "
              f"<- M3", flush=True)
        print(f"  additivity check              {additivity:+10.3f}   "
              f"(B - sum - alld; 0 if the five are independent)", flush=True)
        print(f"  score DMA, total              {passes['dma']:10.3f}   "
              f"<- M2. hardware_map stage C prices its BANDWIDTH at 33.43; "
              f"any excess is exposed LATENCY", flush=True)
        print(f"\n  --- launch, b={b} ---", flush=True)
        print(f"  per extra kernel launch       {kslope:10.3f}  "
              f"(from X4: {kslope4:10.3f})", flush=True)
        print(f"  per extra trivial-op launch   {nslope:10.3f}  "
              f"(from N4: {nslope4:10.3f})", flush=True)
        print(f"  SC program load = k - n       {kslope - nslope:10.3f}   "
              f"<- D4", flush=True)
        print(f"\n  --- skipped levels, b={b} ---", flush=True)
        print(f"  P3T (3 levels, all skipped)   {m['P3T']:10.3f}", flush=True)
        print(f"  per skipped level             {(m['P3T'] - m['B']) / 3.0:10.3f}   "
              f"(probes/07 §4: ~3.8 would close the 2.231-vs-2.03 gap)",
              flush=True)
        print(f"\n  --- D5, b={b} ---", flush=True)
        print(f"  R1 (replication factor 1)     {m['R1']:10.3f}  "
              f"vs B {m['B']:10.3f}  -> "
              f"{100.0 * (m['R1'] - m['B']) / m['B']:+.2f}% "
              f"{'[EXACT]' if r1_exact else '[LOSSY -- timing is not a candidate]'}",
              flush=True)
        print(f"\n  null floor                    {null:10.3f}  "
              f"(two nulls differ by {null_spread:.3f} us)", flush=True)

        # Cross-JOB replication. `B` is byte-identical in construction to
        # probe 07's `L0` arm -- the same single substitution on the same
        # subject md5 -- so it is a free control on the instrument itself,
        # taken hours apart on a different pod. One drift null is a number,
        # not a floor; this is the second reading of the same quantity.
        p07 = {1024: 327.023, 64: 41.104}.get(b)
        if p07:
            print(f"  cross-job control: B={m['B']:.3f} vs probes/07 L0="
                  f"{p07:.3f}  -> {100.0 * (m['B'] - p07) / p07:+.2f}% "
                  f"(same arm, different job)", flush=True)
            results.setdefault("cross_job", {})[str(b)] = {
                "B": m["B"], "probe07_L0": p07,
                "delta_pct": 100.0 * (m["B"] - p07) / p07,
            }

        results["shapes"][str(b)] = {
            "b": b, "n": n, "k": args.k, "P": p, "waves": waves,
            "patch_evidence": ev, "raw": out, "order": last_order(),
            "derived": {
                "base_B": m["B"],
                "pass_costs": passes,
                "sum_of_passes": sum_passes,
                "residual_dynamic": residual_d,
                "residual_static": residual_s,
                "parallel_loop_entry": loop_entry,
                "additivity": additivity,
                "kernel_launch_slope": kslope,
                "kernel_launch_slope_x4": kslope4,
                "null_launch_slope": nslope,
                "null_launch_slope_n4": nslope4,
                "sc_program_load": kslope - nslope,
                "P3T": m["P3T"],
                "per_skipped_level": (m["P3T"] - m["B"]) / 3.0,
                "R1": m["R1"],
                "R1_exact": bool(r1_exact),
                "R1_delta_pct": 100.0 * (m["R1"] - m["B"]) / m["B"],
                "null_mean": null,
                "null_spread": null_spread,
            },
        }

    path = os.path.join(args.outdir, "per_wave_residual.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
