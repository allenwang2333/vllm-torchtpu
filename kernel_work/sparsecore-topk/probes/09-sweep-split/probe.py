"""Probe 09 -- split the L0 sweep into its four components, and clean the residual.

A tightly-specified derivative of probes/08. Same subject, same instrument,
same interleaved-rounds harness, same `patch evidence` gate, same `us_median`
reporting. Only the arm set and the report blocks change.

Two questions.

(1) THE SWEEP SPLIT. probes/08's `-sweep` arm zeroed all four L0-sweep bounds
    at once -- `zero_priv`, `fold_priv`, `scan_glob`, `find_bucket` -- so it
    priced the group and not the members. Here each of the four is dropped
    independently on top of a common base `E0`, plus `E0-swp` which drops all
    four jointly. The sum of the four singles against the exact joint is the
    additivity check; a ratio far from 1 means the four share a cost that no
    single one of them owns.

(2) WHY `fill` MUST BE OFF IN THE SPLIT ARMS. `fill_bound` is `k//LANES = 128`
    whenever `total < k`. Forcing `scan_glob`'s bound to zero makes `total = 0`,
    so `-scan` and `-alld` run a 128-iteration-per-wave fill loop that the base
    arm `B` never runs -- in `B`, `total = 10240 > k`, so `fill_bound = 0`.
    probes/08's `-alld` is contaminated by exactly this. `A0` and the `E0-*`
    family exist to remove it. Keeping the uncorrected `-alld` as well is what
    lets the size of the contamination be read off directly as
    `-alld` - (`A0` + outdma).

Arms (14). All patched textually from the subject, every substitution asserted
to hit an exact count, each arm exec'd into its own module object so no jax.jit
cache can alias. Every arm's patched source is written to arms/.

    B        nothing                              the control
    -out     output DMA gated off                 prices the out DMA alone
    E0       emit off, fill off                   the base for the sweep split
    E0-zero  E0 + zero_priv   bound -> dynamic 0
    E0-fold  E0 + fold_priv   bound -> dynamic 0
    E0-scan  E0 + scan_glob   bound -> dynamic 0
    E0-find  E0 + find_bucket bound -> dynamic 0
    E0-swp   E0 + all four of the above
    -alld    probes/08's -alld, byte-for-byte      CROSS-JOB control
    A0       -alld + fill off + outdma off        the clean residual
    A0_W8    A0, wave loop bound literal 8        wave slope
    A0_W16   A0, wave loop bound literal 16       wave slope
    #null_a #null_b                               the floor. Two of them.

Every arm keeps `REFINEMENT_LEVELS = ()`. There is no `P3T`-style arm here, and
no `X`/`N` launch arms -- probes/08 closed that.

NOT a correctness experiment. Every arm except `B` computes the wrong answer by
construction, and the printed PATCH EVIDENCE is what proves each patch took.
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

SEED = 20260904  # same stream as probes 05, 07 and 08
REPS_LO = 20
REPS_HI = 60

_LAST_ORDER: list[list[str]] = []


# --------------------------------------------------------------------------
# Timing harness -- byte-identical protocol to probes 07 and 08. Prefer the
# FROZEN instrument so the numbers compose with probes 03, 07 and 08 directly;
# the inline copy is the same estimator and prints which one ran.
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

# The three new anchors.
#
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


# Which passes each arm removes. `sweep` is kept as a single group name ONLY so
# that `-alld` is byte-for-byte identical to probes/08's `-alld`; the split arms
# name the four components independently.
DROPS = {
    "B":       (),
    "-out":    ("outdma",),
    "E0":      ("emit", "fill"),
    "E0-zero": ("emit", "fill", "zero"),
    "E0-fold": ("emit", "fill", "fold"),
    "E0-scan": ("emit", "fill", "scan"),
    "E0-find": ("emit", "fill", "find"),
    "E0-swp":  ("emit", "fill", "zero", "fold", "scan", "find"),
    "-alld":   ("dma", "cvt", "hist", "sweep", "emit"),
    "A0":      ("dma", "cvt", "hist", "sweep", "emit", "fill", "outdma"),
    "A0_W8":   ("dma", "cvt", "hist", "sweep", "emit", "fill", "outdma"),
    "A0_W16":  ("dma", "cvt", "hist", "sweep", "emit", "fill", "outdma"),
}

# Wave-loop bound override, for the slope arms only.
WAVES = {"A0_W8": 8, "A0_W16": 16}

# Drops that need the `z_dyn` binding injected. `fill` and `outdma` splice
# `& (w < 0)` in directly and do not.
Z_DYN_DROPS = frozenset(
    ("cvt", "hist", "emit", "sweep", "zero", "fold", "scan", "find"))


def _patch_src(base_src: str, name: str) -> str:
    """The single source of truth for what each arm patches."""
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


def build_arm(base_src: str, name: str) -> types.ModuleType:
    src = _patch_src(base_src, name)
    mod = types.ModuleType(f"sctopk09_{abs(hash(name))}")
    mod.__file__ = f"<probe09:{name}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


PATCH_ARMS = ["B", "-out", "E0", "E0-zero", "E0-fold", "E0-scan", "E0-find",
              "E0-swp", "-alld", "A0", "A0_W8", "A0_W16"]

# probes/08, job j-300d4690-d53e-441e-9d39, us_median.
P08 = {1024: {"B": 327.166, "-alld": 101.584},
       64: {"B": 41.005, "-alld": 26.344}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--ctx", type=int, default=9216)
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 64])
    ap.add_argument("--k", type=int, default=2048)
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

    mods = {}
    for name in PATCH_ARMS:
        mods[name] = build_arm(base_src, name)
        safe = name.replace("-", "no")
        with open(os.path.join(args.outdir, f"arm-{safe}.py"), "w") as fh:
            fh.write(mods[name]._probe_src)
        print(f"  arm {name:8s} drops={','.join(DROPS[name]) or '-':45s} "
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
        _nullf = jax.jit(lambda a: a + 1.0)
        _nullx = jnp.zeros((8, 128), jnp.float32)
        thunks["#null_a"] = lambda: _nullf(_nullx)
        thunks["#null_b"] = lambda: _nullf(_nullx)

        # --- PATCH EVIDENCE, before timing. ---------------------------------
        ref_b = np.asarray(thunks["B"]())
        ev = {}
        for name in PATCH_ARMS:
            got = np.asarray(thunks[name]())
            same = int((np.sort(got, axis=1) == np.sort(ref_b, axis=1))
                       .all(axis=1).sum())
            ev[name] = {"rows_matching_B": same, "rows": int(b)}
            print(f"  patch evidence {name:8s}: {same:5d}/{b} rows match B",
                  flush=True)

        for name in PATCH_ARMS:
            if name == "B":
                continue
            if ev[name]["rows_matching_B"] == b:
                print(f"  !! {name} reproduces B exactly -- the patch was NOT "
                      f"effective. Its delta is not a cost.",
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

        # --- the sweep split -------------------------------------------------
        comps = {"zero_priv": m["E0"] - m["E0-zero"],
                 "fold_priv": m["E0"] - m["E0-fold"],
                 "scan_glob": m["E0"] - m["E0-scan"],
                 "find_bucket": m["E0"] - m["E0-find"]}
        sum_comps = sum(comps.values())
        joint = m["E0"] - m["E0-swp"]
        add_ratio = (sum_comps / joint) if joint else float("nan")

        print(f"\n  --- sweep split, b={b} ---", flush=True)
        print(f"  E0 (emit+fill off)              {m['E0']:10.3f}", flush=True)
        for cname, v in comps.items():
            print(f"    - {cname:26s}  {v:10.3f}", flush=True)
        print(f"    sum of the four               {sum_comps:10.3f}",
              flush=True)
        print(f"    E0 - E0-swp (joint, exact)    {joint:10.3f}", flush=True)
        print(f"    additivity ratio              {add_ratio:10.3f}",
              flush=True)

        # --- the residual ----------------------------------------------------
        contamination = m["-alld"] - m["A0"]
        outdma_alone = m["B"] - m["-out"]

        print(f"\n  --- residual, b={b} ---", flush=True)
        print(f"  -alld (probe 08 control)        {m['-alld']:10.3f}   "
              f"probes/08: 101.584 @b=1024, 26.344 @b=64", flush=True)
        print(f"  A0 (fill + outdma also off)     {m['A0']:10.3f}", flush=True)
        print(f"    fill+outdma contamination     {contamination:10.3f}",
              flush=True)
        print(f"    output DMA alone (B - -out)   {outdma_alone:10.3f}",
              flush=True)
        print(f"    A0 - null                     {m['A0'] - null:10.3f}",
              flush=True)

        # --- wave slope on A0 ------------------------------------------------
        # At b=1024 waves=32, so the denominator is exactly (32-8)=24 and the
        # intercept is A0 - 32*slope, as specified.
        denom = waves - 8
        slope = (m["A0"] - m["A0_W8"]) / denom if denom else float("nan")
        intercept = m["A0"] - waves * slope

        print(f"\n  --- wave slope on A0, b={b} ---", flush=True)
        print(f"  A0_W8 / A0_W16 / A0             {m['A0_W8']:10.3f} "
              f"{m['A0_W16']:10.3f} {m['A0']:10.3f}", flush=True)
        lbl = f"per-wave slope ({waves}-8)"
        print(f"    {lbl:30s}{slope:10.3f}", flush=True)
        print(f"    intercept                     {intercept:10.3f}",
              flush=True)

        print(f"\n  null floor                      {null:10.3f}  "
              f"(two nulls differ by {null_spread:.3f} us)", flush=True)

        # Cross-JOB control. `B` and `-alld` are byte-identical in construction
        # to probes/08's, on the same subject md5, so they are a free control on
        # the instrument itself, taken on a different pod at a different time.
        ref = P08.get(b)
        if ref:
            db = 100.0 * (m["B"] - ref["B"]) / ref["B"]
            da = 100.0 * (m["-alld"] - ref["-alld"]) / ref["-alld"]
            print(f"  cross-job control: B={m['B']:.3f} vs probes/08 "
                  f"{ref['B']:.3f} -> {db:+.2f}%  |  "
                  f"-alld={m['-alld']:.3f} vs probes/08 {ref['-alld']:.3f} -> "
                  f"{da:+.2f}%", flush=True)
            results.setdefault("cross_job", {})[str(b)] = {
                "B": m["B"], "probe08_B": ref["B"], "B_delta_pct": db,
                "alld": m["-alld"], "probe08_alld": ref["-alld"],
                "alld_delta_pct": da,
            }

        results["shapes"][str(b)] = {
            "b": b, "n": n, "k": args.k, "P": p, "waves": waves,
            "patch_evidence": ev, "raw": out, "order": last_order(),
            "derived": {
                "base_B": m["B"],
                "E0": m["E0"],
                "sweep_components": comps,
                "sum_of_components": sum_comps,
                "joint_E0_minus_E0swp": joint,
                "additivity_ratio": add_ratio,
                "alld": m["-alld"],
                "A0": m["A0"],
                "fill_outdma_contamination": contamination,
                "outdma_alone": outdma_alone,
                "A0_minus_null": m["A0"] - null,
                "A0_W8": m["A0_W8"],
                "A0_W16": m["A0_W16"],
                "per_wave_slope": slope,
                "intercept": intercept,
                "null_mean": null,
                "null_spread": null_spread,
            },
        }

    path = os.path.join(args.outdir, "sweep_split.json")
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

    def patched(name):
        return _patch_src(src, name)

    import difflib
    base = patched("B")
    for name in PATCH_ARMS:
        s = patched(name)
        compile(s, f"<probe09:{name}>", "exec")
        d = list(difflib.unified_diff(base.splitlines(), s.splitlines(),
                                      n=0, lineterm=""))
        changed = sum(1 for ln in d
                      if (ln.startswith("+") or ln.startswith("-"))
                      and not ln.startswith(("+++", "---")))
        print(f"  arm {name:8s} compiles OK  changed_lines_vs_B={changed:2d}  "
              f"src_md5={_md5(s.encode())}")
    print("COMPILE_GATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
