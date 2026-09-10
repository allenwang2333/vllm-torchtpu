"""Probe 07 -- split a refinement LEVEL into its fixed sweep and its `L`-walk.

Resolves report.md §5d' . The cost model fits ONE free parameter -- the price of
a refinement level -- and gets 114.3 us. Probe 03 measured `scatter_sparse` at
35.243 us over the same walk with bundle-identical code. 3.2x apart, ~150 us of
a 451 us kernel unattributed. The favoured explanation is that a *level* is not
a walk: it is `zero_priv(256)` + the walk + `fold_priv(16)` + `scan_glob(16)` +
`find_bucket(16)` + a serial scalar chain, and probe 03 priced only the walk.

The split is measured by removing exactly one thing.

    L0    REFINEMENT_LEVELS = ()          -- the kernel with no levels at all
    F1    1 level, `done` pinned False    -- base + 1 x (fixed + walk)
    F1n   1 level, pinned, walk forced 0  -- base + 1 x fixed
    F3    3 levels, pinned                -- base + 3 x (fixed + walk)
    F3n   3 levels, pinned, walk forced 0 -- base + 3 x fixed
    L3    unpatched                       -- the incumbent

`done` is pinned False in the F arms because the incumbent's level count is
data-dependent (probe 05: 2.03 critical-path levels at b=1024). Pinning makes
the count static, so `(F3 - F3n)/3` is a walk price and `(F3n - L0)/3` is a
fixed price, with no distribution assumption in either. F1/F1n are the same two
quantities at a second point, so the level count enters as a SLOPE rather than
as a single difference -- a per-kernel artifact that does not scale with the
level count shows up as a broken slope instead of contaminating the price.

L3 is then predicted from those prices and probe 05's 2.03, which is the
out-of-sample test of the whole decomposition.

Patching is textual against the installed module source, every substitution
asserted to hit exactly once, and each arm exec'd into its own module object so
jax.jit caches cannot alias. The patched source of every arm is written out
next to the results.

Not a correctness experiment: the F*/L0 arms compute the wrong answer by
construction. Two output comparisons are printed as PATCH EVIDENCE only --
`F3n` must NOT match the incumbent (if it does, the walk was not removed) and
`L0` must not either.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

SEED = 20260904  # same stream as probe 05, so its 2.03 applies to these rows
REPS_LO = 20
REPS_HI = 60

# --------------------------------------------------------------------------
# Timing harness.
#
# Preference order: the frozen instrument `glm_dsa_block_bench.measure`, so the
# protocol is literally the one report.md's numbers were taken with. If it is
# not importable on the pod, fall back to the inline copy below, which is the
# same protocol -- warm every thunk, per-round reshuffle, PAIRED lo/hi slope,
# report the median slope. Which one ran is printed.
#
# Every conclusion here is a difference between arms measured inside one
# `measure` call, so the fallback cannot move a conclusion; it can only move
# the absolutes, and the absolutes are cross-checked against report.md §7.
# --------------------------------------------------------------------------
_LAST_ORDER: list[list[str]] = []


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
                  f"(md5 {_md5(open(cand,'rb').read())})", flush=True)
            return mod.measure, (lambda: mod.LAST_ORDER), "frozen"
        except Exception as exc:  # pragma: no cover - pod-side path variance
            print(f"  harness import from {d} failed: {exc!r}", flush=True)
    print("  timing harness: INLINE COPY (frozen instrument not importable)",
          flush=True)
    return _measure_inline, (lambda: _LAST_ORDER), "inline"


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# --------------------------------------------------------------------------
# Arm construction by textual patch of the installed module.
# --------------------------------------------------------------------------
SRC_LEVELS = "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))"
SRC_DONE = "        done = keep_all | (quota == cnt)"
SRC_NEWDONE = "            new_done = done | (quota2 == cnt2) | (quota2 == 0)"
SRC_WALK = "            active_walk = jnp.where(done, 0, walk_chunks)"

# `jnp.zeros((), jnp.bool_)` rather than a Python `False`: it keeps `done` an
# array so every `jnp.where(done, ...)` downstream still traces, while folding
# to a constant so the bounds it gates become static-full. A Python False would
# also work but would change the traced types, which is a second difference.
PIN_DONE = "        done = (keep_all | (quota == cnt)) & jnp.zeros((), jnp.bool_)"
PIN_NEWDONE = "            new_done = done"
NOWALK = "            active_walk = jnp.where(done, 0, 0)"

LEVELS_N = {
    0: "REFINEMENT_LEVELS = ()",
    1: "REFINEMENT_LEVELS = ((16, 0xFF), )",
    2: "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF))",
    3: SRC_LEVELS,
}


def _sub(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    assert n == 1, f"patch {what}: expected 1 occurrence, found {n}"
    return src.replace(old, new)


def build_arm(base_src: str, name: str, levels: int, pin: bool,
              nowalk: bool) -> types.ModuleType:
    src = base_src
    if levels != 3:
        src = _sub(src, SRC_LEVELS, LEVELS_N[levels], f"{name}/levels")
    if pin:
        src = _sub(src, SRC_DONE, PIN_DONE, f"{name}/done")
        src = _sub(src, SRC_NEWDONE, PIN_NEWDONE, f"{name}/new_done")
    if nowalk:
        src = _sub(src, SRC_WALK, NOWALK, f"{name}/walk")
    mod = types.ModuleType(f"sctopk_arm_{name}")
    mod.__file__ = f"<probe07:{name}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


ARMS = [
    # name  levels  pin    nowalk  what it isolates
    ("L0", 0, False, False),
    ("F1n", 1, True, True),
    ("F1", 1, True, False),
    ("F3n", 3, True, True),
    ("F3", 3, True, False),
    ("L3", 3, False, False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--ctx", type=int, default=9216,
                    help="max-model-len; n = ctx + 1024 rounded to LANES")
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
    base_src = open(base_path, "rb").read()
    print(f"subject: {base_path}  md5={_md5(base_src)}", flush=True)
    base_src = base_src.decode()

    measure, last_order, harness_kind = _load_harness(
        list(args.harness_dir) + [
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/microbench",
            "/app/microbench",
            os.getcwd(),
        ])

    mods = {}
    for name, levels, pin, nowalk in ARMS:
        mods[name] = build_arm(base_src, name, levels, pin, nowalk)
        with open(os.path.join(args.outdir, f"arm-{name}.py"), "w") as fh:
            fh.write(mods[name]._probe_src)
        print(f"  arm {name:4s} levels={levels} pin={int(pin)} "
              f"nowalk={int(nowalk)} src_md5="
              f"{_md5(mods[name]._probe_src.encode())}", flush=True)

    # n is derived the way the shipping config derives it (report §4), not
    # typed in: --max-model-len 9216 -> n = 10240.
    n = ((args.ctx + 1024) + 15) // 16 * 16
    results = {
        "n": n, "k": args.k, "rounds": args.rounds,
        "subject_md5": _md5(open(base_path, "rb").read()),
        "harness": harness_kind, "seed": SEED, "shapes": {},
    }

    rng = np.random.default_rng(SEED)
    for b in args.batches:
        p = inc._pick_partition(b, n, args.k)
        waves = (b + 31) // 32
        print(f"\n=== b={b} n={n} k={args.k} P={p} waves={waves} "
              f"{'SINGLE-STAGE' if p == 1 else 'TWO-STAGE'} ===", flush=True)
        if p != 1:
            print("  skipped: this probe is about the single-stage path; a "
                  "two-stage shape would mix stage-1 and stage-2 levels in "
                  "one number.", flush=True)
            continue

        scores = jnp.asarray(rng.standard_normal((b, n), dtype=np.float32))
        lengths = jnp.full((b,), n, jnp.int32)

        thunks = {}
        for name, *_ in ARMS:
            # `sparsecore_topk` is already `jax.jit`-wrapped with `k` static;
            # each arm is a distinct module object, so each carries its own
            # cache and no arm can be served another's compiled executable.
            fn = mods[name].sparsecore_topk
            thunks[name] = (lambda fn=fn: fn(scores, k=args.k,
                                             row_lengths=lengths))
        # Null: the dispatch floor, measured through the same estimator so the
        # arm-to-arm resolution has a stated noise floor rather than an assumed
        # one. Two of them -- one null is a number, not a floor.
        _nullf = jax.jit(lambda a: a + 1.0)
        _nullx = jnp.zeros((8, 128), jnp.float32)
        thunks["#null_a"] = lambda: _nullf(_nullx)
        thunks["#null_b"] = lambda: _nullf(_nullx)

        # PATCH EVIDENCE, before timing: the nowalk arms must NOT reproduce the
        # incumbent's output. If one does, its `active_walk` patch did not take
        # and its "fixed cost" would silently include the walk.
        ref = np.asarray(thunks["L3"]())
        ev = {}
        for name in ("L0", "F1n", "F3n", "F1", "F3"):
            got = np.asarray(thunks[name]())
            same_rows = int(
                (np.sort(got, axis=1) == np.sort(ref, axis=1)).all(axis=1).sum())
            ev[name] = {"rows_matching_incumbent": same_rows, "rows": int(b)}
            print(f"  patch evidence {name:4s}: {same_rows}/{b} rows match "
                  f"the incumbent", flush=True)
        for name in ("F1n", "F3n"):
            if ev[name]["rows_matching_incumbent"] == b:
                print(f"  !! {name} reproduces the incumbent exactly -- the "
                      f"walk was NOT removed. Its number is not a fixed "
                      f"cost.", file=sys.stderr, flush=True)

        out = measure(thunks, rounds=args.rounds)
        for name in list(thunks):
            r = out[name]
            print(f"  {name:8s} us_median={r['us_median']:10.3f}  "
                  f"us={r['us']:10.3f}  hostbound={r['hostbound_pct']:5.1f}%  "
                  f"suspect={r['suspect']}", flush=True)

        m = {k2: out[k2]["us_median"] for k2 in out}
        fixed1 = m["F1n"] - m["L0"]
        walk1 = m["F1"] - m["F1n"]
        fixed3 = (m["F3n"] - m["L0"]) / 3.0
        walk3 = (m["F3"] - m["F3n"]) / 3.0
        # Slope over level count -- immune to any per-kernel constant.
        fixed_slope = (m["F3n"] - m["F1n"]) / 2.0
        walk_slope = ((m["F3"] - m["F3n"]) - (m["F1"] - m["F1n"])) / 2.0
        level1 = m["F1"] - m["L0"]
        level_slope = (m["F3"] - m["F1"]) / 2.0
        null = 0.5 * (m["#null_a"] + m["#null_b"])
        null_spread = abs(m["#null_a"] - m["#null_b"])

        eff_levels = ((m["L3"] - m["L0"]) / level_slope
                      if level_slope > 0 else float("nan"))
        pred_L3 = m["L0"] + 2.03 * level_slope

        print(f"\n  --- decomposition, b={b} (us, us_median) ---", flush=True)
        print(f"  base (L0, no levels)          {m['L0']:10.3f}", flush=True)
        print(f"  fixed sweep / level  @1 lvl   {fixed1:10.3f}", flush=True)
        print(f"                       @3 lvl   {fixed3:10.3f}", flush=True)
        print(f"                       slope    {fixed_slope:10.3f}",
              flush=True)
        print(f"  L-walk / level       @1 lvl   {walk1:10.3f}", flush=True)
        print(f"                       @3 lvl   {walk3:10.3f}", flush=True)
        print(f"                       slope    {walk_slope:10.3f}",
              flush=True)
        print(f"  whole level          @1 lvl   {level1:10.3f}", flush=True)
        print(f"                       slope    {level_slope:10.3f}",
              flush=True)
        print(f"  fixed share of a level        "
              f"{100.0 * fixed_slope / level_slope:9.1f}%"
              if level_slope > 0 else "", flush=True)
        print(f"  incumbent L3                  {m['L3']:10.3f}", flush=True)
        print(f"  implied active levels         {eff_levels:10.3f}  "
              f"(probe 05 critical path: 2.03 at b=1024, 2.50 at b=64)",
              flush=True)
        print(f"  L3 predicted from 2.03        {pred_L3:10.3f}  "
              f"({100.0 * (pred_L3 - m['L3']) / m['L3']:+.2f}% vs measured)",
              flush=True)
        print(f"  null floor                    {null:10.3f}  "
              f"(two nulls differ by {null_spread:.3f} us)", flush=True)

        results["shapes"][str(b)] = {
            "b": b, "n": n, "k": args.k, "P": p, "waves": waves,
            "patch_evidence": ev,
            "raw": out,
            "order": last_order(),
            "derived": {
                "base_L0": m["L0"],
                "fixed_per_level_at1": fixed1,
                "fixed_per_level_at3": fixed3,
                "fixed_per_level_slope": fixed_slope,
                "walk_per_level_at1": walk1,
                "walk_per_level_at3": walk3,
                "walk_per_level_slope": walk_slope,
                "level_at1": level1,
                "level_slope": level_slope,
                "fixed_share_of_level_pct": (
                    100.0 * fixed_slope / level_slope
                    if level_slope > 0 else None),
                "incumbent_L3": m["L3"],
                "implied_active_levels": eff_levels,
                "L3_pred_from_2p03": pred_L3,
                "null_mean": null,
                "null_spread": null_spread,
            },
        }

    path = os.path.join(args.outdir, "refinement_split.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
