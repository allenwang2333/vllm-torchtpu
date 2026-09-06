"""Time the composed candidate against the incumbent, adjacently, in one job.

  python3 run_bench.py --rows B-D1-1M,B-D4-256k --level sc,full --repeats 20

This is step 4's headline measurement and it is deliberately not a sum of the
step-3 numbers. Every arm below is timed inside the SAME process against the
SAME incumbent, and the delta is formed within each repeat.

`instruments/bench.py` supplies the statistic and is imported, never copied and
never edited. `interleaved` times every arm back to back inside each repeat and
differences within the repeat, because run-order drift on this block is 3-9% --
larger than most of the effects being chased. Nothing here forms a delta
between two separately measured numbers, and in particular nothing here
computes `full - sc`: that subtraction moved 2.6x across three measurements of
one shape and is not a quantity this harness can produce.

`interleaved` calls ZERO-ARGUMENT thunks. Every arm is bound with
`functools.partial`; passing a function plus its arguments silently times
something else.

Arms. `final` is the shipping candidate, `inc` the shipping op, and each
`no-*` is `final` with exactly one mechanism removed -- leave-one-out, so a
shortfall attributes to a mechanism rather than to the composition as a whole:

  inc       the shipping kernel, imported and untouched.
  inc_copy  this candidate with every option OFF. Times the transcription
            itself: a gap between `inc` and `inc_copy` is copy drift and
            invalidates every other arm in the job.
  final     everything on. What would be merged.
  no-o8     stage 1 emits indices only; the [B, N] seam gather comes back.
  no-o3     the wave packing off; P pinned to the static ceiling.
  no-o7     the lane-private histogram instead of `scan_count`.
  no-o1     the growth guard, the empty-launch skip and the fill elision off.
  no-o2     the caller's KV tile instead of the regime-conditional one.
            Scoring-side, so it is a `full`-level arm only.

Two levels, because they answer different questions:
  sc    the selector alone on a synthetic score matrix -- where four of the
        five options live, and the only level at which they are not diluted.
  full  the whole op. Says how much survives the scoring pass. It is the
        level the end-to-end benchmark would see, and the one the report
        leads with.
"""
import argparse
import dataclasses
import functools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import api  # noqa: E402
import pallas_mosaic_tpu as policy  # noqa: E402
import tiling  # noqa: E402
from run_correctness import ARMS as ARM_CFG  # noqa: E402
from run_correctness import check_incumbent_sha  # noqa: E402

from bench import interleaved, synth_scores  # noqa: E402
from shapes import K, ROWS, build, padded_n  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    sparsecore_topk as inc_select)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    streamindex_topk as inc_full)

SC_ARMS = ("inc", "inc_copy", "final", "no-o8", "no-o3", "no-o7", "no-o1")
FULL_ARMS = SC_ARMS + ("no-o2", )


def paired(out, ref):
    """Re-difference the per-repeat medians `interleaved` returned, vs `ref`.

    Exactly the arithmetic `interleaved` does internally: the delta is formed
    inside each repeat, then medianed, with the spread of the per-repeat
    deltas beside it. Never a ratio of two medians.
    """
    base = np.array(out[ref]["us"])
    res = {}
    for name, rec in out.items():
        if name == ref:
            continue
        v = np.array(rec["us"])
        d = 100.0 * (v - base) / base
        res[name] = (float(np.median(d)), float(d.max() - d.min()))
    return res


def report(tag, out, refs=("inc", "inc_copy")):
    print(f"\n  -- {tag}: paired d time % (negative is faster) --", flush=True)
    print(f"  {'arm':10} {'median us':>10} " + "  ".join(
        f"{'vs ' + r:>14} {'spread':>9}" for r in refs), flush=True)
    rows = {}
    tables = {r: paired(out, r) for r in refs if r in out}
    for name in out:            # arm order, as `interleaved` was given them
        cells, rec = [], {}
        for r in refs:
            if r not in tables or name not in tables[r]:
                cells.append(f"{'--':>14} {'--':>9}")
                continue
            d, s = tables[r][name]
            verdict = "" if abs(d) > s else "!"
            cells.append(f"{d:>+13.1f}%{verdict:1} {s:8.1f}pt")
            rec[r] = dict(delta_pct=d, spread_pt=s,
                          resolved=bool(abs(d) > s))
        med = float(np.median(out[name]["us"]))
        print(f"  {name:10} {med:10.1f} " + "  ".join(cells), flush=True)
        rows[name] = dict(median_us=med, **rec)
    print("  (! marks |delta| <= its own spread -- UNRESOLVED, not a result)",
          flush=True)
    return rows


def live_lens(meta):
    """Per-token-row effective lengths, as the shared bench derives them."""
    qlen, lens = meta.get("qlen", 1), meta["lens"]
    out = np.zeros(meta["T"], dtype=np.int32)
    t = 0
    for i in range(meta["R"]):
        for u in range(qlen):
            out[t] = min(int(lens[i]) - qlen + u + 1, meta["n"])
            t += 1
    return out


def geometry(b, n, lens):
    """What each arm's selector will actually run at, printed before it runs.

    Reported because two of the five options are INERT at some shapes by
    construction, and an inert option must be read as structural rather than
    as a measured no-op. `p_inc == 1` means the two-stage path is not taken at
    all, so `no-o8` and `no-o3` cannot differ from `final` there whatever the
    clock says.
    """
    live = int(max(lens)) if len(lens) else n
    g_inc = policy.select_geometry(b, n, K, fix_growth=False)
    g_fin = policy.select_geometry(b, n, K, fix_growth=True)
    lad = g_fin.ladder
    rung = tiling.ladder_index(live, g_fin.tn, lad)
    return dict(p_inc=g_inc.p_max, p_max=g_fin.p_max, tn=g_fin.tn,
                ladder=list(lad), p_run=lad[rung], live_max=live,
                C=lad[rung] * min(K, g_fin.tn))


def bench_sc(meta, repeats, iters):
    lens = live_lens(meta)
    sc = synth_scores(meta["T"], meta["n"], lens)
    jax.block_until_ready(sc)
    rl = jnp.asarray(lens, jnp.int32)

    arms = []
    for nm in SC_ARMS:
        if nm == "inc":
            f = functools.partial(inc_select, sc, K, rl)
        else:
            f = functools.partial(api.sparsecore_topk, sc, K, rl,
                                  cfg=ARM_CFG[nm])
        arms.append((nm, f))

    got = {nm: np.asarray(fn()) for nm, fn in arms}
    ref = got["inc"]
    for nm in got:
        if nm == "inc":
            continue
        sd = sum(len(set(ref[t][ref[t] >= 0].tolist())
                     ^ set(got[nm][t][got[nm][t] >= 0].tolist()))
                 for t in range(meta["T"]))
        print(f"  setdiff {nm:9} vs inc: {sd}", flush=True)
    out = interleaved(arms, repeats=repeats, iters=iters, ref="inc")
    del sc, rl, arms, got, ref
    return out


def bench_full(name, repeats, iters):
    args, kwargs, meta = build(name)
    jax.block_until_ready(args[2])
    arms = []
    for nm in FULL_ARMS:
        # Both sides are already `jax.jit` objects with the same static
        # argnames, and neither gets an extra wrapper: an asymmetric one would
        # put a Python dispatch on one arm of a paired difference.
        f = (inc_full if nm == "inc"
             else functools.partial(api.streamindex_topk, cfg=ARM_CFG[nm]))
        arms.append((nm, functools.partial(f, *args, **kwargs)))

    got = {nm: np.asarray(fn()) for nm, fn in arms}
    if bool((got["inc"][0] == np.arange(K)).all()):
        raise SystemExit(f"{name}: took _fast_path -- measures nothing")
    ref = got["inc"]
    T = ref.shape[0]
    for nm in got:
        if nm == "inc":
            continue
        sd = sum(len(set(ref[t][ref[t] >= 0].tolist())
                     ^ set(got[nm][t][got[nm][t] >= 0].tolist()))
                 for t in range(T))
        print(f"  setdiff {nm:9} vs inc: {sd}", flush=True)
    out = interleaved(arms, repeats=repeats, iters=iters, ref="inc")
    del args, arms, got, ref
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--level", default="sc,full")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    levels = [x.strip() for x in a.level.split(",")]
    results = {}

    check_incumbent_sha()
    for nm in FULL_ARMS:
        if ARM_CFG[nm] is not None:
            print(f"  arm {nm:9} "
                  + " ".join(f"{f.name}={int(getattr(ARM_CFG[nm], f.name))}"
                             for f in dataclasses.fields(ARM_CFG[nm])),
                  flush=True)

    def flush_json():
        # Written after every row, not at the end: a row that OOMs must not
        # take the rows already measured down with it.
        if a.json:
            with open(a.json, "w") as f:
                json.dump(results, f, indent=2, default=float)

    for name in [r.strip() for r in a.rows.split(",")]:
        r = ROWS[name]
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}", flush=True)
        if "full" in levels:
            t0 = time.perf_counter()
            out, meta = bench_full(name, a.repeats, a.iters)
            g = geometry(meta["T"], meta["n"], meta["lens"])
            print(f"  [full] built+ran in {time.perf_counter() - t0:.0f}s  "
                  f"B={meta['T']} N={meta['n']} {g}", flush=True)
            results[f"{name}/full"] = dict(
                geom=g, arms=out, table=report(f"{name} full", out))
            flush_json()
        if "sc" in levels:
            lens = np.full(r["R"], r["seqlen"], dtype=np.int64)
            if r.get("seqlens_override") == "ragged":
                lens = np.full(r["R"], 4096, dtype=np.int64)
                lens[::4] = r["seqlen"]
            meta = dict(T=r["T"], R=r["R"], lens=lens,
                        n=padded_n(r["psz"], r["pages_per_seq"]),
                        qlen=r.get("qlen", 1))
            g = geometry(meta["T"], meta["n"], live_lens(meta))
            note = ("   p_inc==1: the two-stage path is NOT taken; no-o8 and "
                    "no-o3 are STRUCTURALLY identical to final here"
                    if g["p_inc"] == 1 else "")
            print(f"\n  [sc] B={meta['T']} N={meta['n']} {g}{note}",
                  flush=True)
            out = bench_sc(meta, a.repeats, a.iters)
            results[f"{name}/sc"] = dict(
                geom=g, arms=out, table=report(f"{name} sc", out))
            flush_json()

    flush_json()
    if a.json:
        print(f"\nwrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
