"""Time O1 against the shipping kernel, six arms adjacently, in one process.

  python3 run_bench.py --rows A-D16-8k,A-D64-8k --json /tmp/o1.json

Six arms per shape, all built from the SAME input arrays and timed back to back
inside every repeat, with the shipping kernel as the reference:

  base      the incumbent, `vllm_torchtpu...streamindex_topk`
  copy-off  this variant with all three switches OFF -- copy fidelity. Its
            delta must be indistinguishable from zero. If it is not, the copy
            diverged from the shipping kernel somewhere and no other arm here
            is interpretable.
  pick      `fix_growth` alone
  cond      `skip_empty_launches` alone
  fill      `elide_score_prefill` alone
  o1        all three

Why one job and not six. Run-order drift on this block is 3-9% -- larger than
two of the three effects being measured -- so arms in different jobs, or in
different phases of one job, cannot resolve them. `bench.interleaved` times the
arms microseconds apart and differences them INSIDE each repeat, which makes
the drift common-mode. `base`, `fill` and `o1` in one repeat is also exactly
the three-arm paired set the design asks for on the fill, whose modelled effect
(1.7-2.3%) sits under the drift.

Read `d spread` before `paired d time %`. An effect smaller than its own spread
is not a result, and this file will say so rather than round it away.

Nothing in `../instruments/` is imported for anything but reading: `shapes` and
`bench` are the shared yardstick and are used unmodified.
"""
import argparse
import functools
import json
import os
import sys
import time
import traceback

import jax
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, "instruments"))

import bench  # noqa: E402  shared, read-only
import shapes  # noqa: E402  shared, read-only

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk as shipping_streamindex_topk  # noqa: E402

import api  # noqa: E402

# name -> the three switches. `base` is the shipping kernel and takes none.
ARMS = (
    ("base", None),
    ("copy-off", dict(fix_growth=False, skip_empty_launches=False,
                      elide_score_prefill=False)),
    ("pick", dict(fix_growth=True, skip_empty_launches=False,
                  elide_score_prefill=False)),
    ("cond", dict(fix_growth=False, skip_empty_launches=True,
                  elide_score_prefill=False)),
    ("fill", dict(fix_growth=False, skip_empty_launches=False,
                  elide_score_prefill=True)),
    ("o1", dict(fix_growth=True, skip_empty_launches=True,
                elide_score_prefill=True)),
)

DEFAULT_ROWS = "A-D16-8k,A-D64-8k,B-D4-256k,B-D1-1M,B-P1024-256k"


def _make_fn(switches, kwargs):
    """One arm's callable. Already jitted -- `api.streamindex_topk` carries the
    decorator and the switches are `static_argnames`, so each arm gets its own
    cache entry and no arm can be handed another's executable."""
    if switches is None:
        return jax.jit(lambda *x: shipping_streamindex_topk(*x, **kwargs))
    return lambda *x: api.streamindex_topk(*x, **kwargs, **switches)


def selected_sets_match(a, b, num_live_rows):
    """Do two results select the same index SET on every live row?

    Not `np.array_equal`: the kernel emits in an arbitrary order, and the whole
    claim O1 makes is about the set, not the ordering. Returns
    (n_rows_differing, worst_symmetric_difference).
    """
    a, b = np.asarray(a), np.asarray(b)
    n_diff, worst = 0, 0
    for t in range(min(num_live_rows, a.shape[0])):
        sa = set(int(x) for x in a[t] if x >= 0)
        sb = set(int(x) for x in b[t] if x >= 0)
        d = len(sa ^ sb)
        if d:
            n_diff += 1
            worst = max(worst, d)
    return n_diff, worst


def run_row(name, repeats, iters):
    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}", flush=True)
    t0 = time.perf_counter()
    args, kwargs, meta = shapes.build(name)
    jax.block_until_ready(args[2])
    print(f"  built in {time.perf_counter() - t0:.1f}s  n={meta['n']}  "
          f"T={meta['T']} R={meta['R']} D={meta['D']}  "
          f"cache={meta['cache_bytes'] / 2**20:.0f}MiB  "
          f"score_buf={meta['score_bytes'] / 2**20:.0f}MiB", flush=True)

    fns = [(nm, _make_fn(sw, kwargs)) for nm, sw in ARMS]

    # -- identity first. A faster arm that selects a different set is not a
    # -- result, and finding that out after five repeats of timing is a waste.
    outs = {}
    for nm, fn in fns:
        outs[nm] = np.asarray(jax.block_until_ready(fn(*args)))
    n_live = int(np.asarray(args[5])[meta["R"]])   # cu_q_lens[R]
    took_fast_path = bool((outs["base"][0] == np.arange(shapes.K)).all())
    if took_fast_path:
        raise SystemExit(f"{name}: took _fast_path -- measures nothing")

    identity = {}
    for nm, _ in ARMS:
        if nm == "base":
            continue
        n_diff, worst = selected_sets_match(outs["base"], outs[nm], n_live)
        identity[nm] = dict(rows_differing=n_diff, worst_symdiff=worst)
        flag = "IDENTICAL" if n_diff == 0 else f"DIFFERS on {n_diff} rows"
        print(f"  selected set vs base: {nm:9} {flag}"
              f"{'' if n_diff == 0 else f', worst symdiff {worst}'}",
              flush=True)
    del outs

    # `bench.interleaved` calls each arm as `fn()` -- zero arguments, inputs
    # already closed over (instruments/bench.py:72,81). Bind them here. The
    # identity check above calls `fn(*args)` directly and so did not need this,
    # which is precisely how the mismatch survived to the first timing job.
    timed = [(nm, functools.partial(fn, *args)) for nm, fn in fns]
    res = bench.interleaved(timed, repeats=repeats, iters=iters, ref="base")

    # Say out loud which effects the measurement can actually resolve. The
    # design's rule, applied here rather than left to the reader: an effect
    # smaller than the spread of its own five paired deltas is not a result.
    print("", flush=True)
    for nm, _ in ARMS:
        r = res[nm]
        if r["delta_pct"] is None:
            continue
        d, s = r["delta_pct"], r["delta_spread_pct"]
        verdict = "resolved" if abs(d) > s else "UNRESOLVED (|d| < spread)"
        print(f"  {nm:9} d time {d:+.1f}%  spread {s:.1f}pt  -> {verdict}",
              flush=True)

    return dict(row=name, n=meta["n"], T=meta["T"], R=meta["R"], D=meta["D"],
                score_MiB=meta["score_bytes"] / 2**20,
                live_rows=n_live, identity=identity, timing=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--repeats", type=int, default=5,
                    help="5, not 3: two of the three effects are under 10%%")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    names = [r.strip() for r in a.rows.split(",") if r.strip()]
    for name in names:
        if name not in shapes.ROWS:
            raise SystemExit(f"unknown row {name!r}; have {sorted(shapes.ROWS)}")

    # One bad row must not cost the other four. A shape can die on its own --
    # a 4 GiB score buffer that does not fit, a compile that fails only at that
    # geometry -- and each row here is an independent measurement, so there is
    # nothing to be gained by throwing away the ones that worked. The failure is
    # printed in full and re-listed at the end so it cannot be read as a pass.
    out, failed = [], []
    for name in names:
        try:
            out.append(run_row(name, a.repeats, a.iters))
        except Exception:                                  # noqa: BLE001
            failed.append(name)
            print(f"\n  !! {name} FAILED:\n{traceback.format_exc()}", flush=True)
    if not out:
        raise SystemExit("every row failed -- no timing result")

    print("\n" + "=" * 78)
    print(f"{'row':15} " + " ".join(f"{nm:>11}" for nm, _ in ARMS[1:]))
    print(f"{'':15} " + " ".join(f"{'d time %':>11}" for _ in ARMS[1:]))
    for r in out:
        cells = []
        for nm, _ in ARMS[1:]:
            t = r["timing"][nm]
            cells.append(f"{t['delta_pct']:+7.1f}"
                         f"/{t['delta_spread_pct']:3.0f}")
        print(f"{r['row']:15} " + " ".join(f"{c:>11}" for c in cells))
    print("(d time % / d spread pt, paired against base inside each repeat; "
          "negative is faster)")
    if failed:
        print(f"\nROWS THAT FAILED and are absent above: {', '.join(failed)}")

    bad = [f"{r['row']}/{nm}" for r in out for nm in r["identity"]
           if r["identity"][nm]["rows_differing"]]
    if bad:
        print(f"\nSELECTED SET DIFFERS from base on: {', '.join(bad)} -- the "
              f"timings above are not a like-for-like comparison")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
