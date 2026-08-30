#!/usr/bin/env python3
"""Summarise a k3_knob_matrix.sh results tree into a noise band and arm deltas.

Reads `cellNN_<arm>/isl*_osl*_c*[.warmupK].json` and reports, per metric:

  within-server sd  -- pooled over the repeats inside each cell
  across-restart sd -- over the cell means of one arm
  arm delta         -- mean(arm) - mean(base), with the across-restart sd
                       of the difference so it can be called significant or not

Also diffs `completions.json` between cells: greedy token agreement is the
numerics gate for a knob (like fp8 KV) that is not expected to be bit-exact.
"""
import argparse
import json
import math
import os
import re
import statistics as st

METRICS = ("median_tpot_ms", "mean_tpot_ms", "median_ttft_ms",
           "total_token_throughput", "median_itl_ms")
CELL_RE = re.compile(r"^cell(\d+)_(.+)$")
REP_RE = re.compile(r"^(isl\d+_osl\d+_c\d+)(?:\.warmup(\d+))?\.json$")


def load_cells(root: str) -> list[dict]:
    cells = []
    for name in sorted(os.listdir(root)):
        m = CELL_RE.match(name)
        if not m or not os.path.isdir(os.path.join(root, name)):
            continue
        cell = {"index": int(m.group(1)), "arm": m.group(2), "dir": name,
                "reps": {}}
        for fn in sorted(os.listdir(os.path.join(root, name))):
            rm = REP_RE.match(fn)
            if not rm:
                continue
            # warmup1..N run before the unsuffixed final run, so N+1 is it.
            order = int(rm.group(2)) if rm.group(2) else 10_000
            try:
                with open(os.path.join(root, name, fn)) as fh:
                    cell["reps"].setdefault(rm.group(1), []).append(
                        (order, json.load(fh)))
            except (OSError, ValueError):
                pass
        for series in cell["reps"].values():
            series.sort(key=lambda t: t[0])
        cells.append(cell)
    return cells


def pooled_sd(groups: list[list[float]]) -> tuple[float, int]:
    """sqrt of the pooled within-group variance, and its degrees of freedom."""
    ss = dof = 0.0
    for g in groups:
        if len(g) < 2:
            continue
        mu = st.fmean(g)
        ss += sum((x - mu) ** 2 for x in g)
        dof += len(g) - 1
    return (math.sqrt(ss / dof) if dof else float("nan")), int(dof)


def fmt(x: float, nd: int = 4) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def report_case(case: str, cells: list[dict], out: list[str]) -> None:
    arms = sorted({c["arm"] for c in cells}, key=lambda a: (a != "base", a))
    out.append(f"\n### {case}")
    out.append("cell listing (rep values, in run order):")
    for c in cells:
        for metric in ("median_tpot_ms",):
            vals = [r[1].get(metric) for r in c["reps"].get(case, [])]
            out.append(f"  {c['dir']:>22}  {metric}: "
                       + ", ".join(fmt(v) for v in vals if v is not None))

    for metric in METRICS:
        out.append(f"\n-- {metric}")
        per_arm_cellmeans: dict[str, list[float]] = {}
        within_groups: list[list[float]] = []
        for c in cells:
            vals = [r[1].get(metric) for r in c["reps"].get(case, [])]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if not vals:
                continue
            within_groups.append(vals)
            per_arm_cellmeans.setdefault(c["arm"], []).append(st.fmean(vals))
        if not within_groups:
            continue
        w_sd, w_dof = pooled_sd(within_groups)
        grand = st.fmean([v for g in within_groups for v in g])
        out.append(f"   within-server  sd={fmt(w_sd)} "
                   f"({fmt(100 * w_sd / grand, 3)}% of {fmt(grand)}), dof={w_dof}")
        for arm in arms:
            means = per_arm_cellmeans.get(arm, [])
            if not means:
                continue
            mu = st.fmean(means)
            sd = st.stdev(means) if len(means) > 1 else float("nan")
            out.append(f"   arm {arm:<10} n_cells={len(means)} mean={fmt(mu)} "
                       f"across-restart sd={fmt(sd)} "
                       f"({fmt(100 * sd / mu, 3)}%)")
        base = per_arm_cellmeans.get("base", [])
        for arm in arms:
            if arm == "base" or not base:
                continue
            means = per_arm_cellmeans[arm]
            if len(base) < 2 or len(means) < 2:
                continue
            mb, ma = st.fmean(base), st.fmean(means)
            sb, sa = st.stdev(base), st.stdev(means)
            se = math.sqrt(sb ** 2 / len(base) + sa ** 2 / len(means))
            delta = ma - mb
            out.append(f"   delta {arm} vs base: {fmt(delta)} "
                       f"({fmt(100 * delta / mb, 3)}%)  se={fmt(se)}  "
                       f"t={fmt(delta / se, 2) if se else 'inf'}")


def report_completions(root: str, cells: list[dict], out: list[str]) -> None:
    caps = {}
    for c in cells:
        p = os.path.join(root, c["dir"], "completions.json")
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    caps[c["dir"]] = json.load(fh)
            except (OSError, ValueError):
                pass
    if len(caps) < 2:
        return
    out.append("\n### greedy-completion agreement (pairwise, vs cell01)")
    names = list(caps)
    ref_name = names[0]
    for other in names[1:]:
        a, b = caps[ref_name], caps[other]
        exact = tot = 0
        first_div = []
        for k in sorted(set(a) & set(b)):
            ta, tb = a[k].get("tokens") or [], b[k].get("tokens") or []
            if not ta or not tb:
                continue
            tot += 1
            if ta == tb:
                exact += 1
                first_div.append(len(ta))
            else:
                first_div.append(next(
                    (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                    min(len(ta), len(tb))))
        if tot:
            out.append(f"  {ref_name} vs {other}: exact={exact}/{tot} "
                       f"mean_first_divergence={fmt(st.fmean(first_div), 1)} "
                       f"tokens")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    cells = load_cells(args.results_dir)
    out = [f"# k3_knob_summary  {args.results_dir}",
           f"cells: {len(cells)}"]
    cases = sorted({c for cell in cells for c in cell["reps"]})
    for case in cases:
        report_case(case, cells, out)
    report_completions(args.results_dir, cells, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
