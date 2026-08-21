"""On-device step time: the distribution within a capture, and across captures.

**What "step time" means here.** The wall-clock on the chip of one execution of
the dominant graph, read off the ``XLA Modules`` line. Not a client-side
latency, not a throughput divided by a token count -- the duration the hardware
spent on one engine step.

**Three kinds of spread, and only one of them is instrument noise.**

*Step-to-step, within one capture.* Reported as ``sd`` and ``cv``. Mostly
**not** noise: consecutive steps carry different batch compositions, and a
phase label is a band rather than a fixed shape. Read it as "how heterogeneous
is this phase", and as what sets how many steps a capture needs.

*Rank-to-rank, within one capture.* Ranks differ by up to 8%, only about half
of them emit a device plane, and which half is not stable between captures.
That is why ``--ranks`` exists and why comparisons here are restricted to ranks
present on every side.

*Capture-to-capture, same bytes.* The one that decides whether the metric can
gate anything. The only way to get it is to capture the same tree more than
once -- see :mod:`.compare` for arm-level statistics over repeated captures.

The standard error printed beside each mean is ``sd / sqrt(n)`` over *steps*
and is **optimistic**: consecutive steps of a ramping benchmark are
autocorrelated, so the effective sample size is smaller than the step count.
Trust capture-to-capture agreement over it.
"""

from __future__ import annotations

import statistics as stats

from .table import describe, pct, table, us
from .trace import Capture, PhaseTrace, shared_ranks

#: Location estimators available to ``--estimator``. The default is the mean.
#: The others exist because they disagree, and the disagreement is information:
#: a phase capture starts and ends mid-ramp, so a few steps are a fraction of a
#: full one -- one rank recorded a 29.7 ms step against a 324 ms median. Those
#: drag the mean without saying anything about the stack. The median ignores
#: them but jumps between discrete samples at nine steps; p90 reads the
#: saturated steps. Agreement across all three is the signal a delta is real.
ESTIMATORS = ("mean", "median", "p90")


def summarise(capture: Capture) -> dict[str, dict[str, dict]]:
    """``{phase: {rank: describe(step_us) + module/op fields}}``."""
    out: dict[str, dict[str, dict]] = {}
    for phase, data in capture.phases.items():
        out[phase] = {
            r.rank: {
                **describe(list(r.step_us)),
                "module": r.step_module,
                "busy_us": r.busy_us,
                "async_us": r.async_us,
            }
            for r in data.ranks
        }
    return out


def pinned(phase: PhaseTrace, ranks: list[str], estimator: str) -> float | None:
    """The estimator averaged over a fixed rank set, so churn is not measured."""
    by_rank = phase.by_rank()
    values = [
        describe(list(by_rank[r].step_us))[estimator]
        for r in ranks if r in by_rank
    ]
    return stats.mean(values) if values else None


def compare(captures: list[Capture], estimators: tuple[str, ...] = ("mean",)) -> dict:
    """Each capture's rank-pinned step time against the first, per phase."""
    out: dict[str, dict] = {}
    all_phases = sorted({p for c in captures for p in c.phases})
    for phase in all_phases:
        present = [(c.label, c.phases[phase]) for c in captures if phase in c.phases]
        if len(present) < 2:
            out[phase] = {"captures": [lbl for lbl, _ in present], "shared_ranks": [],
                          "rows": []}
            continue
        shared = shared_ranks([data for _, data in present])
        refs = {e: pinned(present[0][1], shared, e) for e in estimators}
        rows = []
        for label, data in present:
            row: dict = {"capture": label}
            for est in estimators:
                value = pinned(data, shared, est)
                ref = refs[est]
                row[est] = value
                row[f"{est}_delta_pct"] = (
                    100.0 * (value - ref) / ref
                    if ref and value is not None else None
                )
            rows.append(row)
        out[phase] = {"shared_ranks": shared, "rows": rows}
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_capture(capture: Capture, *, components: bool = False) -> list[str]:
    lines = [f"\n## {capture.label}"]
    if not capture:
        return lines + ["  no device plane in any trace"]
    for phase, data in capture.phases.items():
        lines += ["", f"### {phase}", ""]
        rows = []
        for r in data.ranks:
            s = describe(list(r.step_us))
            rows.append([r.rank, s["n"], us(s["mean"]), us(s["median"]), us(s["sd"]),
                         f"{s['cv_pct']:.2f}", f"{s['sem_pct']:.2f}",
                         us(s["min"]), us(s["max"])])
        lines += table(
            ["rank", "steps", "mean us", "median us", "sd us", "cv %", "sem %",
             "min us", "max us"],
            rows, widths=[5, 5, 11, 11, 10, 6, 6, 11, 11])

        means = [r.mean_us for r in data.ranks]
        if len(means) > 1:
            lo, hi = min(means), max(means)
            lines.append(
                f"\nrank-to-rank: {us(lo)} - {us(hi)} us "
                f"({100.0 * (hi - lo) / stats.mean(means):.2f}% of the mean) "
                f"over {len(means)} ranks")
        for name in data.dropped:
            lines.append(f"dropped partial dump: {name}")

        if components:
            busy = data.mean_over_ranks("busy_us")
            step = data.mean_over_ranks("mean_us")
            ratio = busy / step if step else float("nan")
            note = "" if 0.95 <= ratio <= 1.05 else "   <-- ops no longer track the step"
            lines.append(
                f"summed op time:  {us(busy)} us/step, {ratio:.2f}x the step{note}")
            lines.append(
                f"async op time:   {us(data.mean_over_ranks('async_us'))} us/step "
                f"(overlaps the step; not additive)")

        modules = {r.step_module for r in data.ranks}
        if len(modules) > 1:
            lines.append(f"WARNING  ranks disagree on the dominant graph: {sorted(modules)}")
    return lines


def render_compare(result: dict, estimators: tuple[str, ...]) -> list[str]:
    lines = ["", "## capture to capture", ""]
    for phase, data in result.items():
        if not data.get("rows"):
            only = ", ".join(data.get("captures", [])) or "none"
            lines += [f"### {phase}",
                      f"  present in only one capture ({only}); nothing to compare", ""]
            continue
        lines.append(f"### {phase}   [ranks pinned to {','.join(data['shared_ranks'])}]")
        lines.append("")
        headers = ["capture"]
        widths = [26]
        for est in estimators:
            headers += [f"{est} us", "vs ref"]
            widths += [12, 9]
        rows = []
        for index, row in enumerate(data["rows"]):
            cells: list[object] = [row["capture"][:26]]
            for est in estimators:
                delta = row[f"{est}_delta_pct"]
                # The first capture is the reference by position. Deciding by
                # `delta == 0` instead would label a capture that genuinely
                # matched the reference as the reference -- which is the one
                # result worth seeing spelled out as +0.00%.
                cells += [us(row[est]),
                          "ref" if index == 0 else pct(delta, signed=True)]
            rows.append(cells)
        lines += table(headers, rows, widths=widths)
        lines.append("")
    lines.append("Higher is slower: these are microseconds of work, not throughput.")
    return lines


def emit_metrics(capture: Capture, prefix: str = "") -> list[str]:
    """``key: value`` lines, for a harness that harvests stdout.

    *prefix* namespaces the keys when more than one capture is being emitted;
    without it the second capture's lines would be indistinguishable from the
    first's and a harness would keep whichever it parsed last.
    """
    out = []
    for phase, data in capture.phases.items():
        if not data.ranks:
            continue
        summaries = [describe(list(r.step_us)) for r in data.ranks]
        key = f"{prefix}step_us_{phase}"
        out.append(f"{key}: {stats.mean(s['mean'] for s in summaries):.4f}")
        out.append(f"{key}_cv_pct: {stats.mean(s['cv_pct'] for s in summaries):.4f}")
        out.append(f"{key}_ranks: {len(summaries)}")
        out.append(f"{key}_steps: {data.steps}")
    return out
