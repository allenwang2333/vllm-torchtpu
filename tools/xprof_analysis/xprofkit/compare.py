"""Arm-level comparison: several captures of one tree against several of another.

A single capture of a candidate against a single capture of a baseline is not a
result. Two captures of *byte-identical* baseline code disagreed by 0.46% on
this stack, and a third moved the pair's spread to 0.95% -- wide enough to
manufacture or erase most kernel-level wins. So the unit of comparison here is
an **arm**: a named set of captures of the same tree.

    analyze.py compare --arm base=cap1,cap2,cap3 --arm cand=cap4,cap5,cap6

What comes out is, per phase: each arm's mean step time over its captures, the
spread between them, and a Welch comparison of every other arm against the
first with a 95% confidence interval.

**Read the interval, not the point estimate.** A delta whose interval spans
zero is not a measurement of anything, however good the midpoint looks; and a
delta whose interval spans your promotion threshold has not answered the
question you asked. Both happen routinely at three captures per arm.

Ranks are pinned to the intersection across *every* capture in *every* arm, so
no part of the comparison is measuring which ranks happened to report.
"""

from __future__ import annotations

import statistics as stats

from .steptime import pinned
from .table import describe, pct, table, us, welch
from .trace import Capture, shared_ranks


def compare_arms(arms: dict[str, list[Capture]], estimator: str = "mean") -> dict:
    """Per phase, each arm's captures summarised and tested against the first.

    ``arms`` maps a name to its captures; insertion order matters, because the
    first arm is the reference every other is measured against.
    """
    names = list(arms)
    out: dict[str, dict] = {}
    all_phases = sorted({p for caps in arms.values() for c in caps for p in c.phases})

    for phase in all_phases:
        # Every capture, in every arm, that saw this phase.
        present = {
            name: [c for c in caps if phase in c.phases]
            for name, caps in arms.items()
        }
        if any(not caps for caps in present.values()):
            missing = [n for n, caps in present.items() if not caps]
            out[phase] = {"skipped": f"not captured in arm(s): {', '.join(missing)}"}
            continue

        pins = shared_ranks(
            [c.phases[phase] for caps in present.values() for c in caps])
        if not pins:
            out[phase] = {"skipped": "no rank reported in every capture"}
            continue

        samples: dict[str, list[float]] = {}
        detail: dict[str, list[dict]] = {}
        for name, caps in present.items():
            values = []
            rows = []
            for c in caps:
                value = pinned(c.phases[phase], pins, estimator)
                if value is None:
                    continue
                values.append(value)
                rows.append({"capture": c.label, "value": value,
                             "steps": c.phases[phase].steps})
            samples[name] = values
            detail[name] = rows

        ref = names[0]
        arm_stats = {}
        for name, values in samples.items():
            s = describe(values)
            arm_stats[name] = {
                "n": s.get("n", 0),
                "mean": s.get("mean"),
                "sd": s.get("sd"),
                "cv_pct": s.get("cv_pct"),
                "min": s.get("min"),
                "max": s.get("max"),
                "captures": detail[name],
            }
            arm_stats[name]["vs_ref"] = (
                {} if name == ref else welch(values, samples[ref]))

        out[phase] = {"estimator": estimator, "pinned_ranks": pins,
                      "reference": ref, "arms": arm_stats}
    return out


def _interval(vs: dict) -> str:
    """The 95% CI cell, or why there is not one.

    :func:`~xprofkit.table.welch` returns an unbounded interval when the sample
    cannot support one -- one capture per arm, most often. Printing
    ``[-inf%, +inf%]`` is arithmetically right and reads as a formatting bug, so
    the cell says what happened instead.
    """
    if vs["t"] is None:
        return "n=1: not comparable" if min(vs["n_a"], vs["n_b"]) < 2 \
            else "no spread: no CI"
    return f"[{vs['ci_lo_pct']:+.2f}%, {vs['ci_hi_pct']:+.2f}%]"


def render(result: dict) -> list[str]:
    lines: list[str] = []
    for phase, data in result.items():
        lines += ["", f"## {phase}"]
        if "skipped" in data:
            lines.append(f"  skipped: {data['skipped']}")
            continue
        lines.append(
            f"  estimator: {data['estimator']} step time · "
            f"ranks pinned to {','.join(data['pinned_ranks'])} · "
            f"reference arm: {data['reference']}")
        lines.append("")

        rows = []
        for name, arm in data["arms"].items():
            vs = arm["vs_ref"]
            # "ref" is a fact about *which* arm this is, not about whether a
            # comparison came back: an arm whose captures all failed to pin
            # also has an empty `vs_ref`, and calling that one the reference
            # would silently promote it.
            is_ref = name == data["reference"]
            rows.append([
                name, arm["n"], us(arm["mean"]), us(arm["sd"]),
                pct(arm["cv_pct"]),
                "ref" if is_ref else (pct(vs["delta_pct"], signed=True) if vs
                                      else "no data"),
                "" if not vs or vs["t"] is None else f"{vs['t']:+.2f}",
                "" if not vs else _interval(vs),
            ])
        lines += table(
            ["arm", "n", "step us", "sd us", "cv", "vs ref", "t", "95% CI"],
            rows, widths=[16, 3, 12, 9, 7, 8, 7, 22])
        lines.append("")
        lines.append("  Higher step time is slower. A CI that spans zero means the")
        lines.append("  arms are not distinguishable at this many captures.")

        lines.append("")
        lines.append("  captures behind each arm:")
        for name, arm in data["arms"].items():
            for c in arm["captures"]:
                lines.append(f"    {name:<14} {c['capture']:<22} "
                             f"{us(c['value']):>12} us   ({c['steps']} steps)")
    return lines


def verdict(result: dict, phase: str | None = None) -> list[str]:
    """A one-line-per-arm summary a human can act on without reading tables."""
    lines = ["", "## verdict", ""]
    phases = [phase] if phase else [p for p, d in result.items() if "skipped" not in d]
    for name in phases:
        data = result.get(name) or {}
        if "skipped" in data or not data:
            continue
        for arm, stat in data["arms"].items():
            vs = stat["vs_ref"]
            if arm == data["reference"]:
                lines.append(f"  {name}: {arm} is the reference "
                             f"({us(stat['mean'])} us, n={stat['n']}, "
                             f"cv {pct(stat['cv_pct'])})")
                continue
            if not vs:
                lines.append(f"  {name}: {arm} has no capture that could be "
                             "pinned to the shared ranks; not compared")
                continue
            lo, hi = vs["ci_lo_pct"], vs["ci_hi_pct"]
            # An unbounded interval spans zero, so the "not distinguishable"
            # branch below would swallow this case -- and that wording claims a
            # measurement was made and came back empty. Nothing was measured.
            if vs["t"] is None:
                why = ("one capture per arm measures no run-to-run spread"
                       if min(vs["n_a"], vs["n_b"]) < 2
                       else "the captures show no spread at all")
                lines.append(
                    f"  {name}: {arm} is NOT COMPARABLE to the reference -- "
                    f"{why}; the point estimate is "
                    f"{pct(vs['delta_pct'], signed=True)} and means little "
                    f"(n={stat['n']})")
                continue
            if lo <= 0 <= hi:
                call = "NOT DISTINGUISHABLE from the reference"
            elif hi < 0:
                call = f"FASTER by {-vs['delta_pct']:.2f}%"
            else:
                call = f"SLOWER by {vs['delta_pct']:.2f}%"
            lines.append(f"  {name}: {arm} is {call} "
                         f"(CI [{lo:+.2f}%, {hi:+.2f}%], n={stat['n']})")
    return lines


def spread(captures: list[Capture], estimator: str = "mean") -> dict:
    """Capture-to-capture spread of one arm -- the instrument's noise floor.

    Run this on repeated captures of unchanged code before trusting any
    candidate delta: it is the smallest difference the instrument can resolve,
    and it is the number a promotion threshold should be set against.
    """
    out: dict[str, dict] = {}
    all_phases = sorted({p for c in captures for p in c.phases})
    for phase in all_phases:
        present = [c for c in captures if phase in c.phases]
        if len(present) < 2:
            continue
        pins = shared_ranks([c.phases[phase] for c in present])
        # Keep the capture beside its value while dropping the unusable ones:
        # filtering the values alone leaves the labels off by one, which
        # mislabels the whole table without changing a single number.
        measured = [(c, pinned(c.phases[phase], pins, estimator)) for c in present]
        measured = [(c, v) for c, v in measured if v is not None]
        if len(measured) < 2:
            continue
        values = [v for _, v in measured]
        s = describe(values)
        out[phase] = {
            "n": s["n"], "mean": s["mean"], "sd": s["sd"], "cv_pct": s["cv_pct"],
            "peak_to_peak_pct": 100.0 * (s["max"] - s["min"]) / s["mean"],
            "sem_pct": s["sem_pct"],
            "pinned_ranks": pins,
            "captures": {c.label: v for c, v in measured},
        }
    return out


def render_spread(result: dict, estimator: str) -> list[str]:
    lines = ["", f"## capture-to-capture spread ({estimator} step time)", ""]
    if not result:
        lines.append("  need at least two captures of the same phase")
        return lines
    rows = [
        [phase, d["n"], us(d["mean"]), us(d["sd"]), pct(d["cv_pct"]),
         pct(d["peak_to_peak_pct"]), pct(d["sem_pct"])]
        for phase, d in result.items()
    ]
    lines += table(["phase", "n", "mean us", "sd us", "cv", "peak-peak", "sem"],
                   rows, widths=[18, 3, 12, 9, 7, 9, 7])
    lines += [
        "",
        "  This is the noise floor. A candidate delta smaller than the",
        "  peak-to-peak here is not a result, whatever its sign.",
    ]
    return lines
