"""A/B two captures at op level: which ops moved, and by how much.

Per phase, per op, in microseconds of device time **per engine step**. That
normalisation is the whole design. Two captures do not contain the same number
of steps -- a phased profiler stops after N iterations, and the phases do not
arrive at the same moment in the two runs -- so raw totals are not comparable.
Dividing each side by its own step count makes a step of one tree directly
comparable to a step of the other, which is the only comparison worth making.

**Why per phase.** A change to the chunked-prefill path and a change to the
decode path both collapse into one number in a throughput measurement, and they
can cancel. Split by phase, a prefill win sitting next to a decode regression
is visible as exactly that.

**Read the family table first.** Exact op names carry compiler-assigned
suffixes (``copy.18``, ``reshape.176``) that are not stable across builds, so a
byte-identical op can appear as one delete and one add. The family table strips
the suffix, and renumbering cancels there. The exact-name table is for grepping
a dumped module once the family table has told you where to look.

A diff of two captures of *unchanged* code is the honest way to calibrate this:
whatever it reports is the floor below which a row means nothing.
"""

from __future__ import annotations

from .breakdown import (
    dead_rule_note, match_phase, one_sided_rule_note, src_rule_warning,
    taxonomy_notes,
)
from .table import pct, table, us
from .taxonomy import UNCLASSIFIED, Taxonomy, family
from .trace import Capture, shared_ranks

#: Rows below this many microseconds per step in *both* captures are folded
#: into an "(everything else)" line.
NOISE_US = 0.5

#: There are a few dozen categories at most, and a category folded out of the
#: table is exactly where a regression would hide, so that table has no top-N.
NO_LIMIT = float("inf")


def _categories(phases, taxonomy: Taxonomy) -> tuple[dict[str, str],
                                                     list[set[int]]]:
    """``{op name: category}`` over several phase traces, and what fired where.

    Delegates to :func:`~xprofkit.breakdown.match_phase` rather than
    classifying here, and that is the point of the function: this used to walk
    the ranks itself and keep the first answer, while ``breakdown`` walked them
    and kept the last, so the two commands filed the same microseconds under
    different categories on the same capture -- ``moe_glue`` differed by 114 us
    per step and ``fusion`` by 213x. There is one reading of an op's attributes
    per phase now, and both commands take it.

    The base capture's label wins for an op both captures ran, so a category
    cannot appear to move purely because the candidate's build attributed the
    op differently.

    The second return value is the set of rule indices that claimed at least
    one op, one set per phase trace in argument order. It is free here --
    ``match_phase`` returns rule indices and the category is derived from them
    -- and without it ``diff`` had no way to say a rule was dead, while
    ``breakdown`` said it twice on the same taxonomy. A shadowed rule moves a
    DELTA, which is the number somebody makes a go/no-go on.
    """
    out: dict[str, str] = {}
    fired: list[set[int]] = []
    for phase in phases:
        matched = match_phase(phase, taxonomy)
        fired.append({index for index in matched.values() if index is not None})
        for name, index in matched.items():
            out.setdefault(name, UNCLASSIFIED if index is None
                           else taxonomy.rules[index].category)
    return out, fired


def diff(base: Capture, cand: Capture, taxonomy: Taxonomy) -> dict:
    out: dict[str, dict] = {}
    shared = [p for p in base.phases if p in cand.phases]
    for phase in shared:
        b_phase, c_phase = base.phases[phase], cand.phases[phase]
        pins = shared_ranks([b_phase, c_phase])
        if not pins:
            out[phase] = {"skipped": "no rank reported on both sides"}
            continue

        def per_step(trace) -> dict[str, float]:
            ranks = [r for r in trace.ranks if r.rank in pins]
            totals: dict[str, float] = {}
            for rank in ranks:
                for name, value in rank.op_us.items():
                    totals[name] = totals.get(name, 0.0) + value / len(ranks)
            return totals

        b_ops, c_ops = per_step(b_phase), per_step(c_phase)
        op_cat, (b_fired, c_fired) = _categories([b_phase, c_phase], taxonomy)
        rows = [
            {"op": name,
             "category": op_cat.get(name, UNCLASSIFIED),
             "base_us": b_ops.get(name, 0.0),
             "cand_us": c_ops.get(name, 0.0),
             "delta_us": c_ops.get(name, 0.0) - b_ops.get(name, 0.0)}
            for name in set(b_ops) | set(c_ops)
        ]
        rows.sort(key=lambda r: -abs(r["delta_us"]))

        # Categories come from the ops, not from the families: one family can
        # span several categories (`multiply_convert_fusion` is three different
        # projections here) and rolling it up first would book all of it under
        # whichever category happened to win.
        cats: dict[str, dict] = {}
        for row in rows:
            entry = cats.setdefault(row["category"],
                                    {"op": row["category"], "base_us": 0.0,
                                     "cand_us": 0.0})
            entry["base_us"] += row["base_us"]
            entry["cand_us"] += row["cand_us"]

        fams: dict[str, dict] = {}
        fam_cats: dict[str, dict[str, float]] = {}
        for row in rows:
            key = family(row["op"])
            entry = fams.setdefault(
                key, {"op": key, "base_us": 0.0, "cand_us": 0.0})
            entry["base_us"] += row["base_us"]
            entry["cand_us"] += row["cand_us"]
            seen = fam_cats.setdefault(key, {})
            seen[row["category"]] = (seen.get(row["category"], 0.0)
                                     + row["base_us"] + row["cand_us"])
        for key, entry in fams.items():
            seen = fam_cats[key]
            best = max(seen, key=lambda c: seen[c])
            # `+N`: the row is the whole family, the category named is where
            # most of its time went. The category table above is the exact one.
            entry["category"] = f"{best} +{len(seen) - 1}" if len(seen) > 1 else best
        fam_rows = sorted(
            ({**e, "delta_us": e["cand_us"] - e["base_us"]} for e in fams.values()),
            key=lambda r: -abs(r["delta_us"]))

        cat_rows = sorted(
            ({**e, "delta_us": e["cand_us"] - e["base_us"]} for e in cats.values()),
            key=lambda r: -abs(r["delta_us"]))

        def mean_step(trace) -> float:
            ranks = [r for r in trace.ranks if r.rank in pins]
            return sum(r.mean_us for r in ranks) / len(ranks) if ranks else 0.0

        out[phase] = {
            "pinned_ranks": pins,
            # Whether each side carries source lines at all, kept per side
            # rather than as one flag. A capture with no op-stats sidecar
            # classifies with every `src` rule disabled, so diffing one against
            # a sourced capture compares two different classifications -- a
            # difference in instrumentation, printed as a difference in the
            # code. Which of the two it is, is worth naming.
            "sourced": {label: any(src for _, src in trace.op_attributes().values())
                        for label, trace in (("base", b_phase), ("cand", c_phase))},
            "base_step_us": mean_step(b_phase),
            "cand_step_us": mean_step(c_phase),
            "base_busy_us": sum(b_ops.values()),
            "cand_busy_us": sum(c_ops.values()),
            "base_steps": b_phase.steps,
            "cand_steps": c_phase.steps,
            # Dead means dead on *both* sides. A rule alive on one capture is
            # evidence the two ran different ops, not evidence the taxonomy is
            # wrong, and counting it as dead would report a taxonomy defect
            # every time the candidate stopped emitting something.
            "dead_rules": [
                {"index": i, "category": rule.category}
                for i, rule in enumerate(taxonomy.rules)
                if i not in b_fired and i not in c_fired],
            "one_sided_rules": [
                {"index": i, "category": rule.category,
                 "alive": "base" if i in b_fired else "cand"}
                for i, rule in enumerate(taxonomy.rules)
                if (i in b_fired) != (i in c_fired)],
            "categories": cat_rows,
            "families": fam_rows,
            "ops": rows,
        }
    return {
        "phases": out,
        "only_in_base": [p for p in base.phases if p not in cand.phases],
        "only_in_cand": [p for p in cand.phases if p not in base.phases],
        "base_label": base.label,
        "cand_label": cand.label,
    }


def _rows(rows: list[dict], limit: int, with_category: bool) -> list[list[object]]:
    out: list[list[object]] = []
    folded_b = folded_c = 0.0
    for row in rows:
        if len(out) >= limit or max(row["base_us"], row["cand_us"]) < NOISE_US:
            folded_b += row["base_us"]
            folded_c += row["cand_us"]
            continue
        cells: list[object] = [row["op"][:46]]
        if with_category:
            cells.append(row.get("category", ""))
        cells += [us(row["base_us"], 2), us(row["cand_us"], 2),
                  f"{row['delta_us']:+,.2f}"]
        out.append(cells)
    if folded_b or folded_c:
        cells = ["(everything else)"]
        if with_category:
            cells.append("")
        cells += [us(folded_b, 2), us(folded_c, 2), f"{folded_c - folded_b:+,.2f}"]
        out.append(cells)
    return out


def render(result: dict, *, ops: int = 25, families: int = 20,
           taxonomy: Taxonomy | None = None) -> list[str]:
    base, cand = result["base_label"], result["cand_label"]
    lines = [f"base: {base}    cand: {cand}",
             "Positive delta = the candidate spends more device time. "
             "Microseconds per engine step."]
    for phase, d in result["phases"].items():
        if "skipped" in d:
            lines += ["", f"## {phase}", f"  skipped: {d['skipped']}"]
            continue
        b, c = d["base_step_us"], d["cand_step_us"]
        lines += [
            "",
            f"## {phase}   [ranks pinned to {','.join(d['pinned_ranks'])}]",
            "",
            f"step time  {us(b)} -> {us(c)} us  "
            f"({pct(100 * (c - b) / b if b else float('nan'), signed=True)})   "
            f"[{d['base_steps']} / {d['cand_steps']} steps]",
            f"op time    {us(d['base_busy_us'])} -> {us(d['cand_busy_us'])} us",
            "",
            "### by category",
            "",
        ]
        lines += table(["category", "base us", "cand us", "delta"],
                       _rows(d["categories"], NO_LIMIT, False),
                       widths=[46, 11, 11, 11])
        lines += ["", "### by op family (suffix stripped — read this one)", ""]
        lines += table(["op family", "category", "base us", "cand us", "delta"],
                       _rows(d["families"], families, True),
                       widths=[46, 23, 11, 11, 11])
        lines += ["", "### by exact op name (suffixes are not stable across builds)", ""]
        lines += table(["op", "base us", "cand us", "delta"],
                       _rows(d["ops"], ops, False), widths=[46, 11, 11, 11])
        if taxonomy is not None:
            sourced = d["sourced"]
            lines += src_rule_warning(taxonomy, any(sourced.values()))
            if taxonomy.src_rules and len(set(sourced.values())) > 1:
                side = "base" if sourced["base"] else "cand"
                other = "cand" if side == "base" else "base"
                lines += [
                    "",
                    f"**WARNING** — {side} carries source lines and {other} does "
                    f"not, so the two captures were classified by different rules.",
                    f"  Categories above follow {base if side == 'base' else cand}; "
                    "an op only the other side ran is filed without its `src`",
                    "  rule. Re-capture with the op-stats sidecar on both sides "
                    "before reading any category-level delta.",
                ]
            lines += dead_rule_note(taxonomy, d.get("dead_rules", []),
                                    absent="neither capture contains")
            lines += one_sided_rule_note(taxonomy, d.get("one_sided_rules", []),
                                         base, cand)
    # Outside the phase loop: a note about the taxonomy file is true once, and
    # `diff` renders every phase on one page where `breakdown` renders one.
    if taxonomy is not None:
        lines += taxonomy_notes(taxonomy)
    for phase in result["only_in_base"]:
        lines.append(f"\nphase '{phase}' captured only for {base}")
    for phase in result["only_in_cand"]:
        lines.append(f"\nphase '{phase}' captured only for {cand}")
    return lines
