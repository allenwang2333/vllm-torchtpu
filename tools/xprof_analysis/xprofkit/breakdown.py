"""Where one engine step's device time goes, by category and by op family.

Everything here is **device microseconds per engine step**, with the same
normalisation on the numerator and the denominator. That matters more than it
sounds. The common way to get an "end-to-end share" is to time a kernel in
isolation and divide by a step time measured somewhere else -- which puts
device time over host time and is not a fraction of anything. On this stack
that method produced "the kernel is 10.4% of a step, so a free kernel is worth
11.6%", a figure that survived several rounds of planning before anyone
measured it in situ and got 9.35%.

Three tables come out, in increasing detail:

``categories``
    Op time grouped by the taxonomy's rules. The first thing to read.

``families``
    Individual op families -- ``copy``, ``fusion.13`` folded to ``fusion`` --
    with their instance count and cost per call. Where a category is large and
    you want to know which op made it large.

``blocks``
    Optional, and only when the taxonomy's rules carry ``block:``. Categories
    subtotalled into the part of the model they were declared to belong to,
    *including the compiler glue the op name does not attribute*: on this stack
    an eighth of a kernel's true cost is pads and relayouts filed under
    ``layout``, and a ``src:`` rule with a ``block:`` on it puts them back.
    Declared, never inferred -- see :func:`analyse_blocks`.

**Occupancy** is summed op time over step time. Near 1.0 is expected: the op
line is a partition of the module's own time. Well below 1.0 means the step
contains time no op accounts for. Well *above* 1.0 usually means the phase
directory held two different captures and their means were averaged together --
112% was how that bug announced itself.

Asynchronous work is reported separately and never summed in. It overlaps the
step rather than adding to it, so folding it into a "% of step" column would
produce percentages adding to several hundred.
"""

from __future__ import annotations

from .table import pct, table, us
from .taxonomy import (
    SPARSECORE_THREAD, TENSORCORE, TOTAL, UNASSIGNED, UNCLASSIFIED, Taxonomy,
    device_of, family,
)
from .trace import Capture, PhaseTrace

#: Families below this share of the step are folded into one "(everything
#: else)" row. A serving step touches hundreds of ops and all but a few dozen
#: are rounding error; printing them buries the ones that matter.
FOLD_BELOW_PCT = 0.05


def match_phase(phase: PhaseTrace, taxonomy: Taxonomy,
                attributes: dict[str, tuple[str, str]] | None = None
                ) -> dict[str, int | None]:
    """``{op name: index of the rule that claimed it}``, decided once.

    The rule index rather than the category, because they are not the same
    question and only one of them can be recovered from the other. Several
    rules routinely share a category -- `matmul_other` is three in the shipped
    file -- so "which categories got time" cannot say which *rules* fired, and
    a rule shadowed by an earlier one with the same category leaves no trace at
    all. :func:`dead_rule_note` is the reader of that difference.

    Classification takes the op's signature and source line as well as its
    name, and both of those are read per rank -- so classifying inside a
    per-rank loop lets the answer depend on iteration order, and two commands
    that iterate differently report different categories for the same
    microseconds. :meth:`~xprofkit.trace.PhaseTrace.merge_attributes` is the
    one reading; this is the one classification of it. Both commands call this.

    ``attributes`` is that reading, for a caller that already has it: the merge
    walks every op of every rank, and doing it twice per phase is the whole
    cost of this function.
    """
    if attributes is None:
        attributes = phase.op_attributes()
    return {name: taxonomy.match(name, hlo, src)
            for name, (hlo, src) in attributes.items()}


def analyse_phase(phase: PhaseTrace, taxonomy: Taxonomy) -> dict:
    """Per-step microseconds by category and by op family, averaged over ranks."""
    n = len(phase.ranks)
    attributes, disputed = phase.merge_attributes()
    op_rule = match_phase(phase, taxonomy, attributes)
    op_cat = {name: UNCLASSIFIED if i is None else taxonomy.rules[i].category
              for name, i in op_rule.items()}
    categories: dict[str, float] = {}
    cat_fams: dict[str, dict[str, float]] = {}
    fam_us: dict[str, float] = {}
    fam_n: dict[str, int] = {}
    ops: dict[str, float] = {}
    # (family, category) cells. Finer than the family alone, and the difference
    # matters: `multiply_convert_fusion` spans four categories -- the QKV
    # projection, the linear-attention input projection, the output projection
    # and one the taxonomy leaves in `layout` -- and only the cells can put
    # those four in the three different blocks they belong to.
    cell_us: dict[tuple[str, str], float] = {}
    sources: dict[str, float] = {}
    source_cats: dict[str, dict[str, float]] = {}
    source_fams: dict[str, dict[str, float]] = {}
    no_source = 0.0
    # Ops with no source line, kept split by category. Rolling them into one
    # anonymous row makes a quarter of the step look unexplained when in fact
    # it is ~all SPMD reduce-scatter, whose only recorded stack is the
    # compiler's own and is trimmed away as plumbing.
    no_source_cats: dict[str, float] = {}
    no_source_fams: dict[str, dict[str, float]] = {}

    devices: dict[str, float] = {}
    dev_fams: dict[str, dict[str, float]] = {}

    # Resolve the device per *family*, not per op. An async collective is two
    # instructions -- `.call-start` carries `async_execution_thread`, its
    # matching `.call-done` does not -- and booking the wait against
    # TensorCore because the start-half was labelled would split one
    # collective across two units and halve the SparseCore total.
    fam_device: dict[str, str] = {}
    for name, (hlo, _) in attributes.items():
        if device_of(hlo) == SPARSECORE_THREAD:
            fam_device[family(name)] = SPARSECORE_THREAD

    for rank in phase.ranks:
        for name, value in rank.op_us.items():
            src = attributes[name][1]
            cat = op_cat[name]
            fam, share = family(name), value / n
            categories[cat] = categories.get(cat, 0.0) + share
            members = cat_fams.setdefault(cat, {})
            members[fam] = members.get(fam, 0.0) + share
            fam_us[fam] = fam_us.get(fam, 0.0) + share
            ops[name] = ops.get(name, 0.0) + share
            cell_us[(fam, cat)] = cell_us.get((fam, cat), 0.0) + share

            if src:
                sources[src] = sources.get(src, 0.0) + share
                members = source_cats.setdefault(src, {})
                members[cat] = members.get(cat, 0.0) + share
                sfam = source_fams.setdefault(src, {})
                sfam[fam] = sfam.get(fam, 0.0) + share
            else:
                no_source += share
                no_source_cats[cat] = no_source_cats.get(cat, 0.0) + share
                nsf = no_source_fams.setdefault(cat, {})
                nsf[fam] = nsf.get(fam, 0.0) + share

            # The device axis is orthogonal to the category: an op is filed
            # under one category and one unit, and the two do not nest.
            dev = fam_device.get(fam, TENSORCORE)
            devices[dev] = devices.get(dev, 0.0) + share
            dev_members = dev_fams.setdefault(dev, {})
            dev_members[fam] = dev_members.get(fam, 0.0) + share

    # `n` is the count of distinct HLO *instances* in the compiled module --
    # `kernel.90` through `kernel.134` is 45 -- not the number of events, which
    # is that times the steps traced. Counting events would make every family's
    # count depend on how long the capture ran, so two captures of the same
    # program would disagree. Taken from one rank's key set: all ranks run the
    # same program. It is a size, not evidence of structure: it says how many
    # instructions carry the name, and nothing about which part of the model
    # emitted them. Only `--families` prints it, next to us/call, which is the
    # one question it can answer.
    reference = phase.ranks[0].op_us if phase.ranks else {}
    # Per cell as well as per family. The count was taken *off* the rendered
    # tables because a column of instance counts beside a table of blocks read
    # as structure -- so many of them, so one per layer -- which is the
    # inference this tool no longer makes. That argument is about a page a
    # human reads; it does not apply to a machine-readable field, and without
    # one here `--json` cannot answer "how many instances are in this block"
    # at all.
    cell_n: dict[tuple[str, str], int] = {}
    for name in reference:
        fam = family(name)
        fam_n[fam] = fam_n.get(fam, 0) + 1
        key = (fam, op_cat[name])
        cell_n[key] = cell_n.get(key, 0) + 1

    # Rules that claimed nothing. Computed here rather than in the renderer so
    # that `--json` carries it: a dead rule is the one finding on the page that
    # no other field can be used to reconstruct, because its category may well
    # be alive on the strength of a sibling rule.
    fired = {i for i in op_rule.values() if i is not None}
    dead_rules = [{"index": i, "category": rule.category}
                  for i, rule in enumerate(taxonomy.rules) if i not in fired]

    step_us = phase.mean_over_ranks("mean_us")
    busy_us = phase.mean_over_ranks("busy_us")
    result = {
        "phase": phase.phase,
        "ranks": len(phase.ranks),
        "rank_ids": phase.rank_ids,
        "steps": phase.steps,
        "step_module": phase.ranks[0].step_module if phase.ranks else "",
        "dropped": list(phase.dropped),
        "step_us": step_us,
        "busy_us": busy_us,
        "async_us": phase.mean_over_ranks("async_us"),
        "occupancy": busy_us / step_us if step_us else None,
        "categories": categories,
        # Orthogonal to `categories`: same total, split by execution unit.
        "devices": devices,
        "device_families": {
            dev: dict(sorted(members.items(), key=lambda kv: -kv[1]))
            for dev, members in dev_fams.items()
        },
        # Which families make up each category, largest first. This is what
        # turns "collective is 25% of the step" into something actionable.
        "category_families": {
            cat: dict(sorted(members.items(), key=lambda kv: -kv[1]))
            for cat, members in cat_fams.items()
        },
        "families": {k: {"n": fam_n.get(k, 0), "us": v} for k, v in fam_us.items()},
        # Per op, so a later table never has to re-classify from the name
        # alone: the name is the one input a rule cannot rely on, and
        # re-deriving it here is how two tables on the same page disagree.
        "op_categories": op_cat,
        # One row per (family, category) pair -- the granularity the block
        # table runs at. See `cell_us` above for why not per family.
        "cells": [{"family": f, "category": c, "us": v,
                   "n": cell_n.get((f, c), 0)}
                  for (f, c), v in cell_us.items()],
        # Whose code asked for the work, from the op-stats sidecar. Empty when
        # the capture has no sidecar or the schema is not installed.
        "sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "source_categories": {
            loc: dict(sorted(members.items(), key=lambda kv: -kv[1]))
            for loc, members in source_cats.items()
        },
        "source_families": {
            loc: dict(sorted(members.items(), key=lambda kv: -kv[1]))
            for loc, members in source_fams.items()
        },
        "no_source_us": no_source,
        "no_source_categories": dict(sorted(no_source_cats.items(),
                                            key=lambda kv: -kv[1])),
        "no_source_families": {
            cat: dict(sorted(members.items(), key=lambda kv: -kv[1]))
            for cat, members in no_source_fams.items()
        },
        # Ops whose ranks disagreed and so lost a field. Not arithmetic -- the
        # time is all still counted -- but a field an op lost is a field no
        # `hlo:` or `src:` rule can match on, and a stack where that became
        # common would lose rule coverage with no other symptom.
        "disputed": disputed,
        # `categories[N]` of the taxonomy file, for the rules no op matched.
        "dead_rules": dead_rules,
        "ops": ops,
    }
    # The block split is part of the *analysis*, not of the rendering. It used
    # to be computed inside `render`, which meant the headline table -- moe at
    # 69% of the step -- existed only as fixed-width text and a regression
    # check on block attribution had to scrape it. Merged in rather than
    # nested so `result["blocks"]` is the totals themselves.
    result.update(analyse_blocks(result, taxonomy))
    return result


def analyse(capture: Capture, taxonomy: Taxonomy) -> dict:
    return {phase: analyse_phase(data, taxonomy)
            for phase, data in capture.phases.items()}


def analyse_blocks(data: dict, taxonomy: Taxonomy) -> dict:
    """Category time subtotalled into the blocks the taxonomy declares.

    A block is a part of the model -- ``attn``, ``moe`` -- and which one an op
    belongs to is **declared by the taxonomy, not derived from the trace**. The
    trace cannot say: a step is one flat list of ops with no layer or module
    boundary in it, the compiler glue around a kernel is named for what it does
    rather than for whose it is, and a collective's source stack is the
    compiler's own (see :mod:`~xprofkit.opstats`). So the rule that already
    knows enough to name the category names the block too, and the report says
    on the page that it was told rather than that it found out.

    The predecessor of this function inferred the block from instance counts
    instead; :mod:`~xprofkit.taxonomy` records why that was removed.

    Categories no rule assigned keep their own row rather than being spread
    over blocks they may have nothing to do with. Blocks do not nest and they
    sum to the step.
    """
    block_of = taxonomy.block_of
    blocks = taxonomy.blocks
    by_block: dict[str, dict[str, float]] = {b: {} for b in blocks}
    # The cells behind each row, kept so `--verbose` can name every op that
    # went into a block rather than only its category subtotals.
    cells_of: dict[str, list[dict]] = {b: [] for b in [*blocks, UNASSIGNED]}
    unassigned: dict[str, float] = {}

    for cell in data["cells"]:
        cat, value = cell["category"], cell["us"]
        block = block_of.get(cat)
        if block is None:
            unassigned[cat] = unassigned.get(cat, 0.0) + value
            cells_of[UNASSIGNED].append(cell)
            continue
        by_block[block][cat] = by_block[block].get(cat, 0.0) + value
        cells_of[block].append(cell)

    return {
        "blocks": {
            block: {
                "total_us": sum(by_block[block].values()),
                "categories": by_block[block],
                "cells": sorted(cells_of[block], key=lambda c: -c["us"]),
            }
            for block in blocks
        },
        "unassigned": unassigned,
        "unassigned_cells": sorted(cells_of[UNASSIGNED], key=lambda c: -c["us"]),
        "unassigned_us": sum(unassigned.values()),
    }


def family_categories(cells: list[dict]) -> dict[str, str]:
    """``{family: the category holding most of its time}``, ``+N`` if it spans more.

    A family is not a category. ``multiply_convert_fusion`` is the QKV
    projection, the linear-attention input projection and the output
    projection under one compiler-assigned name; re-classifying that name on
    its own -- with no signature and no source line, which is all a family has
    -- answers ``elementwise`` for all three and contradicts the category
    table on the same page. The cells already carry the real answer.
    """
    spread: dict[str, dict[str, float]] = {}
    for cell in cells:
        seen = spread.setdefault(cell["family"], {})
        seen[cell["category"]] = seen.get(cell["category"], 0.0) + cell["us"]
    out = {}
    for fam, seen in spread.items():
        best = max(seen, key=lambda c: seen[c])
        out[fam] = f"{best} +{len(seen) - 1}" if len(seen) > 1 else best
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
#: Width of the "op families" column on the category table.
NAMES_WIDTH = 52
#: Width of the category/unit column. Wide enough for a split taxonomy's
#: names (`attn_gather_reduce`) without wrapping.
CATEGORY_WIDTH = 20

#: The same column on a table whose rows are families, where a category can
#: carry the ` +N` suffix :func:`family_categories` adds.
FAMILY_CATEGORY_WIDTH = CATEGORY_WIDTH + 3


def name_families(members: dict[str, float], limit: int,
                  width: int = NAMES_WIDTH) -> str:
    """The biggest families in a category, as many as fit on one line.

    ``members`` must already be ordered largest-first. Anything left over is
    reported as a ``+N`` count rather than silently dropped -- a category can
    be one dominant op or a hundred small ones, and which it is changes what
    you would do about it.
    """
    if not members:
        return ""
    names = list(members)
    shown: list[str] = []
    for name in names[:limit]:
        text = ", ".join([*shown, name])
        if shown and len(text) > width - 5:
            break
        shown.append(name)
    text = ", ".join(shown)[: width - 5]
    rest = len(names) - len(shown)
    return f"{text} +{rest}" if rest > 0 else text


#: Width of the op-family column in a `--verbose` expansion. Long enough for
#: `gmm_v2-g_64-m_163840-k_4096-act_silu-n_2048-tm_512`.
FAMILY_WIDTH = 50


def expand(title: str, group: str, groups: list[tuple[str, dict[str, float]]],
           step: float) -> list[str]:
    """Every op family behind each row of a table, nothing folded.

    The summary tables name the three biggest families in a row and count the
    rest, which is the right default -- a category can have ninety members. This
    prints all of them, in one flat table with the parent row as a column, so it
    stays greppable and lines up with the table it expands.

    Time only, no instance count. The count used to print here and got read as
    structure -- *this many of them, so one per layer of that kind* -- which is
    the inference this tool no longer makes anywhere. `--families` still prints
    it, where the neighbouring column is us/call and the reading is unambiguous.
    """
    rows: list[list[object]] = []
    for name, members in groups:
        for i, (fam, value) in enumerate(
                sorted(members.items(), key=lambda kv: -kv[1])):
            # The label prints only on the row's first line; the lookup still
            # uses the real name, which is why the two are separate.
            rows.append([name if i == 0 else "", fam[:FAMILY_WIDTH],
                         us(value), pct(100 * value / step)])
    if not rows:
        return []
    return ["", title, "", *table([group, "op family", "us/step", "% step"], rows,
                                  widths=[CATEGORY_WIDTH, FAMILY_WIDTH, 11, 8],
                                  align="<<>>")]


#: Width of the source-location column. Long enough for
#: `vllm_torchtpu/kernels/quantized_matmul/util.py:308`.
SOURCE_WIDTH = 54

#: A no-source category below this share of the step folds into one row. The
#: point of splitting the remainder is to show it is a couple of known things,
#: not to print a second long tail.
NO_SOURCE_FOLD_PCT = 0.10


def elide(path: str, width: int = SOURCE_WIDTH) -> str:
    """Drop leading directories until a path fits, marking what was dropped.

    Cuts on separators, never mid-component: `.../vllm/x.py:3` reads as a path,
    `chtpu/src/vllm/x.py:3` reads as a typo. The filename and line always
    survive -- they are the part anyone actually wants -- even if that means
    overflowing the column.
    """
    if len(path) <= width:
        return path
    parts = path.split("/")
    while len(parts) > 1:
        parts = parts[1:]
        candidate = "…/" + "/".join(parts)
        if len(candidate) <= width:
            return candidate
    return "…/" + parts[-1]


def render_sources(d: dict, step: float, limit: int,
                   verbose: bool = False) -> list[str]:
    """Top source lines by device time, with the categories each produced."""
    sources = d.get("sources", {})
    if not sources:
        # Asked for and not available: say which of the two reasons it is,
        # rather than printing nothing and looking like the flag was ignored.
        return ["", "### by source line",
                "",
                "  No op in this capture carries a source line. Either the "
                "profiler wrote no `*.op_stats_v2.pb`",
                "  sidecar beside the xplane, or the `xprof` package that "
                "reads it is not installed. Every other",
                "  table is unaffected; `src:` rules in the taxonomy cannot "
                "fire. See xprofkit/opstats.py."]
    shown = list(sources.items())[:limit]
    rows = [
        [elide(loc), us(value), pct(100 * value / step),
         name_families(d["source_categories"].get(loc, {}), 3, width=34)]
        for loc, value in shown
    ]
    rest = sum(sources.values()) - sum(v for _, v in shown)
    if rest > 0:
        rows.append([f"(+{len(sources) - len(shown)} more lines)", us(rest),
                     pct(100 * rest / step), ""])
    # The no-source remainder, split by category rather than left as one
    # anonymous row. On this stack it is a quarter of the step, and printing
    # that as a single blank line reads as "a quarter is unexplained" when what
    # it actually is, is SPMD reduce-scatter and the copies around it.
    no_src_cats = d.get("no_source_categories", {})
    folded = 0.0
    for cat, value in no_src_cats.items():
        if 100 * value / step < NO_SOURCE_FOLD_PCT:
            folded += value
            continue
        rows.append([f"(compiler-generated) {cat}", us(value),
                     pct(100 * value / step),
                     name_families(d["no_source_families"].get(cat, {}), 3,
                                   width=34)])
    if folded > 0:
        rows.append(["(compiler-generated) everything else", us(folded),
                     pct(100 * folded / step), ""])

    detail = expand(
        "#### every op family, by source line", "file:line",
        [(elide(loc, CATEGORY_WIDTH), d["source_families"].get(loc, {}))
         for loc, _ in shown],
        step) if verbose else []

    return ["", f"### by source line — top {limit}, whose code asked for it", "",
            *table(["file:line", "us/step", "% step", "categories"], rows,
                   widths=[SOURCE_WIDTH, 11, 8, 34], align="<>><"),
            *detail,
            "",
            "  From the op-stats sidecar, not the xplane. The "
            "`(compiler-generated)` rows are ops left with no source line:",
            "  SPMD collectives and the copies around them, whose recorded "
            "stack is the compiler's own — a collective is functionalised into",
            "  the graph at trace time, so the model frame that wanted it is "
            "gone and every collective in the module has the same",
            "  stack. They are already counted in the category table above; "
            "this table only says no code of yours asked for them.",
            "",
            "  Each row is the innermost frame that is not dispatch plumbing, "
            "not the innermost frame — that one is",
            "  the Pallas call wrapper for half the step and names nobody. A "
            "kernel's setup glue lands on its caller."]


def src_rule_warning(taxonomy: Taxonomy, sourced: bool) -> list[str]:
    """Warn that the taxonomy's ``src:`` rules cannot fire on this capture.

    Printed whether or not source lines were *asked* for, which is the point.
    A capture with no ``*.op_stats_v2.pb`` beside its xplane -- the profiler
    writes one only when it is configured to -- classifies every op with an
    empty source, so every ``src`` rule misses and the ops fall through to a
    broader name rule. Nothing else on the page changes: the totals still add
    up, ``unclassified`` stays small, and the only symptom is a category that
    is smaller than it was last week. The old warning lived inside the
    ``--sources`` table, so the report that most needed it was the one that
    never printed it.
    """
    rules = taxonomy.src_rules
    if sourced or not rules:
        return []
    categories = ", ".join(dict.fromkeys(rule.category for rule in rules))
    return [
        "",
        f"**WARNING** — no op in this capture carries a source line, so the "
        f"{len(rules)} `src:` rule(s) of taxonomy `{taxonomy.name}` cannot fire:",
        f"  {categories}.",
        "  Those ops fall through to whichever name rule catches them next, so "
        "the categories above are under-reported and the",
        "  ones that absorbed them are over-reported. The capture has no "
        "`*.op_stats_v2.pb` sidecar, or `xprof` is not installed",
        "  to read it; either way this capture is not comparable against one "
        "that has sources.",
    ]


def dead_rule_note(taxonomy: Taxonomy, dead: list[dict],
                   absent: str = "this capture does not contain") -> list[str]:
    """Name the individual rules that claimed no op in this capture.

    Nothing else on the page can say it. A category with no time is dropped
    from every table by construction -- printing a screenful of 0.00% rows
    would be worse -- so a rule that fires zero times is invisible in the
    category table, in ``-v``, and in ``--json`` alike. The empty *block* that
    does print is coarser by an order of magnitude and cannot appear while any
    sibling category in that block is alive.

    Per **rule**, not per category, and that is the whole point of the check.
    Most categories in the shipped file are written by more than one rule --
    ``norm``, ``matmul_other``, ``quant``, ``attn_core`` -- so a category-level
    test sees nothing when one of them dies. Change ``norm``'s ``src:`` from
    ``layernorm\\.py`` to ``layer_norm\\.py``, one character, and the category
    falls from 1,567.8 to 9.9 us/step: 0.51% of the step relocates into
    ``fusion`` and ``layout``, the ``norm`` row still prints, and a
    category-level note still reports nothing wrong.

    Matching is first-match, so the way to kill a rule is to put it below a
    broader one, which is the single most likely mistake when copying a
    taxonomy for a new model: on the shipped qwen file ``attn_core``'s
    ``^rpa_`` claimed ``rpa_metadata_schedule`` out from under a rule that had
    been written for it, and the row simply never appeared.

    ``dead`` comes from the analysis, not from re-deriving it here -- see
    ``dead_rules`` in :func:`analyse_phase` -- so ``--json`` and the page agree.

    ``absent`` is the clause naming what the rule found no work in, because
    ``diff`` prints this too and there the answer is *neither* of two captures.
    """
    if not dead:
        return []
    named = ", ".join(f"categories[{item['index']}] `{item['category']}`"
                      for item in dead)
    return [
        "",
        f"  {len(dead)} rule(s) of taxonomy `{taxonomy.name}` claimed no op "
        f"here: {named}.",
        "  A rule, not a category: the category may still be printing above on "
        "the strength of a sibling rule.",
        "  First match wins, so a rule under a broader one never fires and its "
        "ops are counted in the broader row —",
        "  and in that row's block. Move it above whichever rule shadows it, or "
        "delete it. A rule for work",
        f"  {absent} is dead for a different and harmless reason; the two look "
        "the same here.",
    ]


def one_sided_rule_note(taxonomy: Taxonomy, one_sided: list[dict],
                        base: str, cand: str) -> list[str]:
    """Name rules that fired on one capture and not on the other.

    Deliberately *not* folded into the dead count. Dead means the rule matched
    nothing anywhere, and the reader's next move is to edit the taxonomy. This
    is the opposite finding: the rule works, and the two captures ran different
    op sets, which is a fact about the change under review and is very often
    the answer the diff was run to get. Folding the two together would report a
    taxonomy defect every time a candidate build stopped emitting an op.

    It also warns about a delta that is easy to misread. A category alive on
    one side only has no counterpart row to move against, so its delta is the
    whole of one side rather than a shift inside a row both captures share.

    Lives beside :func:`dead_rule_note` although only ``diff`` calls it: the
    four note emitters in this module are one unit about one subject and are
    worth moving to a module of their own together, not one at a time.
    """
    if not one_sided:
        return []
    label = {"base": base, "cand": cand}
    named = ", ".join(
        f"categories[{item['index']}] `{item['category']}` "
        f"({label[item['alive']]} only)" for item in one_sided)
    return [
        "",
        f"  {len(one_sided)} rule(s) of taxonomy `{taxonomy.name}` fired on one "
        f"capture and not the other: {named}.",
        "  This is not a dead rule and needs no edit to the taxonomy — the two "
        "captures ran different ops, which",
        "  is usually the finding. A category present on one side only has "
        "nothing to move against, so its delta",
        "  is the whole of that side rather than a shift within a shared row.",
    ]


def taxonomy_notes(taxonomy: Taxonomy) -> list[str]:
    """Whatever :meth:`~xprofkit.taxonomy.Taxonomy.load` found odd but allowed.

    Loading raises on anything that is simply wrong. This is the other kind:
    a file that is legal, means something, and probably does not mean what was
    intended -- where correcting it would mean editing the user's names.
    """
    if not taxonomy.notes:
        return []
    return ["", *(f"  {note}" for note in taxonomy.notes)]


def dispute_note(disputed: dict[str, list[str]], verbose: bool = False) -> list[str]:
    """How many ops lost a field because their ranks disagreed about it.

    None of the time is lost -- a disputed op is still counted, in full, under
    whatever its remaining fields classify it as. What is lost is the field, and
    with it every ``hlo:`` or ``src:`` rule that needed it, so a stack where the
    ranks diverge widely would quietly lose rule coverage and show only a
    category that had shrunk. Six ops on this capture, all of them genuinely
    shaped differently on the last rank.
    """
    counts = {field: names for field, names in disputed.items() if names}
    if not counts:
        return []
    listed = ", ".join(f"{len(names)} lost `{field}`"
                       for field, names in sorted(counts.items()))
    lines = ["", f"  ranks disagreed about {listed}: those ops keep their time "
                 "but not the field, so a rule needing it cannot fire (-v names "
                 "them)."]
    if verbose:
        for field, names in sorted(counts.items()):
            lines.append(f"    {field}: " + ", ".join(names[:12])
                         + (" ..." if len(names) > 12 else ""))
    return lines


def render_blocks(blocked: dict, taxonomy: Taxonomy, step: float,
                  verbose: bool = False) -> list[str]:
    """Block subtotals, and the sentence saying where the blocks came from.

    That sentence is not decoration. The table looks derived -- it is made of
    measured microseconds and it sums to the step -- and the one thing it is
    not is derived: the split between the rows is a line of YAML. A reader who
    does not know that will read a wrong `block:` as a measurement.

    ``blocked`` is the analysed phase, which carries the keys
    :func:`analyse_blocks` produced. This renders; it does not compute.
    """
    blocks = blocked["blocks"]
    if not blocks:
        return []
    names = list(blocks)

    lines = ["", "### by block — declared by the taxonomy, summed from the trace", ""]
    rows = []
    total = blocked["unassigned_us"]
    for name in names:
        info = blocks[name]
        total += info["total_us"]
        rows.append([
            name, us(info["total_us"]), pct(100 * info["total_us"] / step),
            name_families(dict(sorted(info["categories"].items(),
                                      key=lambda kv: -kv[1])), 3, width=34),
        ])
    rows.append([UNASSIGNED, us(blocked["unassigned_us"]),
                 pct(100 * blocked["unassigned_us"] / step),
                 name_families(dict(sorted(blocked["unassigned"].items(),
                                           key=lambda kv: -kv[1])), 3, width=34)])
    rows.append([TOTAL, us(total), pct(100 * total / step), ""])
    lines += table(["block", "us/step", "% step", "categories"], rows,
                   widths=[CATEGORY_WIDTH, 11, 8, 34], align="<>><")
    lines += [
        "",
        "  **Declared, not derived.** A row is the sum of the categories whose "
        "taxonomy rule carries that `block:`, and",
        "  nothing in the trace says which part of the model an op belongs to "
        "— there are no layer boundaries in a step,",
        "  the compiler names glue for what it does rather than for whose it "
        "is, and a collective's source line is the",
        "  compiler's own stack. So each row is exactly as right as the rules "
        "behind it (the `categories` column names",
        "  the largest), and a change of sharding can move work between blocks "
        "with nothing here noticing. Before",
        f"  quoting a number from this table, read those rules in "
        f"`{taxonomy.name}` — README, 'Declaring blocks'.",
        f"  **{UNASSIGNED}** is the {100 * blocked['unassigned_us'] / step:.1f}% "
        "in categories no rule assigned to one; the rows do not nest and they "
        "sum to the step.",
    ]

    if verbose:
        # Keyed by family *and* category, because that is the granularity the
        # blocks were built at: one family can appear in two blocks under two
        # categories, and merging them here would misreport both.
        groups = [
            (name, {f"{c['family']} [{c['category']}]": c["us"]
                    for c in blocks[name]["cells"]})
            for name in names
        ]
        groups.append((UNASSIGNED,
                       {f"{c['family']} [{c['category']}]": c["us"]
                        for c in blocked["unassigned_cells"]}))
        lines += expand("#### every op family, by block", "block", groups, step)
    return lines


def render(result: dict, taxonomy: Taxonomy, *, families: int = 0,
           unclassified: int = 8, names: int = 3, sources: int = 0,
           units: bool = False, verbose: bool = False) -> list[str]:
    """Two tables by default -- categories, then the blocks they roll up into.

    Everything else is opt-in. The default output is the one anybody asks for
    first, and a report you have to skim to find the answer in is a worse
    report than a short one plus a flag.

    ``verbose`` follows each table with every op family behind each of its
    rows, nothing folded. The summary tables name three and count the rest;
    this is for when the question is *which* op, not how much.
    """
    lines: list[str] = []
    for phase, d in result.items():
        step = d["step_us"]
        if not step:
            continue
        header = (f"step {us(step)} us · ops {us(d['busy_us'])} us · "
                  f"occupancy {100 * (d['occupancy'] or 0):.1f}% · "
                  f"{d['steps']} steps over rank(s) {','.join(d['rank_ids'])}")
        if d["async_us"]:
            # Overlapped, so it is never summed into a "% of step" column --
            # but worth one clause, because a large value is what says the
            # collectives are being hidden rather than waited on.
            header += (f" · async {us(d['async_us'])} us "
                       f"({d['async_us'] / step:.2f}x, overlapped)")
        lines += ["", f"## {phase}", "", header]
        for name in d["dropped"]:
            lines.append(f"    dropped partial dump: {name}")
        lines.append("")

        rows = []
        for cat in taxonomy.order:
            value = d["categories"].get(cat, 0.0)
            if value:
                rows.append([
                    cat, us(value), pct(100 * value / step),
                    name_families(d.get("category_families", {}).get(cat, {}), names),
                ])
        rows.append([TOTAL, us(d["busy_us"]), pct(100 * d["busy_us"] / step), ""])
        lines += table(["category", "us/step", "% step", "op families"], rows,
                       widths=[CATEGORY_WIDTH, 11, 8, NAMES_WIDTH], align="<>><")
        if verbose:
            lines += expand(
                "#### every op family, by category", "category",
                [(cat, d["category_families"].get(cat, {}))
                 for cat in taxonomy.order if d["categories"].get(cat)],
                step)

        devices = d.get("devices", {})
        if units and devices:
            lines += ["", "### by execution unit — same total, split a different way", ""]
            rows = [
                [dev, us(value), pct(100 * value / step),
                 name_families(d.get("device_families", {}).get(dev, {}), names)]
                for dev, value in sorted(devices.items(), key=lambda kv: -kv[1])
            ]
            lines += table(["unit", "us/step", "% step", "op families"], rows,
                           widths=[CATEGORY_WIDTH, 11, 8, NAMES_WIDTH], align="<>><")
            if devices.get(SPARSECORE_THREAD):
                lines.append(
                    "  SparseCore has no VPU and its own bandwidth, so these ops "
                    "must be judged against the SparseCore roofline.")
            if verbose:
                lines += expand(
                    "#### every op family, by execution unit", "unit",
                    [(dev, d["device_families"].get(dev, {}))
                     for dev in sorted(devices, key=lambda k: -devices[k])],
                    step)

        if sources:
            lines += render_sources(d, step, sources, verbose=verbose)

        if taxonomy.blocks:
            lines += render_blocks(d, taxonomy, step, verbose=verbose)

        if families:
            lines += ["", f"### top {families} op families", ""]
            ordered = sorted(d["families"].items(), key=lambda kv: -kv[1]["us"])
            fam_cat = family_categories(d["cells"])
            rows, folded = [], 0.0
            for name, info in ordered:
                folds = not verbose and 100 * info["us"] / step < FOLD_BELOW_PCT
                if len(rows) >= families or folds:
                    folded += info["us"]
                    continue
                rows.append([name[:44], fam_cat.get(name, UNCLASSIFIED), info["n"],
                             us(info["us"]), pct(100 * info["us"] / step),
                             us(info["us"] / info["n"], 2) if info["n"] else ""])
            if folded:
                rows.append(["(everything else)", "", "", us(folded),
                             pct(100 * folded / step), ""])
            lines += table(
                ["op family", "category", "n", "us/step", "% step", "us/call"],
                rows, widths=[44, FAMILY_CATEGORY_WIDTH, 5, 10, 8, 9])
            lines += [
                "",
                "  `category +N` is a family the compiler gave one name and the "
                "taxonomy splits N+1 ways; the row is",
                "  the whole family and the category named is where most of its "
                "time went. `-v` prints the split.",
            ]

        lines += src_rule_warning(taxonomy, bool(d["sources"]))
        lines += dead_rule_note(taxonomy, d.get("dead_rules", []))
        lines += taxonomy_notes(taxonomy)
        lines += dispute_note(d.get("disputed", {}), verbose=verbose)

        junk = d["categories"].get(UNCLASSIFIED, 0.0)
        if junk > 0.005 * d["busy_us"]:
            lines += [
                "",
                f"**WARNING** — `{UNCLASSIFIED}` is "
                f"{100 * junk / d['busy_us']:.2f}% of op time. The taxonomy needs a "
                "rule; every percentage above is provisional until it has one.",
            ]
        if junk and unclassified:
            # From the category each op was actually filed under, not from a
            # second name-only pass: with `hlo` or `src` rules in play the two
            # disagree, and the list would name ops the table above classified.
            worst = sorted(
                ((k, v) for k, v in d["ops"].items()
                 if d["op_categories"].get(k) == UNCLASSIFIED),
                key=lambda kv: -kv[1])[:unclassified]
            lines.append("")
            lines.append(f"{UNCLASSIFIED} {us(junk)} us/step: "
                         + ", ".join(f"{k} ({v:.2f})" for k, v in worst))
    return lines
