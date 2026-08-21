#!/usr/bin/env python3
"""Tests for the parts of xprofkit that do not need a trace.

Run with ``python3 -m pytest tools/xprof_analysis/tests`` or directly with
``python3 tools/xprof_analysis/tests/test_xprofkit.py``.

Deliberately hermetic: no jax, no protobufs, no chip. The protobuf reading is
one small function (:func:`xprofkit.trace.read_rank`) and is exercised by
running the CLI against a real capture; everything that decides what a number
*means* -- classification, block assignment, dump selection, the statistics,
and every command's exit status -- is pure and is tested here.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xprofkit import breakdown, compare, diff, steptime, trace  # noqa: E402
from xprofkit.table import describe, t95, table, welch  # noqa: E402
from xprofkit.taxonomy import (  # noqa: E402
    SPARSECORE_THREAD, TENSORCORE, UNCLASSIFIED, Rule, Taxonomy, device_of, family,
)
from xprofkit.trace import (  # noqa: E402
    Capture, PhaseTrace, RankTrace, hlo_signature, phase_dirs, rank_key,
    rank_of, select_dump, shared_ranks,
)


# --------------------------------------------------------------------------- #
def test_family_strips_the_serial_number():
    assert family("copy.18") == "copy"
    assert family("fused_conv1d.90") == "fused_conv1d"
    assert family("copy") == "copy"
    # Only one serial, and only if the last component is all digits, so a
    # number that is part of the name survives.
    assert family("reshape.11.2") == "reshape.11"


def test_family_strips_compiler_wrappers():
    # Without this, reduce-scatter fragments into hundreds of small rows and
    # the largest cost in the step disappears from every top-N list.
    assert family("reduce-scatter.399.cloned.1.call-done") == "reduce-scatter"
    assert family("reduce-scatter.399.cloned.1.call-start") == "reduce-scatter"
    assert family("prepare_start_all-gather.60.cloned.1.call-start") \
        == "prepare_start_all-gather"
    assert family("all-gather.4.cloned.2") == "all-gather"
    # Hyphenated names are single components and must not be touched.
    assert family("copy-start.7") == "copy-start"
    assert family("copy-done") == "copy-done"
    assert family("sc_ragged_gather_reduce_v2.278") == "sc_ragged_gather_reduce_v2"


def test_family_strips_the_other_spellings_of_a_clone():
    # `broadcast_in_dim.1697.clone` is 44 of the 45 copies of one instruction.
    # Kept apart from the 45th it is two rows in every table, each of them too
    # small to appear in a top-N list that the merged row would have made.
    assert family("broadcast_in_dim.1697.clone") == "broadcast_in_dim"
    assert family("broadcast_in_dim.1697.clone.3") == "broadcast_in_dim"
    assert family("iota.1141.clone.remat") == "iota"


def test_rank_of():
    assert rank_of("rank12_t1v-n-abc.xplane.pb") == "12"
    assert rank_of("weird.xplane.pb") == "weird.xplane.pb"
    # Numeric order, not lexicographic: rank 2 before rank 10, wherever a rank
    # set is printed for the user to copy into `--ranks`.
    assert sorted(["10", "2", "0", "odd"], key=rank_key) == ["0", "2", "10", "odd"]


def test_taxonomy_classifies_and_orders():
    tax = Taxonomy.load()
    assert tax.classify("fused_conv1d_gdn_per_seq.90") == "linear_attention"
    assert tax.classify("RPAm-p256-b1-q256-k256") == "attention"
    assert tax.classify("gmm_v2-bf16") == "moe"
    assert tax.classify("sort.31") == "sort"
    assert tax.classify("all-reduce.4") == "collective"
    assert tax.classify("copy.18") == "layout"
    assert tax.classify("add.7") == "elementwise"
    assert tax.classify("totally-made-up-op") == UNCLASSIFIED
    assert tax.order[-1] == UNCLASSIFIED


def test_hlo_signature_keeps_what_survives_a_recompile():
    qkv = ('%multiply_convert_fusion.44 = bf16[2048,17408]{1,0:T(8,128)(2,1)} '
           'fusion(f8e4m3fn[2048,4096]{1,0:T(32,128)(4,1)} %param.7, '
           'f8e4m3fn[17408,4096]{1,0:T(32,128)(4,1)} %param.8), kind=kOutput, '
           'calls=%fused_computation.99')
    sig = hlo_signature(qkv)
    assert "kind=kOutput" in sig
    assert "out=bf16[2048,17408]" in sig
    assert "in=f8e4m3fn[2048,4096],f8e4m3fn[17408,4096]" in sig
    # Operand *names* are build-local serials and must not leak in, or a rule
    # written against one compile would stop matching after the next.
    assert "param" not in sig and "fused_computation" not in sig
    # The layout suffix `{1,0:T(8,128)}` is not a shape and must not be read
    # as one.
    assert "[8,128]" not in sig

    started = ('%reduce-scatter.399.call-start = f32[2048]{0:T(1024)} '
               'reduce-scatter-start(f32[16384]{0:T(1024)} %bitcast.9), '
               'async_execution_thread="sparsecore"')
    assert "thread=sparsecore" in hlo_signature(started)
    assert hlo_signature("") == ""


def test_device_of_is_orthogonal_to_the_category():
    assert device_of("kind=kOutput thread=sparsecore out=f32[2048]") \
        == SPARSECORE_THREAD
    assert device_of("kind=kOutput out=f32[2048]") == TENSORCORE
    assert device_of("") == TENSORCORE


def test_rule_requires_every_condition_it_declares():
    import re
    both = Rule(category="c", pattern=re.compile("fusion"),
                hlo=re.compile("kind=kOutput"))
    assert both.matches("multiply_fusion.4", "kind=kOutput out=f32[2,2]")
    assert not both.matches("multiply_fusion.4", "kind=kLoop out=f32[2,2]")
    assert not both.matches("copy.4", "kind=kOutput out=f32[2,2]")
    # An hlo-only rule fires on the signature alone -- the only way to catch a
    # matmul XLA named after its elementwise interior.
    hlo_only = Rule(category="c", hlo=re.compile("kind=kOutput"))
    assert hlo_only.matches("multiply_convert_fusion.44", "kind=kOutput")
    assert not hlo_only.matches("multiply_convert_fusion.44", "")
    # A src-only rule fires on the source line alone -- the only way to tell two
    # ops the compiler gave the same anonymous name apart.
    src_only = Rule(category="c", src=re.compile(r"moe_routing\.py"))
    assert src_only.matches("fusion.7", "", "vllm/moe_routing.py:165")
    assert not src_only.matches("fusion.7", "", "vllm/layernorm.py:22")
    # Omitting the source degrades to not matching, never to matching anyway.
    assert not src_only.matches("fusion.7")
    # A rule with no condition would match everything; it must match nothing.
    assert not Rule(category="c").matches("anything", "anything", "anything")


def _write_taxonomy(text: str) -> Taxonomy:
    """Load a taxonomy from a YAML string, via a real file, as the tool does."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.yaml"
        path.write_text(text)
        return Taxonomy.load(path)


def test_src_rules_split_a_name_the_compiler_reused():
    """Two anonymous fusions, same name, different code. Only src separates them."""
    tax = _write_taxonomy("""
name: srctest
categories:
  - {name: norm,        src: 'layernorm\\.py'}
  - {name: quant,       src: 'quantized_matmul/util\\.py'}
  - {name: fusion,      pattern: 'fusion'}
""")
    assert tax.classify("fusion.7", "", "vllm/layers/layernorm.py:256") == "norm"
    assert tax.classify("fusion.9", "", "kernels/quantized_matmul/util.py:308") == "quant"
    # No source recorded: falls through to the name rule rather than failing.
    assert tax.classify("fusion.9") == "fusion"


def _rejects(text: str, expected: str) -> None:
    """Assert that loading *text* fails with a message naming *expected*."""
    try:
        _write_taxonomy(text)
    except ValueError as exc:
        assert expected in str(exc), f"wrong complaint: {exc}"
        return
    raise AssertionError(f"loaded a taxonomy that should have been rejected "
                         f"({expected})")


def test_taxonomy_rejects_what_would_otherwise_fail_silently():
    """Every one of these has no symptom at all if it is allowed through."""
    # A misspelt condition key leaves a rule with no condition, matching nothing.
    _rejects("categories: [{name: a, patern: 'x'}]", "unknown field")
    _rejects("categories: [{name: a}]", "declares none of")
    _rejects("categories: [{name: a, pattern: '(unclosed'}]", "not a valid regex")
    # A `name` is a row heading, and gets the same scrutiny as a `block` for the
    # same reason: whatever it is, it reaches the renderer as one. All four of
    # these loaded clean and printed something -- a row headed `60`, a row with
    # no heading at all, two rows for one category, and (worst) the residue
    # category twice, followed by a coverage warning listing ops a rule had
    # classified.
    _rejects("categories: [{pattern: 'x'}]", "`name` is None")
    _rejects("categories: [{name: 60, pattern: 'x'}]", "`name` is 60")
    _rejects("categories: [{name: '   ', pattern: 'x'}]", "`name` is '   '")
    _rejects("categories: [{name: ' layout ', pattern: 'x'}]",
             "which has surrounding whitespace; write 'layout'")
    # The renderer spends three row labels on rows of its own. Each was worth
    # rejecting on its own evidence: `unclassified` printed the residue category
    # twice and warned that 16.50% of op time needed a rule while listing ops
    # that rule had classified, and `name: TOTAL, block: TOTAL` printed two
    # TOTAL rows in the category table -- 11,912.7 us above the real 306,221.0
    # -- and two more in the block table. They are one list now so that a fourth
    # cannot be guarded in one field and forgotten in the other.
    _rejects("categories: [{name: unclassified, pattern: '^sort'}]",
             "which is where this tool files ops no rule matched")
    _rejects("categories: [{name: '(no block)', pattern: 'x'}]",
             "which is what this tool prints for categories in no block")
    _rejects("categories: [{name: TOTAL, pattern: 'x'}]",
             "which is the summary row at the foot of every table")
    # A condition YAML read as structure. `[copy]` is a character class and the
    # natural unquoted spelling of one; `re.compile` raises TypeError for it,
    # which the `except re.error` here does not catch and `analyze.py` does not
    # either, so it reached the user as a traceback naming no file and no rule.
    # The hint quotes back the offending value, not a stock example: it used to
    # advise ``quote it, `src: '[copy]'` `` for every condition of every type,
    # naming the right field and the wrong value, and reading as though
    # `[copy]` were the fix.
    _rejects("categories: [{name: cpy, pattern: [copy]}]",
             "`pattern` is a list (['copy']), not a string")
    _rejects("categories: [{name: cpy, pattern: [copy]}]",
             "quote it, `pattern: '[copy]'`")
    _rejects("categories: [{name: cpy, src: 12}]", "`src` is a int (12)")
    _rejects("categories: [{name: cpy, src: 12}]", "quote it, `src: '12'`")
    # A misspelt *section* is the same mistake one level up, and it deletes a
    # whole table with nothing on screen naming the typo.
    _rejects("layer_multiplicity: {attn: 15}", "unknown section")
    _rejects("categorys: [{name: a, pattern: 'x'}]", "unknown section")
    # A block is a label, and these are what somebody writes who thinks the
    # field is still the old layer count. An int reaches the renderer as a row
    # heading of `60`.
    _rejects("categories: [{name: a, pattern: 'x', block: 60}]", "`block` is 60")
    _rejects("categories: [{name: a, pattern: 'x', block: yes}]", "`block` is True")
    _rejects("categories: [{name: a, pattern: 'x', block: '  '}]", "`block` is '  '")
    # One category, two blocks. The table groups by category, so the row would
    # land wherever the first rule happened to be written.
    _rejects("""
categories:
  - {name: proj, block: attn, pattern: 'qkv'}
  - {name: proj, block: moe,  pattern: 'gate'}
""", "is in block 'attn' per categories[0] and block 'moe' per categories[1]")
    # Blocked against UNblocked is the same disagreement and the likelier one:
    # it is what a half-finished edit looks like. While it was allowed through,
    # adding `block: attn` to one of the two `quant` rules of the shipped qwen
    # taxonomy moved 6.11% of the step into `attn` -- a 2.4x change in that row
    # -- with the category table byte-identical either way. Both orders, since
    # neither rule is privileged.
    _rejects("""
categories:
  - {name: quant, block: attn, pattern: 'qkv'}
  - {name: quant, pattern: 'gate'}
""", "is in block 'attn' per categories[0] and no block per categories[1]")
    _rejects("""
categories:
  - {name: quant, pattern: 'gate'}
  - {name: quant, block: attn, pattern: 'qkv'}
""", "is in no block per categories[0] and block 'attn' per categories[1]")
    # Every check `name` gets, because a block is the same kind of thing: a row
    # heading and a join key. Padding used to be silently stripped here, which
    # made the loader the one place that rewrote what the user wrote.
    _rejects("categories: [{name: a, pattern: 'x', block: ' moe '}]",
             "which has surrounding whitespace; write 'moe'")
    _rejects("categories: [{name: a, pattern: 'x', block: unclassified}]",
             "which is where this tool files ops no rule matched")
    _rejects("categories: [{name: a, pattern: 'x', block: '(no block)'}]",
             "which is what this tool prints for categories in no block")
    _rejects("categories: [{name: a, pattern: 'x', block: TOTAL}]",
             "which is the summary row at the foot of every table")
    # Shape mistakes. Each of these used to raise a bare TypeError from inside
    # the load loop -- which `analyze.py` does not catch, so it reached the user
    # as a stack trace naming neither the file nor the line.
    _rejects("categories: {name: a, pattern: 'x'}", "`categories` is a dict")
    _rejects("categories: [null]", "is a NoneType")
    _rejects("categories: ['name: a']", "is a str")
    _rejects("- just a list", "the file is a list")
    # The error says which entry, because a taxonomy has dozens.
    try:
        _write_taxonomy("categories: [{name: a, pattern: 'x'}, {name: b}]")
    except ValueError as exc:
        assert "categories[1]" in str(exc)


def test_shipped_taxonomies_load():
    """The demo taxonomy is a demo, but a broken one is a broken example."""
    for path in sorted((Path(__file__).resolve().parent.parent
                        / "taxonomies").glob("*.yaml")):
        tax = Taxonomy.load(path)
        assert tax.rules, f"{path} has no rules"


def test_opstats_finds_the_sidecar_beside_the_xplane():
    from xprofkit import opstats
    with tempfile.TemporaryDirectory() as tmp:
        xplane = Path(tmp) / "rank0_host.xplane.pb"
        xplane.write_bytes(b"")
        assert opstats.op_stats_path(str(xplane)) is None  # no sidecar written
        sidecar = Path(tmp) / "rank0_host.op_stats_v2.pb"
        sidecar.write_bytes(b"")
        assert opstats.op_stats_path(str(xplane)) == str(sidecar)
        assert opstats.op_stats_path(str(sidecar)) is None  # not an xplane


def test_opstats_shortens_a_path_to_what_identifies_it():
    from xprofkit import opstats
    assert opstats.shorten("/x/y/site-packages/torch/_ops.py") == "torch/_ops.py"
    assert opstats.shorten("/home/me/workspace/vllm/vllm/x.py") == "vllm/vllm/x.py"
    # Nothing recognised: left alone rather than truncated to a guess.
    assert opstats.shorten("/opt/thing.py") == "/opt/thing.py"


def test_caller_of_walks_past_the_dispatch_plumbing():
    """The innermost frame of a Pallas op is the wrapper; the owner is above it."""
    from xprofkit import opstats
    stack = "\n".join([
        "/x/site-packages/torch_tpu/_internal/pallas/pallas.py:638:0",
        "/x/site-packages/torch/_library/custom_ops.py:502:0",
        "/x/site-packages/torch/_dynamo/eval_frame.py:1446:0",
        "/home/me/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/fused_moe.py:239:0",
        "/x/site-packages/torch_tpu/_internal/export/export.py:188:0",
    ])
    assert opstats.caller_of(stack) == (
        "vllm-torchtpu/src/vllm_torchtpu/layers/vllm/fused_moe.py:239")
    # The column is dropped: keeping it splits one call site into several rows.
    assert opstats._location("a/b.py:239:0") == "a/b.py:239"
    assert opstats._location("a/b.py:239") == "a/b.py:239"
    # All plumbing: no owner, and saying so beats naming a dispatch function.
    assert opstats.caller_of("/x/site-packages/torch/_compile.py:54:0") is None
    assert opstats.caller_of("") is None


def test_elide_cuts_a_path_on_separators():
    long = "a/bbbbbbbbbb/cccccccccc/dddddddddd/file.py:12"
    assert breakdown.elide(long, 45) == long          # fits, untouched
    assert breakdown.elide(long, 44) == "…/cccccccccc/dddddddddd/file.py:12"
    # The filename and line always survive, even past the width.
    assert breakdown.elide(long, 5) == "…/file.py:12"


def test_hlo_rules_catch_a_matmul_the_name_hides():
    tax = Taxonomy.load()
    qkv = ("kind=kOutput out=bf16[2048,17408] "
           "in=f8e4m3fn[2048,4096],f8e4m3fn[17408,4096]")
    assert tax.classify("multiply_convert_fusion.44", qkv) == "matmul"
    # Same kind, but a 1-D output: a reduction rooted at kOutput, not a matmul.
    reduce_ = ("kind=kOutput out=f32[2048] "
               "in=bf16[2048,4096],bf16[2048,4096]")
    assert tax.classify("multiply_reduce_fusion.119", reduce_) != "matmul"
    # No signature -> name-only, which is a degrade rather than a failure --
    # and is the misfiling the hlo rule exists to fix: the QKV projection
    # booked as pointwise maths.
    assert tax.classify("multiply_convert_fusion.44") == "elementwise"


def _split_taxonomy() -> Taxonomy:
    """Two blocks and a category left out of both, which is the normal case."""
    import re
    return Taxonomy(
        name="t",
        rules=(Rule("core_a", re.compile("^kernel_a"), block="a"),
               Rule("core_b", re.compile("^kernel_b"), block="b"),
               Rule("moe", re.compile("^gmm"), block="b"),
               Rule("layout", re.compile("^copy"))),
    )


def test_blocks_are_read_off_the_rules_that_declared_them():
    tax = _split_taxonomy()
    # File order, deduplicated: `b` is declared by two rules and is one block.
    assert tax.blocks == ("a", "b")
    assert tax.block_of == {"core_a": "a", "core_b": "b", "moe": "b"}
    # A category with no `block:` is in none of them, not in the first one.
    assert "layout" not in tax.block_of
    assert Taxonomy(name="t", rules=()).blocks == ()


def test_analyse_blocks_sums_the_categories_declared_into_each():
    tax = _split_taxonomy()
    data = {"cells": [
        {"family": "kernel_a", "category": "core_a", "us": 450.0},
        {"family": "kernel_b", "category": "core_b", "us": 150.0},
        {"family": "gmm", "category": "moe", "us": 600.0},
        {"family": "copy", "category": "layout", "us": 999.0},
    ]}
    out = breakdown.analyse_blocks(data, tax)
    a, b = (out["blocks"][k] for k in ("a", "b"))
    assert a["total_us"] == 450.0
    # Two categories declared into one block are one row, and nothing but the
    # rules decides that: the cells carry time and no count at all.
    assert b["total_us"] == 750.0
    assert b["categories"] == {"core_b": 150.0, "moe": 600.0}
    assert "moe" not in a["categories"]
    # A category no rule assigned keeps its own row rather than being spread
    # over blocks it may have nothing to do with.
    assert out["unassigned_us"] == 999.0
    assert out["unassigned"]["layout"] == 999.0
    # Every microsecond lands somewhere exactly once.
    assert a["total_us"] + b["total_us"] + out["unassigned_us"] == 2199.0


def test_block_table_says_the_split_was_declared_not_measured():
    """The one line that stops the table being read as a measurement.

    Every number in it is measured and it sums to the step, so it looks
    derived. The split between the rows is not: it is a line of YAML, and a
    reader who does not know that reads a wrong `block:` as a finding.
    """
    tax = _split_taxonomy()
    data = {"cells": [
        {"family": "kernel_a", "category": "core_a", "us": 450.0},
        {"family": "copy", "category": "layout", "us": 550.0},
    ]}
    text = "\n".join(breakdown.render_blocks(
        breakdown.analyse_blocks(data, tax), tax, step=1000.0))
    assert "Declared, not derived" in text
    # No instance count and no per-layer divisor: both were the arity design,
    # and a column of them is what made the table look derived.
    assert "count" not in text and "us/block" not in text
    assert "(no block)" in text and "55.0%" in text
    # A declared block that nothing landed in still prints, at 0.0. That is a
    # signal about a *block*, which is coarser than a rule by an order of
    # magnitude and cannot fire at all while any sibling rule in the block is
    # alive -- `dead_rule_note` is the one that names rules.
    empty = [line for line in text.splitlines() if line.startswith("| b ")]
    assert len(empty) == 1 and "0.0" in empty[0], text


def test_a_dead_rule_is_named_even_when_its_category_is_alive():
    """A dead rule has no other symptom anywhere in the report.

    A zero-time category is dropped from every table by construction, so a rule
    that never fires is invisible in the category table, under `-v` and in
    `--json` alike -- and first-match-wins makes shadowing the likeliest
    mistake when someone copies a taxonomy for a new model. It shipped in
    `qwen35_397b.yaml`: `attn_core`'s `^rpa_` claimed `rpa_metadata_schedule`
    out from under a rule written for it, and that row never appeared.

    Detection is per *rule*, which is the whole point. Most categories worth
    writing are several rules -- `norm` is two in the shipped file,
    `matmul_other` three -- and a category-level check is blind to every death
    but the last. Here the `norm` pattern rule fires, the `norm` src rule does
    not, and the `norm` row prints either way.
    """
    tax = _write_taxonomy("""
name: t
categories:
  - {name: norm,   pattern: 'rms_norm'}
  - {name: norm,   src: 'layer_norm\\.py'}
  - {name: layout, pattern: '^copy'}
""")
    phase = PhaseTrace(phase="p", ranks=(
        RankTrace("0", "/x", "step", (100.0,),
                  {"rms_norm.1": 40.0, "copy.2": 60.0},
                  op_src={"rms_norm.1": "vllm/layernorm.py:9"}),))
    d = breakdown.analyse_phase(phase, tax)
    # In the analysis, so `--json` carries it and the page is not the only copy.
    assert d["dead_rules"] == [{"index": 1, "category": "norm"}]
    text = "\n".join(breakdown.render({"p": d}, tax))
    assert "| norm" in text, text
    assert "1 rule(s)" in text and "categories[1] `norm`" in text, text
    # Nothing dead, nothing said.
    live = _write_taxonomy("categories: [{name: layout, pattern: '.'}]")
    assert "claimed no op" not in "\n".join(breakdown.render(
        {"p": breakdown.analyse_phase(phase, live)}, live))


def test_diff_says_dead_on_both_sides_and_alive_on_one_differently():
    """`diff` classified through the same taxonomy and said nothing about it.

    `breakdown` printed the dead-rule note and the load-time notes; `diff`
    printed neither, on the same file, though the shadowing that moves a
    breakdown row moves a DELTA -- which is the number a go/no-go is made on.

    The two-capture case splits in two, and conflating them would be the worse
    bug of the pair. Dead on both sides is a defect in the taxonomy and the fix
    is to edit it. Alive on one side is a fact about the change under review --
    the candidate stopped emitting an op -- and is very often the finding
    itself; reporting it as a dead rule would send the reader to edit a file
    that is correct.
    """
    tax = _write_taxonomy("""
name: t
categories:
  - {name: norm,   pattern: 'rms_norm'}
  - {name: sort,   pattern: '^sort'}
  - {name: absent, pattern: 'nothing_here'}
  - {name: Norm,   pattern: 'never_either'}
""")
    def _phase(ops):
        return PhaseTrace(phase="p", ranks=(
            RankTrace("0", "/x", "step", (100.0,), ops),))
    base = Capture("A", "/a", {"p": _phase({"rms_norm.1": 40.0, "sort.2": 10.0})})
    cand = Capture("B", "/b", {"p": _phase({"rms_norm.1": 44.0})})
    d = diff.diff(base, cand, tax)
    phase = d["phases"]["p"]
    # In the result, so `--json` carries both lists and the page is not the
    # only copy -- the rule `breakdown` already follows.
    assert phase["dead_rules"] == [{"index": 2, "category": "absent"},
                                   {"index": 3, "category": "Norm"}]
    assert phase["one_sided_rules"] == [
        {"index": 1, "category": "sort", "alive": "base"}]
    text = "\n".join(diff.render(d, taxonomy=tax))
    assert "2 rule(s) of taxonomy `t` claimed no op here" in text, text
    assert "categories[2] `absent`" in text and "categories[3] `Norm`" in text
    # The one-sided message cannot be read as the dead one: different count,
    # different verb, and it says outright that nothing needs editing.
    assert "1 rule(s) of taxonomy `t` fired on one capture and not the other" \
        in text, text
    assert "categories[1] `sort` (A only)" in text, text
    assert "not a dead rule" in text
    # Neither says "this capture", which would name one of two.
    assert "this capture does not contain" not in text
    assert "neither capture contains is dead" in text
    # And the load-time notes ride along, once for the page rather than once
    # per phase: `norm` beside `Norm` is why the delta is split across two rows.
    assert text.count("differ only in case") == 1, text
    # Nothing dead, nothing one-sided, nothing said.
    live = _write_taxonomy("categories: [{name: layout, pattern: '.'}]")
    quiet = "\n".join(diff.render(diff.diff(base, base, live), taxonomy=live))
    assert "claimed no op" not in quiet and "fired on one capture" not in quiet


def test_analysis_carries_the_block_split_and_a_per_cell_count():
    """`--json` used to have no blocks in it at all.

    `analyse_blocks` was called from inside `render`, so the headline table --
    69% of the step in one row -- existed only as fixed-width text, and a
    regression check on block attribution had to scrape it. The rule the rest
    of the kit follows is `analyse` computes and `render` formats.

    `n` per cell is here for the same reason and against the opposite argument:
    it was taken off the rendered block table because a count beside a block
    reads as structure, and that objection is about a page a human reads.
    """
    tax = _write_taxonomy("""
name: t
categories:
  - {name: norm,   block: attn, pattern: 'rms_norm'}
  - {name: layout, pattern: '^copy'}
""")
    phase = PhaseTrace(phase="p", ranks=(
        RankTrace("0", "/x", "step", (100.0,),
                  {"rms_norm.1": 40.0, "copy.2": 30.0, "copy.3": 30.0}),))
    d = breakdown.analyse_phase(phase, tax)
    assert d["blocks"]["attn"]["total_us"] == 40.0
    assert d["unassigned_us"] == 60.0 and d["unassigned"] == {"layout": 60.0}
    # Two instances of `copy` folded into one family, and the count survives.
    assert {(c["family"], c["n"]) for c in d["cells"]} == {("rms_norm", 1),
                                                           ("copy", 2)}
    # And the rendered table is the stored numbers, not a second computation.
    text = "\n".join(breakdown.render({"p": d}, tax))
    assert "| attn" in text and "40.0" in text
    import json
    json.dumps(d)  # the payload has to survive `--json`


def test_two_blocks_differing_only_in_case_are_reported_not_renamed():
    """`moe` and `MoE` are two rows, each holding part of the other's work.

    Folding the case would silently rename a block somebody capitalised on
    purpose, which is worse than the split. Saying so costs nothing.
    """
    tax = _write_taxonomy("""
name: t
categories:
  - {name: a, block: moe, pattern: 'x'}
  - {name: b, block: MoE, pattern: 'y'}
""")
    assert tax.blocks == ("moe", "MoE")
    note = "\n".join(breakdown.taxonomy_notes(tax))
    assert "differ only in case" in note and "'moe', 'MoE'" in note
    same = _write_taxonomy("categories: [{name: a, block: moe, pattern: 'x'}]")
    assert breakdown.taxonomy_notes(same) == []


def test_two_categories_differing_only_in_case_are_reported_too():
    """The argument is about tables, and the category table is a table.

    The check was written for blocks and read `rule.block` directly, so `norm`
    beside `Norm` -- two rows one table up, two plausible headings, subtotals
    that each look like the whole category -- loaded in silence.
    """
    tax = _write_taxonomy("""
name: t
categories:
  - {name: norm, pattern: 'x'}
  - {name: Norm, pattern: 'y'}
""")
    note = "\n".join(breakdown.taxonomy_notes(tax))
    assert "categorys that differ only in case" not in note
    assert "categories that differ only in case" in note
    assert "'norm', 'Norm'" in note
    # One note per fault, not a fourth caveat block per field: both kinds land
    # in the one `notes` list the renderer already prints.
    both = _write_taxonomy("""
name: t
categories:
  - {name: norm, block: moe, pattern: 'x'}
  - {name: Norm, block: MoE, pattern: 'y'}
""")
    assert len(both.notes) == 2


def test_select_dump_keeps_the_most_complete():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "prefill_heavy"
        full = root / "plugins" / "profile" / "2026_08_20_19_00_00"
        partial = root / "plugins" / "profile" / "2026_08_20_18_00_00"
        for d, ranks in ((full, "0246"), (partial, "0")):
            d.mkdir(parents=True)
            for r in ranks:
                (d / f"rank{r}_host.xplane.pb").write_bytes(b"")
        kept, dropped = select_dump(root)
        assert len(kept) == 4
        assert len(dropped) == 1
        assert "18_00_00" in dropped[0]


def test_phase_dirs_handles_flat_and_nested():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "decode_only").mkdir()
        (root / "decode_only" / "rank0_h.xplane.pb").write_bytes(b"")
        (root / "empty_dir").mkdir()
        assert phase_dirs(root) == ["decode_only"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "rank0_h.xplane.pb").write_bytes(b"")
        assert phase_dirs(root) == ["."]

    with tempfile.TemporaryDirectory() as tmp:
        assert phase_dirs(Path(tmp)) == []


# --------------------------------------------------------------------------- #
def _rank(rank: str, steps: list[float], ops: dict | None = None) -> RankTrace:
    return RankTrace(rank=rank, path=f"/x/rank{rank}_h.xplane.pb",
                     step_module="step", step_us=tuple(steps),
                     op_us=ops or {}, async_us=0.0)


def _capture(label: str, phase: str, ranks: dict[str, list[float]],
             ops: dict | None = None) -> Capture:
    return Capture(label=label, root=f"/x/{label}", phases={
        phase: PhaseTrace(phase=phase,
                          ranks=tuple(_rank(r, s, ops) for r, s in ranks.items()))})


def test_shared_ranks_is_the_intersection():
    a = _capture("a", "p", {"0": [1.0], "2": [1.0], "4": [1.0]})
    b = _capture("b", "p", {"0": [1.0], "4": [1.0], "6": [1.0]})
    assert shared_ranks([a.phases["p"], b.phases["p"]]) == ["0", "4"]


def test_compare_pins_ranks_so_churn_is_not_measured():
    """The bug this guards: an unpinned comparison reads rank churn as a delta.

    Both captures are identical on the ranks they share. Only the rank sets
    differ, and the odd rank out is much slower. A correct comparison reports
    zero.
    """
    a = _capture("a", "p", {"0": [100.0], "2": [100.0], "4": [100.0]})
    b = _capture("b", "p", {"0": [100.0], "2": [100.0], "6": [200.0]})
    result = steptime.compare([a, b], ("mean",))
    row = result["p"]["rows"][1]
    assert result["p"]["shared_ranks"] == ["0", "2"]
    assert row["mean_delta_pct"] == 0.0


def test_describe_and_welch():
    s = describe([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["mean"] == 2.0 and s["median"] == 2.0
    assert abs(s["sd"] - 1.0) < 1e-12

    # Identical arms: zero delta, interval centred on zero.
    same = welch([10.0, 11.0, 12.0], [10.0, 11.0, 12.0])
    assert same["delta_pct"] == 0.0
    assert same["ci_lo_pct"] < 0 < same["ci_hi_pct"]

    # A clearly separated pair should exclude zero.
    apart = welch([110.0, 111.0, 112.0], [10.0, 11.0, 12.0])
    assert apart["ci_lo_pct"] > 0

    # Percentiles interpolate. Truncating the index made p10 the minimum for
    # every sample under ten, and p90 the 87.5th percentile at n=9.
    nine = describe([float(i) for i in range(1, 10)])
    assert nine["p10"] == 1.8 and nine["p90"] == 8.2
    assert describe([7.0])["p10"] == 7.0


def test_welch_refuses_to_put_an_interval_on_one_capture_per_arm():
    """Two single captures printed `CI [+1.31%, +1.31%]` and a verdict of SLOWER.

    n=1 measures no run-to-run spread, and this stack's spread is ~1%. The
    honest interval is unbounded and the honest verdict is that there is none.
    """
    one = welch([10.0], [5.0])
    assert one["delta_pct"] == 100.0        # the point estimate is still real
    assert one["t"] is None
    assert one["ci_lo_pct"] == float("-inf") and one["ci_hi_pct"] == float("inf")
    # Identical repeats are the same situation reached a different way: a
    # sample too coarse to show spread is not a sample with none.
    flat = welch([10.0, 10.0], [5.0, 5.0])
    assert flat["t"] is None and flat["ci_hi_pct"] == float("inf")
    # Two per arm with any spread at all is comparable again.
    assert welch([10.0, 11.0], [5.0, 5.5])["t"] is not None


def test_compare_calls_a_one_capture_arm_not_comparable():
    arms = {"base": [_capture("b", "p", {"0": [100.0]})],
            "cand": [_capture("c", "p", {"0": [101.31]})]}
    result = compare.compare_arms(arms, "mean")
    text = "\n".join(compare.render(result) + compare.verdict(result))
    assert "NOT COMPARABLE" in text
    assert "n=1: not comparable" in text
    # The old output claimed a result from two single captures.
    assert "SLOWER" not in text and "NOT DISTINGUISHABLE" not in text
    # ...while still showing the point estimate, which is what to capture more of.
    assert "+1.31%" in text


def test_t95_rounds_towards_the_wider_interval():
    """The bug: rounding the *df* up narrowed every fractional-df interval.

    Welch's df is fractional almost always. The critical value falls as df
    rises, so closing the gap upwards picks a smaller critical value than the
    data supports -- at df 2.32, 3.182 instead of ~3.95, an interval 20% too
    narrow, which is the difference between clearing a promotion gate and not.
    """
    assert t95(3) == 3.182
    assert t95(3.4) == 3.182      # floors to the df=3 row, not up to df=4
    assert t95(1.2) == 12.706
    assert t95(2.16) == 4.303
    assert t95(3.86) == 3.182
    # Past the old end of the table 1.96 was 4% too small; there are rows now.
    assert t95(31) == 2.042
    assert t95(60) == 2.000
    assert t95(1000) == 1.980     # the largest tabled df, not the normal limit
    # Under one degree of freedom there is no row below; take the widest.
    assert t95(0.5) == 12.706
    assert t95(0) == float("inf")
    # Monotone: more data never widens the interval.
    values = [t95(df) for df in (1, 1.5, 2, 5, 12.7, 30, 45, 200)]
    assert values == sorted(values, reverse=True)


def test_compare_arms_reports_the_reference_and_a_verdict():
    arms = {
        "base": [_capture(f"b{i}", "p", {"0": [100.0 + i]}) for i in range(3)],
        "cand": [_capture(f"c{i}", "p", {"0": [200.0 + i]}) for i in range(3)],
    }
    result = compare.compare_arms(arms, "mean")
    p = result["p"]
    assert p["reference"] == "base"
    assert p["arms"]["base"]["vs_ref"] == {}
    assert p["arms"]["cand"]["vs_ref"]["delta_pct"] > 0
    assert "SLOWER" in "\n".join(compare.verdict(result))


def test_compare_arms_skips_a_phase_one_arm_missed():
    arms = {"base": [_capture("b", "p", {"0": [1.0]})],
            "cand": [_capture("c", "q", {"0": [1.0]})]}
    result = compare.compare_arms(arms, "mean")
    assert "skipped" in result["p"] and "skipped" in result["q"]


def test_breakdown_normalises_per_step_and_keeps_async_out():
    tax = Taxonomy.load()
    ops = {"all-reduce.1": 10.0, "copy.2": 5.0, "sort.3": 85.0}
    phase = PhaseTrace(phase="p", ranks=(
        RankTrace("0", "/x", "step", (100.0, 100.0), ops, async_us=500.0),
        RankTrace("2", "/x", "step", (100.0, 100.0), ops, async_us=500.0),
    ))
    d = breakdown.analyse_phase(phase, tax)
    assert d["step_us"] == 100.0
    assert d["busy_us"] == 100.0
    assert abs(d["occupancy"] - 1.0) < 1e-12
    assert d["categories"]["collective"] == 10.0
    assert d["categories"]["sort"] == 85.0
    # Async is reported but never folded into the categories.
    assert d["async_us"] == 500.0
    assert sum(d["categories"].values()) == d["busy_us"]


def test_name_families_counts_what_it_could_not_show():
    members = {"aaa": 3.0, "bbb": 2.0, "ccc": 1.0, "ddd": 0.5}
    assert breakdown.name_families(members, 3) == "aaa, bbb, ccc +1"
    assert breakdown.name_families(members, 9) == "aaa, bbb, ccc, ddd"
    assert breakdown.name_families({}, 3) == ""
    # A name too long for the column is truncated, not wrapped, and the
    # remainder is still counted.
    wide = {"x" * 200: 1.0, "y": 0.5}
    out = breakdown.name_families(wide, 3, width=20)
    assert len(out) <= 20 and out.endswith("+1")


def test_table_renders_a_header_rule():
    lines = table(["a", "b"], [["x", 1]], widths=[3, 3])
    assert lines[0].startswith("| a")
    assert set(lines[1]) <= {"|", "-"}
    assert len(lines) == 3


def test_table_pads_a_short_alignment_string():
    # A caller that grew a column and forgot its alignment character used to
    # get an IndexError on the last cell of every row.
    lines = table(["a", "b", "c"], [["x", 1, 2]], widths=[3, 3, 3], align="<")
    assert lines[-1].startswith("| x")


# --------------------------------------------------------------------------- #
# one compiler-assigned name, several categories
# --------------------------------------------------------------------------- #
#: Three rules, one per input a classifier can use. Only the first two can
#: separate the three ops below; the name is identical for all of them.
_SPLIT_YAML = """
name: split
categories:
  - {name: matmul,      hlo: 'kind=kOutput'}
  - {name: quant,       src: 'util\\.py'}
  - {name: elementwise, pattern: 'fusion'}
"""


def _mixed_phase() -> PhaseTrace:
    """One op family whose three instances belong to three different categories."""
    rank = RankTrace(
        rank="0", path="/x", step_module="step", step_us=(100.0,),
        op_us={"multiply_convert_fusion.1": 60.0,
               "multiply_convert_fusion.2": 30.0,
               "multiply_convert_fusion.3": 10.0},
        op_hlo={"multiply_convert_fusion.1": "kind=kOutput out=bf16[2,4]"},
        op_src={"multiply_convert_fusion.2": "kernels/util.py:308"})
    return PhaseTrace(phase="p", ranks=(rank,))


def test_family_categories_names_where_the_time_went():
    tax = _write_taxonomy(_SPLIT_YAML)
    d = breakdown.analyse_phase(_mixed_phase(), tax)
    fam = breakdown.family_categories(d["cells"])
    # One row, one name: the largest category, plus a count of the others.
    assert fam["multiply_convert_fusion"] == "matmul +2"


def test_family_table_agrees_with_the_category_table():
    """The bug this guards: the family table re-classified from the name alone.

    The name is the one input an ``hlo`` or ``src`` rule cannot use, so the two
    tables on the same page disagreed -- one calling this family a matmul, the
    other pointwise maths, over the same microseconds.
    """
    tax = _write_taxonomy(_SPLIT_YAML)
    d = breakdown.analyse_phase(_mixed_phase(), tax)
    assert d["categories"] == {"matmul": 60.0, "quant": 30.0, "elementwise": 10.0}
    text = "\n".join(breakdown.render({"p": d}, tax, families=5))
    assert "matmul +2" in text
    assert "| elementwise" not in text.split("op families")[-1]


def test_unclassified_list_names_only_unclassified_ops():
    tax = _write_taxonomy("categories: [{name: quant, src: 'util\\.py'}]")
    d = breakdown.analyse_phase(_mixed_phase(), tax)
    last = "\n".join(breakdown.render({"p": d}, tax)).strip().splitlines()[-1]
    assert last.startswith(UNCLASSIFIED)
    # `.2` was classified by its source line. A second, name-only pass could
    # not see that and listed it here, contradicting the table above.
    assert "multiply_convert_fusion.2" not in last
    assert "multiply_convert_fusion.1" in last and "multiply_convert_fusion.3" in last


def test_diff_classifies_with_the_signature_too():
    """Same defect as above, but whole tables: `diff` shares the taxonomy."""
    tax = _write_taxonomy(_SPLIT_YAML)
    phase = _mixed_phase()
    out = diff.diff(Capture("base", "/x", {"p": phase}),
                    Capture("cand", "/y", {"p": phase}), tax)["phases"]["p"]
    cats = {r["op"]: r for r in out["categories"]}
    assert cats["matmul"]["base_us"] == 60.0 and cats["quant"]["base_us"] == 30.0
    assert UNCLASSIFIED not in cats
    fams = {r["op"]: r for r in out["families"]}
    assert fams["multiply_convert_fusion"]["category"] == "matmul +2"


# --------------------------------------------------------------------------- #
def test_render_compare_marks_the_reference_by_position():
    a = _capture("a", "p", {"0": [100.0]})
    b = _capture("b", "p", {"0": [100.0]})
    lines = steptime.render_compare(steptime.compare([a, b], ("mean",)), ("mean",))
    rows = [ln for ln in lines if ln.startswith("| a") or ln.startswith("| b")]
    # The second capture genuinely matched the first, and that is a result
    # worth printing as +0.00% rather than labelling it a second reference.
    assert "ref" in rows[0] and "+0.00%" in rows[1]


def test_compare_renders_an_arm_that_could_not_be_measured():
    """An empty `vs_ref` means "no comparison", which is not the same as "ref"."""
    result = {"p": {
        "estimator": "mean", "pinned_ranks": ["0"], "reference": "base",
        "arms": {
            "base": {"n": 2, "mean": 100.0, "sd": 1.0, "cv_pct": 1.0,
                     "captures": [], "vs_ref": {}},
            "cand": {"n": 0, "mean": None, "sd": None, "cv_pct": None,
                     "captures": [], "vs_ref": {}},
        }}}
    text = "\n".join(compare.render(result) + compare.verdict(result))
    assert "no data" in text and "not compared" in text
    assert text.count("ref\n") <= 1


def test_spread_keeps_each_capture_beside_its_own_value():
    caps = [_capture(f"c{i}", "p", {"0": [100.0 + i]}) for i in range(3)]
    out = compare.spread(caps, "mean")["p"]
    assert out["captures"] == {"c0": 100.0, "c1": 101.0, "c2": 102.0}


def test_emit_metrics_namespaces_and_skips_an_empty_phase():
    cap = _capture("a", "p", {"0": [100.0, 102.0]})
    assert "step_us_p: 101.0000" in steptime.emit_metrics(cap)
    # Without the prefix, a second capture's lines overwrite the first's in
    # whatever harness is parsing them.
    assert all(ln.startswith("a.") for ln in steptime.emit_metrics(cap, "a."))
    empty = Capture("a", "/x", {"p": PhaseTrace(phase="p", ranks=())})
    assert steptime.emit_metrics(empty) == []


def test_cache_is_invalidated_when_the_sidecar_appears():
    """A capture parsed before its op-stats sidecar existed cached no sources.

    Keying only on the xplane would serve that empty source column forever, and
    a missing source line is indistinguishable from an op that never had one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        xplane = Path(tmp) / "rank0_h.xplane.pb"
        xplane.write_bytes(b"trace")
        trace._cache_store(str(xplane), {"step_module": "m"})
        assert trace._cache_load(str(xplane))["step_module"] == "m"
        (Path(tmp) / "rank0_h.op_stats_v2.pb").write_bytes(b"sources")
        assert trace._cache_load(str(xplane)) is None


def test_cache_is_invalidated_when_the_deriving_config_changes():
    """`op_hlo` and `op_src` are derived, and their inputs are meant to be edited.

    Keyed on the files alone, extending PLUMBING for a new stack -- which the
    module docstring invites -- left every warm capture serving the callers of
    the old list, with nothing on screen saying so. The same held for running
    without `xprof` installed: the empty source column was cached and then
    served to a run that could have read sources.
    """
    from xprofkit import opstats
    with tempfile.TemporaryDirectory() as tmp:
        xplane = Path(tmp) / "rank0_h.xplane.pb"
        xplane.write_bytes(b"trace")
        trace._cache_store(str(xplane), {"step_module": "m"})
        assert trace._cache_load(str(xplane))["step_module"] == "m"

        before = opstats.PLUMBING
        try:
            opstats.PLUMBING = (*before, "some_new_framework/")
            assert trace._cache_load(str(xplane)) is None
        finally:
            opstats.PLUMBING = before
        # Restored: the entry is valid again, so the key tracks the config
        # rather than merely changing whenever anything is touched.
        assert trace._cache_load(str(xplane))["step_module"] == "m"

        # Whether the sidecar schema can be imported is part of it too.
        available = opstats.reader_available
        try:
            opstats.reader_available = lambda: not available()
            assert trace._cache_load(str(xplane)) is None
        finally:
            opstats.reader_available = available

        # The signature *regexes*, not just the names of the fields they fill.
        # This hashed `[key for key, _ in _SIGNATURE_FIELDS]`, so rewriting
        # every pattern in the tuple left the digest unmoved -- and tightening
        # a pattern is the ordinary edit here, the one that rewrites every
        # stored `op_hlo` while nothing on screen says the cache is stale.
        fields = trace._SIGNATURE_FIELDS
        try:
            trace._SIGNATURE_FIELDS = tuple(
                (key, pattern + "(?#edited)") for key, pattern in fields)
            assert trace._cache_load(str(xplane)) is None
        finally:
            trace._SIGNATURE_FIELDS = fields
        # ...and the two limits that rewrite a signature without touching a name.
        for attr, value in (("_MAX_GROUP_MEMBERS", 99), ("_MAX_OPERANDS", 99)):
            before_limit = getattr(trace, attr)
            try:
                setattr(trace, attr, value)
                assert trace._cache_load(str(xplane)) is None, attr
            finally:
                setattr(trace, attr, before_limit)
        assert trace._cache_load(str(xplane))["step_module"] == "m"


# --------------------------------------------------------------------------- #
# one op name, several HLO modules
# --------------------------------------------------------------------------- #
class _FakeMetrics:
    """The three fields :func:`opstats._by_weight` reads off an `OpMetrics`."""

    def __init__(self, name, where, time_ps):
        self.name, self.where, self.time_ps = name, where, time_ps


def test_by_weight_resolves_a_name_several_modules_share():
    """A capture runs side graphs, and each numbers its instructions from zero.

    Rank 0's sidecar held 108 duplicate names. `fusion` appeared three times --
    tpu_runner.py:160, fused_moe.py:239 and utils.py:99 -- and taking the last
    entry made the same op answer differently on different ranks, because the
    sidecar's order is not the same on all of them.
    """
    from xprofkit import opstats
    entries = [_FakeMetrics("fusion", "tpu_runner.py:160", 10_800),
               _FakeMetrics("fusion", "fused_moe.py:239", 1_971_200),
               _FakeMetrics("fusion", "utils.py:99", 11_635_200),
               _FakeMetrics("copy", "layout.py:7", 500)]
    out = opstats._by_weight(entries, lambda m: m.where)
    # The xplane row is the merged sum of all three, so the location owning
    # most of the time is the one that describes most of the row: 85% here.
    assert out["fusion"] == "utils.py:99"
    assert out["copy"] == "layout.py:7"
    # Order must not matter -- that is the whole bug.
    assert opstats._by_weight(list(reversed(entries)), lambda m: m.where) == out
    # No owner above the majority: unattributed beats "whichever was largest".
    mixed = [_FakeMetrics("fusion", "a.py:1", 40),
             _FakeMetrics("fusion", "b.py:2", 35),
             _FakeMetrics("fusion", "c.py:3", 25)]
    assert "fusion" not in opstats._by_weight(mixed, lambda m: m.where)


def test_by_weight_refuses_the_two_ties_that_pass_a_majority_test():
    """Both degenerate splits used to resolve, and both resolve to nothing real.

    An exact 50/50 *passes* a >= 0.5 majority test, and the winner was then
    whichever location sorted first: alphabetical attribution, which is the
    arbitrary answer this function exists to refuse. A name whose locations all
    report `time_ps == 0` carries no evidence of ownership at all, and fell out
    through a falsy total -- the right outcome reached by accident, so nothing
    stopped a later edit from turning it into "take the first".

    Neither occurs in the captures to hand, which is exactly why they need
    tests: nothing on the page would move if they regressed.
    """
    from xprofkit import opstats
    tied = [_FakeMetrics("fusion", "z.py:1", 500),
            _FakeMetrics("fusion", "a.py:2", 500)]
    assert opstats._by_weight(tied, lambda m: m.where) == {}
    # Not the alphabetical winner, and not order-dependent either.
    assert opstats._by_weight(list(reversed(tied)), lambda m: m.where) == {}
    # A three-way near-tie above the threshold in aggregate, still no owner.
    thirds = [_FakeMetrics("f", "a.py:1", 34), _FakeMetrics("f", "b.py:2", 33),
              _FakeMetrics("f", "c.py:3", 33)]
    assert opstats._by_weight(thirds, lambda m: m.where) == {}
    # Timeless rows: two locations is no evidence, one location is not a tie
    # and still resolves -- the name is unambiguous whatever its time was.
    zero = [_FakeMetrics("f", "a.py:1", 0), _FakeMetrics("f", "b.py:2", 0)]
    assert opstats._by_weight(zero, lambda m: m.where) == {}
    assert opstats._by_weight([_FakeMetrics("f", "a.py:1", 0)],
                              lambda m: m.where) == {"f": "a.py:1"}


def _disputed_phase() -> PhaseTrace:
    """Two ranks that disagree about one op's source line, agree on another."""
    common = {"fusion.1": 100.0, "fusion.2": 50.0}
    return PhaseTrace(phase="p", ranks=(
        RankTrace("0", "/x", "step", (100.0,), common,
                  op_src={"fusion.1": "kernels/util.py:308",
                          "fusion.2": "kernels/util.py:308"}),
        RankTrace("6", "/x", "step", (100.0,), common,
                  op_src={"fusion.1": "vllm/moe.py:12",
                          "fusion.2": "kernels/util.py:308"}),
    ))


def test_op_attributes_drop_a_field_the_ranks_disagree_on():
    attrs = _disputed_phase().op_attributes()
    # Agreed: kept. Disputed: dropped, because a contested source line is not
    # evidence of anything and a wrong one moves real time into a wrong row.
    assert attrs["fusion.2"][1] == "kernels/util.py:308"
    assert attrs["fusion.1"][1] == ""


def test_breakdown_and_diff_file_the_same_op_in_the_same_category():
    """The bug: two commands read different ranks and disagreed on the same us.

    `breakdown` classified inside its per-rank loop (last rank won, rank 6)
    while `diff` kept the first (rank 0). On the live capture that made
    `fusion` differ by 213x between the two reports, and `moe_glue` by 114 us
    per step, with no flag or warning to say which was meant.
    """
    tax = _write_taxonomy(_SPLIT_YAML)
    phase = _disputed_phase()
    d = breakdown.analyse_phase(phase, tax)
    out = diff.diff(Capture("base", "/x", {"p": phase}),
                    Capture("cand", "/y", {"p": phase}), tax)["phases"]["p"]
    assert d["op_categories"] == {r["op"]: r["category"] for r in out["ops"]}
    # And both agree with the one shared reading, whichever way it went.
    assert d["op_categories"] == {
        name: tax.classify(name, hlo, src)
        for name, (hlo, src) in phase.op_attributes().items()}
    for row in out["categories"]:
        assert row["base_us"] == d["categories"][row["op"]]


def test_src_rule_warning_fires_without_being_asked_for():
    """A capture with no sidecar silently disables every `src:` rule.

    Nothing else on the page changes -- the totals still add up and
    `unclassified` stays small -- so the warning has to be unconditional. It
    used to live inside the `--sources` table, which the report that needed it
    never printed.
    """
    tax = _write_taxonomy(_SPLIT_YAML)
    unsourced = PhaseTrace(phase="p", ranks=(
        RankTrace("0", "/x", "step", (100.0,), {"multiply_convert_fusion.1": 100.0}),))
    text = "\n".join(breakdown.render(
        {"p": breakdown.analyse_phase(unsourced, tax)}, tax))
    assert "1 `src:` rule(s)" in text and "quant" in text

    # Sources present: nothing to warn about.
    sourced = "\n".join(breakdown.render(
        {"p": breakdown.analyse_phase(_mixed_phase(), tax)}, tax))
    assert "`src:` rule(s)" not in sourced

    # A taxonomy with no `src` rules at all never mentions it either.
    plain = _write_taxonomy("categories: [{name: a, pattern: 'fusion'}]")
    assert "`src:` rule(s)" not in "\n".join(breakdown.render(
        {"p": breakdown.analyse_phase(unsourced, plain)}, plain))


def test_diff_warns_that_a_capture_has_no_sources():
    tax = _write_taxonomy(_SPLIT_YAML)
    unsourced = PhaseTrace(phase="p", ranks=(
        RankTrace("0", "/x", "step", (100.0,), {"multiply_convert_fusion.1": 100.0}),))
    result = diff.diff(Capture("base", "/x", {"p": unsourced}),
                       Capture("cand", "/y", {"p": unsourced}), tax)
    assert result["phases"]["p"]["sourced"] == {"base": False, "cand": False}
    assert "`src:` rule(s)" in "\n".join(diff.render(result, taxonomy=tax))
    # Without a taxonomy `render` cannot know, and says nothing rather than
    # guessing -- the JSON carries `sourced` for a caller that wants it.
    assert "`src:` rule(s)" not in "\n".join(diff.render(result))

    # One side instrumented and the other not is the worse case, and it is the
    # live one: of nine captures on this stack exactly one has a sidecar. The
    # two sides are then classified by different rules, and a category-level
    # delta is measuring that rather than the code.
    mixed = diff.diff(Capture("base", "/x", {"p": _mixed_phase()}),
                      Capture("cand", "/y", {"p": unsourced}), tax)
    text = "\n".join(diff.render(mixed, taxonomy=tax))
    assert "base carries source lines and cand does not" in text
    # ...and that is not the same complaint as "no sources anywhere".
    assert "`src:` rule(s)" not in text


# --------------------------------------------------------------------------- #
def test_every_command_exits_nonzero_when_it_measured_nothing():
    """`compare` printed its diagnostics, an empty verdict, and exit 0.

    The README puts that command behind a promotion gate --
    ``analyze.py compare ... && promote`` -- so a mistyped `--ranks`, a renamed
    `--phase` or a capture that never landed promoted the candidate on the
    strength of a report with no numbers in it. Every command is checked here
    because the hole is a class and not a line: each one has a path where
    nothing was measured, and the status is the only part a shell reads.
    """
    import analyze
    with tempfile.TemporaryDirectory() as tmp:
        for capture in ("a", "b"):
            traces = (Path(tmp) / capture / "prefill_heavy" / "plugins"
                      / "profile" / "2026_08_20_17_34_53")
            traces.mkdir(parents=True)
            # A trace with no device plane: a host-only capture looks like this.
            (traces / "rank0_h.xplane.pb").write_bytes(b"")
        argv = {
            "compare": ["compare", "--arm", f"a={tmp}/a", "--arm", f"b={tmp}/b"],
            "phases": ["phases", f"{tmp}/a"],
            "steptime": ["steptime", f"{tmp}/a"],
            "breakdown": ["breakdown", f"{tmp}/a"],
            "diff": ["diff", f"{tmp}/a", f"{tmp}/b"],
        }
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            codes = {name: analyze.main([*args, "--no-cache"])
                     for name, args in argv.items()}
    assert codes == dict.fromkeys(argv, 1), codes
    # The diagnostics are not the fix and were never missing; the status was.
    assert "no capture with usable step data" in quiet.getvalue()


def test_compare_fails_when_one_arm_of_two_is_unusable():
    """Half a comparison is not a comparison, and it exited 0.

    The old check counted arm *names*, which an arm whose captures all failed
    to load passes: two arms were named, so the command believed it had
    compared two arms.
    """
    import analyze
    arms = {"base": [_capture("b", "p", {"0": [100.0]})],
            "cand": [Capture(label="cand/gone", root="/x", phases={})]}
    problems = analyze._compare_problems(arms, compare.compare_arms(arms, "mean"))
    assert any("'cand'" in problem for problem in problems), problems
    assert any("no phase produced a comparison" in problem for problem in problems)
    # And the converse: a real comparison reports nothing and exits 0.
    good = {"base": [_capture("b", "p", {"0": [100.0], "2": [101.0]})],
            "cand": [_capture("c", "p", {"0": [110.0], "2": [111.0]})]}
    assert analyze._compare_problems(good, compare.compare_arms(good, "mean")) == []


def test_a_directory_of_captures_is_not_a_capture():
    """Pointing one level too high does not fail; it averages a run into a row.

    Every capture below the parent is read as one "phase", so prefill and
    decode are merged and the report looks ordinary while measuring nothing.
    """
    import analyze
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "prof"
        for capture in ("baseline_a", "baseline_b"):
            traces = (root / capture / "prefill_heavy" / "plugins" / "profile"
                      / "2026_08_20_17_34_53")
            traces.mkdir(parents=True)
            (traces / "rank0_h.xplane.pb").write_bytes(b"")
        assert analyze._captures_below(str(root)) == ["baseline_a", "baseline_b"]
        # A capture itself, and a phase directory, are not parents.
        assert analyze._captures_below(str(root / "baseline_a")) == []
        assert analyze._captures_below(
            str(root / "baseline_a" / "prefill_heavy")) == []


def test_why_empty_counts_only_the_traces_it_read():
    """"8 trace(s) found, none of which holds a device plane" -- six unopened.

    Under `--ranks 0,2` the count has to be of the selected ranks, or the
    message describes files nothing looked at and sends the reader after a
    capture-wide fault that is not there.
    """
    import analyze

    class _Args:
        ranks = "0,2"
        phase = None

    with tempfile.TemporaryDirectory() as tmp:
        for rank in range(8):
            (Path(tmp) / f"rank{rank}_h.xplane.pb").write_bytes(b"")
        message = analyze._why_empty(tmp, _Args())
        assert "2 trace(s) found for rank(s) 0,2" in message
        _Args.ranks = None
        assert "8 trace(s) found," in analyze._why_empty(tmp, _Args())


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
