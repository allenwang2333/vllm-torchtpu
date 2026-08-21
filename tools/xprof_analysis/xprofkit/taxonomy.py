"""Turn HLO op names into categories, from a YAML file rather than from code.

Two questions are being answered, and they are not the same question.

``name``
    *What kind of work is this op?* A regex over the op name, the signature or
    the source line; first match wins, so the order in the file is significance
    order.

``block``
    *Which part of the model paid for it?* An optional word on the same rule,
    naming the component its ops belong to -- ``moe``, ``attn``. Categories
    carrying one are subtotalled into blocks; categories without one are
    reported as unassigned. Purely a declaration: the trace does not say which
    block an op belongs to, and this module does not pretend to derive it.

    An earlier version did try to derive it, from *arity*: a family whose
    instance count matched the number of layers of one kind was taken to run
    once per layer of that kind. The counting works and the generalisation does
    not. It needs one model's layer counts written into the taxonomy, so the
    file stops describing a stack and starts describing a checkpoint. It cannot
    separate two layer types of equal depth even in principle. And a count that
    divides two ways needs a tie-break, which is a guess wearing arithmetic --
    on the shipped capture a tolerance of one instance moved 5.4% of the step
    between blocks. Declaring the block costs one word per rule and is right
    for a model nobody has counted the layers of.

Nothing model-specific belongs in this module. See ``taxonomies/default.yaml``
for the shipped rules and the README for how to write your own.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent / "taxonomies" / "default.yaml"

#: Category assigned to an op no rule matched. Reported loudly, because a large
#: unclassified bucket means every other percentage on the page is provisional.
UNCLASSIFIED = "unclassified"

#: Row label for categories no rule put in a block.
UNASSIGNED = "(no block)"

#: Heading of the summary row every table ends with.
TOTAL = "TOTAL"

#: The row labels the renderer spends on rows of its own, and what each one
#: means. A taxonomy may not use any of them for a `name` or a `block`: the
#: table would print two rows under one heading with two meanings and nothing
#: on the page to tell them apart.
#:
#: One list, not three checks, because the first two were each guarded twice
#: and `TOTAL` was guarded nowhere -- `name: TOTAL, block: TOTAL` printed two
#: `TOTAL` rows in the category table (11,912.7 us and the real 306,221.0) and
#: two more in the block table. Any label the renderer reserves in future goes
#: here and is rejected in both fields for free.
RESERVED_LABELS = {
    UNCLASSIFIED: "where this tool files ops no rule matched",
    UNASSIGNED: "what this tool prints for categories in no block",
    TOTAL: "the summary row at the foot of every table",
}


#: Dot-components the compiler appends around an op's serial number. They are
#: wrappers, not part of the op's identity, and they sit *outside* the serial:
#: ``reduce-scatter.399.cloned.1.call-done``.
#:
#: ``clone`` and ``remat`` are the same thing as ``cloned`` under different
#: spellings -- one instruction duplicated per user, or rematerialised instead
#: of kept live. Leaving them on split ``broadcast_in_dim.1697.clone`` off as
#: its own family beside a one-instance ``broadcast_in_dim`` -- copies of a
#: single instruction reported as two unrelated rows, each too small to appear
#: in a top-N list, and neither one the size of the thing it came from.
CHURN_COMPONENTS = frozenset({"cloned", "clone", "remat", "call-start", "call-done"})


def family(op: str) -> str:
    """``copy.18`` -> ``copy``, ``reduce-scatter.399.cloned.1.call-done`` -> ``reduce-scatter``.

    Two things get cut, in this order.

    **Compiler wrappers.** Every component of :data:`CHURN_COMPONENTS` --
    ``cloned``, ``clone``, ``remat``, ``call-start``, ``call-done`` -- plus
    any serial number they carry. Leaving them on is not cosmetic: on this
    stack reduce-scatter fragments into ~480 distinct names that way, each one
    small, so the single largest cost in the step (22% of it) does not appear
    anywhere in a top-N list. ``call-start`` and ``call-done`` fold into the
    same row on purpose -- they are the two halves of one collective.

    **The serial number.** One trailing all-digit component, so ``kernel.90``
    through ``kernel.134`` are one op called once per layer, not many ops.

    Only one serial is cut, and only from the end: ``reshape.11.2`` stays
    ``reshape.11``, because there the ``11`` is part of the name.
    """
    while "." in op:
        head, _, tail = op.rpartition(".")
        if tail in CHURN_COMPONENTS:
            op = head
            continue
        # A serial belonging to a wrapper, e.g. the `1` of `.cloned.1`.
        if tail.isdigit() and head.rpartition(".")[2] in CHURN_COMPONENTS:
            op = head
            continue
        break
    head, _, tail = op.rpartition(".")
    return head if head and tail.isdigit() else op


#: Value of ``async_execution_thread`` that means the op runs on SparseCore.
#: SparseCore is a different unit with a different roofline -- no VPU -- so an
#: op landing there must not be judged against TensorCore peak.
SPARSECORE_THREAD = "sparsecore"
TENSORCORE = "tensorcore"


def device_of(hlo: str) -> str:
    """Which unit an op runs on, from its HLO signature.

    Orthogonal to the category on purpose. ``sc_ragged_gather_reduce_v2`` is
    *both* attention work and SparseCore work, and collapsing that into one
    axis loses whichever of the two you were not thinking about. On this stack
    the SparseCore share is 36% of op time while the ops whose *name* starts
    ``sc_`` are 0.8%.
    """
    return SPARSECORE_THREAD if f"thread={SPARSECORE_THREAD}" in hlo else TENSORCORE


#: The conditions a rule may declare, and everything a YAML entry may carry.
#: Spelled out so that a misspelt key (`patern:`) is an error at load time
#: rather than a rule that silently matches nothing.
CONDITION_FIELDS = ("pattern", "hlo", "src")
RULE_FIELDS = frozenset({"name", "block", *CONDITION_FIELDS})

#: Every section a taxonomy file may declare. Checked for the same reason as
#: :data:`RULE_FIELDS`: ``categorys:`` parses fine and leaves the whole report
#: empty, with nothing on screen naming the typo.
TAXONOMY_FIELDS = frozenset({"name", "categories"})


@dataclass(frozen=True)
class Rule:
    """One category rule. Three independent conditions, all of which must hold.

    ``pattern``
        The op name -- what the compiler called it.
    ``hlo``
        The signature from :func:`~xprofkit.trace.hlo_signature` -- what the
        instruction *is*. The only way to catch a matmul XLA named after its
        elementwise interior.
    ``src``
        ``file.py:123`` -- whose code asked for it, from the op-stats sidecar.
        The only way to separate ops the compiler gave interchangeable names:
        two ``multiply_convert_fusion`` are the fp8 quantise and the routing
        gate, and nothing but the source line says which is which.

    A rule declaring none of the three would match everything, so it matches
    nothing and :meth:`Taxonomy.load` rejects it outright.

    ``block`` is not a condition. It is the answer to the *other* question --
    which part of the model this rule's ops belong to -- and it is optional:
    leave it off and the category is reported on its own, outside every block.
    """

    category: str
    pattern: re.Pattern | None = None
    hlo: re.Pattern | None = None
    src: re.Pattern | None = None
    block: str | None = None

    def matches(self, op: str, hlo: str = "", src: str = "") -> bool:
        if self.pattern is not None and not self.pattern.search(op):
            return False
        if self.hlo is not None and not self.hlo.search(hlo):
            return False
        if self.src is not None and not self.src.search(src):
            return False
        return any(c is not None for c in (self.pattern, self.hlo, self.src))


def _label(where: str, field: str, value: object, role: str) -> str:
    """Validate one of a rule's two row headings, and return it unchanged.

    ``name`` and ``block`` are the same kind of thing -- a string a human wrote,
    which the renderer prints as a row heading and which everything downstream
    joins on by equality -- so they take the same three checks. They did not
    always. ``block`` was silently ``.strip()``ed while ``name`` was rejected
    for the same fault, which made this loader the one place in the tool that
    rewrites what the user wrote rather than telling them about it, and the
    stated reason (the name is a join key) was true of ``block`` too: it keys
    the ``blocks`` dict in ``--json``.

    Each check is a mistake with no symptom. A non-string reaches the renderer
    as a heading anyway -- ``name: 60`` prints a row headed ``60``, which is
    the old layer count written into the field that replaced it. A padded value
    prints under a heading identical to the unpadded one and subtotals
    separately. A reserved label prints beside the renderer's own row of that
    name: ``name: unclassified`` printed the category twice and then warned
    that 16.50% of op time needed a rule while listing ops that rule had
    classified.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{where}: `{field}` is {value!r}; it must be a non-empty string "
            f"{role}")
    if value != value.strip():
        raise ValueError(
            f"{where}: `{field}` is {value!r}, which has surrounding "
            f"whitespace; write {value.strip()!r} -- padded and unpadded are "
            f"two {field}s that print as two rows with the same heading")
    if value in RESERVED_LABELS:
        raise ValueError(
            f"{where}: `{field}` is {value!r}, which is "
            f"{RESERVED_LABELS[value]}; pick another, or the table prints two "
            "rows under one heading with two meanings")
    return value


def _confusable(labels: Iterable[str | None]) -> list[list[str]]:
    """Groups of row headings that differ only in case, in first-seen order.

    Shared by the category and the block heading, which is the whole point:
    the two used to be checked by one loop written for blocks, so `norm`
    beside `Norm` -- the identical fault one table up -- loaded in silence.
    """
    by_case: dict[str, list[str]] = {}
    for label in dict.fromkeys(labels):
        if label:
            by_case.setdefault(label.lower(), []).append(label)
    return [group for group in by_case.values() if len(group) > 1]


def _block_label(block: str | None) -> str:
    """How a declaration reads in an error message, blocked or not.

    ``None`` has to render as words rather than as ``None``: the message
    compares two rules and one side of the comparison is routinely the absence
    of a declaration, which is the case the check exists for.
    """
    return f"block {block!r}" if block else "no block"


@dataclass(frozen=True)
class Taxonomy:
    """Loaded rules. Immutable; load a second one to compare taxonomies."""

    name: str
    rules: tuple[Rule, ...]
    #: Load-time observations that are suspicious but not wrong, printed under
    #: the tables. A separate channel because :meth:`load` must not print --
    #: it is called from a library -- and because the alternative for the one
    #: case there is would be to *rename* the user's block, silently.
    notes: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Taxonomy":
        """Read a taxonomy YAML, rejecting anything that would fail silently.

        Every check below is for a mistake whose only symptom would otherwise
        be a table that quietly says the wrong thing: a misspelt condition key
        makes a rule match nothing, and a misspelt ``block`` deletes a row from
        the block table without a word. The taxonomy is the extension point, so
        it is the one place worth being loud.

        The shape checks come first and are not paranoia. ``categories:`` with
        the list indented one level too far parses as a *string*, and
        ``set(entry) - RULE_FIELDS`` on a string is a set of characters, so the
        complaint used to be ``unknown field(s) ['-', 'a', 'e', ...]``; a
        list item written ``- name: a`` with the rest at the wrong indent
        parses as ``None`` and raised a bare ``TypeError`` from inside this
        loop, which :func:`analyze.py::_taxonomy` does not catch and which
        reached the user as a stack trace naming neither the file nor the line.
        """
        path = Path(path or DEFAULT_TAXONOMY)
        spec = yaml.safe_load(path.read_text()) or {}
        if not isinstance(spec, dict):
            raise ValueError(
                f"{path}: the file is a {type(spec).__name__}, not a mapping; "
                f"a taxonomy is a mapping of {sorted(TAXONOMY_FIELDS)}")
        unknown_sections = set(spec) - TAXONOMY_FIELDS
        if unknown_sections:
            raise ValueError(
                f"{path}: unknown section(s) {sorted(unknown_sections)}; a "
                f"taxonomy takes {sorted(TAXONOMY_FIELDS)}")
        entries = spec.get("categories") or []
        if not isinstance(entries, list):
            raise ValueError(
                f"{path}: `categories` is a {type(entries).__name__}, not a "
                "list; it is a list of rules, one `- {name: ..., pattern: "
                "...}` per line")
        rules = []
        for index, entry in enumerate(entries):
            where = f"categories[{index}] of {path}"
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{where}: is a {type(entry).__name__} ({entry!r}), not a "
                    "mapping; every rule is `- {name: ..., pattern: ...}`")
            unknown = set(entry) - RULE_FIELDS
            if unknown:
                raise ValueError(
                    f"{where}: unknown field(s) {sorted(unknown)}; a rule takes "
                    f"{sorted(RULE_FIELDS)}")
            name = _label(where, "name", entry.get("name"),
                          "naming the category this rule files ops under")
            if not any(entry.get(f) for f in CONDITION_FIELDS):
                raise ValueError(
                    f"{where}: category {name!r} declares none of "
                    "`pattern`, `hlo` or `src`, so it would match nothing")
            # A block is a free-form label -- the set of blocks is whatever the
            # file declares -- and so it takes exactly the checks `name` does.
            # Case is the one fault neither field rejects: folding it would
            # rename a block somebody capitalised on purpose, so `MoE` beside
            # `moe` goes into `notes` below instead.
            block = entry.get("block")
            if block is not None:
                block = _label(where, "block", block,
                               "naming the part of the model these ops "
                               "belong to")
            compiled = {}
            for field_name in CONDITION_FIELDS:
                condition = entry.get(field_name)
                if not condition:
                    continue
                # Checked before `re.compile`, which raises `TypeError` for a
                # non-string -- and the `except re.error` below does not catch
                # it, so it escaped `analyze.py` as a traceback naming neither
                # the file nor the rule. The trigger is the metacharacter set
                # this field exists to hold: unquoted `pattern: [copy]` is a
                # YAML *list*, and `pattern: {copy}` a mapping.
                if not isinstance(condition, str):
                    # The hint quotes back what was written, not a stock
                    # example: `src: 12` advised as ``quote it, `src:
                    # '[copy]'` `` names the right field and the wrong value,
                    # and reads as though `[copy]` were the fix.
                    # Dumped back rather than `repr`ed, so the hint is spelt
                    # the way the file spells it: `['copy']` is Python for
                    # what YAML wrote as `[copy]`.
                    literal = yaml.safe_dump(
                        condition, default_flow_style=True).splitlines()[0]
                    raise ValueError(
                        f"{where}: `{field_name}` is a "
                        f"{type(condition).__name__} ({condition!r}), not a "
                        "string; YAML read the regex as structure -- quote it, "
                        f"`{field_name}: '{literal}'`")
                try:
                    compiled[field_name] = re.compile(condition, re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(
                        f"{where}: `{field_name}` is not a valid regex: {exc}") from exc
            rules.append(Rule(category=name, block=block, **compiled))

        # The block table groups by *category*, because that is the granularity
        # every other table on the page uses and the one the cells are keyed
        # by. Several rules may share a category name -- `matmul_other` is
        # three rules -- and they fold into one row, so they cannot disagree
        # about where that row belongs. Silently taking the first would put the
        # whole category in one block on the strength of whichever rule
        # happened to be written first.
        #
        # *Every* rule is recorded, including the ones with no `block:`. An
        # earlier version compared only the blocked ones, so blocked-vs-
        # unblocked was not a disagreement -- and that is the likelier mistake,
        # because it is what a half-finished edit looks like. Adding `block:
        # attn` to one of the two `quant` rules in the shipped qwen taxonomy
        # loaded without complaint and moved 6.11% of the step out of
        # `(no block)` and into `attn`, a 2.4x change in that row, with the
        # category table byte-identical either way.
        declared: dict[str, tuple[str | None, int]] = {}
        for index, rule in enumerate(rules):
            first, first_index = declared.setdefault(
                rule.category, (rule.block, index))
            if first != rule.block:
                raise ValueError(
                    f"{path}: category {rule.category!r} is in "
                    f"{_block_label(first)} per categories[{first_index}] and "
                    f"{_block_label(rule.block)} per categories[{index}]; the "
                    "block table groups by category, so every rule naming a "
                    "category must give the same `block:` -- or none of them "
                    "must")

        # Case is *not* folded: `MoE` may be somebody's deliberate spelling and
        # renaming it behind their back is worse than the split. But the split
        # itself has no signal -- two rows, two plausible headings, subtotals
        # that each look like the whole block -- so it is said out loud instead.
        # Both row headings get this: the argument is about what a table looks
        # like, and the category table is a table, so `norm` beside `Norm` is
        # the same fault one table up and used to load in silence.
        name = spec.get("name", path.stem)
        notes = [
            f"taxonomy `{name}` declares {field} that differ only in case: "
            f"{', '.join(repr(label) for label in group)}. They are separate "
            "rows and each holds part of the other's work."
            for field, labels in (("categories", [r.category for r in rules]),
                                  ("blocks", [r.block for r in rules]))
            for group in _confusable(labels)
        ]
        return cls(name=name, rules=tuple(rules), notes=tuple(notes))

    @property
    def order(self) -> tuple[str, ...]:
        """Category names in file order, deduplicated, ``unclassified`` last."""
        seen: list[str] = []
        for rule in self.rules:
            if rule.category not in seen:
                seen.append(rule.category)
        return tuple(seen) + (UNCLASSIFIED,)

    @property
    def src_rules(self) -> tuple[Rule, ...]:
        """Rules that need a source line, and so need the op-stats sidecar.

        Against a capture with no sidecar none of them can fire, and the
        failure is silent by construction -- see :meth:`classify`. The commands
        report how many there are, and which categories they feed, whenever a
        capture has no source coverage.
        """
        return tuple(rule for rule in self.rules if rule.src is not None)

    def match(self, op: str, hlo: str = "", src: str = "") -> int | None:
        """Index of the first rule that matches, or ``None`` for no rule.

        The index, not the category, because several rules routinely share a
        category -- `matmul_other` is three -- and the caller that wants to
        know which *rules* ever fired cannot recover that from the name. It is
        also the rule's position in the file's ``categories:`` list, which is
        how a report points at the line.
        """
        for index, rule in enumerate(self.rules):
            if rule.matches(op, hlo, src):
                return index
        return None

    def classify(self, op: str, hlo: str = "", src: str = "") -> str:
        """First matching rule wins.

        Both extra arguments are optional and omitting them degrades to
        name-only classification rather than failing, so an old cache or a
        hand-built trace still works -- but rules conditioned on what is
        missing will not fire, and the report will quietly look like the
        pre-signature one.

        The ``unclassified`` line does **not** detect that. A ``src``-rule miss
        falls through to a later, broader name rule -- the fp8 quantise lands
        in ``elementwise`` rather than in ``quantise`` -- so the report stays
        fully classified and merely wrong. A capture with no sources at all is
        caught by :attr:`src_rules`, which the commands warn on; a capture with
        *partial* coverage shows up as a category smaller than its usual share
        with the same op families still in it.
        """
        index = self.match(op, hlo, src)
        return UNCLASSIFIED if index is None else self.rules[index].category

    # --------------------------------------------------------------- blocks #
    @property
    def blocks(self) -> tuple[str, ...]:
        """Every block the file declares, in file order.

        There is no fixed set. A block is whatever a rule says it is, so a
        stack with an encoder and a decoder writes those and gets those.
        """
        seen: list[str] = []
        for rule in self.rules:
            if rule.block and rule.block not in seen:
                seen.append(rule.block)
        return tuple(seen)

    @property
    def block_of(self) -> dict[str, str]:
        """``{category: block}`` for the categories a rule assigned to one.

        Consistency across rules sharing a category is checked in
        :meth:`load`, so the first declaration is the only declaration.
        """
        out: dict[str, str] = {}
        for rule in self.rules:
            if rule.block:
                out.setdefault(rule.category, rule.block)
        return out
