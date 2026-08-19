# SPDX-License-Identifier: Apache-2.0
"""Which MoE program actually got compiled, counted on the graph itself.

An MoE kernel arm on this stack is selected by a Python branch that is resolved
when vLLM traces a token bucket, so "the flag was on" and "the kernel is in the
program" are different facts. Three device jobs have been spent on serving
numbers that could not be attributed to either, because every observable
reached for was adjacent to the thing: the enable banner proves selection at
weight load, the decline warning appears in the controls too, the boot log's
``compil*`` line count was a *hang warning* containing the word "compilation",
and total compile time tracks the flag rather than the kernel.

The graph handed to the compiler is not adjacent to anything: it either
contains the custom call or it does not. This module counts, and
``TpuCompilerAdaptor.compile`` prints the count next to the compile range, so a
serving log answers "which MoE path was compiled into bucket B" by grep.
"""

from __future__ import annotations

import collections
from typing import Any, Iterable

# (label, substring of the node target). First match wins, so a more specific
# name must precede a prefix of it. These are the op names built by
# ``layers/vllm/fused_moe_ep.py::_get_op`` and
# ``layers/vllm/fused_moe.py::_build_fused_moe_custom_op``; both mint a fresh
# numbered name per kernel config, so the match is on the stem.
MOE_OP_LABELS: tuple[tuple[str, str], ...] = (
    ("fused_ep", "fused_ep_moe_kernel"),
    ("gmm", "fused_moe_kernel"),
)

# Cross-device collectives, counted BY KIND rather than as one number, because
# the kinds do not behave alike here: a production capture has 0 of 5844
# dispatch all-gather pairs overlapping in wall clock while the combine
# reduce-scatter in the same capture is async. What the fused path claims
# is that it *removes* the exposed pair, not that it shrinks it -- and shrinking
# is worth ~nothing at our shapes, where the dispatch all-gather is 89-97%
# fixed cost (alpha = 9.61 us + 4.47 us/MiB) at decode sizes. So the count of
# each kind, per graph, is the graph-level half of that mechanism claim; the
# other half is exposure, which only an xprof capture can see.
COLLECTIVE_LABELS: tuple[tuple[str, str], ...] = (
    ("ag", "all_gather"),
    ("rs", "reduce_scatter"),
    ("ar", "all_reduce"),
    ("a2a", "all_to_all"),
)

_LABELS: tuple[str, ...] = (tuple(label for label, _ in MOE_OP_LABELS) +
                            tuple(label for label, _ in COLLECTIVE_LABELS))

# Boot order, so the table reads the way the run happened.
_PHASE_ORDER = ("boot", "profile", "capture", "serving")


def _range_sort_key(key: str) -> tuple[int, int, str]:
    """Order the report by phase then token bucket, whatever ``Range`` reprs."""
    phase = key.split(" ", 1)[0]
    order = (_PHASE_ORDER.index(phase)
             if phase in _PHASE_ORDER else len(_PHASE_ORDER))
    digits = "".join(c if c.isdigit() else " " for c in key).split()
    return (order, int(digits[0]) if digits else -1, key)


def census(nodes: Iterable[Any]) -> collections.Counter:
    """Count MoE custom calls and collectives in one FX graph's nodes.

    Takes an iterable of nodes rather than a ``GraphModule`` so it can be
    tested without building one, and matches on ``str(node.target)`` because a
    torch custom op's target is an ``OpOverload`` whose repr is the qualified
    name (``pallas.fused_ep_moe_kernel_0.default``).
    """
    counts: collections.Counter = collections.Counter()
    for node in nodes:
        if getattr(node, "op", None) not in ("call_function", "call_method",
                                             "call_module"):
            continue
        target = str(getattr(node, "target", ""))
        for label, needle in MOE_OP_LABELS + COLLECTIVE_LABELS:
            if needle in target:
                counts[label] += 1
                break
    return counts


def format_counts(counts: collections.Counter) -> str:
    """``fused_ep=60 gmm=0 ag=0 rs=0 ar=60 a2a=0`` -- every field, never blank.

    A missing field would read as "not measured"; a zero is a measurement.
    """
    return " ".join(f"{name}={counts.get(name, 0)}" for name in _LABELS)


class _Recorder:
    """Process-wide tally, keyed by boot phase and compile range.

    The phase matters as much as the range: ``profile_run`` compiles the
    max-token bucket *before* the DP coordinator exists and throws the result
    away, so a fused count of zero there is expected and says nothing about
    serving, while a zero in the capture phase is the whole failure.
    """

    def __init__(self) -> None:
        self.by_range: dict[str, collections.Counter] = {}
        self.graphs: int = 0
        self.phase: str = "boot"

    def observe(self, nodes: Iterable[Any],
                compile_range: Any) -> collections.Counter:
        counts = census(nodes)
        key = f"{self.phase} {compile_range}"
        self.by_range.setdefault(key, collections.Counter()).update(counts)
        self.graphs += 1
        return counts

    def total(self, label: str) -> int:
        return sum(c.get(label, 0) for c in self.by_range.values())

    def total_in_phase(self, label: str, phase: str) -> int:
        return sum(
            c.get(label, 0) for key, c in self.by_range.items()
            if key.startswith(f"{phase} "))

    def reset(self) -> None:
        self.by_range.clear()
        self.graphs = 0
        self.phase = "boot"

    def table(self) -> str:
        if not self.by_range:
            return (
                "MoE graph census: NOTHING WAS COMPILED THROUGH THIS "
                "PROCESS -- the census saw 0 graphs, which is not the same "
                "as 0 MoE ops.")
        lines = [f"MoE graph census over {self.graphs} compiled graphs:"]
        for key in sorted(self.by_range, key=_range_sort_key):
            lines.append(f"  range {key}: {format_counts(self.by_range[key])}")
        totals = collections.Counter()
        for counts in self.by_range.values():
            totals.update(counts)
        lines.append(f"  TOTAL: {format_counts(totals)}")
        return "\n".join(lines)


RECORDER = _Recorder()


def enabled() -> bool:
    """Whether to census at all.

    The census is what stops a serving arm being unattributable -- 
    ``fused_moe_ep.report_engagement`` reads it, and a run that cannot say
    which MoE program it compiled has already cost this project two jobs. It is
    also a general compile-time facility, so it stays switched off, and leaves
    the compiler's log lines unchanged, for every deployment that is not using
    the kernel it exists to observe.
    """
    from vllm_torchtpu import envs

    return bool(envs.USE_MOE_FUSED_EP_V2)


def observe(graph: Any, compile_range: Any) -> collections.Counter:
    """Census one ``fx.GraphModule`` and fold it into the process tally.

    Inert unless the fused-EP kernel was asked for. The census is what stops a
    serving arm being unattributable -- ``fused_moe_ep.report_engagement``
    See `enabled`.
    """
    if not enabled():
        return collections.Counter()
    nodes = getattr(getattr(graph, "graph", None), "nodes", ())
    return RECORDER.observe(nodes, compile_range)


def set_phase(phase: str) -> None:
    """Label subsequent compiles: ``profile`` / ``capture`` / ``serving``.

    A compile tagged ``serving`` is a recompile after warmup -- worth seeing on
    its own, whatever it contains.
    """
    RECORDER.phase = phase
