"""Source attribution for device ops, from the profiler's op-stats sidecar.

The xplane holds timing; it does **not** hold source lines. Device op events
carry only offsets, durations and offload ids, and the HLO text in the event
name has no ``metadata={... source_file ...}``. What the profiler writes
*alongside* the xplane is ``rank<N>_<host>.op_stats_v2.pb`` -- XProf's own
aggregation -- and every op in there has an ``OpMetrics.source_info`` with a
file, a line, and the whole stack that emitted it.

That is the difference between "10.10% of the step is elementwise" and "7.9 ms
of it is the fp8 quantise in ``quantized_matmul/util.py:94``". Op names on a
fused stack say what the compiler did; source says whose code asked for it.

Two limits worth knowing before leaning on it:

* **Coverage is partial, and a collective's coverage is worse than none.**
  About three quarters of op time gets a usable source line. Collectives are
  the hole, and not in the way it first looks: they are not missing a stack,
  they have one, and it is the *compilation* stack rather than the execution
  one. A collective is functionalised into the graph at trace time, so by the
  time the instruction exists the model frame that wanted it is gone; what the
  sidecar records is the ~70 frames of ``torch/_library/autograd.py`` ->
  ``_functional_collectives.py`` -> aot_autograd -> the piecewise backend that
  every collective in the module passes through identically. Two all-reduces
  from different parts of one model have **byte-identical** stacks. On this
  capture the whole stack is :data:`PLUMBING`, so ``reduce-scatter`` reports no
  caller at all and ``all-gather`` reports the compiler driver
  (``tpu_compiler.py:396``); neither names a part of the model, and no ``src:``
  rule can ever separate two collectives. A Pallas kernel keeps a real frame
  (``fused_moe.py:239``) only because it dispatches at *runtime*, through
  ``__torch_function__``, with the model's stack still on it.
* **The innermost frame is where the trace stopped, not where the intent was.**
  For anything dispatched through Pallas it is the call wrapper,
  ``torch_tpu/_internal/pallas/pallas.py:638`` -- one undifferentiated row worth
  51.8% of the step. Report the innermost frame that is *not* dispatch plumbing
  instead: :func:`read_callers`, not :func:`read_sources`. See
  :data:`PLUMBING`. A trampoline can still swallow the owner, though --
  ``parameter.py:127`` is vLLM's ``BasevLLMParameter.__torch_function__``.

The join is by op name, and the name is **not unique**. A capture runs several
HLO modules -- the decoder plus small side graphs for sampling and token
scatter -- and each numbers its instructions from scratch, so one capture holds
three instructions called ``fusion`` in three modules with three different
owners. The xplane merges them into one row as well (its op line is keyed by
name too), so the honest join is by *weight*: see :func:`read_callers`. Parsing
costs ~0.1 s against a 54 MB file, so this is not the expensive part of
anything.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re

#: Path prefixes to cut so a location fits in a table column. Site-packages
#: first, since a checkout under a workspace can also live inside one.
_TRIM = (
    re.compile(r"^.*/site-packages/"),
    re.compile(r"^/home/[^/]+/workspace/"),
    re.compile(r"^/usr/lib/python[\d.]+/"),
)

#: Suffix the profiler appends to an xplane to name its op-stats sidecar.
OP_STATS_SUFFIX = ".op_stats_v2.pb"
XPLANE_SUFFIX = ".xplane.pb"

#: Frames that are never the answer to "whose work is this?". Everything here is
#: dispatch machinery that every op in the module passes through identically:
#: torch's operator plumbing and dynamo, the Pallas call wrapper, and the
#: torch_tpu export/compile driver below it. Walking past them is the whole
#: trick -- the innermost frame of a Pallas kernel is always
#: ``pallas/pallas.py:638`` and says nothing, while the frame directly above the
#: plumbing is ``fused_moe.py:239`` or ``attention.py:666``, which says
#: everything. On this stack that is the difference between one 51.8% row and a
#: breakdown of it.
#:
#: Order-insensitive; matched as path prefixes after :func:`shorten`.
PLUMBING = (
    "torch/",
    "torch_tpu/_internal/pallas/",
    "torch_tpu/_internal/export/",
    "torch_tpu/_internal/compile/",
    "contextlib",
    "multiprocessing/",
)


def op_stats_path(xplane_path: str) -> str | None:
    """The op-stats sidecar beside *xplane_path*, if the capture wrote one."""
    if not xplane_path.endswith(XPLANE_SUFFIX):
        return None
    candidate = xplane_path[: -len(XPLANE_SUFFIX)] + OP_STATS_SUFFIX
    return candidate if os.path.exists(candidate) else None


def shorten(path: str) -> str:
    """``/home/me/workspace/vllm/vllm/x.py`` -> ``vllm/vllm/x.py``."""
    for pattern in _TRIM:
        trimmed = pattern.sub("", path)
        if trimmed != path:
            return trimmed
    return path


#: A name shared by several HLO modules keeps the owner of the module holding
#: at least this share of the name's device time. The xplane row is that same
#: merged sum, so above a half the row *is* mostly that module's work; below it
#: the row is a genuine mixture and no single line owns it.
CALLER_MAJORITY = 0.5


def reader_available() -> bool:
    """Whether the sidecar schema can be imported in this interpreter.

    Part of the cache key, not just a diagnostic: parsing a capture without
    ``xprof`` installed produces an empty source column, and a cache entry
    written then must not be served to a run that *can* read sources. See
    :func:`~xprofkit.trace._cache_key`.
    """
    try:
        from xprof.protobuf import op_stats_pb2  # noqa: F401
    except ImportError:
        return False
    return True


def source_fingerprint() -> str:
    """Identity of everything this module's source attribution depends on.

    The attribution is cached beside the trace, and the cache would otherwise
    outlive the thing it was derived from: add a frame to :data:`PLUMBING` --
    which this module's docstring invites -- and a warm capture keeps serving
    the old callers, with nothing on screen to say the column is stale. So the
    inputs are hashed into the cache key.

    Covers the data, not the logic: editing :func:`caller_of` itself still
    needs ``CACHE_VERSION`` bumped in :mod:`~xprofkit.trace`.

    ``reader`` was a bool, so upgrading ``xprof`` in place -- same capture,
    same frames, a reader that reads the sidecar differently -- left the key
    unchanged. The version string closes that; it is unverified in the sense
    that no upgrade has been observed to move an attribution here, only that
    the input is now in the key rather than out of it.
    """
    try:
        installed = importlib.metadata.version("xprof")
    except importlib.metadata.PackageNotFoundError:
        # Absent is a state the rest of this module already handles, and it
        # is a different key from any version, which is what is wanted.
        installed = None
    payload = json.dumps({
        "reader": reader_available(),
        "xprof": installed,
        "plumbing": PLUMBING,
        "trim": [pattern.pattern for pattern in _TRIM],
        "majority": CALLER_MAJORITY,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _device_ops(path: str) -> list:
    """Every device op in the sidecar, or ``[]`` if it cannot be read.

    Returns nothing rather than raising when ``xprof`` is absent: source
    attribution is a bonus, not a dependency, and every other command in this
    kit works without it.
    """
    try:
        from xprof.protobuf import op_stats_pb2
    except ImportError:
        return []
    stats = op_stats_pb2.OpStats()
    with open(path, "rb") as handle:
        stats.ParseFromString(handle.read())
    return list(stats.device_op_metrics_db.metrics_db)


def _by_weight(entries, locate) -> dict[str, str]:
    """``{op name: location}``, resolving a name several modules share.

    One name can carry several locations -- three modules each with their own
    ``fusion`` -- and the sidecar stores them in file order, which differs
    between ranks. Taking the last (or the first) is therefore not a choice at
    all: it makes the same op in the same capture answer differently on rank 0
    and rank 6, and two commands reading different ranks then report different
    categories for the same microseconds. That happened, and it is what this
    function exists to prevent.

    The tie-break is device time. The xplane row this joins onto is the same
    merged sum, so the location owning most of the name's time is the one that
    describes most of the row. Below :data:`CALLER_MAJORITY` no location owns
    the row and the op is left unattributed rather than filed under whichever
    of them was largest -- 40/35/25 is a mixture, not an owner.

    Two degenerate cases are mixtures as well, and both used to resolve. An
    exact 50/50 split *passes* a 0.5 majority test, and the winner was then
    whichever location sorted first -- alphabetical attribution, which is the
    arbitrary answer this function exists to refuse. And a name whose locations
    all report ``time_ps == 0`` carries no evidence of ownership at all, yet
    fell out through a falsy ``total`` as a silent drop rather than a decision.
    Both are now the same refusal, for the same stated reason. Neither occurs
    in the captures to hand -- exact picosecond ties do not happen and every
    sidecar row here has a non-zero time -- so this is about what the rule
    *says*, on the next stack, not about a number that moves on this one.
    """
    weights: dict[str, dict[str, int]] = {}
    for metrics in entries:
        where = locate(metrics)
        if not where:
            continue
        seen = weights.setdefault(metrics.name, {})
        seen[where] = seen.get(where, 0) + metrics.time_ps
    out: dict[str, str] = {}
    for name, seen in weights.items():
        # Sorted by weight, then by location, so that the ordering is the same
        # on every rank and every run rather than by dict insertion order.
        ordered = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ordered) == 1:
            out[name] = ordered[0][0]
            continue
        (best, weight), (_, runner_up) = ordered[0], ordered[1]
        total = sum(seen.values())
        if weight == runner_up or not total:
            continue
        if weight / total >= CALLER_MAJORITY:
            out[name] = best
    return out


def read_sources(path: str) -> dict[str, str]:
    """``{op name: "file.py:123"}`` for every op the sidecar has a source for.

    The *innermost* frame, which is the wrong one for anything dispatched
    through Pallas -- see :func:`read_callers`, which is what the kit uses.
    Kept because it is the raw reading of the field and the one to compare
    against when a caller attribution looks wrong.

    Ops without a source are simply absent rather than mapped to a
    placeholder, so a caller can tell "no source recorded" from "source
    recorded as empty".
    """
    return _by_weight(
        _device_ops(path),
        lambda metrics: (
            f"{shorten(metrics.source_info.file_name)}:{metrics.source_info.line_number}"
            if metrics.source_info.file_name else None))


def _location(frame: str) -> str:
    """``a/b.py:638:0`` -> ``a/b.py:638``. Stack frames carry a column; drop it.

    Keeping it would split one call site into several rows for no information,
    and would stop a stack frame from ever matching a ``source_info`` location.
    Requires *both* trailing components to be numeric, so a frame that carries
    only a line number keeps it rather than being truncated to a bare filename.
    """
    head, _, column = frame.rpartition(":")
    line = head.rpartition(":")[2]
    return head if column.isdigit() and line.isdigit() else frame


def caller_of(stack_frame: str) -> str | None:
    """The innermost frame of *stack_frame* that is not in :data:`PLUMBING`.

    Takes the raw newline-separated block the sidecar stores. Returns ``None``
    when every frame is plumbing, which is the honest answer -- better than
    naming a dispatch function as the owner of the work.
    """
    for frame in stack_frame.strip().split("\n"):
        frame = frame.strip()
        if not frame:
            continue
        location = _location(shorten(frame))
        if not location.startswith(PLUMBING):
            return location
    return None


def read_callers(path: str) -> dict[str, str]:
    """``{op name: "file.py:123"}`` for the innermost frame that is *not* plumbing.

    This is the useful reading of "whose code asked for it". The innermost frame
    alone answers a different and mostly uninteresting question -- for anything
    dispatched through Pallas it is the call wrapper, identically, for half the
    step. Walking out to the first frame in :data:`PLUMBING`'s complement gives
    the subsystem that owns the work, and it attributes the compiler glue with
    it: the reshapes and slices around a kernel land on the kernel's caller
    rather than in an anonymous ``layout`` pile.

    Falls back to the ``source_info`` location when an op has no stack recorded,
    and omits ops that have neither -- or whose owner is genuinely split across
    modules; see :func:`_by_weight`.
    """
    def locate(metrics):
        info = metrics.source_info
        found = caller_of(info.stack_frame)
        if found is None and info.file_name:
            found = f"{shorten(info.file_name)}:{info.line_number}"
        return found

    return _by_weight(_device_ops(path), locate)


def read_stacks(path: str) -> dict[str, list[str]]:
    """``{op name: [innermost frame, ...]}``. For digging, not for the tables.

    What to reach for from a REPL when a ``src`` rule is not firing and the
    question is which frames the op actually carried.
    """
    return {
        metrics.name: [shorten(frame) for frame in
                       metrics.source_info.stack_frame.strip().split("\n") if frame]
        for metrics in _device_ops(path)
        if metrics.source_info.stack_frame
    }
