#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize an xprof/XPlane TPU trace into the numbers a perf triage needs.

The headline number is **idle time**: span minus the union of the device op
intervals. A decode step that is neither bandwidth- nor FLOP-bound spends its
time in host-dispatch gaps and in collective waits, and neither shows up in a
per-kernel microbenchmark. Everything else here exists to attribute that time:
a top-op table, a category rollup, a gap histogram, and a per-rank skew table.

Input is either a single ``*.xplane.pb`` or a profile directory.
``profiler_trace.merge_rank_capture`` lays a run out as::

    <root>/plugins/profile/<timestamp>/rank<N>_<host>.xplane.pb

and the phased profiler (``runner/utils.PhaseBasedProfiler``) puts one such
tree under a per-phase directory named after ``InferencePhase`` in lower case::

    <root>/<phase>/plugins/profile/<timestamp>/rank<N>_<host>.xplane.pb
    <root>/<phase>/rank_<N>/batch_composition_stats_*.json

Both shapes are handled; a directory is grouped by phase and rank and reports
rank 0 unless told otherwise.

Two device timelines matter and they must not be added together:

* ``XLA Ops`` is the serialized TensorCore timeline. Busy, idle and gaps are
  computed from it alone.
* ``Async XLA Ops`` holds the ``async-start`` halves of collectives, the
  SparseCore kernels and HBM copies. These *overlap* the TensorCore timeline,
  so they are summarized separately. The blocking cost of an async op is its
  ``async-done``, which is already on ``XLA Ops``.

Because no trace exists until the profile is captured, parsing is deliberately
tolerant: unknown line names are reported rather than assumed away, missing
stats degrade to name-only bucketing, and ``--self-test`` exercises the
aggregation, gap and bucketing logic on synthetic events with no profiler
data and no TPU.

Examples::

    # one phase directory, rank 0
    python scripts/benchmarking/summarize_xplane.py /tmp/prof/decode_only

    # every rank of every phase, machine readable
    python scripts/benchmarking/summarize_xplane.py /tmp/prof --all-ranks \\
        --json /tmp/prof/summary.json

    # regroup the tables by XLA's `tf_op` metadata, which names the jaxpr
    # scope and so resolves the anonymous `fusion.N` rows
    python scripts/benchmarking/summarize_xplane.py /tmp/prof --by tf_op

    # A/B two runs
    python scripts/benchmarking/summarize_xplane.py \\
        --compare base.json candidate.json
"""

import argparse
import bisect
import contextlib
import dataclasses
import glob
import gzip
import io
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from typing import Any, Iterable, NamedTuple, Optional, Sequence

# Device planes are named `/device:TPU:<n>`; the SparseCore tiles of the same
# chip get their own planes, `/device:TPU:<n> SparseCore <m>`.
TPU_PLANE_RE = re.compile(r"^/device:TPU:(\d+)$")
TPU_SPARSECORE_PLANE_RE = re.compile(r"^/device:TPU:(\d+) SparseCore (\d+)$")

# Line names seen in a torch_xla/xprof TPU capture. Anything not listed is
# still counted and reported under "unrecognized lines" so a renamed line in a
# future xprof cannot silently drop work from the totals.
PRIMARY_OP_LINES = ("XLA Ops", "XLA Op", "TensorFlow Ops")
ASYNC_OP_LINES = ("Async XLA Ops", )
MODULE_LINES = ("XLA Modules", "XLA Module")
TRACEME_LINES = ("XLA TraceMe", "TraceMe")
STEP_LINES = ("Steps", "Step")
# Performance counters and empty overlay lines: no wall-clock meaning.
IGNORED_LINE_RE = re.compile(r"^(_counters_|counters_\d+|Scalar Unit|"
                             r"TC Overlay|Framework Ops|Source)$")

# Gap histogram edges in microseconds. Log-ish, because the interesting range
# spans a sub-microsecond issue gap and a multi-millisecond host stall.
GAP_BUCKET_EDGES_US = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
                       500.0, 1000.0, 2000.0)

# Bucket table: ordered (name, patterns); first match wins, so the specific
# kernels come before the generic HLO shapes. Patterns are matched
# case-insensitively against `<op name> | <tf_op> | <hlo_category>`, which is
# what lets a rule key off either the kernel's own name or the jaxpr scope XLA
# recorded for it.
#
# Every pattern below is taken from a name this repo actually emits or from a
# name observed in a captured trace; see the comment on each line. The two
# naming mechanisms are `jax.named_scope(...)` / `pl.pallas_call(name=...)`
# (which become the HLO instruction name, hence the op name) and torch_tpu's
# `pallas.jax_op`, whose wrapper shows up in `tf_op` as
# `custom_kernel/custom_kernel/<inner scope>/...`.
BUCKET_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # GDN first: its conv kernel would otherwise be caught by the generic
    # `convolution` rule under dense matmuls.
    (
        "gdn/linear-attn",
        (
            r"\bgdn\b|_gdn|gdn_",  # decoding_gdn-bt_2, fused_conv1d_gdn_<mode>
            r"gated_delta_rule",  # ragged_gated_delta_rule_wrapper (tf_op)
            r"causal_conv1d",  # ragged_causal_conv1d_kernel.<n>
            r"kimi_(kda|sconv|chunk_kda|dispatched_kda)",  # kda_attention_op
            r"mamba_state",  # pallas::mamba_state_block_copy
            r"newton_schulz",  # newton_schulz_inverse_kernel
        )),
    (
        "attention",
        (
            r"\bRPA[a-z]*[-_]",  # RPAd-/RPAm-/RPAp-/RPA-HD_64-, MLA-RPA-
            r"rpa_kernel|rpa_metadata_schedule",  # pallas::rpa_kernel[_batched]
            r"ragged_paged_attention",  # ragged_paged_attention_kernel
            r"\bMLA[-_]|mla_attention|sparse_mla",  # MLA-<sym>-bq_..., ops
            r"streamidx|streamindex",  # StreamIdxTC-<sym>-bq_...
            r"deepseek_v4_(indexer|compressor)",  # dsv4 custom ops
            r"pcp_streaming",  # pcp_streaming_attention_* kernels
            r"flash_attention",  # pallas::vision_flash_attention
        )),
    (
        "moe-gmm",
        (
            r"\bgmm_v2-|\bgmm_v2\b",  # gmm_v2-g_64-m_65536-k_2048-act_silu-...
            r"\bgmm\b|gmm_wrapper",  # megablox gmm entry points
            r"fused_moe_k-|fused_moe_kernel",  # fused_moe/v1 scope, torch op
            r"megablox",
        )),
    (
        "moe-permute/combine",
        (
            r"sc_ragged_(gather|gather_reduce|scatter)",  # sparse_core kernels
            r"sc_csa_gather|sc_gather_reduce",  # sparse_core/gather_reduce
            r"sparse-core-data-format-call",  # SC layout call around the above
            r"gather_offload",  # gather_offload_async_{start,done}
            r"(gather|scatter)_custom_fusion",  # XLA's gather/scatter fusions
            r"one_hot|onehot",  # the one-hot permute/combine fallback
            # The routing argsort lives inside the MoE jax function, so key it
            # off the jaxpr scope. A bare `sort` (the sampler's) stays out.
            r"custom_kernel/[^|]*\b(sort|argsort|top_k)\b",
            # The one-hot permute/combine path is a plain `@` and a plain
            # `.sum(axis=1)` inside the fused-MoE jax op (fused_moe_gmm.py
            # `onehot @ hidden_states_local` and `jnp.matmul(combine, ...)`
            # plus the `combine` reduction), so XLA labels it with nothing but
            # the jax primitive directly under torch_tpu's wrapper. Those two
            # primitives are the *only* raw matmul/reduce in
            # `layers/` outside a Pallas kernel, and a captured trace shows
            # them in the expected 2:1 ratio, so match the scope exactly.
            # A dense matmul always carries an inner scope
            # (`jit(_quantized_matmul_jax)/dot_general`, `mm/mm`) and so is
            # not caught here.
            r"custom_kernel/custom_kernel/(dot_general|reduce_sum)\b",
        )),
    (
        "collectives",
        (
            r"all-reduce|all_reduce",  # all-reduce.<n>.cloned.1.call-{start,done}
            r"all-gather|all_gather",  # distributed.all_gather_into_tensor
            r"reduce-scatter|reduce_scatter",
            r"collective-permute|collective_permute",
            r"all-to-all|all_to_all",
            r"barrier-cores",  # XLA TraceMe cross-core wait
            r"megascale",  # DCN plane / multi-slice transfers
        )),
    (
        "dense-matmul",
        (
            r"quantized_matmul",  # quantized_matmul_kernel_<b>_<o>_<i>
            r"\bdot_general\b|\bdot\b",  # tf_op of any XLA matmul
            r"\bmatmul\b|\beinsum\b",
            r"\b(mm|bmm|addmm|linear)/",  # torch aten ops in tf_op
            r"convolution",  # XLA lowers TPU matmuls to `convolution fusion`
        )),
    # Pallas kernels whose inner scope did not survive into the HLO name show
    # up as torch_tpu's wrapper, `custom_kernel.<n>`. Anchored to the name
    # field so it only fires when nothing more specific matched.
    ("pallas-unnamed", (r"^custom_kernel(\.\d+)*\s*(\||$)", )),
    (
        "data-movement",
        (
            r"async-copy|copy-start|copy-done|\bcopy\b",
            r"data formatting",  # hlo_category
            r"\bbitcast\b|\breshape\b|\btranspose\b|xpose_",
            r"\bpad\b|\bslice\b|dynamic-slice|dynamic-update-slice",
            r"\bconcatenate\b|\bcat/|\bsplit\b",
            r"as_strided",
            r"\biota\b|\bbroadcast\b",
        )),
    (
        "rope/norm/elementwise",
        (
            r"\brope\b|rotary",
            r"rms_?norm|layer_?norm|\bnorm\b|\brsqrt\b|\bmean\b",
            r"\bsilu\b|\bgelu\b|swiglu|\bsigmoid\b|logistic|\bsoftmax\b",
            r"non-fusion elementwise",  # hlo_category
            r"_fusion\b",  # add_rsqrt_fusion, multiply_add_fusion, ...
            r"^(add|sub|subtract|mul|multiply|divide|negate|convert|select|"
            r"select_n|compare|clamp|maximum|minimum|exponential|abs|and|or|"
            r"not|eq|lt|gt|ge|le|reduce|reduce_sum|reduce_max|"
            r"convert_element_type)([_;./]|\s*\||$)",
        )),
)

OTHER_BUCKET = "other"
# Always printed, so a hot op that no rule claimed cannot hide in the total.
OTHER_LISTING_LIMIT = 15


class Event(NamedTuple):
    """One device-timeline event, already normalized.

    `dur_ns` is the timeline duration (what gaps are computed from) and
    `device_ns` is the `device_duration_ps` stat when the trace carries one.
    They differ for ops the hardware overlaps internally, so both are kept.
    """

    name: str
    start_ns: int
    end_ns: int
    dur_ns: float
    device_ns: float
    line: str
    match_text: str


@dataclasses.dataclass(frozen=True)
class TraceFile:
    """One rank's xplane file, with the phase/rank it was captured under."""

    path: str
    phase: str
    rank: int
    timestamp: str


Rule = tuple[str, re.Pattern]


def _compile_rules(extra: Sequence[tuple[str, str]] = ()) -> list[Rule]:
    """Compile the bucket table into (name, alternation) pairs, in order.

    `extra` rules are prepended so a caller can override a built-in bucket for
    a one-off investigation without editing the table.
    """
    rules: list[Rule] = []
    for name, pattern in extra:
        rules.append((name, re.compile(pattern, re.IGNORECASE)))
    for name, patterns in BUCKET_RULES:
        joined = "|".join(f"(?:{p})" for p in patterns)
        rules.append((name, re.compile(joined, re.IGNORECASE)))
    return rules


def classify(match_text: str, rules: Sequence[Rule]) -> str:
    for name, pattern in rules:
        if pattern.search(match_text):
            return name
    return OTHER_BUCKET


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# An XEvent's metadata name is sometimes the whole HLO instruction text
# (`%conditional.90 = (f32[...]) conditional(...)`) and sometimes just the
# instruction name, depending on the xprof version and on whether the visitor
# hands back `display_name`. Reduce both to the instruction name.
_HLO_TEXT_RE = re.compile(r"^\s*%?([A-Za-z0-9_.\-]+)\s*=")
_NUMERIC_SUFFIX_RE = re.compile(r"\.\d+")


def short_op_name(raw: str) -> str:
    matched = _HLO_TEXT_RE.match(raw)
    if matched:
        return matched.group(1)
    name = raw.strip()
    if name.startswith("%"):
        name = name[1:]
    # A stray shape or operand list means the name was never an instruction
    # name; keep the head so the row is still readable.
    return name.split(" ")[0] if " " in name else name


def base_op_name(name: str) -> str:
    """Drop HLO's numeric ids so the 124 copies of one op form one row.

    `conditional.90` -> `conditional`;
    `all-reduce.260.cloned.1.call-done` -> `all-reduce.cloned.call-done`.
    Kernel names carry their config in dashes, not dots, so they survive:
    `gmm_v2-g_64-m_65536-...` is unchanged.
    """
    stripped = _NUMERIC_SUFFIX_RE.sub("", name)
    return stripped or name


# ---------------------------------------------------------------------------
# Trace discovery
# ---------------------------------------------------------------------------

_RANK_FILE_RE = re.compile(r"^rank(\d+)_")
_RANK_DIR_RE = re.compile(r"^rank_(\d+)$")


def _classify_trace_path(root: str, path: str) -> TraceFile:
    """Derive (phase, rank, timestamp) from a trace path under `root`."""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    rank = 0
    matched = _RANK_FILE_RE.match(parts[-1])
    if matched:
        rank = int(matched.group(1))
    else:
        for part in parts:
            dir_match = _RANK_DIR_RE.match(part)
            if dir_match:
                rank = int(dir_match.group(1))
                break

    timestamp = ""
    if "profile" in parts:
        idx = parts.index("profile")
        if idx + 1 < len(parts) - 1:
            timestamp = parts[idx + 1]

    # Everything above the `plugins/` (or `rank_<N>/`) subtree is the phase
    # directory the phased profiler created. A plain `torch_profiler_dir`
    # capture has nothing there, hence the "-" placeholder.
    head = parts[:-1]
    for stop in ("plugins", ):
        if stop in head:
            head = head[:head.index(stop)]
    head = [p for p in head if not _RANK_DIR_RE.match(p)]
    phase = "/".join(head) if head else "-"
    return TraceFile(path=path, phase=phase, rank=rank, timestamp=timestamp)


def discover_traces(path: str) -> list[TraceFile]:
    """Return every xplane file under `path`, tagged with phase and rank."""
    if os.path.isfile(path):
        # Root the single file at the directory holding `plugins/`, so the
        # run timestamp and the `rank<N>_` prefix are still recovered. The
        # phase stays "-": one file names no grouping.
        absolute = os.path.abspath(path)
        parts = absolute.split(os.sep)
        if "plugins" in parts:
            root = os.sep.join(parts[:parts.index("plugins")]) or os.sep
        else:
            root = os.path.dirname(absolute) or "."
        return [_classify_trace_path(root, absolute)]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    found: list[str] = []
    for pattern in ("*.xplane.pb", "*.xplane.pb.gz"):
        found.extend(
            glob.glob(os.path.join(path, "**", pattern), recursive=True))
    traces = [_classify_trace_path(path, p) for p in sorted(set(found))]
    if not traces:
        raise FileNotFoundError(f"no *.xplane.pb under {path}")
    return traces


def select_traces(
    traces: Sequence[TraceFile],
    *,
    phase: Optional[str],
    rank: Optional[int],
    all_ranks: bool,
) -> list[TraceFile]:
    """Filter by phase/rank, defaulting to rank 0 of every phase."""
    picked = list(traces)
    if phase:
        picked = [t for t in picked if t.phase == phase]
        if not picked:
            have = sorted({t.phase for t in traces})
            raise SystemExit(f"no traces for phase {phase!r}; have {have}")
    if not all_ranks:
        wanted = 0 if rank is None else rank
        by_rank = [t for t in picked if t.rank == wanted]
        if not by_rank:
            have = sorted({t.rank for t in picked})
            raise SystemExit(f"no traces for rank {wanted}; have {have}")
        picked = by_rank
    return picked


# ---------------------------------------------------------------------------
# XPlane reading
# ---------------------------------------------------------------------------


def _stats_dict(event: Any) -> dict[str, Any]:
    """Best-effort read of an event's stats as a name -> value mapping.

    Different xprof builds expose a different slice of the XStat set (some
    attach `tf_op`/`hlo_category` to the event, some only to the shared event
    metadata that the Python visitor does not walk), so this never raises: a
    missing stat just means name-only bucketing for that op.
    """
    stats: dict[str, Any] = {}
    try:
        for name, value in event.stats:
            stats[name] = value
    except Exception:  # pragma: no cover - depends on the xprof build
        return {}
    return stats


def _event_wall_ns(event: Any, device_ps: float) -> tuple[int, float]:
    """Return (start_ns, duration_ns) for an event, tolerating old builds."""
    start_ns = int(getattr(event, "start_ns", 0) or 0)
    duration_ns = getattr(event, "duration_ns", None)
    if duration_ns is None:
        end_ns = getattr(event, "end_ns", None)
        if end_ns is not None:
            duration_ns = float(end_ns) - start_ns
        else:
            duration_ns = device_ps / 1e3
    return start_ns, float(duration_ns)


def _device_duration_reader():
    """Return `device_time._event_duration_ps`, the shared stat reader.

    Imported lazily and behind a `sys.path` fixup: the sibling `kernels`
    package pulls in jax, and `--self-test` has to run without it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from kernels.device_time import _event_duration_ps
    return _event_duration_ps


def read_plane_events(
        plane: Any) -> tuple[dict[str, list[Event]], dict[str, int]]:
    """Read every line of a plane into normalized events, keyed by line name.

    Also returns the raw event count per line, so lines that were dropped for
    having no duration are still visible in the report.
    """
    _event_duration_ps = _device_duration_reader()

    by_line: dict[str, list[Event]] = {}
    counts: dict[str, int] = {}
    for line in plane.lines:
        line_name = getattr(line, "name", "") or "<unnamed>"
        events: list[Event] = []
        raw_count = 0
        for event in line.events:
            raw_count += 1
            device_ps = float(_event_duration_ps(event) or 0.0)
            start_ns, duration_ns = _event_wall_ns(event, device_ps)
            if duration_ns <= 0:
                continue
            raw_name = getattr(event, "name", "") or ""
            name = short_op_name(raw_name)
            stats = _stats_dict(event)
            match_text = " | ".join((
                name,
                str(stats.get("tf_op", "")),
                str(stats.get("hlo_category", "")),
            ))
            events.append(
                Event(
                    name=name,
                    start_ns=start_ns,
                    end_ns=int(start_ns + duration_ns),
                    dur_ns=duration_ns,
                    device_ns=device_ps / 1e3,
                    line=line_name,
                    match_text=match_text,
                ))
        by_line[line_name] = events
        counts[line_name] = raw_count
    return by_line, counts


def open_profile(path: str) -> tuple[Any, Optional[str]]:
    """Open an xplane file, transparently decompressing a `.gz`.

    Returns (ProfileData, tempdir-to-clean-up).
    """
    import jax

    tmpdir = None
    read_path = path
    if path.endswith(".gz"):
        tmpdir = tempfile.mkdtemp(prefix="summarize_xplane-")
        read_path = os.path.join(tmpdir, os.path.basename(path)[:-len(".gz")])
        with gzip.open(path, "rb") as src, open(read_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return jax.profiler.ProfileData.from_file(read_path), tmpdir


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def merged_busy_ns(events: Sequence[Event]) -> float:
    """Union of the event intervals, in ns.

    A union rather than a sum: the TensorCore line should not overlap itself,
    but a trace that merges several queues into one line can, and a sum would
    then report more busy time than the span contains.
    """
    if not events:
        return 0.0
    ordered = sorted(events, key=lambda e: e.start_ns)
    total = 0.0
    cur_start = ordered[0].start_ns
    cur_end = ordered[0].end_ns
    for event in ordered[1:]:
        if event.start_ns > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = event.start_ns, event.end_ns
        else:
            cur_end = max(cur_end, event.end_ns)
    total += cur_end - cur_start
    return float(total)


class Gap(NamedTuple):
    start_ns: int
    dur_ns: float
    before: str
    after: str


def find_gaps(events: Sequence[Event]) -> list[Gap]:
    """Inter-op gaps on a single timeline, with the ops on either side."""
    if len(events) < 2:
        return []
    ordered = sorted(events, key=lambda e: e.start_ns)
    gaps: list[Gap] = []
    frontier = ordered[0].end_ns
    frontier_name = ordered[0].name
    for event in ordered[1:]:
        if event.start_ns > frontier:
            gaps.append(
                Gap(
                    start_ns=frontier,
                    dur_ns=float(event.start_ns - frontier),
                    before=frontier_name,
                    after=event.name,
                ))
        if event.end_ns >= frontier:
            frontier = event.end_ns
            frontier_name = event.name
    return gaps


def gap_histogram(gaps: Sequence[Gap]) -> list[dict[str, Any]]:
    """Bucket gaps by duration; returns one row per bucket, always all rows."""
    edges = GAP_BUCKET_EDGES_US
    counts = [0] * (len(edges) + 1)
    totals = [0.0] * (len(edges) + 1)
    for gap in gaps:
        us = gap.dur_ns / 1e3
        # bisect_right, so a gap landing exactly on an edge falls in the
        # bucket the label says it does: the ranges are half-open [low, high).
        idx = bisect.bisect_right(edges, us)
        counts[idx] += 1
        totals[idx] += us
    rows = []
    for i, count in enumerate(counts):
        low = 0.0 if i == 0 else edges[i - 1]
        high = edges[i] if i < len(edges) else None
        rows.append({
            "low_us": low,
            "high_us": high,
            "count": count,
            "total_us": totals[i],
        })
    return rows


def _group_key(event: Event, by: str) -> str:
    if by == "tf_op":
        # match_text is "<name> | <tf_op> | <hlo_category>".
        parts = event.match_text.split(" | ")
        return parts[1] if len(parts) > 1 and parts[1] else f"<{event.name}>"
    if by == "category":
        parts = event.match_text.split(" | ")
        return parts[2] if len(parts) > 2 and parts[2] else "<none>"
    if by == "exact":
        return event.name
    return base_op_name(event.name)


def aggregate_ops(
    events: Sequence[Event],
    *,
    by: str,
    rules: Sequence[Rule],
) -> list[dict[str, Any]]:
    """Per-op totals, sorted by total time descending."""
    acc: dict[str, dict[str, Any]] = {}
    for event in events:
        key = _group_key(event, by)
        row = acc.get(key)
        if row is None:
            row = {
                "name": key,
                "count": 0,
                "total_us": 0.0,
                "device_us": 0.0,
                "bucket": classify(event.match_text, rules),
                "sample": event.match_text,
            }
            acc[key] = row
        row["count"] += 1
        row["total_us"] += event.dur_ns / 1e3
        row["device_us"] += event.device_ns / 1e3
    rows = sorted(acc.values(), key=lambda r: r["total_us"], reverse=True)
    for row in rows:
        row["mean_us"] = row["total_us"] / row["count"]
    return rows


def rollup_buckets(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = acc.setdefault(row["bucket"], {
            "bucket": row["bucket"],
            "count": 0,
            "total_us": 0.0,
        })
        bucket["count"] += row["count"]
        bucket["total_us"] += row["total_us"]
    return sorted(acc.values(), key=lambda b: b["total_us"], reverse=True)


def _field_coverage(events: Sequence[Event], field: int) -> float:
    """Fraction of events whose `match_text` carries a non-empty field.

    Bucketing is much sharper when XLA's `tf_op` and `hlo_category` stats are
    visible through the profiler's Python visitor, and which stats it exposes
    varies by build. Measuring the coverage lets the report say so rather than
    quietly dumping half the time into "other".
    """
    if not events:
        return 0.0
    present = 0
    for event in events:
        parts = event.match_text.split(" | ")
        if len(parts) > field and parts[field].strip():
            present += 1
    return present / len(events)


def summarize_events(
    events: Sequence[Event],
    *,
    by: str,
    rules: Sequence[Rule],
    top: int,
) -> dict[str, Any]:
    """The whole per-timeline summary: busy/idle, ops, buckets, gaps."""
    if not events:
        return {
            "num_events": 0,
            "span_us": 0.0,
            "busy_us": 0.0,
            "idle_us": 0.0,
            "idle_pct": 0.0,
            "op_total_us": 0.0,
            "device_busy_us": 0.0,
            "tf_op_coverage": 0.0,
            "category_coverage": 0.0,
            "ops": [],
            "buckets": [],
            "other_ops": [],
            "gap_histogram": gap_histogram([]),
            "num_gaps": 0,
            "gap_total_us": 0.0,
            "largest_gaps": [],
        }
    start = min(e.start_ns for e in events)
    end = max(e.end_ns for e in events)
    span_ns = float(end - start)
    busy_ns = merged_busy_ns(events)
    gaps = find_gaps(events)
    rows = aggregate_ops(events, by=by, rules=rules)
    buckets = rollup_buckets(rows)

    busy_us = busy_ns / 1e3
    span_us = span_ns / 1e3
    for row in rows:
        row["pct_busy"] = 100.0 * row["total_us"] / busy_us if busy_us else 0.0
        row["pct_span"] = 100.0 * row["total_us"] / span_us if span_us else 0.0
    for bucket in buckets:
        bucket["pct_busy"] = (100.0 * bucket["total_us"] /
                              busy_us if busy_us else 0.0)

    idle_us = span_us - busy_us
    other_ops = [r for r in rows if r["bucket"] == OTHER_BUCKET]
    # `op_total_us` is the sum of the per-op durations. It exceeds `busy_us`
    # when the line nests events -- an HLO `conditional` sits on the timeline
    # alongside the ops of the branch it ran -- which is why busy/idle use the
    # interval union and only the op table uses the sum.
    op_total_us = sum(e.dur_ns for e in events) / 1e3
    device_us = sum(e.device_ns for e in events) / 1e3
    largest = [{
        "at_us": (g.start_ns - start) / 1e3,
        "gap_us": g.dur_ns / 1e3,
        "before": g.before,
        "after": g.after,
    } for g in sorted(gaps, key=lambda g: g.dur_ns, reverse=True)[:10]]
    return {
        "num_events": len(events),
        "span_us": span_us,
        "busy_us": busy_us,
        "idle_us": idle_us,
        "idle_pct": 100.0 * idle_us / span_us if span_us else 0.0,
        "op_total_us": op_total_us,
        "device_busy_us": device_us,
        "tf_op_coverage": _field_coverage(events, 1),
        "category_coverage": _field_coverage(events, 2),
        "ops": rows[:top],
        "buckets": buckets,
        "other_ops": other_ops[:OTHER_LISTING_LIMIT],
        "gap_histogram": gap_histogram(gaps),
        "num_gaps": len(gaps),
        "gap_total_us": sum(g.dur_ns for g in gaps) / 1e3,
        "largest_gaps": largest,
    }


def summarize_plane(
    plane_name: str,
    by_line: dict[str, list[Event]],
    counts: dict[str, int],
    *,
    by: str,
    rules: Sequence[Rule],
    top: int,
) -> dict[str, Any]:
    """Summarize one plane: the primary timeline plus the side timelines."""
    primary_name = next((n for n in PRIMARY_OP_LINES if by_line.get(n)), None)
    unrecognized = {}
    known = set(PRIMARY_OP_LINES + ASYNC_OP_LINES + MODULE_LINES +
                TRACEME_LINES + STEP_LINES)
    for name, events in by_line.items():
        if name in known or IGNORED_LINE_RE.match(name):
            continue
        unrecognized[name] = {
            "num_events": counts.get(name, len(events)),
            "total_us": sum(e.dur_ns for e in events) / 1e3,
        }
    if primary_name is None:
        # No line we recognize as the op timeline. Rather than report zero,
        # fall back to the busiest unrecognized line and say so.
        candidates = [(sum(e.dur_ns for e in evs), n)
                      for n, evs in by_line.items()
                      if evs and not IGNORED_LINE_RE.match(n)]
        if candidates:
            primary_name = max(candidates)[1]

    primary = by_line.get(primary_name or "", [])
    summary = summarize_events(primary, by=by, rules=rules, top=top)
    summary["plane"] = plane_name
    summary["primary_line"] = primary_name
    summary["primary_line_recognized"] = primary_name in PRIMARY_OP_LINES
    summary["lines"] = {
        name: {
            "num_events": counts.get(name, len(events)),
            "total_us": sum(e.dur_ns for e in events) / 1e3,
        }
        for name, events in by_line.items()
    }
    summary["unrecognized_lines"] = unrecognized

    async_events: list[Event] = []
    for name in ASYNC_OP_LINES:
        async_events.extend(by_line.get(name, []))
    if async_events:
        async_rows = aggregate_ops(async_events, by=by, rules=rules)
        summary["async"] = {
            "num_events": len(async_events),
            "total_us": sum(e.dur_ns for e in async_events) / 1e3,
            "ops": async_rows[:top],
            "buckets": rollup_buckets(async_rows),
        }

    module_events: list[Event] = []
    for name in MODULE_LINES:
        module_events.extend(by_line.get(name, []))
    if module_events:
        durations = sorted(e.dur_ns / 1e3 for e in module_events)
        summary["modules"] = {
            "count": len(durations),
            "total_us": sum(durations),
            "median_us": statistics.median(durations),
            "max_us": durations[-1],
        }

    traceme: list[Event] = []
    for name in TRACEME_LINES:
        traceme.extend(by_line.get(name, []))
    if traceme:
        rows = aggregate_ops(traceme, by="base", rules=rules)
        summary["traceme"] = rows[:top]
    return summary


def summarize_trace(
    trace: TraceFile,
    *,
    by: str,
    rules: Sequence[Rule],
    top: int,
    include_sparsecore: bool,
) -> dict[str, Any]:
    """Summarize every TPU plane of one rank's xplane file."""
    profile, tmpdir = open_profile(trace.path)
    try:
        planes = []
        for plane in profile.planes:
            name = getattr(plane, "name", "") or ""
            if TPU_PLANE_RE.match(name):
                planes.append(plane)
            elif include_sparsecore and TPU_SPARSECORE_PLANE_RE.match(name):
                planes.append(plane)
        if not planes:
            available = sorted(
                getattr(p, "name", "") or "" for p in profile.planes)
            raise SystemExit(
                f"no /device:TPU:<n> plane in {trace.path}; planes present: "
                f"{available}")
        out = {
            "path": trace.path,
            "phase": trace.phase,
            "rank": trace.rank,
            "timestamp": trace.timestamp,
            "planes": [],
        }
        for plane in planes:
            by_line, counts = read_plane_events(plane)
            out["planes"].append(
                summarize_plane(
                    getattr(plane, "name", "") or "",
                    by_line,
                    counts,
                    by=by,
                    rules=rules,
                    top=top,
                ))
        return out
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_us(value: float) -> str:
    return f"{value:,.1f}"


def print_plane(plane: dict[str, Any], *, top: int) -> None:
    # `.get` on the plane-level extras: --compare happily reads a --json dump
    # written by an older copy of this script, which may not carry them.
    print(f"\n  plane {plane.get('plane', '?')}  (timeline line: "
          f"{plane.get('primary_line')!r})")
    if not plane.get("primary_line_recognized"):
        print("    WARNING: no recognized op line; the numbers below come "
              "from the busiest line found. Check `lines` in --json.")
    print(f"    span        {_fmt_us(plane['span_us']):>14} us")
    print(f"    busy        {_fmt_us(plane['busy_us']):>14} us  "
          f"({100.0 - plane['idle_pct']:5.1f}%)")
    print(f"    IDLE/GAP    {_fmt_us(plane['idle_us']):>14} us  "
          f"({plane['idle_pct']:5.1f}%)   <-- dispatch gaps + waits")
    print(f"    device time {_fmt_us(plane['device_busy_us']):>14} us  "
          "(sum of device_duration_ps; overlaps not removed)")
    op_total = plane.get("op_total_us", 0.0)
    if op_total > plane["busy_us"] * 1.01:
        print(f"    op sum      {_fmt_us(op_total):>14} us  "
              f"({op_total / max(plane['busy_us'], 1e-9):.2f}x busy: the "
              "timeline nests events, e.g. an HLO `conditional` next to its "
              "branch's ops. The op/bucket totals below double-count that "
              "nesting; busy/idle/gaps do not.)")
    if "modules" in plane:
        mod = plane["modules"]
        print(f"    modules     {mod['count']:>14} executions, median "
              f"{_fmt_us(mod['median_us'])} us, max "
              f"{_fmt_us(mod['max_us'])} us")
    tf_cov = plane.get("tf_op_coverage", 0.0)
    cat_cov = plane.get("category_coverage", 0.0)
    print(f"    metadata    {100.0 * tf_cov:13.0f}% of events carry `tf_op`, "
          f"{100.0 * cat_cov:.0f}% carry `hlo_category`")
    if tf_cov < 0.05:
        print("      NOTE: this profiler build does not surface XLA's op "
              "metadata, so buckets fall back to op names alone and "
              "anonymous `fusion.N` rows cannot be attributed. Expect a "
              "large 'other'.")

    if plane.get("unrecognized_lines"):
        print("    unrecognized lines (not in busy/idle):")
        for name, info in sorted(plane["unrecognized_lines"].items(),
                                 key=lambda kv: -kv[1]["total_us"]):
            print(f"      {name:<40} n={info['num_events']:<8} "
                  f"{_fmt_us(info['total_us'])} us")

    print(f"\n    top {top} ops")
    print(f"      {'op':<52} {'n':>7} {'total us':>12} {'mean us':>10} "
          f"{'%busy':>7} {'%span':>7}  bucket")
    for row in plane["ops"]:
        print(f"      {row['name'][:52]:<52} {row['count']:>7} "
              f"{_fmt_us(row['total_us']):>12} "
              f"{row['mean_us']:>10.2f} {row['pct_busy']:>7.1f} "
              f"{row['pct_span']:>7.1f}  {row['bucket']}")

    print("\n    category rollup")
    print(f"      {'bucket':<24} {'n':>8} {'total us':>14} {'%busy':>8} "
          f"{'%span':>8}")
    span_us = plane["span_us"] or 1.0
    for bucket in plane["buckets"]:
        print(f"      {bucket['bucket']:<24} {bucket['count']:>8} "
              f"{_fmt_us(bucket['total_us']):>14} "
              f"{bucket['pct_busy']:>8.1f} "
              f"{100.0 * bucket['total_us'] / span_us:>8.1f}")

    print(f"\n    '{OTHER_BUCKET}' contents (top {OTHER_LISTING_LIMIT}) "
          "-- name | tf_op | hlo_category")
    if not plane["other_ops"]:
        print("      (empty)")
    for row in plane["other_ops"]:
        print(f"      {row['name'][:52]:<52} {row['count']:>7} "
              f"{_fmt_us(row['total_us']):>12}  {row['sample'][:70]}")
    if plane["other_ops"]:
        print("      (anonymous `fusion.N`/`conditional.N` rows carry no "
              "kernel name; rerun with --by tf_op to see the jaxpr scope, "
              "then --bucket NAME=REGEX to claim them)")

    print(f"\n    gaps: {plane['num_gaps']} totalling "
          f"{_fmt_us(plane['gap_total_us'])} us")
    print(f"      {'range (us)':<20} {'count':>9} {'total us':>14} "
          f"{'%idle':>8}")
    idle = plane["idle_us"] or 1.0
    for row in plane["gap_histogram"]:
        high = "inf" if row["high_us"] is None else f"{row['high_us']:g}"
        label = f"[{row['low_us']:g}, {high})"
        print(f"      {label:<20} {row['count']:>9} "
              f"{_fmt_us(row['total_us']):>14} "
              f"{100.0 * row['total_us'] / idle:>8.1f}")

    print("\n    10 largest gaps")
    print(f"      {'at us':>12} {'gap us':>11}  {'op before gap':<34} "
          f"{'op after gap':<34}")
    for gap in plane["largest_gaps"]:
        print(f"      {gap['at_us']:>12,.1f} {gap['gap_us']:>11,.1f}  "
              f"{gap['before'][:34]:<34} {gap['after'][:34]:<34}")

    if "async" in plane:
        async_info = plane["async"]
        print(f"\n    async timeline ({ASYNC_OP_LINES[0]}): "
              f"{async_info['num_events']} events, "
              f"{_fmt_us(async_info['total_us'])} us "
              "(overlaps the timeline above; NOT in busy/idle)")
        for row in async_info["ops"][:min(top, 12)]:
            print(f"      {row['name'][:52]:<52} {row['count']:>7} "
                  f"{_fmt_us(row['total_us']):>12}  {row['bucket']}")

    if plane.get("traceme"):
        print("\n    XLA TraceMe")
        for row in plane["traceme"][:8]:
            print(f"      {row['name'][:52]:<52} {row['count']:>7} "
                  f"{_fmt_us(row['total_us']):>12}")


def print_rank_comparison(profiles: Sequence[dict[str, Any]]) -> None:
    """Per-rank busy/idle, plus the skew across ranks of one phase.

    DP replicas run the same graph, so a spread here means one rank is waiting
    on another and the collective is paying for it.
    """
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        by_phase.setdefault(profile["phase"], []).append(profile)
    for phase, group in sorted(by_phase.items()):
        if len(group) < 2:
            continue
        print(f"\n=== per-rank comparison, phase {phase} ===")
        print(f"  {'rank':>5} {'plane':<18} {'span us':>13} {'busy us':>13} "
              f"{'idle us':>13} {'idle %':>8} {'gaps':>7} {'max gap us':>12}")
        idles = []
        busies = []
        for profile in sorted(group, key=lambda p: p["rank"]):
            for plane in profile["planes"]:
                max_gap = max((g["gap_us"] for g in plane["largest_gaps"]),
                              default=0.0)
                print(f"  {profile['rank']:>5} {plane['plane']:<18} "
                      f"{_fmt_us(plane['span_us']):>13} "
                      f"{_fmt_us(plane['busy_us']):>13} "
                      f"{_fmt_us(plane['idle_us']):>13} "
                      f"{plane['idle_pct']:>8.1f} {plane['num_gaps']:>7} "
                      f"{max_gap:>12,.1f}")
                idles.append(plane["idle_us"])
                busies.append(plane["busy_us"])
        if len(busies) > 1:
            print(f"  skew: busy {max(busies) - min(busies):,.1f} us, "
                  f"idle {max(idles) - min(idles):,.1f} us "
                  f"(max busy {max(busies):,.1f}, min {min(busies):,.1f})")


def print_report(summary: dict[str, Any], *, top: int) -> None:
    print(f"source: {summary['source']}")
    for profile in summary["profiles"]:
        print(f"\n=== phase {profile['phase']}  rank {profile['rank']}  "
              f"ts {profile['timestamp'] or '-'} ===")
        print(f"  {profile['path']}")
        for plane in profile["planes"]:
            print_plane(plane, top=top)
    print_rank_comparison(summary["profiles"])


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def load_summary(
    path: str,
    *,
    by: str,
    rules: Sequence[Rule],
    top: int,
    include_sparsecore: bool,
) -> dict[str, Any]:
    """Accept either a `--json` dump or a raw trace path."""
    if path.endswith(".json") and os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    traces = discover_traces(path)
    picked = select_traces(traces, phase=None, rank=0, all_ranks=False)
    return {
        "source":
        path,
        "profiles": [
            summarize_trace(t,
                            by=by,
                            rules=rules,
                            top=top,
                            include_sparsecore=include_sparsecore)
            for t in picked
        ],
    }


def _plane_index(summary: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    out = {}
    for profile in summary["profiles"]:
        for plane in profile["planes"]:
            out[(profile["phase"], profile["rank"], plane["plane"])] = plane
    return out


def _delta_row(label: str, a_val: float, b_val: float) -> str:
    delta = b_val - a_val
    pct = (100.0 * delta / a_val) if a_val else float("nan")
    return (f"    {label:<26} {a_val:>14,.1f} {b_val:>14,.1f} "
            f"{delta:>+14,.1f} {pct:>+8.1f}%")


def print_compare(a: dict[str, Any], b: dict[str, Any]) -> None:
    print(f"A: {a['source']}\nB: {b['source']}")
    index_a = _plane_index(a)
    index_b = _plane_index(b)
    keys = sorted(set(index_a) & set(index_b), key=str)
    only_a = sorted(set(index_a) - set(index_b), key=str)
    only_b = sorted(set(index_b) - set(index_a), key=str)
    if only_a:
        print(f"only in A: {only_a}")
    if only_b:
        print(f"only in B: {only_b}")
    for key in keys:
        pa, pb = index_a[key], index_b[key]
        print(f"\n=== phase {key[0]}  rank {key[1]}  {key[2]} ===")
        print(f"    {'metric':<26} {'A':>14} {'B':>14} {'delta':>14} "
              f"{'delta %':>9}")
        print(_delta_row("span us", pa["span_us"], pb["span_us"]))
        print(_delta_row("busy us", pa["busy_us"], pb["busy_us"]))
        print(_delta_row("idle us", pa["idle_us"], pb["idle_us"]))
        print(
            _delta_row("gap total us", pa["gap_total_us"], pb["gap_total_us"]))
        print(
            _delta_row("gap count", float(pa["num_gaps"]),
                       float(pb["num_gaps"])))
        buckets_a = {x["bucket"]: x["total_us"] for x in pa["buckets"]}
        buckets_b = {x["bucket"]: x["total_us"] for x in pb["buckets"]}
        print("    -- buckets --")
        for bucket in sorted(set(buckets_a) | set(buckets_b),
                             key=lambda k: -max(buckets_a.get(k, 0.0),
                                                buckets_b.get(k, 0.0))):
            print(
                _delta_row(bucket, buckets_a.get(bucket, 0.0),
                           buckets_b.get(bucket, 0.0)))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _synthetic_events() -> list[Event]:
    """A hand-built decode-step timeline with known busy/idle and gaps.

    Names and `tf_op` strings are copied verbatim from a captured
    vllm-torchtpu TPU trace, so this doubles as a regression test on the
    bucket regexes: if a rule stops matching a real op name, this fails.
    """

    def ev(name: str,
           start_us: float,
           dur_us: float,
           tf_op: str = "",
           category: str = "") -> Event:
        start_ns = int(start_us * 1e3)
        dur_ns = dur_us * 1e3
        return Event(
            name=name,
            start_ns=start_ns,
            end_ns=int(start_ns + dur_ns),
            dur_ns=dur_ns,
            device_ns=dur_ns,
            line="XLA Ops",
            match_text=f"{name} | {tf_op} | {category}",
        )

    # Timeline in us: 0-10 busy, 10-30 GAP, 30-50 busy, 50-50.6 GAP,
    # 50.6-51 busy, 51-71 busy, 71-171 GAP, 171-181 busy.
    # => span 181, busy 60.4, idle 120.6, three gaps (20, 0.6, 100).
    # The 0.6 us gap is the sub-microsecond dispatch bubble the histogram's
    # bottom buckets exist for; the 100 us one is a host stall.
    return [
        ev(
            "RPAd-p_256-bq_1_1-bkv_16384_16384", 0, 10,
            "custom_kernel/custom_kernel/RPAd-p_256-bq_1_1-bkv_16384_16384/"
            "pallas_call:", "custom-call"),
        ev(
            "gmm_v2-g_64-m_65536-k_2048-act_silu-n_1024-tm_128-tk_2048-tn_512",
            30, 20,
            "custom_kernel/custom_kernel/gmm_v2-g_64-m_65536-k_2048-act_silu-"
            "n_1024-tm_128-tk_2048-tn_512/pallas_call:", "custom-call"),
        ev("all-reduce.260.cloned.1.call-done", 50.6, 0.4,
           "distributed.all_reduce/distributed.all_reduce:", "async-done"),
        ev("sc_ragged_gather_reduce.cloned.1.call-done", 51, 20,
           "custom_kernel/custom_kernel/sc_ragged_gather_reduce/mpmd_map:",
           "async-done"),
        ev(
            "ragged_causal_conv1d_kernel.60", 171, 5,
            "custom_kernel/custom_kernel/ragged_causal_conv1d_kernel/"
            "pallas_call:", "custom-call"),
        # No tf_op and no hlo_category: the metadata-free case, which must
        # still bucket from the name and must show up in the coverage number.
        ev("add_rsqrt_fusion.7", 176, 5),
    ]


def _bucket_expectations() -> tuple[tuple[str, str, str], ...]:
    """(op name, tf_op, expected bucket) for every real name pattern."""
    return (
        ("RPAd-p_256-bq_1_1-bkv_16384_16384", "", "attention"),
        ("RPAm-p_256-bq_512_256-bkv_2048_512", "", "attention"),
        ("RPA-HD_64-bq_512-bkvp_8-p_256", "", "attention"),
        ("MLA-RPA-bq_512-bkvp_8-p_64", "", "attention"),
        ("StreamIdxTC-d-bq_1-bkvp_8", "", "attention"),
        ("decoding_gdn-bt_2", "", "gdn/linear-attn"),
        ("fused_conv1d_gdn_decode", "", "gdn/linear-attn"),
        ("ragged_causal_conv1d_kernel.60", "", "gdn/linear-attn"),
        ("fusion.3", "custom_kernel/custom_kernel/"
         "ragged_gated_delta_rule_wrapper/cond/branch_1_fun/slice",
         "gdn/linear-attn"),
        ("gmm_v2-g_64-m_65536-k_512-act_None-n_2048-tm_128-tk_512-tn_2048", "",
         "moe-gmm"),
        ("sc_ragged_gather.cloned.1.call-done", "", "moe-permute/combine"),
        ("sc_ragged_gather_reduce", "", "moe-permute/combine"),
        ("sparse-core-data-format-call", "", "moe-permute/combine"),
        ("gather_custom_fusion.99", "", "moe-permute/combine"),
        ("gather_offload_async_done", "", "moe-permute/combine"),
        ("sort.4", "custom_kernel/custom_kernel/sort:", "moe-permute/combine"),
        # The one-hot permute/combine primitives, which XLA leaves anonymous.
        ("fusion.5", "custom_kernel/custom_kernel/dot_general:",
         "moe-permute/combine"),
        ("fusion.6", "custom_kernel/custom_kernel/reduce_sum:",
         "moe-permute/combine"),
        # The sampler's top-k must NOT be counted as MoE routing.
        ("sort.9", "topk/topk:", OTHER_BUCKET),
        ("all-reduce.260.cloned.1.call-done", "", "collectives"),
        ("all-gather.1", "", "collectives"),
        ("prepare_start_all-reduce.cloned.1.call-start", "", "collectives"),
        ("barrier-cores", "", "collectives"),
        ("quantized_matmul_kernel_256_512_1024", "", "dense-matmul"),
        ("fusion.9", "custom_kernel/custom_kernel/jit(_quantized_matmul_jax)/"
         "dot_general:", "dense-matmul"),
        ("convolution.12", "", "dense-matmul"),
        ("custom_kernel.63", "custom_kernel/custom_kernel/pallas_call:",
         "pallas-unnamed"),
        ("copy-done.44", "", "data-movement"),
        ("async-copy", "", "data-movement"),
        ("add_rsqrt_fusion.7", "", "rope/norm/elementwise"),
        ("multiply_add_fusion.2", "", "rope/norm/elementwise"),
        ("convert.1", "", "rope/norm/elementwise"),
        # Deliberately unclaimed: an anonymous XLA fusion with no metadata is
        # exactly the case the "other" listing exists to surface.
        ("fusion.11", "", OTHER_BUCKET),
        ("conditional.90", "", OTHER_BUCKET),
    )


def run_self_test() -> int:
    """Exercise aggregation, gaps and bucketing with no profiler data."""
    failures: list[str] = []

    def check(label: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    rules = _compile_rules()
    events = _synthetic_events()

    check("busy_ns", merged_busy_ns(events), 60_400.0)

    # Overlapping events must be unioned, not summed.
    overlapping = [
        Event("a", 0, 100, 100.0, 100.0, "L", "a | | "),
        Event("b", 50, 200, 150.0, 150.0, "L", "b | | "),
    ]
    check("busy_ns overlap", merged_busy_ns(overlapping), 200.0)
    check("gaps overlap", find_gaps(overlapping), [])

    gaps = find_gaps(events)
    check("num gaps", len(gaps), 3)
    check("gap durations", [g.dur_ns for g in gaps],
          [20_000.0, 600.0, 100_000.0])
    check("gap neighbours", (gaps[0].before, gaps[0].after),
          ("RPAd-p_256-bq_1_1-bkv_16384_16384",
           "gmm_v2-g_64-m_65536-k_2048-act_silu-n_1024-tm_128-tk_2048-tn_512"))

    histogram = gap_histogram(gaps)
    by_range = {(r["low_us"], r["high_us"]): r["count"] for r in histogram}
    check("hist 20-50us", by_range[(20.0, 50.0)], 1)
    check("hist 0.5-1us", by_range[(0.5, 1.0)], 1)
    check("hist >2000us", by_range[(2000.0, None)], 0)
    check("hist 100-200us", by_range[(100.0, 200.0)], 1)
    check("hist total", sum(r["count"] for r in histogram), len(gaps))

    summary = summarize_events(events, by="base", rules=rules, top=10)
    check("span_us", round(summary["span_us"], 3), 181.0)
    check("busy_us", round(summary["busy_us"], 3), 60.4)
    check("idle_us", round(summary["idle_us"], 3), 120.6)
    check("num_events", summary["num_events"], len(events))
    check("largest gap", round(summary["largest_gaps"][0]["gap_us"], 3), 100.0)
    check("gap_total_us", round(summary["gap_total_us"], 3), 120.6)
    check("op_total_us", round(summary["op_total_us"], 3), 60.4)
    total_bucket_us = sum(b["total_us"] for b in summary["buckets"])
    check("buckets sum to busy", round(total_bucket_us, 3), 60.4)

    # A nested event (a conditional next to its branch's ops) must inflate
    # `op_total_us` but leave busy, idle and the gap list untouched.
    nested = list(events) + [
        Event("conditional.90", 51_000, 71_000, 20_000.0, 20_000.0, "XLA Ops",
              "conditional.90 |  | conditional")
    ]
    nested_summary = summarize_events(nested, by="base", rules=rules, top=10)
    check("nested busy unchanged", round(nested_summary["busy_us"], 3), 60.4)
    check("nested idle unchanged", round(nested_summary["idle_us"], 3), 120.6)
    check("nested gaps unchanged", nested_summary["num_gaps"], 3)
    check("nested op sum inflated", round(nested_summary["op_total_us"], 3),
          80.4)

    # Numeric HLO ids collapse; kernel configs (dashes) survive.
    check("base name conditional", base_op_name("conditional.90"),
          "conditional")
    check("base name all-reduce",
          base_op_name("all-reduce.260.cloned.1.call-done"),
          "all-reduce.cloned.call-done")
    check("base name gmm", base_op_name("gmm_v2-g_64-m_65536-k_2048"),
          "gmm_v2-g_64-m_65536-k_2048")

    # Whole-HLO-text event names reduce to the instruction name.
    check(
        "short name from hlo text",
        short_op_name("%conditional.90 = (f32[513,8,128,128]) conditional("
                      "s32[] %convert_element_type.3619)"), "conditional.90")
    check("short name plain", short_op_name("sc_ragged_gather"),
          "sc_ragged_gather")

    for name, tf_op, want in _bucket_expectations():
        got = classify(f"{name} | {tf_op} | ", rules)
        if got != want:
            failures.append(f"bucket {name!r} (tf_op={tf_op!r}): "
                            f"got {got!r}, want {want!r}")

    # An extra rule must win over the built-ins.
    override = _compile_rules(extra=[("scratch", r"gmm_v2")])
    check("extra rule wins", classify("gmm_v2-g_64 | | ", override), "scratch")

    # Regrouping by tf_op must merge ops that share a jaxpr scope.
    by_tf = aggregate_ops(events, by="tf_op", rules=rules)
    check("tf_op grouping keys", len(by_tf), 6)

    # Empty input must not raise and must report zeros.
    empty = summarize_events([], by="base", rules=rules, top=5)
    check("empty span", empty["span_us"], 0.0)
    check("empty histogram rows", len(empty["gap_histogram"]),
          len(GAP_BUCKET_EDGES_US) + 1)

    # Line dispatch: the primary timeline must be picked out of a plane that
    # also carries the async, module, TraceMe, counter and unknown lines, and
    # only the unknown one may be reported as unrecognized.
    async_event = Event(
        "all-reduce", 40_000, 60_000, 20_000.0, 20_000.0, "Async XLA Ops",
        "all-reduce | distributed.all_reduce/distributed.all_reduce: | "
        "async-start")
    by_line = {
        "XLA Ops":
        list(events),
        "Async XLA Ops": [async_event],
        "XLA Modules": [
            Event("tt_jit_layernorm", 0, 181_000, 181_000.0, 181_000.0,
                  "XLA Modules", "tt_jit_layernorm |  | ")
        ],
        "XLA TraceMe": [
            Event("barrier-cores", 0, 1_000, 1_000.0, 1_000.0, "XLA TraceMe",
                  "barrier-cores |  | ")
        ],
        "counters_0": [],
        "Some Future Line": [
            Event("mystery", 0, 2_000, 2_000.0, 2_000.0, "Some Future Line",
                  "mystery |  | ")
        ],
    }
    counts = {name: len(evs) for name, evs in by_line.items()}
    plane = summarize_plane("/device:TPU:0",
                            by_line,
                            counts,
                            by="base",
                            rules=rules,
                            top=10)
    check("primary line", plane["primary_line"], "XLA Ops")
    check("primary recognized", plane["primary_line_recognized"], True)
    check("busy from primary only", round(plane["busy_us"], 3), 60.4)
    check("unrecognized lines", sorted(plane["unrecognized_lines"]),
          ["Some Future Line"])
    check("async kept separate", plane["async"]["num_events"], 1)
    check("async bucket", plane["async"]["buckets"][0]["bucket"],
          "collectives")
    check("module count", plane["modules"]["count"], 1)
    check("traceme bucket", plane["traceme"][0]["name"], "barrier-cores")
    # Every synthetic op carries a tf_op except the elementwise fusion.
    check("tf_op coverage", round(plane["tf_op_coverage"], 4), round(5 / 6, 4))
    check("category coverage", round(plane["category_coverage"], 4),
          round(5 / 6, 4))
    stripped = [e._replace(match_text=f"{e.name} |  | ") for e in events]
    check(
        "coverage with no metadata",
        summarize_events(stripped, by="base", rules=rules,
                         top=5)["tf_op_coverage"], 0.0)

    # With no recognized op line, fall back to the busiest line and say so.
    fallback = summarize_plane(
        "/device:TPU:0", {"Some Future Line": by_line["Some Future Line"]},
        {"Some Future Line": 1},
        by="base",
        rules=rules,
        top=10)
    check("fallback line", fallback["primary_line"], "Some Future Line")
    check("fallback flagged", fallback["primary_line_recognized"], False)

    # The JSON dump must round-trip, since --compare reads it back.
    fake = {
        "source":
        "synthetic",
        "profiles": [{
            "path": "synthetic",
            "phase": "decode_only",
            "rank": 0,
            "timestamp": "",
            "planes": [plane],
        }],
    }
    restored = json.loads(json.dumps(fake))
    check("json round trip",
          _plane_index(restored).keys(),
          _plane_index(fake).keys())

    # Render both reports. Cheap, but it is the only thing that catches a
    # typo in a format string or a key the printer expects and the summary
    # does not carry -- neither of which any assertion above would see.
    two_ranks = {
        "source":
        "synthetic",
        "profiles": [{
            "path": f"synthetic/rank{rank}",
            "phase": "decode_only",
            "rank": rank,
            "timestamp": "2026_08_15_00_00_00",
            "planes": [dict(plane, plane=f"/device:TPU:{rank}")],
        } for rank in (0, 1)],
    }
    rendered = io.StringIO()
    with contextlib.redirect_stdout(rendered):
        print_report(two_ranks, top=5)
        print_compare(fake, two_ranks)
    text = rendered.getvalue()
    for needle in ("IDLE/GAP", "category rollup", "10 largest gaps",
                   f"'{OTHER_BUCKET}' contents", "per-rank comparison",
                   "skew:", "-- buckets --"):
        if needle not in text:
            failures.append(f"report is missing {needle!r}")

    # Path classification must recognize both capture layouts.
    root = "/tmp/prof"
    phased = _classify_trace_path(
        root, f"{root}/decode_only/plugins/profile/2026_08_15_00_00_00/"
        "rank3_host.xplane.pb")
    check("phase from path", phased.phase, "decode_only")
    check("rank from filename", phased.rank, 3)
    check("timestamp from path", phased.timestamp, "2026_08_15_00_00_00")
    plain = _classify_trace_path(
        root, f"{root}/plugins/profile/2026_08_15_00_00_00/host.xplane.pb")
    check("phase absent", plain.phase, "-")
    check("rank default", plain.rank, 0)
    sandbox = _classify_trace_path(
        root, f"{root}/decode_only/rank_5/plugins/profile/ts/host.xplane.pb")
    check("rank from sandbox dir", sandbox.rank, 5)
    check("phase with sandbox dir", sandbox.phase, "decode_only")

    # Pointed at a single file, discovery still recovers rank and timestamp
    # from the merged layout; the phase is left unset.
    with tempfile.TemporaryDirectory() as tmp:
        one = os.path.join(tmp, "decode_only", "plugins", "profile",
                           "2026_08_15_00_00_00")
        os.makedirs(one)
        target = os.path.join(one, "rank7_host.xplane.pb")
        open(target, "wb").close()
        found = discover_traces(target)
        check("single file rank", found[0].rank, 7)
        check("single file ts", found[0].timestamp, "2026_08_15_00_00_00")
        check("single file phase", found[0].phase, "-")
        # A directory is grouped by phase and rank.
        every = discover_traces(tmp)
        check("dir discovery", [(t.phase, t.rank) for t in every],
              [("decode_only", 7)])
        check("dir default rank",
              select_traces(every, phase=None, rank=None, all_ranks=True),
              every)

    if failures:
        print(f"SELF-TEST FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("self-test OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_extra_buckets(values: Iterable[str]) -> list[tuple[str, str]]:
    extra = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--bucket expects NAME=REGEX, got {value!r}")
        name, pattern = value.split("=", 1)
        extra.append((name, pattern))
    return extra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="a *.xplane.pb file or a profile directory (a run root or a "
        "phased-profiler root with one subdirectory per phase)")
    parser.add_argument("--phase",
                        default=None,
                        help="only this phase (default: all phases found)")
    parser.add_argument("--rank",
                        type=int,
                        default=None,
                        help="rank to report (default: 0)")
    parser.add_argument("--all-ranks",
                        action="store_true",
                        help="report every rank and print the skew table")
    parser.add_argument("--top",
                        type=int,
                        default=25,
                        help="rows in the top-op table (default: 25)")
    parser.add_argument(
        "--by",
        choices=("base", "exact", "tf_op", "category"),
        default="base",
        help="group ops by base name (numeric HLO ids stripped), exact name, "
        "the `tf_op` jaxpr scope, or `hlo_category` (default: base)")
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="prepend a bucket rule; repeatable, first match wins")
    parser.add_argument("--include-sparsecore",
                        action="store_true",
                        help="also summarize the SparseCore planes")
    parser.add_argument("--json",
                        default=None,
                        metavar="FILE",
                        help="write the whole summary as JSON ('-' = stdout)")
    parser.add_argument("--quiet",
                        action="store_true",
                        help="suppress the text report (use with --json)")
    parser.add_argument("--compare",
                        nargs=2,
                        metavar=("A", "B"),
                        help="side-by-side delta of two summaries; each is "
                        "either a --json dump or a profile path")
    parser.add_argument("--self-test",
                        action="store_true",
                        help="run the offline logic tests and exit")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list every trace found (ignores --phase/--rank) and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()

    rules = _compile_rules(_parse_extra_buckets(args.bucket))

    if args.compare:
        a = load_summary(args.compare[0],
                         by=args.by,
                         rules=rules,
                         top=args.top,
                         include_sparsecore=args.include_sparsecore)
        b = load_summary(args.compare[1],
                         by=args.by,
                         rules=rules,
                         top=args.top,
                         include_sparsecore=args.include_sparsecore)
        print_compare(a, b)
        return 0

    if not args.path:
        build_parser().error("path is required unless --compare/--self-test")

    traces = discover_traces(args.path)
    if args.list:
        for trace in traces:
            print(f"phase={trace.phase:<16} rank={trace.rank:<3} "
                  f"ts={trace.timestamp or '-':<20} {trace.path}")
        return 0

    picked = select_traces(traces,
                           phase=args.phase,
                           rank=args.rank,
                           all_ranks=args.all_ranks)
    summary = {
        "source":
        args.path,
        "profiles": [
            summarize_trace(t,
                            by=args.by,
                            rules=rules,
                            top=args.top,
                            include_sparsecore=args.include_sparsecore)
            for t in picked
        ],
    }
    if not args.quiet:
        print_report(summary, top=args.top)
    if args.json:
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.json == "-":
            print(text)
        else:
            with open(args.json, "w") as handle:
                handle.write(text + "\n")
            print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
