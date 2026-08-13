#!/usr/bin/env python3
"""Break down TPU work between consecutive K3 backbone executions.

TorchTPU's Chrome trace writer can emit unescaped quotes in HLO event names,
so loading the complete file with ``json.load`` is not reliable.  This tool
parses one top-level trace-event object at a time and intentionally uses only
TPU device timestamps.  Host and TPU clocks in these captures are not aligned
closely enough to overlay CPU events on a device interval at microsecond
resolution.
"""

from __future__ import annotations

import argparse
import gzip
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    pid: int
    tid: int
    ts: float
    dur: float
    name: str


MODULE_PATTERNS = (
    ("lm_head", "tt_jit_compile_mlir_as_strided"),
    ("fused_lm_sampler", "sample_from_hidden_states_argmax"),
    ("uniform_vocab", "tt_jit_uniform_(18208027922723981442)"),
    ("uniform_seed", "tt_jit_uniform_(6615789334249222255)"),
    ("sample_argmax", "sample_from_logits_argmax"),
    ("token_view", "tt_jit_as_strided(12167799074198170175)"),
    (
        "placeholder_scatter",
        "_substitute_placeholder_token_scatter",
    ),
)


def _field(text: str, key: str, convert=str):
    match = re.search(rf'^    "{re.escape(key)}": (.*?)(?:,)?$', text, re.M)
    if match is None:
        return None
    value = match.group(1).rstrip(",")
    if convert is str:
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        return value
    try:
        return convert(value)
    except ValueError:
        return None


def read_device_events(path: Path) -> list[Event]:
    events: list[Event] = []
    with gzip.open(path, "rt", errors="replace") as trace:
        current: list[str] | None = None
        for line in trace:
            if line == "  {\n":
                current = []
            if current is not None:
                current.append(line)
            if current is None or line not in ("  },\n", "  }\n"):
                continue
            text = "".join(current)
            current = None
            if _field(text, "ph") != "X":
                continue
            pid = _field(text, "pid", int)
            tid = _field(text, "tid", int)
            ts = _field(text, "ts", float)
            dur = _field(text, "dur", float)
            name = _field(text, "name")
            if pid not in (2, 3) or None in (tid, ts, dur, name):
                continue
            events.append(Event(pid, tid, ts, dur, name))
    return events


def union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--region", default="region.268435455")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=1,
        help="skip profiler-boundary backbone regions before forming gaps",
    )
    parser.add_argument(
        "--min-backbone-duration-us",
        type=float,
        default=1000.0,
        help=(
            "ignore same-named short compiled regions; K3's backbone is about "
            "15 ms, while sampling epilogues may reuse the root region name"
        ),
    )
    parser.add_argument(
        "--top-modules",
        type=int,
        default=0,
        help="also print the N device-module names with the most gap time",
    )
    args = parser.parse_args()

    events = read_device_events(args.trace)
    regions = sorted(
        (
            event
            for event in events
            if event.name == args.region
            and event.dur >= args.min_backbone_duration_us
        ),
        key=lambda event: event.ts,
    )[args.skip_first :]
    if len(regions) < 2:
        raise SystemExit("fewer than two usable backbone regions")

    results: dict[str, list[float]] = defaultdict(list)
    module_totals: dict[str, float] = defaultdict(float)
    for current, following in zip(regions, regions[1:]):
        start = current.ts + current.dur
        stop = following.ts
        if stop <= start:
            continue
        intersecting = [
            event
            for event in events
            if event.tid == 2
            and event.ts < stop
            and event.ts + event.dur > start
        ]
        busy = union_duration(
            [
                (max(event.ts, start), min(event.ts + event.dur, stop))
                for event in intersecting
            ]
        )
        results["gap"].append(stop - start)
        results["backbone"].append(current.dur)
        results["start_spacing"].append(following.ts - current.ts)
        results["device_module_union"].append(busy)
        results["uncovered"].append(stop - start - busy)
        for name in {event.name for event in intersecting}:
            module_totals[name] += union_duration(
                [
                    (max(event.ts, start), min(event.ts + event.dur, stop))
                    for event in intersecting
                    if event.name == name
                ]
            )
        for label, pattern in MODULE_PATTERNS:
            matching = [
                (max(event.ts, start), min(event.ts + event.dur, stop))
                for event in intersecting
                if pattern in event.name
            ]
            results[label].append(union_duration(matching))

    print(f"trace={args.trace}")
    print(f"gaps={len(results['gap'])} units=microseconds")
    for key in (
        "backbone",
        "gap",
        "start_spacing",
        "device_module_union",
        "uncovered",
        *(label for label, _ in MODULE_PATTERNS),
    ):
        values = results[key]
        print(
            f"{key:21s} mean={statistics.mean(values):9.5f} "
            f"median={statistics.median(values):9.5f}"
        )
    if args.top_modules:
        print("top_device_modules mean_per_gap=microseconds")
        gap_count = len(results["gap"])
        for name, total in sorted(
            module_totals.items(), key=lambda item: item[1], reverse=True
        )[: args.top_modules]:
            print(f"{total / gap_count:9.5f} {name}")


if __name__ == "__main__":
    main()
