#!/usr/bin/env python3
"""Attribute stable K3 model-executable leaf time by HLO opcode.

TorchTPU HLO names can contain unescaped quotes, so this reuses the tolerant
line-oriented trace reader in ``analyze_trace_tail.py``.  Classification uses
the opcode after the HLO assignment, never substrings from operand names.
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_trace_tail import Event, read_device_events, union_duration


_OPCODE = re.compile(r"\s([a-z][a-z0-9-]*)\(")


def opcode(name: str) -> str:
    """Return the assigned HLO opcode without inspecting its operands."""
    rhs = name.split(" = ", 1)[1] if " = " in name else name
    match = _OPCODE.search(rhs)
    return match.group(1) if match is not None else "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--region", default="region.268435455")
    parser.add_argument("--min-module-duration-us", type=float, default=1000.0)
    parser.add_argument(
        "--min-leaf-fraction",
        type=float,
        default=0.9,
        help="discard profiler-boundary modules without nearly complete leaves",
    )
    parser.add_argument("--show-counts", action="store_true")
    args = parser.parse_args()

    events = read_device_events(args.trace)
    modules = sorted(
        (event for event in events if event.tid == 2
         and event.name == args.region
         and event.dur >= args.min_module_duration_us),
        key=lambda event: event.ts,
    )
    leaves = [event for event in events if event.tid == 3]
    samples: list[tuple[Event, dict[str, float], dict[str, int]]] = []
    for module in modules:
        stop = module.ts + module.dur
        contained = sorted(
            (event for event in leaves if event.ts >= module.ts
             and event.ts + event.dur <= stop),
            key=lambda event: (event.ts, -event.dur),
        )
        # The XLA-op lane contains nested bookkeeping scopes.  Attribute each
        # parent only its self time after subtracting the union of its direct
        # children; dropping the parent entirely would also drop legitimate
        # conditional/fusion time around those children.
        children: dict[int, list[tuple[float, float]]] = defaultdict(list)
        stack: list[int] = []
        for index, event in enumerate(contained):
            while stack and (contained[stack[-1]].ts
                             + contained[stack[-1]].dur <= event.ts):
                stack.pop()
            event_stop = event.ts + event.dur
            while stack and event_stop > (contained[stack[-1]].ts
                                           + contained[stack[-1]].dur):
                stack.pop()
            if stack:
                parent = contained[stack[-1]]
                if event.ts != parent.ts or event_stop != parent.ts + parent.dur:
                    children[stack[-1]].append((event.ts, event_stop))
            stack.append(index)
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for index, event in enumerate(contained):
            counts[opcode(event.name)] += 1
            self_time = event.dur - union_duration(children[index])
            if self_time > 0:
                totals[opcode(event.name)] += self_time
        # Coverage selection uses all contained records: a complete module can
        # legitimately have less than 90% exclusive leaf occupancy after the
        # nested bookkeeping scopes above are removed.
        if sum(event.dur for event in contained) >= (
                module.dur * args.min_leaf_fraction):
            samples.append((module, totals, counts))

    if not samples:
        raise SystemExit("no model modules met the leaf-coverage threshold")
    opcodes = sorted({key for _, totals, _ in samples for key in totals})
    means = {
        key: statistics.mean(totals.get(key, 0.0)
                             for _, totals, _ in samples)
        for key in opcodes
    }
    medians = {
        key: statistics.median(totals.get(key, 0.0)
                               for _, totals, _ in samples)
        for key in opcodes
    }
    module_durations = [module.dur for module, _, _ in samples]
    leaf_sums = [sum(totals.values()) for _, totals, _ in samples]
    print(f"trace={args.trace}")
    print(f"modules={len(samples)} units=microseconds")
    print(f"module_mean={statistics.mean(module_durations):.6f}")
    print(f"module_median={statistics.median(module_durations):.6f}")
    print(f"leaf_sum_mean={statistics.mean(leaf_sums):.6f}")
    print(f"leaf_sum_median={statistics.median(leaf_sums):.6f}")
    print("opcode_mean_per_module")
    for key, duration in sorted(means.items(), key=lambda item: item[1],
                                reverse=True):
        print(f"{duration:12.6f} {key}")
    print("opcode_median_per_module")
    for key, duration in sorted(medians.items(), key=lambda item: item[1],
                                reverse=True):
        print(f"{duration:12.6f} {key}")
    if args.show_counts:
        print("opcode_mean_event_count_per_module")
        for key in sorted(opcodes):
            count = statistics.mean(counts.get(key, 0)
                                    for _, _, counts in samples)
            print(f"{count:12.6f} {key}")


if __name__ == "__main__":
    main()
