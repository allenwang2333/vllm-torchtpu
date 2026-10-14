# SPDX-License-Identifier: Apache-2.0
"""Summarize XLA device events; async spans are NOT pure communication time."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    processes, threads = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        if event["name"] == "process_name":
            processes[event["pid"]] = event["args"]["name"]
        elif event["name"] == "thread_name":
            threads[event["pid"], event["tid"]] = event["args"]["name"]
    groups = collections.defaultdict(list)
    for event in events:
        pid, tid = event.get("pid"), event.get("tid")
        lane = threads.get((pid, tid), "")
        if (
            event.get("ph") == "X"
            and "TPU:" in processes.get(pid, "")
            and lane in ("XLA Modules", "XLA Ops", "Async XLA Ops")
        ):
            groups[processes[pid], lane, event["name"]].append(event["dur"])
    return dict(
        source=str(path),
        units="microseconds; profiler only, not official steady-state timings",
        caveat=(
            "Nested and concurrent events must not be summed as wall time. "
            "Async event spans include waiting/overlap, not pure communication. "
            "The fused custom call contains both compute and remote DMA."
        ),
        events=[
            dict(
                device=device,
                lane=lane,
                name=name,
                count=len(values),
                mean_us=sum(values) / len(values),
                min_us=min(values),
                max_us=max(values),
            )
            for (device, lane, name), values in sorted(groups.items())
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summaries = [summarize(p) for p in sorted(args.results.rglob("*.trace.json.gz"))]
    (args.results / "trace-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(f"Summarized {len(summaries)} traces")


if __name__ == "__main__":
    main()
