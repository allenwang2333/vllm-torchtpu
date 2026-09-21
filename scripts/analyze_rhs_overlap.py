# SPDX-License-Identifier: Apache-2.0
"""Extract per-step device-time slopes from RHS overlap XPlane events."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(root):
    events = json.loads((root / "trace_events.json").read_text())
    configurations = json.loads((root / "configurations.json").read_text())
    rows = []
    for config in configurations:
        if config["steps"] != 128:
            continue
        durations = {}
        for steps in (128, 512):
            label = f"mb_{config['mode']}_m{config['m']}_r{config['reuse']}_s{steps}"
            group = {}
            for event in events[label]:
                if event["line"] == "XLA Ops" and "custom-call" in event["event"]:
                    group.setdefault(event["device"], []).append(event["duration_us"])
            if len(group) != 8 or any(len(values) != 7 for values in group.values()):
                raise ValueError(f"Expected 8 ranks x 7 captures: {label}")
            durations[steps] = {
                device: float(np.median(values)) for device, values in group.items()
            }
        slopes = {
            device: (durations[512][device] - durations[128][device]) / 384
            for device in durations[128]
        }
        if min(slopes.values()) <= 0:
            raise ValueError(f"Nonpositive device-time slope: {config}")
        row = {key: config[key] for key in ("mode", "m", "reuse")}
        row.update(
            run=root.name,
            us_per_rhs=float(np.median(list(slopes.values()))),
            rank_min_us=min(slopes.values()),
            rank_max_us=max(slopes.values()),
            per_rank_us=slopes,
            device_duration_us=durations,
        )
        rows.append(row)
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = summarize(args.root / "full") + summarize(args.root / "refine")
    lookup = {(r["mode"], r["m"], r["reuse"]): r for r in rows}
    dma = {r["run"]: r["us_per_rhs"] for r in rows if r["mode"] == "dma"}
    table = []
    for _, m, reuse in sorted(key for key in lookup if key[0] == "compute"):
        c, s, o = (lookup[mode, m, reuse] for mode in ("compute", "serial", "overlap"))
        table.append(
            dict(
                m=m,
                reuse=reuse,
                effective_tokens=m * reuse,
                dma_us=dma[c["run"]],
                compute_us=c["us_per_rhs"],
                serial_us=s["us_per_rhs"],
                overlap_us=o["us_per_rhs"],
                overlap_over_compute=o["us_per_rhs"] / c["us_per_rhs"],
            )
        )
    (args.root / "comparison.json").write_text(json.dumps(table, indent=2))
    with (args.root / "comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("M reuse DMA(us) compute(us) serial(us) overlap(us) overlap/compute")
    for row in sorted(table, key=lambda r: (r["reuse"] != 1, r["m"], r["reuse"])):
        print(
            row["m"],
            row["reuse"],
            *[
                round(row[k], 4)
                for k in (
                    "dma_us",
                    "compute_us",
                    "serial_us",
                    "overlap_us",
                    "overlap_over_compute",
                )
            ],
        )


if __name__ == "__main__":
    main()
