#!/usr/bin/env python3
"""Run the fixed FP8, capacity-32 MLA matrix on all four TPU hosts."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT"])
BENCH = ROOT / "benchmark"
BENCH.mkdir(parents=True, exist_ok=True)
C = (1, 2, 4, 8, 16, 32)
VERIFY_BUCKETS = (8, 16, 32, 48, 96, 192)
DECODE_TILES = ((1, 3, 4), (1, 4, 4), (1, 8, 4), (1, 8, 2), (1, 16, 1),
                (1, 16, 2), (1, 16, 4), (1, 16, 8))
VERIFY_TILES = ((16, 1, 1), (6, 16, 3), (12, 16, 3), (16, 16, 1), (16, 2, 2),
                (8, 4, 1), (16, 4, 1), (32, 2, 1), (32, 4, 2))
rows = []
for window in (1, 6):
    for concurrency, verify_bucket in zip(C, VERIFY_BUCKETS):
        tokens = concurrency if window == 1 else verify_bucket
        tiles = DECODE_TILES if window == 1 else VERIFY_TILES
        for tile in tiles:
            label = f"w{window}-c{concurrency}-tile" + "-".join(map(str, tile))
            path = BENCH / (label + ".json")
            env = dict(os.environ,
                       TUNER_TRACE_DIR=str(ROOT / "profiling/xprof" / label))
            print("TP32_CANDIDATE_START", label, flush=True)
            subprocess.run([
                sys.executable,
                str(Path(__file__).with_name("k3_fp8_tp32_tuner.py")),
                "--case",
                json.dumps([concurrency, window, tokens]),
                "--tile",
                json.dumps(tile),
                "--output",
                str(path),
            ],
                           env=env,
                           check=True,
                           timeout=1800)
            records = json.loads(path.read_text())
            assert len(records) == 4
            for row in records:
                assert row["status"] == "success" and row["cache_valid"]
                row["capture_label"] = label
                rows.append(row)
                with (BENCH / "metrics.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
            print("TP32_CANDIDATE_VALIDATED", label, flush=True)
summary = dict(status="correctness_complete_device_analysis_required",
               devices_total=32,
               metadata_capacity=32,
               rows=len(rows),
               decode_buckets=list(C),
               verify_buckets=list(VERIFY_BUCKETS),
               draft_model="Inferact/Kimi-K3-DSpark-Block5")
(BENCH / "summary.json").write_text(json.dumps(summary, indent=2))
print("TP32_SWEEP_COMPLETE", json.dumps(summary), flush=True)
