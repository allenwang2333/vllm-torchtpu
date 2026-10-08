#!/usr/bin/env python3
"""Render a static-MoE replay from the successful PR #1734 Falcon baseline.

Input is the JSON from `falcon exp get exp-5iekivkfrn --output json`.
The rendered JSON is accepted by Falcon's YAML experiment submit command.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.parse import urlparse

BASELINE_COMMIT = "63e3476be4e76d7c89d38e1e7e47f64fb41d30be"
BASELINE_EXPERIMENT = "exp-5iekivkfrn"
SOURCE_BRANCH = "develop/dsv41-falcon-prefill"
MODEL_MOUNT = "/models/DeepSeek-V4.1-Flash"


def render(baseline: dict, source_commit: str, model_uri: str) -> dict:
    if baseline.get("exp_id") != BASELINE_EXPERIMENT:
        raise ValueError("Expected the successful PR #1734 Run:AI baseline")
    parsed = urlparse(model_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("--model-uri must name one GCS model directory")
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("--source-commit must be a full Git SHA")
    spec = copy.deepcopy(baseline["role_to_task_spec"])
    if list(spec) != ["worker"]:
        raise ValueError("Unexpected baseline task topology")
    task = spec["worker"]
    if (task["replica"], task["device_type"], task["device_count"]) != (
        1,
        "v7x",
        8,
    ):
        raise ValueError("Baseline is not the C1/TP8 worker")
    source = next(item for item in task["sources"] if item["name"] == "vllm-torchtpu")
    if source["commit"] != BASELINE_COMMIT:
        raise ValueError("Baseline source commit differs")
    source["ref"] = SOURCE_BRANCH
    source["commit"] = source_commit
    model_mount = next(item for item in task["mounts"] if item["name"] == "dsv41-model")
    model_mount["bucket"] = parsed.netloc
    model_mount["prefix"] = parsed.path.strip("/")
    if model_mount["mount_path"] != MODEL_MOUNT:
        raise ValueError("Unexpected baseline model mount")

    command = task["command"]
    old_uri = "gs://inference-model-storage-poc-tpu-hns/DeepSeek-V4.1-Flash/"
    if command.count(BASELINE_COMMIT) < 1 or old_uri not in command:
        raise ValueError("Baseline command no longer has the pinned source/model")
    command = command.replace(BASELINE_COMMIT, source_commit)
    cache_source = BASELINE_COMMIT[:9]
    if command.count(cache_source + "-torchtpu") != 1:
        raise ValueError("Baseline compilation cache key changed")
    command = command.replace(
        cache_source + "-torchtpu", source_commit[:9] + "-torchtpu"
    )
    command = command.replace(old_uri, model_uri.rstrip("/") + "/")
    index_assertions = (
        "assert len(shards) == 48, len(shards)",
        "assert len(shards)==48,len(shards)",
    )
    present = [item for item in index_assertions if command.count(item) == 1]
    if len(present) != 1:
        raise ValueError("Baseline model index check changed")
    command = command.replace(present[0], "assert shards, 'empty model index'")
    marker = (
        "export DSV41_CACHE_DIR=/cache/dsv41-prefill "
        "DEEPSEEK_V41_MODEL=/models/DeepSeek-V4.1-Flash"
    )
    if command.count(marker) != 1:
        raise ValueError("Baseline launch marker changed")
    command = command.replace(
        marker,
        f"export TPU_DSV41_STATIC_MOE_DIR={MODEL_MOUNT}\n{marker}",
    )
    task["command"] = command
    return {
        "name": f"dsv41-100k-prefill-c1-static-moe-{source_commit[:10]}",
        "exp_type": baseline["exp_type"],
        "artifact_type": "GCS",
        "cluster_name": baseline["cluster_name"],
        "priority": baseline.get("priority", 0),
        "config": json.dumps(
            {
                "baseline_exp_id": BASELINE_EXPERIMENT,
                "source_commit": source_commit,
                "model_uri": model_uri,
                "input_len": 100000,
                "output_len": 1,
                "concurrency": 1,
                "compile_sizes": [2048, 8192],
                "static_moe": True,
                "dense_online_requant": True,
            }
        ),
        "role_to_task_spec": spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_json.read_text())
    result = render(baseline, args.source_commit, args.model_uri)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Rendered {result['name']} from {BASELINE_EXPERIMENT}")


if __name__ == "__main__":
    main()
