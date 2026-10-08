#!/usr/bin/env python3
"""Render a CPU job that assembles the completed DSv4.1 static-MoE model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.parse import urlparse

SOURCE_BRANCH = "develop/dsv41-falcon-prefill"
SOURCE_PATH = "/workspace/vllm-torchtpu"
MODEL_PATH = "/models/DeepSeek-V4.1-Flash"
PREVIOUS_PATH = "/previous"
CONVERTED_PATH = "/converted"


def render(conversion: dict, source_commit: str) -> dict:
    if conversion.get("status") != "succeeded":
        raise ValueError("Conversion experiment must have succeeded")
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("--source-commit must be a full Git SHA")
    uri = urlparse(conversion["artifact_uri"])
    if uri.scheme != "gs" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("Conversion artifact must have a GCS URI")
    spec = copy.deepcopy(conversion["role_to_task_spec"])
    if list(spec) != ["worker"]:
        raise ValueError("Unexpected conversion task topology")
    task = spec["worker"]
    if task["device_type"] != "cpu" or task["replica"] != 1:
        raise ValueError("Expected one CPU packing worker")
    mounts = {item["mount_path"]: item for item in task["mounts"]}
    if MODEL_PATH not in mounts or PREVIOUS_PATH not in mounts:
        raise ValueError("Conversion model or previous-cache mount is missing")
    task["mounts"].append(
        {
            "name": "completed-conversion",
            "type": "gcs",
            "bucket": uri.netloc,
            "prefix": uri.path.strip("/"),
            "mount_path": CONVERTED_PATH,
            "read_only": True,
            "mount_options": ["implicit-dirs", "read:enable-buffered-read:true"],
        }
    )
    source = next(item for item in task["sources"] if item["name"] == "vllm-torchtpu")
    source["ref"] = SOURCE_BRANCH
    source["commit"] = source_commit
    if source["path"] != SOURCE_PATH:
        raise ValueError("Unexpected source checkout path")
    task["command"] = (
        "bash <<'PACK_STATIC_MOE'\n"
        "set -euo pipefail\n"
        'mkdir -p "$ARTIFACT_LOCAL_DIR/model"\n'
        "timeout --signal=TERM --kill-after=60s 21600 python -u "
        f"{SOURCE_PATH}/scripts/weights/pack_dsv41_static_moe.py "
        f"--source {MODEL_PATH} "
        f"--prepared {CONVERTED_PATH}/prepared-weights "
        f"--previous {PREVIOUS_PATH}/prepared-weights "
        '--output "$ARTIFACT_LOCAL_DIR/model"\n'
        'test -f "$ARTIFACT_LOCAL_DIR/model/STATIC_MOE_READY.json"\n'
        "PACK_STATIC_MOE\n"
    )
    return {
        "name": f"dsv41-static-moe-pack-{source_commit[:10]}",
        "exp_type": conversion["exp_type"],
        "artifact_type": "GCS",
        "cluster_name": conversion["cluster_name"],
        "config": json.dumps(
            {
                "conversion_exp_id": conversion["exp_id"],
                "conversion_artifact_uri": conversion["artifact_uri"],
                "source_commit": source_commit,
                "model_output": "model/",
            }
        ),
        "role_to_task_spec": spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-json", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conversion = json.loads(args.conversion_json.read_text())
    result = render(conversion, args.source_commit)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Rendered {result['name']} from {conversion['exp_id']}")


if __name__ == "__main__":
    main()
