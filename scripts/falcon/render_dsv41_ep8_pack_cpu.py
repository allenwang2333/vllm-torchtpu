#!/usr/bin/env python3
"""Render a CPU experiment to repack a complete v1 DSv4.1 model for EP8."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.parse import urlparse

SOURCE_BRANCH = "develop/dsv41-falcon-prefill"
MODEL_MOUNT = "/models/DeepSeek-V4.1-Flash"


def render(previous_pack: dict, source_commit: str, model_uri: str) -> dict:
    if previous_pack.get("exp_id") != "exp-8i5xjrno96":
        raise ValueError("Expected the completed v1 DSv4.1 model pack experiment")
    if previous_pack.get("status") != "succeeded":
        raise ValueError("Source model experiment did not succeed")
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("--source-commit must be a full Git SHA")
    uri = urlparse(model_uri)
    if uri.scheme != "gs" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("--model-uri must name the complete v1 model directory")
    spec = copy.deepcopy(previous_pack["role_to_task_spec"])
    task = spec["worker"]
    if task["device_type"] != "cpu" or task["replica"] != 1:
        raise ValueError("Expected one CPU packing worker")
    task["resources"] = {
        "cpu_request": "12",
        "cpu_limit": "16",
        "memory_request": "64Gi",
        "memory_limit": "64Gi",
    }
    task["mounts"] = [
        {
            "name": "v1-complete-model",
            "type": "gcs",
            "bucket": uri.netloc,
            "prefix": uri.path.strip("/"),
            "mount_path": MODEL_MOUNT,
            "read_only": True,
            "mount_options": ["implicit-dirs", "read:enable-buffered-read:true"],
        }
    ]
    source = next(item for item in task["sources"] if item["name"] == "vllm-torchtpu")
    source["ref"] = SOURCE_BRANCH
    source["commit"] = source_commit
    task["command"] = (
        "bash <<'PACK_EP8'\n"
        "set -euo pipefail\n"
        "python -m pip install --no-cache-dir numpy==2.3.3 safetensors==0.6.2\n"
        'mkdir -p "$ARTIFACT_LOCAL_DIR/model"\n'
        "timeout --signal=TERM --kill-after=60s 21600 python -u "
        "/workspace/vllm-torchtpu/scripts/weights/repack_dsv41_ep8.py "
        f"--source {MODEL_MOUNT} "
        '--output "$ARTIFACT_LOCAL_DIR/model" --workers 2\n'
        'test -f "$ARTIFACT_LOCAL_DIR/model/STATIC_MOE_READY.json"\n'
        "PACK_EP8\n"
    )
    return {
        "name": f"dsv41-static-moe-ep8-pack-{source_commit[:10]}",
        "exp_type": previous_pack["exp_type"],
        "artifact_type": "GCS",
        "cluster_name": previous_pack["cluster_name"],
        "config": json.dumps(
            {
                "source_model_uri": model_uri,
                "source_commit": source_commit,
                "model_output": "model/",
                "static_moe_format": "dsv41.static-moe.ep8.v2",
            }
        ),
        "role_to_task_spec": spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-pack-json", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.previous_pack_json.read_text())
    result = render(previous, args.source_commit, args.model_uri)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Rendered {result['name']}")


if __name__ == "__main__":
    main()
