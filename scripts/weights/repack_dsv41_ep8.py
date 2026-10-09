#!/usr/bin/env python3
"""Repack a complete v1 DSv4.1 model into directly readable EP8 shards.

The input checkpoint is read-only. The output includes every ordinary model
file and one contiguous, already requantized MoE safetensors file per layer
and EP rank. No MoE conversion or model-wide copy is needed at serving time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

LAYERS = 40
EP_SIZE = 8
EXPERTS_PER_RANK = 48
QUANTIZATION = {
    "dtype": "float4_e2m1fn_x2",
    "block": 512,
    "layout": "K_N",
    "packing": "low_nibble_first_along_N",
}
SHAPES = {
    "w13": (EXPERTS_PER_RANK, 5120, 2560),
    "w2": (EXPERTS_PER_RANK, 2560, 2560),
    "s13": (EXPERTS_PER_RANK, 10, 1, 5120),
    "s2": (EXPERTS_PER_RANK, 5, 1, 5120),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_index(source: Path) -> dict[str, dict]:
    ready = json.loads((source / "STATIC_MOE_READY.json").read_text())
    manifest_path = source / "tpu-static-moe.json"
    if ready.get("schema") != "dsv41.static-moe.v1":
        raise ValueError("Expected a complete v1 static MoE model")
    if sha256(manifest_path) != ready["static_manifest_sha256"]:
        raise ValueError("Source static MoE manifest checksum mismatch")
    for filename, field in (
        ("config.json", "source_config_sha256"),
        ("model.safetensors.index.json", "model_index_sha256"),
    ):
        if sha256(source / filename) != ready[field]:
            raise ValueError(f"Source {filename} checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["quantization"] != QUANTIZATION:
        raise ValueError("Source static MoE quantization format mismatch")
    files = {item["file"] for item in manifest["files"]}
    records = {}
    for entry in manifest["tensors"]:
        name = "prepared-weights/" + entry["file"]
        if name not in files or entry["key"] in records:
            raise ValueError(f"Invalid source tensor: {entry['key']}")
        records[entry["key"]] = {**entry, "file": name}
    if len(records) != LAYERS * EP_SIZE * EXPERTS_PER_RANK * 3 * 2:
        raise ValueError("Incomplete source MoE tensor set")
    return records


def read_expert(
    source: Path, records: dict[str, dict], layer: int, expert: int
) -> dict[str, np.ndarray]:
    keys = [
        f"layers.{layer}.ffn.experts.{expert}.w{projection}.weight::{kind}"
        for projection in (1, 2, 3)
        for kind in ("fp4", "scale")
    ]
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(records[key]["file"], []).append(key)
    result = {}
    for filename, members in by_file.items():
        with safe_open(str(source / filename), framework="numpy") as stream:
            for key in members:
                entry = records[key]
                tensor = np.array(stream.get_tensor(key), copy=True)
                dtype = np.uint8 if entry["dtype"] == "U8" else np.float32
                if list(tensor.shape) != entry["shape"] or tensor.dtype != dtype:
                    raise ValueError(f"Source tensor shape/dtype mismatch: {key}")
                result[key.rsplit(".", 2)[-2] + "::" + key.rsplit("::", 1)[-1]] = tensor
    return result


def store_expert(
    tensors: dict[str, np.ndarray], parts: dict[str, np.ndarray], slot: int
) -> None:
    w1, w2, w3 = (parts[f"w{i}::fp4"] for i in (1, 2, 3))
    s1, s2, s3 = (parts[f"w{i}::scale"] for i in (1, 2, 3))
    if (
        w1.shape != (5120, 1280)
        or w3.shape != w1.shape
        or w2.shape != (2560, 2560)
        or s1.shape != (10, 1, 2560)
        or s3.shape != s1.shape
        or s2.shape != (5, 1, 5120)
    ):
        raise ValueError("Unexpected MoE projection shape")
    tensors["w13"][slot, :, :1280] = w1
    tensors["w13"][slot, :, 1280:] = w3
    tensors["w2"][slot] = w2
    tensors["s13"][slot, :, :, :2560] = s1
    tensors["s13"][slot, :, :, 2560:] = s3
    tensors["s2"][slot] = s2


def pack_one(
    source: Path, output: Path, records: dict[str, dict], layer: int, rank: int
) -> dict:
    tensors = {
        name: np.empty(shape, dtype=np.uint8 if name.startswith("w") else np.float32)
        for name, shape in SHAPES.items()
    }
    first = rank * EXPERTS_PER_RANK
    for slot, expert in enumerate(range(first, first + EXPERTS_PER_RANK)):
        parts = read_expert(source, records, layer, expert)
        try:
            store_expert(tensors, parts, slot)
        except ValueError as error:
            raise ValueError(f"Unexpected MoE projection: {layer}/{expert}") from error
    relative = f"static-moe/layer-{layer:02d}-rank-{rank}.safetensors"
    target = output / relative
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite packed shard: {target}")
    save_file(tensors, str(target))
    result = {
        "file": relative,
        "layer": layer,
        "rank": rank,
        "expert_ids": list(range(first, first + EXPERTS_PER_RANK)),
        "bytes": target.stat().st_size,
    }
    print(json.dumps({"event": "ep8_shard_complete", **result}), flush=True)
    return result


def copy_model_files(source: Path, output: Path) -> None:
    for item in source.rglob("*"):
        if not item.is_file() or "prepared-weights" in item.relative_to(source).parts:
            continue
        relative = item.relative_to(source)
        if relative.as_posix() in {"STATIC_MOE_READY.json", "tpu-static-moe.json"}:
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite model file: {target}")
        shutil.copyfile(item, target)


def repack(source: Path, output: Path, workers: int) -> None:
    if workers < 1:
        raise ValueError("--workers must be positive")
    records = source_index(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "static-moe").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(
            pool.map(
                lambda pair: pack_one(source, output, records, *pair),
                ((layer, rank) for layer in range(LAYERS) for rank in range(EP_SIZE)),
            )
        )
    copy_model_files(source, output)
    for filename in ("config.json", "model.safetensors.index.json"):
        if sha256(source / filename) != sha256(output / filename):
            raise ValueError(f"Copied {filename} checksum mismatch")
    manifest = {
        "schema": "dsv41.static-moe.ep8.v2",
        "quantization": QUANTIZATION,
        "source_expert_count": EP_SIZE * EXPERTS_PER_RANK,
        "ep_size": EP_SIZE,
        "layers": LAYERS,
        "files": files,
    }
    write_json(output / "tpu-static-moe.json", manifest)
    write_json(
        output / "STATIC_MOE_READY.json",
        {
            "schema": manifest["schema"],
            "static_files": len(files),
            "static_manifest_sha256": sha256(output / "tpu-static-moe.json"),
            "model_index_sha256": sha256(output / "model.safetensors.index.json"),
            "source_config_sha256": sha256(output / "config.json"),
            "self_contained": True,
        },
    )
    print(json.dumps({"event": "ep8_model_ready", "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    repack(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
