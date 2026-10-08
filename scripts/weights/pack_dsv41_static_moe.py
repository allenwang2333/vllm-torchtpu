#!/usr/bin/env python3
"""Assemble one self-contained DeepSeek V4.1 checkpoint with static MoE weights.

The prepared cache and source checkpoint are read-only inputs. Output contains
ordinary safetensors for every non-MoE tensor and private prepared-weights/
files for the MoE FP4 tensors. The READY marker is written last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MOE_KINDS = {"moe_fp4", "moe_scale"}
EXPECTED_PROJECTIONS = 40 * 384 * 3
COPY_CHUNK = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def checkpoint_headers(root: Path) -> dict[str, list[dict]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    result = {}
    for filename in sorted(set(index["weight_map"].values())):
        with (root / filename).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        base = 8 + length
        result[filename] = [
            {"name": name, "base": base, **metadata}
            for name, metadata in header.items()
            if name != "__metadata__"
        ]
    return result


def copy_exact(source, target, size: int) -> None:
    while size:
        chunk = source.read(min(size, COPY_CHUNK))
        if not chunk:
            raise EOFError(f"Short read from {source.name}")
        target.write(chunk)
        size -= len(chunk)


def write_original_shard(
    source: Path, target: Path, entries: list[dict], journal: Path
) -> None:
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for entry in entries:
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        header[entry["name"]] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    expected_size = 8 + len(encoded) + offset
    if journal.exists():
        recorded = json.loads(journal.read_text())
        if recorded != {"file": target.name, "bytes": expected_size}:
            raise ValueError(f"Resume journal differs: {journal}")
        if target.stat().st_size != expected_size:
            raise ValueError(f"Resume file size differs: {target}")
        return
    if target.exists():
        raise FileExistsError(f"Unjournaled output requires inspection: {target}")
    with source.open("rb") as src, target.open("xb") as dst:
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        for entry in entries:
            start, stop = entry["data_offsets"]
            src.seek(entry["base"] + start)
            copy_exact(src, dst, stop - start)
    if target.stat().st_size != expected_size:
        raise OSError(f"Short output shard: {target}")
    write_json(journal, {"file": target.name, "bytes": expected_size})
    print(
        json.dumps(
            {
                "event": "original_shard_complete",
                "file": target.name,
                "bytes": expected_size,
            }
        ),
        flush=True,
    )


def copy_prepared_shard(
    item: dict, prepared: Path, previous: Path, static_dir: Path
) -> dict:
    filename = item["file"]
    source_file = (previous if item.get("external") else prepared) / filename
    target_name = Path(filename).with_suffix(".fp4").name
    target = static_dir / target_name
    if target.exists():
        if target.stat().st_size != source_file.stat().st_size:
            raise ValueError(f"Copied shard size differs: {target}")
    else:
        with source_file.open("rb") as src, target.open("xb") as dst:
            copy_exact(src, dst, source_file.stat().st_size)
        print(
            json.dumps(
                {
                    "event": "moe_shard_copied",
                    "file": target_name,
                    "bytes": target.stat().st_size,
                }
            ),
            flush=True,
        )
    return {"file": f"prepared-weights/{target_name}", "bytes": target.stat().st_size}


def assemble(
    source: Path, prepared: Path, previous: Path, output: Path, workers: int = 8
) -> None:
    if workers < 1:
        raise ValueError("--copy-workers must be positive")
    complete = json.loads((prepared / "CONVERSION_COMPLETE.json").read_text())
    if not complete["cpu_conversion_complete"]:
        raise ValueError("Converter did not complete")
    if sha256(prepared / "manifest.json") != complete["manifest_sha256"]:
        raise ValueError("Converter manifest checksum mismatch")
    converted = json.loads((prepared / "manifest.json").read_text())
    if converted["schema"] != "dsv41.prepared-tensors.v1":
        raise ValueError("Unknown prepared weight schema")
    if converted["source_metadata_sha256"] != {
        name: sha256(source / name)
        for name in ("config.json", "model.safetensors.index.json")
    }:
        raise ValueError("Source checkpoint metadata differs from conversion")

    records = [e for e in converted["tensors"] if e.get("kind") in MOE_KINDS]
    keys = {e["key"] for e in records}
    if len(records) != EXPECTED_PROJECTIONS * 2 or len(keys) != len(records):
        raise ValueError(f"Incomplete or duplicate MoE tensors: {len(records)}")
    sources = {e["source"] for e in records}
    sources.update(e["source_scale"] for e in records)
    if len(sources) != EXPECTED_PROJECTIONS * 2:
        raise ValueError("Incomplete source MoE projections/scales")
    used_files = {e["file"] for e in records}

    output.mkdir(parents=True, exist_ok=True)
    static_dir = output / "prepared-weights"
    static_dir.mkdir(exist_ok=True)
    needed_files = [item for item in converted["files"] if item["file"] in used_files]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        static_files = list(
            pool.map(
                lambda item: copy_prepared_shard(item, prepared, previous, static_dir),
                needed_files,
            )
        )
    if {Path(e["file"]).stem + ".safetensors" for e in static_files} != used_files:
        raise ValueError("MoE shard file missing from converter manifest")

    weight_map = {}
    total_bytes = 0
    original_shards = []
    for ordinal, (filename, entries) in enumerate(checkpoint_headers(source).items()):
        kept = [entry for entry in entries if entry["name"] not in sources]
        if kept:
            original_shards.append((ordinal, filename, kept))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                lambda shard: write_original_shard(
                    source / shard[1],
                    output / f"model-{shard[0]:05d}.safetensors",
                    shard[2],
                    output / f"model-{shard[0]:05d}.json",
                ),
                original_shards,
            )
        )
    for ordinal, _, kept in original_shards:
        target = output / f"model-{ordinal:05d}.safetensors"
        for entry in kept:
            if entry["name"] in weight_map:
                raise ValueError(f"Duplicate source tensor: {entry['name']}")
            weight_map[entry["name"]] = target.name
            total_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]

    for item in source.rglob("*"):
        if not item.is_file() or item.suffix == ".safetensors":
            continue
        if item.relative_to(source).as_posix() == "model.safetensors.index.json":
            continue
        target = output / item.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
    )

    by_file = defaultdict(list)
    for record in records:
        by_file[record["file"]].append(record)
    static_records = [
        {**record, "file": Path(record["file"]).with_suffix(".fp4").name}
        for record in records
    ]
    static_manifest = {
        "schema": "dsv41.static-moe.v1",
        "source_commit": converted["source_commit"],
        "source_metadata_sha256": converted["source_metadata_sha256"],
        "quantization": {
            "dtype": "float4_e2m1fn_x2",
            "block": 512,
            "layout": "K_N",
            "packing": "low_nibble_first_along_N",
        },
        "source_expert_count": 384,
        "files": static_files,
        "tensors": static_records,
    }
    write_json(output / "tpu-static-moe.json", static_manifest)
    write_json(
        output / "STATIC_MOE_READY.json",
        {
            "schema": static_manifest["schema"],
            "moe_projections": EXPECTED_PROJECTIONS,
            "original_tensors": len(weight_map),
            "static_files": len(by_file),
            "model_index_sha256": sha256(output / "model.safetensors.index.json"),
            "static_manifest_sha256": sha256(output / "tpu-static-moe.json"),
            "source_config_sha256": sha256(output / "config.json"),
            "self_contained": True,
        },
    )
    print(
        json.dumps(
            {
                "event": "static_moe_ready",
                "output": str(output),
                "original_tensors": len(weight_map),
                "moe_projections": EXPECTED_PROJECTIONS,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-workers", type=int, default=8)
    args = parser.parse_args()
    assemble(args.source, args.prepared, args.previous, args.output, args.copy_workers)


if __name__ == "__main__":
    main()
