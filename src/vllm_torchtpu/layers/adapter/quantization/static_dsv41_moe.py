"""Read the self-contained DeepSeek V4.1 static MoE tensor cache."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open  # pyrefly: ignore[missing-import]
from safetensors.torch import load  # pyrefly: ignore[missing-import]
from vllm.model_executor.layers.fused_moe import RoutedExperts


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=2)
def _index(root: str) -> dict[str, dict]:
    directory = Path(root)
    ready = json.loads((directory / "STATIC_MOE_READY.json").read_text())
    manifest_path = directory / "tpu-static-moe.json"
    if ready.get("schema") not in ("dsv41.static-moe.v1", "dsv41.static-moe.ep8.v2"):
        raise ValueError("Unknown DeepSeek V4.1 static MoE schema")
    if _sha256(manifest_path) != ready["static_manifest_sha256"]:
        raise ValueError("Static MoE manifest checksum mismatch")
    if _sha256(directory / "config.json") != ready["source_config_sha256"]:
        raise ValueError("Static MoE model config checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != ready["schema"]:
        raise ValueError("Static MoE manifest schema mismatch")
    if manifest["quantization"] != {
        "dtype": "float4_e2m1fn_x2",
        "block": 512,
        "layout": "K_N",
        "packing": "low_nibble_first_along_N",
    }:
        raise ValueError("Static MoE quantization format mismatch")
    if manifest["source_expert_count"] != 384:
        raise ValueError("Static MoE expert count mismatch")
    if ready["schema"] == "dsv41.static-moe.ep8.v2":
        if (
            manifest.get("ep_size") != 8
            or manifest.get("layers") != 40
            or ready.get("static_files") != 320
            or _sha256(directory / "model.safetensors.index.json")
            != ready["model_index_sha256"]
        ):
            raise ValueError("Static MoE EP8 model metadata mismatch")
        files = {}
        for item in manifest["files"]:
            layer, rank = item["layer"], item["rank"]
            filename = Path(item["file"])
            if (
                not 0 <= layer < 40
                or not 0 <= rank < 8
                or (layer, rank) in files
                or filename.is_absolute()
                or ".." in filename.parts
                or item["expert_ids"] != list(range(rank * 48, (rank + 1) * 48))
                or item["bytes"] <= 0
            ):
                raise ValueError(f"Invalid static MoE EP8 file entry: {item}")
            files[layer, rank] = item
        if len(files) != 320:
            raise ValueError("Incomplete static MoE EP8 files")
        return {"__ep8__": files}
    files = {item["file"] for item in manifest["files"]}
    for filename in files:
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"Unsafe static MoE file path: {filename}")
        if not (directory / filename).is_file():
            raise FileNotFoundError(directory / filename)
    result = {}
    for entry in manifest["tensors"]:
        filename = "prepared-weights/" + entry["file"]
        if filename not in files or entry["key"] in result:
            raise ValueError(f"Invalid static MoE tensor entry: {entry['key']}")
        result[entry["key"]] = {**entry, "file": filename}
    if len(result) != 40 * 384 * 3 * 2:
        raise ValueError(f"Incomplete static MoE manifest: {len(result)} tensors")
    return result


def _load_ep8_layer(
    root: Path,
    files: dict[tuple[int, int], dict],
    layer_id: int,
    experts: list[int],
    layer: RoutedExperts,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = min(experts) // 48
    if set(experts) != set(range(rank * 48, (rank + 1) * 48)):
        raise ValueError("Static MoE EP8 shard does not match local experts")
    item = files[layer_id, rank]
    path = root / item["file"]
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"Static MoE EP8 shard size mismatch: {path}")
    tensors = load(path.read_bytes())
    shapes = {
        "w13": (48, 5120, 2560),
        "w2": (48, 2560, 2560),
        "s13": (48, 10, 1, 5120),
        "s2": (48, 5, 1, 5120),
    }
    if set(tensors) != set(shapes):
        raise ValueError(f"Unexpected static MoE EP8 tensors: {path}")
    for name, shape in shapes.items():
        dtype = torch.uint8 if name.startswith("w") else torch.float32
        if tuple(tensors[name].shape) != shape or tensors[name].dtype != dtype:
            raise ValueError(f"Static MoE EP8 tensor shape/dtype mismatch: {name}")
    order: list[int | None] = [None] * 48
    for expert in experts:
        slot = layer.expert_map_manager.map_global_to_local(expert)
        if slot < 0 or slot >= 48 or order[slot] is not None:
            raise ValueError("Invalid static MoE EP8 expert slot mapping")
        order[slot] = expert - rank * 48
    if any(slot is None for slot in order):
        raise ValueError("Incomplete static MoE EP8 expert slots")
    if order != list(range(48)):
        indices = torch.tensor(
            [slot for slot in order if slot is not None], dtype=torch.int64
        )
        tensors = {
            name: value.index_select(0, indices) for name, value in tensors.items()
        }
    native = torch.float4_e2m1fn_x2
    return (
        tensors["w13"].view(native).to(device),
        tensors["w2"].view(native).to(device),
        tensors["s13"].to(device),
        tensors["s2"].to(device),
    )


def _read_tensors(
    root: Path, index: dict[str, dict], keys: list[str]
) -> dict[str, torch.Tensor]:
    by_file = {}
    for key in keys:
        by_file.setdefault(index[key]["file"], []).append(key)
    result = {}
    for filename, members in by_file.items():
        with safe_open(str(root / filename), framework="numpy") as stream:
            for key in members:
                entry = index[key]
                data = stream.get_tensor(key)
                if list(data.shape) != entry["shape"]:
                    raise ValueError(f"Static MoE shape mismatch: {key}")
                if entry["dtype"] == "U8" and data.dtype != np.uint8:
                    raise ValueError(f"Static MoE dtype mismatch: {key}")
                if entry["dtype"] == "F32" and data.dtype != np.float32:
                    raise ValueError(f"Static MoE scale dtype mismatch: {key}")
                result[key] = torch.from_numpy(np.array(data, copy=True))
    return result


def load_layer(
    root: str, layer: RoutedExperts, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
    if match is None or int(match[1]) >= 40:
        raise ValueError(f"Not a DSv4.1 text MoE layer: {layer.layer_name}")
    layer_id = int(match[1])
    directory = Path(root)
    index = _index(str(directory.resolve()))
    experts = layer.expert_map_manager.get_local_expert_ids()
    if not experts or len(experts) != layer.local_num_experts:
        raise ValueError("Static MoE local expert map mismatch")
    if "__ep8__" in index:
        return _load_ep8_layer(
            directory, index["__ep8__"], layer_id, experts, layer, device
        )
    keys = []
    for global_id in experts:
        for projection in (1, 2, 3):
            source = f"layers.{layer_id}.ffn.experts.{global_id}.w{projection}.weight"
            keys.extend((source + "::fp4", source + "::scale"))
    tensors = _read_tensors(directory, index, keys)
    w13 = [None] * len(experts)
    w2 = [None] * len(experts)
    s13 = [None] * len(experts)
    s2 = [None] * len(experts)
    for global_id in experts:
        local_id = layer.expert_map_manager.map_global_to_local(global_id)
        if local_id < 0 or local_id >= len(experts) or w13[local_id] is not None:
            raise ValueError("Invalid static MoE expert slot mapping")
        parts = {}
        for projection in (1, 2, 3):
            source = f"layers.{layer_id}.ffn.experts.{global_id}.w{projection}.weight"
            parts[projection] = (
                tensors[source + "::fp4"],
                tensors[source + "::scale"],
            )
        w13[local_id] = torch.cat((parts[1][0], parts[3][0]), dim=-1)
        s13[local_id] = torch.cat((parts[1][1], parts[3][1]), dim=-1)
        w2[local_id] = parts[2][0]
        s2[local_id] = parts[2][1]
    if any(item is None for item in (*w13, *w2, *s13, *s2)):
        raise ValueError("Incomplete local static MoE expert slots")

    packed_w13 = torch.stack(w13).contiguous()
    packed_w2 = torch.stack(w2).contiguous()
    if packed_w13.shape != (len(experts), 5120, 2560):
        raise ValueError(f"Static MoE w13 shape mismatch: {packed_w13.shape}")
    if packed_w2.shape != (len(experts), 2560, 2560):
        raise ValueError(f"Static MoE w2 shape mismatch: {packed_w2.shape}")
    native = torch.float4_e2m1fn_x2
    return (
        packed_w13.view(native).to(device),
        packed_w2.view(native).to(device),
        torch.stack(s13).contiguous().to(device),
        torch.stack(s2).contiguous().to(device),
    )
