# SPDX-License-Identifier: Apache-2.0
"""Exact qualification for the opt-in K3 KDA Two Shot callsite."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from vllm_torchtpu import envs

SUPPORTED_BATCHES = (1, 2, 4, 8, 16, 32, 64)
COLLECTIVE_ID = 1
_MANIFEST = Path(__file__).with_suffix(".json")
_SOURCE_FILES = (
    "__init__.py",
    "platforms/tpu_platform.py",
    "kernels/collectives/two_shot.py",
    "models/vllm/kimi_k3/collective_ops.py",
    "models/vllm/kimi_k3/attention.py",
    "distributed/two_shot_qualification.py",
    "distributed/sharded_jax_op.py",
    "distributed/pallas_shapes.py",
    "compilation/shape_variants.py",
    "compilation/tpu_compiler.py",
    "runner/tpu_runner.py",
    "runner/tpu_runner_async_output.py",
    "env_override.py",
    "envs.py",
)
_RUNTIME_FLAGS = (
    "LIBTPU_INIT_ARGS",
    "XLA_FLAGS",
    "TORCH_TPU_INTERNAL_XLA_OPTIONS",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
)
_CONTEXT_IDENTITIES: dict[str, dict] = {}


def source_fingerprint() -> str:
    """Hash implementation sources, excluding the qualification document."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        content = (root / name).read_bytes()
        digest.update(name.encode() + b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def runtime_identity() -> dict:
    versions: dict[str, str | None] = {}
    for name in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "flags": {name: getattr(envs, name) for name in _RUNTIME_FLAGS},
    }


def qualification_identity(
    devices: list[dict], rank_device_ids: tuple[int, ...]
) -> dict:
    """Describe the physical slice without changing its process-rank ordering."""
    if len(devices) != 32 or len(rank_device_ids) != 32:
        raise ValueError("Two Shot K3 qualification requires 32 devices")
    by_id = {int(device["id"]): device for device in devices}
    if len(by_id) != 32 or set(by_id) != set(rank_device_ids):
        raise ValueError("Two Shot rank mapping must cover each device exactly once")
    if len({int(device.get("slice_index", 0)) for device in devices}) != 1:
        raise ValueError("Two Shot K3 qualification requires a single TPU slice")
    kinds = {str(device["device_kind"]) for device in devices}
    if len(kinds) != 1 or next(iter(kinds)).lower() not in {
        "tpu v7",
        "tpu v7x",
        "tpu7x",
    }:
        raise ValueError("Two Shot K3 qualification requires TPU v7 devices")
    if any(len(device["coords"]) != 3 for device in devices):
        raise ValueError("Two Shot requires three physical chip coordinates")
    axes = [sorted({int(d["coords"][a]) for d in devices}) for a in range(3)]
    if tuple(map(len, axes)) != (2, 2, 4) or any(
        axis != list(range(axis[0], axis[0] + len(axis))) for axis in axes
    ):
        raise ValueError("Two Shot K3 qualification requires a 2x2x4 slice")
    mapping = [
        [axes[a].index(int(by_id[i]["coords"][a])) for a in range(3)]
        + [int(by_id[i]["core_on_chip"])]
        for i in rank_device_ids
    ]
    expected = {
        (x, y, z, core)
        for x in range(2)
        for y in range(2)
        for z in range(4)
        for core in range(2)
    }
    if {tuple(item) for item in mapping} != expected:
        raise ValueError("Two Shot requires complete two-core physical chips")
    return {
        "source": source_fingerprint(),
        "runtime": runtime_identity(),
        "topology": {
            "device_kind": next(iter(kinds)),
            "shape": [2, 2, 4],
            "cores_per_chip": 2,
        },
        "rank_mapping": mapping,
    }


def load_manifest() -> dict:
    return json.loads(_MANIFEST.read_text())


def has_runtime_match(manifest: dict) -> bool:
    """Avoid device discovery when no entry can match this installed implementation."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        return False
    entries = manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        return False
    source, runtime = source_fingerprint(), runtime_identity()
    return any(
        isinstance(entry, dict)
        and isinstance(entry.get("identity"), dict)
        and entry["identity"].get("source") == source
        and entry["identity"].get("runtime") == runtime
        for entry in entries
    )


def qualified_batches(identity: dict, manifest: dict | None = None) -> tuple[int, ...]:
    """Return only exact measured shapes for an identical execution context."""
    manifest = load_manifest() if manifest is None else manifest
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        return ()
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return ()
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("identity") == identity
    ]
    if len(matches) != 1:
        return ()
    batches = matches[0].get("batches", [])
    if not isinstance(batches, list) or any(
        type(batch) is not int or batch not in SUPPORTED_BATCHES for batch in batches
    ):
        return ()
    return tuple(sorted(set(batches)))


def serving_context_supported(config, *, sp_prefill: bool) -> bool:
    parallel = config.parallel_config
    return (
        not sp_prefill
        # The current bridge requires separate executions to finish before
        # reuse. Async steps and multiple request chunks can overlap without
        # a last-collective-to-next-first-collective dependency.
        and config.scheduler_config.async_scheduling is False
        and config.scheduler_config.max_num_seqs == 1
        and parallel.tensor_parallel_size == 32
        and parallel.data_parallel_size == 1
        and int(os.getenv("TORCH_TPU_DP_SIZE", "0")) in (0, 1)
        and parallel.pipeline_parallel_size == 1
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and getattr(parallel, "decode_context_parallel_size", 1) == 1
        and not getattr(parallel, "enable_dbo", False)
        and getattr(parallel, "ubatch_size", 1) == 1
        and config.speculative_config is None
    )


def register_context(name: str, identity: dict) -> None:
    _CONTEXT_IDENTITIES[name] = identity


def compilation_identity() -> dict:
    """Include admission and resolved placement when replaying compiled graphs."""
    try:
        manifest = hashlib.sha256(_MANIFEST.read_bytes()).hexdigest()
    except OSError:
        # Admission falls back to native for an unavailable manifest. Cache
        # identity must remain available for that same execution path.
        manifest = "unavailable"
    return {
        "manifest": manifest,
        "contexts": dict(sorted(_CONTEXT_IDENTITIES.items())),
    }
