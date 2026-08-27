# SPDX-License-Identifier: Apache-2.0
"""Opt-in StableHLO dumps at TorchTPU's FX-to-MLIR boundary.

TorchTPU normally asks ``fx_to_mlir`` not to retain the MLIR module.  This
module installs a narrow, environment-gated wrapper which changes that flag
only for an explicit dump run and writes both supported representations:

* a versioned StableHLO portable artifact for machine consumption; and
* a debug-info-free MLIR text sidecar for humans.

The wrapper deliberately uses ``ExportedMlir``'s official serialization
methods.  In particular, it never treats a PJRT/Tier-3 executable as StableHLO
bytecode.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import socket
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STABLEHLO_DUMP_DIR_ENV = "VLLM_TORCHTPU_STABLEHLO_DUMP_DIR"
STABLEHLO_DUMP_RANKS_ENV = "VLLM_TORCHTPU_STABLEHLO_DUMP_RANKS"
STABLEHLO_DUMP_SCHEMA = "vllm-torchtpu-stablehlo-dump-v1"

_HOOK_MARKER = "_vllm_torchtpu_stablehlo_dump_hook"
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_PACKAGE_NAMES = (
    "torch",
    "torch-tpu",
    "vllm",
    "vllm-torchtpu",
    "jax",
    "jaxlib",
    "libtpu",
)


def stablehlo_dump_enabled() -> bool:
    """Return whether the current process requested StableHLO dumping."""
    return bool(os.getenv(STABLEHLO_DUMP_DIR_ENV, "").strip())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(value: str, *, fallback: str = "unknown") -> str:
    result = _FILENAME_SAFE.sub("-", value).strip("-._")
    return (result or fallback)[:96]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _distributed_rank() -> int | None:
    """Resolve global rank without initializing torch.distributed."""
    try:
        import torch

        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            return int(distributed.get_rank())
    except Exception:
        # Dumping must not initialize or otherwise perturb distributed state.
        pass

    for name in ("RANK", "VLLM_RANK", "TPU_PROCESS_INDEX", "TPU_WORKER_ID"):
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _selected_rank(rank: int | None, selection: str) -> bool:
    selection = selection.strip().lower()
    if selection in ("", "0"):
        # A single-process program often has no explicit rank and is rank zero
        # by convention.
        return rank in (None, 0)
    if selection in ("*", "all"):
        return True

    requested: set[int] = set()
    for item in selection.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            requested.add(int(item))
        except ValueError as exc:
            raise ValueError(
                f"{STABLEHLO_DUMP_RANKS_ENV} must be 'all' or a "
                f"comma-separated list of integers; got {selection!r}"
            ) from exc
    return (rank or 0) in requested


def _json_value(value: Any) -> Any:
    """Return bounded, JSON-safe metadata for an FX value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    result: dict[str, Any] = {"python_type": type(value).__qualname__}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        result.update({
            "kind":
            "tensor",
            "shape": [str(dim) for dim in value.shape],
            "dtype":
            str(value.dtype),
            "device":
            str(getattr(value, "device", "unknown")),
            "requires_grad":
            bool(getattr(value, "requires_grad", False)),
        })
        try:
            result["stride"] = [str(dim) for dim in value.stride()]
        except (AttributeError, RuntimeError, TypeError):
            pass
        return result

    rendered = repr(value)
    result["repr"] = rendered if len(
        rendered) <= 512 else rendered[:509] + "..."
    return result


def _fx_metadata(graph_module: Any,
                 example_inputs: Sequence[Any]) -> dict[str, Any]:
    placeholders = []
    nodes = []
    graph = getattr(graph_module, "graph", None)
    if graph is not None:
        for index, node in enumerate(graph.nodes):
            node_record = {
                "index": index,
                "name": str(node.name),
                "op": str(node.op),
                "target": str(node.target),
            }
            nodes.append(node_record)
            if node.op == "placeholder":
                value = node.meta.get("val") if isinstance(node.meta,
                                                           Mapping) else None
                placeholders.append({
                    **node_record, "value": _json_value(value)
                })

    code = str(getattr(graph_module, "code", ""))
    inputs = []
    for index, value in enumerate(example_inputs):
        encoded = _json_value(value)
        if isinstance(encoded, dict):
            inputs.append({"index": index, **encoded})
        else:
            inputs.append({"index": index, "value": encoded})

    return {
        "graph_module_type":
        type(graph_module).__qualname__,
        "graph_module_name":
        str(
            getattr(graph_module, "_get_name",
                    lambda: type(graph_module).__name__)()),
        "code_sha256":
        _sha256(code.encode("utf-8")),
        "placeholders":
        placeholders,
        "nodes":
        nodes,
        "example_inputs":
        inputs,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class StableHloDumper:
    """Serialize paired artifacts and their compiler-side ABI manifest."""

    def __init__(self, root: Path, *, rank_selection: str = "0") -> None:
        self.root = root
        self.rank_selection = rank_selection
        self._sequence = itertools.count()
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "StableHloDumper | None":
        value = os.getenv(STABLEHLO_DUMP_DIR_ENV, "").strip()
        if not value:
            return None
        return cls(
            Path(value).expanduser(),
            rank_selection=os.getenv(STABLEHLO_DUMP_RANKS_ENV, "0"),
        )

    def should_dump(self) -> bool:
        return _selected_rank(_distributed_rank(), self.rank_selection)

    def dump(
        self,
        exported_mlir: Any,
        *,
        graph_module: Any,
        example_inputs: Sequence[Any],
    ) -> Path | None:
        if bool(getattr(exported_mlir, "is_noop", False)):
            return None

        rank = _distributed_rank()
        if not _selected_rank(rank, self.rank_selection):
            return None

        # Serialize through TorchTPU while the ContextedModule and its MLIR
        # context are alive.  No parser or home-grown bytecode interpretation is
        # involved here.
        portable = bytes(exported_mlir.serialize_portable_artifact())
        mlir_text = str(exported_mlir.serialize_text(enable_debug_info=False))
        text_bytes = mlir_text.encode("utf-8")
        fx = _fx_metadata(graph_module, example_inputs)
        fx_code = str(getattr(graph_module, "code", ""))

        with self._lock:
            sequence = next(self._sequence)
            host = _safe_filename(socket.gethostname())
            rank_label = "unknown" if rank is None else str(rank)
            artifact_id = (f"{host}-rank{rank_label}-pid{os.getpid()}-"
                           f"graph{sequence:04d}-{_sha256(portable)[:12]}")
            output_dir = self.root / artifact_id
            output_dir.mkdir(parents=True, exist_ok=False)

            binary_path = output_dir / "module.stablehlo.mlirbc"
            text_path = output_dir / "module.stablehlo.mlir"
            fx_path = output_dir / "graph.fx.py"
            manifest_path = output_dir / "manifest.json"

            _atomic_write(binary_path, portable)
            _atomic_write(text_path, text_bytes)
            _atomic_write(fx_path, fx_code.encode("utf-8"))

            manifest = {
                "schema":
                STABLEHLO_DUMP_SCHEMA,
                "artifact_id":
                artifact_id,
                "created_at":
                datetime.now(timezone.utc).isoformat(),
                "process": {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "rank": rank,
                    "local_rank": os.getenv("LOCAL_RANK"),
                    "tpu_worker_id": os.getenv("TPU_WORKER_ID"),
                    "python": sys.version,
                    "platform": platform.platform(),
                },
                "packages":
                _package_versions(),
                "artifacts": {
                    "portable": {
                        "path": binary_path.name,
                        "bytes": len(portable),
                        "sha256": _sha256(portable),
                    },
                    "text": {
                        "path": text_path.name,
                        "bytes": len(text_bytes),
                        "sha256": _sha256(text_bytes),
                        "enable_debug_info": False,
                    },
                    "fx": {
                        "path": fx_path.name,
                        "bytes": len(fx_code.encode("utf-8")),
                        "sha256": _sha256(fx_code.encode("utf-8")),
                    },
                },
                "fx":
                fx,
                "exported_mlir": {
                    "updates_default_generator_state":
                    bool(
                        getattr(exported_mlir,
                                "updates_default_generator_state", False)),
                    "dynamic_outputs":
                    list(
                        getattr(exported_mlir, "dynamic_outputs", None) or []),
                    "unique_output_indices":
                    list(
                        getattr(exported_mlir, "unique_output_indices", None)
                        or []),
                    "mlir_results": [
                        _json_value(value) for value in getattr(
                            exported_mlir, "mlir_result_tensors", [])
                    ],
                },
                "notes": [
                    "portable is the machine-consumption authority; text is a sidecar",
                    "the FX manifest records compiler ABI order, not model-level pytree or checkpoint weight locations",
                ],
            }
            _atomic_write(
                manifest_path,
                (json.dumps(manifest, indent=2, sort_keys=True) +
                 "\n").encode("utf-8"),
            )
            return output_dir


def wrap_fx_to_mlir(original: Callable[..., Any],
                    dumper: StableHloDumper) -> Callable[..., Any]:
    """Wrap ``fx_to_mlir`` while preserving its public call convention."""

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not dumper.should_dump():
            return original(*args, **kwargs)

        positional = list(args)
        if len(positional) >= 3:
            positional[2] = True
        else:
            kwargs["build_mlir_module"] = True

        result = original(*positional, **kwargs)
        if positional:
            graph_module = positional[0]
        else:
            graph_module = kwargs.get("graph_module")
        if len(positional) >= 2:
            example_inputs = positional[1]
        else:
            example_inputs = kwargs.get("args")
        if graph_module is None or example_inputs is None:
            raise TypeError(
                "fx_to_mlir wrapper expected graph_module and args")
        dumper.dump(
            result,
            graph_module=graph_module,
            example_inputs=example_inputs,
        )
        return result

    setattr(wrapped, _HOOK_MARKER, True)
    return wrapped


def install_stablehlo_dump_hook() -> bool:
    """Install the opt-in TorchTPU export hook once in this process."""
    dumper = StableHloDumper.from_environment()
    if dumper is None:
        return False

    from torch_tpu._internal import export as torch_tpu_export

    current = torch_tpu_export.fx_to_mlir
    if getattr(current, _HOOK_MARKER, False):
        return True
    wrapped = wrap_fx_to_mlir(current, dumper)

    # TpuBackend may already have imported fx_to_mlir directly by the time
    # vllm-torchtpu installs this opt-in hook. Replace those cached references
    # by identity as well as the defining export-module attribute. This is
    # deliberately limited to TorchTPU's private modules and only runs in an
    # explicit dump process.
    for module_name, module in tuple(sys.modules.items()):
        if (module is None
                or not module_name.startswith("torch_tpu._internal")):
            continue
        for attribute, value in tuple(vars(module).items()):
            if value is current:
                setattr(module, attribute, wrapped)
    torch_tpu_export.fx_to_mlir = wrapped
    return True


__all__ = [
    "STABLEHLO_DUMP_DIR_ENV",
    "STABLEHLO_DUMP_RANKS_ENV",
    "StableHloDumper",
    "install_stablehlo_dump_hook",
    "stablehlo_dump_enabled",
    "wrap_fx_to_mlir",
]
