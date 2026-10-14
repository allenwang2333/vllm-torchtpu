# SPDX-License-Identifier: Apache-2.0
"""StableHLO (.mlir) model export and save-load round-trip pipeline (RFC 0001).

External entry points (called outside `vllm_torchtpu/export/`):
1. `update_hash_with_loaded_hlo(hash_obj)` — `tpu_compiler.py`
2. `maybe_export_or_load_stablehlo(adaptor, graph, example_inputs, compile_range, key)` — `tpu_compiler.py`
3. `_install_warmup_early_exit_hook(vllm_config)` — `tpu_platform.py`
4. `register_active_model(model)` — `tpu_runner.py`
5. `maybe_export_sampling_subgraphs(runner)` — `tpu_runner.py`
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import dataclasses
import json
import os
import re
import shutil
import struct
import sys
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from vllm.config.utils import Range

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

logger = init_logger(__name__)

_MLIR_DTYPE_TO_TORCH_NAME = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "ui8": "uint8",
    "i16": "int16",
    "ui16": "uint16",
    "i32": "int32",
    "ui32": "uint32",
    "i64": "int64",
    "ui64": "uint64",
    "f8E4M3FN": "float8_e4m3fn",
    "f8E5M2": "float8_e5m2",
}
_TORCH_TENSOR_DTYPES = set(_MLIR_DTYPE_TO_TORCH_NAME.values())

_ALLOWED_CUSTOM_CALL_TARGETS = frozenset(
    {
        "tpu_custom_call",
        "xla_tpu_custom_call",
        "Sharding",
        "SPMDFullToShardShape",
        "SPMDShardToFullShape",
        "pin_to_device",
        "mark_tensor",
    }
)

_TORCH_TO_SAFETENSORS_DTYPE: dict[torch.dtype, str] = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
}

_ACTIVE_PARAM_MAP: dict[int, str] = {}
_ACTIVE_MODEL: Any = None
_CAPTURED_CONSTANTS: dict[str, torch.Tensor] = {}


def update_hash_with_loaded_hlo(hash_obj: Any) -> None:
    """Fold `VLLM_TORCHTPU_LOAD_HLO_DIR` `.mlir` and metadata into compilation hash."""
    load_hlo_dir = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if not load_hlo_dir:
        return
    load_dir = Path(load_hlo_dir)
    if not load_dir.is_dir():
        return
    for hlo_path in sorted(
        [
            *load_dir.glob("*.mlir"),
            load_dir / "metadata.json",
            load_dir / "serving_checkpoint" / "weight.json",
        ]
    ):
        if hlo_path.is_file():
            hash_obj.update(hlo_path.name.encode())
            with contextlib.suppress(OSError):
                hash_obj.update(hlo_path.read_bytes())


def register_active_model(model: Any) -> None:
    """Register an active `torch.nn.Module` for parameter name resolution."""
    global _ACTIVE_PARAM_MAP, _ACTIVE_MODEL
    _ACTIVE_PARAM_MAP.clear()
    _ACTIVE_MODEL = model
    if model is None:
        return
    for iter_fn in ("named_parameters", "named_buffers"):
        if hasattr(model, iter_fn):
            for name, t in getattr(model, iter_fn)():
                if isinstance(t, torch.Tensor):
                    with contextlib.suppress(Exception):
                        _ACTIVE_PARAM_MAP[t.data_ptr()] = name
                        _ACTIVE_PARAM_MAP[t.untyped_storage().data_ptr()] = name


def _dtype_name(val: Any) -> str:
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        return str(dtype).removeprefix("torch.")
    if isinstance(val, torch.dtype):
        return str(val).removeprefix("torch.")
    return type(val).__name__


def _shape_to_list(val: Any) -> list[int | str]:
    shape_raw = getattr(val, "shape", None)
    if shape_raw is None:
        return []
    out: list[int | str] = []
    for dim in shape_raw:
        try:
            out.append(int(dim))
        except Exception:
            out.append(str(dim))
    return out


def _extract_sharding_spec(node: fx.Node | None, val: Any) -> str | None:
    if node is not None:
        for meta_key in ("sharding", "sharding_spec", "partition_spec"):
            if node.meta.get(meta_key) is not None:
                return str(node.meta[meta_key])
    sharding = getattr(val, "sharding", None)
    return str(sharding) if sharding is not None else None


def _ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    out = next((n for n in graph.graph.nodes if n.op == "output"), None)
    if out is None or isinstance(out.args[0], (tuple, list)):
        return graph, False
    with graph.graph.inserting_before(out):
        out.args = ((out.args[0],),)
    graph.graph.lint()
    graph.recompile()
    return graph, True


def _parse_subgraph_index(key: str | None) -> int:
    if not key:
        return 0
    for pat in (r"(?:subgraph|submod)_(\d+)", r"_(\d+)$"):
        m = re.search(pat, key)
        if m is not None:
            return int(m.group(1))
    return 0


def _compute_fn_key(compile_range: Range, subgraph_idx: int) -> str:
    if subgraph_idx == 0:
        return f"forward_bucket{compile_range.start}"
    return f"forward_piece{subgraph_idx}_bucket{compile_range.start}"


def _get_target_hardware_fingerprint() -> dict[str, str | None]:
    fingerprint: dict[str, str | None] = {"chip_type": envs.TPU_ACCELERATOR_TYPE}
    for pkg_key, dist_name in (
        ("torch_tpu", "torch-tpu"),
        ("jaxlib", "jaxlib"),
        ("libtpu", "libtpu"),
    ):
        try:
            fingerprint[pkg_key] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            fingerprint[pkg_key] = None
    return fingerprint


def _parse_mlir_tensor_type(tensor_body: str) -> tuple[list[int | str], str]:
    core_body = tensor_body.split(",", 1)[0].strip()
    parts = core_body.split("x")
    elem = parts[-1]
    shape = [
        "?" if dim == "?" else int(dim)
        for dim in parts[:-1]
        if dim == "?" or dim.isdigit()
    ]
    return shape, _MLIR_DTYPE_TO_TORCH_NAME.get(elem, elem)


def _parse_mlir_main_args(args_str: str) -> list[tuple[list[int | str], str, bool]]:
    return [
        (
            *_parse_mlir_tensor_type(tensor_body),
            bool(attr_str and re.search(r"buffer_donor\s*=\s*true", attr_str)),
        )
        for tensor_body, attr_str in re.findall(
            r"tensor<([^>]+)>(\s*\{[^}]*\})?", args_str
        )
    ]


def _resolve_dynamo_placeholder(
    raw_name: str, model: Any = None
) -> tuple[str, bool, str, bool]:
    """Resolve a TorchDynamo placeholder name into (canonical_name, is_weight, arg_type, is_const)."""
    if not raw_name:
        return "", False, "user_io", False
    m = re.match(
        r"^l_self_modules_(.+)_(parameters|buffers)_([A-Za-z0-9_]+?)_?$", raw_name
    )
    if m:
        mod_tokens, kind, attr = m.group(1), m.group(2), m.group(3)
        sub_path = re.sub(r"_modules_", ".", mod_tokens)
        candidates = [f"model.{sub_path}.{attr}", f"{sub_path}.{attr}"]
        canonical = candidates[0]
        if model is not None:
            known = set()
            for fn in ("named_parameters", "named_buffers"):
                if hasattr(model, fn):
                    with contextlib.suppress(Exception):
                        known.update(k for k, _ in getattr(model, fn)())
            for cand in candidates:
                if cand in known:
                    canonical = cand
                    break
        is_const = kind == "buffers"
        if is_const and attr in ("cos_sin_cache", "cos_sin_cache_bf16"):
            canonical = "rotary_emb_inv_freq"
        return canonical, True, "weight", is_const

    if "kv_cache" in raw_name:
        m_layer = re.search(r"layers_(?:modules_)?(\d+)", raw_name)
        layer_suffix = m_layer.group(1) if m_layer else "0"
        return f"kv_cache_layer_{layer_suffix}", False, "state", False

    if raw_name.startswith("g_import_vllm") or "attn_metadata" in raw_name:
        for suffix, canonical in (
            ("seq_lens", "kv_lens"),
            ("kv_lens", "kv_lens"),
            ("block_tables", "block_tables"),
            ("query_start_loc", "query_start_loc"),
            ("request_distribution", "req_dist"),
            ("req_dist", "req_dist"),
        ):
            if raw_name.endswith(suffix):
                return canonical, False, "user_io", False

    m_local = re.match(r"^l_([A-Za-z0-9_]+?)_?$", raw_name)
    if m_local:
        return m_local.group(1), False, "user_io", False

    return raw_name, False, "user_io", False


def _build_generic_weight_sources(
    weight_name: str, shape: list[int | str], model: Any = None
) -> list[dict[str, Any]] | None:
    """Build RFC 0001 declarative `sources` by inspecting `model` submodules and `packed_modules_mapping`."""
    int_shape = [int(d) for d in shape if isinstance(d, int)]
    if not weight_name or not int_shape:
        return None

    if model is not None and "." in weight_name:
        mod_path, attr = weight_name.rsplit(".", 1)
        submod = None
        for cand_mod_path in (mod_path, mod_path.removeprefix("model.")):
            with contextlib.suppress(Exception):
                submod = model.get_submodule(cand_mod_path)
                if submod is not None:
                    break
        if submod is not None and attr == "weight":
            parent_path, leaf_mod_name = (
                mod_path.rsplit(".", 1) if "." in mod_path else ("", mod_path)
            )
            packed_map = getattr(model, "packed_modules_mapping", None) or {}
            sub_names = packed_map.get(leaf_mod_name)
            output_sizes = getattr(submod, "output_sizes", None)
            if (
                isinstance(sub_names, (list, tuple))
                and isinstance(output_sizes, (list, tuple))
                and len(sub_names) == len(output_sizes)
                and len(int_shape) == 2
            ):
                tp_size = int(getattr(submod, "tp_size", 1) or 1)
                local_sizes = [int(s) // max(tp_size, 1) for s in output_sizes]
                if sum(local_sizes) == int_shape[1]:
                    in_dim = int_shape[0]
                    sources: list[dict[str, Any]] = []
                    offset = 0
                    for sub_n, out_sz in zip(sub_names, local_sizes):
                        src_prefix = f"{parent_path}.{sub_n}" if parent_path else sub_n
                        sources.append(
                            {
                                "path": "",
                                "source_tensor": f"{src_prefix}.{attr}",
                                "source_shape": [out_sz, in_dim],
                                "transpose": [1, 0],
                                "dst_slices": [
                                    {"dim": 1, "start": offset, "length": out_sz}
                                ],
                            }
                        )
                        offset += out_sz
                    return sources
                if sum(local_sizes) == int_shape[0]:
                    in_dim = int_shape[1]
                    sources = []
                    offset = 0
                    for sub_n, out_sz in zip(sub_names, local_sizes):
                        src_prefix = f"{parent_path}.{sub_n}" if parent_path else sub_n
                        sources.append(
                            {
                                "path": "",
                                "source_tensor": f"{src_prefix}.{attr}",
                                "source_shape": [out_sz, in_dim],
                                "dst_slices": [
                                    {"dim": 0, "start": offset, "length": out_sz}
                                ],
                            }
                        )
                        offset += out_sz
                    return sources

            if (
                len(int_shape) == 2
                and not hasattr(submod, "num_embeddings")
                and "embedding" not in type(submod).__name__.lower()
                and (
                    hasattr(submod, "input_size")
                    or hasattr(submod, "input_size_per_partition")
                    or hasattr(submod, "in_features")
                )
            ):
                return [
                    {
                        "path": "",
                        "source_tensor": weight_name,
                        "source_shape": [int_shape[1], int_shape[0]],
                        "transpose": [1, 0],
                    }
                ]

    return [{"path": "", "source_tensor": weight_name, "source_shape": int_shape}]


def _extract_input_specs(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    outer_graph: fx.GraphModule | None = None,
    outer_inputs: list[Any] | None = None,
) -> list[dict[str, Any]]:
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    outer_phs = (
        [node for node in outer_graph.graph.nodes if node.op == "placeholder"]
        if outer_graph is not None and outer_graph is not graph
        else []
    )
    use_outer = len(outer_phs) == len(placeholders) > 0

    specs: list[dict[str, Any]] = []
    idx = 0
    for i in range(max(len(placeholders), len(example_inputs))):
        node = placeholders[i] if i < len(placeholders) else None
        outer_node = outer_phs[i] if use_outer and i < len(outer_phs) else None
        val = example_inputs[i] if i < len(example_inputs) else None
        if val is None and node is not None:
            val = node.meta.get("example_value", node.meta.get("val"))
        if val is None and outer_node is not None:
            val = outer_node.meta.get("example_value", outer_node.meta.get("val"))
        if (
            val is None
            or isinstance(val, (int, float, bool, str))
            or type(val).__name__ == "SymInt"
            or getattr(val, "shape", None) is None
            or _dtype_name(val) not in _TORCH_TENSOR_DTYPES
        ):
            continue

        is_weight, weight_name = False, ""
        arg_type = "user_io"
        is_const = False

        if outer_node is not None:
            cand_name, cand_w, cand_type, cand_const = _resolve_dynamo_placeholder(
                str(outer_node.name or outer_node.target or ""), _ACTIVE_MODEL
            )
            if cand_w:
                is_weight, weight_name, arg_type, is_const = (
                    True,
                    cand_name,
                    "weight",
                    cand_const,
                )
            elif cand_name:
                arg_type = cand_type

        if not is_weight and node is not None:
            cand_name, cand_w, cand_type, cand_const = _resolve_dynamo_placeholder(
                str(node.name or node.target or ""), _ACTIVE_MODEL
            )
            if cand_w:
                is_weight, weight_name, arg_type, is_const = (
                    True,
                    cand_name,
                    "weight",
                    cand_const,
                )
            elif cand_type != "user_io":
                arg_type = cand_type

        if not is_weight and node is not None:
            desc = node.meta.get("desc")
            if (
                desc is not None
                and (
                    getattr(desc, "is_param", False)
                    or getattr(desc, "is_buffer", False)
                    or type(desc).__name__ in ("ParamAOTInput", "BufferAOTInput")
                )
                and getattr(desc, "target", None)
            ):
                is_weight, weight_name, arg_type = True, str(desc.target), "weight"
                is_const = bool(
                    getattr(desc, "is_buffer", False)
                    or type(desc).__name__ == "BufferAOTInput"
                )
            if not is_weight:
                for k in ("param_name", "parameter_name", "fwd_bwd_target"):
                    if node.meta.get(k):
                        is_weight, weight_name, arg_type = (
                            True,
                            str(node.meta[k]),
                            "weight",
                        )
                        break
            if not is_weight:
                target = str(node.target or "")
                name = str(node.name or "")
                if hasattr(graph, target) and isinstance(
                    getattr(graph, target), (torch.nn.Parameter, torch.Tensor)
                ):
                    is_weight, weight_name, arg_type = True, target, "weight"
                elif target.startswith("p_") or name.startswith("p_"):
                    is_weight, weight_name, arg_type = (
                        True,
                        (target[2:] if target.startswith("p_") else name[2:]),
                        "weight",
                    )
        if not is_weight and _ACTIVE_PARAM_MAP:
            for cand_v in (
                val,
                outer_inputs[i]
                if outer_inputs is not None and i < len(outer_inputs)
                else None,
            ):
                if (
                    isinstance(cand_v, torch.Tensor)
                    and type(cand_v).__name__ != "FakeTensor"
                ):
                    with contextlib.suppress(Exception):
                        for ptr in (
                            cand_v.data_ptr(),
                            cand_v.untyped_storage().data_ptr(),
                        ):
                            if ptr in _ACTIVE_PARAM_MAP:
                                is_weight, weight_name, arg_type = (
                                    True,
                                    _ACTIVE_PARAM_MAP[ptr],
                                    "weight",
                                )
                                break
                if is_weight:
                    break
        if not is_weight and node is not None:
            name = str(node.name or "")
            if any(
                s in name.lower()
                for s in (
                    "_weight",
                    "_bias",
                    ".weight",
                    ".bias",
                    "w_q",
                    "w_k",
                    "w_v",
                    "w_o",
                    "embed_tokens",
                    "lm_head",
                )
            ):
                is_weight, weight_name, arg_type = True, name, "weight"
        if not is_weight and (
            isinstance(val, torch.nn.Parameter) or getattr(val, "_is_param", False)
        ):
            is_weight, weight_name, arg_type = (
                True,
                str(getattr(val, "_param_name", getattr(val, "name", ""))),
                "weight",
            )

        resolved_node_name = node.name if node is not None else f"arg_{idx}"
        if outer_node is not None:
            cand_name, _, _, _ = _resolve_dynamo_placeholder(
                str(outer_node.name or outer_node.target or ""), _ACTIVE_MODEL
            )
            if cand_name:
                resolved_node_name = cand_name
        elif is_weight and weight_name:
            resolved_node_name = weight_name

        shape_list = _shape_to_list(val)
        spec_entry: dict[str, Any] = {
            "index": idx,
            "name": weight_name if is_weight and weight_name else resolved_node_name,
            "shape": shape_list,
            "dtype": _dtype_name(val),
            "sharding": _extract_sharding_spec(node or outer_node, val),
            "is_weight": is_weight,
            "weight_name": weight_name,
        }
        if outer_node is not None or is_weight or arg_type != "user_io":
            spec_entry["argument_type"] = "weight" if is_weight else arg_type
        if is_const:
            spec_entry["is_internal_constant"] = True
        if is_weight and weight_name:
            sources = _build_generic_weight_sources(
                weight_name, shape_list, _ACTIVE_MODEL
            )
            if sources:
                spec_entry["sources"] = sources

        specs.append(spec_entry)
        idx += 1

    return specs


def _extract_output_specs(
    graph: fx.GraphModule, exported_mlir: Any | None = None
) -> list[dict[str, Any]]:
    result_tensors = getattr(exported_mlir, "mlir_result_tensors", None)
    if result_tensors:
        return [
            {
                "index": idx,
                "name": f"output_{idx}",
                "shape": _shape_to_list(val),
                "dtype": _dtype_name(val),
                "sharding": _extract_sharding_spec(None, val),
            }
            for idx, val in enumerate(result_tensors)
        ]

    out_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
    if out_node is None:
        return []
    raw_args = out_node.args[0] if out_node.args else ()
    items = raw_args if isinstance(raw_args, (tuple, list)) else (raw_args,)
    specs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        node = item if isinstance(item, fx.Node) else None
        val = (
            node.meta.get("example_value", node.meta.get("val"))
            if node is not None
            else None
        )
        if val is None:
            val = out_node.meta.get("example_value", out_node.meta.get("val"))
            if isinstance(val, (tuple, list)) and idx < len(val):
                val = val[idx]
        specs.append(
            {
                "index": idx,
                "name": node.name if node is not None else f"output_{idx}",
                "shape": _shape_to_list(val),
                "dtype": _dtype_name(val),
                "sharding": _extract_sharding_spec(node or out_node, val),
            }
        )
    return specs


def _write_exported_module(
    export_dir: Path,
    fn_key: str,
    subgraph_idx: int,
    compile_size: int,
    mlir_text: str,
    input_specs: list[dict[str, Any]],
    output_specs: list[dict[str, Any]],
) -> Path:
    """Persist `<fn_key>.mlir`, `metadata.json`, and `serving_checkpoint/weight.json`."""
    export_dir.mkdir(parents=True, exist_ok=True)
    mlir_filename = f"{fn_key}.mlir"
    mlir_path = export_dir / mlir_filename
    mlir_path.write_text(mlir_text)

    metadata_path = export_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt metadata.json in {export_dir}: {exc}") from exc

    annotated_inputs = [dict(inp) for inp in input_specs]
    match = re.search(r"func\.func\s+(?:public\s+)?@main\s*\((.*?)\)\s*->", mlir_text)
    if match is not None:
        mlir_args = _parse_mlir_main_args(match.group(1))
        if len(mlir_args) == len(annotated_inputs):
            for spec, (_, _, is_donor) in zip(annotated_inputs, mlir_args):
                spec["buffer_donor"] = is_donor

    metadata[fn_key] = {
        "file_path": mlir_filename,
        "subgraph_index": subgraph_idx,
        "compile_size": compile_size,
        "target_hardware": _get_target_hardware_fingerprint(),
        "inputs": annotated_inputs,
        "outputs": output_specs,
    }
    metadata.setdefault("format_version", 2)
    tmp_metadata_path = export_dir / f"metadata.json.{os.getpid()}.tmp"
    tmp_metadata_path.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp_metadata_path, metadata_path)

    weight_map: dict[str, Any] = {}
    for inp in annotated_inputs:
        if inp.get("is_weight") and inp.get("weight_name"):
            w_name = str(inp["weight_name"])
            raw_shape = inp.get("shape", [])
            int_shape = [int(d) for d in raw_shape if isinstance(d, int)]
            sources = inp.get("sources") or (
                [{"path": "", "source_tensor": w_name, "source_shape": int_shape}]
                if int_shape
                else None
            )
            entry: dict[str, Any] = {
                "index": inp["index"],
                "shape": raw_shape,
                "dtype": inp["dtype"],
                "sharding": inp.get("sharding"),
            }
            if sources:
                entry["sources"] = sources
            if inp.get("is_internal_constant"):
                entry["is_internal_constant"] = True
            weight_map[w_name] = entry

    if weight_map:
        serving_ckpt_dir = export_dir / "serving_checkpoint"
        serving_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for wj_path in (export_dir / "weight.json", serving_ckpt_dir / "weight.json"):
            existing_weights: dict[str, Any] = {}
            if wj_path.is_file():
                with contextlib.suppress(Exception):
                    raw_wj = json.loads(wj_path.read_text())
                    if isinstance(raw_wj, dict):
                        existing_weights = raw_wj.get("weights", raw_wj)
                        existing_weights.pop("hf_checkpoint_path", None)
                        existing_weights.pop("checkpoint_path", None)
            existing_weights.update(weight_map)
            wj_path.write_text(json.dumps(existing_weights, indent=2))

    logger.info(
        "[TpuCompilerAdaptor] Exported StableHLO module '%s' to %s",
        fn_key,
        mlir_path,
    )
    return mlir_path


def _install_warmup_early_exit_hook(vllm_config: Any = None) -> None:
    """Pin `num_gpu_blocks_override` on load and handle multiprocess early exit on export."""
    load_dir_str = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if (
        vllm_config is not None
        and load_dir_str
        and getattr(vllm_config, "cache_config", None) is not None
        and vllm_config.cache_config.num_gpu_blocks_override is None
    ):
        meta_path = Path(load_dir_str) / "metadata.json"
        if meta_path.is_file():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_path.read_text())
                pinned = None
                for k, entry in meta.items():
                    if k.startswith("forward_bucket") and isinstance(entry, dict):
                        for inp in entry.get("inputs", []):
                            if inp.get("buffer_donor") is True:
                                shape = inp.get("shape", [])
                                if shape and isinstance(shape[0], int) and shape[0] > 0:
                                    pinned = int(shape[0])
                                    break
                        if pinned is not None:
                            break
                if pinned is None:
                    cfg_blocks = (meta.get("config") or {}).get("num_blocks")
                    if isinstance(cfg_blocks, int) and cfg_blocks > 0:
                        pinned = cfg_blocks
                if pinned is not None:
                    vllm_config.cache_config.num_gpu_blocks_override = pinned

    with contextlib.suppress(Exception):
        import vllm.v1.engine.utils as engine_utils

        if not getattr(
            engine_utils.wait_for_engine_startup, "_has_export_early_exit_hook", False
        ):
            engine_utils._orig_wait_for_engine_startup = (  # type: ignore[attr-defined]
                engine_utils.wait_for_engine_startup
            )

            def _wait_for_engine_startup_with_early_exit(
                *args: Any, **kwargs: Any
            ) -> Any:
                try:
                    return engine_utils._orig_wait_for_engine_startup(*args, **kwargs)
                except RuntimeError as exc:
                    export_dir_str = str(
                        envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or ""
                    ).strip()
                    if export_dir_str:
                        export_dir = Path(export_dir_str)
                        if (
                            (export_dir / "metadata.json").is_file()
                            or ": 0}" in str(exc)
                            or "Failed core proc(s):" in str(exc)
                        ):
                            logger.info(
                                "[TpuCompilerAdaptor] Exported all StableHLO modules to %s; exiting early.",
                                export_dir,
                            )
                            sys.exit(0)
                    raise

            _wait_for_engine_startup_with_early_exit._has_export_early_exit_hook = True  # type: ignore[attr-defined]
            engine_utils.wait_for_engine_startup = (
                _wait_for_engine_startup_with_early_exit
            )
            with contextlib.suppress(Exception):
                import vllm.v1.engine.core_client as core_client

                core_client.wait_for_engine_startup = (  # type: ignore[attr-defined]
                    _wait_for_engine_startup_with_early_exit
                )


def maybe_export_or_load_stablehlo(
    adaptor: TpuCompilerAdaptor,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compile_range: Range,
    key: str | None,
) -> Callable[..., Any] | None:
    """Export FX graph to `.mlir` (`EXPORT_HLO_DIR`) or load/compile `.mlir` (`LOAD_HLO_DIR`)."""
    export_dir_str = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    load_dir_str = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if export_dir_str and load_dir_str:
        raise ValueError(
            "VLLM_TORCHTPU_EXPORT_HLO_DIR and VLLM_TORCHTPU_LOAD_HLO_DIR "
            "are mutually exclusive."
        )
    if adaptor.cache_dir and "profile_cache" in adaptor.cache_dir:
        return None
    if not export_dir_str and not load_dir_str:
        return None

    from torch._subclasses.fake_tensor import FakeTensor
    from torch_tpu._internal import export as torch_tpu_export
    from torch_tpu._internal.compile import tpu_torch_compile
    from torch_tpu._internal.compile.compiler import StaticCompiler

    from vllm_torchtpu.compilation import shape_variants

    subgraph_idx = _parse_subgraph_index(key)
    needs_aot = any(isinstance(t, FakeTensor) for t in example_inputs) or any(
        node.op in ("get_attr", "call_module") for node in graph.graph.nodes
    )

    if export_dir_str:
        export_dir = Path(export_dir_str)

        def _save_fx(
            module: fx.GraphModule, args: list[Any], exported_mlir: Any, idx: int
        ) -> None:
            input_specs = _extract_input_specs(
                module, args, outer_graph=graph, outer_inputs=example_inputs
            )
            tensor_args = [
                a
                for a in args
                if a is not None
                and not isinstance(a, (int, float, bool, str))
                and type(a).__name__ != "SymInt"
                and getattr(a, "shape", None) is not None
                and _dtype_name(a) in _TORCH_TENSOR_DTYPES
            ]
            if len(tensor_args) == len(input_specs):
                for spec, val in zip(input_specs, tensor_args):
                    if spec.get("is_internal_constant") and isinstance(
                        val, torch.Tensor
                    ):
                        w_name = str(spec.get("weight_name") or spec.get("name") or "")
                        if (
                            w_name
                            and w_name not in _CAPTURED_CONSTANTS
                            and type(val).__name__ != "FakeTensor"
                            and not getattr(val, "is_meta", False)
                        ):
                            with contextlib.suppress(Exception):
                                _CAPTURED_CONSTANTS[w_name] = (
                                    val.detach().cpu().contiguous()
                                )

            _write_exported_module(
                export_dir=export_dir,
                fn_key=_compute_fn_key(compile_range, idx),
                subgraph_idx=idx,
                compile_size=compile_range.start,
                mlir_text=exported_mlir.serialize_text(),
                input_specs=input_specs,
                output_specs=_extract_output_specs(module, exported_mlir),
            )

        if needs_aot:
            orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
            orig_async = getattr(StaticCompiler, "_async_compile", False)
            part_counter = 0

            def _export_hook(
                module: fx.GraphModule,
                args: list[Any],
                build_mlir_module: bool = True,
                **kw: Any,
            ) -> Any:
                nonlocal part_counter
                exported_mlir = orig_fx_to_mlir(
                    module, args, build_mlir_module=True, **kw
                )
                if not exported_mlir.is_noop and exported_mlir.module is not None:
                    part_idx = subgraph_idx * 100 + part_counter
                    part_counter += 1
                    _save_fx(module, list(args), exported_mlir, part_idx)
                return exported_mlir

            export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
            torch_tpu_export.fx_to_mlir = _export_hook
            StaticCompiler._async_compile = False
            try:
                compiled_fn, _ = adaptor._run_backend(export_graph, example_inputs)
            finally:
                torch_tpu_export.fx_to_mlir = orig_fx_to_mlir
                StaticCompiler._async_compile = orig_async
            if was_wrapped:
                compiled_fn = adaptor._unwrap_compiled_fn(compiled_fn)
        else:
            export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
            exported_mlir = torch_tpu_export.fx_to_mlir(
                export_graph, example_inputs, build_mlir_module=True
            )
            _save_fx(graph, example_inputs, exported_mlir, subgraph_idx)
            templates = [
                (tuple(int(d) for d in t.shape), t.dtype, t.device)
                for t in getattr(exported_mlir, "mlir_result_tensors", [])
                if isinstance(t, torch.Tensor)
            ]

            def _dummy(*_a: Any, **_kw: Any) -> list[torch.Tensor]:
                return [
                    torch.zeros(s, dtype=dt, device=dev) for s, dt, dev in templates
                ]

            out_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
            if out_node is not None and not isinstance(out_node.args[0], (tuple, list)):
                compiled_fn = adaptor._unwrap_compiled_fn(_dummy)
            else:
                compiled_fn = lambda *a, **kw: tuple(_dummy(*a, **kw))

        if key is not None and compile_range.is_single_size():
            shape_variants.remember(adaptor.cache_dir, key, graph, compiled_fn)
        return compiled_fn

    # Load path (`VLLM_TORCHTPU_LOAD_HLO_DIR`)
    load_dir = Path(load_dir_str)
    metadata_path = load_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"StableHLO metadata file missing from {load_dir}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())

    def _validate_and_load(
        module: fx.GraphModule, args: list[Any], idx: int
    ) -> tuple[str, Path, str]:
        fn_key = _compute_fn_key(compile_range, idx)
        if fn_key not in metadata:
            raise FileNotFoundError(
                f"Function key '{fn_key}' missing from metadata.json in {load_dir}"
            )
        entry = metadata[fn_key]

        # Hardware & ABI check
        saved_hw = entry.get("target_hardware")
        if isinstance(saved_hw, dict):
            current_hw = _get_target_hardware_fingerprint()
            if (
                saved_hw.get("chip_type")
                and current_hw.get("chip_type")
                and saved_hw["chip_type"] != current_hw["chip_type"]
            ):
                raise ValueError(
                    f"Target hardware mismatch for '{fn_key}': "
                    f"artifact was exported for chip_type='{saved_hw['chip_type']}', "
                    f"but current runtime has chip_type='{current_hw['chip_type']}'"
                )
            for abi_key in ("torch_tpu", "jaxlib", "libtpu"):
                sv, cv = saved_hw.get(abi_key), current_hw.get(abi_key)
                if sv and cv and sv != cv:
                    logger.warning(
                        "Compiler ABI version difference for '%s' (%s): '%s' vs '%s'.",
                        fn_key,
                        abi_key,
                        sv,
                        cv,
                    )

        # Runtime FX input check
        runtime_inputs = _extract_input_specs(
            module, args, outer_graph=graph, outer_inputs=example_inputs
        )
        saved_inputs = entry.get("inputs", [])
        if len(runtime_inputs) != len(saved_inputs):
            raise ValueError(
                f"Input count mismatch for '{fn_key}': "
                f"expected {len(saved_inputs)}, got {len(runtime_inputs)}"
            )
        for i, (actual, expected) in enumerate(zip(runtime_inputs, saved_inputs)):
            exp_sharding, act_sharding = (
                expected.get("sharding"),
                actual.get("sharding"),
            )
            if (
                actual["shape"] != expected.get("shape")
                or actual["dtype"] != expected.get("dtype")
                or (
                    exp_sharding is not None
                    and act_sharding is not None
                    and act_sharding != exp_sharding
                )
            ):
                raise ValueError(
                    f"Input signature mismatch at index {i} ('{actual['name']}') "
                    f"for '{fn_key}': expected shape={expected.get('shape')}, "
                    f"dtype={expected.get('dtype')}, sharding={exp_sharding}; "
                    f"got shape={actual['shape']}, dtype={actual['dtype']}, "
                    f"sharding={act_sharding}"
                )

        mlir_path = load_dir / entry.get("file_path", f"{fn_key}.mlir")
        if not mlir_path.is_file():
            raise FileNotFoundError(
                f"StableHLO file '{mlir_path}' missing for '{fn_key}'"
            )
        mlir_text = mlir_path.read_text()

        # Custom call allowlist check
        targets = set(
            re.findall(
                r"stablehlo\.custom_call\s+@([A-Za-z0-9_.-]+)", mlir_text, re.DOTALL
            )
        ) | set(
            re.findall(
                r"stablehlo\.custom_call\b[^{]*\{[^}]*call_target_name\s*=\s*\"([^\"]+)\"",
                mlir_text,
                re.DOTALL,
            )
        )
        disallowed = sorted(t for t in targets if t not in _ALLOWED_CUSTOM_CALL_TARGETS)
        if disallowed:
            raise ValueError(
                f"Disallowed stablehlo.custom_call target(s) {disallowed} in '{fn_key}'"
            )

        # MLIR @main signature check
        saved_outputs = entry.get("outputs", [])
        match = re.search(
            r"func\.func\s+(?:public\s+)?@main\s*\((.*?)\)(?:\s*->\s*([^\{\n]+))?",
            mlir_text,
        )
        if match is None:
            if saved_inputs or saved_outputs:
                raise ValueError(
                    f"MLIR @main function signature missing for '{fn_key}'"
                )
        else:
            mlir_inputs = _parse_mlir_main_args(match.group(1))
            mlir_outputs = [
                _parse_mlir_tensor_type(t)
                for t in re.findall(r"tensor<([^>]+)>", match.group(2) or "")
            ]
            if len(mlir_inputs) != len(saved_inputs):
                raise ValueError(
                    f"MLIR @main input count mismatch for '{fn_key}': "
                    f"expected {len(saved_inputs)}, got {len(mlir_inputs)}"
                )
            for i, ((s, dt, donor), exp) in enumerate(zip(mlir_inputs, saved_inputs)):
                if s != exp.get("shape") or dt != exp.get("dtype"):
                    raise ValueError(
                        f"MLIR @main input signature mismatch at index {i} for '{fn_key}'"
                    )
                if exp.get("buffer_donor") is not None and donor != exp.get(
                    "buffer_donor"
                ):
                    raise ValueError(
                        f"MLIR @main buffer_donor mismatch at index {i} for '{fn_key}'"
                    )
            if len(mlir_outputs) != len(saved_outputs):
                raise ValueError(
                    f"MLIR @main output count mismatch for '{fn_key}': "
                    f"expected {len(saved_outputs)}, got {len(mlir_outputs)}"
                )
            for i, ((s, dt), exp) in enumerate(zip(mlir_outputs, saved_outputs)):
                if s != exp.get("shape") or dt != exp.get("dtype"):
                    raise ValueError(
                        f"MLIR @main output signature mismatch at index {i} for '{fn_key}'"
                    )

        return fn_key, mlir_path, mlir_text

    @dataclasses.dataclass(frozen=True)
    class _LoadedCompileResult:
        module: Any
        executable: Any

    if needs_aot:
        orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
        orig_traverse = getattr(tpu_torch_compile, "traverse_and_compile", None)
        orig_async = getattr(StaticCompiler, "_async_compile", False)
        part_counter = 0

        def _load_hook(
            module: fx.GraphModule,
            args: list[Any],
            build_mlir_module: bool = False,
            **kw: Any,
        ) -> Any:
            nonlocal part_counter

            def _load_traverse(**tc_kw: Any) -> _LoadedCompileResult:
                nonlocal part_counter
                part_idx = subgraph_idx * 100 + part_counter
                part_counter += 1
                resolved_key, mlir_path, mlir_text = _validate_and_load(
                    module, list(args), part_idx
                )
                mlir_mod = tpu_torch_compile.parse_mlir_text(mlir_text)
                pjrt_exec = tpu_torch_compile.compile_mlir(
                    mlir_mod, False, tc_kw.get("argument_layouts") or []
                )
                logger.info(
                    "[TpuCompilerAdaptor] Loaded StableHLO module '%s' from %s",
                    resolved_key,
                    mlir_path,
                )
                return _LoadedCompileResult(module=mlir_mod, executable=pjrt_exec)

            tpu_torch_compile.traverse_and_compile = _load_traverse
            try:
                return orig_fx_to_mlir(module, args, build_mlir_module=False, **kw)
            finally:
                if orig_traverse is not None:
                    tpu_torch_compile.traverse_and_compile = orig_traverse
                elif hasattr(tpu_torch_compile, "traverse_and_compile"):
                    delattr(tpu_torch_compile, "traverse_and_compile")

        export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
        torch_tpu_export.fx_to_mlir = _load_hook
        StaticCompiler._async_compile = False
        try:
            compiled_fn, _ = adaptor._run_backend(export_graph, example_inputs)
        finally:
            torch_tpu_export.fx_to_mlir = orig_fx_to_mlir
            StaticCompiler._async_compile = orig_async
        if was_wrapped:
            compiled_fn = adaptor._unwrap_compiled_fn(compiled_fn)
    else:
        from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
            TorchTpuCompiledExecutable,
        )

        resolved_key, mlir_path, mlir_text = _validate_and_load(
            graph, example_inputs, subgraph_idx
        )
        mlir_mod = tpu_torch_compile.parse_mlir_text(mlir_text)
        pjrt_exec = tpu_torch_compile.compile_mlir(mlir_mod)
        exec_fn = TorchTpuCompiledExecutable(
            executable=pjrt_exec,
            reconstruct_fx_outputs_fn=lambda args, outputs: outputs,
            updates_default_generator_state=False,
        )
        out_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
        if out_node is not None and not isinstance(out_node.args[0], (tuple, list)):
            compiled_fn = adaptor._unwrap_compiled_fn(exec_fn)
        else:
            compiled_fn = lambda *a, **kw: tuple(exec_fn(*a, **kw))
        logger.info(
            "[TpuCompilerAdaptor] Loaded StableHLO module '%s' from %s",
            resolved_key,
            mlir_path,
        )

    if key is not None and compile_range.is_single_size():
        shape_variants.remember(adaptor.cache_dir, key, graph, compiled_fn)
    return compiled_fn


def _compile_options_for(tensor_parallel_size: int) -> dict[str, Any]:
    """Returns the `compile_options` object of metadata.json (rvllm RFC 0001 §3.2.1).

    torch_tpu compiles each TP rank's module with `num_partitions` = TP,
    `replica_count` = 1 and `use_spmd_partitioning` = true (from the XLA dump of a
    TP=4 Qwen3-0.6B run). A TP=1 module is compiled for a single device.
    """
    if tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
        )
    return {
        "num_replicas": 1,
        "num_partitions": tensor_parallel_size,
        "use_spmd_partitioning": tensor_parallel_size > 1,
    }


def maybe_export_sampling_subgraphs(runner: Any) -> None:
    """Export sampling subgraphs, write `serving_checkpoint/weight.json`, and exit."""
    export_dir_str = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    if not export_dir_str:
        return
    export_dir = Path(export_dir_str)
    export_dir.mkdir(parents=True, exist_ok=True)

    from torch_tpu._internal import export as torch_tpu_export

    from vllm_torchtpu.runner import greedy_top_tokens

    hsize = runner.model_config.get_hidden_size()
    dtype = getattr(runner, "_hidden_states_dtype", torch.bfloat16)
    device = getattr(runner, "device", torch.device("cpu"))
    num_reqs_paddings = getattr(runner, "num_reqs_paddings", [1])

    lm_head_weight = None
    if hasattr(runner, "model"):
        lm_head = getattr(runner.model, "lm_head", None)
        if lm_head is not None:
            lm_head_weight = getattr(lm_head, "weight", None)
        if lm_head_weight is None and hasattr(runner.model, "named_parameters"):
            for p_name, p in runner.model.named_parameters():
                if "lm_head" in p_name and p_name.endswith(".weight"):
                    lm_head_weight = p
                    break

    class _LogitsExplicitGraph(torch.nn.Module):
        def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return torch.matmul(h, w.t())

    class _TopTokensExplicitGraph(torch.nn.Module):
        def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return torch.argmax(torch.matmul(h, w.t()), dim=-1, keepdim=True)

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.model.compute_logits(hidden_states)

    v_raw = getattr(getattr(runner, "model_config", None), "vocab_size", 151936)
    try:
        vocab_size = (
            int(lm_head_weight.shape[0]) if lm_head_weight is not None else int(v_raw)
        )
    except Exception:
        vocab_size = 151936

    # weight.json `index` is the global HBM weight slot. `lm_head.weight` is arg 1
    # of the sampling subgraphs, which is also the slot of forward's
    # `model.embed_tokens.weight`, so give it the first slot no forward weight uses.
    used_weight_indices: set[int] = set()
    for wj_path in (
        export_dir / "serving_checkpoint" / "weight.json",
        export_dir / "weight.json",
    ):
        if wj_path.is_file():
            with contextlib.suppress(Exception):
                raw_wj = json.loads(wj_path.read_text())
                for w_name, w_spec in raw_wj.get("weights", raw_wj).items():
                    if w_name != "lm_head.weight" and isinstance(w_spec, dict):
                        used_weight_indices.add(int(w_spec["index"]))
            break
    lm_head_index = max(used_weight_indices, default=0) + 1

    def _export_bucket(
        fn_key: str,
        num_reqs: int,
        dummy_hidden: torch.Tensor,
        explicit_mod: torch.nn.Module,
        fallback_mod: torch.nn.Module,
        out_name: str,
        out_shape: list[int],
        out_dtype: str,
    ) -> None:
        if lm_head_weight is not None:
            if str(dummy_hidden.device).startswith("tpu"):
                _ = dummy_hidden.cpu()
                _ = lm_head_weight.cpu()
            traced_gm = fx.symbolic_trace(explicit_mod)
            example_args = [dummy_hidden, lm_head_weight]
        else:
            traced_gm = None
            if hasattr(fx, "symbolic_trace"):
                with contextlib.suppress(Exception):
                    traced_gm = fx.symbolic_trace(fallback_mod)
            if traced_gm is None:
                g = fx.Graph()
                g.output(g.placeholder("hidden_states"))
                traced_gm = fx.GraphModule(torch.nn.Module(), g)
            example_args = [dummy_hidden]

        exported_mlir = torch_tpu_export.fx_to_mlir(
            traced_gm, example_args, build_mlir_module=True
        )
        if (
            getattr(exported_mlir, "is_noop", False)
            or getattr(exported_mlir, "module", None) is None
        ):
            return

        if lm_head_weight is not None:
            lm_shape = [int(d) for d in lm_head_weight.shape]
            input_specs = [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": list(dummy_hidden.shape),
                    "dtype": _dtype_name(dummy_hidden),
                    "is_weight": False,
                    "weight_name": "",
                },
                {
                    "index": lm_head_index,
                    "name": "lm_head.weight",
                    "shape": lm_shape,
                    "dtype": _dtype_name(lm_head_weight),
                    "is_weight": True,
                    "weight_name": "lm_head.weight",
                    "sources": [
                        {
                            "path": "",
                            "source_tensor": "lm_head.weight",
                            "source_shape": lm_shape,
                        }
                    ],
                },
            ]
            output_specs = [
                {"index": 0, "name": out_name, "shape": out_shape, "dtype": out_dtype}
            ]
        else:
            input_specs = _extract_input_specs(traced_gm, [dummy_hidden]) or [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": list(dummy_hidden.shape),
                    "dtype": _dtype_name(dummy_hidden),
                    "is_weight": False,
                    "weight_name": "",
                }
            ]
            output_specs = _extract_output_specs(traced_gm, exported_mlir) or [
                {"index": 0, "name": out_name, "shape": out_shape, "dtype": out_dtype}
            ]

        _write_exported_module(
            export_dir=export_dir,
            fn_key=fn_key,
            subgraph_idx=0,
            compile_size=num_reqs,
            mlir_text=exported_mlir.serialize_text(),
            input_specs=input_specs,
            output_specs=output_specs,
        )

    logits_mod = _LogitsWrapper(runner.model)
    for num_reqs in num_reqs_paddings:
        fn_key = f"compute_logits_req{num_reqs}"
        dummy_hidden = torch.zeros((num_reqs, hsize), dtype=dtype, device=device)
        with contextlib.suppress(Exception):
            _export_bucket(
                fn_key=fn_key,
                num_reqs=num_reqs,
                dummy_hidden=dummy_hidden,
                explicit_mod=_LogitsExplicitGraph(),
                fallback_mod=logits_mod,
                out_name="logits",
                out_shape=[num_reqs, vocab_size],
                out_dtype=_dtype_name(dummy_hidden),
            )

    head_owner = getattr(runner, "_greedy_head_owner", None)
    if head_owner is not None or lm_head_weight is not None:

        class _TopTokensWrapper(torch.nn.Module):
            def __init__(self, owner: Any) -> None:
                super().__init__()
                self.head_owner = owner

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return greedy_top_tokens.top_tokens(self.head_owner, hidden_states)

        top_tokens_mod = _TopTokensWrapper(head_owner)
        for num_reqs in num_reqs_paddings:
            fn_key = f"sample_top_tokens_req{num_reqs}"
            dummy_hidden = torch.zeros((num_reqs, hsize), dtype=dtype, device=device)
            with contextlib.suppress(Exception):
                _export_bucket(
                    fn_key=fn_key,
                    num_reqs=num_reqs,
                    dummy_hidden=dummy_hidden,
                    explicit_mod=_TopTokensExplicitGraph(),
                    fallback_mod=top_tokens_mod,
                    out_name="tokens",
                    out_shape=[num_reqs, 1],
                    out_dtype="int64",
                )

    # Finalize metadata.json global config
    meta_path = export_dir / "metadata.json"
    if meta_path.is_file():
        meta_dict = json.loads(meta_path.read_text())
        model_cfg = getattr(runner, "model_config", None)
        parallel_cfg = getattr(runner, "parallel_config", None)
        cache_cfg = getattr(runner, "cache_config", None)

        meta_dict["format_version"] = 2
        meta_dict["model_name"] = str(getattr(model_cfg, "model", "serving_model"))
        meta_dict["target_tpu"] = _get_target_hardware_fingerprint().get(
            "chip_type", "v5p"
        )

        block_size = int(getattr(cache_cfg, "block_size", 16) or 16)
        num_blocks = int(getattr(cache_cfg, "num_gpu_blocks", 512) or 512)
        for fn_key, entry in meta_dict.items():
            if fn_key.startswith("forward_bucket") and isinstance(entry, dict):
                for inp in entry.get("inputs", []):
                    if inp.get("buffer_donor") is True:
                        shape = inp.get("shape", [])
                        if shape and isinstance(shape[0], int) and shape[0] > 0:
                            num_blocks = int(shape[0])
                            break
                if num_blocks != 512:
                    break

        cfg_vocab = int(
            getattr(model_cfg, "vocab_size", 0)
            or (
                model_cfg.get_vocab_size()
                if hasattr(model_cfg, "get_vocab_size")
                else 0
            )
            or getattr(getattr(model_cfg, "hf_config", None), "vocab_size", 0)
            or 0
        )
        eos_token_id = int(
            getattr(model_cfg, "eos_token_id", 0)
            or getattr(getattr(model_cfg, "hf_config", None), "eos_token_id", 0)
            or 0
        )
        meta_dict["config"] = {
            "vocab_size": cfg_vocab,
            "eos_token_id": eos_token_id,
            "max_num_seqs": 64,
            "max_num_pages": num_blocks,
            "max_cu_seqs": 65,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "max_model_len": int(getattr(model_cfg, "max_model_len", 2048) or 2048),
            "tensor_parallel_size": int(
                getattr(parallel_cfg, "tensor_parallel_size", 1) or 1
            ),
            "pipeline_parallel_size": int(
                getattr(parallel_cfg, "pipeline_parallel_size", 1) or 1
            ),
            "data_parallel_size": 1,
        }
        meta_dict["compile_options"] = _compile_options_for(
            meta_dict["config"]["tensor_parallel_size"]
        )
        tmp_path = export_dir / f"metadata.json.{os.getpid()}.tmp"
        tmp_path.write_text(json.dumps(meta_dict, indent=2))
        os.replace(tmp_path, meta_path)

    # Bundle checkpoint assets, constants.safetensors, and per-source paths
    model_id = getattr(getattr(runner, "model_config", None), "model", None)
    snap_dir: Path | None = None
    if model_id and isinstance(model_id, (str, Path)):
        candidate = Path(model_id)
        if candidate.is_dir():
            snap_dir = candidate
        else:
            with contextlib.suppress(Exception):
                import importlib

                hf_hub = importlib.import_module("huggingface_hub")
                snap_dir = Path(
                    hf_hub.snapshot_download(
                        repo_id=str(model_id), local_files_only=True
                    )
                )
            if snap_dir is None or not snap_dir.is_dir():
                hf_home = Path(
                    os.environ.get(
                        "HF_HOME", str(Path.home() / ".cache" / "huggingface")
                    )
                )
                repo_slug = "models--" + str(model_id).replace("/", "--")
                snaps_root = hf_home / "hub" / repo_slug / "snapshots"
                if snaps_root.is_dir():
                    for cand in sorted(snaps_root.iterdir()):
                        if cand.is_dir() and list(cand.glob("*.safetensors")):
                            snap_dir = cand
                            break

    if snap_dir is not None and snap_dir.is_dir():
        for fname in (
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ):
            src, dst = snap_dir / fname, export_dir / fname
            if src.is_file() and not dst.is_file():
                with contextlib.suppress(OSError):
                    shutil.copy2(src, dst)
        for st in sorted(snap_dir.glob("*.safetensors")):
            dst = export_dir / st.name
            if not dst.exists():
                with contextlib.suppress(OSError):
                    dst.symlink_to(st.resolve())

    serving_ckpt_dir = export_dir / "serving_checkpoint"
    serving_ckpt_dir.mkdir(parents=True, exist_ok=True)
    weight_json_path = export_dir / "weight.json"
    serving_weight_json_path = serving_ckpt_dir / "weight.json"
    wj_source = (
        serving_weight_json_path
        if serving_weight_json_path.is_file()
        else weight_json_path
    )
    if wj_source.is_file():
        with contextlib.suppress(Exception):
            raw_wj = json.loads(wj_source.read_text())
            if isinstance(raw_wj, dict):
                weights_dict: dict[str, Any] = raw_wj.get("weights", raw_wj)
                weights_dict.pop("hf_checkpoint_path", None)
                weights_dict.pop("checkpoint_path", None)

                # Index safetensors headers in snap_dir / export_dir
                tensor_to_file: dict[str, str] = {}
                for search_dir in [snap_dir, export_dir] if snap_dir else [export_dir]:
                    if search_dir and search_dir.is_dir() and not tensor_to_file:
                        idx_json = search_dir / "model.safetensors.index.json"
                        if idx_json.is_file():
                            with contextlib.suppress(Exception):
                                wm = json.loads(idx_json.read_text()).get(
                                    "weight_map", {}
                                )
                                if isinstance(wm, dict):
                                    for tn, sn in wm.items():
                                        tensor_to_file[str(tn)] = str(
                                            (search_dir / str(sn)).absolute()
                                        )
                        for st_file in sorted(search_dir.glob("*.safetensors")):
                            fp_str = str(st_file.absolute())
                            with (
                                contextlib.suppress(Exception),
                                open(st_file, "rb") as f,
                            ):
                                lb = f.read(8)
                                if len(lb) == 8:
                                    hlen = struct.unpack("<Q", lb)[0]
                                    if 0 < hlen <= 100 * 1024 * 1024:
                                        hdr = json.loads(f.read(hlen).decode("utf-8"))
                                        if isinstance(hdr, dict):
                                            for k in hdr:
                                                if k != "__metadata__":
                                                    tensor_to_file[k] = fp_str

                default_st_path = ""
                if snap_dir is not None and (snap_dir / "model.safetensors").exists():
                    default_st_path = str((snap_dir / "model.safetensors").absolute())
                elif (export_dir / "model.safetensors").exists():
                    default_st_path = str((export_dir / "model.safetensors").absolute())

                # Resolve & save internal constants (e.g. rotary_emb_inv_freq)
                internal_tensors: dict[str, torch.Tensor] = {}
                for w_name, w_spec in weights_dict.items():
                    if not isinstance(w_spec, dict):
                        continue
                    if (
                        bool(w_spec.get("is_internal_constant"))
                        or w_name == "rotary_emb_inv_freq"
                    ):
                        shape_list = [
                            int(d)
                            for d in w_spec.get("shape", [])
                            if isinstance(d, int)
                        ] or [40960, 128]
                        if w_name == "rotary_emb_inv_freq":
                            rope_t = _CAPTURED_CONSTANTS.get("rotary_emb_inv_freq")
                            if rope_t is None or list(rope_t.shape) != shape_list:
                                for cand_mod in (
                                    getattr(runner, "model", None),
                                    _ACTIVE_MODEL,
                                ):
                                    if cand_mod is not None and hasattr(
                                        cand_mod, "named_modules"
                                    ):
                                        with contextlib.suppress(Exception):
                                            for _, mod in cand_mod.named_modules():
                                                for attr in (
                                                    "cos_sin_cache_bf16",
                                                    "cos_sin_cache",
                                                ):
                                                    buf = getattr(mod, attr, None)
                                                    if (
                                                        isinstance(buf, torch.Tensor)
                                                        and type(buf).__name__
                                                        != "FakeTensor"
                                                        and not getattr(
                                                            buf, "is_meta", False
                                                        )
                                                        and list(buf.shape)
                                                        == shape_list
                                                    ):
                                                        rope_t = (
                                                            buf.detach()
                                                            .cpu()
                                                            .to(torch.bfloat16)
                                                            .contiguous()
                                                        )
                                                        break
                                                if rope_t is not None:
                                                    break
                            if rope_t is None or list(rope_t.shape) != shape_list:
                                max_pos = (
                                    shape_list[0] if len(shape_list) >= 1 else 40960
                                )
                                head_dim = (
                                    shape_list[1] if len(shape_list) >= 2 else 128
                                )
                                rope_theta = 1000000.0
                                cfg_p = export_dir / "config.json"
                                if cfg_p.is_file():
                                    with contextlib.suppress(Exception):
                                        rope_theta = float(
                                            json.loads(cfg_p.read_text()).get(
                                                "rope_theta", rope_theta
                                            )
                                        )
                                half_dim = head_dim // 2
                                inv_freq = 1.0 / (
                                    rope_theta
                                    ** (
                                        torch.arange(0, half_dim, dtype=torch.float32)
                                        / float(half_dim)
                                    )
                                )
                                freqs = torch.outer(
                                    torch.arange(max_pos, dtype=torch.float32), inv_freq
                                )
                                rope_t = (
                                    torch.cat([freqs.cos(), freqs.sin()], dim=-1)
                                    .to(torch.bfloat16)
                                    .contiguous()
                                )
                            internal_tensors[w_name] = rope_t
                        elif w_name in _CAPTURED_CONSTANTS:
                            internal_tensors[w_name] = _CAPTURED_CONSTANTS[w_name]

                constants_st_path = serving_ckpt_dir / "constants.safetensors"
                if internal_tensors:
                    header: dict[str, Any] = {}
                    payloads: list[bytes] = []
                    offset = 0
                    for name, tensor in internal_tensors.items():
                        cpu_t = tensor.detach().cpu().contiguous()
                        st_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(cpu_t.dtype, "BF16")
                        nbytes = cpu_t.numel() * cpu_t.element_size()
                        raw = (
                            ctypes.string_at(cpu_t.data_ptr(), nbytes)
                            if nbytes > 0
                            else b""
                        )
                        end = offset + len(raw)
                        header[name] = {
                            "dtype": st_dtype,
                            "shape": [int(d) for d in cpu_t.shape],
                            "data_offsets": [offset, end],
                        }
                        payloads.append(raw)
                        offset = end
                    hdr_bytes = json.dumps(header, separators=(",", ":")).encode(
                        "utf-8"
                    )
                    pad = (8 - (len(hdr_bytes) % 8)) % 8
                    if pad:
                        hdr_bytes += b" " * pad
                    tmp_st = (
                        constants_st_path.parent
                        / f"constants.safetensors.{os.getpid()}.tmp"
                    )
                    with open(tmp_st, "wb") as f:
                        f.write(struct.pack("<Q", len(hdr_bytes)))
                        f.write(hdr_bytes)
                        for chunk in payloads:
                            f.write(chunk)
                    os.replace(tmp_st, constants_st_path)
                    for k in internal_tensors:
                        tensor_to_file[k] = str(constants_st_path.absolute())

                for w_name, w_spec in weights_dict.items():
                    if not isinstance(w_spec, dict):
                        continue
                    w_shape = [
                        int(d) for d in w_spec.get("shape", []) if isinstance(d, int)
                    ]
                    sources = w_spec.get("sources")
                    if not isinstance(sources, list) or not sources:
                        sources = [
                            {
                                "path": "",
                                "source_tensor": w_name,
                                "source_shape": w_shape,
                            }
                        ]
                        w_spec["sources"] = sources
                    for src_entry in sources:
                        if not isinstance(src_entry, dict):
                            continue
                        src_tensor = str(src_entry.get("source_tensor") or w_name)
                        if (
                            src_tensor == "lm_head.weight"
                            and "lm_head.weight" not in tensor_to_file
                            and "model.embed_tokens.weight" in tensor_to_file
                        ):
                            src_tensor = "model.embed_tokens.weight"
                            src_entry["source_tensor"] = src_tensor
                        src_entry["path"] = tensor_to_file.get(
                            src_tensor, default_st_path
                        )
                        if "source_shape" not in src_entry and w_shape:
                            src_entry["source_shape"] = w_shape
                    w_spec.pop("is_internal_constant", None)

                formatted = json.dumps(weights_dict, indent=2)
                serving_weight_json_path.write_text(formatted)
                weight_json_path.write_text(formatted)

    world_size = int(
        getattr(getattr(runner, "parallel_config", None), "world_size", 1) or 1
    )
    if world_size <= 1:
        logger.info(
            "[TpuCompilerAdaptor] Exported all StableHLO modules "
            "(backbone + sampling) to %s; exiting early.",
            export_dir,
        )
        sys.exit(0)
