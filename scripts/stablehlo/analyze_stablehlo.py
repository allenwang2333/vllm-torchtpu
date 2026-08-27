#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse and calibrate StableHLO artifacts with official MLIR bindings.

The binary path is always handled by StableHLO's compatibility API (portable
artifact) or MLIR's bytecode parser (raw bytecode).  The text path is always
handled by ``ir.Module.parse``.  Original MLIR text and bytecode are never
decoded with regular expressions or a private binary format implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA = "vllm-torchtpu-stablehlo-analysis-v1"
DEFAULT_ALLOWED_DIALECTS = ("builtin", "func", "stablehlo", "sdy")
ArtifactFormat = Literal["auto", "portable", "bytecode", "text"]


class AnalysisError(RuntimeError):
    """An artifact could not be parsed, verified, or compared."""


@dataclasses.dataclass(frozen=True)
class Bindings:
    ir: Any
    stablehlo: Any
    sdy: Any | None
    provider: str
    distribution: str
    distribution_version: str | None


@dataclasses.dataclass
class Snapshot:
    report: dict[str, Any]
    semantics: dict[str, dict[str, Any]]
    public_operations: dict[str, dict[str, Any]]
    order: list[str]
    canonical_assembly_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value,
                         sort_keys=True,
                         separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return _sha256(payload)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_bindings() -> Bindings:
    """Load the official StableHLO wheel, falling back to JAX's bindings."""
    try:
        ir = importlib.import_module("mlir.ir")
        stablehlo = importlib.import_module("mlir.dialects.stablehlo")
        try:
            sdy = importlib.import_module("mlir.dialects.sdy")
        except ImportError:
            sdy = None
        return Bindings(
            ir=ir,
            stablehlo=stablehlo,
            sdy=sdy,
            provider="stablehlo-wheel",
            distribution="stablehlo",
            distribution_version=_distribution_version("stablehlo"),
        )
    except ImportError:
        pass

    try:
        ir = importlib.import_module("jaxlib.mlir.ir")
        stablehlo = importlib.import_module("jaxlib.mlir.dialects.stablehlo")
        try:
            sdy = importlib.import_module("jaxlib.mlir.dialects.sdy")
        except ImportError:
            sdy = None
        return Bindings(
            ir=ir,
            stablehlo=stablehlo,
            sdy=sdy,
            provider="jaxlib",
            distribution="jaxlib",
            distribution_version=_distribution_version("jaxlib"),
        )
    except ImportError as exc:
        raise AnalysisError(
            "StableHLO Python bindings are unavailable. Install the official "
            "stablehlo wheel or jaxlib.") from exc


def _safe_call(obj: Any, name: str) -> Any | None:
    value = getattr(obj, name, None)
    if not callable(value):
        return None
    try:
        return value()
    except Exception:
        return None


def create_context(bindings: Bindings, *, allow_unregistered: bool) -> Any:
    """Create a context with upstream, StableHLO, and Shardy dialects."""
    ir = bindings.ir
    context = ir.Context()

    # jaxlib intentionally ships upstream func/arith/etc. registration in a
    # small extension rather than loading it into a bare ir.Context().
    if bindings.provider == "jaxlib":
        try:
            extension = importlib.import_module(
                "jaxlib.mlir._mlir_libs._jax_mlir_ext")
            registry = ir.DialectRegistry()
            extension.register_dialects(registry)
            context.append_dialect_registry(registry)
        except (ImportError, AttributeError):
            pass
    else:
        # Some official wheel builds provide the standard MLIR registration
        # extension. Context.load_all_available_dialects() remains the fallback.
        try:
            extension = importlib.import_module(
                "mlir._mlir_libs._mlirRegisterEverything")
            registry = ir.DialectRegistry()
            extension.register_dialects(registry)
            context.append_dialect_registry(registry)
        except (ImportError, AttributeError):
            pass

    load_all = getattr(context, "load_all_available_dialects", None)
    if callable(load_all):
        load_all()
    bindings.stablehlo.register_dialect(context)
    if bindings.sdy is not None:
        bindings.sdy.register_dialect(context)
    context.allow_unregistered_dialects = allow_unregistered
    return context


def _looks_like_text(raw: bytes) -> bool:
    try:
        prefix = raw[:4096].decode("utf-8").lstrip()
    except UnicodeDecodeError:
        return False
    return prefix.startswith(("module", "//", "#loc", "builtin.module"))


def parse_artifact(
    path: Path,
    requested_format: ArtifactFormat,
    *,
    context: Any,
    bindings: Bindings,
) -> tuple[Any, str, bytes]:
    raw = path.read_bytes()
    selected = requested_format
    if selected == "auto":
        selected = "text" if _looks_like_text(raw) else "portable"

    if selected == "text":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AnalysisError(
                f"{path} is not valid UTF-8 MLIR text") from exc
        try:
            return bindings.ir.Module.parse(text, context), "text", raw
        except Exception as exc:
            raise AnalysisError(
                f"official MLIR parser rejected {path}: {exc}") from exc

    if selected == "portable":
        try:
            module = bindings.stablehlo.deserialize_portable_artifact(
                context, raw)
            return module, "portable", raw
        except Exception as portable_error:
            if requested_format != "auto":
                raise AnalysisError(
                    f"StableHLO compatibility API rejected {path}: {portable_error}"
                ) from portable_error
            # ``serialize_bytecode()`` output is a valid secondary input form.
            # Try only MLIR's official bytecode parser and identify it clearly.
            try:
                module = bindings.ir.Module.parse(raw, context)
                return module, "bytecode", raw
            except Exception as bytecode_error:
                raise AnalysisError(
                    f"{path} is neither a StableHLO portable artifact nor "
                    "registered MLIR bytecode; portable error: {portable_error}; "
                    f"bytecode error: {bytecode_error}") from bytecode_error

    if selected == "bytecode":
        try:
            return bindings.ir.Module.parse(raw, context), "bytecode", raw
        except Exception as exc:
            raise AnalysisError(
                f"official MLIR bytecode parser rejected {path}: {exc}"
            ) from exc

    raise AnalysisError(f"unsupported artifact format: {selected}")


def _operation(value: Any) -> Any:
    return getattr(value, "operation", value)


@dataclasses.dataclass
class _Index:
    operations: list[tuple[str, Any]]
    operation_paths: dict[Any, str]
    block_paths: dict[Any, str]
    value_paths: dict[Any, str]


def _index_module(module: Any) -> _Index:
    operations: list[tuple[str, Any]] = []
    operation_paths: dict[Any, str] = {}
    block_paths: dict[Any, str] = {}
    value_paths: dict[Any, str] = {}

    def visit(op_like: Any, path: str) -> None:
        op = _operation(op_like)
        operations.append((path, op))
        operation_paths[op] = path
        for result_index, result in enumerate(op.results):
            value_paths[result] = f"{path}:result{result_index}"
        for region_index, region in enumerate(op.regions):
            for block_index, block in enumerate(region.blocks):
                block_path = f"{path}/region{region_index}/block{block_index}"
                block_paths[block] = block_path
                for argument_index, argument in enumerate(block.arguments):
                    value_paths[argument] = f"{block_path}:arg{argument_index}"
                for operation_index, child in enumerate(block.operations):
                    visit(child, f"{block_path}/op{operation_index}")

    visit(module.operation, "module")
    return _Index(operations, operation_paths, block_paths, value_paths)


def _attribute_items(op: Any) -> list[tuple[str, Any]]:
    return sorted(
        ((str(name), value) for name, value in op.attributes.items()),
        key=lambda item: item[0],
    )


def _attribute_summary(text: str, max_chars: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": _sha256(text.encode("utf-8")),
        "characters": len(text),
    }
    if len(text) <= max_chars:
        result["text"] = text
    else:
        edge = max(1, max_chars // 2)
        result["preview"] = text[:edge] + " ... <elided> ... " + text[-edge:]
    return result


def _type_record(type_value: Any, ir: Any) -> dict[str, Any]:
    text = str(type_value)
    result: dict[str, Any] = {"text": text}
    try:
        ranked = ir.RankedTensorType(type_value)
    except (TypeError, ValueError):
        return result

    shape = [int(value) for value in ranked.shape]
    dynamic_dimensions = [
        index for index in range(ranked.rank) if ranked.is_dynamic_dim(index)
    ]
    result.update({
        "kind": "ranked_tensor",
        "shape": shape,
        "element_type": str(ranked.element_type),
        "dynamic_dimensions": dynamic_dimensions,
    })
    if ranked.encoding is not None:
        result["encoding"] = str(ranked.encoding)
    return result


def _string_attribute(value: Any, ir: Any) -> str:
    try:
        return str(ir.StringAttr(value).value)
    except (TypeError, ValueError):
        rendered = str(value)
        if len(rendered) >= 2 and rendered[0] == rendered[-1] == '"':
            return rendered[1:-1]
        return rendered


def _region_semantics(op: Any, index: _Index, ir: Any) -> list[dict[str, Any]]:
    regions = []
    for region_index, region in enumerate(op.regions):
        blocks = []
        for block_index, block in enumerate(region.blocks):
            blocks.append({
                "path":
                index.block_paths[block],
                "arguments": [
                    _type_record(argument.type, ir)
                    for argument in block.arguments
                ],
                "operations": [
                    index.operation_paths[_operation(child)]
                    for child in block.operations
                ],
            })
        regions.append({"index": region_index, "blocks": blocks})
    return regions


def _operand_record(operand: Any, index: _Index, ir: Any) -> dict[str, Any]:
    source = index.value_paths.get(operand)
    if source is None:
        raise AnalysisError(f"operand {operand} has no indexed producer")
    return {"source": source, "type": _type_record(operand.type, ir)}


def _operation_records(
    index: _Index, ir: Any, *, max_attribute_chars: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    semantics: dict[str, dict[str, Any]] = {}
    public: dict[str, dict[str, Any]] = {}
    order = []

    for path, op in index.operations:
        order.append(path)
        attributes = {name: str(value) for name, value in _attribute_items(op)}
        operands = [_operand_record(value, index, ir) for value in op.operands]
        results = [{
            "source": index.value_paths[value],
            "type": _type_record(value.type, ir),
        } for value in op.results]
        regions = _region_semantics(op, index, ir)
        successors = [index.block_paths[block] for block in op.successors]
        semantic = {
            "path": path,
            "name": str(op.name),
            "operands": operands,
            "results": results,
            "attributes": attributes,
            "regions": regions,
            "successors": successors,
        }
        semantics[path] = semantic
        public[path] = {
            **{
                key: value
                for key, value in semantic.items() if key != "attributes"
            },
            "attributes": {
                name: _attribute_summary(value, max_attribute_chars)
                for name, value in attributes.items()
            },
            "semantic_sha256": _json_digest(semantic),
        }
    return semantics, public, order


def _function_abi(
    index: _Index,
    semantics: dict[str, dict[str, Any]],
    public: dict[str, dict[str, Any]],
    ir: Any,
) -> list[dict[str, Any]]:
    functions = []
    operations = dict(index.operations)
    for path, semantic in semantics.items():
        if semantic["name"] != "func.func":
            continue
        op = operations[path]
        attrs = dict(_attribute_items(op))
        name = _string_attribute(attrs.get("sym_name", "<anonymous>"), ir)
        entry_arguments = []
        returns = []
        if len(op.regions) > 0 and len(op.regions[0].blocks) > 0:
            entry_block = op.regions[0].blocks[0]
            entry_arguments = [
                _type_record(argument.type, ir)
                for argument in entry_block.arguments
            ]
            for child in entry_block.operations:
                child_op = _operation(child)
                if str(child_op.name) == "func.return":
                    returns.append([
                        _type_record(value.type, ir)
                        for value in child_op.operands
                    ])
        functions.append({
            "path":
            path,
            "name":
            name,
            "function_type":
            str(attrs.get("function_type", "")),
            "visibility":
            _string_attribute(attrs["sym_visibility"], ir)
            if "sym_visibility" in attrs else None,
            "inputs":
            entry_arguments,
            "returns":
            returns,
            "attributes":
            public[path]["attributes"],
        })
    return functions


def _type_occurrences(semantics: dict[str, dict[str, Any]],
                      marker: str) -> list[dict[str, Any]]:
    occurrences = []
    for path, op in semantics.items():
        for role in ("operands", "results"):
            for index, value in enumerate(op[role]):
                type_record = value["type"]
                if marker in type_record["text"] or marker in str(
                        type_record.get("encoding", "")):
                    occurrences.append({
                        "path": path,
                        "operator": op["name"],
                        "role": role[:-1],
                        "index": index,
                        "type": type_record,
                    })
    return occurrences


def _dynamic_occurrences(
        semantics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    occurrences = []
    for path, op in semantics.items():
        for role in ("operands", "results"):
            for index, value in enumerate(op[role]):
                dynamic = value["type"].get("dynamic_dimensions", [])
                if dynamic:
                    occurrences.append({
                        "path": path,
                        "operator": op["name"],
                        "role": role[:-1],
                        "index": index,
                        "dynamic_dimensions": dynamic,
                        "type": value["type"]["text"],
                    })
    return occurrences


def _custom_calls(
    index: _Index,
    semantics: dict[str, dict[str, Any]],
    public: dict[str, dict[str, Any]],
    ir: Any,
) -> list[dict[str, Any]]:
    operations = dict(index.operations)
    calls = []
    for path, semantic in semantics.items():
        if semantic["name"] != "stablehlo.custom_call":
            continue
        attrs = dict(_attribute_items(operations[path]))
        target = _string_attribute(attrs.get("call_target_name", "<missing>"),
                                   ir)
        calls.append({
            "path":
            path,
            "target":
            target,
            "operand_types":
            [value["type"]["text"] for value in semantic["operands"]],
            "result_types":
            [value["type"]["text"] for value in semantic["results"]],
            "attributes":
            public[path]["attributes"],
        })
    return calls


def _sharding_occurrences(
        semantics: dict[str, dict[str, Any]],
        public: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for path, op in semantics.items():
        for name, value in op["attributes"].items():
            lowered = name.lower() + " " + value.lower()
            if "sharding" in lowered or "#sdy." in lowered:
                result.append({
                    "path": path,
                    "operator": op["name"],
                    "attribute": name,
                    "value": public[path]["attributes"][name],
                })
    return result


def _rng_occurrences(
        semantics: dict[str, dict[str, Any]],
        custom_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for path, op in semantics.items():
        if op["name"].startswith("stablehlo.") and "rng" in op["name"].lower():
            result.append({"path": path, "operator": op["name"]})
    for call in custom_calls:
        target = call["target"].lower()
        if any(marker in target for marker in ("rng", "random", "philox")):
            result.append({
                "path": call["path"],
                "operator": "stablehlo.custom_call",
                "target": call["target"],
            })
    return result


def snapshot_module(
    module: Any,
    *,
    source_path: Path,
    source_raw: bytes,
    requested_format: ArtifactFormat,
    parsed_format: str,
    bindings: Bindings,
    allowed_dialects: set[str],
    allowed_custom_calls: set[str] | None,
    allow_unregistered: bool,
    max_attribute_chars: int,
) -> Snapshot:
    try:
        verified = bool(module.operation.verify())
    except Exception as exc:
        raise AnalysisError(
            f"MLIR/StableHLO verifier rejected {source_path}: {exc}") from exc
    if not verified:
        raise AnalysisError(f"MLIR/StableHLO verifier rejected {source_path}")

    index = _index_module(module)
    semantics, public, order = _operation_records(
        index, bindings.ir, max_attribute_chars=max_attribute_chars)
    operation_counts = Counter(value["name"] for value in semantics.values())
    dialect_counts = Counter(
        name.split(".", 1)[0] if "." in name else "builtin"
        for name in operation_counts.elements())
    unexpected_dialects = sorted(set(dialect_counts) - allowed_dialects)
    custom_calls = _custom_calls(index, semantics, public, bindings.ir)
    custom_targets = sorted({call["target"] for call in custom_calls})
    unexpected_custom_calls = (
        sorted(set(custom_targets) - allowed_custom_calls)
        if allowed_custom_calls is not None else [])

    canonical_assembly = str(
        module.operation.get_asm(
            enable_debug_info=False,
            print_generic_op_form=True,
            assume_verified=True,
        ))
    canonical_hash = _sha256(canonical_assembly.encode("utf-8"))
    report = {
        "source": {
            "path": str(source_path.resolve()),
            "requested_format": requested_format,
            "parsed_format": parsed_format,
            "bytes": len(source_raw),
            "sha256": _sha256(source_raw),
        },
        "verification": {
            "verified": verified,
            "allow_unregistered_dialects": allow_unregistered,
            "semantic_verification": not allow_unregistered,
        },
        "canonical_assembly_sha256":
        canonical_hash,
        "operation_count":
        len(semantics),
        "stablehlo_operation_count":
        sum(count for name, count in operation_counts.items()
            if name.startswith("stablehlo.")),
        "operation_counts":
        dict(sorted(operation_counts.items())),
        "dialect_counts":
        dict(sorted(dialect_counts.items())),
        "functions":
        _function_abi(index, semantics, public, bindings.ir),
        "custom_calls":
        custom_calls,
        "dynamic_shapes":
        _dynamic_occurrences(semantics),
        "stablehlo_bounds":
        _type_occurrences(semantics, "stablehlo.bounds"),
        "sharding":
        _sharding_occurrences(semantics, public),
        "rng":
        _rng_occurrences(semantics, custom_calls),
        "policy": {
            "allowed_dialects":
            sorted(allowed_dialects),
            "observed_dialects":
            sorted(dialect_counts),
            "unexpected_dialects":
            unexpected_dialects,
            "custom_call_allowlist_enforced":
            allowed_custom_calls is not None,
            "allowed_custom_call_targets":
            sorted(allowed_custom_calls)
            if allowed_custom_calls is not None else None,
            "observed_custom_call_targets":
            custom_targets,
            "unexpected_custom_call_targets":
            unexpected_custom_calls,
            "passed":
            not unexpected_dialects and not unexpected_custom_calls,
        },
        "operations": [public[path] for path in order],
    }
    return Snapshot(report, semantics, public, order, canonical_hash)


def _record_differences(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    left_public: dict[str, Any] | None,
    right_public: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if left is None:
        return [{"field": "operation", "left": None, "right": right_public}]
    if right is None:
        return [{"field": "operation", "left": left_public, "right": None}]

    differences = []
    for field in ("name", "operands", "results", "regions", "successors"):
        if left[field] != right[field]:
            differences.append({
                "field": field,
                "left": left[field],
                "right": right[field]
            })

    left_attrs = left["attributes"]
    right_attrs = right["attributes"]
    for name in sorted(set(left_attrs) | set(right_attrs)):
        if left_attrs.get(name) == right_attrs.get(name):
            continue
        differences.append({
            "field":
            f"attribute:{name}",
            "left": (left_public or {}).get("attributes", {}).get(name),
            "right": (right_public or {}).get("attributes", {}).get(name),
        })
    return differences


def compare_snapshots(left: Snapshot, right: Snapshot) -> dict[str, Any]:
    order = list(left.order)
    order.extend(path for path in right.order if path not in left.semantics)
    calibrations = []
    all_differences = []
    by_operator: dict[str, Counter[str]] = {}

    for path in order:
        left_record = left.semantics.get(path)
        right_record = right.semantics.get(path)
        left_public = left.public_operations.get(path)
        right_public = right.public_operations.get(path)
        matches = left_record == right_record
        left_name = left_record["name"] if left_record else None
        right_name = right_record["name"] if right_record else None
        operator = left_name if left_name == right_name else f"{left_name} -> {right_name}"
        differences = ([] if matches else _record_differences(
            left_record, right_record, left_public, right_public))
        if not matches:
            all_differences.append({
                "path": path,
                "operator": operator,
                "differences": differences
            })

        is_stablehlo = bool(
            (left_name and left_name.startswith("stablehlo."))
            or (right_name and right_name.startswith("stablehlo.")))
        if not is_stablehlo:
            continue
        status = "match" if matches else "mismatch"
        by_operator.setdefault(operator, Counter())[status] += 1
        calibrations.append({
            "path":
            path,
            "operator":
            operator,
            "status":
            status,
            "binary_semantic_sha256":
            left_public.get("semantic_sha256") if left_public else None,
            "text_semantic_sha256":
            right_public.get("semantic_sha256") if right_public else None,
            "differences":
            differences,
        })

    canonical_match = (
        left.canonical_assembly_sha256 == right.canonical_assembly_sha256)
    if not canonical_match and not all_differences:
        all_differences.append({
            "path":
            "module",
            "operator":
            "builtin.module",
            "differences": [{
                "field": "canonical_assembly_sha256",
                "left": left.canonical_assembly_sha256,
                "right": right.canonical_assembly_sha256,
            }],
        })

    matched = sum(item["status"] == "match" for item in calibrations)
    mismatched = len(calibrations) - matched
    return {
        "equivalent": not all_differences and canonical_match,
        "canonical_assembly_match": canonical_match,
        "binary_canonical_assembly_sha256": left.canonical_assembly_sha256,
        "text_canonical_assembly_sha256": right.canonical_assembly_sha256,
        "all_operation_occurrences": {
            "binary": len(left.order),
            "text": len(right.order),
            "mismatched": len(all_differences),
        },
        "stablehlo_operator_occurrences": {
            "total": len(calibrations),
            "matched": matched,
            "mismatched": mismatched,
        },
        "by_operator": {
            name: {
                "total": counts["match"] + counts["mismatch"],
                "matched": counts["match"],
                "mismatched": counts["mismatch"],
            }
            for name, counts in sorted(by_operator.items())
        },
        "operator_calibration": calibrations,
        "differences": all_differences,
    }


def _parser_report(bindings: Bindings) -> dict[str, Any]:
    return {
        "provider":
        bindings.provider,
        "distribution":
        bindings.distribution,
        "distribution_version":
        bindings.distribution_version,
        "stablehlo_api_version":
        _safe_call(bindings.stablehlo, "get_api_version"),
        "stablehlo_current_version":
        _safe_call(bindings.stablehlo, "get_current_version"),
        "stablehlo_minimum_version":
        _safe_call(bindings.stablehlo, "get_minimum_version"),
    }


def _analyze_one(
    path: Path,
    artifact_format: ArtifactFormat,
    *,
    context: Any,
    bindings: Bindings,
    args: argparse.Namespace,
) -> Snapshot:
    module, parsed_format, raw = parse_artifact(path,
                                                artifact_format,
                                                context=context,
                                                bindings=bindings)
    return snapshot_module(
        module,
        source_path=path,
        source_raw=raw,
        requested_format=artifact_format,
        parsed_format=parsed_format,
        bindings=bindings,
        allowed_dialects=set(args.allowed_dialects),
        allowed_custom_calls=set(args.allow_custom_call)
        if args.allow_custom_call is not None else None,
        allow_unregistered=args.allow_unregistered_dialects,
        max_attribute_chars=args.max_attribute_chars,
    )


def analyze_file(args: argparse.Namespace,
                 bindings: Bindings) -> dict[str, Any]:
    path = Path(args.artifact)
    context = create_context(
        bindings, allow_unregistered=args.allow_unregistered_dialects)
    with context:
        snapshot = _analyze_one(
            path,
            args.format,
            context=context,
            bindings=bindings,
            args=args,
        )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parser": _parser_report(bindings),
        "artifact": snapshot.report,
        "passed": snapshot.report["policy"]["passed"],
    }


def compare_files(args: argparse.Namespace,
                  bindings: Bindings) -> dict[str, Any]:
    binary_path = Path(args.binary)
    text_path = Path(args.text)
    context = create_context(
        bindings, allow_unregistered=args.allow_unregistered_dialects)
    with context:
        binary = _analyze_one(
            binary_path,
            args.binary_format,
            context=context,
            bindings=bindings,
            args=args,
        )
        text = _analyze_one(
            text_path,
            "text",
            context=context,
            bindings=bindings,
            args=args,
        )
        comparison = compare_snapshots(binary, text)
    passed = (binary.report["policy"]["passed"]
              and text.report["policy"]["passed"] and comparison["equivalent"])
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parser": _parser_report(bindings),
        "binary": binary.report,
        "text": text.report,
        "comparison": comparison,
        "passed": passed,
    }


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if str(path) == "-":
        sys.stdout.write(rendered)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")


def _validate_dump_manifest(binary: Path, text: Path, *,
                            allow_missing: bool) -> dict[str, Any]:
    manifest_path = binary.parent / "manifest.json"
    if not manifest_path.is_file():
        if allow_missing:
            return {"present": False, "verified": False}
        raise AnalysisError(f"dump manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AnalysisError(
            f"invalid dump manifest {manifest_path}: {exc}") from exc

    artifacts = manifest.get("artifacts", {})
    expected_binary = artifacts.get("portable", {}).get("sha256")
    expected_text = artifacts.get("text", {}).get("sha256")
    actual_binary = _sha256(binary.read_bytes())
    actual_text = _sha256(text.read_bytes())
    mismatches = []
    if expected_binary != actual_binary:
        mismatches.append({
            "artifact": "portable",
            "expected_sha256": expected_binary,
            "actual_sha256": actual_binary,
        })
    if expected_text != actual_text:
        mismatches.append({
            "artifact": "text",
            "expected_sha256": expected_text,
            "actual_sha256": actual_text,
        })
    if mismatches:
        raise AnalysisError(
            f"dump manifest hash mismatch for {binary.parent}: {mismatches}")
    return {
        "present": True,
        "verified": True,
        "path": str(manifest_path),
        "schema": manifest.get("schema"),
        "artifact_id": manifest.get("artifact_id"),
    }


def compare_directory(args: argparse.Namespace,
                      bindings: Bindings) -> dict[str, Any]:
    root = Path(args.dump_root).resolve()
    binaries = sorted(root.rglob("*.stablehlo.mlirbc"))
    if not binaries:
        raise AnalysisError(
            f"no *.stablehlo.mlirbc artifacts found under {root}")

    reports_root = Path(args.reports_dir).resolve()
    graphs = []
    aggregate = Counter()
    passed = True
    for binary in binaries:
        text = binary.with_suffix(".mlir")
        relative_binary = binary.relative_to(root)
        if not text.is_file():
            graphs.append({
                "binary": str(relative_binary),
                "text": str(text.relative_to(root)),
                "passed": False,
                "error": "text sidecar is missing",
            })
            passed = False
            continue

        pair_args = argparse.Namespace(**vars(args))
        pair_args.binary = str(binary)
        pair_args.text = str(text)
        pair_args.binary_format = "portable"
        try:
            manifest = _validate_dump_manifest(
                binary,
                text,
                allow_missing=args.allow_missing_manifest,
            )
            report = compare_files(pair_args, bindings)
        except (AnalysisError, OSError, ValueError) as exc:
            graphs.append({
                "binary": str(relative_binary),
                "text": str(text.relative_to(root)),
                "passed": False,
                "error": str(exc),
            })
            passed = False
            continue
        report_path = reports_root / relative_binary.parent / "analysis.json"
        _write_json(report_path, report)
        comparison = report["comparison"]
        for operator, counts in comparison["by_operator"].items():
            aggregate[(operator, "total")] += counts["total"]
            aggregate[(operator, "matched")] += counts["matched"]
            aggregate[(operator, "mismatched")] += counts["mismatched"]
        graph_passed = bool(report["passed"])
        passed = passed and graph_passed
        graphs.append({
            "binary":
            str(relative_binary),
            "text":
            str(text.relative_to(root)),
            "report":
            str(report_path),
            "manifest":
            manifest,
            "passed":
            graph_passed,
            "binary_sha256":
            report["binary"]["source"]["sha256"],
            "text_sha256":
            report["text"]["source"]["sha256"],
            "operation_count":
            report["binary"]["operation_count"],
            "stablehlo_operator_occurrences":
            comparison["stablehlo_operator_occurrences"],
            "mismatch_paths":
            [difference["path"] for difference in comparison["differences"]],
        })

    operators = sorted({operator for operator, _ in aggregate})
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parser": _parser_report(bindings),
        "dump_root": str(root),
        "reports_dir": str(reports_root),
        "graph_count": len(graphs),
        "passed_graph_count": sum(graph["passed"] for graph in graphs),
        "failed_graph_count": sum(not graph["passed"] for graph in graphs),
        "by_operator": {
            operator: {
                "total": aggregate[(operator, "total")],
                "matched": aggregate[(operator, "matched")],
                "mismatched": aggregate[(operator, "mismatched")],
            }
            for operator in operators
        },
        "graphs": graphs,
        "passed": passed,
    }


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allowed-dialects",
        default=",".join(DEFAULT_ALLOWED_DIALECTS),
        help="comma-separated operation dialect allowlist",
    )
    parser.add_argument(
        "--allow-custom-call",
        action="append",
        default=None,
        metavar="TARGET",
        help=
        "allowed stablehlo.custom_call target; repeat to enforce an allowlist",
    )
    parser.add_argument(
        "--custom-call-allowlist",
        metavar="JSON_FILE",
        help="JSON array of accepted custom-call target strings",
    )
    parser.add_argument(
        "--allow-unregistered-dialects",
        action="store_true",
        help=
        "inventory-only escape hatch; disables the full semantic-verification claim",
    )
    parser.add_argument(
        "--max-attribute-chars",
        type=int,
        default=4096,
        help=
        "maximum inline characters per attribute before hash+preview elision",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze one artifact")
    analyze.add_argument("artifact")
    analyze.add_argument(
        "--format",
        choices=("auto", "portable", "bytecode", "text"),
        default="auto",
    )
    analyze.add_argument("-o", "--output", default="-")
    _add_policy_arguments(analyze)

    compare = subparsers.add_parser(
        "compare", help="calibrate a binary artifact against its text sidecar")
    compare.add_argument("binary")
    compare.add_argument("text")
    compare.add_argument(
        "--binary-format",
        choices=("auto", "portable", "bytecode"),
        default="portable",
    )
    compare.add_argument("-o", "--output", default="-")
    _add_policy_arguments(compare)

    compare_dir = subparsers.add_parser(
        "compare-dir", help="calibrate every paired dump below a directory")
    compare_dir.add_argument("dump_root")
    compare_dir.add_argument(
        "--reports-dir",
        required=True,
        help="directory for full per-graph, per-operator reports",
    )
    compare_dir.add_argument("-o", "--output", required=True)
    compare_dir.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help=
        "permit generic artifact pairs which were not produced by the dump hook",
    )
    _add_policy_arguments(compare_dir)
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    args.allowed_dialects = tuple(value.strip()
                                  for value in args.allowed_dialects.split(",")
                                  if value.strip())
    if not args.allowed_dialects:
        raise AnalysisError("--allowed-dialects must not be empty")
    if args.max_attribute_chars < 64:
        raise AnalysisError("--max-attribute-chars must be at least 64")
    if args.custom_call_allowlist:
        allowlist_path = Path(args.custom_call_allowlist)
        try:
            loaded = json.loads(allowlist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(
                f"invalid custom-call allowlist {allowlist_path}: {exc}"
            ) from exc
        if (not isinstance(loaded, list)
                or any(not isinstance(value, str) for value in loaded)):
            raise AnalysisError(
                "--custom-call-allowlist must contain a JSON array of strings")
        args.allow_custom_call = sorted(
            set(args.allow_custom_call or ()) | set(loaded))


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        _normalize_args(args)
        bindings = load_bindings()
        if args.command == "analyze":
            report = analyze_file(args, bindings)
        elif args.command == "compare":
            report = compare_files(args, bindings)
        else:
            report = compare_directory(args, bindings)
        _write_json(args.output, report)
        if not report["passed"]:
            print("StableHLO analysis failed policy or equivalence checks",
                  file=sys.stderr)
            return 2
        return 0
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"StableHLO analysis error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
