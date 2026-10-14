# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StableHLO export and save-load round-trip API."""

from __future__ import annotations

import copy
import ctypes
import json
import struct
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest  # pyrefly: ignore[missing-import]
import torch
import torch.fx as fx
from vllm.config import VllmConfig
from vllm.config.utils import Range

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
from vllm_torchtpu.export import stablehlo_export


@pytest.fixture(autouse=True)
def _forbid_default_tpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock default TPU backend on CPU so normal compile continuation succeeds."""
    mock_runnable = MagicMock()
    mock_runnable.return_value = torch.ones((16, 64), dtype=torch.bfloat16)
    mock_backend = MagicMock(return_value=mock_runnable)
    monkeypatch.setattr(tpu_compiler, "_tpu_backend", mock_backend)


def _make_sample_graph_and_inputs(
    batch_size: int = 16,
    hidden_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[fx.GraphModule, list[torch.Tensor]]:
    class SimpleModule(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + hidden_states

    graph = fx.symbolic_trace(SimpleModule())
    example_input = torch.empty((batch_size, hidden_size), dtype=dtype)
    for node in graph.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            node.meta["example_value"] = example_input
            node.meta["sharding"] = "PartitionSpec(None, 'model')"
    return graph, [example_input]


def _write_test_safetensors(tensors: dict[str, torch.Tensor], out_path: Path) -> None:
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        cpu_t = tensor.detach().cpu().contiguous()
        nbytes = cpu_t.numel() * cpu_t.element_size()
        raw = ctypes.string_at(cpu_t.data_ptr(), nbytes) if nbytes > 0 else b""
        end = offset + len(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": [int(d) for d in cpu_t.shape],
            "data_offsets": [offset, end],
        }
        payloads.append(raw)
        offset = end
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    if pad:
        hdr_bytes += b" " * pad
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for chunk in payloads:
            f.write(chunk)


def test_env_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", "/tmp/mock_export_hlo")
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", "/tmp/mock_hlo_export")

    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compiler = TpuCompilerAdaptor()
    with pytest.raises(ValueError, match="mutually exclusive"):
        stablehlo_export.maybe_export_or_load_stablehlo(
            compiler, graph, inputs, Range(start=16, end=16), "subgraph_0"
        )


def test_deterministic_key_generation() -> None:
    assert (
        stablehlo_export._compute_fn_key(Range(start=16, end=16), 0)
        == "forward_bucket16"
    )
    assert (
        stablehlo_export._compute_fn_key(Range(start=32, end=32), 0)
        == "forward_bucket32"
    )
    assert (
        stablehlo_export._compute_fn_key(Range(start=16, end=16), 1)
        == "forward_piece1_bucket16"
    )


def test_profile_run_preserves_behavior_and_skips_export_and_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path / "hlo"))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    profile_compiler = TpuCompilerAdaptor()
    profile_cache_dir = tmp_path / "cache" / "profile_cache" / "abcd1234"
    profile_compiler.initialize_cache(cache_dir=str(profile_cache_dir))

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = "module {}"
    mock_mlir_res.mlir_result_tensors = [inputs[0]]
    mock_mlir_res.is_noop = False

    with (
        patch(
            "torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res
        ) as mock_fx_to_mlir,
        patch.object(
            profile_compiler,
            "_run_backend",
            return_value=(lambda *a, **kw: inputs[0], None),
        ) as mock_run_backend,
    ):
        compiled_fn, _ = profile_compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn is not None
        mock_fx_to_mlir.assert_not_called()
        mock_run_backend.assert_called_once()
    assert not (tmp_path / "hlo").exists()

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path / "hlo"))
    with patch.object(
        profile_compiler,
        "_run_backend",
        return_value=(lambda *a, **kw: inputs[0], None),
    ) as mock_run_backend_load:
        compiled_fn_load, _ = profile_compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn_load is not None
        mock_run_backend_load.assert_called_once()


def test_export_stablehlo_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    dummy_mlir = (
        "module @exported_stablehlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )

    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = dummy_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch(
        "torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported
    ) as mock_fx_to_mlir:
        compiled_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )

    assert mock_fx_to_mlir.call_count == 1
    assert compiled_fn is not None
    assert handle is None

    expected_fn_key = "forward_bucket16"
    mlir_path = tmp_path / f"{expected_fn_key}.mlir"
    metadata_path = tmp_path / "metadata.json"
    assert mlir_path.is_file()
    assert mlir_path.read_text() == dummy_mlir
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())
    assert expected_fn_key in metadata
    entry = metadata[expected_fn_key]
    assert entry["file_path"] == f"{expected_fn_key}.mlir"
    assert entry["subgraph_index"] == 0
    assert entry["compile_size"] == 16
    assert len(entry["inputs"]) == 1
    assert entry["inputs"][0] == {
        "index": 0,
        "name": "hidden_states",
        "shape": [16, 64],
        "dtype": "bfloat16",
        "sharding": "PartitionSpec(None, 'model')",
        "buffer_donor": False,
        "is_weight": False,
        "weight_name": "",
    }
    assert len(entry["outputs"]) == 1
    assert entry["outputs"][0]["shape"] == [16, 64]
    assert entry["outputs"][0]["dtype"] == "bfloat16"


def test_load_stablehlo_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    saved_mlir = (
        "module @custom_stablehlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    (tmp_path / f"{fn_key}.mlir").write_text(saved_mlir)
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_0",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")
    expected_output = torch.ones((16, 64), dtype=torch.bfloat16)

    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[expected_output],
        ) as mock_execute,
    ):
        compiled_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn is not None
        assert handle is None
        mock_parse.assert_called_once_with(saved_mlir)
        mock_compile_mlir.assert_called_once_with(mock_mlir_module)

        result = compiled_fn(inputs[0])
        assert mock_execute.call_count == 1
        assert torch.equal(result, expected_output)


def test_load_stablehlo_signature_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    (tmp_path / f"{fn_key}.mlir").write_text("module {}")
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [32, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
            "outputs": [],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(ValueError, match="index 0"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_load_stablehlo_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    (tmp_path / "metadata.json").write_text(json.dumps({}))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing from metadata.json"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_export_and_load_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    exported_mlir_text = (
        "module @initial_export {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = exported_mlir_text
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = "forward_bucket16"
    mlir_file = tmp_path / f"{fn_key}.mlir"
    assert mlir_file.is_file()

    customer_modified_mlir = (
        "module @customer_optimized {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mlir_file.write_text(customer_modified_mlir)

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")
    expected_output = torch.full((16, 64), 2.0, dtype=torch.bfloat16)

    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[expected_output],
        ),
    ):
        reloaded_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert reloaded_fn is not None
        mock_parse.assert_called_once_with(customer_modified_mlir)
        mock_compile_mlir.assert_called_once_with(mock_mlir_module)
        out = reloaded_fn(inputs[0])
        assert torch.equal(out, expected_output)


def test_export_stablehlo_explicit_dir_and_aot_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _WithWeight(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones((16, 64), dtype=torch.bfloat16), requires_grad=False
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.weight

    mod = _WithWeight()
    graph = fx.symbolic_trace(mod)
    inputs = [torch.ones((16, 64), dtype=torch.bfloat16)]
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            node.meta["example_value"] = inputs[0]
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_2"

    explicit_export_dir = tmp_path / "custom_export_dir"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(explicit_export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path / "unused_cache_dir"))

    exported_mlir_text = (
        "module @aot_exported {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.is_noop = False
    mock_exported.module = MagicMock(name="MlirModule")
    mock_exported.serialize_text.return_value = exported_mlir_text
    mock_exported.mlir_result_tensors = [inputs[0]]

    def _fake_run_backend(
        g: fx.GraphModule, ex_inputs: list[Any], *, serializable: bool = True
    ) -> tuple[Any, None]:
        from torch_tpu._internal import export as torch_tpu_export

        torch_tpu_export.fx_to_mlir(g, ex_inputs, build_mlir_module=False)
        return (lambda *a: (a[0] * 2,)), None

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported),
        patch.object(compiler, "_run_backend", side_effect=_fake_run_backend),
    ):
        compiled_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert compiled_fn is not None
        assert torch.equal(compiled_fn(inputs[0]), inputs[0] * 2)

    metadata_file = explicit_export_dir / "metadata.json"
    assert metadata_file.is_file()
    metadata = json.loads(metadata_file.read_text())
    assert "forward_piece200_bucket16" in metadata
    fn_key = "forward_piece200_bucket16"
    mlir_file = explicit_export_dir / metadata[fn_key]["file_path"]
    assert mlir_file.read_text() == exported_mlir_text

    modified_mlir_text = (
        "module @aot_modified {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mlir_file.write_text(modified_mlir_text)

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(explicit_export_dir))

    mock_parsed_mod = MagicMock(name="ParsedModule")
    mock_pjrt_exe = MagicMock(name="CompiledExe")

    def _fake_orig_fx_to_mlir(
        g: fx.GraphModule, ex_inputs: list[Any], **kwargs: Any
    ) -> Any:
        from torch_tpu._internal.compile import tpu_torch_compile

        res = tpu_torch_compile.traverse_and_compile(argument_layouts=[[1, 0]])
        assert res.module is mock_parsed_mod
        assert res.executable is mock_pjrt_exe
        return mock_exported

    with (
        patch(
            "torch_tpu._internal.export.fx_to_mlir",
            side_effect=_fake_orig_fx_to_mlir,
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_parsed_mod,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch.object(compiler, "_run_backend", side_effect=_fake_run_backend),
    ):
        reloaded_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert reloaded_fn is not None
        mock_parse.assert_called_once_with(modified_mlir_text)
        mock_compile_mlir.assert_called_once_with(mock_parsed_mod, False, [[1, 0]])
        assert torch.equal(reloaded_fn(inputs[0]), inputs[0] * 2)


def test_load_stablehlo_exact_key_lookup_and_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(16, 64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"
    expected_key = "forward_bucket16"

    metadata_other = {
        "forward_bucket32": {
            "file_path": "forward_bucket32.mlir",
            "subgraph_index": 0,
            "compile_size": 32,
            "inputs": [],
            "outputs": [],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata_other, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match=expected_key):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    metadata_exact = {
        expected_key: {
            "file_path": f"{expected_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [32, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                }
            ],
            "outputs": [],
        }
    }
    (tmp_path / f"{expected_key}.mlir").write_text("module {}")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata_exact, indent=2))

    with pytest.raises(ValueError, match="Input signature mismatch at index 0"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_export_and_load_integrate_with_shape_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_torchtpu.compilation import shape_variants

    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    dummy_mlir = (
        "module @shape_variant_test {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = dummy_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    with (
        shape_variants.warmup(),
        patch(
            "torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported
        ) as mock_fx_to_mlir,
    ):
        fn_trace1, _ = compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
        fn_trace2, _ = compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
        fn_load_trace3 = compiler.load(
            (cache_key, str(tmp_path / cache_key), True),
            graph,
            inputs,
            0,
            compile_range,
        )
        assert mock_fx_to_mlir.call_count == 1
        assert fn_trace2 is fn_trace1
        assert fn_load_trace3 is fn_trace1

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")

    with (
        shape_variants.warmup(),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
    ):
        loaded_trace1, _ = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        loaded_trace2, _ = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        loaded_trace3 = compiler.load(
            (cache_key, str(tmp_path / cache_key), True),
            graph,
            inputs,
            0,
            compile_range,
        )
        assert mock_parse.call_count == 1
        assert mock_compile_mlir.call_count == 1
        assert loaded_trace2 is loaded_trace1
        assert loaded_trace3 is loaded_trace1


def test_load_stablehlo_mlir_main_signature_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    mlir_path = tmp_path / f"{fn_key}.mlir"
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_0",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x128xbf16> {\n"
        "    return %arg0 : tensor<16x128xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xf32> {\n"
        "    return %arg0 : tensor<16x64xf32>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>)"
        " -> (tensor<16x64xbf16>, tensor<16x64xbf16>) {\n"
        "    return %arg0, %arg0 : tensor<16x64xbf16>, tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output count mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<32x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main input signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_export_and_load_isolate_compilation_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            num_gpu_blocks=1024,
            num_gpu_blocks_override=None,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=16,
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )

    mlir_file = tmp_path / "subgraph_idx0_bucket16_abcdef123456.mlir"
    mlir_file.write_text("module @v1 {\n  func.func @main() { return }\n}\n")

    with patch(
        "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
        return_value=[],
    ):
        monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
        monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
        base_hash = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))

        monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
        export_hash = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))
        assert export_hash != base_hash

        monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
        monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))
        load_hash_v1 = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))
        assert load_hash_v1 != base_hash
        assert load_hash_v1 != export_hash

        mlir_file.write_text("module @v2 {\n  func.func @main() { return }\n}\n")
        load_hash_v2 = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))
        assert load_hash_v2 != load_hash_v1


def test_load_stablehlo_disables_async_compile_and_preserves_tuple_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from torch_tpu._internal.compile.compiler import StaticCompiler

    class _TupleOutputModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x + x, x * 2

    tuple_graph = fx.symbolic_trace(_TupleOutputModule())
    inp = torch.ones((16, 64), dtype=torch.bfloat16)
    for node in tuple_graph.graph.nodes:
        if node.op == "placeholder":
            node.meta["example_value"] = inp
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    (tmp_path / f"{fn_key}.mlir").write_text(
        "module @tuple_out {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>)"
        " -> (tensor<16x64xbf16>, tensor<16x64xbf16>) {\n"
        "    return %arg0, %arg0 : tensor<16x64xbf16>, tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "x",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                }
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_0",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                },
                {
                    "index": 1,
                    "name": "output_1",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                },
            ],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    out0 = torch.full((16, 64), 2.0, dtype=torch.bfloat16)
    out1 = torch.full((16, 64), 3.0, dtype=torch.bfloat16)
    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[out0, out1],
        ),
    ):
        loaded_fn, _ = compiler.compile(
            tuple_graph, [inp], {}, compile_range, key=cache_key
        )
        assert loaded_fn is not None
        res = loaded_fn(inp)
        assert isinstance(res, tuple)
        assert len(res) == 2

    class _WithAttr(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(inp, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w

    aot_graph = fx.symbolic_trace(_WithAttr())
    monkeypatch.setattr(StaticCompiler, "_async_compile", True, raising=False)
    observed_async_flags: list[bool] = []

    def _inspect_async_backend(
        g: fx.GraphModule, ex_inputs: list[Any], *args: Any, **kwargs: Any
    ) -> tuple[Any, None]:
        observed_async_flags.append(
            bool(getattr(StaticCompiler, "_async_compile", False))
        )
        return (lambda *a: (a[0],)), None

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path / "aot_out"))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
    with patch.object(compiler, "_run_backend", side_effect=_inspect_async_backend):
        compiler.compile(aot_graph, [inp], {}, compile_range, key=cache_key)

    assert observed_async_flags == [False]
    assert getattr(StaticCompiler, "_async_compile", None) is True


def test_rfc_contract_safeguards_buffer_donor_hardware_and_custom_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    donor_mlir = (
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16> {jax.buffer_donor = true})"
        " -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = donor_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = "forward_bucket16"
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entry = metadata[fn_key]
    assert entry["inputs"][0]["buffer_donor"] is True
    assert entry["target_hardware"]["chip_type"] == "v5p-8"
    assert "libtpu" in entry["target_hardware"]

    mlir_path = tmp_path / f"{fn_key}.mlir"
    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mlir_path.write_text(
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main buffer_donor mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(donor_mlir)

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    with pytest.raises(ValueError, match="Target hardware mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")

    drifted_metadata = copy.deepcopy(metadata)
    drifted_metadata[fn_key]["target_hardware"]["torch_tpu"] = "0.0.0-different"
    drifted_metadata[fn_key]["target_hardware"]["jaxlib"] = "0.0.0-different"
    drifted_metadata[fn_key]["target_hardware"]["libtpu"] = "0.0.0-different"
    metadata_path.write_text(json.dumps(drifted_metadata, indent=2))
    with (
        patch.object(stablehlo_export.logger, "warning") as mock_warn,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=MagicMock(),
        ),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
        assert mock_warn.call_count >= 1
    metadata_path.write_text(json.dumps(metadata, indent=2))

    mlir_path.write_text(
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16> {jax.buffer_donor = true})"
        " -> tensor<16x64xbf16> {\n"
        "    %0 = stablehlo.custom_call @__xla_custom_call_unvetted(%arg0)"
        " : (tensor<16x64xbf16>) -> tensor<16x64xbf16>\n"
        "    return %0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="Disallowed stablehlo.custom_call target"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_strict_main_signature_and_single_read_invariants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    valid_mlir = (
        "module @valid_hlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16, #stablehlo.bounds<?, ?>>)"
        " -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = valid_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{corrupt_json")
    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported),
        pytest.raises(ValueError, match="Corrupt metadata.json"),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    metadata_path.unlink()

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = "forward_bucket16"
    mlir_path = tmp_path / f"{fn_key}.mlir"

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mlir_path.write_text("module @empty_main {\n  func.func @main() { return }\n}\n")
    with pytest.raises(ValueError, match="MLIR @main input count mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @missing_main {\n"
        "  func.func @other(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main function signature missing"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(valid_mlir)
    orig_read_text = Path.read_text
    mlir_reads: list[Path] = []

    def _spy_read_text(self_path: Path, *a: Any, **kw: Any) -> str:
        if self_path == mlir_path:
            mlir_reads.append(self_path)
        return orig_read_text(self_path, *a, **kw)

    with (
        patch.object(Path, "read_text", autospec=True, side_effect=_spy_read_text),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=MagicMock(),
        ),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    assert len(mlir_reads) == 1


def test_export_early_exit_skips_backend_compile_and_exits_after_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_torchtpu import envs

    assert "VLLM_TORCHTPU_EXPORT_EARLY_EXIT" not in envs.environment_variables
    assert "VLLM_TORCHTPU_EXPORT_HLO_DIR" in envs.environment_variables

    graph16, inputs16 = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    dummy_mlir_16 = (
        "module @early_exit_16 {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exp = MagicMock()
    mock_exp.serialize_text.return_value = dummy_mlir_16
    mock_exp.mlir_result_tensors = [inputs16[0]]
    mock_exp.is_noop = False
    mock_exp.module = MagicMock()

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exp),
        patch.object(compiler, "_run_backend") as mock_run_backend,
    ):
        fn16, handle16 = compiler.compile(
            graph16,
            inputs16,
            {},
            Range(start=16, end=16),
            key="artifact_compile_range_16_16_subgraph_0",
        )
        assert handle16 is None
        assert fn16 is not None
        out16 = fn16(inputs16[0])
        assert out16.shape == inputs16[0].shape
        assert out16.dtype == inputs16[0].dtype
        assert mock_run_backend.call_count == 0

    assert (tmp_path / "forward_bucket16.mlir").is_file()


def test_extract_input_specs_filters_non_tensor_symint_placeholders() -> None:
    class DynamicModule(torch.nn.Module):
        def forward(self, batch_size: int, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + hidden_states

    graph = fx.symbolic_trace(DynamicModule())
    tensor_input = torch.empty((16, 64), dtype=torch.bfloat16)

    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    assert len(placeholders) == 2
    placeholders[0].meta["val"] = 16
    placeholders[1].meta["val"] = tensor_input
    placeholders[1].meta["sharding"] = "PartitionSpec(None, 'model')"

    specs = stablehlo_export._extract_input_specs(graph, [16, tensor_input])
    assert len(specs) == 1
    assert specs[0]["index"] == 0
    assert specs[0]["name"] == "hidden_states"
    assert specs[0]["shape"] == [16, 64]
    assert specs[0]["dtype"] == "bfloat16"


def test_extract_input_specs_annotates_is_weight_and_weight_name() -> None:
    class ModuleWithLiftedWeights(torch.nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, p_q_proj_weight: torch.Tensor
        ) -> torch.Tensor:
            return torch.matmul(hidden_states, p_q_proj_weight)

    model = ModuleWithLiftedWeights()
    graph = fx.symbolic_trace(model)
    tensor_input = torch.empty((16, 64), dtype=torch.bfloat16)
    weight_tensor = torch.nn.Parameter(torch.empty((64, 64), dtype=torch.bfloat16))

    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["val"] = tensor_input
    placeholders[1].meta["val"] = weight_tensor
    placeholders[1].meta["param_name"] = "model.layers.0.self_attn.q_proj.weight"

    specs = stablehlo_export._extract_input_specs(graph, [tensor_input, weight_tensor])
    assert len(specs) == 2
    assert specs[0]["is_weight"] is False
    assert specs[1]["is_weight"] is True
    assert specs[1]["weight_name"] == "model.layers.0.self_attn.q_proj.weight"


def test_extract_input_specs_annotates_is_weight_with_aot_desc() -> None:
    class ModuleWithAot(torch.nn.Module):
        def forward(self, arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
            return arg0 + arg1

    model = ModuleWithAot()
    graph = fx.symbolic_trace(model)
    tensor_input = torch.empty((16, 64), dtype=torch.bfloat16)
    weight_tensor = torch.empty((64, 64), dtype=torch.bfloat16)

    class DummyDesc:
        def __init__(self, target: str, is_param: bool):
            self.target = target
            self.is_param = is_param
            self.is_buffer = False

    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["val"] = tensor_input
    placeholders[0].meta["desc"] = DummyDesc("", False)
    placeholders[1].meta["val"] = weight_tensor
    placeholders[1].meta["desc"] = DummyDesc("model.embed_tokens.weight", True)

    specs = stablehlo_export._extract_input_specs(graph, [tensor_input, weight_tensor])
    assert len(specs) == 2
    assert specs[0]["is_weight"] is False
    assert specs[1]["is_weight"] is True
    assert specs[1]["weight_name"] == "model.embed_tokens.weight"


def test_export_and_load_serving_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _WeightedGraph(torch.nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, p_embed_weight: torch.Tensor
        ) -> torch.Tensor:
            return hidden_states + p_embed_weight

    graph = fx.symbolic_trace(_WeightedGraph())
    inp_x = torch.empty((16, 64), dtype=torch.bfloat16)
    inp_w = torch.nn.Parameter(torch.empty((16, 64), dtype=torch.bfloat16))
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["example_value"] = inp_x
    placeholders[0].meta["sharding"] = "PartitionSpec(None, 'model')"
    placeholders[1].meta["example_value"] = inp_w
    placeholders[1].meta["param_name"] = "model.embed_tokens.weight"
    for node in graph.graph.nodes:
        if node.op == "output":
            node.meta["example_value"] = inp_x

    inputs = [inp_x, inp_w]
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    exported_mlir = (
        "module @mlir_test {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16> {jax.buffer_donor = true},"
        " %arg1: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = exported_mlir
    mock_exported.mlir_result_tensors = [inp_x]
    mock_exported.is_noop = False

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    assert (tmp_path / "forward_bucket16.mlir").is_file()
    assert (tmp_path / "metadata.json").is_file()
    serving_ckpt_dir = tmp_path / "serving_checkpoint"
    assert (serving_ckpt_dir / "weight.json").is_file()
    wj_data = json.loads((serving_ckpt_dir / "weight.json").read_text())
    assert "model.embed_tokens.weight" in wj_data
    assert "sources" in wj_data["model.embed_tokens.weight"]

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")
    expected_out = torch.full((16, 64), 3.0, dtype=torch.bfloat16)

    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[expected_out],
        ),
    ):
        reloaded_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert reloaded_fn is not None
        mock_parse.assert_called_once_with(exported_mlir)
        mock_compile_mlir.assert_called_once_with(mock_mlir_module)
        res = reloaded_fn(inp_x, inp_w)
        assert torch.equal(res, expected_out)


def test_export_sampling_subgraphs_e2e(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / "export_subgraphs"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockModel(torch.nn.Module):
        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                hidden_states,
                torch.ones((hidden_states.shape[-1], 32), dtype=hidden_states.dtype),
            )

    class MockRunner:
        def __init__(self) -> None:
            self.model = MockModel()
            self._greedy_head_owner = MagicMock()
            self.num_reqs_paddings = [1, 2]
            self.model_config = MagicMock()
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = (
        "module @subgraph { func.func @main(%arg0: tensor<1x16xbf16>) "
        "-> tensor<1x32xbf16> { return %arg0 : tensor<1x32xbf16> } }"
    )
    mock_mlir_res.is_noop = False
    mock_mlir_res.module = MagicMock()

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res),
        patch(
            "vllm_torchtpu.runner.greedy_top_tokens.top_tokens",
            return_value=torch.zeros((1, 1), dtype=torch.int64),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(MockRunner())
    assert exc_info.value.code == 0

    assert (export_dir / "compute_logits_req1.mlir").exists()
    assert (export_dir / "compute_logits_req2.mlir").exists()
    assert (export_dir / "sample_top_tokens_req1.mlir").exists()
    assert (export_dir / "sample_top_tokens_req2.mlir").exists()
    assert (export_dir / "metadata.json").exists()


def test_export_sampling_subgraphs_bundles_checkpoint_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt_dir = tmp_path / "hf_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    (ckpt_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrained"}')
    (ckpt_dir / "config.json").write_text(
        '{"model_type": "qwen3", "rope_theta": 1000000.0}'
    )
    _write_test_safetensors(
        {"model.embed_tokens.weight": torch.ones((32, 16), dtype=torch.bfloat16)},
        ckpt_dir / "model.safetensors",
    )

    export_dir = tmp_path / "export_bundled"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    (export_dir / "serving_checkpoint").mkdir(parents=True, exist_ok=True)
    seed_weights = {
        "model.embed_tokens.weight": {
            "index": 1,
            "shape": [32, 16],
            "dtype": "bfloat16",
            "sharding": None,
            "sources": [
                {
                    "path": "",
                    "source_tensor": "model.embed_tokens.weight",
                    "source_shape": [32, 16],
                }
            ],
        },
        "rotary_emb_inv_freq": {
            "index": 7,
            "shape": [64, 16],
            "dtype": "bfloat16",
            "sharding": None,
            "is_internal_constant": True,
            "sources": [
                {
                    "path": "",
                    "source_tensor": "rotary_emb_inv_freq",
                    "source_shape": [64, 16],
                }
            ],
        },
    }
    (export_dir / "weight.json").write_text(json.dumps(seed_weights))
    (export_dir / "serving_checkpoint" / "weight.json").write_text(
        json.dumps(seed_weights)
    )

    class MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.lm_head(hidden_states)

    class MockRunner:
        def __init__(self) -> None:
            self.model = MockModel()
            self._greedy_head_owner = MagicMock()
            self.num_reqs_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(ckpt_dir)
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = (
        "module @subgraph { func.func @main(%arg0: tensor<1x16xbf16>, %arg1: tensor<32x16xbf16>) "
        "-> tensor<1x32xbf16> { return %arg0 : tensor<1x32xbf16> } }"
    )
    mock_mlir_res.is_noop = False
    mock_mlir_res.module = MagicMock()

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res),
        patch(
            "vllm_torchtpu.runner.greedy_top_tokens.top_tokens",
            return_value=torch.zeros((1, 1), dtype=torch.int64),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(MockRunner())
    assert exc_info.value.code == 0

    constants_st = export_dir / "serving_checkpoint" / "constants.safetensors"
    assert constants_st.is_file()
    wj = json.loads((export_dir / "serving_checkpoint" / "weight.json").read_text())
    assert wj["rotary_emb_inv_freq"]["sources"][0]["path"] == str(
        constants_st.resolve()
    )
    assert wj["model.embed_tokens.weight"]["sources"][0]["path"] == str(
        (ckpt_dir / "model.safetensors").resolve()
    )
    assert (
        wj["lm_head.weight"]["sources"][0]["source_tensor"]
        == "model.embed_tokens.weight"
    )


def test_export_sampling_subgraphs_gives_lm_head_its_own_weight_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `lm_head.weight` is arg 1 of compute_logits and `model.embed_tokens.weight`
    # is arg 1 of forward_bucket. weight.json `index` is the global HBM weight
    # slot, so the two tensors must not share it.
    export_dir = tmp_path / "export_lm_head_index"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
    (export_dir / "serving_checkpoint").mkdir(parents=True, exist_ok=True)
    seed_weights = {
        name: {
            "index": idx,
            "shape": [32, 16],
            "dtype": "bfloat16",
            "sharding": None,
            "sources": [{"path": "", "source_tensor": name, "source_shape": [32, 16]}],
        }
        for name, idx in (("model.embed_tokens.weight", 1), ("model.norm.weight", 9))
    }
    for wj_path in (
        export_dir / "weight.json",
        export_dir / "serving_checkpoint" / "weight.json",
    ):
        wj_path.write_text(json.dumps(seed_weights))

    class MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.lm_head(hidden_states)

    class MockRunner:
        def __init__(self) -> None:
            self.model = MockModel()
            self._greedy_head_owner = MagicMock()
            self.num_reqs_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(tmp_path / "no_such_ckpt")
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = (
        "module @subgraph { func.func @main(%arg0: tensor<1x16xbf16>, "
        "%arg1: tensor<32x16xbf16>) -> tensor<1x32xbf16> "
        "{ return %arg0 : tensor<1x32xbf16> } }"
    )
    mock_mlir_res.is_noop = False
    mock_mlir_res.module = MagicMock()

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res),
        patch(
            "vllm_torchtpu.runner.greedy_top_tokens.top_tokens",
            return_value=torch.zeros((1, 1), dtype=torch.int64),
        ),
        pytest.raises(SystemExit),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(MockRunner())

    wj = json.loads((export_dir / "serving_checkpoint" / "weight.json").read_text())
    lm_head_index = wj["lm_head.weight"]["index"]
    assert lm_head_index not in {1, 9}
    metadata = json.loads((export_dir / "metadata.json").read_text())
    for fn_key in ("compute_logits_req1", "sample_top_tokens_req1"):
        assert metadata[fn_key]["inputs"][1]["index"] == lm_head_index


@pytest.mark.parametrize(
    ("tp", "expected"),
    [
        (1, {"num_replicas": 1, "num_partitions": 1, "use_spmd_partitioning": False}),
        # torch_tpu compiles each TP rank's module with num_partitions=TP,
        # replica_count=1 and use_spmd_partitioning=true (XLA dump of a TP=4 run).
        (4, {"num_replicas": 1, "num_partitions": 4, "use_spmd_partitioning": True}),
    ],
)
def test_export_sampling_subgraphs_writes_compile_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tp: int, expected: dict
) -> None:
    export_dir = tmp_path / f"export_compile_options_tp{tp}"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockModel(torch.nn.Module):
        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class MockRunner:
        def __init__(self) -> None:
            self.model = MockModel()
            self._greedy_head_owner = MagicMock()
            self.num_reqs_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(tmp_path / "no_such_ckpt")
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.parallel_config = MagicMock()
            self.parallel_config.tensor_parallel_size = tp
            self.parallel_config.pipeline_parallel_size = 1
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = (
        "module @subgraph { func.func @main(%arg0: tensor<1x16xbf16>) "
        "-> tensor<1x16xbf16> { return %arg0 : tensor<1x16xbf16> } }"
    )
    mock_mlir_res.is_noop = False
    mock_mlir_res.module = MagicMock()

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res),
        patch(
            "vllm_torchtpu.runner.greedy_top_tokens.top_tokens",
            return_value=torch.zeros((1, 1), dtype=torch.int64),
        ),
        pytest.raises(SystemExit),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(MockRunner())

    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["compile_options"] == expected
    assert metadata["config"]["tensor_parallel_size"] == tp


def test_export_sampling_subgraphs_fails_when_metadata_cannot_be_finalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An invalid tensor_parallel_size must fail the export instead of leaving a
    # metadata.json without config/compile_options behind.
    export_dir = tmp_path / "export_bad_parallel_config"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockModel(torch.nn.Module):
        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class MockRunner:
        def __init__(self) -> None:
            self.model = MockModel()
            self._greedy_head_owner = MagicMock()
            self.num_reqs_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(tmp_path / "no_such_ckpt")
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.parallel_config = MagicMock()
            self.parallel_config.tensor_parallel_size = "four"
            self.parallel_config.pipeline_parallel_size = 1
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = (
        "module @subgraph { func.func @main(%arg0: tensor<1x16xbf16>) "
        "-> tensor<1x16xbf16> { return %arg0 : tensor<1x16xbf16> } }"
    )
    mock_mlir_res.is_noop = False
    mock_mlir_res.module = MagicMock()

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res),
        patch(
            "vllm_torchtpu.runner.greedy_top_tokens.top_tokens",
            return_value=torch.zeros((1, 1), dtype=torch.int64),
        ),
        pytest.raises(ValueError),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(MockRunner())


def test_export_early_exit_cleanly_handles_engine_startup_and_multiproc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / "export_mp"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "metadata.json").write_text("{}")

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockModel(torch.nn.Module):
        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class MockTPRunner:
        def __init__(self) -> None:
            self.model = MockModel()
            self._greedy_head_owner = MagicMock()
            self.num_reqs_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.parallel_config = MagicMock()
            self.parallel_config.world_size = 2
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = (
        "module @subgraph { func.func @main(%arg0: tensor<1x16xbf16>) "
        "-> tensor<1x32xbf16> { return %arg0 : tensor<1x32xbf16> } }"
    )
    mock_mlir_res.is_noop = False
    mock_mlir_res.module = MagicMock()

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res),
        patch(
            "vllm_torchtpu.runner.greedy_top_tokens.top_tokens",
            return_value=torch.zeros((1, 1), dtype=torch.int64),
        ),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(MockTPRunner())

    import vllm.v1.engine.utils as engine_utils

    stablehlo_export._install_warmup_early_exit_hook()

    def _fake_startup(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "Engine core initialization failed. See root cause above. "
            "Failed core proc(s): {'EngineCore_DP0': 0}"
        )

    with (
        patch.object(
            engine_utils,
            "_orig_wait_for_engine_startup",
            _fake_startup,
            create=True,
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        engine_utils.wait_for_engine_startup(None, [], None, False, None, None)
    assert exc_info.value.code == 0


def test_load_hlo_dir_pins_num_gpu_blocks_override_from_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    load_dir = tmp_path / "loaded_hlo"
    load_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "forward_bucket16": {
            "inputs": [
                {
                    "index": 8,
                    "name": "kv_cache_layer_0",
                    "shape": [50336, 16, 8, 2, 128],
                    "dtype": "bfloat16",
                    "buffer_donor": True,
                }
            ]
        }
    }
    (load_dir / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(load_dir))

    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None)
    )
    stablehlo_export._install_warmup_early_exit_hook(vllm_cfg)
    assert vllm_cfg.cache_config.num_gpu_blocks_override == 50336


def test_generic_dynamo_placeholder_and_packed_module_reflection() -> None:
    class _FakeFusedProj(torch.nn.Module):
        def __init__(self, in_dim: int, out_sizes: list[int]) -> None:
            super().__init__()
            self.input_size = in_dim
            self.output_sizes = out_sizes
            self.tp_size = 1
            self.weight = torch.nn.Parameter(
                torch.empty((in_dim, sum(out_sizes)), dtype=torch.bfloat16)
            )

    class _FakeLinear(torch.nn.Module):
        def __init__(self, out_dim: int, in_dim: int) -> None:
            super().__init__()
            self.input_size = in_dim
            self.output_size = out_dim
            self.weight = torch.nn.Parameter(
                torch.empty((in_dim, out_dim), dtype=torch.bfloat16)
            )

    class _FakeModel(torch.nn.Module):
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }

        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(
                151936, 1024, dtype=torch.bfloat16
            )
            self.model.embed_tokens.output_size_per_partition = 1024
            layer0 = torch.nn.Module()
            layer0.self_attn = torch.nn.Module()
            layer0.self_attn.qkv_proj = _FakeFusedProj(1024, [2048, 1024, 1024])
            layer0.self_attn.o_proj = _FakeLinear(1024, 2048)
            layer0.self_attn.rotary_emb = torch.nn.Module()
            layer0.self_attn.rotary_emb.register_buffer(
                "cos_sin_cache", torch.empty((40960, 128), dtype=torch.bfloat16)
            )
            layer0.mlp = torch.nn.Module()
            layer0.mlp.gate_up_proj = _FakeFusedProj(1024, [3072, 3072])
            self.model.layers = torch.nn.ModuleList([layer0])

    fake_model = _FakeModel()
    stablehlo_export.register_active_model(fake_model)

    outer_names_and_tensors: list[tuple[str, Any]] = [
        ("l_input_ids_", torch.empty((16,), dtype=torch.int32)),
        ("s72", 16),
        (
            "l_self_modules_embed_tokens_parameters_weight_",
            fake_model.model.embed_tokens.weight,
        ),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_",
            fake_model.model.layers[0].self_attn.qkv_proj.weight,
        ),
        ("l_positions_", torch.empty((16,), dtype=torch.int32)),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_",
            fake_model.model.layers[0].self_attn.rotary_emb.cos_sin_cache,
        ),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_attn_kv_cache",
            torch.empty((512, 128, 8, 2, 128), dtype=torch.bfloat16),
        ),
        (
            "g_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_seq_lens",
            torch.empty((16,), dtype=torch.int32),
        ),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
            fake_model.model.layers[0].self_attn.o_proj.weight,
        ),
        (
            "l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_",
            fake_model.model.layers[0].mlp.gate_up_proj.weight,
        ),
    ]

    outer_g = fx.Graph()
    inner_g = fx.Graph()
    inputs: list[Any] = []
    for idx, (ph_name, val) in enumerate(outer_names_and_tensors):
        outer_g.placeholder(ph_name)
        inner_g.placeholder(f"arg{idx}_1")
        inputs.append(val)
    outer_g.output(())
    inner_g.output(())
    outer_gm = fx.GraphModule(torch.nn.Module(), outer_g)
    inner_gm = fx.GraphModule(torch.nn.Module(), inner_g)

    specs = stablehlo_export._extract_input_specs(
        inner_gm, inputs, outer_graph=outer_gm, outer_inputs=inputs
    )
    assert len(specs) == 9
    assert specs[0]["name"] == "input_ids"
    assert specs[0]["argument_type"] == "user_io"

    assert specs[1]["weight_name"] == "model.embed_tokens.weight"
    assert specs[1]["sources"] == [
        {
            "path": "",
            "source_tensor": "model.embed_tokens.weight",
            "source_shape": [151936, 1024],
        }
    ]

    assert specs[2]["weight_name"] == "model.layers.0.self_attn.qkv_proj.weight"
    assert len(specs[2]["sources"]) == 3
    assert (
        specs[2]["sources"][0]["source_tensor"]
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert specs[2]["sources"][0]["transpose"] == [1, 0]
    assert specs[2]["sources"][0]["dst_slices"] == [
        {"dim": 1, "start": 0, "length": 2048}
    ]

    assert specs[4]["weight_name"] == "rotary_emb_inv_freq"
    assert specs[4].get("is_internal_constant") is True

    assert specs[5]["name"] == "kv_cache_layer_0"
    assert specs[5]["argument_type"] == "state"

    assert specs[6]["name"] == "kv_lens"
    assert specs[6]["argument_type"] == "user_io"

    assert specs[7]["weight_name"] == "model.layers.0.self_attn.o_proj.weight"
    assert specs[7]["sources"][0]["transpose"] == [1, 0]
    assert specs[7]["sources"][0]["source_shape"] == [1024, 2048]

    assert specs[8]["weight_name"] == "model.layers.0.mlp.gate_up_proj.weight"
    assert len(specs[8]["sources"]) == 2
    stablehlo_export.register_active_model(None)
