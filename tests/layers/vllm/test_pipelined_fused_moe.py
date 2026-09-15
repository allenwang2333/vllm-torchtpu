# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for MoE chunk pipelining, mathematical derivation, and contract checks."""

import inspect
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.vllm.fused_moe import fused_moe_gmm
from vllm_torchtpu.layers.vllm.pipelined_fused_moe import (
    _pack_routing, _slice_gathered, _unpack_routing, calculate_moe_chunks,
    enable_pipelined_collective_and_compute, pipelined_fused_moe_gmm)
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import \
    VllmCompressedTensorsW4A16MoEMethod
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
    VllmCompressedTensorsW4ANMxfp4MoEMethod
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8MoEMethodTPU
from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod
from vllm_torchtpu.layers.vllm.quantization.nvfp4 import VllmNvfp4MoEMethod
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod


def _make_mock_moe_config():
    return SimpleNamespace(
        experts_per_token=8,
        moe_parallel_config=SimpleNamespace(use_ep=False),
        group_size=32,
    )


def test_env_var_default_and_override():
    """Verify TPU_MOE_COLLECTION_CHUNK_SIZE default is 0 and parses positive ints."""
    with patch.dict(os.environ, {}, clear=True):
        assert envs.TPU_MOE_COLLECTION_CHUNK_SIZE == 0

    with patch.dict(os.environ, {"TPU_MOE_COLLECTION_CHUNK_SIZE": "16384"}):
        assert envs.TPU_MOE_COLLECTION_CHUNK_SIZE == 16384

    with patch.dict(os.environ, {"TPU_MOE_COLLECTION_CHUNK_SIZE": "0"}):
        assert envs.TPU_MOE_COLLECTION_CHUNK_SIZE == 0


def test_enable_pipelined_collective_and_compute():
    """Verify enable_pipelined_collective_and_compute reflects TPU_MOE_COLLECTION_CHUNK_SIZE > 0."""
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0):
        assert enable_pipelined_collective_and_compute() is False

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert enable_pipelined_collective_and_compute() is True


def test_enable_pipelined_collective_allows_pcp():
    """Verify PCP does not disable MoE collective chunking."""
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert enable_pipelined_collective_and_compute() is True


@pytest.mark.parametrize(
    "method_cls",
    [
        VllmFp8MoEMethodTPU,
        VllmUnquantizedFusedMoEMethod,
        VllmMxfp4MoEMethod,
        VllmNvfp4MoEMethod,
        VllmCompressedTensorsW4A16MoEMethod,
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    ],
)
def test_supports_internal_mk_property(method_cls):
    """Verify supports_internal_mk is True when chunk_size > 0, False when 0 across all 6 quantization classes."""
    prop = getattr(method_cls, "supports_internal_mk")
    assert isinstance(prop, property)

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0):
        assert prop.fget(None) is False

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert prop.fget(None) is True


def test_calculate_moe_chunks_math():
    """Verify chunk calculation: N_chunk = ceil((DP * S) / C), S_chunk = S // N_chunk."""
    # S=8192, DP=4, C=16384 -> T_global=32768 -> N_chunk=2, S_chunk=4096
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=8192,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 4096

    # S=4096, DP=4, C=16384 -> T_global=16384 -> N_chunk=1, S_chunk=4096
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=4096,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 1
    assert chunk_size_local == 4096

    # S=4096, DP=8, C=16384 -> T_global=32768 -> N_chunk=2, S_chunk=2048
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=4096,
                                                        parallel_size=8,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 2048

    # Sub-threshold: S=1024, DP=4, C=16384 -> T_global=4096 -> N_chunk=1, S_chunk=1024
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=1024,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 1
    assert chunk_size_local == 1024

    # Non-multiple global: S=3072, DP=8, C=16384 -> T_global=24576 -> N_chunk=2, S_chunk=1536
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=3072,
                                                        parallel_size=8,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 1536

    # Non-multiple global: S=8000, DP=4, C=16384 -> T_global=32000 -> N_chunk=2, S_chunk=4000
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=8000,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 4000

    # Zero/Empty sequence length guard
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=0,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 1
    assert chunk_size_local == 0


def test_calculate_moe_chunks_validation_errors():
    """Verify invalid token/chunk geometries raise clear ValueErrors."""
    # Local tokens (5001) with DP=4, C=16384 -> T=20004 -> N_chunk=2, but 5001 % 2 != 0
    with pytest.raises(ValueError, match="not divisible by chunk count"):
        calculate_moe_chunks(seq_len=5001, parallel_size=4, chunk_size=16384)

    # Local tokens (3) with DP=4, C=6 -> T=12 -> N_chunk=2, but 3 % 2 != 0
    with pytest.raises(ValueError, match="not divisible by chunk count"):
        calculate_moe_chunks(seq_len=3, parallel_size=4, chunk_size=6)


class _MockCollectiveGroup:
    """Mock collective group tracking calls and verifying order."""

    def __init__(self, world_size=4, rank=0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.call_log = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.call_log.append(("all_gather", tensor.shape))
        # Replicate along dim=0 across world_size
        return torch.cat([tensor] * self.world_size, dim=dim)

    def reduce_scatter(self,
                       tensor: torch.Tensor,
                       dim: int = 0) -> torch.Tensor:
        self.call_log.append(("reduce_scatter", tensor.shape))
        chunks = tensor.chunk(self.world_size, dim=dim)
        return chunks[self.rank_in_group]


def test_pipelined_moe_execution_flow_and_numerical_parity():
    """Verify pipelined execution interleaving collectives and compute with numerical parity."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    seq_len = 8192
    hidden_dim = 128
    topk = 2
    num_experts = 4

    # Dummy inputs
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) / topk
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)

    # Dummy weights
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        # Linear compute for parity check: hs * 2.0
        return hidden_states * 2.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_pcp_group", return_value=None), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    # Output shape should match original local hidden_states (S, H)
    assert out.shape == (seq_len, hidden_dim)

    # Numerical parity check: each token should be exactly hs * 2.0
    torch.testing.assert_close(out, hidden_states * 2.0)

    # Call log should record 2 all-gathers (for hidden_states, topk_weights, topk_ids) and 2 reduce-scatters
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    # 3 tensors gathered per chunk * 2 chunks = 6 all_gather calls
    assert len(ag_calls) == 6
    # 1 tensor reduce-scattered per chunk * 2 chunks = 2 reduce_scatter calls
    assert len(rs_calls) == 2


def test_pcp_uses_chunk_pipeline_and_not_dp_collectives():
    """Verify PCP selects its own group and executes multiple MoE chunks."""
    pcp_group = _MockCollectiveGroup(world_size=4, rank=0)
    dp_group = _MockCollectiveGroup(world_size=1, rank=0)
    seq_len = 8192
    hidden_dim = 16
    topk = 2
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) / topk
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 2.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_pcp_group", return_value=pcp_group), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=torch.empty(0),
            w2=torch.empty(0),
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    torch.testing.assert_close(out, hidden_states * 2.0)
    assert len([c for c in pcp_group.call_log if c[0] == "all_gather"]) == 6
    assert len([c for c in pcp_group.call_log
                if c[0] == "reduce_scatter"]) == 2
    assert dp_group.call_log == []


class TestMoEForwardPrecisionBranches:
    """Tests protecting apply_monolithic call paths and signature parity across all MoE precision methods."""

    def test_fused_moe_and_pipelined_signature_parity(self):
        """Ensure fused_moe_gmm and pipelined_fused_moe_gmm maintain identical signatures.

        Prevents changes to fused_moe_gmm parameter list that break pipelined_fused_moe_gmm.
        """
        sig_fused = inspect.signature(fused_moe_gmm)
        sig_pipelined = inspect.signature(pipelined_fused_moe_gmm)

        assert list(sig_fused.parameters.keys()) == list(
            sig_pipelined.parameters.keys()
        ), (f"Signature parameter mismatch between fused_moe_gmm ({list(sig_fused.parameters.keys())}) "
            f"and pipelined_fused_moe_gmm ({list(sig_pipelined.parameters.keys())})."
            )

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_fp8_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify FP8 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._tpu_activation_str = "silu"
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(4, 64, 64, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn)
        layer.w13_weight_scale_inv = torch.ones(4,
                                                1,
                                                1,
                                                128,
                                                dtype=torch.float32)
        layer.w2_weight_scale_inv = torch.ones(4,
                                               1,
                                               1,
                                               64,
                                               dtype=torch.float32)
        layer.w13_bias = torch.zeros(4, 1, 128, dtype=torch.float32)
        layer.w2_bias = torch.zeros(4, 1, 64, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2
        layer.moe_config.moe_parallel_config.use_ep = False

        method = MagicMock(spec=VllmFp8MoEMethodTPU)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.fp8.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.fp8.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.fp8.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.fp8.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x,
                                                       router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_unquantized_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify unquantized apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._tpu_activation_str = "silu"
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_bias = torch.zeros(4, 1, 128, dtype=torch.float32)
        layer.w2_bias = torch.zeros(4, 1, 64, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2

        method = MagicMock()
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.unquantized.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.unquantized.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.unquantized.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.unquantized.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmUnquantizedFusedMoEMethod._forward_monolithic_tpu(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_mxfp4_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify MXFP4 forward monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._tpu_activation_str = "silu"
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w13_bias = None
        layer.w2_bias = None
        layer.moe_config.experts_per_token = 2

        method = MagicMock()
        method.rhs_quant_dtype = None
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.mxfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.mxfp4.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.mxfp4.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.mxfp4.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmMxfp4MoEMethod._forward_monolithic_tpu(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_nvfp4_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify NVFP4 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._tpu_activation_str = "silu"
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2

        method = MagicMock(spec=VllmNvfp4MoEMethod)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmNvfp4MoEMethod.apply_monolithic(method, layer, x,
                                                      router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_w4a16_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify compressed tensors W4A16 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer.activation = "silu"
        layer.moe_config = _make_mock_moe_config()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight_packed = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight_packed = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)

        method = MagicMock(spec=VllmCompressedTensorsW4A16MoEMethod)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 8, dtype=torch.bfloat16) * 0.125
        topk_ids = torch.zeros(4, 8, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmCompressedTensorsW4A16MoEMethod.apply_monolithic(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight_packed

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_w4an_mxfp4_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify compressed tensors W4AN_MXFP4 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._tpu_activation_str = "silu"
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2

        method = MagicMock(spec=VllmCompressedTensorsW4ANMxfp4MoEMethod)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmCompressedTensorsW4ANMxfp4MoEMethod.apply_monolithic(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight


def test_pipelined_moe_single_chunk_dp_greater_than_one():
    """Verify pipelined MoE execution when num_chunks == 1 with DP > 1."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    seq_len = 1024
    hidden_dim = 64
    topk = 2
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 3.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 3.0)

    # In single-chunk mode (num_chunks=1): 3 all_gathers (hs, weights, ids) and 1 reduce_scatter
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    assert len(ag_calls) == 3
    assert len(rs_calls) == 1


def test_pipelined_moe_dp_size_one_no_collectives():
    """Verify pipelined MoE bypasses collectives when DP == 1 or dp_group is None."""
    seq_len = 1024
    hidden_dim = 64
    topk = 2
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 1.5

    # Case 1: dp_group is None
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=None), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )
    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 1.5)

    # Case 2: dp_group world_size == 1
    dp_group_1 = _MockCollectiveGroup(world_size=1, rank=0)
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=dp_group_1), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )
    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 1.5)
    assert len(dp_group_1.call_log) == 0


def test_pipelined_moe_with_none_routing_tensors():
    """Verify pipelining works cleanly when topk_weights or topk_ids are None."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    seq_len = 8192
    hidden_dim = 64
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states,
                       topk_weights=None,
                       topk_ids=None,
                       *args,
                       **kwargs):
        assert topk_weights is None
        assert topk_ids is None
        return hidden_states * 2.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=None,
            topk_ids=None,
            experts_start=None,
            topk=2,
            activation="silu",
        )

    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 2.0)
    # 1 tensor gathered per chunk * 2 chunks = 2 all_gather calls
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    assert len(ag_calls) == 2
    assert len(rs_calls) == 2


def test_pipelined_moe_four_stage_pipeline():
    """Verify pipelining with 4 chunks exercises multi-iteration pipeline steady state."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    # S=16384, DP=4, C=16384 -> T_global=65536 -> N_chunk=4, S_chunk=4096
    seq_len = 16384
    hidden_dim = 64
    topk = 2
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 4.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 4.0)
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    # 3 tensors * 4 chunks = 12 all_gather calls
    assert len(ag_calls) == 12
    # 4 reduce_scatter calls
    assert len(rs_calls) == 4


# --------------------------------------------------------------------------
# Routing-gather strategies: TPU_MOE_UNCHUNKED_ROUTING_GATHER and
# TPU_MOE_FUSED_ROUTING_GATHER.
#
# These change *which* collective carries the routing tensors, so the only way
# they can fail is by silently handing the MoE kernel correct-looking routing
# for the wrong tokens. Nothing else in this file exercises gathered row order,
# so the ordering identity is asserted directly.
# --------------------------------------------------------------------------

_P, _S, _TOPK = 4, 12, 8


def _fake_all_gather(per_rank):
    """What collective_group.all_gather(x, dim=0) returns: rank-major concatenation."""
    return torch.cat(per_rank, dim=0)


def test_slice_gathered_matches_gathering_the_slice():
    """all_gather(x[a:b]) == _slice_gathered(all_gather(x), P, S, a, b).

    The property the hoisted routing gather rests on. A flat row slice of the
    gathered tensor would pass a same-shape, same-dtype, entirely wrong tensor.
    """
    per_rank = [torch.randn(_S, _TOPK) for _ in range(_P)]
    gathered = _fake_all_gather(per_rank)

    for chunk_size_local in (1, 2, 3, 4, 6, 12):
        for i in range(_S // chunk_size_local):
            a, b = i * chunk_size_local, (i + 1) * chunk_size_local
            want = _fake_all_gather([x[a:b] for x in per_rank])
            got = _slice_gathered(gathered, _P, _S, a, b)
            assert got.shape == want.shape
            torch.testing.assert_close(got, want)


def test_flat_row_slice_would_have_been_wrong():
    """Guard the guard: show the naive slice really does differ, so the test has teeth."""
    per_rank = [torch.full((_S, _TOPK), float(r)) for r in range(_P)]
    gathered = _fake_all_gather(per_rank)
    chunk_size_local = _S // 2
    a, b = chunk_size_local, 2 * chunk_size_local
    correct = _slice_gathered(gathered, _P, _S, a, b)
    naive = gathered[_P * a:_P * b]
    assert correct.shape == naive.shape
    assert not torch.equal(correct, naive)
    # correct holds one block per rank; naive holds only the last two ranks.
    assert set(correct.flatten().tolist()) == {0.0, 1.0, 2.0, 3.0}
    assert set(naive.flatten().tolist()) == {2.0, 3.0}


def test_slice_gathered_preserves_trailing_dims():
    """Routing tensors are 2-D here, but the helper must not assume it."""
    per_rank = [torch.randn(_S, 3, 5) for _ in range(_P)]
    gathered = _fake_all_gather(per_rank)
    want = _fake_all_gather([x[2:6] for x in per_rank])
    torch.testing.assert_close(_slice_gathered(gathered, _P, _S, 2, 6), want)


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_pack_unpack_routing_is_exact(weights_dtype, ids_dtype):
    """f32 packing must be lossless for both operands, not merely close.

    Weights: bf16/f32 -> f32 widens. Ids: integers well below 2**24, the last
    f32 integer with an exact successor. Asserted with equality, not a tolerance.
    """
    weights = torch.randn(_S, _TOPK).to(weights_dtype)
    # 256 routed experts; include both endpoints and a -1 "no expert" sentinel.
    ids = torch.randint(0, 256, (_S, _TOPK)).to(ids_dtype)
    ids[0, 0], ids[0, 1], ids[1, 0] = 0, 255, -1

    packed = _pack_routing(weights, ids)
    assert packed.dtype == torch.float32
    assert packed.shape == (_S, 2 * _TOPK)

    got_w, got_i = _unpack_routing(packed, _TOPK, weights_dtype, ids_dtype)
    assert got_w.dtype == weights_dtype and got_i.dtype == ids_dtype
    assert torch.equal(got_w, weights)
    assert torch.equal(got_i, ids)


def test_pack_unpack_survives_the_gather_and_the_slice():
    """The two helpers compose: pack -> gather -> slice -> unpack == gather the slice."""
    per_rank_w = [torch.randn(_S, _TOPK).to(torch.bfloat16) for _ in range(_P)]
    per_rank_i = [torch.randint(0, 256, (_S, _TOPK)) for _ in range(_P)]

    gathered = _fake_all_gather(
        [_pack_routing(w, i) for w, i in zip(per_rank_w, per_rank_i)])

    chunk_size_local = _S // 3
    for c in range(3):
        a, b = c * chunk_size_local, (c + 1) * chunk_size_local
        got_w, got_i = _unpack_routing(
            _slice_gathered(gathered, _P, _S, a, b), _TOPK, torch.bfloat16,
            torch.int32)
        assert torch.equal(got_w, _fake_all_gather([w[a:b] for w in per_rank_w]))
        assert torch.equal(got_i,
                           _fake_all_gather([i[a:b]
                                             for i in per_rank_i]).to(torch.int32))


def _run_routing_arm(unchunked, fused, hs, weights, ids, parallel_size,
                     seq_len, hidden, chunk_divisor=2):
    """Drive pipelined_fused_moe_gmm once, returning (routing gather count, kernel inputs)."""
    calls = []

    def all_gather(x, dim=0):
        calls.append(tuple(x.shape))
        # Emulate a rank-major gather with *distinguishable* per-rank content. If
        # every rank held the same rows, a wrong slice of the gathered tensor would
        # be indistinguishable from a right one and this test would prove nothing.
        return torch.cat([x + r for r in range(parallel_size)], dim=dim)

    group = MagicMock()
    group.world_size = parallel_size
    group.all_gather.side_effect = all_gather
    group.reduce_scatter.side_effect = (
        lambda x, dim=0: x[:x.shape[0] // parallel_size])

    seen = []

    def fake_kernel(**kw):
        seen.append((kw["topk_weights"].clone(), kw["topk_ids"].clone()))
        return torch.zeros(kw["hidden_states"].shape[0], hidden)

    env = {
        "TPU_MOE_COLLECTION_CHUNK_SIZE":
        str(parallel_size * seq_len // chunk_divisor),
        "TPU_MOE_UNCHUNKED_ROUTING_GATHER": "1" if unchunked else "0",
        "TPU_MOE_FUSED_ROUTING_GATHER": "1" if fused else "0",
    }
    with patch.dict(os.environ, env), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe."
               "_get_moe_collective_group", return_value=(group, "dp")), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm",
               side_effect=fake_kernel):
        out = pipelined_fused_moe_gmm(
            hidden_states=hs, w1=None, w2=None, w1_scale=None, w2_scale=None,
            w1_bias=None, w2_bias=None, topk_weights=weights, topk_ids=ids,
            experts_start=None, topk=_TOPK, activation="silu")

    # Classify by trailing dim: routing gathers carry topk (unfused) or 2*topk
    # (fused) columns. The caller must keep `hidden` clear of both -- a collision
    # silently counts the hidden-state gathers as routing gathers.
    assert hidden not in (_TOPK, 2 * _TOPK), "hidden collides with the classifier"
    routing_gathers = [c for c in calls if c[-1] in (_TOPK, 2 * _TOPK)]
    return len(routing_gathers), seen, out, calls


def test_routing_gather_arms_are_equivalent_and_cheaper():
    """All four knob settings must feed the kernel byte-identical routing.

    num_chunks=2, so the unmodified path issues 4 routing gathers per call. The
    whole point of the change is to reach 1 without altering what the kernel sees,
    so cost and equivalence are asserted in the same test -- a cheaper arm that
    quietly reroutes tokens is the failure mode worth catching.
    """
    # hidden must differ from _TOPK and 2*_TOPK; see _run_routing_arm.
    parallel_size, seq_len, hidden = _P, _S, 5
    hs = torch.randn(seq_len, hidden)
    weights = torch.randn(seq_len, _TOPK).to(torch.bfloat16)
    ids = torch.randint(0, 200, (seq_len, _TOPK)).to(torch.int32)

    expected_counts = {
        (False, False): 4,
        (False, True): 2,
        (True, False): 2,
        (True, True): 1,
    }

    reference = None
    for (unchunked, fused), want in expected_counts.items():
        n, seen, out, calls = _run_routing_arm(unchunked, fused, hs, weights,
                                               ids, parallel_size, seq_len,
                                               hidden)
        assert len(seen) == 2, f"{unchunked=} {fused=}: expected two chunks"
        assert n == want, (f"{unchunked=} {fused=}: {n} routing gathers, "
                           f"expected {want}; all gathers were {calls}")
        assert out.shape == (seq_len, hidden)
        for got_w, got_i in seen:
            assert got_w.dtype == weights.dtype, f"{unchunked=} {fused=}"
            assert got_i.dtype == ids.dtype, f"{unchunked=} {fused=}"
            assert got_w.shape == (parallel_size * seq_len // 2, _TOPK)

        if reference is None:
            reference = seen
        else:
            for c, ((rw, ri), (gw, gi)) in enumerate(zip(reference, seen)):
                assert torch.equal(gw, rw), (
                    f"{unchunked=} {fused=} chunk {c}: topk_weights differ from "
                    "the unmodified path")
                assert torch.equal(gi, ri), (
                    f"{unchunked=} {fused=} chunk {c}: topk_ids differ from "
                    "the unmodified path -- tokens are being routed to the "
                    "wrong experts")


def test_routing_knobs_are_inert_without_both_tensors():
    """topk_weights/topk_ids may be None; packing them must not be attempted then."""
    parallel_size, seq_len, hidden = _P, _S, 5
    hs = torch.randn(seq_len, hidden)
    group = MagicMock()
    group.world_size = parallel_size
    group.all_gather.side_effect = lambda x, dim=0: torch.cat(
        [x] * parallel_size, dim=dim)
    group.reduce_scatter.side_effect = (
        lambda x, dim=0: x[:x.shape[0] // parallel_size])

    env = {
        "TPU_MOE_COLLECTION_CHUNK_SIZE": str(parallel_size * seq_len // 2),
        "TPU_MOE_UNCHUNKED_ROUTING_GATHER": "1",
        "TPU_MOE_FUSED_ROUTING_GATHER": "1",
    }
    with patch.dict(os.environ, env), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe."
               "_get_moe_collective_group", return_value=(group, "dp")), \
         patch("vllm_torchtpu.layers.vllm.pipelined_fused_moe.fused_moe_gmm",
               side_effect=lambda **kw: torch.zeros(
                   kw["hidden_states"].shape[0], hidden)):
        out = pipelined_fused_moe_gmm(
            hidden_states=hs, w1=None, w2=None, w1_scale=None, w2_scale=None,
            w1_bias=None, w2_bias=None, topk_weights=None, topk_ids=None,
            experts_start=None, topk=_TOPK, activation="silu")
    assert out.shape == (seq_len, hidden)
