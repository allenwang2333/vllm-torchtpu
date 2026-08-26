# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness contracts for Kimi-K3 sequence parallelism."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.distributed.sequence_parallel import (
    SequenceParallelMesh, build_sequence_parallel_group_plan)
from vllm_torchtpu.models.vllm.kimi_k3.model import KimiDecoderLayer, KimiModel
from vllm_torchtpu.models.vllm.kimi_k3.sequence_parallel import \
    KimiSequenceParallel


def test_all_configured_compile_buckets_use_sp() -> None:
    plan = build_sequence_parallel_group_plan(tuple(range(8)), 4)
    mesh = SequenceParallelMesh(
        plan=plan,
        tp_group=None,
        lane_group=None,
        sequence_group=None,
        lane_rank=0,
        sequence_rank=2,
    )
    layout = KimiSequenceParallel(
        mesh=mesh,
        token_sizes=(4, 1024, 2048, 4096),
    )

    assert layout.enabled
    for num_tokens in (4, 1024, 2048, 4096):
        hidden_states = torch.arange(num_tokens * 2).view(num_tokens, 2)
        shard = layout.shard_model_input(hidden_states)
        expected = hidden_states.chunk(4, dim=0)[2]
        torch.testing.assert_close(shard, expected)
        assert shard.data_ptr() != hidden_states.data_ptr()

    with pytest.raises(ValueError, match="not configured"):
        layout.shard_model_input(torch.empty(512, 2))


def _sequence_parallel_config(
        *,
        sp_size: int,
        tp_size: int = 8,
        dp_size: int = 1,
        compile_sizes: tuple[int, ...] = (4, 512, 1024),
):
    return SimpleNamespace(
        additional_config={"sequence_parallel_size": sp_size},
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            pipeline_parallel_size=1,
            tensor_parallel_size=tp_size,
        ),
        compilation_config=SimpleNamespace(compile_sizes=compile_sizes),
    )


def test_model_adapter_derives_token_buckets_from_vllm_config(
        monkeypatch: pytest.MonkeyPatch) -> None:
    plan = build_sequence_parallel_group_plan(tuple(range(8)), 2)
    mesh = SequenceParallelMesh(
        plan=plan,
        tp_group=None,
        lane_group=None,
        sequence_group=None,
        lane_rank=0,
        sequence_rank=0,
    )
    monkeypatch.setattr(
        SequenceParallelMesh,
        "from_tensor_parallel_group",
        classmethod(lambda cls, size: mesh),
    )

    adapter = KimiSequenceParallel.from_vllm_config(
        _sequence_parallel_config(sp_size=2))

    assert adapter.enabled
    assert adapter.token_sizes == (4, 512, 1024)


def test_model_adapter_rejects_unsupported_layouts() -> None:
    assert not KimiSequenceParallel.from_vllm_config(
        _sequence_parallel_config(sp_size=1)).enabled

    with pytest.raises(ValueError, match="requires DP1/PP1"):
        KimiSequenceParallel.from_vllm_config(
            _sequence_parallel_config(sp_size=4, dp_size=2))

    with pytest.raises(ValueError, match="must be divisible"):
        KimiSequenceParallel.from_vllm_config(
            _sequence_parallel_config(
                sp_size=4,
                compile_sizes=(4, 1023),
            ))


def test_disabled_model_adapter_preserves_layout() -> None:
    adapter = KimiSequenceParallel.disabled()
    hidden_states = torch.empty(4, 8)

    assert adapter.shard_model_input(hidden_states) is hidden_states
    assert adapter.gather_attention_input(hidden_states) is hidden_states
    assert adapter.reduce_attention_output(hidden_states) is hidden_states
    assert adapter.gather_feed_forward_input(hidden_states) is hidden_states
    assert adapter.shard_feed_forward_output(hidden_states) is hidden_states
    assert adapter.gather_model_output(hidden_states) is hidden_states


class _FakeSPLayout:

    enabled = True

    def __init__(self, sp_rank: int = 1) -> None:
        self.sp_rank = sp_rank
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def _record(self, name: str, tensor: torch.Tensor) -> None:
        self.calls.append((name, tuple(tensor.shape)))

    def gather_attention_input(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("gather_attention_input", tensor)
        return torch.cat([tensor] * 4, dim=0)

    def reduce_attention_output(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("reduce_attention_output", tensor)
        return tensor.chunk(4, dim=0)[self.sp_rank].clone()

    def gather_feed_forward_input(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("gather_feed_forward_input", tensor)
        return torch.cat([tensor] * 4, dim=0)

    def shard_feed_forward_output(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("shard_feed_forward_output", tensor)
        return tensor.chunk(4, dim=0)[self.sp_rank].clone()


class _ShapeRecorder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(tensor.shape))
        return tensor


class _AttentionRecorder(_ShapeRecorder):

    def forward(self, positions: torch.Tensor,
                tensor: torch.Tensor) -> torch.Tensor:
        assert tuple(positions.shape) == (8, )
        return super().forward(tensor)


class _ResidualRecorder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, prefix_sum: torch.Tensor,
                block_residuals: torch.Tensor) -> torch.Tensor:
        self.shapes.append(
            (tuple(prefix_sum.shape), tuple(block_residuals.shape)))
        return prefix_sum


@pytest.mark.parametrize("is_moe", [False, True], ids=["dense", "moe"])
def test_decoder_sequence_parallel_contract_keeps_residuals_local(
        is_moe: bool) -> None:
    layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layout = _FakeSPLayout()
    layer.sequence_parallel = layout
    layer.attn_res_block_size = None
    layer.input_layernorm = _ShapeRecorder()
    layer.self_attn = _AttentionRecorder()
    layer.post_attention_layernorm = _ShapeRecorder()
    layer.is_moe = is_moe
    feed_forward = _ShapeRecorder()
    if is_moe:
        layer.block_sparse_moe = feed_forward
    else:
        layer.mlp = feed_forward

    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    output, block_residuals = layer(torch.arange(8), hidden_states, None)

    assert tuple(output.shape) == (2, 4)
    assert block_residuals is None
    assert layer.input_layernorm.shapes == [(2, 4)]
    assert layer.self_attn.shapes == [(8, 4)]
    assert layer.post_attention_layernorm.shapes == [(2, 4)]
    assert feed_forward.shapes == [(8, 4)]
    assert layout.calls == [
        ("gather_attention_input", (2, 4)),
        ("reduce_attention_output", (8, 4)),
        ("gather_feed_forward_input", (2, 4)),
        ("shard_feed_forward_output", (8, 4)),
    ]


def test_attnres_stack_and_prefix_sum_remain_local() -> None:
    layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layout = _FakeSPLayout()
    layer.sequence_parallel = layout
    layer.attn_res_block_size = 2
    layer.layer_idx = 2
    layer.self_attention_res = _ResidualRecorder()
    layer.mlp_res = _ResidualRecorder()
    layer.input_layernorm = _ShapeRecorder()
    layer.self_attn = _AttentionRecorder()
    layer.post_attention_layernorm = _ShapeRecorder()
    layer.is_moe = True
    layer.block_sparse_moe = _ShapeRecorder()

    hidden_states = torch.ones(2, 4)
    block_residuals = torch.zeros(2, 1, 4)
    output, updated_residuals = layer(torch.arange(8), hidden_states,
                                      block_residuals)

    assert tuple(output.shape) == (2, 4)
    assert updated_residuals is not None
    assert tuple(updated_residuals.shape) == (2, 2, 4)
    assert layer.self_attention_res.shapes == [((2, 4), (2, 1, 4))]
    assert layer.mlp_res.shapes == [((2, 4), (2, 2, 4))]
    assert layer.input_layernorm.shapes == [(2, 4)]
    assert layer.post_attention_layernorm.shapes == [(2, 4)]


class _ModelSPLayout:

    enabled = True

    def __init__(self) -> None:
        self.shard_shape: tuple[int, ...] | None = None
        self.final_shape: tuple[int, ...] | None = None

    def shard_model_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.shard_shape = tuple(hidden_states.shape)
        return hidden_states.chunk(4, dim=0)[0].clone()

    def gather_model_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.final_shape = tuple(hidden_states.shape)
        return torch.cat([hidden_states] * 4, dim=0)


class _Embedding(nn.Module):

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([input_ids.float()] * 4, dim=-1)


class _Decoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.sp_shapes: list[tuple[int, ...]] = []
        self.block_residual_shapes: list[tuple[int, ...] | None] = []

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                block_residuals: torch.Tensor | None):
        assert tuple(positions.shape) == (8, )
        self.sp_shapes.append(tuple(hidden_states.shape))
        self.block_residual_shapes.append(
            None if block_residuals is None else tuple(block_residuals.shape))
        return hidden_states + 1, block_residuals


def test_embedding_first_layer_and_final_norm_boundaries() -> None:
    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    layout = _ModelSPLayout()
    decoder = _Decoder()
    final_norm = _ShapeRecorder()
    model.sequence_parallel = layout
    model.attn_res_block_size = None
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([decoder])
    model.norm = final_norm

    output = model.forward(
        input_ids=torch.arange(8),
        positions=torch.arange(8),
    )

    assert layout.shard_shape == (8, 4)
    assert decoder.sp_shapes == [(2, 4)]
    assert decoder.block_residual_shapes == [None]
    assert final_norm.shapes == [(2, 4)]
    assert layout.final_shape == (2, 4)
    assert tuple(output.shape) == (8, 4)


def test_sp_attnres_empty_stack_uses_local_token_shape() -> None:
    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    layout = _ModelSPLayout()
    decoder = _Decoder()
    output_attn_res = _ResidualRecorder()
    model.sequence_parallel = layout
    model.attn_res_block_size = 2
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([decoder])
    model.output_attn_res = output_attn_res
    model.norm = _ShapeRecorder()

    output = model.forward(
        input_ids=torch.arange(8),
        positions=torch.arange(8),
    )

    assert decoder.block_residual_shapes == [(2, 0, 4)]
    assert output_attn_res.shapes == [((2, 4), (2, 0, 4))]
    assert tuple(output.shape) == (8, 4)
