# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek-V4.1 v2/model.py host loop optimizations (TD7.35)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model
from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import DecoderReplayPlan
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import HOST_ENGRAM_KEY
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
)


def _make_engram_layer(*, cpu_offload: bool, host_dma: bool, layer_hash_index: int = 0):
    embed = SimpleNamespace(cpu_offload=cpu_offload, host_dma=host_dma)
    engram = SimpleNamespace(
        embed_tokens=embed,
        layer_hash_index=layer_hash_index,
        prepare_embeddings=MagicMock(),
    )
    return SimpleNamespace(engram=engram)


def test_engram_offload_flags_cache_and_invalidation():
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = [SimpleNamespace(engram=None), SimpleNamespace(engram=None)]

    # Empty active engrams returns (False, False) immediately
    assert model._has_cpu_offloaded_engram() is False
    assert model._has_host_dma_engram() is False

    l0 = _make_engram_layer(cpu_offload=True, host_dma=False, layer_hash_index=0)
    l1 = _make_engram_layer(cpu_offload=True, host_dma=False, layer_hash_index=1)
    model.layers = [l0, l1]
    model._engram_modules_cache = None

    assert model._has_cpu_offloaded_engram() is True
    # Second call hits _engram_offload_flags_cache
    assert model._has_host_dma_engram() is False
    assert model._engram_offload_flags_cache is not None

    # Mutating host_dma to True invalidates the cached flags automatically
    l0.engram.embed_tokens.host_dma = True
    l1.engram.embed_tokens.host_dma = True
    assert model._has_cpu_offloaded_engram() is False
    assert model._has_host_dma_engram() is True


def test_prepare_engram_reuses_host_image_sentinel_mask(monkeypatch):
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    model.start_layer = 0
    model.end_layer = 1
    l0 = _make_engram_layer(cpu_offload=True, host_dma=False, layer_hash_index=0)
    model.layers = [l0]
    model.engram_swa_prefix = "swa"
    model.engram_dp_shared_memory = True
    model.engram_hash = SimpleNamespace(
        use_slot_cache=True,
        lookback_depth=2,
        ensure_cache=lambda: True,
        hash_on_host=lambda *args: torch.zeros((3, 1, 2), dtype=torch.int64),
    )

    host_obj = SimpleNamespace(
        can_hash_on_host=True,
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        lookback_token_ids=torch.tensor([[1, 2]], dtype=torch.int32),
    )
    swa_meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
    )
    ctx = SimpleNamespace(
        attn_metadata={"swa": swa_meta},
        additional_kwargs={HOST_ENGRAM_KEY: host_obj},
    )

    sentinel_calls = []

    def spy_sentinel(t: torch.Tensor) -> torch.Tensor:
        sentinel_calls.append(t)
        return t == 11

    monkeypatch.setattr(v2_model.upstream, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_model.upstream, "image_sentinel_mask", spy_sentinel)
    monkeypatch.setattr(v2_model, "gather_host_engram_hashes", lambda h, **kw: h)

    input_ids = torch.tensor([10, 11, 12], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    lookback = torch.tensor([[1, 2]], dtype=torch.int32)

    hashes, mask = model._prepare_engram(input_ids, positions, lookback)
    assert hashes is not None
    assert torch.equal(mask, torch.tensor([True, False, True]))
    # Called only twice in _host_hash_inputs (once for ids, once for lookback),
    # NOT a 3rd time in _prepare_engram!
    assert len(sentinel_calls) == 2


def test_causal_lm_forward_short_circuits_host_dma_env_lookup(monkeypatch):
    upstream_lm = v2_model.upstream.DeepseekV41LLMForCausalLM
    monkeypatch.setattr(upstream_lm, "forward", lambda self, *a, **k: "ok")

    moves = []
    lm = object.__new__(DeepseekV41ForCausalLM)
    lm.__dict__["model"] = SimpleNamespace(
        engram_hash=None,
        _engram_rows_pinned=False,
        _move_engram_rows_to_pinned_host=lambda: moves.append(1),
    )

    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", "1")
    assert lm.forward() == "ok"
    assert len(moves) == 1
    assert lm.model._engram_rows_pinned is True

    # Second forward short-circuits without checking envs or calling move again
    assert lm.forward() == "ok"
    assert len(moves) == 1


def test_forward_ced_plan_none_reuses_padding_mask_and_aux_branches(monkeypatch):
    from dataclasses import dataclass, field

    @dataclass
    class _FakeMeta:
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        input_positions: torch.Tensor | None = None

    @dataclass
    class _FakeCtx:
        attn_metadata: dict
        additional_kwargs: dict = field(default_factory=dict)
        is_padding: torch.Tensor | None = None

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    model._ced_enabled = True
    model._ced_boundary = 1
    model._ced_window = 512
    model._ced_compile_sizes = (4,)
    model._ced_dp_sync = False
    model.start_layer = 0
    model.end_layer = 2
    model.layers = []
    model.use_sequence_parallel = False
    model.use_mega_moe = False
    model.config = SimpleNamespace(hidden_size=8)
    model.aux_hidden_state_layers = (1, 2)

    first_meta = _FakeMeta(
        seq_lens=torch.tensor([2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    ctx = _FakeCtx(
        attn_metadata={"l0": first_meta},
        additional_kwargs={},
        is_padding=None,
    )
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx)

    # Case 1: plan.padded_num_tokens == 0 with non-empty aux_hidden_state_layers
    empty_plan = DecoderReplayPlan(
        indices=(),
        query_start_loc=(0, 0),
        replay_starts=(0,),
        padded_num_tokens=0,
        is_identity=False,
    )
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: empty_plan)
    model.encoder_stage = lambda *a: (
        torch.zeros((4, 8)),
        torch.zeros((4, 4)),
        [torch.ones((4, 8))],
    )
    zero_h, zero_aux = model.forward(
        torch.zeros(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32)
    )
    assert zero_h.shape == (4, 8)
    assert len(zero_aux) == 2

    # Case 2: plan is None with first is not None and 2-tuple decoder_stage output
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: None)
    model.decoder_stage = lambda *a, **kw: (
        torch.ones((4, 8)),
        [torch.full((4, 8), 2.0)],
    )
    out_h, out_aux = model.forward(
        torch.zeros(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32)
    )
    assert out_h.shape == (4, 8)
    assert len(out_aux) == 2
    assert model._ced_padding_mask_cache is not None
    assert torch.equal(
        model._ced_padding_mask_cache[1],
        torch.tensor([False, False, True, True]),
    )

    # Case 3: non-identity plan with 2-tuple decoder_stage output (scatter aux)
    non_id_plan = DecoderReplayPlan(
        indices=(0, 1),
        query_start_loc=(0, 2),
        replay_starts=(0,),
        padded_num_tokens=2,
        is_identity=False,
    )
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: non_id_plan)
    model.decoder_stage = lambda *a, **kw: (
        torch.ones((2, 8)),
        [torch.full((2, 8), 3.0)],
    )
    scat_h, scat_aux = model.forward(
        torch.zeros(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32)
    )
    assert scat_h.shape == (4, 8)
    assert len(scat_aux) == 2
    assert scat_aux[1].shape == (4, 8)


def test_encoder_stage_returns_encoder_aux_and_dp_shared_memory_guard(monkeypatch):
    # 1. stage == "encoder" with encoder_aux non-empty (line 934)
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    model.use_sequence_parallel = False
    model.use_mega_moe = False
    model._mtp_hidden_buffer = None
    model.start_layer = 0
    model.end_layer = 2
    model._ced_boundary = 1
    model.aux_hidden_state_layers = (0,)
    model._prepare_engram = lambda *a, **kw: (None, None)
    model.embed_input_ids = lambda ids: torch.zeros((ids.shape[0], 4, 8))

    def enc_layer(h, pos, inp, pm, post_m, res_m, res, eh, em, capture_previous_aux):
        return (
            torch.zeros((2, 8)),
            torch.zeros((2, 4, 8)),
            torch.ones((2, 4, 1)),
            torch.ones((2, 4, 4)),
            torch.ones((2, 4)),
            torch.ones((2, 8)),
        )

    dec_layer = SimpleNamespace(
        attn_norm=lambda x: x,
        attn=SimpleNamespace(write_encoder_global_kv=MagicMock()),
    )
    model.layers = [enc_layer, dec_layer]
    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(v2_model, "post", lambda h, r, pm, rm: r)

    h_out, pm_out, enc_aux = model._forward_stage(
        torch.tensor([1, 2]), torch.tensor([0, 1]), None, stage="encoder"
    )
    assert h_out.shape == (2, 4, 8)
    assert pm_out.shape == (2, 4)
    assert len(enc_aux) == 1

    # 2. DeepseekV41ForCausalLM.__init__ dp_shared_memory guard (line 1131)
    bad_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        engram_config=SimpleNamespace(dp_shared_memory=True),
    )
    with pytest.raises(
        NotImplementedError, match="engram dp_shared_memory is not supported"
    ):
        DeepseekV41ForCausalLM(vllm_config=bad_cfg)
