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


def test_td8255_ced_replay_and_window_starts_int32_with_int64_positions(monkeypatch):
    """TD8.255: Verify _get_ced_replay_starts, window_starts, replay_starts, and
    _get_ced_padding_mask stay native torch.int32 even when positions is torch.int64."""
    from dataclasses import dataclass, field

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import _zero_hits

    @dataclass
    class _FakeMeta:
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        replay_starts_cpu: tuple[int, ...] | None = None
        input_positions: torch.Tensor | None = None

    @dataclass
    class _FakeCtx:
        attn_metadata: dict
        additional_kwargs: dict = field(default_factory=dict)
        is_padding: torch.Tensor | None = None

    dummy = SimpleNamespace()
    pos_i64 = torch.arange(8, dtype=torch.int64)
    pos_i32 = torch.arange(8, dtype=torch.int32)

    z1 = DeepseekV41Model._get_ced_replay_starts(dummy, _zero_hits(4), pos_i64)
    z2 = DeepseekV41Model._get_ced_replay_starts(dummy, (0, 0, 0, 0), pos_i32)
    assert z1.dtype == torch.int32
    assert z1 is z2
    assert torch.equal(z1, torch.zeros(4, dtype=torch.int32))

    nz1 = DeepseekV41Model._get_ced_replay_starts(
        dummy, [0, 64, 128, 0], pos_i64, cache_attr="_nz_cache"
    )
    nz2 = DeepseekV41Model._get_ced_replay_starts(
        dummy, (0, 64, 128, 0), pos_i32, cache_attr="_nz_cache"
    )
    assert nz1.dtype == torch.int32
    assert nz1 is nz2
    assert torch.equal(nz1, torch.tensor([0, 64, 128, 0], dtype=torch.int32))

    arange_dtypes: list[torch.dtype | None] = []
    orig_arange = torch.arange

    def spy_arange(*args, **kwargs):
        arange_dtypes.append(kwargs.get("dtype"))
        return orig_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", spy_arange)
    pad = DeepseekV41Model._get_ced_padding_mask(dummy, 6, 4, pos_i64.device)
    assert arange_dtypes == [torch.int32]
    assert pad.dtype == torch.bool
    assert torch.equal(pad, torch.tensor([False, False, False, False, True, True]))
    monkeypatch.setattr(torch, "arange", orig_arange)

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
    model.aux_hidden_state_layers = ()

    first_meta = _FakeMeta(
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        replay_starts_cpu=(0, 32),
    )
    ctx = _FakeCtx(attn_metadata={"l0": first_meta})

    captured_enc_win: list[torch.Tensor] = []
    captured_dec_rep: list[torch.Tensor] = []

    def fake_enc(*args):
        c = v2_model.upstream.get_forward_context()
        captured_enc_win.append(c.additional_kwargs["deepseek_v41_window_starts"])
        return torch.zeros((4, 8)), torch.zeros((4, 4))

    def fake_dec(*args, **kwargs):
        c = v2_model.upstream.get_forward_context()
        captured_dec_rep.append(c.additional_kwargs["deepseek_v41_replay_starts"])
        return torch.ones((args[1].shape[0], 8))

    model.encoder_stage = fake_enc
    model.decoder_stage = fake_dec

    with v2_model.override_forward_context(ctx):
        # 1) plan is None with int64 positions
        monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: None)
        model.forward(torch.zeros(4, dtype=torch.int32), pos_i64[:4])
        assert captured_enc_win[-1].dtype == torch.int32
        assert captured_dec_rep[-1].dtype == torch.int32
        assert torch.equal(
            captured_enc_win[-1], torch.tensor([0, 32], dtype=torch.int32)
        )
        assert torch.equal(
            captured_dec_rep[-1], torch.tensor([0, 0], dtype=torch.int32)
        )

        # 2) identity plan with int64 positions
        id_plan = DecoderReplayPlan(
            indices=(0, 1, 2, 3),
            query_start_loc=(0, 2, 4),
            replay_starts=(0, 64),
            padded_num_tokens=4,
            is_identity=True,
        )
        monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: id_plan)
        model.forward(torch.zeros(4, dtype=torch.int32), pos_i64[:4])
        assert captured_enc_win[-1].dtype == torch.int32
        assert captured_dec_rep[-1].dtype == torch.int32
        assert torch.equal(
            captured_dec_rep[-1], torch.tensor([0, 64], dtype=torch.int32)
        )

        # 3) non-identity plan with int64 positions
        non_id_plan = DecoderReplayPlan(
            indices=(1, 3),
            query_start_loc=(0, 1, 2),
            replay_starts=(16, 48),
            padded_num_tokens=2,
            is_identity=False,
        )
        monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: non_id_plan)
        model.forward(torch.zeros(4, dtype=torch.int32), pos_i64[:4])
        assert captured_enc_win[-1].dtype == torch.int32
        assert captured_dec_rep[-1].dtype == torch.int32
        assert torch.equal(
            captured_dec_rep[-1], torch.tensor([16, 48], dtype=torch.int32)
        )


def test_td8259_ced_non_identity_int32_gather_inverse_and_pad_scatter(monkeypatch):
    """TD8.259: Verify _ced_inverse_rows, _scatter_decoder_rows,
    _get_ced_non_identity_tensors, and non-identity forward use native int32
    inverse/gather and F.pad + index_select."""
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

    inv = v2_model._ced_inverse_rows((1, 3), 3, 4, torch.device("cpu"))
    assert inv.dtype == torch.int32
    assert torch.equal(inv, torch.tensor([3, 0, 3, 1], dtype=torch.int32))

    src = torch.tensor([[10.0, 20.0], [30.0, 40.0], [99.0, 99.0]])
    scat = v2_model._scatter_decoder_rows(src, inv)
    ref = torch.cat([src, src.new_zeros((1, src.shape[-1]))]).index_select(0, inv)
    assert torch.equal(scat, ref)

    dummy = SimpleNamespace()
    plan = DecoderReplayPlan(
        indices=[1, 3],
        query_start_loc=[0, 1, 2],
        replay_starts=(0, 0),
        padded_num_tokens=4,
        is_identity=False,
    )
    pos = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    inv_t, gather_t, bounds_t = DeepseekV41Model._get_ced_non_identity_tensors(
        dummy, plan, 4, pos
    )
    assert inv_t.dtype == torch.int32
    assert gather_t.dtype == torch.int32
    assert bounds_t.dtype == torch.int32
    assert torch.equal(inv_t, torch.tensor([4, 0, 4, 1], dtype=torch.int32))
    assert torch.equal(gather_t, torch.tensor([1, 3, 3, 3], dtype=torch.int32))
    assert torch.equal(bounds_t, torch.tensor([0, 1, 2], dtype=torch.int32))
    # Cache hit
    cached = DeepseekV41Model._get_ced_non_identity_tensors(dummy, plan, 4, pos)
    assert cached[0] is inv_t and cached[1] is gather_t and cached[2] is bounds_t

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
    model.config = SimpleNamespace(hidden_size=2)
    model.aux_hidden_state_layers = ()

    first_meta = _FakeMeta(
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    ctx = _FakeCtx(attn_metadata={"l0": first_meta})

    captured_dec_args: list[tuple] = []

    model.encoder_stage = lambda *a: (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
    )

    def fake_dec(dec_ids, dec_pos, inputs_embeds=None, encoder_pre_mix=None):
        captured_dec_args.append((dec_ids, dec_pos, inputs_embeds, encoder_pre_mix))
        return inputs_embeds * 2.0

    model.decoder_stage = fake_dec
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: plan)

    with v2_model.override_forward_context(ctx):
        out_with_ids = model.forward(
            torch.tensor([10, 20, 30, 40], dtype=torch.int32), pos
        )
        out_no_ids = model.forward(None, pos, inputs_embeds=torch.zeros((4, 2)))

    dec_ids, dec_pos, dec_h, dec_m = captured_dec_args[0]
    assert dec_ids is not None and torch.equal(
        dec_ids, torch.tensor([20, 40, 40, 40], dtype=torch.int32)
    )
    assert torch.equal(dec_pos, torch.tensor([101, 103, 103, 103], dtype=torch.int64))
    assert torch.equal(
        dec_h, torch.tensor([[3.0, 4.0], [7.0, 8.0], [7.0, 8.0], [7.0, 8.0]])
    )
    assert torch.equal(
        dec_m, torch.tensor([[0.3, 0.4], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8]])
    )
    assert captured_dec_args[1][0] is None
    assert torch.equal(
        out_with_ids,
        torch.tensor([[0.0, 0.0], [6.0, 8.0], [0.0, 0.0], [14.0, 16.0]]),
    )
    assert torch.equal(out_no_ids, out_with_ids)


def test_td8272_2d_comb_res_mix_and_int32_default_pre_mix(monkeypatch):
    """TD8.272: Verify 2D comb_res_mix reshape/slice in v2/model.py, v2/dspark.py,
    and common/mhc.py matches 3D [T, M, M] reference, and _default_pre_mix uses
    int32 torch.arange."""
    import jax.numpy as jnp

    from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc as common_mhc
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as v2_dspark

    tokens, mult, dim = 6, 4, 8
    gen = torch.Generator().manual_seed(272)
    x = torch.randn(tokens, dim, generator=gen, dtype=torch.bfloat16)
    residual = torch.randn(tokens, mult, dim, generator=gen, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(tokens, mult, generator=gen, dtype=torch.float32)
    comb_res_mix = torch.randn(tokens, mult, mult, generator=gen, dtype=torch.float32)
    pre_mix = torch.randn(tokens, mult, generator=gen, dtype=torch.float32)

    # Reference 3D calculation
    flat = residual.reshape(tokens, mult * dim)
    x_f32 = x.reshape(tokens, dim).to(residual.dtype).float()
    post_f32 = post_layer_mix.reshape(tokens, mult).float()
    comb_3d = comb_res_mix.reshape(tokens, mult, mult).float()
    streams = [flat[:, i * dim : (i + 1) * dim].float() for i in range(mult)]
    ref_streams = []
    for j in range(mult):
        acc = post_f32[:, j : j + 1] * x_f32
        for i in range(mult):
            acc = acc + comb_3d[:, i, j : j + 1] * streams[i]
        ref_streams.append(acc.to(residual.dtype).float())

    ref_collapsed = v2_model._collapse_streams(ref_streams, pre_mix, residual.dtype)
    ref_aux = v2_model._mean_flat_streams(ref_streams, residual.dtype)

    col_model, aux_model = v2_model._post_and_collapse(
        x, residual, post_layer_mix, comb_res_mix, pre_mix, capture_aux=True
    )
    assert torch.equal(col_model, ref_collapsed)
    assert aux_model is not None and torch.equal(aux_model, ref_aux)

    col_dspark = v2_dspark._post_and_collapse(
        x, residual, post_layer_mix, comb_res_mix, pre_mix
    )
    assert torch.equal(col_dspark, ref_collapsed)

    jax_streams = common_mhc._post_streams_jax(
        jnp.asarray(x.float().numpy()),
        jnp.asarray(flat.float().numpy()),
        jnp.asarray(post_layer_mix.numpy()),
        jnp.asarray(comb_res_mix.numpy()),
    )
    for j in range(mult):
        ref_jax_j = (
            post_f32[:, j : j + 1] * x.float()
            + sum(
                comb_3d[:, i, j : j + 1] * flat[:, i * dim : (i + 1) * dim].float()
                for i in range(mult)
            )
        ).numpy()
        assert torch.allclose(
            torch.tensor(jax_streams[j].__array__()),
            torch.from_numpy(ref_jax_j),
            atol=1e-5,
        )

    # Verify _default_pre_mix passes dtype=torch.int32 to torch.arange
    seen_arange_dtypes: list[torch.dtype | None] = []
    orig_arange = torch.arange

    def tracking_arange(*args, **kwargs):
        seen_arange_dtypes.append(kwargs.get("dtype"))
        return orig_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", tracking_arange)
    default_pre = common_mhc._default_pre_mix(residual, torch.float32)
    assert seen_arange_dtypes == [torch.int32]
    expected_pre = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * tokens, dtype=torch.float32)
    assert torch.equal(default_pre, expected_pre)


def test_td8367_ced_decoder_owner_kwargs_identity_fast_path_and_single_offload_flags(
    monkeypatch,
):
    import numpy as np

    class _TrackedPlan:
        def __init__(self, indices, qsl, padded):
            self._indices = indices
            self._qsl = qsl
            self.padded_num_tokens = padded
            self.qsl_reads = 0

        @property
        def indices(self):
            return self._indices

        @property
        def query_start_loc(self):
            self.qsl_reads += 1
            return self._qsl

    enc_layout = SimpleNamespace(owners=np.array([0, 1], dtype=np.int32))
    first = SimpleNamespace(
        owner_perm=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        owner_q_loc=torch.zeros((2, 2), dtype=torch.int32),
        owner_token_layout=enc_layout,
    )
    plan = _TrackedPlan((1, 3), (0, 1, 2), 4)
    bounds = torch.tensor([0, 1, 2], dtype=torch.int32)
    holder = SimpleNamespace()

    out1 = DeepseekV41Model._get_ced_decoder_owner_kwargs(first, plan, bounds, holder)
    assert plan.qsl_reads == 1

    # Second call with same (plan, enc_layout, tp_size, bounds.device) hits by identity
    # without reading plan.query_start_loc.
    out2 = DeepseekV41Model._get_ced_decoder_owner_kwargs(first, plan, bounds, holder)
    assert out2 is out1
    assert plan.qsl_reads == 1

    # Equivalent plan instance hits by value-equality fallback.
    plan_eq = _TrackedPlan((1, 3), (0, 1, 2), 4)
    out3 = DeepseekV41Model._get_ced_decoder_owner_kwargs(
        first, plan_eq, bounds, holder
    )
    assert out3 is out1
    assert plan_eq.qsl_reads == 1

    # Verify forward unpacks _engram_offload_flags once when both flags are False.
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    model._ced_enabled = False
    model.start_layer = 0
    model.end_layer = 1
    model.layers = [_make_engram_layer(cpu_offload=False, host_dma=False)]
    model.full_stage = lambda *a, **kw: "done"
    flag_calls = []
    orig_flags = DeepseekV41Model._engram_offload_flags

    def _spy_flags(self):
        flag_calls.append(1)
        return orig_flags(self)

    monkeypatch.setattr(DeepseekV41Model, "_engram_offload_flags", _spy_flags)
    assert (
        model.forward(
            torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
        )
        == "done"
    )
    assert len(flag_calls) == 1

    # TD8.375: SimpleNamespace metadata values clone cleanly via cls.__new__(cls)
    from dataclasses import dataclass

    @dataclass
    class _FakeCtx:
        attn_metadata: object
        additional_kwargs: dict
        is_padding: object = None

    model._ced_enabled = True
    model._ced_boundary = 1
    model._ced_window = 512
    model._ced_compile_sizes = (4,)
    model._ced_dp_sync = False
    model.use_sequence_parallel = False
    model.config = SimpleNamespace(hidden_size=4)
    model.aux_hidden_state_layers = ()
    ns_meta = SimpleNamespace(
        seq_lens=torch.tensor([2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    ctx_ns = _FakeCtx(
        attn_metadata={"l0": ns_meta},
        additional_kwargs={},
        is_padding=None,
    )
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx_ns)
    monkeypatch.setattr(
        v2_model,
        "plan_decoder_replay",
        lambda *a: DecoderReplayPlan(
            indices=(0,),
            query_start_loc=(0, 1),
            replay_starts=(0,),
            padded_num_tokens=2,
            is_identity=False,
        ),
    )
    model.encoder_stage = lambda *a: (torch.zeros((2, 4)), torch.zeros((2, 4)))
    model.decoder_stage = lambda *a, **kw: torch.ones((2, 4))
    out_ns = model.forward(
        torch.zeros(2, dtype=torch.int32), torch.arange(2, dtype=torch.int32)
    )
    assert out_ns.shape == (2, 4)
