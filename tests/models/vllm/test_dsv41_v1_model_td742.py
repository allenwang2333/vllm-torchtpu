# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% full-module coverage tests for deepseek_v41/v1/model.py (TD7.42)."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_cache as cache_ops,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_compressor as compressor_ops,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_indexer as indexer_ops,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import model as v1_model
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import replay_plan as v1_replay_plan


@dataclass
class _DummyAttnMeta:
    seq_lens: torch.Tensor = field(
        default_factory=lambda: torch.tensor([8, 12], dtype=torch.int32)
    )
    query_start_loc: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    input_positions: torch.Tensor | None = None


@dataclass
class _DummyContext:
    attn_metadata: object
    is_padding: object = None
    additional_kwargs: dict = field(default_factory=dict)


class _DummyV1Model(v1_model.DeepseekV41Model):
    def __init__(self):
        pass


def test_engram_modules_caching_and_invalidation():
    m = _DummyV1Model()
    e1 = SimpleNamespace(layer_hash_index=0, prepare_embeddings=mock.MagicMock())
    e2 = SimpleNamespace(layer_hash_index=1, prepare_embeddings=mock.MagicMock())
    m.layers = [
        SimpleNamespace(engram=None),
        SimpleNamespace(engram=e1),
        SimpleNamespace(engram=None),
        SimpleNamespace(engram=e2),
    ]
    m.start_layer = 0
    m.end_layer = 4

    mods1 = m._engram_modules()
    assert mods1 == (e1, e2)
    # Cache hit returns identical tuple
    mods2 = m._engram_modules()
    assert mods2 is mods1

    # Changing end_layer invalidates cache key
    m.end_layer = 2
    mods3 = m._engram_modules()
    assert mods3 == (e1,)
    assert mods3 is not mods1


def test_ced_helper_caches_and_all_zero_vs_nonzero_replay_starts():
    m = _DummyV1Model()
    positions = torch.arange(4, dtype=torch.int64)

    # 1. _get_ced_replay_starts: all-zeros branch (any(replay_tuple) is False)
    r_zero_1 = m._get_ced_replay_starts((0, 0), positions)
    assert torch.equal(r_zero_1, torch.zeros(2, dtype=torch.int64))
    r_zero_2 = m._get_ced_replay_starts([0, 0], positions)
    assert r_zero_2 is r_zero_1

    # 2. _get_ced_replay_starts: non-zero branch (any(replay_tuple) is True)
    r_nz_1 = m._get_ced_replay_starts((0, 64), positions)
    assert torch.equal(r_nz_1, torch.tensor([0, 64], dtype=torch.int64))
    r_nz_2 = m._get_ced_replay_starts((0, 64), positions)
    assert r_nz_2 is r_nz_1

    # 3. _get_ced_padding_mask: cache hit and miss
    pad1 = m._get_ced_padding_mask(4, 2, positions.device)
    assert torch.equal(pad1, torch.tensor([False, False, True, True]))
    pad2 = m._get_ced_padding_mask(4, 2, positions.device)
    assert pad2 is pad1
    pad3 = m._get_ced_padding_mask(4, 3, positions.device)
    assert torch.equal(pad3, torch.tensor([False, False, False, True]))

    # 4. _get_ced_non_identity_tensors: cache hit and miss
    plan_a = v1_replay_plan.DecoderReplayPlan(
        indices=(0, 2),
        query_start_loc=(0, 1, 2),
        replay_starts=(0, 0),
        padded_num_tokens=4,
    )
    t1 = m._get_ced_non_identity_tensors(plan_a, 4, positions)
    t2 = m._get_ced_non_identity_tensors(plan_a, 4, positions)
    assert t2 is t1
    indices, gather, bounds = t1
    assert torch.equal(indices, torch.tensor([0, 2, 4, 4], dtype=torch.int64))
    assert torch.equal(gather, torch.tensor([0, 2, 3, 3], dtype=torch.int64))
    assert torch.equal(bounds, torch.tensor([0, 1, 2], dtype=torch.int32))

    plan_b = v1_replay_plan.DecoderReplayPlan(
        indices=(1, 3),
        query_start_loc=(0, 1, 2),
        replay_starts=(0, 0),
        padded_num_tokens=4,
    )
    t3 = m._get_ced_non_identity_tensors(plan_b, 4, positions)
    assert t3 is not t1


def test_forward_metadata_deduplication_and_padding_mask_reuse():
    m = _DummyV1Model()
    m._ced_enabled = True
    m._ced_window = 64
    m._ced_compile_sizes = (4,)
    m.config = SimpleNamespace(hidden_size=8)
    m.encoder_stage = mock.MagicMock(
        return_value=(torch.zeros(4, 8), torch.zeros(4, 8))
    )
    m.decoder_stage = mock.MagicMock(return_value=torch.ones(4, 8))

    shared_swa = _DummyAttnMeta()
    shared_mla = _DummyAttnMeta()
    metadata = {
        "layer.0": shared_swa,
        "layer.1": shared_swa,
        "layer.2": shared_mla,
        "layer.3": shared_mla,
    }
    captured_contexts = []
    captured_ctx = []

    def _capture_decoder(decoder_ids, decoder_positions, **kwargs):
        captured_contexts.append(captured_ctx[-1])
        return torch.ones(4, 8)

    @contextmanager
    def _fake_override(ctx):
        captured_ctx.append(ctx)
        yield

    plan = v1_replay_plan.DecoderReplayPlan(
        indices=(0, 2),
        query_start_loc=(0, 1, 2),
        replay_starts=(0, 16),
        padded_num_tokens=4,
    )
    positions = torch.arange(4, dtype=torch.int64)

    with (
        mock.patch.object(
            v1_model.upstream,
            "get_forward_context",
            return_value=_DummyContext(attn_metadata=metadata),
        ),
        mock.patch.object(v1_model, "override_forward_context", _fake_override),
        mock.patch.object(v1_model, "plan_decoder_replay", return_value=plan),
    ):
        m.decoder_stage.side_effect = _capture_decoder
        # Step 1: with input_ids=None to cover the `else None` branch of decoder_ids
        out1 = m.forward(None, positions, inputs_embeds=torch.zeros(4, 8))
        # Step 2: with input_ids Tensor, hits all CED caches
        out2 = m.forward(torch.arange(4, dtype=torch.int64), positions)

    assert out1.shape == (4, 8)
    assert out2.shape == (4, 8)
    dec_meta_1 = captured_contexts[0].attn_metadata
    # Shared metadata instances are deduplicated by id(value)
    assert dec_meta_1["layer.0"] is dec_meta_1["layer.1"]
    assert dec_meta_1["layer.2"] is dec_meta_1["layer.3"]
    assert dec_meta_1["layer.0"] is not dec_meta_1["layer.2"]
    # Cached tensors are reused identically across Step 1 and Step 2
    assert captured_contexts[0].is_padding is captured_contexts[1].is_padding
    assert (
        captured_contexts[0].additional_kwargs["deepseek_v41_replay_starts"]
        is captured_contexts[1].additional_kwargs["deepseek_v41_replay_starts"]
    )


class TestV1ModelFullCoverage:
    def test_stage_classes(self):
        owner = mock.MagicMock()
        owner._forward_stage.return_value = "stage_out"

        full = object.__new__(v1_model.DeepseekV41FullStage)
        v1_model._StageBase.__init__(full, owner=owner, vllm_config=None)
        assert full.forward(1, 2, 3, 4, 5) == "stage_out"
        owner._forward_stage.assert_called_with(1, 2, 3, 4, 5, stage="full")

        enc = object.__new__(v1_model.DeepseekV41EncoderStage)
        v1_model._StageBase.__init__(enc, owner=owner, vllm_config=None)
        assert enc.forward(1, 2, 3, 4, 5) == "stage_out"
        owner._forward_stage.assert_called_with(1, 2, 3, 4, 5, stage="encoder")

        dec = object.__new__(v1_model.DeepseekV41DecoderStage)
        v1_model._StageBase.__init__(dec, owner=owner, vllm_config=None)
        assert dec.forward(1, 2, 3, 4) == "stage_out"
        owner._forward_stage.assert_called_with(
            1, 2, None, 3, encoder_pre_mix=4, stage="decoder"
        )

    def test_model_forward_ced_and_incomplete_prefill(self):
        model = _DummyV1Model()
        model._ced_enabled = False
        model.full_stage = mock.MagicMock(return_value="full_stage_out")
        assert (
            v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            == "full_stage_out"
        )

        model._ced_enabled = True
        model._ced_window = 64
        model._ced_compile_sizes = (16, 32)
        model.config = SimpleNamespace(hidden_size=128)
        model.encoder_stage = mock.MagicMock(
            return_value=[(torch.zeros(4, 128), torch.zeros(4, 128))]
        )
        model.decoder_stage = mock.MagicMock(return_value=[torch.ones(4, 128)])

        meta_dict = {"default": _DummyAttnMeta()}
        empty_plan = v1_replay_plan.DecoderReplayPlan((), (0,), (0,), 0)
        with (
            mock.patch.object(
                v1_model.upstream,
                "get_forward_context",
                return_value=_DummyContext(attn_metadata=meta_dict),
            ),
            mock.patch.object(v1_model, "plan_decoder_replay", return_value=empty_plan),
        ):
            res = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res == 0).all()
            assert res.shape == (4, 128)

        # Plan is None with first is not None
        model.encoder_stage.return_value = (
            torch.zeros(4, 128),
            torch.zeros(4, 128),
        )
        model.decoder_stage.return_value = torch.ones(4, 128)
        with (
            mock.patch.object(
                v1_model.upstream,
                "get_forward_context",
                return_value=_DummyContext(attn_metadata=meta_dict),
            ),
            mock.patch.object(v1_model, "plan_decoder_replay", return_value=None),
        ):
            res3 = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res3 == 1).all()

        # Plan is None with first is None (non-dict metadata) and wrapped output list
        model.decoder_stage.return_value = [torch.ones(4, 128)]
        with mock.patch.object(
            v1_model.upstream,
            "get_forward_context",
            return_value=_DummyContext(attn_metadata=None),
        ):
            res4 = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res4 == 1).all()

    def test_model_forward_stage_and_pp_branches(self):
        model = _DummyV1Model()
        model.start_layer = 0
        model.end_layer = 4
        model._ced_boundary = 2
        model.use_mega_moe = True
        model.engram_hash = None
        model.use_sequence_parallel = False
        model.aux_hidden_state_layers = []
        model._mtp_hidden_buffer = None
        model.norm = lambda x: x
        model.embed_input_ids = mock.MagicMock(return_value=torch.ones(4, 128))

        layer = mock.MagicMock(
            return_value=(
                torch.ones(4, 128),
                None,
                None,
                None,
                torch.ones(4, 128),
                None,
            )
        )
        model.layers = [layer] * 4

        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            # stage="decoder"
            res_dec = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="decoder",
            )
            assert res_dec.shape == (4, 128)

            # stage="encoder" publishes global KV
            decoder_layer = mock.MagicMock()
            decoder_layer.attn_norm = lambda x: x
            decoder_layer.attn.write_encoder_global_kv = mock.MagicMock()
            model.layers[2] = decoder_layer
            hidden, pre_mix = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="encoder",
            )
            assert hidden.shape == (4, 128)
            assert pre_mix.shape == (4, 128)
            decoder_layer.attn.write_encoder_global_kv.assert_called_once()

            # First rank, inputs_embeds is None -> embed_input_ids
            model.end_layer = 1
            out = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=None,
                stage="full",
            )
            assert out.shape == (4, 128)
            model.embed_input_ids.assert_called_once()

            # Not first rank, not last rank -> IntermediateTensors
            mock_pp.return_value.is_first_rank = False
            mock_pp.return_value.is_last_rank = False
            interm_in = {
                "hidden_states": torch.ones(4, 128),
                "pre_mix": torch.ones(4, 128),
            }
            res_interm = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=interm_in,
                inputs_embeds=None,
                stage="full",
            )
            assert isinstance(res_interm, v1_model.upstream.IntermediateTensors)

    def test_forward_stage_engram_hash_branches(self):
        model = _DummyV1Model()
        model.start_layer = 0
        model.end_layer = 2
        model._ced_boundary = 1
        model.use_mega_moe = False
        model.use_sequence_parallel = False
        model.aux_hidden_state_layers = []
        model._mtp_hidden_buffer = None
        model.norm = lambda x: x
        model.engram_dp_shared_memory = False
        model.engram_swa_prefix = "swa"

        layer = mock.MagicMock(
            return_value=(
                torch.ones(4, 128),
                None,
                None,
                None,
                torch.ones(4, 128),
                None,
            )
        )
        layer.engram = mock.MagicMock(layer_hash_index=0)
        layer2 = mock.MagicMock(return_value=layer.return_value, engram=None)
        model.layers = [layer, layer2]

        model.engram_hash = mock.MagicMock()
        model.engram_hash.ensure_cache.return_value = True
        model.engram_hash.use_slot_cache = False
        model.engram_hash.lookback_depth = 4
        model.engram_hash.swa_cache_module.block_size = 16
        model.engram_hash.return_value = torch.zeros(4, 2, dtype=torch.int64)

        fwd_ctx = SimpleNamespace(
            attn_metadata={
                "swa": SimpleNamespace(
                    seq_lens=torch.tensor([4]),
                    query_start_loc=torch.tensor([0, 4]),
                )
            }
        )

        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(
                v1_model.upstream, "is_forward_context_available", return_value=True
            ),
            mock.patch.object(
                v1_model.upstream, "get_forward_context", return_value=fwd_ctx
            ),
            mock.patch.object(
                v1_model.upstream,
                "image_sentinel_mask",
                return_value=torch.zeros(4, dtype=torch.bool),
            ),
            mock.patch.object(
                v1_model.upstream,
                "gather_engram_hashes",
                return_value=torch.zeros(4, 2, dtype=torch.int64),
            ),
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
            mock.patch.object(
                cache_ops, "cache_slots", return_value=torch.zeros(4, dtype=torch.int32)
            ),
            mock.patch.object(cache_ops, "block_table", return_value=torch.zeros(2, 4)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            with pytest.raises(
                NotImplementedError, match="engram needs `lookback_token_ids`"
            ):
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )

            model.engram_hash.use_slot_cache = True
            v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                lookback_token_ids=None,
                stage="full",
            )
            layer.engram.prepare_embeddings.assert_called_once()

            fwd_ctx_list = SimpleNamespace(attn_metadata=[fwd_ctx.attn_metadata])
            with (
                mock.patch.object(
                    v1_model.upstream,
                    "get_forward_context",
                    return_value=fwd_ctx_list,
                ),
                mock.patch.object(
                    v1_model.upstream, "dbo_current_ubatch_id", return_value=0
                ),
            ):
                layer.engram.prepare_embeddings.reset_mock()
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )
                layer.engram.prepare_embeddings.assert_called_once()

            model.engram_hash.ensure_cache.return_value = False
            model.engram_hash.dummy_hashes.return_value = (
                torch.zeros(4, 2, dtype=torch.int64),
                torch.ones(4, dtype=torch.bool),
            )
            with mock.patch.object(
                v1_model.upstream, "get_engram_dp_size", return_value=2
            ):
                layer.engram.prepare_embeddings.reset_mock()
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )
                model.engram_hash.dummy_hashes.assert_called_once()

            model.engram_hash.ensure_cache.return_value = True
            v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                lookback_token_ids=torch.zeros(1, 4, dtype=torch.long),
                stage="full",
            )

            model.engram_hash.ensure_cache.return_value = False
            with mock.patch.object(
                v1_model.upstream, "get_engram_dp_size", return_value=1
            ):
                layer.engram.prepare_embeddings.reset_mock()
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )
                layer.engram.prepare_embeddings.assert_not_called()

    def test_forward_stage_sp_aux_and_layer_none(self):
        model = _DummyV1Model()
        model.start_layer = 0
        model.end_layer = 1
        model._ced_boundary = 1
        model.use_mega_moe = False
        model.engram_hash = None
        model.use_sequence_parallel = True
        model.aux_hidden_state_layers = [0, 1]
        model._mtp_hidden_buffer = torch.zeros(8, 128)
        model.norm = lambda x: x

        layer = mock.MagicMock(
            return_value=(
                torch.ones(4, 128),
                None,
                None,
                None,
                torch.ones(4, 128),
                torch.ones(4, 128),
            )
        )
        model.layers = [layer]
        fwd_ctx = SimpleNamespace(is_padding=torch.zeros(4, dtype=torch.bool))

        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(
                v1_model.upstream, "is_forward_context_available", return_value=True
            ),
            mock.patch.object(
                v1_model.upstream, "get_forward_context", return_value=fwd_ctx
            ),
            mock.patch.object(v1_model.upstream.envs, "VLLM_MOE_SKIP_PADDING", True),
            mock.patch.object(
                v1_model.upstream,
                "sp_padding_mask",
                return_value=torch.zeros(4, dtype=torch.bool),
            ),
            mock.patch.object(v1_model.upstream, "sp_shard", side_effect=lambda x: x),
            mock.patch.object(
                v1_model.upstream, "sp_all_gather", side_effect=lambda x: x
            ),
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            res, aux = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert res.shape == (4, 128)
            assert len(aux) == 2

            model._mtp_hidden_buffer = None
            model.aux_hidden_state_layers = []
            res2 = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert res2.shape == (4, 128)

            with mock.patch.object(
                v1_model.upstream.envs, "VLLM_MOE_SKIP_PADDING", False
            ):
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    stage="full",
                )

            model.use_sequence_parallel = False
            model.aux_hidden_state_layers = [0, 1]
            model._mtp_hidden_buffer = torch.zeros(8, 128)
            res4, aux4 = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert res4.shape == (4, 128)
            assert len(aux4) == 2

            # layer is None branch
            def mutating_layer(*args, **kwargs):
                import ctypes
                import inspect

                frame = inspect.currentframe().f_back
                frame.f_locals["layer"] = None
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
                return (
                    torch.ones(4, 128),
                    None,
                    None,
                    None,
                    torch.ones(4, 128),
                    None,
                )

            model.layers = [mutating_layer]
            model._mtp_hidden_buffer = None
            model.aux_hidden_state_layers = []
            out_none = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert out_none.shape == (4, 128)

    def test_finalize_mhc_init_and_causal_lm(self, monkeypatch):
        model = _DummyV1Model()
        layer0 = mock.MagicMock()
        layer0.hc_attn_fn.dtype = torch.bfloat16
        model.layers = [layer0]
        model.start_layer = 0
        model.end_layer = 1

        with mock.patch.object(v1_model, "pack_mix_weights") as mock_pack:
            v1_model.DeepseekV41Model.finalize_mhc_broadcast_weights(model)
            mock_pack.assert_not_called()

            layer0.hc_attn_fn.dtype = torch.float32
            with mock.patch.object(
                v1_model.upstream.DeepseekV4Model,
                "finalize_mhc_broadcast_weights",
            ):
                v1_model.DeepseekV41Model.finalize_mhc_broadcast_weights(model)
                mock_pack.assert_called_once()

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            compilation_config=SimpleNamespace(compile_sizes=[16, 32]),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
            speculative_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
        )

        def make_base_model():
            m = _DummyV1Model()
            m.config = SimpleNamespace(num_hidden_layers=4)
            m.layers = [mock.MagicMock() for _ in range(4)]
            for lyr in m.layers:
                lyr.attn = mock.MagicMock(compressor=None, window_size=64)
                lyr.attn.rotary_emb.cos_sin_cache = torch.zeros(10, 64)
                lyr.engram = None
            m.layers[2].attn.compressor = "compressor"
            m.end_layer = 4
            m.use_sequence_parallel = False
            return m

        with (
            mock.patch.object(cache_ops, "initialize_cache_ops"),
            mock.patch.object(v1_model, "initialize_delayed_mhc"),
            mock.patch.object(v1_model, "initialize_mhc_post"),
            mock.patch.object(compressor_ops, "initialize_compressor_ops"),
            mock.patch.object(v1_model.ced_ops, "initialize_ced_attention"),
            mock.patch.object(indexer_ops, "initialize_indexer"),
            mock.patch.object(v1_model, "compact_rope_caches"),
            mock.patch.object(
                v1_model.upstream.DeepseekV4Model, "__init__", return_value=None
            ),
            mock.patch.object(
                v1_model, "DeepseekV41EncoderStage", return_value=mock.MagicMock()
            ) as mock_enc_cls,
            mock.patch.object(
                v1_model, "DeepseekV41DecoderStage", return_value=mock.MagicMock()
            ) as mock_dec_cls,
            mock.patch.object(
                v1_model, "DeepseekV41FullStage", return_value=mock.MagicMock()
            ) as mock_full_cls,
        ):
            monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", True)
            m1 = make_base_model()
            m1.use_sequence_parallel = True
            with pytest.raises(NotImplementedError, match="unsharded token rows"):
                v1_model.DeepseekV41Model.__init__(m1, vllm_config=vllm_config)

            m2 = make_base_model()
            m2.layers[3].attn.compressor = "compressor"
            with pytest.raises(NotImplementedError, match="one decoder KV source"):
                v1_model.DeepseekV41Model.__init__(m2, vllm_config=vllm_config)

            m3 = make_base_model()
            m3.layers[2].engram = "engram"
            with pytest.raises(
                NotImplementedError, match="Engram layers in the encoder"
            ):
                v1_model.DeepseekV41Model.__init__(m3, vllm_config=vllm_config)

            m4 = make_base_model()
            v1_model.DeepseekV41Model.__init__(m4, vllm_config=vllm_config)
            assert m4.encoder_stage is mock_enc_cls.return_value
            assert m4.decoder_stage is mock_dec_cls.return_value

            monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", False)
            m5 = make_base_model()
            v1_model.DeepseekV41Model.__init__(m5, vllm_config=vllm_config)
            assert m5.full_stage is mock_full_cls.return_value

            monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", True)
            vllm_config.cache_config.enable_prefix_caching = True
            m6 = make_base_model()
            v1_model.DeepseekV41Model.__init__(m6, vllm_config=vllm_config)
            assert m6._ced_enabled is False
            assert m6.full_stage is mock_full_cls.return_value

        # DeepseekV41ForCausalLM tests
        vllm_config.parallel_config.pipeline_parallel_size = 2
        with pytest.raises(
            NotImplementedError, match="V4.1 TPU currently requires PP1 and DP1"
        ):
            v1_model.DeepseekV41ForCausalLM(vllm_config=vllm_config)

        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.speculative_config = "spec"
        with pytest.raises(
            NotImplementedError, match="synchronous scheduling without speculation"
        ):
            v1_model.DeepseekV41ForCausalLM(vllm_config=vllm_config)

        vllm_config.speculative_config = None
        with (
            mock.patch.object(
                v1_model.upstream.DeepseekV41LLMForCausalLM,
                "__init__",
                return_value=None,
            ),
            mock.patch.object(
                v1_model.upstream,
                "_make_deepseek_v4_weights_mapper",
                return_value="mapper",
            ) as mock_mapper,
            mock.patch.object(
                v1_model.upstream.DeepseekV41LLMForCausalLM,
                "forward",
                return_value="lm_out",
            ) as mock_super,
        ):
            lm = object.__new__(v1_model.DeepseekV41ForCausalLM)
            lm.config = SimpleNamespace(expert_dtype="fp4")
            v1_model.DeepseekV41ForCausalLM.__init__(lm, vllm_config=vllm_config)
            assert lm.hf_to_vllm_mapper == "mapper"
            mock_mapper.assert_called_once()

            lm.model = mock.MagicMock()
            lm.model.engram_hash = mock.MagicMock()
            assert v1_model.DeepseekV41ForCausalLM.forward(lm, "arg") == "lm_out"
            lm.model.engram_hash.ensure_cache.assert_called_once()
            mock_super.assert_called_once()

            lm.model.engram_hash = None
            assert v1_model.DeepseekV41ForCausalLM.forward(lm, "arg2") == "lm_out"
