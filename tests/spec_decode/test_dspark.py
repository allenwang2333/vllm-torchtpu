# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.spec_decode.dspark import DSparkProposer
from vllm_torchtpu.models.vllm.kimi_k3.dspark import K3DSparkModel


class _FakeHFConfig:
    """to_dict() reflects attribute writes, like a transformers config."""

    def __init__(self, d):
        self._d = dict(d)

    def to_dict(self):
        out = dict(self._d)
        if hasattr(self, "dflash_config"):
            out["dflash_config"] = self.dflash_config
        return out


def _make_proposer(hf_dict, num_speculative_tokens=4):
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)))
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DSparkProposer(runner=runner, vllm_config=vllm_config)


_LAYER_IDS = {"dspark_target_layer_ids": [1, 15, 30]}


def test_noise_resolution_and_dense_block():
    p = _make_proposer({
        "dspark_noise_token_id": 7,
        "dspark_block_size": 4,
        **_LAYER_IDS,
    })
    assert p.mask_token_id == 7
    assert p.sample_from_anchor
    assert p._query_block_size(4) == 4
    # propose() sizes the query block via this property, so the dense
    # override must reach it (K slots, not DFlash's 1 + K).
    assert p.block_size == 4


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer({
        "mask_token_id": 5,
        "sample_from_anchor": False,
        **_LAYER_IDS,
    })
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5
    assert p.block_size == 5


def test_block_size_guard():
    with pytest.raises(ValueError, match="block size"):
        _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 7,
                **_LAYER_IDS,
            },
            num_speculative_tokens=4)


def test_missing_noise_token():
    with pytest.raises(ValueError, match="noise/mask token"):
        _make_proposer({})


def test_missing_target_layer_ids():
    with pytest.raises(ValueError, match="target layer ids"):
        _make_proposer({"dspark_noise_token_id": 7})


def test_target_layer_id_resolution_chain():
    # Aux ids (target + 1) take precedence and are shifted back down.
    p = _make_proposer({
        "dspark_noise_token_id": 7,
        "eagle_aux_hidden_state_layer_ids": [2, 16, 31],
        "dspark_target_layer_ids": [9, 9, 9],
    })
    assert p.dflash_config["target_layer_ids"] == [1, 15, 30]

    # Dense-checkpoint spelling.
    p = _make_proposer({
        "dspark_noise_token_id": 7,
        "target_layer_ids": [3, 7],
    })
    assert p.dflash_config["target_layer_ids"] == [3, 7]

    # An explicit dflash_config wins unchanged.
    p = _make_proposer({
        "dspark_noise_token_id": 7,
        "dflash_config": {
            "target_layer_ids": [4, 8]
        },
        "dspark_target_layer_ids": [9, 9],
    })
    assert p.dflash_config["target_layer_ids"] == [4, 8]


def test_no_forced_embedding_sharing():
    # A reduced-vocab DSpark draft owns its lm_head; load_model must honor
    # has_own_embed_tokens / has_own_lm_head instead of force-sharing.
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._force_share_target_embeddings is False


def test_k3_context_rope_matches_mla_cache_dtype():
    context = torch.ones((2, 4), dtype=torch.bfloat16)

    class _Core:

        def __init__(self):
            self.kv_cache = torch.empty((1,), dtype=torch.bfloat16)
            self.k_pe = None

        def mla_op(self, _cache, _q_nope, _q_pe, _kv_c, k_pe, *_metadata):
            self.k_pe = k_pe

    core = _Core()
    wrapper = SimpleNamespace(
        kv_a_layernorm=lambda value: value,
        rotary_emb=lambda _positions, q_pe, k_pe: (q_pe.float(),
                                                   k_pe.float()),
        num_heads=1,
        mla_attn=core,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(kv_lora_rank=2, qk_rope_head_dim=2),
        layers=[
            SimpleNamespace(self_attn=SimpleNamespace(mla_attn=wrapper))
        ],
        combine_hidden_states=lambda _hidden_states: context,
        context_kv_proj=lambda _context: (
            torch.ones((2, 4), dtype=torch.bfloat16),
            None,
        ),
    )
    metadata = [
        SimpleNamespace(
            seq_lens=torch.tensor([2]),
            block_tables=torch.tensor([[0]]),
            query_start_loc=torch.tensor([0, 2]),
            request_distribution=torch.tensor([0]),
        )
    ]

    K3DSparkModel.precompute_and_store_context_kv(
        model,
        target_hidden_states=context,
        positions=torch.tensor([0, 1]),
        metadata=metadata,
    )

    assert core.k_pe is not None
    assert core.k_pe.dtype == core.kv_cache.dtype
