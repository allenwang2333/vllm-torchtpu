# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.spec_decode.dspark import DSparkProposer


class _FakeHFConfig:
    """to_dict() reflects attribute writes, like a transformers config."""

    def __init__(self, d):
        self._d = dict(d)

    def to_dict(self):
        out = dict(self._d)
        if hasattr(self, "dflash_config"):
            out["dflash_config"] = self.dflash_config
        return out


def _make_proposer(hf_dict,
                   num_speculative_tokens=4,
                   draft_sample_method="greedy",
                   async_scheduling=False):
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_sample_method=draft_sample_method,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)))
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
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


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer({
        "mask_token_id": 5,
        "dspark_bonus_anchor": True,
        **_LAYER_IDS,
    })
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5


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


_BASE = {"dspark_noise_token_id": 7, **_LAYER_IDS}


def test_greedy_mode_has_no_sampling_state():
    p = _make_proposer(_BASE)
    assert p.draft_sample_probabilistic is False
    assert p.draft_logits_cache is None
    assert p._chunk_sample_args(SimpleNamespace(num_reqs=1, start_index=0),
                                8) == ()
    assert p._dummy_sample_args(8) == ()


def test_probabilistic_mode_flag_and_dummy_args():
    p = _make_proposer(_BASE, draft_sample_method="probabilistic")
    assert p.draft_sample_probabilistic is True
    # Buffers appear at load_model; give the proposer a small fake cache and
    # check the dummy args mirror the compiled signature (seeds int64, temps
    # fp32, rows aimed at the trash row).
    p.draft_logits_cache = torch.zeros(5, 4, 16)
    seeds, temps, rows = p._dummy_sample_args(8)
    assert seeds.dtype == torch.int64 and seeds.shape == (8, )
    assert temps.dtype == torch.float32 and (temps == 1.0).all()
    assert rows.dtype == torch.int64 and (rows == 4).all()


def test_probabilistic_rejects_async_scheduling():
    with pytest.raises(NotImplementedError, match="async"):
        _make_proposer(_BASE,
                       draft_sample_method="probabilistic",
                       async_scheduling=True)
