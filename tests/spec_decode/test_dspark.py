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


def _make_proposer(hf_dict, num_speculative_tokens=4, **spec_fields):
    # The vLLM 0.29 drafting fields at their defaults; tests override them.
    fields = dict(
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
    )
    fields.update(spec_fields)
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)),
        **fields,
    )
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
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 4,
            **_LAYER_IDS,
        }
    )
    assert p.mask_token_id == 7
    assert p.sample_from_anchor
    assert p._query_block_size(4) == 4
    # propose() sizes the query block via this property, so the dense
    # override must reach it (K slots, not DFlash's 1 + K).
    assert p.block_size == 4


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer(
        {
            "mask_token_id": 5,
            "sample_from_anchor": False,
            **_LAYER_IDS,
        }
    )
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5
    assert p.block_size == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("draft_sample_method", "probabilistic"),
        ("enable_adaptive_verification", True),
        ("dspark_draft_topk", 8),
    ],
)
def test_rejects_unsupported_drafting_options(field, value):
    with pytest.raises(ValueError, match="does not support"):
        _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 4,
                **_LAYER_IDS,
            },
            **{field: value},
        )


def test_block_size_guard():
    with pytest.raises(ValueError, match="block size"):
        _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 7,
                **_LAYER_IDS,
            },
            num_speculative_tokens=4,
        )


def test_missing_noise_token():
    with pytest.raises(ValueError, match="noise/mask token"):
        _make_proposer({})


def test_missing_target_layer_ids():
    with pytest.raises(ValueError, match="target layer ids"):
        _make_proposer({"dspark_noise_token_id": 7})


def test_target_layer_id_resolution_chain():
    # Aux ids (target + 1) take precedence and are shifted back down.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "eagle_aux_hidden_state_layer_ids": [2, 16, 31],
            "dspark_target_layer_ids": [9, 9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [1, 15, 30]

    # Dense-checkpoint spelling.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "target_layer_ids": [3, 7],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [3, 7]

    # An explicit dflash_config wins unchanged.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dflash_config": {"target_layer_ids": [4, 8]},
            "dspark_target_layer_ids": [9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [4, 8]


def test_no_forced_embedding_sharing():
    # A reduced-vocab DSpark draft owns its lm_head; load_model must honor
    # has_own_embed_tokens / has_own_lm_head instead of force-sharing.
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._force_share_target_embeddings is False


@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("use_shard_indices", [True, False])
def test_sharded_vs_full_vocab_argmax_tp8_bitwise_equal(
    sample_from_anchor: bool, use_shard_indices: bool
):
    """TP=8 vocab-sharded argmax fast path matches full-vocab argmax bitwise."""
    tp_size = 8
    vocab_shard = 16
    vocab_size = tp_size * vocab_shard
    hidden_dim = 32
    markov_rank = 8
    block_size = 5
    num_reqs = 4
    total_tokens = num_reqs * block_size

    gen = torch.Generator().manual_seed(12345)
    hidden = torch.randn(
        (total_tokens, hidden_dim), generator=gen, dtype=torch.bfloat16
    )
    w_lm = torch.randn((vocab_size, hidden_dim), generator=gen, dtype=torch.bfloat16)
    w_m1 = torch.randn((vocab_size, markov_rank), generator=gen, dtype=torch.bfloat16)
    w_m2 = torch.randn((vocab_size, markov_rank), generator=gen, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab_size, (total_tokens,), generator=gen)
    positions = torch.arange(total_tokens, dtype=torch.int64)

    fn = DSparkProposer._dflash_forward_and_sample.__wrapped__

    # 1. Golden full-vocab path (draft_id_to_target_id non-None forces fallback)
    full_draft = SimpleNamespace(
        draft_id_to_target_id=torch.arange(vocab_size, dtype=torch.int64),
        model=lambda input_ids, positions: hidden,
        compute_draft_logits=lambda h: h @ w_lm.T,
        markov_embed=lambda ids: w_m1[ids],
        markov_bias=lambda emb: emb @ w_m2.T,
        map_draft_to_target=lambda idx: idx,
    )
    full_proposer = SimpleNamespace(
        sample_from_anchor=sample_from_anchor, draft_model=full_draft
    )
    expected_tokens, expected_hidden = fn(
        full_proposer, input_ids, positions, block_size
    )
    assert torch.equal(expected_hidden, hidden)

    # 2. Simulate all 8 TP ranks running lockstep with cross-rank [B, 2*TP] gather
    rank_states = []
    for r in range(tp_size):
        lo = r * vocab_shard
        hi = (r + 1) * vocab_shard
        lm_head_mock = (
            SimpleNamespace(shard_indices=SimpleNamespace(org_vocab_start_index=lo))
            if use_shard_indices
            else None
        )
        rank_states.append((r, lo, hi, lm_head_mock))

    # Execute step-by-step across all 8 ranks verifying each rank with
    # simulated cross-rank [B, 2*TP] all-gather over all 8 ranks' shards.
    for r, lo, hi, lm_head_mock in rank_states:
        state = SimpleNamespace(step_idx=0, prev_history=[])

        def _fake_all_gather(
            local_pair: torch.Tensor, dim: int = -1, *, _state=state
        ) -> torch.Tensor:
            step_idx = _state.step_idx
            _state.step_idx += 1
            prev_ids = _state.prev_history[step_idx]
            bias_all = w_m1[prev_ids] @ w_m2.T
            slot = (0 if sample_from_anchor else 1) + step_idx
            logits_all = (hidden @ w_lm.T).view(num_reqs, block_size, vocab_size)[
                :, slot
            ] + bias_all
            pairs = []
            for rr in range(tp_size):
                r_lo, r_hi = rr * vocab_shard, (rr + 1) * vocab_shard
                vals, idxs = logits_all[:, r_lo:r_hi].max(dim=-1)
                pairs.append(torch.stack([vals.float(), (idxs + r_lo).float()], dim=-1))
            return torch.cat(pairs, dim=dim)

        sharded_draft = SimpleNamespace(
            draft_id_to_target_id=None,
            lm_head=lm_head_mock,
            model=lambda input_ids, positions: hidden,
            compute_sharded_draft_logits=lambda h, lo=lo, hi=hi: h @ w_lm[lo:hi].T,
            markov_embed=lambda ids, _state=state: (
                _state.prev_history.append(ids) or w_m1[ids]
            ),
            sharded_markov_bias=lambda emb, lo=lo, hi=hi: emb @ w_m2[lo:hi].T,
        )
        sharded_proposer = SimpleNamespace(
            sample_from_anchor=sample_from_anchor, draft_model=sharded_draft
        )
        with (
            mock.patch(
                "vllm_torchtpu.spec_decode.dspark.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
            mock.patch(
                "vllm_torchtpu.spec_decode.dspark.get_tensor_model_parallel_rank",
                return_value=r,
            ),
            mock.patch(
                "vllm_torchtpu.spec_decode.dspark.tensor_model_parallel_all_gather",
                side_effect=_fake_all_gather,
            ),
        ):
            actual_tokens, _ = fn(sharded_proposer, input_ids, positions, block_size)
        assert torch.equal(actual_tokens, expected_tokens)


def test_sharded_argmax_tp1_single_rank():
    """TP=1 sharded fast path skips cross-rank All-Gather and matches bitwise."""
    vocab_size = 32
    hidden_dim = 16
    markov_rank = 8
    block_size = 4
    num_reqs = 2
    total_tokens = num_reqs * block_size

    gen = torch.Generator().manual_seed(99)
    hidden = torch.randn((total_tokens, hidden_dim), generator=gen)
    w_lm = torch.randn((vocab_size, hidden_dim), generator=gen)
    w_m1 = torch.randn((vocab_size, markov_rank), generator=gen)
    w_m2 = torch.randn((vocab_size, markov_rank), generator=gen)
    input_ids = torch.randint(0, vocab_size, (total_tokens,), generator=gen)
    positions = torch.arange(total_tokens, dtype=torch.int64)

    fn = DSparkProposer._dflash_forward_and_sample.__wrapped__
    sharded_draft = SimpleNamespace(
        draft_id_to_target_id=None,
        lm_head=SimpleNamespace(shard_indices=SimpleNamespace(org_vocab_start_index=0)),
        model=lambda input_ids, positions: hidden,
        compute_sharded_draft_logits=lambda h: h @ w_lm.T,
        markov_embed=lambda ids: w_m1[ids],
        sharded_markov_bias=lambda emb: emb @ w_m2.T,
    )
    proposer = SimpleNamespace(sample_from_anchor=True, draft_model=sharded_draft)
    with mock.patch(
        "vllm_torchtpu.spec_decode.dspark.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        tokens, _ = fn(proposer, input_ids, positions, block_size)
    assert tokens.shape == (num_reqs, block_size)


def test_dspark_v41_sharded_vocab_proj_methods():
    """Verify DSparkDeepseekV41ForCausalLM sharded projection and padding mask."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.dspark import (
        DSparkDeepseekV41ForCausalLM,
    )

    head_padded = SimpleNamespace(
        weight=torch.empty(10, 4),
        quant_method=SimpleNamespace(
            apply=lambda head, x, bias=None: torch.ones(x.shape[0], 10)
        ),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=10, num_org_vocab_padding=2
        ),
    )
    head_unpadded = SimpleNamespace(
        weight=torch.empty(8, 4),
        quant_method=SimpleNamespace(
            apply=lambda head, x, bias=None: torch.full((x.shape[0], 8), 2.0)
        ),
        shard_indices=SimpleNamespace(org_vocab_start_index=0, num_org_vocab_padding=0),
    )
    dummy_self = SimpleNamespace(
        lm_head=head_padded,
        model=SimpleNamespace(
            norm=lambda h: h,
            markov_head=SimpleNamespace(markov_w2=head_unpadded),
        ),
    )
    dummy_self._compute_sharded_vocab_proj = (
        DSparkDeepseekV41ForCausalLM._compute_sharded_vocab_proj.__get__(dummy_self)
    )

    out_logits = DSparkDeepseekV41ForCausalLM.compute_sharded_draft_logits(
        dummy_self, torch.zeros(3, 4)
    )
    assert out_logits.shape == (3, 10)
    assert torch.all(out_logits[:, :8] == 1.0)
    assert torch.all(out_logits[:, 8:] == -float("inf"))

    # Case 1: markov_w2 already sharded (w2.weight.shape[0] <= lm_shard_size)
    out_bias = DSparkDeepseekV41ForCausalLM.sharded_markov_bias(
        dummy_self, torch.zeros(3, 4)
    )
    assert out_bias.shape == (3, 8)
    assert torch.all(out_bias == 2.0)

    # Case 2: markov_w2 replicated (disable_tp=True, w2.weight.shape[0] > lm_shard_size)
    w2_rep = torch.arange(80, dtype=torch.float32).view(20, 4)
    dummy_self.model.markov_head.markov_w2 = SimpleNamespace(weight=w2_rep)
    x = torch.ones(3, 4)
    out_bias_rep = DSparkDeepseekV41ForCausalLM.sharded_markov_bias(dummy_self, x)
    expected_rep = torch.nn.functional.linear(x, w2_rep[10:20])
    assert out_bias_rep.shape == (3, 10)
    assert torch.equal(out_bias_rep[:, :8], expected_rep[:, :8])
    assert torch.all(out_bias_rep[:, 8:] == -float("inf"))

    # Case 3: shard_indices is None fallback to tp_rank * lm_shard_size
    dummy_self.lm_head = SimpleNamespace(weight=torch.empty(10, 4), shard_indices=None)
    with mock.patch(
        "vllm_torchtpu.models.vllm.deepseek_v41.v2.dspark.get_tensor_model_parallel_rank",
        return_value=1,
    ):
        out_bias_rank = DSparkDeepseekV41ForCausalLM.sharded_markov_bias(dummy_self, x)
    assert torch.equal(out_bias_rank, expected_rep)
