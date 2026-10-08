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


class _FakeDSparkDraftModel:
    def __init__(
        self, hidden: torch.Tensor, lm_weight: torch.Tensor, markov_table: torch.Tensor
    ):
        self._hidden = hidden
        self._lm_weight = lm_weight
        self._markov_table = markov_table
        self.last_logits_input_shape: tuple[int, ...] | None = None

    def model(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del input_ids, positions
        return self._hidden

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.last_logits_input_shape = tuple(hidden_states.shape)
        return hidden_states @ self._lm_weight.T

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._markov_table[token_ids]

    def markov_bias(self, embed: torch.Tensor) -> torch.Tensor:
        return embed

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + 2


@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("pad_tokens", [0, 3])
def test_dflash_forward_and_sample_slot_major_parity(
    sample_from_anchor: bool, pad_tokens: int
):
    num_spec = 5
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": num_spec,
            "sample_from_anchor": sample_from_anchor,
            **_LAYER_IDS,
        },
        num_speculative_tokens=num_spec,
    )
    block_size = p.block_size
    padded_num_reqs = 4
    total_tokens = padded_num_reqs * block_size + pad_tokens
    hidden_size = 8
    draft_vocab = 16
    target_vocab = 24

    gen = torch.Generator().manual_seed(42 + pad_tokens + int(sample_from_anchor))
    hidden = torch.randn(total_tokens, hidden_size, generator=gen)
    lm_weight = torch.randn(draft_vocab, hidden_size, generator=gen)
    markov_table = torch.randn(target_vocab, draft_vocab, generator=gen)
    input_ids = torch.randint(0, 10, (total_tokens,), generator=gen, dtype=torch.int32)
    positions = torch.arange(total_tokens, dtype=torch.int32)

    p.draft_model = _FakeDSparkDraftModel(hidden, lm_weight, markov_table)
    fn = getattr(
        DSparkProposer._dflash_forward_and_sample,
        "__wrapped__",
        DSparkProposer._dflash_forward_and_sample,
    )
    draft_tokens_chunk, out_hidden = fn(p, input_ids, positions, block_size)

    assert torch.equal(out_hidden, hidden)
    assert p.draft_model.last_logits_input_shape == (total_tokens, hidden_size)

    # Reference request-major calculation.
    valid_hidden = hidden[: padded_num_reqs * block_size]
    ref_logits_3d = (valid_hidden @ lm_weight.T).view(
        padded_num_reqs, block_size, draft_vocab
    )
    prev = input_ids[: padded_num_reqs * block_size].view(padded_num_reqs, block_size)[
        :, 0
    ]
    ref_steps = []
    sample_start = 0 if sample_from_anchor else 1
    for i in range(sample_start, block_size):
        bias = markov_table[prev]
        step = (ref_logits_3d[:, i] + bias).argmax(dim=-1) + 2
        ref_steps.append(step)
        prev = step
    ref_chunk = torch.stack(ref_steps, dim=1)
    assert torch.equal(draft_tokens_chunk, ref_chunk)


def test_load_model_validates_dspark_interface(monkeypatch):
    from vllm_torchtpu.spec_decode.dflash import DFlashProposer

    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    monkeypatch.setattr(DFlashProposer, "load_model", lambda self, tm: None)
    p.draft_model = SimpleNamespace(
        compute_draft_logits=lambda x: x,
        markov_embed=lambda x: x,
        markov_bias=lambda x: x,
        map_draft_to_target=lambda x: x,
        has_own_lm_head=True,
        has_own_embed_tokens=True,
    )
    p.load_model(target_model=object())

    p.draft_model = SimpleNamespace(compute_draft_logits=lambda x: x)
    with pytest.raises(RuntimeError, match="DSpark draft model lacks markov_embed"):
        p.load_model(target_model=object())
