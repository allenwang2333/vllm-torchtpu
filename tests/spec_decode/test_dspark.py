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
@pytest.mark.parametrize("padded_num_reqs,pad_tokens", [(4, 0), (4, 3), (8, 0)])
def test_dflash_forward_and_sample_slot_major_parity(
    sample_from_anchor: bool, padded_num_reqs: int, pad_tokens: int
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
    total_tokens = padded_num_reqs * block_size + pad_tokens
    hidden_size = 8
    draft_vocab = 16
    target_vocab = 24

    gen = torch.Generator().manual_seed(
        42 + padded_num_reqs + pad_tokens + int(sample_from_anchor)
    )
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
    sample_start = 0 if sample_from_anchor else 1
    num_sampled_tokens = padded_num_reqs * (block_size - sample_start)
    expected_pad = (
        (-num_sampled_tokens) % 8
        if sample_start > 0 and total_tokens % 8 == 0
        else pad_tokens
    )
    assert p.draft_model.last_logits_input_shape == (
        num_sampled_tokens + expected_pad,
        hidden_size,
    )

    # Reference request-major calculation.
    valid_hidden = hidden[: padded_num_reqs * block_size]
    ref_logits_3d = (valid_hidden @ lm_weight.T).view(
        padded_num_reqs, block_size, draft_vocab
    )
    prev = input_ids[: padded_num_reqs * block_size].view(padded_num_reqs, block_size)[
        :, 0
    ]
    ref_steps = []
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


# --- TP-sharded greedy Markov sampling --------------------------------------

_DSPARK_HF = {"dspark_noise_token_id": 7, "dspark_block_size": 4, **_LAYER_IDS}


def _shard_indices(org_vocab: int, tp_rank: int, tp_size: int, padding: int = 64):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
        pad_vocab_size,
    )

    padded = pad_vocab_size(org_vocab, padding)
    return VocabParallelEmbedding._get_indices(
        padded, padded, org_vocab, org_vocab, tp_rank, tp_size
    ), padded


class _ShardedDraft:
    """Traceable stand-in for a DSpark draft whose lm_head is TP-sharded.

    ``lm_head_weight`` is the full [padded_vocab, hidden] head (pad rows zero)
    and ``w1`` / ``w2`` the replicated Markov embeddings; this rank's shard of
    the base logits is what ``compute_draft_logits_local`` returns.
    """

    has_own_lm_head = False
    has_own_embed_tokens = False

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        org_vocab: int,
        tp_rank: int,
        tp_size: int,
        head_dtype: torch.dtype | None = None,
    ):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        indices, padded = _shard_indices(org_vocab, tp_rank, tp_size)
        width = padded // tp_size
        self._shard = lm_head_weight[
            indices.padded_org_vocab_start_index : indices.padded_org_vocab_end_index
        ]
        self.lm_head = SimpleNamespace(
            shard_indices=indices,
            tp_size=tp_size,
            num_embeddings_per_partition=width,
            num_embeddings_padded=padded,
            org_vocab_size=org_vocab,
        )
        self.model = SimpleNamespace(
            markov_head=SimpleNamespace(
                markov_w2=SimpleNamespace(
                    weight=w2,
                    tp_size=1,
                    quant_method=UnquantizedEmbeddingMethod(),
                )
            )
        )
        self._w1 = w1
        self._w2 = w2
        self._org_vocab = org_vocab
        self.logits_processor = SimpleNamespace(head_dtype=head_dtype)

    def compute_draft_logits_local(self, hidden):
        return hidden @ self._shard.t()

    def markov_embed(self, prev):
        return self._w1[prev]

    def markov_bias(self, embed):
        return (embed @ self._w2.t())[:, : self._org_vocab]

    @staticmethod
    def map_draft_to_target(ids):
        return ids


def _lockstep_all_gather(tp_size: int):
    """Barrier-synchronized stand-in for the TP all-gather across threads."""
    import threading

    barrier = threading.Barrier(tp_size)
    slots: dict[int, torch.Tensor] = {}
    ranks: dict[int, int] = {}
    lock = threading.Lock()

    def register(rank: int) -> None:
        with lock:
            ranks[threading.get_ident()] = rank

    def all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        rank = ranks[threading.get_ident()]
        with lock:
            slots[rank] = x.clone()
        barrier.wait()
        out = torch.cat([slots[r] for r in range(tp_size)], dim=dim)
        barrier.wait()
        return out

    return register, all_gather


def _reference_markov_chain(logits_full, w1, w2, ids_3d, org_vocab, sample_start):
    """The stock full-vocab chain, in the same arithmetic order."""
    block_size = ids_3d.shape[1]
    prev = ids_3d[:, 0]
    out = []
    for i in range(sample_start, block_size):
        bias = (w1[prev] @ w2.t())[:, :org_vocab]
        step = (logits_full[:, i] + bias).argmax(dim=-1)
        out.append(step)
        prev = step
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("org_vocab,tp_size", [(100, 4), (128, 2), (200, 8)])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_sharded_markov_matches_full_vocab_chain(
    org_vocab, tp_size, sample_from_anchor, monkeypatch
):
    import threading

    from vllm_torchtpu.spec_decode import dspark as dspark_mod

    torch.manual_seed(org_vocab * 7 + tp_size)
    hidden, rank_r, num_reqs, K = 8, 4, 5, 4
    block_size = K if sample_from_anchor else K + 1
    _, padded = _shard_indices(org_vocab, 0, tp_size)
    # Coarse values force many exact ties so first-index tie-breaking is
    # exercised both within a shard and across shards.
    lm_head_weight = torch.randint(-2, 3, (padded, hidden)).float()
    lm_head_weight[org_vocab:] = 0
    w1 = torch.randint(-1, 2, (org_vocab, rank_r)).float()
    w2 = torch.randint(-2, 3, (padded, rank_r)).float()
    h = torch.randint(-1, 2, (num_reqs * block_size, hidden)).float()
    ids_3d = torch.randint(0, org_vocab, (num_reqs, block_size))
    sample_start = 0 if sample_from_anchor else 1

    logits_full = (h @ lm_head_weight.t())[:, :org_vocab].view(
        num_reqs, block_size, org_vocab
    )
    expected = _reference_markov_chain(
        logits_full, w1, w2, ids_3d, org_vocab, sample_start
    )

    register, fake_all_gather = _lockstep_all_gather(tp_size)
    monkeypatch.setattr(dspark_mod, "tensor_model_parallel_all_gather", fake_all_gather)
    hf = dict(_DSPARK_HF, dspark_block_size=K, sample_from_anchor=sample_from_anchor)

    results: dict[int, torch.Tensor] = {}
    errors: list[BaseException] = []

    def run(rank: int) -> None:
        try:
            register(rank)
            p = _make_proposer(hf, num_speculative_tokens=K)
            p._draft_replicated = False
            p.draft_model = _ShardedDraft(
                lm_head_weight, w1, w2, org_vocab, rank, tp_size
            )
            p._init_sharded_markov_sampling()
            assert p._sharded_markov_tp == tp_size
            if padded != org_vocab:
                assert p._vocab_shard_mask is not None
            else:
                assert p._vocab_shard_mask is None
            tokens = p._sample_markov_sharded(h, ids_3d, block_size, sample_start)
            results[rank] = torch.stack(tokens, dim=1)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)
            raise

    threads = [threading.Thread(target=run, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for rank in range(tp_size):
        assert torch.equal(results[rank], expected), rank
    assert expected.dtype == results[0].dtype == torch.int64


def test_sharded_markov_padding_mask_and_w2_slice(monkeypatch):
    org_vocab, tp_size, rank_r = 100, 4, 4
    _, padded = _shard_indices(org_vocab, 0, tp_size)
    w2 = torch.randn(padded, rank_r)
    for rank in range(tp_size):
        p = _make_proposer(_DSPARK_HF)
        p._draft_replicated = False
        p.draft_model = _ShardedDraft(
            torch.zeros(padded, 8),
            torch.zeros(org_vocab, rank_r),
            w2,
            org_vocab,
            rank,
            tp_size,
            head_dtype=torch.bfloat16,
        )
        p._init_sharded_markov_sampling()
        width = padded // tp_size
        assert p._vocab_shard_offset.item() == rank * width
        assert p._markov_w2_local.dtype == torch.bfloat16
        assert torch.equal(
            p._markov_w2_local,
            w2[rank * width : (rank + 1) * width].to(torch.bfloat16),
        )
        mask = p._vocab_shard_mask
        assert mask is not None and mask.shape == (width,)
        valid = max(0, min(org_vocab - rank * width, width))
        assert torch.all(mask[:valid] == 0)
        assert torch.all(torch.isinf(mask[valid:]))

    # A Markov head shorter than the lm_head partition is zero-padded.
    p = _make_proposer(_DSPARK_HF)
    p._draft_replicated = False
    p.draft_model = _ShardedDraft(
        torch.zeros(padded, 8),
        torch.zeros(org_vocab, rank_r),
        w2[:org_vocab],
        org_vocab,
        tp_size - 1,
        tp_size,
    )
    p._init_sharded_markov_sampling()
    width = padded // tp_size
    assert p._markov_w2_local.shape == (width, rank_r)
    assert torch.all(p._markov_w2_local[org_vocab - (tp_size - 1) * width :] == 0)


def test_sharded_markov_fallback_reasons(monkeypatch):
    org_vocab, tp_size = 64, 2

    def fresh():
        p = _make_proposer(_DSPARK_HF)
        p._draft_replicated = False
        p.draft_model = _ShardedDraft(
            torch.zeros(org_vocab, 8),
            torch.zeros(org_vocab, 4),
            torch.zeros(org_vocab, 4),
            org_vocab,
            0,
            tp_size,
        )
        return p

    p = fresh()
    p._init_sharded_markov_sampling()
    assert p._sharded_markov_tp == tp_size

    p = fresh()
    p._draft_replicated = True
    assert "replicated" in p._sharded_markov_unsupported_reason(p.draft_model)

    p = fresh()
    assert "compute_draft_logits_local" in p._sharded_markov_unsupported_reason(
        SimpleNamespace(lm_head=p.draft_model.lm_head)
    )

    p = fresh()
    p.draft_model.lm_head.tp_size = 1
    assert "ParallelLMHead" in p._sharded_markov_unsupported_reason(p.draft_model)

    p = fresh()
    p.draft_model.lm_head.shard_indices.padded_added_vocab_end_index += 64
    assert "added vocab" in p._sharded_markov_unsupported_reason(p.draft_model)

    p = fresh()
    p.draft_model.lm_head.num_embeddings_padded = 1 << 24
    assert "float32" in p._sharded_markov_unsupported_reason(p.draft_model)

    p = fresh()
    p.draft_model.model.markov_head.markov_w2.quant_method = object()
    assert "markov_w2" in p._sharded_markov_unsupported_reason(p.draft_model)

    p = fresh()
    p.draft_model.model.markov_head.markov_w2.tp_size = 2
    assert "markov_w2" in p._sharded_markov_unsupported_reason(p.draft_model)

    p = fresh()
    p.draft_model.logits_processor = None
    p.draft_model.model.logits_processor = None
    assert "LogitsProcessor" in p._sharded_markov_unsupported_reason(p.draft_model)
    p._init_sharded_markov_sampling()
    assert p._sharded_markov_tp == 1 and p._markov_w2_local is None


def test_select_gathered_candidate_prefers_lowest_shard_on_ties():
    # Rows are rank-major ([rank 0 req 0, rank 0 req 1, rank 1 req 0, ...]).
    # Request 0: ranks 0 and 2 tie at 5.0 -> rank 0's id 3 wins.
    # Request 1: ranks 1 and 2 tie at 4.0 -> rank 1's id 7 wins.
    gathered = torch.tensor(
        [[5.0, 3.0], [2.0, 9.0], [1.0, 1.0], [4.0, 7.0], [5.0, 0.0], [4.0, 8.0]]
    )
    out = DSparkProposer._select_gathered_candidate(gathered, tp_size=3, num_reqs=2)
    assert out.tolist() == [3, 7] and out.dtype == torch.int64


def test_pack_local_candidate_adds_vocab_offset():
    scores = torch.tensor([[1.0, 4.0, 4.0], [0.5, 0.25, 0.0]])
    packed = DSparkProposer._pack_local_candidate(scores, torch.tensor(10))
    assert packed.dtype == torch.float32
    assert packed.tolist() == [[4.0, 11.0], [0.5, 10.0]]


class _CompiledDraft:
    """Traceable draft for the compiled forward: logits are one-hot on
    round(hidden[:, 0]) and the Markov head is identically zero, so both
    sampling branches must produce round(hidden[:, 0]) per slot."""

    def __init__(self, vocab: int, hidden: int):
        self.vocab = vocab
        self.embed = torch.nn.Embedding(vocab, hidden)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[:, 0] = torch.arange(vocab).float()
        self._w1 = torch.zeros(vocab, 2)
        self._w2 = torch.zeros(vocab, 2)

    def model(self, input_ids, positions):
        return self.embed(input_ids)

    def _logits(self, hidden):
        tokens = hidden[:, 0].round().clamp(min=0).to(torch.int64)
        return torch.nn.functional.one_hot(tokens, self.vocab).to(torch.float32)

    compute_draft_logits = _logits
    compute_draft_logits_local = _logits

    def markov_embed(self, prev):
        return self._w1[prev]

    def markov_bias(self, embed):
        return embed @ self._w2.t()

    @staticmethod
    def map_draft_to_target(ids):
        return ids


@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="torch.compile(backend='tpu') requires a physical TPU accelerator",
)
@pytest.mark.parametrize("sharded_tp", [1, 2])
def test_dflash_forward_and_sample_compiled_branches(sharded_tp, monkeypatch):
    import torch._dynamo

    from vllm_torchtpu.spec_decode import dspark as dspark_mod

    torch._dynamo.reset()
    vocab, hidden, K, num_reqs = 32, 4, 4, 3
    p = _make_proposer(dict(_DSPARK_HF, dspark_block_size=K), num_speculative_tokens=K)
    p.draft_model = _CompiledDraft(vocab, hidden)
    p._sharded_markov_tp = sharded_tp
    p._markov_w2_local = p.draft_model._w2
    p._vocab_shard_offset = torch.tensor(0, dtype=torch.int64)
    p._vocab_shard_mask = None
    # Two identical "ranks": the gathered candidates duplicate the local ones.
    monkeypatch.setattr(
        dspark_mod,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x] * sharded_tp, dim=dim),
    )
    # One extra (padding) block beyond num_reqs is sampled like the rest.
    input_ids = torch.randint(0, vocab, ((num_reqs + 1) * K,))
    positions = torch.arange(input_ids.shape[0])
    tokens, hidden_out = p._dflash_forward_and_sample(input_ids, positions, K)
    assert hidden_out.shape == (input_ids.shape[0], hidden)
    assert torch.equal(tokens, input_ids.view(num_reqs + 1, K))


def test_dflash_forward_and_sample_eager_matches_reference(monkeypatch):
    """Both sampling branches, run eagerly (dynamo disabled) against the
    stock full-vocab chain, including the Markov bias dtype cast."""
    import torch._dynamo

    from vllm_torchtpu.spec_decode import dspark as dspark_mod

    torch.manual_seed(0)
    vocab, hidden, rank_r, K, num_reqs = 64, 8, 4, 3, 2
    lm_head_weight = torch.randint(-2, 3, (vocab, hidden)).float()
    w1 = torch.randint(-1, 2, (vocab, rank_r)).float()
    w2 = torch.randint(-2, 3, (vocab, rank_r)).float()

    class _Draft:
        def model(self, input_ids, positions):
            return torch.nn.functional.one_hot(input_ids, hidden).float()

        def compute_draft_logits(self, h):
            return h @ lm_head_weight.t()

        compute_draft_logits_local = compute_draft_logits

        def markov_embed(self, prev):
            return w1[prev]

        def markov_bias(self, embed):
            return embed @ w2.t()

        @staticmethod
        def map_draft_to_target(ids):
            return ids

    input_ids = torch.randint(0, hidden, (num_reqs * K,))
    positions = torch.arange(num_reqs * K)
    ids_3d = input_ids.view(num_reqs, K)
    logits_full = (
        torch.nn.functional.one_hot(input_ids, hidden).float() @ lm_head_weight.t()
    ).view(num_reqs, K, vocab)
    expected = _reference_markov_chain(logits_full, w1, w2, ids_3d, vocab, 0)

    monkeypatch.setattr(
        dspark_mod,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    # The un-compiled Python body behind the @torch.compile wrapper.
    forward_and_sample = torch._dynamo.eval_frame.innermost_fn(
        DSparkProposer._dflash_forward_and_sample
    )
    p = _make_proposer(dict(_DSPARK_HF, dspark_block_size=K), num_speculative_tokens=K)
    p.draft_model = _Draft()
    tokens, _ = forward_and_sample(p, input_ids, positions, K)
    assert torch.equal(tokens, expected)

    p._sharded_markov_tp = 2
    p._markov_w2_local = w2.to(torch.bfloat16)
    p._vocab_shard_offset = torch.tensor(0, dtype=torch.int64)
    tokens, _ = forward_and_sample(p, input_ids, positions, K)
    # bf16 Markov rows (exact for these small integers) + fp32 embed cast.
    assert torch.equal(tokens, expected)
    bias = p._markov_bias_local(ids_3d[:, 0])
    assert bias.dtype == torch.bfloat16
    assert torch.equal(bias.float(), w1[ids_3d[:, 0]] @ w2.t())

    # Also verify sample_from_anchor=False (sample_start=1), non-None
    # _vocab_shard_mask, and matching fp32 _markov_w2_local on the main thread.
    p_bonus = _make_proposer(
        dict(_DSPARK_HF, dspark_block_size=K, sample_from_anchor=False),
        num_speculative_tokens=K,
    )
    p_bonus.draft_model = _Draft()
    p_bonus._sharded_markov_tp = 2
    p_bonus._markov_w2_local = w2
    p_bonus._vocab_shard_offset = torch.tensor(0, dtype=torch.int64)
    p_bonus._vocab_shard_mask = torch.zeros(vocab, dtype=torch.float32)
    block_bonus = p_bonus.block_size
    ids_bonus = torch.randint(0, hidden, (num_reqs * block_bonus,))
    pos_bonus = torch.arange(num_reqs * block_bonus)
    ids_bonus_3d = ids_bonus.view(num_reqs, block_bonus)
    logits_bonus = (
        torch.nn.functional.one_hot(ids_bonus, hidden).float() @ lm_head_weight.t()
    ).view(num_reqs, block_bonus, vocab)
    expected_bonus = _reference_markov_chain(
        logits_bonus, w1, w2, ids_bonus_3d, vocab, 1
    )
    tokens_bonus, _ = forward_and_sample(p_bonus, ids_bonus, pos_bonus, block_bonus)
    assert torch.equal(tokens_bonus, expected_bonus)


def test_load_model_initializes_sharded_sampling(monkeypatch):
    from vllm_torchtpu.spec_decode import dflash as dflash_mod
    from vllm_torchtpu.spec_decode import utils as spec_decode_utils

    p = _make_proposer(_DSPARK_HF)
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(3, 2), layers=[]),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        get_draft_attn_causal=lambda: [],
        logits_processor=SimpleNamespace(_gather_logits=lambda x: x),
        has_own_embed_tokens=True,
        has_own_lm_head=True,
        compute_draft_logits=lambda h: h,
        markov_embed=lambda t: t,
        markov_bias=lambda e: e,
        map_draft_to_target=lambda t: t,
    )
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(3, 2)),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        set_aux_hidden_state_layers=lambda _: None,
    )

    def load_draft_model():
        p.draft_model = draft

    monkeypatch.setattr(p, "_load_draft_model", load_draft_model)
    monkeypatch.setattr(dflash_mod, "get_layers_from_vllm_config", lambda *a, **k: {})
    monkeypatch.setattr(
        spec_decode_utils, "gather_sharded_weight", _fake_gather_sharded_weight
    )
    p.load_model(target)
    # draft_tp=1 -> the lm_head is replicated, so the full-vocab path stays.
    assert p._sharded_markov_tp == 1 and p._markov_w2_local is None

    del draft.markov_bias
    with pytest.raises(RuntimeError, match="lacks markov_bias"):
        p.load_model(target)


def _fake_gather_sharded_weight(module):
    weight = module.weight.detach().clone()
    return weight, weight.shape[0], weight.shape[1], weight.device
