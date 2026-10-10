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

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.spec_decode import utils


def test_own_vocab_weights_are_kept_when_tp_layout_matches():
    target_embed = object()
    target_head = object()
    draft_embed = object()
    draft_head = object()
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_head,
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=draft_head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )

    utils.maybe_share_embeddings(draft, target, tp_layout_matches=True)
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    assert draft.model.embed_tokens is draft_embed
    assert draft.lm_head is draft_head


class _SharedHeadLayer(torch.nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.shared_head = torch.nn.Module()
        self.shared_head.head = torch.nn.Linear(hidden, vocab, bias=False)


def _make_mtp_draft(num_layers: int = 1, hidden: int = 2, vocab: int = 3):
    """A DeepSeek/GLM-5.2-shaped MTP draft: no top-level lm_head, one head per
    MTP layer at model.layers.<N>.shared_head.head, LogitsProcessor on the
    inner model."""
    layers = torch.nn.ModuleDict(
        {str(78 + i): _SharedHeadLayer(hidden, vocab) for i in range(num_layers)}
    )
    inner = torch.nn.Module()
    inner.layers = layers
    inner.logits_processor = SimpleNamespace(
        _gather_logits=lambda logits: logits.clone()
    )
    draft = torch.nn.Module()
    draft.model = inner
    return draft


def _make_target(hidden: int = 2, vocab: int = 3):
    return SimpleNamespace(lm_head=torch.nn.Linear(hidden, vocab, bias=False))


def test_mtp_shared_head_is_bound_to_target_lm_head():
    """G1: DeepSeekMTP keeps its head at model.layers.<N>.shared_head.head, so
    the plain lm_head attribute walk never reaches it."""
    draft = _make_mtp_draft(num_layers=2)
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    heads = [sh.head for sh in utils.iter_mtp_shared_heads(draft)]
    assert len(heads) == 2
    assert all(h is target.lm_head for h in heads)
    # No dead top-level attribute is invented when shared heads were bound.
    assert not hasattr(draft, "lm_head")


def test_mtp_replicated_draft_overrides_inner_logits_processor():
    """G2: DeepSeekMTP's LogitsProcessor lives on the inner model, so the
    top-level getattr used to return None and skip the TP-gather override."""
    draft = _make_mtp_draft()
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=True, tp_layout_matches=True
    )

    logits = torch.randn(2, 3)
    assert draft.model.logits_processor._gather_logits(logits) is logits


def test_replicated_draft_without_logits_processor_raises():
    draft = _make_mtp_draft()
    del draft.model.logits_processor
    target = _make_target()

    with pytest.raises(RuntimeError, match="LogitsProcessor"):
        utils.maybe_share_lm_head(
            draft, target, draft_replicated=True, tp_layout_matches=True
        )


def test_top_level_lm_head_draft_is_unchanged():
    """Regression: eagle3 / Qwen3.5-MTP drafts have a top-level lm_head and no
    shared_head, so the shared-head walk must be a no-op for them."""
    draft = SimpleNamespace(
        model=SimpleNamespace(),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        logits_processor=SimpleNamespace(_gather_logits=lambda logits: logits.clone()),
    )
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=True, tp_layout_matches=True
    )

    assert draft.lm_head is target.lm_head
    assert list(utils.iter_mtp_shared_heads(draft)) == []
    logits = torch.randn(2, 3)
    assert draft.logits_processor._gather_logits(logits) is logits


def test_spec_decode_utils_all_branches(monkeypatch):
    fake_tp = SimpleNamespace(world_size=8, rank_in_group=3, cpu_group="cpu_grp")
    monkeypatch.setattr(utils, "get_tp_group", lambda: fake_tp)
    import vllm.distributed.parallel_state as ps_mod

    monkeypatch.setattr(ps_mod, "get_tp_group", lambda: fake_tp)
    with utils._force_draft_tp1():
        assert fake_tp.world_size == 1 and fake_tp.rank_in_group == 0
    assert fake_tp.world_size == 8 and fake_tp.rank_in_group == 3

    monkeypatch.setattr(utils.dist, "all_reduce", lambda tensor, group=None: None)
    lora_mod = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=4,
        weight=torch.ones(4, 4),
        shard_indices=SimpleNamespace(
            added_vocab_start_index=0,
            added_vocab_end_index=2,
            org_vocab_start_index=0,
            org_vocab_end_index=4,
        ),
    )
    with pytest.raises(NotImplementedError, match="LoRA"):
        utils.gather_sharded_weight(lora_mod)

    f32_mod = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=4,
        weight=torch.ones(4, 4, dtype=torch.float32),
        shard_indices=SimpleNamespace(
            added_vocab_start_index=0,
            added_vocab_end_index=0,
            org_vocab_start_index=0,
            org_vocab_end_index=4,
        ),
    )
    full_f32, vocab, dim, _ = utils.gather_sharded_weight(f32_mod)
    assert full_f32.dtype == torch.float32 and vocab == 8 and dim == 4

    class _FakeSlice:
        dtype = torch.bfloat16
        device = SimpleNamespace(type="tpu")

        def to(self, dtype):
            res = _FakeSlice()
            res.dtype = dtype
            return res

        def cpu(self):
            return torch.ones(4, 4, dtype=torch.float32)

    class _FakeData:
        def __getitem__(self, s):
            return _FakeSlice()

    class _FakeWeight:
        dtype = torch.bfloat16
        device = torch.device("meta")
        data = _FakeData()

    bf16_mod = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=4,
        weight=_FakeWeight(),
        shard_indices=f32_mod.shard_indices,
    )
    full_bf16, _, _, dev_bf16 = utils.gather_sharded_weight(bf16_mod)
    assert full_bf16.dtype == torch.bfloat16 and dev_bf16.type == "meta"

    synced = []
    monkeypatch.setattr(utils, "synchronize_tensors", lambda t: synced.append(t))
    draft_m = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    utils.populate_draft_embed_from_target(draft_m, f32_mod)
    assert isinstance(draft_m.model.embed_tokens, torch.nn.Embedding)
    assert len(synced) == 0

    monkeypatch.setattr(
        utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="tpu"),
        ),
    )
    draft_m_tpu = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    utils.populate_draft_embed_from_target(draft_m_tpu, f32_mod)
    assert len(synced) == 1

    target_with_getter = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(
            model=SimpleNamespace(embed_tokens=f32_mod)
        )
    )
    draft_no_attr = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    utils.maybe_share_embeddings(
        draft_no_attr, target_with_getter, tp_layout_matches=True
    )
    assert draft_no_attr.model.embed_tokens is f32_mod

    draft_force = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8)),
        has_own_embed_tokens=True,
    )
    utils.maybe_share_embeddings(
        draft_force, target_with_getter, tp_layout_matches=False, force_share=True
    )
    assert isinstance(draft_force.model.embed_tokens, torch.nn.Embedding)

    with pytest.raises(RuntimeError, match="target_lm.model.embed_tokens"):
        utils.maybe_share_embeddings(
            draft_no_attr, SimpleNamespace(model=None), tp_layout_matches=True
        )

    draft_mixed_layers = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(),
                SimpleNamespace(
                    shared_head=SimpleNamespace(head=torch.nn.Linear(4, 8, bias=False))
                ),
            ]
        )
    )
    assert len(list(utils.iter_mtp_shared_heads(draft_mixed_layers))) == 1

    utils.populate_draft_lm_head_from_target(draft_mixed_layers, f32_mod)

    monkeypatch.setattr(
        utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="cpu"),
        ),
    )
    utils.populate_draft_lm_head_from_target(draft_mixed_layers, f32_mod)

    draft_inner_head = SimpleNamespace(
        model=SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False))
    )
    utils.populate_draft_lm_head_from_target(draft_inner_head, f32_mod)

    monkeypatch.setattr(
        utils,
        "ParallelLMHead",
        lambda vocab, dim: torch.nn.Linear(dim, vocab, bias=False),
    )
    monkeypatch.setattr(
        utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="tpu"),
        ),
    )
    draft_no_head = SimpleNamespace(model=SimpleNamespace())
    utils.populate_draft_lm_head_from_target(draft_no_head, f32_mod)
    assert hasattr(draft_no_head, "lm_head")

    monkeypatch.setattr(
        utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="cpu"),
        ),
    )
    draft_top_head = SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False))
    utils.populate_draft_lm_head_from_target(draft_top_head, f32_mod)

    with pytest.raises(RuntimeError, match="have an lm_head"):
        utils.maybe_share_lm_head(
            draft_top_head,
            SimpleNamespace(get_language_model=lambda: SimpleNamespace()),
            draft_replicated=False,
            tp_layout_matches=True,
        )

    target_head_getter = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(lm_head=f32_mod)
    )
    draft_inner_share = SimpleNamespace(
        model=SimpleNamespace(lm_head=object()),
        has_own_lm_head=True,
    )
    utils.maybe_share_lm_head(
        draft_inner_share,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=True,
        force_share=True,
    )
    assert draft_inner_share.model.lm_head is f32_mod

    draft_empty_share = SimpleNamespace(model=SimpleNamespace())
    utils.maybe_share_lm_head(
        draft_empty_share,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=True,
    )
    assert draft_empty_share.lm_head is f32_mod

    draft_mismatch_mat = SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False))
    utils.maybe_share_lm_head(
        draft_mismatch_mat,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=True,
    )
    utils.maybe_share_lm_head(
        draft_mismatch_mat,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=False,
    )
