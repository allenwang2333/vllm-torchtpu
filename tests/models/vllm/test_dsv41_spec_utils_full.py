# SPDX-License-Identifier: Apache-2.0
"""Unit and 100% statement/branch coverage tests for spec_decode.utils."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class _StubMeta(type):
    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()


class _StubModule(ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.isupper():
            return None
        if name[:1].isupper():
            return _StubMeta(name, (), {"__init__": lambda self, *a, **k: None})
        return MagicMock()


class _FakeParallelLMHead(torch.nn.Module):
    def __init__(self, org_vocab_size: int, embedding_dim: int):
        super().__init__()
        self.org_vocab_size = org_vocab_size
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.zeros(org_vocab_size, embedding_dim, dtype=torch.bfloat16)
        )


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in {
            "vllm",
            "torch_tpu",
            "requests",
            "portpicker",
            "einops",
            "psutil",
            "jax_tpu_embedding",
            "compressed_tensors",
        }:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = _StubModule(spec.name)
        if spec.name == "vllm.model_executor.layers.vocab_parallel_embedding":
            mod.ParallelLMHead = _FakeParallelLMHead  # type: ignore[attr-defined]
        return mod

    def exec_module(self, module):
        pass


sys.meta_path.append(_StubFinder())

if "tpu" not in torch._dynamo.list_backends():
    torch._dynamo.register_backend(
        compiler_fn=lambda gm, example_inputs: gm, name="tpu"
    )

from vllm_torchtpu.spec_decode import utils as su  # noqa: E402


class _FakeDeviceTensor:
    """Tensor-like wrapper exposing `.device.type = 'tpu'` or `torch.device('meta')`."""

    def __init__(
        self,
        t: torch.Tensor,
        device_type: str = "tpu",
        non_cpu_slice: bool = False,
    ):
        self._t = t
        self.dtype = t.dtype
        self.device = (
            torch.device("meta")
            if device_type == "meta"
            else SimpleNamespace(type=device_type)
        )
        self._non_cpu_slice = non_cpu_slice

    @property
    def data(self):
        return self

    @property
    def shape(self):
        return self._t.shape

    def __getitem__(self, idx):
        sub = self._t[idx]
        if self._non_cpu_slice:
            return _FakeDeviceTensor(
                sub, device_type=self.device.type, non_cpu_slice=True
            )
        return sub

    def cpu(self):
        return self._t

    def to(self, target):
        return self._t.clone()


def _make_sharded_embed(
    org_vocab: int,
    dim: int,
    start: int,
    end: int,
    added_start: int = 0,
    added_end: int = 0,
    weight_obj=None,
    part_rows: int | None = None,
):
    rows = part_rows if part_rows is not None else (end - start)
    w = (
        weight_obj
        if weight_obj is not None
        else torch.arange(rows * dim, dtype=torch.bfloat16).reshape(rows, dim)
    )
    return SimpleNamespace(
        org_vocab_size=org_vocab,
        embedding_dim=dim,
        weight=w,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            added_vocab_start_index=added_start,
            added_vocab_end_index=added_end,
        ),
    )


def test_draft_chunk_inputs_defaults():
    inp = su.DraftChunkInputs(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        position_ids=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        attn_ctx=SimpleNamespace(),  # type: ignore[arg-type]
        start_index=0,
        num_reqs=1,
        aux_hidden_states=[],
    )
    assert inp.sequence_layout_plan is su._DEFAULT_DRAFT_SEQUENCE_LAYOUT_PLAN
    assert inp.draft_lengths is None
    assert inp.hidden_states is None
    assert inp.attn_metadata is None


def test_force_draft_tp1_restores_on_success_and_exception(monkeypatch):
    tp = SimpleNamespace(world_size=8, rank_in_group=5)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: tp)
    with su._force_draft_tp1():
        assert tp.world_size == 1
        assert tp.rank_in_group == 0
    assert (tp.world_size, tp.rank_in_group) == (8, 5)

    with pytest.raises(RuntimeError, match="boom"), su._force_draft_tp1():
        assert (tp.world_size, tp.rank_in_group) == (1, 0)
        raise RuntimeError("boom")
    assert (tp.world_size, tp.rank_in_group) == (8, 5)


def test_gather_sharded_weight_rejects_lora_added_vocab(monkeypatch):
    monkeypatch.setattr(
        su, "get_tp_group", lambda: SimpleNamespace(cpu_group="g", rank_in_group=0)
    )
    mod = _make_sharded_embed(16, 4, 0, 8, added_start=16, added_end=20)
    with pytest.raises(NotImplementedError, match="LoRA"):
        su.gather_sharded_weight(mod)  # type: ignore[arg-type]


def test_gather_sharded_weight_tpu_fast_path_bit_exact(monkeypatch):
    tp = SimpleNamespace(cpu_group="g", rank_in_group=1)
    monkeypatch.setattr(su, "get_tp_group", lambda: tp)

    full_ref = torch.randn(10, 4, dtype=torch.bfloat16)
    shard1 = torch.cat(
        [full_ref[6:10], torch.full((2, 4), 999.0, dtype=torch.bfloat16)]
    )
    fake_tpu_w = _FakeDeviceTensor(shard1, device_type="tpu")

    monkeypatch.setattr(
        "vllm.distributed.tensor_model_parallel_all_gather",
        lambda tensor, dim=0: torch.cat([full_ref[:6], tensor._t], dim=dim),
    )

    mod = _make_sharded_embed(10, 4, start=6, end=10, weight_obj=fake_tpu_w)
    full_dev, org_vocab, dim, dev = su.gather_sharded_weight(mod)  # type: ignore[arg-type]
    assert org_vocab == 10 and dim == 4 and dev.type == "tpu"
    assert torch.equal(full_dev, full_ref)


def test_gather_sharded_weight_host_fallback_cpu_and_non_cpu(monkeypatch):
    tp = SimpleNamespace(cpu_group="cpu_grp", rank_in_group=1)
    monkeypatch.setattr(su, "get_tp_group", lambda: tp)

    peer = torch.zeros(8, 4, dtype=torch.bfloat16)
    peer[:4] = 3.5

    def fake_all_reduce(t: torch.Tensor, group=None):
        assert group == "cpu_grp"
        t.add_(peer)

    monkeypatch.setattr(su.dist, "all_reduce", fake_all_reduce)

    # 1) CPU weight tensor (`my_slice.device.type == "cpu"` and same device)
    my_cpu = torch.full((4, 4), 7.25, dtype=torch.bfloat16)
    mod_cpu = _make_sharded_embed(8, 4, start=4, end=8, weight_obj=my_cpu)
    out_cpu, v, d, dev = su.gather_sharded_weight(mod_cpu)  # type: ignore[arg-type]
    assert (v, d, dev) == (8, 4, torch.device("cpu"))
    assert torch.equal(out_cpu[:4], peer[:4])
    assert torch.equal(out_cpu[4:], my_cpu)

    # 2) Non-CPU slice fallback (`my_slice.device.type != "cpu"` and cross-device)
    # Triggered when `draft_device.type != "tpu"` (e.g., `torch.device("meta")`)
    fake_dev_w = _FakeDeviceTensor(my_cpu, device_type="meta", non_cpu_slice=True)
    mod_dev = _make_sharded_embed(8, 4, start=4, end=8, weight_obj=fake_dev_w)
    out_dev, _, _, dev2 = su.gather_sharded_weight(mod_dev)  # type: ignore[arg-type]
    assert dev2.type == "meta"
    assert out_dev.shape == (8, 4)


def test_populate_draft_embed_from_target_cpu_and_tpu(monkeypatch):
    synced = []
    monkeypatch.setattr(su, "synchronize_tensors", lambda *a: synced.append(a))

    draft_bad = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=9))
    )
    tgt_bad = SimpleNamespace(org_vocab_size=8)
    with pytest.raises(AssertionError, match="cannot populate"):
        su.populate_draft_embed_from_target(draft_bad, tgt_bad)  # type: ignore[arg-type]

    w = torch.randn(8, 4, dtype=torch.bfloat16)
    # CPU branch
    monkeypatch.setattr(
        su, "gather_sharded_weight", lambda _: (w, 8, 4, torch.device("cpu"))
    )
    draft_cpu = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    su.populate_draft_embed_from_target(draft_cpu, SimpleNamespace(org_vocab_size=8))  # type: ignore[arg-type]
    assert synced == []
    assert not draft_cpu.model.embed_tokens.weight.requires_grad

    # TPU branch
    monkeypatch.setattr(
        su, "gather_sharded_weight", lambda _: (w, 8, 4, SimpleNamespace(type="tpu"))
    )
    draft_tpu = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    su.populate_draft_embed_from_target(draft_tpu, SimpleNamespace(org_vocab_size=8))  # type: ignore[arg-type]
    assert len(synced) == 1


def test_maybe_share_embeddings_all_branches(monkeypatch):
    populated = []
    monkeypatch.setattr(
        su, "populate_draft_embed_from_target", lambda d, t: populated.append((d, t))
    )

    tgt_embed = SimpleNamespace(org_vocab_size=8)
    target_with_getter = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(
            model=SimpleNamespace(embed_tokens=tgt_embed)
        )
    )
    target_direct = SimpleNamespace(model=SimpleNamespace(embed_tokens=tgt_embed))
    target_missing_embed = SimpleNamespace(model=None)

    # 1) has_own_embed_tokens = True, force_share = False -> no-op
    d1 = SimpleNamespace(
        has_own_embed_tokens=True, model=SimpleNamespace(embed_tokens="orig")
    )
    su.maybe_share_embeddings(
        d1, target_direct, tp_layout_matches=True, force_share=False
    )
    assert d1.model.embed_tokens == "orig"

    # 2) force_share = True overrides has_own_embed_tokens = True (tp match)
    su.maybe_share_embeddings(
        d1, target_with_getter, tp_layout_matches=True, force_share=True
    )
    assert d1.model.embed_tokens is tgt_embed

    # 3) Missing has_own_embed_tokens -> share_embed=True; tp mismatch -> populate
    d2 = SimpleNamespace(model=SimpleNamespace(embed_tokens="orig"))
    su.maybe_share_embeddings(d2, target_direct, tp_layout_matches=False)
    assert populated == [(d2, tgt_embed)]

    # 4) Missing target_embed -> raises RuntimeError
    with pytest.raises(RuntimeError, match="embed_tokens"):
        su.maybe_share_embeddings(d2, target_missing_embed, tp_layout_matches=True)


def test_iter_mtp_shared_heads_and_resolve_logits_processor():
    # No model / no layers
    assert list(su.iter_mtp_shared_heads(SimpleNamespace())) == []
    assert (
        list(su.iter_mtp_shared_heads(SimpleNamespace(model=SimpleNamespace()))) == []
    )

    h0 = SimpleNamespace(head="head0")
    h1 = SimpleNamespace(head="head1")
    h_no_head = SimpleNamespace()
    # ModuleDict branch
    md = torch.nn.ModuleDict(
        {
            "0": torch.nn.Module(),
            "1": torch.nn.Module(),
            "2": torch.nn.Module(),
        }
    )
    md["0"].shared_head = h0  # type: ignore[attr-defined]
    md["1"].shared_head = h_no_head  # type: ignore[attr-defined]
    md["2"].shared_head = h1  # type: ignore[attr-defined]
    assert list(
        su.iter_mtp_shared_heads(SimpleNamespace(model=SimpleNamespace(layers=md)))
    ) == [
        h0,
        h1,
    ]
    # List branch
    assert list(
        su.iter_mtp_shared_heads(
            SimpleNamespace(
                model=SimpleNamespace(layers=[SimpleNamespace(shared_head=h0)])
            )
        )
    ) == [h0]

    # resolve_draft_logits_processor top-level vs nested vs None
    assert (
        su.resolve_draft_logits_processor(SimpleNamespace(logits_processor="lp_top"))
        == "lp_top"
    )
    assert (
        su.resolve_draft_logits_processor(
            SimpleNamespace(model=SimpleNamespace(logits_processor="lp_inner"))
        )
        == "lp_inner"
    )
    assert su.resolve_draft_logits_processor(SimpleNamespace()) is None


def test_populate_draft_lm_head_from_target_shared_heads_and_top_level(monkeypatch):
    synced = []
    monkeypatch.setattr(su, "synchronize_tensors", lambda *a: synced.append(a))

    w = torch.randn(8, 4, dtype=torch.bfloat16)
    # 1) MTP shared_heads on TPU: shares single Parameter + syncs once
    monkeypatch.setattr(
        su, "gather_sharded_weight", lambda _: (w, 8, 4, SimpleNamespace(type="tpu"))
    )
    sh0 = SimpleNamespace(head=SimpleNamespace(weight=None))
    sh1 = SimpleNamespace(head=SimpleNamespace(weight=None))
    draft_mtp = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(shared_head=sh0), SimpleNamespace(shared_head=sh1)]
        )
    )
    su.populate_draft_lm_head_from_target(
        draft_mtp, SimpleNamespace(org_vocab_size=8, embedding_dim=4)
    )  # type: ignore[arg-type]
    assert sh0.head.weight is sh1.head.weight
    assert len(synced) == 1

    # 2) MTP shared_heads on CPU: no sync
    monkeypatch.setattr(
        su, "gather_sharded_weight", lambda _: (w, 8, 4, torch.device("cpu"))
    )
    su.populate_draft_lm_head_from_target(
        draft_mtp, SimpleNamespace(org_vocab_size=8, embedding_dim=4)
    )  # type: ignore[arg-type]
    assert len(synced) == 1

    # 3) draft_model.model.lm_head present + CPU
    draft_inner = SimpleNamespace(
        model=SimpleNamespace(lm_head=SimpleNamespace(weight=None))
    )
    su.populate_draft_lm_head_from_target(
        draft_inner, SimpleNamespace(org_vocab_size=8, embedding_dim=4)
    )  # type: ignore[arg-type]
    assert torch.equal(draft_inner.model.lm_head.weight, w)

    # 4) No lm_head anywhere + TPU -> constructs ParallelLMHead and syncs
    monkeypatch.setattr(
        su, "gather_sharded_weight", lambda _: (w, 8, 4, SimpleNamespace(type="tpu"))
    )
    draft_none = SimpleNamespace()
    su.populate_draft_lm_head_from_target(
        draft_none, SimpleNamespace(org_vocab_size=8, embedding_dim=4)
    )  # type: ignore[arg-type]
    assert isinstance(draft_none.lm_head, _FakeParallelLMHead)
    assert len(synced) == 2


def test_maybe_share_lm_head_all_branches(monkeypatch):
    populated = []
    monkeypatch.setattr(
        su, "populate_draft_lm_head_from_target", lambda d, t: populated.append((d, t))
    )

    tgt_head = SimpleNamespace(org_vocab_size=8, embedding_dim=4)
    target_with_getter = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(lm_head=tgt_head)
    )
    target_direct = SimpleNamespace(lm_head=tgt_head)

    # Missing target lm_head -> raises RuntimeError
    with pytest.raises(RuntimeError, match="lm_head"):
        su.maybe_share_lm_head(SimpleNamespace(), SimpleNamespace(), False, True)

    # 1) has_own_lm_head = True, force_share = False, draft_replicated = False -> no-op
    d_own = SimpleNamespace(has_own_lm_head=True, lm_head="orig")
    su.maybe_share_lm_head(
        d_own, target_direct, draft_replicated=False, tp_layout_matches=True
    )
    assert d_own.lm_head == "orig"

    # 2) force_share = True with shared_heads + top-level lm_head + draft_replicated
    sh = SimpleNamespace(head="old_sh")
    lp = SimpleNamespace(_gather_logits=lambda x: x + 1)
    d_sh_top = SimpleNamespace(
        has_own_lm_head=True,
        lm_head="old_top",
        logits_processor=lp,
        model=SimpleNamespace(layers=[SimpleNamespace(shared_head=sh)]),
    )
    su.maybe_share_lm_head(
        d_sh_top,
        target_with_getter,
        draft_replicated=True,
        tp_layout_matches=True,
        force_share=True,
    )
    assert sh.head is tgt_head and d_sh_top.lm_head is tgt_head
    assert lp._gather_logits(5) == 5

    # 3) Default share_lm_head=True, model.lm_head branch
    d_inner = SimpleNamespace(model=SimpleNamespace(lm_head="old_inner"))
    su.maybe_share_lm_head(
        d_inner, target_direct, draft_replicated=False, tp_layout_matches=True
    )
    assert d_inner.model.lm_head is tgt_head

    # 4) Default share_lm_head=True, shared_heads ONLY (no top/model lm_head)
    sh2 = SimpleNamespace(head="old2")
    d_mtp_only = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(shared_head=sh2)])
    )
    su.maybe_share_lm_head(
        d_mtp_only, target_direct, draft_replicated=False, tp_layout_matches=True
    )
    assert sh2.head is tgt_head and not hasattr(d_mtp_only, "lm_head")

    # 5) Default share_lm_head=True, bare draft model -> sets draft_model.lm_head
    d_bare = SimpleNamespace()
    su.maybe_share_lm_head(
        d_bare, target_direct, draft_replicated=False, tp_layout_matches=True
    )
    assert d_bare.lm_head is tgt_head

    # 6) tp_layout_matches = False: materialize_if_mismatched = True vs False
    d_mm = SimpleNamespace()
    su.maybe_share_lm_head(
        d_mm,
        target_direct,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=False,
    )
    assert populated == []
    su.maybe_share_lm_head(
        d_mm,
        target_direct,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=True,
    )
    assert populated == [(d_mm, tgt_head)]

    # 7) draft_replicated = True error paths: no logits_processor & no _gather_logits
    with pytest.raises(RuntimeError, match="LogitsProcessor"):
        su.maybe_share_lm_head(
            SimpleNamespace(),
            target_direct,
            draft_replicated=True,
            tp_layout_matches=True,
        )
    with pytest.raises(AssertionError, match="_gather_logits"):
        su.maybe_share_lm_head(
            SimpleNamespace(logits_processor=SimpleNamespace()),
            target_direct,
            draft_replicated=True,
            tp_layout_matches=True,
        )


def test_resolve_target_layer_ids_and_normalize_draft_config():
    # _resolve_target_layer_ids branches
    assert su._resolve_target_layer_ids(
        {"eagle_aux_hidden_state_layer_ids": [2, 10]}
    ) == [1, 9]
    assert su._resolve_target_layer_ids({"dspark_target_layer_ids": (3, 7)}) == [3, 7]
    assert su._resolve_target_layer_ids({"target_layer_ids": [5]}) == [5]
    assert su._resolve_target_layer_ids({}) is None

    # Missing noise token id raises ValueError (also exercises non-to_dict hf_config)
    with pytest.raises(ValueError, match="noise/mask token id"):
        su.normalize_draft_config(SimpleNamespace())

    # First call resolves dspark_noise_token_id + dspark_target_layer_ids via to_dict()
    to_dict_calls = 0

    class _Cfg:
        def to_dict(self):
            nonlocal to_dict_calls
            to_dict_calls += 1
            return {
                "dspark_noise_token_id": 128001,
                "dspark_target_layer_ids": [12, 24],
            }

    cfg = _Cfg()
    first = su.normalize_draft_config(cfg)
    assert first == {"mask_token_id": 128001, "target_layer_ids": [12, 24]}
    assert to_dict_calls == 1

    # Second (idempotent) call hits fast path without invoking to_dict()
    second = su.normalize_draft_config(cfg)
    assert second == first and second is not first
    assert to_dict_calls == 1

    # Config with mask_token_id inside dflash_config dict and NO target_layer_ids
    cfg_no_layers = SimpleNamespace(
        to_dict=lambda: {"dflash_config": {"mask_token_id": 42}}
    )
    out_no_layers = su.normalize_draft_config(cfg_no_layers)
    assert out_no_layers == {"mask_token_id": 42}

    # Config with no attr dflash_config yet, where to_dict already sets target_layer_ids
    cfg_pre_layers = SimpleNamespace(
        to_dict=lambda: {
            "dflash_config": {"mask_token_id": 99, "target_layer_ids": [4, 8]}
        }
    )
    assert su.normalize_draft_config(cfg_pre_layers) == {
        "mask_token_id": 99,
        "target_layer_ids": [4, 8],
    }
