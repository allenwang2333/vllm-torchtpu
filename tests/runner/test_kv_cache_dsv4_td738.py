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

"""CPU unit tests for TD7.38 optimizations and 100% coverage."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu.runner.kv_cache_dsv4 import (
    DsV4KVCacheAllocator,
    deepseek_v41_cache_spec,
)
from vllm_torchtpu.spec_decode import utils as sd_utils


@dataclass(frozen=True)
class _NoAlignSpec:
    block_size: int = 16
    tokens_per_state: int = 1
    state_content_size_bytes: int = 512
    dtype: torch.dtype = torch.uint8
    state_content_bytes: int = 512
    page_size_padded: int | None = None
    unpadded_page_size_bytes: int = 512


def test_deepseek_v41_cache_spec_all_branches():
    vllm_cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))
    assert deepseek_v41_cache_spec(None, vllm_cfg) is None

    # 1. indexer=True branch.
    mla_spec = MLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=None,
    )
    idx_out = deepseek_v41_cache_spec(mla_spec, vllm_cfg, indexer=True)
    assert idx_out.dtype == torch.uint8
    assert idx_out.alignment is None

    # 2. compressed=True branch.
    comp_spec = MLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=None,
    )
    comp_out = deepseek_v41_cache_spec(comp_spec, vllm_cfg, compressed=True)
    assert comp_out.dtype == torch.uint8

    # 3. CircularBufferSpec branch (not indexer, not compressed).
    circ_spec = CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    circ_out = deepseek_v41_cache_spec(circ_spec, vllm_cfg)
    assert circ_out.block_size == 4

    # 4. Non-CircularBufferSpec without alignment attribute + budget exceeded error.
    no_align = _NoAlignSpec(unpadded_page_size_bytes=1024)
    na_out = deepseek_v41_cache_spec(no_align, vllm_cfg)
    assert na_out.page_size_padded == 40960

    over_budget = _NoAlignSpec(unpadded_page_size_bytes=1_000_000)
    with pytest.raises(ValueError, match="exceeds its native array budget"):
        deepseek_v41_cache_spec(over_budget, vllm_cfg)


def test_initialize_v41_all_branches(monkeypatch):
    import vllm_torchtpu.kv_cache_materializer as mat_mod
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_cache as cache_ops,
    )

    runner = SimpleNamespace(device="cpu", shared_kv_cache_layers={})
    alloc = DsV4KVCacheAllocator(runner)

    # 1. Non-uniform page budget error.
    monkeypatch.setattr(mat_mod, "layer_to_pool_index", lambda cfg: {"l0": 0})
    cfg_non_uniform = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=2560)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=5120)),
        ]
    )
    with pytest.raises(ValueError, match="uniform physical page budget"):
        alloc.initialize_v41(cfg_non_uniform, lambda name: None)

    # 2. page_bytes not divisible by 2560 error.
    cfg_bad_div = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=3000))
        ]
    )
    with pytest.raises(ValueError, match="complete NoPE/RoPE rows"):
        alloc.initialize_v41(cfg_bad_div, lambda name: None)

    # 3. Indexer spec num_states mismatch error.
    page_bytes = 51200  # nope_bytes = 40960, rows = 80, keys_per_page = 320
    monkeypatch.setattr(
        mat_mod,
        "layer_to_pool_index",
        lambda cfg: {"m.layers.0.attn.indexer.k_cache": 0},
    )
    cfg_valid_page = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(page_size_bytes=page_bytes),
                layer_names=["m.layers.0.attn.indexer.k_cache"],
            )
        ],
        kv_cache_tensors=[SimpleNamespace(size=10_000_000)],
    )
    bad_idx_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=None,
    )
    with pytest.raises(ValueError, match="Index spec and native page geometry"):
        alloc.initialize_v41(cfg_valid_page, lambda name: bad_idx_spec)

    # 4. Full valid initialize_v41 across main MLA, SWA, indexer, and compressor state,
    #    followed by budget exceeded check.
    regions = {
        "m.layers.0.attn": 0,
        "m.layers.0.attn.swa_cache": 1,
        "m.layers.0.attn.indexer.k_cache": 2,
        "m.layers.0.attn.compressor.state_cache": 3,
    }
    monkeypatch.setattr(mat_mod, "layer_to_pool_index", lambda cfg: regions)
    monkeypatch.setattr(
        cache_ops,
        "native_cache_view",
        lambda raw, shape, dtype, stride, offset: raw,
    )
    specs_v41 = {
        "m.layers.0.attn": MLAAttentionSpec(
            block_size=80,
            num_kv_heads=1,
            head_size=640,
            dtype=torch.uint8,
            tokens_per_state=1,
            alignment=None,
        ),
        "m.layers.0.attn.swa_cache": SlidingWindowMLASpec(
            block_size=80,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=80,
            alignment=None,
        ),
        "m.layers.0.attn.indexer.k_cache": MLAAttentionSpec(
            block_size=320,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=1,
            alignment=None,
        ),
        "m.layers.0.attn.compressor.state_cache": CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=640,
            dtype=torch.float32,
        ),
    }
    cfg_all = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(page_size_bytes=page_bytes),
                layer_names=list(regions.keys()),
            )
        ],
        kv_cache_tensors=[SimpleNamespace(size=10_000_000)],
    )
    res = alloc.initialize_v41(cfg_all, specs_v41.__getitem__)
    assert set(res.kv_caches.keys()) == set(regions.keys())

    cfg_small_budget = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=cfg_all.kv_cache_groups,
        kv_cache_tensors=[SimpleNamespace(size=16)],
    )
    with pytest.raises(ValueError, match="exceeds"):
        alloc.initialize_v41(cfg_small_budget, specs_v41.__getitem__)


def test_dsv4_allocator_remaining_branches():
    runner = SimpleNamespace(
        device="cpu",
        shared_kv_cache_layers={"m.layers.shared.attn"},
    )
    alloc = DsV4KVCacheAllocator(runner)

    with pytest.raises(ValueError, match="unexpected layer name"):
        alloc._ds_v4_compressed_kv_layer_name("invalid_state_name")

    # 1. Unknown layer role in _classify_ds_v4_layers.
    full_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    cfg_unknown = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["m.layers.shared.attn", "m.layers.0.unknown"])
        ]
    )
    with pytest.raises(ValueError, match="no known role"):
        alloc._classify_ds_v4_layers(cfg_unknown, lambda _: full_spec)

    # 2. No MLAAttentionSpec layers error.
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        sliding_window=64,
        alignment=None,
    )
    cfg_no_mla = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["m.layers.0.swa_cache"])]
    )
    with pytest.raises(ValueError, match="no MLAAttentionSpec layers"):
        alloc._classify_ds_v4_layers(cfg_no_mla, lambda _: swa_spec)

    # 3. _initialize_ds_v4_kv_cache with no CSA layer (only HCA) error.
    hca_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        tokens_per_state=128,
        alignment=None,
    )
    cfg_only_hca = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["m.layers.0.attn"])]
    )
    with pytest.raises(ValueError, match="no CSA layer"):
        alloc._initialize_ds_v4_kv_cache(cfg_only_hca, lambda _: hca_spec, {}, 2)

    # 4. State cache whose compressed-KV layer does not exist in kv_caches.
    csa_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        tokens_per_state=4,
        alignment=None,
    )
    idx_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=None,
    )
    specs_missing_kv = {
        "m.layers.0.attn": csa_spec,
        "m.layers.99.attn.compressor.state_cache": swa_spec,
    }
    cfg_missing_kv = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "m.layers.0.attn",
                    "m.layers.99.attn.compressor.state_cache",
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="no compressed-KV layer"):
        alloc._initialize_ds_v4_kv_cache(
            cfg_missing_kv, specs_missing_kv.__getitem__, {}, 2
        )

    # 5. Full _initialize_ds_v4_kv_cache happy path with CSA, indexer, HCA,
    #    SWA overflow (>1 SWA in group), CSA state, indexer state, and HCA state.
    full_specs = {
        "m.layers.0.attn": csa_spec,
        "m.layers.0.attn.indexer.k_cache": idx_spec,
        "m.layers.1.attn": hca_spec,
        "m.layers.0.swa_cache": swa_spec,
        "m.layers.1.swa_cache": swa_spec,
        "m.layers.0.attn.compressor.state_cache": swa_spec,
        "m.layers.0.attn.indexer.compressor.state_cache": swa_spec,
        "m.layers.1.attn.compressor.state_cache": swa_spec,
    }
    cfg_full = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "m.layers.0.attn",
                    "m.layers.0.attn.indexer.k_cache",
                    "m.layers.1.attn",
                ]
            ),
            SimpleNamespace(
                layer_names=["m.layers.0.swa_cache", "m.layers.1.swa_cache"]
            ),
            SimpleNamespace(
                layer_names=[
                    "m.layers.0.attn.compressor.state_cache",
                    "m.layers.0.attn.indexer.compressor.state_cache",
                ]
            ),
            SimpleNamespace(layer_names=["m.layers.1.attn.compressor.state_cache"]),
        ]
    )
    kv_out: dict[str, torch.Tensor] = {}
    alloc._initialize_ds_v4_kv_cache(cfg_full, full_specs.__getitem__, kv_out, 2)
    assert set(kv_out.keys()) == set(full_specs.keys())

    # 6. _validate_ds_v4_overlay missing layer, duplicate name, and storage collision.
    t0 = torch.zeros(4, dtype=torch.int32)
    DsV4KVCacheAllocator._validate_ds_v4_overlay(
        SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["missing_layer", "l0", "l0"])]
        ),
        {"l0": t0},
    )
    with pytest.raises(ValueError, match="same array or the same storage"):
        DsV4KVCacheAllocator._validate_ds_v4_overlay(
            SimpleNamespace(
                kv_cache_groups=[SimpleNamespace(layer_names=["l0", "l1"])]
            ),
            {"l0": t0, "l1": t0},
        )


def test_spec_decode_utils_all_branches(monkeypatch):
    # 1. _force_draft_tp1 context manager.
    fake_tp = SimpleNamespace(world_size=8, rank_in_group=3, cpu_group="cpu_grp")
    monkeypatch.setattr(sd_utils, "get_tp_group", lambda: fake_tp)
    import vllm.distributed.parallel_state as ps_mod

    monkeypatch.setattr(ps_mod, "get_tp_group", lambda: fake_tp)
    with sd_utils._force_draft_tp1():
        assert fake_tp.world_size == 1 and fake_tp.rank_in_group == 0
    assert fake_tp.world_size == 8 and fake_tp.rank_in_group == 3

    # 2. gather_sharded_weight:
    #    - LoRA added_vocab > 0 raises NotImplementedError
    #    - float32 weight on CPU (fast path: no dtype/device casts)
    #    - bfloat16 weight on simulated non-CPU / different target device
    monkeypatch.setattr(sd_utils.dist, "all_reduce", lambda tensor, group=None: None)
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
        sd_utils.gather_sharded_weight(lora_mod)

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
    full_f32, vocab, dim, dev = sd_utils.gather_sharded_weight(f32_mod)
    assert full_f32.dtype == torch.float32 and vocab == 8 and dim == 4

    # Simulate a weight slice with .device.type != "cpu" and draft_device != cpu.
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
    full_bf16, _, _, dev_bf16 = sd_utils.gather_sharded_weight(bf16_mod)
    assert full_bf16.dtype == torch.bfloat16 and dev_bf16.type == "meta"

    # 3. populate_draft_embed_from_target (both cpu and tpu device.type).
    synced = []
    monkeypatch.setattr(sd_utils, "synchronize_tensors", lambda t: synced.append(t))
    draft_m = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    sd_utils.populate_draft_embed_from_target(draft_m, f32_mod)
    assert isinstance(draft_m.model.embed_tokens, torch.nn.Embedding)
    assert len(synced) == 0

    monkeypatch.setattr(
        sd_utils,
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
    sd_utils.populate_draft_embed_from_target(draft_m_tpu, f32_mod)
    assert len(synced) == 1

    # 4. maybe_share_embeddings all branches.
    target_with_getter = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(
            model=SimpleNamespace(embed_tokens=f32_mod)
        )
    )
    draft_no_attr = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8))
    )
    sd_utils.maybe_share_embeddings(
        draft_no_attr, target_with_getter, tp_layout_matches=True
    )
    assert draft_no_attr.model.embed_tokens is f32_mod

    draft_force = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(org_vocab_size=8)),
        has_own_embed_tokens=True,
    )
    sd_utils.maybe_share_embeddings(
        draft_force, target_with_getter, tp_layout_matches=False, force_share=True
    )
    assert isinstance(draft_force.model.embed_tokens, torch.nn.Embedding)

    with pytest.raises(RuntimeError, match="target_lm.model.embed_tokens"):
        sd_utils.maybe_share_embeddings(
            draft_no_attr, SimpleNamespace(model=None), tp_layout_matches=True
        )

    # 5. iter_mtp_shared_heads with layers without shared_head.
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
    assert len(list(sd_utils.iter_mtp_shared_heads(draft_mixed_layers))) == 1

    # 6. populate_draft_lm_head_from_target & maybe_share_lm_head all branches.
    sd_utils.populate_draft_lm_head_from_target(draft_mixed_layers, f32_mod)

    monkeypatch.setattr(
        sd_utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="cpu"),
        ),
    )
    sd_utils.populate_draft_lm_head_from_target(draft_mixed_layers, f32_mod)

    draft_inner_head = SimpleNamespace(
        model=SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False))
    )
    sd_utils.populate_draft_lm_head_from_target(draft_inner_head, f32_mod)

    monkeypatch.setattr(
        sd_utils,
        "ParallelLMHead",
        lambda vocab, dim: torch.nn.Linear(dim, vocab, bias=False),
    )
    monkeypatch.setattr(
        sd_utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="tpu"),
        ),
    )
    draft_no_head = SimpleNamespace(model=SimpleNamespace())
    sd_utils.populate_draft_lm_head_from_target(draft_no_head, f32_mod)
    assert hasattr(draft_no_head, "lm_head")

    monkeypatch.setattr(
        sd_utils,
        "gather_sharded_weight",
        lambda m: (
            torch.ones(8, 4, dtype=torch.float32),
            8,
            4,
            SimpleNamespace(type="cpu"),
        ),
    )
    draft_top_head = SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False))
    sd_utils.populate_draft_lm_head_from_target(draft_top_head, f32_mod)

    with pytest.raises(RuntimeError, match="have an lm_head"):
        sd_utils.maybe_share_lm_head(
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
    sd_utils.maybe_share_lm_head(
        draft_inner_share,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=True,
        force_share=True,
    )
    assert draft_inner_share.model.lm_head is f32_mod

    draft_empty_share = SimpleNamespace(model=SimpleNamespace())
    sd_utils.maybe_share_lm_head(
        draft_empty_share,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=True,
    )
    assert draft_empty_share.lm_head is f32_mod

    draft_mismatch_mat = SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False))
    sd_utils.maybe_share_lm_head(
        draft_mismatch_mat,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=True,
    )
    sd_utils.maybe_share_lm_head(
        draft_mismatch_mat,
        target_head_getter,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=False,
    )

    # 7. _resolve_target_layer_ids and normalize_draft_config all branches.
    assert sd_utils._resolve_target_layer_ids(
        {"eagle_aux_hidden_state_layer_ids": [2, 5]}
    ) == [1, 4]
    assert sd_utils._resolve_target_layer_ids({"dspark_target_layer_ids": [3, 7]}) == [
        3,
        7,
    ]
    assert sd_utils._resolve_target_layer_ids({}) is None

    with pytest.raises(ValueError, match="noise/mask token id"):
        sd_utils.normalize_draft_config(SimpleNamespace())

    cfg_aux = SimpleNamespace(
        to_dict=lambda: {
            "dspark_noise_token_id": 99,
            "eagle_aux_hidden_state_layer_ids": [1, 2],
        }
    )
    out_aux = sd_utils.normalize_draft_config(cfg_aux)
    assert out_aux == {"mask_token_id": 99, "target_layer_ids": [0, 1]}

    cfg_preset = SimpleNamespace(
        to_dict=lambda: {
            "dflash_config": {"mask_token_id": 42, "target_layer_ids": [5]},
        }
    )
    out_preset = sd_utils.normalize_draft_config(cfg_preset)
    assert out_preset == {"mask_token_id": 42, "target_layer_ids": [5]}

    cfg_no_layers = SimpleNamespace(to_dict=lambda: {"mask_token_id": 7})
    out_no_layers = sd_utils.normalize_draft_config(cfg_no_layers)
    assert out_no_layers == {"mask_token_id": 7}
