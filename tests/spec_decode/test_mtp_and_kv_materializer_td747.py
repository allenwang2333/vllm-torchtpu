# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TD7.47: MTP loop models and KV cache materializer."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import CompilationMode
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.kv_cache_materializer import (
    _can_allocate_attention_cache_directly,
    _pool_attention_geometry,
    _raw_tensor_index,
    _select_representative_kv_cache_spec,
    _storage_ptr,
    _tensor_layout_summary,
    allocate_raw_kv_cache_tensors,
    build_kernel_block_size_by_group_id,
    can_share_attention_cache,
    format_kv_cache_layout_summary,
    layer_to_pool_index,
    make_attention_cache_tensor,
    materialize_kv_cache_tensors,
    materialize_shared_attention_cache,
    resolve_block_major_pool_layout,
)
from vllm_torchtpu.spec_decode import mtp_fused_loop as mfl
from vllm_torchtpu.spec_decode.mtp_fused_loop import (
    MtpDraftModel,
    _MtpFusedLoop,
    build_mtp_fused_loop,
)
from vllm_torchtpu.spec_decode.mtp_loop_model import (
    MTP_DYNAMIC_ARG_DIMS,
    MtpLoopStepModel,
)


class _DummyBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str="auto",
    ):
        return (num_blocks, block_size, num_kv_heads, 2, head_size)


class _UnsupportedSpec(KVCacheSpec):
    @property
    def page_size_bytes(self) -> int:
        return 64


# ---------------------------------------------------------------------------
# MTP Fused Loop & MTP Loop Step Model Tests
# ---------------------------------------------------------------------------


def test_mtp_draft_model_protocol_stubs():
    assert MtpDraftModel.forward(None) is None
    assert MtpDraftModel.compute_logits(None, torch.zeros(1)) is None


def test_mtp_fused_loop_forward_branches_and_int32_guard():
    class _Draft(nn.Module):
        def __init__(self, int32_argmax: bool = False):
            super().__init__()
            self.int32_argmax = int32_argmax
            self.calls = []

        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors=None,
            inputs_embeds=None,
            spec_step_idx=0,
        ):
            self.calls.append((spec_step_idx, positions.clone(), input_ids.clone()))
            base = hidden_states + float(spec_step_idx)
            if spec_step_idx == 1:
                return [base]
            if spec_step_idx == 2:
                return (base, base + 10.0)
            if spec_step_idx == 3:
                return [(base, base + 20.0)]
            return base

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            vocab = torch.stack(
                [hidden_states[:, 0], hidden_states[:, 0] + 5.0], dim=-1
            )
            if self.int32_argmax:

                class _Int32ArgmaxTensor:
                    def argmax(self, dim=-1):
                        return torch.ones(
                            hidden_states.shape[0],
                            dtype=torch.int32,
                            device=hidden_states.device,
                        )

                return _Int32ArgmaxTensor()
            return vocab

    vllm_cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    draft_i64 = _Draft(int32_argmax=False)
    loop_empty = _MtpFusedLoop(
        vllm_config=vllm_cfg, prefix="empty", mtp=draft_i64, num_steps=0
    )
    seq_lens = torch.tensor([4, 7], dtype=torch.int32)
    assert (
        loop_empty(
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3, 6], dtype=torch.int32),
            torch.zeros(2, 4),
            seq_lens,
        )
        == ()
    )

    loop_4 = _MtpFusedLoop(
        vllm_config=vllm_cfg, prefix="four", mtp=draft_i64, num_steps=4
    )
    pos = torch.tensor([3, 6], dtype=torch.int32)
    out = loop_4(
        torch.tensor([1, 2], dtype=torch.int32),
        pos,
        torch.zeros(2, 4),
        seq_lens,
    )
    assert len(out) == 4
    assert all(t.dtype == torch.int32 for t in out)
    # seq_lens was bumped in-place 3 times (for i = 1, 2, 3)
    assert torch.equal(seq_lens, torch.tensor([7, 10], dtype=torch.int32))
    assert [c[0] for c in draft_i64.calls] == [1, 2, 3, 4]
    assert torch.equal(draft_i64.calls[0][1], torch.tensor([3, 6], dtype=torch.int32))
    assert torch.equal(draft_i64.calls[3][1], torch.tensor([6, 9], dtype=torch.int32))

    # Exercise already-int32 argmax fast path
    draft_i32 = _Draft(int32_argmax=True)
    loop_i32 = _MtpFusedLoop(
        vllm_config=vllm_cfg, prefix="i32", mtp=draft_i32, num_steps=1
    )
    out_i32 = loop_i32(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([3, 6], dtype=torch.int32),
        torch.zeros(2, 4),
        torch.tensor([4, 7], dtype=torch.int32),
    )
    assert len(out_i32) == 1
    assert out_i32[0].dtype == torch.int32


def test_build_mtp_fused_loop_caches_compiled_class_by_dynamic_arg_dims(monkeypatch):
    mfl._FUSED_LOOP_CLS_CACHE.clear()
    compile_calls = []

    def fake_support_torch_compile(*, dynamic_arg_dims):
        compile_calls.append(dict(dynamic_arg_dims))
        return lambda cls: cls

    monkeypatch.setattr(mfl, "support_torch_compile", fake_support_torch_compile)
    vllm_cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    dummy_mtp = SimpleNamespace(
        forward=lambda *a, **kw: None, compute_logits=lambda h: h
    )

    dims_a = {"input_ids": 0, "positions": 0}
    dims_b = {"input_ids": 0, "positions": -1}

    mod1 = build_mtp_fused_loop(
        vllm_config=vllm_cfg,
        prefix="p1",
        mtp=dummy_mtp,
        num_steps=2,
        dynamic_arg_dims=dims_a,
    )
    mod2 = build_mtp_fused_loop(
        vllm_config=vllm_cfg,
        prefix="p2",
        mtp=dummy_mtp,
        num_steps=3,
        dynamic_arg_dims={"positions": 0, "input_ids": 0},
    )
    mod3 = build_mtp_fused_loop(
        vllm_config=vllm_cfg,
        prefix="p3",
        mtp=dummy_mtp,
        num_steps=1,
        dynamic_arg_dims=dims_b,
    )

    assert type(mod1) is type(mod2)
    assert type(mod1) is not type(mod3)
    assert len(compile_calls) == 2
    assert mod1.prefix == "p1" and mod1.num_steps == 2
    assert mod2.prefix == "p2" and mod2.num_steps == 3


def test_mtp_loop_step_model_delegates_forward():
    assert "input_ids" in MTP_DYNAMIC_ARG_DIMS

    class _Draft(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors=None,
            inputs_embeds=None,
            spec_step_idx=0,
        ):
            return hidden_states + float(spec_step_idx)

    vllm_cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    draft = _Draft()
    step_model = MtpLoopStepModel(vllm_config=vllm_cfg, prefix="step_loop", mtp=draft)
    assert step_model.prefix == "step_loop"
    assert step_model.mtp is draft

    out = step_model(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        positions=torch.tensor([3, 4], dtype=torch.int32),
        hidden_states=torch.ones(2, 4),
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx=2,
    )
    assert torch.equal(out, torch.full((2, 4), 3.0))


# ---------------------------------------------------------------------------
# KV Cache Materializer Tests
# ---------------------------------------------------------------------------


def _make_attn_spec(
    *,
    block_size: int = 16,
    num_kv_heads: int = 2,
    head_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    page_size_padded: int | None = None,
) -> FullAttentionSpec:
    if page_size_padded is None:
        page_size_padded = block_size * num_kv_heads * 2 * head_size * dtype.itemsize
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
    )


def _make_mamba_spec(
    *,
    block_size: int = 16,
    page_size_padded: int = 1024,
) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((2, 8), (2, 4, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size_padded,
    )


def test_storage_index_and_layout_summary_branches():
    raw0 = torch.zeros((2, 4), dtype=torch.bfloat16)
    raw1 = torch.zeros((2, 4), dtype=torch.bfloat16)
    other = torch.zeros((2, 4), dtype=torch.bfloat16)

    assert _storage_ptr(raw0) == raw0.untyped_storage().data_ptr()
    assert _raw_tensor_index(raw1, [raw0, raw1]) == 1
    assert _raw_tensor_index(other, [raw0, raw1]) is None
    assert _raw_tensor_index(other, []) is None
    assert "raw_idx=0" in _tensor_layout_summary(raw0, [raw0])

    attn_spec = _make_attn_spec()
    mamba_spec = _make_mamba_spec(page_size_padded=attn_spec.page_size_bytes)
    cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["attn.present", "attn.missing"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["mamba.nonempty", "mamba.empty"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )
    kv_caches = {
        "attn.present": raw0,
        "mamba.nonempty": [raw1],
        "mamba.empty": [],
    }

    # With attn_groups=None and empty raw_tensors
    summary_no_groups = format_kv_cache_layout_summary(
        kv_cache_config=cfg,
        kv_caches=kv_caches,
        raw_tensors=[],
        attn_groups=None,
    )
    assert "backend=<none>" in summary_no_groups
    assert "attn.missing: <missing>" in summary_no_groups
    assert "attn.present: attention" in summary_no_groups
    assert "mamba.nonempty: mamba states=1" in summary_no_groups
    assert "mamba.empty: mamba states=0" in summary_no_groups

    # With attn_groups provided (including an empty inner group list) and raw_tensors
    attn_groups = [
        [],
        [
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["attn.present", "attn.missing"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=0,
            )
        ],
    ]
    summary_with_groups = format_kv_cache_layout_summary(
        kv_cache_config=cfg,
        kv_caches=kv_caches,
        raw_tensors=[raw0, raw1],
        attn_groups=attn_groups,
    )
    assert "backend=_DummyBackend" in summary_with_groups
    assert "raw[0]:" in summary_with_groups


def test_pool_attention_geometry_all_branches():
    attn_spec = _make_attn_spec()
    enc_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=attn_spec.page_size_bytes,
    )
    mamba_spec = _make_mamba_spec(page_size_padded=attn_spec.page_size_bytes)

    # No MambaSpec -> None
    cfg_no_mamba = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=attn_spec)],
    )
    assert _pool_attention_geometry(cfg_no_mamba, [], {}) is None

    # Has MambaSpec, but attn_groups only has Encoder/Mamba -> None
    cfg_mamba = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["enc"], kv_cache_spec=enc_spec),
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba_spec),
        ],
    )
    groups_no_full_attn = [
        [],
        [
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["enc"],
                kv_cache_spec=enc_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["m0"],
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=1,
            ),
        ],
    ]
    assert _pool_attention_geometry(cfg_mamba, groups_no_full_attn, {1: 16}) is None

    # Has MambaSpec and a full AttentionSpec -> returns tuple
    groups_with_attn = [
        [
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["a0"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=2,
            )
        ]
    ]
    geom = _pool_attention_geometry(cfg_mamba, groups_with_attn, {})
    assert geom == (attn_spec, _DummyBackend, 16)


def test_layer_to_pool_index_and_resolve_block_major_pool_layout():
    attn_spec = _make_attn_spec()
    mamba_spec = _make_mamba_spec(page_size_padded=attn_spec.page_size_bytes)
    num_blocks = 4
    page_bytes = attn_spec.page_size_bytes
    num_regions = 2
    backing_bytes = num_regions * num_blocks * page_bytes

    # Test both layer-outermost (block_stride == page_bytes) and
    # block-outermost (block_stride == num_regions * page_bytes)
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing_bytes,
                layers=["m0"],
                layer_stride=num_blocks * page_bytes,
                block_stride=page_bytes,
                offset=0,
            ),
            KVCacheTensor(
                size=backing_bytes,
                layers=["a0", "a1"],
                layer_stride=page_bytes,
                block_stride=num_regions * page_bytes,
                offset=0,
            ),
        ],
        # Place MambaSpec first so resolve_block_major_pool_layout iterates past it
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["a0", "a1"], kv_cache_spec=attn_spec),
        ],
    )
    indices = layer_to_pool_index(cfg)
    assert indices == {"m0": 0, "a0": 0, "a1": 1}

    layout = resolve_block_major_pool_layout(cfg, kernel_block_size=8)
    assert layout.num_blocks == num_blocks
    assert layout.num_pools == 2
    assert layout.split == 2
    assert layout.pool_index_by_layer == indices


def test_allocate_raw_kv_cache_tensors_block_major_and_page_mismatch():
    attn_spec = _make_attn_spec(dtype=torch.float8_e4m3fn)
    mamba_spec = _make_mamba_spec(page_size_padded=attn_spec.page_size_bytes)
    num_blocks = 4
    page_bytes = attn_spec.page_size_bytes
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * page_bytes,
                layers=["a0", "m0"],
                layer_stride=num_blocks * page_bytes,
                block_stride=page_bytes,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["a0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba_spec),
        ],
    )

    # Block-major=True with FP8
    layer_to_raw, raw_tensors, layout = allocate_raw_kv_cache_tensors(
        cfg,
        torch.device("cpu"),
        pool_geometry=(attn_spec, _DummyBackend, 8),
        cache_dtype="fp8",
        block_major=True,
    )
    assert layout is not None
    assert len(raw_tensors) == 1
    assert raw_tensors[0].shape == (4, 4, 8, 2, 2, 8)
    assert layer_to_raw["a0"] is raw_tensors[0]
    assert layer_to_raw["m0"] is raw_tensors[0]

    # Block-major=False with bfloat16 (zeros path)
    bf16_spec = _make_attn_spec(dtype=torch.bfloat16)
    bf16_mamba = _make_mamba_spec(page_size_padded=bf16_spec.page_size_bytes)
    bf16_bytes = bf16_spec.page_size_bytes
    bf16_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * bf16_bytes,
                layers=["a0", "m0"],
                layer_stride=num_blocks * bf16_bytes,
                block_stride=bf16_bytes,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["a0"], kv_cache_spec=bf16_spec),
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=bf16_mamba),
        ],
    )
    l2r_bf16, raws_bf16, layout_bf16 = allocate_raw_kv_cache_tensors(
        bf16_cfg,
        torch.device("cpu"),
        pool_geometry=(bf16_spec, _DummyBackend, 8),
        cache_dtype="bfloat16",
        block_major=False,
    )
    assert layout_bf16 is None
    assert len(raws_bf16) == 2
    assert l2r_bf16["a0"] is raws_bf16[0]
    assert l2r_bf16["m0"] is raws_bf16[1]

    # Page mismatch raises AssertionError
    bad_mamba = _make_mamba_spec(page_size_padded=page_bytes * 2)
    bad_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_bytes * 2,
                layers=["a0", "m0"],
                layer_stride=0,
                block_stride=page_bytes * 2,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["a0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=bad_mamba),
        ],
    )
    with pytest.raises(AssertionError, match="pool page mismatch"):
        allocate_raw_kv_cache_tensors(
            bad_cfg,
            torch.device("cpu"),
            pool_geometry=(attn_spec, _DummyBackend, 8),
        )


def test_can_share_attention_cache_and_materialize_shared_attention_cache():
    spec1 = _make_attn_spec(block_size=16, num_kv_heads=2, head_size=8)
    spec2 = _make_attn_spec(block_size=16, num_kv_heads=1, head_size=16)
    mamba = _make_mamba_spec(page_size_padded=spec1.page_size_bytes)
    mla = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=spec1.page_size_bytes,
    )
    bad_page_spec = _make_attn_spec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        page_size_padded=spec1.page_size_bytes + 64,
    )

    assert can_share_attention_cache([], _DummyBackend) is False
    assert can_share_attention_cache([mamba], _DummyBackend) is False
    assert can_share_attention_cache([spec1, mamba], _DummyBackend) is False
    assert can_share_attention_cache([spec1, mla], _DummyBackend) is False
    assert (
        can_share_attention_cache(
            [spec1, replace(spec1, dtype=torch.float32)], _DummyBackend
        )
        is False
    )
    assert (
        can_share_attention_cache([spec1, replace(spec1, block_size=8)], _DummyBackend)
        is False
    )
    assert can_share_attention_cache([spec1, bad_page_spec], _DummyBackend) is False
    assert can_share_attention_cache([bad_page_spec], _DummyBackend) is False
    assert can_share_attention_cache([spec1, spec2], _DummyBackend) is True

    # materialize_shared_attention_cache error branches
    with pytest.raises(ValueError, match="must contain an attention layer"):
        materialize_shared_attention_cache(
            layer_specs=[],
            tensor_size=1024,
            num_blocks=1,
            attn_backend=_DummyBackend,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="Incompatible shared KV cache spec"):
        materialize_shared_attention_cache(
            layer_specs=[("l0", spec1), ("l1", replace(spec1, block_size=8))],
            tensor_size=spec1.page_size_bytes * 2,
            num_blocks=2,
            attn_backend=_DummyBackend,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="whole native pages"):
        materialize_shared_attention_cache(
            layer_specs=[("l0", bad_page_spec)],
            tensor_size=bad_page_spec.page_size_bytes * 2,
            num_blocks=2,
            attn_backend=_DummyBackend,
            device=torch.device("cpu"),
        )

    # Identical shapes and exact page_elements -> native pool_shape
    native_pool = materialize_shared_attention_cache(
        layer_specs=[("l0", spec1), ("l1", spec1)],
        tensor_size=spec1.page_size_bytes * 2,
        num_blocks=2,
        attn_backend=_DummyBackend,
        device=torch.device("cpu"),
    )
    assert native_pool.shape == (2, 16, 2, 2, 8)

    # Mixed shapes with equal page_elements -> packed 4D pool_shape
    mixed_pool = materialize_shared_attention_cache(
        layer_specs=[("l0", spec1), ("l1", spec2)],
        tensor_size=spec1.page_size_bytes * 2,
        num_blocks=2,
        attn_backend=_DummyBackend,
        device=torch.device("cpu"),
    )
    assert mixed_pool.shape == (2, 16, 2, 16)

    # Single layer with padded page_size (2x native) -> packed 4D pool_shape
    double_page_spec = _make_attn_spec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        page_size_padded=spec1.page_size_bytes * 2,
    )
    padded_pool = materialize_shared_attention_cache(
        layer_specs=[("l0", double_page_spec)],
        tensor_size=double_page_spec.page_size_bytes * 2,
        num_blocks=2,
        attn_backend=_DummyBackend,
        device=torch.device("cpu"),
    )
    assert padded_pool.shape == (2, 16, 2, 32)


def test_select_representative_spec_and_build_kernel_block_sizes():
    spec1 = _make_attn_spec()
    enc_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=spec1.page_size_bytes,
    )
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"l0": spec1},
    )
    assert _select_representative_kv_cache_spec(uniform) is spec1

    empty_uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={},
    )
    with pytest.raises(ValueError, match="at least one KV cache spec"):
        _select_representative_kv_cache_spec(empty_uniform)

    unsupported_cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["u0"], kv_cache_spec=_UnsupportedSpec(16))
        ],
    )
    with pytest.raises(NotImplementedError, match="Unsupported KV cache spec"):
        build_kernel_block_size_by_group_id(
            kv_cache_config=unsupported_cfg,
            kernel_block_sizes=[16],
        )

    valid_cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["enc"], kv_cache_spec=enc_spec),
            KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec1),
        ],
    )
    with pytest.raises(ValueError, match="positive values"):
        build_kernel_block_size_by_group_id(
            kv_cache_config=valid_cfg,
            kernel_block_sizes=[0],
        )
    with pytest.raises(ValueError, match="missing an entry"):
        build_kernel_block_size_by_group_id(
            kv_cache_config=valid_cfg,
            kernel_block_sizes=[],
        )
    with pytest.raises(ValueError, match="unused entries"):
        build_kernel_block_size_by_group_id(
            kv_cache_config=valid_cfg,
            kernel_block_sizes=[16, 16],
        )


def test_materialize_kv_cache_tensors_aliased_regions_and_error_branches():
    attn_spec = _make_attn_spec()
    enc_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=attn_spec.page_size_bytes,
    )
    mamba_spec = _make_mamba_spec(page_size_padded=attn_spec.page_size_bytes)
    num_blocks = 2
    page_bytes = attn_spec.page_size_bytes

    # 1. Direct attention with aliased region (matching geometry) + encoder gap
    aliased_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=["a0"],
                layer_stride=0,
                block_stride=page_bytes,
            ),
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=["a1"],
                layer_stride=0,
                block_stride=page_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["enc"], kv_cache_spec=enc_spec),
            KVCacheGroupSpec(layer_names=["a0", "a1"], kv_cache_spec=attn_spec),
        ],
    )
    assert _can_allocate_attention_cache_directly(aliased_cfg) is True
    mat = materialize_kv_cache_tensors(
        kv_cache_config=aliased_cfg,
        attn_groups=[
            [
                AttentionGroup(
                    backend=_DummyBackend,
                    layer_names=["enc"],
                    kv_cache_spec=enc_spec,
                    kv_cache_group_id=0,
                ),
                AttentionGroup(
                    backend=_DummyBackend,
                    layer_names=["a0", "a1"],
                    kv_cache_spec=attn_spec,
                    kv_cache_group_id=1,
                ),
            ]
        ],
        kernel_block_sizes=[16],
        device=torch.device("cpu"),
        cache_dtype=torch.bfloat16,
    )
    assert mat.kv_caches["a0"] is mat.kv_caches["a1"]
    assert make_attention_cache_tensor(
        spec=attn_spec,
        attn_backend=_DummyBackend,
        num_blocks=num_blocks,
        kernel_block_size=16,
        cache_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    ).shape == (2, 16, 2, 2, 8)

    # 1b. Direct attention with aliased region + mismatched geometry -> error
    mismatched_spec = replace(attn_spec, num_kv_heads=1, head_size=16)
    mismatched_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=["a0"],
                layer_stride=0,
                block_stride=page_bytes,
            ),
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=["a1"],
                layer_stride=0,
                block_stride=page_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["a0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["a1"], kv_cache_spec=mismatched_spec),
        ],
    )
    with pytest.raises(NotImplementedError, match="identical native cache geometry"):
        materialize_kv_cache_tensors(
            kv_cache_config=mismatched_cfg,
            attn_groups=[
                [
                    AttentionGroup(
                        backend=_DummyBackend,
                        layer_names=["a0"],
                        kv_cache_spec=attn_spec,
                        kv_cache_group_id=0,
                    ),
                    AttentionGroup(
                        backend=_DummyBackend,
                        layer_names=["a1"],
                        kv_cache_spec=mismatched_spec,
                        kv_cache_group_id=1,
                    ),
                ]
            ],
            kernel_block_sizes=[16, 16],
            device=torch.device("cpu"),
            cache_dtype=torch.bfloat16,
        )

    # 2. Hybrid config without an attention pool in attn_groups raises ValueError
    mamba_only_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=["m0"],
                layer_stride=0,
                block_stride=page_bytes,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba_spec)
        ],
    )
    with pytest.raises(ValueError, match="requires an attention pool"):
        materialize_kv_cache_tensors(
            kv_cache_config=mamba_only_cfg,
            attn_groups=[],
            kernel_block_sizes=[16],
            device=torch.device("cpu"),
            cache_dtype=torch.bfloat16,
        )

    # 3. Hybrid config happy path (with encoder gap) and unsupported spec error path
    hybrid_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=["a0", "m0"],
                layer_stride=0,
                block_stride=page_bytes,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["enc"], kv_cache_spec=enc_spec),
            KVCacheGroupSpec(layer_names=["a0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba_spec),
        ],
    )
    good_hybrid_groups = [
        [
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["enc"],
                kv_cache_spec=enc_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["a0"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=1,
            ),
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["m0"],
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=2,
            ),
        ]
    ]
    hybrid_mat = materialize_kv_cache_tensors(
        kv_cache_config=hybrid_cfg,
        attn_groups=good_hybrid_groups,
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype=torch.bfloat16,
    )
    assert hybrid_mat.kv_caches["m0"] == [hybrid_mat.raw_tensors[0]]
    assert hybrid_mat.kv_caches["a0"] is hybrid_mat.raw_tensors[0]

    bad_attn_groups = [
        [
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["a0"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=1,
            ),
            AttentionGroup(
                backend=_DummyBackend,
                layer_names=["m0"],
                kv_cache_spec=_UnsupportedSpec(16),
                kv_cache_group_id=2,
            ),
        ]
    ]
    with pytest.raises(NotImplementedError, match="Unsupported KV cache spec"):
        materialize_kv_cache_tensors(
            kv_cache_config=hybrid_cfg,
            attn_groups=bad_attn_groups,
            kernel_block_sizes=[16, 16],
            device=torch.device("cpu"),
            cache_dtype=torch.bfloat16,
        )
