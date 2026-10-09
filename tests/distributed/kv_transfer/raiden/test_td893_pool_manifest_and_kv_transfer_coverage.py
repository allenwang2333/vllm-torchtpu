# SPDX-License-Identifier: Apache-2.0
"""TD8.39: 100% statement and branch coverage tests for Raiden KV transfer."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.raiden import (
    pool_manifest,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
    BINDING_ALIASED_RAW,
    DeadStorageError,
    FaLayoutHook,
    GdnHeadGeometry,
    ManifestError,
    PoolEntry,
    PoolManifest,
    RegionSpec,
    build_dsv4_pool_manifest,
    build_glm_mla_pool_manifest,
    build_kimi_k3_pool_manifest,
    build_qwen35_pool_manifest,
    layer_index_from_name,
    materialize_storages,
    verify_storage_binding,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.tags import (
    TAG_FA,
)


@pytest.fixture(autouse=True)
def _clean_pep562_env_dicts(monkeypatch):
    yield
    monkeypatch.undo()
    from vllm import envs as vllm_envs

    for mod in (tpu_envs, vllm_envs):
        env_vars = getattr(mod, "environment_variables", {})
        for key in list(mod.__dict__):
            if key in env_vars:
                delattr(mod, key)


# Re-export existing Raiden test suites so this file achieves 100% standalone coverage.

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _suite_idx, _mod_path in enumerate(
    (
        "tests.distributed.test_pool_byte_spans",
        "tests.distributed.test_pcp_tp_dcp_spans",
        "tests.distributed.test_raiden_pool_manifest",
        "tests.distributed.test_raiden_layout_fingerprint",
        "tests.distributed.test_seq_on_lane",
        "tests.distributed.test_raiden_telemetry_utils",
    )
):
    _mod = importlib.import_module(_mod_path)
    for _attr_name, _attr_val in vars(_mod).items():
        if _attr_name.startswith("test_") and callable(_attr_val):
            globals()[f"test_suite{_suite_idx}_{_attr_name[5:]}"] = _attr_val
        elif _attr_name.startswith("Test") and isinstance(_attr_val, type):
            globals()[f"TestSuite{_suite_idx}_{_attr_name[4:]}"] = _attr_val


# ---------------------------------------------------------------------------
# 1. tags.py & telemetry_utils.py
# ---------------------------------------------------------------------------


def test_pool_manifest_helpers_and_qwen35_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert RegionSpec("r", 12, 64, 32, 0).extent_end_bytes == 12

    p1 = PoolEntry(
        tag="fa.l0",
        layer_name="layers.0",
        storage_index=0,
        base_offset_bytes=0,
        block_stride_bytes=1024,
        num_blocks=2,
        regions=(RegionSpec("r", 0, 1024, 1024, 1, 1),),
        dtype_tag="uint8",
    )
    p2 = PoolEntry(
        tag="fa.l1",
        layer_name="layers.1",
        storage_index=0,
        base_offset_bytes=0,
        block_stride_bytes=2048,
        num_blocks=2,
        regions=(RegionSpec("r", 0, 2048, 2048, 1, 1),),
        dtype_tag="uint8",
    )
    m_div = PoolManifest("private_typed", [object()], [p1, p2])
    assert m_div.dtype_tags() == ["uint8", "uint8"]
    with pytest.raises(ManifestError, match="pool geometry diverges"):
        m_div.geometry_by_tag()

    t_no_nbytes = SimpleNamespace(numel=lambda: 8, element_size=lambda: 2)
    assert pool_manifest._nbytes(t_no_nbytes) == 16

    t_dtype_itemsize = SimpleNamespace(dtype=SimpleNamespace(itemsize=4))
    assert pool_manifest._element_size(t_dtype_itemsize) == 4
    with pytest.raises(ManifestError, match="cannot determine element size"):
        pool_manifest._element_size(SimpleNamespace(dtype=None))

    t_bad_storage = SimpleNamespace(
        untyped_storage=lambda: (_ for _ in ()).throw(RuntimeError("no storage"))
    )
    assert pool_manifest._storage_ptr(t_bad_storage) is None
    assert pool_manifest._storage_offset_bytes(SimpleNamespace()) == 0

    with pytest.raises(ManifestError, match="are not .* rows of .* regions"):
        pool_manifest._region_placement(
            SimpleNamespace(num_blocks=0, num_pools=1, pool_index_by_layer={}),  # type: ignore[arg-type]
            SimpleNamespace(nbytes=1024),
            "layers.0",
        )

    assert layer_index_from_name("model.layers.not_an_int.attn") is None
    assert layer_index_from_name("model.embed_tokens") is None

    with pytest.raises(
        ManifestError, match="live bytes per token exceed the physical token stride"
    ):
        pool_manifest._fa_regions(
            block_size_tokens=16,
            token_stride_bytes=64,
            num_kv_heads=4,
            head_size=16,
            itemsize=2,
        )

    geom = GdnHeadGeometry(
        local_key_heads=2,
        local_value_heads=2,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_size=4,
    )
    # 4D conv shape with singleton at dim 1: (blocks, 1, taps, dim)
    monkeypatch.setattr(tpu_envs, "TPU_GDN_CONV_QK_PAIR_LAYOUT", True)
    regs_pair = pool_manifest._gdn_conv_regions(
        conv_shape=(2, 1, 3, geom.conv_dim), itemsize=2, geometry=geom
    )
    assert len(regs_pair) == 2 and regs_pair[0].name == "gdn_conv_qk"

    monkeypatch.setattr(tpu_envs, "TPU_GDN_CONV_QK_PAIR_LAYOUT", False)
    regs_legacy = pool_manifest._gdn_conv_regions(
        conv_shape=(2, 3, geom.conv_dim), itemsize=2, geometry=geom
    )
    assert len(regs_legacy) == 3 and regs_legacy[0].name == "gdn_conv_q"

    with pytest.raises(ManifestError, match="GDN conv state must be"):
        pool_manifest._gdn_conv_regions(
            conv_shape=(2, geom.conv_dim), itemsize=2, geometry=geom
        )
    with pytest.raises(ManifestError, match="GDN ssm state must be"):
        pool_manifest._gdn_ssm_regions(ssm_shape=(2, 4, 16), itemsize=4)
    with pytest.raises(ManifestError, match="is in no kv cache group"):
        pool_manifest._group_spec_for_layer([], "layers.0")


def test_pooled_gdn_and_kda_and_block_major_edges() -> None:
    raw = SimpleNamespace(
        nbytes=8192,
        dtype="uint8",
        element_size=lambda: 1,
        storage_offset=lambda: 0,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 1000),
    )
    geom_no_taps = GdnHeadGeometry(2, 2, 16, 16, conv_kernel_size=1)
    spec_ok = SimpleNamespace(
        shapes=((3, 96), (2, 16, 16)),
        dtypes=("bfloat16", "float32"),
        page_size_bytes=4096,
    )
    with pytest.raises(ManifestError, match="has no conv taps"):
        pool_manifest._pooled_gdn_state_views(
            pool=raw,
            spec=spec_ok,
            raw_tensors=[raw],
            layer_name="layers.0",
            gdn_geometry=geom_no_taps,
        )

    geom = GdnHeadGeometry(2, 2, 16, 16, conv_kernel_size=4)
    spec_bad_dtype = SimpleNamespace(
        shapes=((3, 96), (2, 16, 16)),
        dtypes=("bfloat16", "unknown_dtype"),
        page_size_bytes=4096,
    )
    with pytest.raises(ManifestError):
        pool_manifest._pooled_gdn_state_views(
            pool=raw,
            spec=spec_bad_dtype,
            raw_tensors=[raw],
            layer_name="layers.0",
            gdn_geometry=geom,
        )

    spec_no_page = SimpleNamespace(
        shapes=((3, 96), (2, 16, 16)),
        dtypes=("bfloat16", "float32"),
        page_size_bytes=None,
    )
    with pytest.raises(ManifestError, match="has no manager page size"):
        pool_manifest._pooled_gdn_state_views(
            pool=raw,
            spec=spec_no_page,
            raw_tensors=[raw],
            layer_name="layers.0",
            gdn_geometry=geom,
        )

    with pytest.raises(ManifestError, match="does not match the manager page"):
        pool_manifest._pooled_gdn_state_views(
            pool=raw,
            spec=spec_ok,
            raw_tensors=[raw],
            layer_name="layers.0",
            gdn_geometry=geom,
            placement=pool_manifest._RegionPlacement(
                num_blocks=2, region_bytes=2048, region_offset_bytes=0
            ),
        )

    with pytest.raises(ManifestError, match="is not aligned to dtype"):
        pool_manifest._pooled_gdn_state_views(
            pool=raw,
            spec=spec_ok,
            raw_tensors=[raw],
            layer_name="layers.0",
            gdn_geometry=geom,
            placement=pool_manifest._RegionPlacement(
                num_blocks=2, region_bytes=4096, region_offset_bytes=1
            ),
        )

    # _pooled_kda_regions errors
    with pytest.raises(
        ManifestError, match="pooled KDA cache requires conv and SSM specs"
    ):
        pool_manifest._pooled_kda_regions(
            spec=SimpleNamespace(shapes=((1,),), dtypes=("bfloat16",)),
            row_bytes=512,
            page_bytes=4096,
        )
    with pytest.raises(ManifestError):
        pool_manifest._pooled_kda_regions(
            spec=SimpleNamespace(
                shapes=((3, 64), (2, 16, 16)),
                dtypes=("bfloat16", "bad_dtype"),
            ),
            row_bytes=512,
            page_bytes=4096,
        )
    with pytest.raises(ManifestError, match="KDA state exceeds its pooled page"):
        pool_manifest._pooled_kda_regions(
            spec=SimpleNamespace(
                shapes=((3, 512), (4, 64, 64)),
                dtypes=("bfloat16", "float32"),
            ),
            row_bytes=512,
            page_bytes=1024,
        )
    with pytest.raises(
        ManifestError,
        match="KDA transfer requires f32 SSM and word-aligned pool rows",
    ):
        pool_manifest._pooled_kda_regions(
            spec=SimpleNamespace(
                shapes=((3, 16), (1, 8, 8)),
                dtypes=("bfloat16", "bfloat16"),
            ),
            row_bytes=512,
            page_bytes=4096,
        )

    # _block_major_fa_view errors
    pool_2b = SimpleNamespace(
        dtype="bfloat16",
        element_size=lambda: 2,
        storage_offset=lambda: 0,
    )
    with pytest.raises(ManifestError, match="whole tokens"):
        pool_manifest._block_major_fa_view(
            pool=pool_2b,
            spec=SimpleNamespace(block_size=16),
            layer_name="layers.0",
            placement=pool_manifest._RegionPlacement(2, 100, 0),
        )
    with pytest.raises(ManifestError, match="is not aligned to its dtype"):
        pool_manifest._block_major_fa_view(
            pool=pool_2b,
            spec=SimpleNamespace(block_size=16),
            layer_name="layers.0",
            placement=pool_manifest._RegionPlacement(2, 1024, 1),
        )

    # _StorageTable with None ptr and FaLayoutHook abstract methods
    st = pool_manifest._StorageTable()
    obj_no_ptr = SimpleNamespace()
    assert st.index_for(obj_no_ptr) == 0
    assert st.index_for(obj_no_ptr) == 0

    base_hook = FaLayoutHook()
    with pytest.raises(NotImplementedError):
        base_hook.total_tokens((2, 16))
    with pytest.raises(NotImplementedError):
        base_hook.regions(
            block_size_tokens=16,
            shape=(2, 16),
            nbytes=1024,
            num_kv_heads=2,
            head_size=16,
            itemsize=1,
        )


def test_build_qwen35_kimi_glm_and_bind_manifest_errors() -> None:
    geom = GdnHeadGeometry(2, 2, 16, 16, conv_kernel_size=4)
    raw = SimpleNamespace(
        nbytes=4096,
        dtype="uint8",
        element_size=lambda: 1,
        storage_offset=lambda: 0,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 500),
    )
    bm = SimpleNamespace(
        num_blocks=2,
        num_pools=1,
        pool_index_by_layer={"layers.0": 0},
    )
    with pytest.raises(
        ManifestError, match="must expose the block-major pool, not private"
    ):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": (raw, raw)},
            kv_cache_groups=[],
            raw_tensors=[raw],
            gdn_geometry=geom,
            block_major=bm,  # type: ignore[arg-type]
        )
    with pytest.raises(
        ManifestError, match="must have a unified pool or \\(conv, ssm\\) states"
    ):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": (raw, raw, raw)},
            kv_cache_groups=[],
            raw_tensors=[],
            gdn_geometry=geom,
        )
    private_fa = SimpleNamespace(
        nbytes=4096,
        shape=(2, 16, 2, 16),
        dtype="uint8",
        element_size=lambda: 1,
        storage_offset=lambda: 0,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 999),
    )
    with pytest.raises(ManifestError, match="must alias the block-major pool"):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": private_fa},
            kv_cache_groups=[],
            raw_tensors=[raw],
            gdn_geometry=geom,
            block_major=bm,  # type: ignore[arg-type]
        )
    with pytest.raises(ManifestError, match="per-layer pool tags need a layer index"):
        build_qwen35_pool_manifest(
            named_kv_caches={"attn_no_index": private_fa},
            kv_cache_groups=[],
            raw_tensors=[],
            gdn_geometry=geom,
            per_layer_tags=True,
        )

    fa_group = SimpleNamespace(
        layer_names=["layers.0"],
        kv_cache_spec=SimpleNamespace(block_size=16, num_kv_heads=2, head_size=16),
    )
    fa_1d = SimpleNamespace(
        nbytes=1024, shape=(1024,), dtype="uint8", element_size=lambda: 1
    )
    with pytest.raises(ManifestError, match="needs a paged shape"):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": fa_1d},
            kv_cache_groups=[fa_group],
            raw_tensors=[],
            gdn_geometry=geom,
        )
    fa_bad_cap = SimpleNamespace(
        nbytes=100, shape=(1, 10), dtype="uint8", element_size=lambda: 1
    )
    with pytest.raises(
        ManifestError, match="is not divisible by the logical block size"
    ):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": fa_bad_cap},
            kv_cache_groups=[fa_group],
            raw_tensors=[],
            gdn_geometry=geom,
        )
    fa_bad_nbytes = SimpleNamespace(
        nbytes=33, shape=(2, 16), dtype="uint8", element_size=lambda: 1
    )
    with pytest.raises(ManifestError, match="is not divisible by token capacity"):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": fa_bad_nbytes},
            kv_cache_groups=[fa_group],
            raw_tensors=[],
            gdn_geometry=geom,
        )

    conv_scalar = SimpleNamespace(
        nbytes=64, shape=(), dtype="bfloat16", element_size=lambda: 2
    )
    with pytest.raises(ManifestError, match="has no shape"):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": (conv_scalar, conv_scalar)},
            kv_cache_groups=[],
            raw_tensors=[],
            gdn_geometry=geom,
        )
    conv_bad_div = SimpleNamespace(
        nbytes=65,
        shape=(2, 3, geom.conv_dim),
        dtype="bfloat16",
        element_size=lambda: 2,
    )
    with pytest.raises(ManifestError, match="is not divisible by num_blocks"):
        build_qwen35_pool_manifest(
            named_kv_caches={"layers.0": (conv_bad_div, conv_bad_div)},
            kv_cache_groups=[],
            raw_tensors=[],
            gdn_geometry=geom,
        )

    # build_kimi_k3_pool_manifest errors
    with pytest.raises(ManifestError, match="must expose one unified pool"):
        build_kimi_k3_pool_manifest(
            named_kv_caches={"layers.0": (raw, raw)},
            kv_cache_groups=[],
            raw_tensors=[],
        )
    kimi_tensor = SimpleNamespace(
        shape=(2, 4, 4, 128),
        dtype="float8_e4m3fn",
        nbytes=4096,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="block size does not match its rows"):
        build_kimi_k3_pool_manifest(
            named_kv_caches={"layers.0": kimi_tensor},
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layers.0"],
                    kv_cache_spec=SimpleNamespace(page_size_bytes=2048, block_size=32),
                )
            ],
            raw_tensors=[],
        )
    with pytest.raises(ManifestError, match="do not match its packed-row stride"):
        build_kimi_k3_pool_manifest(
            named_kv_caches={"layers.0": kimi_tensor},
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layers.0"],
                    kv_cache_spec=SimpleNamespace(page_size_bytes=1024, block_size=16),
                )
            ],
            raw_tensors=[],
        )

    with pytest.raises(ManifestError, match="named_kv_caches is empty"):
        pool_manifest._ordered_layers({})
    with pytest.raises(ManifestError, match="has no mamba group ordinal"):
        pool_manifest._state_tag_suffix("layers.0", {})

    # _bind_pool_manifest errors
    entry_2blocks = PoolEntry(
        tag=TAG_FA,
        layer_name="layers.0",
        storage_index=-1,
        base_offset_bytes=0,
        block_stride_bytes=1024,
        num_blocks=2,
        regions=(RegionSpec("r", 0, 1024, 1024, 1, 1),),
        dtype_tag="uint8",
    )
    raw_odd = SimpleNamespace(
        nbytes=1025,
        storage_offset=lambda: 0,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 777),
    )
    with pytest.raises(ManifestError, match="raw storage bytes .* are not divisible"):
        pool_manifest._bind_pool_manifest([(raw_odd, entry_2blocks)], [raw_odd])

    raw_2048 = SimpleNamespace(
        nbytes=2048,
        element_size=lambda: 1,
        storage_offset=lambda: 0,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 777),
    )
    typed_oob = SimpleNamespace(
        nbytes=1024,
        element_size=lambda: 1,
        storage_offset=lambda: 1024,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 777),
    )
    with pytest.raises(ManifestError, match="outside one raw page"):
        pool_manifest._bind_pool_manifest([(typed_oob, entry_2blocks)], [raw_2048])

    entry_over = PoolEntry(
        tag=TAG_FA,
        layer_name="layers.0",
        storage_index=-1,
        base_offset_bytes=512,
        block_stride_bytes=1024,
        num_blocks=2,
        regions=(RegionSpec("r", 0, 1024, 1024, 1, 1),),
        dtype_tag="uint8",
    )
    with pytest.raises(ManifestError, match="extends through byte"):
        pool_manifest._bind_pool_manifest([(raw_2048, entry_over)], [raw_2048])

    entry_3blocks = dataclasses.replace(
        entry_2blocks, layer_name="layers.1", num_blocks=3
    )
    priv1 = SimpleNamespace(nbytes=2048)
    priv2 = SimpleNamespace(nbytes=3072)
    with pytest.raises(ManifestError, match="pools disagree on num_blocks"):
        pool_manifest._bind_pool_manifest(
            [(priv1, entry_2blocks), (priv2, entry_3blocks)], []
        )


def test_materialize_verify_glm_and_dsv4_manifest_edges() -> None:
    called: list[str] = []

    class _FakeStorageOk:
        def __getitem__(self, item: Any) -> Any:
            return self

        def cpu(self) -> None:
            called.append("ok")

    class _FakeStorageErr:
        def __getitem__(self, item: Any) -> Any:
            raise RuntimeError("slice error")

        def cpu(self) -> None:
            called.append("err")

    materialize_storages(
        PoolManifest(
            "private_typed",
            [_FakeStorageOk(), _FakeStorageErr(), SimpleNamespace()],
            [],
        )
    )
    assert called == ["ok"]

    # verify_storage_binding with None ptrs and aliased_raw errors
    no_ptr = SimpleNamespace()
    raw_ptr_1 = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 101)
    )
    raw_ptr_2 = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 202)
    )
    with pytest.raises(
        DeadStorageError,
        match="aliased_raw pool storages must be raw unified tensors",
    ):
        verify_storage_binding(
            PoolManifest(BINDING_ALIASED_RAW, [raw_ptr_2, no_ptr], []),
            {"layers.0": raw_ptr_1, "layers.1": no_ptr},
            [raw_ptr_1, no_ptr],
        )
    with pytest.raises(
        DeadStorageError,
        match="aliased_raw binding requires every typed KV cache",
    ):
        verify_storage_binding(
            PoolManifest(BINDING_ALIASED_RAW, [raw_ptr_1], []),
            {"layers.0": raw_ptr_2},
            [raw_ptr_1],
        )

    # build_glm_mla_pool_manifest errors
    with pytest.raises(ManifestError, match="block_size_tokens must be positive"):
        build_glm_mla_pool_manifest(
            named_kv_caches={}, raw_tensors=[], block_size_tokens=0
        )
    with pytest.raises(
        ManifestError, match="GLM MLA admission expects \\(nope, rope\\)"
    ):
        build_glm_mla_pool_manifest(
            named_kv_caches={"layers.0": (1, 2, 3)},
            raw_tensors=[],
            block_size_tokens=16,
        )
    with pytest.raises(ManifestError, match="found no latent \\(mla.nope\\) caches"):
        build_glm_mla_pool_manifest(
            named_kv_caches={"layers.0": SimpleNamespace(dtype="uint8")},
            raw_tensors=[],
            block_size_tokens=16,
        )
    nope_ok = SimpleNamespace(
        dtype="uint8",
        shape=(2, 16, 4, 128),
        nbytes=2 * 16 * 512,
        element_size=lambda: 1,
    )
    rope_ok = SimpleNamespace(
        dtype="bfloat16",
        shape=(2, 8, 2, 128),
        nbytes=2 * 8 * 512,
        element_size=lambda: 2,
    )
    with pytest.raises(ManifestError, match="per-layer pool tags need a layer index"):
        build_glm_mla_pool_manifest(
            named_kv_caches={"no_layer_idx": (nope_ok, rope_ok)},
            raw_tensors=[],
            block_size_tokens=16,
            per_layer_tags=True,
        )
    nope_bad_nbytes = SimpleNamespace(
        dtype="uint8",
        shape=(2, 16, 4, 128),
        nbytes=1025,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="is not divisible by num_blocks"):
        build_glm_mla_pool_manifest(
            named_kv_caches={"layers.0": (nope_bad_nbytes, rope_ok)},
            raw_tensors=[],
            block_size_tokens=16,
        )
    nope_bad_stride = SimpleNamespace(
        dtype="uint8",
        shape=(2, 16, 4, 128),
        nbytes=2 * 16 * 256,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="block stride .* does not match"):
        build_glm_mla_pool_manifest(
            named_kv_caches={"layers.0": (nope_bad_stride, rope_ok)},
            raw_tensors=[],
            block_size_tokens=16,
        )

    # DSv4 helper & manifest error branches
    s_direct = SimpleNamespace(num_states=16, tokens_per_state=1)
    assert pool_manifest._dsv4_layer_spec(s_direct, "layers.0") is s_direct
    with pytest.raises(ManifestError, match="is not in its own cache group's spec map"):
        pool_manifest._dsv4_layer_spec(
            SimpleNamespace(kv_cache_specs={"layers.1": s_direct}), "layers.0"
        )
    with pytest.raises(ManifestError, match="vLLM renamed it"):
        pool_manifest._dsv4_spec_field(SimpleNamespace(), "missing", "layers.0")

    with pytest.raises(ManifestError, match="has no allocated KV cache"):
        pool_manifest._dsv4_classify(
            [
                SimpleNamespace(
                    layer_names=["layers.0.swa_cache"],
                    kv_cache_spec=s_direct,
                )
            ],
            {},
        )
    with pytest.raises(
        ManifestError, match="names a compressed-KV layer .* that is not an MLA cache"
    ):
        pool_manifest._dsv4_classify(
            [
                SimpleNamespace(
                    layer_names=["layers.0.attn.compressor.state_cache"],
                    kv_cache_spec=s_direct,
                )
            ],
            {"layers.0.attn.compressor.state_cache": SimpleNamespace()},
        )
    csa_spec = SimpleNamespace(num_states=16, tokens_per_state=4)
    with pytest.raises(ManifestError, match="must bind a \\(nope, rope\\) pair"):
        pool_manifest._dsv4_classify(
            [SimpleNamespace(layer_names=["layers.0.attn"], kv_cache_spec=csa_spec)],
            {"layers.0.attn": SimpleNamespace()},
        )
    with pytest.raises(ManifestError, match="must bind a single array"):
        pool_manifest._dsv4_classify(
            [SimpleNamespace(layer_names=["layers.0.attn"], kv_cache_spec=s_direct)],
            {"layers.0.attn": (SimpleNamespace(), SimpleNamespace())},
        )

    with pytest.raises(ManifestError, match="named_kv_caches is empty"):
        build_dsv4_pool_manifest(named_kv_caches={}, kv_cache_groups=[], raw_tensors=[])
    with pytest.raises(ManifestError, match="needs at least one KV cache group"):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0": SimpleNamespace()},
            kv_cache_groups=[],
            raw_tensors=[],
        )

    hca_arr = SimpleNamespace(
        dtype="uint8",
        shape=(2, 32, 4, 128),
        nbytes=2 * 32 * 512,
        element_size=lambda: 1,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 303),
    )
    hca_group = SimpleNamespace(
        layer_names=["layers.0.attn"],
        kv_cache_spec=s_direct,
    )
    with pytest.raises(
        ManifestError, match="DSv4 admission requires private typed cache tensors"
    ):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": hca_arr},
            kv_cache_groups=[hca_group],
            raw_tensors=[hca_arr],
        )

    arr_fp16 = SimpleNamespace(
        dtype="float16",
        shape=(2, 32, 2, 128),
        nbytes=2 * 32 * 512,
        element_size=lambda: 2,
    )
    with pytest.raises(ManifestError, match="must be uint8; got float16"):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": arr_fp16},
            kv_cache_groups=[hca_group],
            raw_tensors=[],
        )

    arr_3d_u8 = SimpleNamespace(
        dtype="uint8",
        shape=(2, 32, 128),
        nbytes=2 * 32 * 128,
        element_size=lambda: 1,
    )
    with pytest.raises(
        ManifestError, match="must be \\[blocks, rows, packing, width\\]"
    ):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": arr_3d_u8},
            kv_cache_groups=[hca_group],
            raw_tensors=[],
        )

    arr_bad_nbytes = SimpleNamespace(
        dtype="uint8",
        shape=(2, 32, 4, 128),
        nbytes=1025,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="is not divisible by num_blocks"):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": arr_bad_nbytes},
            kv_cache_groups=[hca_group],
            raw_tensors=[],
        )

    arr_bad_pack = SimpleNamespace(
        dtype="uint8",
        shape=(2, 32, 2, 128),
        nbytes=2 * 32 * 256,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="does not fill one 32-bit word"):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": arr_bad_pack},
            kv_cache_groups=[hca_group],
            raw_tensors=[],
        )

    arr_bad_width = SimpleNamespace(
        dtype="uint8",
        shape=(2, 32, 4, 100),
        nbytes=2 * 32 * 400,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="is not lane-aligned"):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": arr_bad_width},
            kv_cache_groups=[hca_group],
            raw_tensors=[],
        )

    arr_bad_stride = SimpleNamespace(
        dtype="uint8",
        shape=(2, 32, 4, 128),
        nbytes=2 * 32 * 256,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="block stride .* does not match"):
        build_dsv4_pool_manifest(
            named_kv_caches={"layers.0.attn": arr_bad_stride},
            kv_cache_groups=[hca_group],
            raw_tensors=[],
        )

    hca_arr_3blocks = SimpleNamespace(
        dtype="uint8",
        shape=(3, 32, 4, 128),
        nbytes=3 * 32 * 512,
        element_size=lambda: 1,
    )
    with pytest.raises(ManifestError, match="pools disagree on num_blocks"):
        build_dsv4_pool_manifest(
            named_kv_caches={
                "layers.0.attn": hca_arr,
                "layers.1.swa_cache": hca_arr_3blocks,
            },
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layers.0.attn", "layers.1.swa_cache"],
                    kv_cache_spec=s_direct,
                )
            ],
            raw_tensors=[],
        )


# ---------------------------------------------------------------------------
# zmq_shm_base.py remaining lines (1233-1234, 1235->1223, 1242)
# ---------------------------------------------------------------------------


def test_zmq_shm_base_coord_stage_waiter_loop_empty_and_done_future():
    import queue
    import threading

    from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import (
        ZmqShmKvConnectorBase,
    )

    dummy_entry = SimpleNamespace()
    items = [queue.Empty(), dummy_entry, None]

    class FakeQueue:
        def get(self, timeout=None):
            item = items.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeFuture:
        def done(self):
            return True

    class FakePool:
        def submit(self, fn, entry):
            return FakeFuture()

    worker = SimpleNamespace(
        _coord_stage_wait_timeout_s=10.0,
        _stop_event=threading.Event(),
        _stage_pending_q=FakeQueue(),
        _coord_stage_waiter_pool=FakePool(),
        _coord_stage_wait_worker=lambda entry: None,
    )
    ZmqShmKvConnectorBase._coord_stage_waiter_loop(worker)


# ---------------------------------------------------------------------------
# tpu_connector.py Part 1 (lines 1 - 3600)
# ---------------------------------------------------------------------------


def test_tpu_connector_helpers_and_utilities(monkeypatch):
    import sys

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs

    # Line 120: _DoneFuture.wait()
    assert tc._DONE_FUTURE.wait() is None

    # Line 232: _stage3_trim_window_pages non-null dropped block
    with pytest.raises(ValueError, match="dropped pages that are not the null block"):
        tc._stage3_trim_window_pages(
            [5, 6, 7],
            num_tokens=64,
            page_tokens=16,
            window_tokens=32,
            group_label="g0",
        )

    # Line 483: _get_extra_config when kv_transfer_config is None
    assert tc._get_extra_config(SimpleNamespace(kv_transfer_config=None)) == {}

    # Lines 489, 492-498: _as_bool
    assert tc._as_bool(None, default=True) is True
    assert tc._as_bool("true") is True
    assert tc._as_bool("0") is False
    assert tc._as_bool("custom") is True
    assert tc._as_bool(1) is True

    # Lines 556, 558, 560, 562: stage3_fa_raiden_id_fields validation
    with pytest.raises(ValueError, match="job_name must not be empty"):
        tc.stage3_fa_raiden_id_fields(job_name=" ", engine_id="e", dp_rank=0)
    with pytest.raises(ValueError, match="engine_id must not be empty"):
        tc.stage3_fa_raiden_id_fields(job_name="j", engine_id=" ", dp_rank=0)
    with pytest.raises(ValueError, match="dp_rank must be non-negative"):
        tc.stage3_fa_raiden_id_fields(job_name="j", engine_id="e", dp_rank=-1)
    with pytest.raises(ValueError, match="transfer_rank must be non-negative"):
        tc.stage3_fa_raiden_id_fields(
            job_name="j", engine_id="e", dp_rank=0, transfer_rank=-1
        )

    # Line 585: _pipeline_parallel_size with parallel_config=None
    assert tc._pipeline_parallel_size(SimpleNamespace(parallel_config=None)) == 1

    # Line 614: _raiden_dst_shards <= 0
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 0)
    with pytest.raises(ValueError, match="TPU_RAIDEN_DST_SHARDS must be positive"):
        tc._raiden_dst_shards()

    # Line 626: _raiden_dst_dcp_size invalid
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 2)
    bad_dcp_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"destination_decode_context_parallel_size": 3}
        )
    )
    with pytest.raises(
        ValueError, match="destination DCP size must divide destination TP size"
    ):
        tc._raiden_dst_dcp_size(bad_dcp_cfg)

    # Lines 634, 640: _reshard_advertise_host
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_ADVERTISE_HOST", " ")
    with pytest.raises(ValueError, match="TPU_RAIDEN_ADVERTISE_HOST is required"):
        tc._reshard_advertise_host(0)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_ADVERTISE_HOST", "h0,h1")
    assert tc._reshard_advertise_host(1) == "h1"
    with pytest.raises(ValueError, match="has no host for data-parallel rank"):
        tc._reshard_advertise_host(2)

    # Lines 708-718: _prefix_aware_load_supported
    tc._prefix_aware_load_supported.cache_clear()
    fake_rc = SimpleNamespace(SUPPORTS_DST_SKIP_BYTES=True)
    fake_torch_mod = SimpleNamespace(reshard_client=fake_rc)
    monkeypatch.setitem(sys.modules, "tpu_sync", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch", fake_torch_mod)
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch.reshard_client", fake_rc)
    assert tc._prefix_aware_load_supported() is True

    tc._prefix_aware_load_supported.cache_clear()
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch", None)
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch.reshard_client", None)
    assert tc._prefix_aware_load_supported() is False
    tc._prefix_aware_load_supported.cache_clear()


def test_tpu_connector_facade_and_base_scheduler_worker(monkeypatch):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.core.kv_cache_utils import KVCacheConfig

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs

    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")
    vllm_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=False,
            kv_connector_extra_config={"use_raiden_connector": False},
        ),
        cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(architecture="Qwen", max_model_len=64),
        speculative_config=None,
    )
    kv_cfg = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[])

    # Line 813->exit: role neither SCHEDULER nor WORKER
    conn_none_role = tc.TPUConnector(vllm_cfg, SimpleNamespace(), kv_cfg)
    assert getattr(conn_none_role, "connector_worker", None) is None

    # Scheduler role methods
    conn_sched = tc.TPUConnector(vllm_cfg, KVConnectorRole.SCHEDULER, kv_cfg)
    req = SimpleNamespace(
        request_id="req1",
        kv_transfer_params=None,
        num_prompt_tokens=16,
        prompt_token_ids=[1] * 16,
        _all_token_ids=[1] * 16,
    )
    conn_sched.on_new_request(req)
    assert conn_sched.get_kv_connector_stats() is None
    assert tc.TPUConnector.build_kv_connector_stats({"k": [1]}) is not None
    assert tc.TPUConnector.build_kv_connector_stats(None) is not None
    monkeypatch.setattr(tc, "TpuKVConnectorPromMetrics", lambda *args: ("prom", args))
    assert tc.TPUConnector.build_prom_metrics(vllm_cfg, {}, [], {})[0] == "prom"
    conn_sched.register_kv_caches({"l0": "v"})
    conn_sched.wait_for_layer_load("l0")
    conn_sched.save_kv_layer()
    conn_sched.wait_for_save()
    conn_sched.connector_worker = SimpleNamespace()
    assert conn_sched.get_block_ids_with_load_errors() == set()
    conn_sched.connector_worker = SimpleNamespace(
        get_block_ids_with_load_errors=lambda: {3}
    )
    assert conn_sched.get_block_ids_with_load_errors() == {3}
    assert conn_sched.get_block_ids_with_load_errors_group_index() is None
    conn_sched.update_connector_output(SimpleNamespace(finished_sending=["r1"]))
    called_out = []
    conn_sched.connector_scheduler.update_connector_output = called_out.append
    conn_sched.update_connector_output("out1")
    assert called_out == ["out1"]

    # Lines 863, 890: request_finished_all_groups missing group errors
    conn_sched.use_raiden = True
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    conn_sched._stage3_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=2, page_tokens=16),
    )
    with pytest.raises(ValueError, match="Stage-3 FA KV cache group is absent"):
        conn_sched.request_finished_all_groups(req, ([1],))
    conn_sched._stage3_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=0, page_tokens=16),
    )
    conn_sched._stage3_mamba_group_indices = [2]
    with pytest.raises(ValueError, match="mamba KV cache group is absent"):
        conn_sched.request_finished_all_groups(req, ([1],))
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")

    # Lines 1080, 1085, 1090: _maybe_truncate_prompt_for_handoff
    sched = conn_sched.connector_scheduler
    r_short = SimpleNamespace(num_prompt_tokens=1)
    sched._maybe_truncate_prompt_for_handoff(r_short)
    r_no_ids = SimpleNamespace(
        num_prompt_tokens=4, kv_transfer_params=None, prompt_token_ids=None
    )
    sched._maybe_truncate_prompt_for_handoff(r_no_ids)
    r_trunc = SimpleNamespace(
        num_prompt_tokens=4,
        kv_transfer_params=None,
        prompt_token_ids=[1, 2, 3, 4],
        _all_token_ids=[1, 2, 3, 4],
    )
    sched._maybe_truncate_prompt_for_handoff(r_trunc)
    assert r_trunc.kv_transfer_params == {"_p_side_truncated": True}

    # Lines 1149, 1159, 1217: _validate_load_params & get_num_new_matched_tokens
    r_valid = SimpleNamespace(
        request_id="rv",
        prompt_token_ids=[1] * 16,
        kv_transfer_params={
            "uuid": 10,
            "remote_block_ids": [1],
            "remote_host": ["127.0.0.1"],
            "remote_port": [5000],
            "remote_side_channel_port": "6000",
        },
    )
    assert sched.get_num_new_matched_tokens(r_valid, 16) == (0, False)
    assert r_valid.kv_transfer_params["remote_side_channel_port"] == 6000

    # Lines 1325-1433: TPUConnectorWorker methods
    import threading

    w = object.__new__(tc.TPUConnectorWorker)
    w._stop_event = threading.Event()
    w._coord_teardown = lambda: None
    w.device = torch.device("cpu")
    w.local_tp_rank = 0
    w.tp_rank = 0
    w.node_id = 0
    w.is_producer = False
    w._kv_scatter_enabled = False
    w.shape = (4, 8)
    w.dtype = torch.float32
    w.runner = SimpleNamespace(kv_caches=[torch.zeros((4, 8), dtype=torch.float32)])
    w._coord_pool = SimpleNamespace(
        layer_view=lambda slot, rank, layer, nblocks: torch.zeros(
            (nblocks, 8), dtype=torch.float32
        )
    )
    monkeypatch.setattr(
        tc.ZmqShmKvConnectorBase, "process_send_load", lambda self, meta: None
    )
    w.process_send_load(tc.TPUConnectorMetadata())

    monkeypatch.setattr(tc, "batch_transfer_d2h", lambda tpu, cpu: "d2h_fut")
    monkeypatch.setattr(tc, "batch_transfer_d2h_sync", lambda tpu, cpu: None)
    monkeypatch.setattr(tc, "batch_transfer_h2d_sync", lambda src, dst: None)
    monkeypatch.setattr(tc, "batch_transfer_h2d", lambda src, dst: "h2d_fut")
    monkeypatch.setattr(tc, "synchronize_tensors", lambda t, wait=False: None)

    fut, tpu_t, cpu_t, nbytes = w._stage_d2h(0, 2, [0, 1])
    assert fut == "d2h_fut" and nbytes == 2 * 8 * 4
    w._stage_d2h_sync(0, 2, [0, 1])
    waited = []
    w._wait_stage(SimpleNamespace(wait=lambda: waited.append(True)))
    assert waited == [True]
    dev_shards = w._h2d_into_device([torch.zeros((2, 8))])
    assert len(dev_shards) == 1
    fut_h2d, dev_shards_async = w._h2d_into_device_async([torch.zeros((2, 8))])
    assert fut_h2d == "h2d_fut" and len(dev_shards_async) == 1
    w._synchronize_device(torch.zeros(1))

    assert w._try_fast_scatter(dev_shards, w.runner.kv_caches, [0, 1]) is None
    w._kv_scatter_enabled = True
    monkeypatch.setattr(tc.kv_scatter, "prepare_scatter_args", lambda b, d: "prep")
    monkeypatch.setattr(
        tc.kv_scatter,
        "multi_layer_scatter_into",
        lambda shards, caches, b, prebuilt_args=None: ["scattered"],
    )
    assert w._try_fast_scatter(dev_shards, w.runner.kv_caches, [0, 1]) == ["scattered"]

    w.is_producer = True
    w._maybe_enable_kv_scatter()
    w.is_producer = False
    monkeypatch.setattr(tc.kv_scatter, "scatter_available", lambda: False)
    w._maybe_enable_kv_scatter()
    monkeypatch.setattr(tc.kv_scatter, "scatter_available", lambda: True)
    monkeypatch.setattr(
        tc.kv_scatter, "smoke_test_multi_layer_scatter", lambda **kw: True
    )
    w._kv_scatter_enabled = False
    w._maybe_enable_kv_scatter()
    assert w._kv_scatter_enabled is True
    w.shape = (4,)
    monkeypatch.setattr(
        tc.kv_scatter, "smoke_test_multi_layer_scatter", lambda **kw: False
    )
    w._maybe_enable_kv_scatter()


def test_tpu_raiden_scheduler_missing_branches(monkeypatch):
    from vllm.v1.request import RequestStatus

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs

    vllm_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=False,
            kv_connector_extra_config={},
            engine_id=None,
        ),
        cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=16,
        ),
        model_config=SimpleNamespace(architecture="Qwen", max_model_len=64),
    )
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")
    sched = tc.TPURaidenConnectorScheduler(vllm_cfg)

    # Line 1594->1602: _reject_kv_transfer_params when not stage3
    req_rej = SimpleNamespace(request_id="r_rej", kv_transfer_params={"bad": 1})
    sched._reject_kv_transfer_params(req_rej, req_rej.kv_transfer_params, "bad")

    # Lines 1643-1644: get_num_new_matched_tokens when not stage3
    monkeypatch.setattr(tc.dist_utils, "get_raiden_inline_load", lambda: False)
    req_leg = SimpleNamespace(
        request_id="r_leg",
        prompt_token_ids=[1] * 16,
        kv_transfer_params={
            "uuid": 1,
            "remote_block_ids": [1],
            "remote_host": "127.0.0.1",
            "remote_port": 5000,
        },
    )
    assert sched.get_num_new_matched_tokens(req_leg, 0) == (16, True)
    assert sched.get_num_new_matched_tokens(req_leg, 16) == (0, False)

    # Line 2022: request_finished when not stage3
    assert sched.request_finished(req_leg, [1]) == (False, None)

    # Switch to stage3
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tc.dist_utils, "get_raiden_inline_load", lambda: True)
    req_s3 = SimpleNamespace(
        request_id="r_s3",
        num_prompt_tokens=16,
        num_computed_tokens=0,
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={
            "req_id": "src_s3",
            "uuid": 10,
            "num_tokens": 16,
            "src_controller_address": "127.0.0.1:9000",
            "src_job_name": "prefill",
            "src_engine_id": "eng0",
            "src_data_replica_idx": 0,
            "src_parallelism": 1,
            "dst_tp_size": 1,
            "dst_dcp_size": 1,
        },
    )
    # Line 1624: stage3 inline load
    assert sched.get_num_new_matched_tokens(req_s3, 0) == (16, False)

    # Line 1747->1749: _update_stage3_state_after_alloc when kv_transfer_params becomes
    # None
    sched._stage3_prefix_aware_load_enabled = True
    orig_enq = sched._enqueue_stage3_release
    sched._enqueue_stage3_release = lambda r, **kw: setattr(
        r, "kv_transfer_params", None
    )
    sched._update_stage3_state_after_alloc(
        SimpleNamespace(
            request_id="r_none", num_computed_tokens=0, kv_transfer_params={"a": 1}
        ),
        SimpleNamespace(get_block_ids=lambda: ([],)),
        0,
    )
    sched._enqueue_stage3_release = orig_enq

    # Line 1773: absent FA group in _update_stage3_state_after_alloc
    sched._stage3_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=1, page_tokens=16),
    )
    with pytest.raises(ValueError, match="Stage-3 FA KV cache group is absent"):
        sched._update_stage3_state_after_alloc(
            req_s3, SimpleNamespace(get_block_ids=lambda: ([1],)), 16
        )

    # Lines 1804-1811: multi_group in _update_stage3_state_after_alloc
    sched._stage3_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=0, page_tokens=16),
        tc._Stage3TransferGroup(cache_group_index=1, page_tokens=16),
    )
    # Incomplete second group
    sched._update_stage3_state_after_alloc(
        req_s3, SimpleNamespace(get_block_ids=lambda: ([1], [])), 16
    )
    assert sched.reqs_to_load["r_s3"].fail_only is True
    # Valid multi_group + external_tokens > num_tokens (1814-1820)
    sched._update_stage3_state_after_alloc(
        req_s3, SimpleNamespace(get_block_ids=lambda: ([1], [2])), 32
    )
    assert sched.reqs_to_load["r_s3"].fail_only is True

    # Lines 1841-1849: unaligned skip_tokens
    sched._stage3_transfer_groups = ()
    sched._stage3_prefix_aware_load_enabled = True
    sched._update_stage3_state_after_alloc(
        req_s3, SimpleNamespace(get_block_ids=lambda: ([1],)), 8
    )
    assert sched.reqs_to_load["r_s3"].fail_only is True

    # Lines 1856-1863: absent mamba group in _update_stage3_state_after_alloc
    sched._stage3_prefix_aware_load_enabled = False
    sched._stage3_mamba_group_indices = [1]
    sched._update_stage3_state_after_alloc(
        req_s3, SimpleNamespace(get_block_ids=lambda: ([1],)), 16
    )
    assert sched.reqs_to_load["r_s3"].fail_only is True
    sched._stage3_mamba_group_indices = []

    # Lines 1900-1908: _raiden_engine_id
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_ENGINE_ID", "")
    assert sched._raiden_engine_id() == "decode-engine"
    vllm_cfg.kv_transfer_config.engine_id = "my_eng"
    assert sched._raiden_engine_id() == "my_eng"

    # Line 1977: _enqueue_stage3_release with non-dict params
    sched._enqueue_stage3_release(
        SimpleNamespace(request_id="r_nodict", kv_transfer_params=None)
    )

    # Line 2027->2032: consumer request_finished when active source req_id belongs to
    # another req
    sched._stage3_active_source_req_ids["src_s3"] = "other_req"
    req_s3.kv_transfer_params["_remote_kv_processed"] = True
    sched.request_finished(req_s3, [1])

    # Producer request_finished branches (2048, 2066, 2142-2261, 2322)
    sched.is_producer = True
    req_prod = SimpleNamespace(
        request_id="rp1",
        status=RequestStatus.RUNNING,
        num_computed_tokens=16,
        num_prompt_tokens=16,
        kv_transfer_params={"_p_side_truncated": True},
    )
    assert sched.request_finished(req_prod, [1]) == (False, None)
    req_prod.status = RequestStatus.FINISHED_LENGTH_CAPPED

    # Lines 2142-2180: group_block_ids validation and recycled sliding window pages
    sched._stage3_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=0, page_tokens=16, window_tokens=16),
        tc._Stage3TransferGroup(
            cache_group_index=1, page_tokens=16, window_tokens=None
        ),
    )
    with pytest.raises(ValueError, match="do not match the transfer plan"):
        sched.request_finished(req_prod, [1], group_block_ids=[[1]])
    # Second group too few blocks -> returns False, {}
    assert sched.request_finished(req_prod, [1], group_block_ids=[[1], []]) == (
        False,
        {},
    )
    # Recycled pages in group 0 (num_tokens=32, window_tokens=16 -> 1 skipped, plus 1
    # dead page)
    req_prod_win = SimpleNamespace(
        request_id="rp_win",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        num_computed_tokens=32,
        num_prompt_tokens=32,
        kv_transfer_params={"_p_side_truncated": True},
    )
    assert sched.request_finished(
        req_prod_win, [0, 0], group_block_ids=[[0, 0], [1, 2]]
    ) == (False, {})

    # No recycled pages when num_tokens=16
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "")
    with pytest.raises(ValueError, match="requires TPU_RAIDEN_CONTROLLER_ADDRESS"):
        sched.request_finished(req_prod, [1], group_block_ids=[[1], [2]])

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "127.0.0.1:9000")
    monkeypatch.setattr(sched, "_raiden_engine_id", lambda: "")
    with pytest.raises(ValueError, match="TPU_RAIDEN_ENGINE_ID must not be empty"):
        sched.request_finished(req_prod, [1])
    monkeypatch.undo()
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "127.0.0.1:9000")

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 0)
    with pytest.raises(
        ValueError, match="TPU_RAIDEN_TRANSFER_PARALLELISM must be positive"
    ):
        sched.request_finished(req_prod, [1])

    # DSv4 admission parallelism check (2203-2204, 2203->2229)
    assert (
        sched._validate_load_params(
            req_s3, {**req_s3.kv_transfer_params, "dst_tp_size": 2}
        )
        is not None
    )
    monkeypatch.setattr(tpu_envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)
    sched.request_finished(req_prod, [1])
    sched._stage3_finished_sends.clear()
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 2)
    with pytest.raises(
        ValueError, match="DSv4 producer transfer parallelism must be 1"
    ):
        sched.request_finished(req_prod, [1])
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", False)

    # TP stage3 source check (2211, 2322)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    with pytest.raises(
        ValueError, match="TP producer transfer parallelism must equal TP size"
    ):
        sched.request_finished(req_prod, [1])
    assert sched.get_finished_count() == 1
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", False)

    # Pipeline producer check (2217-2219, 2218->2229)
    vllm_cfg.parallel_config.pipeline_parallel_size = 2
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 2)
    sched.request_finished(req_prod, [1])
    sched._stage3_finished_sends.clear()
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 4)
    with pytest.raises(
        ValueError, match="pipeline producer transfer parallelism must equal PP size"
    ):
        sched.request_finished(req_prod, [1])
    vllm_cfg.parallel_config.pipeline_parallel_size = 1

    # Default PCP * TP check (2225)
    with pytest.raises(
        ValueError, match="producer transfer parallelism must equal PCP \\* TP size"
    ):
        sched.request_finished(req_prod, [1])

    # Negative dp_rank (2230)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)
    vllm_cfg.parallel_config.data_parallel_rank = -1
    with pytest.raises(ValueError, match="data-parallel rank must be non-negative"):
        sched.request_finished(req_prod, [1])
    vllm_cfg.parallel_config.data_parallel_rank = 0

    # Conflicting duplicate existing (2235)
    sched._stage3_finished_sends["rp1"] = (99, (999,), 16, {"uuid": 99})
    ok, params = sched.request_finished(req_prod, [1])
    assert ok is True and params == {"uuid": 99}
    sched._stage3_finished_sends.clear()

    # Missing mamba_block_ids (2261)
    sched._stage3_mamba_group_indices = [0]
    with pytest.raises(ValueError, match="mamba block tables were not provided"):
        sched.request_finished(req_prod, [1])


# ---------------------------------------------------------------------------
# tpu_connector.py Part 2: TPURaidenConnectorWorker (lines 2330 - 6511)
# ---------------------------------------------------------------------------


def _make_raiden_worker(
    monkeypatch, *, is_producer=True, tp_rank=0, tp_size=1, arch="Qwen"
):
    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc

    monkeypatch.setattr(tc, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(tc, "get_tensor_model_parallel_world_size", lambda: tp_size)
    vllm_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=is_producer,
            kv_connector_extra_config={},
            engine_id=None,
        ),
        cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            data_parallel_size=1,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=16,
        ),
        model_config=SimpleNamespace(
            architecture=arch,
            max_model_len=64,
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(
                    linear_num_key_heads=4,
                    linear_num_value_heads=4,
                    linear_key_head_dim=16,
                    linear_value_head_dim=16,
                    linear_conv_kernel_dim=4,
                    num_key_value_heads=4,
                )
            ),
            hf_text_config=SimpleNamespace(linear_attn_config={"num_heads": 4}),
        ),
        speculative_config=None,
    )
    return tc.TPURaidenConnectorWorker(vllm_cfg)


def test_tpu_raiden_worker_admission_and_topology(monkeypatch):
    import sys

    from vllm import envs as vllm_envs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

    w = _make_raiden_worker(monkeypatch, is_producer=True)

    # Lines 2481->exit, 2482->2485, 2486-2490: _get_raiden_stats
    monkeypatch.setattr(tc.dist_utils, "get_raiden_telemetry_module", lambda: None)
    w._get_raiden_stats()
    monkeypatch.setattr(
        tc.dist_utils, "get_raiden_telemetry_module", lambda: SimpleNamespace()
    )
    w._get_raiden_stats()
    monkeypatch.setattr(
        tc.dist_utils,
        "get_raiden_telemetry_module",
        lambda: SimpleNamespace(get_and_reset_metric_samples=lambda: {"m1": [1.0]}),
    )
    w._get_raiden_stats()
    w._get_raiden_stats()
    assert w.transfer_stats.data["m1"] == [1.0, 1.0]
    monkeypatch.setattr(
        tc.dist_utils,
        "get_raiden_telemetry_module",
        lambda: (_ for _ in ()).throw(RuntimeError("telemetry err")),
    )
    w._get_raiden_stats()

    # Lines 2534, 2536, 2543, 2559-2560, 2562-2563: register_runner
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        kv_cache_raw_tensors=[],
        bm_pool_layout=None,
    )
    monkeypatch.setattr(tpu_envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_QWEN35_ADMISSION", True)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        w.register_runner(runner)

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_QWEN35_ADMISSION", False)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", False)
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    with pytest.raises(RuntimeError, match="requires explicit pool admission"):
        w.register_runner(runner)

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_KIMIK3_ADMISSION", True)
    with pytest.raises(ValueError, match="requires KimiK3ForConditionalGeneration"):
        w.register_runner(runner)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_KIMIK3_ADMISSION", False)

    glm_called = []
    monkeypatch.setattr(w, "_admit_raiden_glm_kv_cache", glm_called.append)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    w.register_runner(runner)
    assert len(glm_called) == 1
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", False)

    dsv4_called = []
    monkeypatch.setattr(w, "_admit_raiden_dsv4_kv_cache", dsv4_called.append)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    w.register_runner(runner)
    assert len(dsv4_called) == 1
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", False)
    monkeypatch.undo()

    # Lines 2573, 2596, 2610-2615, 2628, 2647-2651, 2710-2711:
    # _admit_raiden_hybrid_kv_cache
    w2 = _make_raiden_worker(monkeypatch, is_producer=True)
    w2._raiden_transfer_engine = "already"
    w2._admit_raiden_hybrid_kv_cache(runner)
    w2._raiden_transfer_engine = None
    w2.vllm_config.parallel_config.data_parallel_size = 4
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "127.0.0.1:9000")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 1)
    with pytest.raises(ValueError, match="requires register_kv_caches"):
        w2._admit_raiden_hybrid_kv_cache(runner)

    # Kimi hybrid admission + seq_on_lane hybrid admission
    fake_pool = rpm.PoolEntry(
        tag="fa",
        storage_index=0,
        base_offset_bytes=0,
        num_blocks=2,
        block_stride_bytes=128,
        dtype_tag="uint8",
        layer_name="layers.0.attn",
        regions=(
            rpm.RegionSpec(
                name="kv",
                offset_bytes=0,
                stride_bytes=128,
                unit_bytes=8,
                num_units=16,
            ),
        ),
    )
    fake_manifest = rpm.PoolManifest(
        binding=rpm.BINDING_PRIVATE_TYPED,
        storages=(SimpleNamespace(),),
        pools=(fake_pool,),
    )
    monkeypatch.setattr(rpm, "verify_storage_binding", lambda **kw: 1)
    monkeypatch.setattr(rpm, "materialize_storages", lambda m: None)
    monkeypatch.setattr(rpm, "build_kimi_k3_pool_manifest", lambda **kw: fake_manifest)

    import vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane as sol_mod

    monkeypatch.setattr(
        sol_mod, "build_qwen35_pool_manifest_sol", lambda **kw: fake_manifest
    )

    w_kimi = _make_raiden_worker(
        monkeypatch, is_producer=True, arch="KimiK3ForConditionalGeneration"
    )
    w_kimi.named_kv_caches = {"layers.0.attn": SimpleNamespace()}
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 1)
    monkeypatch.setattr(
        w_kimi,
        "_construct_raiden_transfer_engine",
        lambda storages, num_slots=1: SimpleNamespace(
            register_pools=lambda *a, **kw: {}
        ),
    )
    monkeypatch.setattr(
        w_kimi, "_register_raiden_stage3_work_unit", lambda **kw: {"ok": True}
    )
    mamba_spec = MambaSpec(shapes=((4, 8), (4, 8)), dtypes=(None, None), block_size=16)
    runner_mamba = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=mamba_spec, layer_names=["layers.0.mamba"]
                )
            ]
        ),
        kv_cache_raw_tensors=[],
        bm_pool_layout=None,
    )
    w_kimi._admit_raiden_hybrid_kv_cache(runner_mamba)
    assert w_kimi._stage3_state_group_count == 1
    assert w_kimi._stage3_row_geometry["fa"] == (128, 8)

    # Seq-on-lane hybrid admission (2647-2651)
    w_sol = _make_raiden_worker(monkeypatch, is_producer=True)
    w_sol.vllm_config.parallel_config.data_parallel_size = 4
    w_sol.named_kv_caches = {"layers.0.attn": SimpleNamespace()}
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    monkeypatch.setattr(
        w_sol,
        "_construct_raiden_transfer_engine",
        lambda storages, num_slots=1: SimpleNamespace(
            register_pools=lambda *a, **kw: {}
        ),
    )
    monkeypatch.setattr(
        w_sol, "_register_raiden_stage3_work_unit", lambda **kw: {"ok": True}
    )
    w_sol._admit_raiden_hybrid_kv_cache(runner)
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")

    # Lines 2771-2855: _admit_raiden_glm_kv_cache
    w_glm = _make_raiden_worker(monkeypatch, is_producer=True)
    w_glm._raiden_transfer_engine = "already"
    w_glm._admit_raiden_glm_kv_cache(runner)
    w_glm._raiden_transfer_engine = None
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")
    with pytest.raises(RuntimeError, match="requires TPU_KV_RESHARD_TRANSPORT=raiden"):
        w_glm._admit_raiden_glm_kv_cache(runner)
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "")
    with pytest.raises(ValueError, match="TPU_RAIDEN_CONTROLLER_ADDRESS is required"):
        w_glm._admit_raiden_glm_kv_cache(runner)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "127.0.0.1:9000")
    with pytest.raises(ValueError, match="requires register_kv_caches"):
        w_glm._admit_raiden_glm_kv_cache(runner)
    w_glm.named_kv_caches = {"layers.0.attn": SimpleNamespace()}
    monkeypatch.setattr(rpm, "build_glm_mla_pool_manifest", lambda **kw: fake_manifest)
    monkeypatch.setattr(
        w_glm,
        "_construct_raiden_transfer_engine",
        lambda storages, num_slots=1: SimpleNamespace(
            register_pools=lambda *a, **kw: {}
        ),
    )
    monkeypatch.setattr(
        w_glm, "_register_raiden_stage3_work_unit", lambda **kw: {"ok": True}
    )
    monkeypatch.setattr(
        w_glm, "_measure_glm_tag_geometry", lambda m: {"mla.nope": (128, 8)}
    )
    w_glm._admit_raiden_glm_kv_cache(runner)
    assert w_glm._raiden_admission_summary["topology"] == "tp1_prefill"

    # Lines 2901-2978, 3032, 3039-3049: _admit_raiden_dsv4_kv_cache
    w_dsv4 = _make_raiden_worker(monkeypatch, is_producer=True)
    w_dsv4._raiden_transfer_engine = "already"
    w_dsv4._admit_raiden_dsv4_kv_cache(runner)
    w_dsv4._raiden_transfer_engine = None
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")
    with pytest.raises(RuntimeError, match="requires TPU_KV_RESHARD_TRANSPORT=raiden"):
        w_dsv4._admit_raiden_dsv4_kv_cache(runner)
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "")
    with pytest.raises(ValueError, match="TPU_RAIDEN_CONTROLLER_ADDRESS is required"):
        w_dsv4._admit_raiden_dsv4_kv_cache(runner)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "127.0.0.1:9000")
    with pytest.raises(ValueError, match="requires register_kv_caches"):
        w_dsv4._admit_raiden_dsv4_kv_cache(runner)

    # _raiden_dsv4_admission_topology invalid tp_size
    w_dsv4.tp_size = 2
    with pytest.raises(ValueError, match="supports DP attention only"):
        w_dsv4._raiden_dsv4_admission_topology()
    w_dsv4.tp_size = 1

    # _measure_dsv4_pool_plan empty pools -> RuntimeError (line 3032)
    fa_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=sys.modules["torch"].uint8
    )
    dsv4_groups = [
        SimpleNamespace(kv_cache_spec=fa_spec, layer_names=["layers.0.attn"]),
        SimpleNamespace(kv_cache_spec=fa_spec, layer_names=["layers.1.attn"]),
    ]
    empty_manifest = rpm.PoolManifest(
        binding=rpm.BINDING_PRIVATE_TYPED, storages=(), pools=()
    )
    with pytest.raises(RuntimeError, match="produced no transfer pools"):
        w_dsv4._measure_dsv4_pool_plan(empty_manifest, dsv4_groups)

    dsv4_pool0 = dataclasses.replace(fake_pool, tag="dsv4.csa.nope.g0")
    dsv4_pool1 = dataclasses.replace(
        fake_pool, tag="dsv4.swa.g1", layer_name="layers.1.attn"
    )
    dsv4_manifest = rpm.PoolManifest(
        binding=rpm.BINDING_PRIVATE_TYPED,
        storages=(SimpleNamespace(),),
        pools=(dsv4_pool0, dsv4_pool1),
    )
    w_dsv4.named_kv_caches = {"layers.0.attn": SimpleNamespace()}
    monkeypatch.setattr(rpm, "build_dsv4_pool_manifest", lambda **kw: dsv4_manifest)
    monkeypatch.setattr(
        w_dsv4,
        "_construct_raiden_transfer_engine",
        lambda storages, num_slots=1: SimpleNamespace(
            register_pools=lambda *a, **kw: {}
        ),
    )
    monkeypatch.setattr(
        w_dsv4, "_register_raiden_stage3_work_unit", lambda **kw: {"ok": True}
    )
    runner_dsv4 = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=dsv4_groups),
        kv_cache_raw_tensors=[],
        bm_pool_layout=None,
    )
    w_dsv4._admit_raiden_dsv4_kv_cache(runner_dsv4)
    assert len(w_dsv4._stage3_dsv4_plan) == 2


def test_tpu_raiden_worker_layout_and_registration_helpers(monkeypatch):
    import sys

    from vllm import envs as vllm_envs

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as lf
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol_mod

    w = _make_raiden_worker(monkeypatch, is_producer=True)
    fake_m = SimpleNamespace(
        pool_dicts=lambda: [],
        pools=(),
        geometry_by_tag=lambda: {},
        pools_of_class=lambda c: (),
    )

    # Lines 3055-3136: _measure_raiden_dsv4_layout, _measure_raiden_glm_layout,
    # _measure_stage3_layout
    monkeypatch.setattr(
        lf, "measured_dsv4_layout_fingerprint", lambda m: ("fp_dsv4", {"k": 1})
    )
    monkeypatch.setattr(
        lf,
        "measured_glm_layout_fingerprint",
        lambda m, page_tokens, hash_indexer: ("fp_glm", {"k": 2}),
    )
    monkeypatch.setattr(
        lf,
        "measured_kimi_k3_layout_fingerprint",
        lambda m, **kw: ("fp_kimi", {"k": 3}),
    )
    monkeypatch.setattr(
        sol_mod, "measured_sol_layout_fingerprint", lambda m: ("fp_sol", {"k": 4})
    )
    monkeypatch.setattr(sol_mod, "sol_fa_page_tokens", lambda m: 16)

    monkeypatch.setattr(tpu_envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    assert w._measure_stage3_layout(fake_m) == ("fp_dsv4", {"k": 1}, 16)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", False)

    w._stage3_kimi = True
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 0)
    with pytest.raises(ValueError, match="complete KDA head shards"):
        w._measure_stage3_layout(fake_m)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 2)
    assert w._measure_stage3_layout(fake_m) == ("fp_kimi", {"k": 3}, 16)
    w._stage3_kimi = False

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    assert w._measure_stage3_layout(fake_m) == ("fp_glm", {"k": 2}, 16)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", False)

    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    assert w._measure_stage3_layout(fake_m) == ("fp_sol", {"k": 4}, 16)
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")

    # Lines 3154-3168, 3187: _validate_stage3_transfer_parallelism
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    w._validate_stage3_transfer_parallelism(1)
    with pytest.raises(
        ValueError, match="DSv4 producer transfer parallelism must be 1"
    ):
        w._validate_stage3_transfer_parallelism(2)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", False)

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    w._validate_stage3_transfer_parallelism(1)
    with pytest.raises(ValueError, match="TP producer transfer parallelism must equal"):
        w._validate_stage3_transfer_parallelism(2)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", False)

    with pytest.raises(
        ValueError, match="Producer transfer parallelism must equal the complete PCP"
    ):
        w._validate_stage3_transfer_parallelism(2)

    # Lines 3198-3200, 3209-3211, 4412-4413: _new_raiden_controller_facade,
    # _new_raiden_manager, _raiden_hbm_memory_type
    fake_rc = SimpleNamespace(ReshardClient=lambda addr: ("client", addr))
    fake_mgr = SimpleNamespace(KVCacheManager=lambda **kw: ("mgr", kw))
    fake_rpc = SimpleNamespace(
        RaidenId=lambda **kw: ("id", kw),
        RaidenMemoryType=SimpleNamespace(HBM="HBM_MEM"),
    )
    monkeypatch.setitem(sys.modules, "tpu_sync", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch.reshard_client", fake_rc)
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch.kv_cache_manager", fake_mgr)
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc.raiden_controller", fake_rpc)
    assert w._new_raiden_controller_facade("addr") == ("client", "addr")
    assert w._new_raiden_manager(a=1) == ("mgr", {"a": 1})
    assert w._raiden_hbm_memory_type() == "HBM_MEM"

    # Line 3216: _raiden_transfer_parallelism <= 0
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 0)
    with pytest.raises(
        ValueError, match="TPU_RAIDEN_TRANSFER_PARALLELISM must be positive"
    ):
        w._raiden_transfer_parallelism()
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)

    # Lines 3253-3261: _raiden_engine_id
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_ENGINE_ID", "")
    assert w._raiden_engine_id() == "prefill-engine"
    w.vllm_config.kv_transfer_config.engine_id = "eng_w"
    assert w._raiden_engine_id() == "eng_w"

    # Lines 3271, 3300-3305: _maybe_host_reshard_store
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_RESHARD_IMPL", "store")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_ADVERTISE_HOST", "127.0.0.1")
    monkeypatch.setattr(w, "_local_raiden_transfer_rank", lambda: 1)
    w._maybe_host_reshard_store()
    assert w._reshard_store is None
    monkeypatch.setattr(w, "_local_raiden_transfer_rank", lambda: 0)
    fake_store_mod = SimpleNamespace(
        SUPPORTS_REQUEST_REGISTRY_TTL=False,
        RaidenId=lambda **kw: kw,
        ReshardStore=lambda **kw: SimpleNamespace(request_registry_ttl_s=10.0),
    )
    monkeypatch.setitem(
        sys.modules,
        "tpu_sync.api.torch",
        SimpleNamespace(reshard_store=fake_store_mod),
    )
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch.reshard_store", fake_store_mod)
    monkeypatch.setattr(tc.dist_utils, "get_p2p_wait_pull_timeout", lambda: 100)
    w._maybe_host_reshard_store()
    assert w._reshard_store is not None
    w._reshard_store = None
    monkeypatch.setattr(tc.dist_utils, "get_p2p_wait_pull_timeout", lambda: 5)
    w._maybe_host_reshard_store()
    assert w._reshard_store is not None
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_RESHARD_IMPL", "controller")

    # Lines 3332, 3334, 3346, 3349, 3353: _raiden_interleave_tokens
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        w._raiden_interleave_tokens(0, 1)
    with pytest.raises(ValueError, match="transfer_parallelism must be positive"):
        w._raiden_interleave_tokens(16, 0)
    w._stage3_kimi = True
    assert w._raiden_interleave_tokens(16, 2) == 16
    w._stage3_kimi = False
    w.vllm_config.parallel_config.cp_kv_cache_interleave_size = 0
    with pytest.raises(
        ValueError, match="requires a positive cp_kv_cache_interleave_size"
    ):
        w._raiden_interleave_tokens(16, 2)
    w.vllm_config.parallel_config.cp_kv_cache_interleave_size = 7
    with pytest.raises(ValueError, match="divisible by cp_kv_cache_interleave_size"):
        w._raiden_interleave_tokens(16, 2)
    w.vllm_config.parallel_config.cp_kv_cache_interleave_size = 16

    # Lines 3392, 3396, 3411, 3418, 3439-3448: _register_raiden_stage3_work_unit
    with pytest.raises(RuntimeError, match="did not advertise a data endpoint"):
        w._register_raiden_stage3_work_unit(
            engine=SimpleNamespace(transfer_address="", listener_address="l"),
            manifest=fake_m,
            controller_address="127.0.0.1:9000",
        )
    with pytest.raises(RuntimeError, match="did not advertise a listener endpoint"):
        w._register_raiden_stage3_work_unit(
            engine=SimpleNamespace(transfer_address="d", listener_address=""),
            manifest=fake_m,
            controller_address="127.0.0.1:9000",
        )
    monkeypatch.setattr(w, "_measure_stage3_layout", lambda m: ("fp", {}, 16))
    monkeypatch.setattr(w, "_local_raiden_transfer_rank", lambda: 5)
    with pytest.raises(ValueError, match="outside the admitted parallelism"):
        w._register_raiden_stage3_work_unit(
            engine=SimpleNamespace(transfer_address="d", listener_address="l"),
            manifest=fake_m,
            controller_address="127.0.0.1:9000",
        )
    monkeypatch.setattr(w, "_local_raiden_transfer_rank", lambda: 0)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_RESHARD_IMPL", "store")
    monkeypatch.setattr(w, "_wait_for_address_ready", lambda addr, timeout_s=60.0: None)
    monkeypatch.setattr(tc.time, "sleep", lambda s: None)
    calls = []

    def _flaky_reg(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("not ready")

    monkeypatch.setattr(
        w,
        "_new_raiden_controller_facade",
        lambda addr: SimpleNamespace(register_work_unit=_flaky_reg),
    )
    w._register_raiden_stage3_work_unit(
        engine=SimpleNamespace(transfer_address="d", listener_address="l"),
        manifest=fake_m,
        controller_address="127.0.0.1:9000",
    )
    assert len(calls) == 2

    # Deadline exceeded in _register_raiden_stage3_work_unit (3441)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_RESHARD_IMPL", "controller")
    monkeypatch.setattr(
        w,
        "_new_raiden_controller_facade",
        lambda addr: SimpleNamespace(
            register_work_unit=lambda **kw: (_ for _ in ()).throw(RuntimeError("fail"))
        ),
    )
    with pytest.raises(RuntimeError, match="fail"):
        w._register_raiden_stage3_work_unit(
            engine=SimpleNamespace(transfer_address="d", listener_address="l"),
            manifest=fake_m,
            controller_address="127.0.0.1:9000",
        )

    # Lines 3513, 3533-3537, 3545, 3583, 3589-3600, 3623, 3632
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 0)
    with pytest.raises(ValueError, match="Kimi Stage-3 requires PCP1"):
        w._raiden_kimi_k3_admission_topology()
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 1)

    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    w.vllm_config.parallel_config.data_parallel_size = 1
    w.vllm_config.parallel_config.pipeline_parallel_size = 1
    w.vllm_config.parallel_config.tensor_parallel_size = 2
    w.vllm_config.parallel_config.decode_context_parallel_size = 0
    with pytest.raises(ValueError, match="DCP size must divide TP size"):
        w._raiden_qwen35_admission_topology()
    w.vllm_config.parallel_config.decode_context_parallel_size = 1
    w.is_producer = True
    assert w._raiden_qwen35_admission_topology() == "pcp1tp2_prefill"
    w.is_producer = False
    w.vllm_config.parallel_config.decode_context_parallel_size = 2
    assert w._raiden_qwen35_admission_topology() == "tp2dcp2_decode"
    w.is_producer = True
    w.vllm_config.parallel_config.tensor_parallel_size = 1
    w.vllm_config.parallel_config.decode_context_parallel_size = 1
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")

    w.vllm_config.parallel_config.data_parallel_size = 3
    with pytest.raises(
        ValueError, match="Raiden Qwen3.5 admission topology pcp8_prefill"
    ):
        w._raiden_qwen35_admission_topology()

    w.vllm_config.model_config.hf_config = None
    assert w._model_config_int("missing_attr", 7) == 7

    assert w._local_head_count(0) == 0
    w.vllm_config.parallel_config.tensor_parallel_size = 4
    assert w._local_head_count(2) == 1
    with pytest.raises(ValueError, match="must be divisible by tp_size"):
        w._local_head_count(6)
    assert w._local_head_count(8) == 2

    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 1)
    with pytest.raises(ValueError, match="cannot change Q/K replication"):
        w._validate_gdn_transfer_geometry(tc.derive_gdn_head_geometry(2, 4, 4))
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 2)
    with pytest.raises(
        ValueError, match="GDN destination shards must divide producer parallelism"
    ):
        w._validate_gdn_transfer_geometry(tc.derive_gdn_head_geometry(6, 6, 3))
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 1)


def test_tpu_raiden_worker_producer_registration_paths(monkeypatch):
    from vllm import envs as vllm_envs

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
        LoadMeta,
        SendMeta,
        TPUConnectorMetadata,
    )
    from vllm_torchtpu.distributed.kv_transfer.raiden import byte_spans as bs
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol_mod

    w = _make_raiden_worker(monkeypatch, is_producer=False)
    fake_eng = SimpleNamespace(
        start_read=lambda *a, **kw: None,
        register_read=lambda *a, **kw: None,
        poll_stats=lambda: ([], [], []),
    )
    w._raiden_transfer_engine = fake_eng

    # Lines 3687-3689: consumer stage3 process_send_load with wait_for_completion=True,
    # report_completion=False
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(w, "_submit_stage3_loads", lambda m, e, synchronous=False: None)
    monkeypatch.setattr(w, "_wait_for_recving", lambda reqs: None)
    meta_load = TPUConnectorMetadata()
    meta_load.reqs_to_load = {"r1": SimpleNamespace()}
    w.process_send_load(meta_load, wait_for_completion=True, report_completion=False)
    assert "r1" in w._suppress_done_recving
    w.process_send_load(meta_load, wait_for_completion=True, report_completion=True)

    # Line 3722: legacy process_send_load mismatched remote/local block_ids
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")
    meta_bad = TPUConnectorMetadata()
    meta_bad.reqs_to_load = {
        "r_bad": LoadMeta(
            uuid=1,
            local_block_ids=[1],
            remote_block_ids=None,
            remote_host="127.0.0.1",
            remote_port=5000,
        )
    }
    with pytest.raises(
        ValueError, match="must contain both remote and local block ids"
    ):
        w.process_send_load(meta_bad)

    # Line 3755: _require_stage3_controller
    with pytest.raises(
        RuntimeError, match="requires successful work-unit registration"
    ):
        w._require_stage3_controller()

    # Lines 3771-3777: _maybe_delay_stage3_registration_for_test
    monkeypatch.setattr(
        tc.dist_utils, "get_stage3_test_registration_delay_s", lambda: 0.01
    )
    monkeypatch.setattr(tc.time, "sleep", lambda s: None)
    meta_send = TPUConnectorMetadata()
    meta_send.reqs_to_send = {
        "rs1": SendMeta(
            uuid=1, local_block_ids=[1], expiration_time=100.0, num_tokens=16
        )
    }
    w._maybe_delay_stage3_registration_for_test(meta_send)
    monkeypatch.setattr(
        tc.dist_utils, "get_stage3_test_registration_delay_s", lambda: 0.0
    )

    # Lines 3783, 3787, 3806-3998: _register_stage3_request_blocks
    wp = _make_raiden_worker(monkeypatch, is_producer=True)
    wp._raiden_controller_facade = SimpleNamespace(
        register_request_blocks=lambda **kw: None
    )
    wp._raiden_controller_address = "127.0.0.1:9000"
    with pytest.raises(RuntimeError, match="producer work unit is not registered"):
        wp._register_stage3_request_blocks(meta_send)
    wp._raiden_work_unit = "unit0"
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)
    monkeypatch.setattr(wp, "_local_raiden_transfer_rank", lambda: 2)
    with pytest.raises(ValueError, match="transfer rank is outside parallelism"):
        wp._register_stage3_request_blocks(meta_send)
    monkeypatch.setattr(wp, "_local_raiden_transfer_rank", lambda: 0)

    # Expired terminal cleanup (3806-3808)
    wp._stage3_terminal_sends["expired_r"] = tc._Stage3RegisteredSend(
        uuid=1, local_block_ids=(1,), num_tokens=16, expiration_time=0.0
    )
    # Invalid num_tokens (3815), expiration_time (3817), scheduler_ids (3832)
    m_bad_tok = TPUConnectorMetadata()
    m_bad_tok.reqs_to_send = {
        "r": SendMeta(uuid=1, local_block_ids=[1], expiration_time=100.0, num_tokens=0)
    }
    with pytest.raises(ValueError, match="requires num_tokens"):
        wp._register_stage3_request_blocks(m_bad_tok)
    assert "expired_r" not in wp._stage3_terminal_sends

    m_bad_exp = TPUConnectorMetadata()
    m_bad_exp.reqs_to_send = {
        "r": SendMeta(uuid=1, local_block_ids=[1], expiration_time=0.0, num_tokens=16)
    }
    with pytest.raises(ValueError, match="requires a positive expiration_time"):
        wp._register_stage3_request_blocks(m_bad_exp)

    m_bad_blk = TPUConnectorMetadata()
    m_bad_blk.reqs_to_send = {
        "r": SendMeta(
            uuid=1, local_block_ids=[1, 2], expiration_time=100.0, num_tokens=16
        )
    }
    with pytest.raises(
        ValueError, match="scheduler block count is inconsistent with PCP geometry"
    ):
        wp._register_stage3_request_blocks(m_bad_blk)

    # Line 3887: dst_shards > 1 without seq_on_lane
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 2)
    with pytest.raises(ValueError, match="requires the seq-along-lane FA layout"):
        wp._register_stage3_request_blocks(meta_send)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DST_SHARDS", 1)

    # Line 3998: empty spans + terminal/existing conflict checks (3916, 3928) + HND
    # (3866-3871)
    wp._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {"fa": {"live_bytes_per_block": 128}},
        pools_of_class=lambda c: [SimpleNamespace(tag="fa")],
    )
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    monkeypatch.setattr(
        sol_mod, "sol_fa_geometry_from_manifest", lambda m: SimpleNamespace()
    )
    monkeypatch.setattr(
        sol_mod,
        "lower_fa_spans_sol",
        lambda **kw: SimpleNamespace(spans=(), tag="fa", declared_bytes=0),
    )
    wp._register_stage3_request_blocks(meta_send)
    wp._stage3_registered_sends.clear()
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")
    monkeypatch.setattr(
        bs,
        "lower_fa_spans",
        lambda **kw: SimpleNamespace(spans=(), tag="fa", declared_bytes=0),
    )
    wp._register_stage3_request_blocks(meta_send)
    assert wp._stage3_send_outcomes["rs1"] == "done"

    # Conflicting existing registration (3928)
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send = {
        "rs1": SendMeta(
            uuid=999, local_block_ids=[1], expiration_time=100.0, num_tokens=16
        )
    }
    with pytest.raises(
        ValueError, match="Conflicting duplicate Stage-3 producer registration"
    ):
        wp._register_stage3_request_blocks(meta_conflict)

    # Conflicting terminal registration (3916)
    wp._stage3_registered_sends.clear()
    wp._stage3_terminal_sends["rs1"] = tc._Stage3RegisteredSend(
        uuid=888,
        local_block_ids=(1,),
        num_tokens=16,
        expiration_time=tc.time.perf_counter() + 1000.0,
    )
    with pytest.raises(
        ValueError, match="Conflicting replay after terminal Stage-3 producer"
    ):
        wp._register_stage3_request_blocks(meta_send)

    # Line 4045: _measure_glm_tag_geometry missing tag
    with pytest.raises(RuntimeError, match="Stage-3 GLM manifest is missing tag"):
        wp._measure_glm_tag_geometry(SimpleNamespace(geometry_by_tag=lambda: {}))

    # Lines 4070-4222: _register_stage3_request_row_spans
    wp._raiden_work_unit = None
    with pytest.raises(RuntimeError, match="producer work unit is not registered"):
        wp._register_stage3_request_row_spans(meta_send)
    wp._raiden_work_unit = "unit0"
    wp._stage3_row_geometry = {"mla.nope": (128, 8)}
    wp._stage3_terminal_sends["exp_row"] = tc._Stage3RegisteredSend(
        uuid=1, local_block_ids=(1,), num_tokens=16, expiration_time=0.0
    )
    with pytest.raises(ValueError, match="requires num_tokens"):
        wp._register_stage3_request_row_spans(m_bad_tok)
    assert "exp_row" not in wp._stage3_terminal_sends
    with pytest.raises(ValueError, match="requires a positive expiration_time"):
        wp._register_stage3_request_row_spans(m_bad_exp)

    # Kimi + empty spans (4120-4137, 4208)
    wp._stage3_kimi = True
    wp._stage3_row_geometry = {"fa": (128, 8)}
    wp._stage3_terminal_sends.clear()
    monkeypatch.setattr(
        bs,
        "lower_glm_row_spans",
        lambda **kw: SimpleNamespace(spans=(), tag="fa", declared_bytes=0),
    )
    wp._register_stage3_request_row_spans(meta_send)
    assert wp._stage3_send_outcomes["rs1"] == "done"
    # Duplicate existing (4154-4163) & conflicting existing (4159)
    wp._register_stage3_request_row_spans(meta_send)
    with pytest.raises(
        ValueError, match="Conflicting duplicate Stage-3 producer registration"
    ):
        wp._register_stage3_request_row_spans(meta_conflict)
    wp._stage3_registered_sends.clear()
    # Terminal replay matching & conflicting (4142-4151)
    wp._stage3_terminal_sends["rs1"] = tc._Stage3RegisteredSend(
        uuid=1,
        local_block_ids=(),
        num_tokens=16,
        expiration_time=tc.time.perf_counter() + 1000.0,
    )
    wp._register_stage3_request_row_spans(meta_send)
    with pytest.raises(
        ValueError, match="Conflicting replay after terminal Stage-3 producer"
    ):
        wp._register_stage3_request_row_spans(meta_conflict)
    wp._stage3_terminal_sends.clear()

    # Cancelled error & non-cancelled error in _register_stage3_request_row_spans
    # (4179-4199)
    wp._raiden_controller_facade = SimpleNamespace(
        register_request_blocks=lambda **kw: (_ for _ in ()).throw(
            RuntimeError("Request block registration was cancelled")
        )
    )
    wp._register_stage3_request_row_spans(meta_send)
    assert "rs1" in wp._done_sending
    wp._stage3_terminal_sends.clear()
    wp._raiden_controller_facade = SimpleNamespace(
        register_request_blocks=lambda **kw: (_ for _ in ()).throw(
            RuntimeError("other error")
        )
    )
    with pytest.raises(RuntimeError, match="other error"):
        wp._register_stage3_request_row_spans(meta_send)
    wp._stage3_kimi = False

    # Lines 4238, 4243, 4281-4408: _stage3_dsv4_group_blocks &
    # _register_stage3_request_blocks_dsv4
    wp._stage3_dsv4_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=0, page_tokens=16),
    )
    with pytest.raises(
        RuntimeError, match="require one block table per KV cache group"
    ):
        wp._stage3_dsv4_group_blocks(SimpleNamespace(group_block_ids=None), 16)
    with pytest.raises(RuntimeError, match="do not match the admitted transfer plan"):
        wp._stage3_dsv4_group_blocks(SimpleNamespace(group_block_ids=[[1], [2]]), 16)

    wp._raiden_work_unit = None
    with pytest.raises(RuntimeError, match="producer work unit is not registered"):
        wp._register_stage3_request_blocks_dsv4(meta_send)
    wp._raiden_work_unit = "unit0"
    wp._stage3_dsv4_plan = ()
    with pytest.raises(RuntimeError, match="DSv4 pool plan is missing"):
        wp._register_stage3_request_blocks_dsv4(meta_send)
    wp._stage3_dsv4_plan = (
        tc._Stage3Dsv4Pool(
            tag="dsv4.csa.nope.g0",
            plan_position=0,
            page_tokens=16,
            live_bytes_per_block=128,
            row_bytes=8,
        ),
    )
    wp._stage3_terminal_sends["exp_dsv4"] = tc._Stage3RegisteredSend(
        uuid=1, local_block_ids=(1,), num_tokens=16, expiration_time=0.0
    )
    with pytest.raises(ValueError, match="requires num_tokens"):
        wp._register_stage3_request_blocks_dsv4(m_bad_tok)
    assert "exp_dsv4" not in wp._stage3_terminal_sends
    with pytest.raises(ValueError, match="requires a positive expiration_time"):
        wp._register_stage3_request_blocks_dsv4(m_bad_exp)

    meta_dsv4 = TPUConnectorMetadata()
    meta_dsv4.reqs_to_send = {
        "rd1": SendMeta(
            uuid=1,
            local_block_ids=[1],
            expiration_time=100.0,
            num_tokens=16,
            group_block_ids=[[1]],
        )
    }
    meta_dsv4_conflict = TPUConnectorMetadata()
    meta_dsv4_conflict.reqs_to_send = {
        "rd1": SendMeta(
            uuid=99,
            local_block_ids=[1],
            expiration_time=100.0,
            num_tokens=16,
            group_block_ids=[[1]],
        )
    }
    monkeypatch.setattr(
        bs,
        "lower_dsv4_row_spans",
        lambda **kw: SimpleNamespace(
            spans=(), tag="dsv4.csa.nope.g0", declared_bytes=0
        ),
    )
    wp._raiden_controller_facade = SimpleNamespace(
        register_request_blocks=lambda **kw: None
    )
    wp._register_stage3_request_blocks_dsv4(meta_dsv4)
    assert wp._stage3_send_outcomes["rd1"] == "done"
    # Duplicate existing & conflicting existing (4344-4353)
    wp._register_stage3_request_blocks_dsv4(meta_dsv4)
    with pytest.raises(
        ValueError, match="Conflicting duplicate Stage-3 producer registration"
    ):
        wp._register_stage3_request_blocks_dsv4(meta_dsv4_conflict)
    wp._stage3_registered_sends.clear()
    # Terminal replay matching & conflicting (4332-4341)
    wp._stage3_terminal_sends["rd1"] = tc._Stage3RegisteredSend(
        uuid=1,
        local_block_ids=(),
        num_tokens=16,
        expiration_time=tc.time.perf_counter() + 1000.0,
    )
    wp._register_stage3_request_blocks_dsv4(meta_dsv4)
    with pytest.raises(
        ValueError, match="Conflicting replay after terminal Stage-3 producer"
    ):
        wp._register_stage3_request_blocks_dsv4(meta_dsv4_conflict)
    wp._stage3_terminal_sends.clear()

    # Cancelled & other error in _register_stage3_request_blocks_dsv4 (4362-4382)
    wp._raiden_controller_facade = SimpleNamespace(
        register_request_blocks=lambda **kw: (_ for _ in ()).throw(
            RuntimeError("Request block registration was cancelled")
        )
    )
    wp._register_stage3_request_blocks_dsv4(meta_dsv4)
    wp._stage3_terminal_sends.clear()
    wp._raiden_controller_facade = SimpleNamespace(
        register_request_blocks=lambda **kw: (_ for _ in ()).throw(
            RuntimeError("dsv4 err")
        )
    )
    with pytest.raises(RuntimeError, match="dsv4 err"):
        wp._register_stage3_request_blocks_dsv4(meta_dsv4)


def test_tpu_raiden_worker_state_spans_and_polling_paths(monkeypatch):
    from vllm import envs as vllm_envs

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol_mod

    w = _make_raiden_worker(monkeypatch, is_producer=True)

    # Line 4424: _stage3_source_work_units mismatch
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 2)
    with pytest.raises(
        ValueError, match="source parallelism does not match destination"
    ):
        w._stage3_source_work_units(SimpleNamespace(src_parallelism=1))
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)

    # Lines 4458-4472: _stage3_fa_token_bytes
    w._raiden_manifest = None
    with pytest.raises(RuntimeError, match="requires the admitted pool manifest"):
        w._stage3_fa_token_bytes(16)
    w._raiden_manifest = SimpleNamespace(geometry_by_tag=lambda: {})
    with pytest.raises(RuntimeError, match="contains no FA pools"):
        w._stage3_fa_token_bytes(16)
    w._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {"fa": {"live_bytes_per_block": 15}}
    )
    with pytest.raises(RuntimeError, match="not token-aligned"):
        w._stage3_fa_token_bytes(16)
    w._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {"fa": {"live_bytes_per_block": 128}}
    )
    assert w._stage3_fa_token_bytes(16) == 8

    # Lines 4478, 4483, 4488: _stage3_state_live_bytes
    w._raiden_manifest = None
    with pytest.raises(RuntimeError, match="requires the admitted pool manifest"):
        w._stage3_state_live_bytes("gdn.conv.g0")
    w._raiden_manifest = SimpleNamespace(geometry_by_tag=lambda: {})
    with pytest.raises(RuntimeError, match="is absent from the pool manifest"):
        w._stage3_state_live_bytes("gdn.conv.g0")
    w._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {"gdn.conv.g0": {"live_bytes_per_block": 0}}
    )
    with pytest.raises(RuntimeError, match="has invalid live bytes"):
        w._stage3_state_live_bytes("gdn.conv.g0")

    # Lines 4499, 4518: _stage3_state_regions
    w._raiden_manifest = SimpleNamespace(pools=())
    with pytest.raises(RuntimeError, match="is absent from the pool manifest"):
        w._stage3_state_regions("gdn.conv.g0")
    r1 = rpm.RegionSpec("r", 0, 16, 8, 2)
    r2 = rpm.RegionSpec("r", 0, 32, 8, 2)
    w._raiden_manifest = SimpleNamespace(
        pools=(
            SimpleNamespace(tag="gdn.conv.g0", regions=(r1,)),
            SimpleNamespace(tag="gdn.conv.g0", regions=(r2,)),
        )
    )
    with pytest.raises(RuntimeError, match="disagree on live regions"):
        w._stage3_state_regions("gdn.conv.g0")

    # Lines 4530, 4535, 4552, 4563: _stage3_fa_registration_tags &
    # _stage3_state_registration_tags
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_POOL_TAGS_PER_LAYER", True)
    w._raiden_manifest = None
    with pytest.raises(RuntimeError, match="requires the admitted pool manifest"):
        w._stage3_fa_registration_tags()
    with pytest.raises(RuntimeError, match="requires the admitted pool manifest"):
        w._stage3_state_registration_tags()
    w._raiden_manifest = SimpleNamespace(
        pools_of_class=lambda c: [],
        pools=[SimpleNamespace(tag="gdn.conv.gX.l0")],
    )
    with pytest.raises(RuntimeError, match="contains no FA pools"):
        w._stage3_fa_registration_tags()
    with pytest.raises(RuntimeError, match="carries no mamba group ordinal"):
        w._stage3_state_registration_tags()
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_POOL_TAGS_PER_LAYER", False)

    # Lines 4585, 4590, 4620, 4645, 4663, 4692: _stage3_state_pool_spans
    w._stage3_state_group_count = 1
    with pytest.raises(RuntimeError, match="requires the producer's per-group"):
        w._stage3_state_pool_spans(None, 0, 1)
    with pytest.raises(RuntimeError, match="group count disagrees with the manifest"):
        w._stage3_state_pool_spans([1, 2], 0, 1)

    w._raiden_manifest = SimpleNamespace(
        binding=rpm.BINDING_ALIASED_RAW,
        geometry_by_tag=lambda: {
            "gdn.conv.g0": {"live_bytes_per_block": 16},
            "gdn.ssm.g0": {"live_bytes_per_block": 16},
        },
        pools=(
            SimpleNamespace(
                tag="gdn.conv.g0",
                base_offset_bytes=512,
                block_stride_bytes=1024,
                regions=(r1,),
            ),
        ),
    )
    w._raiden_layout_fingerprint_payload = None
    with pytest.raises(
        RuntimeError, match="requires an admitted physical layout fingerprint"
    ):
        w._stage3_state_pool_spans([1], 0, 1)

    from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as lf

    w._raiden_layout_fingerprint_payload = {
        "minor_to_major": lf.expected_fa_minor_to_major(None),
        "tiles": lf.EXPECTED_FA_TILES,
        "element_size_in_bits": 8,
        "fa_kv_layout": "WRONG",
    }
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    with pytest.raises(RuntimeError, match="require an HND layout fingerprint"):
        w._stage3_state_pool_spans([1], 0, 1)
    w._raiden_layout_fingerprint_payload["fa_kv_layout"] = sol_mod.SOL_LAYOUT_TOKEN
    monkeypatch.setattr(
        sol_mod,
        "sol_fa_geometry_from_manifest",
        lambda m: SimpleNamespace(packing=2, page_tokens=512),
    )
    with pytest.raises(RuntimeError, match="must be physical-token aligned"):
        w._stage3_state_pool_spans([1], 0, 1)
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")

    # Line 4663: unaligned base_offset_bytes
    with pytest.raises(RuntimeError, match="must be physical-token aligned"):
        w._stage3_state_pool_spans([1], 0, 1)

    # Line 4692: declared_bytes != admitted_live_bytes
    w._raiden_manifest.pools = (
        SimpleNamespace(
            tag="gdn.conv.g0",
            base_offset_bytes=0,
            block_stride_bytes=1024,
            regions=(r1,),
        ),
    )
    monkeypatch.setattr(
        sol_mod,
        "lower_gdn_state_shard_spans_sol",
        lambda **kw: SimpleNamespace(declared_bytes=999),
    )
    with pytest.raises(RuntimeError, match="byte lowering disagrees with the admitted"):
        w._stage3_state_pool_spans([1], 0, 1)
    w._stage3_state_group_count = 0

    # Lines 4720-4724, 4720->4709: _stage3_record_producer_send_terminals unmapped &
    # already reported
    w._stage3_reported_sends.add("already_rep")
    w._stage3_record_producer_send_terminals(["unmapped_req", "already_rep"], [])

    # Lines 4780-4787, 4789: _probe_stage3_registrations error & length mismatch
    monkeypatch.setattr(tc.dist_utils, "get_stage3_status_probe_s", lambda: 0.1)
    w._stage3_registered_sends["rp"] = tc._Stage3RegisteredSend(
        uuid=1, local_block_ids=(1,), num_tokens=16, expiration_time=100.0
    )
    w._stage3_status_probe_next["rp"] = 0.0
    w._probe_stage3_registrations(
        SimpleNamespace(
            get_request_block_status=lambda keys: (_ for _ in ()).throw(
                RuntimeError("probe err")
            )
        ),
        10.0,
    )
    w._stage3_status_probe_next["rp"] = 0.0
    with pytest.raises(RuntimeError, match="status probe returned 0 statuses"):
        w._probe_stage3_registrations(
            SimpleNamespace(get_request_block_status=lambda keys: []), 10.0
        )

    # Line 4839->4846: _start_stage3_transfer_with_d5_retry multiple attempts
    monkeypatch.setattr(tc.dist_utils, "get_p2p_wait_pull_timeout", lambda: 5.0)
    monkeypatch.setattr(tc.time, "sleep", lambda s: None)
    att = []

    def _retry_start(**kw):
        att.append(1)
        if len(att) < 3:
            raise RuntimeError("Missing producer block registration")
        return True

    assert (
        w._start_stage3_transfer_with_d5_retry(
            SimpleNamespace(start_transfer=_retry_start)
        )
        is True
    )
    assert len(att) == 3

    # Line 4875: _stage3_source_request_id missing
    with pytest.raises(RuntimeError, match="has no source identity binding"):
        w._stage3_source_request_id("missing_dst")


def test_tpu_raiden_worker_consumer_submit_and_lifecycle_paths(monkeypatch):
    import socket

    from vllm import envs as vllm_envs

    import vllm_torchtpu.distributed.kv_transfer.tpu_connector as tc
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
        TPUConnectorMetadata,
    )
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol_mod

    wc = _make_raiden_worker(monkeypatch, is_producer=False)
    wc._raiden_controller_facade = SimpleNamespace()
    wc._raiden_controller_address = "127.0.0.1:9000"

    # Lines 4913->4915, 4916, 4921-4936, 4953-4954:
    # _stage3_release_producer_registration
    rel_meta = tc._Stage3LoadMeta(
        uuid=1,
        source_req_id="src1",
        local_block_ids=[],
        num_tokens=16,
        src_controller_address="127.0.0.1:9001",
        src_job_name="prefill",
        src_engine_id="eng0",
        src_data_replica_idx=0,
        src_parallelism=1,
        release_only=True,
        report_completion=False,
    )
    monkeypatch.setattr(
        wc,
        "_new_raiden_controller_facade",
        lambda addr: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )
    wc._stage3_release_producer_registration("dst1", rel_meta, synchronous=True)
    rel_meta_rep = dataclasses.replace(rel_meta, report_completion=True)
    wc._stage3_release_producer_registration("dst1b", rel_meta_rep, synchronous=True)
    assert "dst1b" in wc._done_recving

    monkeypatch.setattr(
        wc,
        "_new_raiden_controller_facade",
        lambda addr: SimpleNamespace(
            cancel_request_blocks_if_unclaimed=lambda **kw: (_ for _ in ()).throw(
                RuntimeError("cancel rpc err")
            )
        ),
    )
    wc._stage3_release_producer_registration("dst2", rel_meta, synchronous=True)

    # Lines 4974, 4981, 4999->5003, 5026, 5063-5094, 5161-5180: _submit_stage3_loads
    fake_eng = SimpleNamespace(poll_stats=lambda: ([], [], []))
    wc._raiden_work_unit = None
    with pytest.raises(RuntimeError, match="destination work unit is not registered"):
        wc._submit_stage3_loads(TPUConnectorMetadata(), fake_eng)
    wc._raiden_work_unit = "dst_unit0"

    m_bad_type = TPUConnectorMetadata()
    m_bad_type.reqs_to_load = {"d": SimpleNamespace()}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="requires controller load metadata"):
        wc._submit_stage3_loads(m_bad_type, fake_eng)

    # Conflicting duplicate load submission (5026)
    wc._stage3_submitted_loads["d_dup"] = 10
    wc._stage3_submitted_load_tokens["d_dup"] = 16
    m_dup = TPUConnectorMetadata()
    m_dup.reqs_to_load = {
        "d_dup": dataclasses.replace(rel_meta, release_only=False, uuid=99)
    }
    with pytest.raises(ValueError, match="Conflicting duplicate Stage-3 load"):
        wc._submit_stage3_loads(m_dup, fake_eng)

    # Sharded consumer non-Kimi (5037-5094) and Kimi TP group (4999->5003, 5161-5180)
    wc_tp = _make_raiden_worker(monkeypatch, is_producer=False, tp_rank=1, tp_size=2)
    wc_tp._raiden_controller_facade = SimpleNamespace()
    wc_tp._raiden_controller_address = "127.0.0.1:9000"
    wc_tp._raiden_work_unit = "dst_unit1"
    fake_tp_grp = SimpleNamespace(
        world_size=2,
        cpu_group="cpu_grp",
        broadcast_object=lambda obj, src=0: obj,
    )
    monkeypatch.setattr(wc_tp, "_stage3_tp_group", lambda: fake_tp_grp)
    # Fail_only on tp_rank=1 skips _stage3_release_producer_registration (4999->5003)
    m_fail_only = TPUConnectorMetadata()
    m_fail_only.reqs_to_load = {
        "d_fo": dataclasses.replace(
            rel_meta, release_only=False, fail_only=True, local_block_ids=[1]
        )
    }
    wc_tp._submit_stage3_loads(m_fail_only, fake_eng)

    # Leader outcomes: failed (5060-5072) and uncertain (5073-5094)
    wc_tp.tp_rank = 0
    monkeypatch.setattr(wc_tp, "_raiden_destination_work_units", lambda: ["u0", "u1"])
    monkeypatch.setattr(
        wc_tp,
        "_submit_stage3_load_as_leader",
        lambda **kw: ("failed", None, "pre-arm err", 0, 0),
    )
    m_l1 = TPUConnectorMetadata()
    m_l1.reqs_to_load = {
        "d_l1": dataclasses.replace(
            rel_meta, source_req_id="s_l1", release_only=False, local_block_ids=[1]
        )
    }
    wc_tp._submit_stage3_loads(m_l1, fake_eng)
    assert "d_l1" in wc_tp._failed_recving

    monkeypatch.setattr(
        wc_tp,
        "_submit_stage3_load_as_leader",
        lambda **kw: ("uncertain", None, "post-arm err", 0, 0),
    )
    m_l2 = TPUConnectorMetadata()
    m_l2.reqs_to_load = {
        "d_l2": dataclasses.replace(
            rel_meta, source_req_id="s_l2", release_only=False, local_block_ids=[2]
        )
    }
    wc_tp._submit_stage3_loads(m_l2, fake_eng)
    assert "d_l2" in wc_tp._stage3_pending_controller_failures

    # Kimi sharded consumer path (5161-5180)
    wc_tp._stage3_kimi = True
    monkeypatch.setattr(tc, "get_tp_group", lambda: fake_tp_grp)
    monkeypatch.setattr(
        tc.torch.distributed,
        "all_gather_object",
        lambda out, val, group=None: out.__setitem__(0, False),
    )
    monkeypatch.setattr(wc_tp, "_stage_stage3_submit_task", lambda p, s: None)
    m_k1 = TPUConnectorMetadata()
    m_k1.reqs_to_load = {
        "d_k1": dataclasses.replace(
            rel_meta, source_req_id="s_k1", release_only=False, local_block_ids=[3]
        )
    }
    wc_tp._submit_stage3_loads(m_k1, fake_eng, synchronous=True)
    assert "d_k1" in wc_tp._failed_recving

    # Lines 5200, 5215-5277: _stage_stage3_submit_task branches
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)
    monkeypatch.setattr(wc, "_new_raiden_id", lambda f: f)
    monkeypatch.setattr(
        wc, "_new_raiden_controller_facade", lambda addr: SimpleNamespace()
    )
    monkeypatch.setattr(wc, "_raiden_hbm_memory_type", lambda: "HBM")

    # Empty src_controller_address -> exception caught -> returns None (5200, 5265-5277)
    p_empty_addr = tc._Stage3PendingSubmit(
        req_meta=dataclasses.replace(
            rel_meta, release_only=False, src_controller_address=" "
        ),
        source_req_id="s_emp",
        destination_req_id="d_emp",
        uuid=1,
        num_tokens=16,
        local_blocks=[1],
        dst_controller_address="127.0.0.1:9000",
        dst_units=["u0"],
        defer_deadline=None,
        first_attempt_s=1.0,
    )
    assert wc._stage_stage3_submit_task(p_empty_addr, True) is None

    # DSv4 branch in _stage_stage3_submit_task (5215-5225)
    monkeypatch.setattr(tpu_envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    wc._stage3_dsv4_transfer_groups = (
        tc._Stage3TransferGroup(cache_group_index=0, page_tokens=16),
    )
    wc._stage3_dsv4_plan = (
        tc._Stage3Dsv4Pool(
            tag="dsv4.csa.nope.g0",
            plan_position=0,
            page_tokens=16,
            live_bytes_per_block=128,
            row_bytes=8,
        ),
    )
    p_dsv4 = dataclasses.replace(
        p_empty_addr,
        req_meta=dataclasses.replace(
            rel_meta,
            release_only=False,
            group_block_ids=[[1]],
        ),
        destination_req_id="d_dsv4",
    )
    task_dsv4 = wc._stage_stage3_submit_task(p_dsv4, True)
    assert task_dsv4 is not None
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_DSV4_ADMISSION", False)

    # GLM + state_group_count + skip_tokens in _stage_stage3_submit_task (5228,
    # 5236-5261)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    wc._stage3_row_geometry = {"mla.nope": (128, 8)}
    wc._stage3_state_group_count = 1
    p_missing_mamba = dataclasses.replace(
        p_empty_addr,
        req_meta=dataclasses.replace(
            rel_meta, release_only=False, mamba_state_block_ids=None
        ),
        destination_req_id="d_mm",
    )
    assert wc._stage_stage3_submit_task(p_missing_mamba, True) is None
    wc._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {"fa": {"live_bytes_per_block": 128}}
    )
    p_glm_ok = dataclasses.replace(
        p_empty_addr,
        req_meta=dataclasses.replace(
            rel_meta,
            release_only=False,
            mamba_state_block_ids=[5],
            skip_tokens=16,
        ),
        destination_req_id="d_glm",
    )
    task_glm = wc._stage_stage3_submit_task(p_glm_ok, True)
    assert task_glm is not None and task_glm.fa_skip_bytes == 128
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_GLM_ADMISSION", False)
    wc._stage3_state_group_count = 0

    # Lines 5319, 5323->5328, 5329-5330, 5354-5405, 5436-5475: dispatch, worker loop,
    # execute, apply
    wc._dispatch_stage3_load_submit(p_empty_addr, synchronous=True)
    # Synchronous dispatch with valid task
    task_glm = dataclasses.replace(
        task_glm, facade=SimpleNamespace(start_transfer=lambda **kw: False)
    )
    p_glm_ok.task = task_glm
    wc._stage3_submitted_loads["d_glm"] = 1
    wc._dispatch_stage3_load_submit(p_glm_ok, synchronous=True)

    # Kimi sharded consumer with all(ready) == True on tp_rank=0 (5166->5169,
    # 5323->5328)
    monkeypatch.setattr(
        tc.torch.distributed,
        "all_gather_object",
        lambda out, val, group=None: out.__setitem__(slice(None), [True, True]),
    )
    monkeypatch.setattr(wc_tp, "_stage_stage3_submit_task", lambda p, s: task_glm)
    wc_tp.tp_rank = 0
    m_k2 = TPUConnectorMetadata()
    m_k2.reqs_to_load = {
        "d_k2": dataclasses.replace(
            rel_meta, source_req_id="s_k2", release_only=False, local_block_ids=[3]
        )
    }
    wc_tp._submit_stage3_loads(m_k2, fake_eng, synchronous=True)

    # Execute when destination_req_id in _stage3_finished_loads_pending_cleanup
    # (5379-5389)
    wc._stage3_finished_loads_pending_cleanup.add("d_glm")
    out_aborted = wc._execute_stage3_submit(task_glm)
    assert out_aborted.error is not None
    wc._stage3_finished_loads_pending_cleanup.discard("d_glm")

    # Deferred mode (retry_inline=False) in _execute_stage3_submit (5400)
    task_defer = dataclasses.replace(
        task_glm,
        retry_inline=False,
        facade=SimpleNamespace(
            start_transfer=lambda **kw: (_ for _ in ()).throw(
                RuntimeError("Missing producer block registration")
            )
        ),
    )
    out_defer = wc._execute_stage3_submit(task_defer)
    # _apply_stage3_submit_outcome when already released (5436-5445)
    wc._stage3_submitted_loads.pop("d_glm", None)
    wc._apply_stage3_submit_outcome(out_defer)
    # _apply_stage3_submit_outcome deferred retry with wait_logged=True (5464->5475)
    wc._stage3_submitted_loads["d_glm"] = 1
    task_defer.pending.defer_deadline = tc.time.perf_counter() + 100.0
    task_defer.pending.wait_logged = True
    wc._apply_stage3_submit_outcome(out_defer)
    assert "d_glm" in wc._stage3_pending_submits

    # Worker loop callable & task (5354-5367)
    called_cb = []
    wc._stage3_submit_queue.put(lambda: called_cb.append(1))
    wc._stage3_submit_queue.put(
        dataclasses.replace(
            task_defer,
            facade=SimpleNamespace(start_transfer=lambda **kw: True),
        )
    )

    orig_get = wc._stage3_submit_queue.get

    def _stop_after_two():
        if wc._stage3_submit_queue.empty():
            raise StopIteration
        return orig_get(block=False)

    wc._stage3_submit_queue.get = _stop_after_two
    with pytest.raises(StopIteration):
        wc._stage3_submit_worker_loop()
    wc._stage3_submit_queue.get = orig_get
    assert called_cb == [1]

    # Lines 5597-5618: _sync_stage3_tp_submit_states on rank 0 and rank 1
    wc_tp._stage3_tp_submits = {"r_f", "r_u", "r_a", "r_none"}
    wc_tp._stage3_terminal_loads.add("r_f")
    wc_tp._stage3_pending_controller_failures["r_u"] = 100.0
    wc_tp._stage3_controller_accepted.add("r_a")
    wc_tp.tp_rank = 0
    wc_tp._sync_stage3_tp_submit_states()

    wc_tp._stage3_tp_submits = {"r_f", "r_u", "r_a"}
    wc_tp._stage3_inflight_submits = {"r_f": None, "r_u": None, "r_a": None}
    wc_tp._load_block_ids = {"r_f": [1], "r_u": [2], "r_a": [3]}
    wc_tp.tp_rank = 1
    monkeypatch.setattr(
        tc,
        "get_tp_group",
        lambda: SimpleNamespace(
            broadcast_object=lambda obj, src=0: {
                "r_f": "failed",
                "r_u": "uncertain",
                "r_a": "accepted",
            }
        ),
    )
    wc_tp._sync_stage3_tp_submit_states()
    assert (
        "r_f" in wc_tp._failed_recving
        and "r_u" in wc_tp._stage3_pending_controller_failures
    )

    # Lines 5624, 5632, 5656, 5664: _drain_stage3_pending_submits &
    # _raiden_destination_work_units
    wc_tp.tp_rank = 1
    wc_tp._drain_stage3_pending_submits(set())
    wc._stage3_pending_submits["d_glm"].last_attempt_s = tc.time.perf_counter()
    wc._drain_stage3_pending_submits({"nonexistent_req"})

    wc_single = _make_raiden_worker(monkeypatch, is_producer=False, tp_size=1)
    wc_single._raiden_work_unit = "u_single"
    assert wc_single._raiden_destination_work_units() == ["u_single"]

    # Lines 5719, 5722-5725, 5749, 5778, 5779, 5817, 5819-5823:
    # _submit_stage3_load_as_leader
    wc_lead = _make_raiden_worker(monkeypatch, is_producer=False, tp_size=2)
    monkeypatch.setattr(wc_lead, "_new_raiden_id", lambda f: f)
    monkeypatch.setattr(
        wc_lead,
        "_new_raiden_controller_facade",
        lambda addr: SimpleNamespace(start_transfer=lambda **kw: False),
    )
    monkeypatch.setattr(wc_lead, "_raiden_hbm_memory_type", lambda: "HBM")
    # Empty src_controller_address (5719)
    st, _, _, _, _ = wc_lead._submit_stage3_load_as_leader(
        req_meta=dataclasses.replace(rel_meta, src_controller_address=" "),
        source_req_id="s",
        destination_req_id="d",
        uuid=1,
        num_tokens=16,
        local_blocks=[1],
        dst_controller_address="127.0.0.1:9000",
        dst_units=["u0", "u1"],
    )
    assert st == "failed"

    # Missing mamba_state_block_ids (5749)
    wc_lead._stage3_state_group_count = 1
    st, _, _, _, _ = wc_lead._submit_stage3_load_as_leader(
        req_meta=rel_meta,
        source_req_id="s",
        destination_req_id="d",
        uuid=1,
        num_tokens=16,
        local_blocks=[1],
        dst_controller_address="127.0.0.1:9000",
        dst_units=["u0", "u1"],
    )
    assert st == "failed"
    wc_lead._stage3_state_group_count = 0

    # HND unaligned & aligned skip_tokens (5778, 5779)
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    st, _, _, _, _ = wc_lead._submit_stage3_load_as_leader(
        req_meta=dataclasses.replace(rel_meta, skip_tokens=3),
        source_req_id="s",
        destination_req_id="d",
        uuid=1,
        num_tokens=16,
        local_blocks=[1],
        dst_controller_address="127.0.0.1:9000",
        dst_units=["u0", "u1"],
    )
    assert st == "failed"
    monkeypatch.setattr(
        sol_mod, "sol_fa_geometry_from_manifest", lambda m: SimpleNamespace()
    )
    monkeypatch.setattr(
        sol_mod, "sol_fa_skip_bytes", lambda tok, geometry, dst_shards=1: 64
    )
    st, _, _, _, _ = wc_lead._submit_stage3_load_as_leader(
        req_meta=dataclasses.replace(rel_meta, skip_tokens=16),
        source_req_id="s",
        destination_req_id="d",
        uuid=1,
        num_tokens=16,
        local_blocks=[1],
        dst_controller_address="127.0.0.1:9000",
        dst_units=["u0", "u1"],
    )
    assert st == "uncertain"
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")

    # Controller returns False -> uncertain with NHD skip_tokens (5786, 5817, 5823)
    wc_lead._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {"fa": {"live_bytes_per_block": 128}}
    )
    st, _, _, _, _ = wc_lead._submit_stage3_load_as_leader(
        req_meta=dataclasses.replace(rel_meta, skip_tokens=16),
        source_req_id="s",
        destination_req_id="d",
        uuid=1,
        num_tokens=16,
        local_blocks=[1],
        dst_controller_address="127.0.0.1:9000",
        dst_units=["u0", "u1"],
    )
    assert st == "uncertain"

    # Lines 5887-5889, 5898, 5995, 6003, 6089->6087, 6120->6116, 6127->6129, 6180-6181:
    # get_finished & _poll_finished
    wc_fin = _make_raiden_worker(monkeypatch, is_producer=False)
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    wc_fin._raiden_transfer_engine = SimpleNamespace(
        poll_stats=lambda: ([], ["src_done"], ["src_fail"])
    )
    wc_fin._bind_stage3_request_ids("dst_done", "src_done")
    wc_fin._bind_stage3_request_ids("dst_fail", "src_fail")
    wc_fin._bind_stage3_request_ids("dst_replayed_fail", "src_replayed_fail")
    wc_fin._stage3_inflight_submits["dst_fail"] = tc.time.perf_counter() + 1000.0
    wc_fin._stage3_deferred_native_failures["dst_replayed_fail"] = True
    wc_fin._stage3_controller_accepted.update(
        {"dst_done", "dst_fail", "dst_replayed_fail"}
    )
    wc_fin._reported_recving.add("dst_done")
    wc_fin._poll_finished(wc_fin._raiden_transfer_engine)
    assert wc_fin._stage3_deferred_native_failures.get("dst_fail") is True

    # Latency_ms when start_time is None (6127->6129)
    wc_fin._reported_recving.discard("dst_done")
    wc_fin._stage3_submitted_loads["dst_done"] = 1
    wc_fin._stage3_submitted_load_tokens["dst_done"] = 16
    wc_fin._poll_finished(wc_fin._raiden_transfer_engine)

    # get_finished cleanup branches (5887-5889, 5888->5899, 5898)
    wc_fin._raiden_transfer_engine = SimpleNamespace(poll_stats=lambda: ([], [], []))
    monkeypatch.setattr(
        wc_fin, "_stage3_release_producer_registration", lambda *a, **kw: None
    )
    wc_fin._load_block_ids["c_unreported"] = [1]
    wc_fin._load_block_ids["c_unreported_nometa"] = [2]
    wc_fin._stage3_submitted_load_metas["c_unreported"] = rel_meta
    wc_fin._stage3_submitted_load_metas["c_failed"] = rel_meta
    wc_fin._failed_recving.add("c_failed")
    wc_fin.get_finished({"c_unreported", "c_unreported_nometa", "c_failed"})
    assert (
        wc_fin._resolve_remote_endpoint(
            SimpleNamespace(
                remote_host=["127.0.0.1", "127.0.0.2"], remote_port=[5000, 5002]
            )
        )
        == "127.0.0.1:5000"
    )

    # Producer _poll_finished branches (6089->6087, 6180-6181)
    wp_fin = _make_raiden_worker(monkeypatch, is_producer=True)
    wp_fin._raiden_controller_facade = SimpleNamespace(
        complete_request_blocks=lambda **kw: None
    )
    wp_fin._raiden_controller_address = "127.0.0.1:9000"
    wp_fin._raiden_transfer_engine = SimpleNamespace(
        poll_stats=lambda: (["p_done"], [], [])
    )
    wp_fin._stage3_registered_sends["p_done"] = tc._Stage3RegisteredSend(
        uuid=1, local_block_ids=(1,), num_tokens=16, expiration_time=1000.0
    )
    wp_fin._stage3_reported_sends.add("p_done")
    wp_fin._stage3_terminal_cleanup["ghost_req"] = True
    wp_fin._poll_finished(wp_fin._raiden_transfer_engine)
    assert "ghost_req" not in wp_fin._stage3_terminal_cleanup

    # Lines 6222, 6228, 6234-6241: _wait_for_recving
    wc_fin._wait_for_recving(set())
    wc_fin._done_recving.add("already_done")
    wc_fin._wait_for_recving({"already_done"})
    monkeypatch.setattr(tc.dist_utils, "get_p2p_wait_pull_timeout", lambda: 0.0)
    wc_fin._wait_for_recving({"never_done"})

    # Lines 6258, 6270, 6284->6310, 6295-6297, 6305, 6326-6340, 6358, 6461-6471, 6485:
    # _ensure_raiden_transfer_engine, _construct_raiden_transfer_engine,
    # _wait_for_address_ready, _register_raiden_pools
    w_no_runner = _make_raiden_worker(monkeypatch, is_producer=True)
    with pytest.raises(RuntimeError, match="register_runner must be called"):
        w_no_runner._ensure_raiden_transfer_engine()

    w_no_runner.runner = SimpleNamespace(kv_caches=[])
    monkeypatch.setattr(
        w_no_runner, "_new_raiden_manager", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(tc.dist_utils, "get_raiden_transfer_num_slots", lambda: 2)
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "legacy")
    # Covers 6270, 6284->6310, 6358
    w_no_runner._construct_raiden_transfer_engine([])

    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_RESHARD_IMPL", "store")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_ADVERTISE_HOST", "127.0.0.1")
    monkeypatch.setattr(
        w_no_runner, "_wait_for_address_ready", lambda addr, timeout_s=30.0: None
    )
    # Covers 6295-6297
    w_no_runner._construct_raiden_transfer_engine([], num_slots=1)
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_RESHARD_IMPL", "controller")
    monkeypatch.setattr(tpu_envs, "TPU_RAIDEN_CONTROLLER_ADDRESS", "127.0.0.1:9000")
    # Covers 6305
    w_no_runner._construct_raiden_transfer_engine([], num_slots=1)

    # _wait_for_address_ready (6326-6340)
    conn_attempts = []

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_create_conn(addr, timeout=0.5):
        conn_attempts.append(addr)
        if len(conn_attempts) == 1:
            raise OSError("not ready")
        return DummyConn()

    monkeypatch.setattr(socket, "create_connection", _fake_create_conn)
    monkeypatch.setattr(tc.time, "sleep", lambda s: None)
    tc.TPURaidenConnectorWorker._wait_for_address_ready("127.0.0.1:9000", timeout_s=5.0)
    assert len(conn_attempts) == 2

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda addr, timeout=0.5: (_ for _ in ()).throw(OSError("down")),
    )
    with pytest.raises(TimeoutError, match="Timed out waiting for address"):
        tc.TPURaidenConnectorWorker._wait_for_address_ready(
            "127.0.0.1:9000", timeout_s=0.0
        )

    # _register_raiden_pools (6461-6471, 6485)
    monkeypatch.setattr(tc.dist_utils, "get_raiden_pool_staging_leases", lambda: 2)

    class LegacyEng:
        def register_pools(self, pool_dicts, **kw):
            if kw:
                raise TypeError("unexpected keyword argument")
            return {"ok": 1}

    summary = w_no_runner._register_raiden_pools(
        LegacyEng(),
        SimpleNamespace(pool_dicts=lambda: [], pools=()),
        (),
    )
    assert summary == {"ok": 1}
    monkeypatch.setattr(tc.dist_utils, "get_raiden_pool_staging_leases", lambda: 0)
    assert w_no_runner._register_raiden_pools(
        LegacyEng(),
        SimpleNamespace(pool_dicts=lambda: [], pools=()),
        (),
    ) == {"ok": 1}
