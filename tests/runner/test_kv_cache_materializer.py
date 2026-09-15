import pytest
import torch
from vllm.v1.kv_cache_interface import (EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.kv_cache_materializer import (
    format_kv_cache_layout_summary, materialize_kv_cache_tensors)
from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend


class FakeAttentionBackend:

    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str,
    ):
        return (num_blocks, block_size, num_kv_heads, 2, head_size)


def make_attention_only_config(num_blocks: int = 4) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.0.self_attn"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            )
        ],
    )


def make_hybrid_config(
    num_blocks: int = 4,
    attn_dtype: torch.dtype = torch.bfloat16,
) -> KVCacheConfig:
    # Size the attention page by the actual KV dtype: the unified pool is
    # born attention-shaped, so the page bytes and the spec dtype have to
    # agree (as the real block-size derivation guarantees).
    page_size = 16 * 2 * 2 * 8 * attn_dtype.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=attn_dtype,
        page_size_padded=page_size,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2, 8), (2, 4, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.0.self_attn", "model.layers.1.mamba"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.mamba"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )


def make_attention_encoder_attention_config(
        num_blocks: int = 4) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    encoder_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.0.self_attn"],
            ),
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.2.self_attn"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.encoder_attn"],
                kv_cache_spec=encoder_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.2.self_attn"],
                kv_cache_spec=attn_spec,
            ),
        ],
    )


def make_hybrid_with_encoder_gap_config(num_blocks: int = 4) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    encoder_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2, 8), (2, 4, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=[
                    "model.layers.0.self_attn",
                    "model.layers.2.self_attn",
                    "model.layers.3.mamba",
                ],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.encoder_attn"],
                kv_cache_spec=encoder_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.2.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.3.mamba"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )


def make_attention_groups(cfg: KVCacheConfig) -> list[list[AttentionGroup]]:
    return [[
        AttentionGroup(
            backend=FakeAttentionBackend,
            layer_names=list(group.layer_names),
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=gid,
        )
    ] for gid, group in enumerate(cfg.kv_cache_groups)]


def test_attention_only_allocates_direct_cache_without_raw_backing():
    cfg = make_attention_only_config(num_blocks=4)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group]],
        kernel_block_sizes=[16],
        device=torch.device("cpu"),
        cache_dtype=torch.bfloat16,
    )

    assert materialized.raw_tensors == []
    attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    assert isinstance(attn_cache, torch.Tensor)
    assert attn_cache.shape == (4, 16, 2, 2, 8)
    assert attn_cache.dtype == torch.bfloat16
    assert attn_cache.is_contiguous()


def test_direct_attention_materialization_maps_kernel_block_sizes_after_encoder_only_gap(
):
    cfg = make_attention_encoder_attention_config(num_blocks=4)

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=make_attention_groups(cfg),
        kernel_block_sizes=[16, 8],
        device=torch.device("cpu"),
        cache_dtype=torch.bfloat16,
    )

    assert "model.layers.1.encoder_attn" not in materialized.kv_caches
    first_attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    second_attn_cache = materialized.kv_caches["model.layers.2.self_attn"]
    assert isinstance(first_attn_cache, torch.Tensor)
    assert isinstance(second_attn_cache, torch.Tensor)
    assert first_attn_cache.shape == (4, 16, 2, 2, 8)
    assert second_attn_cache.shape == (8, 8, 2, 2, 8)


@pytest.mark.parametrize("attn_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_hybrid_materializes_one_attention_shaped_pool(attn_dtype):
    cfg = make_hybrid_config(num_blocks=4, attn_dtype=attn_dtype)

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_hybrid_groups(cfg),
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype=("fp8"
                     if attn_dtype == torch.float8_e4m3fn else "bfloat16"),
    )

    pool = materialized.raw_tensors[0]
    assert len(materialized.raw_tensors) == 1
    assert pool.dtype == attn_dtype
    assert pool.shape == (4, 16, 2, 2, 8)
    assert materialized.kv_caches["model.layers.0.self_attn"] is pool
    assert materialized.kv_caches["model.layers.1.mamba"] == [pool]


def test_hybrid_materialization_builds_raiden_logical_regions():
    cfg = make_hybrid_config(num_blocks=4, attn_dtype=torch.bfloat16)
    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_hybrid_groups(cfg),
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )

    manifest = rpm.build_qwen35_pool_manifest(
        named_kv_caches=materialized.kv_caches,
        kv_cache_groups=cfg.kv_cache_groups,
        raw_tensors=materialized.raw_tensors,
        gdn_geometry=rpm.GdnHeadGeometry(local_key_heads=1,
                                         local_value_heads=1,
                                         key_head_dim=2,
                                         value_head_dim=4,
                                         conv_kernel_size=3),
        mamba_group_ordinal_by_layer={"model.layers.1.mamba": 0},
    )

    assert manifest.binding == rpm.BINDING_ALIASED_RAW
    assert len(manifest.storages) == 1
    assert manifest.storages[0] is materialized.raw_tensors[0]
    assert len(manifest.pools) == 3
    fa, conv, ssm = manifest.pools
    assert [fa.tag, conv.tag, ssm.tag] == [
        rpm.TAG_FA,
        f"{rpm.TAG_GDN_CONV}.g0",
        f"{rpm.TAG_GDN_SSM}.g0",
    ]
    assert all(pool.num_blocks == 4 for pool in manifest.pools)
    assert all(pool.block_stride_bytes == 1024 for pool in manifest.pools)
    assert ssm.base_offset_bytes == 0
    # Kernel-tied SSM bytes from the geometry (1 V head × 4 × 2 fp32), not
    # from the declared MambaSpec ssm shape.
    assert conv.base_offset_bytes == 1 * 4 * 2 * torch.float32.itemsize
    rpm.verify_storage_binding(manifest, materialized.kv_caches,
                               materialized.raw_tensors)


def test_hybrid_materialization_skips_encoder_only_kernel_entry():
    cfg = make_hybrid_with_encoder_gap_config(num_blocks=4)

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=make_attention_groups(cfg),
        kernel_block_sizes=[16, 8, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )

    assert "model.layers.1.encoder_attn" not in materialized.kv_caches
    first_attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    second_attn_cache = materialized.kv_caches["model.layers.2.self_attn"]
    mamba_cache = materialized.kv_caches["model.layers.3.mamba"]
    pool = materialized.raw_tensors[0]
    assert pool.shape == (4, 16, 2, 2, 8)
    assert first_attn_cache is pool
    assert second_attn_cache is pool
    assert mamba_cache == [pool]


@pytest.mark.parametrize("kernel_block_sizes", ([16], [16, 8, 16]))
def test_materialization_rejects_mismatched_kernel_block_size_count(
        kernel_block_sizes):
    cfg = make_attention_encoder_attention_config(num_blocks=4)

    with pytest.raises(ValueError):
        materialize_kv_cache_tensors(
            kv_cache_config=cfg,
            attn_groups=make_attention_groups(cfg),
            kernel_block_sizes=kernel_block_sizes,
            device=torch.device("cpu"),
            cache_dtype=torch.bfloat16,
        )


def _hybrid_groups(cfg: KVCacheConfig) -> list[list[AttentionGroup]]:
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )
    mamba_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.1.mamba"],
        kv_cache_spec=cfg.kv_cache_groups[1].kv_cache_spec,
        kv_cache_group_id=1,
    )
    return [[attn_group], [mamba_group]]


def test_pool_fp8_is_zero_filled(monkeypatch):
    """The fp8 pool must be zeroed, and the pool layout must survive it.

    This test previously asserted the opposite -- that the fp8 pool is born
    uninitialized via ``torch.empty``, on the stated grounds that "fp8 pools
    cannot zero-fill on TPU (and slots are overwritten before use)". Both
    halves were wrong, and measurement is what settled it:

    * ``torch.zeros`` works fine for both fp8 dtypes on a v7x pod --
      0 non-zero bytes, and 2.078 ms median for the whole 1.930 GiB 48-tensor
      pool, 1.855 ms more than ``torch.empty`` (``j-4e2906de``), plus a
      one-time ~394 ms fill-kernel compile. Not free, not remotely a startup
      problem.
    * The slots are *not* all overwritten before use. A page holds more rows
      than the sequence occupying it, and the attention kernel reads the whole
      page. RPA v3's default ``RPA_V_MASK_MODE=none`` relies on ``p == 0`` to
      annihilate the rows past ``effective_kv_len``, which annihilates a
      finite V and nothing else -- and ``torch.empty`` on TPU fp8 returns
      524288/524288 non-finite bytes, because ``0xFF`` is a NaN in
      ``f8e4m3fn`` (``j-761bc2df``, reproduced in ``j-f636426b`` and
      ``j-d329490e``). That combination is NaN on every decode *of a model
      that takes this route*, which is the narrower claim: the ``torch.empty``
      fast path lived only in ``allocate_raw_kv_cache_tensors``, reached only
      when ``_can_allocate_attention_cache_directly()`` is False -- i.e.
      hybrid attention+mamba. Qwen3-Coder-30B takes the direct route, which
      was already zeroed (``j-73a8aaba``: ``AFFECTED_BY_CHANGE = False``).

    See ``tests/test_kv_cache_zero_init.py`` for the invariant this upholds
    and ``mask_v()`` in the RPA v3 kernel for who depends on it.
    """
    # Zeroing is the property under test, so prove the fp8 path reaches
    # torch.zeros rather than trusting that it did.
    torch_zeros = torch.zeros
    fp8_zero_calls = []

    def counting_zeros(*args, **kwargs):
        if kwargs.get("dtype") in (torch.float8_e4m3fn, torch.float8_e5m2):
            fp8_zero_calls.append(kwargs["dtype"])
        return torch_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", counting_zeros)
    cfg = make_hybrid_config(num_blocks=4, attn_dtype=torch.float8_e4m3fn)

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_hybrid_groups(cfg),
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="fp8",
    )

    attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    mamba_cache = materialized.kv_caches["model.layers.1.mamba"]

    assert attn_cache.dtype == torch.float8_e4m3fn
    assert attn_cache.shape == (4, 16, 2, 2, 8)
    assert attn_cache is materialized.raw_tensors[0]
    assert isinstance(mamba_cache, list)
    assert len(mamba_cache) == 1
    assert mamba_cache[0] is attn_cache

    assert fp8_zero_calls == [torch.float8_e4m3fn], (
        f"the fp8 pool was not allocated through torch.zeros (saw "
        f"{fp8_zero_calls}); RPA v3's V-mask elision requires finite padding")
    assert not attn_cache.view(torch.uint8).any(), (
        "the fp8 pool came up with non-zero bytes; uninitialized fp8 is "
        "100% non-finite on TPU and 0 * NaN is NaN")


def test_pool_layout_summary_shows_single_pool_state():
    cfg = make_hybrid_config(num_blocks=4)
    attn_groups = _hybrid_groups(cfg)
    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=attn_groups,
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )

    summary = format_kv_cache_layout_summary(
        kv_cache_config=cfg,
        kv_caches=materialized.kv_caches,
        raw_tensors=materialized.raw_tensors,
        attn_groups=attn_groups,
    )

    assert "TPU KV cache layout summary:" in summary
    # The raw backing tensor IS the attention-shaped pool (bf16, contiguous,
    # offset 0), and the mamba entry is the single pool view.
    assert "raw[0]: shape=(4, 16, 2, 2, 8) dtype=torch.bfloat16" in summary
    assert "model.layers.1.mamba: mamba states=1" in summary
    assert "state[0]: shape=(4, 16, 2, 2, 8)" in summary
    assert "contiguous=True" in summary
    assert "storage_offset_bytes=0" in summary


def test_hybrid_materialization_workload_a_geometry() -> None:
    attn_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(274432, )],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    norm_specs = normalize_kv_cache_specs_for_tpu(
        {
            "model.layers.0.self_attn": attn_spec,
            "model.layers.1.mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )
    norm_attn = norm_specs["model.layers.0.self_attn"]
    norm_mamba = norm_specs["model.layers.1.mamba"]
    assert norm_attn.page_size_bytes == 786432
    assert norm_mamba.page_size_bytes == 786432

    num_blocks = 2
    pool_page_bytes = norm_attn.page_size_bytes
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=pool_page_bytes * num_blocks,
                shared_by=[
                    "model.layers.0.self_attn",
                    "model.layers.1.mamba",
                ],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=norm_attn,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.mamba"],
                kv_cache_spec=norm_mamba,
            ),
        ],
    )

    attn_groups = [
        [
            AttentionGroup(
                backend=PallasAttentionBackend,
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=norm_attn,
                kv_cache_group_id=0,
            )
        ],
        [
            AttentionGroup(
                backend=PallasAttentionBackend,
                layer_names=["model.layers.1.mamba"],
                kv_cache_spec=norm_mamba,
                kv_cache_group_id=1,
            )
        ],
    ]

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=attn_groups,
        kernel_block_sizes=[512, 512],
        device=torch.device("cpu"),
        cache_dtype="fp8",
    )

    assert len(materialized.raw_tensors) == 1
    pool = materialized.raw_tensors[0]
    assert pool.dtype == torch.float8_e4m3fn
    assert materialized.kv_caches["model.layers.0.self_attn"] is pool
    assert materialized.kv_caches["model.layers.1.mamba"] == [pool]
