import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec

from vllm_torchtpu.kv_cache_spec_normalizer import normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
)

pytestmark = pytest.mark.cpu_test


def test_attention_spec_with_bf16_normalizes_to_fp8_without_inflation():
    # Defect A regression: when input AttentionSpec has dtype=torch.bfloat16 (2 bytes),
    # normalizer must evaluate real_page_size_bytes on the target FP8 dtype (1 byte),
    # avoiding a 2x inflated page_size_padded.
    bf16_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=None,
    )
    assert bf16_spec.real_page_size_bytes == 16384

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": bf16_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    spec = normalized["layer"]

    pallas_page_size = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        16,
        2,
        128,
        torch.float8_e4m3fn,
    )
    assert pallas_page_size == 8192
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == pallas_page_size
    assert spec.page_size_padded < bf16_spec.real_page_size_bytes


def test_mla_attention_spec_with_bf16_normalizes_to_fp8_without_inflation():
    mla_bf16_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=None,
    )
    assert mla_bf16_spec.real_page_size_bytes == 16 * 1 * 576 * 2

    normalized = normalize_kv_cache_specs_for_tpu(
        {"mla_layer": mla_bf16_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    spec = normalized["mla_layer"]
    pallas_mla_page_size = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        16,
        1,
        576,
        torch.float8_e4m3fn,
    )
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == pallas_mla_page_size
    assert spec.page_size_padded < mla_bf16_spec.real_page_size_bytes


def test_attention_spec_uses_tpu_dtype_for_fp8_page_size():
    gpu_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": gpu_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    spec = normalized["layer"]

    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == max(
        PallasAttentionBackend.get_kv_cache_page_size_bytes(
            16,
            2,
            128,
            torch.float8_e4m3fn,
        ),
        gpu_spec.real_page_size_bytes,
    )


def test_mamba_spec_is_left_unchanged():
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=256,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": mamba_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["layer"] is mamba_spec


def test_hybrid_specs_keep_separate_page_size_by_default():
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["attn"].page_size_bytes != normalized["mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded is None


def test_hybrid_specs_use_uniform_page_size_when_unified_enabled():
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
        enable_unified_kv_layout=True,
    )

    assert normalized["attn"].page_size_bytes == normalized["mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized["attn"].page_size_bytes


def test_hybrid_specs_with_smaller_mamba_padded_size_normalizes_safely() -> None:
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=1572864,
    )
    # MambaSpec where page_size_padded was initialized to a smaller value
    # than unpadded size (e.g. from cache_config.mamba_page_size_padded)
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(6, 1, 8192), (32, 128, 128)],
        dtypes=[torch.float32, torch.bfloat16],
        page_size_padded=1146880,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["attn"].page_size_bytes == normalized["mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized["attn"].page_size_bytes
    assert normalized["mamba"].page_size_bytes == 1572864


def test_hybrid_specs_preserve_attention_pallas_padding_workload_a() -> None:
    # Workload A (Qwen 3.5 35B FP8 TP4):
    # Pallas attention kernel requires 786,432 B per page, whereas Mamba unpadded
    # state is 548,864 B. Normalization must preserve the attention Pallas padding.
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(274432,)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )
    assert mamba_spec.page_size_bytes == 548864

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    pallas_expected_page_size = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        1536,
        1,
        128,
        torch.float8_e4m3fn,
    )
    assert pallas_expected_page_size == 786432
    assert normalized["attn"].page_size_bytes == 786432
    assert normalized["attn"].page_size_padded == 786432
    assert normalized["mamba"].page_size_bytes == 786432
    assert normalized["mamba"].page_size_padded == 786432


def test_hybrid_specs_pad_attention_when_mamba_is_larger() -> None:
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(500000,)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )
    assert mamba_spec.page_size_bytes == 1000000

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["attn"].page_size_bytes == 1000000
    assert normalized["attn"].page_size_padded == 1000000
    assert normalized["mamba"].page_size_bytes == 1000000
    assert normalized["mamba"].page_size_padded == 1000000


def test_hybrid_specs_mamba_page_uses_pooled_physical_dtypes() -> None:
    # K3 TP32 regression: the KDA layer declares its conv state fp32 (the
    # fused kernel wants fp32 dense caches), but the unified pool stores the
    # conv region in bf16. Sizing the mamba page from the declared dtypes
    # (196608 + 13824 = 210432 B) outgrows the pool's attention-shaped page
    # (207360 B) and fails the materializer's page invariant; the pool
    # physical dtypes keep it at 203520 B so the platform-padded 207360 B
    # holds.
    attention_spec = MLAAttentionSpec(
        block_size=162,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
            162, 1, 576, torch.bfloat16
        ),
    )
    assert attention_spec.page_size_bytes == 207360
    mamba_spec = MambaSpec(
        block_size=32768,
        shapes=[(3, 3, 3, 128), (3, 128, 128)],
        dtypes=[torch.float32, torch.float32],
        page_size_padded=207360,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.bfloat16,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["mamba"].page_size_bytes == 207360
    assert normalized["mamba"].page_size_bytes == normalized["attn"].page_size_bytes
    # The retyped spec describes the pool's physical layout: bf16 conv, fp32
    # ssm.
    assert tuple(normalized["mamba"].dtypes) == (torch.bfloat16, torch.float32)


def test_exempt_layers_pass_through_untouched_under_unified_layout() -> None:
    # DeepSeek-V4 custom packed specs where page_size_padded < real_page_size_bytes.
    # If spec.page_size_bytes is called on such an AttentionSpec, it asserts:
    # `assert self.page_size_padded >= self.real_page_size_bytes`.
    # When exempted, this spec must pass through untouched without triggering assertions
    # during unified KV layout normalization.
    ds_v4_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=100,
    )
    assert ds_v4_spec.page_size_padded < ds_v4_spec.real_page_size_bytes

    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    specs = {
        "attn": attention_spec,
        "mamba": mamba_spec,
        "ds_v4": ds_v4_spec,
    }

    normalized = normalize_kv_cache_specs_for_tpu(
        specs,
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        exempt_layers={"ds_v4"},
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["ds_v4"] is ds_v4_spec
    assert normalized["ds_v4"].dtype == torch.bfloat16
    assert normalized["ds_v4"].page_size_padded == 100
    assert normalized["attn"].page_size_bytes == normalized["mamba"].page_size_bytes
    assert normalized["attn"].dtype == torch.float8_e4m3fn


def test_exempt_attention_does_not_trigger_hybrid_unification_with_mamba() -> None:
    ds_v4_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=100,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "ds_v4": ds_v4_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        exempt_layers={"ds_v4"},
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["ds_v4"] is ds_v4_spec
    assert normalized["mamba"] is mamba_spec
    assert normalized["mamba"].page_size_padded is None


def test_normalizer_remaining_branches(monkeypatch) -> None:
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from vllm_torchtpu import kv_cache_spec_normalizer as norm

    # _as_torch_dtype with string
    assert norm._as_torch_dtype("torch.bfloat16") is torch.bfloat16

    # _pooled_physical_mamba_spec when dtypes already match pooled physical dtypes
    already_pooled = MambaSpec(
        block_size=16,
        shapes=[(2, 8), (4, 4)],
        dtypes=[torch.bfloat16, torch.float32],
        page_size_padded=None,
    )
    assert norm._pooled_physical_mamba_spec(already_pooled) is already_pooled

    # Integer-typed MLAAttentionSpec keeps its integer dtype
    mla_uint8 = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        page_size_padded=1000000,
    )
    custom_mla = norm._normalize_one_spec(
        mla_uint8,
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    assert custom_mla.dtype is torch.uint8
    assert custom_mla.page_size_padded == 1000000

    # _has_hybrid_attention_and_mamba and _pad_page_size with generic KVCacheSpec
    class DummySpec(KVCacheSpec):
        @property
        def page_size_bytes(self) -> int:
            return 64

    dummy = DummySpec(block_size=16)
    assert not norm._has_hybrid_attention_and_mamba({"d": dummy})
    assert norm._pad_page_size(dummy, 128) is dummy


def test_lazy_logger_and_tpu_info_full_coverage(monkeypatch) -> None:
    from types import SimpleNamespace

    import requests

    from vllm_torchtpu import tpu_info
    from vllm_torchtpu.logger import init_logger

    lazy = init_logger("td761_test")
    assert lazy._logger is None
    assert callable(lazy.debug)
    assert lazy._logger is not None
    # Second getattr uses cached self._logger without calling _resolve
    assert callable(lazy.info)
    # Direct _resolve when _logger is already populated
    assert lazy._resolve() is lazy._logger

    # get_tpu_metadata success, non-200, and RequestException
    monkeypatch.setattr(
        tpu_info.requests,
        "get",
        lambda *a, **kw: SimpleNamespace(status_code=200, text="v6e-8"),
    )
    assert tpu_info.get_tpu_metadata("accelerator-type") == "v6e-8"

    monkeypatch.setattr(
        tpu_info.requests,
        "get",
        lambda *a, **kw: SimpleNamespace(status_code=404, text=""),
    )
    assert tpu_info.get_tpu_metadata("accelerator-type") is None

    def raise_req(*a, **kw):
        raise requests.RequestException("network down")

    monkeypatch.setattr(tpu_info.requests, "get", raise_req)
    assert tpu_info.get_tpu_metadata("accelerator-type") is None

    # get_tpu_type env vs metadata
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-4")
    assert tpu_info.get_tpu_type() == "v5litepod-4"
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "v6e-4")
    assert tpu_info.get_tpu_type() == "v6e-4"

    # get_chip_version
    tpu_info.get_chip_version.cache_clear()
    assert tpu_info.get_chip_version("v5litepod-4") == tpu_info.pltpu.ChipVersion("v5e")
    assert tpu_info.get_chip_version("tpu7x-8") == tpu_info.pltpu.ChipVersion.TPU_7X
    assert (
        tpu_info.get_chip_version("unknown-chip-99")
        == tpu_info.pltpu.ChipVersion.TPU_7X
    )
    tpu_info.get_chip_version.cache_clear()

    # get_node_name
    monkeypatch.setenv("TPU_NAME", "node-a")
    assert tpu_info.get_node_name() == "node-a"
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "node-b")
    assert tpu_info.get_node_name() == "node-b"

    # get_node_worker_id
    monkeypatch.setenv("TPU_WORKER_ID", "5")
    assert tpu_info.get_node_worker_id() == 5
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "7")
    assert tpu_info.get_node_worker_id() == 7
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": None)
    assert tpu_info.get_node_worker_id() == 0

    # get_num_cores_per_chip
    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: "v6e-8")
    assert tpu_info.get_num_cores_per_chip() == 1
    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: "v5p-8")
    assert tpu_info.get_num_cores_per_chip() == 2

    # get_num_chips
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: ["/dev/accel0"])
    assert tpu_info.get_num_chips() == 1
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(tpu_info.os, "listdir", lambda path: ["0", "1", "vfio"])
    assert tpu_info.get_num_chips() == 2

    def raise_fnf(path):
        raise FileNotFoundError("no vfio")

    monkeypatch.setattr(tpu_info.os, "listdir", raise_fnf)
    assert tpu_info.get_num_chips() == 0


def test_utils_full_coverage(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import jax.numpy as jnp

    from vllm_torchtpu import utils as tpu_utils

    assert tpu_utils.align_to(10, 8) == 16
    assert tpu_utils.get_megacore() is False
    tpu_utils.enable_megacore()
    assert tpu_utils.get_megacore() is True
    monkeypatch.setattr(tpu_utils, "_megacore", False)

    assert tpu_utils.largest_divisor(12, 8) == 6
    assert tpu_utils.largest_divisor(0, 0) == 1

    assert tpu_utils.get_num_kv_heads_by_tp(8, 4) == 8
    assert tpu_utils.get_num_kv_heads_by_tp(2, 8) == 8

    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    assert tpu_utils.get_dp_size(SimpleNamespace(data_parallel_size=2)) == 4
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    assert tpu_utils.get_dp_size(SimpleNamespace(data_parallel_size=2)) == 2

    # hbm_usage_bytes & hbm_usage_gb
    good_dev = MagicMock()
    good_dev.memory_stats.return_value = {
        "bytes_in_use": tpu_utils.GBYTES,
        "bytes_limit": 2 * tpu_utils.GBYTES,
    }
    bad_dev = MagicMock()
    bad_dev.memory_stats.side_effect = RuntimeError("not addressable")

    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    assert tpu_utils.hbm_usage_bytes([bad_dev, good_dev]) == [
        (tpu_utils.GBYTES, 2 * tpu_utils.GBYTES),
        (tpu_utils.GBYTES, 2 * tpu_utils.GBYTES),
    ]
    assert good_dev.memory_stats.call_count == 1
    assert tpu_utils.hbm_usage_bytes([bad_dev]) == []

    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)
    good_dev.memory_stats.reset_mock()
    assert tpu_utils.hbm_usage_gb([good_dev]) == [(1.0, 2.0)]
    assert good_dev.memory_stats.call_count == 1

    # get_device_name
    for raw_kind, num_devs, expected in [
        ("TPU v5 lite", None, "TPU v5e"),
        ("TPU v6e", None, "TPU v6e"),
        ("TPU v5p", None, "TPU v5p"),
        ("TPU v5i", None, "TPU v5i"),
        ("TPU7x", None, "TPU v7"),
        ("TPU8x", 8, "TPU v8-8"),
    ]:
        monkeypatch.setattr(
            tpu_utils.jax,
            "devices",
            lambda k=raw_kind: [SimpleNamespace(device_kind=k)],
        )
        assert tpu_utils.get_device_name(num_devs) == expected

    monkeypatch.setattr(
        tpu_utils.jax, "devices", lambda: [SimpleNamespace(device_kind="CPU")]
    )
    with pytest.raises(RuntimeError, match="Expected TPU devices"):
        tpu_utils.get_device_name()

    # get_mesh_shape_product
    fake_mesh = SimpleNamespace(shape={"tp": 4, "dp": 2})
    assert tpu_utils.get_mesh_shape_product(fake_mesh, None) == 1
    assert tpu_utils.get_mesh_shape_product(fake_mesh, "tp") == 4
    assert tpu_utils.get_mesh_shape_product(fake_mesh, ["tp", "dp"]) == 8

    # compute_hbm_budget
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda dev: (200, 1000))
    budget = tpu_utils.compute_hbm_budget(["d0", "d1"], 0.9)
    assert budget == tpu_utils.HbmBudget(
        total_limit=2000, total_used=1600, cap=1800, available=200
    )

    # estimate_kv_connector_hbm_reserve
    assert (
        tpu_utils.estimate_kv_connector_hbm_reserve(
            SimpleNamespace(kv_transfer_config=None)
        )
        == 0
    )
    assert (
        tpu_utils.estimate_kv_connector_hbm_reserve(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_connector_extra_config=None)
            )
        )
        == 0
    )
    assert (
        tpu_utils.estimate_kv_connector_hbm_reserve(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    kv_connector_extra_config={
                        "spec_name": "MissingSpec",
                        "spec_module_path": "nonexistent.module.td761",
                    }
                )
            )
        )
        == 0
    )
    fake_mod = SimpleNamespace(
        NoEstSpec=SimpleNamespace(),
        WithEstSpec=SimpleNamespace(estimate_hbm_reserve_bytes=lambda cfg: 4096),
    )
    monkeypatch.setitem(sys.modules, "fake_spec_mod_td761", fake_mod)
    assert (
        tpu_utils.estimate_kv_connector_hbm_reserve(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    kv_connector_extra_config={
                        "spec_name": "NoEstSpec",
                        "spec_module_path": "fake_spec_mod_td761",
                    }
                )
            )
        )
        == 0
    )
    assert (
        tpu_utils.estimate_kv_connector_hbm_reserve(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    kv_connector_extra_config={
                        "spec_name": "WithEstSpec",
                        "spec_module_path": "fake_spec_mod_td761",
                    }
                )
            )
        )
        == 4096
    )

    # head_dim, num_heads, dtype_packing, hash_fn, time_function
    assert tpu_utils.get_padded_head_dim(64) == 64
    assert tpu_utils.get_padded_head_dim(80) == 128
    assert tpu_utils.get_padded_num_heads(8, 4) == 8
    assert tpu_utils.get_padded_num_heads(2, 8) == 8
    assert tpu_utils.get_dtype_packing(jnp.bfloat16) == 2
    assert tpu_utils.get_hash_fn_by_name("builtin") is hash
    assert callable(tpu_utils.get_hash_fn_by_name("sha256"))

    @tpu_utils.time_function
    def add_one(x: int) -> int:
        return x + 1

    assert add_one(4) == 5

    # tpu_bind_kv_cache
    t0 = torch.zeros(1)
    t1 = torch.ones(1)

    class NativeLayer:
        tpu_native_kv = True

        def __init__(self):
            self.bound = None

        def bind_kv_cache(self, kv_cache, raw_cache=None, layout=None, rope_pool=None):
            self.bound = (kv_cache, raw_cache, layout, rope_pool)

    class StandardLayer:
        tpu_native_kv = False
        kv_cache = None

    l0 = NativeLayer()
    l1 = StandardLayer()
    runner_caches: list[torch.Tensor] = []
    tpu_utils.tpu_bind_kv_cache(
        {"model.layers.0.attn": t0, "model.layers.1.attn": [t1]},
        {"model.layers.0.attn": l0, "model.layers.1.attn": l1},
        runner_caches,
        native_raw_cache={"model.layers.0.attn": t0},
        native_cache_layouts=None,
    )
    assert runner_caches == [t0, t1]
    assert l0.bound == (t0, t0, None, None)
    assert l1.kv_cache == [t1]

    # With native_cache_layouts and native_rope_pools
    l0_rope = NativeLayer()
    tpu_utils.tpu_bind_kv_cache(
        {"model.layers.0.attn": t0},
        {"model.layers.0.attn": l0_rope},
        [],
        native_raw_cache=t0,
        native_cache_layouts={"model.layers.0.attn": ("layout",)},
        native_rope_pools={"model.layers.0.attn": t1},
    )
    assert l0_rope.bound == (t0, t0, ("layout",), t1)

    # With native_cache_layouts and native_rope_pools=None
    l0_layout = NativeLayer()
    tpu_utils.tpu_bind_kv_cache(
        {"model.layers.0.attn": t0},
        {"model.layers.0.attn": l0_layout},
        [],
        native_raw_cache=t0,
        native_cache_layouts={"model.layers.0.attn": ("layout",)},
        native_rope_pools=None,
    )
    assert l0_layout.bound == (t0, t0, ("layout",), None)

    # synchronize_device and synchronize_tensors
    sync_calls = []
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: sync_calls.append("device")
    )
    tpu_utils.synchronize_device()
    from torch_tpu._internal import sync as tpu_sync

    monkeypatch.setattr(
        tpu_sync,
        "synchronize",
        lambda tensors=None, wait=True: sync_calls.append(("tensors", tensors, wait)),
    )
    tpu_utils.synchronize_tensors([t0], wait=False)
    assert sync_calls == ["device", ("tensors", [t0], False)]
