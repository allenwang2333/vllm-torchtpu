# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TD7.55 platform validation and TPU block-size derivation."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import MultipleOf

from vllm_torchtpu.gdn_pool_layout import QWEN_GDN_ARCHITECTURES, PooledGDNStateLayout
from vllm_torchtpu.platforms import pcp_validation as pcp_val
from vllm_torchtpu.platforms import pp_validation as pp_val
from vllm_torchtpu.platforms import tpu_block_size_utils as bsu

# ---------------------------------------------------------------------------
# 1. pcp_validation.py Tests
# ---------------------------------------------------------------------------


def _make_pcp_vllm_cfg(
    *,
    pcp_size: int = 2,
    interleave_size: int = 128,
    dcp_size: int = 1,
    pp_size: int = 1,
    async_scheduling: bool = False,
    spec_method: str | None = None,
    num_spec_tokens: int = 0,
    is_moe: bool = False,
    ep: bool = False,
):
    spec_cfg = (
        None
        if spec_method is None
        else SimpleNamespace(
            method=spec_method,
            num_speculative_tokens=num_spec_tokens,
        )
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            decode_context_parallel_size=dcp_size,
            pipeline_parallel_size=pp_size,
            is_moe_model=is_moe,
            enable_expert_parallel=ep,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=spec_cfg,
    )


def test_pcp_static_support_validator_all_branches():
    # None vllm_config
    default_cfg = pcp_val.PcpStaticSupportValidator.from_vllm_config(None)
    assert default_cfg.enabled is False

    # Disabled PCP returns immediately
    disabled_vllm = _make_pcp_vllm_cfg(pcp_size=1)
    res_disabled = pcp_val.PcpStaticSupportValidator.validate_platform_config(
        disabled_vllm, kv_cache_layout="HND"
    )
    assert res_disabled.enabled is False

    # HND layout with non-128 interleave_size raises ValueError
    with pytest.raises(ValueError, match="multiple of 128"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, interleave_size=64),
            kv_cache_layout="HND",
        )

    # DCP > 1 raises NotImplementedError
    with pytest.raises(NotImplementedError, match="does not support DCP"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, dcp_size=2),
            kv_cache_layout="NHD",
        )

    # PP != 1 raises NotImplementedError
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, pp_size=2),
            kv_cache_layout="NHD",
        )

    # MoE without EP raises NotImplementedError
    with pytest.raises(NotImplementedError, match="requires --enable-expert-parallel"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, is_moe=True, ep=False),
            kv_cache_layout="NHD",
        )

    # Speculative decoding with non-mtp method raises NotImplementedError
    with pytest.raises(NotImplementedError, match="requires method=mtp"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, spec_method="eagle3", num_spec_tokens=1),
            kv_cache_layout="NHD",
        )

    # Speculative decoding with mtp and num_speculative_tokens != 1 raises
    with pytest.raises(NotImplementedError, match="num_speculative_tokens=1"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, spec_method="mtp", num_spec_tokens=2),
            kv_cache_layout="NHD",
        )

    # interleave_size <= 0 with NHD layout raises ValueError
    with pytest.raises(ValueError, match="cp_kv_cache_interleave_size > 0"):
        pcp_val.PcpStaticSupportValidator.validate_platform_config(
            _make_pcp_vllm_cfg(pcp_size=2, interleave_size=0),
            kv_cache_layout="NHD",
        )

    # Valid PCP with MTP=1 and MoE+EP
    valid_mtp = pcp_val.PcpStaticSupportValidator.validate_platform_config(
        _make_pcp_vllm_cfg(
            pcp_size=2,
            interleave_size=128,
            spec_method="mtp",
            num_spec_tokens=1,
            is_moe=True,
            ep=True,
        ),
        kv_cache_layout="HND",
    )
    assert valid_mtp.enabled is True

    # Valid PCP without speculative decoding
    valid_no_spec = pcp_val.PcpStaticSupportValidator.validate_platform_config(
        _make_pcp_vllm_cfg(pcp_size=2, interleave_size=128),
        kv_cache_layout="HND",
    )
    assert valid_no_spec.speculative_enabled is False


# ---------------------------------------------------------------------------
# 2. pp_validation.py Tests
# ---------------------------------------------------------------------------


def _make_pp_vllm_cfg(
    *,
    pp_size: int = 2,
    tp_size: int = 1,
    async_scheduling: bool = False,
    speculative_config=None,
    kv_connector: str | None = None,
    layer_types: list[str] | None = None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative_config,
        kv_transfer_config=(
            None if kv_connector is None else SimpleNamespace(kv_connector=kv_connector)
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types)
        ),
    )


def test_pp_validation_all_branches(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0")
    assert pp_val._legacy_ray_executor() is True

    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")
    assert pp_val._legacy_ray_executor() is False

    # PP <= 1 is a no-op
    pp_val.validate_pipeline_parallel_config(_make_pp_vllm_cfg(pp_size=1))

    # TP > 1 raises NotImplementedError
    with pytest.raises(NotImplementedError, match="tensor parallelism inside a stage"):
        pp_val.validate_pipeline_parallel_config(
            _make_pp_vllm_cfg(pp_size=2, tp_size=2)
        )

    # async_scheduling raises NotImplementedError
    with pytest.raises(NotImplementedError, match="async scheduling"):
        pp_val.validate_pipeline_parallel_config(
            _make_pp_vllm_cfg(pp_size=2, async_scheduling=True)
        )

    # speculative_config raises NotImplementedError
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        pp_val.validate_pipeline_parallel_config(
            _make_pp_vllm_cfg(pp_size=2, speculative_config=object())
        )

    # Non-Raiden KV connector raises NotImplementedError
    with pytest.raises(NotImplementedError, match="other than TPURaidenConnector"):
        pp_val.validate_pipeline_parallel_config(
            _make_pp_vllm_cfg(pp_size=2, kv_connector="OtherConnector")
        )

    # Raiden connector with a stage lacking full attention raises NotImplementedError
    with pytest.raises(NotImplementedError, match="holding no full-attention layer"):
        pp_val.validate_pipeline_parallel_config(
            _make_pp_vllm_cfg(
                pp_size=2,
                kv_connector="TPURaidenConnector",
                layer_types=[
                    "full_attention",
                    "full_attention",
                    "linear_attention",
                    "mamba",
                ],
            )
        )

    # Legacy Ray executor raises NotImplementedError
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0")
    with pytest.raises(NotImplementedError, match="legacy Ray executor"):
        pp_val.validate_pipeline_parallel_config(
            _make_pp_vllm_cfg(
                pp_size=2,
                kv_connector="TPURaidenConnector",
                layer_types=[
                    "full_attention",
                    "linear_attention",
                    "deepseek_sparse_attention",
                    "linear_attention",
                ],
            )
        )

    # Valid PP config with layer_types=None and Ray V2
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")
    pp_val.validate_pipeline_parallel_config(
        _make_pp_vllm_cfg(
            pp_size=2,
            kv_connector="TPURaidenConnector",
            layer_types=None,
        )
    )


# ---------------------------------------------------------------------------
# 3. tpu_block_size_utils.py Tests
# ---------------------------------------------------------------------------


class _FakeBackend:
    supported = [MultipleOf(16), 32]
    min_page_size = 16
    packing = 1
    return_tensor = False
    break_linearity_at_final = False

    @classmethod
    def get_name(cls) -> str:
        return "FAKE_BACKEND"

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return cls.supported

    @classmethod
    def get_min_page_size(cls, _vllm_config) -> int:
        return cls.min_page_size

    @classmethod
    def get_kv_cache_page_size_bytes(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype: torch.dtype,
    ):
        if cls.break_linearity_at_final and block_size >= 16:
            return 999984
        rows = max(1, (block_size + cls.packing - 1) // cls.packing)
        val = rows * cls.packing * num_kv_heads * 2 * head_size * cache_dtype.itemsize
        if cls.return_tensor:
            return torch.tensor(val, dtype=torch.int64)
        return val


def _make_bsu_vllm_cfg(
    *,
    architecture: str = "LlamaForCausalLM",
    is_hybrid: bool = False,
    cache_dtype="auto",
    model_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    user_specified_block_size: bool = False,
    mamba_block_size: int = 16,
    mamba_cache_mode: str = "align",
    tp_size: int = 1,
    pcp_size: int = 1,
    num_kv_heads: int = 2,
    head_size: int = 8,
):
    hf_text_config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            is_hybrid=is_hybrid,
            dtype=model_dtype,
            hf_text_config=hf_text_config,
            get_num_kv_heads=lambda _pc: num_kv_heads,
            get_head_size=lambda: head_size,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=block_size,
            user_specified_block_size=user_specified_block_size,
            mamba_block_size=mamba_block_size,
            mamba_page_size_padded=None,
            mamba_cache_mode=mamba_cache_mode,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def test_tpu_block_size_basic_helpers(monkeypatch):
    cfg = _make_bsu_vllm_cfg(cache_dtype="auto", model_dtype=torch.bfloat16)
    assert isinstance(bsu.unified_kv_layout_enabled(cfg), bool)
    assert bsu._align_block_to_backend(10, []) == 10
    assert bsu._align_block_to_backend(10, [MultipleOf(16), 32]) == 16

    assert bsu._resolve_tpu_cache_dtype(cfg) == torch.bfloat16
    cfg_fp8 = _make_bsu_vllm_cfg(cache_dtype=" FP8 ")
    assert bsu._resolve_tpu_cache_dtype(cfg_fp8) == torch.float8_e4m3fn
    cfg_bad = _make_bsu_vllm_cfg(cache_dtype="unknown_dtype")
    with pytest.raises(ValueError, match="Unsupported TPU KV cache dtype"):
        bsu._resolve_tpu_cache_dtype(cfg_bad)

    # _tpu_attention_slot_size_bytes with tensor return and int return
    class _TensorBackend(_FakeBackend):
        return_tensor = True
        packing = 2

    assert bsu._tpu_attention_slot_size_bytes(cfg, _TensorBackend, 2) > 0
    assert bsu._pool_row_tokens(cfg, _TensorBackend) == 2

    class _NonLinearBackend(_FakeBackend):
        @classmethod
        def get_kv_cache_page_size_bytes(cls, *_args):
            return 64  # never doubles

    with pytest.raises(ValueError, match="not token-linear"):
        bsu._pool_row_tokens(cfg, _NonLinearBackend)


def test_localize_gdn_state_shapes_and_hybrid_mamba_state_layout(monkeypatch):
    qwen_arch = next(iter(QWEN_GDN_ARCHITECTURES))
    cfg_qwen = _make_bsu_vllm_cfg(
        architecture=qwen_arch, is_hybrid=True, tp_size=2, pcp_size=2
    )

    with pytest.raises(ValueError, match="requires two GDN state regions"):
        bsu._localize_gdn_state_shapes_for_pcp(cfg_qwen, ((16,),), pcp_size=2)

    # tp_size=2 -> tp_local_kq_heads=2 (d_k=8), tp_local_v_heads=4 (d_v=16)
    # expected_conv_dim = 2 * 2 * 8 + 4 * 16 = 32 + 64 = 96
    with pytest.raises(ValueError, match="conv width does not match"):
        bsu._localize_gdn_state_shapes_for_pcp(
            cfg_qwen, ((4, 10), (4, 8, 16)), pcp_size=2
        )

    with pytest.raises(ValueError, match="recurrent state does not match"):
        bsu._localize_gdn_state_shapes_for_pcp(
            cfg_qwen, ((4, 96), (2, 8, 16)), pcp_size=2
        )

    valid_pcp_shapes = bsu._localize_gdn_state_shapes_for_pcp(
        cfg_qwen, ((4, 96), (4, 8, 16)), pcp_size=2
    )
    assert len(valid_pcp_shapes) == 2

    # Non-hybrid returns None
    assert (
        bsu._hybrid_mamba_state_layout(_make_bsu_vllm_cfg(is_hybrid=False), 64) is None
    )

    # Mock ModelRegistry.resolve_model_cls and is_conv_state_dim_first
    import vllm.model_executor.layers.mamba.mamba_utils as mamba_utils
    from vllm.model_executor.models import ModelRegistry

    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: False)

    class _FakeQwenModel:
        @classmethod
        def get_mamba_state_shape_from_config(cls, _vllm_cfg):
            # Full TP=1 conv width: 2*4*8 + 8*16 = 192; V heads = 8
            return ((4, 192), (8, 8, 16))

        @classmethod
        def get_mamba_state_dtype_from_config(cls, _vllm_cfg):
            return (torch.bfloat16, torch.float32)

    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda arch, model_config=None: (_FakeQwenModel, arch),
    )
    layout_qwen_pcp = bsu._hybrid_mamba_state_layout(cfg_qwen, 64)
    assert isinstance(layout_qwen_pcp, PooledGDNStateLayout)

    cfg_qwen_no_pcp = _make_bsu_vllm_cfg(
        architecture=qwen_arch, is_hybrid=True, tp_size=2, pcp_size=1
    )
    assert bsu._hybrid_mamba_state_layout(cfg_qwen_no_pcp, 64) is not None

    # Other hybrid architecture with PCP > 1: valid/invalid ratio, bad shapes/dtypes
    class _FakeOtherHybridModel:
        shapes = ((4, 64), (4, 8, 16))
        dtypes = (torch.bfloat16, torch.float32)
        scale_with_tp = True

        @classmethod
        def get_mamba_state_shape_from_config(cls, vcfg):
            if len(cls.shapes) != 2:
                return cls.shapes
            tp = vcfg.parallel_config.tensor_parallel_size
            factor = tp if cls.scale_with_tp else 1
            return (
                (cls.shapes[0][0], cls.shapes[0][1] // factor),
                (cls.shapes[1][0] // factor, cls.shapes[1][1], cls.shapes[1][2]),
            )

        @classmethod
        def get_mamba_state_dtype_from_config(cls, _vcfg):
            return cls.dtypes

    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda arch, model_config=None: (_FakeOtherHybridModel, arch),
    )
    cfg_other_pcp = _make_bsu_vllm_cfg(
        architecture="OtherHybridModel", is_hybrid=True, tp_size=1, pcp_size=2
    )
    assert bsu._hybrid_mamba_state_layout(cfg_other_pcp, 64) is not None

    _FakeOtherHybridModel.scale_with_tp = False
    with pytest.raises(ValueError, match="exactly 1/pcp_size"):
        bsu._hybrid_mamba_state_layout(cfg_other_pcp, 64)
    _FakeOtherHybridModel.scale_with_tp = True

    _FakeOtherHybridModel.dtypes = (torch.bfloat16,)
    with pytest.raises(ValueError, match="two GDN state dtypes"):
        bsu._hybrid_mamba_state_layout(cfg_other_pcp, 64)
    _FakeOtherHybridModel.dtypes = (torch.bfloat16, torch.float32)

    _FakeOtherHybridModel.shapes = ((4, 64),)
    with pytest.raises(ValueError, match="two GDN state regions"):
        bsu._hybrid_mamba_state_layout(cfg_other_pcp, 64)


def test_derive_and_update_tpu_block_size_and_slot_config(monkeypatch):
    # 1. Early return when input_block_size == 0
    cfg_zero = _make_bsu_vllm_cfg(block_size=0)
    bsu.update_tpu_block_size_and_slot_config(cfg_zero, _FakeBackend)

    # 2. Non-hybrid derivation and min_page_size violation
    class _HighMinPageBackend(_FakeBackend):
        min_page_size = 64

    cfg_non_hybrid = _make_bsu_vllm_cfg(block_size=16, is_hybrid=False)
    bsu.update_tpu_block_size_and_slot_config(cfg_non_hybrid, _FakeBackend)
    assert cfg_non_hybrid.cache_config.block_size == 16
    with pytest.raises(ValueError, match="violate min_page_size"):
        bsu.update_tpu_block_size_and_slot_config(cfg_non_hybrid, _HighMinPageBackend)

    # 3. Hybrid derivation covering block_size_source + mamba_cache_mode branches
    fake_layout = PooledGDNStateLayout(
        ssm_bytes=512,
        conv_bytes=256,
        token_bytes=64,
        ssm_tokens=24,
        conv_tokens=8,
    )
    monkeypatch.setattr(
        bsu, "_hybrid_mamba_state_layout", lambda *_a, **_kw: fake_layout
    )

    # 3a. source = "mamba_state_fit" (32 > min_page=16, user_specified=False) + align
    cfg_fit = _make_bsu_vllm_cfg(
        block_size=16,
        is_hybrid=True,
        user_specified_block_size=False,
        mamba_cache_mode="align",
    )
    bsu.update_tpu_block_size_and_slot_config(cfg_fit, _FakeBackend)
    assert cfg_fit.cache_config.block_size == 32
    assert cfg_fit.cache_config.mamba_block_size == 32
    assert cfg_fit.cache_config.mamba_page_size_padded == 2048

    # 3b. source = "user_block_size_floor" (input=64 > 32, user_specified=True) + all
    cfg_user = _make_bsu_vllm_cfg(
        block_size=64,
        is_hybrid=True,
        user_specified_block_size=True,
        mamba_block_size=16,
        mamba_cache_mode="all",
    )
    bsu.update_tpu_block_size_and_slot_config(cfg_user, _FakeBackend)
    assert cfg_user.cache_config.block_size == 64
    assert cfg_user.cache_config.mamba_block_size == 16

    # 3c. source = "backend_min_page_size" (min_page_size=64 > 32, user_specified=False)
    cfg_min_page = _make_bsu_vllm_cfg(
        block_size=16,
        is_hybrid=True,
        user_specified_block_size=False,
    )
    bsu.update_tpu_block_size_and_slot_config(cfg_min_page, _HighMinPageBackend)
    assert cfg_min_page.cache_config.block_size == 64

    # 3d. Error when final FA page is not token-linear
    class _BrokenFinalBackend(_FakeBackend):
        break_linearity_at_final = True

    with pytest.raises(ValueError, match="not token-linear"):
        bsu._derive_tpu_block_slot_config(
            cfg_fit, _BrokenFinalBackend, input_block_size=16
        )

    # 3e. Error when FA physical slot bytes < mamba_layout.required_bytes
    inconsistent_layout = PooledGDNStateLayout(
        ssm_bytes=512,
        conv_bytes=256,
        token_bytes=64000,
        ssm_tokens=8,
        conv_tokens=8,
    )
    monkeypatch.setattr(
        bsu, "_hybrid_mamba_state_layout", lambda *_a, **_kw: inconsistent_layout
    )
    with pytest.raises(ValueError, match="cannot contain pooled GDN state"):
        bsu._derive_tpu_block_slot_config(cfg_fit, _FakeBackend, input_block_size=16)
