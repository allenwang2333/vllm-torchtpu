# SPDX-License-Identifier: Apache-2.0
"""Comprehensive CPU unit tests for tpu_platform.py (TD7.87)."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.platforms import tpu_platform
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


def test_get_tpu_multihost_topology_all_branches(monkeypatch):
    # 1. Env var with 'x' and without 'x'
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2x4x1")
    assert tpu_platform.get_tpu_multihost_topology(8) == "2,4,1"
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2,4,1")
    assert tpu_platform.get_tpu_multihost_topology(8) == "2,4,1"
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)

    # 2. device_name=None -> calls TpuPlatform.get_device_name()
    monkeypatch.setattr(
        TpuPlatform, "get_device_name", classmethod(lambda cls: "TPU v5e")
    )
    assert tpu_platform.get_tpu_multihost_topology(8) == "2,4,1"

    # 3. device_name=None when TpuPlatform.get_device_name() raises
    def _raise_err(cls):
        raise RuntimeError("no device")

    monkeypatch.setattr(TpuPlatform, "get_device_name", classmethod(_raise_err))
    assert tpu_platform.get_tpu_multihost_topology(8) == "2,2,2,2"

    # 4. Explicit generations: 2D torus, 8i, dual-device (v7/tpu7x), 3D torus
    assert tpu_platform.get_tpu_multihost_topology(4, "TPU v6e") == "2,2,1"
    assert tpu_platform.get_tpu_multihost_topology(4, "TPU v8i") == "4,1,1"
    assert tpu_platform.get_tpu_multihost_topology(16, "TPU7x") == "2,2,2,2"
    assert tpu_platform.get_tpu_multihost_topology(8, "TPU v4") == "2,2,2,2"

    # 5. Unknown world_size -> ValueError
    with pytest.raises(ValueError, match="Cannot find topology"):
        tpu_platform.get_tpu_multihost_topology(999, "TPU v6e")


def test_eager_mode_unwrap_and_api_server_reload_endpoint(monkeypatch):
    # 1. _configure_torchtpu_eager_mode
    from torch_tpu._internal import execution_mode

    orig_mode = execution_mode.eager_mode
    try:
        monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "1")
        tpu_platform._configure_torchtpu_eager_mode()
        monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "0")
        tpu_platform._configure_torchtpu_eager_mode()
        assert execution_mode.eager_mode == execution_mode.EagerMode.DEFER_AND_FUSE
    finally:
        execution_mode.eager_mode = orig_mode

    # 2. _unwrap_dynamic_compile_fns
    def _dummy_wrapped():
        return 42

    _dummy_outer = MagicMock()
    _dummy_outer.__wrapped__ = _dummy_wrapped
    fake_mod_wrapped = SimpleNamespace(fn_a=_dummy_outer)
    fake_mod_plain = SimpleNamespace(fn_b=lambda: 1)

    monkeypatch.setattr(
        tpu_platform,
        "_DYNAMIC_COMPILE_TARGETS",
        [
            ("nonexistent.module.for.td787", "missing_fn"),
            ("fake_wrapped_mod_td787", "fn_a"),
            ("fake_plain_mod_td787", "fn_b"),
        ],
    )
    monkeypatch.setitem(sys.modules, "fake_wrapped_mod_td787", fake_mod_wrapped)
    monkeypatch.setitem(sys.modules, "fake_plain_mod_td787", fake_mod_plain)

    monkeypatch.setattr(tpu_platform, "_dynamic_compile_unwrapped", False)
    tpu_platform._unwrap_dynamic_compile_fns()
    assert fake_mod_wrapped.fn_a is _dummy_wrapped
    # Second call hits early return
    tpu_platform._unwrap_dynamic_compile_fns()

    # 3. _patch_api_server_kernel_reload_endpoint
    monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "0")
    tpu_platform._patch_api_server_kernel_reload_endpoint()

    monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "1")
    monkeypatch.delitem(
        sys.modules, "vllm.entrypoints.openai.api_server", raising=False
    )
    tpu_platform._patch_api_server_kernel_reload_endpoint()

    routes = {}

    class _FakeApp:
        def post(self, path):
            def _decorator(fn):
                routes[path] = fn
                return fn

            return _decorator

    fake_api_server = SimpleNamespace(build_app=lambda *a, **kw: _FakeApp())
    monkeypatch.setitem(
        sys.modules, "vllm.entrypoints.openai.api_server", fake_api_server
    )
    tpu_platform._patch_api_server_kernel_reload_endpoint()
    assert getattr(fake_api_server, "_tpu_kernel_reload_patch", False) is True
    # Calling again when already patched returns early
    tpu_platform._patch_api_server_kernel_reload_endpoint()

    app = fake_api_server.build_app()
    assert "/reload_kernel" in routes
    endpoint = routes["/reload_kernel"]

    drained = []
    rpc_calls = []

    class _ClientWithDrain:
        async def wait_for_requests_to_drain(self):
            drained.append(True)

        async def collective_rpc(self, method, timeout, kwargs):
            rpc_calls.append((method, timeout, kwargs))
            return ["ok"]

    class _ClientWithoutDrain:
        async def collective_rpc(self, method, timeout, kwargs):
            rpc_calls.append((method, timeout, kwargs))
            return ["ok2"]

    class _FakeReq:
        def __init__(self, body_bytes: bytes, client):
            self._body = body_bytes
            self.app = SimpleNamespace(state=SimpleNamespace(engine_client=client))

        async def body(self) -> bytes:
            return self._body

    resp1 = asyncio.run(
        endpoint(_FakeReq(b'{"modules": ["mod_a"]}', _ClientWithDrain()))
    )
    assert resp1.status_code == 200
    assert drained == [True]
    assert rpc_calls[-1] == ("reload_kernels", 1800, {"modules": ["mod_a"]})

    resp2 = asyncio.run(endpoint(_FakeReq(b"", _ClientWithoutDrain())))
    assert resp2.status_code == 200
    assert rpc_calls[-1] == ("reload_kernels", 1800, {"modules": None})
    assert app is not None


def test_phased_profiling_and_kv_connectors(monkeypatch):
    # _validate_phased_profiling_config
    bad_cfg = SimpleNamespace(
        additional_config={"phased_profiling_dir": "/tmp"},
        profiler_config=SimpleNamespace(torch_profiler_dir="/tmp"),
    )
    with pytest.raises(AssertionError, match="phased_profiling_dir"):
        tpu_platform._validate_phased_profiling_config(bad_cfg)

    monkeypatch.setenv("USE_PHASED_PROFILER", "1")
    no_dir_cfg = SimpleNamespace(
        additional_config={},
        profiler_config=SimpleNamespace(torch_profiler_dir=""),
    )
    with pytest.raises(AssertionError, match="USE_PHASED_PROFILER"):
        tpu_platform._validate_phased_profiling_config(no_dir_cfg)

    ok_cfg = SimpleNamespace(
        additional_config={},
        profiler_config=SimpleNamespace(torch_profiler_dir="/tmp/traces"),
    )
    tpu_platform._validate_phased_profiling_config(ok_cfg)

    # _register_tpu_kv_connectors
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    reg_calls = []
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(lambda cls, name, mod, cls_name: reg_calls.append(name)),
    )
    monkeypatch.setattr(tpu_platform, "_tpu_kv_connectors_registered", False)
    tpu_platform._register_tpu_kv_connectors()
    assert len(reg_calls) == 3
    # Second call hits early return
    tpu_platform._register_tpu_kv_connectors()
    assert len(reg_calls) == 3


def test_tpu_platform_device_and_simple_methods(monkeypatch):
    assert TpuPlatform.import_kernels() is None

    # device_id_to_physical_device_id
    tpu_platform._parse_visible_chips.cache_clear()
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "2, 5")
    assert TpuPlatform.device_id_to_physical_device_id(0) == 2
    assert TpuPlatform.device_id_to_physical_device_id(1) == 5
    assert TpuPlatform.device_id_to_physical_device_id(3) == 5

    monkeypatch.setenv("TPU_VISIBLE_CHIPS", " , ")
    monkeypatch.setattr(TpuPlatform, "device_count", classmethod(lambda cls: 4))
    assert TpuPlatform.device_id_to_physical_device_id(6) == 2

    monkeypatch.delenv("TPU_VISIBLE_CHIPS", raising=False)
    assert TpuPlatform.device_id_to_physical_device_id(5) == 1

    # device_count
    import vllm_torchtpu.tpu_info as tpu_info_mod

    monkeypatch.setattr(tpu_info_mod, "get_num_chips", lambda: 8)
    monkeypatch.undo()
    monkeypatch.setattr(tpu_info_mod, "get_num_chips", lambda: 8)
    assert TpuPlatform.device_count() == 8

    # get_worker_distributed_backend
    assert TpuPlatform.get_worker_distributed_backend(1) == "gloo"
    assert TpuPlatform.get_worker_distributed_backend(4) == "tpu_dist"

    # _prepare_singlehost_tpu_env & _get_tpu_topology
    monkeypatch.setenv("WORLD_SIZE", "4")
    TpuPlatform._prepare_singlehost_tpu_env(1)
    assert "WORLD_SIZE" not in tpu_platform.os.environ

    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.delenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", raising=False)
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)
    monkeypatch.setattr(tpu_platform.hardware, "get_tpu_topology", lambda ws: "2,2,1")
    TpuPlatform._prepare_singlehost_tpu_env(4)
    assert "MASTER_PORT" in tpu_platform.os.environ
    assert tpu_platform.os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].count(",") == 3
    assert tpu_platform.os.environ["TORCH_TPU_TOPOLOGY"] == "2,2,1"

    # Re-entry when MASTER_PORT, SLICEBUILDER_ADDRESSES, and TOPOLOGY exist
    TpuPlatform._prepare_singlehost_tpu_env(4)

    monkeypatch.setattr(tpu_platform.hardware, "get_tpu_topology", lambda ws: None)
    with pytest.raises(ValueError, match="No TPU devices found"):
        TpuPlatform._get_tpu_topology(4)

    # get_attn_backend_cls
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    path_mla = TpuPlatform.get_attn_backend_cls(
        AttentionBackendEnum.FLASH_ATTN,
        SimpleNamespace(use_mla=True),
    )
    assert "MLA" in path_mla or "mla" in path_mla.lower()

    path_fallback = TpuPlatform.get_attn_backend_cls(
        AttentionBackendEnum.TRITON_ATTN,
        SimpleNamespace(use_mla=False),
    )
    assert path_fallback == AttentionBackendEnum.FLASH_ATTN.get_path()

    # get_device_name & fp8_dtype
    monkeypatch.setattr(tpu_platform.hardware, "get_tpu_device_name", lambda: "TPU v6e")
    assert TpuPlatform.get_device_name() == "TPU v6e"
    assert TpuPlatform.fp8_dtype() == torch.float8_e5m2

    monkeypatch.setattr(tpu_platform.hardware, "get_tpu_device_name", lambda: "TPU v7")
    assert TpuPlatform.fp8_dtype() == torch.float8_e4m3fn

    with pytest.raises(NotImplementedError):
        TpuPlatform.get_device_total_memory()

    # get_current_memory_usage
    fake_tpu = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setattr(torch, "tpu", fake_tpu, raising=False)
    assert TpuPlatform.get_current_memory_usage() == 0.0
    fake_tpu.is_initialized = lambda: True
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda dev: (100, 500))
    assert TpuPlatform.get_current_memory_usage() == 400.0

    # Simple classmethods
    assert TpuPlatform.is_async_output_supported(False) is False
    assert "PunicaWrapperTPU" in TpuPlatform.get_punica_wrapper()
    mn, mx = TpuPlatform.get_infinity_values(torch.float32)
    assert mn < 0 < mx
    assert TpuPlatform.can_update_inplace() is False
    assert TpuPlatform.get_lora_vocab_padding_size() == 1
    assert TpuPlatform.inference_mode() is True
    assert "TpuCompilerAdaptor" in TpuPlatform.get_compile_backend()
    assert TpuPlatform.is_pin_memory_available() is False
    assert "TpuDeviceCommunicator" in TpuPlatform.get_device_communicator_cls()
    assert TpuPlatform.use_all_gather() is True
    assert TpuPlatform.supports_v1(None) is True  # type: ignore[arg-type]
    assert TpuPlatform.is_kv_cache_dtype_supported("fp8", None) is True  # type: ignore[arg-type]
    assert TpuPlatform.is_kv_cache_dtype_supported("int4", None) is False  # type: ignore[arg-type]
    assert TpuPlatform.use_sync_weight_loader() is False
    assert TpuPlatform.support_hybrid_kv_cache() is True


def test_set_additional_forward_context_and_validate_request():
    # 1. set_additional_forward_context
    assert TpuPlatform.set_additional_forward_context() == {}

    class _FakeTensor:
        def __init__(self, dev_type: str):
            self.device = SimpleNamespace(type=dev_type)

        def to(self, target: str):
            return _FakeTensor(target)

    # Already on tpu -> no .to("tpu")
    dp_meta_tpu = SimpleNamespace(num_tokens_across_dp_cpu=_FakeTensor("tpu"))
    TpuPlatform.set_additional_forward_context(dp_metadata=dp_meta_tpu)
    assert dp_meta_tpu.num_tokens_across_dp_cpu.device.type == "tpu"

    # Mutable CPU dp_metadata
    dp_meta_cpu = SimpleNamespace(num_tokens_across_dp_cpu=_FakeTensor("cpu"))
    TpuPlatform.set_additional_forward_context(dp_metadata=dp_meta_cpu)
    assert dp_meta_cpu.num_tokens_across_dp_cpu.device.type == "tpu"

    # Frozen dataclass CPU dp_metadata
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _FrozenDpMeta:
        num_tokens_across_dp_cpu: object

    frozen_meta = _FrozenDpMeta(num_tokens_across_dp_cpu=_FakeTensor("cpu"))
    TpuPlatform.set_additional_forward_context(dp_metadata=frozen_meta)
    assert frozen_meta.num_tokens_across_dp_cpu.device.type == "tpu"  # type: ignore[attr-defined]

    # 2. validate_request
    from vllm.sampling_params import SamplingParams

    TpuPlatform.validate_request(
        None, SimpleNamespace()
    )  # non-SamplingParams (PoolingParams)
    sp_greedy = SamplingParams(temperature=0.0)
    TpuPlatform.validate_request(None, sp_greedy)

    sp_seed = SamplingParams(temperature=0.8, seed=42)
    with pytest.raises(ValueError, match="per-request seed"):
        TpuPlatform.validate_request(None, sp_seed)

    class _BadSP(SamplingParams):
        @property
        def sampling_type(self):
            return "UNSUPPORTED_TYPE"

    with pytest.raises(ValueError, match="not supported on TPU"):
        TpuPlatform.validate_request(None, _BadSP())


def _make_base_vllm_config():
    from vllm.config import CompilationMode
    from vllm.config.compilation import DynamicShapesType
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

    return SimpleNamespace(
        additional_config={},
        profiler_config=SimpleNamespace(torch_profiler_dir="/tmp"),
        model_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            get_head_size=lambda: 128,
            dtype=torch.bfloat16,
            is_hybrid=False,
        ),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            data_parallel_master_ip="127.0.0.1",
            enable_expert_parallel=False,
            world_size=1,
            world_size_across_dp=1,
            nnodes=1,
            node_rank=0,
            worker_cls="",
            distributed_executor_backend=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=64,
            async_scheduling=False,
            scheduler_cls=None,
            enable_chunked_prefill=True,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="align",
            block_size=16,
            user_specified_block_size=False,
            kv_cache_layout=None,
            get_resolved_kv_cache_layout=lambda: VllmKVCacheLayout.LBNHC,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=[],
            pass_config=SimpleNamespace(
                fuse_norm_quant=True,
                fuse_act_quant=True,
                fuse_attn_quant=True,
                eliminate_noops=True,
            ),
            compile_sizes=None,
            compile_ranges_split_points=[32],
            dynamic_shapes_config=SimpleNamespace(type=DynamicShapesType.BACKED),
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )


def test_check_and_update_config_all_branches(monkeypatch):
    from vllm.config import CompilationMode
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

    monkeypatch.setattr(tpu_platform.patch_registry, "apply", lambda *a, **kw: None)
    monkeypatch.setattr(tpu_platform, "_register_tpu_kv_connectors", lambda: None)
    monkeypatch.setattr(
        tpu_platform, "validate_pipeline_parallel_config", lambda cfg: None
    )
    monkeypatch.setattr(
        TpuPlatform, "_prepare_singlehost_tpu_env", classmethod(lambda cls, ws: None)
    )
    monkeypatch.setattr(
        tpu_platform.PcpStaticSupportValidator,
        "validate_platform_config",
        classmethod(
            lambda cls, cfg, kv_cache_layout: SimpleNamespace(pcp_size=1, enabled=False)
        ),
    )

    # 1. USE_MOE_EP_KERNEL error
    cfg = _make_base_vllm_config()
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    with pytest.raises(ValueError, match="USE_MOE_EP_KERNEL"):
        TpuPlatform.check_and_update_config(cfg)
    monkeypatch.delenv("USE_MOE_EP_KERNEL", raising=False)

    # 2. VLLM_TPU_USING_PATHWAYS error
    monkeypatch.setattr(tpu_platform.vllm_envs, "VLLM_TPU_USING_PATHWAYS", True)
    with pytest.raises(NotImplementedError, match="Pathways"):
        TpuPlatform.check_and_update_config(cfg)
    monkeypatch.setattr(tpu_platform.vllm_envs, "VLLM_TPU_USING_PATHWAYS", False)

    # 3. DeepseekV4ForCausalLM + CUSTOM + NONE + kernel iter + compile_sizes + fp16
    cfg = _make_base_vllm_config()
    cfg.model_config.architectures = ["DeepseekV4ForCausalLM"]
    cfg.attention_config.backend = AttentionBackendEnum.CUSTOM
    cfg.compilation_config.mode = CompilationMode.NONE
    cfg.compilation_config.compile_sizes = [32, 1]
    cfg.model_config.dtype = torch.float16
    monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "1")
    monkeypatch.setattr(
        tpu_platform.PcpStaticSupportValidator,
        "validate_platform_config",
        classmethod(
            lambda cls, cfg, kv_cache_layout: SimpleNamespace(pcp_size=2, enabled=True)
        ),
    )
    TpuPlatform.check_and_update_config(cfg)
    assert len(cfg.compilation_config.splitting_ops) == 128
    assert cfg.model_config.dtype == torch.bfloat16
    monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "0")

    # 4. LBHNC layout with head_size == 64 -> NotImplementedError
    import vllm.v1.attention.backends.utils as attn_utils

    cfg_hd64 = _make_base_vllm_config()
    cfg_hd64.model_config.get_head_size = lambda: 64
    monkeypatch.setattr(
        attn_utils,
        "resolve_kv_cache_layout",
        lambda c, s: VllmKVCacheLayout.LBHNC,
    )
    with pytest.raises(NotImplementedError, match="head_dim==64"):
        TpuPlatform.check_and_update_config(cfg_hd64)
    monkeypatch.undo()

    # Re-apply common stubs after undo
    monkeypatch.setattr(tpu_platform.patch_registry, "apply", lambda *a, **kw: None)
    monkeypatch.setattr(tpu_platform, "_register_tpu_kv_connectors", lambda: None)
    monkeypatch.setattr(
        tpu_platform, "validate_pipeline_parallel_config", lambda cfg: None
    )
    monkeypatch.setattr(
        TpuPlatform, "_prepare_singlehost_tpu_env", classmethod(lambda cls, ws: None)
    )
    monkeypatch.setattr(
        tpu_platform.PcpStaticSupportValidator,
        "validate_platform_config",
        classmethod(
            lambda cls, cfg, kv_cache_layout: SimpleNamespace(pcp_size=1, enabled=False)
        ),
    )

    # 5. pipeline_parallel_size > 1
    import vllm_torchtpu.distributed.pp_push as pp_push_mod

    pp_called = []
    monkeypatch.setattr(
        pp_push_mod, "patch_executor_for_pp_wave", lambda c: pp_called.append(True)
    )
    monkeypatch.setattr(
        tpu_platform, "_configure_pipeline_chunks", lambda c: pp_called.append(True)
    )
    cfg_pp = _make_base_vllm_config()
    cfg_pp.parallel_config.pipeline_parallel_size = 2
    TpuPlatform.check_and_update_config(cfg_pp)
    assert len(pp_called) == 2

    # 6. Speculative config + async_scheduling: non-eagle raises, eagle passes
    cfg_spec = _make_base_vllm_config()
    cfg_spec.scheduler_config.async_scheduling = True
    cfg_spec.speculative_config = SimpleNamespace(
        method="ngram", use_eagle=lambda: False
    )
    with pytest.raises(NotImplementedError, match="Async scheduling"):
        TpuPlatform.check_and_update_config(cfg_spec)
    cfg_spec.speculative_config = SimpleNamespace(
        method="eagle", use_eagle=lambda: True
    )
    TpuPlatform.check_and_update_config(cfg_spec)

    # 7. Hybrid prefix caching checks
    cfg_hyb = _make_base_vllm_config()
    cfg_hyb.model_config.is_hybrid = True
    cfg_hyb.cache_config.enable_prefix_caching = True
    cfg_hyb.cache_config.mamba_cache_mode = "none"
    with pytest.raises(NotImplementedError, match="mamba_cache_mode='align'"):
        TpuPlatform.check_and_update_config(cfg_hyb)

    cfg_hyb.cache_config.mamba_cache_mode = "align"
    monkeypatch.setattr(tpu_platform, "unified_kv_layout_enabled", lambda c: False)
    with pytest.raises(NotImplementedError, match="unified KV pool"):
        TpuPlatform.check_and_update_config(cfg_hyb)

    monkeypatch.setattr(tpu_platform, "unified_kv_layout_enabled", lambda c: True)
    TpuPlatform.check_and_update_config(cfg_hyb)

    # 8. Multihost backends:
    # 8a. nnodes > 1 with mp backend
    import vllm_torchtpu.distributed.tpu_mp_multihost as mp_mh_mod

    mh_called = []
    monkeypatch.setattr(
        mp_mh_mod, "prepare_mp_multihost_env", lambda pc: mh_called.append(True)
    )
    cfg_nnodes = _make_base_vllm_config()
    cfg_nnodes.parallel_config.nnodes = 2
    TpuPlatform.check_and_update_config(cfg_nnodes)
    assert mh_called == [True]

    # 8b. Single-host DP > 1 with EP=True/False and DP_SCHED_ENABLED
    cfg_dp = _make_base_vllm_config()
    cfg_dp.parallel_config.data_parallel_size = 2
    cfg_dp.parallel_config.enable_expert_parallel = True
    monkeypatch.setattr(mp_mh_mod, "prepare_mp_multihost_dp_env", lambda pc, ws: True)
    monkeypatch.setenv("DP_SCHED_ENABLED", "1")
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", True)
    TpuPlatform.check_and_update_config(cfg_dp)
    assert (
        cfg_dp.scheduler_config.scheduler_cls
        == "vllm_torchtpu.core.tpu_scheduler.TpuDpScheduler"
    )
    # Re-run when scheduler_cls is already set and enable_expert_parallel is False
    cfg_dp.parallel_config.enable_expert_parallel = False
    monkeypatch.setattr(mp_mh_mod, "prepare_mp_multihost_dp_env", lambda pc, ws: False)
    TpuPlatform.check_and_update_config(cfg_dp)
    monkeypatch.setenv("DP_SCHED_ENABLED", "0")
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", False)
    TpuPlatform.device_control_env_var = "TPU_VISIBLE_CHIPS"

    # 8c. multihost_backend == "ray"
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    cfg_ray = _make_base_vllm_config()
    cfg_ray.parallel_config.data_parallel_size = 2
    monkeypatch.setattr(
        tpu_platform.PcpStaticSupportValidator,
        "validate_platform_config",
        classmethod(
            lambda cls, cfg, kv_cache_layout: SimpleNamespace(pcp_size=2, enabled=False)
        ),
    )
    with pytest.raises(NotImplementedError, match="Prefill context parallelism"):
        TpuPlatform.check_and_update_config(cfg_ray)

    monkeypatch.setattr(
        tpu_platform.PcpStaticSupportValidator,
        "validate_platform_config",
        classmethod(
            lambda cls, cfg, kv_cache_layout: SimpleNamespace(pcp_size=1, enabled=False)
        ),
    )
    monkeypatch.setattr(
        tpu_platform.vllm_envs, "VLLM_USE_RAY_V2_EXECUTOR_BACKEND", False
    )
    with pytest.raises(NotImplementedError, match="Ray V2"):
        TpuPlatform.check_and_update_config(cfg_ray)

    cfg_ray.parallel_config.data_parallel_size = 1
    TpuPlatform.check_and_update_config(cfg_ray)

    monkeypatch.setattr(
        tpu_platform.vllm_envs, "VLLM_USE_RAY_V2_EXECUTOR_BACKEND", True
    )
    cfg_ray.parallel_config.data_parallel_size = 2
    TpuPlatform.check_and_update_config(cfg_ray)

    # 8d. Unknown multihost_backend
    monkeypatch.setitem(
        tpu_platform.envs.__dict__, "TPU_MULTIHOST_BACKEND", "custom_unknown"
    )
    cfg_unk = _make_base_vllm_config()
    TpuPlatform.check_and_update_config(cfg_unk)
    assert cfg_unk.parallel_config.distributed_executor_backend == "uni"
    tpu_platform.envs.__dict__.pop("TPU_MULTIHOST_BACKEND", None)
    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)

    # 9. kv_transfer_config hybrid offloading checks
    cfg_kv = _make_base_vllm_config()
    cfg_kv.cache_config.kv_cache_layout = VllmKVCacheLayout.LBNHC
    cfg_kv.model_config.is_hybrid = True
    cfg_kv.kv_transfer_config = SimpleNamespace(
        kv_connector="TPURaidenOffloadingConnector",
        kv_connector_module_path=None,
    )
    monkeypatch.setattr(tpu_platform, "unified_kv_layout_enabled", lambda c: False)
    with pytest.raises(ValueError, match="CPU offloading"):
        TpuPlatform.check_and_update_config(cfg_kv)

    monkeypatch.setattr(tpu_platform, "unified_kv_layout_enabled", lambda c: True)
    TpuPlatform.check_and_update_config(cfg_kv)


def test_update_block_size_pipeline_chunks_and_token_paddings(monkeypatch):
    # 1. update_block_size_for_backend
    # 1a. DeepseekV41ForCausalLM with user_specified_block_size False vs True
    cfg_v41 = _make_base_vllm_config()
    cfg_v41.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
    cfg_v41.cache_config.block_size = 64
    cfg_v41.cache_config.user_specified_block_size = False
    TpuPlatform.update_block_size_for_backend(cfg_v41)
    assert cfg_v41.cache_config.block_size == 16

    cfg_v41.cache_config.block_size = 64
    cfg_v41.cache_config.user_specified_block_size = True
    TpuPlatform.update_block_size_for_backend(cfg_v41)
    assert cfg_v41.cache_config.block_size == 64

    # 1b. _find_non_ssm_backend returns None
    monkeypatch.setattr(
        TpuPlatform, "_find_non_ssm_backend", classmethod(lambda cls, c: None)
    )
    cfg_none = _make_base_vllm_config()
    TpuPlatform.update_block_size_for_backend(cfg_none)

    # 1c. DeepseekV4ForCausalLM branch + min_page_size > block_size
    from contextlib import nullcontext

    import vllm.config.vllm as vllm_cfg_mod

    from vllm_torchtpu.layers.adapter import attention as attn_mod

    monkeypatch.setattr(
        vllm_cfg_mod, "set_current_vllm_config", lambda c: nullcontext()
    )
    fake_backend = SimpleNamespace(
        get_page_size=lambda c: 32,
        get_preferred_block_size=lambda d: d,
        get_min_page_size=lambda c: 128,
    )
    monkeypatch.setattr(
        TpuPlatform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, c: fake_backend),
    )
    monkeypatch.setattr(
        attn_mod.PallasMLAttentionBackend,
        "get_page_size",
        staticmethod(lambda c: 64),
    )
    monkeypatch.setattr(tpu_platform, "unified_kv_layout_enabled", lambda c: True)
    monkeypatch.setattr(
        tpu_platform, "update_tpu_block_size_and_slot_config", lambda c, b: None
    )

    cfg_v4 = _make_base_vllm_config()
    cfg_v4.model_config.hf_config.architectures = ["DeepseekV4ForCausalLM"]
    TpuPlatform.update_block_size_for_backend(cfg_v4)
    assert cfg_v4.cache_config.block_size == 128

    # 1d. Standard non-hybrid model with min_page_size <= block_size
    fake_backend_small_min = SimpleNamespace(
        get_page_size=lambda c: 64,
        get_preferred_block_size=lambda d: d,
        get_min_page_size=lambda c: 32,
    )
    monkeypatch.setattr(
        TpuPlatform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, c: fake_backend_small_min),
    )
    monkeypatch.setattr(tpu_platform, "unified_kv_layout_enabled", lambda c: False)
    cfg_std = _make_base_vllm_config()
    TpuPlatform.update_block_size_for_backend(cfg_std)
    assert cfg_std.cache_config.block_size == 64

    # 1e. Hybrid model with user_specified_block_size=True
    cfg_hyb = _make_base_vllm_config()
    cfg_hyb.model_config.is_hybrid = True
    cfg_hyb.cache_config.user_specified_block_size = True
    cfg_hyb.cache_config.block_size = 64
    TpuPlatform.update_block_size_for_backend(cfg_hyb)
    assert cfg_hyb.cache_config.block_size == 64

    # 2. _configure_pipeline_chunks
    import vllm_torchtpu.core.pp_chunk_scheduler as pp_chunk_mod

    patched_pp = []
    monkeypatch.setattr(
        pp_chunk_mod,
        "patch_engine_core_for_pp_chunks",
        lambda c: patched_pp.append(True),
    )

    cfg_pp = _make_base_vllm_config()
    monkeypatch.setenv("TPU_PP_DYNAMIC_CHUNKS", "0")
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_DYNAMIC_CHUNKS", False)
    tpu_platform._configure_pipeline_chunks(cfg_pp)

    monkeypatch.setenv("TPU_PP_DYNAMIC_CHUNKS", "1")
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_DYNAMIC_CHUNKS", True)
    monkeypatch.setenv("DP_SCHED_ENABLED", "1")
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", True)
    cfg_pp.parallel_config.data_parallel_size = 2
    tpu_platform._configure_pipeline_chunks(cfg_pp)
    monkeypatch.setenv("DP_SCHED_ENABLED", "0")
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", False)
    cfg_pp.parallel_config.data_parallel_size = 1

    cfg_pp.scheduler_config.scheduler_cls = "custom.Scheduler"
    tpu_platform._configure_pipeline_chunks(cfg_pp)
    cfg_pp.scheduler_config.scheduler_cls = None

    cfg_pp.scheduler_config.enable_chunked_prefill = False
    tpu_platform._configure_pipeline_chunks(cfg_pp)
    cfg_pp.scheduler_config.enable_chunked_prefill = True

    monkeypatch.setenv("TPU_PP_CHUNK_SLACK", "-0.5")
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_CHUNK_SLACK", -0.5)
    with pytest.raises(ValueError, match="must not be negative"):
        tpu_platform._configure_pipeline_chunks(cfg_pp)

    monkeypatch.setenv("TPU_PP_CHUNK_SLACK", "0.1")
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_CHUNK_SLACK", 0.1)
    tpu_platform._configure_pipeline_chunks(cfg_pp)
    assert cfg_pp.scheduler_config.scheduler_cls == pp_chunk_mod.SCHEDULER_CLS
    assert patched_pp == [True]

    # 3. _get_token_paddings with extra_bucket_sizes
    paddings = tpu_platform._get_token_paddings(16, 64, extra_bucket_sizes=[24, 96])
    assert paddings == [16, 24, 32, 64]


def test_find_non_ssm_backend_compatibility(monkeypatch):
    """Test dual-compatibility across v0.30 and v0.31."""
    from unittest.mock import MagicMock

    from vllm.platforms import Platform

    cfg = MagicMock()

    # Case 1: v0.31 environment where Platform defines _find_non_ssm_backends
    backend_a = MagicMock(name="backend_a")
    backend_b = MagicMock(name="backend_b")
    monkeypatch.setattr(
        Platform,
        "_find_non_ssm_backends",
        classmethod(lambda cls, c: [backend_a, backend_b]),
        raising=False,
    )
    assert TpuPlatform._find_non_ssm_backend(cfg) is backend_a
    assert TpuPlatform._find_non_ssm_backends(cfg) == [backend_a, backend_b]

    # Case 2: v0.30 environment where Platform defines only _find_non_ssm_backend
    if hasattr(Platform, "_find_non_ssm_backends"):
        monkeypatch.delattr(Platform, "_find_non_ssm_backends")
    monkeypatch.setattr(
        Platform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, c: backend_b),
        raising=False,
    )
    assert TpuPlatform._find_non_ssm_backend(cfg) is backend_b
    assert TpuPlatform._find_non_ssm_backends(cfg) == [backend_b]

    # Case 3: Neither defined on superclass
    if hasattr(Platform, "_find_non_ssm_backend"):
        monkeypatch.delattr(Platform, "_find_non_ssm_backend")
    assert TpuPlatform._find_non_ssm_backend(cfg) is None
    assert TpuPlatform._find_non_ssm_backends(cfg) == []
