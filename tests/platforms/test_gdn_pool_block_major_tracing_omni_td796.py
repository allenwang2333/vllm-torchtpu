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

import enum
import importlib
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import block_major_pool, envs, gdn_pool_layout
from vllm_torchtpu.tracing import options as tracing_options


def _install_fake_vllm_omni(monkeypatch):
    class _OmniPlatform:
        pass

    class _OmniPlatformEnum(enum.Enum):
        OOT = "oot"

    class _SDPAImpl:
        def _forward_impl(self, query, key, value, attn_metadata, mask_mode=None):
            return (query + key + value, attn_metadata, mask_mode)

    class _SDPABackend:
        pass

    class _LTXRuntime:
        def _decode_output(self, *args, **kwargs):
            return SimpleNamespace(output=(torch.ones(2), "raw_audio"))

    mod_omni = types.ModuleType("vllm_omni")
    mod_platforms = types.ModuleType("vllm_omni.platforms")
    mod_iface = types.ModuleType("vllm_omni.platforms.interface")
    mod_iface.OmniPlatform = _OmniPlatform
    mod_iface.OmniPlatformEnum = _OmniPlatformEnum

    mod_diff = types.ModuleType("vllm_omni.diffusion")
    mod_ipc = types.ModuleType("vllm_omni.diffusion.ipc")
    mod_ipc._pack_tensor_if_large = lambda val, d2h_stream=None: (
        val.device.type,
        d2h_stream,
    )

    mod_models = types.ModuleType("vllm_omni.diffusion.models")
    mod_ltx2_pkg = types.ModuleType("vllm_omni.diffusion.models.ltx2")
    mod_ltx2_rt = types.ModuleType("vllm_omni.diffusion.models.ltx2.ltx2_runtime")
    mod_ltx2_rt.LTXRuntime = _LTXRuntime

    mod_attn = types.ModuleType("vllm_omni.diffusion.attention")
    mod_backends = types.ModuleType("vllm_omni.diffusion.attention.backends")
    mod_sdpa = types.ModuleType("vllm_omni.diffusion.attention.backends.sdpa")
    mod_sdpa.SDPABackend = _SDPABackend
    mod_sdpa.SDPAImpl = _SDPAImpl

    for name, mod in (
        ("vllm_omni", mod_omni),
        ("vllm_omni.platforms", mod_platforms),
        ("vllm_omni.platforms.interface", mod_iface),
        ("vllm_omni.diffusion", mod_diff),
        ("vllm_omni.diffusion.ipc", mod_ipc),
        ("vllm_omni.diffusion.models", mod_models),
        ("vllm_omni.diffusion.models.ltx2", mod_ltx2_pkg),
        ("vllm_omni.diffusion.models.ltx2.ltx2_runtime", mod_ltx2_rt),
        ("vllm_omni.diffusion.attention", mod_attn),
        ("vllm_omni.diffusion.attention.backends", mod_backends),
        ("vllm_omni.diffusion.attention.backends.sdpa", mod_sdpa),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    return mod_ipc, mod_ltx2_rt


def test_gdn_pool_layout_all_paths(monkeypatch):
    gdn_pool_layout.derive_pooled_gdn_state_layout.cache_clear()

    assert gdn_pool_layout.pooled_gdn_state_dtypes(
        ("torch.float16", "torch.float32")
    ) == (
        "torch.bfloat16",
        "torch.float32",
    )
    assert gdn_pool_layout.pooled_gdn_state_itemsize("torch.bfloat16") == 2
    assert gdn_pool_layout.pooled_gdn_state_itemsize(torch.float32) == 4
    with pytest.raises(ValueError, match="unsupported pooled GDN state dtype"):
        gdn_pool_layout.pooled_gdn_state_itemsize("torch.int8")

    monkeypatch.setattr(envs, "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", True, raising=False)
    assert gdn_pool_layout.unified_kv_layout_enabled_for_architecture(
        "LlamaForCausalLM"
    )
    monkeypatch.setattr(envs, "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", None, raising=False)
    assert gdn_pool_layout.unified_kv_layout_enabled_for_architecture(
        "KimiLinearForCausalLM"
    )
    assert not gdn_pool_layout.unified_kv_layout_enabled_for_architecture(
        "LlamaForCausalLM"
    )

    assert gdn_pool_layout.pooled_gdn_conv_state_bytes(kernel_size=4, conv_dim=16) == 96
    assert (
        gdn_pool_layout.pooled_gdn_ssm_state_bytes(
            num_v_heads=2, head_k_dim=4, head_v_dim=8
        )
        == 256
    )

    with pytest.raises(ValueError, match="must be positive"):
        gdn_pool_layout.derive_pooled_gdn_state_layout(
            ssm_bytes=0, conv_bytes=64, token_bytes=32
        )

    layout = gdn_pool_layout.derive_pooled_gdn_state_layout(
        ssm_bytes=100, conv_bytes=70, token_bytes=32
    )
    assert layout.ssm_tokens == 4
    assert layout.conv_tokens == 4
    assert layout.required_tokens == 8
    assert layout.required_bytes == 256


def test_block_major_pool_all_paths(monkeypatch):
    import vllm_torchtpu.platforms.tpu_block_size_utils as bsu

    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False, raising=False)
    fake_cfg = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True))
    assert not block_major_pool.unified_block_major_enabled(fake_cfg)

    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", True, raising=False)
    monkeypatch.setattr(bsu, "unified_kv_layout_enabled", lambda cfg: True)
    assert block_major_pool.unified_block_major_enabled(fake_cfg)

    layout = block_major_pool.BlockMajorPoolLayout(
        num_blocks=10, num_pools=3, split=2, pool_index_by_layer={"l0": 0}
    )
    assert layout.rows_per_block == 6

    pool = torch.zeros((4, 6, 8, 16))
    assert block_major_pool.kernel_view(pool).shape == (24, 8, 16)

    ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    assert block_major_pool.flat_manager_ids(ids, 0, 1) is ids
    assert torch.equal(
        block_major_pool.flat_manager_ids(ids, 1, 3),
        torch.tensor([1, 4, 7, 10], dtype=torch.int32),
    )

    assert block_major_pool.flat_kernel_ids(ids, 0, 1, 2) is ids
    assert torch.equal(
        block_major_pool.flat_kernel_ids(ids, 1, 3, 1),
        torch.tensor([1, 4, 7, 10], dtype=torch.int32),
    )
    # For num_pools=3, split=2, pool_index=1:
    # manager = [0, 0, 1, 1], expected = [2, 3, 8, 9]
    assert torch.equal(
        block_major_pool.flat_kernel_ids(ids, 1, 3, 2),
        torch.tensor([2, 3, 8, 9], dtype=torch.int32),
    )


def test_tracing_options_all_paths(monkeypatch):
    # parse_profile_options
    assert tracing_options.parse_profile_options(None) == ({}, {})
    assert tracing_options.parse_profile_options("   ") == ({}, {})
    with pytest.raises(ValueError, match="non-ASCII"):
        tracing_options.parse_profile_options("host_tracer_level:\u2603")
    with pytest.raises(ValueError, match="Invalid profile option format"):
        tracing_options.parse_profile_options("bad_part_no_colon")
    with pytest.raises(ValueError, match="Invalid characters in option value"):
        tracing_options.parse_profile_options("custom_key:bad$val")

    std, adv = tracing_options.parse_profile_options(
        "host_tracer_level:2; ; enable_foo:true; disable_bar:false; mode:trace_v1"
    )
    assert std == {"host_tracer_level": 2}
    assert adv == {
        "enable_foo": True,
        "disable_bar": False,
        "mode": "trace_v1",
    }

    # resolve_profile_dir_and_opts: structured prefix & kwargs types
    pdir1, std1, adv1 = tracing_options.resolve_profile_dir_and_opts(
        "/tmp/base",
        profile_prefix=None,
        profiler_kwargs={
            "profile_prefix": "device_tracer_level:1;custom_adv:true",
            "python_tracer_level": 3,
            "extra_bool": False,
            "extra_str": "clean_val",
        },
    )
    assert pdir1 == "/tmp/base"
    assert std1["device_tracer_level"] == 1
    assert std1["python_tracer_level"] == 3
    assert adv1["custom_adv"] is True
    assert adv1["extra_bool"] is False
    assert adv1["extra_str"] == "clean_val"

    pdir2, _, _ = tracing_options.resolve_profile_dir_and_opts(
        "/tmp/base",
        profile_prefix="sub_run",
        profiler_kwargs={"non_str_prefix": 123},
    )
    assert pdir2 == os.path.join("/tmp/base", "sub_run")

    pdir3, _, _ = tracing_options.resolve_profile_dir_and_opts(
        "/tmp/base",
        profile_prefix=None,
        profiler_kwargs={"profile_prefix": 123},
    )
    assert pdir3 == "/tmp/base"

    pdir4, _, _ = tracing_options.resolve_profile_dir_and_opts("/tmp/base")
    assert pdir4 == "/tmp/base"


def test_omni_patches_attention_and_platform(monkeypatch):
    mod_ipc, mod_ltx2_rt = _install_fake_vllm_omni(monkeypatch)

    from vllm_torchtpu.omni import attention as omni_attn
    from vllm_torchtpu.omni import patches as omni_patches
    from vllm_torchtpu.omni import platform as omni_platform

    importlib.reload(omni_patches)
    importlib.reload(omni_attn)
    importlib.reload(omni_platform)

    # 1. omni/patches.py
    orig_interp = torch.nn.functional.interpolate
    monkeypatch.setattr(
        torch.nn.functional,
        "interpolate",
        lambda x, size=None, scale_factor=None, mode="nearest", *a, **kw: (
            x,
            mode,
        ),
    )
    omni_patches.apply_omni_tpu_patches.cache_clear()
    omni_patches.apply_omni_tpu_patches()

    class _FakeTpuTensor:
        device = SimpleNamespace(type="tpu")

        def cpu(self):
            return SimpleNamespace(
                device=SimpleNamespace(type="cpu"),
                to=lambda dev: ("from_cpu_to", dev.type),
            )

    # Mock orig_interpolate inside _safe_interpolate to return an object with .to(...)
    monkeypatch.setattr(torch.nn.functional, "interpolate", orig_interp)
    omni_patches.apply_omni_tpu_patches.cache_clear()

    def _mock_interp(x, size=None, scale_factor=None, mode="nearest", *a, **kw):
        return SimpleNamespace(
            x=x,
            mode=mode,
            to=lambda dev: ("moved_to", dev.type, mode),
        )

    monkeypatch.setattr(torch.nn.functional, "interpolate", _mock_interp)
    omni_patches.apply_omni_tpu_patches()

    tpu_in = SimpleNamespace(
        device=SimpleNamespace(type="tpu"),
        cpu=lambda: "cpu_tensor",
    )
    assert torch.nn.functional.interpolate(tpu_in, mode="linear") == (
        "moved_to",
        "tpu",
        "linear",
    )
    cpu_in = torch.ones((1, 1, 4))
    assert torch.nn.functional.interpolate(cpu_in, mode="nearest").mode == "nearest"

    omni_patches.apply_omni_model_specific_patches.cache_clear()
    omni_patches.apply_omni_model_specific_patches()
    assert mod_ipc._pack_tensor_if_large(torch.ones(2), d2h_stream="s") == (
        "cpu",
        "s",
    )
    rt = mod_ltx2_rt.LTXRuntime()
    decoded = rt._decode_output()
    assert isinstance(decoded.output[0], torch.Tensor)
    assert decoded.output[1] == "raw_audio"

    # 2. omni/attention.py
    assert omni_attn.TpuSDPABackend.get_name() == "TPU_SDPA"
    impl_cls = omni_attn.TpuSDPABackend.get_impl_cls()
    assert impl_cls is omni_attn.TpuSDPAImpl
    impl = impl_cls()
    q = torch.ones(2)
    assert impl.forward(q, q, q, attn_metadata="meta")[2] == "broadcast_k"

    # 3. omni/platform.py
    monkeypatch.setattr(
        omni_platform.TpuPlatform,
        "__init__",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        omni_platform.TpuPlatform,
        "get_worker_distributed_backend",
        classmethod(lambda cls, world_size: f"backend_{world_size}"),
    )
    monkeypatch.setattr(
        omni_platform.TpuPlatform,
        "_prepare_singlehost_tpu_env",
        classmethod(lambda cls, ws: None),
    )
    monkeypatch.setattr(
        omni_platform.TpuPlatform,
        "device_count",
        classmethod(lambda cls: 4),
    )
    plat = omni_platform.OmniTpuPlatform()
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert plat.dist_backend == "backend_2"
    assert (
        omni_platform.OmniTpuPlatform.get_diffusion_attn_backend_cls("sdpa", 128)
        == "vllm_torchtpu.omni.attention.TpuSDPABackend"
    )
    assert not omni_platform.OmniTpuPlatform.supports_torch_inductor()
    assert not omni_platform.OmniTpuPlatform.supports_diffusion_dense_flash_attention()
    assert not omni_platform.OmniTpuPlatform.supports_float64()
    assert not omni_platform.OmniTpuPlatform.supports_cpu_offload()

    orig_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda d: SimpleNamespace(type="tpu") if d == "tpu" else orig_device(d),
    )
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda idx: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: None)
    omni_platform.OmniTpuPlatform.set_device(orig_device("cpu"))
    assert omni_platform.OmniTpuPlatform.get_torch_device().type == "tpu"
    assert omni_platform.OmniTpuPlatform.get_device_count() == 4
    assert omni_platform.OmniTpuPlatform.get_device_version() is None
    omni_platform.OmniTpuPlatform.synchronize()
    omni_platform.OmniTpuPlatform.empty_cache()

    monkeypatch.setattr(
        omni_platform.hardware, "get_hbm_bytes_per_device", lambda: 64 * 1024**3
    )
    assert omni_platform.OmniTpuPlatform.get_device_total_memory() == 64 * 1024**3
    monkeypatch.setattr(
        omni_platform.hardware, "get_hbm_bytes_per_device", lambda: None
    )
    monkeypatch.setattr(
        omni_platform.hardware, "get_tpu_device_name", lambda: "unknown"
    )
    with pytest.raises(ValueError, match="Could not determine HBM"):
        omni_platform.OmniTpuPlatform.get_device_total_memory()

    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda dev=None: (100, 200)
    )
    monkeypatch.setattr(torch.accelerator, "max_memory_reserved", lambda dev=None: 150)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda dev=None: 120)
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda dev=None: None
    )
    assert omni_platform.OmniTpuPlatform.get_free_memory() == 100
    assert omni_platform.OmniTpuPlatform.get_device_memory() == (100, 200)
    assert omni_platform.OmniTpuPlatform.max_memory_reserved() == 150
    assert omni_platform.OmniTpuPlatform.max_memory_allocated() == 120
    omni_platform.OmniTpuPlatform.reset_peak_memory_stats()
    assert (
        omni_platform.OmniTpuPlatform.get_omni_ar_worker_cls()
        == "vllm_torchtpu.worker.tpu_worker.TPUWorker"
    )
    assert (
        omni_platform.OmniTpuPlatform.get_omni_generation_worker_cls()
        == "vllm_torchtpu.worker.tpu_worker.TPUWorker"
    )
    assert (
        omni_platform.OmniTpuPlatform.get_default_stage_config_path()
        == "vllm_omni/deploy"
    )
