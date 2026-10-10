# SPDX-License-Identifier: Apache-2.0
import os
import sys
from unittest.mock import patch

import pytest
import torch

from .mock_omni import vllm_omni_mock  # isort: split

import vllm_omni
from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

from vllm_torchtpu.omni import register_omni_tpu_platform
from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl
from vllm_torchtpu.omni.patches import (
    apply_omni_model_specific_patches,
    apply_omni_tpu_patches,
)
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [pytest.mark.cpu_test, pytest.mark.omni]


@pytest.fixture(scope="module", autouse=True)
def cleanup_mock_omni():
    yield
    for mod_name in list(sys.modules.keys()):
        if mod_name == "vllm_omni" or mod_name.startswith("vllm_omni."):
            del sys.modules[mod_name]


def test_register_omni_tpu_platform(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 0)
    assert register_omni_tpu_platform() is None

    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 8)
    assert register_omni_tpu_platform() == "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_platform_attributes(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    platform = OmniTpuPlatform()
    assert platform.device_name == "tpu"
    assert platform.device_type == "tpu"
    assert platform.dist_backend == "gloo"
    assert platform.supports_torch_inductor() is False
    assert platform.supports_diffusion_dense_flash_attention() is False
    assert (
        platform.get_diffusion_attn_backend_cls(None, 64)
        == "vllm_torchtpu.omni.attention.TpuSDPABackend"
    )
    with patch("torch.device") as mock_dev:
        platform.get_torch_device()
        mock_dev.assert_called_once_with("tpu")


def test_device_control_env_var():
    assert OmniTpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


def test_memory_methods(monkeypatch):
    class MockAccelerator:
        @staticmethod
        def get_memory_info(device=None):
            return (16 * 1024**3, 32 * 1024**3)

        @staticmethod
        def max_memory_reserved(device=None):
            return 16 * 1024**3

        @staticmethod
        def max_memory_allocated(device=None):
            return 16 * 1024**3

    monkeypatch.setattr(torch, "accelerator", MockAccelerator(), raising=False)
    with patch(
        "torch_tpu._internal.utils.hardware.get_hbm_bytes_per_device",
        return_value=32 * 1024**3,
    ):
        assert OmniTpuPlatform.get_device_total_memory() == 32 * 1024**3
        assert OmniTpuPlatform.get_free_memory() == 16 * 1024**3
        assert OmniTpuPlatform.get_device_memory() == (16 * 1024**3, 32 * 1024**3)
        assert OmniTpuPlatform.max_memory_reserved() == 16 * 1024**3
        assert OmniTpuPlatform.max_memory_allocated() == 16 * 1024**3


def test_tpu_sdpa_attention_backend():
    assert TpuSDPABackend.get_name() == "TPU_SDPA"
    assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
    assert TpuSDPABackend.supports_attention_mask() is True

    impl = TpuSDPAImpl(num_heads=4, head_size=16, softmax_scale=0.25)
    query = torch.randn(2, 8, 4, 16)
    key = torch.randn(2, 8, 4, 16)
    value = torch.randn(2, 8, 4, 16)

    res = impl.forward(query, key, value)
    assert res.shape == (2, 8, 4, 16)

    # GQA
    impl_gqa = TpuSDPAImpl(
        num_heads=8, head_size=16, softmax_scale=0.25, num_kv_heads=2
    )
    query_gqa = torch.randn(1, 16, 8, 16)
    key_gqa = torch.randn(1, 16, 2, 16)
    value_gqa = torch.randn(1, 16, 2, 16)
    res_gqa = impl_gqa.forward(query_gqa, key_gqa, value_gqa)
    assert res_gqa.shape == (1, 16, 8, 16)


def test_vllm_omni_is_mocked():
    assert sys.modules["vllm_omni"] is vllm_omni_mock
    assert not hasattr(vllm_omni, "__file__") or vllm_omni.__file__ is None


def test_ltx2_safe_decode_output():
    apply_omni_model_specific_patches()

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    runtime = ltx2_runtime.LTXRuntime()

    decode_kwargs = dict(
        latents=torch.randn(1, 128, 4, 4),
        audio_latents=torch.randn(1, 8, 16),
        output_type="pt",
        connector_prompt_embeds=torch.randn(1, 10, 64),
        generator=None,
        device=torch.device("cpu"),
        decode_timestep=0.0,
        decode_noise_scale=None,
        prompt_batch_size=1,
    )

    # Case 1: Video and audio are both Tensors (verify .cpu() is invoked)
    v = torch.randn(1, 3, 16, 16).as_subclass(TrackedTensor)
    a = torch.randn(1, 100).as_subclass(TrackedTensor)
    runtime.output = (v, a)
    res = runtime._decode_output(**decode_kwargs)
    assert v.cpu_called is True
    assert a.cpu_called is True
    assert res.output[0].device.type == "cpu"
    assert res.output[1].device.type == "cpu"

    # Case 2: Video is Tensor, audio is None
    v = torch.randn(1, 3, 16, 16).as_subclass(TrackedTensor)
    runtime.output = (v, None)
    res = runtime._decode_output(**decode_kwargs)
    assert v.cpu_called is True
    assert res.output[0].device.type == "cpu"
    assert res.output[1] is None

    # Case 3: Both are None (verify non-tensor outputs pass through safely)
    runtime.output = (None, None)
    res = runtime._decode_output(**decode_kwargs)
    assert res.output == (None, None)


def test_safe_pack_tensor_if_large():
    from unittest.mock import MagicMock

    from vllm_omni.diffusion import ipc

    from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

    # Provide an original pack function to mock_omni's ipc
    orig_mock = MagicMock(
        side_effect=lambda tensor, d2h_stream=None: {
            "packed": True,
            "device": getattr(tensor, "device", None).type
            if hasattr(tensor, "device")
            else "unknown",
        }
    )
    ipc._pack_tensor_if_large = orig_mock

    apply_omni_model_specific_patches.cache_clear()
    apply_omni_model_specific_patches()

    class MockDeviceTensor:
        def __init__(self, device_type="tpu"):
            self.device = MagicMock()
            self.device.type = device_type
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return torch.empty(1, device="cpu")

    t = MockDeviceTensor("tpu")
    res = ipc._pack_tensor_if_large(t)
    assert t.cpu_called is True
    assert res["device"] == "cpu"


def test_omni_tpu_patches_configuration(monkeypatch):
    # Call the undecorated body so the run-once cache is left alone, and
    # restore the process-global state it edits (env var, F.interpolate).
    orig_interpolate = torch.nn.functional.interpolate
    monkeypatch.setattr(torch.nn.functional, "interpolate", orig_interpolate)
    with patch.dict(os.environ):
        os.environ.pop("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", None)
        apply_omni_tpu_patches.__wrapped__()
        assert os.environ["VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT"] == "1800.0"

    patched_interp = torch.nn.functional.interpolate
    x_cpu = torch.ones(1, 2, 4)
    out_nearest = patched_interp(x_cpu, size=8, mode="nearest")
    assert out_nearest.shape == (1, 2, 8)

    class _FakeTpuInput:
        device = type("D", (), {"type": "tpu"})()

        def cpu(self):
            return x_cpu

    monkeypatch.setattr(torch.Tensor, "to", lambda self, dev, *a, **kw: self)
    out_linear = patched_interp(_FakeTpuInput(), size=8, mode="linear")
    assert out_linear.shape == (1, 2, 8)


def test_omni_platform_remaining_methods(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert OmniTpuPlatform.supports_float64() is False
    assert OmniTpuPlatform.supports_cpu_offload() is False
    assert OmniTpuPlatform.get_device_version() is None
    assert OmniTpuPlatform.empty_cache() is None
    assert (
        OmniTpuPlatform.get_omni_ar_worker_cls()
        == "vllm_torchtpu.worker.tpu_worker.TPUWorker"
    )
    assert (
        OmniTpuPlatform.get_omni_generation_worker_cls()
        == "vllm_torchtpu.worker.tpu_worker.TPUWorker"
    )
    assert OmniTpuPlatform.get_default_stage_config_path() == "vllm_omni/deploy"

    monkeypatch.setattr(OmniTpuPlatform, "device_count", classmethod(lambda cls: 4))
    assert OmniTpuPlatform.get_device_count() == 4

    prep_calls = []
    sync_calls = []
    reset_calls = []
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        classmethod(lambda cls, ws: prep_calls.append(ws)),
    )
    monkeypatch.setattr(
        torch,
        "accelerator",
        type(
            "Acc",
            (),
            {
                "set_device_index": staticmethod(
                    lambda idx: prep_calls.append(("idx", idx))
                ),
                "synchronize": staticmethod(lambda: sync_calls.append(True)),
                "reset_peak_memory_stats": staticmethod(
                    lambda dev=None: reset_calls.append(dev)
                ),
            },
        )(),
        raising=False,
    )
    cpu_dev = torch.device("cpu")
    monkeypatch.setattr(torch, "device", lambda *_a, **_kw: cpu_dev)
    monkeypatch.setattr(torch, "empty", lambda *_a, **_kw: None)
    OmniTpuPlatform.set_device(cpu_dev)
    assert prep_calls == [1, ("idx", 0)]
    OmniTpuPlatform.synchronize()
    assert len(sync_calls) == 2
    OmniTpuPlatform.reset_peak_memory_stats()
    assert reset_calls == [None]

    with (
        patch(
            "torch_tpu._internal.utils.hardware.get_hbm_bytes_per_device",
            return_value=None,
        ),
        patch(
            "torch_tpu._internal.utils.hardware.get_tpu_device_name",
            return_value="TPUv7",
        ),
        pytest.raises(ValueError, match="Could not determine HBM memory"),
    ):
        OmniTpuPlatform.get_device_total_memory()
