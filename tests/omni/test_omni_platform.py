import os

import torch

from vllm_torchtpu.omni import register_omni_tpu_platform
from vllm_torchtpu.omni.platform import OmniTpuPlatform


def test_get_diffusion_worker_cls():
    assert OmniTpuPlatform.get_diffusion_worker_cls(
    ) == "vllm_omni.diffusion.worker.diffusion_worker.DiffusionWorker"


def test_get_diffusion_model_runner_cls():
    assert OmniTpuPlatform.get_diffusion_model_runner_cls(
    ) == "vllm_torchtpu.worker.tpu_diffusion_model_runner.TpuDiffusionModelRunner"


def test_get_diffusion_attn_backend_cls():
    assert OmniTpuPlatform.get_diffusion_attn_backend_cls(
        None, 64) == "vllm_torchtpu.omni.attention.TpuSDPABackend"


def test_device_control_env_var(monkeypatch):
    monkeypatch.delenv(OmniTpuPlatform.device_control_env_var, raising=False)

    OmniTpuPlatform.set_device_control_env_var("0,1")
    assert os.environ[OmniTpuPlatform.device_control_env_var] == "0,1"

    OmniTpuPlatform.unset_device_control_env_var()
    assert OmniTpuPlatform.device_control_env_var not in os.environ


def test_accelerator_memory_info(monkeypatch):

    class MockAccelerator:

        @staticmethod
        def get_memory_info(device):
            # Returns (free, total)
            return (2 * 1024**3, 16 * 1024**3)

    monkeypatch.setattr(torch, "accelerator", MockAccelerator(), raising=False)
    assert OmniTpuPlatform.get_device_total_memory() == 16 * 1024**3
    assert OmniTpuPlatform.get_free_memory() == 2 * 1024**3
    assert OmniTpuPlatform.max_memory_allocated() == 14 * 1024**3
    assert OmniTpuPlatform.max_memory_reserved() == 14 * 1024**3


def test_register_omni_tpu_platform():
    result = register_omni_tpu_platform()
    assert result == "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_tpu_sdpa_attention_backend():
    from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl

    assert TpuSDPABackend.get_name() == "TPU_SDPA"
    assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl

    impl = TpuSDPAImpl(num_heads=4, head_size=16, softmax_scale=0.25)
    query = torch.randn(1, 8, 4, 16)
    key = torch.randn(1, 8, 4, 16)
    value = torch.randn(1, 8, 4, 16)

    res = impl.forward(query, key, value)
    assert res.shape == (1, 8, 4, 16)
