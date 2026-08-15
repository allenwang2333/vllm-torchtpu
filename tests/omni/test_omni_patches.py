import sys
import types
from unittest.mock import MagicMock, patch

import torch

# Trigger logger resolution before sys.modules patches to avoid vllm reload bug
from vllm_torchtpu.logger import init_logger as _test_init_logger
from vllm_torchtpu.omni import apply_omni_ipc_patch
from vllm_torchtpu.omni.executor import MockWorkerThreadProcess
from vllm_torchtpu.omni.model_patch import apply_omni_rope_patch

_test_init_logger(__name__).info("Initializing tests to resolve logger")


def test_apply_omni_ipc_patch():
    mock_ipc = types.ModuleType("vllm_omni.diffusion.ipc")

    def orig_pack_tensor(val):
        return val

    mock_ipc._pack_tensor_if_large = orig_pack_tensor

    mock_omni = types.ModuleType("vllm_omni")
    mock_diffusion = types.ModuleType("vllm_omni.diffusion")

    with patch.dict(
            sys.modules, {
                "vllm_omni": mock_omni,
                "vllm_omni.diffusion": mock_diffusion,
                "vllm_omni.diffusion.ipc": mock_ipc
            }):
        apply_omni_ipc_patch.cache_clear()
        apply_omni_ipc_patch()

        # Verify patched function returns CPU tensor
        gpu_tensor = MagicMock(spec=torch.Tensor)
        cpu_tensor = MagicMock(spec=torch.Tensor)
        gpu_tensor.detach.return_value.cpu.return_value = cpu_tensor

        res = mock_ipc._pack_tensor_if_large(gpu_tensor)
        assert res is cpu_tensor

        # Verify idempotency
        apply_omni_ipc_patch()


def test_mock_worker_thread_process():
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True

    proc = MockWorkerThreadProcess(fake_thread, name="DiffusionWorker-0")
    assert proc.name == "DiffusionWorker-0"
    assert proc.is_alive() is True
    assert isinstance(proc.sentinel, int)

    proc.join()
    assert proc.sentinel is None


def test_apply_omni_rope_patch():
    mock_omni_abs = types.ModuleType(
        "vllm_omni.diffusion.attention.backends.abstract")
    mock_flux_mod = types.ModuleType(
        "vllm_omni.diffusion.models.flux.flux_transformer")

    mock_omni_abs.AttentionImpl = MagicMock()
    mock_flux_mod.FluxPosEmbed = MagicMock()

    modules = {
        "vllm_omni":
        types.ModuleType("vllm_omni"),
        "vllm_omni.diffusion":
        types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.attention":
        types.ModuleType("vllm_omni.diffusion.attention"),
        "vllm_omni.diffusion.attention.backends":
        types.ModuleType("vllm_omni.diffusion.attention.backends"),
        "vllm_omni.diffusion.attention.backends.abstract":
        mock_omni_abs,
        "vllm_omni.diffusion.models":
        types.ModuleType("vllm_omni.diffusion.models"),
        "vllm_omni.diffusion.models.flux":
        types.ModuleType("vllm_omni.diffusion.models.flux"),
        "vllm_omni.diffusion.models.flux.flux_transformer":
        mock_flux_mod,
    }

    with patch.dict(sys.modules, modules), \
         patch("vllm_torchtpu.omni.model_patch.logger"):
        apply_omni_rope_patch.cache_clear()
        apply_omni_rope_patch()

        # Verify idempotency
        patched_forward = mock_flux_mod.FluxPosEmbed.forward
        apply_omni_rope_patch()
        assert mock_flux_mod.FluxPosEmbed.forward is patched_forward


def test_flux_pos_embed_tpu_cpu_offload():
    mock_omni_abs = types.ModuleType(
        "vllm_omni.diffusion.attention.backends.abstract")
    mock_flux_mod = types.ModuleType(
        "vllm_omni.diffusion.models.flux.flux_transformer")

    class FakeFluxPosEmbed:

        def __init__(self):
            pass

        def forward(self, ids: torch.Tensor):
            return torch.zeros(ids.shape[0], 64), torch.zeros(ids.shape[0], 64)

    mock_omni_abs.AttentionImpl = MagicMock()
    mock_flux_mod.FluxPosEmbed = FakeFluxPosEmbed

    modules = {
        "vllm_omni":
        types.ModuleType("vllm_omni"),
        "vllm_omni.diffusion":
        types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.attention":
        types.ModuleType("vllm_omni.diffusion.attention"),
        "vllm_omni.diffusion.attention.backends":
        types.ModuleType("vllm_omni.diffusion.attention.backends"),
        "vllm_omni.diffusion.attention.backends.abstract":
        mock_omni_abs,
        "vllm_omni.diffusion.models":
        types.ModuleType("vllm_omni.diffusion.models"),
        "vllm_omni.diffusion.models.flux":
        types.ModuleType("vllm_omni.diffusion.models.flux"),
        "vllm_omni.diffusion.models.flux.flux_transformer":
        mock_flux_mod,
    }

    with patch.dict(sys.modules, modules), \
         patch("vllm_torchtpu.omni.model_patch.logger"):
        apply_omni_rope_patch()

        embed = FakeFluxPosEmbed()

        # Create rectangular input ids on TPU
        mock_ids = MagicMock(spec=torch.Tensor)
        mock_ids.device = torch.device("cpu")
        mock_ids.device.type = "tpu"
        cpu_ids = torch.zeros(3584, 3)
        mock_ids.detach.return_value.cpu.return_value = cpu_ids

        cos, sin = embed.forward(mock_ids)
        mock_ids.detach.assert_called_once()
        assert cos.shape == (3584, 64)
        assert sin.shape == (3584, 64)
