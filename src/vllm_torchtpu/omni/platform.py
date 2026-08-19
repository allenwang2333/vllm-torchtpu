import os
from typing import Any

import torch
from vllm.logger import init_logger
from vllm_omni.platforms.interface import (  # type: ignore[missing-import]
    OmniPlatform, OmniPlatformEnum)

from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

logger = init_logger(__name__)


class OmniTpuPlatform(OmniPlatform, TpuPlatform):
    """TPU implementation of OmniPlatform.

    Inherits all TPU implementations from vLLM's TpuPlatform and adds
    Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.OOT

    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "tpu"
    _is_hybrid: bool = False

    def __init__(self) -> None:
        super().__init__()
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        if "TPU_VISIBLE_CHIPS" not in os.environ:
            os.environ["TPU_VISIBLE_CHIPS"] = ""

        try:
            from vllm_omni.diffusion.attention.backends.registry import (  # type: ignore[missing-import]
                DiffusionAttentionBackendEnum, register_diffusion_backend)

            register_diffusion_backend(
                DiffusionAttentionBackendEnum.TORCH_SDPA,
                "vllm_torchtpu.omni.attention.TpuSDPABackend",
            )
            register_diffusion_backend(
                DiffusionAttentionBackendEnum.FLASH_ATTN,
                "vllm_torchtpu.omni.attention.TpuSDPABackend",
            )
        except Exception:
            pass

    @property
    def dist_backend(self) -> str:
        # For single chip / single rank, gloo is lightweight and avoids PJRT cluster init.
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size <= 1:
            return "gloo"
        return "tpu_dist"

    # Explicitly override platform checks to avoid MRO/enum conflicts with
    # OmniPlatform (which checks `self._omni_enum`) and provide has_flash_attn_package
    # which is required by vllm_omni.diffusion.envs.PackagesEnvChecker.
    def is_npu(self) -> bool:
        return False

    def is_xpu(self) -> bool:
        return False

    def is_cuda(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return False

    def is_musa(self) -> bool:
        return False

    def is_out_of_tree(self) -> bool:
        return True

    def has_flash_attn_package(self) -> bool:
        return False

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_torchtpu.platforms.tpu_platform import apply_tpu_patches

        apply_tpu_patches()

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.ar_worker.ARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.generation_worker.GenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    def get_diffusion_worker_cls(cls) -> str:
        return "vllm_omni.diffusion.worker.diffusion_worker.DiffusionWorker"

    @classmethod
    def get_diffusion_model_runner_cls(cls) -> str:
        return "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None = None,
        head_size: int = 0,
        *args,
        **kwargs,
    ) -> str:
        return "vllm_torchtpu.omni.attention.TpuSDPABackend"

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def set_device_control_env_var(cls, devices: str) -> None:
        os.environ[cls.device_control_env_var] = str(devices)

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        if cls.device_control_env_var in os.environ:
            del os.environ[cls.device_control_env_var]

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        return torch.device("tpu")

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        try:
            from torch_tpu._internal.device import _device_ops_backend
            _device_ops_backend._init_runtime_options("tpu")
        except Exception:
            pass
        try:
            torch.accelerator.set_device_index(0)
        except Exception:
            pass
        try:
            _ = torch.empty(1, device=torch.device("tpu"))
            if hasattr(torch, "tpu") and torch.tpu.is_initialized():
                torch.tpu.synchronize()
            logger.info(
                "Successfully initialized TPU device in worker process")
        except Exception as e:
            logger.warning("Failed to eager-initialize TPU device: %s", e)

    @classmethod
    def get_device_count(cls) -> int:
        from vllm_torchtpu.tpu_info import get_num_chips
        return get_num_chips()

    @classmethod
    def get_device_version(cls) -> str | None:
        from torch_tpu._internal.utils import hardware
        return hardware.get_tpu_device_name()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        from torch_tpu._internal.utils import hardware
        return hardware.get_tpu_device_name()

    @classmethod
    def canonical_device_str(cls) -> str:
        return "tpu"

    def uses_host_device_handling(self) -> bool:
        return True

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.get_device_name().lower() == "tpu v6e":
            logger.info(
                "Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.")
            return torch.float8_e5m2
        return torch.float8_e4m3fn

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return TpuPlatform.get_device_total_memory(device_id)

    @classmethod
    def synchronize(cls) -> None:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                torch.tpu.synchronize()
            except Exception:
                pass

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                free, _ = torch.accelerator.get_memory_info(device)
                return free
            except Exception:
                pass
        return 32 * 1024 * 1024 * 1024

    @classmethod
    def get_device_memory(cls,
                          device: torch.device | None = None
                          ) -> tuple[int, int]:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                free, total = torch.accelerator.get_memory_info(device)
                return free, total
            except Exception:
                pass
        return 32 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024

    @classmethod
    def max_memory_reserved(cls, device: torch.device | None = None) -> int:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                return torch.accelerator.max_memory_reserved(device)
            except Exception:
                pass
        return 0

    @classmethod
    def max_memory_allocated(cls, device: torch.device | None = None) -> int:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                return torch.accelerator.max_memory_allocated(device)
            except Exception:
                pass
        return 0

    @classmethod
    def reset_peak_memory_stats(cls,
                                device: torch.device | None = None) -> None:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                torch.accelerator.reset_peak_memory_stats(device)
            except Exception:
                pass

    @classmethod
    def get_current_memory_usage(cls,
                                 device: torch.device | None = None) -> float:
        return 0.0
