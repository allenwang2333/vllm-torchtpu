"""TPU platform plugin for vllm-omni."""

import os

import torch
from vllm.logger import init_logger
from vllm_omni.platforms.interface import (  # type: ignore[missing-import]
    OmniPlatform, OmniPlatformEnum)

from vllm_torchtpu import utils
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

logger = init_logger(__name__)


class OmniTpuPlatform(OmniPlatform, TpuPlatform):
    """TPU implementation of OmniPlatform.

    Inherits all TPU implementations from vLLM's TpuPlatform and adds
    Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.OOT

    dist_backend: str = "tpu_dist"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "tpu"
    _is_hybrid: bool = False

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
        return "vllm_torchtpu.worker.tpu_diffusion_model_runner.TpuDiffusionModelRunner"

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
        pass

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
        return utils.compute_hbm_budget([torch.device("tpu", device_id)],
                                        1.0).total_limit

    @classmethod
    def synchronize(cls) -> None:
        torch.accelerator.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = cls.get_device_memory(device)
        return free

    @classmethod
    def get_device_memory(cls,
                          device: torch.device | None = None
                          ) -> tuple[int, int]:
        budget = utils.compute_hbm_budget([device or torch.device("tpu")], 1.0)
        return budget.available, budget.total_limit

    @classmethod
    def max_memory_reserved(cls, device: torch.device | None = None) -> int:
        return cls.max_memory_allocated(device)

    @classmethod
    def max_memory_allocated(cls, device: torch.device | None = None) -> int:
        return utils.compute_hbm_budget([device or torch.device("tpu")],
                                        1.0).total_used

    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        pass

    @classmethod
    def get_current_memory_usage(cls,
                                 device: torch.device | None = None) -> float:
        return 0.0
