# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni platform implementation for Google Cloud TPU."""

from __future__ import annotations

import os
import shutil
from typing import Any

import torch
from vllm.platforms.interface import PlatformEnum

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

logger = init_logger(__name__)


def _patch_shm_broadcast_size() -> None:
    """Adjust MessageQueue default buffer size to fit within K8s container /dev/shm (64MB)."""
    try:
        from vllm.distributed.device_communicators import shm_broadcast

        original_init = shm_broadcast.MessageQueue.__init__

        def patched_init(
            self,
            n_reader,
            n_local_reader,
            local_reader_ranks=None,
            max_chunk_bytes=1024 * 1024 * 4,
            max_chunks=4,
            connect_ip=None,
        ):
            # Scale down buffer sizes if container /dev/shm is constrained (e.g. 64MB)
            free_bytes = 64 * 1024 * 1024
            if os.path.isdir("/dev/shm"):
                try:
                    free_bytes = shutil.disk_usage("/dev/shm").free
                except Exception:
                    pass
            total_needed = max_chunk_bytes * max_chunks
            if total_needed > free_bytes * 0.7:
                max_chunks = min(max_chunks, 4)
                max_chunk_bytes = min(
                    max_chunk_bytes,
                    max(1024 * 1024, int(free_bytes * 0.35 / max_chunks)),
                )
            original_init(
                self,
                n_reader,
                n_local_reader,
                local_reader_ranks=local_reader_ranks,
                max_chunk_bytes=max_chunk_bytes,
                max_chunks=max_chunks,
                connect_ip=connect_ip,
            )

        shm_broadcast.MessageQueue.__init__ = patched_init
    except Exception:
        pass


class TpuOmniPlatform(OmniPlatform, TpuPlatform):
    """TPU platform implementation for vLLM-Omni."""

    _omni_enum = OmniPlatformEnum.OOT
    _enum = PlatformEnum.OOT
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "TPU"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "tpu"

    @property
    def dist_backend(self) -> str:
        # For single chip / single rank, gloo is lightweight and avoids PJRT cluster init.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            return "gloo"
        return "tpu_dist"

    def __init__(self) -> None:
        super().__init__()
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        if "TPU_VISIBLE_CHIPS" not in os.environ:
            os.environ["TPU_VISIBLE_CHIPS"] = ""
        _patch_shm_broadcast_size()

        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.TORCH_SDPA,
            "vllm_torchtpu.diffusion.attention.backends.sdpa.TpuSDPABackend",
        )
        register_diffusion_backend(
            DiffusionAttentionBackendEnum.FLASH_ATTN,
            "vllm_torchtpu.diffusion.attention.backends.sdpa.TpuSDPABackend",
        )

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        if cls.device_control_env_var is not None:
            os.environ[cls.device_control_env_var] = str(devices) if devices is not None else ""

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        if cls.device_control_env_var is not None:
            os.environ[cls.device_control_env_var] = ""

    @classmethod
    def check_and_update_config(cls, vllm_config: Any) -> None:
        if getattr(vllm_config, "model_config", None) is None:
            # Diffusion worker VllmConfig — skip LLM-specific model checks
            return
        super().check_and_update_config(vllm_config)

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_torchtpu.platforms.tpu_platform import apply_tpu_patches

        apply_tpu_patches()
        world_size = getattr(getattr(vllm_config, "parallel_config", None), "world_size", 1)
        if world_size <= 1:
            os.environ.pop("WORLD_SIZE", None)

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_torchtpu.worker.tpu_worker.TPUWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_torchtpu.worker.tpu_worker.TPUWorker"

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
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        return "vllm_torchtpu.diffusion.attention.backends.sdpa.TpuSDPABackend"

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        return torch.device("tpu")

    @classmethod
    def get_device_count(cls) -> int:
        return cls.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size <= 1:
            os.environ.pop("WORLD_SIZE", None)
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
            # Eagerly initialize PJRT client on main thread so that all future
            # tensor allocations and copies find the PJRT client initialized.
            _ = torch.empty(1, device=torch.device("tpu"))
            if hasattr(torch, "tpu") and torch.tpu.is_initialized():
                torch.tpu.synchronize()
            logger.info("Successfully initialized TPU device in worker process")
        except Exception as e:
            logger.warning("Failed to eager-initialize TPU device: %s", e)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.manual_seed(seed)

    @classmethod
    def synchronize(cls) -> None:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                torch.tpu.synchronize()
            except Exception:
                pass

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def record_device_event(cls):
        return None

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
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                free, total = torch.accelerator.get_memory_info(device)
                return free, total
            except Exception:
                pass
        return 32 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return TpuPlatform.get_device_total_memory(device_id)

    @classmethod
    def max_memory_allocated(cls, device: torch.device | None = None) -> int:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                return torch.accelerator.max_memory_allocated(device)
            except Exception:
                pass
        return 0

    @classmethod
    def max_memory_reserved(cls, device: torch.device | None = None) -> int:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                return torch.accelerator.max_memory_reserved(device)
            except Exception:
                pass
        return 0

    @classmethod
    def reset_peak_memory_stats(cls, device: torch.device | None = None) -> None:
        if hasattr(torch, "tpu") and torch.tpu.is_initialized():
            try:
                torch.accelerator.reset_peak_memory_stats(device)
            except Exception:
                pass

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_torchtpu.platforms.tpu_platform import apply_tpu_patches

        apply_tpu_patches()
        _patch_shm_broadcast_size()

    @classmethod
    def init_diffusion_model_runner_runtime(
        cls,
        vllm_config: Any,
        od_config: Any,
        device: torch.device,
    ) -> None:
        from vllm_torchtpu.platforms.tpu_platform import apply_tpu_patches

        apply_tpu_patches()
        _patch_shm_broadcast_size()


def register_tpu_omni_platform() -> str:
    """Entry point for vLLM-Omni TPU platform plugin discovery."""
    from vllm_torchtpu.platforms import _log_tpu_info

    _log_tpu_info()
    _patch_shm_broadcast_size()
    return "vllm_torchtpu.platforms.omni.TpuOmniPlatform"

