# SPDX-License-Identifier: Apache-2.0
"""TPU platform plugin for vllm-omni."""

import os
from typing import Any

import torch
from torch_tpu._internal.utils import hardware
from vllm.logger import init_logger
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_torchtpu.omni.patches import (
    apply_omni_tpu_patches,
    apply_omni_worker_patches,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

logger = init_logger(__name__)

# Attention head counts of diffusion transformers whose heads are not divisible
# by the power-of-2 Ulysses degrees that TPU slices naturally offer (4, 8, ...).
# Only pipelines listed here are auto-promoted to ``ulysses_mode="advanced_uaa"``
# (see ``_maybe_promote_ulysses_mode``); every other model keeps vLLM-Omni's
# default behaviour. Keys are ``OmniDiffusionConfig.model_class_name``.
_ULYSSES_HEAD_COUNTS: dict[str, int] = {
    # ZImageTransformer2DModel: dim=3840, head_dim=128 -> n_heads=30.
    "ZImagePipeline": 30,
}


class OmniTpuPlatform(OmniPlatform, TpuPlatform):
    """TPU implementation of OmniPlatform.

    Inherits all TPU implementations from vLLM's TpuPlatform and implements
    Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.OOT

    # Diffusion worker world size recorded by set_device (see dist_backend).
    _worker_world_size: int = 1

    @property
    def dist_backend(self) -> str:
        # TODO(go/torchtpu-diffusion-findings, FR4): Remove the _worker_world_size
        # fallback once vLLM-Omni's DiffusionWorker keeps WORLD_SIZE set after
        # building its per-worker VllmConfig (check_and_update_config pops it today).
        #
        # DiffusionWorker exports WORLD_SIZE before set_device, but building
        # the worker's VllmConfig afterwards runs check_and_update_config,
        # which pops WORLD_SIZE for that world_size=1 vLLM config. Fall back
        # to the size set_device saw, so multi-rank process groups get
        # tpu_dist instead of gloo.
        world_size = int(os.environ.get("WORLD_SIZE") or self._worker_world_size)
        return TpuPlatform.get_worker_distributed_backend(world_size=world_size)

    def __init__(self) -> None:
        super().__init__()
        apply_omni_tpu_patches()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

        apply_omni_model_specific_patches()
        logger.info_once(
            "Using TpuSDPABackend for diffusion attention (requested: %s).",
            selected_backend,
        )
        return "vllm_torchtpu.omni.attention.TpuSDPABackend"

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        return False

    @classmethod
    def supports_float64(cls) -> bool:
        return False

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return False

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # Must run before WorkerProc starts its DiffusionAsyncOutput thread;
        # init_device() is the first thing DiffusionWorker.__init__ does.
        apply_omni_worker_patches()
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size > 1:
            cls._worker_world_size = world_size
            cls._set_shared_slicebuilder_addresses(world_size)
        cls._prepare_singlehost_tpu_env(world_size)
        torch.accelerator.set_device_index(0)
        _ = torch.empty(1, device=torch.device("tpu"))
        torch.accelerator.synchronize()

    @staticmethod
    def _set_shared_slicebuilder_addresses(world_size: int) -> None:
        """Give every diffusion worker the same PjRt slicebuilder addresses.

        Root cause: vLLM-Omni spawns diffusion workers with no platform hook
        in the parent, so each worker reaches ``_prepare_singlehost_tpu_env``
        without TORCH_TPU_SLICEBUILDER_ADDRESSES and picks its own random
        ports. torch_tpu needs one identical list on every rank, otherwise
        the slice bootstrap never completes. The workers already share
        MASTER_PORT (DiffusionWorker exports od_config.master_port), so the
        list is derived from it.
        """
        # TODO(go/torchtpu-diffusion-findings, S5): Remove once vLLM-Omni calls a
        # platform hook in the parent before spawning diffusion workers, so every
        # rank inherits one address list, or torch_tpu allows more than one
        # addressable device per process.
        #
        addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
        if addresses and len(addresses.split(",")) == world_size:
            return
        base_port = int(os.environ["MASTER_PORT"])
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
            f"localhost:{base_port + 1 + i}" for i in range(world_size)
        )

    @classmethod
    def init_diffusion_model_runner_runtime(
        cls,
        vllm_config: Any,
        od_config: Any,
        device: torch.device,
    ) -> None:
        """Runs in each diffusion worker right before the pipeline is built."""
        cls._maybe_promote_ulysses_mode(od_config)

    @staticmethod
    def _maybe_promote_ulysses_mode(od_config: Any) -> None:
        """Select ``ulysses_mode="advanced_uaa"`` for known odd-head models.

        Root cause: vLLM-Omni's default ``ulysses_mode="strict"`` requires the
        transformer's head count to be divisible by ``ulysses_degree`` and
        raises ``ValueError`` otherwise. TPU slices come in power-of-2 sizes, so
        a model such as Z-Image (30 heads) cannot use ``ulysses_degree`` 4 or 8
        without the padded-head ``"advanced_uaa"`` mode. ``ulysses_mode`` is only
        read at forward time from ``ForwardContext.omni_diffusion_config``, which
        is this same ``od_config`` object, so mutating it here is sufficient.

        Scope: only pipelines in ``_ULYSSES_HEAD_COUNTS`` are promoted, and only
        when the user left the mode at its default. An explicit
        ``ulysses_mode`` is never overridden.

        Note: ``_patch_z_image_transformer_block_flush`` also flips
        ``ZImageTransformer2DModel._sp_plan["unified_prepare"]`` to
        ``auto_pad=True`` now that per-block async graph flushes bound each
        XLA module to a single block's 4 ``all_to_all_single`` collectives,
        eliminating the per-layer ``_all_gather_int`` host sync in
        ``UlyssesParallelAttention`` without deadlocking at ``ulysses_degree=8``.
        """
        # TODO(go/torchtpu-diffusion-findings): Remove once vLLM-Omni selects
        # advanced_uaa itself when n_heads % ulysses_degree != 0.
        p_cfg = getattr(od_config, "parallel_config", None)
        if p_cfg is None:
            return
        ulysses_degree = int(getattr(p_cfg, "ulysses_degree", 1) or 1)
        ulysses_mode = getattr(p_cfg, "ulysses_mode", "strict")
        if ulysses_degree <= 1 or ulysses_mode != "strict":
            return
        model_class_name = getattr(od_config, "model_class_name", None)
        n_heads = _ULYSSES_HEAD_COUNTS.get(model_class_name or "")
        if n_heads is None or n_heads % ulysses_degree == 0:
            return
        logger.warning(
            "%s has %d attention heads, not divisible by ulysses_degree=%d; "
            "promoting ulysses_mode from 'strict' to 'advanced_uaa'. On TPU this "
            "mode adds a per-layer host sync; pass ulysses_mode explicitly or use "
            "a divisor of %d to opt out.",
            model_class_name,
            n_heads,
            ulysses_degree,
            n_heads,
        )
        p_cfg.ulysses_mode = "advanced_uaa"

    @classmethod
    def get_torch_device(cls, _local_rank: int | None = None) -> torch.device:
        return torch.device("tpu")

    @classmethod
    def get_device_count(cls) -> int:
        return cls.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.accelerator.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        hbm = hardware.get_hbm_bytes_per_device()
        if hbm is not None:
            return hbm
        raise ValueError(
            "Could not determine HBM memory for attached TPU device: "
            f"{hardware.get_tpu_device_name()}"
        )

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.accelerator.get_memory_info(device)
        return free

    @classmethod
    def get_device_memory(
        cls,
        device: torch.device | None = None,
    ) -> tuple[int, int]:
        return torch.accelerator.get_memory_info(device)

    @classmethod
    def max_memory_reserved(cls, device: torch.device | None = None) -> int:
        return torch.accelerator.max_memory_reserved(device)

    @classmethod
    def max_memory_allocated(cls, device: torch.device | None = None) -> int:
        return torch.accelerator.max_memory_allocated(device)

    @classmethod
    def reset_peak_memory_stats(cls, device: torch.device | None = None) -> None:
        torch.accelerator.reset_peak_memory_stats(device)

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_torchtpu.worker.tpu_worker.TPUWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_torchtpu.worker.tpu_worker.TPUWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"
