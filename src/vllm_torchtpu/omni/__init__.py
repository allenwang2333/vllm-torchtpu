# Omni compatibility module
from functools import cache as run_once

import torch

from vllm_torchtpu.logger import init_logger

from .model_patch import apply_omni_rope_patch

logger = init_logger(__name__)


@run_once
def apply_omni_patches():
    logger.info("Applying Omni-specific TPU environment patches.")

    # 1. Configure default diffusion executor backend for in-process single-chip execution
    try:
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        orig_od_init = OmniDiffusionConfig.__init__

        def _tpu_od_config_init(self, *args, **kwargs):
            if kwargs.get("distributed_executor_backend") in (None, "mp"):
                kwargs["distributed_executor_backend"] = (
                    "vllm_torchtpu.omni.executor.TpuDiffusionExecutor")
            orig_od_init(self, *args, **kwargs)

        OmniDiffusionConfig.__init__ = _tpu_od_config_init
    except ImportError:
        pass

    # 2. Patch check_and_update_config and memory hooks for diffusion workers.
    # Note: Underlying vLLM utilities (e.g. startup_plan.py) resolve
    # vllm.platforms.current_platform (the base TpuPlatform singleton) rather than
    # OmniTpuPlatform. We patch base TpuPlatform memory and config methods dynamically
    # when Omni diffusion workers spawn so model_config=None edge cases in diffusion
    # pipelines don't crash core vLLM helpers. This is deferred and only executed
    # during Omni worker spawning.
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
    orig_check_and_update = TpuPlatform.check_and_update_config

    @classmethod
    def _omni_check_and_update_config(cls, vllm_config):
        if vllm_config.model_config is None:
            return
        return orig_check_and_update(vllm_config)

    TpuPlatform.check_and_update_config = _omni_check_and_update_config

    from .platform import OmniTpuPlatform

    TpuPlatform.get_current_memory_usage = classmethod(
        lambda cls, device=None: 0.0)
    TpuPlatform.reset_peak_memory_stats = classmethod(lambda cls: None)
    TpuPlatform.get_device_total_memory = classmethod(
        lambda cls, device_id=0: OmniTpuPlatform.get_device_total_memory(
            device_id))

    # 3. Model and IPC patches
    apply_omni_rope_patch()
    apply_omni_ipc_patch()


@run_once
def apply_omni_ipc_patch():
    """Safety net: Ensure large intermediate tensors serialized via Omni IPC are CPU-backed.

    While TpuDiffusionModelRunner converts primary output tensors to CPU, this patch
    acts as defense-in-depth for any auxiliary tensors passed across Omni IPC channels.
    """
    try:
        import vllm_omni.diffusion.ipc as ipc
        orig_pack = ipc._pack_tensor_if_large

        def _tpu_patched_pack_tensor(val: torch.Tensor):
            ret = orig_pack(val)
            if isinstance(ret, torch.Tensor):
                return ret.detach().cpu()
            return ret

        ipc._pack_tensor_if_large = _tpu_patched_pack_tensor
        logger.info(
            "Applied TPU patch: diffusion.ipc pack_tensor returns CPU tensors."
        )
    except ImportError:
        pass


def register_omni_tpu_platform() -> str:
    """Plugin entry point for vllm_omni.platform_plugins."""
    apply_omni_patches()
    return "vllm_torchtpu.omni.platform.OmniTpuPlatform"
