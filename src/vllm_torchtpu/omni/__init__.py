# Omni compatibility module
from functools import cache as run_once

from vllm_torchtpu.logger import init_logger

from .model_patch import apply_omni_rope_patch

logger = init_logger(__name__)


@run_once
def apply_omni_patches():
    logger.info("Applying Omni-specific TPU environment patches.")
    apply_omni_rope_patch()


def register_omni_tpu_platform() -> str:
    """Plugin entry point for vllm_omni.platform_plugins."""
    apply_omni_patches()
    return "vllm_torchtpu.omni.platform.OmniTpuPlatform"
