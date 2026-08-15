from functools import cache as run_once

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import is_tpu_device

logger = init_logger(__name__)


@run_once
def apply_omni_rope_patch():
    """Patch RoPE frequency generation for TPU diffusion models.

    FLUX models compute 2D/3D rotary position embeddings using float64 precision
    (`torch.float64`) to prevent numerical degradation across high-resolution images.
    However, TPUs do not natively support float64 trigonometry operations.
    Compute float64 RoPE embeddings on CPU and transfer tensors back to TPU.
    """

    try:
        import vllm_omni.diffusion.models.flux.flux_transformer as flux_mod
        orig_embed_forward = flux_mod.FluxPosEmbed.forward

        def _tpu_patched_embed_forward(self, ids: torch.Tensor):
            if is_tpu_device(ids):
                cos, sin = orig_embed_forward(self, ids.detach().cpu())
                return cos.to(ids.device), sin.to(ids.device)
            return orig_embed_forward(self, ids)

        flux_mod.FluxPosEmbed.forward = _tpu_patched_embed_forward
        logger.info(
            "Applied TPU patch: FluxPosEmbed.forward computes float64 RoPE on CPU."
        )
    except ImportError:
        pass
