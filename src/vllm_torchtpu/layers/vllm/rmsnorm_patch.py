# SPDX-License-Identifier: Apache-2.0
"""Patcher that swaps vLLM's fused-add RMSNorm to a Pallas fused kernel.

The stock path on TPU dispatches ``GemmaRMSNorm.forward_native``
(equivalent to Qwen3NextRMSNorm), which composes three separate ops:

    x = x + residual            # bf16 elementwise
    residual = x                # alias, no HBM write actually
    normed = ir.ops.rms_norm(x, weight+1, eps)

Even without the alias write, the intermediate ``x`` is a (M, H) bf16
HBM buffer read/written 120x per Qwen3.5 decode step. This patch
substitutes a single Pallas call that keeps ``new_residual`` in VMEM
across the RMSNorm compute and emits both outputs in one pass.

The patch is installed by ``vllm_torchtpu`` at platform registration
and only fires when:
- ``residual`` is provided (i.e. layer 1+; layer 0 falls through)
- Both tensors live on the ``tpu`` device
- The input dtype is bf16 (kernel is bf16-specialized)
- The last dim is a multiple of 128 (Pallas layout requirement)

Fallback path preserves the exact ``forward_native`` semantics so any
edge case (custom dtype, sub-128 hidden, CPU device during profiling)
lands on the reference implementation.
"""

from __future__ import annotations

from typing import Optional

import torch


def _install_gemma_rmsnorm_pallas_patch() -> None:
    """Monkey-patch ``GemmaRMSNorm.forward_native`` to prefer the Pallas
    fused kernel when the residual + shape + device gate matches.

    Safe to call multiple times; no-op after the first install.
    """
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    if getattr(GemmaRMSNorm, "_torchtpu_pallas_patched", False):
        return

    _orig_forward_native = GemmaRMSNorm.forward_native

    def _patched(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        # Gate: only take the Pallas path when it's a clear win. Falls
        # through to the reference on anything unusual.
        take_pallas = (
            residual is not None
            and x.device.type == "tpu"
            and residual.device.type == "tpu"
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and x.shape == residual.shape
            and (x.shape[-1] % 128 == 0)
        )
        if not take_pallas:
            return _orig_forward_native(self, x, residual)

        # Lazy import so tests without torch_tpu still load this module.
        from vllm_torchtpu.kernels.fused_add_rmsnorm.torch_op import \
            fused_add_rmsnorm_torch

        # Cache (weight + 1) once per module — GemmaRMSNorm semantics.
        eff = getattr(self, "_torchtpu_eff_weight", None)
        if eff is None or eff.dtype != x.dtype or eff.device != x.device:
            eff = (self.weight.data.to(dtype=x.dtype, device=x.device) +
                   1.0)
            self._torchtpu_eff_weight = eff

        normed, new_residual = fused_add_rmsnorm_torch(x, residual, eff)
        return normed, new_residual

    GemmaRMSNorm.forward_native = _patched  # type: ignore[assignment]
    GemmaRMSNorm._torchtpu_pallas_patched = True  # type: ignore[attr-defined]
