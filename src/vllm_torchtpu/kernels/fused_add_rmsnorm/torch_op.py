# SPDX-License-Identifier: Apache-2.0
"""Torch op bridge for :func:`fused_add_rmsnorm`.

Uses ``torch_tpu._internal.pallas.jax_op`` to expose the JAX/Pallas
kernel as a torch custom op, matching the pattern used by
``vllm_torchtpu.layers.vllm.linear_common._get_quantized_matmul_op``.
Exposed via ``fused_add_rmsnorm_torch(x, residual, weight, eps)``.
"""

from typing import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.fused_add_rmsnorm.kernel import fused_add_rmsnorm

# eps as a Python int-cast (Pallas kernel takes fp32; we bake a static
# default here since vLLM's RMSNorm always uses 1e-6 for Qwen3.5).
_EPS = 1.0e-6


def _jax_fn(x: jax.Array, residual: jax.Array,
             weight: jax.Array) -> tuple[jax.Array, jax.Array]:
    return fused_add_rmsnorm(x, residual, weight, eps=_EPS)


def _fake(x: torch.Tensor, residual: torch.Tensor,
           weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    del weight
    return (
        torch.empty_like(x),
        torch.empty_like(residual),
    )


# Eagerly register the op at module import. Lazy registration behind a
# ``threading.Lock`` breaks torch.compile / Dynamo tracing: Dynamo does
# not know how to enter a lock context manager, and the model's forward
# is traced under torch.compile, so a Lock inside the hot path would
# hard-crash the worker at compile time.
_op: Callable = pallas.jax_op("pallas::fused_add_rmsnorm", _jax_fn)
_op.register_fake(_fake)


def _get_op() -> Callable:
    return _op


def fused_add_rmsnorm_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch entry: (normed, new_residual) = RMSNorm(residual + x, w).

    Accepts arbitrary leading dims; flattens to 2-D before dispatch.
    """
    if x.shape != residual.shape:
        raise ValueError(
            f"residual shape {residual.shape} != x shape {x.shape}.")
    if weight.shape != (x.shape[-1], ):
        raise ValueError(
            f"weight shape {weight.shape} != ({x.shape[-1]},).")

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    r_2d = residual.reshape(-1, orig_shape[-1])
    normed_2d, resid_2d = _get_op()(x_2d, r_2d, weight)
    return (normed_2d.reshape(orig_shape), resid_2d.reshape(orig_shape))
