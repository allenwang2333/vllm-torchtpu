# SPDX-License-Identifier: Apache-2.0
"""Torch op bridge for :func:`fused_add_rmsnorm_qkv_fp8`.

Same eager-registration pattern as
:mod:`vllm_torchtpu.kernels.fused_add_rmsnorm.torch_op` — lazy
initialization with ``threading.Lock`` breaks Dynamo tracing.
"""

from typing import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.fused_add_rmsnorm_qkv_fp8.kernel import \
    fused_add_rmsnorm_qkv_fp8


def _jax_fn(x: jax.Array, residual: jax.Array, rms_w: jax.Array,
             qkv_w_q: jax.Array,
             qkv_w_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    return fused_add_rmsnorm_qkv_fp8(x, residual, rms_w, qkv_w_q, qkv_w_scale)


def _fake(x: torch.Tensor, residual: torch.Tensor, rms_w: torch.Tensor,
           qkv_w_q: torch.Tensor,
           qkv_w_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    del rms_w, qkv_w_scale
    qkv = torch.empty(x.shape[0],
                       qkv_w_q.shape[0],
                       dtype=x.dtype,
                       device=x.device)
    new_residual = torch.empty_like(residual)
    return qkv, new_residual


# Eagerly register at module import — see fused_add_rmsnorm/torch_op.py
# for the Dynamo-vs-Lock rationale.
_op: Callable = pallas.jax_op("pallas::fused_add_rmsnorm_qkv_fp8", _jax_fn)
_op.register_fake(_fake)


def fused_add_rmsnorm_qkv_fp8_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_w: torch.Tensor,
    qkv_w_q: torch.Tensor,
    qkv_w_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch entry: (qkv_out, new_residual) = fused pre-attention.

    Accepts arbitrary leading dims on x/residual; flattens to 2-D
    before dispatch.
    """
    if x.shape != residual.shape:
        raise ValueError(
            f"residual shape {residual.shape} != x shape {x.shape}.")
    if x.shape[-1] != qkv_w_q.shape[-1]:
        raise ValueError(
            f"hidden dim mismatch: x has {x.shape[-1]}, qkv_w_q has "
            f"{qkv_w_q.shape[-1]}.")
    if rms_w.shape != (x.shape[-1], ):
        raise ValueError(
            f"rms_w shape {rms_w.shape} != ({x.shape[-1]},).")
    if qkv_w_scale.shape != (qkv_w_q.shape[0], ):
        raise ValueError(
            f"qkv_w_scale shape {qkv_w_scale.shape} != "
            f"({qkv_w_q.shape[0]},).")

    x_2d = x.reshape(-1, x.shape[-1])
    resid_2d = residual.reshape(-1, residual.shape[-1])
    qkv_2d, new_resid_2d = _op(x_2d, resid_2d, rms_w, qkv_w_q, qkv_w_scale)
    qkv = qkv_2d.reshape(*x.shape[:-1], qkv_w_q.shape[0])
    new_residual = new_resid_2d.reshape(residual.shape)
    return qkv, new_residual
