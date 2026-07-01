# SPDX-License-Identifier: Apache-2.0
"""Torch op wrapper for :func:`rmsnorm_qkv_fuse`.

Uses ``torch_tpu._internal.pallas.jax_op`` to bridge the JAX/Pallas
kernel into PyTorch. Mirrors the exact pattern used by
``vllm_torchtpu.layers.vllm.linear_common._get_quantized_matmul_op``:

- ``pallas.jax_op("pallas::name", jax_fn)`` wraps the JAX callable.
- ``op.register_fake(...)`` provides shape/dtype inference for FX.
- The wrapper is lazily initialized once per process behind a Lock.

Public entry point ``fused_rmsnorm_qkv`` reshapes ``x`` to 2-D, dispatches
through the wrapped op, and reshapes the output back — matching the
convention in ``linear_common.quantized_matmul``.
"""

from threading import Lock
from typing import Callable, Optional

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.rmsnorm_qkv_fuse.kernel import rmsnorm_qkv_fuse

_op: Optional[Callable] = None
_op_lock = Lock()


def _jax_fn(x: jax.Array, rms_w: jax.Array, qkv_w_q: jax.Array,
            qkv_w_scale: jax.Array) -> jax.Array:
    return rmsnorm_qkv_fuse(x, rms_w, qkv_w_q, qkv_w_scale)


def _fake(x: torch.Tensor, rms_w: torch.Tensor, qkv_w_q: torch.Tensor,
          qkv_w_scale: torch.Tensor) -> torch.Tensor:
    del rms_w, qkv_w_scale
    return torch.empty(x.shape[0],
                       qkv_w_q.shape[0],
                       dtype=x.dtype,
                       device=x.device)


def _get_op() -> Callable:
    global _op
    if _op is not None:
        return _op
    with _op_lock:
        if _op is not None:
            return _op
        op = pallas.jax_op("pallas::rmsnorm_qkv_fuse", _jax_fn)
        op.register_fake(_fake)
        _op = op
        return op


def fused_rmsnorm_qkv(x: torch.Tensor, rms_w: torch.Tensor,
                      qkv_w_q: torch.Tensor,
                      qkv_w_scale: torch.Tensor) -> torch.Tensor:
    """Torch entry: fused RMSNorm(x, rms_w) → int8 QKV projection.

    Shapes:
      - ``x``           : (..., H)   bf16
      - ``rms_w``       : (H,)       bf16
      - ``qkv_w_q``     : (QKV, H)   int8
      - ``qkv_w_scale`` : (QKV,)     fp32

    Returns (``..., QKV``) bf16. Leading dims of ``x`` are flattened to a
    single batch dim before dispatch and restored on return.
    """
    if x.shape[-1] != qkv_w_q.shape[-1]:
        raise ValueError(
            f"Hidden dim mismatch: x has {x.shape[-1]}, qkv_w_q has "
            f"{qkv_w_q.shape[-1]}.")
    if rms_w.shape != (x.shape[-1], ):
        raise ValueError(
            f"rms_w shape {tuple(rms_w.shape)} does not match hidden dim "
            f"{x.shape[-1]}.")
    if qkv_w_scale.shape != (qkv_w_q.shape[0], ):
        raise ValueError(
            f"qkv_w_scale shape {tuple(qkv_w_scale.shape)} does not match "
            f"QKV output dim {qkv_w_q.shape[0]}.")

    orig_out_shape = (*x.shape[:-1], qkv_w_q.shape[0])
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _get_op()(x_2d, rms_w, qkv_w_q, qkv_w_scale)
    return out_2d.reshape(orig_out_shape)
