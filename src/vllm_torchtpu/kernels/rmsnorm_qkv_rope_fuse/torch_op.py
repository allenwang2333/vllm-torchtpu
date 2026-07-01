# SPDX-License-Identifier: Apache-2.0
"""Torch op wrapper for :func:`rmsnorm_qkv_rope_fuse`.

Same ``pallas.jax_op`` bridge pattern as
:mod:`vllm_torchtpu.kernels.rmsnorm_qkv_fuse.torch_op` and
:mod:`vllm_torchtpu.layers.vllm.linear_common`.

Note: ``q_dim`` and ``k_dim`` are static ints (not tensors); we bake
them into a per-``(q_dim, k_dim)`` cached op so ``pallas.jax_op`` sees a
pure tensors-in signature. Callers within a single model typically use
one ``(q_dim, k_dim)`` pair, so the cache never grows past ~1 entry.
"""

from threading import Lock
from typing import Callable, Dict, Tuple

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.rmsnorm_qkv_rope_fuse.kernel import \
    rmsnorm_qkv_rope_fuse

_ops: Dict[Tuple[int, int], Callable] = {}
_ops_lock = Lock()


def _fake_for(q_dim: int, k_dim: int):

    def _fake(x: torch.Tensor, rms_w: torch.Tensor, qkv_w_q: torch.Tensor,
              qkv_w_scale: torch.Tensor, cos: torch.Tensor,
              sin: torch.Tensor) -> torch.Tensor:
        del rms_w, qkv_w_scale, cos, sin
        return torch.empty(x.shape[0],
                           qkv_w_q.shape[0],
                           dtype=x.dtype,
                           device=x.device)

    return _fake


def _get_op(q_dim: int, k_dim: int) -> Callable:
    key = (q_dim, k_dim)
    op = _ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _ops.get(key)
        if op is not None:
            return op

        def _jax_fn(x: jax.Array, rms_w: jax.Array, qkv_w_q: jax.Array,
                    qkv_w_scale: jax.Array, cos: jax.Array,
                    sin: jax.Array) -> jax.Array:
            return rmsnorm_qkv_rope_fuse(x, rms_w, qkv_w_q, qkv_w_scale, cos,
                                         sin, q_dim, k_dim)

        # jax_op names must be unique per (q_dim, k_dim) — otherwise
        # a second Linear layer with different split sizes would reuse
        # the wrong lowered op.
        name = f"pallas::rmsnorm_qkv_rope_fuse_q{q_dim}_k{k_dim}"
        wrapped = pallas.jax_op(name, _jax_fn)
        wrapped.register_fake(_fake_for(q_dim, k_dim))
        _ops[key] = wrapped
        return wrapped


def fused_rmsnorm_qkv_rope(x: torch.Tensor, rms_w: torch.Tensor,
                           qkv_w_q: torch.Tensor, qkv_w_scale: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor, q_dim: int,
                           k_dim: int) -> torch.Tensor:
    """Torch entry: fused RMSNorm → int8 QKV → RoPE(q,k).

    Shapes (leading dims of ``x`` are flattened before dispatch):
      - ``x``           : (..., H)     bf16
      - ``rms_w``       : (H,)         bf16
      - ``qkv_w_q``     : (QKV, H)     int8
      - ``qkv_w_scale`` : (QKV,)       fp32
      - ``cos``, ``sin``: (..., hd/2)  bf16 or fp32

    Returns (``..., QKV``) bf16 with RoPE applied to the first ``q_dim``
    output columns as ``q`` and the next ``k_dim`` as ``k``.
    """
    if x.shape[-1] != qkv_w_q.shape[-1]:
        raise ValueError(
            f"Hidden dim mismatch: x has {x.shape[-1]}, qkv_w_q has "
            f"{qkv_w_q.shape[-1]}.")

    orig_out_shape = (*x.shape[:-1], qkv_w_q.shape[0])
    x_2d = x.reshape(-1, x.shape[-1])
    cos_2d = cos.reshape(-1, cos.shape[-1])
    sin_2d = sin.reshape(-1, sin.shape[-1])
    out_2d = _get_op(q_dim, k_dim)(x_2d, rms_w, qkv_w_q, qkv_w_scale, cos_2d,
                                   sin_2d)
    return out_2d.reshape(orig_out_shape)
