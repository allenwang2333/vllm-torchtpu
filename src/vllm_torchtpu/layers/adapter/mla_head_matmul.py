# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``TPU_MLA_ABSORB_TOKEN_MAJOR``: the MLA absorbs as a token-major Pallas op.

Torch entry for `kernels/mla/head_matmul.py`: `[T, H, Dq] x [H, p, L] ->
[T, H, L]`, replacing `torch.bmm(x.transpose(0, 1), w).transpose(0, 1)` and the
relayout XLA materialises on each side of it.

Build the ops with `build_ops()` outside Dynamo (load time; it takes a lock).
"""

from __future__ import annotations

from threading import Lock

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.mla import head_matmul as kernel

_OUT_DTYPES = {torch.bfloat16: jnp.bfloat16, torch.float32: jnp.float32}
_ops: dict[tuple[bool, bool, torch.dtype], object] = {}
_ops_lock = Lock()


def _make_op(round_before_scale: bool, has_scale: bool, out_dtype: torch.dtype):
    jdt = _OUT_DTYPES[out_dtype]
    tag = f"r{int(round_before_scale)}_s{int(has_scale)}_{str(out_dtype).split('.')[-1]}"
    if has_scale:

        def fn(x: jax.Array, w: jax.Array, scale: jax.Array) -> jax.Array:
            return kernel.head_matmul(
                x, w, scale, round_before_scale=round_before_scale, out_dtype=jdt
            )

        def fake(x, w, scale):
            return x.new_empty((x.shape[0], x.shape[1], w.shape[2]), dtype=out_dtype)

    else:

        def fn(x: jax.Array, w: jax.Array) -> jax.Array:
            return kernel.head_matmul(
                x, w, None, round_before_scale=round_before_scale, out_dtype=jdt
            )

        def fake(x, w):
            return x.new_empty((x.shape[0], x.shape[1], w.shape[2]), dtype=out_dtype)

    op = pallas.jax_op(f"pallas::mla_head_matmul_{tag}", fn)
    op.register_fake(fake)
    return op


def build_ops() -> None:
    """Build every variant once, outside Dynamo."""
    if _ops:
        return
    with _ops_lock:
        if _ops:
            return
        for rb in (True, False):
            for hs in (True, False):
                for dt in _OUT_DTYPES:
                    _ops[(rb, hs, dt)] = _make_op(rb, hs, dt)


def head_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    round_before_scale: bool = True,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`out[t, h] = x[t, h, :w.shape[1]] @ w[h]`, token-major in and out.

    Same contract as the bmm it replaces: `round_before_scale` rounds the f32
    product to `x.dtype` before multiplying by `scale` (`bmm(...) * scale`),
    otherwise the scale is applied in f32.
    """
    out_dtype = out_dtype or x.dtype
    op = _ops.get((round_before_scale, scale is not None, out_dtype))
    assert op is not None, "mla_head_matmul.build_ops() must run at load time"
    if scale is None:
        return op(x, w)
    return op(x, w, scale)
