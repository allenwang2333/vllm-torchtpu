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
"""``TPU_FUSED_ADD_RMSNORM``: prefill-sized RMSNorm as one Pallas pass.

At prefill/mixed token counts XLA splits ``fused_add_rms_norm`` into a row
reduction and an apply pass that re-reads both inputs (and recomputes the
add), so each norm moves ~1.75x the bytes it needs. Rows at or above
``TPU_FUSED_ADD_RMSNORM_MIN_TOKENS`` go to ``kernels/fused_add_rmsnorm``,
which reads ``x``/``residual`` once and writes ``y``/``residual`` once. Smaller
(decode) batches keep the upstream path untouched.

Imported only when the flag is set, so the flag-off model builds the stock
``RMSNorm`` class.
"""

from threading import Lock

import jax
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import fused_add_rmsnorm as kernel

_ops: dict[tuple[float, bool], object] = {}
_ops_lock = Lock()


def _get_op(eps: float, has_residual: bool):
    """Build (once, at model construction -- outside Dynamo) the torch op."""
    key = (float(eps), has_residual)
    op = _ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _ops.get(key)
        if op is not None:
            return op
        tag = f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")
        if has_residual:

            def fn(
                x: jax.Array, residual: jax.Array, weight: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                return kernel.fused_add_rms_norm(x, residual, weight, eps=float(eps))

            def fake(x, residual, weight):
                return torch.empty_like(x), torch.empty_like(residual)

        else:

            def fn(x: jax.Array, weight: jax.Array) -> jax.Array:
                return kernel.fused_add_rms_norm(x, None, weight, eps=float(eps))

            def fake(x, weight):
                return torch.empty_like(x)

        op = pallas.jax_op(f"pallas::fused_add_rmsnorm_r{int(has_residual)}_{tag}", fn)
        op.register_fake(fake)
        _ops[key] = op
        return op


_barrier_op = None


def _get_barrier_op():
    """TPU_RMSNORM_DECODE_MATERIALIZE: identity on (f32 sum, bf16 residual)
    behind one optimization barrier, so XLA materializes both once."""
    global _barrier_op
    if _barrier_op is None:
        with _ops_lock:
            if _barrier_op is None:

                def fn(xf: jax.Array, res: jax.Array) -> tuple[jax.Array, jax.Array]:
                    return jax.lax.optimization_barrier((xf, res))

                def fake(xf, res):
                    return torch.empty_like(xf), torch.empty_like(res)

                op = pallas.jax_op("pallas::rmsnorm_decode_materialize", fn)
                op.register_fake(fake)
                _barrier_op = op
    return _barrier_op


def keep_upstream_path(model: torch.nn.Module) -> int:
    """Pin every TpuRMSNorm in ``model`` to the upstream path; returns the count.

    For modules traced once for all token buckets (the spec-decode draft
    model): the token-count branch in ``forward`` would specialize their graph.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, TpuRMSNorm) and m._pallas_ok:
            m._pallas_ok = False
            n += 1
    return n


@RMSNorm.register_oot
class TpuRMSNorm(RMSNorm):
    """RMSNorm whose prefill-sized calls run the one-pass Pallas kernel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._min_rows = max(envs.TPU_FUSED_ADD_RMSNORM_MIN_TOKENS, kernel.DEFAULT_BLOCK_ROWS)
        self._pallas_ok = (
            self.has_weight
            and self.variance_size_override is None
            and self.hidden_size % 128 == 0
            # Only the residual-stream norms: the small q/kv latent norms read
            # strided slices XLA fuses into their neighbours.
            and self.hidden_size >= envs.TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN
        )
        if self._pallas_ok:
            self._op_res = _get_op(self.variance_epsilon, True)
            self._op_nores = _get_op(self.variance_epsilon, False)
        # TPU_RMSNORM_DECODE_MATERIALIZE: residual-stream norms only (same
        # admission as the Pallas path; a smaller token count keeps the
        # rows-threshold branch) -- see envs.py.
        self._materialize = (
            envs.TPU_RMSNORM_DECODE_MATERIALIZE
            and self.has_weight
            and self.variance_size_override is None
            and self.hidden_size >= envs.TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN
        )
        if self._materialize:
            self._barrier = _get_barrier_op()

    def _use_pallas(self, x: torch.Tensor, residual: torch.Tensor | None) -> bool:
        if not self._pallas_ok or x.device.type != "tpu":
            return False
        if x.dtype != self.weight.dtype or x.shape[-1] != self.hidden_size:
            return False
        if not x.is_contiguous():
            return False
        if residual is not None and (
            residual.dtype != x.dtype
            or residual.shape != x.shape
            or not residual.is_contiguous()
        ):
            return False
        rows = x.numel() // self.hidden_size
        return rows >= self._min_rows and kernel.supports(rows, self.hidden_size)

    def _materialized_add_norm(self, x: torch.Tensor, residual: torch.Tensor):
        """vllm.ir fused_add_rms_norm op for op, with the f32 sum and the
        bf16 residual pinned behind one optimization barrier (same values)."""
        orig_dtype = x.dtype
        xf = x.to(torch.float32) + residual.to(torch.float32)
        xf, new_res = self._barrier(xf, xf.to(orig_dtype))
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(variance + self.variance_epsilon)
        y = y.to(self.weight.dtype) * self.weight.data
        return y.to(orig_dtype), new_res

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if not self._use_pallas(x, residual):
            if (
                self._materialize
                and residual is not None
                and x.device.type == "tpu"
                and x.dtype == residual.dtype == self.weight.dtype
                and x.shape == residual.shape
                and x.shape[-1] == self.hidden_size
            ):
                return self._materialized_add_norm(x, residual)
            return super().forward(x, residual)
        if x.dim() == 2:
            # No view ops on the 2-D residual stream: a view the backend
            # lowers as a relayout would cost a full pass at these sizes.
            if residual is None:
                return self._op_nores(x, self.weight.data)
            return self._op_res(x, residual, self.weight.data)
        shape = x.shape
        x2 = x.reshape(-1, self.hidden_size)
        if residual is None:
            return self._op_nores(x2, self.weight.data).reshape(shape)
        y, new_res = self._op_res(
            x2, residual.reshape(-1, self.hidden_size), self.weight.data
        )
        return y.reshape(shape), new_res.reshape(shape)
