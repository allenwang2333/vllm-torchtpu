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
# ==============================================================================
"""The arithmetic of one tile, with no memory layout in sight."""

import jax
import jax.numpy as jnp
from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import config
from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import memory_ref


def dot(
    lhs: jax.Array,
    rhs: memory_ref.FusedWeightsRef,
    cfg: config.PermuteGmmConfig,
) -> jax.Array:
  """`lhs @ w`, or `silu(lhs @ gate) * (lhs @ up)` when the halves are fused.

  A narrower rhs is met by quantizing the LHS to it rather than dequantizing it,
  which keeps the MXU at the rate the weights were quantized for; one scale per
  `quant_block_k` contracting elements is the blocking `gmm_v2` uses, so the
  result is bit-identical to it rather than merely close.
  """
  half = _quantized(lhs, cfg) if cfg.quantize_lhs else _plain(lhs)
  if cfg.fuse_act is None:
    return half(rhs.gate)
  assert cfg.fuse_act == "silu", cfg.fuse_act
  return jax.nn.silu(half(rhs.gate)) * half(rhs.up)


def _plain(lhs: jax.Array):
  def half(ref: memory_ref.WeightsRef) -> jax.Array:
    return jnp.matmul(lhs, ref.weight[...], preferred_element_type=jnp.float32)

  return half


def _quantized(lhs: jax.Array, cfg: config.PermuteGmmConfig):
  qbk = cfg.quant_block_k
  blocks = []
  for s in range(0, cfg.size_k, qbk):
    block = lhs[:, s : s + qbk]
    scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / cfg.quant_max
    blocks.append((
        (block * jnp.where(scale == 0, 0, 1 / scale)).astype(cfg.rhs_dtype),
        scale.astype(cfg.dtype),
    ))

  def half(ref: memory_ref.WeightsRef) -> jax.Array:
    rhs_scale = ref.scale[...].reshape(1, -1).astype(cfg.dtype)
    return sum(
        jnp.matmul(
            q,
            ref.weight[i * qbk : (i + 1) * qbk, :],
            preferred_element_type=jnp.float32,
        ).astype(cfg.dtype)
        * scale
        * rhs_scale
        for i, (q, scale) in enumerate(blocks)
    )

  return half
