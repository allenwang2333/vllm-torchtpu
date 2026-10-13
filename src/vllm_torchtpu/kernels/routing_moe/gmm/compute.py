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

"""Functions required to perform matrix multiplication."""

import jax
import jax.numpy as jnp
from vllm_torchtpu.kernels.routing_moe.common import layout_util
from vllm_torchtpu.kernels.routing_moe.gmm import config


def _swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
  """Activation used in some models such as GPT-OSS."""

  # Use the limit value provided in GPT-OSS config.
  # TODO(kyuyeunk): Add ability configure limit values.
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  glu = gate * jax.nn.sigmoid(alpha * gate)
  return (up + 1.0) * glu


def _silu_and_mul_with_clamp(
    gate: jax.Array, up: jax.Array, limit: float = 10.0
) -> jax.Array:
  """Activation used in some models DeepSeek V4."""
  # Use the limit value provided in DSV4's config.
  # TODO(kyuyeunk): Add ability configure limit values.
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  return jax.nn.silu(gate) * up


def apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
  """Optionally applies a fused activation function to the accumulator."""

  if fuse_act is None:
    return acc

  acc_gate, acc_up = layout_util.large_deinterleave_lane(acc)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "gelu_tanh":
      return jax.nn.gelu(acc_gate, approximate=True) * acc_up
    case "swigluoai":
      return _swigluoai(acc_gate, acc_up)
    case "silu_and_mul_with_clamp":
      return _silu_and_mul_with_clamp(acc_gate, acc_up)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def matmul(
    cfg: config.RoutingGmmConfig,
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array | None,
    w_bias: jax.Array | None,
    is_last_k_step=True,
    is_first_k_step=True,
) -> jax.Array:
  """Perform stateless matrix multiplication for a given arguments."""

  # TODO(kyuyeunk): Support multiple N/K tiles.
  assert is_last_k_step == is_first_k_step == True

  tpu_info = cfg.tpu_info
  mxu_size = tpu_info.mxu_column_size

  # Step 1: Input pre-processing.
  tile_m, tile_k = x.shape
  tile_n = w.shape[-1]

  # Step 2: Matmul.
  acc_list = []

  if not cfg.x.should_quantize:
    block_acc = jnp.matmul(x, w, preferred_element_type=jnp.float32)
    acc_list.append(block_acc.astype(cfg.acc_dtype))
  else:
    # Quantized matmul path.
    x_q_dtype = cfg.x.quant_dtype
    x_block_size = cfg.x.quant_block_size
    assert x_block_size is not None
    assert x_q_dtype is not None
    # w_scale holds one row per weight block: [k // w_block_size, tile_n].
    w_block_size = cfg.w.quant_block_size or tile_k
    # Each k step lies within a single activation block and a single weight
    # block, so it sees one scale of each.
    k_step = min(x_block_size, w_block_size)

    if jnp.issubdtype(x_q_dtype, jnp.floating):
      dtype_max = float(jnp.finfo(x_q_dtype).max)
      preferred_element_type = jnp.float32
    else:
      dtype_max = float(jnp.iinfo(x_q_dtype).max)
      preferred_element_type = jnp.int32

    # Without n outer loop, result of quantized matmul becomes available only
    # at the last iteration of the loop. This means [tile_m, tile_n] value
    # needs to be stored until the last iteration. By adding n outer loop,
    # result of [tile_m, mxu_size] becomes available at the end of every k
    # inner loop which can be used to pipeline subsequent VPU or VST ops with
    # MXU ops for the next [tile_m, mxu_size].
    for start_n in range(0, tile_n, mxu_size):
      end_n = min(tile_n, start_n + mxu_size)
      col_size = end_n - start_n

      acc_n = jnp.zeros((tile_m, col_size), dtype=cfg.acc_dtype)
      for start_k in range(0, tile_k, k_step):
        end_k = min(tile_k, start_k + k_step)

        block_w = w[start_k:end_k, start_n:end_n]

        # Perform lhs quantization over the whole activation block holding
        # this k step. Note that for every block_lhs, same computation will be
        # performed tiles_n//mxu_size times (and once per k step inside it).
        # But we can let compiler perform CSE and avoid recomputation.
        start_x = start_k // x_block_size * x_block_size
        block_x_full = x[:, start_x : min(tile_k, start_x + x_block_size)]
        block_x = x[:, start_k:end_k]
        block_abs_max = jnp.max(jnp.abs(block_x_full), axis=1, keepdims=True)
        block_scale = block_abs_max / dtype_max

        # If block_scale=0, it will cause division by zero and return either
        # NaN or Inf. Since this can cause numeric issue when downcasting to
        # quantized value, we convert them into 0.
        block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
        # Convert lhs into quantized dtype.
        block_x_q = (block_x * block_scale_inv).astype(x_q_dtype)

        # Unlike unquantized path, compiler may not perform implicit type
        # conversion due to numeric concerns. As this can cause unsupported
        # matmul error, explicit type conversion is performed.
        if not tpu_info.is_matmul_supported(x_q_dtype, block_w.dtype):
          block_w = block_w.astype(x_q_dtype)

        block_acc = jnp.matmul(
            block_x_q,
            block_w,
            preferred_element_type=preferred_element_type,
        ).astype(cfg.acc_dtype)

        if w_scale is not None:
          w_block = start_k // w_block_size
          block_acc *= w_scale[w_block : w_block + 1, start_n:end_n].astype(
              cfg.acc_dtype
          )

        block_acc *= block_scale.astype(cfg.acc_dtype)

        acc_n += block_acc
      acc_list.append(acc_n)
  acc = jnp.concatenate(acc_list, axis=1)

  if w_bias is not None:
    acc += w_bias

  return acc
