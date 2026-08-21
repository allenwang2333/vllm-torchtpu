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

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils


def flash_attention_qk_softmax(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    cfgs: configs.RpaConfigs,
    bq_start: int,
    k_scale: jax.Array | float | None = None,  # [B, KV, S]
):
  """Flash attention kernel."""
  b, k_heads, tq, h_size = q.shape

  if cfgs.serve.scale_q is not None:
    q = q / cfgs.serve.scale_q
    if jnp.issubdtype(k.dtype, jnp.floating):
      dtype_info = jnp.finfo(k.dtype)
      minval = float(dtype_info.min)
      maxval = float(dtype_info.max)
      q = jnp.clip(q, min=minval, max=maxval)
    q = q.astype(k.dtype)

  if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
    s = k.shape[-1]
    qk = lax.dot(
        q.reshape(-1, tq, h_size),
        k.reshape(-1, h_size, s),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  else:
    s = k.shape[-2]
    qk = lax.dot(
        q.reshape(-1, tq, h_size),
        k.reshape(-1, s, h_size),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  qk = qk.reshape(b, k_heads, tq, s)

  qk *= cfgs.model.sm_scale
  if cfgs.serve.scale_q is not None:
    qk *= cfgs.serve.scale_q
  if k_scale is not None:
    qk *= k_scale

  # We convert to the output dtype after scaling, because especially for very
  # low precision, we want to have 32-bit scale factors and do scaling in fp32.
  qk = qk.astype(cfgs.serve.dtype_out)

  if cfgs.model.soft_cap is not None:
    qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

  qk_masked = []

  kv_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 2)
  q_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
  q_iota //= cfgs.aligned_num_q_heads_per_kv_head
  q_kv_diff = q_iota - kv_iota

  for b_idx in range(cfgs.block.batch_size):
    # NOTE: Goal is to compute q_len >= kv_len. But we want to utilize scalar
    # compute as much as possible before involving vector compute. Therefore, we
    # break down a computational steps into following equations to separate out
    # scalar and vector compute.
    # q_len = q_iota + (bq_start + processed_q_len)
    # kv_len = kv_iota + processed_kv_len
    # Step 1: q_len >= kv_len:
    # Step 2:
    #   q_iota + (bq_start + processed_q_len)
    #   >= kv_iota + processed_kv_len
    # Step 3:
    #   q_iota - kv_iota
    #   >= processed_kv_len - (bq_start + processed_q_len)
    # Step 4:
    #   q_kv_delta = q_iota - kv_iota
    #   offset = processed_kv_len - (bq_start + processed_q_len)
    offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
    mask_b = q_kv_diff >= offset

    if (sliding_window := cfgs.model.sliding_window) is not None:
      # NOTE: Goal is to compute q_len < sliding_window + kv_len. And similar
      # to above, we want to minimize vector compute. Therefore, we break
      # down computes into following steps.
      # Step 1: q_len < sliding_window + kv_len
      # Step 2:
      #   q_iota + (bq_start + processed_q_len)
      #   < sliding_window + kv_iota + processed_kv_len
      # Step 3:
      #   q_iota - kv_iota
      #   < sliding_window + processed_kv_len - (bq_start + processed_q_len)
      # Step 4:
      #   q_kv_diff < sliding_window + offset
      mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

    qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
  qk = jnp.stack(qk_masked, axis=0)

  m_curr = jnp.max(qk, axis=-1, keepdims=True)
  m_next = jnp.maximum(m_prev, m_curr)
  p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

  alpha = jnp.exp(m_prev - m_next)
  l_next = alpha * l_prev + p_rowsum

  return p, alpha, m_next, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha: jax.Array,  # [B, KV, TQ, 128]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    cfgs: configs.RpaConfigs,
    v_scale: jax.Array | None = None,  # [B, KV, S]
):
  """Flash attention kernel."""
  b, k_heads, tq, s = p.shape

  if cfgs.serve.per_token_scale:
    assert v_scale is not None
    v_sc = v_scale[:, :, jnp.newaxis, :]
    # We can scale p or v, and because p will be smaller especially for decode
    # workflows, we scale p to save some time.
    p = p.astype(v_sc.dtype) * v_sc

  if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
    h_size = v.shape[-2]
    pv = lax.dot(
        p.reshape(-1, tq, s),
        v.reshape(-1, h_size, s),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  else:
    h_size = v.shape[-1]
    pv = lax.dot(
        p.reshape(-1, tq, s),
        v.reshape(-1, s, h_size),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )

  if not cfgs.serve.per_token_scale and cfgs.serve.scale_v is not None:
    pv = pv * cfgs.serve.scale_v

  pv = pv.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, h_size)
  o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

  return o_next
