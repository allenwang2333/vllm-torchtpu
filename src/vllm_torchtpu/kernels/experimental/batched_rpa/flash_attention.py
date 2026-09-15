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

# Set False to force the general mask path even where the collapsed one
# applies. Exists so the two can be A/B'd at an identical shape and identical
# block sizes -- monkeypatch it in a test and the kernel must produce
# bit-identical output, which is a strictly stronger check than a tolerance.
# Read at trace time, so flipping it costs nothing at runtime.
_COLLAPSE_DECODE_MASK = True


def flash_attention_qk_softmax(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    effective_kv_len: list[jax.Array],  # [B]
    cfgs: configs.RpaConfigs,
    bq_start: int,
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
        # `-1` folds k_heads into the batch dim; upstream's `b` here silently
        # assumes a single KV head per block.
        qk = lax.dot(
            q.reshape(-1, tq, h_size),
            k.reshape(-1, s, h_size),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    qk = qk.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, s)

    qk *= cfgs.model.sm_scale
    if cfgs.serve.scale_k is not None:
        qk *= cfgs.serve.scale_k
    if cfgs.serve.scale_q is not None:
        qk *= cfgs.serve.scale_q

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    qk_masked = []

    int_ty = cfgs.serve.int_ty
    sliding_window = cfgs.model.sliding_window

    # Decode fast path. `tq == bq_c_sz * aligned_num_q_heads_per_kv_head`,
    # because kernel.py slices q in units of the aligned head count, so
    # `bq_c_sz == 1` makes `iota(axis=1) // aligned_num_q_heads_per_kv_head`
    # identically zero and the whole of `q_idx_b` a *scalar*. Every mask term
    # then folds into a bound on the kv iota: the iota is built once and shared
    # by all `batch_size` lanes, instead of two broadcast adds, a compare and a
    # logical_and per lane.
    #
    # Unlike RPA v3, this kernel has no decode specialisation that kills the
    # causal term -- there is no `use_causal_mask` here at all (the wrapper
    # documents the argument as unused and rejects False), so the q-side iota,
    # divide, cast and adds this removes are live work in decode rather than a
    # trace-time-false branch. That is why the collapse is worth more here than
    # it was in v3, where only the kv-side adds survived to be removed.
    #
    # Gate on the structural property rather than on `cfgs.mode`: a caller can
    # pass `decode_block_sizes` explicitly, and the shape is what the rewrite
    # actually depends on.
    collapse_bounds = (_COLLAPSE_DECODE_MASK and cfgs.bq_c_sz == 1
                       and tq == cfgs.aligned_num_q_heads_per_kv_head
                       and sliding_window is None)

    if collapse_bounds:
        kv_iota = lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2)
        for b_idx in range(cfgs.block.batch_size):
            # Widen to i32 before subtracting: Mosaic implements `arith.subi`
            # for i32 only ("Only i32 arith.subi is supported, but got i16"),
            # and kernel.py hands these in already narrowed to `int_ty`, which
            # is i16 whenever q is packed and max_kv_len fits. These are scalar
            # ops on the scalar core, so the wider type costs nothing.
            pq = processed_q_len[b_idx].astype(jnp.int32) + bq_start
            pk = processed_kv_len[b_idx].astype(jnp.int32)
            eff = effective_kv_len[b_idx].astype(jnp.int32)

            # q_idx_b >= kv_idx_b  <=>  iota < processed_q_len - processed_kv_len + 1
            bound = pq - pk + 1
            # `q_idx_b < eff_kv_len_b` is constant across the tile once q_idx_b
            # is scalar, so a false one masks the entire lane. It must be kept,
            # not dropped as implied by causality: on a padded step
            # `mask_out_steps` writes s_idx=-1, which yields pq == pk == eff == 0
            # and leaves `bound == 1` -- unmasking kv column 0 of a dead lane.
            bound = jnp.where(pq < eff, bound, 0)
            # Clamp before narrowing so the cast is total for any `int_ty`, and
            # so the comparison cannot wrap the way `iota + processed_kv_len`
            # could in i16.
            bound = jnp.clip(bound, 0, s).astype(int_ty)

            qk_masked.append(
                jnp.where(kv_iota < bound, qk[b_idx], cfgs.model.mask_value))
    else:
        for b_idx in range(cfgs.block.batch_size):
            kv_idx_b = (lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2) +
                        processed_kv_len[b_idx])
            q_idx_b = (lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1) //
                       cfgs.aligned_num_q_heads_per_kv_head +
                       bq_start).astype(int_ty) + processed_q_len[b_idx]

            eff_kv_len_b = effective_kv_len[b_idx]
            mask_b = q_idx_b < eff_kv_len_b
            mask_b = jnp.logical_and(mask_b, q_idx_b >= kv_idx_b)

            if sliding_window is not None:
                mask_b = jnp.logical_and(mask_b, q_idx_b
                                         < kv_idx_b + sliding_window)

            qk_masked.append(
                jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
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
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape

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
    pv = pv.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, h_size)

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return o_next
