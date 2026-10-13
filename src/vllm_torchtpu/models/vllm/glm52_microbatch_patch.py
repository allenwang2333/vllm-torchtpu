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
"""Two-window token micro-batching for GLM-5.2 / DeepSeek-V3.2 MLA + MoE models.

When `TPU_TOKEN_MICROBATCHES=2` and the padded token bucket is at least
`TPU_MICROBATCH_MIN_TOKENS` (default 1024), splits each `DeepseekV2Model`
backbone step into two token windows `[0:mid]` and `[mid:N]` (`mid = N // 2`)
across the decoder layer stack while keeping the three Pallas/SparseCore
custom ops (`streamindex_topk_dcp`, `sparse_mla_dcp_op`, and `fused_moe_func`)
as single full-batch `[0:N]` kernel launches per layer:

  1. Window 0 pre-attention + indexer projections (`input_layernorm`,
     `fused_qkv_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_layernorm`, RoPE,
     indexer `wq_b` / `wk_weights_proj` / `k_norm` / RoPE / FP8 quant, and
     `W_UK_T` BMM) are emitted first so XLA overlaps them with Layer `l-1`'s
     Window 1 MoE TP All-Reduce.
  2. Window 1 pre-attention + indexer projections run next, followed by a
     single full-batch `[0:N]` call to `streamindex_topk_dcp` and a single
     full-batch `[0:N]` call to `sparse_mla_dcp_op`.
  3. Window 0 merges its DCP attention partials (`cp_lse_ag_out_rs`), runs
     `W_UV` BMM and `o_proj` (launching Window 0's `o_proj` TP All-Reduce),
     while Window 1 merges its DCP partials and runs `W_UV` BMM and `o_proj`.
  4. Window 0 runs `post_attention_layernorm`, `mlp.gate`, and
     `moe_tp_decode_pre_mb2` (Window 0 routing + shared expert GEMMs) while
     Window 1's `o_proj` TP All-Reduce is in flight.
  5. Window 1 runs `post_attention_layernorm` and `mlp.gate`, then
     `moe_tp_decode_combine_mb2` runs Window 1 routing, a single full-batch
     `[0:N]` `fused_moe_func` call, Window 0's MoE TP All-Reduce overlapped
     with Window 1's shared expert GEMMs, and Window 1's MoE TP All-Reduce
     overlapped with Layer `l+1`'s Window 0 pre-attention stack.
"""

from __future__ import annotations

from itertools import islice
from typing import Any, NamedTuple, cast

import torch
from vllm.compilation import backends as vllm_compilation_backends
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.linear import DCPGroupColumnParallelLinear
from vllm.model_executor.models import deepseek_v2
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs
from vllm_torchtpu.layers.adapter import cp_mla_attention as _cp_mla_attention
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMLAAttention,
    VllmTPUMultiHeadLatentAttentionWrapper,
    rope_half_split,
    rope_half_split_leading,
    rope_half_split_leading_supported,
)
from vllm_torchtpu.layers.adapter.moe_tp_decode import (
    moe_tp_decode_combine_mb2,
    moe_tp_decode_pre_mb2,
    moe_tp_decode_supported,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv, quantize_tensor
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_DEEPSEEK_V2_MODEL_FORWARD = deepseek_v2.DeepseekV2Model.forward


class _WindowAttnPre(NamedTuple):
    ql_nope_flat: torch.Tensor
    q_pe_flat: torch.Tensor
    kv_c_normed_flat: torch.Tensor
    k_pe_flat: torch.Tensor
    input_dtype: torch.dtype
    use_qrep: bool
    output_gate: torch.Tensor | None
    idx_q_fp8: torch.Tensor | None
    idx_weights: torch.Tensor | None
    idx_k_packed: torch.Tensor | None


def _has_bound_kv_cache(model: deepseek_v2.DeepseekV2Model) -> bool:
    """False during `profile_run` (before KV caches are allocated), True once bound."""
    if model.start_layer >= model.end_layer:
        return False
    first_layer = model.layers[model.start_layer]
    self_attn = getattr(first_layer, "self_attn", None)
    wrapper = getattr(self_attn, "mla_attn", None)
    mla_layer = getattr(wrapper, "mla_attn", None)
    if not isinstance(mla_layer, VllmTPUMLAAttention):
        return False
    _, _, kv_cache, _ = get_attention_context(mla_layer.layer_name)
    if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
        return False
    return True


def _can_use_mb2(
    model: deepseek_v2.DeepseekV2Model,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> bool:
    if envs.TPU_TOKEN_MICROBATCHES != 2:
        return False
    if vllm_compilation_backends.model_tag != "backbone":
        return False
    if model.use_mha:
        return False
    if not _has_bound_kv_cache(model):
        return False
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        if getattr(layer, "use_sequence_parallel_moe", False):
            return False
        wrapper = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
        if not isinstance(wrapper, VllmTPUMultiHeadLatentAttentionWrapper):
            return False
        if getattr(wrapper, "kimi_preprocess", None) is not None:
            return False
        if getattr(wrapper, "g_proj", None) is not None:
            return False
    if hidden_states.shape[0] != positions.shape[0]:
        return False
    num_tokens = positions.shape[0]
    return num_tokens >= envs.TPU_MICROBATCH_MIN_TOKENS and (num_tokens % 16 == 0)


def _attn_pre_window(
    wrapper: VllmTPUMultiHeadLatentAttentionWrapper,
    pos_w: torch.Tensor,
    h_w: torch.Tensor,
    llama_4_scaling_w: torch.Tensor | None,
) -> _WindowAttnPre:
    q_c = None
    kv_lora = None
    output_gate = None

    if wrapper.q_lora_rank is not None:
        assert wrapper.fused_qkv_a_proj is not None
        assert wrapper.q_a_layernorm is not None
        assert wrapper.q_b_proj is not None

        qkv_lora = wrapper.fused_qkv_a_proj(h_w)[0]
        if wrapper.gate_is_fused:
            q_c, kv_lora, output_gate = qkv_lora.split(
                [
                    wrapper.q_lora_rank,
                    wrapper.kv_lora_rank + wrapper.qk_rope_head_dim,
                    wrapper.num_heads * wrapper.v_head_dim,
                ],
                dim=-1,
            )
        else:
            q_c, kv_lora = qkv_lora.split(
                [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
                dim=-1,
            )
        q_c = wrapper.q_a_layernorm(q_c)
        q = wrapper.q_b_proj(q_c)[0]
    else:
        assert wrapper.kv_a_proj_with_mqa is not None
        assert wrapper.q_proj is not None
        kv_lora = wrapper.kv_a_proj_with_mqa(h_w)[0]
        q = wrapper.q_proj(h_w)[0]

    assert kv_lora is not None
    kv_c, k_pe = kv_lora.split(
        [wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1
    )
    kv_c_normed = wrapper.kv_a_layernorm(kv_c)

    q = q.view(-1, wrapper.num_heads * wrapper.dcp_group_size, wrapper.qk_head_dim)
    q_nope, q_pe = q.split([wrapper.qk_nope_head_dim, wrapper.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.unsqueeze(1)

    if wrapper.rotary_emb is not None:
        q_pe_dtype = q_pe.dtype
        k_pe_dtype = k_pe.dtype
        if getattr(wrapper, "_rope_neox_permuted", False):
            q_pe, k_pe = rope_half_split(wrapper.rotary_emb, pos_w, q_pe, k_pe)
        else:
            q_pe, k_pe = wrapper.rotary_emb(pos_w, q_pe, k_pe)
        q_pe = q_pe.to(dtype=q_pe_dtype)
        k_pe = k_pe.to(dtype=k_pe_dtype)

    idx_q_fp8 = None
    idx_weights = None
    idx_k_packed = None
    if wrapper.is_sparse and wrapper.indexer is not None and not wrapper.skip_topk:
        indexer = wrapper.indexer
        rotary_emb = wrapper.indexer_rope_emb
        assert q_c is not None
        idx_q, _ = indexer.wq_b(q_c)
        idx_q = idx_q.view(-1, indexer.n_head, indexer.head_dim)
        kw, _ = indexer.wk_weights_proj(h_w)
        idx_k = kw[:, : indexer.head_dim]
        idx_weights = kw[:, indexer.head_dim :]

        idx_k = indexer.k_norm(idx_k)
        permuted = getattr(indexer, "_rope_neox_permuted", False)
        if permuted and rope_half_split_leading_supported(rotary_emb):
            idx_q = rope_half_split_leading(
                rotary_emb, pos_w, idx_q, indexer.rope_dim
            )
            idx_k = rope_half_split_leading(
                rotary_emb, pos_w, idx_k, indexer.rope_dim
            )
        else:
            assert rotary_emb is not None
            idx_q_pe, idx_q_nope = torch.split(
                idx_q, [indexer.rope_dim, indexer.head_dim - indexer.rope_dim], dim=-1
            )
            idx_k_pe, idx_k_nope = torch.split(
                idx_k, [indexer.rope_dim, indexer.head_dim - indexer.rope_dim], dim=-1
            )
            if permuted:
                idx_q_pe, idx_k_pe = rope_half_split(
                    rotary_emb, pos_w, idx_q_pe, idx_k_pe.unsqueeze(1)
                )
            else:
                idx_q_pe, idx_k_pe = rotary_emb(
                    pos_w, idx_q_pe, idx_k_pe.unsqueeze(1)
                )
            idx_q_pe = idx_q_pe.reshape(-1, indexer.n_head, indexer.rope_dim)
            idx_k_pe = idx_k_pe.reshape(-1, 1, indexer.rope_dim)
            idx_q = torch.cat([idx_q_pe, idx_q_nope], dim=-1)
            idx_k = torch.cat([idx_k_pe.squeeze(-2), idx_k_nope], dim=-1)

        idx_q = idx_q.view(-1, indexer.head_dim)
        idx_q_fp8, idx_q_scale = quantize_tensor(
            idx_q, torch.float8_e4m3fn, axis=-1, use_ue8m0=False
        )
        idx_q_fp8 = idx_q_fp8.view(-1, indexer.n_head, indexer.head_dim)
        idx_q_scale = idx_q_scale.view(-1, indexer.n_head, 1)

        weights_dtype = idx_weights.dtype
        idx_weights = (
            idx_weights.unsqueeze(-1)
            * idx_q_scale
            * indexer.softmax_scale
            * indexer.n_head**-0.5
        )
        idx_weights = idx_weights.squeeze(-1).to(weights_dtype)

        k_quant, k_scale = quantize_tensor(
            idx_k,
            torch.float8_e4m3fn,
            axis=-1,
            use_ue8m0=True,
        )
        idx_k_packed = torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)

    if llama_4_scaling_w is not None:
        q_nope = q_nope * llama_4_scaling_w
        q_pe = q_pe * llama_4_scaling_w

    q_dcp_replicated = None
    if wrapper.dcp_q_replicate:
        if wrapper.is_sparse:
            q_dcp_replicated = (q_nope, q_pe)
        local_view = cast(
            DCPGroupColumnParallelLinear, wrapper._q_proj_layer()
        )._local_view
        q_nope, q_pe = local_view(q_nope), local_view(q_pe)

    mla_layer = wrapper.mla_attn
    impl = cast(PallasMLAttentionBackendImpl, mla_layer.impl)
    input_dtype = q_nope.dtype
    use_qrep = (
        q_dcp_replicated is not None
        and impl.dcp_size > 1
        and wrapper.is_sparse
    )
    if use_qrep:
        assert mla_layer.W_UK_T_dcp_qrep is not None
        q_nope_use, q_pe_use = q_dcp_replicated
        w_uk_t_src = mla_layer.W_UK_T_dcp_qrep
        w_uk_t_scale = mla_layer.W_UK_T_dcp_qrep_scale
    else:
        q_nope_use, q_pe_use = q_nope, q_pe
        w_uk_t_src = mla_layer.W_UK_T
        w_uk_t_scale = getattr(mla_layer, "W_UK_T_scale", None)
    attn_heads = mla_layer.num_heads * (impl.dcp_size if use_qrep else 1)

    q_nope_t = q_nope_use.transpose(0, 1)
    w_uk_t = (
        w_uk_t_src.to(q_nope_t.dtype)
        if w_uk_t_src.dtype != q_nope_t.dtype
        else w_uk_t_src
    )
    ql_nope = torch.bmm(q_nope_t, w_uk_t)
    if w_uk_t_scale is not None:
        ql_nope = ql_nope * w_uk_t_scale
    ql_nope = ql_nope.transpose(0, 1).to(input_dtype)

    _, k_scale, _ = impl._get_kv_scales(mla_layer)
    if mla_layer.kv_cache_quantized_dtype is not None:
        assert k_scale is not None
        kv_c_normed, _ = quantize_kv(
            mla_layer.kv_cache_quantized_dtype,
            kv_c_normed,
            value=None,
            k_scale=k_scale,
        )
        k_pe, _ = quantize_kv(
            mla_layer.kv_cache_quantized_dtype,
            k_pe,
            value=None,
            k_scale=k_scale,
        )

    ql_nope_flat = ql_nope.view(-1, attn_heads, mla_layer.kv_lora_rank)
    q_pe_flat = q_pe_use.view(-1, attn_heads, mla_layer.qk_rope_head_dim)
    kv_c_normed_flat = kv_c_normed.view(-1, mla_layer.kv_lora_rank)
    k_pe_flat = k_pe.view(-1, mla_layer.qk_rope_head_dim)

    return _WindowAttnPre(
        ql_nope_flat=ql_nope_flat,
        q_pe_flat=q_pe_flat,
        kv_c_normed_flat=kv_c_normed_flat,
        k_pe_flat=k_pe_flat,
        input_dtype=input_dtype,
        use_qrep=use_qrep,
        output_gate=output_gate,
        idx_q_fp8=idx_q_fp8,
        idx_weights=idx_weights,
        idx_k_packed=idx_k_packed,
    )


def _merge_window_lse(
    partial_w: torch.Tensor,
    lse_w: torch.Tensor,
    full_num_tokens: int,
) -> torch.Tensor:
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return partial_w
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and full_num_tokens <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return _cp_mla_attention.merge_lse_partials_all_to_all(
            partial_w, lse_w, group
        )
    full_chunks = _cp_mla_attention._merge_chunks(full_num_tokens)
    window_chunks = max(1, full_chunks // 2)
    if window_chunks > 1 and partial_w.shape[0] % (8 * window_chunks) == 0:
        return _cp_mla_attention._merge_chunked(
            partial_w, lse_w, group, window_chunks
        )
    return cp_lse_ag_out_rs(partial_w, lse_w, group)


def _attn_post_window(
    wrapper: VllmTPUMultiHeadLatentAttentionWrapper,
    mla_layer: VllmTPUMLAAttention,
    outputs_w: torch.Tensor,
    input_dtype: torch.dtype,
    output_gate_w: torch.Tensor | None,
) -> torch.Tensor:
    outputs_t = outputs_w.reshape(
        -1, mla_layer.num_heads, mla_layer.kv_lora_rank
    ).transpose(0, 1)
    w_uv = (
        mla_layer.W_UV.to(outputs_t.dtype)
        if mla_layer.W_UV.dtype != outputs_t.dtype
        else mla_layer.W_UV
    )
    out_proj = torch.bmm(outputs_t, w_uv)
    if hasattr(mla_layer, "W_UV_scale"):
        out_proj = out_proj * mla_layer.W_UV_scale
    attn_out_w = (
        out_proj.transpose(0, 1)
        .to(input_dtype)
        .reshape(-1, mla_layer.num_heads * mla_layer.v_head_dim)
    )
    if output_gate_w is not None:
        attn_out_w = attn_out_w * output_gate_w.sigmoid()
    return wrapper.o_proj(attn_out_w)[0]


def _forward_layer_mb2(
    layer: deepseek_v2.DeepseekV2DecoderLayer,
    pos0: torch.Tensor,
    pos1: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    res0: torch.Tensor | None,
    res1: torch.Tensor | None,
    llama_4_scaling_0: torch.Tensor | None,
    llama_4_scaling_1: torch.Tensor | None,
    full_num_tokens: int,
    mid: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. Input LayerNorm + Pre-Attention + Indexer projections:
    # Emit Window 0 first so XLA overlaps it with Layer l-1's Window 1 MoE All-Reduce.
    if res0 is None:
        res0 = h0
        h0 = layer.input_layernorm(h0)
    else:
        h0, res0 = layer.input_layernorm(h0, res0)

    wrapper = cast(VllmTPUMultiHeadLatentAttentionWrapper, layer.self_attn.mla_attn)
    pre0 = _attn_pre_window(wrapper, pos0, h0, llama_4_scaling_0)

    if res1 is None:
        res1 = h1
        h1 = layer.input_layernorm(h1)
    else:
        h1, res1 = layer.input_layernorm(h1, res1)

    pre1 = _attn_pre_window(wrapper, pos1, h1, llama_4_scaling_1)

    # 2. Full-batch Pallas Indexer Top-K + Sparse MLA Attention (1x per layer on [0:N]):
    topk_indices = None
    if wrapper.is_sparse:
        if wrapper.indexer is not None and not wrapper.skip_topk:
            indexer_op = wrapper.indexer.indexer_op
            assert pre0.idx_q_fp8 is not None and pre1.idx_q_fp8 is not None
            assert pre0.idx_weights is not None and pre1.idx_weights is not None
            assert pre0.idx_k_packed is not None and pre1.idx_k_packed is not None

            attn_metadata_dict = get_forward_context().attn_metadata
            idx_metadata = None
            if isinstance(attn_metadata_dict, dict):
                idx_metadata = attn_metadata_dict.get(indexer_op.k_cache.prefix)
            kv_cache_idx = indexer_op.k_cache.kv_cache
            if (
                not isinstance(idx_metadata, AttentionMetadata)
                or kv_cache_idx.numel() == 0
            ):
                topk_index = indexer_op.topk_indices_buffer
            else:
                q_bytes_full = torch.cat(
                    [
                        pre0.idx_q_fp8.view(torch.uint8),
                        pre1.idx_q_fp8.view(torch.uint8),
                    ],
                    dim=0,
                )
                weights_full = torch.cat([pre0.idx_weights, pre1.idx_weights], dim=0)
                k_packed_full = torch.cat(
                    [pre0.idx_k_packed, pre1.idx_k_packed], dim=0
                )
                seq_lens = idx_metadata.seq_lens
                page_indices = idx_metadata.block_tables
                cu_q_lens = idx_metadata.query_start_loc
                rd = idx_metadata.request_distribution
                distribution = torch.stack([rd[0], rd[0], rd[2]])
                if indexer_op.topk_op_dcp is not None:
                    raw_topk = indexer_op.topk_op_dcp(
                        q_bytes_full,
                        weights_full,
                        kv_cache_idx,
                        k_packed_full,
                        seq_lens,
                        page_indices,
                        cu_q_lens,
                        distribution,
                    )
                else:
                    raw_topk = indexer_op.topk_op(
                        q_bytes_full,
                        weights_full,
                        kv_cache_idx,
                        k_packed_full,
                        seq_lens,
                        page_indices,
                        cu_q_lens,
                        distribution,
                    )
                rows = indexer_op._write_topk_rows(full_num_tokens, raw_topk)
                if envs.TPU_INDEXER_TOPK_SHARED_READ:
                    topk_index = indexer_op.topk_indices_buffer
                else:
                    topk_index = rows
        else:
            topk_index = wrapper.topk_indices_buffer
        assert topk_index is not None
        topk_indices = topk_index[:full_num_tokens]

    mla_layer = wrapper.mla_attn
    impl = cast(PallasMLAttentionBackendImpl, mla_layer.impl)
    attn_metadata_mla, _, kv_cache_mla, _ = get_attention_context(mla_layer.layer_name)

    ql_nope_full = torch.cat([pre0.ql_nope_flat, pre1.ql_nope_flat], dim=0)
    q_pe_full = torch.cat([pre0.q_pe_flat, pre1.q_pe_flat], dim=0)
    kv_c_normed_full = torch.cat([pre0.kv_c_normed_flat, pre1.kv_c_normed_flat], dim=0)
    k_pe_full = torch.cat([pre0.k_pe_flat, pre1.k_pe_flat], dim=0)

    # 3. Post-Attention DCP Merge + W_UV BMM + o_proj + TP All-Reduce per window:
    # Emit Window 0's complete post-attention + o_proj All-Reduce before Window 1's
    # so Window 0's o_proj All-Reduce overlaps with Window 1's LSE merge + W_UV + o_proj.
    if topk_indices is not None:
        assert isinstance(kv_cache_mla, (tuple, list)) and len(kv_cache_mla) == 2
        if impl.dcp_size > 1:
            assert getattr(mla_layer, "sparse_mla_dcp_op", None) is not None
            if pre0.use_qrep:
                ql_nope_dcp, q_pe_dcp = ql_nope_full, q_pe_full
            elif (
                envs.TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER
                and ql_nope_full.dtype == q_pe_full.dtype
            ):
                ql_nope_dcp, q_pe_dcp = _cp_mla_attention.all_gather_q_nope_pe_heads(
                    ql_nope_full, q_pe_full, mla_layer.kv_lora_rank
                )
            else:
                ql_nope_dcp = _cp_mla_attention.all_gather_heads(ql_nope_full)
                q_pe_dcp = _cp_mla_attention.all_gather_heads(q_pe_full)

            partial_full, lse_full = mla_layer.sparse_mla_dcp_op(
                kv_cache_mla,
                ql_nope_dcp,
                q_pe_dcp,
                kv_c_normed_full,
                k_pe_full,
                topk_indices,
                attn_metadata_mla.seq_lens,
                attn_metadata_mla.block_tables,
                attn_metadata_mla.query_start_loc,
                attn_metadata_mla.request_distribution,
            )
            partial_0, partial_1 = partial_full[:mid], partial_full[mid:]
            lse_0, lse_1 = lse_full[:mid], lse_full[mid:]

            outputs_0 = _merge_window_lse(partial_0, lse_0, full_num_tokens)
            h_attn_0 = _attn_post_window(
                wrapper, mla_layer, outputs_0, pre0.input_dtype, pre0.output_gate
            )

            outputs_1 = _merge_window_lse(partial_1, lse_1, full_num_tokens)
            h_attn_1 = _attn_post_window(
                wrapper, mla_layer, outputs_1, pre1.input_dtype, pre1.output_gate
            )
        else:
            if not hasattr(mla_layer, "sparse_mla_op") or mla_layer.sparse_mla_op is None:
                mla_layer.sparse_mla_op = impl._build_sparse_mla_op(mla_layer)
            outputs_full = mla_layer.sparse_mla_op(
                kv_cache_mla,
                ql_nope_full,
                q_pe_full,
                kv_c_normed_full,
                k_pe_full,
                topk_indices,
                attn_metadata_mla.seq_lens,
                attn_metadata_mla.block_tables,
                attn_metadata_mla.query_start_loc,
                attn_metadata_mla.request_distribution,
            )
            outputs_0, outputs_1 = outputs_full[:mid], outputs_full[mid:]
            h_attn_0 = _attn_post_window(
                wrapper, mla_layer, outputs_0, pre0.input_dtype, pre0.output_gate
            )
            h_attn_1 = _attn_post_window(
                wrapper, mla_layer, outputs_1, pre1.input_dtype, pre1.output_gate
            )
    else:
        q_scale, k_scale, v_scale = impl._get_kv_scales(mla_layer)
        if not hasattr(mla_layer, "mla_op") or mla_layer.mla_op is None:
            mla_layer.mla_op = impl._build_mla_op(
                mla_layer, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
            )
        assert isinstance(kv_cache_mla, torch.Tensor)
        outputs_full = mla_layer.mla_op(
            kv_cache_mla,
            ql_nope_full,
            q_pe_full,
            kv_c_normed_full,
            k_pe_full,
            attn_metadata_mla.seq_lens,
            attn_metadata_mla.block_tables,
            attn_metadata_mla.query_start_loc,
            attn_metadata_mla.request_distribution,
        )
        outputs_0, outputs_1 = outputs_full[:mid], outputs_full[mid:]
        h_attn_0 = _attn_post_window(
            wrapper, mla_layer, outputs_0, pre0.input_dtype, pre0.output_gate
        )
        h_attn_1 = _attn_post_window(
            wrapper, mla_layer, outputs_1, pre1.input_dtype, pre1.output_gate
        )

    if (
        not isinstance(layer.self_attn, deepseek_v2.DeepseekAttention)
        and h_attn_0.dtype == torch.float16
    ):
        h_attn_0 = h_attn_0 * (1.0 / layer.routed_scaling_factor)
        h_attn_1 = h_attn_1 * (1.0 / layer.routed_scaling_factor)
        if layer.layer_idx == 0:
            res0 = res0 * (1.0 / layer.routed_scaling_factor)
            res1 = res1 * (1.0 / layer.routed_scaling_factor)

    # 4. Post-Attention LayerNorm + MLP / MoE:
    # Emit Window 0's pre-MoE (or dense MLP) BEFORE consuming `h_attn_1` so
    # Window 0 overlaps with Window 1's o_proj TP All-Reduce.
    h_mlp_in_0, res0 = layer.post_attention_layernorm(h_attn_0, res0)
    mlp = layer.mlp
    if isinstance(mlp, deepseek_v2.DeepseekV2MoE):
        runner = mlp.experts
        routed_experts = runner.routed_experts
        if moe_tp_decode_supported(routed_experts.quant_method):
            logits_0, _ = runner.gate(h_mlp_in_0)
            w_0, ids_0, shared_0 = moe_tp_decode_pre_mb2(
                routed_experts.quant_method,
                routed_experts,
                h_mlp_in_0,
                logits_0,
            )
            h_mlp_in_1, res1 = layer.post_attention_layernorm(h_attn_1, res1)
            logits_1, _ = runner.gate(h_mlp_in_1)
            h0, h1 = moe_tp_decode_combine_mb2(
                routed_experts.quant_method,
                routed_experts,
                h_mlp_in_0,
                h_mlp_in_1,
                w_0,
                ids_0,
                shared_0,
                logits_1,
            )
        else:
            h0 = mlp(h_mlp_in_0)
            h_mlp_in_1, res1 = layer.post_attention_layernorm(h_attn_1, res1)
            h1 = mlp(h_mlp_in_1)
    else:
        h0 = mlp(h_mlp_in_0)
        h_mlp_in_1, res1 = layer.post_attention_layernorm(h_attn_1, res1)
        h1 = mlp(h_mlp_in_1)
        if isinstance(mlp, deepseek_v2.DeepseekV2MLP) and h0.dtype == torch.float16:
            h0 = h0 * (1.0 / layer.routed_scaling_factor)
            h1 = h1 * (1.0 / layer.routed_scaling_factor)

    return h0, h1, res0, res1


def patched_deepseek_v2_model_forward(
    self: deepseek_v2.DeepseekV2Model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "Either input_ids or inputs_embeds must be provided "
                    "to DeepseekV2Model.forward"
                )
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = deepseek_v2._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    if _can_use_mb2(self, positions, hidden_states):
        full_num_tokens = positions.shape[0]
        mid = full_num_tokens // 2
        pos0, pos1 = positions[:mid], positions[mid:]
        h0, h1 = hidden_states[:mid], hidden_states[mid:]
        if residual is None:
            res0: torch.Tensor | None = None
            res1: torch.Tensor | None = None
        else:
            res0, res1 = residual[:mid], residual[mid:]
        if llama_4_scaling is None:
            l4_0 = None
            l4_1 = None
        else:
            l4_0, l4_1 = llama_4_scaling[:mid], llama_4_scaling[mid:]

        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                assert res0 is not None and res1 is not None
                aux_hidden_states.append(torch.cat([h0 + res0, h1 + res1], dim=0))
            h0, h1, res0, res1 = _forward_layer_mb2(
                layer,
                pos0,
                pos1,
                h0,
                h1,
                res0,
                res1,
                l4_0,
                l4_1,
                full_num_tokens,
                mid,
            )

        hidden_states = torch.cat([h0, h1], dim=0)
        assert res0 is not None and res1 is not None
        residual = torch.cat([res0, res1], dim=0)
    else:
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != positions.shape[0]
                and not layer.use_sequence_parallel_moe
            ):
                combined_states = torch.cat([hidden_states, residual], dim=-1)
                combined_states = tensor_model_parallel_all_gather(combined_states, 0)
                combined_states = combined_states[: positions.shape[0]]
                hidden_states, residual = combined_states.split(
                    [self.hidden_size, self.hidden_size], dim=-1
                )
            if idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                if aux_hidden_state.shape[0] != positions.shape[0]:
                    aux_hidden_state = tensor_model_parallel_all_gather(
                        aux_hidden_state, 0
                    )
                    aux_hidden_state = aux_hidden_state[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    if hidden_states.shape[0] != positions.shape[0]:
        combined_states = torch.cat([hidden_states, residual], dim=-1)
        combined_states = tensor_model_parallel_all_gather(combined_states, 0)
        combined_states = combined_states[: positions.shape[0]]
        hidden_states, residual = combined_states.split(
            [self.hidden_size, self.hidden_size], dim=-1
        )

    if self.end_layer in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


def maybe_patch_glm52_microbatch(*args: object, **kwargs: object) -> None:
    """Install two-window token micro-batching on `DeepseekV2Model.forward`."""
    if envs.TPU_TOKEN_MICROBATCHES != 2:
        return None
    deepseek_v2.DeepseekV2Model.forward = patched_deepseek_v2_model_forward
    logger.info(
        "Applied TPU patch: DeepseekV2Model / GLM-5.2 two-window token micro-batching "
        "(TPU_TOKEN_MICROBATCHES=%d, TPU_MICROBATCH_MIN_TOKENS=%d).",
        envs.TPU_TOKEN_MICROBATCHES,
        envs.TPU_MICROBATCH_MIN_TOKENS,
    )
    return None
