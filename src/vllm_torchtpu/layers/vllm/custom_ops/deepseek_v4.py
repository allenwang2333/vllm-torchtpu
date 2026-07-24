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

import os
import sys
import functools
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
torch._dynamo.config.reorderable_logging_functions.add(print)
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.patches import pallas_patched as pallas
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionBackend, AttentionLayerBase
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4 import attention as dsv4_attention

def check_nan(name: str, tensor: torch.Tensor, prefix: str):
    if tensor is None:
        return
    is_fake = (tensor.device.type == "meta" or
               type(tensor).__name__ == "FakeTensor" or
               (hasattr(torch, "compiler") and torch.compiler.is_compiling()))
    if not is_fake:
        # isnan/isinf are not implemented for Float8 types; cast to float32 first.
        is_float8 = (tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2)
        if tensor.dtype.is_floating_point or is_float8:
            t_check = tensor.to(torch.float32) if is_float8 else tensor
            has_nan = torch.isnan(t_check).any().item()
            has_inf = torch.isinf(t_check).any().item()
            if has_nan or has_inf:
                nan_token_0 = (torch.isnan(t_check[0]).any().item() or torch.isinf(t_check[0]).any().item()) if t_check.shape[0] > 0 else False
                print(f"[NAN_DETECTION] NaN/Inf detected in {name} at layer {prefix}! shape={list(tensor.shape)} has_nan={has_nan} has_inf={has_inf} token0_has_nan_inf={nan_token_0}", flush=True)
                print(f"    min={t_check.min().item():.6f} max={t_check.max().item():.6f} mean={t_check.mean().item():.6f}", flush=True)
            elif os.getenv("DSV4_TRACE_ACTIVATIONS") == "1":
                t_mean = t_check.mean().item()
                t_std = (t_check - t_mean).pow(2).mean().sqrt().item()
                print(f"[ACT_TRACE] [{prefix}] {name}: shape={list(tensor.shape)} mean={t_mean:.6f} std={t_std:.6f} min={t_check.min().item():.6f} max={t_check.max().item():.6f}", flush=True)



import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v4.compressor import (
    CompressorStateCache,
    DeepseekCompressor,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec, MLAAttentionSpec
from vllm.distributed.parallel_state import get_tp_group
from vllm_torchtpu.layers.vllm.linear_common import quantized_matmul

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context

# Custom JAX operator caches to prevent compile-time program leaks
_compressor_op_cache = {}
_indexer_op_cache = {}
_attn_op_cache = {}

BATCH_AXIS = None

from vllm_torchtpu.kernels.deepseek_v4.compressor import (
    compressor_forward, compressor_forward_indexer)
from vllm_torchtpu.kernels.deepseek_v4.mla import \
    mla_ragged_paged_attention
from vllm_torchtpu.kernels.deepseek_v4.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk

logger = init_logger(__name__)

_LATEST_KV_CACHES: Dict[str, torch.Tensor] = {}


def cdiv(a: int, b: int) -> int:
    assert b != 0
    return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


def get_valid_spec(mesh: jax.sharding.Mesh, *axes) -> P:
    valid_axes = []
    for axis in axes:
        if axis is None:
            valid_axes.append(None)
        elif isinstance(axis, str):
            valid_axes.append(axis if axis in mesh.axis_names else None)
        else:
            filtered = [a for a in axis if a in mesh.axis_names]
            if len(filtered) == 0:
                valid_axes.append(None)
            elif len(filtered) == 1:
                valid_axes.append(filtered[0])
            else:
                valid_axes.append(tuple(filtered))
    return P(*valid_axes)


def _jax_quantize_tensor(dtype, tensor, axis=-1):
    dtype_info = jnp.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
    scale = abs_max / dtype_max
    scale_inv = jnp.where(scale == 0, jnp.inf, 1 / scale)

    tensor_q = jnp.clip(tensor * scale_inv, dtype_min, dtype_max)
    tensor_q = tensor_q.astype(dtype)
    scale = jnp.squeeze(scale, axis).astype(jnp.float32)
    return tensor_q, scale


def _deepseek_v4_compressor_jax(
    kv_score: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    request_distribution: jax.Array,
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    state_block_tables: jax.Array,
    cache: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> jax.Array:

    forward_fn = (compressor_forward
                  if head_dim == 512 else compressor_forward_indexer)

    def _compressor_local(
        kv_score, ape, norm_weight, cos_sin_cache, positions,
        request_distribution, query_start_loc, seq_lens,
        state_block_tables, cache
    ):
        num_valid_reqs = request_distribution[2]
        num_tokens = kv_score.shape[0]
        num_valid_tokens = query_start_loc[num_valid_reqs]

        q_per_req = query_start_loc[1:] - query_start_loc[:-1]
        max_num_seqs = seq_lens.shape[0]
        token_to_req_indices = jnp.repeat(jnp.arange(max_num_seqs),
                                          q_per_req,
                                          total_repeat_length=num_tokens)

        assert state_block_tables.shape[0] % max_num_seqs == 0
        max_num_blocks_per_req = state_block_tables.shape[0] // max_num_seqs
        state_cache_block_num = positions // state_block_size + token_to_req_indices * max_num_blocks_per_req
        state_cache_block_offset = positions % state_block_size
        slot_mapping = (
            state_block_tables[state_cache_block_num] * state_block_size +
            state_cache_block_offset)
        slot_mapping = jnp.where(
            jnp.arange(num_tokens) < num_valid_tokens, slot_mapping, -1)

        k_block_size = cache.shape[1] * cache.shape[2]
        k_cache_block_num = (
            positions // compress_ratio
        ) // k_block_size + token_to_req_indices * max_num_blocks_per_req
        k_cache_block_offset = (positions //
                                compress_ratio) % k_block_size
        kv_slot_mapping = (
            state_block_tables[k_cache_block_num] * k_block_size +
            k_cache_block_offset)
        kv_slot_mapping = jnp.where(
            jnp.arange(num_tokens) < num_valid_tokens, kv_slot_mapping, -1)

        block_table = state_block_tables.reshape(max_num_seqs, -1)

        out = forward_fn(
            kv_score=kv_score,
            ape=ape,
            norm_weight=norm_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            cache=cache,
            state_block_size=state_block_size,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            rms_eps=rms_eps,
            quant_block=quant_block,
        )

        dummy_sum = 0.0
        for arg in (
            kv_score, ape, norm_weight, cos_sin_cache, positions,
            request_distribution, query_start_loc, seq_lens,
            state_block_tables
        ):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0

        return out + dummy_sum.astype(out.dtype)

    data_spec = get_valid_spec(mesh, None)
    cache_spec = get_valid_spec(mesh, BATCH_AXIS)
    in_specs = (
        data_spec,  # kv_score
        P(),  # ape
        P(),  # norm_weight
        P(),  # cos_sin_cache
        data_spec,  # positions
        P(),  # request_distribution
        P(),  # query_start_loc
        data_spec,  # seq_lens
        data_spec,  # state_block_tables
        P(),  # cache
    )
    out_specs = P()  # updated cache

    if cache.shape[0] == 0:
        dummy_sum = 0.0
        for arg in (kv_score, ape, norm_weight, cos_sin_cache, positions,
                    request_distribution, query_start_loc, seq_lens,
                    state_block_tables):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0
        return cache + dummy_sum

    cache_was_5d = False
    if cache.ndim == 5:
        cache_was_5d = True
        cache = jnp.squeeze(cache, axis=2)

    updated_cache = jax.shard_map(
        _compressor_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        kv_score, ape, norm_weight, cos_sin_cache, positions,
        request_distribution, query_start_loc, seq_lens,
        state_block_tables, cache
    )

    if cache_was_5d:
        updated_cache = jnp.expand_dims(updated_cache, axis=2)
    return updated_cache


def _deepseek_v4_attention_jax(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    is_csa: bool,
    two_caches_same_buffer: bool,
) -> tuple[jax.Array, jax.Array]:
    if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0):
        print(f"[DEBUG_ZERO_CACHE] main_cache_kv.shape={main_cache_kv.shape} sw_cache.shape={sw_cache.shape} swa_only={swa_only}", flush=True)
        t = q.shape[0]
        n_local_heads = q.shape[1]
        head_dim = q.shape[2]
        dummy_out = jnp.zeros((t, n_local_heads, head_dim), dtype=q.dtype)
        dummy_sum = 0.0
        for arg in (q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                    swa_cu_q_lens, swa_distribution, main_cache_kv,
                    main_kv_lens, extra, main_page_indices, main_cu_q_lens,
                    main_distribution, attention_sinks):
            if arg is not None:
                is_f8 = (arg.dtype == jnp.float8_e4m3fn or arg.dtype == jnp.float8_e5m2)
                arg_check = arg.astype(jnp.float32) if is_f8 else arg
                dummy_sum = dummy_sum + jnp.sum(arg_check) * 0
        return dummy_out + dummy_sum, sw_cache + dummy_sum

    def _attention_local(
        q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
        swa_cu_q_lens, swa_distribution, main_cache_kv,
        main_kv_lens, extra, main_page_indices, main_cu_q_lens,
        main_distribution, attention_sinks
    ):
        sw_cache_was_5d = (sw_cache.ndim == 5)
        if sw_cache_was_5d:
            sw_cache = jnp.squeeze(sw_cache, axis=2)
        if main_cache_kv.ndim == 5:
            main_cache_kv = jnp.squeeze(main_cache_kv, axis=2)

        if swa_page_indices is not None:
            swa_page_indices = swa_page_indices.flatten()
        if main_page_indices is not None:
            main_page_indices = main_page_indices.flatten()

        if sw_cache.shape[0] == 0:
            swa_output = jnp.zeros_like(q)
            updated_sw_cache = sw_cache
            swa_l = jnp.zeros((q.shape[0], q.shape[1], 1), dtype=jnp.float32)
            swa_m = jnp.full((q.shape[0], q.shape[1], 1), -1e9, dtype=jnp.float32)
        else:
            swa_output, updated_sw_cache, swa_l, swa_m = (
                mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=new_kv,
                    cache_kv=sw_cache,
                    kv_lens=swa_kv_lens,
                    page_indices=swa_page_indices,
                    cu_q_lens=swa_cu_q_lens,
                    distribution=swa_distribution,
                    attention_sinks=attention_sinks,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    logical_page_size=logical_page_size,
                    num_kv_pages_per_block=1,
                    num_queries_per_block=1,
                    unnormalized_output=False if swa_only else True,
                )
            )

        dummy_sum = 0.0
        for arg in (
            q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
            swa_cu_q_lens, swa_distribution, main_cache_kv,
            main_kv_lens, extra, main_page_indices, main_cu_q_lens,
            main_distribution, attention_sinks
        ):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg.astype(jnp.float32)) * 0

        if swa_only:
            if sw_cache_was_5d:
                updated_sw_cache = jnp.expand_dims(updated_sw_cache, axis=2)
            return swa_output + dummy_sum, updated_sw_cache

        if two_caches_same_buffer and not is_csa:
            main_cache_kv = updated_sw_cache

        if swa_l is not None and swa_l.ndim == 3:
            swa_l = jnp.squeeze(swa_l, axis=-1)
        if swa_m is not None and swa_m.ndim == 3:
            swa_m = jnp.squeeze(swa_m, axis=-1)

        output = mla_ragged_paged_attention(
            q=q,
            cache_kv=main_cache_kv,
            kv_lens=main_kv_lens,
            kv_lens_to_attend=None if is_csa else extra,
            topk_indices=extra if is_csa else None,
            page_indices=main_page_indices,
            cu_q_lens=main_cu_q_lens,
            distribution=main_distribution,
            attention_sinks=attention_sinks,
            swa_accumution=swa_output,
            swa_l=swa_l,
            swa_m=swa_m,
            sm_scale=sm_scale,
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )
        if sw_cache_was_5d:
            updated_sw_cache = jnp.expand_dims(updated_sw_cache, axis=2)
        return output + dummy_sum, updated_sw_cache

    data_spec = get_valid_spec(mesh, None, "model")
    batch_spec = get_valid_spec(mesh, None)
    extra_spec = batch_spec if not is_csa else get_valid_spec(mesh, None, None)
    in_specs = (
        data_spec,  # q
        P(),  # new_kv
        P(),  # sw_cache
        batch_spec,  # swa_kv_lens
        batch_spec,  # swa_page_indices
        P(),  # swa_cu_q_lens
        P(),  # swa_distribution
        P(),  # main_cache_kv
        batch_spec,  # main_kv_lens
        extra_spec,  # extra
        batch_spec,  # main_page_indices
        P(),  # main_cu_q_lens
        P(),  # main_distribution
        P(),  # attention_sinks
    )
    out_specs = (
        data_spec,  # output
        P(),  # updated_sw_cache
    )

    res_out, res_cache = jax.shard_map(
        _attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
        swa_cu_q_lens, swa_distribution, main_cache_kv,
        main_kv_lens, extra, main_page_indices, main_cu_q_lens,
        main_distribution, attention_sinks
    )
    return res_out, res_cache


class VllmCompressorStateCache(CompressorStateCache):

    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__(
            state_dim,
            dtype,
            compress_ratio,
            prefix,
        )
        coff = 1 + (compress_ratio == 4)
        self.head_dim = state_dim // 2 // coff
        compressed_kv_cache_bz = max(
            1,
            get_current_vllm_config().cache_config.block_size // compress_ratio,
        )
        self.block_size = max(1, compressed_kv_cache_bz // 4 // 2 // coff)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
        )


class VllmDeepseekCompressor(DeepseekCompressor):

    def __init__(self, *args, **kwargs) -> None:
        orig_state_cache = dsv4_compressor.CompressorStateCache
        dsv4_compressor.CompressorStateCache = VllmCompressorStateCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_compressor.CompressorStateCache = orig_state_cache
        self.compressor_op = self._build_compressor_op()
        self.num_layers = get_current_vllm_config().model_config.hf_config.num_hidden_layers

    def _build_compressor_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh
        
        wrapped_fn = functools.partial(
            _deepseek_v4_compressor_jax,
            mesh=mesh,
            state_block_size=self.state_cache.block_size,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            rms_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
        )
        
        op_name = f"pallas::deepseek_v4_compressor_{self.head_dim}_{self.compress_ratio}_{int(self.overlap)}"
        global _compressor_op_cache
        if op_name in _compressor_op_cache:
            return _compressor_op_cache[op_name]
            
        from jax.sharding import PartitionSpec as P
        attn_data_axis = None
        batch_axis = BATCH_AXIS
        compressor_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                get_valid_spec(mesh, attn_data_axis, None),  # kv_score
                P(),  # ape
                P(),  # norm_weight
                P(),  # cos_sin_cache
                get_valid_spec(mesh, attn_data_axis),  # positions
                P(),  # request_distribution
                P(),  # query_start_loc
                get_valid_spec(mesh, batch_axis),  # seq_lens
                get_valid_spec(mesh, batch_axis),  # state_block_tables
                P(),  # cache
            ),
            output_partition_specs=(
                get_valid_spec(mesh),  # updated_cache
            ),
        )
        
        def _fake_compressor(
            kv_score, ape, norm_weight, cos_sin_cache, positions,
            request_distribution, query_start_loc, seq_lens,
            state_block_tables, cache, *args, **kwargs
        ):
            return torch.empty_like(cache)
            
        compressor_jax_op.register_fake(_fake_compressor)
        
        _compressor_op_cache[op_name] = compressor_jax_op
        return compressor_jax_op

    def forward(
        self,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        kv_score = kv_score.to(torch.float32)
        
        attn_ctx = get_forward_context().attn_metadata
        if isinstance(attn_ctx, dict):
            prefix_key = getattr(getattr(self, "state_cache", None), "prefix", None)
            state_metadata = attn_ctx.get(prefix_key, next(iter(attn_ctx.values()), None))
        else:
            state_metadata = attn_ctx

        if state_metadata is not None:
            state_req_dist = state_metadata.request_distribution
            state_q_start_loc = state_metadata.query_start_loc
            state_seq_lens = state_metadata.seq_lens
            state_block_tables = state_metadata.block_tables.flatten()
        else:
            state_req_dist = state_q_start_loc = state_seq_lens = state_block_tables = None
        
        orig_cache = self.k_cache.kv_cache
        updated_cache = self.compressor_op(
            kv_score,
            self.ape.clone(),
            self.norm.weight.clone(),
            rotary_emb.cos_sin_cache,
            positions,
            state_req_dist,
            state_q_start_loc,
            state_seq_lens,
            state_block_tables,
            orig_cache,
        )

        
        orig_cache.copy_(updated_cache)


def _deepseek_v4_indexer_jax(
    q_rope: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    head_dim: int,
) -> jax.Array:
    if cache_kv.shape[0] == 0:
        t = q_rope.shape[0]
        dummy_out = jnp.zeros((t, k), dtype=jnp.int32)
        dummy_sum = 0.0
        for arg in (q_rope, indexer_weights, cache_kv, seq_lens,
                    page_indices, cu_q_lens, distribution):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0
        return dummy_out + dummy_sum.astype(jnp.int32)

    def _indexer_local(
        q_rope, indexer_weights, cache_kv, seq_lens,
        page_indices, cu_q_lens, distribution
    ):
        q_quant, q_scales = _jax_quantize_tensor(
            jnp.float8_e4m3fn, q_rope, axis=-1)
        
        weights = (indexer_weights.astype(q_rope.dtype) * softmax_scale *
                   (head_dim ** -0.5) * q_scales)
                   
        return streamindex_topk(
            q=q_quant,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices.flatten(),
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )

    data_spec = get_valid_spec(mesh, None)
    cache_spec = get_valid_spec(mesh, BATCH_AXIS)
    in_specs = (
        get_valid_spec(mesh, None, "model"),  # q_rope
        get_valid_spec(mesh, None, "model"),  # indexer_weights
        P(),  # cache_kv
        data_spec,  # seq_lens
        data_spec,  # page_indices
        P(),  # cu_q_lens
        P(),  # distribution
    )
    out_specs = get_valid_spec(mesh, None, None)  # topk_indices

    if cache_kv.ndim == 5:
        cache_kv = jnp.squeeze(cache_kv, axis=2)

    return jax.shard_map(
        _indexer_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        q_rope, indexer_weights, cache_kv, seq_lens,
        page_indices, cu_q_lens, distribution
    )


class VllmDeepseekV4IndexerCache(DeepseekV4IndexerCache):

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = self.cache_config.block_size
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=align_to(128 + 1, 128),
            dtype=torch.uint8,
            compress_ratio=min(self.compress_ratio, block_size),
            alignment=None,
        )


class VllmDeepseekV4Indexer(DeepseekV4Indexer):

    def __init__(self, *args, **kwargs) -> None:
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_cache = dsv4_attention.DeepseekV4IndexerCache
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4IndexerCache = VllmDeepseekV4IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4IndexerCache = orig_cache
        object.__setattr__(self, "indexer_op", self._build_indexer_op())
        object.__setattr__(self.compressor, "k_cache", self.k_cache)

    def _build_indexer_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh
        
        wrapped_fn = functools.partial(
            _deepseek_v4_indexer_jax,
            mesh=mesh,
            k=self.topk_tokens,
            compression_ratio=self.compress_ratio,
            softmax_scale=self.softmax_scale,
            head_dim=self.head_dim,
        )
        
        op_name = "pallas::deepseek_v4_indexer"
        global _indexer_op_cache
        if op_name in _indexer_op_cache:
            return _indexer_op_cache[op_name]
            
        from jax.sharding import PartitionSpec as P
        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        indexer_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                get_valid_spec(mesh, attn_data_axis, attn_head_axis, None),  # q_rope
                get_valid_spec(mesh, attn_data_axis, attn_head_axis),  # indexer_weights
                P(),  # cache_kv
                get_valid_spec(mesh, batch_axis),  # seq_lens
                get_valid_spec(mesh, batch_axis),  # page_indices
                P(),  # cu_q_lens
                P(),  # distribution
            ),
            output_partition_specs=(
                get_valid_spec(mesh, batch_axis, None),  # indices
            ),
        )
        
        def _fake_indexer(q_rope, indexer_weights, *args, **kwargs):
            return torch.empty((q_rope.shape[0], self.topk_tokens), dtype=torch.int32, device=q_rope.device)
        indexer_jax_op.register_fake(_fake_indexer)
        
        _indexer_op_cache[op_name] = indexer_jax_op
        return indexer_jax_op

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(query)
        q = q.view(-1, self.n_head, self.head_dim)
        
        q_rope, _ = rotary_emb(positions, q)
        
        self.compressor(compressed_kv_score, positions, rotary_emb)
        
        kv_cache = self.k_cache.kv_cache
        
        attn_ctx = get_forward_context().attn_metadata
        if isinstance(attn_ctx, dict):
            prefix_key = getattr(getattr(self, "k_cache", None), "prefix", None)
            attn_metadata = attn_ctx.get(prefix_key, next(iter(attn_ctx.values()), None))
        else:
            attn_metadata = attn_ctx

        if attn_metadata is not None:
            idx_seq_lens = attn_metadata.seq_lens
            idx_block_tables = attn_metadata.block_tables
            idx_q_start_loc = attn_metadata.query_start_loc
            idx_req_dist = attn_metadata.request_distribution
        else:
            idx_seq_lens = idx_block_tables = idx_q_start_loc = idx_req_dist = None

        topk_indices = self.indexer_op(
            q_rope,
            indexer_weights,
            kv_cache,
            idx_seq_lens,
            idx_block_tables,
            idx_q_start_loc,
            idx_req_dist,
        )

        return topk_indices


class VllmDeepseekV4SWACache(DeepseekV4SWACache):

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
    ) -> None:
        super().__init__(head_dim, window_size, dtype, prefix, cache_config)
        compressed_kv_cache_bz = cache_config.block_size
        csa_compression_ratio = 4
        if window_size is None:
            self.block_size = compressed_kv_cache_bz // csa_compression_ratio
        else:
            self.block_size = min(compressed_kv_cache_bz // csa_compression_ratio,
                                  window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if getattr(self, "is_unused", False):
            return None
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )


class VllmDeepseekV4MLAAttention(dsv4_attention.DeepseekV4Attention, AttentionLayerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        orig_indexer = dsv4_attention.DeepseekV4Indexer
        orig_amd_indexer = getattr(ds_v4_amd_model, "DeepseekV4Indexer", None)
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_swa_cache = dsv4_attention.DeepseekV4SWACache
        dsv4_attention.DeepseekV4Indexer = VllmDeepseekV4Indexer
        if orig_amd_indexer is not None:
            ds_v4_amd_model.DeepseekV4Indexer = VllmDeepseekV4Indexer
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
        try:
            super().__init__(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
            self.custom_prefix = prefix
            self.__dict__['mla_attn'] = self
            for module in self.modules():
                if module is not self and hasattr(module, "get_kv_cache_spec"):
                    if module.__class__.__name__ in ("DeepseekV4MultiHeadLatentAttentionWrapper", "DeepseekV4MLAAttention"):
                        module.get_kv_cache_spec = self.get_kv_cache_spec
        finally:
            dsv4_attention.DeepseekV4Indexer = orig_indexer
            if orig_amd_indexer is not None:
                ds_v4_amd_model.DeepseekV4Indexer = orig_amd_indexer
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4SWACache = orig_swa_cache
        if hasattr(self.mla_attn, "compressor") and self.mla_attn.compressor is not None:
            if self.compress_ratio <= 1:
                object.__setattr__(self.mla_attn.compressor, "k_cache", self.mla_attn.swa_cache_layer)
            else:
                object.__setattr__(self.mla_attn.compressor, "k_cache", self.mla_attn.mla_attn)
        if self.compress_ratio > 1:
            if hasattr(self.mla_attn, "swa_cache_layer"):
                object.__setattr__(self.mla_attn.swa_cache_layer, "is_unused", True)
        if hasattr(self.mla_attn, "swa_cache_layer"):
            print(f"[DEBUG_SWA] Layer {prefix} compress_ratio={self.compress_ratio} class={self.mla_attn.swa_cache_layer.__class__.__name__} is_unused={getattr(self.mla_attn.swa_cache_layer, 'is_unused', False)}", flush=True)
        hf_config = vllm_config.model_config.hf_config
        object.__setattr__(self, "attn_out_dim", hf_config.num_attention_heads * hf_config.head_dim)
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.n_groups = self.n_local_groups * self.tp_size
        object.__setattr__(self, "attn_op", self._build_attn_op())
        self.num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        compilation_config = vllm_config.compilation_config
        if compilation_config is not None:
            compilation_config.static_forward_context[prefix] = self
            if hasattr(self.mla_attn, "swa_cache_layer"):
                compilation_config.static_forward_context[self.mla_attn.swa_cache_layer.prefix] = self.mla_attn.swa_cache_layer
            if hasattr(self.mla_attn, "indexer") and hasattr(self.mla_attn.indexer, "k_cache"):
                compilation_config.static_forward_context[self.mla_attn.indexer.k_cache.prefix] = self.mla_attn.indexer.k_cache
            if hasattr(self.mla_attn, "compressor") and hasattr(self.mla_attn.compressor, "state_cache"):
                compilation_config.static_forward_context[self.mla_attn.compressor.state_cache.prefix] = self.mla_attn.compressor.state_cache

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_attn_backend(self) -> type[AttentionBackend]:
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        block_size = vllm_config.cache_config.block_size
        comp_ratio = max(1, self.compress_ratio)
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            compress_ratio=min(comp_ratio, block_size),
            alignment=None,
        )

    def _build_attn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh
        
        swa_only = self.compress_ratio <= 1
        is_csa = self.compress_ratio == 4
        two_caches_same_buffer = not swa_only
        logical_page_size = self.mla_attn.swa_cache_layer.block_size if self.mla_attn.swa_cache_layer is not None else 0
        
        wrapped_fn = functools.partial(
            _deepseek_v4_attention_jax,
            mesh=mesh,
            sm_scale=self.mla_attn.scale,
            sliding_window=self.window_size,
            logical_page_size=logical_page_size,
            swa_only=swa_only,
            is_csa=is_csa,
            two_caches_same_buffer=two_caches_same_buffer,
        )
        
        op_type = "swa" if swa_only else ("csa" if is_csa else "hca")
        op_name = f"pallas::deepseek_v4_attention_{op_type}"
        global _attn_op_cache
        if op_name in _attn_op_cache:
            return _attn_op_cache[op_name]
            
        from jax.sharding import PartitionSpec as P
        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        extra_spec = get_valid_spec(mesh, batch_axis) if not is_csa else get_valid_spec(mesh, batch_axis, None)
        attn_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                get_valid_spec(mesh, attn_data_axis, attn_head_axis, None),  # q
                P(),  # new_kv
                P(),  # sw_cache
                get_valid_spec(mesh, batch_axis),  # swa_kv_lens
                get_valid_spec(mesh, batch_axis),  # swa_page_indices
                P(),  # swa_cu_q_lens
                P(),  # swa_distribution
                P(),  # main_cache_kv
                get_valid_spec(mesh, batch_axis),  # main_kv_lens
                extra_spec,  # extra
                get_valid_spec(mesh, batch_axis),  # main_page_indices
                P(),  # main_cu_q_lens
                P(),  # main_distribution
                P(),  # attention_sinks
            ),
            output_partition_specs=(
                get_valid_spec(mesh, attn_data_axis, attn_head_axis, None),  # out (sharded along attention heads)
                P(),                      # new_sw_cache
            ),
        )

        
        def _fake_attn(q, new_kv, sw_cache, *args, **kwargs):
            num_tokens = q.shape[0]
            out_tensor = torch.empty((num_tokens, self.mla_attn.n_local_heads, self.mla_attn.head_dim), dtype=q.dtype, device=q.device)
            return out_tensor, torch.empty_like(sw_cache)
        attn_jax_op.register_fake(_fake_attn)
        
        _attn_op_cache[op_name] = attn_jax_op
        return attn_jax_op

    def attn_gemm(self, hidden_states):
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        if self.mla_attn.compressor is not None:
            compressor = self.mla_attn.compressor
            kv_score = hidden_states @ compressor.fused_wkv_wgate.weight.T
        else:
            kv_score = None

        if self.indexer is not None:
            indexer = self.indexer
            indexer_weights, _ = indexer.weights_proj(hidden_states)
            indexer_kv_score = hidden_states @ indexer.compressor.fused_wkv_wgate.weight.T
        else:
            indexer_weights = None
            indexer_kv_score = None

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def qnorm_rope(
            self,
            q: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = q.dtype
        qf = q.to(torch.float32)
        
        # Per-head RMSNorm (no weight) over the full head_dim.
        rms = torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + self.mla_attn.eps)
        qf = qf * rms
        
        # Apply rotary embedding to trailing channels.
        rot_dim = self.mla_attn.rotary_emb.rotary_dim
        nope_dim = qf.shape[-1] - rot_dim
        
        q_nope = qf[..., :nope_dim]
        q_rot = qf[..., nope_dim:]
        
        q_padded = torch.cat([q_rot, torch.zeros_like(q_nope)], dim=-1)
        q_padded_rotated, _ = self.mla_attn.rotary_emb(positions, q_padded)
        q_rot_rotated = q_padded_rotated[..., :rot_dim]
        
        return torch.cat([q_nope, q_rot_rotated], dim=-1).to(orig_dtype)

    def kv_rope(
            self,
            kv: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = kv.dtype
        kv_f = kv.unsqueeze(1).to(torch.float32)
        
        rot_dim = self.mla_attn.rotary_emb.rotary_dim
        nope_dim = kv_f.shape[-1] - rot_dim
        
        kv_nope = kv_f[..., :nope_dim]
        kv_rot = kv_f[..., nope_dim:]
        
        kv_padded = torch.cat([kv_rot, torch.zeros_like(kv_nope)], dim=-1)
        kv_padded_rotated, _ = self.mla_attn.rotary_emb(positions, kv_padded)
        kv_rot_rotated = kv_padded_rotated[..., :rot_dim]
        
        kv_rotated = torch.cat([kv_nope, kv_rot_rotated], dim=-1)
        return kv_rotated.squeeze(1).to(orig_dtype)

    def attention_impl(
            self,
            hidden_states: torch.Tensor,
            qr: torch.Tensor,
            kv: torch.Tensor,
            kv_score: torch.Tensor,
            indexer_kv_score: torch.Tensor,
            indexer_weights: torch.Tensor,
            positions: torch.Tensor,
            out: torch.Tensor,
    ) -> torch.Tensor:
        import os
        print(f"[DEBUG_SHAPES] pid={os.getpid()} qr shape={list(qr.shape)} | n_local_heads={self.mla_attn.n_local_heads} head_dim={self.mla_attn.head_dim}", flush=True)
        q = self.wq_b(qr).view(qr.shape[0], self.mla_attn.n_local_heads, self.mla_attn.head_dim)
        q = self.qnorm_rope(q, positions)
        kv = self.kv_rope(kv, positions)
        print(f"[DEBUG_SHAPES] pid={os.getpid()} after rope q shape={list(q.shape)} kv shape={list(kv.shape)}", flush=True)
        check_nan("q (after rope)", q, self.custom_prefix)
        check_nan("kv (after rope)", kv, self.custom_prefix)

        topk_indices = None
        if self.indexer is not None:
            assert self.mla_attn.compressor is not None
            topk_indices = self.indexer(hidden_states, qr, indexer_kv_score,
                                        indexer_weights, positions,
                                        self.mla_attn.indexer_rotary_emb)
            self.mla_attn.compressor(kv_score, positions, self.mla_attn.rotary_emb)
        elif self.mla_attn.compressor is not None:
            self.mla_attn.compressor(kv_score, positions, self.mla_attn.rotary_emb)

        res = self.forward_mqa(
            q,
            kv,
            positions,
            None,
            topk_indices=topk_indices,
        )
        return res

    def _get_active_sw_cache(self) -> torch.Tensor | None:
        if hasattr(self.mla_attn, "swa_cache_layer") and getattr(self.mla_attn.swa_cache_layer, "kv_cache", None) is not None:
            cache = getattr(self.mla_attn.swa_cache_layer, "kv_cache")
            if cache.dtype == torch.uint8:
                return cache
        if hasattr(self, "kv_cache") and getattr(self, "kv_cache", None) is not None:
            cache = getattr(self, "kv_cache")
            if cache.dtype == torch.uint8:
                return cache
        if hasattr(self.mla_attn, "kv_cache") and getattr(self.mla_attn, "kv_cache", None) is not None:
            cache = getattr(self.mla_attn, "kv_cache")
            if cache.dtype == torch.uint8:
                return cache
        return None

    def _get_active_main_cache(self) -> torch.Tensor | None:
        if hasattr(self, "kv_cache") and getattr(self, "kv_cache", None) is not None:
            cache = getattr(self, "kv_cache")
            if cache.dtype == torch.uint8 and cache.shape[-1] > 512:
                return cache
        if hasattr(self.mla_attn, "kv_cache") and getattr(self.mla_attn, "kv_cache", None) is not None:
            cache = getattr(self.mla_attn, "kv_cache")
            if cache.dtype == torch.uint8 and cache.shape[-1] > 512:
                return cache
        return self._get_active_sw_cache()

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        *,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        attn_ctx = get_forward_context().attn_metadata
        if rank == 0 and not hasattr(self, "_printed_ctx_keys"):
            self._printed_ctx_keys = True
            print(f"[DEBUG_ATTN_CTX_TYPE] prefix={self.mla_attn.prefix} type={type(attn_ctx)} val={attn_ctx}", flush=True)

        def _get_field(meta, field_name):
            if meta is None:
                return None
            if isinstance(meta, dict):
                return meta.get(field_name)
            return getattr(meta, field_name, None)

        if isinstance(attn_ctx, dict):
            swa_layer_name = getattr(getattr(self.mla_attn, "swa_cache_layer", None), "prefix", self.mla_attn.prefix)
            swa_attn_metadata = attn_ctx.get(swa_layer_name, attn_ctx.get(self.mla_attn.prefix, next(iter(attn_ctx.values()), None)))
            compressor_state_prefix = getattr(getattr(getattr(self.mla_attn, "compressor", None), "state_cache", None), "prefix", None)
            main_layer_name = self.mla_attn.prefix
            main_attn_metadata = attn_ctx.get(compressor_state_prefix) or attn_ctx.get(main_layer_name) or swa_attn_metadata
        else:
            swa_attn_metadata = attn_ctx
            main_attn_metadata = attn_ctx

        orig_sw_cache = self._get_active_sw_cache()
        sw_cache = orig_sw_cache
        if sw_cache is None:
            sw_cache = torch.zeros((0, 16, 1, self.mla_attn.head_dim), dtype=q.dtype, device=q.device)

        swa_only = self.compress_ratio <= 1
        is_csa = self.compress_ratio == 4

        if swa_attn_metadata is not None:
            swa_seq_lens = _get_field(swa_attn_metadata, "seq_lens")
            swa_block_tables = _get_field(swa_attn_metadata, "block_tables")
            swa_query_start_loc = _get_field(swa_attn_metadata, "query_start_loc")
            swa_request_distribution = _get_field(swa_attn_metadata, "request_distribution")
        else:
            swa_seq_lens = None
            swa_block_tables = None
            swa_query_start_loc = None
            swa_request_distribution = None

        if not swa_only and main_attn_metadata is not None:
            main_cache_kv = self._get_active_main_cache()
            if main_cache_kv is None:
                main_cache_kv = sw_cache
            m_seq_lens = _get_field(main_attn_metadata, "seq_lens")
            ratio = self.compress_ratio if is_csa else 1
            main_kv_lens = m_seq_lens // ratio if m_seq_lens is not None else None
            main_page_indices = _get_field(main_attn_metadata, "block_tables")
            main_cu_q_lens = _get_field(main_attn_metadata, "query_start_loc")
            main_distribution = _get_field(main_attn_metadata, "request_distribution")
        else:
            main_cache_kv = sw_cache
            main_kv_lens = None
            main_page_indices = None
            main_cu_q_lens = None
            main_distribution = None

        if is_csa:
            assert topk_indices is not None
            max_valid_kv = torch.clamp((positions + 1).to(torch.int32) // self.compress_ratio, min=1).unsqueeze(1)
            extra = torch.where((topk_indices >= 0) & (topk_indices < max_valid_kv), topk_indices, torch.full_like(topk_indices, -1))
        else:
            extra = torch.clamp((positions + 1).to(torch.int32) // self.compress_ratio, min=1)
            
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        if rank == 0:
            kv_lens_str = main_kv_lens.tolist() if main_kv_lens is not None else "None"
            print(f"[DEBUG_ATTN_INPUTS] prefix={self.custom_prefix} main_cache_kv.shape={list(main_cache_kv.shape)} mean={main_cache_kv.to(torch.float32).mean().item():.6f} is_csa={is_csa} extra.shape={list(extra.shape)} extra_min={extra.min().item()} extra_max={extra.max().item()} main_kv_lens={kv_lens_str}", flush=True)

        if rank == 0:
            print(f"[DEBUG_FORWARD_MQA_CACHES] prefix={self.mla_attn.prefix} sw_cache.shape={sw_cache.shape if sw_cache is not None else None} main_cache_kv.shape={main_cache_kv.shape if main_cache_kv is not None else None}", flush=True)
        output, new_sw_cache = self.attn_op(
            q,
            kv,
            sw_cache,
            swa_seq_lens,
            swa_block_tables,
            swa_query_start_loc,
            swa_request_distribution,
            main_cache_kv,
            main_kv_lens,
            extra,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            self.attn_sink,
        )

        if orig_sw_cache is not None and new_sw_cache is not None and new_sw_cache.numel() > 0:
            orig_sw_cache.copy_(new_sw_cache)
        return output

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        check_nan("o (input to o_proj)", o, self.custom_prefix)
        t = o.shape[0]
        o_f = o.to(torch.float32).view(t, self.mla_attn.n_local_heads, self.mla_attn.head_dim)
        rot_dim = self.mla_attn.rotary_emb.rotary_dim
        nope_dim = o_f.shape[-1] - rot_dim
        o_nope = o_f[..., :nope_dim]
        o_rot = o_f[..., nope_dim:]
        o_padded = torch.cat([o_rot, torch.zeros_like(o_nope)], dim=-1)
        o_padded_rotated, _ = self.mla_attn.rotary_emb(positions, o_padded, inverse=True)
        o_rot_rotated = o_padded_rotated[..., :rot_dim]
        o_ref = torch.cat([o_nope, o_rot_rotated], dim=-1).to(torch.bfloat16)
        
        # Dequantize wo_a.weight to match w.shape dynamically
        w = self.wo_a.weight
        s = self.wo_a.weight_scale
        if s.ndim == 2:
            block_n = w.shape[0] // s.shape[0]
            block_k = w.shape[1] // s.shape[1]
            s_flat = s.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
        elif s.ndim == 3:
            s_2d = s.squeeze(1) if s.shape[1] == 1 else s.squeeze()
            if s_2d.shape[0] != w.shape[0] // 128 and s_2d.shape[1] == w.shape[0] // 128:
                s_2d = s_2d.transpose(0, 1)
            block_n = w.shape[0] // s_2d.shape[0]
            block_k = w.shape[1] // s_2d.shape[1]
            s_flat = s_2d.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
        else:
            s_flat = s.view(w.shape[0], 1)
            
        s_flat = torch.nan_to_num(s_flat, nan=1.0)
        wo_a_dequant = w.to(o_ref.dtype) * s_flat.to(o_ref.dtype)
        check_nan("wo_a_dequant", wo_a_dequant, self.custom_prefix)
        
        # Grouped einsum projection matching reference ROCm implementation
        n_local_groups = self.n_local_groups
        o_lora_rank = self.o_lora_rank
        
        o_ref_grouped = o_ref.view(t, n_local_groups, -1)
        wo_a_grouped = wo_a_dequant.view(n_local_groups, o_lora_rank, -1)
        
        z = torch.einsum("tgd,grd->tgr", o_ref_grouped, wo_a_grouped)
        z_flat = torch.nan_to_num(z.flatten(1), nan=0.0)
        check_nan("z_flat", z_flat, self.custom_prefix)

        # --- wo_b: RowParallelLinear back to hidden_size (returns (out, bias)).
        out = self.wo_b(z_flat)
        if isinstance(out, tuple):
            out = out[0]
        out = torch.nan_to_num(out, nan=0.0)
        check_nan("out_o_proj", out, self.custom_prefix)
        return out



    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_nan("hidden_states (forward input)", hidden_states, self.custom_prefix)
        
        # Check weights on the very first forward step
        if hasattr(self, "fused_wqa_wkv") and hasattr(self.fused_wqa_wkv, "weight"):
            check_nan("self.fused_wqa_wkv.weight", self.fused_wqa_wkv.weight, self.custom_prefix)
        check_nan("self.q_norm.weight", self.q_norm.weight, self.custom_prefix)
        check_nan("self.kv_norm.weight", self.kv_norm.weight, self.custom_prefix)

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm(hidden_states))
            
        check_nan("qr_kv (raw gemm output)", qr_kv, self.custom_prefix)
        if kv_score is not None:
            check_nan("kv_score (raw)", kv_score, self.custom_prefix)
        if indexer_weights is not None:
            check_nan("indexer_weights (raw)", indexer_weights, self.custom_prefix)

        qr, kv = qr_kv.split([self.mla_attn.q_lora_rank, self.mla_attn.head_dim], dim=-1)
        qr = self.q_norm(qr)
        
        kv = self.kv_norm(kv)
        
        check_nan("qr (after norm)", qr, self.custom_prefix)
        check_nan("kv (after norm)", kv, self.custom_prefix)

        attn_output = self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            None,
        )
        attn_output = torch.nan_to_num(attn_output, nan=0.0)
        check_nan("attn_output (attention_impl)", attn_output, self.custom_prefix)

        out = torch.nan_to_num(self._o_proj(attn_output, positions), nan=0.0)
        check_nan("forward_out", out, self.custom_prefix)
        return out


def patch_deepseek_v4_mla_cls() -> None:
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    ds_v4_amd_model.DeepseekV4Attention = VllmDeepseekV4MLAAttention
    if hasattr(ds_v4_amd_model, "DeepseekV4ROCMAiterMLAAttention"):
        ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention = VllmDeepseekV4MLAAttention
    if hasattr(ds_v4_amd_model, "DeepseekV4MLAAttention"):
        ds_v4_amd_model.DeepseekV4MLAAttention = VllmDeepseekV4MLAAttention
    dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
    logger.info(
        "Patched DeepseekV4Attention, DeepseekV4ROCMAiterMLAAttention and DeepseekV4SWACache for TPU."
    )
