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

import functools
from typing import Optional

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import (DeepseekV4Indexer,
                                               DeepseekV4IndexerCache)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

_indexer_op_cache = {}

BATCH_AXIS = None


def cdiv(a: int, b: int) -> int:
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
        for arg in (q_rope, indexer_weights, cache_kv, seq_lens, page_indices,
                    cu_q_lens, distribution):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0
        return dummy_out + dummy_sum.astype(jnp.int32)

    def _indexer_local(q_rope, indexer_weights, cache_kv, seq_lens,
                       page_indices, cu_q_lens, distribution):
        q_quant, q_scales = _jax_quantize_tensor(jnp.float8_e4m3fn,
                                                 q_rope,
                                                 axis=-1)

        # Matches the reference fused_indexer_q_rope_quant exactly:
        #   weights_out = weights * q_scale * softmax_scale * head_dim**-0.5
        # (the standard scaled-dot-product-attention scale factor). This
        # used to read `num_heads ** -0.5` -- a different quantity entirely
        # (n_head, not head_dim) -- which was a real divergence from the
        # reference, though likely harmless in practice since it's a
        # uniform positive scalar applied before a pure top-k ranking
        # (doesn't change relative order at full precision; could matter
        # under fp8 rounding near tie boundaries).
        weights = (indexer_weights.astype(q_rope.dtype) * softmax_scale *
                   (head_dim**-0.5) * q_scales)

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
    _cache_spec = get_valid_spec(mesh, BATCH_AXIS)
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

    res = jax.shard_map(
        _indexer_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(q_rope, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
      distribution)
    if res.shape[0] < q_rope.shape[0]:
        res = jnp.pad(res, ((0, q_rope.shape[0] - res.shape[0]), (0, 0)),
                      constant_values=-1)
    return res


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
                get_valid_spec(mesh, attn_data_axis, attn_head_axis,
                               None),  # q_rope
                get_valid_spec(mesh, attn_data_axis,
                               attn_head_axis),  # indexer_weights
                P(),  # cache_kv
                get_valid_spec(mesh, batch_axis),  # seq_lens
                get_valid_spec(mesh, batch_axis),  # page_indices
                P(),  # cu_q_lens
                P(),  # distribution
            ),
        )

        def _fake_indexer(q_rope, indexer_weights, *args, **kwargs):
            return torch.empty((q_rope.shape[0], self.topk_tokens),
                               dtype=torch.int32,
                               device=q_rope.device)

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

        # During the pre-allocation profiling/dummy forward pass, no KV cache
        # has been bound to k_cache yet; return dummy indices of the right
        # shape (matches _fake_indexer above) since values are unused.
        if getattr(self.k_cache, "kv_cache", None) is None:
            return torch.zeros((q_rope.shape[0], self.topk_tokens),
                               dtype=torch.int32,
                               device=q_rope.device)

        kv_cache = self.k_cache.kv_cache

        attn_ctx = get_forward_context().attn_metadata
        if isinstance(attn_ctx, dict):
            prefix_key = getattr(getattr(self, "k_cache", None), "prefix",
                                 None)
            attn_metadata = attn_ctx.get(prefix_key,
                                         next(iter(attn_ctx.values()), None))
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
        fill_pad = torch.full((positions.shape[0], topk_indices.shape[1]),
                              -1,
                              dtype=topk_indices.dtype,
                              device=topk_indices.device)
        topk_indices = torch.cat([topk_indices, fill_pad],
                                 dim=0)[:positions.shape[0]]

        return topk_indices
