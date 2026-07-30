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

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import (CompressorStateCache,
                                                DeepseekCompressor)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu.kernels.deepseek_v4.compressor import (
    compressor_forward, compressor_forward_indexer)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.patches import pallas_patched as pallas

logger = init_logger(__name__)

_compressor_op_cache = {}

BATCH_AXIS = None


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
    k_block_tables: jax.Array,
    state_cache_buf: jax.Array,
    cache: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    state_block_size: int,
    head_dim: int,
    state_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> tuple[jax.Array, jax.Array]:

    forward_fn = (compressor_forward
                  if head_dim == 512 else compressor_forward_indexer)

    def _compressor_local(kv_score, ape, norm_weight, cos_sin_cache, positions,
                          request_distribution, query_start_loc, seq_lens,
                          state_block_tables, k_block_tables, state_cache_buf,
                          cache):
        num_valid_reqs = request_distribution[2]
        num_tokens = kv_score.shape[0]
        num_valid_tokens = query_start_loc[num_valid_reqs]

        q_per_req = query_start_loc[1:] - query_start_loc[:-1]
        max_num_seqs = seq_lens.shape[0]
        token_to_req_indices = jnp.repeat(jnp.arange(max_num_seqs),
                                          q_per_req,
                                          total_repeat_length=num_tokens)

        # `state_block_tables` addresses the compressor's own dedicated
        # scratch buffer (state_cache_buf) -- its own independent
        # vLLM-managed KV-cache-spec allocation, physically disjoint from
        # any real attention KV data. `k_block_tables` addresses the shared
        # main/compressed-KV buffer (`cache`), i.e. the same page-table space
        # `main_page_indices` uses on the attention read side. These two
        # block tables must be kept separate: previously this code reused
        # `state_block_tables` for both, which let the compressor's per-token
        # state write (which, prior to the dedicated buffer, could touch
        # every row-slot of whatever page it landed on) alias onto and
        # clobber real/compressed KV bytes living in the shared buffer.
        assert state_block_tables.shape[0] % max_num_seqs == 0
        max_num_state_blocks_per_req = (state_block_tables.shape[0] //
                                        max_num_seqs)
        # Wrap within this request's own state-block allocation (a
        # circular/sliding-window buffer, per SlidingWindowMLASpec) --
        # without the modulo, any position beyond
        # max_num_state_blocks_per_req * state_block_size aliases into the
        # *next* request's block-table slice.
        state_cache_block_num = (
            positions // state_block_size) % max_num_state_blocks_per_req + (
                token_to_req_indices * max_num_state_blocks_per_req)
        state_cache_block_offset = positions % state_block_size
        slot_mapping = (
            state_block_tables[state_cache_block_num] * state_block_size +
            state_cache_block_offset)
        slot_mapping = jnp.where(
            jnp.arange(num_tokens) < num_valid_tokens, slot_mapping, -1)

        assert k_block_tables.shape[0] % max_num_seqs == 0
        max_num_k_blocks_per_req = k_block_tables.shape[0] // max_num_seqs
        k_block_size = cache.shape[1] * cache.shape[2]
        k_cache_block_num = (
            (positions // compress_ratio) //
            k_block_size) % max_num_k_blocks_per_req + (
                token_to_req_indices * max_num_k_blocks_per_req)
        k_cache_block_offset = (positions // compress_ratio) % k_block_size
        kv_slot_mapping = (k_block_tables[k_cache_block_num] * k_block_size +
                           k_cache_block_offset)
        kv_slot_mapping = jnp.where(
            jnp.arange(num_tokens) < num_valid_tokens, kv_slot_mapping, -1)

        block_table = state_block_tables.reshape(max_num_seqs, -1)

        out, updated_state_cache_buf = forward_fn(
            kv_score=kv_score,
            ape=ape,
            norm_weight=norm_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            state_cache_buf=state_cache_buf,
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
        for arg in (kv_score, ape, norm_weight, cos_sin_cache, positions,
                    request_distribution, query_start_loc, seq_lens,
                    state_block_tables, k_block_tables):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0

        return (out + dummy_sum.astype(out.dtype), updated_state_cache_buf +
                dummy_sum.astype(updated_state_cache_buf.dtype))

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
        data_spec,  # k_block_tables
        P(),  # state_cache_buf
        P(),  # cache
    )
    out_specs = (P(), P())  # updated cache, updated state_cache_buf

    if cache.shape[0] == 0 or state_cache_buf.shape[0] == 0:
        dummy_sum = 0.0
        for arg in (kv_score, ape, norm_weight, cos_sin_cache, positions,
                    request_distribution, query_start_loc, seq_lens,
                    state_block_tables, k_block_tables):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0
        return cache + dummy_sum, state_cache_buf + dummy_sum

    cache_was_5d = False
    if cache.ndim == 5:
        cache_was_5d = True
        cache = jnp.squeeze(cache, axis=2)

    state_cache_buf_was_5d = False
    if state_cache_buf.ndim == 5:
        state_cache_buf_was_5d = True
        state_cache_buf = jnp.squeeze(state_cache_buf, axis=2)

    updated_cache, updated_state_cache_buf = jax.shard_map(
        _compressor_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(kv_score, ape, norm_weight, cos_sin_cache, positions,
      request_distribution, query_start_loc, seq_lens, state_block_tables,
      k_block_tables, state_cache_buf, cache)

    if cache_was_5d:
        updated_cache = jnp.expand_dims(updated_cache, axis=2)
    if state_cache_buf_was_5d:
        updated_state_cache_buf = jnp.expand_dims(updated_state_cache_buf,
                                                  axis=2)
    return updated_cache, updated_state_cache_buf


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
        # NOTE: don't recompute self.block_size here. The base
        # CompressorStateCache.__init__ (vllm.models.deepseek_v4.compressor)
        # already sets it correctly (4 for CSA/compress_ratio=4, 8 for
        # HCA/compress_ratio=128), derived from the exact packed-cache row
        # geometry (see its comment: KV block shape [64, 584] -> block_size
        # 4/8). A previous override here recomputed it from
        # cache_config.block_size and produced 1 for both CSA and HCA, which
        # makes one compressor-state token span an entire physical page
        # (page_size // 1 rows) instead of the intended 16 (CSA) / 8 (HCA)
        # rows -- since compressor state and the shared MLA cache buffer
        # overlay the same physical pages, every state write was clobbering
        # all 64 rows of whatever page it landed on, wiping real KV and
        # compressed-KV bytes living on the same page.
        coff = 1 + (compress_ratio == 4)
        self.head_dim = state_dim // 2 // coff

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Declare this as a raw-uint8, self-managed-layout buffer (matching
        # the shared MLA/SWA cache convention) rather than dtype=float32.
        # vllm_torchtpu's kv_cache_spec_normalizer force-overrides every
        # spec's dtype to the model's global KV-cache quantization dtype
        # (e.g. fp8) *unless* the spec already declares torch.uint8 -- that's
        # the deliberate escape hatch the shared buffers use. Declaring
        # float32 here got silently swept into fp8, truncating the raw
        # (pre-quantization) state values. Packing as uint8 byte-lanes here
        # reuses the exact same pack_state_cache/unpack_state_cache byte
        # layout already used for the shared buffer (see
        # kernels/deepseek_v4/compress_norm_rope.py), just against this
        # dedicated, disjoint tensor instead.
        #
        # This tensor's physical row geometry (kernels/deepseek_v4/mla.py's
        # get_kv_cache_shape) packs kv_slots = align_to(block_size, 4) rows,
        # i.e. kv_slots == self.block_size (4 for CSA, 8 for HCA) -- unlike
        # the shared main/SWA cache, which packs a much larger token count
        # per physical page. So compress_norm_rope.py's _state_chunk_dims
        # (run later against the real allocated tensor) always resolves
        # rows_per_token = kv_slots // state_block_size == 1 here, meaning
        # the whole state_dim vector lives in a single row-slot: bytes_per_row
        # is simply state_dim * 4, not state_dim divided across multiple rows.
        bytes_per_row = self.state_dim * 4
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(bytes_per_row, 128),
            dtype=torch.uint8,
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
        self.num_layers = get_current_vllm_config(
        ).model_config.hf_config.num_hidden_layers

    def _build_compressor_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        state_dim = getattr(self.state_cache, "state_dim",
                            getattr(self, "state_dim", 512))

        wrapped_fn = functools.partial(
            _deepseek_v4_compressor_jax,
            mesh=mesh,
            state_block_size=self.state_cache.block_size,
            head_dim=self.head_dim,
            state_dim=state_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            rms_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
        )

        op_name = f"pallas::deepseek_v4_compressor_{self.head_dim}_{self.compress_ratio}_{int(self.overlap)}_{self.state_cache.block_size}_{state_dim}_v6"

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
                get_valid_spec(mesh, batch_axis),  # k_block_tables
                P(),  # state_cache_buf
                P(),  # cache
            ),
            output_partition_specs=(
                get_valid_spec(mesh),  # updated_cache
                get_valid_spec(mesh),  # updated_state_cache_buf
            ),
        )

        def _fake_compressor(kv_score, ape, norm_weight, cos_sin_cache,
                             positions, request_distribution, query_start_loc,
                             seq_lens, state_block_tables, k_block_tables,
                             state_cache_buf, cache, *args, **kwargs):
            return torch.empty_like(cache), torch.empty_like(state_cache_buf)

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
            prefix_key = getattr(getattr(self, "state_cache", None), "prefix",
                                 None)
            state_metadata = attn_ctx.get(prefix_key,
                                          next(iter(attn_ctx.values()), None))
            _k_cache_obj = getattr(self, "k_cache", None)
            k_cache_prefix_key = getattr(
                _k_cache_obj, "prefix",
                getattr(_k_cache_obj, "custom_prefix", None))
            k_cache_metadata = attn_ctx.get(k_cache_prefix_key, state_metadata)
        else:
            state_metadata = attn_ctx
            k_cache_metadata = attn_ctx

        if state_metadata is not None:
            state_req_dist = state_metadata.request_distribution
            state_q_start_loc = state_metadata.query_start_loc
            state_seq_lens = state_metadata.seq_lens
            state_block_tables = state_metadata.block_tables
            if state_block_tables is not None:
                state_block_tables = state_block_tables.flatten()
        else:
            state_req_dist = state_q_start_loc = state_seq_lens = state_block_tables = None

        # `k_block_tables` addresses the shared main/compressed-KV buffer
        # (self.k_cache.kv_cache) -- the same page-table space the outer
        # attention layer's own reads use -- which is why it must come from
        # k_cache's own metadata, not the compressor's state_cache metadata.
        if k_cache_metadata is not None:
            k_block_tables = k_cache_metadata.block_tables
            if k_block_tables is not None:
                k_block_tables = k_block_tables.flatten()
        else:
            k_block_tables = None

        # During the pre-allocation profiling/dummy forward pass, no KV cache
        # has been bound to k_cache yet; there is nothing to compress/store.
        if getattr(self.k_cache, "kv_cache", None) is None:
            return

        orig_cache = self.k_cache.kv_cache
        state_cache_buf = self.state_cache.kv_cache
        updated_cache, updated_state_cache_buf = self.compressor_op(
            kv_score,
            self.ape.clone(),
            self.norm.weight.clone(),
            rotary_emb.cos_sin_cache,
            positions,
            state_req_dist,
            state_q_start_loc,
            state_seq_lens,
            state_block_tables,
            k_block_tables,
            state_cache_buf,
            orig_cache,
        )

        orig_cache.copy_(updated_cache)
        if state_cache_buf.numel() > 0:
            state_cache_buf.copy_(updated_state_cache_buf)
