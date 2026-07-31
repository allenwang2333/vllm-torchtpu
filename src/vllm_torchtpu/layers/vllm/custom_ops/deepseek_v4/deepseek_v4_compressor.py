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
import os

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
            if (os.getenv("DSV4_DEBUG_COMPRESSOR") == "1"
                    and prefix_key is not None and "layers.2." in prefix_key):
                print(
                    f"[COMPRESSOR_META_DEBUG] prefix_key={prefix_key!r} "
                    f"prefix_key_found={prefix_key in attn_ctx} "
                    f"k_cache_prefix_key={k_cache_prefix_key!r} "
                    f"k_cache_prefix_found={k_cache_prefix_key in attn_ctx} "
                    f"attn_ctx_keys={[k for k in attn_ctx if 'layers.2.' in str(k)]}",
                    flush=True)
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

        if (os.getenv("DSV4_DEBUG_COMPRESSOR") == "1"
                and isinstance(attn_ctx, dict) and prefix_key is not None
                and "layers.2." in prefix_key):
            num_valid_reqs = state_req_dist[2].item(
            ) if state_req_dist is not None else None
            num_valid_tokens = (state_q_start_loc[num_valid_reqs].item()
                                if state_q_start_loc is not None
                                and num_valid_reqs is not None else None)
            print(
                f"[COMPRESSOR_META_DEBUG] positions={positions[:8].tolist()} "
                f"state_req_dist={state_req_dist.tolist() if state_req_dist is not None else None} "
                f"state_q_start_loc={state_q_start_loc.tolist() if state_q_start_loc is not None else None} "
                f"num_valid_reqs={num_valid_reqs} num_valid_tokens={num_valid_tokens}",
                flush=True)

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

        # The compressor kernels treat both of these as raw *byte* buffers:
        # compress_norm_rope_store packs `[fp8 nope | bf16 rope | e8m0 scale]`
        # into uint8 records, and pack_state_cache writes f32 state through
        # `_to_byte_lane` (also uint8). But vLLM allocates the underlying
        # tensors as `kv_cache_dtype` (torch.float8_e4m3fn here), and a JAX
        # `.at[...].set(uint8_values)` into an fp8-typed array performs a
        # NUMERIC CONVERSION, not a bitcast -- byte 108 becomes the fp8
        # *number* 108.0, with an entirely different bit pattern. Verified:
        # setting bytes [108,119,112,109] into an fp8 array reads back as
        # [110,111,110,110].
        #
        # That silently destroyed every record the compressor wrote. The state
        # cache then decoded back as ~7.87e31, which overflows RMSNorm's
        # variance to inf -> rsqrt(inf)=0 -> an all-zero compressed vector ->
        # an empty KV record. The attention path never hit this because it
        # routes through `_as_kernel_cache_view`, which applies exactly this
        # `.view(torch.uint8)`; the compressor was the one path that passed the
        # raw fp8-typed tensors straight through.
        #
        # `.view()` shares storage, so copying back through these views writes
        # the same memory the caches are bound to.
        orig_cache = self.k_cache.kv_cache
        state_cache_buf = self.state_cache.kv_cache
        if orig_cache is not None and orig_cache.dtype != torch.uint8:
            orig_cache = orig_cache.view(torch.uint8)
        if (state_cache_buf is not None
                and state_cache_buf.dtype != torch.uint8):
            state_cache_buf = state_cache_buf.view(torch.uint8)

        debug = (os.getenv("DSV4_DEBUG_COMPRESSOR") == "1"
                 and "layers.2." in str(k_cache_prefix_key)
                 and getattr(self, "head_dim", None) == 512
                 and orig_cache.numel() > 0)

        if debug:
            # kv_score is the op's primary data input and the only one never
            # inspected. The written record is provably not a
            # compress_norm_rope product (RMSNorm bounds |normed| to
            # ~sqrt(512)*max|w| ~= 22.6 and cos_sin is in [-1,1], so 2**97 is
            # unreachable; A1 showed the byte pattern is unreachable from
            # quantize_fp8_ue8m0; A5 showed the real function is healthy on
            # both CPU and TPU). So the corruption must enter through the
            # inputs -- and kv_score feeds both save_partial_states and, via
            # the state cache, the gather window.
            _ks = kv_score.detach().to(torch.float32)
            _t0 = _ks[0]
            print(
                f"[COMPRESSOR_KVSCORE] positions[0]={int(positions[0].item())} "
                f"shape={list(kv_score.shape)} dtype={kv_score.dtype} "
                f"tok0[absmax={_t0.abs().max().item():.6g} "
                f"nan={int(torch.isnan(_t0).sum().item())} "
                f"inf={int(torch.isinf(_t0).sum().item())} "
                f"nonzero={int((_t0 != 0).sum().item())}/{_t0.numel()}] "
                f"all[absmax={_ks.abs().max().item():.6g} "
                f"nan={int(torch.isnan(_ks).sum().item())} "
                f"inf={int(torch.isinf(_ks).sum().item())}]",
                flush=True)

            # Read the target row's bytes BEFORE the op. CPU experiment A1
            # (scratch/cpu_repro/a1_quantize_reachability.py) proved that the
            # observed row-259 signature -- all-zero fp8 bytes together with
            # all seven scale bytes == 109 -- is UNREACHABLE from any input to
            # quantize_fp8_ue8m0 (all-zero input gives byte 105; byte 109
            # requires amax ~1e-3, which always yields nonzero fp8). So the row
            # is not a single coherent record, and the "one atomic
            # .at[dest].set(record)" assumption is false.
            #
            # This splits the remaining possibilities:
            #   before == after  -> the compressor's scatter never lands, and
            #                       the bytes are residue from an earlier step
            #                       (note layer 2's SWA cache IS this same
            #                       tensor, and forward_mqa does
            #                       orig_sw_cache.copy_(new_sw_cache) over the
            #                       whole buffer after attention).
            #   before clean     -> the scatter itself produces the bad row.
            # Deliberately NOT a clone-diff (see bisection report 6.2 trap 3);
            # this reads one known address and prints its bytes.
            _cr = 4
            _kbs = orig_cache.shape[1] * orig_cache.shape[2]
            _mns = state_seq_lens.shape[0]
            _mkb = k_block_tables.shape[0] // _mns
            _p0 = int(positions[0].item())
            _bn = ((_p0 // _cr) // _kbs) % _mkb
            _slot = int(k_block_tables[_bn].item()) * _kbs + (
                (_p0 // _cr) % _kbs)
            _u8 = (orig_cache.view(torch.uint8)
                   if orig_cache.dtype != torch.uint8 else orig_cache)
            _row = _u8.reshape(-1, _u8.shape[-1])[_slot].detach().cpu().numpy()
            print(
                f"[COMPRESSOR_ROW_BEFORE] positions[0]={_p0} slot={_slot} "
                f"nz[nope/rope/scale/pad]={int((_row[:448] != 0).sum())}/"
                f"{int((_row[448:576] != 0).sum())}/"
                f"{int((_row[576:583] != 0).sum())}/"
                f"{int((_row[583:640] != 0).sum())} "
                f"scale_bytes={_row[576:583].tolist()}",
                flush=True)

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

        if debug:
            # No before/after diffing (torch/XLA buffer donation + async
            # execution made that unreliable -- two different snapshot
            # techniques gave two different, mutually-inconsistent answers).
            # Instead directly recompute token 0's (the real token) own
            # kv_slot_mapping in torch, mirroring _deepseek_v4_compressor_jax's
            # exact formula, and dequantize *that* row via the same
            # unpack_sparse_kv_cache() helper the passing kernel unit test
            # (tests/kernels/test_dsv4_compressor.py) already trusts -- a
            # genuine floating-point NaN check, not a byte-pattern guess.
            import jax.numpy as jnp
            import numpy as np

            from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
                unpack_sparse_kv_cache
            compress_ratio = 4
            k_block_size = orig_cache.shape[1] * orig_cache.shape[2]
            max_num_seqs = state_seq_lens.shape[0]
            max_num_k_blocks_per_req = k_block_tables.shape[0] // max_num_seqs
            pos0 = int(positions[0].item())
            k_cache_block_num = ((pos0 // compress_ratio) //
                                 k_block_size) % max_num_k_blocks_per_req
            k_cache_block_offset = (pos0 // compress_ratio) % k_block_size
            kv_slot0 = int(
                k_block_tables[k_cache_block_num].item()) * k_block_size + \
                k_cache_block_offset
            # Dump the state-cache rows that gather_state_windows reads for
            # this boundary window (positions pos0-compress_ratio+1 .. pos0),
            # mirroring _deepseek_v4_compressor_jax's slot_mapping formula.
            # The raw-byte probe showed compressed slot 3 is *written* but with
            # an essentially empty payload (0/448 nope bytes, 5/128 rope bytes)
            # rather than corrupted data -- i.e. the compressor had nothing to
            # compress. Slots 0-2 are healthy because their whole window lives
            # inside the single prefill call; slot 3's window straddles the
            # prefill->decode boundary and is the only one needing state to
            # persist across calls. If positions 12-14 read back as all-zero
            # here, the state simply is not persisting.
            # NOTE on addressing: `slot_mapping` indexes the *state view*
            # ([num_pages, state_block_size, state_dim]), NOT the raw byte
            # buffer ([num_pages, rows, packing, width]). Those are different
            # flattenings -- a naive reshape(-1, width) lands on the wrong
            # page entirely. Mirror unpack_state_cache's own indexing (via its
            # _state_chunk_dims helper) rather than re-deriving it.
            from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
                _state_chunk_dims

            # ROOT CAUSE (offline recompute): the gathered state window reads
            # back at ~7.87e31, which overflows RMSNorm's variance to inf ->
            # rsqrt(inf)=0 -> normed=0 -> an all-zero compressed vector -> an
            # empty record that destroys the healthy row at 259.
            # Reading fp8/KV-style bytes (~100-120) as f32 lands exactly in the
            # 1e26-1e31 range, so the state buffer is very likely aliased onto
            # a real KV buffer. Compare data_ptrs to find out.
            _sp = state_cache_buf.data_ptr()
            _kp = orig_cache.data_ptr()
            _sn = state_cache_buf.numel() * state_cache_buf.element_size()
            _kn = orig_cache.numel() * orig_cache.element_size()
            _overlap = (_sp < _kp + _kn) and (_kp < _sp + _sn)
            print(
                f"[COMPRESSOR_ALIAS_CHECK] positions[0]={pos0} "
                f"state_ptr={_sp} state_bytes={_sn} "
                f"kv_ptr={_kp} kv_bytes={_kn} "
                f"same_ptr={_sp == _kp} ranges_overlap={_overlap} "
                f"state_shape={list(state_cache_buf.shape)} "
                f"state_dtype={state_cache_buf.dtype}",
                flush=True)

            _sb = int(self.state_cache.block_size)
            _max_state_blocks = state_block_tables.shape[0] // max_num_seqs
            _st = state_cache_buf
            if _st.ndim == 5:
                _st = _st.squeeze(2)
            _st_u8 = _st.view(torch.uint8) if _st.dtype != torch.uint8 else _st
            _coff = 2 if self.overlap else 1
            _state_dim = 2 * _coff * self.head_dim
            _kv_slots, _rpt, _fpr, _bpr = _state_chunk_dims(
                tuple(_st_u8.shape), _sb, _state_dim)
            _npages = _st_u8.shape[0]
            _width = _st_u8.shape[-1]
            _slots_view = _st_u8.reshape(_npages, _kv_slots, _width)
            # The gather window is coff*compress_ratio wide (8 for CSA, since
            # overlap=True), NOT compress_ratio -- w_idx < compress_ratio
            # selects head-block 0 and the rest select block 1, so the two
            # halves of the window live in different state blocks/pages.
            _window = _coff * compress_ratio
            _win = []
            for _p in range(max(0, pos0 - _window + 1), pos0 + 1):
                _bn = (_p // _sb) % _max_state_blocks
                _slot = int(state_block_tables[_bn].item()) * _sb + (_p % _sb)
                _pg, _off = _slot // _sb, _slot % _sb
                if 0 <= _pg < _npages:
                    _chunk = _slots_view[_pg,
                                         _off * _rpt:(_off + 1) * _rpt, :_bpr]
                    _nz = int((_chunk != 0).sum().item())
                    # Per-head-block breakdown. unpack_state_cache reshapes
                    # this chunk to [rpt, fpr] and flattens to state_dim, so
                    # chunk row r is exactly head-block r -- and
                    # save_partial_states packs the row as
                    # [kv_a | kv_b | score_a | score_b]. Whole-row nz came out
                    # at ~6120/8192 == 3/4, which means one of these four
                    # blocks is entirely zero; this says which.
                    _blk = [
                        int((_chunk[_r] != 0).sum().item())
                        for _r in range(_rpt)
                    ]
                else:
                    _nz = -1
                    _blk = []
                _win.append(
                    f"pos{_p}->slot{_slot}(pg{_pg},off{_off}):nz={_nz} "
                    f"blk[kv_a/kv_b/sc_a/sc_b]={'/'.join(map(str, _blk))}")
            print(
                f"[COMPRESSOR_STATE_WINDOW] positions[0]={pos0} "
                f"state_cache_buf.shape={list(state_cache_buf.shape)} "
                f"state_block_size={_sb} max_state_blocks={_max_state_blocks} "
                f"state_bt[:6]={state_block_tables[:6].tolist()} | " +
                " | ".join(_win),
                flush=True)

            updated_cache_u8 = (updated_cache.view(torch.uint8)
                                if updated_cache.dtype != torch.uint8 else
                                updated_cache)
            width = updated_cache_u8.shape[-1]
            flat_after = updated_cache_u8.reshape(-1, width)
            row0 = flat_after[kv_slot0:kv_slot0 + 1].detach().cpu().numpy()
            row0_jax = jnp.asarray(row0)
            nope_dim = 448
            rope_head_dim = 64
            quant_block = 64
            nope, rope, scale = unpack_sparse_kv_cache(row0_jax, nope_dim,
                                                       rope_head_dim,
                                                       quant_block)
            nope_f32 = np.asarray(nope.astype(jnp.float32))
            rope_f32 = np.asarray(rope.astype(jnp.float32))
            # Exact nonzero byte map of the written row. Standalone CPU *and*
            # TPU runs of compress_norm_rope_store both produce a healthy
            # 448/128/7 record (scratch/cpu_repro/a5_store_cpu_vs_tpu.py), so
            # the kernel and its TPU lowering are both exonerated -- the row
            # written in the real run is malformed in a way the function
            # cannot produce. The byte positions say whether the record is
            # shifted (nope fp8 data landing in the rope/scale regions: note
            # 0x70/0x74 are ordinary fp8_e4m3 values, and reading them as bf16
            # pairs yields exactly the observed 1e29/1e31) or something else.
            _rowb = row0.reshape(-1).astype(np.uint8)
            _nzi = np.nonzero(_rowb)[0]
            print(
                f"[COMPRESSOR_ROW_BYTEMAP] positions[0]={pos0} "
                f"slot={kv_slot0} total_nonzero={_nzi.size} "
                f"first32_idx={_nzi[:32].tolist()} "
                f"first32_val={_rowb[_nzi[:32]].tolist()}",
                flush=True)
            # Recompute token 0's record OFFLINE from the REAL inputs and
            # compare with what the op actually wrote. Every component is now
            # individually verified correct (kv_score healthy; gather correct;
            # compress_norm_rope_store byte-identically healthy on CPU AND TPU
            # at real geometry; pack/unpack round-trip lossless) -- yet the
            # integrated op emits a 12-nonzero-byte record. This isolates the
            # op wrapper (pallas.jax_op + shard_map + donation) from the math.
            try:
                from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import (
                    _state_chunk_dims, compress_norm_rope, quantize_fp8_ue8m0)
                _coff = 2 if self.overlap else 1
                _sdim = 2 * _coff * self.head_dim
                _sb = int(self.state_cache.block_size)
                _mnsb = state_block_tables.shape[0] // max_num_seqs
                _stb = (state_cache_buf.squeeze(2)
                        if state_cache_buf.ndim == 5 else state_cache_buf)
                _stb8 = (_stb.view(torch.uint8)
                         if _stb.dtype != torch.uint8 else _stb)
                _ks_, _rpt_, _fpr_, _bpr_ = _state_chunk_dims(
                    tuple(_stb8.shape), _sb, _sdim)
                _sv = _stb8.reshape(_stb8.shape[0], _ks_, _stb8.shape[-1])
                _win = _coff * compress_ratio
                _rows = []
                for _p in range(pos0 - _win + 1, pos0 + 1):
                    if _p == pos0:
                        _kvs = kv_score[0].detach().to(torch.float32).cpu()
                        _sw = _coff * self.head_dim
                        _apr = self.ape[_p % compress_ratio].detach().to(
                            torch.float32).cpu()
                        _rows.append(
                            torch.cat([_kvs[:_sw],
                                       _kvs[_sw:2 * _sw] + _apr]).numpy())
                    elif _p < 0:
                        _rows.append(np.zeros(_sdim, dtype=np.float32))
                    else:
                        _b = (_p // _sb) % _mnsb
                        _s = int(state_block_tables[_b].item()) * _sb + (_p %
                                                                         _sb)
                        _pg, _of = _s // _sb, _s % _sb
                        _ch = _sv[_pg, _of * _rpt_:(_of + 1) * _rpt_, :_bpr_]
                        # [rpt, bpr] uint8 -> rpt*bpr bytes -> state_dim f32,
                        # matching unpack_state_cache's own reshape order.
                        _by = np.ascontiguousarray(
                            _ch.detach().cpu().numpy().reshape(-1))
                        _rows.append(_by.view(np.float32))
                _w = jnp.asarray(np.stack(_rows)[None])  # [1, win, state_dim]
                _wr = _w.reshape(1, _win, 4, self.head_dim)
                _wc = (jnp.arange(_win) < compress_ratio)[None, :, None]
                _kvw = jnp.where(_wc, _wr[:, :, 0, :], _wr[:, :, 1, :])
                _scw = jnp.where(_wc, _wr[:, :, 2, :], _wr[:, :, 3, :])
                _cmp = compress_norm_rope(
                    kv_window=_kvw,
                    score_window=_scw,
                    valid_mask=jnp.ones((1, _win), dtype=bool),
                    rms_weight=jnp.asarray(self.norm.weight.detach().to(
                        torch.float32).cpu().numpy()),
                    cos_sin_cache=jnp.asarray(
                        rotary_emb.cos_sin_cache.detach().to(
                            torch.float32).cpu().numpy()),
                    compressed_pos=jnp.asarray([
                        (pos0 // compress_ratio) * compress_ratio
                    ]),
                    rms_eps=self.rms_norm_eps,
                    rope_head_dim=self.rope_head_dim)
                _c = np.asarray(_cmp)[0]
                _nd = self.head_dim - self.rope_head_dim
                _q, _sc = quantize_fp8_ue8m0(_cmp[:, :_nd], self._quant_block)
                print(
                    f"[COMPRESSOR_OFFLINE_RECOMPUTE] positions[0]={pos0} "
                    f"window={_win} "
                    f"win_absmax={float(np.abs(np.stack(_rows)).max()):.6g} "
                    f"compressed[nope_absmax={np.abs(_c[:_nd]).max():.6g} "
                    f"rope_absmax={np.abs(_c[_nd:]).max():.6g}] "
                    f"q_nonzero={int(np.count_nonzero(np.asarray(jnp.asarray(_q).view(jnp.uint8))))}/{_nd} "
                    f"scale_bytes={np.asarray(_sc).reshape(-1).tolist()}",
                    flush=True)
            except Exception as _e:  # diagnostic only; never break the run
                print(f"[COMPRESSOR_OFFLINE_RECOMPUTE] failed: {_e!r}",
                      flush=True)

            print(
                f"[COMPRESSOR_DEQUANT_DEBUG] positions[0]={pos0} kv_slot0={kv_slot0} "
                f"nope_nan_count={int(np.isnan(nope_f32).sum())}/{nope_f32.size} "
                f"nope_absmax={float(np.nanmax(np.abs(nope_f32))) if not np.all(np.isnan(nope_f32)) else float('nan')} "
                f"rope_nan_count={int(np.isnan(rope_f32).sum())}/{rope_f32.size} "
                # rope_absmax is the decisive number: the attention-side probe
                # (DSV4_DEBUG_MAIN_KV) sees rope~1.585e+29 in this same row at
                # read time, but this print previously reported only
                # rope_nan_count, so it was never known whether the compressor
                # itself emitted that value or whether a later writer into the
                # shared main/SWA buffer overwrote the row's rope bytes after
                # this point. nope_absmax=0 alongside scale=2^-18 is internally
                # inconsistent for a single coherent write (that scale implies a
                # quantized max of ~224-448, not 0), which is what suggests two
                # different writers touching one row.
                f"rope_absmax={float(np.nanmax(np.abs(rope_f32))) if not np.all(np.isnan(rope_f32)) else float('nan')} "
                f"scale_min={float(np.asarray(scale).min())} scale_max={float(np.asarray(scale).max())}",
                flush=True)

        if debug:
            # The op's SECOND output is what poisons the state buffer if the
            # wrapper (pallas.jax_op / shard_map / donation) is at fault:
            # state_cache_buf.copy_() faithfully stores whatever comes back, so
            # a corrupt updated_state_cache_buf at prefill is read as ~7.87e31
            # by the decode-step gather. Compare the buffer's own state slots
            # before and after this copy, decoded as f32 through the kernel's
            # real geometry (never hand-derived -- see the addressing trap in
            # bisection report 8.3).
            def _slot_absmax(_buf, _slot):
                _b = _buf.squeeze(2) if _buf.ndim == 5 else _buf
                _b8 = _b.view(torch.uint8) if _b.dtype != torch.uint8 else _b
                _ks, _rp, _fp, _bp = _state_chunk_dims(
                    tuple(_b8.shape), int(self.state_cache.block_size),
                    2 * (2 if self.overlap else 1) * self.head_dim)
                _sv = _b8.reshape(_b8.shape[0], _ks, _b8.shape[-1])
                _pg, _of = _slot // int(self.state_cache.block_size), _slot % \
                    int(self.state_cache.block_size)
                _ch = _sv[_pg, _of * _rp:(_of + 1) * _rp, :_bp]
                _f = np.ascontiguousarray(
                    _ch.detach().cpu().numpy().reshape(-1)).view(np.float32)
                return float(np.abs(_f).max()), int((_f != 0).sum())

            try:
                from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
                    _state_chunk_dims
                _sbz = int(self.state_cache.block_size)
                _mnsb2 = state_block_tables.shape[0] // state_seq_lens.shape[0]
                _bn2 = (pos0 // _sbz) % _mnsb2
                _slot2 = int(state_block_tables[_bn2].item()) * _sbz + (pos0 %
                                                                        _sbz)
                _pre = _slot_absmax(state_cache_buf, _slot2)
                _new = _slot_absmax(updated_state_cache_buf, _slot2)
                print(
                    f"[COMPRESSOR_STATE_OUT] positions[0]={pos0} "
                    f"slot={_slot2} buf_before[absmax={_pre[0]:.6g} "
                    f"nonzero={_pre[1]}] op_output[absmax={_new[0]:.6g} "
                    f"nonzero={_new[1]}]",
                    flush=True)
            except Exception as _e2:
                print(f"[COMPRESSOR_STATE_OUT] failed: {_e2!r}", flush=True)

        orig_cache.copy_(updated_cache)
        if state_cache_buf.numel() > 0:
            state_cache_buf.copy_(updated_state_cache_buf)
