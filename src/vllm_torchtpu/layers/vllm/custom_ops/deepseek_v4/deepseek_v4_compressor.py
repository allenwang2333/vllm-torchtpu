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
from torch_tpu._internal import pallas
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

logger = init_logger(__name__)


def _is_tracing() -> bool:
    if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
        return True
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return True
    return False


_compressor_op_cache = {}

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
) -> jax.Array:

    forward_fn = (compressor_forward
                  if head_dim == 512 else compressor_forward_indexer)

    def _compressor_local(kv_score, ape, norm_weight, cos_sin_cache, positions,
                          request_distribution, query_start_loc, seq_lens,
                          state_block_tables, k_block_tables, cache):
        num_valid_reqs = request_distribution[2]
        num_tokens = kv_score.shape[0]
        num_valid_tokens = query_start_loc[num_valid_reqs]

        q_per_req = query_start_loc[1:] - query_start_loc[:-1]
        max_num_seqs = seq_lens.shape[0]
        token_to_req_indices = jnp.repeat(jnp.arange(max_num_seqs),
                                          q_per_req,
                                          total_repeat_length=num_tokens)

        # Compressor state and compressed KV overlay the same physical
        # `cache` buffer, but are addressed by two independent block tables
        # -- `state_block_tables` (from the state-cache "layer"'s own vLLM
        # metadata) at `state_block_size` granularity, `k_block_tables`
        # (from the k-cache "layer"'s metadata) at full-page granularity.
        # vLLM's hybrid-KV-cache-group allocator is what keeps these two
        # logical layers' page-index ranges disjoint within the one shared
        # tensor; this mirrors tpu-inference's reference design exactly
        # (confirmed via a real-hardware `layer_name_to_kvcache_index`
        # check: state and k-cache land on the same tensor, same shape, for
        # every CSA/HCA/indexer layer).
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
        for arg in (kv_score, ape, norm_weight, cos_sin_cache, positions,
                    request_distribution, query_start_loc, seq_lens,
                    state_block_tables, k_block_tables):
            if arg is not None:
                dummy_sum = dummy_sum + jnp.sum(arg) * 0

        return out + dummy_sum.astype(out.dtype)

    data_spec = get_valid_spec(mesh, None)
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
        P(),  # cache
    )
    out_specs = P()  # updated cache

    if cache.shape[0] == 0:
        dummy_sum = 0.0
        for arg in (kv_score, ape, norm_weight, cos_sin_cache, positions,
                    request_distribution, query_start_loc, seq_lens,
                    state_block_tables, k_block_tables):
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
    )(kv_score, ape, norm_weight, cos_sin_cache, positions,
      request_distribution, query_start_loc, seq_lens, state_block_tables,
      k_block_tables, cache)

    if cache_was_5d:
        updated_cache = jnp.expand_dims(updated_cache, axis=2)
    return updated_cache


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
        self.compress_ratio = compress_ratio
        self._state_coff = coff
        # On TPU, cache_block_size is always 1024. Initialize self.block_size
        # directly at construction time so self.compressor_op can be pre-built
        # cleanly before profile_run / determine_available_memory.
        self.block_size = self._state_block_size(1024)

    def _state_block_size(self, cache_block_size: int) -> int:
        """Rows of compressor state packed into one host compressed-KV page."""
        compressed_kv_cache_bz = cache_block_size // self.compress_ratio
        # 4: bytes per f32. 2: kv-dim + score-dim.
        block_size = compressed_kv_cache_bz // 4 // 2 // self._state_coff
        if block_size <= 0:
            # Degenerates below block_size=1024: at 512 an HCA host page is 4
            # rows and this floors to 0. Fail loudly rather than emit a
            # zero-sized page.
            raise ValueError(
                f"DeepSeek-V4 compressor state cache {self.prefix!r} computed "
                f"a non-positive block size ({block_size}) from cache block "
                f"size {cache_block_size} and compress_ratio "
                f"{self.compress_ratio}: the compressed-KV host page "
                f"({compressed_kv_cache_bz} rows) is too small to pack this "
                "layer's compressor state into.")
        return block_size

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
        # kernels/deepseek_v4/compress_norm_rope.py).
        #
        # This spec still describes its own real KV-cache-group (own
        # block_size, own page-index allocation -- vLLM's scheduler handles
        # that exactly as it would for any other attention layer). It is
        # NOT what makes this cache overlay the main compressed-KV tensor,
        # though -- page-size-bucket matching between this spec and the
        # main spec is *not* how tpu-inference achieves that overlay either
        # (confirmed by reading its runner: `get_kv_cache_spec` is a
        # documented no-op there). The actual physical-tensor overlay here
        # is done by `TPUModelRunner._alias_ds_v4_compressor_state_caches`
        # (`runner/tpu_runner.py`), a post-allocation step that swaps this
        # layer's tensor for its host's, after this spec's own group/
        # metadata machinery has already run normally.
        self.block_size = self._state_block_size(
            vllm_config.cache_config.block_size)
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
        # NOTE: `compressor_op` is deliberately NOT built here -- see the
        # property below.
        self.num_layers = get_current_vllm_config(
        ).model_config.hf_config.num_hidden_layers
        # The state-cache/compressed-KV alias is checked on this layer's first
        # forward only; see `forward()`. It is decided at
        # `initialize_kv_cache` and cannot change per step, and the check
        # itself costs a host sync.
        self._alias_checked = False

    @property
    def compressor_op(self):
        """Built on first use, not in `__init__`.

        `_build_compressor_op` bakes `state_block_size` into the traced op as a
        static constant. At construction time that attribute still holds the
        base `CompressorStateCache.__init__` value (4 for CSA / 8 for HCA),
        which is derived from the CUDA layout and assumes a 64-row host page;
        `VllmCompressorStateCache.get_kv_cache_spec` only replaces it with the
        value this host page can actually take (16 / 1) later, during KV cache
        initialization.

        Baking the stale 8 in gave `state row needs 4096B but cache width is
        640B` on HCA layers -- with the correct 1 the same packing needs 512B
        and fits. Same failure mode, same fix as
        `VllmDeepseekV4MLAAttention.attn_op`.
        """
        op = self.__dict__.get("_compressor_op_instance")
        if op is None:
            op = self._build_compressor_op()
            object.__setattr__(self, "_compressor_op_instance", op)
        return op

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
                P(),  # cache
            ),
        )

        def _fake_compressor(kv_score, ape, norm_weight, cos_sin_cache,
                             positions, request_distribution, query_start_loc,
                             seq_lens, state_block_tables, k_block_tables,
                             cache, *args, **kwargs):
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
                    and not _is_tracing() and prefix_key is not None
                    and "layers.2." in prefix_key):
                logger.debug(
                    f"[COMPRESSOR_META_DEBUG] prefix_key={prefix_key!r} "
                    f"prefix_key_found={prefix_key in attn_ctx} "
                    f"k_cache_prefix_key={k_cache_prefix_key!r} "
                    f"k_cache_prefix_found={k_cache_prefix_key in attn_ctx} "
                    f"attn_ctx_keys={[k for k in attn_ctx if 'layers.2.' in str(k)]}"
                )
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

        if (os.getenv("DSV4_DEBUG_COMPRESSOR") == "1" and not _is_tracing()
                and isinstance(attn_ctx, dict) and prefix_key is not None
                and "layers.2." in prefix_key):
            num_valid_reqs = state_req_dist[2].item(
            ) if state_req_dist is not None else None
            num_valid_tokens = (state_q_start_loc[num_valid_reqs].item()
                                if state_q_start_loc is not None
                                and num_valid_reqs is not None else None)
            logger.debug(
                f"[COMPRESSOR_META_DEBUG] positions={positions[:8].tolist()} "
                f"state_req_dist={state_req_dist.tolist() if state_req_dist is not None else None} "
                f"state_q_start_loc={state_q_start_loc.tolist() if state_q_start_loc is not None else None} "
                f"num_valid_reqs={num_valid_reqs} num_valid_tokens={num_valid_tokens}"
            )

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
        state_cache_tensor = self.state_cache.kv_cache
        if orig_cache is not None and orig_cache.dtype != torch.uint8:
            orig_cache = orig_cache.view(torch.uint8)
        if (state_cache_tensor is not None
                and state_cache_tensor.dtype != torch.uint8):
            state_cache_tensor = state_cache_tensor.view(torch.uint8)

        # Compressor state and compressed KV overlay one physical tensor --
        # `TPUModelRunner._alias_ds_v4_compressor_state_caches` points this
        # layer's cache at its host's tensor object after normal allocation
        # -- so only `orig_cache` is passed to the kernel. `.data_ptr()`
        # (not `is` -- `.view()` above always returns a fresh Tensor object
        # even when storage is unchanged) guards that the alias actually
        # took effect for this layer rather than assuming it, mirroring
        # tpu-inference's own unconditional `assert state_cache_index ==
        # cache_index` in its compressor forward().
        #
        # Checked ONCE per layer, not per step. `.data_ptr()` materializes a
        # lazy tensor and so costs a host sync, and this ran on all 41
        # compressor layers of every decode step. The aliasing it guards is
        # decided once, at `initialize_kv_cache`, and cannot change between
        # steps -- so a per-step check bought nothing and cut the lazy graph 41
        # times a step. Deleting it outright was the other option and is worse:
        # this invariant has silently corrupted state here before.
        if (not self._alias_checked and orig_cache is not None
                and state_cache_tensor is not None and orig_cache.numel() > 0
                and not _is_tracing()):
            self._alias_checked = True
            assert orig_cache.data_ptr() == state_cache_tensor.data_ptr(), (
                "compressor state cache and compressed-KV cache did not "
                "land on the same physical tensor -- the runner-level DSV4 "
                "alias did not take effect for this layer; writing state "
                "through orig_cache alone would silently corrupt or orphan "
                "data")

        debug = (os.getenv("DSV4_DEBUG_COMPRESSOR") == "1"
                 and "layers.2." in str(k_cache_prefix_key)
                 and getattr(self, "head_dim", None) == 512
                 and orig_cache.numel() > 0)

        # `DSV4_DEBUG_GEOM=1` dumps every geometry input to the compressor,
        # including values we believe we already know. The state-cache slot
        # mapping wraps (`(positions // state_block_size) %
        # max_num_state_blocks_per_req`, see the jax path in this file), and
        # whether that wrap ever fires is a function of numbers none of which
        # have been measured -- they have only been derived from source. Print
        # them rather than infer them.
        if os.getenv("DSV4_DEBUG_GEOM") == "1" and not _is_tracing():

            def _t(x, n=96):
                if x is None:
                    return "None"
                try:
                    flat = x.flatten()
                    return (f"shape={tuple(x.shape)} dtype={x.dtype} "
                            f"vals[:{n}]={flat[:n].tolist()}")
                except Exception as exc:
                    return f"<unprintable: {exc}>"

            try:
                _mns = None
                for _obj in (getattr(self, "cache_config", None),
                             getattr(self.state_cache, "cache_config", None)):
                    _mns = getattr(_obj, "max_num_seqs", None) or _mns
                if _mns is None:
                    from vllm.config import get_current_vllm_config
                    _mns = get_current_vllm_config().scheduler_config.\
                        max_num_seqs
                _sbs = getattr(self, "state_block_size", None)
                _nblk = (state_block_tables.shape[0] // _mns
                         if state_block_tables is not None and _mns else None)
                _thresh = (_nblk * _sbs if _nblk and _sbs else None)
                _pmax = int(positions.max()) if positions.numel() else -1
                logger.info(
                    "[DSV4_GEOM] layer=%s\n"
                    "  positions: %s min=%s max=%s\n"
                    "  state_block_tables: %s\n"
                    "  k_block_tables: %s\n"
                    "  state_seq_lens: %s\n"
                    "  state_q_start_loc: %s\n"
                    "  state_req_dist: %s\n"
                    "  state_block_size=%s k_block_size=%s compress_ratio=%s "
                    "head_dim=%s max_num_seqs=%s\n"
                    "  max_num_state_blocks_per_req=%s  WRAP_THRESHOLD=%s  "
                    "positions.max=%s  WRAPS=%s\n"
                    "  orig_cache: %s\n"
                    "  state_cache_tensor: %s  same_ptr=%s",
                    k_cache_prefix_key,
                    _t(positions),
                    int(positions.min()) if positions.numel() else -1,
                    _pmax,
                    _t(state_block_tables),
                    _t(k_block_tables),
                    _t(state_seq_lens),
                    _t(state_q_start_loc),
                    _t(state_req_dist),
                    _sbs,
                    getattr(self, "k_block_size", None),
                    getattr(self, "compress_ratio", None),
                    getattr(self, "head_dim", None),
                    _mns,
                    _nblk,
                    _thresh,
                    _pmax,
                    (_thresh is not None and _pmax >= _thresh),
                    (f"shape={tuple(orig_cache.shape)} dtype={orig_cache.dtype}"
                     if orig_cache is not None else "None"),
                    (f"shape={tuple(state_cache_tensor.shape)}"
                     if state_cache_tensor is not None else "None"),
                    (orig_cache is not None and state_cache_tensor is not None
                     and orig_cache.data_ptr()
                     == state_cache_tensor.data_ptr()),
                )
            except Exception as exc:  # debug only: never fail a run
                logger.info("[DSV4_GEOM] dump failed: %s", exc)

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
            k_block_tables,
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
            logger.debug(
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
                f"scale_min={float(np.asarray(scale).min())} scale_max={float(np.asarray(scale).max())}"
            )

        if (orig_cache is not None and updated_cache is not None
                and updated_cache.numel() > 0
                and orig_cache.shape == updated_cache.shape):
            orig_cache.copy_(updated_cache)
