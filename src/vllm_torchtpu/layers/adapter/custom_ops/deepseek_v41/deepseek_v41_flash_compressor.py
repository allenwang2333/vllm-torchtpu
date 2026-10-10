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
"""TPU Compressor and Indexer custom ops and layers for DeepSeek-V4.1."""

import functools
import inspect
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v41 import attention as dsv41_attention
from vllm.models.deepseek_v41 import compressor as dsv41_compressor
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import tp_all_gather, tp_shard_slice
from vllm_torchtpu.kernels.deepseek_v41.compressor import (
    compress_r2_pool_norm_jax,
    compressor_insert_cache_jax,
    indexer_produce_k_jax,
    indexer_v41_jax,
    indexer_v41_tp_jax,
    indexer_v41_tp_write_jax,
    indexer_v41_write_jax,
)
from vllm_torchtpu.kernels.deepseek_v41.streamindex_topk import (
    MAX_INDEXER_CHUNKS,
    compute_indexer_step_metadata_jax,
    compute_meta_buf_len,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    quantized_matmul_prequant,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import align_to
from vllm_torchtpu.kernels.deepseek_v41.compressor import (
    compress_r1_wkv_norm_jax,
    compress_r2_wkv_pool_norm_jax,
    fused_indexer_wq_b_rope_quant,
    indexer_produce_k_wk_norm_jax,
    wkv_wgate_f32_jax,
)

_pallas_op_cache: dict[str, Any] = {}
_R = P()


def _name_float(x: float) -> str:
    return f"{x:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _get_or_register_op(
    op_name: str,
    fn: Any,
    n_inputs: int,
    fake_fn: Any,
    donate_argnums: tuple[int, ...] = (),
):
    op = _pallas_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            fn,
            mesh=get_vllm_model_wrapper_context().mesh,
            donate_argnums=donate_argnums,
            input_partition_specs=(_R,) * n_inputs,
        )
        op.register_fake(fake_fn)
        _pallas_op_cache[op_name] = op
    return op


def _as_kernel_cache_view(cache: torch.Tensor | None) -> torch.Tensor | None:
    if cache is None or cache.numel() == 0:
        return None
    return (
        cache if cache.dtype in (torch.uint8, torch.int32) else cache.view(torch.uint8)
    )


class VllmCompressorV41StateCache(dsv41_compressor.CompressorStateCache):
    """TPU state cache for DeepSeek-V4.1 ratio-2 compressor."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        self.block_size = vllm_config.cache_config.block_size // 2 // 8
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(self.state_dim * 4, 128),
            dtype=torch.uint8,
            sliding_window=2,
            alignment=None,
        )


class VllmDeepseekV41Compressor(dsv41_compressor.DeepseekCompressor):
    """TPU implementation of DeepSeek-V4.1 Compressor (compress_ratio in {1, 2})."""

    def __init__(self, *args, **kwargs) -> None:
        orig_state_cache = dsv41_compressor.CompressorStateCache
        dsv41_compressor.CompressorStateCache = VllmCompressorV41StateCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv41_compressor.CompressorStateCache = orig_state_cache

        if self.has_gate:
            # Ratio-2 wkv and wgate are computed in float32 before softmax pooling.
            self.fused_wkv_wgate = MergedColumnParallelLinear(
                self.hidden_size,
                [self.head_dim, self.head_dim],
                bias=False,
                return_bias=False,
                quant_config=None,
                params_dtype=torch.float32,
                disable_tp=True,
                prefix=f"{self.prefix}.fused_wkv_wgate",
            )

    def _get_r2_pool_op(self):
        if (op := self.__dict__.get("_r2_pool_op")) is not None:
            return op
        assert self.state_cache is not None
        sb = self.state_cache.block_size
        op = _get_or_register_op(
            f"pallas::deepseek_v41_compress_r2_pool_sb{sb}_e{_name_float(self.rms_norm_eps)}",
            functools.partial(
                compress_r2_pool_norm_jax,
                state_block_size=sb,
                rms_eps=float(self.rms_norm_eps),
            ),
            7,
            lambda wkv_wgate, norm_weight, *args: (
                wkv_wgate.new_empty(
                    (wkv_wgate.shape[0], norm_weight.shape[0]), dtype=torch.bfloat16
                ),
                torch.empty_like(args[-1]),
            ),
            donate_argnums=(6,),
        )
        object.__setattr__(self, "_r2_pool_op", op)
        return op

    def _get_r2_wkv_f32_op(self):
        if (op := self.__dict__.get("_r2_wkv_f32_op")) is not None:
            return op
        op = _get_or_register_op(
            f"pallas::deepseek_v41_compress_r2_wkv_f32_d{self.head_dim}",
            wkv_wgate_f32_jax,
            2,
            lambda hidden_states, w_fused: hidden_states.new_empty(
                (
                    hidden_states.shape[0],
                    w_fused.shape[0]
                    if w_fused.shape[0] != hidden_states.shape[-1]
                    else w_fused.shape[1],
                ),
                dtype=torch.float32,
            ),
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "_r2_wkv_f32_op", op)
        return op

    def _get_r2_wkv_pool_op(self):
        if (op := self.__dict__.get("_r2_wkv_pool_op")) is not None:
            return op
        assert self.state_cache is not None
        sb = self.state_cache.block_size
        op = _get_or_register_op(
            f"pallas::deepseek_v41_compress_r2_wkv_pool_sb{sb}_e{_name_float(self.rms_norm_eps)}",
            functools.partial(
                compress_r2_wkv_pool_norm_jax,
                state_block_size=sb,
                rms_eps=float(self.rms_norm_eps),
            ),
            8,
            lambda hidden_states, w_fused, norm_weight, *args: (
                hidden_states.new_empty(
                    (hidden_states.shape[0], norm_weight.shape[0]),
                    dtype=torch.bfloat16,
                ),
                torch.empty_like(args[-1]),
            ),
            donate_argnums=(7,),
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "_r2_wkv_pool_op", op)
        return op

    def _get_r1_wkv_norm_op(self):
        if (op := self.__dict__.get("_r1_wkv_norm_op")) is not None:
            return op
        op = _get_or_register_op(
            f"pallas::deepseek_v41_compress_r1_wkv_norm_d{self.head_dim}_e{_name_float(self.rms_norm_eps)}",
            functools.partial(
                compress_r1_wkv_norm_jax,
                rms_eps=float(self.rms_norm_eps),
            ),
            3,
            lambda hidden_states, w_fused, norm_weight: hidden_states.new_empty(
                (hidden_states.shape[0], norm_weight.shape[0]),
                dtype=hidden_states.dtype,
            ),
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "_r1_wkv_norm_op", op)
        return op

    def _get_insert_cache_op(self):
        if (op := self.__dict__.get("_insert_cache_op")) is not None:
            return op
        op = _get_or_register_op(
            f"pallas::deepseek_v41_compress_insert_r{self.compress_ratio}",
            functools.partial(
                compressor_insert_cache_jax, compress_ratio=self.compress_ratio
            ),
            8,
            lambda *args: (torch.empty_like(args[6]), torch.empty_like(args[7])),
            donate_argnums=(6, 7),
        )
        object.__setattr__(self, "_insert_cache_op", op)
        return op

    def _build_compressor_op(self):
        for k in (
            "_r2_pool_op",
            "_r2_wkv_f32_op",
            "_r2_wkv_pool_op",
            "_r1_wkv_norm_op",
            "_insert_cache_op",
        ):
            self.__dict__.pop(k, None)
        r2_pool = self._get_r2_pool_op() if self.has_gate else None
        if self.has_gate:
            self._get_r2_wkv_f32_op()
            if self.state_cache is not None:
                self._get_r2_wkv_pool_op()
        else:
            self._get_r1_wkv_norm_op()
        return (r2_pool, self._get_insert_cache_op())

    compressor_op = property(_build_compressor_op)

    def _can_use_fused_wkv(self, hidden_states: torch.Tensor) -> bool:
        return (
            hidden_states.device.type != "cpu"
            and self.fused_wkv_wgate._parameters.get("weight") is not None
            and self.fused_wkv_wgate._parameters.get("bias") is None
            and self.norm._parameters.get("weight") is not None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = positions.shape[0]
        if not self.has_gate:
            if self._can_use_fused_wkv(hidden_states):
                latent = self._get_r1_wkv_norm_op()(
                    hidden_states,
                    self.fused_wkv_wgate.weight,
                    self.norm.weight,
                )
            else:
                latent = self.norm(self.fused_wkv_wgate(hidden_states))
            if latent.shape[0] != num_tokens:
                latent = tp_all_gather(latent, dim=0)
            return latent

        assert self.state_cache is not None
        attn_ctx = get_forward_context().attn_metadata
        state_entry = getattr(self.state_cache, "kv_cache", None)
        state_raw = state_entry[0] if isinstance(state_entry, tuple) else state_entry
        state_cache = _as_kernel_cache_view(state_raw)
        if self._can_use_fused_wkv(hidden_states):
            if (
                isinstance(attn_ctx, dict)
                and state_cache is not None
                and hidden_states.shape[0] == num_tokens
            ):
                state_meta = attn_ctx[self.state_cache.prefix]
                latent, updated_state = self._get_r2_wkv_pool_op()(
                    hidden_states,
                    self.fused_wkv_wgate.weight,
                    self.norm.weight,
                    positions.to(torch.int32),
                    state_meta.block_tables.reshape(-1),
                    state_meta.query_start_loc,
                    state_meta.request_distribution,
                    state_cache,
                )
                state_cache.copy_(updated_state)
                return latent
            wkv_wgate = self._get_r2_wkv_f32_op()(
                hidden_states, self.fused_wkv_wgate.weight
            )
        else:
            wkv_wgate = self.fused_wkv_wgate(hidden_states.float())
        if wkv_wgate.shape[0] != num_tokens:
            wkv_wgate = tp_all_gather(wkv_wgate, dim=0)
        if not isinstance(attn_ctx, dict) or state_cache is None:
            return self.norm(wkv_wgate[:, : self.head_dim].to(hidden_states.dtype))

        state_meta = attn_ctx[self.state_cache.prefix]
        latent, updated_state = self._get_r2_pool_op()(
            wkv_wgate,
            self.norm.weight,
            positions.to(torch.int32),
            state_meta.block_tables.reshape(-1),
            state_meta.query_start_loc,
            state_meta.request_distribution,
            state_cache,
        )
        state_cache.copy_(updated_state)
        return latent

    def insert_cache(
        self,
        latent: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        attn_ctx = get_forward_context().attn_metadata
        k_cache_obj = self._static_forward_context.get(self.k_cache_prefix)
        entry = getattr(k_cache_obj, "kv_cache", None)
        if not isinstance(attn_ctx, dict) or not isinstance(entry, tuple):
            return
        nope_cache = _as_kernel_cache_view(entry[0])
        rope_cache = _as_kernel_cache_view(entry[1])
        if nope_cache is None or rope_cache is None:
            return
        kv_meta = attn_ctx[self.k_cache_prefix]
        new_nope, new_rope = self._get_insert_cache_op()(
            latent,
            rotary_emb.cos_sin_cache,
            positions.to(torch.int32),
            kv_meta.block_tables.reshape(-1),
            kv_meta.query_start_loc,
            kv_meta.request_distribution,
            nope_cache,
            rope_cache,
        )
        nope_cache.copy_(new_nope)
        rope_cache.copy_(new_rope)


class VllmDeepseekV41IndexerCache(dsv41_attention.DeepseekV4IndexerCache):
    """Indexer K cache as a packed uint8 SEQ_ALONG_LANE record on TPU."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = self.cache_config.block_size
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            tokens_per_state=min(self.compress_ratio, block_size),
            alignment=None,
        )


def _indexer_v41_tp_op_fn(
    rank: jax.Array,
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    cache_kv: jax.Array,
    main_page_indices: jax.Array,
    candidates: jax.Array | None = None,
    *,
    tp_size: int,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    cand_mode: str | None = None,
    cand_blocks: int = 0,
    cand_block_size: int = 0,
) -> jax.Array:
    return indexer_v41_tp_jax(
        rank,
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        main_page_indices,
        candidates,
        tp_size=tp_size,
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        cand_mode=cand_mode,
        cand_blocks=cand_blocks,
        cand_block_size=cand_block_size,
    )


def _indexer_v41_tp_write_op_fn(*args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return _indexer_v41_tp_op_fn(*args, **kwargs)  # type: ignore[return-value]


_indexer_v41_tp_write_op_fn.__signature__ = inspect.signature(  # type: ignore[attr-defined]
    _indexer_v41_tp_op_fn
).replace(return_annotation=tuple[jax.Array, jax.Array])


class VllmDeepseekV41Indexer(dsv41_attention.DeepseekV4Indexer):
    """TPU DeepSeek-V4.1 Indexer; returns full index_topk indices on every TP rank."""

    def __init__(self, *args, **kwargs) -> None:
        cand_buf = kwargs.get("candidate_block_buffer")
        object.__setattr__(self, "_cand_buf", cand_buf)
        self.cand_mode = (
            None
            if cand_buf is None
            else ("write" if kwargs.get("candidate_write") else "read")
        )
        self.cand_block_size = int(kwargs.get("candidate_block_size") or 0)
        self.cand_blocks = 0 if cand_buf is None else int(cand_buf.shape[1])
        orig_sparse_attn = dsv41_attention.SparseAttnIndexer
        orig_sparse_mqa = dsv41_attention.SparseMQAIndexer
        dsv41_attention.SparseAttnIndexer = lambda *a, **kw: None
        dsv41_attention.SparseMQAIndexer = lambda *a, **kw: None
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv41_attention.SparseAttnIndexer = orig_sparse_attn
            dsv41_attention.SparseMQAIndexer = orig_sparse_mqa
        object.__setattr__(self, "indexer_op", self._build_indexer_op())
        object.__setattr__(
            self, "indexer_op_hm", self._build_indexer_op(use_step_meta=True)
        )
        if envs.VLLM_DSV41_TP_INDEXER and get_tensor_model_parallel_world_size() > 1:
            self.register_buffer(
                "_tp_rank_idx",
                torch.tensor([get_tensor_model_parallel_rank()], dtype=torch.int64),
                persistent=False,
            )
            object.__setattr__(self, "indexer_op_tp", self._build_tp_indexer_op())
        if self.owns_k:
            object.__setattr__(self, "produce_k_op", self._get_produce_k_op())
        bs = int(
            getattr(getattr(self.k_cache, "cache_config", None), "block_size", 1024)
        )
        num_states = bs // max(1, min(self.compress_ratio, bs))
        object.__setattr__(
            self, "step_metadata_op", self._get_step_metadata_op(num_states)
        )
        object.__setattr__(self, "_step_metadata_page_size", num_states)
        self._init_fused_ops()

    def _build_indexer_op(self, use_step_meta: bool = False):
        op_name = (
            f"pallas::deepseek_v41_indexer_k{self.topk_tokens}"
            f"_c{self.compress_ratio}_h{self.n_head}"
            f"_s{_name_float(self.softmax_scale)}"
        )
        if self.cand_mode is not None:
            op_name += (
                f"_cand{self.cand_mode}_b{self.cand_blocks}x{self.cand_block_size}"
            )
        if use_step_meta:
            op_name += "_hm"

        def _fake_indexer(q, *args, **kwargs):
            topk = q.new_empty((q.shape[0], self.topk_tokens), dtype=torch.int32)
            if self.cand_mode == "write":
                cands = q.new_empty((q.shape[0], self.cand_blocks), dtype=torch.int32)
                return topk, cands
            return topk

        jax_fn = indexer_v41_write_jax if self.cand_mode == "write" else indexer_v41_jax
        n_inputs = 11 if self.cand_mode == "read" else 10
        if use_step_meta:
            n_inputs += 4

        return _get_or_register_op(
            op_name,
            functools.partial(
                jax_fn,
                k=self.topk_tokens,
                compression_ratio=self.compress_ratio,
                softmax_scale=self.softmax_scale,
                n_head=self.n_head,
                cand_mode=self.cand_mode,
                cand_blocks=self.cand_blocks,
                cand_block_size=self.cand_block_size,
                has_precomputed_meta=use_step_meta,
            ),
            n_inputs,
            _fake_indexer,
        )

    def _build_tp_indexer_op(self):
        tp_size = get_tensor_model_parallel_world_size()
        op_name = (
            f"pallas::deepseek_v41_indexer_tp{tp_size}_k{self.topk_tokens}"
            f"_c{self.compress_ratio}_h{self.n_head}"
            f"_s{_name_float(self.softmax_scale)}"
        )
        if self.cand_mode is not None:
            op_name += (
                f"_cand{self.cand_mode}_b{self.cand_blocks}x{self.cand_block_size}"
            )

        def _fake_tp_indexer(rank, q, *args, **kwargs):
            topk = q.new_empty((q.shape[0], self.topk_tokens), dtype=torch.int32)
            if self.cand_mode == "write":
                cands = q.new_empty((q.shape[0], self.cand_blocks), dtype=torch.int32)
                return topk, cands
            return topk

        jax_fn = (
            _indexer_v41_tp_write_op_fn
            if self.cand_mode == "write"
            else _indexer_v41_tp_op_fn
        )
        return _get_or_register_op(
            op_name,
            functools.partial(
                jax_fn,
                tp_size=tp_size,
                k=self.topk_tokens,
                compression_ratio=self.compress_ratio,
                softmax_scale=self.softmax_scale,
                n_head=self.n_head,
                cand_mode=self.cand_mode,
                cand_blocks=self.cand_blocks,
                cand_block_size=self.cand_block_size,
            ),
            12 if self.cand_mode == "read" else 11,
            _fake_tp_indexer,
        )

    def use_tp_indexer(self, num_tokens: int) -> bool:
        """Token-parallel indexer for this batch size (static, per bucket)."""
        if "indexer_op_tp" not in self.__dict__:
            return False
        tp_size = get_tensor_model_parallel_world_size()
        t_l = num_tokens // tp_size
        return (
            num_tokens % tp_size == 0
            # 2 or 4 rows per rank (and t_l % 8 != 0, e.g. 10 at T=80) run padded
            # to a multiple of 8 inside indexer_v41_tp_jax.
            and (t_l in (2, 4) or t_l >= 8)
            and num_tokens >= envs.VLLM_DSV41_TP_INDEXER_MIN_TOKENS
        )

    def _get_step_metadata_op(self, page_size: int):
        if (
            self.__dict__.get("_step_metadata_page_size") == page_size
            and (cached_op := self.__dict__.get("step_metadata_op")) is not None
        ):
            return cached_op
        page_size_r2 = page_size if self.compress_ratio == 2 else max(1, page_size // 2)
        page_size_r1 = page_size * 2 if self.compress_ratio == 2 else page_size
        block_stride = max(1, (page_size * self.compress_ratio) // 1024)
        op_name = (
            f"pallas::deepseek_v41_indexer_step_meta_k{self.topk_tokens}"
            f"_p{page_size_r2}x{page_size_r1}_s{block_stride}"
        )

        def _fake_step_meta(pos, seq_lens, page_indices, cu_q_lens, dist):
            t = pos.shape[0]
            max_num_seqs = int(seq_lens.shape[0])
            max_blocks_per_seq = int(page_indices.numel()) // max_num_seqs
            pages_per_seq = (max_blocks_per_seq + block_stride - 1) // block_stride
            meta_buf_len = compute_meta_buf_len(
                max_num_seqs,
                pages_per_seq,
                None if isinstance(t, torch.SymInt) else int(t),
            )
            idx_pages_len = max_num_seqs * pages_per_seq
            return (
                pos.new_empty((t,), dtype=torch.int32),
                pos.new_empty((t,), dtype=torch.int32),
                pos.new_empty((idx_pages_len,), dtype=torch.int32),
                pos.new_empty((6, MAX_INDEXER_CHUNKS), dtype=torch.int32),
                pos.new_empty((18, meta_buf_len), dtype=torch.int32),
            )

        op = _get_or_register_op(
            op_name,
            functools.partial(
                compute_indexer_step_metadata_jax,
                k=self.topk_tokens,
                page_size_r2=page_size_r2,
                page_size_r1=page_size_r1,
                block_stride=block_stride,
                max_chunks=MAX_INDEXER_CHUNKS,
            ),
            5,
            _fake_step_meta,
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "step_metadata_op", op)
            object.__setattr__(self, "_step_metadata_page_size", page_size)
        return op

    def compute_step_metadata(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        kv_cache = _as_kernel_cache_view(getattr(self.k_cache, "kv_cache", None))
        if kv_cache is None:
            return None
        attn_ctx = get_forward_context().attn_metadata
        if not isinstance(attn_ctx, dict):
            return None
        attn_metadata = attn_ctx.get(self.k_cache.prefix)
        if attn_metadata is None:
            return None
        op = self._get_step_metadata_op(int(kv_cache.shape[3]))
        return op(
            positions.to(torch.int32),
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )

    def _get_produce_k_op(self):
        if (op := self.__dict__.get("produce_k_op")) is not None:
            return op
        op = _get_or_register_op(
            f"pallas::deepseek_v41_indexer_produce_k_r{self.compress_ratio}",
            functools.partial(
                indexer_produce_k_jax, compress_ratio=self.compress_ratio
            ),
            7,
            lambda *args: torch.empty_like(args[6]),
            donate_argnums=(6,),
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "produce_k_op", op)
        return op

    def _get_produce_k_wk_norm_op(self):
        if (op := self.__dict__.get("produce_k_wk_norm_op")) is not None:
            return op
        eps = float(getattr(self.k_norm, "eps", 1e-6))
        op = _get_or_register_op(
            f"pallas::deepseek_v41_indexer_produce_k_wk_norm_r{self.compress_ratio}_e{_name_float(eps)}",
            functools.partial(
                indexer_produce_k_wk_norm_jax,
                compress_ratio=self.compress_ratio,
                k_norm_eps=eps,
            ),
            9,
            lambda *args: torch.empty_like(args[8]),
            donate_argnums=(8,),
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "produce_k_wk_norm_op", op)
        return op

    def _get_wq_b_rope_quant_op(self):
        if (op := self.__dict__.get("wq_b_rope_quant_op")) is not None:
            return op
        op_name = (
            f"pallas::deepseek_v41_indexer_wq_b_rope_quant"
            f"_h{self.n_head}_d{self.head_dim}_s{_name_float(self.softmax_scale)}"
        )

        def _fake_wq_b_rq(q_fp8, q_s, w_q, w_s, pos, cs, idx_w):
            return (
                q_fp8.new_empty(
                    (q_fp8.shape[0], self.n_head, self.head_dim),
                    dtype=torch.float8_e4m3fn,
                ),
                q_s.new_empty((q_fp8.shape[0], self.n_head), dtype=torch.float32),
            )

        op = _get_or_register_op(
            op_name,
            functools.partial(
                fused_indexer_wq_b_rope_quant,
                softmax_scale=self.softmax_scale,
                n_head=self.n_head,
                head_dim=self.head_dim,
            ),
            7,
            _fake_wq_b_rq,
        )
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "wq_b_rope_quant_op", op)
        return op

    def _init_fused_ops(self) -> None:
        object.__setattr__(self, "_prequant_q", [None])
        if self.owns_k and self.__dict__.get("produce_k_op") is not None:
            object.__setattr__(
                self, "produce_k_wk_norm_op", self._get_produce_k_wk_norm_op()
            )
        if (
            self.__dict__.get("indexer_op") is None
            or self.head_dim != 128
            or self.n_head > 128
        ):
            return
        wq_b_rq_op = self._get_wq_b_rope_quant_op()

        def _wrap_idx_op(orig_op, is_tp: bool = False):
            def _wrapped(*args):
                pq = self._prequant_q[0]
                if pq is not None:
                    self._prequant_q[0] = None
                    q_fp8, q_s, w_q, w_s = pq
                    if is_tp:
                        rank_idx, _, pos, cs, idx_w, *rest = args
                        q_proj, idx_w = wq_b_rq_op(
                            q_fp8, q_s, w_q, w_s, pos, cs, idx_w
                        )
                        return orig_op(rank_idx, q_proj, pos, cs, idx_w, *rest)
                    _, pos, cs, idx_w, *rest = args
                    q_proj, idx_w = wq_b_rq_op(
                        q_fp8, q_s, w_q, w_s, pos, cs, idx_w
                    )
                    return orig_op(q_proj, pos, cs, idx_w, *rest)
                return orig_op(*args)

            return _wrapped

        object.__setattr__(self, "indexer_op", _wrap_idx_op(self.indexer_op))
        if self.__dict__.get("indexer_op_hm") is not None:
            object.__setattr__(
                self, "indexer_op_hm", _wrap_idx_op(self.indexer_op_hm)
            )
        if self.__dict__.get("indexer_op_tp") is not None:
            object.__setattr__(
                self,
                "indexer_op_tp",
                _wrap_idx_op(self.indexer_op_tp, is_tp=True),
            )

    def write_k_cache(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        if not self.owns_k or latent is None:
            return
        kv_cache = _as_kernel_cache_view(getattr(self.k_cache, "kv_cache", None))
        if kv_cache is None:
            return
        attn_ctx = get_forward_context().attn_metadata
        if not isinstance(attn_ctx, dict):
            return
        attn_metadata = attn_ctx.get(self.k_cache.prefix)
        if attn_metadata is None:
            return
        if (
            "produce_k_wk_norm_op" in self.__dict__
            and latent.device.type != "cpu"
            and self.wk._parameters.get("weight") is not None
            and self.wk._parameters.get("bias") is None
            and self.k_norm._parameters.get("weight") is not None
            and self.k_norm._parameters.get("bias") is None
            and (self.compress_ratio == 2 or latent.shape[0] <= 2048)
        ):
            new_kv_cache = self.produce_k_wk_norm_op(
                latent,
                self.wk.weight,
                self.k_norm.weight,
                rotary_emb.cos_sin_cache,
                positions.to(torch.int32),
                attn_metadata.block_tables.reshape(-1),
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                kv_cache,
            )
        else:
            new_kv_cache = self.produce_k_op(
                self.k_norm(self.wk(latent)[0]),
                rotary_emb.cos_sin_cache,
                positions.to(torch.int32),
                attn_metadata.block_tables.reshape(-1),
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                kv_cache,
            )
        kv_cache.copy_(new_kv_cache)

    def can_accept_prequant_query(self) -> bool:
        w_s = self.wq_b._parameters.get("weight_scale", None)
        return (
            w_s is not None
            and w_s.ndim == 1
            and self.wq_b._parameters.get("bias", None) is None
        )

    def _apply_wq_b(
        self,
        query: torch.Tensor | None,
        query_fp8: torch.Tensor | None = None,
        query_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_fp8 is not None and query_scale is not None:
            weight = self.wq_b.weight
            if getattr(self.wq_b, KEEP_VLLM_LAYOUT_ATTR, False):
                weight = weight.transpose(0, 1)
            if query_scale.ndim == 2 and query_scale.shape[1] > 1:
                query_scale = query_scale[:, :1]
            if (
                "wq_b_rope_quant_op" in self.__dict__
                and query_fp8.device.type != "cpu"
                and query_fp8.dtype == torch.float8_e4m3fn
            ):
                self._prequant_q[0] = (
                    query_fp8,
                    query_scale,
                    weight,
                    self.wq_b.weight_scale,
                )
                return query_fp8.new_empty(
                    (query_fp8.shape[0], self.n_head * self.head_dim),
                    dtype=torch.bfloat16,
                )
            return quantized_matmul_prequant(
                query_fp8,
                query_scale,
                weight,
                self.wq_b.weight_scale,
            )
        assert query is not None
        return self.wq_b(query)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        latent: torch.Tensor | None = None,
        step_metadata: tuple[torch.Tensor, ...] | None = None,
        step_metadata_fn: Any = None,
        tp_rank_idx: torch.Tensor | None = None,
        keep_topk_local: bool = False,
        query_fp8: torch.Tensor | None = None,
        query_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        num_tokens = positions.shape[0]
        kv_cache = _as_kernel_cache_view(getattr(self.k_cache, "kv_cache", None))
        if kv_cache is None:
            return positions.new_zeros((num_tokens, self.topk_tokens), dtype=torch.int32)

        attn_ctx = get_forward_context().attn_metadata
        assert isinstance(attn_ctx, dict)
        prefix_key = self.k_cache.prefix
        attn_metadata = attn_ctx[prefix_key]

        if self.owns_k and latent is not None:
            self.write_k_cache(latent, positions, rotary_emb)

        main_metadata = attn_ctx[prefix_key.removesuffix(".indexer.k_cache")]
        if self.use_tp_indexer(num_tokens):
            # Token-parallel: this rank projects and scores tokens
            # [rank * T/tp, (rank + 1) * T/tp) against the replicated K cache,
            # then the top-k indices are all-gathered in rank order.
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            rank_idx = tp_rank_idx if tp_rank_idx is not None else self._tp_rank_idx
            t_local = num_tokens // tp_size

            def local(x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] == t_local:
                    return x
                if x.element_size() == 1:
                    return tp_shard_slice(
                        x.view(torch.uint8), tp_size, tp_rank, rank_idx
                    ).view(x.dtype)
                return tp_shard_slice(x, tp_size, tp_rank, rank_idx)

            if query_fp8 is not None and query_scale is not None:
                q_proj = self._apply_wq_b(
                    None, local(query_fp8), local(query_scale)
                ).view(-1, self.n_head, self.head_dim)
            else:
                assert query is not None
                q_proj = self._apply_wq_b(local(query)).view(
                    -1, self.n_head, self.head_dim
                )
            op_args = (
                rank_idx.to(torch.int32),
                q_proj,
                local(positions.to(torch.int32)),
                rotary_emb.cos_sin_cache,
                local(indexer_weights),
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                kv_cache,
                main_metadata.block_tables,
            )
            if self.cand_mode == "write":
                topk, cands = self.indexer_op_tp(*op_args)
                self._cand_buf[:t_local].copy_(cands)
            elif self.cand_mode == "read":
                topk = self.indexer_op_tp(*op_args, self._cand_buf[:t_local])
            else:
                topk = self.indexer_op_tp(*op_args)
            if not keep_topk_local:
                topk = tp_all_gather(topk, dim=0)
        else:
            # wq_b / weights_proj are replicated, so every rank scores all heads
            # and all tokens against its replicated K cache.
            q_proj = self._apply_wq_b(query, query_fp8, query_scale).view(
                -1, self.n_head, self.head_dim
            )
            if step_metadata is None and step_metadata_fn is not None:
                step_metadata = step_metadata_fn()
            meta_args: tuple[torch.Tensor, ...] = ()
            if step_metadata is not None:
                eff_rl_r2, eff_rl_r1, idx_page_indices, pass_num_steps, pass_schedules = (
                    step_metadata
                )
                eff_rl = eff_rl_r2 if self.compress_ratio == 2 else eff_rl_r1
                meta_args = (eff_rl, idx_page_indices, pass_num_steps, pass_schedules)
            idx_op = self.indexer_op_hm if step_metadata is not None else self.indexer_op

            op_args = (
                q_proj,
                positions.to(torch.int32),
                rotary_emb.cos_sin_cache,
                indexer_weights,
                kv_cache,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                main_metadata.block_tables,
            )
            if self.cand_mode == "write":
                topk, cands = idx_op(*op_args, *meta_args)
                self._cand_buf[:num_tokens].copy_(cands)
            elif self.cand_mode == "read":
                topk = idx_op(*op_args, self._cand_buf[:num_tokens], *meta_args)
            else:
                topk = idx_op(*op_args, *meta_args)
        if self.topk_indices_buffer is not None:
            n_tk = min(num_tokens, int(topk.shape[0]))
            self.topk_indices_buffer[:n_tk].copy_(topk[:n_tk])
        return topk

