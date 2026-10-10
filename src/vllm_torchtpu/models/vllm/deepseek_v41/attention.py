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
"""TPU Attention implementation for DeepSeek-V4.1."""

import functools
import os
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v41 import attention as deepseek_v41_flash_attention
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_all_to_all,
    tp_reduce_scatter,
    tp_shard_slice,
)
from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_kernel
from vllm_torchtpu.kernels.deepseek_v4.o_projection import (
    fused_wq_b_q_rope_prequant_projection,
    fused_wq_b_q_rope_projection,
)
from vllm_torchtpu.kernels.deepseek_v41 import tp_attention
from vllm_torchtpu.kernels.deepseek_v41.sparse_mla import (
    precompute_csa_union_and_bitmask,
    resolve_csa_union_params,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_attention_op as _dsv4_attn_op,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_flash_attention_op as _deepseek_v41_flash_attn_op,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_compressor import (  # noqa: E501
    VllmDeepseekV41Compressor,
    VllmDeepseekV41Indexer,
    VllmDeepseekV41IndexerCache,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    _get_quantized_matmul_op,
    _get_quantized_matmul_prequant_op,
    quantized_matmul,
    quantized_matmul_prequant,
)
from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
    DeepseekV4TPUAttentionBackend,
    VllmDeepseekV4MLAAttention,
    _fake_o_proj,
    _fake_rope,
    _live_cache,
    _name_float,
    _o_proj_jax,
    _pallas_op_cache,
    _rope_jax,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import synchronize_tensors

BATCH_AXIS = _dsv4_attn_op.BATCH_AXIS
VllmDeepseekV4SWACache = _dsv4_attn_op.VllmDeepseekV4SWACache
_attention_csa = _deepseek_v41_flash_attn_op._attention_csa
_attention_swa = _deepseek_v41_flash_attn_op._attention_swa
get_packed_mla_head_size = _dsv4_attn_op.get_packed_mla_head_size
_Q_ROPE_SPECS = (P(None, "model", None), P(), P())
_KV_NORM_ROPE_SPECS = (P(), P(), P(), P())
_QKV_NORM_ROPE_QUANT_SPECS = (P(), P(), P(), P(), P())
_O_PROJ_SPECS = (P(None, "model", None), P(), P(), P(), P())
_WQ_B_Q_ROPE_SPECS = (P(), P(), P(), P(), P())
_WQ_B_Q_ROPE_PREQUANT_SPECS = (P(), P(), P(), P(), P(), P())
_CSA_UNION_SPECS = (P(BATCH_AXIS, None),)
_FLASHMLA_TP8: bool = os.environ.get("VLLM_DSV41_FLASHMLA_TP8", "0") == "1"


def _kvnorm_rope_jax(
    x: jax.Array,
    weight: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    return rope_kernel.rmsnorm_rope(x, weight, positions, cos_sin_cache, eps=eps)


def _qkv_norm_rope_quant_jax(
    qr_kv: jax.Array,
    q_weight: jax.Array,
    kv_weight: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    q_eps: float,
    kv_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return rope_kernel.qkv_norm_rope_quant(
        qr_kv,
        q_weight,
        kv_weight,
        positions,
        cos_sin_cache,
        q_eps=q_eps,
        kv_eps=kv_eps,
    )


def _fake_qkv_norm_rope_quant(
    qr_kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del positions, cos_sin_cache
    t = qr_kv.shape[0]
    q_dim = q_weight.shape[0]
    kv_dim = kv_weight.shape[0]
    return (
        torch.empty((t, q_dim), dtype=torch.float8_e4m3fn, device=qr_kv.device),
        torch.empty((t, rope_kernel.LANE), dtype=torch.float32, device=qr_kv.device),
        torch.empty((t, kv_dim), dtype=qr_kv.dtype, device=qr_kv.device),
    )


def _wq_b_q_rope_jax(
    qr: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wq_b: jax.Array,
    wq_b_scale: jax.Array,
    *,
    head_dim: int,
) -> jax.Array:
    return fused_wq_b_q_rope_projection(
        qr,
        positions,
        cos_sin_cache,
        wq_b,
        wq_b_scale.reshape(-1),
        head_dim=head_dim,
    )


def _wq_b_q_rope_prequant_jax(
    qr_fp8: jax.Array,
    qr_scale: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wq_b: jax.Array,
    wq_b_scale: jax.Array,
    *,
    head_dim: int,
) -> jax.Array:
    return fused_wq_b_q_rope_prequant_projection(
        qr_fp8,
        qr_scale,
        positions,
        cos_sin_cache,
        wq_b,
        wq_b_scale.reshape(-1),
        head_dim=head_dim,
    )


def _fake_wq_b_q_rope(
    qr: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    del positions, cos_sin_cache, wq_b_scale
    num_heads = wq_b.shape[1] // head_dim
    return torch.empty(
        (qr.shape[0], num_heads, head_dim), dtype=qr.dtype, device=qr.device
    )


def _fake_wq_b_q_rope_prequant(
    qr_fp8: torch.Tensor,
    qr_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    del qr_scale, positions, cos_sin_cache, wq_b_scale
    num_heads = wq_b.shape[1] // head_dim
    return torch.empty(
        (qr_fp8.shape[0], num_heads, head_dim),
        dtype=torch.bfloat16,
        device=qr_fp8.device,
    )


def _get_or_register_op(
    self: Any, op_name: str, fn: Any, in_specs: tuple, fake_fn: Any, **kw: Any
):
    if (op := self.__dict__.get(op_name)) is not None:
        return op
    if (op := _pallas_op_cache.get(op_name)) is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(fn, **kw) if kw else fn,
            mesh=get_vllm_model_wrapper_context().mesh,
            input_partition_specs=in_specs,
        )
        op.register_fake(fake_fn)
        _pallas_op_cache[op_name] = op
    object.__setattr__(self, op_name, op)
    return op


def _unpack_meta(meta: Any) -> tuple[Any, Any, Any, Any]:
    if meta is None:
        return None, None, None, None
    get = meta.get if isinstance(meta, dict) else lambda k: getattr(meta, k, None)
    return (
        get("seq_lens"),
        get("block_tables"),
        get("query_start_loc"),
        get("request_distribution"),
    )


class VllmDeepseekV41SWACache(VllmDeepseekV4SWACache):
    """SWA cache wrapper accepting DeepSeek-V4.1's extra constructor kwargs."""

    def __init__(
        self,
        head_dim: int,
        window_size: int | None,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        backend_cls=None,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__(
            head_dim, window_size or 0, dtype, prefix, cache_config, backend_cls
        )


class VllmDeepseekV41MLAAttention(
    deepseek_v41_flash_attention.DeepseekV4Attention, AttentionLayerBase
):
    """TPU sparse MLA attention for DeepSeek-V4.1 (SWA-only + CSA2 shared KV/index)."""

    backend_cls = DeepseekV4TPUAttentionBackend

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        orig_indexer = deepseek_v41_flash_attention.DeepseekV4Indexer
        orig_compressor = deepseek_v41_flash_attention.DeepseekCompressor
        orig_swa_cache = deepseek_v41_flash_attention.DeepseekV4SWACache
        orig_indexer_cache = deepseek_v41_flash_attention.DeepseekV4IndexerCache
        orig_cuda_event = torch.cuda.Event
        orig_jit_warmup = vllm_config.kernel_config.enable_jit_warmup

        deepseek_v41_flash_attention.DeepseekV4Indexer = VllmDeepseekV41Indexer
        deepseek_v41_flash_attention.DeepseekCompressor = VllmDeepseekV41Compressor
        deepseek_v41_flash_attention.DeepseekV4SWACache = VllmDeepseekV41SWACache
        deepseek_v41_flash_attention.DeepseekV4IndexerCache = (
            VllmDeepseekV41IndexerCache
        )
        torch.cuda.Event = lambda *args, **kwargs: None
        vllm_config.kernel_config.enable_jit_warmup = False

        try:
            super().__init__(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                candidate_block_buffer=candidate_block_buffer,
            )
        finally:
            deepseek_v41_flash_attention.DeepseekV4Indexer = orig_indexer
            deepseek_v41_flash_attention.DeepseekCompressor = orig_compressor
            deepseek_v41_flash_attention.DeepseekV4SWACache = orig_swa_cache
            deepseek_v41_flash_attention.DeepseekV4IndexerCache = orig_indexer_cache
            torch.cuda.Event = orig_cuda_event
            vllm_config.kernel_config.enable_jit_warmup = orig_jit_warmup

        self.attn_sink.weight_loader = sharded_weight_loader(0)
        self._step_indexer_metadata: tuple[torch.Tensor, ...] | None = None
        self._step_csa_union: tuple[torch.Tensor, ...] | None = None
        self._ced_decoder: bool = False
        self._ced_global_from_encoder: bool = False
        self._in_encoder_stage: bool = False
        self._tp_attn_encoder_only: bool = False
        self.non_causal_multi_token_decode: bool = bool(
            self.layer_id >= vllm_config.model_config.hf_config.num_hidden_layers
        )

        max_len = vllm_config.model_config.max_model_len
        cache = self.rotary_emb.cos_sin_cache
        if envs.TPU_ROPE_CACHE_TRUNCATE and cache.shape[0] > max_len:
            self.rotary_emb.cos_sin_cache = cache[:max_len].clone(
                memory_format=torch.contiguous_format
            )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward_mqa(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _o_proj(self, z: torch.Tensor) -> torch.Tensor:
        return self.wo_b(z)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if not self.is_kv_source:
            return None
        block_size = vllm_config.cache_config.block_size
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=get_packed_mla_head_size(vllm_config.model_config.hf_config),
            dtype=torch.uint8,
            tokens_per_state=min(max(1, self.compress_ratio), block_size),
            alignment=None,
        )

    def _kv_source_module(self) -> Any:
        if self.is_kv_source or not self.compressed_cache_prefix:
            return self
        return self._static_forward_context[self.compressed_cache_prefix]

    def _leader_group_bounds(self, ced_decoder: bool | None = None) -> tuple[Any, int]:
        eff_ced = self._ced_decoder if ced_decoder is None else ced_decoder
        cache_attr = (
            "_cached_leader_group_bounds_ced"
            if eff_ced
            else "_cached_leader_group_bounds"
        )
        cached = self.__dict__.get(cache_attr)
        if cached is not None:
            return cached
        my_lid = extract_layer_index(self.prefix)
        attn_mods = [
            mod
            for mod in self._static_forward_context.values()
            if isinstance(mod, VllmDeepseekV41MLAAttention)
            and not getattr(mod, "non_causal_multi_token_decode", False)
        ]
        ced_boundary = len(attn_mods) // 2
        min_lid = my_lid
        max_lid = my_lid
        first_leader: Any = self
        for mod in attn_mods:
            if mod.indexer is not None:
                lid = extract_layer_index(mod.prefix)
                if eff_ced and lid < ced_boundary:
                    continue
                if lid < min_lid:
                    min_lid = lid
                    first_leader = mod
                if lid > max_lid:
                    max_lid = lid
        res = (first_leader, max_lid)
        if not torch.compiler.is_compiling():
            object.__setattr__(self, cache_attr, res)
        return res

    def _get_or_compute_indexer_step_metadata(
        self, positions: torch.Tensor, *, ced_decoder: bool | None = None
    ) -> tuple[torch.Tensor, ...]:
        # Indexer step metadata depends only on positions and attention metadata,
        # so the first indexer layer of a stage computes it and the rest reuse it.
        first_leader, last_leader_lid = self._leader_group_bounds(ced_decoder)
        my_lid = extract_layer_index(self.prefix)
        cached_meta = getattr(first_leader, "_step_indexer_metadata", None)
        if my_lid == extract_layer_index(first_leader.prefix) or cached_meta is None:
            step_meta = self.indexer.compute_step_metadata(positions)
            if last_leader_lid > my_lid:
                first_leader._step_indexer_metadata = step_meta
            return step_meta
        if my_lid == last_leader_lid:
            first_leader._step_indexer_metadata = None
        return cached_meta

    def _csa_kv_group_bounds(
        self, ced_decoder: bool | None = None
    ) -> tuple[int, int]:
        eff_ced = self._ced_decoder if ced_decoder is None else ced_decoder
        cache_attr = (
            "_cached_csa_kv_group_bounds_ced"
            if eff_ced
            else "_cached_csa_kv_group_bounds"
        )
        cached = self.__dict__.get(cache_attr)
        if cached is not None:
            return cached
        kv_holder = self._kv_source_module()
        my_lid = extract_layer_index(self.prefix)
        attn_mods = [
            mod
            for mod in self._static_forward_context.values()
            if isinstance(mod, VllmDeepseekV41MLAAttention)
        ]
        ced_boundary = len(attn_mods) // 2
        min_lid = my_lid
        max_lid = my_lid
        for mod in attn_mods:
            if mod.compress_ratio > 0 and mod._kv_source_module() is kv_holder:
                lid = extract_layer_index(mod.prefix)
                if eff_ced and lid < ced_boundary:
                    continue
                if lid < min_lid:
                    min_lid = lid
                if lid > max_lid:
                    max_lid = lid
        res = (int(min_lid), int(max_lid))
        if not torch.compiler.is_compiling():
            object.__setattr__(self, cache_attr, res)
        return res

    def _get_or_compute_csa_union(
        self,
        topk_indices: torch.Tensor,
        total_cache_rows: int,
        m_union: int,
        b_q: int,
        *,
        ced_decoder: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        kv_holder = self._kv_source_module()
        first_lid, last_lid = self._csa_kv_group_bounds(ced_decoder)
        my_lid = extract_layer_index(self.prefix)
        cached_entry = getattr(kv_holder, "_step_csa_union", None)
        if (
            self.indexer is not None
            or my_lid == first_lid
            or cached_entry is None
            or cached_entry[0] != b_q
            or cached_entry[1] != m_union
        ):
            cached_u = self._get_csa_union_op(
                total_cache_rows, m_union, b_q=b_q
            )(topk_indices)
            if last_lid > my_lid:
                kv_holder._step_csa_union = (b_q, m_union, cached_u)
            else:
                kv_holder._step_csa_union = None
            return cached_u
        if my_lid == last_lid:
            kv_holder._step_csa_union = None
        return cached_entry[2]

    def _get_csa_union_op(
        self, total_cache_rows: int, m_union: int, b_q: int | None = None
    ):
        b_q = envs.VLLM_DSV41_CSA_UNION_BQ if b_q is None else b_q
        u_chunk = min(512, m_union)
        op_name = f"pallas::deepseek_v41_csa_union_r{total_cache_rows}_bq{b_q}_m{m_union}_uc{u_chunk}"
        if (op := self.__dict__.get(op_name)) is not None:
            return op

        def _fake_union(topk_indices: torch.Tensor):
            num_groups = topk_indices.shape[0] // b_q
            return (
                torch.empty(
                    (num_groups * m_union,),
                    dtype=torch.int32,
                    device=topk_indices.device,
                ),
                torch.empty(
                    (num_groups, m_union // 128, 128),
                    dtype=torch.int32,
                    device=topk_indices.device,
                ),
                torch.empty(
                    (num_groups,),
                    dtype=torch.int32,
                    device=topk_indices.device,
                ),
            )

        return _get_or_register_op(
            self,
            op_name,
            precompute_csa_union_and_bitmask,
            _CSA_UNION_SPECS,
            _fake_union,
            total_cache_rows=total_cache_rows,
            b_q=b_q,
            m_union=m_union,
            u_chunk=u_chunk,
        )

    @property
    def attn_op(self):
        op = self._get_attn_op(False)
        if (
            getattr(self, "_supports_ced_decoder", False)
            and not torch.compiler.is_compiling()
        ):
            _ = self._get_attn_op(False, ced_decoder=True)
            if self.compress_ratio > 0:
                _ = self._get_attn_op(True, ced_decoder=True)
        if self.non_causal_multi_token_decode and not torch.compiler.is_compiling():
            _ = self.context_kv_attn_op
            _ = self.kv_rope_op
        return op

    @property
    def context_kv_attn_op(self):
        op = self.__dict__.get("_ctx_kv_attn_op_instance")
        if op is not None:
            return op
        _ = self.kv_rope_op
        swa_tensor = _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        if swa_tensor is None:
            prof_op = self.__dict__.get("_prof_ctx_kv_attn_op_instance")
            if prof_op is None:
                prof_op, _ = self._build_attn_op(non_causal_override=False)
                object.__setattr__(self, "_prof_ctx_kv_attn_op_instance", prof_op)
            return prof_op
        op, is_real = self._build_attn_op(non_causal_override=False)
        assert is_real
        object.__setattr__(self, "_ctx_kv_attn_op_instance", op)
        self.__dict__.pop("_prof_ctx_kv_attn_op_instance", None)
        return op

    def _get_attn_op(self, has_union: bool = False, ced_decoder: bool | None = None):
        eff_ced = self._ced_decoder if ced_decoder is None else ced_decoder
        ced_tag = "_ced" if eff_ced else ""
        u_tag = "_u1" if has_union else "_instance"
        attr_name = f"_attn_op{ced_tag}{u_tag}"
        prof_attr = f"_prof_attn_op{ced_tag}{u_tag}"
        op = self.__dict__.get(attr_name)
        if op is not None:
            return op
        _ = self.kvnorm_rope_op
        _ = self.o_proj_op
        _ = self.q_rope_op
        _ = getattr(self, "qkv_norm_rope_op", None)
        _ = getattr(self, "qkv_norm_rope_and_quant_op", None)
        if self.non_causal_multi_token_decode and not torch.compiler.is_compiling():
            _ = self.kv_rope_op
            _ = self.context_kv_attn_op
        if envs.VLLM_DSV41_FUSED_Q_ROPE:
            _ = self.wq_b_q_rope_op
            _ = self.qkv_norm_rope_quant_op
            _ = self.wq_b_q_rope_prequant_op
        if envs.VLLM_DSV41_FUSED_SEAM_QUANT or envs.VLLM_DSV41_FUSED_Q_ROPE:
            _ = _get_quantized_matmul_prequant_op()
        if "_tp_attn_wo_b_w" in self._buffers:
            _ = _get_quantized_matmul_op()
        if self.indexer is not None and not torch.compiler.is_compiling():
            _ = self._leader_group_bounds(eff_ced)
        swa_tensor = _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        if swa_tensor is None:
            prof_op = self.__dict__.get(prof_attr)
            if prof_op is None:
                prof_op, _ = self._build_attn_op(
                    has_union=has_union, ced_decoder=eff_ced
                )
                object.__setattr__(self, prof_attr, prof_op)
            if (
                not has_union
                and self.compress_ratio > 0
                and not torch.compiler.is_compiling()
            ):
                _ = self._get_attn_op(True, ced_decoder=eff_ced)
            return prof_op
        op, is_real = self._build_attn_op(has_union=has_union, ced_decoder=eff_ced)
        assert is_real
        object.__setattr__(self, attr_name, op)
        self.__dict__.pop(prof_attr, None)
        if not torch.compiler.is_compiling():
            if not has_union and self.compress_ratio > 0:
                _ = self._get_attn_op(True, ced_decoder=eff_ced)
                kv_holder = self._kv_source_module()
                main_entry = getattr(kv_holder, "kv_cache", None)
                main_cache = _live_cache(
                    main_entry[0] if isinstance(main_entry, tuple) else main_entry
                )
                if main_cache is not None and main_cache.shape[0] > 0:
                    total_r = int(main_cache.shape[0] * main_cache.shape[1])
                    m_u = (
                        envs.VLLM_DSV41_CSA_UNION_M
                        if envs.VLLM_DSV41_CSA_UNION_M > 0
                        else ((3 * int(self.topk_indices_buffer.shape[1]) + 127) // 128) * 128
                    )
                    m_u_v = (
                        envs.VLLM_DSV41_CSA_UNION_M
                        if envs.VLLM_DSV41_CSA_UNION_M > 0
                        else ((2 * int(self.topk_indices_buffer.shape[1]) + 127) // 128) * 128
                    )
                    m_u_k3 = (
                        envs.VLLM_DSV41_CSA_UNION_M
                        if envs.VLLM_DSV41_CSA_UNION_M > 0
                        else ((4 * int(self.topk_indices_buffer.shape[1]) + 127) // 128) * 128
                    )
                    _ = self._get_csa_union_op(total_r, m_u)
                    _ = self._get_csa_union_op(total_r, m_u_v, b_q=6)
                    _ = self._get_csa_union_op(total_r, m_u_v, b_q=5)
                    _ = self._get_csa_union_op(total_r, m_u_k3, b_q=5)
                    _ = self._get_csa_union_op(total_r, m_u_k3, b_q=4)
                    if not torch.compiler.is_compiling():
                        self._csa_kv_group_bounds(False)
                        self._csa_kv_group_bounds(True)
            if self.indexer is not None:
                idx_kv = _live_cache(getattr(self.indexer.k_cache, "kv_cache", None))
                if idx_kv is not None:
                    _ = self.indexer._get_step_metadata_op(int(idx_kv.shape[3]))
            tp_size = get_tensor_model_parallel_world_size()
            if (
                envs.VLLM_DSV41_TP_ATTN
                and "_tp_attn_wq_b_w" in self._buffers
                and self._tp_attn_ready(tp_size, tp_size)
            ):
                self.__dict__.pop("_tp_attn_op", None)
                _ = self._get_tp_attn_op()
                if envs.VLLM_DSV41_TP_ATTN_LOCAL_TOPK and self.compress_ratio > 0:
                    _ = self._tp_attn_topk_indexer()
        return op

    def _build_attn_op(
        self,
        has_union: bool = False,
        ced_decoder: bool | None = None,
        non_causal_override: bool | None = None,
    ):
        eff_ced = self._ced_decoder if ced_decoder is None else ced_decoder
        non_causal = (
            bool(getattr(self, "non_causal_multi_token_decode", False))
            if non_causal_override is None
            else non_causal_override
        )
        is_ctx_kv_only = non_causal_override is False
        vllm_context = get_vllm_model_wrapper_context()
        is_csa = self.compress_ratio > 0
        kv_holder = self._kv_source_module()
        main_entry = getattr(kv_holder, "kv_cache", None)
        main_cache = _live_cache(
            main_entry[0] if isinstance(main_entry, tuple) else main_entry
        )
        swa_tensor = _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        two_caches_same_buffer = bool(
            is_csa
            and main_cache is not None
            and swa_tensor is not None
            and main_cache.data_ptr() == swa_tensor.data_ptr()
        )
        built_from_real_caches = swa_tensor is not None

        vllm_cfg = getattr(vllm_context, "vllm_config", None)
        assert vllm_cfg is not None
        logical_page_size = VllmDeepseekV4SWACache._swa_block_size(
            vllm_cfg.cache_config.block_size, self.window_size
        )

        use_flashmla_tp8 = (
            _FLASHMLA_TP8
            and get_tensor_model_parallel_world_size() == 8
            and not non_causal
            and not is_ctx_kv_only
        )
        if use_flashmla_tp8:
            from vllm_torchtpu.kernels.deepseek_v41.flashmla_split_kv import (
                flashmla_attention_csa_tp8,
                flashmla_attention_swa_tp8,
            )

            attn_fn = (
                flashmla_attention_csa_tp8 if is_csa else flashmla_attention_swa_tp8
            )
        else:
            attn_fn = _attention_csa if is_csa else _attention_swa
        extra_kw: dict[str, Any] = {}
        if is_csa:
            extra_kw["two_caches_same_buffer"] = two_caches_same_buffer
        elif not use_flashmla_tp8:
            extra_kw["non_causal_block"] = non_causal
        wrapped_fn = functools.partial(
            attn_fn,
            sm_scale=self.scale,
            sliding_window=1 if is_ctx_kv_only else self.window_size,
            logical_page_size=logical_page_size,
            ced_decoder=eff_ced,
            **extra_kw,
        )
        op_type = "csa" if is_csa else "swa"
        op_name = (
            f"pallas::deepseek_v41_attention_{op_type}"
            f"{'_flashmla_tp8' if use_flashmla_tp8 else ''}"
            f"{'_ced' if eff_ced else ''}"
            f"_p{logical_page_size}"
            f"{'_aliased' if two_caches_same_buffer else ''}"
            f"{'_ncb' if non_causal else ''}"
            f"{'_ctxkv' if is_ctx_kv_only else ''}"
            f"{'_u1' if (is_csa and has_union) else ''}"
            f"{'' if built_from_real_caches else '_prof'}"
        )
        if op_name in _pallas_op_cache:
            return _pallas_op_cache[op_name], built_from_real_caches

        # Head-parallel under TP: q and sinks are split by head, caches replicated.
        b_ax = BATCH_AXIS
        in_specs: tuple = (
            P(None, "model", None),
            P(),
            P(),
            P(b_ax),
            P(b_ax),
            P(),
            P(),
            P(),
            P(b_ax),
            P(b_ax, None) if is_csa else P(b_ax),
            P(b_ax),
            P(),
            P(),
            P("model"),
            *((P(),) if is_csa else ()),
            *(
                (P(b_ax), P(b_ax, None, None), P(b_ax))
                if (is_csa and has_union)
                else ()
            ),
        )
        attn_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=vllm_context.mesh,
            donate_argnums=(2,),
            input_partition_specs=in_specs,
        )
        attn_jax_op.register_fake(
            lambda q, new_kv, sw_cache, *a, **kw: (
                torch.empty_like(q),
                torch.empty_like(sw_cache),
            )
        )
        _pallas_op_cache[op_name] = attn_jax_op
        return attn_jax_op, built_from_real_caches

    def _tp_attn_ready(self, num_tokens: int, tp_size: int) -> bool:
        """Whether VLLM_DSV41_TP_ATTN can run this step (CED is checked by the caller)."""
        if not (
            envs.VLLM_DSV41_TP_ATTN
            and not self.non_causal_multi_token_decode
            and tp_size > 1
            and num_tokens % tp_size == 0
            # A token-count threshold splits the bucket ladder into traces
            # (compilation/shape_variants.py), and this path's extra graph
            # inputs then fail the bucket hand-over's signature check. <= 0
            # skips the comparison so no guard is recorded.
            and (
                envs.VLLM_DSV41_TP_ATTN_MIN_TOKENS <= 0
                or num_tokens >= envs.VLLM_DSV41_TP_ATTN_MIN_TOKENS
            )
            and "_tp_attn_wq_b_w" in self._buffers
            and _live_cache(getattr(self.swa_cache_layer, "kv_cache", None)) is not None
        ):
            return False
        if self.compress_ratio == 0:
            return True
        main_entry = getattr(self._kv_source_module(), "kv_cache", None)
        main_cache = _live_cache(
            main_entry[0] if isinstance(main_entry, tuple) else main_entry
        )
        return main_cache is not None and main_cache.shape[0] > 0

    def _tp_attn_topk_indexer(self) -> Any | None:
        """Indexer of the CSA layers sharing this layer's top-k rows.

        None unless all of them can run token-parallel attention. Cached from
        `_get_attn_op` outside torch.compile, so a trace touches no other
        layer's module.
        """
        indexer = self.__dict__.get("_cached_tp_attn_topk_indexer")
        if indexer is not None or torch.compiler.is_compiling():
            return indexer
        my_lid = extract_layer_index(self.prefix)
        csa_mods = sorted(
            (
                (extract_layer_index(mod.prefix), mod)
                for mod in self._static_forward_context.values()
                if isinstance(mod, VllmDeepseekV41MLAAttention)
                and mod.compress_ratio > 0
                and not getattr(mod, "non_causal_multi_token_decode", False)
            ),
            key=lambda item: item[0],
        )
        members: list[Any] = []
        for lid, mod in csa_mods:
            if mod.indexer is not None:
                if lid > my_lid:
                    break
                members = []
            members.append(mod)
        tp_size = get_tensor_model_parallel_world_size()
        if (
            not members
            or members[0].indexer is None
            or not all(m._tp_attn_ready(tp_size, tp_size) for m in members)
        ):
            return None
        indexer = members[0].indexer
        object.__setattr__(self, "_cached_tp_attn_topk_indexer", indexer)
        return indexer

    def _keep_topk_local(self, num_tokens: int) -> bool:
        """Whether the top-k rows this layer uses can stay rank-local (T/tp).

        Called only when this layer runs token-parallel attention, which reads
        only this rank's token rows. The layers sharing the rows run in one
        stage loop with the same token-sharding, CED and encoder-stage
        settings, so they all run it too.
        """
        if not envs.VLLM_DSV41_TP_ATTN_LOCAL_TOPK or self.compress_ratio == 0:
            return False
        indexer = self._tp_attn_topk_indexer()
        return indexer is not None and indexer.use_tp_indexer(num_tokens)

    def _get_tp_attn_op(self):
        """Token-parallel attention op: all heads for this rank's T/tp tokens."""
        op = self.__dict__.get("_tp_attn_op")
        if op is not None:
            return op
        vllm_context = get_vllm_model_wrapper_context()
        vllm_cfg = getattr(vllm_context, "vllm_config", None)
        assert vllm_cfg is not None
        is_csa = self.compress_ratio > 0
        main_entry = getattr(self._kv_source_module(), "kv_cache", None)
        main_cache = _live_cache(
            main_entry[0] if isinstance(main_entry, tuple) else main_entry
        )
        swa_tensor = _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        assert swa_tensor is not None
        two_caches_same_buffer = bool(
            is_csa
            and main_cache is not None
            and main_cache.data_ptr() == swa_tensor.data_ptr()
        )
        logical_page_size = VllmDeepseekV4SWACache._swa_block_size(
            vllm_cfg.cache_config.block_size, self.window_size
        )
        base_fn = functools.partial(
            _attention_csa if is_csa else _attention_swa,
            sm_scale=self.scale,
            sliding_window=self.window_size,
            logical_page_size=logical_page_size,
            token_parallel=True,
            **({"two_caches_same_buffer": two_caches_same_buffer} if is_csa else {}),
        )
        op_name = (
            f"pallas::deepseek_v41_attention_{'csa' if is_csa else 'swa'}"
            f"_tpattn{get_tensor_model_parallel_world_size()}"
            f"_p{logical_page_size}"
            f"{'_aliased' if two_caches_same_buffer else ''}"
        )
        op = _pallas_op_cache.get(op_name)
        if op is None:
            # Every operand but q is whole on each rank (q, the token slice,
            # differs per rank as in the head-parallel op).
            op = pallas.jax_op(
                op_name,
                functools.partial(
                    tp_attention.attention_tp_local,
                    base_fn=base_fn,
                    logical_page_size=logical_page_size,
                ),
                mesh=vllm_context.mesh,
                donate_argnums=(2,),
                input_partition_specs=(P(None, "model", None),)
                + (P(),) * (15 if is_csa else 14),
            )
            op.register_fake(
                lambda q, new_kv, sw_cache, *a, **kw: (
                    torch.empty_like(q),
                    torch.empty_like(sw_cache),
                )
            )
            _pallas_op_cache[op_name] = op
        object.__setattr__(self, "_tp_attn_op", op)
        return op

    def _build_tp_attn_weights(self) -> None:
        """Replicates wq_b, wo_a, wo_b and attn_sink across TP ranks.

        For VLLM_DSV41_TP_ATTN; the head-sharded originals stay for the
        head-parallel path (small or CED steps).
        """
        tp_size = get_tensor_model_parallel_world_size()

        def gather(x: torch.Tensor, dim: int) -> torch.Tensor:
            if x.element_size() == 1:  # fp8 / e8m0: gather the raw bytes.
                out = tp_all_gather(x.view(torch.uint8), dim).view(x.dtype)
            else:
                out = tp_all_gather(x, dim)
            # Materialize once in the native dtype so the compiled step does
            # not relayout the replicated weights on every call.
            return out.clone(memory_format=torch.contiguous_format)

        wq_b_s, wo_a_s, wo_b_s = (
            mod.weight_scale.data for mod in (self.wq_b, self.wo_a, self.wo_b)
        )
        assert wo_a_s.ndim == 1 and wq_b_s.ndim in (1, 4) and wo_b_s.ndim in (1, 4), (
            f"{self.prefix}: VLLM_DSV41_TP_ATTN needs 1-D or rank-4 weight scales, "
            f"got wq_b {list(wq_b_s.shape)}, wo_a {list(wo_a_s.shape)}, "
            f"wo_b {list(wo_b_s.shape)}."
        )
        device = self.wq_b.weight.device
        rank = get_tensor_model_parallel_rank()
        buffers = {
            # Column-parallel wq_b / wo_a: heads (groups) concatenate along n.
            "_tp_attn_wq_b_w": gather(self.wq_b.weight.data, 1),
            "_tp_attn_wq_b_s": gather(wq_b_s, wq_b_s.ndim - 1),
            "_tp_attn_wo_a_w": gather(self.wo_a.weight.data, 1),
            "_tp_attn_wo_a_s": gather(wo_a_s, 0),
            # Row-parallel wo_b: groups concatenate along k, and a per-rank
            # 1-D scale becomes that rank's k-block of a blockwise scale.
            "_tp_attn_wo_b_w": gather(self.wo_b.weight.data, 0),
            "_tp_attn_wo_b_s": (
                gather(wo_b_s, 1)
                if wo_b_s.ndim == 4
                else gather(wo_b_s.unsqueeze(0), 0).view(1, tp_size, 1, -1)
            ),
            "_tp_attn_sink": gather(self.attn_sink.data, 0),
            "_tp_attn_rank_idx": torch.tensor([rank], dtype=torch.int64, device=device),
            "_tp_attn_rank_i32": torch.tensor([rank], dtype=torch.int32, device=device),
        }
        if device.type == "tpu":
            synchronize_tensors(list(buffers.values()), wait=True)
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=False)

    @property
    def q_rope_op(self):
        return _get_or_register_op(
            self, "pallas::deepseek_v41_q_rope", _rope_jax, _Q_ROPE_SPECS, _fake_rope
        )

    qnorm_rope_op = q_rope_op
    kv_rope_op = VllmDeepseekV4MLAAttention.kv_rope_op

    @property
    def kvnorm_rope_op(self):
        eps = float(self.kv_norm.variance_epsilon)
        return _get_or_register_op(
            self,
            f"pallas::deepseek_v41_kvnorm_rope_e{_name_float(eps)}",
            _kvnorm_rope_jax,
            _KV_NORM_ROPE_SPECS,
            _fake_rope,
            eps=eps,
        )

    @property
    def qkv_norm_rope_quant_op(self):
        q_eps = float(self.q_norm.variance_epsilon)
        kv_eps = float(self.kv_norm.variance_epsilon)
        return _get_or_register_op(
            self,
            (
                f"pallas::deepseek_v41_qkv_norm_rope_quant_qe{_name_float(q_eps)}"
                f"_kve{_name_float(kv_eps)}"
            ),
            _qkv_norm_rope_quant_jax,
            _QKV_NORM_ROPE_QUANT_SPECS,
            _fake_qkv_norm_rope_quant,
            q_eps=q_eps,
            kv_eps=kv_eps,
        )

    @property
    def o_proj_op(self):
        head_dim = int(self.head_dim)
        return _get_or_register_op(
            self,
            f"pallas::deepseek_v4_o_proj_d{head_dim}",
            _o_proj_jax,
            _O_PROJ_SPECS,
            _fake_o_proj,
            head_dim=head_dim,
        )

    @property
    def wq_b_q_rope_op(self):
        head_dim = int(self.head_dim)
        return _get_or_register_op(
            self,
            f"pallas::deepseek_v41_wq_b_q_rope_d{head_dim}",
            _wq_b_q_rope_jax,
            _WQ_B_Q_ROPE_SPECS,
            functools.partial(_fake_wq_b_q_rope, head_dim=head_dim),
            head_dim=head_dim,
        )

    @property
    def wq_b_q_rope_prequant_op(self):
        head_dim = int(self.head_dim)
        return _get_or_register_op(
            self,
            f"pallas::deepseek_v41_wq_b_q_rope_prequant_d{head_dim}",
            _wq_b_q_rope_prequant_jax,
            _WQ_B_Q_ROPE_PREQUANT_SPECS,
            functools.partial(_fake_wq_b_q_rope_prequant, head_dim=head_dim),
            head_dim=head_dim,
        )

    def can_accept_prequant_hidden(
        self,
        num_tokens: int,
        *,
        token_sharded: bool,
        ced_global_from_encoder: bool | None = None,
    ) -> bool:
        if not envs.VLLM_DSV41_FUSED_SEAM_QUANT:
            return False
        w_s = self.fused_wqa_wkv._parameters.get("weight_scale", None)
        if (
            w_s is None
            or w_s.ndim != 1
            or self.fused_wqa_wkv._parameters.get("bias", None) is not None
        ):
            return False
        if not token_sharded:
            return True
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return True
        use_tp_idx = (
            self.indexer is not None
            and getattr(self.indexer, "use_tp_indexer", None) is not None
            and self.indexer.use_tp_indexer(num_tokens)
        )
        eff_ced_global = (
            self._ced_global_from_encoder
            if ced_global_from_encoder is None
            else ced_global_from_encoder
        )
        compressor = None if eff_ced_global else getattr(self, "compressor", None)
        use_tp_attn = (
            envs.VLLM_DSV41_TP_ATTN
            and num_tokens >= envs.VLLM_DSV41_TP_ATTN_MIN_TOKENS
        )
        return (
            num_tokens % tp_size == 0
            and (compressor is None or use_tp_attn)
            and (self.indexer is None or use_tp_idx)
        )

    def _apply_fused_wqa_wkv(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp8: torch.Tensor | None,
        hidden_states_scale: torch.Tensor | None,
        w_wqa: torch.Tensor | None = None,
        ws_wqa: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.fused_wqa_wkv.weight if w_wqa is None else w_wqa
        weight_scale = (
            self.fused_wqa_wkv.weight_scale if ws_wqa is None else ws_wqa
        )
        if hidden_states_fp8 is not None and hidden_states_scale is not None:
            if getattr(self.fused_wqa_wkv, KEEP_VLLM_LAYOUT_ATTR, False):
                weight = weight.transpose(0, 1)
            return quantized_matmul_prequant(
                hidden_states_fp8,
                hidden_states_scale,
                weight,
                weight_scale,
            )
        return self.fused_wqa_wkv(hidden_states)[0]

    @staticmethod
    def _as_kernel_cache_view(cache: torch.Tensor | None) -> torch.Tensor | None:
        if cache is None:
            return None
        return (
            cache
            if cache.dtype in (torch.uint8, torch.int32)
            else cache.view(torch.uint8)
        )

    def _get_dummy_main_cache(self, device: torch.device) -> torch.Tensor:
        dummy = self.__dict__.get("_dummy_main_cache_kv")
        if dummy is None or dummy.device != device:
            dummy = (
                torch.zeros((0, 1, 128), dtype=torch.int32, device=device)
                if self.compress_ratio > 0
                else torch.zeros(
                    (1, 1, 1, self.head_dim), dtype=torch.uint8, device=device
                )
            )
            object.__setattr__(self, "_dummy_main_cache_kv", dummy)
        return dummy

    def project_wkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "wkv_weight"):
            return quantized_matmul(
                hidden_states, self.wkv_weight, self.wkv_weight_scale
            )
        qr_kv = self.fused_wqa_wkv(hidden_states)[0]
        _, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return kv

    def store_context_kv(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_attn_metadata: Any,
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        kv = self.kv_rope_op(kv, positions, self.rotary_emb.cos_sin_cache)
        orig_sw_cache = self._as_kernel_cache_view(
            _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        )
        sw_cache = (
            orig_sw_cache
            if orig_sw_cache is not None
            else torch.zeros(
                (0, 1, 1, self.head_dim), dtype=torch.uint8, device=kv.device
            )
        )
        swa_lens, swa_tables, swa_q_loc, swa_dist = _unpack_meta(swa_attn_metadata)
        # Only the SWA cache write is needed; a 2-head dummy query with the
        # sliding_window=1 op keeps the attention compute negligible.
        out_tensor, new_sw_cache = self.context_kv_attn_op(
            kv.new_zeros((kv.shape[0], 2, self.head_dim)),
            kv,
            sw_cache,
            swa_lens,
            swa_tables,
            swa_q_loc,
            swa_dist,
            self._get_dummy_main_cache(kv.device),
            swa_lens,
            (positions + 1).to(torch.int32),
            swa_tables,
            swa_q_loc,
            swa_dist,
            self.attn_sink.new_zeros((2,), device=kv.device),
        )
        if (
            orig_sw_cache is not None
            and new_sw_cache is not None
            and new_sw_cache.numel() > 0
        ):
            orig_sw_cache.copy_(new_sw_cache)
        return out_tensor

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
        # Cache the wkv half of fused_wqa_wkv so DSpark skips the wq_a GEMM.
        if self.non_causal_multi_token_decode and not hasattr(self, "wkv_weight"):
            f_w = self.fused_wqa_wkv.weight
            if getattr(self.fused_wqa_wkv, KEEP_VLLM_LAYOUT_ATTR, False):
                f_w = f_w.transpose(0, 1)
            f_s = getattr(self.fused_wqa_wkv, "weight_scale", None)
            if f_s is not None and f_w.shape[1] == self.q_lora_rank + self.head_dim:
                for attr, t in (
                    ("wkv_weight", f_w[:, self.q_lora_rank :]),
                    ("wkv_weight_scale", f_s[..., self.q_lora_rank :]),
                ):
                    buf = torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)
                    setattr(self, attr, torch.nn.Parameter(buf, requires_grad=False))
                if f_w.device.type == "tpu":
                    synchronize_tensors(
                        [self.wkv_weight, self.wkv_weight_scale], wait=True
                    )
        for lin_mod, expected_shape, name in (
            (
                self.wq_b,
                (self.q_lora_rank, self.n_local_heads * self.head_dim),
                "wq_b",
            ),
            (
                self.wo_a,
                (
                    self.n_local_heads * self.head_dim // self.n_local_groups,
                    self.n_local_groups * self.o_lora_rank,
                ),
                "wo_a",
            ),
            (
                self.wo_b,
                (self.n_local_groups * self.o_lora_rank, self.hidden_size),
                "wo_b",
            ),
        ):
            weight = lin_mod.weight
            if tuple(weight.shape) != expected_shape:
                assert tuple(weight.shape) == expected_shape[::-1], (
                    f"{self.prefix}.{name}: expected {list(expected_shape[::-1])}, "
                    f"got {list(weight.shape)}."
                )
                out = torch.empty(
                    expected_shape, dtype=weight.dtype, device=weight.device
                )
                out.copy_(weight.data.t())
                lin_mod.weight = torch.nn.Parameter(out, requires_grad=False)
        if (
            envs.VLLM_DSV41_TP_ATTN
            and not self.non_causal_multi_token_decode
            and get_tensor_model_parallel_world_size() > 1
            and not (
                self._tp_attn_encoder_only
                and getattr(self, "_supports_ced_decoder", False)
            )
        ):
            self._build_tp_attn_weights()
        elif get_tensor_model_parallel_world_size() > 1:
            rank = get_tensor_model_parallel_rank()
            self.register_buffer(
                "_tp_attn_rank_idx",
                torch.tensor([rank], dtype=torch.int64, device=self.wq_b.weight.device),
                persistent=False,
            )
            if not self.non_causal_multi_token_decode:
                self._build_tp_wo_b_weights()

    def _build_tp_wo_b_weights(self) -> None:
        """Replicates wo_b across TP ranks for token-sharded AllToAll wo_b."""
        wo_b_scale = getattr(self.wo_b, "weight_scale", None)
        if wo_b_scale is None:
            return
        wo_b_s = wo_b_scale.data
        if wo_b_s.ndim not in (1, 4):
            return
        tp_size = get_tensor_model_parallel_world_size()

        def gather(x: torch.Tensor, dim: int) -> torch.Tensor:
            if x.element_size() == 1:
                out = tp_all_gather(x.view(torch.uint8), dim).view(x.dtype)
            else:
                out = tp_all_gather(x, dim)
            return out.clone(memory_format=torch.contiguous_format)

        buffers = {
            "_tp_attn_wo_b_w": gather(self.wo_b.weight.data, 0),
            "_tp_attn_wo_b_s": (
                gather(wo_b_s, 1)
                if wo_b_s.ndim == 4
                else gather(wo_b_s.unsqueeze(0), 0).view(1, tp_size, 1, -1)
            ),
        }
        device = self.wo_b.weight.device
        if device.type == "tpu":
            synchronize_tensors(list(buffers.values()), wait=True)
            _ = _get_quantized_matmul_op()
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=False)

    def write_encoder_global_kv(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Write Layer 20 global KV and Indexer k_cache at EncoderStage end."""
        if positions.ndim > 1:
            positions = positions.flatten()
        compressor = getattr(self, "compressor", None)
        latent = (
            compressor(hidden_states, positions) if compressor is not None else None
        )
        if compressor is not None and latent is not None:
            compressor.insert_cache(latent, positions, self.rotary_emb)
        if self.indexer is not None and self.indexer.owns_k:
            assert latent is not None
            self.indexer.write_k_cache(latent, positions, self.indexer_rotary_emb)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        input_is_token_sharded: bool = False,
        return_token_sharded: bool = False,
        ced_decoder: bool | None = None,
        ced_global_from_encoder: bool | None = None,
        tp_rank_idx: torch.Tensor | None = None,
        hidden_states_fp8: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        num_tokens = positions.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        is_csa = self.compress_ratio > 0
        eff_ced_decoder = self._ced_decoder if ced_decoder is None else ced_decoder
        eff_ced_global_from_encoder = (
            self._ced_global_from_encoder
            if ced_global_from_encoder is None
            else ced_global_from_encoder
        )
        compressor = (
            None if eff_ced_global_from_encoder else getattr(self, "compressor", None)
        )
        cos_sin = self.rotary_emb.cos_sin_cache
        if (
            envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS > 0
            and "_tp_attn_wq_b_w" in self._buffers
            and not eff_ced_decoder
            and (not self._tp_attn_encoder_only or self._in_encoder_stage)
        ):
            # VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS splits this module's bucket
            # ladder into head-parallel (replicated) and TP_ATTN (token-sharded)
            # shape_variants traces. Lift both branches' module tensors here in a
            # fixed order so their FX placeholder signatures agree (unused
            # aliases are DCE'd), as in l3b_min_tokens_shape_variants_fix.
            _ = self._tp_attn_ready(num_tokens, tp_size)
            _ = [
                torch.ops.aten.alias.default(t)
                for t in (
                    self.wq_b.weight,
                    getattr(self.wq_b, "weight_scale", None),
                    self.wo_a.weight,
                    getattr(self.wo_a, "weight_scale", None),
                    self.wo_b.weight,
                    getattr(self.wo_b, "weight_scale", None),
                    self.attn_sink,
                    self._tp_attn_wq_b_w,
                    self._tp_attn_wq_b_s,
                    self._tp_attn_wo_a_w,
                    self._tp_attn_wo_a_s,
                    self._tp_attn_wo_b_w,
                    self._tp_attn_wo_b_s,
                    self._tp_attn_sink,
                    self._tp_attn_rank_idx,
                    self._tp_attn_rank_i32,
                )
                if t is not None
            ]
        elif "_tp_attn_wo_b_w" in self._buffers:
            _ = [
                torch.ops.aten.alias.default(t)
                for t in (
                    self.wo_b.weight,
                    (
                        self.wo_b.weight_scale
                        if "weight_scale" in self.wo_b._parameters
                        else None
                    ),
                    self._tp_attn_wo_b_w,
                    self._tp_attn_wo_b_s,
                )
                if t is not None
            ]
        use_tp_idx = self.indexer is not None and self.indexer.use_tp_indexer(
            num_tokens
        )
        qr: torch.Tensor | None = None
        qr_for_indexer: torch.Tensor | None = None
        qr_fp8_for_indexer: torch.Tensor | None = None
        qr_scale_for_indexer: torch.Tensor | None = None
        qr_q_fp8: torch.Tensor | None = None
        qr_q_scale: torch.Tensor | None = None
        # Token-parallel attention: all heads for this rank's T/tp tokens.
        use_tp_attn = (
            input_is_token_sharded
            and return_token_sharded
            and not eff_ced_decoder
            and (not self._tp_attn_encoder_only or self._in_encoder_stage)
            and self._tp_attn_ready(num_tokens, tp_size)
        )
        w_wqa = self.fused_wqa_wkv.weight
        ws_wqa = self.fused_wqa_wkv._parameters.get("weight_scale", None)
        w_qnorm = self.q_norm.weight
        w_kvnorm = self.kv_norm.weight
        w_wqb = self.wq_b.weight
        ws_wqb = self.wq_b._parameters.get("weight_scale", None)
        wq_b_s = ws_wqb
        has_wqb_bias = self.wq_b._parameters.get("bias", None) is not None
        if use_tp_attn:
            use_fused_q_rope = bool(
                envs.VLLM_DSV41_FUSED_Q_ROPE
                and self._tp_attn_wq_b_s.ndim == 1
                and not has_wqb_bias
                and (self.n_local_heads * tp_size) % 8 == 0
            )
        else:
            use_fused_q_rope = bool(
                envs.VLLM_DSV41_FUSED_Q_ROPE
                and wq_b_s is not None
                and wq_b_s.ndim == 1
                and not has_wqb_bias
                and self.n_local_heads % 8 == 0
            )
        rank_idx = (
            tp_rank_idx
            if tp_rank_idx is not None
            else getattr(self, "_tp_attn_rank_idx", None)
        )
        if use_tp_attn:
            tp_rank = get_tensor_model_parallel_rank()
            positions_local = tp_shard_slice(positions, tp_size, tp_rank, rank_idx)
        token_sharded_qkv = (
            input_is_token_sharded
            and tp_size > 1
            and num_tokens % tp_size == 0
            and (compressor is None or use_tp_attn)
            and (self.indexer is None or use_tp_idx)
        )
        use_qkv_prequant = (
            use_fused_q_rope
            and (not token_sharded_qkv or use_tp_attn)
            and ("weight" in self.q_norm._parameters)
            and self.indexer is None
        )
        if token_sharded_qkv:
            if not use_tp_attn:
                tp_rank = get_tensor_model_parallel_rank()
                positions_local = tp_shard_slice(positions, tp_size, tp_rank, rank_idx)
            qr_kv_local = self._apply_fused_wqa_wkv(
                hidden_states,
                hidden_states_fp8,
                hidden_states_scale,
                w_wqa=w_wqa,
                ws_wqa=ws_wqa,
            )
            if use_qkv_prequant:
                qr_q_fp8, qr_q_scale, kv_local = self.qkv_norm_rope_quant_op(
                    qr_kv_local,
                    w_qnorm,
                    w_kvnorm,
                    positions_local,
                    cos_sin,
                )
                if use_tp_idx:
                    qr_fp8_for_indexer = qr_q_fp8
                    qr_scale_for_indexer = qr_q_scale
                kv = tp_all_gather(kv_local, dim=0)
            else:
                qr_local, kv_local = qr_kv_local.split(
                    [self.q_lora_rank, self.head_dim], dim=-1
                )
                qr_local = self.q_norm(qr_local)
                kv_local = self.kvnorm_rope_op(
                    kv_local, w_kvnorm, positions_local, cos_sin
                )
                if use_tp_idx:
                    qr_for_indexer = qr_local
                if use_tp_attn:
                    # qr stays local; only the normed, roped kv is gathered.
                    qr = qr_local
                    kv = tp_all_gather(kv_local, dim=0)
                else:
                    qr_kv_normed = tp_all_gather(
                        torch.cat([qr_local, kv_local], dim=-1), dim=0
                    )
                    qr, kv = qr_kv_normed.split(
                        [self.q_lora_rank, self.head_dim], dim=-1
                    )
        else:
            if input_is_token_sharded:
                hidden_states = tp_all_gather(hidden_states, dim=0)
                hidden_states_fp8 = None
                hidden_states_scale = None
            qr_kv = self._apply_fused_wqa_wkv(
                hidden_states,
                hidden_states_fp8,
                hidden_states_scale,
                w_wqa=w_wqa,
                ws_wqa=ws_wqa,
            )
            if use_qkv_prequant:
                qr_fp8, qr_scale, kv = self.qkv_norm_rope_quant_op(
                    qr_kv,
                    w_qnorm,
                    w_kvnorm,
                    positions,
                    cos_sin,
                )
                if self.indexer is not None:
                    qr_fp8_for_indexer = qr_fp8
                    qr_scale_for_indexer = qr_scale
                if use_tp_attn:
                    qr_q_fp8 = tp_shard_slice(
                        qr_fp8.view(torch.uint8),
                        tp_size,
                        tp_rank,
                        rank_idx,
                    ).view(qr_fp8.dtype)
                    qr_q_scale = tp_shard_slice(qr_scale, tp_size, tp_rank, rank_idx)
                else:
                    qr_q_fp8 = qr_fp8
                    qr_q_scale = qr_scale
            else:
                qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
                qr = self.q_norm(qr)
                kv = self.kvnorm_rope_op(kv, w_kvnorm, positions, cos_sin)

        if use_tp_attn:
            if qr_q_fp8 is not None and qr_q_scale is not None:
                q = self.wq_b_q_rope_prequant_op(
                    qr_q_fp8,
                    qr_q_scale,
                    positions_local,
                    cos_sin,
                    self._tp_attn_wq_b_w,
                    self._tp_attn_wq_b_s,
                )
            else:
                assert qr is not None
                qr_q = (
                    qr
                    if qr.shape[0] != num_tokens
                    else tp_shard_slice(qr, tp_size, tp_rank, rank_idx)
                )
                if use_fused_q_rope:
                    q = self.wq_b_q_rope_op(
                        qr_q,
                        positions_local,
                        cos_sin,
                        self._tp_attn_wq_b_w,
                        self._tp_attn_wq_b_s,
                    )
                else:
                    q = quantized_matmul(
                        qr_q, self._tp_attn_wq_b_w, self._tp_attn_wq_b_s
                    ).view(-1, self.n_local_heads * tp_size, self.head_dim)
                    q = self.q_rope_op(q, positions_local, cos_sin)
        else:
            if qr_q_fp8 is not None and qr_q_scale is not None:
                wq_b_w = w_wqb
                if getattr(self.wq_b, KEEP_VLLM_LAYOUT_ATTR, False):
                    wq_b_w = wq_b_w.transpose(0, 1)
                assert wq_b_s is not None
                q = self.wq_b_q_rope_prequant_op(
                    qr_q_fp8,
                    qr_q_scale,
                    positions,
                    cos_sin,
                    wq_b_w,
                    wq_b_s,
                )
            elif use_fused_q_rope:
                assert qr is not None and wq_b_s is not None
                wq_b_w = w_wqb
                if getattr(self.wq_b, KEEP_VLLM_LAYOUT_ATTR, False):
                    wq_b_w = wq_b_w.transpose(0, 1)
                q = self.wq_b_q_rope_op(
                    qr,
                    positions,
                    cos_sin,
                    wq_b_w,
                    wq_b_s,
                )
            else:
                assert qr is not None
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self.q_rope_op(q, positions, cos_sin)

        latent = (
            compressor(hidden_states, positions) if compressor is not None else None
        )
        topk_local = use_tp_attn and self._keep_topk_local(num_tokens)
        if self.indexer is not None:
            if self.indexer.owns_k and latent is not None:
                self.indexer.write_k_cache(latent, positions, self.indexer_rotary_emb)
            indexer_weights = self.indexer.weights_proj(hidden_states)[0]
            topk_indices = self.indexer(
                hidden_states,
                qr_for_indexer if qr_for_indexer is not None else qr,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
                latent=None,
                step_metadata_fn=lambda: self._get_or_compute_indexer_step_metadata(
                    positions, ced_decoder=eff_ced_decoder
                ),
                tp_rank_idx=tp_rank_idx,
                keep_topk_local=topk_local,
                query_fp8=qr_fp8_for_indexer,
                query_scale=qr_scale_for_indexer,
            )
        elif is_csa:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[
                : num_tokens // tp_size if topk_local else num_tokens
            ]
        else:
            topk_indices = None

        if compressor is not None and latent is not None:
            compressor.insert_cache(latent, positions, self.rotary_emb)

        attn_ctx = get_forward_context().attn_metadata
        kv_holder = self._kv_source_module()
        if isinstance(attn_ctx, dict):
            swa_meta = attn_ctx[self.swa_cache_layer.prefix]
            main_meta = attn_ctx.get(kv_holder.prefix)
        else:
            swa_meta = main_meta = attn_ctx

        orig_sw_cache = self._as_kernel_cache_view(
            _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        )
        sw_cache = (
            orig_sw_cache
            if orig_sw_cache is not None
            else torch.zeros(
                (0, 1, 1, self.head_dim), dtype=torch.uint8, device=kv.device
            )
        )
        swa_lens, swa_tables, swa_q_loc, swa_dist = _unpack_meta(swa_meta)

        kv_entry = getattr(kv_holder, "kv_cache", None)
        nope_entry, rope_entry = (
            kv_entry if isinstance(kv_entry, tuple) else (kv_entry, None)
        )
        dummy_main = self._get_dummy_main_cache(kv.device)
        if is_csa and main_meta is not None:
            main_cache_kv = self._as_kernel_cache_view(_live_cache(nope_entry))
            if main_cache_kv is None:
                main_cache_kv = dummy_main
            m_lens, main_tables, main_q_loc, main_dist = _unpack_meta(main_meta)
            main_kv_lens = (
                m_lens // kv_holder.compress_ratio if m_lens is not None else None
            )
        else:
            main_cache_kv = dummy_main
            main_kv_lens, main_tables, main_q_loc, main_dist = _unpack_meta(swa_meta)

        extra = topk_indices if is_csa else (positions + 1).to(torch.int32)
        if use_tp_attn:
            main_cache_rope = (
                self._as_kernel_cache_view(_live_cache(rope_entry)) if is_csa else None
            )
            attn_output, new_sw_cache = self._get_tp_attn_op()(
                q,
                kv,
                sw_cache,
                swa_lens,
                swa_tables,
                swa_q_loc,
                swa_dist,
                main_cache_kv,
                main_kv_lens,
                extra,
                main_tables,
                main_q_loc,
                main_dist,
                self._tp_attn_sink,
                self._tp_attn_rank_i32,
                *(
                    ()
                    if not is_csa
                    else (dummy_main if main_cache_rope is None else main_cache_rope,)
                ),
            )
            orig_sw_cache.copy_(new_sw_cache)
            z = self.o_proj_op(
                attn_output,
                positions_local,
                cos_sin,
                self._tp_attn_wo_a_w,
                self._tp_attn_wo_a_s,
            )
            return quantized_matmul(z, self._tp_attn_wo_b_w, self._tp_attn_wo_b_s)
        tail_operands: tuple = (
            kv,
            sw_cache,
            swa_lens,
            swa_tables,
            swa_q_loc,
            swa_dist,
            main_cache_kv,
            main_kv_lens,
            extra,
            main_tables,
            main_q_loc,
            main_dist,
            self.attn_sink,
        )
        has_union = False
        if is_csa:
            main_cache_rope = self._as_kernel_cache_view(_live_cache(rope_entry))
            tail_operands += (
                dummy_main if main_cache_rope is None else main_cache_rope,
            )
            max_num_seqs = (
                int(main_q_loc.shape[0]) - 1 if main_q_loc is not None else num_tokens
            )
            u_params = (
                resolve_csa_union_params(
                    num_tokens,
                    int(topk_indices.shape[1]),
                    max_num_seqs,
                    ced_decoder=eff_ced_decoder,
                )
                if envs.VLLM_DSV41_CSA_UNION
                and main_cache_kv.shape[0] > 0
                and self.n_local_heads <= 128
                and topk_indices is not None
                and topk_indices.shape[1] % 128 == 0
                else None
            )
            if u_params is not None:
                b_q, m_union = u_params
                total_cache_rows = int(main_cache_kv.shape[0] * main_cache_kv.shape[1])
                cached_u = self._get_or_compute_csa_union(
                    topk_indices,
                    total_cache_rows,
                    m_union,
                    b_q,
                    ced_decoder=eff_ced_decoder,
                )
                tail_operands += tuple(cached_u)
                has_union = True

        attn_output, new_sw_cache = self._get_attn_op(
            has_union, ced_decoder=eff_ced_decoder
        )(q, *tail_operands)
        if orig_sw_cache is not None:
            orig_sw_cache.copy_(new_sw_cache)

        z = self.o_proj_op(
            attn_output,
            positions,
            cos_sin,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        if return_token_sharded and tp_size > 1:
            if (
                num_tokens <= 512
                and num_tokens % (8 * tp_size) == 0
                and "_tp_attn_wo_b_w" in self._buffers
            ):
                z_a2a = tp_all_to_all(z)
                z_local = (
                    z_a2a.view(tp_size, -1, int(z.shape[1]))
                    .transpose(0, 1)
                    .reshape(-1, tp_size * int(z.shape[1]))
                )
                return quantized_matmul(
                    z_local, self._tp_attn_wo_b_w, self._tp_attn_wo_b_s
                )
            out_parallel = self.wo_b.quant_method.apply(self.wo_b, z, None)
            return tp_reduce_scatter(out_parallel, dim=0)
        return self.wo_b(z)
