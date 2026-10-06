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
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v41 import attention as deepseek_v41_flash_attention
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import tp_all_gather, tp_reduce_scatter
from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_kernel
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

BATCH_AXIS = _dsv4_attn_op.BATCH_AXIS
VllmDeepseekV4SWACache = _dsv4_attn_op.VllmDeepseekV4SWACache
_attention_csa = _deepseek_v41_flash_attn_op._attention_csa
_attention_swa = _deepseek_v41_flash_attn_op._attention_swa
get_packed_mla_head_size = _dsv4_attn_op.get_packed_mla_head_size
_Q_ROPE_SPECS = (P(None, "model", None), P(), P())
_KV_NORM_ROPE_SPECS = (P(), P(), P(), P())
_O_PROJ_SPECS = (P(None, "model", None), P(), P(), P(), P())


def _kvnorm_rope_jax(
    x: jax.Array,
    weight: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    return rope_kernel.rmsnorm_rope(x, weight, positions, cos_sin_cache, eps=eps)


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
        self._ced_decoder: bool = False
        self._ced_global_from_encoder: bool = False

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

    def _leader_group_bounds(self) -> tuple[Any, int]:
        cached = self.__dict__.get("_cached_leader_group_bounds")
        if cached is not None:
            return cached
        my_lid = extract_layer_index(self.prefix)
        my_ced = bool(getattr(self, "_ced_decoder", False))
        min_lid = my_lid
        max_lid = my_lid
        first_leader: Any = self
        for mod in self._static_forward_context.values():
            if (
                isinstance(mod, VllmDeepseekV41MLAAttention)
                and mod.indexer is not None
                and bool(getattr(mod, "_ced_decoder", False)) == my_ced
            ):
                lid = extract_layer_index(mod.prefix)
                if lid < min_lid:
                    min_lid = lid
                    first_leader = mod
                if lid > max_lid:
                    max_lid = lid
        res = (first_leader, max_lid)
        if not torch.compiler.is_compiling():
            object.__setattr__(self, "_cached_leader_group_bounds", res)
        return res

    def _get_or_compute_indexer_step_metadata(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # Indexer step metadata depends only on positions and attention metadata,
        # so the first indexer layer of a stage computes it and the rest reuse it.
        first_leader, last_leader_lid = self._leader_group_bounds()
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

    @property
    def attn_op(self):
        op = self.__dict__.get("_attn_op_instance")
        if op is not None:
            return op
        _ = self.kvnorm_rope_op
        if self.indexer is not None and not torch.compiler.is_compiling():
            _ = self._leader_group_bounds()
        swa_tensor = _live_cache(getattr(self.swa_cache_layer, "kv_cache", None))
        if swa_tensor is None:
            prof_op = self.__dict__.get("_prof_attn_op_instance")
            if prof_op is None:
                prof_op, _ = self._build_attn_op()
                object.__setattr__(self, "_prof_attn_op_instance", prof_op)
            return prof_op
        op, is_real = self._build_attn_op()
        assert is_real
        object.__setattr__(self, "_attn_op_instance", op)
        self.__dict__.pop("_prof_attn_op_instance", None)
        if self.indexer is not None and not torch.compiler.is_compiling():
            idx_kv = _live_cache(getattr(self.indexer.k_cache, "kv_cache", None))
            if idx_kv is not None:
                _ = self.indexer._get_step_metadata_op(int(idx_kv.shape[3]))
        return op

    def _build_attn_op(self):
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

        attn_fn = _attention_csa if is_csa else _attention_swa
        wrapped_fn = functools.partial(
            attn_fn,
            sm_scale=self.scale,
            sliding_window=self.window_size,
            logical_page_size=logical_page_size,
            ced_decoder=self._ced_decoder,
            **({"two_caches_same_buffer": two_caches_same_buffer} if is_csa else {}),
        )
        op_type = "csa" if is_csa else "swa"
        op_name = (
            f"pallas::deepseek_v41_attention_{op_type}"
            f"{'_ced' if self._ced_decoder else ''}"
            f"_p{logical_page_size}"
            f"{'_aliased' if two_caches_same_buffer else ''}"
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

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
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
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        num_tokens = positions.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        is_csa = self.compress_ratio > 0
        compressor = (
            None if self._ced_global_from_encoder else getattr(self, "compressor", None)
        )
        if input_is_token_sharded:
            hidden_states = tp_all_gather(hidden_states, dim=0)

        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(qr)

        cos_sin = self.rotary_emb.cos_sin_cache
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_rope_op(q, positions, cos_sin)
        kv = self.kvnorm_rope_op(kv, self.kv_norm.weight, positions, cos_sin)

        latent = (
            compressor(hidden_states, positions) if compressor is not None else None
        )
        if self.indexer is not None:
            indexer_weights = self.indexer.weights_proj(hidden_states)[0]
            topk_indices = self.indexer(
                hidden_states,
                qr,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
                latent=latent,
                step_metadata=self._get_or_compute_indexer_step_metadata(positions),
            )
        elif is_csa:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[:num_tokens]
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
                (0, 1, 1, self.head_dim), dtype=torch.uint8, device=qr.device
            )
        )
        swa_lens, swa_tables, swa_q_loc, swa_dist = _unpack_meta(swa_meta)

        kv_entry = getattr(kv_holder, "kv_cache", None)
        nope_entry, rope_entry = (
            kv_entry if isinstance(kv_entry, tuple) else (kv_entry, None)
        )
        dummy_main = self._get_dummy_main_cache(qr.device)
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
        if is_csa:
            main_cache_rope = self._as_kernel_cache_view(_live_cache(rope_entry))
            tail_operands += (
                dummy_main if main_cache_rope is None else main_cache_rope,
            )

        attn_output, new_sw_cache = self.attn_op(q, *tail_operands)
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
            out_parallel = self.wo_b.quant_method.apply(self.wo_b, z, None)
            return tp_reduce_scatter(out_parallel, dim=0)
        return self.wo_b(z)
