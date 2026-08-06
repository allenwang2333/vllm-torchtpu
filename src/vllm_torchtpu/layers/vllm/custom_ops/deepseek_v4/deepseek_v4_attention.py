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
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (KVCacheSpec, MLAAttentionSpec,
                                        SlidingWindowMLASpec)

from vllm_torchtpu.kernels.deepseek_v4.mla import mla_ragged_paged_attention
from vllm_torchtpu.kernels.deepseek_v4.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_indexer import \
    VllmDeepseekV4Indexer
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

from vllm.model_executor.layers.attention_layer_base import (  # isort: skip
    AttentionBackend, AttentionLayerBase)

logger = init_logger(__name__)


def _is_tracing() -> bool:
    if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
        return True
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return True
    return False


_attn_op_cache = {}

BATCH_AXIS = None


def check_nan(name: str, tensor: torch.Tensor, prefix: str):
    if os.getenv("VLLM_CHECK_NAN") != "1" or tensor is None:
        return
    if (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()) or (
            hasattr(torch, "compiler") and torch.compiler.is_compiling()):
        return
    is_fake = (tensor.device.type in ("meta", "cpu") or type(tensor).__name__
               in ("FakeTensor", "SymbolicVariable", "TensorVariable"))
    if is_fake:
        return

    # isnan/isinf are not implemented for Float8 types; cast to float32 first.
    is_float8 = (tensor.dtype == torch.float8_e4m3fn
                 or tensor.dtype == torch.float8_e5m2)
    if tensor.dtype.is_floating_point or is_float8:
        t_check = tensor.to(torch.float32) if is_float8 else tensor
        has_nan = torch.isnan(t_check).any().item()
        has_inf = torch.isinf(t_check).any().item()
        if has_nan or has_inf:
            nan_token_0 = (torch.isnan(t_check[0]).any().item() or torch.isinf(
                t_check[0]).any().item()) if t_check.shape[0] > 0 else False
            logger.debug(
                f"[NAN_DETECTION] NaN/Inf detected in {name} at layer {prefix}! shape={list(tensor.shape)} has_nan={has_nan} has_inf={has_inf} token0_has_nan_inf={nan_token_0}"
            )
            logger.debug(
                f"    min={t_check.min().item():.6f} max={t_check.max().item():.6f} mean={t_check.mean().item():.6f}"
            )
        elif os.getenv("DSV4_TRACE_ACTIVATIONS") == "1":
            t_mean = t_check.mean().item()
            t_std = (t_check - t_mean).pow(2).mean().sqrt().item()
            logger.debug(
                f"[ACT_TRACE] [{prefix}] {name}: shape={list(tensor.shape)} mean={t_mean:.6f} std={t_std:.6f} min={t_check.min().item():.6f} max={t_check.max().item():.6f}"
            )


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
                if jnp.issubdtype(arg.dtype, jnp.floating):
                    is_f8 = (arg.dtype == jnp.float8_e4m3fn
                             or arg.dtype == jnp.float8_e5m2)
                    arg_check = arg.astype(
                        jnp.float32) if is_f8 else arg.astype(jnp.float32)
                    clean_arg = jnp.where(
                        jnp.isinf(arg_check) | jnp.isnan(arg_check), 0.0,
                        arg_check)
                    dummy_sum = dummy_sum + jnp.sum(clean_arg) * 0
                else:
                    dummy_sum = dummy_sum + jnp.sum(arg) * 0
        return dummy_out + dummy_sum, sw_cache + dummy_sum

    def _attention_local(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                         swa_cu_q_lens, swa_distribution, main_cache_kv,
                         main_kv_lens, extra, main_page_indices,
                         main_cu_q_lens, main_distribution, attention_sinks):
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
            swa_m = jnp.full((q.shape[0], q.shape[1], 1),
                             -1e9,
                             dtype=jnp.float32)
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
                ))

        dummy_sum = 0.0
        for arg in (q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                    swa_cu_q_lens, swa_distribution, main_cache_kv,
                    main_kv_lens, extra, main_page_indices, main_cu_q_lens,
                    main_distribution, attention_sinks):
            if arg is not None:
                if jnp.issubdtype(arg.dtype, jnp.floating):
                    arg_check = arg.astype(jnp.float32)
                    clean_arg = jnp.where(
                        jnp.isinf(arg_check) | jnp.isnan(arg_check), 0.0,
                        arg_check)
                    dummy_sum = dummy_sum + jnp.sum(clean_arg) * 0
                else:
                    dummy_sum = dummy_sum + jnp.sum(arg) * 0

        if swa_only:
            if sw_cache_was_5d:
                updated_sw_cache = jnp.expand_dims(updated_sw_cache, axis=2)
            return swa_output + dummy_sum, updated_sw_cache

        # Matches the reference implementation exactly (no is_csa gate):
        # when the SWA and compressed/main KV caches are the same physical
        # buffer, the compressed-KV read must see the buffer AFTER the SWA
        # kernel's write in this same pass (updated_sw_cache), not the stale
        # pre-write snapshot -- for both CSA and HCA layers. A prior commit
        # (3cd0d62f) added `and not is_csa` here in the same change that
        # widened `two_caches_same_buffer` from `is_csa` to `not swa_only`,
        # net effect: inverted which layer type gets this substitution, with
        # no stated rationale covering this specific line.
        if two_caches_same_buffer:
            main_cache_kv = updated_sw_cache

        if swa_l is not None and swa_l.ndim == 3:
            swa_l = jnp.squeeze(swa_l, axis=-1)
        if swa_m is not None and swa_m.ndim == 3:
            swa_m = jnp.squeeze(swa_m, axis=-1)

        # Bisection probe for the CSA/HCA decode blowup: |out| <= |acc| in
        # mla.py (its epilogue clamps the denominator to >= 1), and acc has
        # exactly two sources -- this SWA unnormalized-accumulator handoff,
        # and pv = p @ kv with p <= 1. Returning swa_output here in place of
        # the MLA output routes the SWA accumulator through the *existing*
        # dump path (sub_02b_post_kernel_output) with an identical shape and
        # dtype, so no extra kernel outputs or host callbacks are needed
        # (jax.debug.* breaks this codebase's AOT compile cache -- see
        # docs/bisection_progress_report.md 6.2). If the ~1e27 magnitude
        # shows up here, the bug is in mla_swa.py's unnormalized path; if
        # this is bounded, the bug is in mla.py's own KV read/accumulate.
        # NOTE: this edit is inside the traced function, so it requires
        # `rm -rf /dev/shm/torch_tpu_cache/default/*` to take effect.
        if os.getenv("DSV4_DEBUG_RETURN_SWA") == "1":
            if sw_cache_was_5d:
                updated_sw_cache = jnp.expand_dims(updated_sw_cache, axis=2)
            return swa_output + dummy_sum, updated_sw_cache

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
            swa_accumulation=swa_output,
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
        get_valid_spec(mesh, "model"),  # attention_sinks
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
    )(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices, swa_cu_q_lens,
      swa_distribution, main_cache_kv, main_kv_lens, extra, main_page_indices,
      main_cu_q_lens, main_distribution, attention_sinks)
    return res_out, res_cache


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
            self.block_size = min(
                compressed_kv_cache_bz // csa_compression_ratio, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )


_orig_ds_v4_attention_cls = getattr(
    ds_v4_amd_model, "DeepseekV4Attention",
    getattr(ds_v4_amd_model, "DeepseekV4ROCMAiterMLAAttention", object))


# ds_v4_amd_model.DeepseekV4Attention must come first: on vLLM builds where it
# resolves to DeepseekV4ROCMAiterMLAAttention (see sitecustomize.py), that
# class already inherits AttentionLayerBase itself, so listing
# AttentionLayerBase first here creates an unresolvable MRO (it would need to
# precede a class it's also reachable through). AttentionLayerBase is a pure
# ABC with no concrete methods, so this ordering doesn't change dispatch.
class VllmDeepseekV4MLAAttention(_orig_ds_v4_attention_cls,
                                 AttentionLayerBase):

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
            cast(Any, _orig_ds_v4_attention_cls).__init__(
                self,
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
            self.custom_prefix = prefix
            _real_mla_attn = getattr(self, "mla_attn", self)
            # self.mla_attn is collapsed to alias `self` below (to sidestep the
            # grandchild-module get_kv_cache_spec overwrite issue), which would
            # otherwise silently drop access to swa_cache_layer/compressor/indexer
            # since those live on the wrapper (_real_mla_attn), not on `self`.
            for _attr in ("swa_cache_layer", "compressor", "indexer"):
                if hasattr(_real_mla_attn, _attr) and not hasattr(self, _attr):
                    object.__setattr__(self, _attr,
                                       getattr(_real_mla_attn, _attr))
            self.__dict__['mla_attn'] = self
            for module in self.modules():
                if module is not self and hasattr(module, "get_kv_cache_spec"):
                    if module.__class__.__name__ in (
                            "DeepseekV4MultiHeadLatentAttentionWrapper",
                            "DeepseekV4MLAAttention", "DeepseekV4Attention",
                            "DeepseekV4ROCMAiterMLAAttention"):
                        module.get_kv_cache_spec = self.get_kv_cache_spec
        finally:
            dsv4_attention.DeepseekV4Indexer = orig_indexer
            if orig_amd_indexer is not None:
                ds_v4_amd_model.DeepseekV4Indexer = orig_amd_indexer
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4SWACache = orig_swa_cache
        if hasattr(self.mla_attn,
                   "compressor") and self.mla_attn.compressor is not None:
            if self.compress_ratio <= 1:
                object.__setattr__(self.mla_attn.compressor, "k_cache",
                                   self.mla_attn.swa_cache_layer)
            else:
                object.__setattr__(self.mla_attn.compressor, "k_cache",
                                   self.mla_attn.mla_attn)
        hf_config = vllm_config.model_config.hf_config
        object.__setattr__(self, "attn_out_dim",
                           hf_config.num_attention_heads * hf_config.head_dim)
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.n_groups = self.n_local_groups * self.tp_size
        object.__setattr__(self, "attn_op", self._build_attn_op())
        self.num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        compilation_config = vllm_config.compilation_config
        if compilation_config is not None:
            compilation_config.static_forward_context[prefix] = self
            if hasattr(self.mla_attn, "swa_cache_layer"):
                compilation_config.static_forward_context[
                    self.mla_attn.swa_cache_layer.
                    prefix] = self.mla_attn.swa_cache_layer
            if hasattr(self.mla_attn, "indexer") and hasattr(
                    self.mla_attn.indexer, "k_cache"):
                compilation_config.static_forward_context[
                    self.mla_attn.indexer.k_cache.
                    prefix] = self.mla_attn.indexer.k_cache
            if hasattr(self.mla_attn, "compressor") and hasattr(
                    self.mla_attn.compressor, "state_cache"):
                compilation_config.static_forward_context[
                    self.mla_attn.compressor.state_cache.
                    prefix] = self.mla_attn.compressor.state_cache

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_attn_backend(self) -> type[AttentionBackend]:
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        block_size = vllm_config.cache_config.block_size
        comp_ratio = max(1, self.compress_ratio)
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            compress_ratio=min(comp_ratio, block_size),
            alignment=None,
        )
        return spec

    def _build_attn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        # DSV4_FORCE_SWA_ONLY=1 disables the sparse-attention path
        # (CSA top-k + HCA long-range) for every layer, leaving dense
        # local SWA. One-run ablation: if output becomes prompt-faithful
        # the accuracy gap lives in the sparse path; if not, that whole
        # subsystem is exonerated. See docs/dsv4/00_NEXT_PHASE_PROMPT.md.
        swa_only = (self.compress_ratio <= 1
                    or os.getenv("DSV4_FORCE_SWA_ONLY") == "1")
        is_csa = self.compress_ratio == 4
        two_caches_same_buffer = False
        swa_cache = getattr(self.mla_attn, "swa_cache_layer", None)
        if swa_cache is not None and getattr(swa_cache, "block_size", 0) > 0:
            logical_page_size = swa_cache.block_size
        else:
            # Never guess: a wrong logical page size (e.g. 16 vs the real 64)
            # silently misaddresses every SWA cache slot. Missing block_size
            # here usually means the layer was constructed by tpu_inference
            # instead of vllm_torchtpu — fix plugin isolation instead.
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer.block_size unavailable "
                "at attention-op build time; refusing to guess the SWA cache "
                "logical page size.")

        hf_config = vllm_context.vllm_config.model_config.hf_config
        sm_scale = (getattr(self.mla_attn, "scale", None)
                    or getattr(self.mla_attn, "softmax_scale", None)
                    or getattr(self, "softmax_scale", None)
                    or getattr(self, "scale", None)
                    or (hf_config.head_dim**-0.5 if hasattr(
                        hf_config, "head_dim") else None))
        if sm_scale is None:
            raise RuntimeError(
                f"{self.custom_prefix}: attention scale unavailable at "
                "attention-op build time (expected head_dim**-0.5 from "
                "DeepseekV4Attention.__init__).")

        wrapped_fn = functools.partial(
            _deepseek_v4_attention_jax,
            mesh=mesh,
            sm_scale=sm_scale,
            sliding_window=self.window_size,
            logical_page_size=logical_page_size,
            swa_only=swa_only,
            is_csa=is_csa,
            two_caches_same_buffer=two_caches_same_buffer,
        )

        op_type = "swa" if swa_only else ("csa" if is_csa else "hca")
        op_name = f"pallas::deepseek_v4_attention_{op_type}_v2"

        global _attn_op_cache
        if op_name in _attn_op_cache:
            return _attn_op_cache[op_name]

        from jax.sharding import PartitionSpec as P
        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        extra_spec = get_valid_spec(
            mesh, batch_axis) if not is_csa else get_valid_spec(
                mesh, batch_axis, None)
        attn_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                get_valid_spec(mesh, attn_data_axis, attn_head_axis,
                               None),  # q
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
                get_valid_spec(mesh, attn_head_axis),  # attention_sinks
            ),
        )

        def _fake_attn(q, new_kv, sw_cache, *args, **kwargs):
            num_tokens = q.shape[0]
            out_tensor = torch.empty((num_tokens, self.mla_attn.n_local_heads,
                                      self.mla_attn.head_dim),
                                     dtype=q.dtype,
                                     device=q.device)
            return out_tensor, torch.empty_like(sw_cache)

        attn_jax_op.register_fake(_fake_attn)

        _attn_op_cache[op_name] = attn_jax_op
        return attn_jax_op

    def attn_gemm(self, hidden_states):
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        compressor = getattr(self.mla_attn, "compressor", None)
        if compressor is not None:
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
        rms = torch.rsqrt(
            qf.pow(2).mean(dim=-1, keepdim=True) + self.mla_attn.eps)
        qf = qf * rms

        # DeepseekV4ScalingRotaryEmbedding rotates the trailing rotary_dim
        # channels internally, so pass the full head.
        q_rotated, _ = self.mla_attn.rotary_emb(positions, qf)
        return q_rotated.to(orig_dtype)

    def kv_rope(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = kv.dtype
        # DeepseekV4ScalingRotaryEmbedding rotates the trailing rotary_dim
        # channels internally; unsqueeze a head dim so cos/sin broadcast.
        kv_rotated, _ = self.mla_attn.rotary_emb(
            positions,
            kv.unsqueeze(1).to(torch.float32))
        return kv_rotated.squeeze(1).to(orig_dtype)

    def _maybe_dump_sublayer(self, name: str, tensor: torch.Tensor,
                             step: int) -> None:
        # Finer-grained companion to deepseek_v4_patch.py's
        # sub_01..sub_06 dumps (which bracket the *whole* attention module
        # as one black box, "02_post_attn"). These bisect further within
        # that range: 02a is q/kv right after RoPE (before the kernel
        # call), 02b is the kernel's raw output (before _o_proj) -- mirrors
        # tpu-inference's identically-named checkpoints in its
        # attention_impl/forward_mqa so the two can be diffed directly.
        if os.getenv("DSV4_DUMP_HIDDEN_STATES") != "1":
            return
        if torch.compiler.is_compiling():
            return
        prefix = getattr(self, "custom_prefix", "")
        target_layers = os.getenv("DSV4_DUMP_LAYERS", "0,2").split(",")
        target_layer = next(
            (layer for layer in target_layers if f"layers.{layer}." in prefix),
            None)
        if target_layer is None:
            return
        layer_suffix = "" if target_layer == "0" else f"_L{target_layer}"
        name = f"{name}{layer_suffix}"
        from vllm.distributed.parallel_state import \
            get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return
        import numpy as np
        outdir = "/home/patemotter_google_com/workspace/scratch/hidden_states/vllm_torchtpu"
        os.makedirs(outdir, exist_ok=True)
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().to(torch.float32).numpy()
        else:
            arr = np.asarray(tensor)
        np.save(f"{outdir}/sub_{name}_step{step:03d}.npy", arr)

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
        dump_step = getattr(self, "_dsv4_dump_step", 0)
        self._dsv4_dump_step = dump_step + 1

        q = self.wq_b(qr).view(qr.shape[0], self.mla_attn.n_local_heads,
                               self.mla_attn.head_dim)
        q = self.qnorm_rope(q, positions)
        kv = self.kv_rope(kv, positions)
        self._maybe_dump_sublayer("02a_pre_kernel_q", q, dump_step)
        self._maybe_dump_sublayer("02a_pre_kernel_kv", kv, dump_step)
        check_nan("q (after rope)", q, self.custom_prefix)
        check_nan("kv (after rope)", kv, self.custom_prefix)

        topk_indices = None
        compressor = getattr(self.mla_attn, "compressor", None)
        if self.indexer is not None:
            indexer_emb = getattr(self.mla_attn, "indexer_rotary_emb",
                                  getattr(self.mla_attn, "rotary_emb", None))
            topk_indices = self.indexer(hidden_states, qr, indexer_kv_score,
                                        indexer_weights, positions,
                                        indexer_emb)
            if compressor is not None:
                compressor(kv_score, positions, self.mla_attn.rotary_emb)
        elif compressor is not None:
            compressor(kv_score, positions, self.mla_attn.rotary_emb)

        self._dsv4_dump_step_current = dump_step
        res = self.forward_mqa(
            q,
            kv,
            positions,
            None,
            topk_indices=topk_indices,
        )
        return res

    @staticmethod
    def _as_kernel_cache_view(cache: torch.Tensor) -> torch.Tensor:
        # The Pallas MLA/SWA kernels (mla.py, mla_swa.py) hard-require
        # `cache_kv.dtype == uint8`: the cache stores a custom packed byte
        # layout (quantized fp8 nope-values + raw rope bits + per-block
        # scales, see quantize_kv_inputs) rather than a uniform semantic
        # dtype. But vLLM's own KV-cache-spec/allocation pipeline allocates
        # the actual backing tensor as `kv_cache_dtype` (e.g.
        # torch.float8_e4m3fn for `--kv-cache-dtype=fp8_e4m3`), not the
        # `torch.uint8` this class's own get_kv_cache_spec() declares --
        # vLLM does not preserve that per-layer override through its
        # spec-merging/allocation path. Since every candidate dtype here is
        # exactly 1 byte/element (same as uint8), `.view(torch.uint8)` is a
        # free, lossless bitcast of the identical underlying bytes -- no
        # repacking needed, same trick used for MXFP4 weights elsewhere in
        # this codebase (mxfp4.py's `loaded_weight.view(torch.uint8)`).
        if cache.dtype != torch.uint8:
            return cache.view(torch.uint8)
        return cache

    def _get_active_sw_cache(self) -> torch.Tensor | None:
        # NOTE: this used to additionally require `cache.dtype == torch.uint8`
        # before accepting a candidate cache -- a leftover from an earlier
        # stage of this project when the SWA cache really was instantiated as
        # raw uint8 (see docs/old_attempts.md Step 3). Checkpoints now
        # commonly use a real kv_cache_dtype (e.g. fp8_e4m3), so the real,
        # correctly-bound cache tensor never matched that dtype check and
        # this always fell through to `None` -- silently degrading every
        # SWA-only/dense layer's attention to a zero-page empty cache instead
        # of raising or using the real cache. Just check presence (dtype
        # mismatches are handled by _as_kernel_cache_view instead).
        #
        # Before initialize_kv_cache() actually runs (e.g. during
        # profile_run's memory-sizing pass), vLLM binds `.kv_cache` to a
        # degenerate numel==0 placeholder rather than leaving it unset. That
        # placeholder is `is not None` but carries no real data; treating it
        # as a real cache sends a genuinely empty tensor into the Pallas
        # kernel purely through operations XLA can constant-fold away (an
        # empty array's contents never need reading), which silently drops
        # it from the compiled kernel's operand list and desyncs it from the
        # statically-declared signature (Mosaic "number of refinements must
        # match number of op operands"). Skip it like a real None so
        # forward_mqa's own zero-page dummy-cache fallback is used instead.
        if hasattr(self.mla_attn, "swa_cache_layer") and getattr(
                self.mla_attn.swa_cache_layer, "kv_cache", None) is not None:
            cache = getattr(self.mla_attn.swa_cache_layer, "kv_cache")
            if cache.numel() > 0:
                return self._as_kernel_cache_view(cache)
        if hasattr(self, "kv_cache") and getattr(self, "kv_cache",
                                                 None) is not None:
            cache = getattr(self, "kv_cache")
            if cache.numel() > 0:
                return self._as_kernel_cache_view(cache)
        if hasattr(self.mla_attn, "kv_cache") and getattr(
                self.mla_attn, "kv_cache", None) is not None:
            cache = getattr(self.mla_attn, "kv_cache")
            if cache.numel() > 0:
                return self._as_kernel_cache_view(cache)
        return None

    def _get_active_main_cache(self) -> torch.Tensor | None:
        # See _get_active_sw_cache's note: dropped the stale
        # `dtype == torch.uint8` requirement for the same reason. The
        # `shape[-1] > 512` check is kept -- it distinguishes the main
        # (uncompressed head_dim, e.g. 640) cache from the SWA cache
        # (head_dim <= 512), which is a real structural distinction, not a
        # dtype artifact.
        if hasattr(self, "kv_cache") and getattr(self, "kv_cache",
                                                 None) is not None:
            cache = getattr(self, "kv_cache")
            if cache.numel() > 0 and cache.shape[-1] > 512:
                return self._as_kernel_cache_view(cache)
        if hasattr(self.mla_attn, "kv_cache") and getattr(
                self.mla_attn, "kv_cache", None) is not None:
            cache = getattr(self.mla_attn, "kv_cache")
            if cache.numel() > 0 and cache.shape[-1] > 512:
                return self._as_kernel_cache_view(cache)
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
        from vllm.distributed.parallel_state import \
            get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        attn_ctx = get_forward_context().attn_metadata
        main_prefix = getattr(self.mla_attn, "prefix",
                              getattr(self, "custom_prefix", ""))
        if rank == 0 and not hasattr(
                self, "_printed_ctx_keys") and not _is_tracing():
            self._printed_ctx_keys = True
            logger.debug(
                f"[DEBUG_ATTN_CTX_TYPE] prefix={main_prefix} type={type(attn_ctx)}"
            )

        def _get_field(meta, field_name):
            if meta is None:
                return None
            if isinstance(meta, dict):
                return meta.get(field_name)
            return getattr(meta, field_name, None)

        if isinstance(attn_ctx, dict):
            swa_layer_name = getattr(
                getattr(self.mla_attn, "swa_cache_layer", None), "prefix",
                main_prefix)
            swa_attn_metadata = attn_ctx.get(
                swa_layer_name,
                attn_ctx.get(main_prefix, next(iter(attn_ctx.values()), None)))
            compressor_state_prefix = getattr(
                getattr(getattr(self.mla_attn, "compressor", None),
                        "state_cache", None), "prefix", None)
            main_layer_name = main_prefix
            # `main_layer_name` (this layer's own attention prefix) must be
            # tried *first* -- it addresses the shared main/compressed-KV
            # buffer that `main_page_indices` below reads from. Trying
            # `compressor_state_prefix` first (as introduced by 3cd0d62f) was
            # only ever harmless because, at the time, the compressor's state
            # cache and the shared main/compressed-KV cache aliased the same
            # physical buffer. A later fix (see docs/dsv4_running_and_testing.md
            # session 3) gave the compressor's state cache its own genuinely
            # separate, disjoint physical buffer -- since then, this ordering
            # has silently pointed CSA/HCA layers' attention reads at the
            # *state* cache's block table instead of the main cache's own,
            # causing reads from unrelated (uninitialized/garbage) physical
            # pages once a query's valid range first includes a compressed
            # slot that requires this metadata.
            main_attn_metadata = attn_ctx.get(main_layer_name) or attn_ctx.get(
                compressor_state_prefix) or swa_attn_metadata
            if (os.getenv("DSV4_DEBUG_MAIN_META") == "1"
                    and "layers.2." in str(main_layer_name)
                    and not _is_tracing()):
                _main_bt = attn_ctx.get(main_layer_name)
                _state_bt = attn_ctx.get(compressor_state_prefix)
                same_obj = _main_bt is _state_bt
                same_bt = (_main_bt is not None and _state_bt is not None
                           and _main_bt.block_tables is not None
                           and _state_bt.block_tables is not None and bool(
                               torch.equal(_main_bt.block_tables,
                                           _state_bt.block_tables)))
                logger.debug(
                    f"[MAIN_META_DEBUG] positions={positions[:6].tolist()} "
                    f"main_layer_name={main_layer_name!r} "
                    f"compressor_state_prefix={compressor_state_prefix!r} "
                    f"same_metadata_object={same_obj} same_block_tables={same_bt} "
                    f"main_bt_present={_main_bt is not None} state_bt_present={_state_bt is not None} "
                    f"chosen_is_main={main_attn_metadata is _main_bt} "
                    f"main_bt={_main_bt.block_tables[:8].tolist() if _main_bt is not None and _main_bt.block_tables is not None else None} "
                    f"state_bt={_state_bt.block_tables[:8].tolist() if _state_bt is not None and _state_bt.block_tables is not None else None}"
                )
        else:
            swa_attn_metadata = attn_ctx
            main_attn_metadata = attn_ctx

        orig_sw_cache = self._get_active_sw_cache()
        sw_cache = orig_sw_cache
        if sw_cache is None:
            sw_cache = torch.zeros((0, 16, 1, self.mla_attn.head_dim),
                                   dtype=torch.uint8,
                                   device=q.device)

        # DSV4_FORCE_SWA_ONLY=1 disables the sparse-attention path
        # (CSA top-k + HCA long-range) for every layer, leaving dense
        # local SWA. One-run ablation: if output becomes prompt-faithful
        # the accuracy gap lives in the sparse path; if not, that whole
        # subsystem is exonerated. See docs/dsv4/00_NEXT_PHASE_PROMPT.md.
        swa_only = (self.compress_ratio <= 1
                    or os.getenv("DSV4_FORCE_SWA_ONLY") == "1")
        is_csa = self.compress_ratio == 4

        if swa_attn_metadata is not None:
            swa_seq_lens = _get_field(swa_attn_metadata, "seq_lens")
            swa_block_tables = _get_field(swa_attn_metadata, "block_tables")
            swa_query_start_loc = _get_field(swa_attn_metadata,
                                             "query_start_loc")
            swa_request_distribution = _get_field(swa_attn_metadata,
                                                  "request_distribution")
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
            # Divide unconditionally by compress_ratio for both CSA (4) and
            # HCA (128) layers -- matches the reference implementation
            # (tpu-inference's deepseek_v4_attention.py) exactly. A prior
            # commit (64855fed) narrowed this to CSA-only ("ratio = ... if
            # is_csa else 1"), based on a rationale about SWA-only layers
            # that doesn't apply here: this code already lives inside
            # `if not swa_only`, so SWA-only layers never reach it. The
            # change silently left HCA's main_kv_lens undivided (the raw,
            # uncompressed sequence length instead of seq_len //
            # compress_ratio), inflating the compressed-cache read-loop
            # bound (`num_bkv = cdiv(kv_len, bkv_sz)` in mla.py) far beyond
            # the number of physically valid compressed rows.
            main_kv_lens = m_seq_lens // self.compress_ratio if m_seq_lens is not None else None
            main_page_indices = _get_field(main_attn_metadata, "block_tables")
            main_cu_q_lens = _get_field(main_attn_metadata, "query_start_loc")
            main_distribution = _get_field(main_attn_metadata,
                                           "request_distribution")
        else:
            # A plain alias (`main_cache_kv = sw_cache`, same buffer) makes
            # the Pallas custom kernel's two separate `sw_cache`/
            # `main_cache_kv` formal parameters resolve to one identical
            # input value; XLA's input deduplication then merges them into a
            # single physical operand in the compiled kernel, desyncing it
            # from the statically-declared 14-operand
            # `input_partition_specs` (Mosaic "number of refinements must
            # match number of op operands 14 vs 13"). `.clone()` gives it a
            # distinct buffer so both formal parameters stay real, separate
            # operands; unused for real math in the swa_only path anyway
            # (see _deepseek_v4_attention_jax's swa_only branch).
            main_cache_kv = sw_cache.clone()
            # Reuse the SWA-side metadata tensors (same underlying batch)
            # instead of Python `None`. `_deepseek_v4_attention_jax`'s
            # swa_only branch never reads these for real math, but a `None`
            # is an empty JAX pytree leaf -- it silently drops out of the
            # jax_op's operand list instead of becoming a real argument,
            # which desyncs the traced/compiled operand count from the
            # statically-declared `input_partition_specs` length (surfaces
            # as a Mosaic "number of refinements must match number of op
            # operands" crash once this branch is actually exercised, e.g.
            # once _get_active_sw_cache/_get_active_main_cache correctly
            # find a real cache instead of always returning None). Cloned
            # (not aliased) for the same reason main_cache_kv is cloned
            # above: a plain alias is the same buffer as its swa_* twin,
            # which XLA's input deduplication merges into one operand,
            # desyncing the compiled kernel from the declared 14-operand
            # signature.
            main_kv_lens = swa_seq_lens.clone(
            ) if swa_seq_lens is not None else None
            main_page_indices = swa_block_tables.clone(
            ) if swa_block_tables is not None else None
            main_cu_q_lens = swa_query_start_loc.clone(
            ) if swa_query_start_loc is not None else None
            main_distribution = swa_request_distribution.clone(
            ) if swa_request_distribution is not None else None

        if is_csa:
            assert topk_indices is not None
            fill_pad = torch.full((positions.shape[0], topk_indices.shape[1]),
                                  -1,
                                  dtype=topk_indices.dtype,
                                  device=topk_indices.device)
            topk_indices = torch.cat([topk_indices, fill_pad],
                                     dim=0)[:positions.shape[0]]
            # Compressed slot i covers positions [4i, 4i+4); a query at pos p may
            # attend slot i only when i < (p+1)//4 (matches the reference
            # topk_len = (pos+1)//COMPRESS_RATIO — no clamp, so early tokens
            # attend via SWA only instead of reading unwritten slot 0).
            max_valid_kv = ((positions + 1).to(torch.int32) //
                            self.compress_ratio).unsqueeze(1)
            extra = torch.where(
                (topk_indices >= 0) & (topk_indices < max_valid_kv),
                topk_indices, torch.full_like(topk_indices, -1))
        else:
            extra = (positions + 1).to(torch.int32) // self.compress_ratio

        if os.getenv("DSV4_DEBUG_MAIN_META") == "1" and not _is_tracing():
            from vllm.distributed.parallel_state import \
                get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            if rank == 0:
                logger.debug(
                    f"[DEBUG_FORWARD_MQA_CACHES] prefix={main_prefix} sw_cache.shape={sw_cache.shape if sw_cache is not None else None} sw_cache.dtype={sw_cache.dtype if sw_cache is not None else None} main_cache_kv.shape={main_cache_kv.shape if main_cache_kv is not None else None} main_cache_kv.dtype={main_cache_kv.dtype if main_cache_kv is not None else None}"
                )
            if rank == 0 and main_prefix in ("model.layers.0.attn",
                                             "model.layers.2.attn"):
                logger.debug(
                    f"[DEBUG_DECODE_STEP] prefix={main_prefix} positions={positions[:6].tolist()} "
                    f"swa_seq_lens={swa_seq_lens[:3].tolist() if swa_seq_lens is not None else None} "
                    f"main_kv_lens={main_kv_lens[:3].tolist() if main_kv_lens is not None else None} "
                    f"swa_query_start_loc={swa_query_start_loc[:3].tolist() if swa_query_start_loc is not None else None} "
                    f"main_cu_q_lens={main_cu_q_lens[:3].tolist() if main_cu_q_lens is not None else None} "
                    f"extra[:6]={(extra[:6].tolist() if extra.dim() == 1 else extra[:6].tolist())}"
                )

        # Read-side counterpart to DSV4_DEBUG_COMPRESSOR's dequant check:
        # dequantize the compressed-KV rows this layer's attention is about
        # to actually read (slots 0..main_kv_lens-1 of the sequence's first
        # compressed page) and report their magnitude. mla.py's epilogue
        # clamps L = max(l + exp(sinks), 1.0) >= 1 before out = acc / L, so
        # |out| <= |acc| always -- an output of ~1e27 therefore *cannot*
        # originate in the attention-sinks epilogue and must already be
        # present in acc, whose only two sources are the SWA accumulator
        # handoff and pv = p @ kv (p <= 1). This probe distinguishes the
        # "kv rows are garbage" branch from the "SWA handoff is garbage"
        # branch. Host-side only: no kernel-body change, so it does not by
        # itself require a compile-cache clear.
        if (os.getenv("DSV4_DEBUG_MAIN_KV") == "1" and not _is_tracing()
                and rank == 0 and main_prefix == "model.layers.2.attn"
                and not swa_only and main_cache_kv is not None
                and main_cache_kv.numel() > 0 and main_page_indices is not None
                and main_kv_lens is not None):
            import jax.numpy as jnp
            import numpy as np

            from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
                unpack_sparse_kv_cache
            _mc = main_cache_kv
            if _mc.ndim == 5:
                _mc = _mc.squeeze(2)
            # [num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
            _k_block_size = _mc.shape[1] * _mc.shape[2]
            _width = _mc.shape[-1]
            _flat = _mc.reshape(-1, _width)
            _bt = main_page_indices.flatten()
            _kv_len = int(main_kv_lens[0].item())
            _page0 = int(_bt[0].item())
            _rows = []
            # Include one slot well past kv_len as a never-written control:
            # if its raw bytes match slot 3's, slot 3 was never written and the
            # "garbage" is just the buffer's initial contents.
            _probe_slots = list(range(min(_kv_len, 8))) + [40]
            for _slot in _probe_slots:
                _row_idx = _page0 * _k_block_size + _slot
                _row = _flat[_row_idx:_row_idx + 1].detach().cpu().numpy()
                _nope, _rope, _scale = unpack_sparse_kv_cache(
                    jnp.asarray(_row), 448, 64, 64)
                _n = np.asarray(_nope.astype(jnp.float32))
                _r = np.asarray(_rope.astype(jnp.float32))
                _s = np.asarray(_scale).astype(np.float64)
                # Raw bytes: the decoded views have repeatedly been ambiguous
                # (rope decoding to exactly 2**97 == bf16 0x7000 == bytes
                # "00 70", where 0x70 read as fp8_e4m3 is a perfectly ordinary
                # 128.0). Reading the actual byte regions distinguishes "bad
                # values" from "right values at the wrong offset".
                _b = _row.reshape(-1).astype(np.uint8)
                _nz_nope = int(np.count_nonzero(_b[:448]))
                _nz_rope = int(np.count_nonzero(_b[448:576]))
                _nz_scale = int(np.count_nonzero(_b[576:583]))
                _nz_pad = int(np.count_nonzero(_b[583:640]))
                _rows.append(f"slot{_slot}(row{_row_idx}): "
                             f"nope_absmax={np.nanmax(np.abs(_n)):.4g} "
                             f"rope_absmax={np.nanmax(np.abs(_r)):.4g} "
                             f"scale_max={_s.max():.4g} "
                             f"nz[nope/rope/scale/pad]={_nz_nope}/{_nz_rope}/"
                             f"{_nz_scale}/{_nz_pad} "
                             f"b[444:452]={_b[444:452].tolist()} "
                             f"b[574:584]={_b[574:584].tolist()}")
            logger.debug(
                f"[MAIN_KV_DEQUANT_DEBUG] positions={positions[:4].tolist()} "
                f"main_kv_lens[0]={_kv_len} page0={_page0} "
                f"k_block_size={_k_block_size} | " + " | ".join(_rows))
            # `two_caches_same_buffer` is hardcoded to `not swa_only` here,
            # whereas the reference derives it (main_cache_index ==
            # swa_cache_index). When it is True the compressed-KV read is
            # redirected to `updated_sw_cache`; if the two caches are in fact
            # distinct buffers that redirect points CSA/HCA attention at the
            # wrong physical allocation entirely.
            logger.debug(
                f"[CACHE_IDENTITY_DEBUG] positions={positions[:4].tolist()} "
                f"same_tensor_obj={sw_cache is main_cache_kv} "
                f"same_data_ptr={sw_cache.data_ptr() == main_cache_kv.data_ptr()} "
                f"sw_ptr={sw_cache.data_ptr()} main_ptr={main_cache_kv.data_ptr()} "
                f"sw_shape={list(sw_cache.shape)} main_shape={list(main_cache_kv.shape)} "
                f"redirect={not swa_only}")

        _dump_step_now = getattr(self, "_dsv4_dump_step_current", 0)
        if topk_indices is not None:
            self._maybe_dump_sublayer("02_topk_indices", topk_indices,
                                      _dump_step_now)
        if main_kv_lens is not None:
            self._maybe_dump_sublayer("02_main_kv_lens", main_kv_lens,
                                      _dump_step_now)
        if main_page_indices is not None:
            self._maybe_dump_sublayer("02_main_page_indices",
                                      main_page_indices, _dump_step_now)
        self._maybe_dump_sublayer("02_positions", positions, _dump_step_now)
        if main_cache_kv is not None and os.getenv(
                "DSV4_DUMP_KV_CACHE") == "1":
            # Opt-in, separate from DSV4_DUMP_HIDDEN_STATES: the full paged
            # cache is large (unlike q/kv/topk_indices), so only dump it
            # when explicitly requested (targeted kernel-bisection runs),
            # not on every layer-sweep step.
            self._maybe_dump_sublayer("02_main_cache_kv", main_cache_kv,
                                      _dump_step_now)

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

        self._maybe_dump_sublayer("02b_post_kernel_output", output,
                                  getattr(self, "_dsv4_dump_step_current", 0))

        # Discriminator: is new_sw_cache (the torch tensor returned by
        # self.attn_op, before the write-back copy_ a few lines below) already
        # wrong for compressed-KV rows outside the SWA's own address range, or
        # is it correct at this point (meaning the copy_ call itself, not the
        # JAX kernel's return value, introduces the corruption)? Plain torch
        # read, no host callback needed -- see
        # docs/dsv4/CURRENT_STATUS_20260801.md sec 6 item 0.
        if (os.getenv("DSV4_DEBUG_MAIN_KV") == "1" and not _is_tracing()
                and rank == 0 and main_prefix == "model.layers.2.attn"
                and not swa_only and main_page_indices is not None
                and main_kv_lens is not None and new_sw_cache is not None):
            _nsc = new_sw_cache
            if _nsc.ndim == 5:
                _nsc = _nsc.squeeze(2)
            _page = int(main_page_indices.flatten()[0].item())
            _n = int(main_kv_lens[0].item())
            _rows = []
            for _slot in range(min(_n, 4)):
                _row = _nsc[_page, 0,
                            _slot, :].detach().cpu().numpy().astype("uint8")
                _rows.append(f"slot{_slot}: b[444:452]="
                             f"{_row[444:452].tolist()}")
            logger.debug(
                f"[NEW_SW_CACHE_PRECOPY_DEBUG] positions={positions[:4].tolist()} "
                f"page={_page} kv_lens={_n} | " + " | ".join(_rows))

        if not _is_tracing():
            has_nan = torch.isnan(output).any().item()
            if has_nan:
                rank = os.getenv("RANK", "0")
                clean_out = output[~torch.isnan(output)]

                if clean_out.numel() > 0:
                    logger.debug(
                        f"[PYTORCH_DEBUG_NAN] Rank={rank} output non-nan min={clean_out.min().item():.6f} | max={clean_out.max().item():.6f} | mean={clean_out.mean().item():.6f}"
                    )
                else:
                    logger.debug(
                        f"[PYTORCH_DEBUG_NAN] Rank={rank} output is 100% NaN!")
                dump_path = f"/home/patemotter_google_com/workspace/vllm-torchtpu/scratch/nan_dump_{main_prefix}_rank{rank}.pt"
                logger.debug(
                    f"[NAN_DETECTION] NaN detected in output! Dumping all inputs to {dump_path} ..."
                )
                dump_data = {
                    "q":
                    q.cpu(),
                    "kv":
                    kv.cpu(),
                    "sw_cache":
                    sw_cache.cpu(),
                    "swa_seq_lens":
                    swa_seq_lens.cpu() if swa_seq_lens is not None else None,
                    "swa_block_tables":
                    swa_block_tables.cpu()
                    if swa_block_tables is not None else None,
                    "swa_query_start_loc":
                    swa_query_start_loc.cpu()
                    if swa_query_start_loc is not None else None,
                    "swa_request_distribution":
                    swa_request_distribution.cpu()
                    if swa_request_distribution is not None else None,
                    "main_cache_kv":
                    main_cache_kv.cpu() if main_cache_kv is not None else None,
                    "main_kv_lens":
                    main_kv_lens.cpu() if main_kv_lens is not None else None,
                    "extra":
                    extra.cpu() if isinstance(extra, torch.Tensor) else extra,
                    "main_page_indices":
                    main_page_indices.cpu()
                    if main_page_indices is not None else None,
                    "main_cu_q_lens":
                    main_cu_q_lens.cpu()
                    if main_cu_q_lens is not None else None,
                    "main_distribution":
                    main_distribution.cpu()
                    if main_distribution is not None else None,
                    "attn_sink":
                    self.attn_sink.cpu()
                    if self.attn_sink is not None else None,
                    "output":
                    output.cpu(),
                    "positions":
                    positions.cpu(),
                    "compress_ratio":
                    self.compress_ratio,
                }
                torch.save(dump_data, dump_path)

        if orig_sw_cache is not None and new_sw_cache is not None and new_sw_cache.numel(
        ) > 0:
            orig_sw_cache.copy_(new_sw_cache)

            # Discriminating probe for the "what corrupts slots 0-2 between
            # the prefill capture and the first decode capture" question
            # (docs/dsv4/CURRENT_STATUS_20260801.md §6 item 0): re-run the
            # exact same MAIN_KV_DEQUANT_DEBUG probe immediately after this
            # copy_, reading orig_sw_cache post-write-back. If bytes here
            # differ from the pre-attn_op probe above, this copy_ is the
            # corrupting write; if they're already wrong before attn_op was
            # even called, this write-back is exonerated and the bug is
            # upstream. Reuses _page0/_k_block_size/_kv_len/_probe_slots
            # computed above -- only runs when that block ran.
            if (os.getenv("DSV4_DEBUG_MAIN_KV") == "1" and not _is_tracing()
                    and rank == 0 and main_prefix == "model.layers.2.attn"
                    and not swa_only and main_cache_kv is not None
                    and main_cache_kv.numel() > 0
                    and main_page_indices is not None
                    and main_kv_lens is not None):
                import jax.numpy as jnp
                import numpy as np

                from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
                    unpack_sparse_kv_cache
                _post = orig_sw_cache
                if _post.ndim == 5:
                    _post = _post.squeeze(2)
                _post_flat = _post.reshape(-1, _width)
                _post_rows = []
                for _slot in _probe_slots:
                    _row_idx = _page0 * _k_block_size + _slot
                    _row = _post_flat[_row_idx:_row_idx +
                                      1].detach().cpu().numpy()
                    _nope, _rope, _scale = unpack_sparse_kv_cache(
                        jnp.asarray(_row), 448, 64, 64)
                    _n = np.asarray(_nope.astype(jnp.float32))
                    _r = np.asarray(_rope.astype(jnp.float32))
                    _s = np.asarray(_scale).astype(np.float64)
                    _b = _row.reshape(-1).astype(np.uint8)
                    _post_rows.append(
                        f"slot{_slot}(row{_row_idx}): "
                        f"nope_absmax={np.nanmax(np.abs(_n)):.4g} "
                        f"rope_absmax={np.nanmax(np.abs(_r)):.4g} "
                        f"scale_max={_s.max():.4g} "
                        f"b[444:452]={_b[444:452].tolist()} "
                        f"b[574:584]={_b[574:584].tolist()}")
                logger.debug(
                    f"[MAIN_KV_POST_COPY_DEBUG] positions={positions[:4].tolist()} "
                    f"main_kv_lens[0]={_kv_len} page0={_page0} "
                    f"k_block_size={_k_block_size} | " +
                    " | ".join(_post_rows))
        return output

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        check_nan("o (input to o_proj)", o, self.custom_prefix)
        t = o.shape[0]
        o_f = o.to(torch.float32).view(t, self.mla_attn.n_local_heads,
                                       self.mla_attn.head_dim)
        # Inverse GPT-J RoPE on the trailing rotary_dim channels of each head
        # (mirrors rocm_inv_rope_einsum's _fused_inverse_rope_gptj).
        o_ref, _ = self.mla_attn.rotary_emb(positions, o_f, inverse=True)
        o_ref = o_ref.to(torch.bfloat16)

        # Dequantize wo_a.weight to match w.shape dynamically
        w = self.wo_a.weight
        s = self.wo_a.weight_scale
        if s.ndim == 2:
            block_n = w.shape[0] // s.shape[0]
            block_k = w.shape[1] // s.shape[1]
            s_flat = s.repeat_interleave(block_n,
                                         dim=0).repeat_interleave(block_k,
                                                                  dim=1)
        elif s.ndim == 3:
            s_2d = s.squeeze(1) if s.shape[1] == 1 else s.squeeze()
            if s_2d.shape[0] != w.shape[0] // 128 and s_2d.shape[
                    1] == w.shape[0] // 128:
                s_2d = s_2d.transpose(0, 1)
            block_n = w.shape[0] // s_2d.shape[0]
            block_k = w.shape[1] // s_2d.shape[1]
            s_flat = s_2d.repeat_interleave(block_n,
                                            dim=0).repeat_interleave(block_k,
                                                                     dim=1)
        else:
            s_flat = s.view(w.shape[0], 1)

        wo_a_dequant = w.to(o_ref.dtype) * s_flat.to(o_ref.dtype)
        check_nan("wo_a_dequant", wo_a_dequant, self.custom_prefix)

        # Grouped einsum projection matching reference ROCm implementation
        n_local_groups = self.n_local_groups
        o_lora_rank = self.o_lora_rank

        o_ref_grouped = o_ref.view(t, n_local_groups, -1)
        wo_a_grouped = wo_a_dequant.view(n_local_groups, o_lora_rank, -1)

        z = torch.einsum("tgd,grd->tgr", o_ref_grouped, wo_a_grouped)
        z_flat = z.flatten(1)
        check_nan("z_flat", z_flat, self.custom_prefix)

        # --- wo_b: RowParallelLinear back to hidden_size (returns (out, bias)).
        out = self.wo_b(z_flat)
        if isinstance(out, tuple):
            out = out[0]
        check_nan("out_o_proj", out, self.custom_prefix)
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        check_nan("hidden_states (forward input)", hidden_states,
                  self.custom_prefix)

        # Check weights on the very first forward step
        if not hasattr(self,
                       "_printed_wo_a") and not torch.compiler.is_compiling():
            self._printed_wo_a = True
            if hasattr(self, "wo_a") and hasattr(self.wo_a, "weight_scale"):
                s_val = self.wo_a.weight_scale.float()
                logger.debug(
                    f"[DEBUG_WO_A] {self.custom_prefix} weight shape={self.wo_a.weight.shape} scale shape={s_val.shape} scale min={s_val.min().item():.6f} max={s_val.max().item():.6f} mean={s_val.mean().item():.6f}"
                )
            else:
                logger.debug(
                    f"[DEBUG_WO_A] {self.custom_prefix} no weight_scale on wo_a"
                )

        if hasattr(self, "fused_wqa_wkv") and hasattr(self.fused_wqa_wkv,
                                                      "weight"):
            check_nan("self.fused_wqa_wkv.weight", self.fused_wqa_wkv.weight,
                      self.custom_prefix)
        check_nan("self.q_norm.weight", self.q_norm.weight, self.custom_prefix)
        check_nan("self.kv_norm.weight", self.kv_norm.weight,
                  self.custom_prefix)

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm(hidden_states))

        check_nan("qr_kv (raw gemm output)", qr_kv, self.custom_prefix)
        if kv_score is not None:
            check_nan("kv_score (raw)", kv_score, self.custom_prefix)
        if indexer_weights is not None:
            check_nan("indexer_weights (raw)", indexer_weights,
                      self.custom_prefix)

        qr, kv = qr_kv.split(
            [self.mla_attn.q_lora_rank, self.mla_attn.head_dim], dim=-1)
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
        check_nan("attn_output (attention_impl)", attn_output,
                  self.custom_prefix)

        out = self._o_proj(attn_output, positions)
        check_nan("forward_out", out, self.custom_prefix)
        return out


def patch_deepseek_v4_mla_cls() -> None:
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    ds_v4_amd_model.DeepseekV4Attention = VllmDeepseekV4MLAAttention
    if hasattr(ds_v4_amd_model, "DeepseekV4ROCMAiterMLAAttention"):
        ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention = VllmDeepseekV4MLAAttention
    dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
    logger.info(
        "Patched DeepseekV4Attention, DeepseekV4ROCMAiterMLAAttention and DeepseekV4SWACache for TPU."
    )
