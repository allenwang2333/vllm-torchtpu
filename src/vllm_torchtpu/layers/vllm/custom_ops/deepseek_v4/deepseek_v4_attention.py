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
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend, DeepseekV4SWACache)
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


def _DEBUG_LAYER_PREFIX():
    """Which layer the content/metadata probes read. Default layer 2 (CSA).

    Every probe in this file was hardcoded to `model.layers.2.attn`, so every
    "the cache is correct / addressing is correct / selection is correct"
    result in this investigation was measured on a CSA layer. Layer 3 is the
    first HCA layer (compress_ratio 128) and had never been read. Set
    DSV4_DEBUG_LAYER=3 to point the same probes at it.
    """
    return f"model.layers.{os.getenv('DSV4_DEBUG_LAYER', '2')}.attn"


def _DEBUG_LAYER_INFIX():
    return f"layers.{os.getenv('DSV4_DEBUG_LAYER', '2')}."


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
    # Trace-time active-marker (runs once per trace, i.e. per kernel_key):
    # records the exact static geometry baked into THIS compiled kernel.
    # The 20260812 SWA-cache addressing hunt found the device executing
    # logical_page_size=4 while every host-side probe read 128 -- this line
    # is the ground truth for what the kernel was actually traced with.
    logger.info(
        "[ATTN_TRACE] swa_only=%s is_csa=%s logical_page_size=%s "
        "sliding_window=%s two_caches_same_buffer=%s q=%s sw_cache=%s "
        "main_cache=%s", swa_only, is_csa,
        logical_page_size, sliding_window, two_caches_same_buffer,
        getattr(q, "shape", None), getattr(sw_cache, "shape", None),
        getattr(main_cache_kv, "shape", None))
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


class VllmDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """TPU attention backend for DeepSeek-V4 SWA cache."""

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        return 1


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
        # NOTE: `cache_config.block_size` is NOT final here. The platform's
        # `update_block_size_for_backend` override has not reached this config
        # object yet at model-construction time -- measured 16 (vLLM's default)
        # while the real value is 1024. `_swa_block_size` recomputes from the
        # final config; this is only a provisional value so the attribute
        # always exists.
        self.block_size = self._swa_block_size(cache_config.block_size,
                                               window_size)

    @staticmethod
    def _swa_block_size(compressed_kv_cache_bz: int,
                        window_size: int | None) -> int:
        """SWA page size, sized to fit inside a CSA main cache page.

        A CSA layer's main cache stores `block_size // 4` compressed rows per
        page (`compress_ratio == 4`), so the SWA cache uses the same divisor,
        then clamps to the window it actually has to hold. Identical to the
        reference (`tpu_inference/.../deepseek_v4_attention.py:73`), which
        runs at the same `block_size=1024` we do
        (`tpu_inference/layers/vllm/backends/flash_attn_mla.py:46`).

        The clamp is load-bearing at 1024: `min(256, 128)` = 128 against a
        256-row host page. The overlay does not require equality, only that
        the host page is large enough -- the reference's compatibility test is
        `physical_page_entries >= swa_spec.storage_block_size`
        (`tpu_inference/runner/kv_cache_manager.py:1171`). So a 128-row SWA
        page inside a 256-row CSA page is the intended arrangement, and
        dropping the clamp would only waste half of it.
        """
        csa_compression_ratio = 4
        block_size = compressed_kv_cache_bz // csa_compression_ratio
        if window_size is None:
            return block_size
        return min(block_size, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Recompute from the final, post-override block size and write it back:
        # `_build_attn_op` reads this same attribute as the kernel's
        # `logical_page_size`, so the spec and the kernel must not disagree.
        self.block_size = self._swa_block_size(
            vllm_config.cache_config.block_size, self.window_size)
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return VllmDeepseekSparseSWABackend


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
        # NOTE: `attn_op` is deliberately NOT built here -- see the `attn_op`
        # property below. Model construction is too early to read the geometry
        # the kernel needs to be specialized on.
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

    @property
    def attn_op(self):
        """The Pallas attention op, built on first use rather than in `__init__`.

        Building at construction time reads state that is not final yet:

        - `cache_config.block_size` is still vLLM's default 16 during model
          construction (the platform's `update_block_size_for_backend` override
          is not visible on any config object reachable from here), so the SWA
          logical page size came out 4 instead of 256.
        - the KV caches are still `torch.tensor([])` placeholders, so nothing
          about real tensor identity or geometry can be observed.

        By first forward both are settled, and the model-wrapper context is
        active (`tpu_runner.py:2713`), so `_build_attn_op` can read what it
        needs. The op is memoized per instance, and `_attn_op_cache` still
        dedupes the underlying jax op across layers.
        """
        op = self.__dict__.get("_attn_op_instance")
        if op is None:
            op, built_from_real_caches = self._build_attn_op()
            # vLLM's memory-profiling pass (determine_available_memory ->
            # profile_run) is ALSO a forward, and it runs before
            # initialize_kv_cache: the caches are still numel==0
            # placeholders then, so geometry derived from real tensor
            # identity (two_caches_same_buffer) is wrong-by-construction.
            # An op built in that phase must NOT be memoized, or the first
            # real forward -- and every one after it -- inherits the
            # placeholder-built kernel. Only pin the op once it was built
            # against real caches.
            if built_from_real_caches:
                object.__setattr__(self, "_attn_op_instance", op)
        return op

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
        swa_cache = getattr(self.mla_attn, "swa_cache_layer", None)
        if swa_cache is None:
            # Never guess: a wrong logical page size silently misaddresses
            # every SWA cache slot. A missing swa_cache_layer usually means the
            # layer was constructed by tpu_inference instead of vllm_torchtpu —
            # fix plugin isolation instead.
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer unavailable at "
                "attention-op build time; refusing to guess the SWA cache "
                "logical page size.")
        # Safe to read directly, matching the reference
        # (`tpu_inference/.../deepseek_v4_attention.py:397` passes
        # `logical_page_size=self.swa_cache_layer.block_size`): this op is
        # built lazily on first forward (see the `attn_op` property), by which
        # point `get_kv_cache_spec` has run and replaced the provisional
        # construction-time value with one derived from the final block size.
        logical_page_size = swa_cache.block_size
        if logical_page_size <= 0:
            raise RuntimeError(
                f"{self.custom_prefix}: SWA logical page size is "
                f"{logical_page_size}; expected `get_kv_cache_spec` to have "
                "set it before the first forward.")

        # Derived from real tensor identity, matching the reference's
        # `two_caches_same_buffer = (main_cache_index == swa_cache_index)`
        # (`tpu_inference/.../deepseek_v4_attention.py:344`). When the runner
        # overlays this layer's SWA cache onto its compressed-KV array, the
        # compressed-KV read must see the buffer *after* the SWA kernel's write
        # in this same pass (`updated_sw_cache`), not the stale pre-write
        # snapshot. When they are distinct arrays it must not be redirected.
        #
        # Only correct because `attn_op` is built lazily on first forward: at
        # construction time both are `torch.tensor([])` placeholders and would
        # compare equal-ish for all the wrong reasons.
        main_cache = getattr(self.mla_attn, "kv_cache", None)
        swa_tensor = getattr(swa_cache, "kv_cache", None)
        two_caches_same_buffer = bool(
            not swa_only and main_cache is not None and swa_tensor is not None
            and main_cache.numel() > 0 and swa_tensor.numel() > 0
            and main_cache.data_ptr() == swa_tensor.data_ptr())

        # Are the caches real yet, or still the pre-initialize_kv_cache
        # numel==0 placeholders (profile_run phase)? Placeholder-built ops
        # are throwaways: see the `attn_op` property.
        built_from_real_caches = bool(swa_tensor is not None
                                      and swa_tensor.numel() > 0)

        # Cross-check the layer attribute against the same formula
        # get_kv_cache_spec uses on the final config. These must agree; a
        # mismatch means the op is about to bake a page size the runner's
        # block tables were not built for (the 20260812 defect: kernel
        # addressing with the provisional block_size=4 while the runner
        # assigned one 128-token logical page per request).
        cfg_lps = VllmDeepseekV4SWACache._swa_block_size(
            vllm_context.vllm_config.cache_config.block_size, self.window_size)
        if cfg_lps != logical_page_size:
            logger.warning(
                "[ATTN_BUILD] %s: swa_cache.block_size=%s disagrees with "
                "config-derived SWA page size %s (cache_config.block_size=%s,"
                " window=%s). Using the config-derived value.",
                self.custom_prefix, logical_page_size, cfg_lps,
                vllm_context.vllm_config.cache_config.block_size,
                self.window_size)
            logical_page_size = cfg_lps
        logger.info(
            "[ATTN_BUILD] %s: logical_page_size=%s window=%s swa_only=%s "
            "is_csa=%s two_caches_same_buffer=%s real_caches=%s "
            "sw_numel=%s main_numel=%s", self.custom_prefix, logical_page_size,
            self.window_size, swa_only, is_csa, two_caches_same_buffer,
            built_from_real_caches,
            swa_tensor.numel() if swa_tensor is not None else None,
            main_cache.numel() if main_cache is not None else None)

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
        # `two_caches_same_buffer` is baked into the traced op, and it is NOT a
        # function of op_type: the SWA overlay maps only as many sliding-window
        # caches as there are compatible compressed-KV arrays, so two CSA layers
        # can legitimately disagree (one overlaid, one with its own array).
        # Keying the cache on op_type alone would hand the second layer the
        # first one's op.
        # `_prof` marks a placeholder-built op (load_model's eager
        # kernel pre-build and profile_run both run BEFORE
        # initialize_kv_cache): a separate name keeps it out of the real
        # ops' registries so real forwards rebuild. `_p{page}` bakes the
        # geometry into the cache key: the 20260812 defect was exactly a
        # load_model-era op (provisional SWA page size 4) being returned
        # from this name-keyed cache after initialize_kv_cache's
        # force-rebuild recomputed the same name.
        op_name = (f"pallas::deepseek_v4_attention_{op_type}_v2"
                   f"_p{logical_page_size}"
                   f"{'_aliased' if two_caches_same_buffer else ''}"
                   f"{'' if built_from_real_caches else '_prof'}")

        global _attn_op_cache
        if op_name in _attn_op_cache:
            return _attn_op_cache[op_name], built_from_real_caches

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

        # `_prof` ops are stored too (all 43 layers share one registration
        # during profile_run instead of re-registering the same torch op
        # name), but under their own key -- the real-cache build always
        # misses this entry and re-derives geometry from real tensors.
        _attn_op_cache[op_name] = attn_jax_op
        return attn_jax_op, built_from_real_caches

    def attn_gemm(self, hidden_states):
        # DSV4_KVSCORE_FIRST=1 -- intervention 3 for the deterministic decode
        # corruption. Interventions 1 (host sync) and 2 (async scheduling
        # off) were both byte-inert, so the corruption is NOT a timing race:
        # the mutation is ordered between the two consumers' value capture,
        # and the only thing executing between these adjacent lines is
        # fused_wqa_wkv's own dispatch. Suspicion: the quantized-linear
        # custom op donates/reuses its input buffer, clobbering
        # hidden_states for every LATER reader (the bare matmuls, the synced
        # dumps -- all taken after it and all polluted) while the op itself
        # got the true value (byte-perfect SWA rows). Computing the bypass
        # projections BEFORE fused_wqa_wkv captures the clean value; if
        # generation becomes prompt-faithful this is both the proof and the
        # minimal fix.
        # DSV4_DUMP_HIDDEN_FULL=1: dump the FULL hidden tensor, synced, at
        # the TOP of attn_gemm (before any op dispatch), plus positions --
        # the measurement that splits the post-intervention-3 fork: is the
        # decode hidden row for token t equal to the prefill hidden row for
        # the same token (pollution enters later), or different (the
        # "forward path differs" finding was right and the SWA byte-identity
        # is the anomaly)?
        _hf_layers = [
            x for x in os.getenv("DSV4_DUMP_HIDDEN_LAYERS", "").split(",") if x
        ]
        _hf_this = None
        if _hf_layers and not _is_tracing():
            for _l in _hf_layers:
                if f"layers.{_l}." in getattr(self, "custom_prefix", ""):
                    _hf_this = _l
                    break
        if ((os.getenv("DSV4_DUMP_HIDDEN_FULL") == "1" and not _is_tracing()
             and _DEBUG_LAYER_INFIX() in getattr(self, "custom_prefix", ""))
                or _hf_this is not None):
            from vllm.distributed.parallel_state import \
                get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                import numpy as np

                from vllm_torchtpu.models.vllm.deepseek_v4_patch import \
                    _dump_to_numpy
                _step = getattr(self, "_dsv4_hf_step", 0)
                self._dsv4_hf_step = _step + 1
                _root = os.path.join(
                    os.getenv("DSV4_DUMP_ROOT",
                              os.path.expanduser("~/workspace/scratch")),
                    "page_dumps", os.getenv("DSV4_DUMP_LABEL", "run"))
                os.makedirs(_root, exist_ok=True)
                np.savez_compressed(os.path.join(
                    _root, f"hidden_L{_hf_this or '?'}_step{_step:03d}.npz"),
                                    hidden=_dump_to_numpy(hidden_states))

        _kvscore_first = os.getenv("DSV4_KVSCORE_FIRST") == "1"
        _pre = {}
        if _kvscore_first:
            _c = getattr(self.mla_attn, "compressor", None)
            if _c is not None:
                _pre["kv_score"] = (
                    hidden_states @ _c.fused_wkv_wgate.weight.T)
            if self.indexer is not None:
                _pre["indexer_weights"] = self.indexer.weights_proj(
                    hidden_states)[0]
                _pre["indexer_kv_score"] = (
                    hidden_states
                    @ self.indexer.compressor.fused_wkv_wgate.weight.T)

        qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        compressor = getattr(self.mla_attn, "compressor", None)
        if compressor is not None:
            kv_score = (_pre["kv_score"] if _kvscore_first else
                        hidden_states @ compressor.fused_wkv_wgate.weight.T)
            # DSV4_KVSCORE_SYNC=1 -- intervention experiment for the decode
            # corruption. Root-cause chain (2026-08-12, page dumps v2-v4 +
            # offline weight check): at decode, this kv_score materializes
            # against a MUTATED hidden_states buffer -- offline
            # `dumped_hidden @ W` reproduces the polluted runtime kv_score at
            # cos 0.999999 while fused_wqa_wkv of the same variable is
            # byte-perfect (its consumer, the attention kernel, forces early
            # materialization; kv_score's only consumer is the compressor's
            # cache side effect, which nothing in the step's logits depends
            # on, so the lazy engine defers it past an untracked in-place
            # stream-buffer write). Forcing materialization here pins the
            # value at program order. If this fixes generation, the mechanism
            # is proven by intervention; the proper fix is to stop the
            # in-place aliasing, not to keep this sync.
            if os.getenv("DSV4_KVSCORE_SYNC") == "1" and not _is_tracing():
                try:
                    from torch_tpu._internal import sync as _sync
                    _sync.synchronize(kv_score, wait=True)
                except Exception:
                    pass
        else:
            kv_score = None

        # DSV4_DUMP_PAGES matmul-integrity probe (debug layer, rank 0): the
        # v3 dumps showed kv_score row 0 at decode carries an admixture of
        # other batch rows' content while fused_wqa_wkv of the SAME tensor is
        # byte-perfect. A per-row matmul cannot mix rows, so either this
        # batched `@` lowers wrong at the decode shape, or hidden_states row 0
        # is already polluted and the other path is insensitive. Dump row 0
        # computed BOTH ways (batched vs single-row) plus the hidden row
        # itself, all through synced reads.
        if (os.getenv("DSV4_DUMP_PAGES") == "1" and not _is_tracing()
                and kv_score is not None and _DEBUG_LAYER_INFIX() in getattr(
                    self, "custom_prefix", "")):
            from vllm.distributed.parallel_state import \
                get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                import numpy as np

                from vllm_torchtpu.models.vllm.deepseek_v4_patch import \
                    _dump_to_numpy
                _kvs_single = (
                    hidden_states[0:1] @ compressor.fused_wkv_wgate.weight.T)
                _step = getattr(self, "_dsv4_gemm_dump_step", 0)
                self._dsv4_gemm_dump_step = _step + 1
                _root = os.path.join(
                    os.getenv("DSV4_DUMP_ROOT",
                              os.path.expanduser("~/workspace/scratch")),
                    "page_dumps", os.getenv("DSV4_DUMP_LABEL", "run"))
                os.makedirs(_root, exist_ok=True)
                np.savez_compressed(
                    os.path.join(_root, f"gemm_step{_step:03d}.npz"),
                    hidden_row0=_dump_to_numpy(hidden_states[0:1]),
                    hidden_lastrow=_dump_to_numpy(hidden_states[-1:]),
                    kvs_batched_row0=_dump_to_numpy(kv_score[0:1]),
                    kvs_single_row0=_dump_to_numpy(_kvs_single),
                    num_rows=np.asarray(hidden_states.shape[0]))

        if self.indexer is not None:
            if _kvscore_first:
                indexer_weights = _pre["indexer_weights"]
                indexer_kv_score = _pre["indexer_kv_score"]
            else:
                indexer = self.indexer
                indexer_weights, _ = indexer.weights_proj(hidden_states)
                indexer_kv_score = (
                    hidden_states
                    @ indexer.compressor.fused_wkv_wgate.weight.T)
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

        from vllm_torchtpu.models.vllm.deepseek_v4_patch import _dump_root
        outdir = _dump_root("hidden_states", "vllm_torchtpu")
        os.makedirs(outdir, exist_ok=True)
        if isinstance(tensor, torch.Tensor):
            # Materialize the lazy TPU tensor before the host read; a bare
            # `.cpu()` is not ordered against the deferred graph and copies
            # unmaterialized bytes. See `_dump_to_numpy` in
            # models/vllm/deepseek_v4_patch.py for the full account (it
            # manufactured a 2.884e38 "blowup" that did not exist in the real
            # computation).
            from vllm_torchtpu.models.vllm.deepseek_v4_patch import \
                _dump_to_numpy
            arr = _dump_to_numpy(tensor)
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
                    and _DEBUG_LAYER_INFIX() in str(main_layer_name)
                    and not _is_tracing()):
                _main_bt = attn_ctx.get(main_layer_name)
                _state_bt = attn_ctx.get(compressor_state_prefix)
                same_obj = _main_bt is _state_bt
                same_bt = (_main_bt is not None and _state_bt is not None
                           and _main_bt.block_tables is not None
                           and _state_bt.block_tables is not None and bool(
                               torch.equal(_main_bt.block_tables,
                                           _state_bt.block_tables)))

                # Print SHAPES and full contents, not a `[:8]` slice. That
                # slice previously produced `state_bt=[4, 5, 6, 7, 8, 9, 0, 0]`,
                # which was misread as "the block table is 8 wide" and led to
                # two wrong conclusions about where a modulo wrap would fire.
                # The width is what matters and it was never printed.
                def _bt(md, n=96):
                    if md is None or getattr(md, "block_tables", None) is None:
                        return "None"
                    bt = md.block_tables
                    return (f"shape={tuple(bt.shape)} dtype={bt.dtype} "
                            f"vals[:{n}]={bt.flatten()[:n].tolist()}")

                def _same_ip(md, pos):
                    """Is this layer's input_positions the model positions?"""
                    ip = getattr(md, "input_positions",
                                 None) if md is not None else None
                    if ip is None:
                        return "input_positions=None"
                    try:
                        if ip.shape != pos.shape:
                            return f"shape_differs {tuple(ip.shape)} vs {tuple(pos.shape)}"
                        return bool(torch.equal(ip, pos))
                    except Exception as exc:
                        return f"<{exc}>"

                def _fld(md, name, n=48):
                    v = getattr(md, name, None) if md is not None else None
                    if v is None:
                        return "None"
                    try:
                        return (f"shape={tuple(v.shape)} "
                                f"vals[:{n}]={v.flatten()[:n].tolist()}")
                    except Exception:
                        return str(v)

                logger.info(
                    "[MAIN_META_DEBUG] layer=%s state_prefix=%s\n"
                    "  positions: shape=%s min=%s max=%s first8=%s\n"
                    "  same_metadata_object=%s same_block_tables=%s "
                    "chosen_is_main=%s\n"
                    "  main_bt:  %s\n"
                    "  state_bt: %s\n"
                    "  main.seq_lens: %s\n"
                    "  main.query_start_loc: %s\n"
                    "  main.request_distribution: %s\n"
                    "  state.seq_lens: %s\n"
                    # The reference derives each slot mapping from that
                    # layer's OWN metadata (`state_metadata.input_positions`
                    # for the state map, `k_cache_metadata.input_positions`
                    # for the k map, `positions` only for RoPE --
                    # tpu_inference/.../deepseek_v4_compressor.py:240-246).
                    # We pass the model `positions` to all three, and our
                    # AttentionMetadata sets input_positions=None for
                    # attention layers (attention.py:121). These three lines
                    # show whether that distinction would have mattered.
                    "  main.input_positions: %s\n"
                    "  state.input_positions: %s\n"
                    "  positions_vs_state_input_equal=%s\n"
                    "  compress_ratio=%s window_size=%s",
                    main_layer_name,
                    compressor_state_prefix,
                    tuple(positions.shape),
                    int(positions.min()) if positions.numel() else -1,
                    int(positions.max()) if positions.numel() else -1,
                    positions[:8].tolist(),
                    same_obj,
                    same_bt,
                    main_attn_metadata is _main_bt,
                    _bt(_main_bt),
                    _bt(_state_bt),
                    _fld(_main_bt, "seq_lens"),
                    _fld(_main_bt, "query_start_loc"),
                    _fld(_main_bt, "request_distribution"),
                    _fld(_state_bt, "seq_lens"),
                    _fld(_main_bt, "input_positions"),
                    _fld(_state_bt, "input_positions"),
                    _same_ip(_state_bt, positions),
                    getattr(self, "compress_ratio", None),
                    getattr(self, "window_size", None))
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
            # The max_valid_kv filter is ON by default, and is LOAD-BEARING --
            # do not drop it again to match the reference without replacing
            # what it protects against.
            #
            # The reference has no such filter
            # (`tpu_inference/.../deepseek_v4_attention.py:322-323` is simply
            # `extra = jax_view(topk_indices)`), and it looks redundant here
            # too: our score kernel already masks non-causal and out-of-range
            # candidates to -inf (`streamindex_topk.py:350-353`) and maps any
            # -inf winner to -1 (`:805`). On that reasoning it was defaulted
            # OFF in b79202e0 -- a static-comparison argument, never measured.
            #
            # Measured 2026-08-08, A/B on this flag alone (runA vs runB in
            # scratchpad/, same commit, same prompts):
            #   filter OFF: 81-token needle FAILS (" to the code's. The...")
            #   filter ON:  81-token needle PASSES (7391-QUARTZ verbatim)
            # The needle at 81 tokens sits inside the 128-token sliding window,
            # so this breaks even the short-range path.
            #
            # Why the -1 masking is not enough: `streamindex_topk` selects via
            # `jax.lax.approx_max_k`, which is confirmed NOT equivalent to
            # top_k here (see the WARNING in that file) -- with k >= the
            # candidate count it returns indices positionally rather than by
            # score, so `top_vals` no longer correspond to `top_idxs` and the
            # `where(top_vals == -inf, -1, top_idxs)` guard at :805 masks the
            # wrong entries. Invalid rows then reach the MLA kernel, whose CSA
            # branch applies no causal or kv_len mask of its own (`mla.py:185-195`
            # masks purely by membership in topk_indices) and attends them.
            #
            # The right end state is to fix approx_max_k selection and then
            # retire this filter; until then it is the only thing enforcing
            # causality on the CSA path. DSV4_CSA_FILTER=0 disables it for
            # bisection.
            if os.getenv("DSV4_CSA_FILTER", "1") == "1":
                fill_pad = torch.full(
                    (positions.shape[0], topk_indices.shape[1]),
                    -1,
                    dtype=topk_indices.dtype,
                    device=topk_indices.device)
                topk_indices = torch.cat([topk_indices, fill_pad],
                                         dim=0)[:positions.shape[0]]
                max_valid_kv = ((positions + 1).to(torch.int32) //
                                self.compress_ratio).unsqueeze(1)
                extra = torch.where(
                    (topk_indices >= 0) & (topk_indices < max_valid_kv),
                    topk_indices, torch.full_like(topk_indices, -1))
            else:
                extra = topk_indices
        else:
            extra = (positions + 1).to(torch.int32) // self.compress_ratio

        # DSV4_ABLATE_MAIN=1 empties the long-range branch's selection while
        # leaving everything else -- kernel, merge, sinks, SWA path, shapes,
        # dtypes -- bit-for-bit unchanged. CSA attends no compressed rows
        # (all -1); HCA attends zero compressed keys (kv_lens_to_attend=0).
        #
        # This is the ablation `DSV4_FORCE_SWA_ONLY` was mistaken for. That
        # switch reroutes the layer to a different code path entirely and was
        # correctly written off as "a lobotomy, not an ablation"; this one
        # changes only the values of `extra`.
        #
        # Reading it: if output is byte-identical to a normal run, the main
        # branch is already contributing nothing and the fault is in the
        # merge/weighting, not in the compressed data or its addressing. If
        # output changes, the main branch does contribute and the fault is in
        # what it contributes.
        # NOT gated on `_is_tracing()`, deliberately. Every other probe in this
        # file is, because they only log. This one changes a *value*: if it
        # were skipped under tracing it would silently become a no-op, and a
        # no-op ablation yields byte-identical output -- precisely the reading
        # that would be taken as "the main branch contributes nothing". The
        # failure mode would confirm the hypothesis it was built to test.
        # Values: "1"/"all" (both layer types), "csa", "hca".
        #
        # The selective modes break a confound that has sat under this whole
        # investigation unexamined. 128 is BOTH the SWA window size AND the HCA
        # compress_ratio, so `main_kv_lens = seq_len // 128` means the HCA
        # long-range branch holds ZERO rows below 128 tokens and switches on at
        # exactly the length where retrieval starts failing:
        #
        #   81 tokens (PASSES): HCA rows = 0  -- path is a complete no-op
        #   162 tokens (FAILS): HCA rows = 1
        #   256 tokens (FAILS): HCA rows = 2
        #
        # "the needle left the sliding window" and "the HCA path switched on"
        # are therefore perfectly confounded, and NO prompt length can separate
        # them: if total length < 128 the needle distance is < 128 too, so it is
        # inside the window by construction. The documented "sharp cliff at the
        # sliding-window boundary" assumes the first explanation without ever
        # testing the second.
        #
        # Ablating one layer type at a fixed 256 tokens separates them:
        #   DSV4_ABLATE_MAIN=hca restores 81-token HCA behaviour (no rows) while
        #     leaving CSA live. Retrieval returns -> HCA is the bug.
        #   DSV4_ABLATE_MAIN=csa is the complement and brackets the result.
        # NOTE the `not swa_only` guard on the hca arm. SWA-only layers (0, 1,
        # 42) also report is_csa=False, so a bare `not is_csa` would ablate them
        # too -- and those are precisely the layers carrying the 81-token
        # success. That would reintroduce the confound this switch exists to
        # remove, and turn the arm back into the lobotomy DSV4_FORCE_SWA_ONLY
        # already was. Their `extra` is inert anyway (the swa_only branch never
        # reads it for real math), but ablating them would still be sloppy.
        _ablate = os.getenv("DSV4_ABLATE_MAIN", "")
        _ablate_this = (_ablate in ("1", "all")
                        or (_ablate == "csa" and is_csa)
                        or (_ablate == "hca" and not is_csa and not swa_only))
        if _ablate_this:
            extra = (torch.full_like(extra, -1)
                     if is_csa else torch.zeros_like(extra))
            if not _is_tracing():
                # Positive control that it actually fired, per layer per step.
                logger.info(
                    "[ABLATE_MAIN] mode=%s layer=%s is_csa=%s extra_shape=%s "
                    "extra_first8=%s", _ablate, getattr(self, "prefix", "?"),
                    is_csa, tuple(extra.shape),
                    extra.flatten()[:8].tolist())

        if os.getenv("DSV4_DEBUG_MAIN_META") == "1" and not _is_tracing():
            from vllm.distributed.parallel_state import \
                get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            if rank == 0:
                logger.debug(
                    f"[DEBUG_FORWARD_MQA_CACHES] prefix={main_prefix} sw_cache.shape={sw_cache.shape if sw_cache is not None else None} sw_cache.dtype={sw_cache.dtype if sw_cache is not None else None} main_cache_kv.shape={main_cache_kv.shape if main_cache_kv is not None else None} main_cache_kv.dtype={main_cache_kv.dtype if main_cache_kv is not None else None}"
                )
            if rank == 0 and main_prefix in ("model.layers.0.attn",
                                             _DEBUG_LAYER_PREFIX()):
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
                and rank == 0 and main_prefix == _DEBUG_LAYER_PREFIX()
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
            logger.info(
                f"[MAIN_KV_DEQUANT_DEBUG] positions={positions[:4].tolist()} "
                f"main_kv_lens[0]={_kv_len} page0={_page0} "
                f"k_block_size={_k_block_size} | " + " | ".join(_rows))
            # `two_caches_same_buffer` decides whether the compressed-KV read
            # is redirected to `updated_sw_cache` (the post-SWA-write view of
            # the shared buffer). It is derived from real tensor identity in
            # `_build_attn_op` (see the `main_cache.data_ptr() ==
            # swa_tensor.data_ptr()` expression there), matching the
            # reference's `main_cache_index == swa_cache_index`.
            #
            # This field used to print `not swa_only` under the name
            # `redirect=`, together with a comment claiming the flag was
            # hardcoded to that. Both were stale -- the hardcoding was already
            # gone. A `redirect=True` line therefore meant only "this is not an
            # SWA-only layer", not "the buffer was redirected", which reads as
            # a serious bug on a layer whose two caches are distinct. Print the
            # quantity the kernel actually branches on.
            _redirect = (not swa_only
                         and sw_cache.data_ptr() == main_cache_kv.data_ptr())
            logger.info(
                f"[CACHE_IDENTITY_DEBUG] positions={positions[:4].tolist()} "
                f"same_tensor_obj={sw_cache is main_cache_kv} "
                f"same_data_ptr={sw_cache.data_ptr() == main_cache_kv.data_ptr()} "
                f"sw_ptr={sw_cache.data_ptr()} main_ptr={main_cache_kv.data_ptr()} "
                f"sw_shape={list(sw_cache.shape)} main_shape={list(main_cache_kv.shape)} "
                f"not_swa_only={not swa_only} two_caches_same_buffer={_redirect}"
            )

        # SWA-side probe, mirror of MAIN_KV_DEQUANT_DEBUG for the sliding
        # window cache. Reads the rows the SWA branch serves for the EARLY
        # positions (the prompt) plus one unwritten control row, per step.
        # Indexing mirrors the kernel's contract exactly (mla_swa.py:312):
        # a vLLM page id addresses a full PHYSICAL page
        # (`page_indices_ref[i] * physical_page_size_per_kv_packing`), and only
        # the first `logical_page_size` rows of it are used, so position p in
        # logical block b=p//logical_page_size lives at flat row
        # `page_id*physical_rows + (p % logical_page_size)`. SWA rows are the
        # same 640-byte packed record as the main cache.
        if (os.getenv("DSV4_DEBUG_SWA_KV") == "1" and not _is_tracing()
                and rank == 0 and main_prefix == _DEBUG_LAYER_PREFIX()
                and sw_cache is not None and sw_cache.numel() > 0
                and swa_block_tables is not None and swa_seq_lens is not None):
            import jax.numpy as jnp
            import numpy as np

            from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
                unpack_sparse_kv_cache
            _sc = sw_cache
            if _sc.ndim == 5:
                _sc = _sc.squeeze(2)
            _phys_rows = _sc.shape[1] * _sc.shape[2]
            _lp = getattr(getattr(self.mla_attn, "swa_cache_layer", None),
                          "block_size", None) or _phys_rows
            _width = _sc.shape[-1]
            _sflat = _sc.reshape(-1, _width)
            _sbt = swa_block_tables.flatten()
            _slen = int(swa_seq_lens.flatten()[0].item())
            _probe_pos = list(range(min(_slen,
                                        13))) + [min(_lp, _phys_rows) - 8]
            _rows = []
            for _p in _probe_pos:
                _blk = _p // _lp
                _page = int(_sbt[_blk].item())
                _row_idx = _page * _phys_rows + (_p % _lp)
                _row = _sflat[_row_idx:_row_idx + 1].detach().cpu().numpy()
                _nope, _rope, _scale = unpack_sparse_kv_cache(
                    jnp.asarray(_row), 448, 64, 64)
                _n = np.asarray(_nope.astype(jnp.float32))
                _r = np.asarray(_rope.astype(jnp.float32))
                _b = _row.reshape(-1).astype(np.uint8)
                _rows.append(f"pos{_p}(row{_row_idx}): "
                             f"nope_absmax={np.nanmax(np.abs(_n)):.4g} "
                             f"rope_absmax={np.nanmax(np.abs(_r)):.4g} "
                             f"nz[nope/rope]="
                             f"{int(np.count_nonzero(_b[:448]))}/"
                             f"{int(np.count_nonzero(_b[448:576]))} "
                             f"b[444:452]={_b[444:452].tolist()}")
            logger.info(
                f"[SWA_KV_DEQUANT_DEBUG] positions={positions[:2].tolist()} "
                f"swa_seq_len={_slen} logical_page={_lp} phys_rows={_phys_rows} "
                f"swa_bt[:8]={_sbt[:8].tolist()} | " + " | ".join(_rows))

        # Full page-level dump (DSV4_DUMP_PAGES=1): everything the attention
        # of the debug layer consumes, per call, to .npy -- both caches'
        # request-relevant PHYSICAL PAGES (not 8-byte keyholes), all metadata,
        # q/kv/extra, and the kernel-returned SWA buffer after the call is
        # handled below. Request-scale pages are tiny (640B x 1024 rows), so a
        # 150-step run costs ~300MB, and one decode run + one prefill control
        # answers every remaining content question offline in a single pass.
        # Uses the same direct host reads the byte probes use (proven via bug
        # #4), NOT the DSV4_DUMP_HIDDEN_STATES layer-forward replacement.
        _dump_pages = (os.getenv("DSV4_DUMP_PAGES") == "1"
                       and not _is_tracing() and rank == 0
                       and main_prefix == _DEBUG_LAYER_PREFIX())
        if _dump_pages:
            import numpy as np
            _step = getattr(self, "_dsv4_page_dump_step", 0)
            self._dsv4_page_dump_step = _step + 1
            _root = os.path.join(
                os.getenv("DSV4_DUMP_ROOT",
                          os.path.expanduser("~/workspace/scratch")),
                "page_dumps", os.getenv("DSV4_DUMP_LABEL", "run"))
            os.makedirs(_root, exist_ok=True)

            def _np(t):
                # _dump_to_numpy, not .cpu(): plain host reads of lazy
                # activation tensors are unordered against the deferred graph
                # and fabricate values (this exact probe produced a fake
                # "decode q/kv drift" before this fix; cache tensors happened
                # to read fine, activations did not).
                if t is None:
                    return None
                if isinstance(t, torch.Tensor):
                    from vllm_torchtpu.models.vllm.deepseek_v4_patch import \
                        _dump_to_numpy
                    return _dump_to_numpy(t)
                return np.asarray(t)

            _bundle = {
                "positions": _np(positions),
                "q": _np(q),
                "kv": _np(kv),
                "extra": _np(extra),
                "swa_seq_lens": _np(swa_seq_lens),
                "swa_block_tables": _np(swa_block_tables),
                "swa_query_start_loc": _np(swa_query_start_loc),
                "swa_request_distribution": _np(swa_request_distribution),
                "main_kv_lens": _np(main_kv_lens),
                "main_page_indices": _np(main_page_indices),
                "main_cu_q_lens": _np(main_cu_q_lens),
                "main_distribution": _np(main_distribution),
            }

            # Dump the pages the request ACTUALLY uses, indexed via the block
            # tables (a fixed [:N] window missed them -- fresh requests
            # allocate pages anywhere in the pool). Page ids are saved
            # alongside so the analyzer can map positions to dumped rows.
            def _pages_by_table(cache_t, table, extra=()):
                ids = []
                if table is not None:
                    ids += [int(x) for x in table.flatten()[:2].tolist()]
                ids += [int(x) for x in extra]
                ids = [
                    i for i in dict.fromkeys(ids) if 0 <= i < cache_t.shape[0]
                ]
                return cache_t[ids], np.asarray(ids, dtype=np.int32)

            _state_md = None
            if isinstance(attn_ctx, dict):
                _state_md = attn_ctx.get(compressor_state_prefix)
            _state_bt = getattr(_state_md, "block_tables", None) \
                if _state_md is not None else None
            _mc = main_cache_kv
            if _mc is not None and _mc.numel() > 0:
                if _mc.ndim == 5:
                    _mc = _mc.squeeze(2)
                _extra = ([int(x) for x in _state_bt.flatten()[:2].tolist()]
                          if _state_bt is not None else [])
                pages, ids = _pages_by_table(_mc, main_page_indices, _extra)
                _bundle["main_pages"] = _np(pages)
                _bundle["main_page_ids"] = ids
                if _state_bt is not None:
                    _bundle["state_block_tables"] = _np(_state_bt)
            _sc = sw_cache
            if _sc is not None and _sc.numel() > 0:
                if _sc.ndim == 5:
                    _sc = _sc.squeeze(2)
                pages, ids = _pages_by_table(_sc, swa_block_tables)
                _bundle["sw_pages"] = _np(pages)
                _bundle["sw_page_ids"] = ids
            np.savez_compressed(
                os.path.join(_root, f"attn_step{_step:03d}.npz"), **{
                    k: v
                    for k, v in _bundle.items() if v is not None
                })

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

        # DSV4_ZERO_PAD=1: zero the batch's PADDING rows of q/kv before the
        # kernel. History: this mask changed generation when the attention
        # op was (wrongly) running with logical_page_size=4 -- the padding
        # "sensitivity" it mitigated was that geometry bug, not a kernel
        # property (see docs/dsv4_decode_race_postmortem.md, "ROOT CAUSE
        # FOUND OFFLINE"). With the correct page size the mask is measured
        # INERT: the 13-prompt pool A/B (pool_zp0 vs pool_lpsfix, 20260812)
        # is logprob-identical at every step with it on or off, and the
        # lps=128 kernel is pad-clean in isolation
        # (scratchpad/kernel_micro_repro.py). Kept default-ON as cheap
        # hygiene only: our runner does leave full-magnitude stale data in
        # pad rows (unlike the reference, whose pads are zeros), and two
        # elementwise multiplies per layer buy insurance against any future
        # pad-sensitive consumer. DSV4_ZERO_PAD=0 opts out.
        if os.getenv("DSV4_ZERO_PAD", "1") == "1" and swa_query_start_loc is not None \
                and swa_request_distribution is not None:
            _nvalid = swa_query_start_loc[swa_request_distribution[2]]
            _rowmask = (torch.arange(q.shape[0], device=q.device) < _nvalid)
            q = q * _rowmask.view(-1, 1, 1).to(q.dtype)
            kv = kv * _rowmask.view(-1, 1).to(kv.dtype)
            if not _is_tracing() and not getattr(self, "_zp_logged", False):
                self._zp_logged = True
                logger.info("[ZERO_PAD_ACTIVE] %s masking rows >= nvalid",
                            getattr(self, "custom_prefix", "?"))

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

        if _dump_pages and new_sw_cache is not None and new_sw_cache.numel(
        ) > 0:
            import numpy as np
            _nsc = new_sw_cache
            if _nsc.ndim == 5:
                _nsc = _nsc.squeeze(2)
            _step_done = self._dsv4_page_dump_step - 1
            _pages_post, _ids_post = _pages_by_table(_nsc, swa_block_tables)
            np.savez_compressed(os.path.join(
                _root, f"attn_step{_step_done:03d}_post.npz"),
                                sw_pages_post=_np(_pages_post),
                                sw_page_ids_post=_ids_post,
                                attn_output=_np(output))

        # Discriminator: is new_sw_cache (the torch tensor returned by
        # self.attn_op, before the write-back copy_ a few lines below) already
        # wrong for compressed-KV rows outside the SWA's own address range, or
        # is it correct at this point (meaning the copy_ call itself, not the
        # JAX kernel's return value, introduces the corruption)? Plain torch
        # read, no host callback needed -- see
        # docs/dsv4/CURRENT_STATUS_20260801.md sec 6 item 0.
        if (os.getenv("DSV4_DEBUG_MAIN_KV") == "1" and not _is_tracing()
                and rank == 0 and main_prefix == _DEBUG_LAYER_PREFIX()
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

        # `.item()` here is a device->host sync on every attention layer of
        # every step -- 43 per decode step, each one cutting the lazy graph and
        # blocking the host on the device. Gated behind the same
        # `VLLM_CHECK_NAN` flag as `check_nan()` above rather than a second
        # convention. Unset (the default) this whole block is skipped, which is
        # what a perf or production run wants; set it when hunting NaNs.
        if os.getenv("VLLM_CHECK_NAN") == "1" and not _is_tracing():
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
                from vllm_torchtpu.models.vllm.deepseek_v4_patch import \
                    _dump_root
                dump_path = _dump_root(f"nan_dump_{main_prefix}_rank{rank}.pt")
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
                    and rank == 0 and main_prefix == _DEBUG_LAYER_PREFIX()
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
                logger.info(
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
    if hasattr(ds_v4_amd_model, "DeepseekV4SWACache"):
        ds_v4_amd_model.DeepseekV4SWACache = VllmDeepseekV4SWACache
    if not hasattr(DeepseekSparseSWABackend, "get_min_page_size"):
        DeepseekSparseSWABackend.get_min_page_size = classmethod(
            lambda cls, vllm_config: 1)
    logger.info(
        "Patched DeepseekV4Attention, DeepseekV4ROCMAiterMLAAttention and DeepseekV4SWACache for TPU."
    )
