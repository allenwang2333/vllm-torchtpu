# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU ragged-paged-attention, backed by tokamax.

One kernel, one impl. Every attention layer on this backend routes onto
tokamax's `PallasMosaicTpuBatchedRpa` — a process-wide singleton that registers
its own torch custom op and keys its compilations on the argument shapes and
the keyword config by value. There is no in-tree kernel left to fall back to,
so `_tokamax_rpa_closure` refuses at build time every configuration tokamax
cannot express rather than silently attending over the wrong thing.

`PallasAttentionBackend` and `PallasBatchedRPAAttentionBackend` are the same
backend under two names: vLLM's registry keys backends by
`AttentionBackendEnum`, and `TpuPlatform.get_attn_backend_cls` accepts both
`FLASH_ATTN` (the default, and what every unsupported request falls back to)
and `CUSTOM` (`--attention-backend CUSTOM`). The subclass exists only so
`get_name()` reports the enum the user asked for; both return the same impl.

MLA lives next door in `mla_attention.py`, which takes only the three KV-cache
dtype and page-size helpers below.
"""

import dataclasses
import functools
from typing import Any, ClassVar

import torch
from vllm import envs as vllm_envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer, AttentionType)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# TPU requires the head size to be a multiple of 128.
TPU_HEAD_SIZE_ALIGNMENT = 128

# Note: TPU can fp8 as storage dtype but doesn't support converting from uint8
# from to fp32 directly. That's why it has a dtype mapping different from GPU
TPU_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_ds_mla": torch.uint8,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def _is_fp8_kv_cache_dtype(cache_dtype_str: str) -> bool:
    return cache_dtype_str.lower().strip() in frozenset(
        ("fp8", "fp8_e4m3", "fp8_e5m2"))


def _resolve_kv_cache_dtype(cache_dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype
    normalized = cache_dtype.lower().strip()
    if normalized == "auto":
        raise ValueError(
            "cache_dtype='auto' must be resolved to a concrete torch.dtype "
            "before calling get_kv_cache_shape.")
    dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported KV cache dtype string: {cache_dtype}")
    return dtype


def get_dtype_packing(dtype: torch.dtype, packing_bits: int = 32) -> int:
    """Return number of dtype values packed into a packing_bits lane."""
    bits = torch.empty((), dtype=dtype).element_size() * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"The bit width must divide {packing_bits}, but got {bits} for "
            f"dtype={dtype}.")
    return packing_bits // bits


# HND is SEQ_ALONG_LANE: head_dim, not KV heads, fills the 32-bit words, so an
# fp8 page is genuinely half a bf16 one (b/510425663). tokamax only serves
# HEAD_ALONG_SUBLANE; the HND entries are here so a request for one is named in
# the error rather than silently reinterpreted.
KV_LAYOUT_BY_VLLM_LAYOUT: dict[
    VllmKVCacheLayout, batched_rpa_configs.KVLayout] = {
        VllmKVCacheLayout.LBNHC:
        batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        VllmKVCacheLayout.BLNHC:
        batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        VllmKVCacheLayout.LBHNC: batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
        VllmKVCacheLayout.BLHNC: batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
    }

# vLLM still accepts the pre-refactor layout names on
# `VLLM_KV_CACHE_LAYOUT`; `vllm.config.cache` aliases them privately, so the
# two entries are repeated here rather than reached through a `_`-name.
_VLLM_LAYOUT_NAME_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}


def _resolve_kv_layout() -> batched_rpa_configs.KVLayout:
    """Resolve the layout the KV pool is physically laid out in.

    Prefers the layout the current vLLM config negotiated against the backend's
    `supported_kv_cache_layouts()` -- the engine core resolves it once and
    pushes it to every worker before any layer is built, so in a live server
    this is always the branch taken.

    The fallback covers a bare impl constructed outside a
    `set_current_vllm_config()` scope, or before resolution has run: read the
    same `VLLM_KV_CACHE_LAYOUT` override the resolver reads, and otherwise take
    the NHD default the pool is allocated in. (There is no process-wide getter
    to fall back on any more -- `get_kv_cache_layout` went away with the layout
    refactor that introduced `vllm.v1.kv_cache_layout`.)
    """
    vllm_config = get_current_vllm_config_or_none()
    if (vllm_config is not None
            and vllm_config.cache_config.kv_cache_layout is not None):
        resolved = vllm_config.cache_config.get_resolved_kv_cache_layout()
    elif (requested := vllm_envs.VLLM_KV_CACHE_LAYOUT) is not None:
        name = _VLLM_LAYOUT_NAME_ALIASES.get(requested, requested)
        resolved = VllmKVCacheLayout[name]
    else:
        return batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE

    layout = KV_LAYOUT_BY_VLLM_LAYOUT.get(resolved)
    if layout is None:
        raise NotImplementedError(
            f"KV cache layout {resolved.name} has no batched-RPA equivalent; "
            f"supported: {sorted(v.name for v in KV_LAYOUT_BY_VLLM_LAYOUT)}.")
    return layout


def get_tpu_min_page_size(vllm_config: VllmConfig) -> int:
    max_num_page_per_req = (1024 * 1024 // 2 //
                            vllm_config.scheduler_config.max_num_seqs // 4)
    min_page_size = cdiv(vllm_config.model_config.max_model_len,
                         max_num_page_per_req)
    min_page_size = 1 << (min_page_size - 1).bit_length()
    return min_page_size


# =========================================================================================
# VLLM Attention Backend
# =========================================================================================


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class PallasAttentionBackend(AttentionBackend):
    supported_kv_cache_dtypes = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        """The 5D packed HEAD_ALONG_SUBLANE pool tokamax reads and writes.

        `[pages, page_size, kv_heads_x2 // packing, packing, head_dim]`, with
        K and V packed along the head axis. tokamax folds the two minor head
        dims back together on the far side of the trace, where the reshape is
        free.
        """
        padded_head_size = (cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) *
                            TPU_HEAD_SIZE_ALIGNMENT)
        num_kv_heads_x2 = num_kv_heads * 2
        # vLLM's OffloadingConnectorWorker.register_kv_caches probes this
        # method without a concrete dtype to discover the num_blocks logical
        # dimension (it only reads test_shape.index(num_blocks); see
        # vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py).
        # num_blocks is at dim 0 in our layout regardless of dtype, so return
        # a placeholder shape that satisfies the probe without resolving the
        # dtype.
        if (isinstance(cache_dtype_str, str)
                and cache_dtype_str.lower().strip() == "auto"):
            return (num_blocks, block_size, num_kv_heads_x2, 1,
                    padded_head_size)
        kv_dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        kv_packing = get_dtype_packing(kv_dtype)

        num_kv_heads_x2 = cdiv(num_kv_heads_x2, kv_packing) * kv_packing
        return (
            num_blocks,
            block_size,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            padded_head_size,
        )

    @classmethod
    def get_kv_cache_page_size_bytes(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        """Return one Pallas KV-cache page size including layout padding."""
        dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        shape = cls.get_kv_cache_shape(
            1,
            block_size,
            num_kv_heads,
            head_size,
            dtype,
        )
        num_elements = functools.reduce(lambda x, y: x * y, shape, 1)
        return num_elements * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # In recent TPU generations, up to v6e, the SMEM size is 1MB. The
    # block_tables within the PallasMetadata constitute almost the entire SMEM
    # requirement. Its size is max_num_seqs * num_page_per_seq * 4 (Int). Here
    # we simply make sure that the size is smaller than half of SMEM capacity.
    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        return get_tpu_min_page_size(vllm_config)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        # tokamax's autotuning config pins the layout to HEAD_ALONG_SUBLANE.
        return (VllmKVCacheLayout.LBNHC, )

    @staticmethod
    def get_supported_kernel_block_sizes():
        # HEAD_ALONG_SUBLANE puts one page on one 128-lane tile.
        return [256]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # TPU kernels consume packed records, including head/lane padding.
        # Publish their byte geometry so vLLM computes native L/B strides
        # from the same page size as the TPU allocation.
        page_bytes = cls.get_kv_cache_page_size_bytes(spec.block_size,
                                                      spec.num_kv_heads,
                                                      spec.head_size,
                                                      spec.dtype)
        return dataclasses.replace(spec,
                                   num_head_slots=1,
                                   state_content_bytes=page_bytes //
                                   spec.block_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False, ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(model_len, page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4

    # TPU has limited SREGs (scalar registers), if page_size is too small, we
    # can spill SREGs easily which leads to bad performance. The strategy we
    # apply here is trying to split max-model-len to 16 pages which make the
    # spill less likely. Meanwhile we make sure the page size is in [16, 256].
    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # TODO: This is a temporary fix for vmem OOM.
        # For long model length, we use 16 page-size to avoid too much
        # VMEM spill. A more robust solution should be implemented to
        # handle VREG spills.
        if vllm_config.model_config.max_model_len > 8192:
            return 16
        page_size = next_power_of_2(
            vllm_config.model_config.max_model_len) // 16
        if page_size <= 16:
            return 16
        if page_size >= 256:
            return 256
        return page_size


@register_backend(AttentionBackendEnum.CUSTOM)
class PallasBatchedRPAAttentionBackend(PallasAttentionBackend):
    """`PallasAttentionBackend` under `AttentionBackendEnum.CUSTOM`.

    Same pool, same impl, same tokamax kernel. It exists because vLLM's
    registry is keyed by enum and `--attention-backend CUSTOM` has to resolve
    to something; the only difference is the name it reports.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"


class PallasAttentionBackendImpl(AttentionImpl):
    """Ragged paged attention on tokamax's batched RPA operator."""

    # Process-wide dedup of the closure built below. tokamax's operator already
    # shares compilations across layers that agree on shapes and config; this
    # only keeps the wrapper from being rebuilt per layer.
    _kernel_registry: ClassVar[dict] = {}

    # Read by the runner to size speculative verify windows. tokamax pins the
    # decode blocks to bq_sz=1, so this is a constant rather than a knob.
    decode_query_size: int = 1

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.use_causal_mask = kwargs.get("use_causal_mask", True)

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")

        if sinks is not None:
            # Not in tokamax's batched RPA signature at all, so there is
            # nowhere to forward them; refuse at construction rather than let
            # the layer attend without its sink.
            raise NotImplementedError(
                "Attention sinks are not supported: they are not in tokamax's "
                "batched RPA signature.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        self.kv_cache_quantized_dtype = None
        # Only set kv_cache_quantized_dtype for fp8 KV cache
        if _is_fp8_kv_cache_dtype(kv_cache_dtype):
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                kv_cache_dtype.lower().strip()]

        # Cache the resolved config value during construction: compiled
        # forwards have no active vLLM config and must not look it up.
        self.kv_layout = _resolve_kv_layout()
        # Populated by initialize_kernel() before torch.compile traces the
        # model, so forward can reuse the custom op without rebuilding it from
        # inside Dynamo's fullgraph capture.
        self.rpa_kernel = None
        self._kernel_config_cache: dict = {}

    def _build_rpa_kernel(
        self,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        skip_kv_update: bool = False,
    ):
        config_key = (self.sliding_window, q_scale, k_scale, v_scale,
                      skip_kv_update)
        existing = self._kernel_config_cache.get(config_key)
        if existing is not None:
            return existing

        ctx = get_vllm_model_wrapper_context()
        kernel = self._tokamax_rpa_closure(ctx, q_scale, k_scale, v_scale,
                                           skip_kv_update)
        self._kernel_config_cache[config_key] = kernel
        return kernel

    def _tokamax_rpa_closure(
        self,
        ctx,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        skip_kv_update: bool,
    ):
        """Validate this layer against tokamax's batched RPA and configure it.

        Not a second builder: `_build_rpa_kernel` stays the only entry point
        and owns the per-instance cache. This half holds the guards for
        everything tokamax's operator cannot express and the keyword config
        derived from the layer; `_tokamax_rpa_kernel` holds the call.
        """
        unsupported = []
        if ctx.mesh.shape.get("model", 1) > 1:
            # tokamax's torch operator registers itself once, at import, with
            # no mesh and no `input_partition_specs`, so there is nothing to
            # shard over and every rank would attend over every rank's heads.
            # Multi-process TP, where each worker drives one chip and sees
            # `model=1`, is unaffected.
            unsupported.append(
                f"{ctx.mesh.shape['model']} model-axis devices in one process")
        if envs.TPU_KERNEL_ITER_MODE:
            # tokamax donates the KV pool, so the second iteration of a timing
            # loop would read a donated buffer.
            unsupported.append("TPU_KERNEL_ITER_MODE (the pool is donated)")
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            # The bundled kernel took the whole pool plus a dynamic layer
            # index; tokamax's operator takes one layer's pool.
            unsupported.append("VLLM_TPU_BLOCK_MAJOR_KV (no bundled variant)")
        if self.head_size == 64:
            unsupported.append("head_dim 64 (the hd64 kernel and its pool)")
        nhd = batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
        if self.kv_layout is not nhd:
            unsupported.append("VLLM_KV_CACHE_LAYOUT=HND")
        if skip_kv_update:
            # tokamax still lists `skip_kv_update` in `static_argnames` but
            # never reads it, so a read-only layer would clobber its target's
            # slots rather than attend without writing.
            unsupported.append("Cross-layer KV sharing")
        if self.decode_query_size != 1:
            # `_fwd` pins the decode blocks to bq_sz=1.
            unsupported.append(f"decode_query_size={self.decode_query_size} "
                               "(speculative verify windows)")
        if not self.use_causal_mask:
            # `kernel/wrapper.py` raises for a non-causal mask mid-compile.
            unsupported.append("use_causal_mask=False")
        if unsupported:
            raise NotImplementedError(
                "tokamax's batched RPA operator does not support: "
                f"{', '.join(unsupported)}.")

        # Passed by value, because this tuple is tokamax's compilation cache
        # key. `kv_layout` and `decode_query_size` are deliberately left at
        # tokamax's defaults -- HEAD_ALONG_SUBLANE and 1 are exactly the only
        # values the guards above let through.
        config = dict(
            use_causal_mask=self.use_causal_mask,
            sm_scale=self.scale,
            sliding_window=self.sliding_window,
            soft_cap=self.logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        return self._tokamax_rpa_kernel(config, q_scale, k_scale, v_scale,
                                        skip_kv_update)

    def _tokamax_rpa_kernel(
        self,
        config: dict[str, Any],
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        skip_kv_update: bool,
    ):
        """Close over tokamax's batched RPA operator for this layer.

        `PallasMosaicTpuBatchedRpa` is a process-wide singleton: it registers
        itself with `pallas.jax_op` once, on import, and keys its compilations
        on the argument shapes and the keyword config by value, so layers that
        agree on both already share one kernel a level below this one. What is
        left here is the writeback and a registry entry so the layers also
        share this closure.

        The pool crosses into the operator in the packed 5-D shape the runner
        allocated, not folded onto tokamax's 4-D cache first: tokamax folds it
        on the far side, inside the JAX trace, where the reshape is free. A
        fold on this side would be a tensor of its own, and since the pool is
        donated the operator would then reuse *that* buffer -- 40 GiB of extra
        HBM temporaries per worker on a 397B-class model, which is enough to
        fail the backbone compile outright.

        Donation is also why the writeback below is a `copy_` into the tensor
        the runner owns rather than a rebind: the buffer tokamax hands back is
        the same one it was given.
        """
        # Imported lazily so a missing or moved tokamax surfaces here, against
        # a concrete layer, rather than at module import.
        from tokamax.experimental.torch_tpu.ops.batched_rpa import \
            torch_pallas_mosaic_tpu as tokamax_rpa

        registry_key = (self.sliding_window, self.scale, self.logits_soft_cap,
                        q_scale, k_scale, v_scale, skip_kv_update,
                        self.use_causal_mask, self.decode_query_size)
        existing = self._kernel_registry.get(registry_key)
        if existing is not None:
            return existing

        tokamax_batched_rpa = tokamax_rpa.PallasMosaicTpuBatchedRpa

        def tokamax_rpa_kernel_impl(kv_cache, query, key, value, seq_lens,
                                    block_tables, query_start_loc,
                                    request_distribution):
            outputs, new_kv_cache = tokamax_batched_rpa(
                query,
                key,
                value,
                kv_cache,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                **config,
            )
            # Checked before the copy, which would otherwise fail with a shape
            # error that says nothing about where the cache came from. Shapes
            # are static here, so this costs nothing at run time.
            if new_kv_cache.shape != kv_cache.shape:
                raise RuntimeError(
                    "tokamax returned an incompatible KV cache: expected "
                    f"{tuple(kv_cache.shape)}, got "
                    f"{tuple(new_kv_cache.shape)}.")
            kv_cache.copy_(new_kv_cache)
            return outputs

        self._kernel_registry[registry_key] = tokamax_rpa_kernel_impl
        return tokamax_rpa_kernel_impl

    def initialize_kernel(self, layer: AttentionLayer) -> None:
        """Pre-build the RPA kernel before torch.compile traces the model.

        Must be called after model weights are loaded (so scales are known)
        and after the mesh is available via the model wrapper context, but
        before forward().
        """
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale_value = layer._k_scale_float
            v_scale_value = layer._v_scale_float
            if k_scale_value != 0.0 and v_scale_value != 0.0:
                k_scale = k_scale_value
                v_scale = v_scale_value

        ctx = get_vllm_model_wrapper_context()
        vllm_config = ctx.vllm_config
        parallel_config = (None if vllm_config is None else
                           vllm_config.parallel_config)
        if parallel_config is not None:
            # Both context-parallel modes needed a kernel that returns a
            # partial output plus its log-sum-exp so the ranks can be combined.
            # tokamax's operator returns the attention output only.
            for size_attr, name in (("prefill_context_parallel_size",
                                     "Prefill"),
                                    ("decode_context_parallel_size",
                                     "Decode")):
                if getattr(parallel_config, size_attr, 1) > 1:
                    raise NotImplementedError(
                        f"{name} context parallelism needs a kernel that "
                        "returns per-rank log-sum-exp; tokamax's batched RPA "
                        "operator does not.")

        # KV-sharing (cross-layer) layers reuse the target layer's K/V already
        # written into the shared cache, so their kernel must attend without
        # writing; `_tokamax_rpa_closure` refuses that outright.
        skip_kv_update = self.kv_sharing_target_layer_name is not None
        self.rpa_kernel = self._build_rpa_kernel(q_scale,
                                                 k_scale,
                                                 v_scale,
                                                 skip_kv_update=skip_kv_update)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads_x2 // kv_packing,
                 kv_packing, padded_head_size]
            attn_metadata: Metadata for attention.
            output: buffer written in place, shape
                = [num_tokens, num_heads, head_size_v]
        Returns:
            shape = [num_tokens, num_heads, head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for PallasAttentionBackendImpl")

        # For determine_available_memory case.
        if kv_cache.numel() == 0:
            if output is None:
                output = torch.ones_like(query)
            return output

        query_dim = query.dim()
        if query_dim == 3:
            q_len = query.shape[0]
            q_compute_dim = query.shape[1] * query.shape[2]
        else:
            q_len, q_compute_dim = query.shape

        if key.dim() == 3:
            k_len = key.shape[0]
            k_compute_dim = key.shape[1] * key.shape[2]
        else:
            k_len, k_compute_dim = key.shape

        assert q_compute_dim == self.head_size * self.num_heads
        assert k_compute_dim == self.head_size * self.num_kv_heads

        query = query.view(q_len, self.num_heads, self.head_size)
        key = key.view(k_len, self.num_kv_heads, self.head_size)
        value = value.view(k_len, self.num_kv_heads, self.head_size)
        assert key.shape == value.shape

        # A layer whose head_size is narrower than the page it writes into —
        # a DFlash draft layer sharing the target's unified pool, whose page
        # geometry comes from whichever attention spec defined the pool (see
        # `kv_cache_materializer._pool_attention_geometry`). Pad q/k/v out to
        # the page width so the kernel's slot writes line up, and slice the
        # result back below. Zeros contribute nothing to the QK dot product,
        # `self.scale` comes from the real head_size and is passed separately,
        # and the padded V columns are dropped — so the output is unchanged.
        #
        # Only the head dim is reconciled. A layer whose `num_kv_heads`
        # disagrees with the pool's would write the wrong slots and padding
        # cannot fix that, so the two must already agree.
        pool_head_dim = kv_cache.shape[-1]
        if pool_head_dim > self.head_size:
            pad_size = pool_head_dim - self.head_size
            query = torch.nn.functional.pad(query, (0, pad_size))
            key = torch.nn.functional.pad(key, (0, pad_size))
            value = torch.nn.functional.pad(value, (0, pad_size))
        if self.kv_cache_quantized_dtype:
            k_scale_value = layer._k_scale_float
            v_scale_value = layer._v_scale_float
            if k_scale_value == 0.0 or v_scale_value == 0.0:
                raise ValueError(
                    "k_scale_float and v_scale_float must be non-zero")
            key, value = quantize_kv(self.kv_cache_quantized_dtype, key, value,
                                     k_scale_value, v_scale_value)

        rpa_kernel = self._build_rpa_kernel(
            None,
            layer._k_scale_float if self.kv_cache_quantized_dtype else None,
            layer._v_scale_float if self.kv_cache_quantized_dtype else None,
            skip_kv_update=self.kv_sharing_target_layer_name is not None,
        )

        outputs = rpa_kernel(
            kv_cache,
            query,
            key,
            value,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )

        # Drop the padding added above so callers see the layer's own width.
        if outputs.shape[-1] > self.head_size:
            outputs = outputs[..., :self.head_size]
        # TODO (geyuhao) ideally we don't want this
        if not torch.compiler.is_compiling():
            synchronize_tensors(kv_cache, wait=False)

        if query_dim == 2:
            outputs = outputs.reshape(q_len, self.num_heads * self.head_size)

        if output is not None:
            output.copy_(outputs)
        return outputs
