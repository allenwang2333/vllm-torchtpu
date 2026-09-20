# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU ragged-paged-attention backends.

Two backends live here and they do not share a kernel:

* `PallasAttentionBackend` (FLASH_ATTN, the default) runs the in-tree Pallas
  flash kernel. Its four JAX entry points live in `rpa_kernels.py`; this module
  only names two of them as class defaults and wraps them with `pallas.jax_op`.
  It is the only path that supports in-process TP > 1, head_dim 64, HND pools,
  speculative verify windows, non-causal masks, cross-layer KV sharing and
  attention sinks.
* `PallasBatchedRPAAttentionBackend` (CUSTOM) has no kernel entry at all: it
  routes onto tokamax's `PallasMosaicTpuBatchedRpa` singleton, built inline by
  `_tokamax_rpa_closure`, which refuses at build time every configuration in
  the list above.

What is left here is the backend and impl classes plus the KV-cache geometry
helpers. The two other halves that used to be in this file are

* `rpa_kernels.py`   -- the flash kernel entries, which import nothing back.
* `mla_attention.py` -- MLA, which takes only the three KV-cache dtype and
                        page-size helpers below.
"""

import dataclasses
import functools
from typing import Any, ClassVar

import torch
from torch_tpu._internal import pallas
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
from vllm_torchtpu.distributed.dcp import get_dcp_group as _get_dcp_group
from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.layers.adapter.cp_attention import \
    build_dcp_kernels as _build_dcp_kernels
from vllm_torchtpu.layers.adapter.cp_attention import \
    forward_with_dcp as _forward_with_dcp
from vllm_torchtpu.layers.adapter.rpa_kernels import (
    _pallas_rpa_kernel_default, _pallas_rpa_kernel_default_bundled)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# The op prefix that routes a layer onto tokamax's batched RPA operator.
# Matched by value rather than by `isinstance`, because the runner rebinds
# both the entry and the prefix per instance (see `tpu_runner`'s eagle3 draft
# path) and a rebound layer must follow its new prefix, not its class.
_TOKAMAX_RPA_OP_PREFIX = "pallas::rpa_kernel_batched"

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
# fp8 page is genuinely half a bf16 one (b/510425663).
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
        padded_head_size = (cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) *
                            TPU_HEAD_SIZE_ALIGNMENT)
        # Two different RPA kernels have different KV cache layouts:
        # - hd64 (head_dim=64): K/V packed along head_dim
        # - v3 (head_dim!=64): K/V packed along heads
        # The Pallas kernels expect a 5D KV cache: [L, S, Kx2 / kv_packing, kv_packing, H]
        # where Kx2 = num_kv_heads for hd64 and Kx2 = num_kv_heads * 2 for v3.
        use_hd64 = (head_size == 64)
        # vLLM's OffloadingConnectorWorker.register_kv_caches probes this
        # method without a concrete dtype to discover the num_blocks logical
        # dimension (it only reads test_shape.index(num_blocks); see
        # vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py).
        # num_blocks is at dim 0 in our layout regardless of dtype, so return
        # a placeholder shape that satisfies the probe without resolving the
        # dtype.
        if (isinstance(cache_dtype_str, str)
                and cache_dtype_str.lower().strip() == "auto"):
            num_kv_heads_x2 = num_kv_heads if use_hd64 else num_kv_heads * 2
            return (num_blocks, block_size, num_kv_heads_x2, 1,
                    padded_head_size)
        kv_dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        kv_packing = get_dtype_packing(kv_dtype)

        num_kv_heads_x2 = num_kv_heads if use_hd64 else num_kv_heads * 2
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
        """Return one Pallas KV-cache page size including layout padding.

        Sized through `cls` so a subclass reports what it actually allocates.
        """
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
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            return (VllmKVCacheLayout.BLNHC, )
        return (VllmKVCacheLayout.LBNHC, )

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
    """Pallas attention backend wired to tokamax's batched-RPA operator.

    Registered under `AttentionBackendEnum.CUSTOM` by
    `TpuPlatform.pre_register_and_update`; user opts in via
    `--attention-backend CUSTOM`. The Impl subclass is a marker: it carries no
    in-tree kernel of its own, and `PallasAttentionBackendImpl`'s builder
    routes its `_kernel_op_prefix` to tokamax's `PallasMosaicTpuBatchedRpa`.

    The KV pool is the inherited 5D packed HEAD_ALONG_SUBLANE one; the
    SEQ_ALONG_LANE (HND) shape and the PCP-streaming page sizes belonged to the
    in-tree kernels and are gone with them.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["PallasBatchedRPAAttentionBackendImpl"]:
        return PallasBatchedRPAAttentionBackendImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        # tokamax's autotuning config pins the layout to HEAD_ALONG_SUBLANE, so
        # HND is no longer negotiable for this backend.
        return (VllmKVCacheLayout.LBNHC, )

    @staticmethod
    def get_supported_kernel_block_sizes():
        # HEAD_ALONG_SUBLANE puts one page on one 128-lane tile. The other
        # multiples of 128 belonged to the HND and PCP-streaming kernels.
        return [256]


class PallasAttentionBackendImpl(AttentionImpl):
    _kernel_instance_counter = 0
    # Registry of shared custom ops keyed to avoid registering duplicate Pallas
    # kernels for layers with identical configs.
    # Mapping of (sliding_window, mesh, q_scale, k_scale, v_scale) -> custom op
    _kernel_registry: dict = {}

    # Each Impl subclass points at its own RPA kernel entry function. Wrapped
    # with `staticmethod` so attribute access through `self` doesn't bind it
    # as a method. Subclasses override this to select a different kernel.
    _kernel_entry: ClassVar = staticmethod(_pallas_rpa_kernel_default)
    _kernel_op_prefix: ClassVar[str] = "pallas::rpa_kernel"
    decode_query_size: int = 1

    # Bundled (block-major) RPA kernel entry: accepts the full KV cache bundle
    # and a dynamic scalar layer_idx. Registered under a dedicated prefix to avoid
    # collisions with layer-major ops in the shared registry.
    _kernel_entry_bundled: ClassVar = staticmethod(
        _pallas_rpa_kernel_default_bundled)
    _kernel_op_prefix_bundled: ClassVar[str] = "pallas::rpa_kernel_bundled"

    # Class-level registry of bundled kernel instances. Layers sharing identical
    # configurations reuse a single custom op, emitting one shared OpOverload node
    # across the FX graph rather than duplicating JAX tracing and MLIR compilation.
    _bundled_kernel_registry: ClassVar[dict] = {}

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

        # Store sinks for attention sink optimization
        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")
        # Cache the resolved config value during construction: compiled
        # forwards have no active vLLM config and must not look it up.
        self.kv_layout = _resolve_kv_layout()
        self.rpa_kernel = None
        # Populated by initialize_kernel() before torch.compile traces the
        # model, so forward can reuse the custom op without rebuilding it from
        # inside Dynamo's fullgraph capture.
        self._kernel_config_cache: dict = {}
        # DCP (Decode Context Parallelism) state. Populated by initialize_kernel()
        # when decode_context_parallel_size > 1.
        self.dcp_world_size = 1
        self.dcp_rank = 0
        self.rpa_dcp_cache_kernel = None  # CACHE_ONLY pass, returns (out, lse)
        self.rpa_dcp_new_kernel = None  # NEW_TOKENS_ONLY pass, returns (out, lse)
        # Block-major bundled kernel and pre-allocated layer index. Initialized by
        # setup_bundled() after bundle allocation and prior to torch.compile tracing.
        # During profile_run, ctx.kv_cache_bundle is None and forward falls back to
        # standard execution.
        self.rpa_kernel_bundled = None
        self._bundle_layer_idx_tensor: torch.Tensor | None = None

    @classmethod
    def _allocate_kernel_instance_id(cls) -> int:
        kernel_instance_id = cls._kernel_instance_counter
        cls._kernel_instance_counter += 1
        return kernel_instance_id

    def _build_rpa_kernel(
        self,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        skip_kv_update: bool = False,
    ):
        config_key = (self._kernel_op_prefix, self.sliding_window, q_scale,
                      k_scale, v_scale, skip_kv_update, self.decode_query_size)
        existing = self._kernel_config_cache.get(config_key)
        if existing is not None:
            return existing

        ctx = get_vllm_model_wrapper_context()
        if self._kernel_op_prefix == _TOKAMAX_RPA_OP_PREFIX:
            kernel = self._tokamax_rpa_closure(ctx, q_scale, k_scale, v_scale,
                                               skip_kv_update)
            # Cached per instance only. Sharing happens a level down: tokamax's
            # operator is a process singleton that keys its compilations on the
            # keyword arguments by value.
            self._kernel_config_cache[config_key] = kernel
            return kernel

        # Reuse an existing custom op if one with the same config already exists.
        # `_kernel_op_prefix` is included so variants (e.g. the runner's
        # single-device local kernel) don't collide with the base impl in the
        # shared registry. `skip_kv_update` is part of the key so KV-sharing
        # (read-only) layers get their own kernel variant and never share an op
        # with KV-owning layers of an otherwise-identical config.
        mesh = ctx.mesh
        registry_key = (self._kernel_op_prefix,
                        self.sliding_window, self.scale, self.logits_soft_cap,
                        id(mesh), q_scale, k_scale, v_scale, skip_kv_update,
                        self.use_causal_mask, self.decode_query_size)
        existing = self._kernel_registry.get(registry_key)
        if existing is not None:
            self._kernel_config_cache[config_key] = existing
            return existing

        kernel_instance_id = self._allocate_kernel_instance_id()
        op_name = f"{self._kernel_op_prefix}_{kernel_instance_id}"

        wrapped_fn = functools.partial(
            self._kernel_entry,
            mesh=mesh,
            sliding_window=self.sliding_window,
            sm_scale=self.scale,
            soft_cap=self.logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            skip_kv_update=skip_kv_update,
            use_causal_mask=self.use_causal_mask,
        )

        # Register as a custom op to mark it as an op boundary in Dynamo.
        # This prevents torch.compile from tracing into the Pallas kernel internals.
        #
        # Kernel-iteration mode runs the op eagerly between compiled pieces
        # (splitting_ops); donating kv_cache there kills the buffer that
        # rpa_kernel_impl's copy_ writes back into. Inside a compiled full
        # graph XLA rewires the alias, so donation is only safe when the op
        # is compiled.
        rpa_donate_argnums = (None if envs.TPU_KERNEL_ITER_MODE else (0, ))
        rpa_kernel_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=rpa_donate_argnums,
        )

        # We must overwrite the default fake implementation as vLLM uses dynamic
        # dimensions for the query.
        def _fake_rpa_op(kv_cache: torch.Tensor, query: torch.Tensor, *args,
                         **kwargs):
            return torch.empty_like(kv_cache), torch.empty_like(query)

        rpa_kernel_op.register_fake(_fake_rpa_op)

        def rpa_kernel_impl(kv_cache, *args, **kwargs):
            new_kv_cache, output = rpa_kernel_op(kv_cache, *args, **kwargs)
            if new_kv_cache.shape != kv_cache.shape:
                raise RuntimeError(
                    "RPA kernel returned an incompatible KV cache shape: "
                    f"expected {tuple(kv_cache.shape)}, got "
                    f"{tuple(new_kv_cache.shape)}.")
            # Plain donation + copy_ writeback: XLA lifts the copy_ input
            # mutation and aliases the op output onto the donated pool, so this
            # compiles to an in-place pool update.
            kv_cache.copy_(new_kv_cache)
            return output

        self._kernel_registry[registry_key] = rpa_kernel_impl
        self._kernel_config_cache[config_key] = rpa_kernel_impl
        return rpa_kernel_impl

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
        and owns the cache. This half holds the guards for everything
        tokamax's operator cannot express and the keyword config derived from
        the layer; `_tokamax_rpa_kernel` holds the call.
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

        registry_key = ("tokamax", self.sliding_window, self.scale,
                        self.logits_soft_cap, q_scale, k_scale, v_scale,
                        skip_kv_update, self.use_causal_mask,
                        self.decode_query_size)
        existing = self._kernel_registry.get(registry_key)
        if existing is not None:
            return existing

        tokamax_batched_rpa = tokamax_rpa.PallasMosaicTpuBatchedRpa

        def tokamax_rpa_kernel_impl(kv_cache, query, key, value, seq_lens,
                                    block_tables, query_start_loc,
                                    request_distribution, sinks):
            if sinks is not None:
                raise NotImplementedError(
                    "Attention sinks are not in tokamax's batched RPA "
                    "signature.")
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

    def _run_dcp_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return _forward_with_dcp(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            sliding_window=self.sliding_window,
            sm_scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            kv_cache_quantized_dtype=self.kv_cache_quantized_dtype,
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            kv_layout=self.kv_layout,
        )

    def _build_bundled_kernel(
        self,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
    ):
        """Constructs and caches the bundled RPA custom op via `pallas.jax_op`.

        Uses in-place buffer donation on the full bundle (arg 0) with a dynamic
        `layer_idx` scalar. Layers sharing identical kernel configurations reuse
        the cached op via `_bundled_kernel_registry`.
        """
        # The batched impl carries no `_kernel_entry` of its own -- its layer-
        # major kernel is tokamax's, reached by op prefix -- so the identity
        # check below would not notice it and block-major KV would silently run
        # the *default* bundled kernel. Refuse by type first.
        if isinstance(self, PallasBatchedRPAAttentionBackendImpl):
            raise NotImplementedError(
                "VLLM_TPU_BLOCK_MAJOR_KV=1: tokamax's batched RPA operator "
                "has no bundled variant; refusing to fall back to the "
                "default bundled kernel.")
        # Fail closed when the layer-major kernel is overridden (per-instance
        # rebinds like single-device local kernels) without a corresponding
        # bundled counterpart, preventing silent fallback to default kernels.
        # Read via `self` to detect instance overrides.
        if (self._kernel_entry is not PallasAttentionBackendImpl._kernel_entry
                and self._kernel_entry_bundled
                is PallasAttentionBackendImpl._kernel_entry_bundled):
            raise NotImplementedError(
                "VLLM_TPU_BLOCK_MAJOR_KV=1: "
                f"{type(self).__name__} overrides the layer-major RPA "
                "kernel but has no bundled variant; refusing to fall "
                "back to the default bundled kernel.")
        ctx = get_vllm_model_wrapper_context()
        mesh = ctx.mesh
        registry_key = (self._kernel_op_prefix_bundled, self.sliding_window,
                        self.scale, self.logits_soft_cap, id(mesh), q_scale,
                        k_scale, v_scale, self.use_causal_mask)
        existing = self._bundled_kernel_registry.get(registry_key)
        if existing is not None:
            return existing

        kernel_instance_id = self._allocate_kernel_instance_id()
        op_name = f"{self._kernel_op_prefix_bundled}_{kernel_instance_id}"
        wrapped_fn = functools.partial(
            self._kernel_entry_bundled,
            mesh=mesh,
            sliding_window=self.sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=self.scale,
            soft_cap=self.logits_soft_cap,
            use_causal_mask=self.use_causal_mask,
        )

        # Eager kernel iteration mode runs without JAX tracing; omit donation
        # to prevent prematurely releasing the buffer needed for copy_ writeback.
        bundled_donate_argnums = (None if envs.TPU_KERNEL_ITER_MODE else (0, ))
        rpa_kernel_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=bundled_donate_argnums,
        )

        def _fake_rpa_op(kv_cache_bundle: torch.Tensor,
                         layer_idx: torch.Tensor, query: torch.Tensor, *args,
                         **kwargs):
            return (torch.empty_like(kv_cache_bundle), torch.empty_like(query))

        rpa_kernel_op.register_fake(_fake_rpa_op)

        def rpa_kernel_bundled_impl(kv_cache_bundle, *args, **kwargs):
            new_bundle, output = rpa_kernel_op(kv_cache_bundle, *args,
                                               **kwargs)
            if new_bundle.shape != kv_cache_bundle.shape:
                raise RuntimeError(
                    "Bundled RPA kernel returned an incompatible bundle "
                    f"shape: expected {tuple(kv_cache_bundle.shape)}, got "
                    f"{tuple(new_bundle.shape)}.")
            # In-place writeback: XLA aliases the donated input buffer to the op
            # output, compiling the copy_ operation into an in-place HBM update.
            kv_cache_bundle.copy_(new_bundle)
            return output

        self._bundled_kernel_registry[registry_key] = rpa_kernel_bundled_impl
        return rpa_kernel_bundled_impl

    def setup_bundled(self, layer_idx: int, bundle_device: torch.device,
                      layer: "AttentionLayer") -> None:
        """Initializes the bundled kernel and pre-allocates the layer index tensor.

        Called by the TPU runner immediately after bundle allocation and BEFORE
        `torch.compile` traces `capture_model`. Mesh, sliding window, and scaling
        parameters are bound into the compiled op.
        """
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale_value = layer._k_scale_float
            v_scale_value = layer._v_scale_float
            if k_scale_value != 0.0 and v_scale_value != 0.0:
                k_scale = k_scale_value
                v_scale = v_scale_value
        # Pre-allocate layer index tensor on device to avoid per-step Host-to-Device copies.
        self._bundle_layer_idx_tensor = torch.tensor(int(layer_idx),
                                                     dtype=torch.int32,
                                                     device=bundle_device)
        self.rpa_kernel_bundled = self._build_bundled_kernel(
            q_scale, k_scale, v_scale)

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
        # KV-sharing (cross-layer) layers reuse the target layer's K/V already
        # written into the shared cache, so their kernel must attend without
        # writing. See `PallasAttentionBackendImpl.__init__` and the runner's
        # cache aliasing for how the shared cache tensor is set up.
        skip_kv_update = self.kv_sharing_target_layer_name is not None
        ctx = get_vllm_model_wrapper_context()
        vllm_config = ctx.vllm_config
        parallel_config = (None if vllm_config is None else
                           vllm_config.parallel_config)
        pcp_configured = (parallel_config is not None and
                          parallel_config.prefill_context_parallel_size > 1)
        if pcp_configured:
            raise NotImplementedError(
                "Prefill context parallelism needs the streaming RPA kernel, "
                "which was removed along with the in-tree batched RPA "
                "kernels; tokamax has no streaming variant.")

        dcp_configured = (parallel_config is not None and getattr(
            parallel_config, 'decode_context_parallel_size', 1) > 1)
        if dcp_configured:
            # TODO(kwang3939): forward decode_query_size to the DCP ops
            self.decode_query_size = 1
            # Every backend now allocates the NHD pool, so passing HND to the
            # DCP kernels would reinterpret the physical KV pool. This used to
            # be reachable on the CUSTOM backend, which allocated its own HND
            # pool; that shape override is gone with the in-tree kernel.
            if self.kv_layout is batched_rpa_configs.KVLayout.SEQ_ALONG_LANE:
                raise NotImplementedError(
                    "DCP with VLLM_KV_CACHE_LAYOUT=HND is not supported: the "
                    "KV pool is allocated NHD.")
            dcp_group = _get_dcp_group()
            if dcp_group is not None:
                self.dcp_world_size = int(dcp_group.world_size)
                self.dcp_rank = int(dcp_group.rank_in_group)
                self.rpa_dcp_cache_kernel, self.rpa_dcp_new_kernel = (
                    _build_dcp_kernels(sliding_window=self.sliding_window,
                                       sm_scale=self.scale,
                                       logits_soft_cap=self.logits_soft_cap,
                                       q_scale=q_scale,
                                       k_scale=k_scale,
                                       v_scale=v_scale,
                                       cp_group_size=self.dcp_world_size,
                                       cp_rank=self.dcp_rank,
                                       kv_layout=self.kv_layout))
            return

        self.rpa_kernel = self._build_rpa_kernel(q_scale,
                                                 k_scale,
                                                 v_scale,
                                                 skip_kv_update=skip_kv_update)

    def runs_batched_rpa_schedule(self) -> bool:
        """Whether forward runs a kernel that keeps a whole step's (query
        block, KV block) schedule in SMEM, which the runner bounds.

        Always False now: the only remaining batched RPA implementation is
        tokamax's, which spills its schedule to HBM. Kept because the runner
        calls it.
        """
        return False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """Process sinks after model loading - convert to float32 as required by RPA kernel."""
        if self.sinks is not None:
            # RPA v3 kernel requires sinks to be float32
            self.sinks = torch.nn.Parameter(self.sinks.to(torch.float32),
                                            requires_grad=False)

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
                 kv_packing, padded_head_size] (preferred)
                or legacy 4D
                [num_blocks, block_size, num_kv_heads_x2, padded_head_size]
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

        sink = self.sinks
        ctx = get_vllm_model_wrapper_context()

        if self.dcp_world_size > 1:
            if sink is not None or ctx.kv_cache_bundle is not None:
                raise NotImplementedError(
                    "DCP attention does not support attention sinks or "
                    "the bundled KV cache path.")
            outputs = self._run_dcp_forward(layer, query, key, value, kv_cache,
                                            attn_metadata)
            if outputs.shape[-1] > self.head_size:
                outputs = outputs[..., :self.head_size]
            if query_dim == 2:
                outputs = outputs.reshape(q_len,
                                          self.num_heads * self.head_size)
            if output is not None:
                output.copy_(outputs)
            return outputs

        skip_kv_update = self.kv_sharing_target_layer_name is not None
        rpa_kernel = self._build_rpa_kernel(
            None,
            layer._k_scale_float if self.kv_cache_quantized_dtype else None,
            layer._v_scale_float if self.kv_cache_quantized_dtype else None,
            skip_kv_update=skip_kv_update,
        )

        # TODO (geyuhao) the support of this API is pending discussion.
        # This line will only influence performance, not functionality
        # # Mark kv_cache avaliable for donation
        # pallas.set_buffer_donor_(kv_cache, True)

        # Block-major execution path: when a KV cache bundle is mounted on the context,
        # route through the bundled RPA kernel. The per-layer `kv_cache` argument
        # passed by vLLM (a strided view) is bypassed in favor of direct, in-place
        # updates to the full bundle. During profile_run, `ctx.kv_cache_bundle` is None;
        # Dynamo guards on this check trigger retracing once the bundle is initialized.
        if ctx.kv_cache_bundle is not None:
            if skip_kv_update:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: the bundled RPA path does "
                    "not support KV sharing")
            outputs = self.rpa_kernel_bundled(
                ctx.kv_cache_bundle,
                self._bundle_layer_idx_tensor,
                query,
                key,
                value,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                sink,
            )
        else:
            # Call the operator
            outputs = rpa_kernel(
                kv_cache,
                query,
                key,
                value,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                sink,
            )

        # Drop the padding added above so callers see the layer's own width.
        if outputs.shape[-1] > self.head_size:
            outputs = outputs[..., :self.head_size]
        # TODO (geyuhao) ideally we don't want this
        if not torch.compiler.is_compiling():
            synchronize_tensors(ctx.kv_cache_bundle if ctx.kv_cache_bundle
                                is not None else kv_cache,
                                wait=False)

        if query_dim == 2:
            outputs = outputs.reshape(q_len, self.num_heads * self.head_size)

        if output is not None:
            output.copy_(outputs)
        return outputs


class PallasBatchedRPAAttentionBackendImpl(PallasAttentionBackendImpl):
    """Marker impl for the CUSTOM backend; the kernel itself is tokamax's.

    It exists so the backend and `_build_bundled_kernel` have something to
    dispatch on. `_build_rpa_kernel` routes on `_kernel_op_prefix` rather than
    on this type, because the runner rebinds the prefix per instance.
    """
    _kernel_op_prefix = _TOKAMAX_RPA_OP_PREFIX
