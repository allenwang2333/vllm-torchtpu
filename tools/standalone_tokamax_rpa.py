#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for the tokamax batched-RPA path in `attention.py`.

Self-contained by construction: this file imports nothing from
`vllm_torchtpu` and nothing from `vllm`. Every helper that
`src/vllm_torchtpu/layers/adapter/attention.py` reaches for across a package
boundary is vendored into Part 1 below, and the routines under test are
transcribed into Part 2. `torch` is the only hard requirement.

Run it::

    python3 tools/standalone_tokamax_rpa.py           # CPU checks
    python3 tools/standalone_tokamax_rpa.py --verbose # per-check detail

Everything except the last check runs on CPU with no TPU, no JAX and no
tokamax. `check_real_tokamax_operator` needs all three and reports itself as
SKIP without them; point it at a checkout with
`PYTHONPATH=/path/to/tokamax`.

What this actually tests
------------------------
The adapter seam, end to end, against an oracle that never touches the paged
cache:

  * `get_kv_cache_shape` -- the 5D packed pool geometry.
  * `tokamax_kv_cache_view` -- the free 5D->4D fold tokamax performs on its
    own side of the boundary, and its invertibility.
  * `build_tokamax_rpa_kernel` -- the seven fail-closed guards, and the fact
    that `skip_kv_update` is *not* forwarded to tokamax.
  * `attention_forward` -- the 3D/2D query reshape, narrow-head padding, fp8
    quantization, the nine-argument call, the output slice.
  * `reference_rpa_kernel` -- a pure-torch stand-in with tokamax's exact
    signature and semantics, so the whole pipeline can be run and checked for
    numerics without a TPU.

Fidelity notes (read these before trusting a green run)
-------------------------------------------------------
1. K/V interleave. The fused 4D cache axis is K,V per head: index ``2*h`` is
   key head ``h``, ``2*h + 1`` is value head ``h``. Taken from tokamax's
   `_src/ops/experimental/batched_rpa/reference.py`, which unpacks with
   ``flat_kv[:, :, 0::2]`` / ``[:, :, 1::2]``.

2. Scale convention. `reference_rpa_kernel` *multiplies* by `k_scale` and
   `v_scale`, matching tokamax's kernel
   (`kernel/flash_attention.py:67,137` -- ``qk *= scale_k``, ``pv *=
   scale_v``) and matching this repo. Note that tokamax's own
   `reference.py` *divides* instead; the two disagree, and the kernel is the
   one that ships. See MISMATCH note in `reference_rpa_kernel`.

3. `operator` injection. `build_tokamax_rpa_kernel` takes the tokamax
   callable as an argument, defaulting to a lazy import of the real
   singleton -- the same singleton `attention.py` imports. `attention.py`
   has no such seam; it always imports. This is the one deliberate
   divergence, and it is what lets the guards and the nine-argument call be
   checked with no tokamax installed.

4. No tensor parallelism. The singleton registers itself once, at import,
   with no mesh and no partition specs, so `attention.py` refuses a
   `model` axis wider than one device in a process rather than sharding.
   This file mirrors that refusal; there is no `shard_map` stand-in here
   because there is no `shard_map` left to stand in for.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch

# =====================================================================
# Part 1 -- vendored dependencies
#
# These are the cross-package helpers `attention.py` imports. Each is a
# faithful copy; the provenance comment says where the original lives.
# =====================================================================

# `vllm_torchtpu.utils.TPU_HEAD_SIZE_ALIGNMENT`
TPU_HEAD_SIZE_ALIGNMENT = 128

# `vllm.utils.math_utils.cdiv`
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# `vllm.platforms.tpu.TPU_STR_DTYPE_TO_TORCH_DTYPE`, trimmed to the dtypes
# the batched RPA path can actually be handed.
TPU_STR_DTYPE_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
}


# `attention.py::_resolve_kv_cache_dtype`
def resolve_kv_cache_dtype(cache_dtype: str | torch.dtype) -> torch.dtype:
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


# `attention.py::get_dtype_packing`
def get_dtype_packing(dtype: torch.dtype, packing_bits: int = 32) -> int:
    """Number of `dtype` values packed into a `packing_bits` lane."""
    bits = torch.empty((), dtype=dtype).element_size() * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"The bit width must divide {packing_bits}, but got {bits} for "
            f"dtype={dtype}.")
    return packing_bits // bits


class KVLayout(enum.Enum):
    """`vllm_torchtpu.kernels.experimental.batched_rpa.configs.KVLayout`.

    HEAD_ALONG_SUBLANE is NHD: one page is `[page_size, heads, head_dim]`,
    so a page lands on one 128-lane tile. SEQ_ALONG_LANE is HND, where
    head_dim rather than the KV heads fills the 32-bit words. Only the
    former survives on this path.
    """

    HEAD_ALONG_SUBLANE = "head_along_sublane"
    SEQ_ALONG_LANE = "seq_along_lane"


@dataclass
class AttentionMetadata:
    """The five fields the RPA kernels read, from
    `vllm_torchtpu.layers.core.attention_metadata.AttentionMetadata`.

    The real dataclass is a registered JAX pytree with a dozen more fields
    for mamba and sequence layouts; none of them reach this kernel.
    """

    # (max_num_seqs * max_num_blocks_per_req,) -- FLAT, not 2D. tokamax's
    # `RpaConfigs` asserts `page_indices.ndim == 1`.
    block_tables: torch.Tensor
    # (max_num_seqs,) total context length per request after this step.
    seq_lens: torch.Tensor
    # (max_num_seqs + 1,) cumulative scheduled query tokens.
    query_start_loc: torch.Tensor
    # (3,) [decode_end, prefill_end, mixed_end]; the last is the live count.
    request_distribution: torch.Tensor
    # (padded_total_num_scheduled_tokens,) -- carried but unused here, since
    # vLLM applies RoPE before attention.
    input_positions: torch.Tensor | None = None


# `vllm_torchtpu.layers.core.quantization`
def static_per_tensor_quantize_tensor(
    tensor: torch.Tensor,
    quant_dtype: torch.dtype,
    scale: float,
) -> torch.Tensor:
    """Quantize with a fixed scalar scale and clamp to the dtype range."""
    if scale == 0.0:
        raise ValueError("Quantization scale must be non-zero.")
    if quant_dtype.is_floating_point:
        dtype_info = torch.finfo(quant_dtype)
    else:
        dtype_info = torch.iinfo(quant_dtype)
    quantized = torch.clamp(
        tensor.to(torch.float32) / scale, dtype_info.min, dtype_info.max)
    return quantized.to(quant_dtype)


def quantize_kv(
    quant_dtype: torch.dtype,
    key: torch.Tensor,
    value: torch.Tensor | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Static per-tensor quantization of key/value."""
    key = static_per_tensor_quantize_tensor(key, quant_dtype, k_scale)
    if value is None:
        return key, None
    value = static_per_tensor_quantize_tensor(value, quant_dtype, v_scale)
    return key, value


# =====================================================================
# Part 2 -- the routines under test, transcribed from attention.py
# =====================================================================

# `attention.py::_TOKAMAX_RPA_OP_PREFIX`
TOKAMAX_RPA_OP_PREFIX = "pallas::rpa_kernel_batched"

# tokamax's *torch* singleton: the same object `attention.py` calls. It
# registers itself once, at import, through `torch_tpu`'s `pallas.jax_op`, and
# that registration -- its signature, its keyword config, its KV-pool donation
# and the shapes it accepts -- is the contract this file exercises.
_TOKAMAX_OP_PATH = (
    "tokamax.experimental.torch_tpu.ops.batched_rpa."
    "torch_pallas_mosaic_tpu")
_TOKAMAX_OP_ATTR = "PallasMosaicTpuBatchedRpa"


def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str | torch.dtype = "auto",
) -> tuple[int, ...]:
    """`PallasAttentionBackend.get_kv_cache_shape`.

    The 5D packed pool:
    `[num_blocks, block_size, kv_heads_x2 // packing, packing, head_dim]`.
    Since the CUSTOM backend's HND override was deleted, this is the only
    pool geometry on either backend.
    """
    padded_head_size = (cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) *
                        TPU_HEAD_SIZE_ALIGNMENT)
    # Two different RPA kernels have different KV cache layouts:
    # - hd64 (head_dim=64): K/V packed along head_dim
    # - v3 (head_dim!=64): K/V packed along heads
    use_hd64 = (head_size == 64)
    # vLLM's offloading connector probes this without a concrete dtype just
    # to find the num_blocks axis, so "auto" returns a placeholder whose
    # dim 0 is still num_blocks.
    if (isinstance(cache_dtype_str, str)
            and cache_dtype_str.lower().strip() == "auto"):
        num_kv_heads_x2 = num_kv_heads if use_hd64 else num_kv_heads * 2
        return (num_blocks, block_size, num_kv_heads_x2, 1, padded_head_size)
    kv_dtype = resolve_kv_cache_dtype(cache_dtype_str)
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


def tokamax_kv_cache_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """The fold tokamax's `op_impl_call` does, in torch.

    `[pages, page_size, kv_heads_x2 // packing, packing, head_dim]` ->
    `[pages, page_size, kv_heads_x2, head_dim]`. The packing lane is the
    minor of the merged head axis, so this is free and exactly invertible,
    which is why tokamax can take the pool in the shape the runner allocated
    and hand that same shape back.

    Deliberately *not* called by `build_tokamax_rpa_kernel`: on this side of
    the custom-op boundary the result would be a tensor of its own, and the
    pool is donated -- see `check_pool_crosses_the_boundary_unfolded`. It is
    used here by the fixtures and the stand-in operators, which live on
    tokamax's side.
    """
    pages, page_size, packed_heads, packing, head_dim = kv_cache.shape
    return kv_cache.view(pages, page_size, packed_heads * packing, head_dim)


@dataclass
class LayerSpec:
    """The `PallasAttentionBackendImpl` attributes this path reads.

    Standing in for the impl object so the routines can be exercised without
    constructing a vLLM backend (which needs a live `VllmConfig`, a device
    mesh and a registered TPU platform).
    """

    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    sliding_window: int | None = None
    logits_soft_cap: float | None = None
    use_causal_mask: bool = True
    decode_query_size: int = 1
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE
    # None for an unquantized pool; a torch dtype turns on `quantize_kv`.
    kv_cache_quantized_dtype: torch.dtype | None = None
    kv_sharing_target_layer_name: str | None = None
    sinks: torch.Tensor | None = None
    kernel_op_prefix: str = TOKAMAX_RPA_OP_PREFIX

    @classmethod
    def llama3_8b(cls, **overrides: Any) -> "LayerSpec":
        """One TP=1 layer of a Llama-3-8B-shaped decoder."""
        base = dict(num_heads=32, num_kv_heads=8, head_size=128,
                    scale=128**-0.5)
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]


def build_tokamax_rpa_kernel(
    spec: LayerSpec,
    *,
    mesh_shape: dict[str, int] | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    skip_kv_update: bool = False,
    kernel_iter_mode: bool = False,
    operator: Callable[..., Any] | None = None,
) -> Callable[..., torch.Tensor]:
    """`PallasAttentionBackendImpl._tokamax_rpa_closure`.

    Returns the nine-argument closure `attention_forward` calls. Every
    configuration tokamax's operator cannot express is refused here, at
    build time, rather than mid-compile or -- worse -- silently.

    `operator` is the injection seam described in the module docstring; the
    real builder imports the same singleton (see `_TOKAMAX_OP_PATH`).
    `kernel_iter_mode` stands in for `envs.TPU_KERNEL_ITER_MODE`.
    """
    if operator is None:
        # Imported lazily so a missing or moved tokamax surfaces here,
        # against a concrete layer, rather than at module import.
        import importlib
        operator = getattr(importlib.import_module(_TOKAMAX_OP_PATH),
                           _TOKAMAX_OP_ATTR)

    mesh_shape = mesh_shape or {}
    model_axis = mesh_shape.get("model", 1)
    unsupported: list[str] = []
    if model_axis > 1:
        # The singleton registers itself once, at import, with no mesh and no
        # `input_partition_specs`, so there is nothing to shard over and every
        # rank would attend over every rank's heads. Multi-process TP, where
        # each worker drives one chip and sees `model=1`, is unaffected.
        unsupported.append(f"{model_axis} model-axis devices in one process")
    if kernel_iter_mode:
        # tokamax donates the KV pool, so the second iteration of a timing
        # loop would read a donated buffer.
        unsupported.append("TPU_KERNEL_ITER_MODE (the pool is donated)")
    if spec.head_size == 64:
        unsupported.append("head_dim 64 (the hd64 kernel and its pool)")
    if spec.kv_layout is not KVLayout.HEAD_ALONG_SUBLANE:
        unsupported.append("VLLM_KV_CACHE_LAYOUT=HND")
    if skip_kv_update:
        # tokamax still lists `skip_kv_update` in `static_argnames` but never
        # reads it, so a read-only layer would clobber its target's slots
        # rather than attend without writing.
        unsupported.append("Cross-layer KV sharing")
    if spec.decode_query_size != 1:
        # `_fwd` pins the decode blocks to bq_sz=1.
        unsupported.append(f"decode_query_size={spec.decode_query_size} "
                           "(speculative verify windows)")
    if not spec.use_causal_mask:
        # `kernel/wrapper.py` raises for a non-causal mask mid-compile.
        unsupported.append("use_causal_mask=False")
    if unsupported:
        raise NotImplementedError(
            "tokamax's batched RPA operator does not support: "
            f"{', '.join(unsupported)}.")

    # Passed by value, because this tuple is tokamax's compilation cache key.
    # `kv_layout` and `decode_query_size` are deliberately left at tokamax's
    # defaults -- HEAD_ALONG_SUBLANE and 1 are exactly the only values the
    # guards above let through. `skip_kv_update` is deliberately absent: it
    # is dead in tokamax, which is why KV sharing raises above instead.
    config = dict(
        use_causal_mask=spec.use_causal_mask,
        sm_scale=spec.scale,
        sliding_window=spec.sliding_window,
        soft_cap=spec.logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    def tokamax_rpa_kernel_impl(kv_cache, query, key, value, seq_lens,
                                block_tables, query_start_loc,
                                request_distribution, sinks):
        if sinks is not None:
            raise NotImplementedError(
                "Attention sinks are not in tokamax's batched RPA "
                "signature.")
        # The pool goes in in the packed 5D shape it was allocated with, not
        # folded onto tokamax's 4D cache first: tokamax folds it on the far
        # side, inside the JAX trace, where the reshape is free. A fold here
        # would be a tensor of its own, and since the pool is donated the
        # operator would reuse *that* buffer -- one extra pool-sized HBM
        # allocation per call, 40 GiB per worker on a 397B-class model.
        outputs, new_kv_cache = operator(
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
        # error that says nothing about where the cache came from.
        if new_kv_cache.shape != kv_cache.shape:
            raise RuntimeError(
                "tokamax returned an incompatible KV cache: expected "
                f"{tuple(kv_cache.shape)}, got "
                f"{tuple(new_kv_cache.shape)}.")
        # A `copy_` into the runner's tensor rather than a rebind: the pool is
        # donated, so the buffer tokamax hands back is the one it was given.
        kv_cache.copy_(new_kv_cache)
        return outputs

    return tokamax_rpa_kernel_impl


def attention_forward(
    spec: LayerSpec,
    rpa_kernel: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
    *,
    k_scale_float: float = 1.0,
    v_scale_float: float = 1.0,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """`PallasAttentionBackendImpl.forward`, minus DCP and block-major KV.

    Both of those branches are refused on this path anyway: DCP returns
    before the tokamax builder, and `_build_bundled_kernel` raises for the
    batched impl outright.

    query: `[num_tokens, num_heads, head_size]` or the flat 2D form.
    key/value: `[num_tokens, num_kv_heads, head_size]` or flat 2D.
    """
    if output_scale is not None:
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

    assert q_compute_dim == spec.head_size * spec.num_heads
    assert k_compute_dim == spec.head_size * spec.num_kv_heads

    query = query.view(q_len, spec.num_heads, spec.head_size)
    key = key.view(k_len, spec.num_kv_heads, spec.head_size)
    value = value.view(k_len, spec.num_kv_heads, spec.head_size)
    assert key.shape == value.shape

    # A layer whose head_size is narrower than the page it writes into -- a
    # DFlash draft layer sharing the target's unified pool. Pad q/k/v out to
    # the page width so the kernel's slot writes line up, and slice the
    # result back below. Zeros contribute nothing to the QK dot product,
    # `spec.scale` comes from the real head_size and is passed separately,
    # and the padded V columns are dropped -- so the output is unchanged.
    pool_head_dim = kv_cache.shape[-1]
    if pool_head_dim > spec.head_size:
        pad_size = pool_head_dim - spec.head_size
        query = torch.nn.functional.pad(query, (0, pad_size))
        key = torch.nn.functional.pad(key, (0, pad_size))
        value = torch.nn.functional.pad(value, (0, pad_size))
    if spec.kv_cache_quantized_dtype:
        if k_scale_float == 0.0 or v_scale_float == 0.0:
            raise ValueError(
                "k_scale_float and v_scale_float must be non-zero")
        key, value = quantize_kv(spec.kv_cache_quantized_dtype, key, value,
                                 k_scale_float, v_scale_float)

    outputs = rpa_kernel(
        kv_cache,
        query,
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
        spec.sinks,
    )

    # Drop the padding added above so callers see the layer's own width.
    if outputs.shape[-1] > spec.head_size:
        outputs = outputs[..., :spec.head_size]

    if query_dim == 2:
        outputs = outputs.reshape(q_len, spec.num_heads * spec.head_size)

    if output is not None:
        output.copy_(outputs)
    return outputs


# =====================================================================
# Part 3 -- example tensors
# =====================================================================


@dataclass(frozen=True)
class Workload:
    """One continuous-batching step of a GQA decoder.

    This is what the kernel actually sees in production: a handful of
    1-token decodes at the front of the persistent batch, then a chunked
    prefill resuming mid-sequence, then a fresh prefill. Shapes follow
    Llama-3-8B (32 query heads over 8 KV heads, head_dim 128) at the only
    page size the CUSTOM backend still offers.

    The pool is deliberately small (24 pages) so the harness stays fast; the
    *structure* -- 5D packed layout, a page count covering every active
    request, padded batch rows -- is the part that has to be faithful.
    """

    num_heads: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    # The single entry in `get_supported_kernel_block_sizes()`.
    page_size: int = 256
    max_model_len: int = 1024
    # The persistent batch is padded to a fixed width; only `num_reqs` rows
    # are live.
    max_num_seqs: int = 8
    num_reqs: int = 6
    num_decodes: int = 4
    num_blocks: int = 24
    # Scheduled tokens padded up to a compile bucket, as the runner does.
    num_tokens: int = 772
    padded_num_tokens: int = 1024
    dtype: torch.dtype = torch.bfloat16
    seed: int = 20260919

    # 4 decodes (1 token each), then a 512-token prefill chunk on a request
    # with 512 tokens already cached, then a fresh 256-token prefill:
    # 4 + 512 + 256 = 772 scheduled tokens.
    q_lens: tuple[int, ...] = (1, 1, 1, 1, 512, 256)
    # Total context each request will have attended after this step.
    seq_lens: tuple[int, ...] = (1024, 768, 512, 257, 1024, 256)

    @property
    def pages_per_req(self) -> int:
        return self.max_model_len // self.page_size

    @property
    def kv_packing(self) -> int:
        return get_dtype_packing(self.dtype)

    @property
    def kv_cache_shape(self) -> tuple[int, ...]:
        return get_kv_cache_shape(self.num_blocks, self.page_size,
                                  self.num_kv_heads, self.head_size,
                                  self.dtype)

    def empty_kv_cache(self) -> torch.Tensor:
        return torch.zeros(self.kv_cache_shape, dtype=self.dtype)

    def metadata(self) -> AttentionMetadata:
        starts = [0]
        for q_len in self.q_lens:
            starts.append(starts[-1] + q_len)
        assert starts[-1] == self.num_tokens, starts
        starts += [self.num_tokens] * (self.max_num_seqs + 1 - len(starts))

        seq_lens = list(self.seq_lens)
        seq_lens += [0] * (self.max_num_seqs - len(seq_lens))

        # Distinct pages per live request; padded rows park on page 0. Flat,
        # not [max_num_seqs, pages_per_req]: `AttentionMetadata` documents
        # `(max_num_seqs * max_num_blocks_per_req,)` and tokamax's
        # `RpaConfigs` asserts `page_indices.ndim == 1`.
        block_tables = torch.zeros(self.max_num_seqs * self.pages_per_req,
                                   dtype=torch.int32)
        for req in range(self.num_reqs):
            base = req * self.pages_per_req
            block_tables[base:base + self.pages_per_req] = torch.arange(
                base, base + self.pages_per_req, dtype=torch.int32)

        return AttentionMetadata(
            input_positions=torch.arange(self.padded_num_tokens,
                                         dtype=torch.int32),
            block_tables=block_tables,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            # V3 convention: [decode_end, prefill_end, mixed_end].
            request_distribution=torch.tensor(
                [self.num_decodes, self.num_decodes, self.num_reqs],
                dtype=torch.int32),
        )

    def qkv(self, head_size: int | None = None,
            num_heads: int | None = None,
            num_kv_heads: int | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q/k/v for the whole padded step, in vLLM's 3D layout."""
        head_size = head_size or self.head_size
        num_heads = num_heads or self.num_heads
        num_kv_heads = num_kv_heads or self.num_kv_heads
        gen = torch.Generator().manual_seed(self.seed)

        def _rand(heads: int) -> torch.Tensor:
            return (torch.randn(self.padded_num_tokens, heads, head_size,
                                generator=gen) * 0.5).to(self.dtype)

        return (_rand(num_heads), _rand(num_kv_heads), _rand(num_kv_heads))

    def history(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Full per-request K/V history, `[seq_len, kv_heads, head_size]`.

        The oracle's ground truth. `seed_cache` writes the prefix of this
        into the pool and `new_kv` hands the suffix to the kernel, so the
        two together reconstruct exactly these tensors -- if the paging
        arithmetic is right.
        """
        gen = torch.Generator().manual_seed(self.seed + 1)
        keys, values = [], []
        for kv_len in self.seq_lens:
            shape = (kv_len, self.num_kv_heads, self.head_size)
            keys.append((torch.randn(shape, generator=gen) * 0.5).to(
                self.dtype))
            values.append((torch.randn(shape, generator=gen) * 0.5).to(
                self.dtype))
        return keys, values

    def seed_cache(self, kv_cache: torch.Tensor,
                   keys: list[torch.Tensor],
                   values: list[torch.Tensor]) -> None:
        """Write each request's *pre-existing* history into the pool.

        Positions `[0, seq_len - q_len)`; the last `q_len` are the new
        tokens the kernel is responsible for writing.
        """
        view = tokamax_kv_cache_view(kv_cache)
        table = self.metadata().block_tables.view(self.max_num_seqs,
                                                  self.pages_per_req)
        # The history may be narrower than the page (a 96-wide layer in a
        # 128-wide pool); the tail columns stay zero, exactly as the
        # forward pad leaves them for the new tokens.
        width = keys[0].shape[-1]
        for req, (kv_len, q_len) in enumerate(zip(self.seq_lens,
                                                  self.q_lens)):
            for pos in range(kv_len - q_len):
                page = int(table[req, pos // self.page_size])
                off = pos % self.page_size
                view[page, off, 0::2, :width] = keys[req][pos]
                view[page, off, 1::2, :width] = values[req][pos]

    def new_kv(self, keys: list[torch.Tensor], values: list[torch.Tensor]
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's new K/V, packed into the padded token buffer."""
        shape = (self.padded_num_tokens, self.num_kv_heads, self.head_size)
        key = torch.zeros(shape, dtype=self.dtype)
        value = torch.zeros(shape, dtype=self.dtype)
        start = 0
        for req, (kv_len, q_len) in enumerate(zip(self.seq_lens,
                                                  self.q_lens)):
            key[start:start + q_len] = keys[req][kv_len - q_len:kv_len]
            value[start:start + q_len] = values[req][kv_len - q_len:kv_len]
            start += q_len
        return key, value

    def query(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.seed + 2)
        return (torch.randn(self.padded_num_tokens, self.num_heads,
                            self.head_size, generator=gen) * 0.5).to(
                                self.dtype)


WORKLOAD = Workload()


# =====================================================================
# Part 4 -- a pure-torch stand-in for tokamax's operator
# =====================================================================


def reference_rpa_kernel(
    kv_cache: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_distribution: torch.Tensor,
    sinks: torch.Tensor | None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    use_causal_mask: bool = True,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> torch.Tensor:
    """Batched ragged paged attention in pure torch, in place on `kv_cache`.

    Same nine-argument signature as the closure `build_tokamax_rpa_kernel`
    returns, so it can be dropped straight into `attention_forward`.
    Semantics ported from tokamax's
    `_src/ops/experimental/batched_rpa/reference.py`: write every live
    token's K/V into its page first, then attend over `[0, kv_len)`.

    MISMATCH, resolved in favour of the kernel: tokamax's `reference.py`
    dequantizes with ``all_k / k_scale`` and ``all_v / v_scale``, but
    tokamax's kernel multiplies -- ``qk *= scale_k`` at
    `kernel/flash_attention.py:67` and ``pv *= scale_v`` at :137 -- and so
    does this repo (`tests/layers/adapter/test_attention_dcp.py::_reference`
    uses ``keys * _K_SCALE``). This function multiplies, matching the code
    that actually ships.
    """
    if sinks is not None:
        raise NotImplementedError(
            "Attention sinks are not in tokamax's batched RPA signature.")

    view = tokamax_kv_cache_view(kv_cache)
    total_pages, page_size, kv_heads_x2, head_dim = view.shape
    num_kv_heads = kv_heads_x2 // 2
    num_heads = query.shape[1]
    q_per_kv = num_heads // num_kv_heads
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.numel() // max_num_seqs
    max_kv_len = pages_per_seq * page_size
    table = block_tables.view(max_num_seqs, pages_per_seq)
    cu = query_start_loc.tolist()
    lens = seq_lens.tolist()
    active = int(request_distribution[-1])

    # Step 1: write every live token's K/V into its slot, before any
    # attention, so a chunked prefill sees its own earlier tokens.
    for s in range(active):
        q_start, q_end = cu[s], cu[s + 1]
        q_len = q_end - q_start
        kv_len = lens[s]
        for i in range(q_len):
            pos = kv_len - q_len + i
            page = int(table[s, pos // page_size])
            off = pos % page_size
            view[page, off, 0::2] = key[q_start + i]
            view[page, off, 1::2] = value[q_start + i]

    # Step 2: attend, per request, over the unpaged history.
    outputs = torch.zeros(query.shape[0], num_heads, head_dim,
                          dtype=torch.float32)
    positions = torch.arange(max_kv_len)
    for s in range(active):
        q_start, q_end = cu[s], cu[s + 1]
        q_len = q_end - q_start
        if q_len == 0:
            continue
        kv_len = lens[s]
        # Unpage: [pages_per_seq, page_size, kv_heads_x2, d] -> flat.
        flat = view[table[s].long()].reshape(max_kv_len, kv_heads_x2,
                                             head_dim).to(torch.float32)
        all_k = flat[:, 0::2].repeat_interleave(q_per_kv, dim=1)
        all_v = flat[:, 1::2].repeat_interleave(q_per_kv, dim=1)

        q = query[q_start:q_end].to(torch.float32)
        # [heads, q_len, kv_len]
        scores = torch.einsum("qhd,khd->hqk", q, all_k) * sm_scale
        if k_scale is not None:
            scores = scores * k_scale
        if q_scale is not None:
            scores = scores * q_scale
        if soft_cap is not None:
            scores = soft_cap * torch.tanh(scores / soft_cap)

        token_kv_pos = torch.arange(kv_len - q_len, kv_len)
        mask = positions[None, :] < kv_len
        if use_causal_mask:
            mask = mask & (positions[None, :] <= token_kv_pos[:, None])
        if sliding_window is not None:
            mask = mask & (positions[None, :] >
                           token_kv_pos[:, None] - sliding_window)
        scores = torch.where(mask[None, :, :], scores,
                             torch.full_like(scores, -1e30))

        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", weights, all_v)
        if v_scale is not None:
            out = out * v_scale
        outputs[q_start:q_end] = out

    return outputs.to(query.dtype)


def make_reference_kernel(spec: LayerSpec, *, q_scale=None, k_scale=None,
                          v_scale=None) -> Callable[..., torch.Tensor]:
    """Bind a `LayerSpec`'s config onto `reference_rpa_kernel`.

    The same binding `build_tokamax_rpa_kernel` does, so the two closures are
    interchangeable in `attention_forward`.
    """

    def kernel(*args):
        return reference_rpa_kernel(
            *args,
            sm_scale=spec.scale,
            sliding_window=spec.sliding_window,
            soft_cap=spec.logits_soft_cap,
            use_causal_mask=spec.use_causal_mask,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    return kernel


def make_reference_operator() -> Callable[..., Any]:
    """A tokamax-shaped stand-in for the operator that actually attends.

    `Recorder` proves the call convention but returns zeros. This one has
    tokamax's operator signature -- queries first, the cache fourth,
    everything else by keyword -- and computes the real thing, which is what
    lets the sharded path be compared against the unsharded one.

    Like the real operator it takes the pool either packed (5-D) or already
    fused (4-D), folds it itself, and returns whichever shape it was given.
    """

    def operator(queries, keys, values, kv_cache, kv_lens, page_indices,
                 cu_q_lens, distribution, **config):
        if kv_cache.ndim == 5:
            pool = kv_cache
        else:
            pages, page_size, kv_heads_x2, head_dim = kv_cache.shape
            packing = get_dtype_packing(kv_cache.dtype)
            pool = kv_cache.view(pages, page_size, kv_heads_x2 // packing,
                                 packing, head_dim)
        out = reference_rpa_kernel(pool, queries, keys, values, kv_lens,
                                   page_indices, cu_q_lens, distribution,
                                   None, **config)
        return out, kv_cache

    return operator


def oracle_attention(
    workload: Workload,
    spec: LayerSpec,
    query: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> torch.Tensor:
    """Ground truth, computed without any pool, paging or block table.

    Straight dense attention over each request's contiguous history. If this
    and the paged path agree, the page arithmetic, the K/V interleave and
    the writeback are all correct.
    """
    q_per_kv = spec.num_heads // spec.num_kv_heads
    outputs = torch.zeros(query.shape[0], spec.num_heads, spec.head_size,
                          dtype=torch.float32)
    start = 0
    for req, (kv_len, q_len) in enumerate(zip(workload.seq_lens,
                                              workload.q_lens)):
        q = query[start:start + q_len].to(torch.float32)
        k = keys[req].to(torch.float32).repeat_interleave(q_per_kv, dim=1)
        v = values[req].to(torch.float32).repeat_interleave(q_per_kv, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q, k) * spec.scale
        if spec.logits_soft_cap is not None:
            scores = spec.logits_soft_cap * torch.tanh(
                scores / spec.logits_soft_cap)
        token_kv_pos = torch.arange(kv_len - q_len, kv_len)
        positions = torch.arange(kv_len)
        mask = positions[None, :] <= token_kv_pos[:, None]
        if spec.sliding_window is not None:
            mask = mask & (positions[None, :] >
                           token_kv_pos[:, None] - spec.sliding_window)
        scores = torch.where(mask[None, :, :], scores,
                             torch.full_like(scores, -1e30))
        weights = torch.softmax(scores, dim=-1)
        outputs[start:start + q_len] = torch.einsum("hqk,khd->qhd", weights,
                                                    v)
        start += q_len
    return outputs


# =====================================================================
# Part 5 -- the checks
# =====================================================================

_CHECKS: list[Callable[[], str | None]] = []


def check(fn: Callable[[], str | None]) -> Callable[[], str | None]:
    _CHECKS.append(fn)
    return fn


class Recorder:
    """A stand-in for tokamax's operator that records its call."""

    def __init__(self, kv_heads_x2: int | None = None):
        self.calls: list[dict[str, Any]] = []
        self.kv_heads_x2 = kv_heads_x2

    def __call__(self, queries, keys, values, kv_cache, kv_lens,
                 page_indices, cu_q_lens, distribution, **kwargs):
        self.calls.append(dict(queries=queries, keys=keys, values=values,
                               kv_cache=kv_cache, kv_lens=kv_lens,
                               page_indices=page_indices,
                               cu_q_lens=cu_q_lens,
                               distribution=distribution, kwargs=kwargs))
        out = torch.zeros(queries.shape, dtype=queries.dtype)
        return out, kv_cache


@check
def check_kv_cache_shape() -> str:
    """The 5D packed pool geometry, on both dtypes and both head widths."""
    # bf16: packing 2, 8 KV heads -> 16 fused heads -> 8 x 2.
    assert get_kv_cache_shape(24, 256, 8, 128, torch.bfloat16) == (
        24, 256, 8, 2, 128)
    # fp8: packing 4 -> 4 x 4. Same byte count, different tiling.
    assert get_kv_cache_shape(24, 256, 8, 128, torch.float8_e4m3fn) == (
        24, 256, 4, 4, 128)
    # head_size 64 is the hd64 layout: K/V packed along head_dim, so the
    # fused axis is num_kv_heads, not 2x. It is also refused by the builder.
    assert get_kv_cache_shape(24, 256, 8, 64, torch.bfloat16) == (
        24, 256, 4, 2, 128)
    # Narrow heads round up to the 128 alignment.
    assert get_kv_cache_shape(4, 128, 1, 96, torch.bfloat16) == (
        4, 128, 1, 2, 128)
    # The dtype-less probe still puts num_blocks at dim 0.
    assert get_kv_cache_shape(4242, 128, 2, 128, "auto")[0] == 4242
    assert get_kv_cache_shape(4242, 128, 2, 128, torch.bfloat16)[0] == 4242
    # A string dtype resolves the same as the torch dtype.
    assert (get_kv_cache_shape(4, 128, 2, 128, "fp8") ==
            get_kv_cache_shape(4, 128, 2, 128, torch.float8_e4m3fn))
    try:
        get_kv_cache_shape(4, 128, 2, 128, "not-a-dtype")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown dtype string should raise")
    return "5 dtype/width combinations"


@check
def check_kv_cache_view_fold() -> str:
    """The 5D->4D fold is free, exactly invertible, and order-preserving."""
    cache = torch.arange(24 * 256 * 8 * 2 * 128,
                         dtype=torch.float32).reshape(24, 256, 8, 2, 128)
    view = tokamax_kv_cache_view(cache)
    assert view.shape == (24, 256, 16, 128)
    # Free: a view, not a copy.
    assert view.data_ptr() == cache.data_ptr()
    # Exactly invertible -- this is what lets tokamax fold the pool inside its
    # own trace and hand back the shape the caller gave it.
    assert torch.equal(view.view(cache.shape), cache)
    # The packing lane is the minor of the merged axis: 5D [g, p] lands on
    # 4D [g * packing + p].
    for group in range(8):
        for lane in range(2):
            assert torch.equal(cache[3, 7, group, lane],
                               view[3, 7, group * 2 + lane])
    return "shape, aliasing, invertibility, lane order"


@check
def check_pool_crosses_the_boundary_unfolded() -> str:
    """The operator is handed the runner's pool itself, not a reshape of it.

    Free as the fold is, a reshape on this side of the custom-op boundary is
    a tensor in its own right, and the pool is donated: the operator would
    then reuse *that* buffer and leave the runner's pool live alongside it.
    On a 397B-class model that is an extra 40 GiB of HBM temporaries per
    worker, which fails the backbone compile outright.
    """
    spec = LayerSpec.llama3_8b()
    recorder = Recorder()
    kernel = build_tokamax_rpa_kernel(spec, operator=recorder)
    q, k, v = WORKLOAD.qkv()
    cache = WORKLOAD.empty_kv_cache()
    attention_forward(spec, kernel, q, k, v, cache, WORKLOAD.metadata())

    got = recorder.calls[0]["kv_cache"]
    assert got is cache, f"{type(got)} {tuple(got.shape)}"
    assert tuple(got.shape) == WORKLOAD.kv_cache_shape
    return "the pool arrives 5D and unaliased"


@check
def check_builder_guards() -> str:
    """Every configuration tokamax cannot serve fails at build time."""
    recorder = Recorder()
    cases = [
        ("model-axis devices",
         dict(spec=LayerSpec.llama3_8b(), mesh_shape={"model": 4})),
        ("TPU_KERNEL_ITER_MODE",
         dict(spec=LayerSpec.llama3_8b(), kernel_iter_mode=True)),
        ("hd64", dict(spec=LayerSpec.llama3_8b(head_size=64))),
        ("VLLM_KV_CACHE_LAYOUT",
         dict(spec=LayerSpec.llama3_8b(
             kv_layout=KVLayout.SEQ_ALONG_LANE))),
        ("Cross-layer KV sharing",
         dict(spec=LayerSpec.llama3_8b(), skip_kv_update=True)),
        ("decode_query_size=4",
         dict(spec=LayerSpec.llama3_8b(decode_query_size=4))),
        ("use_causal_mask=False",
         dict(spec=LayerSpec.llama3_8b(use_causal_mask=False))),
    ]
    for fragment, kwargs in cases:
        try:
            build_tokamax_rpa_kernel(operator=recorder, **kwargs)
        except NotImplementedError as exc:
            assert fragment in str(exc), (fragment, str(exc))
        else:
            raise AssertionError(f"expected a raise for {fragment!r}")

    # Only the "model" axis is refused. A mesh that is wide on some other
    # axis -- attention data parallelism, say -- is one chip per model shard
    # and builds fine, as does an absent mesh.
    build_tokamax_rpa_kernel(LayerSpec.llama3_8b(), operator=recorder,
                             mesh_shape={"model": 1, "attn_dp": 8})
    build_tokamax_rpa_kernel(LayerSpec.llama3_8b(), operator=recorder)

    # Several at once are reported together, not one at a time.
    try:
        build_tokamax_rpa_kernel(
            LayerSpec.llama3_8b(head_size=64, decode_query_size=4),
            operator=recorder)
    except NotImplementedError as exc:
        for fragment in ("hd64", "decode_query_size=4"):
            assert fragment in str(exc), (fragment, str(exc))
    else:
        raise AssertionError("expected a combined raise")

    # Sinks are refused at call time -- they are not in the signature.
    kernel = build_tokamax_rpa_kernel(LayerSpec.llama3_8b(),
                                      operator=recorder)
    try:
        kernel(*([None] * 8), torch.ones(4))
    except NotImplementedError as exc:
        assert "Attention sinks" in str(exc)
    else:
        raise AssertionError("expected a raise for sinks")
    assert not recorder.calls, "no guard should have reached the operator"
    return f"{len(cases)} guards + combined + sinks"


@check
def check_operator_call_convention() -> str:
    """The nine-argument closure maps onto tokamax's positional signature."""
    workload = WORKLOAD
    spec = LayerSpec.llama3_8b()
    recorder = Recorder()
    kernel = build_tokamax_rpa_kernel(spec, operator=recorder, k_scale=0.5,
                                      v_scale=0.25)
    md = workload.metadata()
    q, k, v = workload.qkv()
    cache = workload.empty_kv_cache()
    attention_forward(spec, kernel, q, k, v, cache, md)

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    # Order: tokamax takes (queries, keys, values, kv_cache, kv_lens,
    # page_indices, cu_q_lens, distribution); the repo's closure takes
    # kv_cache first. Getting this permutation wrong is silent.
    assert call["queries"].shape == (1024, 32, 128)
    assert call["keys"].shape == (1024, 8, 128)
    assert call["values"].shape == (1024, 8, 128)
    # The pool arrives in the packed 5D shape it was allocated with; tokamax
    # folds it onto its 4D cache jax-side.
    assert call["kv_cache"].shape == (24, 256, 8, 2, 128)
    assert torch.equal(call["kv_lens"], md.seq_lens)
    assert torch.equal(call["page_indices"], md.block_tables)
    assert call["page_indices"].ndim == 1, "tokamax asserts ndim == 1"
    assert torch.equal(call["cu_q_lens"], md.query_start_loc)
    assert torch.equal(call["distribution"], md.request_distribution)

    kwargs = call["kwargs"]
    assert kwargs == dict(use_causal_mask=True, sm_scale=128**-0.5,
                          sliding_window=None, soft_cap=None, q_scale=None,
                          k_scale=0.5, v_scale=0.25), kwargs
    # The three deliberate omissions. `skip_kv_update` is dead in tokamax
    # (static_argnames only); kv_layout and decode_query_size are pinned to
    # tokamax's defaults by the guards.
    for absent in ("skip_kv_update", "kv_layout", "decode_query_size"):
        assert absent not in kwargs, absent
    return "8 positionals, 7 keywords, 3 omissions"


@check
def check_forward_numerics() -> str:
    """The whole pipeline against an oracle that never sees the pool."""
    workload = WORKLOAD
    spec = LayerSpec.llama3_8b()
    keys, values = workload.history()
    cache = workload.empty_kv_cache()
    workload.seed_cache(cache, keys, values)
    key, value = workload.new_kv(keys, values)
    query = workload.query()
    md = workload.metadata()

    got = attention_forward(spec, make_reference_kernel(spec), query, key,
                            value, cache, md)
    want = oracle_attention(workload, spec, query, keys, values)

    live = md.query_start_loc[workload.num_reqs].item()
    torch.testing.assert_close(got[:live].to(torch.float32), want[:live],
                               atol=2e-2, rtol=2e-2)
    # Padding rows past the live tokens must stay zero, not garbage.
    assert torch.all(got[live:] == 0)
    return f"{live} live tokens over {workload.num_reqs} requests"


@check
def check_multi_process_tensor_parallelism() -> str:
    """Four workers, each `model=1`, reconstruct the whole-model answer.

    This is the only tensor parallelism the tokamax path supports, and the
    one the deployment actually runs: the heads are split *above* this file,
    one worker per chip, so each worker builds an ordinary TP=1 closure over
    its own narrower pool. In-process sharding is refused by the guard --
    the singleton has no mesh to shard over -- so there is no `shard_map`
    here to get the partition axes wrong in.

    What this rules out is the pool geometry drifting with the head count:
    a worker's 5D pool has to stay a faithful narrow copy of the global
    one, or the paging arithmetic lands on different slots per worker.
    """
    workload = WORKLOAD
    spec = LayerSpec.llama3_8b()
    keys, values = workload.history()
    key, value = workload.new_kv(keys, values)
    query = workload.query()
    md = workload.metadata()
    operator = make_reference_operator()

    whole_cache = workload.empty_kv_cache()
    workload.seed_cache(whole_cache, keys, values)
    whole = attention_forward(
        spec, build_tokamax_rpa_kernel(spec, operator=operator), query, key,
        value, whole_cache, md)

    tp = 4
    q_per_worker = spec.num_heads // tp
    kv_per_worker = spec.num_kv_heads // tp
    shard_spec = dataclasses.replace(spec, num_heads=q_per_worker,
                                     num_kv_heads=kv_per_worker)
    shard_workload = dataclasses.replace(workload, num_heads=q_per_worker,
                                         num_kv_heads=kv_per_worker)

    outs = []
    for rank in range(tp):
        kv_slice = slice(rank * kv_per_worker, (rank + 1) * kv_per_worker)
        my_keys = [k[:, kv_slice] for k in keys]
        my_values = [v[:, kv_slice] for v in values]
        my_key, my_value = shard_workload.new_kv(my_keys, my_values)
        my_query = query[:, rank * q_per_worker:(rank + 1) * q_per_worker]

        cache = shard_workload.empty_kv_cache()
        shard_workload.seed_cache(cache, my_keys, my_values)
        # `mesh_shape` is what a worker sees: one chip on the model axis.
        kernel = build_tokamax_rpa_kernel(shard_spec, operator=operator,
                                          mesh_shape={"model": 1})
        outs.append(attention_forward(shard_spec, kernel, my_query, my_key,
                                      my_value, cache, md))

    got = torch.cat(outs, dim=1)
    torch.testing.assert_close(got, whole, atol=0, rtol=0)

    # And against the oracle, so "equal to the whole-model run" cannot be
    # equal-and-wrong.
    live = md.query_start_loc[workload.num_reqs].item()
    want = oracle_attention(workload, spec, query, keys, values)
    torch.testing.assert_close(got[:live].to(torch.float32), want[:live],
                               atol=2e-2, rtol=2e-2)
    return (f"{spec.num_kv_heads} KV heads over {tp} single-chip workers, "
            "bit-identical to the whole-model run")


@check
def check_kv_writeback() -> str:
    """The step's new K/V land in exactly the right slots, and only those."""
    workload = WORKLOAD
    spec = LayerSpec.llama3_8b()
    keys, values = workload.history()
    cache = workload.empty_kv_cache()
    workload.seed_cache(cache, keys, values)
    before = cache.clone()
    key, value = workload.new_kv(keys, values)
    md = workload.metadata()

    attention_forward(spec, make_reference_kernel(spec), workload.query(),
                      key, value, cache, md)

    view = tokamax_kv_cache_view(cache)
    table = md.block_tables.view(workload.max_num_seqs,
                                 workload.pages_per_req)
    touched = set()
    for req, (kv_len, q_len) in enumerate(zip(workload.seq_lens,
                                              workload.q_lens)):
        for i in range(q_len):
            pos = kv_len - q_len + i
            page = int(table[req, pos // workload.page_size])
            off = pos % workload.page_size
            touched.add((page, off))
            assert torch.equal(view[page, off, 0::2], keys[req][pos])
            assert torch.equal(view[page, off, 1::2], values[req][pos])

    # Nothing outside those slots moved -- no stray writes from the padded
    # batch rows or the padded token buffer.
    changed = (cache != before).any(dim=-1).any(dim=-1).any(dim=-1)
    for page in range(workload.num_blocks):
        for off in range(workload.page_size):
            if bool(changed[page, off]) != ((page, off) in touched):
                raise AssertionError(
                    f"page {page} slot {off}: changed="
                    f"{bool(changed[page, off])}, expected="
                    f"{(page, off) in touched}")
    return f"{len(touched)} slots written, 0 stray"


@check
def check_narrow_head_padding() -> str:
    """A 96-wide layer sharing a 128-wide pool: pad in, slice out."""
    workload = dataclasses.replace(WORKLOAD, num_heads=4, num_kv_heads=2,
                                   head_size=96)
    spec = LayerSpec(num_heads=4, num_kv_heads=2, head_size=96,
                     scale=96**-0.5)
    cache = workload.empty_kv_cache()
    assert cache.shape[-1] == 128, "the pool is 128 wide"

    recorder = Recorder()
    kernel = build_tokamax_rpa_kernel(spec, operator=recorder)
    q, k, v = workload.qkv(head_size=96, num_heads=4, num_kv_heads=2)
    out = attention_forward(spec, kernel, q, k, v, cache,
                            workload.metadata())

    call = recorder.calls[0]
    # The kernel saw the full page width...
    assert call["queries"].shape[-1] == 128
    assert torch.all(call["queries"][..., 96:] == 0)
    assert torch.all(call["keys"][..., 96:] == 0)
    assert torch.all(call["values"][..., 96:] == 0)
    # ...and the layer got its own width back.
    assert out.shape[-1] == 96

    # And the padding really is inert: the padded run matches an unpadded
    # oracle over 96 dims.
    keys, values = workload.history()
    keys = [t[..., :96] for t in keys]
    values = [t[..., :96] for t in values]
    cache = workload.empty_kv_cache()
    workload.seed_cache(cache, keys, values)
    key, value = workload.new_kv(keys, values)
    query = workload.query()[:, :4, :96].contiguous()
    md = workload.metadata()
    got = attention_forward(spec, make_reference_kernel(spec), query, key,
                            value, cache, md)
    want = oracle_attention(workload, spec, query, keys, values)
    live = md.query_start_loc[workload.num_reqs].item()
    torch.testing.assert_close(got[:live].to(torch.float32), want[:live],
                               atol=2e-2, rtol=2e-2)
    return "head_size 96 into a 128-wide page"


@check
def check_flat_query_reshape() -> str:
    """A 2D `[tokens, heads * dim]` query round-trips to 2D."""
    workload = WORKLOAD
    spec = LayerSpec.llama3_8b()
    recorder = Recorder()
    kernel = build_tokamax_rpa_kernel(spec, operator=recorder)
    q, k, v = workload.qkv()
    flat_q = q.reshape(workload.padded_num_tokens, 32 * 128)
    flat_k = k.reshape(workload.padded_num_tokens, 8 * 128)
    out = attention_forward(spec, kernel, flat_q, flat_k,
                            flat_k.clone(), workload.empty_kv_cache(),
                            workload.metadata())
    # The kernel still sees 3D...
    assert recorder.calls[0]["queries"].shape == (1024, 32, 128)
    # ...and the caller gets 2D back.
    assert out.shape == (1024, 32 * 128)

    # A 3D query gives a 3D result from the same code path.
    out3d = attention_forward(spec, kernel, q, k, v,
                              workload.empty_kv_cache(),
                              workload.metadata())
    assert out3d.shape == (1024, 32, 128)
    return "2D in -> 2D out, 3D in -> 3D out"


@check
def check_fp8_quantization() -> str:
    """fp8 pool: quantize on the way in, scale on the way out."""
    workload = dataclasses.replace(WORKLOAD, dtype=torch.bfloat16)
    k_scale, v_scale = 0.5, 0.25
    spec = LayerSpec.llama3_8b(
        kv_cache_quantized_dtype=torch.float8_e4m3fn)
    shape = get_kv_cache_shape(workload.num_blocks, workload.page_size,
                               workload.num_kv_heads, workload.head_size,
                               torch.float8_e4m3fn)
    assert shape == (24, 256, 4, 4, 128), shape
    cache = torch.zeros(shape, dtype=torch.float8_e4m3fn)

    recorder = Recorder()
    kernel = build_tokamax_rpa_kernel(spec, operator=recorder,
                                      k_scale=k_scale, v_scale=v_scale)
    q, k, v = workload.qkv()
    attention_forward(spec, kernel, q, k, v, cache, workload.metadata(),
                      k_scale_float=k_scale, v_scale_float=v_scale)

    call = recorder.calls[0]
    assert call["keys"].dtype == torch.float8_e4m3fn
    assert call["values"].dtype == torch.float8_e4m3fn
    # The query is NOT quantized -- q_scale stays None on this path.
    assert call["queries"].dtype == torch.bfloat16
    assert call["kwargs"]["q_scale"] is None
    assert call["kwargs"]["k_scale"] == k_scale
    # Stored as x / scale, so `stored * scale` recovers x. This is the
    # convention the kernel's `qk *= scale_k` implements.
    recovered = call["keys"].to(torch.float32) * k_scale
    torch.testing.assert_close(recovered, k.to(torch.float32), atol=6e-2,
                               rtol=6e-2)

    # A zero scale is a configuration error, not a silent divide.
    try:
        attention_forward(spec, kernel, q, k, v, cache,
                          workload.metadata(), k_scale_float=0.0)
    except ValueError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("expected a raise for a zero scale")
    return "fp8 e4m3 round-trip within 6e-2"


@check
def check_sliding_window_and_soft_cap() -> str:
    """Both optional model features reach the kernel and change the math."""
    workload = WORKLOAD
    keys, values = workload.history()
    query = workload.query()
    md = workload.metadata()
    live = md.query_start_loc[workload.num_reqs].item()

    for spec in (LayerSpec.llama3_8b(sliding_window=128),
                 LayerSpec.llama3_8b(logits_soft_cap=30.0)):
        recorder = Recorder()
        kernel = build_tokamax_rpa_kernel(spec, operator=recorder)
        q, k, v = workload.qkv()
        attention_forward(spec, kernel, q, k, v,
                          workload.empty_kv_cache(), md)
        kwargs = recorder.calls[0]["kwargs"]
        assert kwargs["sliding_window"] == spec.sliding_window
        assert kwargs["soft_cap"] == spec.logits_soft_cap

        cache = workload.empty_kv_cache()
        workload.seed_cache(cache, keys, values)
        key, value = workload.new_kv(keys, values)
        got = attention_forward(spec, make_reference_kernel(spec), query,
                                key, value, cache, md)
        want = oracle_attention(workload, spec, query, keys, values)
        torch.testing.assert_close(got[:live].to(torch.float32),
                                   want[:live], atol=2e-2, rtol=2e-2)

    # And they really do change the result, so the checks above are not
    # vacuously comparing two no-ops.
    plain = LayerSpec.llama3_8b()
    cache = workload.empty_kv_cache()
    workload.seed_cache(cache, keys, values)
    key, value = workload.new_kv(keys, values)
    base = attention_forward(plain, make_reference_kernel(plain), query,
                             key, value, cache, md)
    windowed_spec = LayerSpec.llama3_8b(sliding_window=128)
    cache = workload.empty_kv_cache()
    workload.seed_cache(cache, keys, values)
    windowed = attention_forward(windowed_spec,
                                 make_reference_kernel(windowed_spec),
                                 query, key, value, cache, md)
    assert not torch.allclose(base[:live].to(torch.float32),
                              windowed[:live].to(torch.float32),
                              atol=1e-3)
    return "sliding_window=128 and soft_cap=30 vs the oracle"


@check
def check_memory_profiling_noop() -> str:
    """`determine_available_memory` calls forward with an empty pool."""
    spec = LayerSpec.llama3_8b()
    workload = WORKLOAD
    q, k, v = workload.qkv()

    def exploding_kernel(*_args):
        raise AssertionError("the kernel must not be reached")

    out = attention_forward(spec, exploding_kernel, q, k, v,
                            torch.zeros(0, dtype=torch.bfloat16),
                            workload.metadata())
    assert out.shape == q.shape
    assert torch.all(out == 1)
    return "empty pool short-circuits to ones_like"


@check
def check_fused_output_quantization_refused() -> str:
    """The one pre-existing guard on this path."""
    spec = LayerSpec.llama3_8b()
    workload = WORKLOAD
    q, k, v = workload.qkv()
    try:
        attention_forward(spec, lambda *a: None, q, k, v,
                          workload.empty_kv_cache(), workload.metadata(),
                          output_scale=torch.ones(1))
    except NotImplementedError as exc:
        assert "output quantization" in str(exc)
    else:
        raise AssertionError("expected a raise")
    return "output_scale rejected"


@check
def check_incompatible_cache_is_caught() -> str:
    """A pool tokamax hands back at the wrong shape must not be copied in."""
    spec = LayerSpec.llama3_8b()
    workload = WORKLOAD

    class Shrinker(Recorder):

        def __call__(self, queries, *args, **kwargs):
            super().__call__(queries, *args, **kwargs)
            return (torch.zeros(queries.shape, dtype=queries.dtype),
                    torch.zeros(4, 4))

    kernel = build_tokamax_rpa_kernel(spec, operator=Shrinker())
    q, k, v = workload.qkv()
    try:
        attention_forward(spec, kernel, q, k, v,
                          workload.empty_kv_cache(), workload.metadata())
    except RuntimeError as exc:
        assert "incompatible KV cache" in str(exc), str(exc)
    else:
        raise AssertionError("expected a raise")
    return "shape mismatch caught before the copy"


@check
def check_real_tokamax_operator() -> str | None:
    """The only check that needs a TPU, JAX and tokamax on the path.

    Returns None (SKIP) without them. This is the one that proves the
    folded 4D cache, the flat ragged metadata and the keyword config are
    shapes tokamax actually accepts -- everything above proves wiring.
    """
    try:
        import importlib
        operator = getattr(importlib.import_module(_TOKAMAX_OP_PATH),
                           _TOKAMAX_OP_ATTR)
    except Exception as exc:  # noqa: BLE001 - any import failure is a skip
        _skip(f"tokamax not importable ({type(exc).__name__}: {exc})")
        return None

    try:
        import torch_tpu  # noqa: F401
        device = "tpu"
        torch.zeros(1, device=device)
    except Exception as exc:  # noqa: BLE001
        _skip(f"no TPU device ({type(exc).__name__}: {exc})")
        return None

    workload = WORKLOAD
    spec = LayerSpec.llama3_8b()
    keys, values = workload.history()
    cache = workload.empty_kv_cache()
    workload.seed_cache(cache, keys, values)
    key, value = workload.new_kv(keys, values)
    query = workload.query()
    md = workload.metadata()
    want = oracle_attention(workload, spec, query, keys, values)

    kernel = build_tokamax_rpa_kernel(spec, operator=operator)
    to_dev = lambda t: t.to(device)  # noqa: E731
    got = attention_forward(
        spec, kernel, to_dev(query), to_dev(key), to_dev(value),
        to_dev(cache),
        AttentionMetadata(
            block_tables=to_dev(md.block_tables),
            seq_lens=to_dev(md.seq_lens),
            query_start_loc=to_dev(md.query_start_loc),
            request_distribution=to_dev(md.request_distribution),
        ))
    live = md.query_start_loc[workload.num_reqs].item()
    torch.testing.assert_close(got[:live].cpu().to(torch.float32),
                               want[:live], atol=5e-2, rtol=5e-2)
    return f"real operator matches the oracle on {live} tokens"


_SKIPS: list[str] = []


def _skip(reason: str) -> bool:
    _SKIPS.append(reason)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the traceback for each failure")
    parser.add_argument("-k", metavar="SUBSTRING", default="",
                        help="only run checks whose name contains this")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    selected = [c for c in _CHECKS if args.k in c.__name__]
    width = max(len(c.__name__) for c in selected) if selected else 0
    passed = failed = skipped = 0

    for fn in selected:
        before = len(_SKIPS)
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - this is the reporter
            failed += 1
            print(f"FAIL  {fn.__name__:<{width}}  {type(exc).__name__}: "
                  f"{exc}")
            if args.verbose:
                traceback.print_exc()
            continue
        if detail is None:
            skipped += 1
            reason = _SKIPS[before] if len(_SKIPS) > before else "skipped"
            print(f"SKIP  {fn.__name__:<{width}}  {reason}")
        else:
            passed += 1
            print(f"ok    {fn.__name__:<{width}}  {detail}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
