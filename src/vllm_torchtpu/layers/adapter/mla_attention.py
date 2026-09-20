# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU multi-head latent attention (MLA) backends.

Split out of `attention.py`, which is now only the ragged-paged-attention
half: the FLASH_ATTN flash kernel and the CUSTOM backend that routes onto
tokamax's batched RPA operator. Nothing here shares a kernel, a KV pool
layout or a code path with that half -- MLA has its own cache geometry
(`kernels.mla.v2`), its own sparse (nope, rope) split cache and its own DCP
implementation -- so the only things this module still takes from
`attention.py` are three KV-cache dtype/page-size helpers.

This module must be imported for its `@register_backend` side effect;
`layers/__init__.py` does that alongside `attention`.
"""

import dataclasses
import functools
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (AttentionBackend, AttentionLayer,
                                       MLAAttentionImpl)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_or_create_dcp_mesh
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import \
    pcp_streaming_jax_op
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (KVCacheLayout,
                                                      KVCacheType,
                                                      SparseMLAKVCacheSpec)
from vllm_torchtpu.kernels.mla.v2 import kernel as mla_v2_kernel
from vllm_torchtpu.layers.adapter.attention import (_resolve_kv_cache_dtype,
                                                    get_dtype_packing,
                                                    get_tpu_min_page_size)
from vllm_torchtpu.layers.adapter.cp_mla_attention import (
    all_gather_heads, merge_lse_partials_scatter_heads)
from vllm_torchtpu.layers.core.attention_interface import mla_attention
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

_SPARSE_MLA_DCP_INPUT_PARTITION_SPECS = (
    P(DCP_AXIS_NAME),  # kv_cache_nope, this rank's position shard
    P(DCP_AXIS_NAME),  # kv_cache_rope
    P(),  # ql_nope
    P(),  # q_pe
    P(),  # kv_c_normed -- no all-gather: every rank already has it
    P(),  # k_pe
    # This rank's slice of every token's global top-k, already local indices.
    # Sharded, so the global leading dim is `dcp_size * num_tokens`.
    P(DCP_AXIS_NAME),  # local_topk_indices
    P(),  # seq_lens
    P(),  # block_tables (virtual page ordinals)
    P(),  # query_start_loc
    P(),  # request_distribution
)
_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS = (
    P(DCP_AXIS_NAME),  # updated nope cache
    P(DCP_AXIS_NAME),  # updated rope cache
    P(DCP_AXIS_NAME),  # partial attention output over this rank's KV positions
    # Matching log-sum-exps, `-inf` where this rank owns nothing.
    P(DCP_AXIS_NAME),  # partial lse
)


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class PallasMLAttentionBackend(AttentionBackend):
    """TPU attention backend utilizing customized Pallas kernels for DeepSeek MLA.
    """
    supported_kv_cache_dtypes = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_ds_mla",
    ]

    # DeepSeek-V4 MLA packed-cache constants (64-dim RoPE tail, 64-wide UE8M0 blocks).
    _DS_MLA_ROPE_HEAD_DIM = 64
    _DS_MLA_QUANT_BLOCK = 64

    @staticmethod
    def _ds_mla_packed_width(nope_dim: int, rope_head_dim: int,
                             quant_block: int) -> int:
        """Bytes per token in the packed sparse (head_dim=512) KV cache."""
        # nope fp8 (1B) + rope bf16 (2B) + UE8M0 block scale (1B)
        return nope_dim + rope_head_dim * 2 + (nope_dim // quant_block)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def is_mla() -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["PallasMLAttentionBackendImpl"]:
        return PallasMLAttentionBackendImpl

    @staticmethod
    def _is_ds_mla_packed_cache(cache_dtype_str: str | torch.dtype) -> bool:
        # Recognizes both raw "fp8_ds_mla" string and resolved torch.uint8 dtype.
        if isinstance(cache_dtype_str,
                      str) and cache_dtype_str.lower().strip() == "fp8_ds_mla":
            return True
        return isinstance(cache_dtype_str,
                          torch.dtype) and cache_dtype_str == torch.uint8

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
        *,
        head_size_is_packed_width: bool = False,
    ) -> tuple[int, ...]:
        """Shape of one MLA KV cache tensor.

        Set `head_size_is_packed_width` when `head_size` is already the packed
        byte width; callers passing `kv_lora_rank` leave it False.
        """
        if (isinstance(cache_dtype_str, str)
                and cache_dtype_str.lower().strip() == "auto"):
            return (num_blocks, block_size, 1, cdiv(head_size, 128) * 128)
        if PallasMLAttentionBackend._is_ds_mla_packed_cache(cache_dtype_str):
            # Packed layout is [nope fp8 | rope bf16 | UE8M0 scales] padded to 128-aligned minor dim.
            if head_size_is_packed_width:
                packed_width = head_size
            else:
                rope_head_dim = PallasMLAttentionBackend._DS_MLA_ROPE_HEAD_DIM
                quant_block = PallasMLAttentionBackend._DS_MLA_QUANT_BLOCK
                nope_dim = head_size - rope_head_dim
                packed_width = (PallasMLAttentionBackend._ds_mla_packed_width(
                    nope_dim, rope_head_dim, quant_block))
            kv_packing = get_dtype_packing(torch.uint8)
            return mla_v2_kernel.get_kv_cache_shape(
                total_num_pages=num_blocks,
                page_size=block_size,
                kv_dim=packed_width,
                kv_dtype=None,
                kv_packing=kv_packing,
            )
        kv_dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        kv_packing = get_dtype_packing(kv_dtype)
        return mla_v2_kernel.get_kv_cache_shape(
            total_num_pages=num_blocks,
            page_size=block_size,
            kv_dim=head_size,
            kv_dtype=None,
            kv_packing=kv_packing,
        )

    @staticmethod
    def get_sparse_kv_cache_specs(
        num_blocks: int,
        block_size: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[SparseMLAKVCacheSpec, SparseMLAKVCacheSpec]:
        """The (nope, rope) specs for one sparse-MLA layer."""
        rope_dim = PallasMLAttentionBackend._DS_MLA_ROPE_HEAD_DIM
        kv_packing = get_dtype_packing(
            _resolve_kv_cache_dtype(cache_dtype_str))
        return (
            SparseMLAKVCacheSpec.create(
                KVCacheType.NOPE,
                KVCacheLayout(envs.TPU_SPARSE_MLA_NOPE_LAYOUT),
                num_blocks,
                block_size,
                head_size - rope_dim,
                kv_packing,
            ),
            SparseMLAKVCacheSpec.create(
                KVCacheType.ROPE,
                KVCacheLayout(envs.TPU_SPARSE_MLA_ROPE_LAYOUT),
                num_blocks,
                block_size,
                rope_dim,
                kv_packing,
            ),
        )

    @staticmethod
    def get_kv_cache_page_size_bytes(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        shape = PallasMLAttentionBackend.get_kv_cache_shape(
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

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        return 1024

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
            return (1, 0, 2, 3, 4)
        return (0, 1, 2, 3)


class VllmTPUDeepseekV32IndexerBackend(DeepseekV32IndexerBackend):
    """Indexer backend for TPU.

    The TPU materializer retains the native 3-D indexer K cache shape
    `(num_blocks, block_size, head_size)` consumed by streamindex. vLLM 0.29
    no longer owns this backend shape hook.

    The page size hooks below delegate to `PallasMLAttentionBackend`. They are
    not the indexer's own page size: `cache_config.block_size` is global across
    KV cache groups, and `update_block_size_for_backend` consults whichever
    backend `Platform._find_non_ssm_backend` returns first. An indexer cache
    registers before its `mla_attn` sibling, so on a sparse-MLA model these are
    the hooks that decide the block size for the MLA latent cache too --
    hardcoding a second copy of the constant would let the two drift, and the
    effective value would silently depend on layer registration order.
    """

    @staticmethod
    def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str | torch.dtype = "auto") -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        return (VllmKVCacheLayout.LBNHC, )

    @staticmethod
    def get_name() -> str:
        return "TPU_STREAMINDEX_INDEXER"

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        return PallasMLAttentionBackend.get_page_size(vllm_config)

    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        return PallasMLAttentionBackend.get_min_page_size(vllm_config)


class PallasMLAttentionBackendImpl(MLAAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_head_dim: int | None = None,
        v_head_dim: int | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        parallel_config = getattr(get_current_vllm_config_or_none(),
                                  "parallel_config", None)
        dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        self.dcp_size = dcp_size if isinstance(dcp_size,
                                               int) and dcp_size > 1 else 1
        self.dcp_interleave_size = getattr(parallel_config,
                                           "cp_kv_cache_interleave_size", 1)
        self._dcp_mesh = None
        if self.dcp_size > 1:
            # The indexer refuses pcp>1 and dcp>1 together (see
            # `VllmTPUSparseAttnIndexer.__init__`)
            pcp_size = getattr(parallel_config,
                               "prefill_context_parallel_size", 1)
            if isinstance(pcp_size, int) and pcp_size > 1:
                raise NotImplementedError(
                    f"pcp_size={pcp_size} and dcp_size={self.dcp_size} are "
                    "both >1; sparse MLA has one KV-position interleave, "
                    "not two.")
            if self.dcp_interleave_size % kv_cache_utils.WORD_BYTES:
                raise ValueError(
                    "DCP sparse MLA needs --cp-kv-cache-interleave-size to be "
                    f"a multiple of {kv_cache_utils.WORD_BYTES} "
                    f"(kv_cache_utils.WORD_BYTES), got "
                    f"{self.dcp_interleave_size}. It is also the best value for "
                    "top-k load balance across the shards -- coarser cycles "
                    "let a contiguous run of winners land on fewer ranks.")
            self._dcp_mesh = get_or_create_dcp_mesh()

    def _get_kv_scales(
            self,
            layer: Any) -> tuple[float | None, float | None, float | None]:
        """Harvest scalar quantization scales from heterogeneous layer attributes.

        Extracting deterministic scalar float values guarantees purely static execution inside
        Pallas FX graphs without tracing dynamic tensor-shape overhead right when consuming
        heterogeneous checkpoint FP8 scales.
        """
        q_scale = getattr(layer, "_q_scale_float", None)
        if q_scale is None and hasattr(layer, "_q_scale"):
            q_scale = (layer._q_scale.item()
                       if isinstance(layer._q_scale, torch.Tensor)
                       and layer._q_scale.ndim == 0 else getattr(
                           layer._q_scale, "tolist", lambda: layer._q_scale)())

        k_scale = getattr(layer, "_k_scale_float", None)
        if k_scale is None and hasattr(layer, "_k_scale"):
            k_scale = (layer._k_scale.item()
                       if isinstance(layer._k_scale, torch.Tensor)
                       and layer._k_scale.ndim == 0 else getattr(
                           layer._k_scale, "tolist", lambda: layer._k_scale)())

        v_scale = getattr(layer, "_v_scale_float", None)
        if v_scale is None and hasattr(layer, "_v_scale"):
            v_scale = (layer._v_scale.item()
                       if isinstance(layer._v_scale, torch.Tensor)
                       and layer._v_scale.ndim == 0 else getattr(
                           layer._v_scale, "tolist", lambda: layer._v_scale)())
        if v_scale is None:
            v_scale = k_scale

        return q_scale, k_scale, v_scale

    def _build_mla_op(
        self,
        layer: Any,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        vllm_context = get_vllm_model_wrapper_context()

        def mla_attention_core_tpu(
            kv_cache: jax.Array,
            q_nope: jax.Array,
            q_pe: jax.Array,
            kv_c_normed: jax.Array,
            k_pe: jax.Array,
            seq_lens: jax.Array,
            block_tables: jax.Array,
            query_start_loc: jax.Array,
            request_distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            metadata = AttentionMetadata(
                input_positions=None,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )
            return mla_attention(
                q_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache,
                metadata,
                vllm_context.mesh,
                layer.num_heads,
                layer.qk_nope_head_dim,
                sm_scale=layer.scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        op_name = f"pallas::mla_attention_{layer.layer_name.replace('.', '_')}"
        mla_jax_op = pallas.jax_op(op_name,
                                   mla_attention_core_tpu,
                                   donate_argnums=(0, ))

        def _fake_mla(kv_cache, q_nope, q_pe, kv_c_normed, k_pe, *args,
                      **kwargs):
            num_tokens = q_nope.size(0)
            out_shape = (num_tokens, layer.num_heads, layer.kv_lora_rank)
            return torch.empty_like(kv_cache), torch.empty(
                out_shape, dtype=q_nope.dtype, device=q_nope.device)

        mla_jax_op.register_fake(_fake_mla)

        def mla_impl(kv_cache: torch.Tensor, q_nope: torch.Tensor,
                     q_pe: torch.Tensor, kv_c_normed: torch.Tensor,
                     k_pe: torch.Tensor, seq_lens: torch.Tensor,
                     block_tables: torch.Tensor, query_start_loc: torch.Tensor,
                     request_distribution: torch.Tensor) -> torch.Tensor:
            new_kv, outputs = mla_jax_op(kv_cache, q_nope, q_pe, kv_c_normed,
                                         k_pe, seq_lens, block_tables,
                                         query_start_loc, request_distribution)

            kv_cache.copy_(new_kv)
            return outputs

        return mla_impl

    def _build_sparse_mla_op(self, layer: Any):
        from vllm_torchtpu.layers.core.attention_interface import \
            sparse_mla_attention

        vllm_context = get_vllm_model_wrapper_context()

        # Per-tensor fp8 dequant scale for the packed nope cache -- the
        # same checkpoint k_scale the dense path uses.
        k_scale = None
        if layer.kv_cache_quantized_dtype is not None:
            _, k_scale, _ = self._get_kv_scales(layer)

        def sparse_mla_attention_core_tpu(
            kv_cache_nope: jax.Array,
            kv_cache_rope: jax.Array,
            ql_nope: jax.Array,
            q_pe: jax.Array,
            kv_c_normed: jax.Array,
            k_pe: jax.Array,
            topk_indices: jax.Array,
            seq_lens: jax.Array,
            block_tables: jax.Array,
            query_start_loc: jax.Array,
            request_distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            nope_spec, rope_spec = layer.mla_kv_spec
            return sparse_mla_attention(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache_nope,
                kv_cache_rope,
                topk_indices,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                vllm_context.mesh,
                nope_spec,
                rope_spec,
                sm_scale=layer.scale,
                k_scale=k_scale,
            )

        op_name = ("pallas::sparse_mla_attention_"
                   f"{layer.layer_name.replace('.', '_')}")
        sparse_mla_jax_op = pallas.jax_op(op_name,
                                          sparse_mla_attention_core_tpu,
                                          donate_argnums=(0, 1))

        def _fake_sparse_mla(kv_cache_nope, kv_cache_rope, ql_nope, q_pe,
                             kv_c_normed, k_pe, topk_indices, *args, **kwargs):
            num_tokens = ql_nope.size(0)
            out_shape = (num_tokens, layer.num_heads, layer.kv_lora_rank)
            return (torch.empty_like(kv_cache_nope),
                    torch.empty_like(kv_cache_rope),
                    torch.empty(out_shape,
                                dtype=ql_nope.dtype,
                                device=ql_nope.device))

        sparse_mla_jax_op.register_fake(_fake_sparse_mla)

        def sparse_mla_impl(
                kv_cache: tuple[torch.Tensor, torch.Tensor],
                ql_nope: torch.Tensor, q_pe: torch.Tensor,
                kv_c_normed: torch.Tensor, k_pe: torch.Tensor,
                topk_indices: torch.Tensor, seq_lens: torch.Tensor,
                block_tables: torch.Tensor, query_start_loc: torch.Tensor,
                request_distribution: torch.Tensor) -> torch.Tensor:
            nope_cache, rope_cache = kv_cache
            new_nope, new_rope, outputs = sparse_mla_jax_op(
                nope_cache, rope_cache, ql_nope, q_pe, kv_c_normed, k_pe,
                topk_indices, seq_lens, block_tables, query_start_loc,
                request_distribution)
            nope_cache.copy_(new_nope)
            rope_cache.copy_(new_rope)
            return outputs

        return sparse_mla_impl

    def _build_sparse_mla_dcp_op(self, layer: Any, dcp_size: int,
                                 interleave_size: int):
        """DCP variant of `_build_sparse_mla_op`: shard the KV cache only.

        In DCP the queries are replicated. `kv_c_normed`/`k_pe` in particular
        need no all-gather -- they are computed from a replicated residual
        stream, so every rank already holds every row and only its write mask
        (owner rank) differs.

        The op returns unmerged partials. Combining them is one log-sum-exp
        reduction over the DCP group, is done by the torch
        caller (`cp_mla_attention.merge_lse_partials_scatter_heads`), which
        also narrows the head axis back down.
        """
        from vllm_torchtpu.layers.core.attention_interface import \
            sparse_mla_attention_dcp

        mesh = self._dcp_mesh

        k_scale = None
        if layer.kv_cache_quantized_dtype is not None:
            _, k_scale, _ = self._get_kv_scales(layer)

        def sparse_mla_attention_core_tpu_dcp(
            kv_cache_nope: jax.Array,
            kv_cache_rope: jax.Array,
            ql_nope: jax.Array,
            q_pe: jax.Array,
            kv_c_normed: jax.Array,
            k_pe: jax.Array,
            local_topk_indices: jax.Array,
            seq_lens: jax.Array,
            block_tables: jax.Array,
            query_start_loc: jax.Array,
            request_distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            nope_spec, rope_spec = layer.mla_kv_spec
            return sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache_nope,
                kv_cache_rope,
                local_topk_indices,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                mesh,
                nope_spec,
                rope_spec,
                sm_scale=layer.scale,
                k_scale=k_scale,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )

        op_name = ("pallas::sparse_mla_attention_dcp_"
                   f"{layer.layer_name.replace('.', '_')}")
        sparse_mla_jax_op = pcp_streaming_jax_op(
            op_name,
            sparse_mla_attention_core_tpu_dcp,
            donate_argnums=(0, 1),
            mesh=mesh,
            input_partition_specs=_SPARSE_MLA_DCP_INPUT_PARTITION_SPECS,
            output_partition_specs=_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS,
        )

        # We must overwrite the default fake implementation as vLLM uses
        # dynamic dimensions for the query.
        def _fake_sparse_mla_dcp(kv_cache_nope, kv_cache_rope, ql_nope, *rest,
                                 **kwargs):
            del rest, kwargs
            return (
                torch.empty_like(kv_cache_nope),
                torch.empty_like(kv_cache_rope),
                torch.empty_like(ql_nope),
                # LSE: one f32 per (token, head).
                ql_nope.new_empty(ql_nope.shape[:2], dtype=torch.float32),
            )

        sparse_mla_jax_op.register_fake(_fake_sparse_mla_dcp)

        def sparse_mla_dcp_impl(
            kv_cache: tuple[torch.Tensor, torch.Tensor],
            ql_nope: torch.Tensor,
            q_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            local_topk_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            block_tables: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nope_cache, rope_cache = kv_cache
            new_nope, new_rope, outputs, lse = sparse_mla_jax_op(
                nope_cache, rope_cache, ql_nope, q_pe, kv_c_normed, k_pe,
                local_topk_indices, seq_lens, block_tables, query_start_loc,
                request_distribution)
            nope_cache.copy_(new_nope)
            rope_cache.copy_(new_rope)
            return outputs, lse

        return sparse_mla_dcp_impl

    def forward(
        self,
        layer: Any,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Executes complete TPU multi-head latent attention evaluation."""
        assert isinstance(
            q, tuple) and len(q) == 2, "q must be a tuple of (q_nope, q_pe)"
        q_nope, q_pe = q
        input_dtype = q_nope.dtype

        # For determine_available_memory when cache memory buffer is empty right during probe
        # (before binding, `layer.kv_cache` is the layer's default empty
        # tensor; after binding, sparse layers hold a (nope, rope) pair).
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
            if output is None:
                # Preserve symbolic token dimensions during the memory probe.
                template = q_nope.flatten(1)[:, :1].expand(
                    -1, layer.num_heads * layer.v_head_dim)
                return torch.ones_like(template)
            output.fill_(1)
            return output

        # Evaluate projection matrices directly across input precision (`bfloat16`/`float16`/`fp8`)
        # without dynamic `.to(torch.float32)` casting right before `torch.bmm`.
        q_nope_t = q_nope.transpose(0, 1)
        w_uk_t = layer.W_UK_T.to(
            q_nope_t.dtype
        ) if layer.W_UK_T.dtype != q_nope_t.dtype else layer.W_UK_T
        ql_nope = torch.bmm(q_nope_t, w_uk_t)
        if hasattr(layer, "W_UK_T_scale"):
            ql_nope = ql_nope * layer.W_UK_T_scale
        ql_nope = ql_nope.transpose(0, 1).to(input_dtype)

        q_scale, k_scale, v_scale = self._get_kv_scales(layer)

        if layer.kv_cache_quantized_dtype is not None:
            kv_c_normed, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                         kv_c_normed,
                                         value=None,
                                         k_scale=k_scale)
            k_pe, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                  k_pe,
                                  value=None,
                                  k_scale=k_scale)

        ql_nope_flat = ql_nope.view(-1, layer.num_heads, layer.kv_lora_rank)
        q_pe_flat = q_pe.view(-1, layer.num_heads, layer.qk_rope_head_dim)
        kv_c_normed_flat = kv_c_normed.view(-1, layer.kv_lora_rank)
        k_pe_flat = k_pe.view(-1, layer.qk_rope_head_dim)

        topk_indices = kwargs.get("topk_indices")
        if topk_indices is not None:
            assert (isinstance(kv_cache, (tuple, list))
                    and len(kv_cache) == 2), (
                        "sparse MLA layers use a native (nope, rope) split "
                        f"cache; got {type(kv_cache)}")
            if self.dcp_size > 1:
                assert getattr(layer, "sparse_mla_dcp_op", None) is not None, (
                    "DCP sparse MLA op was never built; "
                    "`process_weights_after_loading` must run first.")
                # DCP re-spends `dcp_size` of TP's ways on positions instead of
                # heads, so attention runs at `tp // dcp` head sharding: every
                # rank attends with the whole DCP group's heads over its own
                # KV shard, and the head axis is scattered back after the
                # merge.
                ql_nope_dcp = all_gather_heads(ql_nope_flat)
                q_pe_dcp = all_gather_heads(q_pe_flat)
                # `topk_indices` here is not global positions: the DCP indexer
                # already resolved each token's global top-k into this rank's
                # own local cache indices, `-1` where it owns nothing.
                partial, lse = layer.sparse_mla_dcp_op(
                    kv_cache,
                    ql_nope_dcp,
                    q_pe_dcp,
                    kv_c_normed_flat,
                    k_pe_flat,
                    topk_indices,
                    attn_metadata.seq_lens,
                    attn_metadata.block_tables,
                    attn_metadata.query_start_loc,
                    attn_metadata.request_distribution,
                )
                # The scatter undoes the head all-gather above.
                outputs = merge_lse_partials_scatter_heads(partial, lse)
            else:
                if (not hasattr(layer, "sparse_mla_op")
                        or layer.sparse_mla_op is None):
                    layer.sparse_mla_op = self._build_sparse_mla_op(layer)

                outputs = layer.sparse_mla_op(
                    kv_cache,
                    ql_nope_flat,
                    q_pe_flat,
                    kv_c_normed_flat,
                    k_pe_flat,
                    topk_indices,
                    attn_metadata.seq_lens,
                    attn_metadata.block_tables,
                    attn_metadata.query_start_loc,
                    attn_metadata.request_distribution,
                )
        else:
            if not hasattr(layer, "mla_op") or layer.mla_op is None:
                layer.mla_op = self._build_mla_op(layer,
                                                  q_scale=q_scale,
                                                  k_scale=k_scale,
                                                  v_scale=v_scale)

            outputs = layer.mla_op(
                kv_cache,
                ql_nope_flat,
                q_pe_flat,
                kv_c_normed_flat,
                k_pe_flat,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
            )

        outputs_t = outputs.reshape(-1, layer.num_heads,
                                    layer.kv_lora_rank).transpose(0, 1)
        w_uv = layer.W_UV.to(
            outputs_t.dtype
        ) if layer.W_UV.dtype != outputs_t.dtype else layer.W_UV
        out_proj = torch.bmm(outputs_t, w_uv)
        if hasattr(layer, "W_UV_scale"):
            out_proj = out_proj * layer.W_UV_scale
        outputs = out_proj.transpose(0, 1).to(input_dtype).reshape(
            -1, layer.num_heads * layer.v_head_dim)

        if output is not None and outputs is not output:
            output.copy_(outputs)

        return outputs

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Structural no-op. TPU Pallas MLA unifies projection and evaluation inside `self.forward`."""
        pass

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Structural no-op. TPU Pallas MLA unifies projection and evaluation inside `self.forward`."""
        pass

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Structural no-op. KV cache updates occur in-place during ragged paged attention kernel run."""
        pass
