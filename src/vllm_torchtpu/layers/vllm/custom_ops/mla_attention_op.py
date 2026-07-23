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
"""Out-of-tree (OOT) custom operator layers and wrappers for Multi-Head Latent Attention (MLA)."""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch.nn import Parameter
from torch_tpu._internal import pallas, sync
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention.attention import \
    get_attention_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (rotate_gptj,
                                                                rotate_neox)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import \
    DeepseekScalingRotaryEmbedding
from vllm.v1.attention.backend import AttentionType

from vllm_torchtpu.layers.vllm.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import Indexer, DeepseekV32IndexerCache
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
)
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm_torchtpu.kernels.lightning_indexer.streamindex_topk import streamindex_topk

# streamindex_topk block-tuning knobs; bkv pages/block must keep
# page_size * bkv_p a multiple of 128 (TPU DMA constraint).
STREAMIDX_NUM_KV_PAGES_PER_BLOCK = 8
STREAMIDX_NUM_QUERIES_PER_BLOCK = 32

def _per_token_group_quant_fp8_tpu(
    x: torch.Tensor, group_size: int = 128, eps: float = 1e-10, use_ue8m0: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    x_grouped = x.reshape(-1, group_size)
    amax = torch.clamp(x_grouped.abs().max(dim=-1, keepdim=True).values, min=eps)
    fp8_max = 448.0  # float8_e4m3fn max value
    if use_ue8m0:
        # ue8m0: scale is a power of two, stored as its e8m0 bit pattern
        # (biased exponent, 1 byte/group) so it can be packed next to the
        # fp8 values and bitcast to float8_e8m0fnu inside the TPU kernel.
        exp = torch.ceil(torch.log2(amax / fp8_max))
        scale = torch.exp2(exp)
        scale_bits = (exp + 127).clamp(0, 254).to(torch.uint8)
    else:
        scale = amax / fp8_max
        scale_bits = None
    x_q = (x_grouped / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    out_scale = scale_bits if scale_bits is not None else scale
    out_scale = out_scale.reshape(orig_shape[:-1] + (orig_shape[-1] // group_size,))
    return x_q.reshape(orig_shape), out_scale


@SparseAttnIndexer.register_oot
class VllmTPUSparseAttnIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build eagerly: pallas.jax_op runs inspect.signature(), which Dynamo
        # cannot trace if the op is first built inside a compiled forward.
        self._get_streamidx_op()

    def _get_streamidx_op(self):
        """Build (once) the torch-callable wrapper around streamindex_topk."""
        op = getattr(self, "_streamidx_op", None)
        if op is not None:
            return op

        topk = self.topk_tokens

        def _streamidx_topk_jax(
            q_bytes: jax.Array,  # uint8 view of fp8 q [num_tokens, H, D]
            weights: jax.Array,  # [num_tokens, H]
            cache_kv: jax.Array,  # uint8 [num_blocks, block_size, D + 1]
            k_packed: jax.Array,  # uint8 [num_tokens, D + 1] fp8 k + e8m0 scale
            seq_lens: jax.Array,  # i32 [num_seqs]
            page_indices: jax.Array,  # i32 [num_seqs * pages_per_seq]
            cu_q_lens: jax.Array,  # i32 [num_seqs + 1]
            distribution: jax.Array,  # i32 [3]
        ) -> tuple[jax.Array, jax.Array]:
            num_blocks, block_size, entry_size = cache_kv.shape

            # Fused insert (TPU metadata has no slot_mapping): same position
            # formula and OOB-drop contract as _update_kv_cache_slots.
            tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
            seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                                      side="right").astype(jnp.int32)
            q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
            pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
            page = jnp.where(tok < cu_q_lens[-1],
                             page_indices.reshape(seq_lens.shape[0],
                                                  -1)[seq_id, pos // block_size],
                             jnp.int32(2**30))
            cache_kv = cache_kv.at[page, pos % block_size].set(k_packed)

            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
            kv_packing = 4  # uint8: 32 bits / 8
            cache_view = cache_kv.reshape(
                num_blocks, block_size // kv_packing, kv_packing, entry_size
            )
            topk_indices = streamindex_topk(
                q,
                weights,
                cache_view,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k=topk,
                compression_ratio=1,
                num_kv_pages_per_block=STREAMIDX_NUM_KV_PAGES_PER_BLOCK,
                num_queries_per_block=STREAMIDX_NUM_QUERIES_PER_BLOCK,
            )
            return cache_kv, topk_indices

        op_name = f"pallas::streamidx_topk_{self.k_cache.prefix.replace('.', '_')}"
        jax_op = pallas.jax_op(op_name, _streamidx_topk_jax,
                               donate_argnums=(2, ))

        def _fake(q_bytes, weights, cache_kv, *args, **kwargs):
            return torch.empty_like(cache_kv), torch.empty(
                (q_bytes.shape[0], topk), dtype=torch.int32, device=q_bytes.device
            )

        jax_op.register_fake(_fake)

        def op(*args):
            new_cache, topk_indices = jax_op(*args)
            args[2].copy_(new_cache)
            return topk_indices

        self._streamidx_op = op
        return op

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None

        k_quant, k_scale = _per_token_group_quant_fp8_tpu(
            k, self.quant_block_size, use_ue8m0=True
        )
        # Pack per-token fp8 values + 1-byte e8m0 scale into the uint8 row
        # layout the TPU kernel unpacks (head_dim value bytes + 1 scale byte).
        # cat requires a single dtype, so view the fp8 values as raw bytes.
        k_quant_packed = torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)

        attn_metadata = get_forward_context().attn_metadata
        metadata = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata.get(self.k_cache.prefix)
        if not isinstance(metadata, AttentionMetadata):
            return self.topk_indices_buffer

        kv_cache = self.k_cache.kv_cache  # uint8 [num_blocks, block_size, head_dim + 1]
        if kv_cache.numel() == 0:
            # Profiling run: the runner has not bound the caches yet.
            return self.topk_indices_buffer

        assert q_values.dtype == torch.float8_e4m3fn, (
            f"streamindex_topk expects fp8 q, got {q_values.dtype}"
        )
        # TPU metadata already carries the kernel's flat per-sequence view.
        seq_lens = metadata.seq_lens
        page_indices = metadata.block_tables
        cu_q_lens = metadata.query_start_loc
        # TPU request_distribution is (decode_end, prefill_end, num_seqs); the
        # kernel wants (decode_end, decode_end, num_seqs) so every non-decode
        # sequence routes through its MIXED case.
        rd = metadata.request_distribution
        distribution = torch.stack([rd[0], rd[0], rd[2]])

        topk_indices = self._get_streamidx_op()(
            q_values.view(torch.uint8),
            weights,
            kv_cache,
            k_quant_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

        self.topk_indices_buffer[: hidden_states.shape[0]] = -1
        num_out = min(topk_indices.shape[0], self.topk_indices_buffer.shape[0])
        self.topk_indices_buffer[:num_out, : self.topk_tokens] = topk_indices[
            :num_out
        ]
        return self.topk_indices_buffer

    # Reference path is also the native path (used when the custom op is
    # disabled in CompilationConfig).
    forward_native = forward_oot

class VllmTPUIndexer(Indexer):
    """TPU-optimized out-of-tree wrapper for Indexer."""
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
        is_inplace_rope: bool = False,
    ):
        # Skip Indexer.__init__ (it builds the unfused projections); init nn.Module directly.
        super(Indexer, self).__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"

        # streamindex_topk kernel limitaiton quant_block_size must be same with head_dim
        self.quant_block_size = self.head_dim  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp8 since we use ue8m0 format, so we need to store 1 extra scale value
        #       per self.quant_block_size element
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 1,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = VllmTPUSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

        self.is_inplace_rope = is_inplace_rope

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = _per_token_group_quant_fp8_tpu(
            q,
            self.quant_block_size,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)
        return self.indexer_op(hidden_states, q_fp8, k, weights)

class VllmTPUMLAAttention(MLAAttention):
    """TPU-optimized out-of-tree wrapper for Multi-Head Latent Attention."""

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: Any | None = None,
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        torch.nn.Module.__init__(self)
        super().__init__(num_heads, scale, qk_nope_head_dim, qk_rope_head_dim,
                         v_head_dim, q_lora_rank, kv_lora_rank, kv_b_proj,
                         cache_config, quant_config, prefix, attn_backend,
                         use_sparse, indexer, **extra_impl_args)

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                self.kv_cache_dtype.lower().strip())

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)

        device = torch.device("tpu")

        if self.kv_cache_quantized_dtype is not None:
            from vllm_torchtpu.layers.common.quantization import \
                quantize_tensor
            W_UK_T, W_UK_T_scale = quantize_tensor(
                self.W_UK_T, self.kv_cache_quantized_dtype, axis=1)
            self.W_UK_T = Parameter(W_UK_T.to(device), requires_grad=False)
            self.W_UK_T_scale = Parameter(W_UK_T_scale.to(device),
                                          requires_grad=False)

            W_UV, W_UV_scale = quantize_tensor(self.W_UV,
                                               self.kv_cache_quantized_dtype,
                                               axis=1)
            self.W_UV = Parameter(W_UV.to(device), requires_grad=False)
            self.W_UV_scale = Parameter(W_UV_scale.to(device),
                                        requires_grad=False)
        else:
            self.W_UK_T = Parameter(self.W_UK_T.to(device),
                                    requires_grad=False)
            self.W_UV = Parameter(self.W_UV.to(device), requires_grad=False)

        # Safely detach and clear kv_b_proj parameter buffers without breaking PyTorch attribute integrity
        kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
        for key in kv_b_proj_params.keys():
            if key in self.kv_b_proj._parameters:
                self.kv_b_proj._parameters[key] = None
            elif hasattr(self.kv_b_proj, key):
                delattr(self.kv_b_proj, key)

        if self.W_UK_T.device.type == "tpu":
            sync.synchronize(self.W_UK_T, wait=True)
            if hasattr(self, "W_UK_T_scale"):
                sync.synchronize(self.W_UK_T_scale, wait=True)
            sync.synchronize(self.W_UV, wait=True)
            if hasattr(self, "W_UV_scale"):
                sync.synchronize(self.W_UV_scale, wait=True)

        q_scale, k_scale, v_scale = self.impl._get_kv_scales(self)
        self.mla_op = self.impl._build_mla_op(self,
                                              q_scale=q_scale,
                                              k_scale=k_scale,
                                              v_scale=v_scale)
        self.sparse_mla_op = self.impl._build_sparse_mla_op(self)

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                output: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if getattr(self, "calculate_kv_scales", False):
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe,
                                                self.layer_name)

        attn_metadata, _, kv_cache, _ = get_attention_context(self.layer_name)

        return self.impl.forward(
            layer=self,
            q=q,
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            topk_indices=kwargs.get("topk_indices"),
        )


# Backward compatibility alias right in case legacy external imports reference old name
VllmMLAAttention = VllmTPUMLAAttention


@MultiHeadLatentAttentionWrapper.register_oot
class VllmTPUMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.skip_topk = skip_topk

        # skip_topk layers (GLM-5.2 index sharing) still need topk_tokens and
        # the shared buffer: they don't run their indexer but consume the
        # indices a previous layer wrote into the model-wide buffer.
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = VllmTPUMLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(super(), "process_weights_after_loading"):
            super().process_weights_after_loading(act_dtype)
        if self.rotary_emb is not None and hasattr(self.rotary_emb,
                                                   "cos_sin_cache"):
            device = torch.device("tpu")
            self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(
                device)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None")
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None")
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None")

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None")
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None")
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        topk_indices = None
        if self.indexer and self.is_sparse:
            # Non-skip layers run the indexer, which writes into the
            # model-wide topk_indices_buffer; skip_topk layers (GLM-5.2
            # index sharing) reuse whatever the previous non-skip layer
            # wrote there in this pass. Either way the sparse path consumes
            # the shared buffer, sliced to this batch (a copy-free view).
            # Rows are valid-first with a -1 tail, as the sparse MLA kernel
            # requires.
            if not self.skip_topk:
                self.indexer(hidden_states, q_c, positions,
                             self.indexer_rope_emb)
            assert self.topk_indices_buffer is not None, (
                "sparse MLA layers require the model-wide topk_indices_buffer"
            )
            topk_indices = self.topk_indices_buffer[
                : hidden_states.shape[0], : self.topk_tokens]

        if llama_4_scaling is not None:
            q_nope *= llama_4_scaling
            q_pe *= llama_4_scaling

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_heads * self.v_head_dim),
            topk_indices=topk_indices,
        )

        return self.o_proj(attn_out)[0]


VllmMultiHeadLatentAttentionWrapper = VllmTPUMultiHeadLatentAttentionWrapper


@DeepseekScalingRotaryEmbedding.register_oot
class VllmDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """TPU-compatible DeepSeek Scaling RoPE implementation."""

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert key is not None
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        pos = torch.add(positions,
                        offsets) if offsets is not None else positions
        cos_sin = self.cos_sin_cache.index_select(0, pos)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
