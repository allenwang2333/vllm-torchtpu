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
"""GLM-5.3-Flash full-attention (DSA) layers on TPU.

GLM-5.3-Flash's 11 non-KDA layers are MLA with one property this repo's MLA
stack has never been asked for: **NoPE only**. ``qk_rope_head_dim == 0`` --
there is no rotary component at all, and position information reaches these
layers through the KDA recurrence and the short convolutions instead.

The RoPE half is handled by *not* doing it, which is exact rather than
approximate. What the kernel needs is a non-degenerate rotary dimension, so it
gets a slab of zeros; see :data:`_ZERO_ROPE_SLAB` for why zero width is
structurally impossible and why a zero-valued slab is numerically free.

These layers run as dense causal MLA, which is exact only while
``max_model_len <= index_topk``; ``model._check_context_within_dense_mla_bound``
refuses longer contexts rather than producing fluent nonsense.
"""

import re

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)

from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import \
    VllmTPUMultiHeadLatentAttentionWrapper
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Width of the all-zeros rotary slab handed to the MLA kernel.
#
# The dense MLA v2 kernel cannot take ``r_dim = 0``. It is parametric in
# multiples of 128, but zero is structurally impossible, not merely untested:
# ``kernels/mla/v2/kernel.py:2408`` and ``:2435`` allocate
# ``pltpu.VMEM((..., r_dim))`` scratch that Mosaic cannot size at zero lanes,
# the prologue at ``:2001`` does ``reshape((2, -1, r_dim))`` which is
# unsolvable when ``r_dim`` is 0, several KV DMAs would become zero-byte copies
# sharing a semaphore with a live one, and ``:562`` contracts a matmul over
# ``d == 0``.
#
# Feeding it a 128-wide slab of zeros instead is numerically exact:
# ``q_pe . k_pe == 0``, so the rotary contribution to the logits at
# ``kernel.py:568`` (``s = s_c + s_pe``) is identically zero, and the value
# path never reads the rotary columns. The cost is a wider KV cache
# (``512 -> 640`` in the minor dim, +25%) and ~25% more QK MXU work; the PV
# matmul that dominates decode is untouched.
#
# The alternative -- teaching the kernel ``r_dim = 0`` -- means editing at
# least five sites inside a 2600-line Pallas kernel where the failure mode is a
# DMA-semaphore hang rather than an error. That belongs in the performance
# phase, not bringup.
_ZERO_ROPE_SLAB = 128


def mla_head_size(kv_lora_rank: int) -> int:
    """``MLAAttention.head_size`` for this layer.

    Spelled out because it is the number every KV-cache spec, page-size
    calculation and OOM message carries, and it is *not* the same quantity as
    the rope slab width that produces it.
    """
    return int(kv_lora_rank) + _ZERO_ROPE_SLAB


def layer_index_from_prefix(prefix: str) -> int:
    """``model.layers.<i>.self_attn`` -> ``i``, or ``-1``.

    Mirrors ``Glm5NextDecoderLayer.__init__``'s regex rather than taking the
    index as an argument, so a layer cannot be told it is one index while its
    weights load from another.
    """
    match = re.search(r"layers\.(\d+)", prefix or "")
    return int(match.group(1)) if match else -1


class Glm5NextMLAAttention(VllmTPUMultiHeadLatentAttentionWrapper):
    """MLA for GLM-5.3-Flash's full-attention layers.

    Built with ``qk_rope_head_dim`` set to the zero-slab width so the KV cache
    spec and the Pallas kernel see ordinary DeepSeek geometry, while
    ``forward`` is overridden so nothing tries to *split* a rotary component
    out of projections that do not produce one.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        # Named `zero_rope_slab`, not `_ZERO_ROPE_SLAB`, so `forward` reads
        # the width off the instance and the tensors it synthesises cannot
        # disagree with the KV-cache spec.
        self.zero_rope_slab = _ZERO_ROPE_SLAB

        from vllm.distributed import get_tensor_model_parallel_world_size

        hidden_size = text_config.hidden_size
        num_heads = text_config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must divide the tensor "
                f"parallel size ({tp_size})")
        # The projections below are built with the *global* head count --
        # Column/RowParallelLinear shard them internally -- but the wrapper and
        # everything downstream of it work per rank, so it gets the local
        # count. vLLM's own MLA callers do the same
        # (`deepseek_v2.py:1206` passes `num_local_heads`). Getting this wrong
        # is invisible at TP=1 and a shape error above it.
        num_local_heads = num_heads // tp_size
        self.actual_qk_rope_head_dim = text_config.qk_rope_head_dim
        if self.actual_qk_rope_head_dim != 0:
            raise NotImplementedError(
                "GLM-5.3-Flash is expected to be NoPE-only "
                f"(qk_rope_head_dim=0); got {self.actual_qk_rope_head_dim}. "
                "A real rotary component needs the stock MLA path, not this "
                "zero-slab one.")

        qk_nope_head_dim = text_config.qk_nope_head_dim
        v_head_dim = text_config.v_head_dim
        q_lora_rank = text_config.q_lora_rank
        kv_lora_rank = text_config.kv_lora_rank
        eps = text_config.rms_norm_eps

        # Scale over the real (NoPE-only) query width. The zero slab must not
        # enter the softmax scale -- it carries no information.
        scale = qk_nope_head_dim**-0.5

        # One projection for q_a and kv_a, as DeepSeek-V4 does; note the kv
        # half is `kv_lora_rank` alone, with no rope tail.
        fused_qkv_a_proj = MergedColumnParallelLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        q_a_layernorm = RMSNorm(q_lora_rank, eps=eps)
        q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * qk_nope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        kv_a_layernorm = RMSNorm(kv_lora_rank, eps=eps)
        kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        from vllm.model_executor.layers.mla import MLAModules

        # No indexer. These layers run as dense causal MLA, which is
        # *exact* for `max_model_len <= index_topk` (2048): a query at
        # position p has floor((p+1)/index_kpool) candidate pools against a
        # budget of index_topk // index_kpool == 512, so a top-k over pools
        # would select every valid pool, expansion would cover
        # [0, 4*floor((p+1)/4) - 1], and the always-selected tail would supply
        # the remaining (p + 1) % 4. The union is exactly [0, p], which is
        # what dense causal attention computes anyway.
        #
        # Past that bound it is wrong, and silently so -- plausible tokens,
        # wrong attention -- which is what
        # `model._check_context_within_dense_mla_bound` refuses.
        #
        # `is_sparse=False` is a CONSTRUCTION-TIME choice, not a runtime one:
        # `tpu_runner` allocates either one MLA cache or the split
        # (nope, rope) pair from `attn_module.use_sparse`, and once the caches
        # are a pair the dense branch at `layers/vllm/attention.py:1677` takes
        # a single tensor -- so nothing can fall back later.

        mla_modules = MLAModules(
            fused_qkv_a_proj=fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=q_a_layernorm,
            q_b_proj=q_b_proj,
            q_proj=None,
            kv_a_layernorm=kv_a_layernorm,
            kv_b_proj=kv_b_proj,
            rotary_emb=None,
            o_proj=o_proj,
            indexer=None,
            indexer_rotary_emb=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_local_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=self.zero_rope_slab,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Like the base forward, but the rotary halves are synthesised zeros.

        The base implementation splits ``q`` and ``kv_lora`` into NoPE and RoPE
        parts. GLM's projections have no RoPE part to split off, so the split
        widths and the synthesised slab are kept separate here.

        ``positions`` is unused: nothing here rotates anything.
        """
        del positions, llama_4_scaling  # Neither is a NoPE dense-MLA input.

        num_tokens = hidden_states.shape[0]

        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank],
                                   dim=-1)

        q_c = self.q_a_layernorm(q_c)
        q_nope = self.q_b_proj(q_c)[0].view(-1, self.num_heads,
                                            self.qk_nope_head_dim)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        # The zero slab. `q_pe @ k_pe^T` is exactly 0, so it contributes
        # nothing to the attention logits; it exists only to give the kernel a
        # non-degenerate rotary dimension. Read off the wrapper rather than
        # off `_ZERO_ROPE_SLAB` so the tensors and the KV-cache spec cannot
        # disagree.
        slab = self.qk_rope_head_dim
        zeros = hidden_states.new_zeros(())
        q_pe = zeros.expand(num_tokens, self.num_heads, slab).contiguous()
        k_pe = zeros.expand(num_tokens, 1, slab).contiguous()

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(num_tokens, self.num_heads * self.v_head_dim),
            topk_indices=None,
        )
        return self.o_proj(attn_out)[0]
