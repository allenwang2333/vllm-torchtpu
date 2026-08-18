# SPDX-License-Identifier: Apache-2.0
"""PyTorch SDPA diffusion attention backend for TPU execution."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)


class TpuSDPABackend(AttentionBackend):
    """Diffusion attention backend for TPU using PyTorch SDPA."""

    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return list(range(1024))

    @staticmethod
    def get_name() -> str:
        return "TPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TpuSDPAImpl]:
        return TpuSDPAImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None


class TpuSDPAImpl(AttentionImpl):
    """TPU attention implementation wrapping PyTorch F.scaled_dot_product_attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args: Any,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads or num_heads
        self.prefix = prefix
        self.backend_kwargs = backend_kwargs or {}

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        """Run attention using PyTorch SDPA on TPU.

        Input shapes: (batch_size, seq_len, num_heads, head_size)
        Output shape: (batch_size, seq_len, num_heads, head_size)
        """
        attention_mask = None
        if attn_metadata is not None and getattr(attn_metadata, "attn_mask", None) is not None:
            mask = attn_metadata.attn_mask
            if mask.ndim == 2 and mask.shape[0] == query.shape[0] and mask.shape[1] == key.shape[1]:
                attention_mask = mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
            else:
                attention_mask = mask

        # (B, S, H, D) -> (B, H, S, D)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    f"GQA requires query heads ({query.shape[1]}) to be divisible by "
                    f"KV heads ({key.shape[1]})."
                )
            repeat_num = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_num, dim=1)
            value = value.repeat_interleave(repeat_num, dim=1)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )

        # (B, H, S, D) -> (B, S, H, D)
        return output.permute(0, 2, 1, 3)
