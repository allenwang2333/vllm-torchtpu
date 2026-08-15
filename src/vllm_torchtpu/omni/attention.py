# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

try:
    from vllm_omni.diffusion.attention.backends.sdpa import (  # type: ignore[missing-import]
        SDPABackend, SDPAImpl)
except ImportError:

    class SDPABackend:  # type: ignore
        pass

    class SDPAImpl:  # type: ignore
        pass


class TpuSDPAImpl(SDPAImpl):
    """TPU implementation of SDPA attention for diffusion models."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query,
                                  key,
                                  value,
                                  attn_metadata,
                                  mask_mode="broadcast_k")


class TpuSDPABackend(SDPABackend):
    """TPU attention backend for diffusion models using PyTorch SDPA."""

    @staticmethod
    def get_name() -> str:
        return "TPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TpuSDPAImpl]:
        return TpuSDPAImpl
