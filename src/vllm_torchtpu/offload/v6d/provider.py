# SPDX-License-Identifier: Apache-2.0
"""Wheel-discoverable TPU DMA provider for the external v6d integration."""

from typing import Any


def _load_handler():
    """Load the handler only after v6d selects this provider."""
    from .handler import TpuRaidenOffloadingHandler

    return TpuRaidenOffloadingHandler


class TpuDmaProvider:
    """Provider API v1 factory for rank-local TPU DMA handlers."""

    api_version = 1

    def create_handler(self, context: Any):
        return _load_handler()(rank_id=context.rank)


def get_provider() -> TpuDmaProvider:
    """Return a provider without loading the TPU DMA implementation."""
    return TpuDmaProvider()
