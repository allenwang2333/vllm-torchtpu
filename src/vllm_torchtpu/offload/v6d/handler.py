# SPDX-License-Identifier: Apache-2.0
"""TPU Raiden DMA handler consumed by the external v6d integration."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _load_kv_cache_offloader():
    """Load the native wrapper only when device tensors are registered."""
    from tpu_sync.api.torch.kv_cache_offloader import KVCacheOffloader

    return KVCacheOffloader


def _synchronize_device_tensors(tensors: list[torch.Tensor]) -> None:
    """Order a D2H read after vLLM's deferred writes to the KV tensors."""
    from vllm_torchtpu.utils import synchronize_tensors

    synchronize_tensors(tensors)


class _AsyncTransferFuture(asyncio.Future[None]):
    """Bridge a non-cancellable native DMA future into asyncio."""

    def __init__(self, loop: asyncio.AbstractEventLoop, transfer_future: Any,
                 poll_interval_s: float) -> None:
        super().__init__(loop=loop)
        self._poll_interval_s = poll_interval_s
        self._polling_task = loop.create_task(self._poll(transfer_future))

    async def _poll(self, transfer_future: Any) -> None:
        try:
            while not transfer_future.is_ready():
                await asyncio.sleep(self._poll_interval_s)
            if not transfer_future.ok():
                raise RuntimeError(transfer_future.error_message())
        except Exception as error:
            self.set_exception(error)
        else:
            self.set_result(None)

    def cancel(self, msg: Any = None) -> bool:
        """Reject cancellation because the underlying DMA cannot cancel."""
        return False


class TpuRaidenOffloadingHandler:
    """Thin adapter from v6d transfer requests to TPU Raiden DMA."""

    def __init__(self, rank_id: int, poll_interval_ms: float = 1.0) -> None:
        if poll_interval_ms < 0 or not math.isfinite(poll_interval_ms):
            raise ValueError(
                "poll_interval_ms must be finite and non-negative")
        self._rank_id = rank_id
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._device_tensors: list[torch.Tensor] | None = None
        self._offloader: Any | None = None

    def _require_offloader(self) -> Any:
        if self._offloader is None:
            raise RuntimeError("device KV cache is not registered")
        return self._offloader

    def register_device(
        self,
        kv_cache_tensors: list[torch.Tensor],
        page_nbytes: int,
    ) -> None:
        """Register the device KV cache geometry once for this handler."""
        if self._offloader is not None:
            raise RuntimeError("device KV cache is already registered")

        tensors = list(kv_cache_tensors)
        offloader = _load_kv_cache_offloader()(tensors, page_nbytes)
        self._device_tensors = tensors
        self._offloader = offloader

    def map_shared_memory(self, mapped_address: int,
                          pool_size_bytes: int) -> None:
        """DMA-map a shared-memory pool already mapped in this process."""
        self._require_offloader().map_shared_memory(mapped_address,
                                                    pool_size_bytes)

    def unmap_shared_memory(self) -> None:
        """Drain native copies and release the active DMA registration."""
        self._require_offloader().unmap_shared_memory()

    def transfer(
        self,
        object_tensors: list[torch.Tensor],
        gpu_block_ids: list[int],
        is_h2d: bool,
    ) -> Any:
        """Submit one asynchronous IN/H2D or OUT/D2H page transfer."""
        if not isinstance(is_h2d, bool):
            raise TypeError("is_h2d must be a bool")

        offloader = self._require_offloader()
        if is_h2d:
            return offloader.h2d(gpu_block_ids, object_tensors, self._rank_id)

        assert self._device_tensors is not None, (
            "registered offloader must retain its device tensor references")
        _synchronize_device_tensors(self._device_tensors)
        return offloader.d2h(gpu_block_ids, object_tensors, self._rank_id)

    def transfer_async(
        self,
        object_tensors: list[torch.Tensor],
        gpu_block_ids: list[int],
        is_h2d: bool,
    ) -> asyncio.Future[None]:
        """Submit one transfer and return a non-cancellable asyncio future."""
        loop = asyncio.get_running_loop()
        transfer_future = self.transfer(object_tensors, gpu_block_ids, is_h2d)
        return _AsyncTransferFuture(loop, transfer_future,
                                    self._poll_interval_s)

    @staticmethod
    def poll(transfer_future: Any) -> bool:
        """Poll one backend-private transfer future for completion."""
        if not transfer_future.is_ready():
            return False
        if not transfer_future.ok():
            raise RuntimeError(transfer_future.error_message())
        return True
