# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the v6d TPU DMA handler."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.offload.v6d.handler import (TpuRaidenOffloadingHandler,
                                               _load_kv_cache_offloader)


@pytest.fixture
def native_offloader():
    offloader = MagicMock()
    offloader_cls = MagicMock(return_value=offloader)
    return offloader_cls, offloader


def test_torch_tpu_environment_exposes_real_kv_cache_offloader():
    offloader_cls = _load_kv_cache_offloader()

    assert offloader_cls.__module__ == (
        "tpu_sync.api.torch.kv_cache_offloader")
    assert callable(offloader_cls.map_shared_memory)
    assert callable(offloader_cls.unmap_shared_memory)
    assert callable(offloader_cls.h2d)
    assert callable(offloader_cls.d2h)


def _registered_handler(native_offloader, **handler_kwargs):
    offloader_cls, offloader = native_offloader
    tensors = [object(), object()]
    handler = TpuRaidenOffloadingHandler(rank_id=3, **handler_kwargs)
    with patch("vllm_torchtpu.offload.v6d.handler._load_kv_cache_offloader",
               return_value=offloader_cls):
        handler.register_device(tensors, page_nbytes=4096)
    return handler, offloader, tensors


class _FakeTransferFuture:

    def __init__(self, *, ready_after: int, error_message: str = "") -> None:
        self._ready_after = ready_after
        self._error_message = error_message
        self.poll_count = 0

    def is_ready(self) -> bool:
        self.poll_count += 1
        return self.poll_count >= self._ready_after

    def ok(self) -> bool:
        return not self._error_message

    def error_message(self) -> str:
        return self._error_message

    def wait(self) -> None:
        raise AssertionError("async transfer must not call native wait()")


def test_register_device_constructs_native_offloader_once(native_offloader):
    offloader_cls, _ = native_offloader
    tensors = [object(), object()]
    handler = TpuRaidenOffloadingHandler(rank_id=3)

    with patch("vllm_torchtpu.offload.v6d.handler._load_kv_cache_offloader",
               return_value=offloader_cls):
        handler.register_device(tensors, page_nbytes=4096)
        with pytest.raises(RuntimeError, match="already registered"):
            handler.register_device(tensors, page_nbytes=4096)

    offloader_cls.assert_called_once_with(tensors, 4096)


def test_failed_device_registration_can_be_retried(native_offloader):
    offloader_cls, offloader = native_offloader
    offloader_cls.side_effect = [
        RuntimeError("native construction failed"), offloader
    ]
    tensors = [object()]
    handler = TpuRaidenOffloadingHandler(rank_id=3)

    with patch("vllm_torchtpu.offload.v6d.handler._load_kv_cache_offloader",
               return_value=offloader_cls):
        with pytest.raises(RuntimeError, match="native construction failed"):
            handler.register_device(tensors, page_nbytes=4096)
        handler.register_device(tensors, page_nbytes=4096)

    assert offloader_cls.call_count == 2


@pytest.mark.parametrize(
    "method,args",
    [
        ("map_shared_memory", (17, 8192)),
        ("unmap_shared_memory", ()),
        ("transfer", ([object()], [1], True)),
    ],
)
def test_operations_require_device_registration(method, args):
    handler = TpuRaidenOffloadingHandler(rank_id=0)
    with pytest.raises(RuntimeError,
                       match="device KV cache is not registered"):
        getattr(handler, method)(*args)


def test_map_and_unmap_forward_to_native_offloader(native_offloader):
    handler, offloader, _ = _registered_handler(native_offloader)
    mapped_address = 0x40000000
    handler.map_shared_memory(mapped_address=mapped_address,
                              pool_size_bytes=8192)
    handler.unmap_shared_memory()
    offloader.map_shared_memory.assert_called_once_with(mapped_address, 8192)
    offloader.unmap_shared_memory.assert_called_once_with()


def test_h2d_forwards_lists_and_returns_future_without_sync(native_offloader):
    handler, offloader, _ = _registered_handler(native_offloader)
    objects = [object(), object()]
    block_ids = [4, 9]
    future = object()
    offloader.h2d.return_value = future

    with patch("vllm_torchtpu.offload.v6d.handler._synchronize_device_tensors"
               ) as synchronize:
        result = handler.transfer(objects, block_ids, True)

    assert result is future
    synchronize.assert_not_called()
    offloader.h2d.assert_called_once_with(block_ids, objects, 3)
    forwarded_block_ids, forwarded_objects, _ = offloader.h2d.call_args.args
    assert forwarded_block_ids is block_ids
    assert forwarded_objects is objects
    offloader.d2h.assert_not_called()


def test_d2h_synchronizes_then_forwards_and_returns_future(native_offloader):
    handler, offloader, tensors = _registered_handler(native_offloader)
    objects = [object(), object()]
    block_ids = [4, 9]
    future = object()
    calls = []

    with patch("vllm_torchtpu.offload.v6d.handler._synchronize_device_tensors",
               side_effect=lambda value: calls.append(("sync", value))):
        offloader.d2h.side_effect = (lambda *args: calls.append(
            ("d2h", args)) or future)
        result = handler.transfer(objects, block_ids, False)

    assert result is future
    assert calls == [("sync", tensors), ("d2h", (block_ids, objects, 3))]
    forwarded_block_ids, forwarded_objects, _ = offloader.d2h.call_args.args
    assert forwarded_block_ids is block_ids
    assert forwarded_objects is objects
    offloader.h2d.assert_not_called()


@pytest.mark.parametrize("direction", [0, 1, "in", None])
def test_transfer_rejects_non_bool_direction(native_offloader, direction):
    handler, _, _ = _registered_handler(native_offloader)
    with pytest.raises(TypeError, match="is_h2d must be a bool"):
        handler.transfer([object()], [1], direction)


def test_poll_returns_false_while_transfer_is_pending():
    future = MagicMock()
    future.is_ready.return_value = False

    assert TpuRaidenOffloadingHandler.poll(transfer_future=future) is False
    future.ok.assert_not_called()
    future.error_message.assert_not_called()


def test_poll_returns_true_when_transfer_succeeds():
    future = MagicMock()
    future.is_ready.return_value = True
    future.ok.return_value = True

    assert TpuRaidenOffloadingHandler.poll(future) is True
    future.error_message.assert_not_called()


def test_poll_raises_native_error_when_transfer_fails():
    future = MagicMock()
    future.is_ready.return_value = True
    future.ok.return_value = False
    future.error_message.return_value = "DMA failed"

    with pytest.raises(RuntimeError, match="DMA failed"):
        TpuRaidenOffloadingHandler.poll(future)


def test_transfer_async_submits_immediately_and_yields_until_success(
        native_offloader):

    async def run():
        handler, offloader, _ = _registered_handler(native_offloader)
        native_future = _FakeTransferFuture(ready_after=2)
        offloader.h2d.return_value = native_future
        objects = [object()]
        block_ids = [7]

        transfer_future = handler.transfer_async(objects, block_ids, True)

        assert isinstance(transfer_future, asyncio.Future)
        offloader.h2d.assert_called_once_with(block_ids, objects, 3)

        other_work = []
        asyncio.get_running_loop().call_soon(other_work.append, "ran")

        assert await transfer_future is None
        assert other_work == ["ran"]
        assert native_future.poll_count == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    ("poll_interval_ms", "expected_sleep_seconds"),
    [(None, 0.001), (2.5, 0.0025), (0.0, 0.0)],
)
def test_transfer_async_polls_at_configured_interval(native_offloader,
                                                     poll_interval_ms,
                                                     expected_sleep_seconds):

    async def run():
        handler_kwargs = ({} if poll_interval_ms is None else {
            "poll_interval_ms": poll_interval_ms
        })
        handler, offloader, _ = _registered_handler(native_offloader,
                                                    **handler_kwargs)
        native_future = _FakeTransferFuture(ready_after=2)
        offloader.h2d.return_value = native_future
        sleep_delays = []
        real_sleep = asyncio.sleep

        async def record_sleep(delay):
            sleep_delays.append(delay)
            await real_sleep(0)

        with patch("vllm_torchtpu.offload.v6d.handler.asyncio.sleep",
                   side_effect=record_sleep):
            assert await handler.transfer_async([object()], [7], True) is None

        assert sleep_delays == [expected_sleep_seconds]

    asyncio.run(run())


@pytest.mark.parametrize(
    "poll_interval_ms",
    [-0.1, float("nan"), float("inf"),
     float("-inf")])
def test_handler_rejects_invalid_poll_interval(poll_interval_ms):
    with pytest.raises(ValueError, match="poll_interval_ms"):
        TpuRaidenOffloadingHandler(rank_id=0,
                                   poll_interval_ms=poll_interval_ms)


def test_transfer_async_propagates_native_failure(native_offloader):

    async def run():
        handler, offloader, _ = _registered_handler(native_offloader)
        native_future = _FakeTransferFuture(ready_after=1,
                                            error_message="DMA failed")
        offloader.h2d.return_value = native_future

        transfer_future = handler.transfer_async([object()], [7], True)

        with pytest.raises(RuntimeError, match="DMA failed"):
            await transfer_future

    asyncio.run(run())


def test_transfer_async_cannot_finish_cancellation_before_dma(
        native_offloader):

    async def run():
        handler, offloader, _ = _registered_handler(native_offloader)
        native_future = _FakeTransferFuture(ready_after=2)
        offloader.h2d.return_value = native_future

        transfer_future = handler.transfer_async([object()], [7], True)

        assert transfer_future.cancel() is False
        assert not transfer_future.done()
        assert await transfer_future is None
        assert native_future.poll_count == 2

    asyncio.run(run())


def test_transfer_async_requires_running_loop_before_submitting(
        native_offloader):
    handler, offloader, _ = _registered_handler(native_offloader)

    with pytest.raises(RuntimeError, match="no running event loop"):
        handler.transfer_async([object()], [7], True)

    offloader.h2d.assert_not_called()
