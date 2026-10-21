# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental fixed-buffer upload for packed host Engram rows."""

import ctypes

import torch
from torch_tpu._internal.compile import tpu_torch_compile
from tpu_sync.api.torch import raw_transfer


class PreparedEngramUpload:
    """Own a validated raw host/device pair for one packed token bucket.

    The device tensor must never receive a torch mutation after construction.
    ``copy_from`` fences previous consumers before changing its physical bytes.
    This prototype uses a device-wide fence until a consumer event is available.
    """

    def __init__(self, num_words: int, device: torch.device) -> None:
        probe = torch.arange(num_words, dtype=torch.int32, device="cpu")
        probe.bitwise_xor_(0x13579BDF)
        self.device = probe.to(device)
        torch.accelerator.synchronize()
        self.layout = tpu_torch_compile.get_device_layout_if_materialized(self.device)
        if self.layout is None:
            raise RuntimeError("Engram raw upload requires a materialized layout")
        order, tiles, element_bits = self.layout
        if list(order) != [0] or element_bits not in (0, 32):
            raise NotImplementedError(f"Engram raw int32 layout: {self.layout}")
        capacity_words = num_words
        for tile in tiles:
            if len(tile) != 1 or tile[0] <= 0:
                raise NotImplementedError(f"Engram raw 1-D tiling: {self.layout}")
            capacity_words = -(-capacity_words // tile[0]) * tile[0]
        self.host_buffer = raw_transfer.RawHostBuffer(capacity_words * 4)
        self.transfer = raw_transfer.PreparedTorchRawTransfer(
            self.device, self.host_buffer, unsafe_skip_buffer_lock=False
        )
        self._host_owner = (ctypes.c_uint8 * self.host_buffer.size_bytes).from_address(
            self.host_buffer.data_ptr
        )
        self.host_bytes = torch.frombuffer(self._host_owner, dtype=torch.uint8)
        self.host_words = self.host_bytes[: num_words * 4].view(torch.int32)
        self.host_bytes.zero_()
        self.transfer.d2h()
        if not torch.equal(self.host_words, probe):
            raise RuntimeError(
                "Engram raw physical order differs from packed int32; "
                "use the copy_ upload path"
            )

    def copy_from(self, host_words: torch.Tensor) -> None:
        if host_words.shape != self.host_words.shape:
            raise ValueError("Engram raw upload must use its exact token bucket")
        torch.accelerator.synchronize()
        self.host_words.copy_(host_words)
        self.transfer.h2d()
