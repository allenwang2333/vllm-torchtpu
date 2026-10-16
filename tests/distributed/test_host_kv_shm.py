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
"""Tests for the shared-memory KV staging pool."""

import uuid

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import HostKVShmPool, PoolSpec


def _pool_spec() -> PoolSpec:
    return PoolSpec(
        num_slots=1,
        tp_size=2,
        num_layers=3,
        max_blocks=4,
        layer_shard_shape=(4, 2),
        dtype=torch.uint8,
    )


def test_unpack_rank_layers_copies_used_bytes_and_preserves_padding():
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        pool._shm.buf[:] = b"\xee" * spec.total_bytes
        num_blocks = 2
        used = pool._used_per_layer(num_blocks)
        payloads = [
            memoryview(bytes([layer + 1]) * used) for layer in range(spec.num_layers)
        ]

        pool.unpack_rank_layers(
            slot_idx=0, rank=1, num_blocks=num_blocks, layer_buffers=payloads
        )

        rank_start = spec.per_rank_bytes
        for layer in range(spec.num_layers):
            layer_off = rank_start + layer * spec.per_layer_bytes
            assert bytes(pool._shm.buf[layer_off : layer_off + used]) == (
                bytes([layer + 1]) * used
            )
            assert bytes(
                pool._shm.buf[layer_off + used : layer_off + spec.per_layer_bytes]
            ) == (b"\xee" * (spec.per_layer_bytes - used))
    finally:
        pool.close()


def test_unpack_rank_layers_rejects_wrong_layer_size():
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        with pytest.raises(RuntimeError, match="layer 0 size"):
            pool.unpack_rank_layers(
                slot_idx=0, rank=0, num_blocks=2, layer_buffers=[b"x"] * spec.num_layers
            )
    finally:
        pool.close()


def test_unpack_rank_layers_translates_fp8_to_bfloat16():
    spec = PoolSpec(
        num_slots=1,
        tp_size=1,
        num_layers=2,
        max_blocks=4,
        layer_shard_shape=(4, 16, 1, 2, 128),
        dtype=torch.bfloat16,
    )
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        num_blocks = 2
        src_phys = torch.zeros((num_blocks, 16, 1, 128, 4), dtype=torch.bfloat16)
        src_phys[..., 0] = 2.5
        src_phys[..., 1] = -0.0625
        src_fp8_bytes = src_phys.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()

        pool.unpack_rank_layers(
            slot_idx=0,
            rank=0,
            num_blocks=num_blocks,
            layer_buffers=[memoryview(src_fp8_bytes)] * spec.num_layers,
            src_dtype="torch.float8_e4m3fn",
            src_layer_shard_shape=(16, 1, 4, 128),
        )

        for li in range(spec.num_layers):
            view = pool.layer_view(0, 0, li, num_blocks).view(num_blocks, 16, 1, 128, 2)
            assert torch.all(view[..., 0] == 2.5)
            assert torch.all(view[..., 1] == -0.0625)
    finally:
        pool.close()
