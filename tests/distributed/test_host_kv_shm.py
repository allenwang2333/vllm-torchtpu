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

from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import (HostKVShmPool,
                                                               PoolSpec)


def _pool_spec(block_major: bool = False) -> PoolSpec:
    return PoolSpec(
        num_slots=1,
        tp_size=2,
        num_layers=3,
        max_blocks=4,
        layer_shard_shape=(4, 2),
        dtype=torch.uint8,
        block_major=block_major,
    )


def test_unpack_rank_layers_copies_used_bytes_and_preserves_padding():
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        pool._shm.buf[:] = b"\xee" * spec.total_bytes
        num_blocks = 2
        used = pool._used_per_layer(num_blocks)
        payloads = [
            memoryview(bytes([layer + 1]) * used)
            for layer in range(spec.num_layers)
        ]

        pool.unpack_rank_layers(slot_idx=0,
                                rank=1,
                                num_blocks=num_blocks,
                                layer_buffers=payloads)

        rank_start = spec.per_rank_bytes
        for layer in range(spec.num_layers):
            layer_off = rank_start + layer * spec.per_layer_bytes
            assert bytes(pool._shm.buf[layer_off:layer_off +
                                       used]) == (bytes([layer + 1]) * used)
            assert bytes(
                pool._shm.buf[layer_off + used:layer_off +
                              spec.per_layer_bytes]) == (
                                  b"\xee" * (spec.per_layer_bytes - used))
    finally:
        pool.close()


def test_unpack_rank_layers_rejects_wrong_layer_size():
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        with pytest.raises(RuntimeError, match="layer 0 size"):
            pool.unpack_rank_layers(slot_idx=0,
                                    rank=0,
                                    num_blocks=2,
                                    layer_buffers=[b"x"] * spec.num_layers)
    finally:
        pool.close()


def test_bundle_view_at_max_blocks():
    """bundle_view returns a tensor that spans the per-rank region with
    shape `[num_layers, *layer_shard_shape]`. Writing through the view
    propagates to the underlying shm bytes (sanity for the would-be
    bundled DMA dst). Only valid at num_blocks == max_blocks because
    per-layer regions are otherwise padded apart."""
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        view = pool.bundle_view(slot_idx=0, rank=1, num_blocks=spec.max_blocks)
        assert tuple(
            view.shape) == (spec.num_layers, ) + spec.layer_shard_shape
        # Fill via the view; verify the corresponding rank region holds
        # the data in the expected layer-by-layer layout.
        for layer in range(spec.num_layers):
            view[layer].fill_(layer + 1)
        rank_start = spec.per_rank_bytes  # rank=1 offset
        for layer in range(spec.num_layers):
            layer_off = rank_start + layer * spec.per_layer_bytes
            assert bytes(pool._shm.buf[layer_off:layer_off +
                                       spec.per_layer_bytes]) == bytes(
                                           [layer + 1] * spec.per_layer_bytes)
    finally:
        pool.close()


def test_bundle_view_rejects_partial_num_blocks():
    """bundle_view at num_blocks < max_blocks is not a valid contiguous
    view in layer-major mode (per-layer slots are padded apart). Verify
    the helper rejects."""
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        with pytest.raises(ValueError, match="bundle_view requires"):
            pool.bundle_view(slot_idx=0, rank=0, num_blocks=2)
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Block-major layout
# ---------------------------------------------------------------------------


def test_block_major_bundle_view_works_at_partial_num_blocks():
    """In block-major mode the per-rank region is laid out
    [max_blocks, num_layers, *trailing] contiguously, so bundle_view is
    valid for any num_blocks ≤ max_blocks (slicing on dim 0 keeps
    contiguity)."""
    spec = _pool_spec(block_major=True)
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        view = pool.bundle_view(slot_idx=0, rank=1, num_blocks=2)
        assert tuple(
            view.shape) == (2, spec.num_layers) + spec.layer_shard_shape[1:]
        # Fill block 0 layer 1 distinctly from block 1 layer 0.
        view[0, 1].fill_(0xAB)
        view[1, 0].fill_(0xCD)
        # Verify the bytes show up at the expected offsets: in block-major
        # layout, rank region is [max_blocks, num_layers, *trailing], so
        # block i layer l starts at (i * num_layers + l) * trailing_bytes.
        rank_start = spec.per_rank_bytes  # rank=1
        trailing_bytes = spec._trailing_bytes
        off_b0_l1 = rank_start + (0 * spec.num_layers + 1) * trailing_bytes
        off_b1_l0 = rank_start + (1 * spec.num_layers + 0) * trailing_bytes
        assert bytes(pool._shm.buf[off_b0_l1:off_b0_l1 +
                                   trailing_bytes]) == b"\xab" * trailing_bytes
        assert bytes(pool._shm.buf[off_b1_l0:off_b1_l0 +
                                   trailing_bytes]) == b"\xcd" * trailing_bytes
    finally:
        pool.close()


def test_block_major_rank_bundle_view_and_unpack_roundtrip():
    """rank_bundle_view returns a contiguous memoryview suitable as a
    ZMQ frame; unpack_rank_bundle is its inverse on the consumer side."""
    spec = _pool_spec(block_major=True)
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        num_blocks = 3
        used = pool._used_per_rank_bundle(num_blocks)
        payload = bytes((i & 0xFF) for i in range(used))
        pool.unpack_rank_bundle(slot_idx=0,
                                rank=0,
                                num_blocks=num_blocks,
                                buf=payload)
        view = pool.rank_bundle_view(slot_idx=0, rank=0, num_blocks=num_blocks)
        assert view.nbytes == used
        assert bytes(view) == payload
        # Release the memoryview before closing shm, otherwise the
        # mmap.close() call inside pool.close() raises BufferError.
        view.release()
    finally:
        pool.close()


def test_block_major_layer_view_is_strided_into_bundle():
    """layer_view in block-major mode returns a strided slice of the
    per-rank bundle region. Writes through it propagate to the
    underlying bytes at the bundle-strided offsets."""
    spec = _pool_spec(block_major=True)
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        num_blocks = spec.max_blocks
        layer_idx = 1
        view = pool.layer_view(slot_idx=0,
                               rank=0,
                               layer_idx=layer_idx,
                               num_blocks=num_blocks)
        assert tuple(view.shape) == (num_blocks, ) + spec.layer_shard_shape[1:]
        view.fill_(0x77)
        # Layer 1's data should be at offsets
        # (i * num_layers + 1) * trailing for i in [0, num_blocks).
        trailing_bytes = spec._trailing_bytes
        for i in range(num_blocks):
            off = (i * spec.num_layers + layer_idx) * trailing_bytes
            assert bytes(
                pool._shm.buf[off:off +
                              trailing_bytes]) == b"\x77" * trailing_bytes
        # Layer 0/2 untouched.
        for i in range(num_blocks):
            for other in (0, 2):
                off = (i * spec.num_layers + other) * trailing_bytes
                assert bytes(
                    pool._shm.buf[off:off +
                                  trailing_bytes]) == b"\x00" * trailing_bytes
    finally:
        pool.close()


def test_block_major_rejects_layer_views_and_layer_unpack():
    """rank_layer_views / unpack_rank_layers are layer-major-only
    helpers; calling them on a block-major pool should raise."""
    spec = _pool_spec(block_major=True)
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        with pytest.raises(RuntimeError, match="rank_layer_views"):
            pool.rank_layer_views(slot_idx=0, rank=0, num_blocks=2)
        with pytest.raises(RuntimeError, match="unpack_rank_layers"):
            pool.unpack_rank_layers(slot_idx=0,
                                    rank=0,
                                    num_blocks=2,
                                    layer_buffers=[b"x"] * spec.num_layers)
    finally:
        pool.close()


def test_layer_major_rejects_bundle_helpers():
    """rank_bundle_view / unpack_rank_bundle are block-major-only
    helpers; calling them on a layer-major pool should raise."""
    spec = _pool_spec(block_major=False)
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        with pytest.raises(RuntimeError, match="rank_bundle_view"):
            pool.rank_bundle_view(slot_idx=0, rank=0, num_blocks=2)
        with pytest.raises(RuntimeError, match="unpack_rank_bundle"):
            pool.unpack_rank_bundle(slot_idx=0, rank=0, num_blocks=2, buf=b"x")
    finally:
        pool.close()
