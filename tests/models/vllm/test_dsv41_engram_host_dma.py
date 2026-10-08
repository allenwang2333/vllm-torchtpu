# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the pinned_host Engram row gather (``ENGRAM_HOST_DMA``).

The Pallas host DMA itself needs a TPU; here the kernel's tile fetch is
emulated in NumPy so the layout (``pack_rows``), the in-tile selection
(``select_rows``) and the embedding's device lookup are checked bitwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v41 import host_row_gather as hrg
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common import host_row_table


@pytest.fixture(autouse=True)
def setup_torch_accelerator(monkeypatch: pytest.MonkeyPatch):
    """Enable CPU execution for upstream Engram constructor checks."""
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: 0, raising=False
    )


def _fetch_tiles(packed: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """What the kernel DMAs: tile ``t`` = ``packed[t // 8, :, 128 * (t % 8):]``."""
    return np.stack(
        [
            packed[
                t // hrg.TILES_PER_MAJOR, :, (t % hrg.TILES_PER_MAJOR) * hrg.LANES :
            ][:, : hrg.LANES]
            for t in tiles
        ]
    )


@pytest.mark.parametrize("row_bytes", [32, 256, 512])
def test_pack_then_select_round_trips_every_row(row_bytes):
    rng = np.random.default_rng(row_bytes)
    per_major = hrg.TILES_PER_MAJOR * hrg.rows_per_tile(row_bytes)
    num_rows = 3 * per_major + 5  # partial last major
    rows = rng.integers(0, 256, (num_rows, row_bytes), dtype=np.uint8)
    packed = hrg.pack_rows(rows)
    assert packed.shape == (4, hrg.SUBLANES, hrg.MAJOR_LANES)
    assert packed.dtype == np.int32
    ids = np.arange(num_rows)
    per_tile = hrg.rows_per_tile(row_bytes)
    tiles = _fetch_tiles(packed, ids // per_tile)
    got = np.asarray(
        hrg.select_rows(jnp.asarray(tiles), jnp.asarray(ids % per_tile), row_bytes)
    )
    np.testing.assert_array_equal(got, rows.view(np.int32))
    # Tail rows beyond num_rows are zero.
    tail = _fetch_tiles(packed, [(num_rows + 1) // per_tile])
    assert not tail.reshape(-1)[-(row_bytes // 4) :].any()


def test_pack_rows_explicit_majors_pads_with_zeros():
    rows = np.full((10, 256), 7, dtype=np.uint8)
    packed = hrg.pack_rows(rows, majors=3)
    assert packed.shape == (3, hrg.SUBLANES, hrg.MAJOR_LANES)
    assert (packed != 0).sum() == 10 * 64
    with pytest.raises(AssertionError):
        hrg.pack_rows(np.zeros((129, 256), dtype=np.uint8), majors=1)


def test_row_width_must_tile_the_lanes():
    with pytest.raises(AssertionError):
        hrg.row_words(96)  # 24 words do not divide 128 lanes


def test_chunk_majors():
    assert host_row_table.chunk_majors(256, 2 << 30) == 65536
    assert host_row_table.chunk_majors(256, 1) == 1


class _NumpyRows:
    """Stand-in for ``HostRowTable`` with the same ``gather`` contract."""

    def __init__(self, rows: torch.Tensor):
        self.rows = rows.view(torch.uint8).clone()

    def gather(self, row_ids: torch.Tensor) -> torch.Tensor:
        assert row_ids.min() >= 0 and row_ids.max() < self.rows.shape[0]
        return self.rows[row_ids.long()]


def _embedding(monkeypatch, tp_size, rank, head_sizes, dim=64):
    monkeypatch.setattr(
        common_engram.upstream, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        common_engram.upstream, "get_tensor_model_parallel_rank", lambda: rank
    )
    emb = common_engram.ParallelEngramEmbedding(
        sum(head_sizes), dim, head_sizes, cpu_offload=True
    )
    gen = torch.Generator().manual_seed(rank)
    raw = torch.randint(0, 256, emb.weight.shape, dtype=torch.uint8, generator=gen)
    raw[raw == 0x7F] = 0  # no FP8 NaN in a real table
    raw[raw == 0xFF] = 0
    emb.weight.data.copy_(raw.view(torch.float8_e4m3fn))
    emb.weight_scale_inv.data.copy_(
        torch.randint(100, 150, emb.weight_scale_inv.shape, dtype=torch.uint8)
    )
    return emb


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (4, 1), (4, 3)])
def test_host_dma_lookup_matches_cpu_lookup(monkeypatch, tp_size, rank):
    head_sizes = (37, 41, 43, 47, 53, 59, 61, 67, 71, 73)
    emb = _embedding(monkeypatch, tp_size, rank, head_sizes)
    total = sum(head_sizes)
    gen = torch.Generator().manual_seed(7)
    # Ids span every shard plus out-of-range values for other ranks' heads.
    indices = torch.randint(0, total, (11, len(head_sizes)), generator=gen)
    out_shape = (11, emb.part_n_hash_cols, emb.dim)
    expected = torch.empty(out_shape, dtype=torch.bfloat16)
    emb.lookup(indices, expected)

    emb._host_rows = _NumpyRows(emb.weight.data)
    got = torch.full(out_shape, float("nan"), dtype=torch.bfloat16)
    emb.lookup(indices, got)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def test_host_dma_flag_requires_cpu_offload(monkeypatch):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", "1")
    monkeypatch.setattr(
        common_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        common_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0
    )
    on = common_engram.ParallelEngramEmbedding(64, 32, (8, 8), cpu_offload=True)
    off = common_engram.ParallelEngramEmbedding(64, 32, (8, 8), cpu_offload=False)
    assert on.host_dma and not off.host_dma
    # Without a converted table the lookup stays on the Host-DRAM rows.
    assert on._host_rows is None


def test_move_rows_to_pinned_host_frees_cpu_rows(monkeypatch):
    emb = _embedding(monkeypatch, 1, 0, (8, 8), dim=32)
    emb.host_dma = True
    weight = emb.weight.data.clone()
    scales = emb.weight_scale_inv.data.clone()
    built = []

    def fake_table(rows, device):
        built.append((rows.clone(), device))
        return _NumpyRows(rows)

    monkeypatch.setattr(common_engram, "HostRowTable", fake_table)
    emb.move_rows_to_pinned_host(torch.device("cpu"))
    emb.move_rows_to_pinned_host(torch.device("cpu"))  # idempotent
    assert len(built) == 1
    assert torch.equal(built[0][0].view(torch.uint8), weight.view(torch.uint8))
    assert emb.weight.shape == (0, 32)
    assert torch.equal(emb.weight_scale_inv.data, scales)
