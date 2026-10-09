# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the pinned_host Engram row gather (``ENGRAM_HOST_DMA``).

The kernel runs in the Pallas interpreter (``interpret=True`` supports the
``pl.HOST`` operands on CPU); the layout (``pack_rows``), the in-tile
selection (``select_rows``), the TorchTPU glue and the embedding's device
lookup are checked bitwise.
"""

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v41 import host_row_gather as hrg
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common import host_row_table
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model


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
    """Stand-in for ``HostRowTable`` with ``gather`` and ``lookup_and_dequant``."""

    def __init__(self, rows: torch.Tensor):
        self.rows = rows.view(torch.uint8).clone()

    def gather(self, row_ids: torch.Tensor) -> torch.Tensor:
        assert row_ids.min() >= 0 and row_ids.max() < self.rows.shape[0]
        return self.rows[row_ids.long()]

    def lookup_and_dequant(
        self,
        row_ids: torch.Tensor,
        valid: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        raw = self.gather(row_ids).numpy().view(np.int32)
        scale = weight_scale_inv[row_ids.long()].clone()
        scale.masked_fill_(~valid[:, None], hrg.INVALID_SCALE)
        dequant = hrg.dequant_fp8_rows_jax(
            jnp.asarray(raw), jnp.asarray(scale.numpy()), block_size
        )
        bits = np.asarray(jax.lax.bitcast_convert_type(dequant, jnp.int16))
        return torch.from_numpy(bits.copy()).view(torch.bfloat16)


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

    def fake_table(rows, device, capacity_rows):
        built.append((rows.clone(), device, capacity_rows))
        return _NumpyRows(rows)

    monkeypatch.setattr(common_engram, "HostRowTable", fake_table)
    emb.move_rows_to_pinned_host(torch.device("cpu"))
    emb.move_rows_to_pinned_host(torch.device("cpu"))  # idempotent
    assert len(built) == 1
    assert torch.equal(built[0][0].view(torch.uint8), weight.view(torch.uint8))
    assert built[0][2] == 16
    assert emb.weight.shape == (0, 32)
    assert torch.equal(emb.weight_scale_inv.data, scales)


@pytest.mark.parametrize(
    "tp_size,expected", [(1, 552), (2, 331), (4, 199), (5, 144), (10, 73)]
)
def test_max_shard_rows_is_rank_independent(monkeypatch, tp_size, expected):
    """Every rank sizes its pinned table for the largest shard."""
    head_sizes = (37, 41, 43, 47, 53, 59, 61, 67, 71, 73)
    sizes = set()
    for rank in range(tp_size):
        emb = _embedding(monkeypatch, tp_size, rank, head_sizes, dim=32)
        assert emb.part_num_embeddings <= emb.max_shard_rows()
        sizes.add(emb.max_shard_rows())
    assert sizes == {expected}


# --- Kernel through the Pallas interpreter -----------------------------------


@pytest.mark.parametrize("per_chunk", [6, 2, 4])
def test_gather_rows_jax_interpret(per_chunk):
    """Single, equal and partial-tail multi-chunk kernels match NumPy bitwise."""
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 256, (700, 256), dtype=np.uint8)
    packed = hrg.pack_rows(rows)  # 6 majors of 128 rows
    chunks = [jnp.asarray(packed[i : i + per_chunk]) for i in range(0, 6, per_chunk)]
    ids = np.array([0, 5, 127, 128, 255, 256, 383, 699, 3], dtype=np.int32)
    got = hrg.gather_rows_jax(jnp.asarray(ids), *chunks, row_bytes=256, interpret=True)
    np.testing.assert_array_equal(np.asarray(got), rows[ids].view(np.int32))


def test_lookup_and_dequant_rows_jax_interpret():
    """Fused Pallas gather + barrier + scale gather + dequant matches Torch."""
    rng = np.random.default_rng(11)
    rows = rng.integers(0, 256, (300, 256), dtype=np.uint8)
    rows[rows == 0x7F] = 0x7E
    rows[rows == 0xFF] = 0xFE
    rows[13, 128:] = 0x7F  # FP8 NaN masked by invalid=False
    scales = rng.integers(100, 150, (300, 2), dtype=np.uint8)
    scales[0, 0] = 0  # subnormal 2**-127 flushes to +0.0
    packed = hrg.pack_rows(rows)
    chunks = [jnp.asarray(packed[:2]), jnp.asarray(packed[2:])]
    ids = np.array([0, 13, 127, 128, 299], dtype=np.int32)
    valid = np.array([True, False, True, True, False], dtype=np.bool_)
    got = hrg.lookup_and_dequant_rows_jax(
        jnp.asarray(ids),
        jnp.asarray(valid),
        jnp.asarray(scales),
        *chunks,
        row_bytes=256,
        block_size=128,
        interpret=True,
    )
    got_bits = np.asarray(jax.lax.bitcast_convert_type(got, jnp.int16))
    expected_scale = torch.from_numpy(scales[ids].copy())
    expected_scale[~torch.from_numpy(valid)] = hrg.INVALID_SCALE
    expected = common_engram.dequant_fp8_rows(
        torch.from_numpy(rows[ids]), expected_scale, 128
    )
    np.testing.assert_array_equal(got_bits, expected.view(torch.int16).numpy())


# --- TorchTPU glue -------------------------------------------------------------


@torch.library.custom_op(
    f"{host_row_table.HOST_OP_NAMESPACE}::test_gather", mutates_args=()
)
def _test_gather(row_ids: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
    return row_ids.clone()


@torch.library.custom_op(
    f"{host_row_table.HOST_OP_NAMESPACE}::lookup_and_dequant_test", mutates_args=()
)
def _test_lookup_and_dequant(
    row_ids: torch.Tensor,
    valid: torch.Tensor,
    scale: torch.Tensor,
    chunk: torch.Tensor,
) -> torch.Tensor:
    return row_ids.clone()


def test_install_patches_placeholder_and_mark(monkeypatch):
    from torch_tpu._internal.compile import _backend as tpu_backend
    from torch_tpu._internal.compile import compiler as tpu_compiler
    from torch_tpu._internal.pallas import pallas as tpu_pallas

    pinned = torch.zeros(2, 8, 1024, dtype=torch.int32)
    plain = torch.zeros(4, dtype=torch.int32)

    def base_placeholder(tensor, mesh=None, partition_spec=None):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return jax.ShapeDtypeStruct(tuple(tensor.shape), jnp.int32)

    base_marks = []
    monkeypatch.setattr(tpu_pallas, "jax_placeholder", base_placeholder)
    monkeypatch.setattr(tpu_compiler, "is_pinned_host_tensor", lambda t: t is pinned)
    monkeypatch.setattr(
        tpu_backend,
        "_mark_pinned_host_inputs",
        lambda gm, inputs: base_marks.append(len(inputs)),
    )
    host_row_table._install.__wrapped__()

    placeholder = tpu_pallas.jax_placeholder
    assert placeholder(pinned).sharding.memory_kind == "pinned_host"
    assert placeholder(plain).sharding is None
    assert placeholder(pinned, mesh=object()).sharding is None  # mesh: untouched
    assert placeholder(3) == 3

    def fn(row_ids, chunk, other, valid, scale, chunk2):
        # ``other + 0`` is not a placeholder: it must stay unmarked.
        extra = torch.ops.dsv41_host_rows.test_gather(row_ids, other + 0)
        deq = torch.ops.dsv41_host_rows.lookup_and_dequant_test(
            row_ids, valid, scale, chunk2
        )
        return torch.ops.dsv41_host_rows.test_gather(row_ids, chunk) + extra + deq

    graph = torch.fx.symbolic_trace(fn)
    inputs = [torch.zeros(4, dtype=torch.int32) for _ in range(6)]
    tpu_backend._mark_pinned_host_inputs(graph, inputs)
    assert base_marks == [6]
    assert [getattr(t, "_is_pinned_host", False) for t in inputs] == [
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_gather_and_lookup_op_signatures_and_fakes(monkeypatch):
    from torch_tpu._internal.pallas import pallas as tpu_pallas

    made = {}

    class FakeOp:
        def register_fake(self, fn):
            made["fake"] = fn

    def jax_op(name, fn):
        made["name"], made["fn"] = name, fn
        return FakeOp()

    calls = []
    monkeypatch.setattr(tpu_pallas, "jax_op", jax_op)
    monkeypatch.setattr(
        hrg,
        "gather_rows_jax",
        lambda rows, *chunks, row_bytes: calls.append((rows, chunks, row_bytes)),
    )
    op = host_row_table._gather_op.__wrapped__(2, 256)
    assert isinstance(op, FakeOp)
    assert made["name"] == "dsv41_host_rows::gather_c2_b256"
    params = inspect.signature(made["fn"]).parameters.values()
    assert [p.annotation for p in params] == [jax.Array] * 3
    made["fn"]("ids", "c0", "c1")
    assert calls == [("ids", ("c0", "c1"), 256)]
    out = made["fake"](torch.zeros(5, dtype=torch.int32), torch.zeros(1, 8, 1024))
    assert out.shape == (5, 64) and out.dtype == torch.int32

    deq_calls = []
    monkeypatch.setattr(
        hrg,
        "lookup_and_dequant_rows_jax",
        lambda rows, valid, scale, *chunks, row_bytes, block_size: deq_calls.append(
            (rows, valid, scale, chunks, row_bytes, block_size)
        ),
    )
    deq_op = host_row_table._lookup_and_dequant_op.__wrapped__(2, 256, 128)
    assert isinstance(deq_op, FakeOp)
    assert made["name"] == "dsv41_host_rows::lookup_and_dequant_c2_b256_s128"
    deq_params = inspect.signature(made["fn"]).parameters.values()
    assert [p.annotation for p in deq_params] == [jax.Array] * 5
    made["fn"]("ids", "valid", "scale", "c0", "c1")
    assert deq_calls == [("ids", "valid", "scale", ("c0", "c1"), 256, 128)]
    deq_out = made["fake"](
        torch.zeros(5, dtype=torch.int32),
        torch.ones(5, dtype=torch.bool),
        torch.ones(10, 2, dtype=torch.uint8),
        torch.zeros(1, 8, 1024),
    )
    assert deq_out.shape == (5, 256) and deq_out.dtype == torch.bfloat16


def test_host_row_table_chunks_and_gathers(monkeypatch):
    """Chunked pack/offload, gather, and lookup_and_dequant end to end."""
    monkeypatch.setattr(host_row_table, "_install", lambda: None)
    monkeypatch.setattr(
        torch.ops, "ao", SimpleNamespace(offload=lambda t: t.clone()), raising=False
    )

    def gather_op(num_chunks, row_bytes):
        def op(row_ids, *chunks):
            assert len(chunks) == num_chunks
            out = hrg.gather_rows_jax(
                jnp.asarray(row_ids.numpy()),
                *(jnp.asarray(c.numpy()) for c in chunks),
                row_bytes=row_bytes,
                interpret=True,
            )
            return torch.from_numpy(np.asarray(out).copy())

        return op

    def lookup_and_dequant_op(num_chunks, row_bytes, block_size):
        def op(row_ids, valid, scale, *chunks):
            assert len(chunks) == num_chunks
            out = hrg.lookup_and_dequant_rows_jax(
                jnp.asarray(row_ids.numpy()),
                jnp.asarray(valid.numpy()),
                jnp.asarray(scale.numpy()),
                *(jnp.asarray(c.numpy()) for c in chunks),
                row_bytes=row_bytes,
                block_size=block_size,
                interpret=True,
            )
            bits = np.asarray(jax.lax.bitcast_convert_type(out, jnp.int16))
            return torch.from_numpy(bits.copy()).view(torch.bfloat16)

        return op

    monkeypatch.setattr(host_row_table, "_gather_op", gather_op)
    monkeypatch.setattr(host_row_table, "_lookup_and_dequant_op", lookup_and_dequant_op)
    rng = np.random.default_rng(5)
    rows = rng.integers(0, 256, (700, 256), dtype=np.uint8)
    rows[rows == 0x7F] = 0x7E
    rows[rows == 0xFF] = 0xFE
    scales = rng.integers(100, 150, (700, 2), dtype=np.uint8)
    chunk_bytes = 4 * hrg.SUBLANES * hrg.MAJOR_LANES * 4
    # 700 rows = 6 majors of 128; at most 4 majors per chunk: 2 x 3 majors.
    table = host_row_table.HostRowTable(
        torch.from_numpy(rows).view(torch.float8_e4m3fn),
        torch.device("cpu"),
        chunk_bytes=chunk_bytes,
    )
    assert [tuple(c.shape) for c in table.chunks] == [(3, 8, 1024)] * 2
    ids = torch.tensor([699, 0, 511, 512, 640, 17], dtype=torch.int64)
    got = table.gather(ids)
    assert got.dtype == torch.uint8
    np.testing.assert_array_equal(got.numpy(), rows[ids.numpy()])

    valid = torch.tensor([True, False, True, True, False, True])
    scales_t = torch.from_numpy(scales)
    got_deq = table.lookup_and_dequant(ids, valid, scales_t, 128)
    exp_scale = scales_t[ids].clone()
    exp_scale[~valid] = hrg.INVALID_SCALE
    exp_deq = common_engram.dequant_fp8_rows(got, exp_scale, 128)
    assert torch.equal(got_deq.view(torch.int16), exp_deq.view(torch.int16))

    # A smaller shard sized for a 1100-row capacity: 9 majors -> 3 x 3, with
    # the chunk past its rows all zero.
    small = host_row_table.HostRowTable(
        torch.from_numpy(rows[:300]), torch.device("cpu"), chunk_bytes, 1100
    )
    assert [tuple(c.shape) for c in small.chunks] == [(3, 8, 1024)] * 3
    assert not small.chunks[2].any()
    ids = torch.tensor([299, 0, 128, 255], dtype=torch.int64)
    np.testing.assert_array_equal(small.gather(ids).numpy(), rows[ids.numpy()])


def test_engram_host_dma_staging_reuses_bucket_rows(monkeypatch):
    emb = _embedding(monkeypatch, 1, 0, (16, 16), dim=64)
    emb.host_dma = True
    emb._host_rows = _NumpyRows(emb.weight.data)
    engram = object.__new__(common_engram.Engram)
    engram.__dict__["embed_tokens"] = emb
    engram._init_staging(max_tokens=16, head_dim=64)
    # Before prepare_embeddings, _ready_rows falls back to staged_rows[:n].
    assert engram._ready_rows(3).shape == (3, 2, 64)
    # Single-token bucket (num_tokens=1 < 2) and multi-token bucket (num_tokens=4 >= 2).
    for num_tokens in (1, 4):
        hashes = torch.randint(0, 32, (num_tokens, 2))
        engram.prepare_embeddings(hashes)
        first = engram._ready_rows(num_tokens)
        assert first.shape == (num_tokens, 2, 64)
        engram.prepare_embeddings(hashes)
        assert engram._ready_rows(num_tokens) is first


# --- Model and runner hooks ----------------------------------------------------


def _layer(cpu_offload=True, host_dma=True, engram=True, layer_hash_index=0):
    if not engram:
        return SimpleNamespace(engram=None)
    moved = []

    def move(device):
        moved.append(device)
        embed._host_rows = SimpleNamespace(chunks=[0, 1])

    embed = SimpleNamespace(
        cpu_offload=cpu_offload,
        host_dma=host_dma,
        _host_rows=None,
        move_rows_to_pinned_host=move,
        moved=moved,
    )
    prepared = []
    active_rows = torch.empty(2, 2, 4)

    def prepare(hashes):
        prepared.append(hashes.clone())

    return SimpleNamespace(
        engram=SimpleNamespace(
            embed_tokens=embed,
            staged_rows=torch.empty(1),
            _active_rows=active_rows,
            layer_hash_index=layer_hash_index,
            prepare_embeddings=prepare,
            prepared=prepared,
        )
    )


def _model(layers):
    model = object.__new__(v2_model.DeepseekV41Model)
    model.__dict__.update(layers=layers, start_layer=0, end_layer=len(layers))
    return model


def test_has_cpu_offloaded_and_host_dma_engram():
    assert not _model([_layer(engram=False), _layer()])._has_cpu_offloaded_engram()
    assert _model([_layer(engram=False), _layer()])._has_host_dma_engram()
    assert _model([_layer(host_dma=False)])._has_cpu_offloaded_engram()
    assert not _model([_layer(host_dma=False)])._has_host_dma_engram()
    assert not _model(
        [_layer(cpu_offload=False, host_dma=False)]
    )._has_cpu_offloaded_engram()
    assert not v2_model.DeepseekV41Model._has_host_dma_engram(SimpleNamespace())


def test_prepare_engram_and_forward_host_dma_outside_stage(monkeypatch):
    class FakeTpuTensor:
        def __init__(self, t, dev_type="tpu"):
            self.t = t
            self.device = SimpleNamespace(type=dev_type)
            self.shape = t.shape

        def to(self, dev):
            return self.t.to(dev)

        def __getitem__(self, idx):
            return self.t[idx]

    synced = []
    monkeypatch.setattr(
        v2_model,
        "synchronize_tensors",
        lambda tensors, wait=True: synced.append((tuple(tensors), wait)),
    )
    monkeypatch.setattr(v2_model.upstream, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        v2_model.upstream,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "swa": SimpleNamespace(
                    query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
                    seq_lens=torch.tensor([2], dtype=torch.int32),
                )
            },
            additional_kwargs={},
        ),
    )
    monkeypatch.setattr(
        v2_model.upstream,
        "image_sentinel_mask",
        lambda t: torch.zeros_like(t, dtype=torch.bool),
    )
    monkeypatch.setattr(v2_model, "_cache_slots", lambda kv, meta, pos: None)
    monkeypatch.setattr(v2_model, "_block_table", lambda meta: None)
    raw_hashes = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32)
    tpu_hashes = FakeTpuTensor(raw_hashes, "tpu")
    monkeypatch.setattr(
        v2_model,
        "gather_host_engram_hashes",
        lambda h, dp_shared_memory, gather_fn: tpu_hashes,
    )

    class FakeHash:
        use_slot_cache = True
        lookback_depth = 2
        multipliers = torch.ones(2, 3, dtype=torch.int64)
        swa_cache_module = SimpleNamespace(kv_cache=torch.ones(1))

        def ensure_cache(self):
            return True

        def __call__(self, *args):
            return raw_hashes

    l0 = _layer(engram=False)
    l1 = _layer(cpu_offload=True, host_dma=True, layer_hash_index=0)
    l2 = _layer(cpu_offload=True, host_dma=True, layer_hash_index=1)
    tpu_active = FakeTpuTensor(torch.zeros(2, 2, 4), "tpu")
    l1.engram._active_rows = tpu_active
    l2.engram._active_rows = torch.zeros(2, 2, 4)  # cpu device: no sync

    model = _model([l0, l1, l2])
    model.__dict__.update(
        engram_hash=FakeHash(),
        engram_swa_prefix="swa",
        engram_dp_shared_memory=True,
        use_mega_moe=False,
        _ced_enabled=False,
    )
    seen_outside = []
    model.__dict__["full_stage"] = lambda *a: (
        seen_outside.append(model._engram_prepared_outside_stage) or "full_out"
    )
    input_ids = torch.tensor([10, 11], dtype=torch.int32)
    positions = torch.tensor([0, 1], dtype=torch.int32)
    assert model.forward(input_ids, positions) == "full_out"
    assert seen_outside == [True]
    assert not model._engram_prepared_outside_stage
    assert len(l1.engram.prepared) == 1 and len(l2.engram.prepared) == 1
    assert synced == [((tpu_hashes,), False), ((tpu_active,), False)]

    # When gathered_hashes is on CPU under host_dma, no sync is issued.
    synced.clear()
    tpu_hashes.device.type = "cpu"
    l1.engram._active_rows = None
    model._prepare_engram(input_ids, positions, None)
    assert synced == []

    # Under cpu_offload=True, host_dma=False, gathered_hashes moves to CPU and no
    # host_dma sync runs.
    l1.engram.embed_tokens.host_dma = False
    l2.engram.embed_tokens.host_dma = False
    model._prepare_engram(input_ids, positions, None)
    assert synced == []

    # Under _forward_stage while compiling with host_dma=True, skip_lookup is True.
    l1.engram.embed_tokens.host_dma = True
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    model.__dict__.update(
        use_sequence_parallel=False,
    )
    skip_seen = []
    orig_prepare = model._prepare_engram

    class _StopStage(Exception):
        pass

    def spy_prepare(*args, **kwargs):
        skip_seen.append(kwargs.get("skip_lookup"))
        orig_prepare(*args, **kwargs)
        raise _StopStage

    monkeypatch.setattr(model, "_prepare_engram", spy_prepare)
    with pytest.raises(_StopStage):
        model._forward_stage(
            input_ids, positions, None, inputs_embeds=torch.zeros(2, 4, 8)
        )
    assert skip_seen == [True]


def test_move_engram_rows_to_pinned_host_runs_once():
    layers = [_layer(engram=False), _layer(), _layer()]
    model = _model(layers)
    model._move_engram_rows_to_pinned_host()
    model._move_engram_rows_to_pinned_host()
    for layer in layers[1:]:
        assert layer.engram.embed_tokens.moved == [torch.device("cpu")]
    assert model._engram_rows_pinned


@pytest.mark.parametrize("host_dma", ["0", "1"])
def test_causal_lm_converts_after_load_and_before_forward(monkeypatch, host_dma):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", host_dma)
    upstream_lm = v2_model.upstream.DeepseekV41LLMForCausalLM
    monkeypatch.setattr(upstream_lm, "process_weights_after_loading", lambda self: None)
    monkeypatch.setattr(upstream_lm, "forward", lambda self, *a, **k: "out")
    moves = []
    lm = object.__new__(v2_model.DeepseekV41ForCausalLM)
    lm.__dict__["model"] = SimpleNamespace(
        engram_hash=None, _move_engram_rows_to_pinned_host=lambda: moves.append(1)
    )
    lm.process_weights_after_loading()
    assert lm.forward() == "out"
    assert len(moves) == (2 if host_dma == "1" else 0)


def test_select_rows_lowers_without_stablehlo_gather():
    dummy_tiles = jnp.zeros((16, 8, 128), dtype=jnp.int32)
    dummy_slot = jnp.zeros((16,), dtype=jnp.int32)
    hlo_text = str(
        jax.jit(hrg.select_rows, static_argnums=(2,))
        .lower(dummy_tiles, dummy_slot, 64)
        .compiler_ir("stablehlo")
    )
    assert "stablehlo.gather" not in hlo_text


def test_ngram_hash_state_forward_reuses_mapped_batch():
    class CountingTokenMap:
        def __init__(self, tensor: torch.Tensor):
            self._tensor = tensor
            self.calls = 0

        def __getitem__(self, idx):
            self.calls += 1
            return self._tensor[idx]

    class SpyState(common_engram.NgramHashState):
        def write_cache(self, cache, slots, values, valid):
            pass

    state = object.__new__(SpyState)
    torch.nn.Module.__init__(state)
    state.token_map = CountingTokenMap(torch.arange(100, dtype=torch.int64))
    state.multipliers = torch.ones((2, 3), dtype=torch.int64)
    state.primes = torch.full((2, 2, 4), 97, dtype=torch.int64)
    state.offsets = torch.zeros((2, 8), dtype=torch.int64)
    state.lookback_depth = 4
    state.pad_id = 0
    state.use_slot_cache = True
    state.block_size = 8
    state._cache = torch.zeros((64,), dtype=torch.int64)
    state._kv_cache_ref = None
    state.swa_cache_module = SimpleNamespace(
        kv_cache=torch.zeros((8, 8), dtype=torch.int64),
        bound_cache=torch.zeros((8, 8), dtype=torch.int64),
        block_size=8,
    )

    ids = torch.tensor([1, 2], dtype=torch.int32)
    pos = torch.tensor([0, 1], dtype=torch.int64)
    qsl = torch.tensor([0, 2], dtype=torch.int32)
    dead = torch.zeros(2, dtype=torch.bool)
    lookback = torch.full((1, 4), -1, dtype=torch.int32)
    dead_lb = torch.zeros((1, 4), dtype=torch.bool)
    slots = torch.tensor([0, 1], dtype=torch.int64)
    block_table = torch.zeros((1, 4), dtype=torch.int32)

    out = state.forward(ids, pos, qsl, dead, lookback, dead_lb, slots, block_table)
    assert out.shape == (2, 2, 8)
    assert state.token_map.calls == 2


def test_td714_hash_on_host_active_lookback_and_direct_scale_pack(monkeypatch):
    state = object.__new__(common_engram.NgramHashState)
    torch.nn.Module.__init__(state)
    state.token_map = torch.arange(1, 101, dtype=torch.int64)
    state.multipliers = torch.tensor([[3, 5, 7], [11, 13, 17]], dtype=torch.int64)
    state.primes = torch.full((2, 2, 4), 97, dtype=torch.int64)
    state.offsets = torch.arange(16, dtype=torch.int64).reshape(2, 8)
    state.lookback_depth = 4
    state.pad_id = 0

    ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
    pos = torch.tensor([0, 1, 10, 11, 20, 21], dtype=torch.int64)
    qsl = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    dead = torch.zeros(6, dtype=torch.bool)
    lb_small = torch.tensor(
        [[10, 11, -1, 13], [20, 21, 22, -1], [30, -1, 32, 33]], dtype=torch.int32
    )
    lb_dead_small = torch.tensor(
        [
            [False, True, False, False],
            [False, False, False, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
    )
    lb_large = torch.full((1024, 4), 99, dtype=torch.int32)
    lb_large[:3] = lb_small
    lb_dead_large = torch.ones((1024, 4), dtype=torch.bool)
    lb_dead_large[:3] = lb_dead_small

    out_small = state.hash_on_host(ids, pos, qsl, dead, lb_small, lb_dead_small)
    tables = state._host_tables
    scratch_6 = tables._ensure_scratch(6)
    out_large = state.hash_on_host(ids, pos, qsl, dead, lb_large, lb_dead_large)
    assert torch.equal(out_small, out_large)
    assert all(
        a is b for a, b in zip(scratch_6, tables._ensure_scratch(6), strict=True)
    )

    # Padded tail (`qsl[-1] < count`) includes row `qsl.shape[0] - 1` (`index 3`)
    # while still ignoring inactive rows `4..1023`.
    qsl_padded = torch.tensor([0, 2, 4, 4], dtype=torch.int32)
    lb_large_4 = lb_large.clone()
    lb_large_4[3] = torch.tensor([40, 41, 42, 43], dtype=torch.int32)
    lb_dead_4 = lb_dead_large.clone()
    lb_dead_4[3] = False
    out_pad_4 = state.hash_on_host(
        ids, pos, qsl_padded, dead, lb_large_4[:4], lb_dead_4[:4]
    )
    out_pad_1024 = state.hash_on_host(ids, pos, qsl_padded, dead, lb_large_4, lb_dead_4)
    assert torch.equal(out_pad_4, out_pad_1024)

    # Direct scale gather into `_host_packed_buf` keeps `_pending_scale_sync`
    # deferred through `_ready_rows` until `_host_scale` / `_host_scale_buf` is read.
    emb = _embedding(monkeypatch, 1, 0, (16, 16), dim=64)
    emb.host_dma = False
    engram = object.__new__(common_engram.Engram)
    torch.nn.Module.__init__(engram)
    engram.embed_tokens = emb
    engram._init_staging(max_tokens=8, head_dim=64)
    engram._raw_scale_buf.fill_(111)
    hashes = torch.tensor([[0, 16], [5, 999], [15, 20]], dtype=torch.int32)
    engram.prepare_embeddings(hashes)
    assert engram._pending_scale_sync is not None
    assert torch.all(engram._raw_scale_buf == 111)
    ready = engram._ready_rows(3)
    assert ready.shape == (3, 2, 64)
    assert engram._pending_scale_sync is not None
    # Reading `_host_scale` triggers the lazy sync of the active scale words.
    assert torch.all(engram._host_scale[1, 1] == common_engram._INVALID_SCALE)
    assert engram._pending_scale_sync is None
    assert torch.all(engram._host_scale[3:] == 111)
