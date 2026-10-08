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
from vllm_torchtpu.runner import utils as runner_utils


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


# --- TorchTPU glue -------------------------------------------------------------


@torch.library.custom_op(
    f"{host_row_table.HOST_OP_NAMESPACE}::test_gather", mutates_args=()
)
def _test_gather(row_ids: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
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

    def fn(row_ids, chunk, other):
        # ``other + 0`` is not a placeholder: it must stay unmarked.
        extra = torch.ops.dsv41_host_rows.test_gather(row_ids, other + 0)
        return torch.ops.dsv41_host_rows.test_gather(row_ids, chunk) + extra

    graph = torch.fx.symbolic_trace(fn)
    inputs = [torch.zeros(4, dtype=torch.int32) for _ in range(3)]
    tpu_backend._mark_pinned_host_inputs(graph, inputs)
    assert base_marks == [3]
    assert [getattr(t, "_is_pinned_host", False) for t in inputs] == [
        False,
        True,
        False,
    ]


def test_gather_op_signature_and_fake(monkeypatch):
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


def test_host_row_table_chunks_and_gathers(monkeypatch):
    """Chunked pack/offload and gather end to end, kernel interpreted."""
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
            return torch.from_numpy(np.asarray(out))

        return op

    monkeypatch.setattr(host_row_table, "_gather_op", gather_op)
    rng = np.random.default_rng(5)
    rows = rng.integers(0, 256, (700, 256), dtype=np.uint8)
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
    # A smaller shard sized for a 1100-row capacity: 9 majors -> 3 x 3, with
    # the chunk past its rows all zero.
    small = host_row_table.HostRowTable(
        torch.from_numpy(rows[:300]), torch.device("cpu"), chunk_bytes, 1100
    )
    assert [tuple(c.shape) for c in small.chunks] == [(3, 8, 1024)] * 3
    assert not small.chunks[2].any()
    ids = torch.tensor([299, 0, 128, 255], dtype=torch.int64)
    np.testing.assert_array_equal(small.gather(ids).numpy(), rows[ids.numpy()])


# --- Model and runner hooks ----------------------------------------------------


def _layer(cpu_offload=True, host_dma=True, engram=True):
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
    return SimpleNamespace(
        engram=SimpleNamespace(embed_tokens=embed, staged_rows=torch.empty(1))
    )


def _model(layers):
    model = object.__new__(v2_model.DeepseekV41Model)
    model.__dict__.update(layers=layers, start_layer=0, end_layer=len(layers))
    return model


def test_has_cpu_offloaded_engram_excludes_host_dma():
    assert not _model([_layer(engram=False), _layer()])._has_cpu_offloaded_engram()
    assert _model([_layer(host_dma=False)])._has_cpu_offloaded_engram()
    assert not _model([_layer(cpu_offload=False)])._has_cpu_offloaded_engram()


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


@pytest.mark.parametrize("host_dma,expected", [("0", True), ("1", False)])
def test_entry_layout_pins_supported(monkeypatch, host_dma, expected):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", host_dma)
    assert runner_utils.entry_layout_pins_supported() is expected
