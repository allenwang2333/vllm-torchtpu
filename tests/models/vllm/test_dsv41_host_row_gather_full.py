# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for host_row_gather.py."""

import os
import sys
import types

import numpy as np
import torch

_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)


def _ref_dequant_fp8_rows(
    raw_u8: np.ndarray, scale_u8: np.ndarray, block_size: int
) -> np.ndarray:
    vals = (
        torch.from_numpy(raw_u8.copy())
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .reshape(raw_u8.shape[0], scale_u8.shape[1], block_size)
    )
    sc_i32 = torch.from_numpy(scale_u8.astype(np.int32)) << 23
    power = sc_i32.view(torch.float32).unsqueeze(-1)
    invalid = torch.from_numpy(scale_u8 == 255).unsqueeze(-1)
    out = torch.where(invalid, torch.zeros_like(vals), vals * power)
    return out.reshape(raw_u8.shape[0], -1).to(torch.bfloat16).to(torch.float32).numpy()


class _FakeLax:
    @staticmethod
    def fori_loop(lower: int, upper: int, body_fun, init_val):
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val

    @staticmethod
    def bitcast_convert_type(x: np.ndarray, new_dtype) -> np.ndarray:
        arr = np.ascontiguousarray(x)
        if new_dtype == "float8_e4m3fn":
            u8 = arr.view(np.uint8)
            t = torch.from_numpy(u8.copy()).view(torch.float8_e4m3fn)
            return t.to(torch.float32).numpy().reshape(*arr.shape, 4)
        if arr.dtype == np.uint16 or new_dtype == "bfloat16":
            u16 = arr.astype(np.uint16, copy=False)
            t = torch.from_numpy(u16.copy()).view(torch.bfloat16)
            return t.to(torch.float32).numpy()
        return arr.view(np.dtype(new_dtype))

    @staticmethod
    def optimization_barrier(x):
        return x


class _FakeJnp:
    int32 = np.int32
    uint16 = np.uint16
    uint8 = np.uint8
    float32 = np.float32
    bfloat16 = np.float32
    float8_e4m3fn = "float8_e4m3fn"

    @staticmethod
    def arange(n: int, dtype=np.int32) -> np.ndarray:
        return np.arange(n, dtype=dtype)

    @staticmethod
    def where(cond, x, y) -> np.ndarray:
        return np.where(cond, x, y)

    @staticmethod
    def sum(x, axis=None, dtype=None) -> np.ndarray:
        return np.sum(x, axis=axis, dtype=dtype)

    @staticmethod
    def take(a, indices, axis=0) -> np.ndarray:
        return np.take(a, indices, axis=axis)

    @staticmethod
    def take_along_axis(arr, indices, axis) -> np.ndarray:
        return np.take_along_axis(arr, indices, axis=axis)


class _RefView:
    def __init__(self, arr: np.ndarray, slices=None):
        self.arr = arr
        self.slices = slices

    @property
    def shape(self):
        return self.view().shape

    def view(self) -> np.ndarray:
        return self.arr if self.slices is None else self.arr[self.slices]

    @property
    def at(self):
        return self

    def __getitem__(self, item):
        if self.slices is None:
            return _RefView(self.arr, item)
        return _RefView(self.view(), item)


class _FakeCopy:
    def __init__(self, src: _RefView, dst: _RefView):
        self.src = src
        self.dst = dst

    def start(self) -> None:
        self.dst.view()[...] = self.src.view()

    def wait(self) -> None:
        return None


class _FakePlTpu:
    HBM = staticmethod(lambda shape=None, dtype=None: (shape, dtype))

    class SemaphoreType:
        @staticmethod
        def DMA(shape):
            return ("DMA", shape)

    class PrefetchScalarGridSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    @staticmethod
    def make_async_copy(src: _RefView, dst: _RefView, sem) -> _FakeCopy:
        return _FakeCopy(src, dst)


class _FakePl:
    HOST = "HOST"
    tpu = _FakePlTpu

    class BlockSpec:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def ds(start: int, size: int) -> slice:
        return slice(int(start), int(start) + int(size))

    @staticmethod
    def multiple_of(x: int, m: int) -> int:
        return int(x)

    @staticmethod
    def when(cond: bool):
        def _decorator(fn):
            if bool(cond):
                fn()
            return fn

        return _decorator

    @staticmethod
    def pallas_call(kernel_fn, *, grid_spec, out_shape, name: str, interpret: bool):
        shape, dtype = out_shape

        def _runner(tile_ids: np.ndarray, *chunks: np.ndarray) -> np.ndarray:
            out = np.zeros(shape, dtype=dtype)
            chunk_views = [_RefView(c) for c in chunks]
            out_view = _RefView(out)
            kernel_fn(tile_ids, *chunk_views, out_view, "sem")
            return out

        return _runner


fake_src = types.ModuleType("jax._src")
fake_src.xla_bridge = types.SimpleNamespace(  # type: ignore[attr-defined]
    backends_are_initialized=lambda: False
)
fake_jax = types.ModuleType("jax")
fake_jax.Array = np.ndarray  # type: ignore[attr-defined]
fake_jax.lax = _FakeLax  # type: ignore[attr-defined]
fake_jax.numpy = _FakeJnp  # type: ignore[attr-defined]
fake_jax._src = fake_src  # type: ignore[attr-defined]
fake_exp = types.ModuleType("jax.experimental")
fake_exp.pallas = _FakePl  # type: ignore[attr-defined]
fake_pallas = types.ModuleType("jax.experimental.pallas")
fake_pallas.tpu = _FakePlTpu  # type: ignore[attr-defined]

_FAKE_MODULES = {
    "jax": fake_jax,
    "jax._src": fake_src,
    "jax.numpy": _FakeJnp,
    "jax.experimental": fake_exp,
    "jax.experimental.pallas": _FakePl,
    "jax.experimental.pallas.tpu": _FakePlTpu,
}
_SAVED_MODULES = {k: sys.modules.get(k) for k in _FAKE_MODULES}
sys.modules.update(_FAKE_MODULES)  # type: ignore[arg-type]
from vllm_torchtpu.kernels.deepseek_v41 import host_row_gather as hrg  # noqa: E402

for _k, _v in _SAVED_MODULES.items():
    if _v is None:
        sys.modules.pop(_k, None)
    else:
        sys.modules[_k] = _v


def test_pack_rows_geometry_and_branches() -> None:
    row_bytes = 512  # 128 words -> 8 rows per tile, 64 rows per major
    assert hrg.row_words(row_bytes) == 128
    assert hrg.rows_per_tile(row_bytes) == 8
    assert hrg.num_majors(64, row_bytes) == 1
    assert hrg.num_majors(65, row_bytes) == 2

    rng = np.random.default_rng(42)
    # 1. Exact full major (`full_majors == majors`, `full_majors > 0`, no tail):
    rows_exact = rng.integers(0, 256, size=(64, row_bytes), dtype=np.uint8)
    packed_exact = hrg.pack_rows(rows_exact)
    assert packed_exact.shape == (1, hrg.SUBLANES, hrg.MAJOR_LANES)

    # 2. Exact full major + extra zero-only padded major (`rem == 0`):
    packed_padded = hrg.pack_rows(rows_exact, majors=2)
    assert packed_padded.shape == (2, hrg.SUBLANES, hrg.MAJOR_LANES)
    assert np.array_equal(packed_padded[0], packed_exact[0])
    assert np.all(packed_padded[1] == 0)

    # 3. Partial major only (`full_majors == 0`, `rem > 0`):
    rows_partial = rows_exact[:19]
    packed_partial = hrg.pack_rows(rows_partial)
    assert packed_partial.shape == (1, hrg.SUBLANES, hrg.MAJOR_LANES)


def test_gather_and_lookup_dequant_single_and_multi_chunk() -> None:
    rng = np.random.default_rng(7)
    row_bytes = 512
    block_size = 128
    rows_raw = rng.integers(0, 256, size=(1200, row_bytes), dtype=np.uint8)
    rows_raw[(rows_raw == 0x7F) | (rows_raw == 0xFF)] = 0x7E
    # Inject NaN FP8 byte (`0x7f` and `0xff`) into row 5 so invalid_scale=255 masking is
    # tested.
    rows_raw[5, :4] = np.array([0x7F, 0xFF, 0x80, 0x00], dtype=np.uint8)

    chunk0 = hrg.pack_rows(rows_raw[:512], majors=8)
    chunk1 = hrg.pack_rows(rows_raw[512:1024], majors=8)
    chunk2 = hrg.pack_rows(rows_raw[1024:], majors=4)

    # Single chunk gather with int64 rows (covers rows.dtype != int32 conversion):
    idx_single = np.array([0, 5, 511], dtype=np.int64)
    g_single = hrg.gather_rows_jax(
        idx_single, chunk0, row_bytes=row_bytes, interpret=True
    )
    assert np.array_equal(g_single.view(np.uint8), rows_raw[idx_single])

    # Also test select_rows directly with non-int32 slot and >1024 tiles:
    tiles_dummy = np.zeros((2, hrg.SUBLANES, hrg.LANES), dtype=np.int32)
    sel_out = hrg.select_rows(
        tiles_dummy, np.array([0, 1], dtype=np.int64), row_bytes=row_bytes
    )
    assert sel_out.shape == (2, row_bytes // 4)
    tiles_large = np.zeros((1025, hrg.SUBLANES, hrg.LANES), dtype=np.int32)
    tiles_large[1024, 3, :] = 77
    sel_large = hrg.select_rows(
        tiles_large, np.full((1025,), 3, dtype=np.int32), row_bytes=row_bytes
    )
    assert sel_large.shape == (1025, row_bytes // 4)
    assert np.all(sel_large[1024] == 77)

    # Multi-chunk gather + lookup_and_dequant with int64 rows and int32 rows:
    idx_multi = np.array([5, 520, 1100], dtype=np.int64)
    valid = np.array([False, True, True], dtype=np.bool_)
    scales = rng.integers(
        120, 132, size=(1200, row_bytes // block_size), dtype=np.uint8
    )
    scales[520, 0] = 0  # subnormal scale -> flush to 0
    deq = hrg.lookup_and_dequant_rows_jax(
        idx_multi,
        valid,
        scales,
        chunk0,
        chunk1,
        chunk2,
        row_bytes=row_bytes,
        block_size=block_size,
        interpret=True,
    )
    deq_i32 = hrg.lookup_and_dequant_rows_jax(
        idx_multi.astype(np.int32),
        valid,
        scales,
        chunk0,
        chunk1,
        chunk2,
        row_bytes=row_bytes,
        block_size=block_size,
        interpret=True,
    )
    assert np.array_equal(deq, deq_i32)

    # Reference verification against torch FP8 dequant:
    ref_scales = scales[idx_multi].copy()
    ref_scales[~valid] = hrg.INVALID_SCALE
    ref = _ref_dequant_fp8_rows(rows_raw[idx_multi], ref_scales, block_size=block_size)
    assert np.allclose(deq, ref, atol=0.0, rtol=0.0)
    assert np.all(deq[0] == 0.0)
    assert np.all(deq[1, :block_size] == 0.0)

    # Also test > 256 rows path in dequant_fp8_rows_jax (including invalid_scale=255):
    raw_300 = np.ascontiguousarray(rows_raw[:300]).view(np.int32)
    scales_300 = scales[:300].copy()
    scales_300[5, :] = hrg.INVALID_SCALE
    deq_300 = hrg.dequant_fp8_rows_jax(raw_300, scales_300, block_size=block_size)
    ref_300 = _ref_dequant_fp8_rows(rows_raw[:300], scales_300, block_size=block_size)
    assert np.allclose(deq_300, ref_300, atol=0.0, rtol=0.0)
    assert np.all(deq_300[5] == 0.0)
