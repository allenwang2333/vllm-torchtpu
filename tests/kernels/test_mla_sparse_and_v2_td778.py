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
"""CPU unit tests for TD7.78: MLA sparse gather and MLA v2 KV/transpose utils."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import (
    dsa_gather,
    native_sc_cache_layout,
    native_sc_gather,
)
from vllm_torchtpu.kernels.mla.v2 import kv_utils, transpose

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# 1. native_sc_cache_layout.py tests
# ---------------------------------------------------------------------------


def test_native_sc_cache_layout_all_branches() -> None:
    # _require_u8 error branches
    with pytest.raises(ValueError, match="nope_u8 must be uint8"):
        native_sc_cache_layout.pack_nope(np.zeros((2, 4, 128), dtype=np.int32))
    with pytest.raises(ValueError, match="nope_u8 must be uint8"):
        native_sc_cache_layout.pack_nope(np.zeros((2, 4, 64), dtype=np.uint8))

    # pack_nope / unpack_nope round-trip and error branch
    nope_u8 = np.arange(2 * 4 * 128, dtype=np.uint8).reshape(2, 4, 128)
    packed_nope = native_sc_cache_layout.pack_nope(nope_u8)
    assert packed_nope.shape == (2, 128)
    assert packed_nope.dtype == np.uint32
    np.testing.assert_array_equal(
        native_sc_cache_layout.unpack_nope(packed_nope), nope_u8
    )
    with pytest.raises(ValueError, match="nope_i32 must be uint32"):
        native_sc_cache_layout.unpack_nope(np.zeros((2, 64), dtype=np.uint32))

    # pack_rope_banded / unpack_rope_banded round-trip and error branch
    rope_u8 = np.arange(3 * 4 * 128, dtype=np.uint8).reshape(3, 4, 128)
    packed_rope = native_sc_cache_layout.pack_rope_banded(rope_u8)
    assert packed_rope.shape == (3, 128)
    assert packed_rope.dtype == np.uint32
    np.testing.assert_array_equal(
        native_sc_cache_layout.unpack_rope_banded(packed_rope), rope_u8
    )
    with pytest.raises(ValueError, match="rope_i32 must be uint32"):
        native_sc_cache_layout.unpack_rope_banded(np.zeros((3, 64), dtype=np.uint32))

    # pack_native_caches: 3D nope + 2D rope, and 4D nope + 4D rope
    nope_3d = np.zeros((8, 4, 128), dtype=np.uint8)
    rope_2d = np.zeros((8, 128), dtype=np.uint8)
    pn3, pr2 = native_sc_cache_layout.pack_native_caches(nope_3d, rope_2d, page_size=4)
    assert pn3.shape == (2, 4, 128)
    assert pr2.shape == (2, 1, 128)

    nope_4d = np.zeros((2, 4, 4, 128), dtype=np.uint8)
    rope_4d = np.zeros((2, 1, 4, 128), dtype=np.uint8)
    pn4, pr4 = native_sc_cache_layout.pack_native_caches(nope_4d, rope_4d, page_size=4)
    assert pn4.shape == (2, 4, 128)
    assert pr4.shape == (2, 1, 128)

    # pack_selected_rope_for_tc
    sel_tc = native_sc_cache_layout.pack_selected_rope_for_tc(
        np.arange(8 * 128, dtype=np.uint8).reshape(8, 128)
    )
    assert sel_tc.shape == (2, 128)
    assert sel_tc.dtype == np.uint32

    # flatten_nope_rows / flatten_rope_rows + error branches
    assert native_sc_cache_layout.flatten_nope_rows(pn4).shape == (8, 128)
    assert native_sc_cache_layout.flatten_rope_rows(pr4).shape == (8, 32)
    with pytest.raises(ValueError, match="expected uint32 NOPE cache"):
        native_sc_cache_layout.flatten_nope_rows(np.zeros((2, 64), dtype=np.uint32))
    with pytest.raises(ValueError, match="expected uint32 ROPE cache"):
        native_sc_cache_layout.flatten_rope_rows(np.zeros((2, 64), dtype=np.uint32))

    # source_bytes_per_token + mismatch error branch
    assert native_sc_cache_layout.source_bytes_per_token(pn4, pr4) == (512, 128)
    with pytest.raises(ValueError, match="cache token counts differ"):
        native_sc_cache_layout.source_bytes_per_token(
            pn4, np.zeros((3, 1, 128), dtype=np.uint32)
        )


# ---------------------------------------------------------------------------
# 2. kv_utils.py tests
# ---------------------------------------------------------------------------


class _FakeVmemRef:
    def __init__(self, arr: jax.Array):
        self.arr = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, val):
        self.arr = self.arr.at[idx].set(val)


def test_kv_utils_helpers_and_pack_new_kv(monkeypatch) -> None:
    # Power-of-2 and non-power-of-2 branches for unsigned_*
    assert kv_utils.unsigned_mod(19, 8) == 3
    assert kv_utils.unsigned_mod(19, 6) == 1

    assert kv_utils.unsigned_cdiv(19, 8) == 3
    assert kv_utils.unsigned_cdiv(19, 6) == 4

    assert kv_utils.unsigned_floor_div(19, 8) == 2
    assert kv_utils.unsigned_floor_div(19, 6) == 3

    assert kv_utils.unsigned_align_to(19, 8) == 24
    assert kv_utils.unsigned_align_to(19, 6) == 24

    assert kv_utils.align_to(130, 128) == 256
    assert kv_utils.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert kv_utils.get_dtype_packing(jnp.bfloat16) == 2

    # Stub pltpu.bitcast, pltpu.roll, and lax.fori_loop for CPU execution
    monkeypatch.setattr(
        kv_utils.pltpu,
        "bitcast",
        lambda x, dt: jax.lax.bitcast_convert_type(
            x,
            jnp.uint16
            if jnp.dtype(dt) == jnp.uint32 and x.dtype == jnp.bfloat16
            else dt,
        )
        if False
        else x.view(dt),
    )
    monkeypatch.setattr(
        kv_utils.pltpu,
        "roll",
        lambda x, shift, axis=0: jnp.roll(x, shift=shift, axis=axis),
    )

    def _py_fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    monkeypatch.setattr(kv_utils.lax, "fori_loop", _py_fori_loop)

    # 1. pack_new_kv with num_sublanes_for_kv_packing == 1 (kv_packing=2, bfloat16)
    kvc_ref_1 = _FakeVmemRef(jnp.ones((8, 2, 4), dtype=jnp.bfloat16))
    kpe_ref_1 = _FakeVmemRef(jnp.ones((8, 2, 2), dtype=jnp.bfloat16))
    kv_utils.pack_new_kv(
        kvc_ref_1,
        kpe_ref_1,
        offset=jnp.int32(1),
        update_sz=jnp.int32(2),
        q_end=jnp.int32(4),
        kv_len=jnp.int32(3),
        bkv_sz=8,
    )

    # 2. pack_new_kv with num_sublanes_for_kv_packing > 1 (kv_packing=4, bfloat16)
    kvc_ref_2 = _FakeVmemRef(jnp.ones((8, 4, 4), dtype=jnp.bfloat16))
    kpe_ref_2 = _FakeVmemRef(jnp.ones((8, 4, 2), dtype=jnp.bfloat16))
    kv_utils.pack_new_kv(
        kvc_ref_2,
        kpe_ref_2,
        offset=jnp.int32(2),
        update_sz=jnp.int32(3),
        q_end=jnp.int32(6),
        kv_len=jnp.int32(5),
        bkv_sz=8,
    )

    # 3. pack_new_kv_reference
    kvc_ref_3 = _FakeVmemRef(jnp.ones((8, 4, 4), dtype=jnp.bfloat16))
    kpe_ref_3 = _FakeVmemRef(jnp.ones((8, 4, 2), dtype=jnp.bfloat16))
    kv_utils.pack_new_kv_reference(
        kvc_ref_3,
        kpe_ref_3,
        offset=jnp.int32(2),
        update_sz=jnp.int32(3),
        q_end=jnp.int32(6),
        kv_len=jnp.int32(5),
        bkv_sz=8,
    )


# ---------------------------------------------------------------------------
# 3. transpose.py tests
# ---------------------------------------------------------------------------


def test_transpose_all_functions_and_branches(monkeypatch) -> None:
    # 1. prev_closest_valid_divisor all branches
    assert transpose.prev_closest_valid_divisor(64, 0, multiple_of=8) == 1
    with pytest.raises(ValueError, match="no valid tile size exists"):
        transpose.prev_closest_valid_divisor(16, 8, multiple_of=16)
    assert transpose.prev_closest_valid_divisor(8, 16, multiple_of=16) == 8
    # Fast-path bound divides number and is multiple of multiple_of
    assert transpose.prev_closest_valid_divisor(256, 128, multiple_of=16) == 128
    # Slow-path sympy divisors: bound=96 does not divide 128, falls back to 64
    assert transpose.prev_closest_valid_divisor(128, 96, multiple_of=16) == 64
    # Slow-path error: no divisor of 30 is a multiple of 8
    with pytest.raises(ValueError, match="No divisor of 30"):
        transpose.prev_closest_valid_divisor(30, 20, multiple_of=8)

    # 2. get_reshape_dimension
    assert transpose.get_reshape_dimension((4, 8), (2, 16)) == (2, 16)

    # 3. identity_fn_generator
    class _ScalarArg:
        ndim = 0

        def __init__(self, val: int = 0):
            self.val = val

        def get(self) -> int:
            return self.val

        def set(self, v: int) -> None:
            self.val = v

    s_in = _ScalarArg(42)
    s_out = _ScalarArg(0)
    v_in = np.array([10, 20])
    v_out = np.zeros(2, dtype=np.int64)
    buf_in = np.ones((2, 2))
    buf_out = np.zeros((2, 2))

    ident_fn = transpose.identity_fn_generator(num_scalars=2)
    ident_fn(s_in, v_in, buf_in, s_out, v_out, buf_out)
    assert s_out.val == 42
    np.testing.assert_array_equal(v_out, v_in)
    np.testing.assert_array_equal(buf_out, buf_in)

    # Also test num_scalars=0 (loop with no scalars)
    ident_fn0 = transpose.identity_fn_generator(num_scalars=0)
    ident_fn0()

    # 4. xpose_full, pin_vmem_custom_call, and xpose_pipeline via fake pallas_call
    def _fake_pallas_call(
        kernel_fn,
        *,
        in_specs=None,
        out_specs=None,
        out_shape=None,
        grid=None,
        compiler_params=None,
        name=None,
    ):
        if in_specs and getattr(in_specs[0], "index_map", None) is not None:
            in_specs[0].index_map(0, 1)
        if out_specs and getattr(out_specs[0], "index_map", None) is not None:
            out_specs[0].index_map(0, 1)

        def _runner(inp):
            out_buf = np.zeros(out_shape[0].shape, dtype=np.float32)
            if name == "prefetch":
                return inp
            kernel_fn(np.asarray(inp), out_buf)
            return jnp.asarray(out_buf, dtype=inp.dtype)

        return _runner

    monkeypatch.setattr(transpose.pl, "pallas_call", _fake_pallas_call)

    arr = jnp.ones((32, 64), dtype=jnp.float32)
    res_full = transpose.xpose_full.__wrapped__(arr, transpose_axes=(1, 0))
    assert res_full.shape == (64, 32)

    res_pin = transpose.pin_vmem_custom_call.__wrapped__(arr, num_scalars=0)
    assert res_pin.shape == (32, 64)

    # xpose_pipeline: exact tiles (n_tile > shape clamps to shape)
    res_pipe = transpose.xpose_pipeline.__wrapped__(
        arr, transpose_axes=(1, 0), n_tile=128, m_tile=128
    )
    assert res_pipe.shape == (64, 32)

    # xpose_pipeline: adjusted n_tile and m_tile (triggers both logger.warning branches)
    res_pipe_adj = transpose.xpose_pipeline.__wrapped__(
        arr, transpose_axes=(1, 0), n_tile=24, m_tile=48
    )
    assert res_pipe_adj.shape == (64, 32)

    # xpose_pipeline: defensive non-divisor error branches
    orig_prev = transpose.prev_closest_valid_divisor
    monkeypatch.setattr(
        transpose, "prev_closest_valid_divisor", lambda num, div, multiple_of=1: 7
    )
    with pytest.raises(ValueError, match="No divisor of parallel axis size"):
        transpose.xpose_pipeline.__wrapped__(
            arr, transpose_axes=(1, 0), n_tile=16, m_tile=16
        )

    calls = [0]

    def _second_bad(num, div, multiple_of=1):
        calls[0] += 1
        return 16 if calls[0] == 1 else 7

    monkeypatch.setattr(transpose, "prev_closest_valid_divisor", _second_bad)
    with pytest.raises(ValueError, match="No divisor of pipeline axis size"):
        transpose.xpose_pipeline.__wrapped__(
            arr, transpose_axes=(1, 0), n_tile=16, m_tile=16
        )
    monkeypatch.setattr(transpose, "prev_closest_valid_divisor", orig_prev)


# ---------------------------------------------------------------------------
# 4. dsa_gather.py tests
# ---------------------------------------------------------------------------


class _FakeGatherRef:
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.at = self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self):
        return jnp.dtype(self.arr.dtype)

    def bitcast(self, dt):
        return _FakeGatherRef(self.arr.astype(np.dtype(dt)))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            conv = tuple(self._conv(i) for i in idx)
            return self.arr[conv]
        return self.arr[self._conv(idx)]

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            conv = tuple(self._conv(i) for i in idx)
            self.arr[conv] = np.asarray(val)
        else:
            self.arr[self._conv(idx)] = np.asarray(val)

    @staticmethod
    def _conv(s):
        if hasattr(s, "start") and hasattr(s, "size"):
            start = int(s.start)
            return slice(start, start + int(s.size))
        return s


def test_dsa_gather_and_main_kernel(monkeypatch) -> None:
    fake_sc = SimpleNamespace(num_lanes=4, num_cores=1, num_subcores=1)
    monkeypatch.setattr(
        dsa_gather.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=fake_sc),
    )
    monkeypatch.setattr(
        dsa_gather.plsc,
        "VectorSubcoreMesh",
        lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(dsa_gather.jax.lax, "axis_size", lambda axes: 1)
    monkeypatch.setattr(dsa_gather.lax, "axis_index", lambda axes: 0)
    monkeypatch.setattr(dsa_gather.pl, "program_id", lambda axis: 0)
    monkeypatch.setattr(dsa_gather.pl, "when", lambda cond: (lambda fn: fn() or fn))

    class _FakeCopy:
        def start(self) -> None:
            pass

        def wait(self) -> None:
            pass

    monkeypatch.setattr(
        dsa_gather.pltpu, "make_async_copy", lambda src, dst, sem: _FakeCopy()
    )

    def _fake_emit_pipeline(body_fn, *, grid, in_specs=(), out_specs=(), **kw):
        for spec in in_specs:
            if spec.index_map is not None:
                spec.index_map(0)
        for spec in out_specs:
            if spec.index_map is not None:
                spec.index_map(0)

        def _runner(*args):
            if len(args) == 2:
                body_fn(*args)
            else:
                rope_out_ref = args[-1]
                nope_g = [
                    _FakeGatherRef(np.zeros((4, 128), dtype=np.int32)) for _ in range(4)
                ]
                rope_g = [
                    _FakeGatherRef(np.zeros((4, 128), dtype=np.int32)) for _ in range(4)
                ]
                rope_o = _FakeGatherRef(
                    np.zeros((4, rope_out_ref.shape[1]), dtype=np.int32)
                )
                body_fn(*nope_g, *rope_g, rope_o)

        return _runner

    monkeypatch.setattr(dsa_gather.pltpu, "emit_pipeline", _fake_emit_pipeline)

    def _fake_pl_kernel(kernel_fn, *, out_type, **kwargs):
        def _call(nope_c, rope_c, idx, valid_idx):
            nope_in_ref = _FakeGatherRef(np.asarray(nope_c))
            rope_in_ref = _FakeGatherRef(np.asarray(rope_c))
            idx_ref = _FakeGatherRef(np.asarray(idx))
            valid_ref = _FakeGatherRef(np.asarray(valid_idx))
            nope_out_ref = _FakeGatherRef(np.zeros(out_type[0].shape, dtype=np.uint8))
            rope_out_ref = _FakeGatherRef(np.zeros(out_type[1].shape, dtype=np.uint8))
            sem_ref = _FakeGatherRef(np.zeros((4,), dtype=np.int32))
            kernel_fn(
                nope_in_ref,
                rope_in_ref,
                idx_ref,
                valid_ref,
                nope_out_ref,
                rope_out_ref,
                sem_ref,
            )
            return (
                jnp.asarray(nope_out_ref.arr),
                jnp.asarray(rope_out_ref.arr),
            )

        return _call

    monkeypatch.setattr(dsa_gather.pl, "kernel", _fake_pl_kernel)

    nope_cache = jnp.zeros((2, 4, 4, 128), dtype=jnp.uint8)
    rope_cache = jnp.zeros((2, 1, 4, 128), dtype=jnp.uint8)

    # 1. Exact block_size=128 (out_pad_size == 0) and num_valid_indices=None
    idx_aligned = jnp.zeros((128,), dtype=jnp.int32)
    n_out1, r_out1 = dsa_gather.dsa_gather.__wrapped__(
        nope_cache, rope_cache, idx_aligned, num_valid_indices=None
    )
    assert n_out1.shape == (128, 4, 128)
    assert r_out1.shape == (128, 128)

    # 2. Unaligned out_size=10 (out_pad_size > 0) and explicit num_valid_indices
    idx_unaligned = jnp.zeros((10,), dtype=jnp.int32)
    n_out2, r_out2 = dsa_gather.dsa_gather.__wrapped__(
        nope_cache,
        rope_cache,
        idx_unaligned,
        num_valid_indices=jnp.int32(8),
    )
    assert n_out2.shape == (10, 4, 128)
    assert r_out2.shape == (10, 128)


# ---------------------------------------------------------------------------
# 5. native_sc_gather.py tests
# ---------------------------------------------------------------------------


def test_native_sc_gather_and_kernel(monkeypatch) -> None:
    fake_sc = SimpleNamespace(num_cores=1, num_subcores=1)
    monkeypatch.setattr(
        native_sc_gather.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=fake_sc),
    )
    monkeypatch.setattr(
        native_sc_gather.plsc,
        "VectorSubcoreMesh",
        lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(native_sc_gather.lax, "axis_index", lambda axes: 0)
    monkeypatch.setattr(native_sc_gather.pl, "program_id", lambda axis: 0)

    class _FakeCopy:
        def start(self) -> None:
            pass

        def wait(self) -> None:
            pass

    monkeypatch.setattr(
        native_sc_gather.pltpu,
        "make_async_copy",
        lambda src, dst, sem: _FakeCopy(),
    )

    def _fake_emit_pipeline(body_fn, *, grid, in_specs=(), out_specs=(), **kw):
        for spec in in_specs:
            if spec.index_map is not None:
                spec.index_map(0)
        for spec in out_specs:
            if spec.index_map is not None:
                spec.index_map(0)

        def _runner(*args):
            if len(args) == 1:
                body_fn(args[0])
            else:
                nope_refs = [
                    _FakeGatherRef(np.zeros((16, 128), dtype=np.uint32))
                    for _ in range(4)
                ]
                rope_refs = [
                    _FakeGatherRef(np.zeros((16, 32), dtype=np.uint32))
                    for _ in range(4)
                ]
                rope_out = _FakeGatherRef(np.zeros((16, 128), dtype=np.uint32))
                body_fn(*nope_refs, *rope_refs, rope_out)

        return _runner

    monkeypatch.setattr(native_sc_gather.pltpu, "emit_pipeline", _fake_emit_pipeline)

    def _fake_cm_kernel(kernel_fn, *, out_type, **kwargs):
        def _call(nope_c, rope_c, idx):
            nope_ref = _FakeGatherRef(np.asarray(nope_c))
            rope_ref = _FakeGatherRef(np.asarray(rope_c))
            idx_ref = _FakeGatherRef(np.asarray(idx))
            nope_out_ref = _FakeGatherRef(np.zeros(out_type[0].shape, dtype=np.uint32))
            rope_out_ref = _FakeGatherRef(np.zeros(out_type[1].shape, dtype=np.uint32))
            sem_ref = _FakeGatherRef(np.zeros((4,), dtype=np.int32))
            kernel_fn(
                nope_ref,
                rope_ref,
                idx_ref,
                nope_out_ref,
                rope_out_ref,
                sem_ref,
            )
            return (
                jnp.asarray(nope_out_ref.arr),
                jnp.asarray(rope_out_ref.arr),
            )

        return _call

    monkeypatch.setattr(native_sc_gather.core_map_helper, "kernel", _fake_cm_kernel)

    nope_c = jnp.zeros((64, 128), dtype=jnp.uint32)
    rope_c = jnp.zeros((16, 128), dtype=jnp.uint32)
    idx_valid = jnp.zeros((1, 64), dtype=jnp.int32)

    # Validation error branches
    with pytest.raises(ValueError, match="unknown wait_mode"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            nope_c, rope_c, idx_valid, out_size=64, wait_mode="bad"
        )
    with pytest.raises(ValueError, match="native-SC NOPE must be explicit"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            jnp.zeros((64, 64), dtype=jnp.uint32),
            rope_c,
            idx_valid,
            out_size=64,
        )
    with pytest.raises(ValueError, match="native-SC ROPE must be quarter-major"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            nope_c,
            jnp.zeros((16, 64), dtype=jnp.uint32),
            idx_valid,
            out_size=64,
        )
    with pytest.raises(ValueError, match="indices must be int32"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            nope_c,
            rope_c,
            jnp.zeros((64,), dtype=jnp.int32),
            out_size=64,
        )
    with pytest.raises(ValueError, match="indices has 64 values, expected 128"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            nope_c, rope_c, idx_valid, out_size=128
        )
    with pytest.raises(ValueError, match="atoms_per_batch .* must be > 0"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            nope_c, rope_c, idx_valid, out_size=64, atoms_per_batch=0
        )
    idx_96 = jnp.zeros((1, 96), dtype=jnp.int32)
    with pytest.raises(ValueError, match="out_size .* must be divisible"):
        native_sc_gather.dsa_gather_native_sc.__wrapped__(
            nope_c, rope_c, idx_96, out_size=96, atoms_per_batch=1
        )

    # Valid run 1: atoms_per_batch clamped to max_possible_atoms=1, overlap_rope
    n_out1, r_out1 = native_sc_gather.dsa_gather_native_sc.__wrapped__(
        nope_c,
        rope_c,
        idx_valid,
        out_size=64,
        atoms_per_batch=16,
        wait_mode="overlap_rope",
        serve_pipelined=True,
    )
    assert n_out1.shape == (64, 128)
    assert r_out1.shape == (16, 128)

    # Valid run 2: wait_mode="before_rope", serve_pipelined=False
    n_out2, r_out2 = native_sc_gather.dsa_gather_native_sc.__wrapped__(
        nope_c,
        rope_c,
        idx_valid,
        out_size=64,
        atoms_per_batch=1,
        wait_mode="before_rope",
        serve_pipelined=False,
    )
    assert n_out2.shape == (64, 128)
    assert r_out2.shape == (16, 128)
