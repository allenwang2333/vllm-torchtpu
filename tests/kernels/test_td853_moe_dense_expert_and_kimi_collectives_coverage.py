# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for TD8.53 moe_dense_expert and
kimi_k3 collectives modules."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # type: ignore[missing-import]
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import moe_dense_expert as mde
from vllm_torchtpu.kernels.kimi_k3 import collectives as k3c

pytestmark = pytest.mark.cpu_test


class _FakeCopy:
    def start(self):
        return None

    def wait(self):
        return None


class _FakeSlice:
    def __init__(self, ref: _FakeRef, norm_idx):
        self.ref = ref
        self.norm_idx = norm_idx

    @property
    def shape(self):
        return self.ref.arr[self.norm_idx].shape

    @property
    def dtype(self):
        return self.ref.arr.dtype

    def __getitem__(self, idx):
        sub = self.ref.arr[self.norm_idx]
        if idx is Ellipsis:
            return sub
        return sub[self.ref._norm_idx(idx)]

    def __setitem__(self, idx, val):
        if isinstance(val, _FakeRef):
            val = val.arr
        target = self.ref.arr[self.norm_idx]
        arr_val = jnp.broadcast_to(
            jnp.asarray(val, dtype=self.ref.arr.dtype), target.shape
        )
        self.ref.arr = self.ref.arr.at[self.norm_idx].set(arr_val)


class _FakeAt:
    def __init__(self, ref: _FakeRef):
        self.ref = ref

    def __getitem__(self, idx):
        return _FakeSlice(self.ref, self.ref._norm_idx(idx))


class _FakeRef:
    """Lightweight array-backed reference supporting .at[...] and slicing."""

    def __init__(self, arr):
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def at(self):
        return _FakeAt(self)

    def _norm_idx(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._norm_idx(i) for i in idx)
        if hasattr(idx, "start") and hasattr(idx, "size"):
            start = int(idx.start)
            size = int(idx.size)
            return slice(start, start + size)
        if isinstance(idx, jnp.ndarray) and idx.ndim == 0:
            return int(idx)
        return idx

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.arr
        return self.arr[self._norm_idx(idx)]

    def __setitem__(self, idx, val):
        if isinstance(val, _FakeRef):
            val = val.arr
        if idx is Ellipsis:
            self.arr = jnp.broadcast_to(
                jnp.asarray(val, dtype=self.arr.dtype), self.arr.shape
            )
            return
        norm = self._norm_idx(idx)
        target = self.arr[norm]
        arr_val = jnp.broadcast_to(
            jnp.asarray(val, dtype=self.arr.dtype),
            target.shape if isinstance(target, jnp.ndarray) else (),
        )
        self.arr = self.arr.at[norm].set(arr_val)

    def __array__(self, dtype=None):
        return np.asarray(self.arr, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. moe_dense_expert.py tests
# ---------------------------------------------------------------------------


def test_moe_dense_expert_selectors_and_validation():
    mde._select_expert_buffers.cache_clear()
    n1 = mde._select_expert_buffers(16, 512, 512, 8, 64 * 1024 * 1024)
    assert n1 == 3
    assert mde._select_expert_buffers(16, 512, 512, 8, 64 * 1024 * 1024) == 3
    n2 = mde._select_expert_buffers(
        16, 512, 512, 8, 64 * 1024 * 1024, packing_factor=2, s1_blocks=2, s2_blocks=2
    )
    assert 1 <= n2 <= 3

    w_fp8 = jnp.zeros((2, 512, 1024), dtype=jnp.float8_e4m3fn)
    w2_fp8 = jnp.zeros((2, 512, 512), dtype=jnp.float8_e4m3fn)
    assert mde._packing_factor_for(w_fp8, w2_fp8, None) == 1

    w_fp4 = jnp.zeros((2, 512, 1024), dtype=jnp.float4_e2m1fn)
    w2_fp4 = jnp.zeros((2, 512, 512), dtype=jnp.float4_e2m1fn)
    assert mde._packing_factor_for(w_fp4, w2_fp4, jnp.float4_e2m1fn) == 1

    w_u8 = jnp.zeros((2, 256, 1024), dtype=jnp.uint8)
    w2_u8 = jnp.zeros((2, 256, 512), dtype=jnp.uint8)
    assert mde._packing_factor_for(w_u8, w2_u8, jnp.float4_e2m1fn) == 2
    assert mde._packing_factor_for(w_u8, w2_u8, None) is None

    hidden = jnp.zeros((16, 512), dtype=jnp.bfloat16)
    s1 = jnp.ones((2, 1, 1, 1024), dtype=jnp.float32)
    s2 = jnp.ones((2, 1, 1, 512), dtype=jnp.float32)

    assert mde.can_use_dense_expert(
        hidden,
        w_fp8,
        w2_fp8,
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Unsupported activation
    assert not mde.can_use_dense_expert(
        hidden,
        w_fp8,
        w2_fp8,
        s1,
        s2,
        None,
        None,
        activation="gelu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Bias present or scales missing
    assert not mde.can_use_dense_expert(
        hidden,
        w_fp8,
        w2_fp8,
        s1,
        s2,
        jnp.zeros((1,)),
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Rank mismatch
    assert not mde.can_use_dense_expert(
        hidden[0],
        w_fp8,
        w2_fp8,
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Unsupported packing factor
    assert not mde.can_use_dense_expert(
        hidden,
        w_u8,
        w2_u8,
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Zero dimension
    assert not mde.can_use_dense_expert(
        jnp.zeros((0, 512), dtype=jnp.bfloat16),
        w_fp8,
        w2_fp8,
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Unaligned m for FP8
    assert not mde.can_use_dense_expert(
        jnp.zeros((4, 512), dtype=jnp.bfloat16),
        w_fp8,
        w2_fp8,
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Out != h
    assert not mde.can_use_dense_expert(
        hidden,
        w_fp8,
        jnp.zeros((2, 512, 256), dtype=jnp.float8_e4m3fn),
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # Non-bfloat16 hidden
    assert not mde.can_use_dense_expert(
        hidden.astype(jnp.float32),
        w_fp8,
        w2_fp8,
        s1,
        s2,
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    # FP4 VMEM budget exceeded (> _MAX_DENSE_EXPERT_VMEM_BYTES)
    assert not mde.can_use_dense_expert(
        jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 2048, 8192), jnp.uint8),
        jax.ShapeDtypeStruct((2, 2048, 4096), jnp.uint8),
        jax.ShapeDtypeStruct((2, 8, 1, 8192), jnp.float32),
        jax.ShapeDtypeStruct((2, 8, 1, 4096), jnp.float32),
        None,
        None,
        activation="silu",
        rhs_quant_dtype=jnp.float4_e2m1fn,
        skip_padded_tokens=False,
    )


def test_moe_dense_expert_quantize_activations_and_kernel(monkeypatch):
    gate = jnp.ones((2, 4), dtype=jnp.bfloat16)
    up = jnp.ones((2, 4), dtype=jnp.bfloat16)
    for act in ("silu", "silu_and_mul_with_clamp", "swigluoai"):
        res = mde._apply_activation(gate, up, act)
        assert res.shape == (2, 4)
    with pytest.raises(NotImplementedError, match="Unsupported activation"):
        mde._apply_activation(gate, up, "relu")

    # _quantize with 0 width (0 loop iterations) and 512 width
    q0 = jnp.zeros((2, 0), dtype=jnp.float8_e4m3fn)
    s0 = jnp.zeros((0, 2, 128), dtype=jnp.bfloat16)
    mde._quantize(jnp.zeros((2, 0), dtype=jnp.bfloat16), _FakeRef(q0), _FakeRef(s0))

    # Exercise _kernel directly for both num_buffers=1 (FP8, s1_blocks=1, s2_blocks=1)
    # and num_buffers=2 (packed FP4, s1_blocks=1, s2_blocks=1 + block-scaled)
    monkeypatch.setattr(mde.pl, "when", lambda _cond: lambda fn: fn())
    monkeypatch.setattr(
        mde.pltpu,
        "make_async_copy",
        lambda src, dst, sem: _FakeCopy(),
    )
    monkeypatch.setattr(
        mde.pltpu,
        "bitcast",
        lambda rhs, dt: jnp.repeat(rhs, 2, axis=0).astype(dt),
    )

    def _fori_loop(lo, hi, body, init):
        val = init
        for idx in range(int(lo), int(hi)):
            val = body(jnp.int32(idx), val)
        return val

    monkeypatch.setattr(mde.lax, "fori_loop", _fori_loop)

    m, h, i, experts = 16, 512, 512, 2
    active_ref = _FakeRef(jnp.array([1, 0], dtype=jnp.int32))
    x_ref = _FakeRef(jnp.ones((m, h), dtype=jnp.bfloat16))
    coeff_ref = _FakeRef(jnp.ones((m, 128), dtype=jnp.float32))
    w1_hbm = _FakeRef(jnp.ones((experts, h, 2 * i), dtype=jnp.float8_e4m3fn))
    w2_hbm = _FakeRef(jnp.ones((experts, i, h), dtype=jnp.float8_e4m3fn))
    s1_hbm = _FakeRef(jnp.ones((experts, 1, 1, 2 * i), dtype=jnp.float32))
    s2_hbm = _FakeRef(jnp.ones((experts, 1, 1, h), dtype=jnp.float32))
    out_ref = _FakeRef(jnp.zeros((m, h), dtype=jnp.bfloat16))
    w1_ref = _FakeRef(jnp.ones((1, h, 2 * i), dtype=jnp.float8_e4m3fn))
    w2_ref = _FakeRef(jnp.ones((1, i, h), dtype=jnp.float8_e4m3fn))
    s1_ref = _FakeRef(jnp.ones((1, 1, 2 * i), dtype=jnp.float32))
    s2_ref = _FakeRef(jnp.ones((1, 1, h), dtype=jnp.float32))
    xq_ref = _FakeRef(jnp.zeros((m, h), dtype=jnp.float8_e4m3fn))
    xs_ref = _FakeRef(jnp.zeros((h // 512, m, 128), dtype=jnp.bfloat16))
    a_ref = _FakeRef(jnp.zeros((m, i), dtype=jnp.bfloat16))
    aq_ref = _FakeRef(jnp.zeros((m, i), dtype=jnp.float8_e4m3fn))
    asc_ref = _FakeRef(jnp.zeros((i // 512, m, 128), dtype=jnp.bfloat16))
    owner_ref = _FakeRef(jnp.zeros((m, h), dtype=jnp.float32))
    active_ids_ref = _FakeRef(jnp.zeros((experts,), dtype=jnp.int32))
    sem_ref = _FakeRef(jnp.zeros((2, 4), dtype=jnp.int32))

    # Case A: num_buffers=1, packing_factor=1, FP8
    mde._kernel(
        active_ref,
        x_ref,
        coeff_ref,
        w1_hbm,
        w2_hbm,
        s1_hbm,
        s2_hbm,
        out_ref,
        w1_ref,
        w2_ref,
        s1_ref,
        s2_ref,
        xq_ref,
        xs_ref,
        a_ref,
        aq_ref,
        asc_ref,
        owner_ref,
        active_ids_ref,
        sem_ref,
        m=m,
        h=h,
        i=i,
        experts=experts,
        num_buffers=1,
        packing_factor=1,
        s1_blocks=1,
        s2_blocks=1,
        activation="silu",
        rhs_quant_dtype=None,
    )

    # Case B: num_buffers=2, packing_factor=2, s1_blocks=2, s2_blocks=2, FP4
    w1_ref2 = _FakeRef(jnp.ones((2, h // 2, 2 * i), dtype=jnp.uint8))
    w2_ref2 = _FakeRef(jnp.ones((2, i // 2, h), dtype=jnp.uint8))
    s1_ref2 = _FakeRef(jnp.ones((2, 2, 1, 2 * i), dtype=jnp.float32))
    s2_ref2 = _FakeRef(jnp.ones((2, 2, 1, h), dtype=jnp.float32))
    mde._kernel(
        active_ref,
        x_ref,
        coeff_ref,
        w1_hbm,
        w2_hbm,
        s1_hbm,
        s2_hbm,
        out_ref,
        w1_ref2,
        w2_ref2,
        s1_ref2,
        s2_ref2,
        xq_ref,
        xs_ref,
        a_ref,
        aq_ref,
        asc_ref,
        owner_ref,
        active_ids_ref,
        sem_ref,
        m=m,
        h=h,
        i=i,
        experts=experts,
        num_buffers=2,
        packing_factor=2,
        s1_blocks=2,
        s2_blocks=2,
        activation="silu",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )

    # Case C: h=0, i=0, experts=0 to cover 0-iteration loop branches inside compute
    mde._kernel(
        _FakeRef(jnp.array([1], dtype=jnp.int32)),
        _FakeRef(jnp.zeros((m, 0), dtype=jnp.bfloat16)),
        coeff_ref,
        w1_hbm,
        w2_hbm,
        s1_hbm,
        s2_hbm,
        _FakeRef(jnp.zeros((m, 0), dtype=jnp.bfloat16)),
        w1_ref,
        w2_ref,
        s1_ref,
        s2_ref,
        _FakeRef(jnp.zeros((m, 0), dtype=jnp.float8_e4m3fn)),
        _FakeRef(jnp.zeros((0, m, 128), dtype=jnp.bfloat16)),
        _FakeRef(jnp.zeros((m, 0), dtype=jnp.bfloat16)),
        _FakeRef(jnp.zeros((m, 0), dtype=jnp.float8_e4m3fn)),
        _FakeRef(jnp.zeros((0, m, 128), dtype=jnp.bfloat16)),
        _FakeRef(jnp.zeros((m, 0), dtype=jnp.float32)),
        active_ids_ref,
        sem_ref,
        m=m,
        h=0,
        i=0,
        experts=1,
        num_buffers=1,
    )


def test_dense_expert_moe_entry(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(
        mde.pl,
        "pallas_call",
        lambda fn, *, out_shape, **kw: lambda *args: jnp.zeros(
            out_shape.shape, dtype=out_shape.dtype
        ),
    )

    # 1. Padded m (orig_m=4 -> pad_m=12) and padded experts (experts=2), s1_blocks=1
    out1 = mde.dense_expert_moe.__wrapped__(
        jnp.ones((4, 512), dtype=jnp.bfloat16),
        jnp.ones((2, 256, 1024), dtype=jnp.uint8),
        jnp.ones((2, 256, 512), dtype=jnp.uint8),
        jnp.ones((2, 1, 1, 1024), dtype=jnp.float32),
        jnp.ones((2, 1, 1, 512), dtype=jnp.float32),
        jnp.ones((4, 2), dtype=jnp.float32),
        jnp.zeros((4, 2), dtype=jnp.int32),
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    assert out1.shape == (4, 512)

    # 2. Unpadded m (orig_m=16) and unpadded experts (experts=128), s1_blocks=2
    out2 = mde.dense_expert_moe.__wrapped__(
        jnp.ones((16, 512), dtype=jnp.bfloat16),
        jnp.ones((128, 512, 1024), dtype=jnp.float8_e4m3fn),
        jnp.ones((128, 512, 512), dtype=jnp.float8_e4m3fn),
        jnp.ones((128, 2, 1, 1024), dtype=jnp.float32),
        jnp.ones((128, 2, 1, 512), dtype=jnp.float32),
        jnp.ones((16, 0), dtype=jnp.float32),
        jnp.zeros((16, 0), dtype=jnp.int32),
    )
    assert out2.shape == (16, 512)


# ---------------------------------------------------------------------------
# 2. kimi_k3/collectives.py tests
# ---------------------------------------------------------------------------


def _make_32_devices():
    devices = []
    dev_id = 0
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1, 2, 3):
                for core in (0, 1):
                    devices.append(
                        SimpleNamespace(
                            id=dev_id,
                            coords=(x, y, z),
                            core_on_chip=core,
                        )
                    )
                    dev_id += 1
    return devices


def test_k3_mla_chip_layout_and_ring_tables_and_regions():
    devices = _make_32_devices()
    owners = list(reversed(range(32)))
    pairs, parity, token_order = k3c.mla_chip_layout(devices, owners)
    assert len(pairs) == 16
    assert len(parity) == 2
    assert len(token_order) == 32

    with pytest.raises(ValueError, match="sixteen two-core chips"):
        k3c.mla_chip_layout(devices[:4], owners[:4])

    tables = k3c.ring_tables(devices, owners)
    assert tables.shape == (3, 3, 32)

    with pytest.raises(ValueError, match="32 devices and a token-owner permutation"):
        k3c.ring_tables(devices[:16], list(range(16)))
    bad_id_devs = _make_32_devices()
    bad_id_devs[0].id, bad_id_devs[1].id = 99, 0
    with pytest.raises(ValueError, match="ascending device IDs"):
        k3c.ring_tables(bad_id_devs, owners)

    bad_slice_devs = _make_32_devices()
    for d in bad_slice_devs:
        d.coords = (0, d.coords[1], d.coords[2])
    with pytest.raises(ValueError, match="physical 2x2x4 slice"):
        k3c.ring_tables(bad_slice_devs, owners)

    bad_core_devs = _make_32_devices()
    bad_core_devs[0].core_on_chip = 2
    with pytest.raises(ValueError, match="two cores per chip"):
        k3c.ring_tables(bad_core_devs, owners)

    dup_core_devs = _make_32_devices()
    dup_core_devs[1].core_on_chip = 0
    with pytest.raises(ValueError, match="Duplicate physical cores"):
        k3c.ring_tables(dup_core_devs, owners)

    k3c._regions.cache_clear()
    with pytest.raises(ValueError, match="64-row ownership and 128-column alignment"):
        k3c._regions(32, 384)
    reg = k3c._regions(64, 384)
    assert len(reg) == 12
    assert k3c._regions(64, 384) == reg  # cache hit


def test_k3_mla_preprocess_and_gather_moe_inputs(monkeypatch):
    devices = _make_32_devices()
    owners = list(range(32))
    pairs, _, _ = k3c.mla_chip_layout(devices, owners)
    mesh = SimpleNamespace(
        devices=np.array(devices, dtype=object),
        axis_names=("k3_sp",),
        size=32,
    )

    monkeypatch.setattr(k3c.jax, "shard_map", lambda fn, **kw: lambda *args: fn(*args))
    monkeypatch.setattr(
        k3c.jax.lax,
        "all_gather",
        lambda x, axis_name, *, axis=0, **kw: jnp.repeat(
            x,
            2
            if kw.get("axis_index_groups") and len(kw["axis_index_groups"][0]) == 16
            else (2 if kw.get("axis_index_groups") else 32),
            axis=axis,
        ),
    )
    monkeypatch.setattr(
        k3c.jax.lax,
        "all_to_all",
        lambda g, axis_name, **kw: jnp.zeros(
            (32 * (g.shape[0] // 2), 384), dtype=g.dtype
        ),
    )

    w_gate = jnp.ones((4, 2500), dtype=jnp.bfloat16)
    packed_g = k3c.pack_mla_gate(w_gate, mesh, pairs)
    assert packed_g.shape[0] == 4

    x = jnp.ones((2, 64), dtype=jnp.bfloat16)
    weight = jnp.ones((64, 2496), dtype=jnp.bfloat16)
    packed_gate = jnp.ones((64, 6144), dtype=jnp.bfloat16)
    qnorm = jnp.ones((1536,), dtype=jnp.bfloat16)
    kvnorm = jnp.ones((512,), dtype=jnp.bfloat16)

    q_sp, kv_sp, rope_sp, gate_sp = k3c.mla_preprocess(
        x,
        weight,
        packed_gate,
        qnorm,
        kvnorm,
        mesh,
        owners,
        sequence_parallel=True,
        eps=1e-6,
    )
    assert q_sp.shape == (64, 1536)
    assert kv_sp.shape == (64, 512)
    assert rope_sp.shape == (64, 64)
    assert gate_sp.shape == (64, 384)

    q_nsp, kv_nsp, rope_nsp, gate_nsp = k3c.mla_preprocess(
        x,
        weight,
        packed_gate,
        qnorm,
        kvnorm,
        mesh,
        owners,
        sequence_parallel=False,
        eps=1e-6,
    )
    assert q_nsp.shape == (2, 1536)
    assert gate_nsp.shape == (2, 384)

    # gather_moe_inputs: test both padded (width != 128 multiple) and unpadded (128)
    monkeypatch.setattr(k3c, "ring_collective", lambda packed, *a, **kw: packed)
    x_pad = jnp.ones((4, 16), dtype=jnp.bfloat16)
    w_pad = jnp.ones((4, 2), dtype=jnp.bfloat16)
    ids_pad = jnp.array([[1, -1], [258, 0], [3, 4], [5, 6]], dtype=jnp.int32)
    gx, gw, gids = k3c.gather_moe_inputs(
        x_pad, w_pad, ids_pad, mesh, None, collective_id=0
    )
    assert gx.shape == (4, 16)
    assert gw.shape == (4, 2)
    np.testing.assert_array_equal(np.asarray(gids), np.asarray(ids_pad))

    x_exact = jnp.ones((4, 116), dtype=jnp.bfloat16)
    w_exact = jnp.ones((4, 4), dtype=jnp.bfloat16)
    ids_exact = jnp.zeros((4, 4), dtype=jnp.int32)
    gx2, _, _ = k3c.gather_moe_inputs(
        x_exact, w_exact, ids_exact, mesh, None, collective_id=0
    )
    assert gx2.shape == (4, 116)


def test_k3_ring_collective_project_rs_and_gather_project_wrappers(monkeypatch):
    devices = _make_32_devices()
    owners = list(range(32))
    tables = k3c.ring_tables(devices, owners)
    mesh = SimpleNamespace(axis_names=("k3_sp",), size=32)

    with pytest.raises(ValueError, match="flat 32-device k3_sp mesh"):
        k3c.ring_collective(
            jnp.ones((64, 384), dtype=jnp.bfloat16),
            SimpleNamespace(axis_names=("tp",), size=32),
            tables,
            gather=True,
            collective_id=1,
        )
    with pytest.raises(ValueError, match="transfer and reduce BF16"):
        k3c.ring_collective(
            jnp.ones((64, 384), dtype=jnp.float32),
            mesh,
            tables,
            gather=True,
            collective_id=1,
        )
    with pytest.raises(ValueError, match="Invalid token or partial-shard count"):
        k3c.ring_collective(
            jnp.ones((16, 384), dtype=jnp.bfloat16),
            mesh,
            tables,
            gather=True,
            collective_id=1,
        )

    def _fake_shard_map(fn, *, in_specs=None, out_specs=None, **kw):
        def _wrapped(*args):
            local_args = [
                a[:1] if (a.ndim == 3 and a.shape[0] == 32) else a for a in args
            ]
            res = fn(*local_args)
            if (
                out_specs is not None
                and len(out_specs) == 3
                and res.ndim == 3
                and res.shape[0] == 1
            ):
                return jnp.repeat(res, 32, axis=0)
            return res

        return _wrapped

    monkeypatch.setattr(k3c.jax, "shard_map", _fake_shard_map)
    monkeypatch.setattr(
        k3c.lax,
        "all_gather",
        lambda part, axis_name, **kw: part,
    )
    monkeypatch.setattr(
        k3c.lax,
        "psum_scatter",
        lambda ordered, axis_name, **kw: ordered[: ordered.shape[0] // 32],
    )
    monkeypatch.setattr(
        k3c.pl,
        "pallas_call",
        lambda kernel, *, out_shape, **kw: lambda *args: jnp.zeros(
            out_shape.shape, dtype=out_shape.dtype
        ),
    )

    # XLA fallback (rows % 64 != 0) for gather=True and gather=False
    ag_fb = k3c.ring_collective(
        jnp.ones((32 * 32, 384), dtype=jnp.bfloat16),
        mesh,
        tables,
        gather=True,
        collective_id=1,
    )
    assert ag_fb.shape == (32 * 32, 384)

    rs_fb = k3c.ring_collective(
        jnp.ones((32, 32 * 32, 384), dtype=jnp.bfloat16),
        mesh,
        tables,
        gather=False,
        collective_id=1,
    )
    assert rs_fb.shape == (32, 384)

    # Pallas ring path (rows == 64) for gather=True and gather=False
    ag_pl = k3c.ring_collective(
        jnp.ones((32 * 64, 384), dtype=jnp.bfloat16),
        mesh,
        tables,
        gather=True,
        collective_id=1,
    )
    assert ag_pl.shape == (32 * 64, 384)

    rs_pl = k3c.ring_collective(
        jnp.ones((32, 32 * 64, 384), dtype=jnp.bfloat16),
        mesh,
        tables,
        gather=False,
        collective_id=1,
    )
    assert rs_pl.shape == (64, 384)

    # project_reduce_scatter validation, fallback, and Pallas path
    with pytest.raises(ValueError, match="SP32 and 3 local heads"):
        k3c.project_reduce_scatter(
            jnp.ones((16, 64, 384), dtype=jnp.bfloat16),
            jnp.ones((32, 384, 7168), dtype=jnp.bfloat16),
            mesh,
            tables,
            collective_id=2,
        )
    with pytest.raises(ValueError, match="requires BF16 operands"):
        k3c.project_reduce_scatter(
            jnp.ones((32, 64, 384), dtype=jnp.float32),
            jnp.ones((32, 384, 7168), dtype=jnp.bfloat16),
            mesh,
            tables,
            collective_id=2,
        )

    prs_fb = k3c.project_reduce_scatter(
        jnp.ones((32, 32 * 32, 384), dtype=jnp.bfloat16),
        jnp.ones((32, 384, 7168), dtype=jnp.bfloat16),
        mesh,
        tables,
        collective_id=2,
    )
    assert prs_fb.shape == (32, 7168)

    prs_pl = k3c.project_reduce_scatter(
        jnp.ones((32, 32 * 64, 384), dtype=jnp.bfloat16),
        jnp.ones((32, 384, 7168), dtype=jnp.bfloat16),
        mesh,
        tables,
        collective_id=2,
    )
    assert prs_pl.shape == (64, 7168)

    # gather_project fallback (x.shape[0] % 128 != 0) and Pallas (x.shape[0] == 128)
    gp_fb = k3c.gather_project(
        jnp.ones((64, 7168), dtype=jnp.bfloat16),
        jnp.ones((1, 7168, 1792), dtype=jnp.bfloat16),
        mesh,
        tables,
        collective_id=3,
    )
    assert gp_fb.shape == (32, 64, 1792)

    gp_pl = k3c.gather_project(
        jnp.ones((128, 7168), dtype=jnp.bfloat16),
        jnp.ones((1, 7168, 1792), dtype=jnp.bfloat16),
        mesh,
        tables,
        collective_id=3,
    )
    assert gp_pl.shape == (32, 32 * 128, 1792)


def test_k3_pallas_ring_kernels_direct(monkeypatch):
    monkeypatch.setattr(k3c.pl, "program_id", lambda axis: jnp.int32(0))
    monkeypatch.setattr(k3c.lax, "axis_index", lambda axis_name: jnp.int32(0))
    monkeypatch.setattr(k3c.pl, "when", lambda _cond: lambda fn: fn())
    monkeypatch.setattr(
        k3c.pl,
        "run_scoped",
        lambda fn, **kw: fn(_FakeRef(jnp.zeros((1,), dtype=jnp.int32))),
    )
    monkeypatch.setattr(
        k3c.pltpu,
        "get_barrier_semaphore",
        lambda: _FakeRef(jnp.zeros((1,), dtype=jnp.int32)),
    )
    monkeypatch.setattr(k3c.pl, "semaphore_signal", lambda *a, **kw: None)
    monkeypatch.setattr(k3c.pl, "semaphore_wait", lambda *a, **kw: None)
    monkeypatch.setattr(k3c.pltpu, "make_async_copy", lambda *a, **kw: _FakeCopy())
    monkeypatch.setattr(k3c.pltpu, "async_copy", lambda *a, **kw: _FakeCopy())
    monkeypatch.setattr(
        k3c.pltpu, "make_async_remote_copy", lambda *a, **kw: _FakeCopy()
    )

    tables_ref = _FakeRef(jnp.zeros((3, 3, 32), dtype=jnp.int32))
    sem_ref = _FakeRef(jnp.zeros((4, 12), dtype=jnp.int32))

    # 1. _reduce_scatter_kernel with 4-tuple and 2-tuple streams, plus empty streams
    x_rs = _FakeRef(jnp.zeros((32 * 64, 128), dtype=jnp.bfloat16))
    o_rs = _FakeRef(jnp.zeros((64, 128), dtype=jnp.bfloat16))
    local_buf = _FakeRef(jnp.zeros((64, 128), dtype=jnp.bfloat16))
    payload = _FakeRef(jnp.zeros((64, 128), dtype=jnp.bfloat16))
    received = _FakeRef(jnp.zeros((2, 64, 128), dtype=jnp.bfloat16))
    k3c._reduce_scatter_kernel(
        tables_ref,
        x_rs,
        o_rs,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        local_buf,
        payload,
        received,
        streams=((0, 16, 0, 128), (16, 16)),
        buffers=2,
    )
    k3c._reduce_scatter_kernel(
        tables_ref,
        x_rs,
        o_rs,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        local_buf,
        payload,
        received,
        streams=(),
        buffers=2,
    )

    # 2. _all_gather_kernel with 2 regions and empty regions
    k3c._all_gather_kernel(
        tables_ref,
        o_rs,
        x_rs,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        payload,
        received,
        regions=((0, 16, 0, 128), (16, 16, 0, 128)),
        buffers=2,
    )
    k3c._all_gather_kernel(
        tables_ref,
        o_rs,
        x_rs,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        payload,
        received,
        regions=(),
        buffers=2,
    )

    # 3. _project_ring_kernel with 4-tuple and 2-tuple streams, plus empty streams
    w_proj = _FakeRef(jnp.zeros((32, 128), dtype=jnp.bfloat16))
    inputs_proj = _FakeRef(jnp.zeros((2, 16, 32), dtype=jnp.bfloat16))
    k3c._project_ring_kernel(
        tables_ref,
        _FakeRef(jnp.zeros((32 * 64, 32), dtype=jnp.bfloat16)),
        w_proj,
        o_rs,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        local_buf,
        payload,
        received,
        w_proj,
        inputs_proj,
        sem_ref,
        streams=((0, 16, 0, 128), (16, 16)),
        buffers=2,
    )
    k3c._project_ring_kernel(
        tables_ref,
        _FakeRef(jnp.zeros((32 * 64, 32), dtype=jnp.bfloat16)),
        w_proj,
        o_rs,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        local_buf,
        payload,
        received,
        w_proj,
        inputs_proj,
        sem_ref,
        streams=(),
        buffers=2,
    )

    # 4. _gather_project_kernel with 2 regions and empty regions
    gp_x = _FakeRef(jnp.zeros((128, 128), dtype=jnp.bfloat16))
    gp_w = _FakeRef(jnp.zeros((128, 64), dtype=jnp.bfloat16))
    gp_out = _FakeRef(jnp.zeros((32 * 128, 64), dtype=jnp.bfloat16))
    gp_payload = _FakeRef(jnp.zeros((128, 128), dtype=jnp.bfloat16))
    gp_received = _FakeRef(jnp.zeros((2, 128, 128), dtype=jnp.bfloat16))
    gp_acc = _FakeRef(jnp.zeros((4096, 64), dtype=jnp.float32))
    gp_out_bufs = _FakeRef(jnp.zeros((2, 128, 64), dtype=jnp.bfloat16))
    k3c._gather_project_kernel(
        tables_ref,
        gp_x,
        gp_w,
        gp_out,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        gp_payload,
        gp_received,
        gp_w,
        gp_acc,
        gp_out_bufs,
        sem_ref,
        sem_ref,
        regions=((0, 64, 0, 128), (64, 64, 0, 128)),
    )
    k3c._gather_project_kernel(
        tables_ref,
        gp_x,
        gp_w,
        gp_out,
        sem_ref,
        sem_ref,
        sem_ref,
        sem_ref,
        gp_payload,
        gp_received,
        gp_w,
        gp_acc,
        gp_out_bufs,
        sem_ref,
        sem_ref,
        regions=(),
    )
