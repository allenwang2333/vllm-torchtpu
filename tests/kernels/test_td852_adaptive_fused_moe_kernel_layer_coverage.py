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
"""CPU unit tests and 100% coverage for Adaptive Fused MoE kernel & layer (TD8.52)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    host,
    kernel,
    layer,
)


class _FakeChip:
    num_lanes = 128
    generation = host.MIN_GENERATION

    def __init__(self, vmem_capacity_bytes=128 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


class _FakeCopy:
    def start(self, priority=None):
        return None

    def wait(self):
        return None


class _FakeAt:
    def __init__(self, ref: "_FakeRef"):
        self._ref = ref

    def __getitem__(self, idx):
        return self._ref


class _FakeRef:
    """Lightweight CPU Pallas Ref stand-in supporting .at, .bitcast, and indexing."""

    def __init__(self, shape, dtype=jnp.float32, allow_bitcast=True):
        self.shape = tuple(shape)
        try:
            self.dtype = jnp.dtype(dtype)
        except TypeError:
            self.dtype = jnp.dtype(jnp.int32)
        self._allow_bitcast = allow_bitcast

    def __getattribute__(self, name):
        if name == "bitcast" and not object.__getattribute__(self, "_allow_bitcast"):
            raise AttributeError("bitcast not available")
        return object.__getattribute__(self, name)

    @property
    def at(self):
        return _FakeAt(self)

    def bitcast(self, new_dtype):
        old_bits = max(4, self.dtype.itemsize * 8)
        if self.dtype == jnp.float4_e2m1fn:
            old_bits = 4
        new_dt = jnp.dtype(new_dtype)
        new_bits = 4 if new_dt == jnp.float4_e2m1fn else new_dt.itemsize * 8
        if len(self.shape) >= 2:
            new_k = (self.shape[-2] * old_bits) // new_bits
            new_shape = (*self.shape[:-2], new_k, self.shape[-1])
        else:
            new_shape = self.shape
        return _FakeRef(new_shape, new_dt)

    def _slice_shape(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        out_dims = []
        s_idx = 0
        for item in idx:
            if s_idx >= len(self.shape):
                break
            dim = self.shape[s_idx]
            if isinstance(item, pl.Slice):
                sz = item.size
                out_dims.append(int(sz) if isinstance(sz, int) else dim)
                s_idx += 1
            elif isinstance(item, slice):
                start, stop, _ = item.indices(dim)
                out_dims.append(max(0, stop - start))
                s_idx += 1
            else:
                # Scalar index drops the axis
                s_idx += 1
        out_dims.extend(self.shape[s_idx:])
        return tuple(out_dims)

    def __getitem__(self, idx):
        sub_shape = self._slice_shape(idx)
        if not sub_shape:
            return jnp.zeros((), dtype=self.dtype)
        return jnp.zeros(sub_shape, dtype=self.dtype)

    def __setitem__(self, idx, val):
        return None


@pytest.fixture(autouse=True)
def _mock_tpu_and_caches(monkeypatch):
    chip = _FakeChip()
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: chip)
    monkeypatch.setattr(
        pltpu,
        "roll",
        lambda x, shift, axis: jnp.roll(x, shift, axis=axis),
    )
    monkeypatch.setattr(pltpu, "get_barrier_semaphore", lambda: _FakeRef((1,)))
    monkeypatch.setattr(
        pl,
        "semaphore_signal",
        lambda sem, inc=1, device_id=None, device_id_type=None: None,
    )
    monkeypatch.setattr(pl, "semaphore_wait", lambda sem, val: None)

    def _fake_run_scoped(fn=None, **kw):
        if fn is None:
            return lambda f: _fake_run_scoped(f, **kw)
        fn(*(_FakeRef((1,)) for _ in kw))
        return None

    monkeypatch.setattr(pl, "run_scoped", _fake_run_scoped)
    monkeypatch.setattr(
        pltpu,
        "make_async_copy",
        lambda src_ref, dst_ref, sem: _FakeCopy(),
    )
    monkeypatch.setattr(
        pltpu,
        "make_async_remote_copy",
        lambda src_ref, dst_ref, send_sem, recv_sem, device_id, device_id_type: (
            _FakeCopy()
        ),
    )

    def _fake_bitcast(x, target_dtype):
        dt = jnp.dtype(target_dtype)
        # u32 [qb/8, N] -> fp4 [qb, N]
        new_shape = (*x.shape[:-2], x.shape[-2] * host.PACK4, x.shape[-1])
        return jnp.zeros(new_shape, dtype=dt)

    monkeypatch.setattr(pltpu, "bitcast", _fake_bitcast)
    monkeypatch.setattr(pl, "when", lambda cond: lambda fn: (fn(), fn)[1])
    monkeypatch.setattr(
        lax,
        "fori_loop",
        lambda lo, hi, body, init: body(jnp.int32(0), init),
    )
    kernel._BUILD_CACHE.clear()
    kernel._COMBINE_CACHE.clear()
    layer._LAYER_SM_CACHE.clear()
    kernel.tile_height_rungs.cache_clear()
    kernel.combine_step_tokens.cache_clear()
    layer._exchange_layout.cache_clear()
    layer._validate_mesh_ep_ranks.cache_clear()


def test_math_and_ffn_helpers():
    gate = jnp.ones((8, 16), dtype=jnp.float32)
    up = jnp.ones((8, 16), dtype=jnp.float32) * 2.0
    assert kernel._apply_act(gate, up, "silu").shape == (8, 16)
    assert kernel._apply_act(gate, up, "swigluoai").shape == (8, 16)
    with pytest.raises(NotImplementedError, match="fuses"):
        kernel._apply_act(gate, up, "gelu")

    # tile_height_rungs
    assert kernel.tile_height_rungs(512, 1024, 8, 8) == (128, 256, 384, 512)
    assert kernel.tile_height_rungs(512, 1024, 8, 8) == (128, 256, 384, 512)
    assert kernel.tile_height_rungs(6, 1024, 8, 8) == (6,)
    assert kernel.tile_height_rungs(512, 10**9, 1, 1) == (512,)

    # rowquant_fp8 and wire_row_and_scale
    x = jnp.array([[0.0, 0.0], [1.0, -2.0], [jnp.inf, 1.0]], dtype=jnp.bfloat16)
    q, s = kernel.rowquant_fp8(x)
    assert q.shape == (3, 2) and s.shape == (3, 1)

    acc2 = jnp.ones((8, 16), dtype=jnp.float32)
    mid_s = jnp.ones((8, 1), dtype=jnp.float32) * 0.5
    w2s = jnp.ones((1, 16), dtype=jnp.float32) * 0.25
    w2b = jnp.ones((1, 16), dtype=jnp.float32) * 0.1
    wr_fp8, ws_fp8 = kernel.wire_row_and_scale(
        acc2, mid_s, w2s=w2s, w2b=w2b, wire_dtype=host.FP8
    )
    assert wr_fp8.dtype == host.FP8 and ws_fp8.shape == (8, 1)
    wr_bf16, ws_bf16 = kernel.wire_row_and_scale(
        acc2, None, wire_dtype=jnp.bfloat16, tile_m=8
    )
    assert wr_bf16.dtype == jnp.bfloat16 and ws_bf16.shape == (8, 1)
    wr_bf16_auto, ws_bf16_auto = kernel.wire_row_and_scale(
        acc2, None, wire_dtype=jnp.bfloat16, tile_m=None
    )
    assert wr_bf16_auto.shape == (8, 16) and ws_bf16_auto.shape == (8, 1)

    # _intermediate_rows with multiple chunks (inter > QCHUNK = 512)
    inter_big = kernel.QCHUNK + 128
    acc1 = jnp.ones((4, 2 * inter_big), dtype=jnp.float32)
    act_scale = jnp.ones((4, 1), dtype=jnp.float32)
    w1s = jnp.ones((1, 2 * inter_big), dtype=jnp.float32)
    w1b = jnp.ones((1, 2 * inter_big), dtype=jnp.float32)
    mid_q, mid_scale = kernel._intermediate_rows(
        acc1,
        act_scale,
        inter_big,
        w1s=w1s,
        act_fn="silu",
        w1b=w1b,
        requantize=True,
    )
    assert mid_q.shape == (4, inter_big) and mid_scale.shape == (4, 1)

    # expert_ffn_fp8, expert_ffn_blockscale, expert_ffn_bf16, expert_ffn_int8
    act_q = jnp.ones((4, 16), dtype=host.FP8)
    act_sc = jnp.ones((4, 1), dtype=jnp.float32)
    w1_fp8 = jnp.ones((16, 16), dtype=host.FP8)
    w2_fp8 = jnp.ones((8, 16), dtype=host.FP8)
    a2, ms = kernel.expert_ffn_fp8(
        act_q, act_sc, w1_fp8, lambda: w2_fp8, jnp.ones((1, 16), dtype=jnp.float32)
    )
    assert a2.shape == (4, 16) and ms.shape == (4, 1)

    # blockscale with 2 blocks along both K dimensions
    a2_bs, ms_bs = kernel.expert_ffn_blockscale(
        jnp.ones((4, 16), dtype=host.FP8),
        act_sc,
        lambda b: jnp.ones((8, 32), dtype=host.FP8),
        lambda b: jnp.ones((8, 16), dtype=host.FP8),
        jnp.ones((2, 32), dtype=jnp.float32),
        jnp.ones((2, 16), dtype=jnp.float32),
        qb=8,
    )
    assert a2_bs.shape == (4, 16) and ms_bs.shape == (4, 1)

    a2_bf16, ms_bf16 = kernel.expert_ffn_bf16(
        jnp.ones((4, 16), dtype=jnp.bfloat16),
        jnp.ones((16, 16), dtype=jnp.bfloat16),
        lambda: jnp.ones((8, 16), dtype=jnp.bfloat16),
    )
    assert a2_bf16.shape == (4, 16) and ms_bf16 is None

    a2_int8, ms_int8 = kernel.expert_ffn_int8(
        jnp.ones((4, 16), dtype=jnp.bfloat16),
        lambda b: jnp.ones((8, 32), dtype=jnp.bfloat16),
        lambda b: jnp.ones((8, 16), dtype=jnp.bfloat16),
        jnp.ones((1, 32), dtype=jnp.float32),
        n_chunks1=2,
        n_chunks2=2,
        kb=8,
    )
    assert a2_int8.shape == (4, 16) and ms_int8 is None

    # _tile_row_scales & _and_nonempty
    win = jnp.arange(3 * host.HIDDEN_LANE_BLOCK, dtype=jnp.float32).reshape(
        3, host.HIDDEN_LANE_BLOCK
    )
    trs = kernel._tile_row_scales(win, jnp.int32(8), 16)
    assert trs.shape == (16, 1)
    assert bool(kernel._and_nonempty(jnp.bool_(True), jnp.int32(4))) is True


def test_build_fused_ep_moe_kernel_validations_and_execution(monkeypatch):
    base_kw = dict(
        g_local=2,
        capacity=32,
        hidden=128,
        inter=128,
        ep=2,
        ragged_rows_alloc=64,
    )
    with pytest.raises(NotImplementedError, match="fuses"):
        kernel._build_fused_ep_moe_kernel(**base_kw, act_fn="bad")
    with pytest.raises(ValueError, match="tile height"):
        kernel._build_fused_ep_moe_kernel(**{**base_kw, "capacity": 3})
    with pytest.raises(ValueError, match="smaller than one row"):
        kernel._build_fused_ep_moe_kernel(
            **base_kw, weight_format=host.WeightFormat.FP4, rhs_qb=0
        )
    with pytest.raises(ValueError, match="divide both"):
        kernel._build_fused_ep_moe_kernel(
            **base_kw, weight_format=host.WeightFormat.FP4, rhs_qb=48
        )
    with pytest.raises(ValueError, match="packed-weight row tile"):
        kernel._build_fused_ep_moe_kernel(
            **base_kw, weight_format=host.WeightFormat.FP4, rhs_qb=32
        )
    with pytest.raises(ValueError, match="integer weights widen"):
        kernel._build_fused_ep_moe_kernel(
            **base_kw, weight_format=host.WeightFormat.INT8
        )
    with pytest.raises(ValueError, match="positive multiple"):
        kernel._build_fused_ep_moe_kernel(**{**base_kw, "hidden": 64})
    with pytest.raises(ValueError, match="narrower than one"):
        kernel._build_fused_ep_moe_kernel(**{**base_kw, "inter": 64})
    with pytest.raises(ValueError, match="ragged_rows_alloc"):
        kernel._build_fused_ep_moe_kernel(**{**base_kw, "ragged_rows_alloc": None})

    # Intercept pl.pallas_call to execute the kernel body on _FakeRef instances
    captured_kernels = []

    def _run_pallas_call(kernel_fn, *, grid_spec, out_shape, compiler_params, name):
        captured_kernels.append((kernel_fn, grid_spec, out_shape, name))

        def _invoker(*args):
            refs = []
            # Scalar prefetch refs
            for a in args[: grid_spec.num_scalar_prefetch]:
                refs.append(jnp.asarray(a))
            # HBM input refs
            for a in args[grid_spec.num_scalar_prefetch :]:
                refs.append(_FakeRef(a.shape, a.dtype))
            # HBM output refs
            out_list = out_shape if isinstance(out_shape, list) else [out_shape]
            for o in out_list:
                refs.append(_FakeRef(o.shape, o.dtype))
            # Scratch refs
            for s in grid_spec.scratch_shapes:
                if hasattr(s, "shape") and hasattr(s, "dtype"):
                    refs.append(_FakeRef(s.shape, s.dtype))
                else:
                    refs.append(_FakeRef((4,), jnp.int32))
            kernel_fn(*refs)
            if isinstance(out_shape, list):
                return tuple(jnp.zeros(o.shape, dtype=o.dtype) for o in out_shape)
            return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        return _invoker

    monkeypatch.setattr(pl, "pallas_call", _run_pallas_call)

    def _invoke_built(
        fn,
        *,
        g_local=2,
        ep=2,
        hidden=512,
        inter=512,
        fmt=host.WeightFormat.FP8,
        w1b=False,
        w2b=False,
        rows_alloc=128,
    ):
        tables = (
            jnp.zeros((g_local, ep), dtype=jnp.int32),
            jnp.ones((g_local, ep), dtype=jnp.int32),
            jnp.zeros((g_local, ep), dtype=jnp.int32),
            jnp.zeros((g_local, ep), dtype=jnp.int32),
            jnp.ones((g_local, ep), dtype=jnp.int32) * 8,
        )
        token_gather = jnp.zeros((rows_alloc + 128,), dtype=jnp.int32)
        expert_rows = jnp.ones((g_local,), dtype=jnp.int32) * 16
        expert_base = jnp.zeros((g_local,), dtype=jnp.int32)
        form = host.weight_form(fmt)
        act_q = jnp.zeros((rows_alloc, hidden), dtype=form.act_dtype)
        act_scales = (
            jnp.ones(
                (host.act_scale_slab_rows(rows_alloc), host.HIDDEN_LANE_BLOCK),
                dtype=jnp.float32,
            )
            if form.quantized_activations
            else None
        )
        w1 = jnp.zeros((g_local, hidden, 2 * inter), dtype=form.weight_dtype)
        w2 = jnp.zeros((g_local, inter, hidden), dtype=form.weight_dtype)
        if fmt == host.WeightFormat.FP4:
            w1s = jnp.ones((g_local, hidden // 256, 2 * inter), dtype=jnp.float32)
            w2s = jnp.ones((g_local, inter // 256, hidden), dtype=jnp.float32)
        elif form.has_scales:
            w1s = jnp.ones((g_local, 2 * inter), dtype=jnp.float32)
            w2s = jnp.ones((g_local, hidden), dtype=jnp.float32)
        else:
            w1s = w2s = None
        w1b_arr = jnp.ones((g_local, 2 * inter), dtype=jnp.float32) if w1b else None
        w2b_arr = jnp.ones((g_local, hidden), dtype=jnp.float32) if w2b else None
        visit = jnp.arange(g_local, dtype=jnp.int32)
        counts = jnp.ones((host.N_COUNTS, ep), dtype=jnp.int32)
        rank = jnp.zeros((1, 1), dtype=jnp.int32)
        return fn(
            tables,
            token_gather,
            expert_rows,
            expert_base,
            act_q,
            act_scales,
            w1,
            w2,
            w1s,
            w2s,
            w1b_arr,
            w2b_arr,
            recv_rows=64,
            visit=visit,
            counts=counts,
            rank=rank,
        )

    # 1. FP8 with push_from_vmem=True and multi-rung height ladder
    # Force w1_nbuf=3, w2_nbuf=2 to hit both b < w2_prefetch True/False in prologue
    real_select = host.select_weight_buffers
    monkeypatch.setattr(
        host,
        "select_weight_buffers",
        lambda *a, **kw: SimpleNamespace(w1_nbuf=3, w2_nbuf=2),
    )
    fn_fp8_vmem = kernel.build_fused_ep_moe_kernel(
        g_local=2,
        capacity=32,
        hidden=512,
        inter=512,
        ep=2,
        weight_format=host.WeightFormat.FP8,
        ragged_rows_alloc=64,
        has_w1_bias=True,
        has_w2_bias=True,
    )
    # Cache hit on build_fused_ep_moe_kernel
    assert (
        kernel.build_fused_ep_moe_kernel(
            g_local=2,
            capacity=32,
            hidden=512,
            inter=512,
            ep=2,
            weight_format=host.WeightFormat.FP8,
            ragged_rows_alloc=64,
            has_w1_bias=True,
            has_w2_bias=True,
        )
        is fn_fp8_vmem
    )
    _invoke_built(
        fn_fp8_vmem, fmt=host.WeightFormat.FP8, w1b=True, w2b=True, rows_alloc=64
    )
    # Call again with same recv_rows=64 to hit make_call _call_cache!
    _invoke_built(
        fn_fp8_vmem, fmt=host.WeightFormat.FP8, w1b=True, w2b=True, rows_alloc=64
    )
    monkeypatch.setattr(host, "select_weight_buffers", real_select)

    # Validation errors on make_call and fn operands
    with pytest.raises(ValueError, match="arrival buffer height"):
        fn_fp8_vmem(
            (jnp.zeros((2, 2), dtype=jnp.int32),) * 5,
            jnp.zeros((64,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((64, 512), dtype=host.FP8),
            jnp.ones((host.act_scale_slab_rows(64), 128), dtype=jnp.float32),
            jnp.zeros((2, 512, 1024), dtype=host.FP8),
            jnp.zeros((2, 512, 512), dtype=host.FP8),
            jnp.ones((2, 1024), dtype=jnp.float32),
            jnp.ones((2, 512), dtype=jnp.float32),
            jnp.ones((2, 1024), dtype=jnp.float32),
            jnp.ones((2, 512), dtype=jnp.float32),
            recv_rows=3,
            visit=jnp.zeros((2,), dtype=jnp.int32),
            counts=jnp.zeros((host.N_COUNTS, 2), dtype=jnp.int32),
            rank=jnp.zeros((1, 1), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="wants the gate/up bias"):
        fn_fp8_vmem(
            (jnp.zeros((2, 2), dtype=jnp.int32),) * 5,
            jnp.zeros((64,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((64, 512), dtype=host.FP8),
            jnp.ones((1, 128), dtype=jnp.float32),
            jnp.zeros((2, 512, 1024), dtype=host.FP8),
            jnp.zeros((2, 512, 512), dtype=host.FP8),
            jnp.ones((2, 1024), dtype=jnp.float32),
            jnp.ones((2, 512), dtype=jnp.float32),
            None,
            None,
            recv_rows=64,
            visit=jnp.zeros((2,), dtype=jnp.int32),
            counts=jnp.zeros((host.N_COUNTS, 2), dtype=jnp.int32),
            rank=jnp.zeros((1, 1), dtype=jnp.int32),
        )

    # 2. FP4 with push_from_vmem=False and 2 blocks along inter (inter=512, rhs_qb=256)
    fn_fp4 = kernel._build_fused_ep_moe_kernel(
        g_local=2,
        capacity=32,
        hidden=512,
        inter=512,
        ep=2,
        weight_format=host.WeightFormat.FP4,
        rhs_qb=256,
        ragged_rows_alloc=512,
        act_fn="swigluoai",
    )
    _invoke_built(fn_fp4, fmt=host.WeightFormat.FP4, rows_alloc=512)

    # Also test missing .bitcast on w1_hbm in FP4 kernel body
    fp4_kernel_fn, fp4_grid_spec, fp4_out_shape, _ = captured_kernels[-1]
    bad_refs = []
    for _ in range(fp4_grid_spec.num_scalar_prefetch):
        bad_refs.append(jnp.zeros((2, 2), dtype=jnp.int32))
    for _ in range(len(fp4_grid_spec.in_specs)):
        bad_refs.append(_FakeRef((2, 512, 512), jnp.float4_e2m1fn, allow_bitcast=False))
    for o in fp4_out_shape:
        bad_refs.append(_FakeRef(o.shape, o.dtype))
    for _ in fp4_grid_spec.scratch_shapes:
        bad_refs.append(_FakeRef((4, 4), jnp.float32))
    with pytest.raises(NotImplementedError, match="bitcast"):
        fp4_kernel_fn(*bad_refs)

    # 3. INT8 with n_chunks2 = inter // WIDEN_KCHUNK = 2 (inter=1024, hidden=512)
    fn_int8 = kernel._build_fused_ep_moe_kernel(
        g_local=1,
        capacity=32,
        hidden=512,
        inter=1024,
        ep=1,
        weight_format=host.WeightFormat.INT8,
        ragged_rows_alloc=128,
    )
    _invoke_built(
        fn_int8,
        g_local=1,
        ep=1,
        hidden=512,
        inter=1024,
        fmt=host.WeightFormat.INT8,
        rows_alloc=128,
    )

    # 4. BF16
    fn_bf16 = kernel._build_fused_ep_moe_kernel(
        g_local=2,
        capacity=32,
        hidden=512,
        inter=512,
        ep=2,
        weight_format=host.WeightFormat.BF16,
        ragged_rows_alloc=128,
    )
    _invoke_built(fn_bf16, fmt=host.WeightFormat.BF16, rows_alloc=128)
    with pytest.raises(ValueError, match="has no operand for"):
        fn_bf16(
            (jnp.zeros((2, 2), dtype=jnp.int32),) * 5,
            jnp.zeros((128,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((128, 512), dtype=jnp.bfloat16),
            jnp.ones((1, 128), dtype=jnp.float32),
            jnp.zeros((2, 512, 1024), dtype=jnp.bfloat16),
            jnp.zeros((2, 512, 512), dtype=jnp.bfloat16),
            None,
            None,
            None,
            None,
            recv_rows=64,
            visit=jnp.zeros((2,), dtype=jnp.int32),
            counts=jnp.zeros((host.N_COUNTS, 2), dtype=jnp.int32),
            rank=jnp.zeros((1, 1), dtype=jnp.int32),
        )

    # Test double-checked lock branch on _BUILD_CACHE
    class _PrepopulatingLock:
        def __init__(self, cache, key_fn, val):
            self._cache = cache
            self._key_fn = key_fn
            self._val = val

        def __enter__(self):
            self._cache[self._key_fn()] = self._val

        def __exit__(self, *exc):
            return False

    def sentinel(*a, **kw):
        return None

    kw_lock = dict(
        g_local=4, capacity=32, hidden=512, inter=512, ep=2, ragged_rows_alloc=128
    )
    monkeypatch.setattr(
        kernel,
        "_BUILD_CACHE_LOCK",
        _PrepopulatingLock(
            kernel._BUILD_CACHE, lambda: tuple(sorted(kw_lock.items())), sentinel
        ),
    )
    assert kernel.build_fused_ep_moe_kernel(**kw_lock) is sentinel


def test_combine_step_tokens_and_combine_kernel(monkeypatch):
    assert kernel.combine_step_tokens(64) == 0
    assert kernel.combine_step_tokens(256) == 64
    assert kernel.combine_step_tokens(320) == 16
    assert kernel.combine_step_tokens(260) == 0

    def _run_combine_pallas(kernel_fn, *, grid_spec, out_shape, compiler_params, name):
        def _invoker(pos_flat, coef_flat, arrivals):
            refs = [
                jnp.asarray(pos_flat),
                jnp.asarray(coef_flat),
                _FakeRef(arrivals.shape, arrivals.dtype),
                _FakeRef(out_shape.shape, out_shape.dtype),
            ]
            for s in grid_spec.scratch_shapes:
                if hasattr(s, "shape") and hasattr(s, "dtype"):
                    refs.append(_FakeRef(s.shape, s.dtype))
                else:
                    refs.append(_FakeRef((4,), jnp.int32))
            kernel_fn(*refs)
            return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        return _invoker

    monkeypatch.setattr(pl, "pallas_call", _run_combine_pallas)

    ck = kernel.build_combine_kernel(
        t_local=256,
        topk=2,
        hidden=128,
        lane_blocks=1,
        wire_dtype=host.FP8,
        out_dtype=jnp.bfloat16,
    )
    assert (
        kernel.build_combine_kernel(
            t_local=256,
            topk=2,
            hidden=128,
            lane_blocks=1,
            wire_dtype=host.FP8,
            out_dtype=jnp.bfloat16,
        )
        is ck
    )
    arrivals = jnp.zeros((512, 1, 128), dtype=host.FP8)
    coef = jnp.ones((256, 2), dtype=jnp.float32)
    pos = jnp.zeros((256, 2), dtype=jnp.int32)
    out = ck(arrivals, coef, pos)
    assert out.shape == (256, 128)

    with pytest.raises(ValueError, match="token-major"):
        ck(arrivals, coef, jnp.zeros((128, 2), dtype=jnp.int32))

    # Double-checked lock branch on _COMBINE_CACHE
    kw_c = dict(
        t_local=512,
        topk=2,
        hidden=128,
        lane_blocks=1,
        wire_dtype=host.FP8,
        out_dtype=jnp.bfloat16,
    )

    def sentinel(*a, **kw):
        return None

    class _LockC:
        def __enter__(self):
            kernel._COMBINE_CACHE[
                tuple(sorted((k, str(v)) for k, v in kw_c.items()))
            ] = sentinel

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(kernel, "_COMBINE_CACHE_LOCK", _LockC())
    assert kernel.build_combine_kernel(**kw_c) is sentinel


def test_layer_helpers_and_validations(monkeypatch):
    # _combine_arrivals: both XLA (t_local=8) and Pallas combine kernel (t_local=256)
    monkeypatch.setattr(
        layer,
        "build_combine_kernel",
        lambda **kw: (
            lambda arrivals, coef, rows: jnp.zeros(
                (kw["t_local"], kw["hidden"]), dtype=kw["out_dtype"]
            )
        ),
    )
    arr_small = jnp.ones((16, 1, 128), dtype=host.FP8)
    scl_small = jnp.ones((4, 32), dtype=jnp.float32)
    pos_small = jnp.zeros((8, 2), dtype=jnp.int32)
    mpos_small = jnp.zeros((8, 2), dtype=jnp.int32)
    tw_small = jnp.ones((8, 2), dtype=jnp.float32) * 0.5
    c_small = layer._combine_arrivals(
        arr_small, scl_small, pos_small, mpos_small, tw_small, jnp.bfloat16
    )
    assert c_small.shape == (8, 128)

    arr_big = jnp.ones((512, 1, 128), dtype=host.FP8)
    pos_big = jnp.zeros((256, 2), dtype=jnp.int32)
    mpos_big = jnp.zeros((256, 2), dtype=jnp.int32)
    tw_big = jnp.ones((256, 2), dtype=jnp.float32) * 0.5
    c_big = layer._combine_arrivals(
        arr_big, scl_small, pos_big, mpos_big, tw_big, jnp.bfloat16
    )
    assert c_big.shape == (256, 128)

    # _exchange_layout, _pack_routing_exchange, _unpack_routing_exchange
    lay_q = layer._exchange_layout(64, 2, 8, True)
    assert layer._exchange_layout(64, 2, 8, True) == lay_q
    packed_q = layer._pack_routing_exchange(
        jnp.ones((128,), dtype=jnp.int32),
        jnp.ones((64,), dtype=jnp.int32),
        jnp.ones((8,), dtype=jnp.int32),
        lay_q,
    )
    p_g, sb_g, cnt_g = layer._unpack_routing_exchange(
        packed_q[None, :], lay_q, 64, 2, 8
    )
    assert p_g.shape == (1, 128) and sb_g.shape == (64,) and cnt_g.shape == (1, 8)

    lay_nq = layer._exchange_layout(32, 2, 8, False)
    packed_nq = layer._pack_routing_exchange(
        jnp.ones((64,), dtype=jnp.int32),
        None,
        jnp.ones((8,), dtype=jnp.int32),
        lay_nq,
    )
    _, sb_nq, _ = layer._unpack_routing_exchange(packed_nq[None, :], lay_nq, 32, 2, 8)
    assert sb_nq is None

    # _relabel_expert_ids_to_mesh_order
    idx = jnp.array([[0, 3], [2, 1]], dtype=jnp.int32)
    assert (
        layer._relabel_expert_ids_to_mesh_order(idx, g_local=2, mesh_ep_ranks=None)
        is idx
    )
    # Identity permutation fast-path
    assert (
        layer._relabel_expert_ids_to_mesh_order(idx, g_local=2, mesh_ep_ranks=(0, 1))
        is idx
    )
    # Non-identity permutation
    relabeled = layer._relabel_expert_ids_to_mesh_order(
        idx, g_local=2, mesh_ep_ranks=(1, 0)
    )
    np.testing.assert_array_equal(relabeled, np.array([[2, 1], [0, 3]], dtype=np.int32))
    with pytest.raises(ValueError, match="permutation"):
        layer._relabel_expert_ids_to_mesh_order(idx, g_local=2, mesh_ep_ranks=(0, 2))


def test_adaptive_fused_moe_validations_and_execution(monkeypatch):
    class _FakeMesh:
        axis_names = ("d",)
        shape = {"d": 2}

    fake_mesh = _FakeMesh()
    x = jnp.ones((16, 128), dtype=jnp.bfloat16)
    w1_fp8 = jnp.ones((4, 128, 256), dtype=host.FP8)
    w2_fp8 = jnp.ones((4, 128, 128), dtype=host.FP8)
    w1s_fp8 = jnp.ones((4, 256), dtype=jnp.float32)
    w2s_fp8 = jnp.ones((4, 128), dtype=jnp.float32)
    gate = jnp.ones((16, 4), dtype=jnp.float32)

    # All validation branches in adaptive_fused_moe
    with pytest.raises(ValueError, match="unsupported expert weight dtype"):
        layer.adaptive_fused_moe(
            x,
            jnp.ones((4, 128, 256), dtype=jnp.int32),
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    w1_fp4 = jnp.zeros((4, 512, 1024), dtype=jnp.float4_e2m1fn)
    w2_fp4 = jnp.zeros((4, 512, 512), dtype=jnp.float4_e2m1fn)
    with pytest.raises(ValueError, match="FP4 requires"):
        layer.adaptive_fused_moe(
            jnp.ones((16, 512), dtype=jnp.bfloat16),
            w1_fp4,
            w2_fp4,
            None,
            None,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="FP4 scale blocks must divide"):
        layer.adaptive_fused_moe(
            jnp.ones((16, 512), dtype=jnp.bfloat16),
            w1_fp4,
            w2_fp4,
            jnp.ones((4, 3, 1024), dtype=jnp.float32),
            jnp.ones((4, 3, 512), dtype=jnp.float32),
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="rhs_qb must be positive"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            rhs_qb=0,
        )
    with pytest.raises(ValueError, match="takes float8_e4m3fn expert weights"):
        layer.adaptive_fused_moe(
            x,
            jnp.ones((4, 128, 256), dtype=jnp.bfloat16),
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            weight_format=host.WeightFormat.FP8,
        )
    with pytest.raises(ValueError, match="carries per_channel weight scales"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            None,
            None,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="supplied together"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            None,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="rank-3 expert weights"):
        layer.adaptive_fused_moe(
            x[0],
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="W1/W2 dimensions"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            jnp.ones((4, 64, 128), dtype=host.FP8),
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="gating must have shape"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            jnp.ones((8, 4), dtype=jnp.float32),
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="topk must be between"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=0,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="w1_scale layout"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            jnp.ones((4, 1, 256), dtype=jnp.float32),
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="w2_scale layout"):
        layer.adaptive_fused_moe(
            jnp.ones((16, 512), dtype=jnp.bfloat16),
            w1_fp4,
            w2_fp4,
            jnp.ones((4, 2, 1024), dtype=jnp.float32),
            jnp.ones((4, 512), dtype=jnp.float32),
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="mesh_ep_ranks"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            mesh_ep_ranks=(0, 2),
        )
    with pytest.raises(ValueError, match="divisible by EP size"):
        layer.adaptive_fused_moe(
            jnp.ones((15, 128), dtype=jnp.bfloat16),
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            jnp.ones((15, 4), dtype=jnp.float32),
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
        )
    with pytest.raises(ValueError, match="FP4 block size must divide"):
        layer.adaptive_fused_moe(
            jnp.ones((16, 512), dtype=jnp.bfloat16),
            w1_fp4,
            w2_fp4,
            jnp.ones((4, 2, 1024), dtype=jnp.float32),
            jnp.ones((4, 2, 512), dtype=jnp.float32),
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            rhs_qb=300,
        )
    with pytest.raises(ValueError, match="FP4 scale contraction-block counts"):
        layer.adaptive_fused_moe(
            jnp.ones((16, 512), dtype=jnp.bfloat16),
            w1_fp4,
            w2_fp4,
            jnp.ones((4, 4, 1024), dtype=jnp.float32),
            jnp.ones((4, 4, 512), dtype=jnp.float32),
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            rhs_qb=256,
        )
    with pytest.raises(ValueError, match="multiple of capacity"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            ragged_stride=40,
        )
    with pytest.raises(ValueError, match="below the no-drop bound"):
        layer.adaptive_fused_moe(
            x,
            w1_fp8,
            w2_fp8,
            w1s_fp8,
            w2s_fp8,
            gate,
            topk=2,
            renormalize=True,
            mesh=fake_mesh,
            capacity=32,
            ragged_stride=32,
        )

    # Mock shard_map, device_put, lax.all_gather, pallas_select, and kernel builder
    # to execute local_fn end-to-end on CPU!
    monkeypatch.setattr(jax.sharding, "NamedSharding", lambda mesh, spec: (mesh, spec))
    monkeypatch.setattr(jax, "device_put", lambda a, sharding: a)
    monkeypatch.setattr(
        lax,
        "all_gather",
        lambda val, axis_name, axis=0, tiled=False, **kw: (
            jnp.concatenate([val, val], axis=axis)
            if tiled
            else jnp.stack([val, val], axis=axis)
        ),
    )
    monkeypatch.setattr(
        layer,
        "pallas_select",
        lambda scores, topk, block_rows: (
            jnp.ones((scores.shape[0], topk), dtype=jnp.float32),
            jnp.zeros((scores.shape[0], topk), dtype=jnp.int32),
        ),
    )
    monkeypatch.setattr(
        layer,
        "build_fused_ep_moe_kernel",
        lambda **kw: (
            lambda *a, recv_rows, **k: (
                jnp.zeros((recv_rows, kw["hidden"] // 128, 128), dtype=host.FP8),
                jnp.ones((16, kw["capacity"]), dtype=jnp.float32),
            )
        ),
    )
    monkeypatch.setattr(
        layer,
        "_combine_arrivals",
        lambda arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype: (
            jnp.zeros(
                (pos.shape[0], arrivals.shape[1] * arrivals.shape[2]), dtype=out_dtype
            )
        ),
    )

    def _fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma=False):
        def _wrapped(*args):
            # Slice first axis by ep=2 to pass shard-local views into local_fn
            sliced = [a[: a.shape[0] // 2] for a in args]
            res = fn(*sliced)
            return jnp.concatenate([res, res], axis=0)

        return _wrapped

    monkeypatch.setattr(jax, "shard_map", _fake_shard_map)

    # Run 1: FP8 quantized with token_replica_groups, renormalize=True, biases, none
    out1 = layer.adaptive_fused_moe(
        jnp.ones((32, 512), dtype=jnp.bfloat16),
        jnp.ones((4, 512, 1024), dtype=host.FP8),
        jnp.ones((4, 512, 512), dtype=host.FP8),
        jnp.ones((4, 1024), dtype=jnp.float32),
        jnp.ones((4, 512), dtype=jnp.float32),
        jnp.ones((32, 4), dtype=jnp.float32),
        w1_bias=jnp.ones((4, 1, 1024), dtype=jnp.float32),
        w2_bias=jnp.ones((4, 1, 512), dtype=jnp.float32),
        topk=2,
        renormalize=True,
        mesh=fake_mesh,
        capacity=128,
        mesh_ep_ranks=(0, 1),
        token_replica_groups=((0, 1),),
    )
    assert out1.shape == (32, 512)
    # Cache hit on _LAYER_SM_CACHE
    out1_cached = layer.adaptive_fused_moe(
        jnp.ones((32, 512), dtype=jnp.bfloat16),
        jnp.ones((4, 512, 1024), dtype=host.FP8),
        jnp.ones((4, 512, 512), dtype=host.FP8),
        jnp.ones((4, 1024), dtype=jnp.float32),
        jnp.ones((4, 512), dtype=jnp.float32),
        jnp.ones((32, 4), dtype=jnp.float32),
        w1_bias=jnp.ones((4, 1, 1024), dtype=jnp.float32),
        w2_bias=jnp.ones((4, 1, 512), dtype=jnp.float32),
        topk=2,
        renormalize=True,
        mesh=fake_mesh,
        capacity=128,
        mesh_ep_ranks=(0, 1),
        token_replica_groups=((0, 1),),
    )
    assert out1_cached.shape == (32, 512)

    # Run 2: BF16 unquantized with t_local = 1024 (>= AGQ_MIN_LOCAL_TOKENS)
    out2 = layer.adaptive_fused_moe(
        jnp.ones((2048, 512), dtype=jnp.bfloat16),
        jnp.ones((4, 512, 1024), dtype=jnp.bfloat16),
        jnp.ones((4, 512, 512), dtype=jnp.bfloat16),
        None,
        None,
        jnp.ones((2048, 4), dtype=jnp.float32),
        topk=2,
        renormalize=False,
        mesh=fake_mesh,
        capacity=128,
        block=128,
        ragged_stride=8192,
        rank=jnp.arange(2, dtype=jnp.int32).reshape(2, 1),
    )
    assert out2.shape == (2048, 512)

    # Run 3: FP4 with inferred rhs_qb=None and lock hit on _LAYER_SM_CACHE
    calls = []

    def _mock_get(k):
        calls.append(k)
        if len(calls) == 1:
            return None
        return lambda *a: jnp.zeros((32, 512), dtype=jnp.bfloat16)

    monkeypatch.setattr(layer, "_LAYER_SM_CACHE", SimpleNamespace(get=_mock_get))
    out3 = layer.adaptive_fused_moe(
        jnp.ones((32, 512), dtype=jnp.bfloat16),
        w1_fp4,
        w2_fp4,
        jnp.ones((4, 2, 1024), dtype=jnp.float32),
        jnp.ones((4, 2, 512), dtype=jnp.float32),
        jnp.ones((32, 4), dtype=jnp.float32),
        topk=2,
        renormalize=True,
        mesh=fake_mesh,
        capacity=128,
    )
    assert out3.shape == (32, 512)
