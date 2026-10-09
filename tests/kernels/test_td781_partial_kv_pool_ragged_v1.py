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
"""CPU unit tests for TD7.81: partial_kv_update, pool_adapters, and ragged v1."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels import partial_kv_update, pool_adapters, typed_ldst
from vllm_torchtpu.kernels.sparse_core import (
    core_map_helper,
    ragged_gather,
    ragged_scatter,
)


class _FakeAt:
    def __init__(self, ref: "_FakeRef"):
        self._ref = ref

    def __getitem__(self, idx):
        arr = self._ref.array
        if isinstance(idx, tuple):
            conv = tuple(
                slice(0, x.size) if isinstance(x, pl.Slice) else x for x in idx
            )
            return _FakeRef(arr[conv])
        if isinstance(idx, pl.Slice):
            return _FakeRef(arr[: idx.size])
        return _FakeRef(arr[idx])


class _FakeRef:
    """Minimal Pallas Ref emulator for CPU unit testing."""

    def __init__(self, array: jax.Array):
        self.array = array

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    @property
    def at(self):
        return _FakeAt(self)

    def bitcast(self, target_dtype):
        in_bits = jax.dtypes.itemsize_bits(self.array.dtype)
        out_bits = jax.dtypes.itemsize_bits(target_dtype)
        rows = max(1, (self.array.shape[0] * in_bits) // out_bits)
        return _FakeRef(jnp.zeros((rows,) + self.array.shape[1:], dtype=target_dtype))

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.array
        if isinstance(idx, tuple):
            conv = tuple(
                slice(0, x.size) if isinstance(x, pl.Slice) else x for x in idx
            )
            return self.array[conv]
        if isinstance(idx, pl.Slice):
            return self.array[: idx.size]
        return self.array[idx]

    def __setitem__(self, idx, val):
        raw = val.array if isinstance(val, _FakeRef) else val
        if idx is Ellipsis:
            self.array = jnp.asarray(raw, dtype=self.array.dtype)
        elif isinstance(idx, tuple):
            conv = tuple(
                slice(0, x.size) if isinstance(x, pl.Slice) else x for x in idx
            )
            self.array = self.array.at[conv].set(
                jnp.asarray(raw, dtype=self.array.dtype)
            )
        else:
            self.array = self.array.at[idx].set(
                jnp.asarray(raw, dtype=self.array.dtype)
            )


def _install_fake_pallas_for_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pl, "program_id", lambda axis: 0)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(num_sublanes=8, num_lanes=128),
    )

    def _fake_load_typed(ref, *, view_dtype, lane_split=1):
        total_bytes = math.prod(ref.shape) * jnp.dtype(ref.dtype).itemsize
        out_lanes = ref.shape[-1] // lane_split
        rows = total_bytes // (out_lanes * jnp.dtype(view_dtype).itemsize)
        return jnp.zeros((rows, out_lanes), dtype=view_dtype)

    monkeypatch.setattr(typed_ldst, "load_typed", _fake_load_typed)
    monkeypatch.setattr(
        typed_ldst,
        "store_typed",
        lambda ref, val, *, lane_split=1: None,
    )

    def _fake_pallas_call(
        kernel_fn, *, grid_spec, out_shape, input_output_aliases=None
    ):
        del input_output_aliases

        def _call(*args):
            num_scalar = grid_spec.num_scalar_prefetch
            scalar_args = [_FakeRef(a) for a in args[:num_scalar]]
            array_args = args[num_scalar:]
            in_refs = []
            for a, spec in zip(array_args, grid_spec.in_specs):
                if spec.index_map is not None:
                    spec.index_map(0, *args[:num_scalar])
                bshape = spec.block_shape if spec.block_shape is not None else a.shape
                in_refs.append(_FakeRef(jnp.zeros(bshape, dtype=a.dtype)))

            if grid_spec.out_specs.index_map is not None:
                grid_spec.out_specs.index_map(0, *args[:num_scalar])
            out_bshape = grid_spec.out_specs.block_shape
            out_ref = _FakeRef(jnp.zeros(out_bshape, dtype=out_shape.dtype))
            kernel_fn(*scalar_args, *in_refs, out_ref)
            return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        return _call

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call)


def test_partial_kv_update_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pallas_for_pool(monkeypatch)

    # 3D pool with int32 and int16 indices
    pool_3d = jnp.zeros((16, 8, 128), dtype=jnp.bfloat16)
    vals_3d = jnp.ones((2, 8, 128), dtype=jnp.bfloat16)
    idx_i32 = jnp.array([1, 3], dtype=jnp.int32)
    idx_i16 = jnp.array([1, 3], dtype=jnp.int16)

    out_3d_a = partial_kv_update.pallas_scatter_slots(pool_3d, idx_i32, vals_3d)
    out_3d_b = partial_kv_update.pallas_scatter_slots(pool_3d, idx_i16, vals_3d)
    assert out_3d_a.shape == pool_3d.shape
    assert out_3d_b.shape == pool_3d.shape

    # 2D pool valid and invalid num_slots
    pool_2d = jnp.zeros((16, 128), dtype=jnp.bfloat16)
    vals_2d = jnp.ones((2, 128), dtype=jnp.bfloat16)
    out_2d, first_idx = partial_kv_update._jax_scatter_op(pool_2d, idx_i32, vals_2d)
    assert out_2d.shape == pool_2d.shape
    assert int(first_idx) == 1

    with pytest.raises(ValueError, match="must be a multiple of"):
        partial_kv_update.pallas_scatter_slots(
            jnp.zeros((10, 128), dtype=jnp.bfloat16), idx_i32, vals_2d
        )

    with pytest.raises(ValueError, match="pool dtype"):
        partial_kv_update.pallas_scatter_slots(
            pool_3d, idx_i32, vals_3d.astype(jnp.float32)
        )

    with pytest.raises(ValueError, match="unsupported pool ndim"):
        partial_kv_update.pallas_scatter_slots(
            jnp.zeros((16,), dtype=jnp.bfloat16),
            idx_i32,
            jnp.zeros((2,), dtype=jnp.bfloat16),
        )

    # PyTorch bridge & fake op
    t_pool = torch.zeros((8, 4, 16), dtype=torch.bfloat16)
    t_idx32 = torch.tensor([0, 2], dtype=torch.int32)
    t_idx64 = torch.tensor([0, 2], dtype=torch.int64)
    t_vals = torch.ones((2, 4, 16), dtype=torch.bfloat16)

    fake_out, fake_idx = partial_kv_update._fake_scatter_op(t_pool, t_idx32, t_vals)
    assert fake_out.shape == t_pool.shape and fake_idx.shape == ()

    monkeypatch.setattr(
        partial_kv_update,
        "_pallas_scatter_torch_op",
        lambda c, i, u: (c.clone(), i[0]),
    )
    unwrapped = getattr(
        partial_kv_update.pallas_index_copy_,
        "__wrapped__",
        partial_kv_update.pallas_index_copy_,
    )
    res32 = unwrapped(t_pool, t_idx32, t_vals)
    res64 = unwrapped(t_pool, t_idx64, t_vals)
    assert res32 is t_pool and res64 is t_pool


def test_pool_adapters_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pallas_for_pool(monkeypatch)

    # 1) HND vs NHD pool geometry
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    assert not pool_adapters.is_seq_along_lane_pool()
    pool_4d = jnp.zeros((12, 16, 4, 128), dtype=jnp.bfloat16)
    assert pool_adapters._pool_geometry(pool_4d) == (16, (4,), 128)

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    assert pool_adapters.is_seq_along_lane_pool()
    assert pool_adapters._pool_geometry(pool_4d) == (128, (16, 4), 128)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")

    # _bitcast_payload with empty and non-empty payload
    assert pool_adapters._bitcast_payload((), jnp.bfloat16, jnp.float32) == ()
    assert pool_adapters._bitcast_payload((4,), jnp.bfloat16, jnp.float32) == (2,)

    idx = jnp.array([0, 1], dtype=jnp.int32)

    # 2) gather_region: split > 1 (_gather_window, straddle, single-block recurse)
    g_win1 = pool_adapters.gather_region(
        pool_4d, idx, tok0=0, ntok=32, out_dtype=jnp.bfloat16, split=3
    )
    assert g_win1.shape == (2, 128, 128)

    pool_3d = jnp.zeros((12, 16, 256), dtype=jnp.bfloat16)
    g_win_3d = pool_adapters.gather_region(
        pool_3d, idx, tok0=0, ntok=32, out_dtype=jnp.float32, out_lanes=128, split=3
    )
    assert g_win_3d.shape == (2, 32, 128)

    with pytest.raises(NotImplementedError, match="straddles kernel blocks"):
        pool_adapters.gather_region(
            pool_4d, idx, tok0=12, ntok=8, out_dtype=jnp.bfloat16, split=3
        )

    g_sub = pool_adapters.gather_region(
        pool_4d, idx, tok0=16, ntok=8, out_dtype=jnp.bfloat16, split=3
    )
    assert g_sub.shape == (2, 32, 128)

    # 3) gather_region: split == 1 (unaligned chunks > 1, 3D cross-dtype, lane_split)
    g_unaligned = pool_adapters.gather_region(
        pool_3d, idx, tok0=6, ntok=4, out_dtype=jnp.float32, out_lanes=128
    )
    assert g_unaligned.shape == (2, 4, 128)

    # 4) scatter_region: split > 1 (_scatter_window, straddle, single-block recurse)
    s_win1 = pool_adapters.scatter_region(
        pool_4d, g_win1, idx, tok0=0, ntok=32, split=3
    )
    assert s_win1.shape == pool_4d.shape

    s_win_3d = pool_adapters.scatter_region(
        pool_3d, g_win_3d, idx, tok0=0, ntok=32, split=3
    )
    assert s_win_3d.shape == pool_3d.shape

    with pytest.raises(NotImplementedError, match="straddles kernel blocks"):
        pool_adapters.scatter_region(pool_4d, g_sub, idx, tok0=12, ntok=8, split=3)

    s_sub = pool_adapters.scatter_region(pool_4d, g_sub, idx, tok0=16, ntok=8, split=3)
    assert s_sub.shape == pool_4d.shape

    # 5) scatter_region: split == 1 (aligned chunks == 1 & unaligned chunks > 1)
    s_aligned = pool_adapters.scatter_region(pool_4d, g_sub, idx, tok0=0, ntok=8)
    assert s_aligned.shape == pool_4d.shape

    s_unaligned = pool_adapters.scatter_region(
        pool_3d, g_unaligned, idx, tok0=6, ntok=4
    )
    assert s_unaligned.shape == pool_3d.shape

    # 6) copy_blocks
    copied = pool_adapters.copy_blocks(pool_4d, idx, idx)
    assert copied.shape == pool_4d.shape

    # 7) _conv_qk_pair_rows_perm all branches
    with pytest.raises(NotImplementedError, match="equal Q/K"):
        pool_adapters._conv_qk_pair_rows_perm(3, 320, 1, 128, 128)
    with pytest.raises(NotImplementedError, match="lane-aligned V"):
        pool_adapters._conv_qk_pair_rows_perm(3, 320, 1, 64, 128)
    assert pool_adapters._conv_qk_pair_rows_perm(3, 384, 1, 128, 128) is None
    perm = pool_adapters._conv_qk_pair_rows_perm(2, 768, 2, 128, 128)
    assert perm == (0, 2, 1, 3, 4, 5, 6, 8, 7, 9, 10, 11)

    # 8) v3_state_source all branches
    pool_v3 = jnp.zeros((4, 64, 4, 128), dtype=jnp.bfloat16)
    plan_f32 = pool_adapters.v3_state_source(
        pool_v3,
        split=2,
        ssm_ntok=64,
        conv_tok0=64,
        conv_ntok=8,
        conv_dim=768,
        n_v=2,
        d_k=64,
        d_v=128,
        kernel_size=4,
        recurrent_state_dtype=jnp.float32,
        qk_pair_layout=True,
    )
    assert plan_f32.recurrent.nblocks == 1 and plan_f32.conv.rows_perm is not None

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    pool_hnd = jnp.zeros((4, 4, 128, 64), dtype=jnp.bfloat16)
    plan_bf16 = pool_adapters.v3_state_source(
        pool_hnd,
        split=2,
        ssm_ntok=32,
        conv_tok0=32,
        conv_ntok=8,
        conv_dim=256,
        n_v=2,
        d_k=32,
        d_v=64,
        kernel_size=2,
        recurrent_state_dtype=jnp.bfloat16,
        qk_pair_layout=False,
    )
    assert plan_bf16.whole_block_dma is True
    assert plan_bf16.recurrent.view_dtype == jnp.dtype(jnp.uint32)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")

    with pytest.raises(NotImplementedError, match="conv region partially straddles"):
        pool_adapters.v3_state_source(
            pool_v3,
            split=2,
            ssm_ntok=32,
            conv_tok0=60,
            conv_ntok=8,
            conv_dim=256,
            n_v=2,
            d_k=32,
            d_v=128,
            kernel_size=2,
        )


def test_sparse_core_ragged_gather_and_scatter_all_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = jnp.arange(64, dtype=jnp.bfloat16).reshape(8, 8)
    indices = jnp.array([2, 0, 5, 1], dtype=jnp.int32)

    # 1) Fallback when sparse_core is None (with scalar and 1D start/end)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            sparse_core=None,
            num_lanes=128,
            generation=7,
            vmem_capacity_bytes=1024,
        ),
    )
    g_fb1 = ragged_gather.ragged_gather(x, indices, jnp.int32(0), jnp.int32(4))
    g_fb2 = ragged_gather.ragged_gather(
        x, indices, jnp.array([0], dtype=jnp.int32), jnp.array([4], dtype=jnp.int32)
    )
    assert g_fb1.shape == (4, 8) and g_fb2.shape == (4, 8)

    s_fb1 = ragged_scatter.ragged_scatter(x, indices, jnp.int32(0), jnp.int32(4))
    s_fb2 = ragged_scatter.ragged_scatter(
        x, indices, jnp.array([0], dtype=jnp.int32), jnp.array([4], dtype=jnp.int32)
    )
    assert s_fb1.shape == (4, 8) and s_fb2.shape == (4, 8)

    # 2) calculate_col_size across generations 5, 6, 7
    for gen in (5, 6, 7):
        monkeypatch.setattr(
            pltpu,
            "get_tpu_info",
            lambda g=gen: SimpleNamespace(
                sparse_core=SimpleNamespace(num_lanes=8, num_cores=1, num_subcores=1),
                num_lanes=128,
                generation=g,
                vmem_capacity_bytes=16 * 1024 * 1024,
            ),
        )
        assert ragged_gather.calculate_col_size(256) == 256
        assert ragged_scatter.calculate_col_size(256) == 256

    # 3) ragged_scatter small-tensor VMEM heuristic fallback
    s_small = ragged_scatter.ragged_scatter.__wrapped__(
        x, indices, jnp.int32(0), jnp.int32(4)
    )
    assert s_small.shape == (4, 8)

    # 4) SparseCore dispatch path in ragged_gather & ragged_scatter
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            sparse_core=SimpleNamespace(num_lanes=4, num_cores=1, num_subcores=1),
            num_lanes=8,
            generation=7,
            vmem_capacity_bytes=1,
        ),
    )
    monkeypatch.setattr(
        plsc,
        "VectorSubcoreMesh",
        lambda num_cores,
        num_subcores,
        core_axis_name,
        subcore_axis_name: SimpleNamespace(
            num_cores=num_cores,
            num_subcores=num_subcores,
            core_axis_name=core_axis_name,
            subcore_axis_name=subcore_axis_name,
        ),
    )

    class _FakeDMA:
        def start(self):
            pass

        def wait(self):
            pass

    monkeypatch.setattr(pltpu, "make_async_copy", lambda *a, **k: _FakeDMA())
    monkeypatch.setattr(pltpu, "sync_copy", lambda *a, **k: None)
    monkeypatch.setattr(jax.lax, "axis_size", lambda axes: 1)
    monkeypatch.setattr(pl, "program_id", lambda axis: 0)
    monkeypatch.setattr(pl, "when", lambda cond: lambda fn: fn())

    def _fake_loop(lower, upper, *, step=1):
        def _dec(fn):
            for v in range(lower, upper, step):
                fn(v)
            return fn

        return _dec

    monkeypatch.setattr(pl, "loop", _fake_loop)
    monkeypatch.setattr(pltpu, "emit_pipeline", lambda fn, **kw: fn)

    def _fake_core_map_kernel(fn, *, out_type, scratch_types, **kw):
        def _run(*args):
            in_refs = [_FakeRef(a) for a in args]
            out_ref = _FakeRef(jnp.zeros(out_type.shape, dtype=out_type.dtype))
            sc_refs = []
            for k, v in scratch_types.items():
                if k == "sem_ref":
                    sc_refs.append(_FakeRef(jnp.zeros((2,), dtype=jnp.int32)))
                else:
                    sc_refs.append(_FakeRef(jnp.ones(v.shape, dtype=v.dtype)))
            fn(*in_refs, out_ref, *sc_refs)
            return out_ref.array

        return _run

    monkeypatch.setattr(core_map_helper, "kernel", _fake_core_map_kernel)

    # Aligned (out_pad_size == 0 and aligned_hidden_size == hidden_size)
    x_aligned = jnp.ones((8, 8), dtype=jnp.bfloat16)
    idx_aligned = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    g_al = ragged_gather.ragged_gather.__wrapped__(
        x_aligned, idx_aligned, jnp.int32(0), jnp.int32(4)
    )
    s_al = ragged_scatter.ragged_scatter.__wrapped__(
        x_aligned, idx_aligned, jnp.int32(0), jnp.int32(4)
    )
    assert g_al.shape == (4, 8) and s_al.shape == (4, 8)

    # Unaligned (out_pad_size > 0 and aligned_hidden_size != hidden_size)
    x_unaligned = jnp.ones((8, 6), dtype=jnp.float32)
    idx_unaligned = jnp.array([0, 1, 2], dtype=jnp.int32)
    g_un = ragged_gather.ragged_gather.__wrapped__(
        x_unaligned, idx_unaligned, jnp.int32(0), jnp.int32(3)
    )
    s_un = ragged_scatter.ragged_scatter.__wrapped__(
        x_unaligned, idx_unaligned, jnp.int32(0), jnp.int32(3)
    )
    assert g_un.shape == (3, 6) and s_un.shape == (3, 6)
