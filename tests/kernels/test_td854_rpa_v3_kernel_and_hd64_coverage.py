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

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    kernel as k_v3,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    kernel_hd64 as k_hd64,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import get_dtype_packing


class _FakeAsyncCopy:
    def start(self):
        pass

    def wait(self):
        pass


class _AtProxy:
    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, idx):
        return _RefProxy(self.ref.arr[idx])


class _RefProxy:
    def __init__(self, arr):
        self.arr = jnp.asarray(arr)

    def __jax_array__(self):
        return self.arr

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def at(self):
        return _AtProxy(self)

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, val):
        if isinstance(val, _RefProxy):
            val = val.arr
        if idx is Ellipsis:
            self.arr = jnp.broadcast_to(
                jnp.asarray(val, dtype=self.arr.dtype), self.arr.shape
            )
        else:
            self.arr = self.arr.at[idx].set(jnp.asarray(val, dtype=self.arr.dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _RefProxy(jnp.zeros(shape, dtype=self.arr.dtype))

    def bitcast(self, target_dtype):
        return _RefProxy(jnp.zeros(self.arr.shape, dtype=target_dtype))


def _fake_ds(start, sz, step=1):
    s = int(jnp.asarray(start))
    n = int(jnp.asarray(sz))
    st = int(jnp.asarray(step))
    return slice(s, s + n * st, st)


def _fake_bitcast(x, target_dtype):
    if isinstance(x, _RefProxy):
        x = x.arr
    target_dtype = jnp.dtype(target_dtype)
    in_pack = get_dtype_packing(x.dtype)
    out_pack = get_dtype_packing(target_dtype)
    if in_pack == out_pack:
        return jnp.zeros(x.shape, dtype=target_dtype)
    if in_pack == 1 and out_pack > 1:
        return jnp.zeros((x.shape[0] * out_pack, *x.shape[1:]), dtype=target_dtype)
    return jnp.zeros(
        (*x.shape[:-2], x.shape[-2] // in_pack, x.shape[-1]), dtype=target_dtype
    )


def _install_fake_pallas(monkeypatch):
    def _fake_loop(lower, upper, *, unroll=False):
        del unroll
        lo = int(jnp.asarray(lower))
        hi = int(jnp.asarray(upper))

        def _dec(body):
            for i in range(lo, hi):
                body(jnp.int32(i))

        return _dec

    def _fake_when(cond):
        def _dec(fn):
            if bool(jnp.all(jnp.asarray(cond))):
                fn()

        return _dec

    def _fake_fori_loop(lower, upper, body_fun, init_val, *, unroll=False):
        del unroll
        val = init_val
        for i in range(int(jnp.asarray(lower)), int(jnp.asarray(upper))):
            val = body_fun(jnp.int32(i), val)
        return val

    monkeypatch.setattr(jax.lax, "fori_loop", _fake_fori_loop)
    monkeypatch.setattr(pl, "loop", _fake_loop)
    monkeypatch.setattr(pl, "when", _fake_when)
    monkeypatch.setattr(pl, "ds", _fake_ds)
    monkeypatch.setattr(pl, "debug_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pl,
        "reciprocal",
        lambda x, approx=False: 1.0 / jnp.where(x == 0, 1.0, x),
    )
    monkeypatch.setattr(
        pltpu,
        "make_async_copy",
        lambda src, dst, sem: _FakeAsyncCopy(),
    )
    monkeypatch.setattr(pltpu, "bitcast", _fake_bitcast)


def test_rpa_v3_helpers_and_validations(monkeypatch):
    # RpaCase properties and get_range
    assert k_v3.RpaCase.DECODE.symbol == "d"
    assert k_v3.RpaCase.PREFILL.symbol == "p"
    assert k_v3.RpaCase.MIXED.symbol == "m"
    dist = jnp.array([1, 2, 3], dtype=jnp.int32)
    assert k_v3.RpaCase.DECODE.get_range(dist) == (0, dist[0])
    assert k_v3.RpaCase.PREFILL.get_range(dist) == (dist[0], dist[1])
    assert k_v3.RpaCase.MIXED.get_range(dist) == (dist[1], dist[2])
    with pytest.raises(ValueError, match="Unsupported RPA case"):
        k_v3.RpaCase.get_range(object(), dist)

    # get_smem_estimate_bytes, get_vmem_estimate_bytes, get_kv_cache_shape
    assert k_v3.get_smem_estimate_bytes(8, 4) > 0
    # has_bank_conflicts True (bkv_stride=2) and False (bkv_stride=1)
    assert (
        k_v3.get_vmem_estimate_bytes(2, 4, 128, 8, 16, jnp.bfloat16, jnp.bfloat16) > 0
    )
    assert (
        k_v3.get_vmem_estimate_bytes(1, 4, 128, 8, 16, jnp.bfloat16, jnp.bfloat16) > 0
    )
    assert k_v3.get_kv_cache_shape(4, 16, 2, 128, jnp.bfloat16) == (4, 16, 2, 2, 128)

    # merge_kv, prepare_inputs, prepare_outputs (aligned & unaligned, swap True/False
    q_al = jnp.ones((4, 4, 128), dtype=jnp.bfloat16)
    k_al = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    v_al = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    q_prep, kv_prep = k_v3.prepare_inputs(q_al, k_al, v_al)
    out_al = k_v3.prepare_outputs(q_prep, 2, 128)
    assert out_al.shape == (4, 4, 128)

    q_prep_t, _ = k_v3.prepare_inputs(q_al, k_al, v_al, fuse_non_tiling_axis_swap=True)
    out_al_t = k_v3.prepare_outputs(q_prep_t, 2, 128, fuse_non_tiling_axis_swap=True)
    assert out_al_t.shape == (4, 4, 128)

    # Unaligned q_heads_per_kv_head=1 (padded to 2 for bf16) and head_dim=64 (padded
    q_un = jnp.ones((4, 1, 64), dtype=jnp.bfloat16)
    k_un = jnp.ones((4, 1, 64), dtype=jnp.float8_e4m3fn)
    v_un = jnp.ones((4, 1, 64), dtype=jnp.float8_e4m3fn)
    q_prep_u, kv_prep_u = k_v3.prepare_inputs(
        q_un, k_un, v_un, fuse_non_tiling_axis_swap=False
    )
    assert kv_prep_u.shape == (4, 1, 4, 128)
    out_un = k_v3.prepare_outputs(q_prep_u, 1, 64, fuse_non_tiling_axis_swap=False)
    assert out_un.shape == (4, 1, 64)

    # get_default_block_sizes across TPU versions 5, 6, 7 and unsupported
    for ver in (5, 6, 7):
        k_v3.get_default_block_sizes.cache_clear()
        monkeypatch.setattr(k_v3, "get_tpu_version", lambda v=ver: v)
        for case in (k_v3.RpaCase.DECODE, k_v3.RpaCase.PREFILL, k_v3.RpaCase.MIXED):
            bs = k_v3.get_default_block_sizes(
                jnp.bfloat16, jnp.bfloat16, 8, 2, 128, 16, 64, 8, 4, case=case
            )
            assert bs["bq_sz"] >= 1
    k_v3.get_default_block_sizes.cache_clear()
    monkeypatch.setattr(k_v3, "get_tpu_version", lambda: 4)
    with pytest.raises(NotImplementedError, match="Unsupported tpu_version=4"):
        k_v3.get_default_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 128, 16, 64, 8, 4
        )
    k_v3.get_default_block_sizes.cache_clear()

    # static_validate_inputs & dynamic_validate_inputs error branches
    q = jnp.ones((8, 4, 128), dtype=jnp.bfloat16)
    k = jnp.ones((8, 2, 128), dtype=jnp.bfloat16)
    v = jnp.ones((8, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((4, 4, 2, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([4, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="Can not skip kv mask"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            use_causal_mask=True,
            skip_kv_mask=True,
        )
    with pytest.raises(ValueError, match="Expected 3D array"):
        k_v3.static_validate_inputs(
            q[0], k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
        )
    with pytest.raises(ValueError, match="Expected k.shape="):
        k_v3.static_validate_inputs(
            q,
            k,
            jnp.ones((8, 4, 128), dtype=jnp.bfloat16),
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected q.shape\\[0\\]="):
        k_v3.static_validate_inputs(
            jnp.ones((4, 4, 128), dtype=jnp.bfloat16),
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected q.shape\\[2\\]="):
        k_v3.static_validate_inputs(
            jnp.ones((8, 4, 64), dtype=jnp.bfloat16),
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="to be divisible by"):
        k_v3.static_validate_inputs(
            jnp.ones((8, 3, 128), dtype=jnp.bfloat16),
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected kv_cache.shape="):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            jnp.zeros((4, 4, 3, 2, 128), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

    # Monkeypatch get_kv_cache_shape to reach lines 1444, 1449, 1454, 1456, 1460, 1464
    monkeypatch.setattr(k_v3, "get_kv_cache_shape", lambda *a: a_shape)
    a_shape = (4, 4, 2, 2, 256)
    with pytest.raises(ValueError, match="Expected head_dim="):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            jnp.zeros(a_shape, dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    a_shape = (4, 4, 2, 2, 128)
    with pytest.raises(ValueError, match="Expected kv_cache.dtype="):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            jnp.zeros(a_shape, dtype=jnp.float32),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    q_i = jnp.ones((8, 4, 128), dtype=jnp.int32)
    k_i = jnp.ones((8, 2, 128), dtype=jnp.int32)
    v_i = jnp.ones((8, 2, 128), dtype=jnp.int32)
    with pytest.raises(ValueError, match="to be a floating point"):
        k_v3.static_validate_inputs(
            q_i,
            k_i,
            v_i,
            jnp.zeros(a_shape, dtype=jnp.int32),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    a_shape = (4, 4, 2, 4, 128)
    with pytest.raises(ValueError, match="does not match with"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            jnp.zeros(a_shape, dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    a_shape = (4, 4, 3, 1, 128)
    q_f = jnp.ones((8, 4, 128), dtype=jnp.float32)
    k_f = jnp.ones((8, 2, 128), dtype=jnp.float32)
    v_f = jnp.ones((8, 2, 128), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Combined KV heads must be divisible by 2"):
        k_v3.static_validate_inputs(
            q_f,
            k_f,
            v_f,
            jnp.zeros(a_shape, dtype=jnp.float32),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    a_shape = (4, 4, 1, 2, 128)
    with pytest.raises(ValueError, match="Invalid num_kv_heads_x2="):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            jnp.zeros(a_shape, dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    monkeypatch.undo()

    with pytest.raises(ValueError, match="Expected int32 dtype"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens.astype(jnp.float32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 1D array"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens.reshape(2, 1),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="to be divisible by max_num_seqs"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            jnp.array([0, 1, 2], dtype=jnp.int32),
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected cu_q_lens.shape="):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            jnp.array([0, 2], dtype=jnp.int32),
            distribution,
        )
    with pytest.raises(ValueError, match="Expected distribution.shape="):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([0, 2], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="page_size=1 must be divisible by"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            jnp.zeros((4, 1, 2, 2, 128), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="sliding_window=0 must be positive"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sliding_window=0,
        )
    with pytest.raises(ValueError, match="soft_cap=0.0 must not be 0.0"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            soft_cap=0.0,
        )
    with pytest.raises(ValueError, match="chunk_prefill_size=0 must be positive"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            chunk_prefill_size=0,
        )
    with pytest.raises(ValueError, match="must satisfy \\(0 < bq_csz"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            d_block_sizes=(4, 4, 3, 4),
        )
    with pytest.raises(ValueError, match="must satisfy \\(0 < bkv_csz"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            d_block_sizes=(4, 4, 2, 3),
        )
    with pytest.raises(ValueError, match="bkv_sz=6 must be divisible by page_size"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            d_block_sizes=(4, 6, 2, 6),
        )
    with pytest.raises(ValueError, match="bkv_csz=2 must be divisible by page_size"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            d_block_sizes=(4, 4, 2, 2),
        )
    with pytest.raises(ValueError, match="vmem_limit_bytes=0 must be positive"):
        k_v3.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            vmem_limit_bytes=0,
        )

    # dynamic_validate_inputs errors
    with pytest.raises(ValueError, match="Invalid distribution"):
        k_v3.dynamic_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([2, 1, 2], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="num_seqs=3 must be <="):
        k_v3.dynamic_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([0, 1, 3], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="Total q tokens"):
        k_v3.dynamic_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            jnp.array([0, 4, 9], dtype=jnp.int32),
            distribution,
        )
    with pytest.raises(ValueError, match="Require 0 < q_len="):
        k_v3.dynamic_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            jnp.array([1, 4], dtype=jnp.int32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Require page_cnt="):
        k_v3.dynamic_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            jnp.array([12, 4], dtype=jnp.int32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Require 0 <= page_idx="):
        k_v3.dynamic_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            jnp.array([4, 1, 2, 3], dtype=jnp.int32),
            cu_q_lens,
            distribution,
        )


def test_rpa_v3_ref_and_kernel_execution(monkeypatch):
    _install_fake_pallas(monkeypatch)
    q = jnp.ones((4, 4, 128), dtype=jnp.bfloat16)
    k = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    v = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((4, 4, 2, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([4, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 4], dtype=jnp.int32)
    distribution = jnp.array([1, 1, 2], dtype=jnp.int32)

    # ref_ragged_paged_attention: default out_dtype/mask_value, scales, soft_cap, sli
    out1, cache1 = k_v3.ref_ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=True,
        skip_kv_update=False,
        sliding_window=2,
        soft_cap=10.0,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.5,
    )
    assert out1.shape == (4, 4, 128)

    # ref_ragged_paged_attention: explicit out_dtype/mask_value, skip_kv_update=True,
    out2, _ = k_v3.ref_ragged_paged_attention(
        q.astype(jnp.float32),
        k,
        v,
        cache1,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=True,
        skip_kv_update=True,
        out_dtype=jnp.float32,
        mask_value=-1e4,
        q_scale=None,
        sliding_window=None,
    )
    assert out2.shape == (4, 4, 128)

    orig_issubdtype = jnp.issubdtype
    monkeypatch.setattr(jnp, "issubdtype", lambda a, b: False)
    monkeypatch.setattr(k_v3, "dynamic_validate_inputs", lambda *a, **kw: None)
    out2b, _ = k_v3.ref_ragged_paged_attention(
        q.astype(jnp.float32),
        k,
        v,
        cache1,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=False,
        skip_kv_update=True,
        out_dtype=jnp.float32,
        mask_value=-1e4,
        q_scale=1.0,
    )
    assert out2b.shape == (4, 4, 128)
    monkeypatch.setattr(jnp, "issubdtype", orig_issubdtype)
    monkeypatch.undo()

    _install_fake_pallas(monkeypatch)

    class _FakeTpuInfo:
        vmem_capacity_bytes = 64 * 1024 * 1024

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _FakeTpuInfo())
    monkeypatch.setattr(pltpu, "with_memory_space_constraint", lambda x, space: x)
    monkeypatch.setattr(
        jax, "jit", lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    )

    def _fake_pallas_call(kernel_fn, *, grid_spec, out_shape, **kwargs):
        def _runner(*args):
            n_pre = grid_spec.num_scalar_prefetch
            scalars = [_RefProxy(a) for a in args[:n_pre]]
            ins = [_RefProxy(a) for a in args[n_pre:]]
            outs = [_RefProxy(jnp.zeros(s.shape, dtype=s.dtype)) for s in out_shape]
            scratches = []
            for sc in grid_spec.scratch_shapes:
                if (
                    hasattr(sc, "shape")
                    and hasattr(sc, "dtype")
                    and "sem" not in str(sc.dtype)
                ):
                    scratches.append(_RefProxy(jnp.zeros(sc.shape, dtype=sc.dtype)))
                else:
                    scratches.append(_RefProxy(jnp.zeros((4, 2), dtype=jnp.int32)))
            kernel_fn(*scalars, *ins, *outs, *scratches)
            return tuple(o.arr for o in outs)

        return _runner

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call)

    # Run 1: TPU v7,bf16 (kv_packing=2), sliding_window=2, soft_cap, scales, chunk_pr
    monkeypatch.setattr(k_v3, "get_tpu_version", lambda: 7)
    k_v3.get_default_block_sizes.cache_clear()
    cu_q_lens_3 = jnp.array([0, 1, 3, 4], dtype=jnp.int32)
    kv_lens_3 = jnp.array([4, 4, 4], dtype=jnp.int32)
    page_indices_3 = jnp.array([0, 1, 2, 3, 0, 1], dtype=jnp.int32)
    dist_3 = jnp.array([1, 2, 3], dtype=jnp.int32)
    kv_cache_6 = jnp.zeros((6, 2, 2, 2, 128), dtype=jnp.bfloat16)
    out_rpa1, _ = k_v3.ragged_paged_attention.__wrapped__(
        jnp.ones((4, 4, 256), dtype=jnp.bfloat16),
        jnp.ones((4, 2, 256), dtype=jnp.bfloat16),
        jnp.ones((4, 2, 256), dtype=jnp.bfloat16),
        jnp.zeros((6, 2, 2, 2, 256), dtype=jnp.bfloat16),
        kv_lens_3,
        page_indices_3,
        cu_q_lens_3,
        dist_3,
        sliding_window=2,
        soft_cap=10.0,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.5,
        chunk_prefill_size=2,
        d_block_sizes=(1, 2, 1, 2),
        p_block_sizes=(1, 2, 1, 2),
        m_block_sizes=(1, 2, 1, 2),
    )
    assert out_rpa1.shape == (4, 4, 256)

    # Run 2: TPU v6, float32 (kv_packing=1, out_dtype=float32), use_causal_mask=False
    monkeypatch.setattr(k_v3, "get_tpu_version", lambda: 6)
    k_v3.get_default_block_sizes.cache_clear()
    out_rpa2, _ = k_v3.ragged_paged_attention.__wrapped__(
        jnp.ones((4, 1, 128), dtype=jnp.float32),
        jnp.ones((4, 1, 128), dtype=jnp.float32),
        jnp.ones((4, 1, 128), dtype=jnp.float32),
        jnp.zeros((6, 2, 2, 1, 128), dtype=jnp.float32),
        kv_lens_3,
        page_indices_3,
        cu_q_lens_3,
        dist_3,
        use_causal_mask=False,
        skip_kv_mask=True,
        skip_kv_update=True,
        out_dtype=jnp.float32,
        mask_value=-1e4,
        vmem_limit_bytes=32 * 1024 * 1024,
    )
    assert out_rpa2.shape == (4, 1, 128)

    # Run 3: debug_mode=True and actual_num_kv_heads=1 with bf16 (bkv_stride=1, has_b
    out_rpa3, _ = k_v3.ragged_paged_attention.__wrapped__(
        jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
        jnp.ones((4, 1, 128), dtype=jnp.bfloat16),
        jnp.ones((4, 1, 128), dtype=jnp.bfloat16),
        jnp.zeros((6, 2, 1, 2, 128), dtype=jnp.bfloat16),
        kv_lens_3,
        page_indices_3,
        cu_q_lens_3,
        dist_3,
        debug_mode=True,
        d_block_sizes=(1, 2, 1, 2),
        m_block_sizes=(2, 2, 1, 2),
    )
    assert out_rpa3.shape == (4, 2, 128)

    # Non-floating branch in step1_qk_softmax when q_scale is not None
    orig_issub = jnp.issubdtype
    monkeypatch.setattr(jnp, "issubdtype", lambda a, b: False)
    monkeypatch.setattr(k_v3, "static_validate_inputs", lambda *a, **kw: None)
    _ = k_v3.ragged_paged_attention.__wrapped__(
        jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
        jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
        jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
        kv_cache_6,
        kv_lens_3,
        page_indices_3,
        cu_q_lens_3,
        dist_3,
        q_scale=0.5,
        d_block_sizes=(1, 2, 1, 2),
        m_block_sizes=(2, 2, 1, 2),
    )
    monkeypatch.setattr(jnp, "issubdtype", orig_issub)

    # broadcast_minor when src.shape == shape (direct call inside loop with head_dim=
    # ragged_paged_attention_bundled
    monkeypatch.setattr(
        k_v3,
        "ragged_paged_attention",
        lambda q, k, v, flat_b, *a, **kw: (q, flat_b),
    )
    bundle = jnp.zeros((3, 2, 2, 2, 2, 128), dtype=jnp.bfloat16)
    out_b, new_b = k_v3.ragged_paged_attention_bundled.__wrapped__(
        jnp.ones((4, 4, 128), dtype=jnp.bfloat16),
        jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
        jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
        bundle,
        jnp.int32(1),
        kv_lens_3,
        page_indices_3,
        cu_q_lens_3,
        dist_3,
    )
    assert out_b.shape == (4, 4, 128)
    assert new_b.shape == bundle.shape


def test_rpa_v3_hd64_helpers_and_validations(monkeypatch):
    assert k_hd64.get_smem_estimate_bytes(8, 4) > 0
    assert (
        k_hd64.get_vmem_estimate_bytes(2, 4, 64, 8, 16, jnp.bfloat16, jnp.bfloat16) > 0
    )
    assert k_hd64.get_kv_cache_shape(4, 16, 2, 64, jnp.bfloat16) == (4, 16, 1, 2, 128)

    # merge_kv: aligned (actual_num_kv_heads=2, bf16) and unaligned (actual_num_kv_he
    k_al = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
    v_al = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
    assert k_hd64.merge_kv(k_al, v_al).shape == (4, 1, 2, 128)

    k_un = jnp.ones((4, 1, 64), dtype=jnp.bfloat16)
    v_un = jnp.ones((4, 1, 64), dtype=jnp.bfloat16)
    assert k_hd64.merge_kv(k_un, v_un).shape == (4, 1, 2, 128)

    # prepare_inputs with and without attention_sink, and prepare_outputs
    q = jnp.ones((4, 4, 64), dtype=jnp.bfloat16)
    sink = jnp.ones((4,), dtype=jnp.float32)
    q_p, kv_p, sink_p = k_hd64.prepare_inputs(q, k_al, v_al, sink)
    assert sink_p is not None and sink_p.shape == (2, 2, 128)
    q_p2, _, sink_p2 = k_hd64.prepare_inputs(q, k_al, v_al, None)
    assert sink_p2 is None
    out_p = k_hd64.prepare_outputs(q_p2, 2, 64)
    assert out_p.shape == (4, 4, 64)

    # static_validate_inputs & dynamic_validate_inputs error branches
    kv_cache = jnp.zeros((4, 4, 1, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([4, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="Expected 3D array"):
        k_hd64.static_validate_inputs(
            q[0], k_al, v_al, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
        )
    with pytest.raises(ValueError, match="Expected k.shape="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            jnp.ones((4, 4, 64), dtype=jnp.bfloat16),
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected q.shape\\[0\\]="):
        k_hd64.static_validate_inputs(
            jnp.ones((2, 4, 64), dtype=jnp.bfloat16),
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected q.shape\\[2\\]="):
        k_hd64.static_validate_inputs(
            jnp.ones((4, 4, 128), dtype=jnp.bfloat16),
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected attention_sink.shape\\[0\\]="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            jnp.ones((2,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="Expected attention_sink.dtype="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            jnp.ones((4,), dtype=jnp.bfloat16),
        )
    with pytest.raises(ValueError, match="Expected actual_head_dim=128 to be 64"):
        k_hd64.static_validate_inputs(
            jnp.ones((4, 4, 128), dtype=jnp.bfloat16),
            jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
            jnp.ones((4, 2, 128), dtype=jnp.bfloat16),
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="to be divisible by"):
        k_hd64.static_validate_inputs(
            jnp.ones((4, 3, 64), dtype=jnp.bfloat16),
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(
        ValueError, match="Expected actual_head_dim_x2=64 is equal to 128"
    ):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            jnp.zeros((4, 4, 1, 2, 64), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected kv_cache.dtype="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            jnp.zeros((4, 4, 1, 2, 128), dtype=jnp.float32),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="to be a floating point"):
        k_hd64.static_validate_inputs(
            jnp.ones((4, 4, 64), dtype=jnp.int32),
            jnp.ones((4, 2, 64), dtype=jnp.int32),
            jnp.ones((4, 2, 64), dtype=jnp.int32),
            jnp.zeros((4, 4, 2, 1, 128), dtype=jnp.int32),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="does not match with"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            jnp.zeros((4, 4, 1, 4, 128), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Invalid num_kv_heads="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            jnp.zeros((4, 4, 2, 2, 128), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected int32 dtype"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens.astype(jnp.float32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 1D array"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens.reshape(2, 1),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="to be divisible by max_num_seqs"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            jnp.array([0, 1, 2], dtype=jnp.int32),
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected cu_q_lens.shape="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            jnp.array([0, 2], dtype=jnp.int32),
            distribution,
        )
    with pytest.raises(ValueError, match="Expected distribution.shape="):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([0, 2], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="page_size=1 must be divisible by"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            jnp.zeros((4, 1, 1, 2, 128), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="sliding_window=0 must be positive"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sliding_window=0,
        )
    with pytest.raises(ValueError, match="soft_cap=0.0 must not be 0.0"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            soft_cap=0.0,
        )
    with pytest.raises(ValueError, match="chunk_prefill_size=0 must be positive"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            chunk_prefill_size=0,
        )
    with pytest.raises(ValueError, match="num_kv_pages_per_block=0 must be positive"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_kv_pages_per_block=0,
        )
    with pytest.raises(ValueError, match="num_queries_per_block=0 must be positive"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_queries_per_block=0,
        )
    with pytest.raises(ValueError, match="vmem_limit_bytes=0 must be positive"):
        k_hd64.static_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            vmem_limit_bytes=0,
        )

    # dynamic_validate_inputs errors
    with pytest.raises(ValueError, match="Invalid distribution"):
        k_hd64.dynamic_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([2, 1, 2], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="num_seqs=3 must be <="):
        k_hd64.dynamic_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([0, 1, 3], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="Total q tokens"):
        k_hd64.dynamic_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            page_indices,
            jnp.array([0, 2, 5], dtype=jnp.int32),
            distribution,
        )
    with pytest.raises(ValueError, match="Require 0 < q_len="):
        k_hd64.dynamic_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            jnp.array([1, 4], dtype=jnp.int32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Require page_cnt="):
        k_hd64.dynamic_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            jnp.array([12, 4], dtype=jnp.int32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Require 0 <= page_idx="):
        k_hd64.dynamic_validate_inputs(
            q,
            k_al,
            v_al,
            kv_cache,
            kv_lens,
            jnp.array([4, 1, 2, 3], dtype=jnp.int32),
            cu_q_lens,
            distribution,
        )


def test_rpa_v3_hd64_ref_and_kernel_execution(monkeypatch):
    _install_fake_pallas(monkeypatch)
    q = jnp.ones((4, 4, 64), dtype=jnp.bfloat16)
    k = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
    v = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((6, 2, 1, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([4, 4, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 3, 4], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 3], dtype=jnp.int32)
    sink = jnp.ones((4,), dtype=jnp.float32)

    # ref_ragged_paged_attention_hd64: mask_value=None, scales, sliding_window, soft_
    out1, cache1 = k_hd64.ref_ragged_paged_attention_hd64(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink=sink,
        mask_value=None,
        sliding_window=2,
        soft_cap=10.0,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.5,
        skip_kv_update=False,
        use_causal_mask=True,
    )
    assert out1.shape == (4, 4, 64)

    # ref_ragged_paged_attention_hd64: no sink, q_scale=None, skip_kv_update=True, us
    out2a, _ = k_hd64.ref_ragged_paged_attention_hd64(
        q,
        k,
        v,
        cache1,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink=None,
        mask_value=-1e4,
        q_scale=None,
        skip_kv_update=True,
        use_causal_mask=False,
    )
    assert out2a.shape == (4, 4, 64)

    orig_issub = jnp.issubdtype
    monkeypatch.setattr(jnp, "issubdtype", lambda a, b: False)
    monkeypatch.setattr(k_hd64, "dynamic_validate_inputs", lambda *a, **kw: None)
    out2, _ = k_hd64.ref_ragged_paged_attention_hd64(
        q,
        k,
        v,
        cache1,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink=None,
        mask_value=-1e4,
        q_scale=0.5,
        skip_kv_update=True,
        use_causal_mask=False,
    )
    assert out2.shape == (4, 4, 64)
    monkeypatch.setattr(jnp, "issubdtype", orig_issub)
    monkeypatch.undo()

    _install_fake_pallas(monkeypatch)
    monkeypatch.setattr(k_hd64, "get_tuned_block_sizes", lambda *a: (1, 2))

    prog_state = {"seq_idx": 0, "num_seqs": 3}
    monkeypatch.setattr(pl, "program_id", lambda axis: jnp.int32(prog_state["seq_idx"]))
    monkeypatch.setattr(
        pl, "num_programs", lambda axis: jnp.int32(prog_state["num_seqs"])
    )

    def _fake_fori_loop(lower, upper, body_fun, init_val, *, unroll=False):
        del unroll
        val = init_val
        for i in range(int(jnp.asarray(lower)), int(jnp.asarray(upper))):
            val = body_fun(jnp.int32(i), val)
        return val

    monkeypatch.setattr(jax.lax, "fori_loop", _fake_fori_loop)

    def _fake_pallas_call_hd64(kernel_fn, *, grid_spec, out_shape, **kwargs):
        def _runner(*args):
            n_pre = grid_spec.num_scalar_prefetch
            scalars = [_RefProxy(a) for a in args[:n_pre]]
            ins = [None if a is None else _RefProxy(a) for a in args[n_pre:]]
            outs = [_RefProxy(jnp.zeros(s.shape, dtype=s.dtype)) for s in out_shape]
            scratches = []
            for sc in grid_spec.scratch_shapes:
                if (
                    hasattr(sc, "shape")
                    and hasattr(sc, "dtype")
                    and "sem" not in str(sc.dtype)
                ):
                    scratches.append(_RefProxy(jnp.zeros(sc.shape, dtype=sc.dtype)))
                else:
                    scratches.append(_RefProxy(jnp.zeros((4, 2), dtype=jnp.int32)))
            n_seqs = int(jnp.asarray(grid_spec.grid[0]))
            prog_state["num_seqs"] = n_seqs
            for s_idx in range(n_seqs):
                prog_state["seq_idx"] = s_idx
                kernel_fn(*scalars, *ins, *outs, *scratches)
            return tuple(o.arr for o in outs)

        return _runner

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call_hd64)

    # Run 1: with attention_sink, sliding_window=2, strict_sliding_window=True, soft_
    q_1kv = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
    k_1kv = jnp.ones((4, 1, 64), dtype=jnp.bfloat16)
    v_1kv = jnp.ones((4, 1, 64), dtype=jnp.bfloat16)
    out_hd1, _ = k_hd64.ragged_paged_attention_hd64.__wrapped__(
        q_1kv,
        k_1kv,
        v_1kv,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink=jnp.ones((2,), dtype=jnp.float32),
        sliding_window=2,
        strict_sliding_window=True,
        soft_cap=10.0,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.5,
        chunk_prefill_size=2,
    )
    assert out_hd1.shape == (4, 2, 64)

    # Run 2: multiple kv heads (actual_num_kv_heads=2 -> hits prev_bq_shape_0 is not
    out_hd2, _ = k_hd64.ragged_paged_attention_hd64.__wrapped__(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        jnp.zeros((6, 2, 2, 1, 128), dtype=jnp.float32),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink=None,
        sliding_window=None,
        strict_sliding_window=False,
        skip_kv_update=True,
        use_causal_mask=False,
        num_kv_pages_per_block=1,
        num_queries_per_block=2,
        vmem_limit_bytes=32 * 1024 * 1024,
    )
    assert out_hd2.shape == (4, 4, 64)

    # Run 3: debug_mode=True
    out_hd3, _ = k_hd64.ragged_paged_attention_hd64.__wrapped__(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        debug_mode=True,
        num_kv_pages_per_block=1,
        num_queries_per_block=2,
    )
    assert out_hd3.shape == (4, 4, 64)

    # Run 4: non-floating kv.dtype branch in step1_qk_softmax when q_scale is not None
    monkeypatch.setattr(jnp, "issubdtype", lambda a, b: False)
    monkeypatch.setattr(k_hd64, "static_validate_inputs", lambda *a, **kw: None)
    _ = k_hd64.ragged_paged_attention_hd64.__wrapped__(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        q_scale=0.5,
        sliding_window=2,
        strict_sliding_window=False,
        num_kv_pages_per_block=1,
        num_queries_per_block=2,
    )
    monkeypatch.setattr(jnp, "issubdtype", orig_issub)
