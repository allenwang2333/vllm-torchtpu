# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for TD8.98 mla/v2/kernel.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest  # type: ignore[missing-import]

from vllm_torchtpu.kernels.mla.v2 import kernel as mla_v2

pytestmark = pytest.mark.cpu_test


class _FakeCopy:
    def start(self) -> None:
        return None

    def wait(self) -> None:
        return None


class _FakeRef:
    """Array-backed fake Pallas memory reference supporting .at, .bitcast, .reshape."""

    def __init__(self, arr: jax.Array):
        self.arr = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> Any:
        return self.arr.dtype

    @property
    def at(self) -> _FakeRef:
        return self

    def _norm(self, idx: Any) -> Any:
        if isinstance(idx, tuple):
            return tuple(self._norm(i) for i in idx)
        if hasattr(idx, "start") and hasattr(idx, "size"):
            start = int(idx.start)
            size = int(idx.size)
            return slice(start, start + size)
        if isinstance(idx, jax.Array) and idx.ndim == 0:
            return int(idx)
        return idx

    def __getitem__(self, idx: Any) -> Any:
        if idx is Ellipsis:
            return self.arr
        norm = self._norm(idx)
        sub = self.arr[norm]
        if isinstance(sub, jax.Array) and sub.ndim > 0:
            return _FakeRef(sub)
        return sub

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(val, _FakeRef):
            val = val.arr
        if idx is Ellipsis:
            self.arr = jnp.broadcast_to(
                jnp.asarray(val, dtype=self.arr.dtype), self.arr.shape
            )
            return
        norm = self._norm(idx)
        target = self.arr[norm]
        target_shape = target.shape if isinstance(target, jax.Array) else ()
        arr_val = jnp.broadcast_to(jnp.asarray(val, dtype=self.arr.dtype), target_shape)
        self.arr = self.arr.at[norm].set(arr_val)

    def reshape(self, *shape: Any) -> _FakeRef:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _FakeRef(self.arr.reshape(*shape))

    def bitcast(self, target_dtype: Any) -> _FakeRef:
        src_bits = jax.dtypes.itemsize_bits(self.arr.dtype)
        dst_bits = jax.dtypes.itemsize_bits(target_dtype)
        if src_bits == dst_bits:
            return _FakeRef(self.arr.astype(target_dtype))
        if dst_bits > src_bits:
            ratio = dst_bits // src_bits
            for ax in range(self.arr.ndim - 2, -1, -1):
                if self.arr.shape[ax] % ratio == 0:
                    new_shape = (
                        *self.arr.shape[:ax],
                        self.arr.shape[ax] // ratio,
                        *self.arr.shape[ax + 1 :],
                    )
                    return _FakeRef(jnp.zeros(new_shape, dtype=target_dtype))
        else:
            ratio = src_bits // dst_bits
            new_shape = (
                *self.arr.shape[:-2],
                self.arr.shape[-2] * ratio,
                self.arr.shape[-1],
            )
            return _FakeRef(jnp.zeros(new_shape, dtype=target_dtype))
        return _FakeRef(self.arr.astype(target_dtype))

    def __jax_array__(self) -> jax.Array:
        return self.arr


def _fake_bitcast(x: Any, target_dtype: Any) -> jax.Array:
    arr = x.arr if isinstance(x, _FakeRef) else jnp.asarray(x)
    src_bits = jax.dtypes.itemsize_bits(arr.dtype)
    dst_bits = jax.dtypes.itemsize_bits(target_dtype)
    if src_bits == dst_bits:
        return arr.astype(target_dtype)
    if dst_bits < src_bits:
        ratio = src_bits // dst_bits
        new_shape = (*arr.shape[:-2], arr.shape[-2] * ratio, arr.shape[-1])
        return jnp.zeros(new_shape, dtype=target_dtype)
    ratio = dst_bits // src_bits
    new_shape = (*arr.shape[:-2], arr.shape[-2] // ratio, arr.shape[-1])
    return jnp.zeros(new_shape, dtype=target_dtype)


def test_mla_v2_math_and_cache_shape_helpers() -> None:
    # Power-of-two and non-power-of-two branches
    assert mla_v2.unsigned_mod(10, 4) == 2
    assert mla_v2.unsigned_mod(10, 3) == 1

    assert mla_v2.unsigned_cdiv(10, 4) == 3
    assert mla_v2.unsigned_cdiv(10, 3) == 4

    assert mla_v2.unsigned_floor_div(10, 4) == 2
    assert mla_v2.unsigned_floor_div(10, 3) == 3

    assert mla_v2.unsigned_align_to(10, 4) == 12
    assert mla_v2.unsigned_align_to(10, 3) == 12

    assert mla_v2.align_to(130, 128) == 256
    assert mla_v2.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert mla_v2.get_dtype_packing(jnp.bfloat16) == 2

    # get_kv_cache_shape
    assert mla_v2.get_kv_cache_shape(
        4, 128, 256, jnp.bfloat16, transpose_kv_cache=True
    ) == (4, 256, 128)
    assert mla_v2.get_kv_cache_shape(
        4, 128, 256, jnp.bfloat16, kv_packing=None, transpose_kv_cache=False
    ) == (4, 64, 2, 256)
    assert mla_v2.get_kv_cache_shape(
        4, 128, 256, jnp.bfloat16, kv_packing=4, transpose_kv_cache=False
    ) == (4, 32, 4, 256)

    # MlaCase symbols
    assert mla_v2.MlaCase.DECODE.symbol == "d"
    assert mla_v2.MlaCase.PREFILL.symbol == "p"
    assert mla_v2.MlaCase.MIXED.symbol == "m"
    assert mla_v2.MlaCase.BATCHED_DECODE.symbol == "bd"


def test_mla_v2_static_validate_inputs() -> None:
    ql_nope = jnp.zeros((4, 8, 128), dtype=jnp.bfloat16)
    q_pe = jnp.zeros((8, 4, 128), dtype=jnp.bfloat16)
    new_kv_c = jnp.zeros((8, 128), dtype=jnp.bfloat16)
    new_k_pe = jnp.zeros((8, 128), dtype=jnp.bfloat16)
    cache_kv = jnp.zeros((4, 64, 2, 256), dtype=jnp.bfloat16)
    cache_kv_t = jnp.zeros((4, 256, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.zeros((2,), dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu_q_lens = jnp.zeros((3,), dtype=jnp.int32)
    distribution = jnp.zeros((3,), dtype=jnp.int32)

    # Valid non-transposed with optional args
    mla_v2.static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sliding_window=64,
        soft_cap=30.0,
        chunk_prefill_size=4,
        num_kv_pages_per_blocks=(1, 1, 1),
        num_queries_per_blocks=(2, 2, 4),
        mixed_q_split=2,
        vmem_limit_bytes=1024,
        transpose_kv_cache=False,
    )
    # Valid transposed with None optional args
    mla_v2.static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv_t,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        transpose_kv_cache=True,
    )

    with pytest.raises(ValueError, match="Expected 3D array for ql_nope"):
        mla_v2.static_validate_inputs(
            ql_nope[0],
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 3D array for q_pe"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe[0],
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 2D array for new_kv_c"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c[0],
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 2D array for new_k_pe"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe[0],
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="num_heads"):
        mla_v2.static_validate_inputs(
            jnp.zeros((2, 8, 128), dtype=jnp.bfloat16),
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="num_tokens"):
        mla_v2.static_validate_inputs(
            jnp.zeros((4, 4, 128), dtype=jnp.bfloat16),
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="new_kv_c.shape\\[0\\]"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            jnp.zeros((4, 128), dtype=jnp.bfloat16),
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="new_k_pe.shape\\[0\\]"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            jnp.zeros((4, 128), dtype=jnp.bfloat16),
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="new_kv_c.shape\\[1\\]"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            jnp.zeros((8, 64), dtype=jnp.bfloat16),
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="new_k_pe.shape\\[1\\]"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            jnp.zeros((8, 64), dtype=jnp.bfloat16),
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="kv_packing="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            jnp.zeros((4, 64, 3, 256), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="aligned_kv_dim="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            jnp.zeros((4, 64, 2, 128), dtype=jnp.bfloat16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="new_kv_c.dtype="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c.astype(jnp.float32),
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="new_k_pe.dtype="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe.astype(jnp.float32),
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="floating point"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c.astype(jnp.int16),
            new_k_pe.astype(jnp.int16),
            cache_kv.astype(jnp.int16),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected int32 dtype"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens.astype(jnp.float32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 1D array"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens[:, None],
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="divisible by max_num_seqs"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            jnp.zeros((3,), dtype=jnp.int32),
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="cu_q_lens.shape="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            jnp.zeros((4,), dtype=jnp.int32),
            distribution,
        )
    with pytest.raises(ValueError, match="distribution.shape="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.zeros((4,), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="sliding_window="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sliding_window=0,
        )
    with pytest.raises(ValueError, match="soft_cap="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            soft_cap=0.0,
        )
    with pytest.raises(ValueError, match="chunk_prefill_size="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            chunk_prefill_size=0,
        )
    with pytest.raises(ValueError, match="num_kv_pages_per_block="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_kv_pages_per_blocks=(1, 0, 1),
        )
    with pytest.raises(ValueError, match="mixed_q_split="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            mixed_q_split=0,
        )
    with pytest.raises(ValueError, match="num_queries_per_block="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_queries_per_blocks=(1, 0, 2),
        )
    with pytest.raises(ValueError, match="divisible by mixed_q_split"):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_queries_per_blocks=(2, 2, 3),
            mixed_q_split=2,
        )
    with pytest.raises(ValueError, match="vmem_limit_bytes="):
        mla_v2.static_validate_inputs(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            vmem_limit_bytes=0,
        )


def test_mla_v2_prepare_inputs_outputs_and_wrapper(monkeypatch) -> None:
    q = jnp.zeros((5, 3, 64), dtype=jnp.bfloat16)
    q_padded = mla_v2.prepare_q_inputs(q)
    assert q_padded.shape == (5, 4, 128)

    # prepare_q_nope_inputs: xpose_pipeline success vs ValueError fallback
    monkeypatch.setattr(
        mla_v2,
        "xpose_pipeline",
        lambda x, **kw: (jnp.transpose(x, (1, 0, 2)),),
    )
    q_nope_in = jnp.zeros((3, 5, 64), dtype=jnp.bfloat16)
    q_nope_out = mla_v2.prepare_q_nope_inputs(q_nope_in)
    assert q_nope_out.shape == (16, 4, 128)

    out_ok = mla_v2.prepare_outputs(q_nope_out, 3, 5, 64)
    assert out_ok.shape == (3, 5, 64)

    def _raise_val_err(*_a, **_kw):
        raise ValueError("tile mismatch")

    monkeypatch.setattr(mla_v2, "xpose_pipeline", _raise_val_err)
    q_nope_fb = mla_v2.prepare_q_nope_inputs(q_nope_in)
    assert q_nope_fb.shape == (16, 4, 128)
    out_fb = mla_v2.prepare_outputs(q_nope_fb, 3, 5, 64)
    assert out_fb.shape == (3, 5, 64)

    # prepare_kv_inputs: padded and aligned token counts
    kv_unaligned = jnp.zeros((3, 64), dtype=jnp.bfloat16)
    assert mla_v2.prepare_kv_inputs(kv_unaligned, 2).shape == (2, 2, 128)
    kv_aligned = jnp.zeros((4, 128), dtype=jnp.bfloat16)
    assert mla_v2.prepare_kv_inputs(kv_aligned, 2).shape == (2, 2, 128)

    # prepare_kv_inputs_for_transposed_kv_cache: unaligned & aligned
    assert mla_v2.prepare_kv_inputs_for_transposed_kv_cache(kv_unaligned).shape == (
        128,
        128,
    )
    kv_128 = jnp.zeros((128, 128), dtype=jnp.bfloat16)
    assert mla_v2.prepare_kv_inputs_for_transposed_kv_cache(kv_128).shape == (128, 128)

    # mla_ragged_paged_attention wrapper (both transpose_kv_cache=False and True)
    fn = mla_v2.mla_ragged_paged_attention.__wrapped__
    ql_nope = jnp.zeros((4, 16, 128), dtype=jnp.bfloat16)
    q_pe = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
    new_kv_c = jnp.zeros((16, 128), dtype=jnp.bfloat16)
    new_k_pe = jnp.zeros((16, 128), dtype=jnp.bfloat16)
    cache_kv = jnp.zeros((4, 64, 2, 256), dtype=jnp.bfloat16)
    cache_kv_t = jnp.zeros((4, 256, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([4, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="must be specified"):
        fn(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_kv_pages_per_block=None,
            num_queries_per_block=2,
        )

    monkeypatch.setattr(
        mla_v2.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=64 * 1024 * 1024),
    )
    monkeypatch.setattr(
        mla_v2.pl,
        "pallas_call",
        lambda *_a, **_kw: lambda *args: (args[7], args[11]),
    )

    # 1. int block sizes, vmem_limit_bytes=None, mask_value=None, chunk_prefill_size=4
    out1, ukv1 = fn(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_kv_pages_per_block=1,
        num_queries_per_block=2,
        chunk_prefill_size=4,
        transpose_kv_cache=False,
    )
    assert out1.shape == (4, 16, 128)
    assert ukv1.shape == cache_kv.shape

    # 2. tuple block sizes, explicit vmem_limit_bytes & mask_value, transpose=True
    out2, ukv2 = fn(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv_t,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        mask_value=-1e9,
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(2, 2, 2),
        vmem_limit_bytes=32 * 1024 * 1024,
        chunk_prefill_size=None,
        transpose_kv_cache=True,
    )
    assert out2.shape == (4, 16, 128)
    assert ukv2.shape == cache_kv_t.shape


def _run_kernel_variant(
    monkeypatch,
    *,
    transpose_kv_cache: bool,
    two_step_flash_attention: bool,
    bkv_p: int = 1,
    bq_sz: int = 2,
    batch_size: int = 1,
    q_split: int = 1,
    static_q_len: int | None = None,
    use_causal_mask: bool = True,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    p_same_dtype_as_v: bool = True,
    q_dtype: Any = jnp.bfloat16,
    debug_mode: bool = False,
    same_minor_shape: bool = False,
) -> None:
    monkeypatch.setattr(mla_v2.pl, "program_id", lambda _axis: jnp.int32(0))
    monkeypatch.setattr(mla_v2.pl, "when", lambda _cond: lambda fn: fn())
    monkeypatch.setattr(mla_v2.pl, "debug_print", lambda *_a, **_kw: None)
    monkeypatch.setattr(mla_v2.pl, "reciprocal", lambda x, approx=True: 1.0 / x)
    monkeypatch.setattr(
        mla_v2.pltpu, "make_async_copy", lambda src, dst, sem: _FakeCopy()
    )
    monkeypatch.setattr(mla_v2.pltpu, "bitcast", _fake_bitcast)
    monkeypatch.setattr(
        mla_v2.pltpu, "roll", lambda x, shift, axis: jnp.roll(x, shift, axis=axis)
    )
    monkeypatch.setattr(mla_v2.kv_utils, "pack_new_kv", lambda *_a, **_kw: None)

    def _fori_loop(lo, hi, body, init, unroll=False):
        del unroll
        val = init
        for idx in range(int(lo), max(int(hi), 1)):
            val = body(jnp.int32(idx), val)
        return val

    monkeypatch.setattr(mla_v2.lax, "fori_loop", _fori_loop)

    num_q_heads = 4
    lkv_dim = 128
    r_dim = 128
    kv_dim = lkv_dim + r_dim
    page_size = 128
    q_packing = mla_v2.get_dtype_packing(q_dtype)
    kv_packing = mla_v2.get_dtype_packing(q_dtype)
    page_size_per_kv_packing = page_size // kv_packing
    total_num_pages = 4
    max_num_tokens = 128

    kv_lens_ref = _FakeRef(jnp.array([130, 140], dtype=jnp.int32))
    page_indices_ref = _FakeRef(jnp.array([0, 1, 2, 3], dtype=jnp.int32))
    cu_q_lens_ref = _FakeRef(jnp.array([0, 4, 8], dtype=jnp.int32))
    start_end_seq_idx_ref = _FakeRef(jnp.array([0, batch_size], dtype=jnp.int32))
    sem_ids_ref = _FakeRef(jnp.zeros((3,), dtype=jnp.int32))
    bo_ids_ref = _FakeRef(jnp.zeros((4,), dtype=jnp.int32))
    bkv_update_ids_ref = _FakeRef(jnp.ones((batch_size, 6), dtype=jnp.int32))

    ql_nope_hbm_ref = _FakeRef(
        jnp.zeros((max_num_tokens, num_q_heads, lkv_dim), dtype=q_dtype)
    )
    q_pe_hbm_ref = _FakeRef(
        jnp.zeros((max_num_tokens, num_q_heads, r_dim), dtype=q_dtype)
    )
    o_hbm_ref = _FakeRef(
        jnp.zeros((max_num_tokens, num_q_heads, lkv_dim), dtype=q_dtype)
    )

    if not transpose_kv_cache:
        new_kv_c_hbm_ref = _FakeRef(
            jnp.zeros(
                (max_num_tokens // kv_packing, kv_packing, lkv_dim), dtype=q_dtype
            )
        )
        new_k_pe_hbm_ref = _FakeRef(
            jnp.zeros((max_num_tokens // kv_packing, kv_packing, r_dim), dtype=q_dtype)
        )
        cache_kv_hbm_ref = _FakeRef(
            jnp.zeros(
                (total_num_pages, page_size_per_kv_packing, kv_packing, kv_dim),
                dtype=q_dtype,
            )
        )
        updated_cache_kv_hbm_ref = _FakeRef(
            jnp.zeros(
                (total_num_pages, page_size_per_kv_packing, kv_packing, kv_dim),
                dtype=q_dtype,
            )
        )
        bkv_buf_sz = bkv_p * page_size_per_kv_packing + 2
        bkvc_x2_ref = _FakeRef(
            jnp.zeros((2, batch_size, bkv_buf_sz, kv_packing, lkv_dim), dtype=q_dtype)
        )
        bkpe_x2_ref = _FakeRef(
            jnp.zeros((2, batch_size, bkv_buf_sz, kv_packing, r_dim), dtype=q_dtype)
        )
    else:
        new_kv_c_hbm_ref = _FakeRef(jnp.zeros((lkv_dim, max_num_tokens), dtype=q_dtype))
        new_k_pe_hbm_ref = _FakeRef(jnp.zeros((r_dim, max_num_tokens), dtype=q_dtype))
        cache_kv_hbm_ref = _FakeRef(
            jnp.zeros((total_num_pages, kv_dim, page_size), dtype=q_dtype)
        )
        updated_cache_kv_hbm_ref = _FakeRef(
            jnp.zeros((total_num_pages, kv_dim, page_size), dtype=q_dtype)
        )
        bkv_sz = bkv_p * page_size
        bkvc_x2_ref = _FakeRef(
            jnp.zeros((2, batch_size, lkv_dim, bkv_sz + 384), dtype=q_dtype)
        )
        bkpe_x2_ref = _FakeRef(
            jnp.zeros((2, batch_size, r_dim, bkv_sz + 384), dtype=q_dtype)
        )

    padded_num_q_heads = (
        mla_v2.unsigned_align_to(num_q_heads // q_packing, 4) * q_packing
    )
    bq_nope_x2_ref = _FakeRef(
        jnp.zeros((2, batch_size, bq_sz, padded_num_q_heads, lkv_dim), dtype=q_dtype)
    )
    bq_rope_x2_ref = _FakeRef(
        jnp.zeros((2, batch_size, bq_sz, padded_num_q_heads, r_dim), dtype=q_dtype)
    )
    bo_x2_ref = _FakeRef(
        jnp.zeros((2, batch_size, bq_sz, padded_num_q_heads, lkv_dim), dtype=q_dtype)
    )
    sems = _FakeRef(jnp.zeros((4, batch_size, 2), dtype=jnp.int32))
    lm_minor = bkv_p * page_size if same_minor_shape else 128
    l_ref = _FakeRef(
        jnp.ones((batch_size, bq_sz * num_q_heads, lm_minor), dtype=jnp.float32)
    )
    m_ref = _FakeRef(
        jnp.zeros((batch_size, bq_sz * num_q_heads, lm_minor), dtype=jnp.float32)
    )
    acc_ref = _FakeRef(
        jnp.zeros((batch_size, bq_sz * num_q_heads, lkv_dim), dtype=jnp.float32)
    )

    mla_v2._mla_ragged_paged_attention_kernel(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        start_end_seq_idx_ref,
        sem_ids_ref,
        bo_ids_ref,
        bkv_update_ids_ref,
        ql_nope_hbm_ref,
        q_pe_hbm_ref,
        new_kv_c_hbm_ref,
        new_k_pe_hbm_ref,
        cache_kv_hbm_ref,
        o_hbm_ref,
        updated_cache_kv_hbm_ref,
        bkvc_x2_ref,
        bkpe_x2_ref,
        bq_nope_x2_ref,
        bq_rope_x2_ref,
        bo_x2_ref,
        sems,
        l_ref,
        m_ref,
        acc_ref,
        static_q_len=static_q_len,
        sm_scale=1.0,
        mask_value=-1e9,
        s_dtype=jnp.float32,
        transpose_kv_cache=transpose_kv_cache,
        two_step_flash_attention=two_step_flash_attention,
        p_same_dtype_as_v=p_same_dtype_as_v,
        use_causal_mask=use_causal_mask,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        bkv_p=bkv_p,
        bq_sz=bq_sz,
        batch_size=batch_size,
        q_split=q_split,
        debug_mode=debug_mode,
    )


def test_mla_v2_kernel_body_all_branches(monkeypatch) -> None:
    # 1. One-step flash attention, non-transposed, batch_size=2, all scales/caps
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=False,
        two_step_flash_attention=False,
        bkv_p=2,
        bq_sz=2,
        batch_size=2,
        static_q_len=None,
        use_causal_mask=True,
        sliding_window=64,
        soft_cap=30.0,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.5,
        p_same_dtype_as_v=True,
        q_dtype=jnp.bfloat16,
        debug_mode=False,
        same_minor_shape=False,
    )

    # 2. One-step flash attention, transposed, batch_size=1, causal=False, float32
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=True,
        two_step_flash_attention=False,
        bkv_p=1,
        bq_sz=2,
        batch_size=1,
        static_q_len=2,
        use_causal_mask=False,
        sliding_window=None,
        soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        p_same_dtype_as_v=False,
        q_dtype=jnp.float32,
        debug_mode=False,
        same_minor_shape=True,
    )

    # 3. One-step flash attention with debug_mode=True
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=False,
        two_step_flash_attention=False,
        bkv_p=1,
        bq_sz=2,
        batch_size=1,
        static_q_len=1,
        debug_mode=True,
    )

    # 4. Two-step flash attention, non-transposed, bkv_sz=256, causal=True, window=64
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=False,
        two_step_flash_attention=True,
        bkv_p=2,
        bq_sz=4,
        batch_size=1,
        q_split=2,
        static_q_len=4,
        use_causal_mask=True,
        sliding_window=64,
        soft_cap=20.0,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.5,
        p_same_dtype_as_v=True,
    )

    # 5. Two-step flash attention, transposed, bkv_sz=256, causal=True, window=None
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=True,
        two_step_flash_attention=True,
        bkv_p=2,
        bq_sz=4,
        batch_size=1,
        q_split=2,
        static_q_len=4,
        use_causal_mask=True,
        sliding_window=None,
        soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        p_same_dtype_as_v=False,
    )

    # 6. Two-step flash attention, non-transposed, bkv_sz=128, causal=False, window=64
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=False,
        two_step_flash_attention=True,
        bkv_p=1,
        bq_sz=2,
        batch_size=1,
        q_split=1,
        static_q_len=2,
        use_causal_mask=False,
        sliding_window=64,
    )

    # 7. Two-step flash attention, transposed, bkv_sz=128, causal=False, window=None
    _run_kernel_variant(
        monkeypatch,
        transpose_kv_cache=True,
        two_step_flash_attention=True,
        bkv_p=1,
        bq_sz=2,
        batch_size=1,
        q_split=1,
        static_q_len=2,
        use_causal_mask=False,
        sliding_window=None,
    )


def test_masked_dense_cost_model_methods() -> None:
    from vllm_torchtpu.kernels.mla.masked_dense.cost_model import (
        GLM52_TPU7X_PREFILL_COST_MODEL,
    )

    assert GLM52_TPU7X_PREFILL_COST_MODEL.estimate_analytic_cost_ns(10, 2) > 0
    assert GLM52_TPU7X_PREFILL_COST_MODEL.estimate_bitmap_cost_ns(10, 2) > 0
    assert GLM52_TPU7X_PREFILL_COST_MODEL.estimate_sparse_cost_ns(10) > 0
