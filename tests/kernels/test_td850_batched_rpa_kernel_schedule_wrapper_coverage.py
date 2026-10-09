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
"""Comprehensive tests for TD8.50 Batched RPA optimizations and 100% coverage."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    flash_attention,
    kernel,
    schedule,
    schedule_cp,
    wrapper,
)


def _mock_tpu_info(
    generation: int = 7,
    chip_version: str = "v7x",
    num_lanes: int = 128,
    num_sublanes: int = 8,
    mxu_column_size: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    smem_capacity_bytes: int = 1024 * 1024,
    fp8_ops_per_second: int = 2,
    bf16_ops_per_second: int = 1,
):
    info = MagicMock()
    info.generation = generation
    info.chip_version = chip_version
    info.num_lanes = num_lanes
    info.num_sublanes = num_sublanes
    info.mxu_column_size = mxu_column_size
    info.vmem_capacity_bytes = vmem_capacity_bytes
    info.smem_capacity_bytes = smem_capacity_bytes
    info.fp8_ops_per_second = fp8_ops_per_second
    info.bf16_ops_per_second = bf16_ops_per_second
    return info


def _make_cfgs(
    *,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    mask_value: float = -1e9,
    sm_scale: float = 0.5,
    soft_cap: float | None = None,
    sliding_window: int | None = None,
    num_seqs: int = 2,
    page_size: int = 128,
    total_q_tokens: int = 4,
    num_page_indices: int = 8,
    dtype_q=jnp.bfloat16,
    dtype_kv=jnp.bfloat16,
    dtype_out=jnp.float32,
    per_token_scale: bool | None = False,
    per_token_scale_dtype=None,
    scale_q=None,
    scale_k=None,
    scale_v=None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    skip_kv_update: bool = False,
    bq_sz: int = 2,
    bq_c_sz: int = 1,
    bkv_sz: int = 128,
    batch_size: int = 2,
    n_buffer: int = 2,
    mode: configs.RpaCase = configs.RpaCase.DECODE,
    vmem_limit_bytes: int = 64 * 1024 * 1024,
) -> configs.RpaConfigs:
    block = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_c_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=n_buffer,
    )
    model = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        mask_value=mask_value,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    serve = configs.ServingConfigs(
        num_seqs=num_seqs,
        page_size=page_size,
        total_q_tokens=total_q_tokens,
        num_page_indices=num_page_indices,
        dtype_q=jnp.dtype(dtype_q),
        dtype_kv=jnp.dtype(dtype_kv),
        dtype_out=jnp.dtype(dtype_out),
        per_token_scale=per_token_scale,
        per_token_scale_dtype=(
            jnp.dtype(per_token_scale_dtype)
            if per_token_scale_dtype is not None
            else None
        ),
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        skip_kv_update=skip_kv_update,
    )
    return configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        mode=mode,
        vmem_limit_bytes=vmem_limit_bytes,
    )


def _eager_fori_loop(lower, upper, body_fn, init_val):
    val = init_val
    for i in range(int(lower), int(upper)):
        val = body_fn(jnp.int32(i), val)
    return val


def _eager_cond(pred, true_fun, false_fun, *operands):
    if bool(np.asarray(pred)):
        return true_fun(*operands)
    return false_fun(*operands)


def _pallas_bitcast(x, dt):
    target_dt = jnp.dtype(dt)
    if target_dt == jnp.float4_e2m1fn:
        return jnp.zeros(
            (*x.shape[:-2], x.shape[-2] * 8, x.shape[-1]), dtype=jnp.bfloat16
        )
    src_bits = jax.dtypes.itemsize_bits(x.dtype)
    dst_bits = jax.dtypes.itemsize_bits(target_dt)
    if x.ndim >= 2:
        new_second_last = (x.shape[-2] * src_bits) // dst_bits
        return jnp.zeros((*x.shape[:-2], new_second_last, x.shape[-1]), dtype=target_dt)
    return jnp.zeros(((x.shape[0] * src_bits) // dst_bits,), dtype=target_dt)


class _FakeRef:
    """NumPy-backed stand-in for Pallas/JAX Ref."""

    def __init__(self, arr: np.ndarray, memory_space=None):
        self.arr = arr
        self.memory_space = memory_space

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return jnp.dtype(self.arr.dtype)

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def size(self):
        return self.arr.size

    def _convert_idx(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._convert_idx(i) for i in idx)
        if isinstance(idx, pl.Slice):
            start = int(np.asarray(idx.start))
            size = int(np.asarray(idx.size))
            stride = int(np.asarray(idx.stride))
            return slice(start, start + size * stride, stride)
        if isinstance(idx, (jax.Array, np.ndarray, np.generic)):
            arr = np.asarray(idx)
            if arr.ndim == 0:
                return int(arr)
            return arr
        return idx

    def __getitem__(self, idx):
        if idx is Ellipsis:
            if self.arr.shape == (1,):
                return jnp.asarray(self.arr[0])
            return jnp.asarray(self.arr)
        c_idx = self._convert_idx(idx)
        return jnp.asarray(self.arr[c_idx])

    def __setitem__(self, idx, val):
        if idx is Ellipsis:
            self.arr[...] = np.asarray(val, dtype=self.arr.dtype)
            return
        c_idx = self._convert_idx(idx)
        self.arr[c_idx] = np.asarray(val, dtype=self.arr.dtype)

    @property
    def at(self):
        parent = self

        class _At:
            def __getitem__(self, idx):
                c_idx = parent._convert_idx(idx)
                if isinstance(c_idx, int) and parent.arr.ndim == 1:
                    view = parent.arr[c_idx : c_idx + 1]
                else:
                    view = parent.arr[c_idx]
                return _FakeRef(view, memory_space=parent.memory_space)

        return _At()

    def bitcast(self, target_dtype):
        np_dt = np.dtype(jnp.dtype(target_dtype))
        src_bits = self.arr.dtype.itemsize * 8
        dst_bits = np_dt.itemsize * 8
        if self.arr.ndim >= 2:
            new_second_last = (self.arr.shape[-2] * src_bits) // dst_bits
            new_shape = (*self.arr.shape[:-2], new_second_last, self.arr.shape[-1])
        else:
            new_shape = ((self.arr.shape[0] * src_bits) // dst_bits,)
        return _FakeRef(
            np.zeros(new_shape, dtype=np_dt), memory_space=self.memory_space
        )

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _FakeRef(self.arr.reshape(shape), memory_space=self.memory_space)


def _materialize_schedule(
    cfgs: configs.RpaConfigs,
    computer_cls: type[schedule.BaseMetadataComputer] = schedule.BaseMetadataComputer,
    multiplier: int = 1,
    memory_space=None,
) -> schedule.RpaSchedule:
    shaped = computer_cls.get_rpa_schedule(cfgs, multiplier=multiplier)
    return jax.tree.map(
        lambda x: _FakeRef(
            np.zeros(x.shape, dtype=np.dtype(x.dtype)), memory_space=memory_space
        ),
        shaped,
    )


# ---------------------------------------------------------------------------
# 1. flash_attention.py tests
# ---------------------------------------------------------------------------


def test_flash_attention_qk_softmax_and_pv_all_branches():
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_mock_tpu_info(num_lanes=8, num_sublanes=2),
    ):
        # Case A: SEQ_ALONG_LANE, scale_q set, floating k, k_scale set, soft_cap set,
        # per_token_scale=True in pv
        cfgs_a = _make_cfgs(
            batch_size=2,
            num_kv_heads=2,
            num_q_heads=4,
            head_dim=8,
            bkv_sz=8,
            scale_q=2.0,
            soft_cap=30.0,
            per_token_scale=True,
            per_token_scale_dtype=jnp.float32,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            dtype_out=jnp.float32,
        )
        q = jnp.ones((2, 2, 4, 8), dtype=jnp.bfloat16)
        k_seq = jnp.ones((2, 2, 8, 8), dtype=jnp.bfloat16)
        v_seq = jnp.ones((2, 2, 8, 8), dtype=jnp.bfloat16)
        m_prev = jnp.full((2, 4, 8), -jnp.inf, dtype=jnp.float32)
        l_prev = jnp.zeros((2, 4, 8), dtype=jnp.float32)
        is_last_k = jnp.array([[1, 0]], dtype=jnp.int32)
        custom_mask = [
            jnp.ones((2, 4, 8), dtype=jnp.bool_),
            jnp.ones((2, 4, 8), dtype=jnp.bool_),
        ]
        k_scale = jnp.ones((2, 2, 1, 8), dtype=jnp.float32) * 0.5
        v_scale = jnp.ones((2, 2, 8), dtype=jnp.float32) * 0.5

        p, alpha_list, m_next, l_next, m_carry = (
            flash_attention.flash_attention_qk_softmax(
                jnp.int32(0),
                q,
                k_seq,
                m_prev,
                l_prev,
                is_last_k,
                custom_mask=custom_mask,
                cfgs=cfgs_a,
                bq_start=0,
                k_scale=k_scale,
            )
        )
        assert p.shape == (2, 2, 4, 8)
        assert len(alpha_list) == 2
        assert m_next.shape == (2, 2, 4, 8)
        assert l_next.shape == (2, 2, 4, 8)
        assert m_carry.shape == (2, 4, 8)

        o_prev = jnp.zeros((2, 4, 8), dtype=jnp.float32)
        o_next = flash_attention.flash_attention_pv(
            p, v_seq, alpha_list, o_prev, cfgs=cfgs_a, v_scale=v_scale
        )
        assert o_next.shape == (2, 2, 4, 8)

        # Case B: HEAD_ALONG_SUBLANE, scale_q set with integer k (skips floating clip),
        # k_scale=None, soft_cap=None, per_token_scale=False, scale_v set
        cfgs_b = _make_cfgs(
            batch_size=2,
            num_kv_heads=2,
            num_q_heads=4,
            head_dim=8,
            bkv_sz=8,
            scale_q=2.0,
            scale_v=1.5,
            soft_cap=None,
            per_token_scale=False,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            dtype_out=jnp.float32,
        )
        k_int = jnp.ones((2, 2, 8, 8), dtype=jnp.int8)
        v_sub = jnp.ones((2, 2, 8, 8), dtype=jnp.bfloat16)
        p2, alpha_list2, _, _, _ = flash_attention.flash_attention_qk_softmax(
            jnp.int32(0),
            q,
            k_int,
            m_prev,
            l_prev,
            is_last_k,
            custom_mask=custom_mask,
            cfgs=cfgs_b,
            bq_start=0,
            k_scale=None,
        )
        o_next2 = flash_attention.flash_attention_pv(
            p2, v_sub, alpha_list2, o_prev, cfgs=cfgs_b, v_scale=None
        )
        assert o_next2.shape == (2, 2, 4, 8)

        # Case C: scale_q=None, scale_v=None
        cfgs_c = _make_cfgs(
            batch_size=1,
            num_kv_heads=2,
            num_q_heads=4,
            head_dim=8,
            bkv_sz=8,
            scale_q=None,
            scale_v=None,
            per_token_scale=False,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            dtype_out=jnp.float32,
        )
        p3, alpha_list3, _, _, _ = flash_attention.flash_attention_qk_softmax(
            jnp.int32(0),
            q[:1],
            v_sub[:1],
            m_prev,
            l_prev,
            is_last_k,
            custom_mask=custom_mask[:1],
            cfgs=cfgs_c,
            bq_start=0,
            k_scale=None,
        )
        o_next3 = flash_attention.flash_attention_pv(
            p3, v_sub[:1], alpha_list3, o_prev, cfgs=cfgs_c
        )
        assert o_next3.shape == (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# 2. schedule.py & schedule_cp.py tests
# ---------------------------------------------------------------------------


def test_schedule_structs_and_wrappers():
    # Abstract FieldOffset and concrete FieldOffset
    abs_fo = schedule.FieldOffset(is_abstract=True)
    assert abs_fo.offset is None
    conc_fo = schedule.FieldOffset(offset=2, is_abstract=False)
    assert conc_fo.offset == 2

    # Concrete SeqAlongLaneDmaNew, HeadAlongSublaneDmaNew, HeadAlongSublaneDmaNewCP
    assert schedule.SeqAlongLaneDmaNew.num_fields() == 5
    assert schedule.HeadAlongSublaneDmaNew.num_fields() == 4
    assert schedule_cp.HeadAlongSublaneDmaNewCP.num_fields() == 5

    buf_seq = _FakeRef(np.zeros((10,), dtype=np.int32))
    aos_seq = schedule.SmemArrayOfStructs(
        data=buf_seq,
        shape=(2,),
        struct_cls=schedule.SeqAlongLaneDmaNew,
        struct_size=5,
    )
    entry_seq = aos_seq[(1,)]
    entry_seq.fetch_hbm[...] = 11
    entry_seq.fetch_vmem[...] = 22
    entry_seq.wb_hbm[...] = 33
    entry_seq.wb_vmem[...] = 44
    entry_seq.set_flags(1, 1)
    assert int(entry_seq.fetch_val) == 1
    assert int(entry_seq.wb_val) == 1

    buf_head = _FakeRef(np.zeros((8,), dtype=np.int32))
    aos_head = schedule.SmemArrayOfStructs(
        data=buf_head,
        shape=(2,),
        struct_cls=schedule.HeadAlongSublaneDmaNew,
        struct_size=4,
    )
    entry_head = aos_head[(1,)]
    entry_head.set_flags(64, 64)
    assert int(entry_head.fetch_val) == 64
    assert int(entry_head.wb_val) == 64

    buf_cp = _FakeRef(np.zeros((10,), dtype=np.int32))
    aos_cp = schedule.SmemArrayOfStructs(
        data=buf_cp,
        shape=(2,),
        struct_cls=schedule_cp.HeadAlongSublaneDmaNewCP,
        struct_size=5,
    )
    entry_cp = aos_cp[(1,)]
    entry_cp.set_flags(32, 16)
    assert int(entry_cp.fetch_val) == 32
    assert int(entry_cp.wb_val) == 16

    # SmemWrapper 1D and 2D indexing
    w1d = schedule.SmemWrapper(
        data=_FakeRef(np.zeros((4,), dtype=np.int32)), shape=(4,)
    )
    w1d[2] = 99
    assert int(w1d[2]) == 99


def test_schedule_metadata_generation_and_cp_all_branches():
    with (
        patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_mock_tpu_info(
                num_lanes=8, num_sublanes=2, smem_capacity_bytes=64 * 1024
            ),
        ),
        patch.object(jax.lax, "fori_loop", side_effect=_eager_fori_loop),
        patch.object(jax.lax, "cond", side_effect=_eager_cond),
    ):
        # 1. RpaSchedule specs & shapes
        cfgs_seq_mix = _make_cfgs(
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            mode=configs.RpaCase.MIXED,
            bq_sz=2,
            bkv_sz=128,
            page_size=128,
            batch_size=2,
            sliding_window=128,
            attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
            skip_kv_update=False,
        )
        sched_struct = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs_seq_mix)
        assert sched_struct.scratch_shapes() is not None
        assert sched_struct.in_specs() is not None
        assert sched_struct.out_specs() is not None

        # Test BaseMetadataComputer via mocked pallas_call
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        q_offsets = jnp.array([128, 0], dtype=jnp.int32)
        kv_cache_lens = jnp.array([128, 0], dtype=jnp.int32)
        kv_new_lens = jnp.array([2, 2], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

        def _fake_pl_loop(start, stop):
            def _dec(fn):
                for idx in range(int(start), int(stop)):
                    fn(idx)
                return fn

            return _dec

        class _FakeCopy:
            def __init__(self):
                self.started = False
                self.waited = False

            def start(self):
                self.started = True

            def wait(self):
                self.waited = True

        def _run_metadata_kernel(
            cfgs, computer_cls, extra_scalars=(), dist=distribution
        ):
            sched_smem = _materialize_schedule(cfgs, computer_cls=computer_cls)
            sched_hbm = _materialize_schedule(
                cfgs,
                computer_cls=computer_cls,
                multiplier=cfgs.max_schedule_size_multiplier,
            )
            dma_sem = _FakeRef(np.zeros((1,), dtype=np.int32))

            def _fake_pallas_call(kernel_fn, **kwargs):
                def _invoker(*args):
                    kernel_fn(
                        cu_q_lens,
                        q_offsets,
                        kv_cache_lens,
                        kv_new_lens,
                        dist,
                        extra_scalars,
                        sched_hbm,
                        sched_smem,
                        dma_sem,
                    )
                    return sched_hbm

                return _invoker

            with (
                patch.object(pl, "pallas_call", side_effect=_fake_pallas_call),
                patch.object(pl, "loop", side_effect=_fake_pl_loop),
                patch.object(pltpu, "make_async_copy", return_value=_FakeCopy()),
            ):
                return schedule.generate_rpa_metadata(
                    cu_q_lens,
                    q_offsets,
                    kv_cache_lens,
                    kv_new_lens,
                    dist,
                    cfgs=cfgs,
                    computer_cls=computer_cls,
                    extra_scalars=extra_scalars,
                )

        dist_odd = jnp.array([0, 0, 1], dtype=jnp.int32)
        with (
            patch.object(pl, "loop", side_effect=_fake_pl_loop),
            patch.object(pltpu, "make_async_copy", return_value=_FakeCopy()),
        ):
            _run_metadata_kernel(
                cfgs_seq_mix, schedule.BaseMetadataComputer, dist=dist_odd
            )

        # Run BaseMetadataComputer: SEQ_ALONG_LANE, DECODE with bkv_p_new < bkv_p
        cfgs_seq_dec = _make_cfgs(
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            mode=configs.RpaCase.DECODE,
            bq_sz=1,
            bkv_sz=256,
            page_size=128,
            batch_size=2,
            sliding_window=None,
            attention_scope=configs.AttentionScope.FULL,
            skip_kv_update=False,
        )
        _run_metadata_kernel(
            cfgs_seq_dec,
            schedule.BaseMetadataComputer,
            dist=jnp.array([1, 1, 1], dtype=jnp.int32),
        )

        # Run BaseMetadataComputer: HEAD_ALONG_SUBLANE, DECODE, skip_kv_update=True
        cfgs_head_dec = _make_cfgs(
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            mode=configs.RpaCase.DECODE,
            bq_sz=1,
            bkv_sz=256,
            page_size=128,
            batch_size=2,
            sliding_window=None,
            attention_scope=configs.AttentionScope.FULL,
            skip_kv_update=True,
        )
        _run_metadata_kernel(
            cfgs_head_dec,
            schedule.BaseMetadataComputer,
            dist=jnp.array([1, 1, 1], dtype=jnp.int32),
        )

        # Run BaseMetadataComputer: HEAD_ALONG_SUBLANE, MIXED (bkv_p_new >= bkv_p)
        cfgs_head_mix = _make_cfgs(
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            mode=configs.RpaCase.MIXED,
            bq_sz=2,
            bkv_sz=128,
            page_size=128,
            batch_size=1,
            sliding_window=None,
            attention_scope=configs.AttentionScope.FULL,
            skip_kv_update=False,
        )
        sched_smem_dec = _materialize_schedule(
            cfgs_head_mix, computer_cls=schedule.BaseMetadataComputer
        )
        sched_hbm_dec = _materialize_schedule(
            cfgs_head_mix, computer_cls=schedule.BaseMetadataComputer
        )
        dma_sem_dec = _FakeRef(np.zeros((1,), dtype=np.int32))
        comp_dec = schedule.BaseMetadataComputer(
            schedule=sched_smem_dec,
            schedule_hbm_ref=sched_hbm_dec,
            dma_sem=dma_sem_dec,
            cfgs=cfgs_head_mix,
        )
        with (
            patch.object(pl, "loop", side_effect=_fake_pl_loop),
            patch.object(pltpu, "make_async_copy", return_value=_FakeCopy()),
        ):
            comp_dec.compute(
                cu_q_lens_ref=cu_q_lens,
                q_offsets_ref=q_offsets,
                kv_cache_lens_ref=kv_cache_lens,
                kv_new_lens_ref=kv_new_lens,
                distribution_ref=distribution,
            )
            # Explicitly test flush_carry at max_steps_ub * batch_size
            flushed = comp_dec.advance_carry(
                schedule.LoopCarry(
                    0, cfgs_head_mix.max_steps_ub * cfgs_head_mix.batch_size - 1
                )
            )
            assert int(flushed.count) == 0

        # Run CPMetadataComputer across all layout / attention_scope combinations
        for kv_layout in (
            configs.KVLayout.SEQ_ALONG_LANE,
            configs.KVLayout.HEAD_ALONG_SUBLANE,
        ):
            for scope, mode, bq_sz, bkv_sz in (
                (
                    configs.AttentionScope.NEW_TOKENS_ONLY,
                    configs.RpaCase.DECODE,
                    1,
                    256,
                ),
                (configs.AttentionScope.CACHE_ONLY, configs.RpaCase.MIXED, 2, 128),
            ):
                cfgs_cp = _make_cfgs(
                    kv_layout=kv_layout,
                    mode=mode,
                    bq_sz=bq_sz,
                    bkv_sz=bkv_sz,
                    page_size=128,
                    batch_size=2,
                    cp_group_size=2,
                    attention_scope=scope,
                )
                extra = (
                    (jnp.array([1], dtype=jnp.int32),)
                    if scope == configs.AttentionScope.NEW_TOKENS_ONLY
                    else ()
                )
                sched_cp_smem = _materialize_schedule(
                    cfgs_cp, computer_cls=schedule_cp.CPMetadataComputer
                )
                sched_cp_hbm = _materialize_schedule(
                    cfgs_cp, computer_cls=schedule_cp.CPMetadataComputer
                )
                comp_cp = schedule_cp.CPMetadataComputer(
                    schedule=sched_cp_smem,
                    schedule_hbm_ref=sched_cp_hbm,
                    dma_sem=dma_sem_dec,
                    cfgs=cfgs_cp,
                    extra_refs=extra,
                )
                dist = (
                    jnp.array([2, 2, 2], dtype=jnp.int32)
                    if mode == configs.RpaCase.DECODE
                    else distribution
                )
                with (
                    patch.object(pl, "loop", side_effect=_fake_pl_loop),
                    patch.object(pltpu, "make_async_copy", return_value=_FakeCopy()),
                ):
                    comp_cp.compute(
                        cu_q_lens_ref=cu_q_lens,
                        q_offsets_ref=q_offsets,
                        kv_cache_lens_ref=kv_cache_lens,
                        kv_new_lens_ref=kv_new_lens,
                        distribution_ref=dist,
                    )


# ---------------------------------------------------------------------------
# 3. kernel.py tests
# ---------------------------------------------------------------------------


def test_kernel_helpers_and_rpa_body_and_pipeline():
    tpu_info = _mock_tpu_info(num_lanes=8, num_sublanes=2)
    with (
        patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info),
        patch.object(jax.lax, "fori_loop", side_effect=_eager_fori_loop),
        patch.object(jax.lax, "cond", side_effect=_eager_cond),
        patch.object(pltpu, "bitcast", side_effect=_pallas_bitcast),
        patch.object(pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x),
    ):
        # 1. strided_load_bkv: packing_kv == 1 (float32) vs packing_kv > 1 (bfloat16)
        cfgs_f32 = _make_cfgs(
            dtype_kv=jnp.float32,
            head_dim=8,
            bkv_sz=8,
            batch_size=1,
            num_kv_heads=2,
        )
        kv_vref_f32 = _FakeRef(np.zeros(cfgs_f32.kv_vmem_shape, dtype=np.float32))
        res_f32 = kernel.strided_load_bkv(kv_vref_f32, 0, 0, cfgs=cfgs_f32)
        assert len(res_f32) == 1

        cfgs_bf16 = _make_cfgs(
            dtype_kv=jnp.bfloat16,
            head_dim=8,
            bkv_sz=8,
            batch_size=1,
            num_kv_heads=2,
        )
        kv_vref_bf16 = _FakeRef(
            np.zeros(cfgs_bf16.kv_vmem_shape, dtype=np.dtype(jnp.bfloat16))
        )
        res_bf16 = kernel.strided_load_bkv(kv_vref_bf16, 0, 0, cfgs=cfgs_bf16)
        assert len(res_bf16) == 1

        # 2. get_scale_factors: per_token_scale=False, per_token_scale=True with
        # per_token_scale_dtype=None, scale_channels=1, and scale_channels>1
        k_arr = jnp.ones((1, 2, 12, 8), dtype=jnp.int8)
        v_arr = jnp.ones((1, 2, 12, 8), dtype=jnp.int8)
        cfgs_no_pts = _make_cfgs(
            per_token_scale=False, scale_k=0.5, scale_v=0.25, head_dim=8, bkv_sz=8
        )
        _, _, sk, sv = kernel.get_scale_factors(k_arr, v_arr, cfgs=cfgs_no_pts)
        assert sk == 0.5 and sv == 0.25

        cfgs_pts_none = _make_cfgs(
            per_token_scale=True,
            per_token_scale_dtype=None,
            head_dim=8,
            bkv_sz=8,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

        class _MockScaleArray:
            def __init__(self, dt=jnp.float32):
                self.dtype = dt

            def __getitem__(self, idx):
                return self

            def astype(self, dt):
                return self

            def reshape(self, *shape):
                return jnp.ones(shape, dtype=jnp.float32)

        class _MockKV:
            shape = (1, 2, 8, 8)
            dtype = jnp.int8

            def __getitem__(self, idx):
                return _MockScaleArray()

        kernel.get_scale_factors(_MockKV(), _MockKV(), cfgs=cfgs_pts_none)

        # scale_channels == 1 (dtype_kv=bfloat16, per_token_scale_dtype=bfloat16)
        cfgs_pts_1 = _make_cfgs(
            per_token_scale=True,
            per_token_scale_dtype=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            head_dim=8,
            bkv_sz=8,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )
        kv_1 = jnp.ones((1, 2, 9, 8), dtype=jnp.bfloat16)
        _, _, sk1, sv1 = kernel.get_scale_factors(kv_1, kv_1, cfgs=cfgs_pts_1)
        assert sk1.shape == (1, 2, 1, 8)
        assert sv1.shape == (1, 2, 8)

        # scale_channels > 1 (dtype_kv=int8, per_token_scale_dtype=float32)
        cfgs_pts_4 = _make_cfgs(
            per_token_scale=True,
            per_token_scale_dtype=jnp.float32,
            dtype_kv=jnp.int8,
            head_dim=8,
            bkv_sz=8,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )
        kv_4 = jnp.ones((1, 2, 12, 8), dtype=jnp.int8)
        _, _, sk4, sv4 = kernel.get_scale_factors(kv_4, kv_4, cfgs=cfgs_pts_4)
        assert sk4.shape == (1, 2, 1, 8)
        assert sv4.shape == (1, 2, 8)

        # 3. fetch_step_metadata and generate_mask across all scopes and skip_kv_update
        for scope, skip_kv, sw in (
            (configs.AttentionScope.NEW_TOKENS_ONLY, False, 64),
            (configs.AttentionScope.CACHE_ONLY, True, None),
            (configs.AttentionScope.FULL, False, None),
        ):
            cfgs_meta = _make_cfgs(
                attention_scope=scope,
                skip_kv_update=skip_kv,
                sliding_window=sw,
                batch_size=2,
                bkv_sz=8,
                bq_sz=2,
                head_dim=8,
            )
            sched_meta = _materialize_schedule(cfgs_meta)
            sched_meta.s_idx[0, 0] = 0
            sched_meta.s_idx[0, 1] = -1
            cu_q = jnp.array([0, 2, 4], dtype=jnp.int32)
            q_off = jnp.array([4, 0], dtype=jnp.int32)
            kv_c = jnp.array([4, 0], dtype=jnp.int32)
            kv_n = jnp.array([2, 0], dtype=jnp.int32)
            step_meta = kernel.fetch_step_metadata(
                jnp.int32(0),
                sched_meta,
                cu_q,
                q_off,
                kv_c,
                kv_n,
                cfgs=cfgs_meta,
            )
            masks = kernel.generate_mask(
                (2, 2, 2, 8), bq_start=0, step_meta=step_meta, cfgs=cfgs_meta
            )
            assert len(masks) == 2

        # 4. calculate_and_store_out:
        # Case 1: pad_amount > 0 (SEQ_ALONG_LANE with head_dim=2, dtype_kv=bfloat16),
        # dtype_out=bfloat16 with float32 batch_l (hits pl.reciprocal),
        # mode=MIXED (fuse_accum=False), return_lse=True, batch_size=2
        cfgs_out1 = _make_cfgs(
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            head_dim=2,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            mode=configs.RpaCase.MIXED,
            return_lse=True,
            batch_size=2,
            bq_sz=1,
            bkv_sz=8,
        )
        assert cfgs_out1.aligned_q_head_dim > cfgs_out1.compute_kv_head_dim
        sched_out1 = _materialize_schedule(cfgs_out1)
        sched_out1.is_last_k[0, 0] = 1
        sched_out1.is_last_k[0, 1] = 0
        acc_list = [
            jnp.ones((2, 2, cfgs_out1.compute_kv_head_dim), dtype=jnp.float32)
            for _ in range(2)
        ]
        l_list = [jnp.ones((2, 2, 8), dtype=jnp.float32) for _ in range(2)]
        m_list = [jnp.zeros((2, 2, 8), dtype=jnp.float32) for _ in range(2)]
        o_vref = _FakeRef(
            np.zeros(cfgs_out1.q_vmem_shape, dtype=np.dtype(jnp.bfloat16))
        )
        lse_vref = _FakeRef(
            np.zeros(cfgs_out1.lse_vmem_shape, dtype=np.dtype(jnp.bfloat16))
        )
        kernel.calculate_and_store_out(
            jnp.int32(0),
            sched_out1,
            acc_list,
            l_list,
            m_list,
            o_vref,
            lse_vref,
            cfgs=cfgs_out1,
        )

        # Case 2: pad_amount == 0 (head_dim=8 == num_lanes=8), dtype_out=float32,
        # mode=DECODE (fuse_accum=True), return_lse=False
        cfgs_out2 = _make_cfgs(
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            head_dim=8,
            dtype_out=jnp.float32,
            mode=configs.RpaCase.DECODE,
            return_lse=False,
            batch_size=1,
            bq_sz=1,
            bkv_sz=8,
        )
        sched_out2 = _materialize_schedule(cfgs_out2)
        o_vref2 = _FakeRef(np.zeros(cfgs_out2.q_vmem_shape, dtype=np.float32))
        kernel.calculate_and_store_out(
            jnp.int32(0),
            sched_out2,
            [jnp.ones((2, 2, 8), dtype=jnp.float32)],
            [jnp.ones((2, 2, 8), dtype=jnp.float32)],
            [],
            o_vref2,
            None,
            cfgs=cfgs_out2,
        )

        # 5. rpa_body:
        # Test SEQ_ALONG_LANE (both uint8 fp4 and bfloat16) and HEAD_ALONG_SUBLANE,
        # with bq_sz=2, bq_c_sz=1 (multiple bq chunks so prev_p is not None on 2nd!)
        for kv_layout, dtype_kv, return_lse in (
            (configs.KVLayout.SEQ_ALONG_LANE, jnp.uint8, True),
            (configs.KVLayout.SEQ_ALONG_LANE, jnp.bfloat16, False),
            (configs.KVLayout.HEAD_ALONG_SUBLANE, jnp.bfloat16, True),
        ):
            hd = 16 if dtype_kv == jnp.uint8 else 8
            body_tpu_info = _mock_tpu_info(num_lanes=8, num_sublanes=2)
            with patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=body_tpu_info,
            ):
                cfgs_body = _make_cfgs(
                    kv_layout=kv_layout,
                    dtype_kv=dtype_kv,
                    dtype_q=jnp.bfloat16,
                    dtype_out=jnp.float32,
                    head_dim=hd,
                    bkv_sz=8,
                    page_size=8,
                    bq_sz=2,
                    bq_c_sz=1,
                    batch_size=1,
                    num_kv_heads=2,
                    num_q_heads=4,
                    return_lse=return_lse,
                )
                q_vref = _FakeRef(
                    np.zeros(cfgs_body.q_vmem_shape, dtype=np.dtype(jnp.bfloat16))
                )
                kv_vref = _FakeRef(
                    np.zeros(cfgs_body.kv_vmem_shape, dtype=np.dtype(dtype_kv))
                )
                o_vref = _FakeRef(np.zeros(cfgs_body.q_vmem_shape, dtype=np.float32))
                lse_vref = (
                    _FakeRef(np.zeros(cfgs_body.lse_vmem_shape, dtype=np.float32))
                    if return_lse
                    else None
                )
                sched_body = _materialize_schedule(cfgs_body)
                sched_body.is_last_k[0, 0] = 1
                m_scratch = _FakeRef(
                    np.full(cfgs_body.lm_scratch_shape, -np.inf, dtype=np.float32)
                )
                l_scratch = _FakeRef(
                    np.zeros(cfgs_body.lm_scratch_shape, dtype=np.float32)
                )
                acc_scratch = _FakeRef(
                    np.zeros(cfgs_body.acc_scratch_shape, dtype=np.float32)
                )
                cu_q = jnp.array([0, 2, 4], dtype=jnp.int32)
                q_off = jnp.array([0, 0], dtype=jnp.int32)
                kv_c = jnp.array([0, 0], dtype=jnp.int32)
                kv_n = jnp.array([2, 2], dtype=jnp.int32)

                with (
                    patch.object(pl, "program_id", return_value=jnp.int32(0)),
                    patch(
                        "vllm_torchtpu.kernels.experimental.batched_rpa.stitch_utils.stitch_new_kv_lane",
                        return_value="stitched",
                    ),
                    patch(
                        "vllm_torchtpu.kernels.experimental.batched_rpa.stitch_utils.store_new_kv_lane"
                    ),
                ):
                    kernel.rpa_body(
                        q_vref,
                        kv_vref,
                        o_vref,
                        lse_vref,
                        sched_body,
                        m_scratch,
                        l_scratch,
                        acc_scratch,
                        cu_q_lens_ref=cu_q,
                        q_offsets_ref=q_off,
                        kv_cache_lens_ref=kv_c,
                        kv_new_lens_ref=kv_n,
                        cfgs=cfgs_body,
                    )

        # 6. get_kernel_name & get_kernel_metadata
        cfgs_sw = _make_cfgs(sliding_window=256)
        assert "-sw256" in kernel.get_kernel_name(cfgs_sw)
        assert "-sw" not in kernel.get_kernel_name(_make_cfgs(sliding_window=None))
        meta = kernel.get_kernel_metadata(cfgs_sw)
        assert isinstance(meta["serve.dtype_q"], str)

        # 7. create_allocs & rpa_kernel pipeline execution
        for kv_layout, return_lse in (
            (configs.KVLayout.SEQ_ALONG_LANE, True),
            (configs.KVLayout.HEAD_ALONG_SUBLANE, False),
        ):
            cfgs_pipe = _make_cfgs(
                kv_layout=kv_layout,
                return_lse=return_lse,
                head_dim=8,
                bkv_sz=8,
                page_size=8,
                bq_sz=1,
                bq_c_sz=1,
                batch_size=1,
            )
            sched_hbm = _materialize_schedule(cfgs_pipe, memory_space=pltpu.HBM)
            sched_hbm.actual_steps.arr[:] = 1
            sched_hbm.actual_steps.memory_space = pltpu.SMEM
            sched_smem = _materialize_schedule(cfgs_pipe, memory_space=pltpu.SMEM)

            fake_alloc = SimpleNamespace(
                spec=pl.BlockSpec(memory_space=pltpu.VMEM),
                window_ref=_FakeRef(np.zeros((2, 8), dtype=np.uint32)),
            )

            def _fake_with_scoped(_sched_smem=sched_smem, **scoped_kwargs):
                def _dec(run_fn):
                    def _wrapped():
                        scratches = tuple(
                            _FakeRef(np.zeros(s.shape, dtype=np.float32))
                            for s in scoped_kwargs["scratches"]
                        )
                        dma_sem = _FakeRef(np.zeros((1,), dtype=np.int32))
                        run_fn(
                            scoped_kwargs["final_allocs"],
                            _sched_smem,
                            dma_sem,
                            scratches,
                        )

                    return _wrapped

                return _dec

            def _fake_loop(start, stop):
                def _dec(body_fn):
                    body_fn(jnp.int32(0))
                    return body_fn

                return _dec

            class _FakeCopy:
                def start(self):
                    pass

                def wait(self):
                    pass

            def _fake_pallas_call(pipeline_fn, **kwargs):
                def _call(*args):
                    cu_q, q_off, kv_c, kv_n, p_idx, s_hbm, q_h, nkv_h, kvc_h, lse_h = (
                        args
                    )
                    pipeline_fn(
                        cu_q,
                        q_off,
                        kv_c,
                        kv_n,
                        p_idx,
                        s_hbm,
                        q_h,
                        nkv_h,
                        kvc_h,
                        lse_h,
                        q_h,
                        kvc_h,
                        lse_h,
                    )
                    return q_h, kvc_h, lse_h

                return _call

            with (
                patch(
                    "vllm_torchtpu.kernels.experimental.batched_rpa.bref_override.KVBufferedRefSeqAlongLane.input_output",
                    return_value=fake_alloc,
                ),
                patch(
                    "vllm_torchtpu.kernels.experimental.batched_rpa.bref_override.KVBufferedRefHeadAlongSublane.input_output",
                    return_value=fake_alloc,
                ),
                patch(
                    "vllm_torchtpu.kernels.experimental.batched_rpa.bref_override.BatchingQRef.input",
                    return_value=fake_alloc,
                ),
                patch(
                    "vllm_torchtpu.kernels.experimental.batched_rpa.bref_override.BatchingORef.output",
                    return_value=fake_alloc,
                ),
                patch(
                    "vllm_torchtpu.kernels.experimental.batched_rpa.bref_override.BatchingLSERef.output",
                    return_value=fake_alloc,
                ),
                patch.object(pl, "with_scoped", side_effect=_fake_with_scoped),
                patch.object(pl, "loop", side_effect=_fake_loop),
                patch.object(pl, "pallas_call", side_effect=_fake_pallas_call),
                patch.object(pltpu, "make_async_copy", return_value=_FakeCopy()),
                patch.object(
                    pltpu, "emit_pipeline", return_value=lambda *a, **kw: None
                ),
                patch.object(
                    jax,
                    "typeof",
                    side_effect=lambda x: SimpleNamespace(
                        memory_space=getattr(x, "memory_space", pltpu.HBM)
                    ),
                ),
            ):
                q_hbm = jnp.zeros((2, 4, 1, 2, 8), dtype=jnp.bfloat16)
                new_kv_hbm = jnp.zeros((4, 4, 2, 8), dtype=jnp.bfloat16)
                kv_cache_hbm = jnp.zeros((4, 8, 2, 2, 8), dtype=jnp.bfloat16)
                lse_hbm = (
                    jnp.zeros((2, 4, 1, 2, 8), dtype=jnp.float32)
                    if return_lse
                    else None
                )
                out_q, out_kv, out_lse = kernel.rpa_kernel(
                    jnp.array([0, 2], dtype=jnp.int32),
                    jnp.array([0], dtype=jnp.int32),
                    jnp.array([0], dtype=jnp.int32),
                    jnp.array([2], dtype=jnp.int32),
                    jnp.array([0, 1], dtype=jnp.int32),
                    sched_hbm,
                    q_hbm,
                    new_kv_hbm,
                    kv_cache_hbm,
                    lse_hbm,
                    cfgs=cfgs_pipe,
                )
                assert out_q.shape == q_hbm.shape
                assert (out_lse is not None) == return_lse


# ---------------------------------------------------------------------------
# 4. wrapper.py tests
# ---------------------------------------------------------------------------


def test_wrapper_prepare_inputs_outputs_and_kv_cache_shape():
    tpu_info = _mock_tpu_info(num_lanes=128, num_sublanes=8)
    with (
        patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info),
        patch(
            "jax.experimental.pallas.tpu.get_tpu_info_for_chip",
            return_value=tpu_info,
        ),
    ):
        # 1. prepare_inputs: already aligned (skips pad) vs unaligned (pads)
        q_aligned = jnp.ones((128, 4, 128), dtype=jnp.bfloat16)
        k_aligned = jnp.ones((128, 2, 128), dtype=jnp.bfloat16)
        v_aligned = jnp.ones((128, 2, 128), dtype=jnp.bfloat16)
        o_alias, nkv = wrapper.prepare_inputs(
            q_aligned,
            k_aligned,
            v_aligned,
            jnp.bfloat16,
            jnp.bfloat16,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            page_size=128,
        )
        out_unpacked = wrapper.prepare_outputs(o_alias)
        assert out_unpacked.shape == (2, 128, 2, 128)
        assert nkv.shape == (128, 2, 2, 128)

        # Aligned SEQ_ALONG_LANE without scales: total_q_tokens=128, head_dim=128
        _, nkv_seq_aligned = wrapper.prepare_inputs(
            q_aligned,
            k_aligned,
            v_aligned,
            jnp.bfloat16,
            jnp.bfloat16,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            page_size=128,
        )
        assert nkv_seq_aligned.shape == (4, 64, 2, 128)

        # Unaligned SEQ_ALONG_LANE with per-token k_scale and v_scale:
        q_unaligned = jnp.ones((10, 3, 64), dtype=jnp.bfloat16)
        k_unaligned = jnp.ones((10, 1, 64), dtype=jnp.bfloat16)
        v_unaligned = jnp.ones((10, 1, 64), dtype=jnp.bfloat16)
        k_sc = jnp.ones((10, 1), dtype=jnp.float32)
        v_sc = jnp.ones((10, 1), dtype=jnp.float32)
        o_unaligned, nkv_seq_unaligned = wrapper.prepare_inputs(
            q_unaligned,
            k_unaligned,
            v_unaligned,
            jnp.bfloat16,
            jnp.bfloat16,
            k_scale=k_sc,
            v_scale=v_sc,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            page_size=128,
        )
        assert o_unaligned.shape == (1, 10, 2, 2, 128)
        assert nkv_seq_unaligned.shape[-1] == 128

        # Unaligned HEAD_ALONG_SUBLANE:
        _, nkv_head_unaligned = wrapper.prepare_inputs(
            q_unaligned,
            k_unaligned,
            v_unaligned,
            jnp.bfloat16,
            jnp.bfloat16,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            page_size=128,
        )
        assert nkv_head_unaligned.shape == (10, 1, 2, 128)

        # 2. get_kv_cache_shape: SEQ_ALONG_LANE and HEAD_ALONG_SUBLANE
        s_seq_no_pts = wrapper.get_kv_cache_shape(
            16,
            128,
            2,
            128,
            jnp.bfloat16,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            use_per_token_scale=False,
        )
        s_seq_pts = wrapper.get_kv_cache_shape(
            16,
            128,
            2,
            128,
            jnp.bfloat16,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            use_per_token_scale=True,
            scale_dtype=jnp.float32,
        )
        s_head = wrapper.get_kv_cache_shape(
            16,
            128,
            2,
            128,
            jnp.bfloat16,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        )
        assert s_seq_pts[2] >= s_seq_no_pts[2]
        assert s_head == (16, 128, 2, 2, 128)


def test_calculate_block_sizes_all_branches():
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    # 1. SEQ_ALONG_LANE + uint8 (is_packed_fp4) + per_token_scale=True
    serve_seq_fp4 = configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=8,
        num_page_indices=128,
        dtype_q=jnp.float8_e4m3fn,
        dtype_kv=jnp.uint8,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        per_token_scale=True,
        per_token_scale_dtype=jnp.float32,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_mock_tpu_info(generation=8, chip_version="v8i"),
    ):
        dec_b, pref_b = wrapper.calculate_block_sizes(
            model_cfg, serve_seq_fp4, 64 * 1024 * 1024
        )
        assert dec_b.bkv_sz >= 128
        assert pref_b.bkv_sz >= 128

    # 2. No bank conflict in HEAD_ALONG_SUBLANE (num_kv_heads=1, packing_kv=2)
    model_no_conflict = configs.ModelConfigs(
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_no_conflict = configs.ServingConfigs(
        num_seqs=2,
        page_size=16,
        total_q_tokens=8,
        num_page_indices=2,  # pages_per_seq=1 -> max_seq_len=16 < mxu_column_size=128!
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_mock_tpu_info(generation=5, chip_version="v5p"),
    ):
        # Also tests bkv_sz <= bkv_stride when max_seq_len=16 < bkv_stride=128
        dec_c, pref_c = wrapper.calculate_block_sizes(
            model_no_conflict, serve_no_conflict, 64 * 1024 * 1024
        )
        assert dec_c.bkv_sz == 128
        assert pref_c.bkv_sz == 128

    # 3. Tight VMEM reduces batch_size and n_buffer to 1, and OOM when n_buffer=0
    serve_tight = configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=8,
        num_page_indices=64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_mock_tpu_info(generation=7, chip_version="v7x"),
    ):
        # vmem_limit = 8,000,000 -> prefill drops batch_size to 1 and n_buffer to 1
        dec_tight, pref_tight = wrapper.calculate_block_sizes(
            model_cfg, serve_tight, 8_000_000
        )
        assert pref_tight.batch_size == 1
        assert pref_tight.n_buffer == 1

        # OOM when n_buffer drops to 0 (vmem_limit = 700,000 -> capped = 560,000)
        with pytest.raises(
            ValueError, match="Cannot find batch size that fits within VMEM limit"
        ):
            wrapper.calculate_block_sizes(model_cfg, serve_tight, 700_000)

    # 4. bkv_sz == 0 raises ValueError when mxu_column_size == 0 and max_seq_len < 0
    serve_neg_seq = dataclasses.replace(serve_no_conflict, num_page_indices=-2)
    with (
        patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_mock_tpu_info(
                generation=8, chip_version="v8t", mxu_column_size=0
            ),
        ),
        pytest.raises(
            ValueError, match="Cannot find block sizes that fit within VMEM limit"
        ),
    ):
        wrapper.calculate_block_sizes(
            model_cfg, serve_neg_seq, 64 * 1024 * 1024, decode_query_size=0
        )


def test_ragged_paged_attention_validation_and_dispatch():
    tpu_info = _mock_tpu_info(num_lanes=128, num_sublanes=8)
    q = jnp.ones((4, 4, 128), dtype=jnp.bfloat16)
    k = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    v = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    kv_cache_head = jnp.zeros((8, 128, 2, 2, 128), dtype=jnp.bfloat16)
    kv_cache_seq = jnp.zeros((8, 4, 64, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([128, 128], dtype=jnp.int32)
    page_indices = jnp.arange(8, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 2], dtype=jnp.int32)

    fn = wrapper.ragged_paged_attention.__wrapped__

    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info):
        # Error validations
        with pytest.raises(ValueError, match="Only causal attention is supported"):
            fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                use_causal_mask=False,
            )
        with pytest.raises(
            ValueError, match="Specifying chunk prefill size is not supported"
        ):
            fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                chunk_prefill_size=64,
            )
        with pytest.raises(ValueError, match="Debug mode is not supported"):
            fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                debug_mode=True,
            )
        with pytest.raises(
            ValueError, match="Only one of k_scale or dynamic_k_scale can be set"
        ):
            fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k_scale=1.0,
                dynamic_k_scale=jnp.ones((4, 2)),
            )
        with pytest.raises(
            ValueError, match="Only one of v_scale or dynamic_v_scale can be set"
        ):
            fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                v_scale=1.0,
                dynamic_v_scale=jnp.ones((4, 2)),
            )

        with (
            patch(
                "vllm_torchtpu.kernels.experimental.batched_rpa.schedule.generate_rpa_metadata",
                return_value="dummy_schedule",
            ),
            patch(
                "vllm_torchtpu.kernels.experimental.batched_rpa.kernel.rpa_kernel",
                side_effect=lambda *a, **kw: (a[6], a[8], a[9]),
            ),
        ):
            # 1. FULL scope, HEAD_ALONG_SUBLANE, return_lse=False, defaults
            out, new_kv = fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_scope=configs.AttentionScope.FULL,
                return_lse=False,
            )
            assert out.shape == q.shape
            assert new_kv.shape == kv_cache_head.shape

            # 2. CACHE_ONLY scope without CP, SEQ_ALONG_LANE, return_lse=True
            out2, new_kv2, lse2 = fn(
                q,
                k,
                v,
                kv_cache_seq,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                attention_scope=configs.AttentionScope.CACHE_ONLY,
                return_lse=True,
                out_dtype=jnp.float32,
                mask_value=-1e9,
                vmem_limit_bytes=64 * 1024 * 1024,
            )
            assert out2.shape == q.shape
            assert lse2.shape == (4, 4)

            # 3. CACHE_ONLY scope with CP and cp_rank provided
            out3, _, lse3 = fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                cp_group_size=2,
                cp_rank=jnp.array([1], dtype=jnp.int32),
                attention_scope=configs.AttentionScope.CACHE_ONLY,
                return_lse=True,
            )
            assert out3.shape == q.shape
            assert lse3.shape == (4, 4)

            # 4. NEW_TOKENS_ONLY scope with CP and cp_rank=None
            out4, _ = fn(
                q,
                k,
                v,
                kv_cache_head,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                cp_group_size=2,
                cp_rank=None,
                attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
                return_lse=False,
            )
            assert out4.shape == q.shape
