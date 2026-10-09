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
"""Comprehensive CPU unit tests and 100% coverage for Batched RPA TD8.49 modules."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    bref_override,
    configs,
    stitch_utils,
    utils,
)


@pytest.fixture(autouse=True)
def _mock_tpu_info(monkeypatch):
    info = SimpleNamespace(
        num_lanes=8,
        num_sublanes=2,
        vmem_capacity_bytes=16 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


class _FakeAt:
    def __init__(self, ref: "_FakeRef"):
        self._ref = ref

    def __getitem__(self, idx):
        return _FakeRef(self._ref.arr[self._ref._norm_idx(idx)])


class _FakeRef:
    """Lightweight CPU Pallas Ref stand-in (.at, .bitcast, .reshape, indexing)."""

    def __init__(self, arr: jax.Array):
        self.arr = arr

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
        return _FakeAt(self)

    def _norm_idx(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._norm_idx(i) for i in idx)
        if hasattr(idx, "start") and hasattr(idx, "size"):
            start = int(idx.start)
            size = int(idx.size)
            stride = int(getattr(idx, "stride", 1))
            return slice(start, start + size * stride, stride)
        if isinstance(idx, jax.Array) and idx.ndim == 0:
            return int(idx)
        return idx

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.arr
        return self.arr[self._norm_idx(idx)]

    def __setitem__(self, idx, val):
        raw_val = val.arr if isinstance(val, _FakeRef) else val
        if idx is Ellipsis:
            self.arr = jnp.asarray(raw_val, dtype=self.arr.dtype)
        else:
            self.arr = self.arr.at[self._norm_idx(idx)].set(
                jnp.asarray(raw_val, dtype=self.arr.dtype)
            )

    def bitcast(self, dtype):
        return _FakeRef(self.arr.astype(dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple | list):
            shape = tuple(shape[0])
        return _FakeRef(self.arr.reshape(*shape))


# ---------------------------------------------------------------------------
# 1. utils.py tests
# ---------------------------------------------------------------------------


def test_utils_all_functions(monkeypatch):
    assert utils.align_to(10, 8) == 16
    assert utils.align_to(16, 8) == 16

    # cp_local_cache_len
    local_len = utils.cp_local_cache_len(
        jnp.array([20, 40], dtype=jnp.int32),
        cp_group_size=2,
        cp_rank=0,
        page_size=8,
    )
    np.testing.assert_array_equal(local_len, np.array([12, 24], dtype=np.int32))

    # broadcast_minor: equal shape fast path and tiled path
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    assert utils.broadcast_minor(x, (2, 8)) is x
    bcast = utils.broadcast_minor(x, (2, 20))
    assert bcast.shape == (2, 20)

    # get_dtype_packing & get_dtype_bits (cached)
    assert utils.get_dtype_packing(jnp.float32) == 1
    assert utils.get_dtype_packing(jnp.bfloat16) == 2
    assert utils.get_dtype_bits(jnp.float32) == 32
    assert utils.get_dtype_bits(jnp.uint8) == 8

    # strided_load (dtype=None and dtype=jnp.int32)
    monkeypatch.setattr(pltpu, "bitcast", lambda val, dt: val.astype(dt))
    src = jnp.arange(4 * 16, dtype=jnp.float32).reshape(4, 16)
    loaded_none = utils.strided_load(src, start_row=0, num_rows=4, step=2, dtype=None)
    assert loaded_none.shape == (2, 16)
    loaded_cast = utils.strided_load(
        src, start_row=0, num_rows=4, step=2, dtype=jnp.int32
    )
    assert loaded_cast.dtype == jnp.int32

    # strided_store
    dst_ref = _FakeRef(jnp.zeros((4, 16), dtype=jnp.float32))
    val = jnp.ones((4, 16), dtype=jnp.float32)
    utils.strided_store(dst_ref, start=0, sz=4, step=1, val=val)

    # convert_to_target_bitwidth: direct (32 -> 16) and recursive (32 -> 8)
    val_u32 = jnp.ones((2, 8), dtype=jnp.uint32)
    out_16 = utils.convert_to_target_bitwidth(
        val_u32, target_bitwidth=16, kv_dtype=jnp.bfloat16
    )
    assert len(out_16) == 1
    out_8 = utils.convert_to_target_bitwidth(
        val_u32, target_bitwidth=8, kv_dtype=jnp.uint8
    )
    assert len(out_8) == 2

    # has_bank_conflicts: True and False
    assert utils.has_bank_conflicts(2, distance=24, num_banks=32) is True
    assert utils.has_bank_conflicts(1, distance=24, num_banks=32) is False


# ---------------------------------------------------------------------------
# 2. configs.py tests
# ---------------------------------------------------------------------------


def _make_rpa_cfg(
    *,
    mode: configs.RpaCase = configs.RpaCase.DECODE,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    bq_sz: int = 1,
    bkv_sz: int = 128,
    page_size: int = 128,
    num_kv_heads: int = 2,
    dtype_kv=jnp.bfloat16,
    per_token_scale: bool | None = None,
    per_token_scale_dtype=None,
    scale_k: int | None = None,
    scale_v: int | None = None,
    cp_group_size: int | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    sliding_window: int | None = None,
) -> configs.RpaConfigs:
    block = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=2,
        n_buffer=2,
    )
    model = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=num_kv_heads,
        head_dim=16,
        mask_value=-1e9,
        sliding_window=sliding_window,
    )
    serve = configs.ServingConfigs(
        num_seqs=2,
        page_size=page_size,
        total_q_tokens=4,
        num_page_indices=8,
        dtype_q=jnp.bfloat16.dtype,
        dtype_kv=jnp.dtype(dtype_kv),
        dtype_out=jnp.bfloat16.dtype,
        per_token_scale=per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        scale_k=scale_k,
        scale_v=scale_v,
        kv_layout=kv_layout,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
    )
    return configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        mode=mode,
        vmem_limit_bytes=16 * 1024 * 1024,
    )


def test_configs_enums_properties_and_validation():
    assert configs.KVLayout.SEQ_ALONG_LANE.symbol == "snh"
    assert configs.KVLayout.HEAD_ALONG_SUBLANE.symbol == "nhs"

    dist = jnp.array([2, 5, 8], dtype=jnp.int32)
    assert configs.RpaCase.DECODE.symbol == "d"
    assert configs.RpaCase.PREFILL.symbol == "p"
    assert configs.RpaCase.MIXED.symbol == "m"
    assert configs.RpaCase.DECODE.get_range(dist) == (0, dist[0])
    assert configs.RpaCase.PREFILL.get_range(dist) == (dist[0], dist[1])
    assert configs.RpaCase.MIXED.get_range(dist) == (dist[1], dist[2])

    # HEAD_ALONG_SUBLANE + DECODE (bkv_stride with bank conflict -> 3)
    cfg_nhs_dec = _make_rpa_cfg(
        mode=configs.RpaCase.DECODE,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        bq_sz=1,
        num_kv_heads=2,
    )
    assert cfg_nhs_dec.bq_sz == 1
    assert cfg_nhs_dec.bq_c_sz == 1
    assert cfg_nhs_dec.bkv_sz == 128
    assert cfg_nhs_dec.batch_size == 2
    assert cfg_nhs_dec.n_buffer == 2
    assert cfg_nhs_dec.max_steps_ub > 0
    assert cfg_nhs_dec.bkv_p == 1
    assert cfg_nhs_dec.bkv_p_cache == 1
    assert cfg_nhs_dec.bkv_p_new == 1
    assert cfg_nhs_dec.bkv_stride == 3
    assert cfg_nhs_dec.aligned_q_head_dim == 16
    assert cfg_nhs_dec.compute_kv_head_dim == 16
    assert cfg_nhs_dec.aligned_kv_head_dim == 16
    assert cfg_nhs_dec.aligned_num_kv_heads_x2 == 4
    assert cfg_nhs_dec.aligned_num_q_heads_per_kv_head == 4
    assert cfg_nhs_dec.kv_hbm_stride == 2
    assert cfg_nhs_dec.fuse_accum is True
    assert cfg_nhs_dec.kv_bytes_per_token > 0
    assert cfg_nhs_dec.q_bytes_per_token > 0
    assert cfg_nhs_dec.o_bytes_per_token > 0
    assert cfg_nhs_dec.q_vmem_shape == (2, 2, 1, 2, 2, 16)
    assert cfg_nhs_dec.kv_vmem_shape == (2, 128, 3, 2, 16)
    assert cfg_nhs_dec.dma_kv_new_size == 4
    assert cfg_nhs_dec.lm_scratch_shape == (2, 4, 8)
    assert cfg_nhs_dec.lse_vmem_shape == (2, 2, 1, 2, 2, 8)
    assert cfg_nhs_dec.acc_scratch_shape == (2, 4, 16)
    assert cfg_nhs_dec.max_schedule_size_multiplier == 16
    assert cfg_nhs_dec.serve.page_size_log2 == 7
    assert cfg_nhs_dec.serve.page_size_mask == 127
    assert cfg_nhs_dec.serve.scale_channels == 0

    # HEAD_ALONG_SUBLANE + PREFILL + no bank conflict + cp_group_size=2
    cfg_nhs_pref = _make_rpa_cfg(
        mode=configs.RpaCase.PREFILL,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        bq_sz=4,
        num_kv_heads=1,
        cp_group_size=2,
        attention_scope=configs.AttentionScope.CACHE_ONLY,
    )
    assert cfg_nhs_pref.bkv_p_cache == 0
    assert cfg_nhs_pref.bkv_p_new == 1
    assert cfg_nhs_pref.bkv_stride == 1
    assert cfg_nhs_pref.dma_kv_new_size == 5
    assert cfg_nhs_pref.fuse_accum is False

    # SEQ_ALONG_LANE + MIXED + uint8 + per_token_scale
    cfg_snh_u8 = _make_rpa_cfg(
        mode=configs.RpaCase.MIXED,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        bq_sz=4,
        dtype_kv=jnp.uint8,
        per_token_scale=True,
        per_token_scale_dtype=jnp.float32.dtype,
    )
    assert cfg_snh_u8.bkv_p_new == 2
    assert cfg_snh_u8.compute_kv_head_dim == 16
    assert cfg_snh_u8.aligned_kv_head_dim == 16
    assert cfg_snh_u8.aligned_num_kv_heads_x2 == 4
    assert cfg_snh_u8.kv_hbm_stride == 4
    assert cfg_snh_u8.kv_bytes_per_token > 0
    assert cfg_snh_u8.kv_vmem_shape == (2, 4, 4, 4, 384)
    assert cfg_snh_u8.dma_kv_new_size == 5

    # SEQ_ALONG_LANE + bf16 (non-uint8)
    cfg_snh_bf16 = _make_rpa_cfg(
        mode=configs.RpaCase.MIXED,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        bq_sz=4,
        dtype_kv=jnp.bfloat16,
    )
    assert cfg_snh_bf16.aligned_kv_head_dim == 16

    # validate_inputs checks
    q = jnp.zeros((4, 8, 16), dtype=jnp.bfloat16)
    k = jnp.zeros((4, 2, 16), dtype=jnp.bfloat16)
    v = jnp.zeros((4, 2, 16), dtype=jnp.bfloat16)
    kv_cache_nhs = jnp.zeros((8, 128, 2, 2, 16), dtype=jnp.bfloat16)
    kv_lens = jnp.array([10, 20], dtype=jnp.int32)
    page_indices = jnp.arange(8, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    # Valid HEAD_ALONG_SUBLANE
    cfg_nhs_dec.validate_inputs(
        q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
    )

    # Valid HEAD_ALONG_SUBLANE with CP
    kv_cache_nhs_cp = jnp.zeros((8, 128, 1, 2, 16), dtype=jnp.bfloat16)
    cfg_nhs_pref.validate_inputs(
        q,
        k[:, :1],
        v[:, :1],
        kv_cache_nhs_cp,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    # Valid SEQ_ALONG_LANE with uint8 + per_token_scale
    k_u8 = jnp.zeros((4, 2, 8), dtype=jnp.uint8)
    v_u8 = jnp.zeros((4, 2, 8), dtype=jnp.uint8)
    kv_cache_snh_u8 = jnp.zeros((8, 4, 4, 4, 128), dtype=jnp.uint8)
    cfg_snh_u8.validate_inputs(
        q, k_u8, v_u8, kv_cache_snh_u8, kv_lens, page_indices, cu_q_lens, distribution
    )

    # Error branches in validate_inputs:
    with pytest.raises(ValueError, match="Expected 3D array"):
        cfg_nhs_dec.validate_inputs(
            q[0], k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
        )
    with pytest.raises(ValueError, match="to be equal to"):
        cfg_nhs_dec.validate_inputs(
            q,
            k,
            v[:, :1],
            kv_cache_nhs,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="number of sequences"):
        cfg_nhs_dec.validate_inputs(
            q[:2], k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
        )
    with pytest.raises(ValueError, match="number of head dimensions"):
        cfg_nhs_dec.validate_inputs(
            q,
            k[:, :, :8],
            v[:, :, :8],
            kv_cache_nhs,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    cfg_snh_bad_page = _make_rpa_cfg(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE, page_size=64, bkv_sz=64
    )
    with pytest.raises(ValueError, match="multiple of 128"):
        cfg_snh_bad_page.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
        )
    with pytest.raises(ValueError, match="expected_kv_cache_shape"):
        cfg_nhs_dec.validate_inputs(
            q,
            k,
            v,
            kv_cache_nhs[:4, :64],
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    cfg_pts_bad_layout = _make_rpa_cfg(
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        per_token_scale=True,
        per_token_scale_dtype=jnp.float32.dtype,
    )
    with pytest.raises(ValueError, match="kv_layout=KVLayout.SEQ_ALONG_LANE"):
        cfg_pts_bad_layout.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
        )
    cfg_pts_bad_scale = _make_rpa_cfg(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        dtype_kv=jnp.uint8,
        per_token_scale=True,
        per_token_scale_dtype=jnp.float32.dtype,
        scale_k=1,
    )
    with pytest.raises(ValueError, match="scale_k and scale_v to be None"):
        cfg_pts_bad_scale.validate_inputs(
            q,
            k_u8,
            v_u8,
            kv_cache_snh_u8,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected int32 dtype"):
        cfg_nhs_dec.validate_inputs(
            q,
            k,
            v,
            kv_cache_nhs,
            kv_lens.astype(jnp.float32),
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="Expected 1D array"):
        cfg_nhs_dec.validate_inputs(
            q,
            k,
            v,
            kv_cache_nhs,
            kv_lens[:, None],
            page_indices,
            cu_q_lens,
            distribution,
        )
    with pytest.raises(ValueError, match="divisible by"):
        cfg_nhs_dec.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices[:3], cu_q_lens, distribution
        )
    with pytest.raises(ValueError, match="cu_q_lens.shape"):
        cfg_nhs_dec.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens[:2], distribution
        )
    with pytest.raises(ValueError, match="distribution.shape"):
        cfg_nhs_dec.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution[:2]
        )
    cfg_cp_full = _make_rpa_cfg(
        cp_group_size=2, attention_scope=configs.AttentionScope.FULL
    )
    with pytest.raises(ValueError, match="does not support AttentionScope.FULL"):
        cfg_cp_full.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
        )
    cfg_cp_sw = _make_rpa_cfg(
        cp_group_size=2,
        attention_scope=configs.AttentionScope.CACHE_ONLY,
        sliding_window=64,
    )
    with pytest.raises(ValueError, match="does not support sliding window"):
        cfg_cp_sw.validate_inputs(
            q, k, v, kv_cache_nhs, kv_lens, page_indices, cu_q_lens, distribution
        )
    cfg_snh_unaligned = _make_rpa_cfg(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        page_size=128,
        bkv_sz=64,
    )
    kv_cache_snh_bf16 = jnp.zeros((8, 4, 8, 2, 128), dtype=jnp.bfloat16)
    with pytest.raises(NotImplementedError, match="SEQ_ALONG_LANE expects"):
        cfg_snh_unaligned.validate_inputs(
            q, k, v, kv_cache_snh_bf16, kv_lens, page_indices, cu_q_lens, distribution
        )


# ---------------------------------------------------------------------------
# 3. stitch_utils.py tests
# ---------------------------------------------------------------------------


def test_stitch_utils_decode_and_prefill(monkeypatch):
    monkeypatch.setattr(
        pltpu, "roll", lambda x, shift, axis: jnp.roll(x, shift, axis=axis)
    )

    # 1) Decode path with bq_sz == 1 (int32 inputs)
    cfg_dec1 = _make_rpa_cfg(
        mode=configs.RpaCase.DECODE,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        bq_sz=1,
        bkv_sz=16,
        page_size=8,
    )
    v_len = cfg_dec1.bkv_sz + 2 * cfg_dec1.serve.page_size
    total_words = (
        cfg_dec1.model.num_kv_heads
        * 2
        * cfg_dec1.aligned_kv_head_dim
        // cfg_dec1.serve.packing_kv
    )
    vmem_ref1 = _FakeRef(
        jnp.arange(2 * total_words * v_len, dtype=jnp.uint32).reshape(
            2,
            cfg_dec1.model.num_kv_heads * 2,
            cfg_dec1.aligned_kv_head_dim // cfg_dec1.serve.packing_kv,
            1,
            v_len,
        )
    )
    res1 = stitch_utils.stitch_new_kv_lane(
        vmem_ref1,
        0,
        jnp.array(3, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
        cfgs=cfg_dec1,
    )
    stitch_utils.store_new_kv_lane(vmem_ref1, 0, res1, cfgs=cfg_dec1)

    # 2) Decode path with 1 < bq_sz <= num_lanes (bq_sz=4 <= 8, non-int32 inputs)
    cfg_dec4 = _make_rpa_cfg(
        mode=configs.RpaCase.MIXED,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        bq_sz=4,
        bkv_sz=16,
        page_size=8,
    )
    vmem_ref4 = _FakeRef(vmem_ref1.arr.copy())
    res4 = stitch_utils.stitch_new_kv_lane(
        vmem_ref4,
        0,
        jnp.array(5, dtype=jnp.int16),
        jnp.array(2, dtype=jnp.int16),
        jnp.array(3, dtype=jnp.int16),
        cfgs=cfg_dec4,
    )
    stitch_utils.store_new_kv_lane(vmem_ref4, 0, res4, cfgs=cfg_dec4)

    # 3) Prefill path with bq_sz > num_lanes (bq_sz=16 > 8)
    cfg_pref16 = _make_rpa_cfg(
        mode=configs.RpaCase.PREFILL,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        bq_sz=16,
        bkv_sz=16,
        page_size=8,
    )
    vmem_ref16 = _FakeRef(vmem_ref1.arr.copy())
    res16 = stitch_utils.stitch_new_kv_lane(
        vmem_ref16,
        0,
        jnp.array(4, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(8, dtype=jnp.int32),
        cfgs=cfg_pref16,
    )
    stitch_utils.store_new_kv_lane(vmem_ref16, 0, res16, cfgs=cfg_pref16)


# ---------------------------------------------------------------------------
# 4. bref_override.py tests
# ---------------------------------------------------------------------------


def test_bref_override_all_buffered_refs(monkeypatch):
    copies = []

    class _FakeAsyncCopy:
        def __init__(self, src, dst, sem):
            self.src = src
            self.dst = dst
            self.sem = sem

        def start(self):
            copies.append(("start", self.src, self.dst))

        def wait(self):
            copies.append(("wait", self.src, self.dst))

    monkeypatch.setattr(pltpu, "make_async_copy", _FakeAsyncCopy)
    monkeypatch.setattr(
        pltpu.BufferedRef, "current_copy_in_slot", property(lambda _s: 0)
    )
    monkeypatch.setattr(
        pltpu.BufferedRef, "current_wait_in_slot", property(lambda _s: 0)
    )
    monkeypatch.setattr(
        pltpu.BufferedRef, "current_copy_out_slot", property(lambda _s: 0)
    )
    monkeypatch.setattr(
        pltpu.BufferedRef, "current_wait_out_slot", property(lambda _s: 0)
    )

    def _bind(bref, win_shape):
        object.__setattr__(bref, "window_ref", _FakeRef(jnp.zeros((2, *win_shape))))
        object.__setattr__(bref, "sem_recvs", _FakeRef(jnp.zeros((2,))))
        object.__setattr__(bref, "sem_sends", _FakeRef(jnp.zeros((2,))))
        return bref

    class _FakeDmaEntry:
        def __init__(self):
            self.fetch_hbm = _FakeRef(jnp.array(0, dtype=jnp.int32))
            self.fetch_vmem = _FakeRef(jnp.array(0, dtype=jnp.int32))
            self.fetch_val = jnp.array(1, dtype=jnp.int32)
            self.wb_hbm = _FakeRef(jnp.array(0, dtype=jnp.int32))
            self.wb_vmem = _FakeRef(jnp.array(0, dtype=jnp.int32))
            self.wb_val = jnp.array(1, dtype=jnp.int32)

    class _FakeDmaTable:
        def __getitem__(self, _idx):
            return _FakeDmaEntry()

    sched = SimpleNamespace(
        get_dma_kv_cache=lambda _blk, _b, _i: (0, 0, 1),
        dma_kv_new=_FakeDmaTable(),
        do_writeback=jnp.ones((2, 2), dtype=jnp.int32),
        is_last_k=jnp.ones((2, 2), dtype=jnp.int32),
        get_dma_q=lambda _blk, _b: (0, 2),
        total_wait_kv_in=jnp.array([4, 4], dtype=jnp.int32),
        total_wait_kv_out=jnp.array([4, 4], dtype=jnp.int32),
        total_wait_o_out=jnp.array([4, 4], dtype=jnp.int32),
        total_wait_lse_out=jnp.array([4, 4], dtype=jnp.int32),
        total_wait_q_in=jnp.array([4, 4], dtype=jnp.int32),
    )
    page_indices_ref = _FakeRef(jnp.arange(8, dtype=jnp.int32))

    def _spec_for(arr: jax.Array) -> pl.BlockSpec:
        return pl.BlockSpec(
            block_shape=arr.shape,
            index_map=lambda i: (0,) * arr.ndim,
            memory_space=pltpu.VMEM,
        )

    # 1. KVBufferedRefSeqAlongLane
    cfg_snh = _make_rpa_cfg(
        mode=configs.RpaCase.MIXED,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        bq_sz=2,
        bkv_sz=8,
        page_size=8,
    )
    dummy_arr = jnp.zeros((2, 4, 8, 2, 24), dtype=jnp.bfloat16)
    kv_snh_ref = _bind(
        bref_override.KVBufferedRefSeqAlongLane.create(
            spec=_spec_for(dummy_arr),
            dtype_or_type=dummy_arr,
            buffer_type=pltpu.BufferType.INPUT_OUTPUT,
            buffer_count=3,
            use_lookahead=False,
            cfgs=cfg_snh,
        ),
        (2, 4, 8, 2, 24),
    )
    kv_cache_hbm_snh = _FakeRef(jnp.zeros((8, 4, 8, 2, 128), dtype=jnp.bfloat16))
    new_kv_hbm_snh = _FakeRef(jnp.zeros((4, 8, 2, 128), dtype=jnp.bfloat16))
    src_tuple_snh = (kv_cache_hbm_snh, new_kv_hbm_snh, sched, page_indices_ref)
    kv_snh_ref.copy_in(src_tuple_snh, (0,))
    kv_snh_ref.copy_out(src_tuple_snh, (0,))
    kv_snh_ref.wait_in(src_tuple_snh, (0,))
    kv_snh_ref.wait_out(src_tuple_snh, (0,))

    # 2. KVBufferedRefHeadAlongSublane
    cfg_nhs = _make_rpa_cfg(
        mode=configs.RpaCase.MIXED,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        bq_sz=2,
        bkv_sz=8,
        page_size=8,
    )
    kv_nhs_ref = _bind(
        bref_override.KVBufferedRefHeadAlongSublane.create(
            spec=_spec_for(dummy_arr),
            dtype_or_type=dummy_arr,
            buffer_type=pltpu.BufferType.INPUT_OUTPUT,
            buffer_count=3,
            use_lookahead=False,
            cfgs=cfg_nhs,
        ),
        (2, 8, 3, 2, 16),
    )
    kv_cache_hbm_nhs = _FakeRef(jnp.zeros((8, 8, 2, 2, 16), dtype=jnp.bfloat16))
    new_kv_hbm_nhs = _FakeRef(jnp.zeros((64, 2, 16), dtype=jnp.bfloat16))
    src_tuple_nhs = (kv_cache_hbm_nhs, new_kv_hbm_nhs, sched, page_indices_ref)
    kv_nhs_ref.copy_in(src_tuple_nhs, (0,))
    kv_nhs_ref.copy_out(src_tuple_nhs, (0,))
    kv_nhs_ref.wait_in(src_tuple_nhs, (0,))
    kv_nhs_ref.wait_out(src_tuple_nhs, (0,))

    # 3. BatchingORef
    o_dummy = jnp.zeros((2, 2, 8, 16), dtype=jnp.bfloat16)
    o_ref = _bind(
        bref_override.BatchingORef.create(
            spec=_spec_for(o_dummy),
            dtype_or_type=o_dummy,
            buffer_type=pltpu.BufferType.OUTPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfg_nhs,
        ),
        (2, 2, 8, 16),
    )
    o_hbm = _FakeRef(jnp.zeros((2, 16, 128), dtype=jnp.bfloat16))
    o_ref.copy_out((o_hbm, sched), (0,))
    o_ref.wait_out((o_hbm, sched), (0,))

    # 4. BatchingLSERef
    lse_dummy = jnp.zeros((2, 2, 8, 8), dtype=jnp.float32)
    lse_ref = _bind(
        bref_override.BatchingLSERef.create(
            spec=_spec_for(lse_dummy),
            dtype_or_type=lse_dummy,
            buffer_type=pltpu.BufferType.OUTPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfg_nhs,
        ),
        (2, 2, 8, 8),
    )
    lse_hbm = _FakeRef(jnp.zeros((2, 16, 128), dtype=jnp.float32))
    lse_ref.copy_out((lse_hbm, sched), (0,))
    lse_ref.wait_out((lse_hbm, sched), (0,))

    # 5. BatchingQRef
    q_dummy = jnp.zeros((2, 2, 8, 16), dtype=jnp.bfloat16)
    q_ref = _bind(
        bref_override.BatchingQRef.create(
            spec=_spec_for(q_dummy),
            dtype_or_type=q_dummy,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfg_nhs,
        ),
        (2, 2, 8, 16),
    )
    q_hbm = _FakeRef(jnp.zeros((2, 16, 128), dtype=jnp.bfloat16))
    q_ref.copy_in((q_hbm, sched), (0,))
    q_ref.wait_in((q_hbm, sched), (0,))
    assert len(copies) > 0
