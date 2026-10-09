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
"""Comprehensive CPU unit tests and 100% coverage suite for GDN v3 TD8.45 modules."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.gdn.v3 import (
    compute_gdn,
    compute_kda,
    config,
    memory_ref,
    vmem_ldst,
    wrapper,
)

pytestmark = pytest.mark.cpu_test


class _FakeAt:
    def __init__(self, ref: "_FakeRef"):
        self._ref = ref

    def __getitem__(self, idx) -> "_FakeRef":
        norm = self._ref._norm_idx(idx)
        return _FakeRef(self._ref.arr[norm])


class _FakeRef:
    """Lightweight in-memory stand-in for a Pallas/VMEM/HBM Ref on CPU."""

    def __init__(self, arr: jax.Array):
        self.arr = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.arr.dtype

    @property
    def size(self) -> int:
        return self.arr.size

    def reshape(self, *shape) -> "_FakeRef":
        return _FakeRef(self.arr.reshape(*shape))

    def _norm_idx(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._norm_idx(i) for i in idx)
        if isinstance(idx, pl.Slice):
            return slice(int(idx.start), int(idx.start) + int(idx.size))
        return idx

    @property
    def at(self) -> _FakeAt:
        return _FakeAt(self)

    def __getitem__(self, idx) -> jax.Array:
        if idx is Ellipsis:
            return self.arr
        norm = self._norm_idx(idx)
        return self.arr[norm]

    def __setitem__(self, idx, val):
        if isinstance(val, _FakeRef):
            val = val.arr
        if idx is Ellipsis:
            self.arr = jnp.asarray(val, dtype=self.arr.dtype)
        else:
            norm = self._norm_idx(idx)
            self.arr = self.arr.at[norm].set(val)


@pytest.fixture(autouse=True)
def _mock_tpu_info(monkeypatch):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=4,
            num_sublanes=8,
            vmem_capacity_bytes=64 * 1024 * 1024,
        ),
    )


def _make_dtypes(compute=jnp.float32) -> config.Dtypes:
    return config.Dtypes(
        act_in=jnp.float32.dtype,
        act_out=jnp.bfloat16.dtype,
        compute=jnp.dtype(compute),
        recurrent_state=jnp.float32.dtype,
        conv_state=jnp.float32.dtype,
    )


def _make_cfg(
    *,
    mode: config.GDNMode = config.GDNMode.PER_SEQ,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    tile_size: int = 4,
    window_size: int = 1,
    num_kq_heads: int = 2,
    num_v_heads: int = 4,
    kq_head_dim: int = 4,
    v_head_dim: int = 4,
    kernel_size: int = 3,
    gate_lower_bound: float | None = None,
    state_plan: config.StateSourcePlan | None = None,
) -> config.GDNConfig:
    dim_size = 2 * num_kq_heads * kq_head_dim + num_v_heads * v_head_dim
    return config.GDNConfig(
        mode=mode,
        dtypes=_make_dtypes(),
        batch_size=8,
        dim_size=dim_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        num_kq_heads=num_kq_heads,
        num_v_heads=num_v_heads,
        kq_head_dim=kq_head_dim,
        v_head_dim=v_head_dim,
        attention_mode=attention_mode,
        gate_lower_bound=gate_lower_bound,
        state_plan=state_plan,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# 1. compute_gdn.py tests
# ---------------------------------------------------------------------------


def test_compute_gdn_helpers_and_mask_dtypes():
    x = jnp.array([[3.0, 4.0]], dtype=jnp.float32)
    normed = compute_gdn.l2_norm(x, eps=0.0)
    np.testing.assert_allclose(normed, jnp.array([[0.6, 0.8]]), atol=1e-5)

    assert compute_gdn.get_mask_dtype(jnp.float32) == jnp.int32
    assert compute_gdn.get_mask_dtype(jnp.int32) == jnp.int32
    assert compute_gdn.get_mask_dtype(jnp.bfloat16) == jnp.int16
    assert compute_gdn.get_mask_dtype(jnp.float16) == jnp.int16
    assert compute_gdn.get_mask_dtype(jnp.int16) == jnp.int16
    # Fallback match-case for other 4-byte / 2-byte dtypes
    assert compute_gdn.get_mask_dtype(jnp.uint32) == jnp.int32
    assert compute_gdn.get_mask_dtype(jnp.uint16) == jnp.int16
    with pytest.raises(ValueError, match="Unsupported dtype"):
        compute_gdn.get_mask_dtype(jnp.int8)

    # invert_triangular_matrix with multiple blocks (chunk=4, block_size=2)
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4))
    t = eye.at[:, 1, 0].set(0.5).at[:, 2, 1].set(-0.25).at[:, 3, 0].set(0.1)
    t_inv = compute_gdn.invert_triangular_matrix(t, block_size=2)
    prod = jnp.matmul(t, t_inv)
    np.testing.assert_allclose(prod, eye, atol=1e-5)

    # fused_transpose_broadcast
    src = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3)
    trans = compute_gdn.fused_transpose_broadcast(src, src_dim=2, dst_dim=0)
    assert trans.shape == (3, 2, 1)


def test_chunked_and_recurrent_gdn():
    # 1. chunked_gdn with chunk_size=4
    cfg_chunk = _make_cfg(mode=config.GDNMode.PER_SEQ, tile_size=4, window_size=1)
    seq_tile = cfg_chunk.seq_tile_size
    chunk = cfg_chunk.chunk_size
    real_sizes = jnp.array([3], dtype=jnp.int32)
    q_large = jnp.ones(
        (seq_tile, cfg_chunk.num_kq_heads, chunk, cfg_chunk.kq_head_dim),
        dtype=jnp.float32,
    )
    k_large = jnp.ones_like(q_large) * 0.5
    v_large = jnp.ones(
        (seq_tile, cfg_chunk.num_v_heads, chunk, cfg_chunk.v_head_dim),
        dtype=jnp.float32,
    )
    b_large = jnp.zeros((seq_tile, 1, chunk, cfg_chunk.num_v_heads), dtype=jnp.float32)
    a_large = jnp.zeros_like(b_large)
    state_prev = jnp.zeros(
        (
            seq_tile,
            cfg_chunk.num_v_heads,
            cfg_chunk.kq_head_dim,
            cfg_chunk.v_head_dim,
        ),
        dtype=jnp.float32,
    )
    a_log = jnp.zeros((cfg_chunk.num_v_heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((cfg_chunk.num_v_heads,), dtype=jnp.float32)

    out, state = compute_gdn.chunked_gdn(
        real_sizes=real_sizes,
        q_large=q_large,
        k_large=k_large,
        v_large=v_large,
        b_large=b_large,
        a_large=a_large,
        state_prev=state_prev,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg_chunk,
    )
    assert out.shape == (
        seq_tile,
        chunk,
        cfg_chunk.num_v_heads,
        cfg_chunk.v_head_dim,
    )
    assert state.shape == (
        seq_tile,
        1,
        cfg_chunk.num_v_heads,
        cfg_chunk.kq_head_dim,
        cfg_chunk.v_head_dim,
    )

    # Also test chunked_gdn_per_seq with chunk_size=1 to cover empty range(1, 1)
    cfg_c1 = _make_cfg(mode=config.GDNMode.BATCHED, tile_size=1, window_size=1)
    out_c1, state_c1 = compute_gdn.chunked_gdn_per_seq(
        q_large[0, :, :1],
        k_large[0, :, :1],
        v_large[0, :, :1],
        jnp.zeros((1, 1, cfg_c1.num_v_heads), dtype=jnp.float32),
        jnp.zeros((1, 1, cfg_c1.num_v_heads), dtype=jnp.float32),
        state_prev[0],
        cfg_c1,
    )
    assert out_c1.shape == (cfg_c1.num_v_heads, 1, cfg_c1.v_head_dim)
    assert state_c1.shape == (
        cfg_c1.num_v_heads,
        cfg_c1.kq_head_dim,
        cfg_c1.v_head_dim,
    )

    # 2. recurrent_gdn with chunk_size=3, window_size=2 (covers both c_idx < 1 and >= 1)
    cfg_rec = _make_cfg(mode=config.GDNMode.PER_SEQ, tile_size=3, window_size=2)
    q_comp = jnp.ones(
        (1, cfg_rec.num_kq_heads, 3, 1, cfg_rec.kq_head_dim), dtype=jnp.float32
    )
    k_comp = jnp.ones_like(q_comp)
    v_comp = jnp.ones(
        (1, cfg_rec.num_v_heads, 3, 1, cfg_rec.v_head_dim), dtype=jnp.float32
    )
    b_comp = jnp.zeros((1, 1, 3, 1, cfg_rec.num_v_heads), dtype=jnp.float32)
    a_comp = jnp.zeros_like(b_comp)
    out_r, state_r = compute_gdn.recurrent_gdn(
        real_sizes=jnp.array([2], dtype=jnp.int32),
        q_compact=q_comp,
        k_compact=k_comp,
        v_compact=v_comp,
        b_compact=b_comp,
        a_compact=a_comp,
        state_prev=state_prev,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg_rec,
    )
    assert out_r.shape == (1, 3, cfg_rec.num_v_heads, cfg_rec.v_head_dim)
    assert state_r.shape == (
        1,
        2,
        cfg_rec.num_v_heads,
        cfg_rec.kq_head_dim,
        cfg_rec.v_head_dim,
    )


# ---------------------------------------------------------------------------
# 2. compute_kda.py tests
# ---------------------------------------------------------------------------


def test_chunked_and_recurrent_kda(monkeypatch):
    # Force triangular_block_size = 2 on a 4-token chunk so num_blocks = 2
    monkeypatch.setattr(
        config.GDNConfig, "triangular_block_size", property(lambda _self: 2)
    )
    cfg_kda = _make_cfg(
        mode=config.GDNMode.PER_SEQ,
        attention_mode=config.AttentionMode.KDA,
        tile_size=4,
        window_size=1,
        gate_lower_bound=None,
    )
    chunk = cfg_kda.chunk_size
    real_sizes = jnp.array([3], dtype=jnp.int32)
    q_large = jnp.ones(
        (1, cfg_kda.num_kq_heads, chunk, cfg_kda.kq_head_dim), dtype=jnp.float32
    )
    k_large = jnp.ones_like(q_large) * 0.5
    v_large = jnp.ones(
        (1, cfg_kda.num_v_heads, chunk, cfg_kda.v_head_dim), dtype=jnp.float32
    )
    b_large = jnp.zeros((1, 1, chunk, cfg_kda.num_v_heads), dtype=jnp.float32)
    a_large = jnp.zeros(
        (1, cfg_kda.num_v_heads, chunk, cfg_kda.kq_head_dim), dtype=jnp.float32
    )
    state_prev = jnp.zeros(
        (1, cfg_kda.num_v_heads, cfg_kda.kq_head_dim, cfg_kda.v_head_dim),
        dtype=jnp.float32,
    )
    a_log = jnp.zeros((cfg_kda.num_v_heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((cfg_kda.num_v_heads, cfg_kda.kq_head_dim), dtype=jnp.float32)

    out, state = compute_kda.chunked_kda(
        real_sizes=real_sizes,
        q_large=q_large,
        k_large=k_large,
        v_large=v_large,
        b_large=b_large,
        a_large=a_large,
        state_prev=state_prev,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg_kda,
    )
    assert out.shape == (1, chunk, cfg_kda.num_v_heads, cfg_kda.v_head_dim)
    assert state.shape == (
        1,
        1,
        cfg_kda.num_v_heads,
        cfg_kda.kq_head_dim,
        cfg_kda.v_head_dim,
    )

    # Also test chunked_kda_per_seq with chunk_size=1 to cover empty range(1, 1)
    cfg_k1 = _make_cfg(
        mode=config.GDNMode.BATCHED,
        attention_mode=config.AttentionMode.KDA,
        tile_size=1,
        window_size=1,
        gate_lower_bound=-5.0,
    )
    out_k1, _ = compute_kda.chunked_kda_per_seq(
        q_large[0, :, :1],
        k_large[0, :, :1],
        v_large[0, :, :1],
        jnp.zeros((cfg_k1.num_v_heads, 1, cfg_k1.kq_head_dim), dtype=jnp.float32),
        jnp.zeros((1, 1, cfg_k1.num_v_heads), dtype=jnp.float32),
        state_prev[0],
        cfg_k1,
    )
    assert out_k1.shape == (cfg_k1.num_v_heads, 1, cfg_k1.v_head_dim)

    # recurrent_kda with window_size=1 (bounded gate) and window_size=2, tile_size=3
    for win, tile in ((1, 1), (2, 3)):
        cfg_rec = _make_cfg(
            mode=config.GDNMode.PER_SEQ,
            attention_mode=config.AttentionMode.KDA,
            tile_size=tile,
            window_size=win,
            gate_lower_bound=-5.0,
        )
        q_comp = jnp.ones(
            (1, cfg_rec.num_kq_heads, tile, 1, cfg_rec.kq_head_dim),
            dtype=jnp.float32,
        )
        k_comp = jnp.ones_like(q_comp)
        v_comp = jnp.ones(
            (1, cfg_rec.num_v_heads, tile, 1, cfg_rec.v_head_dim),
            dtype=jnp.float32,
        )
        b_comp = jnp.zeros((1, 1, tile, 1, cfg_rec.num_v_heads), dtype=jnp.float32)
        a_comp = jnp.zeros(
            (1, cfg_rec.num_v_heads, tile, 1, cfg_rec.kq_head_dim),
            dtype=jnp.float32,
        )
        out_r, state_r = compute_kda.recurrent_kda(
            real_sizes=jnp.array([tile], dtype=jnp.int32),
            q_compact=q_comp,
            k_compact=k_comp,
            v_compact=v_comp,
            b_compact=b_comp,
            a_compact=a_comp,
            state_prev=state_prev,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg_rec,
        )
        assert out_r.shape == (1, tile, cfg_rec.num_v_heads, cfg_rec.v_head_dim)
        assert state_r.shape == (
            1,
            win,
            cfg_rec.num_v_heads,
            cfg_rec.kq_head_dim,
            cfg_rec.v_head_dim,
        )


# ---------------------------------------------------------------------------
# 3. memory_ref.py tests
# ---------------------------------------------------------------------------


def test_memory_ref_wrappers_and_buffered_refs(monkeypatch):
    # SmemWrapper 2D fast path, 3D general path, and 0D empty loop
    w2d = memory_ref.SmemWrapper(jnp.arange(6), (2, 3))
    assert int(w2d[1, 2]) == 5
    w3d = memory_ref.SmemWrapper(jnp.arange(12), (2, 2, 3))
    assert int(w3d[1, 1, 2]) == 11
    w0d = memory_ref.SmemWrapper(jnp.arange(1), ())
    assert w0d._get_pos(()) == 0

    cfg = _make_cfg(mode=config.GDNMode.BATCHED, tile_size=2, window_size=1)
    zeros = jnp.zeros((2,), dtype=jnp.int32)
    ones = jnp.ones((2,), dtype=jnp.int32)
    m_no_ckpt = memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=jnp.array(1, dtype=jnp.int32),
        p_id_to_s_idx=zeros,
        p_id_to_r_base=zeros,
        p_id_to_r_size=ones,
        p_id_is_first_tile=ones,
        p_id_is_last_tile=ones,
        s_idx_has_initial_state=ones,
        s_idx_to_state_indices=zeros,
        s_idx_to_read_offset=zeros,
        s_idx_to_ckpt_indices=None,
    )
    assert len(m_no_ckpt) == 9

    m_with_ckpt = memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=jnp.array(1, dtype=jnp.int32),
        p_id_to_s_idx=zeros,
        p_id_to_r_base=zeros,
        p_id_to_r_size=ones,
        p_id_is_first_tile=ones,
        p_id_is_last_tile=ones,
        s_idx_has_initial_state=ones,
        s_idx_to_state_indices=zeros,
        s_idx_to_read_offset=zeros,
        s_idx_to_ckpt_indices=jnp.zeros((2, 2), dtype=jnp.int32),
    )
    assert len(m_with_ckpt) == 10

    # Mock pltpu.make_async_copy to record start/wait calls
    copies = []

    class _FakeCopy:
        def __init__(self, src, dst, sem):
            self.src = src
            self.dst = dst
            self.sem = sem

        def start(self):
            copies.append(("start", self.src, self.dst))

        def wait(self):
            copies.append(("wait", self.src, self.dst))

    monkeypatch.setattr(pltpu, "make_async_copy", _FakeCopy)

    # Create allocs (dense, whole_block_dma=True, whole_block_dma=False)
    qkv_ref = _FakeRef(jnp.zeros((8, 1, cfg.dim_size), dtype=jnp.float32))
    b_ref = _FakeRef(jnp.zeros((8, 1, cfg.aligned_num_v_heads), dtype=jnp.float32))
    a_ref = _FakeRef(jnp.zeros((8, 1, cfg.aligned_gate_dim), dtype=jnp.float32))
    out_ref = _FakeRef(
        jnp.zeros((8, cfg.num_v_heads, cfg.v_head_dim), dtype=jnp.bfloat16)
    )
    conv_state_ref = _FakeRef(
        jnp.zeros((4, cfg.prev_kernel_size, 1, cfg.dim_size), dtype=jnp.float32)
    )
    rec_state_ref = _FakeRef(
        jnp.zeros(
            (4, cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim), dtype=jnp.float32
        )
    )

    # Validation errors in create_allocs
    with pytest.raises(ValueError, match="must be set together"):
        memory_ref.create_allocs(
            m_no_ckpt,
            qkv_ref,
            b_ref,
            a_ref,
            out_ref,
            conv_state_ref,
            rec_state_ref,
            cfg,
            conv_state_output_ref=conv_state_ref,
            recurrent_state_output_ref=None,
        )
    with pytest.raises(ValueError, match="require an external state source plan"):
        memory_ref.create_allocs(
            m_no_ckpt,
            qkv_ref,
            b_ref,
            a_ref,
            out_ref,
            conv_state_ref,
            rec_state_ref,
            cfg,
            conv_state_output_ref=conv_state_ref,
            recurrent_state_output_ref=rec_state_ref,
        )

    region_conv = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=4,
    )
    region_rec = config.StateRegion(
        kb0=1,
        nblocks=1,
        row0=2,
        nrows=4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=4,
    )
    plan_partial = config.StateSourcePlan(
        stride=2, conv=region_conv, recurrent=region_rec, whole_block_dma=False
    )
    plan_whole = config.StateSourcePlan(
        stride=2, conv=region_conv, recurrent=region_rec, whole_block_dma=True
    )

    cfg_win2 = _make_cfg(
        mode=config.GDNMode.BATCHED,
        tile_size=2,
        window_size=2,
        state_plan=plan_partial,
    )
    with pytest.raises(ValueError, match="do not support checkpoint windows"):
        memory_ref.create_allocs(
            m_with_ckpt,
            qkv_ref,
            b_ref,
            a_ref,
            out_ref,
            conv_state_ref,
            rec_state_ref,
            cfg_win2,
            conv_state_output_ref=conv_state_ref,
            recurrent_state_output_ref=rec_state_ref,
        )

    # Dense create_allocs
    qkv_a, _b_a, _a_a, conv_a, _rec_a, out_a = memory_ref.create_allocs(
        m_no_ckpt,
        qkv_ref,
        b_ref,
        a_ref,
        out_ref,
        conv_state_ref,
        rec_state_ref,
        cfg,
    )

    # Populate window_ref / sem_recvs / sem_sends on the refs and run copy/wait
    def _bind_runtime_refs(buf_ref, win_shape):
        object.__setattr__(buf_ref, "window_ref", _FakeRef(jnp.zeros((2, *win_shape))))
        object.__setattr__(buf_ref, "sem_recvs", _FakeRef(jnp.zeros((2,))))
        object.__setattr__(buf_ref, "sem_sends", _FakeRef(jnp.zeros((2,))))
        object.__setattr__(buf_ref, "buffer_index", _FakeRef(jnp.array([0])))
        return buf_ref

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

    _bind_runtime_refs(qkv_a, (2, 1, 1, cfg.dim_size))
    qkv_a.copy_in(qkv_ref, (0,))
    qkv_a.wait_in(qkv_ref, (0,))

    _bind_runtime_refs(out_a, (2, 1, cfg.num_v_heads, cfg.v_head_dim))
    out_a.copy_out(out_ref, (0,))
    out_a.wait_out(out_ref, (0,))

    _bind_runtime_refs(conv_a, (2, 1, cfg.prev_kernel_size, 1, cfg.dim_size))
    conv_a.copy_in(conv_state_ref, (0,))
    conv_a.wait_in(conv_state_ref, (0,))
    conv_a.copy_out(conv_state_ref, (0,))
    conv_a.wait_out(conv_state_ref, (0,))

    # External create_allocs: whole_block_dma=True (with compact output refs)
    pool_ref = _FakeRef(jnp.zeros((16, 4, 8), dtype=jnp.float32))
    compact_out = _FakeRef(jnp.zeros((4, 2, 4, 8), dtype=jnp.float32))
    cfg_whole = _make_cfg(
        mode=config.GDNMode.BATCHED,
        tile_size=2,
        window_size=1,
        state_plan=plan_whole,
    )
    _, _, _, ext_conv_whole, ext_rec_whole, _ = memory_ref.create_allocs(
        m_no_ckpt,
        qkv_ref,
        b_ref,
        a_ref,
        out_ref,
        pool_ref,
        pool_ref,
        cfg_whole,
        conv_state_output_ref=compact_out,
        recurrent_state_output_ref=compact_out,
    )
    _bind_runtime_refs(ext_conv_whole, (2, 1, 1, 4, 8))
    ext_conv_whole.copy_in(pool_ref, (0,))
    ext_conv_whole.wait_in(pool_ref, (0,))
    ext_conv_whole.copy_out(pool_ref, (0,))
    ext_conv_whole.wait_out(pool_ref, (0,))

    # External create_allocs: whole_block_dma=False, with ckpt_indices
    pool_ref_large = _FakeRef(jnp.zeros((16, 8, 8), dtype=jnp.float32))
    cfg_part = _make_cfg(
        mode=config.GDNMode.BATCHED,
        tile_size=2,
        window_size=1,
        state_plan=plan_partial,
    )
    _, _, _, ext_conv_part, ext_rec_part, _ = memory_ref.create_allocs(
        m_with_ckpt,
        qkv_ref,
        b_ref,
        a_ref,
        out_ref,
        pool_ref_large,
        pool_ref_large,
        cfg_part,
    )
    _bind_runtime_refs(ext_conv_part, (2, 1, 1, 4, 8))
    _bind_runtime_refs(ext_rec_part, (2, 1, 1, 4, 8))
    # For pool_ref (nrows=4 == shape[1]=4), _region_slice hits 1D slice
    ext_conv_part.copy_in(pool_ref, (0,))
    # For ext_rec_part (row0=2, nrows=4 != shape[1]=8), _region_slice hits 2D slice
    ext_rec_part.copy_in(pool_ref_large, (0,))
    ext_rec_part.copy_out(pool_ref_large, (0,))


# ---------------------------------------------------------------------------
# 4. vmem_ldst.py tests
# ---------------------------------------------------------------------------


def test_vmem_ldst_all_functions(monkeypatch):
    assert vmem_ldst._invert_rows_perm((1, 0, 2)) == (1, 0, 2)

    # load_strided_heads & load_compact_heads & load_compact_to_large
    slot_ref = _FakeRef(jnp.arange(2 * 1 * 16, dtype=jnp.float32).reshape(2, 1, 16))
    strided = vmem_ldst.load_strided_heads(slot_ref, num_heads=2, head_dim=8)
    assert strided.shape == (2, 2, 8)

    comp_ref = _FakeRef(
        jnp.arange(2 * 2 * 1 * 16, dtype=jnp.float32).reshape(2, 2, 1, 16)
    )
    compact = vmem_ldst.load_compact_heads(comp_ref, num_heads=2, head_dim=8)
    assert compact.shape == (2, 2, 2, 1, 8)

    large = vmem_ldst.load_compact_to_large(comp_ref)
    assert large.shape == (2, 2, 16)

    # Mock typed_ldst.load_typed, typed_ldst.store_typed, and pltpu.bitcast
    monkeypatch.setattr(
        typed_ldst,
        "load_typed",
        lambda ref, view_dtype, lane_split: ref.arr.reshape(-1, 4).astype(view_dtype),
    )
    stored_chunks = []
    monkeypatch.setattr(
        typed_ldst,
        "store_typed",
        lambda ref, val, lane_split: stored_chunks.append((ref, val, lane_split)),
    )
    monkeypatch.setattr(
        pltpu,
        "bitcast",
        lambda arr, dtype: (
            jnp.repeat(arr.astype(jnp.bfloat16), 2, axis=-1)
            if dtype == jnp.bfloat16
            else arr[..., ::2].astype(jnp.uint32)
        ),
    )

    # load_state_region & store_state_region:
    # 1) nblocks=1, view_dtype=uint32 with bf16 bitcast, rows_used == capacity
    reg_u32 = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=2,
        view_dtype=jnp.dtype(jnp.uint32),
        lane_split=1,
        rows_used=2,
        rows_perm=None,
    )
    tile_u32 = _FakeRef(jnp.ones((1, 2, 4), dtype=jnp.uint32))
    loaded_u32 = vmem_ldst.load_state_region(tile_u32, reg_u32, (2, 8))
    assert loaded_u32.shape == (2, 8)
    vmem_ldst.store_state_region(tile_u32, reg_u32, jnp.ones((2, 8), dtype=jnp.float32))

    # 2) nblocks=2, view_dtype=float32, rows_perm=(1, 0), rows_used < capacity
    reg_perm = config.StateRegion(
        kb0=0,
        nblocks=2,
        row0=0,
        nrows=2,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=2,
        rows_perm=(1, 0),
    )
    tile_perm = _FakeRef(jnp.ones((2, 2, 4), dtype=jnp.float32))
    loaded_perm = vmem_ldst.load_state_region(tile_perm, reg_perm, (2, 4))
    assert loaded_perm.shape == (2, 4)
    vmem_ldst.store_state_region(
        tile_perm, reg_perm, jnp.ones((2, 4), dtype=jnp.float32)
    )

    # load_and_select_states:
    # Case A: state_plan is None, carry_conv_scratch_ref is None
    cfg_dense = _make_cfg(mode=config.GDNMode.BATCHED, tile_size=1, window_size=1)
    zeros = jnp.zeros((1,), dtype=jnp.int32)
    ones = jnp.ones((1,), dtype=jnp.int32)
    meta = memory_ref.MetadataRef.create(
        cfgs=cfg_dense,
        num_tiles=ones,
        p_id_to_s_idx=zeros,
        p_id_to_r_base=zeros,
        p_id_to_r_size=ones,
        p_id_is_first_tile=ones,
        p_id_is_last_tile=ones,
        s_idx_has_initial_state=ones,
        s_idx_to_state_indices=zeros,
        s_idx_to_read_offset=zeros,
    )
    conv_slot = _FakeRef(
        jnp.ones(
            (1, 1, cfg_dense.prev_kernel_size, 1, cfg_dense.dim_size),
            dtype=jnp.float32,
        )
    )
    rec_slot = _FakeRef(
        jnp.ones(
            (
                1,
                1,
                cfg_dense.num_v_heads,
                cfg_dense.kq_head_dim,
                cfg_dense.v_head_dim,
            ),
            dtype=jnp.float32,
        )
    )
    rs, pc, pr = vmem_ldst.load_and_select_states(
        meta, jnp.array(0), conv_slot, rec_slot, None, None, cfg_dense
    )
    assert rs.shape == (1,)

    # Case B: state_plan is None, carry_conv_scratch_ref is not None
    carry_c = _FakeRef(
        jnp.zeros(
            (1, cfg_dense.prev_kernel_size, 1, cfg_dense.dim_size), dtype=jnp.float32
        )
    )
    carry_r = _FakeRef(
        jnp.zeros(
            (
                1,
                cfg_dense.num_v_heads,
                cfg_dense.kq_head_dim,
                cfg_dense.v_head_dim,
            ),
            dtype=jnp.float32,
        )
    )
    vmem_ldst.load_and_select_states(
        meta, jnp.array(0), conv_slot, rec_slot, carry_c, carry_r, cfg_dense
    )

    # Case C: state_plan is not None, carry_conv_scratch_ref is not None
    monkeypatch.setattr(pl, "when", lambda _cond: lambda fn: fn())
    reg_c = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=cfg_dense.prev_kernel_size * cfg_dense.dim_size // 4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=cfg_dense.prev_kernel_size * cfg_dense.dim_size // 4,
    )
    reg_r = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=cfg_dense.num_v_heads * cfg_dense.kq_head_dim * cfg_dense.v_head_dim // 4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=cfg_dense.num_v_heads
        * cfg_dense.kq_head_dim
        * cfg_dense.v_head_dim
        // 4,
    )
    plan = config.StateSourcePlan(stride=1, conv=reg_c, recurrent=reg_r)
    cfg_pooled = _make_cfg(
        mode=config.GDNMode.BATCHED, tile_size=1, window_size=1, state_plan=plan
    )
    conv_pool_slot = _FakeRef(jnp.ones((1, 1, 1, reg_c.nrows, 4), dtype=jnp.float32))
    rec_pool_slot = _FakeRef(jnp.ones((1, 1, 1, reg_r.nrows, 4), dtype=jnp.float32))
    vmem_ldst.load_and_select_states(
        meta,
        jnp.array(0),
        conv_pool_slot,
        rec_pool_slot,
        carry_c,
        carry_r,
        cfg_pooled,
    )

    # load_activation_as_compact & load_activation_as_large for both GDN and KDA
    for att_mode in (config.AttentionMode.GDN, config.AttentionMode.KDA):
        cfg_act = _make_cfg(
            mode=config.GDNMode.BATCHED,
            attention_mode=att_mode,
            tile_size=1,
            window_size=1,
        )
        qkv_vreg = jnp.ones((1, 1, 1, cfg_act.dim_size), dtype=jnp.float32)
        qkv_vmem = _FakeRef(jnp.zeros_like(qkv_vreg))
        b_vmem = _FakeRef(
            jnp.zeros((1, 1, 1, cfg_act.aligned_num_v_heads), dtype=jnp.float32)
        )
        a_vmem = _FakeRef(
            jnp.zeros((1, 1, 1, cfg_act.aligned_gate_dim), dtype=jnp.float32)
        )
        vmem_ldst.load_activation_as_compact(
            qkv_vreg, qkv_vmem, b_vmem, a_vmem, cfg_act
        )
        vmem_ldst.load_activation_as_large(qkv_vreg, qkv_vmem, b_vmem, a_vmem, cfg_act)


# ---------------------------------------------------------------------------
# 5. wrapper.py tests
# ---------------------------------------------------------------------------


def test_wrapper_inner_and_outer_kernel(monkeypatch):
    monkeypatch.setattr(pl, "program_id", lambda _axis: jnp.array(0))
    monkeypatch.setattr(pl, "when", lambda _cond: lambda fn: fn())
    monkeypatch.setattr(
        typed_ldst,
        "load_typed",
        lambda ref, view_dtype, lane_split: ref.arr.reshape(-1, 4).astype(view_dtype),
    )
    monkeypatch.setattr(
        typed_ldst,
        "store_typed",
        lambda ref, val, lane_split: None,
    )

    # 1. Dense GDN inner_kernel: recurrent (padding_size > 0, conv_bias != None)
    cfg_gdn_rec = _make_cfg(
        mode=config.GDNMode.BATCHED,
        attention_mode=config.AttentionMode.GDN,
        tile_size=1,
        window_size=1,
        num_kq_heads=1,
        num_v_heads=2,  # aligned_num_v_heads=4 -> padding_size=2 > 0
    )
    zeros = jnp.zeros((1,), dtype=jnp.int32)
    ones = jnp.ones((1,), dtype=jnp.int32)
    meta = memory_ref.MetadataRef.create(
        cfgs=cfg_gdn_rec,
        num_tiles=ones,
        p_id_to_s_idx=zeros,
        p_id_to_r_base=zeros,
        p_id_to_r_size=ones,
        p_id_is_first_tile=ones,
        p_id_is_last_tile=ones,
        s_idx_has_initial_state=ones,
        s_idx_to_state_indices=zeros,
        s_idx_to_read_offset=zeros,
    )
    weights_with_bias = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(
            weight=_FakeRef(
                jnp.ones(
                    (cfg_gdn_rec.kernel_size, 1, cfg_gdn_rec.dim_size),
                    dtype=jnp.float32,
                )
            ),
            bias=_FakeRef(jnp.zeros((cfg_gdn_rec.dim_size,), dtype=jnp.float32)),
        ),
        gdn=memory_ref.GDNWeightsRef(
            a_log=_FakeRef(jnp.zeros((cfg_gdn_rec.num_v_heads,), dtype=jnp.float32)),
            dt_bias=_FakeRef(jnp.zeros((cfg_gdn_rec.num_v_heads,), dtype=jnp.float32)),
        ),
    )
    qkv_slot = _FakeRef(jnp.ones((1, 1, 1, cfg_gdn_rec.dim_size), dtype=jnp.float32))
    b_slot = _FakeRef(
        jnp.zeros((1, 1, 1, cfg_gdn_rec.aligned_num_v_heads), dtype=jnp.float32)
    )
    a_slot = _FakeRef(
        jnp.zeros((1, 1, 1, cfg_gdn_rec.aligned_gate_dim), dtype=jnp.float32)
    )
    conv_slot = _FakeRef(
        jnp.zeros(
            (1, 1, cfg_gdn_rec.prev_kernel_size, 1, cfg_gdn_rec.dim_size),
            dtype=jnp.float32,
        )
    )
    rec_slot = _FakeRef(
        jnp.zeros(
            (
                1,
                1,
                cfg_gdn_rec.num_v_heads,
                cfg_gdn_rec.kq_head_dim,
                cfg_gdn_rec.v_head_dim,
            ),
            dtype=jnp.float32,
        )
    )
    out_slot = _FakeRef(
        jnp.zeros(
            (1, 1, cfg_gdn_rec.num_v_heads, cfg_gdn_rec.v_head_dim),
            dtype=jnp.bfloat16,
        )
    )
    wrapper.inner_kernel(
        qkv_slot,
        b_slot,
        a_slot,
        conv_slot,
        rec_slot,
        out_slot,
        meta,
        weights_with_bias,
        None,
        None,
        cfg=cfg_gdn_rec,
        p_id=None,
    )

    # 2. Pooled GDN inner_kernel: chunked (padding_size == 0, conv_bias == None),
    # both with and without carry scratch refs
    reg_c = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=16,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=16,
    )
    reg_r = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=16,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=16,
    )
    plan = config.StateSourcePlan(stride=1, conv=reg_c, recurrent=reg_r)
    cfg_gdn_chunk = _make_cfg(
        mode=config.GDNMode.PER_SEQ,
        attention_mode=config.AttentionMode.GDN,
        tile_size=2,
        window_size=1,
        num_kq_heads=2,
        num_v_heads=4,  # aligned_num_v_heads=4 -> padding_size=0
        state_plan=plan,
    )
    weights_no_bias = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(
            weight=_FakeRef(
                jnp.ones(
                    (cfg_gdn_chunk.kernel_size, 1, cfg_gdn_chunk.dim_size),
                    dtype=jnp.float32,
                )
            ),
            bias=None,
        ),
        gdn=memory_ref.GDNWeightsRef(
            a_log=_FakeRef(jnp.zeros((cfg_gdn_chunk.num_v_heads,), dtype=jnp.float32)),
            dt_bias=_FakeRef(
                jnp.zeros((cfg_gdn_chunk.num_v_heads,), dtype=jnp.float32)
            ),
        ),
    )
    qkv_slot_c = _FakeRef(
        jnp.ones((1, 2, 1, cfg_gdn_chunk.dim_size), dtype=jnp.float32)
    )
    b_slot_c = _FakeRef(
        jnp.zeros((1, 2, 1, cfg_gdn_chunk.aligned_num_v_heads), dtype=jnp.float32)
    )
    a_slot_c = _FakeRef(
        jnp.zeros((1, 2, 1, cfg_gdn_chunk.aligned_gate_dim), dtype=jnp.float32)
    )
    conv_pool_slot = _FakeRef(jnp.zeros((1, 1, 1, 16, 4), dtype=jnp.float32))
    rec_pool_slot = _FakeRef(jnp.zeros((1, 1, 1, 16, 4), dtype=jnp.float32))
    out_slot_c = _FakeRef(
        jnp.zeros(
            (1, 2, cfg_gdn_chunk.num_v_heads, cfg_gdn_chunk.v_head_dim),
            dtype=jnp.bfloat16,
        )
    )
    carry_c = _FakeRef(
        jnp.zeros(
            (1, cfg_gdn_chunk.prev_kernel_size, 1, cfg_gdn_chunk.dim_size),
            dtype=jnp.float32,
        )
    )
    carry_r = _FakeRef(
        jnp.zeros(
            (
                1,
                cfg_gdn_chunk.num_v_heads,
                cfg_gdn_chunk.kq_head_dim,
                cfg_gdn_chunk.v_head_dim,
            ),
            dtype=jnp.float32,
        )
    )
    # Without carry scratch
    wrapper.inner_kernel(
        qkv_slot_c,
        b_slot_c,
        a_slot_c,
        conv_pool_slot,
        rec_pool_slot,
        out_slot_c,
        meta,
        weights_no_bias,
        None,
        None,
        cfg=cfg_gdn_chunk,
        p_id=jnp.array(0),
    )
    # With carry scratch
    wrapper.inner_kernel(
        qkv_slot_c,
        b_slot_c,
        a_slot_c,
        conv_pool_slot,
        rec_pool_slot,
        out_slot_c,
        meta,
        weights_no_bias,
        carry_c,
        carry_r,
        cfg=cfg_gdn_chunk,
        p_id=jnp.array(0),
    )

    # 3. KDA inner_kernel with window_size > 1 (covers lines 149 and 155)
    cfg_kda_win = _make_cfg(
        mode=config.GDNMode.BATCHED,
        attention_mode=config.AttentionMode.KDA,
        tile_size=1,
        window_size=2,
        num_kq_heads=2,
        num_v_heads=4,
    )
    weights_kda = memory_ref.WeightRefs(
        conv=weights_no_bias.conv,
        gdn=memory_ref.GDNWeightsRef(
            a_log=_FakeRef(jnp.zeros((cfg_kda_win.num_v_heads,), dtype=jnp.float32)),
            dt_bias=_FakeRef(
                jnp.zeros(
                    (cfg_kda_win.num_v_heads, cfg_kda_win.kq_head_dim),
                    dtype=jnp.float32,
                )
            ),
        ),
    )
    wrapper.inner_kernel(
        _FakeRef(jnp.ones((1, 2, 1, cfg_kda_win.dim_size), dtype=jnp.float32)),
        _FakeRef(
            jnp.zeros((1, 2, 1, cfg_kda_win.aligned_num_v_heads), dtype=jnp.float32)
        ),
        _FakeRef(jnp.zeros((1, 2, 1, cfg_kda_win.aligned_gate_dim), dtype=jnp.float32)),
        _FakeRef(
            jnp.zeros(
                (1, 2, cfg_kda_win.prev_kernel_size, 1, cfg_kda_win.dim_size),
                dtype=jnp.float32,
            )
        ),
        _FakeRef(
            jnp.zeros(
                (
                    1,
                    2,
                    cfg_kda_win.num_v_heads,
                    cfg_kda_win.kq_head_dim,
                    cfg_kda_win.v_head_dim,
                ),
                dtype=jnp.float32,
            )
        ),
        _FakeRef(
            jnp.zeros(
                (1, 2, cfg_kda_win.num_v_heads, cfg_kda_win.v_head_dim),
                dtype=jnp.bfloat16,
            )
        ),
        meta,
        weights_kda,
        None,
        None,
        cfg=cfg_kda_win,
        p_id=jnp.array(0),
    )

    # 4. outer_kernel (both dense and pooled)
    pipeline_calls = []
    monkeypatch.setattr(
        pltpu,
        "emit_pipeline",
        lambda **kw: lambda *args, **kwargs: pipeline_calls.append((kw, args, kwargs)),
    )
    monkeypatch.setattr(
        pl,
        "with_scoped",
        lambda allocations: lambda fn: lambda: fn(allocations),
    )
    object.__setattr__(meta, "num_tiles", _FakeRef(jnp.array(1, dtype=jnp.int32)))
    wrapper.outer_kernel(
        meta,
        qkv_slot,
        b_slot,
        a_slot,
        conv_slot,
        rec_slot,
        out_slot,
        weights_no_bias,
        out_slot,
        conv_slot,
        rec_slot,
        None,
        None,
        cfg=cfg_gdn_rec,
    )
    wrapper.outer_kernel(
        meta,
        qkv_slot_c,
        b_slot_c,
        a_slot_c,
        conv_pool_slot,
        rec_pool_slot,
        out_slot_c,
        weights_no_bias,
        out_slot_c,
        conv_pool_slot,
        rec_pool_slot,
        carry_c,
        carry_r,
        cfg=cfg_gdn_chunk,
    )
    assert len(pipeline_calls) == 2


def test_fused_conv1d_gdn_validation_and_dispatch(monkeypatch):
    # Mock pl.pallas_call so fused_conv1d_gdn.__wrapped__ runs end-to-end on CPU
    def _fake_pallas_call(kernel_fn, out_shape, **_kwargs):
        def _runner(*_args):
            out_s, conv_s, rec_s = out_shape
            out_arr = (
                jnp.zeros(out_s.shape, dtype=out_s.dtype)
                if isinstance(out_s, jax.ShapeDtypeStruct)
                else out_s
            )
            return out_arr, conv_s, rec_s

        return _runner

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call)

    fn = wrapper.fused_conv1d_gdn.__wrapped__
    n_kq, n_v, d_k, d_v, kernel_size = 2, 4, 4, 4, 3
    dim = 2 * n_kq * d_k + n_v * d_v
    batch_size = 4
    num_seqs = 2

    qkv = jnp.ones((batch_size, dim), dtype=jnp.bfloat16)
    b = jnp.zeros((batch_size, n_v), dtype=jnp.bfloat16)
    a_gdn = jnp.zeros((batch_size, n_v), dtype=jnp.bfloat16)
    a_kda = jnp.zeros((batch_size, n_v * d_k), dtype=jnp.bfloat16)
    conv_state = jnp.zeros((num_seqs + 1, kernel_size - 1, dim), dtype=jnp.bfloat16)
    rec_state = jnp.zeros((num_seqs + 1, n_v, d_k, d_v), dtype=jnp.float32)
    conv_weight = jnp.ones((dim, 1, kernel_size), dtype=jnp.bfloat16)
    conv_bias = jnp.zeros((dim,), dtype=jnp.bfloat16)
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias_gdn = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias_kda = jnp.zeros((n_v, d_k), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 2, 4], dtype=jnp.int32)
    state_indices = jnp.array([1, 2], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
    seq_lens = jnp.array([2, 2], dtype=jnp.int32)

    # 1. Validation errors
    with pytest.raises(ValueError, match="gate_lower_bound only applies to KDA"):
        fn(
            qkv,
            b,
            a_gdn,
            conv_state,
            rec_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias_gdn,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            attention_mode=config.AttentionMode.GDN,
            gate_lower_bound=-5.0,
        )

    with pytest.raises(ValueError, match="gate_lower_bound must be negative"):
        fn(
            qkv,
            b,
            a_kda,
            conv_state,
            rec_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias_kda,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            attention_mode=config.AttentionMode.KDA,
            gate_lower_bound=0.0,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        fn(
            qkv,
            b,
            a_gdn,
            conv_state,
            rec_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias_gdn,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            batched_only=True,
            prefill_only=True,
        )

    # 2. Dense GDN: full, batched_only, prefill_only (zero_init=False)
    (new_c, new_r), out = fn(
        qkv,
        b,
        a_gdn,
        conv_state,
        rec_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias_gdn,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )
    assert out.shape == (batch_size, n_v * d_v)
    assert new_c.shape == conv_state.shape
    assert new_r.shape == rec_state.shape

    # Dense GDN with num_spec_tokens=1, batched_only=True, conv_bias=None
    fn(
        qkv,
        b,
        a_gdn,
        conv_state,
        rec_state,
        conv_weight,
        None,
        a_log,
        dt_bias_gdn,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        read_offsets=jnp.zeros((num_seqs,), dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=1,
        batched_only=True,
    )

    # Dense KDA with num_spec_tokens=1, prefill_only=True, zero_initialize_out=False
    fn(
        qkv,
        b,
        a_kda,
        conv_state,
        rec_state,
        conv_weight,
        None,
        a_log,
        dt_bias_kda,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        read_offsets=jnp.zeros((num_seqs,), dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        attention_mode=config.AttentionMode.KDA,
        gate_lower_bound=-5.0,
        num_spec_tokens=1,
        prefill_only=True,
        zero_initialize_out=False,
    )

    # 3. Pooled mode: full, batched_only, prefill_only, with ckpt_indices
    reg = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=4,
    )
    plan = config.StateSourcePlan(stride=1, conv=reg, recurrent=reg)
    pool_source = jnp.zeros((8, 4, 4), dtype=jnp.float32)
    ckpt_indices = jnp.zeros((num_seqs, 2), dtype=jnp.int32)

    out_src, out_act = fn(
        qkv,
        b,
        a_gdn,
        None,
        None,
        conv_weight,
        None,
        a_log,
        dt_bias_gdn,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        read_offsets=jnp.zeros((num_seqs,), dtype=jnp.int32),
        ckpt_indices=ckpt_indices,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        state_source=pool_source,
        state_plan=plan,
        num_spec_tokens=1,
    )
    assert out_src.shape == pool_source.shape
    assert out_act.shape == (batch_size, n_v * d_v)

    fn(
        qkv,
        b,
        a_gdn,
        None,
        None,
        conv_weight,
        None,
        a_log,
        dt_bias_gdn,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        state_source=pool_source,
        state_plan=plan,
        batched_only=True,
    )
    fn(
        qkv,
        b,
        a_gdn,
        None,
        None,
        conv_weight,
        None,
        a_log,
        dt_bias_gdn,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        state_source=pool_source,
        state_plan=plan,
        prefill_only=True,
    )


def test_td897_optimizations():
    from vllm_torchtpu.kernels.kimi_k3 import decode_kda

    # 1. SmemWrapper._get_pos unit-stride and zero-col fast path
    data = jnp.arange(8, dtype=jnp.int32)
    smem_unit = memory_ref.SmemWrapper(data, (1, 1))
    row_obj = object()
    assert smem_unit._get_pos((row_obj, 0)) is row_obj
    assert smem_unit[3, 0] == 3
    assert smem_unit[3, 1] == 4

    smem_strided = memory_ref.SmemWrapper(data, (1, 4))
    assert smem_strided[1, 0] == 4
    assert smem_strided[1, 2] == 6

    # 2. Cached CompilerParams and decode kernel partial
    assert wrapper._get_compiler_params(4096) is wrapper._get_compiler_params(4096)
    k1 = decode_kda._get_decode_kernel(
        4, 128, 4, -5.0, jnp.dtype(jnp.float32), 0.5, False
    )
    k2 = decode_kda._get_decode_kernel(
        4, 128, 4, -5.0, jnp.dtype(jnp.float32), 0.5, False
    )
    assert k1 is k2
