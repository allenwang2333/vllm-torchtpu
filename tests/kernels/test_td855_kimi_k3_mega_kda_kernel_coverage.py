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
"""Tests for TD8.55 Kimi-K3 Mega-KDA native segids kernel.

Provides 100% statement and branch coverage.
"""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.kimi_k3 import mega_kda
from vllm_torchtpu.kernels.kimi_k3.mega_kda import kernel


class _FakeRef:
    """NumPy-backed stand-in for Pallas/JAX Ref."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return jnp.dtype(self.arr.dtype)

    @property
    def ndim(self):
        return self.arr.ndim

    def _conv(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._conv(i) for i in idx)
        if isinstance(idx, pl.Slice):
            start = 0 if idx.start is None else int(np.asarray(idx.start))
            size = None if idx.size is None else int(np.asarray(idx.size))
            return slice(start, None if size is None else start + size)
        if isinstance(idx, (jax.Array, np.ndarray, np.generic)):
            a = np.asarray(idx)
            if a.ndim == 0:
                return int(a)
            return a
        return idx

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        return jnp.asarray(self.arr[self._conv(idx)])

    def __setitem__(self, idx, val):
        if idx is Ellipsis:
            self.arr[...] = np.asarray(val, dtype=self.arr.dtype)
            return
        self.arr[self._conv(idx)] = np.asarray(val, dtype=self.arr.dtype)

    @property
    def at(self):
        parent = self

        class _At:
            def __getitem__(self, idx):
                return _FakeRef(parent.arr[parent._conv(idx)])

        return _At()


class _FakeCopy:
    def start(self):
        pass

    def wait(self):
        pass


def _fake_pl_when(cond):
    def _dec(fn):
        if bool(np.asarray(cond)):
            fn()
        return fn

    return _dec


def test_align_up_and_build_chunk_metadata():
    assert kernel.align_up(1, 128) == 128
    assert kernel.align_up(128, 128) == 128
    assert kernel.align_up(129, 128) == 256

    # 1D segment_ids covering ALL_PAD (0), FULL_IN_SEGMENT (1),
    # PARTIAL_PAD (1, 0), BOUNDARY (1, 2)
    seg_1d = jnp.array(
        [
            0,
            0,
            0,
            0,  # ALL_PAD
            1,
            1,
            1,
            1,  # FULL_IN_SEGMENT
            1,
            1,
            0,
            0,  # PARTIAL_PAD
            1,
            1,
            2,
            2,  # BOUNDARY
        ],
        dtype=jnp.int32,
    )
    ck1, sf1, sl1, sid1 = kernel._build_chunk_metadata(seg_1d, chunk_size=4)
    assert ck1.tolist() == [
        kernel.CHUNK_KIND_ALL_PAD,
        kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        kernel.CHUNK_KIND_PARTIAL_PAD,
        kernel.CHUNK_KIND_BOUNDARY,
    ]
    assert sf1.tolist() == [0, 1, 1, 1]
    assert sl1.tolist() == [0, 1, 1, 2]
    assert sid1.tolist() == [0, 1, 0, 0]

    # 2D segment_ids
    seg_2d = seg_1d[None, :]
    ck2, sf2, sl2, sid2 = kernel._build_chunk_metadata(seg_2d, chunk_size=4)
    assert ck2.shape == (1, 4)
    assert sf2.shape == (1, 4)
    assert sl2.shape == (1, 4)
    assert sid2.shape == (1, 4)


def _invoke_kernel_body(
    *,
    kind_val: int,
    is_last_real: bool = False,
    first_seg: int = 1,
    last_seg: int = 2,
    prev_seg: int = 1,
    i_c: int = 0,
    BT: int = 8,
    MB: int = 2,
    K_PAD: int = 4,
    V_PAD: int = 4,
    QK_BC: int = 4,
    INV_BC: int = 4,
    safe_gate: bool = True,
    NORMALIZE_QK: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    USE_NEUMANN: bool = True,
    SKIP_STAGE4_MASK: bool = False,
    PACK_HEAD_INV: bool = False,
    CLIP_BETA_IN_KERNEL: bool = False,
    PACKED_METADATA: bool = False,
    HAS_H0: bool = False,
    STORE_RESIDUALS: bool = False,
    STORE_H: bool = False,
    STORE_FINAL_STATE: bool = False,
    BATCH_FIRST: bool = False,
    MANUAL_H0_DMA: bool = False,
    MANUAL_HT_DMA: bool = False,
    OVERLAP_H0_DMA: bool = False,
    OVERLAP_HT_DMA: bool = False,
    RESIDUAL_CHUNK_LAYOUT: bool = False,
    use_packed_wrapper: bool = False,
    beta_4d: bool = False,
):
    encoded = kind_val | (kernel.CHUNK_FLAG_LAST_REAL if is_last_real else 0)
    if PACKED_METADATA:
        packed_val = encoded | (first_seg << 3) | (last_seg << 17)
        ck_ref = _FakeRef(np.full((2, 8), packed_val, dtype=np.int32))
        sf_ref = ck_ref
        sl_ref = ck_ref
    else:
        ck_ref = _FakeRef(np.full((2, 8), encoded, dtype=np.int32))
        sf_ref = _FakeRef(np.full((2, 8), first_seg, dtype=np.int32))
        sl_ref = _FakeRef(np.full((2, 8), last_seg, dtype=np.int32))

    seg_row = np.ones((1, 128), dtype=np.int32)
    if kind_val == kernel.CHUNK_KIND_PARTIAL_PAD:
        seg_row[:, BT // 2 : BT] = 0
    elif kind_val == kernel.CHUNK_KIND_BOUNDARY:
        seg_row[:, : BT // 2] = first_seg
        seg_row[:, BT // 2 : BT] = last_seg
    seg_ids_ref = _FakeRef(seg_row)

    if BATCH_FIRST:
        q_ref = _FakeRef(np.ones((1, BT, MB, K_PAD), dtype=np.float32) * 0.1)
        k_ref = _FakeRef(np.ones((1, BT, MB, K_PAD), dtype=np.float32) * 0.1)
        v_ref = _FakeRef(np.ones((1, BT, MB, V_PAD), dtype=np.float32) * 0.1)
        g_ref = _FakeRef(np.ones((1, BT, MB, K_PAD), dtype=np.float32) * -0.1)
        if beta_4d:
            beta_ref = _FakeRef(np.ones((1, 1, BT, MB), dtype=np.float32) * 0.5)
        else:
            beta_ref = _FakeRef(np.ones((MB, 1, 1, 1, BT), dtype=np.float32) * 0.5)
        o_ref = _FakeRef(np.zeros((1, BT, MB, V_PAD), dtype=np.float32))
    else:
        q_ref = _FakeRef(np.ones((MB, 1, BT, K_PAD), dtype=np.float32) * 0.1)
        k_ref = _FakeRef(np.ones((MB, 1, BT, K_PAD), dtype=np.float32) * 0.1)
        v_ref = _FakeRef(np.ones((MB, 1, BT, V_PAD), dtype=np.float32) * 0.1)
        g_ref = _FakeRef(np.ones((MB, 1, BT, K_PAD), dtype=np.float32) * -0.1)
        beta_ref = _FakeRef(np.ones((MB, 1, 1, 1, BT), dtype=np.float32) * 0.5)
        o_ref = _FakeRef(np.zeros((MB, 1, BT, V_PAD), dtype=np.float32))

    h0_ref = _FakeRef(np.zeros((4, MB, 1, K_PAD, V_PAD), dtype=np.float32))
    A_scale_ref = _FakeRef(np.ones((MB, 1, 1, 1), dtype=np.float32))
    dt_bias_ref = _FakeRef(np.zeros((MB, 1, 1, K_PAD), dtype=np.float32))
    ht_ref = _FakeRef(np.zeros((4, MB, 1, K_PAD, V_PAD), dtype=np.float32))

    if RESIDUAL_CHUNK_LAYOUT:
        Aqk_ref = _FakeRef(np.zeros((MB, 1, 1, BT, BT), dtype=np.float32))
        Akk_ref = _FakeRef(np.zeros((MB, 1, 1, BT, BT), dtype=np.float32))
    else:
        Aqk_ref = _FakeRef(np.zeros((MB, 1, BT, BT), dtype=np.float32))
        Akk_ref = _FakeRef(np.zeros((MB, 1, BT, BT), dtype=np.float32))

    g_cumsum_out_ref = _FakeRef(np.zeros((MB, 1, BT, K_PAD), dtype=np.float32))
    chunk_h_out_ref = _FakeRef(np.zeros((MB, 1, 1, K_PAD, V_PAD), dtype=np.float32))
    scratch_ref = _FakeRef(np.zeros((MB, K_PAD, V_PAD), dtype=np.float32))
    prev_seg_ref = _FakeRef(np.full((128,), prev_seg, dtype=np.int32))
    final_state_dma_ref = _FakeRef(np.zeros((MB, K_PAD, V_PAD), dtype=np.float32))
    state_dma_sem = _FakeRef(np.zeros((1,), dtype=np.int32))

    def _prog_id(axis):
        if axis == 0:
            return jnp.int32(0)
        if axis == 1:
            return jnp.int32(0)
        return jnp.int32(i_c)

    kwargs = dict(
        NT=4,
        BT=BT,
        N_max=4,
        scale=1.0,
        cumsum_scale=kernel._RCP_LN2,
        MB=MB,
        K_PAD=K_PAD,
        V_PAD=V_PAD,
        OUTPUT_PRECISION=jax.lax.Precision.DEFAULT,
        safe_gate=safe_gate,
        NORMALIZE_QK=NORMALIZE_QK,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        USE_NEUMANN=USE_NEUMANN,
        QK_BC=QK_BC,
        INV_BC=INV_BC,
        SKIP_STAGE4_MASK=SKIP_STAGE4_MASK,
        PACK_HEAD_INV=PACK_HEAD_INV,
        CLIP_BETA_IN_KERNEL=CLIP_BETA_IN_KERNEL,
        PACKED_METADATA=PACKED_METADATA,
        HAS_H0=HAS_H0,
        STORE_RESIDUALS=STORE_RESIDUALS,
        STORE_H=STORE_H,
        STORE_FINAL_STATE=STORE_FINAL_STATE,
        BATCH_FIRST=BATCH_FIRST,
        MANUAL_H0_DMA=MANUAL_H0_DMA,
        MANUAL_HT_DMA=MANUAL_HT_DMA,
        OVERLAP_H0_DMA=OVERLAP_H0_DMA,
        OVERLAP_HT_DMA=OVERLAP_HT_DMA,
        RESIDUAL_CHUNK_LAYOUT=RESIDUAL_CHUNK_LAYOUT,
    )

    with (
        patch.object(pl, "program_id", side_effect=_prog_id),
        patch.object(pl, "when", side_effect=_fake_pl_when),
        patch.object(pltpu, "make_async_copy", return_value=_FakeCopy()),
    ):
        if use_packed_wrapper:
            kernel._fwd_mega_kernel_native_segids_packed(
                ck_ref,
                seg_ids_ref,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                h0_ref,
                A_scale_ref,
                dt_bias_ref,
                o_ref,
                ht_ref,
                Aqk_ref,
                Akk_ref,
                g_cumsum_out_ref,
                chunk_h_out_ref,
                scratch_ref,
                prev_seg_ref,
                final_state_dma_ref,
                state_dma_sem,
                **kwargs,
            )
        else:
            kernel._fwd_mega_kernel_native_segids(
                ck_ref,
                sf_ref,
                sl_ref,
                seg_ids_ref,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                h0_ref,
                A_scale_ref,
                dt_bias_ref,
                o_ref,
                ht_ref,
                Aqk_ref,
                Akk_ref,
                g_cumsum_out_ref,
                chunk_h_out_ref,
                scratch_ref,
                prev_seg_ref,
                final_state_dma_ref,
                state_dma_sem,
                **kwargs,
            )


def test_fwd_mega_kernel_all_pad_and_full_in_segment_branches():
    # 1. ALL_PAD: BATCH_FIRST=True/False, STORE_RESIDUALS=True/False,
    # RESIDUAL_CHUNK_LAYOUT=True/False
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_ALL_PAD,
        BATCH_FIRST=True,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=True,
        PACKED_METADATA=True,
        use_packed_wrapper=True,
    )
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_ALL_PAD,
        BATCH_FIRST=False,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=False,
    )
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_ALL_PAD,
        BATCH_FIRST=False,
        STORE_RESIDUALS=False,
    )

    # 2. FULL_IN_SEGMENT:
    # Case A: i_c=1, prev_seg=2 != first_seg=1 (seg_changed & ~chain_start True!),
    # STORE_FINAL_STATE=True, MANUAL_HT_DMA=True, HAS_H0=True, MANUAL_H0_DMA=True,
    # BATCH_FIRST=True, beta_4d=True, CLIP_BETA_IN_KERNEL=True, NORMALIZE_QK=True,
    # use_gate_in_kernel=True (lower_bound=None), safe_gate=True,
    # PACK_HEAD_INV=False, BT=32, QK_BC=16, INV_BC=4 (NC_inv=8 -> num_horner_steps=2),
    # STORE_RESIDUALS=True, RESIDUAL_CHUNK_LAYOUT=True, STORE_H=True,
    # SKIP_STAGE4_MASK=True, is_last_real=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        is_last_real=True,
        first_seg=1,
        prev_seg=2,
        i_c=1,
        BT=32,
        QK_BC=16,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=True,
        HAS_H0=True,
        MANUAL_H0_DMA=True,
        BATCH_FIRST=True,
        beta_4d=True,
        CLIP_BETA_IN_KERNEL=True,
        NORMALIZE_QK=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        safe_gate=True,
        PACK_HEAD_INV=False,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=True,
        STORE_H=True,
        SKIP_STAGE4_MASK=True,
    )

    # Case A_pack: PACK_HEAD_INV=True with STORE_RESIDUALS=False, STORE_H=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        is_last_real=True,
        first_seg=1,
        prev_seg=2,
        i_c=1,
        BT=32,
        QK_BC=16,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=True,
        HAS_H0=True,
        MANUAL_H0_DMA=True,
        BATCH_FIRST=True,
        beta_4d=True,
        CLIP_BETA_IN_KERNEL=True,
        NORMALIZE_QK=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        safe_gate=True,
        PACK_HEAD_INV=True,
        STORE_RESIDUALS=False,
        STORE_H=True,
        SKIP_STAGE4_MASK=True,
    )

    # Case B: i_c=1, prev_seg=2 != first_seg=1, MANUAL_HT_DMA=False,
    # MANUAL_H0_DMA=False, BATCH_FIRST=False, CLIP_BETA_IN_KERNEL=False,
    # NORMALIZE_QK=False, use_gate_in_kernel=True (lower_bound=-5.0), safe_gate=False,
    # PACK_HEAD_INV=False, BT=16, QK_BC=8, INV_BC=4 (NC_inv=4 -> num_horner_steps=1),
    # STORE_RESIDUALS=True, RESIDUAL_CHUNK_LAYOUT=False, STORE_H=False,
    # SKIP_STAGE4_MASK=False, is_last_real=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        is_last_real=True,
        first_seg=1,
        prev_seg=2,
        i_c=1,
        BT=16,
        QK_BC=8,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=False,
        HAS_H0=True,
        MANUAL_H0_DMA=False,
        BATCH_FIRST=False,
        CLIP_BETA_IN_KERNEL=False,
        NORMALIZE_QK=False,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        safe_gate=False,
        PACK_HEAD_INV=False,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=False,
        STORE_H=False,
        SKIP_STAGE4_MASK=False,
    )

    # Case C: i_c=1, prev_seg=1 == first_seg=1 (seg_changed=False!),
    # STORE_FINAL_STATE=False, HAS_H0=False, use_gate_in_kernel=False,
    # BT=8, QK_BC=4, INV_BC=4 (NC_inv=2 -> num_horner_steps=0), STORE_RESIDUALS=False
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        is_last_real=False,
        first_seg=1,
        prev_seg=1,
        i_c=1,
        BT=8,
        QK_BC=4,
        INV_BC=4,
        STORE_FINAL_STATE=False,
        HAS_H0=False,
        use_gate_in_kernel=False,
        STORE_RESIDUALS=False,
    )

    # Case D: NC_inv == 1 (BT=8, INV_BC=8) with STORE_RESIDUALS=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        BT=8,
        QK_BC=4,
        INV_BC=8,
        STORE_FINAL_STATE=True,
        STORE_RESIDUALS=True,
        is_last_real=False,
    )

    # Case D2: NC_inv == 1 (BT=8, INV_BC=8) with STORE_RESIDUALS=False
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
        BT=8,
        QK_BC=4,
        INV_BC=8,
        STORE_FINAL_STATE=False,
        STORE_RESIDUALS=False,
        is_last_real=False,
    )

    # Case E: USE_NEUMANN=False raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Neumann"):
        _invoke_kernel_body(
            kind_val=kernel.CHUNK_KIND_FULL_IN_SEGMENT,
            USE_NEUMANN=False,
        )


def test_fwd_mega_kernel_partial_pad_and_boundary_branches():
    # 1. PARTIAL_PAD:
    # Case A: BT=128 (hits _load_token_segment_ids BT == 128), i_c=1,
    # prev_seg=2 != first_seg=1, STORE_FINAL_STATE=True, MANUAL_HT_DMA=True,
    # HAS_H0=True, MANUAL_H0_DMA=True, BATCH_FIRST=True, beta_4d=True,
    # CLIP_BETA_IN_KERNEL=True, NORMALIZE_QK=True,
    # use_gate_in_kernel=True (lower_bound=None), QK_BC=64,
    # INV_BC=16 (NC_inv=8 -> num_horner_steps=2), STORE_RESIDUALS=True,
    # RESIDUAL_CHUNK_LAYOUT=True, STORE_H=True, SKIP_STAGE4_MASK=True,
    # is_last_real=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_PARTIAL_PAD,
        is_last_real=True,
        first_seg=1,
        prev_seg=2,
        i_c=1,
        BT=128,
        QK_BC=64,
        INV_BC=16,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=True,
        HAS_H0=True,
        MANUAL_H0_DMA=True,
        BATCH_FIRST=True,
        beta_4d=True,
        CLIP_BETA_IN_KERNEL=True,
        NORMALIZE_QK=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=True,
        STORE_H=True,
        SKIP_STAGE4_MASK=True,
    )

    # Case B: BT=16 (hits _load_token_segment_ids BT != 128), i_c=1, prev_seg=2,
    # MANUAL_HT_DMA=False, MANUAL_H0_DMA=False, BATCH_FIRST=False,
    # use_gate_in_kernel=True (lower_bound=-5.0), safe_gate=False,
    # QK_BC=8, INV_BC=4 (NC_inv=4 -> num_horner_steps=1),
    # STORE_RESIDUALS=True, RESIDUAL_CHUNK_LAYOUT=False, STORE_H=False,
    # SKIP_STAGE4_MASK=False, is_last_real=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_PARTIAL_PAD,
        is_last_real=True,
        first_seg=1,
        prev_seg=2,
        i_c=1,
        BT=16,
        QK_BC=8,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=False,
        HAS_H0=True,
        MANUAL_H0_DMA=False,
        BATCH_FIRST=False,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        safe_gate=False,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=False,
        STORE_H=False,
        SKIP_STAGE4_MASK=False,
    )

    # Case C: i_c=1, prev_seg=1 (seg_changed=False), STORE_FINAL_STATE=False,
    # HAS_H0=False, use_gate_in_kernel=False, BT=8, QK_BC=4,
    # INV_BC=4 (NC_inv=2 -> num_horner_steps=0), STORE_RESIDUALS=False
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_PARTIAL_PAD,
        is_last_real=False,
        first_seg=1,
        prev_seg=1,
        i_c=1,
        BT=8,
        QK_BC=4,
        INV_BC=4,
        STORE_FINAL_STATE=False,
        HAS_H0=False,
        use_gate_in_kernel=False,
        STORE_RESIDUALS=False,
    )

    # Case D: NC_inv == 1 (BT=8, INV_BC=8), STORE_FINAL_STATE=True, is_last_real=False
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_PARTIAL_PAD,
        BT=8,
        QK_BC=4,
        INV_BC=8,
        STORE_FINAL_STATE=True,
        is_last_real=False,
    )

    # Case E: USE_NEUMANN=False raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Neumann"):
        _invoke_kernel_body(
            kind_val=kernel.CHUNK_KIND_PARTIAL_PAD,
            USE_NEUMANN=False,
        )

    # 2. BOUNDARY:
    # Case A: i_c=1, prev_seg=3 != first_seg=1,
    # STORE_FINAL_STATE=True, MANUAL_HT_DMA=True, OVERLAP_HT_DMA=True,
    # HAS_H0=True, MANUAL_H0_DMA=True, OVERLAP_H0_DMA=True,
    # BATCH_FIRST=True, beta_4d=True, CLIP_BETA_IN_KERNEL=True, NORMALIZE_QK=True,
    # use_gate_in_kernel=True (lower_bound=None), PACK_HEAD_INV=False,
    # BT=32, QK_BC=16, INV_BC=4 (NC_inv=8 -> num_horner_steps=2),
    # STORE_RESIDUALS=True, RESIDUAL_CHUNK_LAYOUT=True, STORE_H=True,
    # SKIP_STAGE4_MASK=True, is_last_real=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_BOUNDARY,
        is_last_real=True,
        first_seg=1,
        last_seg=2,
        prev_seg=3,
        i_c=1,
        BT=32,
        QK_BC=16,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=True,
        OVERLAP_HT_DMA=True,
        HAS_H0=True,
        MANUAL_H0_DMA=True,
        OVERLAP_H0_DMA=True,
        BATCH_FIRST=True,
        beta_4d=True,
        CLIP_BETA_IN_KERNEL=True,
        NORMALIZE_QK=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        PACK_HEAD_INV=False,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=True,
        STORE_H=True,
        SKIP_STAGE4_MASK=True,
    )

    # Case A_pack: PACK_HEAD_INV=True with STORE_RESIDUALS=False, STORE_H=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_BOUNDARY,
        is_last_real=True,
        first_seg=1,
        last_seg=2,
        prev_seg=3,
        i_c=1,
        BT=32,
        QK_BC=16,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=True,
        OVERLAP_HT_DMA=True,
        HAS_H0=True,
        MANUAL_H0_DMA=True,
        OVERLAP_H0_DMA=True,
        BATCH_FIRST=True,
        beta_4d=True,
        CLIP_BETA_IN_KERNEL=True,
        NORMALIZE_QK=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        PACK_HEAD_INV=True,
        STORE_RESIDUALS=False,
        STORE_H=True,
        SKIP_STAGE4_MASK=True,
    )

    # Case B: MANUAL_HT_DMA=True, OVERLAP_HT_DMA=False, MANUAL_H0_DMA=True,
    # OVERLAP_H0_DMA=False, BT=16, QK_BC=8, INV_BC=4 (NC_inv=4 -> num_horner_steps=1)
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_BOUNDARY,
        is_last_real=False,
        first_seg=1,
        last_seg=2,
        prev_seg=1,
        i_c=1,
        BT=16,
        QK_BC=8,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=True,
        OVERLAP_HT_DMA=False,
        HAS_H0=True,
        MANUAL_H0_DMA=True,
        OVERLAP_H0_DMA=False,
        STORE_RESIDUALS=False,
    )

    # Case C: MANUAL_HT_DMA=False, MANUAL_H0_DMA=False, BATCH_FIRST=False,
    # use_gate_in_kernel=True (lower_bound=-5.0), safe_gate=False,
    # BT=8, QK_BC=4, INV_BC=4 (NC_inv=2 -> num_horner_steps=0),
    # STORE_RESIDUALS=True, RESIDUAL_CHUNK_LAYOUT=False,
    # SKIP_STAGE4_MASK=False, is_last_real=True
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_BOUNDARY,
        is_last_real=True,
        first_seg=1,
        last_seg=2,
        prev_seg=3,
        i_c=1,
        BT=8,
        QK_BC=4,
        INV_BC=4,
        STORE_FINAL_STATE=True,
        MANUAL_HT_DMA=False,
        HAS_H0=True,
        MANUAL_H0_DMA=False,
        BATCH_FIRST=False,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        safe_gate=False,
        STORE_RESIDUALS=True,
        RESIDUAL_CHUNK_LAYOUT=False,
        SKIP_STAGE4_MASK=False,
    )

    # Case D: STORE_FINAL_STATE=False, HAS_H0=False, NC_inv == 1 (BT=8, INV_BC=8)
    _invoke_kernel_body(
        kind_val=kernel.CHUNK_KIND_BOUNDARY,
        BT=8,
        QK_BC=4,
        INV_BC=8,
        STORE_FINAL_STATE=False,
        HAS_H0=False,
    )

    # Case E: USE_NEUMANN=False raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Neumann"):
        _invoke_kernel_body(
            kind_val=kernel.CHUNK_KIND_BOUNDARY,
            USE_NEUMANN=False,
        )


def test_chunk_kda_fwd_native_segids_impl_and_jit_wrappers():
    # Direct unit test of module-level index map helpers
    assert kernel._h0_map(1, 2, 3) == (0, 1, 2, 0, 0)
    assert kernel._out_map(1, 2, 3) == (1, 2, 3, 0)
    assert kernel._out_chunk_map(1, 2, 3) == (1, 2, 3, 0, 0)
    assert kernel._alog_map(1, 2, 3) == (1, 0, 0, 0)
    assert kernel._o_batch_first_map(1, 2, 3) == (2, 3, 1, 0)
    assert kernel._beta_batch_first_map(1, 2, 3) == (2, 3, 0, 1)
    assert kernel._beta_head_first_map(1, 2, 3) == (1, 2, 3, 0, 0)

    def _fake_pallas_call(fn, out_shape, grid_spec, compiler_params):
        # Exercise _seg_map from grid_spec.in_specs[0]
        assert grid_spec.in_specs[0].index_map(0, 0, 0) == (0, 0)

        def _runner(*args):
            return tuple(
                jnp.zeros(s.shape, dtype=s.dtype) if s is not None else None
                for s in out_shape
            )

        return _runner

    with patch.object(pl, "pallas_call", side_effect=_fake_pallas_call):
        # 1. Error check: only_fwd=True with store_h=True
        q_bf = jnp.ones((2, 64, 4, 64), dtype=jnp.bfloat16)
        k_bf = jnp.ones((2, 64, 4, 64), dtype=jnp.bfloat16)
        v_bf = jnp.ones((2, 64, 4, 64), dtype=jnp.bfloat16)
        g_bf = jnp.ones((2, 64, 4, 64), dtype=jnp.bfloat16)
        beta_bf = jnp.ones((2, 64, 4), dtype=jnp.bfloat16)
        seg_1d = jnp.ones((64,), dtype=jnp.int32)

        with pytest.raises(ValueError, match="only_fwd=True is incompatible"):
            kernel._chunk_kda_fwd_native_segids_impl(
                q_bf,
                k_bf,
                v_bf,
                g_bf,
                beta_bf,
                seg_1d,
                only_fwd=True,
                store_h=True,
                batch_first=True,
            )

        # 2. batch_first=True, only_fwd=True, packed_metadata=True (meta_pad > 0),
        # 1D segment_ids with B=2, N_max=None, K=64 < 128, V=64 < 128, T=64 < 128,
        # initial_state 4D [1, 1, 4, 64, 64], use_gate_in_kernel=True with n_rep > 1,
        # output_final_state=True, manual_h0_dma=False, manual_ht_dma=False,
        # env_mini_batch=4
        h0_4d = jnp.ones((1, 4, 64, 64), dtype=jnp.float32)
        A_log = jnp.ones((2,), dtype=jnp.float32)
        dt_bias = jnp.ones((2 * 64,), dtype=jnp.float32)
        with (
            patch.object(envs, "KDA_PACKED_METADATA", True),
            patch.object(envs, "KDA_FWD_MB", 4),
            patch.object(envs, "KDA_MANUAL_H0_DMA", False),
            patch.object(envs, "KDA_MANUAL_HT_DMA", False),
        ):
            out = kernel._chunk_kda_fwd_native_segids_impl(
                q_bf,
                k_bf,
                v_bf,
                g_bf,
                beta_bf,
                seg_1d,
                initial_state=h0_4d,
                output_final_state=True,
                only_fwd=True,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                batch_first=True,
            )
            assert out[0].shape == (2, 64, 4, 64)
            assert out[1].shape == (2, 1, 4, 64, 64)

        # 3. batch_first=False, only_fwd=True, packed_metadata=True but
        # N_max=16384 -> demotes to False! Also test K=128, V=128, T=128,
        # chunk_size=1 -> NT_meta=128 (meta_pad == 0!), segment_ids 1D with B=1,
        # mini_batch=6 with H=4 (while H % MB != 0 loop runs!),
        # manual_h0_dma=True, manual_ht_dma=True, A_log with n_rep == 1
        q_hf = jnp.ones((4, 1, 128, 128), dtype=jnp.float32)
        k_hf = jnp.ones((4, 1, 128, 128), dtype=jnp.float32)
        v_hf = jnp.ones((4, 1, 128, 128), dtype=jnp.float32)
        g_hf = jnp.ones((4, 1, 128, 128), dtype=jnp.float32)
        beta_hf = jnp.ones((1, 128, 4), dtype=jnp.float32)
        seg_128 = jnp.ones((128,), dtype=jnp.int32)
        h0_5d = jnp.ones((1, 16384, 4, 128, 128), dtype=jnp.float32)
        A_log_4 = jnp.ones((4,), dtype=jnp.float32)
        dt_bias_4 = jnp.ones((4 * 128,), dtype=jnp.float32)
        with (
            patch.object(envs, "KDA_PACKED_METADATA", True),
            patch.object(envs, "KDA_FWD_MB", None),
            patch.object(envs, "KDA_MANUAL_H0_DMA", None),
            patch.object(envs, "KDA_MANUAL_HT_DMA", None),
            patch.object(envs, "KDA_MANUAL_STATE_DMA", None),
        ):
            out2 = kernel._chunk_kda_fwd_native_segids_impl(
                q_hf,
                k_hf,
                v_hf,
                g_hf,
                beta_hf,
                seg_128,
                initial_state=h0_5d,
                output_final_state=True,
                only_fwd=True,
                chunk_size=1,
                use_gate_in_kernel=True,
                A_log=A_log_4,
                dt_bias=dt_bias_4,
                mini_batch=6,
                N_max=16384,
                batch_first=False,
            )
            assert out2[0].shape == (4, 1, 128, 128)
            assert out2[1].shape == (1, 16384, 4, 128, 128)

        # 4. packed_metadata=True with meta_pad == 0 (N_max=2, T=128,
        # chunk_size=1 -> NT_meta=128), mini_batch=None and KDA_FWD_MB=None
        # -> MB = min(H, 32) = 8 > 4 (hits _COMPILER_PARAMS_MB_GT4!)
        q_h8 = jnp.ones((8, 1, 128, 128), dtype=jnp.bfloat16)
        v_h8 = jnp.ones((8, 1, 128, 128), dtype=jnp.bfloat16)
        beta_h8 = jnp.ones((1, 128, 8), dtype=jnp.bfloat16)
        with (
            patch.object(envs, "KDA_PACKED_METADATA", True),
            patch.object(envs, "KDA_FWD_MB", None),
        ):
            out3 = kernel._chunk_kda_fwd_native_segids_impl(
                q_h8,
                q_h8,
                v_h8,
                q_h8,
                beta_h8,
                seg_128[None, :],
                only_fwd=True,
                chunk_size=1,
                N_max=2,
                batch_first=False,
            )
            assert out3[0].shape == (8, 1, 128, 128)
            assert out3[1] is None

        # 5. Training / backward-intermediates path: only_fwd=False,
        # disable_recompute=True (sets store_h=True),
        # residual_chunk_layout=True & False, K < 128 & K == 128
        out4 = kernel._chunk_kda_fwd_native_segids_impl(
            q_bf.transpose(2, 0, 1, 3),
            k_bf.transpose(2, 0, 1, 3),
            v_bf.transpose(2, 0, 1, 3),
            g_bf.transpose(2, 0, 1, 3),
            beta_bf,
            jnp.ones((2, 64), dtype=jnp.int32),
            only_fwd=False,
            disable_recompute=True,
            residual_chunk_layout=True,
            batch_first=False,
        )
        assert out4[2].shape == (4, 2, 64, 64)
        assert out4[10].shape == (4, 2, 1, 64, 64)

        out5 = kernel._chunk_kda_fwd_native_segids_impl(
            q_hf,
            k_hf,
            v_hf,
            g_hf,
            beta_hf,
            seg_128[None, :],
            only_fwd=False,
            store_h=False,
            residual_chunk_layout=False,
            batch_first=False,
        )
        assert out5[2].shape == (4, 1, 128, 128)
        assert out5[10] is None

    # 6. _get_native_residual_layout_jit and _native_residual_layout_jit_for_sharding
    assert (
        kernel._get_native_residual_layout_jit(SimpleNamespace(sharding=None))
        is kernel._chunk_kda_fwd_native_segids
    )
    kernel._native_residual_layout_jit_for_sharding.cache_clear()
    with patch("jax.jit", return_value="custom_jit_fn") as mock_jit:
        fake_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        res_jit = kernel._get_native_residual_layout_jit(
            SimpleNamespace(sharding=fake_sharding)
        )
        assert res_jit == "custom_jit_fn"
        mock_jit.assert_called_once()


def test_mega_kda_init_module_coverage():
    # 1. _layout_supported
    with pytest.raises(ValueError, match="full token tiles"):
        mega_kda._layout_supported(jnp.array([0, 64]), num_tokens=65, chunk_size=64)

    assert bool(mega_kda._layout_supported(jnp.array([0, 64, 128]), num_tokens=128))

    # > 3 request slots: True case
    ql_ok = jnp.array([0, 32, 64, 96, 128], dtype=jnp.int32)
    assert bool(mega_kda._layout_supported(ql_ok, num_tokens=128, chunk_size=64))

    # > 3 request slots: False case (three requests in tile [0, 64])
    ql_bad = jnp.array([0, 10, 20, 30, 128], dtype=jnp.int32)
    assert not bool(mega_kda._layout_supported(ql_bad, num_tokens=128, chunk_size=64))

    # 2. kda_forward_inference input validations
    q = jnp.ones((1, 64, 4, 64), dtype=jnp.bfloat16)
    k = jnp.ones((1, 64, 4, 64), dtype=jnp.bfloat16)
    v = jnp.ones((1, 64, 4, 64), dtype=jnp.bfloat16)
    g = jnp.ones((1, 64, 4, 64), dtype=jnp.bfloat16)
    beta = jnp.ones((1, 64, 4), dtype=jnp.bfloat16)
    seg = jnp.ones((1, 64), dtype=jnp.int32)
    A_log = jnp.ones((4,), dtype=jnp.float32)
    dt_bias = jnp.ones((4 * 64,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires BF16"):
        mega_kda.kda_forward_inference(
            q.astype(jnp.float32),
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="same dtype"):
        mega_kda.kda_forward_inference(
            q,
            k.astype(jnp.float32),
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="beta must have the same dtype"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta.astype(jnp.float32),
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="chunk_size=64"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            chunk_size=32,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="agree on"):
        mega_kda.kda_forward_inference(
            q,
            k[:, :, :2],
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="agree with q on"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v[:, :, :2],
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="same key dimension"):
        mega_kda.kda_forward_inference(
            q,
            k[..., :32],
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="multiple of chunk_size"):
        mega_kda.kda_forward_inference(
            q[:, :32],
            k[:, :32],
            v[:, :32],
            g[:, :32],
            beta[:, :32],
            segment_ids=seg[:, :32],
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="segment_ids must have shape"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=jnp.ones((2, 64), dtype=jnp.int32),
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="segment_ids must use int32"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg.astype(jnp.int16),
            A_log=A_log,
            dt_bias=dt_bias,
        )

    with pytest.raises(ValueError, match="requires A_log and dt_bias"):
        mega_kda.kda_forward_inference(
            q, k, v, g, beta, segment_ids=seg, A_log=None, dt_bias=dt_bias
        )

    with pytest.raises(ValueError, match="requires A_log and dt_bias"):
        mega_kda.kda_forward_inference(
            q, k, v, g, beta, segment_ids=seg, A_log=A_log, dt_bias=None
        )

    with pytest.raises(ValueError, match="in-kernel Q/K L2 normalization"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=False,
        )

    with pytest.raises(ValueError, match="safe fused gate activation"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
        )

    with pytest.raises(ValueError, match="safe fused gate activation"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=False,
        )

    with pytest.raises(ValueError, match="lower_bound in"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=None,
        )

    with pytest.raises(ValueError, match="lower_bound in"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=1.0,
        )

    with pytest.raises(ValueError, match="N_max is required"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-2.0,
            N_max=None,
        )

    with pytest.raises(ValueError, match="initial_state must use float32"):
        mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-2.0,
            initial_state=jnp.ones((1, 4, 64, 64), dtype=jnp.bfloat16),
        )

    # Valid invocation with 1D segment_ids, scale=None, N_max from initial_state
    h0 = jnp.zeros((1, 2, 4, 64, 64), dtype=jnp.float32)
    with patch(
        "vllm_torchtpu.kernels.kimi_k3.mega_kda._chunk_kda_fwd_native_segids",
        return_value=(q, h0),
    ) as mock_impl:
        out_o, out_h = mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg[0],
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-2.0,
            initial_state=h0,
            output_final_state=True,
        )
        assert out_o.shape == q.shape
        assert out_h.shape == h0.shape
        mock_impl.assert_called_once()

    # Valid invocation with scale specified and initial_state=None, N_max specified
    with patch(
        "vllm_torchtpu.kernels.kimi_k3.mega_kda._chunk_kda_fwd_native_segids",
        return_value=(q, None),
    ) as mock_impl2:
        out_o2, out_h2 = mega_kda.kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=seg,
            scale=0.5,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-2.0,
            N_max=4,
        )
        assert out_o2.shape == q.shape
        assert out_h2 is None
        mock_impl2.assert_called_once()
