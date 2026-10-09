# SPDX-License-Identifier: Apache-2.0
"""Unit tests and 100% diff coverage for v2/attention_kernels.py fast paths (TD7.45)."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak


def _make_pool(rows: int = 256, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    bf16 = jnp.asarray(rng.uniform(-1.0, 1.0, size=(rows, 256)), dtype=jnp.bfloat16)
    return bf16.view(jnp.int32)


class TestV2AttentionKernelsTD745:
    def test_attend_zero_pad_and_nonzero_pad(self):
        pool = _make_pool(128, seed=1)
        q = jnp.ones((2, 4, 128), dtype=jnp.bfloat16)
        state = ak._attention_state(q)

        # 1. Empty width == 0 returns state untouched
        empty_records = jnp.zeros((2, 0), dtype=jnp.int32)
        assert ak._attend(q, pool, empty_records, state, scale=0.5, tile=64) is state

        # 2. pad == 0 (width=64, tile=64) skips jnp.pad
        records_aligned = jnp.arange(128, dtype=jnp.int32).reshape(2, 64)
        max_a, den_a, num_a = ak._attend(
            q, pool, records_aligned, state, scale=0.5, tile=64
        )
        assert num_a.shape == (2, 4, 128)
        assert jnp.all(den_a > 0)

        # 3. pad > 0 (width=32, tile=64) pads records with -1
        records_unaligned = jnp.arange(64, dtype=jnp.int32).reshape(2, 32)
        max_u, den_u, num_u = ak._attend(
            q, pool, records_unaligned, state, scale=0.5, tile=64
        )
        assert num_u.shape == (2, 4, 128)
        assert jnp.all(den_u > 0)

    def test_select_indices_jax_candidate_write_pad_and_nopad(self):
        rng = np.random.default_rng(2)
        lengths = jnp.array([20, 22], dtype=jnp.int32)
        candidates = jnp.zeros((2, 1), dtype=jnp.int32)

        # 1. pad == 0 (width=32, block_size=16 -> blocks=2, pad=0)
        scores_aligned = jnp.asarray(rng.standard_normal((2, 32)), dtype=jnp.float32)
        sel_a, pub_a = ak.select_indices_jax(
            scores_aligned,
            lengths,
            candidates,
            topk=8,
            block_size=16,
            candidate_write=True,
            candidate_filter=False,
        )
        assert sel_a.shape == (2, 8)
        assert pub_a.shape == (2, 1)

        # 2. pad > 0 (width=24, block_size=16 -> blocks=2, pad=8)
        scores_unaligned = jnp.asarray(rng.standard_normal((2, 24)), dtype=jnp.float32)
        sel_u, pub_u = ak.select_indices_jax(
            scores_unaligned,
            lengths,
            candidates,
            topk=8,
            block_size=16,
            candidate_write=True,
            candidate_filter=False,
        )
        assert sel_u.shape == (2, 8)
        assert pub_u.shape == (2, 1)

    def test_paged_indexer_jax_3d_broadcast_and_padding_branches(self):
        pool = _make_pool(256, seed=3)
        rng = np.random.default_rng(3)
        tokens, heads, dim, width = 2, 2, 128, 128
        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), dtype=jnp.bfloat16)
        weights = jnp.asarray(rng.standard_normal((tokens, heads)), dtype=jnp.float32)
        records = jnp.arange(tokens * width, dtype=jnp.int32).reshape(tokens, width)
        lengths = jnp.array([96, 112], dtype=jnp.int32)
        requests = jnp.array([0, 1], dtype=jnp.int32)
        candidates = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)

        # 1. full_context with page_records=16 (exercises 3D broadcast row_valid)
        sel_fc_paged, pub_fc_paged = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=8,
            block_size=16,
            candidate_write=True,
            candidate_filter=False,
            page_records=16,
        )
        assert sel_fc_paged.shape == (tokens, 8)
        assert pub_fc_paged.shape == (tokens, 2)

        # 2. full_context with page_records=None (exercises else branch)
        sel_fc_plain, _ = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            page_records=None,
        )
        np.testing.assert_array_equal(
            np.asarray(sel_fc_paged), np.asarray(sel_fc_plain)
        )

        # 3. candidate_context with page_records=16, block_size=32 (per_block=2),
        #    and topk=8 <= 64 (padding == 0 fast path)
        sel_cc_nopad, pub_cc_nopad = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=8,
            block_size=32,
            candidate_write=False,
            candidate_filter=True,
            page_records=16,
        )
        assert sel_cc_nopad.shape == (tokens, 8)
        assert pub_cc_nopad is candidates

        # 4. candidate_context with page_records=None and topk=48 > 32
        #    (padding > 0 path: scores.shape[1]=32, output_width=48 -> padding=16)
        sel_cc_pad, pub_cc_pad = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=48,
            block_size=16,
            candidate_write=False,
            candidate_filter=True,
            page_records=None,
        )
        assert sel_cc_pad.shape == (tokens, 48)
        assert pub_cc_pad is candidates

    def test_torch_wrappers_int32_fast_paths_and_conversions(self, monkeypatch):
        class _RecordingOp:
            def __init__(self, ret):
                self.ret = ret
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)
                return self.ret

        out_tensor = torch.zeros((4, 16), dtype=torch.bfloat16)
        op_qo = _RecordingOp(out_tensor)
        op_attn = _RecordingOp(out_tensor)
        op_fp8 = _RecordingOp(out_tensor)
        op_idx = _RecordingOp(
            (
                torch.zeros((16, 128), dtype=torch.int32),
                torch.zeros((4, 8), dtype=torch.int32),
                torch.zeros((4, 2), dtype=torch.int32),
            )
        )

        monkeypatch.setattr(ak, "_fp8_attention_qo_op", op_qo)
        monkeypatch.setattr(ak, "_attention_op", op_attn)
        monkeypatch.setattr(ak, "_fp8_attention_op", op_fp8)
        monkeypatch.setattr(ak, "_indexer_op", op_idx)

        swa_cache = SimpleNamespace(
            raw=torch.zeros((16, 128), dtype=torch.int32),
            raw_hi=None,
            shape=(2, 8),
        )
        main_cache = SimpleNamespace(
            raw=torch.zeros((16, 128), dtype=torch.int32),
            raw_hi=None,
            rope=torch.zeros((16, 32), dtype=torch.int32),
            rope_hi=None,
            shape=(2, 8),
            total_rows=16,
        )

        # 1. _as_int32 and _as_int32_1d direct checks
        t_i32 = torch.tensor([1, 2], dtype=torch.int32)
        t_i64_2d = torch.tensor([[1], [2]], dtype=torch.int64)
        assert ak._as_int32(t_i32) is t_i32
        assert ak._as_int32(t_i64_2d).dtype == torch.int32
        assert ak._as_int32_1d(t_i32) is t_i32
        conv_1d = ak._as_int32_1d(t_i64_2d)
        assert conv_1d.ndim == 1 and conv_1d.dtype == torch.int32

        # 2. _resolve_fp8_pages 1-bank and 2-bank
        ps1, p1, s1 = ak._resolve_fp8_pages(main_cache, t_i64_2d, 4)
        assert ps1 == 16 and p1.dtype == torch.int32 and s1.dtype == torch.int32

        main_2bank = SimpleNamespace(
            raw=torch.zeros((16, 128), dtype=torch.int32),
            raw_hi=torch.zeros((16, 128), dtype=torch.int32),
            shape=(2, 8),
            total_rows=32,
        )
        ps2, p2, s2 = ak._resolve_fp8_pages(main_2bank, t_i64_2d, 4)
        assert ps2 == 16 and p2.dtype == torch.int32 and s2.dtype == torch.int32

        # 3. Attention and indexer wrappers with already-int32 1D inputs
        #    (verifying zero tensor copies)
        swa_rec = torch.zeros((4, 8), dtype=torch.int32)
        main_rec = torch.zeros((4, 8), dtype=torch.int32)
        tp_rank_1d = torch.zeros((1,), dtype=torch.int32)
        valid_i32 = torch.ones((4,), dtype=torch.int32)
        wo_scale = torch.ones((4,), dtype=torch.float32)
        q_dummy = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
        sink_dummy = torch.zeros((2,), dtype=torch.float32)

        ak.sparse_fp8_attention_qo(
            q_dummy,
            q_dummy,
            q_dummy,
            wo_scale,
            q_dummy,
            wo_scale,
            valid_i32,
            q_dummy,
            swa_cache,
            main_cache,
            swa_rec,
            main_rec,
            sink_dummy,
            sink_dummy,
            tp_rank_1d,
            q_dummy,
            wo_scale,
            q_dummy,
            wo_scale,
            valid_i32,
            1.0,
            None,
            None,
            4,
            False,
            128,
        )
        assert op_qo.calls[-1][11] is swa_rec
        assert op_qo.calls[-1][12] is main_rec
        assert op_qo.calls[-1][17] is tp_rank_1d
        assert op_qo.calls[-1][22] is valid_i32

        ak.paged_attention(
            q_dummy,
            swa_cache,
            main_cache,
            swa_rec,
            main_rec,
            sink_dummy,
            1.0,
            tp_rank=tp_rank_1d,
        )
        assert op_attn.calls[-1][3] is swa_rec
        assert op_attn.calls[-1][4] is main_rec
        assert op_attn.calls[-1][7] is tp_rank_1d

        ak.sparse_fp8_attention(
            q_dummy,
            swa_cache,
            main_cache,
            swa_rec,
            main_rec,
            sink_dummy,
            1.0,
            tp_rank=tp_rank_1d,
        )
        assert op_fp8.calls[-1][4] is swa_rec
        assert op_fp8.calls[-1][5] is main_rec
        assert op_fp8.calls[-1][10] is tp_rank_1d

        # paged_indexer with int32 / 1D tensors
        lengths_1d = torch.ones((4,), dtype=torch.int32)
        cands_i32 = torch.zeros((4, 2), dtype=torch.int32)
        wrec_i32 = torch.zeros((4,), dtype=torch.int32)
        req_i32 = torch.zeros((4,), dtype=torch.int32)
        bmax_1d = torch.tensor([8], dtype=torch.int32)
        ak.paged_indexer(
            q_dummy,
            main_cache,
            main_rec,
            wo_scale,
            lengths_1d,
            cands_i32,
            wrec_i32,
            q_dummy[:, 0],
            req_i32,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            batch_max=bmax_1d,
        )
        idx_call = op_idx.calls[-1]
        assert idx_call[2] is main_rec
        assert idx_call[4] is lengths_1d
        assert idx_call[5] is cands_i32
        assert idx_call[6] is wrec_i32
        assert idx_call[8] is req_i32
        assert idx_call[9] is bmax_1d
