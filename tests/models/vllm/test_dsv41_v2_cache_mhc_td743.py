# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Wave 7 Track TD7.43:
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.cache`:
  `PagedCacheRows` static geometry precomputation (`elem_count`, `is_fp32`,
  `pages_aligned`, `page_scale`, `page_offset`), zero-overhead identity fast
  paths in `row_indices` and `pages`, and redundant dtype cast guards in
  `__getitem__`, `write`, `_write_fp8`, and `write_cache`.
- `vllm_torchtpu.models.vllm.deepseek_v41.common.mhc`:
  `_pre_mix_and_gates_jax` extraction and direct invocation from
  `delayed_pre_jax`, `_delayed_pre_flat_small`, and `delayed_post_pre_flat_jax`
  (`tile is None` decode/small-batch XLA path) to eliminate the dead 3D
  `flat.reshape(tokens, mult, dim)` and discarded 3D reduction.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_delayed_torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import ROW_WORDS, PagedCacheRows


@pytest.mark.cpu_test
class TestPagedCacheRowsTD743:
    def test_precomputed_attributes_and_pages_alignment(self):
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # Aligned BF16 layout: rows=4, page_stride=4 (scale=1), row_offset=0 (offset=0)
        layout_bf16 = ((16, 4, 256), torch.bfloat16, 2048, 0)
        paged_bf16 = PagedCacheRows(raw, layout_bf16)
        assert paged_bf16.elem_count == 256
        assert paged_bf16.is_fp32 is False
        assert paged_bf16.pages_aligned is True
        assert paged_bf16.page_scale == 1
        assert paged_bf16.page_offset == 0

        # Aligned FP32 layout with page_scale=2, page_offset=1
        layout_fp32 = ((7, 4, 128), torch.float32, 4096, 2048)
        paged_fp32 = PagedCacheRows(raw, layout_fp32)
        assert paged_fp32.elem_count == 128
        assert paged_fp32.is_fp32 is True
        assert paged_fp32.pages_aligned is True
        assert paged_fp32.page_scale == 2
        assert paged_fp32.page_offset == 1

        # Unaligned layout (row_offset=2 not divisible by rows=4)
        layout_unaligned = ((15, 4, 256), torch.bfloat16, 2048, 1024)
        paged_unaligned = PagedCacheRows(raw, layout_unaligned)
        assert paged_unaligned.pages_aligned is False
        assert paged_unaligned.page_scale == 0
        assert paged_unaligned.page_offset == 0
        assert paged_unaligned.pages(torch.tensor([0, 1], dtype=torch.int32)) is None

    def test_row_indices_and_pages_identity_and_scaled_branches(self):
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        slots = torch.tensor([0, 3, 7], dtype=torch.int32)
        block_table = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)

        # 1. Unsharded, page_stride == shape[1], row_offset == 0 -> returns input tensor
        paged_id = PagedCacheRows(raw, ((16, 4, 256), torch.bfloat16, 2048, 0))
        assert paged_id.row_indices(slots) is slots
        assert paged_id.pages(block_table) is block_table

        # 2. Unsharded, page_stride == shape[1], row_offset != 0 (aligned to rows=4)
        paged_off = PagedCacheRows(raw, ((15, 4, 256), torch.bfloat16, 2048, 2048))
        assert torch.equal(paged_off.row_indices(slots), slots + 4)
        assert torch.equal(paged_off.pages(block_table), block_table + 1)

        # 3. Unsharded, page_stride != shape[1], row_offset == 0 and row_offset != 0
        paged_strided_zero = PagedCacheRows(raw, ((8, 4, 256), torch.bfloat16, 4096, 0))
        expected_strided = (slots // 4) * 8 + (slots % 4)
        assert torch.equal(paged_strided_zero.row_indices(slots), expected_strided)
        assert torch.equal(paged_strided_zero.pages(block_table), block_table * 2)

        paged_strided_off = PagedCacheRows(
            raw, ((7, 4, 256), torch.bfloat16, 4096, 2048)
        )
        assert torch.equal(paged_strided_off.row_indices(slots), expected_strided + 4)
        assert torch.equal(paged_strided_off.pages(block_table), block_table * 2 + 1)

        # 4. Sharded: (scale=1, offset=0) vs (scale=2, offset=1)
        paged_sharded_id = PagedCacheRows(
            raw,
            ((8, 4, 256), torch.bfloat16, 2048, 0),
            block_offset=4,
            replicated_blocks=2,
            sharded=True,
        )
        sharded_slots = torch.tensor([1, 16, 40, -1], dtype=torch.int32)
        sharded_idx = paged_sharded_id.row_indices(sharded_slots)
        assert sharded_idx.tolist() == [1, 8, -1, -1]
        bt_sharded = torch.tensor([0, 4, 10, -1], dtype=torch.int32)
        assert paged_sharded_id.pages(bt_sharded).tolist() == [0, 2, 0, 0]

        paged_sharded_scaled = PagedCacheRows(
            raw,
            ((6, 4, 256), torch.bfloat16, 4096, 2048),
            block_offset=4,
            replicated_blocks=2,
            sharded=True,
        )
        assert paged_sharded_scaled.pages(bt_sharded).tolist() == [1, 5, 0, 0]

    def test_getitem_and_write_int32_vs_int64_and_dtype_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        raw = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        layout = ((8, 4, 256), torch.bfloat16, 2048, 0)
        paged = PagedCacheRows(raw, layout)

        # CPU write with row_offset == 0 (clamp(min=-1) branch) and row_offset != 0
        slots_i32 = torch.tensor([0, -5], dtype=torch.int32)
        vals_bf16 = torch.ones((2, 256), dtype=torch.bfloat16) * 2.0
        paged.write(slots_i32, vals_bf16)
        out_cpu = paged[torch.tensor([0]), torch.tensor([0])]
        assert torch.equal(out_cpu, vals_bf16[:1])

        # CPU write with values needing dtype conversion (float32 -> bfloat16)
        paged_off = PagedCacheRows(raw, ((7, 4, 256), torch.bfloat16, 2048, 2048))
        paged_off.write(slots_i32, torch.ones((2, 256), dtype=torch.float32) * 3.0)
        out_off = paged_off[torch.tensor([0]), torch.tensor([0])]
        assert torch.allclose(out_off, torch.full((1, 256), 3.0, dtype=torch.bfloat16))

        # CPU write with strided pages and sharded pages
        paged_strided = PagedCacheRows(raw, ((4, 4, 256), torch.bfloat16, 4096, 0))
        paged_strided.write(slots_i32, vals_bf16)
        paged_sharded = PagedCacheRows(
            raw, layout, block_offset=0, replicated_blocks=2, sharded=True
        )
        paged_sharded.write(slots_i32, vals_bf16)

        # Non-CPU __getitem__ and write: test both int32 and int64
        paged.raw = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), copy_=MagicMock()
        )
        read_calls = []

        def fake_read(r, rec, elem_count, is_fp32):
            read_calls.append((rec.dtype, elem_count, is_fp32))
            return torch.zeros((rec.numel(), elem_count), dtype=torch.bfloat16)

        monkeypatch.setattr(v2_cache, "_read_values_op", fake_read)
        _ = paged[
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        ]
        _ = paged[
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
        ]
        assert read_calls == [(torch.int32, 256, False), (torch.int32, 256, False)]

        # 2-bank non-CPU __getitem__
        raw_lo = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        raw_hi = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        paged_2b = PagedCacheRows((raw_lo, raw_hi), layout)
        paged_2b.raw = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), copy_=MagicMock()
        )
        paged_2b.raw_hi = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), copy_=MagicMock()
        )
        monkeypatch.setattr(
            v2_cache,
            "_read_values_2bank_op",
            lambda r0, r1, rec, ec, fp32: torch.zeros(
                (rec.numel(), ec), dtype=torch.bfloat16
            ),
        )
        _ = paged_2b[
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        ]

        # Non-CPU write: int32 vs int64 slots
        write_rec_dtypes = []
        monkeypatch.setattr(
            v2_cache,
            "_write_values_op",
            lambda r, rec, v: write_rec_dtypes.append(rec.dtype) or torch.zeros(1),
        )
        paged.write(torch.tensor([0, 1], dtype=torch.int32), vals_bf16)
        paged.write(torch.tensor([0, 1], dtype=torch.int64), vals_bf16)
        assert write_rec_dtypes == [torch.int32, torch.int32]

    def test_write_fp8_and_write_cache_dtype_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        raw = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        rope = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
        layout_fp8 = ((16, 1, 512), torch.uint8, 512, 0)
        paged_fp8 = PagedCacheRows(raw, layout_fp8, rope=rope)

        # 1. CPU _write_fp8 with both bfloat16 and float32 inputs
        slots_i32 = torch.tensor([0, 1], dtype=torch.int32)
        paged_fp8.write(slots_i32, torch.randn(2, 512, dtype=torch.bfloat16))
        paged_fp8.write(slots_i32, torch.randn(2, 512, dtype=torch.float32))

        # 2. Non-CPU _write_fp8 with int32/int64 slots and bf16/fp32 values
        paged_fp8.raw = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), copy_=MagicMock()
        )
        paged_fp8.rope = SimpleNamespace(copy_=MagicMock())
        encode_dtypes = []
        fp8_rec_dtypes = []
        monkeypatch.setattr(
            v2_cache,
            "_encode_fp8_op",
            lambda v: encode_dtypes.append(v.dtype)
            or (
                torch.zeros(2, ROW_WORDS, dtype=torch.int32),
                torch.zeros(2, v2_cache.ROPE_WORDS, dtype=torch.int32),
            ),
        )
        monkeypatch.setattr(
            v2_cache,
            "_write_values_op",
            lambda r, rec, w: fp8_rec_dtypes.append(rec.dtype) or torch.zeros(1),
        )
        paged_fp8.write(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.randn(2, 512, dtype=torch.bfloat16),
        )
        paged_fp8.write(
            torch.tensor([0, 1], dtype=torch.int64),
            torch.randn(2, 512, dtype=torch.float32),
        )
        assert encode_dtypes == [torch.bfloat16, torch.bfloat16]
        assert fp8_rec_dtypes == [torch.int32, torch.int32, torch.int32, torch.int32]

        # 3. write_cache non-CPU tensor path: int32 vs int64 slots
        write_op_dtypes = []
        monkeypatch.setattr(
            v2_cache,
            "_write_op",
            lambda c, s, v: write_op_dtypes.append(s.dtype) or torch.zeros(1),
        )
        fake_tpu_cache = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), copy_=MagicMock()
        )
        valid = torch.tensor([True, False])
        vals = torch.zeros(2, 8, dtype=torch.bfloat16)

        class _FakeSlotsTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

            def int(self):
                raise AssertionError(".int() must not be called when already int32")

        slots_no_cast = _FakeSlotsTensor(torch.tensor([0, 1], dtype=torch.int32))
        v2_cache.write_cache(fake_tpu_cache, slots_no_cast, vals, valid)
        v2_cache.write_cache(
            fake_tpu_cache, torch.tensor([0, 1], dtype=torch.int64), vals, valid
        )
        assert write_op_dtypes == [torch.int32, torch.int32]

    def test_sharded_kv_resolution_and_materialization(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import vllm.distributed as vllm_dist

        from vllm_torchtpu.core import owner_affine_block_pool as oabp

        monkeypatch.setattr(
            vllm_dist, "get_tensor_model_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(vllm_dist, "get_tensor_model_parallel_rank", lambda: 1)

        local_blocks, block_offset, sharded = v2_cache.resolve_kv_block_shard(
            10, kv_shard=True, replicated_blocks=2
        )
        assert (local_blocks, block_offset, sharded) == (6, 6, True)

        with pytest.raises(ValueError, match="KV sharding requires"):
            v2_cache.resolve_kv_block_shard(
                10, tp_size=2, tp_rank=0, kv_shard=True, replicated_blocks=1
            )

        monkeypatch.setattr(oabp, "worker_owner_layout", lambda vc, kc: "owner_layout")
        assert v2_cache.resolve_worker_owner_layout(None, None) == "owner_layout"

        raw = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        paged_6 = PagedCacheRows(raw, ((4, 4, 256), torch.bfloat16, 2048, 0, 2, True))
        paged_7 = PagedCacheRows(
            raw, ((6, 4, 256), torch.bfloat16, 2048, 0, 4, True, 2)
        )
        assert paged_6.sharded and paged_6.replicated_blocks == 0
        assert paged_7.sharded and paged_7.replicated_blocks == 2
        got_sharded = paged_7[torch.tensor([0, 20]), torch.tensor([0, 0])]
        assert got_sharded.shape == (2, 256)
        assert torch.all(got_sharded[1] == 0)

        fake_view = torch.zeros((4, 1, 4, 256), dtype=torch.bfloat16)
        monkeypatch.setattr(
            v2_cache,
            "allocate_upstream_kv_cache",
            lambda cfg, dev, lay: {"l0": fake_view},
        )
        import vllm_torchtpu.kv_cache_materializer as kv_mat

        monkeypatch.setattr(kv_mat, "layer_to_pool_index", lambda cfg: {"l0": 0})
        fake_cfg = SimpleNamespace(
            num_blocks=4,
            kv_cache_tensors=[SimpleNamespace(size=4 * 2048)],
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=2048))
            ],
        )
        fake_layout = SimpleNamespace(is_layer_compact=True)
        with pytest.raises(ValueError, match="Worker local_blocks"):
            v2_cache.materialize_upstream_kv_cache(
                fake_cfg,
                torch.device("cpu"),
                fake_layout,
                word_rows=True,
                kv_shard=True,
                tp_size=2,
                tp_rank=0,
                scheduler_blocks=4,
            )
        mat_rep0 = v2_cache.materialize_upstream_kv_cache(
            fake_cfg,
            torch.device("cpu"),
            fake_layout,
            word_rows=True,
            kv_shard=True,
            tp_size=2,
            tp_rank=0,
            replicated_blocks=0,
        )
        assert len(mat_rep0.native_cache_layouts["l0"]) == 6
        mat_rep1 = v2_cache.materialize_upstream_kv_cache(
            fake_cfg,
            torch.device("cpu"),
            fake_layout,
            word_rows=True,
            kv_shard=True,
            tp_size=2,
            tp_rank=0,
            replicated_blocks=2,
        )
        assert len(mat_rep1.native_cache_layouts["l0"]) == 7


@pytest.mark.cpu_test
class TestMhcPreMixAndGatesTD743:
    @pytest.mark.parametrize("packed", [False, True])
    def test_pre_mix_and_gates_jax_and_no_3d_residual_on_flat_xla_paths(
        self, monkeypatch: pytest.MonkeyPatch, packed: bool
    ):
        tokens, mult, dim = 16, 4, 256
        mixes = mult * (mult + 2)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 20)
        rng = np.random.default_rng(743)

        residual_t = torch.from_numpy(
            rng.standard_normal((tokens, mult, dim))
        ).bfloat16()
        fn_t = torch.from_numpy(0.02 * rng.standard_normal((mixes, mult * dim))).float()
        scale_t = torch.from_numpy(1.0 + 0.1 * rng.standard_normal(3)).float()
        base_t = torch.from_numpy(0.5 * rng.standard_normal(mixes)).float()
        prev_pre_t = torch.from_numpy(rng.uniform(0.0, 1.0, (tokens, mult))).float()
        x_t = torch.from_numpy(rng.standard_normal((tokens, dim))).bfloat16()
        post_mix_t = torch.from_numpy(rng.uniform(0.0, 2.0, (tokens, mult, 1))).float()
        comb_mix_t = torch.from_numpy(
            rng.uniform(0.0, 1.0, (tokens, mult, mult))
        ).float()
        comb_mix_t = comb_mix_t / comb_mix_t.sum(dim=-1, keepdim=True)

        def to_jax(t: torch.Tensor) -> jax.Array:
            if t.dtype == torch.bfloat16:
                return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
            return jnp.asarray(t.numpy())

        weight_t = mhc.pack_mix_weight(fn_t) if packed else fn_t
        res_j = to_jax(residual_t)
        flat_j = to_jax(residual_t.flatten(1))
        weight_j = to_jax(weight_t)
        scale_j = to_jax(scale_t)
        base_j = to_jax(base_t)
        prev_pre_j = to_jax(prev_pre_t)

        # 1. Verify delayed_pre_jax works and matches PyTorch reference
        post_3d, comb_3d, val_3d, next_pre_3d = mhc.delayed_pre_jax(
            res_j, flat_j, weight_j, scale_j, base_j, prev_pre_j, *constants
        )
        want_pre = mhc_pre_delayed_torch(
            residual_t, fn_t, scale_t, base_t, *constants, pre_mix=prev_pre_t
        )
        np.testing.assert_allclose(
            np.asarray(post_3d, dtype=np.float32),
            want_pre[0].float().numpy(),
            rtol=5e-3,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(val_3d.astype(jnp.float32)),
            want_pre[2].float().numpy(),
            rtol=2e-2,
            atol=2e-2,
        )

        # 2. Verify flat XLA paths NEVER invoke delayed_pre_jax and match 3D gates
        def forbid_delayed_pre_jax(*_args, **_kwargs):
            raise AssertionError(
                "Flat XLA path must call _pre_mix_and_gates_jax directly"
            )

        monkeypatch.setattr(mhc, "delayed_pre_jax", forbid_delayed_pre_jax)

        post_flat, comb_flat, val_flat, next_pre_flat = mhc.delayed_pre_flat_jax(
            flat_j, weight_j, scale_j, base_j, prev_pre_j, *constants
        )
        np.testing.assert_array_equal(np.asarray(post_flat), np.asarray(post_3d))
        np.testing.assert_array_equal(np.asarray(comb_flat), np.asarray(comb_3d))
        np.testing.assert_array_equal(
            np.asarray(next_pre_flat), np.asarray(next_pre_3d)
        )
        np.testing.assert_allclose(
            np.asarray(val_flat.astype(jnp.float32)),
            np.asarray(val_3d.astype(jnp.float32)),
            rtol=1e-5,
            atol=1e-5,
        )

        flat_new, post_pp, comb_pp, val_pp, next_pre_pp = mhc.delayed_post_pre_flat_jax(
            to_jax(x_t),
            flat_j,
            to_jax(post_mix_t),
            to_jax(comb_mix_t),
            weight_j,
            scale_j,
            base_j,
            prev_pre_j,
            *constants,
        )
        want_res = mhc_post_torch(x_t, residual_t, post_mix_t, comb_mix_t)
        want_pp = mhc_pre_delayed_torch(
            want_res, fn_t, scale_t, base_t, *constants, pre_mix=prev_pre_t
        )
        np.testing.assert_allclose(
            np.asarray(flat_new.reshape(tokens, mult, dim).astype(jnp.float32)),
            want_res.float().numpy(),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(post_pp, dtype=np.float32),
            want_pp[0].float().numpy(),
            rtol=5e-3,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(comb_pp, dtype=np.float32),
            want_pp[1].float().numpy(),
            rtol=5e-3,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(val_pp.astype(jnp.float32)),
            want_pp[2].float().numpy(),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(next_pre_pp, dtype=np.float32),
            want_pp[3].float().numpy(),
            rtol=5e-3,
            atol=1e-5,
        )
