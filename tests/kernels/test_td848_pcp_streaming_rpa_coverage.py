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
"""Complete statement and branch coverage tests for TD8.48 PCP streaming RPA modules."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as batched_rpa_wrapper,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    pcp_layout,
    reference,
    schedule,
    vllm_adapter,
    wrapper,
)


@pytest.fixture(autouse=True)
def _mock_tpu_info(monkeypatch):
    monkeypatch.setattr(
        batched_rpa_wrapper.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(num_lanes=128, num_sublanes=8),
    )


def _cpu_mesh(with_tp: bool = True) -> Mesh:
    devices = np.array(jax.devices()[:1])
    if with_tp:
        return Mesh(
            devices.reshape(1, 1), (wrapper.PCP_AXIS_NAME, wrapper.TP_AXIS_NAME)
        )
    return Mesh(devices.reshape(1), (wrapper.PCP_AXIS_NAME,))


class TestPcpLayout:
    def test_token_owner_start_offsets_and_resolve(self):
        reqs = np.array([2, 4], dtype=np.int64)
        np.testing.assert_array_equal(
            pcp_layout.pcp_token_owner_start_offsets(reqs, None),
            np.array([0, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            pcp_layout.pcp_token_owner_start_offsets(reqs, [1, 3]),
            np.array([1, 3], dtype=np.int64),
        )
        with pytest.raises(ValueError, match="same number of entries"):
            pcp_layout.pcp_token_owner_start_offsets(reqs, [1])
        with pytest.raises(ValueError, match="non-negative"):
            pcp_layout.pcp_token_owner_start_offsets(reqs, [0, -1])

        with pytest.raises(ValueError, match="legacy alias"):
            pcp_layout._resolve_token_owner_start_offsets(reqs, [0, 2], [0, 2])
        np.testing.assert_array_equal(
            pcp_layout._resolve_token_owner_start_offsets(reqs, None, [2, 6]),
            np.array([2, 6], dtype=np.int64),
        )

    def test_chunk_ranges_and_query_alias(self):
        assert list(pcp_layout.pcp_token_owner_chunk_ranges(0, 0, 0, 2, 2)) == []
        # chunk_start + interleave_size <= token_owner_start is True:
        assert list(pcp_layout.pcp_token_owner_chunk_ranges(4, 3, 0, 2, 2)) == [(4, 6)]
        # Exercise overlap_end <= overlap_start False branch via negative pcp_rank:
        assert list(pcp_layout.pcp_token_owner_chunk_ranges(2, 0, -3, 2, 2)) == []
        assert list(pcp_layout.pcp_query_chunk_ranges(4, 0, 1, 2, 2)) == [(2, 4)]

    def test_pcp_local_token_counts(self):
        np.testing.assert_array_equal(
            pcp_layout.pcp_local_token_counts([3, 5], pcp_size=1, interleave_size=2),
            np.array([8], dtype=np.int32),
        )
        with pytest.raises(ValueError, match="must be positive"):
            pcp_layout.pcp_local_token_counts([4], pcp_size=2, interleave_size=0)
        np.testing.assert_array_equal(
            pcp_layout.pcp_local_token_counts([], pcp_size=2, interleave_size=2),
            np.zeros(2, dtype=np.int32),
        )
        counts = pcp_layout.pcp_local_token_counts(
            [3, 5, 2],
            pcp_size=2,
            interleave_size=2,
            token_owner_start_offsets_per_req=[0, 3, 8],
        )
        np.testing.assert_array_equal(counts, np.array([6, 4], dtype=np.int32))

    def test_build_and_apply_rank_major_token_order_and_logits(self, monkeypatch):
        order_1, inv_1 = pcp_layout.build_pcp_rank_major_token_order(
            [2, 1], pcp_size=1, interleave_size=2, padded_num_tokens=4
        )
        np.testing.assert_array_equal(order_1, np.array([0, 1, 2, -1], dtype=np.int64))
        np.testing.assert_array_equal(inv_1, np.array([0, 1, 2], dtype=np.int64))

        with pytest.raises(ValueError, match="must be positive"):
            pcp_layout.build_pcp_rank_major_token_order(
                [2], pcp_size=2, interleave_size=0, padded_num_tokens=4
            )
        with pytest.raises(ValueError, match="must be divisible"):
            pcp_layout.build_pcp_rank_major_token_order(
                [2], pcp_size=2, interleave_size=2, padded_num_tokens=3
            )

        # Empty requests -> total_num_tokens == 0
        empty_order, empty_inv = pcp_layout.build_pcp_rank_major_token_order(
            [], pcp_size=2, interleave_size=2, padded_num_tokens=4
        )
        np.testing.assert_array_equal(empty_order, np.full(4, -1, dtype=np.int64))
        assert empty_inv.size == 0

        # Exceeds padded local capacity
        with pytest.raises(ValueError, match="exceeds padded local capacity"):
            pcp_layout.build_pcp_rank_major_token_order(
                [4], pcp_size=2, interleave_size=2, padded_num_tokens=2
            )

        # Request with 0 tokens alongside positive tokens, with and without mrope
        input_ids = np.array([10, 11, 12, 13, 99, 99], dtype=np.int32)
        positions = np.array([0, 1, 2, 3, 99, 99], dtype=np.int32)
        mrope = np.array([[0, 1, 2, 3, 9, 9], [10, 11, 12, 13, 9, 9]], dtype=np.int32)
        inv = pcp_layout.apply_pcp_rank_major_token_order(
            input_ids,
            positions,
            [0, 4],
            pcp_size=2,
            interleave_size=1,
            padded_num_tokens=6,
            mrope_positions_cpu=mrope,
            token_owner_start_offsets_per_req=[0, 0],
        )
        np.testing.assert_array_equal(inv, np.array([0, 3, 1, 4], dtype=np.int64))
        np.testing.assert_array_equal(
            input_ids, np.array([10, 12, 0, 11, 13, 0], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            mrope[0], np.array([0, 2, 0, 1, 3, 0], dtype=np.int32)
        )

        # Without mrope
        input_ids_no_mrope = np.array([10, 11, 12, 13], dtype=np.int32)
        positions_no_mrope = np.array([0, 1, 2, 3], dtype=np.int32)
        pcp_layout.apply_pcp_rank_major_token_order(
            input_ids_no_mrope,
            positions_no_mrope,
            [4],
            pcp_size=2,
            interleave_size=1,
            padded_num_tokens=4,
            mrope_positions_cpu=None,
        )

        # Logits indices: empty and non-empty
        assert (
            pcp_layout.build_pcp_logits_indices(
                [], pcp_size=2, interleave_size=1, padded_num_tokens=4
            ).size
            == 0
        )
        logits_idx = pcp_layout.build_pcp_logits_indices(
            [2, 2],
            pcp_size=2,
            interleave_size=1,
            padded_num_tokens=4,
            token_offset=10,
        )
        np.testing.assert_array_equal(logits_idx, np.array([12, 13], dtype=np.int64))

        # Uncovered tokens error branch
        monkeypatch.setattr(
            pcp_layout, "pcp_token_owner_chunk_ranges", lambda *args, **kwargs: iter(())
        )
        with pytest.raises(ValueError, match="did not cover all tokens"):
            pcp_layout.build_pcp_rank_major_token_order(
                [2], pcp_size=2, interleave_size=1, padded_num_tokens=4
            )


class TestScheduleAndReference:
    def test_schedule_helpers_and_validation(self):
        pos = schedule.query_row_positions(
            jnp.asarray(4, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
            pcp_size=2,
            interleave_size=2,
        )
        np.testing.assert_array_equal(
            np.asarray(pos), np.array([4, 7, 8], dtype=np.int32)
        )

        with pytest.raises(ValueError, match="interleave_size must be positive"):
            schedule._validate_page_interleave_size(4, 0, "ctx")
        with pytest.raises(NotImplementedError, match="divisible by interleave_size"):
            schedule._validate_page_interleave_size(4, 3, "ctx")

        with pytest.raises(ValueError, match="at least one sequence slot"):
            schedule._reshape_metadata_page_indices_jax(
                jnp.zeros((4,), dtype=jnp.int32), 0
            )
        with pytest.raises(ValueError, match="divisible by kv_lens"):
            schedule._reshape_metadata_page_indices_jax(
                jnp.zeros((5,), dtype=jnp.int32), 2
            )
        reshaped_1d = schedule._reshape_metadata_page_indices_jax(
            jnp.arange(6, dtype=jnp.int32), 2
        )
        assert reshaped_1d.shape == (2, 3)

        with pytest.raises(ValueError, match="first dimension must match"):
            schedule._reshape_metadata_page_indices_jax(
                jnp.zeros((3, 2), dtype=jnp.int32), 2
            )
        reshaped_2d = schedule._reshape_metadata_page_indices_jax(
            jnp.zeros((2, 3), dtype=jnp.int32), 2
        )
        assert reshaped_2d.shape == (2, 3)

        with pytest.raises(ValueError, match="rank 1 or rank 2"):
            schedule._reshape_metadata_page_indices_jax(
                jnp.zeros((1, 2, 3), dtype=jnp.int32), 1
            )

    def test_build_schedule_inputs_validation_and_execution(self):
        kv_lens = jnp.array([6, 4], dtype=jnp.int32)
        page_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 4, 6], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

        with pytest.raises(NotImplementedError, match="num_lanes == 1"):
            schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                global_bucket_tokens=8,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=2,
                interleave_size=1,
                q_block_size=2,
                num_lanes=2,
            )
        with pytest.raises(NotImplementedError, match="q_block_size to be a multiple"):
            schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                global_bucket_tokens=12,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=2,
                interleave_size=2,
                q_block_size=3,
            )
        with pytest.raises(ValueError, match="positive and divisible by pcp_size"):
            schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                global_bucket_tokens=0,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=2,
                interleave_size=1,
                q_block_size=2,
            )
        with pytest.raises(ValueError, match="pcp_size \\* q_block_size"):
            schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                global_bucket_tokens=6,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=2,
                interleave_size=1,
                q_block_size=2,
            )
        with pytest.raises(
            ValueError, match="local_kv_cache_num_blocks must be positive"
        ):
            schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                global_bucket_tokens=8,
                local_kv_cache_num_blocks=0,
                page_size=2,
                pcp_size=2,
                interleave_size=1,
                q_block_size=2,
            )

        outputs = schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            global_bucket_tokens=8,
            local_kv_cache_num_blocks=4,
            page_size=2,
            pcp_size=2,
            interleave_size=1,
            q_block_size=2,
        )
        assert len(outputs) == 8
        tile_plan, block_tables = outputs[0], outputs[1]
        ref = reference.build_runtime_schedule_reference(
            np.asarray(tile_plan),
            np.asarray(block_tables),
            pcp_size=2,
            page_size=2,
            interleave_size=1,
        )
        assert ref.current_rows.shape[0] == tile_plan.shape[0]

    def test_reference_validation_and_all_branches(self):
        block_tables = np.array([[0, 1], [2, 3]], dtype=np.int32)
        with pytest.raises(ValueError, match="shape \\[tiles, 1, fields\\]"):
            reference.build_runtime_schedule_reference(
                np.zeros((2, 2, 24), dtype=np.int32),
                block_tables,
                pcp_size=2,
                page_size=2,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="rank 2"):
            reference.build_runtime_schedule_reference(
                np.zeros((1, 1, 24), dtype=np.int32),
                np.zeros((2,), dtype=np.int32),
                pcp_size=2,
                page_size=2,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="cover all PCP rank fields"):
            reference.build_runtime_schedule_reference(
                np.zeros((1, 1, 10), dtype=np.int32),
                block_tables,
                pcp_size=2,
                page_size=2,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="must be positive"):
            reference.build_runtime_schedule_reference(
                np.zeros((1, 1, 24), dtype=np.int32),
                block_tables,
                pcp_size=2,
                page_size=0,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="divisible by interleave_size"):
            reference.build_runtime_schedule_reference(
                np.zeros((1, 1, 24), dtype=np.int32),
                block_tables,
                pcp_size=2,
                page_size=2,
                interleave_size=3,
            )

        # Exercise _local_page_valid_len full vs partial chunks and kv_len <= base
        assert (
            reference._local_page_valid_len(
                100, 0, 0, page_size=4, pcp_size=2, interleave_size=2
            )
            == 4
        )
        assert (
            reference._local_page_valid_len(
                1, 0, 0, page_size=4, pcp_size=2, interleave_size=2
            )
            == 1
        )
        assert (
            reference._local_page_valid_len(
                0, 1, 0, page_size=4, pcp_size=2, interleave_size=2
            )
            == 0
        )

        # Craft a 2-rank tile plan with history boundary + dual-rank capture
        # and an inactive tile 1.
        pcp_size = 2
        F = schedule.TilePlanField
        plan_2pcp = np.zeros((2, pcp_size, F.NUM_FIELDS), dtype=np.int32)
        plan_2pcp[0, :, F.REQ_ID] = 0
        plan_2pcp[0, :, F.TOKEN_OWNER_START] = 0  # full chunk overlap of 2
        plan_2pcp[0, :, F.REQUEST_ABSOLUTE_QUERY_START] = (
            5  # unaligned with interleave=2 and virtual_page=8
        )
        plan_2pcp[0, :, F.CURRENT_EFFECTIVE_LEN] = 4
        plan_2pcp[0, :, F.CAPTURE_CURRENT_KV] = 1
        plan_2pcp[0, :, F.CURRENT_NUM_GROUPS] = 2
        plan_2pcp[0, :, F.HISTORY_NUM_GROUPS] = 2
        ref_2 = reference.build_runtime_schedule_reference(
            plan_2pcp.reshape(2, 1, pcp_size * F.NUM_FIELDS),
            block_tables,
            pcp_size=2,
            page_size=4,
            interleave_size=2,
        )
        assert ref_2.current_rows.shape[0] == 2
        assert ref_2.history_rows.shape[0] == 2

        # Craft a 1-rank tile plan where both owns_0 and owns_1 are True
        plan_1pcp = np.zeros((1, 1, F.NUM_FIELDS), dtype=np.int32)
        plan_1pcp[0, 0, F.REQ_ID] = 0
        plan_1pcp[0, 0, F.TOKEN_OWNER_START] = 0
        plan_1pcp[0, 0, F.REQUEST_ABSOLUTE_QUERY_START] = 1
        plan_1pcp[0, 0, F.CURRENT_EFFECTIVE_LEN] = 4
        plan_1pcp[0, 0, F.CAPTURE_CURRENT_KV] = 1
        plan_1pcp[0, 0, F.CURRENT_NUM_GROUPS] = 4  # group 3 has fresh_valid_len == 0
        plan_1pcp[0, 0, F.HISTORY_NUM_GROUPS] = 0
        ref_1 = reference.build_runtime_schedule_reference(
            plan_1pcp.reshape(1, 1, F.NUM_FIELDS),
            block_tables,
            pcp_size=1,
            page_size=4,
            interleave_size=2,
        )
        assert (
            ref_1.current_rows[0, 1, 0, 0, schedule.RuntimeScheduleField.CAPTURE_LEN]
            == 2
        )


class TestVllmAdapter:
    def test_build_callable_and_jax_op_modes(self, monkeypatch):
        mesh = _cpu_mesh(with_tp=True)

        def dummy_fn(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
            return x, y

        class _FakeJaxCallable:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def exported(self, *args, **kwargs):
                return SimpleNamespace(
                    out_avals=args,
                    out_tree=SimpleNamespace(unflatten=lambda it: tuple(it)),
                )

        monkeypatch.setattr(vllm_adapter.pallas_impl, "JaxCallable", _FakeJaxCallable)
        monkeypatch.setattr(
            vllm_adapter.pallas_impl,
            "jax_placeholders",
            lambda args, mesh, partition_specs: args,
        )
        monkeypatch.setattr(
            vllm_adapter.pallas_impl,
            "torch_placeholder",
            lambda aval, mesh, partition_spec: aval,
        )

        class _FakeCustomOp:
            def __init__(self, name, op_target, mutates_args=()):
                self.name = name
                self.op_target = op_target
                self.fake_fn = None

            def register_fake(self, fn):
                self.fake_fn = fn

        monkeypatch.setattr(torch.library, "custom_op", _FakeCustomOp)

        callable_obj = vllm_adapter.build_pcp_streaming_callable(
            "test_pcp_callable",
            dummy_fn,
            donate_argnums=(0,),
            mesh=mesh,
            input_partition_specs=(P(), P()),
            output_partition_specs=(P(), P()),
        )
        assert callable_obj.name == "test_pcp_callable"

        # Register custom op with TPU_KERNEL_ITER_MODE = False
        monkeypatch.setattr(envs, "TPU_KERNEL_ITER_MODE", False)
        op_normal = vllm_adapter.pcp_streaming_jax_op(
            "vllm_td848_test::pcp_op_normal",
            dummy_fn,
            donate_argnums=None,
            mesh=mesh,
            input_partition_specs=(P(), P()),
            output_partition_specs=(P(), P()),
        )
        out_x, out_y = op_normal.fake_fn("x", "y")
        assert (out_x, out_y) == ("x", "y")

        # Register custom op with TPU_KERNEL_ITER_MODE = True
        monkeypatch.setattr(envs, "TPU_KERNEL_ITER_MODE", True)
        op_iter = vllm_adapter.pcp_streaming_jax_op(
            "vllm_td848_test::pcp_op_iter",
            dummy_fn,
            donate_argnums=(0,),
            mesh=mesh,
            input_partition_specs=(P(), P()),
            output_partition_specs=None,
        )
        assert op_iter.fake_fn is not None

    def test_mesh_make_kernel_and_invoke_op(self, monkeypatch):
        mesh = _cpu_mesh(with_tp=True)
        monkeypatch.setattr(
            vllm_adapter,
            "get_or_create_pcp_mesh",
            lambda axis_name, tp_axis_name: (axis_name, tp_axis_name),
        )
        assert vllm_adapter.get_pcp_streaming_mesh() == (
            wrapper.PCP_AXIS_NAME,
            wrapper.TP_AXIS_NAME,
        )

        with pytest.raises(NotImplementedError, match="soft cap"):
            vllm_adapter.make_pcp_streaming_rpa_kernel(
                q_scale=None,
                k_scale=None,
                v_scale=None,
                mesh=mesh,
                sliding_window=None,
                soft_cap=1.0,
                skip_kv_update=False,
                cp_kv_cache_interleave_size=1,
            )
        with pytest.raises(NotImplementedError, match="skip_kv_update"):
            vllm_adapter.make_pcp_streaming_rpa_kernel(
                q_scale=None,
                k_scale=None,
                v_scale=None,
                mesh=mesh,
                sliding_window=None,
                soft_cap=None,
                skip_kv_update=True,
                cp_kv_cache_interleave_size=1,
            )

        monkeypatch.setattr(
            vllm_adapter,
            "sharded_pcp_ragged_paged_attention",
            lambda **kwargs: (kwargs["q"] + 1, kwargs["kv_cache"] + 2),
        )
        kernel_flat = vllm_adapter.make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=None,
            v_scale=None,
            mesh=mesh,
            sliding_window=None,
            skip_kv_update=False,
            cp_kv_cache_interleave_size=1,
            block_major=False,
        )
        kv_flat = jnp.zeros((4, 2, 1, 2, 128), dtype=jnp.float32)
        q = jnp.zeros((4, 2, 128), dtype=jnp.float32)
        new_kv, out = kernel_flat(kv_flat, q, q, q, q, q, q, q)
        assert new_kv.shape == kv_flat.shape
        assert out.shape == q.shape

        kernel_bm = vllm_adapter.make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=None,
            v_scale=None,
            mesh=mesh,
            sliding_window=None,
            skip_kv_update=False,
            cp_kv_cache_interleave_size=1,
            block_major=True,
        )
        kv_bm = jnp.zeros((2, 2, 2, 1, 2, 128), dtype=jnp.float32)
        new_kv_bm, out_bm = kernel_bm(kv_bm, q, q, q, q, q, q, q)
        assert new_kv_bm.shape == kv_bm.shape
        assert out_bm.shape == q.shape

        with pytest.raises(ValueError, match="keyword tensor args"):
            vllm_adapter.invoke_pcp_streaming_op(lambda *a: a, kv_flat, (), {"bad": 1})
        with pytest.raises(ValueError, match="tensor args after kv_cache"):
            vllm_adapter.invoke_pcp_streaming_op(lambda *a: a, kv_flat, (1, 2, 3), {})
        with pytest.raises(NotImplementedError, match="attention sinks"):
            vllm_adapter.invoke_pcp_streaming_op(
                lambda *a: a, kv_flat, (1, 2, 3, 4, 5, 6, 7, "sink"), {}
            )
        res = vllm_adapter.invoke_pcp_streaming_op(
            lambda *a: a, "kv", (1, 2, 3, 4, 5, 6, 7, None), {}
        )
        assert res == ("kv", 1, 2, 3, 4, 5, 6, 7)


class TestWrapper:
    def test_pack_and_prepare_kv_and_reshape_cache(self, monkeypatch):
        k = jnp.ones((4, 1, 128), dtype=jnp.float32)
        v = jnp.ones((4, 1, 128), dtype=jnp.float32)

        with pytest.raises(ValueError, match="requires page_size > 0"):
            wrapper._pack_kv_for_cache(
                k, v, jnp.float32, kv_layout=KVLayout.SEQ_ALONG_LANE, page_size=0
            )
        packed_seq = wrapper._pack_kv_for_cache(
            k, v, jnp.float32, kv_layout=KVLayout.SEQ_ALONG_LANE, page_size=128
        )
        assert packed_seq.shape[-1] == 128

        with pytest.raises(ValueError, match="must be 5D"):
            wrapper._kv_cache_page_size(
                jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                KVLayout.HEAD_ALONG_SUBLANE,
            )

        # SEQ_ALONG_LANE _prepare_packed_kv_for_cache
        cache_seq = jnp.zeros((2, *packed_seq.shape[:-1], 128), dtype=jnp.float32)
        prepared_seq = wrapper._prepare_packed_kv_for_cache(
            k, v, cache_seq, kv_layout=KVLayout.SEQ_ALONG_LANE
        )
        assert prepared_seq.shape == packed_seq.shape

        bad_cache_seq = jnp.zeros((2, 99, 1, 1, 128), dtype=jnp.float32)
        with pytest.raises(ValueError, match="fresh KV prefix must match"):
            wrapper._prepare_packed_kv_for_cache(
                k, v, bad_cache_seq, kv_layout=KVLayout.SEQ_ALONG_LANE
            )

        # Trigger packed_kv.shape[-1] < k.shape[0] error branch
        orig_pack = wrapper._pack_kv_for_cache
        monkeypatch.setattr(
            wrapper,
            "_pack_kv_for_cache",
            lambda *args, **kwargs: jnp.zeros(
                (*cache_seq.shape[1:-1], 2), dtype=jnp.float32
            ),
        )
        with pytest.raises(ValueError, match="token extent must cover"):
            wrapper._prepare_packed_kv_for_cache(
                k, v, cache_seq, kv_layout=KVLayout.SEQ_ALONG_LANE
            )
        monkeypatch.setattr(wrapper, "_pack_kv_for_cache", orig_pack)

        # HEAD_ALONG_SUBLANE _prepare_packed_kv_for_cache: exact, reshape, error
        packed_head = wrapper._pack_kv_for_cache(
            k, v, jnp.float32, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
        )
        cache_exact = jnp.zeros((2, 8, *packed_head.shape[1:]), dtype=jnp.float32)
        res_exact = wrapper._prepare_packed_kv_for_cache(
            k, v, cache_exact, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
        )
        assert res_exact.shape == packed_head.shape

        tail_prod = math.prod(packed_head.shape[1:])
        cache_compat = jnp.zeros((2, 8, 1, 1, tail_prod), dtype=jnp.float32)
        res_compat = wrapper._prepare_packed_kv_for_cache(
            k, v, cache_compat, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
        )
        assert res_compat.shape == (4, 1, 1, tail_prod)

        cache_incompat = jnp.zeros((2, 8, 1, 1, tail_prod + 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="incompatible with KV cache layout"):
            wrapper._prepare_packed_kv_for_cache(
                k, v, cache_incompat, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
            )

        # _reshape_packed_kv_cache_for_attention
        valid_attn_cache = jnp.zeros((2, 8, 1, 1, 128), dtype=jnp.float32)
        assert (
            wrapper._reshape_packed_kv_cache_for_attention(valid_attn_cache)
            is valid_attn_cache
        )
        bad_attn_cache = jnp.zeros((2, 8, 1, 2, 128), dtype=jnp.float32)
        with pytest.raises(ValueError, match="packing does not match"):
            wrapper._reshape_packed_kv_cache_for_attention(bad_attn_cache)

    def test_local_mapping_and_row_mappings(self):
        with pytest.raises(ValueError, match="Replicated PCP tensor axis"):
            wrapper._select_replicated_shard_for_pcp_rank(
                jnp.zeros((2, 3), dtype=jnp.int32),
                wrapper.PCP_AXIS_NAME,
                2,
                axis=-1,
            )

        block_tables = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        pos = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        req_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        with pytest.raises(ValueError, match="positive block_size"):
            wrapper.compute_pcp_local_mapping(pos, req_idx, block_tables, 0, 2, 0)
        with pytest.raises(ValueError, match="positive cp_size"):
            wrapper.compute_pcp_local_mapping(pos, req_idx, block_tables, 2, 0, 0)
        with pytest.raises(ValueError, match="cp_rank in"):
            wrapper.compute_pcp_local_mapping(pos, req_idx, block_tables, 2, 2, 2)
        with pytest.raises(ValueError, match="positive interleave_size"):
            wrapper.compute_pcp_local_mapping(pos, req_idx, block_tables, 2, 2, 0, 0)

        is_local, slot_ids = wrapper.compute_pcp_local_mapping(
            pos, req_idx, block_tables, 2, 2, 0, 1
        )
        np.testing.assert_array_equal(
            np.asarray(is_local), np.array([True, False, True, False])
        )
        np.testing.assert_array_equal(
            np.asarray(slot_ids), np.array([0, -1, 5, -1], dtype=np.int32)
        )

        valid = jnp.array([True, True, False, False])
        with pytest.raises(
            ValueError, match="local_kv_cache_num_blocks must be positive"
        ):
            wrapper._compute_cache_owner_and_slot_ids_for_rows(
                pos,
                req_idx,
                valid,
                block_tables,
                local_kv_cache_num_blocks=0,
                page_size=2,
                pcp_size=2,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="page_size must be positive"):
            wrapper._compute_cache_owner_and_slot_ids_for_rows(
                pos,
                req_idx,
                valid,
                block_tables,
                local_kv_cache_num_blocks=4,
                page_size=0,
                pcp_size=2,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="pcp_size must be positive"):
            wrapper._compute_cache_owner_and_slot_ids_for_rows(
                pos,
                req_idx,
                valid,
                block_tables,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=0,
                interleave_size=1,
            )
        with pytest.raises(ValueError, match="interleave_size must be positive"):
            wrapper._compute_cache_owner_and_slot_ids_for_rows(
                pos,
                req_idx,
                valid,
                block_tables,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=2,
                interleave_size=0,
            )
        with pytest.raises(NotImplementedError, match="divisible by interleave_size"):
            wrapper._compute_cache_owner_and_slot_ids_for_rows(
                pos,
                req_idx,
                valid,
                block_tables,
                local_kv_cache_num_blocks=4,
                page_size=2,
                pcp_size=2,
                interleave_size=3,
            )

        # _compute_pcp_local_query_row_mapping_from_metadata validations
        kv_lens = jnp.array([4, 2], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)
        for bad_kwargs, match in [
            (
                {"local_padded_tokens": 0, "pcp_size": 2, "interleave_size": 1},
                "local_padded_tokens",
            ),
            (
                {"local_padded_tokens": 2, "pcp_size": 0, "interleave_size": 1},
                "pcp_size",
            ),
            (
                {"local_padded_tokens": 2, "pcp_size": 2, "interleave_size": 0},
                "interleave_size",
            ),
        ]:
            with pytest.raises(ValueError, match=match):
                wrapper._compute_pcp_local_query_row_mapping_from_metadata(
                    kv_lens, cu_q_lens, dist, pcp_rank=0, **bad_kwargs
                )

        # _build_writeback_segment_descriptors validation and SEQ_ALONG_LANE path
        compact_slots = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        with pytest.raises(ValueError, match="require page_size > 0"):
            wrapper._build_writeback_segment_descriptors(
                compact_slots,
                jnp.asarray(4, dtype=jnp.int32),
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                page_size=0,
            )
        desc_seq, num_seg_seq = wrapper._build_writeback_segment_descriptors(
            compact_slots,
            jnp.asarray(4, dtype=jnp.int32),
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            page_size=2,
        )
        assert int(num_seg_seq) == 2
        assert desc_seq.shape == (3, 4)

    def test_sharded_pcp_ragged_paged_attention_validation_and_execution(
        self, monkeypatch
    ):
        mesh_tp = _cpu_mesh(with_tp=True)
        mesh_no_tp = _cpu_mesh(with_tp=False)
        mesh_bad = Mesh(np.array(jax.devices()[:1]), ("tp",))

        q = jnp.ones((4, 2, 128), dtype=jnp.float32)
        k = jnp.ones((4, 1, 128), dtype=jnp.float32)
        v = jnp.ones((4, 1, 128), dtype=jnp.float32)
        packed_kv = wrapper._pack_kv_for_cache(k, v, jnp.float32)
        kv_cache = jnp.zeros((4, 2, *packed_kv.shape[1:]), dtype=jnp.float32)
        kv_lens = jnp.array([2, 2], dtype=jnp.int32)
        page_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)

        def _call(**overrides):
            kwargs = dict(
                mesh=mesh_tp,
                q=q,
                k=k,
                v=v,
                kv_cache=kv_cache,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=dist,
                attention_sink=None,
                sm_scale=1.0,
                cp_kv_cache_interleave_size=1,
                q_block_size=2,
            )
            kwargs.update(overrides)
            return wrapper.sharded_pcp_ragged_paged_attention(**kwargs)

        with pytest.raises(NotImplementedError, match="attention sinks"):
            _call(attention_sink=jnp.zeros((1,), dtype=jnp.float32))
        with pytest.raises(NotImplementedError, match="head_dim==64"):
            _call(q=jnp.ones((4, 2, 64), dtype=jnp.float32))
        with pytest.raises(ValueError, match="cp_kv_cache_interleave_size > 0"):
            _call(cp_kv_cache_interleave_size=0)
        with pytest.raises(NotImplementedError, match="mesh with a pcp axis"):
            _call(mesh=mesh_bad)
        with pytest.raises(NotImplementedError, match="full attention only"):
            _call(attention_chunk_size=128)
        with pytest.raises(NotImplementedError, match="quantized Q scales"):
            _call(q_scale=1.0)
        with pytest.raises(ValueError, match="both None or both non-None"):
            _call(k_scale=1.0, v_scale=None)
        with pytest.raises(NotImplementedError, match="update_kv_cache=True"):
            _call(update_kv_cache=False)
        with pytest.raises(NotImplementedError, match="multiples of 128"):
            _call(kv_layout=KVLayout.SEQ_ALONG_LANE, cp_kv_cache_interleave_size=1)
        with pytest.raises(
            NotImplementedError, match="divisible by cp_kv_cache_interleave_size"
        ):
            _call(cp_kv_cache_interleave_size=3)

        mesh_2pcp = SimpleNamespace(
            axis_names=(wrapper.PCP_AXIS_NAME,), shape={wrapper.PCP_AXIS_NAME: 2}
        )
        with pytest.raises(ValueError, match="q tokens to be evenly sharded"):
            _call(mesh=mesh_2pcp, q=jnp.ones((3, 2, 128), dtype=jnp.float32))
        with pytest.raises(ValueError, match="KV cache blocks to be evenly sharded"):
            _call(
                mesh=mesh_2pcp,
                kv_cache=jnp.zeros((3, 2, *packed_kv.shape[1:]), dtype=jnp.float32),
            )

        # Mock kernel entry to run shard_map on CPU
        monkeypatch.setattr(
            wrapper,
            "pcp_streaming_attention_page_groups_packed_local_from_metadata",
            lambda q_streaming, packed_current_kv, attention_kv_cache, *a, **kw: (
                jnp.ones_like(q_streaming),
                attention_kv_cache,
            ),
        )

        # 1) With TP axis and float32 (q_per_kv_padded == q_per_kv)
        out_tp, cache_tp = _call(mesh=mesh_tp)
        assert out_tp.shape == q.shape
        assert cache_tp.shape == kv_cache.shape

        # 2) Without TP axis and bfloat16 with q_per_kv=1 (q_per_kv_padded != q_per_kv)
        q_bf16 = jnp.ones((4, 1, 128), dtype=jnp.bfloat16)
        k_bf16 = jnp.ones((4, 1, 128), dtype=jnp.bfloat16)
        v_bf16 = jnp.ones((4, 1, 128), dtype=jnp.bfloat16)
        packed_bf16 = wrapper._pack_kv_for_cache(k_bf16, v_bf16, jnp.bfloat16)
        cache_bf16 = jnp.zeros((4, 2, *packed_bf16.shape[1:]), dtype=jnp.bfloat16)
        out_no_tp, _ = _call(
            mesh=mesh_no_tp,
            q=q_bf16,
            k=k_bf16,
            v=v_bf16,
            kv_cache=cache_bf16,
        )
        assert out_no_tp.shape == q_bf16.shape

        # 3) Q heads not divisible by KV heads inside _pcp_ragged_paged_attention
        k_2h = jnp.ones((4, 2, 128), dtype=jnp.float32)
        v_2h = jnp.ones((4, 2, 128), dtype=jnp.float32)
        packed_2h = wrapper._pack_kv_for_cache(k_2h, v_2h, jnp.float32)
        cache_2h = jnp.zeros((4, 2, *packed_2h.shape[1:]), dtype=jnp.float32)
        with pytest.raises(ValueError, match="Q heads must be divisible by KV heads"):
            _call(
                q=jnp.ones((4, 3, 128), dtype=jnp.float32),
                k=k_2h,
                v=v_2h,
                kv_cache=cache_2h,
            )

    def test_td8316_schedule_and_wrapper_int32_cumsum(self, monkeypatch):
        """TD8.316: Verify all jnp.cumsum calls in pcp_streaming_rpa/schedule.py
        and pcp_streaming_rpa/wrapper.py pass explicit dtype=jnp.int32, and
        _build_writeback_segment_descriptors accumulates bool segment_starts
        directly without an intermediate .astype(jnp.int32)."""
        cumsum_calls: list[tuple[object, object]] = []
        orig_cumsum = jnp.cumsum

        def _spy_cumsum(a, axis=None, dtype=None, out=None):
            cumsum_calls.append((a.dtype, dtype))
            return orig_cumsum(a, axis=axis, dtype=dtype, out=out)

        monkeypatch.setattr(schedule.jnp, "cumsum", _spy_cumsum)
        monkeypatch.setattr(wrapper.jnp, "cumsum", _spy_cumsum)

        kv_lens = jnp.array([6, 4], dtype=jnp.int32)
        page_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 4, 6], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

        schedule.build_pcp_streaming_schedule_inputs_from_metadata_jax(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            global_bucket_tokens=8,
            local_kv_cache_num_blocks=4,
            page_size=2,
            pcp_size=2,
            interleave_size=1,
            q_block_size=2,
        )
        assert len(cumsum_calls) == 4
        assert all(dt == jnp.int32 for _, dt in cumsum_calls)

        cumsum_calls.clear()
        query_rows = wrapper._compute_pcp_local_query_row_mapping(
            jnp.array([2, 2], dtype=jnp.int32),
            token_owner_starts=jnp.array([0, 2], dtype=jnp.int32),
            request_absolute_starts=jnp.array([0, 0], dtype=jnp.int32),
            local_padded_tokens=4,
            pcp_size=2,
            pcp_rank=0,
            interleave_size=1,
        )
        wb_mapping = wrapper._compute_pcp_kv_writeback_mapping(
            query_rows,
            page_indices,
            local_kv_cache_num_blocks=4,
            page_size=2,
            pcp_size=2,
            interleave_size=1,
        )
        wrapper._build_writeback_segment_descriptors(
            jnp.array([0, 1, 3, -1], dtype=jnp.int32),
            wb_mapping.writeback_counts[0],
        )
        assert len(cumsum_calls) == 3
        assert all(dt == jnp.int32 for _, dt in cumsum_calls)
        assert cumsum_calls[2] == (jnp.bool_, jnp.int32)
