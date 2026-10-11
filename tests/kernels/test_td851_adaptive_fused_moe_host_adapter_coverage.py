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
"""100% statement and branch coverage tests for TD8.51 adaptive_fused_moe host,
router_ops, token_parallel, and vllm_adapter modules."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest  # type: ignore[missing-import]
import torch
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    host,
    router_ops,
    token_parallel,
    vllm_adapter,
)

pytestmark = pytest.mark.cpu_test


def _tpu7x_info():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


# ---------------------------------------------------------------------------
# 1. host.py coverage tests
# ---------------------------------------------------------------------------


def test_host_pow2_shift_and_layout_helpers():
    host.pow2_shift.cache_clear()
    assert host.pow2_shift(1, "one") == 0
    assert host.pow2_shift(8, "eight") == 3
    assert host.pow2_shift(8, "eight") == 3  # cache hit
    for bad in (0, -4, 3, 6):
        with pytest.raises(ValueError, match="not a power of two"):
            host.pow2_shift(bad, "bad_width")

    assert host.align_up(0, 8) == 0
    assert host.align_up(9, 8) == 16
    assert host.token_gather_window_rows(128) == 256
    assert host.token_gather_smem_bytes(128) == 2 * 256 * 4

    run_start = jnp.array([[0, 128], [64, 0]], dtype=jnp.int32)
    run_rows = jnp.array([[0, 64], [128, 0]], dtype=jnp.int32)
    tiles = host.mirror_run_tiles(run_start, run_rows, 128)
    np.testing.assert_array_equal(np.asarray(tiles), np.array([[0, 1], [2, 0]]))

    assert host.contrib_mirror_rows(256, 4, 128) == 2 + 8 + 2
    assert host.arrival_mirror_rows(512, 8, 128) == 4 + 16 + 2
    assert host.row_lane_blocks(512) == 4
    assert host.act_scale_slab_rows(200) == 2 + 1
    assert host.act_scale_window_rows(128) == 1 + 1


def test_host_weight_formats_and_bounds():
    assert str(host.WeightFormat.FP8) == "fp8"
    assert repr(host.WeightFormat.FP4) == "'fp4'"

    fp8_form = host.weight_form(host.WeightFormat.FP8)
    assert fp8_form.has_scales is True
    assert fp8_form.quantized_activations is True

    fp4_form = host.weight_form("fp4")
    assert fp4_form.name == host.WeightFormat.FP4
    assert fp4_form.has_scales is True
    assert fp4_form.quantized_activations is True

    int8_form = host.weight_form("int8")
    assert int8_form.has_scales is True
    assert int8_form.quantized_activations is False

    bf16_form = host.weight_form(host.WeightFormat.BF16)
    assert bf16_form.has_scales is False
    assert bf16_form.quantized_activations is False

    with pytest.raises(NotImplementedError, match="takes weight formats"):
        host.weight_form("fp16_unknown")

    assert host.weight_format_of_dtype(jnp.float8_e4m3fn) == host.WeightFormat.FP8
    assert host.weight_format_of_dtype(jnp.float4_e2m1fn) == host.WeightFormat.FP4
    assert host.weight_format_of_dtype(jnp.int8) == host.WeightFormat.INT8
    assert host.weight_format_of_dtype(jnp.bfloat16) == host.WeightFormat.BF16
    assert host.weight_format_of_dtype(jnp.float32) is None

    assert host.ragged_stride_bound(16, 2, 8, 128) == 256
    assert host.gather_clamp_bounds(0) == (0, 0)
    assert host.gather_clamp_bounds(10) == (0, 9)

    host.routing_block.cache_clear()
    assert host.routing_block(128, 2) == 256
    assert host.routing_block(3, 2) == 2
    assert host.routing_block(3, 2) == 2  # cache hit


def test_host_refuse_bad_plan_args():
    with pytest.raises(ValueError, match="at least one token"):
        host._refuse_bad_plan_args(0, 2, e_total=4, ep=2, t_local=1, block=2, tile_m=2)
    with pytest.raises(ValueError, match="not divisible by the expert-parallel width"):
        host._refuse_bad_plan_args(4, 2, e_total=5, ep=2, t_local=2, block=2, tile_m=2)
    with pytest.raises(ValueError, match="alignment slots"):
        host._refuse_bad_plan_args(
            65, 2, e_total=65, ep=65, t_local=1, block=2, tile_m=2
        )
    with pytest.raises(ValueError, match="corrupt each other silently"):
        host._refuse_bad_plan_args(
            2**25, 2, e_total=4, ep=2, t_local=2**24, block=4, tile_m=2
        )
    with pytest.raises(ValueError, match="routed pairs are not a whole number"):
        host._refuse_bad_plan_args(5, 1, e_total=4, ep=2, t_local=2, block=2, tile_m=2)
    with pytest.raises(ValueError, match="this shard's .* routed pairs"):
        host._refuse_bad_plan_args(8, 1, e_total=4, ep=2, t_local=3, block=4, tile_m=2)
    with pytest.raises(ValueError, match="tile height .* is not a whole number"):
        host._refuse_bad_plan_args(4, 2, e_total=4, ep=2, t_local=2, block=4, tile_m=3)


def test_host_replicated_and_sharded_routing_tables(monkeypatch):
    topk_idx = jnp.array([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=jnp.int32)  # T=4, K=2
    with pytest.raises(ValueError, match="per-shard slab stride"):
        host.build_routing_tables(
            topk_idx,
            e_total=4,
            ep=2,
            t_local=2,
            block=4,
            tile_m=4,
            shard_stride=6,
        )

    rt = host.build_routing_tables(
        topk_idx,
        e_total=4,
        ep=2,
        t_local=2,
        block=4,
        tile_m=4,
        shard_stride=8,
    )
    assert rt.pair_grid == (4, 2)
    assert rt.rows_alloc == 16

    # Sharded builder refusals
    with pytest.raises(ValueError, match="pass exactly one"):
        host.build_routing_tables_sharded(
            topk_idx,
            0,
            e_total=4,
            ep=2,
            t_local=2,
            block=4,
            tile_m=4,
            shard_stride=8,
        )
    with pytest.raises(ValueError, match="pass exactly one"):
        host.build_routing_tables_sharded(
            topk_idx,
            0,
            e_total=4,
            ep=2,
            t_local=2,
            block=4,
            tile_m=4,
            shard_stride=8,
            all_gather_rows=lambda r: r,
            gathered=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="per-shard slab stride"):
        host.build_routing_tables_sharded(
            topk_idx,
            0,
            e_total=4,
            ep=2,
            t_local=2,
            block=4,
            tile_m=4,
            shard_stride=6,
            all_gather_rows=lambda r: jnp.concatenate([r, r], axis=0),
        )

    # Sharded builder with all_gather_rows (unsplit window == block)
    expert_blocks = topk_idx.reshape(2, 4)
    full_hist = host.pair_block_hist(expert_blocks, 4)
    rows_by_dest = full_hist.reshape(2, 1, 4).sum(axis=1)

    srt0 = host.build_routing_tables_sharded(
        topk_idx,
        0,
        e_total=4,
        ep=2,
        t_local=2,
        block=4,
        tile_m=4,
        shard_stride=8,
        all_gather_rows=lambda _r: rows_by_dest,
    )
    assert srt0.pair_grid == (4, 2)
    np.testing.assert_array_equal(np.asarray(srt0.pos), np.asarray(rt.arrival_row[:2]))
    np.testing.assert_array_equal(
        np.asarray(srt0.mirror_pos), np.asarray(rt.mirror_row[:2])
    )

    # Sharded builder with gathered != None and sub-block window < block
    host.rank_window.cache_clear()
    assert host.rank_window(64, 4096) == 32
    assert host.rank_window(24, 4096) == 8
    monkeypatch.setattr(host, "ROUTING_WINDOW_MIN_PAIRS", 2)
    monkeypatch.setattr(host, "ROUTING_WINDOW", 2)
    host.rank_window.cache_clear()
    assert host.rank_window(4, 4) == 2

    gathered = host.GatheredPairs(
        expert_blocks=expert_blocks,
        local_blocks=expert_blocks[:1],
        block_hist=full_hist[:1],
        rows_by_dest=rows_by_dest,
        n_tokens=4,
        topk=2,
    )
    srt_win = host.build_routing_tables_sharded(
        None,
        0,
        e_total=4,
        ep=2,
        t_local=2,
        block=4,
        tile_m=4,
        shard_stride=8,
        gathered=gathered,
    )
    np.testing.assert_array_equal(
        np.asarray(srt_win.slab_row), np.asarray(srt0.slab_row)
    )

    # Downstream shard helpers
    exp_rows, exp_base = host.shard_expert_slabs(rt, 0, e_total=4, ep=2)
    assert exp_rows.shape == (2,)
    assert exp_base.shape == (2,)

    local_rows = host.local_slab_rows(rt, 1, shard_stride=8)
    g1 = host.shard_token_gather(rt, 1, shard_stride=8)
    g2 = host.shard_token_gather(rt, 1, shard_stride=8, rows=local_rows)
    np.testing.assert_array_equal(np.asarray(g1), np.asarray(g2))

    visit, n_visit = host.expert_visit_list(exp_rows, 2)
    assert visit.shape == (2,)
    assert n_visit.shape == (1,)

    true_rows, recv_row_off, totals_rows = host.shard_push_tables_in_rows(
        rt, 0, e_total=4, ep=2
    )
    assert true_rows.shape == (2, 2)
    assert recv_row_off.shape == (2, 2)
    assert totals_rows.shape == (2,)

    transport = host.shard_transport_tables_in_blocks(rt, 0, e_total=4, ep=2)
    assert len(transport) == 7

    counts = host.shard_count_vector(rt, exp_rows, 0, e_total=4, ep=2)
    assert counts.shape == (host.N_COUNTS, 2)


def test_host_vmem_and_device_queries(monkeypatch):
    info = _tpu7x_info()
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)

    assert host.vmem_limit() == int(info.vmem_capacity_bytes * host.VMEM_FRACTION)
    assert host.chip_generation() == info.generation
    assert host.chip_generation(info) == info.generation

    assert host.array_vmem_bytes((64,), jnp.float32, info) == 128 * 4
    assert host.array_vmem_bytes((2, 16, 128), jnp.float32, info) > 0

    host.check_u32_sublane_tile()
    host.check_u32_sublane_tile(info)
    bad_info = SimpleNamespace(get_sublane_tiling=lambda _dt: 16)
    with pytest.raises(ValueError, match="second-minor dimension tiles to"):
        host.check_u32_sublane_tile(bad_info)

    with pytest.raises(ValueError, match="weight buffers require"):
        host.vmem_scratch_arrays(2, 128, 512, 256, nbuf=1)
    with pytest.raises(ValueError, match="weight buffers require"):
        host.vmem_scratch_arrays(2, 128, 512, 256, nbuf=2, w2_nbuf=0)

    for fmt in (
        host.WeightFormat.FP8,
        host.WeightFormat.FP4,
        host.WeightFormat.INT8,
        host.WeightFormat.BF16,
    ):
        scratch = host.vmem_scratch_arrays(
            2,
            128,
            512,
            512,
            weight_format=fmt,
            has_w1_bias=True,
            has_w2_bias=True,
        )
        assert len(scratch) > 0
        body = host.vmem_tile_body_arrays(128, 512, 512, weight_format=fmt)
        assert len(body) > 0

    est_default = host.vmem_estimate_bytes(2, 128, 512, 256)
    est_explicit = host.vmem_estimate_bytes(2, 128, 512, 256, info=info, w2_nbuf=2)
    assert est_default == est_explicit

    plan_default = host.select_weight_buffers(2, 128, 512, 256)
    assert plan_default.w1_nbuf == host.NBUF

    # Force fallback to smaller w1_nbuf=2, then failure when budget is tiny
    est_nbuf2 = host.vmem_estimate_bytes(2, 128, 512, 256, nbuf=2, w2_nbuf=1, info=info)
    mid_info = SimpleNamespace(
        vmem_capacity_bytes=int(est_nbuf2 / host.VMEM_FRACTION) + 1024,
        num_lanes=info.num_lanes,
        get_sublane_tiling=info.get_sublane_tiling,
    )
    plan_mid = host.select_weight_buffers(2, 128, 512, 256, info=mid_info)
    assert (plan_mid.w1_nbuf, plan_mid.w2_nbuf) == (2, 1)

    tiny_info = SimpleNamespace(
        vmem_capacity_bytes=1024,
        num_lanes=info.num_lanes,
        get_sublane_tiling=info.get_sublane_tiling,
    )
    with pytest.raises(ValueError, match="over the .* VMEM budget"):
        host.select_weight_buffers(2, 128, 512, 256, info=tiny_info)


# ---------------------------------------------------------------------------
# 2. router_ops.py coverage tests
# ---------------------------------------------------------------------------


def test_router_ops_select_kernel_and_pallas_select(monkeypatch):
    scores = jnp.array(
        [[1.0, 5.0, 3.0, 5.0], [jnp.nan, jnp.nan, jnp.nan, jnp.nan]],
        dtype=jnp.float32,
    )

    class _Ref:
        def __init__(self, arr):
            self.arr = arr

        def __getitem__(self, idx):
            return self.arr[idx]

        def __setitem__(self, idx, val):
            self.arr = self.arr.at[idx].set(val)

    # Also test topk=0 (zero loop iterations) and topk=2
    x_ref0 = _Ref(scores)
    w_ref0 = _Ref(jnp.zeros((2, 0), dtype=jnp.float32))
    i_ref0 = _Ref(jnp.zeros((2, 0), dtype=jnp.int32))
    router_ops._select_kernel(x_ref0, w_ref0, i_ref0, topk=0, n=4)

    def _fake_pallas_call(kernel_fn, *, grid, in_specs, out_specs, out_shape):
        assert grid == (1,)
        for spec in (*in_specs, *out_specs):
            assert spec.index_map(0) == (0, 0)

        def _runner(inp):
            x_ref = _Ref(inp)
            w_ref = _Ref(jnp.zeros(out_shape[0].shape, dtype=out_shape[0].dtype))
            i_ref = _Ref(jnp.zeros(out_shape[1].shape, dtype=out_shape[1].dtype))
            kernel_fn(x_ref, w_ref, i_ref)
            return w_ref.arr, i_ref.arr

        return _runner

    monkeypatch.setattr(router_ops.pl, "pallas_call", _fake_pallas_call)
    w, idx = router_ops.pallas_select(scores, topk=2, block_rows=2)
    np.testing.assert_allclose(
        np.asarray(w), np.array([[5.0, 5.0], [router_ops.NEG, router_ops.NEG]])
    )
    np.testing.assert_array_equal(np.asarray(idx), np.array([[1, 3], [0, 0]]))

    with pytest.raises(AssertionError):
        router_ops.pallas_select(scores, topk=2, block_rows=4)


# ---------------------------------------------------------------------------
# 3. token_parallel.py coverage tests
# ---------------------------------------------------------------------------


def test_token_replica_layout_create_split_and_restore(monkeypatch):
    with pytest.raises(ValueError, match="must be positive"):
        token_parallel.TokenReplicaLayout.create(ep=0, input_rows=4)
    with pytest.raises(ValueError, match="must be positive"):
        token_parallel.TokenReplicaLayout.create(ep=2, input_rows=0)
    with pytest.raises(ValueError, match="positive power of two"):
        token_parallel.TokenReplicaLayout.create(ep=2, input_rows=4, row_alignment=0)
    with pytest.raises(ValueError, match="positive power of two"):
        token_parallel.TokenReplicaLayout.create(ep=2, input_rows=4, row_alignment=3)

    for bad_groups in (
        (),
        ((),),
        ((0,), (1, 2)),
        ((0, 0), (1, 2)),
    ):
        with pytest.raises(ValueError, match="must partition the mesh"):
            token_parallel.TokenReplicaLayout.create(
                ep=3 if len(bad_groups) == 2 and len(bad_groups[-1]) == 2 else 4,
                input_rows=4,
                groups=bad_groups,
            )

    # Singleton groups: split and restore are identity
    singleton = token_parallel.TokenReplicaLayout.create(ep=2, input_rows=5)
    rows = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    assert singleton.split(rows, jnp.int32(0)) is rows
    assert singleton.restore(rows, "ep") is rows

    # Width > 1 with zero padding (exact multiple)
    exact = token_parallel.TokenReplicaLayout.create(
        ep=2, input_rows=8, groups=((1, 0),), row_alignment=4
    )
    assert exact.local_rows == 4
    assert exact.slots == (1, 0)
    rows8 = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    s_rank1 = exact.split(rows8, jnp.int32(1))
    np.testing.assert_array_equal(np.asarray(s_rank1), np.asarray(rows8[:4]))
    s_rank0 = exact.split(rows8, jnp.int32(0))
    np.testing.assert_array_equal(np.asarray(s_rank0), np.asarray(rows8[4:]))

    monkeypatch.setattr(
        token_parallel.lax,
        "all_gather",
        lambda x, axis_name, **kw: jnp.concatenate([x, x], axis=0),
    )
    restored_exact = exact.restore(s_rank1, "ep")
    assert restored_exact.shape == (8, 2)

    # Width > 1 with non-zero padding (input_rows < local_rows * width)
    padded = token_parallel.TokenReplicaLayout.create(
        ep=2, input_rows=6, groups=((0, 1),), row_alignment=4
    )
    assert padded.local_rows == 4
    rows6 = jnp.ones((6, 2), dtype=jnp.float32)
    s_pad = padded.split(rows6, jnp.int32(1), fill_value=-1.0)
    assert s_pad.shape == (4, 2)
    np.testing.assert_array_equal(
        np.asarray(s_pad),
        np.array(
            [[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32
        ),
    )
    restored_padded = padded.restore(s_pad, "ep")
    assert restored_padded.shape == (6, 2)


# ---------------------------------------------------------------------------
# 4. vllm_adapter.py coverage tests
# ---------------------------------------------------------------------------


def _make_mock_layer(
    *,
    fp4: bool = False,
    use_ep: bool = True,
    pcp_size: int = 1,
    is_sequence_parallel: bool = False,
    global_num_experts: int = 8,
    experts_per_token: int = 2,
    hidden: int = 512,
    inter: int = 256,
    local_experts: int = 1,
):
    layer = SimpleNamespace(
        activation="silu",
        renormalize=True,
        global_num_experts=global_num_experts,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
        w13_bias=None,
        w2_bias=None,
        moe_config=SimpleNamespace(
            experts_per_token=experts_per_token,
            moe_parallel_config=SimpleNamespace(
                use_ep=use_ep,
                pcp_size=pcp_size,
                is_sequence_parallel=is_sequence_parallel,
            ),
        ),
    )
    if fp4:
        layer.w13_weight = torch.empty(
            (local_experts, hidden, inter), dtype=torch.float4_e2m1fn_x2, device="meta"
        )
        layer.w2_weight = torch.empty(
            (local_experts, inter, hidden // 2),
            dtype=torch.float4_e2m1fn_x2,
            device="meta",
        )
        layer.w13_weight_scale = torch.empty(
            (local_experts, hidden // 64, 1, 2 * inter),
            dtype=torch.float32,
            device="meta",
        )
        layer.w2_weight_scale = torch.empty(
            (local_experts, inter // 64, 1, hidden),
            dtype=torch.float32,
            device="meta",
        )
    else:
        layer.w13_weight = torch.empty(
            (local_experts, hidden, 2 * inter),
            dtype=torch.float8_e4m3fn,
            device="cpu",
        )
        layer.w2_weight = torch.empty(
            (local_experts, inter, hidden),
            dtype=torch.float8_e4m3fn,
            device="cpu",
        )
        layer.w13_weight_scale_inv = torch.ones(
            (local_experts, 1, 1, 2 * inter), dtype=torch.float32
        )
        layer.w2_weight_scale_inv = torch.ones(
            (local_experts, 1, 1, hidden), dtype=torch.float32
        )
    return layer


def test_vllm_adapter_helpers_and_unsupported_reasons(monkeypatch):
    info = _tpu7x_info()
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    assert vllm_adapter._device_info() is info

    # _max_node_tokens valid and invalid
    monkeypatch.setattr(
        vllm_adapter,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=512)
        ),
    )
    assert vllm_adapter._max_node_tokens(8, 2) == 2048
    with pytest.raises(ValueError, match="valid scheduler cap and PCP size"):
        vllm_adapter._max_node_tokens(8, 3)

    monkeypatch.setattr(
        vllm_adapter,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=0)
        ),
    )
    with pytest.raises(ValueError, match="valid scheduler cap and PCP size"):
        vllm_adapter._max_node_tokens(8, 1)

    # _squeeze_channel_scale and _squeeze_channel_scale_checked
    assert vllm_adapter._squeeze_channel_scale(None, 1, 4) is None
    assert vllm_adapter._squeeze_channel_scale(torch.ones(1, 2, 1, 4), 1, 4) is None
    assert vllm_adapter._squeeze_channel_scale(torch.ones(1, 1, 1, 8), 1, 4) is None
    contig = torch.ones(2, 1, 1, 4)
    s_contig = vllm_adapter._squeeze_channel_scale(contig, 2, 4)
    assert s_contig is not None and s_contig.shape == (2, 4)

    non_contig = torch.ones(4, 1, 1, 2).transpose(0, 3)
    assert not non_contig.squeeze(1).squeeze(1).is_contiguous()
    s_non_contig = vllm_adapter._squeeze_channel_scale(non_contig, 2, 4)
    assert s_non_contig is not None and s_non_contig.is_contiguous()

    checked = vllm_adapter._squeeze_channel_scale_checked(contig)
    assert checked.shape == (2, 4)

    # _mesh_expert_order
    monkeypatch.setattr(vllm_adapter, "ep_rank_order", lambda: None)
    assert vllm_adapter._mesh_expert_order(4) is None

    monkeypatch.setattr(vllm_adapter, "ep_rank_order", lambda: [0, 1, 2, 3])
    assert vllm_adapter._mesh_expert_order(4) is None

    monkeypatch.setattr(vllm_adapter, "ep_rank_order", lambda: [1, 0, 3, 2])
    assert vllm_adapter._mesh_expert_order(4) == (1, 0, 3, 2)

    monkeypatch.setattr(vllm_adapter, "ep_rank_order", lambda: [0, 0, 1, 2])
    with pytest.raises(ValueError, match="must permute range"):
        vllm_adapter._mesh_expert_order(4)


def test_vllm_adapter_fused_moe_ep_unsupported_reason_all_branches(monkeypatch):
    layer = _make_mock_layer(fp4=False)
    w1 = layer.w13_weight
    w2 = layer.w2_weight
    s1 = layer.w13_weight_scale_inv
    s2 = layer.w2_weight_scale_inv

    # 1. Rank / expert mismatch
    assert "weight rank/expert mismatch" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(layer, (w1[0], w2, s1, s2), "silu")
        or ""
    )
    # 2. w2 shape mismatch
    assert "is not [E," in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            layer,
            (w1, torch.empty(1, 128, 512, dtype=torch.float8_e4m3fn), s1, s2),
            "silu",
        )
        or ""
    )
    # 3. Activation not silu
    assert "not wired to the kernel's ACT_FNS" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(layer, (w1, w2, s1, s2), "gelu")
        or ""
    )
    # 4. FP4 checks
    fp4_layer = _make_mock_layer(fp4=True)
    fw1, fw2, fs1, fs2 = (
        fp4_layer.w13_weight,
        fp4_layer.w2_weight,
        fp4_layer.w13_weight_scale,
        fp4_layer.w2_weight_scale,
    )
    assert "native float4_e2m1fn_x2" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            fp4_layer,
            (fw1.to(torch.uint8), fw2, fs1, fs2),
            "silu",
            weight_format="fp4",
            rhs_qb=64,
        )
        or ""
    )
    assert "positive integer block size" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            fp4_layer, (fw1, fw2, fs1, fs2), "silu", weight_format="fp4", rhs_qb=0
        )
        or ""
    )
    assert "multiple of the packed-weight row tile" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            fp4_layer, (fw1, fw2, fs1, fs2), "silu", weight_format="fp4", rhs_qb=32
        )
        or ""
    )
    assert "must divide both" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            fp4_layer, (fw1, fw2, fs1, fs2), "silu", weight_format="fp4", rhs_qb=192
        )
        or ""
    )
    assert "w1_scale is not the block-64" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            fp4_layer,
            (fw1, fw2, torch.empty(1, 2, 1, 512), fs2),
            "silu",
            weight_format="fp4",
            rhs_qb=64,
        )
        or ""
    )
    assert "w2_scale is not the block-64" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            fp4_layer,
            (fw1, fw2, fs1, torch.empty(1, 2, 1, 512)),
            "silu",
            weight_format="fp4",
            rhs_qb=64,
        )
        or ""
    )
    # 5. FP8 checks
    assert "float8_e4m3fn expert weights" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            layer,
            (w1.to(torch.int8), w2, s1, s2),
            "silu",
            weight_format="fp8",
        )
        or ""
    )
    assert "w1_scale is not the per-output-channel" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            layer,
            (w1, w2, None, s2),
            "silu",
            weight_format="fp8",  # type: ignore[arg-type]
        )
        or ""
    )
    assert "w2_scale is not the per-output-channel" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            layer,
            (w1, w2, s1, None),
            "silu",
            weight_format="fp8",  # type: ignore[arg-type]
        )
        or ""
    )
    # 6. Unknown weight format
    assert "unsupported fused EP weight format" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(
            layer, (w1, w2, s1, s2), "silu", weight_format="int4"
        )
        or ""
    )
    # 7. Expert biases
    layer.w13_bias = torch.zeros(1)
    assert "expert biases are not wired" in (
        vllm_adapter.fused_moe_ep_unsupported_reason(layer, (w1, w2, s1, s2), "silu")
        or ""
    )
    layer.w13_bias = None

    # 8. Routing refusals
    layer.custom_routing_function = lambda *a: a
    assert "custom_routing_function" in (
        vllm_adapter._unsupported_routing_reason(layer) or ""
    )
    layer.custom_routing_function = None

    layer.scoring_func = "sigmoid"
    assert "scoring_func" in (vllm_adapter._unsupported_routing_reason(layer) or "")
    layer.scoring_func = "softmax"

    layer.e_score_correction_bias = torch.zeros(8)
    assert "e_score_correction_bias" in (
        vllm_adapter._unsupported_routing_reason(layer) or ""
    )
    layer.e_score_correction_bias = None

    layer.use_grouped_topk = True
    layer.num_expert_group = 4
    layer.topk_group = 2
    assert "grouped top-k" in (vllm_adapter._unsupported_routing_reason(layer) or "")
    layer.use_grouped_topk = False

    layer.hash_indices_table = torch.zeros(8)
    assert "hash routing" in (vllm_adapter._unsupported_routing_reason(layer) or "")
    layer.hash_indices_table = None

    layer.routed_scaling_factor = 2.0
    assert "routed_scaling_factor" in (
        vllm_adapter._unsupported_routing_reason(layer) or ""
    )
    layer.routed_scaling_factor = 1.0

    monkeypatch.setattr(vllm_adapter.moe_routing, "_SIMULATION_STRATEGY", "uniform")
    assert "VLLM_MOE_ROUTING_SIMULATION_STRATEGY" in (
        vllm_adapter._unsupported_routing_reason(layer) or ""
    )
    monkeypatch.setattr(vllm_adapter.moe_routing, "_SIMULATION_STRATEGY", None)
    assert vllm_adapter._unsupported_routing_reason(layer) is None


def test_vllm_adapter_prebuild_build_op_and_run(monkeypatch):
    info = _tpu7x_info()
    monkeypatch.setattr(vllm_adapter, "_device_info", lambda: info)
    monkeypatch.setattr(vllm_adapter, "_max_node_tokens", lambda ep, pcp: 4096)

    class _MockMesh:
        shape = {vllm_adapter.EP_AXIS_NAME: 8}

    mesh = _MockMesh()
    monkeypatch.setattr(vllm_adapter, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(vllm_adapter, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(vllm_adapter, "ep_rank_order", lambda: None)
    monkeypatch.setattr(
        vllm_adapter,
        "ep_token_replica_groups",
        lambda **kw: tuple((r,) for r in range(8)),
    )

    # Prebuild failure branches
    layer_no_ep = _make_mock_layer(use_ep=False)
    with pytest.raises(ValueError, match="requires expert-parallel weights"):
        vllm_adapter.prebuild_adaptive_fused_moe(layer_no_ep, 2, True, "silu")

    monkeypatch.setattr(vllm_adapter, "build_ep_mesh", lambda: None)
    with pytest.raises(ValueError, match="requires an EP group"):
        vllm_adapter.prebuild_adaptive_fused_moe(_make_mock_layer(), 2, True, "silu")
    monkeypatch.setattr(vllm_adapter, "build_ep_mesh", lambda: mesh)

    monkeypatch.setattr(vllm_adapter, "_max_node_tokens", lambda ep, pcp: 1)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 64)
    with pytest.raises(ValueError, match="MOE_FUSED_EP_KERNEL_MIN_TOKENS"):
        vllm_adapter.prebuild_adaptive_fused_moe(_make_mock_layer(), 2, True, "silu")
    monkeypatch.setattr(vllm_adapter, "_max_node_tokens", lambda ep, pcp: 4096)

    # FP4 weights=None path + bad scale inference
    fp4_layer = _make_mock_layer(fp4=True)
    fp4_bad_scale = _make_mock_layer(fp4=True)
    fp4_bad_scale.w13_weight_scale = torch.empty((1, 0, 1, 512), device="meta")
    with pytest.raises(ValueError, match="cannot infer FP4 block size"):
        vllm_adapter.prebuild_adaptive_fused_moe(fp4_bad_scale, 2, True, "silu")

    # Unsupported dtype
    bad_dtype_layer = _make_mock_layer()
    bad_dtype_layer.w13_weight = torch.empty((1, 512, 512), dtype=torch.float16)
    with pytest.raises(ValueError, match="unsupported adaptive fused MoE weight dtype"):
        vllm_adapter.prebuild_adaptive_fused_moe(bad_dtype_layer, 2, True, "silu")

    # Reason from fused_moe_ep_unsupported_reason
    with pytest.raises(ValueError, match="adaptive fused MoE: activation"):
        vllm_adapter.prebuild_adaptive_fused_moe(_make_mock_layer(), 2, True, "gelu")

    # global_num_experts != local_experts * ep
    bad_exp_layer = _make_mock_layer(global_num_experts=16)
    with pytest.raises(ValueError, match="equal contiguous expert blocks"):
        vllm_adapter.prebuild_adaptive_fused_moe(bad_exp_layer, 2, True, "silu")

    # topk out of range
    with pytest.raises(ValueError, match="topk must be between"):
        vllm_adapter.prebuild_adaptive_fused_moe(_make_mock_layer(), 0, True, "silu")

    # hidden / inter lane alignment
    unaligned_hidden = _make_mock_layer(hidden=64, inter=256)
    with pytest.raises(ValueError, match="do not satisfy lane alignment"):
        vllm_adapter.prebuild_adaptive_fused_moe(unaligned_hidden, 2, True, "silu")

    unaligned_inter_fp8 = _make_mock_layer(hidden=512, inter=160)
    with pytest.raises(
        ValueError, match="FP8 intermediate dimension must be lane aligned"
    ):
        vllm_adapter.prebuild_adaptive_fused_moe(unaligned_inter_fp8, 2, True, "silu")

    # Old chip generation
    v6e_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    monkeypatch.setattr(vllm_adapter, "_device_info", lambda: v6e_info)
    with pytest.raises(ValueError, match="requires TPU generation >="):
        vllm_adapter.prebuild_adaptive_fused_moe(_make_mock_layer(), 2, True, "silu")

    # SMEM exceeded
    tiny_smem_info = SimpleNamespace(
        generation=7,
        smem_capacity_bytes=1024,
        vmem_capacity_bytes=info.vmem_capacity_bytes,
        num_lanes=info.num_lanes,
        get_sublane_tiling=info.get_sublane_tiling,
    )
    monkeypatch.setattr(vllm_adapter, "_device_info", lambda: tiny_smem_info)
    with pytest.raises(ValueError, match="routing working set exceeds SMEM"):
        vllm_adapter.prebuild_adaptive_fused_moe(_make_mock_layer(), 2, True, "silu")
    monkeypatch.setattr(vllm_adapter, "_device_info", lambda: info)

    # _build_op: exercise closure creation, moe inner fn, _fake, and cache hit
    vllm_adapter._OPS.clear()
    captured = {}

    def _fake_jax_op(name, fn, *, mesh, input_partition_specs, output_partition_specs):
        captured["name"] = name
        captured["fn"] = fn

        class _Op:
            def __init__(self):
                self.fake_fn = None

            def register_fake(self, f):
                self.fake_fn = f

            def __call__(self, x, *args):
                return x, args

        op_obj = _Op()
        captured["op"] = op_obj
        return op_obj

    monkeypatch.setattr(vllm_adapter, "jax_op", _fake_jax_op)
    monkeypatch.setattr(
        vllm_adapter.adaptive_kernel,
        "adaptive_fused_moe",
        lambda x, *a, **kw: (x, kw["weight_format"]),
    )

    op_fp8 = vllm_adapter.prebuild_adaptive_fused_moe(
        _make_mock_layer(fp4=False), 2, True, "silu"
    )
    assert captured["name"] == vllm_adapter.FUSED_MOE_EP_OP_NAME
    # Call cached path
    assert (
        vllm_adapter.prebuild_adaptive_fused_moe(
            _make_mock_layer(fp4=False), 2, True, "silu"
        )
        is op_fp8
    )

    # Exercise inner `moe` and `_fake` registered on the op
    dummy_jax = jnp.ones((2, 4), dtype=jnp.bfloat16)
    res, fmt = captured["fn"](
        dummy_jax,
        dummy_jax,
        dummy_jax,
        dummy_jax,
        dummy_jax,
        dummy_jax,
        jnp.zeros((1, 1), dtype=jnp.int32),
    )
    assert fmt == host.WeightFormat.FP8
    fake_out = op_fp8.fake_fn(torch.ones(2, 4))
    assert fake_out.shape == (2, 4)

    # Second distinct op closure uses suffixed name and FP4 weights=None path
    op_fp4 = vllm_adapter.prebuild_adaptive_fused_moe(fp4_layer, 2, True, "silu")
    assert captured["name"] == f"{vllm_adapter.FUSED_MOE_EP_OP_NAME}_1"
    assert op_fp4 is not op_fp8

    # prebuild_adaptive_nvfp4 valid & invalid block
    nvfp4_ckpt_layer = _make_mock_layer(fp4=False)
    nvfp4_ckpt_layer.w13_weight = torch.empty((1, 512, 256), dtype=torch.uint8)
    nvfp4_ckpt_layer.w2_weight = torch.empty((1, 512, 128), dtype=torch.uint8)
    with pytest.raises(ValueError, match="must be positive and divide hidden"):
        vllm_adapter.prebuild_adaptive_nvfp4(
            nvfp4_ckpt_layer, "silu", configured_block=0
        )
    op_nv, blk = vllm_adapter.prebuild_adaptive_nvfp4(nvfp4_ckpt_layer, "silu")
    assert blk == 64
    assert op_nv is op_fp4

    # adaptive_fused_moe_supported & run_adaptive_fused_moe
    owner = SimpleNamespace()
    assert not vllm_adapter.adaptive_fused_moe_supported(owner)
    with pytest.raises(RuntimeError, match="was not prebuilt"):
        vllm_adapter.run_adaptive_fused_moe(
            owner,
            torch.ones(2, 512),
            torch.ones(1, 512, 512),
            torch.ones(1, 256, 512),
            torch.ones(1, 1, 1, 512),
            torch.ones(1, 1, 1, 512),
            torch.ones(2, 8),
        )

    setattr(owner, vllm_adapter.FUSED_MOE_EP_OP_ATTR, op_fp8)
    assert vllm_adapter.adaptive_fused_moe_supported(owner)

    # FP8 run_adaptive_fused_moe
    x = torch.ones(2, 512)
    out_fp8, args_fp8 = vllm_adapter.run_adaptive_fused_moe(
        owner,
        x,
        torch.empty(1, 512, 512, dtype=torch.float8_e4m3fn),
        torch.empty(1, 256, 512, dtype=torch.float8_e4m3fn),
        torch.ones(1, 1, 1, 512),
        torch.ones(1, 1, 1, 512),
        torch.ones(2, 8),
    )
    assert out_fp8 is x
    assert args_fp8[2].shape == (1, 512)

    # FP4 run_adaptive_fused_moe (4D scale, 3D scale, and invalid 2D scale)
    out_fp4, args_fp4 = vllm_adapter.run_adaptive_fused_moe(
        owner,
        x,
        torch.empty(1, 512, 256, dtype=torch.float4_e2m1fn_x2),
        torch.empty(1, 256, 256, dtype=torch.float4_e2m1fn_x2),
        torch.ones(1, 8, 1, 512),
        torch.ones(1, 4, 512),
        torch.ones(2, 8),
    )
    assert out_fp4 is x
    assert args_fp4[2].shape == (1, 8, 512)
    assert args_fp4[3].shape == (1, 4, 512)

    with pytest.raises(ValueError, match="require a contraction-block axis"):
        vllm_adapter.run_adaptive_fused_moe(
            owner,
            x,
            torch.empty(1, 512, 256, dtype=torch.float4_e2m1fn_x2),
            torch.empty(1, 256, 256, dtype=torch.float4_e2m1fn_x2),
            torch.ones(1, 512),
            torch.ones(1, 4, 512),
            torch.ones(2, 8),
        )


def test_td8316_adaptive_fused_moe_host_int32_cumsum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD8.316: Verify all 12 jnp.cumsum calls in adaptive_fused_moe/host.py
    pass explicit dtype=jnp.int32."""
    cumsum_dtypes: list[object] = []
    orig_cumsum = jnp.cumsum

    def _spy_cumsum(a, axis=None, dtype=None, out=None):
        cumsum_dtypes.append(dtype)
        return orig_cumsum(a, axis=axis, dtype=dtype, out=out)

    monkeypatch.setattr(host.jnp, "cumsum", _spy_cumsum)
    topk_idx = jnp.zeros((16, 2), dtype=jnp.int32)
    rt = host.build_routing_tables(
        topk_idx,
        e_total=4,
        ep=2,
        t_local=8,
        block=16,
        tile_m=8,
        shard_stride=16,
    )
    host.build_routing_tables_sharded(
        topk_idx,
        jnp.int32(0),
        e_total=4,
        ep=2,
        t_local=8,
        block=16,
        tile_m=8,
        shard_stride=16,
        all_gather_rows=lambda x: jnp.repeat(x, 2, axis=0),
    )
    host.shard_transport_tables_in_blocks(rt, jnp.int32(0), e_total=4, ep=2)
    assert len(cumsum_dtypes) == 12
    assert all(dt == jnp.int32 for dt in cumsum_dtypes)
