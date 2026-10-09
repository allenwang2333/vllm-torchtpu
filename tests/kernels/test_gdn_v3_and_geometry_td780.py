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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn import head_geometry
from vllm_torchtpu.kernels.gdn.v3 import (
    compute_conv1d,
    config,
    metadata,
    pcp_metadata,
    projection_scale,
)


def _make_dtypes() -> config.Dtypes:
    return config.Dtypes(
        act_in=jnp.bfloat16,
        act_out=jnp.bfloat16,
        compute=jnp.float32,
        recurrent_state=jnp.float32,
        conv_state=jnp.bfloat16,
    )


def test_head_geometry_derivation_and_methods():
    geom_sharded = head_geometry.derive_gdn_head_geometry(8, 16, 4)
    assert head_geometry.derive_gdn_head_geometry(8, 16, 4) is geom_sharded
    assert geom_sharded.local_num_kq_heads == 2
    assert geom_sharded.local_num_v_heads == 4
    assert geom_sharded.kq_replication_factor == 1
    assert geom_sharded.kq_shard_index(3) == 3

    geom_replicated = head_geometry.derive_gdn_head_geometry(2, 8, 4)
    assert geom_replicated.local_num_kq_heads == 1
    assert geom_replicated.local_num_v_heads == 2
    assert geom_replicated.kq_replication_factor == 2
    assert geom_replicated.kq_shard_index(3) == 1

    with pytest.raises(ValueError, match="rank="):
        geom_sharded.kq_shard_index(-1)
    with pytest.raises(ValueError, match="rank="):
        geom_sharded.kq_shard_index(4)

    assert geom_sharded.local_conv_dim(64, 128) == 2 * 2 * 64 + 4 * 128
    with pytest.raises(ValueError, match="positive"):
        geom_sharded.local_conv_dim(0, 128)
    with pytest.raises(ValueError, match="positive"):
        geom_sharded.local_conv_dim(64, -1)

    conv_s, rec_s = geom_sharded.local_state_shapes(
        ((4, 3, 2048), (16, 64, 128)), 64, 128, conv_dim_axis=-1
    )
    assert conv_s == (4, 3, 768)
    assert rec_s == (4, 64, 128)

    conv_s2, rec_s2 = geom_sharded.local_state_shapes(
        ((4, 2048, 3), (16, 64, 128)), 64, 128, conv_dim_axis=1
    )
    assert conv_s2 == (4, 768, 3)
    assert rec_s2 == (4, 64, 128)

    with pytest.raises(ValueError, match="positive"):
        head_geometry.derive_gdn_head_geometry(0, 8, 2)
    with pytest.raises(ValueError, match="num_v_heads=8 must be divisible"):
        head_geometry.derive_gdn_head_geometry(4, 8, 3)
    with pytest.raises(ValueError, match="when Q/K heads are sharded"):
        head_geometry.derive_gdn_head_geometry(6, 8, 4)
    with pytest.raises(ValueError, match="when Q/K heads are replicated"):
        head_geometry.derive_gdn_head_geometry(3, 8, 4)
    with pytest.raises(ValueError, match="integer number of V heads"):
        head_geometry.derive_gdn_head_geometry(6, 8, 2)


def test_gdn_config_properties_and_methods(monkeypatch):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(num_lanes=128, vmem_capacity_bytes=32 * 1024 * 1024),
    )
    dtypes = _make_dtypes()
    region = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=4,
        view_dtype=jnp.bfloat16,
        lane_split=1,
        rows_used=4,
    )
    plan = config.StateSourcePlan(stride=2, conv=region, recurrent=region)

    cfg_per_seq = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=dtypes,
        batch_size=16,
        dim_size=256,
        kernel_size=4,
        tile_size=8,
        num_kq_heads=2,
        num_v_heads=8,
        kq_head_dim=64,
        v_head_dim=64,
        attention_mode=config.AttentionMode.GDN,
        state_plan=plan,
        window_size=1,
    )
    assert cfg_per_seq.chunk_size == 8
    assert cfg_per_seq.seq_tile_size == 1
    assert cfg_per_seq.prev_kernel_size == 3
    assert not cfg_per_seq.is_kda
    assert cfg_per_seq.gate_dim == 8
    assert cfg_per_seq.aligned_gate_dim == 128
    assert cfg_per_seq.triangular_block_size == 16
    assert not cfg_per_seq.use_recurrent
    assert cfg_per_seq.v_dim_size == 512
    assert cfg_per_seq.kq_dim_size == 128
    assert cfg_per_seq.v_per_kq_head == 4
    assert cfg_per_seq.aligned_num_v_heads == 128
    assert cfg_per_seq.get_kernel_name() == "fused_conv1d_gdn_per_seq"
    meta = cfg_per_seq.get_metadata()
    assert meta["batch_size"] == 16
    assert isinstance(meta["dtypes.act_in"], str)
    out_shape = cfg_per_seq.get_out_shape()
    assert out_shape.shape == (16, 8, 64)
    assert cfg_per_seq.get_vmem_limit_bytes() == int(0.7 * 32 * 1024 * 1024)
    scratch_per_seq = cfg_per_seq.get_scratch_shape_dict()
    assert scratch_per_seq["carry_conv_scratch_ref"] is not None
    assert scratch_per_seq["carry_recurrent_scratch_ref"] is not None

    for num_v_heads, expected_b in ((8, 16), (16, 8), (40, 4), (136, 2)):
        cfg_kda = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=dtypes,
            batch_size=16,
            dim_size=256,
            kernel_size=4,
            tile_size=8,
            num_kq_heads=2,
            num_v_heads=num_v_heads,
            kq_head_dim=64,
            v_head_dim=64,
            attention_mode=config.AttentionMode.KDA,
            gate_lower_bound=-5.0 if num_v_heads == 8 else None,
            window_size=2 if num_v_heads == 8 else 1,
        )
        assert cfg_kda.is_kda
        assert cfg_kda.gate_dim == num_v_heads * 64
        assert cfg_kda.triangular_block_size == expected_b
        assert cfg_kda.use_recurrent
        if num_v_heads == 8:
            assert cfg_kda.chunk_size == 2
            assert cfg_kda.seq_tile_size == 8
            assert cfg_kda.get_kernel_name() == "fused_conv1d_kda_batched_w2_bounded"
            assert cfg_kda.get_vmem_limit_bytes() == int(0.9 * 32 * 1024 * 1024)
            scratch_batched = cfg_kda.get_scratch_shape_dict()
            assert scratch_batched["carry_conv_scratch_ref"] is None
        else:
            assert cfg_kda.get_kernel_name() == "fused_conv1d_kda_batched"


def test_causal_conv1d_with_and_without_bias():
    dtypes = _make_dtypes()
    cfg_batched = config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=dtypes,
        batch_size=4,
        dim_size=8,
        kernel_size=3,
        tile_size=2,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=2,
        v_head_dim=2,
        window_size=1,
    )
    real_sizes_1 = jnp.array([1, 1], dtype=jnp.int32)
    lhs_1 = jnp.ones((2, 3, 1, 8), dtype=jnp.float32)
    weight = jnp.ones((3, 1, 8), dtype=jnp.float32)
    out_no_bias, state_no_bias = compute_conv1d.causal_conv1d(
        real_sizes_1, lhs_1, weight, None, cfg_batched
    )
    assert out_no_bias.shape == (2, 1, 1, 8)
    assert state_no_bias.shape == (2, 1, 2, 1, 8)
    np.testing.assert_allclose(np.asarray(out_no_bias), 3.0)

    cfg_per_seq = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=dtypes,
        batch_size=4,
        dim_size=8,
        kernel_size=3,
        tile_size=4,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=2,
        v_head_dim=2,
        window_size=2,
    )
    real_sizes_2 = jnp.array([2], dtype=jnp.int32)
    lhs_2 = jnp.arange(1 * 6 * 1 * 8, dtype=jnp.float32).reshape(1, 6, 1, 8)
    bias = jnp.full((8,), 0.5, dtype=jnp.float32)
    out_bias, state_bias = compute_conv1d.causal_conv1d(
        real_sizes_2, lhs_2, weight, bias, cfg_per_seq
    )
    assert out_bias.shape == (1, 4, 1, 8)
    assert state_bias.shape == (1, 2, 2, 1, 8)


def test_compute_batched_and_per_seq_metadata():
    dtypes = _make_dtypes()
    cfg = config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=dtypes,
        batch_size=8,
        dim_size=16,
        kernel_size=4,
        tile_size=2,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=4,
        v_head_dim=4,
    )
    seq_lens = jnp.array([5, 1, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 1, 2, 2, 2], dtype=jnp.int32)
    state_indices = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    read_offsets = jnp.zeros((4,), dtype=jnp.int32)
    end_seq = jnp.array(2, dtype=jnp.int32)
    ckpt_indices = jnp.arange(8, dtype=jnp.int32).reshape(4, 2)

    batched_meta = metadata.compute_batched_seq_metadata(
        cfg,
        seq_lens,
        query_start_loc,
        state_indices,
        read_offsets,
        end_seq,
        ckpt_indices=ckpt_indices,
    )
    assert int(batched_meta.num_tiles) == 1
    np.testing.assert_array_equal(
        np.asarray(batched_meta.s_idx_has_initial_state),
        np.array([True, False, False, False]),
    )

    cfg_per_seq = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=dtypes,
        batch_size=8,
        dim_size=16,
        kernel_size=4,
        tile_size=2,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=4,
        v_head_dim=4,
    )
    per_seq_meta_no_ckpt = metadata.compute_per_seq_metadata(
        cfg_per_seq,
        jnp.array([4, 3, 0, 0], dtype=jnp.int32),
        jnp.array([0, 3, 6, 6, 6], dtype=jnp.int32),
        state_indices,
        read_offsets,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(2, dtype=jnp.int32),
        ckpt_indices=None,
    )
    assert int(per_seq_meta_no_ckpt.num_tiles) == 4
    assert per_seq_meta_no_ckpt.s_idx_to_ckpt_indices is None

    per_seq_meta_ckpt = metadata.compute_per_seq_metadata(
        cfg_per_seq,
        jnp.array([4, 3, 0, 0], dtype=jnp.int32),
        jnp.array([0, 3, 6, 6, 6], dtype=jnp.int32),
        state_indices,
        read_offsets,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(2, dtype=jnp.int32),
        ckpt_indices=ckpt_indices,
    )
    assert int(per_seq_meta_ckpt.num_tiles) == 2
    assert per_seq_meta_ckpt.s_idx_to_ckpt_indices is not None


def test_pcp_stage_metadata_and_validations():
    dtypes = _make_dtypes()
    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=dtypes,
        batch_size=16,
        dim_size=16,
        kernel_size=4,
        tile_size=2,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=4,
        v_head_dim=4,
    )
    seq_lens = jnp.array([6, 4], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 4, 7], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    start_seq = jnp.array(0, dtype=jnp.int32)
    end_seq = jnp.array(2, dtype=jnp.int32)

    gdn_meta, stage_meta = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start_loc,
        state_indices,
        start_seq,
        end_seq,
        pcp_size=2,
        comm_chunk_size=2,
        projection_token_block_size=2,
        num_qkv_out_blocks=2,
        num_projection_out_blocks=3,
    )
    assert int(stage_meta.num_stages) >= 2
    assert int(gdn_meta.num_tiles) >= 2

    # Also exercise non-int32 input casts across both public and coordinate entrypoints.
    gdn_meta_i16, _ = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens.astype(jnp.int16),
        query_start_loc.astype(jnp.int16),
        state_indices.astype(jnp.int16),
        start_seq.astype(jnp.int16),
        end_seq.astype(jnp.int16),
        pcp_size=2,
        comm_chunk_size=2,
        projection_token_block_size=2,
        num_qkv_out_blocks=2,
        num_projection_out_blocks=3,
    )
    assert int(gdn_meta_i16.num_tiles) == int(gdn_meta.num_tiles)

    pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
        cfg,
        query_start_loc.astype(jnp.int16),
        state_indices.astype(jnp.int16),
        start_seq.astype(jnp.int16),
        end_seq.astype(jnp.int16),
        token_owner_starts=jnp.array([0, 4], dtype=jnp.int16),
        request_absolute_starts=jnp.array([2, 1], dtype=jnp.int16),
        pcp_size=2,
        comm_chunk_size=2,
        projection_token_block_size=2,
        num_qkv_out_blocks=2,
        num_projection_out_blocks=3,
    )

    with pytest.raises(ValueError, match="state_indices and seq_lens"):
        pcp_metadata.compute_pcp_stage_metadata(
            cfg,
            seq_lens,
            query_start_loc,
            jnp.array([0], dtype=jnp.int32),
            start_seq,
            end_seq,
            pcp_size=2,
            comm_chunk_size=2,
            projection_token_block_size=2,
            num_qkv_out_blocks=2,
            num_projection_out_blocks=3,
        )
    with pytest.raises(ValueError, match="query_start_loc must contain"):
        pcp_metadata.compute_pcp_stage_metadata(
            cfg,
            seq_lens,
            jnp.array([0, 4], dtype=jnp.int32),
            state_indices,
            start_seq,
            end_seq,
            pcp_size=2,
            comm_chunk_size=2,
            projection_token_block_size=2,
            num_qkv_out_blocks=2,
            num_projection_out_blocks=3,
        )

    base_kwargs = dict(
        token_owner_starts=jnp.array([0, 4], dtype=jnp.int32),
        request_absolute_starts=jnp.array([2, 1], dtype=jnp.int32),
        pcp_size=2,
        comm_chunk_size=2,
        projection_token_block_size=2,
        num_qkv_out_blocks=2,
        num_projection_out_blocks=3,
    )
    with pytest.raises(ValueError, match="pcp_size > 1"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(base_kwargs | {"pcp_size": 1}),
        )
    with pytest.raises(ValueError, match="comm_chunk_size must be positive"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(base_kwargs | {"comm_chunk_size": 0}),
        )
    with pytest.raises(ValueError, match="projection_token_block_size"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(base_kwargs | {"projection_token_block_size": 0}),
        )
    with pytest.raises(ValueError, match="num_qkv_out_blocks must be positive"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(base_kwargs | {"num_qkv_out_blocks": 0}),
        )
    with pytest.raises(ValueError, match="num_projection_out_blocks must cover"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(base_kwargs | {"num_projection_out_blocks": 1}),
        )
    with pytest.raises(ValueError, match="query_start_loc must contain"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            jnp.array([0, 4], dtype=jnp.int32),
            state_indices,
            start_seq,
            end_seq,
            **base_kwargs,
        )
    with pytest.raises(ValueError, match="token_owner_starts and state_indices"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(base_kwargs | {"token_owner_starts": jnp.array([0], dtype=jnp.int32)}),
        )
    with pytest.raises(ValueError, match="request_absolute_starts and state_indices"):
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start_loc,
            state_indices,
            start_seq,
            end_seq,
            **(
                base_kwargs
                | {"request_absolute_starts": jnp.array([0], dtype=jnp.int32)}
            ),
        )


def test_projection_scale_layout_and_normalization():
    w_shape = (8, 256)
    assert projection_scale.projection_scale_layout(w_shape, [1]) == (
        projection_scale.ProjectionScaleLayout.TENSOR,
        1,
        1,
    )
    assert projection_scale.projection_scale_layout(w_shape, (8,)) == (
        projection_scale.ProjectionScaleLayout.CHANNEL,
        8,
        1,
    )
    assert projection_scale.projection_scale_layout(w_shape, (4, 2)) == (
        projection_scale.ProjectionScaleLayout.BLOCK_GRID,
        4,
        2,
    )
    assert projection_scale.projection_scale_layout(w_shape, (1, 2, 1, 8)) == (
        projection_scale.ProjectionScaleLayout.LINEAR_RUNTIME,
        8,
        2,
    )

    with pytest.raises(ValueError, match="FP8 QKVZ scale must be scalar"):
        projection_scale.projection_scale_layout(w_shape, (2, 2, 2))
    with pytest.raises(ValueError, match="divide N and K"):
        projection_scale.projection_scale_layout(w_shape, (3, 2))
    with pytest.raises(ValueError, match="multiple of 128"):
        projection_scale.projection_scale_layout((8, 128), (4, 2))

    s_tensor = projection_scale.normalize_projection_scale(
        w_shape, jnp.array([2.0], dtype=jnp.float32)
    )
    assert s_tensor.shape == (8,)

    s_channel = projection_scale.normalize_projection_scale(
        w_shape, jnp.arange(8, dtype=jnp.float32)
    )
    assert s_channel.shape == (8,)

    s_grid_repeat = projection_scale.normalize_projection_scale(
        w_shape, jnp.ones((4, 2), dtype=jnp.float32)
    )
    assert s_grid_repeat.shape == (2, 8)

    s_grid_exact = projection_scale.normalize_projection_scale(
        w_shape, jnp.ones((8, 1), dtype=jnp.float32)
    )
    assert s_grid_exact.shape == (8,)

    s_linear = projection_scale.normalize_projection_scale(
        w_shape, jnp.ones((1, 2, 1, 8), dtype=jnp.float32)
    )
    assert s_linear.shape == (2, 8)
