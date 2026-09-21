# SPDX-License-Identifier: Apache-2.0
"""Exercise slice-ring reuse, empty experts and worst-case expert row counts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tests.kernels.experimental.adaptive_fused_moe.test_kernel import (
    _make_inputs, _make_weights)
from vllm_torchtpu.kernels.experimental.streaming_fused_moe import \
    streaming_fused_moe
from vllm_torchtpu.kernels.fused_moe.v2 import (AXIS, WeightFormat,
                                                fused_ep_moe_v2)

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module")
def operands():
    # Do not initialize PJRT during collection: other tests spawn TPU workers.
    if jax.device_count() < 8:
        pytest.skip("Requires eight local TPU devices")
    mesh = Mesh(np.asarray(jax.devices()[:8]), axis_names=(AXIS, ))
    weights = _make_weights(mesh,
                            7,
                            hidden=1024,
                            inter=1024,
                            e_total=16,
                            weight_format=WeightFormat.FP8)
    x, gating = _make_inputs(mesh, 8, tokens=32768, hidden=1024, e_total=16)
    return mesh, x, weights, gating


@pytest.mark.parametrize("slices,concentrated,capacity,tokens", [
    (2, False, 512, 32768),
    (4, False, 1024, 32768),
    (2, False, 1024, 4096),
    (2, True, 128, 32768),
    (2, False, 256, 32768),
    (2, True, 256, 32768),
    (2, False, 256, 128),
])
def test_matches_original_v2(operands, slices, concentrated, capacity, tokens):
    mesh, x, weights, gating = operands
    x, gating = x[:tokens], gating[:tokens]
    if concentrated:
        # Every token selects the same two experts on ranks 0 and 7. This
        # includes six entirely idle ranks and 32K rows per selected expert.
        def skew(logits):
            return jnp.full_like(logits, -1000).at[:, 0].set(1).at[:,
                                                                   15].set(2)

        gating = jax.jit(skew, out_shardings=gating.sharding)(gating)

    def run(implementation, tile_capacity=128, **extra):
        return jax.jit(lambda x, *args: implementation(x,
                                                       *args,
                                                       topk=2,
                                                       renormalize=True,
                                                       mesh=mesh,
                                                       capacity=tile_capacity,
                                                       sharded_plan=True,
                                                       **extra))(x, *weights,
                                                                 gating)

    expected = run(fused_ep_moe_v2)
    actual = run(streaming_fused_moe,
                 tile_capacity=capacity,
                 adaptive_rows=(tokens == 4096 and capacity == 1024),
                 weight_tile_bytes=1024 * (1024 // slices))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("capacity", (128, 512))
@pytest.mark.parametrize("output_buffers", (1, 2))
def test_fp4_block_scales_match_adaptive(capacity, output_buffers):
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import \
        adaptive_fused_moe

    mesh = Mesh(np.asarray(jax.devices()[:8]), axis_names=(AXIS, ))

    def local(_):
        keys = jax.random.split(
            jax.random.fold_in(jax.random.key(37), lax.axis_index(AXIS)), 4)
        weights, scales = [], []
        for index, (k, n) in enumerate(((1024, 2048), (1024, 1024))):
            weight = jax.random.normal(keys[2 * index],
                                       (2, k, n)).astype(jnp.float4_e2m1fn)
            scale = jax.random.uniform(keys[2 * index + 1], (2, k // 64, n),
                                       minval=.02,
                                       maxval=.08)
            weights.append(weight)
            scales.append(scale)
        return weights[0], weights[1], scales[0], scales[1]

    weights = jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(AXIS), ),
                      out_specs=(P(AXIS), ) * 4,
                      check_vma=False))(jnp.zeros(8))
    x, gating = _make_inputs(mesh, 38, tokens=4096, hidden=1024, e_total=16)

    def run(impl, m, **extra):
        return jax.jit(lambda x, *args: impl(x,
                                             *args,
                                             topk=2,
                                             renormalize=True,
                                             mesh=mesh,
                                             capacity=m,
                                             weight_format=WeightFormat.FP4,
                                             rhs_qb=64,
                                             **extra))(x, *weights, gating)

    expected = run(adaptive_fused_moe, 128)
    actual = run(streaming_fused_moe,
                 capacity,
                 weight_tile_bytes=256 * 2**10,
                 output_buffers=output_buffers,
                 down_compute_cols=256,
                 gate_compute_rows=512 if output_buffers == 1 else 256,
                 adaptive_rows=(capacity == 512),
                 sharded_plan=True)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
