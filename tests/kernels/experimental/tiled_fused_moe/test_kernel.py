# SPDX-License-Identifier: Apache-2.0
"""Exercise slice-ring reuse, empty experts and worst-case expert row counts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tests.kernels.experimental.adaptive_fused_moe.test_kernel import (
    _make_inputs, _make_weights)
from vllm_torchtpu.kernels.experimental.tiled_fused_moe import tiled_fused_moe
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
    (2, False, 128, 32768),
    (4, False, 128, 32768),
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
    actual = run(tiled_fused_moe, tile_capacity=capacity, weight_slices=slices)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_rejects_unsupported_output_width():
    from vllm_torchtpu.kernels.experimental.tiled_fused_moe.kernel import \
        _build_fused_ep_moe_kernel
    with pytest.raises(ValueError, match="256 columns"):
        _build_fused_ep_moe_kernel(g_local=2,
                                   capacity=128,
                                   hidden=512,
                                   inter=512,
                                   ep=8,
                                   weight_slices=4)
