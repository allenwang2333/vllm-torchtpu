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
"""Chip-free gate for the fused expert-parallel MoE kernel.

Two things are checked without a TPU, at the shapes we actually serve:

* **Mosaic compiles it.** ``libtpu`` can build a compile-only ``tpu7x`` topology
  on a CPU host, which is enough for a real AOT compile -- this is where VMEM
  and SMEM overflows show up.
* **The torch bridge sizes the output right.** Under the mesh's Explicit axis
  types ``jax.export`` keeps the shard_map out-spec on the output aval, which is
  what ``pallas.jax_op`` divides by; under the default Auto axes it records the
  output as replicated and the torch buffer comes back ``ep_size`` times too
  large. That is the whole reason ``ep_mesh`` pins the axis types.
"""

import functools

import jax
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
topologies = pytest.importorskip("jax.experimental.topologies")

from jax.sharding import AxisType, Mesh, PartitionSpec  # noqa: E402

from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME,  # noqa: E402
                                               EP_MESH_DATA_AXIS)
from vllm_torchtpu.layers.common.fused_moe_ep import \
    fused_ep_moe_func  # noqa: E402

# Qwen3.5-397B-A17B-FP8, one v7x 2x2x1 node: ep=8, 64 local experts.
EP_SIZE = 8
HIDDEN = 4096
INTERMEDIATE = 1024
NUM_EXPERTS = 512
TOP_K = 10


@pytest.fixture(scope="module")
def topology():
    try:
        return topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
    except Exception as exc:  # no libtpu, or a build without tpu7x
        pytest.skip(f"no compile-only tpu7x topology available: {exc}")


def _mesh(topology, axis_types):
    devices = np.asarray(tuple(topology.devices)[:EP_SIZE]).reshape(1, EP_SIZE)
    return Mesh(devices,
                axis_names=(EP_MESH_DATA_AXIS, EP_AXIS_NAME),
                axis_types=axis_types)


@pytest.fixture(scope="module")
def mesh(topology):
    """The mesh ``ep_mesh.get_or_create_ep_mesh`` builds."""
    return _mesh(topology, (AxisType.Explicit, AxisType.Explicit))


def _operands(mesh, local_tokens, concat_layout=True):
    """Global-shaped, EP-sharded operands, as the torch bridge builds them."""
    spec = PartitionSpec(EP_AXIS_NAME)

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape,
                                    dtype,
                                    sharding=jax.sharding.NamedSharding(
                                        mesh, spec))

    tokens = local_tokens * EP_SIZE
    return dict(
        tokens=sds((tokens, HIDDEN), jnp.bfloat16),
        gating_output=sds((tokens, NUM_EXPERTS), jnp.bfloat16),
        w1=sds((NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE), jnp.float8_e4m3fn)
        if concat_layout else sds(
            (NUM_EXPERTS, 2, HIDDEN, INTERMEDIATE), jnp.float8_e4m3fn),
        w2=sds((NUM_EXPERTS, INTERMEDIATE, HIDDEN), jnp.float8_e4m3fn),
        w1_scale=sds((NUM_EXPERTS, 1, 1, 2 * INTERMEDIATE), jnp.float32)
        if concat_layout else sds(
            (NUM_EXPERTS, 2, 1, 1, INTERMEDIATE), jnp.float32),
        w2_scale=sds((NUM_EXPERTS, 1, 1, HIDDEN), jnp.float32),
    )


def _kernel(mesh, **overrides):
    kwargs = dict(mesh=mesh,
                  ep_axis_name=EP_AXIS_NAME,
                  top_k=TOP_K,
                  renormalize=True,
                  activation="silu",
                  scoring_fn="softmax",
                  subc_quant_w1_sz=HIDDEN,
                  subc_quant_w2_sz=INTERMEDIATE,
                  quant_act=True,
                  w13_concat_layout=True)
    kwargs.update(overrides)
    return functools.partial(fused_ep_moe_func, **kwargs)


# 2048 local x 8 ranks = the 16384-token full-prefill point in every layout the
# InferenceX sweep uses; the rest walk down to a decode-sized batch.
@pytest.mark.parametrize("local_tokens", [2048, 256, 8])
@pytest.mark.parametrize("quant_act", [True, False])
def test_mosaic_compiles_at_serving_shapes(mesh, topology, local_tokens,
                                           quant_act):
    operands = _operands(mesh, local_tokens)
    lowered = jax.jit(_kernel(mesh, quant_act=quant_act)).trace(
        **operands).lower(lowering_platforms=("tpu", ))
    compiled = lowered.compile(device_assignment=tuple(topology.devices))

    stats = compiled.memory_analysis()
    assert stats.output_size_in_bytes == local_tokens * HIDDEN * 2


def _exported_local_out_shape(mesh, local_tokens):
    """What ``pallas.jax_op`` would allocate for the torch output buffer.

    Mirrors ``torch_tpu._internal.pallas.torch_placeholder``.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    operands = _operands(mesh, local_tokens)
    with jax._src.config.export_ignore_forward_compatibility(True):
        lowered = jax.export.export(jax.jit(_kernel(mesh)),
                                    platforms=["tpu"])(**operands)
    aval = lowered.out_avals[0]
    spec = getattr(getattr(aval, "sharding", None), "spec", None)
    return tuple(pallas_impl.get_local_shape(aval.shape, mesh, spec))


# A candidate for MOE_FUSED_EP_KERNEL_BLOCKS: half the token block, which
# quarters the a2a scratch (num_experts x bt x hidden).
def test_a_block_size_override_compiles(mesh, topology):
    operands = _operands(mesh, 2048)
    blocks = (64, 512, 1024, 2048, 32, 512, 1024, 2048)
    compiled = jax.jit(_kernel(mesh, blocks=blocks)).trace(**operands).lower(
        lowering_platforms=("tpu", )).compile(
            device_assignment=tuple(topology.devices))
    stats = compiled.memory_analysis()
    print(f"[blocks={blocks}] temp={stats.temp_size_in_bytes / 1e6:.1f} MB")
    assert stats.output_size_in_bytes == 2048 * HIDDEN * 2


def test_the_kernels_own_w13_layout_also_compiles(mesh, topology):
    """The upstream [E, 2, H, F] layout stays supported (and AOT-gated)."""
    operands = _operands(mesh, 256, concat_layout=False)
    compiled = jax.jit(_kernel(mesh, w13_concat_layout=False)).trace(
        **operands).lower(lowering_platforms=("tpu", )).compile(
            device_assignment=tuple(topology.devices))
    assert compiled.memory_analysis().output_size_in_bytes == 256 * HIDDEN * 2


def test_explicit_axes_give_the_bridge_a_local_output_shape(mesh, topology):
    local_tokens = 256
    assert _exported_local_out_shape(mesh, local_tokens) == (local_tokens,
                                                             HIDDEN)

    auto = _mesh(topology, (AxisType.Auto, AxisType.Auto))
    assert _exported_local_out_shape(auto, local_tokens) == (local_tokens *
                                                             EP_SIZE, HIDDEN), (
        "Auto axes are expected to erase the shard_map out-spec; if that ever "
        "changes, ep_mesh no longer needs to pin the axis types")
