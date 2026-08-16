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
import os
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, EP_MESH_DATA_AXIS
from vllm_torchtpu.layers.common.fused_moe_ep import fused_ep_moe_func

# Qwen3.5-397B-A17B-FP8, one v7x 2x2x1 node: ep=8, 64 local experts.
EP_SIZE = 8
HIDDEN = 4096
INTERMEDIATE = 1024
NUM_EXPERTS = 512
TOP_K = 10


@pytest.fixture(scope="module")
def topology():
    try:
        from jax.experimental import topologies
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
        w1=sds(
            (NUM_EXPERTS, HIDDEN,
             2 * INTERMEDIATE), jnp.float8_e4m3fn) if concat_layout else sds(
                 (NUM_EXPERTS, 2, HIDDEN, INTERMEDIATE), jnp.float8_e4m3fn),
        w2=sds((NUM_EXPERTS, INTERMEDIATE, HIDDEN), jnp.float8_e4m3fn),
        w1_scale=sds(
            (NUM_EXPERTS, 1, 1,
             2 * INTERMEDIATE), jnp.float32) if concat_layout else sds(
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
                  w13_concat_layout=True)
    kwargs.update(overrides)
    return functools.partial(fused_ep_moe_func, **kwargs)


# 2048 local x 8 ranks = the 16384-token full-prefill point in every layout the
# InferenceX sweep uses; the rest walk down to a decode-sized batch.
@pytest.mark.parametrize("local_tokens", [2048, 256, 8])
def test_mosaic_compiles_at_serving_shapes(mesh, topology, local_tokens):
    operands = _operands(mesh, local_tokens)
    lowered = jax.jit(
        _kernel(mesh)).trace(**operands).lower(lowering_platforms=("tpu", ))
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


# The kernel's own tuned table has no entry for this shape and falls back to a
# formula, so `blocks=` is how a tuner tries a candidate. Half the token block
# quarters the a2a scratch (num_experts x bt x hidden); gate it here, free,
# before it reaches a chip.
def test_a_non_default_block_size_compiles(mesh, topology):
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
    assert _exported_local_out_shape(mesh,
                                     local_tokens) == (local_tokens, HIDDEN)

    auto = _mesh(topology, (AxisType.Auto, AxisType.Auto))
    assert _exported_local_out_shape(auto, local_tokens) == (
        local_tokens * EP_SIZE, HIDDEN
    ), ("Auto axes are expected to erase the shard_map out-spec; if that ever "
        "changes, ep_mesh no longer needs to pin the axis types")


# Every test file this branch adds. Import hygiene is checked for all of them
# together because the cost of getting it wrong is not local to one file.
_ADDED_TEST_FILES = (
    "tests/kernels/fused_moe/test_fused_ep_aot.py",
    "tests/kernels/fused_moe/test_fused_ep_v1_upstream.py",
    "tests/layers/common/test_fused_moe_ep_numerics.py",
    "tests/layers/vllm/test_fused_moe_ep.py",
    "tests/layers/vllm/test_fused_moe_ep_contract.py",
    "tests/layers/vllm/test_fused_moe_ep_default_off.py",
)

_IMPORT_HYGIENE_PROBE = """
import importlib.util, sys
from jax._src import xla_bridge as xb

def live():
    return sorted(getattr(xb, "_backends", {}))

assert not live(), f"a backend existed before any test module was imported: {live()}"
for i, path in enumerate(sys.argv[1:]):
    spec = importlib.util.spec_from_file_location(f"_probe_{i}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if live():
        raise SystemExit(f"{path} initialised a JAX backend at import: {live()}")
print("OK")
"""


def test_importing_these_modules_opens_no_device():
    """Collection must not create a PJRT client. Run in a child, or the parent's
    own backend would mask the very thing being checked.

    A module-scope ``jax.default_backend()`` reads like a query and is not one:
    it initialises the backend. Doing that at import time means it happens
    during pytest *collection*, so the pytest process holds the TPU's vfio
    groups for the rest of the run and every later test that spawns an engine
    or a worker dies on ``Device or resource busy``. That is not hypothetical
    -- it took down three CI builds (PR #478, builds 2947/2952) and it looked
    convincingly like runner contention, because the set of busy iommu groups
    differed every time.
    """
    root = pathlib.Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable, "-c", _IMPORT_HYGIENE_PROBE,
            *(str(root / f) for f in _ADDED_TEST_FILES)
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ, "PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}"
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Substrings that mean "this test wants a PJRT client of its own": it starts a
# vLLM engine, a multiproc worker, or a spawned isolated process.
_NEEDS_ITS_OWN_DEVICE = ("run_in_isolated_process", "LLM(", "EngineCore",
                         "multiproc", "subprocess", "Process(")

_PORTED_SUITE = "tests/kernels/fused_moe/test_fused_ep_v1_upstream.py"


def test_nothing_needing_a_fresh_device_is_scheduled_after_the_ported_suite():
    """The ported suite runs in-process, so pin what may follow it.

    In the multichip job its tests really execute and build an 8-device mesh,
    which leaves a PJRT client alive in the pytest process (the same reason
    tests/distributed/tpu_test_utils.py exists). That is fine for neighbours
    that use JAX in-process -- they share the client, as the other ~34 kernel
    test files do -- and fatal for a neighbour that spawns an engine, which
    would get `Device or resource busy`.

    Today the two engine-spawning multichip files (`test_generate_tp*.py`) sort
    *before* us and the only one after is `test_gdn_attention_pcp.py`, which
    spawns nothing. CI feeds pytest a sorted glob, so that ordering is a
    property of the paths and is checkable here rather than being an assumption
    a future rename would silently break.
    """
    root = pathlib.Path(__file__).resolve().parents[3]
    marked = sorted(
        p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")
        if "multichip" in p.read_text(errors="ignore"))
    assert _PORTED_SUITE in marked, "the ported suite lost its multichip marker"

    offenders = {
        path:
        sorted(s for s in _NEEDS_ITS_OWN_DEVICE
               if s in (root / path).read_text(errors="ignore"))
        for path in marked if path > _PORTED_SUITE
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        "these multichip files sort after the ported suite and want their own "
        f"device: {offenders}. Either they will fail with 'Device or resource "
        "busy', or the ported suite now needs subprocess isolation.")
