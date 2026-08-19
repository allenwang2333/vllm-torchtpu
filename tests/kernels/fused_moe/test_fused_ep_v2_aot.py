# SPDX-License-Identifier: Apache-2.0
"""Chip-free gate for the **v2** fused expert-parallel MoE kernel.

Same three things the v1 gate checks, at the same serving shapes:

* **Mosaic compiles it** against a compile-only ``tpu7x`` topology, which is
  where VMEM and SMEM overflows and vector-layout failures show up. v2 needs
  ``use_abstract_mesh`` around the trace because its VMEM accounting reads the
  device record (``host.vmem_limit`` -> ``pltpu.get_tpu_info()``).
* **The torch bridge sizes the output right** under the 1-D mesh v2 takes.
* **The loader's scale layout is accepted.** v2 refuses a scale table carrying
  the loader's singleton axes; ``fused_ep_moe_v2_func`` drops them, and this
  gate is what says the drop survives a real compile on a sharded operand.

It also pins the two build parameters that are *not* free: ``capacity`` is 128
because 256 hits a Mosaic vector-layout bug and 512 exceeds VMEM, and the
no-drop slab is what sets the HBM temp.
"""

import functools
import pathlib

import jax
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
topologies = pytest.importorskip("jax.experimental.topologies")

from jax.sharding import AxisType, Mesh, PartitionSpec  # noqa: E402

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME_1D  # noqa: E402
from vllm_torchtpu.layers.common.fused_moe_ep_v2 import (  # noqa: E402
    CAPACITY, fused_ep_moe_v2_func)

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
    devices = np.asarray(tuple(topology.devices)[:EP_SIZE]).reshape(EP_SIZE)
    return Mesh(devices, axis_names=(EP_AXIS_NAME_1D, ), axis_types=axis_types)


@pytest.fixture(scope="module")
def mesh(topology):
    """The mesh ``ep_mesh.get_or_create_ep_mesh_1d`` builds."""
    return _mesh(topology, (AxisType.Explicit, ))


def _operands(mesh, local_tokens, loader_scale_layout=True):
    """Global-shaped, EP-sharded operands, as the torch bridge builds them."""
    spec = PartitionSpec(EP_AXIS_NAME_1D)

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape,
                                    dtype,
                                    sharding=jax.sharding.NamedSharding(
                                        mesh, spec))

    tokens = local_tokens * EP_SIZE
    # process_weights_after_loading leaves the loader's singleton axes on the
    # scales (fp8.py); the kernel layer wants them gone.
    w1_scale = ((NUM_EXPERTS, 1, 1,
                 2 * INTERMEDIATE) if loader_scale_layout else
                (NUM_EXPERTS, 2 * INTERMEDIATE))
    w2_scale = ((NUM_EXPERTS, 1, 1, HIDDEN) if loader_scale_layout else
                (NUM_EXPERTS, HIDDEN))
    return dict(
        ep_ids=sds((EP_SIZE, ), jnp.int32),
        tokens=sds((tokens, HIDDEN), jnp.bfloat16),
        gating_output=sds((tokens, NUM_EXPERTS), jnp.bfloat16),
        w1=sds((NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE), jnp.float8_e4m3fn),
        w2=sds((NUM_EXPERTS, INTERMEDIATE, HIDDEN), jnp.float8_e4m3fn),
        w1_scale=sds(w1_scale, jnp.float32),
        w2_scale=sds(w2_scale, jnp.float32),
    )


def _kernel(mesh, **overrides):
    kwargs = dict(mesh=mesh, top_k=TOP_K, renormalize=True, activation="silu")
    kwargs.update(overrides)
    return functools.partial(fused_ep_moe_v2_func, **kwargs)


def _compile(mesh, topology, operands, **overrides):
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        lowered = jax.jit(_kernel(
            mesh,
            **overrides)).trace(**operands).lower(lowering_platforms=("tpu", ))
    return lowered.compile(device_assignment=tuple(topology.devices))


# 2048 local x 8 ranks = the 16384-token full-prefill point in every layout the
# InferenceX sweep uses; the rest walk down to a decode-sized batch. The whole
# ladder is gated because the build is shape-specialized: one Mosaic program per
# token bucket, and a bucket that fails to compile fails at serving time.
@pytest.mark.parametrize("local_tokens", [2048, 1024, 512, 256, 128, 64, 8])
@pytest.mark.parametrize("capacity_skew", [None, 0])
def test_mosaic_compiles_at_serving_shapes(mesh, topology, local_tokens,
                                           capacity_skew):
    """Both slabs, because both ship: the sized one by default and the
    drop-free one behind ``MOE_FUSED_EP_V2_CAPACITY_SKEW=0``."""
    compiled = _compile(mesh,
                        topology,
                        _operands(mesh, local_tokens),
                        capacity_skew=capacity_skew)
    stats = compiled.memory_analysis()
    print(f"[v2 local_tokens={local_tokens}] "
          f"temp={stats.temp_size_in_bytes / 1e6:.1f} MB "
          f"args={stats.argument_size_in_bytes / 1e6:.1f} MB")
    assert stats.output_size_in_bytes == local_tokens * HIDDEN * 2


def test_the_kernels_own_scale_layout_also_compiles(mesh, topology):
    """A caller that already squeezed the scales is not made worse off."""
    compiled = _compile(mesh, topology,
                        _operands(mesh, 256, loader_scale_layout=False))
    assert compiled.memory_analysis().output_size_in_bytes == 256 * HIDDEN * 2


def test_capacity_128_is_the_only_usable_tile_height(mesh, topology):
    """Why ``CAPACITY`` is a constant and not a tuning knob.

    256 is a Mosaic vector-layout bug and 512 is over the kernel's own VMEM
    budget, so a deployment has nothing to sweep here. Pinned so that a jax or
    kernel bump that fixes either one shows up as a failing test rather than as
    an opportunity nobody re-checks.
    """
    assert CAPACITY == 128
    operands = _operands(mesh, 2048)
    with pytest.raises(Exception, match="shape cast|Mosaic"):
        _compile(mesh, topology, operands, capacity=256)
    with pytest.raises(ValueError, match="VMEM|budget"):
        _compile(mesh, topology, operands, capacity=512)


def _exported_local_out_shape(mesh, local_tokens):
    """What ``pallas.jax_op`` would allocate for the torch output buffer.

    Mirrors ``torch_tpu._internal.pallas.torch_placeholder``.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    operands = _operands(mesh, local_tokens)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = jax.export.export(jax.jit(_kernel(mesh)),
                                        platforms=["tpu"])(**operands)
    aval = lowered.out_avals[0]
    spec = getattr(getattr(aval, "sharding", None), "spec", None)
    return tuple(pallas_impl.get_local_shape(aval.shape, mesh, spec))


def test_explicit_axes_give_the_bridge_a_local_output_shape(mesh, topology):
    local_tokens = 256
    assert _exported_local_out_shape(mesh,
                                     local_tokens) == (local_tokens, HIDDEN)

    auto = _mesh(topology, (AxisType.Auto, ))
    assert _exported_local_out_shape(auto, local_tokens) == (
        local_tokens * EP_SIZE, HIDDEN
    ), ("Auto axes are expected to erase the shard_map out-spec; if that ever "
        "changes, ep_mesh no longer needs to pin the axis types")


def _exported(mesh, local_tokens):
    """The module ``pallas.jax_op`` registers with torch_tpu's runtime.

    ``torch_tpu._internal.pallas.JaxCallable.__call__`` hands
    ``jax.export(...).mlir_module_serialized`` to ``register_custom_kernel``
    and the runtime compiles *that*, on its own.
    """
    operands = _operands(mesh, local_tokens)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        with jax._src.config.export_ignore_forward_compatibility(True):
            return jax.export.export(jax.jit(_kernel(mesh)),
                                     platforms=["tpu"])(**operands)


def _main_params(mlir: str) -> str:
    """The parameter list of the exported module's entry function."""
    import re
    sig = re.search(r"func\.func public @main\((.*?)\) ->", mlir, re.S)
    assert sig, "no @main in the exported module"
    return sig.group(1)


def test_the_shard_identity_is_a_parameter_not_a_constant(mesh):
    """The second half of the partition-id story, found on device a commit
    after the first.

    Asserting the *absence* of `partition_id` was not enough. The first fix
    exported with `partition_id: 0` and still died on device, because the shard
    identity was an `arange(ep)` built inside the trace and sharded over the
    mesh -- and the SPMD partitioner implements a sharded iota **by emitting
    `partition-id`**, inside the standalone kernel-compile that then refuses it.
    The instruction was not removed; its creation moved from jax to XLA, where
    no export-level check can see it.

    So assert the positive property instead, which is the one that actually
    holds: **the shard identity arrives as an input parameter**, sharded over
    the mesh axis. A parameter cannot be folded into an iota, because the
    compiler does not know its contents. ``replica_rank`` in
    ``layers/vllm/fused_moe_ep.py`` is the worked example -- it has always been
    a torch buffer, which is why the TP row-slicing path never had this bug.
    """
    import hashlib

    import vllm_torchtpu.kernels.fused_moe.v2.layer as _layer

    # Which layer.py actually ran: the first fix for this defect passed every
    # gate on the wrong tree, so the log says what was under test.
    src = pathlib.Path(_layer.__file__).read_bytes()
    print(f"[shard identity] layer.py md5={hashlib.md5(src).hexdigest()[:12]}")
    params = _main_params(_exported(mesh, 64).mlir_module())
    assert 'loc("ep_ids")' in params, (
        "the shard identity is not an operand of the exported module; anything "
        "the compiler can derive from the mesh, it derives with partition-id "
        "(see register_ep_rank_buffer)")
    ep_ids = next(p for p in params.split("%arg") if 'loc("ep_ids")' in p)
    assert f"tensor<{EP_SIZE}xi32>" in ep_ids, ep_ids
    assert '"d"' in ep_ids, f"ep_ids is not sharded over the mesh axis: {ep_ids}"


def test_the_gate_bites_when_the_identity_is_built_in_the_trace(mesh):
    """The negative control, because this gate has a bad track record.

    Two earlier versions of the partition-id check were green on a live
    failure. This one is only worth having if it can tell the two programs
    apart, so build the bad one on purpose -- ``ep_ids=None`` makes the layer
    fall back to an in-trace ``arange(ep)``, which is exactly what shipped in
    the first partition-id fix -- and require that the parameter is gone.
    """
    operands = _operands(mesh, 64)
    operands.pop("ep_ids")
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        with jax._src.config.export_ignore_forward_compatibility(True):
            bad = jax.export.export(jax.jit(_kernel(mesh)),
                                    platforms=["tpu"])(**operands)
    assert 'loc("ep_ids")' not in _main_params(bad.mlir_module())


def test_the_exported_module_has_no_partition_id(mesh):
    """The gate the AOT compile above cannot be: it compiles the *jax* program.

    A `lax.axis_index` in the shard_map body -- outside the Pallas kernel --
    lowers to an XLA `partition-id`. That is legal inside a GSPMD manual region,
    so `jax.jit(...).compile()` accepts it and every test above passes; the SPMD
    partitioner then refuses the *exported* module on its own with "partitionId
    instruction is not supported for SPMD partitioning since the meaning is
    ambiguous", and serving dies in `_dummy_run` on the first profile run. That
    is a real incident (seen on device, 2026-08-15), and it cost a TPU slot
    because the two compile paths disagree and only one of them was gated.

    Asserting on the exported MLIR rather than trying to reproduce the
    partitioner's refusal: the refusal needs the runtime's own compile
    invocation, the text does not, and the text is the thing that is wrong.
    """
    mlir = _exported(mesh, 64).mlir_module()
    # The dialect-qualified op name, not the bare word: MLIR carries python
    # symbol names in its `loc(...)` metadata, so a test named after the thing
    # it forbids would fail on its own name.
    asks = [
        op for op in ("stablehlo.partition_id", "mhlo.partition_id")
        if op in mlir
    ]
    assert not asks, (
        f"the exported bridge module asks for {asks}; every axis-derived "
        "scalar in the shard_map body has to arrive as data instead (see "
        "`me` in kernels/fused_moe/v2/layer.py)")


def test_the_smem_ceiling_is_refused_by_name(mesh, topology):
    """SMEM, not VMEM or HBM, is what caps v2's token bucket.

    ``shard_token_gather`` is one scalar-prefetch int32 per drop-free slab row,
    so it lives in SMEM: 654.5 kB of this chip's 1 MiB at the 16384-token
    bucket. Measured by holding the token count and growing the slab, Mosaic
    refuses at 262,144 rows with "Ran out of memory in memory space smem. Used
    1.22M of 1.00M smem. Exceeded smem capacity by 230.7K" -- a message that
    names neither the token count nor the setting that controls it, arriving at
    the first compile of a serving run. So the layer answers first.
    """
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import max_node_tokens

    limit = max_node_tokens(TOP_K, NUM_EXPERTS, 1024 * 1024)
    # 19,801 on the drop-free slab, which is what ships. The engagement band's
    # own cap (8192) is below it, so SMEM is not the binding constraint at the
    # default -- it becomes one for a deployment that sizes the slab and lifts
    # the cap, which is why the message names both knobs.
    assert 8192 < limit < 32768, limit

    # The next power-of-two bucket above the ceiling: pallas_select wants a
    # block height that divides the local row count, so an arbitrary count is
    # refused for an unrelated reason.
    local = 4096
    assert local * EP_SIZE > limit, (local * EP_SIZE, limit)
    # The default is the drop-free slab, so the default refuses here...
    with pytest.raises(ValueError, match="scalar-prefetch|SMEM"):
        _compile(mesh, topology, _operands(mesh, local))
    # ...and a sized slab is several times smaller, so the same shape compiles
    # on it: the ceiling is a property of the slab, not of the token count.
    # (That path is opt-in until its drop path stops writing out of bounds --
    # tests/kernels/fused_moe/test_fused_ep_v2_slab_overflow.py.)
    _compile(mesh, topology, _operands(mesh, local), capacity_skew=3.0)


def test_the_default_slab_is_the_drop_free_one(mesh, topology):
    """A caller driving the kernel from JAX must get the shipped program.

    The env var is read in the serving adapter, so a harness that calls
    ``fused_ep_moe_v2_func`` directly never sees it -- and the failure is a
    silent no-op with a correct-looking run (found exactly that way). The
    default therefore lives on the JAX entry point, and this is what says so.

    The shipped default is the **drop-free** slab: sizing it smaller is what
    makes 16384 affordable, but a shard that overflows a sized slab writes past
    the end of its buffer (``test_fused_ep_v2_slab_overflow.py``), so the
    sizing is opt-in. This asserts the default *is* the bound, and that asking
    for a sized slab still gets you a smaller program -- i.e. the knob works.
    """
    default = _compile(mesh, topology, _operands(mesh, 2048))
    drop_free = _compile(mesh,
                         topology,
                         _operands(mesh, 2048),
                         capacity_skew=0)
    sized = _compile(mesh, topology, _operands(mesh, 2048), capacity_skew=3.0)
    a = default.memory_analysis().temp_size_in_bytes
    b = drop_free.memory_analysis().temp_size_in_bytes
    c = sized.memory_analysis().temp_size_in_bytes
    print(f"[slab] default {a / 1e6:.1f} MB, drop-free {b / 1e6:.1f} MB, "
          f"sized {c / 1e6:.1f} MB")
    assert a == b, (a, b)
    assert c < b / 1.5, (c, b)
