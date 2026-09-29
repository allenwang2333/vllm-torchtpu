# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts and offline TPU lowering for two-shot all-reduce."""

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest
from jax.interpreters import mlir
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import two_shot

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("shape", [(), (1,), (127,), (129,), (3, 257), (4097,)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_owner_padding_preserves_input_and_zeroes_tail(shape, dtype):
    with jax.default_device(jax.devices("cpu")[0]):
        value = (
            jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape).astype(dtype)
        )
        packed = two_shot._pack(value, 32)
        flat = np.asarray(packed).reshape(-1)
    assert packed.shape[0] == 32
    assert packed.shape[-1] == 128
    np.testing.assert_array_equal(flat[: value.size].reshape(shape), value)
    np.testing.assert_array_equal(flat[value.size :], 0)


@pytest.mark.parametrize(
    "shape,dtype,axis_size,collective_id,reason",
    [
        ((0, 128), jnp.float32, 32, 0, "empty"),
        ((1, 128), jnp.float16, 32, 0, "BF16 and FP32"),
        ((1, 128), jnp.int32, 32, 0, "BF16 and FP32"),
        ((-1, 128), jnp.float32, 32, 0, "concrete nonnegative"),
        ((True, 128), jnp.float32, 32, 0, "concrete nonnegative"),
        ((1, 128), jnp.float32, 1, 0, "axis size"),
        ((1, 128), jnp.float32, 3, 0, "axis size"),
        ((1, 128), jnp.float32, True, 0, "integer"),
        ((1, 128), jnp.float32, 32, -1, "collective_id"),
        ((1, 128), jnp.float32, 32, 16, "collective_id"),
        ((1, 128), jnp.float32, 32, True, "collective_id"),
    ],
)
def test_invalid_contracts_are_rejected(shape, dtype, axis_size, collective_id, reason):
    actual = two_shot.two_shot_unsupported_reason(
        shape, dtype, axis_size=axis_size, collective_id=collective_id
    )
    assert reason in actual


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_padded_vmem_boundary(dtype):
    axis_size = 32
    per_owner_element = (3 * axis_size + 1) * jnp.dtype(dtype).itemsize + 12
    owner_rows = two_shot._VMEM_BUDGET_BYTES // (128 * per_owner_element)
    elements = axis_size * owner_rows * 128
    assert (
        two_shot.two_shot_unsupported_reason(
            (elements,), dtype, axis_size=axis_size, collective_id=15
        )
        is None
    )
    assert "VMEM" in two_shot.two_shot_unsupported_reason(
        (elements + 1,), dtype, axis_size=axis_size, collective_id=15
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize(
    "entry,effectful",
    [
        (two_shot.two_shot_all_reduce, True),
        (two_shot._two_shot_all_reduce_uniform_spmd, False),
    ],
    ids=["public-effectful", "private-uniform-spmd"],
)
def test_offline_tpu_lowering(dtype, entry, effectful):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("requires JAX_NUM_CPU_DEVICES=4 before JAX initialization")
    mesh = Mesh(np.asarray(devices[:4]), ("tp",))

    def local(x):
        return entry(x, axis_name="tp", axis_size=4, collective_id=3)

    mapped = jax.shard_map(
        local, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False
    )
    exported = jax.export.export(jax.jit(mapped), platforms=("tpu",))(
        jax.ShapeDtypeStruct((12, 257), dtype)
    )
    assert exported.out_avals[0].shape == (12, 257)
    assert exported.out_avals[0].dtype == dtype
    with mlir.make_ir_context():
        module = mlir.ir.Module.parse(exported.mlir_module())
        calls = []

        def visit(operation):
            if operation.name == "stablehlo.custom_call":
                target = mlir.ir.StringAttr(operation.attributes["call_target_name"])
                if target.value == "tpu_custom_call":
                    calls.append(operation)
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child.operation)

        visit(module.operation)
        assert len(calls) == 1
        attributes = calls[0].attributes
        actual_effectful = (
            mlir.ir.BoolAttr(attributes["has_side_effect"]).value
            if "has_side_effect" in attributes
            else False
        )
        assert actual_effectful == effectful


@pytest.fixture(params=["public-effectful", "private-uniform-spmd"])
def entry(request):
    if request.param == "public-effectful":
        return two_shot.two_shot_all_reduce
    return two_shot._two_shot_all_reduce_uniform_spmd


def test_bound_axis_size_must_match(entry):
    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices[:1]), ("tp",))

    def local(x):
        return entry(x, axis_name="tp", axis_size=2, collective_id=0)

    mapped = jax.shard_map(
        local, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False
    )
    with pytest.raises(ValueError, match="bound collective axis"):
        jax.eval_shape(mapped, jax.ShapeDtypeStruct((1, 128), jnp.float32))


@pytest.mark.parametrize(
    "shape,dtype,kwargs,reason",
    [
        ((1, 128), jnp.float32, {"axis_name": ""}, "axis_name"),
        ((1, 128), jnp.float16, {}, "BF16 and FP32"),
        ((0, 128), jnp.float32, {}, "empty"),
        ((1, 128), jnp.float32, {"collective_id": 16}, "collective_id"),
        ((32 * 1024 * 1024,), jnp.float32, {}, "VMEM"),
    ],
)
def test_entry_contracts_fail_before_axis_binding(entry, shape, dtype, kwargs, reason):
    options = {"axis_name": "tp", "axis_size": 32, "collective_id": 0} | kwargs
    with pytest.raises(ValueError, match=reason):
        entry(jax.ShapeDtypeStruct(shape, dtype), **options)
