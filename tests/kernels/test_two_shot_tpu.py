# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the same test list on every worker of an initialized TPU slice."""

import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.two_shot import (
    _two_shot_all_reduce_uniform_spmd,
    two_shot_all_reduce,
)

pytestmark = [pytest.mark.multichip, pytest.mark.nightly]


@pytest.fixture(scope="module", params=["public-effectful", "private-uniform-spmd"])
def all_reduce(request):
    if request.param == "public-effectful":
        return two_shot_all_reduce
    return _two_shot_all_reduce_uniform_spmd


@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    if len(devices) != 32 or any(d.platform != "tpu" for d in devices):
        pytest.skip("requires an initialized 32-device TPU worker group")
    return Mesh(np.asarray(devices), ("tp",))


def _values(shape, dtype, rank, iteration, exact):
    seed = 1987 + 1009 * rank + 1000003 * iteration
    rng = np.random.default_rng(seed)
    if exact:
        values = rng.integers(-1, 2, size=shape) / 32.0
    else:
        values = rng.uniform(-0.5, 0.5, size=shape)
    return values.astype(dtype)


def _input(mesh, shape, dtype, iteration, exact):
    sharding = NamedSharding(mesh, P("tp"))
    return jax.make_array_from_callback(
        (mesh.size,) + shape,
        sharding,
        lambda index: _values(shape, dtype, index[0].start, iteration, exact)[None],
    )


def _check(output, mesh, shape, dtype, iteration, exact):
    values = [
        _values(shape, dtype, rank, iteration, exact).astype(np.float64)
        for rank in range(mesh.size)
    ]
    reference = np.sum(values, axis=0)
    sum_abs = np.sum(np.abs(values), axis=0)
    gamma = (mesh.size - 1) * 2**-24 / (1 - (mesh.size - 1) * 2**-24)
    accumulation = gamma * sum_abs
    rounding = 2**-8 if dtype == ml_dtypes.bfloat16 else 2**-24
    bound = 0 if exact else accumulation + rounding * (np.abs(reference) + accumulation)
    failures = []
    replica = None
    for shard in output.addressable_shards:
        actual = np.asarray(shard.data)[0]
        failures.append(
            not np.all(np.abs(actual.astype(np.float64) - reference) <= bound)
        )
        if replica is not None:
            failures.append(
                not np.array_equal(actual.reshape(-1).view(np.uint8), replica)
            )
        replica = actual.reshape(-1).view(np.uint8)
    # Compare replicas across hosts without introducing communication inside timing.
    gathered = multihost_utils.process_allgather(np.asarray(replica), tiled=False)
    failures.append(not np.all(gathered == gathered[0]))
    status = multihost_utils.process_allgather(np.asarray(any(failures)), tiled=False)
    assert not np.any(status), "two-shot values or replicas disagree"


def _mapped(
    mesh, shape, *, all_reduce, length=1, gap=False, native_gap=False, collective_id=3
):
    def local(inputs):
        outputs = []
        for index in range(length):
            x = inputs[index][0]
            if outputs:
                previous = lax_barrier(outputs[-1])
                factor = lax_barrier(
                    1.0 + jnp.sum(previous.astype(jnp.float32)) * 1e-30
                )
                x = x * factor.astype(x.dtype)
            if gap and index:
                x = _scratch_gap(x)
            if native_gap and index:
                native = jax.lax.psum(jnp.sum(x.astype(jnp.float32)), "tp")
                x = x * lax_barrier(1.0 + native * 1e-30).astype(x.dtype)
            rank = jax.lax.axis_index("tp")
            delay = jax.lax.fori_loop(
                0, (rank + index) % 4 * 128, lambda _, v: jnp.sin(v) + 0.25, 0.25
            )
            x = x * lax_barrier(1.0 + delay * 1e-30).astype(x.dtype)
            output = all_reduce(
                x, axis_name="tp", axis_size=mesh.size, collective_id=collective_id
            )
            outputs.append(output[None])
        return tuple(outputs)

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=((P("tp"),) * length,),
            out_specs=(P("tp"),) * length,
            check_vma=False,
        )
    )


def lax_barrier(value):
    return jax.lax.optimization_barrier(value)


def _scratch_gap(x):
    def kernel(x_ref, out_ref, temporary, semaphore):
        copy = pltpu.make_async_copy(x_ref, temporary, semaphore)
        copy.start()
        copy.wait()
        copy = pltpu.make_async_copy(temporary, out_ref, semaphore)
        copy.start()
        copy.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
        compiler_params=pltpu.CompilerParams(
            has_side_effects=True, disable_semaphore_checks=False
        ),
        name="two_shot_lifecycle_scratch",
    )(x)


@pytest.mark.parametrize("dtype", [ml_dtypes.bfloat16, np.float32])
@pytest.mark.parametrize(
    "shape", [(), (1, 127), (1, 129), (3, 257), (4095,), (4097,), (4, 7168)]
)
def test_numerical_contract_and_repeated_inputs(mesh, all_reduce, shape, dtype):
    run = _mapped(mesh, shape, all_reduce=all_reduce)
    for exact in (True, False):
        for iteration in range(3):
            x = _input(mesh, shape, dtype, iteration, exact)
            output = run((x,))[0]
            jax.block_until_ready(output)
            _check(output, mesh, shape, dtype, iteration, exact)


@pytest.mark.parametrize("dtype", [ml_dtypes.bfloat16, np.float32])
@pytest.mark.parametrize("length", [7, 8, 31, 32])
def test_barrier_generations_and_scratch_reuse(mesh, all_reduce, dtype, length):
    shape = (1, 7168)
    run = _mapped(mesh, shape, all_reduce=all_reduce, length=length, gap=length < 10)
    for exact in (True, False):
        inputs = tuple(
            _input(mesh, shape, dtype, index, exact) for index in range(length)
        )
        outputs = run(inputs)
        jax.block_until_ready(outputs)
        for index, output in enumerate(outputs):
            _check(output, mesh, shape, dtype, index, exact)


@pytest.mark.parametrize("same_id", [True, False])
def test_separate_executables_and_delayed_host_entry(mesh, all_reduce, same_id):
    cases = [
        ((1, 129), ml_dtypes.bfloat16, 3),
        ((3, 257), np.float32, 3 if same_id else 4),
    ]
    runs = [
        _mapped(mesh, shape, all_reduce=all_reduce, collective_id=cid)
        for shape, _, cid in cases
    ]
    for iteration in range(8):
        shape, dtype, _ = cases[iteration % 2]
        x = _input(mesh, shape, dtype, iteration, False)
        if jax.process_index() == iteration % jax.process_count():
            time.sleep(0.01)
        output = runs[iteration % 2]((x,))[0]
        jax.block_until_ready(output)
        _check(output, mesh, shape, dtype, iteration, False)


def test_reordered_logical_ranks(mesh, all_reduce):
    mesh = Mesh(mesh.devices[::-1], mesh.axis_names)
    shape, dtype = (1, 7168), ml_dtypes.bfloat16
    x = _input(mesh, shape, dtype, 19, False)
    output = _mapped(mesh, shape, all_reduce=all_reduce)((x,))[0]
    jax.block_until_ready(output)
    _check(output, mesh, shape, dtype, 19, False)


def test_native_and_pallas_collectives_in_one_executable(mesh, all_reduce):
    shape, dtype = (1, 7168), ml_dtypes.bfloat16
    inputs = tuple(_input(mesh, shape, dtype, index, False) for index in range(3))
    run = _mapped(mesh, shape, all_reduce=all_reduce, length=3, native_gap=True)
    outputs = run(inputs)
    jax.block_until_ready(outputs)
    for index, output in enumerate(outputs):
        _check(output, mesh, shape, dtype, index, False)
