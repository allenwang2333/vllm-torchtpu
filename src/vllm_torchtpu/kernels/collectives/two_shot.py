# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm-torchtpu PR #567 and tpu-collectives.
"""Shard-local two-shot SUM all-reduce with invocation-local VMEM."""

import functools
import math
from numbers import Integral

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_VECTOR_LANES = 128
_VMEM_BUDGET_BYTES = 32 * 1024 * 1024


def _owner_shape(elements: int, axis_size: int) -> tuple[int, int]:
    owner_elements = (elements + axis_size - 1) // axis_size
    return (owner_elements + _VECTOR_LANES - 1) // _VECTOR_LANES, _VECTOR_LANES


def two_shot_unsupported_reason(
    shape, dtype, *, axis_size: int, collective_id: int
) -> str | None:
    """Check local contracts; the caller separately validates hardware and runtime."""
    if isinstance(axis_size, bool) or not isinstance(axis_size, Integral):
        return "axis_size must be an integer"
    if axis_size not in (2, 4, 8, 16, 32):
        return "two-shot requires an axis size of 2, 4, 8, 16 or 32"
    if (
        isinstance(collective_id, bool)
        or not isinstance(collective_id, Integral)
        or not 0 <= collective_id <= 15
    ):
        return "collective_id must be an integer in [0, 15]"
    if any(isinstance(d, bool) or not isinstance(d, Integral) or d < 0 for d in shape):
        return "input shape must have concrete nonnegative dimensions"
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        return "two-shot supports BF16 and FP32 inputs"
    elements = math.prod(shape)
    if elements == 0:
        return "empty tensors use native all-reduce"
    owner_elements = math.prod(_owner_shape(elements, axis_size))
    required = owner_elements * ((3 * axis_size + 1) * dtype.itemsize + 3 * 4)
    if required > _VMEM_BUDGET_BYTES:
        return "two-shot exceeds the conservative 32 MiB VMEM budget"
    return None


def _pack(x, axis_size):
    rows, lanes = _owner_shape(x.size, axis_size)
    flat = x.reshape(-1)
    return jnp.pad(flat, (0, axis_size * rows * lanes - flat.size)).reshape(
        axis_size, rows, lanes
    )


def _entry_barrier(axis_name, axis_size):
    # Every RS send follows its entry wait. Completing RS requires every peer,
    # so no next-invocation signal can precede any current-invocation wait.
    rank = lax.axis_index(axis_name)
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, axis_size):
        peer = lax.rem(rank + offset, axis_size)
        pl.semaphore_signal(
            barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(barrier, axis_size - 1)


def _kernel(
    x_ref,
    out_ref,
    incoming,
    reduced,
    rs_send,
    rs_recv,
    ag_send,
    ag_recv,
    *,
    axis_name,
    axis_size,
):
    rank = lax.axis_index(axis_name)
    _entry_barrier(axis_name, axis_size)
    copies = []
    for offset in range(1, axis_size):
        peer = lax.rem(rank + offset, axis_size)
        copy = pltpu.make_async_remote_copy(
            x_ref.at[peer],
            incoming.at[offset - 1],
            rs_send,
            rs_recv,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()

    def sum_chunks(start, stop):
        if stop - start == 1:
            chunk = x_ref[rank] if start == 0 else incoming[start - 1]
            return chunk.astype(jnp.float32)
        middle = (start + stop) // 2
        return sum_chunks(start, middle) + sum_chunks(middle, stop)

    reduced[...] = sum_chunks(0, axis_size).astype(reduced.dtype)
    out_ref[rank] = reduced[...]
    copies = []
    for offset in range(1, axis_size):
        peer = lax.rem(rank + offset, axis_size)
        copy = pltpu.make_async_remote_copy(
            reduced,
            out_ref.at[rank],
            ag_send,
            ag_recv,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()


def two_shot_all_reduce(
    x: jax.Array, *, axis_name: str, axis_size: int, collective_id: int
) -> jax.Array:
    """Sum a local array across a flat, single-axis shard_map mesh.

    BF16 inputs accumulate in FP32 and round once to BF16. FP32 inputs retain
    FP32 outputs. All participants must use the same shape, dtype, ID and call
    order. Only sequential dependent invocations are supported; overlapping
    protocols are unsupported. Hardware/runtime support and performance selection
    are the caller's responsibility.
    """
    return _two_shot_all_reduce(
        x,
        axis_name=axis_name,
        axis_size=axis_size,
        collective_id=collective_id,
        has_side_effects=True,
    )


def _two_shot_all_reduce_uniform_spmd(
    x: jax.Array, *, axis_name: str, axis_size: int, collective_id: int
) -> jax.Array:
    """Execute only within an audited uniform SPMD collective program.

    All participants must have the same live collective program after compiler
    optimization. In-graph invocations require genuine previous-output
    dependencies; separate executions must synchronize before the next starts.
    Independent or overlapping invocations are unsupported, including across
    distinct IDs and native collectives. The caller must establish these
    properties; matching source, shapes and dtypes alone is insufficient.

    Remote operands make this unsuitable as a generally pure local operation.
    Under the uniform-program contract, whole-group elimination or merging is
    safe because the protocol drains writes to invocation-local scratch/output.
    The compiler may optimize this entry; the public API remains effectful.
    """
    return _two_shot_all_reduce(
        x,
        axis_name=axis_name,
        axis_size=axis_size,
        collective_id=collective_id,
        has_side_effects=False,
    )


def _two_shot_all_reduce(
    x: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
    collective_id: int,
    has_side_effects: bool,
) -> jax.Array:
    if not isinstance(axis_name, str) or not axis_name:
        raise ValueError("axis_name must be a nonempty string")
    if reason := two_shot_unsupported_reason(
        x.shape, x.dtype, axis_size=axis_size, collective_id=collective_id
    ):
        raise ValueError(reason)
    if lax.axis_size(axis_name) != axis_size:
        raise ValueError("axis_size does not match the bound collective axis")
    packed = _pack(x, axis_size)
    result = pl.pallas_call(
        functools.partial(_kernel, axis_name=axis_name, axis_size=axis_size),
        out_shape=jax.ShapeDtypeStruct(packed.shape, x.dtype),
        in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=(
            pltpu.VMEM((axis_size - 1,) + packed.shape[1:], x.dtype),
            pltpu.VMEM(packed.shape[1:], x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            has_side_effects=has_side_effects,
            disable_semaphore_checks=False,
        ),
        name="two_shot_all_reduce",
    )(packed)
    return result.reshape(-1)[: x.size].reshape(x.shape)
