# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental host-local scatter / cross-host reduce / host-local gather.

Construct before compilation. Host membership comes from the CPU process group,
not JAX process_index: serving runs one process per core, unlike JAX SPMD tests.
"""

from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

HOST_REDUCE_AXIS = "moe_host"


@cache
def get_serving_host_reduce() -> Callable:
    """Initialize once before tracing, using the existing TP CPU group."""
    import socket

    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.logger import init_logger

    group = get_tp_group()
    if group.world_size != 32 or tuple(group.ranks) != tuple(range(32)):
        raise ValueError("Host reduction requires TP to cover the 32-rank world")
    hosts: list[Any] = [None] * group.world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group.cpu_group)
    mesh = get_pallas_mesh(axis_names=(HOST_REDUCE_AXIS,), mesh_shape=(32,))
    op = build_host_reduce(mesh, hosts)
    init_logger(__name__).info(
        "MoE host reduction initialized | TP=32 | hosts=4 | cores_per_host=8"
    )
    return op


def host_replica_groups(
    hosts: Sequence[str], device_ids: Sequence[int]
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return mesh-position groups for the measured four-host/eight-core layout."""
    if len(hosts) != 32 or len(device_ids) != 32 or len(set(device_ids)) != 32:
        raise ValueError("Host reduction requires 32 distinct devices")
    by_host: dict[str, list[int]] = {}
    for position, host in enumerate(hosts):
        by_host.setdefault(host, []).append(position)
    groups = tuple(
        tuple(sorted(positions, key=lambda i: device_ids[i]))
        for positions in by_host.values()
    )
    if len(groups) != 4 or any(len(group) != 8 for group in groups):
        raise ValueError("Host reduction requires four hosts with eight cores each")
    # Canonical order independent of process-group rank ordering.
    groups = tuple(sorted(groups, key=lambda group: device_ids[group[0]]))
    cross = tuple(zip(*groups))
    return groups, cross


def build_host_reduce(mesh: Any, hosts: Sequence[str]) -> Callable:
    """Build a TorchTPU custom op over a rank-aligned, one-dimensional mesh."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import jax_op

    if len(mesh.axis_names) != 1 or mesh.size != 32:
        raise ValueError("Host reduction requires a one-dimensional 32-core mesh")
    axis = mesh.axis_names[0]
    groups, cross = host_replica_groups(hosts, [int(d.id) for d in mesh.devices.flat])

    def collective(x: jax.Array) -> jax.Array:
        def local(part: jax.Array) -> jax.Array:
            part = jax.lax.psum_scatter(
                part, axis, scatter_dimension=1, tiled=True, axis_index_groups=groups
            )
            part = jax.lax.psum(part, axis, axis_index_groups=cross)
            return jax.lax.all_gather(
                part, axis, axis=1, tiled=True, axis_index_groups=groups
            )

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P(axis, None),
            out_specs=P(),
            check_vma=False,
        )(x)

    # Inputs stack the rank-local partials. The sum is genuinely replicated,
    # so its global output shape equals the local shape. Declaring a sharded
    # output here incorrectly inflates TorchTPU placeholders by world_size.
    op = jax_op(
        "pallas::moe_host_reduce_4x8",
        collective,
        mesh=mesh,
        input_partition_specs=(P(axis, None),),
    )
    op.register_fake(lambda x: x.new_empty(x.shape))
    return op
