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
"""A full-slice (dp, tp) JAX mesh for the MoE DP all-to-all ops.

Same construction family as `distributed.ep_mesh` / `distributed.pcp`: the
runner's model mesh is single-device per worker and ranks are crossed through
torch.distributed, but a single op can still run as an SPMD program over its
own mesh as long as every worker's `jax.devices()` exposes the whole slice
(verified here, loudly).

The MoE a2a exchanges within each TP lane's 4-member DP group via collectives
on the ``dp`` axis. The mesh still spans every device of the slice because
the compiled model program is a 32-partition SPMD module and XLA requires a
collective's replica groups to cover ALL partitions: a 4-device op-local mesh
fails the verifier ("replica groups should contain 32 partitions, but found
4"). With a full-slice (dp, tp) mesh each lane's DP group is one dp-axis
subgroup and the groups tile all 32 partitions.
"""

from typing import Any

import torch

from vllm_torchtpu.distributed.chip_topology import get_chip_topology
from vllm_torchtpu.distributed.pcp import (_get_current_global_rank,
                                           _get_tpu_global_device_id)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DP_AXIS_NAME = "dp"
TP_AXIS_NAME = "tp"

_MESH: Any = None


def _get_dp_group() -> Any | None:
    from vllm.distributed.parallel_state import get_dp_group
    try:
        group = get_dp_group()
    except (AssertionError, AttributeError, RuntimeError):
        return None
    return group if group is not None and int(group.world_size) > 1 else None


def build_dp_mesh() -> Any | None:
    """The (dp, tp) mesh over the whole slice, or None when DP=1.

    ``mesh.devices[d, t]`` is the device of (dp_rank d, tp_rank t), gathered
    over the world group rather than derived from rank arithmetic (chip
    coordinates do not promise any particular host ordering). Cached: the
    mapping comes from a collective, so this must not be called for the first
    time inside a traced region.
    """
    global _MESH
    if _MESH is not None:
        return _MESH
    group = _get_dp_group()
    if group is None:
        return None

    from vllm.distributed.parallel_state import (get_tp_group,
                                                 get_world_group)
    tp_group = get_tp_group()
    my_dp_rank = int(group.rank_in_group)
    my_tp_rank = int(tp_group.rank_in_group)
    my_device_id = _get_tpu_global_device_id()

    world = get_world_group()
    gathered: list[tuple[int, int, int] | None] = [None] * int(
        world.world_size)
    torch.distributed.all_gather_object(
        gathered, (my_device_id, my_dp_rank, my_tp_rank), group=world.cpu_group)

    dp_size = int(group.world_size)
    tp_size = int(tp_group.world_size)
    device_of: dict[tuple[int, int], int] = {}
    for item in gathered:
        if item is None:
            raise RuntimeError("DP mesh world gather returned an empty entry.")
        device_id, dp_rank, tp_rank = item
        device_of[(dp_rank, tp_rank)] = device_id
    if len(device_of) != dp_size * tp_size:
        raise RuntimeError(
            f"World gather found {len(device_of)} (dp, tp) pairs, expected "
            f"{dp_size * tp_size}.")

    import jax
    import numpy as np
    from jax.sharding import Mesh

    visible = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    missing = [d for d in device_of.values() if d not in visible]
    if missing:
        raise RuntimeError(
            "Cannot build the DP a2a mesh: JAX does not expose every device "
            f"in this worker. missing={missing}, "
            f"jax_visible_ids={sorted(visible)}.")

    ids = [[device_of[(d, t)] for t in range(tp_size)] for d in range(dp_size)]
    _MESH = Mesh(np.asarray([[visible[i] for i in row] for row in ids]),
                 axis_names=(DP_AXIS_NAME, TP_AXIS_NAME))
    logger.info("Built DP a2a mesh | axes=(%s, %s) | shape=(%d, %d) | ids=%s",
                DP_AXIS_NAME, TP_AXIS_NAME, dp_size, tp_size, ids)
    return _MESH


def build_chip_block_dest_table(num_blocks: int,
                                device: torch.device) -> torch.Tensor:
    """``table[b]`` = DP rank (== mesh ``dp`` coordinate) owning chip block b.

    Under hierarchical EP each chip block holds one contiguous block of
    experts (`moe_hierarchical`), so expert ``e`` lives on chip block
    ``e // experts_per_block`` and must be exchanged with the DP rank this
    table maps that block to. The mapping is a fact about the whole slice, so
    every rank computes the same table; it is gathered over the world group
    rather than derived from rank arithmetic because chip coordinates do not
    promise a host-contiguous numbering (measured on 2x2x4: the table is
    interleaved, e.g. [3, 2, 0, 1, ...]).

    Runs a world-group ``all_gather_object`` -- load-time only, never inside a
    traced region.
    """
    from vllm.distributed.parallel_state import get_world_group

    topology = get_chip_topology()
    if topology is None:
        raise RuntimeError(
            "Chip topology unavailable; cannot map expert blocks to DP ranks.")
    if topology.num_chips != num_blocks:
        raise RuntimeError(
            f"Expected {num_blocks} chip blocks (one per hierarchical EP "
            f"rank) but the topology reports {topology.num_chips} chips.")

    my_device_id = _get_tpu_global_device_id()
    my_position = topology.device_ids.index(my_device_id)
    my_chip, _ = topology.chip_of(my_position)
    dp_group = _get_dp_group()
    my_dp_rank = int(dp_group.rank_in_group) if dp_group is not None else 0

    world = get_world_group()
    gathered: list[tuple[int, int] | None] = [None] * int(world.world_size)
    torch.distributed.all_gather_object(gathered, (my_chip, my_dp_rank),
                                        group=world.cpu_group)

    table = [-1] * num_blocks
    for item in gathered:
        if item is None:
            raise RuntimeError("DP a2a chip-block gather returned an empty "
                               "entry.")
        chip, dp_rank = item
        if 0 <= chip < num_blocks:
            if table[chip] not in (-1, dp_rank):
                raise RuntimeError(
                    f"Chip block {chip} is claimed by two DP ranks "
                    f"({table[chip]} and {dp_rank}).")
            table[chip] = dp_rank
    if any(v < 0 for v in table):
        raise RuntimeError(
            f"No DP rank reported owning chip blocks "
            f"{[b for b, v in enumerate(table) if v < 0]}.")
    logger.info("DP a2a chip-block destination table: %s", table)
    return torch.tensor(table, dtype=torch.int32, device=device)
