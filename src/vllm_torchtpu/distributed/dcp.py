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
"""DCP (Decode Context Parallelism) process-group accessors.

Mirrors the role of distributed/pcp.py for PCP: a narrow, safe accessor
layer every call site shares, rather than each re-deriving group state
independently. The layout and mesh construction the two axes have in common
lives in mesh_utils.py.

"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
    get_or_create_cp_mesh,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The JAX mesh axis the DCP kernels shard the KV cache over.
DCP_AXIS_NAME = "dcp"


def get_dcp_group() -> Any | None:
    """Return vLLM's DCP GroupCoordinator, or None if not initialized."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group as _get_group

        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_dcp_rank() -> int:
    group = get_dcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_dcp_world_size() -> int:
    group = get_dcp_group()
    return 1 if group is None else int(group.world_size)


def get_dcp_group_layout() -> CpGroupLayout:
    """The DCP group's rank order and the TPU device each rank sits on."""
    return get_cp_group_layout(get_dcp_group())


def get_or_create_dcp_mesh(axis_name: str | None = None) -> Any:
    """Build the op-local JAX mesh for the DCP axis.

    Under `tp=N, dcp=N` the DCP group covers the same devices as the TP group
    and the mesh is one axis over it. A smaller group (`dcp < tp`, or `dp > 1`)
    gets a two-axis mesh over the whole world instead; see
    `_get_or_create_world_dcp_mesh`. The mesh cache is keyed on the axis name as well as the device ids, so
    a run with both CP axes live keeps two distinct meshes rather than
    aliasing them.

    The axis name defaults to `DCP_AXIS_NAME`; the kernels take the same
    constant from here, so a caller that overrides it has to override it in
    both places or the mesh and the partition specs will not agree.
    """
    if axis_name is None:
        axis_name = DCP_AXIS_NAME
    group = get_dcp_group()
    world_size = _torch_world_size()
    if group is None or world_size <= int(group.world_size):
        return get_or_create_cp_mesh(axis_name=axis_name, group=group)
    return _get_or_create_world_dcp_mesh(axis_name, group, world_size)


# The outer axis of the world-wide DCP mesh: one row per DCP group. No
# partition spec names it, so every operand is replicated across it.
DCP_REPLICA_AXIS_NAME = "dcp_replica"

_WORLD_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}


def _torch_world_size() -> int:
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _get_or_create_world_dcp_mesh(axis_name: str, group: Any, world_size: int) -> Any:
    """A `(world / dcp, dcp)` mesh for a DCP group smaller than the world.

    TorchTPU compiles an op-local program over the whole torch.distributed
    world (see `ep_mesh.py`), so a one-axis mesh over the DCP group alone is
    rejected at compile time when `dcp < tp` or `dp > 1`: "tile_assignment
    should have <world> devices but has <dcp>". Each row here is one DCP group
    in its rank order, so a `P(axis_name)` spec shards over the group and
    replicates over the rows, and collectives over `axis_name` stay inside a
    group.
    """
    import jax
    import numpy as np
    import torch
    from jax.sharding import Mesh
    from vllm.distributed.parallel_state import get_world_group

    layout = get_dcp_group_layout()
    cache_key = (axis_name, layout.device_ids)
    cached = _WORLD_MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    gathered: list[tuple[int, ...] | None] = [None] * world_size
    torch.distributed.all_gather_object(
        gathered, layout.device_ids, group=get_world_group().cpu_group
    )
    rows = sorted({tuple(row) for row in gathered if row is not None})
    flat = [device_id for row in rows for device_id in row]
    if len(flat) != world_size or len(set(flat)) != world_size:
        raise RuntimeError(
            f"DCP groups do not partition the world of {world_size}: {rows}"
        )
    devices_by_id = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    missing = [device_id for device_id in flat if device_id not in devices_by_id]
    if missing:
        raise RuntimeError(
            f"Cannot build the world-wide '{axis_name}' mesh: JAX does not "
            f"expose device ids {missing}; rows={rows}"
        )
    mesh = Mesh(
        np.asarray([[devices_by_id[d] for d in row] for row in rows]),
        axis_names=(DCP_REPLICA_AXIS_NAME, axis_name),
    )
    logger.info(
        "DCP world mesh | axis=%s dcp=%d world=%d rows=%s",
        axis_name,
        layout.world_size,
        world_size,
        rows,
    )
    _WORLD_MESH_CACHE[cache_key] = mesh
    return mesh
