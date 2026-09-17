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
"""Group layout and JAX mesh construction shared by the context-parallel axes.

`pcp.py` and `dcp.py` each need the same two things: the order of the ranks in
their process group paired with the TPU device each one sits on, and a
one-axis JAX mesh laid out in that order. Neither is specific to which axis
asked -- a layout is just "this group's ranks, in order, and where they live"

`ep_mesh.py` builds its own mesh but takes the rank/device helpers from here
too, which is why this module is named for the job rather than for an axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class CpGroupLayout:
    """A CP process group's rank order plus the matching TPU device ids."""

    ranks: tuple[int, ...]
    device_ids: tuple[int, ...]
    rank_in_group: int
    world_size: int

    @property
    def device_id(self) -> int:
        return self.device_ids[self.rank_in_group]

    @property
    def prev_rank_in_group(self) -> int:
        return (self.rank_in_group - 1) % self.world_size

    @property
    def next_rank_in_group(self) -> int:
        return (self.rank_in_group + 1) % self.world_size


_LAYOUT_CACHE: dict[tuple[int, tuple[int, ...], int, int], CpGroupLayout] = {}
_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}


def _get_current_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _get_tpu_global_device_id() -> int:
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        return int(tpu_distributed.global_device_id())
    except Exception:
        # Single-process tests may exercise layout code before TorchTPU/PJRT is
        # initialized. Use JAX's local device id when available, otherwise the
        # only sensible single-device fallback is 0.
        try:
            import jax

            local_devices = jax.local_devices()
            if local_devices:
                return int(getattr(local_devices[0], "id", 0))
        except Exception:
            pass
        return 0


def _collect_rank_to_device_id(group: Any, global_rank: int,
                               device_id: int) -> dict[int, int]:
    gathered: list[tuple[int, int] | None] = [None] * int(group.world_size)
    torch.distributed.all_gather_object(gathered, (global_rank, device_id),
                                        group=group.cpu_group)
    rank_to_device_id: dict[int, int] = {}
    for item in gathered:
        if item is None:
            raise RuntimeError("CP device id gather returned an empty entry.")
        rank, gathered_device_id = item
        rank_to_device_id[int(rank)] = int(gathered_device_id)
    return rank_to_device_id


def get_cp_group_layout(group: Any | None) -> CpGroupLayout:
    """Return `group`'s rank order and the matching TPU device-id order.

    The order of ``device_ids`` mirrors ``group.ranks``. This is the order a CP
    Pallas mesh should use, so that a TP+CP layout forms one CP ring per TP
    lane.

    Args:
      group: the process group to describe, or None before distributed init
        (in which case the layout describes this worker alone).
    """

    global_rank = _get_current_global_rank()
    device_id = _get_tpu_global_device_id()
    if group is None or int(group.world_size) == 1:
        return CpGroupLayout(
            ranks=(global_rank, ),
            device_ids=(device_id, ),
            rank_in_group=0,
            world_size=1,
        )

    ranks = tuple(int(rank) for rank in group.ranks)
    rank_in_group = int(group.rank_in_group)
    cache_key = (global_rank, ranks, rank_in_group, device_id)
    cached = _LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rank_to_device_id = _collect_rank_to_device_id(group, global_rank,
                                                   device_id)
    missing = [rank for rank in ranks if rank not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all CP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}")
    device_ids = tuple(rank_to_device_id[rank] for rank in ranks)
    layout = CpGroupLayout(
        ranks=ranks,
        device_ids=device_ids,
        rank_in_group=rank_in_group,
        world_size=int(group.world_size),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def get_or_create_cp_mesh(axis_name: str, group: Any | None) -> Any:
    """Build a one-axis JAX mesh ordered by `group`.

    This helper is intentionally op-local: it does not replace the runner's
    normal single-device model mesh.

    The cache is keyed on `(axis_name, device_ids)`, so two axes coexist even
    when they cover the same devices -- which they do under `tp=N, dcp=N`.
    """

    layout = get_cp_group_layout(group)
    cache_key = (axis_name, layout.device_ids)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import jax
    import numpy as np
    from jax.sharding import Mesh

    devices_by_id = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    missing = [
        device_id for device_id in layout.device_ids
        if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(
            f"Cannot build the '{axis_name}' mesh because JAX does not "
            f"expose all of the group's TPU device ids: missing={missing}, "
            f"layout={layout}, "
            f"jax_device_ids={sorted(devices_by_id)}")

    mesh_devices = np.asarray(
        [devices_by_id[device_id] for device_id in layout.device_ids])
    mesh = Mesh(mesh_devices, axis_names=(axis_name, ))

    # This is the mesh the CP kernel runs on, so it is the only place that
    # sees the rank -> device correspondence the ring actually uses.
    # ``lax.axis_index(axis_name)`` inside the kernel does *not* return a
    # position in this array: jax.export drops the mesh's device array and the
    # runtime rebuilds the assignment by walking the client's devices in order,
    # so the kernel sees that order instead. A kernel that indexes host-built
    # per-rank tables with it needs partition_order_for_axis to translate.
    # Logging the two side by side makes a divergence visible instead of
    # silent.
    # Cached per layout, so this logs once per worker rather than per step.
    positions = ", ".join(
        f"pos{position}=rank{rank}/dev{device_id}"
        f"@{tuple(getattr(device, 'coords', ()))}"
        f"c{getattr(device, 'core_on_chip', '?')}"
        for position, (rank, device_id, device) in enumerate(
            zip(layout.ranks, layout.device_ids, mesh_devices)))
    logger.info(
        "CP device mesh | axis=%s world_size=%d my_rank_in_group=%d | %s",
        axis_name, layout.world_size, layout.rank_in_group, positions)

    _MESH_CACHE[cache_key] = mesh
    return mesh


def partition_order_for_axis(mesh: Any, axis_name: str) -> tuple[int, ...]:
    """Map each position in ``mesh`` along ``axis_name`` to its mesh coordinate.

    A kernel has two different numbers for the same peer. The host builds its
    token plan from the group layout, so plan entry ``i`` belongs to the device
    sitting at position ``i`` of ``mesh``. ``lax.axis_index`` and
    ``DeviceIdType.MESH`` instead speak the runtime device assignment, which
    ``jax.export`` rebuilds by walking the client's devices in order -- the
    mesh's own device array does not survive export. The two agree only when
    the group happens to be laid out in device order.

    Returns a tuple whose ``i``-th entry is the mesh coordinate addressing the
    device at position ``i``, so a caller can translate plan positions into
    peer addresses. Raises if the runtime assignment cannot be expressed that
    way, because silently addressing the wrong peer corrupts results instead
    of failing.
    """
    import jax
    import numpy as np

    axis_names = tuple(mesh.axis_names)
    if axis_names.count(axis_name) != 1:
        raise ValueError(f"Axis {axis_name!r} must occur exactly once in "
                         f"{axis_names!r}.")
    mesh_devices = np.asarray(mesh.devices)
    axis = axis_names.index(axis_name)
    axis_size = mesh_devices.shape[axis]

    # How jax.export reconstitutes the device assignment: the client's devices
    # in enumeration order, folded into the mesh shape.
    runtime_devices = jax.devices()
    if len(runtime_devices) < mesh_devices.size:
        raise RuntimeError(
            f"The {axis_name!r} mesh spans {mesh_devices.size} devices but "
            f"JAX exposes only {len(runtime_devices)}.")
    runtime_grid = np.asarray(runtime_devices[:mesh_devices.size]).reshape(
        mesh_devices.shape)
    runtime_position = {
        device.id: coordinate
        for coordinate, device in np.ndenumerate(runtime_grid)
    }

    order: list[int | None] = [None] * axis_size
    for coordinate, device in np.ndenumerate(mesh_devices):
        target = runtime_position.get(device.id)
        if target is None:
            raise RuntimeError(
                f"Device {device.id} of the {axis_name!r} mesh is absent from "
                f"the runtime device assignment.")
        off_axis = tuple(value for index, value in enumerate(coordinate)
                         if index != axis)
        target_off_axis = tuple(value for index, value in enumerate(target)
                                if index != axis)
        if off_axis != target_off_axis:
            raise RuntimeError(
                f"Device {device.id} moves off the {axis_name!r} axis between "
                f"the mesh and the runtime device assignment "
                f"({coordinate} -> {target}); peer addressing that only "
                f"rewrites the {axis_name!r} coordinate cannot express it.")
        position = coordinate[axis]
        if order[position] not in (None, target[axis]):
            raise RuntimeError(
                f"Position {position} of the {axis_name!r} axis maps to more "
                f"than one mesh coordinate ({order[position]} and "
                f"{target[axis]}).")
        order[position] = target[axis]

    if sorted(order) != list(range(axis_size)):  # type: ignore[type-var]
        raise RuntimeError(
            f"The {axis_name!r} position -> mesh coordinate map {order} is not "
            f"a permutation of 0..{axis_size - 1}.")
    return tuple(order)  # type: ignore[arg-type]
