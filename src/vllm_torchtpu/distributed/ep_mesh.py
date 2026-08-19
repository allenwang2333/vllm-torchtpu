# SPDX-License-Identifier: Apache-2.0
"""JAX mesh over vLLM's expert-parallel process group.

Same mechanism as :mod:`vllm_torchtpu.distributed.pcp`: every rank agrees on an
order for ``jax.devices()`` by gathering each peer's TPU global device id over
the group's CPU process group.

The mesh goes in **ascending device-id order, which is not EP-rank order**. A
Mosaic kernel that addresses a peer with ``DeviceIdType.MESH`` resolves the
coordinate positionally against the runtime's device list rather than through
the device array the mesh was built with, so mesh index ``i`` has to *be* device
id ``i``. On the tpu7x-8 we serve on, whose EP ranks come up as device ids
``(0, 1, 4, 5, 6, 7, 2, 3)``, a mesh in EP-rank order sends every routed row to
the wrong peer and the ranks deadlock on the first call.

Ordering by device id means mesh index ``i`` is no longer EP rank ``i``: the
shard there holds the experts of EP rank ``ep_rank_at_mesh_index[i]``. A kernel
that takes ownership as data can be told so -- see
``layers/vllm/fused_moe_ep.py`` -- and one that infers it from its own axis
index cannot, which is what ``validate_ep_mesh``'s ``reindexes_expert_order``
separates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.distributed.pcp import (_collect_rank_to_device_id,
                                           _get_current_global_rank,
                                           _get_tpu_global_device_id)

# The v1 kernel takes a 2-D mesh and requires every non-EP axis to have size 1;
# it reads the other axis by this name (kernels/fused_moe/v1/kernel.py:342).
EP_MESH_DATA_AXIS = "data"
EP_AXIS_NAME = "model"

# The v2 kernel takes a 1-D mesh and names its one axis in its own body
# (kernels/fused_moe/v2/kernel.py:42 ``AXIS = "d"``, read by ``lax.axis_index``
# and by every ``device_id=(i,)`` remote copy), so the name is a contract, not a
# choice. Upstream collapses its own multi-axis mesh the same way
# (tpu-inference ``moe_fused_ep._single_axis_mesh``).
EP_AXIS_NAME_1D = "d"


@dataclass(frozen=True)
class EpGroupLayout:
    """EP process-group order plus the matching TPU global device ids."""

    ranks: tuple[int, ...]
    device_ids: tuple[int, ...]
    rank_in_group: int
    world_size: int

    @property
    def mesh_device_ids(self) -> tuple[int, ...]:
        """Device ids in the order the mesh builders lay them out."""
        if len(set(self.device_ids)) != len(self.device_ids):
            raise RuntimeError(
                "Two EP ranks report the same TPU global device id, so no mesh "
                f"order can address them: ranks={self.ranks}, "
                f"device_ids={self.device_ids}")
        return tuple(sorted(self.device_ids))

    @property
    def ep_rank_at_mesh_index(self) -> tuple[int, ...]:
        """The EP rank whose expert block sits at each mesh index."""
        ep_rank_of_device = {d: r for r, d in enumerate(self.device_ids)}
        return tuple(ep_rank_of_device[d] for d in self.mesh_device_ids)

    @property
    def mesh_index(self) -> int:
        """This rank's index into that mesh -- what the kernel's own
        ``lax.axis_index`` returns, and the peer coordinate others address."""
        return self.mesh_device_ids.index(self.device_ids[self.rank_in_group])

    @property
    def mesh_order_is_ep_rank_order(self) -> bool:
        return self.ep_rank_at_mesh_index == tuple(range(self.world_size))


_LAYOUT_CACHE: dict[tuple[int, tuple[int, ...], int, int], EpGroupLayout] = {}
_MESH_CACHE: dict[tuple[tuple[str, ...], tuple[int, ...]], Any] = {}


def get_ep_group() -> Any | None:
    """Return vLLM's EP group, or None before distributed init / for dense."""
    try:
        from vllm.distributed.parallel_state import get_ep_group as _get_group

        return _get_group()
    except (AssertionError, ImportError, AttributeError):
        return None


def get_ep_group_layout() -> EpGroupLayout:
    """Rank order of the EP group and the TPU device id each rank drives."""

    group = get_ep_group()
    global_rank = _get_current_global_rank()
    device_id = _get_tpu_global_device_id()
    if group is None or int(group.world_size) == 1:
        return EpGroupLayout(ranks=(global_rank, ),
                             device_ids=(device_id, ),
                             rank_in_group=0,
                             world_size=1)

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
            "Could not map all EP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}")
    layout = EpGroupLayout(
        ranks=ranks,
        device_ids=tuple(rank_to_device_id[rank] for rank in ranks),
        rank_in_group=rank_in_group,
        world_size=int(group.world_size),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def get_or_create_ep_mesh(axis_name: str = EP_AXIS_NAME) -> Any:
    """Build the ``(1, ep_size)`` JAX mesh in ascending device-id order."""
    return _get_or_create_mesh((EP_MESH_DATA_AXIS, axis_name))


def get_or_create_ep_mesh_1d(axis_name: str = EP_AXIS_NAME_1D) -> Any:
    """The same devices in the same order, as the ``(ep_size,)`` mesh v2 takes."""
    return _get_or_create_mesh((axis_name, ))


def _get_or_create_mesh(axis_names: tuple[str, ...]) -> Any:
    """Device-id-ordered mesh whose LAST axis is the expert-parallel one."""

    layout = get_ep_group_layout()
    cache_key = (axis_names, layout.mesh_device_ids)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import jax
    import numpy as np
    from jax.sharding import AxisType, Mesh

    devices_by_id = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    device_ids = layout.mesh_device_ids
    missing = [
        device_id for device_id in device_ids if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(
            "Cannot build EP mesh because JAX does not expose all EP TPU "
            f"device ids: missing={missing}, layout={layout}, "
            f"jax_device_ids={sorted(devices_by_id)}")

    # Ascending order buys positional addressing only when the ids *are* the
    # positions: a Mosaic peer coordinate resolves against the runtime's device
    # list, so mesh index i reaches device id i whatever this mesh says. A group
    # on some other set of devices -- a sub-slice, or two groups sharing a node
    # -- cannot be fixed by reordering, and the expert reindex does not reach it
    # either: that moves ownership between *ranks*, not between devices.
    if device_ids != tuple(range(layout.world_size)):
        raise NotImplementedError(
            "EP mesh needs the group's TPU global device ids to be "
            f"0..{layout.world_size - 1}; got {sorted(layout.device_ids)}. Mesh "
            "index i must be device id i, because a Mosaic kernel resolves a "
            "MESH peer coordinate positionally against the device list.")
    shape = (1, ) * (len(axis_names) - 1) + (layout.world_size, )
    mesh_devices = np.asarray(
        [devices_by_id[device_id] for device_id in device_ids]).reshape(shape)
    # Explicit axes keep the shard_map out-spec on the exported output aval,
    # which is what torch_tpu's `torch_placeholder` divides by to size the torch
    # buffer (pallas.py:get_local_shape). Under the default Auto axes
    # `jax.export` records the output as replicated and the buffer comes back
    # ep_size times too large -- measured on jax 0.10.2 in
    # tests/kernels/fused_moe/test_fused_ep_aot.py.
    mesh = Mesh(mesh_devices,
                axis_names=axis_names,
                axis_types=(AxisType.Explicit, ) * len(axis_names))
    _MESH_CACHE[cache_key] = mesh
    return mesh


def validate_ep_mesh(layer, *, reindexes_expert_order: bool = False) -> None:
    """Fail loudly if vLLM's expert ownership and the mesh order can disagree.

    Expert block ``r`` lives on EP rank ``r`` (``moe_routing.get_experts_start``)
    while the mesh is in device-id order, so the shard at mesh index ``i`` holds
    block ``ep_rank_at_mesh_index[i]``. Pass ``reindexes_expert_order=True`` if
    the caller carries that permutation into the kernel; a kernel that reads
    ownership off its own axis index must not run on a permuted node at all,
    because the rows land on the wrong experts and no shape or row-order test
    can see it.

    Nothing upstream asserts ``moe_config.ep_rank ==
    get_ep_group().rank_in_group``; they agree by construction, and a mismatch
    would silently route rows to the wrong experts.
    """
    group = get_ep_group()
    if group is None:
        return
    moe = layer.moe_config
    if int(group.rank_in_group) != int(moe.ep_rank):
        raise RuntimeError("EP mesh order would not match expert ownership: "
                           f"ep_group.rank_in_group={group.rank_in_group} != "
                           f"moe_config.ep_rank={moe.ep_rank}")
    if int(group.world_size) != int(moe.ep_size):
        raise RuntimeError(f"ep_group.world_size={group.world_size} != "
                           f"moe_config.ep_size={moe.ep_size}")
    layout = get_ep_group_layout()
    if not reindexes_expert_order and not layout.mesh_order_is_ep_rank_order:
        raise NotImplementedError(
            "This MoE kernel takes expert ownership from its mesh index, and "
            "the mesh is in device-id order: the node drives EP ranks "
            f"{layout.ranks} from device ids {layout.device_ids}, which puts EP "
            f"ranks {layout.ep_rank_at_mesh_index} at mesh indices "
            f"0..{layout.world_size - 1}.")
    if not (torch.distributed.is_available()
            and torch.distributed.is_initialized()):
        return
    world_size = torch.distributed.get_world_size()
    if list(group.ranks) != list(range(world_size)):
        # ep_size == tp*dp does not imply this: pp=2,dp=2,tp=2 satisfies it
        # while EP splits into [[0,1,4,5],[2,3,6,7]] and ep_rank != global rank.
        raise NotImplementedError(
            "Fused EP MoE assumes the EP group spans the whole job (pp=1, "
            f"external_dp=1); got ranks={list(group.ranks)}, "
            f"world_size={world_size}.")


def ep_rank_index_tensor(device: torch.device) -> torch.Tensor:
    """This rank's index in the EP group, as 0-d int32 tensor *data*.

    Never bind it as a Python constant into a traced program: ranks would then
    compile structurally different programs under one op name and the in-graph
    collectives desync (see ``layers/vllm/moe_routing.py`` and fa8faaf5).
    """
    return torch.tensor(get_ep_group_layout().rank_in_group,
                        dtype=torch.int32,
                        device=device)
