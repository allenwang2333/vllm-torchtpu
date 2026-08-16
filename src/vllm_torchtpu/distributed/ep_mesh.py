# SPDX-License-Identifier: Apache-2.0
"""JAX mesh over vLLM's expert-parallel process group.

Same mechanism as :mod:`vllm_torchtpu.distributed.pcp`: every rank agrees on an
order for ``jax.devices()`` by gathering each peer's TPU global device id over
the group's CPU process group. A shard_map'd Pallas kernel needs that order to
match EP rank order, because expert ``e`` lives on rank ``e // local_experts``
and the kernel addresses peers by mesh index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.distributed.pcp import (_collect_rank_to_device_id,
                                           _get_current_global_rank,
                                           _get_tpu_global_device_id)

# The kernel takes a 2-D mesh and requires every non-EP axis to have size 1;
# it reads the other axis by this name (kernels/fused_moe/v1/kernel.py:318).
EP_MESH_DATA_AXIS = "data"
EP_AXIS_NAME = "model"


@dataclass(frozen=True)
class EpGroupLayout:
    """EP process-group order plus the matching TPU global device ids."""

    ranks: tuple[int, ...]
    device_ids: tuple[int, ...]
    rank_in_group: int
    world_size: int


_LAYOUT_CACHE: dict[tuple[int, tuple[int, ...], int, int], EpGroupLayout] = {}
_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}


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
    """Build the ``(1, ep_size)`` JAX mesh ordered by EP rank."""

    layout = get_ep_group_layout()
    cache_key = (axis_name, layout.device_ids)
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
    missing = [
        device_id for device_id in layout.device_ids
        if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(
            "Cannot build EP mesh because JAX does not expose all EP TPU "
            f"device ids: missing={missing}, layout={layout}, "
            f"jax_device_ids={sorted(devices_by_id)}")

    mesh_devices = np.asarray([
        devices_by_id[device_id] for device_id in layout.device_ids
    ]).reshape(1, layout.world_size)
    # Explicit axes keep the shard_map out-spec on the exported output aval,
    # which is what torch_tpu's `torch_placeholder` divides by to size the torch
    # buffer (pallas.py:get_local_shape). Under the default Auto axes
    # `jax.export` records the output as replicated and the buffer comes back
    # ep_size times too large -- measured on jax 0.10.2 in
    # tests/kernels/fused_moe/test_fused_ep_aot.py.
    mesh = Mesh(mesh_devices,
                axis_names=(EP_MESH_DATA_AXIS, axis_name),
                axis_types=(AxisType.Explicit, AxisType.Explicit))
    _MESH_CACHE[cache_key] = mesh
    return mesh


def validate_ep_mesh(layer) -> None:
    """Fail loudly if vLLM's expert ownership and the mesh order can disagree.

    Mesh index i must be EP rank i, because expert block i lives on EP rank i
    (``moe_routing.get_experts_start``) and the kernel addresses peers by mesh
    index. Nothing upstream asserts ``moe_config.ep_rank ==
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
