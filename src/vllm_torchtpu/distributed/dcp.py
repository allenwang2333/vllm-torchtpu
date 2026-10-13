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

import sys
from typing import Any

from vllm_torchtpu.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
    get_or_create_cp_mesh,
)

# The JAX mesh axis the DCP kernels shard the KV cache over.
DCP_AXIS_NAME = "dcp"

_SYS_MODULES = sys.modules
_vllm_get_dcp_group = None
_vllm_get_dcp_group_resolved = False

_cached_group: Any = None
_cached_rank: int = 0
_cached_world_size: int = 1
_cached_layout_group: Any = None
_cached_layout_fn: Any = None
_cached_layout_val: Any = None
_cached_mesh_group: Any = None
_cached_mesh_fn: Any = None
_cached_mesh_axis: str | None = None
_cached_mesh_val: Any = None


def get_dcp_group() -> Any | None:
    """Return vLLM's DCP GroupCoordinator, or None if not initialized."""
    global _vllm_get_dcp_group, _vllm_get_dcp_group_resolved
    if not _vllm_get_dcp_group_resolved:
        _ps = _SYS_MODULES.get("vllm.distributed.parallel_state")
        if _ps is not None:
            _vllm_get_dcp_group = getattr(_ps, "get_dcp_group", None)
        else:
            try:
                from vllm.distributed.parallel_state import (
                    get_dcp_group as _get_group,
                )

                _vllm_get_dcp_group = _get_group
            except (AssertionError, ImportError):
                _vllm_get_dcp_group = None
        _vllm_get_dcp_group_resolved = True

    getter = _vllm_get_dcp_group
    if getter is not None:
        try:
            return getter()
        except (AssertionError, ImportError):
            _vllm_get_dcp_group = None
            return None
    return None


def get_dcp_rank() -> int:
    global _cached_group, _cached_rank, _cached_world_size
    group = get_dcp_group()
    if group is None:
        return 0
    if group is _cached_group:
        return _cached_rank
    rank = int(group.rank_in_group)
    _cached_group = group
    _cached_rank = rank
    _cached_world_size = int(group.world_size)
    return rank


def get_dcp_world_size() -> int:
    global _cached_group, _cached_rank, _cached_world_size
    group = get_dcp_group()
    if group is None:
        return 1
    if group is _cached_group:
        return _cached_world_size
    world_size = int(group.world_size)
    _cached_group = group
    _cached_rank = int(group.rank_in_group)
    _cached_world_size = world_size
    return world_size


def get_dcp_group_layout() -> CpGroupLayout:
    """The DCP group's rank order and the TPU device each rank sits on."""
    global _cached_layout_group, _cached_layout_fn, _cached_layout_val
    group = get_dcp_group()
    fn = get_cp_group_layout
    if group is not None and group is _cached_layout_group and fn is _cached_layout_fn:
        return _cached_layout_val
    layout = fn(group)
    _cached_layout_group = group
    _cached_layout_fn = fn
    _cached_layout_val = layout
    return layout


def get_or_create_dcp_mesh(axis_name: str | None = DCP_AXIS_NAME) -> Any:
    """Build the op-local JAX mesh for the DCP axis.

    Under `tp=N, dcp=N` the DCP group covers the same devices as the TP group,
    and the mesh cache is keyed on the axis name as well as the device ids, so
    a run with both CP axes live keeps two distinct meshes rather than
    aliasing them.

    The axis name defaults to `DCP_AXIS_NAME`; the kernels take the same
    constant from here, so a caller that overrides it has to override it in
    both places or the mesh and the partition specs will not agree.
    """
    global _cached_mesh_group, _cached_mesh_fn, _cached_mesh_axis, _cached_mesh_val
    name = DCP_AXIS_NAME if axis_name is None else axis_name
    group = get_dcp_group()
    fn = get_or_create_cp_mesh
    if (
        group is not None
        and group is _cached_mesh_group
        and fn is _cached_mesh_fn
        and name == _cached_mesh_axis
    ):
        return _cached_mesh_val
    mesh = fn(axis_name=name, group=group)
    _cached_mesh_group = group
    _cached_mesh_fn = fn
    _cached_mesh_axis = name
    _cached_mesh_val = mesh
    return mesh
