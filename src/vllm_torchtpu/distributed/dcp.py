# SPDX-License-Identifier: Apache-2.0
"""DCP (Decode Context Parallelism) process-group accessors.

Mirrors the role of distributed/pcp.py for PCP: a narrow, safe accessor
layer every call site shares, rather than each re-deriving group state
independently.

Unlike PCP and EP, DCP's current kernel design does two separate per-device
Pallas kernel calls glued by ordinary torch.distributed collectives
(`GroupCoordinator.all_gather`), not a JAX multi-device mesh fused into one
kernel invocation. So this file stays a thin group/rank accessor -- there is
no `PcpGroupLayout` or `get_or_create_*_mesh` equivalent here, because
nothing in the DCP path needs a JAX mesh or a device-id ordering.
"""

from __future__ import annotations

from typing import Any


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
