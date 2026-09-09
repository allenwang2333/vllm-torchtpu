# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU device communicator for EP across data-parallel engines.

Dispatch gathers tokens across DP ranks that share the same TP rank. Combine
sums those matching-TP partials across DP and returns this DP rank's token
slice. vLLM's existing ``reduce_results`` TP all-reduce remains responsible for
the final TP-side reduction. Callers must pass equal token counts on each DP
rank.
"""

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase, DeviceCommunicatorBase)
from vllm.distributed.parallel_state import get_dp_group

# The blocked pool's minor dim. Resolved from the device once, at op-build time; the traced
# path reads the int. `dispatch_router_logits` runs inside the Dynamo-traced region, where a
# nanobind call is untraceable, so the accessor must never be reached from there.
_LANES: list[int] = []


def resolve_hardware() -> None:
    """Read the lane count off the device. Call at op-build time, never while tracing."""
    from jax.experimental.pallas import tpu as pltpu
    try:
        _LANES[:] = [pltpu.get_tpu_info().num_lanes]
        return
    except Exception:  # noqa: BLE001 -- no device yet, or an unsupported kind
        pass
    # Op construction can precede a device. Every generation the compiler knows
    # has the same lane count, so take that; a future split has to be resolved.
    lanes = {
        pltpu.get_tpu_info_for_chip(v, 1).num_lanes
        for v in pltpu.ChipVersion
    }
    if len(lanes) != 1:
        raise RuntimeError(
            f"no device and the lane count is not uniform: {lanes}")
    _LANES[:] = [lanes.pop()]


# Side channel for the lane-blocked MoE pool, set by dispatch_router_logits and
# consumed by the MoE quant method in the SAME traced forward.  A single slot is
# enough because dispatch and apply strictly alternate, once per MoE layer.
_BLOCKED_POOL = [None]


def set_blocked_pool(t) -> None:
    _BLOCKED_POOL[0] = t


# Global token ceiling of the fused permute path, published by the MoE op builder.
_FPG_MAX_TOKENS = [None]


def set_fpg_max_tokens(n: int) -> None:
    """Publish the fused permute path's global-token ceiling (0 = ineligible).

    Blocking a pool that path cannot read is worse than not blocking it: the
    full-pool un-blocking costs a SparseCore data-format call per layer, more
    than the shard-side blocking saved.  The most restrictive layer wins, since
    blocking is decided per step for the whole model while eligibility is a
    per-layer property.
    """
    cur = _FPG_MAX_TOKENS[0]
    _FPG_MAX_TOKENS[0] = n if cur is None else min(cur, n)


def fpg_max_tokens() -> int:
    return _FPG_MAX_TOKENS[0] or 0


def take_blocked_pool():
    """Return the pool stashed by the last dispatch and clear the slot."""
    t, _BLOCKED_POOL[0] = _BLOCKED_POOL[0], None
    return t


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


class TpuDeviceCommunicator(DeviceCommunicatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU performs EP dispatch/combine directly (see dispatch/combine
        # below), not through an all2all manager. The stub only satisfies
        # vLLM's DP>1 + MoE fault-tolerance probe
        # (support_fault_tolerance=False); it is never used for compute.
        if self.use_all2all:
            self.all2all_manager = TpuNullAll2AllManager(self.cpu_group)

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, router_logits

        # A blockable shape is not enough: the fused path has to be able to
        # read it, or the full-pool un-blocking costs more than blocking saved.
        budget = fpg_max_tokens()
        if (budget and hidden_states.dim() == 2
                and hidden_states.shape[-1] % _LANES[0] == 0
                and hidden_states.shape[0] * dp.world_size <= budget):
            # Block the pool on the pre-gather shard, the layout
            # fused_permute_gmm reads rows in: the same SparseCore data-format
            # conversion as after the gather, on 1/dp of the bytes.
            n, k = hidden_states.shape
            blocked = self._dp_gather(
                hidden_states.view(n, k // _LANES[0], _LANES[0]))
            # Side channel: `_maybe_dispatch` unpacks a 2-tuple. The gate above
            # is what makes the returned view unread, so its reshape is dead
            # code and the relayout is not paid.
            set_blocked_pool(blocked)
            hidden_states = blocked.view(blocked.shape[0], k)
        else:
            set_blocked_pool(None)
            hidden_states = self._dp_gather(hidden_states)

        router_logits = self._dp_gather(router_logits)
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = self._dp_gather(hidden_states)
        topk_weights = self._dp_gather(topk_weights)
        topk_ids = self._dp_gather(topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states
        return dp.reduce_scatter(hidden_states, dim=0)
