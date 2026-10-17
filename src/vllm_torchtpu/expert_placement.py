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
"""TPU_EXPERT_PLACEMENT=<json>: a static expert placement for expert parallelism.

The file names, per MoE layer, which global experts each EP rank holds and in
what slot order:

    {"experts": 384, "shards": 16,
     "layers": {"3": [[17, 250, ...24 ids...], ...16 ranks...], ...}}

Every expert of a listed layer is held exactly once (a permutation, no
copies). Layers that are not listed keep the checkpoint's contiguous blocks.

What follows from it, all decided at load:
  * the layer's expert map puts expert `ranks[r][j]` in slot j of rank r, so
    vLLM's weight loader and the fetch-time expert filter load exactly those;
  * `expert_slot_table` maps each global expert to its SLOT id,
    `r * slots + j`, which is the contiguous-block numbering the fused EP
    kernel addresses. The kernel applies it to the selected ids after top-k,
    so selection, tie-breaking, the router weights and the order of the k
    combine terms are those of the unplaced model: per token the output is
    the same bits, only where each expert's matmuls run moves.

Only the fused EP kernel reads slot ids, so a placed layer must be armed for
it; `check_layer_served` refuses anything else.
"""

import functools
import json
import re
from dataclasses import dataclass

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `model.layers.7.ffn.experts` (a module prefix) and `layers.7.ffn.experts.3.w1`
# (a checkpoint tensor) both name layer 7. A drafter's `mtp.0.` does not match.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# The layer attribute the slot table is registered under.
EXPERT_SLOT_ATTR = "_tpu_expert_slot"

# Layer indices whose expert map follows the placement in this process. The
# fetch-time filter consults the placement only for these, so a model whose
# MoE method never applied it keeps the default filter.
_APPLIED_LAYERS: set[int] = set()


@dataclass(frozen=True)
class Placement:
    experts: int
    shards: int
    layers: dict[int, tuple[tuple[int, ...], ...]]

    @property
    def slots(self) -> int:
        return self.experts // self.shards

    def ranks(self, layer: int) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order; None if unlisted."""
        return self.layers.get(layer)

    def slot_ids(self, layer: int) -> list[int]:
        """slot_ids[e] = r * slots + j for the rank r and slot j holding e."""
        ranks = self.layers[layer]
        table = [0] * self.experts
        for r, held in enumerate(ranks):
            for j, e in enumerate(held):
                table[e] = r * self.slots + j
        return table


def parse(table: dict, source: str = "<dict>") -> Placement:
    """Validate a placement table: every listed layer is a permutation."""
    experts = int(table["experts"])
    shards = int(table["shards"])
    if experts % shards:
        raise ValueError(
            f"expert placement {source}: {experts} experts do not divide over "
            f"{shards} ranks"
        )
    slots = experts // shards
    layers = {}
    for key, ranks in table["layers"].items():
        if len(ranks) != shards or any(len(held) != slots for held in ranks):
            raise ValueError(
                f"expert placement {source}: layer {key} must list {shards} "
                f"ranks of {slots} experts"
            )
        flat = sorted(int(e) for held in ranks for e in held)
        if flat != list(range(experts)):
            raise ValueError(
                f"expert placement {source}: layer {key} must hold every one "
                f"of the {experts} experts exactly once"
            )
        layers[int(key)] = tuple(tuple(int(e) for e in held) for held in ranks)
    return Placement(experts=experts, shards=shards, layers=layers)


@functools.cache
def _load(path: str) -> Placement:
    with open(path) as f:
        return parse(json.load(f), path)


def current() -> Placement | None:
    """The process's placement, or None when TPU_EXPERT_PLACEMENT is unset."""
    path = envs.TPU_EXPERT_PLACEMENT
    return _load(path) if path else None


def layer_index(name: str) -> int | None:
    """The decoder layer a module prefix or checkpoint tensor name belongs to."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def placed_ranks(name: str) -> tuple[tuple[int, ...], ...] | None:
    """The placement of the layer `name` belongs to, or None if it has none."""
    placement = current()
    layer = layer_index(name)
    if placement is None or layer is None:
        return None
    return placement.ranks(layer)


def has_applied_layers() -> bool:
    return bool(_APPLIED_LAYERS)


def fetch_filter_ids(name: str, ep_rank: int) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for this tensor's layer.

    None defers to the default contiguous filter: either there is no
    placement for the layer, or no MoE method applied one to it.
    """
    layer = layer_index(name)
    if layer is None or layer not in _APPLIED_LAYERS:
        return None
    ranks = current().ranks(layer)
    return frozenset(ranks[ep_rank])


def apply_to_layer(layer) -> bool:
    """Point a RoutedExperts layer's expert map at its placement.

    Must run before the weights load (from the MoE method's create_weights).
    Registers the slot table the fused EP op takes. Returns whether the layer
    is placed.
    """
    import torch

    placement = current()
    ranks = placed_ranks(layer.layer_name)
    if ranks is None:
        return False
    manager = layer.expert_map_manager
    if manager.global_num_experts != placement.experts or manager.ep_size != (
        placement.shards
    ):
        raise ValueError(
            f"expert placement {envs.TPU_EXPERT_PLACEMENT} is for "
            f"{placement.shards} ranks x {placement.experts} experts; "
            f"{layer.layer_name} has {manager.ep_size} x "
            f"{manager.global_num_experts}"
        )
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        raise ValueError(
            f"expert placement: {layer.layer_name} fuses shared experts into "
            "its expert map, which a placement does not describe"
        )
    _refuse_drafter_layer(layer.layer_name)
    rank = manager.ep_rank
    expert_map = torch.full((placement.experts,), -1, dtype=torch.int32)
    expert_map[list(ranks[rank])] = torch.arange(placement.slots, dtype=torch.int32)
    manager._expert_map = expert_map.to(manager.expert_map.device)
    layer.update_expert_map_info()
    layer.register_buffer(
        EXPERT_SLOT_ATTR,
        torch.tensor(
            placement.slot_ids(layer_index(layer.layer_name)),
            dtype=torch.int32,
            device=manager.expert_map.device,
        ),
        persistent=False,
    )
    _APPLIED_LAYERS.add(layer_index(layer.layer_name))
    logger.info_once(
        "Expert placement %s applied: rank %d of %d, %d placed layers; "
        "first placed layer holds %s",
        envs.TPU_EXPERT_PLACEMENT,
        rank,
        placement.shards,
        len(placement.layers),
        ranks[rank][:6],
    )
    return True


def _refuse_drafter_layer(name: str) -> None:
    """A drafter's MoE layers are numbered after the target's (`layers.61.` on
    a 61-layer model) but load from `mtp.*` tensors, which the fetch filter
    cannot place, so the placed slots would hold nothing."""
    from vllm.config import get_current_vllm_config

    target_layers = get_current_vllm_config().model_config.hf_config.num_hidden_layers
    if layer_index(name) >= target_layers:
        raise ValueError(
            f"expert placement lists {name}, a drafter layer (the target has "
            f"{target_layers} decoder layers); only target layers can be placed"
        )


def expert_slot_table(layer):
    """The [experts] i32 slot table of a placed layer, or None."""
    return getattr(layer, EXPERT_SLOT_ATTR, None)


def check_layer_served(layer, fused_armed: bool) -> None:
    """A placed layer's weights sit in slot order only the fused op undoes."""
    if expert_slot_table(layer) is not None and not fused_armed:
        raise RuntimeError(
            f"expert placement covers {layer.layer_name}, but the fused EP "
            "MoE kernel is not armed for it; the grouped-matmul path reads "
            "contiguous expert blocks and would run the wrong experts"
        )
