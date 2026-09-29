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
"""TPU_EXPERT_PLACEMENT=<json>: a per-layer expert placement for expert
parallelism, applied as a permutation of the checkpoint at load time.

The file says, per MoE layer, which global experts each expert-parallel rank
holds and in what slot order: `layers[<n>].cores[r]` is a list of
`slots_per_core` global expert ids, and the primary slots hold every expert exactly once. Optional copy slots
follow the primaries; the small-batch MoE kernel assigns each active expert
to exactly one holder. vLLM's EPLB policy run on a
routing recording produces such a file. Off by default (no file): the
checkpoint's own contiguous placement, nothing touched.

What it changes, and where:
  * the layer's `expert_map` (vLLM's global-to-local slot map) is this rank's
    slot index of each global expert, so expert `cores[r][j]` lands in slot
    j. The kernels' contiguous-block assumption (rank r holds virtual ids
    r * owned .. (r + 1) * owned - 1; virtual id v = r * owned + j for primary slot j
    of rank r) then holds by construction;
  * the loader's expert filter keeps the experts the placement gives this
    rank (model_loader_patches);
  * the router's rows (one per expert) are reordered into the same virtual
    order at load, so the top-k it produces already names slots; nothing
    runs per token.

Numerics: each token selects the same experts with the same weights and
runs the same expert weights; only the order the k expert contributions are
summed in the combine can differ (rounding level).
"""

import functools
import json
import re

from vllm_torchtpu import envs as _envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DATA: dict | None = None
_DATA_PATH = ""
_LOGGED: set[str] = set()
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_ROUTER_RE = re.compile(r"\.mlp\.gate\.(weight|bias)$|e_score_correction_bias$")


def _path() -> str:
    return _envs.TPU_EXPERT_PLACEMENT


def enabled() -> bool:
    """Whether a placement table is set (TPU_EXPERT_PLACEMENT)."""
    return bool(_path())


def _log_once(key: str, message: str) -> None:
    """One line per process per event kind, so a boot log shows the
    placement was applied."""
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info(message)


def data() -> dict:
    """The placement table, loaded once per path and checked to be a
    primary permutation and optional copies."""
    global _DATA, _DATA_PATH
    path = _path()
    if _DATA is None or path != _DATA_PATH:
        with open(path) as f:
            table = json.load(f)
        experts = int(table["experts"])
        ranks = int(table["shards"])
        per_rank = int(table["slots_per_core"])
        # a copies table: the first owned_per_core slots of every rank are a
        # permutation of the experts, the slots after them hold copies
        copies = bool(table.get("copies", False))
        owned_n = int(table.get("owned_per_core", per_rank)) if copies else per_rank
        if ranks * owned_n != experts or owned_n > per_rank:
            raise ValueError(
                f"expert placement {path}: {ranks} ranks x {owned_n} owned slots "
                f"must equal {experts} experts (slots per rank {per_rank})"
            )
        for n, layer in table["layers"].items():
            cores = layer["cores"]
            if len(cores) != ranks or any(len(c) != per_rank for c in cores):
                raise ValueError(
                    f"expert placement {path}: layer {n} must list {ranks} "
                    f"ranks x {per_rank} slots"
                )
            if sorted(e for c in cores for e in c[:owned_n]) != list(range(experts)):
                raise ValueError(
                    f"expert placement {path}: layer {n} must hold every "
                    "expert exactly once among the owned slots"
                )
            if copies and any(e in c[:owned_n] for c in cores for e in c[owned_n:]):
                raise ValueError(
                    f"expert placement {path}: layer {n} holds an expert as a "
                    "copy on the rank that owns it"
                )
            if copies:
                copy_slots = [e for c in cores for e in c[owned_n:]]
                if any(e < 0 or e >= experts for e in copy_slots):
                    raise ValueError(
                        f"expert placement: layer {n} has an invalid copy id"
                    )
                if len(copy_slots) != len(set(copy_slots)):
                    raise ValueError(
                        f"expert placement: layer {n} allows at most one extra copy per expert"
                    )
                declared = layer.get("copies")
                if isinstance(declared, dict):
                    actual = {
                        str(e): r for r, c in enumerate(cores) for e in c[owned_n:]
                    }
                    if declared != actual:
                        raise ValueError(
                            f"expert placement: layer {n} copy order must match the copy slots"
                        )
        _DATA, _DATA_PATH = table, path
    return _DATA


def layer_index(name: str) -> int | None:
    """The MoE layer index in a parameter name or prefix, or None."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def shards() -> int:
    """The number of expert-parallel ranks the table was built for."""
    return int(data()["shards"])


def slots() -> int:
    """The number of expert slots each rank holds."""
    return int(data()["slots_per_core"])


def global_count() -> int:
    """The number of global experts per layer."""
    return int(data()["experts"])


def cores(layer: int) -> list[list[int]]:
    """Per rank, the global experts it holds for the layer, in slot order."""
    return data()["layers"][str(layer)]["cores"]


@functools.cache
def _local_ids(path: str, layer: int, rank: int) -> frozenset[int]:
    del path  # part of the key: a new table gives new sets
    return frozenset(cores(layer)[rank])


def local_ids(layer: int, rank: int) -> frozenset[int]:
    """The global experts rank r holds for this layer."""
    return _local_ids(_path(), layer, rank)


def expert_map(layer: int, rank: int) -> list[int]:
    """vLLM's expert_map for this layer on this rank: the local slot of each
    global expert, -1 when absent."""
    m = [-1] * global_count()
    for j, e in enumerate(cores(layer)[rank]):
        m[e] = j
    return m


def virtual_order(layer: int) -> list[int]:
    """The global expert of each virtual id v = r * owned + j (the router's
    row order after the permutation): every rank's primaries, copy slots
    left out."""
    o = owned()
    return [e for c in cores(layer) for e in c[:o]]


def has_copies() -> bool:
    """Whether the placement holds experts on more than one rank (copy slots
    after each rank's owned primaries)."""
    return enabled() and bool(data().get("copies", False))


def owned() -> int:
    """The primaries per rank: the slots the grouped path and the router
    permutation know; copy slots follow them."""
    return int(data().get("owned_per_core", slots()))


def holder_tables(layer: int, rank: int):
    """For the few-token kernel's scores entry, in VIRTUAL ids (the router's
    row order after the permutation: rank r's primaries are v = r*owned ..
    (r+1)*owned - 1): this rank's slot -> virtual id table (primaries first,
    then its copy slots), and per virtual id the rank of the primary (v //
    owned, by construction) and of the copy (-1 without one)."""
    cs = cores(layer)
    o = owned()
    vid = {}  # global expert -> virtual id
    for r, lst in enumerate(cs):
        for j, e in enumerate(lst[:o]):
            vid[e] = r * o + j
    n = len(vid)
    prim = [v // o for v in range(n)]
    copy = [-1] * n
    for r, lst in enumerate(cs):
        for e in lst[o:]:
            copy[vid[e]] = r
    gid = [vid[e] for e in cs[rank]]
    return gid, prim, copy


def hot_order(layer: int) -> list[int]:
    """The copied experts of a layer as VIRTUAL ids, hottest first: the placement file's per-layer `copies`
    (an insertion-ordered mapping, written hottest first), else the copy slots in core order."""
    cs = cores(layer)
    o = owned()
    vid = {}
    for r, lst in enumerate(cs):
        for j, e in enumerate(lst[:o]):
            vid[e] = r * o + j
    entry = data()["layers"][str(layer)]
    if isinstance(entry.get("copies"), dict):
        return [vid[int(e)] for e in entry["copies"]]
    return [vid[e] for lst in cs for e in lst[o:]]


def permute_router_rows(name: str, tensor):
    """The router's weight (rows = experts) or per-expert bias reordered into
    the virtual order at load. Returns the tensor unchanged when the name is
    not a router parameter of a placed layer."""
    if not enabled() or not _ROUTER_RE.search(name):
        return tensor
    layer = layer_index(name)
    if layer is None or str(layer) not in data()["layers"]:
        return tensor
    order = virtual_order(layer)
    if tensor.shape[0] != global_count():
        raise ValueError(
            f"expert placement: {name} has {tensor.shape[0]} rows, the table "
            f"{global_count()} experts"
        )
    import torch

    _log_once(
        "router",
        "[expert placement] router rows reordered into the virtual "
        f"order at load: {name} {tuple(tensor.shape)}",
    )
    return tensor[
        torch.tensor(order, dtype=torch.long, device=tensor.device)
    ].contiguous()


# ---------------------------------------------------------------------------
# Model construction: each MoE layer gets its own expert map
# ---------------------------------------------------------------------------
_CURRENT_PREFIX = ""


def install_model_patches() -> None:
    """Give every MoE layer the placement's expert map for its layer (slot
    index per global expert, -1 when absent) and its slot count, by wrapping
    vLLM's layer factory (which knows the layer's prefix) and the map
    computation it does inside; reorder the router rows in the loaders'
    weight streams. A no-op without TPU_EXPERT_PLACEMENT."""
    if not enabled():
        return
    import inspect
    import sys

    import torch
    from vllm.model_executor.layers import fused_moe as fused_pkg
    from vllm.model_executor.layers.fused_moe import expert_map_manager as emm
    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    if getattr(fused_layer, "_tpu_placement_patch", False):
        return
    original_factory = fused_layer.FusedMoEFactory
    original_calc = emm.ExpertMapManager._calculate_expert_maps
    signature = inspect.signature(original_factory)

    def FusedMoEFactory(*args, **kwargs):
        global _CURRENT_PREFIX
        bound = signature.bind_partial(*args, **kwargs)
        _CURRENT_PREFIX = str(bound.arguments.get("prefix", "") or "")
        try:
            return original_factory(*args, **kwargs)
        finally:
            _CURRENT_PREFIX = ""

    def _calculate_expert_maps(self):
        original_calc(self)
        if not self.moe_parallel_config.use_ep:
            return
        n = layer_index(_CURRENT_PREFIX)
        if n is None:
            raise RuntimeError(
                "expert placement: an expert map is computed outside the "
                "layer factory (no layer prefix), so the placement cannot "
                "be applied"
            )
        if str(n) not in data()["layers"]:
            raise RuntimeError(
                f"expert placement: the table has no layer {n} (prefix "
                f"{_CURRENT_PREFIX})"
            )
        if shards() != int(self.ep_size) or global_count() != int(
            self.global_num_experts
        ):
            raise ValueError(
                f"expert placement: the table is for {shards()} ranks x "
                f"{global_count()} experts; this layer has {self.ep_size} "
                f"ranks x {self.global_num_experts}"
            )
        rank = int(self.ep_rank)
        self._local_num_experts = slots() + self.num_fused_shared_experts
        self._expert_map = torch.tensor(expert_map(n, rank), dtype=torch.int32)
        if self._expert_mask is not None:
            self._expert_mask = (self._expert_map >= 0).to(self._expert_mask.dtype)
        _log_once(
            "map",
            f"[expert placement] {_path()}: layer {n} rank {rank} "
            f"holds {slots()} slots; first slots {cores(n)[rank][:4]}",
        )

    # the factory by every name it is already bound to (the package export
    # and any model module that imported it)
    fused_layer.FusedMoEFactory = FusedMoEFactory
    fused_pkg.FusedMoEFactory = FusedMoEFactory
    for mod in list(sys.modules.values()):
        if getattr(mod, "FusedMoEFactory", None) is original_factory:
            mod.FusedMoEFactory = FusedMoEFactory
    emm.ExpertMapManager._calculate_expert_maps = _calculate_expert_maps
    fused_layer._tpu_placement_patch = True

    # The router rows reordered in both loaders' weight streams: vLLM's
    # default safetensors loader and the streaming loader (the gs:// path,
    # whose per-rank iterator main's loader patches install first).
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    original_get_all = dl.DefaultModelLoader.get_all_weights

    def get_all_weights(self, model_config, model):
        for name, tensor in original_get_all(self, model_config, model):
            yield name, permute_router_rows(name, tensor)

    dl.DefaultModelLoader.get_all_weights = get_all_weights
    original_streamed = rsl.RunaiModelStreamerLoader._get_weights_iterator

    def _get_weights_iterator(self, *args, **kwargs):
        for name, tensor in original_streamed(self, *args, **kwargs):
            yield name, permute_router_rows(name, tensor)

    rsl.RunaiModelStreamerLoader._get_weights_iterator = _get_weights_iterator
