# SPDX-License-Identifier: Apache-2.0
"""Static per-layer expert relabelling for balanced contiguous EP blocks.

The TPU fused-MoE kernel needs a **contiguous-block** expert map: it derives
this shard's first global expert id from a formula
(``moe_routing.get_experts_start``) rather than reading a per-expert map, and
``validate_linear_ep_placement`` rejects any non-linear placement outright.
That rules out EPLB-style re-placement, which works by handing the kernel an
arbitrary expert->rank map.

Relabelling sidesteps the constraint. Pick a permutation pi of the expert ids,
per layer, and apply it at weight load:

* physical (checkpoint) expert ``g`` is stored in slot ``pi[g] % local`` on
  rank ``pi[g] // local`` -- expressed as a permuted ``expert_map``, which is
  what the weight loader consults;
* row ``pi[g]`` of the router gate gets physical expert ``g``'s row -- and so
  does entry ``pi[g]`` of any per-expert routing bias, e.g. the
  ``e_score_correction_bias`` that ``noaux_tc`` adds before top-k.

The two moves cancel: for logit index ``n`` the kernel picks local slot
``n - local * ep_rank`` on rank ``n // local``, which holds physical expert
``pi^-1[n]`` -- exactly the expert whose gate row now sits at ``n``. So the
model computes the identical function, with the same top-k set and the same
weights, and the blocks are contiguous as the kernel requires. There is no
runtime cost and no kernel change; the only thing that moves is which rank
owns which expert.

Choosing pi so the 8 contiguous blocks carry equal load is then an offline
balanced-partition problem over measured expert load
(``layers/common/expert_load.py`` produces the histogram).

``pi = identity`` must be a bit-exact no-op. That is the cheapest correctness
smoke test available for this path and it is worth running before trusting a
real permutation.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

import vllm_torchtpu.envs as envs

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

_CACHE: dict[str, Any] = {"loaded": False, "perm": None, "meta": None}


def uri() -> str:
    return envs.TPU_MOE_EXPERT_PERM_URI


def enabled() -> bool:
    return bool(uri())


def _read(u: str) -> str:
    if u.startswith("gs://"):
        import subprocess
        out = subprocess.run(["gcloud", "storage", "cat", u],
                             check=True,
                             capture_output=True)
        return out.stdout.decode()
    with open(u) as f:
        return f.read()


def _perms() -> dict[int, np.ndarray] | None:
    """``{layer_index: int64[num_experts]}`` mapping physical id -> new id."""
    if _CACHE["loaded"]:
        return _CACHE["perm"]
    _CACHE["loaded"] = True
    u = uri()
    if not u:
        return None
    doc = json.loads(_read(u))
    raw = doc["perm"]
    perm: dict[int, np.ndarray] = {}
    for key, values in (raw.items() if isinstance(raw, dict) else enumerate(
            raw)):
        arr = np.asarray(values, dtype=np.int64)
        idx = int(key)
        if arr.ndim != 1:
            raise ValueError(f"perm[{idx}] is not 1-D")
        if not np.array_equal(np.sort(arr), np.arange(arr.shape[0])):
            raise ValueError(
                f"perm[{idx}] is not a permutation of 0..{arr.shape[0] - 1}")
        perm[idx] = arr
    _CACHE["perm"] = perm
    _CACHE["meta"] = {k: v for k, v in doc.items() if k != "perm"}
    logger.info(
        "Expert permutation loaded from %s: %d layers, %d experts, meta=%s", u,
        len(perm), next(iter(perm.values())).shape[0] if perm else 0,
        _CACHE["meta"])
    return perm


def _layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name or "")
    return int(m.group(1)) if m else None


def permuted_expert_map(perm: np.ndarray, local_num_experts: int,
                        ep_rank: int) -> torch.Tensor:
    """Global->local slot map placing physical ``g`` at new id ``perm[g]``."""
    new_ids = perm
    owner = new_ids // local_num_experts
    mine = owner == ep_rank
    out = np.full(new_ids.shape[0], -1, dtype=np.int32)
    out[mine] = (new_ids[mine] % local_num_experts).astype(np.int32)
    if int(mine.sum()) != local_num_experts:
        raise ValueError(
            f"permutation gives rank {ep_rank} {int(mine.sum())} experts, "
            f"expected {local_num_experts}")
    return torch.from_numpy(out)


def patch_routed_experts() -> None:
    """Rewrite each MoE layer's ``expert_map`` to the permuted placement.

    Hooks ``update_expert_map_info``, which runs inside ``RoutedExperts``
    ``__init__`` and so lands before any weight is loaded. The weight loader
    reaches the map through ``ExpertMapManager.map_global_to_local``, so both
    copies are replaced.
    """
    if not enabled():
        return
    from vllm.model_executor.layers.fused_moe.routed_experts import \
        RoutedExperts

    if getattr(RoutedExperts, "_tpu_expert_perm_patch", False):
        return
    original = RoutedExperts.update_expert_map_info

    def update_expert_map_info(self) -> None:
        original(self)
        perms = _perms()
        if not perms:
            return
        idx = _layer_index(getattr(self, "layer_name", ""))
        if idx is None or idx not in perms:
            return
        if self._expert_map is None:
            return
        perm = perms[idx]
        if perm.shape[0] != self.global_num_experts:
            raise ValueError(
                f"perm[{idx}] has {perm.shape[0]} entries but the layer has "
                f"{self.global_num_experts} experts")
        ep_rank = self.moe_config.moe_parallel_config.ep_rank
        new_map = permuted_expert_map(perm, self.local_num_experts, ep_rank)
        new_map = new_map.to(self._expert_map.device)
        self.register_buffer("_expert_map", new_map, persistent=False)
        self.expert_map_manager._expert_map = new_map

    RoutedExperts.update_expert_map_info = update_expert_map_info
    RoutedExperts._tpu_expert_perm_patch = True
    logger.info("Applied TPU patch: permuted MoE expert placement from %s",
                uri())


def permute_router_gates(model: torch.nn.Module) -> None:
    """Reorder router gate rows to match the permuted expert placement.

    Runs after weight loading: the gate is a replicated ``[num_experts,
    hidden]`` tensor, so this is a local row shuffle with no communication.
    Row ``n`` must end up holding physical expert ``pi^-1[n]``'s row, which is
    exactly ``weight[argsort(pi)]``.
    """
    if not enabled():
        return
    perms = _perms()
    if not perms:
        return
    touched = 0
    permuted_score_bias = 0
    for name, module in model.named_modules():
        if not name.endswith(".mlp.gate"):
            continue
        idx = _layer_index(name)
        if idx is None or idx not in perms:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        perm = perms[idx]
        if weight.shape[0] != perm.shape[0]:
            raise ValueError(
                f"{name}.weight has {weight.shape[0]} rows but perm[{idx}] "
                f"has {perm.shape[0]} entries")
        inverse = torch.from_numpy(np.argsort(perm)).to(weight.device)
        with torch.no_grad():
            weight.data = weight.data.index_select(0, inverse).contiguous()
        bias = getattr(module, "bias", None)
        if bias is not None and bias.shape[0] == perm.shape[0]:
            with torch.no_grad():
                bias.data = bias.data.index_select(0, inverse).contiguous()
        # `noaux_tc` routing (DeepSeek-V3/GLM-5.2) adds a per-expert
        # `e_score_correction_bias` to the scores *before* top-k
        # (`moe_routing.py`), so it is indexed by expert id exactly like the
        # gate rows and must move with them -- otherwise the relabelling stops
        # being a no-op and genuinely changes which experts are selected.
        # It lives on the gate module but the experts layer captured the same
        # Parameter object at construction, so mutate `.data` in place; a
        # rebind would leave the experts layer on the unpermuted tensor.
        score_bias = getattr(module, "e_score_correction_bias", None)
        if score_bias is not None:
            if score_bias.shape[0] != perm.shape[0]:
                raise ValueError(
                    f"{name}.e_score_correction_bias has {score_bias.shape[0]} "
                    f"entries but perm[{idx}] has {perm.shape[0]}")
            with torch.no_grad():
                score_bias.data = score_bias.data.index_select(
                    0, inverse).contiguous()
            permuted_score_bias += 1
        touched += 1
    # This module's `logger.info` does not reach the job log; print so the
    # per-layer counts are actually observable in a benchmark run.
    print(
        f"[expert_permutation] permuted router gate rows on {touched} MoE "
        f"layers ({permuted_score_bias} with e_score_correction_bias)",
        flush=True)
    logger.info("Permuted router gate rows on %d MoE layers", touched)
    if touched != len(perms):
        raise ValueError(
            f"permutation file covers {len(perms)} layers but {touched} "
            "router gates were found; refusing to run a half-permuted model")
