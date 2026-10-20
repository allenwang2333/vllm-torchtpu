# SPDX-License-Identifier: Apache-2.0
"""Static per-layer expert relabelling for balanced contiguous EP blocks.

The fused EP MoE kernel assigns global experts to EP ranks in contiguous
blocks (rank ``r`` owns ids ``[r * local, (r + 1) * local)``). With the
checkpoint's own numbering, a layer's busiest experts can land on one rank,
and every rank then waits for that rank inside the kernel.

Relabelling moves experts between blocks without touching the kernel. Pick a
permutation ``perm`` per layer: checkpoint expert ``g`` becomes expert
``perm[g]``. It is applied to the checkpoint stream at weight load:

* ``layers.<l>.ffn.experts.<g>.*`` is loaded as expert ``perm[g]``;
* row ``perm[g]`` of the router gate weight, score-correction bias and
  vision bias takes checkpoint row ``g``.

The router then scores expert ``perm[g]`` exactly as it scored ``g``, so it
selects the same experts with the same weights. Each expert computes on the
same rows as before, only on a different rank. ``perm`` = identity changes
nothing.

Enabled by ``TPU_MOE_EXPERT_PERM_PATH``: a JSON file whose ``"perm"`` maps
layer index -> list of new ids, one per checkpoint expert. Layers absent from
the file keep their numbering (the DSpark draft layers are named ``mtp.<i>``
and never match). Nothing here runs inside a compiled forward: the served
program is the same with the switch on or off.
"""

import functools
import json
import re
from collections.abc import Iterable, Iterator

import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The target model's checkpoint names: ``layers.<l>.ffn.(gate|experts.<g>).``,
# optionally under a ``model.`` style prefix.
_EXPERT_RE = re.compile(r"(^|\.)layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_GATE_RE = re.compile(r"(^|\.)layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")
# A hash-routed layer's table holds expert ids as VALUES; relabelling it is
# not implemented, so such a layer is refused rather than silently misrouted.
_HASH_RE = re.compile(r"(^|\.)layers\.(\d+)\.ffn\.gate\.tid2eid$")


def load_permutations(path: str) -> dict[int, np.ndarray]:
    """``{layer: int64[experts]}`` (checkpoint id -> new id) from ``path``."""
    with open(path) as f:
        doc = json.load(f)
    perms = {}
    for key, values in doc["perm"].items():
        perm = np.asarray(values, dtype=np.int64)
        if perm.ndim != 1 or not np.array_equal(
            np.sort(perm), np.arange(perm.shape[0])
        ):
            raise ValueError(f"{path}: perm[{key}] is not a permutation")
        perms[int(key)] = perm
    return perms


@functools.cache
def _configured() -> dict[int, np.ndarray]:
    path = envs.TPU_MOE_EXPERT_PERM_PATH
    if not path:
        return {}
    perms = load_permutations(path)
    moved = sum(int((p != np.arange(p.shape[0])).sum()) for p in perms.values())
    logger.info(
        "Expert relabelling from %s: %d layers, %d experts moved",
        path,
        len(perms),
        moved,
    )
    return perms


def enabled() -> bool:
    return bool(envs.TPU_MOE_EXPERT_PERM_PATH)


def _perm_for(perms: dict[int, np.ndarray], layer: int, experts: int):
    perm = perms.get(layer)
    if perm is not None and perm.shape[0] != experts:
        raise ValueError(
            f"layer {layer} has {experts} experts but its permutation has "
            f"{perm.shape[0]}"
        )
    return perm


def relabelled_expert_id(name: str, expert_id: int, perms=None) -> int:
    """The id checkpoint tensor ``name`` (of expert ``expert_id``) loads as."""
    perms = _configured() if perms is None else perms
    match = _EXPERT_RE.search(name)
    if not perms or match is None:
        return expert_id
    perm = perms.get(int(match.group(2)))
    return expert_id if perm is None else int(perm[expert_id])


def relabel_checkpoint(
    weights: Iterable[tuple[str, torch.Tensor]], perms=None
) -> Iterator[tuple[str, torch.Tensor]]:
    """The checkpoint stream with every permuted layer relabelled."""
    perms = _configured() if perms is None else perms
    if not perms:
        yield from weights
        return
    for name, tensor in weights:
        expert = _EXPERT_RE.search(name)
        if expert is not None:
            perm = perms.get(int(expert.group(2)))
            if perm is not None:
                start, end = expert.span(3)
                name = f"{name[:start]}{int(perm[int(expert.group(3))])}{name[end:]}"
        elif (hashed := _HASH_RE.search(name)) is not None:
            if int(hashed.group(2)) in perms:
                raise NotImplementedError(
                    f"{name}: relabelling a hash-routed MoE layer is not supported"
                )
        else:
            gate = _GATE_RE.search(name)
            if gate is not None:
                perm = _perm_for(perms, int(gate.group(2)), tensor.shape[0])
                if perm is not None:
                    # Row perm[g] takes checkpoint row g.
                    inverse = torch.from_numpy(np.argsort(perm)).to(tensor.device)
                    tensor = tensor.index_select(0, inverse)
        yield name, tensor
