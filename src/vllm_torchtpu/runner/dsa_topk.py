# SPDX-License-Identifier: Apache-2.0
"""The shared top-k index table of a sparse-attention (DSA) model.

Deepseek-family sparse models choose, every few layers, which earlier tokens
each query attends to, and the layers in between reuse that choice: one
process-wide ``int32[max_num_batched_tokens, index_topk]`` table is written by
a layer that carries indexer weights and read by the layers that do not.

Under pipeline parallelism the reader and the writer can sit on different
chips, and a reading stage has no way to recompute the table -- the weights
are not there. So the table travels between stages as a further hand-off lane
(``distributed/pp_wave.py``), and this module is how the runner gets hold of
it without knowing a model's attribute layout.
"""
from typing import Any

import torch

# The hand-off lane, and the key of the table in a model's intermediate
# tensors. Only the runner and the wave use it; the model never sees it.
TOPK_INDICES = "topk_indices"

_ATTRIBUTE = "topk_indices_buffer"


def model_uses_sparse_indexer(vllm_config: Any) -> bool:
    """Whether the model picks its attended tokens with a sparse indexer.

    ``index_topk`` is what the model code itself keys on to build the table
    (a v3.2-style sparse attention), so it is what decides here too.
    """
    hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
    return hasattr(hf_config, "index_topk")


def find_topk_indices_buffer(model: torch.nn.Module) -> torch.Tensor | None:
    """The one top-k table this process's layers share, or None.

    Every layer of a sparse model is handed the same tensor, so a sweep of
    the modules that hold one must find a single object; more than one would
    mean a stage's readers and writers disagree about where the indices are,
    which the hand-off could not repair. Layers belonging to other stages are
    ``PPMissingLayer`` and hold nothing.
    """
    found: dict[int, torch.Tensor] = {}
    for module in model.modules():
        buffer = getattr(module, _ATTRIBUTE, None)
        if isinstance(buffer, torch.Tensor):
            found.setdefault(id(buffer), buffer)
    if not found:
        return None
    if len(found) > 1:
        shapes = sorted(tuple(b.shape) for b in found.values())
        raise RuntimeError(
            "the model's layers hold several distinct top-k index tables "
            f"{shapes}; the pipeline hand-off carries one per stage")
    return next(iter(found.values()))
