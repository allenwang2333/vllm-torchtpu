# SPDX-License-Identifier: Apache-2.0
"""Spec-verify top-k windows from the vocabulary-parallel LM head.

With ``TPU_SPARSE_SPEC_SAMPLING`` a verify chunk whose requests all have
``0 < top_k <= TPU_SAMPLING_TOPK_BOUND`` (or are greedy) is sampled from each
logits row's top-``bound`` window. ``TPU_SAMPLER_VOCAB_PARALLEL_TOPK`` builds
that window without gathering the full logits: each tensor-parallel rank
takes the exact top-``bound`` of its own vocabulary slice, the ranks exchange
only those ``bound`` (value, id) pairs, and every rank keeps the top
``bound`` of the ``tp_size * bound`` candidates. The global top-``bound``
values are the top-``bound`` of the union of the per-slice top-``bound``
values, so the window holds exactly the values of ``_chunked_topk`` over the
gathered logits; only which of several tied ids it names can differ, and
tied ids have equal logits.
"""

from __future__ import annotations

import torch
from vllm.distributed import tensor_model_parallel_all_gather

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    shard_topk,
    shard_topk_keys,
    unpack_value_index,
)


def window_from_hidden_states(
    owner: torch.nn.Module, hidden_states: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-``width`` window of every row of ``compute_logits(hidden)``:
    float32 values sorted descending ``[rows, width]`` and their int32
    vocab ids. ``owner`` holds ``lm_head`` and ``logits_processor``
    (``greedy_top_tokens.resolve_head_owner``)."""
    processor = owner.logits_processor
    head = owner.lm_head
    logits = processor._apply_head(head, hidden_states, None)
    if processor.soft_cap is not None:
        logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        logits = logits * processor.scale
    num_pad = head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        cols = torch.arange(logits.shape[-1], device=logits.device)
        logits = torch.where(
            cols < logits.shape[-1] - num_pad,
            logits,
            torch.full_like(logits, float("-inf")),
        )
    tp_size = head.tp_size
    n = logits.shape[-1]
    if tp_size > 1 and getattr(head.shard_indices, "num_added_elements", 0) == 0:
        keys = shard_topk_keys(logits, width)
        if keys is not None:
            # [rows, width] int32 keys per rank, one collective; rank r owns
            # ids [r * n, (r + 1) * n) (a rank past the vocabulary holds
            # only masked padding, which never reaches a window). Equal keys
            # keep the earlier, lower rank: ties resolve to the lowest id.
            gathered = tensor_model_parallel_all_gather(keys, dim=-1)
            keys, pos = torch.topk(gathered, width, dim=-1, largest=True, sorted=True)
            vals, local = unpack_value_index(keys)
            idx = (pos // width).to(torch.int32) * n + local
            return vals.to(torch.float32), idx
    vals, idx = shard_topk(logits, width)
    idx = idx.to(torch.int32) + head.shard_indices.org_vocab_start_index
    vals = vals.to(torch.float32)
    if tp_size == 1:
        return vals, idx
    rows = hidden_states.shape[0]
    # One collective for both: ids < 2**24 are exact in float32.
    pair = torch.stack([vals, idx.to(torch.float32)], dim=-1).view(rows, 2 * width)
    gathered = tensor_model_parallel_all_gather(pair, dim=-1)
    gathered = gathered.view(rows, tp_size * width, 2)
    # Candidates sit in rank order (rank r owns the r-th id range) and each
    # rank's are sorted descending; topk keeps the earlier of tied entries.
    vals, pos = torch.topk(gathered[..., 0], width, dim=-1, largest=True, sorted=True)
    idx = torch.gather(gathered[..., 1], 1, pos).to(torch.int32)
    return vals, idx
