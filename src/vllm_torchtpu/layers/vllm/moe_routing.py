# Copyright 2025 Google LLC
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
"""Routing helpers for TPU local MoE kernels."""

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Warn once at import time rather than inside maybe_force_random_routing():
# the notice then reaches every worker's startup log without adding a Python
# call to the compiled forward pass, where it could trigger a graph break in
# the exact code path we are trying to profile.
if envs.FORCE_MOE_RANDOM_ROUTING:
    logger.warning(
        "FORCE_MOE_RANDOM_ROUTING is enabled: MoE expert routing is RANDOM. "
        "This is a test-only feature and model output is meaningless. Never "
        "enable it for serving or accuracy evaluation.")


def maybe_force_random_routing(router_logits: torch.Tensor) -> torch.Tensor:
    """Optionally replace router logits with uniform noise, for profiling.

    Motivation: the TPU profiler (xprof) can only trace a single device, but
    under expert parallelism each device owns a different slice of the experts.
    Real routing is data-dependent and typically skewed, so which experts --
    and therefore which devices -- do work depends on the input. A one-device
    trace is then not representative of the full mesh.

    With ``FORCE_MOE_RANDOM_ROUTING`` set, we route on uniform noise instead of
    the model's gate. Top-k over i.i.d. noise selects a uniformly random set of
    experts per token, so expert load is balanced across all EP shards in
    expectation and any single device's profile represents the whole mesh.

    This deliberately discards the trained gate, so model output is garbage.
    Use it only for profiling and performance benchmarking -- never for serving
    or accuracy evaluation. Disabled by default; a no-op when the flag is unset.
    """
    if not envs.FORCE_MOE_RANDOM_ROUTING:
        return router_logits
    return torch.rand_like(router_logits)


def _apply_scoring_fn(scoring_fn: str,
                      router_logits: torch.Tensor) -> torch.Tensor:
    scores = router_logits.float()
    if scoring_fn == "softmax":
        return scores.softmax(dim=-1)
    if scoring_fn == "sigmoid":
        return scores.sigmoid()
    raise NotImplementedError(
        f"FusedMoE does not support {scoring_fn} scoring function for TPU.")


def _grouped_topk(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    layer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped (expert-group) top-k routing, eager and TPU-traceable.

    A faithful port of ``grouped_topk`` in
    ``vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py``:
    score the experts, score each expert *group*, keep the best ``topk_group``
    groups, mask the rest to ``-inf``, and only then take the top ``topk``
    experts. When a correction bias is present the biased scores drive both the
    group scoring and the expert *selection*, while the unbiased scores supply
    the routing *weights*; renormalization and ``routed_scaling_factor`` come
    last.

    Why reimplement it here rather than call vLLM's: that function carries
    ``@torch.compile(dynamic=True)``, and torch_tpu's backend rejects the
    resulting SymInt with "TPU backend: does not support dynamic shape". This
    version keeps every shape static, so it traces cleanly.

    Two deliberate deviations from the reference, neither changing the result:

    * The group mask is built by comparison against ``arange`` instead of
      ``torch.zeros_like(...).scatter_(...)``, avoiding an in-place scatter in
      the compiled graph.
    * Scores are computed in float32 (see ``_apply_scoring_fn``) rather than the
      logits' dtype. Strictly more accurate, and it matches what the rest of
      this module already does for ungrouped routing.

    ``VLLM_BATCH_INVARIANT``'s ``sorted=True`` tie-breaking is not plumbed
    through; ``torch.topk`` defaults to ``sorted=True`` here anyway.
    """
    scores = _apply_scoring_fn(scoring_fn, router_logits)  # [n_tokens, n_expert]
    bias = getattr(layer, "e_score_correction_bias", None)
    n_group = getattr(layer, "num_expert_group", None) or 1
    topk_group = getattr(layer, "topk_group", None) or n_group

    # NOTE: do not log from this function. It runs inside
    # `@torch.compile(fullgraph=True)` regions, and `vllm_torchtpu.logger`
    # returns a lazy proxy whose first attribute access calls `init_logger` ->
    # `logging.getLogger`, which Dynamo cannot trace: the capture dies with
    # "Attempted to call function marked as skipped". It is invisible under
    # --enforce-eager and only fails once the model is compiled.

    # Biased scores select; unbiased scores weight.
    original_scores = scores
    if bias is not None:
        scores = scores + bias.to(scores.dtype).unsqueeze(0)

    # With one group, or with every group kept, the mask admits all experts and
    # this reduces to a plain top-k -- skip the work rather than build an
    # all-true mask. (Kimi-Linear is exactly this case: 1 group of 256.)
    if n_group > 1 and topk_group < n_group:
        num_token = scores.shape[0]
        group_size = scores.shape[-1] // n_group
        grouped = scores.view(num_token, n_group, group_size)
        if bias is not None:
            # Reference scores a group by its two best experts, not its best.
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        else:
            group_scores = grouped.max(dim=-1).values  # [n_tokens, n_group]

        group_idx = torch.topk(group_scores, k=topk_group,
                               dim=-1)[1]  # [n_tokens, topk_group]
        keep_group = (torch.arange(n_group, device=group_idx.device).view(
            1, 1, -1) == group_idx.unsqueeze(-1)).any(dim=1)  # [n_tok, n_group]
        score_mask = keep_group.unsqueeze(-1).expand(num_token, n_group,
                                                     group_size).reshape(
                                                         num_token, -1)
        scores = torch.where(score_mask, scores,
                             torch.full_like(scores, float("-inf")))

    if bias is not None:
        topk_ids = torch.topk(scores, k=topk, dim=-1)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    scaling = getattr(layer, "routed_scaling_factor", 1.0) or 1.0
    if scaling != 1.0:
        topk_weights = topk_weights * scaling

    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    layer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local routed ids/weights for the non-EP path."""
    # Profiling-only override; a no-op unless FORCE_MOE_RANDOM_ROUTING is set.
    router_logits = maybe_force_random_routing(router_logits)
    if layer is not None and layer.use_grouped_topk:
        return _grouped_topk(
            hidden_states,
            router_logits,
            topk=topk,
            renormalize=renormalize,
            scoring_fn=scoring_fn,
            layer=layer,
        )

    topk_weights, topk_ids = torch.topk(_apply_scoring_fn(
        scoring_fn, router_logits),
                                        k=topk,
                                        dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


def get_experts_start(layer) -> int | None:
    """Return the first global expert id owned by this shard, or None for non-EP.

    Mirrors the formula vLLM's ``determine_expert_map`` uses for linear
    placement (``vllm/model_executor/layers/fused_moe/layer.py``), so this is
    a pure function of ``ep_rank``, ``ep_size``, and ``global_num_experts`` --
    known at layer construction time. ``validate_linear_ep_placement`` enforces
    that the placement is linear before this value is used.
    """
    if not layer.moe_config.moe_parallel_config.use_ep:
        return None
    base = layer.global_num_experts // layer.moe_config.ep_size
    remainder = layer.global_num_experts % layer.moe_config.ep_size
    return (layer.moe_config.ep_rank * base +
            min(layer.moe_config.ep_rank, remainder))


def validate_linear_ep_placement(layer) -> None:
    """Validate that this layer uses linear EP placement.

    The kernel relies on a contiguous-block ``expert_map``, which vLLM only
    produces when ``expert_placement_strategy == "linear"``.
    """
    strategy = layer.expert_placement_strategy
    if strategy != "linear":
        raise NotImplementedError(
            "fused MoE kernel currently requires linear EP placement; got "
            f"expert_placement_strategy={strategy!r}.")
