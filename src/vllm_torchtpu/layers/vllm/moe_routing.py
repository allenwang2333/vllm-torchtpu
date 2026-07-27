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
from typing import List, Tuple, Dict

class SortedExpertProfileRouter:
    """
    Sorted-Expert Profile Padding (SEPP) routing algorithm tailored for TPU static graphs.
    """
    def __init__(self, num_experts: int, total_tokens: int, top_k: int, alignment: int = 32):
        self.num_experts = num_experts
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.alignment = alignment
        
        # 1. Initialize a default static profile based on typical power-law distribution
        self.static_profile = self._generate_default_profile()
        print(f"[SEPP] Static descending bucket profile initialized successfully: {self.static_profile} (Total capacity: {sum(self.static_profile)})")

    def _generate_default_profile(self) -> List[int]:
        """
        Generate a highly utilized, asymmetric descending static profile based on total token count, aligned to TPU memory boundaries.
        """
        max_assignments = self.total_tokens * self.top_k
        profile = []
        remaining = max_assignments
        decay_factor = 0.65  # Simulated skewness factor for MoE routing
        
        for i in range(self.num_experts):
            if i == self.num_experts - 1:
                cap = remaining
            else:
                cap = int(remaining * (1 - decay_factor))
                remaining -= cap
            
            # Align to optimal byte boundaries for TPU HBM transfers (usually 16 or 32)
            aligned_cap = ((cap + self.alignment - 1) // self.alignment) * self.alignment
            profile.append(max(aligned_cap, self.alignment))
            
        profile.sort(reverse=True)
        return profile

    def calibrate_profile_offline(self, history_counts: List[List[int]], percentile: float = 99.5):
        """
        [Optional advanced feature] Offline profile calibration using actual distribution history from warm-up runs.
        """
        sorted_history = [sorted(step, reverse=True) for step in history_counts]
        transposed = list(zip(*sorted_history))
        
        calibrated_profile = []
        for rank_history in transposed:
            rank_tensor = torch.tensor(rank_history, dtype=torch.float32)
            val = torch.quantile(rank_tensor, percentile / 100.0).item()
            aligned_val = ((int(val) + self.alignment - 1) // self.alignment) * self.alignment
            calibrated_profile.append(max(aligned_val, self.alignment))
            
        calibrated_profile.sort(reverse=True)
        self.static_profile = calibrated_profile
        print(f"[SEPP] Profile boundaries updated via warm-up history calibration: {self.static_profile}")

    def route_and_generate_indices(self, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, List[int], Dict[int, int]]:
        """
        Generate the static profile index mapping based on current step routing results.
        
        Input:
            topk_ids: Shape [total_tokens, top_k], top-K physical expert IDs selected by each token
        Returns:
            sorted_actual_counts: Sorted physical expert token counts for current step
            static_profile: Unchanged static graph compilation shape
            physical_to_profile_map: Dictionary mapping physical expert ID to profile slot index (used for permutation phase)
        """
        # Step 1: Count the actual number of tokens assigned to each physical expert in the current step
        flat_gates = topk_ids.flatten()
        
        # Safety check: filter out negative IDs (e.g., padding) and out-of-bound IDs
        valid_mask = (flat_gates >= 0) & (flat_gates < self.num_experts)
        valid_gates = flat_gates[valid_mask]
        
        actual_counts = torch.bincount(valid_gates, minlength=self.num_experts)
        # Ensure we strictly only have `num_experts` counts
        actual_counts = actual_counts[:self.num_experts]
        
        # Step 2: Sort in descending order to identify the hottest and coldest experts
        sorted_actual_counts, sorted_expert_ids = torch.sort(actual_counts, descending=True)
        
        # Step 3: Create mapping dictionary
        # e.g.: sorted_expert_ids[0] (hottest physical expert) -> Maps to profile index 0 (largest capacity)
        #       sorted_expert_ids[-1] (coldest physical expert) -> Maps to profile index 7 (smallest capacity)
        physical_to_profile_map = {
            expert_id.item(): profile_idx 
            for profile_idx, expert_id in enumerate(sorted_expert_ids)
        }
        
        # Step 4: Calculate and output overflow rate monitoring (verifies if any physical expert exceeds mapped slot static capacity)
        overflow_tokens = 0
        for i, phys_id in enumerate(sorted_expert_ids):
            actual = sorted_actual_counts[i].item()
            allocated = self.static_profile[i]
            if actual > allocated:
                overflow_tokens += (actual - allocated)
                
        if overflow_tokens > 0:
            overflow_rate = (overflow_tokens / flat_gates.numel()) * 100
            # Print low-frequency warning only when there is a risk of token dropping
            if overflow_rate > 1.0:
                print(f"[SEPP WARNING] Communication overflow detected! Overflow tokens: {overflow_tokens} ({overflow_rate:.2f}%), suggest fine-tuning profile.")
                
        return sorted_actual_counts, self.static_profile, physical_to_profile_map

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
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import \
            grouped_topk
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            scoring_func=scoring_fn,
            routed_scaling_factor=layer.routed_scaling_factor,
            e_score_correction_bias=layer.e_score_correction_bias,
        )
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)

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
