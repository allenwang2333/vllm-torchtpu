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
"""Dynamic Expert Hotness Profiler and Snake Load-Balancer for TPU MoE."""

import threading
import torch
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class ExpertHotnessTracker:
    """Thread-safe online rolling window tracker for MoE expert activation hotness."""

    def __init__(self, num_experts: int = 512, window_decay: float = 0.95) -> None:
        self.num_experts = num_experts
        self.window_decay = window_decay
        self._hotness_counts = torch.zeros(num_experts, dtype=torch.float32)
        self._total_tokens = 0
        self._lock = threading.Lock()

    def update(self, topk_ids: torch.Tensor) -> None:
        """Update activation counts from a routing step."""
        if topk_ids is None or topk_ids.numel() == 0:
            return

        with self._lock:
            counts = torch.bincount(topk_ids.reshape(-1).cpu(), minlength=self.num_experts)
            counts_float = counts.to(torch.float32)
            self._hotness_counts = (
                self.window_decay * self._hotness_counts
                + (1.0 - self.window_decay) * counts_float
            )
            self._total_tokens += topk_ids.shape[0]

    def get_sorted_experts(self) -> list[int]:
        """Get expert IDs sorted by hotness (most hot to least hot)."""
        with self._lock:
            _, indices = torch.sort(self._hotness_counts, descending=True)
            return indices.tolist()

    def compute_snake_expert_map(self, num_ranks: int) -> dict[int, int]:
        """Compute Snake/Zig-zag load-balanced mapping of global experts to EP ranks.

        Returns:
            map_expert_to_rank: Dict mapping global expert ID -> assigned EP rank ID
        """
        sorted_experts = self.get_sorted_experts()
        expert_to_rank = {}

        for idx, global_expert_id in enumerate(sorted_experts):
            cycle = idx // num_ranks
            rank_in_cycle = idx % num_ranks
            # Zig-zag: even cycles go left-to-right, odd cycles go right-to-left
            if cycle % 2 == 0:
                assigned_rank = rank_in_cycle
            else:
                assigned_rank = num_ranks - 1 - rank_in_cycle
            expert_to_rank[global_expert_id] = assigned_rank

        return expert_to_rank


_GLOBAL_HOTNESS_TRACKER: ExpertHotnessTracker | None = None


def get_global_hotness_tracker(num_experts: int = 512) -> ExpertHotnessTracker:
    global _GLOBAL_HOTNESS_TRACKER
    if _GLOBAL_HOTNESS_TRACKER is None:
        _GLOBAL_HOTNESS_TRACKER = ExpertHotnessTracker(num_experts=num_experts)
    return _GLOBAL_HOTNESS_TRACKER
