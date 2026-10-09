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

from dataclasses import dataclass, field

import torch
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context, is_forward_context_available

_padding_state: "TokenPaddingState | None" = None


@dataclass(slots=True)
class TokenPaddingState:
    """Track token padding."""

    local_padding_mask: torch.Tensor
    # Host mirror of ``local_padding_mask`` and the ``(start, stop)`` padding
    # range it holds; ``None`` until the first ``update`` writes the device mask.
    _host_cache: tuple[torch.Tensor, tuple[int, int]] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def create(cls, max_local_tokens: int, device) -> "TokenPaddingState":
        local_padding_mask = torch.zeros(
            max_local_tokens, dtype=torch.bool, device=device
        )
        return cls(local_padding_mask=local_padding_mask)

    def update(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """Write this rank's token padding mask.

        Steady-state steps repeat the previous padding range; they skip the
        synchronous host-to-device copy.
        """
        # Same bounds as ``range(n)[valid:padded]``. The runner always passes
        # 0 <= valid <= padded <= n, so take that case with plain compares and
        # only build the range for negative or out-of-range bounds (CPython:
        # ~100 ns vs ~240 ns for the range slice; pure min/max clamping is
        # slower than the range slice, ~480 ns).
        n = self.local_padding_mask.shape[0]
        if 0 <= local_valid_tokens < local_padded_tokens <= n:
            padding_range = (local_valid_tokens, local_padded_tokens)
        elif 0 <= local_padded_tokens <= local_valid_tokens:
            padding_range = (0, 0)
        else:
            rows = range(n)[local_valid_tokens:local_padded_tokens]
            padding_range = (rows.start, rows.stop) if rows else (0, 0)
        cached = self._host_cache
        if cached is None:
            host = torch.zeros(n, dtype=torch.bool)
        elif cached[1] == padding_range:
            return
        else:
            host, (start, stop) = cached
            host[start:stop] = False
        host[padding_range[0] : padding_range[1]] = True
        # Synchronous copy: ``host`` is rewritten by the next cache miss.
        self.local_padding_mask.copy_(host)
        self._host_cache = (host, padding_range)

    def get_local_padding_mask(self, num_tokens: int) -> torch.Tensor:
        return self.local_padding_mask[:num_tokens]


def set_padding_state(state: "TokenPaddingState | None") -> None:
    global _padding_state
    _padding_state = state


def zero_routing_weights_for_padding(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, is_local_tensor: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set routing weights to 0 for padded tokens.

    A padded token keeps its selected expert ids but gets zero gate
    weight, which tells the fused MoE to skip the expert computation.

    Args:
        topk_ids: Selected expert IDs for each token.
        topk_weights: Routing weights for each token.
        is_local_tensor: Whether topk_ids and topk_weights are local tensors
            (i.e. not gathered across DP ranks). If True, skips DP all_gather.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The (topk_ids, topk_weights) tensors
            with routing weights zeroed for padded tokens.
    """
    # A model may compact a sub-batch after the runner prepared its persistent
    # mask. Such a stage supplies a mask in its own routing-row layout; do not
    # reuse the enclosing batch's valid-token prefix for these rows.
    if is_forward_context_available():
        stage_mask = get_forward_context().additional_kwargs.get("token_padding_mask")
        if stage_mask is not None:
            return topk_ids, topk_weights.masked_fill(stage_mask[:, None], 0)
    if _padding_state is None:
        return topk_ids, topk_weights
    num_tokens = topk_ids.shape[0]
    if is_local_tensor:
        is_padding = _padding_state.get_local_padding_mask(num_tokens)
    else:
        dp_group = get_dp_group()
        dp_size = dp_group.world_size
        if dp_size > 1:
            local_padding = _padding_state.get_local_padding_mask(num_tokens // dp_size)
            # TPU collectives may not support bool, so encode the mask as int8.
            gathered_padding = dp_group.all_gather(local_padding.to(torch.int8), dim=0)
            is_padding = gathered_padding.ne(0)
        else:
            is_padding = _padding_state.get_local_padding_mask(num_tokens)
    return topk_ids, topk_weights.masked_fill(is_padding[:, None], 0)
