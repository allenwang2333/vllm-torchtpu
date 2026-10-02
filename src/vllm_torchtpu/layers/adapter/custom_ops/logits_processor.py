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

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


@LogitsProcessor.register_oot
class TpuLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that applies elementwise soft_cap and scale before TP all-gather.

    Upstream applies soft_cap and scale after _gather_logits(), which duplicates
    the elementwise tanh/division/multiplication across all TP ranks over the full
    vocabulary [B, V]. Applying soft_cap and scale on the local shard [B, V // TP]
    before all-gather is bit-for-bit identical, reduces VPU work by tp_size, and
    allows XLA to fuse the elementwise epilogue directly into the lm_head GEMM
    before the collective.
    """

    def _apply_soft_cap_and_scale(self, logits: torch.Tensor) -> torch.Tensor:
        if self.soft_cap is not None:
            logits = logits / self.soft_cap
            logits = torch.tanh(logits)
            logits = logits * self.soft_cap
        if self.scale != 1.0:
            logits *= self.scale
        return logits

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
            if logits is not None:
                logits = self._apply_soft_cap_and_scale(logits)
            return logits

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if logits is not None:
            logits = self._apply_soft_cap_and_scale(logits)

        if skip_gather:
            return logits

        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits
