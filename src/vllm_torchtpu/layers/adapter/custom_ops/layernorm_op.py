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
"""Out-of-tree (OOT) custom wrappers for GemmaRMSNorm."""

import torch
from vllm import ir
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

@GemmaRMSNorm.register_oot
class WeightPreprocessedGemmaRMSNorm(GemmaRMSNorm):

    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
        ) -> None:
            super().__init__(hidden_size, eps)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
            # super().process_weights_after_loading(act_dtype)
            self.processed_weight = self.weight.float() + 1.0
            logger.info("processed weight for GemmaRMSNorm has been computed and stored.")

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            """PyTorch-native implementation equivalent to forward()."""
            weight = self.processed_weight
            if residual is None:
                return ir.ops.rms_norm(x, weight, self.variance_epsilon)
            return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)
    






