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
import torch.nn as nn
from vllm import ir
from vllm.model_executor.layers.layernorm import GemmaRMSNorm


@GemmaRMSNorm.register_oot
class WeightPreprocessedGemmaRMSNorm(GemmaRMSNorm):
    """Holds ``1 + w`` in fp32 so forward skips the per-step upcast-and-add.

    Gemma normalizes with ``x * (1 + w)``, and upstream's ``forward_native``
    rebuilds ``self.weight.float() + 1.0`` on every call. The result is a
    function of the checkpoint alone, so it is folded into the parameter once,
    at load time, by the parameter's own ``weight_loader`` -- the hook every
    Gemma loader already honours (``AutoWeightsLoader`` in
    ``vllm/model_executor/models/utils.py``, and gemma4's hand-written
    ``load_weights``). Using that hook rather than a post-load sweep is what
    keeps the target model and both speculative-decoding draft models working
    with no extra call at their load sites.

    Two consequences worth knowing:

    - ``self.weight`` no longer holds the checkpoint value; it holds
      ``1 + w``. Any consumer that reads this weight raw and expects the
      checkpoint would be wrong. Nothing does today -- the weight is read only
      by the forward below -- but it is the same hazard class that
      ``layers/vllm/linear_common.py`` documents for flipped linear layouts.
    - Under ``load_format=dummy`` the fold never happens:
      ``initialize_dummy_weights`` fills parameters in place through
      ``model.state_dict()`` and never calls ``weight_loader``. The parameter
      is then noise rather than ``1 + noise``, which is harmless for profiling
      (same magnitude, no missing state) but means dummy-loaded numerics are
      not bit-comparable to a real load.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps)
        # fp32 explicitly: __init__ runs under
        # `set_default_torch_dtype(model_config.dtype)`, so the upstream
        # parameter would be bf16, which cannot hold `1 + w` at the precision
        # the per-step `.float()` upcast was providing.
        self.weight = nn.Parameter(torch.zeros(hidden_size,
                                               dtype=torch.float32),
                                   requires_grad=False)
        self.weight.weight_loader = self._load_folded_weight

    @staticmethod
    def _load_folded_weight(param: nn.Parameter,
                            loaded_weight: torch.Tensor) -> None:
        """Fold Gemma's implicit ``1 +`` into the weight as it is loaded.

        Mirrors `default_weight_loader`, which this replaces, including its
        shape assertion; the copy upcasts to the parameter's fp32.
        """
        assert param.size() == loaded_weight.size(), (
            f"Attempted to load weight of shape {loaded_weight.size()} "
            f"into parameter of shape {param.size()}")
        param.data.copy_(loaded_weight.float() + 1.0)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(x, self.weight, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(x, residual, self.weight,
                                         self.variance_epsilon)
