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
import torch.nn as nn
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op as gdn_attention_op
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as kda_attention_op
from vllm_torchtpu.layers.adapter.custom_ops import layernorm_op as layernorm_op
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_attention_op


def _register_custom_ops() -> None:
    """Explicitly imported to trigger custom operation registration decorators."""
    pass

_OPS_PROCESS_WEIGHTS_AFTER_LOADING = (
    "WeightPreprocessedGemmaRMSNorm",
)

def force_process_weights_after_loading(model: nn.Module) -> None:
    for _, module in model.named_modules():
        # Defensive check
        if not hasattr(module, "process_weights_after_loading"):
            continue

        if module.__class__.__name__ in _OPS_PROCESS_WEIGHTS_AFTER_LOADING:
            module.process_weights_after_loading(torch.float32)