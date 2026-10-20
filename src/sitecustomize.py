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
"""Process-wide defaults required before torch_tpu autoloads."""

import contextlib
import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Early patch Gemma4 multimodal models:
# 1. First patch layer config so that full-attention layers receive head_dim=512
#    before any model class module imports gemma4_layer_config.
# 2. Then patch supports_encoder_tp_data = True on model classes.
with contextlib.suppress(Exception):
    from vllm_torchtpu.models.vllm.gemma4_mm_patch import (
        patch_gemma4_layer_config,
        patch_gemma4_mm_model_classes,
    )

    patch_gemma4_layer_config()
    patch_gemma4_mm_model_classes()

# Heterogeneous architecture support for Gemma-4: allow global per-layer
# attribute access so head_dim access does not raise
# AmbiguousGlobalPerLayerAttributeError during ModelConfig conversion.
with contextlib.suppress(Exception):
    from transformers.configuration_utils import PretrainedConfig

    PretrainedConfig.allow_global_per_layer_attribute_access = True
