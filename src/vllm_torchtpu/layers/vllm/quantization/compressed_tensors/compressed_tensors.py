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
"""Compressed Tensors Quantization Config for TPU."""

from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig

from vllm_torchtpu.layers.common.quant_methods import (COMPRESSED_TENSORS,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import \
    VllmCompressedTensorsMoEMethod
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig


@register_quantization_config(get_tpu_quant_method(COMPRESSED_TENSORS))
class VllmCompressedTensorsConfig(CompressedTensorsConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return COMPRESSED_TENSORS

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            # Attention weights are pure BF16
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return VllmCompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix)
        return None
