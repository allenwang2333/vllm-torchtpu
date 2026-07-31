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
"""Compressed-tensors MXFP4 MoE for TPU."""

import torch
from vllm.model_executor.layers.fused_moe import (FusedMoeWeightScaleSupported,
                                                  RoutedExperts)
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod


class VllmCompressedTensorsMxfp4MoEMethod(VllmMxfp4MoEMethod):
    """Load compressed-tensors MXFP4 parameters and run the TPU Pallas MoE."""

    group_size = 32

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % self.group_size != 0:
            raise ValueError(
                f"MXFP4 hidden size must be divisible by 32, got {hidden_size}."
            )
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                "MXFP4 intermediate size per partition must be divisible by "
                f"32, got {intermediate_size_per_partition}.")

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
            dtype=torch.uint8,
        ),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=torch.uint8,
        ),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = (
            FusedMoeWeightScaleSupported.GROUP.value)

        w13_scale = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // self.group_size,
            dtype=torch.uint8,
        ),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, scale_attrs)

        w2_scale = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // self.group_size,
            dtype=torch.uint8,
        ),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, scale_attrs)

        layer.w13_bias = None
        layer.w2_bias = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data,
                                              requires_grad=False)
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data,
                                             requires_grad=False)
        delattr(layer, "w2_weight_packed")
        super().process_weights_after_loading(layer)
