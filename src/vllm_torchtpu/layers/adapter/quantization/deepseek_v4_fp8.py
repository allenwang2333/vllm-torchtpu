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
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    VllmFp8MoEMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import VllmDeepseekV4Mxfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import (
    DEEPSEEK_V4_FP8,
    get_tpu_quant_method,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LinearBase = vllm_linear.LinearBase
_BASE_IS_SCALE_E8M0 = DeepseekV4FP8Config.is_scale_e8m0.fget


@register_quantization_config(DEEPSEEK_V4_FP8)
@register_quantization_config(get_tpu_quant_method(DEEPSEEK_V4_FP8))
class VllmDeepseekV4Fp8Config(DeepseekV4FP8Config, VllmQuantConfig):
    """TPU quantization config for "deepseek_v4_fp8" format.

    Registered under "tpu-deepseek_v4_fp8".
    Dispatches MoE quant methods based on the checkpoint's expert_dtype:
      - "fp4" → VllmDeepseekV4Mxfp4MoEMethod
      - "fp8" → VllmFp8MoEMethodTPU
    Linear layers always use VllmFp8LinearMethodTPU (FP8 block-quant).
    """

    @property
    def is_scale_e8m0(self) -> bool:
        try:
            quant_cfg = (
                get_current_vllm_config().model_config.hf_config.quantization_config
            )
            if type(quant_cfg) is dict and quant_cfg.get("scale_fmt") == "ue8m0":
                return True
        except Exception:
            pass
        return bool(_BASE_IS_SCALE_E8M0(self))  # type: ignore[misc]

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return DEEPSEEK_V4_FP8

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> vllm_linear.LinearMethodBase | QuantizeMethodBase | None:
        if isinstance(layer, _LinearBase):
            ignored = self.ignored_layers
            if ignored and is_layer_skipped(
                prefix=prefix,
                ignored_layers=ignored,
                fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedLinearMethod()
            return VllmFp8LinearMethodTPU(
                self, self.get_linear_config(layer), prefix=prefix
            )
        if isinstance(layer, RoutedExperts):
            ignored = self.ignored_layers
            if ignored and is_layer_skipped(
                prefix=prefix,
                ignored_layers=ignored,
                fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedFusedMoEMethod(layer.moe_config)

            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    raise NotImplementedError("NVFP4 is not supported yet.")

                return VllmDeepseekV4Mxfp4MoEMethod(self.get_moe_config(layer))
            return VllmFp8MoEMethodTPU(self, self.get_moe_config(layer))

        return None
