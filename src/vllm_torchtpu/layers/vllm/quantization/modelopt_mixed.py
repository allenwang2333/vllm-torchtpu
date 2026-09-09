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
"""ModelOpt mixed-precision quantization dispatch for TPU."""

from typing import Any, Optional

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import \
    ModelOptMixedPrecisionConfig

from vllm_torchtpu.layers.common.quant_methods import (MODELOPT_MIXED,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.vllm.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod


@register_quantization_config(get_tpu_quant_method(MODELOPT_MIXED))
class VllmModelOptMixedPrecisionConfig(ModelOptMixedPrecisionConfig,
                                       VllmQuantConfig):
    """Dispatch each ModelOpt mixed-precision layer to its TPU backend."""

    @classmethod
    def get_name(cls) -> str:
        return MODELOPT_MIXED

    @classmethod
    def set_configs(cls, vllm_config):
        cls.vllm_config = vllm_config
        VllmFp8Config.set_configs(vllm_config)
        VllmNvfp4Config.set_configs(vllm_config)

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> "VllmModelOptMixedPrecisionConfig":
        config = super()._from_config(
            quant_method=quant_method,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            original_config=original_config,
            group_size=group_size,
            **kwargs,
        )

        # ModelOpt's outer config owns the per-layer algorithm map. Replace its
        # GPU child configs with the native TPU configs used by the standalone
        # FP8 and NVFP4 paths.
        nvfp4_group_size = config.nvfp4_config.group_size
        config.fp8_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            ignored_layers=[],
        )
        config.nvfp4_config = VllmNvfp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=nvfp4_group_size,
        )
        # TPU executes both ModelOpt FP4 variants through its W4A16 kernels.
        config.w4a16_nvfp4_config = config.nvfp4_config
        return config

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, Attention):
            return None

        if self.is_layer_excluded(prefix):
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return VllmUnquantizedFusedMoEMethod(
                    self.get_moe_config(layer))
            return None

        quant_algo = self._resolve_quant_algo(prefix)
        if quant_algo == "FP8":
            return self.fp8_config.get_quant_method(layer, prefix)
        if quant_algo in ("NVFP4", "W4A16_NVFP4"):
            return self.nvfp4_config.get_quant_method(layer, prefix)
        if quant_algo == "MXFP8":
            raise NotImplementedError(
                "ModelOpt mixed-precision MXFP8 layers are not supported on "
                "TPU.")

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
        return None
