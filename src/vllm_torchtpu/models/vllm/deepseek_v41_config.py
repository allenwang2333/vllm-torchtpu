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
"""HF config class for DeepSeek-V4.1 checkpoints (``model_type: deepseek_v41``).

Ported from ``vllm/transformers_utils/configs/deepseek_v41.py`` in upstream
vLLM (Apache-2.0), which post-dates the vLLM release this plugin pins. Neither
that release nor released Transformers know ``deepseek_v41``, so without this
registration ``vllm serve`` fails at ``ModelConfig`` with "Transformers does not
recognize this architecture".

This module must stay import-light (no torch / torch_tpu / kernels): it is
imported by every vLLM process, including the API-server frontends.
"""

from typing import Any

from transformers import PreTrainedConfig

DEEPSEEK_V41_MODEL_TYPE = "deepseek_v41"


class DeepseekV41Config(PreTrainedConfig):
    """DeepSeek V4.1 config.

    The HF config nests the text model under ``text_config`` and the vision
    tower under ``vision_config``. The model code reads flat attributes, so
    both sub-configs are flattened onto the top level here: text fields are
    exposed as-is, vision fields with a ``vision_*`` prefix.
    """

    model_type = DEEPSEEK_V41_MODEL_TYPE

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        text_config = dict(text_config or {})
        vision_config = dict(vision_config or {})

        # ``rope_scaling`` is a property in Transformers v5 (backed by
        # ``rope_parameters``); capture it here and restore after
        # super().__init__, which re-standardizes rope params.
        rope_scaling = text_config.pop("rope_scaling", None)

        for key, value in text_config.items():
            if key == "model_type":
                continue
            # Don't clobber PreTrainedConfig properties (e.g. is_encoder_decoder).
            if isinstance(getattr(type(self), key, None), property):
                continue
            setattr(self, key, value)

        super().__init__(**kwargs)
        if rope_scaling is not None:
            self.rope_parameters = rope_scaling

        # The v4.1 quantization_config carries the expert dtype; the model
        # code reads it from the top-level hf_config.
        quant_cfg = getattr(self, "quantization_config", None) or {}
        if not hasattr(self, "expert_dtype") and "expert_dtype" in quant_cfg:
            self.expert_dtype = quant_cfg["expert_dtype"]

        self.vision_n_layers = vision_config.get("num_hidden_layers", 0)
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_inter_dim = vision_config.get("intermediate_size", 2816)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_rope_theta = vision_config.get("rope_theta", 10000.0)
        self.vision_downsample_ratio = vision_config.get("downsample_ratio", 3)
        self.vision_max_n_token = vision_config.get("max_image_tokens", 1024)
        self.vision_min_pixels = vision_config.get("min_pixels", 295936)
        self.vision_max_wh_ratio = vision_config.get("max_wh_ratio")


def register_deepseek_v41_config() -> None:
    """Make vLLM resolve ``model_type: deepseek_v41`` to :class:`DeepseekV41Config`.

    No-op when the installed vLLM already ships its own entry.
    """
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    if DEEPSEEK_V41_MODEL_TYPE in _CONFIG_REGISTRY:
        return
    # LazyConfigDict returns concrete classes as-is, so inserting the type is
    # enough; vLLM registers it with transformers' AutoConfig on first use.
    _CONFIG_REGISTRY[DEEPSEEK_V41_MODEL_TYPE] = DeepseekV41Config
