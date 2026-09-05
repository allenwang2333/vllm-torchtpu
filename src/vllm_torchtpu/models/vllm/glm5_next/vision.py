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
"""GLM-5.3-Flash's vision tower.

Structurally this is GLM-OCR's tower, and vLLM already has that
(``glm_ocr.py:249``), so almost everything is inherited. transformers says the
same thing: ``modular_glm5_next.py:1373-1420`` defines the whole tower as four
thin subclasses of ``GlmOcr*``.

The entire delta is:

1. **SwiGLU clamping**, in both the block MLP and the patch merger. GLM clamps
   the gate above and the up-projection on both sides before the activation:
   ``silu(clamp(gate, max=L)) * clamp(up, -L, L)``. vLLM's version applies a
   fused ``SiluAndMul()`` straight to the merged projection, so the clamp has
   to go between the matmul and the activation -- ``SiluAndMul`` cannot be
   reused.
2. **Two config sources GlmOcr reads from the wrong place**, both of which
   would load silently-wrong-shaped or silently-wrong-epsilon layers. See
   ``Glm5NextVisionTransformer``.

The tower is bf16: every ``visual.*`` module is listed in the checkpoint's
``quantization_config.modules_to_not_convert``, so no FP8 path runs here.

GLM-5.3-Flash also does **not** use M-RoPE -- ``Glm5NextModel.__init__`` does
``del self.rope_deltas`` and the text model builds plain 1-D positions -- so
none of the mrope plumbing that ``Glm4vForConditionalGeneration`` carries is
needed.
"""

from functools import partial

import torch
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.glm_ocr import (GlmOcrPatchMerger,
                                                GlmOcrVisionBlock,
                                                GlmOcrVisionMLP,
                                                GlmOcrVisionTransformer)


def _clamped_swiglu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """``silu(clamp(gate, max=L)) * clamp(up, -L, L)``.

    ``gate_up`` is the fused projection, gate first -- the layout
    ``SiluAndMul`` assumes, which is what makes splitting it here safe.

    The asymmetry is real and deliberate in the reference
    (``modular_glm5_next.py:1379-1381``): the gate is bounded above only,
    because ``silu`` already handles the negative tail, while the
    up-projection multiplies directly and is bounded both ways.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return nn.functional.silu(gate) * up


class Glm5NextVisionMLP(GlmOcrVisionMLP):
    """GlmOcr's block MLP with GLM-5.3's SwiGLU clamp."""

    def __init__(self, *args, swiglu_limit: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(_clamped_swiglu(gate_up, self.swiglu_limit))
        return x


class Glm5NextVisionPatchMerger(GlmOcrPatchMerger):
    """GlmOcr's merger with the same clamp."""

    def __init__(self, *args, swiglu_limit: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(_clamped_swiglu(gate_up, self.swiglu_limit))
        return x


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    """GlmOcr's block with the clamped MLP swapped in.

    Norms and attention are inherited unchanged: ``q_norm``/``k_norm`` are
    RMSNorm over ``head_dim`` at eps 1e-5, and ``qkv``/``proj`` carry biases,
    which is what ``attention_bias: true`` asks for. The attention itself goes
    through vLLM's ``MMEncoderAttention``, for which this repo registers a
    Pallas kernel (``layers/vllm/vision_attention.py:72``).
    """

    def __init__(self, *args, swiglu_limit: float, mlp_hidden_dim: int,
                 **kwargs):
        super().__init__(*args, mlp_hidden_dim=mlp_hidden_dim, **kwargs)
        self.mlp = Glm5NextVisionMLP(
            kwargs.get("dim") or args[0],
            mlp_hidden_dim,
            bias=True,
            quant_config=kwargs.get("quant_config"),
            prefix=f"{kwargs.get('prefix', '')}.mlp",
            swiglu_limit=swiglu_limit,
        )


class Glm5NextVisionTransformer(GlmOcrVisionTransformer):
    """GLM-5.3-Flash's ViT.

    Two things GlmOcr gets from the wrong place, both silent if left alone:

    * ``context_dim``. ``glm_ocr.py:304`` passes
      ``text_config.intermediate_size``, which for GLM-5.3-Flash is 12288. The
      checkpoint's merger is 10240 wide -- ``projection_intermediate_size`` --
      so the weights would not fit.
    * ``norm_eps``. ``glm_ocr.py:390`` reads a *top-level* ``rms_norm_eps``.
      GLM's config has none, so it falls back to 1e-5, which happens to equal
      ``vision_config.rms_norm_eps``. Correct by luck; read it explicitly.
    """

    def __init__(
        self,
        text_config,
        vision_config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        norm_eps = getattr(vision_config, "rms_norm_eps", 1e-5)
        super().__init__(text_config,
                         vision_config,
                         norm_eps=norm_eps,
                         quant_config=quant_config,
                         prefix=prefix)

        swiglu_limit = getattr(vision_config, "swiglu_limit", 10.0)
        self.swiglu_limit = swiglu_limit
        norm_layer = partial(RMSNorm, eps=norm_eps)

        self.blocks = nn.ModuleList([
            Glm5NextVisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}",
                swiglu_limit=swiglu_limit,
            ) for i in range(vision_config.depth)
        ])

        self.merger = Glm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
            swiglu_limit=swiglu_limit,
        )

        # A third one, inside the inherited attention: ``glm_ocr.py:113-114``
        # hardcodes ``RMSNorm(head_dim, eps=1e-5)`` for q_norm/k_norm, while
        # transformers builds both from ``config.rms_norm_eps``
        # (``modeling_glm5_next.py:1596-1597``). They agree at GLM-5.3-Flash's
        # 1e-5 today, and a mismatch would not raise -- it would shift every
        # query and key by a fraction of a percent, which reads as "the port is
        # a bit off" rather than as a bug. Retie them so it cannot happen.
        for block in self.blocks:
            block.attn.q_norm.variance_epsilon = norm_eps
            block.attn.k_norm.variance_epsilon = norm_eps
