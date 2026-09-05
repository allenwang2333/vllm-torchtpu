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
"""GLM-5.3-Flash pieces that differ from DeepSeek-V4.

GLM-5.3-Flash reuses DeepSeek-V4's multi-hyper-connection residual manifold
almost verbatim -- same per-layer ``hc_attn_*`` / ``hc_ffn_*`` parameters, same
Sinkhorn-normalised combine, same deferred-post chaining. The one place they
part company is the final collapse, and it is a silent difference: both produce
a correctly-shaped tensor.
"""

import torch


def mhc_collapse_head_mean(hidden_states: torch.Tensor) -> torch.Tensor:
    """Collapse GLM-5.3-Flash's residual streams with an unweighted mean.

    DeepSeek-V4 collapses its ``hc_mult`` streams with a *learned, per-token*
    gate -- ``sigmoid(linear(rmsnorm(flat), hc_head_fn) * scale + base)`` --
    see ``models/vllm/deepseek_v4/layers.py:mhc_collapse_head``.
    GLM-5.3-Flash does not: ``Glm5NextTextHyperHead.forward`` in transformers
    is exactly ``hidden_streams.mean(dim=2)``, and the checkpoint confirms it
    by shipping no ``hc_head_fn`` / ``hc_head_base`` / ``hc_head_scale``
    tensors at all.

    Borrowing DeepSeek-V4's head here would need weights that do not exist, and
    filling them with anything plausible would degrade output rather than fail,
    so this stays a separate function.

    Args:
      hidden_states: ``[..., hc_mult, hidden_size]``.

    Returns:
      ``[..., hidden_size]``, averaged over the stream axis.
    """
    # Accumulate in fp32: hc_mult is small, but the streams are the residual
    # manifold and this is the last thing before the LM head.
    return hidden_states.float().mean(dim=-2).to(hidden_states.dtype)
