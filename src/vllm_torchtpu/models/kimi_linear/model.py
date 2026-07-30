# SPDX-License-Identifier: Apache-2.0
"""Kimi-Linear model definition for TPU.

There is deliberately no model class here. vLLM ships an in-tree
``KimiLinearForCausalLM`` (``vllm/model_executor/models/kimi_linear.py``,
registered in ``registry.py``) and it is reused unchanged: the TPU-specific part
is confined to the KDA attention layer, which is substituted via vLLM's
``PluggableLayer`` out-of-tree mechanism in ``kda.py``.

That is the whole point of the pluggable-layer approach over redefining the
model: weight loading, the MoE stack, the MLA layers at
``linear_attn_config["full_attn_layers"]``, and the decoder plumbing all stay
upstream's, so there is nothing here to keep in sync.

This module exists as the place to put a TPU-specific model definition *if* one
becomes necessary -- for example if the MoE or MLA layers need substituting the
way KDA does, which cannot be expressed as a layer swap. Until then it only
records what the target checkpoint looks like, so the assumptions baked into
``kda.py`` are written down somewhere:

``moonshotai/Kimi-Linear-48B-A3B-Instruct``
    hidden_size 2304, 27 layers, vocab 163840, bf16, **no quantization_config**
    linear_attn_config: num_heads 32, head_dim 128, short_conv_kernel_size 4
      kda_layers:        20 layers  [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,
                                     21,22,23,25,26]
      full_attn_layers:   7 layers  [4,8,12,16,20,24,27]  (MLA)
    MoE: 256 experts + shared experts, unquantized
    per-KDA-layer weights: q/k/v/g_a/g_b/f_a/f_b/b_proj, q/k/v_conv1d,
      A_log [1, 1, 32, 1], dt_bias [4096], o_norm [128], o_proj

Note the gate differs from Kimi-K3's: there is no ``gate_lower_bound`` in the
config, so the activation is the unbounded ``-exp(A_log) * softplus(g + dt_bias)``
rather than K3's ``-5.0 * sigmoid(...)``. ``kda.py`` passes
``lower_bound=None`` accordingly.
"""

ARCH_TO_CLASS: dict[str, str] = {
    # Intentionally empty: the in-tree KimiLinearForCausalLM is used as-is.
    # Adding an entry here would override it globally for this environment.
}

__all__ = ["ARCH_TO_CLASS"]
