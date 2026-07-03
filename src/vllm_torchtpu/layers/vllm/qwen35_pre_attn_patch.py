# SPDX-License-Identifier: Apache-2.0
"""Wire the fused_add_rmsnorm_qkv_fp8 Pallas kernel into Qwen3.5.

Monkey-patches ``Qwen3NextDecoderLayer.forward`` and
``Qwen3NextAttention.forward`` so that for full-attention layers with a
non-None residual, the input_layernorm + qkv_proj pair is replaced by a
single Pallas call. Everything downstream (split → q_norm/k_norm →
RoPE → attention → attn_output_gate → o_proj) is preserved verbatim.

Gate:
- ``residual is not None`` (layer 1+)
- layer_type == "full_attention"
- x.device.type == "tpu"
- x.dtype == torch.bfloat16
- qkv_proj is FP8-quantized (``w_q`` dtype ``torch.float8_e4m3fn`` and
  ``weight_scale`` present, 1-D per-channel).
- x.shape[-1] % 128 == 0

Falls through to the original forward when any of these are false.

This patch composes cleanly with the Target B ``fused_add_rmsnorm``
patch (see ``rmsnorm_patch.py``): the DecoderLayer patch runs first for
full-attention layers, which BYPASSES the ``input_layernorm(x, residual)``
call altogether. Post-attention layernorm still hits the Target B patch
path.

.. warning::
   **NOT PRODUCTION-READY — installer intentionally not wired.** An
   adversarial review of this session's work found three blockers that
   must be resolved before ``_install_qwen35_pre_attn_pallas_patch`` is
   called from ``apply_tpu_patches``:

   1. It wires ``fused_add_rmsnorm_qkv_fp8``, which is **W8A16** (bf16
      activations) and does not match Qwen3.5's production **W8A8**
      (fp8-activation) numerics — see that kernel's docstring warning.
      Rework to wire the arena W8A8 winner
      (``agent_winners/qwen35_residual_rmsnorm_qkv_fp8.py``) instead.
   2. It drops the post-MLP ``ffn_layer_scale`` gain
      (upstream ``qwen3_next.py`` applies ``hidden *= (ffn_layer_scale
      + 1)`` after the MLP) — silent wrong output when
      ``config.layer_scale`` is set. The gate must also check
      ``qkv_proj.bias is None`` (``qkv_bias`` is dropped otherwise).
   3. Numeric equivalence must be confirmed with a token-level greedy
      diff vs the unpatched model, not just a smoke test.

   Retained as scaffolding + a documented design; do not install until
   1–3 are addressed.
"""

from __future__ import annotations

from typing import Optional

import torch


def _install_qwen35_pre_attn_pallas_patch() -> None:
    """Install the DecoderLayer + Attention monkey-patches. Idempotent."""

    try:
        from vllm.model_executor.models.qwen3_next import (
            Qwen3NextAttention, Qwen3NextDecoderLayer)
    except ImportError:
        # Qwen3.5 / Qwen3Next model not present in this vLLM install;
        # nothing to patch.
        return

    if getattr(Qwen3NextDecoderLayer, "_torchtpu_fused_pre_attn_patched",
               False):
        return

    _orig_decoder_forward = Qwen3NextDecoderLayer.forward
    _orig_attn_forward = Qwen3NextAttention.forward

    def _decoder_patched(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        # Only intercept the full-attention + fused-add case. Layer 0 (no
        # residual) and DeltaNet layers go through the original path.
        if (residual is None or self.layer_type != "full_attention"
                or hidden_states.device.type != "tpu"
                or hidden_states.dtype != torch.bfloat16
                or hidden_states.shape[-1] % 128 != 0):
            return _orig_decoder_forward(self,
                                         hidden_states,
                                         residual,
                                         positions=positions,
                                         **kwargs)

        # Confirm qkv_proj is FP8-quantized per-channel — the fused
        # kernel's contract requires this layout.
        qkv_layer = self.self_attn.qkv_proj
        w_q = getattr(qkv_layer, "weight", None)
        w_s = getattr(qkv_layer, "weight_scale", None)
        if (w_q is None or w_s is None or w_q.dtype != torch.float8_e4m3fn
                or w_s.dim() != 1):
            return _orig_decoder_forward(self,
                                         hidden_states,
                                         residual,
                                         positions=positions,
                                         **kwargs)

        # Fused kernel call — replaces
        #   hidden_states, residual = self.input_layernorm(hidden_states, residual)
        #   qkv, _ = self.self_attn.qkv_proj(hidden_states)
        from vllm_torchtpu.kernels.fused_add_rmsnorm_qkv_fp8.torch_op import \
            fused_add_rmsnorm_qkv_fp8_torch

        # GemmaRMSNorm applies (weight + 1) — same convention as
        # Target B (see rmsnorm_patch.py).
        rms_w_eff = (
            self.input_layernorm.weight.data.to(dtype=hidden_states.dtype) +
            1.0)
        qkv, new_residual = fused_add_rmsnorm_qkv_fp8_torch(
            hidden_states, residual, rms_w_eff, w_q, w_s)

        # Downstream attention path (replicates
        # Qwen3NextAttention.forward starting from ``qkv``).
        self_attn = self.self_attn
        self_attention_output = torch.empty_like(new_residual)
        _attn_forward_from_qkv(self_attn, qkv, positions,
                               self_attention_output)

        hidden_states = self_attention_output

        # Layer scale (matches the original decoder forward).
        if getattr(self, "layer_scale", False):
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1)

        # Post-attention layernorm + MLP (unchanged from upstream). The
        # Target B patch will intercept post_attention_layernorm for us.
        hidden_states, new_residual = self.post_attention_layernorm(
            hidden_states, new_residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, new_residual

    def _attn_forward_from_qkv(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """The tail of Qwen3NextAttention.forward starting from qkv."""
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)

    Qwen3NextDecoderLayer.forward = _decoder_patched  # type: ignore[assignment]
    Qwen3NextDecoderLayer._torchtpu_fused_pre_attn_patched = True  # type: ignore[attr-defined]
