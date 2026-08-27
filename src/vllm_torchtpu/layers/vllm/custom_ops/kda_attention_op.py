# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch custom-op bridge for Kimi KDA on the fused conv1d + GDN v3 kernel."""

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper


def _gated_output_norm(
    output: jax.Array,  # [num_rows, H, D]
    output_gate: jax.Array,  # [num_rows, H * D] raw gate
    norm_weight: jax.Array,  # [D]
    eps: float,
    activation_dtype,
) -> jax.Array:
    """Per-head ``o_norm`` followed by the sigmoid output gate."""
    output = output.astype(activation_dtype).astype(jnp.float32)
    output *= jax.lax.rsqrt(
        jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output *
              norm_weight.astype(jnp.float32)).astype(activation_dtype)
    return output * jax.nn.sigmoid(output_gate.reshape(output.shape))


def _check_kda_abi(
    mixed_qkv: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    recurrent_state: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
) -> tuple[int, int, int, int]:
    """Validate the shared KDA operand contract; return the shape tuple.

    Deliberately the same contract `ragged_kda` enforces, so swapping ops cannot
    silently change what a caller is allowed to pass.
    """
    num_tokens, mixed_dim = mixed_qkv.shape
    num_heads, head_dim, state_v_dim = recurrent_state.shape[1:]
    if state_v_dim != head_dim or mixed_dim != 3 * num_heads * head_dim:
        raise ValueError("Incompatible KDA activation and state shapes")
    if raw_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA gate shape")
    if beta.shape != (num_tokens, num_heads):
        raise ValueError("Incompatible KDA beta shape")
    if output_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA output-gate shape")
    num_sequences = state_indices.shape[0]
    if a_log.shape != (num_heads, ):
        raise ValueError("KDA A_log shape does not match the head count")
    if dt_bias.shape != (num_heads * head_dim, ):
        raise ValueError("KDA dt_bias shape does not match the projection")
    if norm_weight.shape != (head_dim, ):
        raise ValueError("KDA norm weight shape does not match the head size")
    if query_start_loc.shape != (num_sequences + 1, ):
        raise ValueError("KDA query starts and state indices disagree")
    if seq_lens.shape != (num_sequences, ):
        raise ValueError("KDA sequence lengths and state indices disagree")
    if (raw_gate.dtype != mixed_qkv.dtype or beta.dtype != mixed_qkv.dtype
            or output_gate.dtype != mixed_qkv.dtype):
        raise TypeError("KDA activations must have one common dtype")
    if recurrent_state.dtype != jnp.float32:
        raise TypeError("KDA recurrent state must be float32")
    if a_log.dtype != jnp.float32 or dt_bias.dtype != jnp.float32:
        raise TypeError("KDA A_log and dt_bias must be float32")
    if (query_start_loc.dtype != jnp.int32 or state_indices.dtype != jnp.int32
            or seq_lens.dtype != jnp.int32):
        raise TypeError("KDA metadata tensors must be int32")
    return num_tokens, num_sequences, num_heads, head_dim


def _build_kda_core(lower_bound: float | None, eps: float):
    """The KDA op, on the fused conv1d + GDN v3 kernel.

    One Pallas call covers the whole batch: the fused kernel owns the
    convolution, silu, q/k L2-norm, the gate activation, beta's sigmoid, both
    recurrences and the state write-back, and it dispatches decode against
    prefill/mixed off `distribution` inside the kernel rather than by emitting
    both paths and selecting per token.

    Only `_gated_output_norm` stays outside, since the fused kernel has no
    notion of the output gate.
    """

    def kda_core(
        mixed_qkv: jax.Array,  # [T, 3 * H * D], pre-convolution
        raw_gate: jax.Array,  # [T, H * D]
        beta: jax.Array,  # [T, H]
        output_gate: jax.Array,  # [T, H * D]
        conv_state: jax.Array,  # [num_slots, K - 1, 3, H, D]
        recurrent_state: jax.Array,  # [num_slots, H, K, V]
        conv_weight: jax.Array,  # [kernel_size, 3, H, D], fused at load time
        a_log: jax.Array,  # [H]
        dt_bias: jax.Array,  # [H * D]
        norm_weight: jax.Array,  # [D]
        query_start_loc: jax.Array,  # [N + 1]
        state_indices: jax.Array,  # [N]
        seq_lens: jax.Array,  # [N]
        distribution: jax.Array,  # [3] int32: [decode_end, _, mixed_end]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        num_tokens, _, num_heads, head_dim = _check_kda_abi(
            mixed_qkv, raw_gate, beta, output_gate, recurrent_state, a_log,
            dt_bias, norm_weight, query_start_loc, state_indices, seq_lens)
        kernel_size = conv_weight.shape[0]

        # `mixed_qkv`, `conv_state` and `conv_weight` all lay their channels
        # out as (3, H, D) in row-major order, which is exactly the flat
        # `dim` axis the fused kernel indexes -- the reshapes are free. The
        # kernel wants the weight transposed to [dim, 1, kernel_size].
        conv_weight_flat = conv_weight.reshape(kernel_size, -1)
        conv_weight_flat = jnp.transpose(conv_weight_flat, (1, 0))[:, None]

        (new_conv_state, new_pool), out = gdn_wrapper.fused_conv1d_gdn(
            qkv=mixed_qkv,
            # Raw: the kernel applies sigmoid to `b` and the gate activation
            # to `a` itself.
            b=beta,
            a=raw_gate,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight_flat,
            # Kimi's short convolution is bias-free.
            conv_bias=None,
            a_log=a_log,
            # Per-channel under KDA, and Mosaic rejects the flat form.
            dt_bias=dt_bias.reshape(num_heads, head_dim),
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=num_heads,
            n_v=num_heads,
            d_k=head_dim,
            d_v=head_dim,
            kernel_size=kernel_size,
            attention_mode=gdn_config.AttentionMode.KDA,
            gate_lower_bound=lower_bound,
        )

        output = out.reshape(num_tokens, num_heads, head_dim)
        output = _gated_output_norm(output, output_gate, norm_weight, eps,
                                    mixed_qkv.dtype)
        return output, new_conv_state, new_pool

    return kda_core


def build_kimi_dispatched_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
):
    """Build the KDA op on the fused conv1d + GDN v3 kernel.

    Takes the *pre-convolution* projection output and owns both caches: the
    kernel fuses the short convolution into the recurrence, so it cannot be
    fed by a separate convolution op.
    """
    core = _build_kda_core(lower_bound, eps)

    op_name = f"pallas::kimi_dispatched_kda_{prefix.replace('.', '_')}"
    dispatched_op = pallas.jax_op(op_name, core, donate_argnums=(4, 5))

    def _fake_dispatched(mixed_qkv, _raw_gate, _beta, _output_gate, conv_state,
                         recurrent_state, *args, **kwargs):
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty((mixed_qkv.size(0), num_heads, head_dim),
                             dtype=mixed_qkv.dtype,
                             device=mixed_qkv.device)
        return (output, torch.empty_like(conv_state),
                torch.empty_like(recurrent_state))

    dispatched_op.register_fake(_fake_dispatched)

    def dispatched_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        output, new_conv_state, new_pool = dispatched_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            conv_state,
            recurrent_state,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
            distribution,
        )
        conv_state.copy_(new_conv_state)
        recurrent_state.copy_(new_pool)
        return output

    return dispatched_impl
