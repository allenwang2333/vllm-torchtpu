# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Ragged Kimi Delta Attention Pallas kernel."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kda_step(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    state: jax.Array,
    decay: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    *,
    lower_bound: float | None,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """One fp32 KDA recurrence step, shared by the kernel and tests."""
    activation_dtype = query.dtype
    query = query.astype(jnp.float32)
    query *= jax.lax.rsqrt(
        jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    query *= query.shape[-1]**-0.5
    key = key.astype(jnp.float32)
    key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    value = value.astype(jnp.float32)

    gate_input = raw_gate.astype(jnp.float32) + dt_bias
    if lower_bound is None:
        gate = -decay * jax.nn.softplus(gate_input)
    else:
        gate = lower_bound * jax.nn.sigmoid(decay * gate_input)

    state = state.astype(jnp.float32) * jnp.exp(gate)[..., None]
    prediction = jnp.einsum("hk,hkv->hv",
                            key,
                            state,
                            preferred_element_type=jnp.float32)
    delta = jax.nn.sigmoid(beta.astype(
        jnp.float32))[:, None] * (value - prediction)
    state += key[..., None] * delta[:, None, :]

    output = jnp.einsum("hk,hkv->hv",
                        query,
                        state,
                        preferred_element_type=jnp.float32)
    output = output.astype(activation_dtype).astype(jnp.float32)
    output *= jax.lax.rsqrt(
        jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output *
              norm_weight.astype(jnp.float32)).astype(activation_dtype)
    output *= jax.nn.sigmoid(output_gate)
    return output, state


def _kda_kernel(
    mixed_qkv_ref,
    raw_gate_ref,
    beta_ref,
    output_gate_ref,
    a_log_ref,
    dt_bias_ref,
    norm_weight_ref,
    query_start_loc_ref,
    state_indices_ref,
    seq_lens_ref,
    _output_init_ref,
    state_in_ref,
    output_ref,
    state_out_ref,
    state_scratch_ref,
    qkv_scratch_ref,
    gate_scratch_ref,
    beta_scratch_ref,
    output_gate_scratch_ref,
    output_scratch_ref,
    read_semaphores,
    write_semaphores,
    *,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    eps: float,
):
    request_idx = pl.program_id(0)
    start = query_start_loc_ref[request_idx]
    end = query_start_loc_ref[request_idx + 1]
    query_len = end - start
    state_idx = state_indices_ref[request_idx]
    has_initial_state = (state_idx >= 0) & (seq_lens_ref[request_idx]
                                            > query_len)

    state_scratch_ref[0] = jnp.zeros((num_heads, head_dim, head_dim),
                                     state_scratch_ref.dtype)

    @pl.when(has_initial_state)
    def _load_state():
        state_copy = pltpu.make_async_copy(
            state_in_ref.at[pl.ds(state_idx, 1), :, :, :],
            state_scratch_ref,
            read_semaphores.at[0],
        )
        state_copy.start()
        state_copy.wait()

    decay = jnp.exp(a_log_ref[...].astype(jnp.float32))[:, None]
    dt_bias = dt_bias_ref[...].reshape(num_heads, head_dim).astype(jnp.float32)
    norm_weight = norm_weight_ref[...]

    def process_token(token_offset, _):
        token_idx = start + token_offset
        qkv_copy = pltpu.make_async_copy(
            mixed_qkv_ref.at[pl.ds(token_idx, 1), :],
            qkv_scratch_ref,
            read_semaphores.at[1],
        )
        gate_copy = pltpu.make_async_copy(
            raw_gate_ref.at[pl.ds(token_idx, 1), :],
            gate_scratch_ref,
            read_semaphores.at[2],
        )
        beta_copy = pltpu.make_async_copy(
            beta_ref.at[pl.ds(token_idx, 1), :],
            beta_scratch_ref,
            read_semaphores.at[3],
        )
        output_gate_copy = pltpu.make_async_copy(
            output_gate_ref.at[pl.ds(token_idx, 1), :],
            output_gate_scratch_ref,
            read_semaphores.at[4],
        )
        qkv_copy.start()
        gate_copy.start()
        beta_copy.start()
        output_gate_copy.start()
        qkv_copy.wait()
        gate_copy.wait()
        beta_copy.wait()
        output_gate_copy.wait()

        qkv = qkv_scratch_ref[0].reshape(3, num_heads, head_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]
        raw_gate = gate_scratch_ref[0].reshape(num_heads, head_dim)
        beta = beta_scratch_ref[0]
        output_gate = output_gate_scratch_ref[0].reshape(num_heads, head_dim)
        output, state = kda_step(
            query,
            key,
            value,
            raw_gate,
            beta,
            output_gate,
            state_scratch_ref[0],
            decay,
            dt_bias,
            norm_weight,
            lower_bound=lower_bound,
            eps=eps,
        )
        state_scratch_ref[0] = state.astype(state_scratch_ref.dtype)

        output_scratch_ref[0] = output.astype(output_scratch_ref.dtype)

        output_copy = pltpu.make_async_copy(
            output_scratch_ref,
            output_ref.at[pl.ds(token_idx, 1), :, :],
            write_semaphores.at[0],
        )
        output_copy.start()
        output_copy.wait()

    jax.lax.fori_loop(0, query_len, process_token, None)

    @pl.when((state_idx >= 0) & (query_len > 0))
    def _store_state():
        state_copy = pltpu.make_async_copy(
            state_scratch_ref,
            state_out_ref.at[pl.ds(state_idx, 1), :, :, :],
            write_semaphores.at[1],
        )
        state_copy.start()
        state_copy.wait()


@functools.partial(
    jax.jit,
    static_argnames=("lower_bound", "eps"),
    donate_argnames=("recurrent_state", ),
)
def ragged_kda(
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
    *,
    lower_bound: float | None,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Run KDA over flattened ragged requests and update their recurrent state."""
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

    output_init = jnp.zeros((num_tokens, num_heads, head_dim),
                            dtype=mixed_qkv.dtype)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    output, new_state = pl.pallas_call(
        functools.partial(
            _kda_kernel,
            num_heads=num_heads,
            head_dim=head_dim,
            lower_bound=lower_bound,
            eps=eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(output_init.shape, output_init.dtype),
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
        ),
        grid_spec=pl.GridSpec(
            grid=(state_indices.shape[0], ),
            in_specs=[
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                vmem_spec,
                vmem_spec,
                vmem_spec,
                smem_spec,
                smem_spec,
                smem_spec,
                hbm_spec,
                hbm_spec,
            ],
            out_specs=[hbm_spec, hbm_spec],
            scratch_shapes=[
                pltpu.VMEM((1, num_heads, head_dim, head_dim),
                           recurrent_state.dtype),
                pltpu.VMEM((1, mixed_dim), mixed_qkv.dtype),
                pltpu.VMEM((1, num_heads * head_dim), raw_gate.dtype),
                pltpu.VMEM((1, num_heads), beta.dtype),
                pltpu.VMEM((1, num_heads * head_dim), output_gate.dtype),
                pltpu.VMEM((1, num_heads, head_dim), mixed_qkv.dtype),
                pltpu.SemaphoreType.DMA((5, )),
                pltpu.SemaphoreType.DMA((2, )),
            ],
        ),
        input_output_aliases={
            10: 0,
            11: 1,
        },
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="kimi_k3_ragged_kda",
    )(
        mixed_qkv,
        raw_gate,
        beta,
        output_gate,
        a_log,
        dt_bias,
        norm_weight,
        query_start_loc,
        state_indices,
        seq_lens,
        output_init,
        recurrent_state,
    )
    return output, new_state
