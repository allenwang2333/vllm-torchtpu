# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Kimi Delta Attention kernels and custom-op helpers."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.kernels.kimi_k3 import kda_step, ragged_kda
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import \
    kimi_short_conv_scan


def test_kda_pallas_abi_shapes_and_dtypes() -> None:
    num_tokens, num_heads, head_dim = 5, 2, 128
    recurrent_state = jnp.zeros((8, num_heads, head_dim, head_dim),
                                jnp.float32)
    args = (
        jnp.zeros((num_tokens, 3 * num_heads * head_dim), jnp.bfloat16),
        jnp.zeros((num_tokens, num_heads * head_dim), jnp.bfloat16),
        jnp.zeros((num_tokens, num_heads), jnp.bfloat16),
        jnp.zeros((num_tokens, num_heads * head_dim), jnp.bfloat16),
        recurrent_state,
        jnp.zeros((num_heads, ), jnp.float32),
        jnp.zeros((num_heads * head_dim, ), jnp.float32),
        jnp.ones((head_dim, ), jnp.bfloat16),
        jnp.asarray([0, 2, 5, 5], jnp.int32),
        jnp.asarray([0, 1, -1], jnp.int32),
        jnp.asarray([2, 7, 0], jnp.int32),
    )
    output, new_state = jax.eval_shape(
        functools.partial(ragged_kda, lower_bound=None, eps=1e-5),
        *args,
    )
    assert output.shape == (num_tokens, num_heads, head_dim)
    assert output.dtype == jnp.bfloat16
    assert new_state.shape == recurrent_state.shape
    assert new_state.dtype == jnp.float32


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_kda_step_matches_independent_torch_reference(
        lower_bound: float | None) -> None:
    torch.manual_seed(10)
    num_tokens, num_heads, head_dim = 4, 2, 4

    def activation(shape):
        return torch.randn(shape).bfloat16().float()

    query = activation((num_tokens, num_heads, head_dim))
    key = activation((num_tokens, num_heads, head_dim))
    value = activation((num_tokens, num_heads, head_dim))
    raw_gate = activation((num_tokens, num_heads, head_dim))
    beta = activation((num_tokens, num_heads))
    output_gate = activation((num_tokens, num_heads, head_dim))
    state = torch.randn(num_heads, head_dim, head_dim)
    decay = torch.randn(num_heads).exp()[:, None]
    dt_bias = torch.randn(num_heads, head_dim)
    norm_weight = activation((head_dim, ))
    eps = 1e-5

    jax_state = jnp.asarray(state.numpy())
    actual_outputs = []
    for token_idx in range(num_tokens):
        output, jax_state = kda_step(
            jnp.asarray(query[token_idx].numpy(), dtype=jnp.bfloat16),
            jnp.asarray(key[token_idx].numpy(), dtype=jnp.bfloat16),
            jnp.asarray(value[token_idx].numpy(), dtype=jnp.bfloat16),
            jnp.asarray(raw_gate[token_idx].numpy(), dtype=jnp.bfloat16),
            jnp.asarray(beta[token_idx].numpy(), dtype=jnp.bfloat16),
            jnp.asarray(output_gate[token_idx].numpy(), dtype=jnp.bfloat16),
            jax_state,
            jnp.asarray(decay.numpy()),
            jnp.asarray(dt_bias.numpy()),
            jnp.asarray(norm_weight.numpy(), dtype=jnp.bfloat16),
            lower_bound=lower_bound,
            eps=eps,
        )
        actual_outputs.append(np.asarray(output.astype(jnp.float32)))

    expected_state = state
    expected_outputs = []
    for token_idx in range(num_tokens):
        q = query[token_idx]
        q *= torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
        q *= head_dim**-0.5
        k = key[token_idx]
        k *= torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
        gate_input = raw_gate[token_idx] + dt_bias
        if lower_bound is None:
            gate = -decay * F.softplus(gate_input)
        else:
            gate = lower_bound * torch.sigmoid(decay * gate_input)
        expected_state = expected_state * gate.exp().unsqueeze(-1)
        prediction = torch.einsum("hk,hkv->hv", k, expected_state)
        delta = torch.sigmoid(
            beta[token_idx])[:, None] * (value[token_idx] - prediction)
        expected_state += torch.einsum("hk,hv->hkv", k, delta)
        output = torch.einsum("hk,hkv->hv", q, expected_state).bfloat16()
        output_float = output.float()
        output_float *= torch.rsqrt(
            output_float.square().mean(dim=-1, keepdim=True) + eps)
        output = (output_float * norm_weight).bfloat16()
        output *= torch.sigmoid(output_gate[token_idx].bfloat16())
        expected_outputs.append(output.float())

    torch.testing.assert_close(
        torch.from_numpy(np.stack(actual_outputs)),
        torch.stack(expected_outputs),
        rtol=5e-2,
        atol=4e-3,
    )
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(jax_state).copy()),
        expected_state,
        rtol=1e-2,
        atol=5e-4,
    )


def test_kimi_short_conv_scan_resets_each_new_sequence() -> None:
    x = jnp.asarray([[1, 2], [3, 4], [5, 6]], dtype=jnp.bfloat16)
    state = jnp.zeros((2, 2, 2), dtype=jnp.bfloat16)
    weight = jnp.ones((3, 2), dtype=jnp.bfloat16)

    output, new_state = kimi_short_conv_scan(
        x,
        state,
        weight,
        jnp.asarray([0, 2, 3], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([2, 1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(
        np.asarray(output),
        np.asarray([[1, 2], [4, 6], [5, 6]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(new_state),
        np.asarray([[[1, 2], [3, 4]], [[0, 0], [5, 6]]], dtype=np.float32),
    )


def test_kimi_short_conv_scan_isolates_full_mixed_batch_slots() -> None:
    """One decode plus seven prefills must not share convolution history."""
    x = jnp.arange(1, 17, dtype=jnp.bfloat16)[:, None]
    state = jnp.asarray(
        [
            [[99], [98]],
            [[10], [20]],
            [[77], [88]],
            [[77], [88]],
            [[77], [88]],
            [[77], [88]],
            [[77], [88]],
            [[77], [88]],
            [[77], [88]],
        ],
        dtype=jnp.bfloat16,
    )
    weight = jnp.ones((3, 1), dtype=jnp.bfloat16)

    output, new_state = kimi_short_conv_scan(
        x,
        state,
        weight,
        jnp.asarray([0, 1, 3, 5, 7, 9, 11, 13, 15], dtype=jnp.int32),
        jnp.arange(1, 9, dtype=jnp.int32),
        jnp.asarray([5, 2, 2, 2, 2, 2, 2, 2], dtype=jnp.int32),
    )

    # Slot 1 carries the decode request's history. Every fresh prefill must
    # ignore its deliberately stale slot, and the final padded row must write
    # neither output nor state. Slot 0 is the reserved null block.
    np.testing.assert_array_equal(
        np.asarray(output).reshape(-1),
        np.asarray(
            [31, 2, 5, 4, 9, 6, 13, 8, 17, 10, 21, 12, 25, 14, 29, 0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(new_state).reshape(9, 2),
        np.asarray(
            [
                [99, 98],
                [20, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize('length', [0, 1, 2, 127, 256])
@pytest.mark.parametrize('carry', [False, True])
def test_single_prefill_convolution_preserves_output_and_cache(length, carry):
    from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
        _single_prefill_conv, kimi_short_conv_scan)

    rng = np.random.default_rng(621)
    x = jnp.asarray(rng.normal(size=(256, 768)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(0, 0.1, (4, 768)), jnp.bfloat16)
    state = jnp.asarray(rng.normal(0, 0.1, (4, 3, 3, 2, 128)), jnp.float32)
    starts = jnp.asarray([0, length, length], jnp.int32)
    indices = jnp.asarray([2, 0], jnp.int32)
    lengths = jnp.asarray([length + (10 if carry else 0), 0], jnp.int32)
    expected = jax.jit(kimi_short_conv_scan)(x, state, weights, starts,
                                             indices, lengths)
    actual = jax.jit(_single_prefill_conv)(x, state, weights, indices[0],
                                           jnp.int32(length),
                                           jnp.asarray(carry))
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(actual[1][jnp.array([0, 1, 3])],
                                  state[jnp.array([0, 1, 3])])


@pytest.mark.parametrize('case', ['fresh', 'carried', 'mixed', 'padded'])
@pytest.mark.parametrize('tokens',
                         [1, 32, 64, 96, 1024, 2048, 4096, 8192, 16384])
def test_mega_dispatch_preserves_state_and_fused_fallback(
        monkeypatch, case, tokens):
    from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as op

    heads, dim, slots = 1, 128, 3
    x = jnp.ones((tokens, 3 * heads * dim), jnp.bfloat16)
    conv = jnp.ones((slots, 3, 3 * heads * dim), jnp.bfloat16)
    state = jnp.ones((slots, heads, dim, dim), jnp.float32)

    def fused(**kw):
        return ((kw['conv_state'] + 2, kw['recurrent_state'] + 2),
                jnp.zeros((tokens, heads, dim), x.dtype))

    def mega(q, k, v, g, beta, **kw):
        return jnp.zeros_like(q), kw['initial_state'] + 3

    monkeypatch.setattr(op.gdn_wrapper, 'fused_conv1d_gdn', fused)
    monkeypatch.setattr(op, 'kda_forward_inference', mega)
    starts = [0, 1, tokens] if case == 'mixed' else [0, tokens, tokens]
    if case == 'padded':
        starts = [0, tokens - 1, tokens - 1]
    args = (x, jnp.zeros(
        (tokens, heads * dim), x.dtype), jnp.zeros(
            (tokens, heads), x.dtype), jnp.zeros(
                (tokens, heads * dim),
                x.dtype), conv, state, jnp.ones((4, 3, heads, dim), x.dtype),
            jnp.zeros(
                (heads, ), jnp.float32), jnp.zeros(
                    (heads * dim, ), jnp.float32), jnp.ones((dim, ), x.dtype),
            jnp.array(starts, jnp.int32), jnp.array([2, 0], jnp.int32),
            jnp.array([tokens + (10 if case == 'carried' else 0), 0],
                      jnp.int32),
            jnp.array([1 if case == 'mixed' else 0, 0, 1], jnp.int32))
    core = op._build_fused_core(-5.0, 1e-5)
    output, new_conv, new_state = jax.jit(core)(*args)
    assert output.shape == (tokens, heads, dim)
    assert output.dtype == x.dtype
    if case in ('fresh', 'carried') and tokens % 64 == 0:
        np.testing.assert_array_equal(new_state[:2], state[:2])
        np.testing.assert_array_equal(new_state[2],
                                      4 if case == 'carried' else 3)
        np.testing.assert_array_equal(new_conv, conv)
    else:
        np.testing.assert_array_equal(new_state, state + 2)
        np.testing.assert_array_equal(new_conv, conv + 2)
