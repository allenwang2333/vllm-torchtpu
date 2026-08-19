# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``decode_kda`` against a plain-JAX restatement of its contract.

Runs the Pallas call in interpret mode, so this executes on any backend and
needs no TPU. Interpret mode simulates the kernel, not the hardware: it does
not model memory spaces, DMA, or lane/sublane layout, so passing here means
the arithmetic and indexing are right, never that the kernel is correct on
device. It exists to catch logic regressions before a device run is spent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.kimi_k3.decode_kda import decode_kda

NUM_SEQS = 2
NUM_HEADS = 2
HEAD_DIM = 128
KERNEL_SIZE = 4
NUM_SLOTS = 4
SCALE = HEAD_DIM**-0.5
RTOL = 1e-5


@pytest.fixture(autouse=True)
def _force_interpret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PALLAS_INTERPRET", "1")


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _l2(x: jax.Array) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def _inputs(seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    return {
        "mixed_qkv":
        jax.random.normal(keys[0], (NUM_SEQS, 3 * NUM_HEADS * HEAD_DIM)) * 0.5,
        "g":
        jax.random.normal(keys[1], (NUM_SEQS, NUM_HEADS, HEAD_DIM)) * 0.5,
        "beta":
        jax.nn.sigmoid(jax.random.normal(keys[2], (NUM_SEQS, NUM_HEADS))),
        "conv_state":
        jax.random.normal(
            keys[3],
            (NUM_SLOTS, KERNEL_SIZE - 1, 3, NUM_HEADS, HEAD_DIM)) * 0.5,
        "conv_weight":
        jax.random.normal(keys[4],
                          (KERNEL_SIZE, 3, NUM_HEADS, HEAD_DIM)) * 0.5,
        "state":
        jax.random.normal(keys[5],
                          (NUM_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM)) * 0.1,
        "a_log":
        jax.random.normal(keys[6], (NUM_HEADS, )) * 0.2,
        "dt_bias":
        jax.random.normal(keys[7], (NUM_HEADS, HEAD_DIM)) * 0.1,
    }


def _reference(inp, state_indices, has_initial_state, *, lower_bound):
    """One sequence at a time, in the order the kernel performs them."""
    conv_state, state = inp["conv_state"], inp["state"]
    outs = []
    new_conv, new_state = list(conv_state), list(state)
    for s in range(NUM_SEQS):
        slot = int(state_indices[s])
        valid = slot > 0
        carried = valid and bool(has_initial_state[s])

        prev = conv_state[slot] if carried else jnp.zeros_like(conv_state[0])
        token = inp["mixed_qkv"][s].reshape(3, NUM_HEADS, HEAD_DIM)
        conv_out = token * inp["conv_weight"][KERNEL_SIZE - 1]
        for tap in range(KERNEL_SIZE - 1):
            conv_out = conv_out + prev[tap] * inp["conv_weight"][tap]
        conv_out = _silu(conv_out)

        q = _l2(conv_out[0]) * SCALE
        k = _l2(conv_out[1])
        v = conv_out[2]

        gate_in = inp["g"][s] + inp["dt_bias"]
        a = jnp.exp(inp["a_log"]).reshape(NUM_HEADS, 1)
        gate = (-a * jax.nn.softplus(gate_in) if lower_bound is None else
                lower_bound * jax.nn.sigmoid(a * gate_in))

        decayed = (state[slot] if carried else
                   jnp.zeros_like(state[0])) * jnp.exp(gate)[..., None]
        v_updated = jnp.einsum("hk,hkv->hv", k, decayed)
        v_new = inp["beta"][s][:, None] * (v - v_updated)
        updated = decayed + k[..., None] * v_new[:, None, :]

        outs.append(
            jnp.where(valid, jnp.einsum("hk,hkv->hv", q, updated), 0.0))
        if valid:
            new_conv[slot] = jnp.concatenate([prev[1:], token[None]], axis=0)
            new_state[slot] = updated
    return jnp.stack(outs), jnp.stack(new_conv), jnp.stack(new_state)


def _assert_close(name: str, got: jax.Array, want: jax.Array) -> None:
    scale = float(jnp.max(jnp.abs(want))) + 1e-12
    rel = float(jnp.max(jnp.abs(got - want))) / scale
    assert rel < RTOL, f"{name} relative error {rel:.3e} exceeds {RTOL}"


@pytest.mark.parametrize("lower_bound", [None, -1.0])
def test_decode_kda_matches_reference(lower_bound: float | None) -> None:
    inp = _inputs()
    state_indices = jnp.array([1, 2], jnp.int32)
    has_initial_state = jnp.array([1, 1], jnp.int32)

    got = decode_kda(inp["mixed_qkv"],
                     inp["g"],
                     inp["beta"],
                     inp["conv_state"],
                     inp["conv_weight"],
                     inp["state"],
                     inp["a_log"],
                     inp["dt_bias"],
                     state_indices,
                     has_initial_state,
                     lower_bound=lower_bound,
                     compute_dtype=jnp.float32,
                     scale=SCALE)
    want = _reference(inp,
                      state_indices,
                      has_initial_state,
                      lower_bound=lower_bound)

    for name, a, b in zip(("out", "conv_state", "state"), got, want):
        _assert_close(name, a, b)


def test_first_step_ignores_slot_contents() -> None:
    """``has_initial_state=0`` must start from zero, not from stale slot bytes."""
    inp = _inputs(seed=5)
    state_indices = jnp.array([1, 2], jnp.int32)
    has_initial_state = jnp.array([0, 0], jnp.int32)

    got = decode_kda(inp["mixed_qkv"],
                     inp["g"],
                     inp["beta"],
                     inp["conv_state"],
                     inp["conv_weight"],
                     inp["state"],
                     inp["a_log"],
                     inp["dt_bias"],
                     state_indices,
                     has_initial_state,
                     lower_bound=None,
                     compute_dtype=jnp.float32,
                     scale=SCALE)
    want = _reference(inp, state_indices, has_initial_state, lower_bound=None)
    for name, a, b in zip(("out", "conv_state", "state"), got, want):
        _assert_close(name, a, b)


def test_null_slot_emits_zeros_and_leaves_state() -> None:
    """Slot 0 is the reserved null block: emit nothing, write nothing."""
    inp = _inputs(seed=9)
    state_indices = jnp.array([1, 0], jnp.int32)
    has_initial_state = jnp.array([1, 1], jnp.int32)

    out, _, state_out = decode_kda(inp["mixed_qkv"],
                                   inp["g"],
                                   inp["beta"],
                                   inp["conv_state"],
                                   inp["conv_weight"],
                                   inp["state"],
                                   inp["a_log"],
                                   inp["dt_bias"],
                                   state_indices,
                                   has_initial_state,
                                   lower_bound=None,
                                   compute_dtype=jnp.float32,
                                   scale=SCALE)
    assert jnp.array_equal(out[1], jnp.zeros_like(out[1]))
    assert jnp.array_equal(state_out[0], inp["state"][0])
