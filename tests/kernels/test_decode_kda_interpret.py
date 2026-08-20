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


def _step(inp, token, state_indices, has_initial, conv_state, state,
          read_offsets, write_offsets):
    return decode_kda(token,
                      inp["g"][:1],
                      inp["beta"][:1],
                      conv_state,
                      inp["conv_weight"],
                      state,
                      inp["a_log"],
                      inp["dt_bias"],
                      state_indices,
                      has_initial,
                      read_offsets,
                      write_offsets,
                      lower_bound=None,
                      compute_dtype=jnp.float32,
                      scale=SCALE)


def test_checkpoint_window_matches_sequential_decode() -> None:
    """A verify window's checkpoints equal sequential decode at each position.

    Walks a 3-token window for one request whose slot group starts at 1,
    reading `base + read_offset` and writing `base + t`, then checks that slot
    `base + t` holds exactly what decoding `t + 1` tokens in place would have
    produced. This is the invariant rollback-by-selection rests on.
    """
    window = 3
    base = 1
    inp = _inputs(seed=17)
    slots = 8
    tokens = [
        jax.random.normal(jax.random.PRNGKey(100 + t),
                          (1, 3 * NUM_HEADS * HEAD_DIM)) * 0.5
        for t in range(window)
    ]

    conv_shape = (slots, KERNEL_SIZE - 1, 3, NUM_HEADS, HEAD_DIM)
    state_shape = (slots, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    conv_win = jax.random.normal(jax.random.PRNGKey(3), conv_shape) * 0.5
    state_win = jax.random.normal(jax.random.PRNGKey(4), state_shape) * 0.1
    conv_seq, state_seq = conv_win, state_win

    idx = jnp.array([base], jnp.int32)
    carried = jnp.array([1], jnp.int32)
    zero = jnp.array([0], jnp.int32)

    for t in range(window):
        # Windowed: resume from the previous position, lay down checkpoint t.
        _, conv_win, state_win = _step(inp, tokens[t], idx, carried, conv_win,
                                       state_win,
                                       jnp.array([max(t - 1, 0)], jnp.int32),
                                       jnp.array([t], jnp.int32))
        # Sequential: no window, everything in the group's base slot.
        _, conv_seq, state_seq = _step(inp, tokens[t], idx, carried, conv_seq,
                                       state_seq, zero, zero)

        _assert_close(f"state@ckpt{t}", state_win[base + t], state_seq[base])
        _assert_close(f"conv@ckpt{t}", conv_win[base + t], conv_seq[base])


def test_checkpoint_window_leaves_other_groups_untouched() -> None:
    """Writing checkpoint t must not disturb a neighbouring request's group."""
    inp = _inputs(seed=23)
    slots = 8
    conv = jax.random.normal(jax.random.PRNGKey(3),
                             (slots, KERNEL_SIZE - 1, 3, NUM_HEADS,
                              HEAD_DIM)) * 0.5
    state = jax.random.normal(
        jax.random.PRNGKey(4),
        (slots, NUM_HEADS, HEAD_DIM, HEAD_DIM)) * 0.1
    base, write_offset = 1, 2

    _, _, state_out = _step(inp, inp["mixed_qkv"][:1],
                            jnp.array([base], jnp.int32),
                            jnp.array([1], jnp.int32), conv, state,
                            jnp.array([0], jnp.int32),
                            jnp.array([write_offset], jnp.int32))

    # Exactly one slot moves: the group's checkpoint `write_offset`.
    for slot in range(state.shape[0]):
        if slot == base + write_offset:
            assert not jnp.array_equal(state_out[slot], state[slot]), (
                f"checkpoint slot {slot} should have been written")
        else:
            assert jnp.array_equal(state_out[slot], state[slot]), (
                f"slot {slot} should be untouched")


def _gated_output_norm_reference(out, output_gate, norm_weight, eps, dtype):
    """The epilogue as `kda_attention_op._gated_output_norm` performs it."""
    normed = out.astype(dtype).astype(jnp.float32)
    normed *= jax.lax.rsqrt(
        jnp.mean(normed * normed, axis=-1, keepdims=True) + eps)
    normed = (normed * norm_weight.astype(jnp.float32)).astype(dtype)
    return normed * jax.nn.sigmoid(output_gate.reshape(normed.shape))


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_fused_epilogue_matches_unfused(eps: float) -> None:
    """Fusing o_norm and the output gate must not change the result.

    The caller used to apply both after the kernel returned. Folding them in
    keeps the output resident instead of writing it out and reading it back
    for two elementwise passes, so the two paths have to agree.
    """
    inp = _inputs(seed=21)
    state_indices = jnp.array([1, 2], jnp.int32)
    has_initial_state = jnp.array([1, 1], jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(77), 2)
    output_gate = jax.random.normal(keys[0],
                                    (NUM_SEQS, NUM_HEADS * HEAD_DIM)) * 0.5
    norm_weight = jax.random.normal(keys[1], (HEAD_DIM, )) * 0.5

    common = (inp["mixed_qkv"], inp["g"], inp["beta"], inp["conv_state"],
              inp["conv_weight"], inp["state"], inp["a_log"], inp["dt_bias"],
              state_indices, has_initial_state)

    raw, conv_unfused, state_unfused = decode_kda(*common,
                                                  lower_bound=None,
                                                  compute_dtype=jnp.float32,
                                                  scale=SCALE)
    want = _gated_output_norm_reference(raw, output_gate, norm_weight, eps,
                                        raw.dtype)

    got, conv_fused, state_fused = decode_kda(*common,
                                              None,
                                              None,
                                              output_gate,
                                              norm_weight,
                                              eps=eps,
                                              lower_bound=None,
                                              compute_dtype=jnp.float32,
                                              scale=SCALE)

    _assert_close("fused epilogue output", got, want)
    # The recurrence itself must be untouched by the epilogue.
    assert jnp.array_equal(conv_fused, conv_unfused)
    assert jnp.array_equal(state_fused, state_unfused)


def test_fused_epilogue_keeps_null_rows_zero() -> None:
    """A null slot emits zeros, and the epilogue must not resurrect them."""
    inp = _inputs(seed=23)
    state_indices = jnp.array([1, 0], jnp.int32)
    has_initial_state = jnp.array([1, 1], jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(79), 2)

    out, _, _ = decode_kda(inp["mixed_qkv"],
                           inp["g"],
                           inp["beta"],
                           inp["conv_state"],
                           inp["conv_weight"],
                           inp["state"],
                           inp["a_log"],
                           inp["dt_bias"],
                           state_indices,
                           has_initial_state,
                           None,
                           None,
                           jax.random.normal(keys[0],
                                             (NUM_SEQS,
                                              NUM_HEADS * HEAD_DIM)) * 0.5,
                           jax.random.normal(keys[1], (HEAD_DIM, )) * 0.5,
                           eps=1e-6,
                           lower_bound=None,
                           compute_dtype=jnp.float32,
                           scale=SCALE)
    assert jnp.array_equal(out[1], jnp.zeros_like(out[1]))


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
