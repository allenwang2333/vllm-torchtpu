# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract for KDA speculative-decode checkpoint windows.

These tests pin down the semantics the KDA kernels must implement to support
speculative decoding, which is the protocol ``kernels/gdn/v3`` already speaks:

  * a verify step covers ``W = num_speculative_tokens + 1`` window positions;
  * the kernel writes one state checkpoint per position to
    ``state_indices[s] + t``;
  * the next step resumes from ``state_indices[s] + read_offset[s]``, where
    ``read_offset = num_accepted - 1``.

Rejected drafts are therefore rolled back by *selecting* a checkpoint rather
than by copying or recomputing anything, which makes the rollback bit-exact.
That exactness is what these tests assert: selecting checkpoint ``t`` must be
indistinguishable from having decoded exactly ``t + 1`` tokens sequentially.

The reference here is ``kda_step``, the same per-token recurrence the kernels
are validated against elsewhere. When the kernels learn to emit windows, point
these tests at them; the expected values do not change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.kimi_k3 import kda_step

NUM_HEADS = 2
HEAD_DIM = 8
EPS = 1e-6


def _inputs(seed: int, num_tokens: int) -> dict[str, jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    shape = (num_tokens, NUM_HEADS, HEAD_DIM)
    return {
        "query": jax.random.normal(keys[0], shape),
        "key": jax.random.normal(keys[1], shape),
        "value": jax.random.normal(keys[2], shape),
        "raw_gate": jax.random.normal(keys[3], shape),
        "beta": jax.random.normal(keys[4], (num_tokens, NUM_HEADS)),
        "output_gate": jax.random.normal(keys[5], shape),
    }


def _params(seed: int = 99) -> dict[str, jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    a_log = jax.random.normal(keys[0], (NUM_HEADS, ))
    return {
        "decay": jnp.exp(a_log)[:, None],
        "dt_bias": jax.random.normal(keys[1], (NUM_HEADS, HEAD_DIM)),
        "norm_weight": jax.random.normal(keys[2], (HEAD_DIM, )),
    }


def _decode(inputs, params, state, num_tokens, *, lower_bound=None):
    """Decode ``num_tokens`` sequentially, returning every state along the way."""
    states = []
    for t in range(num_tokens):
        _, state = kda_step(
            inputs["query"][t],
            inputs["key"][t],
            inputs["value"][t],
            inputs["raw_gate"][t],
            inputs["beta"][t],
            inputs["output_gate"][t],
            state,
            params["decay"],
            params["dt_bias"],
            params["norm_weight"],
            lower_bound=lower_bound,
            eps=EPS,
        )
        states.append(state)
    return states


def _window(inputs, params, state, window, *, lower_bound=None):
    """The checkpoints a verify step must write, one per window position."""
    return jnp.stack(
        _decode(inputs, params, state, window, lower_bound=lower_bound))


def _initial_state(seed: int = 7) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed),
                             (NUM_HEADS, HEAD_DIM, HEAD_DIM))


@pytest.mark.parametrize("window", [1, 2, 4, 5])
@pytest.mark.parametrize("lower_bound", [None, -1.0])
def test_checkpoint_selection_matches_sequential_decode(
        window: int, lower_bound: float | None) -> None:
    """Checkpoint ``num_accepted - 1`` equals decoding ``num_accepted`` tokens."""
    inputs = _inputs(seed=0, num_tokens=window)
    params = _params()
    state = _initial_state()

    checkpoints = _window(inputs,
                          params,
                          state,
                          window,
                          lower_bound=lower_bound)
    assert checkpoints.shape == (window, NUM_HEADS, HEAD_DIM, HEAD_DIM)

    for num_accepted in range(1, window + 1):
        expected = _decode(inputs,
                           params,
                           state,
                           num_accepted,
                           lower_bound=lower_bound)[-1]
        selected = checkpoints[num_accepted - 1]
        # Selection, not arithmetic: this must be exact, not merely close.
        assert jnp.array_equal(selected, expected), (
            f"window={window} num_accepted={num_accepted}")


def test_window_size_one_is_plain_decode() -> None:
    """Without speculation the window degenerates to today's single state."""
    inputs = _inputs(seed=3, num_tokens=1)
    params = _params()
    state = _initial_state()

    checkpoints = _window(inputs, params, state, 1)
    expected = _decode(inputs, params, state, 1)[-1]
    assert jnp.array_equal(checkpoints[0], expected)


def test_resume_after_partial_acceptance() -> None:
    """Resuming from a rolled-back checkpoint matches never having speculated."""
    window, num_accepted, follow_on = 4, 2, 3
    inputs = _inputs(seed=11, num_tokens=window + follow_on)
    params = _params()
    state = _initial_state()

    rolled_back = _window(inputs, params, state, window)[num_accepted - 1]
    committed = _decode(inputs, params, state, num_accepted)[-1]
    assert jnp.array_equal(rolled_back, committed)

    tail = {name: value[num_accepted:] for name, value in inputs.items()}
    from_rollback = _decode(tail, params, rolled_back, follow_on)[-1]
    from_commit = _decode(tail, params, committed, follow_on)[-1]
    assert jnp.array_equal(from_rollback, from_commit)


def test_full_rejection_keeps_the_incoming_state() -> None:
    """Accepting only the bonus token must not advance past position 0."""
    inputs = _inputs(seed=13, num_tokens=4)
    params = _params()
    state = _initial_state()

    checkpoints = _window(inputs, params, state, 4)
    expected = _decode(inputs, params, state, 1)[-1]
    # num_accepted == 1 (bonus only) -> read_offset 0.
    assert jnp.array_equal(checkpoints[0], expected)
    assert not jnp.array_equal(checkpoints[0], state)
