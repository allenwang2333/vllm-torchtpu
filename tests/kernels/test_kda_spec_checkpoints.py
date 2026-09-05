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
"""KDA emits one recurrent-state checkpoint per speculative window position.

Speculative decode rolls a rejected draft back by *selecting* the checkpoint
of the last accepted token -- never by copying or recomputing. That only works
if the kernel writes a state per window position, which is what
``recurrent_kda`` now does and ``chunked_kda`` deliberately does not (it runs
only in PER_SEQ mode, where the window is 1).

These run on CPU: ``compute_kda.recurrent_kda`` is plain JAX -- the kernel
*body*, not a Pallas call -- so the recurrence can be checked against an
independent reference without a TPU. What still needs a device is the Pallas
plumbing around it (VMEM windows, DMA writeback), covered by the
``test_gdn_attention_v3*`` suites.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vllm_torchtpu.kernels.gdn.v3 import compute_kda  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config  # noqa: E402

_N_HEADS = 2
_D = 128  # kq_head_dim == v_head_dim; the kernel tiles on these


def _cfg(window_size):
    """A GDNConfig shaped like a KDA verify window.

    In BATCHED mode ``tile_size`` is the sequence count and ``chunk_size``
    derives from ``window_size``; ``seq_tile_size`` and ``prev_kernel_size``
    are properties, not fields.
    """
    dtypes = gdn_config.Dtypes(
        act_in=jnp.bfloat16,
        act_out=jnp.bfloat16,
        compute=jnp.float32,
        recurrent_state=jnp.float32,
        conv_state=jnp.float32,
    )
    return gdn_config.GDNConfig(
        mode=gdn_config.GDNMode.BATCHED,
        dtypes=dtypes,
        batch_size=1,
        dim_size=2 * _N_HEADS * _D + _N_HEADS * _D,
        kernel_size=4,
        tile_size=1,
        num_kq_heads=_N_HEADS,
        num_v_heads=_N_HEADS,
        kq_head_dim=_D,
        v_head_dim=_D,
        attention_mode=gdn_config.AttentionMode.KDA,
        gate_lower_bound=-5.0,
        window_size=window_size,
    )


def _inputs(rng, chunk):
    """Compact-layout activations: [seq, heads, chunk, 1, dim]."""
    shape = (1, _N_HEADS, chunk, 1, _D)
    return dict(
        q_compact=jnp.asarray(rng.standard_normal(shape), jnp.float32),
        k_compact=jnp.asarray(rng.standard_normal(shape), jnp.float32),
        v_compact=jnp.asarray(rng.standard_normal(shape), jnp.float32),
        # b_compact is (seq, 1, chunk, 1, lanes) with heads on the lane axis.
        b_compact=jnp.asarray(
            rng.standard_normal((1, 1, chunk, 1, max(_N_HEADS, 128))),
            jnp.float32),
        a_compact=jnp.asarray(rng.standard_normal(shape), jnp.float32),
    )


def _run(cfg, real_size, rng):
    inp = _inputs(rng, cfg.chunk_size)
    state_prev = jnp.zeros((1, _N_HEADS, _D, _D), jnp.float32)
    a_log = jnp.asarray(rng.standard_normal(_N_HEADS), jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal(_N_HEADS * _D), jnp.float32)
    out, state = compute_kda.recurrent_kda(
        jnp.asarray([real_size], jnp.int32),
        inp["q_compact"],
        inp["k_compact"],
        inp["v_compact"],
        inp["b_compact"],
        inp["a_compact"],
        state_prev,
        a_log,
        dt_bias,
        cfg,
    )
    return np.asarray(out), np.asarray(state)


def test_state_has_one_checkpoint_per_window_position():
    rng = np.random.default_rng(0)
    for window in (1, 2, 4):
        cfg = _cfg(window_size=window)
        _, state = _run(cfg, real_size=window, rng=rng)
        assert state.shape == (1, window, _N_HEADS, _D,
                               _D), (f"window={window} gave {state.shape}")


def test_checkpoints_are_the_running_state_not_repeats():
    """Each checkpoint must be that position's state, not the final one.

    A plausible wrong implementation broadcasts the end-of-tile state across
    the window axis. Shapes would match and only rollback would be wrong, so
    assert the checkpoints actually differ.
    """
    rng = np.random.default_rng(1)
    cfg = _cfg(window_size=4)
    _, state = _run(cfg, real_size=4, rng=rng)
    for t in range(3):
        assert not np.allclose(state[0, t], state[0, t + 1], atol=1e-6), (
            f"checkpoint {t} equals {t+1}; the window axis looks broadcast")


def test_prefix_of_a_longer_window_matches_a_shorter_run():
    """Checkpoint t must equal the final state of a t+1-token run.

    This is the property rollback depends on: resuming from checkpoint
    `num_accepted - 1` has to be indistinguishable from having only ever run
    that many tokens.
    """
    cfg4 = _cfg(window_size=4)
    _, state4 = _run(cfg4, real_size=4, rng=np.random.default_rng(2))

    for n in (1, 2, 3):
        # Same inputs (same seed), but only `n` positions are real.
        cfg = _cfg(window_size=4)
        _, state_n = _run(cfg, real_size=n, rng=np.random.default_rng(2))
        np.testing.assert_allclose(
            state_n[0, n - 1],
            state4[0, n - 1],
            rtol=2e-5,
            atol=2e-5,
            err_msg=f"checkpoint {n-1} diverges when the window is truncated")


def test_masked_positions_leave_the_state_untouched():
    """Padding must not advance the recurrence.

    The gate is masked to 0 so its decay is `exp(0) == 1`. Masking the raw
    gate input instead would decay the state on padded positions, which is
    silent: shapes and finiteness are unaffected.
    """
    rng = np.random.default_rng(3)
    cfg = _cfg(window_size=4)
    _, state = _run(cfg, real_size=2, rng=rng)
    # Checkpoints 1..3 all sit past the last real token (index 1).
    for t in (2, 3):
        np.testing.assert_allclose(
            state[0, t],
            state[0, 1],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"checkpoint {t} moved past the last real token")


def test_output_keeps_the_chunk_axis():
    rng = np.random.default_rng(4)
    cfg = _cfg(window_size=4)
    out, _ = _run(cfg, real_size=4, rng=rng)
    assert out.shape == (1, 4, _N_HEADS, _D), out.shape
    assert np.isfinite(out).all()


def test_window_one_is_unchanged_behaviour():
    """The non-speculative path must be untouched by this change."""
    rng = np.random.default_rng(5)
    cfg = _cfg(window_size=1)
    out, state = _run(cfg, real_size=1, rng=rng)
    assert out.shape == (1, 1, _N_HEADS, _D)
    assert state.shape == (1, 1, _N_HEADS, _D, _D)
    assert np.isfinite(out).all() and np.isfinite(state).all()
