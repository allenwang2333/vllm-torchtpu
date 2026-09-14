# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation tests for the Kimi K3 mega KDA kernel."""

from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.kimi_k3.mega_kda import (_layout_supported,
                                                    kda_forward_inference)


@pytest.mark.parametrize(
    ("query_start_loc", "num_tokens", "supported"),
    [
        ([0, 64], 64, True),
        ([0, 32, 64], 64, True),
        ([0, 20, 40, 64], 64, False),
        ([0, 37, 92, 116], 128, True),
        ([0, 64, 128, 192, 256], 256, True),
        ([0, 20, 20, 40, 40], 64, True),
    ],
)
def test_mega_kda_layout_guard(query_start_loc: list[int], num_tokens: int,
                               supported: bool) -> None:
    actual = _layout_supported(jnp.asarray(query_start_loc, dtype=jnp.int32),
                               num_tokens)
    assert bool(actual) is supported


def test_mega_kda_layout_guard_requires_full_tiles() -> None:
    with pytest.raises(ValueError, match="requires full token tiles"):
        _layout_supported(jnp.asarray([0, 63], dtype=jnp.int32), 63)


def _valid_mega_inputs(tokens: int = 64) -> dict:
    shape = (1, tokens, 1, 128)
    return {
        "q": jnp.zeros(shape, dtype=jnp.bfloat16),
        "k": jnp.zeros(shape, dtype=jnp.bfloat16),
        "v": jnp.zeros(shape, dtype=jnp.bfloat16),
        "g": jnp.zeros(shape, dtype=jnp.bfloat16),
        "beta": jnp.zeros(shape[:3], dtype=jnp.bfloat16),
        "segment_ids": jnp.ones(shape[:2], dtype=jnp.int32),
        "A_log": jnp.zeros((1, ), dtype=jnp.float32),
        "dt_bias": jnp.zeros((128, ), dtype=jnp.float32),
        "initial_state": jnp.zeros((1, 1, 1, 128, 128), dtype=jnp.float32),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "N_max": 1,
    }


def test_mega_kda_requires_bfloat16_inputs() -> None:
    inputs = _valid_mega_inputs()
    inputs["q"] = inputs["q"].astype(jnp.float32)
    with pytest.raises(ValueError, match="requires BF16"):
        kda_forward_inference(**inputs)


def test_mega_kda_requires_64_token_padding() -> None:
    inputs = _valid_mega_inputs(tokens=63)
    with pytest.raises(ValueError, match="multiple of chunk_size"):
        kda_forward_inference(**inputs)


@pytest.mark.parametrize("override", [{
    "safe_gate": False
}, {
    "lower_bound": None
}])
def test_mega_kda_requires_safe_fused_gate(override: dict) -> None:
    inputs = _valid_mega_inputs()
    inputs.update(override)
    with pytest.raises(ValueError, match="safe fused gate"):
        kda_forward_inference(**inputs)


def _run_mega_kda(**kwargs):
    """Execute kda_forward_inference on TPU or in CPU interpret mode."""
    if jax.default_backend() == "cpu":
        orig_pallas_call = pl.pallas_call

        def _interpret_call(*p_args, **p_kwargs):
            p_kwargs["interpret"] = True
            return orig_pallas_call(*p_args, **p_kwargs)

        with mock.patch("jax.experimental.pallas.pallas_call",
                        side_effect=_interpret_call):
            return kda_forward_inference(**kwargs)
    return kda_forward_inference(**kwargs)


def _reference_kda_recurrence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    segment_ids: np.ndarray,
    A_log: np.ndarray,
    dt_bias: np.ndarray,
    initial_state: np.ndarray | None = None,
    scale: float | None = None,
    lower_bound: float | None = -5.0,
    N_max: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Float64 reference for KDA recurrence token-by-token."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q = q.astype(np.float64)
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    g = g.astype(np.float64)
    beta = beta.astype(np.float64)
    dt_bias = dt_bias.astype(np.float64).reshape(H, K)
    A_scale = np.exp(A_log.astype(np.float64))

    # Q / K L2-normalization with epsilon
    q_norm = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k_norm = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    # The kernel casts to bfloat16 after L2-norm
    q_norm = np.asarray(jnp.asarray(q_norm, jnp.bfloat16), dtype=np.float64)
    k_norm = np.asarray(jnp.asarray(k_norm, jnp.bfloat16), dtype=np.float64)

    # Gate activation: lower_bound * sigmoid(A_scale * (g + dt_bias))
    g_input = g + dt_bias[None, None, :, :]
    gate = lower_bound / (1.0 +
                          np.exp(-A_scale[None, None, :, None] * g_input))
    decay = np.exp(gate)

    out = np.zeros((B, T, H, V), dtype=np.float64)
    if initial_state is not None:
        states = initial_state.astype(np.float64).copy()
        if states.ndim == 4:
            states = states[None, ...]
    else:
        states = np.zeros((B, N_max, H, K, V), dtype=np.float64)

    for b in range(B):
        for t in range(T):
            seg = int(segment_ids[b, t])
            if seg == 0:
                continue
            seg_idx = seg - 1
            s = states[b, seg_idx]
            d = decay[b, t]
            s = s * d[:, :, None]
            k_t = k_norm[b, t]
            pred = np.einsum("hk,hkv->hv", k_t, s)
            b_t = np.clip(beta[b, t], 0.0, 1.0)
            v_t = v[b, t]
            delta = b_t[:, None] * (v_t - pred)
            s = s + np.einsum("hk,hv->hkv", k_t, delta)
            states[b, seg_idx] = s
            q_t = q_norm[b, t]
            out[b, t] = scale * np.einsum("hk,hkv->hv", q_t, s)

    return out, states


@pytest.mark.parametrize("gate_bias", [0.0, -4.0])
@pytest.mark.parametrize("seed", [42, 123])
def test_mega_kda_forward_matches_recurrence(seed: int,
                                             gate_bias: float) -> None:
    """Validate mega-KDA single-chunk forward against float64 recurrence.

    Parametrizing gate_bias exercises both fast decay (gate_bias=0.0,
    per-token decay ~ 0.08) and slow decay (gate_bias=-4.0, per-token
    decay ~ 0.97). The slow decay regime preserves state across the
    entire chunk and rigorously exercises the chunked WY cross-token
    accumulation math.
    """
    rng = np.random.default_rng(seed)
    B, T, H, K, V = 1, 64, 1, 128, 128
    shape_q = (B, T, H, K)
    shape_v = (B, T, H, V)

    q_np = rng.standard_normal(shape_q).astype(np.float32)
    k_np = rng.standard_normal(shape_q).astype(np.float32)
    v_np = rng.standard_normal(shape_v).astype(np.float32)
    g_np = (rng.standard_normal(shape_q) + gate_bias).astype(np.float32)
    beta_np = rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32)
    seg_np = np.ones((B, T), dtype=np.int32)
    A_log_np = rng.standard_normal((H, )).astype(np.float32) * 0.1
    dt_bias_np = rng.standard_normal((H * K, )).astype(np.float32) * 0.1

    ref_out, ref_state = _reference_kda_recurrence(
        q=q_np,
        k=k_np,
        v=v_np,
        g=g_np,
        beta=beta_np,
        segment_ids=seg_np,
        A_log=A_log_np,
        dt_bias=dt_bias_np,
        lower_bound=-5.0,
        N_max=1,
    )

    inputs = {
        "q": jnp.asarray(q_np, dtype=jnp.bfloat16),
        "k": jnp.asarray(k_np, dtype=jnp.bfloat16),
        "v": jnp.asarray(v_np, dtype=jnp.bfloat16),
        "g": jnp.asarray(g_np, dtype=jnp.bfloat16),
        "beta": jnp.asarray(beta_np, dtype=jnp.bfloat16),
        "segment_ids": jnp.asarray(seg_np, dtype=jnp.int32),
        "A_log": jnp.asarray(A_log_np, dtype=jnp.float32),
        "dt_bias": jnp.asarray(dt_bias_np, dtype=jnp.float32),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "N_max": 1,
    }

    act_out, act_state = _run_mega_kda(**inputs)

    np.testing.assert_allclose(
        np.asarray(act_out, dtype=np.float64),
        ref_out,
        rtol=1e-3,
        atol=1e-3,
        err_msg="mega_kda forward output diverges from recurrence reference",
    )
    np.testing.assert_allclose(
        np.asarray(act_state, dtype=np.float64),
        ref_state,
        rtol=5e-3,
        atol=5e-3,
        err_msg="mega_kda final state diverges from recurrence reference",
    )
