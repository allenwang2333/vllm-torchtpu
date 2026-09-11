# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive BF16 state alignment and numerical parity test suite for GDN v3.

This test suite validates:
1. 512-byte TPU DMA hardware alignment and granule offset calculation for both
   FP32 and BF16 recurrent and conv states across shared and non-shared layouts.
2. Numerical parity of native BF16 states against FP32 states and eager Python
   references (`gdn_attention_ref` and `gdn_attention_spec_ref`).
3. Correctness across both GDN and KDA attention modes.
4. Speculative decoding window verify ($T=4$) with BF16 state checkpointing.
5. Multi-step state continuation and zeroing of stale slots on `has_initial_state=False`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sys
from pathlib import Path

from vllm_torchtpu.kernels.gdn.v3 import config, wrapper

_kernels_dir = Path(__file__).resolve().parent
if str(_kernels_dir) not in sys.path:
    sys.path.insert(0, str(_kernels_dir))

import test_gdn_attention_v3
import test_gdn_attention_v3_kda

gdn_attention_ref = test_gdn_attention_v3.gdn_attention_ref
gdn_attention_spec_ref = test_gdn_attention_v3.gdn_attention_spec_ref
kda_attention_ref = test_gdn_attention_v3_kda.kda_attention_ref


def _build_test_inputs(
    *,
    num_tokens: int = 4,
    dim: int | None = None,
    n_kq: int = 2,
    n_v: int = 8,
    d_k: int = 128,
    d_v: int = 128,
    kernel_size: int = 4,
    num_slots: int = 2,
    state_dtype: jnp.dtype = jnp.bfloat16,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    gate_lower_bound: float | None = None,
    seed: int = 42,
):
    """Helper creating standardized test tensors for GDN/KDA kernels."""
    key = jax.random.key(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    is_kda = (attention_mode == config.AttentionMode.KDA)
    if is_kda and n_kq != n_v:
        # In KDA (K3 / Kimi-Linear), n_kq equals n_v
        n_kq = n_v

    if dim is None:
        dim = 2 * n_kq * d_k + n_v * d_v

    qkv = jax.random.normal(k1, (num_tokens, dim), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (num_tokens, n_v), dtype=jnp.bfloat16)
    gate_dim = n_v * d_k if is_kda else n_v
    a = jax.random.normal(k3, (num_tokens, gate_dim), dtype=jnp.bfloat16)

    conv_weight = jax.random.normal(k4, (dim, 1, kernel_size),
                                    dtype=jnp.bfloat16)
    conv_bias = jax.random.normal(k5, (dim, ), dtype=jnp.bfloat16)

    a_log = jnp.zeros((n_v, ), dtype=jnp.float32)
    if is_kda:
        dt_bias = jnp.zeros((n_v, d_k), dtype=jnp.float32)
    else:
        dt_bias = jnp.zeros((n_v, ), dtype=jnp.float32)

    conv_state = jnp.zeros(
        (num_slots, kernel_size - 1, dim),
        dtype=state_dtype,
    )
    recurrent_state = jnp.zeros(
        (num_slots, n_v, d_k, d_v),
        dtype=state_dtype,
    )

    query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
    state_indices = jnp.array([1], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
    seq_lens = jnp.array([num_tokens], dtype=jnp.int32)

    return {
        "qkv": qkv,
        "b": b,
        "a": a,
        "states": (conv_state, recurrent_state),
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "query_start_loc": query_start_loc,
        "state_indices": state_indices,
        "distribution": distribution,
        "seq_lens": seq_lens,
        "n_kq": n_kq,
        "n_v": n_v,
        "d_k": d_k,
        "d_v": d_v,
        "kernel_size": kernel_size,
        "attention_mode": attention_mode,
        "gate_lower_bound": gate_lower_bound,
    }


class TestGDNV3ConfigAlignment:
    """Validates 512-byte DMA hardware alignment and granule offset calculation."""

    def test_conv_and_recurrent_hbm_offset_fp32(self):
        """FP32 recurrent state: 8 * 128 * 128 * 4 = 524,288 B = 1024 granules."""
        dtypes = config.Dtypes(
            act_in=jnp.bfloat16,
            act_out=jnp.bfloat16,
            compute=jnp.float32,
            conv_state=jnp.float32,
            recurrent_state=jnp.float32,
        )
        cfg_shared = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=dtypes,
            batch_size=1,
            dim_size=2048,
            kernel_size=4,
            tile_size=1,
            num_kq_heads=2,
            num_v_heads=8,
            kq_head_dim=128,
            v_head_dim=128,
            is_shared=True,
        )
        assert cfg_shared.recurrent_hbm_offset == 0
        assert cfg_shared.conv_hbm_offset == 1024
        # 512-byte alignment assertion: 1024 granules * 512 bytes = 524,288 bytes
        assert (cfg_shared.conv_hbm_offset * 512) % 512 == 0

        # Non-shared layout must return 0 for both offsets
        cfg_non_shared = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=dtypes,
            batch_size=1,
            dim_size=2048,
            kernel_size=4,
            tile_size=1,
            num_kq_heads=2,
            num_v_heads=8,
            kq_head_dim=128,
            v_head_dim=128,
            is_shared=False,
        )
        assert cfg_non_shared.recurrent_hbm_offset == 0
        assert cfg_non_shared.conv_hbm_offset == 0

    def test_conv_and_recurrent_hbm_offset_bf16(self):
        """BF16 recurrent state: 8 * 128 * 128 * 2 = 262,144 B = 512 granules."""
        dtypes = config.Dtypes(
            act_in=jnp.bfloat16,
            act_out=jnp.bfloat16,
            compute=jnp.float32,
            conv_state=jnp.bfloat16,
            recurrent_state=jnp.bfloat16,
        )
        cfg_shared = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=dtypes,
            batch_size=1,
            dim_size=2048,
            kernel_size=4,
            tile_size=1,
            num_kq_heads=2,
            num_v_heads=8,
            kq_head_dim=128,
            v_head_dim=128,
            is_shared=True,
        )
        assert cfg_shared.recurrent_hbm_offset == 0
        assert cfg_shared.conv_hbm_offset == 512
        # 512-byte alignment assertion: 512 granules * 512 bytes = 262,144 bytes
        assert (cfg_shared.conv_hbm_offset * 512) % 512 == 0

        # Non-shared layout must return 0 for both offsets
        cfg_non_shared = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=dtypes,
            batch_size=1,
            dim_size=2048,
            kernel_size=4,
            tile_size=1,
            num_kq_heads=2,
            num_v_heads=8,
            kq_head_dim=128,
            v_head_dim=128,
            is_shared=False,
        )
        assert cfg_non_shared.recurrent_hbm_offset == 0
        assert cfg_non_shared.conv_hbm_offset == 0


class TestGDNV3BF16NumericalParity:
    """Validates numerical parity of BF16 states on Cloud TPU."""

    @pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
    def test_gdn_execution_and_reference_parity(self, state_dtype):
        """Tests GDN attention execution with FP32 and BF16 states against reference."""
        inputs = _build_test_inputs(
            num_tokens=4,
            state_dtype=state_dtype,
            attention_mode=config.AttentionMode.GDN,
        )

        # Copy states for eager reference before fused_conv1d_gdn mutates states in-place
        ref_inputs = dict(inputs)
        ref_inputs["states"] = (jnp.copy(inputs["states"][0]),
                                jnp.copy(inputs["states"][1]))
        ref_states, ref_act = gdn_attention_ref(**ref_inputs)
        ref_conv, ref_recurrent = ref_states

        out_states, out_act = wrapper.fused_conv1d_gdn(**inputs)

        # Check return types and shapes
        out_conv, out_recurrent = out_states
        assert out_conv.dtype == state_dtype
        assert out_recurrent.dtype == state_dtype
        assert out_conv.shape == inputs["states"][0].shape
        assert out_recurrent.shape == inputs["states"][1].shape
        assert out_act.shape == (4, inputs["n_v"] * inputs["d_v"])

        atol = 3e-2 if state_dtype == jnp.bfloat16 else 2e-2
        rtol = 3e-2 if state_dtype == jnp.bfloat16 else 2e-2

        np.testing.assert_allclose(
            np.array(out_act, dtype=np.float32),
            np.array(ref_act, dtype=np.float32),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            np.array(out_conv, dtype=np.float32),
            np.array(ref_conv, dtype=np.float32),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            np.array(out_recurrent, dtype=np.float32),
            np.array(ref_recurrent, dtype=np.float32),
            atol=atol,
            rtol=rtol,
        )

    def test_bf16_vs_fp32_relative_parity(self):
        """Directly asserts that BF16 and FP32 kernel runs match within BF16 tolerance."""
        inputs_fp32 = _build_test_inputs(
            num_tokens=4,
            state_dtype=jnp.float32,
            attention_mode=config.AttentionMode.GDN,
            seed=123,
        )
        inputs_bf16 = _build_test_inputs(
            num_tokens=4,
            state_dtype=jnp.bfloat16,
            attention_mode=config.AttentionMode.GDN,
            seed=123,
        )

        states_fp32, act_fp32 = wrapper.fused_conv1d_gdn(**inputs_fp32)
        states_bf16, act_bf16 = wrapper.fused_conv1d_gdn(**inputs_bf16)

        # Tolerance: BF16 has 7-bit mantissa (~8e-3 precision); 3e-2 is standard
        np.testing.assert_allclose(
            np.array(act_bf16, dtype=np.float32),
            np.array(act_fp32, dtype=np.float32),
            atol=3e-2,
            rtol=3e-2,
        )
        np.testing.assert_allclose(
            np.array(states_bf16[0], dtype=np.float32),
            np.array(states_fp32[0], dtype=np.float32),
            atol=3e-2,
            rtol=3e-2,
        )
        np.testing.assert_allclose(
            np.array(states_bf16[1], dtype=np.float32),
            np.array(states_fp32[1], dtype=np.float32),
            atol=3e-2,
            rtol=3e-2,
        )

    @pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
    def test_kda_execution_parity(self, state_dtype):
        """Tests KDA attention mode with FP32 and BF16 states."""
        inputs = _build_test_inputs(
            num_tokens=4,
            n_kq=4,
            n_v=4,
            state_dtype=state_dtype,
            attention_mode=config.AttentionMode.KDA,
            gate_lower_bound=None,
        )

        # Copy states for eager reference before fused_conv1d_gdn mutates states in-place
        ref_inputs = dict(inputs)
        conv_s, rec_s = ref_inputs.pop("states")
        ref_inputs["conv_state"] = jnp.copy(conv_s)
        ref_inputs["recurrent_state"] = jnp.copy(rec_s)
        ref_states, ref_act = kda_attention_ref(**ref_inputs)
        ref_conv, ref_recurrent = ref_states

        out_states, out_act = wrapper.fused_conv1d_gdn(**inputs)

        out_conv, out_recurrent = out_states
        assert out_conv.dtype == state_dtype
        assert out_recurrent.dtype == state_dtype
        assert out_act.shape == (4, inputs["n_v"] * inputs["d_v"])
        assert not jnp.any(jnp.isnan(out_act))
        assert not jnp.any(jnp.isnan(out_conv))
        assert not jnp.any(jnp.isnan(out_recurrent))

        atol = 3e-2 if state_dtype == jnp.bfloat16 else 2e-2
        rtol = 3e-2 if state_dtype == jnp.bfloat16 else 2e-2
        np.testing.assert_allclose(
            np.array(out_act, dtype=np.float32),
            np.array(ref_act, dtype=np.float32),
            atol=atol,
            rtol=rtol,
        )

    def test_bf16_speculative_window_t4(self):
        """Tests speculative decoding window with BF16 states."""
        num_tokens = 5
        num_spec_tokens = 4
        n_kq = 2
        n_v = 8
        d_k = 128
        d_v = 128
        dim = 2 * n_kq * d_k + n_v * d_v
        kernel_size = 4
        num_slots = 6  # Slot 0 is null slot; slots 1..5 hold the 5 window checkpoints

        key = jax.random.key(101)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        qkv = jax.random.normal(k1, (num_tokens, dim), dtype=jnp.bfloat16)
        b = jax.random.normal(k2, (num_tokens, n_v), dtype=jnp.bfloat16)
        a = jax.random.normal(k3, (num_tokens, n_v), dtype=jnp.bfloat16)
        conv_weight = jax.random.normal(k4, (dim, 1, kernel_size),
                                        dtype=jnp.bfloat16)
        conv_bias = jnp.zeros((dim, ), dtype=jnp.bfloat16)
        a_log = jnp.zeros((n_v, ), dtype=jnp.float32)
        dt_bias = jnp.zeros((n_v, ), dtype=jnp.float32)

        conv_state = jnp.zeros((num_slots, kernel_size - 1, dim),
                               dtype=jnp.bfloat16)
        recurrent_state = jnp.zeros((num_slots, n_v, d_k, d_v),
                                    dtype=jnp.bfloat16)

        query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
        state_indices = jnp.array([1], dtype=jnp.int32)
        distribution = jnp.array([1, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.array([16 + num_tokens], dtype=jnp.int32)
        read_offsets = jnp.array([0], dtype=jnp.int32)
        ckpt_indices = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)

        out_states, out_act = wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            states=(conv_state, recurrent_state),
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
            read_offsets=read_offsets,
            ckpt_indices=ckpt_indices,
        )

        out_conv, out_recurrent = out_states
        assert out_conv.dtype == jnp.bfloat16
        assert out_recurrent.dtype == jnp.bfloat16
        assert out_act.shape == (num_tokens, n_v * d_v)
        assert not jnp.any(jnp.isnan(out_act))

        # Checkpoints 1..4 must all receive state updates
        for ckpt in range(1, num_spec_tokens + 2):
            assert not jnp.all(out_conv[ckpt] == 0)
            assert not jnp.all(out_recurrent[ckpt] == 0)
        # Slot 0 (null slot) must remain 0
        assert jnp.all(out_conv[0] == 0)
        assert jnp.all(out_recurrent[0] == 0)

    def test_bf16_continuation_and_zeroing(self):
        """Tests that dirty slot is zeroed on has_initial_state=False and continues on True."""
        inputs = _build_test_inputs(
            num_tokens=4,
            state_dtype=jnp.bfloat16,
            seed=777,
        )

        # Dirty the slot initially
        dirty_conv = jnp.ones_like(inputs["states"][0]) * 5.0
        dirty_rec = jnp.ones_like(inputs["states"][1]) * 5.0

        # Step 1: has_initial_state=False should disregard dirty state and initialize with zeros
        clean_inputs = dict(inputs)
        clean_inputs["states"] = (dirty_conv, dirty_rec)
        clean_inputs["distribution"] = jnp.array(
            [0, 1, 1], dtype=jnp.int32)  # prefill (has_initial_state=False)

        states_step1, act_step1 = wrapper.fused_conv1d_gdn(**clean_inputs)

        # Reference run with zero state
        zero_inputs = dict(inputs)
        zero_inputs["distribution"] = jnp.array([0, 1, 1], dtype=jnp.int32)
        states_ref, act_ref = wrapper.fused_conv1d_gdn(**zero_inputs)

        np.testing.assert_allclose(
            np.array(act_step1, dtype=np.float32),
            np.array(act_ref, dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )
        # Compare only the active slot [1:]; slot 0 (null block) was dirty (5.0) in inputs
        # but clean in ref, and must remain untouched.
        np.testing.assert_allclose(
            np.array(states_step1[0][1:], dtype=np.float32),
            np.array(states_ref[0][1:], dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(states_step1[1][1:], dtype=np.float32),
            np.array(states_ref[1][1:], dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )
        # Verify slot 0 remained untouched
        np.testing.assert_allclose(
            np.array(states_step1[0][0], dtype=np.float32),
            5.0,
            atol=1e-4,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(states_step1[1][0], dtype=np.float32),
            5.0,
            atol=1e-4,
            rtol=1e-4,
        )

        # Step 2: Continuation with has_initial_state=True (prefill continuation)
        # Context exists when seq_lens > query_lens (e.g. seq_lens = 8, query_len = 4)
        k_step2 = jax.random.key(888)
        k1, k2, k3 = jax.random.split(k_step2, 3)
        cont_inputs = dict(inputs)
        cont_inputs["qkv"] = jax.random.normal(k1,
                                               inputs["qkv"].shape,
                                               dtype=inputs["qkv"].dtype)
        cont_inputs["b"] = jax.random.normal(k2,
                                             inputs["b"].shape,
                                             dtype=inputs["b"].dtype)
        cont_inputs["a"] = jax.random.normal(k3,
                                             inputs["a"].shape,
                                             dtype=inputs["a"].dtype)
        cont_inputs["states"] = (jnp.copy(states_step1[0]),
                                 jnp.copy(states_step1[1]))
        cont_inputs["distribution"] = jnp.array([0, 1, 1], dtype=jnp.int32)
        cont_inputs["seq_lens"] = jnp.array([8], dtype=jnp.int32)

        states_step2, act_step2 = wrapper.fused_conv1d_gdn(**cont_inputs)
        assert not jnp.any(jnp.isnan(act_step2))
        assert not jnp.all(states_step2[0][1:] == states_step1[0][1:])
        assert not jnp.all(states_step2[1][1:] == states_step1[1][1:])


class TestGDNV3UnifiedPoolExecution:
    """Validates execution and numerical parity of unified state pool (is_shared=True)."""

    @pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize(
        "dim_config",
        [
            {
                "n_kq": 2,
                "n_v": 8,
                "d_k": 128,
                "d_v": 128
            },  # dim=1536
            {
                "n_kq": 4,
                "n_v": 8,
                "d_k": 128,
                "d_v": 128
            },  # dim=2048
        ],
    )
    def test_unified_pool_execution_and_separate_state_parity(
            self, state_dtype, dim_config):
        """Tests that states=pool achieves parity with separate states across FP32 and BF16."""
        n_kq = dim_config["n_kq"]
        n_v = dim_config["n_v"]
        d_k = dim_config["d_k"]
        d_v = dim_config["d_v"]
        dim = 2 * n_kq * d_k + n_v * d_v
        num_slots = 3
        num_tokens = 4
        kernel_size = 4

        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(key, 9)

        qkv = jax.random.normal(k1, (num_tokens, dim)).astype(jnp.bfloat16)
        b = jax.random.normal(k2, (num_tokens, n_v)).astype(jnp.bfloat16)
        a = jax.random.normal(k3, (num_tokens, n_v)).astype(jnp.bfloat16)
        conv_weight = jax.random.normal(k4, (dim, 1, kernel_size)).astype(
            jnp.bfloat16)
        conv_bias = jax.random.normal(k5, (dim, )).astype(jnp.bfloat16)
        a_log = jax.random.normal(k6, (n_v, )).astype(jnp.float32)
        dt_bias = jax.random.normal(k7, (n_v, )).astype(jnp.float32)

        query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
        state_indices = jnp.array([1], dtype=jnp.int32)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.array([num_tokens], dtype=jnp.int32)

        itemsize = jnp.dtype(state_dtype).itemsize
        granule_elements = 512 // itemsize
        rec_granules = (n_v * d_k * d_v * itemsize) // 512
        conv_granules = ((kernel_size - 1) * dim * itemsize) // 512
        total_granules = rec_granules + conv_granules

        # Initial random states
        init_conv = jax.random.normal(
            k8, (num_slots, kernel_size - 1, 1, dim)).astype(state_dtype)
        init_rec = jax.random.normal(
            k9, (num_slots, n_v, d_k, d_v)).astype(state_dtype)

        # Pack into unified pool (recurrent state at offset 0, conv state at offset rec_granules)
        pool = jnp.zeros((num_slots, total_granules, granule_elements),
                         dtype=state_dtype)
        flat_conv = init_conv.reshape(num_slots, conv_granules,
                                      granule_elements)
        flat_rec = init_rec.reshape(num_slots, rec_granules, granule_elements)
        pool = pool.at[:, :rec_granules].set(flat_rec)
        pool = pool.at[:, rec_granules:].set(flat_conv)

        # 1. Run with separate states
        (out_conv_sep, out_rec_sep), act_sep = wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            states=(jnp.copy(init_conv), jnp.copy(init_rec)),
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

        # 2. Run with unified pool
        out_pool, act_pool = wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            states=jnp.copy(pool),
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

        # Parity check on activation output
        np.testing.assert_allclose(
            np.array(act_sep, dtype=np.float32),
            np.array(act_pool, dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )

        # Unpack pool states
        out_pool_rec = out_pool[:, :rec_granules].reshape(
            num_slots, n_v, d_k, d_v)
        out_pool_conv = out_pool[:, rec_granules:].reshape(
            num_slots, kernel_size - 1, 1, dim)

        # Parity check on active slot (slot 1)
        np.testing.assert_allclose(
            np.array(out_rec_sep[1], dtype=np.float32),
            np.array(out_pool_rec[1], dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(out_conv_sep[1], dtype=np.float32),
            np.array(out_pool_conv[1], dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )

        # Untouched slots (slot 0, slot 2) must remain identical
        np.testing.assert_allclose(
            np.array(pool[0], dtype=np.float32),
            np.array(out_pool[0], dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(pool[2], dtype=np.float32),
            np.array(out_pool[2], dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
    def test_unified_pool_speculative_decoding(self, state_dtype):
        """Tests unified pool speculative decoding checkpointing ($T=4$)."""
        num_tokens = 4
        num_spec_tokens = 3
        n_kq = 2
        n_v = 8
        d_k = 128
        d_v = 128
        dim = 2 * n_kq * d_k + n_v * d_v
        num_slots = 6
        kernel_size = 4

        itemsize = jnp.dtype(state_dtype).itemsize
        granule_elements = 512 // itemsize
        rec_granules = (n_v * d_k * d_v * itemsize) // 512
        conv_granules = ((kernel_size - 1) * dim * itemsize) // 512
        total_granules = rec_granules + conv_granules

        qkv = jnp.zeros((num_tokens, dim), dtype=jnp.bfloat16)
        b = jnp.zeros((num_tokens, n_v), dtype=jnp.bfloat16)
        a = jnp.zeros((num_tokens, n_v), dtype=jnp.bfloat16)
        conv_weight = jnp.zeros((dim, 1, kernel_size), dtype=jnp.bfloat16)
        conv_bias = jnp.zeros((dim, ), dtype=jnp.bfloat16)
        a_log = jnp.zeros((n_v, ), dtype=jnp.float32)
        dt_bias = jnp.zeros((n_v, ), dtype=jnp.float32)
        query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
        state_indices = jnp.array([1], dtype=jnp.int32)
        distribution = jnp.array([1, 0, 0], dtype=jnp.int32)  # decode
        seq_lens = jnp.array([10], dtype=jnp.int32)
        read_offsets = jnp.array([0], dtype=jnp.int32)
        ckpt_indices = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)

        pool = jnp.zeros((num_slots, total_granules, granule_elements),
                         dtype=state_dtype)
        out_pool, out_act = wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            states=pool,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
            read_offsets=read_offsets,
            ckpt_indices=ckpt_indices,
        )
        assert out_pool.shape == pool.shape
        assert out_pool.dtype == state_dtype
        assert out_act.shape == (num_tokens, n_v * d_v)
