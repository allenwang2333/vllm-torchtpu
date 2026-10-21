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
"""Independent exact arithmetic, not the copied QSA pipeline reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qwen4_exp_exact_bf16 import corpus, fma_bits, round_bits

from vllm_torchtpu.kernels.qwen4_exp.qsa import _fma
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16


def _check():
    cases, _ = corpus()
    bits = np.asarray(cases, np.uint16)
    args = [
        jax.lax.bitcast_convert_type(jnp.asarray(bits[:, i]), jnp.bfloat16)
        for i in range(3)
    ]
    actual = np.asarray(jax.jit(_fma)(*args)).view(np.uint16)
    expected = np.asarray([fma_bits(*row) for row in cases], np.uint16)
    np.testing.assert_array_equal(actual, expected)


def test_exact_oracle_ties_and_cancellation():
    assert round_bits(257, -8) == 0x3F80
    assert round_bits(259, -8) == 0x3F82
    assert round_bits(-257, -8) == 0xBF80
    assert fma_bits(0x3F80, 0x3F80, 0xBF80) == 0
    assert fma_bits(0, 0, 1) == 1  # IEEE subnormal, distinct from backend FTZ.
    assert fma_bits(0x8000, 0x3F80, 0x8000) == 0x8000


def test_cpu_jit_index_fma_matches_exact_normal_bf16_arithmetic():
    _check()


def test_naive_fp32_add_has_concrete_double_rounding_counterexample():
    cases = [(51784, 48944, 46164), (51898, 15136, 45408)]
    bits = jnp.asarray(cases, jnp.uint16)
    a, b, c = [jax.lax.bitcast_convert_type(bits[:, i], jnp.bfloat16) for i in range(3)]
    naive = jax.jit(
        lambda a, b, c: bf16(
            a.astype(jnp.float32) * b.astype(jnp.float32) + c.astype(jnp.float32)
        )
    )(a, b, c)
    expected = np.asarray([fma_bits(*row) for row in cases], np.uint16)
    assert np.all(np.asarray(naive).view(np.uint16) != expected)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(_fma)(a, b, c)).view(np.uint16), expected
    )


@pytest.mark.tpu_test
def test_tpu_jit_index_fma_matches_exact_normal_bf16_arithmetic():
    assert jax.default_backend() == "tpu"
    _check()


def test_index_rope_matches_explicit_contract_and_detects_reassociation():
    from qwen4_exp_exact_bf16 import rope_bits, rope_cases

    from vllm_torchtpu.kernels.qwen4_exp.qsa import index_rope

    rows = rope_cases()
    bits = np.asarray(rows, np.uint16)
    x = jax.lax.bitcast_convert_type(jnp.asarray(bits[:, :2]), jnp.bfloat16)
    trig = jax.lax.bitcast_convert_type(jnp.asarray(bits[:, 2:]), jnp.bfloat16)
    trig = jnp.pad(trig, ((0, 0), (0, 126)))
    got = jax.jit(lambda x, t: index_rope(x, jnp.arange(len(x)), t, 2))(x, trig)
    expected = np.asarray([rope_bits(r, target="inner_b_cos") for r in rows], np.uint16)
    other = np.asarray([rope_bits(r, target="inner_a_sin") for r in rows], np.uint16)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), expected)
    assert np.any(expected != other), (
        "Target-specific fusion is not universally identical"
    )
