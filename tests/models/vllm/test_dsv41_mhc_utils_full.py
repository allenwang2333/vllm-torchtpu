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

"""Full statement and branch coverage and parity tests for mhc/utils.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils


def _ref_mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_tokens = mixes.shape[0]
    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
        )
    return pre_mix, post_mix, comb_mix


class TestMhcUtilsFull:
    """100% statement and branch coverage and parity tests for mhc/utils.py."""

    def test_round_up_and_compiler_params(self) -> None:
        assert utils.round_up(0, 16) == 0
        assert utils.round_up(1, 16) == 16
        assert utils.round_up(15, 16) == 16
        assert utils.round_up(16, 16) == 16
        assert utils.round_up(17, 16) == 32

        p1 = utils.compiler_params(utils.DEFAULT_VMEM_LIMIT_BYTES)
        p2 = utils.compiler_params(utils.DEFAULT_VMEM_LIMIT_BYTES)
        assert p1 is p2
        assert p1 == pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        )
        p3 = utils.compiler_params(utils.DEFAULT_VMEM_LIMIT_BYTES // 2)
        assert p3 is not p1
        assert p3.vmem_limit_bytes == utils.DEFAULT_VMEM_LIMIT_BYTES // 2

    def test_hc_mult_from_mix_dim(self) -> None:
        for hc_mult in (1, 2, 4, 8, 16, 64):
            hc_mult3 = hc_mult * (hc_mult + 2)
            assert utils.hc_mult_from_mix_dim(hc_mult3) == hc_mult
            # Second call hits functools.cache
            assert utils.hc_mult_from_mix_dim(hc_mult3) == hc_mult

        for invalid in (-1, 0, 1, 2, 4, 7, 9, 23, 25):
            with pytest.raises(ValueError, match="hc_mult3"):
                utils.hc_mult_from_mix_dim(invalid)

    def test_select_token_block(self) -> None:
        # 1. Without vmem_need
        assert utils.select_token_block(10, 64) == (16, 16)
        assert utils.select_token_block(16, 64) == (16, 16)
        assert utils.select_token_block(100, 64) == (64, 128)

        # 2. With vmem_need that already fits on the first check (while loop 0 iters)
        assert utils.select_token_block(
            128, 64, vmem_need=lambda tb: tb * 10, vmem_limit_bytes=1000
        ) == (64, 128)

        # 3. With vmem_need that halves tb until it fits
        assert utils.select_token_block(
            128,
            64,
            vmem_need=lambda tb: 200 if tb > 32 else 50,
            vmem_limit_bytes=100,
        ) == (32, 128)

        # 4. With vmem_need that hits SUBLANE floor
        assert utils.select_token_block(
            128,
            64,
            vmem_need=lambda _tb: 1000,
            vmem_limit_bytes=100,
        ) == (utils.SUBLANE, 128)

    def test_pad_to_and_trim_to(self) -> None:
        a = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
        b = jnp.ones((4, 16), dtype=jnp.bfloat16)

        # pad == 0 fast path
        unchanged = utils.pad_to(4, a, b)
        assert unchanged[0] is a
        assert unchanged[1] is b

        # pad > 0 path
        padded = utils.pad_to(8, a, b)
        assert padded[0].shape == (8, 8)
        assert padded[1].shape == (8, 16)
        np.testing.assert_array_equal(np.asarray(padded[0][:4]), np.asarray(a))
        np.testing.assert_array_equal(np.asarray(padded[0][4:]), 0.0)

        # trim_to no-op fast path
        untrimmed = utils.trim_to(8, *padded)
        assert untrimmed[0] is padded[0]
        assert untrimmed[1] is padded[1]

        # trim_to slice path
        trimmed = utils.trim_to(4, *padded)
        assert trimmed[0].shape == (4, 8)
        assert trimmed[1].shape == (4, 16)
        np.testing.assert_array_equal(np.asarray(trimmed[0]), np.asarray(a))

    def test_split_fn3_accuracy(self) -> None:
        rng = np.random.default_rng(42)
        fn_np = rng.standard_normal((24, 256), dtype=np.float32) * 0.05
        fn = jnp.asarray(fn_np)

        fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
        assert fn_hi.dtype == jnp.bfloat16
        assert fn_mid.dtype == jnp.bfloat16
        assert fn_lo.dtype == jnp.bfloat16
        assert fn_hi.shape == (24, 256)
        assert fn_mid.shape == (24, 256)
        assert fn_lo.shape == (24, 256)

        reconstructed = (
            fn_hi.astype(jnp.float32)
            + fn_mid.astype(jnp.float32)
            + fn_lo.astype(jnp.float32)
        )
        np.testing.assert_allclose(
            np.asarray(reconstructed), fn_np, rtol=1e-6, atol=1e-7
        )

    @pytest.mark.parametrize("sinkhorn_repeat", [1, 2, 20])
    @pytest.mark.parametrize("hc_mult", [1, 4])
    def test_mhc_pre_gates_bit_exact_parity(
        self, sinkhorn_repeat: int, hc_mult: int
    ) -> None:
        rng = np.random.default_rng(100 + sinkhorn_repeat * 10 + hc_mult)
        num_tokens = 17
        hidden_size = 256
        hc_mult3 = hc_mult * (hc_mult + 2)

        mixes = jnp.asarray(
            rng.standard_normal((num_tokens, hc_mult3), dtype=np.float32)
        )
        sqrsum = jnp.asarray(
            rng.uniform(10.0, 500.0, size=(num_tokens, 1)).astype(np.float32)
        )
        hc_scale = jnp.asarray((1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32))
        hc_base = jnp.asarray((0.5 * rng.standard_normal(hc_mult3)).astype(np.float32))

        got_pre, got_post, got_comb = utils.mhc_pre_gates(
            mixes,
            sqrsum,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        ref_jit = jax.jit(
            _ref_mhc_pre_gates,
            static_argnames=(
                "hc_mult",
                "hidden_size",
                "rms_eps",
                "hc_pre_eps",
                "hc_sinkhorn_eps",
                "hc_post_mult_value",
                "sinkhorn_repeat",
            ),
        )
        ref_pre, ref_post, ref_comb = ref_jit(
            mixes,
            sqrsum,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_repeat,
        )

        assert got_pre.shape == (num_tokens, hc_mult)
        assert got_post.shape == (num_tokens, hc_mult)
        assert got_comb.shape == (num_tokens, hc_mult, hc_mult)

        np.testing.assert_array_equal(np.asarray(got_pre), np.asarray(ref_pre))
        np.testing.assert_array_equal(np.asarray(got_post), np.asarray(ref_post))
        np.testing.assert_array_equal(np.asarray(got_comb), np.asarray(ref_comb))

        eager_pre, eager_post, eager_comb = _ref_mhc_pre_gates(
            mixes,
            sqrsum,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        np.testing.assert_allclose(
            np.asarray(got_pre), np.asarray(eager_pre), rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(got_post), np.asarray(eager_post), rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(got_comb), np.asarray(eager_comb), rtol=1e-6, atol=1e-7
        )
