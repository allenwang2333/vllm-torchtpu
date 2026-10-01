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
"""Per-row dequantization scales and FP8 rows in the v3 SparseCore combine."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
    ragged_gather_reduce,
)

jax.config.parse_flags_with_absl()

_FP8 = jnp.float8_e4m3fn
_FP8_MAX = float(jnp.finfo(_FP8).max)


def _reference(x, indices, topk_weights, valid_rows_mask, reduce_group_size, scales):
    """Float64 oracle: dequantize, gather, weight, mask, reduce."""
    x = np.asarray(x.astype(jnp.float32), np.float64)
    if scales is not None:
        x = x * np.asarray(scales, np.float64)[:, None]
    indices = np.asarray(indices)
    out = x[indices] * np.asarray(topk_weights.astype(jnp.float32), np.float64)[:, None]
    out = np.where(np.asarray(valid_rows_mask)[:, None], out, 0.0)
    return out.reshape(-1, reduce_group_size, x.shape[-1]).sum(axis=1)


def _quantize_rows(x_f32):
    scales = jnp.max(jnp.abs(x_f32), axis=-1) / _FP8_MAX
    scales = jnp.maximum(scales, 1e-12).astype(jnp.float32)
    return (x_f32 / scales[:, None]).astype(_FP8), scales


def _uses_sparse_core(*args, **kwargs):
    text = ragged_gather_reduce.lower(*args, **kwargs).as_text()
    return "sc_ragged_gather_reduce_v3" in text


def _ep_routes(key, num_tokens, topk, valid_fraction):
    """EP-shaped routes: a permutation of all T*topk slots, of which a
    contiguous source range (this rank's experts) is valid."""
    input_size = num_tokens * topk
    k_perm, k_w = jax.random.split(key)
    indices = jax.random.permutation(k_perm, input_size).astype(jnp.int32)
    start = input_size // 3
    valid = (indices >= start) & (indices < start + int(input_size * valid_fraction))
    weights = jax.random.uniform(k_w, (input_size,), jnp.float32, 0.05, 1.0)
    return indices, weights.astype(jnp.bfloat16), valid


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RowScalesTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        info = pltpu.get_tpu_info()
        if info.sparse_core is None:
            self.skipTest("no SparseCore on this TPU")

    def test_fp8_decode_is_exact_for_every_bit_pattern(self):
        """Every e4m3fn value, in every byte lane of the packed word, comes
        back exactly (BF16 holds every e4m3 value)."""
        num_tokens, hidden = 4096, 6144
        source_rows = num_tokens * 2
        patterns = np.arange(256, dtype=np.uint8)
        # Row r holds the 256 patterns rotated by r, so each pattern lands in
        # each of the four rows sharing a 32-bit word.
        cols = (np.arange(hidden)[None, :] + np.arange(source_rows)[:, None]) % 256
        x_bits = patterns[cols]
        x_bits = np.where((x_bits & 0x7F) == 0x7F, 0, x_bits)  # drop NaNs
        x = jax.lax.bitcast_convert_type(jnp.asarray(x_bits), _FP8)
        # One valid route per token (slot 0), weight 1, scale 1.
        indices = jnp.arange(source_rows, dtype=jnp.int32)
        indices = jnp.flip(indices.reshape(num_tokens, 2), axis=0).reshape(-1)
        weights = jnp.ones((source_rows,), jnp.bfloat16)
        valid = jnp.tile(jnp.array([True, False]), num_tokens)
        scales = jnp.ones((source_rows,), jnp.float32)
        args = (x, indices, weights, valid, 2)
        self.assertTrue(_uses_sparse_core(*args, row_scales=scales))

        actual = np.asarray(ragged_gather_reduce(*args, row_scales=scales), np.float32)
        expected = np.asarray(
            x.astype(jnp.float32)[np.asarray(indices)[::2]], np.float32
        )
        subnormal = (x_bits[np.asarray(indices)[::2]] & 0x78) == 0
        np.testing.assert_array_equal(actual[~subnormal], expected[~subnormal])
        # Subnormals (< 2**-6) are either exact or flushed to zero.
        sub_ok = (actual[subnormal] == expected[subnormal]) | (actual[subnormal] == 0)
        self.assertTrue(np.all(sub_ok))

    @parameterized.product(
        # GLM-5.2 served (top-8, EP16) and the scaled-mode twin (top-4, EP8):
        # both give each rank half a local route per token.
        shape=[(4096, 8, 1 / 16), (4096, 4, 1 / 8), (1000, 8, 1 / 2)],
        dtype=["bfloat16", "float8_e4m3fn"],
    )
    def test_row_scales_match_oracle(self, shape, dtype):
        num_tokens, topk, valid_fraction = shape
        hidden = 6144
        key = jax.random.key(0)
        k_x, k_r, k_s = jax.random.split(key, 3)
        indices, weights, valid = _ep_routes(k_r, num_tokens, topk, valid_fraction)
        x_f32 = jax.random.normal(k_x, (num_tokens * topk, hidden), jnp.float32)
        if dtype == "float8_e4m3fn":
            x, scales = _quantize_rows(x_f32)
        else:
            x = x_f32.astype(jnp.bfloat16)
            scales = jax.random.uniform(k_s, (x.shape[0],), jnp.float32, 0.25, 4.0)
        args = (x, indices, weights, valid, topk)
        if dtype == "float8_e4m3fn" or x.size * 2 * 2 >= (
            pltpu.get_tpu_info().vmem_capacity_bytes * 0.6
        ):
            self.assertTrue(_uses_sparse_core(*args, row_scales=scales))

        actual = np.asarray(ragged_gather_reduce(*args, row_scales=scales), np.float64)
        self.assertEqual(actual.shape, (num_tokens, hidden))
        expected = _reference(*args, scales)
        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=1e-2)

        # Positive control: the oracle tolerance rejects a kernel that ignores
        # the scales or reads another row's scale.
        for wrong in (None, jnp.roll(scales, 1)):
            off = np.abs(_reference(*args, wrong) - expected)
            self.assertGreater(np.max(off - (2e-2 + 1e-2 * np.abs(expected))), 0.0)

    def test_without_row_scales_is_unchanged(self):
        """BF16 without scales takes the same kernel and gives the same
        result as before scales were added."""
        num_tokens, topk, hidden = 4096, 8, 6144
        indices, weights, valid = _ep_routes(
            jax.random.key(1), num_tokens, topk, 1 / 16
        )
        x = jax.random.normal(
            jax.random.key(2), (num_tokens * topk, hidden), jnp.float32
        ).astype(jnp.bfloat16)
        args = (x, indices, weights, valid, topk)
        self.assertTrue(_uses_sparse_core(*args))
        actual = np.asarray(ragged_gather_reduce(*args), np.float64)
        with_ones = np.asarray(
            ragged_gather_reduce(
                *args, row_scales=jnp.ones((x.shape[0],), jnp.float32)
            ),
            np.float64,
        )
        np.testing.assert_array_equal(actual, with_ones)
        np.testing.assert_allclose(
            actual, _reference(*args, None), atol=2e-2, rtol=1e-2
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
