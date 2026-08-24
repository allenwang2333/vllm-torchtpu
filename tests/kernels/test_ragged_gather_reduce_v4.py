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

"""Unit tests for SparseCore ragged_gather_reduce_v4 kernel with continuous prefetch."""

from __future__ import annotations

import os
import pathlib
import unittest

_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[3]
os.environ.setdefault("VLLM_XLA_CACHE_PATH", str(_WORKSPACE_ROOT / "cache" / "xla"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(_WORKSPACE_ROOT / "cache" / "vllm"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_WORKSPACE_ROOT / "cache" / "torchinductor"))
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
os.environ.setdefault("TMPDIR", str(_WORKSPACE_ROOT / "cache" / "tmp"))

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v4 import (
    ragged_gather_reduce_v4,
)


def reference_ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce in pure JAX."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0.0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


class TestRaggedGatherReduceV4(unittest.TestCase):
    """Test suite covering ragged_gather_reduce_v4 correctness and edge cases."""

    def test_random_routing_configurations(self):
        """Test ragged gather reduce against reference across multiple MoE configs."""
        configs = [
            # (num_tokens, hidden_size, topk, num_sparse_cores)
            (128, 512, 2, 1),
            (128, 512, 2, 2),
            (256, 1024, 8, 1),
            (256, 1024, 8, 2),
            (512, 4096, 10, 1),
            (512, 4096, 10, 2),
            (1024, 2048, 5, 1),
            (1024, 2048, 5, 2),
            (4096, 2048, 4, 1),
            (4096, 2048, 4, 2),
            (8192, 4096, 8, 1),
            (8192, 4096, 8, 2),
            (16384, 4096, 10, 1),
            (16384, 4096, 10, 2),
        ]

        for num_tokens, hidden_size, topk, num_sparse_cores in configs:
            with self.subTest(
                tokens=num_tokens,
                hidden=hidden_size,
                topk=topk,
                sc=num_sparse_cores,
            ):
                key = jax.random.key(1234 + num_tokens + hidden_size)
                total_routes = num_tokens * topk

                k_x, k_idx, k_w, k_val = jax.random.split(key, 4)
                x = jax.random.normal(k_x, (total_routes, hidden_size), dtype=jnp.bfloat16)
                indices = jax.random.permutation(k_idx, total_routes).astype(jnp.int32)
                topk_weights = jax.random.uniform(
                    k_w, (total_routes,), dtype=jnp.bfloat16, minval=0.01, maxval=1.0
                )
                valid_rows_mask = jax.random.bernoulli(k_val, p=0.125, shape=(total_routes,))

                ref_out = reference_ragged_gather_reduce(
                    x, indices, topk_weights, valid_rows_mask, reduce_group_size=topk
                )
                kernel_out = ragged_gather_reduce_v4(
                    x, indices, topk_weights, valid_rows_mask,
                    reduce_group_size=topk,
                    num_sparse_cores=num_sparse_cores,
                )

                ref_f32 = np.array(ref_out.astype(jnp.float32))
                kernel_f32 = np.array(kernel_out.astype(jnp.float32))

                np.testing.assert_allclose(kernel_f32, ref_f32, atol=1e-2, rtol=1e-2)

    def test_all_valid_routes(self):
        """Test 100% valid routing density for single and dual SparseCores."""
        for num_sparse_cores in (1, 2):
            with self.subTest(sc=num_sparse_cores):
                num_tokens = 256
                hidden_size = 512
                topk = 4
                total_routes = num_tokens * topk

                key = jax.random.key(42)
                x = jax.random.normal(key, (total_routes, hidden_size), dtype=jnp.bfloat16)
                indices = jnp.arange(total_routes, dtype=jnp.int32)
                topk_weights = jnp.ones((total_routes,), dtype=jnp.bfloat16) * 0.25
                valid_rows_mask = jnp.ones((total_routes,), dtype=jnp.bool_)

                ref_out = reference_ragged_gather_reduce(
                    x, indices, topk_weights, valid_rows_mask, reduce_group_size=topk
                )
                kernel_out = ragged_gather_reduce_v4(
                    x, indices, topk_weights, valid_rows_mask,
                    reduce_group_size=topk,
                    num_sparse_cores=num_sparse_cores,
                )

                np.testing.assert_allclose(
                    np.array(kernel_out.astype(jnp.float32)),
                    np.array(ref_out.astype(jnp.float32)),
                    atol=1e-2,
                    rtol=1e-2,
                )

    def test_zero_valid_routes(self):
        """Test zero valid routes (should output pure zeros)."""
        for num_sparse_cores in (1, 2):
            with self.subTest(sc=num_sparse_cores):
                num_tokens = 128
                hidden_size = 256
                topk = 2
                total_routes = num_tokens * topk

                key = jax.random.key(99)
                x = jax.random.normal(key, (total_routes, hidden_size), dtype=jnp.bfloat16)
                indices = jnp.arange(total_routes, dtype=jnp.int32)
                topk_weights = jnp.ones((total_routes,), dtype=jnp.bfloat16)
                valid_rows_mask = jnp.zeros((total_routes,), dtype=jnp.bool_)

                kernel_out = ragged_gather_reduce_v4(
                    x, indices, topk_weights, valid_rows_mask,
                    reduce_group_size=topk,
                    num_sparse_cores=num_sparse_cores,
                )

                np.testing.assert_allclose(
                    np.array(kernel_out.astype(jnp.float32)),
                    np.zeros((num_tokens, hidden_size), dtype=np.float32),
                    atol=1e-5,
                )

    def test_default_num_sparse_cores(self):
        """Test that default num_sparse_cores (omitted argument) executes with 2 sparse cores."""
        num_tokens = 512
        hidden_size = 2048
        topk = 8
        total_routes = num_tokens * topk

        key = jax.random.key(777)
        k_x, k_idx, k_w, k_val = jax.random.split(key, 4)
        x = jax.random.normal(k_x, (total_routes, hidden_size), dtype=jnp.bfloat16)
        indices = jax.random.permutation(k_idx, total_routes).astype(jnp.int32)
        topk_weights = jax.random.uniform(
            k_w, (total_routes,), dtype=jnp.bfloat16, minval=0.01, maxval=1.0
        )
        valid_rows_mask = jax.random.bernoulli(k_val, p=0.5, shape=(total_routes,))

        ref_out = reference_ragged_gather_reduce(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size=topk
        )
        # Omitted num_sparse_cores argument to verify default is 2
        default_out = ragged_gather_reduce_v4(
            x, indices, topk_weights, valid_rows_mask,
            reduce_group_size=topk,
        )
        # Explicit num_sparse_cores=2 for exact comparison
        explicit_2sc_out = ragged_gather_reduce_v4(
            x, indices, topk_weights, valid_rows_mask,
            reduce_group_size=topk,
            num_sparse_cores=2,
        )

        np.testing.assert_allclose(
            np.array(default_out.astype(jnp.float32)),
            np.array(ref_out.astype(jnp.float32)),
            atol=1e-2,
            rtol=1e-2,
        )
        np.testing.assert_array_equal(
            np.array(default_out),
            np.array(explicit_2sc_out),
        )


if __name__ == "__main__":
    unittest.main()
