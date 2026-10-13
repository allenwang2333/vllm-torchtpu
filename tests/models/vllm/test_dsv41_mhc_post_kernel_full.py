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

"""Full statement and branch coverage and parity tests for mhc/post_kernel.py."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.mhc import post_kernel


class _MockRef:
    def __init__(self, arr: jax.Array) -> None:
        self.arr = arr
        self.dtype = arr.dtype
        self.shape = arr.shape

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, val) -> None:
        self.arr = self.arr.at[idx].set(val)


def _ref_post_kernel(
    x_ref, res_ref, post_ref, comb_ref, out_ref, *, hc_mult, hidden_size
) -> None:
    x = x_ref[...].astype(jnp.float32)
    post = post_ref[...]
    comb = comb_ref[...]

    streams = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size].astype(jnp.float32)
        for i in range(hc_mult)
    ]
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * streams[i]
        out_ref[:, j * hidden_size : (j + 1) * hidden_size] = acc.astype(out_ref.dtype)


class TestMhcPostKernelFull:
    """100% statement and branch coverage and parity tests for mhc/post_kernel.py."""

    @pytest.mark.parametrize("hc_mult", [1, 4])
    def test_post_kernel_bit_exact_parity(self, hc_mult: int) -> None:
        rng = np.random.default_rng(100 + hc_mult)
        tb = 16
        hidden_size = 32

        x = jnp.asarray(
            rng.standard_normal((tb, hidden_size), dtype=np.float32)
        ).astype(jnp.bfloat16)
        res = jnp.asarray(
            rng.standard_normal((tb, hc_mult * hidden_size), dtype=np.float32)
        ).astype(jnp.bfloat16)
        post = jnp.asarray(rng.uniform(0.1, 2.0, size=(tb, hc_mult)).astype(np.float32))
        comb = jnp.asarray(
            rng.uniform(0.05, 0.5, size=(tb, hc_mult * hc_mult)).astype(np.float32)
        )

        out_got = _MockRef(jnp.zeros_like(res))
        out_ref = _MockRef(jnp.zeros_like(res))

        post_kernel._post_kernel(
            _MockRef(x),
            _MockRef(res),
            _MockRef(post),
            _MockRef(comb),
            out_got,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )
        _ref_post_kernel(
            _MockRef(x),
            _MockRef(res),
            _MockRef(post),
            _MockRef(comb),
            out_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )

        np.testing.assert_array_equal(
            np.asarray(out_got.arr.astype(jnp.float32)),
            np.asarray(out_ref.arr.astype(jnp.float32)),
        )
        assert post_kernel._token_index_map(3) == (3, 0)

    def test_mhc_post_2d_and_mhc_post_branches(self) -> None:
        rng = np.random.default_rng(2026)
        T = 32
        hc_mult = 4
        hidden_size = 32

        x_bf16 = jnp.asarray(
            rng.standard_normal((T, hidden_size), dtype=np.float32)
        ).astype(jnp.bfloat16)
        x_f32_3d = jnp.asarray(
            rng.standard_normal((2, T // 2, hidden_size), dtype=np.float32)
        )
        res2d = jnp.asarray(
            rng.standard_normal((T, hc_mult * hidden_size), dtype=np.float32)
        ).astype(jnp.bfloat16)
        post2d = jnp.asarray(
            rng.uniform(0.1, 2.0, size=(T, hc_mult)).astype(np.float32)
        )
        post3d = post2d.reshape(T, hc_mult, 1)
        comb3d = jnp.asarray(
            rng.uniform(0.05, 0.5, size=(T, hc_mult, hc_mult)).astype(np.float32)
        )

        # Assertions on mhc_post_2d
        with pytest.raises(AssertionError):
            post_kernel.mhc_post_2d(x_bf16, res2d.astype(jnp.float32), post2d, comb3d)
        with pytest.raises(AssertionError):
            post_kernel.mhc_post_2d(x_bf16, res2d, post2d.astype(jnp.bfloat16), comb3d)
        with pytest.raises(AssertionError):
            post_kernel.mhc_post_2d(x_bf16, res2d, post2d, comb3d.astype(jnp.bfloat16))

        # Assertion on mhc_post
        with pytest.raises(AssertionError):
            post_kernel.mhc_post(
                x_bf16,
                res2d.reshape(T, hc_mult, hidden_size).astype(jnp.float32),
                post3d,
                comb3d,
            )

        def fake_pallas_call(
            kernel_fn, *, grid, in_specs, out_specs, out_shape, compiler_params
        ):
            assert compiler_params is not None
            for spec in in_specs:
                assert spec.index_map(0) == (0, 0)
            assert out_specs.index_map(0) == (0, 0)

            def run(x_in, res_in, post_in, comb_in):
                tb = in_specs[0].block_shape[0]
                blocks = []
                for step in range(grid[0]):
                    sl = slice(step * tb, (step + 1) * tb)
                    out_blk = _MockRef(
                        jnp.zeros((tb, out_shape.shape[1]), dtype=out_shape.dtype)
                    )
                    kernel_fn(
                        _MockRef(x_in[sl]),
                        _MockRef(res_in[sl]),
                        _MockRef(post_in[sl]),
                        _MockRef(comb_in[sl]),
                        out_blk,
                    )
                    blocks.append(out_blk.arr)
                return jnp.concatenate(blocks, axis=0)

            return run

        with patch.object(post_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            # 1. 2D x (bfloat16), 2D post_layer_mix, unpadded T=16
            out1 = post_kernel.mhc_post_2d(
                x_bf16, res2d, post2d, comb3d, token_block_size=16
            )
            assert out1.shape == (T, hc_mult * hidden_size)
            assert out1.dtype == jnp.bfloat16

            # 2. 3D x (float32 -> cast to bf16), 3D post_layer_mix, vmem halving
            out2 = post_kernel.mhc_post_2d(
                x_f32_3d,
                res2d,
                post3d,
                comb3d,
                token_block_size=32,
                vmem_limit_bytes=1024,
            )
            assert out2.shape == (T, hc_mult * hidden_size)
            np.testing.assert_array_equal(
                np.asarray(out1.astype(jnp.float32)),
                np.asarray(
                    post_kernel.mhc_post_2d(
                        x_bf16.reshape(2, T // 2, hidden_size).astype(jnp.float32),
                        res2d,
                        post3d,
                        comb3d,
                        token_block_size=32,
                        vmem_limit_bytes=1024,
                    ).astype(jnp.float32)
                ),
            )

            # 3. Padded token count (T=10 -> padded to 16, then trimmed to 10)
            out_pad = post_kernel.mhc_post_2d(
                x_bf16[:10],
                res2d[:10],
                post2d[:10],
                comb3d[:10],
                token_block_size=16,
            )
            assert out_pad.shape == (10, hc_mult * hidden_size)
            np.testing.assert_array_equal(
                np.asarray(out_pad.astype(jnp.float32)),
                np.asarray(out1[:10].astype(jnp.float32)),
            )

            # 4. mhc_post wrapper with 4D residual (B=2, S=16, M=4, H=32)
            out_3d = post_kernel.mhc_post(
                x_bf16.reshape(2, 16, hidden_size),
                res2d.reshape(2, 16, hc_mult, hidden_size),
                post3d.reshape(2, 16, hc_mult, 1),
                comb3d.reshape(2, 16, hc_mult, hc_mult),
                token_block_size=16,
            )
            assert out_3d.shape == (2, 16, hc_mult, hidden_size)
            np.testing.assert_array_equal(
                np.asarray(
                    out_3d.reshape(T, hc_mult * hidden_size).astype(jnp.float32)
                ),
                np.asarray(out1.astype(jnp.float32)),
            )
