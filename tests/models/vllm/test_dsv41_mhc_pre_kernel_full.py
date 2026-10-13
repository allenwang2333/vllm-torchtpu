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

"""Full statement and branch coverage and parity tests for mhc/pre_kernel.py."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel, utils


class _MockRef:
    def __init__(self, arr: jax.Array) -> None:
        self.arr = arr
        self.dtype = arr.dtype
        self.shape = arr.shape

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, val) -> None:
        self.arr = self.arr.at[idx].set(val)


def _ref_mixes_collapse_kernel(
    x_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    mixes_ref,
    sqrsum_ref,
    layer_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
) -> None:
    m, h = hc_mult, hidden_size
    x = x_ref[...]
    acc = pre_kernel._dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref)
    mixes_ref[...] = acc
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    sqrsum_ref[...] = sqr
    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    lay = pre_mix[:, 0:1] * xf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)


class TestMhcPreKernelFull:
    """100% statement and branch coverage and parity tests for mhc/pre_kernel.py."""

    @pytest.mark.parametrize("hc_mult", [1, 4])
    def test_kernels_direct_and_bit_exact_parity(self, hc_mult: int) -> None:
        rng = np.random.default_rng(50 + hc_mult)
        tb = 16
        hidden_size = 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x = jnp.asarray(rng.standard_normal((tb, hc_hidden), dtype=np.float32)).astype(
            jnp.bfloat16
        )
        fn = jnp.asarray(
            rng.standard_normal((hc_mult3, hc_hidden), dtype=np.float32) * 0.02
        )
        fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
        sc = jnp.asarray((1.0 + 0.1 * rng.standard_normal((1, 3))).astype(np.float32))
        hb = jnp.asarray((0.5 * rng.standard_normal((1, hc_mult3))).astype(np.float32))

        # 1. _mixes_kernel
        mixes_got = _MockRef(jnp.zeros((tb, hc_mult3), dtype=jnp.float32))
        sqr_got = _MockRef(jnp.zeros((tb, 1), dtype=jnp.float32))
        pre_kernel._mixes_kernel(
            _MockRef(x),
            _MockRef(fn_hi),
            _MockRef(fn_mid),
            _MockRef(fn_lo),
            mixes_got,
            sqr_got,
        )
        assert mixes_got.arr.shape == (tb, hc_mult3)
        assert sqr_got.arr.shape == (tb, 1)

        # 2. _mixes_collapse_kernel vs reference
        mixes_c_got = _MockRef(jnp.zeros((tb, hc_mult3), dtype=jnp.float32))
        sqr_c_got = _MockRef(jnp.zeros((tb, 1), dtype=jnp.float32))
        lay_c_got = _MockRef(jnp.zeros((tb, hidden_size), dtype=jnp.bfloat16))

        mixes_c_ref = _MockRef(jnp.zeros((tb, hc_mult3), dtype=jnp.float32))
        sqr_c_ref = _MockRef(jnp.zeros((tb, 1), dtype=jnp.float32))
        lay_c_ref = _MockRef(jnp.zeros((tb, hidden_size), dtype=jnp.bfloat16))

        pre_kernel._mixes_collapse_kernel(
            _MockRef(x),
            _MockRef(fn_hi),
            _MockRef(fn_mid),
            _MockRef(fn_lo),
            _MockRef(sc),
            _MockRef(hb),
            mixes_c_got,
            sqr_c_got,
            lay_c_got,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
        )
        _ref_mixes_collapse_kernel(
            _MockRef(x),
            _MockRef(fn_hi),
            _MockRef(fn_mid),
            _MockRef(fn_lo),
            _MockRef(sc),
            _MockRef(hb),
            mixes_c_ref,
            sqr_c_ref,
            lay_c_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
        )

        np.testing.assert_array_equal(
            np.asarray(mixes_c_got.arr), np.asarray(mixes_c_ref.arr)
        )
        np.testing.assert_array_equal(
            np.asarray(sqr_c_got.arr), np.asarray(sqr_c_ref.arr)
        )
        np.testing.assert_array_equal(
            np.asarray(lay_c_got.arr.astype(jnp.float32)),
            np.asarray(lay_c_ref.arr.astype(jnp.float32)),
        )
        assert pre_kernel._token_index_map(2) == (2, 0)
        assert pre_kernel._zero_index_map(2) == (0, 0)

    def test_mhc_pre_mixes_and_collapse_and_wrapper_branches(self) -> None:
        rng = np.random.default_rng(2026)
        T = 32
        hc_mult = 4
        hidden_size = 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x2d = jnp.asarray(rng.standard_normal((T, hc_hidden), dtype=np.float32)).astype(
            jnp.bfloat16
        )
        fn = jnp.asarray(
            rng.standard_normal((hc_mult3, hc_hidden), dtype=np.float32) * 0.02
        )
        hc_scale = jnp.asarray((1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32))
        hc_base = jnp.asarray((0.5 * rng.standard_normal(hc_mult3)).astype(np.float32))

        # Assertions on mhc_pre_mixes
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes(x2d.astype(jnp.float32), fn)
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes(x2d, fn.astype(jnp.bfloat16))
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes(
                x2d, jnp.zeros((hc_mult3, hc_hidden + 1), dtype=jnp.float32)
            )

        # Assertions on mhc_pre_mixes_collapse
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes_collapse(
                x2d.astype(jnp.float32), fn, hc_scale, hc_base, 1e-6, 1e-6
            )
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes_collapse(
                x2d, fn.astype(jnp.bfloat16), hc_scale, hc_base, 1e-6, 1e-6
            )
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes_collapse(
                x2d,
                jnp.zeros((hc_mult3, hc_hidden + 1), dtype=jnp.float32),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )

        # Assertions on mhc_pre
        res4d = x2d.reshape(2, 16, hc_mult, hidden_size)
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre(
                res4d.astype(jnp.float32),
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre(
                res4d,
                fn.astype(jnp.bfloat16),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )

        def fake_pallas_call(
            kernel_fn, *, grid, in_specs, out_specs, out_shape, compiler_params
        ):
            assert compiler_params is not None
            for spec in in_specs:
                spec.index_map(0)
            for spec in out_specs:
                spec.index_map(0)

            def run(*args):
                tb = in_specs[0].block_shape[0]
                out_lists = [[] for _ in out_shape]
                for step in range(grid[0]):
                    sl = slice(step * tb, (step + 1) * tb)
                    in_refs = [
                        _MockRef(a[sl] if idx == 0 else a) for idx, a in enumerate(args)
                    ]
                    out_refs = [
                        _MockRef(jnp.zeros((tb, s.shape[1]), dtype=s.dtype))
                        for s in out_shape
                    ]
                    kernel_fn(*in_refs, *out_refs)
                    for o_idx, o_ref in enumerate(out_refs):
                        out_lists[o_idx].append(o_ref.arr)
                return tuple(jnp.concatenate(lst, axis=0) for lst in out_lists)

            return run

        with patch.object(pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            # 1. mhc_pre_mixes with vmem_need halving (tb=32 > SUBLANE=16) and padding
            m1, s1 = pre_kernel.mhc_pre_mixes(
                x2d[:20], fn, token_block_size=32, vmem_limit_bytes=1024
            )
            assert m1.shape == (20, hc_mult3)
            assert s1.shape == (20, 1)

            # 2. mhc_pre_mixes_collapse with float32 and non-float32 hc_scale/hc_base
            mc1, sc1, lc1 = pre_kernel.mhc_pre_mixes_collapse(
                x2d, fn, hc_scale, hc_base, 1e-6, 1e-6, token_block_size=32
            )
            assert mc1.shape == (T, hc_mult3)
            assert sc1.shape == (T, 1)
            assert lc1.shape == (T, hidden_size)

            mc2, sc2, lc2 = pre_kernel.mhc_pre_mixes_collapse(
                x2d[:20],
                fn,
                hc_scale.astype(jnp.bfloat16),
                hc_base.astype(jnp.bfloat16),
                1e-6,
                1e-6,
                token_block_size=32,
                vmem_limit_bytes=1024,
            )
            assert mc2.shape == (20, hc_mult3)
            assert sc2.shape == (20, 1)
            assert lc2.shape == (20, hidden_size)

            # 3. mhc_pre wrapper end-to-end
            post_mix, comb_mix, layer_out = pre_kernel.mhc_pre(
                res4d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
                token_block_size=32,
            )
            assert post_mix.shape == (2, 16, hc_mult, 1)
            assert comb_mix.shape == (2, 16, hc_mult, hc_mult)
            assert layer_out.shape == (2, 16, hidden_size)
            assert layer_out.dtype == jnp.bfloat16
