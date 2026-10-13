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

"""Full unit & parity test suite for mhc/fused_post_pre_kernel.py."""

import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel, utils


class _FakeRef:
    """Array-backed ref supporting [...] and slice indexing for Pallas tests."""

    def __init__(self, arr: jax.Array):
        self.arr = arr

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.arr
        return self.arr[idx]

    def __setitem__(self, idx, val):
        if idx is Ellipsis:
            self.arr = val
        else:
            self.arr = self.arr.at[idx].set(val)


def _reference_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gemm_precision: str = "highest",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    num_tokens = int(np.prod(outer_shape))
    hc_mult3 = fn.shape[0]

    x_f = x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16).astype(jnp.float32)
    res2d = residual.reshape(num_tokens, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(num_tokens, hc_mult).astype(jnp.float32)
    comb2d = comb_res_mix.reshape(num_tokens, hc_mult * hc_mult).astype(jnp.float32)

    old_bf = [res2d[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)]
    streams_bf = []
    for j in range(hc_mult):
        acc = post2d[:, j : j + 1] * x_f
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb2d[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        streams_bf.append(acc.astype(jnp.bfloat16))

    g_bf = jnp.concatenate(streams_bf, axis=1)
    fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
    dn = (((1,), (1,)), ((), ()))
    acc = jax.lax.dot_general(g_bf, fn_hi, dn, preferred_element_type=jnp.float32)
    if gemm_precision == "highest":
        acc = acc + jax.lax.dot_general(
            g_bf, fn_mid, dn, preferred_element_type=jnp.float32
        )
        acc = acc + jax.lax.dot_general(
            g_bf, fn_lo, dn, preferred_element_type=jnp.float32
        )
    mixes = acc
    gf = g_bf.astype(jnp.float32)
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)

    sc_f = hc_scale.astype(jnp.float32).reshape(1, 3)
    hb_f = hc_base.astype(jnp.float32).reshape(1, hc_mult3)
    scaled = mixes[:, :hc_mult] * jax.lax.rsqrt(sqr / (hc_mult * hidden_size) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_f[0, 0] + hb_f[:, :hc_mult]) + hc_pre_eps
    lay = pre_mix[:, 0:1] * gf[:, :hidden_size]
    for i in range(1, hc_mult):
        lay = (
            lay + pre_mix[:, i : i + 1] * gf[:, i * hidden_size : (i + 1) * hidden_size]
        )
    layer2d = lay.astype(jnp.bfloat16)

    _, post_mix_cur, comb_mix_cur = functools.partial(
        jax.jit,
        static_argnames=(
            "hc_mult",
            "hidden_size",
            "rms_eps",
            "hc_pre_eps",
            "hc_sinkhorn_eps",
            "hc_post_mult_value",
            "sinkhorn_repeat",
        ),
    )(utils.mhc_pre_gates)(
        mixes,
        sqr,
        hc_mult,
        hidden_size,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (
        g_bf.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )


class TestMhcFusedPostPreKernelFull:
    """Achieves 100% statement and branch coverage on mhc/fused_post_pre_kernel.py."""

    @pytest.mark.parametrize("hc_mult", [1, 4])
    @pytest.mark.parametrize("gemm_precision", ["highest", "default"])
    def test_fused_kernel_direct_and_parity(
        self, hc_mult: int, gemm_precision: str
    ) -> None:
        assert fused_post_pre_kernel._token_index_map(5) == (5, 0)
        assert fused_post_pre_kernel._zero_index_map(5) == (0, 0)

        rng = np.random.default_rng(42 + hc_mult)
        tb = 8
        hidden_size = 128
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)
        rms_eps = 1e-6
        hc_pre_eps = 1e-6

        x2d = jnp.asarray(
            rng.standard_normal((tb, hidden_size), dtype=np.float32)
        ).astype(jnp.bfloat16)
        res2d = jnp.asarray(
            rng.standard_normal((tb, hc_hidden), dtype=np.float32)
        ).astype(jnp.bfloat16)
        post2d = jnp.asarray(rng.standard_normal((tb, hc_mult), dtype=np.float32))
        comb2d = jnp.asarray(
            rng.standard_normal((tb, hc_mult * hc_mult), dtype=np.float32)
        )
        fn = jnp.asarray(
            rng.standard_normal((hc_mult3, hc_hidden), dtype=np.float32) * 0.05
        )
        hc_scale = jnp.asarray([1.1, 0.9, 1.05], dtype=jnp.float32)
        hc_base = jnp.asarray(
            rng.standard_normal((hc_mult3,), dtype=np.float32) * 0.2,
            dtype=jnp.float32,
        )

        fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
        newres_ref = _FakeRef(jnp.zeros((tb, hc_hidden), dtype=jnp.bfloat16))
        mixes_ref = _FakeRef(jnp.zeros((tb, hc_mult3), dtype=jnp.float32))
        sqrsum_ref = _FakeRef(jnp.zeros((tb, 1), dtype=jnp.float32))
        layer_ref = _FakeRef(jnp.zeros((tb, hidden_size), dtype=jnp.bfloat16))

        fused_post_pre_kernel._fused_kernel(
            _FakeRef(x2d),
            _FakeRef(res2d),
            _FakeRef(post2d),
            _FakeRef(comb2d),
            _FakeRef(fn_hi),
            _FakeRef(fn_mid),
            _FakeRef(fn_lo),
            _FakeRef(hc_scale.reshape(1, 3)),
            _FakeRef(hc_base.reshape(1, hc_mult3)),
            newres_ref,
            mixes_ref,
            sqrsum_ref,
            layer_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
        )

        ref_res, _, _, ref_layer = _reference_fused_post_pre(
            x2d,
            res2d.reshape(tb, hc_mult, hidden_size),
            post2d.reshape(tb, hc_mult, 1),
            comb2d.reshape(tb, hc_mult, hc_mult),
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            1e-6,
            2.0,
            4,
            gemm_precision=gemm_precision,
        )
        np.testing.assert_array_equal(
            np.asarray(newres_ref.arr.astype(jnp.float32)),
            np.asarray(ref_res.reshape(tb, hc_hidden).astype(jnp.float32)),
        )
        np.testing.assert_array_equal(
            np.asarray(layer_ref.arr.astype(jnp.float32)),
            np.asarray(ref_layer.astype(jnp.float32)),
        )

    def test_fused_post_pre_mixes_and_mhc_fused_post_pre_branches(self) -> None:
        rng = np.random.default_rng(2026)
        hc_mult = 4
        hidden_size = 128
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)
        num_tokens = 12
        rms_eps = 1e-6
        hc_pre_eps = 1e-6
        hc_sinkhorn_eps = 1e-6
        hc_post_mult_value = 2.0
        sinkhorn_repeat = 4

        x_3d_f32 = jnp.asarray(
            rng.standard_normal((3, 4, hidden_size), dtype=np.float32)
        )
        res_4d_bf16 = jnp.asarray(
            rng.standard_normal((3, 4, hc_mult, hidden_size), dtype=np.float32)
        ).astype(jnp.bfloat16)
        post_4d = jnp.asarray(rng.standard_normal((3, 4, hc_mult, 1), dtype=np.float32))
        comb_4d = jnp.asarray(
            rng.standard_normal((3, 4, hc_mult, hc_mult), dtype=np.float32)
        )
        fn = jnp.asarray(
            rng.standard_normal((hc_mult3, hc_hidden), dtype=np.float32) * 0.05
        )
        hc_scale_f32 = jnp.asarray([1.05, 0.95, 1.0], dtype=jnp.float32)
        hc_base_f32 = jnp.asarray(
            rng.standard_normal((hc_mult3,), dtype=np.float32) * 0.1
        )
        hc_scale_bf16 = hc_scale_f32.astype(jnp.bfloat16)
        hc_base_bf16 = hc_base_f32.astype(jnp.bfloat16)

        def _fake_pallas_call(
            kernel_fn, *, grid, in_specs, out_specs, out_shape, compiler_params
        ):
            del compiler_params

            def _runner(
                x_in, res_in, post_in, comb_in, fn_hi, fn_mid, fn_lo, sc_in, hb_in
            ):
                tb = in_specs[0].block_shape[0]
                out_res = jnp.zeros(out_shape[0].shape, dtype=out_shape[0].dtype)
                out_mix = jnp.zeros(out_shape[1].shape, dtype=out_shape[1].dtype)
                out_sqr = jnp.zeros(out_shape[2].shape, dtype=out_shape[2].dtype)
                out_lay = jnp.zeros(out_shape[3].shape, dtype=out_shape[3].dtype)
                for step in range(grid[0]):
                    r0, _ = in_specs[0].index_map(step)
                    _ = in_specs[4].index_map(step)
                    _ = out_specs[0].index_map(step)
                    sl = slice(r0 * tb, (r0 + 1) * tb)
                    nr_ref = _FakeRef(out_res[sl])
                    mx_ref = _FakeRef(out_mix[sl])
                    sq_ref = _FakeRef(out_sqr[sl])
                    ly_ref = _FakeRef(out_lay[sl])
                    kernel_fn(
                        _FakeRef(x_in[sl]),
                        _FakeRef(res_in[sl]),
                        _FakeRef(post_in[sl]),
                        _FakeRef(comb_in[sl]),
                        _FakeRef(fn_hi),
                        _FakeRef(fn_mid),
                        _FakeRef(fn_lo),
                        _FakeRef(sc_in),
                        _FakeRef(hb_in),
                        nr_ref,
                        mx_ref,
                        sq_ref,
                        ly_ref,
                    )
                    out_res = out_res.at[sl].set(nr_ref.arr)
                    out_mix = out_mix.at[sl].set(mx_ref.arr)
                    out_sqr = out_sqr.at[sl].set(sq_ref.arr)
                    out_lay = out_lay.at[sl].set(ly_ref.arr)
                return out_res, out_mix, out_sqr, out_lay

            return _runner

        with mock.patch.object(
            fused_post_pre_kernel.pl, "pallas_call", side_effect=_fake_pallas_call
        ):
            # 1) Call fused_post_pre_mixes with 2D inputs and f32 hc_scale/hc_base
            x_2d_bf16 = x_3d_f32.reshape(num_tokens, hidden_size).astype(jnp.bfloat16)
            res_2d_bf16 = res_4d_bf16.reshape(num_tokens, hc_hidden)
            post_2d = post_4d.reshape(num_tokens, hc_mult)
            comb_2d = comb_4d.reshape(num_tokens, hc_mult * hc_mult)
            nr_a, mx_a, sq_a, ly_a = fused_post_pre_kernel.fused_post_pre_mixes(
                x_2d_bf16,
                res_2d_bf16,
                post_2d,
                comb_2d,
                fn,
                hc_scale_f32,
                hc_base_f32,
                rms_eps,
                hc_pre_eps,
                token_block_size=8,
            )
            assert nr_a.shape == (num_tokens, hc_hidden)
            assert mx_a.shape == (num_tokens, hc_mult3)
            assert sq_a.shape == (num_tokens, 1)
            assert ly_a.shape == (num_tokens, hidden_size)

            # 2) Call mhc_fused_post_pre with 4D residual, f32 x, and bf16 scale/base
            res_4d_out, post_4d_out, comb_4d_out, lay_4d_out = (
                fused_post_pre_kernel.mhc_fused_post_pre(
                    x_3d_f32,
                    res_4d_bf16,
                    post_4d,
                    comb_4d,
                    fn,
                    hc_scale_bf16,
                    hc_base_bf16,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                    token_block_size=8,
                )
            )
            ref_res_4d, ref_post_4d, ref_comb_4d, ref_lay_4d = (
                _reference_fused_post_pre(
                    x_3d_f32,
                    res_4d_bf16,
                    post_4d,
                    comb_4d,
                    fn,
                    hc_scale_bf16,
                    hc_base_bf16,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                )
            )
            np.testing.assert_array_equal(
                np.asarray(res_4d_out.astype(jnp.float32)),
                np.asarray(ref_res_4d.astype(jnp.float32)),
            )
            np.testing.assert_allclose(
                np.asarray(post_4d_out), np.asarray(ref_post_4d), rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                np.asarray(comb_4d_out), np.asarray(ref_comb_4d), rtol=0.0, atol=0.0
            )
            np.testing.assert_array_equal(
                np.asarray(lay_4d_out.astype(jnp.float32)),
                np.asarray(ref_lay_4d.astype(jnp.float32)),
            )

            # 3) Call mhc_fused_post_pre with 3D residual (len(outer_shape) == 1)
            res_3d_bf16 = res_4d_bf16.reshape(num_tokens, hc_mult, hidden_size)
            post_3d = post_4d.reshape(num_tokens, hc_mult, 1)
            comb_3d = comb_4d.reshape(num_tokens, hc_mult, hc_mult)
            res_3d_out, post_3d_out, comb_3d_out, lay_3d_out = (
                fused_post_pre_kernel.mhc_fused_post_pre(
                    x_2d_bf16,
                    res_3d_bf16,
                    post_3d,
                    comb_3d,
                    fn,
                    hc_scale_f32,
                    hc_base_f32,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                    token_block_size=8,
                )
            )
            assert res_3d_out.shape == (num_tokens, hc_mult, hidden_size)
            assert post_3d_out.shape == (num_tokens, hc_mult, 1)
            assert comb_3d_out.shape == (num_tokens, hc_mult, hc_mult)
            assert lay_3d_out.shape == (num_tokens, hidden_size)
