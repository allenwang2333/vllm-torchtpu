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
"""CPU unit tests for TD8.96 compress_and_store, mHC, and streamindex_topk."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compress_store_ref,
    compute,
    config,
    kernel,
)
from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    fused_post_pre_kernel,
    post_kernel,
    pre_kernel,
)
from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    utils as mhc_utils,
)


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    return idx


class MockArrayRef:
    """Mock buffer reference for simulating Pallas kernels on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        return jnp.asarray(self.arr[_norm_idx(idx)])

    def __setitem__(self, idx, val):
        if isinstance(val, MockArrayRef):
            val = val.arr
        self.arr[_norm_idx(idx)] = np.asarray(val)

    def bitcast(self, dtype):
        return MockArrayRef(self.arr.view(dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return MockArrayRef(self.arr.reshape(*shape))


class _DummyRef:
    """Numpy-backed array wrapper supporting `[...]` and slice writes."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        sub = self.arr[idx]
        if isinstance(sub, np.ndarray):
            return _DummyRef(sub)
        return sub

    def __setitem__(self, idx, val):
        if isinstance(val, _DummyRef):
            val = val.arr
        self.arr[idx] = np.asarray(val)


class TestMhcUtilsTD896:
    """100% line and branch coverage of mhc/utils.py."""

    def test_round_up_and_select_token_block(self):
        assert mhc_utils.round_up(15, 16) == 16
        assert mhc_utils.round_up(16, 16) == 16
        assert mhc_utils.round_up(17, 16) == 32
        assert mhc_utils.round_up(0, 16) == 0

        tb, pad = mhc_utils.select_token_block(10, 64)
        assert (tb, pad) == (16, 16)

        tb, pad = mhc_utils.select_token_block(100, 64)
        assert (tb, pad) == (64, 128)

        tb, pad = mhc_utils.select_token_block(
            128, 64, vmem_need=lambda b: 200 if b > 16 else 50, vmem_limit_bytes=100
        )
        assert (tb, pad) == (16, 128)

    def test_pad_to_trim_to_split_fn3_and_gates(self):
        a = jnp.ones((4, 8), dtype=jnp.float32)
        b = jnp.zeros((4, 16), dtype=jnp.bfloat16)

        assert mhc_utils.pad_to(4, a, b)[0].shape == (4, 8)
        res_pad = mhc_utils.pad_to(6, a, b)
        assert res_pad[0].shape == (6, 8)
        assert (res_pad[0][4:] == 0).all()

        assert mhc_utils.trim_to(6, *res_pad)[0].shape == (6, 8)
        assert mhc_utils.trim_to(4, *res_pad)[0].shape == (4, 8)

        fn = jnp.ones((8, 64), dtype=jnp.float32) * 1.5
        fn_hi, fn_mid, fn_lo = mhc_utils.split_fn3(fn)
        assert fn_hi.dtype == jnp.bfloat16
        assert fn_mid.dtype == jnp.bfloat16
        assert fn_lo.dtype == jnp.bfloat16

        T, hc_mult, hidden_size = 4, 4, 64
        hc_mult3 = hc_mult * (hc_mult + 2)
        mixes = jnp.ones((T, hc_mult3), dtype=jnp.float32) * 0.1
        sqrsum = jnp.ones((T, 1), dtype=jnp.float32) * 10.0
        hc_scale = jnp.ones((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        for repeat in (1, 3):
            pre_mix, post_mix, comb_mix = mhc_utils.mhc_pre_gates(
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
                sinkhorn_repeat=repeat,
            )
            assert pre_mix.shape == (T, hc_mult)
            assert post_mix.shape == (T, hc_mult)
            assert comb_mix.shape == (T, hc_mult, hc_mult)


class TestMhcPostKernelTD896:
    """100% line and branch coverage of mhc/post_kernel.py."""

    def test_post_kernel_and_wrappers(self):
        tb, hc_mult, hidden_size = 4, 2, 8
        x_ref = MockArrayRef(np.ones((tb, hidden_size), dtype=np.float16))
        res_ref = MockArrayRef(
            np.ones((tb, hc_mult * hidden_size), dtype=np.float16) * 0.5
        )
        post_ref = MockArrayRef(np.ones((tb, hc_mult), dtype=np.float32) * 2.0)
        comb_ref = MockArrayRef(
            np.ones((tb, hc_mult * hc_mult), dtype=np.float32) * 0.1
        )
        out_ref = MockArrayRef(np.zeros((tb, hc_mult * hidden_size), dtype=np.float16))

        post_kernel._post_kernel(
            x_ref,
            res_ref,
            post_ref,
            comb_ref,
            out_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )
        assert np.isfinite(out_ref.arr).all()

        T, hc_mult, hidden_size = 8, 4, 32
        x = jnp.zeros((T, hidden_size), dtype=jnp.bfloat16)
        res2d = jnp.zeros((T, hc_mult * hidden_size), dtype=jnp.bfloat16)
        post_layer_mix = jnp.zeros((T, hc_mult), dtype=jnp.float32)
        comb_res_mix = jnp.zeros((T, hc_mult, hc_mult), dtype=jnp.float32)

        post_2d_fn = post_kernel.mhc_post_2d.__wrapped__
        with pytest.raises(AssertionError):
            post_2d_fn(x, res2d.astype(jnp.float32), post_layer_mix, comb_res_mix)
        with pytest.raises(AssertionError):
            post_2d_fn(x, res2d, post_layer_mix.astype(jnp.bfloat16), comb_res_mix)
        with pytest.raises(AssertionError):
            post_2d_fn(x, res2d, post_layer_mix, comb_res_mix.astype(jnp.bfloat16))

        def fake_pallas_call(fn, out_shape=None, in_specs=None, out_specs=None, **kw):
            for spec in in_specs or []:
                spec.index_map(0)
            if out_specs is not None:
                out_specs.index_map(0)
            return lambda *call_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        with (
            patch.object(post_kernel.pl, "pallas_call", side_effect=fake_pallas_call),
            patch.object(post_kernel, "mhc_post_2d", side_effect=post_2d_fn),
        ):
            out = post_2d_fn(x, res2d, post_layer_mix, comb_res_mix)
            assert out.shape == (T, hc_mult * hidden_size)

            out_pad = post_2d_fn(
                jnp.zeros((10, hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((10, hc_mult * hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((10, hc_mult), dtype=jnp.float32),
                jnp.zeros((10, hc_mult, hc_mult), dtype=jnp.float32),
                token_block_size=16,
            )
            assert out_pad.shape == (10, hc_mult * hidden_size)

            out_vmem = post_2d_fn(
                jnp.zeros((32, hidden_size), dtype=jnp.float32),
                jnp.zeros((32, hc_mult * hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((32, hc_mult), dtype=jnp.float32),
                jnp.zeros((32, hc_mult, hc_mult), dtype=jnp.float32),
                token_block_size=32,
                vmem_limit_bytes=1024,
            )
            assert out_vmem.shape == (32, hc_mult * hidden_size)

            B, S = 2, 4
            xb = jnp.zeros((B, S, hidden_size), dtype=jnp.bfloat16)
            resb = jnp.zeros((B, S, hc_mult, hidden_size), dtype=jnp.bfloat16)
            postb = jnp.zeros((B, S, hc_mult, 1), dtype=jnp.float32)
            combb = jnp.zeros((B, S, hc_mult, hc_mult), dtype=jnp.float32)
            with pytest.raises(AssertionError):
                post_kernel.mhc_post(xb, resb.astype(jnp.float32), postb, combb)
            out_w = post_kernel.mhc_post(xb, resb, postb, combb)
            assert out_w.shape == (B, S, hc_mult, hidden_size)


class TestMhcPreKernelTD896:
    """100% line and branch coverage of mhc/pre_kernel.py."""

    def test_pre_kernel_and_wrappers(self):
        T, hc_mult, hidden_size = 4, 2, 8
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x_ref = MockArrayRef(np.ones((T, hc_hidden), dtype=np.float16))
        fn_hi_ref = MockArrayRef(np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.1)
        fn_mid_ref = MockArrayRef(
            np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.01
        )
        fn_lo_ref = MockArrayRef(
            np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.001
        )
        sc_ref = MockArrayRef(np.ones((1, 3), dtype=np.float32))
        hb_ref = MockArrayRef(np.zeros((1, hc_mult3), dtype=np.float32))
        mixes_ref = MockArrayRef(np.zeros((T, hc_mult3), dtype=np.float32))
        sqrsum_ref = MockArrayRef(np.zeros((T, 1), dtype=np.float32))
        layer_ref = MockArrayRef(np.zeros((T, hidden_size), dtype=np.float16))

        pre_kernel._mixes_kernel(
            x_ref, fn_hi_ref, fn_mid_ref, fn_lo_ref, mixes_ref, sqrsum_ref
        )
        assert np.isfinite(mixes_ref.arr).all()

        pre_kernel._mixes_collapse_kernel(
            x_ref,
            fn_hi_ref,
            fn_mid_ref,
            fn_lo_ref,
            sc_ref,
            hb_ref,
            mixes_ref,
            sqrsum_ref,
            layer_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
        )
        assert np.isfinite(layer_ref.arr).all()

        T, hc_mult, hidden_size = 8, 4, 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)
        x2d = jnp.zeros((T, hc_hidden), dtype=jnp.bfloat16)
        fn = jnp.zeros((hc_mult3, hc_hidden), dtype=jnp.float32)
        hc_scale = jnp.zeros((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        mixes_fn = pre_kernel.mhc_pre_mixes.__wrapped__
        collapse_fn = pre_kernel.mhc_pre_mixes_collapse.__wrapped__

        with pytest.raises(AssertionError):
            mixes_fn(x2d.astype(jnp.float32), fn)
        with pytest.raises(AssertionError):
            mixes_fn(x2d, fn.astype(jnp.bfloat16))
        with pytest.raises(AssertionError):
            mixes_fn(x2d, jnp.zeros((hc_mult3, hc_hidden + 1)))

        with pytest.raises(AssertionError):
            collapse_fn(x2d.astype(jnp.float32), fn, hc_scale, hc_base, 1e-6, 1e-6)
        with pytest.raises(AssertionError):
            collapse_fn(x2d, fn.astype(jnp.bfloat16), hc_scale, hc_base, 1e-6, 1e-6)
        with pytest.raises(AssertionError):
            collapse_fn(
                x2d,
                jnp.zeros((hc_mult3, hc_hidden + 1)),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )

        def fake_pallas_call(fn, out_shape=None, in_specs=None, out_specs=None, **kw):
            for spec in (in_specs or []) + (out_specs or []):
                spec.index_map(0)
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with (
            patch.object(pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call),
            patch.object(pre_kernel, "mhc_pre_mixes_collapse", side_effect=collapse_fn),
        ):
            mixes, sqrsum = mixes_fn(x2d, fn)
            assert mixes.shape == (T, hc_mult3)
            assert sqrsum.shape == (T, 1)

            m32, s32 = mixes_fn(
                jnp.zeros((32, hc_hidden), dtype=jnp.bfloat16),
                fn,
                token_block_size=32,
                vmem_limit_bytes=1024,
            )
            assert m32.shape == (32, hc_mult3)

            mixes_c, sqr_c, lay_c = collapse_fn(x2d, fn, hc_scale, hc_base, 1e-6, 1e-6)
            assert lay_c.shape == (T, hidden_size)

            m32c, _, _ = collapse_fn(
                jnp.zeros((32, hc_hidden), dtype=jnp.bfloat16),
                fn,
                hc_scale.astype(jnp.bfloat16),
                hc_base.astype(jnp.bfloat16),
                1e-6,
                1e-6,
                token_block_size=32,
                vmem_limit_bytes=1024,
            )
            assert m32c.shape == (32, hc_mult3)

            B, S = 2, 4
            residual = jnp.zeros((B, S, hc_mult, hidden_size), dtype=jnp.bfloat16)
            with pytest.raises(AssertionError):
                pre_kernel.mhc_pre(
                    residual.astype(jnp.float32),
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
                    residual,
                    fn.astype(jnp.bfloat16),
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    2,
                )
            pm, cm, lay = pre_kernel.mhc_pre(
                residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 2
            )
            assert pm.shape == (B, S, hc_mult, 1)
            assert cm.shape == (B, S, hc_mult, hc_mult)
            assert lay.shape == (B, S, hidden_size)


class TestMhcFusedPostPreKernelTD896:
    """100% line and branch coverage of mhc/fused_post_pre_kernel.py."""

    def test_fused_kernel_and_wrappers(self):
        tb, hc_mult, hidden_size = 4, 2, 8
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x_ref = MockArrayRef(np.ones((tb, hidden_size), dtype=np.float16))
        res_ref = MockArrayRef(np.ones((tb, hc_hidden), dtype=np.float16))
        post_ref = MockArrayRef(np.ones((tb, hc_mult), dtype=np.float32))
        comb_ref = MockArrayRef(np.ones((tb, hc_mult * hc_mult), dtype=np.float32))
        fn_hi_ref = MockArrayRef(np.ones((hc_mult3, hc_hidden), dtype=np.float16))
        fn_mid_ref = MockArrayRef(
            np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.1
        )
        fn_lo_ref = MockArrayRef(
            np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.01
        )
        sc_ref = MockArrayRef(np.ones((1, 3), dtype=np.float32))
        hb_ref = MockArrayRef(np.zeros((1, hc_mult3), dtype=np.float32))
        newres_ref = MockArrayRef(np.zeros((tb, hc_hidden), dtype=np.float16))
        mixes_ref = MockArrayRef(np.zeros((tb, hc_mult3), dtype=np.float32))
        sqrsum_ref = MockArrayRef(np.zeros((tb, 1), dtype=np.float32))
        layer_ref = MockArrayRef(np.zeros((tb, hidden_size), dtype=np.float16))

        for prec in ("highest", "default"):
            fused_post_pre_kernel._fused_kernel(
                x_ref,
                res_ref,
                post_ref,
                comb_ref,
                fn_hi_ref,
                fn_mid_ref,
                fn_lo_ref,
                sc_ref,
                hb_ref,
                newres_ref,
                mixes_ref,
                sqrsum_ref,
                layer_ref,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                gemm_precision=prec,
                rms_eps=1e-6,
                hc_pre_eps=1e-6,
            )
            assert np.isfinite(newres_ref.arr).all()

        T, hc_mult, hidden_size = 8, 4, 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)
        x2d = jnp.zeros((T, hidden_size), dtype=jnp.bfloat16)
        res2d = jnp.zeros((T, hc_hidden), dtype=jnp.bfloat16)
        post2d = jnp.zeros((T, hc_mult), dtype=jnp.float32)
        comb2d = jnp.zeros((T, hc_mult * hc_mult), dtype=jnp.float32)
        fn = jnp.zeros((hc_mult3, hc_hidden), dtype=jnp.float32)
        hc_scale = jnp.zeros((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        fused_mixes_fn = fused_post_pre_kernel.fused_post_pre_mixes.__wrapped__
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d.astype(jnp.float32),
                res2d,
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                res2d.astype(jnp.float32),
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                res2d,
                post2d.astype(jnp.bfloat16),
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                res2d,
                post2d,
                comb2d.astype(jnp.bfloat16),
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                res2d,
                post2d,
                comb2d,
                fn.astype(jnp.bfloat16),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                jnp.zeros((T + 1, hc_hidden), dtype=jnp.bfloat16),
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                res2d,
                post2d,
                comb2d,
                jnp.zeros((hc_mult3, hc_hidden + 1)),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_mixes_fn(
                x2d,
                res2d,
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                gemm_precision="invalid",
            )

        def fake_pallas_call(fn, out_shape=None, in_specs=None, out_specs=None, **kw):
            for spec in (in_specs or []) + (out_specs or []):
                spec.index_map(0)
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with (
            patch.object(
                fused_post_pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call
            ),
            patch.object(
                fused_post_pre_kernel,
                "fused_post_pre_mixes",
                side_effect=fused_mixes_fn,
            ),
        ):
            B, S = 2, 4
            xb = jnp.zeros((B, S, hidden_size), dtype=jnp.bfloat16)
            resb = jnp.zeros((B, S, hc_mult, hidden_size), dtype=jnp.bfloat16)
            postb = jnp.zeros((B, S, hc_mult), dtype=jnp.float32)
            combb = jnp.zeros((B, S, hc_mult, hc_mult), dtype=jnp.float32)

            with pytest.raises(AssertionError):
                fused_post_pre_kernel.mhc_fused_post_pre(
                    xb,
                    resb.astype(jnp.float32),
                    postb,
                    combb,
                    fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    2,
                )

            new_res, post_mix, comb_mix, layer_in = (
                fused_post_pre_kernel.mhc_fused_post_pre(
                    xb,
                    resb,
                    postb,
                    combb,
                    fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    2,
                )
            )
            assert new_res.shape == (B, S, hc_mult, hidden_size)
            assert post_mix.shape == (B, S, hc_mult, 1)
            assert comb_mix.shape == (B, S, hc_mult, hc_mult)
            assert layer_in.shape == (B, S, hidden_size)

            new_res2, _, _, _ = fused_post_pre_kernel.mhc_fused_post_pre(
                xb.astype(jnp.float32),
                resb,
                postb,
                combb,
                fn,
                hc_scale.astype(jnp.bfloat16),
                hc_base.astype(jnp.bfloat16),
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )
            assert new_res2.shape == (B, S, hc_mult, hidden_size)


def test_config_modes_and_properties():
    config.state_block_size.cache_clear()

    assert config.select_mode(128, True) is config.Mode.CSA_INDEXER
    assert config.select_mode(512, True) is config.Mode.CSA
    assert config.select_mode(512, False) is config.Mode.HCA

    assert config.physical_page_size(config.Mode.CSA_INDEXER, 128, 4) == 8
    assert config.physical_page_size(config.Mode.CSA, 128, 4) == 32
    assert config.physical_page_size(config.Mode.HCA, 128, 128) == 2

    assert config.state_host_page_size(config.Mode.HCA, 128, 128) == 32
    assert config.state_host_page_size(config.Mode.CSA, 128, 4) == 32
    assert config.state_host_page_size(config.Mode.CSA_INDEXER, 128, 4) == 8

    assert config.last_dim_size(config.Mode.CSA_INDEXER) == 256
    assert config.last_dim_size(config.Mode.CSA) == 128
    assert config.row_size_bytes(config.Mode.CSA_INDEXER) == 1024
    assert config.row_size_bytes(config.Mode.CSA) == 512

    assert config.state_block_size(config.Mode.HCA, 128) == 4
    assert config.state_block_size(config.Mode.CSA, 128) == 2
    assert config.state_block_size(config.Mode.CSA_INDEXER, 128, 4, 128) == 2
    assert config.state_block_size(config.Mode.CSA, 128) == 2
    assert config.state_block_size.cache_info().hits >= 1

    cfg_hca = config.Configs.make(
        config.Mode.HCA,
        size_n=8,
        physical_page_size=2,
        state_block_size=4,
        state_physical_page_size=32,
    )
    assert not cfg_hca.dims.is_quantized
    assert cfg_hca.dims.has_rope
    assert cfg_hca.dims.nope_dtype == jnp.bfloat16
    assert cfg_hca.dims.rope_dtype == jnp.bfloat16
    assert cfg_hca.overlap_factor == 1
    assert cfg_hca.state_width == 512
    assert cfg_hca.head_tiles == 4
    assert cfg_hca.window == 128
    assert cfg_hca.nope_dim == 448
    assert cfg_hca.nope_store_dim == 512
    assert cfg_hca.half_rope == 32
    assert cfg_hca.rope_slot == 3
    assert cfg_hca.record_bytes == 1024
    assert cfg_hca.record_rows == 2
    assert cfg_hca.tokens_in_second_minor == 1
    assert cfg_hca.kv_stride == 2
    assert cfg_hca.physical_page_size == 2
    assert cfg_hca.state_physical_page_size == 32
    assert cfg_hca.state_rows_per_token == 8
    assert cfg_hca.field_rows == 4
    assert cfg_hca.state_page_capacity == 4
    assert cfg_hca.state_block_size == 4
    assert cfg_hca.kv_block_size == 1
    assert cfg_hca.rope_page_size == 0
    assert cfg_hca.pages_to_buffer_per_token == 33
    assert cfg_hca.window_shape() == (2, 4, 128, 4, 128)
    assert cfg_hca.window_bytes_shape() == (2, 4, 128, 4, 4, 128)
    assert cfg_hca.output_shape() == (4, 2, 4, 128)
    assert cfg_hca.page_buffer_shape() == (4, 33, 32, 4, 128)
    assert cfg_hca.cos_sin_shape() == (4, 64)
    assert cfg_hca.cache_shape(10) == (10, 2, 4, 128)
    assert cfg_hca.state_cache_shape(10) == (10, 32, 4, 128)

    cfg_csa = config.Configs.make(
        config.Mode.CSA,
        size_n=8,
        physical_page_size=32,
        state_block_size=2,
    )
    assert cfg_csa.dims.is_quantized
    assert cfg_csa.dims.nope_dtype == jnp.uint8
    assert cfg_csa.nope_store_dim == 448
    assert cfg_csa.record_bytes == 512
    assert cfg_csa.kv_stride == 1
    assert cfg_csa.rope_output_shape() == (4, 4, 128)
    assert cfg_csa.rope_cache_shape(10) == (10, 8, 4, 128)

    cfg_idx = config.Configs.make(
        config.Mode.CSA_INDEXER,
        size_n=8,
        physical_page_size=8,
        state_block_size=2,
    )
    assert cfg_idx.record_bytes == 256
    assert cfg_idx.tokens_in_second_minor == 4
    assert cfg_idx.kv_stride == 1


def test_compute_functions_and_coverage():
    key = jax.random.PRNGKey(42)
    x_rope = jax.random.normal(key, (2, 1, 128), dtype=jnp.float32)
    cos_32 = jnp.ones((2, 1, 32), dtype=jnp.float32)
    sin_32 = jnp.zeros((2, 1, 32), dtype=jnp.float32)
    out_rope = compute.interleaved_rope_vector(x_rope, cos_32, sin_32)
    np.testing.assert_allclose(np.asarray(out_rope), np.asarray(x_rope), atol=1e-6)

    with (
        patch.object(
            compute.pltpu, "bitcast", side_effect=jax.lax.bitcast_convert_type
        ),
        patch.object(
            compute.pltpu,
            "roll",
            side_effect=lambda x, shift, axis: jnp.roll(x, shift, axis=axis),
        ),
    ):
        x_q = jax.random.normal(key, (2, 4, 128), dtype=jnp.float32) * 10.0
        q_tiled, s_tiled = compute.quantize_fp8_tiled(x_q, 64)
        assert q_tiled.shape == (2, 4, 128)
        assert s_tiled.shape == (2, 4, 2)

        q_lp, s_lp = compute.quantize_fp8_lane_periodic(x_q, 64, 448)
        assert q_lp.shape == (2, 4, 128)
        assert s_lp.shape == (2, 1, 64)

        packed_lp = compute.pack_nope_lane_periodic(q_lp, s_lp, 448, 512, 128)
        assert packed_lp.shape == (2, 4, 128)

        # pack_nope_tiled: nope_dim < q_flat and nope_blocks < scale_flat
        packed_t1 = compute.pack_nope_tiled(
            q_tiled, s_tiled, nope_dim=128, block_size=64, nope_width_bytes=512
        )
        assert packed_t1.shape == (2, 4, 128)

        # pack_nope_tiled: nope_dim == q_flat and nope_blocks == scale_flat
        packed_t2 = compute.pack_nope_tiled(
            q_tiled[:, :1, :],
            s_tiled[:, :1, :],
            nope_dim=128,
            block_size=64,
            nope_width_bytes=256,
        )
        assert packed_t2.shape == (2, 2, 128)

        words = compute.rope_words_tiled(x_rope, 64)
        assert words.shape == (2, 1, 128)
        row = jnp.zeros((1, 128), dtype=jnp.int32)
        kv_slots = jnp.array([0, 1], dtype=jnp.int32)
        merged_row = compute.merge_rope_row(row, kv_slots, words, 64, 0)
        assert merged_row.shape == (1, 128)

        slots_val = jnp.zeros((4, 128), dtype=jnp.uint8)
        val_padded = jnp.ones((2, 1, 128), dtype=jnp.uint8)
        updated_slots = compute.merge_slot_updates(slots_val, kv_slots, val_padded, 0)
        assert updated_slots.shape == (4, 128)

    with (
        patch.object(compute.pl, "when", side_effect=lambda cond: lambda fn: fn()),
        patch.object(
            compute.pl,
            "loop",
            side_effect=lambda lo, hi, **kw: lambda fn: [fn(i) for i in range(lo, hi)],
        ),
    ):
        for overlap in (True, False):
            page_buf_idx = _DummyRef(np.ones((2, 3, 32, 4, 256), dtype=np.uint8))
            pos_ref = jnp.array([3, 7], dtype=jnp.int32)
            kv_win_idx = _DummyRef(np.zeros((2, 4, 1, 4, 128), dtype=np.uint8))
            sc_win_idx = _DummyRef(np.zeros((2, 4, 1, 4, 128), dtype=np.uint8))
            compute.gather_from_page_buffer(
                page_buf_idx,
                pos_ref,
                kv_win_idx,
                sc_win_idx,
                global_idx=0,
                num_tokens=2,
                window=4,
                block_size=4,
                pages_to_buffer_per_token=2,
                field_rows=1,
                state_rows_per_token=2,
                overlap=overlap,
                is_indexer=True,
            )
            assert np.all(kv_win_idx.arr == 1)
            assert np.all(sc_win_idx.arr == 1)

            page_buf_std = _DummyRef(np.ones((2, 3, 32, 4, 128), dtype=np.uint8))
            kv_win_std = _DummyRef(np.zeros((2, 4, 2, 4, 128), dtype=np.uint8))
            sc_win_std = _DummyRef(np.zeros((2, 4, 2, 4, 128), dtype=np.uint8))
            compute.gather_from_page_buffer(
                page_buf_std,
                pos_ref,
                kv_win_std,
                sc_win_std,
                global_idx=0,
                num_tokens=2,
                window=4,
                block_size=4,
                pages_to_buffer_per_token=2,
                field_rows=2,
                state_rows_per_token=8,
                overlap=overlap,
                is_indexer=False,
            )
            assert np.all(kv_win_std.arr == 1)
            assert np.all(sc_win_std.arr == 1)


def test_compress_store_ref_and_kernel_trace_all_modes():
    # Cover compress_norm_rope with differing rms_weight dtype (bfloat16 vs float32)
    kv_w = jnp.ones((2, 4, 8), dtype=jnp.float32)
    sc_w = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    vmask = jnp.ones((2, 4), dtype=jnp.bool_)
    cos_sin_8 = jnp.ones((8, 4), dtype=jnp.float32)
    c_pos = jnp.array([0, 4], dtype=jnp.int32)
    res_f32 = compress_store_ref.compress_norm_rope(
        kv_w, sc_w, vmask, jnp.ones((8,), dtype=jnp.float32), cos_sin_8, c_pos, 1e-6, 4
    )
    res_bf16 = compress_store_ref.compress_norm_rope(
        kv_w, sc_w, vmask, jnp.ones((8,), dtype=jnp.bfloat16), cos_sin_8, c_pos, 1e-6, 4
    )
    np.testing.assert_allclose(np.asarray(res_f32), np.asarray(res_bf16), atol=1e-5)

    # Trace kernel.compress_norm_rope_store across CSA, CSA_INDEXER, and HCA
    # 1. CSA mode: rope_cache=None raises ValueError
    cache_words = jnp.zeros((2, 32, 128), dtype=jnp.int32)
    state_words = jnp.zeros((2, 32, 128), dtype=jnp.int32)
    rope_words = jnp.zeros((2, 8, 128), dtype=jnp.int32)
    pos_4 = jnp.array([3, 7, 11, 15], dtype=jnp.int32)
    bt = jnp.array([0, 1], dtype=jnp.int32)
    req_4 = jnp.zeros((4,), dtype=jnp.int32)
    kv_slot_4 = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    rms_512 = jnp.ones((512,), dtype=jnp.float32)
    cos_sin_64 = jnp.ones((16, 64), dtype=jnp.float32)

    with pytest.raises(ValueError, match="rope_cache must be provided"):
        kernel.compress_norm_rope_store.trace(
            cache_words,
            pos_4,
            bt,
            req_4,
            kv_slot_4,
            rms_512,
            block_table_stride=2,
            state_block_size=2,
            state_cache=None,
            rope_cache=None,
            cos_sin_cache=cos_sin_64,
            compress_ratio=4,
            overlap=True,
            quant_block=64,
        )

    # Trace CSA with separate state_cache and unaligned num_tokens=6 (pad_len > 0)
    pos_6 = jnp.arange(6, dtype=jnp.int32)
    req_6 = jnp.zeros((6,), dtype=jnp.int32)
    kv_slot_6 = jnp.array([0, 1, 2, 3, 4, -1], dtype=jnp.int32)
    traced_csa = kernel.compress_norm_rope_store.trace(
        cache_words,
        pos_6,
        bt,
        req_6,
        kv_slot_6,
        rms_512,
        block_table_stride=2,
        state_block_size=2,
        state_cache=state_words,
        rope_cache=rope_words,
        cos_sin_cache=cos_sin_64,
        compress_ratio=4,
        overlap=True,
        quant_block=64,
    )
    assert traced_csa is not None

    # 2. Trace CSA_INDEXER mode (head_dim=128, head_tiles == 1, has_rope=True)
    cache_idx = jnp.zeros((2, 8, 4, 256), dtype=jnp.uint8)
    rms_128 = jnp.ones((128,), dtype=jnp.float32)
    traced_idx = kernel.compress_norm_rope_store.trace(
        cache_idx,
        pos_4,
        bt,
        req_4,
        kv_slot_4,
        rms_128,
        block_table_stride=2,
        state_block_size=2,
        state_cache=None,
        rope_cache=None,
        cos_sin_cache=cos_sin_64,
        compress_ratio=4,
        overlap=True,
        quant_block=128,
    )
    assert traced_idx is not None

    # 3. Trace HCA mode with RoPE (cos_sin_cache provided, head_tiles > 1)
    cache_hca = jnp.zeros((2, 32, 4, 128), dtype=jnp.uint8)
    traced_hca_rope = kernel.compress_norm_rope_store.trace(
        cache_hca,
        pos_4,
        bt,
        req_4,
        kv_slot_4,
        rms_512,
        block_table_stride=2,
        state_block_size=4,
        state_cache=None,
        rope_cache=None,
        cos_sin_cache=cos_sin_64,
        compress_ratio=4,
        overlap=False,
        quant_block=0,
    )
    assert traced_hca_rope is not None

    # 4. Trace HCA mode WITHOUT RoPE (cos_sin_cache=None -> has_rope=False)
    traced_hca_no_rope = kernel.compress_norm_rope_store.trace(
        cache_hca,
        pos_4,
        bt,
        req_4,
        kv_slot_4,
        rms_512,
        block_table_stride=2,
        state_block_size=4,
        state_cache=None,
        rope_cache=None,
        cos_sin_cache=None,
        compress_ratio=4,
        overlap=False,
        quant_block=0,
    )
    assert traced_hca_no_rope is not None
