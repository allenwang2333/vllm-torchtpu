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
"""TPU_MLA_ABSORB_TOKEN_MAJOR kernel: token-major per-head matmul vs the bmm.

The reference is the bmm formulation the MLA absorbs use today (transpose in,
batched dot, transpose out) with the same epilogue, compiled by XLA. Shapes are
GLM-5.2's per rank: W_UK `[64, 192, 512]` (qrep group heads; 4 heads without
qrep) fed the whole `[T, H, 256]` q head or its 192-wide nope slice, and W_UV
`[64, 512, 256]`. T covers every serving token padding, including the ones
below the kernel's 8-token tile (they take the reference path).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import head_matmul as hm

pytestmark = pytest.mark.tpu_test

F8 = jnp.float8_e4m3fn


def _inputs(t, h, dq, p, l, wdt, with_scale, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed + t + h + dq + l), 3)
    x = jax.random.normal(k1, (t, h, dq), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (h, p, l), jnp.float32) * 0.05).astype(wdt)
    s = None
    if with_scale:
        s = jax.random.uniform(k3, (h, 1, l), jnp.float32, 0.5, 1.5).astype(
            jnp.bfloat16
        )
    return x, w, s


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)


# (heads, q width, contraction p, out width): W_UK qrep with the whole q head
# and with the nope slice, W_UK without qrep, W_UV qrep / no qrep.
SHAPES = [(64, 256, 192, 512), (64, 192, 192, 512), (4, 192, 192, 512),
          (64, 512, 512, 256), (4, 512, 512, 256)]
TOKENS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.mark.parametrize("t", TOKENS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "wdt,with_scale,round_before_scale,out_dtype",
    [
        (jnp.bfloat16, False, True, jnp.bfloat16),  # W_UK / W_UV, bf16 weights
        (F8, True, True, jnp.bfloat16),  # W_UK / end-of-forward W_UV under QOL
        (F8, True, False, jnp.bfloat16),  # project_uv_heads f32 path, narrow
        (F8, True, False, jnp.float32),  # project_uv_heads f32 path
    ],
)
def test_matches_bmm_bitwise(t, shape, wdt, with_scale, round_before_scale, out_dtype):
    h, dq, p, l = shape
    x, w, s = _inputs(t, h, dq, p, l, wdt, with_scale)
    kw = dict(round_before_scale=round_before_scale, out_dtype=out_dtype)
    got = hm.head_matmul(x, w, s, **kw)
    want = jax.jit(lambda x, w, s: hm.reference(x, w, s, **kw))(x, w, s)
    assert got.shape == (t, h, l) and got.dtype == jnp.dtype(out_dtype)
    np.testing.assert_array_equal(_bits(got), _bits(want))


def test_whole_q_head_equals_nope_slice():
    x, w, s = _inputs(256, 64, 256, 192, 512, F8, True)
    a = hm.head_matmul(x, w, s)
    b = hm.head_matmul(x[..., :192], w, s)
    np.testing.assert_array_equal(_bits(a), _bits(b))


def test_kernel_path_taken_and_catches_a_head_swap():
    """Positive control: the comparison above sees a one-head permutation."""
    assert hm.supported(4096, 64) and hm.supported(8, 4) and not hm.supported(4, 64)
    x, w, s = _inputs(512, 64, 256, 192, 512, jnp.bfloat16, False)
    got = hm.head_matmul(x, w, s)
    w_bad = w.at[jnp.array([3, 4])].set(w[jnp.array([4, 3])])
    bad = hm.reference(x, w_bad, s)
    assert not np.array_equal(_bits(got), _bits(bad))
    text = jax.jit(hm.head_matmul).lower(x, w).as_text()
    assert "tpu_custom_call" in text
