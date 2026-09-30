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
"""Fused sparse-MLA prefill q kernel vs the packed-q reference."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse.fused_q import FusedQConfig, fused_sparse_q

NOPE, ROPE, LKV = 192, 64, 512


def _problem(t, h, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = (jax.random.normal(k[0], (t, h * (NOPE + ROPE))) * 2).astype(jnp.bfloat16)
    w = (jax.random.normal(k[1], (h, NOPE, LKV)) * 0.05).astype(jnp.bfloat16)
    ang = jax.random.uniform(k[2], (t, ROPE // 2)) * 6
    cos, sin = jnp.cos(ang).astype(jnp.bfloat16), jnp.sin(ang).astype(jnp.bfloat16)
    c = jnp.repeat(cos, 2, axis=-1)
    s = jnp.stack([-sin, sin], axis=-1).reshape(t, ROPE)
    return q, w, c, s


def _reference(q, w, c, s, rounding, f32_dot):
    t, h = q.shape[0], w.shape[0]
    q3 = q.reshape(t, h, NOPE + ROPE)
    qn, qp = q3[..., :NOPE], q3[..., NOPE:]
    even = (jnp.arange(ROPE) % 2) == 0
    sw = jnp.where(even, jnp.roll(qp, -1, -1), jnp.roll(qp, 1, -1))
    f = lambda x: x.astype(jnp.float32)  # noqa: E731
    pc, ps = f(qp) * f(c)[:, None], f(sw) * f(s)[:, None]
    if rounding == "b":
        pc, ps = f(pc.astype(jnp.bfloat16)), f(ps.astype(jnp.bfloat16))
    rot = (pc + ps).astype(jnp.bfloat16)
    if f32_dot:
        qn, w = f(qn), f(w)
    acc = jnp.einsum("thd,hdl->thl", qn, w, preferred_element_type=jnp.float32)
    zeros = jnp.zeros((t, h, 128 - ROPE), jnp.bfloat16)
    return jnp.concatenate([acc.astype(jnp.bfloat16), rot, zeros], axis=-1)


class FusedSparseQTest(parameterized.TestCase):
    @parameterized.product(
        opts=[
            "",
            "rope=roll",
            "round=a",
            "k=256",
            "tt=64,hb=4",
            "out=swap,hb=16",
            "out=einshape,hb=16",
            "out=sst",
        ],
        t=[256, 200],
    )
    def test_matches_reference(self, opts, t):
        interpret = jax.default_backend() != "tpu"
        cfg = FusedQConfig.parse(opts)
        q, w, c, s = _problem(t, 16)
        pad = ((0, 0), (0, 128 - ROPE))
        got = fused_sparse_q(
            q, jnp.pad(c, pad), jnp.pad(s, pad), w, cfg=cfg, interpret=interpret
        )
        want = _reference(q, w, c, s, cfg.rounding, interpret)
        self.assertEqual(got.shape, (t, 16, LKV + 128))
        g = np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint16))
        r = np.asarray(jax.lax.bitcast_convert_type(want, jnp.uint16))
        # RoPE columns and the zero pad: exact on every backend.
        np.testing.assert_array_equal(g[..., LKV:], r[..., LKV:])
        if interpret:
            np.testing.assert_allclose(
                np.asarray(got[..., :LKV], np.float32),
                np.asarray(want[..., :LKV], np.float32),
                rtol=1e-2,
                atol=1e-2,
            )
        else:
            np.testing.assert_array_equal(g[..., :LKV], r[..., :LKV])

        # Positive control: a one-element q change reaches the output.
        q_bad = q.at[t - 1, 0].add(1)
        bad = fused_sparse_q(
            q_bad, jnp.pad(c, pad), jnp.pad(s, pad), w, cfg=cfg, interpret=interpret
        )
        self.assertFalse(np.array_equal(np.asarray(bad), np.asarray(got)))


if __name__ == "__main__":
    absltest.main()
