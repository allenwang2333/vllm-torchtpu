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
"""CPU tests of step_impl: cache / block-table addressing for the attention
phase, the stock tail, and the contract fn end to end (interpret, R=4, in a
subprocess for its simulated devices)."""

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI

EPS = 1e-6

CR, LC = KI.CacheRef, KI.LayerCaches
LAYERS = (
    LC("hca", 128, swa=CR(0, 0), kv=CR(1, 1), state=CR(0, 2)),
    LC("hca", 128, swa=CR(2, 3), kv=CR(3, 1), state=CR(2, 2)),
    LC("csa", 4, swa=CR(0, 0), kv=CR(4, 1), state=CR(4, 2), kv_rope=CR(5, 1),
       indexer_k=CR(6, 1), indexer_state=CR(6, 2)),
)  # fmt: skip
WIDTHS = (3, 5, 4, 2)


def _cfg():
    spec = KI.ModelSpec(
        hidden_size=512, num_layers=3, compress_ratios=(128, 128, 4),
        dtype="bfloat16", q=7, mode="dp", axis_name="d", num_ranks=4,
        aux_layers=(), sliding_window=128,
    )  # fmt: skip
    return KI.StepConfig(
        model=spec, layers=LAYERS, cache_shapes=((), ) * 7,
        group_widths=WIDTHS, swa_page=128,
    )  # fmt: skip


def np_tail(res, aux, hc_fn, hc_scale, hc_base, norm_w):
    """The stock tail in numpy f32: layers.mhc_collapse_head + RMSNorm, and
    the aux mean over hc streams, with the stock bf16 rounding points."""

    def bf(x):
        return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16), np.float32)

    m, hch = res.shape
    h = hch // 4
    flat = np.asarray(res, np.float32)
    normed = flat / np.sqrt((flat * flat).mean(-1, keepdims=True) + EPS)
    mix = normed @ np.asarray(hc_fn, np.float32).T
    mix = 1 / (1 + np.exp(-(mix * np.asarray(hc_scale) + np.asarray(hc_base)))) + EPS
    x = bf((mix[:, :, None] * flat.reshape(m, 4, h)).sum(1))
    x = bf(x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS))
    hidden = bf(x * np.asarray(norm_w, np.float32))
    a = np.asarray(aux, np.float32)
    aux_h = bf(a.reshape(a.shape[0], m, 4, h).mean(2))
    return hidden, aux_h


def test_cache_plan_follows_the_layer_refs():
    plan = SI.CachePlan.of(_cfg())
    assert plan.layers["swa"] == (0, 1, 2) and plan.arrays["swa"] == (0, 2, 0)
    assert plan.layers["kv_rope"] == (2,) and plan.arrays["kv_rope"] == (5,)
    assert plan.arrays["state"] == (0, 2, 4) and plan.groups["state"] == (2, 2, 2)
    assert plan.swa_groups == (0, 3)


def test_attn_metadata_offsets_and_swa_pages():
    cfg = _cfg()
    rng = np.random.default_rng(0)
    B, q = 2, 7
    bts = [rng.integers(1, 99, (B, w)).astype(np.int32) for w in WIDTHS]
    seq = np.asarray([300, 0], np.int32)  # request 1 inactive
    md = SI.attn_metadata(cfg, SI.CachePlan.of(cfg), [jnp.asarray(b) for b in bts],
                          jnp.asarray(seq), q)  # fmt: skip
    flat = np.asarray(md["bt_flat"])
    # the group tables, then one [B, 3] window-page block per SWA group (0, 3)
    assert flat.shape == (B * sum(WIDTHS) + 2 * B * 3,)
    goff = np.cumsum((0,) + tuple(B * w for w in WIDTHS))[:-1]

    def table(role, li, r):
        off, w = int(md["bt_off"][role][li]), int(md["bt_w"][role][li])
        return flat[off + r * w : off + (r + 1) * w]

    for role in SI.ROLES:
        for li, lc in enumerate(LAYERS):
            ref = getattr(lc, role)
            if ref is None:
                assert int(md["bt_off"][role][li]) == int(md["bt_w"][role][li]) == 0
                continue
            assert int(md["bt_off"][role][li]) == goff[ref.group]
            for r in range(B):
                np.testing.assert_array_equal(table(role, li, r), bts[ref.group][r])
    for li, lc in enumerate(LAYERS):
        g = lc.swa.group
        want = attn_core.swa_window_pages(
            jnp.asarray(seq), jnp.asarray(bts[g]), q_len=q, window=128, swa_page=128
        )
        assert int(md["bt_w"]["swa_pages"][li]) == 3
        np.testing.assert_array_equal(table("swa_pages", li, 0), np.asarray(want)[0])
        np.testing.assert_array_equal(table("swa_pages", li, 1), 0)  # inactive
    # layers 0 and 2 share SWA group 0's page block
    assert int(md["bt_off"]["swa_pages"][0]) == int(md["bt_off"]["swa_pages"][2])
    assert int(md["bt_off"]["swa_pages"][1]) != int(md["bt_off"]["swa_pages"][0])


def test_tail_matches_an_independent_reading():
    rng = np.random.default_rng(1)
    M, H = 6, 256
    res = jnp.asarray(rng.standard_normal((M, 4 * H)), jnp.bfloat16)
    aux = jnp.asarray(rng.standard_normal((2, M, 4 * H)), jnp.bfloat16)
    fn = jnp.asarray(rng.standard_normal((4, 4 * H)) * 0.02, jnp.float32)
    base = jnp.asarray(rng.standard_normal(4), jnp.float32)
    scale = jnp.asarray([0.5], jnp.float32)
    norm_w = jnp.asarray(1 + 0.1 * rng.standard_normal(H), jnp.bfloat16)
    got = SI.collapse_head(res, fn, scale, base, norm_w, hc_mult=4, rms_eps=1e-6,
                           hc_eps=1e-6)  # fmt: skip
    want_h, want_a = np_tail(res, aux, fn, scale, base, norm_w)
    np.testing.assert_array_equal(np.asarray(got, np.float32), want_h)
    for a in range(2):
        np.testing.assert_array_equal(
            np.asarray(SI.hc_mean(aux[a], 4), np.float32), want_a[a]
        )


def test_contract_fn_end_to_end_interpret():
    env = dict(os.environ, JAX_PLATFORMS="cpu", STEP_R="4")
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
    ).strip()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))  # fmt: skip
    check = "tests.kernels.deepseek_v4_megakernel.step_impl_interpret_check"
    out = subprocess.run(
        [sys.executable, "-m", check],
        env=env, capture_output=True, text=True, check=True, cwd=root,
    ).stdout  # fmt: skip
    r = json.loads(next(ln for ln in out.splitlines() if ln.startswith("RESULT "))[7:])
    assert r["hidden_shape"] == [14, 512] and r["n_aux"] == 2, r
    assert r["hidden_rel"] == 0.0 and r["aux_rel"] == 0.0, r
    assert r["live_rows_nonzero"] and r["idle_rank_zero"] and r["caches_unchanged"], r
