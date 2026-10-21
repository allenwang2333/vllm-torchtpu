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
"""owner_attention_phase kind "swa" (DSpark draft) vs the stock-path reference, CPU interpret."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import attn_swa_harness as H  # noqa: E402

K = 6


def _check(ss, layer, inp, caches, out, num_req=None, heads=128, h0=0):
    B = len(ss.bases)
    num_req = B if num_req is None else num_req
    exp_c, exp_o = H.reference(ss, layer, inp, num_req)
    for i, (got, exp) in enumerate(zip(caches, exp_c)):
        assert np.array_equal(got, exp), f"cache of layer {i} differs"
    o = out[:num_req, :K * heads].reshape(num_req, K, heads, 512)
    e = exp_o[:num_req, :, h0:h0 + heads]
    err = np.abs(o - e).max() / np.abs(e).max()
    assert err <= 1e-2, err
    return err


@pytest.mark.parametrize("bases,ctx", [
    ([1000, 300], [(995, 7), (299, 3)]),       # 5 / 1 context rows accepted, block after
    ([129, 6], [(124, 7), (0, 7)]),            # window start at 0..; block over a page edge
    ([253, 2000], [(250, 4), (1994, 7)]),      # draft block straddles pages 1 / 2
])
def test_draft_layer(bases, ctx):
    rng = np.random.default_rng(sum(bases))
    ss = H.build(rng, 3, bases, ctx, K, 4096)
    inp = H.make_inputs(rng, ss)
    caches, out = H.run_phase(ss, 1, inp)
    _check(ss, 1, inp, caches, out)


def test_idle_and_partial():
    rng = np.random.default_rng(3)
    ss = H.build(rng, 3, [700, 900], [(694, 7), (894, 7)], K, 2048)
    inp = H.make_inputs(rng, ss)
    caches, _ = H.run_phase(ss, 2, inp, num_req=0)
    for got, exp in zip(caches, ss.arrays):
        assert np.array_equal(got, exp)
    caches, out = H.run_phase(ss, 2, inp, num_req=1)
    _check(ss, 2, inp, caches, out, num_req=1)


def test_positive_control():
    """Causal (not bidirectional) in-block visibility must fail the check."""
    rng = np.random.default_rng(4)
    ss = H.build(rng, 3, [1000], [(995, 7)], K, 2048)
    inp = H.make_inputs(rng, ss)
    caches, out = H.run_phase(ss, 0, inp)
    _check(ss, 0, inp, caches, out)
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP
    orig = AP.window_config
    AP.window_config = lambda d, kind="": orig(d, "")  # causal block
    try:
        caches, out = H.run_phase(ss, 0, inp)
    finally:
        AP.window_config = orig
    with pytest.raises(AssertionError):
        _check(ss, 0, inp, caches, out)


@pytest.mark.parametrize("h0", [0, 48])
def test_group_tp_heads(h0):
    """attn_heads=16 (group TP): the 16-head slice of the full-width reference; a wrong
    slice fails (positive control)."""
    rng = np.random.default_rng(21)
    ss = H.build(rng, 3, [1000, 300], [(995, 7), (299, 3)], K, 4096)
    inp = H.make_inputs(rng, ss)
    caches, out = H.run_phase(ss, 1, inp, attn_heads=16, h0=h0)
    _check(ss, 1, inp, caches, out, heads=16, h0=h0)
    with pytest.raises(AssertionError):
        _check(ss, 1, inp, caches, out, heads=16, h0=h0 + 16)
