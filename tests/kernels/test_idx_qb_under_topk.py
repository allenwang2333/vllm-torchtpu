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
"""`TPU_IDX_QB_UNDER_TOPK`: q_b_proj inside the DCP indexer's merge window.

`streamindex_topk_dcp(..., side_fn, side_args)` must return the same top-k,
bit for bit, as the call without the side work, and a side output equal bit
for bit to `side_fn(*side_args)` run on its own. And it must actually move the
work: in the compiled module the fp8_skinny kernel sits between the
cross-rank merge's SparseCore call-update and call-done. A side output that
differs from the reference (a perturbed weight) is the positive control for
the value check.
"""

import importlib.util
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk_dcp

pytestmark = pytest.mark.tpu_test


def _load(rel):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)
    spec = importlib.util.spec_from_file_location(
        "_qb_" + os.path.basename(rel)[:-3], path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _side_args(seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((16, 2048)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((2048, 16384)) * 0.05, jnp.float8_e4m3fn)
    s = jnp.asarray(rng.uniform(0.5, 1.5, (16384,)), jnp.float32)
    return x, w, s


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [4, 8])
def test_q_b_under_topk_same_values_and_inside_window(dcp_size):
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import _q_b_side

    if len(jax.devices()) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices")
    case = _load("kernels/test_dsv4_streamindex_topk_dcp.py")._dcp_case
    # A decode-sized bucket: 16 tokens, the all-gather merge.
    _, args, kwargs = case(dcp_size, 3, q_len=16)
    kwargs["num_queries_per_block"] = 16
    side = _side_args()

    base = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    topk, q = streamindex_topk_dcp(
        *args, **kwargs, side_fn=_q_b_side, side_args=side
    )
    np.testing.assert_array_equal(np.asarray(topk), base)
    ref = np.asarray(jax.jit(_q_b_side)(*side))
    np.testing.assert_array_equal(np.asarray(q), ref)
    # Positive control: the value check sees a one-element weight change.
    x, w, s = side
    w2 = w.at[7, 9].set(jnp.float8_e4m3fn(3.0))
    assert not np.array_equal(np.asarray(jax.jit(_q_b_side)(x, w2, s)), ref)

    txt = (
        streamindex_topk_dcp.lower(*args, **kwargs, side_fn=_q_b_side, side_args=side)
        .compile()
        .as_text()
    )
    entry = txt[txt.index("ENTRY") :]
    order = []
    for line in entry.splitlines():
        m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = ", line)
        if not m:
            continue
        if "fp8_skinny" in line and "custom-call" in line:
            order.append("SKINNY")
        elif re.search(r"sc_topk.*(update|done)", m.group(1)):
            order.append("upd" if "update" in m.group(1) else "done")
    print("ORDER", order)
    # The cross-rank merge is the last SparseCore top-k of the program.
    i = order.index("SKINNY")
    merge_upd = max(j for j, o in enumerate(order) if o == "upd")
    merge_done = max(j for j, o in enumerate(order) if o == "done")
    assert merge_upd < i < merge_done, order
