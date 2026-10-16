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
"""`TPU_MLA_DECODE_BODY_LEVERS`: the bitwise body levers on the decode body.

`sparse_ragged_paged_attention(decode_body_levers=...)` must return the same
partials and lse (or send buffer), bit for bit, as without it, on the served
decode path (row widths, prologue rope, kernel-owned lse, capped SparseCore
gather) -- with per-token live counts that put every width bucket, the
overflow fix-up, empty rows and token pairs with one empty member into some
grid step, and on buckets 4 (one step: "nofill"), 8 / 16 (send layout) and
32. Positive control: the `row_width_undersize` fault must still be caught
with the levers on (the lever body consumes the step's width).
"""

import importlib.util
import os

import jax
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

pytestmark = pytest.mark.tpu_test

WIDTHS = (128, 256, 384)


def _load(rel):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)
    spec = importlib.util.spec_from_file_location(
        "_bl_" + os.path.basename(rel)[:-3], path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RW = _load("kernels/test_mla_decode_row_widths.py")
P = RW.P


def _call(p, levers, fault=None, send_world=0, batch=4):
    cache = P._cos_sin_cache()
    res = sparse_mla.sparse_ragged_paged_attention(
        p["ql_nope"],
        p["nope"],
        p["rope"],
        p["topk"],
        p["page_table"],
        p["cu_q_lens"],
        p["distribution"],
        sm_scale=P.SM_SCALE,
        k_scale=P.K_SCALE,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=256,
        topk_cap=512,
        small_chunk_batch_size=batch,
        small_chunk_tokens=32,
        lse_lane_dense=True,
        lse_mask_empty=True,
        fused_window_prep=True,
        mixed_live_gather=True,
        mixed_live_widths=(256, 384),
        mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT,
        q_pe=p["q_raw"],
        q_rope_table=P.rope_table(p["positions"], cache),
        lse_in_kernel=True,
        prologue_fault=fault,
        decode_row_widths=WIDTHS,
        decode_body_levers=levers,
        send_world=send_world,
    )
    out, lse = jax.block_until_ready(res[:2])
    if send_world:
        return np.asarray(lse)  # the send buffer; `out` is not written
    return P._bits16(out[..., : P.NOPE_DIM]), P._bits32(lse)


def _same(a, b):
    if isinstance(a, tuple):
        return all(np.array_equal(x, y) for x, y in zip(a, b))
    return np.array_equal(a.view(np.uint16), b.view(np.uint16))


CELLS = (
    (4, (4,), 0),
    (4, (3,), 0),
    (8, (4, 4), 0),
    (16, (4, 4, 4, 4), 0),
    (16, (4, 4, 4, 1), 0),
    (32, (4,) * 8, 0),
    (8, (4, 4), 8),
    (16, (4, 4, 4), 8),
)
LEVER_SETS = (
    sparse_mla.DECODE_BODY_DEFAULT,
    ("tr", "zu", "pv512", "noinf"),
    ("pair",),
    ("fxskip", "nofill"),
    ("q128",),
    ("tr", "zu", "pv512", "noinf", "quad", "fxskip", "nofill"),
    ("lrows",),
)


@pytest.mark.parametrize("query_tokens,seq_q,send_world", CELLS)
@pytest.mark.parametrize("shift", [0, 8])
def test_bit_identical(query_tokens, seq_q, send_world, shift):
    P._require_tpu()
    p = RW._problem(4040 + query_tokens, query_tokens, seq_q)
    counts = RW.COUNTS[shift:] + RW.COUNTS[:shift]
    p = RW._with_counts(p, counts)
    ref = _call(p, (), send_world=send_world)
    for levers in LEVER_SETS:
        got = _call(p, levers, send_world=send_world)
        assert _same(got, ref), levers
    # Eight tokens per grid step (TPU_DCP_MLA_SMALL_CHUNK_BATCH=8) with the
    # default levers: the same bits as the served four-token steps.
    if query_tokens >= 8:
        got = _call(p, sparse_mla.DECODE_BODY_DEFAULT, send_world=send_world, batch=8)
        assert _same(got, ref), "bz 8"


def test_positive_control_fails():
    P._require_tpu()
    p = RW._with_counts(RW._problem(2043, 32, (4,) * 8), RW.COUNTS)
    ref = _call(p, sparse_mla.DECODE_BODY_DEFAULT)
    bad = _call(p, sparse_mla.DECODE_BODY_DEFAULT, fault="row_width_undersize")
    assert not _same(bad, ref), "row_width_undersize not caught with the levers on"


def test_rejects_bad_levers():
    p = RW._with_counts(RW._problem(5, 16, (4,)), RW.COUNTS)
    for bad in (("zu",), ("xqk",), ("nope",), ("pair", "quad")):
        with pytest.raises(ValueError):
            _call(p, bad)
