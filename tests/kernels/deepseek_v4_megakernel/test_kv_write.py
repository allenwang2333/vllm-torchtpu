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
"""Compressor state + compressed-row store vs the stock plain-JAX goldens
(ref_wkv_proj_and_save_state, ref_compress_norm_rope_store): bit-exact."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_write_cases as K  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import kv_write  # noqa: E402

SEQ = {"hca": [130, 259, 1030], "csa": [11, 40, 103], "idx": [11, 40, 103]}


def _run(case):
    return kv_write.compress_store(
        case.mode, case.states, case.seq_lens, case.state_host, case.state_table, case.kv,
        case.kv_table, case.norm_w, case.cos_sin, case.rope,
        state_block_size=K.GEOM[case.mode][1], interpret=True)


def _sal_records(words):
    """int32 [N, 64, 256] seq-along-lane pages -> uint8 [N, 256, 129] records."""
    w = np.asarray(words).view(np.uint32)
    rec = np.zeros((w.shape[0], w.shape[2], 129), np.uint8)
    for g in range(32):
        for i in range(4):
            rec[:, :, 4 * g + i] = (w[:, g] >> (8 * i)) & 0xFF
    rec[:, :, 128] = w[:, 32] & 0xFF
    return rec


def _compare(case, out):
    gs, gk, gr = K.golden(case)
    ok = np.array_equal(np.asarray(out[0]), np.asarray(gs))
    if case.mode == "idx":
        has = np.asarray(gk).reshape(gk.shape[0], 256, 256)
        written = has.any(-1)
        ok &= np.array_equal(_sal_records(out[1])[written], has[written][:, :129])
        ok &= int(written.sum()) > 0
    else:
        ok &= np.array_equal(np.asarray(out[1]), np.asarray(gk))
        ok &= bool(np.any(np.asarray(out[1]) != np.asarray(case.kv)))  # something was stored
    if case.mode == "csa":
        ok &= np.array_equal(np.asarray(out[2]), np.asarray(gr))
    return ok


@pytest.mark.parametrize("mode", ["hca", "csa", "idx"])
def test_bitexact_vs_stock_goldens(mode):
    case = K.make_case(np.random.default_rng(0), mode, SEQ[mode])
    assert _compare(case, _run(case))


def test_positive_control():
    """One window state perturbed in the cache (not in the in-flight rows) must
    change the stored HCA record."""
    case = K.make_case(np.random.default_rng(0), "hca", SEQ["hca"])
    out = _run(case)
    host = np.asarray(case.state_host).copy()
    page = int(case.state_table[0, 0])  # positions 0..31 of request 0: in its window
    host[page, 0:8] = np.float32(3.0).view(np.int32)
    bad = K.dataclasses.replace(case, state_host=host)
    assert not _compare(bad, out)
