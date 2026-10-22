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
"""TPU_SEGMENT_MEGAKERNEL x TPU_GROUP_BATCHED_GATHER: a skip layer that is a
segment megakernel head (its KV write arrives as the megakernel's packed
per-rank `kv_rows`) and a group member (its rows pre-gathered at the group
head) runs `sparse_mla_attention_dcp(kv_rows=..., group_pre_gathered=...)`:
the patch call writes the megakernel's rows and re-gathers the stale atoms.

Checked through `sparse_mla_attention_dcp` on DCP ranks with
test_group_batched_gather's harness (test_mla_decode_prologue's prefill +
decode steps, production env): on the two decode steps (16 and 32 tokens),
`kv_rows` alone and `kv_rows` + the group gather each give caches, partial
outputs and lse bit-identical to the incumbent (latents in, per-layer gather).
`kv_rows` is packed per rank exactly as the incumbent packs its write
(`_get_dcp_page_and_slot` with that rank's owner mask -> `pack_write_rows`).
Positive control: rows packed with the next rank's owner mask must change
the caches (so the rows are really what gets written).
"""

import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_group_batched_gather as gbg  # noqa: E402
import test_mla_decode_prologue as prologue  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather  # noqa: E402

pytestmark = pytest.mark.tpu_test


def _kv_rows(kv_c, k_pe, seq_lens, bt, qsl, mesh, ns, rs, dcp, il, shift=0):
    """Each rank's packed KV write rows (int32 bits, sharded over DCP), as
    the segment megakernel hands them to the head layer."""
    from jax.experimental import shard_map
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.kernels.mla import kv_cache_utils
    from vllm_torchtpu.layers.core import attention_interface

    axis = streamindex_topk.DCP_AXIS_NAME

    def local(kv_c_, k_pe_, seq_lens_, bt_, qsl_):
        rank = (jax.lax.axis_index(axis) + shift) % dcp
        page, slot = kv_cache_utils._get_dcp_page_and_slot(
            kv_c_.shape[0],
            seq_lens_,
            bt_,
            qsl_,
            rank,
            nope_spec=ns,
            dcp_size=dcp,
            interleave_size=il,
        )
        rows = native_sc_gather.pack_write_rows(
            kv_cache_utils._dst_rows(page, slot, ns.page_size),
            attention_interface._pack_lane_dense(kv_c_, ns.token_bytes),
            attention_interface._pack_lane_dense(k_pe_, rs.token_bytes),
        )
        return jax.lax.bitcast_convert_type(rows, jnp.int32)

    return jax.jit(
        shard_map.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(None, None), P(None, None), P(None), P(None), P(None)),
            out_specs=P(axis, None),
            check_rep=False,
        )
    )(kv_c, k_pe, seq_lens, bt, qsl)


def _run(dcp, *, kv_rows, group, shift=0):
    from vllm_torchtpu.layers.core import attention_interface

    env = dict(
        prologue.PROD_ENV,
        TPU_MLA_DECODE_PROLOGUE="1",
        TPU_MLA_KV_WRITE_IN_GATHER="1",
    )
    if group:
        env[gbg.FLAG] = "1"
    orig = attention_interface.sparse_mla_attention_dcp
    taken = []

    def stacked(*args, **kw):
        (_, _, kv_c, k_pe, nope, rope, topk, sl, bt, qsl, dist, mesh, ns, rs) = args[
            :14
        ]
        # The decode steps (16 and 32 tokens): the fused-prep decode gather
        # with the KV write in it, the buckets both levers ride
        # (test_group_batched_gather asserts the group path takes exactly these).
        if kv_c.shape[0] not in (16, 32):
            return orig(*args, **kw)
        args = list(args)
        if group:
            others = [
                (nope ^ jnp.uint32(m), rope ^ jnp.uint32(m))
                for m in (0x5A5A5A5A, 0x0F0F0F0F)
            ]
            pre = attention_interface.sparse_mla_group_gather_dcp(
                [(nope, rope), *others],
                nope,
                topk,
                bt,
                qsl,
                dist,
                mesh,
                ns,
                dcp_size=kw["dcp_size"],
            )
            kw = dict(kw, group_pre_gathered=pre[:2])
        if kv_rows:
            rows = _kv_rows(
                kv_c,
                k_pe,
                sl,
                bt,
                qsl,
                mesh,
                ns,
                rs,
                kw["dcp_size"],
                kw["interleave_size"],
                shift=shift,
            )
            args[2] = args[3] = None
            kw = dict(kw, kv_rows=rows)
        taken.append(kv_c.shape[0])
        return orig(*args, **kw)

    jax.clear_caches()
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(attention_interface, "sparse_mla_attention_dcp", stacked),
        mock.patch.object(
            native_sc_gather,
            "dsa_gather_native_sc_group",
            wraps=native_sc_gather.dsa_gather_native_sc_group,
        ) as group_spy,
    ):
        if not group:
            os.environ.pop(gbg.FLAG, None)
        res = prologue._dcp_run(dcp, candidate=True)
        patches = sum(
            c.kwargs.get("gathered") is not None
            and c.kwargs.get("kv_write") is not None
            for c in group_spy.call_args_list
        )
    return res, taken, patches


def _dcp():
    gbg._require_tpu()
    dcp = gbg._dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    return dcp


def _assert_equal(ref, got, tag):
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope", "out", "lse"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"{tag} step {i} {k}")


def test_kv_rows_alone_bit_identical():
    dcp = _dcp()
    ref, _, _ = _run(dcp, kv_rows=False, group=False)
    got, taken, patches = _run(dcp, kv_rows=True, group=False)
    assert taken == [16, 32] and patches == 0, (taken, patches)
    _assert_equal(ref, got, "kv_rows")


def test_kv_rows_with_group_gather_bit_identical():
    dcp = _dcp()
    ref, _, _ = _run(dcp, kv_rows=False, group=False)
    got, taken, patches = _run(dcp, kv_rows=True, group=True)
    # Both decode steps take kv_rows + the pre-gathered rows: one patch each.
    assert taken == [16, 32] and patches == 2, (taken, patches)
    _assert_equal(ref, got, "kv_rows+group")


def test_kv_rows_positive_control_fails():
    dcp = _dcp()
    ref, _, _ = _run(dcp, kv_rows=False, group=False)
    got, taken, _ = _run(dcp, kv_rows=True, group=True, shift=1)
    assert taken == [16, 32]
    diff = {
        k: [not np.array_equal(a[k], b[k]) for a, b in zip(ref, got)]
        for k in ("nope", "rope", "out", "lse")
    }
    print(f"wrong-owner kv_rows: differs per step {diff}")
    assert any(diff["nope"] + diff["rope"]), "wrong-owner kv_rows not caught"
