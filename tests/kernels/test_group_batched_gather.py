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
"""TPU_GROUP_BATCHED_GATHER: one SparseCore gather for a layer group's skip
layers at the group head, then a per-layer write + stale-atom patch.

The claim is bit-identity with the incumbent per-layer gather (with the KV
write riding it, TPU_MLA_KV_WRITE_IN_GATHER). Checked

* multi mode == the incumbent gather of each layer alone (every live row);
* patch mode on the multi-mode rows taken *before* the write == the
  incumbent's write-then-gather, every rank of a DCP16/il4 group on the
  `test_mla_tc_kv_writer` cells, the step's own rows planted in the gather:
  every live gathered row and both whole caches;
* through `sparse_mla_attention_dcp` on DCP ranks (test_mla_decode_prologue's
  prefill + decode steps, production env incl. TPU_MLA_DECODE_PROLOGUE and
  TPU_MLA_KV_WRITE_IN_GATHER): each rank's caches, partial outputs and lse,
  with the group head's gather taken from the layer's pre-write cache;

with the positive control `no_patch` (write, but no re-gather) caught at both
levels: the decode steps select their own new rows, so stale rows must show.
"""

import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_mla_decode_prologue as prologue  # noqa: E402
import test_mla_kv_write_in_gather as kvw  # noqa: E402
import test_mla_tc_kv_writer as tcw  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather  # noqa: E402

pytestmark = pytest.mark.tpu_test

FLAG = "TPU_GROUP_BATCHED_GATHER"
OUT_SIZE = kvw.OUT_SIZE  # 4 query tokens x topk 2048
NUM_OWNERS = 32
LIVE_MULTIPLE = 2048  # whole owner rounds: no tail guard, as at decode
META_ROWS = 8


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the native SparseCore gather needs a TPU")


def _with_live(idx, num_live):
    """[OUT_SIZE] indices + the live vector right after them, the fused
    prep's single-operand layout (`live_vec_offset=OUT_SIZE`)."""
    occ = native_sc_gather.OCCURRENCES_PER_ATOM  # atoms_per_batch = 1
    live_b = min(-(-num_live // occ), OUT_SIZE // occ)
    owner_b = -(-live_b // NUM_OWNERS)
    flat = np.zeros(OUT_SIZE + META_ROWS * 128, np.int32)
    flat[:OUT_SIZE] = np.asarray(idx).reshape(-1)
    flat[OUT_SIZE : OUT_SIZE + 16] = live_b
    flat[OUT_SIZE + 16 : OUT_SIZE + 32] = owner_b
    return jnp.asarray(flat.reshape(-1, 128))


def _incumbent(caches, ind, kv_write=None):
    return jax.block_until_ready(
        native_sc_gather.dsa_gather_native_sc(
            caches[0],
            caches[1],
            ind,
            out_size=OUT_SIZE,
            atoms_per_batch=1,
            linear_indices=True,
            live_vec_offset=OUT_SIZE,
            live_multiple=LIVE_MULTIPLE,
            kv_write=kv_write,
        )
    )


def _group(layer_caches, ind, gathered=None, kv_write=None, fault=None):
    return jax.block_until_ready(
        native_sc_gather.dsa_gather_native_sc_group(
            tuple(layer_caches),
            ind,
            out_size=OUT_SIZE,
            live_vec_offset=OUT_SIZE,
            atoms_per_batch=1,
            live_multiple=LIVE_MULTIPLE,
            gathered=gathered,
            kv_write=kv_write,
            fault=fault,
        )
    )


def _rows_equal(got, want, num_live):
    groups = num_live // native_sc_gather.ROPE_PACKING
    return [
        np.array_equal(np.asarray(got[0])[:num_live], np.asarray(want[0])[:num_live]),
        np.array_equal(np.asarray(got[1])[:groups], np.asarray(want[1])[:groups]),
    ]


@pytest.mark.parametrize("num_live", [OUT_SIZE, 3 * 2048, 0])
def test_multi_matches_each_layer_alone(num_live):
    _require_tpu()
    layers = [
        tcw._inputs(*tcw._CELLS["decode_t16"], seed=s)["caches"] for s in (1, 2, 3)
    ]
    num_rows = layers[0][0].shape[0] * layers[0][0].shape[1]
    idx = np.random.default_rng(5).integers(0, num_rows, OUT_SIZE, dtype=np.int32)
    ind = _with_live(idx, num_live)
    outs = _group(layers, ind)
    assert len(outs) == 2 * len(layers)
    for g, caches in enumerate(layers):
        ok = _rows_equal(outs[2 * g : 2 * g + 2], _incumbent(caches, ind), num_live)
        assert all(ok), f"layer {g} live {num_live}: nope/rope {ok}"


def _patch_rank(inp, rank, num_live, fault=None):
    dst = tcw._dst_rows(inp, rank)
    num_rows = inp["caches"][0].shape[0] * inp["caches"][0].shape[1]
    ind = _with_live(kvw._indices(dst, num_rows, seed=rank), num_live)
    kv_write = native_sc_gather.pack_write_rows(dst, *kvw._packed(inp))
    # The head's read: this layer's cache before the step's write.
    stale = _group([inp["caches"]], ind)
    got = _group([inp["caches"]], ind, gathered=stale, kv_write=kv_write, fault=fault)
    want = _incumbent(inp["caches"], ind, kv_write)
    ok = _rows_equal(got, want, num_live) + [
        np.array_equal(np.asarray(got[2]), np.asarray(want[2])),
        np.array_equal(np.asarray(got[3]), np.asarray(want[3])),
    ]
    return ok, dst


@pytest.mark.parametrize("num_live", [OUT_SIZE, 0])
@pytest.mark.parametrize("cell", kvw._WRITE_CELLS)
def test_patch_matches_write_then_gather(cell, num_live):
    _require_tpu()
    seqs, num_tokens = tcw._CELLS[cell]
    inp = tcw._inputs(seqs, num_tokens)
    owned = 0
    for rank in range(tcw.DCP_SIZE):
        ok, dst = _patch_rank(inp, rank, num_live)
        owned += int(jnp.sum(dst >= 0))
        assert all(ok), f"{cell} rank {rank} live {num_live}: nope/rope/caches {ok}"
    assert owned == inp["num_valid"]


@pytest.mark.parametrize("cell", tcw._CONTROL_CELLS)
def test_patch_positive_control_fails(cell):
    """No re-gather: the caches still end right, but every owning rank's
    gathered rows (the planted step rows) must go stale."""
    _require_tpu()
    seqs, num_tokens = tcw._CELLS[cell]
    inp = tcw._inputs(seqs, num_tokens)
    owning = caught = 0
    for rank in range(tcw.DCP_SIZE):
        ok, dst = _patch_rank(inp, rank, OUT_SIZE, fault="no_patch")
        assert ok[2] and ok[3], f"{cell} rank {rank}: caches must still be right"
        if bool(jnp.any(dst >= 0)):
            owning += 1
            caught += not (ok[0] and ok[1])
    assert owning > 1
    assert caught == owning, f"no_patch missed on {owning - caught} ranks"


# --- DCP: sparse_mla_attention_dcp, caches + partials on every rank --------


def _dcp(dcp, group, group_fault=None):
    from vllm_torchtpu.layers.core import attention_interface

    env = dict(
        prologue.PROD_ENV,
        TPU_MLA_DECODE_PROLOGUE="1",
        TPU_MLA_KV_WRITE_IN_GATHER="1",
    )
    if group:
        env[FLAG] = "1"
    orig = attention_interface.sparse_mla_attention_dcp
    grouped = []

    def with_group_head(*args, **kw):
        (_, _, kv_c, _, nope, rope, topk, _, bt, qsl, dist, mesh, ns, rs) = args[:14]
        if group and attention_interface.group_gather_applies(
            ns, rs, kv_c.shape[0], topk.shape[-1]
        ):
            # Two more skip layers of the group (other cache contents) ride
            # the same call; this layer is the first.
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
            assert len(pre) == 6
            kw = dict(kw, group_pre_gathered=pre[:2], group_fault=group_fault)
            grouped.append(kv_c.shape[0])
        return orig(*args, **kw)

    jax.clear_caches()
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            attention_interface, "sparse_mla_attention_dcp", with_group_head
        ),
        mock.patch.object(
            native_sc_gather,
            "dsa_gather_native_sc_group",
            wraps=native_sc_gather.dsa_gather_native_sc_group,
        ) as group_spy,
    ):
        if not group:
            os.environ.pop(FLAG, None)
        res = prologue._dcp_run(dcp, candidate=True)
        patches = sum(
            c.kwargs.get("gathered") is not None for c in group_spy.call_args_list
        )
    return res, grouped, patches


def _dcp_size():
    return min(8, len(jax.devices()))


def test_dcp_caches_partials_and_lse_bit_identical():
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref, ref_grouped, ref_patches = _dcp(dcp, group=False)
    got, got_grouped, got_patches = _dcp(dcp, group=True)
    # Wiring: the two decode steps (16 and 32 tokens) take the group path,
    # each with one patch call; the mixed step does not.
    assert (ref_grouped, ref_patches) == ([], 0)
    assert got_grouped == [16, 32] and got_patches == 2, (got_grouped, got_patches)
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope", "out", "lse"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")


def test_dcp_positive_control_fails():
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref, _, _ = _dcp(dcp, group=False)
    got, grouped, _ = _dcp(dcp, group=True, group_fault="no_patch")
    assert grouped == [16, 32]
    diff = {
        k: [not np.array_equal(a[k], b[k]) for a, b in zip(ref, got)]
        for k in ("nope", "rope", "out", "lse")
    }
    print(f"no_patch: differs per step {diff}")
    # The caches end right; the attention reads stale rows.
    assert not any(diff["nope"] + diff["rope"])
    assert any(diff["out"] + diff["lse"]), "no_patch not caught"


def test_flag_defaults_off():
    from vllm_torchtpu import envs

    with mock.patch.dict(os.environ):
        os.environ.pop(FLAG, None)
        assert not envs.TPU_GROUP_BATCHED_GATHER


def test_group_registry_links_skip_layers_to_their_head():
    """GLM-5.2's pattern: 0-2 indexed (2 heads a group of 3, 0/1 none),
    3-5 skip, 6 indexed, 7-9 skip; the MTP layer (10: skip_topk toggled, but
    with an indexer) joins no group; a layer in another container, a 4th skip
    and a skip after the head froze its op are refused too."""
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as m

    class Wrapper:
        def __init__(self, skip, indexer=None):
            self.skip_topk = skip
            self.indexer = indexer if indexer is not None or skip else object()
            self._group_skips = ()

    skip = [i not in (0, 1, 2, 6) for i in range(10)]
    with mock.patch.dict(m._GROUP_LAYERS, clear=True):
        ws = [Wrapper(s) for s in skip]
        for i, w in enumerate(ws):
            m._register_group_layer(w, f"model.layers.{i}.self_attn")
        mtp = Wrapper(True, indexer=object())
        m._register_group_layer(mtp, "model.layers.10.self_attn")
        other = Wrapper(True)
        m._register_group_layer(other, "mtp.layers.3.self_attn")
        fourth = Wrapper(True)
        m._register_group_layer(fourth, "model.layers.10.self_attn")
        heads = [Wrapper(False), Wrapper(True)]
        m._register_group_layer(heads[0], "x.layers.0.self_attn")
        object.__setattr__(heads[0], "_group_frozen", True)
        m._register_group_layer(heads[1], "x.layers.1.self_attn")
    assert ws[0]._group_skips == () and ws[1]._group_skips == ()
    assert ws[2]._group_skips == (ws[3], ws[4], ws[5])
    assert ws[6]._group_skips == (ws[7], ws[8], ws[9])
    assert all(w._group_skips == () for w in ws if w.skip_topk)
    assert heads[0]._group_skips == ()
