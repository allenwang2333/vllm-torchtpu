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
"""GDN v3 in-chunk state checkpoints ("virtual segments") vs the multi-step path.

One `fused_conv1d_gdn` call with `seg_bounds` / `seg_state_indices` must store
the same checkpoint states, final state and outputs as running each segment
as its own prefill step (state block seeded from the previous one), which is
what the Mamba "align" split does today.

On CPU the Pallas kernel runs in TPU interpret mode. Two strided-ref helpers
of `vmem_ldst` are swapped for value-level equivalents because the
interpreter has no ref-reshape transform; both arms of every comparison use
the same helpers. On a TPU backend the kernel runs natively and unpatched.
"""

import functools
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """CPU hosts without torch_tpu: `vllm_torchtpu.layers` imports it at
    module scope, but nothing on the JAX kernel path below touches it."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch_tpu" or name.startswith("torch_tpu."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        m.__spec__ = spec
        return m

    def exec_module(self, module):
        pass


if importlib.util.find_spec("torch_tpu") is None:
    sys.meta_path.insert(0, _StubFinder())
    import torch._dynamo

    if "tpu" not in torch._dynamo.list_backends(exclude_tags=()):
        torch._dynamo.register_backend(name="tpu")(lambda gm, _: gm.forward)

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import vmem_ldst, wrapper
from vllm_torchtpu.layers.core import gdn_attention

N_KQ, N_V, D_K, D_V, K = 2, 4, 128, 128, 4
DIM = 2 * N_KQ * D_K + N_V * D_V
TILE = 128
NUM_SPEC = 3


@pytest.fixture(autouse=True)
def _cpu_interpret(monkeypatch):
    if jax.default_backend() != "cpu":
        yield
        return
    from jax._src import tpu_info

    info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    orig = pl.pallas_call

    def pallas_call(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams()
        return orig(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", pallas_call)

    def load_strided_heads(vmem_ref, num_heads, head_dim, lane_offset=0):
        lanes = info.num_lanes
        lanes_per_col = vmem_ref.shape[-1] // lanes
        lanes_per_head = head_dim // lanes
        flat = vmem_ref[...].reshape(-1, lanes)
        return jnp.stack(
            [
                jnp.concat(
                    [
                        flat[lane_offset + h * lanes_per_head + l :: lanes_per_col]
                        for l in range(lanes_per_head)
                    ],
                    axis=-1,
                )
                for h in range(num_heads)
            ],
            axis=0,
        )

    def load_compact_to_large(vmem_ref):
        return vmem_ref[...].reshape(vmem_ref.shape[:-2] + (vmem_ref.shape[-1],))

    monkeypatch.setattr(vmem_ldst, "load_strided_heads", load_strided_heads)
    monkeypatch.setattr(vmem_ldst, "load_compact_to_large", load_compact_to_large)

    # Typed views of raw pool bytes: Mosaic `ref.bitcast` + ref reshape have
    # no interpret rule. Use a linear-byte view instead; it is a bijection,
    # so store/load round-trip exactly and both arms see the same layout.
    def _view(raw, view_dtype, lane_split):
        src = jnp.dtype(raw.dtype)
        view_dtype = jnp.dtype(view_dtype)
        lanes = raw.shape[-1] // lane_split
        flat = raw.reshape(-1)
        if src != view_dtype:
            ratio = view_dtype.itemsize // src.itemsize
            if ratio >= 1:
                flat = jax.lax.bitcast_convert_type(flat.reshape(-1, ratio), view_dtype)
            else:
                flat = jax.lax.bitcast_convert_type(flat, view_dtype).reshape(-1)
        return flat.reshape(-1, lanes)

    def _unview(arr, like):
        src = jnp.dtype(like.dtype)
        arr = arr.reshape(-1)
        if jnp.dtype(arr.dtype) != src:
            ratio = jnp.dtype(arr.dtype).itemsize // src.itemsize
            if ratio >= 1:
                arr = jax.lax.bitcast_convert_type(arr, src)
            else:
                arr = jax.lax.bitcast_convert_type(
                    arr.reshape(-1, src.itemsize // jnp.dtype(arr.dtype).itemsize), src
                )
        return arr.reshape(like.shape)

    def load_typed(block_ref, *, view_dtype, lane_split=1, row0=0, rows=None):
        arr = _view(block_ref[...], view_dtype, lane_split)
        return arr if rows is None else arr[row0 : row0 + rows]

    def store_typed(block_ref, values, *, lane_split=1, row0=0, rows=None):
        lanes = block_ref.shape[-1] // lane_split
        bytes_per_typed_row = lanes * jnp.dtype(values.dtype).itemsize
        bytes_per_raw_row = int(np.prod(block_ref.shape[1:])) * jnp.dtype(
            block_ref.dtype
        ).itemsize
        if rows is None:
            rows = block_ref.shape[0] * bytes_per_raw_row // bytes_per_typed_row
        lo, hi = row0 * bytes_per_typed_row, (row0 + rows) * bytes_per_typed_row
        assert lo % bytes_per_raw_row == 0 and hi % bytes_per_raw_row == 0
        r0, r1 = lo // bytes_per_raw_row, hi // bytes_per_raw_row
        like = jax.ShapeDtypeStruct((r1 - r0,) + block_ref.shape[1:], block_ref.dtype)
        block_ref[r0:r1] = _unview(values.reshape(rows, lanes), like)

    from vllm_torchtpu.kernels import typed_ldst

    monkeypatch.setattr(typed_ldst, "load_typed", load_typed)
    monkeypatch.setattr(typed_ldst, "store_typed", store_typed)
    yield


# (hit tokens, query tokens, cut1, cut2): cut offsets inside the query, 0 = none.
# A: two cuts (f640 + f128 shape); B: one cut, non-final segment ends mid-tile;
# C: cold, no cut; D: only the second cut (segment 0 empty).
SEQS = (
    (512, 400, 128, 384),
    (300, 333, 200, 0),
    (0, 150, 0, 0),
    (896, 260, 0, 130),
)


def _segments(q, c1, c2):
    cuts = [c for c in (c1, c2) if 0 < c < q]
    bounds = [0] + cuts + [q]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    total = sum(s[1] for s in SEQS)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    return dict(
        qkv=jnp.asarray(f(total, DIM)).astype(jnp.bfloat16),
        b=jnp.asarray(f(total, N_V)).astype(jnp.bfloat16),
        a=jnp.asarray(f(total, N_V)).astype(jnp.bfloat16),
        w=jnp.asarray(f(DIM, 1, K) * 0.3),
        a_log=jnp.asarray(f(N_V) * 0.1),
        dt_bias=jnp.asarray(f(N_V) * 0.1),
        conv0=jnp.asarray(f(len(SEQS), K - 1, DIM)).astype(jnp.bfloat16),
        rec0=jnp.asarray(f(len(SEQS), N_V, D_K, D_V) * 0.1),
    )


def _pad_rows(x, rows):
    return jnp.pad(x, ((0, rows - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))


# Slot plan per sequence s: init 10+s, seg-0 write 20+s, seg-1 write 30+s,
# running/final 40+s. Slot 0 is the null slot.
def _slots(s):
    return dict(init=10 + s, w0=20 + s, w1=30 + s, run=40 + s)


NUM_SLOTS = 48


# ----------------------------------------------------------------- dense arms
def _dense_call(x, conv, rec, rows, qsl, si, sl, ro, padded, seg=None):
    num = len(si)
    kwargs = {}
    if seg is not None:
        kwargs = dict(
            seg_bounds=jnp.asarray(seg[0], jnp.int32),
            seg_state_indices=jnp.asarray(seg[1], jnp.int32),
        )
    (conv, rec), out = wrapper.fused_conv1d_gdn(
        _pad_rows(x["qkv"][rows], padded),
        _pad_rows(x["b"][rows], padded),
        _pad_rows(x["a"][rows], padded),
        conv,
        rec,
        x["w"],
        None,
        x["a_log"],
        x["dt_bias"],
        jnp.asarray(qsl, jnp.int32),
        jnp.asarray(si, jnp.int32),
        jnp.asarray([0, num, num], jnp.int32),
        jnp.asarray(sl, jnp.int32),
        jnp.asarray(ro, jnp.int32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=K,
        prefill_only=True,
        mixed_tile_size=TILE,
        **kwargs,
    )
    return conv, rec, out


def _dense_init(x, rec_dtype):
    conv = jnp.zeros((NUM_SLOTS, K - 1, DIM), jnp.float32)
    rec = jnp.zeros((NUM_SLOTS, N_V, D_K, D_V), rec_dtype)
    for s in range(len(SEQS)):
        conv = conv.at[_slots(s)["init"]].set(x["conv0"][s].astype(jnp.float32))
        rec = rec.at[_slots(s)["init"]].set(x["rec0"][s].astype(rec_dtype))
    return conv, rec


def _row_offsets():
    return np.cumsum([0] + [s[1] for s in SEQS])


def run_dense_multistep(x, rec_dtype, padded):
    """Reference: segment j of every sequence runs in step j, each in its own
    call; the destination slot is seeded from the previous one first."""
    conv, rec = _dense_init(x, rec_dtype)
    offs = _row_offsets()
    out = np.zeros((offs[-1], N_V * D_V), np.float32)
    segs = [_segments(q, c1, c2) for _, q, c1, c2 in SEQS]
    for step in range(3):
        rows, qsl, si, sl, plan = [], [0], [], [], []
        for s, (h, _, _, _) in enumerate(SEQS):
            if step >= len(segs[s]):
                continue
            lo, hi = segs[s][step]
            sl_ = _slots(s)
            dests = [sl_["w0"], sl_["w1"]][: len(segs[s]) - 1] + [sl_["run"]]
            if len(segs[s]) == 2 and SEQS[s][2] == 0:
                dests = [sl_["w1"], sl_["run"]]
            src = sl_["init"] if step == 0 else dests[step - 1]
            dst = dests[step]
            conv = conv.at[dst].set(conv[src])
            rec = rec.at[dst].set(rec[src])
            rows.extend(range(offs[s] + lo, offs[s] + hi))
            qsl.append(qsl[-1] + hi - lo)
            si.append(dst)
            sl.append(h + hi)
            plan.append((s, lo, hi))
        conv, rec, o = _dense_call(
            x, conv, rec, np.asarray(rows), qsl, si, sl, [0] * len(si), padded
        )
        o = np.asarray(o.astype(jnp.float32))
        for i, (s, lo, hi) in enumerate(plan):
            out[offs[s] + lo : offs[s] + hi] = o[qsl[i] : qsl[i + 1]]
    return conv, rec, out


def run_dense_segments(x, rec_dtype, padded):
    conv, rec = _dense_init(x, rec_dtype)
    offs = _row_offsets()
    si, sl, bounds, seg_si = [], [], [], []
    for s, (h, q, c1, c2) in enumerate(SEQS):
        sl_ = _slots(s)
        conv = conv.at[sl_["run"]].set(conv[sl_["init"]])
        rec = rec.at[sl_["run"]].set(rec[sl_["init"]])
        si.append(sl_["run"])
        sl.append(h + q)
        bounds.append((c1, c2))
        seg_si.append((sl_["w0"], sl_["w1"]))
    conv, rec, out = _dense_call(
        x, conv, rec, np.arange(offs[-1]), list(offs), si, sl, [0] * len(si),
        padded, seg=(bounds, seg_si),
    )
    return conv, rec, np.asarray(out.astype(jnp.float32))[: offs[-1]]


def _written_slots():
    out = []
    for s, (_, q, c1, c2) in enumerate(SEQS):
        sl_ = _slots(s)
        if 0 < c1 < q:
            out.append(sl_["w0"])
        if 0 < c2 < q:
            out.append(sl_["w1"])
        out.append(sl_["run"])
    return out


PADDED = 1152


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint16)


def test_dense_fp32_store_bit_identical():
    x = _inputs(0)
    ref = run_dense_multistep(x, jnp.float32, PADDED)
    seg = run_dense_segments(x, jnp.float32, PADDED)
    for slot in _written_slots():
        np.testing.assert_array_equal(_bits(ref[0][slot]), _bits(seg[0][slot]))
        np.testing.assert_array_equal(_bits(ref[1][slot]), _bits(seg[1][slot]))
    np.testing.assert_array_equal(_bits(ref[2]), _bits(seg[2]))
    # Unrelated slots untouched.
    untouched = sorted(set(range(NUM_SLOTS)) - set(_written_slots()))
    np.testing.assert_array_equal(
        _bits(seg[1][np.asarray(untouched)]),
        _bits(_dense_init(x, jnp.float32)[1][np.asarray(untouched)]),
    )


def test_segments_without_cuts_match_plain_call():
    """All-zero seg_bounds reproduce the unsegmented kernel exactly."""
    x = _inputs(1)
    offs = _row_offsets()
    si = [_slots(s)["init"] for s in range(len(SEQS))]
    sl = [h + q for h, q, _, _ in SEQS]
    zeros = np.zeros((len(SEQS), 2), np.int32)
    conv, rec = _dense_init(x, jnp.float32)  # donated by the call
    a = _dense_call(x, conv, rec, np.arange(offs[-1]), list(offs), si, sl,
                    [0] * len(si), PADDED)
    conv, rec = _dense_init(x, jnp.float32)
    b = _dense_call(x, conv, rec, np.arange(offs[-1]), list(offs), si, sl,
                    [0] * len(si), PADDED, seg=(zeros, zeros))
    for i in range(3):
        np.testing.assert_array_equal(_bits(a[i]), _bits(b[i]))


def test_dense_segments_behind_a_decode_sequence():
    """Mixed batch: a 1-token decode (BATCHED kernel) ahead of the prefills,
    so the PER_SEQ metadata starts at start_seq = 1. Prefill states/outputs
    must equal the prefill-only segmented call; the decode must equal the
    same batch without segments."""
    x = _inputs(3)
    offs = _row_offsets()
    dec_slot, dec_tok = 5, jnp.asarray(np.random.default_rng(9).standard_normal((1, DIM)), jnp.float32).astype(jnp.bfloat16)

    def mixed(seg):
        conv, rec = _dense_init(x, jnp.float32)
        conv = conv.at[dec_slot].set(conv[_slots(0)["init"]])
        rec = rec.at[dec_slot].set(rec[_slots(1)["init"]])
        si, sl, bounds, seg_si = [dec_slot], [777], [(0, 0)], [(0, 0)]
        for s_, (h, q, c1, c2) in enumerate(SEQS):
            sl_ = _slots(s_)
            conv = conv.at[sl_["run"]].set(conv[sl_["init"]])
            rec = rec.at[sl_["run"]].set(rec[sl_["init"]])
            si.append(sl_["run"])
            sl.append(h + q)
            bounds.append((c1, c2))
            seg_si.append((sl_["w0"], sl_["w1"]))
        xx = dict(x, qkv=jnp.concatenate([dec_tok, x["qkv"]]),
                  b=jnp.concatenate([x["b"][:1], x["b"]]),
                  a=jnp.concatenate([x["a"][:1], x["a"]]))
        n = len(si)
        (conv, rec), out = wrapper.fused_conv1d_gdn(
            _pad_rows(xx["qkv"], PADDED + 16), _pad_rows(xx["b"], PADDED + 16),
            _pad_rows(xx["a"], PADDED + 16), conv, rec, x["w"], None, x["a_log"],
            x["dt_bias"], jnp.asarray([0] + [1 + o for o in offs], jnp.int32),
            jnp.asarray(si, jnp.int32), jnp.asarray([1, n, n], jnp.int32),
            jnp.asarray(sl, jnp.int32), jnp.zeros((n,), jnp.int32),
            n_kq=N_KQ, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=K, mixed_tile_size=TILE,
            **({} if not seg else dict(
                seg_bounds=jnp.asarray(bounds, jnp.int32),
                seg_state_indices=jnp.asarray(seg_si, jnp.int32))),
        )
        return conv, rec, np.asarray(out.astype(jnp.float32))

    seg_only = run_dense_segments(x, jnp.float32, PADDED)
    m_seg = mixed(True)
    m_plain = mixed(False)
    for slot in _written_slots():
        np.testing.assert_array_equal(_bits(m_seg[1][slot]), _bits(seg_only[1][slot]))
        np.testing.assert_array_equal(_bits(m_seg[0][slot]), _bits(seg_only[0][slot]))
    np.testing.assert_array_equal(_bits(m_seg[2][1 : 1 + offs[-1]]), _bits(seg_only[2]))
    np.testing.assert_array_equal(_bits(m_seg[1][dec_slot]), _bits(m_plain[1][dec_slot]))
    np.testing.assert_array_equal(_bits(m_seg[2][0]), _bits(m_plain[2][0]))


# ---------------------------------------------------------------- pooled arms
POOL_KBS = 256
POOL_LANES = 256
POOL_PAYLOAD = (1, 4)
POOL_BLOCK_TOKENS = 512  # manager block = 2 kernel blocks


def _pooled_call(x, pool, rows, qsl, si, sl, ro, ckpt, padded, rec_dtype, seg=None):
    kwargs = {}
    if seg is not None:
        kwargs = dict(
            seg_bounds=jnp.asarray(seg[0], jnp.int32),
            seg_state_indices=jnp.asarray(seg[1], jnp.int32),
        )
    num = len(si)
    pool, out = gdn_attention.run_jax_gdn_attention_pooled_local(
        _pad_rows(x["qkv"][rows], padded),
        _pad_rows(x["b"][rows], padded),
        _pad_rows(x["a"][rows], padded),
        pool,
        x["w"],
        None,
        x["a_log"],
        x["dt_bias"],
        jnp.asarray(qsl, jnp.int32),
        jnp.asarray(si, jnp.int32),
        jnp.asarray([0, num, num], jnp.int32),
        jnp.asarray(sl, jnp.int32),
        jnp.asarray(ro, jnp.int32),
        jnp.asarray(ckpt, jnp.int32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=K,
        pool_block_tokens=POOL_BLOCK_TOKENS,
        recurrent_state_dtype=rec_dtype,
        num_spec_tokens=NUM_SPEC,
        **kwargs,
    )
    return pool, out


def _spec_blocks(s):
    # Speculative checkpoint blocks: PER_SEQ never writes them and, with read
    # offset 0, never reads them.
    del s
    return [44, 45, 46]


def _pool_seed(pool, src, dst):
    """Manager-block seed copy: a manager block is `split` kernel blocks
    (TPUModelRunner._expand_pool_split)."""
    split = POOL_BLOCK_TOKENS // POOL_KBS
    return pool_adapters.copy_blocks(
        pool,
        jnp.asarray([src * split + j for j in range(split)], jnp.int32),
        jnp.asarray([dst * split + j for j in range(split)], jnp.int32),
    )


def _pool_init(x, rec_dtype):
    """Pool of random bytes whose init blocks hold real states."""
    pool = jnp.zeros((NUM_SLOTS * 2, POOL_KBS) + POOL_PAYLOAD + (POOL_LANES,), jnp.int8)
    # Bring the init blocks to a valid state encoding: run a cold 64-token
    # prefill into each init block (the stored state is then a real state in
    # the pool's encoding).
    n = len(SEQS)
    warm = jax.random.normal(jax.random.PRNGKey(3), (64 * n, DIM)).astype(jnp.bfloat16)
    wb = jax.random.normal(jax.random.PRNGKey(4), (64 * n, N_V)).astype(jnp.bfloat16)
    xx = dict(x, qkv=warm, b=wb, a=wb)
    si = [_slots(s)["init"] for s in range(n)]
    ckpt = [[si[s]] + [0] * NUM_SPEC for s in range(n)]
    pool, _ = _pooled_call(
        xx, pool, np.arange(64 * n), list(range(0, 64 * n + 1, 64)), si,
        [64] * n, [0] * n, ckpt, 256, rec_dtype,
    )
    return pool


def run_pooled_multistep(x, rec_dtype, padded):
    pool = _pool_init(x, rec_dtype)
    offs = _row_offsets()
    out = np.zeros((offs[-1], N_V * D_V), np.float32)
    segs = [_segments(q, c1, c2) for _, q, c1, c2 in SEQS]
    for step in range(3):
        rows, qsl, si, sl, ckpt, plan = [], [0], [], [], [], []
        for s, (h, _, _, _) in enumerate(SEQS):
            if step >= len(segs[s]):
                continue
            lo, hi = segs[s][step]
            sl_ = _slots(s)
            dests = [sl_["w0"], sl_["w1"]][: len(segs[s]) - 1] + [sl_["run"]]
            if len(segs[s]) == 2 and SEQS[s][2] == 0:
                dests = [sl_["w1"], sl_["run"]]
            src = sl_["init"] if step == 0 else dests[step - 1]
            dst = dests[step]
            pool = _pool_seed(pool, src, dst)
            rows.extend(range(offs[s] + lo, offs[s] + hi))
            qsl.append(qsl[-1] + hi - lo)
            si.append(dst)
            sl.append(h + hi)
            ckpt.append([dst] + _spec_blocks(s))
            plan.append((s, lo, hi))
        pool, o = _pooled_call(
            x, pool, np.asarray(rows), qsl, si, sl, [0] * len(si), ckpt, padded,
            rec_dtype,
        )
        o = np.asarray(o.astype(jnp.float32))
        for i, (s, lo, hi) in enumerate(plan):
            out[offs[s] + lo : offs[s] + hi] = o[qsl[i] : qsl[i + 1]]
    return pool, out


def run_pooled_segments(x, rec_dtype, padded):
    pool = _pool_init(x, rec_dtype)
    offs = _row_offsets()
    si, sl, bounds, seg_si, ckpt = [], [], [], [], []
    for s, (h, q, c1, c2) in enumerate(SEQS):
        sl_ = _slots(s)
        pool = _pool_seed(pool, sl_["init"], sl_["run"])
        si.append(sl_["run"])
        sl.append(h + q)
        bounds.append((c1, c2))
        seg_si.append((sl_["w0"], sl_["w1"]))
        ckpt.append([sl_["run"]] + _spec_blocks(s))
    pool, out = _pooled_call(
        x, pool, np.arange(offs[-1]), list(offs), si, sl, [0] * len(si), ckpt,
        padded, rec_dtype, seg=(bounds, seg_si),
    )
    return pool, np.asarray(out.astype(jnp.float32))[: offs[-1]]


def _block_rows(pool, mgr):
    return np.asarray(pool[2 * mgr : 2 * mgr + 2])


def _read_rec(pool, mgr, rec_dtype):
    """Decode manager block `mgr`'s recurrent state with the kernel's own
    reader (`vmem_ldst.load_state_region` through the production plan's
    region), so the typed view matches what the kernel stored: on TPU the
    bf16 state is packed as row pairs in uint32 (Mosaic bitcast), which a
    linear byte view of the pool does not decode."""
    plan = gdn_attention._build_v3_pool_state_plan(
        pool, conv_dim=DIM, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=K,
        pool_block_tokens=POOL_BLOCK_TOKENS, qk_pair_layout=False,
        recurrent_state_dtype=rec_dtype,
    )
    region = plan.recurrent
    split = POOL_BLOCK_TOKENS // POOL_KBS
    base = mgr * split + region.kb0
    tile = pool[base : base + region.nblocks, region.row0 : region.row0 + region.nrows]

    def kernel(src_ref, out_ref):
        out_ref[...] = vmem_ldst.load_state_region(
            src_ref, region, (N_V, D_K, D_V)
        ).astype(jnp.float32)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N_V, D_K, D_V), jnp.float32),
    )(tile)
    return np.asarray(out)


@pytest.mark.parametrize("rec_dtype", [jnp.float32, jnp.bfloat16])
def test_pooled_checkpoints(rec_dtype, record_property):
    x = _inputs(2)
    ref_pool, ref_out = run_pooled_multistep(x, rec_dtype, PADDED)
    seg_pool, seg_out = run_pooled_segments(x, rec_dtype, PADDED)
    first_ckpt = set()
    for s, (_, q, c1, c2) in enumerate(SEQS):
        first = [b for b, c in ((_slots(s)["w0"], c1), (_slots(s)["w1"], c2)) if 0 < c < q]
        first_ckpt.add((first or [_slots(s)["run"]])[0])
    written = _written_slots()
    if rec_dtype == jnp.float32:
        for mgr in written:
            np.testing.assert_array_equal(_block_rows(ref_pool, mgr), _block_rows(seg_pool, mgr))
        np.testing.assert_array_equal(_bits(ref_out), _bits(seg_out))
    else:
        # The first stored state of each sequence is computed from the same
        # seeded block in both arms: identical bytes. Later states differ only
        # because the multi-step path rounds the state to bf16 at each step
        # boundary while the single step carries it in fp32.
        diffs = {}
        for mgr in written:
            same = np.array_equal(_block_rows(ref_pool, mgr), _block_rows(seg_pool, mgr))
            if mgr in first_ckpt:
                assert same, mgr
            a = _read_rec(ref_pool, mgr, rec_dtype)
            b = _read_rec(seg_pool, mgr, rec_dtype)
            # Written states must be finite in both arms.
            assert np.isfinite(a).all() and np.isfinite(b).all(), mgr
            diffs[mgr] = (same, float(np.max(np.abs(a - b))), float(np.max(np.abs(b))))
        assert np.isfinite(ref_out).all() and np.isfinite(seg_out).all()
        out_diff = float(np.max(np.abs(ref_out - seg_out)))
        out_scale = float(np.max(np.abs(seg_out)))
        record_property("bf16_state_diffs", diffs)
        record_property("bf16_out_maxabs", (out_diff, out_scale))
        print("bf16 state diffs (same_bytes, max|d|, max|x|):", diffs)
        print("bf16 output max|d| / max|x|:", out_diff, out_scale)
        for same, d, scale in diffs.values():
            assert d <= 2.0**-6 * max(scale, 1.0)
        assert out_diff <= 2.0**-5 * max(out_scale, 1.0)
