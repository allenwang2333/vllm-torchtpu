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
"""Full-size synthetic attention caches for one rank, built on device.

`synth_rank(key, kinds, seq_lens, max_ctx)` returns what the step's attention
operands need, in the served layouts (int32 word views; `served_dtypes=True`
gives the HCA kv / indexer arrays as uint8 slabs) with finite random contents:

  arrays          distinct cache arrays
  cache_index     role -> array index per layer of the role
  cache_layers    role -> decoder layer of each entry
  bt_flat         int32 [*]   all tables (roles of attn_phase_harness.META_ROLES)
  bt_off, bt_w    role -> int32 [L]
  weights         AttnWeights fields
  tables          (role, layer) -> [B, W] numpy, the tables bt_flat holds

Shapes depend only on (kinds, B = len(seq_lens), max_ctx), so ranks stack.
Per layer: CSA one NoPE+state array, a RoPE array, one indexer K+state array
and an SWA array; HCA a kv array, a state array and an SWA array.
"""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP

ROLES = ("swa", "kv", "state", "kv_rope", "indexer_k", "indexer_state")
META_ROLES = ROLES + ("swa_pages",)
Q = 7

# DSv4-Pro decoder layers: ratio 128 -> HCA, 4 -> CSA.
DSV4_KINDS = tuple("hca" if i in (0, 1) or i % 2 == 1 else "csa" for i in range(61))


def _bits(key, shape):
    return jax.lax.bitcast_convert_type(jax.random.bits(key, shape, jnp.uint32), jnp.int32)


def _fp8_words(key, shape):  # four small finite e4m3 codes a word
    return _bits(key, shape) & 0x37373737


def _bf16_words(key, shape):  # two finite bf16 (|x| < 2) a word
    return _bits(key, shape) & np.int32(np.uint32(0xBF7FBF7F).view(np.int32))


def _f32_words(key, shape):
    return jax.lax.bitcast_convert_type(jax.random.normal(key, shape, jnp.float32), jnp.int32)


def _nope_pages(key, n):
    w = _fp8_words(key, (n, AP.CSA_PAGE, 128))
    lane = jnp.arange(128)[None, None, :]
    return jnp.where(lane >= 64, (w & 0x00FFFFFF) | (126 << 24), w)  # e8m0 scales 2^-1


def _idx_pages(key, n):
    w = _fp8_words(key, (n, 64, 256))
    g = jnp.arange(64)[None, :, None]
    return jnp.where(g == 32, 124, w)  # group 32 byte 0: e8m0 scale 2^-3


def synth_rank(key, kinds, seq_lens, max_ctx, *, served_dtypes=False):
    seq_lens = np.asarray(seq_lens, np.int32)
    B = len(seq_lens)
    nl = len(kinds)
    arrays, cidx, clay = [], {r: [] for r in ROLES}, {r: [] for r in ROLES}
    tables = {}
    keys = iter(jax.random.split(key, 8 * nl + 16))

    def add(a):
        arrays.append(a)
        return len(arrays) - 1

    def bind(role, lid, a, table):
        cidx[role].append(a)
        clay[role].append(lid)
        tables[(role, lid)] = np.asarray(table, np.int32)

    swa_w = -(-max_ctx // AP.SWA_PAGE)
    for lid, kind in enumerate(kinds):
        # SWA: 3 pages per slot under its window, page 0 elsewhere.
        st = np.zeros((B, swa_w), np.int32)
        for r in range(B):
            lp = max(int(seq_lens[r]) - Q - 127, 0) // AP.SWA_PAGE
            for i in range(3):
                if lp + i < swa_w:
                    st[r, lp + i] = 1 + 3 * r + i
        bind("swa", lid, add(_bf16_words(next(keys), (3 * B + 1, 2 * AP.SWA_PAGE, 128))), st)
        if kind == "hca":
            kp = -(-(max_ctx // AP.HCA_RATIO + 1) // AP.HCA_PAGE)
            kt = 1 + np.arange(B * kp, dtype=np.int32).reshape(B, kp)
            bind("kv", lid, add(_bf16_words(next(keys), (B * kp + 1, 2 * AP.HCA_PAGE, 128))), kt)
            sp, sbs = -(-max_ctx // AP.SBS["hca"]), AP.SBS["hca"]
            ns = 8
            bind("state", lid, add(_f32_words(next(keys), (B * ns + 1, 256, 128))),
                 _state_table(B, sp, sbs, seq_lens, ns))
        else:
            kp = -(-(max_ctx // AP.CSA_RATIO + 1) // AP.CSA_PAGE)
            nkv = B * kp + 1
            kt = 1 + np.arange(B * kp, dtype=np.int32).reshape(B, kp)
            ns = 12
            arr = jnp.concatenate([_nope_pages(next(keys), nkv),
                                   _f32_words(next(keys), (B * ns, 256, 128))])
            a = add(arr)
            bind("kv", lid, a, kt)
            bind("state", lid, a, nkv + _state_table(B, -(-max_ctx // AP.SBS["csa"]),
                                                     AP.SBS["csa"], seq_lens, ns))
            rope = _bf16_words(next(keys), (nkv + B * ns, AP.CSA_PAGE // 4, 128))
            bind("kv_rope", lid, add(rope), kt)
            iarr = jnp.concatenate([_idx_pages(next(keys), nkv),
                                    _f32_words(next(keys), (B * ns, 64, 256))])
            ia = add(iarr)
            bind("indexer_k", lid, ia, kt)
            bind("indexer_state", lid, ia, nkv + _state_table(
                B, -(-max_ctx // AP.SBS["idx"]), AP.SBS["idx"], seq_lens, ns))
    # Flat tables, one per (role, kind) shared by every layer of it (as serving
    # shares one block table per KV-cache group); the contents are identical
    # across layers of a kind by construction.
    flat, off = [], {r: np.zeros(nl, np.int32) for r in META_ROLES}
    wid = {r: np.ones(nl, np.int32) for r in META_ROLES}
    n = 0
    seen, seen_lid = {}, {}
    for (role, lid), t in sorted(tables.items()):
        key_ = (role, kinds[lid])
        if key_ not in seen:
            seen[key_] = n
            flat.append(t.reshape(-1))
            n += t.size
        else:
            assert np.array_equal(tables[(role, seen_lid[key_])], t)
        seen_lid.setdefault(key_, lid)
        off[role][lid], wid[role][lid] = seen[key_], t.shape[1]
    seen_pg = {}
    for lid in range(nl):
        t = tables[("swa", lid)]  # attn_core.swa_window_pages, in numpy
        lp = np.maximum(seq_lens - Q - 127, 0) // AP.SWA_PAGE
        cols = np.minimum(lp[:, None] + np.arange(3)[None, :], t.shape[1] - 1)
        pg = np.take_along_axis(t, cols, axis=1).reshape(-1)
        k_ = pg.tobytes()
        if k_ not in seen_pg:
            seen_pg[k_] = n
            flat.append(pg)
            n += pg.size
        off["swa_pages"][lid], wid["swa_pages"][lid] = seen_pg[k_], 3
    if served_dtypes:
        u8 = {cidx["kv"][i] for i, l in enumerate(clay["kv"]) if kinds[l] == "hca"}
        u8 |= set(cidx["indexer_k"])
        arrays = [csa_cache_layout.words_to_slabs(a) if i in u8 else a for i, a in enumerate(arrays)]
    ncsa = sum(k == "csa" for k in kinds)
    maxpos = max_ctx + 8
    ang = jnp.arange(maxpos, dtype=jnp.float32)[:, None] * (0.01 * jnp.arange(1, 33))[None, :]
    cs = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=1)
    k = list(jax.random.split(next(keys), 5))
    weights = dict(
        attn_sink=jax.random.normal(k[0], (nl, 128), jnp.float32),
        comp_ape=0.1 * jax.random.normal(k[1], (nl, 128, 1024), jnp.float32),
        comp_norm=1 + 0.1 * jax.random.normal(k[2], (nl, 512), jnp.float32),
        idx_comp_ape=0.1 * jax.random.normal(k[3], (max(ncsa, 1), 4, 256), jnp.float32),
        idx_comp_norm=1 + 0.1 * jax.random.normal(k[4], (max(ncsa, 1), 128), jnp.float32),
        **{f: AP.pack_cos_sin(cs) for f in AP.COS_SIN_FIELDS},
    )
    return dict(arrays=arrays, cache_index=cidx, cache_layers=clay, tables=tables,
                bt_flat=jnp.asarray(np.concatenate(flat)), bt_off={r: jnp.asarray(v) for r, v in off.items()},
                bt_w={r: jnp.asarray(v) for r, v in wid.items()}, weights=weights)


def _state_table(B, width, sbs, seq_lens, ns):
    """State pages: the ~160 positions before each slot's end on distinct pool
    pages [r * ns, (r + 1) * ns), page 0 elsewhere (never read)."""
    t = np.zeros((B, width), np.int32)
    for r in range(B):
        lo = max(int(seq_lens[r]) - 160, 0) // sbs
        hi = min(width, -(-int(seq_lens[r]) // sbs))
        assert hi - lo <= ns, (hi - lo, ns)
        t[r, lo:hi] = r * ns + np.arange(hi - lo)
    return t


def hbm_bytes(d):
    return sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in d["arrays"])


def as_layer_set(d, kinds, seq_lens, max_ctx):
    """The harness view (attn_phase_harness.LayerSet) of a synth_rank result."""
    import attn_phase_harness as H

    return H.LayerSet(tuple(kinds), np.asarray(seq_lens, np.int32), list(d["arrays"]),
                      d["cache_index"], d["cache_layers"], d["tables"],
                      {l: d["tables"][("swa", l)] for l in range(len(kinds))},
                      d["weights"], max_ctx)
