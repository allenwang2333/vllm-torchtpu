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
"""dp step (MKConfig.dp = 4: DP4TP2 attention, EP8 MoE) against the TP8 wide step.

CPU interpret, 8 ranks, a 4-layer hybrid stack (3 GatedDeltaNet + 1 attention) with EP copy
slots (no copies in use) and forced routing. One TP8 model serves 16 sequences (64 verify rows)
in the TP8 wide step; the same model in the dp layout (`dp_weights_from_tp8`) serves them as 4
chips x 4 sequences. Compared: every chip's hidden rows (on both of its cores), the KV pages
the step writes and the GatedDeltaNet checkpoints of every virtual rank. The all-reduce orders
differ (TP8: 8 bf16 partials; dp mixer: 4 virtual partials in f32, then the core pair's f32
sums; dp MoE: 4 bf16 chip partials per core index, then the core pair's f32 sums), hence the
tolerance.

Imports only the kernel modules (no vllm), so it runs with the mkshim sitecustomize.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel.dp_step import (
    dp_weights_from_tp8,
    make_dp_step,
    virtual_ranks,
)

R, DPG, NV, T = 8, 4, 4, 4
NSG = 4  # sequences per chip
S = DPG * NSG
MB = 16
PS = 128
MAXP = 2


def _cfg(**kw):
    base = dict(
        R=R, L=4, att_every=4, H=512, NH=16, NKV=2, HD=256, KH=8, VH=16, KD=128, VD=128,
        E=128, TOPK=2, I=128, IS=1024, ctx=256, T=T, moe_tp=False, stack_e=False, e_slots=20,
        paged_kv=True, fused_pool_io=True, num_pages=448, page_size=PS, max_pages_per_seq=MAXP,
        ckpt_k=T,
    )
    base.update(kw)
    return K.MKConfig(**base)


C_TP8 = _cfg(B=S * T, mb_rows=MB)
C_DP = _cfg(B=NSG * T, mb_rows=MB, dp=DPG)


def _tiles(c):
    return dataclasses.replace(
        K.default_tiles(c), nbd=4, nbe=4, nkv=2, hier=0, rsag=False, kv_blk=256, kv_sub=128
    )


def _slot_tp8(sub, s, t):
    return 1 + (sub * S + s) * T + t


def _slot_dp(sub, v, j, t):
    return 1 + ((sub * NV + v) * NSG + j) * T + t


GDN_SLOTS = 1 + 3 * S * T
KV0 = GDN_SLOTS * C_TP8.gdn_stride


def _model(rng):
    """TP8 per-rank weights [R, ...] of one consistent model: the ranks of a KV-head group share
    the head's k / v projection, scales and k norm; norms, shared-expert gate and router are the
    same on every rank."""
    c = dataclasses.replace(C_TP8, B=MB, mb_rows=0)
    shapes = K.weight_shapes(c, _tiles(C_TP8))
    lay, _ = K.smalls_layout(c)
    G = R // c.NKV
    kv0 = c.nh * 2 * c.HD  # first k lane of the attention in-projection
    w = {}
    for k in K.W_NAMES:
        spec = shapes[k]
        if k in ("e13", "e2"):
            w[k] = tuple(
                jnp.asarray(rng.standard_normal((R, *s_.shape)).astype(np.float32) * 0.05, s_.dtype)
                for s_ in spec
            )
            continue
        s_, dt = spec
        v = rng.standard_normal((R, *s_)).astype(np.float32) * 0.02
        if k == "smalls":
            v[...] = np.abs(v)
            for name in ("in_norm", "post_norm", "sh_gate"):
                o, n = lay[name]
                v[:, :, :, o:o + n] = v[0:1, :, :, o:o + n]
            o, n = lay["ecopy"]
            v[:, :, :, o:o + n] = 0.0
            o, n = lay["alog"]
            v[:, :, :, o:o + 128] = -1.0
            for r in range(R):
                g0 = (r // G) * G
                o, _ = lay["in_s"]
                v[r, :, :, o + kv0:o + kv0 + 2 * c.HD] = v[g0, :, :, o + kv0:o + kv0 + 2 * c.HD]
                o, n = lay["kn"]
                v[r, :, :, o:o + n] = v[g0, :, :, o:o + n]
        elif k == "att_in":
            for r in range(R):
                g0 = (r // G) * G
                v[r, ..., kv0:kv0 + 2 * c.HD] = v[g0, ..., kv0:kv0 + 2 * c.HD]
        elif k == "router":
            v[:] = v[0:1]
        elif k == "final_norm":
            v[:] = np.abs(v[0:1])
        elif k == "es":
            v = rng.uniform(0.5, 1.5, (R, *s_)).astype(np.float32)
        w[k] = jnp.asarray(v, dt) if k not in ("e13", "e2") else v
    return w


def _unpack_rec(raw_pages_u8, vh, kd=128, vd=128):
    """(rec_pages, 2, 64, 4, 128) uint8 -> (vh, kd, vd) f32 (qwen35_megakernel_test_utils)."""
    rows = (vh * kd) // 2
    n = raw_pages_u8.shape[0]
    u32 = np.swapaxes(raw_pages_u8, -2, -1).copy().view(np.uint32).reshape(-1, n * 128, 128)
    bf = np.swapaxes(u32[:, :rows, :][..., None].view(jnp.dtype(K.BF16)), -2, -1)
    return bf.reshape(-1, vh, kd, vd).astype(np.float32)


def _unpack_conv(raw_page_u8, qkv):
    """(..., 2, 64, 4, 128) uint8 -> (..., 3, qkv) f32 (qwen35_megakernel_test_utils)."""
    rows = (3 * qkv) // 128
    bf = np.swapaxes(np.swapaxes(raw_page_u8, -2, -1).copy().view(jnp.dtype(K.BF16)), -2, -1)
    return bf.reshape(-1, 256, 128)[:, :rows, :].reshape(-1, 3, qkv).astype(np.float32)


def _shard(mesh, x):
    return jax.device_put(x, NamedSharding(mesh, P("tp")))


def _rep(mesh, x):
    return jax.device_put(x, NamedSharding(mesh, P()))


@pytest.mark.parametrize("live", [NSG, NSG - 1])
def test_dp_step_matches_tp8_wide(live):
    """live: sequences per chip; below NSG the chip's last grid sequences are inactive padding
    (checkpoint 0, position 0), as when a chip's rows do not fill the microbatch grid."""
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip("needs 8 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    S_ = DPG * live  # sequences served
    Bg = live * T  # live rows per chip
    BWg = NSG * T  # the chip's row grid (one microbatch)
    c8 = _cfg(B=S_ * T, mb_rows=MB)
    cd = _cfg(B=Bg, mb_rows=MB, dp=DPG)
    rng = np.random.default_rng(7)
    w_tp8 = _model(rng)
    w_dp = dp_weights_from_tp8(cd, {k: (tuple(np.asarray(x) for x in v) if isinstance(v, tuple)
                                        else np.asarray(v)) for k, v in w_tp8.items()})
    shard_w = lambda w: {k: (tuple(_shard(mesh, jnp.asarray(x)) for x in v) if isinstance(v, tuple)
                             else _shard(mesh, jnp.asarray(v))) for k, v in w.items()}

    # ---- sequences, pages (sequence s = g * live + j is chip g's j-th) ----
    ctx_len = rng.integers(130, 240, S_)
    pos = np.concatenate([ctx_len[s] + np.arange(T) for s in range(S_)]).astype(np.int32)
    pt = np.zeros((S_, MAXP), np.int32)
    for s in range(S_):
        pt[s] = KV0 + s * MAXP + np.arange(MAXP)
    n_pages = c8.num_pages
    assert KV0 + S * MAXP <= n_pages
    ck_tp8 = np.array([[[_slot_tp8(sub, s, t) for t in range(T)] for s in range(S_)]
                       for sub in range(3)], np.int32)
    roff = np.zeros((3, S_), np.int32)
    pools_tp8 = rng.integers(0, 40, (R, n_pages, 2, 64, 4, PS), dtype=np.uint8)
    G = R // c8.NKV
    for r in range(R):  # the KV pages of a head are the same on every rank of its group
        g0 = (r // G) * G
        pools_tp8[r, KV0:] = pools_tp8[g0, KV0:]
    gp = c8.gdn_pages
    stride = c8.gdn_stride
    pools_dp = np.zeros_like(pools_tp8)
    ck_dp = np.zeros((R, 3, NV * NSG, T), np.int32)
    for r in range(R):
        g, core = r // 2, r % 2
        pools_dp[r, KV0:] = pools_tp8[core * G, KV0:]
        for v, q in enumerate(virtual_ranks(r, NV)):
            for sub in range(3):
                for j in range(live):
                    s = g * live + j
                    for t in range(T):
                        a, b = _slot_tp8(sub, s, t) * stride, _slot_dp(sub, v, j, t) * stride
                        pools_dp[r, b:b + gp] = pools_tp8[q, a:a + gp]
                        ck_dp[r, sub, v * NSG + j, t] = _slot_dp(sub, v, j, t)

    hid_rows = (rng.standard_normal((S_ * T, c8.H)) * 0.1).astype(np.float32)
    # forced routing: every row's TOPK picks; dp: chip g's rows at g * BWg
    m_rb = 128
    rb = np.full((c8.L, m_rb, c8.E), 1e-3, np.float32)
    rb += np.arange(c8.E, dtype=np.float32)[None, None, :] * 1e-6
    rb_dp = rb.copy()
    for l in range(c8.L):
        for b in range(S_ * T):
            picks = rng.choice(c8.E, c8.TOPK, replace=False)
            rb[l, b, picks] = 2.0 - 0.01 * np.arange(c8.TOPK)
            g, i = divmod(b, Bg)
            rb_dp[l, g * BWg + i, picks] = 2.0 - 0.01 * np.arange(c8.TOPK)
    to_raw = lambda p: (_shard(mesh, jax.lax.bitcast_convert_type(
        jnp.asarray(p.reshape(R * n_pages, 2, 64, 4, PS)), K.FP8)),)

    # ---- TP8 wide reference ----
    step8, _ = pool_io.make_pooled_step(mesh, c8, _tiles(c8), interpret=True,
                                        forced_routing=True, fused_pool_io=True)
    hid8 = np.zeros((c8.M, c8.H), np.float32)
    hid8[:S_ * T] = hid_rows
    out8, raw8 = step8(
        _rep(mesh, jnp.asarray(pos)), _shard(mesh, jnp.asarray(np.tile(hid8, (R, 1)), K.BF16)),
        shard_w(w_tp8), to_raw(pools_tp8), _rep(mesh, jnp.asarray(pt)),
        _rep(mesh, jnp.asarray(ck_tp8)), _rep(mesh, jnp.asarray(roff)),
        _shard(mesh, jnp.asarray(np.tile(rb, (R, 1, 1)))),
    )
    h8 = np.asarray(out8.astype(jnp.float32)).reshape(R, c8.M, c8.H)
    p8 = np.asarray(jax.lax.bitcast_convert_type(raw8[0], jnp.uint8)).reshape(pools_tp8.shape)

    # ---- dp (operands on the chip's grid of NSG sequences) ----
    stepd, _ = make_dp_step(mesh, cd, _tiles(cd), interpret=True, forced_routing=True)
    hidd = np.zeros((R, cd.M, cd.H), np.float32)
    posd = np.zeros((R, BWg), np.int32)
    ptd = np.zeros((R, NSG, MAXP), np.int32)
    rod = np.zeros((R, 3, NSG), np.int32)
    for r in range(R):
        g = r // 2
        hidd[r, :Bg] = hid_rows[g * Bg:(g + 1) * Bg]
        posd[r, :Bg] = pos[g * Bg:(g + 1) * Bg]
        ptd[r, :live] = pt[g * live:(g + 1) * live]
    outd, rawd = stepd(
        _shard(mesh, jnp.asarray(posd)),
        _shard(mesh, jnp.asarray(hidd.reshape(R * cd.M, cd.H), K.BF16)),
        shard_w(w_dp), to_raw(pools_dp), _shard(mesh, jnp.asarray(ptd)),
        _shard(mesh, jnp.asarray(ck_dp)), _shard(mesh, jnp.asarray(rod)),
        _shard(mesh, jnp.asarray(np.tile(rb_dp, (R, 1, 1)))),
    )
    hd = np.asarray(outd.astype(jnp.float32)).reshape(R, cd.M, cd.H)
    pd = np.asarray(jax.lax.bitcast_convert_type(rawd[0], jnp.uint8)).reshape(pools_dp.shape)

    # ---- compare ----
    assert np.isfinite(h8).all() and np.isfinite(hd[:, :Bg]).all()
    for r in range(1, R):
        assert np.array_equal(h8[r, :S_ * T], h8[0, :S_ * T])
    for g in range(DPG):
        assert np.array_equal(hd[2 * g, :Bg], hd[2 * g + 1, :Bg]), f"chip {g}: cores differ"
    ref = np.concatenate([h8[0, g * Bg:(g + 1) * Bg] for g in range(DPG)])
    got = np.concatenate([hd[2 * g, :Bg] for g in range(DPG)])
    close = np.isclose(got, ref, atol=2e-2, rtol=2e-2)
    print("live", live, "hidden close frac", float(close.mean()), "max abs diff",
          float(np.abs(got - ref).max()), "ref max", float(np.abs(ref).max()))
    assert close.mean() > 0.999 and np.abs(got - ref).max() < 0.1
    # the inactive grid sequences wrote nothing: page 0 (the null page) is untouched
    assert np.array_equal(pd[:, 0], pools_dp[:, 0])

    kvf = lambda a: a.view(jnp.dtype(K.FP8)).astype(np.float32)
    for r in range(R):
        g, core = r // 2, r % 2
        pages = pt[g * live:(g + 1) * live].reshape(-1)
        np.testing.assert_allclose(kvf(pd[r, pages]), kvf(p8[core * G, pages]), atol=0.25, rtol=0.13)
        for v, q in enumerate(virtual_ranks(r, NV)):
            for sub in range(3):
                for j in range(live):
                    s = g * live + j
                    for t in range(T):
                        a, b = _slot_tp8(sub, s, t) * stride, _slot_dp(sub, v, j, t) * stride
                        msg = f"rank {r} vr {v} sub {sub} seq {s} token {t}"
                        rp = c8.rec_pages
                        np.testing.assert_allclose(
                            _unpack_rec(pd[r, b:b + rp], c8.vh),
                            _unpack_rec(p8[q, a:a + rp], c8.vh), atol=2e-2, rtol=2e-2,
                            err_msg="rec " + msg)
                        np.testing.assert_allclose(
                            _unpack_conv(pd[r, b + rp], c8.QKV),
                            _unpack_conv(p8[q, a + rp], c8.QKV), atol=2e-2, rtol=2e-2,
                            err_msg="conv " + msg)
