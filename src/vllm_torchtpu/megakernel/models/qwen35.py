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
"""Qwen3.5-397B-A17B geometry, weight packer, logical reference, and step builders."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.blocks.gqa import GQABlock
from vllm_torchtpu.megakernel.blocks.moe import MoEBlock
from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    rnd,
    rope_partial,
)
from vllm_torchtpu.megakernel.spec import (
    Block,
    ExecSpec,
    LayerType,
    ModelSpec,
    SwitchSlot,
)


@dataclass(frozen=True)
class Qwen35Geometry:
    """Pure model architecture geometry for Qwen3.5-397B-A17B."""

    L: int = 60
    att_every: int = 4
    H: int = 4096
    NH: int = 32
    NKV: int = 2
    HD: int = 256
    ROT: int = 64
    KH: int = 16
    VH: int = 64
    KD: int = 128
    VD: int = 128
    E: int = 512
    TOPK: int = 10
    I: int = 1024  # noqa: E741
    IS: int = 1024
    moe_tp: bool = True
    e_slots: int = 0
    eps: float = 1e-6
    theta: float = 10_000_000.0
    zero_centered_norm: bool = True

    def nh(self, R: int) -> int:
        return self.NH // R

    def G(self, R: int) -> int:
        return R // self.NKV

    def kh(self, R: int) -> int:
        return self.KH // R

    def vh(self, R: int) -> int:
        return self.VH // R

    def QKV(self, R: int) -> int:
        return 2 * self.kh(R) * self.KD + self.vh(R) * self.VD

    def Z(self, R: int) -> int:
        return self.vh(R) * self.VD

    def NLI(self, R: int) -> int:
        return self.QKV(R) + self.Z(R) + 128

    def NAI(self, R: int) -> int:
        return (2 * self.nh(R) + 2) * self.HD

    def ER(self, R: int) -> int:
        return self.E // R

    def ISR(self, R: int) -> int:
        return self.IS // R

    def IR(self, R: int) -> int:
        return self.I // R if self.moe_tp else self.I

    def EW(self, R: int) -> int:
        return (
            self.e_slots
            if self.e_slots > 0
            else (self.E if self.moe_tp else self.ER(R))
        )

    @property
    def NGDN(self) -> int:
        return self.L - self.L // self.att_every

    @property
    def NATT(self) -> int:
        return self.L // self.att_every

    def check(self, ex: ExecSpec) -> None:
        R, B, T, M = ex.R, ex.B, ex.T, ex.M
        assert self.L % self.att_every == 0 and self.NH % R == 0 and R % self.NKV == 0
        assert self.KH % R == 0 and self.VH % R == 0 and self.E % R == 0
        assert self.vh(R) <= 64 and B * self.TOPK <= 640 and 1 <= B <= M <= 64
        pc = min(1024, self.H)
        assert (
            self.E % 128 == 0
            and self.E <= 1024
            and self.H % pc == 0
            and self.nh(R) <= 8
        )
        assert self.ROT % 2 == 0 and (R & (R - 1) == 0) and self.VH % self.KH == 0
        assert T >= 1 and B % T == 0, (B, T)
        assert self.e_slots == 0 or self.e_slots >= (
            self.E if self.moe_tp else self.ER(R)
        )
        for n in (
            self.H,
            self.QKV(R),
            self.Z(R),
            self.KD,
            self.VD,
            self.HD,
            self.IR(R),
            self.ISR(R),
        ):
            assert n % 128 == 0, n


# Alias for callers using MKConfig name
MKConfig = Qwen35Geometry

FULL = Qwen35Geometry()
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
TINY = Qwen35Geometry(L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)


def qwen35_spec(g: Qwen35Geometry, gdn_block: Block | None = None) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.5."""
    gqa = GQABlock()
    moe = MoEBlock()
    if g.att_every == 1 or gdn_block is None:
        mixer: Block | SwitchSlot = SwitchSlot(
            "mixer", branches=(gqa,), branch_counts=(1,)
        )
    else:
        mixer = SwitchSlot(
            "mixer",
            branches=(gdn_block, gqa),
            branch_counts=(g.att_every - 1, 1),
        )
    lt = LayerType(name="qwen35_layer", slots=(mixer, moe))
    return ModelSpec(name="qwen3_5", config=g, layer_type=lt)


def build(
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_block: Block | None = None,
):
    """Build Qwen3.5 megakernel `(call, info)` via the generic `kernel.build` engine."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return kernel.build(spec, ex)


def make_step(
    mesh,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_block: Block | None = None,
    donate_states: bool = False,
):
    """Build Qwen3.5 `jax.jit` + `shard_map` step `(step, info)`."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return kernel.make_step(mesh, spec, ex, donate_states=donate_states)


def random_logical_inputs(
    key: jax.Array,
    g: Qwen35Geometry,
    ex: ExecSpec,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate small random logical inputs for `reference_step` and `pack_inputs`."""
    assert g.att_every == 1, (
        "M1 random_logical_inputs supports att_every=1 (GQA + MoE); M2 adds GDN"
    )
    g.check(ex)
    R, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    L, H, E, HD = g.L, g.H, g.E, g.HD
    nh = g.nh(R)
    isr, ir, ew = g.ISR(R), g.IR(R), g.EW(R)
    natt = g.NATT

    keys = jax.random.split(key, 24)
    it = iter(keys)

    base_pos = jnp.arange(1, NS + 1, dtype=I32) * min(16, ctx // 4)
    if T == 1:
        pos = base_pos
    else:
        pos = (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)

    hidden = jax.random.normal(next(it), (B, H), dtype=BF16) * 0.1

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    G = g.G(R)
    nkv = g.NKV
    koff = nh * 2 * HD

    att_q = rfp8((R, natt, H, koff))
    att_kv = jnp.repeat(rfp8((nkv, natt, H, 2 * HD)), G, axis=0)
    att_in = jnp.concatenate([att_q, att_kv], axis=-1)

    att_q_s = rscale((R, L, koff))
    att_kv_s = jnp.repeat(rscale((nkv, L, 2 * HD)), G, axis=0)
    att_in_s = jnp.concatenate([att_q_s, att_kv_s], axis=-1)

    kn = jnp.repeat(
        jax.random.normal(next(it), (nkv, L, HD), dtype=F32) * 0.05, G, axis=0
    )
    sh_gate = jnp.broadcast_to(
        (jax.random.normal(next(it), (L, H), dtype=F32) * 0.1)[None, :, :],
        (R, L, H),
    )

    logical_weights = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "att_in": att_in,
        "att_in_s": att_in_s,
        "att_out": rfp8((R, natt, nh * HD, H)),
        "att_out_s": rscale((R, L, H)),
        "qn": jax.random.normal(next(it), (R, L, HD), dtype=F32) * 0.05,
        "kn": kn,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": sh_gate,
        "sh13": rfp8((R, L, H, 2 * isr)),
        "sh13_s": rscale((R, L, 2 * isr)),
        "sh2": rfp8((R, L, isr, H)),
        "sh2_s": rscale((R, L, H)),
        "e13": rfp8((R, L, ew, H, 2 * ir)),
        "e13_s": rscale((R, L, ew, 2 * ir)),
        "e2": rfp8((R, L, ew, ir, H)),
        "e2_s": rscale((R, L, ew, H)),
    }
    kc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    vc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    logical_states = {
        "kc": jnp.repeat(kc_nkv, G, axis=0),
        "vc": jnp.repeat(vc_nkv, G, axis=0),
    }
    return pos, hidden, logical_weights, logical_states


def pack_inputs(
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    mesh: Any = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical model weights and states into the tiled/folded megakernel layout."""
    ms = qwen35_spec(g)
    p = planner.plan(ms, ex)
    R, B, M = ex.R, ex.B, ex.M
    L = g.L

    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, g.H)

    def tile_w(arr: jax.Array, kind: str) -> jax.Array:
        *lead, K_, N_ = arr.shape
        nt_, tk_ = p.nt[kind], p.tk[kind]
        assert nt_ * tk_ == K_, (kind, K_, nt_, tk_)
        return arr.reshape((*lead, nt_, tk_, N_))

    # Pack smalls [R, L, 1, S]
    lay = p.smalls_layout
    S = p.smalls_width
    sm = jnp.zeros((R, L, 1, S), dtype=F32)

    def set_sm(name: str, val: jax.Array):
        nonlocal sm
        if name not in lay:
            return
        o, w = lay[name]
        # val is either [L, w] (replicated) or [R, L, w]
        if val.ndim == 2:
            val = jnp.broadcast_to(val[None, :, :], (R, L, w))
        sm = sm.at[:, :, 0, o : o + w].set(val[:, :, :w].astype(F32))

    set_sm("in_norm", logical_weights["in_norm"])
    set_sm("post_norm", logical_weights["post_norm"])
    set_sm("in_s", logical_weights["att_in_s"])
    set_sm("out_s", logical_weights["att_out_s"])
    set_sm("qn", logical_weights["qn"])
    set_sm("kn", logical_weights["kn"])
    set_sm("sh_gate", logical_weights["sh_gate"])
    set_sm("sh13_s", logical_weights["sh13_s"])
    set_sm("sh2_s", logical_weights["sh2_s"])

    router_rep = jnp.broadcast_to(
        logical_weights["router"][None, ...], (R, L, g.H, g.E)
    )
    e13_tiled = tile_w(logical_weights["e13"], "e13")
    e2_tiled = tile_w(logical_weights["e2"], "e2")
    if ex.stack_e:
        e13_mk: Any = e13_tiled
        e2_mk: Any = e2_tiled
    else:
        e13_mk = tuple(e13_tiled[:, lyr] for lyr in range(L))
        e2_mk = tuple(e2_tiled[:, lyr] for lyr in range(L))

    es_mk = jnp.concatenate(
        [logical_weights["e13_s"], logical_weights["e2_s"]], axis=-1
    )[:, :, :, None, :].astype(F32)
    fn_mk = jnp.broadcast_to(
        logical_weights["final_norm"][None, None, :], (R, 1, g.H)
    ).astype(F32)

    weights_mk: dict[str, Any] = {
        "att_in": tile_w(logical_weights["att_in"], "att_in"),
        "att_out": tile_w(logical_weights["att_out"], "att_out"),
        "router": tile_w(router_rep, "router"),
        "sh13": tile_w(logical_weights["sh13"], "sh13"),
        "sh2": tile_w(logical_weights["sh2"], "sh2"),
        "e13": e13_mk,
        "e2": e2_mk,
        "es": es_mk,
        "smalls": sm,
        "final_norm": fn_mk,
    }
    states_mk = {
        "kc": kernel.fold(logical_states["kc"]),
        "vc": kernel.fold(logical_states["vc"]),
    }

    if mesh is not None:
        rep = NamedSharding(mesh, P())
        tp = NamedSharding(mesh, P("tp"))
        pos = jax.device_put(pos, rep)
        hid_mk = jax.device_put(hid_mk, tp)
        weights_mk = {
            k: tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
            for k, v in weights_mk.items()
        }
        states_mk = {k: jax.device_put(v, tp) for k, v in states_mk.items()}

    return pos, hid_mk, weights_mk, states_mk


def random_inputs(
    key: jax.Array,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    mesh: Any = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate packed megakernel inputs `(pos, hidden, weights, states)`."""
    pos, hid_l, w_l, st_l = random_logical_inputs(key, g, ex)
    return pack_inputs(pos, hid_l, w_l, st_l, g, ex, mesh=mesh)


def reference_step(
    g: Qwen35Geometry,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
) -> tuple[jax.Array, dict[str, Any]]:
    """Pure-JAX multi-rank reference oracle consuming logical model weights."""
    assert g.att_every == 1, (
        "M1 reference_step supports att_every=1 (GQA + MoE); M2 adds GDN"
    )
    R, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    L = g.L
    nh, HD, IR, ISR, ER = g.nh(R), g.HD, g.IR(R), g.ISR(R), g.ER(R)

    kc_all = logical_states["kc"]  # [R, NATT, NS, ctx, HD]
    vc_all = logical_states["vc"]  # [R, NATT, NS, ctx, HD]
    res = hidden[:B].astype(F32)

    half = g.ROT // 2
    inv = 1.0 / (g.theta ** (jnp.arange(half, dtype=F32)[None, :] * 2.0 / g.ROT))

    for l in range(L):  # noqa: E741
        a = l // g.att_every
        in_norm = logical_weights["in_norm"][l]
        xb = (
            rmsnorm(res, in_norm, g.eps, zero_centered=g.zero_centered_norm)
            .astype(BF16)
            .astype(F32)
        )

        att_partials = []
        knew_ranks = []
        vnew_ranks = []
        for r in range(R):
            w_in = logical_weights["att_in"][r, a].astype(F32)
            in_s = logical_weights["att_in_s"][r, l]
            y = rnd((xb @ w_in) * in_s)

            qn = logical_weights["qn"][r, l]
            kn = logical_weights["kn"][r, l]
            koff = nh * 2 * HD

            qg = y[:, :koff].reshape(B, nh, 2 * HD)
            gate = qg[:, :, HD:].reshape(B, nh * HD)
            q_all = []
            k_new_r = []
            v_new_r = []
            for b in range(B):
                ang = pos[b].astype(F32) * inv
                cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                qb = rmsnorm(
                    qg[b, :, :HD], qn, g.eps, zero_centered=g.zero_centered_norm
                )
                kb = rmsnorm(
                    y[b : b + 1, koff : koff + HD],
                    kn,
                    g.eps,
                    zero_centered=g.zero_centered_norm,
                )
                vb = y[b : b + 1, koff + HD : koff + 2 * HD]
                qb = rnd(rope_partial(qb, cos, sin, g.ROT) * (HD**-0.5))
                kb = rope_partial(kb, cos, sin, g.ROT)
                q_all.append(qb)
                k_new_r.append(kb[0])
                v_new_r.append(vb[0])
            q_r = jnp.stack(q_all, axis=0)  # [B, nh, HD]
            kn_r = jnp.stack(k_new_r, axis=0)  # [B, HD]
            vn_r = jnp.stack(v_new_r, axis=0)  # [B, HD]
            knew_ranks.append(kn_r)
            vnew_ranks.append(vn_r)

            mix_r = []
            for b in range(B):
                sq, tb = divmod(b, T)
                plen = pos[sq * T]
                k_cache = kc_all[r, a, sq].astype(F32)
                v_cache = vc_all[r, a, sq].astype(F32)
                idx = jnp.arange(ctx)
                scores_c = jnp.where(
                    idx[None, :] < plen,
                    q_r[b].astype(BF16).astype(F32) @ k_cache.T,
                    -1e30,
                )
                new_k = jnp.stack(
                    [knew_ranks[r][sq * T + j].astype(F32) for j in range(tb + 1)],
                    axis=0,
                )
                new_v = jnp.stack(
                    [vnew_ranks[r][sq * T + j].astype(F32) for j in range(tb + 1)],
                    axis=0,
                )
                scores_n = q_r[b] @ new_k.T
                scores = jnp.concatenate([scores_c, scores_n], axis=-1)
                vals = jnp.concatenate([v_cache, new_v], axis=0)
                probs = jax.nn.softmax(scores, axis=-1)
                out_b = probs @ vals
                oh = rnd(out_b).reshape(nh * HD)
                gh = rnd(jax.nn.sigmoid(gate[b]))
                mix_r.append(rnd(oh * gh))
            mix_r_arr = jnp.stack(mix_r, axis=0).astype(BF16).astype(F32)
            w_out = logical_weights["att_out"][r, a].astype(F32)
            out_s = logical_weights["att_out_s"][r, l]
            yo = (mix_r_arr @ w_out) * out_s
            att_partials.append(yo.astype(BF16).astype(F32))

        for r in range(R):
            for s_idx in range(NS):
                b0 = s_idx * T
                kc_all = kc_all.at[r, a, s_idx, pos[b0] : pos[b0] + T].set(
                    knew_ranks[r][b0 : b0 + T].astype(BF16)
                )
                vc_all = vc_all.at[r, a, s_idx, pos[b0] : pos[b0] + T].set(
                    vnew_ranks[r][b0 : b0 + T].astype(BF16)
                )

        res = rnd(res + jnp.sum(jnp.stack(att_partials, axis=0), axis=0))

        # Post-attention RMSNorm + MoE
        post_norm = logical_weights["post_norm"][l]
        x_moe = rmsnorm(res, post_norm, g.eps, zero_centered=g.zero_centered_norm)
        xb_moe = x_moe.astype(BF16).astype(F32)

        w_router = logical_weights["router"][l].astype(F32)
        logits = xb_moe @ w_router
        probs = jax.nn.softmax(logits, axis=-1)
        top_vals, top_idx = lax.top_k(probs, g.TOPK)
        top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

        moe_partials = []
        for r in range(R):
            sh_gate = logical_weights["sh_gate"][r, l]
            sg = jax.nn.sigmoid(
                jnp.sum(x_moe * sh_gate[None, :], axis=-1, keepdims=True)
            )
            w_sh13 = logical_weights["sh13"][r, l].astype(F32)
            sh13_s = logical_weights["sh13_s"][r, l]
            g13 = (xb_moe @ w_sh13) * sh13_s
            mid_sh = (
                rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
                .astype(BF16)
                .astype(F32)
            )
            w_sh2 = logical_weights["sh2"][r, l].astype(F32)
            sh2_s = logical_weights["sh2_s"][r, l]
            part_r = (mid_sh @ w_sh2) * sh2_s * sg

            e13_l = logical_weights["e13"][r, l]
            e13_s_l = logical_weights["e13_s"][r, l]
            e2_l = logical_weights["e2"][r, l]
            e2_s_l = logical_weights["e2_s"][r, l]

            for b in range(B):
                for k in range(g.TOPK):
                    e = int(top_idx[b, k])
                    if not g.moe_tp and (e // ER) != r:
                        continue
                    w_slot = e if g.moe_tp else (e - r * ER)
                    w13_e = e13_l[w_slot].astype(F32)
                    w2_e = e2_l[w_slot].astype(F32)
                    s13 = e13_s_l[w_slot]
                    s2 = e2_s_l[w_slot]
                    h_e = (xb_moe[b : b + 1] @ w13_e) * s13
                    mid_e = (
                        rnd(rnd(jax.nn.silu(rnd(h_e[:, :IR]))) * rnd(h_e[:, IR:]))
                        .astype(BF16)
                        .astype(F32)
                    )
                    y_e = rnd((mid_e @ w2_e) * s2)
                    part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)

            moe_partials.append(part_r.astype(BF16).astype(F32))

        res = rnd(res + jnp.sum(jnp.stack(moe_partials, axis=0), axis=0))

    fn = logical_weights["final_norm"].astype(F32)
    out_b = rmsnorm(res, fn, g.eps, zero_centered=g.zero_centered_norm).astype(BF16)
    return out_b, {"kc": kc_all, "vc": vc_all}
