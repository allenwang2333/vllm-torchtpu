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
"""Qwen3.5-397B-A17B model specification, pure-JAX reference oracle, and step builders."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

from vllm_torchtpu.kernels.megakernel import kernel
from vllm_torchtpu.kernels.megakernel.blocks.gqa import GQABlock
from vllm_torchtpu.kernels.megakernel.blocks.moe import MoEBlock
from vllm_torchtpu.kernels.megakernel.ops import (
    BF16,
    F32,
    I32,
    rms_zc,
    rnd,
    rope_partial,
    rup,
)
from vllm_torchtpu.kernels.megakernel.planner import (
    kernel_out_shapes,
    plan,
    smalls_layout,
    state_shapes,
    stream_kinds,
    weight_shapes,
)
from vllm_torchtpu.kernels.megakernel.spec import (
    DEFAULT_TILES,
    TINY_TILES,
    VERIFY_TILES,
    WIDE_TILES,
    Block,
    LayerType,
    ModelSpec,
    Spec,
    SwitchSlot,
    Tiles,
    default_tiles,
)


@dataclass(frozen=True)
class MKConfig:
    """Qwen3.5-397B-A17B geometry and execution configuration."""

    R: int = 8
    B: int = 4
    T: int = 1
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
    I: int = 1024
    IS: int = 1024
    moe_tp: bool = True
    e_slots: int = 0
    stack_e: bool = False
    ctx: int = 4096
    paged_kv: bool = False
    page_size: int = 128
    num_pages: int = 64
    max_pages_per_seq: int = 16
    fused_pool_io: bool = False
    act_mode: str = "none"
    ckpt_2d: bool = False
    roff_1d: bool = False
    ckpt_k: int = 0
    eps: float = 1e-6
    theta: float = 10_000_000.0

    @property
    def NS(self) -> int:
        return self.B // self.T

    @property
    def HR(self) -> int:
        return rup(3 + self.T, 8)

    @property
    def MR(self) -> int:
        return rup(self.B, 8)

    @property
    def M(self) -> int:
        return max(16, rup(self.B, 16))

    @property
    def nh(self) -> int:
        return self.NH // self.R

    @property
    def G(self) -> int:
        return self.R // self.NKV

    @property
    def kh(self) -> int:
        return self.KH // self.R

    @property
    def vh(self) -> int:
        return self.VH // self.R

    @property
    def QKV(self) -> int:
        return 2 * self.kh * self.KD + self.vh * self.VD

    @property
    def Z(self) -> int:
        return self.vh * self.VD

    @property
    def BA(self) -> int:
        return 128

    @property
    def NLI(self) -> int:
        return self.QKV + self.Z + self.BA

    @property
    def NAI(self) -> int:
        return (2 * self.nh + 2) * self.HD

    @property
    def ER(self) -> int:
        return self.E // self.R

    @property
    def ISR(self) -> int:
        return self.IS // self.R

    @property
    def IR(self) -> int:
        return self.I // self.R if self.moe_tp else self.I

    @property
    def EW(self) -> int:
        return self.e_slots if self.e_slots > 0 else (self.E if self.moe_tp else self.ER)

    @property
    def NGDN(self) -> int:
        return self.L - self.L // self.att_every

    @property
    def NATT(self) -> int:
        return self.L // self.att_every

    @property
    def ngdn(self) -> int:
        return self.NGDN

    @property
    def natt(self) -> int:
        return self.NATT

    @property
    def CR(self) -> int:
        return rup(3 * self.B, 8)

    @property
    def PC(self) -> int:
        return min(1024, self.H)

    @property
    def PR(self) -> int:
        return rup(self.B * self.H // self.PC, 16)

    def check(self) -> None:
        assert self.L % self.att_every == 0 and self.NH % self.R == 0 and self.R % self.NKV == 0
        assert self.KH % self.R == 0 and self.VH % self.R == 0 and self.E % self.R == 0
        assert self.vh <= 64 and self.B * self.TOPK <= 640 and 1 <= self.B <= self.M <= 64
        assert self.E % 128 == 0 and self.E <= 1024 and self.H % self.PC == 0 and self.nh <= 8
        assert self.ROT % 2 == 0 and self.R & (self.R - 1) == 0 and self.VH % self.KH == 0
        assert self.T >= 1 and self.B % self.T == 0, (self.B, self.T)
        assert self.e_slots == 0 or self.e_slots >= (self.E if self.moe_tp else self.ER)
        if self.paged_kv:
            assert self.page_size == 128 and self.num_pages >= 1 and self.max_pages_per_seq >= 1
        for n in (self.H, self.QKV, self.Z, self.KD, self.VD, self.HD, self.IR, self.ISR):
            assert n % 128 == 0, n


FULL = MKConfig()
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
TINY = MKConfig(L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024, ctx=2048)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)


def qwen35_spec(c: MKConfig, gdn_block: Block | None = None) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.5."""
    gqa = GQABlock()
    moe = MoEBlock()
    if c.att_every == 1 or gdn_block is None:
        mixer: Block | SwitchSlot = SwitchSlot("mixer", branches=(gqa,), period=c.att_every)
    else:
        mixer = SwitchSlot("mixer", branches=(gdn_block, gqa), period=c.att_every)
    lt = LayerType(name="qwen35_layer", slots=(mixer, moe))
    return ModelSpec(name="qwen35", config=c, layer_type=lt)


def build(
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    gdn_block: Block | None = None,
    interpret: bool = False,
    forced_routing: bool = False,
    skip=frozenset(),
    scopes: int | bool = False,
    row_mask: bool = False,
):
    """Build Qwen3.5 megakernel `(call, info)` via the generic `kernel.build` engine."""
    spec = qwen35_spec(c, gdn_block=gdn_block)
    return kernel.build(
        spec,
        tiles=tiles if tiles is not None else default_tiles(c.B),
        interpret=interpret,
        forced_routing=forced_routing,
        skip=skip,
        scopes=scopes,
        row_mask=row_mask,
    )


def make_step(
    mesh,
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    gdn_block: Block | None = None,
    interpret: bool = False,
    forced_routing: bool = False,
    skip=frozenset(),
    scopes: int | bool = False,
):
    """Build Qwen3.5 `shard_map` step `(step, info)` via the generic `kernel.make_step` engine."""
    spec = qwen35_spec(c, gdn_block=gdn_block)
    return kernel.make_step(
        mesh,
        spec,
        tiles=tiles if tiles is not None else default_tiles(c.B),
        interpret=interpret,
        forced_routing=forced_routing,
        skip=skip,
        scopes=scopes,
    )


def reference_step(
    c: MKConfig,
    tiles: Tiles | None,
    pos: jax.Array,
    hidden: jax.Array,
    weights: dict[str, Any],
    states: dict[str, Any],
) -> tuple[jax.Array, dict[str, Any]]:
    """Pure-JAX multi-rank reference oracle for Qwen3.5 (`att_every=1` in M1).

    Takes the exact same `(pos, hidden, weights, states)` arguments as `make_step`
    and computes the forward pass using standard JAX array ops.
    """
    assert c.att_every == 1, "M1 reference_step supports att_every=1 (GQA + MoE); M2 adds GDN"
    tiles = default_tiles(c.B) if tiles is None else tiles
    lay, _ = smalls_layout(c)
    R, B, T, NS, M, H, L, E = c.R, c.B, c.T, c.NS, c.M, c.H, c.L, c.E
    nh, HD, IR, ISR, ER = c.nh, c.HD, c.IR, c.ISR, c.ER

    def get_sm(r: int, l: int, name: str, n: int | None = None) -> jax.Array:
        o, w = lay[name]
        width = w if n is None else n
        return weights["smalls"][r, l, 0, o : o + width].astype(F32)

    def untile(w_tiled: jax.Array) -> jax.Array:
        # [..., nt, tk, N] -> [..., nt * tk, N]
        *lead, nt_, tk_, N_ = w_tiled.shape
        return w_tiled.reshape((*lead, nt_ * tk_, N_)).astype(F32)

    kc_all = kernel.unfold(states["kc"], R)  # [R, NATT, NS, ctx, HD]
    vc_all = kernel.unfold(states["vc"], R)  # [R, NATT, NS, ctx, HD]
    # Initial residual on rank 0 (replicated across ranks)
    res = hidden[:B].astype(F32)

    half = c.ROT // 2
    inv = 1.0 / (
        c.theta ** (jnp.arange(half, dtype=F32)[None, :] * 2.0 / c.ROT)
    )

    for l in range(L):
        a = l // c.att_every
        # 1. Input RMSNorm (identical across ranks since `in_norm` and `res` are replicated)
        in_norm = get_sm(0, l, "in_norm")
        xb = rms_zc(res, in_norm, c.eps).astype(BF16).astype(F32)

        # 2. Per-rank GQA branch + KV cache update
        att_partials = []
        knew_ranks = []
        vnew_ranks = []
        for r in range(R):
            w_in = untile(weights["att_in"][r, a])  # [H, NAI]
            in_s = get_sm(r, l, "in_s", c.NAI)
            y = rnd((xb @ w_in) * in_s)

            qn = get_sm(r, l, "qn")
            kn = get_sm(r, l, "kn")
            koff = nh * 2 * HD

            qg = y[:, :koff].reshape(B, nh, 2 * HD)
            gate = qg[:, :, HD:].reshape(B, nh * HD)
            q_all = []
            k_new_r = []
            v_new_r = []
            for b in range(B):
                ang = pos[b].astype(F32) * inv
                cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                qb = rms_zc(qg[b, :, :HD], qn, c.eps)
                kb = rms_zc(y[b : b + 1, koff : koff + HD], kn, c.eps)
                vb = y[b : b + 1, koff + HD : koff + 2 * HD]
                qb = rnd(rope_partial(qb, cos, sin, c.ROT) * (HD**-0.5))
                kb = rope_partial(kb, cos, sin, c.ROT)
                q_all.append(qb)
                k_new_r.append(kb[0])
                v_new_r.append(vb[0])
            q_r = jnp.stack(q_all, axis=0)      # [B, nh, HD]
            kn_r = jnp.stack(k_new_r, axis=0)   # [B, HD]
            vn_r = jnp.stack(v_new_r, axis=0)   # [B, HD]
            knew_ranks.append(kn_r.astype(BF16))
            vnew_ranks.append(vn_r.astype(BF16))

            # Update KV cache for sequence s and attend over 0..pos[b]
            mix_r = []
            for b in range(B):
                sq, tb = divmod(b, T)
                plen = pos[sq * T]
                # Cached keys/values [ctx, HD]
                k_cache = kc_all[r, a, sq].astype(F32)
                v_cache = vc_all[r, a, sq].astype(F32)
                idx = jnp.arange(c.ctx)
                scores_c = jnp.where(
                    idx[None, :] < plen,
                    q_r[b].astype(BF16).astype(F32) @ k_cache.T,
                    -1e30,
                )
                # New tokens 0..tb in sequence sq
                new_k = jnp.stack([knew_ranks[r][sq * T + j].astype(F32) for j in range(tb + 1)], axis=0)
                new_v = jnp.stack([vnew_ranks[r][sq * T + j].astype(F32) for j in range(tb + 1)], axis=0)
                scores_n = q_r[b] @ new_k.T
                scores = jnp.concatenate([scores_c, scores_n], axis=-1)
                vals = jnp.concatenate([v_cache, new_v], axis=0)
                probs = jax.nn.softmax(scores, axis=-1)
                out_b = probs @ vals  # [nh, HD]
                oh = rnd(out_b).reshape(nh * HD)
                gh = rnd(jax.nn.sigmoid(gate[b]))
                mix_r.append(rnd(oh * gh))
            mix_r_arr = jnp.stack(mix_r, axis=0).astype(BF16).astype(F32)  # [B, nh * HD]
            w_out = untile(weights["att_out"][r, a])  # [nh * HD, H]
            out_s = get_sm(r, l, "out_s")
            yo = (mix_r_arr @ w_out) * out_s
            att_partials.append(yo.astype(BF16).astype(F32))

        for r in range(R):
            for s_idx in range(NS):
                b0 = s_idx * T
                kc_all = kc_all.at[r, a, s_idx, pos[b0] : pos[b0] + T].set(
                    knew_ranks[r][b0 : b0 + T]
                )
                vc_all = vc_all.at[r, a, s_idx, pos[b0] : pos[b0] + T].set(
                    vnew_ranks[r][b0 : b0 + T]
                )

        res = rnd(res + jnp.sum(jnp.stack(att_partials, axis=0), axis=0))

        # 3. Post-attention RMSNorm + MoE (shared + routed experts)
        post_norm = get_sm(0, l, "post_norm")
        x_moe = rms_zc(res, post_norm, c.eps)
        xb_moe = x_moe.astype(BF16).astype(F32)

        # Router is replicated across ranks
        w_router = untile(weights["router"][0, l])  # [H, E]
        logits = xb_moe @ w_router
        probs = jax.nn.softmax(logits, axis=-1)
        top_vals, top_idx = lax.top_k(probs, c.TOPK)
        top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)  # [B, TOPK]

        moe_partials = []
        for r in range(R):
            sh_gate = get_sm(r, l, "sh_gate")
            sg = jax.nn.sigmoid(jnp.sum(x_moe * sh_gate[None, :], axis=-1, keepdims=True))
            w_sh13 = untile(weights["sh13"][r, l])
            sh13_s = get_sm(r, l, "sh13_s")
            g13 = (xb_moe @ w_sh13) * sh13_s
            mid_sh = rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:])).astype(BF16).astype(F32)
            w_sh2 = untile(weights["sh2"][r, l])
            sh2_s = get_sm(r, l, "sh2_s")
            part_r = (mid_sh @ w_sh2) * sh2_s * sg

            e13_l = weights["e13"][r, l] if c.stack_e else weights["e13"][l][r]
            e2_l = weights["e2"][r, l] if c.stack_e else weights["e2"][l][r]
            es_l = weights["es"][r, l, :, 0, :]  # [EW, 2*IR + H]

            for b in range(B):
                for k in range(c.TOPK):
                    e = int(top_idx[b, k])
                    if not c.moe_tp and (e // ER) != r:
                        continue
                    w_slot = e if c.moe_tp else (e - r * ER)
                    w13_e = untile(e13_l[w_slot])  # [H, 2*IR]
                    w2_e = untile(e2_l[w_slot])    # [IR, H]
                    s13 = es_l[w_slot, : 2 * IR]
                    s2 = es_l[w_slot, 2 * IR :]
                    h_e = (xb_moe[b : b + 1] @ w13_e) * s13
                    mid_e = rnd(rnd(jax.nn.silu(rnd(h_e[:, :IR]))) * rnd(h_e[:, IR:])).astype(BF16).astype(F32)
                    y_e = rnd((mid_e @ w2_e) * s2)
                    part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)

            moe_partials.append(part_r.astype(BF16).astype(F32))

        res = rnd(res + jnp.sum(jnp.stack(moe_partials, axis=0), axis=0))

    # Final RMSNorm
    fn = weights["final_norm"][0, 0].astype(F32)
    out_b = rms_zc(res, fn, c.eps).astype(BF16)
    out_pad = jnp.pad(out_b, ((0, M - B), (0, 0)))
    hidden_out = jnp.tile(out_pad, (R, 1))
    new_states = {
        **states,
        "kc": kernel.fold(kc_all),
        "vc": kernel.fold(vc_all),
    }
    return hidden_out, new_states

