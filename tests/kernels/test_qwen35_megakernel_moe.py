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
"""Unit tests for Qwen3.5 megakernel `MoEBlock`, `WeightRings`, and expert weight packing."""

from __future__ import annotations

import dataclasses

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import torch

from vllm_torchtpu import expert_placement
from vllm_torchtpu.kernels.qwen35_megakernel import (
    MoEBlock,
    WeightRings,
    megakernel as K,
)
from vllm_torchtpu.kernels.qwen35_megakernel.megakernel import BF16, F32, I32, rnd
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import (
    pack_router_and_shared_layer,
    pack_weights,
    reshard_layer_experts_ep_to_tp,
)

from qwen35_megakernel_test_utils import (
    make_test_ctx_and_inputs,
    make_tiny_backbone,
    ref_dmatmul,
    rms_ref,
    setup_8_cpu_devices,
)


def test_standalone_moe_block_interpret():
    """Test MoEBlock standalone in pl.pallas_call(interpret=True) vs pure-JAX reference."""
    c, _, ctx, w_sh, st_sh, lay, w_vals, st_vals, pos, hid = (
        make_test_ctx_and_inputs()
    )
    mesh1 = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))
    seg_r = ctx.nt["lin_in"] + ctx.nt["lin_out"]

    def moe_kernel(*refs):
        ctx.bind_refs(refs)
        rings = WeightRings(ctx)
        moe_blk = MoEBlock(ctx, rings)
        scr = ctx.scr
        scr["hitv"][...] = jnp.zeros((ctx.MR, 128), I32)
        zero_seen = pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        )
        zero_seen.start()
        zero_seen.wait()
        for i in range(ctx.N_MOE):
            rings.fetch(jnp.int32(seg_r + i))
        cp_sm = pltpu.make_async_copy(
            ctx.W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        )
        cp_sm.start()
        cp_sm.wait()
        scr["res"][...] = jnp.where(ctx.live, ctx.hid_ref[: ctx.MB].astype(F32), 0.0)
        part = moe_blk(jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False))
        ctx.out_ref[: ctx.MB] = part.astype(BF16)

    call = pl.pallas_call(
        moe_kernel,
        out_shape=ctx.out_shape,
        in_specs=ctx.in_specs,
        out_specs=ctx.out_specs,
        scratch_shapes=[ctx.scratch[k] for k in ctx.s_names],
        interpret=pltpu.InterpretParams(),
    )
    args = (pos, hid, *[w_vals[k] for k in w_sh], *[st_vals[k] for k in st_sh])
    hid_moe, _, _, _, _ = jax.shard_map(
        lambda *a: call(*a),
        mesh=mesh1,
        in_specs=tuple(P() for _ in args),
        out_specs=tuple(P() for _ in ctx.out_shape),
        check_vma=False,
    )(*args)

    def sm_val(l_idx, name, width=None):
        o, n = lay[name]
        w = n if width is None else width
        return w_vals["smalls"][l_idx, :, o : o + w]

    x_moe = rms_ref(hid[: c.B].astype(F32), sm_val(0, "post_norm"), c.eps)
    xb_moe = x_moe.astype(BF16)
    logits = ref_dmatmul(xb_moe, w_vals["router"][0], ctx.MB)
    probs = jax.nn.softmax(logits, axis=-1)
    top_vals, top_ids = lax.top_k(probs, c.TOPK)
    wn = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)
    glog = jnp.sum(x_moe * sm_val(0, "sh_gate"), axis=-1, keepdims=True)
    sg = jax.nn.sigmoid(glog)
    g13 = ref_dmatmul(xb_moe, w_vals["sh13"][0], ctx.MB) * sm_val(0, "sh13_s")
    mid_sh = rnd(
        rnd(jax.nn.silu(rnd(g13[:, : c.ISR]))) * rnd(g13[:, c.ISR :])
    ).astype(BF16)
    ysh = ref_dmatmul(mid_sh, w_vals["sh2"][0], ctx.MB) * sm_val(0, "sh2_s") * sg
    part_ref = ysh
    for b in range(c.B):
        for k_i in range(c.TOPK):
            e = int(top_ids[b, k_i])
            w_e = wn[b, k_i]
            es_e = w_vals["es"][0, e, 0]
            h_e = (
                ref_dmatmul(xb_moe[b : b + 1], w_vals["e13"][0, e], ctx.MB)
                * es_e[: 2 * c.IR]
            )
            mid_e = rnd(
                rnd(jax.nn.silu(rnd(h_e[:, : c.IR]))) * rnd(h_e[:, c.IR :])
            ).astype(BF16)
            y_e = rnd(
                ref_dmatmul(mid_e, w_vals["e2"][0, e], ctx.MB) * es_e[2 * c.IR :]
            )
            part_ref = part_ref.at[b : b + 1].add(w_e * y_e)
    ref_moe = part_ref.astype(BF16)

    assert (
        float(jnp.max(jnp.abs(hid_moe[: c.B].astype(F32) - ref_moe.astype(F32)))) == 0.0
    )


def test_pack_weights_ep_and_tp():
    c_ep = dataclasses.replace(
        K.TINY_EP, stack_e=False, e_slots=20, paged_kv=True
    )
    tiles = K.TINY_TILES
    backbone = make_tiny_backbone(c_ep, e_slots=20)
    packed_ep = pack_weights(backbone, c_ep, tp_rank=0, tiles=tiles)

    assert len(packed_ep["e13"]) == c_ep.L
    assert len(packed_ep["e2"]) == c_ep.L
    for l in range(c_ep.L):
        assert (
            packed_ep["e13"][l].data_ptr()
            == backbone.layers[l].mlp.experts.w13_weight.data_ptr()
        )
        assert (
            packed_ep["e2"][l].data_ptr()
            == backbone.layers[l].mlp.experts.w2_weight.data_ptr()
        )
    assert packed_ep["smalls_16x4096"].shape == (c_ep.L, 16, c_ep.H)
    assert packed_ep["es_8x8192"].shape == (c_ep.L, 20, 8, 8192)

    c_tp = dataclasses.replace(K.TINY, stack_e=False, e_slots=0, paged_kv=True)
    for l in range(c_tp.L):
        reshard_layer_experts_ep_to_tp(
            backbone.layers[l].mlp,
            tp_rank=1,
            c=c_tp,
            all_gather_fn=lambda x: x.repeat(c_tp.R, *([1] * (x.ndim - 1))),
        )
    packed_tp = pack_weights(backbone, c_tp, tp_rank=1, tiles=tiles)
    assert packed_tp["e13"][0].shape[0] == c_tp.E
    assert packed_tp["e13"][0].shape[-1] == 2 * c_tp.IR
    assert packed_tp["e2"][0].shape[0] == c_tp.E


def test_pack_weights_ecopy_placement(monkeypatch):
    c_ep = dataclasses.replace(
        K.TINY_EP, stack_e=False, e_slots=20, paged_kv=True
    )
    er, ew = c_ep.ER, c_ep.EW
    extra = ew - er
    tiles = K.TINY_TILES
    backbone = make_tiny_backbone(c_ep, e_slots=ew)

    for l in range(c_ep.L):
        cores = [list(range(r * er, (r + 1) * er)) for r in range(c_ep.R)]
        for r in range(c_ep.R):
            for k in range(extra):
                r_src = (r + 1 + (k + l) % (c_ep.R - 1)) % c_ep.R
                cores[r].append(r_src * er + er - extra + k)
        backbone.layers[l].mlp.experts._ecopy_cores = cores

    packed = pack_weights(backbone, c_ep, tp_rank=0, tiles=tiles)
    lay, S = K.smalls_layout(c_ep)
    assert "ecopy" in lay
    o_ec, w_ec = lay["ecopy"]
    assert packed["smalls"].shape == (c_ep.L, 1, S)
    for l in range(c_ep.L):
        ec_row = packed["smalls"][l, 0, o_ec : o_ec + w_ec]
        assert int((ec_row > 0).sum().item()) == c_ep.R * extra

    table_path = expert_placement.auto_table(dcp_size=1, max_num_seqs=16)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", table_path)
    slots = expert_placement.slots()
    assert slots == 78
    c_full78 = dataclasses.replace(
        K.FULL_EP, stack_e=False, e_slots=slots, paged_kv=True
    )
    cores0 = expert_placement.cores(0)
    ec0 = K.encode_ecopy(c_full78, cores0)
    assert len(ec0) == 512
    assert sum(1 for x in ec0 if x > 0) == 112


def test_tp16_shared_and_routed_expert_packing():
    """Verify R=16 64 -> 128 zero-padded shared expert and TP routed expert packing."""
    c_tiny16_ep = dataclasses.replace(
        K.TINY_EP,
        R=16,
        NH=16,
        KH=16,
        VH=32,
        E=128,
        I=1024,
        IS=1024,
        stack_e=False,
        e_slots=8,
        paged_kv=True,
    )
    c_tiny16_ep.check()
    assert c_tiny16_ep.IS_SLICE == 64 and c_tiny16_ep.ISR == 128

    backbone = make_tiny_backbone(c_tiny16_ep, e_slots=8)
    mlp0 = backbone.layers[0].mlp
    torch.manual_seed(0)
    x_test = torch.randn((4, c_tiny16_ep.H), dtype=torch.float32) * 0.01

    w_gu_full = mlp0._fused_gate_gu_fp8_weight[:, c_tiny16_ep.E :].float()
    s_gu_full = mlp0._fused_gate_gu_fp8_scale.reshape(-1)[c_tiny16_ep.E :].float()
    w_down_full = mlp0.shared_expert.down_proj.weight.float()
    s_down_full = mlp0.shared_expert.down_proj.weight_scale.reshape(-1).float()
    g_full = (x_test @ w_gu_full[:, : c_tiny16_ep.IS]) * s_gu_full[: c_tiny16_ep.IS]
    u_full = (x_test @ w_gu_full[:, c_tiny16_ep.IS :]) * s_gu_full[c_tiny16_ep.IS :]
    sh_ref = (torch.nn.functional.silu(g_full) * u_full) @ w_down_full * s_down_full

    sh_sum = torch.zeros_like(sh_ref)
    for r in range(16):
        pk_r = pack_router_and_shared_layer(
            mlp0,
            c=c_tiny16_ep,
            tp_rank=r,
            tk_router=c_tiny16_ep.H,
            tk_sh13=c_tiny16_ep.H,
            tk_sh2=c_tiny16_ep.ISR,
        )
        w13_r = pk_r["sh13"].reshape(c_tiny16_ep.H, 2 * c_tiny16_ep.ISR).float()
        s13_r = pk_r["sh13_s"].float()
        w2_r = pk_r["sh2"].reshape(c_tiny16_ep.ISR, c_tiny16_ep.H).float()
        s2_r = pk_r["sh2_s"].float()
        assert torch.all(w13_r[:, 64:128] == 0)
        assert torch.all(w13_r[:, 192:256] == 0)
        assert torch.all(w2_r[64:128, :] == 0)
        h_r = (x_test @ w13_r) * s13_r
        mid_r = torch.nn.functional.silu(h_r[:, :128]) * h_r[:, 128:]
        out_r_padded = (mid_r @ w2_r) * s2_r
        out_r_unpadded = (mid_r[:, :64] @ w2_r[:64, :]) * s2_r
        assert torch.equal(out_r_padded, out_r_unpadded)
        sh_sum = sh_sum + out_r_padded

    torch.testing.assert_close(sh_sum, sh_ref, rtol=5e-4, atol=5e-2)

    c_tiny16_tp = dataclasses.replace(c_tiny16_ep, moe_tp=True, e_slots=0)
    assert c_tiny16_tp.I_SLICE == 64 and c_tiny16_tp.IR == 128
    for layer_idx in range(c_tiny16_tp.L):
        reshard_layer_experts_ep_to_tp(
            backbone.layers[layer_idx].mlp,
            tp_rank=3,
            c=c_tiny16_tp,
            all_gather_fn=lambda x: x.repeat(c_tiny16_tp.R, *([1] * (x.ndim - 1))),
        )
    packed_tp16 = pack_weights(
        backbone,
        c_tiny16_tp,
        tp_rank=3,
        tiles=dataclasses.replace(K.TINY_TILES, bank=256 << 10),
    )
    assert packed_tp16["e13"][0].shape == (c_tiny16_tp.E, c_tiny16_tp.H, 256)
    assert packed_tp16["e2"][0].shape == (c_tiny16_tp.E, 128, c_tiny16_tp.H)
