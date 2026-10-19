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
"""Test and bench harness for the serving form of the prefill megakernel: builds a
multi-sequence step (padded layout, SMEM metadata, region pools holding each sequence's cached
K/V and GDN states) and the per-sequence reference to check it against."""

import dataclasses

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen35_prefill_mk import megakernel as MK
from vllm_torchtpu.kernels.qwen35_prefill_mk import phases as P
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R

PAGE = P.PAGE
FP8_NP = ml_dtypes.float8_e4m3fn


@dataclasses.dataclass
class Case:
    t: int
    starts: list
    lens: list
    ctxs: list
    meta: dict
    pools: list
    attn_blocks: list          # per sequence, its attention manager blocks
    gdn_slots: list            # [group][seq]


def padded_layout(lens, t=None):
    starts, pos = [], 0
    for n in lens:
        starts.append(pos)
        pos += -(-n // P.SEG_ALIGN) * P.SEG_ALIGN
    t = t or pos
    assert pos <= t, (pos, t)
    return starts, t


def kv_page(bt_seq, p, split, block_tokens):
    """Kernel page of position p for a sequence with manager blocks bt_seq."""
    return bt_seq[p // block_tokens] * split + (p % block_tokens) // PAGE


def make_case(c, layout, lens, ctxs, *, max_seqs, pps, block_tokens, t=None, seed=0):
    """Metadata and empty pools for a step of len(lens) sequences."""
    starts, t = padded_layout(lens, t)
    split = block_tokens // PAGE
    tile_seg = np.full((t // P.BLK,), -1, np.int32)
    for s, (st, n) in enumerate(zip(starts, lens)):
        tile_seg[st // P.BLK:(st + -(-n // P.SEG_ALIGN) * P.SEG_ALIGN) // P.BLK] = s
    nxt = [1]                                   # block 0 left unused

    def alloc(k):
        b = list(range(nxt[0], nxt[0] + k))
        nxt[0] += k
        return b
    attn_blocks = [alloc(-(-(cx + n) // block_tokens)) for n, cx in zip(lens, ctxs)]
    gdn_slots = [[alloc(1)[0] for _ in lens] for _ in range(3)]
    bt = np.zeros((max_seqs * pps,), np.int32)
    for s, blocks in enumerate(attn_blocks):
        for p in range(0, (ctxs[s] + lens[s] + PAGE - 1) // PAGE * PAGE, PAGE):
            bt[s * pps + p // PAGE] = kv_page(blocks, p, split, block_tokens)
    pad = lambda a: np.pad(np.asarray(a, np.int32), (0, max_seqs - len(a)))
    meta = {"tile_seg": tile_seg, "seg_start": pad(starts), "seg_len": pad(lens),
            "seg_ctx": pad(ctxs), "bt": bt}
    for g in range(3):
        meta[f"gslot{g}"] = pad(gdn_slots[g])
    n_pages = nxt[0] * split
    pools = [np.zeros((n_pages, 2 * c.NKV, c.HD // 4, 4, PAGE), FP8_NP)
             for _ in range(layout.n_regions)]
    return Case(t, starts, lens, ctxs, {k: jnp.asarray(v) for k, v in meta.items()}, pools,
                attn_blocks, gdn_slots)


def write_kv(pool, blocks, k, v, p0, split, block_tokens):
    """pool[page, 2h | 2h+1, d // 4, d % 4, lane] = k | v[p - p0, h, d] for the given positions."""
    n, nkv, hd = k.shape
    for i in range(n):
        p = p0 + i
        pg, ln = kv_page(blocks, p, split, block_tokens), p % PAGE
        for h in range(nkv):
            pool[pg, 2 * h, :, :, ln] = np.asarray(k[i, h]).reshape(hd // 4, 4)
            pool[pg, 2 * h + 1, :, :, ln] = np.asarray(v[i, h]).reshape(hd // 4, 4)


def read_kv(pool, blocks, p0, n, split, block_tokens, nkv, hd):
    k = np.zeros((n, nkv, hd), FP8_NP)
    v = np.zeros((n, nkv, hd), FP8_NP)
    for i in range(n):
        p = p0 + i
        pg, ln = kv_page(blocks, p, split, block_tokens), p % PAGE
        for h in range(nkv):
            k[i, h] = pool[pg, 2 * h, :, :, ln].reshape(hd)
            v[i, h] = pool[pg, 2 * h + 1, :, :, ln].reshape(hd)
    return k, v


def _state_kernel(plan, stride, n, conv_shape, rec_shape, store, interpret):
    """A pallas_call packing (store) or unpacking n slots' GDN states into / from a pool through
    the plan, with the serving GDN kernel's own vmem_ldst helpers."""
    from vllm_torchtpu.kernels.gdn.v3 import vmem_ldst

    def kernel(slots, pool_in, conv_ref, rec_ref, pool_out, conv_out, rec_out, cwin, rwin, sem):
        del pool_in
        for i in range(n):
            sl = slots[i]
            for reg, win in ((plan.conv, cwin), (plan.recurrent, rwin)):
                src = pool_out.at[pl.ds(sl * stride + reg.kb0, reg.nblocks)]
                if store:
                    if reg is plan.conv:
                        vmem_ldst.store_state_region(win, reg, conv_ref[i].reshape(
                            conv_shape[0], 1, conv_shape[1]).astype(jnp.float32))
                    else:
                        vmem_ldst.store_state_region(win, reg, rec_ref[i])
                    cp = pltpu.make_async_copy(win, src, sem.at[0])
                    cp.start()
                    cp.wait()
                else:
                    cp = pltpu.make_async_copy(src, win, sem.at[0])
                    cp.start()
                    cp.wait()
                    if reg is plan.conv:
                        conv_out[i] = vmem_ldst.load_state_region(
                            win, reg, (conv_shape[0], 1, conv_shape[1])).reshape(conv_shape)
                    else:
                        rec_out[i] = vmem_ldst.load_state_region(win, reg, rec_shape).astype(
                            jnp.float32)

    def call(pool, slots, convs, recs):
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        vm = pl.BlockSpec(memory_space=pltpu.VMEM)
        page = pool.shape[1:]
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(pool.shape, pool.dtype),
                       jax.ShapeDtypeStruct((n, *conv_shape), jnp.float32),
                       jax.ShapeDtypeStruct((n, *rec_shape), jnp.float32)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1, grid=(), in_specs=[hbm, vm, vm], out_specs=[hbm, vm, vm],
                scratch_shapes=[pltpu.VMEM((plan.conv.nblocks, *page), pool.dtype),
                                pltpu.VMEM((plan.recurrent.nblocks, *page), pool.dtype),
                                pltpu.SemaphoreType.DMA((1,))]),
            input_output_aliases={1: 0},
            interpret=pltpu.InterpretParams() if interpret else False,
        )(slots, pool, convs, recs)
    return call


def store_states(pool, plan, stride, slots, convs, recs, *, interpret=False):
    """Write GDN states (convs [n, ks-1, QKV], recs [n, VH, KD, VD]) into the pool's slots."""
    n = len(slots)
    call = _state_kernel(plan, stride, n, convs.shape[1:], recs.shape[1:], True, interpret)
    out, _, _ = call(jnp.asarray(pool), jnp.asarray(slots, jnp.int32),
                     jnp.asarray(convs, jnp.float32), jnp.asarray(recs, jnp.float32))
    return out


def load_states(pool, plan, stride, slots, conv_shape, rec_shape, *, interpret=False):
    n = len(slots)
    call = _state_kernel(plan, stride, n, conv_shape, rec_shape, False, interpret)
    _, convs, recs = call(jnp.asarray(pool), jnp.asarray(slots, jnp.int32),
                          jnp.zeros((n, *conv_shape), jnp.float32),
                          jnp.zeros((n, *rec_shape), jnp.float32))
    return convs, recs


def reference_step(c, layout, weights, hidden, residual, case, states, *, flash_bkv=None,
                   moe_fn=None):
    """reference.stage_step over a multi-sequence step: each sequence's rows through each mixer
    with its own context and states (states[li][s]: gdn {"conv", "rec"}, attn {"k", "v"} with
    the cached context in [:ctx]); MoE over all rows (padding rows included, as the kernel
    does). Returns hidden, residual and the new per-sequence states."""
    hidden, residual = hidden, residual
    new_states = []
    for li, lt in enumerate(layout.layer_types):
        w = weights[li]
        res = hidden.astype(R.F32) + residual.astype(R.F32)
        residual = res.astype(R.BF16)
        x = R.rmsnorm_zc(res, w["in_norm"], c.eps).astype(R.BF16)
        y = jnp.zeros((hidden.shape[0], c.H), R.F32)
        layer_states = []
        for s, (st, n, cx) in enumerate(zip(case.starts, case.lens, case.ctxs)):
            xs = x[st:st + n]
            if lt == "gdn":
                ys, conv, rec = R.gdn_layer(c, w, xs, states[li][s]["conv"], states[li][s]["rec"])
                layer_states.append({"conv": conv, "rec": rec})
            else:
                ys, kc, vc = R.attn_layer(c, w, xs, states[li][s]["k"], states[li][s]["v"], cx,
                                          flash_bkv)
                layer_states.append({"k": kc, "v": vc})
            y = y.at[st:st + n].set(ys)
        new_states.append(layer_states)
        res = y.astype(R.BF16).astype(R.F32) + residual.astype(R.F32)
        residual = res.astype(R.BF16)
        x = R.rmsnorm_zc(res, w["post_norm"], c.eps).astype(R.BF16)
        hidden = (moe_fn or R.moe)(c, w, x).astype(R.BF16)
    return hidden, residual, new_states
