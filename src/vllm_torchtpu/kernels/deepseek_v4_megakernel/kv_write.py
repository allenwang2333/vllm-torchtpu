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
"""Compressor state update and compressed-row store for verify windows.

What `compress_and_store.compressor_forward` does after its projection, for
the ``q_len`` rows of each request, as a megakernel step:

  1. store each row's f32 state (``kv | score + ape``, the projection
     epilogue's output) into its state slot;
  2. for every row that closes a compressed token (``(pos + 1) % ratio == 0``):
     read the window's states back (``coff * ratio`` positions; with overlap
     the older half contributes its second-head fields), softmax-pool them per
     dim, RMSNorm, interleaved RoPE at ``(pos // ratio) * ratio``, quantize as
     the mode stores it and write the record:
       hca  raw bf16 latent (2 rows of the HCA page)
       csa  e4m3 lane-periodic NoPE record + bf16 RoPE words (RoPE row RMW)
       idx  e4m3 block-128 + e8m0 record into the seq-along-lane indexer page

States and records are read and written as int32 words: a state host array
row of L lanes holds f32 elements ``[row * L, row * L + L)`` of the states laid
token after token (``spt = 2 * state_width / L`` rows a token).
"""

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8_MAX = 448.0

_MODES = {  # head_dim, ratio, overlap, quant_block
    "hca": (512, 128, False, 0),
    "csa": (512, 4, True, 64),
    "idx": (128, 4, True, 128),
}


@dataclasses.dataclass(frozen=True)
class CompressConfig:
    mode: str
    max_req: int
    state_block_size: int  # positions per state page (state block table unit)
    state_pages_per_req: int
    kv_pages_per_req: int
    kv_page_tokens: int  # compressed rows per KV page
    state_lanes: int = 128  # L: lanes of the state host array
    q_len: int = 7
    rms_eps: float = 1e-6
    rope_dim: int = 64

    @property
    def head_dim(self):
        return _MODES[self.mode][0]

    @property
    def ratio(self):
        return _MODES[self.mode][1]

    @property
    def overlap(self):
        return _MODES[self.mode][2]

    @property
    def quant_block(self):
        return _MODES[self.mode][3]

    @property
    def coff(self):
        return 1 + int(self.overlap)

    @property
    def state_width(self):
        return self.coff * self.head_dim

    @property
    def spt(self):  # state rows per token
        return 2 * self.state_width // self.state_lanes

    @property
    def window(self):
        return self.coff * self.ratio


class CompressScratch(NamedTuple):
    states: object  # int32 [8, spt, L] staged states of the request
    win: object  # int32 [window, spt, L]
    rec: object  # int32 [2, 128] record words (hca: 2 rows; csa NoPE: row 0)
    aux: object  # int32 [40, 128]: RoPE row / indexer SAL lane tile (RMW)
    cs: object  # f32 [8, rope_dim]: cos|sin row
    sems: object  # DMA [4]: state writes, window reads, record, RMW


def compress_scratch_shapes(cfg: CompressConfig) -> CompressScratch:
    L = cfg.state_lanes
    return CompressScratch(
        states=pltpu.VMEM((8, cfg.spt, L), jnp.int32),
        win=pltpu.VMEM((cfg.window, cfg.spt, L), jnp.int32),
        rec=pltpu.VMEM((2, 128), jnp.int32),
        aux=pltpu.VMEM((40, 128), jnp.int32),
        cs=pltpu.VMEM((8, cfg.rope_dim), jnp.float32),
        sems=pltpu.SemaphoreType.DMA((4,)),
    )


def _fp8_codes(x):
    """f32 -> int32 holding the e4m3 code (as `.astype(float8_e4m3fn)`)."""
    return lax.bitcast_convert_type(x.astype(jnp.float8_e4m3fn), jnp.uint8).astype(jnp.int32)


def _bf16_bits(x):
    """f32 -> int32 holding the bf16 bits of `x.astype(bfloat16)` (low 16)."""
    b = lax.bitcast_convert_type(x.astype(jnp.bfloat16).astype(jnp.float32), jnp.int32)
    return (b >> 16) & 0xFFFF


def _elems(f, a, n, L):
    """f32 [W, spt, L] state rows -> elements [a, a + n) as [W, n] (a, n % 128 == 0)."""
    return jnp.concatenate(
        [f[:, (a + c) // L, (a + c) % L:(a + c) % L + 128] for c in range(0, n, 128)], axis=1)


def compress_rows(cfg: CompressConfig, f, valid, norm_w, cos, sin):
    """Window states -> the compressed token (f32 [1, head_dim]).

    f: f32 [window, spt, L]; valid: bool [window, 1]; norm_w: f32 [1, D];
    cos, sin: f32 [1, rope_dim // 2]. Same formulas as `compress_store_ref`.
    """
    D, R, SW, L = cfg.head_dim, cfg.ratio, cfg.state_width, cfg.state_lanes
    if cfg.overlap:  # older half: second-head fields; newer half: first-head
        kv = jnp.concatenate([_elems(f[:R], 0, D, L), _elems(f[R:], D, D, L)], axis=0)
        sc = jnp.concatenate([_elems(f[:R], SW, D, L), _elems(f[R:], SW + D, D, L)], axis=0)
    else:
        kv, sc = _elems(f, 0, D, L), _elems(f, SW, D, L)
    sc = jnp.where(valid, sc, -jnp.inf)
    m = jnp.max(sc, axis=0, keepdims=True)
    e = jnp.exp(sc - m)
    w = e / jnp.sum(e, axis=0, keepdims=True)
    c = jnp.sum(w * kv, axis=0, keepdims=True)  # [1, D]
    var = jnp.mean(jnp.square(c), axis=-1, keepdims=True)
    c = c * lax.rsqrt(var + cfg.rms_eps) * norm_w
    # Interleaved RoPE on the last rope_dim dims: pairs (2i, 2i + 1).
    half = cfg.rope_dim // 2
    lane = lax.broadcasted_iota(jnp.int32, (1, D), 1)
    pair = (lane - (D - cfg.rope_dim)) // 2
    is_rope = lane >= D - cfg.rope_dim
    cos_full = jnp.ones((1, D), jnp.float32)
    sin_full = jnp.zeros((1, D), jnp.float32)
    for i in range(half):  # place cos/sin of pair i on both of its lanes
        hit = is_rope & (pair == i)
        cos_full = jnp.where(hit, cos[:, i:i + 1], cos_full)
        sin_full = jnp.where(hit, sin[:, i:i + 1], sin_full)
    odd = (lane % 2) == 1
    partner = jnp.where(odd, pltpu.roll(c, 1, 1), pltpu.roll(c, D - 1, 1))
    # even: x_e cos - x_o sin; odd: x_o cos + x_e sin
    return c * cos_full + partner * jnp.where(odd, sin_full, -sin_full)


def compress_store_step(
    cfg: CompressConfig,
    *,
    num_req,
    seq_lens,  # SMEM int32 [max_req] (incl. the q_len rows)
    st_pt,  # SMEM int32 [max_req * state_pages_per_req]
    kv_pt,  # SMEM int32 [max_req * kv_pages_per_req]
    states_ref,  # VMEM f32 [max_req, 8, 2 * state_width] (row t < q_len)
    norm_ref,  # VMEM f32 [1, head_dim]
    cos_sin_hbm,  # f32 [max_pos, rope_dim]
    state_hbm,  # int32 [N, rows, L]
    kv_hbm,  # hca: int32 [N, 2 * page, 128]; csa: NoPE int32 [N, page, 128]; idx: [N, 64, 256]
    rope_hbm,  # csa: int32 [N, page // 4, 128]; else None
    scr: CompressScratch,
):
    T, L, SBS, R = cfg.q_len, cfg.state_lanes, cfg.state_block_size, cfg.ratio
    spt, W, D = cfg.spt, cfg.window, cfg.head_dim
    SP, KP, KT = cfg.state_pages_per_req, cfg.kv_pages_per_req, cfg.kv_page_tokens

    def state_slot(r, pos):
        page = jnp.clip(st_pt[r * SP + jnp.clip(pos // SBS, 0, SP - 1)], 0, state_hbm.shape[0] - 1)
        return page, (pos % SBS) * spt

    def copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, scr.sems.at[sem])
        cp.wait() if wait else cp.start()

    def req_body(r, carry):
        p0 = seq_lens[r] - T
        # 1. states
        st = lax.bitcast_convert_type(states_ref[r], jnp.int32)  # [8, 2 SW]
        scr.states[...] = st.reshape(8, spt, L)
        for wait in (False, True):
            for t in range(T):
                page, row = state_slot(r, p0 + t)
                copy(scr.states.at[t], state_hbm.at[page, pl.ds(row, spt)], 0, wait)

        # 2. compressed tokens
        def boundary(t, c):
            pos = p0 + t

            @pl.when((pos + 1) % R == 0)
            def _():
                start = pos - W + 1
                for wait in (False, True):
                    for w in range(W):
                        page, row = state_slot(r, jnp.maximum(start + w, 0))
                        copy(state_hbm.at[page, pl.ds(row, spt)], scr.win.at[w], 1, wait)
                    copy(cos_sin_hbm.at[pl.ds((pos // R) * R, 1)], scr.cs.at[pl.ds(0, 1)], 1, wait)
                f = lax.bitcast_convert_type(scr.win[...], jnp.float32)
                valid = start + lax.broadcasted_iota(jnp.int32, (W, 1), 0) >= 0
                cs = scr.cs[0:1, :]
                half = cfg.rope_dim // 2
                y = compress_rows(cfg, f, valid, norm_ref[...], cs[:, :half], cs[:, half:])
                store(r, pos // R, y)

            return c

        lax.fori_loop(0, T, boundary, 0)
        return carry

    def store(r, k, y):  # y: f32 [1, D] compressed token k of request r
        page = jnp.clip(kv_pt[r * KP + jnp.clip(k // KT, 0, KP - 1)], 0, kv_hbm.shape[0] - 1)
        slot = k % KT
        lane = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        if cfg.mode == "hca":
            b = _bf16_bits(y)  # [1, 512]
            words = [b[:, 256 * s: 256 * s + 128] | (b[:, 256 * s + 128: 256 * s + 256] << 16)
                     for s in range(2)]
            scr.rec[...] = jnp.concatenate(words, axis=0)
            for wait in (False, True):
                copy(scr.rec, kv_hbm.at[page, pl.ds(2 * slot, 2)], 2, wait)
            return
        if cfg.mode == "csa":
            nope = D - cfg.rope_dim  # 448
            period = cfg.quant_block  # 64
            a = jnp.where(lax.broadcasted_iota(jnp.int32, (1, D), 1) < nope, jnp.abs(y), 0.0)
            folded = a[:, 0:period]
            for j in range(1, D // period):
                folded = jnp.maximum(folded, a[:, j * period:(j + 1) * period])
            amax = jnp.maximum(folded, 1e-4)
            scale = jnp.exp2(jnp.ceil(jnp.log2(amax / FP8_MAX)))  # [1, 64]
            inv = 1.0 / jnp.concatenate([scale] * (D // period), axis=1)
            q = _fp8_codes(y * inv)  # [1, 512]
            sbyte = (lax.bitcast_convert_type(scale, jnp.int32) >> 23) & 0xFF
            b3 = jnp.where(lane < 64, q[:, 384:512], pltpu.roll(jnp.concatenate([sbyte, sbyte], 1), 64, 1))
            scr.rec[0:1, :] = q[:, 0:128] | (q[:, 128:256] << 8) | (q[:, 256:384] << 16) | (b3 << 24)
            for wait in (False, True):
                copy(scr.rec.at[pl.ds(0, 1)], kv_hbm.at[page, pl.ds(slot, 1)], 2, wait)
            # RoPE words: k -> bf16 dim nope + k | bf16 dim nope + 32 + k << 16, into
            # segment slot % 4 of row slot // 4 (read-modify-write).
            rb = _bf16_bits(y[:, nope:])  # [1, 64]
            words = rb[:, 0:32] | (rb[:, 32:64] << 16)  # [1, 32]
            words = jnp.concatenate([words] * 4, axis=1)  # [1, 128]
            copy(rope_hbm.at[page, pl.ds(slot // 4, 1)], scr.aux.at[pl.ds(0, 1)], 3, False)
            copy(rope_hbm.at[page, pl.ds(slot // 4, 1)], scr.aux.at[pl.ds(0, 1)], 3, True)
            seg = lane // 32
            scr.aux[0:1, :] = jnp.where(seg == slot % 4, words, scr.aux[0:1, :])
            for wait in (False, True):
                copy(scr.aux.at[pl.ds(0, 1)], rope_hbm.at[page, pl.ds(slot // 4, 1)], 3, wait)
            return
        # idx: block-128 e4m3 + e8m0, column `slot` of the seq-along-lane page.
        amax = jnp.maximum(jnp.max(jnp.abs(y), axis=1, keepdims=True), 1e-4)
        scale = jnp.exp2(jnp.ceil(jnp.log2(amax / FP8_MAX)))  # [1, 1]
        q = _fp8_codes(y * (1.0 / scale))  # [1, 128]
        sbyte = (lax.bitcast_convert_type(scale, jnp.int32) >> 23) & 0xFF
        # word (g, slot): bytes = dims 4g..4g+3; group 32: the scale byte.
        qt = jnp.transpose(jnp.broadcast_to(q, (8, 128)))[:, 0:1]  # [128, 1] dims
        grp = [qt[4 * g: 4 * g + 4] for g in range(32)]
        col = jnp.concatenate(
            [g[0:1] | (g[1:2] << 8) | (g[2:3] << 16) | (g[3:4] << 24) for g in grp]
            + [sbyte] + [jnp.zeros((1, 1), jnp.int32)] * 7, axis=0)  # [40, 1]
        tile = (slot // 128) * 128
        dst = kv_hbm.at[page, pl.ds(0, 40), pl.ds(tile, 128)]
        copy(dst, scr.aux, 3, False)
        copy(dst, scr.aux, 3, True)
        row = lax.broadcasted_iota(jnp.int32, (40, 1), 0)
        keep = (lane != slot % 128) | (row >= 33)
        scr.aux[...] = jnp.where(keep, scr.aux[...], col)
        for wait in (False, True):
            copy(scr.aux, dst, 3, wait)

    lax.fori_loop(0, num_req, req_body, 0)


def _kernel(cfg, shared, nreq, seq, st_pt, kv_pt, states, norm, cos_sin, *rest):
    n = (1 if shared else 2) + (cfg.mode == "csa")
    outs, scratch = rest[n:2 * n], rest[2 * n:]
    state_hbm, kv_hbm = (outs[0], outs[0]) if shared else (outs[0], outs[1])
    rope_hbm = outs[-1] if cfg.mode == "csa" else None
    compress_store_step(cfg, num_req=nreq[0], seq_lens=seq, st_pt=st_pt, kv_pt=kv_pt,
                        states_ref=states, norm_ref=norm, cos_sin_hbm=cos_sin,
                        state_hbm=state_hbm, kv_hbm=kv_hbm, rope_hbm=rope_hbm,
                        scr=CompressScratch(*scratch))


def compress_store(
    mode: str,
    states,  # f32 [R, q_len, 2 * state_width]: kv | score + ape per row
    seq_lens,  # int32 [R]
    state_host,  # int32 [N, rows, L]
    state_table,  # int32 [R, SP]
    kv_cache,  # see compress_store_step
    kv_table,  # int32 [R, KP]
    norm_w,  # f32 [head_dim]
    cos_sin_cache,  # f32 [max_pos, 64]
    rope_cache=None,  # csa only
    *,
    state_block_size: int,
    shared_state: bool = False,  # the states live in kv_cache (csa / idx serving)
    rms_eps: float = 1e-6,
    interpret=False,
):
    """Standalone wrapper: returns the updated caches, each aliased to its
    input: (state_host, kv_cache[, rope_cache]), or (kv_cache[, rope_cache]) with
    `shared_state` (state_host is then ignored; the states use kv_cache rows)."""
    R, T, _ = states.shape
    kv_page_tokens = {"hca": kv_cache.shape[1] // 2, "csa": kv_cache.shape[1],
                      "idx": kv_cache.shape[2]}[mode]
    cfg = CompressConfig(mode=mode, max_req=R, state_block_size=state_block_size,
                         state_pages_per_req=state_table.shape[1],
                         kv_pages_per_req=kv_table.shape[1], kv_page_tokens=kv_page_tokens,
                         state_lanes=(kv_cache if shared_state else state_host).shape[2],
                         q_len=T, rms_eps=rms_eps)
    st8 = jnp.pad(states.astype(jnp.float32), ((0, 0), (0, 8 - T), (0, 0)))
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    caches = ([] if shared_state else [state_host]) + [kv_cache] + (
        [rope_cache] if mode == "csa" else [])
    n_in = 7 + len(caches)
    return pl.pallas_call(
        functools.partial(_kernel, cfg, shared_state),
        out_shape=[jax.ShapeDtypeStruct(c.shape, c.dtype) for c in caches],
        in_specs=[smem] * 4 + [vmem, vmem, hbm] + [hbm] * len(caches),
        out_specs=[hbm] * len(caches),
        input_output_aliases={7 + i: i for i in range(len(caches))},
        scratch_shapes=list(compress_scratch_shapes(cfg)),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"mk_compress_{mode}_r{R}",
    )(
        jnp.array([R], jnp.int32), seq_lens.astype(jnp.int32),
        state_table.astype(jnp.int32).reshape(-1), kv_table.astype(jnp.int32).reshape(-1),
        st8, norm_w.astype(jnp.float32).reshape(1, -1), cos_sin_cache.astype(jnp.float32),
        *caches,
    )[: n_in - 7]
