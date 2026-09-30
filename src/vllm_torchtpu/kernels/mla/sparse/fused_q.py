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
"""Fused sparse-MLA prefill q: q_b output -> the sparse kernel's packed q.

One TensorCore pass replaces the XLA q chain (slices, RoPE rolls + copies,
the RoPE fusion, the absorbed W_UK bmm and the [H, T, 640] -> [T, H, 640]
transpose) with a kernel that reads the q_b projection output in its natural
T-major [T, H * (nope + rope)] layout and writes the packed [T, H * 640] q the
sparse MLA kernel consumes, also T-major, so no data-format pass runs on
either side.

Per head h and T tile:
  out[:, 640h : 640h + 512]  = x_nope @ W_UK_T[h]          (bf16 in, f32 acc)
  out[:, 640h + 512 : +64]   = gptj_rope(x_pe, cos, sin)
  out[:, 640h + 576 : +64]   = 0
which is `_packed_sparse_q(layer, q_nope, gptj_rope(q_pe))` column for column.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128


@dataclasses.dataclass(frozen=True)
class FusedQConfig:
    # "roll": lane rolls + even/odd select (XLU); "mxu": one [128, 256]
    # 0/1 matmul yields (pe, swap(pe)) exactly.
    rope: str = "mxu"
    # "a": products and sum in f32, one bf16 rounding.
    # "b": each product rounded to bf16, bf16 sum (per-op bf16 semantics).
    rounding: str = "b"
    # Contraction depth of the W_UK dot: 192 (the real nope width) or 256
    # (the whole per-head q row against a zero-padded weight).
    k: int = 192
    t_tile: int = 256
    head_block: int = 16
    # Output layout. "flat": [T, H * 640] tiles over (T, lanes) -- its reshape
    # to [T, H, 640] is a (T, H) sublane relayout XLA runs as an extra SC
    # data-format pass. The others write [T, H, 640] tiles over (H, lanes)
    # directly: "swap" (jnp.swapaxes of the stacked [hb, tT, 640] heads),
    # "einshape" (pltpu.einshape htd->thd), "sst" (per-head o_ref[:, h, :]
    # sublane-strided stores). "pack*" first packs head pairs (2k, 2k+1) into
    # one u32 row -- the bf16 (2, 1) sublane packing of the [T, H, 640] tile --
    # then writes through a u32 view of the block: "pack" (swapaxes),
    # "packes" (einshape), "psst" (per-pair 32-bit sublane-strided stores).
    # Default "pack": fastest standalone at T=1024, H=64 (84.6 us vs 88.2 swap,
    # 106-112 einshape, 125.9 flat + its relayout; copy floor 63.8), probe 36 j3.
    out: str = "pack"

    @classmethod
    def parse(cls, opts: str | None) -> "FusedQConfig":
        cfg = {}
        for item in (opts or "").split(","):
            item = item.strip()
            if not item:
                continue
            key, val = item.split("=")
            key = {"round": "rounding", "tt": "t_tile", "hb": "head_block"}.get(key, key)
            field_type = cls.__dataclass_fields__[key].type
            cfg[key] = int(val) if field_type in (int, "int") else val
        return cls(**cfg)


def rope_swap_matrix(dtype=jnp.bfloat16) -> jax.Array:
    """E [128, 256]: for a lane block (nope_tail[64] | pe[64]),
    (x @ E)[:, 0:64] = pe and (x @ E)[:, 128:192] = swap_pairs(pe)."""
    e = jnp.zeros((LANES, 2 * LANES), dtype)
    idx = jnp.arange(64)
    e = e.at[64 + idx, idx].set(1)
    e = e.at[64 + (idx ^ 1), LANES + idx].set(1)
    return e


def _pack_pair(lo: jax.Array, hi: jax.Array) -> jax.Array:
    """bf16 rows (lo, hi) -> u32 with lo in bits 0:16, hi in 16:32 (the layout
    `pltpu.bitcast` / a u32 ref view gives two adjacent bf16 rows)."""
    lo32 = jax.lax.bitcast_convert_type(lo.astype(jnp.float32), jnp.uint32)
    hi32 = jax.lax.bitcast_convert_type(hi.astype(jnp.float32), jnp.uint32)
    return (lo32 >> 16) | (hi32 & jnp.uint32(0xFFFF0000))


def _kernel(
    q_ref, c_ref, s_ref, w_ref, e_ref, o_ref, *scratch, cfg, nope, rope, lkv, width, hb, f32_dot
):
    head_in = nope + rope

    def dot(a, b):
        # The CPU interpreter has no bf16 x bf16 -> f32 dot; bf16 -> f32 is
        # exact, so upcasting only changes the accumulation order there.
        if f32_dot:
            a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    if cfg.k == 256:
        (w256,) = scratch

        @pl.when(pl.program_id(1) == 0)
        def _():
            w256[:, :nope, :] = w_ref[...]
            w256[:, nope:, :] = jnp.zeros((hb, head_in - nope, lkv), w256.dtype)

    c = c_ref[...].astype(jnp.float32)
    s = s_ref[...].astype(jnp.float32)
    lane = jax.lax.broadcasted_iota(jnp.int32, c.shape, 1)
    keep = lane < rope
    rows = []
    packed = []
    for h in range(hb):
        base = h * head_in
        if cfg.k == 256:
            acc = dot(q_ref[:, base : base + head_in], w256[h])
        else:
            acc = dot(q_ref[:, base : base + nope], w_ref[h])
        # The 128-lane block holding pe in its upper half.
        hi = q_ref[:, base + head_in - LANES : base + head_in]
        if cfg.rope == "mxu":
            y = dot(hi, e_ref[...])
            pe, sw = y[:, :LANES], y[:, LANES:]
        else:
            v = pltpu.roll(hi.astype(jnp.float32), LANES - rope, axis=1)
            even = (lane % 2) == 0
            sw = jnp.where(
                even, pltpu.roll(v, LANES - 1, axis=1), pltpu.roll(v, 1, axis=1)
            )
            pe = v
        if cfg.rounding == "a":
            rot = pe * c + sw * s
        else:
            pc = (pe * c).astype(jnp.bfloat16).astype(jnp.float32)
            ps = (sw * s).astype(jnp.bfloat16).astype(jnp.float32)
            rot = pc + ps
        rot = jnp.where(keep, rot, 0.0)
        if cfg.out == "flat":
            ob = h * width
            o_ref[:, ob : ob + lkv] = acc.astype(o_ref.dtype)
            o_ref[:, ob + lkv : ob + width] = rot.astype(o_ref.dtype)
        elif cfg.out == "sst":
            o_ref[:, h, :lkv] = acc.astype(o_ref.dtype)
            o_ref[:, h, lkv:] = rot.astype(o_ref.dtype)
        elif cfg.out.startswith("p"):
            rows.append((acc.astype(o_ref.dtype), rot.astype(o_ref.dtype)))
            if h % 2:
                (a0, r0), (a1, r1) = rows.pop(-2), rows.pop(-1)
                packed.append(
                    jnp.concatenate([_pack_pair(a0, a1), _pack_pair(r0, r1)], axis=-1)
                )
        else:
            rows.append(
                jnp.concatenate([acc.astype(o_ref.dtype), rot.astype(o_ref.dtype)], axis=-1)
            )
    if packed and f32_dot:
        # Interpreter: no ref bitcast; unpack with pltpu.bitcast's convention
        # (row 2k in the low half) and store bf16.
        u = jax.lax.bitcast_convert_type(jnp.stack(packed, axis=1), o_ref.dtype)
        o_ref[...] = jnp.swapaxes(u, -1, -2).reshape(o_ref.shape)
    elif packed:
        o32 = o_ref.bitcast(jnp.uint32)  # (tT, hb // 2, width)
        if cfg.out == "psst":
            for k, row in enumerate(packed):
                o32[:, k, :] = row
        elif cfg.out == "packes":
            o32[...] = pltpu.einshape("htd->thd", jnp.stack(packed, axis=0))
        else:
            o32[...] = jnp.swapaxes(jnp.stack(packed, axis=0), 0, 1)
    if rows:
        hm = jnp.stack(rows, axis=0)  # [hb, tT, width]
        if cfg.out == "einshape":
            o_ref[...] = pltpu.einshape("htd->thd", hm)
        else:
            o_ref[...] = jnp.swapaxes(hm, 0, 1)


@functools.partial(jax.jit, static_argnames=("cfg", "interpret"))
def fused_sparse_q(
    q: jax.Array,  # [T, H * (nope + rope)] bf16, the q_b projection output
    c: jax.Array,  # [T, 128]: repeat_interleave(cos, 2) in lanes [0, rope), 0 after
    s: jax.Array,  # [T, 128]: interleave(-sin, sin) in lanes [0, rope), 0 after
    w_uk_t: jax.Array,  # [H, nope, lkv]
    *,
    cfg: FusedQConfig = FusedQConfig(),
    interpret: bool = False,
) -> jax.Array:
    """-> packed sparse-MLA q, [T, H, align_to(lkv + rope, 128)]."""
    t = q.shape[0]
    h_total, nope, lkv = w_uk_t.shape
    head_in = q.shape[1] // h_total
    rope = head_in - nope
    width = -(-(lkv + rope) // LANES) * LANES
    assert head_in % LANES == 0 and lkv % LANES == 0, (head_in, lkv)
    assert rope <= LANES // 2 and width == lkv + LANES, (rope, width)
    assert c.shape == s.shape == (t, LANES), (c.shape, t)
    hb = min(cfg.head_block, h_total)
    assert h_total % hb == 0, (h_total, hb)
    if cfg.out != "flat" and (
        (hb % 8 and hb != h_total) or (cfg.out.startswith("p") and hb % 2)
    ):
        # A (tT, hb, width) out block needs hb % 8 == 0 or hb == H (Mosaic's
        # second-minor rule), and the pack variants pair heads. The flat
        # (tT, hb * width) block is legal for any hb.
        cfg = dataclasses.replace(cfg, out="flat")
    tt = min(cfg.t_tile, t)
    grid = (h_total // hb, pl.cdiv(t, tt))
    scratch = []
    if cfg.k == 256:
        assert head_in == 256
        scratch.append(pltpu.VMEM((hb, head_in, lkv), w_uk_t.dtype))
    if cfg.out == "flat":
        out_spec = pl.BlockSpec((tt, hb * width), lambda g, i: (i, g))
        out_shape = jax.ShapeDtypeStruct((t, h_total * width), q.dtype)
    else:
        out_spec = pl.BlockSpec((tt, hb, width), lambda g, i: (i, g, 0))
        out_shape = jax.ShapeDtypeStruct((t, h_total, width), q.dtype)
    e = rope_swap_matrix(q.dtype)
    kernel = functools.partial(
        _kernel,
        cfg=cfg,
        nope=nope,
        rope=rope,
        lkv=lkv,
        width=width,
        hb=hb,
        f32_dot=interpret,
    )
    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((tt, hb * head_in), lambda g, i: (i, g)),
            pl.BlockSpec((tt, LANES), lambda g, i: (i, 0)),
            pl.BlockSpec((tt, LANES), lambda g, i: (i, 0)),
            pl.BlockSpec((hb, nope, lkv), lambda g, i: (g, 0, 0)),
            pl.BlockSpec((LANES, 2 * LANES), lambda g, i: (0, 0)),
        ],
        out_specs=out_spec,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        interpret=interpret,
        name="fused_sparse_q",
    )(q, c, s, w_uk_t, e)
    return out.reshape(t, h_total, width)
