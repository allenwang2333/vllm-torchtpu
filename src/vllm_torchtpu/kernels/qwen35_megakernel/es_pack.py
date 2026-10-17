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
"""Pack per-expert FP32 scales into FP8 expert weight tiles (MKConfig.fuse_es / es_pad layouts).

TPU bitcasts a (32, 128)-tiled FP8 value to an (8, 128)-tiled f32 value by packing 4 consecutive
FP8 rows into each 32-bit sublane: f32 row k holds FP8 rows 4k .. 4k + 3, with bits 8b .. 8b + 7
(little-endian byte b) in row 4k + b. These helpers lay the scales out so that
`pltpu.bitcast(x, F32)` in `MoEBlock` recovers them.
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from .config import FP8, MKConfig, es_geom, es_right_cols


def f32_rows_to_fp8(v: jax.Array) -> jax.Array:
    """[..., n, N] f32 -> [..., 4 * n, N] FP8 whose `pltpu.bitcast(., F32)` view is v."""
    u = lax.bitcast_convert_type(v.astype(jnp.float32), jnp.uint32)
    b = jnp.stack([((u >> (8 * k)) & 0xFF).astype(jnp.uint8) for k in range(4)], axis=-2)
    return lax.bitcast_convert_type(
        b.reshape(*v.shape[:-2], 4 * v.shape[-2], v.shape[-1]), FP8
    )


def pack_rows(s: jax.Array) -> jax.Array:
    """[..., N] f32 scale rows -> [..., 32, N] FP8 ("bottom" layout: the row on all 8 sublanes)."""
    v = jnp.broadcast_to(
        s[..., None, :].astype(jnp.float32), (*s.shape[:-1], 8, s.shape[-1])
    )
    return f32_rows_to_fp8(v)


def pack_right(c: MKConfig, s2: jax.Array) -> jax.Array:
    """[..., H] f32 down scales -> [..., IR, es_right_cols(c)] FP8 ("right" layout): lanes
    g * 128 .. g * 128 + 127 sit at `es_right_slot(c, g)` of the block's [IR // 4, cols] f32
    view (group g = tile * (IR // 4) + row)."""
    nrow, cols = c.IR // 4, es_right_cols(c)
    ncb, ng = cols // 128, c.H // 128
    lead = s2.shape[:-1]
    pad = [(0, 0)] * len(lead) + [(0, (nrow * ncb - ng) * 128)]
    v = jnp.pad(s2.astype(jnp.float32), pad).reshape(*lead, ncb, nrow, 128)
    v = jnp.swapaxes(v, -3, -2).reshape(*lead, nrow, cols)
    return f32_rows_to_fp8(v)


def _cat_fp8(xs, axis: int) -> jax.Array:
    u8 = [lax.bitcast_convert_type(x, jnp.uint8) for x in xs]
    return lax.bitcast_convert_type(jnp.concatenate(u8, axis=axis), FP8)


def fuse_e13(c: MKConfig, e13: jax.Array, s13: jax.Array) -> jax.Array:
    """e13 [..., H, 2 * IR] FP8 + s13 [..., 2 * IR] f32 -> [..., H + 32, 2 * IR] FP8 (fuse_es:
    packed scale rows; es_pad: zero rows); unchanged without either."""
    x13, _, _ = es_geom(c)
    if not x13:
        return e13
    if c.fuse_es:
        rows = pack_rows(s13)
    else:
        rows = jnp.zeros((*e13.shape[:-2], x13, e13.shape[-1]), FP8)
    return _cat_fp8([e13, rows], axis=-2)


def fuse_e2(c: MKConfig, e2: jax.Array, s2: jax.Array) -> jax.Array:
    """e2 [..., IR, H] FP8 + s2 [..., H] f32 -> [..., IR + 32, H] ("bottom") or
    [..., IR, H + es_right_cols(c)] ("right") FP8 (fuse_es: packed scales; es_pad: zeros)."""
    _, x2r, x2c = es_geom(c)
    if not (x2r or x2c):
        return e2
    if c.fuse_es:
        blk = pack_right(c, s2) if x2c else pack_rows(s2)
    else:
        blk = jnp.zeros((*e2.shape[:-2], x2r or e2.shape[-2], x2c or e2.shape[-1]), FP8)
    return _cat_fp8([e2, blk], axis=-1 if x2c else -2)


def expert_dma_bytes(c: MKConfig) -> int:
    """HBM->VMEM bytes DMA'd per listed single-tile expert (e13 + e2 tiles, plus the es row)."""
    x13, x2r, x2c = es_geom(c)
    b = (c.H + x13) * 2 * c.IR + (c.IR + x2r) * (c.H + x2c)
    if not c.fuse_es:
        b += 4 * (2 * c.IR + c.H)
    return b
