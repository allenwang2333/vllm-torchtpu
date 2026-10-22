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
"""shard_map step and operand layouts of the dp (DP4TP2 attention / EP8 MoE) megakernel.

Every operand is per rank (P("tp") on a leading rank axis): rank r = 2g + c serves chip g's
sequences, so positions, hidden rows, page table, checkpoint indices and read offsets are chip g's
(the same on both cores), and the routing bias covers the rows of all chips (chip g's at
g * BW ..). The per-core mixer weights stack the TP8 slices of virtual ranks 4c .. 4c + 3
(`dp_weights_from_tp8`).
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
from jax.sharding import PartitionSpec as P

from .config import W_NAMES, MKConfig, Tiles, smalls_layout
from .megakernel import build

MIXER_KINDS = ("lin_in", "lin_out", "att_in", "att_out")
# smalls segments of the mixer (per virtual rank); the rest belongs to the MoE phase and the
# layer norms (per physical rank)
MIXER_SMALLS = ("in_s", "out_s", "conv", "alog", "dtb", "gnorm", "qn", "kn")


def virtual_ranks(r: int, nv: int = 4) -> list[int]:
    """TP8 ranks whose mixer slices physical rank r runs: 4 * core .. 4 * core + 3."""
    core = r % 2
    return [core * nv + v for v in range(nv)]


def dp_weights_from_tp8(c: MKConfig, w: dict) -> dict:
    """TP8 per-rank weights (leading rank axis, numpy) -> the dp layout of the same model.

    Mixer streams: rank r's [n_layers * NV, ...] = for each layer, the slices of its virtual
    ranks. smalls: [R, L, NV, 1, S], row v = virtual rank v's, with the MoE / norm segments of
    rank r. Router, shared expert, experts and scales stay rank r's (EP over the 8 ranks)."""
    nv, R = c.NV, c.R
    lay, _ = smalls_layout(c)
    out = {}
    for k in W_NAMES:
        x = w[k]
        if k in MIXER_KINDS:
            x = np.asarray(x)
            stk = np.stack([np.stack([x[q] for q in virtual_ranks(r, nv)], axis=1)
                            for r in range(R)])  # [R, n_layers, NV, ...]
            out[k] = stk.reshape((R, x.shape[1] * nv) + x.shape[2:])
        elif k == "smalls":
            x = np.asarray(x)  # [R, L, 1, S]
            sm = np.stack([np.stack([x[q] for q in virtual_ranks(r, nv)], axis=1)
                           for r in range(R)])  # [R, L, NV, 1, S]
            for name, (o, n) in lay.items():
                if name not in MIXER_SMALLS:
                    sm[:, :, :, :, o:o + n] = x[:, :, None, :, o:o + n]
            out[k] = sm
        else:
            out[k] = x
    return out


def make_dp_step(
    mesh,
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    interpret=False,
    forced_routing=False,
    skip=frozenset(),
    scopes=False,
):
    """`hid, raw = step(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets,
    [rbias])`, all per rank on a leading rank axis. The per-row / per-sequence operands cover the
    chip's microbatch grid, BW = n_mb * mb_rows rows (NSW = BW / T sequences; those past the
    chip's B rows are inactive: checkpoint index 0, position 0): pos [R, BW], hidden [R * M, H],
    page_table [R, NSW, P], ckpt_indices [R, 3, NV * NSW, K] (virtual rank v's sequences at
    v * NSW ..), read_offsets [R, 3, NSW]; weights as make_pooled_step (leading R), raw_pools
    NATT x [R * num_pages, ...], rbias [R * L, M_rb, E]: chip g's rows at g * BW .., M_rb the
    all-chips rows rounded up to 128."""
    assert c.dp > 1 and c.wide, (c.dp, c.mb_rows)
    c = dataclasses.replace(c, paged_kv=True, fused_pool_io=True, act_mode="from_ckpt",
                            ckpt_2d=False, roff_1d=False, ckpt_k=c.ckpt_k or c.T)
    call, info = build(c, tiles, interpret=interpret, forced_routing=forced_routing, skip=skip,
                       scopes=scopes, pool_gather=False)
    A = P(mesh.axis_names[0])

    def local(pos, hidden, weights, raw, pt, ck, ro, *rb):
        w = [tuple(x[0] for x in weights[k]) if isinstance(weights[k], tuple) else weights[k][0]
             for k in W_NAMES]
        out = call(pos[0], hidden, *w, tuple(raw), pt[0], ck[0], ro[0], *rb)
        if bool(interpret):
            hid, raw_out = out
            return hid, tuple(raw_out)
        hid = out[0] if isinstance(out, (tuple, list)) else out
        return hid, tuple(raw)

    def step(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets, *rb):
        ws = {k: tuple(A for _ in weights[k]) if isinstance(weights[k], tuple) else A
              for k in weights}
        raw_spec = tuple(A for _ in raw_pools)
        in_specs = (A, A, ws, raw_spec, A, A, A) + ((A,) if forced_routing else ())
        return jax.shard_map(
            local, mesh=mesh, in_specs=in_specs, out_specs=(A, raw_spec), check_vma=False
        )(pos, hidden, weights, tuple(raw_pools), page_table, ckpt_indices, read_offsets, *rb)

    return step, info
