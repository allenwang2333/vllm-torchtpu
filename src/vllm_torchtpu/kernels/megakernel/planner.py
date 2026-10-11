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
"""Build-time planner for weight tiling, static stream schedules, and smalls layout."""

from __future__ import annotations

from typing import Any
from vllm_torchtpu.kernels.megakernel.ops import BF16, F32, FP8, I32, rup
from vllm_torchtpu.kernels.megakernel.spec import ModelSpec, Spec, SwitchSlot, Tiles

DENSE_KINDS = ("lin_in", "lin_out", "att_in", "att_out", "router", "sh13", "sh2")
EXPERT_KINDS = ("e13", "e2")


def stream_kinds(c: Any) -> dict[str, tuple[int, int, Any]]:
    """Streamed weight kinds: (K, N, dtype) of the per-rank matrix."""
    return {
        "lin_in": (c.H, c.NLI, FP8),
        "lin_out": (c.Z, c.H, FP8),
        "att_in": (c.H, c.NAI, FP8),
        "att_out": (c.nh * c.HD, c.H, FP8),
        "router": (c.H, c.E, BF16),
        "sh13": (c.H, 2 * c.ISR, FP8),
        "sh2": (c.ISR, c.H, FP8),
        "e13": (c.H, 2 * c.IR, FP8),
        "e2": (c.IR, c.H, FP8),
    }


def smalls_layout(c: Any) -> tuple[dict[str, tuple[int, int]], int]:
    """Map `name -> (lane_offset, width)` in the per-layer f32 row of small parameters."""
    segs = [
        ("in_norm", c.H),
        ("post_norm", c.H),
        ("in_s", max(c.NLI, c.NAI)),
        ("out_s", c.H),
        ("sh13_s", 2 * c.ISR),
        ("sh2_s", c.H),
        ("sh_gate", c.H),
        ("conv", 4 * c.QKV),
        ("alog", 128),
        ("dtb", 128),
        ("gnorm", c.VD),
        ("qn", c.HD),
        ("kn", c.HD),
    ]
    if not c.moe_tp and c.e_slots > c.ER:
        segs.append(("ecopy", c.E))
    lay: dict[str, tuple[int, int]] = {}
    o = 0
    for name, n in segs:
        lay[name] = (o, n)
        o += rup(n, 128)
    return lay, o


def encode_ecopy(c: Any, cores) -> list[float]:
    """Pack per-layer redundant expert copy descriptors into an [E] f32 row for smalls['ecopy']."""
    er, ew = c.ER, c.EW
    extra = ew - er
    out = [0.0] * c.E
    if c.moe_tp or extra <= 0:
        return out
    vid = {int(g): r0 * er + s0 for r0, lst in enumerate(cores) for s0, g in enumerate(lst[:er])}
    for r1, lst in enumerate(cores):
        for k, g in enumerate(lst[er:ew]):
            v = vid.get(int(g))
            if v is not None and v // er != r1:
                out[v] = float(1 + r1 * extra + k)
    return out


def plan(c: Any, tiles: Tiles, model_spec: ModelSpec | None = None) -> dict[str, Any]:
    """Compute K-tile sizes, tile counts, and static stream segment schedules."""
    kinds = stream_kinds(c)
    tk = {
        k: tiles.tk(K, N, dt, tiles.ebank if k in EXPERT_KINDS else tiles.bank)
        for k, (K, N, dt) in kinds.items()
    }
    nt = {k: kinds[k][0] // tk[k] for k in kinds}
    t_gdn = nt["lin_in"] + nt["lin_out"] + nt["router"] + nt["sh13"] + nt["sh2"]
    t_att = nt["att_in"] + nt["att_out"] + nt["router"] + nt["sh13"] + nt["sh2"]
    t_block = (c.att_every - 1) * t_gdn + t_att

    # Build static segment tables for GDN-branch layers and Attention-branch layers
    def _build_segs(branch_kinds: tuple[tuple[str, int], ...]):
        segs = []
        cur = 0
        for kind, ordi in branch_kinds:
            nxt = cur + nt[kind]
            segs.append((kind, cur, nxt, ordi))
            cur = nxt
        return segs

    segs_gdn = _build_segs(
        (("lin_in", 0), ("lin_out", 0), ("router", 1), ("sh13", 1), ("sh2", 1))
    )
    segs_att = _build_segs(
        (("att_in", 0), ("att_out", 0), ("router", 1), ("sh13", 1), ("sh2", 1))
    )

    lay, S = smalls_layout(c)
    return {
        "kinds": kinds,
        "tk": tk,
        "nt": nt,
        "t_gdn": t_gdn,
        "t_att": t_att,
        "t_block": t_block,
        "segs_gdn": segs_gdn,
        "segs_att": segs_att,
        "smalls_layout": lay,
        "smalls_width": S,
        "model_spec": model_spec,
    }


def weight_shapes(c: Any, tiles: Tiles) -> dict[str, Any]:
    """Per-rank HBM operand shapes (streamed weights pre-tiled [.., n_tiles, tk, N])."""
    p = plan(c, tiles)
    tk, nt = p["tk"], p["nt"]
    S = p["smalls_width"]
    ngdn = max(1, c.NGDN)

    def tiled(kind: str, lead: tuple[int, ...]):
        _, N, dt = p["kinds"][kind]
        return ((*lead, nt[kind], tk[kind], N), dt)

    if c.stack_e:
        e13 = tiled("e13", (c.L, c.EW))
        e2 = tiled("e2", (c.L, c.EW))
    else:
        e13 = tuple(Spec(*tiled("e13", (c.EW,))) for _ in range(c.L))
        e2 = tuple(Spec(*tiled("e2", (c.EW,))) for _ in range(c.L))
    return {
        "lin_in": tiled("lin_in", (ngdn,)),
        "lin_out": tiled("lin_out", (ngdn,)),
        "att_in": tiled("att_in", (c.NATT,)),
        "att_out": tiled("att_out", (c.NATT,)),
        "router": tiled("router", (c.L,)),
        "sh13": tiled("sh13", (c.L,)),
        "sh2": tiled("sh2", (c.L,)),
        "e13": e13,
        "e2": e2,
        "es": ((c.L, c.EW, 1, 2 * c.IR + c.H), F32),
        "smalls": ((c.L, 1, S), F32),
        "final_norm": ((1, c.H), F32),
    }


def state_shapes(c: Any) -> dict[str, Any]:
    """Per-rank input state shapes."""
    ngdn = max(1, c.NGDN)
    conv = (ngdn, c.CR, c.QKV) if c.T == 1 else (ngdn, c.NS, c.HR, c.QKV)
    if c.paged_kv and c.fused_pool_io:
        G_PER_ATT = max(1, c.att_every - 1)
        K_ckpt = c.ckpt_k if c.ckpt_k > 0 else c.T
        ckpt_shape = (c.NS, K_ckpt) if c.ckpt_2d else (G_PER_ATT, c.NS, K_ckpt)
        roff_shape = (c.NS,) if c.roff_1d else (G_PER_ATT, c.NS)
        st = {
            "raw": tuple(Spec((c.num_pages, 2, 64, 4, c.page_size), FP8) for _ in range(c.NATT)),
            "page_table": Spec((c.NS, c.max_pages_per_seq), I32),
            "ckpt_indices": Spec(ckpt_shape, I32),
            "read_offsets": Spec(roff_shape, I32),
        }
        if c.act_mode == "ref":
            st["active_seqs"] = Spec((c.NS,), I32)
        return st
    if c.paged_kv:
        return {
            "conv": (conv, F32),
            "rec": ((ngdn, c.NS * c.vh, c.KD, c.VD), BF16),
            "raw": tuple(Spec((c.num_pages, 2, 64, 4, c.page_size), FP8) for _ in range(c.NATT)),
            "page_table": Spec((c.NS, c.max_pages_per_seq), I32),
        }
    return {
        "conv": (conv, F32),
        "rec": ((ngdn, c.NS * c.vh, c.KD, c.VD), BF16),
        "kc": ((c.NATT, c.NS, c.ctx, c.HD), BF16),
        "vc": ((c.NATT, c.NS, c.ctx, c.HD), BF16),
    }


def kernel_out_shapes(c: Any) -> dict[str, Any]:
    """Per-rank output tensor shapes."""
    if c.paged_kv and c.fused_pool_io:
        return {
            "hidden": ((c.M, c.H), BF16),
            "raw": tuple(Spec((c.num_pages, 2, 64, 4, c.page_size), FP8) for _ in range(c.NATT)),
        }
    sh = state_shapes(c)
    ngdn = max(1, c.NGDN)
    kv_dt = FP8 if c.paged_kv else F32
    return {
        "hidden": ((c.M, c.H), BF16),
        "conv": sh["conv"],
        "rec": ((ngdn, c.B * c.vh, c.KD, c.VD), BF16),
        "knew": ((c.NATT, c.MR, c.HD), kv_dt),
        "vnew": ((c.NATT, c.MR, c.HD), kv_dt),
    }

