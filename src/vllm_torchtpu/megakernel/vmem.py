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
"""Core VMEM/SMEM/Semaphore allocator and `StreamParams` / `CommParams` builders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from . import pool_alias
from .ops import BF16, F32, rup
from .planner import Plan, plan
from .spec import ExecSpec, ModelSpec, ScratchReq

U8 = jnp.uint8


@dataclass(frozen=True)
class StreamParams:
    """Core weight-streaming ring parameters derived from `Plan` and `ExecSpec`."""

    L: int
    MB: int
    NBD: int
    NBE: int
    BROWS: int
    EBROWS: int
    tk: dict[str, int]
    nt: dict[str, int]
    kinds: dict[str, tuple[int, int, Any]]
    static_kinds: tuple[str, ...]
    dynamic_kinds: tuple[str, ...]
    vshape: dict[str, tuple[int, int]]
    vdtype: dict[str, Any]
    period: int
    branch_counts: tuple[int, ...]
    branch_tiles: tuple[int, ...]
    branch_segs: tuple[list[tuple[str, int, int, int]], ...]
    t_cycle: int
    T_TOTAL: int
    block_branch_idx: dict[str, int | None]
    lay: dict[str, tuple[int, int]]
    S: int
    typed: bool

    def layer_start_tile(self, lyr: Any) -> Any:
        """Return global static stream tile index at the start of layer `lyr`."""
        if len(self.branch_counts) == 1:
            return lyr * self.branch_tiles[0]
        c0 = self.branch_counts[0]
        t0, t1 = self.branch_tiles[0], self.branch_tiles[1]
        rem = lyr % self.period
        return (
            (lyr // self.period) * self.t_cycle
            + jnp.minimum(rem, c0) * t0
            + jnp.maximum(0, rem - c0) * t1
        )

    def block_ord(self, lyr: Any, block_name: str) -> Any:
        """Return layer index into `block_name` stacked weights at layer `lyr`."""
        b_idx = self.block_branch_idx.get(block_name)
        if b_idx is None or len(self.branch_counts) == 1:
            return lyr
        c_b = self.branch_counts[b_idx]
        start_b = sum(self.branch_counts[:b_idx])
        return (lyr // self.period) * c_b + jnp.clip(
            (lyr % self.period) - start_b, 0, c_b - 1
        )


@dataclass(frozen=True)
class CommParams:
    """Core all-reduce and inter-rank collective parameters."""

    R: int
    B: int
    M: int
    MR: int
    MB: int
    H: int
    PR: int
    PC: int
    HPC: int
    HIER: int
    RSAG: bool
    WAR: bool
    AR_R: int
    AR_C: int
    CW: int
    HC2: int
    AR_ROUNDS: tuple[int, ...]
    AR_PEERS: int


@dataclass(frozen=True)
class CorePlan:
    """Complete build-time plan combining `Plan`, `StreamParams`, and `CommParams`."""

    plan: Plan
    stream: StreamParams
    comm: CommParams
    blocks: dict[str, Any]
    phase_map: dict[str, str]
    vmem_bytes: int
    vmem_shapes: dict[str, tuple[tuple[int, ...], Any]]
    NACC: int
    NY: int
    KX2: int
    NMIX: int

    def block(self, name: str) -> Any:
        return self.blocks[name]


def _normalize_scratch_reqs(
    raw: dict[str, Any] | list[ScratchReq],
) -> list[ScratchReq]:
    if isinstance(raw, dict):
        return [ScratchReq(name=k, spec=v, phase="persistent") for k, v in raw.items()]
    return list(raw)


def _merge_scratch(
    scratch: dict[str, Any],
    phase_map: dict[str, str],
    reqs: list[ScratchReq],
) -> None:
    """Merge block `ScratchReq`s into `scratch`, overlaying compatible phases."""
    for r in reqs:
        if r.name not in scratch:
            scratch[r.name] = r.spec
            phase_map[r.name] = r.phase
            continue

        cur_spec = scratch[r.name]
        cur_phase = phase_map[r.name]
        if (
            r.phase != "persistent"
            and cur_phase != "persistent"
            and r.phase != cur_phase
            and hasattr(r.spec, "shape")
            and hasattr(cur_spec, "shape")
            and getattr(r.spec, "dtype", None) == getattr(cur_spec, "dtype", None)
            and getattr(r.spec, "memory_space", None)
            == getattr(cur_spec, "memory_space", None)
            and len(r.spec.shape) == len(cur_spec.shape)
        ):
            max_shape = tuple(max(a, b) for a, b in zip(cur_spec.shape, r.spec.shape))
            ms = getattr(cur_spec, "memory_space", pltpu.VMEM)
            scratch[r.name] = ms(max_shape, cur_spec.dtype)
            phase_map[r.name] = f"{cur_phase}|{r.phase}"
        elif r.spec == cur_spec and r.phase == cur_phase:
            continue
        else:
            raise ValueError(
                f"Conflicting scratch buffer {r.name!r}: "
                f"phase {r.phase!r} vs {cur_phase!r}"
            )


def build_scratch(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> tuple[CorePlan, dict[str, Any]]:
    """Build `CorePlan` and allocate all core and block VMEM scratch buffers."""
    if p is None:
        p = plan(model_spec, ex)
    g = model_spec.config
    tiles = ex.resolved_tiles

    R, B, M, MR, MB = ex.R, ex.B, ex.M, ex.MR, ex.MB
    H, L = g.H, g.L
    PC = min(1024, H)
    assert H % PC == 0, (H, PC)
    HPC = H // PC
    PR = rup(B * HPC, 16)

    tk, nt, kinds = p.tk, p.nt, p.kinds
    lay, S = p.smalls_layout, p.smalls_width
    NBD, NBE = tiles.nbd, tiles.nbe
    BROWS, EBROWS = tiles.bank // 128, tiles.ebank // 128

    NACC = p.shared_widths["acc"]
    NY = p.shared_widths["ybuf"]
    KX2 = p.shared_widths["xb2"]
    NMIX = p.shared_widths["mix"]
    typed = bool(ex.interpret)

    HIER = tiles.hier if R > 1 else 0
    RSAG = tiles.rsag and R > 1 and HIER == 0
    WAR = (ex.WIDE or MR * H <= PR * PC) if tiles.war < 0 else bool(tiles.war)
    AR_R = MR if WAR else PR
    AR_C = H if WAR else PC
    CW = AR_C // R
    HC2 = AR_C // 2
    assert not RSAG or CW % 128 == 0
    assert HIER in (0, 1, 2, 3) and (HIER == 0 or R == 8)
    assert HIER != 2 or HC2 % 128 == 0
    AR_ROUNDS = (
        (0, 1, 2) if HIER in (2, 3) else ((0, 1) if (RSAG or HIER == 1) else (0,))
    )
    AR_PEERS = 1 if HIER == 3 else (3 if HIER in (1, 2) else max(R - 1, 1))

    vshape = {k: (tk[k], kinds[k][1]) for k in kinds}
    vdtype = {k: kinds[k][2] for k in kinds}
    for k in kinds:
        assert pool_alias.storage_size(vshape[k], jnp.dtype(vdtype[k])) <= (
            tiles.ebank if k in p.dynamic_kinds else tiles.bank
        ), k

    scratch: dict[str, Any] = {
        "dsem": pltpu.SemaphoreType.DMA((NBD,)),
        "sm": pltpu.VMEM((2, 1, S), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        "res": pltpu.VMEM((MB, H), F32),
        "xb": pltpu.VMEM((MB, H), BF16),
        "xb2": pltpu.VMEM((MB, KX2), BF16),
        "acc": pltpu.VMEM((MB, NACC), F32),
        "ybuf": pltpu.VMEM((MB, NY), F32),
        "part": pltpu.VMEM((MB, H), F32),
        "mix": pltpu.VMEM((MB, NMIX), F32),
        "send": pltpu.VMEM(
            (3, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, R, AR_R, CW)
            if RSAG
            else (2, AR_R, AR_C),
            BF16,
        ),
        "recv": pltpu.VMEM(
            (3, 2, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, 2, AR_R, AR_C)
            if HIER == 1
            else (2, R, AR_R, CW)
            if RSAG
            else (2, R, AR_R, AR_C),
            BF16,
        ),
        "ar_ssem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        "ar_rsem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        "fn": pltpu.VMEM((1, H), F32),
        "fn_sem": pltpu.SemaphoreType.DMA(()),
    }
    if p.dynamic_kinds:
        scratch["esem"] = pltpu.SemaphoreType.DMA((NBE,))

    if RSAG:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, CW), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, R, AR_R, CW), BF16)
    elif HIER == 1:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, AR_C), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, AR_C), BF16)
    elif HIER == 2:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, HC2), BF16)
        scratch["s2_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["s2_recv"] = pltpu.VMEM((2, 2, AR_R, HC2), BF16)

    phase_map: dict[str, str] = {k: "persistent" for k in scratch}

    block_plans: dict[str, Any] = dict(p.block_params)
    for b in model_spec.all_blocks():
        reqs = _normalize_scratch_reqs(b.scratch(g, ex, p))
        _merge_scratch(scratch, phase_map, reqs)

    if typed:
        for k in p.static_kinds:
            scratch["dring_" + k] = pltpu.VMEM((NBD,) + vshape[k], vdtype[k])
            phase_map["dring_" + k] = "persistent"
        for k in p.dynamic_kinds:
            scratch["ering_" + k] = pltpu.VMEM((NBE,) + vshape[k], vdtype[k])
            phase_map["ering_" + k] = "persistent"
    else:
        scratch["dpool"] = pltpu.VMEM((NBD * BROWS, 128), U8)
        phase_map["dpool"] = "persistent"
        if p.dynamic_kinds:
            scratch["epool"] = pltpu.VMEM((NBE * EBROWS, 128), U8)
            phase_map["epool"] = "persistent"

    vmem_bytes = 0
    vmem_shapes: dict[str, tuple[tuple[int, ...], Any]] = {}
    for k, v in scratch.items():
        if (
            hasattr(v, "shape")
            and "VMEM" in str(getattr(v, "memory_space", "")).upper()
        ):
            vmem_bytes += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
            vmem_shapes[k] = (tuple(v.shape), v.dtype)

    vmem_limit_bytes = tiles.vmem_mib * 1024 * 1024
    if vmem_bytes > vmem_limit_bytes:
        breakdown = sorted(
            (
                (math.prod(sh) * jnp.dtype(dt).itemsize, k, sh, dt)
                for k, (sh, dt) in vmem_shapes.items()
            ),
            reverse=True,
        )
        top_str = ", ".join(
            f"{k}={sz / (1024 * 1024):.2f}MiB {sh} {jnp.dtype(dt).name}"
            for sz, k, sh, dt in breakdown[:10]
        )
        vmem_mib = vmem_bytes / (1024 * 1024)
        raise ValueError(
            f"VMEM scratch allocation ({vmem_mib:.2f} MiB = {vmem_bytes:,} B) "
            f"exceeds tiles.vmem_mib budget "
            f"({tiles.vmem_mib} MiB = {vmem_limit_bytes:,} B). "
            f"Top buffers: {top_str}"
        )

    sp = StreamParams(
        L=L,
        MB=MB,
        NBD=NBD,
        NBE=NBE,
        BROWS=BROWS,
        EBROWS=EBROWS,
        tk=tk,
        nt=nt,
        kinds=kinds,
        static_kinds=p.static_kinds,
        dynamic_kinds=p.dynamic_kinds,
        vshape=vshape,
        vdtype=vdtype,
        period=p.period,
        branch_counts=p.branch_counts,
        branch_tiles=p.branch_tiles,
        branch_segs=p.branch_segs,
        t_cycle=p.t_cycle,
        T_TOTAL=p.T_TOTAL,
        block_branch_idx=p.block_branch_idx,
        lay=lay,
        S=S,
        typed=typed,
    )
    cp = CommParams(
        R=R,
        B=B,
        M=M,
        MR=MR,
        MB=MB,
        H=H,
        PR=PR,
        PC=PC,
        HPC=HPC,
        HIER=HIER,
        RSAG=RSAG,
        WAR=WAR,
        AR_R=AR_R,
        AR_C=AR_C,
        CW=CW,
        HC2=HC2,
        AR_ROUNDS=AR_ROUNDS,
        AR_PEERS=AR_PEERS,
    )
    core_plan = CorePlan(
        plan=p,
        stream=sp,
        comm=cp,
        blocks=block_plans,
        phase_map=phase_map,
        vmem_bytes=vmem_bytes,
        vmem_shapes=vmem_shapes,
        NACC=NACC,
        NY=NY,
        KX2=KX2,
        NMIX=NMIX,
    )
    return core_plan, scratch
