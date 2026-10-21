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
"""`MoEBlock` build-time parameters and tunables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...exec_spec import NumericsClass


@dataclass(frozen=True)
class MoEParams:
    """Block-owned build-time parameters for `MoEBlock`."""

    H: int
    E: int
    TOPK: int
    IR: int
    ISR: int
    ER: int
    EW: int
    EXTRA: int
    CH: int
    T13: int
    T2: int
    TE: int
    NBE: int
    NSS: int
    PIPE: bool
    PIPE_EP: bool
    LAZY_MOE: bool
    N_MOE: int
    ECOPY: bool
    moe_tp: bool
    stack_e: bool
    topk_packed: bool
    topk_lanes: bool
    route_vec: bool
    r2_direct_store: bool
    unroll: bool
    # Paired expert-ring releases + unrolled refill loop (unrolled / PIPE_EP
    # expert paths) for buckets up to `MoETunables.moe_group_rows` rows.
    group_rel: bool
    ecopy_opt: int
    eps: float
    zero_centered_norm: bool


@dataclass(frozen=True)
class MoETunables:
    """`MoEBlock` knobs (`Tiles.blocks["moe"]`); `NUMERICS` classes each field."""

    pipe: bool = True  # Software-pipeline single-tile experts
    topk: bool = True  # One-reduction top-k picks on packed (value, lane) keys
    ecopy_opt: int = 0  # Redundant expert dedup optimization (0 or 1)
    # Unrolled / PIPE_EP experts: paired ring releases + unrolled refills up to
    # this many rows
    moe_group_rows: int = 40
    # Opt-in performance flags (off by default; named profiles turn them on).
    pipe_ep: bool = False  # Software-pipeline EP experts with T13 == 2 * T2
    r2_direct_store: bool = False  # Direct VMEM store on r2 tile 0
    # Vectorized MXU expert route packing (a distinct routing/Po2C policy that
    # supersedes `ecopy_opt` whenever its static gate in `MoEBlock.route` holds)
    route_vec: bool = False
    topk_lanes: bool = False  # Batcher odd-even top-k on lanes for E >= 128
    moe_unroll: bool = False  # Unrolled expert tile loops + grouped expert refills

    # `exact`: moving the field keeps hidden states and caches bit-identical
    # (scheduling only); `approx`: changes or may change the arithmetic.
    NUMERICS: ClassVar[dict[str, NumericsClass]] = {
        "pipe": "approx",  # single-tile expert pipelining
        "topk": "approx",  # packed-key top-k
        "ecopy_opt": "approx",  # redundant-expert dispatch policy
        "moe_group_rows": "exact",  # paired expert-ring releases (scheduling)
        "pipe_ep": "approx",  # EP expert software pipeline
        "r2_direct_store": "approx",  # `=` instead of zero-fill + `+=` (signed zero)
        "route_vec": "approx",  # vectorized routing policy
        "topk_lanes": "approx",  # Batcher odd-even lane top-k
        "moe_unroll": "approx",  # unrolled expert tiles (not exact on TPU, bisect)
    }


@dataclass(frozen=True)
class MoEDebug:
    """`MoEBlock` debug flags (`DebugOptions.blocks["moe"]`), not tuning knobs.

    Each flag adds a kernel input (`MoEBlock.extra_inputs`): `forced_routing`
    routes from an external `rbias` HBM input `[L, M, E]` per rank instead of
    the router's probabilities (the expert weights still come from the
    router); `row_mask` builds the routed-expert list only from the rows whose
    `moe_rows` SMEM entry (`[B]` int32) is non-zero (every row still adds its
    weighted output of the experts in that list).
    """

    forced_routing: bool = False
    row_mask: bool = False
