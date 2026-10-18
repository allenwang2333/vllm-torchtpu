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
"""`qwen35-candidate30`: the M5 Qwen3.5 tuning, moved out of library defaults.

Select with `VLLM_TPU_MEGAKERNEL="profile=qwen35-candidate30"` or
`additional_config={"megakernel": {"profile": "qwen35-candidate30"}}`.

This reproduces the tiles and perf flags that `spec.default_tiles` /
`Tiles` defaults carried at M5 head `bb5316a0a` (including the code paths that
were unconditional there and are now flags: `pipe_ep`, `gdn_head_major`,
`gqa_qv`, `moe_unroll`, `rel_fetch`), plus `pool_gather="auto"`.
Fused pool I/O stays off here (experimental); layer it on explicitly with
`fused_pool_io=auto` for ablations.
"""

from __future__ import annotations

import dataclasses

from ..profile import Baseline, Profile, register_profile
from ..spec import Tiles

# Opt-in perf flags that were default-on at M5 (`bb5316a0a`).
_C30_FLAGS = dict(
    kv_sub=1024,
    ecopy_opt=1,
    pipe_ep=True,
    r2_direct_store=True,
    route_vec=True,
    dense_acc_pair=True,
    kv_ering=True,
    pair_kv=True,
    dmm_unroll=8,
    gq_pack=True,
    topk_lanes=True,
    conv_stage=True,
    gdn_head_major=True,
    gqa_qv=True,
    moe_unroll=True,
    rel_fetch=True,
)
_BASE = Tiles(hier=1, pack=1, mb8=2, **_C30_FLAGS)
_VERIFY = dataclasses.replace(_BASE, nkv=4, rsag=True, hier=2)
_WIDE = dataclasses.replace(_VERIFY, nbd=10, nkv=3, vmem_mib=62, hier=0)


def candidate30_tiles(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
    """Per-bucket tiles of the M5 tuning (deeper EP expert rings, hier=2, HIER3)."""
    nbe_ep = 12 if not moe_tp else 10
    nbe_small = 20 if not moe_tp else nbe_ep
    if B > 48:
        t = dataclasses.replace(_WIDE, nbd=8 if B <= 56 else 7, nbe=nbe_ep, hier=2)
    elif B > 32:
        t = dataclasses.replace(_WIDE, nbd=8, nbe=nbe_ep, hier=2)
    elif B > 24:
        t = dataclasses.replace(_WIDE, nbe=nbe_ep, hier=2)
    elif B > 16:
        t = dataclasses.replace(_WIDE, nbd=12, nbe=nbe_ep, hier=2)
    elif B > 8:
        t = dataclasses.replace(_VERIFY, nbe=nbe_ep)
    elif B > 4:
        t = dataclasses.replace(_BASE, nbe=nbe_small, rsag=True, hier=2)
    else:
        t = dataclasses.replace(_BASE, nbe=nbe_small)
    if R != 8 and t.hier in (1, 2):
        is_pow2_r = R >= 2 and (R & (R - 1)) == 0
        t = dataclasses.replace(t, hier=3 if (B <= 32 and is_pow2_r) else 0)
    return t


PROFILE = register_profile(
    Profile(
        name="qwen35-candidate30",
        model="qwen3_5",
        numerics="approx",
        tiles=candidate30_tiles,
        fused_pool_io=False,
        pool_gather=None,
        baseline=Baseline(ref="candidate30"),
        description=(
            "M5 Qwen3.5 tuning (bb5316a0a defaults): kv_sub=1024, route_vec, "
            "dense_acc_pair, pair_kv, KVE, QV/GQ, Batcher top-k, unrolled experts "
            "+ PIPE_EP, head-major GDN, deeper EP expert rings, hier=2 for B>16, "
            "GPR auto. Numerics: approx."
        ),
    )
)

__all__ = ["PROFILE", "candidate30_tiles"]
