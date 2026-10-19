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
"""Tile rule of the `default` profile: the general, frozen-parity schedule.

All per-bucket (`B`) tuning of the default path lives here; blocks and
`spec.py` only read the resulting `Tiles`. `spec.default_tiles` delegates to
`frozen_parity_tiles`, and the `default` profile (`profile.DEFAULT_PROFILE`)
uses it as its tile rule.
"""

from __future__ import annotations

import dataclasses

from ..spec import DEFAULT_TILES, VERIFY_TILES, WIDE_TILES, Tiles


def frozen_parity_tiles(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
    """General default tiles for batch row count `B`.

    Selects the per-bucket ring depths, all-reduce mode, and VMEM budget (using
    `nbe=6` for `B > 48` so 64-row MoE scratch fits VMEM). Every opt-in
    performance flag stays at its `Tiles` default (off). Tuned,
    numerics-changing schedules live in named profiles (`profile.py`).

    `moe_tp` is accepted for signature compatibility with profile tile rules; the
    default schedule does not depend on it. Hierarchical all-reduce (`hier` 1/2)
    is v7x-8 specific, so `R` outside `{1, 8}` falls back to the flat ring.
    """
    del moe_tp
    if B > 48:
        t = dataclasses.replace(WIDE_TILES, nbe=6)
    elif B > 32:
        t = WIDE_TILES
    elif B > 24:
        t = dataclasses.replace(WIDE_TILES, hier=2)
    elif B > 16:
        t = dataclasses.replace(WIDE_TILES, nbd=12, hier=2)
    elif B > 8:
        t = VERIFY_TILES
    elif B > 4:
        t = dataclasses.replace(DEFAULT_TILES, rsag=True, hier=2, ecopy_opt=1)
    else:
        t = DEFAULT_TILES
    if R not in (1, 8) and t.hier != 0:
        t = dataclasses.replace(t, hier=0)
    return t


__all__ = ["frozen_parity_tiles"]
