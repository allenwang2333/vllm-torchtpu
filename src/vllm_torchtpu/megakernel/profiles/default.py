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
"""Model-agnostic default tile rule: one schedule for every bucket.

`default_tiles` returns the same tiles for every bucket: library field defaults,
the flat-ring all-reduce (`hier=0`) and the chip's VMEM / SMEM budgets. It does
not branch on the batch size, the rank count or the model; ring depths may be
refined at build time to fit the VMEM budget when `ExecSpec.tiles` is left
`None`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..exec_spec import Tiles


@dataclass(frozen=True)
class ChipBudget:
    """On-chip memory budgets the default tiles are sized for."""

    name: str
    vmem_mib: int  # VMEM limit handed to Mosaic (MiB)
    smem_mib: int  # SMEM limit (MiB)


# TPU v7x: 64 MiB VMEM per core, 2 MiB kept free for compiler-internal scratch.
TPU_V7X = ChipBudget(name="tpu_v7x", vmem_mib=62, smem_mib=1)
DEFAULT_CHIP = TPU_V7X


def default_tiles(
    B: int, *, moe_tp: bool = False, R: int = 8, chip: ChipBudget = DEFAULT_CHIP
) -> Tiles:
    """Model-agnostic default tiles (identical for every bucket).

    `B`, `moe_tp` and `R` are accepted for tile-rule call-site compatibility
    and ignored: field defaults everywhere, flat-ring all-reduce (`hier=0`),
    and `chip`'s VMEM / SMEM budgets.
    """
    del B, moe_tp, R
    return Tiles(hier=0, vmem_mib=chip.vmem_mib, smem_mib=chip.smem_mib)


__all__ = ["DEFAULT_CHIP", "TPU_V7X", "ChipBudget", "default_tiles"]
