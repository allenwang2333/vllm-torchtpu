# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2026 Ant Group. All Rights Reserved.
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
# ==============================================================================
"""KDA helpers shared by the Pallas TPU forward and backward paths."""

from __future__ import annotations

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.kda.utils import (block_align_minor, cdiv,
                                             vmem_limit_bytes)
RCP_LN2 = 1.0 / math.log(2)

# =============================================================================
# Mini-batch sizing and shared forward recurrence
# =============================================================================


def estimate_mini_batch(
    per_tile_bytes: int,
    total: int,
    *,
    max_mb: int = 16,
    vmem_budget: int | None = None,
    align_minor: int | None = None,
) -> int:
  """Estimate the optimal mini-batch size to maximise VMEM utilisation.

  This mirrors the backward auto-tune pattern used throughout the KDA kernels:
  compute the largest MB that fits within the hardware VMEM budget, cap it,
  and then adjust downward so that ``total`` is evenly divisible by ``MB``.

  Args:
      per_tile_bytes: Estimated VMEM footprint (bytes) for **one** tile/head.
      total: Number of tiles (or heads) to partition.
      max_mb: Upper bound on MB (default 16).
      vmem_budget: VMEM budget in bytes. ``None`` (default) queries
          ``vmem_limit_bytes()`` at call time.
      align_minor: TPU block_align_minor constraint. ``None`` (default)
          queries ``block_align_minor()``.  When provided,
          the function prefers MB values that satisfy this alignment.

  Returns:
      Mini-batch size ``MB`` such that ``total % MB == 0`` (best-effort).
  """
  if vmem_budget is None:
    vmem_budget = vmem_limit_bytes()
  if align_minor is None:
    align_minor = block_align_minor()

  per_tile_bytes = max(1, per_tile_bytes)
  MB = max(1, vmem_budget // per_tile_bytes)
  MB = max(1, min(MB, total, max_mb))

  # Try to find an MB that divides total evenly.
  while total % MB != 0 and MB > 1:
    MB -= 1

  return MB


# =============================================================================
# Chunk-local cumsum helpers
# =============================================================================

# =============================================================================
# Fixed-length: Hillis-Steele scan (optimal for head_first=False)
# =============================================================================



# =============================================================================
# Public API
# =============================================================================


# =============================================================================
# Delta-rule hidden-state kernels
# =============================================================================

# ---------------------------------------------------------------------------
# Pallas TPU kernels for delta-rule inter-chunk state propagation
# ---------------------------------------------------------------------------


# ── Varlen Pallas kernel ────────────────────────────────────────────────────






# ── Non-varlen Pallas kernel ────────────────────────────────────────────────






# ── Public dispatch function ────────────────────────────────────────────────




