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
"""Shared data contracts for the Pallas TPU KDA implementation."""

import dataclasses
from typing import TypeAlias

import jax


CpMetadata: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
] | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class KdaResiduals:
  """Prepared forward values consumed by the custom backward."""

  q: jax.Array  # [H, B, T_ALIGNED, K]
  k: jax.Array  # [H, B, T_ALIGNED, K]
  v: jax.Array  # [H, B, T_ALIGNED, V]
  beta: jax.Array  # [H, B, T_ALIGNED]
  g_cumsum: jax.Array | None  # [H, B, T_ALIGNED, K]
  aqk: jax.Array  # [H, B, T_ALIGNED, BT]
  akk: jax.Array  # [H, B, T_ALIGNED, BT]
  initial_state: (
      jax.Array | None  # [B, H, K, V] or [B, N, H, K, V]
  )
  g_org: jax.Array | None  # [H, B, T_ALIGNED, K]
  a_log: jax.Array | None  # [H]
  dt_bias: jax.Array | None  # [H*K]
  h: jax.Array | None  # [H, B, NT, K, V]
  g_dtype_marker: jax.Array  # scalar
  q_rstd: jax.Array | None  # [H, B, T_ALIGNED]
  k_rstd: jax.Array | None  # [H, B, T_ALIGNED]
  cu_seqlens: jax.Array | None  # [B, N_CU] or [N_CU]
  aligned_cu_seqlens: jax.Array | None  # [B, N_CU] or [N_CU]
  chunk_indices: jax.Array | None  # [B, NT, 2] or [NT, 2]
  aligned_segment_ids: jax.Array | None  # [B, T_ALIGNED]
  segment_ids: jax.Array | None  # [B, T_ORIG]
  cp_metadata: CpMetadata
