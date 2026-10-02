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
# ==============================================================================
"""Grouped matmul with one side permuted per row inside the kernel.

The gather that feeds an MoE's first stage and the scatter that follows its
second are the same ring of VMEM tiles crossed by per-row DMAs, so neither
permute ever materializes. A narrower rhs quantizes the lhs to it, one scale per
512 contracting elements as `gmm_v2` does, so each op equals `gmm_v2` on the
materialized permute over the valid rows, bit for bit up to the sign of zero.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import config
from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import kernel
from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import memory_ref


@jax.jit(static_argnames=["fusion", "tile_m", "fuse_act"])
def _fused_gmm(
    lhs: jax.Array,  # [rows, cols // 256, 2, 128] bf16, blocked by the caller
    rhs: jax.Array,  # per-expert weights, gate then up on the last axis
    rhs_scale: jax.Array | None,  # one per output column; None if rhs is bf16
    group_sizes: jax.Array,
    row_indices: jax.Array,  # int32[size_out_m], the HBM row each routed row uses
    group_offset: jax.Array | None = None,  # int32[1], LOCAL ADDITION
    *,
    fusion: config.Fusion,
    tile_m: int,
    fuse_act: str | None,
) -> jax.Array:
  """The grouped matmul both ops are; `fusion` picks which side is per-row."""
  cfg = config.PermuteGmmConfig(
      size_m=lhs.shape[0],
      size_out_m=row_indices.shape[0],
      size_k=lhs.shape[1] * lhs.shape[2] * lhs.shape[3],
      size_n=rhs.shape[2],
      num_groups=rhs.shape[0],
      tile_m=tile_m,
      dtype=lhs.dtype,
      rhs_dtype=rhs.dtype,
      tpu_info=pltpu.get_tpu_info(),
      fusion=fusion,
      fuse_act=fuse_act,
  )
  assert lhs.shape == cfg.lhs_shape, (lhs.shape, cfg.lhs_shape)
  # LOCAL ADDITION: upstream asserts equality here. `group_sizes` may now be
  # GLOBAL while `rhs` carries only this shard's experts, so it only has to be
  # long enough to cover `group_offset .. group_offset + num_groups`. That bound
  # cannot be checked at trace time because `group_offset` is a runtime value.
  assert group_sizes.shape[0] >= cfg.num_groups, (
      group_sizes.shape,
      cfg.num_groups,
  )
  # `gmm_v2`'s sub-channel form carries one scale per k block; this applies one.
  assert cfg.quantize_lhs == (rhs_scale is not None), rhs.dtype
  assert rhs_scale is None or rhs_scale.shape == (
      cfg.num_groups,
      1,
      1,
      cfg.size_n,
  ), rhs_scale.shape

  if group_offset is None:
    group_offset = jnp.zeros((1,), jnp.int32)
  scalar = memory_ref.ScalarRef(
      group_cum_sum=jnp.cumulative_sum(group_sizes, include_initial=True),
      row_idx=row_indices,
      group_offset=group_offset.astype(jnp.int32).reshape((1,)),
  )
  weights = memory_ref.WeightsRef(weight=rhs, scale=rhs_scale)
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  return pl.pallas_call(
      functools.partial(kernel.kernel_main, cfg=cfg),
      out_shape=jax.ShapeDtypeStruct(cfg.out_shape, cfg.dtype),
      in_specs=[
          jax.tree.map(lambda _: smem, scalar),
          hbm,
          jax.tree.map(lambda _: hbm, weights),
      ],
      out_specs=hbm,
      scratch_shapes=[memory_ref.ScratchRef.create(cfg)],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.tpu_info.vmem_capacity_bytes,
          disable_bounds_checks=True,
          flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
      ),
      name=(
          f"fused_{fusion.name.lower()}_gmm-{cfg.size_out_m}-{cfg.size_k}"
          f"-{cfg.size_n}"
      ),
  )(scalar, lhs, weights)


def fused_permute_gmm(
    lhs,
    rhs,
    rhs_scale,
    group_sizes,
    gather_indices,
    *,
    tile_m: int = 256,
    fuse_act: str | None = "silu",
    group_offset=None,
) -> jax.Array:
  """Grouped matmul over `lhs[gather_indices]`, never materialized.

  The pool is blocked by the CALLER and the output comes back blocked the same
  way: doing it here would relayout the whole pool on every call, since a bf16
  `[m, k]` tiles (16, 128) where the blocked form tiles (2, 128). Every gather
  index must be in `[0, size_m)` -- rows are DMA'd with bounds checks off, so
  one
  past the end reads outside the pool rather than wrapping. Rows past
  `sum(group_sizes)` are not written.
  """
  return _fused_gmm(
      lhs,
      rhs,
      rhs_scale,
      group_sizes,
      gather_indices,
      group_offset,
      fusion=config.Fusion.PERMUTE,
      tile_m=tile_m,
      fuse_act=fuse_act,
  )


def fused_unpermute_gmm(
    lhs,
    rhs,
    rhs_scale,
    group_sizes,
    scatter_indices,
    *,
    tile_m: int = 256,
    fuse_act: str | None = None,
    group_offset=None,
) -> jax.Array:
  """Grouped matmul whose result rows go straight to their token slots.

  The mirror of the above: the lhs arrives in routed order and the combine's
  un-permute never materializes. Only the rows named by the first
  `sum(group_sizes)` scatter indices are written, and each must be distinct and
  in `[0, size_out_m)` -- with bounds checks off a repeat is a lost row and an
  index past the end is a write outside the output.

  `group_offset` is a LOCAL ADDITION (not in cl/983441168): under expert
  parallelism `group_sizes` is global, `rhs` holds only experts
  `group_offset .. group_offset + rhs.shape[0]`, and `lhs` stays globally
  row-ordered so this shard's rows start at `cumsum(group_sizes)[group_offset]`.
  Leave it None for the upstream behaviour.
  """
  return _fused_gmm(
      lhs,
      rhs,
      rhs_scale,
      group_sizes,
      scatter_indices,
      group_offset,
      fusion=config.Fusion.UNPERMUTE,
      tile_m=tile_m,
      fuse_act=fuse_act,
  )
