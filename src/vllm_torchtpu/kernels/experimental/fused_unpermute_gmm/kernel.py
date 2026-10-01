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
"""The kernel: the grid's metadata, then rows in, dot, rows out, per tile."""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import compute
from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import config
from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import memory_ref


def fill_metadata(
    scalar: memory_ref.ScalarRef,
    metadata: memory_ref.MetadataRef,
    cfg: config.PermuteGmmConfig,
) -> jax.Array:
  """Writes one (group, routed-row offset) pair per gm tile; returns the count.

  A group of `g` rows contributes `cdiv(g, tile_m)` tiles at exact offsets: no
  sublane term, because the out block needs no aligned start.

  `group_offset` (LOCAL ADDITION, see `ScalarRef`) shifts which slice of a
  GLOBAL `group_cum_sum` this shard walks. The loop still runs over the
  `num_groups` LOCAL experts and stores the LOCAL id, so `weight_map` indexes
  `rhs` unchanged; only the boundary lookups move. The offsets written are
  absolute routed-row positions, which is what the row DMAs and the out block
  already expect, because the activation stays globally row-ordered.
  """
  # Warm-up reads tile 0 before the grid is known to have one, so it must name
  # a real group even when no group has rows at all.
  metadata.gm_id_to_g_id[0] = 0
  metadata.gm_id_to_hbm_offset[0] = 0
  base = 0 if scalar.group_offset is None else scalar.group_offset[0]

  def write_tile(gm_id, offset, *, group_id):
    metadata.gm_id_to_g_id[gm_id] = group_id
    metadata.gm_id_to_hbm_offset[gm_id] = offset
    return offset + cfg.tile_m

  def write_group(group_id, num_gm):
    start = scalar.group_cum_sum[base + group_id]
    end = scalar.group_cum_sum[base + group_id + 1]
    next_num_gm = num_gm + pl.cdiv(end - start, cfg.tile_m)
    lax.fori_loop(
        num_gm,
        next_num_gm,
        functools.partial(write_tile, group_id=group_id),
        start,
    )
    return next_num_gm

  return lax.fori_loop(0, cfg.num_groups, write_group, 0)


def pipeline_body(
    pipe: memory_ref.PipeRefs,
    out_ref=None,
    *,
    ring: memory_ref.RowRing,
    cfg: config.PermuteGmmConfig,
) -> None:
  """One gm tile: take its rows in, dot them, put them out.

  Which side is the ring and which the pipelined block is the whole difference
  between the two modes; the gather fills a slot `depth` tiles before the dot
  reads it, the scatter empties one the tile after the dot wrote it.
  """
  gm_id = pl.program_id(0)
  permute = cfg.fusion is config.Fusion.PERMUTE
  slot = ring.slot(gm_id)
  ring.wait(gm_id if permute else gm_id - cfg.num_buffers)
  lhs = ring.tile(slot) if permute else memory_ref.BlockedRef(pipe.lhs, cfg)
  out = memory_ref.BlockedRef(out_ref, cfg) if permute else ring.tile(slot)

  def dot(rows: int) -> None:
    acc = compute.dot(lhs.head(rows).load(), pipe.rhs, cfg)
    # Inside the branch the issue shares VLIW bundles with this dot; outside
    # the switch it is a scheduling region of its own and none of it can.
    ring.start(gm_id + cfg.depth if permute else gm_id - 1)
    out.head(rows).store(acc)

  # A tile is `tile_m` tall but its group may hold far fewer rows, so dot the
  # smallest bucket that covers them and move only that many.
  live = ring.tiles.live_rows(gm_id)
  idx = jnp.maximum(
      0, (live - cfg.bucket_rows[0] + cfg.bucket_m - 1) // cfg.bucket_m
  )
  lax.switch(idx, [functools.partial(dot, rows) for rows in cfg.bucket_rows])


def kernel_main(
    scalar: memory_ref.ScalarRef,
    lhs_hbm_ref,
    rhs_hbm_ref: memory_ref.WeightsRef,
    out_hbm_ref,
    scratch: memory_ref.ScratchRef,
    *,
    cfg: config.PermuteGmmConfig,
) -> None:
  """The `pallas_call` body: the routing, the pool, the weights, the output."""
  num_gm = fill_metadata(scalar, scratch.metadata, cfg)
  scratch.num_gm[0] = num_gm
  tiles = memory_ref.TileMap(scalar, scratch.metadata, scratch.num_gm, cfg)
  permute = cfg.fusion is config.Fusion.PERMUTE
  # LOCAL ADDITION (not in cl/983441168): drop compiler dependence edges between
  # slot stores and the per-row DMAs on other slots. `RowRing.slot()` derives the
  # slot from a traced `lax.rem`, which Mosaic cannot prove yields distinct
  # buffers, so without this it must order them conservatively. Slot rotation is
  # already synchronized by the DMA semaphores. Mirrors what our kernel does at
  # fused_unpermute_gmm/kernel.py:53. See b/519211134#comment14.
  if hasattr(pltpu, "annotate"):
    ring_vmem = pltpu.annotate(scratch.ring_vmem, no_hazard_no_deps=True)
  else:
    ring_vmem = scratch.ring_vmem
  ring = memory_ref.RowRing(
      hbm_ref=lhs_hbm_ref if permute else out_hbm_ref,
      vmem_ref=ring_vmem,
      sem_ref=scratch.ring_sem,
      row_idx=scalar.row_idx,
      tiles=tiles,
      cfg=cfg,
      cols=cfg.ring_cols,
      to_hbm=not permute,
  )
  for b in range(cfg.depth if permute else 0):
    ring.start(b)

  up = (
      None
      if cfg.fuse_act is None
      else jax.tree.map(lambda x: x.at[..., cfg.out_n :], rhs_hbm_ref)
  )
  operands = memory_ref.PipeRefs(
      rhs=memory_ref.FusedWeightsRef(gate=rhs_hbm_ref, up=up),
      lhs=None if permute else lhs_hbm_ref,
  )
  in_specs, out_specs = memory_ref.block_specs(tiles)
  pltpu.emit_pipeline(
      functools.partial(pipeline_body, ring=ring, cfg=cfg),
      grid=(num_gm,),
      in_specs=(in_specs,),
      out_specs=out_specs,
  )(operands, *((out_hbm_ref,) if permute else ()))

  if not permute:
    # Each step drains its predecessor, so the last tile leaves here; then every
    # slot still in flight is waited, because a nonzero DMA semaphore at Mosaic
    # exit halts the core.
    ring.start(num_gm - 1)
    for b in range(cfg.num_buffers):
      ring.wait(num_gm - cfg.num_buffers + b)
