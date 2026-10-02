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
"""Everything that touches memory: the ref bundles, the layout, the gather."""

import dataclasses
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.fused_unpermute_gmm import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScalarRef:
  """The routing, in SMEM: group boundaries, and the HBM row each routed row

  is gathered from or scattered to.

  `group_offset` is a LOCAL ADDITION, not part of cl/983441168. It is the index
  of the first expert this shard owns inside a GLOBAL `group_cum_sum`: under
  expert parallelism the activation stays globally row-ordered and each shard
  computes only its own slice of it, so a shard's routed rows begin part-way
  down the array rather than at zero. With it zero the kernel behaves exactly
  as upstream.
  """

  group_cum_sum: Any
  row_idx: Any
  group_offset: Any = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """Per gm tile: the group it belongs to, and its routed-row offset."""

  gm_id_to_g_id: Any
  gm_id_to_hbm_offset: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef:
  weight: Any
  scale: Any = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef:
  gate: WeightsRef
  up: WeightsRef | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipeRefs:
  """What the pipeline emitter carries in: the weights, and the lhs if it is

  not being gathered per row.
  """

  rhs: FusedWeightsRef
  lhs: Any = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScratchRef:
  """Everything the kernel allocates: the gathered-tile ring and the metadata."""

  ring_vmem: Any
  ring_sem: Any
  num_gm: Any
  metadata: MetadataRef

  @classmethod
  def create(cls, cfg: config.PermuteGmmConfig) -> "ScratchRef":
    return cls(
        ring_vmem=pltpu.VMEM(
            (cfg.num_buffers,) + cfg.ring_shape(cfg.ring_cols), cfg.dtype
        ),
        ring_sem=pltpu.SemaphoreType.DMA((cfg.num_buffers,)),
        num_gm=pltpu.SMEM((1,), jnp.int32),
        metadata=MetadataRef(
            gm_id_to_g_id=pltpu.SMEM((cfg.max_num_gm,), jnp.int32),
            gm_id_to_hbm_offset=pltpu.SMEM((cfg.max_num_gm,), jnp.int32),
        ),
    )


@dataclasses.dataclass(frozen=True)
class BlockedRef:
  """A `[rows, blocks, packing, lanes]` ref, read and written as `[rows, cols]`.

  One word holds two adjacent lane-blocks of a row, so a column is one strided
  load per word with no cross-sublane move. Load and store are inverses, so the
  gathered tile and the output share this one codec.
  """

  ref: Any
  cfg: config.PermuteGmmConfig
  blocks: int = 0  # payload blocks; 0 = all of them, as the output has

  def head(self, rows: int) -> "BlockedRef":
    return BlockedRef(self.ref.at[pl.ds(0, rows)], self.cfg, self.blocks)

  @property
  def _words(self):
    return self.ref.bitcast(jnp.uint32).reshape(-1, self.cfg.tpu_info.num_lanes)

  def load(self) -> jax.Array:
    words, stride = self._words, self.ref.shape[1]
    return jnp.concat(
        [
            pltpu.unpack_elementwise(
                words[i::stride],
                index=j,
                packed_dtype=self.cfg.dtype,
                unpacked_dtype=jnp.float32,
            ).astype(self.cfg.dtype)
            for i in range(self.blocks or stride)
            for j in (0, 1)
        ],
        axis=-1,
    )

  def store(self, val: jax.Array) -> None:
    words, stride = self._words, self.ref.shape[1]
    blocks = self.blocks or stride
    chunks = jnp.split(val.astype(jnp.float32), 2 * blocks, axis=-1)
    for i in range(blocks):
      packed = pltpu.pack_elementwise(
          chunks[2 * i : 2 * i + 2], packed_dtype=self.cfg.dtype
      )
      words[i::stride] = pltpu.bitcast(packed, jnp.uint32)


@dataclasses.dataclass(frozen=True)
class TileMap:
  """Which routed rows a gm tile owns. One window, read by everyone.

  The gather, the dot's height and the out block all mean the same rows, so
  they are one function of `gm_id` rather than three that must agree.
  """

  scalar: ScalarRef
  metadata: MetadataRef
  num_gm_ref: Any
  cfg: config.PermuteGmmConfig

  def _global_g_id(self, local_g_id):
    """Local expert index -> index into the GLOBAL `group_cum_sum`.

    LOCAL ADDITION (see `ScalarRef.group_offset`). `gm_id_to_g_id` stores the
    LOCAL expert so it can index `rhs` directly; the cumsum is global, so every
    read of it has to be shifted back up by the shard's first expert.
    """
    if self.scalar.group_offset is None:
      return local_g_id
    return local_g_id + self.scalar.group_offset[0]

  def offset(self, gm_id) -> jax.Array:
    return self.metadata.gm_id_to_hbm_offset[gm_id]

  def live_rows(self, gm_id) -> jax.Array:
    """Rows of tile `gm_id` that its group actually has: 1 .. tile_m."""
    start = self.offset(gm_id)
    g_id = self._global_g_id(self.metadata.gm_id_to_g_id[gm_id])
    end = self.scalar.group_cum_sum[g_id + 1]
    return jnp.minimum(end, start + self.cfg.tile_m) - start

  def window(self, gm_id, signed: bool) -> tuple[jax.Array, jax.Array]:
    """`(first routed row, rows)`, and no rows at all outside the grid."""
    oob = gm_id >= self.num_gm_ref[0]
    if signed:
      oob = jnp.logical_or(gm_id < 0, oob)
    safe = jnp.where(oob, 0, gm_id)
    return self.offset(safe), jnp.where(oob, 0, self.live_rows(safe))

  def row_block(self, gm_id):
    """The block spec index map of the side pipelined by rows."""
    return (pl.ds(self.offset(gm_id), self.live_rows(gm_id)), 0, 0, 0)

  def in_row_block(self, gm_id):
    """The same window, CLAMPED: a ring deeper than two evaluates it past the

    end of the grid, where an unclamped slice neither compiles nor names a row
    inside the array.
    """
    gm_id = jnp.clip(gm_id, 0, jnp.maximum(self.num_gm_ref[0] - 1, 0))
    return (
        pl.ds(
            jnp.minimum(self.offset(gm_id), self.cfg.size_out_m - 1),
            jnp.clip(self.live_rows(gm_id), 0, self.cfg.tile_m),
        ),
        0,
        0,
        0,
    )


@dataclasses.dataclass(frozen=True)
class RowRing:
  """A ring of VMEM tiles, and the per-row DMAs that move them to or from HBM.

  Kept `depth` tiles ahead of the dot: `start` issues a tile's rows against
  `row_idx`, `wait` blocks on exactly those. `to_hbm` is the only difference
  between a gather that fills a slot before the dot reads it and a scatter that
  empties one after the dot wrote it.
  """

  hbm_ref: Any
  vmem_ref: Any
  sem_ref: Any
  row_idx: Any
  tiles: TileMap
  cfg: config.PermuteGmmConfig
  cols: int
  to_hbm: bool = False

  @property
  def blocks(self) -> int:
    """Payload blocks of a tile; the ring is one wider when that is even."""
    return self.cfg.blocked(self.cfg.tile_m, self.cols)[1]

  def tile(self, slot) -> BlockedRef:
    return BlockedRef(self.vmem_ref.at[slot], self.cfg, self.blocks)

  def slot(self, gm_id) -> jax.Array:
    """The ring slot a tile owns -- never passed in, so a lag and a slot cannot

    disagree. A scatter names the tile BEFORE the current one, so its id can be
    -1 and the remainder needs a non-negative argument.
    """
    if self.to_hbm:
      gm_id += self.cfg.num_buffers
    return lax.rem(gm_id, self.cfg.num_buffers)

  def start(self, gm_id) -> None:
    """Starts one DMA per row of the tile, sized 0 past the group's last row --

    which is also what leaves the semaphore at zero when the grid ends.
    """
    offset, rows = self.tiles.window(gm_id, self.to_hbm)
    slot = self.slot(gm_id)
    vmem, sem, blocks = (
        self.vmem_ref.at[slot],
        self.sem_ref.at[slot],
        self.blocks,
    )

    for r in range(self.cfg.tile_m):
      size = jnp.where(r < rows, 1, 0)
      hbm = self.hbm_ref.at[pl.ds(self.row_idx[offset + r], size)]
      tile = vmem.at[pl.ds(r, size), :blocks]
      src, dst = (tile, hbm) if self.to_hbm else (hbm, tile)
      pltpu.make_async_copy(src, dst, sem).start()

  def wait(self, gm_id) -> None:
    """Waits for exactly what was issued: one window over those same rows.

    Only a scatter can name a tile outside the grid here -- a gather waits on
    the step it is in, which the grid bound already put inside.
    """
    rows = (
        self.tiles.window(gm_id, True)[1]
        if self.to_hbm
        else self.tiles.live_rows(gm_id)
    )
    win = self.vmem_ref.at[0, pl.ds(0, rows), : self.blocks]
    pltpu.make_async_copy(win, win, self.sem_ref.at[self.slot(gm_id)]).wait()


def block_specs(tiles: TileMap):
  """The pipeline emitter's in and out specs, both indexed by gm tile.

  The out block starts at the group's EXACT routed-row offset, so two tiles of
  a split group never overlap and the kernel needs no accumulator.
  """
  cfg, metadata = tiles.cfg, tiles.metadata

  def weight_map(gm_id):
    return (metadata.gm_id_to_g_id[gm_id], 0, 0)

  def scale_map(gm_id):
    return (metadata.gm_id_to_g_id[gm_id], 0, 0, 0)

  buffered = pl.Buffered(buffer_count=cfg.num_buffers)
  # LOCAL DEVIATION FROM cl/983441168 -- not Guowei's code.
  # His weights use the plain `buffered` above, i.e. `use_lookahead=False`.
  # Weights are indexed by EXPERT (`weight_map`), so a group spanning several
  # tiles maps many consecutive grid steps to one block. Without lookahead the
  # next expert's block is fetched exactly one grid step before use, which at
  # K=1024 N=4096 means one tile-dot (1.6-2.8us) of cover for an 8.39MB
  # transfer; the remainder is exposed once per expert. With it,
  # `fetch_with_lookahead` scans forward past the unchanged steps and keeps
  # `buffer_count` DISTINCT experts in flight. Matches
  # moe_gmm_common/config.py:197. Flip back to `buffered` to reproduce his
  # original numbers.
  weights_buffered = pl.Buffered(
      buffer_count=cfg.num_buffers, use_lookahead=True
  )
  half = WeightsRef(
      weight=pl.BlockSpec(
          (None, cfg.size_k, cfg.out_n),
          weight_map,
          pipeline_mode=weights_buffered,
      ),
      scale=None
      if not cfg.quantize_lhs
      else pl.BlockSpec(
          (None, 1, 1, cfg.out_n), scale_map, pipeline_mode=weights_buffered
      ),
  )
  rhs = FusedWeightsRef(gate=half, up=half if cfg.fuse_act else None)
  permute = cfg.fusion is config.Fusion.PERMUTE
  _, blocks, packing, lanes = cfg.blocked(
      cfg.tile_m, cfg.out_n if permute else cfg.size_k
  )
  row = (pl.BoundedSlice(cfg.tile_m), blocks, packing, lanes)
  # The emitter caps a pipelined output at two buffers, which is its default;
  # the pipelined lhs is as deep as the ring it stands in for.
  out = pl.BlockSpec(row, tiles.row_block) if permute else ()
  lhs = (
      None
      if permute
      else pl.BlockSpec(row, tiles.in_row_block, pipeline_mode=buffered)
  )
  return PipeRefs(rhs=rhs, lhs=lhs), out
