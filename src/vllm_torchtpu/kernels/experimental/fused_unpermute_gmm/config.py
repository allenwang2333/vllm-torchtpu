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
"""Static resolution of one call: every trace-time quantity is derived here."""

import dataclasses
import enum

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


class Fusion(enum.Enum):
  """Which side of the matmul the kernel fuses, per row, inside the kernel.

  Never both: two unrolled per-row descriptor chains in one scheduling region
  cannot overlap, and the pair costs an order of magnitude.
  """

  PERMUTE = enum.auto()  # the lhs is gathered from a pool
  UNPERMUTE = enum.auto()  # the result is scattered to its token slot


@dataclasses.dataclass(frozen=True)
class PermuteGmmConfig:
  """One call's shapes, plus the knobs that are not derived from them."""

  size_m: int  # rows of the lhs in HBM: the pool when gathering
  size_out_m: int  # routed rows, i.e. len(gather_indices)
  size_k: int
  size_n: int  # gate and up concatenated
  num_groups: int
  tile_m: int
  dtype: jnp.dtype
  rhs_dtype: jnp.dtype
  tpu_info: pltpu.TpuInfo
  fusion: Fusion = Fusion.PERMUTE
  fuse_act: str | None = "silu"
  num_buffers: int = 3
  # Tiles gathered ahead of the dot. Below `num_buffers` the refill lands in a
  # slot the dot is not reading, so nothing has to be ordered against it.
  depth: int = 2
  num_buckets: int = 8

  @property
  def out_n(self) -> int:
    """Columns of the output: one of the two halves when they are fused."""
    return self.size_n // 2 if self.fuse_act else self.size_n

  @property
  def quantize_lhs(self) -> bool:
    """Whether the dot runs at the rhs rate rather than the lhs one."""
    return jnp.dtype(self.rhs_dtype).itemsize < jnp.dtype(self.dtype).itemsize

  def blocked(self, rows: int, cols: int) -> tuple[int, int, int, int]:
    """`[rows, cols]` with the sub-word pair as its own axis.

    A row is then whole 32-bit words, which makes a per-row DMA legal, and the
    row axis is outside the tile, which lets a block start at any offset.
    """
    packing = 4 // jnp.dtype(self.dtype).itemsize
    lanes = self.tpu_info.num_lanes
    return (rows, cols // lanes // packing, packing, lanes)

  @property
  def ring_cols(self) -> int:
    """Columns the ring carries: the lhs when gathering, the result when

    scattering.
    """
    return self.size_k if self.fusion is Fusion.PERMUTE else self.out_n

  @property
  def lhs_shape(self) -> tuple[int, int, int, int]:
    return self.blocked(self.size_m, self.size_k)

  def ring_shape(self, cols: int) -> tuple[int, int, int, int]:
    """A tile of `cols` columns at an ODD block count, the ring's row stride.

    A tile is read one block at a time down its rows, so `32 // gcd(stride, 32)`
    of the 32 VMEM banks answer that read, and a power-of-two column count makes
    the stride a power of two too. An odd stride answers with all 32.
    """
    rows, blocks, packing, lanes = self.blocked(self.tile_m, cols)
    return (rows, blocks | 1, packing, lanes)

  @property
  def out_shape(self) -> tuple[int, int, int, int]:
    return self.blocked(self.size_out_m, self.out_n)

  @property
  def max_num_gm(self) -> int:
    # Every group contributes a tile, and every `tile_m` routed rows one more;
    # the last term is slack for the warm-up ids, which `RowRing.start` clamps.
    return (
        self.num_groups
        + pl.cdiv(self.size_out_m, self.tile_m)
        + self.num_buffers
    )

  @property
  def bucket_m(self) -> int:
    """Quantum of the dot's height: a tile dots `ceil(live / this) * this`."""
    step = max(
        self.tpu_info.get_sublane_tiling(self.dtype),
        self.tile_m // self.num_buckets,
    )
    assert self.tile_m % step == 0, (self.tile_m, step)
    return step

  @property
  def min_bucket(self) -> int:
    """No dot shorter than this: the MXU is latch-bound below 128 rows."""
    return min(128, self.tile_m)

  @property
  def bucket_rows(self) -> tuple[int, ...]:
    """The dot heights a tile picks from, shortest first.

    Counted DOWN from the tile: a ladder short of `tile_m` clamps the switch and
    stores too few rows.
    """
    return tuple(
        reversed(range(self.tile_m, self.min_bucket - 1, -self.bucket_m))
    )

  @property
  def quant_block_k(self) -> int:
    """Contracting elements per lhs quantization scale; `gmm_v2`'s is 512."""
    return min(self.size_k, 512)

  @property
  def quant_max(self) -> float:
    return float(jnp.finfo(self.rhs_dtype).max)
