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
"""Configuration for the fused permute + grouped matmul.

`Tuning` is the knob set with its measured defaults. `Config` is one call's
resolution of it against gmm_v2's own configs: the gather ring, the index
table and the tile the gather wants.
"""
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2

# The invariant index table's rows, as the wrapper casts them.
INDEX_DTYPE = jnp.int32
# TPU7x SMEM. Stands in for the device when op construction runs before one exists.
V7X_SMEM_CAPACITY_BYTES = 1 << 20
# Where a scalar-prefetch operand starts to cliff on v7x. Measured, not derived.
V7X_SMEM_INDEX_LIMIT_BYTES = 480_000
# gmm_v2's own zero-init staging budget, kept so both kernels tile the zero fill alike.
ZERO_TILE_TARGET_BYTES = 2 * 1024 * 1024
# `gather_tiling`'s law, fitted on v7x at the serving fill; the constants are per-generation.
TILE_SPREAD_SIGMAS = 2.5  # headroom over the mean so a fat group still fits one tile
TILE_ALIGN = 32  # the granularity the tile sweep resolved
TILE_M_MIN = 64  # a skewed routing still gets a tile a group can fit in
TILE_M_MAX = 512  # gmm_v2's ceiling at this dtype pair


@dataclasses.dataclass(frozen=True)
class Tuning:
    """The fused kernel's knobs.

    Every default here was measured on TPU7x for Qwen3.5-397B at the serving
    fill (0.6-12.5% of the padded pool live), so another generation or routing
    regime wants a re-sweep. The exception is `smem_capacity_bytes`, which
    `from_device` reads off the device.

    Attributes:
        num_slots: VMEM ring buffers for gathered tiles. Must exceed `lookahead`,
            or the ring aliases a slot before its wait.
        lookahead: tiles issued ahead of the dot. This is the depth knob; extra
            slots deepen nothing, since the issue site names tile `s + lookahead`.
        issue_unroll: rows per unrolled block of the row-issue loop. The loop's
            trip count is the tile's live row count, because `pl.when` predicates
            an instruction and a dead row would still take its issue slot.
        num_buckets: gmm_v2 buckets per tile (`bucket_base = tile_m // num_buckets`)
            so the dot covers the live rows rather than the tile. Saturates at 6 on
            the serving fill; a quotient that is not a sublane multiple dividing
            `tile_m` silently degrades to one bucket.
        tile_m: overrides the tile `gather_tiling` derives, which reads mean live
            rows per group and is blind to `max_group`, where the cliff is. None
            keeps the law; a value must be a multiple of the lhs sublane block.
        ring_pad_slots: lane-blocks of pad per ring row, which takes the LHS
            load's row stride off a power of two, where VMEM banking costs most.
            `ring_chunks` rounds any even value up to a legal row.
        smem_index_fraction: the share of SMEM the index table may take.
        smem_capacity_bytes: the device's SMEM; `from_device` reads it.
    """
    num_slots: int = 4
    lookahead: int = 3
    issue_unroll: int = 16
    num_buckets: int = 6
    tile_m: int | None = None
    ring_pad_slots: int = 8
    smem_index_fraction: float = (V7X_SMEM_INDEX_LIMIT_BYTES /
                                  V7X_SMEM_CAPACITY_BYTES)
    smem_capacity_bytes: int = V7X_SMEM_CAPACITY_BYTES

    def __post_init__(self):
        if self.num_slots <= self.lookahead:
            raise ValueError("num_slots must exceed lookahead: the ring would "
                             "alias a slot before its wait")
        if self.ring_pad_slots % 2:
            raise ValueError("ring_pad_slots must be even: two lane-blocks "
                             "share a u32 sublane")

    @classmethod
    def from_device(cls, **overrides) -> "Tuning":
        """The defaults with the SMEM capacity read off the device.

        Op construction can run before a device exists; the v7x value then
        stands. Never call this from a Dynamo-traced region: the accessor is a
        nanobind call and untraceable there.
        """
        try:
            capacity = pltpu.get_tpu_info().smem_capacity_bytes
        except Exception:  # noqa: BLE001 -- no device yet, or an unsupported kind
            return cls(**overrides)
        return cls(smem_capacity_bytes=capacity, **overrides)

    @property
    def smem_index_limit_bytes(self) -> int:
        """Bytes of SMEM this kernel will spend on the invariant index table."""
        return int(self.smem_capacity_bytes * self.smem_index_fraction)

    @property
    def max_index_rows(self) -> int:
        """Routed rows the index table can hold; above it the path declines."""
        return self.smem_index_limit_bytes // jnp.dtype(INDEX_DTYPE).itemsize


def gather_tiling(rows: int, size_group: int) -> int:
    """tile_m for the gather path: mean live rows per group, plus spread.

    A gm tile spans one group and the dot runs the height of its bucket, so the
    tile follows the live rows per group rather than the VMEM fit
    `calculate_tiling` takes. Not a gather cost: with no gather in either arm,
    this tile and its bucket beat gmm_v2's own by 0.5-4.1% across the serving
    fills. The two are a pair -- a bucket that tracks live rows costs
    `tile_m // bucket_base` switch branches, which only a small tile affords.
    Undersizing is the worse error: a tile below the largest group splits it and
    the split dots run at reduced MXU height, hence the spread term and floor.

    `rows` is the routed row count local to this shard; the caller folds in
    ep_size.
    """
    live = -(-rows // max(size_group, 1))
    # Ceiling of the spread term; the .999 tolerates float noise below 1e-3.
    live += int(TILE_SPREAD_SIGMAS * live**0.5 + 0.999)
    return int(
        min(TILE_M_MAX, max(TILE_M_MIN, gmm_v2.align_to(live, TILE_ALIGN))))


def ring_chunks(row_chunks: int, pad: int, num_sublanes: int) -> int:
    """Lane-blocks a padded ring row occupies.

    Measured on v7x: past four lane-blocks, which every ring is, the LHS
    build's flatten of the ring's u32 view takes only a chunk count that is a
    whole number of sublanes, so the pad grows to the next legal one rather
    than the shape being refused.
    A count landing back on a power-of-two stride is the VMEM banking case the
    pad exists to avoid. `estimate_vmem_bytes` prices the ring through here.
    """
    chunks = row_chunks + pad
    chunks += -chunks % num_sublanes
    return chunks + num_sublanes if chunks & (chunks - 1) == 0 else chunks


def estimate_vmem_bytes(dims: gmm_v2.Dimensions, lhs_cfgs: gmm_v2.InputConfigs,
                        rhs_cfgs: gmm_v2.InputConfigs, tile_m: int,
                        tile_n: int, fuse_factor: int, *, tuning: "Tuning",
                        num_lanes: int, num_sublanes: int) -> int:
    """This kernel's VMEM footprint at tile_k == size_k.

    `calculate_tiling`'s estimate does not apply: the pipelined LHS tile is
    replaced by the gather slots, and tile_m comes from the live-row law rather
    than from the fit.
    """
    tile_k = dims.size_k
    lhs_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8

    # The ring is padded per row; pricing it any other way lets the tile law pick
    # a tile whose ring does not fit.
    ring_k = ring_chunks(tile_k // num_lanes, tuning.ring_pad_slots,
                         num_sublanes) * num_lanes
    gather = tuning.num_slots * tile_m * ring_k * lhs_bytes
    lhs = tile_m * tile_k * lhs_bytes  # the rows as the dot wants them
    if lhs_cfgs.should_quantize:
        lhs += tile_m * tile_k * (
            jax.dtypes.itemsize_bits(lhs_cfgs.quant_dtype) // 8)

    rhs = 3 * tile_k * tile_n * jax.dtypes.itemsize_bits(rhs_cfgs.dtype) // 8
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
        rhs += 2 * pl.cdiv(tile_k, rhs_cfgs.quant_block_size) * tile_n * 4
    if rhs_cfgs.has_bias:
        rhs += 2 * tile_n * 4
    rhs *= fuse_factor

    acc_bytes = 2 if lhs_cfgs.quant_dtype is not None else 4
    acc = tile_m * fuse_factor * tile_n * acc_bytes
    out = 2 * tile_m * tile_n * lhs_bytes
    return gather + lhs + rhs + acc + out


def bucket_base(tile_m: int, sublane: int, num_buckets: int) -> int:
    """gmm_v2's bucket step: must divide tile_m and stay a sublane multiple.

    Otherwise the bucket switch indexes a bucket the block specs do not cover,
    so an unusable quotient falls back to one bucket of the whole tile.
    """
    if num_buckets <= 1:
        return tile_m
    cand = (tile_m // num_buckets // sublane) * sublane
    if cand > 0 and tile_m % cand == 0:
        return cand
    return tile_m


def gather_tile_sizes(dims: gmm_v2.Dimensions, lhs_cfgs: gmm_v2.InputConfigs,
                      rhs_cfgs: gmm_v2.InputConfigs, vmem_limit_bytes: int,
                      fuse_act: str | None, *, tuning: Tuning,
                      ep_size: int) -> gmm_v2.TileSizes:
    """Tile sizes for the gather path: live-row tile_m, whole-K tile_k.

    Splitting k is not available here (the gather DMA reads whole rows, and the
    kernel enforces tile_k == size_k), so the fit trades tile_n and tile_m. Each
    n tile re-issues the whole per-row gather, so tile_n is only taken below
    `calculate_tiling`'s MXU floor once tile_m has nothing left to give.
    """
    info = pltpu.get_tpu_info()
    lanes = info.num_lanes
    sublane = dims.size_lhs_sublane
    fuse_factor = 2 if fuse_act is not None else 1
    law = gather_tiling(dims.size_m // max(ep_size, 1), dims.size_group)
    pick = tuning.tile_m or law
    if pick % sublane:
        raise ValueError(f"tile_m={pick} is not a multiple of the lhs sublane "
                         f"block {sublane}")
    tile_m = min(pick, gmm_v2.align_to(dims.size_m, sublane))
    size_n_per_rhs = dims.size_n // fuse_factor
    num_n_tiles = 1
    tile_n = gmm_v2.align_to(size_n_per_rhs, lanes)
    # gmm_v2's MXU floor: below it the dot stalls.
    tile_n_floor = min(2 * info.mxu_column_size // fuse_factor,
                       gmm_v2.align_to(size_n_per_rhs, lanes))
    while (estimate_vmem_bytes(dims,
                               lhs_cfgs,
                               rhs_cfgs,
                               tile_m,
                               tile_n,
                               fuse_factor,
                               tuning=tuning,
                               num_lanes=lanes,
                               num_sublanes=info.num_sublanes)
           > vmem_limit_bytes):
        if tile_n > tile_n_floor or (tile_m <= sublane and tile_n > lanes):
            num_n_tiles += 1
            tile_n = gmm_v2.align_to(size_n_per_rhs,
                                     num_n_tiles * lanes) // num_n_tiles
        elif tile_m > sublane:
            # Keep the sublane multiple generate_block_specs' BoundedSlice
            # assumes, or the out block covers fewer rows than acc_ref.
            tile_m = max(sublane, (tile_m // 2) // sublane * sublane)
        else:
            raise ValueError(
                f"fused_permute_gmm does not fit in {vmem_limit_bytes} B "
                f"of VMEM at {dims=}")
    return gmm_v2.TileSizes(tile_m=tile_m,
                            tile_k=dims.size_k,
                            tile_n=tile_n,
                            bucket_base=bucket_base(tile_m, sublane,
                                                    tuning.num_buckets))


def gather_tile_fn(ep_size: int,
                   tuning: Tuning | None = None) -> gmm_v2.TileFn:
    """`gather_tile_sizes` bound for `make_gmm_configs(tile_info=...)`."""
    return functools.partial(gather_tile_sizes,
                             tuning=tuning or Tuning(),
                             ep_size=ep_size)


@dataclasses.dataclass(frozen=True)
class Config:
    """One call's resolved configuration: gmm_v2's, plus what the gather adds."""
    gmm: gmm_v2.GmmConfigs
    tuning: Tuning
    num_lanes: int
    num_sublanes: int
    mxu_columns: int

    @property
    def dims(self) -> gmm_v2.Dimensions:
        return self.gmm.dims

    @property
    def tiles(self) -> gmm_v2.TileSizes:
        return self.gmm.tiles

    @property
    def sublane(self) -> int:
        return self.gmm.dims.size_lhs_sublane

    @property
    def num_n_tiles(self) -> int:
        return pl.cdiv(self.gmm.out_size_n, self.tiles.tile_n)

    @property
    def max_num_gm(self) -> int:
        """gmm_v2's bound on gm tiles: every group plus every tile boundary."""
        return (self.dims.size_group +
                pl.cdiv(self.dims.size_m, self.tiles.tile_m) - 1)

    @property
    def index_rows(self) -> int:
        """Length of the padded index table.

        Every row of every tile then names a valid pool row, so a tile that runs
        past the last live row reads a real one (bounds checks are off).
        """
        return (gmm_v2.align_to(self.dims.size_m, self.tiles.tile_m) +
                self.tiles.tile_m)

    @property
    def acc_columns(self) -> int:
        if self.gmm.fuse_act is not None:
            return 2 * self.tiles.tile_n
        return self.tiles.tile_n

    @property
    def row_chunks(self) -> int:
        """Lane-blocks of a gathered row that hold data.

        The LHS build walks the row two lane-blocks at a time, so an odd count
        would drop the last one. The layer guard keeps those shapes on the
        materialised path; this is the backstop for a direct caller.
        """
        chunks = self.tiles.tile_k // self.num_lanes
        if chunks % 2:
            raise ValueError(
                f"size_k must be a multiple of {2 * self.num_lanes}: the LHS "
                f"build pairs lane-blocks, and {self.tiles.tile_k} leaves one "
                "unpaired")
        return chunks

    @property
    def ring_chunks(self) -> int:
        """Lane-blocks a ring row occupies, pad included."""
        return ring_chunks(self.row_chunks, self.tuning.ring_pad_slots,
                           self.num_sublanes)

    @property
    def gather_buffer_shape(self) -> tuple[int, int, int, int]:
        """The ring: [num_slots, tile_m, ring_chunks, lanes] in the pool dtype."""
        return (self.tuning.num_slots, self.tiles.tile_m, self.ring_chunks,
                self.num_lanes)

    @property
    def dot_blocks_per_tile(self) -> int:
        """How many times the dot calls the issue hook, from `inner_kernel` itself.

        The hook slices its rows by this count, so a second derivation that
        disagrees would issue fewer rows than the tile waits for.
        """
        return self.gmm.blocks_per_tile(self.mxu_columns)

    @property
    def zero_tile_rows(self) -> int:
        out_bytes = jnp.dtype(self.gmm.out_dtype).itemsize
        return min(ZERO_TILE_TARGET_BYTES // self.num_lanes // out_bytes,
                   self.dims.size_m)

    @property
    def out_shape(self) -> jax.ShapeDtypeStruct:
        aligned_n = gmm_v2.align_to(self.gmm.out_size_n, self.num_lanes)
        return jax.ShapeDtypeStruct((self.dims.size_m, aligned_n),
                                    self.gmm.out_dtype)

    @property
    def kernel_name(self) -> str:
        return "fpg-" + gmm_v2.get_scope_name(self.gmm)

    @property
    def kernel_metadata(self) -> dict[str, Any]:
        return gmm_v2.get_metadata(self.gmm)

    @property
    def cost_estimate(self) -> pl.CostEstimate:
        return gmm_v2.get_cost_estimate(self.gmm)
