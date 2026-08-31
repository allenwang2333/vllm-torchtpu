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
"""Fused permute + grouped matmul.

Reads the LHS per-row from an un-permuted pool inside the kernel, so the MoE
dispatch permute never materialises. Equal to
`gmm_v2(lhs[gather_indices], ...)` over the valid prefix, bit for bit up to the
sign of zero -- a revisited gm tile accumulates onto its own output, which turns
-0.0 into +0.0.

Design, relative to a contiguous `gmm_v2`:

  * The gather tax is per-gm-tile, not per launch, so `tile_m` is sized from
    the live rows per group rather than to fill VMEM -- see `gather_tiling`.
  * The index table is a kernel-invariant SMEM scalar-prefetch operand, so
    `tile_m` carries no window-copy alignment constraint.
  * A bf16 pool is read as `bf16[size_src, size_k // 128, 128]`, which puts
    the (2, 1) sub-word packing on the block axis instead of on rows, so a
    row is whole 32-bit words and a plain per-row DMA is legal. Reading it
    through an int32 row-pair view instead would move twice the bytes and
    need a per-row parity bit -- as a lane-padded `[size_m, 1]` operand,
    that bit cost more than the kernel saved.
"""
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2

# A scalar-prefetch operand cliffs well below SMEM capacity; the budget is measured on v7x,
# whose SMEM is 1 MiB, and carried as a fraction so it follows the hardware. The invariant
# index table costs 4 bytes per row.
_SMEM_INDEX_FRACTION = 480_000 / (1 << 20)
_SMEM_INDEX_LIMIT: list[int | None] = [None]


def smem_index_limit_bytes() -> int:
    """Bytes of SMEM this kernel will spend on the invariant index table.

    Resolved from the device once and cached. Falls back to the v7x measurement when there is
    no device: op construction can run before one exists, and this is reached from there.
    """
    if _SMEM_INDEX_LIMIT[0] is None:
        try:
            _SMEM_INDEX_LIMIT[0] = int(
                pltpu.get_tpu_info().smem_capacity_bytes *
                _SMEM_INDEX_FRACTION)
        except Exception:
            _SMEM_INDEX_LIMIT[0] = 480_000
    return _SMEM_INDEX_LIMIT[0]


# The gather runs `_LOOKAHEAD` tiles ahead in `_NUM_SLOTS` buffers: the issue
# site names tile `s + _LOOKAHEAD`, so depth is the lookahead and not the slot
# count, and the ring needs `_NUM_SLOTS > _LOOKAHEAD` to stay race-free.
_NUM_SLOTS = 4
_LOOKAHEAD = 3
assert _NUM_SLOTS > _LOOKAHEAD, "the ring aliases a slot before its wait"

# Unroll factor for the row-issue loop, whose trip count is the tile's LIVE row
# count: `pl.when` only predicates an instruction, so a dead row would still
# take its issue slot.
_DYN_LOOP = 16

# `bucket_base = tile_m // _NUM_BUCKETS` sizes gmm_v2's dot to the rows a tile
# carries; unless that quotient floors to a sublane multiple dividing `tile_m`
# it silently becomes ONE bucket -- a full `tile_m`-row dot at any fill (512).
_NUM_BUCKETS = 6

# Where the gather buffer's relayout happens: 1 hands `inner_kernel` the blocked
# buffer, which reshapes whatever it gets; 0 does it here instead.
_BLOCKED_ROWS = 1

# Overrides the tile `gather_tiling` derives, which reads mean live rows per
# group and is blind to `max_group` -- where the cliff is, since a tile below it
# splits groups. None keeps the law; a value must be a sublane multiple.
_TILE_M = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GatherRefs:
    buf: Any  # [_NUM_SLOTS, tile_m, tile_k // 128, 128] pool dtype
    sem: Any  # DMA[_NUM_SLOTS]


def gather_tiling(size_m: int, size_group: int) -> int:
    """tile_m for the gather path: mean live rows per group, plus spread.

    Sizing to fill VMEM as `calculate_tiling` does is wrong here -- a gm tile
    carries only ~size_m/size_group live rows, and tile_m-sized work is paid in
    full per tile. Undersizing is worse than oversizing: a tile smaller than a
    group splits it, and the split tiles' dots are exposed at reduced MXU height
    on top of the per-tile cost, hence the spread term and the 64 floor for
    skewed routing. Under EP only ~1/ep_size of the gathered rows are local,
    which the caller folds into `size_m`.
    """
    live = -(-max(16, size_m) // max(size_group, 1))
    live += int(2.5 * live**0.5 + 0.999)
    return int(min(512, max(64, gmm_v2.align_to(live, 32))))


def _vmem_estimate(dims, lhs_cfgs, rhs_cfgs, tile_m, tile_n,
                   fuse_factor) -> int:
    """This kernel's VMEM footprint at tile_k == size_k.

    `calculate_tiling`'s estimate does not apply: the pipelined LHS tile is
    replaced by the gather slots, and tile_m comes from the live-row law
    rather than from the fit.
    """
    tile_k = dims.size_k
    lhs_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8

    gather = _NUM_SLOTS * tile_m * tile_k * lhs_bytes
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


def _gather_tile_fn(ep_size: int):
    """Tile sizes for the gather path: live-row tile_m, whole-K tile_k.

    Splitting k is not available here (the gather DMA reads whole rows, and
    the kernel enforces tile_k == size_k), so the fit trades tile_n and
    tile_m. Each n tile re-issues the whole per-row gather, so tile_n is
    only taken below `calculate_tiling`'s MXU floor once tile_m has nothing
    left to give.
    """

    def tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act):
        info = pltpu.get_tpu_info()
        lanes = info.num_lanes
        fuse_factor = 2 if fuse_act is not None else 1
        law = gather_tiling(dims.size_m // max(ep_size, 1), dims.size_group)
        pick = _TILE_M or law
        tile_m = min(pick, gmm_v2.align_to(dims.size_m, dims.size_lhs_sublane))
        size_n_per_rhs = dims.size_n // fuse_factor
        num_n_tiles = 1
        tile_n = gmm_v2.align_to(size_n_per_rhs, lanes)
        # gmm_v2's MXU floor: below it the dot stalls.
        tile_n_floor = min(2 * info.mxu_column_size // fuse_factor,
                           gmm_v2.align_to(size_n_per_rhs, lanes))
        while (_vmem_estimate(dims, lhs_cfgs, rhs_cfgs, tile_m, tile_n,
                              fuse_factor) > vmem_limit_bytes):
            if tile_n > tile_n_floor or (tile_m <= dims.size_lhs_sublane
                                         and tile_n > lanes):
                num_n_tiles += 1
                tile_n = gmm_v2.align_to(size_n_per_rhs,
                                         num_n_tiles * lanes) // num_n_tiles
            elif tile_m > dims.size_lhs_sublane:
                # Keep the sublane multiple generate_block_specs' BoundedSlice
                # assumes, or the out block covers fewer rows than acc_ref.
                tile_m = max(dims.size_lhs_sublane, (tile_m // 2) //
                             dims.size_lhs_sublane * dims.size_lhs_sublane)
            else:
                raise ValueError(
                    f"fused_permute_gmm does not fit in {vmem_limit_bytes} B "
                    f"of VMEM at {dims=}")
        # `bucket_base` must divide tile_m and stay a sublane multiple, or
        # gmm_v2's bucket switch indexes a bucket the block specs do not cover.
        bucket_base = tile_m
        if _NUM_BUCKETS > 1:
            cand = tile_m // _NUM_BUCKETS
            cand = (cand // dims.size_lhs_sublane) * dims.size_lhs_sublane
            if cand > 0 and tile_m % cand == 0:
                bucket_base = cand
        return gmm_v2.TileSizes(tile_m=tile_m,
                                tile_k=dims.size_k,
                                tile_n=tile_n,
                                bucket_base=bucket_base)

    return tile_fn


def _fused_permute_gmm_inner(
        # In (pipelined)
        tiled_rhs_ref,
        # Out (pipelined)
        tiled_out_ref,
        # Scratch
        partial_out_ref,
        acc_ref,
        metadata_ref: gmm_v2.MetadataRef,
        gather: GatherRefs,
        *,
        cfgs: gmm_v2.GmmConfigs,
        idx_smem,  # closed over: int32[padded_m] SMEM, kernel-invariant
        lhs_pool,  # closed over: [size_src, size_k // 128, 128] HBM
):
    """Pipeline body. rhs/out stay pipelined; the LHS tile is gathered."""
    tile_m = cfgs.tiles.tile_m
    tile_k = cfgs.tiles.tile_k
    sublane = cfgs.dims.size_lhs_sublane

    # tile_k == size_k (enforced below), so the grid's k extent is 1 and a
    # gathered row is a whole pool row.
    num_gm = pl.num_programs(1)
    s = pl.program_id(0) * num_gm + pl.program_id(1)
    num_steps = pl.num_programs(0) * num_gm

    def m_base(step):
        """Sublane-aligned base row of `step`'s gm tile, and its live count.

        Mirrors gmm_v2.inner_kernel's m_offset/m_end_local, so gathered row j is the
        row the contiguous path's tiled_lhs_ref[j] holds.
        """
        gm = lax.rem(step, num_gm)
        m_start = metadata_ref.gm_id_to_m_offset[gm]
        m_end = metadata_ref.gm_id_to_m_offset[gm + 1]
        m_offset = m_start - m_start % sublane
        return m_offset, m_end - m_offset

    def issue_range(step, slot):
        """Issue the tile's LIVE rows.

        The trip count is a runtime value, so a dead row costs nothing rather
        than an issue slot each -- `pl.when` predicates the instruction, not its
        issue. The body is unrolled in blocks because a pure loop gives up the
        cross-row ILP an unrolled one keeps, and `fori_loop`'s own `unroll=`
        rejects dynamic bounds, so the blocking is manual.

        Reading past the live rows would still be safe: the index table is
        edge-padded to `align_to(size_m, tile_m) + tile_m`, so every row of
        every tile names a valid pool row, and `mode="edge"` makes the pad
        re-read one row rather than reach past the pool.
        """
        m_offset, live = m_base(step)

        def issue_at(j):
            pltpu.make_async_copy(
                lhs_pool.at[pl.ds(idx_smem[m_offset + j], 1)],
                gather.buf.at[slot, pl.ds(j, 1)],
                gather.sem.at[slot],
            ).start()

        u = _DYN_LOOP
        n = jnp.minimum(live, tile_m)

        def issue_block(b, carry):
            base = b * u
            for t in range(u):
                issue_at(base + t)
            return carry

        n_full = n // u
        lax.fori_loop(0, n_full, issue_block, 0)

        def issue_rem(j, carry):
            issue_at(n_full * u + j)
            return carry

        lax.fori_loop(0, n - n_full * u, issue_rem, 0)

    def wait_rows(step, slot):
        """Wait for exactly what `issue_range` issued.

        The live rows are the tile's FIRST `live`, so one descriptor of that
        many rows describes exactly the bytes issued. Same-src/dst is the
        documented idiom for waiting once on N DMAs sharing a semaphore.
        """
        _, live = m_base(step)
        pltpu.make_async_copy(gather.buf.at[slot, pl.ds(0, live)],
                              gather.buf.at[slot, pl.ds(0, live)],
                              gather.sem.at[slot]).wait()

    def rows(slot):
        """The tile's gathered rows, for the dot.

        `inner_kernel` reshapes to `tile_k` whatever it is given, so passing the
        blocked buffer straight through leaves that relayout to it rather than
        doing a second one here.
        """
        if _BLOCKED_ROWS:
            return gather.buf[slot]
        return gather.buf[slot].reshape(tile_m, tile_k)

    def run(slot):
        nxt = (slot + _LOOKAHEAD) % _NUM_SLOTS

        # Issue-first: feed the engine for the next tile before this tile's
        # data is touched, so the next gather runs under this tile's dot.
        @pl.when(s + _LOOKAHEAD < num_steps)
        def _():
            issue_range(s + _LOOKAHEAD, nxt)

        wait_rows(s, slot)
        gmm_v2.inner_kernel(rows(slot),
                            tiled_rhs_ref,
                            tiled_out_ref,
                            partial_out_ref,
                            acc_ref,
                            metadata_ref,
                            cfgs=cfgs)

    @pl.when(s == 0)
    def _():
        # Tile t lands in slot t, is waited by step t and by nothing else, so
        # a grid shorter than the lookahead must not issue it: a nonzero DMA
        # semaphore at exit halts the core and fails the whole slice.
        for _t in range(_LOOKAHEAD):

            def _warm(_t=_t):
                issue_range(_t, _t % _NUM_SLOTS)

            # Step 0 always exists once the body runs.
            if _t == 0:
                _warm()
            else:
                pl.when(_t < num_steps)(_warm)

    run(lax.rem(s, _NUM_SLOTS))


def kernel_main_fpg(
    # Scalar prefetch
    lhs_group_sizes_ref,
    group_offset_ref,
    idx_smem,
    # In
    lhs_ref,
    rhs_ref: gmm_v2.WeightsRef,
    # Out
    out_ref,
    # Scratch
    partial_out_ref,
    acc_ref,
    metadata_ref: gmm_v2.MetadataRef,
    gather: GatherRefs,
    zero_ref,
    semaphore_ref,
    *,
    cfgs: gmm_v2.GmmConfigs,
):
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    num_gm = gmm_v2.fill_metadata(lhs_group_sizes_ref,
                                  group_offset_ref,
                                  metadata_ref,
                                  cfgs=cfgs)

    if cfgs.zero_init:
        zero_size = gmm_v2.zero_out_start(out_ref,
                                          zero_ref,
                                          semaphore_ref,
                                          metadata_ref,
                                          num_gm,
                                          dims=cfgs.dims)

    (_, rhs_spec), out_spec = gmm_v2.generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n:],
                                  rhs_ref)
        rhs_ref = gmm_v2.FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        rhs_spec = gmm_v2.FusedWeightsRef(gate=rhs_spec, up=rhs_spec)

    body = functools.partial(_fused_permute_gmm_inner,
                             cfgs=cfgs,
                             idx_smem=idx_smem,
                             lhs_pool=lhs_ref)

    pipeline_fn = pltpu.emit_pipeline(body,
                                      grid=(num_n, num_gm, 1),
                                      in_specs=(rhs_spec, ),
                                      out_specs=out_spec)

    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    pipeline_fn(rhs_ref,
                out_in,
                scratches=[partial_out_ref, acc_ref, metadata_ref, gather])

    if cfgs.zero_init:
        gmm_v2.zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


@jax.jit(static_argnames=[
    "vmem_limit_bytes",
    "preferred_element_type",
    "acc_dtype",
    "maybe_quantize_lhs",
    "zero_initialize",
    "fuse_act",
    "ep_size",
])
def fused_permute_gmm(
    lhs: jax.Array,  # [size_src, size_k] un-permuted pool
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    gather_indices: jax.Array,  # int32[size_m]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    vmem_limit_bytes: int | None = None,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    ep_size: int = 1,
) -> jax.Array:
    """Grouped matmul over `lhs[gather_indices]` without materialising it.

    Row `i` of the effective LHS is `lhs[gather_indices[i]]`; the output is
    [size_m, size_n] with size_m = gather_indices.shape[0]. `ep_size` is the
    shard count the indices span; it feeds `gather_tiling`.

    `lhs` is the un-permuted pool, either `[size_src, size_k]` or the blocked
    `[size_src, size_k // 128, 128]` the rows are actually read in. size_k
    must be a multiple of 128 either way; a caller that cannot meet that
    should materialise the permute and use `gmm_v2`.
    """
    if lhs.dtype.itemsize not in (2, 4):
        raise NotImplementedError(
            f"pool dtype {lhs.dtype} unsupported: only 2- and 4-byte pools "
            "are implemented")
    if gather_indices.ndim != 1:
        raise ValueError("gather_indices must be 1-D")

    lanes = pltpu.get_tpu_info().num_lanes
    if lhs.ndim == 3:
        if lhs.shape[-1] != lanes:
            raise ValueError(
                f"a blocked pool must be [size_src, size_k // {lanes}, "
                f"{lanes}], got {lhs.shape}")
        pool = lhs
    elif lhs.ndim == 2:
        if lhs.shape[1] % lanes:
            raise NotImplementedError(
                f"size_k {lhs.shape[1]} must be a multiple of {lanes}")
        pool = lhs.reshape(lhs.shape[0], lhs.shape[1] // lanes, lanes)
    else:
        raise ValueError(f"pool must be 2-D or 3-D, got {lhs.shape}")
    size_k = pool.shape[1] * lanes

    size_m = gather_indices.shape[0]
    # The kernel reshapes the output by its sublane block, so pad the rows up to
    # it, as gmm_v2 pads its lhs. The extra rows sit past every group, so they
    # gather row 0 and add no work, and they are sliced back off below.
    sublane = gmm_v2._get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = gmm_v2.align_to(size_m, sublane)
    if padded_size_m != size_m:
        gather_indices = jnp.pad(gather_indices, (0, padded_size_m - size_m))
    smem_limit = smem_index_limit_bytes()
    if padded_size_m * 4 > smem_limit:
        raise NotImplementedError(
            f"{size_m} indices exceed the SMEM scalar-operand budget "
            f"({smem_limit} B); materialise the permute and use "
            "gmm_v2")

    if group_offset is None:
        group_offset = jnp.zeros((1, ), jnp.int32)
    elif group_offset.ndim == 0:
        group_offset = group_offset[None]
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    cfgs = gmm_v2.make_gmm_configs(jax.ShapeDtypeStruct(
        (padded_size_m, size_k), lhs.dtype),
                                   rhs,
                                   rhs_scale,
                                   rhs_bias,
                                   group_sizes,
                                   group_offset,
                                   tile_info=_gather_tile_fn(ep_size),
                                   vmem_limit_bytes=vmem_limit_bytes,
                                   out_dtype=preferred_element_type,
                                   acc_dtype=acc_dtype,
                                   maybe_quantize_lhs=maybe_quantize_lhs,
                                   zero_initialize=zero_initialize,
                                   fuse_act=fuse_act)

    dims, tiles = cfgs.dims, cfgs.tiles
    if tiles.tile_k != dims.size_k:
        raise NotImplementedError(
            "fused_permute_gmm gathers whole rows, so tile_k must be size_k "
            f"({tiles.tile_k} != {dims.size_k})")
    # Pad the index table so a tile that runs past the last live row still
    # names a valid pool row (bounds checks are off).
    pad_to = gmm_v2.align_to(dims.size_m, tiles.tile_m) + tiles.tile_m
    idx = jnp.pad(gather_indices.astype(jnp.int32),
                  (0, pad_to - padded_size_m),
                  mode="edge")

    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    rhs_weights = gmm_v2.WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    # Scratch layout is gmm_v2's, verbatim, so inner_kernel and zero_out_*
    # keep their contract; the gather bundle is spliced in ahead of the
    # zero-init pair, which must stay last.
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        gmm_v2.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        GatherRefs(
            buf=pltpu.VMEM(
                (_NUM_SLOTS, tiles.tile_m, tiles.tile_k // lanes, lanes),
                lhs.dtype),
            sem=pltpu.SemaphoreType.DMA((_NUM_SLOTS, )),
        ),
    ]
    if cfgs.zero_init:
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = min(2 * 1024 * 1024 // lanes // out_bytes, dims.size_m)
        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1, )),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = gmm_v2.align_to(cfgs.out_size_n, lanes)
    out = pl.pallas_call(
        functools.partial(kernel_main_fpg, cfgs=cfgs),
        out_shape=jax.ShapeDtypeStruct((dims.size_m, aligned_n),
                                       cfgs.out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                gmm_v2.WeightsRef(weight=pl.BlockSpec(memory_space=pltpu.HBM),
                                  scale=rhs_scale_spec,
                                  bias=rhs_bias_spec),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name="fpg-" + gmm_v2.get_scope_name(cfgs),
        cost_estimate=gmm_v2.get_cost_estimate(cfgs),
        metadata=gmm_v2.get_metadata(cfgs),
    )(group_sizes, group_offset, idx, pool, rhs_weights)
    return out[:size_m, :cfgs.out_size_n]
