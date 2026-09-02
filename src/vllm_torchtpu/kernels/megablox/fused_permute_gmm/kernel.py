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
"""Kernel body of the fused permute + grouped matmul.

rhs and out stay pipelined by `emit_pipeline`; the LHS tile is gathered per
row into a VMEM ring, `lookahead` tiles ahead of the dot, and handed to
`gmm_v2.inner_kernel` unchanged.
"""
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.megablox.fused_permute_gmm import config, memory_ref

RowIssueFn = Callable[[jax.Array], None]
TileIssueFn = Callable[[jax.Array | int, jax.Array | int], None]


def tile_rows(
    step: jax.Array,
    *,
    metadata: gmm_v2.MetadataRef,
    num_gm: jax.Array,
    sublane: int,
) -> tuple[jax.Array, jax.Array]:
    """Sublane-aligned base row of `step`'s gm tile, and its live count.

    Mirrors `gmm_v2.inner_kernel`'s m_offset / m_end_local, so gathered row j is
    the row the contiguous path's tiled_lhs_ref[j] holds. tile_k == size_k, so
    the grid's k extent is 1 and a gathered row is a whole pool row.
    """
    gm = lax.rem(step, num_gm)
    m_start = metadata.gm_id_to_m_offset[gm]
    m_end = metadata.gm_id_to_m_offset[gm + 1]
    m_offset = m_start - m_start % sublane
    return m_offset, m_end - m_offset


def issue_row(
    row: jax.Array,
    *,
    pool_ref: jax.Ref,
    idx_smem: jax.Ref,
    gather: memory_ref.GatherRefs,
    slot: jax.Array,
    m_offset: jax.Array,
) -> None:
    """Start the DMA of gathered row `row` of the tile into `slot`."""
    pltpu.make_async_copy(
        pool_ref.at[pl.ds(idx_smem[m_offset + row], 1)],
        gather.buf.at[slot, pl.ds(row, 1)],
        gather.sem.at[slot],
    ).start()


def issue_row_sized(
    row: int,
    *,
    pool_ref: jax.Ref,
    idx_smem: jax.Ref,
    gather: memory_ref.GatherRefs,
    slot: jax.Array,
    m_offset: jax.Array,
    live: jax.Array,
) -> None:
    """One row DMA of size 0 past the live rows: no bytes, no signal, one descriptor."""
    size = jnp.where(row < live, 1, 0)
    pltpu.make_async_copy(
        pool_ref.at[pl.ds(idx_smem[m_offset + row], size)],
        gather.buf.at[slot, pl.ds(row, size)],
        gather.sem.at[slot],
    ).start()


class DotIssueHook:
    """Issues the next tile's rows from inside the dot, one slice per matmul block.

    A DMA start has no consumer in its region, so LLO sinks it to the region's
    end unless a later load may alias its target. Each slice is therefore
    followed by a one-vreg load from the ring at a traced slot, folded into the
    block's product through a select that is always false at run time: the
    load pins the slice inside the matmul stream and adds 0.0.
    """

    def __init__(self, *, pool_ref: jax.Ref, idx_smem: jax.Ref,
                 gather: memory_ref.GatherRefs, slot: jax.Array,
                 next_slot: jax.Array, m_offset_next: jax.Array,
                 live_next: jax.Array, never: jax.Array, cfg: config.Config):
        self.issue = functools.partial(issue_row_sized,
                                       pool_ref=pool_ref,
                                       idx_smem=idx_smem,
                                       gather=gather,
                                       slot=next_slot,
                                       m_offset=m_offset_next,
                                       live=live_next)
        self.ring = gather.buf.at[slot]
        self.never = never
        self.num_blocks = cfg.dot_blocks_per_tile
        self.lanes = cfg.num_lanes

    def at_block(self, block: int, bucket_m: int,
                 col_size: int) -> jax.Array | int:
        rows = -(-bucket_m // self.num_blocks)
        if block * rows >= bucket_m:
            return 0  # every row of the tile is issued; nothing left to pin
        for row in range(block * rows, min((block + 1) * rows, bucket_m)):
            self.issue(row)
        sentinel = self.ring[pl.ds(0, 1),
                             pl.ds(0, col_size // self.lanes), :].reshape(
                                 1, col_size)
        return jnp.where(self.never, sentinel, jnp.zeros_like(sentinel))


def _issue_block(block: jax.Array, carry: int, *, issue: RowIssueFn,
                 unroll: int) -> int:
    base = block * unroll
    for t in range(unroll):
        issue(base + t)
    return carry


def _issue_tail(j: jax.Array, carry: int, *, issue: RowIssueFn,
                n_full: jax.Array, unroll: int) -> int:
    issue(n_full * unroll + j)
    return carry


def issue_tile(
    step: jax.Array,
    slot: jax.Array,
    *,
    pool_ref: jax.Ref,
    idx_smem: jax.Ref,
    scratch: memory_ref.PipelineScratchRef,
    num_gm: jax.Array,
    cfg: config.Config,
    first: jax.Array | int = 0,
    live_cap: jax.Array | None = None,
) -> None:
    """Issue the tile's LIVE rows, from `first` on, up to `live_cap` if given.

    The trip count is a runtime value, so a dead row costs nothing rather than
    an issue slot each: `pl.when` predicates the instruction, not its issue.
    The body is unrolled in blocks because a pure loop gives up the cross-row
    ILP an unrolled one keeps, and `fori_loop`'s own `unroll=` rejects dynamic
    bounds, so the blocking is manual.

    Reading past the live rows would still be safe: the index table is
    edge-padded to `align_to(size_m, tile_m) + tile_m`, so every row of every
    tile names a valid pool row, and `mode="edge"` makes the pad re-read one
    row rather than reach past the pool.
    """
    m_offset, live = tile_rows(step,
                               metadata=scratch.metadata,
                               num_gm=num_gm,
                               sublane=cfg.sublane)
    issue = functools.partial(issue_row,
                              pool_ref=pool_ref,
                              idx_smem=idx_smem,
                              gather=scratch.gather,
                              slot=slot,
                              m_offset=m_offset)
    unroll = cfg.tuning.issue_unroll
    n = jnp.minimum(live if live_cap is None else live_cap, cfg.tiles.tile_m)
    if not isinstance(first, int) or first:
        # The rows past what the dot already issued: shift the loop's origin.
        issue = functools.partial(_issue_from, issue=issue, first=first)
        n = jnp.maximum(n - first, 0)
    n_full = n // unroll
    lax.fori_loop(0, n_full,
                  functools.partial(_issue_block, issue=issue, unroll=unroll),
                  0)
    lax.fori_loop(
        0, n - n_full * unroll,
        functools.partial(_issue_tail,
                          issue=issue,
                          n_full=n_full,
                          unroll=unroll), 0)


def _issue_from(row: jax.Array, *, issue: RowIssueFn,
                first: jax.Array | int) -> None:
    issue(first + row)


def wait_tile(
    step: jax.Array,
    slot: jax.Array,
    *,
    scratch: memory_ref.PipelineScratchRef,
    num_gm: jax.Array,
    cfg: config.Config,
) -> None:
    """Wait for exactly what `issue_tile` issued.

    The live rows are the tile's FIRST `live`, so one descriptor of that many
    rows describes exactly the bytes issued. Same-src/dst is the documented
    idiom for waiting once on N DMAs sharing a semaphore.
    """
    _, live = tile_rows(step,
                        metadata=scratch.metadata,
                        num_gm=num_gm,
                        sublane=cfg.sublane)
    gather = scratch.gather
    pltpu.make_async_copy(gather.buf.at[slot, pl.ds(0, live)],
                          gather.buf.at[slot, pl.ds(0, live)],
                          gather.sem.at[slot]).wait()


def gathered_rows(gather: memory_ref.GatherRefs, slot: jax.Array,
                  cfg: config.Config) -> jax.Array:
    """The tile's gathered rows, for the dot.

    `inner_kernel` reshapes to `tile_k` whatever it is given, so passing the
    blocked buffer straight through leaves that relayout to it rather than
    doing a second one here.
    """
    if cfg.tuning.blocked_rows:
        return gather.buf[slot]
    return gather.buf[slot].reshape(cfg.tiles.tile_m, cfg.tiles.tile_k)


def warm_up(*, issue: TileIssueFn, num_steps: jax.Array,
            tuning: config.Tuning) -> None:
    """Issue the first `lookahead` tiles, each only if the grid has its step.

    Tile t lands in slot t, is waited by step t and by nothing else, so a grid
    shorter than the lookahead must not issue it: a nonzero DMA semaphore at
    exit halts the core and fails the whole slice. Step 0 always exists once
    the body runs.
    """
    for tile in range(tuning.lookahead):
        if tile == 0:
            issue(0, 0)
        else:
            pl.when(tile < num_steps)(functools.partial(
                issue, tile, tile % tuning.num_slots))


def _issue_lookahead(*, issue: TileIssueFn, step: jax.Array, slot: jax.Array,
                     lookahead: int) -> None:
    issue(step + lookahead, slot)


def pipeline_body(
        # In (pipelined)
        tiled_rhs_ref: gmm_v2.RhsRef,
        # Out (pipelined)
        tiled_out_ref: jax.Ref,
        # Scratch
        scratch: memory_ref.PipelineScratchRef,
        *,
        cfg: config.Config,
        idx_smem: jax.Ref,  # int32[index_rows] SMEM, kernel-invariant
        pool_ref: jax.Ref,  # [size_src, size_k // lanes, lanes] HBM
) -> None:
    """One grid step: issue tile s + lookahead, wait tile s, run the dot on it."""
    tuning = cfg.tuning
    num_gm = pl.num_programs(1)
    step = pl.program_id(0) * num_gm + pl.program_id(1)
    num_steps = pl.num_programs(0) * num_gm
    issue = functools.partial(issue_tile,
                              pool_ref=pool_ref,
                              idx_smem=idx_smem,
                              scratch=scratch,
                              num_gm=num_gm,
                              cfg=cfg)

    pl.when(step == 0)(functools.partial(warm_up,
                                         issue=issue,
                                         num_steps=num_steps,
                                         tuning=tuning))

    slot = lax.rem(step, tuning.num_slots)
    next_slot = (slot + tuning.lookahead) % tuning.num_slots
    if tuning.issue_in_dot:
        hook, tail = _in_dot_issue(step,
                                   slot,
                                   next_slot,
                                   num_steps,
                                   pool_ref=pool_ref,
                                   idx_smem=idx_smem,
                                   scratch=scratch,
                                   num_gm=num_gm,
                                   cfg=cfg)
    else:
        hook, tail = None, None
        # Issue-first: feed the engine for the next tile before this tile's
        # data is touched, so the next gather runs under this tile's dot.
        pl.when(step + tuning.lookahead < num_steps)(functools.partial(
            _issue_lookahead,
            issue=issue,
            step=step,
            slot=next_slot,
            lookahead=tuning.lookahead))

    wait_tile(step, slot, scratch=scratch, num_gm=num_gm, cfg=cfg)
    gmm_v2.inner_kernel(gathered_rows(scratch.gather, slot, cfg),
                        tiled_rhs_ref,
                        tiled_out_ref,
                        scratch.partial_out,
                        scratch.acc,
                        scratch.metadata,
                        cfgs=cfg.gmm,
                        inject=hook)
    if tail is not None:
        tail()


def _in_dot_issue(
    step: jax.Array,
    slot: jax.Array,
    next_slot: jax.Array,
    num_steps: jax.Array,
    *,
    pool_ref: jax.Ref,
    idx_smem: jax.Ref,
    scratch: memory_ref.PipelineScratchRef,
    num_gm: jax.Array,
    cfg: config.Config,
) -> tuple[DotIssueHook, Any]:
    """The hook issuing tile `step + lookahead`'s first `bucket_m` rows from
    inside the dot, and the loop issuing the rest after it.

    The step guard is a size, not a branch: past the grid every descriptor is
    0-size. The dot's bucket height is recomputed the way gmm_v2 picks it, so
    the tail knows where the hook stopped; the tail's rows are still two dots
    ahead of their wait at the default lookahead.
    """
    tuning = cfg.tuning
    next_step = step + tuning.lookahead
    m_offset_next, live_next = tile_rows(next_step,
                                         metadata=scratch.metadata,
                                         num_gm=num_gm,
                                         sublane=cfg.sublane)
    live_next = jnp.where(next_step < num_steps,
                          jnp.minimum(live_next, cfg.tiles.tile_m), 0)
    _, live = tile_rows(step,
                        metadata=scratch.metadata,
                        num_gm=num_gm,
                        sublane=cfg.sublane)
    bucket_base = cfg.tiles.bucket_base
    issued = jnp.minimum(bucket_base * (live // bucket_base + 1),
                         cfg.tiles.tile_m)
    hook = DotIssueHook(pool_ref=pool_ref,
                        idx_smem=idx_smem,
                        gather=scratch.gather,
                        slot=slot,
                        next_slot=next_slot,
                        m_offset_next=m_offset_next,
                        live_next=live_next,
                        never=step < 0,
                        cfg=cfg)
    tail = functools.partial(issue_tile,
                             next_step,
                             next_slot,
                             pool_ref=pool_ref,
                             idx_smem=idx_smem,
                             scratch=scratch,
                             num_gm=num_gm,
                             cfg=cfg,
                             first=issued,
                             live_cap=live_next)
    return hook, tail


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Ref,
    group_offset_ref: jax.Ref,
    idx_smem: jax.Ref,
    # In
    lhs_ref: jax.Ref,
    rhs_ref: gmm_v2.WeightsRef,
    # Out
    out_ref: jax.Ref,
    # Scratch
    scratch: memory_ref.ScratchRef,
    *,
    cfg: config.Config,
) -> None:
    metadata = scratch.pipeline.metadata
    num_gm = gmm_v2.fill_metadata(lhs_group_sizes_ref,
                                  group_offset_ref,
                                  metadata,
                                  cfgs=cfg.gmm)

    if cfg.gmm.zero_init:
        zero_size = gmm_v2.zero_out_start(out_ref,
                                          scratch.zero,
                                          scratch.zero_sem,
                                          metadata,
                                          num_gm,
                                          dims=cfg.dims)

    (_, rhs_spec), out_spec = gmm_v2.generate_block_specs(metadata, cfg.gmm)

    if cfg.gmm.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfg.gmm.out_size_n:],
                                  rhs_ref)
        rhs_ref = gmm_v2.FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        rhs_spec = gmm_v2.FusedWeightsRef(gate=rhs_spec, up=rhs_spec)

    body = functools.partial(pipeline_body,
                             cfg=cfg,
                             idx_smem=idx_smem,
                             pool_ref=lhs_ref)
    pipeline_fn = pltpu.emit_pipeline(body,
                                      grid=(cfg.num_n_tiles, num_gm, 1),
                                      in_specs=(rhs_spec, ),
                                      out_specs=out_spec)

    out_tiled = out_ref.reshape(-1, cfg.sublane, out_ref.shape[-1])
    pipeline_fn(rhs_ref, out_tiled, scratches=[scratch.pipeline])

    if cfg.gmm.zero_init:
        gmm_v2.zero_out_end(out_ref,
                            scratch.zero_sem,
                            zero_size,
                            dims=cfg.dims)
