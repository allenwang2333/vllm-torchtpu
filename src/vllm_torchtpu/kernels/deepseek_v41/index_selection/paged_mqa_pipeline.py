# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource ownership and scheduling for paged MQA.

Separates the static schedule from numerical operations, following sglang-jax
PR #1800, flash_attention_v2/{config,pipeline}.py (e6cffb7e). The separate K,
accumulator and output rings adapt DeepGEMM's sparse MQA pipeline to TPU DMA
and MRB. MQA needs no softmax/PV stages.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PagedMQAConfig,
)

if TYPE_CHECKING:
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.paged_mqa_logits import (
        MQAOps,
    )


class QueryTile(NamedTuple):
    start: jax.Array
    count: jax.Array
    key_group: jax.Array


class QueryState(NamedTuple):
    query: QueryTile
    step: jax.Array
    key_slot: jax.Array


@dataclass(frozen=True)
class BufferPlan:
    """Static allocation plan; compiler temporaries/spills are additional."""

    config: PagedMQAConfig
    heads: int
    page_count: int
    mxu_size: int
    q_dtype: jnp.dtype = jnp.dtype(jnp.bfloat16)
    paired: bool = False

    @property
    def key_slots(self) -> int:
        return 4

    @property
    def groups(self) -> int:
        return pl.cdiv(self.config.query_tile, self.config.query_group)

    @property
    def fragments(self) -> int:
        return self.config.key_tile // self.compute_size

    @property
    def compute_size(self) -> int:
        return self.mxu_size

    def scratch_shapes(self) -> tuple:
        c = self.config
        return (
            pltpu.VMEM((2 * c.query_tile, self.heads, DIM), self.q_dtype),
            pltpu.VMEM((c.key_buffers, c.key_tile // 128 * 33, ROW_WORDS), jnp.int32),
            pltpu.VMEM(
                (1, 2, (self.page_count // 2 + 1) * 128)
                if self.paired
                else (2, c.query_tile, c.key_tile),
                jnp.float32,
            ),
            pltpu.SMEM(
                (2, 4, self.page_count) if self.paired else (2, self.page_count),
                jnp.int32,
            ),
            pltpu.SemaphoreType.DMA((c.key_buffers,)),
            pltpu.SemaphoreType.DMA((1 if self.paired else 2, c.query_tile)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((c.issue_depth * self.groups, 1, self.mxu_size), jnp.float32),
            pltpu.VMEM((c.issue_depth * self.groups, 1, self.mxu_size), jnp.float32),
        )

    def accumulator_shapes(self) -> tuple:
        shape = (self.config.query_group * self.heads, self.mxu_size)
        primary = tuple(
            pltpu.ACC(i // self.groups % 2)(shape, jnp.float32)
            for i in range(self.config.issue_depth * self.groups)
        )
        weight_shape = (pl.cdiv(3 * self.config.query_group, 16) * 16, self.mxu_size)
        return primary + tuple(
            pltpu.ACC(i // self.groups % 2)(weight_shape, jnp.float32)
            for i in range(self.config.issue_depth * self.groups)
        )

    def vmem_bytes(self, info: pltpu.TpuInfo, tokens: int, key_groups: int) -> int:
        """Tiled explicit VMEM, including resident weights and page tables."""

        def size(shape, dtype):
            rows = info.get_sublane_tiling(jnp.dtype(dtype))
            return (
                pl.cdiv(shape[-2], rows)
                * rows
                * pl.cdiv(shape[-1], info.num_lanes)
                * info.num_lanes
                * jnp.dtype(dtype).itemsize
                * (shape[0] if len(shape) == 3 else 1)
            )

        scratch = self.scratch_shapes()
        scale_bytes = size((tokens, self.heads), jnp.float32) + sum(
            size(v.shape, v.dtype) for v in scratch[-2:]
        )
        return (
            sum(size(s.shape, s.dtype) for s in scratch[:3])
            + scale_bytes
            + (
                size((tokens, self.heads), jnp.float32)
                + size(
                    (tokens, 4, self.page_count)
                    if self.paired
                    else (key_groups, self.page_count),
                    jnp.int32,
                )
            )
        )

    def smem_bytes(self, tokens: int, key_groups: int) -> int:
        return (
            (8 if self.paired else 2) * self.page_count
            + tokens * (2 if self.paired else 1)
            + key_groups
            + 1
        ) * 4

    def mrb_bytes_per_mxu(self) -> int:
        return (
            pl.cdiv(self.config.issue_depth, 2)
            * self.groups
            * self.config.query_group
            * self.heads
            * self.mxu_size
            * 4
            + pl.cdiv(self.config.issue_depth, 2)
            * self.groups
            * pl.cdiv(3 * self.config.query_group, 16)
            * 16
            * self.mxu_size
            * 4
        )

    def validate(self, info: pltpu.TpuInfo, tokens: int, key_groups: int) -> None:
        if self.vmem_bytes(info, tokens, key_groups) >= info.vmem_capacity_bytes:
            raise ValueError("Paged MQA buffers exceed VMEM; reduce tile sizes")
        if self.smem_bytes(tokens, key_groups) >= info.smem_capacity_bytes:
            raise ValueError("Paged MQA metadata exceeds SMEM; reduce batch/context")
        capacity = info.num_accumulators * info.num_sublanes * info.mxu_column_size * 4
        if self.mrb_bytes_per_mxu() > capacity:
            raise ValueError("Paged MQA exceeds MRB; reduce query tile or issue depth")


@dataclass(frozen=True)
class Pipeline:
    """Trace-time producer/consumer schedule; no runtime Python dispatch.

    K slots are released after their final RHS read. MRB slots are popped before
    reuse. A score slot is published after its final MRB pop, and its DMA must
    finish before the slot is overwritten. Q and page metadata each have two
    slots: the next Q is loaded while the current Q's K tiles are processed.
    A Q slot is reused only after its query block finishes.
    """

    plan: BufferPlan
    page_records: int

    def prefill(self, ops: "MQAOps") -> None:
        buffers = self.plan.config.key_buffers
        for tile in range(buffers - 1):

            @pl.when((ops.state.step == 0) & (tile < ops.live_tiles))
            def current(tile=tile):
                ops.fetch(tile, ops.metadata_slot, ops.key_slot(tile))

            @pl.when((ops.live_tiles + tile < buffers - 1) & (tile < ops.next_tiles))
            def following(tile=tile):
                ops.fetch(
                    tile, 1 - ops.metadata_slot, ops.key_slot(ops.live_tiles + tile)
                )

    def emit_tile(self, ops: "MQAOps", tile, *, prefetch: bool) -> None:
        size = self.plan.compute_size
        following = tile + self.plan.config.key_buffers - 1
        following_tile = jnp.where(
            following < ops.live_tiles, following, following - ops.live_tiles
        )
        metadata = jnp.where(
            following < ops.live_tiles, ops.metadata_slot, 1 - ops.metadata_slot
        )
        depth = self.plan.config.issue_depth
        fragments = self.plan.fragments
        for stage in range(fragments + 2 * depth):
            if stage < fragments and prefetch:
                for page in range(
                    stage * size // self.page_records,
                    (stage + 1) * size // self.page_records,
                ):
                    ops.load_page(
                        following_tile, page, metadata, ops.key_slot(following)
                    )
            parts = []
            for group in range(self.plan.groups):
                if stage >= 2 * depth:
                    parts.append(ops.consume(stage - 2 * depth, group))
                if depth <= stage < fragments + depth:
                    ops.reduce(stage - depth, group)
                if stage < fragments:
                    ops.issue(tile, stage, group)
            if parts:
                ops.store(tile, stage - 2 * depth, parts)

    def step(self, ops: "MQAOps", tile, *, initial: bool) -> None:
        ops.wait_keys(tile)

        if not self.plan.paired:

            @pl.when(tile >= 2)
            def release_output():
                ops.publish(tile - 2, wait=True)

        pending = (
            tile + self.plan.config.key_buffers - 1 < ops.live_tiles + ops.next_tiles
        )

        @pl.when(pending)
        def overlap():
            self.emit_tile(ops, tile, prefetch=True)

        @pl.when(~pending)
        def finish():
            self.emit_tile(ops, tile, prefetch=False)

        if not self.plan.paired:
            ops.publish(tile)

    def drain(self, ops: "MQAOps") -> None:
        if self.plan.paired:
            ops.publish(0)
            ops.publish(0, wait=True)
            return

        @pl.when(ops.live_tiles > 0)
        def finish():
            ops.publish(ops.live_tiles - 1, wait=True)

            @pl.when(ops.live_tiles > 1)
            def previous():
                ops.publish(ops.live_tiles - 2, wait=True)

        ops.fill_padding()

    def run(
        self,
        make_ops: Callable[[QueryState], "MQAOps"],
        next_query: Callable[[QueryTile], QueryTile],
        first_query: QueryTile,
        tokens: int,
    ) -> None:
        def query_block(state):
            ops = make_ops(state)
            following_query = next_query(state.query)
            ops.begin_query(following_query)
            self.prefill(ops)
            ops.wait_query()

            def body(tile, _):
                self.step(ops, tile, initial=False)

            jax.lax.fori_loop(0, ops.live_tiles, body, None)
            self.drain(ops)
            return QueryState(
                following_query, state.step + 1, ops.key_slot(ops.live_tiles)
            )

        initial = QueryState(first_query, jnp.int32(0), jnp.int32(0))
        initial_ops = make_ops(initial)
        initial_ops.load_page_table(first_query, 0)
        initial_ops.prefetch_query(first_query, 0)
        jax.lax.while_loop(
            lambda state: state.query.start < tokens, query_block, initial
        )
