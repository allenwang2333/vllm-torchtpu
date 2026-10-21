# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged decode MQA: sum_h(weight_h * relu(dot(q_h, k))).

The resident-Q / paged-K organization follows DeepGEMM's
impls/sm90_fp8_paged_mqa_logits.cuh (5081f851). The issue/consume schedule
follows sglang-jax PR #1800, flash_attention_v2/pipeline.py (e6cffb7e),
specialized to weighted ReLU, with JAX 0.11 accumulator refs. Neither
softmax nor a gathered HBM key workspace is needed.
Query-pair math fragments follow sm100_sparse_mqa_logits.cuh at the same
DeepGEMM revision; resident Q and K reuse remain independent of this tile.
"""

import functools
import math
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.ref import Ref

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS
from vllm_torchtpu.kernels.deepseek_v41.index_selection.paged_mqa_pipeline import (
    BufferPlan,
    Pipeline,
    QueryState,
    QueryTile,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PagedMQAConfig,
)


@dataclass
class MQAOps:
    """Per-query-block operations, instantiated only while tracing Pipeline.run."""

    state: QueryState
    plan: BufferPlan
    lengths: Ref
    q: Ref
    weights: Ref
    pool: tuple[Ref, ...]
    page_table: Ref
    q_scale: Ref | None
    k_scale: tuple[Ref, ...]
    out: Ref
    queries: Ref
    keys: Ref
    scores: Ref
    pages: Ref
    read_sem: Ref
    write_sem: Ref
    q_sem: Ref
    reduction_scales: Ref
    accumulator_scales: Ref | None
    accumulators: tuple[Ref, ...]
    page_records: int
    next_tiles: jax.Array = field(init=False)
    live_tiles: jax.Array = field(init=False)
    row_lengths: jax.Array = field(init=False)
    weight: jax.Array = field(init=False)
    lhs: jax.Array = field(init=False)
    weight_matrix: jax.Array = field(init=False)
    query_scale: jax.Array | None = field(init=False, default=None)

    @property
    def count(self) -> jax.Array:
        return self.state.query.count

    @property
    def metadata_slot(self) -> jax.Array:
        return self.state.step % 2

    @property
    def total_pages(self) -> int:
        return sum(bank.shape[0] for bank in self.pool) // (
            self.page_records // self.plan.key_slots + 1
        )

    def key_slot(self, tile: int | jax.Array) -> jax.Array:
        return (self.state.key_slot + tile) % self.plan.config.key_buffers

    def load_page_table(self, query: QueryTile, slot: int | jax.Array) -> None:
        pltpu.sync_copy(self.page_table.at[query.key_group], self.pages.at[slot])

    def begin_query(self, following: QueryTile) -> None:
        c = self.plan.config
        next_limit = jnp.int32(0)
        for local in range(c.query_tile):
            row = jnp.minimum(following.start + local, self.q.shape[0] - 1)
            next_limit = jnp.maximum(
                next_limit, jnp.where(local < following.count, self.lengths[row], 0)
            )
        self.next_tiles = pl.cdiv(
            jnp.minimum(next_limit, self.plan.page_count * self.page_records),
            c.key_tile,
        )
        self.load_page_table(following, 1 - self.metadata_slot)
        self.prefetch_query(following, 1 - self.metadata_slot)
        limit = jnp.int32(0)
        row_lengths, row_weights, row_scales = ([], [], [])
        for local in range(c.query_tile):
            row = jnp.minimum(self.state.query.start + local, self.q.shape[0] - 1)
            length = jnp.where(local < self.count, self.lengths[row], 0)
            row_lengths.append(length)
            row_weights.append(self.weights[row])
            if self.q_scale is not None:
                row_scales.append(self.q_scale[row])
            limit = jnp.maximum(limit, length)
        self.live_tiles = pl.cdiv(
            jnp.minimum(limit, self.plan.page_count * self.page_records), c.key_tile
        )
        row_weights.extend(
            [jnp.zeros((self.plan.heads,), jnp.float32)]
            * (self.plan.groups * c.query_group - c.query_tile)
        )
        self.weight = jnp.concatenate([w[:, None] for w in row_weights], axis=0)
        if row_scales:
            row_scales.extend(
                [jnp.ones((self.plan.heads,), jnp.float32)]
                * (self.plan.groups * c.query_group - c.query_tile)
            )
            self.query_scale = jnp.concatenate([s[:, None] for s in row_scales], axis=0)
        self.row_lengths = jnp.stack(row_lengths)
        query_count = self.plan.groups * c.query_group
        owner = jnp.arange(query_count * self.plan.heads) // self.plan.heads
        residual = jnp.where(
            jnp.arange(query_count)[:, None] == owner[None, :],
            (self.weight * self.query_scale).T
            if self.query_scale is not None
            else self.weight.T,
            0,
        )
        parts = []
        for _ in range(3):
            part = residual.astype(jnp.bfloat16)
            parts.append(part)
            residual = residual - part.astype(jnp.float32)
        self.weight_matrix = jnp.concatenate(parts, axis=0)

    def wait_query(self) -> None:
        self.copy_queries(self.state.query, self.metadata_slot, wait=True)
        c = self.plan.config
        lhs = self.queries[
            pl.ds(self.metadata_slot * c.query_tile, c.query_tile)
        ].reshape(c.query_tile * self.plan.heads, DIM)
        if self.plan.groups * c.query_group != c.query_tile:
            lhs = jnp.concatenate(
                (
                    lhs,
                    jnp.zeros(
                        (
                            (self.plan.groups * c.query_group - c.query_tile)
                            * self.plan.heads,
                            DIM,
                        ),
                        self.q.dtype,
                    ),
                ),
                axis=0,
            )
        self.lhs = jnp.concatenate((lhs, jnp.zeros_like(lhs)), axis=1)

    def prefetch_query(self, query: QueryTile, slot) -> None:
        """Load the next Q tile, zeroing unused rows for ragged groups."""
        tile = self.plan.config.query_tile
        self.queries[pl.ds(slot * tile, tile)] = jnp.zeros(
            (tile, self.plan.heads, DIM), self.q.dtype
        )
        self.copy_queries(query, slot)

    def copy_queries(self, query: QueryTile, slot, wait: bool = False) -> None:
        for local in range(self.plan.config.query_tile):

            @pl.when(local < query.count)
            def copy(local=local):
                transfer = pltpu.make_async_copy(
                    self.q.at[query.start + local],
                    self.queries.at[slot * self.plan.config.query_tile + local],
                    self.q_sem.at[slot],
                )
                if wait:
                    transfer.wait()
                else:
                    transfer.start()

    def load_page(self, tile, i, metadata_slot, slot):
        page = jnp.clip(
            self.pages[
                metadata_slot,
                tile * (self.plan.config.key_tile // self.page_records) + i,
            ],
            0,
            self.total_pages - 1,
        )
        rows = self.page_records // self.plan.key_slots + 1
        address = page * rows

        def load(bank_id, base):
            pltpu.make_async_copy(
                self.pool[bank_id].at[pl.ds(address - base, rows)],
                self.keys.at[slot, pl.ds(i * rows, rows)],
                self.read_sem.at[slot],
            ).start()

        if len(self.pool) == 1:
            load(0, 0)
        else:
            boundary = self.pool[0].shape[0]
            in_first = address < boundary

            @pl.when(in_first)
            def low():
                load(0, 0)

            @pl.when(~in_first)
            def high():
                load(1, boundary)

    def fetch(self, tile, metadata_slot, slot):
        def body(i, _):
            self.load_page(tile, i, metadata_slot, slot)

        jax.lax.fori_loop(0, self.plan.config.key_tile // self.page_records, body, None)

    def wait_keys(self, tile):
        slot = (self.state.key_slot + tile) % self.plan.config.key_buffers
        pltpu.make_async_copy(
            self.keys.at[slot], self.keys.at[slot], self.read_sem.at[slot]
        ).wait()

    def publish(self, tile, wait: bool = False, *, slot=None) -> None:
        slot = tile % 2 if slot is None else slot
        for local in range(self.plan.config.query_tile):

            @pl.when(local < self.count)
            def copy(local=local):
                transfer = pltpu.make_async_copy(
                    self.scores.at[slot, local],
                    self.out.at[
                        self.state.query.start + local,
                        0,
                        pl.ds(
                            tile * self.plan.config.key_tile, self.plan.config.key_tile
                        ),
                    ],
                    self.write_sem.at[slot, local],
                )
                if wait:
                    transfer.wait()
                else:
                    transfer.start()

    def issue(self, tile, fragment: int, group: int) -> None:
        c = self.plan.config
        group_heads = c.query_group * self.plan.heads
        index = fragment % c.issue_depth * self.plan.groups + group
        acc = self.accumulators[index]
        query = self.lhs[group * group_heads : (group + 1) * group_heads]
        data_rows = self.page_records // self.plan.key_slots
        page_rows = data_rows + 1
        key_parts, scale_parts = ([], [])
        for page in range(
            fragment * self.plan.mxu_size // self.page_records,
            (fragment + 1) * self.plan.mxu_size // self.page_records,
        ):
            words = self.keys[self.key_slot(tile), pl.ds(page * page_rows, data_rows)]
            key_parts.append(pltpu.bitcast(words, self.q.dtype))
            scale_words = self.keys[self.key_slot(tile), page * page_rows + data_rows]
            scale_parts.append(jax.lax.bitcast_convert_type(scale_words, jnp.float32))
        self.accumulator_scales[index] = jnp.concatenate(scale_parts)[None, :]
        keys = jnp.concatenate(key_parts, axis=0)
        rhs = jnp.concatenate((keys, jnp.zeros_like(keys)), axis=1)
        # Head reduction overwrites staged RHS, so every QK group reloads K.
        pltpu.matmul_push_rhs(rhs, 0, fragment % c.issue_depth % 2, transpose=True)
        pltpu.matmul_acc_lhs(acc, query, load_staged_rhs=0)

    def reduce(self, fragment: int, group: int) -> None:
        c = self.plan.config
        size = self.plan.mxu_size
        group_heads = c.query_group * self.plan.heads
        index = fragment % c.issue_depth * self.plan.groups + group
        acc = self.accumulators[index]
        logits = pltpu.matmul_pop(acc)
        logits = jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))
        matrix = self.weight_matrix.reshape(
            3, self.plan.groups * c.query_group, self.plan.groups * group_heads
        )[
            :,
            group * c.query_group : (group + 1) * c.query_group,
            group * group_heads : (group + 1) * group_heads,
        ].reshape(3 * c.query_group, group_heads)
        reduction = self.accumulators[c.issue_depth * self.plan.groups + index]
        self.reduction_scales[index] = self.accumulator_scales[index]
        mxu = fragment % c.issue_depth % 2
        rhs = jnp.pad(logits, ((0, size - group_heads), (0, 0)))
        lhs = jnp.pad(
            matrix, ((0, reduction.shape[0] - matrix.shape[0]), (0, size - group_heads))
        )
        pltpu.matmul_push_rhs(rhs, 0, mxu)
        pltpu.matmul_acc_lhs(reduction, lhs, load_staged_rhs=0)

    def consume(self, fragment: int, group: int) -> jax.Array:
        c = self.plan.config
        index = fragment % c.issue_depth * self.plan.groups + group
        acc = self.accumulators[c.issue_depth * self.plan.groups + index]
        products = pltpu.matmul_pop(acc)
        products = products[: 3 * c.query_group].reshape(
            3, c.query_group, self.plan.mxu_size
        )
        return (products[0] + products[1] + products[2]) * self.reduction_scales[index]

    def store(self, tile, fragment: int, parts: list[jax.Array]) -> None:
        size = self.plan.compute_size
        values = jnp.concatenate(parts, axis=0)[: self.plan.config.query_tile]
        start = tile * self.plan.config.key_tile + fragment * size
        position = start + jnp.arange(size)
        masks = []
        for i in range(0, size, self.page_records):
            page = self.pages[self.metadata_slot, (start + i) // self.page_records]
            valid = ((page >= 0) & (page < self.total_pages)).astype(jnp.int32)
            masks.append(jnp.broadcast_to(valid, (min(self.page_records, size),)))
        valid_page = jnp.concatenate(masks)
        valid = (valid_page[None, :] != 0) & (
            position[None, :] < self.row_lengths[:, None]
        )
        self.scores[tile % 2, :, pl.ds(fragment * size, size)] = jnp.where(
            valid, values, -jnp.inf
        )

    def fill_padding(self) -> None:
        tiles = self.plan.page_count * self.page_records // self.plan.config.key_tile

        @pl.when(self.live_tiles < tiles)
        def initialize_padding():
            self.scores[0] = jnp.full(self.scores.shape[1:], -jnp.inf, jnp.float32)

        for wait in (False, True):

            def body(tile, _, wait=wait):
                self.publish(tile, wait=wait, slot=0)

            jax.lax.fori_loop(
                self.live_tiles,
                self.plan.page_count * self.page_records // self.plan.config.key_tile,
                body,
                None,
            )


def _next_query_tile(query: QueryTile, offsets: Ref, tile_size: int) -> QueryTile:
    """Advance flat Q using only shared-K group boundaries."""
    start = query.start + query.count
    group = jax.lax.while_loop(
        lambda g: (g < offsets.shape[0] - 2) & (start >= offsets[g + 1]),
        lambda g: g + 1,
        query.key_group,
    )
    return QueryTile(start, jnp.minimum(tile_size, offsets[group + 1] - start), group)


def _kernel(
    offsets,
    lengths,
    q,
    weights,
    pool,
    page_table,
    q_scale,
    k_scale,
    out,
    queries,
    keys,
    scores,
    pages,
    read_sem,
    write_sem,
    q_sem,
    key_scales,
    accumulator_scales,
    accumulators,
    *,
    pipeline,
):
    def make_ops(state):
        return MQAOps(
            state,
            pipeline.plan,
            lengths,
            q,
            weights,
            pool,
            page_table,
            q_scale,
            k_scale,
            out,
            queries,
            keys,
            scores,
            pages,
            read_sem,
            write_sem,
            q_sem,
            key_scales,
            accumulator_scales,
            accumulators,
            pipeline.page_records,
        )

    next_query = functools.partial(
        _next_query_tile, offsets=offsets, tile_size=pipeline.plan.config.query_tile
    )
    first = next_query(QueryTile(jnp.int32(0), jnp.int32(0), jnp.int32(0)))
    pipeline.run(make_ops, next_query, first, q.shape[0])


@functools.partial(jax.jit, static_argnames=("page_records", "config"))
def paged_mqa_logits(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    block_table: jax.Array,
    lengths: jax.Array,
    query_start_loc: jax.Array,
    *,
    page_records: int,
    q_scale: jax.Array | None = None,
    k_scale: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    config: PagedMQAConfig | None = None,
) -> jax.Array:
    """Score native paged index keys, returning FP32 [T, page_count * page_records].

    Args:
        q: FP8 E4M3FN [T,H,128], in the same feature order as K.
        weights: FP32 [T,H]. Heads must be multiples of 32.
        pool: One or two INT32 [word_rows,128] packed FP8 banks. Each
            128-key page contains 32 data rows plus one FP32-scale word row.
        block_table: INT32 [G,P] physical page IDs for each shared-K query group.
            Negative or out-of-range entries are holes, producing -inf.
        lengths: INT32 [T] visible context lengths.
        query_start_loc: INT32 [G+1], monotonic offsets partitioning all T query
            rows into groups sharing the same K page table; groups may be empty.
        page_records: Must be 128; each FP8 page occupies 33 word rows.
        q_scale: FP32 [T,H] dequantization scales for FP8 Q; one per D128 row.
        k_scale: Must be None; FP32 K scales are embedded in each packed page.
        config: Internal DMA and query tiles; larger groups use more tiles.

    This is dense paged MQA. Candidate-consuming sparse MQA keeps its separate
    compact-score contract. No union or second key cache is constructed here.
    QK accumulates in FP32; raw ReLU results are rounded to BF16 before scales.
    Q scales are folded into weights; K scales follow head reduction.
    Three BF16 weight components approximate FP32 weighting; bitwise equivalence
    to FP32 vector reduction is not guaranteed.
    """
    tokens, heads, dim = q.shape
    banks = pool if isinstance(pool, tuple) else (pool,)
    scales = (
        () if k_scale is None else k_scale if isinstance(k_scale, tuple) else (k_scale,)
    )
    quantized = q_scale is not None
    if not quantized or k_scale is not None:
        raise ValueError("Packed FP8 pages require Q scales; K scales are embedded")
    if q_scale.shape != (tokens, heads) or q_scale.dtype != jnp.float32:
        raise ValueError("Expected FP32 per-head Q scales")
    key_slots = 4
    key_groups, page_count = block_table.shape
    width = page_count * page_records
    if (
        dim != DIM
        or heads < 16
        or heads % (32 if quantized else 16)
        or (q.dtype != (jnp.float8_e4m3fn if quantized else jnp.bfloat16))
        or (weights.shape != (tokens, heads))
        or (weights.dtype != jnp.float32)
        or (lengths.shape != (tokens,))
        or (lengths.dtype != jnp.int32)
        or (query_start_loc.shape != (key_groups + 1,))
        or (query_start_loc.dtype != jnp.int32)
        or (block_table.dtype != jnp.int32)
        or (key_groups < 1)
        or (page_records != 128)
        or (len(banks) not in (1, 2))
        or any(
            bank.dtype != jnp.int32
            or bank.ndim != 2
            or bank.shape[1] != ROW_WORDS
            or (bank.shape[0] < page_records // key_slots + 1)
            or bank.shape[0] % (page_records // key_slots + 1)
            for bank in banks
        )
    ):
        raise ValueError("Expected FP8 Q and page-aligned packed FP8 K/scale banks")
    if tokens == 0 or width == 0:
        return jnp.empty((tokens, width), jnp.float32)
    info = pltpu.get_tpu_info()
    mxu_size = info.mxu_column_size if info else 2 * DIM
    if info and (info.generation != 7 or info.num_mxus != 2):
        raise NotImplementedError("The paged MQA pipeline currently targets TPU v7")
    if config is None:
        defaults = PagedMQAConfig()
        key_tile = math.gcd(defaults.key_tile, width)
        config = PagedMQAConfig(
            query_tile=min(defaults.query_tile, pl.cdiv(tokens, key_groups)),
            key_tile=key_tile,
            issue_depth=max(1, min(defaults.issue_depth, key_tile // mxu_size)),
        )
    query_tile = min(config.query_tile, tokens)
    query_group = min(config.query_group, query_tile)
    key_tile = min(config.key_tile, width)
    depth = config.issue_depth
    if (
        mxu_size != 2 * DIM
        or key_tile % math.lcm(page_records, mxu_size)
        or width % key_tile
        or mxu_size % min(page_records, mxu_size)
        or (depth != 1 and depth % 2)
        or key_tile // mxu_size % depth
    ):
        raise ValueError(
            "Paged MQA tiles must align to pages, MXU columns and MRB capacity"
        )
    plan = BufferPlan(
        replace(
            config, query_tile=query_tile, key_tile=key_tile, query_group=query_group
        ),
        heads,
        page_count,
        mxu_size,
        q.dtype,
    )
    if info:
        plan.validate(info, tokens, key_groups)
    pipeline = Pipeline(plan, page_records)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def kernel(*refs):
        pl.run_scoped(
            functools.partial(_kernel, *refs, pipeline=pipeline),
            plan.accumulator_shapes(),
        )

    return pl.pallas_call(
        kernel,
        # The unit axis makes each row contiguous for individual output DMAs.
        out_shape=jax.ShapeDtypeStruct((tokens, 1, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=(
                hbm,
                pl.BlockSpec(memory_space=pltpu.VMEM),
                tuple(hbm for _ in banks),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM) if quantized else None,
                tuple(hbm for _ in scales),
            ),
            out_specs=hbm,
            scratch_shapes=plan.scratch_shapes(),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes if info else None,
            disable_bounds_checks=True,
        ),
        name="paged_mqa_logits",
    )(
        query_start_loc, lengths, q, weights, banks, block_table, q_scale, scales
    ).reshape(tokens, width)
