# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request unions and reusable index keys for candidate-filtered decode.

The caller owns cache lifetime and falls back when any union overflows. IDs are
ascending logical blocks; membership is applied before selection for every Q.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import indexer
from vllm_torchtpu.kernels.deepseek_v41.index_selection import decode, tuned_block_sizes
from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    streaming_topk,
)


class UnionMetadata(NamedTuple):
    ids: jax.Array
    membership: jax.Array
    actual_count: jax.Array
    overflow: jax.Array


def _sorted_union_metadata(
    candidates: jax.Array, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    requests, queries, _ = candidates.shape
    valid = (candidates >= 0) & (candidates < logical_blocks)
    # Include query ownership in the sort key so duplicate IDs within one
    # query contribute its membership bit only once.
    encoded = jnp.where(
        valid,
        candidates * queries + jnp.arange(queries)[None, :, None],
        logical_blocks * queries,
    )
    encoded = jnp.sort(encoded.reshape(requests, -1), axis=1)
    blocks = encoded // queries
    valid = blocks < logical_blocks
    previous = jnp.pad(encoded[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    first = valid & (blocks != previous // queries)
    count = first.sum(1).astype(jnp.int32)
    slots = jnp.where(valid, jnp.cumsum(first, axis=1) - 1, capacity_blocks)
    rows = jnp.arange(requests)[:, None]
    ids = jnp.full((requests, capacity_blocks), -1, jnp.int32)
    ids = ids.at[rows, slots].max(jnp.where(valid, blocks, -1), mode="drop")
    bits = jnp.where(valid & (encoded != previous), 1 << (encoded % queries), 0)
    membership = jnp.zeros_like(ids).at[rows, slots].add(bits, mode="drop")
    return UnionMetadata(ids, membership, count, count > capacity_blocks)


def union_metadata(
    candidates: jax.Array, *, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    """Compact [request, query, candidate] IDs without losing overflow evidence."""
    requests, queries, _ = candidates.shape
    if not 1 <= queries <= 8 or min(logical_blocks, capacity_blocks) < 1:
        raise ValueError("Union requires 1..8 queries and positive block capacities")
    # Preserve the calibrated dense route through 128K index records. For
    # longer contexts, bound metadata work by candidate count, not ID space.
    if logical_blocks > 131072 // 8:
        return _sorted_union_metadata(candidates, logical_blocks, capacity_blocks)
    valid = (candidates >= 0) & (candidates < logical_blocks)
    safe = jnp.where(valid, candidates, logical_blocks)
    presence = jnp.zeros((requests, queries, logical_blocks + 1), jnp.int32)
    presence = presence.at[
        jnp.arange(requests)[:, None, None], jnp.arange(queries)[None, :, None], safe
    ].max(valid.astype(jnp.int32))
    bits = (presence[:, :, :-1] << jnp.arange(queries)[None, :, None]).sum(1)
    count = (bits != 0).sum(1).astype(jnp.int32)
    ids = jax.vmap(
        lambda row: jnp.nonzero(row, size=capacity_blocks, fill_value=-1)[0]
    )(bits)
    membership = jnp.take_along_axis(bits, jnp.maximum(ids, 0), axis=1)
    return UnionMetadata(
        ids, jnp.where(ids >= 0, membership, 0), count, count > capacity_blocks
    )


def union_record_starts(
    metadata: UnionMetadata,
    records: jax.Array,
    *,
    block_size: int,
    pool_records: int,
) -> tuple[jax.Array, jax.Array]:
    """Resolve logical blocks to aligned physical runs; invalidate bad pages."""
    logical = jnp.maximum(metadata.ids, 0) * block_size
    starts = jnp.take_along_axis(records, jnp.minimum(logical, records.shape[1] - 1), 1)
    valid = (metadata.ids >= 0) & (logical < records.shape[1])
    valid &= (starts >= 0) & (starts + block_size <= pool_records)
    # D128 BF16 packs two records into one pool word row. Logical blocks
    # need not start at a multiple of block_size in the physical allocation.
    valid &= starts % 2 == 0
    return jnp.where(valid, starts, -1), jnp.where(valid, metadata.membership, 0)


def _unpack_words(words, scratch):
    rows = words.shape[0]
    scratch[pl.ds(0, rows, stride=2), :] = words
    scratch[pl.ds(1, rows, stride=2), :] = pltpu.roll(words, 64, 1)
    words = scratch[...]
    lo = jax.lax.bitcast_convert_type(words << 16, jnp.float32).astype(jnp.bfloat16)
    hi = jax.lax.bitcast_convert_type(words & jnp.int32(-65536), jnp.float32).astype(
        jnp.bfloat16
    )
    return jnp.where(
        jax.lax.broadcasted_iota(jnp.int32, words.shape, 1) < 64,
        lo,
        pltpu.roll(hi, 64, 1),
    )


def _unpack_kernel(words_ref, out_ref, *scratch, halfplane=False):
    out_ref[0] = (
        pltpu.bitcast(words_ref[0], jnp.bfloat16)
        if halfplane
        else _unpack_words(words_ref[0], scratch[0])
    )


@functools.partial(jax.jit, static_argnames=("interpret", "halfplane"))
def unpack_candidate_words(
    words: jax.Array, *, interpret: bool = False, halfplane: bool = False
) -> jax.Array:
    """Unpack native word pairs once into split-order BF16 [request, key, 128]."""
    groups, rows, dim = words.shape
    if dim != 128 or words.dtype != jnp.int32 or rows < 1:
        raise ValueError("Expected nonempty int32 [requests, word rows, 128]")
    tile = math.gcd(rows, 1024)
    return pl.pallas_call(
        functools.partial(_unpack_kernel, halfplane=halfplane),
        out_shape=jax.ShapeDtypeStruct((groups, rows * 2, dim), jnp.bfloat16),
        grid=(groups, rows // tile),
        in_specs=[pl.BlockSpec((1, tile, dim), lambda g, t: (g, t, 0))],
        out_specs=pl.BlockSpec((1, tile * 2, dim), lambda g, t: (g, t, 0)),
        scratch_shapes=[] if halfplane else [pltpu.VMEM((tile * 2, dim), jnp.int32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * 2),
        interpret=interpret,
        name="candidate_unpack_cache",
    )(words)


def _expand_metadata(values, repeats, group_width):
    if repeats == 1:
        return values
    # Lane-local gather avoids repeat's unsupported narrow-minor reshape.
    width = math.gcd(values.shape[0], group_width)
    source = values.reshape(-1, width)
    indices = jnp.broadcast_to(
        (jnp.arange(width * repeats) // repeats)[None, :, None],
        (source.shape[0], width * repeats, 1),
    )
    return jax.lax.gather(
        source,
        indices,
        jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        ),
        slice_sizes=(1, 1),
        mode="promise_in_bounds",
    ).reshape(-1)


def _sum_head_stripes(logits, weights, queries, stripe):
    query_group = math.gcd(queries, stripe)
    head_group = stripe // query_group
    grouped_rows = queries * head_group
    width = logits.shape[-1]
    accumulated = jnp.zeros((grouped_rows, width), jnp.float32)
    # Consume aligned head/query stripes without padding Q=6 to eight queries.
    # Reassociate the FP32 reduction; identical FP32 bits are not required.
    for chunk in range(weights.shape[0]):
        activated = jnp.maximum(
            logits[chunk * grouped_rows : (chunk + 1) * grouped_rows, :], 0
        )
        accumulated = accumulated + activated * weights[chunk, :, None]
    accumulated = accumulated.reshape(queries // query_group, stripe, width)
    # Each stripe interleaves head_group heads with query_group queries.
    for shift in reversed(range(head_group.bit_length() - 1)):
        accumulated = accumulated + pltpu.roll(
            accumulated, query_group * (1 << shift), axis=1
        )
    return jnp.concatenate(
        [
            accumulated[group, :query_group, :]
            for group in range(queries // query_group)
        ],
        axis=0,
    )


def _score_kernel(
    start,
    q,
    weights,
    keys,
    ids,
    membership,
    lengths,
    offsets,
    out,
    *scratch,
    compute_tile: int,
    head_stripe: int | None,
    halfplane: bool,
):
    queries, dim = out.shape[1], q.shape[-1]
    heads = math.prod(q.shape[1:-1]) // queries
    width = out.shape[-1]
    unpacked = None
    if keys.dtype == jnp.int32:
        unpacked = (
            pltpu.bitcast(keys[0], jnp.bfloat16)
            if halfplane
            else _unpack_words(keys[0], scratch[0])
        )
    row = (pl.program_id(1) + start[0]) % ids.shape[0]
    repeat = width // ids.shape[-1]
    # Unroll compute fragments without shrinking the double-buffered DMA window.
    # A rolled loop adds repeated MXU startup/drain to the measured decode path.
    for key_start in range(0, width, compute_tile):
        section = pl.ds(key_start, compute_tile)
        kk = (
            unpacked[key_start : key_start + compute_tile, :]
            if unpacked is not None
            else keys[0, section, :]
        )
        logits = jax.lax.dot_general(
            q[0].reshape(queries * heads, dim),
            kk,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if head_stripe is None:
            logits = logits.reshape(queries, heads, compute_tile)
            values = (jnp.maximum(logits, 0) * weights[0][:, :, None]).sum(1)
        else:
            values = _sum_head_stripes(logits, weights[0], queries, head_stripe)
        metadata_section = pl.ds(key_start // repeat, compute_tile // repeat)
        base = _expand_metadata(ids[row, metadata_section], repeat, dim)
        logical = base + offsets[section]
        members = _expand_metadata(membership[row, metadata_section], repeat, dim)
        allow = ((members[None, :] >> jnp.arange(queries)[:, None]) & 1) != 0
        allow &= (logical[None, :] >= 0) & (logical[None, :] < lengths[row][:, None])
        out[0, :, section] = jnp.where(allow, values, -jnp.inf)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "key_tile",
        "interpret",
        "request_tile",
        "compute_tile",
        "head_stripe",
        "halfplane",
    ),
)
def _candidate_scores(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    start_request: jax.Array,
    request_tile: int,
    block_size: int,
    key_tile: int = 4096,
    interpret: bool = False,
    compute_tile: int | None = None,
    head_stripe: int | None = None,
    halfplane: bool = False,
) -> jax.Array:
    """Address request groups in the original cache, without HBM slice copies."""
    groups, queries, heads, dim = q.shape
    width = ids.shape[1] * block_size
    packed = keys.dtype == jnp.int32
    key_rows = width // 2 if packed else width
    compute_tile = key_tile if compute_tile is None else compute_tile
    if (
        dim != 128
        or q.dtype != jnp.bfloat16
        or keys.dtype not in (jnp.bfloat16, jnp.int32)
        or weights.dtype != jnp.float32
        or weights.shape != (groups, queries, heads)
        or keys.shape != (groups, key_rows, dim)
        or membership.shape != ids.shape
        or lengths.shape != (groups, queries)
        or key_tile < 1
        or width % key_tile
        or (packed and key_tile % 2)
        or compute_tile < 1
        or key_tile % compute_tile
    ):
        raise ValueError("Expected aligned BF16 or packed int32 candidate inputs")
    if head_stripe is not None:
        if head_stripe < 1 or head_stripe & (head_stripe - 1):
            raise ValueError("head_stripe must be a positive power of two")
        query_group = math.gcd(queries, head_stripe)
        head_group = head_stripe // query_group
        if heads % head_group:
            raise ValueError("head_stripe must not require padding the head axis")
        shape = (
            groups,
            queries // query_group,
            query_group,
            heads // head_group,
            head_group,
        )
        q = (
            q.reshape(*shape, dim)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(groups, queries * heads, dim)
        )
        weights = (
            weights.reshape(shape)
            .transpose(0, 3, 1, 4, 2)
            .reshape(groups, heads // head_group, queries * head_group)
        )
    # Keep compact metadata windows lane-aligned, including sub-block key tiles.
    metadata_stride = math.gcd(block_size, key_tile // min(key_tile, dim))
    metadata_tile = key_tile // metadata_stride
    if compute_tile % metadata_stride:
        raise ValueError("compute_tile must align with compact metadata")
    # A negative block must stay negative after adding any within-block offset.
    bases = jnp.where(ids >= 0, ids * block_size, -block_size)
    bases = jnp.repeat(bases, block_size // metadata_stride, axis=1)
    members = jnp.repeat(membership, block_size // metadata_stride, axis=1)
    offsets = jnp.arange(width, dtype=jnp.int32) % block_size
    metadata_rows = (
        groups if interpret else min(groups, pltpu.get_tpu_info().num_sublanes)
    )
    metadata_spec = pl.BlockSpec(
        (metadata_rows, metadata_tile),
        lambda t, g, start: ((g + start[0]) // metadata_rows, t),
    )
    return pl.pallas_call(
        functools.partial(
            _score_kernel,
            compute_tile=compute_tile,
            head_stripe=head_stripe,
            halfplane=halfplane,
        ),
        out_shape=jax.ShapeDtypeStruct((request_tile, queries, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(width // key_tile, request_tile),
            in_specs=[
                pl.BlockSpec(
                    (1, *q.shape[1:]),
                    lambda t, g, start: (g + start[0],) + (0,) * (q.ndim - 1),
                ),
                pl.BlockSpec(
                    (1, *weights.shape[1:]), lambda t, g, start: (g + start[0], 0, 0)
                ),
                pl.BlockSpec(
                    (1, key_tile // 2 if packed else key_tile, dim),
                    lambda t, g, start: (g + start[0], t, 0),
                ),
                metadata_spec,
                metadata_spec,
                pl.BlockSpec(
                    (metadata_rows, queries),
                    lambda t, g, start: ((g + start[0]) // metadata_rows, 0),
                ),
                pl.BlockSpec((key_tile,), lambda t, g, start: (t,)),
            ],
            out_specs=pl.BlockSpec(
                (1, queries, key_tile), lambda t, g, start: (g, 0, t)
            ),
            scratch_shapes=[pltpu.VMEM((key_tile, dim), jnp.int32)]
            if packed and not halfplane
            else [],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * 2),
        interpret=interpret,
        name="candidate_union_packed_scores" if packed else "candidate_union_scores",
    )(
        start_request.reshape(1),
        q,
        weights,
        keys,
        bases,
        members,
        lengths,
        offsets,
    )


@functools.partial(
    jax.jit, static_argnames=("block_size", "key_tile", "interpret", "halfplane")
)
def candidate_scores(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    key_tile: int = 4096,
    interpret: bool = False,
    halfplane: bool = False,
) -> jax.Array:
    """Per-head dot/ReLU/weight/sum with per-query membership and causal masks."""
    return _candidate_scores(
        q,
        weights,
        keys,
        ids,
        membership,
        lengths,
        start_request=jnp.int32(0),
        request_tile=q.shape[0],
        block_size=block_size,
        key_tile=key_tile,
        interpret=interpret,
        halfplane=halfplane,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "key_tile",
        "request_tile",
        "k",
        "overlap",
        "halfplane",
    ),
)
def candidate_indexer(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    key_tile: int | None = None,
    request_tile: int | None = None,
    k: int = 512,
    overlap: bool = True,
    halfplane: bool = False,
) -> jax.Array:
    """Pipeline whole-request score groups and SC selection over a reusable union."""
    requests, queries = q.shape[:2]
    width = ids.shape[1] * block_size
    config = tuned_block_sizes.get_candidate_block_sizes(
        tuned_block_sizes.CandidateTuningKey(
            jax.devices()[0].device_kind,
            requests,
            queries,
            q.shape[2],
            width,
            key_tile,
            k,
            keys.dtype == jnp.int32,
        )
    )
    if request_tile is None:
        request_tile = config.request_tile
    elif request_tile < 1 or requests % request_tile:
        raise ValueError("request_tile must divide the request-major batch")
    # Unroll only the calibrated metadata width and request grouping.
    fused_pipeline = (
        config.fused_pipeline
        and block_size == 8
        and request_tile == config.request_tile
    )

    def prepare(i):
        return i * request_tile

    def score(start):
        return _candidate_scores(
            q,
            weights,
            keys,
            ids,
            membership,
            lengths,
            start_request=start,
            request_tile=request_tile,
            block_size=block_size,
            key_tile=config.key_tile,
            compute_tile=config.compute_tile,
            head_stripe=config.head_stripe,
            halfplane=halfplane,
        )

    def select(i, values):
        if fused_pipeline:
            return streaming_topk(
                values,
                k=k,
                block_size=config.topk_block_size,
                reconstruct_threshold=config.reconstruct_threshold,
                exclude_neg_inf=True,
                logical_blocks=ids,
                start_request=prepare(i),
            )
        group_ids = jax.lax.dynamic_slice_in_dim(ids, prepare(i), request_tile)
        selected = decode._select(
            values,
            jnp.full((request_tile * queries,), width, jnp.int32),
            k,
            config.topk_block_size,
            config.reconstruct_threshold,
        ).reshape(request_tile, queries, k)
        blocks = jnp.take_along_axis(
            group_ids[:, None, :], jnp.maximum(selected, 0) // block_size, axis=2
        )
        logical = blocks * block_size + selected % block_size
        return jnp.where(selected >= 0, logical, -1).reshape(-1, k)

    def write(i, selected, out):
        return jax.lax.dynamic_update_slice_in_dim(
            out, selected, i * request_tile * queries, axis=0
        )

    count = requests // request_tile
    if overlap and fused_pipeline:
        current = score(jnp.int32(0))
        selected_groups = []
        for i in range(count - 1):
            # Immutable outputs need only TC producer ordering, not SC completion.
            next_start, current = jax.lax.optimization_barrier(
                (jnp.int32(prepare(i + 1)), current)
            )
            following = score(next_start)
            selected_groups.append(select(i, current))
            current = following
        selected_groups.append(select(count - 1, current))
        return selected_groups[0] if count == 1 else jnp.concatenate(selected_groups)

    out = jnp.full((requests * queries, k), -1, jnp.int32)
    if overlap:
        return indexer.pipeline_score_topk(count, prepare, score, select, write, out)

    def serial(i, out):
        args, out = jax.lax.optimization_barrier((prepare(i), out))
        return write(i, select(i, score(args)), out)

    return jax.lax.fori_loop(0, count, serial, out)


def cache_capacity(
    query_shape: tuple[int, ...],
    record_shape: tuple[int, ...],
    candidate_count: int,
    block_size: int,
    page_records: int,
) -> int | None:
    """Static envelope without specializing Torch's dynamic token dimension.

    Request/query alignment is checked after JAX lowering, where bucket shapes
    are concrete. Python guards on ``rows == requests * 6`` break Dynamo's
    explicitly dynamic input contract.
    """
    _, width = record_shape
    _, heads, dim = query_shape
    capacity = tuned_block_sizes.CANDIDATE_KEY_CAPACITY
    if (
        (heads, dim) != (32, 128)
        or (candidate_count, block_size) != (2048, 8)
        or not capacity < width <= 1048576
        or page_records < 16
        or page_records % block_size
        or width % page_records
    ):
        return None
    return capacity
