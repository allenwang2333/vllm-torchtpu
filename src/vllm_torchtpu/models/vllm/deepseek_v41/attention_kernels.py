"""Paged attention, exact index selection and index-cache publication."""

import functools

import jax
import jax.numpy as jnp
import torch

from .cache import gather_cache_values, scatter_cache_values

_attention_op = None
_fp8_attention_op = None


def _attention_state(q):
    tokens, heads, _ = q.shape
    return (
        jnp.full((tokens, heads), -1e30, jnp.float32),
        jnp.zeros((tokens, heads), jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )


def _attend(q, pool, records, state, scale, tile=64):
    """Online-softmax accumulate one record set into (max, denominator, numerator)."""
    tokens, _, dim = q.shape
    width = records.shape[1]
    if not width:
        return state
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)

    # Gather BF16 rows once outside; keep FP32 scratch tiled.
    gathered = gather_cache_values(pool, records.reshape(-1), dim).reshape(
        tokens, records.shape[1], dim
    )

    def step(index, carry):
        maximum, denominator, numerator = carry
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        valid = addresses >= 0
        kv = jax.lax.dynamic_slice_in_dim(gathered, index * tile, tile, axis=1)
        kv = jnp.where(valid[..., None], kv, 0)
        scores = (
            jnp.einsum(
                "thd,tkd->thk",
                q.astype(jnp.float32),
                kv.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        scores = jnp.where(valid[:, None], scores, -jnp.inf)
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        exponentials = jnp.exp(scores - new_max[..., None])
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction[..., None] + jnp.einsum(
            "thk,tkd->thd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        return new_max, denominator, numerator

    return jax.lax.fori_loop(0, records.shape[1] // tile, step, state)


def paged_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array,
    main_pool: jax.Array,
    swa_records: jax.Array,
    main_records: jax.Array,
    sink: jax.Array,
    scale: float,
) -> jax.Array:
    # SWA widths in the released model are multiples of 64, preserving the
    # reference's accumulation boundaries when compressed rows follow.
    maximum, denominator, numerator = _attend(
        q,
        main_pool,
        main_records,
        _attend(q, swa_pool, swa_records, _attention_state(q), scale),
        scale,
    )
    denominator += jnp.exp(sink.astype(jnp.float32)[None] - maximum)
    return (numerator / denominator[..., None]).astype(q.dtype)


def compact_records(records: jax.Array) -> jax.Array:
    """Pad to 128 columns and move -1 to the tail: sparse_mla's valid prefix."""
    width = records.shape[1]
    padded = max(128, -(-width // 128) * 128)
    records = jnp.pad(records, ((0, 0), (0, padded - width)), constant_values=-1)
    limit = jnp.iinfo(jnp.int32).max
    records = jnp.sort(jnp.where(records >= 0, records, limit), axis=-1)
    return jnp.where(records == limit, -1, records)


def sparse_fp8_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array,
    main_pool: jax.Array,
    rope_pool: jax.Array,
    swa_records: jax.Array,
    main_records: jax.Array,
    sink: jax.Array,
    scale: float,
) -> jax.Array:
    """BF16 SWA partial merged into V4's sparse MLA over FP8 NoPE + RoPE rows.

    Records are physical pool rows, so the whole pool is passed as one page
    (identity page table); the RoPE companion holds row r's 32 words at r.
    """
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
        sparse_ragged_paged_attention,
    )

    maximum, denominator, numerator = _attend(
        q, swa_pool, swa_records, _attention_state(q), scale
    )
    rows = main_pool.shape[0]
    return sparse_ragged_paged_attention(
        q,
        main_pool.reshape(1, rows, main_pool.shape[1]),
        rope_pool.reshape(1, *rope_pool.shape),
        compact_records(main_records),
        jnp.zeros((1,), jnp.int32),
        jnp.array([0, q.shape[0]], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sink.astype(jnp.float32),
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
    )


def initialize_paged_attention():
    global _attention_op, _fp8_attention_op
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _attention_op = pallas.jax_op(
        "pallas::deepseek_v41_paged_bf16_attention",
        paged_attention_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), P(), P(), P(), P(), None),
    )
    _attention_op.register_fake(lambda q, *args: q.new_empty(q.shape))
    mesh = get_vllm_model_wrapper_context().mesh

    def fp8_attention(
        q: jax.Array,
        swa_pool: jax.Array,
        main_pool: jax.Array,
        rope_pool: jax.Array,
        swa_records: jax.Array,
        main_records: jax.Array,
        sink: jax.Array,
        scale: float,
    ) -> jax.Array:
        # Mosaic calls require explicit manual partitioning on a multi-device
        # mesh; each device attends over its own replicated cache.
        return jax.shard_map(
            functools.partial(sparse_fp8_attention_jax, scale=scale),
            mesh=mesh,
            in_specs=(P(),) * 7,
            out_specs=P(),
            check_vma=False,
        )(q, swa_pool, main_pool, rope_pool, swa_records, main_records, sink)

    _fp8_attention_op = pallas.jax_op(
        "pallas::deepseek_v41_sparse_fp8_attention",
        fp8_attention,
        mesh=mesh,
        input_partition_specs=(P(), P(), P(), P(), P(), P(), P(), None),
    )
    _fp8_attention_op.register_fake(lambda q, *args: q.new_empty(q.shape))


def paged_attention(q, swa, main, swa_records, main_records, sink, scale):
    assert _attention_op is not None
    return _attention_op(
        q, swa.raw, main.raw, swa_records.int(), main_records.int(), sink, scale
    )


def sparse_fp8_attention(q, swa, main, swa_records, main_records, sink, scale):
    assert _fp8_attention_op is not None and main.rope is not None
    return _fp8_attention_op(
        q,
        swa.raw,
        main.raw,
        main.rope,
        swa_records.int(),
        main_records.int(),
        sink,
        scale,
    )


def paged_index_scores_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array | None = None,
) -> jax.Array:
    """Index scores ``[tokens, width]``.

    ``records`` is ``[tokens, width]``, or ``[requests, width]`` when
    ``requests`` maps each token to its row: every token of a request then
    shares one paged gather of the key tile instead of reading it again.
    """
    tokens, dim = q.shape[0], q.shape[-1]
    width = records.shape[1]
    # Keep about 16K keys per iteration: small (decode) batches otherwise run
    # many latency-bound tiny gathers. Prefill batches stay at 128.
    budget = max(128, min(2048, 16384 // max(tokens, 1), -(-width // 128) * 128))
    tile = 1 << (budget.bit_length() - 1)
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)
    scores = jnp.full((tokens, records.shape[1]), -jnp.inf, jnp.float32)

    def step(index, result):
        # Read only the tile consumed by this iteration. Gathering before
        # the loop materializes [tokens, max_context / ratio, dim] even when
        # lengths restrict execution to a single tile (or no tiles).
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        keys = gather_cache_values(pool, addresses.reshape(-1), dim).reshape(
            records.shape[0], tile, dim
        )
        if requests is not None:
            keys = jnp.take(keys, requests, axis=0)
        logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
        # Upstream quantized MQA logits and the weighted head reduction are
        # FP32. BF16 cache storage is not a request to round the dot product.
        logits = jnp.maximum(logits, 0)
        values = (logits * weights.astype(jnp.float32)[:, :, None]).sum(1)
        return jax.lax.dynamic_update_slice_in_dim(
            result, values.astype(jnp.float32), index * tile, axis=1
        )

    return jax.lax.fori_loop(
        0,
        jnp.minimum(
            (jnp.maximum(lengths.max(), 0) + tile - 1) // tile, records.shape[1] // tile
        ),
        step,
        scores,
    )[:, :width]


_indexer_op = None


def select_indices_jax(
    scores: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
) -> tuple[jax.Array, jax.Array]:
    """Source publishes candidates without filtering its own selection."""
    count, width = scores.shape
    if topk < 1 or width < 1 or block_size < 1:
        raise ValueError("Positive top-k, score width and block size required")
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    if candidates.ndim != 2 or candidates.shape[0] != count:
        raise ValueError("Candidate rows must match query rows")
    lengths = jnp.clip(lengths.reshape(count), 0, width)
    logical = jnp.arange(width)[None, :]
    visible = logical < lengths[:, None]
    scores = jnp.where(visible, scores, -jnp.inf)
    published = candidates
    # When the candidate buffer covers every block, publication/filtering are
    # identities. Keep the caller's buffer untouched in this static case.
    if (candidate_write or candidate_filter) and candidates.shape[
        1
    ] * block_size < width:
        blocks = (width + block_size - 1) // block_size
        if candidate_write:
            block_scores = (
                jnp.pad(
                    scores,
                    ((0, 0), (0, blocks * block_size - width)),
                    constant_values=-jnp.inf,
                )
                .reshape(count, blocks, block_size)
                .max(-1)
            )
            last = (lengths - 1) // block_size
            block_scores = jnp.where(
                jnp.arange(blocks)[None] == last[:, None], jnp.inf, block_scores
            )
            values, ids = jax.lax.top_k(block_scores, candidates.shape[1])
            published = jnp.where(values > -jnp.inf, ids, -1).astype(jnp.int32)
        else:
            # One extra column absorbs invalid candidate IDs; no 3-D membership
            # tensor and no dependence on the order of candidate IDs.
            valid = (candidates >= 0) & (candidates < blocks)
            ids = jnp.where(valid, candidates, blocks)
            membership = jnp.zeros((count, blocks + 1), jnp.bool_)
            membership = membership.at[jnp.arange(count)[:, None], ids].set(True)
            allowed = jnp.take_along_axis(membership, logical // block_size, axis=1)
            # Match the upstream batch-wide short-context bypass.
            scores = jnp.where(allowed | (lengths.max() <= topk), scores, -jnp.inf)
    values, indices = jax.lax.top_k(scores, min(topk, width))
    # Preserve V4.1's logical order and therefore attention accumulation order.
    indices = jnp.sort(indices, axis=-1)
    values = jnp.take_along_axis(scores, indices, axis=1)
    return jnp.where(values > -jnp.inf, indices, -1).astype(jnp.int32), published


def paged_indexer_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
) -> tuple[jax.Array, jax.Array]:
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows."""

    def full_context(_):
        scores = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
        scores = jnp.where(jnp.take(records >= 0, requests, axis=0), scores, -jnp.inf)
        return select_indices_jax(
            scores,
            lengths,
            candidates,
            topk,
            block_size,
            candidate_write,
            candidate_filter,
        )

    width = records.shape[1]
    if not candidate_filter or candidates.shape[1] * block_size >= width:
        return full_context(None)

    def candidate_context(_):
        # Candidates differ per token, so this path reads per-token records.
        token_records = jnp.take(records, requests, axis=0)
        # Sort before expanding so equal-score ties retain the full-context
        # top_k's logical ordering. Duplicate and invalid blocks cannot select
        # the same key twice or read outside this request's causal history.
        blocks = (width + block_size - 1) // block_size
        ids = jnp.sort(
            jnp.where((candidates >= 0) & (candidates < blocks), candidates, blocks),
            axis=-1,
        )
        unique = jnp.concatenate(
            (jnp.ones_like(ids[:, :1], dtype=jnp.bool_), ids[:, 1:] != ids[:, :-1]),
            axis=-1,
        )
        logical = (ids[..., None] * block_size + jnp.arange(block_size)).reshape(
            q.shape[0], -1
        )
        valid = logical < jnp.minimum(lengths[:, None], width)
        valid &= jnp.repeat(unique & (ids < blocks), block_size, axis=-1)
        compact = jnp.take_along_axis(
            token_records, jnp.minimum(logical, width - 1), axis=1
        )
        compact = jnp.where(valid, compact, -1)
        # This bounds both key reads and QK work by the candidate pool size.
        compact_lengths = jnp.full_like(lengths, compact.shape[1])
        scores = paged_index_scores_jax(q, pool, compact, weights, compact_lengths)
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        output_width = min(topk, width)
        padding = max(0, output_width - scores.shape[1])
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        logical = jnp.pad(logical, ((0, 0), (0, padding)), constant_values=width)
        values, indices = jax.lax.top_k(scores, output_width)
        selected = jnp.take_along_axis(logical, indices, axis=1)
        selected_valid = values > -jnp.inf
        # The dense reference fills an underfull top-k with the lowest logical
        # -inf positions, then sorts all positions before masking them. Preserve
        # those padding locations too: moving padding across a 64-key attention
        # tile changes BF16 accumulation boundaries.
        occupied = jnp.zeros((q.shape[0], output_width + 1), jnp.bool_)
        occupied_ids = jnp.where(
            selected_valid & (selected < output_width), selected, output_width
        )
        occupied = occupied.at[jnp.arange(q.shape[0])[:, None], occupied_ids].set(True)
        fillers = jnp.sort(
            jnp.where(~occupied[:, :output_width], jnp.arange(output_width), width),
            axis=-1,
        )
        filler_rank = jnp.maximum(
            jnp.arange(output_width) - selected_valid.sum(-1)[:, None], 0
        )
        selected = jnp.where(
            selected_valid, selected, jnp.take_along_axis(fillers, filler_rank, axis=1)
        )
        order = jnp.argsort(selected, axis=-1)
        selected = jnp.take_along_axis(selected, order, axis=1)
        selected_valid = jnp.take_along_axis(selected_valid, order, axis=1)
        return jnp.where(selected_valid, selected, -1).astype(jnp.int32), candidates

    # Preserve the reference's batch-wide short-context bypass. That branch
    # scans at most topk keys; long contexts never enter the full-history path.
    return jax.lax.cond(lengths.max() <= topk, full_context, candidate_context, None)


def write_index_keys(
    pool: jax.Array, write_records: jax.Array, keys: jax.Array
) -> jax.Array:
    """Publish index records through the same cache writer as KV and state."""
    if keys.dtype != jnp.bfloat16:
        raise ValueError("Index keys must be BF16")
    if keys.ndim != 2 or write_records.shape != (keys.shape[0],):
        raise ValueError("One physical record index per key is required")
    return scatter_cache_values(pool, write_records, keys)


def update_and_select_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    write_records: jax.Array,
    keys: jax.Array,
    requests: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    updated = write_index_keys(pool, write_records, keys)
    selected, published = paged_indexer_jax(
        q,
        updated,
        records,
        weights,
        lengths,
        candidates,
        requests,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
    )
    return updated, selected, published


def initialize_indexer():
    global _indexer_op
    if _indexer_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _indexer_op = pallas.jax_op(
        "pallas::deepseek_v41_exact_indexer",
        update_and_select_jax,
        donate_argnums=(1,),
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 9 + (None,) * 4,
    )

    def fake(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        keys,
        requests,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
    ):
        return (
            torch.empty_like(pool),
            records.new_empty(
                (q.shape[0], min(topk, records.shape[1])), dtype=torch.int32
            ),
            torch.empty_like(candidates),
        )

    _indexer_op.register_fake(fake)


def paged_indexer(
    q,
    cache,
    records,
    weights,
    lengths,
    candidates,
    write_records,
    keys,
    requests,
    *,
    topk,
    block_size,
    candidate_write,
    candidate_filter,
):
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows."""
    assert _indexer_op is not None
    return _indexer_op(
        q,
        cache.raw,
        records.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        write_records.int(),
        keys,
        requests.int(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
    )
