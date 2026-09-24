# SPDX-License-Identifier: Apache-2.0
"""Pipelined CP2 dense MLA decode.

KV lives in two VMEM banks. The next bank's paged DMA is issued before
computing QK/softmax/PV on the current bank. Online statistics never leave
VMEM. Remote query and partial-result transfers use separate semaphore banks.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _attention_group(
    first,
    lengths,
    tables,
    cache,
    kv_banks,
    q_buf,
    acc_buf,
    m_buf,
    l_buf,
    out_buf,
    stat_buf,
    kv_sems,
    start_queries,
    wait_queries,
    *,
    tile_pages,
    group_size,
    scale,
):
    page_size = cache.shape[1] * cache.shape[2]
    tile = tile_pages * page_size
    heads = q_buf.shape[1]
    latent = out_buf.shape[-1]
    max_len = jnp.int32(0)
    for b in range(group_size):
        max_len = jnp.maximum(max_len, lengths[first + b])
    iterations = pl.cdiv(max_len, tile)

    def kv_dma(i, wait=False):
        bank = i % 2
        for b in range(group_size):
            length = lengths[first + b]

            valid_pages = jnp.clip(
                pl.cdiv(length, page_size) - i * tile_pages, 0, tile_pages
            )

            def copy_page(p, _, *, b=b):
                logical = i * tile_pages + p
                physical = tables[first + b, logical]
                copy = pltpu.make_async_copy(
                    cache.at[physical], kv_banks.at[bank, b, p], kv_sems.at[bank, b]
                )
                if wait:
                    copy.wait()
                else:
                    copy.start()

            def copy_tail(*, b=b, valid_pages=valid_pages, copy_page=copy_page):
                if tile_pages > 64:
                    lax.fori_loop(0, valid_pages, copy_page, None)
                    return
                # Aggregate exact complete chunks; only the last <8 pages use
                # scalar copies. Never fetch a page beyond the valid prefix.
                chunk_pages = min(8, tile_pages)

                def chunk(k, _):
                    if wait:
                        ref = kv_banks.at[bank, b, pl.ds(k * chunk_pages, chunk_pages)]
                        pltpu.make_async_copy(ref, ref, kv_sems.at[bank, b]).wait()
                    else:
                        for offset in range(chunk_pages):
                            copy_page(k * chunk_pages + offset, None)

                lax.fori_loop(0, valid_pages // chunk_pages, chunk, None)
                lax.fori_loop(
                    valid_pages // chunk_pages * chunk_pages,
                    valid_pages,
                    copy_page,
                    None,
                )

            if wait:
                # Page copies share a byte-counting DMA semaphore. Consume
                # a complete tile once, rather than replaying its page loop.
                def wait_tile(*, b=b):
                    ref = kv_banks.at[bank, b]
                    pltpu.make_async_copy(ref, ref, kv_sems.at[bank, b]).wait()

                lax.cond(valid_pages == tile_pages, wait_tile, copy_tail)
            else:
                # Expose full-tile DMA submissions to the compiler instead of
                # paying dynamic loop control for every page. Keep tails exact.
                lax.cond(
                    valid_pages == tile_pages,
                    lambda *, copy_page=copy_page: lax.fori_loop(
                        0, tile_pages, copy_page, None, unroll=128
                    ),
                    copy_tail,
                )

    start_queries()

    @pl.when(iterations > 0)
    def prime():
        kv_dma(jnp.int32(0))

    wait_queries()
    acc_buf[...] = jnp.zeros(acc_buf.shape, jnp.float32)
    m_buf[...] = jnp.full(m_buf.shape, -jnp.inf, jnp.float32)
    l_buf[...] = jnp.zeros(l_buf.shape, jnp.float32)

    def broadcast_stat(x, width):
        return jnp.concatenate([x] * pl.cdiv(width, 128), axis=-1)[..., :width]

    def step(i, _):
        bank = i % 2

        # Bank i+1 was drained on its previous use. No wait is moved ahead
        # of independent compute except the one protecting current reads.
        @pl.when(i + 1 < iterations)
        def prefetch():
            kv_dma(i + 1)

        kv_dma(i, wait=True)

        def batched():
            rows = kv_banks[bank].reshape(group_size, tile, cache.shape[-1])
            qs = q_buf[...]
            lens = jnp.stack([lengths[first + b] for b in range(group_size)])
            valid = i * tile + jnp.arange(tile)[None, :] < lens[:, None]
            scores = (
                jnp.einsum("bhd,btd->bht", qs, rows, preferred_element_type=jnp.float32)
                * scale
            )
            scores = jnp.where(valid[:, None, :], scores, -jnp.inf)
            old_m, old_l = m_buf[...], l_buf[...]
            new_m = jnp.maximum(old_m, scores.max(axis=-1, keepdims=True))
            safe_m = jnp.where(jnp.isfinite(new_m), new_m, 0)
            correction = jnp.exp(old_m - safe_m)
            probs = jnp.exp(scores - broadcast_stat(safe_m, tile))
            values = jnp.stack(
                [
                    jnp.where(
                        i * tile + lax.broadcasted_iota(jnp.int32, (tile, latent), 0)
                        < lengths[first + b],
                        rows[b, :, :latent],
                        0,
                    )
                    for b in range(group_size)
                ]
            )
            pv = jnp.einsum(
                "bht,btd->bhd",
                probs.astype(rows.dtype),
                values,
                preferred_element_type=jnp.float32,
            )
            acc_buf[...] = acc_buf[...] * broadcast_stat(correction, latent) + pv
            l_buf[...] = jnp.broadcast_to(
                old_l * correction + probs.sum(axis=-1, keepdims=True),
                (group_size, heads, 128),
            )
            m_buf[...] = jnp.broadcast_to(new_m, (group_size, heads, 128))

        def scalar():
            for b in range(group_size):
                length = lengths[first + b]

                @pl.when(i * tile < length)
                def compute(*, b=b, length=length):
                    rows = kv_banks.at[bank, b].reshape(tile, cache.shape[-1])[...]
                    q = q_buf[b]
                    scores = (
                        jnp.dot(q, rows.T, preferred_element_type=jnp.float32) * scale
                    )
                    valid_scores = (
                        i * tile + lax.broadcasted_iota(jnp.int32, (heads, tile), 1)
                    ) < length
                    valid_values = (
                        i * tile + lax.broadcasted_iota(jnp.int32, (tile, latent), 0)
                    ) < length
                    # Unfetched tail pages must not contribute, including NaNs
                    # in unspecified VMEM. Select the masked score explicitly.
                    scores = jnp.where(valid_scores, scores, -jnp.inf)
                    new_m = jnp.maximum(m_buf[b], scores.max(axis=1, keepdims=True))
                    correction = jnp.exp(m_buf[b] - new_m)
                    probs = jnp.exp(scores - broadcast_stat(new_m, tile))
                    values = jnp.where(valid_values, rows[:, :latent], 0)
                    pv = jnp.dot(
                        probs.astype(rows.dtype),
                        values,
                        preferred_element_type=jnp.float32,
                    )
                    acc_buf[b] = acc_buf[b] * broadcast_stat(correction, latent) + pv
                    l_buf[b] = jnp.broadcast_to(
                        l_buf[b] * correction + probs.sum(axis=1, keepdims=True),
                        (heads, 128),
                    )
                    m_buf[b] = jnp.broadcast_to(new_m, (heads, 128))

        if tile_pages in (32, 64) and group_size >= 4:
            # Keep the scalar skip path for a final partial request group.
            lax.cond(lengths[first + group_size - 1] > 0, batched, scalar)
        else:
            scalar()

    lax.fori_loop(0, iterations, step, None)
    for b in range(group_size):
        denominator = l_buf[b]
        out_buf[b] = (
            acc_buf[b] / broadcast_stat(jnp.maximum(denominator, 1e-30), latent)
        ).astype(out_buf.dtype)
        stat_buf[b] = jnp.where(
            denominator > 0, m_buf[b] + jnp.log(jnp.maximum(l_buf[b], 1e-30)), -jnp.inf
        )


def _local_kernel(
    lengths,
    tables,
    cache,
    queries,
    output,
    stats,
    kv_banks,
    q_buf,
    acc_buf,
    m_buf,
    l_buf,
    out_buf,
    stat_buf,
    kv_sems,
    q_sem,
    out_sem,
    stat_sem,
    *,
    tile_pages,
    group_size,
    scale,
):
    first = pl.program_id(0) * group_size
    query_copy = pltpu.make_async_copy(
        queries.at[pl.ds(first, group_size)], q_buf, q_sem
    )
    _attention_group(
        first,
        lengths,
        tables,
        cache,
        kv_banks,
        q_buf,
        acc_buf,
        m_buf,
        l_buf,
        out_buf,
        stat_buf,
        kv_sems,
        query_copy.start,
        query_copy.wait,
        tile_pages=tile_pages,
        group_size=group_size,
        scale=scale,
    )
    out_copy = pltpu.make_async_copy(
        out_buf, output.at[pl.ds(first, group_size)], out_sem
    )
    stat_copy = pltpu.make_async_copy(
        stat_buf, stats.at[pl.ds(first, group_size)], stat_sem
    )
    out_copy.start()
    stat_copy.start()
    out_copy.wait()
    stat_copy.wait()


def local_attention(
    cache,
    queries,
    lengths,
    tables,
    *,
    scale,
    latent_dim=512,
    tile_pages=128,
    group_size=4,
    interpret=False,
):
    """Read already-updated local KV and return output plus natural-log LSE.

    Queries contain latent+RoPE, zero-padded to the cache row width. Lengths
    and tables are token-indexed, with zero lengths for padded tokens. This
    read-only contract makes cache ownership independent of DMA scheduling.
    """
    tokens, heads, width = queries.shape
    if group_size <= 0 or tile_pages <= 0 or tokens % group_size:
        raise ValueError("Positive group size must divide padded token bucket")
    if cache.ndim != 4 or width != cache.shape[-1]:
        raise ValueError("Expected packed cache and matching padded query width")
    if width % 128 or latent_dim % 128:
        raise ValueError("Cache row and latent widths must be 128-aligned")
    if lengths.shape != (tokens,) or tables.shape[0] != tokens:
        raise ValueError("Lengths and page tables must be token-indexed")
    kv_shape = (2, group_size, tile_pages, *cache.shape[1:])
    out_shape = (tokens, heads, latent_dim)
    stat_shape = (tokens, heads, 128)
    fn = pl.pallas_call(
        functools.partial(
            _local_kernel, tile_pages=tile_pages, group_size=group_size, scale=scale
        ),
        out_shape=(
            jax.ShapeDtypeStruct(out_shape, queries.dtype),
            jax.ShapeDtypeStruct(stat_shape, jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(tokens // group_size,),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 2,
            out_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 2,
            scratch_shapes=(
                pltpu.VMEM(kv_shape, cache.dtype),
                pltpu.VMEM((group_size, heads, width), queries.dtype),
                pltpu.VMEM((group_size, heads, latent_dim), jnp.float32),
                pltpu.VMEM((group_size, heads, 128), jnp.float32),
                pltpu.VMEM((group_size, heads, 128), jnp.float32),
                pltpu.VMEM((group_size, heads, latent_dim), queries.dtype),
                pltpu.VMEM((group_size, heads, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((2, group_size)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=60 * 2**20
        ),
        interpret=interpret,
        name="k3_mla_cp_local_pipeline",
    )
    output, stats = fn(lengths, tables, cache, queries)
    return output, stats[..., 0]


def _distributed_kernel(
    lengths,
    tables,
    active,
    cache,
    queries,
    output,
    kv_banks,
    q_banks,
    acc_buf,
    m_buf,
    l_buf,
    out_buf,
    stat_buf,
    own,
    send,
    recv,
    merged,
    kv_sems,
    q_local_sems,
    q_send_sems,
    q_recv_sems,
    send_sems,
    recv_sems,
    out_sem,
    barrier,
    *,
    tile_pages,
    group_size,
    scale,
    axis_name,
):
    """Two-bank request pipeline, with exactly one owner of each semaphore.

    A group barrier follows draining its reused bank on both ranks. This
    protects peer VMEM from a fast sender overwriting a slow consumer. It
    does not wait on the previous group's partial exchange: that exchange
    remains in flight during the next group's attention.
    """
    peer = (lax.axis_index(axis_name) ^ 1,)
    half = queries.shape[1]
    groups = pl.cdiv(active[0], group_size)

    def synchronize():
        pl.semaphore_signal(
            barrier, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(barrier, 1)

    def wait_dma(ref, sem):
        pltpu.make_async_copy(ref, ref, sem).wait()

    def drain(g):
        bank = g % 2
        wait_dma(recv.at[bank], recv_sems.at[bank])
        wait_dma(send.at[bank], send_sems.at[bank])
        a = own[bank]
        b = recv[bank]
        la, lb = a[..., 512:], b[..., 512:]
        shift = jnp.maximum(la, lb)
        shift = jnp.where(jnp.isfinite(shift), shift, 0)
        wa = jnp.where(jnp.isfinite(la), jnp.exp(la - shift), 0)
        wb = jnp.where(jnp.isfinite(lb), jnp.exp(lb - shift), 0)
        denominator = jnp.maximum(wa + wb, 1e-30)
        wa = jnp.concatenate([wa / denominator] * 4, axis=-1)
        wb = jnp.concatenate([wb / denominator] * 4, axis=-1)
        merged[...] = (a[..., :512] * wa + b[..., :512] * wb).astype(output.dtype)
        copy = pltpu.make_async_copy(
            merged, output.at[pl.ds(g * group_size, group_size)], out_sem
        )
        copy.start()
        copy.wait()

    def run_group(g, _):
        bank = g % 2
        first = g * group_size

        @pl.when(g >= 2)
        def free_bank():
            drain(g - 2)

        # A receive-completion wait alone does not prove peer consumption.
        # Both ranks must free this bank before either issues another write.
        @pl.when(g != 1)
        def protect_bank():
            synchronize()

        local_copy = pltpu.make_async_copy(
            queries.at[pl.ds(first, group_size)],
            q_banks.at[bank, :, :half],
            q_local_sems.at[bank],
        )
        remote_copy = pltpu.make_async_remote_copy(
            queries.at[pl.ds(first, group_size)],
            q_banks.at[bank, :, half:],
            q_send_sems.at[bank],
            q_recv_sems.at[bank],
            device_id=peer,
            device_id_type=pl.DeviceIdType.MESH,
        )

        def start_queries():
            local_copy.start()
            remote_copy.start()

        def wait_queries():
            local_copy.wait()
            # Drain the local receive, not the peer's destination descriptor.
            wait_dma(q_banks.at[bank, :, half:], q_recv_sems.at[bank])
            wait_dma(queries.at[pl.ds(first, group_size)], q_send_sems.at[bank])

        _attention_group(
            first,
            lengths,
            tables,
            cache,
            kv_banks,
            q_banks.at[bank],
            acc_buf,
            m_buf,
            l_buf,
            out_buf,
            stat_buf,
            kv_sems,
            start_queries,
            wait_queries,
            tile_pages=tile_pages,
            group_size=group_size,
            scale=scale,
        )
        own[bank] = jnp.concatenate(
            (out_buf[:, :half].astype(jnp.float32), stat_buf[:, :half]), axis=-1
        )
        send[bank] = jnp.concatenate(
            (out_buf[:, half:].astype(jnp.float32), stat_buf[:, half:]), axis=-1
        )
        pltpu.make_async_remote_copy(
            send.at[bank],
            recv.at[bank],
            send_sems.at[bank],
            recv_sems.at[bank],
            device_id=peer,
            device_id_type=pl.DeviceIdType.MESH,
        ).start()

    lax.fori_loop(0, groups, run_group, None)
    lax.fori_loop(jnp.maximum(groups - 2, 0), groups, lambda g, _: drain(g), None)
    # Every invocation ends with all sends, receives and stores consumed.
    synchronize()


def distributed_attention(
    cache,
    queries,
    lengths,
    tables,
    active,
    *,
    mesh,
    scale,
    tile_pages=128,
    group_size=4,
    axis_name="dcp",
):
    """CP2 attention over already-updated KV, including both exchanges.

    Global arrays: cache/lengths/tables sharded on their leading dimension,
    query and output heads sharded on dimension 1. ``active`` is replicated.
    The local rank order is own heads then peer heads; only owned heads are
    returned, so no global-head permutation or separate reduce-scatter runs.
    """
    if len(mesh.axis_names) != 1 or mesh.shape[axis_name] % 2:
        raise ValueError("The pipeline requires a flat mesh of adjacent CP2 pairs")
    P = jax.sharding.PartitionSpec

    def local(c, q, lens, table, count):
        tokens, heads, width = q.shape
        if width != 576 or c.shape[1:] != (8, 2, 640):
            raise ValueError("Expected BF16 K3 page16 / latent512+RoPE64 cache")
        if q.dtype != jnp.bfloat16 or c.dtype != jnp.bfloat16:
            raise ValueError("The pipeline currently supports BF16 Q and KV")
        if group_size <= 0 or tokens % group_size or tile_pages <= 0:
            raise ValueError("Invalid request group or KV tile")
        padded_heads = pl.cdiv(heads, 8) * 8
        q = jnp.pad(q, ((0, 0), (0, padded_heads - heads), (0, 64)))
        total_heads = 2 * padded_heads
        # Scratch is bounded by tile/group configuration, never context length.
        scratch = (
            pltpu.VMEM((2, group_size, tile_pages, 8, 2, 640), jnp.bfloat16),
            pltpu.VMEM((2, group_size, total_heads, 640), jnp.bfloat16),
            pltpu.VMEM((group_size, total_heads, 512), jnp.float32),
            pltpu.VMEM((group_size, total_heads, 128), jnp.float32),
            pltpu.VMEM((group_size, total_heads, 128), jnp.float32),
            # Keep normalized local partials in FP32 until the CP merge;
            # rounding before cancellation across stripes loses precision.
            pltpu.VMEM((group_size, total_heads, 512), jnp.float32),
            pltpu.VMEM((group_size, total_heads, 128), jnp.float32),
            *[
                pltpu.VMEM((2, group_size, padded_heads, 640), jnp.float32)
                for _ in range(3)
            ],
            pltpu.VMEM((group_size, padded_heads, 512), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2, group_size)),
            *[pltpu.SemaphoreType.DMA((2,)) for _ in range(5)],
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
        )
        fn = pl.pallas_call(
            functools.partial(
                _distributed_kernel,
                tile_pages=tile_pages,
                group_size=group_size,
                scale=scale,
                axis_name=axis_name,
            ),
            out_shape=jax.ShapeDtypeStruct((tokens, padded_heads, 512), q.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 2,
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                grid=(),
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 2**20),
            name="k3_mla_cp_query_kv_partial_pipeline",
        )
        output = fn(lens, table, count, c, q)[:, :heads]
        return jnp.where(jnp.arange(tokens)[:, None, None] < count[0], output, 0)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(axis_name), P(None, axis_name), P(axis_name), P(axis_name), P()),
        out_specs=P(None, axis_name),
        check_vma=False,
    )(cache, queries, lengths, tables, active)


def _prepare_decode(c, q, qp, kv, pe, lengths, tables, starts, *, rank):
    """Scatter query suffixes and give each token its causal striped length.

    ``lengths`` includes every scheduled query token, including MTP candidates.
    Rejected candidates may remain in cache; subsequent steps overwrite them
    and the per-token length prevents attention from reading stale suffixes.
    """
    # The serving metadata stores each rank's block table flattened.
    tables = tables.reshape(lengths.shape[0], -1)
    token = jnp.arange(q.shape[0], dtype=jnp.int32)
    request = jnp.searchsorted(starts[1:], token, side="right")
    safe = jnp.minimum(request, lengths.shape[0] - 1)
    valid = (request < lengths.shape[0]) & (token < starts[-1])
    query_len = starts[safe + 1] - starts[safe]
    position = lengths[safe] - query_len + token - starts[safe]
    valid &= (position >= 0) & (position < lengths[safe])
    pos = jnp.maximum(position, 0)
    owner = pos // 16 % 2
    local_pos = pos // 32 * 16 + pos % 16
    col = local_pos // 16
    page = tables[safe, jnp.minimum(col, tables.shape[1] - 1)]
    write = valid & (owner == rank) & (col < tables.shape[1])
    page = jnp.where(write, page, c.shape[0])
    values = jnp.pad(jnp.concatenate((kv, pe), axis=-1), ((0, 0), (0, 64)))
    c = c.at[page, local_pos % 16 // 2, local_pos % 2].set(values, mode="drop")
    causal_length = pos + 1
    local = causal_length // 32 * 16 + jnp.clip(causal_length % 32 - rank * 16, 0, 16)
    return (
        c,
        jnp.concatenate((q, qp), axis=-1),
        jnp.where(valid, local, 0),
        tables[safe],
        starts[-1:],
    )


def decode_attention(
    cache,
    q_nope,
    q_pe,
    new_kv,
    new_pe,
    seq_lens,
    block_tables,
    query_start_loc,
    *,
    mesh,
    scale,
    tile_pages=128,
    group_size=4,
    axis_name="dcp",
    cp_rank=None,
):
    """Native CP2 entry: owner-only update, metadata, and complete pipeline."""
    P = jax.sharding.PartitionSpec

    def prepare(c, q, qp, kv, pe, lengths, tables, starts):
        rank = lax.axis_index(axis_name) % 2 if cp_rank is None else cp_rank
        return _prepare_decode(c, q, qp, kv, pe, lengths, tables, starts, rank=rank)

    c, q, lengths, tables, count = jax.shard_map(
        prepare,
        mesh=mesh,
        in_specs=(
            P(axis_name),
            P(None, axis_name),
            P(None, axis_name),
            P(),
            P(),
            P(),
            P(axis_name),
            P(),
        ),
        out_specs=(P(axis_name), P(None, axis_name), P(axis_name), P(axis_name), P()),
        check_vma=False,
    )(cache, q_nope, q_pe, new_kv, new_pe, seq_lens, block_tables, query_start_loc)
    out = distributed_attention(
        c,
        q,
        lengths,
        tables,
        count,
        mesh=mesh,
        scale=scale,
        tile_pages=tile_pages,
        group_size=group_size,
        axis_name=axis_name,
    )
    return c, out
