# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundary for native and pooled QSA state owners."""

import functools
import hashlib
import struct
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue
from vllm_torchtpu.kernels.qwen4_exp.ple import (
    decode_embeddings,
    ple_values,
    resident_lookup,
    resident_lookup_raw,
    short_conv,
)
from vllm_torchtpu.kernels.qwen4_exp.pool import pooled_short_conv
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import pooled_qsa_step
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import qsa_step


@functools.cache
def build_qsa_op() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    op = pallas.jax_op(
        "qwen4_exp::qsa",
        qsa_step,
        donate_argnums=(10, 11, 12, 13),
    )

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        packed_kv,
        compressed,
        ring,
        tags,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
    ):
        return (
            *(torch.empty_like(x) for x in (packed_kv, compressed, ring, tags)),
            qgate.new_empty((qgate.shape[0], qgate.shape[1] // 2)),
            positions.new_empty((qgate.shape[0], token_topk + compress_ratio - 1)),
        )

    def run(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        packed_kv,
        compressed,
        ring,
        tags,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
    ):
        planes = torch.stack((tags.to(torch.int32), (tags >> 32).to(torch.int32)), -1)
        packed, c, r, p, output, selected = op(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions.to(torch.int32),
            trig,
            packed_kv,
            compressed,
            ring,
            planes,
            starts.to(torch.int32),
            context.to(torch.int32),
            slots.to(torch.int32),
            main_table.to(torch.int32),
            compressed_table.to(torch.int32),
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            k_scale,
            v_scale,
        )
        for owner, updated in zip((packed_kv, compressed, ring), (packed, c, r)):
            owner.copy_(updated)
        tags.copy_(
            (p[..., 1].to(torch.int64) << 32) | (p[..., 0].to(torch.int64) & 0xFFFFFFFF)
        )
        return output, selected

    return run


@functools.cache
def build_pooled_qsa_op():
    op = pallas.jax_op("qwen4_exp::pooled_qsa", pooled_qsa_step, donate_argnums=(10,))

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
    ):
        return (
            torch.empty_like(pool),
            qgate.new_empty((qgate.shape[0], qgate.shape[-1] // 2)),
            positions.new_empty(
                (qgate.shape[0], token_topk + compress_ratio - 1), dtype=torch.int32
            ),
        )

    def run(*args, **kwargs):
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15):
            args[index] = args[index].to(torch.int32)
        updated, output, blocks = op(*args, **kwargs)
        args[10].copy_(updated)
        return output, blocks

    return run


@functools.cache
def build_raw_lookup_op(vocab_start: int, local_rows: int, head_dim: int):
    def fn(table: jax.Array, ids: jax.Array) -> jax.Array:
        return resident_lookup_raw(
            table,
            ids,
            vocab_start=vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
        )

    op = pallas.jax_op(
        f"qwen4_exp::ple_raw_lookup_{vocab_start}_{local_rows}_{head_dim}", fn
    )

    @op.register_fake
    def fake(table, ids):
        dtype = torch.int16 if table.dtype == torch.int16 else torch.int8
        return table.new_empty((ids.shape[0], ids.shape[1] * head_dim), dtype=dtype)

    return op


@functools.cache
def build_decode_op():
    def fn(raw_embeddings: jax.Array, scale: jax.Array) -> jax.Array:
        return decode_embeddings(raw_embeddings, scale)

    op = pallas.jax_op("qwen4_exp::ple_decode", fn)

    @op.register_fake
    def fake(raw_embeddings, scale):
        return torch.empty_like(raw_embeddings, dtype=torch.bfloat16)

    return op


@functools.cache
def build_lookup_op(vocab_start: int, local_rows: int, head_dim: int):
    def fn(table: jax.Array, ids: jax.Array, scale: jax.Array) -> jax.Array:
        return resident_lookup(
            table,
            ids,
            scale,
            vocab_start=vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
        )

    op = pallas.jax_op(
        f"qwen4_exp::ple_lookup_{vocab_start}_{local_rows}_{head_dim}", fn
    )

    @op.register_fake
    def fake(table, ids, scale):
        return table.new_empty(
            (ids.shape[0], ids.shape[1] * head_dim), dtype=torch.bfloat16
        )

    def run(table, ids, scale):
        # Hash stays signed int64 through remainder. The resident embedding
        # constructor bounds the completed global row IDs below 2**31. Cast
        # before the JAX bridge, whose x64-disabled tracing otherwise refines
        # an int64 Torch argument to an incompatible int32 function input.
        return op(table, ids.to(torch.int32), scale)

    return run


@functools.cache
def build_short_conv_op(dilation: int):
    def fn(
        x: jax.Array,
        weight: jax.Array,
        history: jax.Array,
        starts: jax.Array,
        slots: jax.Array,
        initial: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        output, updated = short_conv(
            x, weight, history, starts, slots, initial, dilation=dilation
        )
        return updated, output

    op = pallas.jax_op(f"qwen4_exp::ple_short_conv_{dilation}", fn, donate_argnums=(2,))

    @op.register_fake
    def fake(x, weight, history, starts, slots, initial):
        return torch.empty_like(history), torch.empty_like(x)

    def run(x, weight, history, starts, slots, initial):
        updated, output = op(x, weight, history, starts, slots, initial)
        history.copy_(updated)
        return output

    return run


@functools.cache
def build_values_op(norm_eps: float):
    def fn(
        embedding: jax.Array,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        nk: jax.Array,
        nq: jax.Array,
        nc: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return ple_values(embedding, query, key, value, nk, nq, nc, norm_eps=norm_eps)

    eps_key = struct.pack("!d", norm_eps).hex()
    op = pallas.jax_op(f"qwen4_exp::ple_values_bf16_{eps_key}", fn)

    @op.register_fake
    def fake(embedding, query, *args):
        return torch.empty_like(query), torch.empty_like(query)

    return op


@functools.cache
def build_pooled_short_conv_op(dilation, state_len):
    fn = functools.partial(pooled_short_conv, dilation=dilation, state_len=state_len)
    op = pallas.jax_op(
        f"qwen4_exp::pooled_short_conv_{dilation}_{state_len}", fn, donate_argnums=(2,)
    )

    @op.register_fake
    def fake(x, weight, pool, starts, ids, initial):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(x, weight, pool, starts, ids, initial):
        updated, output = op(x, weight, pool, starts, ids, initial)
        pool.copy_(updated)
        return output

    return run


@functools.cache
def build_gated_residual_op(branches: int, hidden: int, token_tile: int):
    op = pallas.jax_op(
        f"qwen4_exp::gated_residual_b{branches}_h{hidden}_m{token_tile}_v1",
        gated_residual_fn(branches, hidden, token_tile),
    )

    @op.register_fake
    def fake(residual, block, gates):
        return residual.new_empty(residual.shape)

    return op


@functools.cache
def build_full_mix_op(branches: int, hidden: int, eps: float, token_tile: int):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _full_mix(
            x,
            weight,
            down,
            up,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty(x.shape, dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_norm_op(branches: int, hidden: int, eps: float, token_tile: int):
    """Register a local Pallas op for complete prefill token tiles."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(input_tensor: jax.Array, weight: jax.Array) -> jax.Array:
        return _normalize(
            input_tensor,
            weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_norm_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}_v2",
        core,
    )

    @op.register_fake
    def fake(input_tensor, weight):
        return input_tensor.new_empty(input_tensor.shape)

    return op


@functools.cache
def build_up_epilogue_op(branches: int, hidden: int, token_tile: int):
    """Register a local JAX op; callers gate it to whole prefill tiles."""

    def core(mid: jax.Array, norm: jax.Array, weight: jax.Array) -> jax.Array:
        return _up_epilogue(
            mid, norm, weight, branches=branches, hidden=hidden, token_tile=token_tile
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_up_epilogue_c{branches}_h{hidden}_m{token_tile}_v1", core
    )

    @op.register_fake
    def fake(mid, norm, weight):
        return mid.new_empty((mid.shape[0], hidden), dtype=torch.bfloat16)

    return op


def compute_ngram_ids(
    input_ids,
    query_start_loc,
    ngram_context,
    multipliers,
    vocab_sizes,
    offsets,
    *,
    eos_token_id: int,
):
    """Signed int64 multiply/xor/remainder, including wraparound on overflow.

    Context is oldest->newest. A preceding EOS terminates the predecessor
    chain; a current EOS is hashed normally. Starts delimit real tokens;
    trailing token padding returns zero IDs and never indexes a request OOB.
    """
    tokens = input_ids.reshape(-1).to(torch.int64)
    requests = ngram_context.shape[0]
    if tokens.shape[0] == 0 or requests == 0:
        return tokens.new_zeros((tokens.shape[0], vocab_sizes.shape[0]))
    t = torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.int64)
    req = torch.sum(t[:, None] >= query_start_loc[None, 1:], dim=1)
    req = req.clamp(max=requests - 1)
    rel = t - query_start_loc[req]
    context_len = multipliers.shape[0] - 1
    heads_per_ngram = vocab_sizes.shape[0] // context_len
    mixed = tokens * multipliers[0]
    stopped = torch.zeros_like(tokens, dtype=torch.bool)
    blocks = []
    for distance in range(1, context_len + 1):
        context_index = (rel + context_len - distance).clamp(0, context_len - 1)
        previous = torch.where(
            rel >= distance,
            tokens[(t - distance).clamp(min=0)],
            ngram_context[req, context_index],
        )
        stopped = stopped | (previous == eos_token_id)
        previous = torch.where(stopped, eos_token_id, previous)
        mixed = mixed ^ (previous * multipliers[distance])
        blocks.append(mixed[:, None].expand(-1, heads_per_ngram))
    words = torch.cat(blocks, dim=1)
    ids = torch.remainder(words, vocab_sizes[None]) + offsets[None]
    return torch.where((t < query_start_loc[-1])[:, None], ids, 0)
