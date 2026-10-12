# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib
from collections.abc import Callable
import jax
import torch
from torch_tpu._internal import pallas
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue
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
