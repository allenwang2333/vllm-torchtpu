# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch custom-op bridge for the fused K3 AttentionResidual mix kernel."""

from __future__ import annotations

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.kimi_k3.attn_res_mix import attn_res_mix


@functools.lru_cache(maxsize=32)
def build_k3_attn_res_mix_op(n: int, eps: float, exact_emulation: bool):
    """Build (or fetch) the mix op for stack width ``n``.

    ``eps`` is constant across K3 layers so it is not part of the op name;
    the cache key still separates it, and a second distinct eps would fail
    loudly on duplicate op registration rather than silently aliasing.
    """

    def mix_core(
        prefix_sum: jax.Array,
        block_residuals: jax.Array,
        w_score: jax.Array,
        w_norm: jax.Array,
        w_proj: jax.Array,
    ) -> jax.Array:
        return attn_res_mix(prefix_sum,
                            block_residuals,
                            w_score,
                            w_norm,
                            w_proj,
                            n=n,
                            eps=eps,
                            exact_emulation=exact_emulation)

    suffix = "_exact" if exact_emulation else ""
    mix_op = pallas.jax_op(f"pallas::k3_attn_res_mix_n{n}{suffix}", mix_core)

    def _fake_mix(prefix_sum, *args, **kwargs):
        return torch.empty_like(prefix_sum)

    mix_op.register_fake(_fake_mix)
    return mix_op
