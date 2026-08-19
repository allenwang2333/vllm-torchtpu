# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the fused-EP device tests: shapes, inputs, references.

These lived in the v1 device test until that test left with the v1 kernel, and
the v2 device test still needs them. The reference here is the grouped-matmul
path plus ``ref_moe``, neither of which is a fused kernel, so this outlives
whichever kernels the tree carries.

``_small_inputs`` returns both weight layouts from one draw -- ``(E, 2, H, F)``
for ``ref_moe`` and ``(E, H, 2F)`` for v2 -- so the two paths are compared on
the same numbers without either side reshaping.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v1.kernel import ref_moe
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func

EP_AXIS = "model"

FP8 = jnp.float8_e4m3fn

FP8_MAX = float(jnp.finfo(FP8).max)

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 4096, 1024, 512, 10


def _mesh(ep_size):
    devices = np.asarray(jax.devices()[:ep_size]).reshape(1, ep_size)
    return Mesh(devices,
                axis_names=("data", EP_AXIS),
                axis_types=(AxisType.Explicit, AxisType.Explicit))


def _weights(seed, num_experts, hidden, inter):
    """Per-output-channel fp8, in the layout the FP8 loader produces."""
    rng = np.random.default_rng(seed)
    w13 = rng.normal(0, 0.05, (num_experts, hidden, 2 * inter))
    w2 = rng.normal(0, 0.05, (num_experts, inter, hidden))
    s13 = np.abs(w13).max(axis=1, keepdims=True) / FP8_MAX
    s2 = np.abs(w2).max(axis=1, keepdims=True) / FP8_MAX
    return (jnp.asarray(w13 / s13, FP8),
            jnp.asarray(s13[:, None],
                        jnp.float32).reshape(num_experts, 1, 1, 2 * inter),
            jnp.asarray(w2 / s2, FP8),
            jnp.asarray(s2, jnp.float32).reshape(num_experts, 1, 1, hidden))


def _shard(x, mesh, spec=P(EP_AXIS)):
    return jax.device_put(x, NamedSharding(mesh, spec))


@functools.lru_cache(maxsize=None)
def _gmm_reference_fn(mesh, ep_size, local_experts):
    """Build the jitted reference ONCE per (mesh, sharding).

    `@jax.jit` on a closure defined inside the caller mints a fresh function
    object per call, so JAX's compilation cache misses every time and a timing
    loop measures the compiler. That is not hypothetical: it is why this file
    once reported `gmm=11102.307 ms` and "182x / 2911x" speedups, all of which
    were compile time. The v2 device ladder imports this function for its
    `gmm_ms` column, so the defect propagates.
    """

    @jax.jit
    @jax.shard_map(mesh=mesh,
                   in_specs=(P(EP_AXIS), ) * 6,
                   out_specs=P(EP_AXIS),
                   check_vma=False)
    def run(tokens, logits, w13, s13, w2, s2):
        gathered_tokens = jax.lax.all_gather(tokens, EP_AXIS, tiled=True)
        gathered_logits = jax.lax.all_gather(logits, EP_AXIS, tiled=True)
        probs = jax.nn.softmax(gathered_logits.astype(jnp.float32), axis=-1)
        weights, ids = jax.lax.top_k(probs, TOP_K)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        experts_start = jax.lax.axis_index(EP_AXIS).astype(
            jnp.int32) * local_experts
        out = fused_moe_func(
            hidden_states=gathered_tokens,
            w1=w13,
            w2=w2,
            w1_scale=s13,
            w2_scale=s2,
            w1_bias=None,
            w2_bias=None,
            topk_weights=weights.astype(tokens.dtype),
            topk_ids=ids.astype(jnp.int32),
            experts_start=experts_start,
            topk=TOP_K,
            activation="silu",
            use_ep=True,
        )
        return jax.lax.psum_scatter(out, EP_AXIS, tiled=True)

    return run


def _gmm_reference(tokens, logits, w13, s13, w2, s2, mesh, ep_size):
    """What the general path computes: all-gather, mask, GMM, reduce-scatter.

    Modelled with plain collectives so the comparison is against the *semantics*
    the fused kernel replaces, not against another Pallas kernel's tiling.
    """
    run = _gmm_reference_fn(mesh, ep_size, w2.shape[0] // ep_size)
    return run(tokens, logits, w13, s13, w2, s2)


def _inputs(mesh, ep_size, local_tokens, seed=0):
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(
        rng.normal(0, 1, (local_tokens * ep_size, HIDDEN)) / 4, jnp.bfloat16)
    logits = jnp.asarray(
        rng.normal(0, 1, (local_tokens * ep_size, NUM_EXPERTS)), jnp.bfloat16)
    w13, s13, w2, s2 = _weights(seed + 1, NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    return tuple(_shard(x, mesh) for x in (tokens, logits, w13, s13, w2, s2))


SMALL = dict(hidden=512, inter=256, experts=16, top_k=2, local_tokens=8)


def _small_inputs(ep_size, seed=0):
    """Both weight layouts from one source, plus tokens and logits."""
    rng = np.random.default_rng(seed)
    h, f, e = SMALL["hidden"], SMALL["inter"], SMALL["experts"]
    tokens = jnp.asarray(
        rng.normal(0, 1, (SMALL["local_tokens"] * ep_size, h)) / 4,
        jnp.bfloat16)
    logits = jnp.asarray(
        rng.normal(0, 1, (SMALL["local_tokens"] * ep_size, e)), jnp.bfloat16)
    native = rng.normal(0, 0.05, (e, 2, h, f))
    w2 = rng.normal(0, 0.05, (e, f, h))
    ns = np.abs(native).max(axis=2, keepdims=True) / FP8_MAX  # (e,2,1,f)
    s2 = np.abs(w2).max(axis=1, keepdims=True) / FP8_MAX  # (e,1,h)
    q_native = jnp.asarray(native / ns, FP8)
    q_concat = jnp.asarray(
        np.concatenate(
            [np.asarray(q_native[:, 0]),
             np.asarray(q_native[:, 1])], -1), FP8)
    s_native = jnp.asarray(ns[:, :, :, None, :], jnp.float32)  # (e,2,1,1,f)
    s_concat = jnp.asarray(
        np.concatenate([ns[:, 0], ns[:, 1]], -1)[:, :, None],
        jnp.float32).reshape(e, 1, 1, 2 * f)
    return (tokens, logits, q_native, s_native, q_concat, s_concat,
            jnp.asarray(w2 / s2,
                        FP8), jnp.asarray(s2, jnp.float32).reshape(e, 1, 1, h))


def _small_ref(tokens, logits, q_native, s_native, q2, s2):
    """The unfused answer for the small fp8 case.

    Scales arrive from ``_small_inputs`` with the loader's extra singleton axis
    (``(E, 2, 1, 1, F)``); main's reference documents ``(E, 2, cdiv(H, wsz), F)``
    and both of ours are one block, so the axis is dropped here rather than in
    the caller. **This path runs only on a chip** -- the whole device suite is
    skipped without one -- so it is checked by review and by the first device
    run, not by the CPU gate.
    """
    return np.asarray(
        jax.device_get(
            ref_moe(
                tokens,
                q_native,
                q2,
                logits,
                SMALL["top_k"],
                renormalize_topk_logits=True,
                activation="silu",
                # main's reference takes one sub-channel size for both
                # matrices, and both of ours are the full contracting extent
                # (per-output-channel scales), which is what this asserts.
                subc_quant_wsz=SMALL["hidden"],
                w1_scale=s_native.reshape(s_native.shape[0], 2, 1,
                                          s_native.shape[-1]),
                w2_scale=s2)),
        np.float32)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
