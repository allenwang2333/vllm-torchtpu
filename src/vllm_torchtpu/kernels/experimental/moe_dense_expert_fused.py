# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Dense-expert decode MoE with the routing glue and the shared expert inside.

Derived from ``moe_dense_expert``; the routed arithmetic (activation
quantization, both expert matmuls, rounding points, per-expert masked FP32
accumulation in ascending local-expert order) is the incumbent's, unchanged.
What moves into the kernel:

* the EP global->local remap, the per-(row, local expert) coefficients and
  the compact active-expert list. Top-k ids, top-k weights and the local
  expert offset are scalar-prefetched to SMEM; the active list is built with
  scalar SMEM loops and each expert's coefficient column with vector ops over
  the [tokens, top-k] tile, so no XLA prep runs between router and kernel;
* the first routed-expert DMAs (and the shared-expert weight DMA) are issued
  before the activation quantization instead of after it;
* optionally, one per-channel FP8 shared expert ([H, 2*Is] gate|up and
  [Is, H] down) with the numerics of ``xla_quantized_matmul`` (per-token
  full-K activation quantization), whose BF16 output is added to the routed
  FP32 accumulator (coefficient 1) before the single BF16 output.

The caller still owns the router and the TP/EP all-reduce.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_dense_expert import (
    _MAX_EXPERT_BUFFERS,
    _quantize,
    can_use_dense_expert,
)
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block


def _shared_fixed_bytes(m, h, shared_i):
    """VMEM held by the shared expert: both weights, scales, its full-K FP8
    activation, its BF16 output and the BF16 gate/up and activation tiles."""
    if not shared_i:
        return 0
    weights = h * 2 * shared_i + shared_i * h
    scales = 4 * (2 * shared_i + h)
    acts = m * h * (1 + 2) + m * shared_i * (2 + 1) + m * 2 * shared_i * (4 + 2)
    return weights + scales + acts


def _select_expert_buffers(m, h, i, experts, topk, shared_i, vmem_capacity_bytes):
    """The incumbent's slot heuristic plus the shared expert's fixed bytes."""
    expert_bytes = 3 * h * i + 4 * (2 * i + h)
    fixed_bytes = m * h * (2 + 2 + 1 + 4)
    fixed_bytes += m * i * (2 + 1) + (h // 512 + i // 512) * m * 128 * 2
    # Top-k ids and weights tiles (lane-padded) replace the [M, E] coeff.
    fixed_bytes += 2 * m * pl.cdiv(topk, 128) * 128 * 4
    fixed_bytes += _shared_fixed_bytes(m, h, shared_i)
    budget = vmem_capacity_bytes * 4 // 5
    return max(1, min(_MAX_EXPERT_BUFFERS, (budget - fixed_bytes) // expert_bytes))


def can_use_fused_shared(hidden, ws13, ws13_scale, ws2, ws2_scale):
    """Shared-expert operand check: per-channel FP8, canonical (k, n) layout.

    Unlike the routed I (multiple of 512, quantized in 512-wide blocks), the
    shared I is a separate static size quantized over its full width, so it
    only has to be lane aligned (128 served, 256 twin).
    """
    if ws13 is None or ws2 is None or ws13_scale is None or ws2_scale is None:
        return False
    _, h = hidden.shape
    if ws13.ndim != 2 or ws2.ndim != 2:
        return False
    shared_i, out = ws2.shape
    if shared_i <= 0 or shared_i % 128 or out != h or ws13.shape != (h, 2 * shared_i):
        return False
    if ws13.dtype != jnp.float8_e4m3fn or ws2.dtype != jnp.float8_e4m3fn:
        return False
    if ws13_scale.dtype != jnp.float32 or ws2_scale.dtype != jnp.float32:
        return False
    return (
        ws13_scale.size == 2 * shared_i
        and ws13_scale.shape[-1] == 2 * shared_i
        and ws2_scale.size == h
        and ws2_scale.shape[-1] == h
    )


def can_use_dense_expert_fused(
    hidden,
    w1,
    w2,
    s1,
    s2,
    b1,
    b2,
    shared,
    *,
    activation,
    rhs_quant_dtype,
    skip_padded_tokens,
):
    """Routed admission is exactly the incumbent's; ``shared`` may be None."""
    if not can_use_dense_expert(
        hidden,
        w1,
        w2,
        s1,
        s2,
        b1,
        b2,
        activation=activation,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    ):
        return False
    return shared is None or can_use_fused_shared(hidden, *shared)


def _kernel(
    # Scalar prefetch (SMEM).
    ids_smem,
    wbits_smem,
    offset_smem,
    # Inputs.
    x_ref,
    ids_ref,
    w_ref,
    w1_hbm,
    w2_hbm,
    s1_hbm,
    s2_hbm,
    *rest,
    m,
    h,
    i,
    experts,
    topk,
    num_buffers,
    shared_i,
    routed_scale,
):
    if shared_i:
        ws13_hbm, ss13_ref, ws2_hbm, ss2_ref = rest[:4]
        rest = rest[4:]
    out_ref = rest[0]
    (
        w1_ref,
        w2_ref,
        s1_ref,
        s2_ref,
        xq_ref,
        xs_ref,
        a_ref,
        aq_ref,
        asc_ref,
        owner_ref,
        flags_ref,
        active_ids_ref,
        sem_ref,
    ) = rest[1:14]
    if shared_i:
        sw13_ref, sw2_ref, sh_ref, sgu_ref, shared_sem = rest[14:19]

    # 1. Routing-independent shared weights go first.
    if shared_i:
        shared_copies = (
            pltpu.make_async_copy(ws13_hbm, sw13_ref, shared_sem.at[0]),
            pltpu.make_async_copy(ws2_hbm, sw2_ref, shared_sem.at[1]),
        )
        for copy in shared_copies:
            copy.start()

    # 2. EP remap + active flags from the top-k ids/weights, scalar SMEM only.
    #    A route marks its local expert when it is local and its weight is not
    #    zero (padded rows carry zero weights), exactly the incumbent's
    #    any(coeff != 0) over non-negative routing weights.
    offset = offset_smem[0]

    def clear(e, carry):
        flags_ref[e] = jnp.int32(0)
        return carry

    lax.fori_loop(0, experts, clear, None)

    def mark(t, carry):
        row, col = t // topk, t % topk
        local = ids_smem[row, col] - offset
        nonzero = (wbits_smem[row, col] & jnp.int32(0x7FFFFFFF)) != 0

        @pl.when((local >= 0) & (local < experts) & nonzero)
        def _():
            flags_ref[local] = jnp.int32(1)

        return carry

    lax.fori_loop(0, m * topk, mark, None)

    # Ascending local-expert order, as the incumbent's compaction.
    def compact(e, count):
        @pl.when(flags_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + flags_ref[e]

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = active_ids_ref[step], step % num_buffers
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0], s1_ref.at[slot], sem_ref.at[slot, 2]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, 0], s2_ref.at[slot], sem_ref.at[slot, 3]
            ),
        ]

    # 3. Prime the routed ring before any vector work.
    for initial in range(num_buffers - 1):

        @pl.when(count > initial)
        def prime(*, initial=initial):
            for copy in transfers(jnp.int32(initial)):
                copy.start()

    # 4. Activation quantization (incumbent numerics) while the DMAs fly.
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    _quantize(x_ref[...], xq_ref, xs_ref)

    # 5. Shared expert, coefficient 1 per row in xla_quantized_matmul numerics:
    #    per-token full-K FP8 activation quantization, FP32 MXU accumulation,
    #    * x scale * per-channel w scale, BF16 output; silu(gate) * up in BF16.
    if shared_i:
        xq_s, xsc_s = quantize_block(x_ref[...], axis=1, target_dtype=jnp.float8_e4m3fn)
        xsc_s = xsc_s.astype(jnp.float32)
        for copy in shared_copies:
            copy.wait()
        gu = jnp.zeros((m, 2 * shared_i), jnp.float32)
        for k in range(0, h, 512):
            gu += jnp.matmul(
                xq_s[:, k : k + 512],
                sw13_ref[k : k + 512, :],
                preferred_element_type=jnp.float32,
            )
        # Fusion boundaries follow the XLA path: the gate/up output is a
        # materialized BF16 tensor (its own fusion), while silu(gate) * up is
        # fused into the down activation's quantization, so it is not forced
        # through a BF16 buffer here either. FP8 rounding of the down
        # activation is sensitive to that intermediate precision.
        sgu_ref[...] = ((gu * xsc_s) * ss13_ref[...]).astype(jnp.bfloat16)
        act = jax.nn.silu(sgu_ref[:, :shared_i]) * sgu_ref[:, shared_i:]
        aq_s, asc_s = quantize_block(act, axis=1, target_dtype=jnp.float8_e4m3fn)
        asc_s = asc_s.astype(jnp.float32)
        for n in range(0, h, 512):
            part = jnp.matmul(
                aq_s, sw2_ref[:, n : n + 512], preferred_element_type=jnp.float32
            )
            sh_ref[:, n : n + 512] = ((part * asc_s) * ss2_ref[:, n : n + 512]).astype(
                jnp.bfloat16
            )

    local_ids = ids_ref[...] - offset

    def compute(e, w1_ref, w2_ref, s1_ref, s2_ref):
        # Same values, same FP32 summation order over routes as the
        # incumbent's XLA coeff build: sum_k (local_id_k == e) * w_k.
        coeff = jnp.zeros((m, 1), jnp.float32)
        for k in range(topk):
            coeff = coeff + jnp.where(
                local_ids[:, k : k + 1] == e, w_ref[:, k : k + 1], 0.0
            )
        for n in range(0, i, 128):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            scale = jnp.concatenate(
                (s1_ref[:, n : n + 128], s1_ref[:, i + n : i + n + 128]), axis=1
            )
            for k in range(0, h, 512):
                rhs = jnp.concatenate(
                    (
                        w1_ref[k : k + 512, n : n + 128],
                        w1_ref[k : k + 512, i + n : i + n + 128],
                    ),
                    axis=1,
                )
                part = jnp.matmul(
                    xq_ref[:, k : k + 512], rhs, preferred_element_type=jnp.float32
                ).astype(jnp.bfloat16)
                xs = xs_ref[k // 512, :, :]
                part *= jnp.concatenate((xs, xs), axis=1)
                part *= scale.astype(jnp.bfloat16)
                acc += part
            a_ref[:, n : n + 128] = jax.nn.silu(acc[:, :128]) * acc[:, 128:]
        _quantize(a_ref[...], aq_ref, asc_ref)
        for n in range(0, h, 256):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            for k in range(0, i, 512):
                part = jnp.matmul(
                    aq_ref[:, k : k + 512],
                    w2_ref[k : k + 512, n : n + 256],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                asc = asc_ref[k // 512, :, :]
                part *= jnp.concatenate((asc, asc), axis=1)
                part *= s2_ref[:, n : n + 256].astype(jnp.bfloat16)
                acc += part
            weighted = jnp.where(coeff != 0, acc.astype(jnp.float32) * coeff, 0)
            owner_ref[:, n : n + 256] += weighted

    # 6. Routed ring, unchanged from the incumbent.
    @pl.when(count > 0)
    def run_experts():
        def expert_body(step, _):
            if num_buffers == 1:
                for copy in transfers(step):
                    copy.start()
            for copy in transfers(step):
                copy.wait()

            if num_buffers > 1:

                @pl.when(step + num_buffers - 1 < count)
                def prefetch():
                    for copy in transfers(step + num_buffers - 1):
                        copy.start()

            slot = step % num_buffers
            compute(
                active_ids_ref[step],
                w1_ref.at[slot],
                w2_ref.at[slot],
                s1_ref.at[slot],
                s2_ref.at[slot],
            )
            return None

        lax.fori_loop(0, count, expert_body, None)

    # 7. One BF16 write: routed FP32 sum (x the runner's routed scale, when
    #    it is moved in here) + the shared expert at coefficient 1.
    routed = owner_ref[...]
    if routed_scale != 1.0:
        routed = routed * jnp.float32(routed_scale)
    if shared_i:
        routed = routed + sh_ref[...].astype(jnp.float32)
    out_ref[...] = routed.astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("routed_scale",))
def dense_expert_moe_fused(
    hidden,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    topk_ids,
    experts_start=None,
    shared=None,
    *,
    routed_scale=1.0,
):
    """Local routed (+ optional shared) contribution, BF16 [tokens, hidden].

    ``topk_ids`` are GLOBAL expert ids; ``experts_start`` is this rank's first
    local expert (0-d int32, traced) or None for no EP. Ids outside
    [experts_start, experts_start + E) are ignored. ``topk_weights`` must
    already be renormalized/scaled with padded rows zeroed; they are rounded
    to ``hidden.dtype`` first, as fused_moe_func does for the incumbent.
    ``shared`` is None or (ws13 f8[H, 2*Is], ws13_scale f32[2*Is],
    ws2 f8[Is, H], ws2_scale f32[H]); its BF16 output is added in FP32 with
    coefficient 1. ``routed_scale`` (static) multiplies the routed FP32 sum
    before that add -- the MoE runner's ``routed_scaling_factor`` when the
    caller moves it in here, instead of scaling the BF16 output afterwards.
    Callers must first check can_use_dense_expert_fused.
    """
    m, h = hidden.shape
    experts, i, _ = w2.shape
    topk = topk_ids.shape[1]
    shared_i = 0 if shared is None else shared[2].shape[0]
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    num_buffers = _select_expert_buffers(
        m, h, i, experts, topk, shared_i, vmem_capacity_bytes
    )
    ids = topk_ids.astype(jnp.int32)
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    # A bitcast only (no reshape: a [M, K] -> [M*K] reshape is a relayout
    # copy on TPU); 2D SMEM scalar prefetch as paged attention's page table.
    wbits = lax.bitcast_convert_type(weights, jnp.int32)
    if experts_start is None:
        offset = jnp.zeros((1,), jnp.int32)
    else:
        offset = jnp.asarray(experts_start, jnp.int32).reshape(1)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
        *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
    ]
    operands = [ids, wbits, offset, hidden, ids, weights, w1, w2, s1, s2]
    scratch = [
        pltpu.VMEM((num_buffers, h, 2 * i), jnp.float8_e4m3fn),
        pltpu.VMEM((num_buffers, i, h), jnp.float8_e4m3fn),
        pltpu.VMEM((num_buffers, 1, 2 * i), jnp.float32),
        pltpu.VMEM((num_buffers, 1, h), jnp.float32),
        pltpu.VMEM((m, h), jnp.float8_e4m3fn),
        pltpu.VMEM((h // 512, m, 128), jnp.bfloat16),
        pltpu.VMEM((m, i), jnp.bfloat16),
        pltpu.VMEM((m, i), jnp.float8_e4m3fn),
        pltpu.VMEM((i // 512, m, 128), jnp.bfloat16),
        pltpu.VMEM((m, h), jnp.float32),
        pltpu.SMEM((experts,), jnp.int32),
        pltpu.SMEM((experts,), jnp.int32),
        pltpu.SemaphoreType.DMA((num_buffers, 4)),
    ]
    if shared_i:
        ws13, ws13_scale, ws2, ws2_scale = shared
        in_specs += [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ]
        operands += [
            ws13,
            ws13_scale.reshape(1, 2 * shared_i),
            ws2,
            ws2_scale.reshape(1, h),
        ]
        scratch += [
            pltpu.VMEM((h, 2 * shared_i), jnp.float8_e4m3fn),
            pltpu.VMEM((shared_i, h), jnp.float8_e4m3fn),
            pltpu.VMEM((m, h), jnp.bfloat16),
            pltpu.VMEM((m, 2 * shared_i), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2,)),
        ]
    return pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            topk=topk,
            num_buffers=num_buffers,
            shared_i=shared_i,
            routed_scale=float(routed_scale),
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_capacity_bytes),
        name=(
            f"dense_expert_moe_fused-e_{experts}-m_{m}-h_{h}-i_{i}-k_{topk}"
            f"-s_{shared_i}-buffers_{num_buffers}"
        ),
    )(*operands)
