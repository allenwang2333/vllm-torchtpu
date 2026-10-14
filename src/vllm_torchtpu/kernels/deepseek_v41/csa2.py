# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SWA + CSA + inverse RoPE, stepping through each request in BQ groups.

The group DMA schedule, chronological shared window and lane-expanded RoPE
are adapted from pallas-samples/kernels/deepseek-v4.1/csa2/csa2.py. Q stays
BF16; both FP8 KV branches are dequantized once into BF16 for QK/PV reuse.
All four GEMMs use BF16 operands and FP32 accumulators; P is BF16.

One program handles a request, with BQ groups sequential within that request.
Each group loads its first SWA window and only one new KV row per later Q.
The shared SWA QK/PV merge token and head dimensions. CSA remains per Q.
Group input DMAs start together; output DMAs overlap subsequent Q outputs.
There is no prefetch or double buffering between groups.

Inputs follow the gathered, causal contract in the CSA2 kick-off document:
https://outline.infiscale-tech.com/doc/csa2-kick-off-huszFct1U7
Cache publication/gather, arbitrary record masks, non-causal windows, TP
collectives and wo_a/wo_b projection are outside this kernel.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def main_kernel(
    starts_ref: Any,
    positions_ref: Any,
    q_hbm_ref: Any,
    main_kv_hbm_ref: Any,
    main_scale_hbm_ref: Any,
    swa_kv_hbm_ref: Any,
    swa_scale_hbm_ref: Any,
    sink_hbm_ref: Any,
    rope_hbm_ref: Any,
    output_hbm_ref: Any,
    sink_ref: Any,
    q_ref: Any,
    main_fp8_ref: Any,
    main_scale_ref: Any,
    main_bf16_ref: Any,
    swa_first_ref: Any,
    swa_first_scale_ref: Any,
    swa_rows_ref: Any,
    swa_rows_scale_ref: Any,
    swa_shared_ref: Any,
    rope_ref: Any,
    swa_score_ref: Any,
    main_p_ref: Any,
    swa_p_ref: Any,
    swa_output_ref: Any,
    output_ref: Any,
    dma_sem: Any,
    *,
    scale: float,
    compress_ratio: int,
    bq: int,
) -> None:
    request_index = pl.program_id(0)
    start, end = starts_ref[request_index], starts_ref[request_index + 1]
    num_heads = q_hbm_ref.shape[1]
    window_size = swa_kv_hbm_ref.shape[1]
    shared_window_size = swa_shared_ref.shape[0]
    sink_dma = pltpu.make_async_copy(sink_hbm_ref, sink_ref, dma_sem.at[0])
    sink_dma.start()
    # Incomplete groups still participate in the two combined SWA GEMMs.
    q_ref[...] = jnp.zeros(q_ref.shape, q_ref.dtype)
    swa_p_ref[...] = jnp.zeros(swa_p_ref.shape, swa_p_ref.dtype)
    sink_dma.wait()
    sink = jnp.broadcast_to(sink_ref[...][None, :], (128, num_heads)).T[:, :1]

    @pl.loop(0, pl.cdiv(end - start, bq))
    def process_group(block):
        t0 = start + block * bq
        count = jnp.minimum(bq, end - t0)
        p0 = positions_ref[t0]
        first_position = jnp.maximum(0, p0 - window_size + 1)
        first_count = jnp.minimum(p0 + 1, window_size)

        def query_dmas(i):
            return tuple(
                pltpu.make_async_copy(src, dst, dma_sem.at[3 + 6 * i + k])
                for k, (src, dst) in enumerate(
                    (
                        (q_hbm_ref.at[t0 + i], q_ref.at[i]),
                        (main_kv_hbm_ref.at[t0 + i], main_fp8_ref.at[i]),
                        (main_scale_hbm_ref.at[t0 + i], main_scale_ref.at[i]),
                        (rope_hbm_ref.at[t0 + i], rope_ref.at[i]),
                    )
                )
            )

        def append_dmas(i):
            newest = jnp.minimum(p0 + i, window_size - 1)
            return (
                pltpu.make_async_copy(
                    swa_kv_hbm_ref.at[t0 + i, newest],
                    swa_rows_ref.at[i],
                    dma_sem.at[3 + 6 * i + 4],
                ),
                # Load an aligned scale block, then select the new row's scale.
                pltpu.make_async_copy(
                    swa_scale_hbm_ref.at[t0 + i, 0],
                    swa_rows_scale_ref.at[i],
                    dma_sem.at[3 + 6 * i + 5],
                ),
            )

        first_kv = pltpu.make_async_copy(
            swa_kv_hbm_ref.at[t0], swa_first_ref, dma_sem.at[1]
        )
        first_scale = pltpu.make_async_copy(
            swa_scale_hbm_ref.at[t0, 0], swa_first_scale_ref, dma_sem.at[2]
        )
        first_kv.start()
        first_scale.start()

        @pl.loop(0, count)
        def start_query(i):
            for dma in query_dmas(i):
                dma.start()

            @pl.when(i > 0)
            def start_append():
                for dma in append_dmas(i):
                    dma.start()

        swa_shared_ref[...] = jnp.zeros(swa_shared_ref.shape, jnp.bfloat16)
        first_kv.wait()
        first_scale.wait()
        swa_first = (
            swa_first_ref[...].reshape(window_size, 512).astype(jnp.float32)
            * swa_first_scale_ref[...][:, None]
        ).astype(jnp.bfloat16)
        swa_shared_ref[:window_size, :] = jnp.where(
            jnp.arange(window_size)[:, None] < first_count, swa_first, 0
        )

        @pl.loop(0, count)
        def finish_query(i):
            for dma in query_dmas(i):
                dma.wait()
            main_bf16_ref[i] = (
                main_fp8_ref[i].reshape(512, 512).astype(jnp.float32)
                * main_scale_ref[i, 0][:, None]
            ).astype(jnp.bfloat16)

            @pl.when(i > 0)
            def append_swa():
                for dma in append_dmas(i):
                    dma.wait()
                newest = jnp.minimum(p0 + i, window_size - 1)
                row_scale = jnp.sum(
                    jnp.where(
                        jnp.arange(window_size) == newest, swa_rows_scale_ref[i], 0
                    ),
                    keepdims=True,
                )
                row = (
                    swa_rows_ref[i].reshape(1, 512).astype(jnp.float32) * row_scale
                ).astype(jnp.bfloat16)
                # Avoid a dynamically unaligned VMEM write for the appended row.
                swa_shared_ref[...] = jnp.where(
                    jnp.arange(shared_window_size)[:, None] == first_count + i - 1,
                    row,
                    swa_shared_ref[...],
                )

        swa_score_ref[...] = lax.dot_general(
            q_ref[...].reshape(bq * num_heads, 512),
            swa_shared_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).reshape(bq, num_heads, shared_window_size)

        @pl.loop(0, count)
        def softmax(i):
            main_logits = (
                lax.dot_general(
                    q_ref[i],
                    main_bf16_ref[i],
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                * scale
            )
            main_count = jnp.minimum((p0 + i + 1) // compress_ratio, 512)
            main_logits = jnp.where(
                jnp.arange(512)[None, :] < main_count, main_logits, -jnp.inf
            )
            key_positions = first_position + jnp.arange(shared_window_size)
            swa_valid = (key_positions >= jnp.maximum(0, p0 + i - window_size + 1)) & (
                key_positions <= p0 + i
            )
            swa_logits = jnp.where(
                swa_valid[None, :], swa_score_ref[i] * scale, -jnp.inf
            )
            maximum = jnp.maximum(
                sink,
                jnp.maximum(
                    jnp.max(main_logits, axis=1, keepdims=True),
                    jnp.max(swa_logits, axis=1, keepdims=True),
                ),
            )
            main_exp = jnp.exp(main_logits - maximum)
            swa_exp = jnp.exp(swa_logits - maximum)
            denominator = (
                jnp.exp(sink - maximum)
                + jnp.sum(main_exp, axis=1, keepdims=True)
                + jnp.sum(swa_exp, axis=1, keepdims=True)
            )
            main_p_ref[i] = (main_exp / denominator).astype(jnp.bfloat16)
            swa_p_ref[i] = (swa_exp / denominator).astype(jnp.bfloat16)

        swa_output_ref[...] = lax.dot_general(
            swa_p_ref[...].reshape(bq * num_heads, shared_window_size),
            swa_shared_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).reshape(bq, num_heads, 512)

        @pl.loop(0, count)
        def finish_output(i):
            output = swa_output_ref[i] + lax.dot_general(
                main_p_ref[i],
                main_bf16_ref[i],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            tail = output[:, 384:]
            partner = jnp.where(
                (jnp.arange(128) % 2 == 0)[None, :],
                pltpu.roll(tail, 127, axis=1),
                pltpu.roll(tail, 1, axis=1),
            )
            rotated = tail * rope_ref[i, 0][None, :] + partner * rope_ref[i, 1][None, :]
            output_ref[i, :, :384] = output[:, :384].astype(jnp.bfloat16)
            output_ref[i, :, 384:] = rotated.astype(jnp.bfloat16)
            pltpu.make_async_copy(
                output_ref.at[i], output_hbm_ref.at[t0 + i], dma_sem.at[3 + 6 * bq + i]
            ).start()

        @pl.loop(0, count)
        def wait_output(i):
            pltpu.make_async_copy(
                output_ref.at[i], output_hbm_ref.at[t0 + i], dma_sem.at[3 + 6 * bq + i]
            ).wait()

    @pl.when(request_index == starts_ref.shape[0] - 2)
    def clear_padding():
        output_ref[...] = jnp.zeros(output_ref.shape, jnp.bfloat16)

        @pl.loop(end, q_hbm_ref.shape[0], step=bq)
        def padding_group(t0):
            count = jnp.minimum(bq, q_hbm_ref.shape[0] - t0)

            @pl.loop(0, count)
            def start_zero(i):
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[3 + 6 * bq + i],
                ).start()

            @pl.loop(0, count)
            def wait_zero(i):
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[3 + 6 * bq + i],
                ).wait()


@functools.partial(
    jax.jit, static_argnames=("scale", "compress_ratio", "bq", "interpret")
)
def csa2(
    q: jax.Array,
    main_kv: jax.Array,
    main_kv_scale: jax.Array,
    swa_kv: jax.Array,
    swa_kv_scale: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_cache: jax.Array,
    *,
    scale: float = 512**-0.5,
    compress_ratio: int = 2,
    bq: int = 6,
    interpret: bool = False,
) -> jax.Array:
    """Fused causal SWA + CSA + inverse RoPE in request-local BQ groups.

    Args:
      q: (T, H, 512) BF16. H is divisible by 8, in [8, 128]. Q/KV
        are forward-rotated.
      main_kv: (T, 512, 4, 128) E4M3FN. The caller has compacted valid
        selections into the first min((positions[t]+1)//compress_ratio, 512)
        rows.
      main_kv_scale: (T, 512) FP32. One scale per complete NoPE+RoPE row.
      swa_kv: (T, 128, 512) E4M3FN. Chronological, left-aligned windows:
        slot j is max(0, positions[t]-127)+j, with min(positions[t]+1, 128)
        valid rows. Overlapping rows and scales must match within a request.
      swa_kv_scale: (T, 128) FP32. One scale per SWA row.
      starts: (B_max+1,) INT32, prefetched into SMEM. Nondecreasing request
        boundaries from zero to the valid Q count <= T. Repeats are empty
        requests.
      positions: (T,) INT32, prefetched into SMEM. Nonnegative and consecutive
        within each request.
      sink: (H,) FP32. Denominator-only logits.
      rope_cache: (T, 2, 128) FP32. Plane 0 contains cos, plane 1 signed sin.
        Lanes 0:64 are (1, 0); lanes 64:128 repeat cos for adjacent pairs,
        with sin signs (+, -), giving inverse rotation (x*c+y*s, y*c-x*s).
      scale: Static attention logit scale.
      compress_ratio: Positive static integer compression ratio.
      bq: Static group capacity in [1, 16]. Tails use only their valid Qs.
      interpret: Run the Pallas body on CPU without TPU code generation.

    Returns:
      (T, H, 512) BF16 output, already inverse-rotated, with padding zeroed.

    Values and FP32 intermediates must be finite, scales nonnegative.
    The caller is responsible for metadata values and window ordering.
    """
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("CSA2 q must have shape [T,H,512]")
    num_tokens, num_heads, _ = q.shape
    if num_heads < 8 or num_heads > 128 or num_heads % 8:
        raise NotImplementedError("CSA2 requires H divisible by 8, with 8 <= H <= 128")
    if type(bq) is not int or not 1 <= bq <= 16:
        raise ValueError("CSA2 bq must be a static Python int in [1,16]")
    if type(compress_ratio) is not int or compress_ratio < 1:
        raise ValueError("CSA2 compress_ratio must be a positive static Python int")
    if starts.ndim != 1 or starts.shape[0] < 2:
        raise ValueError("CSA2 starts must have shape [B_max+1], B_max >= 1")
    specs = (
        ("q", q, (num_tokens, num_heads, 512), jnp.bfloat16),
        ("main_kv", main_kv, (num_tokens, 512, 4, 128), jnp.float8_e4m3fn),
        ("main_kv_scale", main_kv_scale, (num_tokens, 512), jnp.float32),
        ("swa_kv", swa_kv, (num_tokens, 128, 512), jnp.float8_e4m3fn),
        ("swa_kv_scale", swa_kv_scale, (num_tokens, 128), jnp.float32),
        ("starts", starts, starts.shape, jnp.int32),
        ("positions", positions, (num_tokens,), jnp.int32),
        ("sink", sink, (num_heads,), jnp.float32),
        ("rope_cache", rope_cache, (num_tokens, 2, 128), jnp.float32),
    )
    for name, value, shape, dtype in specs:
        if value.shape != shape or value.dtype != jnp.dtype(dtype):
            raise ValueError(f"CSA2 {name} must have shape {shape} and dtype {dtype}")
    if num_tokens == 0:
        return jnp.zeros(q.shape, q.dtype)
    shared_window_size = pl.cdiv(128 + bq - 1, 128) * 128
    return pl.pallas_call(
        functools.partial(
            main_kernel, scale=scale, compress_ratio=compress_ratio, bq=bq
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(starts.shape[0] - 1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 7,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((num_heads,), jnp.float32),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, 512, 4, 128), jnp.float8_e4m3fn),
                pltpu.VMEM((bq, 1, 512), jnp.float32),
                pltpu.VMEM((bq, 512, 512), jnp.bfloat16),
                pltpu.VMEM((128, 4, 128), jnp.float8_e4m3fn),
                pltpu.VMEM((128,), jnp.float32),
                pltpu.VMEM((bq, 4, 128), jnp.float8_e4m3fn),
                pltpu.VMEM((bq, 128), jnp.float32),
                pltpu.VMEM((shared_window_size, 512), jnp.bfloat16),
                pltpu.VMEM((bq, 2, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, shared_window_size), jnp.float32),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, num_heads, shared_window_size), jnp.bfloat16),
                pltpu.VMEM((bq, num_heads, 512), jnp.float32),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((3 + 7 * bq,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="csa2_swa_csa_inverse_rope",
    )(
        starts,
        positions,
        q,
        main_kv,
        main_kv_scale[:, None, :],
        swa_kv.reshape(num_tokens, 128, 4, 128),
        swa_kv_scale[:, None, :],
        sink,
        rope_cache,
    )
