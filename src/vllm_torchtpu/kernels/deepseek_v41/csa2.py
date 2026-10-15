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
"""Fused BF16 SWA + FP8 CSA + inverse RoPE in request-local BQ groups.

SWA keeps two fixed 128-token blocks and prefetches the following block into
separate staging memory. Causal/window masks advance without shifting KV lanes.
An MTP group that needs a third block is split; overlapping request-tail blocks
are masked to disjoint ranges. MXU RHS loading performs the K transposes.

CSA dequantizes KV to BF16 once for QK/PV reuse. All GEMMs use BF16 operands
and FP32 accumulators, softmax uses FP32, and probabilities are stored in BF16.
Each QK launches before the preceding Q's softmax. CSA PV starts as soon as
its P is ready; SWA PV reuses its RHS for two Qs. Both PV results stay in
independent MRB regions while the next tile runs softmax. QK uses two slots,
and each PV branch uses four. When ten slots exceed MRB capacity, one-Q tiles
use six slots instead. Startup, odd/even tails and final pops are explicit.
CPU interpret uses ordinary dots to check the math; it does not emulate the
asynchronous MXU protocol.

The initial grouped DMA/RoPE organization was adapted from
pallas-samples/kernels/deepseek-v4.1/csa2/csa2.py. Gather/cache publication,
non-causal windows, TP collectives and output projection remain caller-owned.
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
    sink_hbm_ref: Any,
    rope_hbm_ref: Any,
    output_hbm_ref: Any,
    sink_ref: Any,
    q_ref: Any,
    main_fp8_ref: Any,
    main_scale_ref: Any,
    main_bf16_ref: Any,
    swa_first_ref: Any,
    swa_shared_ref: Any,
    swa_cache_ids: Any,
    swa_cache_bases: Any,
    swa_prefetched_id: Any,
    rope_ref: Any,
    swa_score_ref: Any,
    main_p_ref: Any,
    swa_p_ref: Any,
    output_ref: Any,
    dma_sem: Any,
    *,
    scale: float,
    compress_ratio: int,
    bq: int,
    interpret: bool,
) -> None:
    request_index = pl.program_id(0)
    start, end = starts_ref[request_index], starts_ref[request_index + 1]
    num_heads = q_hbm_ref.shape[1]
    mxu_heads = pl.cdiv(num_heads, 16) * 16
    mrb_stride = mxu_heads // 4
    window_size = swa_kv_hbm_ref.shape[1]
    sink_dma = pltpu.make_async_copy(sink_hbm_ref, sink_ref, dma_sem.at[0])
    sink_dma.start()
    # Incomplete groups still participate in the grouped SWA QK.
    q_ref[...] = jnp.zeros(q_ref.shape, q_ref.dtype)
    sink_dma.wait()
    sink = jnp.broadcast_to(sink_ref[...][None, :], (128, num_heads)).T[:, :1]

    # Two physical 128-token slots never shift: key validity moves instead.
    swa_cache_ids[0] = -1
    swa_cache_ids[1] = -1
    swa_prefetched_id[0] = -1
    first_query_position = positions_ref[jnp.minimum(start, q_hbm_ref.shape[0] - 1)]
    anchor = jnp.maximum(0, first_query_position - 127)

    def block_dma(wanted):
        source_q = jnp.minimum(
            end - 1,
            start + jnp.maximum(0, anchor + wanted * 128 + 127 - first_query_position),
        )
        return pltpu.make_async_copy(
            swa_kv_hbm_ref.at[source_q], swa_first_ref, dma_sem.at[1]
        )

    def process_group(t0):
        p0 = positions_ref[t0]
        first_position = jnp.maximum(0, p0 - 127)
        block_id = (first_position - anchor) // 128
        # Split a BQ group whose +MTP tail would require a third block.
        count = jnp.minimum(
            jnp.minimum(bq, end - t0), anchor + (block_id + 2) * 128 - p0
        )

        def query_dmas(i):
            return tuple(
                pltpu.make_async_copy(src, dst, dma_sem.at[2 + 4 * i + k])
                for k, (src, dst) in enumerate(
                    (
                        (q_hbm_ref.at[t0 + i], q_ref.at[i]),
                        (main_kv_hbm_ref.at[t0 + i], main_fp8_ref.at[i]),
                        (main_scale_hbm_ref.at[t0 + i], main_scale_ref.at[i]),
                        (rope_hbm_ref.at[t0 + i], rope_ref.at[i]),
                    )
                )
            )

        @pl.loop(0, count)
        def start_query(i):
            for dma in query_dmas(i):
                dma.start()

        for offset in range(2):
            wanted = block_id + offset
            slot = wanted % 2

            @pl.when(swa_cache_ids[slot] != wanted)
            def fill_block():
                # Gathered windows expose a complete block at its last Q.
                # At request end the available window overlaps the previous
                # block; the score mask removes that overlap, including +5 MTP.
                source_q = jnp.minimum(
                    end - 1,
                    start
                    + jnp.maximum(
                        0, anchor + wanted * 128 + 127 - first_query_position
                    ),
                )
                dma = block_dma(wanted)

                @pl.when(swa_prefetched_id[0] != wanted)
                def start_block():
                    dma.start()

                dma.wait()
                swa_prefetched_id[0] = -1
                rows = pl.ds(slot * 128, 128)
                swa_shared_ref[rows, :] = swa_first_ref[...]
                swa_cache_ids[slot] = wanted
                swa_cache_bases[slot] = jnp.maximum(0, positions_ref[source_q] - 127)

                @pl.when(
                    (wanted >= 1)
                    & (anchor + (wanted + 1) * 128 - 1 <= positions_ref[end - 1])
                )
                def prefetch_block():
                    # Stage the following block without overwriting either live K.
                    block_dma(wanted + 1).start()
                    swa_prefetched_id[0] = wanted + 1

        def dequant(i):
            # Byte g of each word holds feature g * 128 + lane.
            words_ref = main_fp8_ref.at[i].bitcast(jnp.uint32).reshape(512, 128)
            for row in range(0, 512, 128):
                rows = pl.ds(row, 128)
                words = words_ref[rows, :]
                scales = main_scale_ref[i, 0, rows][:, None]
                for group in range(4):
                    if interpret:
                        # unpack_elementwise has no CPU lowering.
                        bits = (words >> (8 * group)) & 0xFF
                    else:
                        bits = pltpu.unpack_elementwise(
                            words,
                            index=group,
                            packed_dtype=jnp.uint8,
                            unpacked_dtype=jnp.uint32,
                        )
                    values = lax.bitcast_convert_type(
                        bits.astype(jnp.uint8), main_fp8_ref.dtype
                    ).astype(jnp.float32)
                    main_bf16_ref[i, rows, pl.ds(group * 128, 128)] = (
                        values * scales
                    ).astype(jnp.bfloat16)

        @pl.loop(0, count)
        def finish_query(i):
            for dma in query_dmas(i):
                dma.wait()

        def csa_launch(i, addr, *, pv):
            for k in range(2):
                ks = pl.ds(k * 256, 256)
                for mxu in range(2):
                    ns = pl.ds(mxu * 256, 256)
                    rhs = main_bf16_ref[i, ks, ns] if pv else main_bf16_ref[i, ns, ks]
                    pltpu.matmul_push_rhs(
                        rhs, staging_register=0, mxu_index=mxu, transpose=not pv
                    )
                    lhs = main_p_ref[i, :, ks] if pv else q_ref[i, :, ks]
                    lhs = jnp.pad(lhs, ((0, mxu_heads - num_heads), (0, 0)))
                    pltpu.matmul_acc_lhs(addr, lhs, mxu_index=mxu, load_staged_rhs=0)

        def csa_pop(i, addr, *, pv):
            if interpret:
                lhs = main_p_ref[i] if pv else q_ref[i]
                rhs = main_bf16_ref[i] if pv else main_bf16_ref[i].T
                return jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
            return jnp.concatenate(
                tuple(
                    pltpu.matmul_pop(
                        addr, (mxu_heads, 256), jnp.float32, mxu_index=mxu
                    )[:num_heads]
                    for mxu in range(2)
                ),
                axis=1,
            )

        if interpret:
            swa_score_ref[...] = jnp.dot(
                q_ref[...].reshape(bq * num_heads, 512),
                swa_shared_ref[...].T,
                preferred_element_type=jnp.float32,
            ).reshape(bq, num_heads, 256)
        else:
            # Split legal head domains; a sub-16-row head tail is indivisible.
            first_heads = min(num_heads, ((num_heads + 16) // 32) * 16)
            partitions = [(0, 0, first_heads)] if first_heads else []
            if num_heads > first_heads:
                partitions.append(
                    (len(partitions), first_heads, num_heads - first_heads)
                )
            for k in range(2):
                ks = pl.ds(k * 256, 256)
                rhs = swa_shared_ref[:, ks]
                for mxu, head_start, heads in partitions:
                    rows = pl.cdiv(heads, 16) * 16
                    pltpu.matmul_push_rhs(
                        rhs, staging_register=0, mxu_index=mxu, transpose=True
                    )
                    for i in range(bq):
                        lhs = q_ref[i, pl.ds(head_start, heads), ks]
                        lhs = jnp.pad(lhs, ((0, rows - heads), (0, 0)))
                        pltpu.matmul_acc_lhs(
                            i * rows // 4,
                            lhs,
                            mxu_index=mxu,
                            load_staged_rhs=0 if i == 0 else None,
                        )
            for i in range(bq):
                for mxu, head_start, heads in partitions:
                    rows = pl.cdiv(heads, 16) * 16
                    swa_score_ref[i, pl.ds(head_start, heads), :] = pltpu.matmul_pop(
                        i * rows // 4, (rows, 256), jnp.float32, mxu_index=mxu
                    )[:heads]

        dequant(0)
        if not interpret:
            csa_launch(0, 0, pv=False)

        def softmax(i, addr, *, has_next=True):
            if has_next:
                dequant(i + 1)
            if not interpret and has_next:
                csa_launch(i + 1, mrb_stride - addr, pv=False)
            main_logits = csa_pop(i, addr, pv=False) * scale
            main_count = jnp.minimum((p0 + i + 1) // compress_ratio, 512)
            main_logits = jnp.where(
                jnp.arange(512)[None, :] < main_count, main_logits, -jnp.inf
            )
            lanes = jnp.arange(128)
            key_positions = jnp.concatenate(
                (swa_cache_bases[0] + lanes, swa_cache_bases[1] + lanes)
            )
            # Concatenate integer boundaries before constructing the boolean mask.
            unique = key_positions >= jnp.concatenate(
                tuple(
                    jnp.full((128,), anchor + swa_cache_ids[slot] * 128, jnp.int32)
                    for slot in range(2)
                )
            )
            swa_valid = (key_positions >= jnp.maximum(0, p0 + i - window_size + 1)) & (
                key_positions <= p0 + i
            )
            swa_logits = jnp.where(
                (swa_valid & unique)[None, :], swa_score_ref[i] * scale, -jnp.inf
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

        # Each 256-wide FP32 MRB has 256 addresses, four rows per address.
        if 10 * mrb_stride <= 256:

            def launch_tile(first, slot, *, size=2, has_next=True):
                for row in range(size):
                    softmax(
                        first + row,
                        row * mrb_stride,
                        has_next=row + 1 < size or has_next,
                    )
                    if not interpret:
                        csa_launch(
                            first + row, (2 + 2 * slot + row) * mrb_stride, pv=True
                        )
                if not interpret:
                    for mxu in range(2):
                        pltpu.matmul_push_rhs(
                            swa_shared_ref[:, pl.ds(mxu * 256, 256)],
                            staging_register=0,
                            mxu_index=mxu,
                        )
                        for row in range(size):
                            lhs = jnp.pad(
                                swa_p_ref[first + row],
                                ((0, mxu_heads - num_heads), (0, 0)),
                            )
                            pltpu.matmul_acc_lhs(
                                (6 + 2 * slot + row) * mrb_stride,
                                lhs,
                                mxu_index=mxu,
                                load_staged_rhs=0 if row == 0 else None,
                            )

            def finish_tile_output(i, slot, row):
                main_output = csa_pop(i, (2 + 2 * slot + row) * mrb_stride, pv=True)
                if interpret:
                    swa_output = jnp.dot(
                        swa_p_ref[i],
                        swa_shared_ref[...],
                        preferred_element_type=jnp.float32,
                    )
                else:
                    swa_output = jnp.concatenate(
                        tuple(
                            pltpu.matmul_pop(
                                (6 + 2 * slot + row) * mrb_stride,
                                (mxu_heads, 256),
                                jnp.float32,
                                mxu_index=mxu,
                            )[:num_heads]
                            for mxu in range(2)
                        ),
                        axis=1,
                    )
                output = main_output + swa_output
                tail = output[:, 384:]
                partner = jnp.where(
                    (jnp.arange(128) % 2 == 0)[None, :],
                    pltpu.roll(tail, 127, axis=1),
                    pltpu.roll(tail, 1, axis=1),
                )
                rotated = (
                    tail * rope_ref[i, 0][None, :] + partner * rope_ref[i, 1][None, :]
                )
                output_ref[i, :, :384] = output[:, :384].astype(jnp.bfloat16)
                output_ref[i, :, 384:] = rotated.astype(jnp.bfloat16)
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 4 * bq + i],
                ).start()

            def finish_tile(first, slot, *, size=2):
                for row in range(size):
                    finish_tile_output(first + row, slot, row)

            def advance_tile(first, slot, *, size=2, has_next=True):
                launch_tile(first, slot, size=size, has_next=has_next)
                finish_tile(first - 2, 1 - slot)

            @pl.when(count == 1)
            def one():
                launch_tile(0, 0, size=1, has_next=False)
                finish_tile(0, 0, size=1)

            @pl.when(count == 2)
            def two():
                launch_tile(0, 0, has_next=False)
                finish_tile(0, 0)

            @pl.when(count > 2)
            def tiled_pipeline():
                full_tiles = (count - 1) // 2
                launch_tile(0, 0)

                @pl.loop(0, (full_tiles - 1) // 2)
                def tile_pairs(pair):
                    advance_tile(4 * pair + 2, 1)
                    advance_tile(4 * pair + 4, 0)

                def tail(slot):
                    @pl.when(count % 2 == 0)
                    def even():
                        advance_tile(2 * full_tiles, slot, has_next=False)
                        finish_tile(2 * full_tiles, slot)

                    @pl.when(count % 2 != 0)
                    def odd():
                        advance_tile(2 * full_tiles, slot, size=1, has_next=False)
                        finish_tile(2 * full_tiles, slot, size=1)

                @pl.when(full_tiles % 2 == 0)
                def even_full():
                    advance_tile(2 * (full_tiles - 1), 1)
                    tail(0)

                @pl.when(full_tiles % 2 != 0)
                def odd_full():
                    tail(1)

        else:

            def pv_launch(i, slot):
                if interpret:
                    return
                # Keep QK, CSA PV and SWA PV in disjoint MRB ranges. Each side
                # owns N=256, so H<=128 needs at most 6*32 of 256 addresses.
                csa_launch(i, (2 + slot) * mrb_stride, pv=True)
                for mxu in range(2):
                    pltpu.matmul_push_rhs(
                        swa_shared_ref[:, pl.ds(mxu * 256, 256)],
                        staging_register=0,
                        mxu_index=mxu,
                    )
                    lhs = jnp.pad(swa_p_ref[i], ((0, mxu_heads - num_heads), (0, 0)))
                    pltpu.matmul_acc_lhs(
                        (4 + slot) * mrb_stride, lhs, mxu_index=mxu, load_staged_rhs=0
                    )

            def finish_output(i, slot):
                main_output = csa_pop(i, (2 + slot) * mrb_stride, pv=True)
                if interpret:
                    swa_output = jnp.dot(
                        swa_p_ref[i],
                        swa_shared_ref[...],
                        preferred_element_type=jnp.float32,
                    )
                else:
                    swa_output = jnp.concatenate(
                        tuple(
                            pltpu.matmul_pop(
                                (4 + slot) * mrb_stride,
                                (mxu_heads, 256),
                                jnp.float32,
                                mxu_index=mxu,
                            )[:num_heads]
                            for mxu in range(2)
                        ),
                        axis=1,
                    )
                output = main_output + swa_output
                tail = output[:, 384:]
                partner = jnp.where(
                    (jnp.arange(128) % 2 == 0)[None, :],
                    pltpu.roll(tail, 127, axis=1),
                    pltpu.roll(tail, 1, axis=1),
                )
                rotated = (
                    tail * rope_ref[i, 0][None, :] + partner * rope_ref[i, 1][None, :]
                )
                output_ref[i, :, :384] = output[:, :384].astype(jnp.bfloat16)
                output_ref[i, :, 384:] = rotated.astype(jnp.bfloat16)
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 4 * bq + i],
                ).start()

            def advance(i, slot, *, has_next=True):
                softmax(i, slot * mrb_stride, has_next=has_next)
                pv_launch(i, slot)
                # The previous P@V remains in its own MRB while the current Q
                # runs softmax. Output DMA uses a distinct per-Q buffer.
                finish_output(i - 1, 1 - slot)

            @pl.when(count == 1)
            def single_query():
                softmax(0, 0, has_next=False)
                pv_launch(0, 0)
                finish_output(0, 0)

            @pl.when(count > 1)
            def query_pipeline():
                softmax(0, 0)
                pv_launch(0, 0)

                @pl.loop(0, (count - 2) // 2)
                def pairs(pair):
                    advance(2 * pair + 1, 1)
                    advance(2 * pair + 2, 0)

                @pl.when(count % 2 == 0)
                def even_tail():
                    advance(count - 1, 1, has_next=False)
                    finish_output(count - 1, 1)

                @pl.when(count % 2 != 0)
                def odd_tail():
                    advance(count - 2, 1)
                    advance(count - 1, 0, has_next=False)
                    finish_output(count - 1, 0)

        @pl.loop(0, count)
        def wait_output(i):
            pltpu.make_async_copy(
                output_ref.at[i], output_hbm_ref.at[t0 + i], dma_sem.at[2 + 4 * bq + i]
            ).wait()

        return t0 + count

    lax.while_loop(lambda t: t < end, process_group, start)

    @pl.when(swa_prefetched_id[0] >= 0)
    def drain_block_prefetch():
        # A final partial BQ can finish before the prefetched block is needed.
        block_dma(swa_prefetched_id[0]).wait()

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
                    dma_sem.at[2 + 4 * bq + i],
                ).start()

            @pl.loop(0, count)
            def wait_zero(i):
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 4 * bq + i],
                ).wait()


@functools.partial(
    jax.jit, static_argnames=("scale", "compress_ratio", "bq", "interpret")
)
def csa2(
    q: jax.Array,
    main_kv: jax.Array,
    main_kv_scale: jax.Array,
    swa_kv: jax.Array,
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
      swa_kv: (T, 128, 512) BF16. Chronological, left-aligned windows:
        slot j is max(0, positions[t]-127)+j, with min(positions[t]+1, 128)
        valid rows. Overlapping rows must match within a request.
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
        ("swa_kv", swa_kv, (num_tokens, 128, 512), jnp.bfloat16),
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
    shared_window_size = 256
    return pl.pallas_call(
        functools.partial(
            main_kernel,
            scale=scale,
            compress_ratio=compress_ratio,
            bq=bq,
            interpret=interpret,
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(starts.shape[0] - 1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 6,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((num_heads,), jnp.float32),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, 512, 4, 128), jnp.float8_e4m3fn),
                pltpu.VMEM((bq, 1, 512), jnp.float32),
                pltpu.VMEM((bq, 512, 512), jnp.bfloat16),
                pltpu.VMEM((128, 512), jnp.bfloat16),
                pltpu.VMEM((shared_window_size, 512), jnp.bfloat16),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SMEM((1,), jnp.int32),
                pltpu.VMEM((bq, 2, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, shared_window_size), jnp.float32),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, num_heads, shared_window_size), jnp.bfloat16),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2 + 5 * bq,)),
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
        swa_kv,
        sink,
        rope_cache,
    )
