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
"""CSA sparse MLA for DeepSeek-V4.1: LxL block-grouped candidate union for prefill
and double-buffered pipelined decode on pre-encoded physical cache rows."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    DEFAULT_VMEM_LIMIT_BYTES,
    _attention_kernel,
    _dequant_and_merge_bkv,
    _dequant_dsv4_fp8,
    _largest_divisor,
    align_to,
    cdiv,
    gather_page_ids,
    prepare_outputs,
    prepare_q_inputs,
    prepare_swa_inputs,
)

_INT32_MAX = 2147483647


_ENCODE_PAGE_CHUNK = 128


def _fused_encode_csa_kernel(
    token_start_ref,  # i32[1] scalar prefetch
    act_chunks_ref,  # i32[1] scalar prefetch
    windows_ref,  # i32[block_tokens, padded_pps]
    topk_ref,  # i32[block_tokens, topk]
    out_ref,  # i32[block_tokens, topk]
    *,
    num_chunks: int,
    pages_per_seq: int,
    total_num_pages: int,
    page_size: int,
    use_dynamic_chunks: bool = False,
):
    tk = topk_ref[...]
    is_padding = tk == -1
    logical = jnp.clip(tk // page_size, 0, pages_per_seq - 1)
    token_offset = tk % page_size

    page_ids = jnp.zeros_like(logical)
    if use_dynamic_chunks:
        act_chunks = act_chunks_ref[0]

        def _chunk_body(c, acc):
            window_chunk = windows_ref[
                :, pl.ds(c * _ENCODE_PAGE_CHUNK, _ENCODE_PAGE_CHUNK)
            ]
            local = logical - c * _ENCODE_PAGE_CHUNK
            gathered = jnp.take_along_axis(
                window_chunk, jnp.clip(local, 0, _ENCODE_PAGE_CHUNK - 1), axis=1
            )
            return jnp.where(
                (local >= 0) & (local < _ENCODE_PAGE_CHUNK), gathered, acc
            )

        page_ids = lax.fori_loop(0, act_chunks, _chunk_body, page_ids)
    else:
        for c in range(num_chunks):
            window_chunk = windows_ref[
                :, c * _ENCODE_PAGE_CHUNK : (c + 1) * _ENCODE_PAGE_CHUNK
            ]
            local = logical - c * _ENCODE_PAGE_CHUNK
            gathered = jnp.take_along_axis(
                window_chunk, jnp.clip(local, 0, _ENCODE_PAGE_CHUNK - 1), axis=1
            )
            page_ids = jnp.where(
                (local >= 0) & (local < _ENCODE_PAGE_CHUNK), gathered, page_ids
            )

    block_tokens, topk = tk.shape
    row_start = token_start_ref[0] + pl.program_id(0) * block_tokens
    rows = row_start + lax.broadcasted_iota(jnp.int32, (block_tokens, 1), 0)
    cols = lax.broadcasted_iota(jnp.int32, (1, topk), 1)
    flat_idx = rows * topk + cols

    if total_num_pages & (total_num_pages - 1) == 0:
        scattered_page_ids = jnp.bitwise_and(
            flat_idx * 104729, total_num_pages - 1
        )
    else:
        rem_p = lax.rem(flat_idx * 104729, total_num_pages)
        scattered_page_ids = jnp.where(
            rem_p < 0, rem_p + total_num_pages, rem_p
        )
    if page_size & (page_size - 1) == 0:
        scattered_token_offset = jnp.bitwise_and(
            flat_idx * 15485863, page_size - 1
        )
    else:
        rem_t = lax.rem(flat_idx * 15485863, page_size)
        scattered_token_offset = jnp.where(rem_t < 0, rem_t + page_size, rem_t)
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    raw = page_ids * page_size + token_offset
    out_ref[...] = jnp.where(is_padding, ~raw, raw)


def encode_csa_gather_indices(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    total_num_pages: int,
    page_size: int,
    token_start: jax.Array | int = 0,
) -> jax.Array:
    """Maps CSA top-k token indices to physical cache rows; padding (-1) gets a
    scattered row, stored bitwise inverted so it stays negative."""
    num_tokens, topk = topk_indices.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    if eff_max_seqs < max_num_seqs:
        page_indices = page_indices[: eff_max_seqs * pages_per_seq]
        cu_q_lens = cu_q_lens[: eff_max_seqs + 1]
        max_num_seqs = eff_max_seqs

    if num_tokens <= 1024:
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
        seq_ids_segment = jnp.sum(
            token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs],
            axis=1,
            dtype=jnp.int32,
        )
    elif jax.default_backend() != "tpu":
        tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
        seq_ids_segment = jnp.repeat(
            jnp.arange(max_num_seqs, dtype=jnp.int32),
            tokens_per_seq,
            total_repeat_length=num_tokens,
        )
    else:
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
        cmp_bf16 = (
            token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs]
        ).astype(jnp.bfloat16)
        ones_bf16 = jnp.ones((max_num_seqs - 1,), dtype=jnp.bfloat16)
        seq_ids_segment = jnp.einsum(
            "ts,s->t",
            cmp_bf16,
            ones_bf16,
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)

    if jax.default_backend() == "tpu" and topk % 128 == 0:
        num_chunks = cdiv(pages_per_seq, _ENCODE_PAGE_CHUNK)
        padded_pps = num_chunks * _ENCODE_PAGE_CHUNK
        page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
        if padded_pps != pages_per_seq:
            page_table = jnp.pad(
                page_table, ((0, 0), (0, padded_pps - pages_per_seq))
            )
        use_dynamic_chunks = num_chunks > 8 and num_tokens >= 128
        if use_dynamic_chunks:
            max_tk = jnp.max(topk_indices)
            act_chunks = jnp.clip(
                (max_tk // (page_size * _ENCODE_PAGE_CHUNK)) + 1, 1, num_chunks
            )
            act_chunks_arr = jnp.asarray([act_chunks], dtype=jnp.int32)
        else:
            act_chunks_arr = jnp.asarray([num_chunks], dtype=jnp.int32)

        eff_block_tokens = min(128, max(8, align_to(num_tokens, 8)))
        padded_tokens = align_to(num_tokens, eff_block_tokens)
        tk_in = topk_indices
        seq_seg = seq_ids_segment
        if padded_tokens != num_tokens:
            pad = padded_tokens - num_tokens
            seq_seg = jnp.pad(seq_seg, (0, pad))
            tk_in = jnp.pad(tk_in, ((0, pad), (0, 0)), constant_values=-1)

        max_chunk_tokens = max(
            eff_block_tokens,
            align_to(
                min(padded_tokens, (512 * 8192) // padded_pps), eff_block_tokens
            ),
        )
        out_chunks = []
        for c_start in range(0, padded_tokens, max_chunk_tokens):
            c_end = min(c_start + max_chunk_tokens, padded_tokens)
            c_len = c_end - c_start
            windows_chunk = page_table[seq_seg[c_start:c_end]]
            tok_start_arr = jnp.asarray(
                [token_start + c_start], dtype=jnp.int32
            )
            out_c = pl.pallas_call(
                functools.partial(
                    _fused_encode_csa_kernel,
                    num_chunks=num_chunks,
                    pages_per_seq=pages_per_seq,
                    total_num_pages=total_num_pages,
                    page_size=page_size,
                    use_dynamic_chunks=use_dynamic_chunks,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=2,
                    in_specs=[
                        pl.BlockSpec(
                            (eff_block_tokens, padded_pps), lambda t, *_: (t, 0)
                        ),
                        pl.BlockSpec(
                            (eff_block_tokens, topk), lambda t, *_: (t, 0)
                        ),
                    ],
                    out_specs=pl.BlockSpec(
                        (eff_block_tokens, topk), lambda t, *_: (t, 0)
                    ),
                    grid=(c_len // eff_block_tokens,),
                ),
                out_shape=jax.ShapeDtypeStruct((c_len, topk), jnp.int32),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    disable_bounds_checks=True,
                ),
                name="fused_encode_csa_gather_indices",
            )(
                tok_start_arr,
                act_chunks_arr,
                windows_chunk,
                tk_in[c_start:c_end],
            )
            out_chunks.append(out_c)
        out = (
            out_chunks[0]
            if len(out_chunks) == 1
            else jnp.concatenate(out_chunks, axis=0)
        )
        return out[:num_tokens]

    seq_page_ids = topk_indices // page_size
    token_offset = topk_indices % page_size
    page_ids = gather_page_ids(
        page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
    )
    is_padding = topk_indices == -1
    flat_element_index = (
        jnp.arange(num_tokens * topk, dtype=jnp.int32)
        + jnp.asarray(token_start, dtype=jnp.int32) * topk
    ).reshape(num_tokens, topk)
    # Large primes spread the padding rows across the cache.
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    raw_gather_indices = page_ids * page_size + token_offset
    return jnp.where(is_padding, ~raw_gather_indices, raw_gather_indices)


def _build_candidate_union_and_bitmask(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    b_q: int,
    m_union: int,
    total_cache_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes per-tile unique sorted candidate union [num_groups * m_union] and
    bit-packed 32-query membership bitmask [num_groups, m_union // 128, 128]
    using sort_key_val + parallel bitwise-OR scan + lax.top_k compaction.
    """
    num_tokens, topk = topk_indices.shape
    num_groups = num_tokens // b_q
    grouped_keys = topk_indices.reshape(num_groups, b_q * topk)
    q_idx = jnp.broadcast_to(
        jnp.arange(b_q, dtype=jnp.int32)[None, :, None], (num_groups, b_q, topk)
    ).reshape(num_groups, b_q * topk)
    packed_in = jnp.where(
        grouped_keys >= 0,
        jnp.bitwise_or(jnp.left_shift(grouped_keys, 5), q_idx),
        _INT32_MAX,
    )
    s_packed = lax.sort(packed_in, dimension=-1, is_stable=False)
    valid_s = s_packed < _INT32_MAX
    s_keys = jnp.where(
        valid_s, lax.shift_right_logical(s_packed, 5), _INT32_MAX
    )
    s_bits = jnp.where(
        valid_s,
        jnp.left_shift(jnp.int32(1), jnp.bitwise_and(s_packed, 31)),
        jnp.int32(0),
    )

    # Segmented right-to-left bitwise-OR across equal keys (run length <= 64)
    for shift in (1, 2, 4, 8, 16, 32):
        rhs_keys = jnp.concatenate(
            [
                s_keys[:, shift:],
                jnp.full((num_groups, shift), _INT32_MAX, dtype=jnp.int32),
            ],
            axis=-1,
        )
        rhs_bits = jnp.concatenate(
            [
                s_bits[:, shift:],
                jnp.zeros((num_groups, shift), dtype=jnp.int32),
            ],
            axis=-1,
        )
        s_bits = jnp.where(
            s_keys == rhs_keys, jnp.bitwise_or(s_bits, rhs_bits), s_bits
        )

    prev_keys = jnp.concatenate(
        [jnp.full((num_groups, 1), -1, dtype=jnp.int32), s_keys[:, :-1]],
        axis=-1,
    )
    is_unique = (s_keys < _INT32_MAX) & (s_keys != prev_keys)
    u_keys = jnp.where(is_unique, s_keys, _INT32_MAX)
    n_cols = b_q * topk
    if (
        jax.default_backend() == "tpu"
        and n_cols % 16 == 0
        and m_union % 16 == 0
        and hasattr(pltpu.get_tpu_info(), "sparse_core")
    ):
        from jax.experimental.pallas import tpu_sc as plsc

        sc = pltpu.get_tpu_info().sparse_core
        lanes = 16
        mesh = plsc.VectorSubcoreMesh(
            num_cores=sc.num_cores,
            num_subcores=sc.num_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore",
        )
        cores_per_chip = sc.num_cores * sc.num_subcores
        num_waves = (num_groups + cores_per_chip - 1) // cores_per_chip
        padded_g = num_waves * cores_per_chip
        u_keys_pad = (
            jnp.pad(
                u_keys,
                ((0, padded_g - num_groups), (0, 0)),
                constant_values=_INT32_MAX,
            )
            if padded_g > num_groups
            else u_keys
        )
        s_bits_pad = (
            jnp.pad(
                s_bits, ((0, padded_g - num_groups), (0, 0)), constant_values=0
            )
            if padded_g > num_groups
            else s_bits
        )
        out_stride = m_union + lanes

        def _sc_body(k_hbm, b_hbm, out_k_hbm, out_b_hbm, out_n_hbm):
            flat = (
                lax.axis_index("core") * sc.num_subcores
                + lax.axis_index("subcore")
            )

            def step(k_in_vmem, b_in_vmem, k_out_vmem, b_out_vmem, n_out_vmem):
                def init_out(i):
                    k_out_vmem[pl.ds(i * lanes, lanes)] = jnp.full(
                        (lanes,), _INT32_MAX, jnp.int32
                    )
                    b_out_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros(
                        (lanes,), jnp.int32
                    )

                plsc.parallel_loop(0, m_union // lanes, unroll=16)(init_out)
                has_valid = (
                    plsc.all_reduce_population_count(
                        k_in_vmem[pl.ds(0, lanes)] < _INT32_MAX
                    )[0]
                    > 0
                )

                @pl.when(~has_valid)
                def _empty_grp():
                    n_out_vmem[pl.ds(0, lanes)] = jnp.zeros((lanes,), jnp.int32)

                @pl.when(has_valid)
                def _active_grp():
                    def compact_step(i, woff):
                        kv = k_in_vmem[pl.ds(i * lanes, lanes)]
                        bv = b_in_vmem[pl.ds(i * lanes, lanes)]
                        sel = kv < _INT32_MAX
                        safe_woff = jnp.minimum(woff, m_union)
                        plsc.store_compressed(
                            k_out_vmem.at[pl.ds(safe_woff, lanes)], kv, mask=sel
                        )
                        plsc.store_compressed(
                            b_out_vmem.at[pl.ds(safe_woff, lanes)], bv, mask=sel
                        )
                        return woff + plsc.all_reduce_population_count(sel)[0]

                    total_u = plsc.parallel_loop(
                        0, n_cols // lanes, unroll=16, carry=jnp.int32(0)
                    )(compact_step)
                    n_out_vmem[pl.ds(0, lanes)] = jnp.full(
                        (lanes,), total_u, jnp.int32
                    )

            mode = pl.Buffered(1 if num_waves == 1 else 2)

            def spec(sz):
                return pl.BlockSpec(
                    (sz,),
                    lambda w: (w * cores_per_chip + flat,),
                    pipeline_mode=mode,
                )

            pltpu.emit_pipeline(
                step,
                grid=(num_waves,),
                in_specs=(spec(n_cols), spec(n_cols)),
                out_specs=(spec(out_stride), spec(out_stride), spec(lanes)),
            )(k_hbm, b_hbm, out_k_hbm, out_b_hbm, out_n_hbm)

        out_k, out_b, out_n = pl.kernel(
            _sc_body,
            out_type=(
                jax.ShapeDtypeStruct((padded_g * out_stride,), jnp.int32),
                jax.ShapeDtypeStruct((padded_g * out_stride,), jnp.int32),
                jax.ShapeDtypeStruct((padded_g * lanes,), jnp.int32),
            ),
            scratch_types=(),
            mesh=mesh,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True, needs_layout_passes=False
            ),
            name=f"sc_compact_direct_g{num_groups}_m{m_union}",
        )(u_keys_pad.reshape(-1), s_bits_pad.reshape(-1))

        union_sorted = out_k.reshape(padded_g, out_stride)[:num_groups, :m_union]
        cand_bits_tiled = out_b.reshape(padded_g, out_stride)[
            :num_groups, :m_union
        ].reshape(num_groups, m_union // 128, 128)
        n_unique = out_n.reshape(padded_g, lanes)[:num_groups, 0]
    else:
        n_unique = jnp.sum(is_unique.astype(jnp.int32), axis=-1)
        s_bits_clean = jnp.where(is_unique, s_bits, jnp.int32(0))
        u_sorted, bits_sorted = lax.sort(
            [u_keys, s_bits_clean], dimension=-1, is_stable=False, num_keys=1
        )
        union_sorted = u_sorted[:, :m_union]
        cand_bits_tiled = bits_sorted[:, :m_union].reshape(
            num_groups, m_union // 128, 128
        )

    flat_grp_idx = jnp.arange(num_groups * m_union, dtype=jnp.int32).reshape(
        num_groups, m_union
    )
    scattered_rows = (flat_grp_idx * 104729) % total_cache_rows
    union_gather_indices = jnp.where(
        union_sorted < _INT32_MAX, union_sorted, scattered_rows
    ).reshape(-1)
    return union_gather_indices, cand_bits_tiled, n_unique


@functools.partial(
    jax.jit,
    static_argnames=("b_q", "m_union", "total_cache_rows"),
)
def precompute_csa_union_and_bitmask(
    topk_indices: jax.Array,
    *,
    b_q: int = 32,
    m_union: int = 1536,
    total_cache_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precomputes the LxL Block-Grouped Candidate Union once per step for all
    Consumer CSA layers sharing the same Indexer topk_indices buffer."""
    return _build_candidate_union_and_bitmask(
        topk_indices, b_q, m_union, total_cache_rows
    )


def _grouped_attention_kernel(
    n_unique_ref,  # i32[num_groups]
    meta_ref,  # i32[3] (start_group, num_active_groups, num_valid_tokens)
    sem_ids_ref,  # i32[2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # i32[2] group_idx of in-flight bo DMA
    attention_sinks_ref,  # f32[l_stride]
    q_hbm_ref,  # bf16[num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # int32[num_groups, m_union, 128]
    cache_kv_rope_hbm_ref,  # int32[num_groups, m_union // 4, 128]
    cand_bits_hbm_ref,  # int32[num_groups, m_union // 128, 128]
    swa_accumution_hbm_ref,  # bf16[num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # f32[num_groups * l_stride]
    swa_m_hbm_ref,  # f32[num_groups * l_stride]
    o_hbm_ref,  # bf16[num_tokens, num_q_heads, head_dim]
    bkv_nope_x2_ref,  # u8[2, m_union * 4, 128]
    bkv_rope_x2_ref,  # bf16[2, m_union // 2, 128]
    bq_x2_ref,  # bf16[2, b_q, num_q_heads, head_dim]
    bo_x2_ref,  # bf16[2, b_q, num_q_heads, head_dim]
    bl_x2_ref,  # f32[2, l_stride]
    bm_x2_ref,  # f32[2, l_stride]
    swa_acc_x2_ref,  # bf16[2, b_q, num_q_heads, head_dim]
    cand_bits_x2_ref,  # int32[2, m_union // 128, 128]
    sems,  # DMA[(8, 2)]
    *,
    sm_scale: float,
    b_q: int,
    m_union: int,
    l_stride: int,
):
    """Fused LxL Block-Grouped Candidate Union Attention Kernel for Prefill."""
    _, num_q_heads, head_dim = q_hbm_ref.shape
    n_qh = b_q * num_q_heads
    q_dtype = q_hbm_ref.dtype
    cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
    cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)

    start_group = meta_ref[0]
    num_active_groups = meta_ref[1]
    num_valid_tokens = meta_ref[2]
    num_run_groups = num_active_groups - start_group
    p_id = pl.program_id(0)
    g = start_group + p_id

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_inputs(grp_idx, sem_idx, *, wait=False):
        tok_start = grp_idx * b_q
        l_start = grp_idx * l_stride
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[grp_idx, :, :],
                bkv_nope_x2_ref.at[sem_idx, :, :],
                sems.at[0, sem_idx],
                False,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[grp_idx, :, :],
                bkv_rope_x2_ref.at[sem_idx, :, :],
                sems.at[6, sem_idx],
                False,
            )
            _async_copy(
                q_hbm_ref.at[pl.ds(tok_start, b_q)],
                bq_x2_ref.at[sem_idx],
                sems.at[1, sem_idx],
                False,
            )
            _async_copy(
                cand_bits_hbm_ref.at[grp_idx],
                cand_bits_x2_ref.at[sem_idx],
                sems.at[7, sem_idx],
                False,
            )
            _async_copy(
                swa_accumution_hbm_ref.at[pl.ds(tok_start, b_q)],
                swa_acc_x2_ref.at[sem_idx],
                sems.at[3, sem_idx],
                False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(l_start, l_stride)],
                bl_x2_ref.at[sem_idx],
                sems.at[4, sem_idx],
                False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(l_start, l_stride)],
                bm_x2_ref.at[sem_idx],
                sems.at[5, sem_idx],
                False,
            )
        else:
            for s_id, ref in (
                (0, bkv_nope_x2_ref.at[sem_idx, :, :]),
                (6, bkv_rope_x2_ref.at[sem_idx, :, :]),
                (1, bq_x2_ref.at[sem_idx]),
                (7, cand_bits_x2_ref.at[sem_idx]),
                (3, swa_acc_x2_ref.at[sem_idx]),
                (4, bl_x2_ref.at[sem_idx]),
                (5, bm_x2_ref.at[sem_idx]),
            ):
                _async_copy(src=ref, dst=ref, sem=sems.at[s_id, sem_idx], wait=True)

    def _send_output(grp_idx, sem_idx, *, wait=False):
        tok_start = grp_idx * b_q
        sz = jnp.clip(num_valid_tokens - tok_start, 1, b_q)
        vmem_ref = bo_x2_ref.at[sem_idx, pl.ds(0, sz)]
        sem = sems.at[2, sem_idx]
        if not wait:
            bo_ids_ref[sem_idx] = grp_idx
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(tok_start, sz)],
                sem,
                False,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _wait_send_output(sem_idx):
        old_grp = bo_ids_ref[sem_idx]

        @pl.when(old_grp >= 0)
        def _():
            _send_output(old_grp, sem_idx, wait=True)

    @pl.when(p_id == 0)
    def _prologue():
        _fetch_inputs(start_group, 0, wait=False)

    bi_sem_idx = sem_ids_ref[0]
    next_g = g + 1
    next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

    @pl.when(next_g < num_active_groups)
    def _prefetch_next():
        sem_ids_ref[0] = next_bi_sem_idx
        _fetch_inputs(next_g, next_bi_sem_idx, wait=False)

    bo_sem_idx = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

    _fetch_inputs(g, bi_sem_idx, wait=True)

    @pl.when(p_id >= 2)
    def _wait_prev_send():
        _wait_send_output(bo_sem_idx)

    # 1. Dequantize union KV block ONCE for all `b_q` queries in this group
    bkv_nope = bkv_nope_x2_ref.at[bi_sem_idx][...].reshape(m_union, -1)
    bkv_rope = bkv_rope_x2_ref.at[bi_sem_idx][...].reshape(m_union // 2, -1)
    if bkv_nope.shape[1] == 512 and bkv_rope.shape[1] == 128:
        bkv = _dequant_and_merge_bkv(bkv_nope, bkv_rope)
    else:
        bkv_nope = _dequant_dsv4_fp8(bkv_nope)
        rope_dim = bkv_rope.shape[-1] // 2
        bkv_rope = jnp.concatenate(
            [bkv_rope[:, :rope_dim], bkv_rope[:, rope_dim:]], axis=0
        )
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)  # [m_union, head_dim]
    n_uniq = n_unique_ref[g]
    m_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
    bkv = jnp.where(m_span < n_uniq, bkv, 0)

    # 2. Full MXU QK^T across all `b_q * num_q_heads` queries simultaneously
    bq_flat = bq_x2_ref.at[bi_sem_idx][...].reshape(n_qh, head_dim)
    s = jnp.einsum("nd,md->nm", bq_flat, bkv, preferred_element_type=jnp.float32)
    s = s * sm_scale

    # 3. Apply in-VMEM bitpacked 32-query membership bitmask [1, m_union] & [512, 1]
    bits_row = cand_bits_x2_ref.at[bi_sem_idx][...].reshape(1, m_union)
    q_bit_col = jnp.left_shift(
        jnp.int32(1),
        lax.broadcasted_iota(jnp.int32, (n_qh, 1), 0) // num_q_heads,
    )
    s = jnp.where(jnp.bitwise_and(bits_row, q_bit_col) != 0, s, jnp.finfo(s.dtype).min)

    # 4. FlashAttention-2 Softmax with SWA state
    swa_m_flat = bm_x2_ref.at[bi_sem_idx, :n_qh][...][:, None]
    swa_l_flat = bl_x2_ref.at[bi_sem_idx, :n_qh][...][:, None]
    sinks_flat = attention_sinks_ref.at[:n_qh][...][:, None]

    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_curr = jnp.maximum(swa_m_flat, s_rowmax)
    p = jnp.exp(s - m_curr)
    exp_m_diff = jnp.exp(swa_m_flat - m_curr)
    p_rowsum = jnp.sum(p, axis=1, keepdims=True)
    l_total = exp_m_diff * swa_l_flat + p_rowsum + jnp.exp(sinks_flat - m_curr)

    # 5. Full MXU PV across all `b_q * num_q_heads` queries simultaneously
    pv = jnp.einsum("nm,md->nd", p, bkv, preferred_element_type=jnp.float32)
    swa_acc_flat = swa_acc_x2_ref.at[bi_sem_idx][...].reshape(-1, head_dim)
    acc = exp_m_diff * swa_acc_flat + pv
    out_flat = (
        lax.div(acc, l_total)
        if q_dtype == jnp.float32
        else (acc * pl.reciprocal(l_total, approx=True)).astype(q_dtype)
    )
    bo_x2_ref.at[bo_sem_idx][...] = out_flat.reshape(b_q, num_q_heads, head_dim)
    _send_output(g, bo_sem_idx, wait=False)

    @pl.when(p_id == num_run_groups - 1)
    def _epilogue():
        _wait_send_output(0)

        @pl.when(num_run_groups >= 2)
        def _wait_1():
            _wait_send_output(1)


def _run_chunked_fallback(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,
    num_valid_tokens: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float,
    gather_and_attention_chunk_size: int,
    attention_kernel_batch_size: int,
) -> jax.Array:
    """Single-launch / larger-chunk double-buffered path without host HLO loop bloat."""
    _, num_q_heads, head_dim = q.shape
    topk = topk_indices.shape[-1]
    kv_lens = jnp.sum(topk_indices >= 0, axis=-1)
    gather_indices = jnp.where(topk_indices < 0, ~topk_indices, topk_indices).ravel()

    # For decode / verify (<= 384 tokens), process the entire batch in 1 launch with
    # batch_size tuned so grid >= 2 for B=16 (8) and 16 for B>=64. For 512-1024 tokens,
    # use 256-token chunks (gcz_256) so SparseCore csa_gather overlaps TensorCore attention.
    if q.shape[0] <= 384:
        eff_chunk_size = q.shape[0]
        eff_batch_size = 8 if q.shape[0] <= 32 else attention_kernel_batch_size
    elif q.shape[0] <= 1024:
        eff_chunk_size = 256
        eff_batch_size = attention_kernel_batch_size
    else:
        eff_chunk_size = max(gather_and_attention_chunk_size, min(q.shape[0], 1024))
        eff_batch_size = attention_kernel_batch_size

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 6

    for i in range(cdiv(q.shape[0], eff_chunk_size)):
        start_pos = i * eff_chunk_size
        end_pos = min(start_pos + eff_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk : end_pos * topk]

        batch_size = _largest_divisor(chunk_size, eff_batch_size)
        valid_in_chunk = jnp.maximum(
            0, jnp.minimum(num_valid_tokens, end_pos) - start_pos
        )
        batch_end = start_pos + cdiv(valid_in_chunk, batch_size) * batch_size
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        gathered_nope, gathered_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
        )
        gathered_nope = gathered_nope.reshape(chunk_size, topk, -1)
        gathered_rope = gathered_rope.reshape(chunk_size, topk // 4, -1)
        lanes = gathered_nope.shape[-1]
        end_seq_idx = jnp.maximum(start_pos, batch_end)
        grid = (cdiv(end_seq_idx - start_pos, batch_size),)
        bq_buf = pltpu.VMEM((2, batch_size, num_q_heads, head_dim), q.dtype)
        bl_buf = pltpu.VMEM((2, batch_size, align_to(num_q_heads, 128)), jnp.float32)
        scratch_shapes = [
            pltpu.VMEM((2, batch_size, topk * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, batch_size, (topk // 4) * 2, lanes), jnp.bfloat16),
            bq_buf,
            bq_buf,
            bl_buf,
            bl_buf,
            bq_buf,
            pltpu.SemaphoreType.DMA((7, 2)),
        ]
        scalar_prefetches = (
            kv_lens,
            jnp.array([start_pos, end_seq_idx, num_valid_tokens], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        scope_name = f"SparseMLA-p_{topk}-bz_{batch_size}-gcz_{chunk_size}"
        q = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel, sm_scale=sm_scale, batch_size=batch_size
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=hbm,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={5: 0},
                name=scope_name,
            )
        )(
            *scalar_prefetches,
            attention_sinks,
            q,
            gathered_nope,
            gathered_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )
    return q


# The compressor writes this step's compressed KV tokens to the cache before
# this runs, so the encoded indices may point at them.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "ced_decoder",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk], already encoded
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    precomputed_union: tuple[jax.Array, jax.Array, jax.Array]
    | None = None,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int = 16,
    ced_decoder: bool = False,
) -> jax.Array:
    """DeepSeek-V4 sparse_ragged_paged_attention on encode_csa_gather_indices output."""
    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    num_tokens, num_q_heads, head_dim = q.shape
    if num_q_heads != actual_num_q_heads:
        attention_sinks = jnp.pad(
            attention_sinks,
            (0, num_q_heads - actual_num_q_heads),
            constant_values=jnp.finfo(attention_sinks.dtype).min,
        )
    assert swa_accumution.dtype == q.dtype

    topk = topk_indices.shape[-1]
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_valid_tokens = cu_q_lens[distribution[2]]
    total_cache_rows = cache_kv_nope.shape[0] * cache_kv_nope.shape[1]

    # Use LxL Block-Grouped Candidate Union only for non-CED prefill batches
    # (num_tokens > max_num_seqs and >= 512, divisible by 32), routing any
    # front-of-batch decode tokens [0, cu_q_lens[distribution[0]]) to the
    # per-token kernel so 32 distinct decode requests never overflow m_union=1536.
    b_q = 32
    m_union = align_to(3 * topk, 128)
    if (
        envs.VLLM_DSV41_CSA_UNION
        and not ced_decoder
        and num_tokens > max_num_seqs
        and num_tokens >= 512
        and num_tokens % b_q == 0
        and topk % 128 == 0
    ):
        if (
            swa_accumution.shape[1] != num_q_heads
            or swa_accumution.shape[2] != head_dim
        ):
            swa_accumution = jnp.pad(
                swa_accumution,
                (
                    (0, 0),
                    (0, num_q_heads - actual_num_q_heads),
                    (0, head_dim - actual_head_dim),
                ),
                constant_values=0,
            )
        if swa_l.shape[1] != num_q_heads:
            swa_l_qh = jnp.pad(
                swa_l, ((0, 0), (0, num_q_heads - actual_num_q_heads))
            )
            swa_m_qh = jnp.pad(
                swa_m, ((0, 0), (0, num_q_heads - actual_num_q_heads))
            )
        else:
            swa_l_qh = swa_l
            swa_m_qh = swa_m

        num_decode_tokens = cu_q_lens[distribution[0]]
        start_group = cdiv(num_decode_tokens, b_q)
        decode_end_tok = jnp.minimum(num_valid_tokens, start_group * b_q)
        decode_cap = min(num_tokens, align_to(max(1, max_num_seqs), 64))
        dec_batch_size = _largest_divisor(
            decode_cap, attention_kernel_batch_size
        )
        dec_batch_end = cdiv(decode_end_tok, dec_batch_size) * dec_batch_size
        dec_indices = jnp.where(
            topk_indices[:decode_cap] < 0,
            ~topk_indices[:decode_cap],
            topk_indices[:decode_cap],
        ).ravel()
        dec_nope, dec_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            dec_indices,
            num_valid_indices=dec_batch_end * topk,
            top_k=topk,
        )
        dec_nope = dec_nope.reshape(decode_cap, topk, -1)
        dec_rope = dec_rope.reshape(decode_cap, topk // 4, -1)

        if precomputed_union is not None:
            union_gather_indices, cand_bits_tiled, n_unique = precomputed_union
        else:
            union_gather_indices, cand_bits_tiled, n_unique = (
                _build_candidate_union_and_bitmask(
                    topk_indices, b_q, m_union, total_cache_rows
                )
            )
        num_groups = num_tokens // b_q
        num_active_groups = jnp.maximum(
            start_group, jnp.maximum(1, cdiv(num_valid_tokens, b_q))
        )
        num_run_groups = jnp.maximum(0, num_active_groups - start_group)
        num_valid_union_indices = num_active_groups * m_union
        gathered_nope, gathered_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            union_gather_indices,
            num_valid_indices=num_valid_union_indices,
            top_k=m_union,
        )
        gathered_nope = gathered_nope.reshape(num_groups, m_union, -1)
        gathered_rope = gathered_rope.reshape(num_groups, m_union // 4, -1)

        def _run_decode_tc(q_in):
            kv_lens = jnp.sum(topk_indices[:decode_cap] >= 0, axis=-1)
            if decode_cap < num_tokens:
                kv_lens = jnp.pad(kv_lens, (0, num_tokens - decode_cap))
            _, swa_l_dec, swa_m_dec = prepare_swa_inputs(
                swa_accumution[:decode_cap],
                swa_l[:decode_cap],
                swa_m[:decode_cap],
            )
            hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
            dec_in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [
                hbm_spec
            ] * 6
            dec_lanes = dec_nope.shape[-1]
            dec_bq_buf = pltpu.VMEM(
                (2, dec_batch_size, num_q_heads, head_dim), q_in.dtype
            )
            dec_bl_buf = pltpu.VMEM(
                (2, dec_batch_size, align_to(num_q_heads, 128)), jnp.float32
            )
            dec_scratch = [
                pltpu.VMEM(
                    (2, dec_batch_size, topk * 4, dec_lanes), jnp.uint8
                ),
                pltpu.VMEM(
                    (2, dec_batch_size, (topk // 4) * 2, dec_lanes),
                    jnp.bfloat16,
                ),
                dec_bq_buf,
                dec_bq_buf,
                dec_bl_buf,
                dec_bl_buf,
                dec_bq_buf,
                pltpu.SemaphoreType.DMA((7, 2)),
            ]
            dec_scalars = (
                kv_lens,
                jnp.array([0, dec_batch_end, decode_end_tok], jnp.int32),
                jnp.zeros((2,), jnp.int32),
                jnp.full((2,), -1, jnp.int32),
            )
            dec_scope = (
                f"SparseMLA-p_{topk}-bz_{dec_batch_size}-gcz_{decode_cap}"
            )
            return jax.named_scope(dec_scope)(
                pl.pallas_call(
                    functools.partial(
                        _attention_kernel,
                        sm_scale=sm_scale,
                        batch_size=dec_batch_size,
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=len(dec_scalars),
                        in_specs=dec_in_specs,
                        out_specs=hbm_spec,
                        grid=(cdiv(dec_batch_end, dec_batch_size),),
                        scratch_shapes=dec_scratch,
                    ),
                    compiler_params=pltpu.CompilerParams(
                        dimension_semantics=("arbitrary",),
                        vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                        disable_bounds_checks=True,
                    ),
                    out_shape=jax.ShapeDtypeStruct(
                        shape=q_in.shape, dtype=q_in.dtype
                    ),
                    input_output_aliases={5: 0},
                    name=dec_scope,
                )
            )(
                *dec_scalars,
                attention_sinks,
                q_in,
                dec_nope,
                dec_rope,
                swa_accumution,
                swa_l_dec,
                swa_m_dec,
            )

        q = lax.cond(decode_end_tok > 0, _run_decode_tc, lambda x: x, q)

        n_qh = b_q * num_q_heads
        l_stride = align_to(n_qh, 1024)
        if l_stride == n_qh:
            swa_l_flat = swa_l_qh.reshape(-1)
            swa_m_flat = swa_m_qh.reshape(-1)
            sinks_flat = jnp.tile(attention_sinks, (b_q,))
        else:
            swa_l_flat = jnp.pad(
                swa_l_qh.reshape(num_groups, n_qh),
                ((0, 0), (0, l_stride - n_qh)),
            ).reshape(-1)
            swa_m_flat = jnp.pad(
                swa_m_qh.reshape(num_groups, n_qh),
                ((0, 0), (0, l_stride - n_qh)),
            ).reshape(-1)
            sinks_flat = jnp.pad(
                jnp.tile(attention_sinks, (b_q,)), (0, l_stride - n_qh)
            )
        lanes = gathered_nope.shape[-1]
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 7
        bq_buf = pltpu.VMEM((2, b_q, num_q_heads, head_dim), q.dtype)
        bl_buf = pltpu.VMEM((2, l_stride), jnp.float32)
        scratch_shapes = [
            pltpu.VMEM((2, m_union * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, m_union // 2, lanes), jnp.bfloat16),
            bq_buf,
            bq_buf,
            bl_buf,
            bl_buf,
            bq_buf,
            pltpu.VMEM((2, m_union // 128, 128), jnp.int32),
            pltpu.SemaphoreType.DMA((8, 2)),
        ]
        scalar_prefetches = (
            n_unique,
            jnp.array([start_group, num_active_groups, num_valid_tokens], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        scope_name = f"SparseMLA-Grouped-bq_{b_q}-mu_{m_union}"
        q = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _grouped_attention_kernel,
                    sm_scale=sm_scale,
                    b_q=b_q,
                    m_union=m_union,
                    l_stride=l_stride,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=hbm,
                    grid=(num_run_groups,),
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={5: 0},
                name=scope_name,
            )
        )(
            *scalar_prefetches,
            sinks_flat,
            q,
            gathered_nope,
            gathered_rope,
            cand_bits_tiled,
            swa_accumution,
            swa_l_flat,
            swa_m_flat,
        )
    else:
        swa_accumution, swa_l, swa_m = prepare_swa_inputs(
            swa_accumution, swa_l, swa_m
        )
        q = _run_chunked_fallback(
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            num_valid_tokens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            sm_scale=sm_scale,
            gather_and_attention_chunk_size=gather_and_attention_chunk_size,
            attention_kernel_batch_size=attention_kernel_batch_size,
        )

    return prepare_outputs(q, actual_num_q_heads, actual_head_dim)
