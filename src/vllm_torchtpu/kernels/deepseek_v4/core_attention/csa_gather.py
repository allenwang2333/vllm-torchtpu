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

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    valid_indices_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    valid_vmem_ref: Any,
    idx_chunk_ref: Any,
    active_steps_ref: Any,
    nope_sem: Any,
    idx_sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
    top_k: int,
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    step_rows = num_streams * row_subchunk_size
    steps_per_period = num_row_subchunks // num_streams
    steps_per_half = top_k // 2 // step_rows
    num_periods = indices_hbm_ref.shape[0] // top_k
    base_periods = num_periods // num_cores
    rem_periods = num_periods % num_cores
    max_chunk_periods = idx_chunk_ref.shape[0] // top_k
    max_core_periods = base_periods + (1 if rem_periods > 0 else 0)
    num_chunks = pl.cdiv(max_core_periods, max_chunk_periods)

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))
    my_start_period = core_index * base_periods + jnp.minimum(core_index, rem_periods)
    my_count_periods = base_periods + jnp.where(core_index < rem_periods, 1, 0)

    nope_in_cols = nope_in_hbm_ref.shape[1]
    rope_in_cols = rope_in_hbm_ref.shape[1]
    # A `bf16[n, 64]` output is tiled to 128 lanes, so XLA pads it and it
    # costs twice its own bytes -- on this kernel's write and on the
    # consumer's read. The output is 128 lanes wide instead, pairing entry `i`
    # of each `top_k`-entry period with entry `i + top_k // 2`, which the
    # consumer splits apart with a lane slice and a row concatenate.
    rope_out_cols = rope_out_hbm_ref.shape[1] // 2
    rope_rows_per_stream = row_subchunk_size // 2
    valid_dma = pltpu.make_async_copy(valid_indices_ref, valid_vmem_ref, idx_sem)
    valid_dma.start()
    valid_dma.wait()
    valid_indices = valid_vmem_ref[pl.ds(0, row_subchunk_size)][0]

    def process_rope(gather_ref, out_ref, out_row_base=0):
        # bf16 output rows 2t and 2t + 1 are the low and high halves of int32
        # row t, so each output word pairs one rope value of an even entry
        # with the same value of the next entry. A cache rope word k holds
        # values k (low) and k + 32 (high).
        for t in range(num_simd_lanes // 2):
            even = gather_ref[pl.ds(2 * t, 1), :]
            odd = gather_ref[pl.ds(2 * t + 1, 1), :]
            row = pl.ds(out_row_base + t, 1)
            out_ref[row, pl.ds(0, rope_in_cols)] = jnp.bitwise_or(
                jnp.bitwise_and(even, 0xFFFF), jnp.left_shift(odd, 16)
            )
            out_ref[row, pl.ds(rope_in_cols, rope_in_cols)] = jnp.bitwise_or(
                lax.shift_right_logical(even, 16),
                jnp.bitwise_and(odd, jnp.int32(-65536)),  # 0xFFFF0000
            )

    my_chunks = pl.cdiv(my_count_periods, max_chunk_periods)

    def _chunk_body(chunk_idx, _):
        chunk_p_offset = chunk_idx * max_chunk_periods
        cur_periods = jnp.clip(my_count_periods - chunk_p_offset, 0, max_chunk_periods)
        chunk_start_p = my_start_period + chunk_p_offset
        dma_start_p = jnp.minimum(chunk_start_p, num_periods - max_chunk_periods)
        p_shift = chunk_start_p - dma_start_p
        base_step = dma_start_p * steps_per_period

        idx_dma = pltpu.make_async_copy(
            indices_hbm_ref.at[pl.ds(dma_start_p * top_k, max_chunk_periods * top_k)],
            idx_chunk_ref,
            idx_sem,
        )
        idx_dma.start()
        idx_dma.wait()

        def _compact_period(lp, n_act):
            g_row = (chunk_start_p + lp) * top_k
            first_idx = idx_chunk_ref[pl.ds((p_shift + lp) * top_k, row_subchunk_size)][0]
            is_act = (g_row < valid_indices) & (first_idx >= 0)
            base_s_rel = (p_shift + lp) * steps_per_period
            for s_in_p in range(steps_per_period):
                active_steps_ref[n_act] = base_s_rel + s_in_p
                n_act = n_act + jnp.where(is_act, 1, 0)
            return n_act

        n_active = lax.fori_loop(0, cur_periods, _compact_period, jnp.int32(0))

        def idx_window(r, s):
            s_in_chunk = active_steps_ref[r]
            return idx_chunk_ref[
                pl.ds(s_in_chunk * step_rows + s * row_subchunk_size, row_subchunk_size)
            ]

        def rope_out_block(r):
            s_in_chunk = active_steps_ref[r]
            half, step = divmod(base_step + s_in_chunk, steps_per_half)
            return (half // 2) * steps_per_half + step, half % 2

        def _body(*refs):
            r = pl.program_id(0)
            s_in_chunk = active_steps_ref[r]
            out_step = base_step + s_in_chunk
            nope_g = refs[0 * num_streams : 1 * num_streams]
            rope_g = refs[1 * num_streams : 2 * num_streams]
            rope_o = refs[2 * num_streams]

            nope_copies = []
            for s in range(num_streams):
                out_row = out_step * step_rows + s * row_subchunk_size
                copy = pltpu.make_async_copy(
                    nope_g[s],
                    nope_out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
                    nope_sem.at[s],
                )
                copy.start()
                nope_copies.append(copy)

            for s in range(num_streams):
                process_rope(
                    gather_ref=rope_g[s],
                    out_ref=rope_o,
                    out_row_base=s * rope_rows_per_stream,
                )

            for copy in nope_copies:
                copy.wait()

        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )
        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_cols), rope_out_block
        )

        @pl.when(n_active > 0)
        def _run_gather():
            pltpu.emit_pipeline(
                _body,
                grid=(n_active,),
                in_specs=nope_in_specs + rope_in_specs,
                out_specs=(rope_out_spec,),
            )(
                *([nope_in_hbm_ref] * num_streams),
                *([rope_in_hbm_ref] * num_streams),
                rope_out_hbm_ref,
            )

        return None

    lax.fori_loop(0, my_chunks, _chunk_body, None)



@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

    Args:
      nope_cache: (total_pages, page_size, 128) int32, one row per token: the
        (4, 128) uint8 nope slab (see `csa_cache_layout`).
      rope_cache: (total_pages, page_size // 4, 128) int32, 32 words per token
        (see `csa_cache_layout`).
      indices: (N,) int32. Token indices into the caches; N a multiple of
        top_k.
      num_valid_indices: Optional (1,) or scalar int32. Number of valid indices to
        gather, a multiple of top_k. A subcore skips every top_k period that
        starts at or past this count, leaving those output rows unwritten.
      top_k: the consumer's row block (the attention kernel's top-k), a
        multiple of 128. `rope_out` pairs entry i of a period with entry
        i + top_k // 2.

    Returns:
      Both outputs are int32 with 128 lanes, which XLA lays out byte-for-byte
      like the uint8 / bf16 arrays they encode, so the consumer views them with
      a free `ref.bitcast`.
      nope_out: (N, 128) int32. Row i is entry i's (4, 128) uint8 nope slab.
      rope_out: (N // 4, 128) int32, i.e. (N // 2, 128) bf16 once bitcast.
        For i < top_k // 2, bf16 row `period * (top_k // 2) + i` holds entry
        i of that period in lanes 0:64 and entry i + top_k // 2 in lanes
        64:128 -- 128 lanes so XLA does not pad the buffer to twice its size.
        The consumer restores a period's (top_k, 64) from its
        (top_k // 2, 128) block `blk` with
        `jnp.concatenate([blk[:, :64], blk[:, 64:]], axis=0)`.
    """
    assert indices.ndim == 1, "Indices must be 1D."
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.int32, "Caches must be int32."
    assert nope_cache.shape[2] == csa_cache_layout.ROW_WORDS
    assert rope_cache.shape[2] == csa_cache_layout.ROW_WORDS

    # Free reshapes: a 128-lane int32 array's T(8,128) tiling is byte-identical
    # to row-major, and the kernel takes its operands untiled
    # (`use_tc_tiling_on_sc=False`), so token t's rope is words
    # [32 t, 32 t + 32).
    nope_cache = nope_cache.reshape(-1, csa_cache_layout.ROW_WORDS)
    rope_cache = rope_cache.reshape(-1, csa_cache_layout.ROPE_WORDS)
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    row_subchunk_size = num_simd_lanes

    if num_valid_indices is None:
        valid_indices = jnp.full((row_subchunk_size,), out_size, dtype=jnp.int32)
    else:
        valid_indices = jnp.full(
            (row_subchunk_size,), num_valid_indices, dtype=jnp.int32
        )

    # `num_streams` independent `pl.Indirect` gathers are issued per
    # pipeline step to keep multiple gather DMAs in flight.
    # See `outer_pipeline` for details.
    num_streams = min(4, max(1, (top_k // 2) // row_subchunk_size))
    assert out_size % top_k == 0
    num_row_subchunks = top_k // row_subchunk_size
    steps_per_period = num_row_subchunks // num_streams
    num_periods = out_size // top_k
    max_core_periods = (num_periods + num_cores - 1) // num_cores
    max_chunk_periods = min(
        16, max(1, num_simd_lanes // steps_per_period), max_core_periods
    )
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return pl.kernel(
        functools.partial(
            main_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
            top_k=top_k,
        ),
        out_type=(
            jax.ShapeDtypeStruct(
                (out_size, csa_cache_layout.ROW_WORDS), jnp.int32
            ),
            jax.ShapeDtypeStruct(
                (out_size // 4, csa_cache_layout.ROW_WORDS),
                jnp.int32,
            ),
        ),
        scratch_types=(
            pltpu.VMEM((num_simd_lanes,), jnp.int32),
            pltpu.VMEM((max_chunk_periods * top_k,), jnp.int32),
            pltpu.SMEM((num_simd_lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA((num_streams,)),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name="sc_csa_gather",
    )(nope_cache, rope_cache, indices, valid_indices)
