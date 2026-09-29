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
"""Left-pack the merged DCP top-k winners one rank owns, on SparseCore.

One ``plsc.store_compressed`` scan replaces a scatter that TPU serializes.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata

LANES = 16
# Rows are dealt one per subcore, so a wave covers every subcore on the chip.
ROWS_PER_WAVE = 32
SCAN_UNROLL = 8


def _pack_body(
    merged_hbm,  # i32[padded_b * width]
    rank_hbm,  # i32[padded_b * LANES]
    out_hbm,  # i32[padded_b * width]
    rowbuf_vmem,  # i32[width + LANES]
    *,
    b: int,
    width: int,
    dcp_size: int,
    interleave_c: int,
    num_waves: int,
):
    num_subcores = 16
    num_sc_cores = 2
    flat = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    chunks = width // LANES

    def pipeline_step(merged_vmem, rank_vmem, out_vmem):
        row = pl.program_id(0) * (num_sc_cores * num_subcores) + flat
        rank = rank_vmem[pl.ds(0, LANES)][0]

        def fill(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

        # Pad up front; the scan below only overwrites the owned prefix.
        plsc.parallel_loop(0, chunks, unroll=4)(fill)

        def emit(i, woff):
            g = merged_vmem[pl.ds(i * LANES, LANES)]
            # `-1` needs its own test: floor-div of a negative can alias a real rank.
            owned = (g >= 0) & (
                metadata.cp_owner_rank(g, dcp_size, interleave_c) == rank
            )
            local = metadata.cp_global_to_local(
                jnp.maximum(g, 0), dcp_size, interleave_c
            )
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)], local, mask=owned)
            return woff + plsc.all_reduce_population_count(owned)[0]

        # `woff` is carried, so this is a sequential scan: ascending column order,
        # and `woff <= width` since at most every entry is owned, so no truncation.
        plsc.parallel_loop(
            0, jnp.where(row < b, chunks, 0), unroll=SCAN_UNROLL, carry=jnp.int32(0)
        )(emit)

        def copy_out(i):
            out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]

        plsc.parallel_loop(0, chunks, unroll=4)(copy_out)

    cores_per_chip = num_sc_cores * num_subcores
    buf_count = 1 if num_waves == 1 else 2
    in_specs = (
        pl.BlockSpec(
            (width,),
            lambda w: (w * cores_per_chip + flat,),
            pipeline_mode=pl.Buffered(buf_count),
        ),
        pl.BlockSpec(
            (LANES,),
            lambda w: (w * cores_per_chip + flat,),
            pipeline_mode=pl.Buffered(buf_count),
        ),
    )
    out_specs = pl.BlockSpec(
        (width,),
        lambda w: (w * cores_per_chip + flat,),
        pipeline_mode=pl.Buffered(buf_count),
    )
    pltpu.emit_pipeline(
        pipeline_step, grid=(num_waves,), in_specs=in_specs, out_specs=out_specs
    )(merged_hbm, rank_hbm, out_hbm)


def pack_owned_winners(
    merged: jax.Array,  # i32[num_rows, width]
    dcp_size: int,
    interleave_c: int,
    dcp_rank: jax.Array | int,
) -> jax.Array:
    """Compacts the winners `dcp_rank` owns into a prefix of rank-local indices.

    `cp_owner_rank` partitions `merged`, so the ranks' outputs partition the
    global top-k and a rank owns at most all `width` entries: never truncates.
    """
    b, width = merged.shape
    if merged.dtype != jnp.int32:
        raise ValueError(f"merged must be int32, got {merged.dtype}")
    if width % LANES != 0:
        raise ValueError(f"width ({width}) must be a multiple of {LANES}")
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    num_waves = (b + ROWS_PER_WAVE - 1) // ROWS_PER_WAVE
    padded_b = num_waves * ROWS_PER_WAVE
    if padded_b > b:
        # `-1`, not the default 0: 0 is a real position that rank 0 owns.
        merged = jnp.pad(merged, ((0, padded_b - b), (0, 0)), constant_values=-1)
    rank = jnp.full((padded_b, LANES), dcp_rank, dtype=jnp.int32)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = pl.kernel(
        functools.partial(
            _pack_body,
            b=b,
            width=width,
            dcp_size=dcp_size,
            interleave_c=interleave_c,
            num_waves=num_waves,
        ),
        out_type=jax.ShapeDtypeStruct((padded_b * width,), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=(pltpu.VMEM((width + LANES,), jnp.int32),),
        mesh=mesh,
        name=f"sc_pack_owned_b{b}_w{width}_dcp{dcp_size}",
    )(merged.reshape(-1), rank.reshape(-1))
    return out.reshape(padded_b, width)[:b]
