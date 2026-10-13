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
"""Left-pack the merged DCP top-k winners that one rank owns, on SparseCore.

The winners are rank-local already: only scores cross the all-gather, so the
indices never leave the rank that produced them.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

SCAN_UNROLL = 8
_PACK_COMPILER_PARAMS = pltpu.CompilerParams(
    disable_bounds_checks=True, needs_layout_passes=False
)


def _pack_body(
    vals_hbm,  # i32[padded_b * width]
    out_hbm,  # i32[padded_b * width]
    rowbuf_vmem,  # i32[width + lanes]
    *,
    b: int,
    width: int,
    num_waves: int,
    lanes: int,
    num_cores: int,
    num_subcores: int,
):
    flat = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    chunks = width // lanes
    cores_per_chip = num_cores * num_subcores
    neg_ones = jnp.full((lanes,), -1, jnp.int32)

    def pipeline_step(vals_vmem, out_vmem):
        row = pl.program_id(0) * cores_per_chip + flat

        def fill(i):
            rowbuf_vmem[pl.ds(i * lanes, lanes)] = neg_ones

        # Pad up front; the scan below only overwrites the kept prefix.
        plsc.parallel_loop(0, chunks, unroll=4)(fill)

        def emit(i, woff):
            v = vals_vmem[pl.ds(i * lanes, lanes)]
            keep = v >= 0
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)], v, mask=keep)
            return woff + plsc.all_reduce_population_count(keep)[0]

        # Carried `woff` makes this sequential: ascending order, and `woff <= width`.
        plsc.parallel_loop(
            0, jnp.where(row < b, chunks, 0), unroll=SCAN_UNROLL, carry=jnp.int32(0)
        )(emit)

        def copy_out(i):
            out_vmem[pl.ds(i * lanes, lanes)] = rowbuf_vmem[pl.ds(i * lanes, lanes)]

        plsc.parallel_loop(0, chunks, unroll=4)(copy_out)

    buf_count = 1 if num_waves == 1 else 2
    block_spec = pl.BlockSpec(
        (width,),
        lambda w: (w * cores_per_chip + flat,),
        pipeline_mode=pl.Buffered(buf_count),
    )
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_waves,),
        in_specs=(block_spec,),
        out_specs=block_spec,
    )(vals_hbm, out_hbm)


@functools.lru_cache(maxsize=16)
def _get_sc_mesh(mesh_cls, num_cores: int, num_subcores: int):
    return mesh_cls(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )


@functools.lru_cache(maxsize=64)
def _get_pack_kernel(
    kernel_fn_builder,
    mesh_cls,
    b: int,
    width: int,
    num_waves: int,
    padded_b: int,
    lanes: int,
    num_cores: int,
    num_subcores: int,
):
    mesh = _get_sc_mesh(mesh_cls, num_cores, num_subcores)
    raw_kernel = kernel_fn_builder(
        functools.partial(
            _pack_body,
            b=b,
            width=width,
            num_waves=num_waves,
            lanes=lanes,
            num_cores=num_cores,
            num_subcores=num_subcores,
        ),
        out_type=jax.ShapeDtypeStruct((padded_b * width,), jnp.int32),
        compiler_params=_PACK_COMPILER_PARAMS,
        scratch_types=(pltpu.VMEM((width + lanes,), jnp.int32),),
        mesh=mesh,
        name=f"sc_pack_nonneg_b{b}_w{width}",
    )

    @jax.jit
    def _compiled_pack(vals: jax.Array) -> jax.Array:
        # `-1`, not the default 0: 0 is a real position.
        if padded_b > b:
            vals = jnp.pad(vals, ((0, padded_b - b), (0, 0)), constant_values=-1)
        out = raw_kernel(vals.reshape(-1))
        out_2d = out.reshape(padded_b, width)
        return out_2d[:b] if padded_b > b else out_2d

    return _compiled_pack


def pack_nonnegative(vals: jax.Array) -> jax.Array:  # i32[num_rows, width]
    """Left-packs each row's non-negative entries into a `-1` padded prefix.

    Relative order is preserved, and a row of all non-negatives fills exactly
    `width`, so this never truncates.
    """
    b, width = vals.shape
    if vals.dtype != jnp.int32:
        raise ValueError(f"vals must be int32, got {vals.dtype}")
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    # Geometry off the chip: a row per subcore, a chunk per lane.
    lanes = sc.num_lanes
    rows_per_wave = sc.num_cores * sc.num_subcores
    if width % lanes != 0:
        raise ValueError(f"width ({width}) must be a multiple of {lanes}")

    num_waves = (b + rows_per_wave - 1) // rows_per_wave
    padded_b = num_waves * rows_per_wave
    return _get_pack_kernel(
        pl.kernel,
        plsc.VectorSubcoreMesh,
        b,
        width,
        num_waves,
        # `-1`, not the default 0: 0 is a real position.
        padded_b,
        lanes,
        sc.num_cores,
        sc.num_subcores,
    )(vals)


@functools.partial(jax.jit, static_argnames=("k",))
def _filter_owned_candidates(
    local_idxs: jax.Array,
    slots: jax.Array,
    k: int,
    dcp_rank: jax.Array | int,
) -> jax.Array:
    rel = slots - dcp_rank * k
    ours = (slots >= 0) & (rel >= 0) & (rel < k)
    safe_rel = jnp.where(ours, rel, 0)
    mine = jnp.take_along_axis(local_idxs, safe_rel, axis=1)
    return jnp.where(ours, mine, -1)


def resolve_owned_winners(
    local_idxs: jax.Array,  # i32[num_rows, k], this rank's rank-local candidates
    slots: jax.Array,  # i32[num_rows, k], flat columns of the merged score list
    k: int,
    dcp_rank: jax.Array | int,
) -> jax.Array:
    """Picks the merged winners this rank produced out of its own list, packed.

    The candidate all-gather is rank-major, so flat column `c` came from rank
    `c // k` at that rank's own slot `c % k`; both are functions of the chunk
    width alone, not of the order the chunks arrived in. Every surviving entry
    is rank-local already, because `local_idxs` never crossed the gather --
    which is also why this is the only place ownership is decided.
    """
    return pack_nonnegative(_filter_owned_candidates(local_idxs, slots, k, dcp_rank))
