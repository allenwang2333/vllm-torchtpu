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
"""Combined native-SparseCore NOPE/ROPE gather for GLM-5.2."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

NUM_CORES = 2
NUM_SUBCORES = 16
NUM_LANES = 16
ROPE_PACKING = 4
QUADS_PER_ATOM = 16
OCCURRENCES_PER_ATOM = QUADS_PER_ATOM * ROPE_PACKING
# A token's ROPE record, in uint32 words. The PADDED cache
# (`kv_cache_utils.align_to(rope_dim=64, TILE_LANE_BYTES=128)`) reserves 128 B
# per token, of which only the first 64 B carry data. The PACKED cache stores
# the 64 B, i.e. 16 words, so the gather reads and writes 576 B/token instead
# of 640. The layout is selected by the caches' size ratio (NOPE is 4x the
# PADDED ROPE cache and 8x the PACKED one), so both layouts keep working.
ROPE_WORDS_PADDED = 32
ROPE_WORDS_PACKED = 16
# PACKED: two ROPE_PACKING-groups (8 tokens) share one 128-lane output row.
# An atom fills ONE 64-lane half of its 16 output rows rather than both halves
# of 8, which is what makes the attention's two-slice/one-row-concat unpack
# land in NATURAL token order with no remap.
ROPE_SUBROWS_PACKED = 2


def rope_words_for_caches(nope_cache: jax.Array, rope_cache: jax.Array) -> int:
    """ROPE words per token implied by the caches' size ratio."""
    ratio, rem = divmod(nope_cache.size, rope_cache.size)
    if rem == 0 and ratio == 4:
        return ROPE_WORDS_PADDED
    if rem == 0 and ratio == 8:
        return ROPE_WORDS_PACKED
    raise ValueError(
        f"NOPE cache size ({nope_cache.size}) must be 4x (padded 128 B ROPE) "
        f"or 8x (packed 64 B ROPE) the ROPE cache size ({rope_cache.size})"
    )


def _delta_swap(x, y, shift, mask):
    tmp = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(x, shift), y), mask)
    return (
        jnp.bitwise_xor(x, jnp.left_shift(tmp, shift)),
        jnp.bitwise_xor(y, tmp),
    )


def _kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    wait_mode: str,
    serve_pipelined: bool,
    rope_words: int,
    rope_atoms_per_row: int,
):
    packed = rope_words == ROPE_WORDS_PACKED
    rope_half = rope_atoms_per_row // 2
    owner = lax.axis_index((core_axis_name, subcore_axis_name))

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners
        occurrence_base = logical_batch * atoms_per_batch * OCCURRENCES_PER_ATOM

        def values(atom, stream):
            return indices_ref[0, atom, stream, :]

        def destinations(atom, stream):
            return pl.ds(
                occurrence_base + atom * OCCURRENCES_PER_ATOM + stream * NUM_LANES,
                NUM_LANES,
            )

        def body(*refs):
            atom = pl.program_id(0)
            nope_inputs = refs[:ROPE_PACKING]
            rope_inputs = refs[ROPE_PACKING : 2 * ROPE_PACKING]
            rope_output = refs[-1]

            nope_copies = []
            for stream in range(ROPE_PACKING):
                copy = pltpu.make_async_copy(
                    nope_inputs[stream],
                    nope_out_hbm_ref.at[destinations(atom, stream), :],
                    sem_ref.at[stream],
                )
                copy.start()
                nope_copies.append(copy)
            if wait_mode == "before_rope":
                for copy in nope_copies:
                    copy.wait()

            for row in range(QUADS_PER_ATOM):
                data = []
                for token_slot in range(ROPE_PACKING):
                    logical_rank = row * ROPE_PACKING + token_slot
                    input_stream = logical_rank // NUM_LANES
                    input_lane = logical_rank % NUM_LANES
                    data.append(
                        rope_inputs[input_stream][
                            pl.ds(input_lane, 1), pl.ds(0, rope_words)
                        ]
                    )

                low_16 = jnp.uint32(0x0000FFFF)
                low_byte_each_half = jnp.uint32(0x00FF00FF)
                data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
                data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
                data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
                data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
                for band in range(ROPE_PACKING):
                    rope_output[
                        pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                    ] = data[band]

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 128),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), rope_words),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        if packed:

            def rope_out_blocks(atom):
                # Atoms 0..apr/2-1 of a query row take the LOW 64-lane half
                # of their 16 rows, atoms apr/2..apr-1 the HIGH half.
                # HAZARD: at rope_atoms_per_row < 2 this computes `g % 0`,
                # whose block indices are undefined and which wedged a shard
                # for 40 minutes (j-ca91a6d1). `dsa_gather_native_sc` refuses
                # that shape in Python before anything is dispatched.
                g = logical_batch * atoms_per_batch + atom
                return (
                    g // rope_atoms_per_row * rope_half + g % rope_half,
                    (g % rope_atoms_per_row) // rope_half,
                )

            rope_out_spec = pl.BlockSpec(
                (QUADS_PER_ATOM, rope_words * ROPE_PACKING), rope_out_blocks
            )
        else:
            rope_out_spec = pl.BlockSpec(
                (QUADS_PER_ATOM, 128),
                lambda atom: (logical_batch * atoms_per_batch + atom, 0),
            )
        pltpu.emit_pipeline(
            body,
            grid=(atoms_per_batch,),
            in_specs=nope_specs + rope_specs,
            out_specs=(rope_out_spec,),
            no_pipelining=not serve_pipelined,
        )(
            *([nope_hbm_ref] * ROPE_PACKING),
            *([rope_hbm_ref] * ROPE_PACKING),
            rope_out_hbm_ref,
        )

    indices_spec = pl.BlockSpec(
        (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
        lambda batch: (owner + batch * num_owners, 0, 0, 0),
    )
    pltpu.emit_pipeline(
        serve_batch,
        grid=(batches_per_owner,),
        in_specs=(indices_spec,),
    )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
    ),
)
def dsa_gather_native_sc(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    *,
    out_size: int,
    atoms_per_batch: int = 16,
    wait_mode: str = "overlap_rope",
    serve_pipelined: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` is either
    PADDED -- one 128-word row per group of four tokens, each token owning a
    consecutive 32-word quarter -- or PACKED -- one 128-word row per group of
    eight tokens, each owning a consecutive 16-word (64 B) eighth. The layout
    is read off the caches' size ratio. The returned ROPE is
    register-transposed into the grouped TensorCore format expected by the
    shared attention consumer; in the PACKED layout each atom writes one
    64-lane half so that the consumer's two-slice unpack lands in natural
    token order.
    """
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError("native-SC ROPE must be quarter-major uint32[...,128]")
    if indices.dtype != jnp.int32 or indices.ndim != 2:
        raise ValueError("indices must be int32[query, topk]")
    if indices.size != out_size:
        raise ValueError(f"indices has {indices.size} values, expected {out_size}")
    rope_words = rope_words_for_caches(nope_cache, rope_cache)
    packed = rope_words == ROPE_WORDS_PACKED
    if packed and indices.shape[1] < 2 * OCCURRENCES_PER_ATOM:
        # SAFETY. With fewer than two atoms per query row `rope_half` is 0 and
        # the half-split index map computes `g % 0`; that wedged a shard for
        # 40 minutes at topk=64. Refuse it here, before anything is dispatched.
        raise ValueError(
            f"the packed half-split ROPE layout needs topk >= "
            f"{2 * OCCURRENCES_PER_ATOM}, got {indices.shape[1]}"
        )

    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = sc_info.num_subcores if sc_info is not None else NUM_SUBCORES
    num_owners = num_cores * num_subcores

    max_possible_atoms = out_size // (num_owners * OCCURRENCES_PER_ATOM)
    if atoms_per_batch > max_possible_atoms > 0:
        atoms_per_batch = max_possible_atoms
    if atoms_per_batch <= 0:
        raise ValueError(
            f"atoms_per_batch ({atoms_per_batch}) must be > 0 "
            f"(out_size={out_size}, num_owners={num_owners})"
        )

    occurrences_per_batch = atoms_per_batch * OCCURRENCES_PER_ATOM
    owner_batch_size = num_owners * occurrences_per_batch
    if out_size % owner_batch_size != 0:
        raise ValueError(
            f"out_size ({out_size}) must be divisible by num_owners * "
            f"occurrences_per_batch ({owner_batch_size})"
        )
    num_batches = out_size // occurrences_per_batch
    batches_per_owner = num_batches // num_owners
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return core_map_helper.kernel(
        functools.partial(
            _kernel,
            core_axis_name=mesh.core_axis_name,
            subcore_axis_name=mesh.subcore_axis_name,
            num_owners=num_owners,
            atoms_per_batch=atoms_per_batch,
            batches_per_owner=batches_per_owner,
            wait_mode=wait_mode,
            serve_pipelined=serve_pipelined,
            rope_words=rope_words,
            rope_atoms_per_row=indices.shape[1] // OCCURRENCES_PER_ATOM,
        ),
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct(
                (out_size * rope_words // 128, 128), jnp.uint32
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types={"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,))},
        mesh=mesh,
        name=(
            f"sc_native_sc_gather{'_r16' if packed else ''}"
            f"_a{atoms_per_batch}_{wait_mode}_"
            f"{'pipe' if serve_pipelined else 'nopipe'}"
        ),
    )(
        nope_cache.reshape(-1, 128),
        rope_cache.reshape(-1, rope_words),
        indices.reshape(num_batches, atoms_per_batch, ROPE_PACKING, NUM_LANES),
    )
