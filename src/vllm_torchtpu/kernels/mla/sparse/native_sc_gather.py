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
"""Combined native-SparseCore NOPE/ROPE gather for GLM-5.2.

An optional traced live-row count (``num_live_indices``) bounds the outer
emit_pipeline grid so only the logical batches holding live rows are gathered;
rows beyond are left unwritten (the attention grid never reads them). Without
it every row is gathered.
"""

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
    live_vmem_ref: Any = None,
    live_sem_ref: Any = None,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    wait_mode: str,
    serve_pipelined: bool,
    live_hbm_ref: Any = None,
    guard_tail: bool = False,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    live_batches = None
    if live_hbm_ref is not None:
        # The live logical-batch count arrives as a broadcast i32[NUM_LANES]
        # vector. On SparseCore a scalar load is legal only from SMEM and SMEM
        # only at rank 0 (jax/_src/pallas/mosaic/sc_lowering.py), so the count is
        # DMA'd into a VMEM vector and lane 0 extracted -- the same shape as the
        # in-repo dsa_gather.py:90 (`valid_ref[pl.ds(0, 16)][0]`).
        cp = pltpu.make_async_copy(live_hbm_ref, live_vmem_ref, live_sem_ref.at[0])
        cp.start()
        cp.wait()
        # Lanes [0,16) broadcast the live logical-batch count, lanes [16,32)
        # the per-owner trip count; only lane 0 of a vector is ever extracted.
        live_batches = live_vmem_ref[pl.ds(0, NUM_LANES)][0]

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
                        rope_inputs[input_stream][pl.ds(input_lane, 1), pl.ds(0, 32)]
                    )

                low_16 = jnp.uint32(0x0000FFFF)
                low_byte_each_half = jnp.uint32(0x00FF00FF)
                data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
                data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
                data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
                data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
                for band in range(ROPE_PACKING):
                    rope_output[pl.ds(row, 1), pl.ds(band * 32, 32)] = data[band]

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
                (pl.Indirect(NUM_LANES), 32),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        rope_out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM, 128),
            lambda atom: (logical_batch * atoms_per_batch + atom, 0),
        )

        def run_inner():
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

        if live_batches is not None and guard_tail:
            # Owners share a uniform traced grid of cdiv(live, num_owners);
            # the ragged last round is skipped per owner.
            pl.when(logical_batch < live_batches)(run_inner)
        else:
            run_inner()

    indices_spec = pl.BlockSpec(
        (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
        lambda batch: (owner + batch * num_owners, 0, 0, 0),
    )
    if live_batches is None:
        pltpu.emit_pipeline(
            serve_batch,
            grid=(batches_per_owner,),
            in_specs=(indices_spec,),
        )(indices_hbm_ref)
    else:
        # Traced grid length for emit_pipeline on a VectorSubcoreMesh, and
        # pl.when around it for the zero-trip case -- both lower on v7x SC.
        # No upstream SparseCore test traces an emit_pipeline grid (the
        # tpu_sparsecore_pallas_test.py grids are static); the nearest
        # upstream form, tests/pallas/tpu_pallas_pipeline_test.py:2420-2452
        # @ce57049, is a TensorCore test. This SC form is measured in-repo.
        # Uniform per-owner trip count; no scalar division on SC: the wrapper
        # sends live_batches as a multiple-of-num_owners-rounded pair.
        owner_batches = live_vmem_ref[pl.ds(NUM_LANES, NUM_LANES)][0]

        @pl.when(owner_batches > 0)
        def _serve_live():
            pltpu.emit_pipeline(
                serve_batch,
                grid=(owner_batches,),
                in_specs=(indices_spec,),
            )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
        "live_multiple",
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
    num_live_indices: jax.Array | None = None,
    live_multiple: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` has one
    128-word row per group of four tokens; each token owns a consecutive
    32-word quarter. The returned ROPE is register-transposed into the grouped
    TensorCore format expected by the shared attention consumer.

    ``num_live_indices`` (traced i32 scalar, optional) gathers only the logical
    batches covering the first ``num_live_indices`` rows; later rows are left
    unwritten. ``live_multiple`` is a static promise that ``num_live_indices``
    is a multiple of it; when that multiple covers whole owner rounds the
    per-owner tail guard is dropped.
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
    live = num_live_indices is not None
    scratch_types = {"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,))}
    operands = [
        nope_cache.reshape(-1, 128),
        rope_cache.reshape(-1, 32),
        indices.reshape(num_batches, atoms_per_batch, ROPE_PACKING, NUM_LANES),
    ]
    guard_tail = False
    if live:
        # Live logical batches = cdiv(num_live_indices, occurrences/batch),
        # clamped to the static count; per-owner trip = cdiv(live, num_owners).
        live_b = jnp.clip(
            (jnp.asarray(num_live_indices, jnp.int32) + occurrences_per_batch - 1)
            // occurrences_per_batch,
            0,
            num_batches,
        )
        owner_b = (live_b + num_owners - 1) // num_owners
        # Lanes [0,16) = live logical batches, [16,32) = per-owner grid length.
        live_vec = jnp.concatenate(
            [
                jnp.full((NUM_LANES,), live_b, jnp.int32),
                jnp.full((NUM_LANES,), owner_b, jnp.int32),
            ]
        )
        operands.append(live_vec)
        scratch_types["live_vmem_ref"] = pltpu.VMEM((2 * NUM_LANES,), jnp.int32)
        scratch_types["live_sem_ref"] = pltpu.SemaphoreType.DMA((1,))
        # Per-owner guard is needed only if a live count can leave a ragged
        # last round, i.e. unless the caller promises num_live_indices is a
        # multiple of num_owners * occurrences_per_batch (at topk=2048,
        # atoms=1: 32 * 64 = 2048 = one token, so decode needs no guard).
        guard_tail = live_multiple % owner_batch_size != 0
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    body = functools.partial(
        _kernel,
        core_axis_name=mesh.core_axis_name,
        subcore_axis_name=mesh.subcore_axis_name,
        num_owners=num_owners,
        atoms_per_batch=atoms_per_batch,
        batches_per_owner=batches_per_owner,
        wait_mode=wait_mode,
        serve_pipelined=serve_pipelined,
        guard_tail=guard_tail,
    )
    if live:
        inner = body

        def body(nope, rope, idx, live_hbm, nope_o, rope_o, **scr):
            return inner(nope, rope, idx, nope_o, rope_o, live_hbm_ref=live_hbm, **scr)

    return core_map_helper.kernel(
        body,
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // ROPE_PACKING, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=(
            f"sc_native_sc_gather_a{atoms_per_batch}_{wait_mode}_"
            f"{'pipe' if serve_pipelined else 'nopipe'}{'_live' if live else ''}"
        ),
    )(*operands)


def _serve_atom(
    nope_inputs, rope_inputs, rope_output, nope_out_hbm_ref, sem_ref, occurrence
):
    """One 64-row atom, exactly as `_kernel`'s body in `overlap_rope` mode.

    NOPE rows go straight to HBM at `occurrence + stream * 16`; the four
    tokens' ROPE quarters are register-transposed into `rope_output`.
    """
    copies = []
    for stream in range(ROPE_PACKING):
        copy = pltpu.make_async_copy(
            nope_inputs[stream],
            nope_out_hbm_ref.at[pl.ds(occurrence + stream * NUM_LANES, NUM_LANES), :],
            sem_ref.at[stream],
        )
        copy.start()
        copies.append(copy)
    for row in range(QUADS_PER_ATOM):
        data = []
        for token_slot in range(ROPE_PACKING):
            logical_rank = row * ROPE_PACKING + token_slot
            data.append(
                rope_inputs[logical_rank // NUM_LANES][
                    pl.ds(logical_rank % NUM_LANES, 1), pl.ds(0, 32)
                ]
            )
        low_16 = jnp.uint32(0x0000FFFF)
        low_byte_each_half = jnp.uint32(0x00FF00FF)
        data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
        data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
        data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
        data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
        for band in range(ROPE_PACKING):
            rope_output[pl.ds(row, 1), pl.ds(band * 32, 32)] = data[band]
    for copy in copies:
        copy.wait()


def _live_prefix_kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    prefix_indices_hbm_ref: Any,
    full_indices_hbm_ref: Any,
    live_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    live_vmem_ref: Any,
    live_sem_ref: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    tokens_per_batch: int,
    atoms_per_window: int,
    prefix_atoms: tuple[int, ...],
    full_atoms_per_batch: int,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    # Runtime scalars arrive as a lane vector (SparseCore: scalar loads only
    # from SMEM, SMEM only at rank 0): [prefix batches | prefix trip | n_c |
    # full batches | full trip], 16 lanes each, lane 0 extracted.
    cp = pltpu.make_async_copy(live_hbm_ref, live_vmem_ref, live_sem_ref.at[0])
    cp.start()
    cp.wait()

    def lane(i):
        return live_vmem_ref[pl.ds(i * NUM_LANES, NUM_LANES)][0]

    prefix_batches, prefix_trip, n_c = lane(0), lane(1), lane(2)
    full_batches, full_trip = lane(3), lane(4)

    def rope_spec_set(index_of):
        nope = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 128),
                lambda *g, s=s: (index_of(*g, s), 0),
            )
            for s in range(ROPE_PACKING)
        )
        rope = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 32),
                lambda *g, s=s: (index_of(*g, s), 0),
            )
            for s in range(ROPE_PACKING)
        )
        return nope + rope

    def make_prefix_serve(nc):
        # Owner batch = `tokens_per_batch` tokens x their first `nc` atoms of
        # window 0, written in place at the window-major position of each
        # atom (token * atoms_per_window + atom); later atoms stay unwritten.
        def serve(indices_ref):
            logical_batch = owner + pl.program_id(0) * num_owners
            token0 = logical_batch * tokens_per_batch

            def global_atom(tt, a):
                return (token0 + tt) * atoms_per_window + a

            def body(*refs):
                g = global_atom(pl.program_id(0), pl.program_id(1))
                _serve_atom(
                    refs[:ROPE_PACKING],
                    refs[ROPE_PACKING : 2 * ROPE_PACKING],
                    refs[-1],
                    nope_out_hbm_ref,
                    sem_ref,
                    g * OCCURRENCES_PER_ATOM,
                )

            in_specs = rope_spec_set(
                lambda tt, a, s: indices_ref[0, tt * atoms_per_window + a, s, :]
            )
            out_spec = pl.BlockSpec(
                (QUADS_PER_ATOM, 128), lambda tt, a: (global_atom(tt, a), 0)
            )

            @pl.when(logical_batch < prefix_batches)
            def _():
                pltpu.emit_pipeline(
                    body,
                    grid=(tokens_per_batch, nc),
                    in_specs=in_specs,
                    out_specs=(out_spec,),
                )(
                    *([nope_hbm_ref] * ROPE_PACKING),
                    *([rope_hbm_ref] * ROPE_PACKING),
                    rope_out_hbm_ref,
                )

        return serve

    def full_serve(indices_ref):
        # Today's linear traversal (every slot of the live windows).
        logical_batch = owner + pl.program_id(0) * num_owners

        def body(*refs):
            g = logical_batch * full_atoms_per_batch + pl.program_id(0)
            _serve_atom(
                refs[:ROPE_PACKING],
                refs[ROPE_PACKING : 2 * ROPE_PACKING],
                refs[-1],
                nope_out_hbm_ref,
                sem_ref,
                g * OCCURRENCES_PER_ATOM,
            )

        in_specs = rope_spec_set(lambda atom, s: indices_ref[0, atom, s, :])
        out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM, 128),
            lambda atom: (logical_batch * full_atoms_per_batch + atom, 0),
        )

        @pl.when(logical_batch < full_batches)
        def _():
            pltpu.emit_pipeline(
                body,
                grid=(full_atoms_per_batch,),
                in_specs=in_specs,
                out_specs=(out_spec,),
            )(
                *([nope_hbm_ref] * ROPE_PACKING),
                *([rope_hbm_ref] * ROPE_PACKING),
                rope_out_hbm_ref,
            )

    prefix_spec = pl.BlockSpec(
        (1, tokens_per_batch * atoms_per_window, ROPE_PACKING, NUM_LANES),
        lambda b: (owner + b * num_owners, 0, 0, 0),
    )
    full_spec = pl.BlockSpec(
        (1, full_atoms_per_batch, ROPE_PACKING, NUM_LANES),
        lambda b: (owner + b * num_owners, 0, 0, 0),
    )
    for nc in prefix_atoms:

        @pl.when(jnp.logical_and(n_c == nc, prefix_trip > 0))
        def _(nc=nc):
            pltpu.emit_pipeline(
                make_prefix_serve(nc), grid=(prefix_trip,), in_specs=(prefix_spec,)
            )(prefix_indices_hbm_ref)

    @pl.when(jnp.logical_and(n_c == 0, full_trip > 0))
    def _():
        pltpu.emit_pipeline(full_serve, grid=(full_trip,), in_specs=(full_spec,))(
            full_indices_hbm_ref
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "prefix_atoms",
        "tokens_per_batch",
        "full_atoms_per_batch",
    ),
)
def dsa_gather_native_sc_live_prefix(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    window_indices: jax.Array,
    n_c: jax.Array,
    live_tokens: jax.Array,
    num_live_full: jax.Array,
    *,
    chunk_size: int,
    prefix_atoms: tuple[int, ...],
    tokens_per_batch: int = 8,
    full_atoms_per_batch: int = 16,
) -> tuple[jax.Array, jax.Array]:
    """Window-major capped gather that moves only each token's live prefix.

    ``window_indices`` is the chunk's window-major ``int32[W * C, topk_cap]``
    (window 0 = rows ``[0, C)``), and the outputs have exactly the shapes and
    layout of ``dsa_gather_native_sc(window_indices, out_size=C * W *
    topk_cap)``. With ``n_c`` (traced) in ``prefix_atoms``, only atoms ``[0,
    n_c)`` (rows ``[0, 64 * n_c)``) of window 0 are gathered for the first
    ``live_tokens`` tokens, in place; every other row is left unwritten. With
    ``n_c == 0`` it falls back to today's linear traversal of the first
    ``num_live_full`` rows (the overflow case: windows >= 1). Gathered rows are
    bit-identical to ``dsa_gather_native_sc``'s.
    """
    num_rows, topk_cap = window_indices.shape
    if window_indices.dtype != jnp.int32 or topk_cap % OCCURRENCES_PER_ATOM:
        raise ValueError("window_indices must be int32[W*C, multiple of 64]")
    if num_rows % chunk_size or chunk_size % tokens_per_batch:
        raise ValueError(
            f"bad shapes: rows {num_rows}, chunk {chunk_size}, "
            f"tokens_per_batch {tokens_per_batch}"
        )
    atoms_per_window = topk_cap // OCCURRENCES_PER_ATOM
    if any(not 0 < a <= atoms_per_window for a in prefix_atoms):
        raise ValueError(f"prefix_atoms {prefix_atoms} outside (0, {atoms_per_window}]")
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = sc_info.num_subcores if sc_info is not None else NUM_SUBCORES
    num_owners = num_cores * num_subcores
    out_size = num_rows * topk_cap
    rows_per_full_batch = full_atoms_per_batch * OCCURRENCES_PER_ATOM
    if out_size % (num_owners * rows_per_full_batch):
        raise ValueError("out_size must divide into whole full-gather owner rounds")
    num_full_batches = out_size // rows_per_full_batch
    num_prefix_batches = chunk_size // tokens_per_batch

    def trip(batches):
        return (batches + num_owners - 1) // num_owners

    prefix_b = jnp.clip(
        (jnp.asarray(live_tokens, jnp.int32) + tokens_per_batch - 1)
        // tokens_per_batch,
        0,
        num_prefix_batches,
    )
    full_b = jnp.clip(
        (jnp.asarray(num_live_full, jnp.int32) + rows_per_full_batch - 1)
        // rows_per_full_batch,
        0,
        num_full_batches,
    )
    live_vec = jnp.concatenate(
        [
            jnp.full((NUM_LANES,), v, jnp.int32)
            for v in (prefix_b, trip(prefix_b), n_c, full_b, trip(full_b))
        ]
    )
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    body = functools.partial(
        _live_prefix_kernel,
        core_axis_name=mesh.core_axis_name,
        subcore_axis_name=mesh.subcore_axis_name,
        num_owners=num_owners,
        tokens_per_batch=tokens_per_batch,
        atoms_per_window=atoms_per_window,
        prefix_atoms=tuple(prefix_atoms),
        full_atoms_per_batch=full_atoms_per_batch,
    )
    return core_map_helper.kernel(
        body,
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // ROPE_PACKING, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types={
            "sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,)),
            "live_vmem_ref": pltpu.VMEM((5 * NUM_LANES,), jnp.int32),
            "live_sem_ref": pltpu.SemaphoreType.DMA((1,)),
        },
        mesh=mesh,
        name=f"sc_native_sc_gather_live_prefix_t{tokens_per_batch}",
    )(
        nope_cache.reshape(-1, 128),
        rope_cache.reshape(-1, 32),
        window_indices[:chunk_size].reshape(
            num_prefix_batches,
            tokens_per_batch * atoms_per_window,
            ROPE_PACKING,
            NUM_LANES,
        ),
        window_indices.reshape(
            num_full_batches, full_atoms_per_batch, ROPE_PACKING, NUM_LANES
        ),
        live_vec,
    )
