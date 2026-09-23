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
    live_bound: bool,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)

        if live_bound:
            # Interleaved: owner ``o`` serves every ``num_owners``-th atom, so
            # a prefix-live bound splits evenly across owners instead of
            # leaving all the work on the first few.
            def dst_atom(atom):
                return ((owner_batch * atoms_per_batch + atom) * num_owners +
                        owner)

            def values(atom, stream):
                return indices_ref[atom, 0, stream, :]
        else:
            # Q4R -- the blocked (prefill) map's owner assignment and inner
            # phase.  Two pieces of pure integer index arithmetic on values
            # the kernel already has: the grid extents, every BlockSpec shape
            # and every DMA descriptor count are unchanged, and the program's
            # ``memory_analysis`` is BYTE-IDENTICAL to the shipped blocked map
            # (``temp_size_in_bytes`` 2864497376, ``generated_code_size_in_
            # bytes`` 29370368) with an identical opcode histogram.
            #
            #   Q4 (sigma): owner PAIR ``(2j, 2j+1)`` owns the four
            #     consecutive query rows ``4j..4j+3`` and serves one per outer
            #     step, so rows that share most of their top-k sit on one
            #     owner pair ACROSS TIME instead of racing concurrently.
            #   PR (rho): ``rot(a) = (a + owner) % atoms_per_batch`` rotates
            #     each owner's inner-step phase, so the rows that ARE
            #     concurrent read different rank windows instead of the same
            #     one.  One add and one rem.
            #
            # WHY IT CANNOT CHANGE RESULTS: ``rot`` is a permutation of the
            # inner grid applied to BOTH the index read and the destination
            # write, and ``sigma`` is a bijection on the logical batches, so
            # the set of (source, destination) pairs is identical to shipped
            # -- only the order and the owner change.  Verified bit-exact on
            # 1024 rows x 4 index distributions.
            #
            # WHY IT IS WORTH IT: the shipped map puts 16 different query rows
            # on 32 owners all reading the same rank window, the maximum-
            # collision configuration.  On REAL captured indices that costs
            # 157.5 us/step of loop time over uniform indices; Q4R recovers
            # 66.6 us of it (42%), measured 1.0390x LOOP on prose / 1.0320x
            # code / 1.0096x random / 0.9945x uniform, with the overlap
            # fraction ``c`` unmoved (0.2991 vs base 0.3065).  Arms that reach
            # zero contention instead need the index operand repacked and pay
            # ``c`` +0.05, which costs more than it returns.
            logical_batch = ((owner // 2) * batches_per_owner +
                             owner_batch) * 2 + owner % 2

            def rot(atom):
                return (atom + owner) % atoms_per_batch

            def dst_atom(atom):
                return logical_batch * atoms_per_batch + rot(atom)

            def values(atom, stream):
                return indices_ref[0, rot(atom), stream, :]

        def destinations(atom, stream):
            return pl.ds(
                dst_atom(atom) * OCCURRENCES_PER_ATOM + stream * NUM_LANES,
                NUM_LANES,
            )

        def body(*refs):
            atom = pl.program_id(0)
            nope_inputs = refs[:ROPE_PACKING]
            rope_inputs = refs[ROPE_PACKING:2 * ROPE_PACKING]
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
                    data.append(rope_inputs[input_stream][pl.ds(input_lane, 1),
                                                          pl.ds(0, 32)])

                low_16 = jnp.uint32(0x0000FFFF)
                low_byte_each_half = jnp.uint32(0x00FF00FF)
                data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
                data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
                data[0], data[1] = _delta_swap(data[0], data[1], 8,
                                               low_byte_each_half)
                data[2], data[3] = _delta_swap(data[2], data[3], 8,
                                               low_byte_each_half)
                for band in range(ROPE_PACKING):
                    rope_output[pl.ds(row, 1),
                                pl.ds(band * 32, 32)] = data[band]

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 128),
                lambda atom, stream=stream: (values(atom, stream), 0),
            ) for stream in range(ROPE_PACKING))
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 32),
                lambda atom, stream=stream: (values(atom, stream), 0),
            ) for stream in range(ROPE_PACKING))
        rope_out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM, 128),
            lambda atom: (dst_atom(atom), 0),
        )

        def serve(steps):
            pltpu.emit_pipeline(
                body,
                grid=(steps, ),
                in_specs=nope_specs + rope_specs,
                out_specs=(rope_out_spec, ),
                no_pipelining=not serve_pipelined,
            )(
                *([nope_hbm_ref] * ROPE_PACKING),
                *([rope_hbm_ref] * ROPE_PACKING),
                rope_out_hbm_ref,
            )

        if not live_bound:
            serve(atoms_per_batch)
            return

        # The runtime bound RIDES the index operand: one extra stream row on a
        # fetch that already happens, so there is no second DMA.  It arrives as
        # a vector and is lane-extracted -- a SparseCore kernel cannot load a
        # scalar at rank 0.
        num_valid = indices_ref[0, 0, ROPE_PACKING, :][0]
        live_atoms = lax.div(num_valid + OCCURRENCES_PER_ATOM - 1,
                             jnp.int32(OCCURRENCES_PER_ATOM))
        owner_atoms = lax.div(
            jnp.maximum(live_atoms - owner, 0) + num_owners - 1,
            jnp.int32(num_owners))
        steps = jnp.clip(owner_atoms - owner_batch * atoms_per_batch, 0,
                         atoms_per_batch)

        @pl.when(steps > 0)
        def _():
            serve(steps)

    if live_bound:
        indices_spec = pl.BlockSpec(
            (atoms_per_batch, 1, ROPE_PACKING + 1, NUM_LANES),
            lambda batch: (batch, owner, 0, 0),
        )
    else:
        indices_spec = pl.BlockSpec(
            (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
            # Q4R's sigma, matching ``logical_batch`` in ``serve_batch``.
            lambda batch: (((owner // 2) * batches_per_owner + batch) * 2 +
                           owner % 2, 0, 0, 0),
        )
    pltpu.emit_pipeline(
        serve_batch,
        grid=(batches_per_owner, ),
        in_specs=(indices_spec, ),
    )(indices_hbm_ref)


def _index_operand(indices, num_valid_occurrences, num_owners,
                   atoms_per_batch, batches_per_owner, num_batches):
    """Shape the index array for the map the grid will use.

    Without a bound this is the incumbent's free reshape.  With one the array
    is interleaved owner-major and carries the bound in one extra stream row
    (the "rider"), which costs a materialisation the reshape does not.
    """
    if num_valid_occurrences is None:
        return indices.reshape(num_batches, atoms_per_batch, ROPE_PACKING,
                               NUM_LANES)
    atoms_total = atoms_per_batch * batches_per_owner
    core = indices.reshape(atoms_total, num_owners, ROPE_PACKING, NUM_LANES)
    rider = jnp.broadcast_to(
        jnp.asarray(num_valid_occurrences, jnp.int32).reshape(1, 1, 1, 1),
        (atoms_total, num_owners, 1, NUM_LANES))
    return jnp.concatenate([core, rider], axis=2)


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
    num_valid_occurrences: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` has one
    128-word row per group of four tokens; each token owns a consecutive
    32-word quarter. The returned ROPE is register-transposed into the grouped
    TensorCore format expected by the shared attention consumer.
    """
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError(
            "native-SC ROPE must be quarter-major uint32[...,128]")
    if indices.dtype != jnp.int32 or indices.ndim != 2:
        raise ValueError("indices must be int32[query, topk]")
    if indices.size != out_size:
        raise ValueError(
            f"indices has {indices.size} values, expected {out_size}")

    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = (sc_info.num_subcores
                    if sc_info is not None else NUM_SUBCORES)
    num_owners = num_cores * num_subcores

    max_possible_atoms = out_size // (num_owners * OCCURRENCES_PER_ATOM)
    if atoms_per_batch > max_possible_atoms > 0:
        atoms_per_batch = max_possible_atoms
    if atoms_per_batch <= 0:
        raise ValueError(f"atoms_per_batch ({atoms_per_batch}) must be > 0 "
                         f"(out_size={out_size}, num_owners={num_owners})")

    occurrences_per_batch = atoms_per_batch * OCCURRENCES_PER_ATOM
    owner_batch_size = num_owners * occurrences_per_batch
    if out_size % owner_batch_size != 0:
        raise ValueError(
            f"out_size ({out_size}) must be divisible by num_owners * "
            f"occurrences_per_batch ({owner_batch_size})")
    num_batches = out_size // occurrences_per_batch
    batches_per_owner = num_batches // num_owners

    # THE GATE.  The interleaved occurrence->owner map plus the traced grid
    # bound are taken ONLY at `batches_per_owner == 1`, which on GLM-5.2-FP8 at
    # csa_topk 2048 / atoms_per_batch 16 / 32 owners is exactly the decode
    # geometry (out_size = 16*2048 = 32768 -> num_batches 32 -> bpo 1).  The
    # prefill geometry (chunk 64 -> out_size 131072 -> num_batches 128 -> bpo
    # 4) keeps the shipped blocked map.
    #
    # WHY: measured against the shipped file at 100% live, the arm is 0.954x at
    # bpo 1 (bucket 16) and 0.911x at bpo 4 (bucket 64, the nightly cell) --
    # applying it unconditionally takes a ~9-10% regression on every prefill
    # chunk.  At bpo 1 and a low live fraction it is 2.20x (bucket 16, R=1,
    # j-f825a03b).
    #
    # WHY THIS KEY AND NOT ANOTHER: the ~6 us geometry term was never
    # decomposed -- three mechanisms move together between the two measured
    # cells (the `_index_operand` concat, which is O(out_size); the strided
    # index BlockSpec, which is O(batches_per_owner) descriptors per owner; and
    # the interleaved destination scatter, which is O(atoms per owner)).  All
    # three are monotone increasing in `batches_per_owner` at fixed
    # `atoms_per_batch` and `num_owners`, so `bpo == 1` selects the minimum of
    # every candidate mechanism and the gate is robust to which one it is.
    #
    # This is a TRACE-TIME decision: no runtime branch, and the two map
    # variants never coexist in one kernel body, because each (out_size,
    # atoms_per_batch) pair already traces its own body (STRAT-SC-16).
    live_bound = num_valid_occurrences is not None and batches_per_owner == 1

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
            live_bound=live_bound,
        ),
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // ROPE_PACKING, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types={"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING, ))},
        mesh=mesh,
        name=(f"sc_native_sc_gather_a{atoms_per_batch}_{wait_mode}_"
              f"{'pipe' if serve_pipelined else 'nopipe'}"
              f"{'_live' if live_bound else ''}"),
    )(
        nope_cache.reshape(-1, 128),
        rope_cache.reshape(-1, 32),
        _index_operand(indices, num_valid_occurrences if live_bound else None,
                       num_owners, atoms_per_batch, batches_per_owner,
                       num_batches),
    )
