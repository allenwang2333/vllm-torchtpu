# Copyright 2025 Google LLC
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
"""A progress signal the host can read while a forward pass is still running.

Today the only thing that tells the host a prefill has produced KV is the pass
finishing. Every layer's cache is then shipped at once, so the network sits
idle for the length of the forward and then has to move the whole payload.
A doorbell breaks that into fifteen smaller signals: each full-attention layer
stores a counter into HBM right after it writes its slice of the cache, so
progress becomes visible while the pass is still running.

This module is only the writing half, and deliberately so. Nothing acts on the
value yet, because the way this idea fails is cost rather than correctness:
the store sits in the compiled graph, where it may serialize work XLA would
otherwise overlap. That is a throughput question and it wants answering on its
own, against a baseline, before anything is built on top.

The reader is not here and will not be a Python thread. It belongs in raiden's
C++ transfer loop, which is where the decision it feeds already lives and
where there is no GIL to hold. A Python stand-in would price the wrong thing.
What this module does offer is `DoorbellSet.maybe_verify`, a handful of reads
during warmup, so that a ring which silently does nothing is distinguishable
from one that works.

All fifteen counters live in one `(groups, 8, 128)` int32 buffer rather than
fifteen separate ones, so the eventual reader pulls every group's state in a
single transfer. That shape costs nothing: (8, 128) is one native tile, the
smallest thing a Pallas store can address, and fifteen of them is 60 KiB.

The value is a step counter, not a group index. The buffer outlives the
request that filled it, and an index would leave the previous request's marks
in place, which any reader would take for progress that has not happened yet.
A counter that only ever rises cannot be misread that way.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# One native TPU tile. A Pallas store cannot address less.
BELL_SUBLANES = 8
BELL_LANES = 128

# Rings start at 1 so that a freshly zeroed buffer is distinguishable from a
# buffer whose first pass has already rung.
FIRST_GENERATION = 1


def _bell_kernel(sel_ref, gen_ref, bell_in_ref, out_ref):
    """Store the generation into the selected group, preserving the others.

    `bell_in_ref` is aliased onto `out_ref`, so this writes the caller's
    buffer instead of a fresh one. Aliasing donates the memory; it does not
    make unwritten elements keep their old contents, so the groups this ring
    does not own have to be copied through explicitly.

    Choosing the group with a mask rather than an index is deliberate. A
    dynamic index into a Pallas ref has to travel through SMEM as a scalar,
    and selecting across a 60 KiB buffer is cheaper than that plumbing.
    """
    out_ref[...] = jnp.where(sel_ref[...] != 0, gen_ref[...], bell_in_ref[...])


def _make_bell_fn(num_groups: int):
    bell_call = pl.pallas_call(
        _bell_kernel,
        out_shape=jax.ShapeDtypeStruct((num_groups, BELL_SUBLANES, BELL_LANES),
                                       jnp.int32),
        input_output_aliases={2: 0},
    )

    # The annotations are load-bearing: `pallas.jax_op` reads the signature to
    # build the torch op schema and rejects a bare parameter.
    def bell_fn(announced: jax.Array, doorbells: jax.Array,
                selector: jax.Array, generation: jax.Array) -> jax.Array:
        """Ring one group's bell, ordered after `announced` is written.

        The ordering rests on two things. `announced` is a parameter of the
        exported function, so at the call site the ring is a consumer of the
        KV cache the attention kernel just produced and XLA cannot schedule it
        first. `optimization_barrier` keeps that edge alive through inlining,
        since ordinary dead-code elimination will not drop a barrier operand
        even when its result goes unused.

        `selector` and `generation` arrive already broadcast to the buffer's
        shape so the kernel needs no scalar operands.
        """
        announced, generation = jax.lax.optimization_barrier(
            (announced, generation))
        del announced
        return bell_call(selector, generation, doorbells)

    return bell_fn


def _make_sharded_bell_fn(num_groups: int, mesh, announced_spec):
    """Wrap the ring in `shard_map` for the meshes PCP attention runs on.

    Mosaic kernels are not automatically partitioned, so under a mesh the call
    has to say what each rank sees. Only `announced` is sharded, because it is
    the KV cache. The doorbell buffer, the selector and the generation are
    replicated, so every rank rings its own copy of the same value.
    """
    from jax.sharding import PartitionSpec as P

    inner = _make_bell_fn(num_groups)
    replicated = P()

    return jax.shard_map(
        inner,
        mesh=mesh,
        in_specs=(announced_spec, replicated, replicated, replicated),
        out_specs=replicated,
        check_vma=False,
    )


# Each `pallas.jax_op` call registers into the global torch op table under a
# name that has to be unique, so ops are counted rather than keyed.
_op_counter = 0


class DoorbellSet:
    """The buffer, the layer-to-group map, and the step counter.

    One instance per worker, built in `initialize_kv_cache` once the set of
    attention layers is known and held on the model wrapper context so
    attention forward can reach it.
    """

    def __init__(self,
                 layer_names: Sequence[str],
                 device: torch.device,
                 verify_steps: int = 8):
        self._group_of = {name: i for i, name in enumerate(layer_names)}
        self.num_groups = len(layer_names)
        self._verify_budget = verify_steps

        # The bells themselves, and one pre-built one-hot selector per group.
        # Building the selectors here rather than in the ring keeps an
        # `arange` and a comparison out of every layer's graph.
        self.bells = torch.zeros((self.num_groups, BELL_SUBLANES, BELL_LANES),
                                 dtype=torch.int32,
                                 device=device)
        eye = torch.eye(self.num_groups, dtype=torch.int32)
        self.selectors = eye[:, :, None,
                             None].expand(self.num_groups, self.num_groups,
                                          BELL_SUBLANES,
                                          BELL_LANES).contiguous().to(device)

        # A device tensor, not a Python int, so that bumping it between steps
        # does not retrace the graph. Shaped like the buffer because the
        # kernel selects elementwise and has no scalar operands.
        self.generation = torch.full(
            (self.num_groups, BELL_SUBLANES, BELL_LANES),
            FIRST_GENERATION - 1,
            dtype=torch.int32,
            device=device)

        # Set by `prebuild`. Held here rather than in a module-level cache
        # because under PCP the mesh comes from `get_pcp_streaming_mesh()`,
        # which need not return the same object twice, so there is nothing
        # stable to key a cache on.
        self.op = None

    def prebuild(self, mesh, announced_spec) -> None:
        """Register the ring's custom op before `torch.compile` traces.

        This cannot be deferred to the first ring. `pallas.jax_op` reads the
        wrapped function's signature with `eval`, which dynamo refuses to
        trace. Under `fullgraph=True` that raises outright. Without it dynamo
        takes a graph break and runs the ring eagerly, and an eager in-place
        write on torch_tpu rebinds the tensor to a fresh buffer, leaving the
        reader watching an address nothing will be written to again.
        """
        global _op_counter
        if self.op is not None:
            return

        if mesh is None:
            fn = _make_bell_fn(self.num_groups)
            specs = None
        else:
            from jax.sharding import PartitionSpec
            fn = _make_sharded_bell_fn(self.num_groups, mesh, announced_spec)
            replicated = PartitionSpec()
            specs = [announced_spec, replicated, replicated, replicated]

        op = pallas.jax_op(
            f"pallas::kv_doorbell_{_op_counter}",
            fn,
            # Donating the doorbell buffer is what makes the store land in
            # the memory the reader watches. Without it the op returns a
            # fresh buffer and the writeback lands somewhere nobody reads.
            donate_argnums=(1, ),
            mesh=mesh,
            input_partition_specs=specs,
        )
        op.register_fake(lambda announced, doorbells, selector, generation:
                         torch.empty_like(doorbells))
        _op_counter += 1
        self.op = op

    def group_for(self, layer_name: str) -> Optional[int]:
        return self._group_of.get(layer_name)

    def next_generation(self) -> int:
        """Advance the counter. Host-side, between steps, never under trace."""
        self.generation += 1
        return int(self.generation[0, 0, 0])

    def read(self) -> list[int]:
        """One value per group, as of now. Costs one 60 KiB device read."""
        return self.bells[:, 0, 0].cpu().tolist()

    def maybe_verify(self) -> None:
        """Log the counters for the first few steps, then stop reading.

        Without this the buffer is write-only and a ring that silently does
        nothing looks exactly like a ring that works. A handful of reads is
        enough to tell those apart.

        It is bounded rather than continuous because reading forces a device
        sync, and a sync every step would show up in the very throughput
        number this stage exists to measure. After the budget runs out the
        check costs one integer comparison.
        """
        if self._verify_budget <= 0:
            return
        self._verify_budget -= 1
        logger.info("KV doorbell after step %d: %s",
                    int(self.generation[0, 0, 0]), self.read())


def ring_for(bells: DoorbellSet, layer_name: str,
             announced: torch.Tensor) -> None:
    """Ring `layer_name`'s bell, ordered after `announced` is written.

    A no-op for a layer with no group, which is every layer on a worker that
    is not the KV source. `announced` should be the tensor the attention
    kernel just wrote, since that dependency is the whole ordering guarantee.
    """
    group = bells.group_for(layer_name)
    if group is None or bells.op is None:
        return

    rung = bells.op(announced, bells.bells, bells.selectors[group],
                    bells.generation)
    # The op returns the donated buffer, but torch_tpu hands back a new tensor
    # object bound to the same memory. Copying it back keeps `bells.bells`
    # pointing at the storage the eventual reader watches.
    bells.bells.copy_(rung)
