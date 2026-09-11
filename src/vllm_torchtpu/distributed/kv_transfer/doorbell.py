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

Nothing acts on the value yet, and deliberately so. The way this idea fails is
cost rather than correctness: the store sits in the compiled graph, where it
may serialize work XLA would otherwise overlap. That is a throughput question
and it wants answering on its own, against a baseline, before anything is
built on top.

The reading is done by raiden's `DoorbellReader`, a C++ thread that copies the
buffer to the host on a fixed cadence and records when each counter first rose.
It is in C++ rather than Python because that is where the decision it will
eventually feed already lives, and because a Python thread would price the GIL
rather than the poll. `DoorbellWatcher` below owns that reader and turns its
sightings into the one number this stage exists to produce: the gap, within a
single pass, between the first group's bell and the last. If that gap is
milliseconds, shipping early layers early wins them back. If the bells all
land together, there is nothing to overlap and the feature is not worth
building.

`DoorbellSet.maybe_verify` is a separate and cruder check, a handful of direct
reads during warmup, so that a ring which silently does nothing is
distinguishable from one that works.

All fifteen counters live in one `(groups, 8, 128)` int32 buffer rather than
fifteen separate ones, so the eventual reader pulls every group's state in a
single transfer. That shape costs nothing: (8, 128) is one native tile, the
smallest thing a Pallas store can address, and fifteen of them is 60 KiB.

The buffer is sharded per chip because the thing it reports is per chip. Under
PCP each chip holds a different slice of the same layer's KV, and a chip can
only honestly say that its own slice is written. A replicated bell would force
the chips to agree on a value none of them can vouch for, which needs a
collective on a path whose whole purpose is to be cheap.

Sharding does not change the buffer, only the spec. A torch tensor handed to a
Pallas op is one chip's shard, and `jax_placeholder` multiplies it up to the
global shape, so the tensor stays fifteen tiles and the eight chips together
describe a `(8 * 15, 8, 128)` array. Everything below sees the local view, and
so does the reader: a worker polls its own sixty kilobytes and speaks only for
its own chip.

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
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# One native TPU tile. A Pallas store cannot address less.
BELL_SUBLANES = 8
BELL_LANES = 128

# Rings start at 1 so that a freshly zeroed buffer is distinguishable from a
# buffer whose first pass has already rung.
FIRST_GENERATION = 1


def _bell_kernel(group_ref, gen_ref, ann_ref, bell_in_ref, out_ref):
    """Store the generation into one group's tile.

    The output block is exactly the tile named by `group_ref`, and it is fully
    written, so the other fourteen survive through the HBM alias without ever
    being read. That is why the kernel body is a single store: there is no
    read-modify-write and no mask.

    Three of the four refs go untouched. `group_ref` is consumed by the output
    block's index map rather than by the body. `ann_ref` is the attention
    output and exists only so the call has it as an operand; see `bell_fn` for
    why that is what orders the store. `bell_in_ref` exists only to give
    `input_output_aliases` something to alias. Both sit in HBM, so declaring
    them costs no transfer.
    """
    del group_ref, ann_ref, bell_in_ref
    out_ref[...] = gen_ref[...]


def _make_bell_fn(num_groups: int, interpret: bool = False):
    """`interpret` is for the tests, which have to run without a TPU."""
    block = (1, BELL_SUBLANES, BELL_LANES)
    bell_call = pl.pallas_call(
        _bell_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            # The group index has to reach the index map, and an index map can
            # only read scalars that were prefetched into SMEM. It cannot come
            # from the doorbell buffer or from a VMEM operand.
            num_scalar_prefetch=1,
            grid=(1, ),
            in_specs=[
                pl.BlockSpec(block, lambda i, s: (0, 0, 0)),
                # The attention output. Never read, so it never leaves HBM.
                pl.BlockSpec(memory_space=pltpu.HBM),
                # Aliased with the output. Never read, so it stays in HBM too.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(block, lambda i, s: (s[0], 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((num_groups, BELL_SUBLANES, BELL_LANES),
                                       jnp.int32),
        input_output_aliases={3: 0},
        interpret=interpret,
    )

    # The annotations are load-bearing: `pallas.jax_op` reads the signature to
    # build the torch op schema and rejects a bare parameter.
    def bell_fn(
        announced: jax.Array, doorbells: jax.Array, group: jax.Array,
        generation: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Ring one group's bell, pinned between `announced` and its readers.

        `announced` is both an operand of the kernel and a result of the call,
        and the two roles fence the ring from opposite sides. A schedule is a
        topological order of the dataflow graph, so an instruction can only be
        held in place by the things it reads and the things that read it, and
        the bell needs one of each.

        The operand supplies the predecessor. It is passed to the kernel
        unread, purely to be an operand, which keeps the ring downstream of the
        layer whose KV it reports. This used to be an `optimization_barrier`
        over `(announced, generation)` whose `announced` result was discarded,
        on the theory that a barrier operand survives dead-code elimination. It
        does not: with the result unused XLA drops the operand and the edge
        with it, and every bell was then scheduled at the top of the program,
        ahead of the layers it claimed to announce. An operand of a Pallas call
        cannot be dropped that way, because XLA cannot see inside the custom
        call to prove it unused.

        Returning `announced` supplies the successor, which the operand alone
        does not. With the ring's only reader being the program's own return of
        the doorbell buffer, XLA is free to place the whole chain of them
        anywhere after their attention kernels, and it sinks them to the end to
        shorten that buffer's live range. On the 35B under the production
        config all ten bells landed within twenty instructions of each other at
        the tail of a 54790 instruction schedule, while their attention kernels
        were spread over the preceding five thousand. Barriering `announced`
        against `rung` gives the caller a value that cannot exist until the
        ring has run; the caller feeds it to the rest of the layer, so every
        instruction after this attention is now after this bell.
        """
        rung = bell_call(group, generation, announced, doorbells)
        return jax.lax.optimization_barrier((rung, announced))

    return bell_fn


def _make_sharded_bell_fn(num_groups: int, mesh, announced_spec, bells_spec):
    """Wrap the ring in `shard_map` for the meshes PCP attention runs on.

    Mosaic kernels are not automatically partitioned, so under a mesh the call
    has to say what each rank sees. The bells carry `bells_spec`, the same
    sharding the KV cache has, because a bell is a claim about one chip's KV
    and only that chip can make it. The group index and the generation are
    replicated, since which group is ringing and which pass it is are the same
    everywhere.
    """
    from jax.sharding import PartitionSpec as P

    inner = _make_bell_fn(num_groups)
    replicated = P()

    sharded = jax.shard_map(
        inner,
        mesh=mesh,
        in_specs=(announced_spec, bells_spec, replicated, replicated),
        # Two results now: the rung buffer, and `announced` passed back through
        # the barrier so the ring has a consumer. The second carries the same
        # sharding it came in with, since the barrier is an identity.
        out_specs=(bells_spec, announced_spec),
        check_vma=False,
    )

    # The annotations have to be on the function `prebuild` hands over, not
    # merely on `inner`. `pcp_streaming_jax_op` reads the signature with
    # `follow_wrapped=False`, and `shard_map` presents a bare
    # `(*args, **kwargs)`, which it rejects.
    def sharded_bell_fn(
        announced: jax.Array, doorbells: jax.Array, group: jax.Array,
        generation: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return sharded(announced, doorbells, group, generation)

    return sharded_bell_fn


def shards_for(mesh, announced_spec) -> int:
    """How many chips the KV cache is split across, per the mesh itself.

    The caller could compute this from the parallel config, but then two
    numbers would have to stay in step and a mismatch would surface as an
    incomprehensible shape error deep inside `shard_map`. Asking the mesh
    leaves one source of truth.
    """
    if mesh is None or announced_spec is None or not announced_spec:
        return 1
    axis = announced_spec[0]
    return 1 if axis is None else int(mesh.shape[axis])


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
                 verify_steps: int = 0):
        self._group_of = {name: i for i, name in enumerate(layer_names)}
        self.num_groups = len(layer_names)
        self._verify_budget = verify_steps

        # This chip's fifteen counters. A bell says "this chip's slice of this
        # layer's KV is written", which is a claim only that chip is in a
        # position to make: under PCP each chip holds a different slice and
        # they need not finish together. The buffer is the same size sharded
        # or not, because it is already the local view.
        self.bells = torch.zeros((self.num_groups, BELL_SUBLANES, BELL_LANES),
                                 dtype=torch.int32,
                                 device=device)

        # Which group each ring writes, one pre-built `(1,)` tensor per group.
        # The kernel's index map reads this out of SMEM, so it has to arrive as
        # its own operand rather than as a Python int, which would bake a
        # different graph per layer.
        self.group_ids = torch.arange(self.num_groups,
                                      dtype=torch.int32,
                                      device=device).reshape(
                                          self.num_groups, 1)

        # A device tensor, not a Python int, so that bumping it between steps
        # does not retrace the graph. One tile, because that is the whole of
        # what a ring writes.
        self.generation = torch.full((1, BELL_SUBLANES, BELL_LANES),
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

        name = f"pallas::kv_doorbell_{_op_counter}"
        # Donating the doorbell buffer is what makes the store land in the
        # memory the reader watches. Without it the op returns a fresh buffer
        # and the writeback lands somewhere nobody reads.
        donated = (1, )

        if mesh is None:
            op = pallas.jax_op(name,
                               _make_bell_fn(self.num_groups),
                               donate_argnums=donated)
        else:
            from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.\
                vllm_adapter import pcp_streaming_jax_op
            from jax.sharding import PartitionSpec
            replicated = PartitionSpec()
            # A worker without PCP still has a mesh, but attention hands back
            # no spec for it, and `NamedSharding` will not accept None.
            # Replicated is what "on this mesh, unsharded" means.
            if announced_spec is None:
                announced_spec = replicated
            # The bells follow the KV cache onto whatever axis it is sharded
            # on, read off `announced_spec` rather than named here, so that
            # the two cannot drift apart if the attention mesh changes.
            bells_spec = PartitionSpec(
                announced_spec[0] if announced_spec else None)
            fn = _make_sharded_bell_fn(self.num_groups, mesh, announced_spec,
                                       bells_spec)
            # Not `pallas.jax_op`, despite the PCP name on this one. Stock
            # `jax_op` reads output shapes off `lowered.out_avals`, and
            # `jax.export` records every `shard_map` output there as
            # replicated, so a sharded output comes back at its global shape.
            # That is survivable while the only output is the donated bell
            # buffer, which the writeback reshapes anyway, but the barrier's
            # `announced` goes straight into the next layer and arrived eight
            # times too long under PCP8. This wrapper carries
            # `output_partition_specs` through to the allocation instead.
            op = pcp_streaming_jax_op(
                name,
                fn,
                donate_argnums=donated,
                mesh=mesh,
                input_partition_specs=[
                    announced_spec, bells_spec, replicated, replicated
                ],
                output_partition_specs=[bells_spec, announced_spec],
            )

        # Both wrappers ship a fake kernel and neither will do. Each derives
        # the output shape from the lowered JAX avals, which are concrete, and
        # the token dimension of `announced` is symbolic: dynamo has it tied to
        # `input_ids.size(0)` and rejects the graph the moment a constant shows
        # up in its place. `empty_like` copies the input's shape whatever it is
        # made of, so the SymInt survives. It also gives the local shard rather
        # than the global array, which is what the caller is holding.
        op.register_fake(lambda announced, doorbells, group, generation:
                         (torch.empty_like(doorbells),
                          torch.empty_like(announced)))
        _op_counter += 1
        self.op = op

    def group_for(self, layer_name: str) -> Optional[int]:
        return self._group_of.get(layer_name)

    def next_generation(self) -> int:
        """Advance the counter. Host-side, between steps, never under trace."""
        self.generation += 1
        return int(self.generation[0, 0, 0])

    def read(self) -> list[int]:
        """This chip's value per group. Costs one 60 KiB device read.

        Every element of a group's tile holds the same counter, so element
        (0, 0) stands for the whole tile.
        """
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

        Bounded is still not free. The first eight reads land inside the first
        benchmark leg, which is why that leg alone showed a cost the others
        did not, so the budget is zero unless TPU_KV_DOORBELL_VERIFY asks for
        it.
        """
        if self._verify_budget <= 0:
            return
        self._verify_budget -= 1
        # This worker's chip only. Chips need not agree, and comparing them is
        # a matter of lining up eight workers' logs.
        logger.info("KV doorbell after step %d: %s",
                    int(self.generation[0, 0, 0]), self.read())


def _make_reader(bells: torch.Tensor):
    """Build raiden's C++ reader over the bells buffer.

    Its own function so that the watcher's bookkeeping can be tested with a
    stand-in, which needs neither a TPU nor a wheel that ships the reader.
    """
    from tpu_sync.api.torch import doorbell as raiden_doorbell
    return raiden_doorbell.reader(bells)


class DoorbellWatcher:
    """Owns raiden's reader and reports the stagger it observes.

    One per worker, alive for as long as the model is. It starts a C++ polling
    thread at construction and stops it at `close`, and in between it does
    nothing except summarize, every so often, what the thread has seen.
    """

    def __init__(self, bells: DoorbellSet, poll_interval_ns: int,
                 report_every: int):
        self._bells = bells
        self._report_every = report_every
        self._steps = 0
        self._pending: dict[int, dict[int, int]] = {}
        # Every pass whose groups all rang, retired out of `_pending` as soon
        # as it completed. One float per pass, so a long run costs nothing.
        self._spans_ms: list[float] = []
        self._zero_spans = 0
        self._newest_offsets: list[float] | None = None
        # Set once the reader has reported anything at all. Until it does, a
        # step whose direct read shows rung bells means the reader is watching
        # the wrong memory, which is worth saying out loud exactly once.
        self._saw_anything = False
        self._warned_stale = False

        self._reader = _make_reader(bells.bells)
        self._reader.start(poll_interval_ns)
        logger.info("KV doorbell reader polling %d groups every %.0f us",
                    self._reader.num_groups(), poll_interval_ns / 1000.0)

    def after_step(self) -> None:
        """Called once per forward pass, from the host, between steps."""
        self._steps += 1
        if self._steps % self._report_every:
            return
        self._collect()
        self._report()

    def _collect(self) -> None:
        for sighting in self._reader.drain_sightings():
            self._saw_anything = True
            gen = self._pending.setdefault(sighting.generation, {})
            gen[sighting.group] = sighting.host_nanos
        # A generation is only interesting once every group has rung. Retire
        # every complete one into the running sample before pruning. Pruning
        # first would leave exactly one pass behind, which is how an earlier
        # run reported a median over a single sample.
        complete = sorted(g for g, seen in self._pending.items()
                          if len(seen) == self._bells.num_groups)
        for gen in complete:
            seen = self._pending[gen]
            first = min(seen.values())
            span_ms = (max(seen.values()) - first) / 1e6
            self._spans_ms.append(span_ms)
            if span_ms == 0.0:
                self._zero_spans += 1
            self._newest_offsets = [(seen[g] - first) / 1e6
                                    for g in range(self._bells.num_groups)]
        # Generations arrive in order, so anything not newer than the newest
        # retired one will never fill in and is dropped rather than kept.
        if complete:
            self._pending = {
                g: seen
                for g, seen in self._pending.items() if g > complete[-1]
            }

    def _report(self) -> None:
        stats = self._reader.stats()
        if not self._saw_anything:
            self._report_nothing_seen(stats)
            return

        mean_poll_us = (stats.poll_nanos_total / stats.polls /
                        1000.0) if stats.polls else 0.0
        logger.info(
            "KV doorbell reader after %d steps: %d polls, mean %.0f us, "
            "max %.0f us, %d failures, %d torn tiles, %d sightings, "
            "%d dropped", self._steps, stats.polls, mean_poll_us,
            stats.poll_nanos_max / 1000.0, stats.poll_failures,
            stats.torn_tiles, stats.sightings_recorded,
            stats.sightings_dropped)

        if not self._spans_ms:
            logger.info(
                "KV doorbell stagger: no pass yet had all %d groups sighted, "
                "so the poll is slower than the pass or some group never "
                "rings", self._bells.num_groups)
            return

        spans_ms = sorted(self._spans_ms)
        # Passes whose span is exactly zero had all their groups first seen in
        # one poll. That is the difference between a pass with no stagger and
        # a stagger too small for the poll interval to resolve, and without it
        # a median of zero is ambiguous.
        logger.info(
            "KV doorbell stagger over %d complete passes: median %.2f ms, "
            "max %.2f ms, %d passes within one poll", len(spans_ms),
            spans_ms[len(spans_ms) // 2], spans_ms[-1], self._zero_spans)
        # The shape matters as much as the span. A stagger that rises steadily
        # across groups is real pipelining; one where fourteen groups land
        # together and one lags is a straggler and not an opportunity.
        logger.info("KV doorbell per-group offsets (ms): %s",
                    ", ".join(f"{off:.2f}"
                              for off in self._newest_offsets))

    def _report_nothing_seen(self, stats) -> None:
        """Distinguish a reader watching dead memory from bells that never ring.

        The ring donates its buffer and writes back with `copy_`, and whether
        that keeps the storage the reader resolved at construction is the one
        thing about this path that cannot be settled by reading code. If the
        storage moved, the reader polls a buffer nobody writes, sees zeros
        forever, and looks exactly like a model that never rang. A direct read
        through torch tells the two apart, so pay for one here rather than
        spend a job guessing.
        """
        if self._warned_stale:
            return
        self._warned_stale = True
        direct = self._bells.read()
        if any(direct):
            logger.error(
                "KV doorbell reader saw nothing in %d polls but a direct read "
                "shows %s. The ring's writeback moved the buffer and the "
                "reader is watching stale memory.", stats.polls, direct)
        else:
            logger.warning(
                "KV doorbell reader saw nothing in %d polls and a direct read "
                "agrees the bells are unrung.", stats.polls)

    def close(self) -> None:
        self._reader.stop()
        self._collect()
        self._report()


def ring_for(bells: DoorbellSet, layer_name: str,
             announced: torch.Tensor) -> torch.Tensor:
    """Ring `layer_name`'s bell and return the value the layer should use.

    The return is `announced`, but it is not the tensor that came in: it has
    been through the barrier in `bell_fn` and so depends on the ring. Callers
    have to use it in place of the original, or the ring keeps no consumer and
    XLA sinks it to the end of the program. For a layer with no group, which is
    every layer on a worker that is not the KV source, the original comes
    straight back.

    Two conditions have to hold, and only one of them is about `announced`.

    `announced` must be a value the attention kernel *returned*, not the cache
    it wrote. The cache is written in place through donation, so it enters the
    traced graph as a program input, and an operand edge to a program input
    orders nothing: the value exists before the first instruction, so the bell
    is free to run first. The kernel's output carries the same guarantee
    without that hole, since it cannot exist until the call that wrote the
    cache has run.

    The call must also be traced. `install` exists because the obvious place to
    ring, inside the attention implementation, is not: vLLM reaches that
    implementation through `torch.ops.vllm.unified_attention_with_output`, and
    a custom op's body is opaque to dynamo, so anything in it runs at execution
    time instead of becoming graph instructions. On a lazy backend fifteen such
    rings pile up and flush together in one small program, which is a flat pass
    however carefully their operands are chosen.
    """
    group = bells.group_for(layer_name)
    if group is None or bells.op is None:
        return announced

    rung, announced = bells.op(announced, bells.bells,
                               bells.group_ids[group], bells.generation)
    # The op returns the donated buffer, but torch_tpu hands back a new tensor
    # object bound to the same memory. Copying it back keeps `bells.bells`
    # pointing at the storage the eventual reader watches.
    bells.bells.copy_(rung)
    return announced


def install(bells: DoorbellSet, layers: dict) -> int:
    """Ring after each attention layer by wrapping its `forward`.

    `layers` maps layer name to the vLLM `Attention` module. Only the ones with
    a group are wrapped, so a worker that is not the KV source is untouched.

    Wrapping the module rather than editing the implementation is what puts the
    store in the model's own XLA program: `Attention.forward` is ordinary traced
    Python, and only the custom op it calls is opaque. The tensor that forward
    returns is the one that op wrote, so it is both a real graph value and the
    right thing to depend on.

    The wrapper returns what `ring_for` hands back rather than what attention
    returned, and that substitution is the point of the whole file. The output
    aliasing chains the bells to each other, so their relative order is safe,
    but a chain with nothing downstream of it is free to slide as a block and
    XLA slides it to the end. Feeding the layer the post-ring value nails the
    block to the layers it reports on. Returns how many layers were wrapped.
    """
    wrapped = 0
    for name, layer in layers.items():
        if bells.group_for(name) is None:
            continue

        def ringing(original, name):
            def forward(*args, **kwargs):
                out = original(*args, **kwargs)
                return ring_for(bells, name, out)

            return forward

        # Assigning to the instance shadows the class method for this layer
        # only, which matters because the same `Attention` class serves layers
        # that have no group.
        layer.forward = ringing(layer.forward, name)
        wrapped += 1
    return wrapped
