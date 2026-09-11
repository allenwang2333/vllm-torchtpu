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
    block's index map rather than by the body. `ann_ref` is the KV cache and
    exists only so the call has it as an operand; see `bell_fn` for why that
    is what orders the store. `bell_in_ref` exists only to give
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
                # The KV cache. Never read, so it never leaves HBM.
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
    def bell_fn(announced: jax.Array, doorbells: jax.Array, group: jax.Array,
                generation: jax.Array) -> jax.Array:
        """Ring one group's bell, ordered after `announced` is written.

        `announced` is passed to the kernel, unread, purely to be an operand
        of the call. That is what keeps the ring downstream of the layer whose
        KV it reports. This used to be an `optimization_barrier` over
        `(announced, generation)` whose `announced` result was discarded, on
        the theory that a barrier operand survives dead-code elimination. It
        does not: with the result unused XLA drops the operand and the edge
        with it, and every bell was then scheduled at the top of the program,
        ahead of the layers it claimed to announce. An operand of a Pallas
        call cannot be dropped that way, because XLA cannot see inside the
        custom call to prove it unused.
        """
        return bell_call(group, generation, announced, doorbells)

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

    return jax.shard_map(
        inner,
        mesh=mesh,
        in_specs=(announced_spec, bells_spec, replicated, replicated),
        out_specs=bells_spec,
        check_vma=False,
    )


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

        if mesh is None:
            fn = _make_bell_fn(self.num_groups)
            specs = None
        else:
            from jax.sharding import PartitionSpec
            # The bells follow the KV cache onto whatever axis it is sharded
            # on, read off `announced_spec` rather than named here, so that
            # the two cannot drift apart if the attention mesh changes.
            bells_spec = PartitionSpec(
                announced_spec[0] if announced_spec else None)
            fn = _make_sharded_bell_fn(self.num_groups, mesh, announced_spec,
                                       bells_spec)
            replicated = PartitionSpec()
            specs = [announced_spec, bells_spec, replicated, replicated]

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
        op.register_fake(lambda announced, doorbells, group, generation:
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
             announced: torch.Tensor) -> None:
    """Ring `layer_name`'s bell, ordered after `announced` is produced.

    A no-op for a layer with no group, which is every layer on a worker that
    is not the KV source.

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
        return

    rung = bells.op(announced, bells.bells, bells.group_ids[group],
                    bells.generation)
    # The op returns the donated buffer, but torch_tpu hands back a new tensor
    # object bound to the same memory. Copying it back keeps `bells.bells`
    # pointing at the storage the eventual reader watches.
    bells.bells.copy_(rung)


def install(bells: DoorbellSet, layers: dict) -> int:
    """Ring after each attention layer by wrapping its `forward`.

    `layers` maps layer name to the vLLM `Attention` module. Only the ones with
    a group are wrapped, so a worker that is not the KV source is untouched.

    Wrapping the module rather than editing the implementation is what puts the
    store in the model's own XLA program: `Attention.forward` is ordinary traced
    Python, and only the custom op it calls is opaque. The tensor that forward
    returns is the one that op wrote, so it is both a real graph value and the
    right thing to depend on.

    Nothing here orders the bells against each other, and nothing needs to. The
    output aliasing already chains them: each call takes the previous call's
    buffer as an operand, so the fifteen stores form a sequence XLA cannot
    reshuffle. Returns how many layers were wrapped.
    """
    wrapped = 0
    for name, layer in layers.items():
        if bells.group_for(name) is None:
            continue

        def ringing(original, name):
            def forward(*args, **kwargs):
                out = original(*args, **kwargs)
                ring_for(bells, name, out)
                return out

            return forward

        # Assigning to the instance shadows the class method for this layer
        # only, which matters because the same `Attention` class serves layers
        # that have no group.
        layer.forward = ringing(layer.forward, name)
        wrapped += 1
    return wrapped
