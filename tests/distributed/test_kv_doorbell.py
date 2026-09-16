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
"""The doorbell kernel's semantics, checked without a TPU.

All fifteen counters share one buffer, so the property that matters is that a
ring touches its own group and no other. Pallas interpret mode runs the kernel
as plain JAX, which settles that without competing for a chip. It says nothing
about whether Mosaic lowers the kernel or whether donation holds, both of
which only a device can answer.
"""

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.experimental import pallas as pl

from vllm_torchtpu.distributed.kv_transfer import doorbell as doorbell_module
from vllm_torchtpu.distributed.kv_transfer.doorbell import (BELL_LANES,
                                                            BELL_SUBLANES,
                                                            FIRST_GENERATION,
                                                            DoorbellSet,
                                                            _make_bell_fn)

NUM_GROUPS = 15
SHAPE = (NUM_GROUPS, BELL_SUBLANES, BELL_LANES)


def _ring(bells, group, generation):
    """One ring through the real `bell_fn`, interpreted so it needs no TPU.

    `announced` stands in for the attention output. The kernel never reads it,
    so its contents and shape do not matter; it is passed because the
    production call passes it, which is what orders the ring after the layer.

    `bell_fn` returns `(rung, announced)`, and only the rung buffer is of
    interest here. The second result exists to give the ring a consumer, which
    is a scheduling property and so has nothing to observe in interpret mode.
    """
    fn = _make_bell_fn(NUM_GROUPS, interpret=True)
    announced = jnp.zeros((BELL_SUBLANES, BELL_LANES), jnp.int32)
    group_id = jnp.asarray([group], jnp.int32)
    gen = jnp.full((1, BELL_SUBLANES, BELL_LANES), generation, jnp.int32)
    rung, _ = fn(announced, bells, group_id, gen)
    return rung


def _counters(bells):
    return [int(bells[g, 0, 0]) for g in range(NUM_GROUPS)]


def test_ring_writes_only_its_own_group():
    bells = _ring(jnp.zeros(SHAPE, jnp.int32), 3, 47)
    expected = [0] * NUM_GROUPS
    expected[3] = 47
    assert _counters(bells) == expected


def test_rings_accumulate_across_groups():
    bells = jnp.zeros(SHAPE, jnp.int32)
    bells = _ring(bells, 3, 47)
    bells = _ring(bells, 7, 47)
    expected = [0] * NUM_GROUPS
    expected[3] = 47
    expected[7] = 47
    assert _counters(bells) == expected


def test_reringing_a_group_replaces_only_that_group():
    bells = jnp.zeros(SHAPE, jnp.int32)
    bells = _ring(bells, 3, 47)
    bells = _ring(bells, 7, 47)
    bells = _ring(bells, 3, 48)
    expected = [0] * NUM_GROUPS
    expected[3] = 48
    expected[7] = 47
    assert _counters(bells) == expected


@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_group_is_reachable_and_fully_written(group):
    """A reader may sample any element of the tile, not just [0, 0]."""
    bells = _ring(jnp.zeros(SHAPE, jnp.int32), group, 99)
    assert int(bells[group].min()) == 99
    assert int(bells[group].max()) == 99
    others = [g for g in range(NUM_GROUPS) if g != group]
    assert all(int(bells[g].max()) == 0 for g in others)


def _layer_names():
    return [f"model.layers.{4 * i + 3}.self_attn.attn" for i in range(15)]


def _cpu_set(**kwargs):
    names = _layer_names()
    return DoorbellSet(names, torch.device("cpu"), **kwargs), names


def test_operands_are_sized_for_a_single_tile_write():
    """A ring writes one tile, so only the buffer is buffer-shaped.

    The generation is a tile because that is all the output block covers, and
    the group index is a one-element array because the index map reads it from
    SMEM. Growing either back to the buffer's shape would put 60 KiB through
    VMEM on every ring for no reason.
    """
    bells, _ = _cpu_set()
    assert tuple(bells.bells.shape) == (NUM_GROUPS, BELL_SUBLANES, BELL_LANES)
    assert tuple(bells.generation.shape) == (1, BELL_SUBLANES, BELL_LANES)
    assert tuple(bells.group_ids[0].shape) == (1, )


def test_sharding_does_not_change_the_buffer():
    """A torch tensor handed to a Pallas op is one chip's shard, which
    `jax_placeholder` multiplies up to the global shape. So sharding the bells
    changes the partition spec and nothing about the allocation."""
    from jax.sharding import PartitionSpec

    mesh = jax.make_mesh((1, ), ("pcp", ), devices=jax.devices("cpu")[:1])
    want = (NUM_GROUPS, BELL_SUBLANES, BELL_LANES)

    plain, _ = _cpu_set()
    plain.prebuild(None, None)
    sharded, _ = _cpu_set()
    sharded.prebuild(mesh, PartitionSpec("pcp"))

    assert tuple(plain.bells.shape) == want
    assert tuple(sharded.bells.shape) == want


def test_each_group_id_names_its_own_group():
    bells, _ = _cpu_set()
    for group in range(NUM_GROUPS):
        assert int(bells.group_ids[group][0]) == group


def test_layers_map_to_groups_in_order_and_gdn_layers_do_not():
    bells, names = _cpu_set()
    assert [bells.group_for(n) for n in names] == list(range(NUM_GROUPS))
    # A GDN layer never reaches this backend, so it has no group and the ring
    # is a no-op for it.
    assert bells.group_for("model.layers.4.linear_attn") is None


class _FakeAttention:
    """Enough of vLLM's `Attention` for `install` to wrap: a forward that
    returns a value derived from its input, and a record of being called."""

    def __init__(self):
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * 2


def test_install_wraps_only_the_layers_that_have_a_group(monkeypatch):
    bells, names = _cpu_set()
    layers = {n: _FakeAttention() for n in names}
    # A GDN layer sits in the same mapping on a real worker and must be left
    # alone, since wrapping it would ring a bell that does not exist.
    layers["model.layers.4.linear_attn"] = _FakeAttention()

    rung = []

    def fake_ring(b, name, announced):
        rung.append((name, announced))
        return announced

    monkeypatch.setattr(doorbell_module, "ring_for", fake_ring)

    assert doorbell_module.install(bells, layers) == NUM_GROUPS

    for name in names:
        assert layers[name].forward(3) == 6
    assert [n for n, _ in rung] == names
    # The bell hears what the layer returned, not what it was handed. That is
    # the whole ordering guarantee.
    assert [a for _, a in rung] == [6] * NUM_GROUPS

    layers["model.layers.4.linear_attn"].forward(3)
    assert len(rung) == NUM_GROUPS


def test_install_keeps_each_layers_own_name(monkeypatch):
    """The wrapper closes over the loop variable, so a naive closure would give
    every layer the last name and every bell the same group."""
    bells, names = _cpu_set()
    layers = {n: _FakeAttention() for n in names}

    rung = []

    def fake_ring(b, name, announced):
        rung.append(name)
        return announced

    monkeypatch.setattr(doorbell_module, "ring_for", fake_ring)

    doorbell_module.install(bells, layers)
    layers[names[0]].forward(1)
    assert rung == [names[0]]


def test_install_returns_the_post_ring_value(monkeypatch):
    """The wrapper must hand the layer what `ring_for` returned.

    Returning the original instead leaves the ring with no consumer, and XLA
    then schedules every bell at the end of the program rather than beside the
    attention it reports on. Nothing about the numbers changes when this
    regresses, so only a test of the plumbing catches it.
    """
    bells, names = _cpu_set()
    layers = {n: _FakeAttention() for n in names}

    monkeypatch.setattr(doorbell_module, "ring_for",
                        lambda b, name, announced: announced + 1)

    doorbell_module.install(bells, layers)
    # `_FakeAttention.forward` doubles, so the unwrapped answer is 6.
    assert layers[names[0]].forward(3) == 7


def test_ring_for_passes_through_an_unwatched_layer():
    """A layer with no group has to get its own value back untouched."""
    bells, _ = _cpu_set()
    sentinel = object()
    assert doorbell_module.ring_for(bells, "model.layers.4.linear_attn",
                                    sentinel) is sentinel


def test_verify_reads_are_bounded():
    """A read forces a device sync, so it must not happen every step."""
    bells, _ = _cpu_set(verify_steps=3)

    reads = []
    bells.read = lambda: reads.append(1) or [0] * NUM_GROUPS

    for _ in range(10):
        bells.maybe_verify()
    assert len(reads) == 3


def test_prebuild_registers_a_torch_op():
    """`pallas.jax_op` reads the JAX function's signature to build the torch
    op schema, so an unannotated parameter fails at registration. That happens
    during KV cache setup on a real worker, long after these tests would have
    passed, so register once here where it costs nothing."""
    bells, _ = _cpu_set()
    bells.prebuild(None, None)
    assert bells.op is not None

    # Idempotent: both KV cache paths call the initializer.
    first = bells.op
    bells.prebuild(None, None)
    assert bells.op is first


def test_prebuild_registers_a_torch_op_under_shard_map():
    """PCP prefill takes the sharded path, and `shard_map` wraps the function
    the schema is read from. A one-device CPU mesh is enough to check the
    annotations survive that wrapper."""
    from jax.sharding import PartitionSpec

    mesh = jax.make_mesh((1, ), ("x", ), devices=jax.devices("cpu")[:1])
    bells, _ = _cpu_set()
    bells.prebuild(mesh, PartitionSpec("x"))
    assert bells.op is not None


def test_shards_come_from_the_mesh_not_from_a_second_config_read():
    from jax.sharding import PartitionSpec

    from vllm_torchtpu.distributed.kv_transfer.doorbell import shards_for

    mesh = jax.make_mesh((1, ), ("pcp", ), devices=jax.devices("cpu")[:1])
    assert shards_for(mesh, PartitionSpec("pcp")) == 1
    # No mesh, no spec, or a spec that leaves dimension zero alone all mean
    # the KV cache lives on one chip.
    assert shards_for(None, None) == 1
    assert shards_for(mesh, None) == 1
    assert shards_for(mesh, PartitionSpec(None)) == 1


def test_the_bells_follow_the_kv_cache_onto_its_axis():
    """The axis is read off the KV cache's own spec rather than named in the
    doorbell, so the two cannot drift apart. Recording the specs `shard_map`
    was handed is the only way to see that from outside."""
    from jax.sharding import PartitionSpec

    import vllm_torchtpu.distributed.kv_transfer.doorbell as mod

    seen = {}
    original = mod._make_sharded_bell_fn

    def spy(num_groups, mesh, announced_spec, bells_spec):
        seen["bells"] = bells_spec
        return original(num_groups, mesh, announced_spec, bells_spec)

    mesh = jax.make_mesh((1, ), ("pcp", ), devices=jax.devices("cpu")[:1])
    bells, _ = _cpu_set()
    mod._make_sharded_bell_fn = spy
    try:
        bells.prebuild(mesh, PartitionSpec("pcp"))
    finally:
        mod._make_sharded_bell_fn = original

    assert seen["bells"] == PartitionSpec("pcp")


class _FakeStats:

    def __init__(self, polls):
        self.polls = polls
        self.poll_failures = 0
        self.poll_nanos_total = polls * 40_000
        self.poll_nanos_max = 90_000
        self.torn_tiles = 0
        self.sightings_dropped = 0
        self.sightings_recorded = 0


class _FakeSighting:

    def __init__(self, group, generation, host_nanos):
        self.group = group
        self.generation = generation
        self.host_nanos = host_nanos


class _FakeReader:
    """Stands in for raiden's C++ reader.

    The watcher's job is bookkeeping: grouping sightings by generation,
    deciding which generations are complete, and turning the complete ones
    into a stagger. None of that needs a device, and testing it here means the
    only thing the TPU has to answer is whether the counters move at all.
    """

    def __init__(self, num_groups):
        self._num_groups = num_groups
        self.queued = []
        self.started_with = None
        self.stopped = False
        self.fired = []
        self.armed = []
        self.cancelled = []
        self.next_watch_id = 1

    def start(self, poll_interval_nanos):
        self.started_with = poll_interval_nanos

    def stop(self):
        self.stopped = True

    def num_groups(self):
        return self._num_groups

    def drain_sightings(self):
        out, self.queued = self.queued, []
        return out

    def drain_firings(self):
        out, self.fired = self.fired, []
        return out

    def arm_transfer(self, group, generation, client, req_id, uuid):
        watch_id = self.next_watch_id
        self.next_watch_id += 1
        self.armed.append((group, generation, client, req_id, uuid, watch_id))
        return watch_id

    def cancel_watch(self, watch_id):
        self.cancelled.append(watch_id)
        return watch_id in {entry[5] for entry in self.armed}

    def stats(self):
        return _FakeStats(polls=1000)


def _watcher_with(monkeypatch, reader, report_every=1):
    import vllm_torchtpu.distributed.kv_transfer.doorbell as mod

    bells, _ = _cpu_set()
    monkeypatch.setattr(mod, "_make_reader", lambda tensor: reader)
    return mod.DoorbellWatcher(bells,
                               poll_interval_ns=200_000,
                               report_every=report_every), bells


def test_watcher_starts_the_reader_at_the_requested_cadence(monkeypatch):
    reader = _FakeReader(NUM_GROUPS)
    watcher, _ = _watcher_with(monkeypatch, reader)
    assert reader.started_with == 200_000
    watcher.close()
    assert reader.stopped


def test_watcher_reports_only_passes_where_every_group_rang(
        monkeypatch, caplog):
    """A pass missing a group has no stagger, since the span would be measured
    against whichever groups happened to be sighted."""
    reader = _FakeReader(NUM_GROUPS)
    watcher, _ = _watcher_with(monkeypatch, reader)

    reader.queued = [
        _FakeSighting(g, 1, 1_000_000 * g) for g in range(NUM_GROUPS - 1)
    ]
    with caplog.at_level("INFO"):
        watcher.after_step()
    assert "no pass yet had all" in caplog.text
    assert "stagger over" not in caplog.text


def test_watcher_measures_the_span_from_first_bell_to_last(
        monkeypatch, caplog):
    reader = _FakeReader(NUM_GROUPS)
    watcher, _ = _watcher_with(monkeypatch, reader)

    # Group g rings one millisecond after group g-1, so fifteen groups span
    # fourteen milliseconds.
    reader.queued = [
        _FakeSighting(g, 1, 1_000_000 * g) for g in range(NUM_GROUPS)
    ]
    with caplog.at_level("INFO"):
        watcher.after_step()
    assert "median 14.00 ms" in caplog.text
    assert "0.00, 1.00, 2.00" in caplog.text


def test_watcher_reports_a_stale_buffer_differently_from_silent_bells(
        monkeypatch, caplog):
    """These fail identically from the reader's side and have opposite causes,
    so the direct read that tells them apart is worth its one device sync."""
    reader = _FakeReader(NUM_GROUPS)
    watcher, bells = _watcher_with(monkeypatch, reader)
    bells.read = lambda: [7] * NUM_GROUPS

    with caplog.at_level("INFO"):
        watcher.after_step()
    assert "watching stale memory" in caplog.text


def test_watcher_stays_quiet_when_the_bells_are_genuinely_unrung(
        monkeypatch, caplog):
    reader = _FakeReader(NUM_GROUPS)
    watcher, bells = _watcher_with(monkeypatch, reader)
    bells.read = lambda: [0] * NUM_GROUPS

    with caplog.at_level("INFO"):
        watcher.after_step()
    assert "agrees the bells are unrung" in caplog.text
    assert "watching stale memory" not in caplog.text


def test_watcher_reports_every_nth_step_and_not_between(monkeypatch):
    reader = _FakeReader(NUM_GROUPS)
    watcher, _ = _watcher_with(monkeypatch, reader, report_every=5)

    drains = []
    original = reader.drain_sightings
    reader.drain_sightings = lambda: drains.append(1) or original()

    for _ in range(12):
        watcher.after_step()
    assert len(drains) == 2


def test_watcher_forgets_passes_older_than_the_newest_complete_one(
        monkeypatch):
    """Generations arrive in order, so a pass still missing a group once a
    later pass has finished will never fill in. Keeping it would leak."""
    reader = _FakeReader(NUM_GROUPS)
    watcher, _ = _watcher_with(monkeypatch, reader)

    reader.queued = (
        [_FakeSighting(0, 1, 0)] +
        [_FakeSighting(g, 2, 1000 * g) for g in range(NUM_GROUPS)])
    watcher.after_step()
    # Generation 2 completed, so it is measured and retired; generation 1 never
    # will and is dropped. Nothing is left waiting.
    assert list(watcher._pending) == []
    assert len(watcher._spans_ms) == 1


def test_watcher_measures_every_complete_pass_not_just_the_last(monkeypatch):
    """Pruning used to run before the spans were taken, which left one pass
    behind and reported its span as a median over the whole run."""
    reader = _FakeReader(NUM_GROUPS)
    watcher, _ = _watcher_with(monkeypatch, reader)

    # Three complete passes, each wider than the one before.
    reader.queued = [
        _FakeSighting(g, gen, 1000 * g * gen)
        for gen in (1, 2, 3) for g in range(NUM_GROUPS)
    ]
    watcher.after_step()
    assert len(watcher._spans_ms) == 3
    assert watcher._spans_ms == sorted(watcher._spans_ms)


def test_generation_starts_below_first_and_rises_once_per_step():
    bells, _ = _cpu_set()
    assert int(bells.generation[0, 0, 0]) == FIRST_GENERATION - 1
    assert bells.next_generation() == FIRST_GENERATION
    assert bells.next_generation() == FIRST_GENERATION + 1
    # A freshly zeroed buffer must not read as already rung.
    assert bells.read() == [0] * NUM_GROUPS
    assert FIRST_GENERATION > 0


class _FakeCachedAttention(_FakeAttention):
    """An attention layer with a KV cache bound, as vLLM leaves it.

    The cache is a small paged buffer whose pages hold distinct bytes, so a
    checksum over the wrong pages produces a different number rather than
    coincidentally the right one.
    """

    def __init__(self, fill: int):
        super().__init__()
        pages = torch.arange(4 * 8, dtype=torch.uint8).reshape(4, 8)
        self.kv_cache = (pages + fill).to(torch.uint8)


def test_page_checksum_sums_the_named_pages_only():
    """The number has to depend on which pages were asked for."""
    layer = _FakeCachedAttention(fill=0)
    first = doorbell_module.page_checksum(layer.kv_cache, [0])
    second = doorbell_module.page_checksum(layer.kv_cache, [1])
    assert first == sum(range(0, 8))
    assert second == sum(range(8, 16))
    assert doorbell_module.page_checksum(layer.kv_cache, [0, 1]) == (first +
                                                                     second)


def test_page_checksum_is_none_when_there_is_no_cache():
    """The memory-probe forward runs before any cache is bound."""
    assert doorbell_module.page_checksum(None, [0]) is None
    assert doorbell_module.page_checksum(torch.zeros(0), [0]) is None
    assert doorbell_module.page_checksum(torch.zeros(4, 8), []) is None


def test_page_checksum_ignores_pages_outside_the_cache():
    """A short block-table row pads with ids the cache does not have."""
    layer = _FakeCachedAttention(fill=0)
    assert doorbell_module.page_checksum(layer.kv_cache, [0, 999]) == sum(
        range(0, 8))


def test_page_check_is_bounded_like_the_verify_read(monkeypatch):
    """It copies pages to the host, so it cannot run on every step."""
    bells, names = _cpu_set(check_steps=3)
    layers = {n: _FakeCachedAttention(fill=i) for i, n in enumerate(names)}
    monkeypatch.setattr(doorbell_module, "ring_for",
                        lambda b, name, announced: announced)
    doorbell_module.install(bells, layers)

    seen = []
    monkeypatch.setattr(doorbell_module, "page_checksum",
                        lambda cache, ids, tokens: seen.append(1) or 7)
    for _ in range(10):
        if bells.wants_page_check():
            bells.maybe_check_pages([0, 1])
    # Three checked steps, each hashing all fifteen layers.
    assert len(seen) == 3 * NUM_GROUPS
    assert not bells.wants_page_check()


def test_page_check_walks_layers_in_group_order(monkeypatch):
    """Index i of the check has to be the same layer as bell i.

    `install` iterates the dict vLLM hands it, and nothing guarantees that
    order matches the group numbering. If the two ever diverge, a check that
    trusted dict order would compare each bell against a different layer's
    pages and report a mismatch that is really its own bug.
    """
    bells, names = _cpu_set(check_steps=1)
    monkeypatch.setattr(doorbell_module, "ring_for",
                        lambda b, name, announced: announced)
    # Hand `install` the layers back to front.
    layers = {n: _FakeCachedAttention(fill=i) for i, n in enumerate(names)}
    doorbell_module.install(bells, dict(reversed(list(layers.items()))))

    assert [id(x) for x in bells._layers] == [id(layers[n]) for n in names]


def test_page_check_says_so_when_every_layer_hashes_alike(monkeypatch, caplog):
    """Identical sums across layers means the bells share memory."""
    bells, names = _cpu_set(check_steps=1)
    monkeypatch.setattr(doorbell_module, "ring_for",
                        lambda b, name, announced: announced)
    # Every layer given the same cache contents.
    doorbell_module.install(bells,
                            {n: _FakeCachedAttention(fill=0)
                             for n in names})
    with caplog.at_level("ERROR"):
        bells.maybe_check_pages([0])
    assert "same" in caplog.text


def test_page_checksum_reads_only_the_requested_slots():
    """The slice width has to change the number, or it is not being applied.

    The production config runs `--block-size 8192`, so a page is far larger
    than any request that fits in one, and reading all of it would drown the
    request's own bytes in whatever the allocator left behind.
    """
    cache = torch.arange(2 * 16, dtype=torch.uint8).reshape(2, 16)
    narrow = doorbell_module.page_checksum(cache, [0], tokens=4)
    wide = doorbell_module.page_checksum(cache, [0], tokens=16)
    assert narrow == sum(range(0, 4))
    assert wide == sum(range(0, 16))
