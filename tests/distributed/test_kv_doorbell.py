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

from vllm_torchtpu.distributed.kv_transfer.doorbell import (BELL_LANES,
                                                            BELL_SUBLANES,
                                                            FIRST_GENERATION,
                                                            DoorbellSet,
                                                            _bell_kernel)

NUM_GROUPS = 15
SHAPE = (NUM_GROUPS, BELL_SUBLANES, BELL_LANES)


def _ring(bells, group, generation):
    """One ring, built the way `DoorbellSet` builds its operands."""
    call = pl.pallas_call(
        _bell_kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.int32),
        input_output_aliases={2: 0},
        interpret=True,
    )
    onehot = (jnp.arange(NUM_GROUPS,
                         dtype=jnp.int32) == group).astype(jnp.int32)
    selector = jnp.broadcast_to(onehot[:, None, None], SHAPE)
    gen = jnp.full(SHAPE, generation, jnp.int32)
    return call(selector, gen, bells)


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


def _cpu_set():
    names = [f"model.layers.{4 * i + 3}.self_attn.attn" for i in range(15)]
    return DoorbellSet(names, torch.device("cpu")), names


def test_operands_all_share_the_buffer_shape():
    """The kernel selects elementwise, so nothing may be a scalar or a tile."""
    bells, _ = _cpu_set()
    want = (NUM_GROUPS, BELL_SUBLANES, BELL_LANES)
    assert tuple(bells.bells.shape) == want
    assert tuple(bells.generation.shape) == want
    assert tuple(bells.selectors[0].shape) == want


def test_each_selector_picks_out_exactly_one_group():
    bells, _ = _cpu_set()
    for group in range(NUM_GROUPS):
        selector = bells.selectors[group]
        assert int(selector[group].min()) == 1
        others = [g for g in range(NUM_GROUPS) if g != group]
        assert all(int(selector[g].max()) == 0 for g in others)


def test_layers_map_to_groups_in_order_and_gdn_layers_do_not():
    bells, names = _cpu_set()
    assert [bells.group_for(n) for n in names] == list(range(NUM_GROUPS))
    # A GDN layer never reaches this backend, so it has no group and the ring
    # is a no-op for it.
    assert bells.group_for("model.layers.4.linear_attn") is None


def test_verify_reads_are_bounded():
    """A read forces a device sync, so it must not happen every step."""
    names = [f"model.layers.{4 * i + 3}.self_attn.attn" for i in range(15)]
    bells = DoorbellSet(names, torch.device("cpu"), verify_steps=3)

    reads = []
    bells.read = lambda: reads.append(1) or [0] * NUM_GROUPS

    for _ in range(10):
        bells.maybe_verify()
    assert len(reads) == 3


def test_generation_starts_below_first_and_rises_once_per_step():
    bells, _ = _cpu_set()
    assert int(bells.generation[0, 0, 0]) == FIRST_GENERATION - 1
    assert bells.next_generation() == FIRST_GENERATION
    assert bells.next_generation() == FIRST_GENERATION + 1
    # A freshly zeroed buffer must not read as already rung.
    assert bells.read() == [0] * NUM_GROUPS
    assert FIRST_GENERATION > 0
