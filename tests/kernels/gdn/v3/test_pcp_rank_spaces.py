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
"""Which number each part of the GDN PCP path is allowed to speak.

Two numbering spaces meet in this kernel, and they coincide only when the PCP
process group happens to be laid out in device order:

  pcp rank         the host's index into the PCP group, i.e. the group's
                   rank_in_group. Every ``rank_*`` and ``projection_*`` table
                   the host builds has one row per PCP rank.
  mesh coordinate  what ``lax.axis_index`` returns, what a
                   ``DeviceIdType.MESH`` peer address resolves against, and
                   the order a native collective stacks its participants in.

Mixing them corrupts results in silence: shapes stay right and nothing raises.
The eight-chip reproducer cannot see it either, because its only pcp8 case
runs two equal-length requests -- every per-rank vector is then constant, and
permuting a constant vector changes nothing. So the rules are pinned here
instead, as CPU tests that run on every change rather than only on a pod.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.distributed import mesh_utils
from vllm_torchtpu.kernels.gdn.v3 import pcp_wrapper
from vllm_torchtpu.layers.core import gdn_attention

# A tpu7x-8 host: the PCP group's rank order is not the device order. This is
# the configuration every rule below exists for; under the identity order the
# two spaces agree and none of this is observable.
_MEASURED_DEVICE_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)


class _FakeDevice:
    """Only ``.id`` is read, by both the mesh walk and the runtime walk."""

    def __init__(self, device_id: int):
        self.id = device_id

    def __repr__(self) -> str:
        return f"_FakeDevice(id={self.id})"


def _fake_mesh(device_ids, axis_names=("pcp", )):
    """A stand-in for jax.sharding.Mesh with just what the helper reads."""
    return type(
        "_FakeMesh", (), {
            "axis_names":
            tuple(axis_names),
            "devices":
            np.array([_FakeDevice(i) for i in device_ids], dtype=object),
        })()


def _runtime_devices(monkeypatch, device_ids):
    """jax.export rebuilds the assignment by walking the client's devices."""
    monkeypatch.setattr(jax, "devices",
                        lambda *a, **k: [_FakeDevice(i) for i in device_ids])


# --- mesh_coordinates_for_axis: PCP rank -> mesh coordinate ---------------


def test_mesh_coordinates_are_the_identity_when_the_group_is_in_device_order(
        monkeypatch):
    _runtime_devices(monkeypatch, range(8))
    mesh = _fake_mesh(range(8))
    assert mesh_utils.mesh_coordinates_for_axis(mesh, "pcp") == tuple(range(8))


def test_mesh_coordinates_translate_the_measured_tpu7x_layout(monkeypatch):
    # The runtime walks devices in id order, so device d sits at coordinate d.
    # The mesh is in PCP rank order, so PCP rank i holds device
    # _MEASURED_DEVICE_ORDER[i] and answers to that coordinate.
    _runtime_devices(monkeypatch, range(8))
    mesh = _fake_mesh(_MEASURED_DEVICE_ORDER)
    assert (mesh_utils.mesh_coordinates_for_axis(
        mesh, "pcp") == _MEASURED_DEVICE_ORDER)


def test_mesh_coordinates_are_a_permutation_for_every_group_order(monkeypatch):
    _runtime_devices(monkeypatch, range(8))
    rng = np.random.default_rng(0)
    for _ in range(16):
        order = rng.permutation(8)
        got = mesh_utils.mesh_coordinates_for_axis(_fake_mesh(order), "pcp")
        assert sorted(got) == list(range(8))


def test_mesh_coordinates_reject_a_device_the_runtime_does_not_expose(
        monkeypatch):
    # Addressing a peer that is not in the assignment cannot be expressed, and
    # guessing would deliver the rows to whoever is there instead.
    _runtime_devices(monkeypatch, range(8))
    mesh = _fake_mesh((0, 1, 2, 3, 4, 5, 6, 99))
    with pytest.raises(RuntimeError, match="absent from the runtime"):
        mesh_utils.mesh_coordinates_for_axis(mesh, "pcp")


def test_mesh_coordinates_reject_a_short_runtime_assignment(monkeypatch):
    _runtime_devices(monkeypatch, range(4))
    with pytest.raises(RuntimeError, match="JAX exposes only"):
        mesh_utils.mesh_coordinates_for_axis(_fake_mesh(range(8)), "pcp")


def test_mesh_coordinates_reject_an_axis_that_is_not_unique():
    mesh = _fake_mesh(range(8), axis_names=("pcp", "pcp"))
    with pytest.raises(ValueError, match="exactly once"):
        mesh_utils.mesh_coordinates_for_axis(mesh, "pcp")


# --- _pcp_rank_for_mesh_coordinate: mesh coordinate -> PCP rank -----------


def test_pcp_rank_for_mesh_coordinate_inverts_the_map():
    for pcp_rank, coordinate in enumerate(_MEASURED_DEVICE_ORDER):
        got = pcp_wrapper._pcp_rank_for_mesh_coordinate(
            _MEASURED_DEVICE_ORDER, jnp.int32(coordinate))
        assert int(got) == pcp_rank, (
            f"mesh coordinate {coordinate} must resolve to PCP rank "
            f"{pcp_rank}")


def test_pcp_rank_for_mesh_coordinate_is_the_identity_under_an_identity_map():
    order = tuple(range(8))
    for coordinate in range(8):
        got = pcp_wrapper._pcp_rank_for_mesh_coordinate(
            order, jnp.int32(coordinate))
        assert int(got) == coordinate


# --- _reorder_collective_output_to_pcp_rank: collective output -> rank ----


def test_collective_blocks_are_reordered_to_pcp_rank():
    # Label every row with the mesh coordinate that contributed it: a
    # collective stacks its participants in that order, so block p is
    # coordinate p's. Stating the expectation as "which participant's data
    # ends up at PCP rank i" keeps this a test of the contract rather
    # than a restatement of the gather the implementation happens to use.
    rows_per_block = 3
    labels = jnp.repeat(jnp.arange(8, dtype=jnp.float32),
                        rows_per_block).reshape(8 * rows_per_block, 1)
    got = np.asarray(
        gdn_attention._reorder_collective_output_to_pcp_rank(
            labels, _MEASURED_DEVICE_ORDER))
    for pcp_rank, coordinate in enumerate(_MEASURED_DEVICE_ORDER):
        block = got[pcp_rank * rows_per_block:(pcp_rank + 1) * rows_per_block,
                    0]
        assert set(block.tolist()) == {
            float(coordinate)
        }, (f"PCP rank {pcp_rank} must be handed the block that "
            f"collective participant {coordinate} contributed, got {block}")


def test_collective_reorder_is_a_no_op_under_an_identity_map():
    tensor = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    got = gdn_attention._reorder_collective_output_to_pcp_rank(
        tensor, tuple(range(8)))
    assert got is tensor


def test_collective_reorder_rejects_an_axis_it_cannot_split():
    tensor = jnp.zeros((9, 2), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Cannot split axis 0"):
        gdn_attention._reorder_collective_output_to_pcp_rank(
            tensor, _MEASURED_DEVICE_ORDER)


# --- the kernel's own indexing, pinned at the source ----------------------
#
# The rules below are decisions inside a Pallas kernel that only an eight-chip
# run executes, so no Python call can reach them. Reading the source is what
# keeps a locally reasonable sweep from undoing them.

# Built by the host, one row per PCP group position.
_RANK_ORDERED_TABLES = (
    "rank_active_row_end",
    "rank_row_start",
    "rank_valid_rows",
    "rank_recv_start",
    "projection_catchup_start",
    "projection_catchup_count",
    "projection_work_offset",
    "projection_work_offset_end",
)


def _wrapper_source() -> str:
    with open(pcp_wrapper.__file__, encoding="utf-8") as handle:
        return handle.read()


def _subscripts(source: str, expression: str):
    """Yield the text inside each ``expression[...]``, newlines included."""
    for match in re.finditer(re.escape(expression) + r"\[", source):
        depth = 1
        start = match.end()
        for index in range(start, len(source)):
            if source[index] == "[":
                depth += 1
            elif source[index] == "]":
                depth -= 1
                if depth == 0:
                    yield source[start:index]
                    break


@pytest.mark.parametrize("table", _RANK_ORDERED_TABLES)
def test_host_built_tables_are_never_indexed_by_the_mesh_coordinate(table):
    source = _wrapper_source()
    seen = 0
    for subscript in _subscripts(source, f"stage_metadata_ref.{table}"):
        seen += 1
        assert not re.search(r"\bmesh_coordinate\b", subscript), (
            f"stage_metadata_ref.{table}[{' '.join(subscript.split())}] "
            "indexes a host-built, rank-ordered table with "
            "`mesh_coordinate`. Use `pcp_rank`: on a group that is not laid "
            "out in device order this silently reads another peer's row.")
    assert seen, (f"stage_metadata_ref.{table}[...] never appears; the table "
                  "was renamed or dropped, so update this list.")


def test_the_head_shard_column_stays_a_mesh_coordinate():
    # The counter-rule. packed_out's second axis is a head shard, and head
    # shards are handed out by the all_to_all in
    # _select_replicated_shard_by_mesh_coordinate, which assigns them by mesh
    # coordinate. Rewriting this one to pcp_rank would introduce the same
    # class of bug in the opposite direction.
    subscripts = list(_subscripts(_wrapper_source(), "packed_out_ref.at"))
    assert subscripts, "packed_out_ref.at[...] not found; update this test"
    assert any(re.search(r"\bmesh_coordinate\b", s) for s in subscripts), (
        "packed_out_ref's head-shard column must stay indexed by "
        "`mesh_coordinate`, not by `pcp_rank`.")


def test_peer_addressing_goes_through_the_partition_order():
    # A DeviceIdType.MESH coordinate is a mesh coordinate. `destination` in
    # the q/k/v push loop is a PCP rank, so it has to be translated before
    # it can name a chip.
    assert "mesh_coordinate_by_pcp_rank[destination]" in _wrapper_source(), (
        "the q/k/v push must address its peer as "
        "mesh_coordinate_by_pcp_rank[destination]; `destination` alone is a "
        "PCP rank and names the wrong chip.")


def test_the_coordinate_table_is_a_static_kernel_argument():
    # Closed over as a Python tuple rather than passed as data: _pcp_rank_for_mesh_coordinate
    # unrolls it into selects because Pallas rejects a closed-over array and
    # SPMD forbids handing each device its own scalar.
    assert re.search(r'^\s*"mesh_coordinate_by_pcp_rank",\s*$',
                     _wrapper_source(),
                     flags=re.MULTILINE), (
                         "mesh_coordinate_by_pcp_rank must stay in the static "
                         "argument list, or the kernel would be asked to "
                         "trace a traced value.")


def test_the_default_coordinate_table_is_the_identity():
    # Callers that have not been taught the difference keep the old
    # behaviour rather than getting a silent reordering.
    source = _wrapper_source()
    assert "mesh_coordinate_by_pcp_rank = tuple(range(pcp_size))" in source, (
        "an absent mesh_coordinate_by_pcp_rank must default to the identity.")
