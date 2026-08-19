# SPDX-License-Identifier: Apache-2.0
"""Mesh index <-> EP rank <-> expert block, pinned on a permuted device order.

The tpu7x-8 we serve on drives EP ranks 0..7 from device ids
``(0, 1, 4, 5, 6, 7, 2, 3)``. The EP mesh has to be in *device-id* order because
a Mosaic kernel resolves a ``DeviceIdType.MESH`` peer coordinate positionally
against the runtime's device list, so mesh index ``i`` has to be device id
``i``; mesh index is then no longer EP rank, and the shard at index ``i`` holds
another rank's experts.

Every hop of that chain is one composition away from a plausible wrong answer:
the permutation and its inverse are both permutations of 0..7, nothing
type-checks the difference, and applying the wrong one produces finite numbers,
in the right rows, from the wrong experts. So the mapping is pinned here as
arithmetic -- no devices, no kernel, no tolerance -- with the node's real
permutation. What the arithmetic controls is pinned by value in
``tests/kernels/fused_moe/test_fused_ep_v2_numerics.py``.
"""

import pytest
import torch

from vllm_torchtpu.distributed import ep_mesh
from vllm_torchtpu.distributed.ep_mesh import EpGroupLayout

# EP rank r drives NODE_DEVICE_IDS[r]; sorting for the mesh puts EP rank
# NODE_EP_RANK_AT_MESH_INDEX[i] at mesh index i. The second is the first
# inverted, and they differ, which is what makes this node a test case.
NODE_DEVICE_IDS = (0, 1, 4, 5, 6, 7, 2, 3)
NODE_EP_RANK_AT_MESH_INDEX = (0, 1, 6, 7, 2, 3, 4, 5)
EP = 8
EXPERTS = 512  # Qwen3.5-397B: 64 experts per rank


def _layout(rank, device_ids=NODE_DEVICE_IDS):
    return EpGroupLayout(ranks=tuple(range(len(device_ids))),
                         device_ids=tuple(device_ids),
                         rank_in_group=rank,
                         world_size=len(device_ids))


def _block_bases(order, experts=EXPERTS, ep=EP):
    """First global expert id of the weights sitting at each mesh index.

    vLLM gives EP rank ``r`` the contiguous block starting at ``r * E/ep``
    (``moe_routing.get_experts_start``, linear placement), and mesh index ``i``
    runs on the process whose rank is ``order[i]``. This is the table the v2
    relabel inverts: it renumbers a global expert id into the mesh-block space
    these bases define.
    """
    assert sorted(order) == list(range(ep)), f"{order} is not a permutation"
    return tuple(rank * (experts // ep) for rank in order)


# --- the mapping itself ------------------------------------------------------


def test_the_mesh_is_the_device_ids_in_ascending_order():
    assert _layout(0).mesh_device_ids == tuple(range(EP))


def test_the_node_permutation_is_not_the_identity():
    """If it were, this whole file would be testing nothing."""
    layout = _layout(0)
    assert not layout.mesh_order_is_ep_rank_order
    assert _layout(0, tuple(range(EP))).mesh_order_is_ep_rank_order


def test_which_ep_rank_sits_at_each_mesh_index():
    assert _layout(0).ep_rank_at_mesh_index == NODE_EP_RANK_AT_MESH_INDEX


def test_the_mesh_index_is_where_this_rank_s_device_sorts():
    assert [_layout(r).mesh_index for r in range(EP)
            ] == [sorted(NODE_DEVICE_IDS).index(d) for d in NODE_DEVICE_IDS]


@pytest.mark.parametrize("rank", range(EP))
def test_a_rank_finds_its_own_expert_block_at_its_own_mesh_index(rank):
    """The round trip the kernel needs, stated end to end.

    Rank r sits at mesh index ``mesh_index``; the shard there is the one vLLM
    loaded rank r's experts into, so the block the relabel must send to mesh
    index ``mesh_index`` is the one starting at ``r * E/ep``. Anything else
    routes rows to weights that are not on that chip.
    """
    layout = _layout(rank)
    index = layout.mesh_index
    assert layout.ep_rank_at_mesh_index[index] == rank
    assert _block_bases(
        layout.ep_rank_at_mesh_index)[index] == rank * (EXPERTS // EP)


def test_the_expert_block_table_for_this_node():
    """The literal table, so a change of convention has to be deliberate."""
    assert _block_bases(NODE_EP_RANK_AT_MESH_INDEX) == (0, 64, 384, 448, 128,
                                                        192, 256, 320)


def test_an_identity_order_is_the_arange_it_replaces():
    """Deployments in device-id order must get exactly the old table."""
    assert _block_bases(tuple(range(EP))) == tuple(
        range(0, EXPERTS, EXPERTS // EP))


def test_the_uninverted_permutation_is_a_different_table():
    """NEGATIVE CONTROL: using the device order where its inverse belongs.

    Both are permutations of 0..7 so nothing type-checks the difference, and on
    a node whose permutation happened to be an involution the two would agree
    and this bug would hide. This one is not an involution: six of the eight
    mesh indices get a different block, and the two that do not are the
    permutation's fixed points.
    """
    right = _block_bases(NODE_EP_RANK_AT_MESH_INDEX)
    wrong = _block_bases(NODE_DEVICE_IDS)
    assert right != wrong
    assert [i for i, (a, b) in enumerate(zip(right, wrong))
            if a != b] == [2, 3, 4, 5, 6, 7]


def test_two_ranks_on_one_device_is_a_loud_error():
    """No mesh order can address them, so there is nothing to reindex."""
    with pytest.raises(RuntimeError, match="same TPU global device id"):
        _ = _layout(0, (0, 1, 2, 3, 4, 5, 6, 6)).mesh_device_ids


# --- who is allowed to run on such a node ------------------------------------


class _Group:

    def __init__(self, layout):
        self.ranks = layout.ranks
        self.rank_in_group = layout.rank_in_group
        self.world_size = layout.world_size


class _Layer:

    def __init__(self, layout):
        self.moe_config = type("MoEConfig", (), {
            "ep_rank": layout.rank_in_group,
            "ep_size": layout.world_size,
        })()


@pytest.fixture
def on_the_permuted_node(monkeypatch):
    layout = _layout(2)
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: _Group(layout))
    monkeypatch.setattr(ep_mesh, "get_ep_group_layout", lambda: layout)
    return layout


def test_a_kernel_that_reindexes_may_run_on_a_permuted_node(
        on_the_permuted_node):
    ep_mesh.validate_ep_mesh(_Layer(on_the_permuted_node),
                             reindexes_expert_order=True)


def test_a_kernel_that_reads_ownership_off_its_axis_index_may_not(
        on_the_permuted_node):
    """The v1 kernel's case, and the default.

    v1 builds ``e_id = my_id * local_num_experts + local_e_id`` from
    ``lax.axis_index`` *inside* the kernel, so nothing a driver passes can move
    its ownership. Keeping the default False is what stops the v2 reindex from
    silently arming such a kernel on a node it cannot serve -- and what makes
    ``reindexes_expert_order=True`` a claim about the caller rather than a
    switch that turns the check off.
    """
    with pytest.raises(NotImplementedError, match="ownership from its mesh"):
        ep_mesh.validate_ep_mesh(_Layer(on_the_permuted_node))


def test_an_unpermuted_node_is_fine_for_both(monkeypatch):
    layout = _layout(2, tuple(range(EP)))
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: _Group(layout))
    monkeypatch.setattr(ep_mesh, "get_ep_group_layout", lambda: layout)
    ep_mesh.validate_ep_mesh(_Layer(layout))
    ep_mesh.validate_ep_mesh(_Layer(layout), reindexes_expert_order=True)


# --- the mesh that actually gets built ---------------------------------------


class _FakeDevice:
    """Enough of a device for the checks that must fire before ``Mesh``."""

    def __init__(self, device_id):
        self.id = device_id


def _with_visible_devices(monkeypatch, device_ids):
    import jax

    monkeypatch.setattr(
        jax, "devices",
        lambda *a, **k: [_FakeDevice(d) for d in sorted(device_ids)])
    monkeypatch.setattr(ep_mesh, "_MESH_CACHE", {})


def _capture_mesh(monkeypatch):
    """``Mesh`` captured rather than built, so the order can be asserted
    without eight real devices; that it accepts eight simulated ones is what
    the CPU numerics gate exercises."""
    import jax.sharding
    captured = {}

    class _CapturedMesh:

        def __init__(self, devices, axis_names, axis_types):
            captured.update(ids=[d.id for d in devices.reshape(-1)],
                            shape=devices.shape,
                            axis_names=axis_names,
                            axis_types=axis_types)

    monkeypatch.setattr(jax.sharding, "Mesh", _CapturedMesh)
    return captured


@pytest.mark.parametrize("builder,shape,axes", [
    ("get_or_create_ep_mesh_1d", (EP, ), (ep_mesh.EP_AXIS_NAME_1D, )),
    ("get_or_create_ep_mesh", (1, EP),
     (ep_mesh.EP_MESH_DATA_AXIS, ep_mesh.EP_AXIS_NAME)),
])
def test_the_permuted_node_gets_a_mesh_in_ascending_device_id_order(
        monkeypatch, builder, shape, axes):
    """The node this port exists for: refused before, built now -- and sorted.

    Both builders share ``_get_or_create_mesh``, so both are asserted: the 1-D
    one is what v2 takes and the 2-D one is what v1 took.
    """
    captured = _capture_mesh(monkeypatch)
    monkeypatch.setattr(ep_mesh, "get_ep_group_layout", lambda: _layout(0))
    _with_visible_devices(monkeypatch, range(EP))

    getattr(ep_mesh, builder)()

    assert captured["ids"] == list(range(EP)), (
        "the mesh must be in ascending device-id order, whatever order the EP "
        "ranks came up in")
    assert captured["shape"] == shape
    assert captured["axis_names"] == axes


def test_an_ep_group_on_devices_other_than_0_to_n_is_refused(monkeypatch):
    """The fallback the reindex cannot cover, and why it is not a regression.

    Reordering fixes rank-to-position; it cannot fix id-to-position. A group
    holding devices 4..7 of a node would have mesh index 0 addressing device 0,
    which is not in the group at all -- so refuse. The reindex does not reach
    this either: it moves ownership between *ranks*, not between devices.
    """
    monkeypatch.setattr(ep_mesh, "get_ep_group_layout",
                        lambda: _layout(0, (4, 5, 6, 7)))
    _with_visible_devices(monkeypatch, range(8))
    with pytest.raises(NotImplementedError, match="device ids to be 0..3"):
        ep_mesh.get_or_create_ep_mesh_1d()


# --- what each process actually ships ----------------------------------------


@pytest.mark.parametrize("rank", range(EP))
def test_each_rank_ships_its_own_mesh_index_not_its_ep_rank(monkeypatch, rank):
    """``register_ep_rank_buffer`` after the reorder.

    ``_ep_rank_ids`` is the shard identity the kernel reads for BOTH of its
    roles, and one of them -- the peer coordinate others push to -- is a mesh
    coordinate by construction. So it ships the mesh index; the expert-block
    role is what ``mesh_expert_order`` compensates. Shipping the EP rank here
    is the spelling that deadlocks the group, and on this node six ranks would
    ship a different number, so it is not a subtle difference.
    """
    from vllm_torchtpu.layers.vllm import fused_moe_ep

    layout = _layout(rank)
    monkeypatch.setattr(ep_mesh, "get_ep_group_layout", lambda: layout)
    layer = type("Layer", (), {})()
    layer.register_buffer = (
        lambda name, tensor, persistent=True: setattr(layer, name, tensor))

    fused_moe_ep.register_ep_rank_buffer(layer, device=torch.device("cpu"))

    assert layer._ep_rank_ids.tolist() == [layout.mesh_index]
    assert layer._ep_rank_ids.dtype == torch.int32
    # Assembled over the group -- the operand is sharded, so element i comes
    # from the process at mesh index i -- it is the arange the kernel assumes.
    by_mesh_index = sorted(range(EP), key=lambda r: _layout(r).mesh_index)
    assert [_layout(r).mesh_index for r in by_mesh_index] == list(range(EP))


def test_the_relabel_table_the_driver_ships_is_the_mesh_order(monkeypatch):
    """``mesh_expert_order`` is the other half: which rank's block is where."""
    from vllm_torchtpu.layers.vllm import fused_moe_ep

    monkeypatch.setattr(ep_mesh, "get_ep_group_layout", lambda: _layout(3))
    assert fused_moe_ep.mesh_expert_order() == NODE_EP_RANK_AT_MESH_INDEX


def test_an_unpermuted_node_ships_no_table_at_all(monkeypatch):
    """None, not ``arange(ep)``: it keeps the traced program byte-identical
    where the reindex is a no-op, so the shipped kernel is unchanged for every
    deployment whose devices come up in rank order."""
    from vllm_torchtpu.layers.vllm import fused_moe_ep

    monkeypatch.setattr(ep_mesh, "get_ep_group_layout",
                        lambda: _layout(3, tuple(range(EP))))
    assert fused_moe_ep.mesh_expert_order() is None
