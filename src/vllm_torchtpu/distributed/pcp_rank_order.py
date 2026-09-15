# SPDX-License-Identifier: Apache-2.0
"""Topology-aware ordering of the PCP group's ranks.

The PCP attention kernel is a ring: rank *r* passes its KV block to rank
*r+1*, ``pcp_size - 1`` times (``next_rank = (my_id + 1) % pcp_size``). The
ring order is the mesh axis order, and that mesh is built from the PCP group's
rank list -- so reordering the group reorders the ring, with no kernel change.

Placement is *not* decided here. Workers bind to chips naturally
(``LOCAL_RANK == rank``); this only changes which position in the ring each
rank occupies. That distinction matters: ``LOCAL_RANK`` restarts at 0 on every
host and so can never express a cross-host ordering, whereas a rank is a
logical label and a group may span hosts.

Why this runs after startup rather than before it: choosing chips requires
knowing which PCI index yields which coordinates, and that is only observable
by opening a chip -- which is the choice itself. ``topology_aware_mesh`` needs
open chips and an initialised process group, so it can only answer once
placement is already fixed. Reordering data is the one lever still available
at that point.

Both the PCP and TP groups come from the same mesh, and they replace vLLM's
arithmetic layout rather than being reconciled with it. ``mesh[d, :, t]`` is a
PCP ring, ``mesh[d, p, :]`` is a TP lane, so one tensor gives every rank a
single ``(dp, pcp, tp)`` position: a rank's TP partners are by construction the
ranks sharing its ``pcp`` coordinate, and they therefore hold the same sequence
chunk. Ordering PCP alone would leave TP on vLLM's arange layout, and where the
two differ TP partners land at different ring positions -- holding different
sequence chunks while the TP all-reduce combines partials computed over
different tokens. Taking both axes from one mesh makes that unrepresentable.

Only ``tp``, ``pcp`` and (degenerately) ``dp``/``pp`` exist at the supported
scope, and at ``dp == 1, pp == 1`` the DP and PP groups are singletons and the
EP group is the whole world, so no other group constrains this layout.

One ordering constraint outranks topology, though, and is applied last. A
kernel reads ``lax.axis_index``, which is a partition id -- a chip's index in
``pjrt_client->devices()`` -- and uses it to index tables the host built per
``rank_in_group``. Those are the same number only if ring position equals chip
position. Before #414 a ``LOCAL_RANK`` remap forced that; #414 removed the
remap, correctly, and nothing took over the invariant, so GDN has been reading
another rank's schedule ever since. ``_order_rings_by_device`` takes it over
here, which is where ``tpu_rank_binding``'s docstring says it belongs. On one
ring that overrides the topology order outright and costs 2 ICI hops; the two
stop competing once the DeviceAssignment can follow the mesh, and at that
point the reordering should be removed, not kept alongside.

"""

import contextlib
import inspect
from collections.abc import Iterator

import numpy

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _rank_to_device_position() -> dict[int, int]:
    """Global rank -> its chip's index in ``pjrt_client->devices()``.

    That index is the partition id the kernel will see. ``torch_tpu`` fills
    the DeviceAssignment with ``pjrt_client->devices()`` in enumeration order
    (``SetDefaultDeviceAssignment``), so ``lax.axis_index`` returns a chip's
    position in that list and nothing else. It is the one numbering this
    module cannot choose, which is why the ring is sorted to match it rather
    than the other way round.

    Gathered rather than read locally: a worker only knows its own chip.
    ``init_distributed_environment`` has run by the time
    ``resolve_pcp_topology_order`` is called, so the world group exists and
    has the gloo side ``all_gather_object`` needs.
    """
    import torch.distributed as dist
    from torch_tpu._internal.distributed import tpu_distributed
    from vllm.distributed.parallel_state import get_world_group

    device_id = int(tpu_distributed.global_device_id())
    pjrt_order = [
        int(value) for value in tpu_distributed.all_global_device_ids()
    ]
    if device_id not in pjrt_order:
        raise RuntimeError(
            f"This worker's TPU device id {device_id} is not in torch_tpu's "
            f"device list {pjrt_order}, so its partition id is undefined.")

    world = get_world_group()
    payload = (int(dist.get_rank()), pjrt_order.index(device_id))
    gathered: list[tuple[int, int] | None] = [None] * int(world.world_size)
    dist.all_gather_object(gathered, payload, group=world.cpu_group)

    rank_to_position: dict[int, int] = {}
    for item in gathered:
        if item is None:
            raise RuntimeError(
                "PCP device position gather returned an empty entry.")
        rank, position = item
        rank_to_position[int(rank)] = int(position)
    return rank_to_position


def _order_rings_by_device(
        rings: list[list[int]],
        rank_to_position: dict[int, int]) -> list[list[int]]:
    """Relabel the rings so ring position i is the rank on the i-th chip.

    The host slices tokens by ``rank_in_group`` (pcp_sequence_layout), while a
    kernel indexes the per-rank schedule it was handed with
    ``lax.axis_index`` -- a partition id. GDN does this literally:
    ``stage_metadata_ref.rank_active_row_end[rank]`` in
    ``kernels/gdn/v3/pcp_wrapper.py``. The two are the same number only when
    ring position == partition id. A remap used to force that and #414 removed
    it, on the correct grounds that permuting ``LOCAL_RANK`` solves a data
    placement problem in the device binding layer. This does it the way
    ``tpu_rank_binding``'s docstring says it should be done -- "the fix belongs
    in the PCP group's rank order, not here" -- by moving the label instead of
    the chip. The binding is untouched; only which ring position each rank
    occupies changes.

    One permutation of the positions is applied to every ring, never one sort
    per ring. A rank's TP partners are the ranks at its position in the other
    rings, so sorting rings independently would scatter a lane across
    positions and break the invariant this module's docstring is about.
    """
    if not rings:
        return rings
    missing = sorted({
        rank
        for ring in rings for rank in ring if rank not in rank_to_position
    })
    if missing:
        raise RuntimeError(
            f"No device position was gathered for PCP ranks {missing}; "
            f"gathered={rank_to_position}.")
    order = sorted(range(len(rings[0])),
                   key=lambda index: rank_to_position[rings[0][index]])
    return [[ring[index] for index in order] for ring in rings]


def resolve_pcp_topology_order(
        vllm_config) -> dict[str, list[list[int]]] | None:
    """Groups to build for each axis, or ``None`` if PCP does not apply.

    Returns ``{"pcp": rings, "tp": lanes}`` -- the ``group_ranks`` vLLM
    should build for each ``group_name``, taken from the topology mesh.

    ``None`` means only one thing: this configuration does not use topology
    ordering at all. Anything that goes *wrong* raises, because a silent
    fallback to the arithmetic layout would hide a real problem behind a small
    and unmeasurable slowdown.
    """
    parallel_config = vllm_config.parallel_config
    pcp_size = int(parallel_config.prefill_context_parallel_size)

    # Not applicable: no ring exists.
    if pcp_size <= 1:
        return None

    # Not applicable: PCP runner path does not support TPU multihost yet.
    nnodes = int(parallel_config.nnodes)
    if nnodes > 1:
        logger.info(
            "Topology aware mesh not applied: nnodes=%d. Topology aware "
            "mesh is currently only supported for single host", nnodes)
        return None

    dp_size = int(parallel_config.data_parallel_size)
    tp_size = int(parallel_config.tensor_parallel_size)
    pp_size = int(parallel_config.pipeline_parallel_size)

    # PcpStaticSupportValidator rejects PCP with pipeline parallelism before
    # any worker spawns, so reaching here with pp > 1 means that gate moved or
    # broke. Fail rather than build a layout whose pp axis we never modelled.
    if pp_size != 1:
        raise RuntimeError(
            f"Topology aware mesh assignment reached with "
            f"pipeline_parallel_size={pp_size}. "
            f"PcpStaticSupportValidator should have rejected PCP "
            f"with pipeline parallelism before worker startup.")

    # dp > 1 is only visible here under expert parallelism: TPUWorker
    # collapses data_parallel_size to 1 for dense DP, and preserves it only
    # for EP, which needs one unified DP*TP world for its expert collectives.
    # Either way vLLM's ParallelConfig rejects pcp>1 together with dp>1
    # before any worker starts, so reaching here means that gate moved or was
    # lifted -- and the mesh's dp axis has never been exercised.
    if dp_size != 1:
        raise RuntimeError(
            f"Topology aware mesh assignment reached with "
            f"data_parallel_size={dp_size}. vLLM's ParallelConfig rejects "
            f"prefill_context_parallel_size>1 together with "
            f"data_parallel_size>1 ('PCP does not support data parallelism "
            f"yet'), so reaching here means that gate moved or was lifted. "
            f"The mesh's dp axis has never been exercised.")

    # Ordered by increasing network intensity, matching vLLM's own
    # (dp, pp, pcp, tp) decomposition. jax maps the last axis onto the
    # highest-bandwidth link, which on v7 is the intra-chip core axis.
    mesh_shape = (dp_size, pcp_size, tp_size)

    import torch

    from vllm_torchtpu import envs as tpu_envs
    if not tpu_envs.TPU_PCP_TOPOLOGY_AWARE_MESH:
        logger.info(
            "Topology aware mesh not applied: TPU_PCP_TOPOLOGY_AWARE_MESH=0 "
            "(device-order PCP/TP layout).")
        return None
    if not hasattr(torch.tpu, "topology_aware_mesh"):
        raise RuntimeError(
            "torch.tpu.topology_aware_mesh is not available in the installed "
            "torch_tpu (needs >= 0.1.1.dev20260813160135); set "
            "TPU_PCP_TOPOLOGY_AWARE_MESH=0 to use the device-order layout.")
    mesh = torch.tpu.topology_aware_mesh(mesh_shape)

    # Keyed by vLLM group_name so the caller can look up the groups for the
    # axis it is building. A ring is mesh[d, :, t] -- reading down the pcp
    # axis, hence the swap. A lane is mesh[d, p, :] -- along it, already
    # memory order. int() because these reach torch.distributed.new_group.
    group_ranks_by_name = {
        "pcp": [[int(r) for r in ring]
                for ring in numpy.swapaxes(mesh, 1, 2).reshape(-1, pcp_size)],
        "tp": [[int(r) for r in lane]
               for lane in numpy.asarray(mesh).reshape(-1, tp_size)],
    }

    # topology_aware_mesh promises a permutation of the global ranks. If it is
    # not one, some rank is missing or duplicated and the groups we build from
    # it would silently drop or double a worker.
    expected = list(range(dp_size * pcp_size * tp_size))
    for axis, groups in group_ranks_by_name.items():
        flat = sorted(r for group in groups for r in group)
        if flat != expected:
            raise RuntimeError(
                f"topology_aware_mesh({mesh_shape}) produced {axis} groups "
                f"{groups} whose ranks {flat} are not a permutation of "
                f"{expected}.")

    # Sorting the ring by chip position is what keeps `lax.axis_index` equal
    # to `rank_in_group`. It is applied on top of the topology order rather
    # than instead of it: the mesh still decides *which* ranks share a ring
    # (and, under tp>1, which lane), this only decides where in the ring each
    # one sits. With one ring the two orderings cannot both be honoured, and
    # correctness wins -- the ring becomes chip-enumeration order, which on
    # 2x2x1 costs 6 ICI hops rather than 4. That trade goes away once the
    # DeviceAssignment can follow the mesh (google-pytorch/torch_tpu#3522
    # plus the collectives half of it), at which point this reordering should
    # be deleted rather than kept.
    rank_to_position = _rank_to_device_position()
    group_ranks_by_name["pcp"] = _order_rings_by_device(
        group_ranks_by_name["pcp"], rank_to_position)
    for ring in group_ranks_by_name["pcp"]:
        logger.info("PCP ring by device | ranks=%s chip_positions=%s", ring,
                    [rank_to_position[rank] for rank in ring])
    if tp_size != 1:
        # Each ring is now a proper subgroup of the slice, so its positions
        # run 0..pcp_size-1 while partition ids run 0..world_size-1. No
        # ordering can make those equal; only a DeviceAssignment built from
        # the mesh can. Said here rather than left to the reader, because the
        # reordering above still runs and still looks like it did something.
        logger.warning(
            "PCP ring reordered by chip position, but tensor_parallel_"
            "size=%d means a ring does not span the slice: partition ids "
            "run 0..%d while ring positions run 0..%d, so they cannot be "
            "made equal here. verify_pcp_topology_order will reject this.",
            tp_size, dp_size * pcp_size * tp_size - 1, pcp_size - 1)

    logger.info("Topology mesh | mesh_shape=%s -> %s", mesh_shape,
                group_ranks_by_name)
    return group_ranks_by_name


@contextlib.contextmanager
def pcp_topology_order(
        group_ranks_by_name: dict[str, list[list[int]]] | None
) -> Iterator[None]:
    """Install the mesh's groups while model-parallel groups are built.

    Patches ``init_model_parallel_group`` for the duration rather than
    permanently, so nothing is left installed for later callers to trip over.
    Only the axes present in ``group_ranks_by_name`` are replaced -- today
    ``pcp`` and ``tp``. Every other group, including ``dp``, ``pp`` and
    ``ep``, goes through the original function untouched; at the supported
    scope those are singletons or the whole world, so they cannot conflict
    with this layout.

    ``group_ranks_by_name`` is ``None`` only when topology aware mesh does
    not apply to this configuration. The patch is still installed in that
    case, because it also records what vLLM builds on its own: without it a
    baseline run logs no groups at all and there is nothing to diff a
    topology run against.
    """
    import torch
    from vllm.distributed import parallel_state

    original = parallel_state.init_model_parallel_group
    replacements = group_ranks_by_name or {}

    # torch.distributed.new_group sorts its rank list unless sort_ranks=False
    # ("if sort_ranks: ranks = sorted(ranks)"), so without the wrapper below
    # the order resolved above is discarded before any collective sees it --
    # while vLLM still derives rank_in_group from the unsorted list it was
    # handed ("ranks.index(self.rank)"). The two then disagree and every shard
    # an all_gather assembles lands in the wrong slot.
    # The wrapper below turns that sort off, and only for the axes replaced
    # above.
    preserving = False

    def patched(group_ranks, local_rank, backend, *args, **kwargs):
        nonlocal preserving
        axis = kwargs.get("group_name")
        # Logged for every axis, whether or not it is being replaced, so the
        # log shows vLLM's own layout even when the mesh does not apply.
        logger.info("vLLM layout | %s: vLLM built %s", axis,
                    [list(g) for g in group_ranks])
        replacement = replacements.get(axis) if axis else None
        if replacement is not None:
            logger.info("Topology layout | %s: replaced with %s", axis,
                        replacement)
            group_ranks = replacement

        preserving = replacement is not None
        try:
            return original(group_ranks, local_rank, backend, *args, **kwargs)
        finally:
            # Cleared even if the delegation raises, so no later call can see
            # a stale True.
            preserving = False

    def new_group_preserving_order(*args, **kwargs):
        if preserving:
            kwargs.setdefault("sort_ranks", False)
        return original_new_group(*args, **kwargs)

    # Only installed when there is something to replace, so configurations
    # that opt out of topology ordering never see it at all.
    patch_new_group = bool(replacements)
    original_new_group = torch.distributed.new_group
    if patch_new_group:
        if "sort_ranks" not in inspect.signature(
                original_new_group).parameters:
            raise RuntimeError(
                "torch.distributed.new_group has no sort_ranks parameter on "
                "this build, so the resolved rank order cannot survive group "
                "construction. Ordering would be silently discarded.")
        torch.distributed.new_group = new_group_preserving_order

    parallel_state.init_model_parallel_group = patched
    try:
        yield
    finally:
        parallel_state.init_model_parallel_group = original
        if patch_new_group:
            torch.distributed.new_group = original_new_group


def _verify_ring_sits_on_chips_in_order() -> None:
    """Fail startup unless ring position i is the worker on the i-th chip.

    ``lax.axis_index`` returns a partition id, and the host builds its
    per-rank tables by ``rank_in_group``. Every PCP kernel reads one with the
    other -- GDN indexes its stage schedule with ``lax.axis_index`` directly,
    in ``kernels/gdn/v3/pcp_wrapper.py`` -- so the two have to be the same
    number. They were, by construction, until #414 removed the remap that
    arranged it, and nothing has checked since.

    Nothing checking is the reason it mattered. A permuted schedule keeps
    every shape correct, completes every collective, and changes no timing;
    it reached end-to-end output twice before anyone could say which number
    was wrong. This is the check that turns that into a startup failure.

    Distinct from the verification above, which asks whether vLLM adopted the
    order this module requested. This asks whether that order is the right
    one, and so runs even when no order was requested at all.
    """
    from vllm.distributed.parallel_state import get_pcp_group

    try:
        group = get_pcp_group()
    except (AssertionError, ImportError):
        return
    if group is None or int(group.world_size) <= 1:
        return

    ranks = [int(rank) for rank in group.ranks]
    rank_to_position = _rank_to_device_position()
    missing = [rank for rank in ranks if rank not in rank_to_position]
    if missing:
        raise RuntimeError(
            f"No chip position was gathered for PCP ranks {missing}, so "
            f"whether the ring matches the partition order is unknown.")

    positions = [rank_to_position[rank] for rank in ranks]
    if positions == list(range(len(ranks))):
        logger.info("PCP ring | verified position == partition for %s", ranks)
        return

    raise RuntimeError(
        "The PCP group does not sit on the chips in enumeration order, so "
        "lax.axis_index inside a kernel is not rank_in_group and every "
        "per-rank table the host builds is read by the wrong worker. Ring "
        f"{ranks} sits on chip positions {positions}, which should have been "
        f"{list(range(len(ranks)))}. resolve_pcp_topology_order sorts the "
        "ring to match, so reaching here means the ring came from somewhere "
        "it does not run -- TPU_PCP_TOPOLOGY_AWARE_MESH=0, or a PCP group "
        "smaller than the slice, which cannot satisfy this at all because "
        "its positions run 0..pcp_size-1 while partition ids run over every "
        "chip.")


def verify_pcp_topology_order(
        group_ranks_by_name: dict[str, list[list[int]]] | None) -> None:
    """Assert vLLM adopted the mesh's order, by reading the built groups back.

    ``pcp_topology_order`` substitutes rank lists on their way into
    ``init_model_parallel_group``. That is a statement about the call, not
    about the result. This checks the result.

    Verifying the outcome rather than the patch is the point. The patch can
    stop applying without failing: upstream may rename ``group_name``, pass it
    positionally, build the groups from somewhere other than
    ``init_model_parallel_group``, or rebuild them after this module has
    restored the original. Each leaves the ring on vLLM's arithmetic layout,
    costing a few percent and raising nothing. Reading the group back catches
    all of them without needing to know which one happened, and keeps this
    module's promise that a fallback to the arithmetic layout is an error
    rather than an unmeasurable slowdown.

    ``GroupCoordinator.ranks`` is the same attribute ``get_pcp_group_layout``
    builds the device mesh from, so this asserts the literal input to the
    kernel's ring rather than a proxy for it.

    ``_verify_ring_sits_on_chips_in_order`` below runs whether or not any
    order was requested, because the invariant it checks is not this module's
    to opt out of.
    """
    _verify_ring_sits_on_chips_in_order()

    if not group_ranks_by_name:
        return

    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_pcp_group, get_tp_group

    getters = {"pcp": get_pcp_group, "tp": get_tp_group}
    my_rank = int(dist.get_rank())

    for axis, wanted in group_ranks_by_name.items():
        getter = getters.get(axis)
        if getter is None:
            raise RuntimeError(
                f"No group accessor is known for axis {axis!r}. Every axis "
                f"pcp_topology_order replaces must also be readable back, "
                f"otherwise it is applied without being verified.")

        expected = next((group for group in wanted if my_rank in group), None)
        if expected is None:
            raise RuntimeError(
                f"Rank {my_rank} does not appear in the {axis} groups "
                f"{wanted} resolved from the topology mesh.")

        # Compared as a sequence, not a set: the ring is the order, and a
        # group with the right members in the wrong order is the exact
        # failure this exists to catch.
        actual = [int(rank) for rank in getter().ranks]
        if actual != expected:
            raise RuntimeError(
                f"The {axis} group built for rank {my_rank} is {actual}, but "
                f"the topology mesh asked for {expected}. vLLM did not adopt "
                f"the requested order, so the ring is running on a layout "
                f"this module did not choose.")

        logger.info("Topology layout | %s: verified %s", axis, actual)
