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
by opening a chip -- which is the choice itself. On one host,
``topology_aware_mesh`` needs open chips and an initialised process group. On
multiple hosts, every worker must first report the global device id it opened.
Both answers are therefore available only once placement is fixed. Reordering
the logical groups is the one lever still available at that point.

Both the PCP and TP groups come from the same mesh, and they replace vLLM's
arithmetic layout rather than being reconciled with it. ``mesh[d, :, t]`` is a
PCP ring, ``mesh[d, p, :]`` is a TP lane, so one tensor gives every rank a
single ``(dp, pcp, tp)`` position: a rank's TP partners are by construction the
ranks sharing its ``pcp`` coordinate, and they therefore hold the same sequence
chunk. Ordering PCP alone would leave TP on vLLM's arange layout, and where the
two differ TP partners land at different ring positions -- holding different
sequence chunks while the TP all-reduce combines partials computed over
different tokens. Taking both axes from one mesh makes that unrepresentable.

At the supported ``dp == 1, pp == 1`` scope, DP and PP groups are singletons.
For an MoE model using expert parallelism, vLLM derives ``ep_rank`` by
flattening ``(pcp_rank, tp_rank)``. The EP group (and EPLB group when enabled)
must therefore use the same flattened mesh order as PCP and TP; otherwise
expert ownership and the communication group's rank disagree.

"""

import contextlib
import inspect
from collections.abc import Iterator

import numpy

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _collect_world_rank_to_device_id() -> dict[int, int]:
    """Collect the physical TPU device selected by every world rank."""
    from vllm.distributed.parallel_state import get_world_group

    from vllm_torchtpu.distributed.mesh_utils import (
        _collect_rank_to_device_id, _get_current_global_rank,
        _get_tpu_global_device_id)

    world_group = get_world_group()
    return _collect_rank_to_device_id(world_group, _get_current_global_rank(),
                                      _get_tpu_global_device_id())


def _multihost_runtime_rank_mesh(
        mesh_shape: tuple[int, int, int]) -> numpy.ndarray:
    """Return world ranks in TorchTPU custom-op partition order.

    A multi-host Mosaic custom op assigns its flattened partitions in
    increasing global-device-id order.  vLLM assigns ``(pcp, tp)`` coordinates
    from world-rank arithmetic, while process-to-device binding is not required
    to preserve that order.  Build the logical rank mesh from the observed
    binding so vLLM's coordinates describe the partitions the custom op will
    actually execute.
    """
    expected_world_size = int(numpy.prod(mesh_shape))
    expected_ranks = set(range(expected_world_size))
    rank_to_device_id = _collect_world_rank_to_device_id()
    actual_ranks = set(rank_to_device_id)
    if actual_ranks != expected_ranks:
        missing = sorted(expected_ranks - actual_ranks)
        extra = sorted(actual_ranks - expected_ranks)
        raise RuntimeError(
            "Multi-host PCP device mapping does not cover the model-parallel "
            f"world exactly once: missing_ranks={missing}, "
            f"extra_ranks={extra}, gathered={rank_to_device_id}.")

    device_ids = list(rank_to_device_id.values())
    if len(set(device_ids)) != expected_world_size:
        raise RuntimeError(
            "Multi-host PCP device mapping contains duplicate global device "
            f"ids: gathered={rank_to_device_id}.")

    ranks_in_runtime_order = sorted(expected_ranks,
                                    key=rank_to_device_id.__getitem__)
    return numpy.asarray(ranks_in_runtime_order).reshape(mesh_shape)


def resolve_pcp_topology_order(
        vllm_config) -> dict[str, list[list[int]]] | None:
    """Groups to build for each axis, or ``None`` if PCP does not apply.

    Returns the ``group_ranks`` vLLM should build for each affected
    ``group_name``, taken from one topology mesh. PCP and TP are always
    present; EP and EPLB are included when expert parallelism uses them.

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

    nnodes = int(parallel_config.nnodes)
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

    if nnodes > 1:
        mesh = _multihost_runtime_rank_mesh(mesh_shape)
        logger.info(
            "Multi-host runtime mesh | nnodes=%d | rank_to_device order=%s",
            nnodes,
            numpy.asarray(mesh).reshape(-1).tolist(),
        )
    else:
        import torch

        from vllm_torchtpu import envs as tpu_envs
        if not tpu_envs.TPU_PCP_TOPOLOGY_AWARE_MESH:
            logger.info("Topology aware mesh not applied: "
                        "TPU_PCP_TOPOLOGY_AWARE_MESH=0 "
                        "(device-order PCP/TP layout).")
            return None
        if not hasattr(torch.tpu, "topology_aware_mesh"):
            raise RuntimeError(
                "torch.tpu.topology_aware_mesh is not available in the "
                "installed torch_tpu (needs >= "
                "0.1.1.dev20260813160135); set "
                "TPU_PCP_TOPOLOGY_AWARE_MESH=0 to use the device-order "
                "layout.")
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
    model_config = getattr(vllm_config, "model_config", None)
    uses_ep = (bool(getattr(model_config, "is_moe", False)) and bool(
        getattr(parallel_config, "enable_expert_parallel", False)))
    if uses_ep:
        flat_mesh = [int(r) for r in numpy.asarray(mesh).reshape(-1)]
        group_ranks_by_name["ep"] = [flat_mesh]
        if bool(getattr(parallel_config, "enable_eplb", False)):
            group_ranks_by_name["eplb"] = [flat_mesh]

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
    Only the axes present in ``group_ranks_by_name`` are replaced. PCP and TP
    always come from the same mesh. MoE expert parallelism also supplies EP
    and, when enabled, EPLB from that mesh's flattened order. Every other group
    goes through the original function untouched.

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
    """
    if not group_ranks_by_name:
        return

    import torch.distributed as dist
    from vllm.distributed.parallel_state import (get_ep_group, get_eplb_group,
                                                 get_pcp_group, get_tp_group)

    getters = {
        "pcp": get_pcp_group,
        "tp": get_tp_group,
        "ep": get_ep_group,
        "eplb": get_eplb_group,
    }
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
