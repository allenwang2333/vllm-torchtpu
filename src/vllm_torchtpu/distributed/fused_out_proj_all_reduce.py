# SPDX-License-Identifier: Apache-2.0
"""Fused out-projection + two-shot all-reduce, T=1 decode only.

The kernel (kernels/collectives/fused_out_proj_all_reduce.py) fuses the
o_proj matmul and the cross-rank reduction into one custom call, keeping
the payload in its packed [tiles, 2, 128] tile layout end to end. This
module is the torch-facing half, and it follows the same rules as the
two-shot low-latency all-reduce it derives from (see
distributed/low_latency_all_reduce.py's module docstring for the full
post-mortems):

* ONE opaque custom op whose signature never changes --
  ``fused_out_proj_ar_<width>(x, w, rs, ag, slot, barrier_first)`` -- so
  dynamo never sees a branch on the token count. T=1 is the only wired
  bucket, so there is one workspace pair, not a list.
* ``slot`` alternates per layer and ``barrier_first`` is set on the first
  layer of every execution; both are compile-time properties of the
  callsite, baked into per-variant inner ops because static args cannot
  flow through sharded_jax_op.
* The workspaces are per-call scratch, fully rewritten before every read;
  they are arguments (not captures) and are never written back.

Integration status: the op expects bf16 weight shards. K3's o_proj weights
are compressed-tensors quantized, so wiring this into the model needs an
in-kernel dequant (or a bf16 weight copy) that does not exist yet -- this
module is built and unit-tested but nothing calls get_fused_out_proj_entry
yet. TPU_FUSED_OUT_PROJ_ALLREDUCE gates it and stays default-off.
"""

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Barrier-semaphore ids must not collide with another kernel that can be in
# flight at the same time. The fused-MoE kernel uses 0, the two-shot
# low-latency all-reduce uses 8.
_COLLECTIVE_ID_BASE = 16


def _replicated_spec():
    """See distributed/low_latency_all_reduce.py::_replicated_spec -- the
    operands are per-rank shards declared "replicated" so global shape ==
    local shape and shard_map hands the body this rank's own buffer. The
    fused op's output IS genuinely replicated (every rank assembles the same
    reduced payload)."""
    from jax.sharding import PartitionSpec

    return PartitionSpec()


def _build_fused_fn(mesh, num_devices: int, collective_id: int, slot: int,
                    barrier_first: int):
    """The shard_map-wrapped JAX fn for one (slot, barrier_first) variant.

    Signature is fixed -- ``(x, w, rs, ag)`` with x the [1, k_shard]
    activation shard and w the [width, k_shard] weight shard. The only
    non-custom-call op in the body is the packed [tiles, 2, 128] -> [1,
    width] reshape of the result (a small relayout copy; the two layouts
    tile differently). slot/barrier_first must be baked into the closure:
    torch_tpu's jax_placeholders zips operands with partition specs and
    silently drops trailing static args.
    """
    import jax

    from vllm_torchtpu.kernels.collectives.fused_out_proj_all_reduce import \
        fused_out_proj_all_reduce

    _REPLICATED = _replicated_spec()

    def _fn(x: jax.Array, w: jax.Array, rs: jax.Array,
            ag: jax.Array) -> jax.Array:

        def body(x, w, rs, ag):
            # The kernel returns the natural [1, width] layout directly --
            # see the kernel's assembly comment for why the packed form
            # must not cross the custom-call boundary (layout contagion).
            return fused_out_proj_all_reduce(x,
                                             w,
                                             rs,
                                             ag,
                                             slot=slot,
                                             barrier_first=barrier_first,
                                             axis_name="tp",
                                             num_devices=num_devices,
                                             collective_id=collective_id,
                                             has_side_effects=False)

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(_REPLICATED, ) * 4,
                             out_specs=_REPLICATED,
                             check_vma=False)(x, w, rs, ag)

    return _fn


# (width, k_shard, dtype, device) -> (op, rs, ag) or None once build failed.
_ENTRIES: dict[tuple, tuple | None] = {}


def get_fused_out_proj_entry(width: int, k_shard: int, dtype: torch.dtype):
    """The shared fused o_proj+reduce op for this TP group, or None.

    One entry per (width, k_shard, dtype). Returns None -- meaning the
    caller keeps the stock matmul + all-reduce -- when the flag is off, the
    group is trivial, the mesh cannot be built, or the shape fails the
    kernel's constraints (width must be a multiple of 256 with
    width // 256 <= world_size).

    Same construction-time rules as the two-shot op: call after the process
    groups exist and before torch.compile sees a forward.
    """
    from vllm_torchtpu import envs

    if not envs.TPU_FUSED_OUT_PROJ_ALLREDUCE:
        return None
    try:
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
    except Exception as e:  # noqa: BLE001
        logger.warning("Fused out-proj all-reduce unavailable (%s).", e)
        return None
    if communicator is None or int(getattr(communicator, "world_size", 1)) < 2:
        return None

    key = (width, k_shard, dtype, str(communicator.device))
    if key in _ENTRIES:
        return _ENTRIES[key]

    entry = None
    try:
        entry = _build_entry(communicator, width, k_shard, dtype)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not build the fused out-proj all-reduce for width %d "
            "(%s); o_proj keeps the stock matmul + XLA ring.", width, e)
        entry = None
    _ENTRIES[key] = entry
    return entry


def _build_entry(communicator, width: int, k_shard: int,
                 dtype: torch.dtype) -> tuple | None:
    """Build the opaque op and its workspace pair for (width, k_shard)."""
    import torch.distributed as dist

    from vllm_torchtpu.distributed.low_latency_all_reduce import _TpMesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
    from vllm_torchtpu.kernels.collectives.fused_out_proj_all_reduce import (
        LANES, PACK)

    num_devices = int(communicator.world_size)
    tiles = width // (PACK * LANES)
    if width % (PACK * LANES) != 0 or tiles > num_devices:
        logger.info(
            "Fused out-proj all-reduce: width %d over %d devices fails the "
            "kernel's constraints (width %% %d == 0, tiles <= devices); "
            "o_proj keeps the stock path.", width, num_devices, PACK * LANES)
        return None
    mesh = _TpMesh(communicator).get()
    if mesh is None:
        return None

    # One inner op per (slot, barrier_first) combination that occurs: the
    # first layer of an execution is (0, 1); every other layer is (0, 0) or
    # (1, 0) alternating.
    _REPLICATED = _replicated_spec()
    inner_map = {}
    for slot, first in ((0, 1), (0, 0), (1, 0)):
        fn = _build_fused_fn(mesh, num_devices, _COLLECTIVE_ID_BASE, slot,
                             first)
        inner_map[(slot, first)] = sharded_jax_op(
            f"vllm_torchtpu::fused_out_proj_ar_{width}_{k_shard}_"
            f"s{slot}f{first}",
            fn,
            mesh=mesh,
            input_partition_specs=(_REPLICATED, ) * 4,
            output_partition_specs=_REPLICATED,
        )

    device = torch.device("tpu")
    rs = torch.zeros((2, num_devices, PACK, LANES), dtype=dtype, device=device)
    ag = torch.zeros((2, tiles + 1, PACK, LANES), dtype=dtype, device=device)

    def xla_fallback(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Off the wired bucket: stock path, byte-for-byte what
        # RowParallelLinear would have done. x @ w is the partial from this
        # rank's shards (w is [k_shard, width], contracting dim first -- the
        # torch_tpu RowParallelLinear layout); the ring reduction sums them.
        out = torch.matmul(x, w)
        dist.all_reduce(out, group=communicator.device_group)
        return out

    op = _build_outer_op(f"vllm_torchtpu::fused_out_proj_ar_{width}_{k_shard}",
                         inner_map,
                         xla_fallback,
                         width=width)

    logger.info(
        "Fused out-proj all-reduce: T=1 o_proj reduces at width %d take the "
        "fused two-shot kernel (%d devices); other token counts keep the "
        "stock path inside the same op.", width, num_devices)
    return op, rs, ag


def _build_outer_op(name: str, inner_map: dict, xla_fallback, *, width: int):
    """The single opaque custom op a layer calls on every bucket.

    Same rules as low_latency_all_reduce._build_outer_op: dynamo must not
    see the token-count switch (reading a dynamic dim in traced Python pins
    it and fails the whole compilation); the custom-op boundary keeps the
    read at execution time, where it is an ordinary int. Only T=1 is wired:
    the kernel implements single-token decode (tiles <= num_devices), and
    every rank executes the same layer sequence with the same padded token
    count, so the branch is unanimous and the kernel's slot parity stays
    consistent.

    mutates_args stays empty even though the kernel writes its workspace
    operands: they are private per-call scratch, fully rewritten before
    every read, read by nothing else in the graph. See
    low_latency_all_reduce._build_outer_op for why declaring the mutation
    breaks the compiled artifact's output count.
    """

    def _impl(x: torch.Tensor, w: torch.Tensor, rs: torch.Tensor,
              ag: torch.Tensor, slot: int, barrier_first: int) -> torch.Tensor:
        if x.shape[0] == 1:
            return inner_map[(slot, barrier_first)](x, w, rs, ag)
        return xla_fallback(x, w)

    op = torch.library.custom_op(name, _impl, mutates_args=())
    op.register_fake(lambda x, w, rs, ag, slot, barrier_first: torch.empty(
        (x.shape[0], width), dtype=x.dtype, device=x.device))
    return op
