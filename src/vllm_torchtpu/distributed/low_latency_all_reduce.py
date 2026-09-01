# SPDX-License-Identifier: Apache-2.0
"""Two-shot low-latency all-reduce for decode-sized TP payloads.

XLA lowers an all-reduce to a *ring* reduce-scatter + all-gather: bandwidth
optimal, but ``2(N-1)`` sequential hops. At decode payloads the hop count, not
the bytes, is the cost. ``kernels/collectives/low_latency_all_reduce.py``
``two_shot_all_reduce`` trades egress volume for hops -- 2 instead of
``2(N-1)`` -- and measures 1.43x XLA at the c=4 payload ([4, 7168] bf16,
56 KiB, 32 devices: 19.86 us -> 13.88 us), winning out to ~112 KiB, tying at
224 KiB, and losing beyond.

Unlike the earlier integration branch (gnovack/llar-k3-integration), the
kernel-or-XLA choice is NOT made in traced Python. This vLLM stack traces the
backbone once with the token dim dynamic and drops all guards, so a dict
lookup on the token count -- or any read of a traced dim -- raises
ConstraintViolationError at profile_run, and wiring workspaces per bucket
gives compile_all_ranges' single FX graph a different interface per bucket
(ShapeSpecializationError; see gnovack/llar-uniform-op's commit message for
the full post-mortem). Instead there is ONE opaque custom op whose signature
never changes:

    llar_out_proj_reduce_<width>(x, rs_list, ag_list) -> Tensor

The lists hold ONE workspace pair per wired token bucket, so every bucket
presents the same graph interface (same tensors, same order, every call) yet
each bucket's JAX module is fully static -- no runtime slicing of a max-size
workspace. The op's *implementation* -- which runs with concrete shapes,
invisible to dynamo -- routes: for a wired bucket the payload goes through
the two-shot kernel over a multi-device shard_map, otherwise through an
ordinary ``dist.all_reduce``. Every rank executes the same layer sequence
with the same padded token count, so the branch is unanimous and the kernel's
slot parity stays consistent.

Batch R (arm-1) measured the cost of doing this the other way -- one max-size
workspace sliced per shape in the JAX body: +1.33 ms/step of on-device glue
(2 slices + 2 dynamic-update-slice fusions + relayout reshapes/reduces per
layer, plus writeback copies) against a kernel win of only ~0.3 ms/step. So
this revision keeps the JAX body a bare custom-call:

* workspaces are per-bucket and exactly sized -- nothing to slice or update;
* the kernel takes the payload in its natural [tokens, width] layout
  (kernels/collectives/low_latency_all_reduce.py::two_shot_all_reduce_natural),
  so the [T, 7168] -> [32, T*224] flatten -- a real relayout copy in both
  directions, because the two shapes tile differently -- disappears;
* the kernel drives its slot rotation from a STATIC argument -- the layer's
  ordinal among the 93 callsites, known at compile time
  (kernels/collectives/low_latency_all_reduce.py::two_shot_all_reduce_static).
  A device-side step counter cannot work on this stack: XLA stages custom-call
  operands into fresh VMEM/SMEM buffers per call, so a counter written through
  its input ref never reaches the persistent buffer (measured on device). And
  the counter is unnecessary: consecutive calls alternate slots, and the first
  callsite of every execution barriers before sending, which is the whole
  protocol. The workspaces carry no value across calls -- every slot is fully
  rewritten before it is read -- so they are NOT returned, NOT donated, and
  never written back with ``copy_``: the JAX body is a bare custom-call.

The kernels need a mesh, and vLLM has none. Each vLLM worker drives one core
and reduces through ``torch.distributed``; a Pallas collective instead needs a
``shard_map`` over every peer. Inside a worker ``jax.devices()`` exposes the
whole slice even though ``jax.local_devices()`` is one core, so a mesh can be
built over the group's TPU *global device ids* -- the same trick
``distributed/pcp.py`` uses for context-parallel attention. Rank order is not
device-id order, so the mapping is gathered rather than assumed.
"""

# NOTE: deliberately no `from __future__ import annotations` anywhere a
# wrapper is handed to jax_op-style registration: it validates signatures by
# inspecting real annotation *objects*, and under PEP 563 they arrive as
# strings it cannot resolve.
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Barrier-semaphore ids must not collide with another kernel that can be in
# flight at the same time. The fused-MoE kernel uses 0, so start clear of it.
_COLLECTIVE_ID_BASE = 8

# Dispatch windows in bytes, read off the measured sweep in the kernel's
# design doc (bfloat16). two_shot loses to XLA's ring once the payload is
# large enough for bandwidth to dominate hop count, so each entry is a window
# rather than a ceiling.
#
#   (two_shot_min, two_shot_max)
#
# Measured, device self time, us:
#
#            payload    XLA   two_shot
#    8 dev    14 KiB  10.12       7.76
#    8 dev    42 KiB  10.61       8.21
#   32 dev    14 KiB  17.31      12.36
#   32 dev    28 KiB  17.97      12.67
#   32 dev    42 KiB  18.93      13.20
#   32 dev   112 KiB  21.88      17.38
#   32 dev   224 KiB  23.27      25.69   <- past the crossover
#
# The 32/16-device ceiling is 128 KiB: past ~112 KiB two_shot goes linear at
# +8.5 us per 112 KiB while XLA's ring amortises, and they cross near
# 150 KiB. The 8-device ceiling is 64 KiB: 8 devices is only measured to
# 42 KiB, and 64 KiB is one modest step past it. 2 and 4 devices are absent
# deliberately: the kernel's win is collapsing 2(N-1) hops to 2, which is
# nothing at N=2 and one hop at N=4.
_GATE: dict[int, tuple[int, int]] = {
    8: (0, 64 * 1024),
    16: (0, 128 * 1024),
    32: (0, 128 * 1024),
}

# Default for TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE. At 32 devices the
# _GATE ceiling binds first at the widths K3 reduces (128 KiB is ~9 tokens of
# a 7168-wide bf16 payload), so this default governs only which *buckets*
# workspaces are wired for. 8 is the derived crossover: T=16 is the measured
# tie point (224 KiB) and is deliberately NOT wired by default.
_DEFAULT_MAX_BATCH_SIZE = 8

# The natural-shape kernel chunks the payload along the flat row-major index,
# which on the [tokens, width] layout lands on whole lane tiles only when the
# per-device chunk is a multiple of 128 elements (see
# two_shot_all_reduce_natural). Buckets that fail this are never wired; the
# fallback serves them. At width 7168 over 32 devices the wired buckets are
# the multiples of 4 -- 4 and 8 by default.
_LANE_TILE = 128


def max_batch_size() -> int:
    """Largest padded token count that may take the kernel.

    This is the knob that decides *which batches* go to the kernel; ``_GATE``
    decides which *payloads* may. Both have to pass, and in practice the gate
    is the binding one at the widths real models reduce. Expressed in padded
    tokens because that is what the kernel is compiled for. For decode, where
    this kernel is aimed, padded tokens is the batch size.
    """
    from vllm_torchtpu import envs

    raw = envs.TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE
    if not raw.strip():
        return _DEFAULT_MAX_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE=%s is not an integer; "
            "using %d.", raw, _DEFAULT_MAX_BATCH_SIZE)
        return _DEFAULT_MAX_BATCH_SIZE
    if value < 1:
        logger.warning(
            "TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE=%d is below 1, which "
            "would disable the kernel entirely; unset TPU_LOW_LATENCY_ALLREDUCE "
            "instead. Using %d.", value, _DEFAULT_MAX_BATCH_SIZE)
        return _DEFAULT_MAX_BATCH_SIZE
    return value


def _choose_kernel(num_devices: int, payload_bytes: int, numel: int) -> str:
    """Pick a kernel from the measured table, or ``"xla"`` to leave it alone."""
    gate = _GATE.get(num_devices)
    if gate is None:
        return "xla"
    two_shot_min, two_shot_max = gate
    if (two_shot_max and two_shot_min <= payload_bytes <= two_shot_max
            and numel % num_devices == 0):
        # two-shot splits the payload into num_devices contiguous chunks of
        # the flat row-major index; on the natural [tokens, hidden] layout
        # that is a lane-aligned column run per device whenever the bucket is
        # wired (see wired_buckets -- misaligned buckets are never built).
        return "two_shot"
    return "xla"


def _chunk_width(numel: int, width: int, num_devices: int) -> int:
    """Per-device chunk of this payload, or 0 if the kernel can't serve it.

    The kernel needs the chunk to be a whole number of lane tiles contained
    in one row of the natural [tokens, width] layout: ``width % w == 0``
    keeps every chunk inside a single row and ``w % 128 == 0`` keeps its
    transfers lane-aligned. w > width (more tokens than devices) would need
    multi-row chunks, which the natural kernel does not implement -- no K3
    decode bucket is there (tokens <= 32 = num_devices).
    """
    if numel % num_devices != 0:
        return 0
    w = numel // num_devices
    if w > width or width % w != 0 or w % _LANE_TILE != 0:
        return 0
    return w


def wired_buckets(width: int, dtype: torch.dtype, num_devices: int) -> list:
    """The token counts that get a kernel workspace, in increasing order.

    Every bucket from 1 to ``max_batch_size()`` that both clears the measured
    gate window and yields a lane-aligned chunk. Buckets not listed take the
    XLA fallback inside the op.
    """
    esize = torch.empty((), dtype=dtype).element_size()
    buckets = []
    for tokens in range(1, max_batch_size() + 1):
        numel = tokens * width
        if _chunk_width(numel, width, num_devices) == 0:
            continue
        if _choose_kernel(num_devices, numel * esize, numel) == "two_shot":
            buckets.append(tokens)
    return buckets


def wired_max_numel(width: int, dtype: torch.dtype, num_devices: int) -> int:
    """Largest payload, in elements, that this op routes to the kernel.

    0 means "no payload takes the kernel" and the op is never built.
    """
    buckets = wired_buckets(width, dtype, num_devices)
    return buckets[-1] * width if buckets else 0


class _TpMesh:
    """A JAX mesh over the TP group's devices, in rank-in-group order."""

    def __init__(self, communicator):
        self._comm = communicator
        self._world_size = int(communicator.world_size)
        self._mesh = None
        self._failed = False

    def _device_id_order(self) -> list[int] | None:
        """This group's TPU global device ids, in rank-in-group order."""
        import torch.distributed as dist
        from torch_tpu._internal.distributed import tpu_distributed

        try:
            my_device_id = int(tpu_distributed.global_device_id())
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Cannot read the TPU global device id (%s); "
                "low-latency all-reduce disabled.", e)
            return None

        gathered: list = [None] * self._world_size
        dist.all_gather_object(gathered,
                               (int(self._comm.rank_in_group), my_device_id),
                               group=self._comm.cpu_group)
        by_rank = {}
        for item in gathered:
            if item is None:
                return None
            rank_in_group, device_id = item
            by_rank[int(rank_in_group)] = int(device_id)
        if sorted(by_rank) != list(range(self._world_size)):
            logger.warning(
                "Incomplete rank->device map %s; low-latency all-reduce "
                "disabled.", by_rank)
            return None
        return [by_rank[r] for r in range(self._world_size)]

    def get(self):
        if self._mesh is not None or self._failed:
            return self._mesh
        try:
            import jax
            import numpy as np
            from jax.sharding import Mesh

            order = self._device_id_order()
            if order is None:
                self._failed = True
                return None
            by_id = {
                int(getattr(d, "id", i)): d
                for i, d in enumerate(jax.devices())
            }
            missing = [d for d in order if d not in by_id]
            if missing:
                # Expected whenever a process cannot address its peers'
                # devices; falling back to XLA is always correct.
                logger.warning(
                    "JAX does not expose device ids %s (sees %s); "
                    "low-latency all-reduce disabled.", missing, sorted(by_id))
                self._failed = True
                return None
            self._mesh = Mesh(np.asarray([by_id[d] for d in order]),
                              axis_names=("tp", ))
            logger.info(
                "Low-latency all-reduce: mesh over %d devices "
                "(rank-in-group order -> device ids %s).", self._world_size,
                order)
            return self._mesh
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not build the low-latency all-reduce mesh "
                "(%s); falling back to XLA.", e)
            self._failed = True
            return None


def _replicated_spec():
    """Partition spec declaring every operand replicated over the mesh.

    Counter-intuitive but deliberate. The operands are *not* replicated -- each
    rank holds its own partial sum, which is the entire point of reducing them.
    What this declares is the mapping between the torch tensor a rank passes in
    and the global array the shard_map sees, and "replicated" is the spec that
    makes the two the same shape.

    The alternative, ``P("tp")``, describes the array as sharded along its
    leading axis, so a [1, 7168] shard becomes a [32, 7168] global. torch_tpu
    then registers a module whose entry wants [32, 7168] and calls it with the
    rank's [1, 7168] tensor, and its specialization pass refuses:

        invalid refinement for argument 0, refinement dimension sizes must
        match for static dimensions in tensor<32x7168> -> tensor<1x7168>

    Declaring replication keeps global shape == local shape, so the entry
    signature matches what is actually passed, and ``shard_map`` hands the body
    each rank's own buffer -- exactly the manual semantics the kernel wants.
    Nothing reads the operands as though the ranks agreed: the only consumer is
    the Pallas kernel, which treats its argument as this device's shard.
    """
    from jax.sharding import PartitionSpec

    return PartitionSpec()


def _build_two_shot_fn(mesh, num_devices: int, collective_id: int, slot: int,
                       barrier_first: int):
    """The shard_map-wrapped JAX fn for one (slot, barrier_first) variant.

    The signature is fixed -- ``(x, rs, ag)`` with the workspaces in the
    bucket's exact shape -- and the body is a bare custom-call: the only
    other ops are the [T, width] -> [T, per_row, 1, w] view and back, which
    split/merge the minor dim at a lane-tile boundary (checked against the
    dumped HLO on the check job). slot and barrier_first are compile-time
    constants of the layer's ordinal; they MUST be baked into the closure
    rather than passed as static args: torch_tpu's jax_placeholders zips
    operands with partition specs, silently dropping trailing static args
    (measured: '_fn() missing 1 required positional argument'). The
    workspaces are per-call scratch; the kernel carries no runtime state
    across calls (two_shot_all_reduce_static).
    """
    import jax

    from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
        two_shot_all_reduce_static

    _REPLICATED = _replicated_spec()

    def _fn(x: jax.Array, rs: jax.Array, ag: jax.Array) -> jax.Array:

        def body(x, rs, ag):
            tokens, width = x.shape
            w = (tokens * width) // num_devices
            per_row = width // w
            out = two_shot_all_reduce_static(x.reshape(tokens, per_row, 1, w),
                                             rs,
                                             ag,
                                             slot=slot,
                                             barrier_first=barrier_first,
                                             axis_name="tp",
                                             num_devices=num_devices,
                                             collective_id=collective_id,
                                             has_side_effects=False)
            return out.reshape(x.shape)

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(_REPLICATED, ) * 3,
                             out_specs=_REPLICATED,
                             check_vma=False)(x, rs, ag)

    return _fn


# (width, dtype, device) -> (op, rs, ag, step), or None once building failed.
_ENTRIES: dict[tuple, tuple | None] = {}


def get_out_proj_reduce_entry(width: int, dtype: torch.dtype):
    """The shared out-projection reduce op for this TP group, or None.

    One entry per (width, dtype): every o_proj in the model reduces the same
    shape and calls it the same number of times in the same order on every
    rank, so they can share one workspace and one step counter. Returns None
    -- meaning the caller keeps the stock all-reduce -- when the flag is off,
    the group is trivial, the mesh cannot be built, or no payload at this
    width clears the gate.

    Must be called after the process groups exist and before torch.compile
    sees a forward: it registers torch custom ops, gathers a device map, and
    allocates persistent workspaces, none of which dynamo can trace. The
    runner calls it from ``_initialize_attention_kernels``, the same point the
    PCP/RPA kernels are built.
    """
    from vllm_torchtpu import envs

    if not envs.TPU_LOW_LATENCY_ALLREDUCE:
        return None
    try:
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
    except Exception as e:  # noqa: BLE001
        logger.warning("Low-latency all-reduce unavailable (%s).", e)
        return None
    if communicator is None or int(getattr(communicator, "world_size", 1)) < 2:
        return None

    key = (width, dtype, str(communicator.device))
    if key in _ENTRIES:
        return _ENTRIES[key]

    entry = None
    num_devices = int(communicator.world_size)
    buckets = wired_buckets(width, dtype, num_devices)
    if not buckets:
        logger.info(
            "Low-latency all-reduce: no payload at width %d clears the gate "
            "for %d devices (batch limit %d); out-proj reduces stay on XLA.",
            width, num_devices, max_batch_size())
    else:
        try:
            entry = _build_entry(communicator, width, dtype, buckets)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not build the low-latency all-reduce for width %d "
                "(%s); out-proj reduces stay on XLA.", width, e)
            entry = None
    _ENTRIES[key] = entry
    return entry


def _build_entry(communicator, width: int, dtype: torch.dtype,
                 buckets: list) -> tuple:
    """Build the opaque op and its per-bucket workspaces for (width, dtype)."""
    import torch.distributed as dist

    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    num_devices = int(communicator.world_size)
    mesh = _TpMesh(communicator).get()
    if mesh is None:
        return None

    # One inner op per (slot, barrier_first) combination that occurs: the
    # first layer of an execution is (0, 1); every other layer is (0, 0) or
    # (1, 0) alternating. Static args cannot flow through sharded_jax_op, so
    # each variant is its own op with the statics baked into its closure.
    _REPLICATED = _replicated_spec()
    inner_map = {}
    for slot, first in ((0, 1), (0, 0), (1, 0)):
        fn = _build_two_shot_fn(mesh, num_devices, _COLLECTIVE_ID_BASE, slot,
                                first)
        inner_map[(slot, first)] = sharded_jax_op(
            f"vllm_torchtpu::llar_two_shot_{width}_s{slot}f{first}",
            fn,
            mesh=mesh,
            input_partition_specs=(_REPLICATED, ) * 3,
            output_partition_specs=_REPLICATED,
        )

    device = torch.device("tpu")
    rs_list, ag_list = [], []
    bucket_index = {}
    for tokens in buckets:
        w = _chunk_width(tokens * width, width, num_devices)
        assert w > 0  # wired_buckets already filtered
        bucket_index[tokens] = len(rs_list)
        rs_list.append(
            torch.zeros((2, num_devices - 1, 1, w), dtype=dtype,
                        device=device))
        ag_list.append(
            torch.zeros((2, tokens, width // w, 1, w),
                        dtype=dtype,
                        device=device))

    def xla_fallback(x: torch.Tensor) -> torch.Tensor:
        # Off the wired buckets this is byte-for-byte what RowParallelLinear
        # would have done: the same XLA ring reduction over the same group.
        # The clone is forced by the custom-op contract -- the op declares no
        # mutated arguments, so it cannot reduce x in place.
        out = x.clone()
        dist.all_reduce(out, group=communicator.device_group)
        return out

    op = _build_outer_op(f"vllm_torchtpu::llar_out_proj_reduce_{width}",
                         inner_map,
                         width=width,
                         bucket_index=bucket_index,
                         xla_fallback=xla_fallback)

    logger.info(
        "Low-latency all-reduce: out-proj reduces at width %d take the "
        "two-shot kernel at token buckets %s (%d devices, batch limit %d "
        "tokens); other payloads keep the XLA ring, inside the same op.",
        width, buckets, num_devices, max_batch_size())
    return op, rs_list, ag_list


def _build_outer_op(name: str, inner_map: dict, *, width: int,
                    bucket_index: dict, xla_fallback):
    """The single opaque custom op a layer calls on every bucket.

    Dynamo must not see the switch. Selecting a kernel in traced Python means
    reading dims, and reading vLLM's dynamic token dimension pins it and fails
    the whole compilation ("Constraints violated (L['input_ids'].size()[0])").
    ``torch.library.custom_op`` is not traced into, so the graph gets one call
    node and the dims are read later, when they are ordinary ints -- which is
    what lets the branch below exist at all.

    The workspaces are *arguments* -- two parallel lists, one exactly-sized
    pair per wired bucket, so every token bucket presents the same graph
    interface while each bucket's JAX module is fully static. They cannot be
    captured inside the implementation: a tensor the compiled program
    references but was never handed aborts it with "identified an argument
    that was not provided: DeviceBufferRef".

    The kernel branch does NO workspace writeback. The workspaces are
    per-call scratch whose contents are fully rewritten before every read in
    a round (the slot rotation is driven by the static ``slot`` argument, not
    by runtime state), so there is no copy_ for XLA to pay per layer and no
    donation to rely on. ``slot`` and ``barrier_first`` come from the layer's
    ordinal: consecutive layers alternate slots, and the first layer of every
    execution barriers before sending.
    """

    def _impl(x: torch.Tensor, rs_list: list[torch.Tensor],
              ag_list: list[torch.Tensor], slot: int,
              barrier_first: int) -> torch.Tensor:
        # The token-count switch. x is a concrete tensor here -- the custom op
        # boundary is exactly what keeps this read out of the traced graph.
        idx = bucket_index.get(x.numel() // width)
        if idx is not None:
            inner = inner_map[(slot, barrier_first)]
            return inner(x, rs_list[idx], ag_list[idx])
        return xla_fallback(x)

    # mutates_args stays empty even though the *kernel* writes its staged
    # workspace operands. The mutation is invisible to torch and irrelevant
    # to XLA's value semantics: the buffers are pure scratch (rewritten
    # before every read) and no graph value depends on their contents.
    # Declaring a mutation would make AOTAutograd add the buffers to the
    # graph's
    # outputs, and the compiled artifact then returns one where the runtime
    # wrapper expects several:
    #
    #   AssertionError: expected 3 outputs, got 1
    #
    # Leaving it undeclared is a deliberate deviation from the custom-op
    # contract, and safe only because of what these buffers are: private to
    # this op, read by nothing else in the graph, and never aliased to a
    # value the compiler could fold. Two calls are also never identical --
    # each layer passes a different x -- so nothing can be CSE'd on the
    # assumption that the op is pure.
    op = torch.library.custom_op(name, _impl, mutates_args=())
    op.register_fake(
        lambda x, rs_list, ag_list, slot, barrier_first: torch.empty_like(x))
    return op
