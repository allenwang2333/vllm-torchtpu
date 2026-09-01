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

    llar_out_proj_reduce_<width>(x, rs, ag, step) -> Tensor

The workspaces are sized for the largest wired payload, so every token bucket
presents the same graph interface, and the op's *implementation* -- which runs
eagerly with concrete shapes, invisible to dynamo -- routes: at or below the
wired threshold the payload goes through the two-shot kernel over a
multi-device shard_map (the slice of the workspace it needs), above it through
an ordinary ``dist.all_reduce``. Every rank executes the same layer sequence
with the same padded token count, so the branch is unanimous and the kernel's
slot parity stays consistent.

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
# warmup-sized workspaces must cover. 8 is the derived crossover: T=16 is the
# measured tie point (224 KiB) and is deliberately NOT wired by default.
_DEFAULT_MAX_BATCH_SIZE = 8


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
        # two-shot scatters along rows. Flattening the payload and viewing it
        # as (num_devices, numel/num_devices) satisfies its divisibility
        # constraint for any token count, which the natural [tokens, hidden]
        # view would not (decode has far fewer rows than devices).
        return "two_shot"
    return "xla"


def wired_max_numel(width: int, dtype: torch.dtype, num_devices: int) -> int:
    """Largest payload, in elements, that this op routes to the kernel.

    The largest multiple-of-``num_devices`` element count that is both at or
    below ``max_batch_size()`` tokens of ``width`` and inside the measured
    gate window. 0 means "no payload takes the kernel" and the op is never
    built.
    """
    esize = torch.empty((), dtype=dtype).element_size()
    for tokens in range(max_batch_size(), 0, -1):
        numel = tokens * width
        if _choose_kernel(num_devices, numel * esize, numel) == "two_shot":
            return numel
    return 0


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


def _build_two_shot_fn(mesh, num_devices: int, collective_id: int):
    """The shard_map-wrapped JAX fn the inner op lowers per input shape.

    The signature is fixed -- ``(x, rs, ag, step)`` -- and the workspaces are
    sized for the largest wired payload, so one op serves every token bucket:
    for a concrete ``x`` the body slices the workspace down to this payload's
    column count and scatters the result back. ``x.size`` is a concrete int
    here (the inner op lowers per shape), so any per-shape decision would be
    ordinary trace-time Python; none is needed because the outer op only calls
    this for payloads the kernel serves.
    """
    import jax

    from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
        two_shot_all_reduce

    _REPLICATED = _replicated_spec()

    def _fn(x: jax.Array, rs: jax.Array, ag: jax.Array, step: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        def body(x, rs, ag, step):
            cols = x.size // num_devices
            flat = x.reshape(num_devices, cols)
            rs_slice = rs[..., :cols]
            ag_slice = ag[..., :cols]
            out, rs_slice, ag_slice, step = two_shot_all_reduce(
                flat,
                rs_slice,
                ag_slice,
                step,
                axis_name="tp",
                num_devices=num_devices,
                collective_id=collective_id,
                has_side_effects=False)
            rs = rs.at[..., :cols].set(rs_slice)
            ag = ag.at[..., :cols].set(ag_slice)
            return out.reshape(x.shape), rs, ag, step

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(_REPLICATED, ) * 4,
                             out_specs=(_REPLICATED, ) * 4,
                             check_vma=False)(x, rs, ag, step)

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
    max_numel = wired_max_numel(width, dtype, num_devices)
    if max_numel == 0:
        logger.info(
            "Low-latency all-reduce: no payload at width %d clears the gate "
            "for %d devices (batch limit %d); out-proj reduces stay on XLA.",
            width, num_devices, max_batch_size())
    else:
        try:
            entry = _build_entry(communicator, width, dtype, max_numel)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not build the low-latency all-reduce for width %d "
                "(%s); out-proj reduces stay on XLA.", width, e)
            entry = None
    _ENTRIES[key] = entry
    return entry


def _build_entry(communicator, width: int, dtype: torch.dtype,
                 max_numel: int) -> tuple:
    """Build the opaque op and its workspace for one (width, dtype)."""
    import torch.distributed as dist

    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    num_devices = int(communicator.world_size)
    mesh = _TpMesh(communicator).get()
    if mesh is None:
        return None
    cols_max = max_numel // num_devices

    fn = _build_two_shot_fn(mesh, num_devices, _COLLECTIVE_ID_BASE)
    _REPLICATED = _replicated_spec()
    inner_name = f"vllm_torchtpu::llar_two_shot_{width}"
    inner = sharded_jax_op(
        inner_name,
        fn,
        mesh=mesh,
        input_partition_specs=(_REPLICATED, ) * 4,
        output_partition_specs=(_REPLICATED, ) * 4,
    )

    device = torch.device("tpu")
    rs = torch.zeros((2, num_devices - 1, 1, cols_max),
                     dtype=dtype,
                     device=device)
    ag = torch.zeros((2, num_devices, 1, cols_max), dtype=dtype, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)

    def xla_fallback(x: torch.Tensor) -> torch.Tensor:
        # Above the threshold this is byte-for-byte what RowParallelLinear
        # would have done: the same XLA ring reduction over the same group.
        # The clone is forced by the custom-op contract -- the op declares no
        # mutated arguments, so it cannot reduce x in place.
        out = x.clone()
        dist.all_reduce(out, group=communicator.device_group)
        return out

    op = _build_outer_op(f"vllm_torchtpu::llar_out_proj_reduce_{width}",
                         inner,
                         max_numel=max_numel,
                         num_devices=num_devices,
                         xla_fallback=xla_fallback)

    logger.info(
        "Low-latency all-reduce: out-proj reduces at width %d take the "
        "two-shot kernel at or below %d elements (%.1f KiB at %d devices, "
        "batch limit %d tokens); larger payloads keep the XLA ring, inside "
        "the same op.", width, max_numel,
        max_numel * torch.empty((), dtype=dtype).element_size() / 1024,
        num_devices, max_batch_size())
    return op, rs, ag, step


def _takes_kernel(numel: int, max_numel: int, num_devices: int) -> bool:
    """The switch predicate. Torch and test code share exactly this."""
    return numel <= max_numel and numel % num_devices == 0


def _build_outer_op(name: str, inner, *, max_numel: int, num_devices: int,
                    xla_fallback):
    """The single opaque custom op a layer calls on every bucket.

    Dynamo must not see the switch. Selecting a kernel in traced Python means
    reading dims, and reading vLLM's dynamic token dimension pins it and fails
    the whole compilation ("Constraints violated (L['input_ids'].size()[0])").
    ``torch.library.custom_op`` is not traced into, so the graph gets one call
    node and the dims are read later, at execution, when they are ordinary
    ints -- which is what lets the branch below exist at all.

    The workspaces are *arguments*, sized once for the largest wired payload,
    so every token bucket presents the same graph interface. They cannot be
    captured inside the implementation: a tensor the compiled program
    references but was never handed aborts it with "identified an argument
    that was not provided: DeviceBufferRef".
    """

    def _impl(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
              step: torch.Tensor) -> torch.Tensor:
        # The token-count switch. x is a concrete tensor here -- the custom op
        # boundary is exactly what keeps this read out of the traced graph.
        if _takes_kernel(x.numel(), max_numel, num_devices):
            out, new_rs, new_ag, new_step = inner(x, rs, ag, step)
            # The workspace has to be written back explicitly: the op returns
            # fresh buffers, and the kernel needs the *same* buffer next call
            # for its slot parity to mean anything. Only the kernel branch
            # advances the workspace, so only it needs the writeback.
            rs.copy_(new_rs)
            ag.copy_(new_ag)
            step.copy_(new_step)
            return out
        return xla_fallback(x)

    # mutates_args stays empty even though the implementation writes the
    # workspace in place. Declaring the mutation makes AOTAutograd add the
    # buffers to the graph's outputs, and the compiled artifact then returns
    # one where the runtime wrapper expects several:
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
    op.register_fake(lambda x, rs, ag, step: torch.empty_like(x))
    return op
