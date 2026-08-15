# SPDX-License-Identifier: Apache-2.0
"""Route small TP all-reduces to the low-latency Pallas kernels.

XLA lowers an all-reduce to a *ring* reduce-scatter + all-gather: bandwidth
optimal, but ``2(N-1)`` sequential hops.  At decode payloads the hop count, not
the bytes, is the cost.  ``kernels/collectives/low_latency_all_reduce.py``
trades egress volume for hops -- one-shot (1 hop) below the crossover, two-shot
(2 hops) above it -- and measures 1.1-2.5x faster than XLA in that range.  See
``docs/low_latency_allreduce.md`` for the sweep this module's dispatch table is
read off.

Three things make this more than a function swap:

**The kernels need a mesh, and vLLM has none.**  Each vLLM worker drives one
core and reduces through ``torch.distributed``; the runner's JAX mesh is a
single device (``_create_mesh_for_parallelism``).  A Pallas collective instead
needs a ``shard_map`` over every peer.  That turns out to be constructible:
inside a worker, ``jax.devices()`` exposes the whole slice even though
``jax.local_devices()`` is one core, so a mesh can be built over the group's
TPU *global device ids* -- the same trick
``vllm_torchtpu/distributed/pcp.py`` uses for context-parallel attention.
Rank order is **not** device-id order (rank 2 sits on device 4 on this host),
so the mapping is gathered rather than assumed.

**The ``_ws`` kernels are stateful.**  They keep a persistent HBM workspace and
a round counter so the barrier can move after the sends and overlap them.  Both
must be threaded call to call, which here means a per-shape buffer donated into
the op and written back with ``copy_`` -- the same idiom the RPA kernel uses for
the KV cache, where XLA aliases the donation and the copy compiles away.

**The gate is mandatory.**  Off-crossover the one-shot family is
catastrophically worse (857 us at 1 MiB / 32 devices against XLA's 38).  The
decision is made from the *static* shape at trace time, so every rank builds an
identical graph -- a requirement here, not a nicety: mismatched per-rank graphs
give the collectives mismatched XLA channel ids (see the RowParallelLinear bias
patch in ``vllm_torchtpu/__init__.py``).
"""

# NOTE: deliberately no `from __future__ import annotations`. The kernel
# wrappers below are handed to jax_op, which validates their signature by
# inspecting real annotation *objects*; under PEP 563 they arrive as strings
# that it cannot resolve (`jax` is imported inside the enclosing function, so
# eval_str has no module global to find), and every shape silently falls back
# to XLA.
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Barrier-semaphore ids must not collide with another kernel that can be in
# flight at the same time. The fused-MoE kernel uses 0, so start clear of it.
_COLLECTIVE_ID_BASE = 8


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


# Dispatch thresholds in bytes, read off the measured sweep in
# docs/low_latency_allreduce.md (bfloat16, cols=1024). Both kernels lose to
# XLA's ring once the payload is large enough for bandwidth to dominate hop
# count, and one-shot's egress is M*(N-1), so its useful range shrinks as the
# device count grows.
#
#   (one_shot_max, two_shot_min, two_shot_max)
#
# The 32-device gap at 32 KiB is real and deliberate: one-shot is already
# bandwidth-bound at 31 peers (0.70x) and two-shot's measured range starts at
# 64 KiB, so that cell stays on XLA.
_GATE: dict[int, tuple[int, int, int]] = {
    2: (256 * 1024, 0, 0),
    4: (128 * 1024, 0, 0),
    8: (32 * 1024, 64 * 1024, 512 * 1024),
    16: (16 * 1024, 64 * 1024, 256 * 1024),
    32: (16 * 1024, 64 * 1024, 256 * 1024),
}


class _Plan:
    """What to do with one (shape, dtype) of all-reduce on one group."""

    __slots__ = ("kernel", "op", "buffers")

    def __init__(self, kernel: str, op: Any, buffers: tuple):
        self.kernel = kernel  # "one_shot" | "two_shot" | "xla"
        self.op = op
        self.buffers = buffers


def _two_shot_enabled() -> bool:
    """Whether two-shot may be used. Off unless explicitly asked for.

    two-shot is correct -- it matches ``funcol`` exactly in float32 across
    every shape and a 6-round chain -- but it is not yet *reliable*: a 256 KiB
    reduction that had passed twice took the core down on the third run with
    ``RuntimeUnexpectedCoreHalt``. That nondeterminism is the same shape as the
    semaphore-overlap fragility documented in docs/low_latency_allreduce.md
    section 3a, and it has not been root-caused for this call path.

    Leaving it off costs little at the sizes that matter here: K3 decode at
    TP=32 reduces 7-28 KiB, which is one-shot territory. Set
    ``TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT=1`` to include it.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "0") == "1"


def _choose_kernel(num_devices: int, payload_bytes: int, numel: int) -> str:
    """Pick a kernel from the measured table, or ``"xla"`` to leave it alone."""
    gate = _GATE.get(num_devices)
    if gate is None:
        return "xla"
    one_shot_max, two_shot_min, two_shot_max = gate
    if payload_bytes <= one_shot_max:
        return "one_shot"
    if (two_shot_max and _two_shot_enabled()
            and two_shot_min <= payload_bytes <= two_shot_max
            and numel % num_devices == 0):
        # two-shot scatters along rows. Flattening the payload and viewing it
        # as (num_devices, numel/num_devices) satisfies its divisibility
        # constraint for any token count, which the natural [tokens, hidden]
        # view would not (decode has far fewer rows than devices).
        return "two_shot"
    return "xla"


class LowLatencyAllReduce:
    """Per-process-group dispatcher, built lazily on first use."""

    def __init__(self, communicator):
        self._comm = communicator
        self._world_size = int(communicator.world_size)
        self._mesh = None
        self._mesh_failed = False
        self._plans: dict[tuple, _Plan] = {}
        self._next_collective_id = _COLLECTIVE_ID_BASE
        self._axis = "tp"

    # -- mesh ---------------------------------------------------------------

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
                "Incomplete rank->device map %s; low-latency "
                "all-reduce disabled.", by_rank)
            return None
        return [by_rank[r] for r in range(self._world_size)]

    def _get_mesh(self):
        if self._mesh is not None or self._mesh_failed:
            return self._mesh
        try:
            import jax
            import numpy as np
            from jax.sharding import Mesh

            order = self._device_id_order()
            if order is None:
                self._mesh_failed = True
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
                self._mesh_failed = True
                return None
            self._mesh = Mesh(np.asarray([by_id[d] for d in order]),
                              axis_names=(self._axis, ))
            logger.info(
                "Low-latency all-reduce: mesh over %d devices "
                "(rank-in-group order -> device ids %s).", self._world_size,
                order)
            return self._mesh
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not build the low-latency all-reduce mesh "
                "(%s); falling back to XLA.", e)
            self._mesh_failed = True
            return None

    # -- op construction ----------------------------------------------------

    def _build_one_shot(self, shape: tuple[int, ...], dtype: torch.dtype):
        import jax

        from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
            one_shot_all_reduce_ws
        from vllm_torchtpu.kernels.collectives.sharded_jax_op import \
            sharded_jax_op

        mesh = self._mesh
        axis = self._axis
        n = self._world_size
        _REPLICATED = _replicated_spec()
        collective_id = self._next_collective_id
        self._next_collective_id += 1

        def _fn(x: jax.Array, ws: jax.Array,
                step: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:

            def body(x, ws, step):
                # has_side_effects=False is required to compile through
                # torch_tpu, and is safe here; see the kernel's docstring.
                return one_shot_all_reduce_ws(x,
                                              ws,
                                              step,
                                              axis_name=axis,
                                              num_devices=n,
                                              collective_id=collective_id,
                                              has_side_effects=False)

            return jax.shard_map(body,
                                 mesh=mesh,
                                 in_specs=(_REPLICATED, ) * 3,
                                 out_specs=(_REPLICATED, ) * 3,
                                 check_vma=False)(x, ws, step)

        name = (f"vllm_torchtpu::ll_all_reduce_one_shot_{collective_id}")
        op = sharded_jax_op(
            name,
            _fn,
            mesh=mesh,
            input_partition_specs=(_REPLICATED, ) * 3,
            output_partition_specs=(_REPLICATED, ) * 3,
        )
        device = self._comm.device
        ws = torch.zeros((2, n - 1, *shape), dtype=dtype, device=device)
        step = torch.zeros(1, dtype=torch.int32, device=device)
        return op, (ws, step)

    def _build_two_shot(self, shape: tuple[int, ...], dtype: torch.dtype):
        import jax

        from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
            two_shot_all_reduce_ws
        from vllm_torchtpu.kernels.collectives.sharded_jax_op import \
            sharded_jax_op

        mesh = self._mesh
        axis = self._axis
        n = self._world_size
        _REPLICATED = _replicated_spec()
        collective_id = self._next_collective_id
        self._next_collective_id += 1

        numel = 1
        for d in shape:
            numel *= d
        cols = numel // n
        chunk = 1

        def _fn(
            x: jax.Array, rs: jax.Array, ag: jax.Array, step: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

            def body(x, rs, ag, step):
                flat = x.reshape(n, cols)
                out, rs, ag, step = two_shot_all_reduce_ws(
                    flat,
                    rs,
                    ag,
                    step,
                    axis_name=axis,
                    num_devices=n,
                    collective_id=collective_id,
                    has_side_effects=False)
                return out.reshape(x.shape), rs, ag, step

            return jax.shard_map(body,
                                 mesh=mesh,
                                 in_specs=(_REPLICATED, ) * 4,
                                 out_specs=(_REPLICATED, ) * 4,
                                 check_vma=False)(x, rs, ag, step)

        name = (f"vllm_torchtpu::ll_all_reduce_two_shot_{collective_id}")
        op = sharded_jax_op(
            name,
            _fn,
            mesh=mesh,
            input_partition_specs=(_REPLICATED, ) * 4,
            output_partition_specs=(_REPLICATED, ) * 4,
        )
        device = self._comm.device
        rs = torch.zeros((2, n - 1, chunk, cols), dtype=dtype, device=device)
        ag = torch.zeros((2, n, chunk, cols), dtype=dtype, device=device)
        step = torch.zeros(1, dtype=torch.int32, device=device)
        return op, (rs, ag, step)

    # Everything this builds -- a process group gather, a torch custom-op
    # registration, persistent workspace buffers, a log line -- is a side
    # effect, and it runs the first time a shape is seen, which is *during*
    # torch.compile tracing. Dynamo cannot trace any of it; on K3 it died on
    # the log call alone:
    #
    #   Unresolved GetAttrVariable comparison ... LoggingLoggerVariable
    #
    # and the workspace allocation would have been worse than an error --
    # tensors created inside the traced region are graph temporaries, not the
    # persistent state these kernels require. Disabling dynamo here runs the
    # builder eagerly and hands the traced region ready-made buffers to close
    # over, which is exactly how the RPA kernel's KV cache reaches the graph.
    @torch._dynamo.disable(recursive=True)
    def _build_plan(self, x: torch.Tensor, key: tuple) -> _Plan:
        numel = x.numel()
        payload = numel * x.element_size()
        kernel = _choose_kernel(self._world_size, payload, numel)

        if kernel != "xla" and self._get_mesh() is None:
            kernel = "xla"

        if kernel == "xla":
            plan = _Plan("xla", None, ())
        else:
            try:
                if kernel == "one_shot":
                    op, buffers = self._build_one_shot(tuple(x.shape), x.dtype)
                else:
                    op, buffers = self._build_two_shot(tuple(x.shape), x.dtype)
                plan = _Plan(kernel, op, buffers)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Could not build the %s all-reduce for %s %s (%s); "
                    "this shape stays on XLA.", kernel, tuple(x.shape),
                    x.dtype, e)
                plan = _Plan("xla", None, ())

        logger.info(
            "Low-latency all-reduce [%s]: shape=%s dtype=%s payload=%.1f KiB "
            "devices=%d -> %s", self._comm.unique_name, tuple(x.shape),
            x.dtype, payload / 1024, self._world_size, plan.kernel)
        self._plans[key] = plan
        return plan

    # -- entry point --------------------------------------------------------

    def maybe_all_reduce(self, x: torch.Tensor) -> torch.Tensor | None:
        """Reduced tensor, or ``None`` to let the caller use XLA's all-reduce.

        Kept deliberately thin: past the cache lookup this is only the custom
        op and its writeback, so what dynamo traces is the collective and
        nothing else. All construction happens in ``_build_plan``, which dynamo
        does not enter.
        """
        if self._world_size < 2:
            return None
        if not x.is_contiguous():
            return None
        key = (tuple(x.shape), x.dtype)
        plan = self._plans.get(key)
        if plan is None:
            if torch.compiler.is_compiling():
                # Never build inside the traced region -- see _build_plan. A
                # shape that reaches here unbuilt is one `prebuild` did not
                # anticipate; XLA's all-reduce is always correct, so it takes
                # it. `prebuild` logs what it covered, which is where to look
                # if the kernels appear to be doing nothing.
                return None
            plan = self._build_plan(x, key)
        if plan.kernel == "xla":
            return None

        if plan.kernel == "one_shot":
            ws, step = plan.buffers
            out, new_ws, new_step = plan.op(x, ws, step)
            # Donation + copy_ writeback, as the RPA kernel does for the KV
            # cache: XLA aliases the donated buffer onto the output and the
            # copy compiles away, leaving the workspace updated in place and
            # its identity stable for the next call.
            ws.copy_(new_ws)
            step.copy_(new_step)
            return out

        rs, ag, step = plan.buffers
        out, new_rs, new_ag, new_step = plan.op(x, rs, ag, step)
        rs.copy_(new_rs)
        ag.copy_(new_ag)
        step.copy_(new_step)
        return out


def _candidate_shapes(vllm_config) -> tuple[list[tuple[int, int]], Any]:
    """(rows, cols) an all-reduce might carry, derived from the config.

    Rows are the static compile buckets, since a decode step is padded up to
    one of them. Columns are the widths a TP reduction can have: the hidden
    size, plus the routed-expert latent width when the model has one -- K3
    reduces in both, 7168 after attention and 3584 around the latent MoE.

    Over-generating is free: a shape outside the crossover gets an "xla" plan,
    which allocates nothing.
    """
    model_config = vllm_config.model_config
    hf = getattr(model_config, "hf_text_config",
                 None) or model_config.hf_config

    widths = []
    for attr in ("hidden_size", "routed_expert_hidden_size"):
        value = getattr(hf, attr, None)
        if isinstance(value, int) and value > 0 and value not in widths:
            widths.append(value)

    rows = []
    compile_sizes = getattr(vllm_config.compilation_config, "compile_sizes",
                            None) or []
    for size in compile_sizes:
        if isinstance(size, int) and size > 0 and size not in rows:
            rows.append(size)
    max_batched = getattr(vllm_config.scheduler_config,
                          "max_num_batched_tokens", None)
    if isinstance(max_batched, int) and max_batched > 0 \
            and max_batched not in rows:
        rows.append(max_batched)

    return [(r, c) for r in sorted(rows) for c in widths], model_config.dtype


def warmup(vllm_config) -> None:
    """Build every plan the model is expected to need, before compile traces.

    Must run after the weights are loaded (the process groups exist) and
    before torch.compile sees a forward pass. ``_build_plan`` registers a torch
    custom op, gathers a device map and allocates persistent workspaces, none
    of which dynamo can trace; doing it here is what keeps the traced region
    down to the collective itself.
    """
    from vllm_torchtpu import envs

    if not envs.TPU_LOW_LATENCY_ALLREDUCE:
        return
    try:
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
    except Exception as e:  # noqa: BLE001
        logger.warning("Low-latency all-reduce warmup skipped (%s).", e)
        return
    if communicator is None or int(getattr(communicator, "world_size", 1)) < 2:
        return

    dispatcher = get_dispatcher(communicator)
    shapes, dtype = _candidate_shapes(vllm_config)
    if not shapes:
        logger.warning("Low-latency all-reduce warmup found no candidate "
                       "shapes; every all-reduce will stay on XLA.")
        return

    built = 0
    for rows, cols in shapes:
        probe = torch.empty((rows, cols),
                            dtype=dtype,
                            device=communicator.device)
        plan = dispatcher._build_plan(probe, ((rows, cols), dtype))
        del probe
        if plan.kernel != "xla":
            built += 1
    logger.info(
        "Low-latency all-reduce warmup: %d of %d candidate shapes take a "
        "kernel.", built, len(shapes))


def get_dispatcher(communicator) -> LowLatencyAllReduce:
    """One dispatcher per communicator instance."""
    existing = getattr(communicator, "_ll_all_reduce", None)
    if existing is None:
        existing = LowLatencyAllReduce(communicator)
        communicator._ll_all_reduce = existing
    return existing
