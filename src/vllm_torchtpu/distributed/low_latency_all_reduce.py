# SPDX-License-Identifier: Apache-2.0
"""Route small TP all-reduces to the low-latency Pallas kernels.

XLA lowers an all-reduce to a *ring* reduce-scatter + all-gather: bandwidth
optimal, but ``2(N-1)`` sequential hops.  At decode payloads the hop count, not
the bytes, is the cost.  ``kernels/collectives/low_latency_all_reduce.py``
``two_shot_all_reduce`` trades egress volume for hops -- 2 instead of
``2(N-1)`` -- and measures 1.3-1.4x faster than XLA over the decode band.  See
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

**The kernel is stateful.**  It keeps a persistent HBM workspace and a round
counter so the barrier can move after the sends and overlap them.  Both must be
threaded call to call, which here means a per-shape buffer passed into the op
and written back with ``copy_`` -- the same idiom the RPA kernel uses for the KV
cache.

**The gate is mandatory.**  Past the crossover the kernel is 0.61x XLA at
448 KiB, and it only gets worse from there -- egress is flat in peers but
linear in bytes, so bandwidth eventually dominates the hops it saved.  The
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
# docs/low_latency_allreduce.md (bfloat16). two_shot loses to XLA's ring once
# the payload is large enough for bandwidth to dominate hop count, so each
# entry is a window rather than a ceiling.
#
#   (two_shot_min, two_shot_max)
#
# 2 and 4 devices are absent deliberately: the kernel's advantage comes from
# collapsing 2(N-1) hops to 2, which is nothing at N=2 and one hop at N=4, so
# neither was ever measured to win and both fall through to XLA.
#
# The floor is 0 rather than a measured smallest payload. That direction is the
# safe one: two_shot is nearly flat in payload across this range and XLA is flat
# but higher -- at 32 devices two_shot reads 12.36 / 12.67 / 13.20 us at 14 / 28
# / 42 KiB against XLA's 17.31 / 17.97 / 18.93 -- so shrinking the payload makes
# two_shot's job easier while XLA stays put. The gap only closes going the other
# way, which is what the ceiling is for.
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
# The 32- and 16-device ceiling is 128 KiB because two_shot is only *locally*
# flat: past ~112 KiB it goes linear at +8.5 us per 112 KiB while XLA's ring
# amortises, and they cross near 150 KiB. 128 KiB sits inside the measured
# winning region rather than on the crossover.
#
# The 8-device ceiling is 64 KiB, not the 512 KiB it used to be. That 512 KiB
# was never a measurement -- it was the top of the original sweep's range, the
# same way its 64 KiB floor was the bottom -- and 8 devices is only measured to
# 42 KiB. 64 KiB is one modest step past the last measured point instead of an
# order of magnitude past it. K3's pruned 8-device proxy tops out at the 42 KiB
# pair, so nothing in use crosses it.
_GATE: dict[int, tuple[int, int]] = {
    8: (0, 64 * 1024),
    16: (0, 128 * 1024),
    32: (0, 128 * 1024),
}

# Default ceiling for TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE. Deliberately
# above where _GATE bites at every device count in the table, so the default
# behaviour is "the measured payload gate decides" rather than a second,
# unmeasured bound layered on top of it. See max_batch_size().
_DEFAULT_MAX_BATCH_SIZE = 16


class _Plan:
    """What to do with one (shape, dtype) of all-reduce on one group."""

    __slots__ = ("kernel", "op", "buffers", "shape", "dtype")

    def __init__(self,
                 kernel: str,
                 op: Any,
                 buffers: tuple,
                 shape: tuple = (),
                 dtype: Any = None):
        self.kernel = kernel  # "two_shot" | "xla"
        self.op = op
        self.buffers = buffers
        # The kernel is compiled for exactly this shape; the entry op checks
        # before using it, because a mismatch would re-lower at runtime.
        self.shape = shape
        self.dtype = dtype


def _two_shot_enabled() -> bool:
    """Kill switch for the kernel, on by default.

    On rather than off because it is the only kernel here: defaulting it off
    would send every payload to XLA and make the whole dispatcher a no-op, which
    is what ``TPU_LOW_LATENCY_ALLREDUCE`` itself is for.

    One reliability caveat is worth carrying: a 256 KiB reduction that had passed twice once
    took the core down on a third run with ``RuntimeUnexpectedCoreHalt``, the
    same shape as the semaphore-overlap fragility in section 3a of
    docs/low_latency_allreduce.md, and it was never root-caused. Two things
    bound it. It was seen at 256 KiB, and the gate's ceiling is now 128 KiB at
    16/32 devices and 64 KiB at 8, so that payload can no longer reach the
    kernel through this path. And the kernel has since run three full K3
    end-to-end jobs at TP=32 (286, 287 and the barrier-free A/B), each with a
    coherent canary and 3x1024-token sweeps, without a halt.

    Set ``TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT=0`` to force everything back to
    XLA, which is also the fastest way to tell whether a suspected regression
    is this kernel's fault.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1") != "0"


def max_batch_size() -> int:
    """Largest padded token count that may take the kernel.

    This is the knob that decides *which batches* go to the kernel; ``_GATE``
    decides which *payloads* may. Both have to pass, and in practice the gate is
    the binding one at the widths real models reduce -- at 32 devices its
    128 KiB ceiling is already reached by 9 tokens of a 7168-wide bf16 payload.
    So the default is permissive and the gate governs; lower this to be more
    conservative than the measurement, or to bound how many workspaces warmup
    allocates.

    Expressed in padded tokens because that is what the forward context reports
    and what the kernel is compiled for. For decode, where this kernel is
    aimed, padded tokens is the batch size.

    A batch threshold rather than an explicit shape. An earlier revision took
    ``TPU_LOW_LATENCY_ALLREDUCE_SHAPE=ROWSxCOLS`` and wired exactly that one
    shape, which was a trap twice over: the value is ``1x7168``, not ``7168``,
    and a bare width parses to a 1-tuple, matches nothing, and falls back with
    a warning in the server log only; and forgetting it entirely left the
    dispatcher picking the *smallest* payload that took a kernel -- the 7 KiB
    latent reduce -- while the 14 KiB post-attention one it was aimed at stayed
    on XLA. Both cost a slice. Nothing here now names a shape.
    """
    import os

    raw = os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "")
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
            "would disable the kernel entirely; unset "
            "TPU_LOW_LATENCY_ALLREDUCE instead. Using %d.", value,
            _DEFAULT_MAX_BATCH_SIZE)
        return _DEFAULT_MAX_BATCH_SIZE
    return value


def _choose_kernel(num_devices: int, payload_bytes: int, numel: int) -> str:
    """Pick a kernel from the measured table, or ``"xla"`` to leave it alone."""
    gate = _GATE.get(num_devices)
    if gate is None:
        return "xla"
    two_shot_min, two_shot_max = gate
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
        # Torch op names must be unique and stable; the group name is both.
        self.op_suffix = str(getattr(communicator, "unique_name",
                                     "tp")).replace(":", "_")

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

    def _build_two_shot(self, shape: tuple[int, ...], dtype: torch.dtype):
        """The reduce op for this group: kernel where it wins, psum elsewhere.

        One op, emitted at every call site on every bucket, choosing inside.
        That uniformity is the point, not an accident of convenience. The
        alternative -- emit the op only where the kernel applies -- makes the
        workspaces graph inputs on some buckets and not others, and vLLM
        compiles every bucket from *one* Dynamo trace: ``compile_all_ranges``
        feeds one FX graph concrete inputs per size, so structure is fixed once
        and shapes vary. ``compilation/shape_variants.py`` can re-trace per
        group, but only from ShapeEnv guards, and the token count cannot leave
        one here -- reading that dim pins vLLM's marked-dynamic input_ids
        dimension and fails with ConstraintViolationError. The trace that stays
        live then has to serve every bucket, and ``BucketExecutables.get``
        checks each handed-over executable against ``_signature(graph)``, the
        subgraph's placeholder list. Two placeholder lists cannot belong to one
        live graph, which is the ShapeSpecializationError this design avoids by
        never producing a second list.

        The choice itself is free. ``sharded_jax_op`` lowers per input shape, so
        inside the traced JAX function ``x.size`` is a concrete int and the
        branch below is ordinary Python at trace time -- the same arrangement
        ``layers/common/fused_moe_gmm.py`` uses for
        ``onehot_moe_permute_threshold``, which is a ``jax.jit`` static argname
        branching on ``token_indices_sorted.shape[0]`` inside the pallas op.
        Nothing about it is visible to dynamo, so nothing about it can change
        the torch graph.

        The fallback is ``jax.lax.psum`` rather than the torch-level
        ``xla_all_reduce``: it is the same ring reduction XLA would have
        emitted, and it avoids the out-of-place clone the custom-op contract
        forces on the torch path -- 7 MiB per reduction at a 512-token bucket.
        What it does not avoid is the custom-call boundary: XLA's collective
        combiner cannot merge a psum inside an opaque call with the collectives
        around it. That is the cost of uniformity and it is unmeasured.
        """
        import jax

        from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
            two_shot_all_reduce
        from vllm_torchtpu.kernels.collectives.sharded_jax_op import \
            sharded_jax_op

        mesh = self._mesh
        axis = self._axis
        n = self._world_size
        _REPLICATED = _replicated_spec()
        collective_id = self._next_collective_id
        self._next_collective_id += 1
        steady = steady_barrier_enabled()

        numel = 1
        for d in shape:
            numel *= d
        cols = numel // n
        chunk = 1
        # The workspaces are sized for this payload, so the kernel branch can
        # only fire for it. Any other shape reaching this op takes the psum
        # branch, which reads neither workspace.
        # The kernel branch fires for exactly the payload the workspaces were
        # sized for, tested by element count. build_reduce_op derives the same
        # number from the same plan shape and gates its writeback on it; the two
        # must agree, and this comment is the only thing holding them together.
        # dtype is not part of the test because it cannot vary -- warmup takes
        # it from model_config.dtype and builds one op per width -- and a
        # mismatch would fail loudly at lowering rather than silently, since the
        # workspaces would not type-check against the payload.
        wired_numel = numel

        def _fn(
            x: jax.Array, rs: jax.Array, ag: jax.Array, step: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

            def body(x, rs, ag, step):
                # Concrete ints at JAX trace time; this is not a lax branch and
                # emits no select. See the docstring.
                if x.size != wired_numel:
                    return jax.lax.psum(x, axis_name=axis), rs, ag, step
                flat = x.reshape(n, cols)
                out, rs, ag, step = two_shot_all_reduce(
                    flat,
                    rs,
                    ag,
                    step,
                    axis_name=axis,
                    num_devices=n,
                    collective_id=collective_id,
                    steady_barrier=steady,
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
                op, buffers = self._build_two_shot(tuple(x.shape), x.dtype)
                plan = _Plan(kernel, op, buffers, tuple(x.shape), x.dtype)
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

    def xla_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """What the communicator would have done, out of place.

        The clone is forced by the custom-op contract: the entry op declares
        no mutated arguments, so it cannot reduce ``x`` in place the way
        ``DeviceCommunicatorBase.all_reduce`` does. It is a real difference
        between the arms on shapes that stay on XLA, small (a 28 KiB copy) but
        worth knowing about when reading a result.
        """
        import torch.distributed as dist

        out = x.clone()
        dist.all_reduce(out, group=self._comm.device_group)
        return out


_ENTRY_OPS: dict[str, Any] = {}


def select_entry(entries: dict) -> tuple | None:
    """The wired entry for this step's padded token count, or None.

    ``entries`` maps a padded token count to ``(op, buffers)``. Warmup fills it
    with every compile bucket at or below ``max_batch_size()`` whose payload
    also clears ``_GATE``; anything else is absent and reduces through XLA.

    Why a lookup keyed on the token count rather than a check inside the op.
    The op *does* check its own shape and fall back to ``xla_all_reduce``, which
    is correct but not free: the fallback happens inside an opaque custom call,
    so XLA sees a collective it cannot fuse, combine, or reorder, and the caller
    has already done whatever work it does to prepare the payload. On K3's
    latent-MoE tail that preparation is a ``torch.cat`` plus two slices per
    layer, pure loss when the kernel does not run -- the c4/c8 arms of job 139
    regressed 5.2% and 4.0% on TPOT for exactly this reason, while c1, a bucket
    the kernel was wired for, improved 3.3%. So the caller asks first.

    The token count is the one thing it cannot ask a tensor for: reading a dim
    of a traced tensor pins vLLM's dynamic token dimension and fails compilation
    outright (see ``build_reduce_op``). The forward context is the way around
    it. The TPU runner passes ``num_tokens=num_tokens_padded`` into
    ``set_forward_context`` -- already rounded up to a compile bucket, a plain
    Python int, set *outside* anything traced -- so this indexes a dict with an
    int and dynamo folds the whole thing to a constant. It never touches a
    tensor.

    Reading the forward context from inside a traced forward is upstream's own
    pattern on this exact path: ``MoERunner._encode_layer_name`` branches on
    ``get_forward_context().all_moe_layers is not None``.

    Missing context or a missing descriptor means "cannot tell", which resolves
    to None -- the caller keeps the path it would have had without any of this.
    """
    from vllm.forward_context import (get_forward_context,
                                      is_forward_context_available)

    if not is_forward_context_available():
        return None
    descriptor = get_forward_context().batch_descriptor
    if descriptor is None:
        return None
    return entries.get(descriptor.num_tokens)


# Set by warmup() when TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT=1, and read by
# models/vllm/kimi_k3/latent_moe_runner.KimiK3LatentMoERunner. Maps a padded
# token count to (op, buffers), same shape as the communicator's own dict.
#
# K3's MoE block reduces twice: the routed experts' output at the latent width
# and the shared experts' at the hidden width. They are independent, so XLA's
# combiner merges them into one 42 KiB collective costing 33.97 us -- and that
# amortisation is why every attempt to put a kernel on one half lost. Doing the
# same job with one kernel over the concatenated pair is the only arrangement
# that keeps the amortisation and gets the faster collective, measured in
# scripts/debug/fusion_probe.py at 15.07 us against the combiner's 21.14, or
# -10.5 us/layer of collective time reproduced across three runs.
#
# Delivered through its own global rather than the communicator, because the
# call site is explicit and so nothing has to infer which reduction this is.
#
# Note what that buys, and why the two halves of this feature ship together.
# The communicator routes by *shape*, and K3's post-attention reduction and the
# MoE block's shared-expert one are both the hidden width -- indistinguishable
# there. Routing that shape while leaving the pair alone splits what the
# combiner was fusing, and the lost fusion costs more than the faster kernel
# gains: +5.5% device time, measured. Handling the pair here takes it out of
# the communicator's view entirely, so the shape route becomes unambiguous and
# the post-attention reduction can take the kernel on its own.
_MOE_CONCAT_ENTRIES: dict[int, tuple] = {}


def moe_concat_enabled() -> bool:
    """Whether to reduce K3's latent-MoE pair as one concatenated collective.

    Off by default: it replaces two upstream reductions with one, which is a
    real change to vLLM's latent-MoE tail, and it only pays off where the
    combiner was already fusing the pair.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT", "0") == "1"


def steady_barrier_enabled() -> bool:
    """Whether the kernel keeps its steady-state barrier.

    True by default. Setting TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER=1
    removes the barrier from every round except round 0, leaving the collective
    with no synchronization primitive in its steady state -- the recv wait
    already proves every peer finished the round that last read this slot. Worth
    ~1.5 us per collective (see the kernel docstring for the argument and
    docs/low_latency_allreduce.md section 5 for the measurements).

    Off by default because the failure mode is a silent stale read and soundness
    requires every rank to call the collective the same number of times in the
    same order. Speculative decode, early exit, a _dummy_run firing on a subset
    of ranks, or a non-unanimous bucket switch all break that.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER",
                          "0") != "1"


def build_reduce_op(dispatcher: LowLatencyAllReduce, plan: _Plan,
                    name: str) -> Any:
    """An opaque torch op that reduces one wired shape with the kernel.

    Two constraints meet here and between them leave only this shape.

    Dynamo must not see the shape. Selecting a kernel in traced Python means
    reading dims, and reading vLLM's dynamic token dimension pins it and fails
    the whole compilation ("Constraints violated (L['input_ids'].size()[0])").
    No guard avoids it: dynamo presents a SymInt as an int, so even
    ``isinstance(dim, int)`` is True. ``torch.library.custom_op`` is not traced
    into, so the graph gets one call node and the dims are read later, at
    execution, when they are ordinary ints -- which is what lets the check
    below exist at all.

    The workspaces must be *arguments*. They cannot be captured inside the
    implementation: a tensor the compiled program references but was never
    handed aborts it with "identified an argument that was not provided:
    DeviceBufferRef". A custom op's schema is fixed at registration, so the
    arity is fixed too, and one op can therefore serve exactly one shape. That
    is why warmup builds an op *per* wired bucket and hands the caller a dict to
    index rather than one op that dispatches internally.

    Every bucket goes through this op; what varies is what the op does, chosen
    inside the traced JAX body from a concrete shape. See ``_build_two_shot``
    for why the uniformity is load-bearing rather than incidental.
    """
    existing = _ENTRY_OPS.get(name)
    if existing is not None:
        return existing

    plan_numel = 1
    for dim in plan.shape:
        plan_numel *= dim

    def _impl(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
              step: torch.Tensor) -> torch.Tensor:
        out, new_rs, new_ag, new_step = plan.op(x, rs, ag, step)
        # Only the kernel branch advances the workspace, so only it needs the
        # writeback -- the op returns fresh buffers and the kernel needs the
        # *same* buffer next call for its slot parity to mean anything. The psum
        # branch returns them untouched, and copying them back would be pure
        # cost on every bucket the kernel does not serve.
        #
        # This predicate must stay identical to the one inside _build_two_shot's
        # traced body -- both are "element count equals the plan's". If torch
        # thought a call took the psum branch while JAX emitted the kernel, the
        # workspace would stop advancing and the slot parity would silently go
        # stale: wrong numbers, not an error.
        if x.numel() == plan_numel:
            rs.copy_(new_rs)
            ag.copy_(new_ag)
            step.copy_(new_step)
        return out

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
    _ENTRY_OPS[name] = op
    return op


def _moe_concat_width(vllm_config) -> int | None:
    """Width of the concatenated latent-MoE pair, or None if there is none.

    The pair's two reductions carry the hidden size and the routed-expert latent
    width, so reducing them together means one payload of their sum. Returns
    None unless the model has both, which is what makes it a latent MoE.

    The only place a model attribute other than the hidden size is read. It is
    here rather than in the K3 runner that consumes it because warmup has to
    allocate the workspace before anything is traced, and warmup is here.
    """
    model_config = vllm_config.model_config
    hf = getattr(model_config, "hf_text_config",
                 None) or model_config.hf_config
    hidden = getattr(hf, "hidden_size", None)
    latent = getattr(hf, "routed_expert_hidden_size", None)
    if not (isinstance(hidden, int) and hidden > 0):
        return None
    if not (isinstance(latent, int) and latent > 0):
        return None
    return hidden + latent


def _candidate_rows(vllm_config) -> list[int]:
    """Padded token counts worth building a plan for.

    The compile buckets, since a step is padded up to one of them, capped at
    ``max_batch_size()``. Uncompiled sizes are excluded on purpose: a plan for a
    token count the model never runs at would allocate a workspace nothing ever
    reads, and the entry could never be selected because the forward context
    only ever reports a bucket.
    """
    limit = max_batch_size()
    rows = set()
    compile_sizes = getattr(vllm_config.compilation_config, "compile_sizes",
                            None) or []
    for size in compile_sizes:
        if isinstance(size, int) and 0 < size <= limit:
            rows.add(size)
    max_batched = getattr(vllm_config.scheduler_config,
                          "max_num_batched_tokens", None)
    if isinstance(max_batched, int) and 0 < max_batched <= limit:
        rows.add(max_batched)
    return sorted(rows)


def _hidden_width(vllm_config) -> int | None:
    """The width a plain TP all-reduce carries.

    Every tensor-parallel reduction in a transformer -- attention's output
    projection, the MLP's down projection -- reduces at the hidden size, so this
    is the whole of the width derivation for the communicator's path. Anything
    narrower is a model-specific arrangement whose call site knows its own
    width and asks for it, which is what the latent-MoE pair does above.
    """
    model_config = vllm_config.model_config
    hf = getattr(model_config, "hf_text_config",
                 None) or model_config.hf_config
    width = getattr(hf, "hidden_size", None)
    return width if isinstance(width, int) and width > 0 else None


def warmup(vllm_config) -> None:
    """Build every plan the model is expected to need, before compile traces.

    Must run after the weights are loaded (the process groups exist) and before
    torch.compile sees a forward pass. ``_build_plan`` registers a torch custom
    op, gathers a device map and allocates persistent workspaces, none of which
    dynamo can trace; doing it here is what keeps the traced region down to the
    collective itself.

    Wires one entry per (bucket, width) that clears both ``max_batch_size()``
    and ``_GATE``. Each gets its own op and its own workspaces -- a custom op's
    schema is fixed at registration, so one op cannot serve two shapes -- and
    the callers index by token count. Nothing competes for a single slot, so
    there is no shape to name and no "pick one" heuristic to get wrong.
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

    # Cleared first so a second warmup (a re-created communicator, a test)
    # cannot leave stale entries pointing at workspaces the old group owned.
    global _MOE_CONCAT_ENTRIES
    _MOE_CONCAT_ENTRIES = {}
    communicator._ll_op = None

    rows = _candidate_rows(vllm_config)
    if not rows:
        logger.warning(
            "Low-latency all-reduce: no compile bucket is at or below "
            "TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE=%d, so every all-reduce "
            "stays on XLA.", max_batch_size())
        return

    dtype = vllm_config.model_config.dtype
    hidden = _hidden_width(vllm_config)
    concat = _moe_concat_width(vllm_config) if moe_concat_enabled() else None

    # (width, destination dict, op-name infix). The concatenated pair is wired
    # separately from the communicator's width and never competes with it: its
    # call site is explicit -- KimiK3LatentMoERunner -- so it gets its own ops
    # and its own workspaces.
    def _wire(width: int, infix: str):
        """One op for this width, bound to the smallest bucket that wins.

        Smallest rather than all of them: the op is emitted on *every* bucket,
        so it can carry only one workspace and therefore serve the kernel at
        only one shape. Widening that to several buckets means several workspace
        sets as arguments of one fixed-arity op -- mechanical, and deliberately
        not done here so this prototype changes one thing at a time.

        Smallest is the right one of the several: it is the lowest-concurrency
        decode step, where a latency-bound collective is worth the most.
        """
        for tokens in rows:
            probe = torch.empty((tokens, width),
                                dtype=dtype,
                                device=communicator.device)
            plan = dispatcher._build_plan(probe, ((tokens, width), dtype))
            del probe
            if plan.kernel == "xla":
                continue
            # Registered here, with everything else, because registration calls
            # infer_schema, which dynamo refuses to trace -- doing it on first
            # use would put it inside the traced forward and abort compilation.
            name = (f"vllm_torchtpu::ll_all_reduce_{infix}_"
                    f"{dispatcher.op_suffix}_{tokens}")
            return tokens, (build_reduce_op(dispatcher, plan,
                                            name), plan.buffers)
        return None, None

    if hidden is not None:
        wired_at, entry = _wire(hidden, "ts")
        communicator._ll_op = entry
        logger.info(
            "Low-latency all-reduce: the TP reduce op is emitted on every "
            "bucket and runs the kernel at %s tokens, jax.lax.psum elsewhere. "
            "Candidates %s, limit %d.", wired_at, rows, max_batch_size())

    # The concatenated latent-MoE pair keeps the per-bucket dict: its call site
    # prepares a payload (a cat and two slices) that is wasted when the kernel
    # does not run, so it has to decide *before* committing, which means
    # deciding in traced python. That leaves this path structurally
    # bucket-dependent and so still single-trace-group only. Giving it the same
    # treatment means always concatenating, whose cost is what the c4/c8
    # regression measured -- a separate change with a separate measurement.
    if concat is not None:
        for tokens in rows:
            probe = torch.empty((tokens, concat),
                                dtype=dtype,
                                device=communicator.device)
            plan = dispatcher._build_plan(probe, ((tokens, concat), dtype))
            del probe
            if plan.kernel == "xla":
                continue
            name = (f"vllm_torchtpu::ll_all_reduce_moe_concat_"
                    f"{dispatcher.op_suffix}_{tokens}")
            _MOE_CONCAT_ENTRIES[tokens] = (build_reduce_op(
                dispatcher, plan, name), plan.buffers)

    if concat is not None and not _MOE_CONCAT_ENTRIES:
        logger.warning(
            "TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT is set but no bucket takes a "
            "kernel at width %d, so the latent-MoE tail keeps its two separate "
            "reductions. Is the payload inside the two_shot band for %d "
            "devices?", concat, dispatcher._world_size)


def get_dispatcher(communicator) -> LowLatencyAllReduce:
    """One dispatcher per communicator instance."""
    existing = getattr(communicator, "_ll_all_reduce", None)
    if existing is None:
        existing = LowLatencyAllReduce(communicator)
        communicator._ll_all_reduce = existing
    return existing
