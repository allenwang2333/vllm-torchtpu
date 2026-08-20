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
    # two_shot's floor was 64 KiB only because the original sweep started
    # there, the same stale bound already corrected for 16/32 devices below.
    # Measured at 8 devices, device self time:
    #
    #   payload   XLA    one_shot   two_shot
    #    14 KiB  10.12       4.84       7.76
    #    42 KiB  10.61          -       8.21
    #
    # two_shot beats XLA at 42 KiB and one_shot still wins under 32 KiB, so
    # dropping the floor to one_shot_max closes the gap with no overlap. The
    # gap mattered: K3's concatenated latent-MoE pair is 42 KiB and fell
    # straight through it to XLA, which made the 8-device pruned-K3 proxy
    # unable to exercise the concat path at all.
    8: (32 * 1024, 32 * 1024, 512 * 1024),
    # two_shot's floor starts well below the 64 KiB the original sweep began
    # at. That sweep could not measure lower because plain two_shot scatters
    # along rows and needs rows % N == 0; _build_two_shot reshapes to
    # (N, numel/N) instead, so the real constraint is numel % N == 0 and
    # decode shapes qualify. Measured at 32 devices, device self time:
    #
    #   payload   XLA    one_shot   two_shot
    #    14 KiB  17.31     13.59     12.36
    #    28 KiB  17.97     22.39     12.67
    #    42 KiB  18.93     33.20     13.20
    #
    # two_shot wins everywhere in that band and is nearly flat in payload
    # (egress 2M(N-1)/N), where one_shot's M(N-1) egress makes it lose past
    # ~16 KiB.
    #
    # The one_shot boundary is 13 KiB, just under the smallest payload where
    # two_shot is measured, because at 14 KiB two_shot is the faster of the two
    # -- 12.36 against one_shot's 13.59, 9% -- and a 16 KiB boundary handed that
    # payload to one_shot. 14 KiB is not a corner case: it is K3's
    # post-attention reduction at a single-token compile bucket, and it is the
    # same kernel invocation as the (2,3584) the row was measured at, since
    # _build_two_shot flattens both to (32, 224). The effect was backwards --
    # halving the tokens from bucket 2 to bucket 1 moved that collective from
    # two_shot at 12.67 to one_shot at 13.59, so a smaller batch reduced
    # *slower*. Below 13 KiB two_shot is genuinely unmeasured and one_shot keeps
    # the small end.
    #
    # two_shot_min moves with it, and must: leaving the floor at 16 KiB while
    # dropping one_shot_max to 13 opens a 13-16 KiB hole that falls through to
    # XLA. That is exactly the gap that sent K3's 42 KiB pair to XLA at 8
    # devices and left the pruned-K3 proxy unable to exercise the concat path.
    #
    # The ceiling is 128 KiB because two_shot is only *locally* flat in
    # payload. Extending the same sweep upwards (job 143, 32 devices, device
    # self time; the slope metric agrees to within 4%):
    #
    #     shape        payload   XLA   two_shot   ratio
    #     (2,10752)     42 KiB  18.95    13.19    1.44x
    #     (16,3584)    112 KiB  21.88    17.38    1.26x
    #     (16,7168)    224 KiB  23.27    25.69    0.91x
    #     (16,10752)   336 KiB  25.99    34.49    0.75x
    #     (16,14336)   448 KiB  26.78    43.71    0.61x
    #     (32,10752)   672 KiB  34.48    65.41    0.53x
    #
    # Past ~112 KiB two_shot is linear in payload (+8.5 us per 112 KiB) while
    # XLA's ring is nearly flat -- 21.88 to 26.78 across four times the bytes.
    # two_shot is bandwidth-bound there and the ring amortises, so they cross
    # at roughly 150 KiB and two_shot loses by more the further past it goes.
    # 128 KiB sits inside the measured winning region rather than on the
    # crossover.
    #
    # This ceiling used to be 256 KiB, which was not a measurement: it was the
    # top of the original sweep's range, exactly as the 64 KiB floor above was
    # the bottom of it. At 256 KiB (16,7168) took a kernel 9% slower than the
    # XLA it replaced, which is the opposite of what this gate exists to do.
    #
    # The 8-device entry is deliberately left alone: it is measured only up to
    # 42 KiB, so its 512 KiB ceiling is an extrapolation of the same kind and
    # should be swept before being relied on above ~64 KiB.
    #
    # 16 devices keeps the 16 KiB one_shot boundary rather than inheriting 32's
    # 13 KiB. The table above is a 32-device sweep, one_shot's egress is M(N-1)
    # and two_shot's 2M(N-1)/N, so the crossover between them genuinely moves
    # with the device count -- and extrapolating a threshold across device
    # counts is how the 42 KiB hole at 8 devices got there.
    16: (16 * 1024, 16 * 1024, 128 * 1024),
    32: (13 * 1024, 13 * 1024, 128 * 1024),
}


class _Plan:
    """What to do with one (shape, dtype) of all-reduce on one group."""

    __slots__ = ("kernel", "op", "buffers", "shape", "dtype")

    def __init__(self,
                 kernel: str,
                 op: Any,
                 buffers: tuple,
                 shape: tuple = (),
                 dtype: Any = None):
        self.kernel = kernel  # "one_shot" | "two_shot" | "xla"
        self.op = op
        self.buffers = buffers
        # The kernel is compiled for exactly this shape; the entry op checks
        # before using it, because a mismatch would re-lower at runtime.
        self.shape = shape
        self.dtype = dtype


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


def donation_enabled() -> bool:
    """Whether to donate the workspace instead of copying it back.

    The ``_ws`` kernels thread a workspace call-to-call. The entry op has always
    updated it with ``copy_`` on the returned buffer, on the assumption -- taken
    from the RPA kernel's KV cache -- that XLA aliases the donation and the copy
    compiles away. In a small graph it does: the ablation harness shows 0.01 ms
    of data formatting across 300 calls. In a real graph it does not. Production
    paid 5.45 ms, 1.82 us per call, and the fusion probe shows the concatenated
    arm saving 10.16 us of collective time per layer while delivering only 5.38
    -- about half of it handed back here.

    ``sharded_jax_op`` already accepts ``donate_argnums``; nothing passed it. So
    this asks JAX to donate the workspace buffers, and the entry op then skips
    the writeback because the buffers were updated in place.

    **Measured, and it is a regression. Leave this off.** Paired fusion-probe
    runs at 32 devices (jobs 130 without, 131 with) put every kernel call 3-4 us
    *slower* and nearly tripled data formatting:

        kernel                no donate   donate
        concat [2,10752]          15.02    18.30
        concat [2,7168]           13.45    16.57
        p_only [2,7168]           14.25    18.01
        data formatting (ms)       0.15     0.44

    So donation adds copies instead of removing them -- presumably XLA
    inserting defensive ones to keep the donated buffer valid for the tensor the
    caller still holds. The xla arm was unchanged across the pair (109.13 vs
    109.03), so the effect is confined to the kernel arms.

    Kept, gated and off, because the hypothesis is an obvious one to retry and
    this records that it was tried. It also disproves the theory it was built to
    test: the gap between concat's collective saving (-10.41 us/layer) and its
    realised layer saving (-5.15) is *not* the copy_ writeback, so that gap is
    still unattributed.

    Correctness held in both runs, so donation is slow rather than unsafe here.
    That was not a given, and it is why the probe's check runs a multi-call
    chain: donation invalidates the donated array in JAX's semantics, and if
    torch_tpu freed the buffer and allocated a fresh one, the workspace the
    caller keeps handing in would go stale -- silent corruption, not an error,
    since the kernel would simply read the wrong slot. The round counter
    alternates slots, so a single call cannot exercise the parity that breaks.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_DONATE", "0") == "1"


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
        # Torch op names must be unique and stable; the group name is both.
        self.op_suffix = str(getattr(communicator, "unique_name",
                                     "tp")).replace(":", "_")
        # The one shape the entry op is wired to reduce with a kernel. Chosen
        # at warmup; everything else goes to XLA. Single rather than
        # shape-dispatched because the buffers have to be *arguments* of the
        # entry op to become graph inputs -- captured tensors are not provided
        # to the compiled program and abort it with
        # "identified an argument that was not provided: DeviceBufferRef".
        self.primary_plan: _Plan | None = None

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
        steady = steady_barrier_enabled()

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
                                              steady_barrier=steady,
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
            # ws, step -- args 1..2. See donation_enabled().
            donate_argnums=(1, 2) if donation_enabled() else None,
        )
        device = self._comm.device
        ws = torch.zeros((2, n - 1, *shape), dtype=dtype, device=device)
        step = torch.zeros(1, dtype=torch.int32, device=device)
        return op, (ws, step)

    def _build_two_shot_variadic(self, shape_a: tuple[int, ...],
                                 shape_b: tuple[int, ...], dtype: torch.dtype):
        """Two-operand two-shot: reduces K3's MoE pair without concatenating.

        The ``torch.cat`` this replaces exists only to turn two collectives into
        one, because two launches measured ~2x one (32.60 vs 17.35 us at 32
        devices, job 265). This gets the single launch without the join, so each
        operand keeps the layout its producer emits.

        Returns ``(op, buffers)`` where ``buffers`` is
        ``(rsa, rsb, aga, agb, step)``.
        """
        import jax

        from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
            two_shot_all_reduce_variadic_ws
        from vllm_torchtpu.kernels.collectives.sharded_jax_op import \
            sharded_jax_op

        mesh = self._mesh
        axis = self._axis
        n = self._world_size
        _REPLICATED = _replicated_spec()
        collective_id = self._next_collective_id
        self._next_collective_id += 1
        steady = steady_barrier_enabled()

        numel_a = 1
        for d in shape_a:
            numel_a *= d
        numel_b = 1
        for d in shape_b:
            numel_b *= d

        def _fn(a, b, rsa, rsb, aga, agb, step):

            def body(a, b, rsa, rsb, aga, agb, step):
                # has_side_effects=False to compile through torch_tpu; see the
                # kernel docstring.
                return two_shot_all_reduce_variadic_ws(
                    a,
                    b,
                    rsa,
                    rsb,
                    aga,
                    agb,
                    step,
                    axis_name=axis,
                    num_devices=n,
                    collective_id=collective_id,
                    steady_barrier=steady,
                    has_side_effects=False)

            return jax.shard_map(body,
                                 mesh=mesh,
                                 in_specs=(_REPLICATED, ) * 7,
                                 out_specs=(_REPLICATED, ) * 7,
                                 check_vma=False)(a, b, rsa, rsb, aga, agb,
                                                  step)

        name = (f"vllm_torchtpu::ll_all_reduce_variadic_{collective_id}")
        op = sharded_jax_op(
            name,
            _fn,
            mesh=mesh,
            input_partition_specs=(_REPLICATED, ) * 7,
            output_partition_specs=(_REPLICATED, ) * 7,
            # rsa, rsb, aga, agb, step -- args 2..6. See donation_enabled().
            donate_argnums=tuple(range(2, 7)) if donation_enabled() else None,
        )
        # Allocated as torch tensors on the comm's device, matching
        # _build_two_shot -- they are op arguments, not jax values. Shapes must
        # mirror make_two_shot_variadic_workspace exactly.
        ca = numel_a // n
        cb = numel_b // n
        device = self._comm.device
        rsa = torch.zeros((2, n - 1, 1, ca), dtype=dtype, device=device)
        rsb = torch.zeros((2, n - 1, 1, cb), dtype=dtype, device=device)
        aga = torch.zeros((2, n, 1, ca), dtype=dtype, device=device)
        agb = torch.zeros((2, n, 1, cb), dtype=dtype, device=device)
        step = torch.zeros(1, dtype=torch.int32, device=device)
        return op, (rsa, rsb, aga, agb, step)

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
        steady = steady_barrier_enabled()

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
            # rs, ag, step -- args 1..3. See donation_enabled().
            donate_argnums=(1, 2, 3) if donation_enabled() else None,
        )
        device = self._comm.device
        rs = torch.zeros((2, n - 1, chunk, cols), dtype=dtype, device=device)
        ag = torch.zeros((2, n, chunk, cols), dtype=dtype, device=device)
        step = torch.zeros(1, dtype=torch.int32, device=device)
        return op, (rs, ag, step)

    def _build_two_shot_pair(self, shape_a: tuple[int, ...],
                             shape_b: tuple[int, ...], dtype: torch.dtype):
        """One two-shot over two operands, joined inside the Pallas kernel.

        Reducing K3's latent-MoE pair as one collective needs both halves in a
        single payload. Doing that with ``torch.cat`` on the caller's side works
        -- measured on K3 job 139 at -3.3% TPOT -- but hands back 301 us per
        decode step, because the cat plus the two slices become ~4400 standalone
        data-formatting ops at roughly 1.1 us each. Copying 42 KiB costs about
        0.04 us, so that cost is op *overhead*: it is the number of ops that
        matters, not the bytes, which is why it barely shrinks with the payload
        and why it is proportionally worse at a single-token bucket, where the
        collective it sits beside is smaller.

        Two ways to remove it were tried. Moving the join into the jitted region
        as a ``jnp.concatenate`` does *not* work: XLA materialises the
        concatenation to HBM either way and merely accounts it in the same
        ``data formatting`` bucket. Job 142, both arms with every other
        collective identical --

            arm                    us/layer   pair kernel   data formatting
            concat (caller cat)      100.59      15.19 us         0.15 ms
            concat_pair (in-jit)     101.53      15.24 us         0.15 ms

        -- formatting unchanged to two decimals, 0.94 us/layer worse overall.
        Relocating a concatenate does not remove it.

        So this builds on ``two_shot_all_reduce_pair_ws`` instead, which takes
        the two operands as separate refs and joins them with a VMEM store
        inside the kernel. No joined tensor is ever materialised, in HBM or
        anywhere else, and the caller passes two tensors and gets two back.
        See ``two_shot_pair_layout`` for the per-device interleave that makes
        the join lane-aligned, and what its ~14% padding buys.

        Both operands must divide across the mesh *independently* -- ``numel %
        n`` for each, not merely for the sum -- so every device's chunk is a
        clean piece of both. K3 qualifies at either decode bucket: 3584 and 7168
        at one token, 7168 and 14336 at two.
        """
        import jax

        from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import (
            two_shot_all_reduce_pair_ws, two_shot_pair_layout)
        from vllm_torchtpu.kernels.collectives.sharded_jax_op import \
            sharded_jax_op

        mesh = self._mesh
        axis = self._axis
        n = self._world_size
        _REPLICATED = _replicated_spec()
        collective_id = self._next_collective_id
        self._next_collective_id += 1

        numel_a = 1
        for d in shape_a:
            numel_a *= d
        numel_b = 1
        for d in shape_b:
            numel_b *= d
        # Raises if either operand does not divide across the mesh.
        _, _, _, cols = two_shot_pair_layout(numel_a, numel_b, n)

        def _fn(
            a: jax.Array, b: jax.Array, rs: jax.Array, ag: jax.Array,
            step: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

            def body(a, b, rs, ag, step):
                return two_shot_all_reduce_pair_ws(a,
                                                   b,
                                                   rs,
                                                   ag,
                                                   step,
                                                   axis_name=axis,
                                                   num_devices=n,
                                                   collective_id=collective_id,
                                                   has_side_effects=False)

            return jax.shard_map(body,
                                 mesh=mesh,
                                 in_specs=(_REPLICATED, ) * 5,
                                 out_specs=(_REPLICATED, ) * 5,
                                 check_vma=False)(a, b, rs, ag, step)

        name = (f"vllm_torchtpu::ll_all_reduce_two_shot_pair_{collective_id}")
        op = sharded_jax_op(
            name,
            _fn,
            mesh=mesh,
            input_partition_specs=(_REPLICATED, ) * 5,
            output_partition_specs=(_REPLICATED, ) * 5,
            donate_argnums=(2, 3, 4) if donation_enabled() else None,
        )
        device = self._comm.device
        # Sized on the padded per-device width: the workspaces carry what the
        # kernel exchanges, which is the interleaved-and-padded layout rather
        # than the bare sum of the two operands.
        rs = torch.zeros((2, n - 1, 1, cols), dtype=dtype, device=device)
        ag = torch.zeros((2, n, 1, cols), dtype=dtype, device=device)
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

# Set by warmup() when TPU_LOW_LATENCY_ALLREDUCE_SITE=rowparallel, and read by
# the patched RowParallelLinear.forward in vllm_torchtpu/__init__.py.
#
# Why a second delivery path exists at all. The communicator's ``_ll_entry``
# routes by *shape*, and on K3 two different reductions share the shape
# [2, 7168]: the post-attention output projection, which is standalone in the
# graph, and the one inside the MoE block, which XLA's combiner fuses with the
# latent [2, 3584] beside it. Wiring the shape captures both, the fused pair
# splits, and the lost fusion costs more than the faster kernel gains -- +5.5%
# device time, measured. Shape cannot separate them because they have the same
# shape; only the call site can.
#
# RowParallelLinear is that discriminator on K3: it issues the 95 standalone
# reductions per decode step and none of the 92 fused ones, which come from
# FusedMoE's reduce_results instead. Delivering the entry op here rather than
# through the communicator therefore reaches exactly the site the kernel helps
# and leaves the combiner's pair untouched.
#
# A module global rather than a context manager or a flag toggled mid-forward:
# warmup() runs before torch.compile traces anything, so by trace time this is
# already final and dynamo folds it to a constant with a guard. A flag that
# changed *during* the forward would be a branch dynamo has to trace, which is
# the thing this whole module is arranged to avoid.
#
# Shape is ``(rows, op, buffers)``: the token bucket the kernel was compiled
# for, then what to call. See ``bucket_matches`` for what the rows are for.
_ROWPARALLEL_ENTRY: tuple | None = None


def bucket_matches(entry: tuple) -> bool:
    """Whether this step's padded token count is the one ``entry`` was built for.

    A kernel is compiled for exactly one shape, and warmup can only wire one of
    them, so on any other token bucket the reduction has to go somewhere else.
    The op already checks its own shape and falls back to ``xla_all_reduce``,
    which is *correct* but not free: the fallback happens inside an opaque
    custom op, so XLA sees a call it cannot fuse, combine, or reorder, and the
    caller has already done whatever work it does to prepare the payload. On
    K3's latent-MoE tail that preparation is a ``torch.cat`` plus two slices per
    layer, which is pure loss when the kernel does not run -- the c4/c8 arms of
    job 139 regressed 5.2% and 4.0% on TPOT for exactly this reason, while c1,
    the bucket the kernel *was* wired for, improved 3.3%.

    So the caller needs to know before it commits, and the token count is the
    one thing it cannot ask for: reading a dim of a traced tensor pins vLLM's
    dynamic token dimension and fails compilation outright (see
    ``get_entry_op``). The forward context is the way around it. The TPU runner
    passes ``num_tokens=num_tokens_padded`` into ``set_forward_context`` --
    already rounded up to a compile bucket, a plain Python int, set *outside*
    anything traced -- so this reads an int and compares it to an int, and
    dynamo folds the whole thing to a constant. It never touches a tensor.

    Reading the forward context from inside a traced forward is upstream's own
    pattern on this exact path: ``MoERunner._encode_layer_name``, called by the
    ``forward`` this guards, branches on
    ``get_forward_context().all_moe_layers is not None``.

    Missing context or a missing descriptor means "cannot tell", which resolves
    to False -- the caller keeps the path it would have had without any of this.
    """
    from vllm.forward_context import (get_forward_context,
                                      is_forward_context_available)

    if not is_forward_context_available():
        return False
    descriptor = get_forward_context().batch_descriptor
    return descriptor is not None and descriptor.num_tokens == entry[0]


# Set by warmup() when TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT=1, and read by the
# patched latent-MoE tail in vllm_torchtpu/__init__.py.
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
# Delivered through its own global rather than the communicator for the same
# reason as _ROWPARALLEL_ENTRY: the call site is explicit, so nothing has to
# infer which reduction this is. Note what that buys -- because the
# concatenated payload is a *different shape* from the post-attention one, the
# communicator can keep routing by shape with no ambiguity, so this needs none
# of the call-site patching that halted the SparseCore.
_MOE_CONCAT_ENTRY: tuple | None = None
# (num_tokens, op, buffers) for the variadic pair reduction.
_MOE_VARIADIC_ENTRY: tuple | None = None


def moe_concat_enabled() -> bool:
    """Whether to reduce K3's latent-MoE pair as one concatenated collective.

    Off by default: it replaces two upstream reductions with one, which is a
    real change to vLLM's latent-MoE tail, and it only pays off where the
    combiner was already fusing the pair.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT", "0") == "1"


def steady_barrier_enabled() -> bool:
    """Whether the _ws kernels keep their steady-state barrier.

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


def moe_variadic_enabled() -> bool:
    """Whether to reduce K3's latent-MoE pair with the variadic kernel.

    Off by default. It supersedes TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT rather
    than composing with it: both wire the same call site, and this one removes
    the ``torch.cat`` that one depends on. Requires the concat path to be
    enabled too, since that is what patches the MoE tail at all.
    """
    import os

    return os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC", "0") == "1"


def get_moe_variadic_op(dispatcher: LowLatencyAllReduce, shape_a: tuple[int,
                                                                        ...],
                        shape_b: tuple[int, ...], dtype: torch.dtype) -> Any:
    """An opaque op that reduces the latent-MoE pair as two operands.

    Same reasoning as ``get_moe_concat_op`` for why this is a custom_op bound to
    fixed shapes rather than anything that inspects dims: reading a dim in
    traced Python pins vLLM's dynamic token dimension and fails compilation.
    """
    name = (f"vllm_torchtpu::ll_all_reduce_moe_variadic_"
            f"{dispatcher.op_suffix}")
    existing = _ENTRY_OPS.get(name)
    if existing is not None:
        return existing

    inner, buffers = dispatcher._build_two_shot_variadic(
        shape_a, shape_b, dtype)
    donate = donation_enabled()
    expect_a, expect_b = tuple(shape_a), tuple(shape_b)

    def _impl(a: torch.Tensor, b: torch.Tensor, rsa: torch.Tensor,
              rsb: torch.Tensor, aga: torch.Tensor, agb: torch.Tensor,
              step: torch.Tensor) -> list[torch.Tensor]:
        if (tuple(a.shape) == expect_a and tuple(b.shape) == expect_b
                and a.dtype == dtype and b.dtype == dtype
                and a.is_contiguous() and b.is_contiguous()):
            out = inner(a, b, rsa, rsb, aga, agb, step)
            oa, ob, new = out[0], out[1], out[2:]
            if not donate:
                for dst, src in zip((rsa, rsb, aga, agb, step), new):
                    dst.copy_(src)
            return [oa, ob]
        # Any shape we were not built for falls back; correctness first.
        return [
            dispatcher.xla_all_reduce(a),
            dispatcher.xla_all_reduce(b),
        ]

    op = torch.library.custom_op(name, _impl, mutates_args=())
    op.register_fake(
        lambda a, b, rsa, rsb, aga, agb, step:
        [torch.empty_like(a), torch.empty_like(b)])
    _ENTRY_OPS[name] = op
    return op, buffers


def get_entry_op(dispatcher: LowLatencyAllReduce) -> Any:
    """An opaque torch op that reduces a tensor, built once per group.

    Two constraints meet here and between them leave only this shape.

    Dynamo must not see the shape. Selecting a kernel in traced Python means
    reading dims, and reading vLLM's dynamic token dimension pins it and fails
    the whole compilation ("Constraints violated (L['input_ids'].size()[0])").
    No guard avoids it: dynamo presents a SymInt as an int, so even
    ``isinstance(dim, int)`` is True. ``torch.library.custom_op`` is not traced
    into, so the graph gets one call node and the dims are read later, at
    execution, when they are ordinary ints.

    The workspaces must be *arguments*. They cannot be captured inside the
    implementation: a tensor the compiled program references but was never
    handed aborts it with "identified an argument that was not provided:
    DeviceBufferRef". So the op takes the primary plan's ``ws`` and ``step``
    directly, which is also why exactly one shape is wired up rather than a
    dispatch table -- a second shape would need a second pair of arguments.

    Anything that is not that shape reduces through XLA, which is always
    correct.
    """
    # one_shot carries (ws, step); two_shot carries (rs, ag, step). A custom
    # op's schema is fixed at registration, so the arity is chosen here from
    # the plan that warmup already selected -- registering both and picking at
    # call time would put a branch in front of the op, which is exactly what
    # dynamo must not see.
    plan0 = dispatcher.primary_plan
    n_buffers = len(plan0.buffers) if plan0 is not None else 2
    name = (f"vllm_torchtpu::ll_all_reduce_{dispatcher.op_suffix}"
            f"{'_ts' if n_buffers == 3 else ''}")
    existing = _ENTRY_OPS.get(name)
    if existing is not None:
        return existing

    # Read once, here, rather than per call: the value is fixed before anything
    # is traced, and reading it inside the implementation would put an env
    # lookup on the critical path of every reduction.
    donate = donation_enabled()

    def _impl(x: torch.Tensor, ws: torch.Tensor,
              step: torch.Tensor) -> torch.Tensor:
        plan = dispatcher.primary_plan
        if (plan is not None and tuple(x.shape) == plan.shape
                and x.dtype == plan.dtype and x.is_contiguous()):
            out, new_ws, new_step = plan.op(x, ws, step)
            # Without donation the workspace has to be written back explicitly:
            # the op returns fresh buffers, and the kernel needs the *same*
            # buffer next call for its slot parity to mean anything. With
            # donation the op wrote through them already, and copying would
            # only pay for it twice. See donation_enabled().
            if not donate:
                ws.copy_(new_ws)
                step.copy_(new_step)
            return out
        return dispatcher.xla_all_reduce(x)

    def _impl_two_shot(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
                       step: torch.Tensor) -> torch.Tensor:
        plan = dispatcher.primary_plan
        if (plan is not None and tuple(x.shape) == plan.shape
                and x.dtype == plan.dtype and x.is_contiguous()):
            out, new_rs, new_ag, new_step = plan.op(x, rs, ag, step)
            if not donate:
                rs.copy_(new_rs)
                ag.copy_(new_ag)
                step.copy_(new_step)
            return out
        return dispatcher.xla_all_reduce(x)

    if n_buffers == 3:
        op = torch.library.custom_op(name, _impl_two_shot, mutates_args=())
        op.register_fake(lambda x, rs, ag, step: torch.empty_like(x))
        _ENTRY_OPS[name] = op
        return op

    # mutates_args stays empty even though the implementation writes the
    # workspace in place. Declaring the mutation makes AOTAutograd add both
    # buffers to the graph's outputs, and the compiled artifact then returns
    # one where the runtime wrapper expects three:
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
    op.register_fake(lambda x, ws, step: torch.empty_like(x))
    _ENTRY_OPS[name] = op
    return op


def get_moe_concat_op(dispatcher: LowLatencyAllReduce, plan: _Plan) -> Any:
    """An opaque op that reduces the concatenated latent-MoE pair.

    Bound to one plan directly, rather than reading ``primary_plan`` the way
    ``get_entry_op`` does, because two shapes are wired at once here and each op
    must know which is its own. Two ops rather than one shape-dispatching op for
    the reason the whole module is arranged around: choosing between them would
    mean reading a dim in traced Python, which pins vLLM's dynamic token
    dimension and fails compilation outright.

    two-shot only. The concatenated payload is 42 KiB at K3's widths, well past
    one-shot's crossover, so a one-shot plan here would be a bug rather than a
    fallback -- and its arity differs, which a fixed op schema cannot express.
    """
    if len(plan.buffers) != 3:
        raise ValueError(
            f"the concatenated MoE reduction needs a two_shot plan with three "
            f"buffers, got {plan.kernel} with {len(plan.buffers)}")

    name = f"vllm_torchtpu::ll_all_reduce_moe_concat_{dispatcher.op_suffix}"
    existing = _ENTRY_OPS.get(name)
    if existing is not None:
        return existing

    donate = donation_enabled()

    def _impl(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
              step: torch.Tensor) -> torch.Tensor:
        if (tuple(x.shape) == plan.shape and x.dtype == plan.dtype
                and x.is_contiguous()):
            out, new_rs, new_ag, new_step = plan.op(x, rs, ag, step)
            if not donate:
                rs.copy_(new_rs)
                ag.copy_(new_ag)
                step.copy_(new_step)
            return out
        # Any other shape is a token count this plan was not built for. XLA is
        # always correct, and the caller has already concatenated, so this still
        # produces the pair's sum -- just without the kernel.
        return dispatcher.xla_all_reduce(x)

    op = torch.library.custom_op(name, _impl, mutates_args=())
    op.register_fake(lambda x, rs, ag, step: torch.empty_like(x))
    _ENTRY_OPS[name] = op
    return op


def _moe_concat_width(vllm_config) -> int | None:
    """Width of the concatenated latent-MoE pair, or None if the model has none.

    The two reductions carry the hidden size and the routed-expert latent width,
    so reducing them together means one payload of their sum. Returns None
    unless the model actually has both, which is what makes it a latent MoE.
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


def _moe_latent_width(vllm_config) -> int | None:
    """Routed-expert latent width, or None. The narrower half of the pair."""
    model_config = vllm_config.model_config
    hf = getattr(model_config, "hf_text_config",
                 None) or model_config.hf_config
    latent = getattr(hf, "routed_expert_hidden_size", None)
    return latent if isinstance(latent, int) and latent > 0 else None


def _moe_hidden_width(vllm_config) -> int | None:
    """Hidden size, or None. The wider half of the pair."""
    model_config = vllm_config.model_config
    hf = getattr(model_config, "hf_text_config",
                 None) or model_config.hf_config
    hidden = getattr(hf, "hidden_size", None)
    return hidden if isinstance(hidden, int) and hidden > 0 else None


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

    # The concatenated pair, when the tail is going to reduce them together.
    # Generated unconditionally rather than behind the flag so that turning the
    # flag on never has to build a plan late, inside a traced region.
    concat = _moe_concat_width(vllm_config)
    if concat is not None and concat not in widths:
        widths.append(concat)

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
    import os

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

    kernel_plans = []
    for rows, cols in shapes:
        probe = torch.empty((rows, cols),
                            dtype=dtype,
                            device=communicator.device)
        plan = dispatcher._build_plan(probe, ((rows, cols), dtype))
        del probe
        if plan.kernel != "xla":
            kernel_plans.append(plan)

    # The concatenated latent-MoE pair, wired separately from the
    # communicator's shape. Its call site is explicit -- the patched MoE tail --
    # so it gets its own op and its own workspaces, and the two never compete.
    #
    # Held out of the primary selection below on purpose. It is the largest
    # payload that takes a kernel, so "smallest" would not pick it anyway, but
    # more importantly the communicator must never route to it: a token count
    # whose hidden-width payload happened to equal the concat width would
    # otherwise reduce the wrong pair.
    concat_width = _moe_concat_width(vllm_config)
    concat_plan = None
    if moe_concat_enabled() and concat_width is not None:
        candidates = [p for p in kernel_plans if p.shape[1] == concat_width]
        concat_plan = min(candidates,
                          key=lambda p: p.shape[0] * p.shape[1],
                          default=None)
        if concat_plan is None:
            logger.warning(
                "TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT is set but no width-%d "
                "shape takes a kernel (candidates %s); the MoE tail keeps its "
                "two separate reductions.", concat_width,
                [p.shape for p in kernel_plans])
        elif concat_plan.kernel != "two_shot":
            logger.warning(
                "The concatenated MoE shape %s chose %s, not two_shot; at this "
                "payload one-shot loses badly, so the tail keeps its two "
                "separate reductions. Is "
                "TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT set?", concat_plan.shape,
                concat_plan.kernel)
            concat_plan = None
    if concat_width is not None:
        kernel_plans = [p for p in kernel_plans if p.shape[1] != concat_width]

    # Exactly one *other* shape can be wired to the communicator's entry op (its
    # workspaces are op arguments), so pick the smallest payload that takes a
    # kernel: that is the low-concurrency decode reduction this is aimed at.
    # Everything else reduces through XLA.
    override = os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_SHAPE", "").strip()
    chosen = None
    if override:
        try:
            want = tuple(int(v) for v in override.lower().split("x"))
            chosen = next(p for p in kernel_plans if p.shape == want)
        except (ValueError, StopIteration):
            logger.warning(
                "TPU_LOW_LATENCY_ALLREDUCE_SHAPE=%s does not match any shape "
                "that takes a kernel (%s); falling back to the smallest.",
                override, [p.shape for p in kernel_plans])
    if chosen is None and kernel_plans:
        chosen = min(kernel_plans, key=lambda p: p.shape[0] * p.shape[1])
    dispatcher.primary_plan = chosen

    # Register the entry op here too, not on first use. Registering it is what
    # torch.library.custom_op does, and that calls infer_schema, which dynamo
    # refuses to trace ("Attempted to call function marked as skipped"). First
    # use is inside the traced forward, so lazy registration is fatal.
    #
    # Everything the traced path needs is then stashed as one attribute, so
    # what dynamo sees in TpuDeviceCommunicator.all_reduce is an attribute
    # read and an opaque op call -- no lookups, no string work, nothing that
    # can register or allocate.
    # Which call sites the kernel is delivered to. "all" routes every
    # reduction on this group through the communicator, which is what the
    # kernels were first wired up as; "rowparallel" routes only
    # RowParallelLinear's, leaving XLA's collective combiner free to fuse the
    # MoE pair. See _ROWPARALLEL_ENTRY for why shape alone cannot separate the
    # two sites. Defaults to "all" so this changes nothing until asked for.
    site = os.environ.get("TPU_LOW_LATENCY_ALLREDUCE_SITE",
                          "all").strip().lower()
    if site not in ("all", "rowparallel"):
        logger.warning(
            "TPU_LOW_LATENCY_ALLREDUCE_SITE=%s is not 'all' or "
            "'rowparallel'; using 'all'.", site)
        site = "all"

    global _ROWPARALLEL_ENTRY, _MOE_CONCAT_ENTRY, _MOE_VARIADIC_ENTRY
    _ROWPARALLEL_ENTRY = None
    _MOE_CONCAT_ENTRY = None
    _MOE_VARIADIC_ENTRY = None
    communicator._ll_entry = None

    # Entries carry the token bucket they were compiled for as element 0, so a
    # caller can ask bucket_matches() *before* preparing a payload the kernel
    # will not accept. See bucket_matches for why the op's own shape check is
    # not enough on its own.
    if concat_plan is not None:
        # Registered here, with everything else, because registration calls
        # infer_schema, which dynamo refuses to trace -- doing it on first use
        # would put it inside the traced forward and abort compilation.
        _MOE_CONCAT_ENTRY = (concat_plan.shape[0],
                             get_moe_concat_op(dispatcher, concat_plan),
                             concat_plan.buffers)
        logger.info(
            "Low-latency all-reduce: latent-MoE pair wired to one %s over %s; "
            "other token buckets keep upstream's two reductions.",
            concat_plan.kernel, concat_plan.shape)

        # The variadic kernel supersedes the concat for the same call site: it
        # reduces both operands in one launch without joining them, so the
        # torch.cat and both reshapes at the call site disappear. Built from the
        # component widths rather than concat_plan's summed one.
        if moe_variadic_enabled():
            latent = _moe_latent_width(vllm_config)
            hidden = _moe_hidden_width(vllm_config)
            rows = concat_plan.shape[0]
            if latent is None or hidden is None:
                logger.warning(
                    "TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC is set but the "
                    "component widths could not be read; keeping the concat.")
            elif (latent * rows) % dispatcher._world_size or (
                    hidden * rows) % dispatcher._world_size:
                logger.warning(
                    "Variadic pair needs both element counts divisible by %d; "
                    "got %d and %d. Keeping the concat.",
                    dispatcher._world_size, latent * rows, hidden * rows)
            elif (
                (latent * rows) // dispatcher._world_size == (hidden * rows) //
                    dispatcher._world_size):
                # The kernel corrupts both operands when their per-device chunk
                # widths match; it raises rather than risk it, so decline here
                # and keep the concat instead of taking down warmup. K3 is
                # 112 against 224 and unaffected. See the kernel's ValueError.
                logger.warning(
                    "Variadic pair declined: latent and hidden give equal "
                    "per-device chunks (%d), which the kernel does not support. "
                    "Keeping the concat.",
                    (latent * rows) // dispatcher._world_size)
            else:
                op, buffers = get_moe_variadic_op(dispatcher, (rows, latent),
                                                  (rows, hidden),
                                                  concat_plan.dtype)
                _MOE_VARIADIC_ENTRY = (rows, op, buffers)
                logger.info(
                    "Low-latency all-reduce: latent-MoE pair wired to the "
                    "VARIADIC two_shot over (%d,%d)+(%d,%d) -- no concat, no "
                    "reshape at the call site.", rows, latent, rows, hidden)

    if chosen is not None:
        # get_entry_op reads primary_plan to fix the op's arity, so it must be
        # set first. The buffers go through positionally -- two for one_shot,
        # three for two_shot -- because they have to be op *arguments* to
        # reach the compiled program at all.
        entry_op = get_entry_op(dispatcher)
        entry = (chosen.shape[0], entry_op, chosen.buffers)
        if site == "rowparallel":
            _ROWPARALLEL_ENTRY = entry
        else:
            communicator._ll_entry = entry

    logger.info(
        "Low-latency all-reduce warmup: %d of %d candidate shapes take a "
        "kernel; wired to %s (%s) at site=%s. Every other shape uses XLA.",
        len(kernel_plans), len(shapes), chosen.shape if chosen else None,
        chosen.kernel if chosen else "none", site)


def get_dispatcher(communicator) -> LowLatencyAllReduce:
    """One dispatcher per communicator instance."""
    existing = getattr(communicator, "_ll_all_reduce", None)
    if existing is None:
        existing = LowLatencyAllReduce(communicator)
        communicator._ll_all_reduce = existing
    return existing
