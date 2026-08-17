#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decompose the two-shot all-reduce's 12.67 us into its parts.

The kernel costs 12.67 us/occ at 2x7168 bf16 on 32 devices, of which a line fit
across 14/28/42 KiB says ~11.9 us is fixed and ~0.8 us is payload. So "is this
implementation good?" is almost entirely "is ~11.9 us a reasonable fixed cost
for two synchronized phases across 32 devices?", and that cannot be answered by
staring at the total.

This runs a ladder of kernels that are *byte-identical to the real one in
everything except which parts of the body execute*: same pallas_call signature,
same scratch, same input_output_aliases, same compiler params, same shard_map,
same torch custom-op wrapper, same chain. Only the body differs. Subtracting
adjacent rungs attributes the time:

    noop        launch + VMEM in/out, nothing else        -> launch floor
    barrier1    + one all-to-all barrier                  -> barrier1 - noop
    barrier2    + the double barrier the kernel uses      -> barrier2 - barrier1
    p1          real kernel minus phase 2's remote sends  -> p1 - barrier2
    full        replica of two_shot_all_reduce_ws         -> full - p1
    full_1bar   full with a single barrier                -> what a merge saves
    full_bf     full with barrier_first=True              -> what the overlap is worth

``full`` is the control for the whole exercise: if it does not reproduce the
12.67 us the production harness measured, the replica has drifted and every
subtraction below it is meaningless. It is also the only rung checked for
correctness -- the others deliberately compute the wrong answer, which is the
point.

Read the result as a floor: the recoverable slack is
``full - (noop + barriers that cannot be removed + transit)``. If that is under
a microsecond, the kernel is done and effort belongs elsewhere (fewer calls,
not faster ones).

Measured (jobs 118/119, 2x7168 bf16, 32 devices, us/occ): noop 1.42,
barrier1 4.35, barrier2 7.27, p1 8.00, full 12.65, full_1bar 11.08,
full_bf 16.77, xla 17.99. So each barrier costs ~2.93 us and the two of them
are 46% of the kernel. See K3_LOW_LATENCY_ALLREDUCE_RESULTS.md.

Known wart: ``noop`` reports fewer occurrences than the other rungs (180 against
300), because an identity kernel carrying no collective_id and no side effects
is the one rung XLA is free to eliminate calls from. Its per-occurrence number
is still self time over calls that ran, and is corroborated by barrier #1 and
barrier #2 landing 0.01 us apart when both are derived through it -- but treat
it as approximate in a way the other rungs are not.

Run (32 devices, 4 hosts):
    sbatch --partition=ici-2x2x4-b scripts/debug/two_shot_ablation.sbatch
"""

import argparse
import functools
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device and backends
from torch.distributed._functional_collectives import \
    all_reduce as funcol_all_reduce
from torch_tpu._internal.profiler import TpuProfilerConfig

from vllm_torchtpu import profiler_trace

# Rungs in ladder order. Each is a body variant of _two_shot_ws_kernel; see the
# module docstring for what the differences between neighbours attribute to.
MODES = ("noop", "barrier1", "barrier2", "p1", "full", "full_1bar", "full_bf")


class _StubCommunicator:
    """The attributes LowLatencyAllReduce reads off a real communicator."""

    def __init__(self, cpu_group, device_group, rank_in_group: int,
                 world_size: int):
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.device = torch.device("tpu")
        self.unique_name = "tp:0"


def _ablation_kernel(x_ref, rs_in_ref, ag_in_ref, step_in_ref, o_ref, rs_ref,
                     ag_ref, step_ref, recv_vmem, red_vmem, local_sem, s1_send,
                     s1_recv, s2_send, s2_recv, *, num_devices: int,
                     axis_name: str, mode: str):
    """One body for every rung; ``mode`` selects how far down the ladder it goes.

    Deliberately a copy of ``_two_shot_ws_kernel`` rather than a refactor of it.
    The production kernel is what the 12.67 us was measured on, and a diagnostic
    that reaches into it can only make that number harder to trust.
    """
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
        _full_barrier

    del rs_in_ref, ag_in_ref  # aliased onto the output views
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)

    # Rungs below the data path: barrier (if any) then straight to the output,
    # so the VMEM in/out traffic matches the full kernel's and cancels in the
    # subtraction. The barrier sits ahead of everything here because there are
    # no sends for it to overlap; that makes barrier2 an *upper* bound on what
    # synchronization contributes to the full kernel, which is the bound worth
    # having.
    if mode in ("noop", "barrier1", "barrier2"):
        if mode == "barrier1":
            _full_barrier(my_id, num_devices, double_barrier=False)
        elif mode == "barrier2":
            _full_barrier(my_id, num_devices, double_barrier=True)
        o_ref[...] = x_ref[...]
        step_ref[0] = step + 1
        return

    double = mode != "full_1bar"
    barrier_first = mode == "full_bf"

    if barrier_first:
        _full_barrier(my_id, num_devices, double_barrier=double)
    else:

        @pl.when(step == 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=double)

    # Phase 1 -- reduce-scatter.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[peer],
            dst_ref=rs_ref.at[slot, i - 1],
            send_sem=s1_send.at[slot],
            recv_sem=s1_recv.at[slot],
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    if not barrier_first:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=double)

    for copy in copies:
        copy.wait()

    local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[my_id]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather. ``p1`` keeps the local write and the copy-out but
    # sends nothing, so subtracting it from ``full`` isolates the phase-2 wire
    # time and nothing else.
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[slot, my_id],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    if mode != "p1":
        for i in range(1, num_devices):
            copy = pltpu.make_async_remote_copy(
                src_ref=red_vmem,
                dst_ref=ag_ref.at[slot, my_id],
                send_sem=s2_send.at[slot],
                recv_sem=s2_recv.at[slot],
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
    local_ag.wait()
    for copy in copies:
        copy.wait()

    out_copy = pltpu.make_async_copy(src_ref=ag_ref.at[slot],
                                     dst_ref=o_ref,
                                     sem=local_sem)
    out_copy.start()
    out_copy.wait()
    step_ref[0] = step + 1


def _ablation_call(x, rs, ag, step, *, axis_name: str, num_devices: int,
                   collective_id: int, mode: str):
    """``two_shot_all_reduce_ws``'s pallas_call, with the ablation body."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, cols = x.shape
    chunk = rows // num_devices

    kernel = functools.partial(_ablation_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               mode=mode)
    out, rs, ag, step = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_devices, chunk, cols), x.dtype),
            jax.ShapeDtypeStruct(rs.shape, rs.dtype),
            jax.ShapeDtypeStruct(ag.shape, ag.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        input_output_aliases={
            1: 1,
            2: 2,
            3: 3
        },
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, chunk, cols), x.dtype),
            pltpu.VMEM((chunk, cols), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(
            # Mosaic rejects a collective_id on a kernel that never takes the
            # barrier semaphore ("collective_id has to be unspecified or None
            # when not using a custom barrier"), and noop is exactly that. It
            # still issues the remote-copy-free body through the identical
            # pallas_call, so the launch floor it measures stays comparable.
            collective_id=None if mode == "noop" else collective_id,
            # False for the same reason the production kernel needs it: XLA's
            # partitioner rejects a side-effecting custom call inside the
            # manual region shard_map lowers to.
            has_side_effects=False,
            disable_semaphore_checks=True),
    )(x.reshape(num_devices, chunk, cols), rs, ag, step)
    return out.reshape(rows, cols), rs, ag, step


def build_ablation_op(dispatcher, shape, dtype, mode: str, collective_id: int):
    """A torch op for one rung, wired exactly as ``_build_two_shot`` wires the real one."""
    import jax

    from vllm_torchtpu.distributed.low_latency_all_reduce import \
        _replicated_spec
    from vllm_torchtpu.kernels.collectives.sharded_jax_op import sharded_jax_op

    mesh = dispatcher._mesh
    axis = dispatcher._axis
    n = dispatcher._world_size
    spec = _replicated_spec()

    numel = 1
    for d in shape:
        numel *= d
    cols = numel // n
    chunk = 1

    def _fn(x: jax.Array, rs: jax.Array, ag: jax.Array, step: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        def body(x, rs, ag, step):
            flat = x.reshape(n, cols)
            out, rs, ag, step = _ablation_call(flat,
                                               rs,
                                               ag,
                                               step,
                                               axis_name=axis,
                                               num_devices=n,
                                               collective_id=collective_id,
                                               mode=mode)
            return out.reshape(x.shape), rs, ag, step

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(spec, ) * 4,
                             out_specs=(spec, ) * 4,
                             check_vma=False)(x, rs, ag, step)

    jax_op = sharded_jax_op(
        f"vllm_torchtpu::ll_ablation_{mode}_{collective_id}",
        _fn,
        mesh=mesh,
        input_partition_specs=(spec, ) * 4,
        output_partition_specs=(spec, ) * 4,
    )

    device = dispatcher._comm.device
    rs = torch.zeros((2, n - 1, chunk, cols), dtype=dtype, device=device)
    ag = torch.zeros((2, n, chunk, cols), dtype=dtype, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)

    # Same contract as the production entry op: opaque to dynamo, buffers as
    # arguments rather than captures, mutation undeclared. See get_entry_op.
    name = f"vllm_torchtpu::ll_abl_{mode}_{collective_id}"

    def _impl(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
              step: torch.Tensor) -> torch.Tensor:
        out, nrs, nag, nstep = jax_op(x, rs, ag, step)
        rs.copy_(nrs)
        ag.copy_(nag)
        step.copy_(nstep)
        return out

    op = torch.library.custom_op(name, _impl, mutates_args=())
    op.register_fake(lambda x, rs, ag, step: torch.empty_like(x))
    return op, (rs, ag, step)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape",
                   default="2x7168",
                   help="The post-attention reduction. One shape only: this "
                   "decomposes a cost, it does not sweep.")
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--profile-reps", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # One code object compiled once per (mode, chain length); the default limit
    # of 8 aborts partway through the ladder.
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    from vllm_torchtpu.distributed.low_latency_all_reduce import \
        LowLatencyAllReduce

    comm = _StubCommunicator(gloo, dist.group.WORLD, rank, world_size)
    dispatcher = LowLatencyAllReduce(comm)

    device = torch.device("tpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    itemsize = torch.finfo(dtype).bits // 8
    r, _, c = args.shape.strip().lower().partition("x")
    rows, cols = int(r), int(c)
    numel = rows * cols
    payload = numel * itemsize

    def lead(*a):
        if rank == 0:
            print(*a, flush=True)

    if dispatcher._get_mesh() is None:
        lead("!! no mesh; cannot build any kernel")
        dist.destroy_process_group()
        return 2
    if numel % world_size:
        lead(f"!! numel {numel} not divisible by {world_size}")
        dist.destroy_process_group()
        return 2

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    lead(f"world_size={world_size} shape={rows}x{cols} dtype={args.dtype} "
         f"payload={payload / 1024:.1f} KiB")
    lead(f"slope {args.iters_lo}->{args.iters_hi}, median of {args.repeats}\n")

    def time_fn(fn, x) -> float:
        for _ in range(args.warmup):
            fn(x).cpu()
        samples = []
        for _ in range(args.repeats):
            dist.barrier()
            t0 = time.perf_counter()
            fn(x).cpu()
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    def profile_chain(fn, x, tag: str) -> None:
        run_dir = os.path.join(args.profile_dir, tag)
        cap_dir = profiler_trace.rank_capture_dir(run_dir, rank)
        os.makedirs(cap_dir, exist_ok=True)
        fn(x).cpu()
        dist.barrier()
        cfg = TpuProfilerConfig(
            run_dir=cap_dir,
            host_tracer_level=2,
            device_tracer_level=1,
            python_tracer_level=1,
            experimental_options={
                "tpu_trace_mode": "TRACE_COMPUTE",
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            },
        )
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=cap_dir, use_gzip=True),
                experimental_config=cfg,
        ):
            for _ in range(args.profile_reps):
                fn(x).cpu()
        dist.barrier()

    x = torch.randn(rows, cols, dtype=torch.float32).to(dtype).to(device)
    results = {}
    failures = []

    def run_arm(label: str, reduce_fn) -> None:

        def build(n_ops: int):

            def f(t):
                for _ in range(n_ops):
                    t = reduce_fn(t)
                    t = t * (1.0 / world_size)
                return t

            return torch.compile(f,
                                 backend="tpu",
                                 fullgraph=True,
                                 dynamic=False)

        lo = build(args.iters_lo)
        hi = build(args.iters_hi)
        t_lo = time_fn(lo, x)
        t_hi = time_fn(hi, x)
        slope = ((t_hi - t_lo) / (args.iters_hi - args.iters_lo)) * 1e6
        results[label] = slope
        lead(f"{label:>12} {slope:>9.2f} us/call")
        if args.profile_dir:
            profile_chain(hi, x, label)

    # XLA in the same conditions, as the reference the ladder is read against.
    run_arm("xla",
            lambda t: funcol_all_reduce(t, "sum", list(range(world_size))))

    for i, mode in enumerate(modes):
        try:
            op, buffers = build_ablation_op(dispatcher, (rows, cols), dtype,
                                            mode, 40 + i)
        except Exception as e:  # noqa: BLE001
            lead(f"{mode:>12} !! build failed: {type(e).__name__}: {e}")
            continue
        rs, ag, step = buffers

        def reduce_fn(t, op=op, rs=rs, ag=ag, step=step):
            return op(t.contiguous(), rs, ag, step)

        # Only `full` is supposed to be right. The other rungs omit parts of
        # the algorithm on purpose, so checking them would only report the
        # ablation back at us.
        if mode == "full":
            one = torch.compile(reduce_fn,
                                backend="tpu",
                                fullgraph=True,
                                dynamic=False)
            got = one(x)
            want = funcol_all_reduce(x, "sum", list(range(world_size)))
            d = (got.float() - want.float()).abs().max().cpu().item()
            scale = max(1.0, want.float().abs().max().cpu().item())
            tol = 1e-5 if dtype == torch.float32 else 0.01 * world_size**0.5
            if d / scale >= tol:
                failures.append(f"full: rel={d / scale:.5f} tol={tol:.5f}")

        try:
            run_arm(mode, reduce_fn)
        except Exception as e:  # noqa: BLE001
            lead(f"{mode:>12} !! run failed: {type(e).__name__}: {e}")

    # The subtractions the ladder exists for. Slope only -- the profiled
    # per-occurrence numbers come out of allreduce_profile_stats.py, and the
    # two agreed within 0.3 us on the production harness.
    def delta(a: str, b: str) -> str:
        if a not in results or b not in results:
            return "     n/a"
        return f"{results[a] - results[b]:>8.2f}"

    lead("\nattribution (us):")
    lead(
        f"  launch + VMEM in/out          {results.get('noop', float('nan')):>8.2f}"
    )
    lead(f"  barrier #1                    {delta('barrier1', 'noop')}")
    lead(f"  barrier #2                    {delta('barrier2', 'barrier1')}")
    lead(f"  phase 1 (reduce-scatter)      {delta('p1', 'barrier2')}")
    lead(f"  phase 2 (all-gather sends)    {delta('full', 'p1')}")
    lead("  ---")
    lead(f"  saved by merging the barriers {delta('full', 'full_1bar')}")
    lead(f"  gained by overlapping barrier {delta('full_bf', 'full')}")
    lead(f"  full vs xla                   {delta('full', 'xla')}")

    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, failures, group=gloo)
    every = [f for per_rank in gathered for f in per_rank]
    if rank == 0 and every:
        print("\nCORRECTNESS FAILURES:", flush=True)
        for f in every:
            print(f"  {f}", flush=True)
    dist.destroy_process_group()
    return 1 if every else 0


if __name__ == "__main__":
    raise SystemExit(main())
