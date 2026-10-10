# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the candidate collective through the serving TorchTPU bridge."""

import argparse
import contextlib
import importlib.util
import json
import os
import socket
import statistics
import time
from pathlib import Path


def worker(local_rank: int, args: argparse.Namespace) -> None:
    rank = args.rank * 8 + local_rank
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(local_rank),
        LOCAL_WORLD_SIZE="8",
        WORLD_SIZE="32",
        GROUP_RANK=str(args.rank),
    )
    import torch
    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh

    torch.set_num_threads(1)
    dist.init_process_group("tpu")
    cpu_group = dist.new_group(backend="gloo")
    hosts = [None] * 32
    dist.all_gather_object(hosts, socket.gethostname(), group=cpu_group)
    mesh = get_pallas_mesh(axis_names=("moe_host",), mesh_shape=(32,))
    spec = importlib.util.spec_from_file_location(
        "host_reduce", Path(__file__).with_name("host_hierarchical_reduce.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = module.build_host_reduce(mesh, hosts)
    if rank == 0:
        print(
            json.dumps(
                {
                    "phase": "torch_mesh",
                    "groups": module.host_replica_groups(
                        hosts, [int(d.id) for d in mesh.devices.flat]
                    ),
                }
            ),
            flush=True,
        )

    def flat(x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    def capture(directory):
        if not args.profile or rank != 0:
            return contextlib.nullcontext()
        from torch_tpu._internal.profiler import TpuProfilerConfig

        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.PrivateUse1],
            experimental_config=TpuProfilerConfig(
                run_dir=str(directory),
                host_tracer_level=0,
                device_tracer_level=1,
                python_tracer_level=0,
            ),
        )

    with capture(args.output.parent / "profile-bridge-prime"):
        pass
    dist.barrier(group=cpu_group)
    results = []
    for batch in args.batches:
        torch.manual_seed(1900 + rank + batch)
        x_cpu = (torch.randn(batch, 8192) * 0.02).to(torch.bfloat16)
        want = x_cpu.float()
        dist.all_reduce(want, group=cpu_group)
        x = x_cpu.to("tpu")
        paths = [("flat", flat), ("host", candidate)]
        if args.reverse:
            paths.reverse()
        for name, fn in paths:
            # Each shape is an independent compile; do not exhaust Dynamo's
            # per-code-object recompile limit during a long shape sweep.
            torch._dynamo.reset()

            def forward(x, fn=fn):
                return fn(x)

            compiled = torch.compile(
                forward, backend="tpu", fullgraph=True, dynamic=False
            )
            # Exact rank sum catches duplication, missing ranks and shard ordering.
            exact_cpu = torch.zeros_like(x_cpu)
            exact_cpu[:, rank::32] = 1
            exact = exact_cpu.to("tpu")
            exact_got = compiled(exact).cpu()
            assert exact_got.shape == x_cpu.shape
            assert torch.equal(exact_got, torch.ones_like(exact_got))
            got = compiled(x).cpu().float()
            assert got.shape == want.shape
            error = float(
                torch.linalg.vector_norm(got - want) / torch.linalg.vector_norm(want)
            )
            max_abs = float((got - want).abs().max())
            assert error < 0.03 and max_abs < 0.05, (name, batch, error, max_abs)

            def chain(x, fn=fn):
                for _ in range(args.inner):
                    x = x + fn(x) * 0.0001
                return x

            executable = torch.compile(
                chain, backend="tpu", fullgraph=True, dynamic=False
            )
            for _ in range(3):
                executable(x)
                torch.accelerator.synchronize()
            times = []
            for _ in range(args.repeats):
                dist.barrier(group=cpu_group)
                start = time.perf_counter()
                executable(x)
                torch.accelerator.synchronize()
                elapsed = torch.tensor((time.perf_counter() - start) * 1e6 / args.inner)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=cpu_group)
                times.append(float(elapsed))
            row = dict(
                path=name,
                batch=batch,
                relative_l2=error,
                max_abs=max_abs,
                median_us=statistics.median(times),
                min_us=min(times),
                max_us=max(times),
                inner=args.inner,
                samples_us=times,
            )
            results.append(row)
            if rank == 0:
                print(json.dumps({"bridge_result": row}), flush=True)
            if args.profile and batch != 8:
                directory = args.output.parent / f"profile-{args.tag}-{name}-b{batch}"
                with capture(directory):
                    dist.barrier(group=cpu_group)
                    for _ in range(3):
                        executable(x)
                        torch.accelerator.synchronize()
                dist.barrier(group=cpu_group)
    if rank == 0:
        args.output.write_text(json.dumps(results, indent=2))
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--coordinator", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inner", type=int, default=30)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--batches", type=int, nargs="+", default=[8, 64, 256])
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--tag", default="bridge")
    args = p.parse_args()
    os.environ.update(
        NODE_RANK=str(args.rank),
        JOB_COMPLETION_INDEX=str(args.rank),
        TORCH_TPU_TOPOLOGY="2,2,4,2",
        TPU_NUM_HOSTS="4",
        WORLD_SIZE="32",
        NNODES="4",
        MASTER_ADDR=args.coordinator.split(":")[0],
        MASTER_PORT="29441",
        TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS="false",
        TORCHINDUCTOR_AUTOGRAD_CACHE="0",
        OMP_NUM_THREADS="1",
    )
    import torch
    from torch_tpu._internal.distributed.environment import set_tpu_launch_env

    set_tpu_launch_env(nnodes=4, nproc_per_node=8)
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=8, join=True)


if __name__ == "__main__":
    main()
