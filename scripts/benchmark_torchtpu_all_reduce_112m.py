#!/usr/bin/env python3
"""Self-contained TorchTPU benchmark for a 112-MiB TP32 all-reduce.
The default local tensor is BF16 ``[8192, 7168]``:
    8192 * 7168 * 2 = 117,440,512 bytes = 112 MiB
The timed interval contains a compiled clone, all-reduce, trivial consumer,
and an explicit TPU synchronization. Use the optional XPlane trace to isolate
the physical collective kernel from those API-boundary costs.
Launch one torchrun agent per host. For a four-host, eight-process-per-host
TP32 slice, the command on host ``NODE_RANK`` is:
    export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
    export TPU_HOST_BOUNDS=1,1,4
    export TORCH_TPU_TOPOLOGY=2,2,4,2
    export TORCH_TPU_SLICEBUILDER_ADDRESSES=<comma-separated worker addresses>
    torchrun \
      --nnodes=4 --nproc-per-node=8 --node-rank=${NODE_RANK} \
      --master-addr=<rank-0-host> --master-port=30201 \
      scripts/benchmark_torchtpu_all_reduce_112m.py \
      --warmups=10 --iterations=100
Every rank emits one ``TORCHTPU_ALL_REDUCE_RESULT`` JSON record. Relevant
compiler experiments can be supplied normally, for example:
    LIBTPU_INIT_ARGS=--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true
Use ``--topology-aware`` to create the full-world group from
``torch.tpu.topology_aware_mesh`` while preserving its rank order. Use
``--chunks=2`` to reproduce the experimental tuple of two 56-MiB operands.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--width", type=int, default=7168)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--expected-world-size", type=int, default=32)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--split-dim", type=int, choices=(0, 1), default=1)
    parser.add_argument("--topology-aware", action="store_true")
    parser.add_argument(
        "--topology-mesh-shape",
        help="comma-separated logical mesh shape; default is 1,1,WORLD_SIZE",
    )
    parser.add_argument("--profile-dir", type=Path)
    return parser.parse_args()


def synchronize(tensor) -> None:
    from torch_tpu._internal import sync

    sync.synchronize(tensors=[tensor], wait=True)


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def main() -> None:
    args = parse_args()

    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist
    from torch_tpu._internal import compile as tpu_compile

    if args.warmups < 0 or args.iterations <= 0:
        raise ValueError("warmups must be nonnegative and iterations positive")
    if args.chunks <= 0:
        raise ValueError("chunks must be positive")

    dist.init_process_group(backend="tpu_dist")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if (args.expected_world_size > 0
                and world_size != args.expected_world_size):
            raise RuntimeError(
                f"expected world size {args.expected_world_size}, got "
                f"{world_size}; pass --expected-world-size=0 to allow any")

        group = None
        group_ranks = list(range(world_size))
        mesh_shape = None
        if args.topology_aware:
            mesh_shape = (tuple(
                int(dim) for dim in args.topology_mesh_shape.split(","))
                          if args.topology_mesh_shape else (1, 1, world_size))
            if not mesh_shape or any(dim <= 0 for dim in mesh_shape):
                raise ValueError(f"invalid topology mesh shape: {mesh_shape}")
            product = 1
            for dim in mesh_shape:
                product *= dim
            if product != world_size:
                raise ValueError(
                    f"topology mesh {mesh_shape} has {product} devices, "
                    f"but world size is {world_size}")
            mesh = torch.tpu.topology_aware_mesh(mesh_shape)
            group_ranks = [int(value) for value in mesh.reshape(-1)]
            if sorted(group_ranks) != list(range(world_size)):
                raise RuntimeError(
                    f"topology-aware order is not a rank permutation: "
                    f"{group_ranks}")
            group = dist.new_group(
                group_ranks,
                backend="tpu_dist",
                sort_ranks=False,
            )

        if args.tokens % args.chunks and args.split_dim == 0:
            raise ValueError("tokens must be divisible by chunks")
        if args.width % args.chunks and args.split_dim == 1:
            raise ValueError("width must be divisible by chunks")

        # Initialize PJRT and verify that the selected process group works
        # before paying the compilation cost of the large operation.
        ready = torch.ones((1, ), dtype=torch.bfloat16, device="tpu")
        dist.all_reduce(ready, group=group)
        synchronize(ready)

        rank_value = (rank + 1) / world_size
        source = torch.full(
            (args.tokens, args.width),
            rank_value,
            dtype=torch.bfloat16,
            device="tpu",
        )

        def reduce_then_consume(value):
            reduced = []
            for part in value.chunk(args.chunks, dim=args.split_dim):
                # clone makes each chunk a contiguous collective operand and
                # leaves the reusable benchmark input unchanged.
                result = part.clone()
                dist.all_reduce(result, group=group)
                reduced.append(result)
            return torch.cat(reduced, dim=args.split_dim) + 1

        compiled = torch.compile(
            reduce_then_consume,
            fullgraph=True,
            dynamic=False,
            backend=tpu_compile.TpuBackend(),
        )

        output = compiled(source)
        synchronize(output)
        observed = float(output[0, 0].cpu())
        expected = (world_size + 1) / 2 + 1
        if observed != expected:
            raise RuntimeError(
                f"all-reduce validation failed: observed {observed}, "
                f"expected {expected}")

        for _ in range(args.warmups):
            output = compiled(source)
            synchronize(output)

        latencies_ms = []
        for _ in range(args.iterations):
            start_ns = time.perf_counter_ns()
            output = compiled(source)
            synchronize(output)
            latencies_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

        message_bytes = args.tokens * args.width * 2
        median_ms = statistics.median(latencies_ms)
        algorithm_gbps = message_bytes / (median_ms / 1e3) / 1e9
        bus_gbps = algorithm_gbps * 2 * (world_size - 1) / world_size

        if args.profile_dir is not None:
            # Every peer must execute the profiled collective. Rank 0 alone
            # records a trace; the final tiny collective keeps peers alive
            # while rank 0 serializes it.
            if rank == 0:
                import torch.profiler
                from torch_tpu._internal.profiler import TpuProfilerConfig

                rank_dir = args.profile_dir / "rank_0"
                rank_dir.mkdir(parents=True, exist_ok=True)
                handler = torch.profiler.tensorboard_trace_handler(
                    str(rank_dir), use_gzip=True)
                config = TpuProfilerConfig(
                    run_dir=str(rank_dir),
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
                        on_trace_ready=handler,
                        experimental_config=config,
                ):
                    output = compiled(source)
                    synchronize(output)
            else:
                output = compiled(source)
                synchronize(output)

            done = torch.ones((1, ), dtype=torch.bfloat16, device="tpu")
            dist.all_reduce(done, group=group)
            synchronize(done)

        result = {
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
            "world_size": world_size,
            "shape": [args.tokens, args.width],
            "dtype": "bfloat16",
            "message_bytes": message_bytes,
            "message_mib": message_bytes / (1 << 20),
            "chunks": args.chunks,
            "chunk_mib": message_bytes / args.chunks / (1 << 20),
            "split_dim": args.split_dim,
            "topology_aware": args.topology_aware,
            "topology_mesh_shape": mesh_shape,
            "group_ranks": group_ranks,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "median_ms": median_ms,
            "mean_ms": statistics.fmean(latencies_ms),
            "min_ms": min(latencies_ms),
            "p90_ms": percentile(latencies_ms, 0.90),
            "p99_ms": percentile(latencies_ms, 0.99),
            "max_ms": max(latencies_ms),
            "algorithm_gbps": algorithm_gbps,
            "ring_equivalent_bus_gbps": bus_gbps,
            "observed": observed,
            "expected": expected,
        }
        print("TORCHTPU_ALL_REDUCE_RESULT " + json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
