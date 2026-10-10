# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed routed-MoE correctness and performance without a serving stack.

Reuses production routing, FP4 GMM and combine, including the final collective.
Gate projection, shared expert and attention are outside this benchmark.
"""

import argparse
import dataclasses
import functools
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

EP = "expert"
TP = "tensor"
P = PartitionSpec


def emit(value: dict) -> None:
    if jax.process_index() == 0:
        print(json.dumps(value), flush=True)


def make_weights(
    mesh: Mesh, experts: int, hidden: int, intermediate: int, quant_block: int
) -> tuple[tuple[jax.Array, ...], tuple[PartitionSpec, ...]]:
    from jax.experimental.pallas import tpu as pltpu

    lanes = pltpu.get_tpu_info().num_lanes
    assert intermediate % (mesh.shape[TP] * lanes) == 0
    shapes = [
        (experts, hidden, 2, intermediate),
        (experts, intermediate, hidden),
        (experts, hidden // quant_block, 1, 2, intermediate),
        (experts, intermediate // quant_block, 1, hidden),
    ]
    specs = [P(EP, None, None, TP), P(EP, TP), P(EP, None, None, None, TP), P(EP, TP)]
    arrays = []
    for i, (shape, spec) in enumerate(zip(shapes, specs)):

        @functools.partial(jax.jit, out_shardings=NamedSharding(mesh, spec))
        def generate(key, *, shape=shape, i=i):
            bits = jax.random.bits(key, shape, dtype=jnp.uint32)
            if i < 2:
                return ((bits % 13).astype(jnp.float32) - 6).astype(jnp.float4_e2m1fn)
            fanin = hidden if i == 2 else intermediate
            return (0.8 + (bits % 101).astype(jnp.float32) * 0.004) / (4 * fanin**0.5)

        array = generate(jax.random.key(100 + i))
        array.block_until_ready()
        arrays.append(array)
    packed_specs = (P(EP, None, TP), P(EP, TP), P(EP, None, None, TP), P(EP, TP))

    def pack(w1, w2, s1, s2):
        return (
            w1.reshape(w1.shape[0], w1.shape[1], -1),
            w2,
            s1.reshape(s1.shape[0], s1.shape[1], 1, -1),
            s2,
        )

    prepare = jax.jit(
        jax.shard_map(
            pack,
            mesh=mesh,
            in_specs=tuple(specs),
            out_specs=packed_specs,
            check_vma=False,
        )
    )
    prepared = prepare(*arrays)
    jax.block_until_ready(prepared)
    return prepared, packed_specs


def build_forward(
    mesh: Mesh,
    specs: tuple,
    topk: int,
    threshold: int,
    reference: bool = False,
    collective: str = "flat",
) -> Callable[..., jax.Array]:
    from vllm_torchtpu.kernels.router_topk import router_topk
    from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

    host_groups = []
    if collective != "flat":
        assert mesh.shape[TP] == 1
        by_host = {}
        for index, device in enumerate(mesh.devices.flat):
            by_host.setdefault(device.process_index, []).append(index)
        host_groups = list(by_host.values())
        assert len(host_groups) == 4 and all(len(g) == 8 for g in host_groups)
    cross_groups = list(map(list, zip(*host_groups)))

    def local(x, logits, w1, w2, s1, s2):
        routing_logits = logits + x[:, : logits.shape[1]] * 0.01
        if reference:
            scores, ids = jax.lax.top_k(routing_logits, topk)
            weights = jax.nn.softmax(scores.astype(jnp.float32)).astype(jnp.bfloat16)
        else:
            scores, ids = router_topk(jax.nn.softmax(routing_logits), topk)
            weights = (scores / scores.sum(axis=-1, keepdims=True)).astype(jnp.bfloat16)
        start = jax.lax.axis_index(EP) * w1.shape[0]
        if reference:
            logical_w1 = (
                w1.astype(jnp.float32).reshape(
                    w1.shape[0], s1.shape[1], w1.shape[1] // s1.shape[1], *w1.shape[2:]
                )
                * s1
            ).reshape(w1.shape)
            gate, up = jnp.split(logical_w1, 2, axis=-1)
            down = (
                w2.astype(jnp.float32).reshape(
                    w2.shape[0], s2.shape[1], w2.shape[1] // s2.shape[1], w2.shape[-1]
                )
                * s2
            ).reshape(w2.shape)
            g = jnp.einsum("bh,ehi->bei", x.astype(jnp.float32), gate)
            u = jnp.einsum("bh,ehi->bei", x.astype(jnp.float32), up)
            values = jnp.einsum("bei,eih->beh", jax.nn.silu(g) * u, down)
            mask = jax.nn.one_hot(ids - start, w1.shape[0], dtype=jnp.float32)
            mixture = (mask * weights.astype(jnp.float32)[..., None]).sum(axis=1)
            y = (values * mixture[..., None]).sum(axis=1)
        else:
            y = fused_moe_func(
                x,
                w1.reshape(w1.shape[0], w1.shape[1], -1),
                w2,
                s1.reshape(s1.shape[0], s1.shape[1], 1, -1),
                s2,
                None,
                None,
                weights,
                ids,
                start,
                topk=topk,
                activation="silu",
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=threshold,
            )
        if collective == "two-stage":
            y = jax.lax.psum(y, EP, axis_index_groups=host_groups)
            return jax.lax.psum(y, EP, axis_index_groups=cross_groups)
        if collective == "scatter-reduce-gather":
            y = jax.lax.psum_scatter(
                y, EP, scatter_dimension=1, axis_index_groups=host_groups, tiled=True
            )
            y = jax.lax.psum(y, EP, axis_index_groups=cross_groups)
            return jax.lax.all_gather(
                y, EP, axis=1, axis_index_groups=host_groups, tiled=True
            )
        return jax.lax.psum(y, (EP, TP))

    mapped = jax.shard_map(
        local, mesh=mesh, in_specs=(P(), P(), *specs), out_specs=P(), check_vma=False
    )
    return mapped


def host_array(array: jax.Array) -> np.ndarray:
    return np.asarray(array.addressable_shards[0].data, dtype=np.float32)


def run_case(
    args: argparse.Namespace, devices: list[jax.Device], tp: int, small: bool
) -> list[dict]:
    experts, hidden, intermediate, topk = (
        (32, 256, 1024, 4) if small else (512, 8192, 2048, 10)
    )
    mesh = Mesh(np.array(devices).reshape(-1, tp), (EP, TP))
    emit({"phase": "weights", "ep": mesh.shape[EP], "tp": tp, "small": small})
    weights, specs = make_weights(mesh, experts, hidden, intermediate, args.quant_block)
    forward = build_forward(
        mesh, specs, topk, args.threshold, collective=args.collective
    )
    compiled_forward = jax.jit(forward)
    rows = []
    for batch in [8] if small else args.batches:
        for skew in [0.0] if small else args.skews:
            replicated = NamedSharding(mesh, P())

            @functools.partial(jax.jit, out_shardings=(replicated, replicated))
            def inputs(key, *, batch=batch, skew=skew):
                x = jax.random.normal(key, (batch, hidden), dtype=jnp.bfloat16)
                logits = jax.random.normal(jax.random.fold_in(key, 1), (batch, experts))
                logits += (jnp.arange(experts) < (args.hot_experts or experts // 8))[
                    None, :
                ] * skew
                return x, logits

            x, logits = inputs(jax.random.key(args.seed + batch))
            t = time.monotonic()
            output = compiled_forward(x, logits, *weights)
            output.block_until_ready()
            emit(
                {
                    "phase": "compiled",
                    "tp": tp,
                    "batch": batch,
                    "skew": skew,
                    "seconds": time.monotonic() - t,
                    "small": small,
                }
            )
            got = host_array(output)
            assert np.isfinite(got).all(), "Non-finite MoE output"
            key = f"{small}-{batch}-{float(skew):g}"
            ref_tag = args.reference_tag or args.tag
            ref_path = args.output.parent / (
                f"reference-{ref_tag}-{key}-rank{args.rank}.npy"
            )
            if small or (args.dense_reference and tp == args.tps[0]):
                reference = jax.jit(
                    build_forward(mesh, specs, topk, args.threshold, True)
                )
                want = host_array(reference(x, logits, *weights))
                if not small and not args.reference_tag:
                    np.save(ref_path, got)
            elif tp == args.tps[0] and not args.reference_tag:
                np.save(ref_path, got)
                want = got
            else:
                want = np.load(ref_path)
            error = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12))
            max_abs = float(np.max(np.abs(got - want)))
            assert error < 0.03 and max_abs < 0.05, (error, max_abs)
            row = {
                "ep": mesh.shape[EP],
                "tp": tp,
                "batch": batch,
                "skew": skew,
                "small": small,
                "relative_l2": error,
                "max_abs": max_abs,
                "rms_output": float(np.sqrt(np.mean(got**2))),
                "threshold": args.threshold,
                "quant_block": args.quant_block,
                "tile_m": args.tile_m,
                "bucket_base": args.bucket_base,
                "single_sort_max": args.single_sort_max,
                "seed": args.seed,
                "hot_experts": args.hot_experts or experts // 8,
                "permute_threshold": args.permute_threshold,
                "collective": args.collective,
            }
            if not small:

                def chain(x, logits, *weights):
                    def step(x, _):
                        y = forward(x, logits, *weights)
                        return x + y * jnp.bfloat16(0.01), None

                    return jax.lax.scan(step, x, None, length=args.inner)[0]

                lowered = jax.jit(chain).lower(x, logits, *weights)
                executable = lowered.compile()
                emit(
                    {"phase": "chain_compiled", "tp": tp, "batch": batch, "skew": skew}
                )
                hlo = str(lowered.compiler_ir(dialect="stablehlo"))
                (
                    args.output.parent
                    / f"stablehlo-{args.tag}-tp{tp}-b{batch}-rank{args.rank}.txt"
                ).write_text(hlo)
                for _ in range(3):
                    executable(x, logits, *weights).block_until_ready()
                times = []
                for repeat in range(args.repeats):
                    multihost_utils.sync_global_devices(f"timing-{key}-{tp}-{repeat}")
                    t = time.perf_counter()
                    executable(x, logits, *weights).block_until_ready()
                    elapsed = (time.perf_counter() - t) * 1e6 / args.inner
                    all_times = multihost_utils.process_allgather(np.array(elapsed))
                    times.append(float(np.max(all_times)))
                row.update(
                    median_us=statistics.median(times),
                    min_us=min(times),
                    max_us=max(times),
                    samples_us=times,
                )
                emit({"timing": row})
                if args.profile:
                    multihost_utils.sync_global_devices(f"profile-{key}-{tp}")
                    import torch
                    from torch_tpu._internal.profiler import TpuProfilerConfig

                    directory = args.output.parent / (
                        f"profile-{args.tag}-tp{tp}-b{batch}-s{skew}"
                    )
                    config = TpuProfilerConfig(
                        run_dir=str(directory),
                        host_tracer_level=0,
                        device_tracer_level=1,
                        python_tracer_level=0,
                    )
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.PrivateUse1],
                        experimental_config=config,
                    ):
                        for _ in range(3):
                            executable(x, logits, *weights).block_until_ready()
            routing_logits = host_array(logits) + host_array(x)[:, :experts] * 0.01
            ids = np.argsort(routing_logits, axis=-1)[:, -topk:]
            loads = np.bincount(
                ids.ravel() // (experts // mesh.shape[EP]), minlength=mesh.shape[EP]
            )
            row.update(
                max_routes_per_ep=int(loads.max()),
                min_routes_per_ep=int(loads.min()),
                mean_routes_per_ep=float(loads.mean()),
            )
            expert_counts = np.bincount(ids.ravel(), minlength=experts).reshape(
                mesh.shape[EP], -1
            )
            active_experts = (expert_counts > 0).sum(axis=1)
            row.update(
                max_active_experts=int(active_experts.max()),
                min_active_experts=int(active_experts.min()),
                total_active_experts=int(active_experts.sum()),
            )
            rows.append(row)
            emit({"result": row})
    del weights
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--tps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--skews", nargs="+", type=float, default=[0, 2])
    parser.add_argument("--threshold", type=int, default=32768)
    parser.add_argument("--quant-block", type=int, default=256)
    parser.add_argument("--inner", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--reference-tag")
    parser.add_argument("--dense-reference", action="store_true")
    parser.add_argument("--tile-m", type=int)
    parser.add_argument("--bucket-base", type=int)
    parser.add_argument("--single-sort-max", type=int)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--hot-experts", type=int)
    parser.add_argument("--permute-threshold", type=int)
    parser.add_argument(
        "--collective",
        choices=["flat", "two-stage", "scatter-reduce-gather"],
        default="flat",
    )
    args = parser.parse_args()
    jax.distributed.initialize(
        coordinator_address=args.coordinator, num_processes=4, process_id=args.rank
    )
    devices = sorted(jax.devices(), key=lambda d: (tuple(d.coords), d.core_on_chip))
    emit(
        {
            "phase": "devices",
            "devices": [str(d) for d in devices],
            "local": [str(d) for d in jax.local_devices()],
        }
    )
    assert len(devices) == 32, f"Expected the service topology32, got {len(devices)}"
    if args.permute_threshold is not None:
        from vllm_torchtpu.layers.core import fused_moe_gmm

        original_prepare = fused_moe_gmm.prepare_routed_gmm_inputs

        def prepare_with_threshold(*positional, **keywords):
            keywords["onehot_moe_permute_threshold"] = args.permute_threshold
            return original_prepare(*positional, **keywords)

        fused_moe_gmm.prepare_routed_gmm_inputs = prepare_with_threshold
    if args.single_sort_max:
        from vllm_torchtpu.layers.core import fused_moe_gmm

        fused_moe_gmm._SINGLE_SORT_MAX_ROWS = args.single_sort_max
    if args.tile_m or args.bucket_base:
        from vllm_torchtpu.kernels.megablox.gmm_v2 import calculate_tiling, gmm_v2
        from vllm_torchtpu.layers.core import fused_moe_gmm

        def tile_policy(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act):
            original = calculate_tiling(
                dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act
            )
            rows = min(args.tile_m or original.tile_m, original.tile_m)
            base = args.bucket_base or rows
            assert rows % base == 0
            return dataclasses.replace(original, tile_m=rows, bucket_base=base)

        fused_moe_gmm.gmm_v2 = functools.partial(gmm_v2, tile_info=tile_policy)
    if args.profile:
        import torch
        from torch_tpu._internal.profiler import TpuProfilerConfig

        # Prime backend registration before collecting hardware evidence.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.PrivateUse1],
            experimental_config=TpuProfilerConfig(
                run_dir=str(args.output.parent / "profile-prime"),
                host_tracer_level=0,
                device_tracer_level=1,
                python_tracer_level=0,
            ),
        ):
            pass
    results = []
    if not args.probe:
        for small in [True] if args.small_only else [True, False]:
            for tp in args.tps:
                results += run_case(args, devices, tp, small)
                args.output.write_text(json.dumps(results, indent=2))
    multihost_utils.sync_global_devices("done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
