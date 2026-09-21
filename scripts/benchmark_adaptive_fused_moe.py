# SPDX-License-Identifier: Apache-2.0
"""Single-host SPMD benchmark of a complete routed MoE layer.

Uses deterministic synthetic weights/inputs with the model's geometry and
the kernel's FP8 or NVFP4 runtime layout. This is not a checkpoint or
end-to-end serving benchmark. Run with PYTHONPATH=src and JAX_PLATFORMS=tpu.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    AXIS,
    WeightFormat,
    adaptive_fused_moe,
    select_weight_buffers,
)

ROOT = Path(__file__).resolve().parents[1]


def test_helpers():
    path = ROOT / "tests/kernels/experimental/adaptive_fused_moe/test_kernel.py"
    spec = importlib.util.spec_from_file_location("moe_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_nvfp4_weights(mesh, seed, *, hidden, inter, e_total):
    from vllm_torchtpu.layers.core.fused_moe_gmm import requant_unpack_kmajor

    (axis,) = mesh.axis_names
    ep = mesh.shape[axis]

    def local(_):
        key = jax.random.fold_in(jax.random.key(seed), jax.lax.axis_index(axis))
        keys = jax.random.split(key, 4)
        weights, scales = [], []
        for i, (n, k) in enumerate(((2 * inter, hidden), (hidden, inter))):
            packed = jax.random.bits(keys[2 * i], (e_total // ep, n, k // 2), jnp.uint8)
            scale = (
                jax.random.uniform(
                    keys[2 * i + 1],
                    (e_total // ep, n, k // 16),
                    minval=0.01,
                    maxval=0.05,
                )
                .astype(jnp.float8_e4m3fn)
                .astype(jnp.float32)
            )
            weight, scale = requant_unpack_kmajor(packed, scale, block=64)
            weights.append(weight)
            scales.append(jnp.squeeze(scale, axis=-2))
        return weights[0], weights[1], scales[0], scales[1]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis),),
            out_specs=(P(axis),) * 4,
            check_vma=False,
        )
    )(jnp.zeros((ep,), jnp.float32))


@jax.jit
def decode_nvfp4(weight, scale):
    e, k, n = weight.shape
    return (
        weight.astype(jnp.float32).reshape(e, k // 64, 64, n) * scale[:, :, None, :]
    ).reshape(e, k, n)


def capture_profile(executable, inputs, output_dir, iterations):
    import torch
    from torch_tpu._internal import profiler

    from vllm_torchtpu.tools import xplane_pb2

    # Register the native TPU profiler factory outside the capture. The
    # measured layer remains the already compiled JAX executable.
    torch.ones((1,), device="tpu").cpu()
    for _ in range(3):
        executable(*inputs).block_until_ready()
    profile_dir = output_dir / "profile"
    print(f"Capturing {iterations} warmed-up MoE steps", flush=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.TPU],
        on_trace_ready=profiler.xprof_trace_handler(profile_dir),
    ):
        for step in range(iterations):
            with jax.profiler.StepTraceAnnotation("moe_benchmark", step_num=step):
                executable(*inputs).block_until_ready()
    traces = list(profile_dir.rglob("*.xplane.pb"))
    if not traces:
        raise RuntimeError("Profiler did not produce an XPlane file")
    captured = []
    for path in traces:
        space = xplane_pb2.XSpace()
        space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            if not plane.name.removeprefix("/device:TPU:").isdigit():
                continue
            modules = [
                event
                for line in plane.lines
                if line.name == "XLA Modules"
                for event in line.events
                if "jit_layer" in plane.event_metadata[event.metadata_id].name
            ]
            captured.append({"plane": plane.name, "moe_calls": len(modules)})
    (output_dir / "profile_validation.json").write_text(json.dumps(captured, indent=2))
    if len(captured) != 8 or any(row["moe_calls"] != iterations for row in captured):
        raise RuntimeError(f"Incomplete TPU trace: {captured}")
    print(f"Validated profile: {captured}", flush=True)
    try:
        hlo = executable.as_text()
    except jax.errors.JaxRuntimeError as error:
        # Optional text export may fail on TPU async instructions even when
        # the executable and its XPlane capture are valid.
        (output_dir / "hlo_export_error.txt").write_text(str(error))
        print(f"Optional HLO text export unavailable: {error}", flush=True)
    else:
        (output_dir / "compiled_hlo.txt").write_text(hlo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--reference-tokens", type=int, default=64)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--weight-format", choices=("fp8", "nvfp4"), default="fp8")
    parser.add_argument(
        "--implementation",
        choices=("adaptive", "v2", "tiled", "streaming"),
        default="adaptive",
    )
    parser.add_argument("--gate-compute-rows", type=int, default=256)
    parser.add_argument("--down-compute-cols", type=int, default=0)
    parser.add_argument("--adaptive-rows", action="store_true")
    parser.add_argument("--output-buffers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--weight-tile-mib", type=float, default=2)
    parser.add_argument("--weight-slices", type=int, default=2)
    parser.add_argument(
        "--capacity", type=int, choices=(128, 256, 512, 1024), default=128
    )
    parser.add_argument(
        "--routing", choices=("random", "concentrated"), default="random"
    )
    parser.add_argument("--vmem-budget-mib", type=float)
    args = parser.parse_args()
    if min(args.tokens_per_rank, args.iterations, args.reference_tokens) < 1:
        parser.error("token and iteration counts must be positive")
    if args.warmup < 1:
        parser.error("at least one warmup is required")
    if args.profile_iterations < 0:
        parser.error("profile iterations must be nonnegative")
    if args.weight_format == "nvfp4" and args.implementation not in (
        "adaptive",
        "streaming",
    ):
        parser.error("NVFP4 benchmark requires the adaptive implementation")
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output / "source"
    source_dir.mkdir()
    import shutil

    shutil.copy2(__file__, source_dir)
    if args.implementation in ("tiled", "streaming"):
        package = f"{args.implementation}_fused_moe"
        shutil.copytree(
            ROOT / "src/vllm_torchtpu/kernels/experimental" / package,
            source_dir / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    config_path = args.model / "config.json" if args.model.is_dir() else args.model
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config = config.get("text_config", config)
    geometry = dict(
        hidden=config["hidden_size"],
        inter=config["moe_intermediate_size"],
        e_total=config["num_experts"],
        topk=config["num_experts_per_tok"],
    )
    nvfp4 = args.weight_format == "nvfp4"
    qwen35 = dict(hidden=4096, inter=1024, e_total=512, topk=10)
    qwen38 = dict(hidden=8192, inter=2048, e_total=512, topk=10)
    supported = (qwen38,) if nvfp4 else (qwen35, qwen38)
    if geometry not in supported:
        raise ValueError(f"Expected one of {supported}, got {geometry}")
    weight_format = WeightFormat.FP4 if nvfp4 else WeightFormat.FP8
    devices = jax.local_devices()
    if jax.default_backend() != "tpu" or jax.process_count() != 1 or len(devices) != 8:
        raise RuntimeError("Requires one process with exactly eight local TPU devices")
    mesh = Mesh(np.asarray(devices), axis_names=(AXIS,))
    total = 8 * args.tokens_per_rank
    capacity = args.capacity
    # Streaming/tiled kernels allocate weight slices, so a whole-expert
    # adaptive buffer plan does not describe their VMEM requirements.
    plan = None
    if args.implementation in ("adaptive", "v2"):
        plan = select_weight_buffers(
            64,
            128,
            geometry["hidden"],
            geometry["inter"],
            weight_format=weight_format,
            rhs_qb=64,
        )
        if nvfp4 and (plan.w1_nbuf, plan.w2_nbuf) != (2, 1):
            raise AssertionError(f"Expected NVFP4 2+1 buffers, got {plan}")
    helper = test_helpers()
    run = adaptive_fused_moe
    extra = {}
    if args.implementation == "v2":
        from vllm_torchtpu.kernels.fused_moe.v2 import fused_ep_moe_v2

        run = fused_ep_moe_v2
        extra["sharded_plan"] = True
    if args.implementation == "tiled":
        from vllm_torchtpu.kernels.experimental.tiled_fused_moe import tiled_fused_moe

        run = tiled_fused_moe
        extra.update(
            sharded_plan=True,
            weight_slices=args.weight_slices,
            vmem_budget_bytes=None
            if args.vmem_budget_mib is None
            else int(args.vmem_budget_mib * 2**20),
        )
    if args.implementation == "streaming":
        from vllm_torchtpu.kernels.experimental.streaming_fused_moe import (
            streaming_fused_moe,
        )

        run = streaming_fused_moe
        extra.update(
            sharded_plan=True,
            weight_format=weight_format,
            rhs_qb=64 if nvfp4 else 512,
            weight_tile_bytes=int(args.weight_tile_mib * 2**20),
            output_buffers=args.output_buffers,
            adaptive_rows=args.adaptive_rows,
            down_compute_cols=args.down_compute_cols,
            gate_compute_rows=args.gate_compute_rows,
            vmem_budget_bytes=None
            if args.vmem_budget_mib is None
            else int(args.vmem_budget_mib * 2**20),
        )
    metadata = {
        "weight_tile_mib": args.weight_tile_mib,
        "output_buffers": args.output_buffers,
        "adaptive_rows": args.adaptive_rows,
        "down_compute_cols": args.down_compute_cols,
        "gate_compute_rows": args.gate_compute_rows,
        "vmem_budget_mib": args.vmem_budget_mib,
        "weight_slices": args.weight_slices if args.implementation == "tiled" else 1,
        "implementation": args.implementation,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_diff": subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src"], cwd=ROOT, text=True
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_path": str(args.model),
        "geometry": geometry,
        "ep": 8,
        "tokens_per_rank": args.tokens_per_rank,
        "total_input_tokens": total,
        "total_routed_rows": total * geometry["topk"],
        "capacity": capacity,
        "renormalize": True,
        "activation": "silu",
        "synthetic_data": True,
        "weights": "NVFP4 block16 -> FP4 E2M1 block64 runtime layout"
        if nvfp4
        else "FP8 e4m3 per-channel runtime layout",
        "weight_format": args.weight_format,
        "weight_conversion_timed": False,
        "input_dtype": "bfloat16",
        "routing": args.routing,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "weight_buffer_counts": {"w1": 3, "w2": 3}
        if args.implementation in ("v2", "tiled")
        else (
            {"gate_up_shared": 2, "down": 2}
            if args.implementation == "streaming"
            else {"w1": plan.w1_nbuf, "w2": plan.w2_nbuf}
        ),
        "adaptive_buffer_plan": plan._asdict() if plan is not None else None,
        "devices": [str(d) for d in devices],
        "device_kind": devices[0].device_kind,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("jax", "jaxlib", "libtpu", "numpy")
        },
        "environment": {
            k: os.environ.get(k)
            for k in (
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "LIBTPU_INIT_ARGS",
                "TPU_SKIP_MDS_QUERY",
                "TPU_VISIBLE_CHIPS",
            )
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    print("Generating sharded weights and inputs", flush=True)
    make_weights = make_nvfp4_weights if nvfp4 else helper._make_weights
    weight_extra = {} if nvfp4 else {"weight_format": weight_format}
    w1, w2, s1, s2 = make_weights(
        mesh,
        args.seed,
        hidden=geometry["hidden"],
        inter=geometry["inter"],
        e_total=geometry["e_total"],
        **weight_extra,
    )
    x, gating = helper._make_inputs(
        mesh,
        args.seed + 1,
        tokens=total,
        hidden=geometry["hidden"],
        e_total=geometry["e_total"],
    )
    if args.routing == "concentrated":
        # All tokens select the same top-k experts, spread across ranks.
        chosen = np.asarray([(i % 8) * 64 + i // 8 for i in range(geometry["topk"])])

        def concentrated(logits):
            logits = jnp.full_like(logits, -1000)
            return logits.at[:, chosen].set(
                jnp.arange(geometry["topk"], dtype=logits.dtype)
            )

        gating = jax.jit(concentrated, out_shardings=gating.sharding)(gating)
    inputs = (x, w1, w2, s1, s2, gating)
    jax.block_until_ready(inputs)

    def layer(*operands):
        return run(
            *operands,
            topk=geometry["topk"],
            renormalize=True,
            mesh=mesh,
            capacity=capacity,
            **extra,
        )

    print("Compiling complete routed MoE layer", flush=True)
    start = time.perf_counter()
    executable = jax.jit(layer).lower(*inputs).compile()
    compile_s = time.perf_counter() - start
    print(f"Compiled in {compile_s:.2f}s; warming up", flush=True)
    for _ in range(args.warmup):
        output = executable(*inputs).block_until_ready()
    if args.compare_output is not None:
        actual = np.asarray(output, dtype=np.float32)
        expected_output = np.load(args.compare_output, mmap_mode="r")
        comparison = dict(
            reference=str(args.compare_output),
            bit_identical=bool(np.array_equal(actual, expected_output)),
            relative_l2=float(
                np.linalg.norm(actual - expected_output)
                / max(np.linalg.norm(expected_output), 1e-12)
            ),
            max_abs=float(np.max(np.abs(actual - expected_output))),
        )
        (args.output / "baseline_comparison.json").write_text(
            json.dumps(comparison, indent=2)
        )
        print(json.dumps(comparison), flush=True)
        if not comparison["bit_identical"]:
            np.save(args.output / "mismatched_output.npy", actual)
            raise AssertionError(comparison)
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        output = executable(*inputs).block_until_ready()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    values = np.asarray(output, dtype=np.float32)
    if values.shape != (total, geometry["hidden"]) or not np.isfinite(values).all():
        raise AssertionError("Invalid output shape or non-finite values")
    np.save(args.output / "output.npy", values)
    result = {
        "compile_seconds": compile_s,
        "latency_ms": {
            "median": float(np.median(samples)),
            "mean": float(np.mean(samples)),
            "min": float(np.min(samples)),
            "p05": float(np.percentile(samples, 5)),
            "p95": float(np.percentile(samples, 95)),
            "max": float(np.max(samples)),
        },
        "samples_ms": samples,
        "layer_input_tokens_per_second": total / (float(np.median(samples)) / 1000),
        "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "executable_memory_analysis": str(executable.memory_analysis()),
        "note": "Synchronized host wall time; compile, warmup and reference excluded. "
        "Memory analysis is not a compiler VMEM peak measurement.",
    }
    (args.output / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.profile_iterations:
        capture_profile(executable, inputs, args.output, args.profile_iterations)
    print("Checking sampled rows from all ranks against dense reference", flush=True)
    count = min(args.reference_tokens, total)
    indices = np.linspace(0, total - 1, count, dtype=np.int32)
    reference_weights = (
        (decode_nvfp4(w1, s1), decode_nvfp4(w2, s2), None, None)
        if nvfp4
        else (w1, w2, s1, s2)
    )
    want = helper._dense_reference(
        mesh, x[indices], *reference_weights, gating[indices], topk=geometry["topk"]
    )
    actual = values[indices]
    error = helper._relative_l2(actual, want)
    worst = helper._worst_token_relative_l2(actual, want)
    check = {
        "sampled_rows": count,
        "indices": indices.tolist(),
        "relative_l2": error,
        "worst_token_relative_l2": worst,
        "relative_bound": helper.FP8_RELATIVE_BOUND,
        "token_bound": helper.FP8_TOKEN_BOUND,
        "passed": error < helper.FP8_RELATIVE_BOUND and worst < helper.FP8_TOKEN_BOUND,
    }
    (args.output / "correctness.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2), flush=True)
    assert check["passed"], check


if __name__ == "__main__":
    main()
