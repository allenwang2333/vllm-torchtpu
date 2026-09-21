# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 synthetic external FP4-to-FP8 experiment, including conversion time.

This is a quality-unvalidated research benchmark, not a serving implementation.
The channel mode changes FP4 quantization at load time. The block64 mode keeps
block-scaled FP4 storage and applies scales/requantizes to FP8 each forward.
"""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import benchmark_adaptive_fused_moe as bench
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.streaming_fused_moe import streaming_fused_moe
from vllm_torchtpu.kernels.fused_moe.v2.host import WeightFormat
from vllm_torchtpu.layers.core.fused_moe_gmm import requant_unpack_kmajor

ROOT = Path(__file__).resolve().parents[1]
H, INTERMEDIATE, E, T, TOPK = 8192, 2048, 512, 32768, 10


def make_channel_weights(mesh):
    def local(_):
        key = jax.random.fold_in(jax.random.key(42), jax.lax.axis_index("expert"))
        keys = jax.random.split(key, 4)
        weights, scales = [], []
        for i, (n, k) in enumerate(((2 * INTERMEDIATE, H), (H, INTERMEDIATE))):
            packed = jax.random.bits(keys[2 * i], (E // 8, n, k // 2), jnp.uint8)
            scale = (
                jax.random.uniform(
                    keys[2 * i + 1], (E // 8, n, k // 16), minval=0.01, maxval=0.05
                )
                .astype(jnp.float8_e4m3fn)
                .astype(jnp.float32)
            )
            weight, scale = requant_unpack_kmajor(packed, scale, block=k)
            weights.append(weight)
            scales.append(scale[:, 0, 0, :])
        return weights[0], weights[1], scales[0], scales[1]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P("expert"),
            out_specs=(P("expert"),) * 4,
            check_vma=False,
        )
    )(jnp.zeros((8,)))


def expand_block64(weight, scale):
    experts, contraction, width = weight.shape
    # Each nonzero block produced by requant_unpack_kmajor reaches magnitude 6.
    channel = jnp.max(scale, axis=1) * (6.0 / 448.0)
    inverse = jnp.where(channel > 0, 1.0 / channel, 0.0)
    multiplier = scale * inverse[:, None, :]
    value = weight.astype(jnp.float32).reshape(experts, contraction // 64, 64, width)
    value = value * multiplier[:, :, None, :]
    value = jnp.clip(value, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    return value.reshape(experts, contraction, width), channel


def experiment_wrapper(output, mode):
    """Generate an isolated wrapper; the package's FP8 dtype contract is unchanged."""
    original = (
        ROOT / "src/vllm_torchtpu/kernels/experimental/streaming_fused_moe/layer.py"
    )
    source = original.read_text()

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Wrapper changed; review experimental insertion: {old}")
        source = source.replace(old, new)

    replace_once(
        "if actual != weight_format:",
        "if actual != weight_format and not (actual == WeightFormat.FP4 and weight_format == WeightFormat.FP8):",
    )
    if mode == "channel":
        conversion = """        with jax.named_scope("external_fp4_widen"):
            w1_l = w1_l.astype(jnp.float8_e4m3fn)
            w2_l = w2_l.astype(jnp.float8_e4m3fn)
"""
    else:
        replace_once(
            'ndim = 3 if form.scale_layout == "per_contraction_block" else 2',
            "ndim = 3  # Experimental input block scales, converted below.",
        )
        conversion = """        with jax.named_scope("external_block64_to_fp8"):
            w1_l, w1s_l = external_expand_block64(w1_l, w1s_l)
            w2_l, w2s_l = external_expand_block64(w2_l, w2s_l)
"""
    replace_once(
        "        rows_bf16 = x_l.astype(jnp.bfloat16)",
        conversion + "        rows_bf16 = x_l.astype(jnp.bfloat16)",
    )
    path = output / "external_layer.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("external_layer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.external_expand_block64 = expand_block64
    return module.streaming_fused_moe


def run_case(directory, run, inputs, common):
    directory.mkdir()

    def layer(*operands):
        return run(*operands, **common)

    started = time.perf_counter()
    executable = jax.jit(layer).lower(*inputs).compile()
    compile_seconds = time.perf_counter() - started
    for _ in range(10):
        result = executable(*inputs).block_until_ready()
    samples = []
    for _ in range(50):
        started = time.perf_counter()
        result = executable(*inputs).block_until_ready()
        samples.append((time.perf_counter() - started) * 1000)
    values = np.asarray(result, dtype=np.float32)
    if values.shape != (T, H) or not np.isfinite(values).all():
        raise AssertionError("Invalid output shape or nonfinite values")
    np.save(directory / "output.npy", values)
    (directory / "timing.json").write_text(
        json.dumps(
            {
                "compile_seconds": compile_seconds,
                "samples_ms": samples,
                "median_host_ms": float(np.median(samples)),
                "memory": str(executable.memory_analysis()),
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "geometry": dict(hidden=H, inter=INTERMEDIATE, e_total=E, topk=TOPK),
                "total_input_tokens": T,
                "external_conversion_timed": directory.name == "external",
                "synthetic_weights": True,
            },
            indent=2,
        )
        + "\n"
    )
    bench.capture_profile(executable, inputs, directory, 10)
    print(directory.name, "host median ms", np.median(samples), flush=True)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage-scale", choices=("channel", "block64"), default="channel"
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        help="Existing same-seed block64 baseline output.npy",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output)
    helper = bench.test_helpers()
    mesh = Mesh(np.asarray(jax.local_devices()), ("expert",))
    if mesh.size != 8 or jax.process_count() != 1 or jax.default_backend() != "tpu":
        raise RuntimeError("Requires one TPU process with eight local devices")
    if args.storage_scale == "channel":
        weights = make_channel_weights(mesh)
    else:
        weights = bench.make_nvfp4_weights(
            mesh, 42, hidden=H, inter=INTERMEDIATE, e_total=E
        )
    x, gating = helper._make_inputs(mesh, 43, tokens=T, hidden=H, e_total=E)

    def local_convert(a, b, c, d):
        if args.storage_scale == "block64":
            a, c = expand_block64(a, c)
            b, d = expand_block64(b, d)
        else:
            a, b = a.astype(jnp.float8_e4m3fn), b.astype(jnp.float8_e4m3fn)
        return a, b, c, d

    expanded = jax.jit(
        jax.shard_map(
            local_convert,
            mesh=mesh,
            in_specs=(P("expert"),) * 4,
            out_specs=(P("expert"),) * 4,
            check_vma=False,
        )
    )(*weights)
    jax.block_until_ready((x, gating, weights, expanded))
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=128,
        sharded_plan=True,
        weight_format=WeightFormat.FP8,
        rhs_qb=512,
        weight_tile_bytes=2**21,
        output_buffers=1,
        adaptive_rows=False,
        down_compute_cols=512,
        gate_compute_rows=128,
    )
    actual = run_case(
        args.output / "external",
        experiment_wrapper(args.output, args.storage_scale),
        (x, *weights, gating),
        common,
    )
    expected = run_case(
        args.output / "preexpanded", streaming_fused_moe, (x, *expanded, gating), common
    )
    check = {
        "storage_scale": args.storage_scale,
        "external_vs_preexpanded_bit_identical": bool(np.array_equal(actual, expected)),
    }
    if not check["external_vs_preexpanded_bit_identical"]:
        raise AssertionError(check)
    if args.compare_output:
        baseline = np.load(args.compare_output, mmap_mode="r")
        check["output_vs_block64_baseline_relative_l2"] = helper._relative_l2(
            actual, baseline
        )
        check["output_vs_block64_baseline_worst_token_relative_l2"] = (
            helper._worst_token_relative_l2(actual, baseline)
        )
    indices = np.linspace(0, T - 1, 64, dtype=np.int32)
    dense = helper._dense_reference(
        mesh, x[indices], *expanded, gating[indices], topk=TOPK
    )
    check["kernel_vs_converted_dense_relative_l2"] = helper._relative_l2(
        actual[indices], dense
    )
    check["kernel_vs_converted_dense_worst_token_relative_l2"] = (
        helper._worst_token_relative_l2(actual[indices], dense)
    )
    check["kernel_passed"] = (
        check["kernel_vs_converted_dense_relative_l2"] < helper.FP8_RELATIVE_BOUND
        and check["kernel_vs_converted_dense_worst_token_relative_l2"]
        < helper.FP8_TOKEN_BOUND
    )
    check["quality_validated"] = False
    check["scope"] = (
        "Kernel checks use already-converted weights. Real checkpoint/model-quality evaluation is still required."
    )
    (args.output / "correctness.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2), flush=True)
    if not check["kernel_passed"]:
        raise AssertionError(check)


if __name__ == "__main__":
    main()
