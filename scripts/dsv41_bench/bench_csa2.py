# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure the current CSA2 snapshot on eight independent TPU7x ranks.

Inputs and the independent numerical oracle come from test_csa2.py. Timed
calls consume already gathered, requantized KV. Cache gather, conversion from
the serving cache format, and projections are outside this benchmark.
MBU uses source-derived DMA payload bytes, not hardware traffic counters.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def load_tests(root: pathlib.Path):
    """Load the standalone tests without importing the serving framework."""
    path = root / "tests/kernels/deepseek_v41/test_csa2.py"
    spec = importlib.util.spec_from_file_location("csa2_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dma_bytes(lengths, first_positions, tokens, heads=64, bq=6):
    """Count explicit DMA payload, including unused final SWA prefetch."""
    window_loads = 0
    for length, position in zip(lengths, first_positions):
        if not length:
            continue
        anchor = max(0, position - 127)
        cache = [-1, -1]
        prefetched = -1
        t = 0
        while t < length:
            p = position + t
            block = (max(0, p - 127) - anchor) // 128
            count = min(bq, length - t, anchor + (block + 2) * 128 - p)
            for wanted in (block, block + 1):
                slot = wanted % 2
                if cache[slot] == wanted:
                    continue
                window_loads += prefetched != wanted
                prefetched = -1
                cache[slot] = wanted
                if (
                    wanted >= 1
                    and anchor + (wanted + 1) * 128 - 1 <= position + length - 1
                ):
                    window_loads += 1
                    prefetched = wanted + 1
            t += count
    valid = sum(lengths)
    parts = {
        "q_read": valid * heads * 512 * 2,
        "main_kv_read": valid * 512 * 512,
        "main_scale_read": valid * 512 * 4,
        "rope_read": valid * 2 * 128 * 4,
        "swa_read": window_loads * 128 * 512 * 2,
        "sink_read": len(lengths) * heads * 4,
        "output_write_including_padding": tokens * heads * 512 * 2,
        "metadata_logical_bytes": (len(lengths) + 1 + tokens) * 4,
    }
    return {
        "components": parts,
        "swa_window_loads": window_loads,
        "total": sum(parts.values()),
    }


def verify(tests, inputs, output, ratio, valid):
    """Compare sampled rows on every rank with the independent NumPy oracle."""
    rows = np.unique(
        np.concatenate(
            (np.arange(min(valid, 8)), np.linspace(0, valid - 1, 16, dtype=int))
        )
    )
    sample = {
        key: (value if key == "sink" else value[rows])
        for key, value in inputs.items()
        if key != "starts"
    }
    sample["starts"] = jnp.array([0, len(rows)], jnp.int32)
    expected = tests._reference(sample, compress_ratio=ratio)
    errors = []
    for shard in output.addressable_shards:
        actual = np.asarray(shard.data[0, rows], np.float32)
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)
        relative = float(
            np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
        )
        if relative >= 0.006:
            raise AssertionError(f"relative L2 {relative} exceeds 0.006")
        np.testing.assert_array_equal(np.asarray(shard.data[0, valid:]), 0)
        errors.append(
            {
                "device": str(shard.device),
                "relative_l2": relative,
                "max_abs": float(np.max(np.abs(actual - expected))),
            }
        )
    return {"sample_rows": rows.tolist(), "ranks": errors, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--baseline-snapshot", type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    tests = load_tests(root)
    devices = jax.local_devices()
    if len(devices) != 8 or jax.default_backend() != "tpu":
        raise RuntimeError(f"Expected eight TPU7x devices; got {devices}")
    args.output.mkdir(parents=True, exist_ok=True)
    source = root / "src/vllm_torchtpu/kernels/deepseek_v41/csa2.py"
    manifest = {
        "schema_version": 1,
        "workflow": "operator-optimization",
        "operator_family": "attention",
        "operator_name": "csa2",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "hardware": {"device_type": "v7x", "device_count": 8, "physical_chips": 4},
        "versions": {
            p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")
        },
        "dimensions": {"heads": 64, "head_dim": 512, "topk": 512, "swa": 128, "bq": 6},
        "bandwidth": {
            "per_chip_bytes_s": 7.38e12,
            "per_logical_device_bytes_s": 3.69e12,
            "source": "https://docs.cloud.google.com/docs/compute-area/tpus?hl=en",
        },
        "scope": (
            "pregathered SWA + CSA + inverse RoPE; "
            "no cache gather or cache-format conversion"
        ),
    }
    if args.baseline_snapshot is not None:
        manifest["baseline"] = json.loads(
            (args.baseline_snapshot / "provenance.json").read_text()
        )
        manifest["baseline_scope"] = (
            "native staged-KV attention, including packed SWA gather, "
            "fused window attention and full-stage SparseMLA; inverse RoPE adapter"
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    if args.profile:
        import torch
        import torch_tpu  # noqa: F401
        from torch_tpu._internal.profiler import TpuProfilerConfig

        # Initialize the profiler's PJRT client before the first capture.
        torch.tpu.synchronize()
    cases = (
        ("decode_encoder", (6,) * 48, (98304,) * 48, 0, 2),
        ("decode_decoder", (6,) * 48, (98304,) * 48, 0, 1),
        ("prefill_encoder", (2044,), (100220,), 4, 2),
    )
    metrics = args.output / "rank-0/benchmark/metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    for name, lengths, positions, padding, ratio in cases:
        print(f"CASE {name}: generating inputs", flush=True)
        inputs = tests._inputs(
            lengths=lengths, starts=positions, padding=padding, heads=64
        )
        jax.block_until_ready(inputs)
        variants = [
            (
                "candidate",
                functools.partial(tests.csa2.csa2, compress_ratio=ratio),
                inputs,
            )
        ]
        if args.baseline_snapshot is not None:
            import csa2_baseline_adapter

            baseline, repack = csa2_baseline_adapter.load(args.baseline_snapshot)
            baseline_inputs = csa2_baseline_adapter.prepare(inputs, repack)
            variants.append(("baseline", baseline, baseline_inputs))
        for variant, call, operands in variants:
            print(f"CASE {name} VARIANT {variant}", flush=True)
            names = tuple(operands)
            values = tuple(operands.values())

            def execute(rank, *arrays, repetitions, names=names, call=call):
                del rank
                kwargs = dict(zip(names, arrays))
                q = kwargs.pop("q")
                return jax.lax.fori_loop(
                    0,
                    repetitions,
                    lambda _, q: call(q, **kwargs),
                    q,
                )

            single = jax.pmap(
                functools.partial(execute, repetitions=1),
                in_axes=(0,) + (None,) * len(values),
            )
            repeated = jax.pmap(
                functools.partial(execute, repetitions=args.reps),
                in_axes=(0,) + (None,) * len(values),
            )
            ranks = np.arange(len(devices), dtype=np.int32)
            start = time.perf_counter()
            compiled = single.lower(ranks, *values).compile()
            resident = jax.tree.map(
                jax.device_put, (ranks, *values), compiled.input_shardings[0]
            )
            jax.block_until_ready(resident)
            output = compiled(*resident)
            jax.block_until_ready(output)
            compile_s = time.perf_counter() - start
            valid, tokens = sum(lengths), inputs["q"].shape[0]
            correctness = verify(tests, inputs, output, ratio, valid)
            print(f"CASE {name}: correctness passed, compiling timed loop", flush=True)
            compiled_repeated = repeated.lower(ranks, *values).compile()
            resident_repeated = jax.tree.map(
                jax.device_put, resident, compiled_repeated.input_shardings[0]
            )
            jax.block_until_ready(resident_repeated)
            for _ in range(3):
                jax.block_until_ready(compiled_repeated(*resident_repeated))
            samples = []
            for _ in range(args.samples):
                start = time.perf_counter_ns()
                result = compiled_repeated(*resident_repeated)
                jax.block_until_ready(result)
                samples.append((time.perf_counter_ns() - start) / 1e6 / args.reps)
            traffic = (
                dma_bytes(lengths, positions, tokens)
                if variant == "candidate"
                else None
            )
            median = statistics.median(samples)
            row = {
                "variant": variant,
                "case": name,
                "tokens": tokens,
                "valid_tokens": valid,
                "batch_size": len(lengths),
                "q_len": lengths[0],
                "seq_len": positions[0],
                "compress_ratio": ratio,
                "bq": 6,
                "latency_ms": median,
                "samples_ms": samples,
                "compile_time_s": compile_s,
                "dma_bytes_per_rank": traffic,
                "effective_bandwidth_gb_s": (
                    traffic["total"] / (median * 1e6) if traffic else None
                ),
                "mbu_percent": (
                    traffic["total"] / (median / 1000 * 3.69e12) * 100
                    if traffic
                    else None
                ),
                "correctness": correctness,
            }
            with metrics.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print("RESULT " + json.dumps(row), flush=True)
            if args.profile:
                path = args.output / "rank-0/profiling/xprof" / (name + "-" + variant)
                path.mkdir(parents=True, exist_ok=True)
                # Match the serving benchmark's profiler. JAX's plugin profiler
                # ABI is incompatible with this pinned libtpu runtime.
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.PrivateUse1,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(path)),
                    experimental_config=TpuProfilerConfig(
                        run_dir=str(path),
                        host_tracer_level=1,
                        device_tracer_level=1,
                        python_tracer_level=0,
                        experimental_options={"tpu_trace_mode": "TRACE_COMPUTE"},
                    ),
                ):
                    for i in range(5):
                        with torch.profiler.record_function(f"{name}_step{i}"):
                            jax.block_until_ready(compiled(*resident))
            del (
                output,
                result,
                values,
                resident,
                resident_repeated,
                compiled,
                compiled_repeated,
            )
            jax.clear_caches()
    (args.output / "complete.json").write_text(json.dumps({"status": "succeeded"}))


if __name__ == "__main__":
    main()
