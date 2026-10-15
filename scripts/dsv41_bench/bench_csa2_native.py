# SPDX-License-Identifier: Apache-2.0
"""Compare the native serving attention bridge, including BF16 SWA gathering.

The candidate uses serving's dispatch: CSA2 for multi-request six-token
batches, the baseline window path for contiguous prefill. Compressed FP8 KV
is staged in both paths, as in the serving stage cache.
Projection and full-model execution are measured separately by server_steps.
"""

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.profiler import TpuProfilerConfig

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels

ROOT = Path(__file__).resolve().parents[2]


def load_fixture():
    path = ROOT / "tests/kernels/deepseek_v41/test_csa2_serving.py"
    spec = importlib.util.spec_from_file_location("csa2_serving_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_baseline(commit):
    path = "src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py"
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)
    name = attention_kernels.__package__ + "._csa2_baseline_snapshot"
    module = types.ModuleType(name)
    module.__package__ = attention_kernels.__package__
    sys.modules[name] = module
    exec(compile(source, f"{commit}/{path}", "exec"), module.__dict__)
    return module.sparse_fp8_attention_jax


def fixture(module, tokens, noncausal):
    lengths = (tokens,) if tokens != 288 else (6,) * 48
    rng = np.random.default_rng(37)
    base, dense = module.oracle._packed_inputs(
        lengths=(2,), first=(98304,), heads=64, padding=0
    )
    positions = np.concatenate([98304 + np.arange(n) for n in lengths]).astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    width = 256 if noncausal else 128
    swa = np.zeros((tokens, width, 512), dtype=jnp.bfloat16)
    lower = np.zeros(tokens, np.int32) if noncausal else positions - 127
    upper = np.zeros(tokens, np.int32) if noncausal else positions + 1
    for begin, end in zip(starts[:-1], starts[1:]):
        length = end - begin
        values = rng.normal(size=(length + 128, 512)).astype(jnp.bfloat16)
        for t in range(begin, end):
            if noncausal:
                swa[t, : length + 128] = values
                upper[t] = length + 128
            else:
                swa[t] = values[t - begin : t - begin + 128]
    inputs = dict(
        q=jnp.asarray(rng.normal(size=(tokens, 64, 512)) * 0.3, jnp.bfloat16),
        main_kv=jnp.broadcast_to(base["main_kv"][1:2], (tokens, 512, 128)),
        main_rope=jnp.broadcast_to(base["main_rope"][1:2], (tokens, 128, 128)),
        swa_kv=jnp.asarray(swa),
        starts=jnp.asarray(starts),
        positions=jnp.asarray(positions),
        main_lengths=jnp.full((tokens,), 512, jnp.int32),
        swa_lower=jnp.asarray(lower),
        swa_upper=jnp.asarray(upper),
        sink=base["sink"],
    )
    return inputs, dense[1], module.serving_inputs(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 288])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    devices = jax.local_devices()
    assert jax.default_backend() == "tpu" and len(devices) == 8, devices
    mesh = Mesh(np.array(devices), ("bench",))
    shard = NamedSharding(mesh, P("bench"))
    module, baseline = load_fixture(), load_baseline(args.baseline_commit)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.profile:
        # Initialize TorchTPU's profiler client before its first capture. The
        # measured work still executes via JAX on all eight devices.
        torch.tpu.synchronize()
    results = []
    for tokens in args.tokens:
        for noncausal in (False, True) if tokens == 288 else (False,):
            inputs, dense, kwargs = fixture(module, tokens, noncausal)
            arrays = {k: v for k, v in kwargs.items() if isinstance(v, jax.Array)}
            static = {k: v for k, v in kwargs.items() if k not in arrays}
            data = {}
            for key, value in arrays.items():
                host = np.asarray(value)[None]
                data[key] = jax.make_array_from_callback(
                    (8, *value.shape), shard, lambda _, host=host: host
                )
            case = dict(tokens=tokens, noncausal=noncausal, ranks=8)
            sample = np.unique(np.linspace(0, tokens - 1, 9, dtype=int))
            oracle_inputs = {
                k: (v if k == "sink" else v[sample])
                for k, v in inputs.items()
                if k != "starts"
            }
            oracle_inputs["starts"] = jnp.array([0, len(sample)], jnp.int32)
            expected = module.oracle._packed_reference(
                oracle_inputs, np.broadcast_to(dense, (len(sample), *dense.shape))
            )
            for name, function in (
                ("baseline", baseline),
                ("candidate", attention_kernels.sparse_fp8_attention_jax),
            ):

                def local(batch, function=function, static=static, name=name):
                    kw = {k: v[0] for k, v in batch.items()}
                    if name == "baseline":
                        kw.pop("positions")
                    return function(**kw, **static)[None]

                run = jax.jit(
                    jax.shard_map(
                        local,
                        mesh=mesh,
                        in_specs=({k: P("bench") for k in data},),
                        out_specs=P("bench"),
                        check_vma=False,
                    )
                )
                begin = time.perf_counter()
                output = run(data).block_until_ready()
                compile_s = time.perf_counter() - begin
                errors = []
                for part in output.addressable_shards:
                    actual = np.asarray(part.data)[0, sample].astype(np.float32)
                    relative = float(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    )
                    if relative >= 0.012:
                        raise AssertionError(f"{name}: relative L2 {relative}")
                    errors.append(relative)
                for _ in range(3):
                    run(data).block_until_ready()
                samples = []
                for _ in range(args.samples):
                    begin = time.perf_counter_ns()
                    for _ in range(10):
                        run(data).block_until_ready()
                    samples.append((time.perf_counter_ns() - begin) / 1e7)
                case[name] = dict(
                    median_ms=statistics.median(samples),
                    samples_ms=samples,
                    compile_s=compile_s,
                    relative_l2=errors,
                )
                if args.profile:
                    profile_dir = str(args.output / f"{tokens}-{noncausal}-{name}")
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.PrivateUse1,
                        ],
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(
                            profile_dir, use_gzip=True
                        ),
                        experimental_config=TpuProfilerConfig(
                            run_dir=profile_dir,
                            host_tracer_level=2,
                            device_tracer_level=1,
                            python_tracer_level=0,
                        ),
                    ):
                        for _ in range(3):
                            run(data).block_until_ready()
                del run, output
            case["speedup"] = (
                case["baseline"]["median_ms"] / case["candidate"]["median_ms"]
            )
            results.append(case)
            print(json.dumps(case), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
