# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare explicit-MXU fusion and the retained high-level FP8 MQA kernel."""

import argparse
import dataclasses
import functools
import importlib.metadata
import json
import os
import time
import types
from pathlib import Path

import bench_packed_mqa_metadata as metadata
import jax
import jax.numpy as jnp
import numpy as np

CONFIG = metadata.PagedMQAConfig(6, 32768, 8, 6, 4)
VARIANTS = ("baseline", "highlevel", "fused")


def load_module(directory: Path, name: str) -> types.ModuleType:
    prefix = "vllm_torchtpu.kernels.deepseek_v41.index_selection."
    suffix = "fused" if name == "highlevel" else "control"
    metadata.load(prefix + "paged_mqa_pipeline", directory / f"pipeline_{suffix}.py")
    path = directory / f"logits_{suffix}.py"
    source = path.read_text().replace(
        'name="paged_mqa_logits"', f'name="paged_mqa_full_{name}"'
    )
    return metadata.load(prefix + f"fused_reduce_{name}", path, source)


def compile_variant(
    module: types.ModuleType, inputs: tuple, name: str, out: Path
) -> jax.stages.Compiled:
    def score(q, weights, pool, pages, lengths, offsets, q_scale):
        return module.paged_mqa_logits(
            q,
            weights,
            pool,
            pages,
            lengths,
            offsets,
            page_records=128,
            q_scale=q_scale,
            config=CONFIG,
        )

    score.__name__ = f"metadata_trial_full_{name}"
    print("COMPILE", name, flush=True)
    executable = jax.jit(score).lower(*inputs).compile()
    (out / f"{name}.hlo.txt").write_text(executable.as_text())
    return executable


def validate(runs: dict, inputs: tuple, seed: int, out: Path) -> None:
    first = inputs[2][0].shape[0] // 33
    total = sum(bank.shape[0] for bank in inputs[2]) // 33
    records = []
    for edge in (False, True):
        pages, lengths, offsets = inputs[3:6]
        if edge:
            pages = pages.at[0, :256].set(0)
            pages = pages.at[1, :256].set(total - 1)
            pages = pages.at[2, jnp.array([0, 255, 256, 511, 512, 767])].set(
                jnp.array(
                    [
                        np.iinfo(np.int32).min,
                        -1,
                        first - 1,
                        first,
                        total,
                        np.iinfo(np.int32).max,
                    ],
                    jnp.int32,
                )
            )
            lengths = lengths.at[:18].set(
                jnp.array(
                    [
                        0,
                        1,
                        127,
                        128,
                        32767,
                        32768,
                        32769,
                        65535,
                        65536,
                        65537,
                        98304,
                        98303,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    jnp.int32,
                )
            )
            offsets = offsets.at[1].set(0)
        data = (*inputs[:3], pages, lengths, offsets, inputs[6])
        baseline = runs["baseline"](*data)
        for name in runs:
            if name == "baseline":
                continue
            candidate = runs[name](*data)
            same_bits = bool(metadata.same_bits(baseline, candidate, mode="full"))
            expected, actual = np.asarray(baseline), np.asarray(candidate)
            if name == "fused":
                assert same_bits, (name, seed, edge)
            # The retained high-level path was validated with this tolerance.
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-4)
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
            finite = np.isfinite(expected)
            record = dict(
                variant=name,
                seed=seed,
                edge=edge,
                bitwise=same_bits,
                max_abs_error=float(np.max(np.abs(actual[finite] - expected[finite]))),
            )
            records.append(record)
            print("PASS", json.dumps(record), flush=True)
    (out / f"correctness_s{seed}.json").write_text(json.dumps(records, indent=2))


def measure(
    runs: dict,
    inputs: tuple,
    seed: int,
    out: Path,
    pstate: int,
    *,
    correctness: bool | None = True,
) -> None:
    calls = {name: functools.partial(run, *inputs) for name, run in runs.items()}
    order = [*runs, *reversed(runs)]
    for _ in range(5):
        for name in order:
            jax.block_until_ready(calls[name]())
    sentinel_arg = jnp.asarray(0, jnp.int32)
    sentinel = jax.jit(lambda x: x + 1).lower(sentinel_arg).compile()
    jax.block_until_ready(sentinel(sentinel_arg))
    samples = []
    for capture in range(4):
        for attempt in range(5):
            path = (
                out.parents[1]
                / "profiling/xprof"
                / (f"compute_p{pstate}_s{seed}_{capture}_{attempt}")
            )
            with jax.profiler.trace(str(path)):
                time.sleep(0.1)
                jax.block_until_ready(sentinel(sentinel_arg))
                for name in order * 5:
                    jax.block_until_ready(calls[name]())
                    jax.block_until_ready(sentinel(sentinel_arg))
                time.sleep(0.1)
            try:
                rows = metadata.parse_capture(path, "full", order * 5)
            except ValueError as error:
                if not str(error).startswith("Incomplete trace:") or attempt == 4:
                    raise
                print("RETRY", error, flush=True)
            else:
                samples.extend(
                    dict(row, seed=seed, requested_pstate=pstate) for row in rows
                )
                break
    (out / f"samples_s{seed}.json").write_text(json.dumps(samples, indent=2))
    for name in runs:
        selected = [r for r in samples if r["variant"] == name]
        record = dict(
            variant=name,
            seed=seed,
            correctness=correctness,
            latency_ms=float(np.median([r["module_us"] for r in selected])) / 1000,
            kernel_us=float(np.median([r["kernel_us"] for r in selected])),
            samples=len(selected),
            config=dataclasses.asdict(CONFIG),
            cache_bank_count=2,
            requested_pstate=pstate,
            timing="raw XProf device_duration_ps; matched interleaved variant order",
        )
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--highlevel-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=["baseline", "highlevel"]
    )
    parser.add_argument("--pstate", type=int, choices=(-1, 7), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "metrics.jsonl").exists():
        raise ValueError("Use a new output directory")
    if "baseline" not in args.variants:
        parser.error("The baseline is required for correctness validation")
    directories = dict(
        baseline=args.baseline_dir,
        highlevel=args.highlevel_dir,
        fused=args.candidate_dir,
    )
    if "fused" in args.variants and args.candidate_dir is None:
        parser.error("--candidate-dir is required for the fused variant")
    modules = {name: load_module(directories[name], name) for name in args.variants}
    if args.compile_only:
        shape = jax.ShapeDtypeStruct
        bank = shape((608256, 128), jnp.int32)
        inputs = (
            shape((288, 32, 128), jnp.float8_e4m3fn),
            shape((288, 32), jnp.float32),
            (bank, bank),
            shape((48, 768), jnp.int32),
            shape((288,), jnp.int32),
            shape((49,), jnp.int32),
            shape((288, 32), jnp.float32),
        )
        for name in modules:
            compile_variant(modules[name], inputs, name, args.out)
        return
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    assert versions == {"jax": "0.11.2", "jaxlib": "0.11.2", "libtpu": "0.0.50"}
    flags = os.environ.get("LIBTPU_INIT_ARGS", "")
    assert [f for f in flags.split() if f.startswith("--xla_tpu_dvfs_p_state=")] == [
        f"--xla_tpu_dvfs_p_state={args.pstate}"
    ]
    (args.out / "environment.json").write_text(
        json.dumps(
            dict(versions=versions, libtpu_flags=flags, device=str(jax.devices()[0])),
            indent=2,
        )
    )
    packed = metadata.load("fused_reduce_packed", args.baseline_dir / "packed_cache.py")
    runs = {}
    with jax.default_device(jax.devices()[0]):
        assert jax.devices()[0].platform == "tpu"
        for seed in (2026, 2027):
            inputs = metadata.make_inputs(seed, packed)
            if not runs:
                runs = {
                    name: compile_variant(module, inputs, name, args.out)
                    for name, module in modules.items()
                }
            validate(runs, inputs, seed, args.out)
            measure(runs, inputs, seed, args.out, args.pstate)
    print("COMPLETE fused reduction comparison", flush=True)


if __name__ == "__main__":
    main()
