# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the maintained packed FP8 MQA entry point (TPU v7, P-state 7)."""

import argparse
import importlib.metadata
import json
import os
import types
from pathlib import Path

import bench_packed_mqa_compute as compute
import bench_packed_mqa_metadata as metadata
import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    pack_fp8_pages,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
    packed_fp8_mqa_logits,
)


def compile_kernel(inputs: tuple, out: Path) -> jax.stages.Compiled:
    def metadata_trial_full_packed_fp8(
        q, weights, pool, pages, lengths, offsets, scales
    ):
        return packed_fp8_mqa_logits(
            q, weights, pool, pages, lengths, offsets, q_scale=scales
        )

    executable = jax.jit(metadata_trial_full_packed_fp8).lower(*inputs).compile()
    (out / "packed_fp8.hlo.txt").write_text(executable.as_text())
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--pstate", type=int, choices=(7,), default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
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
        compile_kernel(inputs, args.out)
        return
    flags = os.environ.get("LIBTPU_INIT_ARGS", "")
    assert [f for f in flags.split() if f.startswith("--xla_tpu_dvfs_p_state=")] == [
        "--xla_tpu_dvfs_p_state=7"
    ]
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    (args.out / "environment.json").write_text(
        json.dumps(dict(versions=versions, flags=flags), indent=2)
    )
    packed = types.SimpleNamespace(pack_pages=pack_fp8_pages)
    executable = None
    with jax.default_device(jax.devices()[0]):
        for seed in (2026, 2027):
            inputs = metadata.make_inputs(seed, packed)
            if executable is None:
                executable = compile_kernel(inputs, args.out)
            compute.measure(
                {"packed_fp8": executable},
                inputs,
                seed,
                args.out,
                args.pstate,
                correctness=None,
            )


if __name__ == "__main__":
    main()
