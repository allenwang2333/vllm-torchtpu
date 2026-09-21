# SPDX-License-Identifier: Apache-2.0
"""Compare one/two FP8 matmuls with operands already resident in VMEM."""

import argparse
import contextlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_rhs_overlap import read_trace
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

FP8 = jnp.float8_e4m3fn
K, N = 512, 256
DN = (((1,), (0,)), ((), ()))
MODES = ("single", "shared_lhs", "shared_rhs", "distinct")


def build(mesh, mode, m, steps):
    label = f"pair_{mode}_m{m}_s{steps}"

    def kernel(lhs, rhs, out, lv, rv, acc, sem):
        def sync(src, dst):
            op = pltpu.make_async_copy(src, dst, sem)
            op.start()
            op.wait()

        sync(lhs, lv)
        sync(rhs, rv)
        acc[...] = jnp.zeros((2, m, N), jnp.float32)

        def step(i, unused):
            slot = lax.rem(i, jnp.int32(2))
            acc[0] = acc[0] + lax.dot_general(
                lv[0], rv[slot], DN, preferred_element_type=jnp.float32
            )
            if mode != "single":
                lhs_slot = 0 if mode == "shared_lhs" else 1
                rhs_slot = slot if mode == "shared_rhs" else 1 - slot
                acc[1] = acc[1] + lax.dot_general(
                    lv[lhs_slot],
                    rv[rhs_slot],
                    DN,
                    preferred_element_type=jnp.float32,
                )
            return unused

        with jax.named_scope(label):
            lax.fori_loop(0, steps, step, jnp.int32(0))
        sync(acc, out)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, m, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm, hbm),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, m, K), FP8),
                pltpu.VMEM((2, K, N), FP8),
                pltpu.VMEM((2, m, N), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 2**20),
        name=label,
        debug=True,
    )
    return label, jax.jit(
        jax.shard_map(
            call,
            mesh=mesh,
            in_specs=(P("d"), P("d")),
            out_specs=P("d"),
            check_vma=False,
        )
    )


def summarize(root, repeats):
    configs = json.loads((root / "configurations.json").read_text())
    records, inventory = read_trace(root / "profile", [c["label"] for c in configs])
    (root / "trace_events.json").write_text(json.dumps(records, indent=2))
    (root / "trace_inventory.json").write_text(json.dumps(inventory, indent=2))
    rows = []
    for config in configs:
        if config["steps"] != 128:
            continue
        times = {}
        for steps in (128, 512):
            label = f"pair_{config['mode']}_m{config['m']}_s{steps}"
            ranks = {}
            for event in records[label]:
                if event["line"] == "XLA Ops" and "custom-call" in event["event"]:
                    ranks.setdefault(event["device"], []).append(event["duration_us"])
            if len(ranks) != 8 or any(len(v) != repeats for v in ranks.values()):
                raise ValueError(f"Unexpected capture counts: {label} {ranks}")
            times[steps] = {r: float(np.median(v)) for r, v in ranks.items()}
        slopes = {r: (times[512][r] - times[128][r]) / 384 for r in times[128]}
        if min(slopes.values()) <= 0:
            raise ValueError(f"Nonpositive slope: {config}")
        total_us = float(np.median(list(slopes.values())))
        count = 1 if config["mode"] == "single" else 2
        rows.append(
            dict(
                mode=config["mode"],
                m=config["m"],
                matmuls=count,
                total_us=total_us,
                us_per_matmul=total_us / count,
                effective_tflops=2 * config["m"] * K * N * count / total_us / 1e6,
                per_rank_us=slopes,
                device_duration_us=times,
            )
        )
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row["m"], row["mode"], round(row["total_us"], 5), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output)
    shutil.copy2(Path(__file__).with_name("benchmark_rhs_overlap.py"), args.output)
    devices = jax.local_devices()
    if len(devices) != 8 or jax.process_count() != 1:
        raise RuntimeError("Requires a single host with eight TPU devices")
    mesh = Mesh(np.array(devices), ("d",))
    sharding = NamedSharding(mesh, P("d"))
    rng = np.random.default_rng(20260921)
    rhs_np = rng.integers(-1, 2, (2, K, N), dtype=np.int8).astype(np.float32)
    rhs = jax.device_put(np.tile(rhs_np, (8, 1, 1)), sharding).astype(FP8)
    rhs.block_until_ready()
    compiled, configs = [], []
    ms = (
        [128]
        if args.pilot
        else [32, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 6144, 8192]
    )
    ir_dir = args.output / "ir"
    ir_dir.mkdir()
    for m in ms:
        lhs_np = rng.integers(-1, 2, (2, m, K), dtype=np.int8).astype(np.float32)
        lhs = jax.device_put(np.tile(lhs_np, (8, 1, 1)), sharding).astype(FP8)
        expected_base = np.stack([x @ (rhs_np[0] + rhs_np[1]) for x in lhs_np])
        for mode in MODES:
            for steps in (128, 512):
                label, fn = build(mesh, mode, m, steps)
                with (ir_dir / f"{label}.txt").open("w") as ir_file:
                    with contextlib.redirect_stdout(ir_file):
                        lowered = fn.lower(lhs, rhs)
                        executable = lowered.compile()
                ir = (ir_dir / f"{label}.txt").read_text()
                matmuls = len(re.findall(r"\btpu\.matmul\b", ir))
                expected_count = 1 if mode == "single" else 2
                if matmuls != expected_count:
                    raise ValueError(
                        f"{label}: expected {expected_count} matmuls, found {matmuls}"
                    )
                (ir_dir / f"{label}.hlo.txt").write_text(executable.as_text())
                expected = np.zeros((2, m, N), np.float32)
                expected[0] = expected_base[0] * (steps // 2)
                if mode != "single":
                    expected[1] = expected_base[0 if mode == "shared_lhs" else 1] * (
                        steps // 2
                    )
                for _ in range(3):
                    output = executable(lhs, rhs)
                    jax.block_until_ready(output)
                np.testing.assert_array_equal(
                    np.asarray(output), np.tile(expected, (8, 1, 1))
                )
                compiled.append((label, executable, lhs))
                configs.append(
                    dict(
                        label=label,
                        mode=mode,
                        m=m,
                        steps=steps,
                        mosaic_matmul_count=matmuls,
                    )
                )
                print(
                    f"Compiled and validated {label}, Mosaic matmuls={matmuls}",
                    flush=True,
                )
    (args.output / "configurations.json").write_text(json.dumps(configs, indent=2))
    (args.output / "metadata.json").write_text(
        json.dumps(
            dict(
                devices=[str(d) for d in devices],
                device_kind=devices[0].device_kind,
                versions={
                    p: importlib.metadata.version(p)
                    for p in ("jax", "jaxlib", "libtpu", "numpy")
                },
                repeats=args.repeats,
                warmups=3,
                dtype="FP8 E4M3; FP32 accumulators",
                timing="(T512-T128)/384, median repetitions per rank then median rank slopes",
                residency="Both LHS and RHS arrays preloaded to VMEM; no DMA inside timed loop",
                consumption="One independent FP32 running accumulator per matmul result",
                anti_hoisting="RHS slot alternates every loop iteration; paired outputs remain separate",
                validation="Full outputs exactly compared on all 8 ranks; Mosaic IR matmul count asserted",
            ),
            indent=2,
        )
    )
    import torch
    from torch_tpu._internal import profiler

    torch.ones((1,), device="tpu").cpu()
    print("Capturing device profile", flush=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.TPU],
        on_trace_ready=profiler.xprof_trace_handler(args.output / "profile"),
    ):
        for repeat in range(args.repeats):
            for label, executable, lhs in compiled:
                with jax.profiler.StepTraceAnnotation(label, step_num=repeat):
                    jax.block_until_ready(executable(lhs, rhs))
    summarize(args.output, args.repeats)


if __name__ == "__main__":
    main()
