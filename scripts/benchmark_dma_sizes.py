# SPDX-License-Identifier: Apache-2.0
"""FP8 HBM->VMEM DMA sizes and source-column-stride sweep on eight TPU ranks."""

import argparse
import csv
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

FP8 = jnp.float8_e4m3fn
SHAPES = [
    (256, 512),
    (512, 512),
    (1024, 512),
    (1024, 1024),
    (2048, 1024),
    (2048, 2048),
    (4096, 2048),
    (512, 256),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
]


def build(mesh, k, n, stride, pool, steps):
    label = f"dma_k{k}_n{n}_pitch{stride}_s{steps}"
    offset = n if stride == 2 else 0

    def kernel(src, out, vm, sems, out_sem):
        def copy(i, slot):
            index = lax.rem(i, jnp.int32(pool))
            return pltpu.make_async_copy(
                src.at[index, :, pl.ds(offset, n)], vm.at[slot], sems.at[slot]
            )

        def step(i, unused):
            slot = lax.rem(i, jnp.int32(2))
            copy(i, slot).start()
            copy(i, slot).wait()
            return unused

        lax.fori_loop(0, steps, step, jnp.int32(0))
        op = pltpu.make_async_copy(vm.at[(steps - 1) % 2], out, out_sem)
        op.start()
        op.wait()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((k, n), FP8),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm,),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, k, n), FP8),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 2**20),
        name=label,
    )
    fn = jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=(P("d"),), out_specs=P("d"), check_vma=False
        )
    )
    return label, fn


def read_dma_trace(profile_dir, configs, repeats):
    from vllm_torchtpu.tools import xplane_pb2

    paths = list(profile_dir.rglob("*.xplane.pb"))
    if len(paths) != 1:
        raise ValueError(f"Expected one local-host trace, got {paths}")
    space = xplane_pb2.XSpace()
    space.ParseFromString(paths[0].read_bytes())
    records, aliases = {}, []
    for plane in space.planes:
        if not plane.name.startswith("/device:TPU:"):
            continue
        events = []
        for line in plane.lines:
            if line.name != "XLA Ops":
                continue
            for event in line.events:
                name = plane.event_metadata[event.metadata_id].name
                if "dma_k" in name and "custom-call" in name:
                    events.append(
                        (
                            line.timestamp_ns * 1000 + event.offset_ps,
                            name,
                            event.duration_ps / 1e6,
                        )
                    )
        events.sort()
        if len(events) != len(configs) * repeats:
            raise ValueError(f"Incomplete capture on {plane.name}: {len(events)}")
        for index, (_, name, duration) in enumerate(events):
            config = configs[index % len(configs)]
            match = re.search(r"dma_k(\d+)_n(\d+)_pitch(\d+)_s(\d+)", name)
            k, n, stride, steps = map(int, match.groups())
            if (k * n, stride, steps) != (
                config["bytes"],
                config["stride"],
                config["steps"],
            ):
                raise ValueError(f"Capture order mismatch: {config} vs {name}")
            if stride != 1 and (k, n) != (config["k"], config["n"]):
                raise ValueError("Unexpected alias for a strided source")
            label = config["label"]
            if label not in name:
                aliases.append(
                    dict(device=plane.name, requested=label, actual=match.group())
                )
            records.setdefault(label, []).append(
                dict(
                    device=plane.name,
                    line="XLA Ops",
                    event=name,
                    duration_us=duration,
                    repeat=index // len(configs),
                )
            )
    if len({r["device"] for rows in records.values() for r in rows}) != 8:
        raise ValueError("Expected eight device planes")
    return records, aliases


def summarize(root, repeats):
    configs = json.loads((root / "configurations.json").read_text())
    events = json.loads((root / "trace_events.json").read_text())
    rows = []
    for c in configs:
        if c["steps"] != 128:
            continue
        durations = {}
        for steps in (128, 512):
            label = f"dma_k{c['k']}_n{c['n']}_pitch{c['stride']}_s{steps}"
            per = {}
            for e in events[label]:
                if e["line"] == "XLA Ops" and "custom-call" in e["event"]:
                    per.setdefault(e["device"], []).append(e["duration_us"])
            if len(per) != 8 or any(len(v) != repeats for v in per.values()):
                raise ValueError(f"Incomplete trace for {label}")
            durations[steps] = {d: float(np.median(v)) for d, v in per.items()}
        slopes = {
            d: (durations[512][d] - durations[128][d]) / 384 for d in durations[128]
        }
        latency = float(np.median(list(slopes.values())))
        row = {k: c[k] for k in ("k", "n", "stride", "pool", "bytes")}
        row.update(
            latency_us=latency,
            effective_GB_s=c["bytes"] / latency / 1000,
            rank_min_us=min(slopes.values()),
            rank_max_us=max(slopes.values()),
            per_rank_us=slopes,
            device_duration_us=durations,
        )
        rows.append(row)
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    fields = [
        "k",
        "n",
        "stride",
        "pool",
        "bytes",
        "latency_us",
        "effective_GB_s",
        "rank_min_us",
        "rank_max_us",
    ]
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(
            f"{r['k']}x{r['n']} stride={r['stride']} {r['bytes'] / 2**20:.3f} MiB: "
            f"{r['latency_us']:.3f} us, {r['effective_GB_s']:.1f} GB/s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output)
    shutil.copy2(Path(__file__).with_name("benchmark_rhs_overlap.py"), args.output)
    devices = jax.local_devices()
    if len(devices) != 8 or jax.process_count() != 1 or jax.default_backend() != "tpu":
        raise RuntimeError("Requires one host with eight local TPU devices")
    mesh = Mesh(np.asarray(devices), ("d",))
    compiled = []
    configs = []
    for k, n in SHAPES:
        pool = max(16, 64 * 2**20 // (k * n))
        for stride in (1, 2):

            def generate(_):
                blocks = jnp.arange(pool, dtype=jnp.int32)[:, None, None]
                rows = jnp.arange(k, dtype=jnp.int32)[None, :, None]
                cols = jnp.arange(n * stride, dtype=jnp.int32)[None, None, :]
                values = (
                    blocks * 7
                    + rows * 5
                    + cols * 3
                    + cols // 128
                    + lax.axis_index("d") * 11
                ) % 17 - 8
                return values.astype(FP8)

            src = jax.jit(
                jax.shard_map(
                    generate,
                    mesh=mesh,
                    in_specs=P("d"),
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(jnp.zeros((8,), jnp.int32))
            src.block_until_ready()
            for steps in (128, 512):
                label, fn = build(mesh, k, n, stride, pool, steps)
                exe = fn.lower(src).compile()
                for _ in range(3):
                    output = exe(src).block_until_ready()
                # Independent full-array comparison for the last source block.
                offset = n if stride == 2 else 0
                ranks = np.arange(8, dtype=np.int32)[:, None, None]
                rows = np.arange(k, dtype=np.int32)[None, :, None]
                cols = np.arange(offset, offset + n, dtype=np.int32)[None, None, :]
                expected = (
                    ((steps - 1) % pool) * 7
                    + rows * 5
                    + cols * 3
                    + cols // 128
                    + ranks * 11
                ) % 17 - 8
                expected = expected.astype(np.dtype(FP8)).reshape(8 * k, n)
                np.testing.assert_array_equal(
                    np.asarray(output).view(np.uint8), expected.view(np.uint8)
                )
                compiled.append((label, exe, src))
                configs.append(
                    dict(
                        label=label,
                        k=k,
                        n=n,
                        stride=stride,
                        pool=pool,
                        steps=steps,
                        bytes=k * n,
                    )
                )
                print("Compiled and validated " + label, flush=True)
    (args.output / "configurations.json").write_text(json.dumps(configs, indent=2))
    (args.output / "metadata.json").write_text(
        json.dumps(
            dict(
                devices=[str(d) for d in devices],
                device_kind=devices[0].device_kind,
                repeats=args.repeats,
                warmups=3,
                dtype="FP8 E4M3",
                versions={
                    p: importlib.metadata.version(p)
                    for p in ("jax", "jaxlib", "libtpu", "numpy")
                },
                timing="Device XPlane XLA Ops slopes: (T512-T128)/384, per-rank median then median across ranks.",
                source="stride=1: whole contiguous matrix. stride=2: right half columns of a matrix twice as wide.",
                schedule="One outstanding DMA at a time; alternating two VMEM buffers; source pool cycles at >=64 MiB of payload per rank.",
                validation="Every warmed executable output checked exactly against the last source block on all eight ranks.",
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
            for label, exe, src in compiled:
                with jax.profiler.StepTraceAnnotation(label, step_num=repeat):
                    exe(src).block_until_ready()
    records, aliases = read_dma_trace(args.output / "profile", configs, args.repeats)
    (args.output / "trace_events.json").write_text(json.dumps(records, indent=2))
    (args.output / "trace_aliases.json").write_text(json.dumps(aliases, indent=2))
    summarize(args.output, args.repeats)


if __name__ == "__main__":
    main()
