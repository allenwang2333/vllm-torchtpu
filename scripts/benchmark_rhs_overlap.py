# SPDX-License-Identifier: Apache-2.0
"""Device-traced FP8 RHS DMA / matmul / overlap microbenchmark on eight ranks."""

import argparse
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

FP8 = jnp.float8_e4m3fn
K, N, POOL = 512, 256, 512
DN = (((1,), (0,)), ((), ()))


def build(mesh, mode, m, reuse, steps):
    label = f"mb_{mode}_m{m}_r{reuse}_s{steps}"
    lhs_count = max(1, reuse)

    def kernel(lhs, rhs, out, last_rhs, lv, rv, acc, sems, io_sem):
        def copy_rhs(i, slot):
            return pltpu.make_async_copy(rhs.at[i], rv.at[slot], sems.at[slot])

        def sync(src, dst):
            op = pltpu.make_async_copy(src, dst, io_sem)
            op.start()
            op.wait()

        if mode != "dma":
            sync(lhs, lv)
        acc[...] = jnp.zeros((m, N), jnp.float32)
        if mode == "compute":
            for slot in range(2):
                copy_rhs(slot, slot).start()
                copy_rhs(slot, slot).wait()
        elif mode == "overlap":
            copy_rhs(jnp.int32(0), jnp.int32(0)).start()

        def step(i, unused):
            slot = lax.rem(i, jnp.int32(2))
            if mode in ("dma", "serial"):
                copy_rhs(i, slot).start()
                copy_rhs(i, slot).wait()
            elif mode == "overlap":
                copy_rhs(i, slot).wait()

                @pl.when(i + 1 < steps)
                def _():
                    copy_rhs(i + 1, 1 - slot).start()

            if mode != "dma":
                # Consume every matmul in the result, preventing DCE/hoisting.
                # The FP32 checksum accumulation is included in the timing.
                for batch in range(reuse):
                    acc[...] = acc[...] + lax.dot_general(
                        lv[batch], rv[slot], DN, preferred_element_type=jnp.float32
                    )
            return unused

        with jax.named_scope(label):
            lax.fori_loop(0, steps, step, jnp.int32(0))
        sync(acc, out)
        sync(rv.at[(steps - 1) % 2], last_rhs)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, N), jnp.float32),
            jax.ShapeDtypeStruct((K, N), FP8),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((lhs_count, m, K), FP8),
                pltpu.VMEM((2, K, N), FP8),
                pltpu.VMEM((m, N), jnp.float32),
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
            call,
            mesh=mesh,
            in_specs=(P("d"), P("d")),
            out_specs=(P("d"), P("d")),
            check_vma=False,
        )
    )
    return label, fn


def read_trace(profile_dir, labels):
    from vllm_torchtpu.tools import xplane_pb2

    records, inventory = {}, {}
    for path in profile_dir.rglob("*.xplane.pb"):
        space = xplane_pb2.XSpace()
        space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            if not plane.name.startswith("/device:TPU:"):
                continue
            for line in plane.lines:
                for event in line.events:
                    name = plane.event_metadata[event.metadata_id].name
                    matching = [label for label in labels if label in name]
                    if not matching:
                        continue
                    key = (line.name, name)
                    inventory[str(key)] = inventory.get(str(key), 0) + 1
                    for label in matching:
                        records.setdefault(label, []).append(
                            {
                                "device": plane.name,
                                "line": line.name,
                                "event": name,
                                "duration_us": event.duration_ps / 1e6,
                            }
                        )
    return records, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output)
    devices = jax.local_devices()
    if len(devices) != 8 or jax.process_count() != 1:
        raise RuntimeError("Requires one host, eight local TPU devices")
    mesh = Mesh(np.asarray(devices), ("d",))
    sharding = NamedSharding(mesh, P("d"))
    rng = np.random.default_rng(20260920)
    rbase = rng.integers(-1, 2, size=(2, K, N), dtype=np.int8).astype(np.float32)
    # Distinct HBM addresses, but alternating known values for exact checking.
    rhs = jax.device_put(np.tile(rbase, (8 * POOL // 2, 1, 1)), sharding).astype(FP8)
    rhs.block_until_ready()
    configurations = (
        [(128, 1)]
        if args.pilot
        else [(m, 1) for m in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
    )
    if not args.pilot:
        configurations += [(m, reuse) for m in (128, 256) for reuse in (2, 4, 8, 16)]
    if args.refine:
        configurations = [(m, 1) for m in (5120, 5376, 5504, 5632, 6144)]
        configurations += [(128, r) for r in (32, 48, 64)]
        configurations += [(256, r) for r in (24, 32)]
    compiled, metadata = [], []
    lengths = (128, 512)
    for m, reuse in configurations:
        lhs_np = rng.integers(-1, 2, size=(reuse, m, K), dtype=np.int8).astype(
            np.float32
        )
        lhs = jax.device_put(np.tile(lhs_np, (8, 1, 1)), sharding).astype(FP8)
        for mode in ("compute", "serial", "overlap"):
            for steps in lengths:
                label, fn = build(mesh, mode, m, reuse, steps)
                executable = fn.lower(lhs, rhs).compile()
                expected = sum(
                    lhs_np[b] @ (rbase[0] + rbase[1]) for b in range(reuse)
                ) * (steps // 2)
                for _ in range(3):
                    output = executable(lhs, rhs)
                    jax.block_until_ready(output)
                np.testing.assert_array_equal(
                    np.asarray(output[0]), np.tile(expected, (8, 1))
                )
                np.testing.assert_array_equal(
                    np.asarray(output[1]), np.tile(rbase[(steps - 1) % 2], (8, 1))
                )
                compiled.append((label, executable, lhs))
                metadata.append(
                    dict(label=label, mode=mode, m=m, reuse=reuse, steps=steps)
                )
                print(f"Compiled and validated {label}", flush=True)
    lhs = jax.device_put(np.zeros((8, 32, K), np.float32), sharding).astype(FP8)
    for steps in lengths:
        label, fn = build(mesh, "dma", 32, 1, steps)
        executable = fn.lower(lhs, rhs).compile()
        for _ in range(3):
            output = executable(lhs, rhs)
            jax.block_until_ready(output)
        np.testing.assert_array_equal(
            np.asarray(output[1]), np.tile(rbase[(steps - 1) % 2], (8, 1))
        )
        compiled.append((label, executable, lhs))
        metadata.append(dict(label=label, mode="dma", m=32, reuse=1, steps=steps))
    (args.output / "configurations.json").write_text(json.dumps(metadata, indent=2))
    import torch
    from torch_tpu._internal import profiler

    torch.ones((1,), device="tpu").cpu()
    print("Capturing device profile", flush=True)
    started = time.time()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.TPU],
        on_trace_ready=profiler.xprof_trace_handler(args.output / "profile"),
    ):
        for repeat in range(args.repeats):
            for label, executable, lhs in compiled:
                with jax.profiler.StepTraceAnnotation(label, step_num=repeat):
                    jax.block_until_ready(executable(lhs, rhs))
    records, inventory = read_trace(
        args.output / "profile", [row["label"] for row in metadata]
    )
    (args.output / "trace_events.json").write_text(json.dumps(records, indent=2))
    (args.output / "trace_inventory.json").write_text(json.dumps(inventory, indent=2))
    print(
        f"Captured in {time.time() - started:.1f}s; {len(records)} configurations",
        flush=True,
    )
    print(json.dumps(dict(list(inventory.items())[:20]), indent=2), flush=True)


if __name__ == "__main__":
    main()
