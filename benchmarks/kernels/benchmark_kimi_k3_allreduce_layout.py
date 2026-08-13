#!/usr/bin/env python3
"""Compare Kimi-K3's variadic and concatenated TP32 all-reduces.

Run one coordinated process per TPU host.  Each sample executes a dependent
loop inside one compiled program, so host dispatch is amortized and the result
tracks the device collective rather than Python launch overhead.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

HIDDEN = 7168
LATENT = 3584
AXIS = "tp"

# Match the TP32 replica order emitted by the current K3 executable.  Device
# IDs are resolved explicitly because jax.devices() ordering is not a stable
# topology contract.
K3_TP32_ORDER = (
    16, 17, 20, 21, 22, 23, 18, 19,
    24, 25, 28, 29, 30, 31, 26, 27,
    0, 1, 4, 5, 6, 7, 2, 3,
    8, 9, 12, 13, 14, 15, 10, 11,
)


def _mesh() -> Mesh:
    devices = {device.id: device for device in jax.devices()}
    missing = sorted(set(K3_TP32_ORDER) - devices.keys())
    if missing:
        raise RuntimeError(
            f"expected global TPU IDs 0..31; missing device IDs {missing}")
    return Mesh(np.asarray([devices[index] for index in K3_TP32_ORDER]),
                (AXIS, ))


def _variadic(hidden: jax.Array,
              latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Separate psums are intentionally adjacent.  With the accepted async
    # collective merger, the full K3 executable lowers the analogous pair as
    # one variadic all-reduce with two result buffers.
    return lax.psum(hidden, AXIS), lax.psum(latent, AXIS)


def _concatenated(hidden: jax.Array,
                  latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    fused = jnp.concatenate((hidden, latent), axis=-1)
    fused = lax.psum(fused, AXIS)
    return fused[..., :HIDDEN], fused[..., HIDDEN:]


def _mapped(fn: Callable, mesh: Mesh) -> Callable:
    return jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )


def _compile_loop(fn: Callable, hidden: jax.Array, latent: jax.Array,
                  iterations: int) -> Callable:

    def loop_fn(hidden_value, latent_value):
        return lax.fori_loop(
            0,
            iterations,
            lambda _, values: fn(values[0], values[1]),
            (hidden_value, latent_value),
        )

    return jax.jit(loop_fn).lower(hidden, latent).compile()


def _measure(name: str, fn: Callable, hidden: jax.Array, latent: jax.Array,
             iterations: int, samples: int, expected_shapes: tuple[str, ...],
             validate_hlo: bool) -> list[float]:
    compiled = _compile_loop(fn, hidden, latent, iterations)
    all_reduce_lines = [
        line.strip() for line in compiled.as_text().splitlines()
        if "all-reduce(" in line
    ]
    hlo_matches = (len(all_reduce_lines) == 1 and all(
        shape in all_reduce_lines[0] for shape in expected_shapes))
    if jax.process_index() == 0:
        print(f"{name}_optimized_hlo_all_reduce_lines="
              f"{len(all_reduce_lines)}",
              flush=True)
        for index, line in enumerate(all_reduce_lines[:4]):
            print(f"{name}_all_reduce_hlo_{index}={line[:1000]}",
                  flush=True)
        print(f"{name}_hlo_matches_expected={str(hlo_matches).lower()}",
              flush=True)
    if validate_hlo and not hlo_matches:
        raise RuntimeError(
            f"{name} optimized HLO does not contain exactly one all-reduce "
            f"with shapes {expected_shapes}; refusing a misleading timing")
    values = compiled(hidden, latent)
    jax.block_until_ready(values)
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        values = compiled(*values)
        jax.block_until_ready(values)
        timings.append((time.perf_counter() - start) * 1e6 / iterations)
    if jax.process_index() == 0:
        rendered = ",".join(f"{timing:.3f}" for timing in timings)
        print(f"{name}_device_loop_us={rendered}", flush=True)
        print(f"{name}_median_us={statistics.median(timings):.3f}",
              flush=True)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--skip-hlo-validation", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.samples <= 0:
        parser.error("--iterations and --samples must be positive")

    mesh = _mesh()
    hidden = jnp.zeros((1, HIDDEN), dtype=jnp.bfloat16)
    latent = jnp.zeros((1, LATENT), dtype=jnp.bfloat16)
    variadic = _mapped(_variadic, mesh)
    concatenated = _mapped(_concatenated, mesh)

    if jax.process_index() == 0:
        print(f"device_kind={jax.devices()[0].device_kind} "
              f"device_count={jax.device_count()} "
              f"process_count={jax.process_count()} "
              f"iterations={args.iterations} samples={args.samples}",
              flush=True)

    variadic_times = _measure("variadic", variadic, hidden, latent,
                              args.iterations, args.samples,
                              ("bf16[1,7168]", "bf16[1,3584]"),
                              not args.skip_hlo_validation)
    concat_times = _measure("concatenated", concatenated, hidden, latent,
                            args.iterations, args.samples,
                            ("bf16[1,10752]", ),
                            not args.skip_hlo_validation)
    if jax.process_index() == 0:
        delta = (statistics.median(concat_times) -
                 statistics.median(variadic_times))
        print(f"concatenated_minus_variadic_median_us={delta:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
