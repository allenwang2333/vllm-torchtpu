#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Characterize the equal-width failure in two_shot_all_reduce_variadic_ws.

Equal operand widths fail at a single round on 32 devices (multi-process) and
pass at 8 (single process); unequal widths pass everywhere. Three hypotheses have
already been wrong -- stale barrier semaphores, input CSE of the zero
workspaces, and dead workspace outputs -- so this stops guessing and reads out
what the kernel actually produced.

``a`` is filled with 1.0 and ``b`` with 2.0, so with N devices the correct
all-reduce is exactly N and 2N. The observed values then name the fault:

    a=2N, b=2N   -> a sourced b's reduced chunk (reda/redb collapsed)
    a=N,  b=N    -> b sourced a's
    a=1.5N       -> the two mixed within a chunk
    ragged       -> only some chunks crossed over, i.e. a slot/index fault
"""

import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--devices", type=int, default=32)
p.add_argument("--width", type=int, default=3584)
p.add_argument("--width-b", type=int, default=None)
p.add_argument("--rounds", type=int, default=1)
p.add_argument("--multihost", action="store_true")
ARGS = p.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

if ARGS.multihost or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize()

AXIS = "x"
LEAD = jax.process_index() == 0


def rp(*a):
    if LEAD:
        print(*a, flush=True)


import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "llar",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
    "/../src/vllm_torchtpu/kernels/collectives/low_latency_all_reduce.py")
llar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llar)


def main():
    n = ARGS.devices
    wa = ARGS.width
    wb = ARGS.width_b if ARGS.width_b is not None else ARGS.width
    dtype = jnp.float32
    mesh = Mesh(np.asarray(sorted(jax.devices(), key=lambda d: d.id)[:n]),
                axis_names=(AXIS, ))
    shard = NamedSharding(mesh, P(AXIS, None, None))

    a = jax.jit(lambda: jnp.ones((n, 1, wa), dtype), out_shardings=shard)()
    b = jax.jit(lambda: jnp.full((n, 1, wb), 2.0, dtype),
                out_shardings=shard)()

    def body(la, lb):
        x, y = la[0], lb[0]
        rsa, rsb, aga, agb, step = llar.make_two_shot_variadic_workspace(
            wa, wb, dtype, n)
        for _ in range(ARGS.rounds):
            x, y, rsa, rsb, aga, agb, step = (
                llar.two_shot_all_reduce_variadic_ws(x,
                                                     y,
                                                     rsa,
                                                     rsb,
                                                     aga,
                                                     agb,
                                                     step,
                                                     axis_name=AXIS,
                                                     num_devices=n))
        # Per-chunk min/max for this device, so a partial crossover shows up
        # as some chunks wrong rather than being averaged away. Shaped
        # (n_chunks, 2) per device and gathered tiled -- a replicated out_spec
        # is not fully addressable under multi-process, which is what the first
        # version of this probe got wrong.
        xr = x.reshape(n, -1)
        yr = y.reshape(n, -1)
        return (jnp.stack([xr.min(axis=1), xr.max(axis=1)], axis=1),
                jnp.stack([yr.min(axis=1), yr.max(axis=1)], axis=1))

    fn = jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS, None, None), P(AXIS, None, None)),
                      out_specs=(P(AXIS, None), P(AXIS, None)),
                      check_vma=False))
    ra, rb = jax.block_until_ready(fn(a, b))
    ra = np.asarray(multihost_utils.process_allgather(ra, tiled=True))
    rb = np.asarray(multihost_utils.process_allgather(rb, tiled=True))
    # Every device holds the whole reduced result, so device 0's view suffices.
    ra = ra.reshape(n, n, 2)[0]
    rb = rb.reshape(n, n, 2)[0]

    rp(f"== {n} devices, a width {wa} (chunk {wa//n}), "
       f"b width {wb} (chunk {wb//n}), rounds={ARGS.rounds}")
    rp(f"== correct: a = {float(n)}  b = {float(2*n)}")
    rp(f"   a distinct values: {np.unique(np.round(ra, 3))}")
    rp(f"   b distinct values: {np.unique(np.round(rb, 3))}")
    bad_a = np.where(np.abs(ra[:, 1] - n) > 1e-3)[0]
    bad_b = np.where(np.abs(rb[:, 1] - 2 * n) > 1e-3)[0]
    rp(f"   chunks wrong in a: {len(bad_a)}/{n}  {bad_a[:12]}")
    rp(f"   chunks wrong in b: {len(bad_b)}/{n}  {bad_b[:12]}")


if __name__ == "__main__":
    main()
