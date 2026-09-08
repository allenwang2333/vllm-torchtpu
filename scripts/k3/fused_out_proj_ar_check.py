#!/usr/bin/env python3
"""On-device check for the SHIPPED fused o_proj+all-reduce kernel.

Exercises kernels/collectives/fused_out_proj_all_reduce.py exactly as the
distributed wrapper drives it: CALLS sequential invocations through a
shard_map over the full 32-device slice, static slot alternating per call
and barrier_first on the first call. Two things are under test that the
single-shot probe never covered:

* slot rotation: call k writes workspace slot k%2 while a skewed peer may
  still be reading slot (k-1)%2 -- inputs are perturbed per call (x + k),
  so a rotation race produces a wrong number rather than silently reusing
  a stale slot;
* the execution boundary: the jitted function is run TWICE, and its first
  call carries barrier_first=1, which is what makes the second run's slot-0
  writes safe against a peer still finishing the first run.

x is the [1, 7168] activation sharded along the contracting dim (each rank
gets [1, 224]); w is the [7168, 7168] weight sharded the same way. Verified
against the global fp32 matmul per call. Run under srun on a 4-host
ici-2x2x4 slice; see fused_out_proj_ar_check.sbatch.
"""

import os
import sys
import time

# These slices boot libtpu in multislice mode: a JAX process that does not
# join the distributed runtime hangs in backend init waiting for the other
# hosts ("Did you run your code on all TPU hosts?"). Join on every task.
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.fused_out_proj_all_reduce import (
    fused_out_proj_all_reduce, make_fused_out_proj_workspace)

HID = 7168  # T=1 payload: [1, 7168] over 32 devices -> 224-wide shards
NUM_DEVICES = 32
CALLS = 8


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True

    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    def fn(x, w, rs, ag):

        def body(xl, wl, rs, ag):
            outs = []
            for k in range(CALLS):
                # Perturb per call so a stale-slot read is a wrong number.
                xk = xl + jnp.asarray(k, xl.dtype)
                out = fused_out_proj_all_reduce(
                    xk,
                    wl,
                    rs,
                    ag,
                    slot=k % 2,
                    barrier_first=1 if k == 0 else 0,
                    axis_name="tp",
                    num_devices=NUM_DEVICES,
                    has_side_effects=False)
                outs.append(out.reshape(1, HID))
            return outs

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(P(None, "tp"), P("tp", None), P(), P()),
                             out_specs=[P()] * CALLS,
                             check_vma=False)(x, w, rs, ag)

    x = jax.jit(lambda: jax.random.normal(jax.random.key(3),
                                          (1, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P(None, "tp")))()
    # w is the global [HID, HID] weight with the CONTRACTING dim sharded
    # across ranks (each rank holds [HID/32, HID]) -- the torch_tpu
    # RowParallelLinear layout, [k_shard, width].
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1),
                                          (HID, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("tp", None)))()
    rs, ag = jax.jit(
        lambda: make_fused_out_proj_workspace(HID, jnp.bfloat16, NUM_DEVICES),
        out_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())))()

    fn_jit = jax.jit(fn)
    t0 = time.time()
    compiled = fn_jit.lower(x, w, rs, ag).compile()
    _stamp(rank, f"lowered+compiled in {time.time() - t0:.1f}s")

    # Reference: the global fp32 matmul per call, (x + k) @ w.T, computed ON
    # DEVICE (the sharded global arrays are not host-addressable on a
    # multislice). XLA inserts the cross-shard reduction itself.
    wants = [
        jax.jit(lambda x, w, k=k: jnp.dot(
            (x + jnp.asarray(k, x.dtype)).astype(jnp.float32),
            w.astype(jnp.float32)),
                out_shardings=NamedSharding(mesh, P()))(x, w)
        for k in range(CALLS)
    ]

    def check_body(out, want):
        err = jnp.max(jnp.abs(out.astype(jnp.float32) - want))
        scale = jnp.max(jnp.abs(want))
        return lax.pmax(err / jnp.maximum(scale, 1e-6), "tp")

    checker = jax.jit(
        jax.shard_map(check_body,
                      mesh=mesh,
                      in_specs=(P(), P()),
                      out_specs=P(),
                      check_vma=False))

    for run in (0, 1):  # second run exercises the execution boundary
        outs = compiled(x, w, rs, ag)
        jax.block_until_ready(outs)
        for k in range(CALLS):
            rel = float(checker(outs[k], wants[k]))
            _stamp(
                rank, f"run {run} call {k} (slot {k % 2}, "
                f"barrier_first={1 if k == 0 else 0}): rel err = {rel:.3g}")
            ok = ok and rel < 0.05

    # Steady-state timing, host dispatch included.
    t0 = time.time()
    for _ in range(100):
        outs = compiled(x, w, rs, ag)
    jax.block_until_ready(outs)
    _stamp(
        rank,
        f"{CALLS} fused calls: {(time.time() - t0) / 100 / CALLS * 1e6:.2f} "
        "us/call (host dispatch included)")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
