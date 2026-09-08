#!/usr/bin/env python3
"""Bisect the barrier_first score corruption (job 791/792/794).

The shipped kernel's tailscore epilogue computes score = sum(hidden*fw)*inv
correctly on every call EXCEPT when barrier_first=1 (deterministic ~4.3x
wrong dot while hidden and inv_rms are exact). This script isolates what
the barrier variant corrupts: variant "fwsum" writes sum(fw) (no hidden
involved) as the score; variant "dotsum" writes the raw dot (no inv);
variant "plain" is the shipped epilogue. One call per config, values
printed for rank 0.
"""

import os
import sys
import time

if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.fused_out_proj_all_reduce import (
    LANES, PACK, fused_out_proj_all_reduce)

HID = 7168
NUM_DEVICES = 32
TILES = HID // (PACK * LANES)
EPS = 1e-5


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _full_barrier(my_id, num_devices):
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(1, num_devices):
        pl.semaphore_signal(barrier_sem,
                            1,
                            device_id=(lax.rem(my_id + i, num_devices), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, num_devices - 1)

    @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
    def _(sem):
        for i in range(1, num_devices):
            pl.semaphore_signal(sem,
                                1,
                                device_id=(lax.rem(my_id + i, num_devices), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(sem, num_devices - 1)


def _fwsum_kernel(fw_ref, o_ref, *, num_devices, axis_name, do_barrier):
    """Writes sum(fw) -- probes whether the entry barrier corrupts an input.
    """
    my_id = lax.axis_index(axis_name)
    if do_barrier:
        _full_barrier(my_id, num_devices)
    o_ref[...] = jnp.broadcast_to(jnp.sum(fw_ref[...]), (1, ))


def fwsum(fw, *, do_barrier, num_devices, axis_name):
    (out, ) = pl.pallas_call(
        functools.partial(_fwsum_kernel,
                          num_devices=num_devices,
                          axis_name=axis_name,
                          do_barrier=do_barrier),
        out_shape=(jax.ShapeDtypeStruct((1, ), jnp.float32), ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        compiler_params=pltpu.CompilerParams(
            collective_id=8 if do_barrier else None,
            has_side_effects=False,
            disable_semaphore_checks=True),
    )(fw)
    return out


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))

    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    def fn(x, w, resid, fw, rs, ag):

        def body(xl, wl, residl, fwl, rs, ag):
            return fused_out_proj_all_reduce(xl,
                                             wl,
                                             residl.reshape(
                                                 TILES, PACK, LANES),
                                             fwl.reshape(TILES, PACK, LANES),
                                             rs,
                                             ag,
                                             slot=0,
                                             barrier_first=1,
                                             axis_name="tp",
                                             num_devices=NUM_DEVICES,
                                             eps=EPS,
                                             collective_id=8,
                                             has_side_effects=False)

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(P(None,
                                         "tp"), P("tp",
                                                  None), P(), P(), P(), P()),
                             out_specs=(P(), P(), P()),
                             check_vma=False)(x, w, resid, fw, rs, ag)

    x = jax.jit(lambda: jax.random.normal(jax.random.key(3),
                                          (1, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P(None, "tp")))()
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1),
                                          (HID, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("tp", None)))()
    resid = jax.jit(lambda: jax.random.normal(jax.random.key(5),
                                              (1, HID), jnp.bfloat16),
                    out_shardings=NamedSharding(mesh, P()))()
    fw = jax.jit(lambda: jax.random.normal(jax.random.key(7),
                                           (HID, ), jnp.float32),
                 out_shardings=NamedSharding(mesh, P()))()
    rs = jax.jit(lambda: jnp.zeros(
        (2, NUM_DEVICES, PACK, LANES), jnp.bfloat16),
                 out_shardings=NamedSharding(mesh, P()))()
    ag = jax.jit(lambda: jnp.zeros((2, TILES + 1, PACK, LANES), jnp.bfloat16),
                 out_shardings=NamedSharding(mesh, P()))()

    # Trusted references, computed by plain XLA on the same device values.
    ref = jax.jit(lambda x, w, resid, fw: (
        jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32)) + resid.astype(
            jnp.float32),
        jnp.sum(fw),
    ),
                  out_shardings=NamedSharding(mesh, P()))
    hid_ref, fw_sum_ref = ref(x, w, resid, fw)
    fw_sum_ref = float(fw_sum_ref)
    inv_ref = float(lax.rsqrt(jnp.sum(hid_ref * hid_ref) / HID + EPS))
    score_ref = float(jnp.sum(hid_ref * fw) * inv_ref)

    compiled = jax.jit(fn).lower(x, w, resid, fw, rs, ag).compile()
    hid, inv, score = compiled(x, w, resid, fw, rs, ag)
    jax.block_until_ready((hid, inv, score))

    hid_err = float(
        jnp.max(jnp.abs(hid.astype(jnp.float32) - hid_ref)) /
        jnp.max(jnp.abs(hid_ref)))
    _stamp(
        rank, f"hid rel err {hid_err:.3g} | inv {float(inv[0]):.6g} vs "
        f"{inv_ref:.6g} | score {float(score[0]):.6g} vs {score_ref:.6g} "
        f"| fw_sum ref {fw_sum_ref:.6g}")

    # fw integrity probes: sum(fw) through a bare kernel, with and without
    # the entry barrier.
    def fw_fn(fw, do_barrier):

        def body(fwl):
            return fwsum(fwl.reshape(TILES, PACK, LANES),
                         do_barrier=do_barrier,
                         num_devices=NUM_DEVICES,
                         axis_name="tp")

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(P(), ),
                             out_specs=P(),
                             check_vma=False)(fw)

    for do_barrier in (False, True):
        got = jax.jit(functools.partial(fw_fn, do_barrier=do_barrier))(fw)
        _stamp(
            rank, f"fwsum barrier={do_barrier}: {float(got[0]):.6g} vs "
            f"ref {fw_sum_ref:.6g}")
    _stamp(rank, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
