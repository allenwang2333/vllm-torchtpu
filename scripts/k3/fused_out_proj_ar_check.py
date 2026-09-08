#!/usr/bin/env python3
"""On-device check for the SHIPPED fused o_proj+all-reduce kernel (T=1).

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

Since the tail-epilogue fusion (bench jobs 787/790) the kernel also takes
the replicated residual stream and the f32 folded AttentionResidual scorer
weight (both packed [tiles, 2, 128] at the callsite) and returns
(hidden = AR + resid, inv_rms, score); all three are checked against the
global fp32 reference per call.

x is the [1, 7168] activation sharded along the contracting dim (each rank
gets [1, 224]); w is the [7168, 7168] weight sharded the same way. Run
under srun on a 4-host ici-2x2x4 slice; see fused_out_proj_ar_check.sbatch.
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
    LANES, PACK, fused_out_proj_all_reduce, make_fused_out_proj_workspace)

HID = 7168  # T=1 payload: [1, 7168] over 32 devices -> 224-wide shards
NUM_DEVICES = 32
CALLS = 8
TILES = HID // (PACK * LANES)
EPS = 1e-5


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True

    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    def fn(x, w, resid, fw, rs, ag):

        def body(xl, wl, residl, fwl, rs, ag):
            outs = []
            for k in range(CALLS):
                # Perturb per call so a stale-slot read is a wrong number.
                xk = xl + jnp.asarray(k, xl.dtype)
                out = fused_out_proj_all_reduce(
                    xk,
                    wl,
                    residl.reshape(TILES, PACK, LANES),
                    fwl.reshape(TILES, PACK, LANES),
                    rs,
                    ag,
                    slot=k % 2,
                    barrier_first=1 if k == 0 else 0,
                    axis_name="tp",
                    num_devices=NUM_DEVICES,
                    eps=EPS,
                    collective_id=8,
                    has_side_effects=False)
                outs.append(out)
            return outs

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(P(None,
                                         "tp"), P("tp",
                                                  None), P(), P(), P(), P()),
                             out_specs=[(P(), P(), P())] * CALLS,
                             check_vma=False)(x, w, resid, fw, rs, ag)

    x = jax.jit(lambda: jax.random.normal(jax.random.key(3),
                                          (1, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P(None, "tp")))()
    # w is the global [HID, HID] weight with the CONTRACTING dim sharded
    # across ranks (each rank holds [HID/32, HID]) -- the torch_tpu
    # RowParallelLinear layout, [k_shard, width].
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1),
                                          (HID, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("tp", None)))()
    resid = jax.jit(lambda: jax.random.normal(jax.random.key(5),
                                              (1, HID), jnp.bfloat16),
                    out_shardings=NamedSharding(mesh, P()))()
    fw = jax.jit(lambda: jax.random.normal(jax.random.key(7),
                                           (HID, ), jnp.float32),
                 out_shardings=NamedSharding(mesh, P()))()
    rs, ag = jax.jit(
        lambda: make_fused_out_proj_workspace(HID, jnp.bfloat16, NUM_DEVICES),
        out_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())))()

    fn_jit = jax.jit(fn)
    t0 = time.time()
    compiled = fn_jit.lower(x, w, resid, fw, rs, ag).compile()
    _stamp(rank, f"lowered+compiled in {time.time() - t0:.1f}s")

    # Reference: the global fp32 matmul per call, (x + k) @ w.T, plus the
    # epilogue stats, computed ON DEVICE (the sharded global arrays are not
    # host-addressable on a multislice). XLA inserts the cross-shard
    # reduction itself.
    wants_hid = [
        jax.jit(lambda x, w, resid, k=k: jnp.dot(
            (x + jnp.asarray(k, x.dtype)).astype(jnp.float32),
            w.astype(jnp.float32)) + resid.astype(jnp.float32),
                out_shardings=NamedSharding(mesh, P()))(x, w, resid)
        for k in range(CALLS)
    ]
    wants_inv = [lax.rsqrt(jnp.sum(h * h) / HID + EPS) for h in wants_hid]
    wants_score = [
        jnp.sum(h * fw) * inv for h, inv in zip(wants_hid, wants_inv)
    ]

    def check_body(out, want):
        err = jnp.max(jnp.abs(out.astype(jnp.float32) - want))
        scale = jnp.max(jnp.abs(want))
        return lax.pmax(err / jnp.maximum(scale, 1e-6), "tp")

    def check_scalar_body(out, want):
        return lax.pmax(
            jnp.abs(out[0] - want) / jnp.maximum(jnp.abs(want), 1e-12), "tp")

    def check_score_body(out, want, scale):
        # score = sum(hidden * fw) * inv is a cancellation-heavy dot:
        # per-element bf16 noise in hidden (~0.4% of ~85-magnitude terms)
        # gives an absolute noise floor ~O(1), so measure against
        # |want| + floor where the floor is 1% of the dot's magnitude scale.
        return lax.pmax(
            jnp.abs(out[0] - want) / (jnp.abs(want) + 0.01 * scale), "tp")

    checker = jax.jit(
        jax.shard_map(check_body,
                      mesh=mesh,
                      in_specs=(P(), P()),
                      out_specs=P(),
                      check_vma=False))
    scalar_checker = jax.jit(
        jax.shard_map(check_scalar_body,
                      mesh=mesh,
                      in_specs=(P(), P()),
                      out_specs=P(),
                      check_vma=False))
    score_checker = jax.jit(
        jax.shard_map(check_score_body,
                      mesh=mesh,
                      in_specs=(P(), P(), P()),
                      out_specs=P(),
                      check_vma=False))
    # Per-call dot magnitude scale: sqrt(sum((hid*fw)^2)).
    wants_scale = [jnp.sqrt(jnp.sum((h * fw)**2)) for h in wants_hid]

    for run in (0, 1):  # second run exercises the execution boundary
        outs = compiled(x, w, resid, fw, rs, ag)
        jax.block_until_ready(outs)
        for k in range(CALLS):
            hid_out, inv_out, score_out = outs[k]
            rel = float(checker(hid_out, wants_hid[k]))
            rel_inv = float(scalar_checker(inv_out, wants_inv[k]))
            rel_score = float(
                score_checker(score_out, wants_score[k], wants_scale[k]))
            worst = max(rel, rel_inv, rel_score)
            _stamp(
                rank, f"run {run} call {k} (slot {k % 2}, "
                f"barrier_first={1 if k == 0 else 0}): rel err = {rel:.3g} "
                f"(inv {rel_inv:.3g}, score {rel_score:.3g})")
            ok = ok and worst < 0.05

    # Steady-state timing, host dispatch included.
    t0 = time.time()
    for _ in range(100):
        outs = compiled(x, w, resid, fw, rs, ag)
    jax.block_until_ready(outs)
    _stamp(
        rank,
        f"{CALLS} fused calls: {(time.time() - t0) / 100 / CALLS * 1e6:.2f} "
        "us/call (host dispatch included)")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
