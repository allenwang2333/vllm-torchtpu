#!/usr/bin/env python3
"""On-device check of the REAL combined sequence: fused KDA core with the
o_proj weight prefetch -> fused o_proj+AR kernel (T=1, K3 TP=32 shapes).

This is the microbenchmark for the combined layer: per rank, the fused
KDA core runs at the K3 decode shapes (n_v=3, d_k=128) with the o_proj
weight shard [384,7168] prefetched under its compute; the fused o_proj+AR
op consumes the VMEM wv directly. Every call is checked against the
in-body psum(dot) reference (like the packed-layout probe), so slot
rotation + the producer->consumer VMEM chaining are verified per call over
8 alternating-slot calls x 2 executions.

Env knobs: VLLM_TORCHTPU_GDN_WPF_HBM_MARK=0 drops the HBM mark on w_o
(A/B for the wv memory-space color issue; the HLO dump carries the wv
memory-space evidence); CHAIN_PROFILE=<dir> captures an xprof trace.
"""

import os
import sys
import time

# These slices boot libtpu in multislice mode: a JAX process that does not
# join the distributed runtime hangs in backend init waiting for the other
# hosts. Join on every task.
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
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper

WIDTH, K_SHARD = 7168, 384  # K3 KDA o_proj per-rank shapes at TP=32
NUM_DEVICES = 32
CALLS = 8


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _kda_wpf(w, prefill_state):
    """The production fused KDA core at T=1 with the o_proj prefetch riding
    it. Returns (attention_out [1, 384], wv [384, 7168])."""
    (_, _), out, wv = gdn_wrapper.fused_conv1d_gdn._fun(w_o_prefetch=w,
                                                        **prefill_state)
    return out.reshape(1, -1), wv


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True

    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    # Fixed KDA inputs, identical on every rank (the fused op's contract is
    # what is under test; KDA numerics are covered by the gdn v3 suite).
    n_v, d_k = 3, 128
    dim_size = 3 * n_v * d_k
    rngs = iter(jax.random.split(jax.random.key(0), 6))
    prefill_state = dict(
        qkv=jax.random.normal(next(rngs), (1, dim_size), jnp.bfloat16),
        b=jax.random.normal(next(rngs), (1, n_v), jnp.bfloat16),
        a=jax.random.normal(next(rngs), (1, n_v * d_k), jnp.bfloat16),
        conv_state=jnp.zeros((2, 3, dim_size), jnp.bfloat16),
        recurrent_state=jnp.zeros((2, n_v, d_k, d_k), jnp.float32),
        conv_weight=jax.random.normal(next(rngs), (dim_size, 1, 4),
                                      jnp.bfloat16),
        conv_bias=None,
        a_log=jax.random.normal(next(rngs), (n_v, ), jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (n_v, d_k), jnp.float32),
        query_start_loc=jnp.array([0, 1], jnp.int32),
        state_indices=jnp.array([1], jnp.int32),
        distribution=jnp.array([1, 1, 1], jnp.int32),
        seq_lens=jnp.array([5], jnp.int32),
        n_kq=n_v,
        n_v=n_v,
        d_k=d_k,
        d_v=d_k,
        kernel_size=4,
        attention_mode=gdn_config.AttentionMode.KDA,
        gate_lower_bound=-5.0,
    )

    def fn(w, rs, ag):

        def body(wl, rs, ag):
            errs = []
            for k in range(CALLS):
                wk = wl + jnp.asarray(k, wl.dtype)
                kda_out, wv = _kda_wpf(wk, prefill_state)
                xo = kda_out + jnp.asarray(k, kda_out.dtype)
                out = fused_out_proj_all_reduce(
                    xo,
                    wv,
                    rs,
                    ag,
                    slot=k % 2,
                    barrier_first=1 if k == 0 else 0,
                    axis_name="tp",
                    num_devices=NUM_DEVICES,
                    has_side_effects=False)
                # Reference: the same math, psum'd, in fp32 (kda_out is
                # identical on every rank by construction).
                ref = lax.psum(
                    jax.lax.dot_general(xo.astype(jnp.float32),
                                        wk.astype(jnp.float32),
                                        dimension_numbers=(((1, ), (0, )),
                                                           ((), ()))), "tp")
                scale = jnp.max(jnp.abs(ref))
                err = jnp.max(
                    jnp.abs(out.reshape(1, -1).astype(jnp.float32) - ref))
                errs.append(lax.pmax(err / jnp.maximum(scale, 1e-6), "tp"))
            return errs

        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(P("tp", None), P(), P()),
                             out_specs=[P()] * CALLS,
                             check_vma=False)(w, rs, ag)

    # w: the global [12288, 7168] weight, contracting-dim sharded per rank.
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1), (
        K_SHARD * NUM_DEVICES, WIDTH), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("tp", None)))()
    rs, ag = jax.jit(lambda: make_fused_out_proj_workspace(
        WIDTH, jnp.bfloat16, NUM_DEVICES),
                     out_shardings=(NamedSharding(mesh, P()),
                                    NamedSharding(mesh, P())))()

    fn_jit = jax.jit(fn)
    t0 = time.time()
    compiled = fn_jit.lower(w, rs, ag).compile()
    _stamp(rank, f"lowered+compiled in {time.time() - t0:.1f}s")

    for run in (0, 1):
        errs = compiled(w, rs, ag)
        jax.block_until_ready(errs)
        for k in range(CALLS):
            rel = float(errs[k])
            _stamp(
                rank, f"run {run} call {k} (slot {k % 2}, "
                f"barrier_first={1 if k == 0 else 0}): rel err = {rel:.3g}")
            ok = ok and rel < 0.05

    t0 = time.time()
    for _ in range(50):
        errs = compiled(w, rs, ag)
    jax.block_until_ready(errs)
    _stamp(
        rank, f"{CALLS} chained KDA+fused calls: "
        f"{(time.time() - t0) / 50 / CALLS * 1e6:.2f} us/call "
        "(host dispatch included)")

    prof_root = os.environ.get("CHAIN_PROFILE")
    if prof_root:
        from jax import profiler
        tdir = os.path.join(prof_root, f"rank{rank}")
        profiler.start_trace(tdir)
        for _ in range(30):
            errs = compiled(w, rs, ag)
        jax.block_until_ready(errs)
        profiler.stop_trace()
        _stamp(rank, f"trace captured -> {tdir}")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
