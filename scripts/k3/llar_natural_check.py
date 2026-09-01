#!/usr/bin/env python3
"""On-device semantics check for the stateless two-shot all-reduce.

The production wiring (distributed/low_latency_all_reduce.py) relies on the
kernel being STATELESS: one workspace slot, a full barrier at the start of
every call, and workspaces whose contents are per-call scratch (every write
to them is consumed in the round that received it, and a peer's writes for
call k+1 start only after everyone enters call k+1 -- the entry barrier --
by which point our call k has finished reading). The barrier placement also
orders any per-call operand restaging: staging completes before the kernel
starts, and the kernel's barrier comes before any peer may send.

What this verifies, at the JAX level, without any torch:

1. Chained calls inside ONE compiled program (the production regime: 93
   back-to-back calls per step) match a psum chain, three repetitions.
2. Independent calls on the same buffers (separate executions) also match,
   with the workspace values never threaded.
3. With XLA_FLAGS=--xla_dump_to set, the compiled chain module lands on disk
   so the staging can be audited: count the copy ops per custom-call and
   confirm the reshapes around the custom-call do not materialize relayout
   copies beyond the (unavoidable) HBM->VMEM staging of the payload itself.

Run under srun on a 4-host ici-2x2x4 slice (32 devices), like the bench.
"""

import os
import sys

if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402


def _load_kernels():
    """Load the kernel module straight from this checkout (see the bench)."""
    import importlib.util
    path = (os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +
            "/src/vllm_torchtpu/kernels/collectives/low_latency_all_reduce.py")
    spec = importlib.util.spec_from_file_location("low_latency_all_reduce",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


llar = _load_kernels()  # noqa: E402

AXIS = "x"
IS_LEAD = jax.process_index() == 0


def rprint(*a, **kw):
    if IS_LEAD:
        print(*a, **kw, flush=True)


def main() -> int:
    num_devices = len(jax.devices())
    tokens = int(os.environ.get("LLAR_CHECK_TOKENS", "4"))
    width = int(os.environ.get("LLAR_CHECK_WIDTH", "7168"))
    n_chain = int(os.environ.get("LLAR_CHECK_CHAIN", "20"))
    n_calls = int(os.environ.get("LLAR_CHECK_CALLS", "5"))
    dtype = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32
    }[os.environ.get("LLAR_CHECK_DTYPE", "bfloat16")]
    devices = mesh_utils.create_device_mesh((num_devices, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, (AXIS, ))
    spec = P(AXIS, None, None)

    w = (tokens * width) // num_devices
    per_row = width // w
    assert w % 128 == 0 and width % w == 0, f"unaligned chunk w={w}"

    def body(local_x, rs, ag):
        v = local_x[0]
        # A one-call execution: always the barrier-first variant.
        out = llar.two_shot_all_reduce_static(v.reshape(tokens, per_row, 1, w),
                                              rs,
                                              ag,
                                              slot=0,
                                              barrier_first=True,
                                              axis_name=AXIS,
                                              num_devices=num_devices,
                                              collective_id=8,
                                              has_side_effects=False)
        return out.reshape(v.shape)[None]

    fn = jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(spec, P(), P()),
                      out_specs=spec,
                      check_vma=False))

    def chain_body(local_x, rs, ag):
        v = local_x[0]
        scale = 1.0 / num_devices
        # Production regime: slot = call index % 2, barrier-first on the
        # first call of each execution.
        for i in range(n_chain):
            out = llar.two_shot_all_reduce_static(v.reshape(
                tokens, per_row, 1, w),
                                                  rs,
                                                  ag,
                                                  slot=i % 2,
                                                  barrier_first=(i == 0),
                                                  axis_name=AXIS,
                                                  num_devices=num_devices,
                                                  collective_id=8,
                                                  has_side_effects=False)
            v = out.reshape(v.shape) * scale
        return v[None]

    chain = jax.jit(
        jax.shard_map(chain_body,
                      mesh=mesh,
                      in_specs=(spec, P(), P()),
                      out_specs=spec,
                      check_vma=False))

    def ref_chain_body(local_x):
        v = local_x[0]
        scale = 1.0 / num_devices
        for _ in range(n_chain):
            v = jax.lax.psum(v, AXIS) * scale
        return v[None]

    ref_chain = jax.jit(
        jax.shard_map(ref_chain_body,
                      mesh=mesh,
                      in_specs=spec,
                      out_specs=spec,
                      check_vma=False))

    x_dev = jax.jit(
        lambda: jax.random.normal(jax.random.key(0),
                                  (num_devices, tokens, width), dtype),
        out_shardings=NamedSharding(mesh, spec))()
    ws = tuple(
        jax.device_put(a, NamedSharding(mesh, P()))
        for a in llar.make_two_shot_workspace_static((
            tokens, width), dtype, num_devices))

    rel_fn = jax.jit(
        lambda a, b:
        (jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))) / jnp.
         maximum(jnp.max(jnp.abs(b.astype(jnp.float32))), 1e-6)),
        out_shardings=NamedSharding(mesh, P()))
    tol = 1e-3 if dtype == jnp.float32 else 0.05

    ok = True

    # 1. The chained regime (production): n_chain kernel calls in one program.
    want = ref_chain(x_dev)
    for rep in range(3):
        got = chain(x_dev, *ws)
        rel = float(rel_fn(got, want))
        rprint(f"chain rep {rep}: rel err vs psum chain = {rel:.3g}")
        if rel > tol:
            rprint("FAIL: chained numerics diverged")
            ok = False

    # 2. The per-execution regime: independent calls, nothing threaded.
    want1 = jax.jit(
        jax.shard_map(lambda lx: jax.lax.psum(lx, AXIS),
                      mesh=mesh,
                      in_specs=spec,
                      out_specs=spec,
                      check_vma=False))(x_dev)
    for call in range(n_calls):
        got = fn(x_dev, *ws)
        rel = float(rel_fn(got, want1))
        rprint(f"single-call {call}: rel err vs psum = {rel:.3g}")
        if rel > tol:
            rprint("FAIL: single-call numerics diverged")
            ok = False

    rprint("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
