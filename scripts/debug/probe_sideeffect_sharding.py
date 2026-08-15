#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Which property of the collective Pallas call does XLA choke on?

Compiling a shard_map'd collective Pallas kernel through torch_tpu fails with

    RET_CHECK failure (spmd_partitioner.cc:6496) hlo->has_sharding()
    Side-effect HLO must have sharding: %shard_map = ... custom-call(...)
    custom_call_has_side_effect=true

The module leaves JAX in Shardy form (`sdy.manual_computation`), and torch_tpu's
own specialization stage requires Shardy -- a GSPMD-lowered module is rejected
with "lowered targeting GSPMD, but Shardy was then enabled in XLA". So the
module must be Shardy, and the failure happens later, when XLA converts that
region for the GSPMD partitioner and the side-effecting custom call comes out
with no sharding.

This isolates the trigger by compiling three variants of the *same* one-shot
kernel and reporting which survive:

  A. as shipped -- has_side_effects=True
  B. has_side_effects=False -- tests whether that flag alone is the trigger
  C. as shipped, plus a sharding constraint on the outputs inside the body

Only compilation matters here, so a variant that compiles is then checked for
correctness against funcol before being believed.

    torchrun --nproc_per_node=8 scripts/debug/probe_sideeffect_sharding.py
"""

import functools
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401
from torch.distributed._functional_collectives import \
    all_reduce as funcol_all_reduce


def _build_one_shot(has_side_effects: bool):
    """A copy of one_shot_all_reduce with the side-effect flag exposed."""
    import jax
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
        _full_barrier

    def kernel(x_ref, o_ref, recv_ref, send_sem, recv_sem, *, num_devices,
               axis_name):
        my_id = lax.axis_index(axis_name)
        _full_barrier(my_id, num_devices, True)
        copies = []
        for i in range(1, num_devices):
            copy = pltpu.make_async_remote_copy(
                src_ref=x_ref,
                dst_ref=recv_ref.at[i - 1],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        acc = x_ref[...]
        for i in range(num_devices - 1):
            acc = acc + recv_ref[i]
        o_ref[...] = acc

    def call(x, *, axis_name, num_devices, collective_id):
        k = functools.partial(kernel,
                              num_devices=num_devices,
                              axis_name=axis_name)
        return pl.pallas_call(
            k,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((num_devices - 1, *x.shape), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                has_side_effects=has_side_effects),
        )(x)

    return call


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.distributed import tpu_distributed

    from vllm_torchtpu.kernels.collectives.sharded_jax_op import sharded_jax_op

    my_device_id = int(tpu_distributed.global_device_id())
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, (rank, my_device_id), group=gloo)
    order = [0] * world_size
    for r, d in gathered:
        order[int(r)] = int(d)
    by_id = {int(getattr(d, "id", i)): d for i, d in enumerate(jax.devices())}
    mesh = Mesh(np.asarray([by_id[d] for d in order]), axis_names=("tp", ))

    device = torch.device("tpu")
    rows, cols = 4, 1024
    n = world_size
    R = P()

    torch.manual_seed(7 + rank)
    x = (torch.randn(rows, cols,
                     dtype=torch.float32).to(torch.bfloat16).to(device))
    want = funcol_all_reduce(x, "sum", list(range(world_size))).cpu()

    def make(variant: str, cid: int):
        side_effects = variant != "B_no_side_effect"
        call = _build_one_shot(side_effects)

        # Returns a 1-tuple so the jit's out_shardings pytree (also a 1-tuple)
        # matches the output structure.
        def _fn(x: jax.Array) -> tuple[jax.Array]:

            def body(x):
                out = call(x, axis_name="tp", num_devices=n, collective_id=cid)
                if variant == "C_sharding_constraint":
                    out = jax.lax.with_sharding_constraint(
                        out, NamedSharding(mesh, R))
                return out

            return (jax.shard_map(body,
                                  mesh=mesh,
                                  in_specs=(R, ),
                                  out_specs=R,
                                  check_vma=False)(x), )

        return sharded_jax_op(f"probe::se_{variant}",
                              _fn,
                              mesh=mesh,
                              input_partition_specs=(R, ),
                              output_partition_specs=(R, ))

    outcomes = {}
    for i, variant in enumerate(
        ["A_as_shipped", "B_no_side_effect", "C_sharding_constraint"]):
        try:
            op = make(variant, 20 + i)
            got = op(x)[0].cpu()
            err = (got.float() - want.float()).abs().max().item()
            outcomes[variant] = ("compiled", err)
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")[:160]
            outcomes[variant] = ("failed", msg)
            if rank == 0:
                print(f"\n--- {variant} ---\n{traceback.format_exc()[-700:]}",
                      flush=True)

    if rank == 0:
        print("\n=== variant outcomes (rank 0) ===", flush=True)
        for variant, (state, detail) in outcomes.items():
            if state == "compiled":
                ok = "OK" if detail < 0.05 else "WRONG VALUES"
                print(
                    f"  {variant:>24}: compiled, max|diff|={detail:.5f} {ok}",
                    flush=True)
            else:
                print(f"  {variant:>24}: FAILED  {detail}", flush=True)

    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
