#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Can a cross-device Pallas collective run from vLLM's process layout?

vLLM-torchtpu runs one process per TPU core and does its TP collectives through
``torch.distributed`` (``tpu_dist``), which lowers to XLA cross-process
collectives.  The low-latency all-reduce kernels in
``vllm_torchtpu.kernels.collectives.low_latency_all_reduce`` are JAX Pallas and
must run inside a ``shard_map`` over a mesh that spans every peer, addressing
them with ``DeviceIdType.MESH``.

Whether those two things can meet is the single assumption the whole
integration rests on, so this probe answers it before any of it is built:

  1. What does JAX see in such a process -- all devices in the slice, or only
     the one this rank owns?  (``get_or_create_pcp_mesh`` in
     ``vllm_torchtpu/distributed/pcp.py`` assumes the former.)
  2. Can a Pallas collective be registered through ``pallas.jax_op(mesh=...,
     input_partition_specs=...)`` and called on torch tensors?
  3. Does it produce the same values as ``funcol.all_reduce``?

Run it exactly the way the torchtpu microbenchmark runs (single host, 8 cores):

    eval "$(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
    torchrun --nproc_per_node=8 scripts/debug/probe_pallas_collective.py

Prints one summary block from rank 0 and exits nonzero if the bridge does not
work, so it can gate the rest of the work.
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device and backends
from torch.distributed._functional_collectives import \
    all_reduce as funcol_all_reduce


def log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    # Object gather needs a CPU-capable group; tpu_dist cannot carry pickles.
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    import jax
    from torch_tpu._internal.distributed import tpu_distributed

    global_device_id = int(tpu_distributed.global_device_id())
    jax_devices = jax.devices()
    jax_local = jax.local_devices()
    visible_ids = sorted(int(getattr(d, "id", -1)) for d in jax_devices)

    facts = {
        "rank": rank,
        "global_device_id": global_device_id,
        "jax_process_index": int(jax.process_index()),
        "jax_process_count": int(jax.process_count()),
        "jax_device_count": len(jax_devices),
        "jax_local_device_count": len(jax_local),
        "jax_visible_ids": visible_ids,
    }
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, facts, group=gloo)

    if rank == 0:
        print("\n=== what JAX sees inside a tpu_dist worker ===", flush=True)
        for f in gathered:
            print(
                f"  rank {f['rank']:>2}  tpu_global_device_id="
                f"{f['global_device_id']:>2}  jax.process={f['jax_process_index']}"
                f"/{f['jax_process_count']}  jax.devices()="
                f"{f['jax_device_count']}  local={f['jax_local_device_count']}"
                f"  ids={f['jax_visible_ids']}",
                flush=True)

    all_ids = {f["global_device_id"] for f in gathered}
    covered = all_ids.issubset(set(visible_ids))
    if rank == 0:
        print(
            f"\nevery peer's device id visible to this process: {covered}\n"
            f"  peers need: {sorted(all_ids)}\n"
            f"  jax exposes: {visible_ids}",
            flush=True)

    if not covered:
        if rank == 0:
            print(
                "\nVERDICT: a shard_map over the TP group is NOT constructible "
                "in this layout -- JAX only exposes this rank's own device, so "
                "the Pallas kernel has no way to address peers.\n"
                "The kernels cannot be called through pallas.jax_op here "
                "without a different device/process layout.",
                flush=True)
        dist.destroy_process_group()
        return 3

    # ---- The devices are addressable; try to actually run the kernel. ----
    import numpy as np
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
        one_shot_all_reduce_ws

    rank_to_device = {f["rank"]: f["global_device_id"] for f in gathered}
    by_id = {int(getattr(d, "id", i)): d for i, d in enumerate(jax_devices)}
    ordered = np.asarray([by_id[rank_to_device[r]] for r in range(world_size)])
    mesh = Mesh(ordered, axis_names=("tp", ))
    if rank == 0:
        print(f"\nmesh built over {ordered.size} devices, axis 'tp'",
              flush=True)

    rows, cols = 4, 1024
    axis = "tp"

    # jax_op reads the signature to infer static argnums, so every parameter
    # needs an annotation -- and so does the return, since an *absent* return
    # annotation is `Signature.empty` rather than None and fails validation.
    def _all_reduce_ws(
            x: jax.Array, ws: jax.Array,
            step: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:

        def body(x, ws, step):
            return one_shot_all_reduce_ws(x,
                                          ws,
                                          step,
                                          axis_name=axis,
                                          num_devices=world_size,
                                          collective_id=7)

        return shard_map(body,
                         mesh=mesh,
                         in_specs=(P(axis), P(axis), P(axis)),
                         out_specs=(P(axis), P(axis), P(axis)),
                         check_rep=False)(x, ws, step)

    try:
        op = pallas.jax_op(
            "probe::one_shot_all_reduce_ws",
            _all_reduce_ws,
            mesh=mesh,
            input_partition_specs=(P(axis), P(axis), P(axis)),
        )
    except Exception:
        if rank == 0:
            print("\nVERDICT: jax_op registration failed:\n" +
                  traceback.format_exc(),
                  flush=True)
        dist.destroy_process_group()
        return 4

    device = torch.device("tpu")
    # Same layout make_workspace() produces, allocated in torch so it can be
    # threaded call-to-call as ordinary model state.
    ws = torch.zeros((2, world_size - 1, rows, cols),
                     dtype=torch.bfloat16,
                     device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)

    torch.manual_seed(rank)
    x = torch.randn(rows, cols,
                    dtype=torch.float32).to(torch.bfloat16).to(device)

    reference = funcol_all_reduce(x, "sum", list(range(world_size))).cpu()

    try:
        out, ws, step = op(x, ws, step)
        got = out.cpu()
    except Exception:
        if rank == 0:
            print("\nVERDICT: calling the kernel failed:\n" +
                  traceback.format_exc(),
                  flush=True)
        dist.destroy_process_group()
        return 5

    max_abs = (got.float() - reference.float()).abs().max().item()
    ok = max_abs < 0.05
    results: list = [None] * world_size
    dist.all_gather_object(results, {
        "rank": rank,
        "max_abs": max_abs,
        "ok": ok
    },
                           group=gloo)
    if rank == 0:
        print("\n=== correctness vs funcol all_reduce ===", flush=True)
        for r in results:
            print(
                f"  rank {r['rank']:>2}  max|diff|={r['max_abs']:.5f}  "
                f"{'OK' if r['ok'] else 'MISMATCH'}",
                flush=True)
        every = all(r["ok"] for r in results)
        print(
            f"\nVERDICT: the Pallas collective {'WORKS' if every else 'RUNS BUT IS WRONG'} "
            f"through pallas.jax_op in vLLM's process layout.",
            flush=True)

    try:
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"(ignored teardown error: {type(e).__name__}: {e})")
    return 0 if all(r["ok"] for r in results) else 6


if __name__ == "__main__":
    sys.exit(main())
