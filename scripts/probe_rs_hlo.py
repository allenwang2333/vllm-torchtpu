"""Minimal 8-rank reproducer for the DP reduce_scatter path that vLLM's
TpuDeviceCommunicator.combine invokes. Emits HLO dumps via LIBTPU_INIT_ARGS
so we can inspect whether libtpu is lowering stablehlo.reduce_scatter as
native reduce-scatter or emulating it via all-reduce + dynamic-slice.

Runs on a single host with 8 TPU chips (topology 2,2,1,2).

Usage:
  LIBTPU_INIT_ARGS="--xla_dump_to=/tmp/hlo_probe_rs --xla_dump_hlo_as_text" \\
    python scripts/probe_rs_hlo.py
"""

import os
import time

# Match the launcher's TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT convention:
# use a unique subdir per run so the reduce_scatter graph actually compiles
# (and XLA dumps HLO) instead of being served from cache.
date = time.strftime("%Y-%m-%d", time.localtime())
cache_root = os.environ.get(
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
    "/tmp/torchtpu_probe_cache",
)
os.environ["TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"] = os.path.join(
    cache_root, f"{date}.{os.getpid()}"
)

import portpicker
import torch  # pylint: disable=g-import-not-at-top
from torch import distributed as dist
import torch.multiprocessing as mp

from torch_tpu._internal.distributed.launchers import singlehost_wrapper


DP_SIZE = 8


def _worker_fn(rank, world_size, master_port, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["GROUP_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    # Route each rank's HLO dumps into a private subdir so the after_optimizations
    # files don't stomp each other. XLA reads XLA_FLAGS at libtpu init time,
    # which happens on first tensor allocation, so setting it here (before we
    # import torch_tpu-triggering ops) is early enough.
    base = os.environ.get("XLA_DUMP_BASE")
    if base:
        rank_dump = os.path.join(base, f"rank{rank}")
        os.makedirs(rank_dump, exist_ok=True)
        os.environ["XLA_FLAGS"] = (
            f"--xla_dump_to={rank_dump} --xla_dump_hlo_as_text "
            "--xla_dump_hlo_as_proto"
        )
    fn()
# The MoE combine output shape at max-num-batched-tokens=1024 per DP rank,
# after the allgather in naive_dp_ep dispatch: (dp_size * per_rank, hidden).
# Qwen3.5-397B: hidden_size=4096, dtype=bfloat16.
SHAPES = [
    (8 * 1024, 4096),   # max batched-tokens config
    (8 * 512, 4096),    # bucketed smaller
    (8 * 128, 4096),    # decode-ish
    (8 * 64, 4096),
]


def _worker():
    dist.init_process_group(backend="tpu_dist")
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        assert world_size == DP_SIZE, (world_size, DP_SIZE)

        for i, (M, H) in enumerate(SHAPES):
            # Every rank contributes a distinct input so reduction is nontrivial.
            x = torch.full(
                (M, H), fill_value=float(rank + 1), dtype=torch.bfloat16, device="tpu"
            )
            out = torch.empty((M // world_size, H), dtype=torch.bfloat16, device="tpu")
            torch.distributed.reduce_scatter_tensor(out, x)
            # Materialize to force execution + trigger compile.
            _ = out.cpu()
            if rank == 0:
                s = float(out.float().sum().item())
                expected = (world_size * (world_size + 1) // 2) * (M // world_size) * H
                print(f"[shape {i}] out sum = {s:.1f} expected = {expected:.1f}")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    wrapped = singlehost_wrapper.tpu_env_wrapper(_worker, world_size=DP_SIZE)
    master_port = portpicker.pick_unused_port()
    mp.spawn(
        _worker_fn,
        args=(DP_SIZE, master_port, wrapped),
        nprocs=DP_SIZE,
        join=True,
    )
