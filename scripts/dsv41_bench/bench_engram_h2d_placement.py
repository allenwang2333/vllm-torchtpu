# SPDX-License-Identifier: Apache-2.0
"""Host time of the decode Engram packed H2D, by source-buffer placement.

Each of the 8 ranks (one TPU device each, as in DP8 serving) copies the C384
decode packed buffer (s32[456192], 1.82 MB) host -> device from sources that
differ only in where their pages live:

  default     first 1.82 MB of ``torch.zeros(13 MB)`` (today's allocation)
  staging     ``host_pages.alloc_staging(huge=True)`` (TPU_DEEPSEEK_V41_ENGRAM_
              HUGEPAGE_STAGING)
  local_thp   the device's NUMA node, 2 MiB pages
  remote_thp  the other node, 2 MiB pages
  local_4k    the device's node, 4 KiB pages (``MADV_NOHUGEPAGE``)
  remote_4k   the other node, 4 KiB pages

then repeats ``staging`` with every thread of the process pinned to the CPUs
of the device's node (``staging_cpus_local``) and of the other node
(``staging_cpus_remote``), and reports the median host-blocking ms of
``dst.copy_(src)`` and of the following sync, per rank and source. The copied
words are checked on device.

    perfkit device run --purpose ... -- dsv41-docker \\
        python scripts/dsv41_bench/bench_engram_h2d_placement.py --result-dir DIR
"""

import argparse
import ctypes
import mmap
import os
import statistics
import time

import common

WORDS = 456192  # 288 tokens x 24 columns x 264 B / 4


def _host_pages():
    from vllm_torchtpu.models.vllm.deepseek_v41.common import host_pages

    return host_pages


def _on_node(nbytes: int, node: int, huge: bool):
    """2 MiB-aligned zeroed buffer bound (MPOL_BIND) to ``node``."""
    import torch

    hp = _host_pages()
    huge_page = 2 << 20
    buf = mmap.mmap(
        -1, nbytes + 2 * huge_page, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    )
    base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    offset = (-base) % huge_page
    addr = base + offset
    size = -(-nbytes // huge_page) * huge_page
    advice = 14 if huge else 15  # MADV_HUGEPAGE / MADV_NOHUGEPAGE
    hp._check(hp._libc.madvise(ctypes.c_void_p(addr), size, advice), "madvise")
    mask = ctypes.c_ulong(1 << node)
    rc = hp._libc.syscall(
        237,  # mbind, x86_64
        ctypes.c_void_p(addr),
        ctypes.c_ulong(size),
        2,  # MPOL_BIND
        ctypes.byref(mask),
        ctypes.c_ulong(64),
        1,  # MPOL_MF_STRICT
    )
    hp._check(int(rc), "mbind")
    ctypes.memset(addr, 0, size)
    return torch.frombuffer(buf, dtype=torch.uint8, count=nbytes, offset=offset)


def _node_cpus(node: int) -> set[int]:
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        spans = f.read().strip().split(",")
    cpus = set()
    for span in spans:
        lo, _, hi = span.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _pin_process(cpus: set[int]) -> None:
    for tid in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(tid), cpus)


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict:
    import torch

    hp = _host_pages()
    node = hp.tpu_numa_node()
    other = 1 - node
    nbytes = WORDS * 4
    sources = {
        "default": torch.zeros(13_000_000, dtype=torch.uint8)[:nbytes],
        "staging": hp.alloc_staging(nbytes, huge=True).tensor,
        "local_thp": _on_node(nbytes, node, huge=True),
        "remote_thp": _on_node(nbytes, other, huge=True),
        "local_4k": _on_node(nbytes, node, huge=False),
        "remote_4k": _on_node(nbytes, other, huge=False),
    }
    dst = torch.zeros(WORDS, dtype=torch.int32, device="tpu")
    out: dict = {"device_node": node, "sources": {}}
    all_cpus = os.sched_getaffinity(0)
    runs = [(name, raw, all_cpus) for name, raw in sources.items()]
    runs += [
        ("staging_cpus_local", sources["staging"], _node_cpus(node)),
        ("staging_cpus_remote", sources["staging"], _node_cpus(other)),
    ]
    for name, raw, cpus in runs:
        _pin_process(cpus)
        words = raw.view(torch.int32)
        words.copy_(torch.randint(-(2**31), 2**31 - 1, (WORDS,), dtype=torch.int32))
        pages = hp.summarize_pages(raw.data_ptr(), nbytes)
        copy_ms, sync_ms = [], []
        for i in range(args.warmup + args.iters):
            torch.distributed.barrier(group=ctx.cpu_group)
            t0 = time.perf_counter()
            dst.copy_(words)
            t1 = time.perf_counter()
            common.synchronize(dst)
            t2 = time.perf_counter()
            if i >= args.warmup:
                copy_ms.append((t1 - t0) * 1e3)
                sync_ms.append((t2 - t1) * 1e3)
        if not torch.equal(dst.cpu(), words):
            raise AssertionError(f"{name}: device words differ from the source")
        out["sources"][name] = {
            "node_pages": pages["node_pages"],
            "thp_pages": pages["thp_pages"],
            "copy_ms_median": statistics.median(copy_ms),
            "copy_ms_p90": sorted(copy_ms)[int(0.9 * len(copy_ms))],
            "sync_ms_median": statistics.median(sync_ms),
        }
    _pin_process(all_cpus)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    common.add_common_args(parser, ["h2d"])
    parser.set_defaults(iters=50, warmup=5)
    common.run_main("engram_h2d_placement", parser, worker)


if __name__ == "__main__":
    main()
