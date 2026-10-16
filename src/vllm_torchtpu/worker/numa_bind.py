# SPDX-License-Identifier: Apache-2.0
"""Experimental: bind a TPU worker's threads to the CPUs local to its chip.

Enabled by ``TPU_DSV41_NUMA_BIND``:
  ``node``  -- all CPUs of the chip's NUMA node (intersected with the cgroup set);
  ``split`` -- that set divided evenly among the workers sharing the node.
The chip is ``local_rank // TPU_DSV41_NUMA_DEVICES_PER_CHIP`` (default 2, v7x).
Chips are the TPU PCI functions (vendor 0x1ae0) in PCI address order.
"""

import ctypes
import glob
import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_TPU_VENDOR = "0x1ae0"
_MPOL_PREFERRED = 1


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _tpu_devices() -> list[str]:
    devices = []
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            if _read(f"{dev}/vendor") == _TPU_VENDOR and _read(
                f"{dev}/class"
            ).startswith("0x12"):
                devices.append(dev)
        except OSError:
            continue
    return devices


def _set_preferred_node(node: int) -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mask = ctypes.c_ulong(1 << node)
        # set_mempolicy(MPOL_PREFERRED, &mask, maxnode); syscall 238 on x86_64.
        return libc.syscall(238, _MPOL_PREFERRED, ctypes.byref(mask), 64) == 0
    except OSError:
        return False


def maybe_bind_worker(local_rank: int, local_world: int) -> None:
    mode = os.environ.get("TPU_DSV41_NUMA_BIND", "")
    if not mode:
        return
    per_chip = int(os.environ.get("TPU_DSV41_NUMA_DEVICES_PER_CHIP", "2"))
    devices = _tpu_devices()
    chip = local_rank // per_chip
    if chip >= len(devices):
        logger.warning("[numa_bind] chip %d not in %s; skipping", chip, devices)
        return
    dev = devices[chip]
    node = int(_read(f"{dev}/numa_node"))
    cpus = _parse_cpulist(_read(f"{dev}/local_cpulist")) & os.sched_getaffinity(0)
    if not cpus:
        logger.warning("[numa_bind] no allowed CPUs local to %s; skipping", dev)
        return
    if mode == "split":
        peers = [
            r
            for r in range(local_world)
            if r // per_chip < len(devices)
            and _read(f"{devices[r // per_chip]}/numa_node") == str(node)
        ]
        ordered = sorted(cpus)
        share = len(ordered) // max(len(peers), 1)
        index = peers.index(local_rank)
        cpus = set(ordered[index * share : (index + 1) * share]) or cpus
    bound = 0
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), cpus)
            bound += 1
        except OSError:
            pass
    mem = _set_preferred_node(node) if node >= 0 else False
    logger.info(
        "[numa_bind] local_rank=%d chip=%d dev=%s node=%d mode=%s cpus=%d "
        "threads=%d mempolicy=%s",
        local_rank,
        chip,
        os.path.basename(dev),
        node,
        mode,
        len(cpus),
        bound,
        mem,
    )
