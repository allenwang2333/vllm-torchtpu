# SPDX-License-Identifier: Apache-2.0
"""Host Engram staging buffers backed by 2 MiB pages, and page reports.

TorchTPU's host-to-device copy costs ~20 us per 4 KiB page of the source
against ~0.25 ms for the whole 1.82 MB decode packed buffer on 2 MiB pages
(``scripts/dsv41_bench/bench_engram_h2d_placement.py``); the NUMA node of the
pages does not matter. ``torch.zeros`` leaves the head of its mapping, where
the decode rows sit, on 4 KiB pages whenever the mapping is not 2 MiB aligned.
``alloc_staging`` allocates the buffer 2 MiB aligned and populated at init, on
transparent huge pages when the host's THP mode allows ``madvise``. The report
helpers record the node and huge-page backing of every page of a buffer.
"""

import ctypes
import json
import mmap
import os
import re
import struct
import threading
from dataclasses import dataclass

import torch

_HUGE = 2 << 20
_PAGE = mmap.PAGESIZE
_SYS_MOVE_PAGES = 279  # x86_64
_MADV_HUGEPAGE = 14
_MADV_NOHUGEPAGE = 15
_KPF_THP = 1 << 22
_PFN_MASK = (1 << 55) - 1

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _check(rc: int, what: str) -> None:
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def tpu_vfio_groups(pid: int | str = "self") -> list[int]:
    """IOMMU groups of the ``/dev/vfio/<n>`` files ``pid`` holds open."""
    groups = set()
    fd_dir = f"/proc/{pid}/fd"
    for fd in os.listdir(fd_dir):
        try:
            target = os.readlink(f"{fd_dir}/{fd}")
        except FileNotFoundError:
            continue  # closed since listdir
        match = re.fullmatch(r"/dev/vfio/(\d+)", target)
        if match:
            groups.add(int(match.group(1)))
    return sorted(groups)


def vfio_group_numa_nodes(group: int) -> dict[str, int]:
    """PCI functions of an IOMMU group and their NUMA nodes."""
    devices = f"/sys/kernel/iommu_groups/{group}/devices"
    nodes = {}
    for dev in sorted(os.listdir(devices)):
        with open(f"{devices}/{dev}/numa_node") as f:
            nodes[dev] = int(f.read())
    return nodes


def tpu_numa_node() -> int:
    """NUMA node of the TPU PCI function(s) this process has opened."""
    groups = tpu_vfio_groups()
    nodes = {node for group in groups for node in vfio_group_numa_nodes(group).values()}
    if len(nodes) != 1 or min(nodes) < 0:
        raise RuntimeError(
            f"cannot derive one TPU NUMA node: vfio groups {groups}, nodes {nodes}"
        )
    return nodes.pop()


def _vma_anon_huge_kb(addr: int) -> int:
    """``AnonHugePages`` of the mapping of this process holding ``addr``."""
    inside = False
    with open("/proc/self/smaps") as f:
        for line in f:
            head = line.split(" ", 1)[0]
            if "-" in head and ":" not in head:
                lo, hi = (int(x, 16) for x in head.split("-"))
                inside = lo <= addr < hi
            elif inside and head == "AnonHugePages:":
                return int(line.split()[1])
    raise RuntimeError(f"no mapping holds {addr:#x}")


@dataclass(frozen=True)
class Staging:
    tensor: torch.Tensor
    huge_kb: int  # KiB of the mapping backed by transparent 2 MiB pages
    mapped_kb: int


def alloc_staging(nbytes: int, *, huge: bool) -> Staging:
    """Zeroed uint8 host tensor, 2 MiB aligned, advised onto transparent 2 MiB
    pages (``huge``) or onto 4 KiB pages; reports what the kernel gave. The
    tensor owns the mapping."""
    size = -(-nbytes // _HUGE) * _HUGE
    # Over-map by one huge page so the tensor can start 2 MiB aligned.
    buf = mmap.mmap(-1, size + _HUGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    offset = (-base) % _HUGE
    addr = base + offset
    advice = _MADV_HUGEPAGE if huge else _MADV_NOHUGEPAGE
    _check(_libc.madvise(ctypes.c_void_p(addr), size, advice), "madvise")
    ctypes.memset(addr, 0, size)
    tensor = torch.frombuffer(buf, dtype=torch.uint8, count=nbytes, offset=offset)
    return Staging(tensor, _vma_anon_huge_kb(addr), size // 1024)


def thp_mode() -> str:
    """The host's transparent huge page mode, e.g. ``always [madvise] never``."""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "unsupported"


def page_nodes(addr: int, nbytes: int, pid: int = 0) -> list[int]:
    """NUMA node of each page in ``[addr, addr + nbytes)`` (negative errno if
    not present), via ``move_pages`` in query mode."""
    first = addr - addr % _PAGE
    count = -(-(addr + nbytes - first) // _PAGE)
    pages = (ctypes.c_void_p * count)(*(first + i * _PAGE for i in range(count)))
    status = (ctypes.c_int * count)()
    rc = _libc.syscall(
        _SYS_MOVE_PAGES, pid, ctypes.c_ulong(count), pages, None, status, 0
    )
    _check(int(rc), "move_pages")
    return list(status)


def page_thp(addr: int, nbytes: int, pid: int | str = "self") -> list[bool] | None:
    """Whether each page is part of a transparent huge page, or None when
    PFNs are not readable (needs CAP_SYS_ADMIN)."""
    first = addr - addr % _PAGE
    count = -(-(addr + nbytes - first) // _PAGE)
    with open(f"/proc/{pid}/pagemap", "rb") as f:
        f.seek(first // _PAGE * 8)
        entries = struct.unpack(f"<{count}Q", f.read(count * 8))
    pfns = [e & _PFN_MASK for e in entries]
    if not any(pfns):
        return None
    flags = []
    with open("/proc/kpageflags", "rb") as f:
        for pfn in pfns:
            f.seek(pfn * 8)
            flags.append(bool(struct.unpack("<Q", f.read(8))[0] & _KPF_THP))
    return flags


def summarize_pages(addr: int, nbytes: int, pid: int = 0) -> dict:
    """Per-node and THP page counts of a host range, plus its pages as
    ``[node, thp, count]`` runs in address order."""
    nodes = page_nodes(addr, nbytes, pid)
    thp = page_thp(addr, nbytes, pid or "self")
    counts: dict[str, int] = {}
    runs: list[list] = []
    for i, node in enumerate(nodes):
        counts[str(node)] = counts.get(str(node), 0) + 1
        key = [node, None if thp is None else thp[i]]
        if runs and runs[-1][:2] == key:
            runs[-1][2] += 1
        else:
            runs.append([*key, 1])
    return {
        "pages": len(nodes),
        "node_pages": counts,
        "thp_pages": None if thp is None else sum(thp),
        "runs": runs,
    }


def log_staging_buffers(log_dir: str, buffers: dict[str, torch.Tensor], **meta) -> None:
    """Append one JSON line per buffer: address, size, page nodes and the
    process's TPU node and CPU affinity, for offline NUMA attribution."""
    os.makedirs(log_dir, exist_ok=True)
    pid = os.getpid()
    groups = tpu_vfio_groups()
    record = {
        "pid": pid,
        "vfio_groups": {g: vfio_group_numa_nodes(g) for g in groups},
        "cpus_allowed": sorted(os.sched_getaffinity(0)),
        **meta,
    }
    with open(os.path.join(log_dir, f"staging-{pid}.jsonl"), "a") as f:
        for name, tensor in buffers.items():
            addr, nbytes = tensor.data_ptr(), tensor.numel() * tensor.element_size()
            line = {
                **record,
                "buffer": name,
                "addr": addr,
                "nbytes": nbytes,
                **summarize_pages(addr, nbytes),
            }
            f.write(json.dumps(line) + "\n")


class H2DStepLog:
    """Per-upload host time and CPU of the issuing thread, flushed to
    ``<log_dir>/h2d-<pid>.jsonl`` in batches to stay off the step's path."""

    def __init__(self, log_dir: str, **meta) -> None:
        self._path = os.path.join(log_dir, f"h2d-{os.getpid()}.jsonl")
        self._meta = meta
        self._rows: list[tuple] = []
        _libc.sched_getcpu.restype = ctypes.c_int

    @staticmethod
    def cpu() -> int:
        return _libc.sched_getcpu()

    def add(self, start_ns: int, end_ns: int, cpu0: int, cpu1: int, **fields) -> None:
        self._rows.append((start_ns, end_ns, cpu0, cpu1, fields))
        if len(self._rows) >= 256:
            self.flush()

    def flush(self) -> None:
        with open(self._path, "a") as f:
            for start_ns, end_ns, cpu0, cpu1, fields in self._rows:
                row = {
                    "start_ns": start_ns,
                    "ms": (end_ns - start_ns) / 1e6,
                    "cpu": [cpu0, cpu1],
                    "tid": threading.get_native_id(),
                    **self._meta,
                    **fields,
                }
                f.write(json.dumps(row) + "\n")
        self._rows.clear()
