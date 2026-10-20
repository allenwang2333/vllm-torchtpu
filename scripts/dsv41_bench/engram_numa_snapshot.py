# SPDX-License-Identifier: Apache-2.0
"""Snapshot where each DSv4.1 worker's Engram staging buffers live.

Run inside the serving container (``docker exec``) while the server is up,
after it started with ``TPU_DEEPSEEK_V41_ENGRAM_STAGING_LOG=<log_dir>``:

    python engram_numa_snapshot.py <log_dir> <out.json> [--full]

For every buffer the workers logged at init it records the current node and
THP backing of each page, the ``numa_maps`` line of the VMA holding it, the
worker's CPU affinity and the NUMA node of the TPU PCI function it opened.
``--smaps`` records every mapping of each worker of at least 1 MiB (size,
RSS, AnonHugePages, Locked, VmFlags), to find the runtime's own host buffers
and their page backing. ``--threads`` samples every worker thread's last CPU
and CPU time twice, one second apart, to show which threads are busy and
where. ``--full`` also totals
each worker's resident pages per node (reads the whole ``numa_maps``, which
walks every page table; take it outside timed windows).
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import time

_SPEC = importlib.util.spec_from_file_location(
    "host_pages",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../src/vllm_torchtpu/models/vllm/deepseek_v41/common/host_pages.py",
    ),
)
host_pages = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(host_pages)


def _vma(pid: int, addr: int) -> tuple[int, int, str] | None:
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            span, rest = line.split(" ", 1)
            lo, hi = (int(x, 16) for x in span.split("-"))
            if lo <= addr < hi:
                return lo, hi, rest.strip()
    return None


def _numa_maps(pid: int, starts: set[int] | None) -> tuple[dict, dict]:
    """numa_maps lines for VMAs starting at ``starts`` (all when None), and
    the per-node resident page totals of the lines read."""
    lines, totals = {}, {}
    with open(f"/proc/{pid}/numa_maps") as f:
        for line in f:
            start = int(line.split(" ", 1)[0], 16)
            if starts is None:
                for node, pages in re.findall(r" N(\d+)=(\d+)", line):
                    size = re.search(r"kernelpagesize_kB=(\d+)", line)
                    kb = int(size.group(1)) if size else 4
                    totals[node] = totals.get(node, 0) + int(pages) * kb
            if starts is not None and start in starts:
                lines[start] = line.strip()
                if len(lines) == len(starts):
                    break
            elif starts is None:
                lines[start] = line.strip()
    return lines, totals


def _status(pid: int) -> dict:
    out = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key in ("Cpus_allowed_list", "Mems_allowed_list", "VmRSS"):
                out[key] = value.strip()
    return out


def _thread_times(pid: int) -> dict[int, tuple[str, int, int]]:
    """tid -> (comm, last CPU, utime + stime in ticks)."""
    out = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                stat = f.read()
        except FileNotFoundError:
            continue  # exited since listdir
        comm = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        out[int(tid)] = (comm, int(fields[36]), int(fields[11]) + int(fields[12]))
    return out


_SMAPS_KEYS = ("Rss", "AnonHugePages", "Locked", "THPeligible")


def smaps_report(pid: int, min_kb: int = 1024) -> list[dict]:
    """Mappings of at least ``min_kb`` with their huge-page backing."""
    out, cur = [], None
    with open(f"/proc/{pid}/smaps") as f:
        for line in f:
            head = line.split(" ", 1)[0]
            if "-" in head and ":" not in head:
                lo, hi = (int(x, 16) for x in head.split("-"))
                parts = line.split()
                cur = {
                    "start": hex(lo),
                    "kb": (hi - lo) // 1024,
                    "perms": parts[1],
                    "name": parts[5] if len(parts) > 5 else "",
                }
                if cur["kb"] >= min_kb:
                    out.append(cur)
            elif cur is not None and head.rstrip(":") in _SMAPS_KEYS:
                cur[head.rstrip(":")] = int(line.split()[1])
            elif cur is not None and head == "VmFlags:":
                cur["VmFlags"] = line.split(":", 1)[1].strip()
    return out


def sample_threads(pids: list[int]) -> dict[int, list]:
    """Per worker, threads busier than 5% over one second: [tid, comm, cpu, busy]."""
    tick = os.sysconf("SC_CLK_TCK")
    first = {pid: _thread_times(pid) for pid in pids}
    time.sleep(1.0)
    out = {}
    for pid in pids:
        rows = []
        for tid, (comm, cpu, ticks) in _thread_times(pid).items():
            if tid in first[pid]:
                busy = (ticks - first[pid][tid][2]) / tick
                if busy > 0.05:
                    rows.append([tid, comm, cpu, round(busy, 2)])
        out[pid] = sorted(rows, key=lambda r: -r[3])
    return out


def snapshot(log_dir: str, full: bool) -> dict:
    procs: dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "staging-*.jsonl"))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                procs.setdefault(rec["pid"], {"init": []})["init"].append(rec)
    for pid, proc in procs.items():
        if not os.path.exists(f"/proc/{pid}"):
            proc["alive"] = False
            continue
        proc["alive"] = True
        proc["status"] = _status(pid)
        groups = host_pages.tpu_vfio_groups(pid)
        proc["vfio"] = {g: host_pages.vfio_group_numa_nodes(g) for g in groups}
        buffers, vma_starts = [], set()
        for rec in proc["init"]:
            vma = _vma(pid, rec["addr"])
            if vma is not None:
                vma_starts.add(vma[0])
            buffers.append(
                {
                    "buffer": rec["buffer"],
                    "layer_hash_index": rec.get("layer_hash_index"),
                    "addr": rec["addr"],
                    "nbytes": rec["nbytes"],
                    "vma": vma and [hex(vma[0]), hex(vma[1]), vma[2]],
                    "now": host_pages.summarize_pages(rec["addr"], rec["nbytes"], pid),
                }
            )
        lines, totals = _numa_maps(pid, None if full else vma_starts)
        for buf in buffers:
            if buf["vma"] is not None:
                buf["numa_maps"] = lines.get(int(buf["vma"][0], 16))
        proc["buffers"] = buffers
        if full:
            proc["resident_kb_per_node"] = totals
    return {"time": time.time(), "full": full, "procs": procs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("log_dir")
    parser.add_argument("out")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--threads", action="store_true")
    parser.add_argument("--smaps", action="store_true")
    args = parser.parse_args()
    snap = snapshot(args.log_dir, args.full)
    if args.smaps:
        snap["smaps"] = {
            pid: smaps_report(pid)
            for pid, proc in snap["procs"].items()
            if proc["alive"]
        }
    if args.threads:
        alive = [pid for pid, proc in snap["procs"].items() if proc["alive"]]
        snap["busy_threads"] = sample_threads(alive)
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=1)
    print(f"{len(snap['procs'])} workers -> {args.out}")


if __name__ == "__main__":
    main()
