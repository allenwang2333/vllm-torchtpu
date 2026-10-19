#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Head-unique landing gather on one tpu7x host: correctness and time.

Eight rank processes (one TPU core each) hold POOLS fp8 storages shaped like
the Qwen3.5-397B FA pools ([pages x 5 kernel pages, 2, 64, 4, 128]) filled with
a rank- and row-specific byte pattern. Ranks 0-3 and 4-7 are the two KV-head
groups. For each case the ranks run the decode connector's
``_stage3_gather_landed_pages`` over a block list and check on every rank
that each page holds the bytes of its owner replica (page index % 4), then
time repeated gathers. Prints one ``RESULT gather {...}`` line per case and
``RESULT: PASS`` or ``RESULT: FAIL``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

WORLD = 8
GROUP = 4
KPS = 5  # kernel pages per 640-token block
ROW_SHAPE = (2, 64, 4, 128)  # one kernel page: 64 KiB of fp8


def log(msg: str) -> None:
    print(f"[gather {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pick_ports(n: int) -> list[int]:
    socks = []
    try:
        for _ in range(n):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _tpu_topology(world: int) -> str:
    code = (
        "from torch_tpu._internal.utils import hardware;"
        f"print('TOPO=' + str(hardware.get_tpu_topology({world})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300).stdout
    for line in out.splitlines():
        if line.startswith("TOPO=") and line[5:] not in ("", "None"):
            return line[5:]
    log(f"topology probe printed no topology: {out[-300:]!r}")
    return ""


def prepare_launcher_env(world: int) -> dict:
    env = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_pick_ports(1)[0]),
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
            f"localhost:{p}" for p in _pick_ports(world)
        ),
        "TORCH_TPU_XPROF_SESSION_ID": str(time.time_ns()),
    }
    topology = os.environ.get("TORCH_TPU_TOPOLOGY") or _tpu_topology(world)
    if topology:
        env["TORCH_TPU_TOPOLOGY"] = topology
    log(f"launcher env {env}")
    return env


def pattern_rows(rank: int, pool: int, rows):
    """The byte every element of each row carries: rank, pool and row specific.

    Built on the host explicitly: the rank processes run with the TPU as
    the default device, where a per-row tensor op is a compiled program.
    """
    import torch

    base = (rank * 37 + pool * 11) & 0x7F
    row_ids = torch.tensor(list(rows), dtype=torch.int32, device="cpu")
    values = ((row_ids + base) & 0x7F).to(torch.int8)
    return values.view(-1, 1, 1, 1, 1).expand((len(row_ids),) + ROW_SHAPE).contiguous()


def rank_main(rank: int, args, launcher_env: dict, result_queue) -> None:
    for name in ("MEGASCALE_COORDINATOR_ADDRESS", "MEGASCALE_PORT",
                 "MEGASCALE_NUM_SLICES", "MEGASCALE_SLICE_ID"):
        os.environ.pop(name, None)
    os.environ.update(launcher_env)
    os.environ["TPU_VISIBLE_DEVICES"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
    os.environ["GROUP_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = str(WORLD)
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401  (registers the tpu device and tpu_dist backend)

    device = torch.device("tpu")
    torch.zeros(1, device=device)
    dist.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://127.0.0.1:{launcher_env['MASTER_PORT']}",
        rank=rank,
        world_size=WORLD,
    )
    subgroups = [dist.new_group(list(range(g * GROUP, (g + 1) * GROUP)))
                 for g in range(WORLD // GROUP)]
    own_group = subgroups[rank // GROUP]
    replica = rank % GROUP

    def all_gather(t, dim=0):
        assert dim == 0
        out = torch.empty((GROUP * t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype,
                          device=t.device)
        dist.all_gather_into_tensor(out, t, group=own_group)
        return out

    group = SimpleNamespace(world_size=GROUP, rank_in_group=replica, all_gather=all_gather)

    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnectorWorker

    rows_total = args.num_blocks * KPS
    # Pools the size of the cell's (5 GiB each at 16384 blocks) plus a filler
    # that leaves about --free-mib of HBM, the state a decode leg gathers in.
    storages = [
        torch.empty((rows_total,) + ROW_SHAPE, dtype=torch.float8_e4m3fn, device=device)
        for _ in range(args.pools)
    ]
    pool_bytes = rows_total * 65536 * args.pools
    filler = None
    if args.hbm_gib > 0:
        filler_bytes = int(args.hbm_gib * 2**30) - pool_bytes - int(args.free_mib * 2**20)
        if filler_bytes > 0:
            filler = torch.empty((filler_bytes // 2 // 1024, 1024), dtype=torch.bfloat16,
                                 device=device)
            filler.fill_(0)
    torch.accelerator.synchronize()
    ptrs = [s.untyped_storage().data_ptr() for s in storages]
    log(f"rank {rank}: {args.pools} storages of {rows_total} rows "
        f"({pool_bytes / 2**30:.1f} GiB) ready, filler "
        f"{0 if filler is None else filler.numel() * 2 / 2**30:.1f} GiB")

    from vllm_torchtpu.distributed.kv_transfer.raiden import gather_kernels

    def read_rows(pool: int, rows: list[int]) -> torch.Tensor:
        """The pool's rows as int8 on the host, through the row-copy kernel.

        A slice of a 5 GiB pool is not read directly: eager slicing and
        index ops on a pool of that size run out of place and need a second
        pool-sized buffer, which the filled HBM does not have.
        """
        out = gather_kernels.copy_rows(
            storages[pool],
            torch.empty((len(rows),) + ROW_SHAPE, dtype=torch.float8_e4m3fn, device=device),
            gather_kernels.chunk_args(tuple(rows), tuple(range(len(rows))), device))
        return out.view(torch.int8).cpu()

    def stamp(block_ids):
        """Writes every pool's rows of these blocks with this rank's pattern, in place."""
        rows = [b * KPS + k for b in block_ids for k in range(KPS)]
        chunks = gather_kernels.chunk_args(tuple(range(len(rows))), tuple(rows), device)
        t0 = time.perf_counter()
        for pool in range(args.pools):
            # pattern_rows keys on the pool row, not the list position.
            src = pattern_rows(rank, pool, rows)
            t1 = time.perf_counter()
            gather_kernels.copy_rows_into(
                src.to(device).view(torch.float8_e4m3fn), storages[pool], chunks)
            torch.accelerator.synchronize()
            if pool < 2:
                log(f"rank {rank}: stamp pool {pool} {len(rows)} rows: "
                    f"{(time.perf_counter() - t1) * 1e3:.1f} ms")
        log(f"rank {rank}: stamp of {len(block_ids)} pages x {args.pools} pools: "
            f"{(time.perf_counter() - t0) * 1e3:.0f} ms")
        # The stamp itself is an in-place kernel write into the pool: check it.
        got = read_rows(args.check_pools[0], rows[:KPS])
        want = pattern_rows(rank, args.check_pools[0], range(rows[0], rows[0] + KPS))
        if not torch.equal(got, want):
            raise RuntimeError(
                f"rank {rank}: in-place stamp of pool {args.check_pools[0]} rows "
                f"{rows[:KPS]} did not land (got byte {int(got.flatten()[0])}, "
                f"want {int(want.flatten()[0])})")

    worker = TPURaidenConnectorWorker.__new__(TPURaidenConnectorWorker)
    worker.tp_rank = rank
    worker.tp_size = WORLD
    worker._raiden_manifest = SimpleNamespace(storages=storages)
    worker._stage3_head_unique_pools = [(i, 0, KPS, KPS) for i in range(args.pools)]
    worker._stage3_kv_head_group = group
    worker._stage3_head_unique_group_size = GROUP

    rng = random.Random(args.seed)
    blocks = list(range(args.num_blocks))
    rng.shuffle(blocks)
    cases = [int(x) for x in args.cases.split(",")]
    if sum(cases) > args.num_blocks:
        raise SystemExit("cases need more blocks than --num-blocks")
    cursor = 0
    results = []
    ok = True
    for n in cases:
        block_ids = blocks[cursor:cursor + n]
        cursor += n
        req = f"case-{n}"
        stamp(block_ids)
        timings = []
        for i in range(args.repeats):
            dist.barrier()
            t0 = time.perf_counter()
            worker._stage3_gather_landed_pages({req: block_ids})
            torch.accelerator.synchronize()
            timings.append((time.perf_counter() - t0) * 1e3)
        # Every page must now hold its owner replica's bytes on this rank.
        mismatches = 0
        checked = 0
        t_check = time.perf_counter()
        for pool in args.check_pools:
            all_rows = [b * KPS + k for b in block_ids for k in range(KPS)]
            t1 = time.perf_counter()
            got_all = read_rows(pool, all_rows)
            log(f"rank {rank}: read {len(all_rows)} rows of pool {pool}: "
                f"{(time.perf_counter() - t1) * 1e3:.1f} ms")
            for page, block in enumerate(block_ids):
                owner = (rank // GROUP) * GROUP + page % GROUP
                rows = range(block * KPS, block * KPS + KPS)
                got = got_all[page * KPS:(page + 1) * KPS]
                want = pattern_rows(owner, pool, rows)
                checked += 1
                if not torch.equal(got, want):
                    mismatches += 1
                    if mismatches <= 3:
                        log(f"rank {rank} pool {pool} page {page} block {block} owner {owner}: "
                            f"got byte {int(got.flatten()[0])}, want {int(want.flatten()[0])}")
        ok = ok and mismatches == 0
        probe = {"ptr_kept": [s.untyped_storage().data_ptr() for s in storages] == ptrs,
                 "check_ms": (time.perf_counter() - t_check) * 1e3}
        row = {
            "rank": rank, "pages": n, "pools": args.pools, "repeats": args.repeats,
            "gather_ms_median": statistics.median(timings),
            "gather_ms_min": min(timings), "gather_ms_max": max(timings),
            "mb_in_per_rank": n * (GROUP - 1) / GROUP * KPS * 65536 * args.pools / 1e6,
            "pages_checked": checked, "mismatches": mismatches,
            "probe": probe,
        }
        results.append(row)
        log(f"RESULT gather {json.dumps(row)}")
    del filler
    dist.barrier()
    result_queue.put((rank, ok, results))
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=16384)
    parser.add_argument("--cases", default="4,25,150,100,200")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hbm-gib", type=float, default=94.74,
                        help="HBM to occupy with pools plus a filler (0 = no filler)")
    parser.add_argument("--free-mib", type=float, default=400.0,
                        help="HBM left free beside the pools and the filler")
    parser.add_argument("--check-pools", default="0,7,15")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deadline-s", type=float, default=3 * 3600.0)
    parser.add_argument("--out", default=os.environ.get("P3_OUT_ROOT", "/tmp"))
    args = parser.parse_args()
    args.check_pools = [int(x) for x in args.check_pools.split(",")]
    launcher_env = prepare_launcher_env(WORLD)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=rank_main, args=(r, args, launcher_env, queue))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    reports = {}
    deadline = time.time() + args.deadline_s
    while len(reports) < WORLD and time.time() < deadline:
        try:
            rank, ok, rows = queue.get(timeout=5)
            reports[rank] = (ok, rows)
        except Exception:  # noqa: BLE001
            if any(p.exitcode not in (None, 0) for p in procs):
                break
    for p in procs:
        p.join(timeout=60)
    all_ok = len(reports) == WORLD and all(ok for ok, _ in reports.values())
    summary = {rank: rows for rank, (_, rows) in sorted(reports.items())}
    try:
        with open(os.path.join(args.out, "head_unique_gather.json"), "w") as f:
            json.dump(summary, f, indent=1)
    except OSError as exc:
        log(f"could not write the summary: {exc}")
    for rank, (ok, rows) in sorted(reports.items()):
        for row in rows:
            log(f"rank {rank} pages {row['pages']}: {row['gather_ms_median']:.2f} ms median, "
                f"{row['mismatches']} mismatches of {row['pages_checked']}")
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'} ({len(reports)}/{WORLD} ranks reported)",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
