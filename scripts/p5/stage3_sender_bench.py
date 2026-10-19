#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 sender-path microbench with the Qwen3.5-397B TP8 cell geometry.

No model and no weights: every rank process holds synthetic device storages
shaped like the cell's unified KV pool (15 regions of one FA layer plus three
GDN layers, and one MTP draft FA layer), builds the pool manifest and the
request byte spans with the same vllm_torchtpu helpers the connector uses, and
drives the same tpu_sync calls (KVCacheManager, ReshardStore,
register_work_unit, register_request_blocks, start_transfer, poll_stats).

Modes (``--mode``):

  check  CPU only: build the manifest and one request's spans, print the
         geometry summary. Runs on the VM.
  d2h    One pod, R rank processes on one host. Each rank registers the pools
         on a full host mirror and times d2h_pool_blocks over the FA pools for
         requests of N pages (issue = the calls returning, done = every future
         ready), all ranks in step. Reads the raw D2H cost on this stack.
  pair   Two pods. ``--role p``: R source ranks, rank 0 hosts the source
         ReshardStore, every rank registers its work unit and the request
         blocks of every request the sweep will pull. ``--role d``: R
         destination ranks, rank 0 hosts the destination store, and the
         launcher process drives the sweep: submit threads call
         start_transfer on the source store (the decode connector's call),
         capped at ``--inflight`` unlanded transfers, and a transfer lands
         when every destination rank reports it in poll_stats.

Outputs go to ``--out``: ``env.json``, per-config ``result.json`` lines in
``results.jsonl``, per-request rows in ``requests.jsonl`` (pair, D side),
``sends-r<k>.jsonl`` (pair, P side), and the source coordinator's
``raiden_pool_reshard_senders_dispatched`` lines in the P pod's log.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import queue
import socket
import statistics
import sys
import threading
import time
import types
from typing import Any

# ---------------------------------------------------------------------------
# Cell geometry (Qwen3.5-397B-A17B-FP8, TP8 head-sharded plane, 640-token
# pages, HND seq-on-lane FA, MTP draft layer).
# ---------------------------------------------------------------------------

PAGE_TOKENS = 640
KERNEL_PAGE_TOKENS = 128
TP = 8
TOTAL_KV_HEADS = 2
HEAD_DIM = 256
NUM_LAYERS = 60
FA_INTERVAL = 4
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 64
LINEAR_HEAD_DIM = 128
CONV_KERNEL = 4
GDN_GROUPS = FA_INTERVAL - 1
MAX_REQUEST_PAGES = 410  # ceil(262144 / 640)
FINGERPRINT = "p5-item2-bench-hnd128"


class FakeStorage:
    _next = [1 << 20]

    def __init__(self):
        FakeStorage._next[0] += 1 << 20
        self._ptr = FakeStorage._next[0]

    def data_ptr(self):
        return self._ptr


class FakeTensor:
    """Shape and storage identity only (the manifest builder's duck type)."""

    def __init__(self, shape, element_size, dtype, storage=None, offset_elems=0):
        self.shape = tuple(int(d) for d in shape)
        self._element_size = int(element_size)
        self.dtype = dtype
        self._storage = storage if storage is not None else FakeStorage()
        self._offset = int(offset_elems)
        numel = 1
        for dim in self.shape:
            numel *= dim
        self.nbytes = numel * self._element_size

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self._storage

    def storage_offset(self):
        return self._offset


@dataclasses.dataclass
class Geometry:
    manifest: Any
    fa_geometry: Any
    page_bytes: int
    fa_pool_indices: list[int]
    state_tags: list[tuple[str, int]]  # (exact tag, mamba group ordinal)
    staging_hints: list[int]
    num_blocks: int


def _fa_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def _gdn_name(layer: int) -> str:
    return f"model.layers.{layer}.linear_attn"


MTP_FA_NAME = "mtp.layers.0.self_attn.attn"


def build_geometry(num_blocks: int, mtp: bool = True) -> Geometry:
    """Builds the cell's per-rank manifest over fake storages."""
    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

    kv_rows = 2 * max(1, TOTAL_KV_HEADS // TP)
    packed_rows = HEAD_DIM // sol.PACKING
    kernel_pages = num_blocks * (PAGE_TOKENS // KERNEL_PAGE_TOKENS)
    fa_shape = (kernel_pages, kv_rows, packed_rows, sol.PACKING, KERNEL_PAGE_TOKENS)
    num_regions = NUM_LAYERS // FA_INTERVAL
    raws = [
        FakeTensor(fa_shape, 1, "torch.float8_e4m3fn")
        for _ in range(num_regions + (1 if mtp else 0))
    ]
    page_bytes = raws[0].nbytes // num_blocks

    fa_spec = types.SimpleNamespace(
        block_size=PAGE_TOKENS, num_kv_heads=kv_rows // 2, head_size=HEAD_DIM
    )
    local_k = LINEAR_KEY_HEADS // TP
    local_v = LINEAR_VALUE_HEADS // TP
    conv_dim = 2 * local_k * LINEAR_HEAD_DIM + local_v * LINEAR_HEAD_DIM
    mamba_spec = types.SimpleNamespace(
        shapes=((CONV_KERNEL - 1, conv_dim), (local_v, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM)),
        dtypes=("torch.bfloat16", "torch.bfloat16"),
        page_size_bytes=page_bytes,
        block_size=PAGE_TOKENS,
    )
    named: dict[str, Any] = {}
    fa_layers: list[str] = []
    gdn_layers: list[list[str]] = [[] for _ in range(GDN_GROUPS)]
    ordinals: dict[str, int] = {}
    for region in range(num_regions):
        for j in range(GDN_GROUPS):
            name = _gdn_name(FA_INTERVAL * region + j)
            named[name] = [raws[region]]
            gdn_layers[j].append(name)
            ordinals[name] = j
        name = _fa_name(FA_INTERVAL * region + GDN_GROUPS)
        named[name] = raws[region]
        fa_layers.append(name)
    layer_index_by_name = {}
    groups = [types.SimpleNamespace(layer_names=fa_layers, kv_cache_spec=fa_spec)]
    for j in range(GDN_GROUPS):
        groups.append(
            types.SimpleNamespace(layer_names=gdn_layers[j], kv_cache_spec=mamba_spec)
        )
    if mtp:
        named[MTP_FA_NAME] = raws[-1]
        layer_index_by_name[MTP_FA_NAME] = NUM_LAYERS
        groups.append(
            types.SimpleNamespace(layer_names=[MTP_FA_NAME], kv_cache_spec=fa_spec)
        )
    gdn_geometry = rpm.GdnHeadGeometry(
        local_key_heads=local_k,
        local_value_heads=local_v,
        key_head_dim=LINEAR_HEAD_DIM,
        value_head_dim=LINEAR_HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
    )
    manifest = sol.build_qwen35_pool_manifest_sol(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=raws,
        gdn_geometry=gdn_geometry,
        mamba_group_ordinal_by_layer=ordinals,
        per_layer_tags=False,
        layer_index_by_name=layer_index_by_name,
    )
    fa_pool_indices = [
        i for i, pool in enumerate(manifest.pools) if rpm.class_tag(pool.tag) == rpm.TAG_FA
    ]
    state_tags: list[tuple[str, int]] = []
    for j in range(GDN_GROUPS):
        state_tags.append((f"{rpm.TAG_GDN_CONV}.g{j}", j))
        state_tags.append((f"{rpm.TAG_GDN_SSM}.g{j}", j))
    # The connector's _raiden_staging_blocks_per_pool: per storage, pages per
    # max-length request for the FA table plus one per GDN table.
    tables: dict[int, dict[str, int]] = {}
    for pool in manifest.pools:
        tag = str(pool.tag)
        blocks = MAX_REQUEST_PAGES if tag.startswith(rpm.TAG_FA) else 1
        table = "fa" if tag.startswith(rpm.TAG_FA) else tag.split(".")[-1]
        if pool.layer_name == MTP_FA_NAME:
            table = "mtp"
        per = tables.setdefault(int(pool.storage_index), {})
        per[table] = max(per.get(table, 0), blocks)
    hints = [sum(tables[int(pool.storage_index)].values()) for pool in manifest.pools]
    return Geometry(
        manifest=manifest,
        fa_geometry=sol.sol_fa_geometry_from_manifest(manifest),
        page_bytes=page_bytes,
        fa_pool_indices=fa_pool_indices,
        state_tags=state_tags,
        staging_hints=hints,
        num_blocks=num_blocks,
    )


def request_spans(geo: Geometry, tp_rank: int, pages: int, block_ids, state_block_ids):
    """The producer rank's pool_spans for one request (connector parity)."""
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

    num_tokens = pages * PAGE_TOKENS
    fa = sol.lower_fa_spans_sol(
        num_tokens=num_tokens,
        transfer_rank=0,
        parallelism=1,
        interleave_tokens=PAGE_TOKENS,
        page_tokens=PAGE_TOKENS,
        geometry=geo.fa_geometry,
        dst_shards=TP,
        block_ids=list(block_ids),
        producer_tp_size=TP,
        producer_tp_rank=tp_rank,
        total_kv_heads=TOTAL_KV_HEADS,
        dst_dcp_size=1,
    )
    spans = [fa] if fa.spans else []
    physical_granule = geo.fa_geometry.packing * geo.fa_geometry.page_tokens
    conv_lane_bytes = sol.sol_gdn_conv_lane_bytes(geo.fa_geometry)
    for tag, ordinal in geo.state_tags:
        regions = next(p.regions for p in geo.manifest.pools if str(p.tag) == tag)
        spans.append(
            sol.lower_gdn_state_shard_spans_sol(
                tag=tag,
                block_id=int(state_block_ids[ordinal]),
                transfer_rank=0,
                parallelism=1,
                regions=regions,
                dst_shards=TP,
                producer_tp_size=TP,
                producer_tp_rank=tp_rank,
                physical_granule_bytes=physical_granule,
                num_key_heads=LINEAR_KEY_HEADS,
                num_value_heads=LINEAR_VALUE_HEADS,
                conv_lane_bytes=conv_lane_bytes,
            )
        )
    return spans


def unit_fields(job: str, engine: str, rank: int) -> dict:
    return {
        "job_name": job,
        "job_replica_id": f"{engine}-rank{rank}",
        "data_name": "kv.fa",
        "data_replica_idx": 0,
    }


def transfer_tags(geo: Geometry) -> list[str]:
    return ["fa"] + [tag for tag, _ in geo.state_tags]


# ---------------------------------------------------------------------------
# Sweep plan shared by both pods.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Config:
    index: int
    pages: tuple[int, ...]  # cycled per request
    inflight: int
    threads: int
    seconds: float

    @property
    def name(self) -> str:
        pages = "+".join(str(p) for p in self.pages)
        return f"c{self.index}-p{pages}-f{self.inflight}-t{self.threads}"


def parse_sweep(text: str, seconds: float) -> list[Config]:
    """``pages:inflight:threads`` items separated by ``;``; pages may be a
    ``/``-separated cycle (``25/150``)."""
    configs = []
    for i, item in enumerate(x for x in text.split(";") if x.strip()):
        parts = item.strip().split(":")
        pages = tuple(int(p) for p in parts[0].split("/"))
        inflight = int(parts[1]) if len(parts) > 1 else 8
        threads = int(parts[2]) if len(parts) > 2 else 4
        configs.append(Config(i, pages, inflight, threads, seconds))
    return configs


def request_budget(config: Config, max_pages_per_s: float) -> int:
    mean_pages = sum(config.pages) / len(config.pages)
    return int(math.ceil(max_pages_per_s * (config.seconds + 30) / mean_pages)) + 16


def request_plan(configs: list[Config], max_pages_per_s: float,
                 verify_pages: int = 0, verify_requests: int = 0):
    """Every request the sweep may pull: (config index, request index, pages)."""
    plan = [(VERIFY_CONFIG, i, verify_pages) for i in range(verify_requests)
            if verify_pages > 0]
    for config in configs:
        for i in range(request_budget(config, max_pages_per_s)):
            plan.append((config.index, i, config.pages[i % len(config.pages)]))
    return plan


def req_name(config_index: int, i: int) -> str:
    return f"c{config_index}-r{i}"


def req_uuid(config_index: int, i: int) -> int:
    return 10_000_000 * (config_index + 1) + i + 1


# ---------------------------------------------------------------------------
# Rank process plumbing.
# ---------------------------------------------------------------------------


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
    """torch_tpu's own topology string for ``world`` local devices, read in a
    child process so the launcher never opens the TPU."""
    import subprocess

    code = (
        "from torch_tpu._internal.utils import hardware;"
        f"print('TOPO=' + str(hardware.get_tpu_topology({world})))"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=300).stdout
    except Exception as exc:  # noqa: BLE001
        log(f"topology probe failed: {exc}")
        return ""
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


def init_rank(rank: int, world: int, launcher_env: dict) -> None:
    # GKE derives multi-slice MegaScale settings from the JobSet's job count;
    # every pod here is its own single-host slice.
    for name in ("MEGASCALE_COORDINATOR_ADDRESS", "MEGASCALE_PORT",
                 "MEGASCALE_NUM_SLICES", "MEGASCALE_SLICE_ID"):
        os.environ.pop(name, None)
    os.environ.update(launcher_env)
    os.environ["TPU_VISIBLE_DEVICES"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
    os.environ["GROUP_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = str(world)
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{launcher_env['MASTER_PORT']}",
        rank=rank,
        world_size=world,
    )


def make_storages(geo: Geometry, device):
    import torch

    rows = geo.page_bytes // 128
    tensors = [
        torch.zeros((geo.num_blocks, rows, 128), dtype=torch.int8, device=device)
        for _ in geo.manifest.storages
    ]
    torch.accelerator.synchronize()
    return tensors


VERIFY_CONFIG = 99


def block_value(block: int, storage: int, rank: int) -> int:
    """The byte every position of a source block holds under --verify-pages."""
    return (block * 7 + storage * 13 + rank * 31) % 120 + 1


def fill_patterns(storages, rank: int, group: int, state_rows) -> None:
    """Writes the source pattern into every row of every source storage.

    Page rows hold block_value of the rank's head group (rank // group), so the
    replicas of a head carry identical pages as the cell's KV does and any of
    them may send a page under head-unique transfer; the GDN state rows, which
    every rank holds for its own heads, hold the rank's own value.
    """
    import torch

    for s, tensor in enumerate(storages):
        n = tensor.shape[0]
        blocks = torch.arange(n, dtype=torch.int32)
        vals = ((blocks * 7 + s * 13 + (rank // group) * 31) % 120 + 1).to(torch.int8)
        for row in state_rows:
            vals[row] = block_value(row, s, rank)
        tensor.copy_(vals.to(tensor.device).view(n, 1, 1).expand(tensor.shape))
    torch.accelerator.synchronize()


def verify_blocks(storages, geo: Geometry, rank: int, group: int, spec: dict) -> dict:
    """Checks landed destination blocks against the source pattern.

    An FA page row must be either untouched (zero: under head-unique transfer
    another rank of the head group lands it) or equal to its source page
    everywhere (the FA region is the whole page); the caller checks that each
    page landed on exactly one rank per group, or on every rank when group is 1.
    A GDN state block must hold its source value in at least the state's live
    bytes.
    """
    import numpy as np
    import torch

    state_live = sum(
        p.live_bytes_per_block for p in geo.manifest.pools
        if p.storage_index == 0 and not str(p.tag).startswith("fa")
    ) // GDN_GROUPS
    bad = []
    checked = 0
    landed = 0
    fa_storages = sorted({geo.manifest.pools[i].storage_index for i in geo.fa_pool_indices})
    state_storages = sorted({p.storage_index for p in geo.manifest.pools
                             if not str(p.tag).startswith("fa")})
    for s in fa_storages:
        dst = [d for d, _ in spec["fa"]]
        rows = storages[s][torch.tensor(dst, dtype=torch.long, device=storages[s].device)]
        rows = rows.cpu().numpy().reshape(len(dst), -1)
        for k, (d, src) in enumerate(spec["fa"]):
            checked += 1
            exp = np.int8(block_value(src, s, rank // group))
            if np.all(rows[k] == exp):
                landed += 1
            elif not np.all(rows[k] == 0):
                bad.append({"kind": "fa", "storage": s, "dst": d, "src": src,
                            "match": int(np.sum(rows[k] == exp))})
    for s in state_storages:
        dst = [d for d, _ in spec["state"]]
        rows = storages[s][torch.tensor(dst, dtype=torch.long, device=storages[s].device)]
        rows = rows.cpu().numpy().reshape(len(dst), -1)
        for k, (d, src) in enumerate(spec["state"]):
            checked += 1
            match = int(np.sum(rows[k] == np.int8(block_value(src, s, rank))))
            if match < state_live:
                bad.append({"kind": "state", "storage": s, "dst": d, "src": src,
                            "match": match, "need": state_live})
    return {"rank": rank, "checked": checked, "fa_landed": landed, "bad": len(bad),
            "first_bad": bad[:4], "fa_storages": len(fa_storages)}


def log(msg: str) -> None:
    print(f"[s3bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * q
    i = int(k)
    j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (k - i)


class BusyLoop:
    """Keeps the rank's TPU core busy with matmuls (prefill stand-in)."""

    def __init__(self, device, n: int):
        import torch

        self._stop = threading.Event()
        self._a = torch.randn((n, n), dtype=torch.bfloat16, device=device)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import torch

        x = self._a
        while not self._stop.is_set():
            for _ in range(8):
                x = torch.matmul(x, self._a)
            torch.accelerator.synchronize()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# d2h mode.
# ---------------------------------------------------------------------------


def d2h_rank(rank: int, args, launcher_env: dict) -> None:
    init_rank(rank, args.ranks, launcher_env)
    import torch
    import torch.distributed as dist
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    device = torch.device("tpu")
    geo = build_geometry(args.num_blocks, mtp=not args.no_mtp)
    storages = make_storages(geo, device)
    manager = KVCacheManager(
        kv_caches=storages,
        local_control_port=0,
        max_blocks=MAX_REQUEST_PAGES,
        num_slots=1,
        node_id=rank,
        parallelism=TP,
    )
    summary = manager.register_pools(geo.manifest.pool_dicts())
    if rank == 0:
        log(f"d2h: pools={summary['pools']} storages={summary['storages']} "
            f"staging={json.dumps(summary['host_staging'])[:400]}")
    busy = BusyLoop(device, args.busy) if args.busy else None
    results = []
    state_pools = [i for i in range(len(geo.manifest.pools)) if i not in geo.fa_pool_indices]
    for pages in [int(p) for p in args.d2h_pages.split(",")]:
        for pattern in args.d2h_patterns.split(","):
            stride = 1 if pattern == "contig" else 2
            if pages * stride > geo.num_blocks:
                continue
            for active in [int(a) for a in args.d2h_active.split(",")]:
                ids = [(i * stride) % geo.num_blocks for i in range(pages)]
                issue, done = [], []
                for it in range(args.iters + 2):
                    dist.barrier()
                    if rank >= active:
                        continue
                    t0 = time.perf_counter()
                    futures = [manager.d2h_pool_blocks(p, ids) for p in geo.fa_pool_indices]
                    if args.d2h_state:
                        futures += [manager.d2h_pool_blocks(p, [geo.num_blocks - 1])
                                    for p in state_pools]
                    t1 = time.perf_counter()
                    for f in futures:
                        f.wait()
                    t2 = time.perf_counter()
                    if it >= 2:
                        issue.append(t1 - t0)
                        done.append(t2 - t0)
                dist.barrier()
                if rank < active:
                    nbytes = pages * geo.page_bytes * len(geo.fa_pool_indices)
                    row = {
                        "rank": rank, "pages": pages, "pattern": pattern,
                        "active_ranks": active, "busy": args.busy,
                        "issue_ms_p50": 1e3 * statistics.median(issue),
                        "done_ms_p50": 1e3 * statistics.median(done),
                        "done_ms_min": 1e3 * min(done),
                        "issue_ms_per_page": 1e3 * statistics.median(issue) / pages,
                        "done_ms_per_page": 1e3 * statistics.median(done) / pages,
                        "gbytes_per_s_rank": nbytes / statistics.median(done) / 1e9,
                        "copies": pages * len(geo.fa_pool_indices) if stride > 1 else len(geo.fa_pool_indices),
                    }
                    results.append(row)
                    if rank == 0:
                        print("RESULT d2h " + json.dumps(row), flush=True)
    if busy:
        busy.stop()
    out = os.path.join(args.out, f"d2h-r{rank}.jsonl")
    with open(out, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    dist.barrier()
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# pair mode, both roles.
# ---------------------------------------------------------------------------


def pod_ip() -> str:
    ip = os.environ.get("POD_IP", "").strip()
    if ip:
        return ip
    return socket.gethostbyname(socket.gethostname())


def resolve(host: str, timeout_s: float = 900.0) -> str:
    deadline = time.time() + timeout_s
    while True:
        try:
            return socket.gethostbyname(host)
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(2)


def wait_port(addr: str, timeout_s: float = 900.0) -> None:
    host, port = addr.rsplit(":", 1)
    deadline = time.time() + timeout_s
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(1)


def pair_rank(rank: int, args, launcher_env: dict, events) -> None:
    init_rank(rank, args.ranks, launcher_env)
    import torch
    import torch.distributed as dist
    from tpu_sync.api.torch import reshard_store as rs
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager
    from tpu_sync.api.torch.reshard_client import ReshardClient

    producer = args.role == "p"
    job = "prefill" if producer else "decode"
    engine = f"{job}-engine"
    ip = pod_ip()
    service_port = args.store_port
    dispatch_port = args.store_port + 100
    store = None
    if rank == 0:
        store = rs.ReshardStore(
            rs.RaidenId(job_name=job, job_replica_id=engine,
                        data_name="reshard_store", data_replica_idx=0),
            ip, dispatch_port, service_port,
            request_registry_ttl_s=float(args.registry_ttl_s),
        )
        log(f"{job} store service={ip}:{store.reshard_service_port} "
            f"dispatch={store.raiden_controller_address}")
    dist.barrier()
    service_addr = f"{ip}:{service_port}"
    dispatch_addr = f"{ip}:{dispatch_port}"
    wait_port(dispatch_addr)
    device = torch.device("tpu")
    geo = build_geometry(args.num_blocks, mtp=not args.no_mtp)
    storages = make_storages(geo, device)
    if producer and args.verify_pages > 0:
        fill_patterns(storages, rank, max(1, args.dst_replica_group_size),
                      [args.num_blocks - 1 - j for j in range(GDN_GROUPS)])
    manager = KVCacheManager(
        kv_caches=storages,
        local_control_port=0,
        max_blocks=MAX_REQUEST_PAGES,
        num_slots=1,
        timeout_s=300.0,
        node_id=rank,
        listener_port=0,
        parallelism=TP if producer else 1,
        raiden_worker_port=0,
        raiden_controller_address=dispatch_addr,
        worker_id=f"worker_{rank}",
    )
    leases = args.p_leases if producer else args.d_leases
    summary = manager.register_pools(
        geo.manifest.pool_dicts(),
        staging_leases=leases,
        staging_blocks_per_pool=geo.staging_hints if leases > 0 else None,
    )
    if rank == 0:
        hs = summary["host_staging"]
        log(f"{job} pools={summary['pools']} storages={summary['storages']} "
            f"staging mode={hs.get('mode')} leases={hs.get('leases')} "
            f"bounded_bytes_per_shard={hs.get('bounded_storage_bytes_per_shard')} "
            f"full_bytes_per_shard={hs.get('full_storage_bytes_per_shard')}")
    client = ReshardClient(service_addr)
    unit = types.SimpleNamespace(**unit_fields(job, engine, rank))
    deadline = time.time() + 300
    while True:
        try:
            client.register_work_unit(
                unit=unit,
                shards=[manager.transfer_address],
                control_plane_rpc_address=manager.listener_address,
                pool_manifest=geo.manifest.pool_dicts(),
                layout_fingerprint=FINGERPRINT,
                page_tokens=PAGE_TOKENS,
                transfer_parallelism=TP,
                transfer_rank=rank,
            )
            break
        except Exception as exc:  # noqa: BLE001
            if time.time() > deadline:
                raise
            log(f"rank{rank} register_work_unit retry: {exc}")
            time.sleep(0.5)
    configs = parse_sweep(args.sweep, args.seconds)
    if producer:
        plan = request_plan(configs, args.max_pages_per_s, args.verify_pages,
                            args.verify_requests)
        t0 = time.time()
        src_ids = list(range(1, 1 + max([max(c.pages) for c in configs]
                                        + [args.verify_pages])))
        state_ids = [geo.num_blocks - 1 - j for j in range(GDN_GROUPS)]
        for config_index, i, pages in plan:
            spans = request_spans(geo, rank, pages, src_ids[:pages], state_ids)
            client.register_request_blocks(
                req_name(config_index, i), req_uuid(config_index, i), unit,
                src_ids[:pages], spans)
        log(f"rank{rank} registered {len(plan)} requests in {time.time() - t0:.1f}s")
    busy = BusyLoop(device, args.busy) if (args.busy and producer) else None
    dist.barrier()
    if rank == 0:
        open(os.path.join(args.out, f"READY-{args.role}"), "w").write(str(time.time()))
    # Completion polling until the sweep is done.
    done_marker = os.path.join(args.out, "SWEEP_DONE")
    sink = open(os.path.join(args.out, f"{'sends' if producer else 'recvs'}-r{rank}.jsonl"), "w")
    give_up = time.time() + args.max_runtime_s
    verify_done = set()
    while not os.path.exists(done_marker):
        if not producer and args.verify_pages > 0:
            for n in range(args.verify_requests + 1):
                req_path = os.path.join(args.out, f"VERIFY-{n}.json")
                if n in verify_done or not os.path.exists(req_path):
                    continue
                try:
                    spec = json.load(open(req_path))
                except ValueError:
                    continue
                verify_done.add(n)
                result = verify_blocks(storages, geo, rank,
                                       max(1, args.dst_replica_group_size), spec)
                with open(os.path.join(args.out, f"VERIFY-{n}-r{rank}.json"), "w") as f:
                    json.dump(result, f)
        if time.time() > give_up:
            log(f"rank{rank} no SWEEP_DONE after {args.max_runtime_s:.0f}s")
            break
        sent, received, failed = manager.poll_stats()
        now = time.time()
        for req in (sent if producer else received):
            sink.write(json.dumps({"req": req, "t": now}) + "\n")
            if events is not None:
                events.put((rank, req, now, True))
        for req in failed:
            sink.write(json.dumps({"req": req, "t": now, "failed": True}) + "\n")
            if events is not None:
                events.put((rank, req, now, False))
        if not (sent or received or failed):
            time.sleep(args.poll_s)
    sink.close()
    if busy:
        busy.stop()
    log(f"rank{rank} exiting")
    sys.stdout.flush()
    os._exit(0)


class Driver:
    """Submits the sweep's pulls from the destination pod."""

    def __init__(self, args, events, p_service: str, d_service: str):
        from tpu_sync.api.torch.reshard_client import ReshardClient

        self.args = args
        self.events = events
        self.p_service = p_service
        self.d_service = d_service
        self.geo = build_geometry(args.num_blocks, mtp=not args.no_mtp)
        self.tags = transfer_tags(self.geo)
        self.src_units = [
            types.SimpleNamespace(**unit_fields("prefill", "prefill-engine", r))
            for r in range(args.ranks)
        ]
        self.dst_units = [
            types.SimpleNamespace(**unit_fields("decode", "decode-engine", r))
            for r in range(args.ranks)
        ]
        self.clients = [ReshardClient(p_service) for _ in range(16)]
        # Head-unique transfer: the replica group size rides on every
        # start_transfer; a replicated run leaves the kwarg out.
        self.transfer_kwargs = (
            {"dst_replica_group_size": int(args.dst_replica_group_size)}
            if int(args.dst_replica_group_size) > 1
            else {}
        )
        self.lock = threading.Condition()
        self.landed: dict[str, set] = {}
        self.failed: set = set()
        self.pending: dict[str, dict] = {}
        self.rows = open(os.path.join(args.out, "requests.jsonl"), "w")
        self.listener = threading.Thread(target=self._listen, daemon=True)
        self.listener.start()

    def _listen(self):
        while True:
            rank, req, t, ok = self.events.get()
            with self.lock:
                if not ok:
                    self.failed.add(req)
                ranks = self.landed.setdefault(req, set())
                ranks.add(rank)
                row = self.pending.get(req)
                if row is not None and (len(ranks) == self.args.ranks or not ok):
                    row["landed"] = t
                    row["ok"] = ok and req not in self.failed
                    self.rows.write(json.dumps(row) + "\n")
                    self.rows.flush()
                    del self.pending[req]
                    self.lock.notify_all()

    def run_verify(self) -> dict:
        """Pulls --verify-requests requests one at a time into fresh blocks and
        has every destination rank check them against the source pattern."""
        args = self.args
        pages = args.verify_pages
        state_base = args.num_blocks - 1 - GDN_GROUPS * args.verify_requests
        client = self.clients[0]
        out = {"verify_requests": args.verify_requests, "pages": pages,
               "landed": 0, "bad": 0, "checked": 0, "errors": []}
        for i in range(args.verify_requests):
            name = req_name(VERIFY_CONFIG, i)
            dst_blocks = [i * pages + k for k in range(pages)]
            state_blocks = [state_base + GDN_GROUPS * i + j for j in range(GDN_GROUPS)]
            dst_ids = list(dst_blocks)
            counts = [pages]
            for tag, ordinal in self.geo.state_tags:
                dst_ids.append(state_blocks[ordinal])
                counts.append(1)
            row = {"config": "verify", "req": name, "pages": pages, "slot": i, "thread": 0,
                   "submit": time.time()}
            with self.lock:
                self.pending[name] = row
            for attempt in range(600):
                try:
                    client.start_transfer(
                        src_units=self.src_units, dst_units=self.dst_units, req_id=name,
                        dst_device_block_ids=dst_ids, dst_mem_type=2, use_block_chunks=True,
                        src_controller_address=self.p_service,
                        dst_controller_address=self.d_service,
                        uuid=req_uuid(VERIFY_CONFIG, i), is_sender=True,
                        num_tokens=pages * PAGE_TOKENS, transfer_pool_tags=self.tags,
                        dst_block_counts=counts, **self.transfer_kwargs)
                    break
                except RuntimeError as exc:
                    if "Missing producer block registration" in str(exc):
                        time.sleep(0.5)
                        continue
                    out["errors"].append(str(exc)[:300])
                    break
            deadline = time.time() + 120
            with self.lock:
                while name in self.pending and time.time() < deadline:
                    self.lock.wait(timeout=1.0)
                landed = name not in self.pending
                self.pending.pop(name, None)
            if not landed:
                out["errors"].append(f"{name} did not land")
                continue
            out["landed"] += 1
            src_state = [args.num_blocks - 1 - j for j in range(GDN_GROUPS)]
            spec = {"fa": [[dst_blocks[k], 1 + k] for k in range(pages)],
                    "state": [[state_blocks[j], src_state[j]] for j in range(GDN_GROUPS)]}
            with open(os.path.join(args.out, f"VERIFY-{i}.json"), "w") as f:
                json.dump(spec, f)
            deadline = time.time() + 600
            paths = [os.path.join(args.out, f"VERIFY-{i}-r{r}.json") for r in range(args.ranks)]
            while not all(os.path.exists(p) for p in paths) and time.time() < deadline:
                time.sleep(1)
            group = max(1, args.dst_replica_group_size)
            landed_by_group: dict[int, int] = {}
            fa_storages = 0
            for p in paths:
                try:
                    res = json.load(open(p))
                except (OSError, ValueError):
                    out["errors"].append(f"no result {os.path.basename(p)}")
                    continue
                out["checked"] += res["checked"]
                out["bad"] += res["bad"]
                fa_storages = res["fa_storages"]
                g = res["rank"] // group
                landed_by_group[g] = landed_by_group.get(g, 0) + res["fa_landed"]
                if res["bad"] and len(out["errors"]) < 8:
                    out["errors"].append(json.dumps(res["first_bad"])[:400])
            # Each page lands once per head group: on exactly one of its ranks under
            # head-unique transfer, on every rank (a group of one) when replicated.
            want = pages * fa_storages
            out.setdefault("fa_landed_by_group", []).append(landed_by_group)
            for g, n in landed_by_group.items():
                if n != want and len(out["errors"]) < 8:
                    out["errors"].append(f"{name}: group {g} landed {n} FA rows, want {want}")
        out["ok"] = (out["landed"] == args.verify_requests and out["bad"] == 0
                     and not out["errors"])
        print("RESULT verify " + json.dumps(out), flush=True)
        return out

    def run_config(self, config: Config) -> dict:
        args = self.args
        budget = request_budget(config, args.max_pages_per_s)
        slots = list(range(config.inflight))
        slot_pages = max(config.pages)
        state_base = args.num_blocks - 1 - GDN_GROUPS * config.inflight
        if slot_pages * config.inflight > state_base:
            raise ValueError("destination pool too small for the in-flight slots")
        next_req = [0]
        submit_rows: list[dict] = []
        t_start = time.time()
        t_end = t_start + config.seconds
        stop = threading.Event()

        def worker(tid: int):
            client = self.clients[tid % len(self.clients)]
            while not stop.is_set():
                with self.lock:
                    while len(self.pending) >= config.inflight and not stop.is_set():
                        self.lock.wait(timeout=0.5)
                    if stop.is_set() or time.time() >= t_end:
                        return
                    i = next_req[0]
                    if i >= budget:
                        return
                    next_req[0] += 1
                    used = {r["slot"] for r in self.pending.values()}
                    slot = next(s for s in slots if s not in used)
                    pages = config.pages[i % len(config.pages)]
                    name = req_name(config.index, i)
                    row = {"config": config.name, "req": name, "pages": pages,
                           "slot": slot, "thread": tid}
                    self.pending[name] = row
                dst_blocks = [slot * slot_pages + k for k in range(pages)]
                state_blocks = [state_base + GDN_GROUPS * slot + j for j in range(GDN_GROUPS)]
                dst_ids = list(dst_blocks)
                counts = [pages]
                for tag, ordinal in self.geo.state_tags:
                    dst_ids.append(state_blocks[ordinal])
                    counts.append(1)
                row["submit"] = time.time()
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        ok = client.start_transfer(
                            src_units=self.src_units,
                            dst_units=self.dst_units,
                            req_id=name,
                            dst_device_block_ids=dst_ids,
                            dst_mem_type=2,
                            use_block_chunks=True,
                            src_controller_address=self.p_service,
                            dst_controller_address=self.d_service,
                            uuid=req_uuid(config.index, i),
                            is_sender=True,
                            num_tokens=pages * PAGE_TOKENS,
                            transfer_pool_tags=self.tags,
                            dst_block_counts=counts,
                            **self.transfer_kwargs,
                        )
                        row["accepted"] = bool(ok)
                        break
                    except RuntimeError as exc:
                        if "Missing producer block registration" in str(exc) and attempt < 600:
                            time.sleep(0.5)
                            continue
                        row["error"] = str(exc)[:300]
                        break
                row["submitted"] = time.time()
                row["attempts"] = attempt
                with self.lock:
                    submit_rows.append(row)
                    if "error" in row and name in self.pending:
                        del self.pending[name]
                        self.rows.write(json.dumps(row) + "\n")
                        self.lock.notify_all()

        threads = [threading.Thread(target=worker, args=(t,), daemon=True)
                   for t in range(config.threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        drain_deadline = time.time() + args.drain_s
        with self.lock:
            while self.pending and time.time() < drain_deadline:
                self.lock.wait(timeout=1.0)
            stuck = list(self.pending)
        stop.set()
        return self._summarize(config, t_start, submit_rows, stuck)

    def _summarize(self, config, t_start, submit_rows, stuck) -> dict:
        rows = [r for r in submit_rows if "landed" in r and r.get("ok")]
        errors = [r for r in submit_rows if "error" in r]
        warm = self.args.warmup_s
        steady = [r for r in rows if r["submit"] >= t_start + warm]
        out = {"config": config.name, "pages": list(config.pages),
               "replica_group_size": self.args.dst_replica_group_size,
               "inflight": config.inflight, "threads": config.threads,
               "submitted": len(submit_rows), "landed": len(rows),
               "errors": len(errors), "stuck": len(stuck),
               "first_error": errors[0]["error"] if errors else ""}
        if steady:
            t0 = min(r["submit"] for r in steady)
            t1 = max(r["landed"] for r in steady)
            pages = sum(r["pages"] for r in steady)
            submit_ms = [1e3 * (r["submitted"] - r["submit"]) for r in steady]
            per_page = [1e3 * (r["submitted"] - r["submit"]) / r["pages"] for r in steady]
            e2e = [1e3 * (r["landed"] - r["submit"]) for r in steady]
            out.update({
                "steady_requests": len(steady),
                "steady_window_s": t1 - t0,
                "pages_per_s": pages / max(t1 - t0, 1e-9),
                "pages_per_min": 60 * pages / max(t1 - t0, 1e-9),
                "gbytes_per_s_host": pages * self.geo.page_bytes
                * len(self.geo.fa_pool_indices) * self.args.ranks / max(t1 - t0, 1e-9) / 1e9,
                # FA bytes on the wire: a head-unique transfer moves each page
                # from one replica of its head group.
                "wire_gbytes_per_s_host": pages * self.geo.page_bytes
                * len(self.geo.fa_pool_indices) * self.args.ranks
                / max(1, self.args.dst_replica_group_size) / max(t1 - t0, 1e-9) / 1e9,
                "submit_ms_p10": pct(submit_ms, 0.1), "submit_ms_p50": pct(submit_ms, 0.5),
                "submit_ms_per_page_p10": pct(per_page, 0.1),
                "submit_ms_per_page_p50": pct(per_page, 0.5),
                "e2e_ms_p50": pct(e2e, 0.5), "e2e_ms_p90": pct(e2e, 0.9),
            })
        print("RESULT pair " + json.dumps(out), flush=True)
        return out


def launcher(args) -> int:
    import multiprocessing as mp

    os.makedirs(args.out, exist_ok=True)
    env_info = {k: v for k, v in os.environ.items()
                if k.startswith(("TPU_", "RAIDEN_", "TORCH_TPU", "NODE_NAME", "POD_"))}
    try:
        from importlib.metadata import version
        env_info["versions"] = {n: version(n) for n in ("torch", "torch-tpu", "tpu-sync-torch", "libtpu")}
    except Exception as exc:  # noqa: BLE001
        env_info["versions_error"] = str(exc)
    env_info["argv"] = sys.argv
    with open(os.path.join(args.out, f"env-{args.mode}-{args.role or 'x'}.json"), "w") as f:
        json.dump(env_info, f, indent=1, sort_keys=True)
    log(f"node={os.environ.get('NODE_NAME', '?')} mode={args.mode} role={args.role} "
        f"versions={env_info.get('versions')}")
    ctx = mp.get_context("spawn")
    launcher_env = prepare_launcher_env(args.ranks)
    if args.mode == "d2h":
        procs = [ctx.Process(target=d2h_rank, args=(r, args, launcher_env))
                 for r in range(args.ranks)]
        for p in procs:
            p.start()
        rc = 0
        for p in procs:
            p.join()
            rc = rc or (p.exitcode or 0)
        print(f"RESULT: {'PASS' if rc == 0 else 'FAIL'} (d2h ranks exit {rc})", flush=True)
        return rc
    events = ctx.Queue() if args.role == "d" else None
    procs = [ctx.Process(target=pair_rank, args=(r, args, launcher_env, events))
             for r in range(args.ranks)]
    for p in procs:
        p.start()
    if args.role == "p":
        # Source pod: ranks serve until the destination writes SWEEP_DONE.
        rc = 0
        for p in procs:
            p.join()
            rc = rc or (p.exitcode or 0)
        print(f"RESULT: {'PASS' if rc == 0 else 'FAIL'} (p ranks exit {rc})", flush=True)
        return rc
    ready_d = os.path.join(args.out, "READY-d")
    ready_p = os.path.join(args.out, "READY-p")
    deadline = time.time() + args.ready_timeout_s
    while not (os.path.exists(ready_d) and os.path.exists(ready_p)):
        if any(p.exitcode not in (None, 0) for p in procs) or time.time() > deadline:
            open(os.path.join(args.out, "SWEEP_DONE"), "w").write("abort")
            print("RESULT: FAIL (ranks not ready)", flush=True)
            return 1
        time.sleep(2)
    p_ip = resolve(args.p_host)
    p_service = f"{p_ip}:{args.store_port}"
    d_service = f"{pod_ip()}:{args.store_port}"
    log(f"driver: source={p_service} destination={d_service}")
    driver = Driver(args, events, p_service, d_service)
    results = []
    verify = driver.run_verify() if args.verify_pages > 0 else {"ok": True}
    for config in parse_sweep(args.sweep, args.seconds):
        log(f"config {config.name}")
        results.append(driver.run_config(config))
        with open(os.path.join(args.out, "results.jsonl"), "a") as f:
            f.write(json.dumps(results[-1]) + "\n")
        time.sleep(args.gap_s)
    open(os.path.join(args.out, "SWEEP_DONE"), "w").write(str(time.time()))
    rc = 0
    for p in procs:
        p.join(timeout=120)
        if p.exitcode is None:
            p.kill()
        rc = rc or (p.exitcode or 0)
    bad = [r for r in results if r.get("errors") or r.get("stuck") or not r.get("landed")]
    ok = rc == 0 and not bad and verify["ok"]
    print(f"RESULT: {'PASS' if ok else 'FAIL'} ({len(results)} configs, "
          f"{len(bad)} with errors or stuck transfers, verify ok={verify['ok']}, "
          f"ranks exit {rc})", flush=True)
    return 0 if ok else 1


def check(args) -> int:
    geo = build_geometry(args.num_blocks, mtp=not args.no_mtp)
    from collections import Counter

    tags = Counter(str(p.tag) for p in geo.manifest.pools)
    print(f"storages={len(geo.manifest.storages)} pools={len(geo.manifest.pools)} "
          f"page_bytes={geo.page_bytes} fa_pools={len(geo.fa_pool_indices)} tags={dict(tags)}")
    for pool in geo.manifest.pools[:8]:
        print(" ", pool.tag, pool.layer_name, "storage", pool.storage_index, "base",
              pool.base_offset_bytes, "stride", pool.block_stride_bytes,
              "live", pool.live_bytes_per_block,
              [(r.name, r.offset_bytes, r.stride_bytes, r.unit_bytes, r.num_units,
                r.units_per_stride) for r in pool.regions])
    print("staging hints", sorted(Counter(geo.staging_hints).items()))
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

    for rank in (0, 5):
        spans = request_spans(geo, rank, 150, list(range(1, 151)),
                              [geo.num_blocks - 1 - j for j in range(GDN_GROUPS)])
        for reg in spans:
            print(f"rank{rank} tag={reg.tag} blocks={len(reg.block_ids)} spans={len(reg.spans)} "
                  f"declared={reg.declared_bytes} src_bytes={sol.registration_source_bytes(reg)} "
                  f"dst_ordinals={sorted({getattr(s, 'dst_unit_ordinal', None) for s in reg.spans}, key=str)}")
    print("transfer tags", transfer_tags(geo))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("check", "d2h", "pair"), required=True)
    parser.add_argument("--role", choices=("p", "d"), default=None)
    parser.add_argument("--out", default="/tmp/s3bench")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--busy", type=int, default=0,
                        help="matmul edge for a TPU busy loop on source ranks (0 = idle)")
    # d2h
    parser.add_argument("--d2h-pages", default="25,150")
    parser.add_argument("--d2h-patterns", default="scatter,contig")
    parser.add_argument("--d2h-active", default="1,8")
    parser.add_argument("--d2h-state", action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    # pair
    parser.add_argument("--p-host", default="")
    parser.add_argument("--store-port", type=int, default=27000)
    parser.add_argument("--p-leases", type=int, default=2)
    parser.add_argument("--d-leases", type=int, default=8)
    parser.add_argument("--sweep", default="150:8:4")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup-s", type=float, default=10.0)
    parser.add_argument("--gap-s", type=float, default=5.0)
    parser.add_argument("--drain-s", type=float, default=120.0)
    parser.add_argument("--max-pages-per-s", type=float, default=900.0)
    parser.add_argument("--registry-ttl-s", type=float, default=7200.0)
    parser.add_argument("--poll-s", type=float, default=0.002)
    parser.add_argument("--ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-runtime-s", type=float, default=3 * 3600.0)
    parser.add_argument("--verify-pages", type=int, default=0,
                        help="pages per byte-checked request before the sweep (0 = off)")
    parser.add_argument("--verify-requests", type=int, default=2)
    parser.add_argument("--dst-replica-group-size", type=int, default=1,
                        help="head-unique transfer: destination replica group size "
                             "passed to start_transfer (1 = replicated)")
    args = parser.parse_args()
    if args.mode == "check":
        return check(args)
    if args.mode == "pair" and args.role is None:
        parser.error("--mode pair needs --role")
    if args.mode == "pair" and args.role == "d" and not args.p_host:
        parser.error("--role d needs --p-host")
    return launcher(args)


if __name__ == "__main__":
    sys.exit(main())
