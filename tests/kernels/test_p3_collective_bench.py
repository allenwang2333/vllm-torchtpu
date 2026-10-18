# SPDX-License-Identifier: Apache-2.0
"""ICI collective microbench by group, operation and message size (phase 3 harness).

Measures all-reduce (psum), all-gather, reduce-scatter (psum_scatter), all-to-all
and ppermute (ring shift by one) on real TPU ranks with jax.shard_map, for the
groups the parallelism candidates use:

  chiplet2    the two TensorCores of one chip (isolated: only they take part)
  chiplet2x   every chip's two cores at once, as independent 2-rank groups
  chip4       four cores (two chips) of one host
  chip4x      all 4-core groups of the slice at once
  host8       the eight cores of one host
  host16      two hosts (2x2x2 slice)
  host32      four hosts (2x2x4 slice)
  longaxis    one core per host in a ring (ppermute along the host axis)

Message size is the per-rank buffer: the per-rank input of all-reduce,
reduce-scatter, all-to-all and ppermute, and the per-rank output of all-gather,
so the five operations move comparable data. Sizes run from 64 KiB to 512 MiB
in powers of two (P3_COLL_SIZES overrides, in KiB). Bytes moved per rank, the
basis of the effective bandwidth column, follow the ring conventions:
all-reduce 2 (g-1)/g S, all-gather and reduce-scatter and all-to-all (g-1)/g S,
ppermute S; the bus-bandwidth number is what the hardware moved, the
algorithmic one is S / time.

Timing: the operation is chained D times inside one jitted program with a data
dependence between iterations, so nothing overlaps or is elided; D grows for
small messages so every program runs for tens of milliseconds, and the host
wall of block_until_ready around it, divided by D, is the per-operation time
(median of 5 after 2 warmups). The 64 KiB row is the launch-latency floor. A
jax.profiler trace of three calls is saved for a subset of sizes
(P3_COLL_TRACE_SIZES, KiB) under <out_dir>/trace/ for
scripts/vllm/benchmark/p3_xplane_device_time.py, which gives the per-core
device time of the same chain. Sub-millisecond operations are never timed one
at a time from the host.

Multi-host: every host runs this same module (SPMD); only process 0 writes
results.jsonl and env.json, every process saves its own trace. Collected only
with P3_BENCH=1; needs at least 8 devices.
"""

from __future__ import annotations

import functools
import gc
import json
import os
import statistics
import sys
import time

import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("phase 3 harness bench; set P3_BENCH=1 to run", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "vllm", "benchmark"))
import p3_bench_common as common  # noqa: E402

try:
    from jax import shard_map as _shard_map
except ImportError:  # older jax
    from jax.experimental.shard_map import shard_map as _shard_map

OUT = os.environ.get("P3_COLL_OUT") or common.run_dir("collective")
KIB = 1024
SIZES_KIB = [int(x) for x in os.environ.get(
    "P3_COLL_SIZES", ",".join(str(64 * 2 ** i) for i in range(14))).split(",") if x]
TRACE_KIB = {int(x) for x in os.environ.get("P3_COLL_TRACE_SIZES", "64,1024,16384,131072").split(",") if x}
OPS = [x for x in os.environ.get("P3_COLL_OPS", "all_reduce,all_gather,reduce_scatter,all_to_all,ppermute").split(",") if x]
GROUPS = [x for x in os.environ.get("P3_COLL_GROUPS", "").split(",") if x]
HOST_ITERS = 5
TRACE_ITERS = 3
if os.path.isdir("/workspace/tmp"):
    jax.config.update("jax_compilation_cache_dir", "/workspace/tmp/p3-harness/jax-cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _chips(devs):
    """Devices grouped by chip (same coords), each ordered by core_on_chip."""
    groups: dict = {}
    for d in devs:
        groups.setdefault(tuple(getattr(d, "coords", (d.id,))), []).append(d)
    for g in groups.values():
        g.sort(key=lambda d: getattr(d, "core_on_chip", d.id))
    return [groups[k] for k in sorted(groups)]


def _devices():
    """Returns global JAX devices after optional multi-host initialization."""
    if not jax.distributed.is_initialized():
        addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
        nproc = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
        if addr and nproc > 1:
            jax.distributed.initialize(
                coordinator_address=addr,
                num_processes=nproc,
                process_id=int(os.environ.get("JAX_PROCESS_ID", "0")),
            )
    return jax.devices()


def _groups() -> dict[str, tuple[np.ndarray, str]]:
    """name -> (device array of shape [independent groups, group size], collective axis)."""
    devs = _devices()
    by_proc: dict[int, list] = {}
    for d in devs:
        by_proc.setdefault(d.process_index, []).append(d)
    local = by_proc[min(by_proc)]
    chips = _chips(local)
    out: dict[str, tuple[np.ndarray, str]] = {}
    if len(chips[0]) >= 2:
        out["chiplet2"] = (np.array([chips[0][:2]]), "g")
        all_pairs = [c[:2] for c in _chips(devs) if len(c) >= 2]
        if len(all_pairs) >= 2:
            out["chiplet2x"] = (np.array(all_pairs), "g")
    if len(chips) >= 2 and len(chips[0]) >= 2:
        out["chip4"] = (np.array([chips[0][:2] + chips[1][:2]]), "g")
        quads = [c[:2] for c in _chips(devs) if len(c) >= 2]
        quads = [quads[i] + quads[i + 1] for i in range(0, len(quads) - 1, 2)]
        if len(quads) >= 2:
            out["chip4x"] = (np.array(quads), "g")
    if len(local) >= 8:
        out["host8"] = (np.array([local[:8]]), "g")
    if len(devs) >= 16:
        out["host16"] = (np.array([devs[:16]]), "g")
    if len(devs) >= 32:
        out["host32"] = (np.array([devs[:32]]), "g")
    if len(by_proc) >= 2:
        ring = [by_proc[p][0] for p in sorted(by_proc)]
        out["longaxis"] = (np.array([ring]), "g")
    return out


def _mesh(arr: np.ndarray) -> Mesh:
    return Mesh(arr, ("i", "g"))


def _bytes_moved(op: str, g: int, size: int) -> int:
    if op == "all_reduce":
        return 2 * (g - 1) * size // g
    if op in ("all_gather", "reduce_scatter", "all_to_all"):
        return (g - 1) * size // g
    return size  # ppermute


def _chain_depth(size: int) -> int:
    if size <= 128 * KIB:
        return 1000
    if size <= 4 * KIB * KIB:
        return 200
    return 20


def _shmap(body, mesh, spec):
    try:
        return _shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
    except TypeError:
        return _shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec, check_rep=False)


def _program(op: str, mesh: Mesh, g: int, size: int, depth: int):
    """Returns (jitted fn, per-rank input array, per-rank input bytes).

    The per-rank buffer is [rows, 1024] f32 with rows a multiple of g, so
    reduce-scatter and all-to-all split it evenly; the recorded input bytes
    are the rounded size.
    """
    n = size // 4  # f32 elements of the per-rank buffer (all-gather: of the output)
    lanes = 1024
    rows = n // g // lanes if op == "all_gather" else n // lanes
    rows = max(g, -(-rows // g) * g)
    shape_in = (rows, lanes)
    perm = [(k, (k + 1) % g) for k in range(g)]
    inv_g = 1.0 / g

    def body(x):
        def step(_, x):
            if op == "all_reduce":
                return lax.psum(x, "g") * inv_g
            if op == "all_gather":
                full = lax.all_gather(x, "g", axis=0, tiled=True)
                k = lax.axis_index("g")
                return lax.dynamic_slice_in_dim(full, k * x.shape[0], x.shape[0], 0) * 0.5 + x * 0.5
            if op == "reduce_scatter":
                part = lax.psum_scatter(x, "g", scatter_dimension=0, tiled=True) * inv_g
                return jnp.concatenate([part] * g, axis=0)
            if op == "all_to_all":
                return lax.all_to_all(x, "g", split_axis=0, concat_axis=0, tiled=True)
            return lax.ppermute(x, "g", perm)

        return lax.fori_loop(0, depth, step, x)

    spec = P(("i", "g"))
    fn = _shmap(body, mesh, spec)
    n_dev = mesh.devices.size
    global_shape = (n_dev * rows, lanes)
    x = jax.make_array_from_callback(
        global_shape, NamedSharding(mesh, spec),
        lambda idx: np.ones((rows, lanes), np.float32),
    )
    return jax.jit(fn), x, rows * lanes * 4


_ENV: dict = {}


def _record(row: dict) -> None:
    if jax.process_index() != 0:
        return
    if not _ENV:
        os.makedirs(OUT, exist_ok=True)
        _ENV.update(common.write_env(OUT, {"bench": "collective"}, devices=True))
    row = {"environment_id": _ENV["id"], **row}
    common.append_jsonl(os.path.join(OUT, "results.jsonl"), row)
    print("COLLECTIVE " + json.dumps(row), flush=True)


def _trace_dir(name: str) -> str:
    return os.path.join(OUT, "trace", f"{name}_p{jax.process_index()}")


@functools.lru_cache(maxsize=None)
def _all_groups():
    groups = _groups()
    if GROUPS:
        groups = {k: v for k, v in groups.items() if k in GROUPS}
    return groups


_CANDIDATE_GROUPS = (
    "chiplet2", "chiplet2x", "chip4", "chip4x", "host8", "host16", "host32", "longaxis",
)


def _params():
    # Enumerate candidate groups statically so pytest collection never initializes JAX.
    names = [g for g in _CANDIDATE_GROUPS if not GROUPS or g in GROUPS]
    out = []
    for name in names:
        for op in OPS:
            if name == "longaxis" and op != "ppermute":
                continue
            for kib in SIZES_KIB:
                out.append(pytest.param(name, op, kib, id=f"{name}-{op}-{kib}k"))
    return out


def test_devices():
    if len(_devices()) < 8:
        pytest.skip("needs at least 8 TPU devices")
    _record({"label": "devices", "groups": {k: [[d.id for d in row] for row in v[0]]
                                             for k, v in _all_groups().items()},
             "jax": jax.__version__, "process_count": jax.process_count()})


@pytest.mark.parametrize("group,op,kib", _params())
def test_collective(group, op, kib):
    if len(_devices()) < 8:
        pytest.skip("needs at least 8 TPU devices")
    groups = _all_groups()
    if group not in groups:
        pytest.skip(f"group {group} is unavailable on {len(_devices())} devices")
    arr, axis = groups[group]
    mesh = _mesh(arr)
    g = arr.shape[1]
    size = kib * KIB
    depth = _chain_depth(size)
    t_build = time.time()
    fn, x, in_bytes = _program(op, mesh, g, size, depth)
    try:
        jax.block_until_ready(fn(x))
    except Exception as e:  # noqa: BLE001
        if arr.size < len(jax.local_devices()) and "UNIMPLEMENTED: HLO all-reduce-start" in str(e):
            _record({"group": group, "op": op, "ranks": g, "concurrent_groups": int(arr.shape[0]),
                     "size_bytes": size, "size_kib": kib,
                     "unsupported": f"{type(e).__name__}: {str(e)[:300]}"})
            pytest.skip(f"XLA TPU compiler limitation on partial-device sub-mesh for {group} {op} {kib}k")
        _record({"group": group, "op": op, "ranks": g, "concurrent_groups": int(arr.shape[0]),
                 "size_bytes": size, "failed": f"{type(e).__name__}: {str(e)[:300]}"})
        pytest.fail(f"{group} {op} {kib}k: {e}")
    compile_s = time.time() - t_build
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(HOST_ITERS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append((time.perf_counter() - t0) * 1e6 / depth)
    med, best = statistics.median(ts), min(ts)
    trace_dir = None
    if kib in TRACE_KIB:
        trace_dir = _trace_dir(f"{group}_{op}_{kib}k")
        jax.profiler.start_trace(trace_dir)
        for _ in range(TRACE_ITERS):
            jax.block_until_ready(fn(x))
            time.sleep(0.005)
        jax.profiler.stop_trace()
    moved = _bytes_moved(op, g, size)
    _record({
        "group": group, "op": op, "ranks": g, "concurrent_groups": int(arr.shape[0]),
        "size_bytes": size, "size_kib": kib, "input_bytes_per_rank": in_bytes,
        "bytes_moved_per_rank": moved, "chain_depth": depth,
        "host_us_per_op_median": round(med, 2), "host_us_per_op_min": round(best, 2),
        "bus_GBps_median": round(moved / (med * 1e-6) / 1e9, 2),
        "alg_GBps_median": round(size / (med * 1e-6) / 1e9, 2),
        "compile_s": round(compile_s, 2), "trace_dir": trace_dir,
        "devices": [[d.id for d in row] for row in arr],
    })
    del fn, x
    gc.collect()
