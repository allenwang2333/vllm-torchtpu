"""Sub-group collective independence probe: eight TPU cores as four device pairs.

Groups created on every rank, in this order: world (tpu_dist), a gloo copy of the
world for host-only barriers, pairs {0,1},{2,3},{4,5},{6,7}, columns {0,2,4,6},{1,3,5,7}.
Tests (one MICRO line per rank per test, PASS/FAIL, timings in ms):
  T0   world all-reduce, then one lockstep pair and one lockstep column all-reduce
  T4   timing: pair all-gather of [8192, 2880] bf16; world permute (jax ppermute or,
       with --permute a2a, a uniform world all-to-all of the same block)
  T1   pair k runs 3+k pair all-reduces, then 3+k pair all-gathers (skewed counts)
  T2   --iters x [pair k does k+1 pair all-reduces, then everyone one world all-reduce]
  T3   one fused program (matmul + pair all-reduce + pair all-gather), 2+k launches per
       pair, then one world all-reduce
  T3c  the same body under torch.compile
A rank that makes no checkpoint progress for --hang-secs dies by SIGALRM; the parent
kills the rest at --timeout and exits non-zero on any non-zero rank exit code.
"""
import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

WORLD = 8
PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
COLS = ((0, 2, 4, 6), (1, 3, 5, 7))
AG_ROWS, AG_COLS = 8192, 2880


def _log(rank, msg):
    print(f"MICRO rank{rank} {msg}", flush=True)


class Watchdog:
    """Prints STUCK lines after `secs` without a checkpoint; SIGALRM after `hang` seconds."""

    def __init__(self, rank, secs, hang):
        import threading
        self.rank, self.secs, self.hang = rank, secs, hang
        self.mark, self.last = "start", time.monotonic()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def at(self, mark):                      # main thread only (signal.alarm)
        self.mark, self.last = mark, time.monotonic()
        signal.alarm(self.hang)              # default SIGALRM action terminates the process

    def _run(self):
        while not self._stop.wait(10):
            idle = time.monotonic() - self.last
            if idle > self.secs:
                _log(self.rank, f"STUCK for {int(idle)} s at {self.mark}")

    def stop(self):
        self._stop.set()
        signal.alarm(0)


def build_world_permute(dev_ids):
    """x_out = permute(x): rank r's block goes to rank r+1 (mod 8) as one collective-permute
    over the slice, exported like sglang's pp_wave (jax_op over a mesh of abstract devices)."""
    import jax
    import numpy as np
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec
    from torch_tpu._internal.pallas import jax_op
    from torch_tpu._internal.utils.hardware import get_tpu_device_name

    name = get_tpu_device_name()
    kind = {"TPU v7": "TPU7x", "TPU v6e": "TPU v6 lite"}.get(name, name)

    class _Client:                            # compile-only stand-in JAX inspects on mesh devices
        platform, platform_version, runtime_type = "tpu", name, "tpu"
        def process_index(self): return 0
        def device_count(self): return WORLD
        def devices(self): return []
        def local_devices(self): return []

    class _Device:
        def __init__(self, i):
            self.id, self.device_kind, self.num_cores = i, kind, 1
            self.platform, self.process_index, self.client = "tpu", 0, _Client()
        def __eq__(self, o): return isinstance(o, _Device) and o.id == self.id
        def __hash__(self): return hash(self.id)
        def __repr__(self): return f"AbstractDevice({self.id})"

    mesh = Mesh(np.asarray([_Device(i) for i in range(WORLD)]), ("w",))
    perm = [(dev_ids[r], dev_ids[(r + 1) % WORLD]) for r in range(WORLD)]   # mesh position = device id
    spec = PartitionSpec()

    def shift(x: jax.Array) -> jax.Array:
        return shard_map(lambda xb: jax.lax.ppermute(xb, "w", perm), mesh=mesh,
                         in_specs=(spec,), out_specs=spec, check_rep=False)(x)

    return jax_op("probe::world_permute", shift)


def worker(rank, port, args):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD),
                      LOCAL_WORLD_SIZE=str(WORLD), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    import torch
    import torch_tpu  # noqa: F401  (registers the "tpu" device and the tpu_dist backend)
    import torch.distributed as dist
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.sync import sync

    tests = set(args.tests.split(","))
    wd = Watchdog(rank, args.stuck_secs, args.hang_secs)
    wd.at("init_process_group")
    dist.init_process_group(backend="tpu_dist")          # env://; opens this rank's single TPU device
    cpu = dist.new_group(backend="gloo")                  # host-only barriers, never a device launch
    wd.at("new_group pairs/columns")
    pair_groups = [dist.new_group(ranks=list(p)) for p in PAIRS]   # every rank, same order
    col_groups = [dist.new_group(ranks=list(c)) for c in COLS]
    k = rank // 2
    my_pair, my_col = pair_groups[k], col_groups[rank % 2]
    if args.standalone:
        # torch_tpu#4387: every member of a group calls this before the group's
        # first collective; the pair's programs then compile for the pair only.
        from torch_tpu._internal.distributed import spmd_util as _spmd
        wd.at("make_standalone(pair)")
        _spmd.make_standalone(my_pair)
        _log(rank, f"make_standalone({PAIRS[k]}) ok")
    pair_sum = PAIRS[k][0] + PAIRS[k][1]
    dev = torch.device("tpu")
    dev_ids = [None] * WORLD
    dist.all_gather_object(dev_ids, int(tpu_distributed.global_device_id()), group=cpu)
    _log(rank, f"init ok: device {dev_ids[rank]}, rank->device {dev_ids}, pair {PAIRS[k]}, column {COLS[rank % 2]}")
    results = {}

    def done(t):                              # wait for a device tensor without a large D2H copy
        sync.synchronize([t], wait=True)
        return t

    def barrier(tag):
        wd.at(f"barrier {tag}")
        dist.barrier(group=cpu)

    def full(v, shape=(256, 256), dtype=torch.float32):
        return torch.full(shape, float(v), dtype=dtype, device=dev)

    def same(t, v):
        t = t.cpu()
        return bool(torch.equal(t, torch.full_like(t, float(v))))

    def record(name, ok, detail):
        results[name] = bool(ok)
        _log(rank, f"{name}: {detail} -> {'PASS' if ok else 'FAIL'}")

    def ms_list(ts):
        return [round(t, 2) for t in ts]

    # --- T0: sanity, everything lockstep ---------------------------------------------
    if "T0" in tests:
        barrier("T0")
        wd.at("T0 world all-reduce")
        t0 = time.perf_counter()
        x = torch.tensor([1.0, float(rank), float(rank ** 2)], device=dev)
        dist.all_reduce(x)
        got = x.cpu().tolist()
        ms_world = 1e3 * (time.perf_counter() - t0)
        ok = got == [8.0, 28.0, 140.0]
        wd.at("T0 pair all-reduce")
        y = full(rank + 1); dist.all_reduce(y, group=my_pair); ok &= same(y, pair_sum + 2)
        wd.at("T0 column all-reduce")
        z = full(rank + 1); dist.all_reduce(z, group=my_col); ok &= same(z, sum(COLS[rank % 2]) + 4)
        record("T0", ok, f"world {got} ({ms_world:.1f} ms incl. first compile), pair and column all-reduce lockstep")

    # --- T4: timing (lockstep on every arm) ---------------------------------------------
    if "T4" in tests:
        barrier("T4")
        xb = torch.full((AG_ROWS, AG_COLS), float(rank + 1), dtype=torch.bfloat16, device=dev)
        out = torch.empty((2 * AG_ROWS, AG_COLS), dtype=torch.bfloat16, device=dev)

        def pair_ag():
            dist.all_gather_into_tensor(out, xb, group=my_pair)
            return out

        wd.at("T4 pair all-gather warm-up")
        done(pair_ag()); o = done(pair_ag()).cpu()
        ok = same(o[:AG_ROWS], PAIRS[k][0] + 1) and same(o[AG_ROWS:], PAIRS[k][1] + 1)
        barrier("T4 pair all-gather timed")
        ts = []
        for _ in range(args.iters):
            wd.at("T4 pair all-gather timed")
            t0 = time.perf_counter(); done(pair_ag()); ts.append(1e3 * (time.perf_counter() - t0))
        ts.sort()
        mb = AG_ROWS * AG_COLS * 2 / 1e6
        record("T4-pair-ag", ok, f"[{AG_ROWS},{AG_COLS}] bf16, {mb:.1f} MB received/device: "
                                 f"median {ts[len(ts) // 2]:.2f} ms, min {ts[0]:.2f} ms, max {ts[-1]:.2f} ms")
        wd.at("T4 world permute build")
        try:
            if args.permute == "ppermute":
                op = build_world_permute(dev_ids)
                label = "world ppermute (jax_op, closed 8-cycle in rank order)"
                def world_perm(): return op(xb)
                def check(y): return same(y, ((rank - 1) % WORLD) + 1)
            else:
                a2a = torch.empty_like(xb)
                label = "world uniform all-to-all (permute proxy)"
                def world_perm(): dist.all_to_all_single(a2a, xb); return a2a
                def check(y):
                    y = y.cpu(); n = AG_ROWS // WORLD
                    return all(same(y[j * n:(j + 1) * n], j + 1) for j in range(WORLD))
            done(world_perm()); okp = check(done(world_perm()))
            barrier("T4 world permute timed")
            tp = []
            for _ in range(args.iters):
                wd.at("T4 world permute timed")
                t0 = time.perf_counter(); done(world_perm()); tp.append(1e3 * (time.perf_counter() - t0))
            tp.sort()
            record("T4-world-permute", okp, f"{label}: median {tp[len(tp) // 2]:.2f} ms, "
                                            f"min {tp[0]:.2f} ms, max {tp[-1]:.2f} ms")
        except Exception as e:  # noqa: BLE001  (a build failure must not abort the remaining tests)
            record("T4-world-permute", False, f"FAILED {type(e).__name__}: {str(e)[:300]}")
        del xb, out

    # --- T1: skewed counts of pair collectives -----------------------------------------
    if "T1" in tests:
        barrier("T1")
        n = 3 + k
        ok, t_ar, t_ag = True, [], []
        for i in range(n):
            wd.at(f"T1 pair all-reduce {i + 1}/{n}")
            x = full(rank + 1 + 10 * i)
            t0 = time.perf_counter(); dist.all_reduce(x, group=my_pair); done(x)
            t_ar.append(1e3 * (time.perf_counter() - t0))
            ok &= same(x, pair_sum + 2 + 20 * i)
        for i in range(n):
            wd.at(f"T1 pair all-gather {i + 1}/{n}")
            x = full(rank + 1 + 10 * i)
            out = torch.empty((512, 256), dtype=torch.float32, device=dev)
            t0 = time.perf_counter(); dist.all_gather_into_tensor(out, x, group=my_pair); done(out)
            t_ag.append(1e3 * (time.perf_counter() - t0))
            o = out.cpu()
            ok &= same(o[:256], PAIRS[k][0] + 1 + 10 * i) and same(o[256:], PAIRS[k][1] + 1 + 10 * i)
        record("T1", ok, f"{n} pair all-reduces {ms_list(t_ar)} ms, {n} pair all-gathers {ms_list(t_ag)} ms")

    # --- T2: interleaved skewed pair collectives and world collectives -----------------
    if "T2" in tests:
        barrier("T2")
        ok, per_iter = True, []
        for it in range(args.iters):
            t0 = time.perf_counter()
            x = full(1.0)
            for j in range(k + 1):
                wd.at(f"T2 iter {it} pair all-reduce {j + 1}/{k + 1}")
                dist.all_reduce(x, group=my_pair)            # both members hold equal values: x doubles
            wd.at(f"T2 iter {it} world all-reduce")
            w = full(rank + 1, shape=(64,))
            dist.all_reduce(w)
            ok &= same(w, 36) and same(x, 2 ** (k + 1))      # sum(r + 1 for r in 0..7) = 36
            per_iter.append(1e3 * (time.perf_counter() - t0))
        record("T2", ok, f"{args.iters} iters x ({k + 1} pair all-reduces + 1 world all-reduce): "
                         f"{sum(per_iter) / len(per_iter):.2f} ms/iter, first {per_iter[0]:.1f} ms")

    # --- T3 / T3c: one program holding two pair collectives ---------------------------------
    if "T3" in tests or "T3c" in tests:
        from torch_tpu._internal import execution_mode
        from torch_tpu._internal.distributed import spmd_util
        rows, hid = 512, 512
        w = torch.eye(hid, device=dev) * 2.0                   # matmul doubles the input

        body_group = None if args.world_body else my_pair      # None = WORLD
        body_size = WORLD if args.world_body else 2
        body_sum = sum(range(1, WORLD + 1)) if args.world_body else pair_sum + 2

        def body(x, wm):
            y = torch.matmul(x, wm)                            # 2 * (rank + 1 + 10 i)
            dist.all_reduce(y, group=body_group)               # 2 * (group_sum + size * 10 i)
            z = y * 0.5 + 1.0                                  # group_sum + size * 10 i + 1
            out = torch.empty((body_size * rows, hid), dtype=torch.float32, device=dev)
            dist.all_gather_into_tensor(out, z, group=body_group)
            return out + 1.0                                   # group_sum + size * 10 i + 2

        def want(i):
            return body_sum + body_size * 10 * i + 2

    if "T3" in tests:
        barrier("T3")
        n = 2 if args.even_counts else 2 + k
        ok, ts, counts = True, [], None
        for i in range(n):
            wd.at(f"T3 fused program {i + 1}/{n}")
            x = full(rank + 1 + 10 * i, shape=(rows, hid))
            sync.synchronize([x, w], wait=True)                # inputs materialized: the region holds only the body
            t0 = time.perf_counter()
            tpu_distributed.enter_spmd_safe_region()           # collectives stay inside the deferred graph
            try:
                with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
                    out = body(x, w)
                    if counts is None:
                        mlir = sync.computation_mlir(out)
                        counts = (mlir.count("stablehlo.all_reduce"), mlir.count("stablehlo.all_gather"),
                                  mlir.count("stablehlo.dot"))
                    sync.synchronize([out], wait=False)        # one program: matmul + both collectives
            finally:
                tpu_distributed.exit_spmd_safe_region()
            done(out)
            ts.append(1e3 * (time.perf_counter() - t0))
            ok &= same(out, want(i))
        wd.at("T3 world all-reduce")
        wv = full(rank + 1, shape=(64,)); dist.all_reduce(wv); ok &= same(wv, 36)
        ok &= counts is not None and counts[0] == 1 and counts[1] == 1
        record("T3", ok, f"{n} launches of one fused program with (all_reduce, all_gather, dot) = {counts}: "
                         f"{ms_list(ts)} ms, then 1 world all-reduce")

    if "T3c" in tests:
        barrier("T3c")
        n = 2 if args.even_counts else 2 + k
        ok, ts, info = True, [], "n/a"
        try:
            from torch_tpu._internal.compile import TpuBackend
            backend = TpuBackend(debug=True) if args.compile_backend == "debug" else "tpu"
            fn = torch.compile(spmd_util.spmd_safe(body), backend=backend)   # spmd_safe: no split at collectives
            for i in range(n):
                wd.at(f"T3c compiled program {i + 1}/{n}")
                x = full(rank + 1 + 10 * i, shape=(rows, hid))
                t0 = time.perf_counter(); out = done(fn(x, w)); ts.append(1e3 * (time.perf_counter() - t0))
                ok &= same(out, want(i))
            try:                                                  # debug-only introspection, precedent: split_collective_ops_test.py
                execs = backend._compiled_executables
                subs = [c.submod for c in execs[0]._split_gm.children()] if hasattr(execs[0], "_split_gm") else []
                info = f"{len(execs)} executables, {len(subs)} submodules, all_reduce/all_gather per submodule " + str(
                    [(s.mlir_text.count("stablehlo.all_reduce"), s.mlir_text.count("stablehlo.all_gather")) for s in subs])
            except Exception as e:  # noqa: BLE001
                info = f"introspection unavailable ({type(e).__name__})"
        except Exception as e:  # noqa: BLE001
            ok = False; info = f"FAILED {type(e).__name__}: {str(e)[:300]}"
        wd.at("T3c world all-reduce")
        wv = full(rank + 1, shape=(64,)); dist.all_reduce(wv); ok &= same(wv, 36)
        record("T3c", ok, f"{n} launches of torch.compile'd body {ms_list(ts)} ms ({info}), then 1 world all-reduce")

    # --- T5: torch.compile'd body with ONLY a pair all-reduce (the PP x TP stage forward shape) ----
    if "T5" in tests:
        from torch_tpu._internal.distributed import spmd_util
        barrier("T5")
        rows, hid = 512, 512
        w5 = torch.eye(hid, device=dev) * 2.0

        def body5(x, wm):
            y = torch.matmul(x, wm)                            # 2 * (rank + 1 + 10 i)
            dist.all_reduce(y, group=my_pair)                  # 2 * (pair_sum + 2 + 20 i)
            return y * 0.5                                     # pair_sum + 2 + 20 i

        n = 2 if args.even_counts else 2 + k
        ok, ts, info = True, [], "ok"
        try:
            fn5 = torch.compile(spmd_util.spmd_safe(body5), backend="tpu")
            for i in range(n):
                wd.at(f"T5 compiled all-reduce program {i + 1}/{n}")
                x = full(rank + 1 + 10 * i, shape=(rows, hid))
                t0 = time.perf_counter(); out = done(fn5(x, w5)); ts.append(1e3 * (time.perf_counter() - t0))
                ok &= same(out, pair_sum + 2 + 20 * i)
        except Exception as e:  # noqa: BLE001
            ok = False; info = f"FAILED {type(e).__name__}: {str(e)[:400]}"
        wd.at("T5 world all-reduce")
        wv = full(rank + 1, shape=(64,)); dist.all_reduce(wv); ok &= same(wv, 36)
        record("T5", ok, f"{n} launches of torch.compile(backend='tpu') matmul+pair all-reduce {ms_list(ts)} ms ({info}), then 1 world all-reduce")

    barrier("end")
    failed = [n for n, r in results.items() if not r]
    _log(rank, f"SUMMARY pass={[n for n, r in results.items() if r]} fail={failed}")
    dist.destroy_process_group()
    wd.stop()
    sys.exit(2 if failed else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="T0,T1,T3c,T5")
    ap.add_argument("--standalone", action="store_true", help="call torch_tpu#4387 make_standalone on each pair before its first collective")
    ap.add_argument("--compile-backend", choices=["debug", "tpu"], default="tpu", help="T3c: TpuBackend(debug=True) for introspection, or the plain 'tpu' backend used by vLLM")
    ap.add_argument("--world-body", action="store_true", help="T3/T3c: the fused body runs over the WORLD group (no restriction; teardown control)")
    ap.add_argument("--even-counts", action="store_true", help="T3/T3c: every pair launches twice (lockstep control)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--permute", choices=["ppermute", "a2a"], default="ppermute")
    ap.add_argument("--stuck-secs", type=int, default=30, help="seconds without progress before STUCK lines")
    ap.add_argument("--hang-secs", type=int, default=180, help="seconds without progress before SIGALRM")
    ap.add_argument("--timeout", type=int, default=900, help="seconds before the parent kills the workers")
    a = ap.parse_args()
    import portpicker
    from torch_tpu._internal.distributed.launchers import singlehost_wrapper
    singlehost_wrapper.prepare_tpu_environment(WORLD)      # slicebuilder ports + TORCH_TPU_TOPOLOGY, before spawning
    port = portpicker.pick_unused_port()
    print(f"MICRO tests={a.tests} iters={a.iters} permute={a.permute} port={port} "
          f"topology={os.environ.get('TORCH_TPU_TOPOLOGY')}", flush=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, port, a)) for r in range(WORLD)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + a.timeout
    for p in procs:
        p.join(max(1, deadline - time.monotonic()))
    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    if hung:
        print(f"MICRO TIMEOUT after {a.timeout} s: ranks {hung} still running; killing", flush=True)
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(30)
    codes = [p.exitcode for p in procs]
    print(f"MICRO exit codes {codes}", flush=True)
    print(f"MICRO RESULT {'PASS' if all(c == 0 for c in codes) else 'FAIL'} tests={a.tests}", flush=True)
    sys.exit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
