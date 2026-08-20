#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pairwise ICI latency across a TPU slice: is the mesh flat or hierarchical?

Every design decision in ``low_latency_all_reduce.py`` currently treats the 32
devices as interchangeable.  ``_GATE`` keys on the device *count* alone, both
kernels fan out to ``peer = (my_id + i) % N`` with no notion of near or far, and
``_get_mesh`` builds a 1-D mesh in TP-rank order -- there is no axis on which a
hierarchy could even be expressed.  The kernel module's own docstring flags the
consequence: "Peer addressing uses ``DeviceIdType.MESH``, so the mesh axis order
determines the physical routing."  Nobody has ever measured what that order
costs.

We do know the slice is hierarchical in *principle*: ``chip_topology.py`` reports
``16 chip(s) x 2 core(s)`` on an ``ici-2x2x4`` TPU7x slice, and the repo already
exploits that for layer partitioning (``TPU_MOE_HIERARCHICAL_EP``,
``TPU_LATENT_PROJ_INTRA_CHIP_TP``).  What we do *not* know, and cannot derive
from a coords listing, is the shape of the latency penalty: how many distinct
tiers there are, how big the on-chip discount really is, and whether the far tail
is what sets the one-shot floor.  Every hop-count claim about this slice is
inference until this probe runs.

What is measured
----------------
One bidirectional exchange between a pair of devices, at a payload small enough
(2 KiB default) that wire time is negligible and what remains is issue + transit
+ semaphore delivery.  Both sides send and both sides wait, so the number is a
one-way latency under symmetric load -- which is what a collective actually
pays, rather than an RTT.

Two schedules, both driven by a *runtime* partner vector so the Pallas kernel
compiles once and only the schedule changes:

``fanout`` (run this first)
    ``N-1`` configurations, ``root <-> r``, with every other device sitting the
    round out.  Zero contention, so the number is the pair's own latency and
    nothing else.  Gives the full distance profile from one vertex, which is
    what reveals the tier structure.

``matrix``
    ``N-1`` rounds of ``N/2`` disjoint pairs (circle method), covering all
    ``C(N,2)`` pairs exactly once.  16 concurrent flows means some route
    sharing, so treat these as contended numbers -- but it is the only way to
    get the whole matrix in 31 configurations instead of 496.

Timing is the two-point slope this repo uses everywhere else: time a chain of
``n_lo`` then ``n_hi`` exchanges in one jitted call and take the difference over
the difference.  Every fixed cost -- dispatch, transfer, host sync -- cancels.
The chain is serial by construction: ``ws`` and ``step`` are
``input_output_aliases``, so call ``k+1`` cannot start before call ``k`` retires
its workspace.

Correctness
-----------
A silently-wrong partner would produce a meaningless matrix that looks perfectly
plausible, so every configuration is wiring-checked before it is timed: device
``d``'s payload is filled with the value ``d``, and after one exchange each
participant must see exactly its partner's id.  ``--skip-verify`` exists only
for timing-only reruns.

No barrier past the cold start, deliberately
--------------------------------------------
The kernel barriers once at ``step == 0`` and never again.  It does not need to:
each round waits both its send *and* recv semaphores, so no DMA is in flight
when the kernel exits, and a partner switch between configurations cannot race.
This also keeps the barrier's ~2.9 us at 32 devices (``two_shot_ablation.py``,
jobs 118/119) out of a number that is supposed to be transit.  ``--cold-barrier
every`` restores a per-round barrier as a control if a tier boundary looks
suspicious.

Run (32 devices, 4 hosts):
    sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_pairwise_latency.sbatch
"""

import argparse
import os
import sys
import time


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",
                   default="fanout",
                   help="Comma-separated: fanout (one root, "
                   "isolated), allpairs (every pair, isolated -- the real "
                   "matrix), matrix (concurrent rounds, contended).")
    p.add_argument("--root",
                   type=int,
                   default=0,
                   help="Hub device for --mode fanout.")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols",
                   type=int,
                   default=128,
                   help="Lane dim; keep a multiple of 128.")
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--n-lo", type=int, default=400)
    p.add_argument("--n-hi", type=int, default=1600)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--cold-barrier",
        default="first",
        choices=["first", "every"],
        help="'first' barriers only at step 0 (default); 'every' is "
        "the control that puts a full barrier in each round.")
    p.add_argument("--sweep-n",
                   default=None,
                   help="Diagnostic: comma-separated chain lengths to time "
                   "absolutely for a single pair, to confirm wall time is "
                   "linear in chain length before any slope is believed. "
                   "Implies --mode fanout and stops after the sweep.")
    p.add_argument("--peer",
                   type=int,
                   default=1,
                   help="Partner for --sweep-n.")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--csv", default=None)
    p.add_argument("--multihost", action="store_true")
    p.add_argument("--sc-offload-bytes", type=int, default=None)
    return p.parse_args(argv)


ARGS = _parse_args()

# Consumed at TPU-client construction, so it must precede the backend import.
if ARGS.sc_offload_bytes is not None:
    _extra = " ".join(
        f"--xla_tpu_sparse_core_{name}_offload_min_size_in_bytes="
        f"{ARGS.sc_offload_bytes}" for name in ("all_reduce", "all_gather"))
    os.environ["LIBTPU_INIT_ARGS"] = (
        _extra + " " + os.environ.get("LIBTPU_INIT_ARGS", "")).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

if ARGS.multihost or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize()

IS_LEAD = jax.process_index() == 0
AXIS = "x"


def rprint(*a, **kw):
    if IS_LEAD:
        print(*a, **kw)
        sys.stdout.flush()


def _load_full_barrier():
    """Pull ``_full_barrier`` out of the kernel module by path.

    Same reason as ``allreduce_latency_bench.py``: importing the package would
    run ``env_override``, which injects ``LIBTPU_INIT_ARGS`` and would change
    the thing being measured. The kernel module itself only needs JAX.
    """
    import importlib.util
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "src", "vllm_torchtpu", "kernels", "collectives",
                        "low_latency_all_reduce.py")
    spec = importlib.util.spec_from_file_location("_llar_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._full_barrier


_full_barrier = _load_full_barrier()

# -----------------------------------------------------------------------------
# Kernel: one bidirectional exchange with a runtime-selected partner.
# -----------------------------------------------------------------------------


def _pingpong_kernel(x_ref, ws_in_ref, step_in_ref, partner_ref, o_ref, ws_ref,
                     step_ref, local_sem, send_sem, recv_sem, *,
                     num_devices: int, axis_name: str, barrier_every: bool):
    del ws_in_ref  # aliased onto ws_ref
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)
    partner = partner_ref[0]

    if barrier_every:
        _full_barrier(my_id, num_devices, double_barrier=True)
    else:

        @pl.when(step == 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    # partner == my_id is the encoding for "sit this round out". The step
    # counter still advances on every device, so all ranks stay in lockstep on
    # the slot parity even when only two of them move data.
    @pl.when(partner != my_id)
    def _():
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=ws_ref.at[slot],
            send_sem=send_sem.at[slot],
            recv_sem=recv_sem.at[slot],
            device_id=(partner, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        # Waiting both semaphores is what makes the kernel self-synchronizing:
        # nothing is in flight at exit, so no barrier is needed to switch
        # partners between configurations.
        copy.wait()

    # ws lives in HBM so its address survives between calls; stage it out.
    out = pltpu.make_async_copy(src_ref=ws_ref.at[slot],
                                dst_ref=o_ref,
                                sem=local_sem)
    out.start()
    out.wait()
    step_ref[0] = step + 1


def pingpong(x,
             ws,
             step,
             partner,
             *,
             num_devices: int,
             barrier_every: bool,
             collective_id: int = 11):
    kernel = __import__("functools").partial(_pingpong_kernel,
                                             num_devices=num_devices,
                                             axis_name=AXIS,
                                             barrier_every=barrier_every)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(ws.shape, ws.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        input_output_aliases={
            1: 1,
            2: 2
        },
        scratch_shapes=[
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=True,
                                             disable_semaphore_checks=True),
    )(x, ws, step, partner)


# -----------------------------------------------------------------------------
# Schedules. Every partner vector is an involution; p[d] == d means "idle".
# -----------------------------------------------------------------------------


def fanout_schedule(n: int, root: int):
    """``n-1`` isolated configurations: root <-> r, all others idle."""
    for r in range(n):
        if r == root:
            continue
        p = list(range(n))
        p[root], p[r] = r, root
        yield [(root, r)], p


def allpairs_schedule(n: int):
    """Every one of the ``C(N,2)`` pairs, measured in isolation.

    This is ``fanout`` generalized to every root, and it is the mode that
    actually produces a *matrix*. ``matrix`` below times a whole round of
    concurrent pairs against one host clock, so it can only report a per-round
    figure that every pair in the round then inherits -- which on the 8-device
    validation run inflated the on-chip pair from 3.37 us to 4.61 us purely by
    averaging it with three off-chip pairs. Isolated configurations cost more
    wall time (``C(N,2)`` of them) but each number belongs to exactly one pair.
    """
    for a in range(n):
        for b in range(a + 1, n):
            p = list(range(n))
            p[a], p[b] = b, a
            yield [(a, b)], p


def matrix_schedule(n: int):
    """``n-1`` rounds of ``n/2`` disjoint pairs covering every pair once.

    Circle method: device ``n-1`` is the pivot and the rest rotate. Round ``r``
    pairs the pivot with ``r`` and then ``(r+i)`` with ``(r-i)`` mod ``n-1``,
    which is a perfect matching because ``r+i`` and ``r-i`` for ``i`` in
    ``1..n/2-1`` hit each of the remaining ``n-2`` positions exactly once.
    """
    if n % 2:
        raise ValueError(f"circle method needs an even device count, got {n}")
    m = n - 1
    for r in range(m):
        p = list(range(n))
        pairs = [(m, r)]
        p[m], p[r] = r, m
        for i in range(1, n // 2):
            u = (r + i) % m
            v = (r - i) % m
            pairs.append((min(u, v), max(u, v)))
            p[u], p[v] = v, u
        yield pairs, p


def _check_schedule(n: int, pairs, p):
    """An involution with the advertised pairs, or the matrix is meaningless."""
    for d in range(n):
        if p[p[d]] != d:
            raise AssertionError(f"partner vector is not an involution at {d}")
    for a, b in pairs:
        if p[a] != b or p[b] != a:
            raise AssertionError(f"pair ({a},{b}) not in partner vector")


# -----------------------------------------------------------------------------
# Harness
# -----------------------------------------------------------------------------


def build_mesh():
    """1-D mesh in device-id order -- exactly what production builds.

    ``low_latency_all_reduce._get_mesh`` orders the mesh by TP rank, and TP rank
    ``r`` drives device id ``r``. Mirroring it here is the whole point: the
    numbers have to describe the mesh the kernels actually address.
    """
    devices = sorted(jax.devices(), key=lambda d: d.id)
    return Mesh(np.asarray(devices), axis_names=(AXIS, ))


def describe_devices():
    rows = []
    for d in sorted(jax.devices(), key=lambda d: d.id):
        coords = tuple(getattr(d, "coords", ()) or ())
        rows.append({
            "id": int(d.id),
            "coords": coords,
            "core": getattr(d, "core_on_chip", None),
            "process": int(d.process_index),
        })
    return rows


def make_chain(num_devices: int, barrier_every: bool, mesh, shard,
               scalar_shard):
    """One jitted, runtime-length chain of serial exchanges.

    The chain is a ``fori_loop`` rather than a Python-unrolled sequence for two
    reasons, both learned the hard way on the 8-device smoke run:

    * **Chain length has to be a runtime value.** The fixed cost of one blocked
      dispatch here is ~1.1 ms, so a 20-vs-60 slope was measuring nothing but
      noise -- absolute wall time was flat from n=1 (1.106 ms) to n=80
      (1.174 ms), and the harness happily reported 0.06 us/exchange, well below
      the 1.42 us launch floor. The chain has to be long enough that device time
      dominates, which means hundreds to thousands of exchanges.
    * **Unrolling that many Pallas calls does not compile in reasonable time.**
      A loop keeps compilation O(1) and, as a bonus, one executable serves every
      chain length, so no configuration pays a recompile.

    Serialization is preserved: ``ws`` and ``step`` are loop-carried and
    ``input_output_aliases`` makes each iteration depend on the previous one's
    workspace, so iterations cannot overlap.
    """

    def body(local_x, local_ws, local_step, local_partner, n_steps):
        x = local_x[0]
        partner = local_partner

        def one(_, carry):
            out, ws, step = carry
            del out
            return pingpong(x,
                            ws,
                            step,
                            partner,
                            num_devices=num_devices,
                            barrier_every=barrier_every)

        out, ws, step = lax.fori_loop(0, n_steps[0], one,
                                      (x, local_ws[0], local_step))
        return out[None], ws[None], step

    return jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS, None,
                                  None), P(AXIS, None, None,
                                           None), P(AXIS), P(AXIS), P(AXIS)),
                      out_specs=(P(AXIS, None,
                                   None), P(AXIS, None, None, None), P(AXIS)),
                      check_vma=False))


def time_call(fn, args, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        multihost_utils.sync_global_devices("probe")
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main():
    dtype = jnp.bfloat16 if ARGS.dtype == "bfloat16" else jnp.float32
    n = jax.device_count()
    rows, cols = ARGS.rows, ARGS.cols
    payload = rows * cols * np.dtype(ARGS.dtype).itemsize

    mesh = build_mesh()
    shard = NamedSharding(mesh, P(AXIS, None, None))
    ws_shard = NamedSharding(mesh, P(AXIS, None, None, None))
    scalar_shard = NamedSharding(mesh, P(AXIS))

    devs = describe_devices()
    coords_of = {d["id"]: d["coords"] for d in devs}
    rprint(f"== {n} devices, {jax.process_count()} process(es), "
           f"payload {payload} B ({rows}x{cols} {ARGS.dtype})")
    rprint(f"== chain {ARGS.n_lo} vs {ARGS.n_hi}, barrier={ARGS.cold_barrier}")
    rprint("== devices (id, coords, core_on_chip, process):")
    for d in devs:
        rprint(f"     {d['id']:>3}  {d['coords']}  core={d['core']}  "
               f"proc={d['process']}")

    # Payload filled with the device's own id: after one exchange a participant
    # must see its partner's id, which is the wiring check.
    x = jax.jit(lambda: jnp.broadcast_to(
        jnp.arange(n, dtype=dtype).reshape(n, 1, 1), (n, rows, cols)),
                out_shardings=shard)()
    ws = jax.jit(lambda: jnp.zeros((n, 2, rows, cols), dtype),
                 out_shardings=ws_shard)()
    step = jax.jit(lambda: jnp.zeros((n, ), jnp.int32),
                   out_shardings=scalar_shard)()

    barrier_every = ARGS.cold_barrier == "every"
    chain = make_chain(n, barrier_every, mesh, shard, scalar_shard)

    def n_arr(k):
        return jax.jit(lambda kk=int(k): jnp.full((n, ), kk, jnp.int32),
                       out_shardings=scalar_shard)()

    n_lo_d, n_hi_d, n_one_d = n_arr(ARGS.n_lo), n_arr(ARGS.n_hi), n_arr(1)

    if ARGS.sweep_n:
        lengths = [int(v) for v in ARGS.sweep_n.split(",")]
        p = list(range(n))
        p[ARGS.root], p[ARGS.peer] = ARGS.peer, ARGS.root
        partner = jax.jit(lambda pv=tuple(p): jnp.asarray(pv, jnp.int32),
                          out_shardings=scalar_shard)()
        rprint(f"\n== linearity control: {ARGS.root}<->{ARGS.peer}, "
               f"absolute median wall time vs chain length")
        rprint(f"{'n':>6} {'total_ms':>10} {'us/exchange':>13} "
               f"{'marginal_us':>13}")
        prev = None
        for ln in lengths:
            t = time_call(chain, (x, ws, step, partner, n_arr(ln)),
                          ARGS.repeats, ARGS.warmup)
            marg = ("" if prev is None else
                    f"{(t - prev[1]) / (ln - prev[0]) * 1e6:13.2f}")
            rprint(f"{ln:>6} {t*1e3:>10.3f} {t/ln*1e6:>13.2f} {marg:>13}")
            prev = (ln, t)
        rprint("\nIf total_ms is flat in n, the chain is not serializing and "
               "no slope from this harness means anything.")
        return

    results = []
    for mode in [m.strip() for m in ARGS.mode.split(",") if m.strip()]:
        if mode == "fanout":
            sched = list(fanout_schedule(n, ARGS.root))
        elif mode == "allpairs":
            sched = list(allpairs_schedule(n))
        elif mode == "matrix":
            sched = list(matrix_schedule(n))
        else:
            raise SystemExit(f"unknown mode {mode!r}")

        rprint(f"\n== mode={mode}: {len(sched)} configuration(s)")
        for cfg, (pairs, p) in enumerate(sched):
            _check_schedule(n, pairs, p)
            partner = jax.jit(lambda pv=tuple(p): jnp.asarray(pv, jnp.int32),
                              out_shardings=scalar_shard)()

            if not ARGS.skip_verify:
                out, ws_v, step_v = jax.block_until_ready(
                    chain(x, ws, step, partner, n_one_d))
                ws, step = ws_v, step_v
                got = np.asarray(
                    multihost_utils.process_allgather(out, tiled=True)).astype(
                        np.float32)
                # Only the participants are checked. An idle device keeps
                # whatever it last received, so once it has taken part in an
                # earlier configuration its workspace holds that stale partner
                # id -- which is correct behaviour, not a fault. Asserting
                # idle == 0 only ever held before any data had landed.
                for d in range(n):
                    if p[d] == d:
                        continue
                    want = float(p[d])
                    seen = got[d]
                    if not np.all(seen == want):
                        raise AssertionError(
                            f"{mode} cfg{cfg}: device {d} expected partner "
                            f"{want} but saw {np.unique(seen)[:4]}")

            t_lo = time_call(chain, (x, ws, step, partner, n_lo_d),
                             ARGS.repeats, ARGS.warmup)
            t_hi = time_call(chain, (x, ws, step, partner, n_hi_d),
                             ARGS.repeats, ARGS.warmup)
            us = (t_hi - t_lo) / (ARGS.n_hi - ARGS.n_lo) * 1e6
            # A slope is only meaningful if the chain actually got longer. The
            # smoke run on 8 devices produced 0.06 us/exchange -- below the
            # 1.42 us launch floor -- because t_hi and t_lo differed by noise.
            growth = (t_hi / t_lo) if t_lo > 0 else float("nan")

            # Re-thread ws/step off a real call so the slot parity the next
            # configuration starts from is the one the device actually holds.
            _, ws, step = jax.block_until_ready(
                chain(x, ws, step, partner, n_one_d))

            for a, b in pairs:
                ca, cb = coords_of.get(a, ()), coords_of.get(b, ())
                same_chip = bool(ca and ca == cb)
                manhattan = (sum(abs(i - j) for i, j in zip(ca, cb))
                             if ca and cb and len(ca) == len(cb) else -1)
                results.append({
                    "mode": mode,
                    "cfg": cfg,
                    "a": a,
                    "b": b,
                    "coords_a": ca,
                    "coords_b": cb,
                    "same_chip": same_chip,
                    "manhattan": manhattan,
                    "bytes": payload,
                    "us": us,
                    "t_lo_ms": t_lo * 1e3,
                    "t_hi_ms": t_hi * 1e3,
                })
            tag = (f"{pairs[0][0]:>3}<->{pairs[0][1]:<3}"
                   if len(pairs) == 1 else f"{len(pairs)} pairs")
            extra = ""
            if len(pairs) == 1:
                a, b = pairs[0]
                extra = (
                    f"  coords {coords_of.get(a)} -> {coords_of.get(b)}"
                    f"  manhattan={results[-1]['manhattan']}"
                    f"{'  SAME CHIP' if results[-1]['same_chip'] else ''}")
            rprint(f"   cfg{cfg:>3}  {tag}  {us:7.2f} us  "
                   f"[t{ARGS.n_lo}={t_lo*1e3:.3f} ms "
                   f"t{ARGS.n_hi}={t_hi*1e3:.3f} ms "
                   f"x{growth:.2f}]{extra}")

    _report(results)
    if ARGS.csv and IS_LEAD:
        _write_csv(results, ARGS.csv)
        rprint(f"\n== wrote {ARGS.csv}")


def _report(results):
    if not results:
        return
    for mode in dict.fromkeys(r["mode"] for r in results):
        rows = [r for r in results if r["mode"] == mode]
        rprint(f"\n== {mode}: latency by physical distance")
        if mode == "matrix":
            rprint("   (contended: 1 figure per round of concurrent pairs, "
                   "inherited by every pair in it -- not per-pair)")
        rprint(f"{'distance':>12} {'pairs':>6} {'min':>8} {'median':>8} "
               f"{'max':>8}")
        buckets: dict = {}
        for r in rows:
            key = "same chip" if r[
                "same_chip"] else f"manhattan {r['manhattan']}"
            buckets.setdefault(key, []).append(r["us"])
        for key in sorted(buckets, key=lambda k: (k != "same chip", k)):
            v = np.asarray(buckets[key])
            rprint(f"{key:>12} {len(v):>6} {v.min():>8.2f} "
                   f"{np.median(v):>8.2f} {v.max():>8.2f}")
        v = np.asarray([r["us"] for r in rows])
        rprint(f"{'ALL':>12} {len(v):>6} {v.min():>8.2f} "
               f"{np.median(v):>8.2f} {v.max():>8.2f}")
        if v.min() > 0:
            rprint(f"   spread far/near = {v.max() / v.min():.2f}x")


def _write_csv(results, path):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "mode", "cfg", "a", "b", "coords_a", "coords_b", "same_chip",
            "manhattan", "bytes", "us"
        ])
        for r in results:
            w.writerow([
                r["mode"], r["cfg"], r["a"], r["b"],
                "/".join(map(str,
                             r["coords_a"])), "/".join(map(str,
                                                           r["coords_b"])),
                int(r["same_chip"]), r["manhattan"], r["bytes"],
                f"{r['us']:.4f}", f"{r['t_lo_ms']:.4f}", f"{r['t_hi_ms']:.4f}"
            ])


if __name__ == "__main__":
    main()
