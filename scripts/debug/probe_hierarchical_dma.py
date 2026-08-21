#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does a two-level all-reduce beat a flat one at K3's payloads?

The pairwise probe (``probe_pairwise_latency.py``) established the cost model
that makes this worth asking: invoking a Pallas kernel costs 2.59 us with no
network at all, while an on-package DMA costs 0.94 us and each extra ICI hop
only 0.46 us. Launch is paid per *kernel*; a DMA phase is paid per *phase*. So a
hierarchical collective built as two ``pallas_call``s pays 5.18 us of pure
invocation overhead -- 40% of two-shot's 12.6 us budget, spent on nothing -- but
one built as a single kernel with extra DMA phases pays only ~0.9 us per phase.

That is the claim under test. Rather than implement a production kernel to find
out, each arm here issues the exact DMA pattern its scheme would use, inside one
kernel, and is timed against flat controls on the same harness.

Every arm computes a genuine all-reduce and is checked against ``jax.lax.psum``.
That is the point of doing it this way: a pattern that is merely
"representative" can quietly move less data than the real thing (the hazard that
made the first pairwise run report sub-launch-floor latencies), whereas a
pattern that produces the correct sum provably moved all of it.

Arms
----
``flat_oneshot``
    Control. Every device pushes the whole payload to all 31 peers. Should track
    ``oneshot_ws`` from ``allreduce_latency_bench.py`` modulo the barrier.
``flat_twoshot``
    Control. Reduce-scatter row ``peer`` to ``peer``, reduce, all-gather. The
    production choice at 14-21 KiB.
``hier_oneshot``
    Core ``c`` owns half ``c``. Phase A exchanges the other half with the
    on-chip partner (``my_id ^ 1``) and reduces 2 contributions; phase B
    one-shots that half across the corresponding core of the 15 other chips and
    reduces 16; phase C returns the result to the partner. Egress 17 halves =
    8.5x payload against flat's 31x.
``hier_oneshot_noc``
    Same without phase C, so each core keeps only its own half. Not a usable
    collective -- it measures what phase C costs, which matters because at the
    MoE call site a half-buffer result might be free given the ``torch.cat``
    already there.
An all-1/1-class arm (core 0 hands everything to core 1, which does the
inter-chip round alone so every leg is the fast core1->core1 class) is
deliberately absent. It does not compose with the split: if core c owns half c,
then core 0 of chip i must send half 0 to whoever owns half 0 on chip j, which is
core 0 of chip j, so the slow 0/0 legs are forced. Buying the far-peer term
requires idling core 0, which halves the egress cut -- and the egress cut is the
lever that matters, since oneshot_ws is already steeply bandwidth-bound below
K3's payloads (10.39 -> 15.36 -> 26.06 us across 8 -> 16 -> 32 KiB, job 255).
``hier_twoshot``
    The arm that matters, since the gate sends K3's payloads to two-shot. Phase
    A intra-chip, then a two-shot over 16 chips on the half, then phase C.

Chip mapping is index arithmetic on the 1-D mesh, which is exact here: the
pairwise probe and ``chip_topology`` agree that devices ``2g`` and ``2g+1`` share
chip ``g`` (dev-node coords put ids 0,1 at (0,0,0) and 2,3 at (1,0,0); the
32-device log reports chip0=[0,1], chip8=[2,3]). A production kernel would need
a real 2-D mesh from ``chip_topology`` -- see the note in
docs/low_latency_allreduce.md section 6 -- but for measuring a DMA pattern,
``DeviceIdType.MESH`` maps mesh index to device either way.

Barrier: every arm opens with the same double barrier and uses scratch-VMEM
destinations, i.e. a stateless kernel rather than a workspace one. That
inflates all arms equally and keeps the comparison clean; absolute numbers are
therefore not comparable to the ``_ws`` figures in the doc, only to each other.

Run (32 devices, 4 hosts):
    sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_hierarchical_dma.sbatch
"""

import argparse
import functools
import os
import sys
import time

ARMS = ("flat_oneshot", "flat_twoshot", "hier_oneshot", "hier_oneshot_noc",
        "hier_twoshot")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--cols",
                   default="224,336,448",
                   help="Trailing dim per row; payload = 32*cols*2 bytes. "
                   "224 -> 14 KiB (K3 post-attention), 336 -> 21 KiB (K3 "
                   "latent-MoE pair), 448 -> 28 KiB.")
    p.add_argument("--n-lo", type=int, default=200)
    p.add_argument("--n-hi", type=int, default=800)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sweep-n",
                   default=None,
                   help="Diagnostic: absolute times across chain lengths, to "
                   "confirm wall time is linear before any slope is believed.")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--csv", default=None)
    p.add_argument("--multihost", action="store_true")
    return p.parse_args(argv)


ARGS = _parse_args()

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
CORES_PER_CHIP = 2


def rprint(*a, **kw):
    if IS_LEAD:
        print(*a, **kw)
        sys.stdout.flush()


def _barrier(my_id, n):
    sem = pltpu.get_barrier_semaphore()
    for i in range(1, n):
        pl.semaphore_signal(sem,
                            1,
                            device_id=(lax.rem(my_id + i, n), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, n - 1)

    @functools.partial(pl.run_scoped, s2=pltpu.SemaphoreType.REGULAR)
    def _(s2):
        for i in range(1, n):
            pl.semaphore_signal(s2,
                                1,
                                device_id=(lax.rem(my_id + i, n), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(s2, n - 1)


def _rc(src, dst, ssem, rsem, peer):
    return pltpu.make_async_remote_copy(src_ref=src,
                                        dst_ref=dst,
                                        send_sem=ssem,
                                        recv_sem=rsem,
                                        device_id=(peer, ),
                                        device_id_type=pl.DeviceIdType.MESH)


def _kernel(x_ref, o_ref, a_buf, b_buf, c_buf, d_buf, sa_s, sa_r, sb_s, sb_r,
            sc_s, sc_r, sd_s, sd_r, *, n: int, arm: str):
    """One kernel, one launch, however many DMA phases the arm needs.

    The payload is 3-D, ``(n, chunk, cols)``. That is not cosmetic: Mosaic
    requires slices along *tiled* dimensions to be tiling-aligned, and for a 2-D
    array the leading dim is tiled, so a dynamic single-row slice is rejected --
    the same constraint that made a variadic two-shot unreachable
    (scripts/debug/probe_variadic_dma.py). Putting the chunk index on an untiled
    leading dimension is what makes ``x_ref.at[peer]`` legal, and it is why
    ``_build_two_shot`` reshapes to ``(N, numel/N)`` in production.
    """
    my_id = lax.axis_index(AXIS)
    _barrier(my_id, n)
    n_chips = n // CORES_PER_CHIP
    half = n // CORES_PER_CHIP  # chunks per half
    core = lax.rem(my_id, CORES_PER_CHIP)
    chip = lax.div(my_id, CORES_PER_CHIP)
    partner = lax.bitwise_xor(my_id, 1)

    if arm == "flat_oneshot":
        cps = [
            _rc(x_ref, a_buf.at[i - 1], sa_s, sa_r, lax.rem(my_id + i, n))
            for i in range(1, n)
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        acc = x_ref[...]
        for i in range(n - 1):
            acc = acc + a_buf[i]
        o_ref[...] = acc
        return

    if arm == "flat_twoshot":
        cps = []
        for i in range(1, n):
            peer = lax.rem(my_id + i, n)
            cps.append(_rc(x_ref.at[peer], a_buf.at[i - 1], sa_s, sa_r, peer))
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        acc = x_ref[my_id]
        for i in range(n - 1):
            acc = acc + a_buf[i]
        c_buf[0] = acc
        cps = [
            _rc(c_buf.at[0], b_buf.at[my_id], sb_s, sb_r,
                lax.rem(my_id + i, n)) for i in range(1, n)
        ]
        for c in cps:
            c.start()
        b_buf[my_id] = acc
        for c in cps:
            c.wait()
        o_ref[...] = b_buf[...]
        return

    # hier_*: core c owns half c (chunks [half*c, half*c+half)).
    mine = pl.ds(half * core, half)
    theirs = pl.ds(half * (1 - core), half)

    # Phase A -- hand the partner the half it owns, take the half we own.
    cp = _rc(x_ref.at[theirs], a_buf, sa_s, sa_r, partner)
    cp.start()
    cp.wait()
    c_buf[...] = x_ref[mine] + a_buf[...]

    if arm == "hier_twoshot":
        # Two-shot across chips on our half: chip j owns chunk j of it.
        cps = []
        for i in range(1, n_chips):
            pc = lax.rem(chip + i, n_chips)
            cps.append(
                _rc(c_buf.at[pc], b_buf.at[i - 1], sb_s, sb_r,
                    pc * CORES_PER_CHIP + core))
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        acc = c_buf[chip]
        for i in range(n_chips - 1):
            acc = acc + b_buf[i]
        # Gather into d_buf, not c_buf: c_buf is still the source of our own
        # scatter sends, and a peer that has already finished its scatter would
        # otherwise clobber a slot we are reading.
        d_buf[chip] = acc
        cps = [
            _rc(d_buf.at[chip], d_buf.at[chip], sc_s, sc_r,
                lax.rem(chip + i, n_chips) * CORES_PER_CHIP + core)
            for i in range(1, n_chips)
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        reduced = d_buf[...]
    else:
        # Phase B -- one-shot our half across the matching core of every chip.
        cps = [
            _rc(c_buf, b_buf.at[i - 1], sb_s, sb_r,
                lax.rem(chip + i, n_chips) * CORES_PER_CHIP + core)
            for i in range(1, n_chips)
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        reduced = c_buf[...]
        for i in range(n_chips - 1):
            reduced = reduced + b_buf[i]

    if arm == "hier_oneshot_noc":
        # Not a usable collective -- each core holds only its own half. This
        # measures what phase C costs, which matters because at the MoE call
        # site a half-buffer result may be free given the torch.cat already
        # there.
        o_ref[mine] = reduced
        o_ref[theirs] = reduced
        return

    # Phase C -- swap reduced halves with the partner.
    c_buf[...] = reduced
    cp = _rc(c_buf, a_buf, sd_s, sd_r, partner)
    cp.start()
    cp.wait()
    o_ref[mine] = reduced
    o_ref[theirs] = a_buf[...]


def make_op(arm: str, n: int, chunk: int, cols: int, dtype):
    n_chips = n // CORES_PER_CHIP
    half = n // CORES_PER_CHIP
    if arm == "flat_oneshot":
        a, b, c, d = (n - 1, n, chunk), (1, 1, 1), (1, 1, 1), (1, 1, 1)
    elif arm == "flat_twoshot":
        a, b, c, d = (n - 1, chunk), (n, chunk), (1, chunk), (1, chunk)
    elif arm == "hier_twoshot":
        a, b, c, d = ((half, chunk), (n_chips - 1, chunk), (half, chunk),
                      (half, chunk))
    else:
        a, b, c, d = ((half, chunk), (n_chips - 1, half, chunk), (half, chunk),
                      (1, 1, 1))

    def shp(t):
        return pltpu.VMEM((*t, cols), dtype)

    kernel = functools.partial(_kernel, n=n, arm=arm)
    return lambda v: pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n, chunk, cols), dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[shp(a), shp(b), shp(c), shp(d)] +
        [pltpu.SemaphoreType.DMA] * 8,
        compiler_params=pltpu.CompilerParams(collective_id=7,
                                             has_side_effects=True,
                                             disable_semaphore_checks=True),
    )(v)


def build_mesh():
    return Mesh(np.asarray(sorted(jax.devices(), key=lambda d: d.id)),
                axis_names=(AXIS, ))


def make_chain(arm, n, chunk, cols, dtype, mesh):
    op = make_op(arm, n, chunk, cols, dtype)
    scale = np.float32(1.0 / n)

    def body(local_x, n_steps):
        v = local_x[0]

        def one(_, acc):
            return op(acc) * scale.astype(dtype)

        return lax.fori_loop(0, n_steps[0], one, v)[None]

    return jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS, None, None, None), P(AXIS)),
                      out_specs=P(AXIS, None, None, None),
                      check_vma=False))


def make_verify(arm, n, chunk, cols, dtype, mesh):
    op = make_op(arm, n, chunk, cols, dtype)

    def body(local_x):
        v = local_x[0]
        got = op(v).astype(jnp.float32)
        want = lax.psum(v, axis_name=AXIS).astype(jnp.float32)
        rel = jnp.max(jnp.abs(got - want)) / jnp.maximum(
            jnp.max(jnp.abs(want)), 1e-6)
        return lax.pmax(rel, axis_name=AXIS)

    return jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS, None, None, None), ),
                      out_specs=P(),
                      check_vma=False))


def time_call(fn, args, repeats, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(repeats):
        multihost_utils.sync_global_devices("probe")
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append(time.perf_counter() - t0)
    local = float(np.median(s))
    if jax.process_count() == 1:
        return local
    return float(
        np.max(
            np.asarray(multihost_utils.process_allgather(np.asarray([local
                                                                     ])))))


def main():
    dtype = jnp.bfloat16
    n = jax.device_count()
    chunk = 1
    mesh = build_mesh()
    shard = NamedSharding(mesh, P(AXIS, None, None, None))
    scalar = NamedSharding(mesh, P(AXIS))
    arms = [a.strip() for a in ARGS.arms.split(",") if a.strip()]

    def n_arr(k):
        return jax.jit(lambda kk=int(k): jnp.full((n, ), kk, jnp.int32),
                       out_shardings=scalar)()

    rprint(f"== {n} devices ({n // CORES_PER_CHIP} chips x {CORES_PER_CHIP} "
           f"cores), {jax.process_count()} process(es)")
    rprint(f"== payload (n={n}, chunk={chunk}, cols), chain {ARGS.n_lo} vs "
           f"{ARGS.n_hi}, median of {ARGS.repeats}")
    rows_out = []
    for cols in [int(c) for c in ARGS.cols.split(",")]:
        kib = n * chunk * cols * 2 / 1024
        x = jax.jit(lambda c=cols: jax.random.normal(jax.random.key(0),
                                                     (n, n, chunk, c), dtype),
                    out_shardings=shard)()
        rprint(f"\n== ({n},{chunk},{cols}) = {kib:.1f} KiB")
        rprint(f"{'arm':>22} {'us':>9} {'verify':>10}   note")
        base = {}
        for arm in arms:
            try:
                err = "skipped"
                if not ARGS.skip_verify:
                    e = float(make_verify(arm, n, chunk, cols, dtype, mesh)(x))
                    err = f"{e:.2e}"
                    if arm != "hier_oneshot_noc" and e > 0.05:
                        err += " BAD"
                fn = make_chain(arm, n, chunk, cols, dtype, mesh)
                t_lo = time_call(fn, (x, n_arr(ARGS.n_lo)), ARGS.repeats,
                                 ARGS.warmup)
                t_hi = time_call(fn, (x, n_arr(ARGS.n_hi)), ARGS.repeats,
                                 ARGS.warmup)
                us = (t_hi - t_lo) / (ARGS.n_hi - ARGS.n_lo) * 1e6
                base[arm] = us
                note = ""
                ref = ("flat_twoshot"
                       if arm == "hier_twoshot" else "flat_oneshot")
                if arm.startswith("hier") and ref in base:
                    d = base[ref] - us
                    note = (f"{'faster' if d > 0 else 'SLOWER'} than {ref} "
                            f"by {abs(d):.2f}")
                rprint(f"{arm:>22} {us:>9.2f} {err:>10}   {note}")
                rows_out.append({
                    "cols": cols,
                    "kib": kib,
                    "arm": arm,
                    "us": us,
                    "err": err
                })
            except Exception as exc:  # noqa: BLE001
                rprint(f"{arm:>22} {'--':>9} {'--':>10}   FAILED: "
                       f"{str(exc)[:90]}")
    if ARGS.csv and IS_LEAD and rows_out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(ARGS.csv)), exist_ok=True)
        with open(ARGS.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        rprint(f"\n== wrote {ARGS.csv}")


if __name__ == "__main__":
    main()
