"""P-H -- is the ~140 us floor under every arm in this corpus the HARNESS or a
real per-launch cost?

Every timing in the corpus sits above ~140 us. P-SCBW's n=4096 arm -- a
SparseCore call that DMAs 16 KiB and walks sixteen words -- costs 140.1 us
against a modelled ceiling of <= 20 us. Call that floor H.

bench.timeit / bench.interleaved time exactly one
    t0 -> fn() -> jax.block_until_ready(out) -> t1
per iteration. That interval contains three separable terms:

    host_dispatch   Python + JAX call overhead to enqueue one execution
    device_exec     what the chip actually does
    sync            the block_until_ready round trip (host <-> device)

A single un-amortized call measures  max(host_dispatch, device_exec) + sync,
because dispatch and execution overlap but the sync cannot.  N chained calls
inside ONE block_until_ready measure  N * max(host_dispatch, device_exec) +
sync, so total/N -> max(host_dispatch, device_exec) as N grows.  The gap
between the two is `sync`, and `sync` is pure harness.

Arms, all driven through bench.interleaved so whatever H is, it is present
identically in each and the 3-9% run-order drift is common-mode:

  null              jit(x -> x+1) on a scalar.  device_exec ~ 0, so this is
                    the direct estimate of host_dispatch + sync.
  null_x100         the same function, 100 calls chained by a TRUE data
                    dependency inside one block_until_ready.
  sc_tiny_k2048     sparsecore_topk(b=32, n=4096, k=2048, row_lengths=16).
                    _pick_partition(32,4096,2048) = 1, so this lands in
                    _sc_topk_direct and is byte-for-byte P-SCBW's n=4096 arm.
  sc_tiny_k2048x100 the decisive arm.  A per-launch DEVICE cost survives
                    amortization; a per-dispatch HOST cost does not.
  sc_tiny_k16       k as small as legal (multiple of LANES=16).
  sc_tiny_k16_x100

DCE guard.  jax.jit calls dispatch eagerly and asynchronously; JAX does not
memoize outputs and does not fuse across two Python-level jit calls, so a
repeated call is a repeated execution.  That is an argument, not a
measurement, so the script also measures it two ways:
  (a) the null chain is a genuine value dependency (x = f(x)), which cannot
      be eliminated even in principle;
  (b) the sc chain is given N DISTINCT row_lengths device buffers (all equal
      to 16, so the work is identical) and then TOTAL block time is swept
      over N = 1..100 and least-squares fitted.  A linear total with a tight
      fit is positive evidence that all N executions happened; a flat or
      sublinear total would mean they did not, and the amortized number would
      then not be a measurement.  The fit is printed.
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get("INST_DIR",
                                  os.path.join(_HERE, "..", "..",
                                               "instruments")))

from bench import interleaved, synth_scores  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _pick_partition, sparsecore_topk)

B = 32
N_SC = 4096
ROWLEN = 16
NCHAIN = int(os.environ.get("NCHAIN", 100))
SWEEP = (1, 2, 5, 10, 25, 50, 100)


def build_null():
    f = jax.jit(lambda x: x + 1.0)
    x0 = jnp.asarray(1.0, jnp.float32)
    jax.block_until_ready(f(x0))

    def one():
        return f(x0)

    def chain(n=NCHAIN):
        x = x0
        for _ in range(n):
            x = f(x)          # true data dependency: cannot be eliminated
        return x

    return one, chain


def build_sc(k):
    p = _pick_partition(B, N_SC, k)
    s = synth_scores(B, N_SC, np.full(B, N_SC, np.int32))
    jax.block_until_ready(s)
    # NCHAIN distinct device buffers, every one holding the same value, so the
    # work per call is identical but no call can be served from another call.
    rls = [jnp.asarray(np.full(B, ROWLEN, np.int32)) for _ in range(NCHAIN)]
    jax.block_until_ready(rls)

    def f(r):
        return sparsecore_topk(s, k, r, write_empty_rows=True)

    out0 = f(rls[0])
    jax.block_until_ready(out0)
    filled = int((np.asarray(out0[0]) >= 0).sum())

    def one():
        return f(rls[0])

    def chain(n=NCHAIN):
        out = None
        for i in range(n):
            out = f(rls[i % NCHAIN])
        return out

    print(f"  sc k={k:<5} partition p={p} "
          f"({'DIRECT single-stage' if p == 1 else '2-STAGE'})  "
          f"out{tuple(out0.shape)}  filled slots row0 = {filled}/{k} "
          f"(row_lengths={ROWLEN}, so {min(ROWLEN, k)} is correct)", flush=True)
    return one, chain


def sweep_total(chain, label):
    """Total wall time of ONE chained block, vs N.  The DCE guard."""
    print(f"\n  {label}: total block time vs N "
          f"(each entry is the median of 5 blocks)", flush=True)
    xs, ys = [], []
    for n in SWEEP:
        jax.block_until_ready(chain(n))          # compile/warm at this N
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            out = chain(n)
            jax.block_until_ready(out)
            ts.append((time.perf_counter() - t0) * 1e6)
        med = float(np.median(ts))
        xs.append(float(n))
        ys.append(med)
        print(f"    N={n:<5} total {med:10.1f} us   per call {med / n:8.2f} us",
              flush=True)
    xs, ys = np.array(xs), np.array(ys)
    slope, inter = np.polyfit(xs, ys, 1)
    pred = slope * xs + inter
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"    least squares: total(N) = {slope:.2f} us * N + {inter:.1f} us "
          f"  R^2 = {r2:.5f}", flush=True)
    print(f"      slope     {slope:8.2f} us  <- per-call cost that SURVIVES "
          f"amortization", flush=True)
    print(f"      intercept {inter:8.1f} us  <- one-off per-block cost "
          f"(the block_until_ready sync)", flush=True)
    return dict(slope=float(slope), intercept=float(inter), r2=r2,
                totals=dict(zip([int(v) for v in xs], ys)))


def main():
    repeats = int(os.environ.get("REPEATS", 5))
    iters = int(os.environ.get("ITERS", 20))
    print(f"jax {jax.__version__}  devices {jax.devices()}", flush=True)
    print(f"NCHAIN={NCHAIN} repeats={repeats} iters={iters}\n", flush=True)

    n_one, n_chain = build_null()
    sc2k_one, sc2k_chain = build_sc(2048)
    sc16_one, sc16_chain = build_sc(16)

    arms = [
        ("null", n_one),
        (f"null_x{NCHAIN}", n_chain),
        ("sc_tiny_k2048", sc2k_one),
        (f"sc_tiny_k2048_x{NCHAIN}", sc2k_chain),
        ("sc_tiny_k16", sc16_one),
        (f"sc_tiny_k16_x{NCHAIN}", sc16_chain),
    ]

    print(f"\n{'=' * 74}\ninterleaved, {repeats} repeats x {iters} iters, "
          f"arms timed adjacently\n{'=' * 74}", flush=True)
    res = interleaved(arms, repeats=repeats, iters=iters, ref="null")

    print(f"\n{'=' * 74}\nper-call figures (x{NCHAIN} arms divided by "
          f"{NCHAIN})\n{'=' * 74}", flush=True)
    print(f"  {'arm':26} {'raw med us':>12} {'per call us':>13} "
          f"{'across-repeat drift':>20}", flush=True)
    per = {}
    for name, _ in arms:
        div = NCHAIN if name.endswith(f"_x{NCHAIN}") else 1
        pc = res[name]["median"] / div
        per[name] = pc
        print(f"  {name:26} {res[name]['median']:12.1f} {pc:13.2f} "
              f"{res[name]['drift']:19.1%}", flush=True)

    print(f"\n{'=' * 74}\nDCE guard: does the chain really execute N times?"
          f"\n{'=' * 74}", flush=True)
    fit_null = sweep_total(n_chain, "null")
    fit_sc = sweep_total(sc2k_chain, "sc_tiny_k2048")

    print(f"\n{'=' * 74}\ndecomposition\n{'=' * 74}", flush=True)
    for label, one_name, fit in (
            ("null", "null", fit_null),
            ("sc_tiny_k2048", "sc_tiny_k2048", fit_sc)):
        un = res[one_name]["median"]
        amort = fit["slope"]
        print(f"  {label}", flush=True)
        print(f"    un-amortized (what bench.timeit reports) {un:9.1f} us",
              flush=True)
        print(f"    per-call floor that survives amortization {amort:8.2f} us",
              flush=True)
        print(f"    difference = per-iteration sync/harness  {un - amort:9.1f} us"
              f"  ({100 * (un - amort) / un:.0f}% of the reported number)",
              flush=True)
    print(f"\n  H (the corpus floor) is ~140 us.  Of that, the part that is "
          f"harness\n  is the sc_tiny difference above; the part that is a real "
          f"per-launch\n  cost is the sc_tiny amortized slope above.", flush=True)


if __name__ == "__main__":
    main()
