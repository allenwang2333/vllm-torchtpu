"""Interleaved microbenchmark: O7's arms against the incumbent, in one job.

Every arm is timed adjacently inside each repeat and differenced there, because
run-order drift at these shapes is 3-9% -- larger than the effects being looked
for. `bench.interleaved` from ../instruments is the yardstick; this file only
builds the arms and hands them over.

    python3 run_bench.py --row B-D4-256k --arms inc,base,fz,scnt,cf,fc,all
    python3 run_bench.py --row A-D64-8k --mode sc     # selector alone

Two modes, and they answer different questions:
  full  the whole streamindex op. This is the headline: the design's predicted
        bands are fractions of the full-block time.
  sc    the selector alone on a synthetic score matrix. Higher resolution on the
        change itself, since the scoring pass is not in the denominator.
NEVER subtract one from the other. `full - sc` moved 2.6x across three
measurements of one shape (j-470d481e); it is not a quantity.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arms  # noqa: E402

INSTRUMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "instruments")
sys.path.insert(0, INSTRUMENTS)


def load_instrument(name):
    path = os.path.join(INSTRUMENTS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"instr_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--arms", default="inc,base,fz,scnt,cf,fc,all")
    ap.add_argument("--mode", default="full", choices=("full", "sc"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    bench = load_instrument("bench")
    shapes = load_instrument("shapes")
    names = [s.strip() for s in a.arms.split(",")]
    for nm in names:
        if nm not in arms.CONFIGS:
            raise SystemExit(f"unknown arm {nm!r}; have {sorted(arms.CONFIGS)}")

    print(f"\n{'='*74}\n{a.row}  mode={a.mode}  arms={','.join(names)}\n"
          f"{'='*74}", flush=True)
    t0 = time.perf_counter()
    args, kwargs, meta = shapes.build(a.row)
    jax.block_until_ready(args[2])
    print(f"  built in {time.perf_counter()-t0:.1f}s  n={meta['n']}  "
          f"num_blocks={meta['num_blocks']}  "
          f"cache={meta['cache_bytes']/2**20:.0f}MiB", flush=True)

    built = []
    if a.mode == "full":
        for nm in names:
            t1 = time.perf_counter()
            fn = arms.full_op(nm, kwargs, args=args)
            idx = np.asarray(fn(*args)[:1])
            if bool((idx[0] == np.arange(shapes.K)).all()):
                raise SystemExit(f"{a.row}: took _fast_path -- measures nothing")
            n_filled = int((idx[0] >= 0).sum())
            print(f"  arm {nm:6} compiled in {time.perf_counter()-t1:5.1f}s   "
                  f"filled slots row0: {n_filled}/{shapes.K}", flush=True)
            built.append((nm, (lambda f=fn: f(*args))))
    else:
        # Per-token-row effective lengths, matching bench.py's --decompose so
        # the synthetic matrix is the one the other measurements used.
        qlen, lens = meta.get("qlen", 1), meta["lens"]
        lens_live = np.zeros(meta["T"], dtype=np.int32)
        t = 0
        for i in range(meta["R"]):
            for u in range(qlen):
                lens_live[t] = min(int(lens[i]) - qlen + u + 1, meta["n"])
                t += 1
        sc = bench.synth_scores(meta["T"], meta["n"], lens_live)
        rl = jnp.asarray(lens_live, jnp.int32)
        jax.block_until_ready(sc)
        for nm in names:
            t1 = time.perf_counter()
            sel = arms.selector(nm)
            fn = jax.jit(lambda s, r, _sel=sel: _sel(s, shapes.K, r))
            jax.block_until_ready(fn(sc, rl))
            print(f"  arm {nm:6} compiled in {time.perf_counter()-t1:5.1f}s",
                  flush=True)
            built.append((nm, (lambda f=fn: f(sc, rl))))

    res = bench.interleaved(built, repeats=a.repeats, iters=a.iters,
                            ref=names[0])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(row=a.row, mode=a.mode, n=meta["n"], result=res),
                      f, indent=2)
        print(f"wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
