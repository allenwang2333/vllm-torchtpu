"""Interleaved microbenchmark: O9's arms against the incumbent, in one job.

Arms are timed adjacently inside each repeat and differenced there, because
run-order drift at these shapes is 3-9% -- larger than the effect being looked
for. `bench.interleaved` from ../instruments is the yardstick; this file only
builds the thunks and hands them over.

    python3 run_bench.py --row B-D4-256k --mode sc --arms inc,base,r2,r3
    python3 run_bench.py --row B-D4-256k --mode full --arms inc,base,c4d2

Two modes, two questions. `sc` is the selector alone on a synthetic score
matrix -- higher resolution on a selector change, since the scoring pass is not
in the denominator. `full` is the whole streamindex op, which is what ships.
NEVER subtract one from the other: `full - sc` moved 2.6x across three
measurements of one shape (j-470d481e); it is not a quantity.

`bench.interleaved` calls ZERO-ARGUMENT thunks, so every arm is bound with a
default argument rather than a closure over the loop variable.
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

INSTRUMENTS = os.environ.get(
    "INST_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "instruments"))
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
    ap.add_argument("--arms", default="inc,base,r2,r3")
    ap.add_argument("--mode", default="sc", choices=("full", "sc"))
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
          f"  gate/yardstick: {os.path.realpath(INSTRUMENTS)}\n{'='*74}",
          flush=True)
    bad = [nm for nm in names if nm in arms.ABLATION_ARMS]
    if bad:
        print(f"  NOTE: {','.join(bad)} are ABLATIONS -- their output is "
              f"deliberately wrong and they are attribution instruments, not "
              f"candidates.", flush=True)

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
            print(f"  arm {nm:6} compiled in {time.perf_counter()-t1:5.1f}s   "
                  f"filled slots row0: {int((idx[0] >= 0).sum())}/{shapes.K}",
                  flush=True)
            built.append((nm, lambda f=fn: f(*args)))
    else:
        # Per-token-row effective lengths, matching bench.py --decompose so the
        # synthetic matrix is the one every other measurement used.
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
        ref_idx = None
        for nm in names:
            t1 = time.perf_counter()
            sel = arms.selector(nm)
            fn = jax.jit(lambda s, r, _s=sel: _s(s, shapes.K, r))
            out = np.asarray(jax.block_until_ready(fn(sc, rl)))
            # Byte-identity against the incumbent's selected set. A schedule
            # change must not move one index; an ablation is expected to.
            note = ""
            if ref_idx is None:
                ref_idx = out
            else:
                d = symmetric_difference(ref_idx, out)
                note = (f"  index-set symdiff vs {names[0]}: {d} "
                        f"({'IDENTICAL' if d == 0 else 'DIFFERS'})")
            print(f"  arm {nm:6} compiled in {time.perf_counter()-t1:5.1f}s"
                  f"{note}", flush=True)
            built.append((nm, lambda f=fn: f(sc, rl)))

    res = bench.interleaved(built, repeats=a.repeats, iters=a.iters,
                            ref=names[0])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(row=a.row, mode=a.mode, n=meta["n"], result=res), f,
                      indent=2)
        print(f"wrote {a.json}", flush=True)


def symmetric_difference(a, b):
    """Total |A xor B| over rows, ignoring -1 padding and emission order."""
    total = 0
    for ra, rb in zip(a, b):
        sa = set(int(x) for x in ra if x >= 0)
        sb = set(int(x) for x in rb if x >= 0)
        total += len(sa ^ sb)
    return total


if __name__ == "__main__":
    main()
