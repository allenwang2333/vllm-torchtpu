"""P-P16 -- is `_pick_partition`'s growth branch a net loss at b = 16?

The claim under test (design/combined.md a0.4): at A-D16-8k the growth loop
raises p from 1 to 2, which does NOT reduce the subcore-wave count
(cdiv(16*1,32) == cdiv(16*2,32) == 1), so it buys a slice half as wide and pays
a second pl.kernel dispatch plus two jnp.take_along_axis gathers over a padded
[16, 10240] row. Predicted `d time -9.0%` at A-D16-8k and exactly 0 at every
other reachable shape, `modelled`.

Six arms, all in ONE process, timed adjacently and repeated (bench.interleaved).
Two levels, because they answer different halves of the claim:

  full-op   fA32   A-D32-8k                incumbent, p = 1
            fA16   A-D16-8k                incumbent, p = 2
            fA16n  A-D16-8k, growth off    p = 1     <- the paired A/B
  sc-only   sA32   sparsecore_topk[32,10240]   p = 1
            sA16   sparsecore_topk[16,10240]   p = 2
            sA16n  the same, growth off        p = 1  <- the paired A/B

fA16 vs fA16n and sA16 vs sA16n are true paired A/Bs: one shape, one input,
one line of source different. fA16 vs fA32 is NOT -- it is the anomaly being
explained, two different shapes, and it is here only so the inversion is
reproduced inside one process instead of across two jobs.

The growth-off arms import probes/sc_nogrow.py and probes/sit_nogrow.py, copies
of the two shipping files carrying that one guard and the one import that
reaches it. Neither shipping file is modified. Copies rather than a
monkey-patch because both shipping entry points are themselves `jax.jit`
objects: patched in place, the two arms would collide in one pjit cache entry
and the second would time the first arm's executable.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get("INST_DIR",
                                  os.path.join(_HERE, "..", "instruments")))

from bench import interleaved, synth_scores  # noqa: E402
from shapes import K, build  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _pick_partition, sparsecore_topk)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402

import sc_nogrow  # noqa: E402
from sit_nogrow import streamindex_topk as streamindex_topk_nogrow  # noqa: E402


def lens_live_for(meta):
    """Per-token effective row length, matching _effective_row_lengths."""
    qlen = meta.get("qlen", 1)
    out = np.zeros(meta["T"], np.int32)
    t = 0
    for i in range(meta["R"]):
        for u in range(qlen):
            out[t] = min(int(meta["lens"][i]) - qlen + u + 1, meta["n"])
            t += 1
    return out


def full_arm(row, nogrow):
    """Build a compiled full-op callable for `row`, with growth on or off."""
    args, kwargs, meta = build(row)
    jax.block_until_ready(args[2])
    impl = streamindex_topk_nogrow if nogrow else streamindex_topk
    fn = jax.jit(lambda *x: impl(*x, **kwargs))
    idx = fn(*args)
    jax.block_until_ready(idx)
    h = np.asarray(idx[0])
    if bool((h == np.arange(K)).all()):
        raise SystemExit(f"{row}: took _fast_path -- measures nothing")
    filled = int((h >= 0).sum())
    print(f"  {row} nogrow={nogrow}: n={meta['n']} p="
          f"{(sc_nogrow._pick_partition if nogrow else _pick_partition)(meta['T'], meta['n'], K)}"
          f"  filled row0 {filled}/{K}", flush=True)
    return (lambda: fn(*args)), meta, np.asarray(idx)


def sc_arm(T, meta_n, lens_live, nogrow, seed=1):
    s = synth_scores(T, meta_n, lens_live, seed=seed)
    jax.block_until_ready(s)
    rl = jnp.asarray(lens_live, jnp.int32)
    impl = sc_nogrow.sparsecore_topk if nogrow else sparsecore_topk
    fn = jax.jit(lambda a, b: impl(a, K, b))
    out = fn(s, rl)
    jax.block_until_ready(out)
    pp = sc_nogrow._pick_partition if nogrow else _pick_partition
    print(f"  sc b={T} n={meta_n} nogrow={nogrow}: p={pp(T, meta_n, K)} "
          f"filled row0 {int((np.asarray(out[0]) >= 0).sum())}/{K}", flush=True)
    return (lambda: fn(s, rl)), np.asarray(out)


def main():
    repeats = int(os.environ.get("REPEATS", 5))
    iters = int(os.environ.get("ITERS", 20))

    print("== building ==", flush=True)
    f32, m32, i32 = full_arm("A-D32-8k", False)
    f16, m16, i16 = full_arm("A-D16-8k", False)
    f16n, _, i16n = full_arm("A-D16-8k", True)

    # Same-set check on the two A-D16 arms. Suppressing the growth branch must
    # not change WHICH columns come back; if it does, the -9.0% is a kernel
    # doing less work, not a kernel doing the same work faster.
    same = all(set(i16[t][i16[t] >= 0].tolist())
               == set(i16n[t][i16n[t] >= 0].tolist())
               for t in range(i16.shape[0]))
    n_diff = sum(len(set(i16[t][i16[t] >= 0].tolist())
                     ^ set(i16n[t][i16n[t] >= 0].tolist()))
                 for t in range(i16.shape[0]))
    print(f"\n  A-D16-8k grow vs nogrow: identical selected SETS on all "
          f"{i16.shape[0]} rows: {same}  (symmetric-difference total {n_diff} "
          f"of {i16.shape[0] * K} slots)", flush=True)

    ll16, ll32 = lens_live_for(m16), lens_live_for(m32)
    s32, _ = sc_arm(32, m32["n"], ll32, False)
    s16, o16 = sc_arm(16, m16["n"], ll16, False)
    s16n, o16n = sc_arm(16, m16["n"], ll16, True)
    sc_same = all(set(o16[t][o16[t] >= 0].tolist())
                  == set(o16n[t][o16n[t] >= 0].tolist())
                  for t in range(o16.shape[0]))
    print(f"  sparsecore_topk b=16 grow vs nogrow: identical sets: {sc_same}",
          flush=True)

    print("\n== full op, arms adjacent ==", flush=True)
    r1 = interleaved([("fA16 (p=2, incumbent)", f16),
                      ("fA16n (p=1, growth off)", f16n),
                      ("fA32 (p=1, b=32)", f32)],
                     repeats=repeats, iters=iters, ref="fA16 (p=2, incumbent)")

    print("\n== sparsecore_topk alone, arms adjacent ==", flush=True)
    r2 = interleaved([("sA16 (p=2, incumbent)", s16),
                      ("sA16n (p=1, growth off)", s16n),
                      ("sA32 (p=1, b=32)", s32)],
                     repeats=repeats, iters=iters, ref="sA16 (p=2, incumbent)")
    return r1, r2, same, sc_same


if __name__ == "__main__":
    main()
