"""Microbenchmark for streamindex_topk at reachable GLM-5.2 shapes.

  python3 bench.py --rows A-D64-8k,A-P1024-8k          # end-to-end op
  python3 bench.py --rows B-D4-1M --decompose          # also time SC alone

Reports min / median / p90 over two independent measurement blocks so the
run-to-run spread is visible rather than hidden behind a single number. Nothing
here is a verdict on its own: read the spread before reading the delta.
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import (ROWS, H, K, build, cdiv)  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    sparsecore_topk  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402


def timeit(fn, iters=20, warmup=5, blocks=2):
    """Time `fn` in two independent blocks. Returns per-block stats in us."""
    jax.block_until_ready(fn())
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)
    res = []
    for _ in range(blocks):
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            ts.append((time.perf_counter() - t0) * 1e6)
        ts = np.array(ts)
        res.append(dict(min=float(ts.min()), median=float(np.median(ts)),
                        p90=float(np.percentile(ts, 90)), max=float(ts.max())))
    return res


def synth_scores(T, n, lens_live, seed=1):
    """A score matrix with the distribution the kernel actually produces.

    Do not use uniform f32 here: the radix select terminates early on a flat
    distribution and would flatter the kernel. The real score is
    sum_h w_h * relu(q_h . k), a weighted sum of 32 rectified dot products --
    right-skewed with an atom at the ReLU floor.
    """
    key = jax.random.key(seed)
    acc = jnp.zeros((T, n), jnp.float32)
    for h in range(H):
        key, k1, k2 = jax.random.split(key, 3)
        g = jax.random.normal(k1, (T, n), jnp.float32) * np.sqrt(128.0)
        w = jax.random.uniform(k2, (T, 1), jnp.float32, -1.5, 1.5)
        acc = acc + w * jnp.maximum(g, 0.0)
    col = jnp.arange(n, dtype=jnp.int32)[None, :]
    live = jnp.asarray(lens_live, jnp.int32)[:, None]
    return jnp.where(col < live, acc, -jnp.inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--decompose", action="store_true",
                    help="also time sparsecore_topk alone on an equivalent "
                         "score matrix, to split scoring from selection")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    out = []
    for name in a.rows.split(","):
        name = name.strip()
        if name not in ROWS:
            raise SystemExit(f"unknown row {name!r}; have {sorted(ROWS)}")
        print(f"\n{'='*74}\n{name}\n{'='*74}", flush=True)
        t0 = time.perf_counter()
        args, kwargs, meta = build(name)
        jax.block_until_ready(args[2])
        print(f"  built in {time.perf_counter()-t0:.1f}s  "
              f"n={meta['n']}  num_blocks={meta['num_blocks']}  "
              f"cache={meta['cache_bytes']/2**20:.0f}MiB  "
              f"score_buf={meta['score_bytes']/2**20:.0f}MiB", flush=True)

        fn = jax.jit(lambda *x: streamindex_topk(*x, **kwargs))
        idx = fn(*args)
        jax.block_until_ready(idx)

        # Correctness smoke: the op must not have taken _fast_path (which emits
        # [0,1,2,...] and measures nothing), and must fill k slots on live rows.
        h = np.asarray(idx[:min(4, meta['R'])])
        took_fast_path = bool((h[0] == np.arange(K)).all())
        n_filled = int((h[0] >= 0).sum())
        print(f"  _fast_path taken: {took_fast_path}   filled slots row0: "
              f"{n_filled}/{K}", flush=True)
        if took_fast_path:
            raise SystemExit(f"{name}: took _fast_path -- measures nothing")

        blocks = timeit(lambda: fn(*args), iters=a.iters)
        for i, b in enumerate(blocks):
            print(f"  full  block{i}  min {b['min']:9.1f}  med {b['median']:9.1f}"
                  f"  p90 {b['p90']:9.1f}  max {b['max']:9.1f}  us", flush=True)
        spread = abs(blocks[1]['median'] - blocks[0]['median']) / blocks[0]['median']
        print(f"  full  block-to-block median spread: {spread:.1%}", flush=True)

        rec = dict(row=name, n=meta['n'], num_blocks=meta['num_blocks'],
                   cache_MiB=meta['cache_bytes'] / 2**20,
                   score_MiB=meta['score_bytes'] / 2**20,
                   full=blocks, spread=spread)

        if a.decompose:
            # Per-token-row effective length, matching _effective_row_lengths:
            # a decode row sees its whole sequence; prefill row t of a chunk of
            # qlen sees up to its own position. Padded rows are 0 and the
            # kernel skips their DMA, so they must be 0 here too.
            qlen, lens = meta.get('qlen', 1), meta['lens']
            lens_live = np.zeros(meta['T'], dtype=np.int32)
            t = 0
            for i in range(meta['R']):
                for u in range(qlen):
                    lens_live[t] = min(int(lens[i]) - qlen + u + 1, meta['n'])
                    t += 1
            sc = synth_scores(meta['T'], meta['n'], lens_live)
            jax.block_until_ready(sc)
            rl = jnp.asarray(lens_live, jnp.int32)
            scfn = jax.jit(lambda s, r: sparsecore_topk(s, K, r))
            jax.block_until_ready(scfn(sc, rl))
            scb = timeit(lambda: scfn(sc, rl), iters=a.iters)
            for i, b in enumerate(scb):
                print(f"  sc    block{i}  min {b['min']:9.1f}  "
                      f"med {b['median']:9.1f}  p90 {b['p90']:9.1f}  us",
                      flush=True)
            impl = blocks[0]['median'] - scb[0]['median']
            print(f"  implied scoring+fill = full - sc = {impl:9.1f} us "
                  f"({impl/blocks[0]['median']:.0%} of full)", flush=True)
            rec['sc'] = scb
            rec['implied_scoring_us'] = impl

        out.append(rec)

    print("\n" + "=" * 74)
    print(f"{'row':16} {'n':>9} {'full med us':>12} {'spread':>8} {'sc med us':>11}")
    for r in out:
        s = f"{r['sc'][0]['median']:11.1f}" if 'sc' in r else " " * 11
        print(f"{r['row']:16} {r['n']:>9} {r['full'][0]['median']:12.1f} "
              f"{r['spread']:>7.1%} {s}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
