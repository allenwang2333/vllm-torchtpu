"""P-ITER -- is the in-body fixed cost per ITERATION, per make_async_copy, or
scalar index arithmetic?

design/combined.md a1 has three live hypotheses and says they predict opposite
levers. This runs the arms that separate the first two, at B-D1-1M, with the
bytes held exactly fixed.

`_fetch_bkv` (streamindex_topk.py:160-197) starts `seq_batch_size * bkv_p`
one-page copies per in-body iteration, and the loop runs
`nbkv = cdiv(kv_len_max, page_size * bkv_p)` times. At B-D1-1M, R = D = 1 so
`decode_batch_end = (1 // 4) * 4 = 0` and the live launch is the sbs = 1 one.
Raising bkv_p therefore holds

    total copies = nbkv * sbs * bkv_p = 1048576/512 = 2048   -- CONSTANT
    KV bytes     = 1048576 * 256 B    = 256 MiB              -- CONSTANT
    bytes/copy   = 512 * 256 B        = 128 KiB              -- CONSTANT
    score n      = align_to(2048, bkv_p) * 512 = 1048576     -- CONSTANT, so
                   the SparseCore half of the op does not move at all

and varies only

    iterations   = 2048 / bkv_p                              -- 8x across arms

which is exactly the quantity the two hypotheses disagree about:

    per-iteration (area 3's 512 ns grid-step constant, transferred to the
      fori_loop body): time falls, by ~0.5 us per removed iteration
    per-copy (area 2): FLAT -- the copies were never removed
    scalar index arithmetic (area 7): falls, because the clamp-and-bounds
      arithmetic is per copy per iteration... no: it is per copy, and copies
      are constant, so area 7 also predicts flat here. Arm 5 of a1 (the
      host-precomputed byte-offset table) is what separates area 7, and it is
      NOT run here -- see design/probes.md.

All four arms share one built input; only the static `num_kv_pages_per_block`
differs, so each is a separately compiled executable over identical data.
"""
import os
import sys

import jax
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get("INST_DIR",
                                  os.path.join(_HERE, "..", "instruments")))

from bench import interleaved  # noqa: E402
from shapes import K, build, cdiv  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402

ROW = os.environ.get("ROW", "B-D1-1M")
BKV_PS = (2, 4, 8, 16)


def main():
    repeats = int(os.environ.get("REPEATS", 5))
    iters = int(os.environ.get("ITERS", 20))

    args, kwargs, meta = build(ROW)
    jax.block_until_ready(args[2])
    print(f"  {ROW}: n={meta['n']}  cache="
          f"{meta['cache_bytes'] / 2**20:.0f}MiB  seqlen={meta['seqlen']}",
          flush=True)

    arms, ref = [], None
    for bp in BKV_PS:
        kw = dict(kwargs, num_kv_pages_per_block=(bp, bp, bp))
        fn = jax.jit(lambda *x, _kw=kw: streamindex_topk(*x, **_kw))
        idx = fn(*args)
        jax.block_until_ready(idx)
        o = np.asarray(idx)
        if bool((o[0] == np.arange(K)).all()):
            raise SystemExit(f"{ROW} bkv_p={bp}: took _fast_path")
        nbkv = cdiv(meta["seqlen"], meta["psz"] * bp)
        name = (f"bkv_p={bp:<2} iters={nbkv:<5} copies/iter={bp}")
        print(f"  {name}  total_copies={nbkv * bp}  filled row0="
              f"{int((o[0] >= 0).sum())}/{K}", flush=True)
        if ref is None:
            ref, ref_out = name, o
        else:
            same = all(set(ref_out[t][ref_out[t] >= 0].tolist())
                       == set(o[t][o[t] >= 0].tolist())
                       for t in range(o.shape[0]))
            print(f"      selected sets identical to bkv_p={BKV_PS[0]}: {same}",
                  flush=True)
        arms.append((name, (lambda f=fn: f(*args))))

    return interleaved(arms, repeats=repeats, iters=iters, ref=ref)


if __name__ == "__main__":
    main()
