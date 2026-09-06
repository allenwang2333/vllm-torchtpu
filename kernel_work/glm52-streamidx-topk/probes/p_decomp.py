"""P-DECOMP -- where does the 16 ms B-P1024-1M / B-P1024-256k gap live?

design/combined.md a0.5 measures the gap at +16041.9 us but across two jobs
(j-31ca6c31 and j-59d0b337), and roofline.py:93 says no cross-job comparison
here is worth anything. It then closes 82% of it only by charging the SC's
stage-1 aggregate read (4 GiB vs 1 GiB) at an ASSUMED 337 GB/s, which is in
apparent tension with M1's padding-only form -- the two rows have identical
X_pad (34,603,008) and M1 predicts identical SC time.

This runs both rows in ONE process with all four arms adjacent:

    full-1M    full-256k     -- the gap itself, no longer cross-job
    sc-1M      sc-256k       -- sparsecore_topk alone on an equivalent score
                               matrix, which is the term a0.5's closure rests
                               on

The two SC arms share nothing but shape: same b=1024, same n=1048576, same
p=16, same X_pad, same executable, same 4 GiB input array -- only the i32[1024]
row-lengths vector differs, so the 256k arm has 4 of 16 stage-1 slices live per
row and the 1M arm has 16.

  If sc-1M and sc-256k are equal, M1's padding-only form survives at these
  rows and the gap is TC-side, contradicting a0.5's 82% closure.
  If sc-1M exceeds sc-256k by roughly the 9.5 ms a0.5 predicts, the SC bytes
  term is real and M1 must never be extrapolated to B-P1024 without it.

`full - sc` is NOT computed. It is a difference of two independently drifting
quantities and it moved 2.6x across three measurements of one shape in
j-470d481e.
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

ROWS = ("B-P1024-1M", "B-P1024-256k")
# jitted so the 32-head accumulation over a 4 GiB f32 array fuses instead of
# materialising 32 intermediates of that size.
_synth = jax.jit(lambda T, n, ll: synth_scores(T, n, ll), static_argnums=(0, 1))


def lens_live_for(meta):
    qlen = meta.get("qlen", 1)
    out = np.zeros(meta["T"], np.int32)
    t = 0
    for i in range(meta["R"]):
        for u in range(qlen):
            out[t] = min(int(meta["lens"][i]) - qlen + u + 1, meta["n"])
            t += 1
    return out


def main():
    repeats = int(os.environ.get("REPEATS", 5))
    iters = int(os.environ.get("ITERS", 10))

    full_arms, metas = [], {}
    for row in ROWS:
        args, kwargs, meta = build(row)
        jax.block_until_ready(args[2])
        fn = jax.jit(lambda *x, _kw=kwargs: streamindex_topk(*x, **_kw))
        idx = fn(*args)
        jax.block_until_ready(idx)
        o = np.asarray(idx[0])
        if bool((o == np.arange(K)).all()):
            raise SystemExit(f"{row}: took _fast_path")
        print(f"  full {row}: n={meta['n']} cache="
              f"{meta['cache_bytes'] / 2**20:.0f}MiB filled row0="
              f"{int((o >= 0).sum())}/{K}", flush=True)
        full_arms.append((f"full {row}", (lambda f=fn, a=args: f(*a))))
        metas[row] = meta
        del args, idx

    print("\n== full op, both rows adjacent ==", flush=True)
    rF = interleaved(full_arms, repeats=repeats, iters=iters,
                     ref="full B-P1024-256k")

    del full_arms

    # One score array, one executable; only the lengths vector differs.
    m = metas["B-P1024-1M"]
    print(f"\n  _pick_partition(1024, {m['n']}, {K}) = "
          f"{_pick_partition(m['T'], m['n'], K)}", flush=True)
    s = _synth(m["T"], m["n"], jnp.asarray(lens_live_for(m)))
    jax.block_until_ready(s)
    scfn = jax.jit(lambda a, r: sparsecore_topk(a, K, r))
    sc_arms = []
    for row in ROWS:
        rl = jnp.asarray(lens_live_for(metas[row]))
        out = scfn(s, rl)
        jax.block_until_ready(out)
        print(f"  sc {row}: live_slices_per_row="
              f"{-(-int(np.asarray(rl)[0]) // 65536)}/16 filled row0="
              f"{int((np.asarray(out[0]) >= 0).sum())}/{K}", flush=True)
        sc_arms.append((f"sc {row}", (lambda f=scfn, a=s, r=rl: f(a, r))))

    print("\n== sparsecore_topk alone, both lengths adjacent ==", flush=True)
    rS = interleaved(sc_arms, repeats=repeats, iters=iters,
                     ref="sc B-P1024-256k")
    return rF, rS


if __name__ == "__main__":
    main()
