"""P-SCBW -- the SparseCore's achieved HBM read bandwidth at a resident slice.

design/combined.md a0.5 closes 82% of the 16 ms B-P1024-1M/256k gap only by
charging the SC's stage-1 read at 337 GB/s. That 337 is `assumed` there
(carried from area 5). If the real figure is materially higher the closure
fails and the gap is `unmodelled`. This probe measures it.

Method. `_sc_topk_direct` DMAs the whole resident slice unconditionally
(sparsecore_topk.py:160, `pl.ds(s, slice_len)`) and only then walks
`cdiv(i_hi, 16)` chunks. So handing it row_lengths = 16 buys a full-width copy
with a one-chunk walk: every pass, every scan, the -1 fill and the output DMA
are identical across arms, and the only thing that scales with `n` is the
input DMA. b = 32 keeps it to a single subcore wave, so the aggregate moved
per call is 32 * n * 4 bytes.

A least-squares line through (bytes, time) gives the slope directly. The
intercept is everything else and is not a bandwidth.

Two controls, because a slope on four points is easy to over-read:
  - `n=65536 fullwalk` (row_lengths = n) is the same DMA with the walk turned
    on. Its distance above `n=65536` is the walk, and it says how much of the
    incumbent's time at a 256 KiB slice is transfer and how much is walk.
  - the fit's residuals are printed per point.
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
from shapes import K  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    _sc_topk_direct  # noqa: E402

B = 32                                   # one subcore wave
NS = (4096, 16384, 32768, 65536)         # 16 / 64 / 128 / 256 KiB per subcore


def main():
    repeats = int(os.environ.get("REPEATS", 5))
    iters = int(os.environ.get("ITERS", 20))

    arms, bytes_per_call = [], {}
    for n in NS:
        s = synth_scores(B, n, np.full(B, n, np.int32))
        jax.block_until_ready(s)
        rl = jnp.asarray(np.full(B, 16, np.int32))
        fn = jax.jit(lambda a, r: _sc_topk_direct(a, K, r, write_empty=True))
        jax.block_until_ready(fn(s, rl))
        name = f"n={n:<6} ({n * 4 // 1024:>3} KiB/subcore)"
        arms.append((name, (lambda f=fn, a=s, r=rl: f(a, r))))
        bytes_per_call[name] = B * n * 4

    n = NS[-1]
    s = synth_scores(B, n, np.full(B, n, np.int32))
    jax.block_until_ready(s)
    rl = jnp.asarray(np.full(B, n, np.int32))
    fnw = jax.jit(lambda a, r: _sc_topk_direct(a, K, r, write_empty=True))
    jax.block_until_ready(fnw(s, rl))
    wname = f"n={n} FULLWALK (control)"
    arms.append((wname, (lambda f=fnw, a=s, r=rl: f(a, r))))

    res = interleaved(arms, repeats=repeats, iters=iters, ref=arms[0][0])

    xs = np.array([bytes_per_call[nm] for nm, _ in arms[:-1]], float)
    ys = np.array([res[nm]["median"] for nm, _ in arms[:-1]], float) * 1e-6
    slope, inter = np.polyfit(xs, ys, 1)
    print(f"\n  least squares over {len(xs)} DMA-only arms:")
    print(f"    slope     {slope:.4e} s/byte  ->  aggregate SC read "
          f"{1 / slope / 1e9:.1f} GB/s")
    print(f"    intercept {inter * 1e6:.1f} us  (everything that is not the "
          f"input DMA)")
    for x, y, (nm, _) in zip(xs, ys, arms[:-1]):
        pred = slope * x + inter
        print(f"    {nm:32} {x / 2**20:7.2f} MiB  meas {y*1e6:8.1f} us  "
              f"fit {pred*1e6:8.1f} us  resid {100*(y-pred)/y:+6.1f}%")
    walk = res[wname]["median"] - res[arms[-2][0]]["median"]
    print(f"\n  FULLWALK control: {res[wname]['median']:.1f} us vs "
          f"{res[arms[-2][0]]['median']:.1f} us DMA-only at the same 256 KiB "
          f"slice -> the 65536-word walk itself costs {walk:.1f} us "
          f"({walk / 65536 * 1e3:.3f} ns/word)")
    return res, slope, inter


if __name__ == "__main__":
    main()
