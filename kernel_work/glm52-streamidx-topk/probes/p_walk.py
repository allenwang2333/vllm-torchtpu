"""P-WALK -- does SparseCore time depend on WALKED words at fixed padded extent?

design/combined.md fits M1, `T_sc = C + f * X_pad`, where X_pad = sum over
subcore-waves of W * slice_len -- a pure PADDING quantity with no live-data
term. M1 therefore predicts EXACTLY ZERO for E3 (block-max pruning), which
removes walked columns without touching W or slice_len. That is a falsifiable
prediction and this probe is the test of it, not a confirmation of it.

Two levels.

LEVEL A -- `_sc_topk_direct`, the cleanest form of the question.
  One score array f32[256, 65536], generated once and never regenerated.
  One compiled executable. The ONLY thing that differs between arms is the
  i32[256] row-lengths vector handed in as data. So:
    - `slice_len` is 65536 in every arm            (static, identical)
    - `num_waves` is cdiv(256,32) = 8 in every arm (static, identical)
    - X_pad = 8 * 65536 = 524288 in every arm      -> M1 predicts one time
    - the HBM->VMEM DMA copies the full slice in every arm (:160 is not
      bounded by eff_len), so BYTES read are identical too
    - `walk_chunks = cdiv(i_hi, 16)` varies 8x
  Nothing but walked words moves. If the arms are flat, M1's padding-only form
  is confirmed and E3 buys nothing on the walk. If they scale with the length,
  M1 needs a second term and E3 has a value.

  The `dead-half` arm varies the same total walked words a different way --
  half the ROWS have length 0, so their DMA and every pass is skipped by
  `pl.when(i_hi > 0)` and only the histogram scans remain. It separates "a
  dead slice is cheap" from "a short walk is cheap", which the (B) decode rows
  conflate.

LEVEL B -- the production two-stage form, `sparsecore_topk(f32[16, 1048576])`,
  p = 16, at row_lengths 1M and 256k. Same array, same executable, same
  X_pad = 557056. This is B-D4-1M's and B-D4-256k's SC component measured
  ADJACENTLY IN ONE PROCESS; roofline.py:93 says the cross-job comparison of
  those two (823.5 vs 795.9) was worth nothing.
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

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _pick_partition, _sc_topk_direct, sparsecore_topk)

B_DIRECT, N_DIRECT = 256, 65536      # X_pad = cdiv(256,32) * 65536 = 524288
B_PROD, N_PROD = 16, 1048576         # p = 16, X_pad = 8*65536 + 32768


def main():
    repeats = int(os.environ.get("REPEATS", 5))
    iters = int(os.environ.get("ITERS", 20))

    # ---- level A --------------------------------------------------------
    print("== level A: _sc_topk_direct, one array, lengths as data ==",
          flush=True)
    full = np.full(B_DIRECT, N_DIRECT, np.int32)
    s = synth_scores(B_DIRECT, N_DIRECT, full)   # every column finite
    jax.block_until_ready(s)
    fn = jax.jit(lambda a, r: _sc_topk_direct(a, K, r, write_empty=True))

    def L(v):
        return jnp.asarray(np.full(B_DIRECT, v, np.int32))

    dead = np.full(B_DIRECT, N_DIRECT, np.int32)
    dead[1::2] = 0
    lens = [("walk-100%  (65536)", L(65536)),
            ("walk-50%   (32768)", L(32768)),
            ("walk-25%   (16384)", L(16384)),
            ("walk-12.5% (8192)", L(8192)),
            ("dead-half  (65536/0)", jnp.asarray(dead))]

    arms = []
    for name, rl in lens:
        out = fn(s, rl)
        jax.block_until_ready(out)
        o = np.asarray(out)
        nsel = [int((o[i] >= 0).sum()) for i in (0, 1)]
        walked = int(np.asarray(rl).sum())
        print(f"  {name:22} walked_words={walked:>9}  "
              f"X_pad=524288  filled rows0,1={nsel}", flush=True)
        arms.append((name, (lambda f=fn, a=s, r=rl: f(a, r))))
    rA = interleaved(arms, repeats=repeats, iters=iters,
                     ref="walk-100%  (65536)")

    del s
    # ---- level B --------------------------------------------------------
    print("\n== level B: sparsecore_topk two-stage, p=16, b=16, n=1048576 ==",
          flush=True)
    print(f"  _pick_partition(16, 1048576, 2048) = "
          f"{_pick_partition(B_PROD, N_PROD, K)}", flush=True)
    fullp = np.full(B_PROD, N_PROD, np.int32)
    s2 = synth_scores(B_PROD, N_PROD, fullp)
    jax.block_until_ready(s2)
    fn2 = jax.jit(lambda a, r: sparsecore_topk(a, K, r))
    armsB = []
    for name, v in (("p16-full    (1M live)", 1048576),
                    ("p16-quarter (256k live)", 262144),
                    ("p16-16th    (64k live)", 65536)):
        rl = jnp.asarray(np.full(B_PROD, v, np.int32))
        out = fn2(s2, rl)
        jax.block_until_ready(out)
        print(f"  {name:24} live_slices_per_row="
              f"{-(-v // 65536)}/16  filled row0="
              f"{int((np.asarray(out[0]) >= 0).sum())}/{K}", flush=True)
        armsB.append((name, (lambda f=fn2, a=s2, r=rl: f(a, r))))
    rB = interleaved(armsB, repeats=repeats, iters=iters,
                     ref="p16-full    (1M live)")
    return rA, rB


if __name__ == "__main__":
    main()
