"""Off-device trace check: `jax.eval_shape` every config against tpu7x info.

Catches shape errors, missing primitives and illegal ref slices in about a
second on CPU. It does NOT check lowering -- Mosaic never runs -- so a clean
run here means only that the pod job will get as far as the compiler.

    JAX_PLATFORMS=cpu python3 trace_check.py
"""
import os
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X
pltpu.get_tpu_info = lambda: _TPU7X

from pallas_mosaic_tpu import (Config, _candidate_cap,  # noqa: E402
                               _get_autotuning_configs, sc_topk_direct)
import api  # noqa: E402

api.sc_topk_direct = sc_topk_direct

# (b, n, k) triples spanning the four measured shapes' kernel geometries.
#   (A) direct: b=T rows of n=10240, p=1
#   (B) two-stage stage 1: b=T*16 rows of n=65536, and stage 2 b=T rows of 32768
GEOMS = [(16, 10240, 2048), (64, 10240, 2048), (16, 1048576, 2048),
         (16, 262144, 2048), (2, 8192 + 1024, 2048)]


def main():
    bad = 0
    for cfg in _get_autotuning_configs():
        for b, n, k in GEOMS:
            scores = jax.ShapeDtypeStruct((b, n), jnp.float32)
            lens = jax.ShapeDtypeStruct((b, ), jnp.int32)
            try:
                out = jax.eval_shape(
                    lambda s, r, cfg=cfg, k=k: api.topk_indices(
                        s, k, r, cfg=cfg), scores, lens)
                assert out.shape == (b, k), out.shape
                print(f"  ok   {cfg.tag:52} b={b:5} n={n:8} k={k}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"  FAIL {cfg.tag:52} b={b:5} n={n:8} k={k}\n"
                      f"       {type(e).__name__}: {str(e)[:300]}")
    # Report the candidate-buffer sizing, since it decides whether
    # carry-forward engages at all at (B).
    sc = _TPU7X.sparse_core
    print("\ncandidate cap by slice_len (carry_forward on, 8-bit digits):")
    for slice_len in (10240, 16384, 32768, 65536):
        for scan in (False, True):
            cfg = Config(carry_forward=True, scan_count_hist=scan)
            cap = _candidate_cap(cfg, sc, slice_len, 2048, 256, 16)
            print(f"  slice_len={slice_len:6} scan_count={scan!s:5} "
                  f"cap={cap:6} ({cap / slice_len:.0%} of the slice)")
    print("\nFAILURES:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
