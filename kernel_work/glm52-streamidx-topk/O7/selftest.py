"""Prove the arm swap took, and that each config's kernel really ran.

The correctness gate reported bit-identical deficit and churn for all eight
arms. For an exact top-k selector that is the expected result -- every arm
returns the same set, and the two metrics are functions of the set. But it is
also exactly what a swap that silently did not happen would look like, and in
that case none of these kernels would ever have been compiled by Mosaic at all.

Two checks settle it, neither of which touches instruments/:

  direct  every config is called on its own and compared against base._fwd's
          exact top-k. This cannot pass unless the config's kernel lowered and
          ran, because nothing else produces the answer.
  swap    the full op is run per arm and each arm's indices are compared to the
          incumbent's POSITION BY POSITION. Two selectors that agree as a set
          will still emit in different orders once the traversal changes, so a
          nonzero positional difference is positive evidence that a different
          executable ran. (Zero is not proof of failure -- `base` is meant to be
          the incumbent algorithm -- so the direct check is the load-bearing
          one.)
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arms  # noqa: E402
import api  # noqa: E402
from base import _fwd  # noqa: E402
from run_correctness import load_instrument  # noqa: E402


def direct():
    """Every config against an exact reference, selector only."""
    print("=== direct: api.topk_indices vs base._fwd ===", flush=True)
    bad = 0
    for b, n, k, ragged in ((4, 8192, 2048, False), (4, 8192, 2048, True),
                            (2, 4096, 2048, False), (8, 16384, 2048, True)):
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        # Right-skewed like the real score distribution, so the radix descent
        # has to actually descend rather than terminating at level 0.
        x = jnp.maximum(jax.random.normal(k1, (b, n), jnp.float32), 0.0) * 3.0
        x = x + jax.random.normal(k2, (b, n), jnp.float32) * 0.01
        lens = jnp.full((b, ), n, jnp.int32)
        if ragged:
            lens = jnp.asarray([n, n // 2, 2048 + 16, 0] * (b // 4 + 1),
                               jnp.int32)[:b]
        want = np.asarray(_fwd(x, k, lens))
        xn = np.asarray(x)
        for name, cfg in arms.CONFIGS.items():
            if cfg is None:
                got = np.asarray(arms.incumbent_selector(x, k, lens))
            else:
                got = np.asarray(api.topk_indices(x, k, lens, cfg=cfg))
            ok = True
            for r in range(b):
                g = got[r][got[r] >= 0]
                w = want[r][want[r] >= 0]
                if len(g) != len(w) or len(set(g.tolist())) != len(g):
                    ok = False
                    break
                # Compare by SCORE, not index: equal boundary scores may be
                # broken differently and that is inside the documented contract.
                if not np.array_equal(np.sort(xn[r][g]), np.sort(xn[r][w])):
                    ok = False
                    break
            print(f"  {'ok  ' if ok else 'FAIL'} b={b} n={n} ragged={ragged!s:5} "
                  f"arm={name}", flush=True)
            bad += not ok
    return bad


def swap():
    """Positional agreement between each full-op arm and the incumbent."""
    print("\n=== swap: full op, indices position by position vs inc ===",
          flush=True)
    corr = load_instrument("correctness")
    case_args, _ = corr.build_case(corr.CASES["dec-2x8k"])
    kwargs = dict(k=corr.K, compression_ratio=corr.CR,
                  num_kv_pages_per_block=corr.BKV_P,
                  num_queries_per_block=corr.BQ_SZ,
                  vmem_limit_bytes=corr.VMEM_LIMIT,
                  decode_req_batch_size=corr.DECODE_REQ_BATCH_SIZE,
                  enable_early_exit=True)
    ref = None
    for name in arms.CONFIGS:
        fn = arms.full_op(name, kwargs, args=case_args)
        idx = np.asarray(fn(*case_args))
        if ref is None:
            ref = idx
            print(f"  {name:6} reference", flush=True)
            continue
        diff = int((idx != ref).sum())
        same_set = all(
            set(idx[r][idx[r] >= 0].tolist()) == set(ref[r][ref[r] >= 0].tolist())
            for r in range(idx.shape[0]))
        print(f"  {name:6} positional differences vs inc: {diff:6}   "
              f"same index set: {same_set}", flush=True)
    return 0


if __name__ == "__main__":
    bad = direct()
    swap()
    print(f"\nSELFTEST: {'PASS' if not bad else 'FAIL'}  ({bad} failures)",
          flush=True)
    sys.exit(1 if bad else 0)
