"""Gate O3 with the SHARED instrument, unmodified: patch the selector, run correctness.py.

  python3 run_correctness.py

The instrument is not copied and not edited -- `instruments/correctness.py` is
imported and its `main()` is called. All this file does is swap the selector
that `streamindex_topk` reaches for, then add the one check the instrument
cannot make because it only ever runs one implementation: whether O3's selected
sets are identical to the incumbent's.

The bucketing hazard is the reason `dec-ragged` gets called out separately. Its
`lens=[8193, 3000, 9216, 2049]` is the batch-max case, and its L=2049 row sits
one word above k=2048.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import tiling  # noqa: E402
from pallas_mosaic_tpu import _get_heuristics_config  # noqa: E402
from pallas_mosaic_tpu_kernel import sparsecore_topk as o3_topk  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    sparsecore_topk as incumbent_topk  # noqa: E402
import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as sit  # noqa: E402


def report_rungs():
    """What p each correctness case and each benchmarked shape lands on."""
    import correctness
    from shapes import ROWS, K, padded_n
    print("== rung selection (host-side arithmetic, same rule as the switch) ==",
          flush=True)
    rows = [(f"case:{n}", c["T"], padded_n(c["psz"], c["pps"]), max(c["lens"]))
            for n, c in correctness.CASES.items()]
    for n, r in ROWS.items():
        if n not in ("A-D64-8k", "B-D4-256k", "B-D1-1M", "B-RAG16",
                     "B-P1024-256k"):
            continue
        npad = padded_n(r["psz"], r["pages_per_seq"])
        rows.append((f"row:{n}", r["T"], npad, min(r["seqlen"], npad)))
    for name, T, npad, live in rows:
        cfg = _get_heuristics_config(T, npad, K)
        lad = cfg.ladder
        p = lad[tiling.ladder_index(live, cfg.tn, lad)]
        cover = p * cfg.tn
        ok = "OK" if cover >= live else "TRUNCATES"
        print(f"  {name:18} T={T:>5} n={npad:>8} live_max={live:>8}  "
              f"p_max={cfg.p_max:>2} -> p={p:>2}  covers {cover:>8}  {ok}",
              flush=True)
        if cover < live:
            raise SystemExit("bucketed DOWNWARD -- candidate set truncated")


def same_sets_as_incumbent():
    """Selector-level A/B: identical selected sets, O3 vs the incumbent.

    Run on the selector directly rather than through the full op, because the
    question is about the partition and nothing else. Both (B) shapes that
    change rung are covered, plus one that does not.
    """
    from bench import synth_scores
    from shapes import K
    print("\n== O3 vs incumbent selected sets (sparsecore_topk level) ==",
          flush=True)
    worst = 0
    for label, T, n, live in (("b16 n1M live256k (p 16->4)", 16, 1048576, 262144),
                              ("b16 n1M live1M   (p 16->16)", 16, 1048576, 1048576),
                              ("b16 n1M live64k  (p 16->1)", 16, 1048576, 65536),
                              ("b16 n10240 live8192 (A, p 2->2)", 16, 10240, 8192)):
        lens = np.full(T, live, np.int32)
        lens[T // 2:] = 0        # padded token rows, as the real call site has
        s = synth_scores(T, n, lens)
        rl = jnp.asarray(lens, jnp.int32)
        a = np.asarray(jax.jit(lambda x, r: incumbent_topk(x, K, r))(s, rl))
        b = np.asarray(jax.jit(lambda x, r: o3_topk(x, K, r))(s, rl))
        diff = sum(len(set(a[t][a[t] >= 0].tolist())
                       ^ set(b[t][b[t] >= 0].tolist())) for t in range(T))
        counts = all((a[t] >= 0).sum() == (b[t] >= 0).sum() for t in range(T))
        worst = max(worst, diff)
        print(f"  {label:34} symmetric difference {diff:>6} of {T * K} slots, "
              f"equal fill counts: {counts}", flush=True)
        del s, a, b
    print(f"  byte-identical selected sets overall: {worst == 0}", flush=True)
    return worst


def main():
    report_rungs()
    diff = same_sets_as_incumbent()

    # The swap. `streamindex_topk` holds `sparsecore_topk` as a module global
    # and resolves it at call time, so rebinding it here is enough and the
    # instrument imports nothing of O3's.
    sit.sparsecore_topk = o3_topk
    print("\n== instruments/correctness.py, O3 selector ==", flush=True)
    import correctness
    try:
        correctness.main()
    except SystemExit as e:
        code = int(e.code or 0)
        if diff:
            print(f"\nNOTE: selected sets differ from the incumbent "
                  f"({diff} slots)", flush=True)
        sys.exit(code)


if __name__ == "__main__":
    main()
