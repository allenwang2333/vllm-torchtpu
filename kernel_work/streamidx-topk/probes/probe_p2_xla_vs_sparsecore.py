#!/usr/bin/env python3
"""P2 -- is the SparseCore top-k actually better than XLA?

Commit `6142d5e7` replaced `lax.approx_max_k(scores, k, recall_target=1.0)`
-- which is **exact**, not approximate -- with `sparsecore_topk`, for
performance.  The margin was never recorded in the commit, in a benchmark, or
anywhere in this repo (CONTRADICTIONS.md §12), and no GLM-5.2 eval baseline
exists.  So the size of the win the kernel bought is unknown.

This is the only probe in the set that can *remove* a design option: if XLA is
close, a step-3 candidate may propose deleting stage B's 457 lines, which also
dissolves the TensorCore<->SparseCore unit boundary the rest of the design is
built around.

Times all three at the nightly stage-B shape, in one process, same job:

    1. jax.lax.approx_max_k(scores, k=2048, recall_target=1.0)
    2. jax.lax.top_k(scores, k=2048)
    3. sparsecore_topk(scores, k=2048, ...)   -- as it exists on the branch

and checks that the three agree, comparing the *selected scores* sorted
descending (ties may pick different indices, so index sets are only compared
when the score vectors match exactly).

Two arms, because the op does not feed a plain dense matrix:

  `full`      -- every row length n = 10240, all-finite N(0,1) scores.  This is
                 the apples-to-apples arm: no `row_lengths` concept exists in
                 either XLA op, so this is the only shape where all three see
                 identical work.
  `prodrows`  -- the shape the op actually produces: columns at or past the
                 row's visible length (8704 at ctx 8704) are -inf, and
                 `sparsecore_topk` is given the matching `row_lengths`.  XLA
                 sees the -inf-padded matrix, which is what it would see if the
                 kernel were deleted.

Run:  python probe_p2_xla_vs_sparsecore.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

B = 64  # num_tokens at nightly decode: 64 seqs x 1 token
N = 10240  # num_positions = align_to(9,2)*1024
K = 2048  # index_topk from the checkpoint config
CTX = 8704  # the row report.md models


def build_scores(row_len: int | None, seed: int = 1):
    """f32[B, N] N(0,1), with -inf at and past `row_len` if given."""
    s = jax.random.normal(jax.random.key(seed), (B, N), jnp.float32)
    if row_len is None:
        return s, jnp.full((B, ), N, jnp.int32)
    col = jnp.arange(N, dtype=jnp.int32)
    s = jnp.where(col[None, :] < row_len, s, -jnp.inf)
    return s, jnp.full((B, ), row_len, jnp.int32)


def selected_scores(scores_np, idx_np):
    """The k selected values per row, sorted descending, -inf for idx < 0."""
    out = np.full(idx_np.shape, -np.inf, np.float32)
    for r in range(idx_np.shape[0]):
        ok = idx_np[r] >= 0
        out[r, :ok.sum()] = scores_np[r, idx_np[r][ok]]
    out.sort(axis=1)
    return out[:, ::-1]


def main():
    tpu_banner()
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    p("sparsecore_topk_module", scmod.__file__)
    p("pick_partition", f"_pick_partition({B},{N},{K}) ="
      f" {scmod._pick_partition(B, N, K)}")

    results = {"shape": dict(b=B, n=N, k=K, dtype="f32", ctx=CTX), "arms": {}}

    for arm, row_len in (("full", None), ("prodrows", CTX)):
        p("arm", arm, f"row_len={row_len if row_len else N}")
        scores, row_lengths = build_scores(row_len)
        scores = jax.block_until_ready(scores)
        scores_np = np.asarray(scores)
        arm_out = {"row_len": row_len or N, "timings": {}, "outputs": {}}

        fns = {
            "approx_max_k_rt1.0":
            jax.jit(lambda s: lax.approx_max_k(
                s, k=K, recall_target=1.0, reduction_dimension=-1)),
            "lax_top_k":
            jax.jit(lambda s: lax.top_k(s, K)),
            "sparsecore_topk":
            jax.jit(lambda s, rl: scmod.sparsecore_topk(
                s, K, row_lengths=rl, write_empty_rows=True)),
        }
        args = {
            "approx_max_k_rt1.0": (scores, ),
            "lax_top_k": (scores, ),
            "sparsecore_topk": (scores, row_lengths),
        }

        idxs = {}
        for name, fn in fns.items():
            try:
                t = time_arm(fn, args[name], warmup=3, trials=7,
                             target_s=0.05)
                out = jax.block_until_ready(fn(*args[name]))
                idx = np.asarray(out if name == "sparsecore_topk" else out[1])
                idxs[name] = idx
                arm_out["timings"][name] = {
                    "best_us": t["best_s"] * 1e6,
                    "median_us": t["median_s"] * 1e6,
                    "iters": t["iters"],
                    "compile_s": t["compile_s"],
                    "trials_us": [x * 1e6 for x in t["trials_s"]],
                }
                p(f"{arm}/{name}",
                  f"best={t['best_s']*1e6:.1f}us "
                  f"median={t['median_s']*1e6:.1f}us "
                  f"iters={t['iters']} compile={t['compile_s']:.1f}s")
            except Exception as e:
                arm_out["timings"][name] = {"error": repr(e)}
                p(f"{arm}/{name}", "FAILED", repr(e))
                traceback.print_exc()

        # ---- agreement -------------------------------------------------
        ref_name = "lax_top_k"
        if ref_name in idxs:
            ref_scores = selected_scores(scores_np, idxs[ref_name])
            for name, idx in idxs.items():
                got = selected_scores(scores_np, idx)
                finite = np.isfinite(ref_scores) & np.isfinite(got)
                d = np.abs(got - ref_scores)
                max_disc = float(d[finite].max()) if finite.any() else 0.0
                n_bad = int((d[finite] > 0).sum())
                same_set = all(
                    set(idx[r][idx[r] >= 0].tolist()) == set(
                        idxs[ref_name][r][idxs[ref_name][r] >= 0].tolist())
                    for r in range(B))
                n_sel = int((idx >= 0).sum())
                arm_out["outputs"][name] = {
                    "vs": ref_name,
                    "max_score_discrepancy": max_disc,
                    "n_positions_differing": n_bad,
                    "index_sets_identical": bool(same_set),
                    "n_selected_total": n_sel,
                    "n_expected_total": B * K,
                }
                p(f"{arm}/{name} agreement",
                  f"max|dscore|={max_disc:g} differing={n_bad} "
                  f"index_sets_identical={same_set} selected={n_sel}/{B*K}")

        # margins
        tm = arm_out["timings"]
        if all("best_us" in tm.get(x, {})
               for x in ("sparsecore_topk", "lax_top_k")):
            arm_out["speedup_sc_vs_lax_top_k"] = (tm["lax_top_k"]["best_us"] /
                                                  tm["sparsecore_topk"]
                                                  ["best_us"])
        if all("best_us" in tm.get(x, {})
               for x in ("sparsecore_topk", "approx_max_k_rt1.0")):
            arm_out["speedup_sc_vs_approx_max_k"] = (
                tm["approx_max_k_rt1.0"]["best_us"] /
                tm["sparsecore_topk"]["best_us"])
        for kk in ("speedup_sc_vs_lax_top_k", "speedup_sc_vs_approx_max_k"):
            if kk in arm_out:
                p(f"{arm}/{kk}", f"{arm_out[kk]:.2f}x")

        results["arms"][arm] = arm_out

    emit_json("P2-xla-vs-sparsecore", results)


if __name__ == "__main__":
    main()
