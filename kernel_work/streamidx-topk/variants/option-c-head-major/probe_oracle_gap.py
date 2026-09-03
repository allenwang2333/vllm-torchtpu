"""Option C follow-up 2: the f64 oracle disagrees with the SHIPPING kernel.

WHAT THIS IS ABOUT
------------------
`run_correctness_c.py --mode recall` reported, at `T=8 S_valid=2048 k=512`:

    incumbent  vs f64 oracle: min_recall=0.996094 missed=10 worst_gap=4.738e-01
    head_minor vs f64 oracle: min_recall=0.996094 missed=10 worst_gap=4.738e-01
    head_major vs f64 oracle: min_recall=0.996094 missed=10 worst_gap=4.738e-01

Byte-identical across the three arms, so it is NOT Option C.  It is either
(a) a benign near-tie -- the oracle and the kernel picked different members of
a pair whose scores differ by f32 rounding, which no consumer can distinguish;
or (b) a real selection defect in code that ships today.  Recall alone cannot
tell those apart.  The GAP can, so this probe reports the gaps.

HOW IT DISCRIMINATES
--------------------
The oracle is not the suspect and neither is quantisation: `_build_g2` reads
the fp8 `q` and the fp8xe8m0 `K` back out and scores THOSE in float64, so the
oracle already sees exactly the numbers the MXU sees.  What is left between
oracle and answer is three things, and this probe separates all three:

  1. STAGE A -- the f32 score.  Captured directly, not re-derived: this patches
     `streamindex_topk.sparsecore_topk` with a spy and calls the shipping
     `streamindex_topk.__wrapped__` inside one trace, so what is compared is
     the exact array the production kernel hands to stage B.  No transcription
     of the launcher is involved and none can drift.
  2. STAGE B -- the selection.  `numpy.argsort` of the kernel's OWN captured
     f32 scores, compared to the indices the kernel returned.  If those two
     agree, stage B selected the true top-k of what it was given, and probe P2's
     finding (`sparsecore_topk` bit-identical to `approx_max_k(1.0)`) is
     confirmed against a different reference.
  3. ROW LENGTHS -- `_effective_row_lengths`.  Stage B only searches a prefix
     of each row; if that prefix is shorter than the oracle's visible range,
     items get dropped for a reason that has nothing to do with arithmetic.
     The spy captures `row_lengths` and it is compared against the oracle mask.

VERDICT RULE, fixed before the numbers are seen
-----------------------------------------------
Let `gap = s_oracle[missed] - s_oracle[displacer]`, the f64 score difference
between an item the oracle ranks in the top-k and the item the kernel took in
its place, and let `sigma` be the f32 accumulation scale, taken as the measured
`max|s_kernel - s_oracle|` over all valid positions.

  * every gap <= a few sigma  ->  BENIGN near-tie.
  * any gap >> sigma          ->  REAL DEFECT: the kernel dropped an item it
                                  had no numerical excuse to drop.  Say so
                                  loudly.

Nothing here is a timing.  Safe to fan out.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROBES = _HERE.parent.parent / "probes"
for _p in (str(_HERE), str(_PROBES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import run_correctness_c as RC  # noqa: E402

P = "[oracle-gap]"


def _p(*a):
    print(P, *a, flush=True)


def capture(cfg, b, arm="incumbent"):
    """Run the kernel and capture what stage B was actually given.

    Returns (indices, scores_f32, row_lengths).  `enable_early_exit` is left at
    its default False, so `_common_path` is called directly rather than through
    `lax.cond` -- which is what makes the captured tracer legal to return, and
    is not a behaviour change here anyway (`max(seq_lens)//cr = 2048 > k`, so
    the fast path could not have fired).
    """
    import functools

    import jax
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod
    import streamindex_topk_c as sxc

    if arm == "incumbent":
        target = sxmod.streamindex_topk.__wrapped__
        extra = {}
    else:
        target = sxc.streamindex_topk.__wrapped__
        extra = {"q_layout": arm}

    real = sxmod.sparsecore_topk
    cap = {}

    def spy(scores, k, *, row_lengths=None, **kw):
        cap["scores"] = scores
        cap["row_lengths"] = row_lengths
        return real(scores, k, row_lengths=row_lengths, **kw)

    static = dict(k=cfg["k"],
                  compression_ratio=cfg["comp_ratio"],
                  num_kv_pages_per_block=cfg["bkv_p"],
                  num_queries_per_block=cfg["bq_sz"],
                  **extra)

    sxmod.sparsecore_topk = spy
    # the variant module imported its own reference to the same symbol
    had_sxc = hasattr(sxc, "sparsecore_topk")
    real_sxc = getattr(sxc, "sparsecore_topk", None)
    if had_sxc:
        sxc.sparsecore_topk = spy
    try:

        @functools.partial(jax.jit, static_argnames=tuple(static))
        def f(q, w, cache_kv, seq_lens, page_indices, cu_q_lens, distribution,
              **st):
            idx = target(q, w, cache_kv, seq_lens, page_indices, cu_q_lens,
                         distribution, **st)
            if "scores" not in cap:
                raise RuntimeError(
                    "the sparsecore_topk spy never fired -- the kernel took a "
                    "path that does not call it, so nothing was captured")
            return idx, cap["scores"], cap["row_lengths"]

        idx, scores, rl = f(b["q"], b["w"], b["cache_kv"], b["seq_lens"],
                            b["page_indices"], b["cu_q_lens"],
                            b["distribution"], **static)
    finally:
        sxmod.sparsecore_topk = real
        if had_sxc:
            sxc.sparsecore_topk = real_sxc
    return (np.asarray(idx), np.asarray(scores),
            None if rl is None else np.asarray(rl))


def analyse(cfg, arm="incumbent", seed=0, dump_n=64):
    b = RC._build_g2(cfg, seed)
    T, k = cfg["T_seq"], cfg["k"]
    S_valid = b["S_valid"]

    s_or, n_masked = RC._oracle(b, cfg)  # (T, S_valid) f64, -inf where masked
    idx, s_ker_full, rl = capture(cfg, b, arm)
    _p(f"arm={arm} captured: indices {idx.shape} scores {s_ker_full.shape} "
       f"row_lengths={'None' if rl is None else rl.tolist()}")

    s_ker = s_ker_full[:T, :S_valid].astype(np.float64)
    vis = np.isfinite(s_or)
    _p(f"oracle: S_valid={S_valid} masked={n_masked} "
       f"visible per row = {vis.sum(axis=1).tolist()}")
    if rl is not None:
        agree = [int(vis[t].sum()) == int(rl[t]) for t in range(T)]
        _p(f"_effective_row_lengths vs oracle visible count: "
           f"{'ALL AGREE' if all(agree) else 'MISMATCH ' + str(agree)}")

    # ---- 1. STAGE A: how far is the kernel's f32 score from exact? ---------
    both = vis & np.isfinite(s_ker)
    d = np.abs(s_ker - s_or)[both]
    scale = np.abs(s_or)[both]
    sigma = float(d.max())
    stage_a = dict(
        n_compared=int(both.sum()),
        max_abs=sigma,
        p99_abs=float(np.percentile(d, 99)),
        median_abs=float(np.median(d)),
        max_rel=float((d / np.maximum(scale, 1e-30)).max()),
        score_absmax=float(scale.max()),
        mask_agrees=bool(np.array_equal(vis, np.isfinite(s_ker))),
    )
    _p(f"STAGE A  |s_f32 - s_f64|: max={stage_a['max_abs']:.6e} "
       f"p99={stage_a['p99_abs']:.6e} median={stage_a['median_abs']:.6e} "
       f"max_rel={stage_a['max_rel']:.3e} over {stage_a['n_compared']} "
       f"positions; |score| max {stage_a['score_absmax']:.3f}; "
       f"mask_agrees={stage_a['mask_agrees']}")

    # ---- 2. STAGE B: did it select the true top-k of ITS OWN input? -------
    got = [set(int(i) for i in idx[t] if i >= 0) for t in range(T)]
    ker_order = np.argsort(-s_ker, axis=-1, kind="stable")
    stage_b_bad, stage_b_rows = 0, 0
    for t in range(T):
        n_take = min(k, int(np.isfinite(s_ker[t]).sum()))
        want = set(int(i) for i in ker_order[t, :n_take])
        diff = want ^ got[t]
        if diff:
            stage_b_rows += 1
            stage_b_bad += len(diff)
    stage_b = dict(rows_differing=stage_b_rows, indices_differing=stage_b_bad,
                   exact=(stage_b_bad == 0))
    _p(f"STAGE B  top-k of the kernel's OWN f32 scores vs returned indices: "
       f"rows_differing={stage_b_rows}/{T} indices_differing={stage_b_bad} "
       f"-> {'EXACT' if stage_b['exact'] else 'NOT EXACT'}")

    # ---- 3. the disagreements, with their actual gaps ---------------------
    order = np.argsort(-s_or, axis=-1, kind="stable")
    oracle_top = order[:, :k]
    rows = []
    for t in range(T):
        want = set(int(i) for i in oracle_top[t])
        missed = sorted(want - got[t])
        extra = sorted(got[t] - want)
        if not missed:
            continue
        # the items the kernel took that the oracle did not, ranked worst-first
        extra_sorted = sorted(extra, key=lambda i: s_or[t, i])
        for n, i in enumerate(sorted(missed, key=lambda j: -s_or[t, j])):
            # pair the n-th best missed item with the n-th worst displacer
            j = extra_sorted[n] if n < len(extra_sorted) else None
            rows.append(
                dict(token=t,
                     missed_idx=int(i),
                     oracle_rank=int(np.where(order[t] == i)[0][0]),
                     s_oracle_missed=float(s_or[t, i]),
                     s_kernel_missed=float(s_ker[t, i]),
                     displacer_idx=(None if j is None else int(j)),
                     s_oracle_displacer=(None if j is None else float(
                         s_or[t, j])),
                     s_kernel_displacer=(None if j is None else float(
                         s_ker[t, j])),
                     gap_oracle=(None if j is None else float(s_or[t, i] -
                                                              s_or[t, j])),
                     gap_kernel=(None if j is None else float(s_ker[t, i] -
                                                              s_ker[t, j]))))

    _p(f"-- {len(rows)} index(es) the oracle ranks in the top-{k} that the "
       f"kernel did not return --")
    if rows:
        print(f"{P} {'tok':>3} {'idx':>6} {'orank':>6} "
              f"{'s_f64(missed)':>15} {'s_f32(missed)':>15} "
              f"{'disp':>6} {'s_f64(disp)':>14} "
              f"{'gap_f64':>12} {'gap_f32':>12} {'gap/sigma':>10}",
              flush=True)
        for r in rows[:dump_n]:
            g64 = r["gap_oracle"]
            print(f"{P} {r['token']:3d} {r['missed_idx']:6d} "
                  f"{r['oracle_rank']:6d} {r['s_oracle_missed']:15.8f} "
                  f"{r['s_kernel_missed']:15.8f} "
                  f"{(r['displacer_idx'] if r['displacer_idx'] is not None else -1):6d} "
                  f"{(r['s_oracle_displacer'] if r['s_oracle_displacer'] is not None else float('nan')):14.8f} "
                  f"{(g64 if g64 is not None else float('nan')):12.3e} "
                  f"{(r['gap_kernel'] if r['gap_kernel'] is not None else float('nan')):12.3e} "
                  f"{((g64 / sigma) if (g64 is not None and sigma > 0) else float('nan')):10.2f}",
                  flush=True)

    gaps = [abs(r["gap_oracle"]) for r in rows if r["gap_oracle"] is not None]
    max_gap = max(gaps) if gaps else 0.0
    ratio = (max_gap / sigma) if sigma > 0 else float("inf")
    # the rule was fixed in the docstring, before the numbers existed
    if not rows:
        verdict = "NO DISAGREEMENT"
    elif ratio <= 4.0:
        verdict = "BENIGN NEAR-TIE"
    else:
        verdict = "REAL DEFECT"
    _p(f"VERDICT: {verdict}  (max |gap_f64| = {max_gap:.6e}, "
       f"sigma = max|s_f32-s_f64| = {sigma:.6e}, ratio = {ratio:.2f})")

    return dict(arm=arm, cfg=cfg, S_valid=int(S_valid), n_masked=int(n_masked),
                row_lengths=(None if rl is None else rl.tolist()),
                oracle_visible_per_row=vis.sum(axis=1).tolist(),
                stage_a=stage_a, stage_b=stage_b,
                disagreements=rows, max_gap_oracle=max_gap,
                sigma_stage_a=sigma, gap_over_sigma=ratio, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="incumbent,head_major")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = dict(RC.G2)
    _p("cfg", cfg)
    out = []
    for seed in (int(s) for s in args.seeds.split(",") if s):
        for arm in (a for a in args.arms.split(",") if a):
            _p("=" * 70)
            _p(f"seed={seed} arm={arm}")
            r = analyse(cfg, arm=arm, seed=seed)
            r["seed"] = seed
            out.append(r)

    print("\n===== ORACLE-GAP SUMMARY =====")
    print(f"{'seed':>4} {'arm':16} {'sigma(stageA)':>14} {'max|gap|':>12} "
          f"{'ratio':>8} {'stageB':>10} {'verdict':>18}")
    for r in out:
        print(f"{r['seed']:4d} {r['arm']:16} {r['sigma_stage_a']:14.6e} "
              f"{r['max_gap_oracle']:12.6e} {r['gap_over_sigma']:8.2f} "
              f"{('EXACT' if r['stage_b']['exact'] else 'NOT EXACT'):>10} "
              f"{r['verdict']:>18}")

    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        _p("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
