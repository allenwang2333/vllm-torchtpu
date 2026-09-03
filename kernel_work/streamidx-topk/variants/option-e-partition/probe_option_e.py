#!/usr/bin/env python3
"""Option E -- is `p = 1` valid at small `b`, and where does two-stage pay?

Two questions, in order, and the first gates the second.

**1. Correctness.**  `probes/stage-b-ksweep.md` Sect 5 forced `p=1` at `b=8` to
get a timing number and never checked the output.  `--arm correctness` closes
that: for every feasible `p` in the incumbent's own chain, at every `b` whose
behaviour Option E changes, it compares

  * the index **set** against a numpy exact-top-k oracle, and
  * the index set against the **incumbent's** own output at its chosen `p`.

Exact set equality is the bar (the radix select is exact), so the score
distribution is drawn continuous -- ties are the one documented licence to
differ (`sparsecore_topk.py:25-28`) and a tied arm is run separately, judged on
the selected *score* multiset instead.

**2. Timing.**  Chain-slope device time, the method
`probes/stage-b-ksweep.md` Sect 0 established: R calls inside one `jax.jit`,
dependency-linked through `rl + min(out[0,0], 0)` (which `x*0` would not
survive -- XLA folds that), slope over R.  The slope is marginal device time
per call with the host term differenced out; the intercept is the host term.
A batched wall clock is `max(device, host)` and the dispatch floor is a clamp,
not an addend -- nothing here subtracts one.

Every timing arm re-measures two **anchors** from the k-sweep probe --
`b=8,p=4` (50.09 us there) and `b=64,p=1` (29.00 us) -- so contention on a
fanned-out pod is visible in the artifact rather than assumed away.

Run:  python probe_option_e.py --arm {correctness,ties,timing_prod,timing_k,
                                      timing_n,glue}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "probes"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402
from partition_v2 import (  # noqa: E402
    feasible_partitions, modelled_us, pick_partition, sparsecore_topk_forced,
)

N_PROD = 10240  # num_positions at the nightly GLM-5.2 decode shape
CTX_PROD = 8704  # the mid-decode row report.md models
K_PROD = 2048

CHAIN_R = (1, 2, 3, 4)
TRIALS = 9
WARMUP = 3
TARGET_S = 0.05

# The two points every timing shard re-measures, from stage-b-ksweep.md Sect 5.
ANCHORS = ((8, 4, 50.09), (64, 1, 29.00))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_scores(b, n, row_lengths, seed=1, distinct=None):
    """f32[b, n], -inf at and past each row's length.

    `distinct=None` draws N(0,1), which is tie-free with probability 1 in
    fp32 at these sizes.  An integer `distinct` quantises to that many levels,
    which manufactures ties on purpose.
    """
    s = jax.random.normal(jax.random.key(seed), (b, n), jnp.float32)
    if distinct is not None:
        s = jnp.round(s * (distinct / 8.0)) / (distinct / 8.0)
    col = jnp.arange(n, dtype=jnp.int32)
    s = jnp.where(col[None, :] < row_lengths[:, None], s, -jnp.inf)
    return jax.block_until_ready(s)


def row_length_patterns(b, n, k, seed=0):
    """The shapes of `row_lengths` the op actually sees, plus the corners."""
    rng = np.random.default_rng(seed)
    pats = {
        "full": np.full(b, n, np.int32),
        "prod_decode": np.full(b, min(CTX_PROD, n), np.int32),
        "ragged": rng.integers(0, n + 1, size=b).astype(np.int32),
        "short_lt_k": np.full(b, max(LANES_ := 16, k // 2), np.int32),
        "some_zero": np.where(np.arange(b) % 3 == 0, 0,
                              min(CTX_PROD, n)).astype(np.int32),
    }
    del LANES_
    if b > 1:
        # The prefill ramp: `_effective_row_lengths` for one chunked-prefill
        # call, reproduced from instruments/bench_streamidx_topk.py.
        ctx = min(9216, n)
        i = np.arange(b)
        pats["prefill_ramp"] = np.minimum(np.minimum(ctx, ctx - b + i + 1),
                                          n).clip(0).astype(np.int32)
    return {name: np.clip(v, 0, n).astype(np.int32) for name, v in pats.items()}


def oracle_sets(scores_np, row_lengths_np, k):
    """Exact top-k index sets, and the selected score multiset, per row."""
    b, n = scores_np.shape
    sets, score_bags, counts = [], [], []
    for r in range(b):
        rl = int(row_lengths_np[r])
        row = scores_np[r, :rl]
        finite = np.isfinite(row)
        idx = np.nonzero(finite)[0]
        if idx.size == 0:
            sets.append(frozenset())
            score_bags.append(tuple())
            counts.append(0)
            continue
        vals = row[idx]
        take = min(k, idx.size)
        order = np.argsort(-vals, kind="stable")[:take]
        chosen = idx[order]
        sets.append(frozenset(int(x) for x in chosen))
        score_bags.append(tuple(sorted(float(v) for v in vals[order])))
        counts.append(take)
    return sets, score_bags, counts


def out_sets(out_np, scores_np):
    """Index sets, score multisets, and the emitted count per kernel row.

    `n_emitted` is kept next to `len(set)` on purpose: a duplicated index
    would collapse silently inside a set and read as a short row rather than
    as a wrong one.
    """
    sets, bags, emitted = [], [], []
    for r in range(out_np.shape[0]):
        idx = out_np[r]
        idx = idx[idx >= 0]
        emitted.append(int(idx.size))
        sets.append(frozenset(int(x) for x in idx))
        bags.append(tuple(sorted(float(scores_np[r, int(x)]) for x in idx)))
    return sets, bags, emitted


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def make_chain(call, k, r):
    def f(scores, rl):
        cur = rl
        out = None
        for _ in range(r):
            out = call(scores, k, cur)
            cur = rl + jnp.minimum(out[0, 0], jnp.int32(0))
        return out

    return jax.jit(f)


def chain_slope(call, k, scores, rl, chain_r=CHAIN_R):
    pts = {}
    for r in chain_r:
        fn = make_chain(call, k, r)
        t = time_arm(fn, (scores, rl), WARMUP, TRIALS, TARGET_S)
        pts[r] = {"best_us": t["best_s"] * 1e6, "iters": t["iters"]}
    xs = np.array(chain_r, float)
    ys = np.array([pts[r]["best_us"] for r in chain_r], float)
    a = np.vstack([np.ones_like(xs), xs]).T
    (intercept, slope), *_ = np.linalg.lstsq(a, ys, rcond=None)
    pred = intercept + slope * xs
    ss_res = float(((ys - pred)**2).sum())
    ss_tot = float(((ys - ys.mean())**2).sum())
    return {
        "points": pts,
        "intercept_us": float(intercept),
        "slope_us_per_call": float(slope),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def time_point(label, b, n, k, pp, scores, rl):
    row = {"label": label, "b": b, "n": n, "k": k, "p": pp}
    try:
        call = lambda s, kk, r: sparsecore_topk_forced(  # noqa: E731
            s, kk, r, p=pp, write_empty_rows=True)
        t0 = time.perf_counter()
        c = chain_slope(call, k, scores, rl)
        row["chain"] = c
        row["device_us"] = c["slope_us_per_call"]
        row["modelled_us"] = modelled_us(b, n, k, pp)
        row["wall_s"] = time.perf_counter() - t0
        p(label,
          f"b={b} n={n} k={k} p={pp} device={row['device_us']:.2f}us "
          f"(r2={c['r2']:.4f}, host={c['intercept_us']:.1f}us) "
          f"modelled={row['modelled_us']:.2f}us")
    except Exception as e:
        row["error"] = repr(e)
        p(label, "FAILED", repr(e))
        traceback.print_exc()
    return row


def anchor_rows():
    out = []
    for b, pp, expect in ANCHORS:
        rl = jnp.full((b, ), CTX_PROD, jnp.int32)
        sc = make_scores(b, N_PROD, rl)
        r = time_point(f"anchor_b{b}_p{pp}", b, N_PROD, K_PROD, pp, sc, rl)
        r["ksweep_reference_us"] = expect
        if "device_us" in r:
            r["vs_reference_pct"] = 100.0 * (r["device_us"] - expect) / expect
            p("anchor", f"b={b} p={pp}: {r['device_us']:.2f}us vs "
              f"{expect:.2f}us reference -> {r['vs_reference_pct']:+.1f}%")
        del sc
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def arm_correctness(res, tied=False):
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod

    bs = (1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 32, 64)
    ks = (2048, 512, 64)
    rows = []
    n = N_PROD
    for b in bs:
        for k in ks:
            inc_p = scmod._pick_partition(b, n, k)
            new_p = pick_partition(b, n, k)
            chain = feasible_partitions(b, n, k)
            pats = row_length_patterns(b, n, k, seed=b * 31 + k)
            for pname, rl_np in pats.items():
                for seed in (1, 2):
                    rl = jnp.asarray(rl_np)
                    sc = make_scores(b, n, rl, seed=seed,
                                     distinct=7 if tied else None)
                    sc_np = np.asarray(jax.device_get(sc))
                    ref_sets, ref_bags, ref_counts = oracle_sets(
                        sc_np, rl_np, k)
                    row = {
                        "b": b, "n": n, "k": k, "pattern": pname,
                        "seed": seed, "incumbent_p": inc_p,
                        "option_e_p": new_p, "feasible_p": chain,
                        "expected_counts": ref_counts, "arms": {},
                    }
                    inc_out = np.asarray(
                        jax.device_get(
                            scmod.sparsecore_topk(sc, k, row_lengths=rl,
                                                  write_empty_rows=True)))
                    inc_sets, inc_bags, inc_em = out_sets(inc_out, sc_np)
                    for pp in chain:
                        o = np.asarray(
                            jax.device_get(
                                sparsecore_topk_forced(
                                    sc, k, rl, p=pp, write_empty_rows=True)))
                        s, bags, em = out_sets(o, sc_np)
                        row["arms"][str(pp)] = {
                            "set_eq_oracle": bool(s == ref_sets),
                            "set_eq_incumbent": bool(s == inc_sets),
                            "bag_eq_oracle": bool(bags == ref_bags),
                            "counts": [len(x) for x in s],
                            "n_emitted": em,
                            "has_duplicates": bool(
                                any(len(s[i]) != em[i] for i in range(b))),
                            "count_ok": bool(
                                [len(x) for x in s] == ref_counts),
                            "bad_rows": [
                                i for i in range(b)
                                if s[i] != ref_sets[i]
                            ][:4],
                        }
                    row["incumbent_set_eq_oracle"] = bool(inc_sets == ref_sets)
                    row["incumbent_bag_eq_oracle"] = bool(inc_bags == ref_bags)
                    row["incumbent_n_emitted"] = inc_em
                    rows.append(row)
                    del sc
            bad = [
                r for r in rows[-len(pats) * 2:]
                if not all(a["set_eq_oracle"] for a in r["arms"].values())
                or not r["incumbent_set_eq_oracle"]
            ]
            p("correctness",
              f"b={b} k={k} inc_p={inc_p} newE_p={new_p} chain={chain} "
              f"-> {'OK' if not bad else f'MISMATCH x{len(bad)}'}")
    res["arms"]["ties" if tied else "correctness"] = rows
    n_bad = sum(1 for r in rows
                if not all(a["set_eq_oracle"] for a in r["arms"].values()))
    n_bad_bag = sum(1 for r in rows
                    if not all(a["bag_eq_oracle"] for a in r["arms"].values()))
    res["summary" + ("_ties" if tied else "")] = {
        "cases": len(rows), "index_set_mismatches": n_bad,
        "score_multiset_mismatches": n_bad_bag,
    }
    p("SUMMARY", f"{'ties ' if tied else ''}cases={len(rows)} "
      f"index_set_mismatches={n_bad} score_bag_mismatches={n_bad_bag}")


def arm_timing_prod(res):
    res["arms"]["anchors"] = anchor_rows()
    rows = []
    for b in (1, 2, 4, 8, 16, 32, 64):
        rl = jnp.full((b, ), CTX_PROD, jnp.int32)
        sc = make_scores(b, N_PROD, rl)
        ps = [q for q in (1, 2, 4, 8, 16, 32)
              if b * q <= 32 and N_PROD % q == 0 and N_PROD // q >= 16]
        for pp in ps:
            r = time_point(f"prod_b{b}_p{pp}", b, N_PROD, K_PROD, pp, sc, rl)
            r["eff_len"] = CTX_PROD
            rows.append(r)
        del sc
    res["arms"]["timing_prod"] = rows


def arm_timing_k(res):
    res["arms"]["anchors"] = anchor_rows()
    rows = []
    for b in (8, 16):
        rl = jnp.full((b, ), CTX_PROD, jnp.int32)
        sc = make_scores(b, N_PROD, rl)
        for k in (2048, 1024, 512, 256, 128, 64):
            ps = [q for q in (1, 2, 4) if b * q <= 32]
            for pp in ps:
                rows.append(
                    time_point(f"k_b{b}_k{k}_p{pp}", b, N_PROD, k, pp, sc, rl))
        del sc
    res["arms"]["timing_k"] = rows


def arm_timing_n(res):
    res["arms"]["anchors"] = anchor_rows()
    rows = []
    # n = 65536: the last width where p=1 still fits the 64K-word slice.
    n = 65536
    for b, ks in ((1, (2048, 128)), ):
        rl = jnp.full((b, ), n, jnp.int32)
        sc = make_scores(b, n, rl)
        for k in ks:
            for pp in (1, 2, 4, 8, 16, 32):
                rows.append(time_point(f"n{n}_b{b}_k{k}_p{pp}", b, n, k, pp,
                                       sc, rl))
        del sc
    res["arms"]["timing_n65536"] = rows

    # n = 262144: the AgentX deployment width.  p >= 4 is mandatory here --
    # p=1..2 cannot be built, which is the capacity clause doing its job.
    rows2 = []
    n = 262144
    for b in (1, 16):
        rl = jnp.full((b, ), n, jnp.int32)
        sc = make_scores(b, n, rl)
        for pp in (4, 8, 16, 32):
            rows2.append(time_point(f"n{n}_b{b}_k2048_p{pp}", b, n, 2048, pp,
                                    sc, rl))
        del sc
    res["arms"]["timing_n262144"] = rows2


def arm_glue(res):
    """Price the two `take_along_axis` gathers the two-stage path adds.

    Measures the whole two-stage call and, separately, each of its two
    `_sc_topk_direct` launches at exactly the shapes it feeds them.  The
    residual is the XLA glue -- it is a *difference of two chain slopes*, so it
    carries both their errors and is reported as such.
    """
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod

    res["arms"]["anchors"] = anchor_rows()
    rows = []
    for b, pp, k in ((8, 4, 2048), (8, 2, 2048), (16, 2, 2048), (4, 8, 1024),
                     (2, 16, 512), (8, 4, 512), (1, 32, 256)):
        n = N_PROD
        n_p = n // pp
        k_p = min(k, n_p)
        rl = jnp.full((b, ), CTX_PROD, jnp.int32)
        sc = make_scores(b, n, rl)
        row = {"b": b, "n": n, "k": k, "p": pp, "n_p": n_p, "k_p": k_p,
               "cand": pp * k_p, "gather_elems": b * pp * k_p}
        try:
            whole = time_point(f"glue_whole_b{b}_p{pp}_k{k}", b, n, k, pp, sc,
                               rl)
            # stage 1: b*p rows of n_p, lengths clipped the same way
            offs = jnp.arange(pp, dtype=jnp.int32) * n_p
            rl1 = jnp.clip(rl[:, None] - offs[None, :], 0, n_p).reshape(b * pp)
            sc1 = sc.reshape(b * pp, n_p)
            c1 = lambda s, kk, r: scmod._sc_topk_direct(  # noqa: E731
                s, kk, r, write_empty=True)
            s1 = chain_slope(c1, k_p, sc1, rl1)
            # stage 2: b rows of p*k_p candidates
            rl2 = jnp.full((b, ), pp * k_p, jnp.int32)
            sc2 = make_scores(b, pp * k_p, rl2, seed=3)
            s2 = chain_slope(c1, k, sc2, rl2)
            row["whole_us"] = whole.get("device_us")
            row["stage1_us"] = s1["slope_us_per_call"]
            row["stage2_us"] = s2["slope_us_per_call"]
            row["stage1_r2"] = s1["r2"]
            row["stage2_r2"] = s2["r2"]
            if row["whole_us"] is not None:
                row["glue_us"] = (row["whole_us"] - row["stage1_us"] -
                                  row["stage2_us"])
            w = row["whole_us"]
            g = row.get("glue_us")
            p("glue",
              f"b={b} p={pp} k={k}: "
              f"whole={'n/a' if w is None else f'{w:.2f}'} "
              f"s1={row['stage1_us']:.2f} s2={row['stage2_us']:.2f} "
              f"-> glue={'n/a' if g is None else f'{g:.2f}'}us "
              f"(gather elems {row['gather_elems']})")
            del sc1, sc2
        except Exception as e:
            row["error"] = repr(e)
            p("glue", "FAILED", repr(e))
            traceback.print_exc()
        rows.append(row)
        del sc
    res["arms"]["glue"] = rows


ARMS = {
    "correctness": lambda r: arm_correctness(r, tied=False),
    "ties": lambda r: arm_correctness(r, tied=True),
    "timing_prod": arm_timing_prod,
    "timing_k": arm_timing_k,
    "timing_n": arm_timing_n,
    "glue": arm_glue,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()

    tpu_banner()
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    p("sparsecore_topk_module", scmod.__file__)
    p("partition_v2_module", os.path.join(_HERE, "partition_v2.py"))
    p("arm", args.arm)

    res = {
        "arm": args.arm,
        "methodology": {
            "trials": TRIALS, "warmup": WARMUP, "target_s": TARGET_S,
            "chain_r": list(CHAIN_R),
            "device_time": ("chain slope: R calls in one jit, dependency "
                            "linked; slope = device us/call, intercept = host"),
            "note": ("batched per-call = max(device, host enqueue); the "
                     "dispatch floor is a clamp, not an addend, and nothing "
                     "here subtracts one"),
        },
        "arms": {},
    }
    t0 = time.perf_counter()
    ARMS[args.arm](res)
    res["arm_wall_s"] = time.perf_counter() - t0
    emit_json(f"option-e-{args.arm}", res)


if __name__ == "__main__":
    main()
