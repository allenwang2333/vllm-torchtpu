#!/usr/bin/env python3
"""Do Option B (chunked stage-A/stage-B pipeline) and Option C (head-major
score layout) compose at prefill, or collide?

The 2x3 grid, all at the production prefill point (1 sequence, q_len 1024,
ctx 9216):

                     head_minor          head_major
    G = 1            baseline            C alone
    G = 2            B alone             B x C
    G = 4            B alone             B x C

plus an `incumbent` control at G = 1 that launches the *unmodified* src kernel
through the same wrapper, so a drift in Option C's verbatim copy shows up as a
`head_minor` vs `incumbent` gap rather than being absorbed into the result.

Timing is `probe_flip3.py`'s chain slope, unchanged: R in {1,2,3,4} chained
copies inside ONE `jax.jit`, best-of-N per point, least squares on t(R), fits
below r^2 = 0.999 rejected.  The slope is dispatch-free device time and the
host term lands in the intercept -- a per-call wall clock is max(device, host)
and subtracting a dispatch floor is invalid, so nothing is subtracted anywhere.
`chain_slope` and `linfit` are imported from `probe_flip3.py`; they are not
re-implemented here.

Correctness (`--arm correct`) compares the top-k index *set* of every cell
against `src/.../streamindex_topk.py::streamindex_topk` run on the whole
unchunked op.  Sets, not sequences: the kernel documents its output as
unsorted.  A row that differs is re-examined against the f32 scores at the two
differing indices, because a tie broken differently is not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, os.path.join(_ROOT, "probes"),
           os.path.join(_ROOT, "variants", "option-b-pipeline"),
           os.path.join(_ROOT, "variants", "option-c-head-major")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import pipeline_c as PC  # noqa: E402
from _probe_util import emit_json, p, tpu_banner  # noqa: E402
from probe_flip3 import chain_slope  # noqa: E402

PHASE = "prefill"
GS = (1, 2, 4)
# (q_arm, G) cells.  `incumbent` only at G=1: it is a drift control on the
# copy, not a row of the grid.
CELLS = [("incumbent", 1)] + [(a, g) for a in ("head_minor", "head_major")
                              for g in GS]


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------


def _capture_premade(phase, G, q_arm, chunk_arrs, cache_kv):
    """Stage A's actual output for this q_arm, captured once outside the timed
    region and handed to arm `b`.  Same device as `probe_flip3._capture_premade`
    -- arm `b` and arm `pipe` must select over the same numbers, because stage
    B is data-dependent at the ~6 % level."""
    g = PC.geometry(phase)
    body = PC.build_body(phase, G, "a", 1, q_arm=q_arm, force_p1=True,
                         return_outputs=True)
    outs = jax.jit(body)(*PC.flat_args(chunk_arrs, cache_kv))
    jax.block_until_ready(outs)
    n = g["num_positions"]
    return [
        jax.block_until_ready(s.reshape(s.shape[0], n)) for s in outs[1:]
    ]


def run_perf(trials=9, cells=CELLS, with_chained=True):
    g, shared = PC.build_inputs(PHASE)
    p("geometry", {k: v for k, v in g.items() if k != "case"})
    cache_kv = shared["cache_kv"]
    res = {"phase": PHASE, "geometry": {k: str(v) for k, v in g.items()},
           "arms": {}, "guards": {}, "derived": {}}

    chunk_cache = {}
    for q_arm, G in cells:
        if G not in chunk_cache:
            chunk_cache[G] = PC.chunk_arrays(PHASE, G, shared)
            b = PC.chunk_plan(PHASE, G)[0]["num_tokens"]
            wg = PC.waves_guard(b)
            res["guards"][f"G{G}"] = wg
            p(f"G={G} waves_guard",
              f"b={wg['b']} b_pad={wg['b_pad']} waves={wg['num_waves']} "
              f"-> {'PASS' if wg['ok'] else 'FAIL'}")
        chunk_arrs = chunk_cache[G]

        premade = _capture_premade(PHASE, G, q_arm, chunk_arrs, cache_kv)
        args = PC.flat_args(chunk_arrs, cache_kv, premade)

        arms = ["a", "b", "pipe"]
        if with_chained and G > 1:
            arms.append("b_chained")
        for arm in arms:
            key = f"{q_arm}/{arm}_G{G}"
            try:
                with PC.p_tripwire(active=True):
                    out = chain_slope(
                        lambda R, arm=arm, G=G, qa=q_arm: PC.build_body(
                            PHASE, G, arm, R, q_arm=qa, force_p1=True),
                        args, trials=trials, tag=key)
                out["q_arm"] = q_arm
                out["G"] = G
                out["arm"] = arm
                res["arms"][key] = out
            except PC.PickPartitionCalled as e:
                res["arms"][key] = {"error": f"TRIPWIRE {e}"}
                p(key, "TRIPWIRE FIRED", repr(e))
            except Exception as e:  # noqa: BLE001
                res["arms"][key] = {"error": repr(e)}
                p(key, "FAILED", repr(e))
                traceback.print_exc()
        del premade, args

    # ---- the arithmetic ---------------------------------------------------
    def slope(key):
        a = res["arms"].get(key)
        if not a or "slope_us" not in a or not a.get("accepted"):
            return None
        return a["slope_us"]

    for q_arm, G in cells:
        ta, tb, tp = (slope(f"{q_arm}/a_G{G}"), slope(f"{q_arm}/b_G{G}"),
                      slope(f"{q_arm}/pipe_G{G}"))
        if None in (ta, tb, tp):
            continue
        hidden = ta + tb - tp
        ideal = min(ta / G, tb / G) * (G - 1)
        res["derived"][f"{q_arm}/G{G}"] = dict(
            q_arm=q_arm, G=G, t_a_us=ta, t_b_us=tb, t_pipe_us=tp,
            t_a_plus_t_b_us=ta + tb, hidden_us=hidden, ideal_saving_us=ideal,
            overlap_fraction=(hidden / ideal) if ideal > 0 else None,
            t_b_chained_us=slope(f"{q_arm}/b_chained_G{G}"))

    base = slope("head_minor/pipe_G1")
    if base is not None:
        grid = {}
        for q_arm in ("head_minor", "head_major"):
            for G in GS:
                tp = slope(f"{q_arm}/pipe_G{G}")
                if tp is None:
                    continue
                grid[f"{q_arm}/G{G}"] = dict(
                    t_pipe_us=tp, delta_us=tp - base,
                    delta_pct=100.0 * (tp - base) / base)
        res["grid_vs_baseline"] = {"baseline_key": "head_minor/pipe_G1",
                                   "baseline_us": base, "cells": grid}

        # additivity: is  BxC(G) - base  ==  (B(G) - base) + (C - base) ?
        dC = None
        if slope("head_major/pipe_G1") is not None:
            dC = slope("head_major/pipe_G1") - base
        add = {}
        for G in (2, 4):
            tb_ = slope(f"head_minor/pipe_G{G}")
            tbc = slope(f"head_major/pipe_G{G}")
            if None in (tb_, tbc) or dC is None:
                continue
            dB = tb_ - base
            dBC = tbc - base
            add[f"G{G}"] = dict(
                delta_B_us=dB, delta_C_us=dC, delta_BC_us=dBC,
                additive_prediction_us=dB + dC,
                miss_us=dBC - (dB + dC),
                miss_pct_of_baseline=100.0 * (dBC - (dB + dC)) / base,
                fraction_of_predicted_saving_realised=(
                    dBC / (dB + dC) if (dB + dC) != 0 else None))
        res["additivity"] = add

        # Option C's stage-A win as a function of G -- the num_bq mechanism.
        cwin = {}
        for G in GS:
            am, aj = slope(f"head_minor/a_G{G}"), slope(f"head_major/a_G{G}")
            if None in (am, aj):
                continue
            plan = PC.chunk_plan(PHASE, G)
            nt = plan[0]["num_tokens"]
            cwin[f"G{G}"] = dict(
                tokens_per_chunk=nt,
                num_bq_per_chunk=max(1, -(-nt // g["bq_per_block"])),
                t_a_head_minor_us=am, t_a_head_major_us=aj,
                delta_us=aj - am, delta_pct=100.0 * (aj - am) / am)
        res["stage_a_C_win_vs_G"] = cwin

    for k, v in sorted(res.get("derived", {}).items()):
        p(f"DERIVED {k}",
          f"t_a={v['t_a_us']:.2f} t_b={v['t_b_us']:.2f} "
          f"t_pipe={v['t_pipe_us']:.2f} hidden={v['hidden_us']:.2f} "
          f"overlap={v['overlap_fraction']}")
    for k, v in sorted(res.get("grid_vs_baseline", {}).get("cells",
                                                           {}).items()):
        p(f"GRID {k}",
          f"t_pipe={v['t_pipe_us']:.2f}us delta={v['delta_us']:+.2f}us "
          f"({v['delta_pct']:+.2f}%)")
    for k, v in sorted(res.get("additivity", {}).items()):
        p(f"ADDITIVITY {k}",
          f"dB={v['delta_B_us']:+.2f} dC={v['delta_C_us']:+.2f} "
          f"pred={v['additive_prediction_us']:+.2f} "
          f"actual={v['delta_BC_us']:+.2f} miss={v['miss_us']:+.2f}us")
    for k, v in sorted(res.get("stage_a_C_win_vs_G", {}).items()):
        p(f"C-STAGE-A {k}",
          f"num_bq={v['num_bq_per_chunk']} minor={v['t_a_head_minor_us']:.2f} "
          f"major={v['t_a_head_major_us']:.2f} "
          f"delta={v['delta_us']:+.2f}us ({v['delta_pct']:+.2f}%)")
    return res


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------


def _rowwise_setcmp(ref, got, ref_scores=None, got_scores=None, max_report=8):
    """Per-row index-set comparison, ignoring order and -1 padding.

    A row that differs is not immediately a failure: the two index sets are
    diffed and, when scores are available, the f32 score at each
    reference-only and candidate-only index is quoted so a tie broken
    differently can be told from a wrong answer.
    """
    assert ref.shape == got.shape, (ref.shape, got.shape)
    n_rows = ref.shape[0]
    bad = []
    for r in range(n_rows):
        a = set(int(x) for x in ref[r] if x >= 0)
        b = set(int(x) for x in got[r] if x >= 0)
        if a == b:
            continue
        only_ref = sorted(a - b)
        only_got = sorted(b - a)
        rec = dict(row=r, n_only_ref=len(only_ref), n_only_got=len(only_got),
                   only_ref=only_ref[:16], only_got=only_got[:16])
        if ref_scores is not None:
            rec["scores_only_ref"] = [
                float(ref_scores[r, i]) for i in only_ref[:16]
            ]
            rec["scores_only_got"] = [
                float((got_scores if got_scores is not None else
                       ref_scores)[r, i]) for i in only_got[:16]
            ]
            sr = set(rec["scores_only_ref"])
            sg = set(rec["scores_only_got"])
            rec["all_swapped_scores_equal"] = bool(sr == sg)
        bad.append(rec)
        if len(bad) >= max_report:
            break
    n_bad_total = 0
    for r in range(n_rows):
        a = set(int(x) for x in ref[r] if x >= 0)
        b = set(int(x) for x in got[r] if x >= 0)
        if a != b:
            n_bad_total += 1
    return dict(rows=n_rows, mismatching_rows=n_bad_total,
                identical=bool(n_bad_total == 0), examples=bad)


def run_correct(cells=CELLS):
    g, ref = PC.reference_topk(PHASE)
    p("reference",
      f"src streamindex_topk unchunked: shape={ref.shape} dtype={ref.dtype} "
      f"min={ref.min()} max={ref.max()}")
    res = {"phase": PHASE, "reference": "src streamindex_topk, unchunked",
           "reference_shape": list(ref.shape), "cells": {}, "scores": {}}

    _, shared = PC.build_inputs(PHASE)
    cache_kv = shared["cache_kv"]

    # Reference scores, for adjudicating any mismatch.  Stage A over the whole
    # 1024 tokens with the unmodified src kernel, i.e. the same numbers the
    # reference selected from.
    ref_arrs = PC.chunk_arrays(PHASE, 1, shared)
    sbody = PC.build_body(PHASE, 1, "a", 1, q_arm="incumbent", force_p1=True,
                          return_outputs=True)
    souts = jax.jit(sbody)(*PC.flat_args(ref_arrs, cache_kv))
    ref_scores = np.asarray(
        jax.block_until_ready(souts[1]).reshape(g["num_tokens"], -1))
    del souts

    # Stage-A score equality between the q layouts, at every G.  A stronger
    # statement than index equality when it holds, and the first thing to look
    # at when index equality does not.
    for G in GS:
        arrs = PC.chunk_arrays(PHASE, G, shared)
        blocks = {}
        for q_arm in ("head_minor", "head_major"):
            body = PC.build_body(PHASE, G, "a", 1, q_arm=q_arm,
                                 force_p1=True, return_outputs=True)
            outs = jax.jit(body)(*PC.flat_args(arrs, cache_kv))
            blocks[q_arm] = np.concatenate([
                np.asarray(jax.block_until_ready(s)).reshape(s.shape[0], -1)
                for s in outs[1:]
            ], axis=0)
            del outs
        a, b = blocks["head_minor"], blocks["head_major"]
        fin = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[fin].astype(np.float64) - b[fin].astype(np.float64))
        res["scores"][f"G{G}"] = dict(
            bitwise_identical=bool(np.array_equal(
                a.view(np.int32), b.view(np.int32))),
            max_abs_diff=float(d.max()) if d.size else 0.0,
            n_differing=int((a.view(np.int32) != b.view(np.int32)).sum()),
            n_elems=int(a.size),
            inf_pattern_identical=bool(
                np.array_equal(np.isfinite(a), np.isfinite(b))),
        )
        p(f"SCORES G={G}", json.dumps(res["scores"][f"G{G}"]))
        del blocks, a, b

    # Index sets, per cell.
    for q_arm, G in cells:
        key = f"{q_arm}/G{G}"
        try:
            arrs = PC.chunk_arrays(PHASE, G, shared)
            body = PC.build_body(PHASE, G, "pipe", 1, q_arm=q_arm,
                                 force_p1=True, return_outputs=True)
            with PC.p_tripwire(active=True):
                outs = jax.jit(body)(*PC.flat_args(arrs, cache_kv))
            got = np.concatenate(
                [np.asarray(jax.block_until_ready(t)) for t in outs[1:]],
                axis=0)
            del outs
            cmp = _rowwise_setcmp(ref, got, ref_scores=ref_scores)
            res["cells"][key] = cmp
            p(f"CORRECT {key}",
              f"{'IDENTICAL' if cmp['identical'] else 'MISMATCH'} "
              f"({cmp['mismatching_rows']}/{cmp['rows']} rows differ)")
            if not cmp["identical"]:
                p(f"CORRECT {key} examples",
                  json.dumps(cmp["examples"], default=str)[:4000])
        except Exception as e:  # noqa: BLE001
            res["cells"][key] = {"error": repr(e)}
            p(f"CORRECT {key}", "FAILED", repr(e))
            traceback.print_exc()

    res["overall"] = "PASS" if all(
        v.get("identical") for v in res["cells"].values()) else "FAIL"
    p("CORRECT OVERALL", res["overall"])
    return res


# ---------------------------------------------------------------------------


def run_plan():
    g = PC.geometry(PHASE)
    out = {"geometry": {k: str(v) for k, v in g.items()}, "chunks": {}}
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
    for G in GS:
        plan = PC.chunk_plan(PHASE, G)
        nt = plan[0]["num_tokens"]
        out["chunks"][f"G{G}"] = dict(
            tokens_per_chunk=nt,
            num_bq_per_chunk=max(1, -(-nt // g["bq_per_block"])),
            num_bkv=max(1, -(-g["context_len"] // (g["bkv_p"] *
                                                   g["page_size"]))),
            waves_guard=PC.waves_guard(nt),
            incumbent_pick_partition=sc._pick_partition(
                nt, g["num_positions"], g["k"]),
            seq_lens=[int(x) for c in plan for x in c["seq_lens"]],
        )
        p(f"plan G={G}", json.dumps(out["chunks"][f"G{G}"], default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("plan", "perf", "correct"))
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--outdir", default=os.environ.get("CDK_OUTPUT_DIR"))
    a = ap.parse_args()

    p("arm", a.arm)
    p("module_md5s", json.dumps(PC.module_md5s(), indent=2))
    if a.arm != "plan":
        tpu_banner()

    if a.arm == "plan":
        payload = run_plan()
    elif a.arm == "perf":
        payload = run_perf(trials=a.trials)
    else:
        payload = run_correct()

    payload["module_md5s"] = PC.module_md5s()
    emit_json(f"bxc-{a.arm}", payload, outdir=a.outdir)


if __name__ == "__main__":
    main()
