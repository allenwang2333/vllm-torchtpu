#!/usr/bin/env python3
"""P3 -- can a TensorCore grid and a SparseCore `VectorSubcoreMesh` be in
flight at the same time?

This gates the largest lever identified in step 2.  `CONTRADICTIONS.md` §9
separates the A->B *kernel* edge (the score matrix's HBM round trip, modelled at
1.75 us = 0.79% of the 221.4 us nightly decode call) from the A->B *unit* edge
(TensorCore stalled while SparseCore works, measured at ~174.6 us/layer of TC
stall at AgentX, of which ~71 us/layer is attributable to `sc_topk_direct`).
The unit edge is worth ~40x the kernel edge, and attacking it needs no
TC<->SC shared memory: stage A can score sequence group *g* while stage B
selects on group *g-1*, pipelined **through HBM** -- *if the two units can be in
flight at once*.  That "if" is what this measures.

Method
------
Arm A is the real stage-A decode `pallas_call` on TensorCore at the nightly
shape (grid 16, `bq_sz=1`, `bkv_sz=2048`), so the TC work is the work the design
actually wants to overlap.  Arm B is `sparsecore_topk` at the nightly stage-B
shape (b=64, n=10240, k=2048), which runs on SparseCore via
`jax.experimental.pallas.tpu_sc.VectorSubcoreMesh`.

**Independence.**  Arm B is fed a *pre-made* score matrix that arm A never
writes, so there is no value flowing from A to B.  They share no buffer: each
stage-A rep allocates its own `-inf` score buffer and aliases only that.  The
arms are combined by returning both outputs from one `jax.jit`, so nothing is
dead-code eliminated, and XLA is free -- but not obliged -- to schedule them
concurrently.

**Controls, which are the part that makes the answer readable.**  `t_a(2)` and
`t_b(2)` run two *independent* copies of the same unit's work.  Those cannot
overlap (one unit), so they establish what "no overlap" looks like in this
harness: `t_x(2) ~ 2*t_x(1)`.  Only against that baseline does `t_both <
t_a + t_b` mean anything.

**Three dispatch shapes are tried**, because "can they overlap" has a different
answer per mechanism and a clean "no" is a valuable result:
  `one_jit`         -- both custom calls in a single XLA program
  `two_executables` -- two jitted programs enqueued back to back, blocked once
  `two_threads`     -- the same two programs enqueued from two Python threads

Overlap shows up as `t_both < t_a + t_b`.  Overlap fraction is reported as
`(t_a + t_b - t_both) / min(t_a, t_b)`: 0 = fully serial, 1 = the shorter arm is
entirely hidden.

Caveat carried into the report: on a fanned-out pod every arm is contended by
the other three shards.  The *ratio* between arms in this probe is measured in
one process against its own controls, which is what the question needs; the
absolute microseconds are not clean timings.

Run:  python probe_p3_tc_sc_overlap.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import _stage_a  # noqa: E402
from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

REPS = (1, 2)


def main():
    tpu_banner()
    from jax.experimental.pallas import tpu as pltpu
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod

    sc = pltpu.get_tpu_info().sparse_core
    p("sparse_core", sc)
    if sc is None:
        p("VERDICT", "UNSUPPORTED - no SparseCore on this stack; P3 cannot run")
        emit_json("P3-tc-sc-overlap", {"verdict": "UNSUPPORTED_no_sparsecore"})
        return
    for f in ("num_cores", "num_subcores", "num_lanes", "vmem_capacity_bytes"):
        p(f"sparse_core.{f}", getattr(sc, f, "<absent>"))

    g, inp = _stage_a.build_nightly_inputs()
    p("stage_a_source_md5", _stage_a.source_md5())
    stage_a = _stage_a.nightly_decode_call1(g)  # seq_batch_size 4, grid 16

    B, N, K = g["num_tokens"], g["num_positions"], g["k"]
    p("shapes", f"stage A: 64 seqs x 1 tok, grid 16, out "
      f"({B},{g['num_sublanes_total']},128) f32 = "
      f"{g['score_bytes']/1e6:.2f} MB | stage B: ({B},{N}) f32 -> ({B},{K}) i32"
      )

    n_seqs = jnp.int32(g["num_seqs"])
    zero = jnp.int32(0)

    max_reps = max(REPS)
    # Independent inputs per rep, so nothing can be CSE'd.
    qs = [
        jax.random.normal(jax.random.key(100 + i),
                          (B, g["h_idx"], g["d_idx"]),
                          jnp.float32).astype(jnp.float8_e4m3fn)
        for i in range(max_reps)
    ]
    qs = [jax.block_until_ready(_stage_a.prepare_q_inputs(q)) for q in qs]
    scores_b = [
        jax.block_until_ready(
            jax.random.normal(jax.random.key(200 + i), (B, N), jnp.float32))
        for i in range(max_reps)
    ]
    row_lengths = jnp.full((B, ), g["context_len"], jnp.int32)

    def build(do_a, do_b, reps):

        def body(qs_, scores_b_, w, cache_kv, seq_lens, page_indices,
                 cu_q_lens, rl, start, end):
            outs = []
            if do_a:
                for i in range(reps):
                    init = jnp.full(
                        (B, g["num_sublanes_total"], 128), -jnp.inf,
                        jnp.float32)
                    outs.append(
                        stage_a(qs_[i], w, cache_kv, init, seq_lens,
                                page_indices, cu_q_lens, start, end))
            if do_b:
                for i in range(reps):
                    outs.append(
                        scmod.sparsecore_topk(scores_b_[i], K, row_lengths=rl,
                                              write_empty_rows=True))
            return tuple(outs)

        return jax.jit(body)

    def args_for(reps):
        return (tuple(qs[:reps]), tuple(scores_b[:reps]), inp["weights"],
                inp["cache_kv"], inp["seq_lens"], inp["page_indices"],
                inp["cu_q_lens"], row_lengths, zero, n_seqs)

    results = {
        "geometry": g,
        "sparse_core": {
            f: getattr(sc, f, None)
            for f in ("num_cores", "num_subcores", "num_lanes",
                      "vmem_capacity_bytes")
        },
        "stage_a_source_md5": _stage_a.source_md5(),
        "one_jit": {},
        "two_executables": {},
        "two_threads": {},
        "errors": {},
    }

    # ---------------- mechanism 1: one jit -------------------------------
    t = {}
    for reps in REPS:
        for tag, (da, db) in (("a", (True, False)), ("b", (False, True)),
                              ("both", (True, True))):
            key = f"{tag}{reps}"
            try:
                fn = build(da, db, reps)
                r = time_arm(fn, args_for(reps), warmup=3, trials=7,
                             target_s=0.08)
                t[key] = r["best_s"] * 1e6
                results["one_jit"][key] = {
                    "best_us": r["best_s"] * 1e6,
                    "median_us": r["median_s"] * 1e6,
                    "iters": r["iters"],
                    "compile_s": r["compile_s"],
                }
                p(f"one_jit/{key}", f"best={r['best_s']*1e6:.2f}us "
                  f"median={r['median_s']*1e6:.2f}us iters={r['iters']} "
                  f"compile={r['compile_s']:.1f}s")
            except Exception as e:
                results["errors"][f"one_jit/{key}"] = repr(e)
                p(f"one_jit/{key}", "FAILED", repr(e))
                traceback.print_exc()

    for reps in REPS:
        ka, kb, kab = f"a{reps}", f"b{reps}", f"both{reps}"
        if all(x in t for x in (ka, kb, kab)):
            serial = t[ka] + t[kb]
            frac = (serial - t[kab]) / min(t[ka], t[kb])
            results["one_jit"][f"overlap_reps{reps}"] = {
                "t_a_us": t[ka],
                "t_b_us": t[kb],
                "t_both_us": t[kab],
                "t_a_plus_t_b_us": serial,
                "overlap_fraction": frac,
            }
            p(f"one_jit OVERLAP reps={reps}",
              f"t_a={t[ka]:.2f} t_b={t[kb]:.2f} t_both={t[kab]:.2f} "
              f"t_a+t_b={serial:.2f} overlap_fraction={frac:.3f}")
    # control: does the harness's additive model hold within one unit?
    for tag in ("a", "b"):
        if f"{tag}1" in t and f"{tag}2" in t:
            ratio = t[f"{tag}2"] / t[f"{tag}1"]
            results["one_jit"][f"control_{tag}2_over_{tag}1"] = ratio
            p(f"one_jit CONTROL {tag}2/{tag}1", f"{ratio:.3f} "
              f"(2.00 = perfectly serial, <2 = the harness already overlaps "
              f"within one unit and the overlap number above is not clean)")

    # ---------------- mechanism 2: two executables -----------------------
    try:
        fa = build(True, False, 1)
        fb = build(False, True, 1)
        rest = args_for(1)[2:]
        # warm
        for _ in range(3):
            jax.block_until_ready(fa((qs[0], ), (scores_b[0], ), *rest))
            jax.block_until_ready(fb((qs[0], ), (scores_b[0], ), *rest))
        iters = 40

        def timeit(fn_list):
            best = float("inf")
            for _ in range(7):
                t0 = time.perf_counter()
                for _ in range(iters):
                    outs = [
                        f((qs[0], ), (scores_b[0], ), *rest) for f in fn_list
                    ]
                jax.block_until_ready(outs)
                best = min(best, (time.perf_counter() - t0) / iters)
            return best * 1e6

        ta = timeit([fa])
        tb = timeit([fb])
        tab = timeit([fa, fb])
        frac = (ta + tb - tab) / min(ta, tb)
        results["two_executables"] = {
            "t_a_us": ta,
            "t_b_us": tb,
            "t_both_us": tab,
            "t_a_plus_t_b_us": ta + tb,
            "overlap_fraction": frac,
            "iters": iters,
        }
        p("two_executables",
          f"t_a={ta:.2f} t_b={tb:.2f} t_both={tab:.2f} "
          f"t_a+t_b={ta+tb:.2f} overlap_fraction={frac:.3f}")
    except Exception as e:
        results["errors"]["two_executables"] = repr(e)
        p("two_executables", "FAILED", repr(e))
        traceback.print_exc()

    # ---------------- mechanism 3: two threads ---------------------------
    try:
        rest = args_for(1)[2:]
        out_box = {}

        def run(name, fn, n):
            o = None
            for _ in range(n):
                o = fn((qs[0], ), (scores_b[0], ), *rest)
            jax.block_until_ready(o)
            out_box[name] = o

        iters = 40
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            th = [
                threading.Thread(target=run, args=("a", fa, iters)),
                threading.Thread(target=run, args=("b", fb, iters)),
            ]
            for x in th:
                x.start()
            for x in th:
                x.join()
            best = min(best, (time.perf_counter() - t0) / iters)
        tab_thr = best * 1e6
        base = results["two_executables"]
        frac = ((base["t_a_us"] + base["t_b_us"] - tab_thr) /
                min(base["t_a_us"], base["t_b_us"]))
        results["two_threads"] = {
            "t_both_us": tab_thr,
            "t_a_plus_t_b_us": base["t_a_us"] + base["t_b_us"],
            "overlap_fraction": frac,
            "iters": iters,
        }
        p("two_threads", f"t_both={tab_thr:.2f} "
          f"t_a+t_b={base['t_a_us']+base['t_b_us']:.2f} "
          f"overlap_fraction={frac:.3f}")
    except Exception as e:
        results["errors"]["two_threads"] = repr(e)
        p("two_threads", "FAILED", repr(e))
        traceback.print_exc()

    emit_json("P3-tc-sc-overlap", results)


if __name__ == "__main__":
    main()
