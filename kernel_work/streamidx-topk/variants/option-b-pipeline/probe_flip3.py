#!/usr/bin/env python3
"""Flip 3 -- does P3's 41.6 % TC/SC overlap survive a *real* cross-chunk
dependency?

P3 measured the overlap with two arms that shared no data: "arm B is fed a
pre-made score matrix that arm A never writes".  §5 of
`measurement_and_design.md` calls that "the one thing P3 deliberately did not
do".  This probe closes it: arm `pipe` runs G chunks in which stage B of chunk
i consumes stage A of chunk i's actual output, through HBM, inside one
`jax.jit`.

Arms and the arithmetic
-----------------------
    t_a(G)     G stage-A calls, no stage B
    t_b(G)     G stage-B calls on pre-made scores, no stage A
    t_pipe(G)  the pipeline, stage B(i) consuming stage A(i)

    hidden(G)  = t_a(G) + t_b(G) - t_pipe(G)
    ideal(G)   = min(t_a(G)/G, t_b(G)/G) * (G - 1)      <- §3.4's ideal saving
    overlap(G) = hidden(G) / ideal(G)

`overlap` is the quantity §3.4 applies P3's 41.6 % to, so it is the one the
§5 gate (>= 20 % decode, >= 7 % prefill) is stated in.  Because every arm is a
chain slope, the fixed per-program cost is in the intercept and drops out --
P3's `F = 2*t_a(1) - t_a(2)` correction is not needed and is not applied.

`t_pipe(1)` is the *serial reference*: one stage A, then one stage B that
depends on it.  No overlap is possible inside it, and it is the incumbent's
structure.  `t_pipe(1) - t_pipe(G)` is therefore the decision number, measured
directly and with no model in it.

Timing: chain slope over R copies inside one jit, best-of-N per point.
`probes/stage-b-ksweep.md` §0 -- a batched async measurement is
`max(device, host)`, so a dispatch floor is a clamp, not an addend, and
subtracting it is invalid.  Fits below r^2 = 0.999 are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "probes"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import pipeline as P  # noqa: E402
from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

REPS = (1, 2, 3, 4)


def linfit(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred)**2))
    ss_tot = float(np.sum((ys - ys.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def chain_slope(build, args, reps=REPS, trials=9, tag=""):
    """Device time per copy = slope of best-of-N wall clock against R."""
    xs, ys, detail = [], [], []
    for R in reps:
        t0 = time.perf_counter()
        fn = jax.jit(build(R))
        r = time_arm(fn, args, warmup=3, trials=trials, target_s=0.08)
        us = r["best_s"] * 1e6
        xs.append(R)
        ys.append(us)
        detail.append(
            dict(R=R, best_us=us, median_us=r["median_s"] * 1e6,
                 iters=r["iters"], compile_s=r["compile_s"],
                 wall_s=time.perf_counter() - t0))
        p(f"{tag} R={R}", f"best={us:.2f}us median={r['median_s']*1e6:.2f}us "
          f"iters={r['iters']} compile={r['compile_s']:.1f}s")
        del fn
    slope, intercept, r2 = linfit(xs, ys)
    p(f"{tag} FIT",
      f"slope={slope:.2f}us/copy intercept={intercept:.2f}us r2={r2:.5f}"
      f"{'' if r2 >= 0.999 else '   <-- REJECTED, r2 < 0.999'}")
    return dict(slope_us=slope, intercept_us=intercept, r2=r2, points=detail,
                accepted=bool(r2 >= 0.999))


# ---------------------------------------------------------------------------


def _capture_premade(phase, G, chunk_arrs, cache_kv):
    """Stage A's *actual* output, captured once outside the timed region and
    handed to arm `b` as its input.  This removes the score-distribution
    confound: arm `b` and arm `pipe` select over the same numbers, and stage B
    is data-dependent at the ~6 % level (`stage-b-ksweep.md` §1)."""
    g = P.geometry(phase)
    body = P.build_body(phase, G, "a", 1, force_p1=True, return_outputs=True)
    fn = jax.jit(body)
    outs = fn(*P.flat_args(chunk_arrs, cache_kv))
    jax.block_until_ready(outs)
    scores = outs[1:]
    n = g["num_positions"]
    return [jax.block_until_ready(s.reshape(s.shape[0], n)) for s in scores]


def run_phase(phase, gs=(1, 2, 4), also_nop1=(), trials=9):
    g, shared = P.build_inputs(phase)
    p("geometry", {
        k: v
        for k, v in g.items() if k not in ("case", )
    })
    cache_kv = shared["cache_kv"]
    res = {"phase": phase, "geometry": {k: str(v) for k, v in g.items()},
           "arms": {}, "guards": {}, "derived": {}}

    for G in gs:
        chunk_arrs = P.chunk_arrays(phase, G, shared)
        b = P.chunk_plan(phase, G)[0]["num_tokens"]
        wg = P.waves_guard(b)
        res["guards"][f"G{G}"] = wg
        p(f"G={G} waves_guard",
          f"b={wg['b']} b_pad={wg['b_pad']} num_waves={wg['num_waves']} "
          f"num_waves*32={wg['lhs']} -> {'PASS' if wg['ok'] else 'FAIL'}")

        premade = _capture_premade(phase, G, chunk_arrs, cache_kv)
        args = P.flat_args(chunk_arrs, cache_kv, premade)

        arms = ["a", "b", "pipe"] + (["b_chained"] if G > 1 else [])
        variants = [(True, "")] + [(False, "_nop1")] * (G in also_nop1)
        for force_p1, sfx in variants:
            for arm in arms:
                if sfx and arm == "a":
                    continue  # stage A does not depend on p
                key = f"{arm}{sfx}_G{G}"
                try:
                    with P.p_tripwire(active=force_p1):
                        out = chain_slope(
                            lambda R, arm=arm, G=G, fp=force_p1: P.build_body(
                                phase, G, arm, R, force_p1=fp),
                            args, trials=trials, tag=f"{phase}/{key}")
                    out["force_p1"] = force_p1
                    res["arms"][key] = out
                except P.PickPartitionCalled as e:
                    res["arms"][key] = {"error": f"TRIPWIRE {e}"}
                    p(f"{phase}/{key}", "TRIPWIRE FIRED", repr(e))
                except Exception as e:  # noqa: BLE001
                    res["arms"][key] = {"error": repr(e)}
                    p(f"{phase}/{key}", "FAILED", repr(e))
                    traceback.print_exc()

    # ---- the arithmetic -------------------------------------------------
    def slope(key):
        a = res["arms"].get(key)
        if not a or "slope_us" not in a or not a["accepted"]:
            return None
        return a["slope_us"]

    for G in gs:
        for sfx in ("", "_nop1"):
            ta, tb, tp = (slope(f"a_G{G}"), slope(f"b{sfx}_G{G}"),
                          slope(f"pipe{sfx}_G{G}"))
            if None in (ta, tb, tp):
                continue
            hidden = ta + tb - tp
            ideal = min(ta / G, tb / G) * (G - 1)
            d = dict(t_a_us=ta, t_b_us=tb, t_pipe_us=tp,
                     t_a_plus_t_b_us=ta + tb, hidden_us=hidden,
                     ideal_saving_us=ideal,
                     overlap_fraction=(hidden / ideal) if ideal > 0 else None,
                     hidden_as_frac_of_b=hidden / tb)
            ref = slope("pipe_G1")
            if ref is not None:
                d["t_pipe_G1_us"] = ref
                d["vs_serial_us"] = ref - tp
                d["vs_serial_pct"] = 100.0 * (ref - tp) / ref
            res["derived"][f"G{G}{sfx}"] = d
            p(f"{phase} DERIVED G={G}{sfx}",
              f"t_a={ta:.2f} t_b={tb:.2f} t_pipe={tp:.2f} "
              f"hidden={hidden:.2f} ideal={ideal:.2f} "
              f"overlap={d['overlap_fraction'] if d['overlap_fraction'] is None else round(d['overlap_fraction'],4)} "
              f"vs_serial={d.get('vs_serial_us', float('nan')):.2f}us "
              f"({d.get('vs_serial_pct', float('nan')):.2f}%)")
    return res


# ---------------------------------------------------------------------------
# HLO verification -- did the dependency survive into the compiled program?
# ---------------------------------------------------------------------------

_LHS = re.compile(r"^\s*(?:ROOT\s+)?(%?[\w.\-]+)\s*=\s*")
_OPCODE = re.compile(r"(?<=[\s\)\]\}])([a-z][\w\-]*)\(")


def _parse_hlo(text):
    """name -> (opcode, [operands], raw line).

    The opcode is the identifier immediately before the operand list; the
    operand list is read to its matching close paren, so a tuple-shaped
    result or a nested call in the metadata does not confuse it.
    """
    g, order = {}, []
    for line in text.splitlines():
        m = _LHS.match(line)
        if not m or "=" not in line:
            continue
        rest = line[m.end():]
        om = _OPCODE.search(" " + rest)
        if not om:
            continue
        opcode = om.group(1)
        i = om.end() - 1 - 1  # index of "(" in `rest`
        if i < 0 or rest[i] != "(":
            continue
        depth, j = 0, i
        for j in range(i, len(rest)):
            if rest[j] == "(":
                depth += 1
            elif rest[j] == ")":
                depth -= 1
                if depth == 0:
                    break
        ops = re.findall(r"%[\w.\-]+", rest[i:j + 1])
        name = m.group(1).lstrip("%")
        g[name] = (opcode, [o.lstrip("%") for o in ops], line)
        order.append(name)
    return g, order


def _reaches(graph, src, dst, limit=200000):
    """Is `dst` reachable from `src` by walking src's operands?"""
    seen, stack, steps = set(), [src], 0
    while stack:
        cur = stack.pop()
        steps += 1
        if steps > limit:
            return None
        if cur == dst:
            return True
        if cur in seen or cur not in graph:
            continue
        seen.add(cur)
        stack.extend(graph[cur][1])
    return False


def run_hlo(phase, gs=(1, 2, 4), R=2, force_p1=True, outdir=None):
    g, shared = P.build_inputs(phase)
    cache_kv = shared["cache_kv"]
    out = {"phase": phase, "R": R, "force_p1": force_p1, "cases": {}}
    for G in gs:
        chunk_arrs = P.chunk_arrays(phase, G, shared)
        n = g["num_positions"]
        premade = [
            jnp.zeros((c["num_tokens"], n), jnp.float32)
            for c in P.chunk_plan(phase, G)
        ]
        args = P.flat_args(chunk_arrs, cache_kv, premade)
        body = P.build_body(phase, G, "pipe", R, force_p1=force_p1)
        with P.p_tripwire(active=force_p1):
            comp = jax.jit(body).lower(*args).compile()
        text = comp.as_text()
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"{phase}_G{G}_R{R}.hlo.txt"),
                      "w") as f:
                f.write(text)

        graph, order = _parse_hlo(text)
        cc = [(nm, v) for nm, v in graph.items() if v[0] == "custom-call"]
        stage_a = [nm for nm, v in cc if "StreamIdxTC" in v[2]]
        stage_b = [nm for nm, v in cc if "sc_topk" in v[2]]
        # schedule position, from the order instructions appear in the
        # entry computation of an already-scheduled module
        pos = {nm: i for i, nm in enumerate(order)}
        stage_a.sort(key=lambda nm: pos[nm])
        stage_b.sort(key=lambda nm: pos[nm])

        case = {
            "num_custom_calls": len(cc),
            "num_stage_a": len(stage_a),
            "num_stage_b": len(stage_b),
            "expected_each": R * G,
            "stage_a_names": stage_a,
            "stage_b_names": stage_b,
            "schedule_order": [(nm, "A" if nm in stage_a else "B")
                               for nm in order if nm in set(stage_a + stage_b)
                               ],
            "async_starts": sum(1 for _, v in graph.items()
                                if v[0].endswith("-start")),
            "is_scheduled": "is_scheduled=true" in text,
            "stage_b_operands": {
                nm: [(o, graph[o][0] if o in graph else "?")
                     for o in graph[nm][1]]
                for nm in stage_b
            },
        }
        # 1. every stage B reaches SOME stage A  (dependency survived)
        # 2. stage B i reaches stage A i and no later one (pairing is right)
        # 3. stage A i+1 does NOT reach stage B i    (overlap is legal)
        dep, pair, free = [], [], []
        for i, bnm in enumerate(stage_b):
            hits = [j for j, anm in enumerate(stage_a)
                    if _reaches(graph, bnm, anm)]
            dep.append(bool(hits))
            pair.append(hits)
        for i in range(len(stage_a) - 1):
            if i < len(stage_b):
                free.append(bool(_reaches(graph, stage_a[i + 1],
                                          stage_b[i])))
        case["every_stage_b_depends_on_a_stage_a"] = all(dep) if dep else None
        case["stage_b_operand_reachability"] = pair
        case["stage_a_next_depends_on_stage_b_prev"] = free
        case["overlap_is_legal_in_the_graph"] = (not any(free)) if free else None
        out["cases"][f"G{G}"] = case
        p(f"HLO {phase} G={G}",
          f"custom-calls={len(cc)} A={len(stage_a)} B={len(stage_b)} "
          f"(expected {R*G} each) "
          f"every_B_depends_on_an_A={case['every_stage_b_depends_on_a_stage_a']} "
          f"B_i<-A_i pairing={pair} "
          f"A_(i+1)_depends_on_B_i={free}")
        for nm in stage_a[:2] + stage_b[:2]:
            p("  hlo line", graph[nm][2].strip()[:400])
    return out


# ---------------------------------------------------------------------------


def run_plan():
    out = {}
    for phase in ("decode", "prefill"):
        g = P.geometry(phase)
        out[phase] = {"geometry": {k: str(v) for k, v in g.items()},
                      "chunks": {}}
        for G in (1, 2, 4):
            plan = P.chunk_plan(phase, G)
            b = plan[0]["num_tokens"]
            wg = P.waves_guard(b)
            from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
            inc_p = sc._pick_partition(b, g["num_positions"], g["k"])
            out[phase]["chunks"][f"G{G}"] = dict(
                b_per_chunk=b, waves_guard=wg,
                incumbent_pick_partition=inc_p)
            p(f"{phase} G={G}",
              f"b={b} waves_guard={'PASS' if wg['ok'] else 'FAIL'} "
              f"(num_waves*32={wg['lhs']} vs b_pad={wg['b_pad']}) "
              f"incumbent _pick_partition={inc_p}")
    return out


def run_anchors(trials=9):
    """Contention control: re-measure stage B alone at the three chunk widths
    against `probes/stage-b-ksweep.md`'s unpinned chain slopes."""
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
    ref = {64: 29.00, 32: 17.95, 16: 17.35}  # ksweep: b=16 is the p=1 forced
    out = {}
    n, k = 10240, 2048
    for b in (64, 32, 16):
        scores = jax.random.normal(jax.random.key(7), (b, n), jnp.float32)
        rl = jnp.full((b, ), 8704, jnp.int32)

        def build(R, b=b):

            def body(s, rl):
                eps = jnp.int32(0)
                o = None
                for _ in range(R):
                    o = sc._sc_topk_direct(s, k, rl + eps, write_empty=True)
                    eps = jnp.minimum(o[0, 0], jnp.int32(0))
                return (eps, )

            return body

        r = chain_slope(build, (scores, rl), trials=trials,
                        tag=f"anchor/b{b}_p1")
        r["ksweep_reference_us"] = ref[b]
        r["delta_pct"] = 100.0 * (r["slope_us"] - ref[b]) / ref[b]
        p(f"anchor b={b}",
          f"{r['slope_us']:.2f}us vs ksweep {ref[b]:.2f}us "
          f"({r['delta_pct']:+.1f}%)")
        out[f"b{b}"] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("plan", "decode", "prefill", "hlo", "anchors"))
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--outdir", default=os.environ.get("CDK_OUTPUT_DIR"))
    a = ap.parse_args()

    p("arm", a.arm)
    p("module_md5s", json.dumps(P.module_md5s(), indent=2))
    if a.arm != "plan":
        tpu_banner()

    if a.arm == "plan":
        payload = run_plan()
    elif a.arm == "anchors":
        payload = run_anchors(trials=a.trials)
    elif a.arm == "hlo":
        payload = {
            ph: run_hlo(ph, outdir=os.path.join(a.outdir or "/tmp", "hlo"))
            for ph in ("decode", "prefill")
        }
    elif a.arm == "decode":
        payload = run_phase("decode", also_nop1=(4, ), trials=a.trials)
    else:
        payload = run_phase("prefill", trials=a.trials)

    payload["module_md5s"] = P.module_md5s()
    emit_json(f"flip3-{a.arm}", payload, outdir=a.outdir)


if __name__ == "__main__":
    main()
