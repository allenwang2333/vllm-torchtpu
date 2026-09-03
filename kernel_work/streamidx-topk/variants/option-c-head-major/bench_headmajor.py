"""Option C measurement 1: prefill stage-A device time, both row orderings.

Instrument named in `measurement_and_design.md` §5's per-option measurement
plan ("`instruments/bench_headmajor.py` (new file)"); it lives in the variant
folder instead because `instruments/` is read-only after step 1 and this file
is specific to one candidate.

METHODOLOGY -- this is the part `probes/stage-b-ksweep.md` §0 settled and it is
not negotiable here.  `time_arm` enqueues `iters` calls asynchronously and
blocks once, so its per-call figure is `max(device, host enqueue)` plus an
un-amortised first call.  A dispatch floor is a **clamp, not an addend**;
subtracting one from a batched async measurement is invalid.  Device time is
therefore taken as the **slope of R chained calls inside one `jit`**:

    s = full(-inf); for r in range(R): s = stage_a(..., scores_init=s)

The dependency is the real one -- each call's `scores_init` operand *is* the
previous call's output, and the pallas call carries
`input_output_aliases={9: 0}`, so the R launches can be neither CSEd,
reordered nor overlapped.  Nothing artificial (`x * 0`, which XLA folds) is
needed.  The slope is marginal device time per call; the intercept is the host
term plus the one `full(-inf)`.

Three statistics are reported per arm and they must not be confused:
  chain slope   -- dispatch-free device time per call.  THIS is the number.
  batched best  -- `max(device, host enqueue)`, `iters` stated.  A wall clock.
  serial best   -- one call with a block after it.  A wall clock.

Best-of-N is optimistic by construction; every figure is a lower bound.
Run this UNPINNED on the whole 4-chip pod -- a fanned-out capture is contended
and yields ratios only.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_PROBES = _HERE.parent.parent / "probes"
for _p in (str(_HERE), str(_PROBES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import stage_a_c as S  # noqa: E402
from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

CHAIN_R = (1, 2, 3, 4)


def _fit(xs, ys):
    """Least squares y = a + b x, plus r^2.  Returns (intercept, slope, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx)**2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my)**2 for y in ys)
    ss_res = sum((y - (a + b * x))**2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return a, b, r2


def build_arm(arm, g, raw, trials):
    """Returns (single_call_fn, chain_fns_by_R, prepared_q, prepared_w)."""
    f = S.prefill_stage_a(arm, g)
    q, w = S.prepared_operands(arm, g, raw)
    q = jax.block_until_ready(q)
    w = jax.block_until_ready(w)

    out_shape = (g["num_tokens"], g["num_sublanes_total"], 128)

    def make_chain(R):

        def chained(q, w, cache_kv, seq_lens, page_indices, cu_q_lens, lo, hi):
            s = jnp.full(out_shape, -jnp.inf, jnp.float32)
            for _ in range(R):
                s = f(q, w, cache_kv, s, seq_lens, page_indices, cu_q_lens, lo,
                      hi)
            return s

        return jax.jit(chained)

    return f, q, w, make_chain


def run_arm(arm, g, raw, trials, do_chain=True):
    p("arm", arm, "building")
    f, q, w, make_chain = build_arm(arm, g, raw, trials)

    args = (q, w, raw["cache_kv"], raw["seq_lens"], raw["page_indices"],
            raw["cu_q_lens"], jnp.array(0, jnp.int32),
            jnp.array(g["num_seqs"], jnp.int32))

    rec = {"arm": arm}

    # --- one call, for the output the correctness compare reads ------------
    out = make_chain(1)(*args)
    out = jax.block_until_ready(out)
    rec["out_shape"] = list(out.shape)
    rec["finite_frac"] = float(jnp.mean(jnp.isfinite(out)))
    rec["checksum"] = float(jnp.sum(jnp.where(jnp.isfinite(out), out, 0.0)))

    if not do_chain:
        return rec, out

    # --- chain fit ----------------------------------------------------------
    xs, ys, per_r = [], [], {}
    for R in CHAIN_R:
        fn = make_chain(R)
        t = time_arm(fn, args, warmup=3, trials=trials, target_s=0.05)
        xs.append(R)
        ys.append(t["best_s"])
        per_r[R] = dict(best_us=t["best_s"] * 1e6,
                        median_us=t["median_s"] * 1e6,
                        iters=t["iters"],
                        compile_s=t["compile_s"])
        p(arm, f"R={R}", f"best {t['best_s']*1e6:.2f} us",
          f"iters={t['iters']}")
    a, b, r2 = _fit(xs, ys)
    rec["chain"] = per_r
    rec["chain_slope_us"] = b * 1e6
    rec["chain_intercept_us"] = a * 1e6
    rec["chain_r2"] = r2
    p(arm, "CHAIN SLOPE (device us/call)", f"{b*1e6:.2f}",
      f"intercept {a*1e6:.2f} us", f"r2 {r2:.6f}")

    # --- wall clocks, labelled as such --------------------------------------
    t1 = time_arm(make_chain(1), args, warmup=3, trials=trials, target_s=0.05)
    rec["batched_best_us"] = t1["best_s"] * 1e6
    rec["batched_iters"] = t1["iters"]

    serial = []
    fn1 = make_chain(1)
    for _ in range(trials):
        t0 = time.perf_counter()
        o = fn1(*args)
        jax.block_until_ready(o)
        serial.append(time.perf_counter() - t0)
    rec["serial_best_us"] = min(serial) * 1e6
    p(arm, "batched best (wall)", f"{rec['batched_best_us']:.2f} us",
      f"iters={t1['iters']}", "| serial best (wall)",
      f"{rec['serial_best_us']:.2f} us")
    return rec, out


def transpose_cost(g, raw, trials):
    """What `head_major_hbm` moves OUT of stage A and into the caller.

    The op is (T,H,D) -> (H,T,D) on the padded fp8 q.  It is NOT chain-fitted:
    a single transpose changes shape so it cannot be chained, and the obvious
    round-trip link `transpose(transpose(x))` is folded to the identity by
    XLA's algebraic simplifier -- exactly the failure the ksweep probe warns
    about with `x * 0`.  So two things are reported and both are labelled:

      * a batched best-of-N **wall clock**, i.e. `max(device, host enqueue)`,
        which at this size is host-dominated and is an UPPER bound on the
        device cost;
      * the analytic HBM floor, read + write at the measured 3.70 TB/s, which
        is a LOWER bound.

    Neither is a chain slope and neither is comparable to the stage-A numbers.
    In production this transpose would most likely fuse into whatever produces
    `q`, so its true marginal cost is somewhere below the wall clock; that is
    an argument for reading `head_major_hbm` as an optimistic bound, not as a
    measured configuration.
    """
    qr = S.sxc.prepare_q_inputs(raw["q_raw"], "head_minor")
    qr = jax.block_until_ready(qr)
    fn = jax.jit(lambda x: jnp.transpose(x, (1, 0, 2)))
    t = time_arm(fn, (qr, ), warmup=3, trials=trials, target_s=0.05)
    nbytes = int(qr.size)  # fp8 -> 1 byte each
    floor_us = 2 * nbytes / 3.70e12 * 1e6
    p("q head-major transpose (outside stage A)",
      f"batched WALL {t['best_s']*1e6:.2f} us (iters={t['iters']})",
      f"| HBM floor {floor_us:.2f} us")
    return dict(batched_wall_us=t["best_s"] * 1e6,
                iters=t["iters"],
                hbm_floor_us=floor_us,
                bytes_moved=2 * nbytes,
                note="wall clock, not a chain slope; upper bound. HBM floor "
                "is modelled at 3.70e12 B/s and is a lower bound.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms",
                    default="incumbent,head_minor,head_major,head_major_hbm")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--q-len", type=int, default=None)
    ap.add_argument("--context-len", type=int, default=None)
    ap.add_argument("--no-chain", action="store_true")
    args = ap.parse_args()

    tpu_banner()
    p("incumbent module", S.sx.__file__, S.source_md5(S.sx))
    p("variant module", S.sxc.__file__, S.source_md5(S.sxc))

    g = S.prefill_geometry(args.q_len, args.context_len)
    p("geometry", {
        k: g[k]
        for k in ("num_seqs", "q_len", "context_len", "bq_sz", "bkv_sz",
                  "num_bq", "num_bkv", "num_tokens", "num_sublanes_total",
                  "total_pages")
    })

    raw = S.build_prefill_inputs(g)
    for k, v in list(raw.items()):
        raw[k] = jax.block_until_ready(v)

    arms = [a for a in args.arms.split(",") if a]
    recs, outs = [], {}
    for arm in arms:
        try:
            rec, out = run_arm(arm, g, raw, args.trials,
                               do_chain=not args.no_chain)
            recs.append(rec)
            outs[arm] = out
        except Exception as e:  # a failed arm is a result
            import traceback
            traceback.print_exc()
            recs.append({"arm": arm, "error": repr(e)[:2000]})
            p("arm", arm, "FAILED", repr(e)[:400])

    # --- numeric agreement, arm vs incumbent --------------------------------
    cmp = {}
    ref_name = "incumbent" if "incumbent" in outs else (
        "head_minor" if "head_minor" in outs else None)
    if ref_name:
        ref = outs[ref_name]
        finite = jnp.isfinite(ref)
        for arm, out in outs.items():
            same_inf = bool(jnp.all(jnp.isfinite(out) == finite))
            d = jnp.where(finite, jnp.abs(out - ref), 0.0)
            den = jnp.where(finite, jnp.abs(ref), 1.0)
            cmp[arm] = dict(
                vs=ref_name,
                bit_identical=bool(jnp.all(jnp.where(finite, out == ref,
                                                     True))) and same_inf,
                same_mask=same_inf,
                max_abs=float(jnp.max(d)),
                max_rel=float(jnp.max(d / jnp.maximum(den, 1e-30))),
            )
            p("numeric", arm, "vs", ref_name, cmp[arm])

    payload = dict(
        what="Option C prefill stage-A timing, head_minor vs head_major",
        geometry={k: g[k] for k in sorted(g)},
        trials=args.trials,
        chain_R=list(CHAIN_R),
        arms=recs,
        numeric_vs_reference=cmp,
        incumbent_md5=S.source_md5(S.sx),
        variant_md5=S.source_md5(S.sxc),
    )
    if "head_major_hbm" in outs:
        payload["q_transpose_outside_stage_a"] = transpose_cost(
            g, raw, args.trials)

    # headline table
    print("\n===== STAGE A PREFILL, bq_sz=%d =====" % g["bq_sz"])
    print(f"{'arm':16} {'chain slope us':>15} {'r2':>9} "
          f"{'batched wall us':>16} {'serial wall us':>15}")
    for r in recs:
        if "chain_slope_us" not in r:
            print(f"{r['arm']:16} {'FAILED/skipped':>15}")
            continue
        print(f"{r['arm']:16} {r['chain_slope_us']:15.2f} {r['chain_r2']:9.6f} "
              f"{r['batched_best_us']:16.2f} {r['serial_best_us']:15.2f}")

    base = next((r for r in recs if r["arm"] == "head_minor"
                 and "chain_slope_us" in r), None)
    if base:
        for r in recs:
            if "chain_slope_us" in r and r["arm"] != "head_minor":
                d = r["chain_slope_us"] / base["chain_slope_us"] - 1.0
                print(f"  {r['arm']:20} vs head_minor: {d*100:+.2f}%")

    emit_json("option_c_stage_a_timing", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
