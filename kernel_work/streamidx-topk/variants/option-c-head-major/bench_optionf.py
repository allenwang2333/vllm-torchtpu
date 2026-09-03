"""Option F -- bf16 score-matmul operands -- measured against the incumbent.

Option F was proposed on the premise that stage A "moves q and k as f32 and
gets bf16 arithmetic for it", so storing the operands as bf16 would halve the
bytes at identical accuracy.  **That premise is false for this kernel**: both
operands are already `float8_e4m3fn` (one byte), so bf16 would DOUBLE the
operand bytes, and on generation 7 it also halves MXU throughput -- 1419.9
Tops/s for fp8 against 764.8 for bf16, `05-compute-units.md`.

What is left of F that is implementable without changing the KV cache format
is its arithmetic half, and that is what this measures.  `mm_dtype="bf16"`
widens both operands immediately before the einsum.  The widening is exact (an
e4m3 value has 3 mantissa bits and bf16 has 7), so the arm is numerically a
superset of the incumbent and any time difference is the MXU pass alone, with
HBM traffic held byte-for-byte constant.

That makes this arm useful for something F itself is not: it measures how much
of stage A the score matmul actually owns.  Halving MXU throughput should cost
the MXU's whole share of the runtime again; if it costs much less, stage A is
issue-bound and no matmul-side change can pay.

Arms:
    incumbent   the unmodified src kernel
    f_bf16      Option F's arithmetic half, incumbent layout
    head_major  Option C, for the compose question
    f_bf16_hm   F+C, because a -20% and a +X% do not add if they attack the
                same bound

Reuses `bench_headmajor.run_arm`/`_fit` and `bench_numbkv._run_point` verbatim,
so prefill and decode are measured exactly the way 1 and 7 were.

    python bench_optionf.py [--arms ...] [--trials 7]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_headmajor as B  # noqa: E402
import bench_numbkv as N  # noqa: E402
import stage_a_c as S  # noqa: E402
from _probe_util import emit_json, p, tpu_banner  # noqa: E402

ARMS_DEFAULT = "incumbent,f_bf16,head_major,f_bf16_hm"

# One prefill point and one decode point.  The num_bkv curve is already in 7;
# F is a per-matmul change and does not interact with num_bkv, so a sweep here
# would buy nothing that the two endpoints do not.
PREFILL_CTX = 9216
DECODE_CTX = 8704


def _deltas(rec):
    by = {r["arm"]: r for r in rec["arms"] if "chain_slope_us" in r}
    inc = by.get("incumbent", {}).get("chain_slope_us")
    out = {}
    if inc:
        for arm, r in by.items():
            out[arm] = (r["chain_slope_us"] / inc - 1.0) * 100.0
    return by, inc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=ARMS_DEFAULT)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--prefill-ctx", type=int, default=PREFILL_CTX)
    ap.add_argument("--decode-ctx", type=int, default=DECODE_CTX)
    ap.add_argument("--no-decode", dest="decode", action="store_false",
                    default=True)
    args = ap.parse_args()

    tpu_banner()
    p("incumbent module", S.sx.__file__, S.source_md5(S.sx))
    p("variant module", S.sxc.__file__, S.source_md5(S.sxc))
    arms = [a for a in args.arms.split(",") if a]
    p("arms", {a: (S.ARMS[a][1], S.ARM_MM_DTYPE.get(a, "native"))
               for a in arms})

    points = []
    g = S.prefill_geometry(None, args.prefill_ctx)
    rec = N._run_point(f"prefill_q1024_ctx{args.prefill_ctx}", g, arms,
                       args.trials, S.prefill_stage_a)
    rec["phase"] = "prefill"
    points.append(rec)

    if args.decode:
        g = N.decode_geometry(None, args.decode_ctx)
        rec = N._run_point(f"decode_b{g['num_seqs']}_ctx{args.decode_ctx}", g,
                           arms, args.trials, N.decode_stage_a)
        rec["phase"] = "decode"
        points.append(rec)

    print("\n===== OPTION F (bf16 matmul operands), chain slope us/call =====")
    hdr = f"{'shape':26} {'ph':<8}" + "".join(f"{a:>13}" for a in arms)
    print(hdr)
    for rec in points:
        by, inc, dl = _deltas(rec)
        cells = ""
        for a in arms:
            if a not in by:
                cells += f"{'FAILED':>13}"
            elif a == "incumbent":
                cells += f"{by[a]['chain_slope_us']:13.2f}"
            else:
                cells += f"{by[a]['chain_slope_us']:8.2f}{dl[a]:+5.1f}"
        print(f"{rec['label']:26} {rec['phase']:<8}{cells}")
    print("\nvalue printed for a non-incumbent arm is 'us  delta%_vs_incumbent'.")
    print("delta% for f_bf16 is the cost of moving the score matmul off the "
          "fp8 MXU pass at CONSTANT HBM bytes -- it is a lower bound on the "
          "matmul's share of stage A, not a bandwidth result.")
    for rec in points:
        by, inc, dl = _deltas(rec)
        r2s = [r["chain_r2"] for r in by.values()]
        print(f"  {rec['label']:26} min r2 {min(r2s):.6f}" if r2s else
              f"  {rec['label']:26} no fit")
        print(f"  {rec['label']:26} numeric {rec['numeric_vs_incumbent']}")

    emit_json(
        "option_f_bf16_operands",
        dict(what="stage-A chain slope with bf16 score-matmul operands",
             premise_note=("q and k are already float8_e4m3fn; bf16 operands "
                           "would double HBM bytes, not halve them"),
             method="chain slope, R=%s, best-of-%d, unpinned" %
             (list(B.CHAIN_R), args.trials),
             points=points,
             incumbent_md5=S.source_md5(S.sx),
             variant_md5=S.source_md5(S.sxc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
