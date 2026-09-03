"""Option C follow-up 1: does the win survive at small `num_bkv`?

WHY THIS EXISTS
---------------
`head_major` pays an in-kernel transpose once per *bq block* (`load_bq`) and
saves a cross-lane reduction tree once per *(bq block, bkv block)* pair
(`compute_scores`).  The saving therefore amortises over

    num_bkv = max(1, cdiv(kv_len_max, bkv_sz))

and at `num_bkv == 1` the transpose is paid once per use.  RESULTS.md §6.2(1)
named this "the likeliest way this disappoints".  This script finds where, if
anywhere, the curve crosses zero.

SHAPES
------
`bkv_sz = num_kv_pages_per_block * page_size = 2 * 1024 = 2048` for BOTH the
prefill (MIXED, index [2]) and the decode (index [0]) calls --
`mla_attention_op.py:119` is `(2, 2, 2)` and `attention.py:1323` is a literal
1024.  `compression_ratio = 1` (`mla_attention_op.py:116`), so
`kv_len == context_len` and `num_bkv = cdiv(context_len, 2048)`.

The prefill sweep holds `max_model_len = 9216` for ctx <= 9216, which is what
the `prod_prefill_q1024_ctx*` rows do (all three have blk_tbl 9216, 80
sublanes) -- so `num_sublanes_total`, the score buffer and the output shape are
CONSTANT across those points and `num_bkv` is the only thing moving.  The one
long-context point (16384) comes from the generic `prefill_q1024_ctx*` family,
where `max_model_len` grows with ctx; it is marked `mml_pinned=False` because
more than `num_bkv` moves there and it is not part of the controlled curve.

DECODE
------
Decode uses `num_queries_per_block[0] = 1` and `static_q_len = 1`, so
`bq_sz = min(1, 1) = 1`.  At `bq_sz == 1` the head-major permutation is the
IDENTITY: head-minor row index is `t*H + h = h`, head-major is `h*bq_sz + t
= h`.  So decode cannot gain and should not lose; what is actually being
measured is whether the compiler makes the degenerate transpose free.  Run
because "should" is not "does".

METHOD
------
Identical to `bench_headmajor.py`: chain slope of R calls linked through the
kernel's own `input_output_aliases`.  Not a wall clock.  Run UNPINNED.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROBES = _HERE.parent.parent / "probes"
for _p in (str(_HERE), str(_PROBES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import bench_headmajor as B  # noqa: E402
import stage_a_c as S  # noqa: E402
from _probe_util import emit_json, p, tpu_banner  # noqa: E402

# ctx -> the num_bkv it produces at bkv_sz=2048.  <=9216 keeps mml pinned.
PREFILL_CTX = (2048, 4096, 6144, 8192, 9216, 16384)
MML_PINNED_UPTO = 9216

DECODE = dict(
    num_seqs=64,
    q_len=1,
    context_len=8704,
    max_model_len=9216,
    page_size=1024,
    kv_packing=4,
    lkv_dim=256,
    h_idx=32,
    d_idx=128,
    k=2048,
    compression_ratio=1,
    bkv_p=2,
    bq_per_block=1,  # num_queries_per_block[0]
    decode_req_batch_size=4,  # streamindex_topk.py:536 default
)


def decode_geometry(num_seqs=None, context_len=None):
    g = dict(DECODE)
    if num_seqs is not None:
        g["num_seqs"] = num_seqs
    if context_len is not None:
        g["context_len"] = context_len
        g["max_model_len"] = max(g["max_model_len"], context_len)
    g["pages_per_seq"] = S.cdiv(g["max_model_len"], g["page_size"])
    g["num_sublanes_total"] = (S.align_to(g["pages_per_seq"], g["bkv_p"]) *
                               g["page_size"] // 128)
    g["num_positions"] = g["num_sublanes_total"] * 128
    g["num_tokens"] = g["num_seqs"] * g["q_len"]
    g["total_pages"] = g["num_seqs"] * g["pages_per_seq"]
    g["bq_sz"] = min(g["bq_per_block"], g["q_len"])
    g["bkv_sz"] = g["bkv_p"] * g["page_size"]
    g["num_bq"] = max(1, S.cdiv(g["q_len"], g["bq_sz"]))
    g["num_bkv"] = max(
        1, S.cdiv(g["context_len"] // g["compression_ratio"], g["bkv_sz"]))
    return g


def decode_stage_a(arm, g):
    """The DECODE stage-A call, transcribed from `streamindex_topk.py:755-771`."""
    return S.make_stage_a_fn(
        arm,
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        max_num_tokens=g["num_tokens"],
        num_kv_pages_per_block=g["bkv_p"],
        num_queries_per_block=g["bq_per_block"],
        static_q_len=1,
        seq_batch_size=g["decode_req_batch_size"],
        case=S.MlaCase.DECODE,
        compression_ratio=g["compression_ratio"],
    )


def _run_point(label, g, arms, trials, launcher):
    """One (shape, arms) point.  Returns a dict; a dead arm is a result."""
    p("=" * 70)
    p(label, {
        k: g[k]
        for k in ("num_seqs", "q_len", "context_len", "max_model_len", "bq_sz",
                  "bkv_sz", "num_bq", "num_bkv", "num_tokens",
                  "num_sublanes_total")
    })
    raw = S.build_prefill_inputs(g)
    for k, v in list(raw.items()):
        raw[k] = jax.block_until_ready(v)

    # `bench_headmajor.build_arm` hardcodes the MIXED launcher; for decode we
    # swap it out.  Everything else -- the chain, the fit, the statistics --
    # is reused verbatim so the two phases are measured the same way.
    orig = S.prefill_stage_a
    S.prefill_stage_a = launcher
    try:
        recs, outs = [], {}
        for arm in arms:
            try:
                rec, out = B.run_arm(arm, g, raw, trials, do_chain=True)
                recs.append(rec)
                outs[arm] = out
            except Exception as e:  # noqa: BLE001 - a failed arm is a result
                import traceback
                traceback.print_exc()
                recs.append({"arm": arm, "error": repr(e)[:1500]})
                p(label, "arm", arm, "FAILED", repr(e)[:300])
    finally:
        S.prefill_stage_a = orig

    by = {r["arm"]: r for r in recs if "chain_slope_us" in r}
    delta = None
    if "incumbent" in by and "head_major" in by:
        delta = (by["head_major"]["chain_slope_us"] /
                 by["incumbent"]["chain_slope_us"] - 1.0) * 100.0
        p(label, f"head_major vs incumbent: {delta:+.2f}%")

    # numeric agreement, so a "fast" arm cannot be a wrong arm
    numeric = {}
    if "incumbent" in outs:
        ref = outs["incumbent"]
        fin = jnp.isfinite(ref)
        for arm, out in outs.items():
            d = jnp.where(fin, jnp.abs(out - ref), 0.0)
            numeric[arm] = dict(
                same_mask=bool(jnp.all(jnp.isfinite(out) == fin)),
                max_abs=float(jnp.max(d)),
            )
        p(label, "numeric vs incumbent", numeric)

    return dict(label=label,
                geometry={k: g[k] for k in sorted(g)},
                num_bkv=g["num_bkv"],
                num_bq=g["num_bq"],
                bq_sz=g["bq_sz"],
                arms=recs,
                head_major_vs_incumbent_pct=delta,
                numeric_vs_incumbent=numeric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="incumbent,head_major")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--prefill-ctx",
                    default=",".join(str(c) for c in PREFILL_CTX))
    ap.add_argument("--decode", action="store_true", default=True)
    ap.add_argument("--no-decode", dest="decode", action="store_false")
    ap.add_argument("--decode-ctx", default="8704")
    args = ap.parse_args()

    tpu_banner()
    p("incumbent module", S.sx.__file__, S.source_md5(S.sx))
    p("variant module", S.sxc.__file__, S.source_md5(S.sxc))
    arms = [a for a in args.arms.split(",") if a]

    points = []
    for ctx in (int(c) for c in args.prefill_ctx.split(",") if c):
        g = S.prefill_geometry(None, ctx)
        pinned = ctx <= MML_PINNED_UPTO
        rec = _run_point(f"prefill_q1024_ctx{ctx}", g, arms, args.trials,
                         S.prefill_stage_a)
        rec["phase"] = "prefill"
        rec["mml_pinned"] = pinned
        points.append(rec)

    if args.decode:
        for ctx in (int(c) for c in args.decode_ctx.split(",") if c):
            g = decode_geometry(None, ctx)
            rec = _run_point(f"decode_b{g['num_seqs']}_ctx{ctx}", g, arms,
                             args.trials, decode_stage_a)
            rec["phase"] = "decode"
            rec["mml_pinned"] = True
            points.append(rec)

    print("\n===== OPTION C vs num_bkv (chain slope, device us/call) =====")
    print(f"{'shape':28} {'ph':<8} {'nbkv':>5} {'nbq':>4} {'bq':>4} "
          f"{'incumbent':>11} {'head_major':>11} {'delta%':>8} {'min r2':>10}")
    for rec in points:
        by = {r["arm"]: r for r in rec["arms"] if "chain_slope_us" in r}
        inc = by.get("incumbent", {}).get("chain_slope_us")
        hm = by.get("head_major", {}).get("chain_slope_us")
        r2s = [r["chain_r2"] for r in by.values()]
        d = rec["head_major_vs_incumbent_pct"]
        c_inc = f"{inc:11.2f}" if inc is not None else f"{'FAILED':>11}"
        c_hm = f"{hm:11.2f}" if hm is not None else f"{'FAILED':>11}"
        c_d = f"{d:+8.2f}" if d is not None else f"{'-':>8}"
        c_r2 = f"{min(r2s):10.6f}" if r2s else f"{'-':>10}"
        print(f"{rec['label']:28} {rec['phase']:<8} {rec['num_bkv']:5d} "
              f"{rec['num_bq']:4d} {rec['bq_sz']:4d} "
              f"{c_inc} {c_hm} {c_d} {c_r2}")
    print("\nprefill ctx<=9216 hold max_model_len=9216 (num_sublanes_total "
          "constant); ctx 16384/32768 do not -- more than num_bkv moves there.")
    print("decode bq_sz=1 makes the head-major permutation the identity; any "
          "delta there is compiler noise or degenerate-transpose overhead.")

    emit_json(
        "option_c_numbkv_sweep",
        dict(what="Option C stage-A chain slope vs num_bkv, prefill + decode",
             method="chain slope, R=%s, best-of-%d, unpinned" %
             (list(B.CHAIN_R), args.trials),
             bkv_sz=2048,
             points=points,
             incumbent_md5=S.source_md5(S.sx),
             variant_md5=S.source_md5(S.sxc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
