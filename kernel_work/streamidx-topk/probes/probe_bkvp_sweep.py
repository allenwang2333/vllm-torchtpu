"""Step-2 probe P7a: sweep `bkv_p` (num_kv_pages_per_block[decode]) at the
nightly decode shape.

`bkv_p` is the number of KV pages prefetched per DMA slot in stage A's
hand-rolled fetch (`streamindex_topk.py:160-198`). The incumbent runs
`bkv_p=2` (`mla_attention_op.py:119`, `num_kv_pages_per_block=(2, 2, 2)`).

This arm does double duty:

  * area 2's top candidate -- a one-tuple change, estimated 2.5-3%;
  * area 3's cheap discriminator for whether stage A is DMA-bound. Flat across
    the sweep => the fetch is not the constraint. Moving => it is.

CONFOUND, and it must be read alongside every number below. The kernel derives

    num_sublanes_total = max over the 3-tuple of
                         align_to(pages_per_seq, bkv_p) * page_size // 128
    (streamindex_topk.py:603-605)

so `num_positions` -- the score matrix's row width, and therefore stage B's and
the init's whole workload -- is a function of the tuple. At the nightly decode
shape `pages_per_seq = 9`, and holding the prefill/mixed entries at 2:

    bkv_p=1 -> max(72, 80, 80) = 80 sublanes -> num_positions 10240
    bkv_p=2 -> max(80, 80, 80) = 80          -> 10240
    bkv_p=3 -> max(72, 80, 80) = 80          -> 10240
    bkv_p=4 -> max(96, 80, 80) = 96          -> 12288   <-- +20% score matrix

So bkv_p in {1, 2, 3} is a clean like-for-like: identical score matrix,
identical stage B, identical init, and every difference in `total` is stage A.
bkv_p=4 is NOT like-for-like and its total carries 20% more stage-B and init
work. Both readings are printed; the sweep's conclusion is drawn from {1,2,3}
and bkv_p=4 is reported with its confound named.

Reuses the read-only instrument `instruments/bench_streamidx_topk.py` for
Shape, geometry, build_inputs and time_arm, so the numbers are comparable with
`report.md` §6. The instrument is imported, never modified.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_INSTR = os.path.join(_HERE, os.pardir, "instruments",
                      "bench_streamidx_topk.py")


def _load_instrument():
    spec = importlib.util.spec_from_file_location("bench_streamidx_topk",
                                                  _INSTR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_streamidx_topk"] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load_instrument()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    sparsecore_topk  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402

BKV_P_SWEEP = (1, 2, 3, 4)
INCUMBENT_BKV_P = 2


def _p(tag, *rest):
    print(f"[P7a] {tag}:", *rest, flush=True)


def sublanes_for(pages_per_seq, tup):
    return max(B.align_to(pages_per_seq, p) * B.PAGE_SIZE // 128 for p in tup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="prod_decode_b64_ctx8704")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--target-s", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    dev = jax.devices()[0]
    _p("device", dev.device_kind, "jax", jax.__version__)

    shapes = {s.name: s for s in B.SHAPES}
    if args.shape not in shapes:
        _p("FATAL", f"no such shape {args.shape}; have "
           f"{sorted(k for k in shapes if k.startswith('prod'))}")
        return 2
    shape = shapes[args.shape]
    geo = B.geometry(shape)

    _p("shape", shape.name, f"phase={shape.phase}",
       f"num_seqs={shape.num_seqs}", f"q_len={shape.q_len}",
       f"ctx={shape.context_len}", f"max_model_len={shape.max_model_len}",
       f"pages_per_seq={geo.pages_per_seq}",
       f"incumbent num_positions={geo.num_positions}")
    _p("note", "the incumbent tuple is num_kv_pages_per_block=(2,2,2), "
       "num_queries_per_block=(1,128,128), vmem_limit_bytes=100MiB (the "
       "kernel default; mla_attention_op.py does not override it)")

    # Inputs are built ONCE and shared across every arm: the cache, the block
    # table, q and the weights are all independent of bkv_p, so reusing them
    # removes the PRNG as a variable between arms and keeps HBM pressure equal.
    inputs = B.build_inputs(shape, geo, seed=args.seed)
    _p("inputs", {k: (tuple(v.shape), str(v.dtype))
                  for k, v in inputs.items()})

    rows = []
    for bkv_p in BKV_P_SWEEP:
        tup = (bkv_p, B.NUM_KV_PAGES_PER_BLOCK[1], B.NUM_KV_PAGES_PER_BLOCK[2])
        nsub = sublanes_for(geo.pages_per_seq, tup)
        npos = nsub * 128
        row = {
            "bkv_p": bkv_p,
            "tuple": tup,
            "num_sublanes_total": nsub,
            "num_positions": npos,
            "bkv_sz": bkv_p * B.PAGE_SIZE,
            "like_for_like": npos == geo.num_positions,
        }

        def total_fn(_tup=tup, **kw):
            return streamindex_topk(
                **kw,
                k=B.K_IDX,
                compression_ratio=B.COMPRESSION_RATIO,
                num_kv_pages_per_block=_tup,
                num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
                enable_early_exit=shape.enable_early_exit,
            )

        try:
            t_total, iters, compile_s = B.time_arm(
                lambda: total_fn(**inputs), (), args.warmup, args.trials,
                args.target_s)
        except Exception as e:  # noqa: BLE001 - a tuple that will not lower is a result
            _p(f"bkv_p={bkv_p}", "FAIL", repr(e)[:400])
            row["status"] = "failed"
            row["error"] = repr(e)[:400]
            rows.append(row)
            continue

        row.update(status="ok", total_us=t_total * 1e6, iters=iters,
                   compile_s=compile_s)
        _p(f"bkv_p={bkv_p}",
           f"bkv_sz={row['bkv_sz']:>5}", f"num_positions={npos:>6}",
           f"like_for_like={row['like_for_like']}",
           f"total={t_total*1e6:8.2f}us", f"(iters={iters})")
        rows.append(row)

    # Stage B and the init, measured once per DISTINCT num_positions, so the
    # derived stage A is available for every arm including the confounded one.
    seen = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        npos = row["num_positions"]
        if npos not in seen:
            nsub = row["num_sublanes_total"]
            scores = jax.random.normal(jax.random.key(args.seed + 1),
                                       (geo.num_tokens, npos), jnp.float32)
            row_lengths = B._row_lengths_ref(inputs["seq_lens"],
                                             inputs["cu_q_lens"],
                                             shape.num_seqs, geo.num_tokens,
                                             npos, B.COMPRESSION_RATIO)
            col = jnp.arange(npos, dtype=jnp.int32)
            scores = jnp.where(col[None, :] < row_lengths[:, None], scores,
                               -jnp.inf)
            stage_b_fn = jax.jit(lambda s, rl: sparsecore_topk(
                s, B.K_IDX, row_lengths=rl, write_empty_rows=True))
            t_b, _, _ = B.time_arm(stage_b_fn, (scores, row_lengths),
                                   args.warmup, args.trials, args.target_s)
            scores.delete()

            init_shape = (geo.num_tokens, nsub, 128)
            init_fn = jax.jit(lambda v: jnp.full(init_shape, v, jnp.float32))
            neg_inf = jnp.asarray(-jnp.inf, jnp.float32)
            t_i, _, _ = B.time_arm(init_fn, (neg_inf, ), args.warmup,
                                   args.trials, args.target_s)
            seen[npos] = (t_b * 1e6, t_i * 1e6)
            _p(f"num_positions={npos}", f"stageB={t_b*1e6:8.2f}us",
               f"init={t_i*1e6:8.2f}us")
        row["stage_b_us"], row["init_us"] = seen[npos]
        row["stage_a_derived_us"] = (row["total_us"] - row["stage_b_us"] -
                                     row["init_us"])

    # ---------------- report ----------------
    _p("---", "results")
    hdr = (f"{'bkv_p':>5} {'bkv_sz':>7} {'n_pos':>6} {'l4l':>4} "
           f"{'total_us':>9} {'stageB_us':>9} {'init_us':>8} "
           f"{'stageA_us':>9} {'vs bkv_p=2':>11}")
    print(hdr, flush=True)
    base = next((r for r in rows
                 if r["bkv_p"] == INCUMBENT_BKV_P and r.get("status") == "ok"),
                None)
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['bkv_p']:>5} {'':>7} {r['num_positions']:>6} "
                  f"{'':>4} {'FAILED':>9}", flush=True)
            continue
        rel = ("" if base is None else
               f"{100*(r['total_us']/base['total_us']-1):+10.2f}%")
        print(f"{r['bkv_p']:>5} {r['bkv_sz']:>7} {r['num_positions']:>6} "
              f"{str(r['like_for_like'])[:1]:>4} {r['total_us']:>9.2f} "
              f"{r['stage_b_us']:>9.2f} {r['init_us']:>8.2f} "
              f"{r['stage_a_derived_us']:>9.2f} {rel:>11}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok" and r["like_for_like"]]
    if len(ok) >= 2 and base is not None:
        tot = [r["total_us"] for r in ok]
        sa = [r["stage_a_derived_us"] for r in ok]
        spread_tot = (max(tot) - min(tot)) / base["total_us"]
        spread_sa = ((max(sa) - min(sa)) / base["stage_a_derived_us"]
                     if base["stage_a_derived_us"] else float("nan"))
        _p("like_for_like spread (bkv_p in {1,2,3}, identical num_positions)",
           f"total {100*spread_tot:.2f}%  stage A (derived) "
           f"{100*spread_sa:.2f}%")
        # The DMA-bound reading. Threshold is deliberately loose: report.md
        # §6 measures 8.5% run-to-run spread on the TOTAL at a fixed config,
        # so anything under ~10% on the total is inside the noise this
        # instrument is known to carry.
        if spread_tot < 0.03:
            _p("P7a_DMA_VERDICT",
               f"FLAT - total moves {100*spread_tot:.2f}% across bkv_p "
               "{1,2,3}. Stage A's cost is not set by the prefetch depth, so "
               "the hand-rolled fetch is not the binding constraint at this "
               "shape, and the one-tuple change is not worth taking.")
        elif spread_tot < 0.10:
            _p("P7a_DMA_VERDICT",
               f"WEAK - total moves {100*spread_tot:.2f}%, which is inside "
               "the 8.5% run-to-run spread report.md §6 measured for this "
               "instrument. Not separable from noise in a single job.")
        else:
            _p("P7a_DMA_VERDICT",
               f"MOVES - total moves {100*spread_tot:.2f}% across bkv_p "
               "{1,2,3}, above the 8.5% known run-to-run spread. The fetch "
               "depth is a real term, so stage A is at least partly DMA-bound.")
        best = min(ok, key=lambda r: r["total_us"])
        _p("P7a_BEST_LIKE_FOR_LIKE",
           f"bkv_p={best['bkv_p']} at {best['total_us']:.2f}us, "
           f"{100*(best['total_us']/base['total_us']-1):+.2f}% vs the "
           f"incumbent bkv_p=2")

    b4 = next((r for r in rows if r["bkv_p"] == 4 and r.get("status") == "ok"),
              None)
    if b4 and base:
        _p("bkv_p=4 CONFOUNDED",
           f"num_positions {b4['num_positions']} vs {base['num_positions']} "
           f"(+{100*(b4['num_positions']/base['num_positions']-1):.0f}%). Its "
           f"total {b4['total_us']:.2f}us carries a larger stage B "
           f"({b4['stage_b_us']:.2f} vs {base['stage_b_us']:.2f}us) and init "
           f"({b4['init_us']:.2f} vs {base['init_us']:.2f}us). Compare only "
           f"its derived stage A: {b4['stage_a_derived_us']:.2f} vs "
           f"{base['stage_a_derived_us']:.2f}us.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"shape": args.shape, "rows": rows,
                       "device": dev.device_kind}, f, indent=2)
        _p("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
