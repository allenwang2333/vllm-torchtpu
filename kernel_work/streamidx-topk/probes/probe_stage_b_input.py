#!/usr/bin/env python3
"""Why the isolated stage-B arm is cheaper than stage B inside the op.

The corrected baseline (`probes/baseline-device-time.md`, job `j-79a410b8`)
found chain slope and `hlo_stats` agreeing to within 2 % on the full op and on
stage A, at both production shapes -- and disagreeing on stage B:

                     chain slope (isolated)   hlo_stats (in situ)
    decode                       28.77 us              31.73 us   +10.3 %
    prefill                     336.48 us             431.15 us   +28.1 %

Both numbers are reproducible: 336.48 here against 335.53 in
`probes/stage-b-ksweep.md` §1, and the `hlo_stats` decode arm reproduces R2's
31.807 us.  So this is not noise and it is not a broken method -- the two are
measuring different work, and the deliverable has to say which one is stage B.

The hypothesis this probe tests is the **score distribution**.  The isolated
arm feeds `bench_streamidx_topk.build_stage_b_scores`, which is N(0, 1) with
`-inf` past each row's visible length; its own docstring warns that "the
SparseCore radix select is data dependent" and that the distribution is held
fixed for comparability, not because it is what the indexer emits.  In the op,
stage B consumes stage A's actual output.

Two competing explanations are ruled out before the run, by reading the
baseline job's own JSON rather than by argument:

  * **row lengths** -- the isolated arm reports `eff_len_mean = 8704.5` at
    prefill, which is exactly the mean of `_effective_row_lengths` over
    `8193..9216`, and 8704.0 at decode.  They already match.
  * **shape / partition** -- both run `sc_topk_direct_b1024_n10240_k2048`
    (resp. `_b64_`), the same kernel with the same `_pick_partition` result.
    The `hlo_stats` row name is identical in the two settings.

So: same kernel, same shape, same row lengths, different data.  The arms:

  ``synthetic``  `build_stage_b_scores`, N(0, 1). The control. It has to come
                 back at 336.5 / 28.8 or this probe's own pipeline moved.
  ``real``       stage A's actual output for the same inputs, reshaped exactly
                 as `_common_path` reshapes it, with `_effective_row_lengths`
                 computed by the op's own function. If this comes back near
                 the `hlo_stats` figure, the disagreement is the input and not
                 the method, and the in-situ number is the one that describes
                 production.

Both arms are chain slopes, so the *method* is held fixed and the input is the
only thing that varies.  That is the whole design: a within-method comparison
cannot be explained by the method.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
INSTRUMENTS = os.environ.get("BDT_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

TRIALS, WARMUP, TARGET_S = 9, 3, 0.05
CHAIN_R = (1, 2, 3, 4)
RESULTS: dict = {}


def fit(pts):
    xs = np.array(sorted(pts), float)
    ys = np.array([pts[int(x)] for x in xs], float)
    a = np.vstack([np.ones_like(xs), xs]).T
    (b0, b1), *_ = np.linalg.lstsq(a, ys, rcond=None)
    pred = b0 + b1 * xs
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {"intercept_us": float(b0), "slope_us_per_call": float(b1),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "points_us": {int(x): float(y) for x, y in zip(xs, ys)}}


def main():
    tpu_banner()
    import bench_streamidx_topk as B
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
        sparsecore_topk
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        MlaCase, _effective_row_lengths, prepare_index_weights,
        prepare_q_inputs)
    import _stage_a as SA

    mods = {}
    for m in (sxmod, scmod, B, SA):
        h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
        mods[os.path.basename(m.__file__)] = h
        p("MODULE", f"{m.__file__} md5={h}")
    RESULTS["modules"] = mods

    shapes = {s.name: s for s in B.SHAPES}
    out = RESULTS.setdefault("arms", {})

    for phase, sname in (("decode", "prod_decode_b64_ctx8704"),
                         ("prefill", "prod_prefill_q1024_ctx9216")):
        shape = shapes[sname]
        geo = B.geometry(shape)
        inputs = B.build_inputs(shape, geo, seed=0)

        # ---- the op's own stage-A output, produced by the op's own launcher
        if phase == "decode":
            fn = SA.make_stage_a_fn(
                num_q_heads=B.H_IDX, head_dim=B.D_IDX,
                page_size_per_kv_packing=B.PAGE_SIZE // B.KV_PACKING,
                kv_packing=B.KV_PACKING, lkv_dim=B.LKV_DIM,
                num_sublanes_total=geo.num_sublanes_total,
                num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK[0],
                num_queries_per_block=B.NUM_QUERIES_PER_BLOCK[0],
                static_q_len=1, seq_batch_size=4, case=MlaCase.DECODE,
                compression_ratio=B.COMPRESSION_RATIO)
            start, end = 0, geo.num_tokens
        else:
            fn = SA.make_stage_a_fn(
                num_q_heads=B.H_IDX, head_dim=B.D_IDX,
                page_size_per_kv_packing=B.PAGE_SIZE // B.KV_PACKING,
                kv_packing=B.KV_PACKING, lkv_dim=B.LKV_DIM,
                num_sublanes_total=geo.num_sublanes_total,
                num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK[2],
                num_queries_per_block=B.NUM_QUERIES_PER_BLOCK[2],
                static_q_len=None, seq_batch_size=1, case=MlaCase.MIXED,
                compression_ratio=B.COMPRESSION_RATIO)
            start, end = 0, 1

        real = jax.jit(lambda q, w, ckv, sl, pi, cql: fn(
            q, w, ckv,
            jnp.full((geo.num_tokens, geo.num_sublanes_total, 128),
                     -jnp.inf, jnp.float32),
            sl, pi, cql, jnp.int32(start), jnp.int32(end)
        ).reshape(geo.num_tokens, -1))(
            prepare_q_inputs(inputs["q"]),
            prepare_index_weights(inputs["indexer_weights"],
                                  inputs["q"].dtype),
            inputs["cache_kv"], inputs["seq_lens"], inputs["page_indices"],
            inputs["cu_q_lens"])
        real = jax.block_until_ready(real)

        rl_real = jax.block_until_ready(_effective_row_lengths(
            seq_lens=inputs["seq_lens"], cu_q_lens=inputs["cu_q_lens"],
            distribution=inputs["distribution"], num_tokens=geo.num_tokens,
            num_positions=real.shape[1],
            compression_ratio=B.COMPRESSION_RATIO))

        syn, rl_syn = B.build_stage_b_scores(shape, geo, inputs, seed=1)
        syn = jax.block_until_ready(syn)

        # A distribution summary, so the reader can see *what* differs rather
        # than only that something does. Only finite entries count: the padded
        # tail is -inf in both arms by construction.
        def summarise(a):
            v = np.asarray(a).ravel()
            f = v[np.isfinite(v)]
            return {"n_finite": int(f.size), "n_neg_inf": int(v.size - f.size),
                    "mean": float(f.mean()), "std": float(f.std()),
                    "min": float(f.min()), "max": float(f.max()),
                    "n_distinct_sample": int(np.unique(f[:200000]).size)}

        for arm, sc, rl in (("synthetic", syn, rl_syn), ("real", real, rl_real)):
            rec = {"phase": phase, "shape": sname,
                   "scores_shape": list(sc.shape), "dtype": str(sc.dtype),
                   "row_length_mean": float(np.asarray(rl).mean()),
                   "row_length_min": int(np.asarray(rl).min()),
                   "row_length_max": int(np.asarray(rl).max()),
                   "p_partition": int(scmod._pick_partition(
                       geo.num_tokens, sc.shape[1], B.K_IDX)),
                   "score_distribution": summarise(sc)}
            try:
                def one(s, r_):
                    return sparsecore_topk(s, B.K_IDX, row_lengths=r_,
                                           write_empty_rows=True)

                def build(r):
                    def f(s, r_):
                        cur, o = r_, None
                        for _ in range(r):
                            o = one(s, cur)
                            cur = r_ + jnp.minimum(o[0, 0], jnp.int32(0))
                        return o
                    return jax.jit(f)

                pts = {}
                for r in CHAIN_R:
                    t = time_arm(build(r), (sc, rl), WARMUP, TRIALS, TARGET_S)
                    pts[r] = t["best_s"] * 1e6
                rec.update(fit(pts))
                t1 = time_arm(jax.jit(one), (sc, rl), WARMUP, TRIALS, TARGET_S)
                rec["batched_best_us"] = t1["best_s"] * 1e6
                rec["batched_iters"] = t1["iters"]
            except Exception as e:
                rec["error"] = repr(e)
                traceback.print_exc()
            out[f"{phase}_{arm}"] = rec
            p(f"{phase}/{arm}",
              f"slope={rec.get('slope_us_per_call', float('nan')):.2f}us "
              f"r2={rec.get('r2', float('nan')):.6f} "
              f"rowlen[{rec['row_length_min']},{rec['row_length_max']}] "
              f"dist={rec['score_distribution']}")
            emit_json("stage-b-input", RESULTS)

        # Do the two arms even agree on the answer? They should not -- the
        # scores differ -- but the *number of valid indices* must match, which
        # is what makes them the same amount of selection work.
        del real, syn

    emit_json("stage-b-input", RESULTS)
    a = out.get("prefill_synthetic", {}).get("slope_us_per_call")
    b = out.get("prefill_real", {}).get("slope_us_per_call")
    if a and b:
        p("VERDICT", f"prefill synthetic={a:.2f}us real={b:.2f}us "
                     f"ratio={b / a:.3f}; hlo_stats in situ = 431.15us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
