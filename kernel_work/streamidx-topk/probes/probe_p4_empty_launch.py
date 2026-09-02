#!/usr/bin/env python3
"""P4 -- what does an empty stage-A `pallas_call` cost at the nightly width?

Two of stage A's three `pallas_call`s launch with a **dynamically zero-length
grid at every pure-decode step**, because `mla_attention_op.py:183` builds
`distribution = torch.stack([rd[0], rd[0], rd[2]])`, so `distribution[0] ==
distribution[1]` always and the middle "chunked-prefill-only" band is empty by
construction.  Measured at the AgentX operating point (`j-a71a648c`, hlo_stats)
those two empty launches cost 5.40 us and 5.22 us against a 114.63 us running
call, while carrying a **16.8 MB** output allocation -- which argues the fixed
cost is invocation-driven rather than allocation-driven.  The nightly output is
**2.62 MB**; this probe tests whether ~5.3 us survives the 6.4x smaller buffer.

It decides between two design tiers:
  tier 1 -- merge decode calls 1 and 2 into one batched kernel (recovers the
            empty launch, no scratch growth);
  tier 2 -- `lax.cond` the empty one away (cheaper to write, but adds 21 more
            `conditional`s per step to a bucket already measured at 4.6% of the
            step).

Method -- and this is the part that matters
-------------------------------------------
A single empty launch is a few microseconds, and `report.md` §6.2 measured the
**host dispatch floor at 34.6-44.9 us per call**.  Timing one launch from the
host therefore measures the host, not the launch.  So every arm here times a
**chain of M launches inside one jitted program**: each link takes the previous
link's score buffer as its `scores_init`, exactly as `_common_path` chains its
three calls, so there is a real data dependency and XLA can neither CSE nor
reorder them.  The reported per-launch price is the **slope**,
`(t(M=9) - t(M=1)) / 8`, which contains no host dispatch and no `-inf` init.

`start_seq_idx` / `end_seq_idx` stay traced device scalars, as `_common_path`
passes them, so grid extents 0 / 1 / 16 are three *calls into one executable*
rather than three compilations -- the same property production relies on.

Run:  python probe_p4_empty_launch.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import _stage_a  # noqa: E402
from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

CHAIN_M = (1, 9)  # slope over 8 marginal launches


def make_chain(stage_a_fn, g, m: int, *, guarded: bool):
    """A jitted program containing `m` sequential stage-A launches."""

    def body(q, w, cache_kv, seq_lens, page_indices, cu_q_lens, start, end,
             pred):
        scores = jnp.full(
            (g["num_tokens"], g["num_sublanes_total"], 128), -jnp.inf,
            jnp.float32)
        for _ in range(m):
            if guarded:
                scores = lax.cond(
                    pred,
                    lambda s: stage_a_fn(q, w, cache_kv, s, seq_lens,
                                         page_indices, cu_q_lens, start, end),
                    lambda s: s,
                    scores,
                )
            else:
                scores = stage_a_fn(q, w, cache_kv, scores, seq_lens,
                                    page_indices, cu_q_lens, start, end)
        return scores

    return jax.jit(body)


def main():
    tpu_banner()
    g, inp = _stage_a.build_nightly_inputs()
    p("stage_a_source", _stage_a.source_path())
    p("stage_a_source_md5", _stage_a.source_md5())
    p("geometry", {k: v for k, v in g.items()})
    p("score_buffer_MB", f"{g['score_bytes']/1e6:.3f}")
    p("cache_kv_MB", f"{inp['cache_kv'].size/1e6:.1f}")

    call1 = _stage_a.nightly_decode_call1(g)  # seq_batch_size = 4
    call2 = _stage_a.nightly_decode_call2(g)  # seq_batch_size = 1

    base = (inp["q"], inp["weights"], inp["cache_kv"], inp["seq_lens"],
            inp["page_indices"], inp["cu_q_lens"])

    def scalars(n_seqs):
        return (jnp.int32(0), jnp.int32(n_seqs))

    # (arm name, launcher, guarded, seq_batch_size, grid extent -> n_seqs)
    configs = [
        ("sbs1", call2, False, 1),
        ("sbs4", call1, False, 4),
        ("sbs1_cond", call2, True, 1),
    ]
    # grid extents to exercise, per config
    grids = {
        "sbs1": (0, 1, 16),
        "sbs4": (0, 1, 16),
        "sbs1_cond": (0, 16),
    }
    # the guarded arm is also called with pred=False, which is the design
    # option being priced (skip the launch entirely)
    preds = {
        "sbs1": (True, ),
        "sbs4": (True, ),
        "sbs1_cond": (True, False),
    }

    results = {
        "geometry": g,
        "score_buffer_bytes": g["score_bytes"],
        "stage_a_source_md5": _stage_a.source_md5(),
        "chain_M": list(CHAIN_M),
        "arms": {},
        "controls": {},
    }

    # ---- controls -------------------------------------------------------
    try:
        init_only = jax.jit(lambda: jnp.full(
            (g["num_tokens"], g["num_sublanes_total"], 128), -jnp.inf,
            jnp.float32))
        t = time_arm(init_only, (), warmup=3, trials=7, target_s=0.05)
        results["controls"]["init_only_us"] = t["best_s"] * 1e6
        p("control init_only(-inf broadcast, 2.62MB)",
          f"best={t['best_s']*1e6:.2f}us")
    except Exception as e:
        p("control init_only", "FAILED", repr(e))
        results["controls"]["init_only_us"] = f"FAILED {e!r}"

    try:
        floor = jax.jit(lambda x: x + 1.0)
        t = time_arm(floor, (jnp.asarray(1.0, jnp.float32), ), warmup=3,
                     trials=7, target_s=0.05)
        results["controls"]["dispatch_floor_us"] = t["best_s"] * 1e6
        p("control dispatch_floor(1-arg jit)", f"best={t['best_s']*1e6:.2f}us")
    except Exception as e:
        p("control dispatch_floor", "FAILED", repr(e))

    # ---- chains ---------------------------------------------------------
    for name, fn, guarded, sbs in configs:
        chains = {}
        for m in CHAIN_M:
            try:
                chains[m] = make_chain(fn, g, m, guarded=guarded)
            except Exception as e:
                p(f"{name} M={m}", "COMPILE-BUILD FAILED", repr(e))
                traceback.print_exc()
        for pred in preds[name]:
            for grid in grids[name]:
                n_seqs = grid * sbs
                key = f"{name}_grid{grid}" + ("" if not guarded else
                                              f"_pred{int(pred)}")
                row = {
                    "seq_batch_size": sbs,
                    "grid_extent": grid,
                    "n_seqs": n_seqs,
                    "guarded": guarded,
                    "pred": bool(pred),
                    "M": {},
                }
                for m, cfn in chains.items():
                    args = base + scalars(n_seqs) + (jnp.bool_(pred), )
                    try:
                        t = time_arm(cfn, args, warmup=3, trials=7,
                                     target_s=0.05)
                        row["M"][m] = {
                            "best_us": t["best_s"] * 1e6,
                            "median_us": t["median_s"] * 1e6,
                            "iters": t["iters"],
                            "compile_s": t["compile_s"],
                        }
                        p(f"{key} M={m}",
                          f"best={t['best_s']*1e6:.2f}us "
                          f"median={t['median_s']*1e6:.2f}us "
                          f"iters={t['iters']} compile={t['compile_s']:.1f}s")
                    except Exception as e:
                        row["M"][m] = {"error": repr(e)}
                        p(f"{key} M={m}", "FAILED", repr(e))
                        traceback.print_exc()
                if all(isinstance(row["M"].get(m), dict)
                       and "best_us" in row["M"][m] for m in CHAIN_M):
                    lo, hi = CHAIN_M[0], CHAIN_M[-1]
                    slope = ((row["M"][hi]["best_us"] -
                              row["M"][lo]["best_us"]) / (hi - lo))
                    row["per_launch_us"] = slope
                    p(f"{key} PER-LAUNCH", f"{slope:.2f}us "
                      f"(slope over M={lo}..{hi})")
                results["arms"][key] = row

    # ---- the two numbers the design decision turns on -------------------
    a = results["arms"]
    empty = a.get("sbs1_grid0", {}).get("per_launch_us")
    running = a.get("sbs1_grid16", {}).get("per_launch_us")
    cond_skip = a.get("sbs1_cond_grid0_pred0", {}).get("per_launch_us")
    summary = {
        "empty_launch_us_sbs1": empty,
        "running_launch_us_sbs1_grid16": running,
        "cond_skipped_launch_us": cond_skip,
        "agentx_reference_empty_us": [5.40, 5.22],
        "agentx_reference_output_MB": 16.8,
        "nightly_output_MB": g["score_bytes"] / 1e6,
    }
    if empty is not None and cond_skip is not None:
        summary["tier2_saving_per_launch_us"] = empty - cond_skip
    results["summary"] = summary
    p("SUMMARY", summary)

    emit_json("P4-empty-launch", results)


if __name__ == "__main__":
    main()
