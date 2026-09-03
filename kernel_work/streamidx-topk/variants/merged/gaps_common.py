#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configs, arms and plans for the three gaps a `/critical-review` left open.

`merged_common.py` is imported, not copied: the grid, `Cfg`, `to_shape`, the
`bkv_p` ladder and the like-for-like check are all reused verbatim so a number
here is comparable to a number in `RESULTS.md`.  What this file adds is three
things `merged_common.py` cannot express.

E1 -- a WITHIN-JOB replicate at the production config
-----------------------------------------------------

The sweep has one compilation and one 20-iteration measurement per
(config, arm), so the only error bar in hand is the *between-job* 1.01% quoted
in `RESULTS.md` -- taken across two jobs that ran different module versions,
which is not the error quantity a within-job composition residual should be
compared against.  `PLANS["e1"]` re-enters the whole measurement loop (a fresh
`jax.profiler.trace` block, a fresh 20 iterations) `reps` times per arm, with
the reps interleaved across arms so a drift over the life of the job hits
every arm rather than piling onto whichever ran last.

E2 -- a `c` arm, so the three-way additivity test is actually three-way
-----------------------------------------------------------------------

`RESULTS.md` defines the Option C term as `d C = d(merged) - d(bkvp_d1)`.
Substituting that into `d(merged) - [d(bkvp) + d(d1) + d C]` cancels `merged`
outright and leaves `d(bkvp_d1) - d(bkvp) - d(d1)`: a two-way test of
bkv_p x D1 wearing a three-way label.  Any interaction C has with the other
two is absorbed silently into the term defined as the leftover, and on prefill
C is the dominant term, so that is the axis where it matters.

`c` is the incumbent plus Option C and nothing else -- tuple (2,2,2), cond
live, head-major on the MIXED launch only.  It needs `streamindex_topk_cond.py`
(the merged module with the cond put back), because the merged module deleted
the cond and cannot build an arm with D=0.  `cctl` is the same module with C
off; it must reproduce `inc`, and it is what says whether the port is faithful.

E3 -- decoupling `ctx` from `max_model_len`
--------------------------------------------

`bkv_p` is compile-time static and `num_bkv = max(1, cdiv(kv_len_max, bkv_p *
page_size))` is runtime, so the bkv block is the quantum of overfetch.  Every
config in `merged_common.CONFIGS` except the production point has
`max_model_len == ctx`, so a request always fills its bucket and the downside
of a large `bkv_p` cannot appear.  `PLANS["e3"]` pins `max_model_len` at 9216
(so `bkv_p` is fixed at whatever the recommender returns for it) and sweeps
the *runtime* context underneath it.

Two things about the short end of that sweep, both of which change how the
rows read:

  * At `ctx <= 2048` the incumbent's early-exit fires: `max(seq_lens) //
    compression_ratio <= k` is `2048 <= 2048`.  So `inc` and `bkvp` take
    `_fast_path` and never run the kernel, while `d1`, `bkvp_d1` and `merged`
    (cond deleted) run it in full.  Their Delta-vs-`inc` at those two points is
    dominated by that and says nothing about `bkv_p`.  The bkv_p question at
    the short end is `bkvp_d1` vs `d1`, which is on the scoring path at every
    context.
  * `inc_nee` is the incumbent with `enable_early_exit=False`.  It is the
    scoring-path reference the two short rows need, and it doubles as a check
    that `d1` (the merged module) really is the incumbent-minus-cond: the two
    should land on each other at every context.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import merged_common as MC  # noqa: E402

Cfg = MC.Cfg
cdiv = MC.cdiv
align_to = MC.align_to

_COND = os.path.join(_HERE, "streamindex_topk_cond.py")

PAGE_SIZE = 1024
COMPRESSION_RATIO = 1

# ---------------------------------------------------------------------------
# Arms.  The five from `merged_common.ARMS` keep their exact definitions; the
# three new ones are appended.
# ---------------------------------------------------------------------------

# arm -> (module, decode-slot uses recommended bkv_p?, extra static kwargs)
GAPS_ARM_SPECS = {
    # -- the five from RESULTS.md, restated here so one table has all eight.
    "inc":     ("incumbent", False, {"enable_early_exit": True}),
    "bkvp":    ("incumbent", True,  {"enable_early_exit": True}),
    "d1":      ("merged",    False, {"head_major_prefill": False}),
    "bkvp_d1": ("merged",    True,  {"head_major_prefill": False}),
    "merged":  ("merged",    True,  {"head_major_prefill": True}),
    # -- new.
    # Option C alone, on top of the untouched incumbent.
    "c":       ("cond",      False, {"enable_early_exit": True,
                                     "head_major_prefill": True}),
    # The port-fidelity control: the same module with C off.  Must == `inc`.
    "cctl":    ("cond",      False, {"enable_early_exit": True,
                                     "head_major_prefill": False}),
    # The incumbent with the early-exit compiled out.  Not a candidate -- it
    # is the scoring-path reference for the E3 rows where the fast path fires,
    # and a cross-module check on `d1`.
    "inc_nee": ("incumbent", False, {"enable_early_exit": False}),
}

ALL_ARMS = tuple(GAPS_ARM_SPECS)

E1_ARMS = ("inc", "bkvp", "d1", "bkvp_d1", "merged", "c", "cctl")
E2_ARMS = ("inc", "bkvp", "d1", "bkvp_d1", "merged", "c", "cctl")
E3_ARMS = ("inc", "bkvp", "d1", "bkvp_d1", "merged", "inc_nee")


def load_cond():
    """The merged module with the early-exit cond restored."""
    return MC._load(_COND, "streamindex_topk_cond")


def load_all_modules():
    return {
        "incumbent": MC.load_incumbent(),
        "merged": MC.load_merged(),
        "cond": load_cond(),
    }


def arm_spec(arm: str, decode_bkv_p: int) -> dict:
    """The kwargs that distinguish one arm, given a resolved decode `bkv_p`."""
    module, use_bkvp, extra = GAPS_ARM_SPECS[arm]
    tup = ((decode_bkv_p, MC.PREFILL_BKV_P, MC.PREFILL_BKV_P)
           if use_bkvp else MC.INCUMBENT_TUPLE)
    spec = {"module": module, "num_kv_pages_per_block": tup}
    spec.update(extra)
    return spec


def make_call(B, mods, arm: str, inputs: dict, decode_bkv_p: int):
    """A zero-argument closure that runs one arm on `inputs`."""
    spec = arm_spec(arm, decode_bkv_p)
    static = dict(
        k=B.K_IDX,
        compression_ratio=B.COMPRESSION_RATIO,
        num_kv_pages_per_block=spec["num_kv_pages_per_block"],
        num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
    )
    for key in ("enable_early_exit", "head_major_prefill"):
        if key in spec:
            static[key] = spec[key]
    fn = mods[spec["module"]].streamindex_topk

    def call():
        return fn(**inputs, **static)

    return call, spec


# ---------------------------------------------------------------------------
# Scanned positions -- the quantity E3 exists to expose.
# ---------------------------------------------------------------------------


def scanned_positions(cfg: Cfg, tup) -> dict:
    """KV positions the kernel's bkv loop walks, per sequence, for one tuple.

    `streamindex_topk.py:275-276`: `kv_len_max = max(kv_lens)` and
    `num_bkv = max(1, cdiv(kv_len_max, bkv_sz))`, with `bkv_sz = bkv_p *
    page_size`.  `bkv_p` is a compile-time constant and `num_bkv` is a runtime
    count, so a request shorter than the block still pays the whole block --
    that rounding-up is the entire cost of a large `bkv_p`, and it is invisible
    in a grid where every config has `ctx == max_model_len`.

    Which slot of the 3-tuple applies is the phase's: decode uses [0], the
    mixed/prefill launch uses [2].  Since every arm here holds the prefill and
    mixed slots at 2, the prefill rows are identical across arms by
    construction and only the decode rows move.
    """
    bkv_p = tup[0] if cfg.phase == "decode" else tup[2]
    bkv_sz = bkv_p * PAGE_SIZE
    kv_len = cfg.ctx // COMPRESSION_RATIO
    num_bkv = max(1, cdiv(kv_len, bkv_sz))
    return {
        "bkv_p": bkv_p,
        "bkv_sz": bkv_sz,
        "num_bkv": num_bkv,
        "scanned_positions": num_bkv * bkv_sz,
        "kv_len": kv_len,
        "overfetch": num_bkv * bkv_sz - kv_len,
    }


# ---------------------------------------------------------------------------
# Configs.
# ---------------------------------------------------------------------------

PROD = MC.PROD  # dec_b64_ctx8704_mml9216

# E3: max_model_len pinned at the production 9216, runtime ctx swept under it.
E3_MML = 9216
E3_BATCH = 64
E3_CONTEXTS = (1024, 2048, 4096, 6144, 8704)


def e3_configs() -> list[Cfg]:
    out = []
    for ctx in E3_CONTEXTS:
        if ctx == PROD.ctx:
            out.append(PROD)  # the same Cfg object, same name, same inputs
            continue
        out.append(
            Cfg(f"dec_b{E3_BATCH}_ctx{ctx}_mml{E3_MML}", "decode", E3_BATCH,
                1, ctx, E3_MML))
    return out


E3_CONFIGS = e3_configs()

PREFILL_CONFIGS = [
    MC.CONFIG_BY_NAME[n] for n in (
        "pre_q1024_ctx4096",
        "pre_q1024_ctx16384",
        "pre_q4096_ctx4096",
        "pre_q4096_ctx16384",
    )
]

# ---------------------------------------------------------------------------
# Plans -- one per fan-out shard.
#
# A plan is a list of (Cfg, arms, reps).  Balance rationale: the prefill
# configs carry by far the largest per-call and per-compile cost, so E2 is
# split over two shards with the two q_len families separated; E1 is 7 arms of
# a 170 us decode op and is the cheapest; E3 is 30 small decode traces.
# ---------------------------------------------------------------------------

E1_REPS = 4

PLANS = {
    "e1": [(PROD, E1_ARMS, E1_REPS)],
    "e2a": [(PREFILL_CONFIGS[0], E2_ARMS, 1),
            (PREFILL_CONFIGS[1], E2_ARMS, 1)],
    "e2b": [(PREFILL_CONFIGS[2], E2_ARMS, 1),
            (PREFILL_CONFIGS[3], E2_ARMS, 1)],
    "e3": [(c, E3_ARMS, 1) for c in E3_CONFIGS],
}

# Which experiment each plan belongs to, for the reader's grouping.
PLAN_EXPERIMENT = {"e1": "E1", "e2a": "E2", "e2b": "E2", "e3": "E3"}


def tag(plan: str, cfg: Cfg, arm: str, rep: int, reps: int) -> str:
    """The trace directory name, and the join key between meta and xplane.

    `<plan>-<config>__<arm>` with a `#r<n>` suffix only when there is more
    than one rep, so a single-rep tag reads the same as `RESULTS.md`'s.  The
    plan prefix is what keeps E1's production config and E3's production row
    -- deliberately the same shape, measured on two different chips -- from
    colliding in one output tree.
    """
    base = f"{plan}-{cfg.name}__{arm}"
    return base if reps == 1 else f"{base}#r{rep}"


if __name__ == "__main__":
    # `JAX_PLATFORMS=cpu python gaps_common.py` -- a dry check of the plan.
    for name, plan in PLANS.items():
        n = sum(len(a) * r for _c, a, r in plan)
        print(f"{name:>4}: {len(plan)} config(s), {n} traces")
        for cfg, arms, reps in plan:
            pps = cdiv(cfg.block_table_len, PAGE_SIZE)
            rec = MC.load_merged().recommended_num_kv_pages_per_block(
                cfg.block_table_len, PAGE_SIZE)
            inc_s = scanned_positions(cfg, MC.INCUMBENT_TUPLE)
            rec_s = scanned_positions(cfg, rec)
            print(f"      {cfg.name:<28} phase={cfg.phase} "
                  f"pps={pps} rec={rec} "
                  f"scanned inc={inc_s['scanned_positions']} "
                  f"rec={rec_s['scanned_positions']} "
                  f"ratio={rec_s['scanned_positions'] / inc_s['scanned_positions']:.3f} "
                  f"like4like="
                  f"{MC.check_like_for_like(pps, rec, MC.INCUMBENT_TUPLE)} "
                  f"arms={len(arms)} reps={reps}")
