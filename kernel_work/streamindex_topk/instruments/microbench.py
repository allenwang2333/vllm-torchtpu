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
"""The yardstick. One number per (implementation, regime, arm).

THIS FILE IS READ-ONLY AFTER STEP 1. Every candidate is judged by it, and a
yardstick that changes between candidates is not a yardstick. If a variant
needs a shape this does not sweep, add a row -- do not fork the file.

Sign convention, binding on every document this exercise writes: every
percentage is a change in TIME, `(new - old) / old`. Negative is faster.

What it measures
----------------
`full`   -- the whole `streamindex_topk` call, which is what the model pays.
            This is the headline and the number variants are ranked by.
`topk`   -- `sparsecore_topk` alone on a pre-built scores tensor of the same
            shape. Measured so the scoring half can be attributed by
            difference. It is NOT a result on its own: it omits the aliased
            -inf initialisation that only exists inside the full call, and it
            is reported only as the subtrahend of `full - topk`.

What it cannot measure
----------------------
Single chip, one TensorCore. GLM-5.2 runs TP=1/DP=16/EP=16, so the indexer is
*replicated* across ranks rather than sharded -- every rank runs this exact
kernel on its own sequences (mla_attention_op.py has no collective on this
path). That makes single-chip the right denominator for the kernel itself. It
still cannot see: HBM pressure from the other 77 layers' weights and
activations, contention with the SparseCore work the rest of the step issues,
or rank skew behind the step barrier. A win here is a kernel-level win and
must be labelled as one.
"""

import argparse
import importlib
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from . import inputs as I
from . import shapes as S

WARMUP = 3
REPS = 10


def _median_spread(times_s):
    """Return (median_ms, spread_pct) where spread is (max-min)/median."""
    med = statistics.median(times_s)
    spread = (max(times_s) - min(times_s)) / med * 100.0 if med else float("nan")
    return med * 1e3, spread


def _time(fn, *args, reps=REPS, warmup=WARMUP):
    """Time a device call. Blocks on the result -- a timing that forgets to
    block times dispatch, which reads as an implausibly good number."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return ts


def run_regime(impl, regime: S.Regime, *, early_exit: bool, reps=REPS,
               measure_topk=True):
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        sparsecore_topk)

    d = I.build(regime)
    k = S.INDEX_TOPK

    def call():
        return impl.streamindex_topk(
            d["q"],
            d["indexer_weights"],
            d["cache_kv"],
            d["seq_lens"],
            d["page_indices"],
            d["cu_q_lens"],
            d["distribution"],
            k=k,
            compression_ratio=S.COMPRESSION_RATIO,
            num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
            num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
            decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
            enable_early_exit=early_exit,
        )

    full_ms, full_spread = _median_spread(_time(call, reps=reps))

    row = dict(
        regime=regime.name,
        phase=regime.phase,
        kv_len=regime.kv_len,
        num_tokens=regime.num_tokens,
        max_num_seqs=regime.max_num_seqs,
        num_bq=regime.num_bq,
        num_bkv=regime.num_bkv,
        num_sublanes_total=S.num_sublanes_total(regime.max_model_len),
        scores_MB=S.scores_bytes(regime.num_tokens, regime.max_model_len) / 1e6,
        early_exit=early_exit,
        full_ms=full_ms,
        full_spread_pct=full_spread,
    )

    if measure_topk:
        width = S.num_sublanes_total(regime.max_model_len) * 128
        scores = jnp.zeros((regime.num_tokens, width), jnp.float32)
        row_lengths = jnp.full((regime.num_tokens, ),
                               min(regime.kv_len, width), jnp.int32)
        topk_fn = jax.jit(
            lambda s, r: sparsecore_topk(
                s, k, row_lengths=r, write_empty_rows=True))
        topk_ms, topk_spread = _median_spread(
            _time(topk_fn, scores, row_lengths, reps=reps))
        # CONTROL ARM -- not a result on its own. See the module docstring.
        row["_control_topk_only_ms"] = topk_ms
        row["_control_topk_spread_pct"] = topk_spread
        row["scoring_ms_by_difference"] = full_ms - topk_ms

    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl",
                    default="vllm_torchtpu.kernels.deepseek_v4.streamindex_topk",
                    help="module exposing streamindex_topk(...)")
    ap.add_argument("--regimes", default="",
                    help="comma-separated regime names; default all")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--early-exit", default="both",
                    choices=["false", "true", "both"])
    ap.add_argument("--no-topk-arm", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    impl = importlib.import_module(a.impl)
    names = [n for n in a.regimes.split(",") if n] or [r.name for r in S.REGIMES]
    arms = {"false": [False], "true": [True], "both": [False, True]}[a.early_exit]

    rows = []
    for name in names:
        regime = S.REGIMES_BY_NAME[name]
        for ee in arms:
            try:
                row = run_regime(impl, regime, early_exit=ee, reps=a.reps,
                                 measure_topk=not a.no_topk_arm)
            except Exception as exc:  # noqa: BLE001
                row = dict(regime=name, early_exit=ee, error=repr(exc)[:400])
            row["impl"] = a.impl
            rows.append(row)
            print(json.dumps(row), flush=True)

    print("\n=== summary (median ms/call, single TensorCore) ===")
    hdr = f"{'regime':<14}{'ee':<7}{'nbq':>5}{'nbkv':>6}{'scoresMB':>10}{'full_ms':>10}{'spread%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['regime']:<14}{str(r['early_exit']):<7}  ERROR {r['error'][:60]}")
            continue
        print(f"{r['regime']:<14}{str(r['early_exit']):<7}{r['num_bq']:>5}"
              f"{r['num_bkv']:>6}{r['scores_MB']:>10.1f}{r['full_ms']:>10.3f}"
              f"{r['full_spread_pct']:>9.2f}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
