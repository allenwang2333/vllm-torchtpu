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
"""Correctness for Option B, including the test the design says is the risk.

Four arms per regime, all judged by the READ-ONLY float64 oracle and rank/recall
criterion in `instruments/test_correctness.py`. The oracle is built once per
regime and shared, so every arm is scored against literally the same reference.

  incumbent     the shipped kernel; its margins are the yardstick the other
                arms are read against, measured here rather than quoted from
                step 1 so the comparison is same-process.
  right_size    the variant. Buffer sized from the live kv_len, no -inf fill.
  poison        THE TEST FOR THE THING THAT MAKES B WRONG RATHER THAN SLOW.
  neg_control   the variant with half its query heads zeroed; the criterion is
                required to FAIL. A check that has never failed is not evidence.

Why `poison` is the right test, stated before the store was written
-------------------------------------------------------------------
Deleting the `-inf` prefill removes the thing that made an untouched lane lose
the top-k. If the masked store does not cover exactly the region the prefill
used to, the kernel selects whatever was in that memory from a previous step --
plausible-looking indices, not obvious garbage, so neither a throughput number
nor a negative control designed around a *ranking* change is guaranteed to see
it, and even a rank check may not: stale scores from the previous step are
scores of the same distribution, and a handful of them can land inside the rank
slack.

`poison` removes the luck. It reinstates the aliased prefill with a large
POSITIVE constant (1e30) instead of `-inf`. Now every lane the kernel does not
write is the largest value in its row, so any read of such a lane does not
merely perturb the ranking -- it takes over the top-k and drives `worst_rank`
to the end of the row. Margins bit-identical to the un-poisoned arm therefore
prove that no unwritten lane is reachable by selection, which is the whole
safety claim behind deleting the fill (design section 4.4). It is a strict
test, not an approximation, and it is decisive in one run rather than
probabilistic over many.

`--stale` adds the weaker but more literal version: run the poisoned arm, drop
its output, then run the plain variant on DIFFERENT inputs and require its
index sets to match the incumbent's on those inputs. If the allocator hands the
second call the first call's 1e30 buffer and the kernel reads any of it, the
sets diverge. Best-effort by nature (nothing forces the allocator's hand), so
it is reported as a corroboration of `poison`, never as a substitute.

Index-set equality against the incumbent is reported for every arm. It is
strictly stronger than the rank criterion: design section 4.4 claim (1) is that
exact sizing is BIT-EXACT, not merely within slack, and a set comparison is the
only thing that can show that.
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from kernel_work.streamindex_topk.instruments import inputs as I
from kernel_work.streamindex_topk.instruments import shapes as S
from kernel_work.streamindex_topk.instruments import test_correctness as TC

import _rs_support as RS

POISON = 1e30


def _call(impl, d, *, mutate_q=False):
    q = d["q"]
    if mutate_q:
        qm = np.array(q.astype(jnp.float32), copy=True)
        qm[:, S.INDEX_N_HEADS // 2:, :] = 0.0
        q = jnp.asarray(qm).astype(jnp.float8_e4m3fn)
    idxs = impl.streamindex_topk(
        q, d["indexer_weights"], d["cache_kv"], d["seq_lens"],
        d["page_indices"], d["cu_q_lens"], d["distribution"],
        k=S.INDEX_TOPK,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=False,
    )
    return np.asarray(jax.block_until_ready(idxs))


def _set_equal(a, b):
    """Rows whose returned index SETS differ, ignoring order and -1 padding."""
    bad = 0
    worst = 0
    first = None
    for t in range(a.shape[0]):
        sa = set(int(x) for x in a[t] if x >= 0)
        sb = set(int(x) for x in b[t] if x >= 0)
        if sa != sb:
            bad += 1
            d = len(sa ^ sb)
            if d > worst:
                worst = d
                first = (t, len(sa), len(sb), d)
    return dict(rows_differing=bad, rows_total=int(a.shape[0]),
                worst_symmetric_difference=worst, worst_row=first)


def run_regime(name, *, arms, stale):
    r = RS.regime(name)
    w_inc, w_rs = RS.widths(r)
    inc = RS.load_incumbent()
    var = RS.load_variant()
    rs = RS.bind(var, kv_len_bucket=r.kv_len)
    rs_poison = RS.bind(var, kv_len_bucket=r.kv_len, poison_fill=POISON)

    d = I.build(r, seed=0, with_dequant=True)
    oracle = TC.oracle_scores(d, r)

    out = []
    ref_idxs = None
    for arm in arms:
        impl, mutate = {
            "incumbent": (inc, False),
            "right_size": (rs, False),
            "poison": (rs_poison, False),
            "neg_control": (rs, True),
        }[arm]
        try:
            idxs = _call(impl, d, mutate_q=mutate)
        except Exception as exc:  # noqa: BLE001
            row = dict(regime=name, arm=arm, error=repr(exc)[:3000])
            out.append(row)
            print(json.dumps(row), flush=True)
            continue
        n, failures, st = TC.check(idxs, oracle, S.INDEX_TOPK)
        row = dict(
            regime=name, arm=arm,
            scores_cols=(w_inc if arm == "incumbent" else w_rs),
            kv_len=r.kv_len, max_model_len=r.max_model_len,
            rows_checked=n, n_failures=len(failures),
            worst_rank=st["worst_rank_seen"],
            allowed_rank=S.INDEX_TOPK + TC.RANK_SLACK,
            min_recall=st["min_recall"], required_recall=TC.MIN_RECALL,
            verdict=("FAIL" if failures else "PASS"),
            first_failure=(failures[0] if failures else None),
        )
        if arm == "incumbent":
            ref_idxs = idxs
        elif ref_idxs is not None:
            row["vs_incumbent"] = _set_equal(ref_idxs, idxs)
        out.append(row)
        print(json.dumps(row), flush=True)

    if stale:
        out.append(run_stale(name, r, inc, var))
    return out


def run_stale(name, r, inc, var):
    """Poison a call, free it, then require a clean call to be unaffected."""
    rs = RS.bind(var, kv_len_bucket=r.kv_len)
    rs_poison = RS.bind(var, kv_len_bucket=r.kv_len, poison_fill=POISON)
    d0 = I.build(r, seed=0)
    try:
        junk = _call(rs_poison, d0)
        del junk
        d1 = I.build(r, seed=7)
        got = _call(rs, d1)
        want = _call(inc, d1)
        row = dict(regime=name, arm="stale_after_poison",
                   **_set_equal(want, got))
        row["verdict"] = ("PASS" if row["rows_differing"] == 0 else "FAIL")
    except Exception as exc:  # noqa: BLE001
        row = dict(regime=name, arm="stale_after_poison",
                   error=repr(exc)[:3000])
    print(json.dumps(row), flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--arms",
                    default="incumbent,right_size,poison,neg_control")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    arms = [x for x in a.arms.split(",") if x]
    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        rows += run_regime(name, arms=arms, stale=a.stale)

    print("\n=== Option B correctness (float64 oracle, rank/recall criterion) ===")
    hdr = (f"{'regime':<18}{'arm':<20}{'cols':>7}{'verdict':>9}"
           f"{'worst_rank':>12}{'allowed':>9}{'min_recall':>12}{'req':>7}"
           f"{'set_diff_rows':>15}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['regime']:<18}{r['arm']:<20}  ERROR")
            continue
        if r["arm"] == "stale_after_poison":
            print(f"{r['regime']:<18}{r['arm']:<20}{'':>7}"
                  f"{r['verdict']:>9}{'':>12}{'':>9}{'':>12}{'':>7}"
                  f"{r['rows_differing']:>15}")
            continue
        sd = r.get("vs_incumbent")
        sd_s = "-" if sd is None else str(sd["rows_differing"])
        print(f"{r['regime']:<18}{r['arm']:<20}{r['scores_cols']:>7}"
              f"{r['verdict']:>9}{r['worst_rank']:>12}{r['allowed_rank']:>9}"
              f"{r['min_recall']:>12.5f}{r['required_recall']:>7}{sd_s:>15}")
    print("\nNOTE: for `neg_control` the REQUIRED verdict is FAIL. Any other "
          "arm reading FAIL, or neg_control reading PASS, is a red result.")
    for r in rows:
        if "error" in r:
            print(f"\nERROR [{r['regime']}/{r['arm']}]: {r['error']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    # Exit non-zero on a red result so the shard's rc means something. A
    # `neg_control` that PASSES is red: it means the criterion is vacuous.
    red = 0
    for r in rows:
        if "error" in r:
            red += 1
        elif r["arm"] == "neg_control":
            red += (r["verdict"] != "FAIL")
        else:
            red += (r["verdict"] != "PASS")
    if red:
        print(f"\n{red} RED result(s).")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
