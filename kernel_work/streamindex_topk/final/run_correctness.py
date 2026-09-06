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
"""Correctness for the COMPOSED candidate, with the poison arm fixed.

WHAT WAS WRONG WITH THE STEP-3 CHECK
-------------------------------------
`variants/right-size/run_correctness.py:121-122` binds
`kv_len_bucket = r.kv_len` on EVERY arm, including the poison arm. With
bucket == kv_len the right-sized buffer has ZERO unwritten lanes, so the
`+1e30` poison is 100% overwritten before selection ever reads it. The check
that step 3's RESULTS.md calls "the strongest evidence in the exercise" had
nothing to catch on the one risk it names -- a reachable unwritten lane, which
is precisely what deleting the `-1` / `-inf` fill creates. It could not have
failed.

The configuration in which unwritten lanes exist is `bucket > kv_len`, and that
is the configuration the candidate SHIPS in: the bucket is a compile-time
constant `min(max_model_len, 65536)`, so every request shorter than the cap
leaves a gap. That configuration had never been run with an oracle anywhere --
only perf-only, `measure_topk=False`.

So this driver runs the poison arm in BOTH configurations and reports them
separately:

  bucket=static   bucket = min(max_model_len, 65536). THE SHIPPING
                  CONFIGURATION. `unwritten_sublanes > 0` at most shapes, and
                  the run asserts it is > 0 wherever the shape was chosen for
                  it. This is the arm the recommendation rests on.
  bucket=kvlen    bucket = kv_len, step 3's configuration. Kept, not because
                  it tests the fill, but because at kv_len 8000 it is the
                  masked partial-block STORE TAIL that the static bucket does
                  not reach. It must not be allowed to stand in for the other.

ARMS, per shape per bucket mode
--------------------------------
  lanecount    THE EMPIRICAL UNWRITTEN-LANE COUNT, read off the device buffer
               by shimming `sparsecore_topk`, and cross-checked against the
               analytic prediction. Runs FIRST, because every arm below it is
               only evidence if this one reports a count > 0. Also asserts
               `poison_cells_below_kv_len == 0`: a marker cell at a column
               below kv_len would be a HOLE inside the region selection can
               reach, which is a different and worse bug than an unwritten
               tail.
  incumbent    the shipped kernel. Its margins are the yardstick, measured
               here rather than quoted, so the comparison is same-process.
  composed     B + A, split cache, the bucket for this mode.
  poison       `composed` with the aliased prefill reinstated as +1e30 instead
               of -inf, so an unwritten lane does not merely perturb the
               ranking -- it SEIZES the top-k and drives worst_rank to the end
               of the row. Bit-identical margins to `composed` therefore prove
               no unwritten lane is reachable by selection.
  neg_control  `composed` with half the query heads zeroed. REQUIRED TO FAIL.
  neg_control_unwritten
               THE CONTROL THE WHOLE ARM EXISTS FOR: the same poisoned buffer
               with only the reachability gate (`row_lengths`) removed, so
               selection is allowed to reach the unwritten lanes. REQUIRED TO
               FAIL wherever `lanecount` measured a count > 0 -- and if it
               does not fail there, `poison` passing means nothing.

`lanecount`, `neg_control_unwritten` and the analytic `predict` are IMPORTED
from `probes/bucket_correctness/run_bucket_correctness.py`, the B-alone gate
(job j-6454262e-acd2-4371-b3ba), rather than rebuilt. Rebuilding them would
mean the composed candidate is judged by a second, unreviewed copy of the one
mechanism that makes this test falsifiable.

Index-set equality against the incumbent is reported for every arm, and it is
strictly stronger than the rank/recall criterion. Option A is bit-exact BY
CONSTRUCTION -- e8m0 is a bare 8-bit exponent and both bf16 and f32 have 8
exponent bits, so the split loses nothing -- and Option B's claim is that exact
sizing is bit-exact too. If `set_diff_rows` is not 0 on the `composed` arm,
that is a BUG to investigate, not a tolerance to accept.

CAVEAT, stated here so it reaches the report: the poison arm necessarily runs
with `aliased=True` (it needs a `scores_init` to poison), while the shipped
candidate's first band runs `aliased=False`. It is therefore a faithful proxy
for the COVERAGE question -- does selection ever read a lane the store did not
write -- and not a literal execution of the shipped path.
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

import _support as SUP

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import (  # noqa: E402
    test_correctness as TC)

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
        sa = {int(x) for x in a[t] if x >= 0}
        sb = {int(x) for x in b[t] if x >= 0}
        if sa != sb:
            bad += 1
            dd = len(sa ^ sb)
            if dd > worst:
                worst = dd
                first = (t, len(sa), len(sb), dd)
    return dict(rows_differing=bad, rows_total=int(a.shape[0]),
                worst_symmetric_difference=worst, worst_row=first)


def run_shape(name, mode, *, tup, arms, composed, incumbent, narrow, probe):
    import jax.numpy as jnp

    r = SUP.regime(name)
    bucket = SUP.bucket_for(r, composed, mode)
    with SUP.tuple_bound(tup):
        # The B-alone gate's own arithmetic, so the composed candidate is
        # judged against one prediction rather than two.
        pred = probe.predict(r, bucket, r.phase)
        acct = SUP.lane_accounting(r, bucket)
    tag = f"{name}[{mode}]tuple{''.join(str(t) for t in tup)}"
    print(f"\n[{tag}] {r.phase} kv_len={r.kv_len} mml={r.max_model_len} "
          f"bucket={bucket}", flush=True)
    print(f"[{tag}] incumbent_cols={acct['incumbent_cols']}  "
          f"predicted: {json.dumps(pred, default=str)}", flush=True)

    # Both arms of every comparison see literally the same oracle.
    d_packed = I.build(r, seed=0, with_dequant=True)
    oracle = TC.oracle_scores(d_packed, r)
    with narrow.patched_inputs():
        d_split = I.build(r, seed=0, with_dequant=True)

    real_sct = composed.sparsecore_topk
    out = []
    ref = None
    measured_unwritten = None

    for arm in arms:
        composed.sparsecore_topk = real_sct
        probe._clear(composed)
        probe._clear(incumbent)

        if arm == "lanecount":
            composed.sparsecore_topk = probe._lanecount_shim(
                jnp.float32(probe.POISON_LANECOUNT), r.kv_len)
            impl = SUP.bind(composed, kv_len_bucket=bucket,
                            poison_fill=probe.POISON_LANECOUNT)
            d, mutate = d_split, False
        elif arm == "neg_control_unwritten":
            composed.sparsecore_topk = probe._full_rowlen_shim(real_sct)
            impl = SUP.bind(composed, kv_len_bucket=bucket,
                            poison_fill=probe.POISON_UNWRITTEN)
            d, mutate = d_split, False
        elif arm == "incumbent":
            impl, d, mutate = incumbent, d_packed, False
        elif arm == "composed":
            impl = SUP.bind(composed, kv_len_bucket=bucket)
            d, mutate = d_split, False
        elif arm == "poison":
            impl = SUP.bind(composed, kv_len_bucket=bucket,
                            poison_fill=POISON)
            d, mutate = d_split, False
        elif arm == "neg_control":
            impl = SUP.bind(composed, kv_len_bucket=bucket)
            d, mutate = d_split, True
        else:
            raise ValueError(arm)

        try:
            with SUP.tuple_bound(tup):
                idxs = _call(impl, d, mutate_q=mutate)
        except Exception as exc:  # noqa: BLE001
            row = dict(regime=name, bucket_mode=mode, arm=arm, tuple=list(tup),
                       kv_len_bucket=bucket, error=repr(exc)[:3000])
            out.append(row)
            print(json.dumps(row, default=str), flush=True)
            continue
        finally:
            composed.sparsecore_topk = real_sct
            probe._clear(composed)

        if arm == "lanecount":
            n_, ns, cnt, first, last, cells, below = [int(x)
                                                      for x in idxs[0, :7]]
            row = dict(
                regime=name, bucket_mode=mode, arm=arm, tuple=list(tup),
                kv_len=r.kv_len, kv_len_bucket=bucket, scores_cols=n_,
                sublanes_total=ns,
                sublanes_unwritten_measured=cnt,
                sublanes_written_measured=ns - cnt,
                first_unwritten_sublane=(first if cnt else None),
                last_unwritten_sublane=(last if cnt else None),
                poison_cells_remaining=cells,
                poison_cells_below_kv_len=below,
                sublanes_unwritten_predicted=pred["sublanes_unwritten"],
                agrees_with_prediction=(cnt == pred["sublanes_unwritten"]),
                verdict=("PASS" if (cnt == pred["sublanes_unwritten"]
                                    and below == 0) else "FAIL"),
            )
            measured_unwritten = cnt
            out.append(row)
            print(json.dumps(row, default=str), flush=True)
            continue

        n, failures, st = TC.check(idxs, oracle, S.INDEX_TOPK)
        row = dict(
            regime=name, bucket_mode=mode, arm=arm, tuple=list(tup),
            kv_len=r.kv_len, max_model_len=r.max_model_len,
            kv_len_bucket=bucket,
            scores_cols=(acct["incumbent_cols"] if arm == "incumbent"
                         else acct["buffer_cols"]),
            unwritten_measured=measured_unwritten,
            unwritten_predicted=pred["sublanes_unwritten"],
            tail_store_executes=(False if arm == "incumbent"
                                 else pred["tail_store_executes"]),
            rows_checked=n, n_failures=len(failures),
            worst_rank=st["worst_rank_seen"],
            allowed_rank=S.INDEX_TOPK + TC.RANK_SLACK,
            min_recall=st["min_recall"], required_recall=TC.MIN_RECALL,
            max_index_returned=int(idxs.max()),
            verdict=("FAIL" if failures else "PASS"),
            first_failure=(failures[0] if failures else None),
        )
        if arm == "incumbent":
            ref = idxs
        elif ref is not None and arm in ("composed", "poison"):
            row["vs_incumbent"] = _set_equal(ref, idxs)
        out.append(row)
        print(json.dumps(row, default=str), flush=True)

    composed.sparsecore_topk = real_sct
    del d_packed, d_split, oracle
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--bucket-modes", default="static")
    ap.add_argument("--arms",
                    default="lanecount,incumbent,composed,poison,"
                            "neg_control,neg_control_unwritten")
    ap.add_argument("--tuples", default="2,2,2",
                    help="semicolon-separated, e.g. '2,2,2;4,2,4'")
    ap.add_argument("--require-unwritten", default="",
                    help="comma list of regimes where unwritten_sublanes MUST "
                         "be > 0, else the shape is not testing what it claims")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    SUP.banner()
    incumbent = SUP.load_incumbent()
    composed = SUP.load_composed()
    narrow = SUP.load_narrow_inputs()
    probe = SUP.load_bucket_probe()

    arms = [x for x in a.arms.split(",") if x]
    modes = [x for x in a.bucket_modes.split(",") if x]
    tuples = [tuple(int(v) for v in t.split(",")) for t in a.tuples.split(";")
              if t]
    must = {x for x in a.require_unwritten.split(",") if x}

    rows = []
    hollow = []
    for name in [n for n in a.regimes.split(",") if n]:
        for mode in modes:
            for tup in tuples:
                r = SUP.regime(name)
                bucket = SUP.bucket_for(r, composed, mode)
                with SUP.tuple_bound(tup):
                    acct = SUP.lane_accounting(r, bucket)
                if name in must and acct["unwritten_sublanes"] == 0:
                    # Exactly the step-3 defect. Refuse to report it as a pass.
                    hollow.append(f"{name}[{mode}] tuple={tup}")
                rows += run_shape(name, mode, tup=tup, arms=arms,
                                  composed=composed, incumbent=incumbent,
                                  narrow=narrow, probe=probe)

    print("\n=== COMPOSED candidate: measured unwritten lanes ===")
    hdr = (f"{'regime':<24}{'mode':<8}{'tup':<8}{'bucket':>8}{'total':>7}"
           f"{'unwr':>6}{'pred':>6}{'first':>7}{'last':>6}{'below_kv':>10}"
           f"{'verdict':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("arm") != "lanecount":
            continue
        if "error" in r:
            print(f"{r['regime']:<24}{r['bucket_mode']:<8}  ERROR")
            continue
        t = "".join(str(x) for x in r["tuple"])
        print(f"{r['regime']:<24}{r['bucket_mode']:<8}{t:<8}"
              f"{r['kv_len_bucket']:>8}{r['sublanes_total']:>7}"
              f"{r['sublanes_unwritten_measured']:>6}"
              f"{r['sublanes_unwritten_predicted']:>6}"
              f"{str(r['first_unwritten_sublane']):>7}"
              f"{str(r['last_unwritten_sublane']):>6}"
              f"{r['poison_cells_below_kv_len']:>10}{r['verdict']:>9}")

    print("\n=== COMPOSED candidate correctness "
          "(float64 oracle, rank/recall criterion) ===")
    hdr = (f"{'regime':<24}{'mode':<8}{'tup':<6}{'arm':<22}{'cols':>7}"
           f"{'unwr':>6}{'verdict':>9}{'worst_rank':>12}{'allowed':>9}"
           f"{'min_recall':>12}{'set_diff':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("arm") == "lanecount":
            continue
        if "error" in r:
            print(f"{r['regime']:<24}{r['bucket_mode']:<8}"
                  f"{'':<6}{r['arm']:<22}  ERROR")
            continue
        sd = r.get("vs_incumbent")
        sd_s = "-" if sd is None else str(sd["rows_differing"])
        t = "".join(str(x) for x in r["tuple"])
        um = r.get("unwritten_measured")
        print(f"{r['regime']:<24}{r['bucket_mode']:<8}{t:<6}{r['arm']:<22}"
              f"{r['scores_cols']:>7}{str('-' if um is None else um):>6}"
              f"{r['verdict']:>9}{r['worst_rank']:>12}{r['allowed_rank']:>9}"
              f"{r['min_recall']:>12.5f}{sd_s:>10}")
    print("\nREQUIRED VERDICTS: `neg_control` FAIL always; "
          "`neg_control_unwritten` FAIL wherever the MEASURED unwritten count "
          "is > 0 (and it is vacuous, not passing, where the count is 0); "
          "every other arm PASS with set_diff 0.")
    print("`unwr` is the MEASURED count of sublanes the kernel never writes. "
          "The poison arm is only evidence where it is > 0.")

    for r in rows:
        if "error" in r:
            print(f"\nERROR [{r['regime']}/{r['bucket_mode']}/{r['arm']}]: "
                  f"{r['error']}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)

    red = 0
    vacuous = []
    for r in rows:
        arm = r.get("arm")
        if "error" in r:
            red += 1
        elif arm == "neg_control":
            red += (r["verdict"] != "FAIL")
        elif arm == "neg_control_unwritten":
            um = r.get("unwritten_measured") or 0
            if um == 0:
                # Not a pass and not a failure: with nothing unwritten there
                # is nothing for the removed gate to reach.
                vacuous.append(f"{r['regime']}[{r['bucket_mode']}]")
            else:
                red += (r["verdict"] != "FAIL")
        else:
            red += (r["verdict"] != "PASS")
            sd = r.get("vs_incumbent")
            if sd is not None and sd["rows_differing"] != 0:
                red += 1
    for v in vacuous:
        print(f"\nVACUOUS (not counted either way): neg_control_unwritten at "
              f"{v} had 0 measured unwritten sublanes.")
    for h in hollow:
        print(f"\nHOLLOW CHECK: {h} was required to have unwritten lanes and "
              f"has none. Its poison arm proves nothing; treat as red.")
    red += len(hollow)
    print(f"\nCORRECTNESS DONE red={red}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
