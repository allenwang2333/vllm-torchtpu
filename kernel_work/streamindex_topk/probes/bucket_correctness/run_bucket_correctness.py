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
"""Option B correctness AT `kv_len_bucket > kv_len` -- the configuration it ships in.

Why this file exists
--------------------
`variants/right-size/run_correctness.py` binds `kv_len_bucket = r.kv_len` in
every arm (:121-122). With bucket == kv_len the buffer is exactly
`cdiv(kv_len,128)` sublanes and the scan writes `align_to(kv_len, bkv_sz)`
positions, so the written region covers the whole buffer and there are **zero
unwritten lanes at every shape that test has ever run**. The poison arm is
therefore 100% overwritten before selection reads it: the test could not have
failed on the risk it was designed for.

B ships behind a *bucket ladder* (probes/followup1/bucket_ladder.py), where the
bucket is rounded UP. That is the configuration with unwritten lanes, and it has
only ever been run perf-only, with no oracle.

This file is `run_correctness.py` with one thing changed -- the bucket binding --
plus the instrumentation needed to prove the unwritten lanes are really there.
Same float64 oracle, same rank/recall criterion, same index-set comparison,
same poison arm, same negative control. Nothing in `instruments/` is edited.

What is added, and only this
----------------------------
* `--bucket-mode` : `ladder` (round kv_len up to the next power-of-two bucket,
  what B ships) or `exact` (bucket == kv_len, the old binding, kept as the
  control arm so both configurations appear in one table).

* arm `lanecount` : the EMPIRICAL unwritten-lane count. `sparsecore_topk` is
  replaced, in the variant module's own namespace, by a shim that -- instead of
  selecting -- counts how many 128-column sublanes of the scores buffer are
  still entirely equal to the poison value it was pre-filled with. A sublane
  still holding the poison is a sublane the kernel never wrote. The count comes
  out of the device as an ordinary array element, so it is a measurement, not
  an inference from the width arithmetic. The analytic prediction is printed
  beside it and disagreement is flagged.

* arm `neg_control_unwritten` : THE CONTROL THIS WHOLE JOB EXISTS FOR. Same
  poisoned buffer, but the shim drops `row_lengths` (passes `None`, which
  `sparsecore_topk` documents as "defaults to n", :404-405). Selection is then
  allowed to reach the unwritten lanes. If those lanes exist and hold 1e30 they
  must seize the top-k, the returned indices must run off the end of the
  oracle's `kv_len` columns, and the criterion must go RED. This is the arm
  that proves the poison arm's PASS is worth something: it is the same buffer,
  the same poison and the same oracle, with only the reachability gate removed.
  Required verdict: FAIL, and only at shapes whose measured unwritten count > 0.

* arm `neg_control` : unchanged from `run_correctness.py` (half the query heads
  zeroed). Required verdict: FAIL.

Trace-cache hazard, handled
---------------------------
`streamindex_topk` is `jax.jit`-wrapped and looks up the module-global
`sparsecore_topk` at TRACE time. Rebinding that global does not invalidate a
cached trace, so a shim installed after a plain call of the same shape would
silently not run. Two defences, both applied: every shimmed arm uses a distinct
`poison_fill` (a *static* argname, so a distinct cache key), and the jit cache
is cleared before and after every shimmed arm.
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# repo root, so `kernel_work.…` imports resolve
sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])
# variants/right-size, so `_rs_support` and the variant kernel resolve
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "variants", "right-size")))

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import (  # noqa: E402
    test_correctness as TC)

import _rs_support as RS  # noqa: E402

# One poison value per arm. They differ so that each arm is a distinct jit
# cache key even before the explicit cache clear -- see the module docstring.
POISON = 1e30
POISON_LANECOUNT = 1.001e30
POISON_UNWRITTEN = 1.002e30

# The ladder B would ship behind. Powers of two, as in
# probes/followup1/bucket_ladder.py:POW2_BUCKETS.
POW2_BUCKETS = [2048, 4096, 8192, 16384, 32768, 65536]


# ---------------------------------------------------------------------------
# Shapes. Constructed here, not appended to `instruments/shapes.py`, which is
# READ-ONLY: `S.REGIMES` and `S.REGIMES_BY_NAME` are untouched and the
# instruments take a `Regime` INSTANCE. Same mechanism `extra_regimes.py` uses.
#
# `max_model_len` is chosen so that `pages_per_seq * PAGE_SIZE >= bucket`,
# otherwise `live_positions = min(kv_len_bucket, pages_per_seq * page_size)`
# (streamindex_topk_rs.py:757) clips the bucket back and there is nothing to
# test.
# ---------------------------------------------------------------------------
def _mk(name, phase, kv_len, max_model_len, max_num_seqs, num_tokens, note):
    return S.Regime(name=name, phase=phase, max_num_seqs=max_num_seqs,
                    num_tokens=num_tokens, kv_len=kv_len,
                    max_model_len=max_model_len, note=note)


REGIMES = {}
for _n, _p, _kv, _mml, _ns, _nt, _note in [
    # (a) kv 24576 -> bucket 32768. 256-sublane buffer, 192 written.
    #     DECODE is the sharp case: every row's `visible` is exactly kv_len,
    #     which is exactly the last written position + 1.
    ("bc_a_decode", "decode", 24576, 33792, 6, 6,
     "kv 24576, pow2 bucket 32768. decode: visible == kv_len for every row, "
     "the tightest reach the selection can have."),
    ("bc_a_mixed", "mixed", 24576, 33792, 6, 1024,
     "kv 24576, pow2 bucket 32768, MIXED band."),
    # (b) kv 8000 -> bucket 16384. The non-multiple-of-bkv_sz length; under an
    #     exact bucket this is the only shape that emits the masked tail store.
    ("bc_b_decode", "decode", 8000, 66560, 6, 6,
     "kv 8000 (not a multiple of bkv_sz=2048), pow2 bucket 16384, DECODE "
     "band. max_num_seqs=6 so both the batched (0..3) and remainder (4..5) "
     "decode invocations run."),
    ("bc_b_mixed", "mixed", 8000, 66560, 6, 1024,
     "kv 8000, pow2 bucket 16384, MIXED band."),
    # (c) kv 49152 -> bucket 65536. 512-sublane buffer, 384 written.
    ("bc_c_decode", "decode", 49152, 66560, 6, 6,
     "kv 49152, pow2 bucket 65536, DECODE band."),
    ("bc_c_mixed", "mixed", 49152, 66560, 6, 1024,
     "kv 49152, pow2 bucket 65536, MIXED band."),
    # (d) A bucket that is NOT a multiple of bkv_sz, so `n_tail_sublanes != 0`
    #     and B's masked partial-block store is COMPILED IN, while unwritten
    #     lanes also exist. See the note in main(): with
    #     `num_bkv = min(cdiv(kv_len,bkv_sz), max_bkv)` the tail store can only
    #     EXECUTE when it lands on the last slab, which is exactly the case
    #     with no unwritten lanes -- so this is the closest a single shape can
    #     get to holding both properties at once.
    ("bc_d_mixed", "mixed", 8000, 66560, 6, 1024,
     "kv 8000, bucket 9216 (NOT a multiple of bkv_sz), so n_tail_sublanes=8 "
     "and the masked partial store is emitted while unwritten lanes exist."),
    ("bc_d_decode", "decode", 8000, 66560, 6, 6,
     "kv 8000, bucket 9216, DECODE band."),
]:
    REGIMES[_n] = _mk(_n, _p, _kv, _mml, _ns, _nt, _note)

# Buckets that are not simply "next power of two".
#
#   bc_b_*  the pow2 ladder would hand kv_len=8000 a bucket of 8192, which is
#           only 192 positions of slack -- 0 unwritten sublanes, so it would
#           prove nothing. 16384 is what a *coarse* ladder gives it (a ladder
#           whose lowest rung is 16k, which is a plausible way to bound the
#           recompile count), and it is the worst case in the whole set: the
#           bucket is 2.05x the live length, so half the buffer is unwritten.
#   bc_d_*  9216 is deliberately NOT a multiple of bkv_sz=2048, so
#           `n_tail_sublanes != 0` and the masked partial-block store is
#           compiled in while unwritten lanes exist.
BUCKET_OVERRIDE = {"bc_d_mixed": 9216, "bc_d_decode": 9216,
                   "bc_b_mixed": 16384, "bc_b_decode": 16384}


def pow2_bucket(kv_len):
    for b in POW2_BUCKETS:
        if b >= kv_len:
            return b
    raise ValueError(f"kv_len {kv_len} above the ladder")


def bucket_for(name, r, mode):
    if mode == "exact":
        return r.kv_len
    return BUCKET_OVERRIDE.get(name, pow2_bucket(r.kv_len))


# ---------------------------------------------------------------------------
# Analytic prediction of the geometry, from the variant's own arithmetic.
# Printed BESIDE the measurement, never in place of it.
# ---------------------------------------------------------------------------
def predict(r, bucket, phase):
    idx = 0 if phase == "decode" else 2
    bkv_sz = S.PAGE_SIZE * S.NUM_KV_PAGES_PER_BLOCK[idx]
    num_sublanes_bkv = bkv_sz // 128
    pps = S.pages_per_seq(r.max_model_len)
    live = min(bucket, pps * S.PAGE_SIZE)
    total = max(S.cdiv(live, 128), S.cdiv(S.INDEX_TOPK, 128))
    n_full = total // num_sublanes_bkv
    n_tail = total - n_full * num_sublanes_bkv
    max_bkv = S.cdiv(total, num_sublanes_bkv)
    num_bkv = min(max(1, S.cdiv(r.kv_len, bkv_sz)), max_bkv)
    if n_tail and num_bkv == max_bkv:
        written = n_full * num_sublanes_bkv + n_tail
        tail_executes = True
    else:
        written = min(num_bkv, n_full) * num_sublanes_bkv
        tail_executes = False
    return dict(sublanes_total=total, sublanes_written=written,
                sublanes_unwritten=total - written, num_bkv=num_bkv,
                max_bkv=max_bkv, n_tail_sublanes=n_tail,
                tail_store_emitted=bool(n_tail),
                tail_store_executes=tail_executes,
                bkv_sz=bkv_sz)


# ---------------------------------------------------------------------------
# Shims installed into the VARIANT MODULE's namespace. Trace-time only.
# ---------------------------------------------------------------------------
def _lanecount_shim(marker, kv_len):
    """Count sublanes still entirely equal to `marker` -- i.e. never written."""

    def shim(scores, k, row_lengths=None, **kw):
        del row_lengths, kw
        b, n = scores.shape
        ns = n // 128
        m = marker.astype(scores.dtype)
        is_mark = (scores == m)
        per_sub = is_mark.reshape(b, ns, 128).all(axis=2).all(axis=0)
        cnt = per_sub.sum().astype(jnp.int32)
        idxs = jnp.arange(ns, dtype=jnp.int32)
        first = jnp.where(per_sub, idxs, jnp.int32(ns)).min()
        last = jnp.where(per_sub, idxs, jnp.int32(-1)).max()
        cells = is_mark.sum().astype(jnp.int32)
        # Marker cells at a column below kv_len would mean a HOLE inside the
        # region the selection can reach. Must be 0.
        below = jnp.where(jnp.arange(n) < kv_len, is_mark, False)
        cells_below = below.sum().astype(jnp.int32)
        vals = jnp.stack([jnp.int32(n), jnp.int32(ns), cnt, first, last,
                          cells, cells_below])
        out = jnp.full((b, k), -1, jnp.int32)
        out = out.at[0, :vals.shape[0]].set(vals)
        return out

    return shim


def _full_rowlen_shim(real):
    """Drop `row_lengths` -> selection may reach every column of the buffer."""

    def shim(scores, k, row_lengths=None, **kw):
        del row_lengths
        return real(scores, k, row_lengths=None, **kw)

    return shim


def _clear(mod):
    fn = getattr(mod, "streamindex_topk", None)
    if hasattr(fn, "clear_cache"):
        fn.clear_cache()


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


def run_regime(name, *, mode, arms):
    r = REGIMES[name]
    bucket = bucket_for(name, r, mode)
    pred = predict(r, bucket, r.phase)

    inc = RS.load_incumbent()
    var = RS.load_variant()
    real_sct = var.sparsecore_topk

    d = I.build(r, seed=0, with_dequant=True)
    oracle = TC.oracle_scores(d, r)

    out = []
    ref_idxs = None
    measured_unwritten = None

    hdr = dict(regime=name, phase=r.phase, kv_len=r.kv_len, bucket=bucket,
               mode=mode, max_model_len=r.max_model_len,
               max_num_seqs=r.max_num_seqs, num_tokens=r.num_tokens,
               predicted=pred)
    print(json.dumps(dict(kind="regime_header", **hdr), default=str),
          flush=True)

    for arm in arms:
        var.sparsecore_topk = real_sct
        _clear(var)
        _clear(inc)

        if arm == "lanecount":
            var.sparsecore_topk = _lanecount_shim(
                jnp.float32(POISON_LANECOUNT), r.kv_len)
            impl = RS.bind(var, kv_len_bucket=bucket,
                           poison_fill=POISON_LANECOUNT)
            mutate = False
        elif arm == "neg_control_unwritten":
            var.sparsecore_topk = _full_rowlen_shim(real_sct)
            impl = RS.bind(var, kv_len_bucket=bucket,
                           poison_fill=POISON_UNWRITTEN)
            mutate = False
        elif arm == "incumbent":
            impl, mutate = inc, False
        elif arm == "right_size":
            impl, mutate = RS.bind(var, kv_len_bucket=bucket), False
        elif arm == "poison":
            impl = RS.bind(var, kv_len_bucket=bucket, poison_fill=POISON)
            mutate = False
        elif arm == "neg_control":
            impl, mutate = RS.bind(var, kv_len_bucket=bucket), True
        else:
            raise ValueError(arm)

        try:
            idxs = _call(impl, d, mutate_q=mutate)
        except Exception as exc:  # noqa: BLE001
            row = dict(regime=name, arm=arm, bucket=bucket,
                       error=repr(exc)[:3000])
            out.append(row)
            print(json.dumps(row, default=str), flush=True)
            var.sparsecore_topk = real_sct
            _clear(var)
            continue
        finally:
            var.sparsecore_topk = real_sct

        if arm == "lanecount":
            n, ns, cnt, first, last, cells, below = [int(x)
                                                     for x in idxs[0, :7]]
            row = dict(
                regime=name, arm=arm, bucket=bucket, kv_len=r.kv_len,
                scores_cols=n, sublanes_total=ns,
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
            _clear(var)
            continue

        n, failures, st = TC.check(idxs, oracle, S.INDEX_TOPK)
        row = dict(
            regime=name, arm=arm, bucket=bucket, kv_len=r.kv_len,
            max_model_len=r.max_model_len, rows_checked=n,
            n_failures=len(failures),
            worst_rank=st["worst_rank_seen"],
            allowed_rank=S.INDEX_TOPK + TC.RANK_SLACK,
            min_recall=st["min_recall"], required_recall=TC.MIN_RECALL,
            max_index_returned=int(idxs.max()),
            verdict=("FAIL" if failures else "PASS"),
            first_failure=(failures[0] if failures else None),
        )
        if arm == "incumbent":
            ref_idxs = idxs
        elif ref_idxs is not None:
            row["vs_incumbent"] = _set_equal(ref_idxs, idxs)
        out.append(row)
        print(json.dumps(row, default=str), flush=True)
        _clear(var)

    for row in out:
        row["measured_unwritten_sublanes"] = measured_unwritten
    return out


ALL_ARMS = ("incumbent,right_size,poison,lanecount,neg_control,"
            "neg_control_unwritten")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--bucket-mode", default="ladder",
                    choices=("ladder", "exact"))
    ap.add_argument("--arms", default=ALL_ARMS)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    arms = [x for x in a.arms.split(",") if x]
    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        rows += run_regime(name, mode=a.bucket_mode, arms=arms)

    print(f"\n=== Option B correctness, bucket-mode={a.bucket_mode} "
          "(float64 oracle, rank/recall criterion) ===")
    hdr = (f"{'regime':<16}{'arm':<24}{'bucket':>8}{'unwr':>6}{'verdict':>9}"
           f"{'worst_rank':>12}{'allowed':>9}{'min_recall':>12}"
           f"{'maxidx':>9}{'set_diff':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        u = r.get("measured_unwritten_sublanes")
        us = "?" if u is None else str(u)
        if "error" in r:
            print(f"{r['regime']:<16}{r['arm']:<24}{r['bucket']:>8}{us:>6}"
                  f"{'  ERROR'}")
            continue
        if r["arm"] == "lanecount":
            print(f"{r['regime']:<16}{r['arm']:<24}{r['bucket']:>8}{us:>6}"
                  f"{r['verdict']:>9}"
                  f"   total={r['sublanes_total']} "
                  f"written={r['sublanes_written_measured']} "
                  f"unwritten={r['sublanes_unwritten_measured']} "
                  f"(predicted {r['sublanes_unwritten_predicted']}) "
                  f"first={r['first_unwritten_sublane']} "
                  f"last={r['last_unwritten_sublane']} "
                  f"poison_cells_below_kv_len={r['poison_cells_below_kv_len']}")
            continue
        sd = r.get("vs_incumbent")
        sd_s = "-" if sd is None else str(sd["rows_differing"])
        print(f"{r['regime']:<16}{r['arm']:<24}{r['bucket']:>8}{us:>6}"
              f"{r['verdict']:>9}{r['worst_rank']:>12}{r['allowed_rank']:>9}"
              f"{r['min_recall']:>12.5f}{r['max_index_returned']:>9}"
              f"{sd_s:>10}")

    print("\nREQUIRED VERDICTS: incumbent/right_size/poison = PASS. "
          "lanecount = PASS (measured unwritten count matches the prediction "
          "and no poison survives below kv_len). neg_control = FAIL. "
          "neg_control_unwritten = FAIL wherever the measured unwritten count "
          "is > 0 -- if it PASSES there, the poison arm proves nothing.")

    for r in rows:
        if "error" in r:
            print(f"\nERROR [{r['regime']}/{r['arm']}]: {r['error']}")
        elif r.get("first_failure"):
            print(f"\nFIRST FAILURE [{r['regime']}/{r['arm']}]: "
                  f"{r['first_failure']}")

    red = 0
    for r in rows:
        if "error" in r:
            red += 1
            continue
        arm = r["arm"]
        u = r.get("measured_unwritten_sublanes")
        if arm == "neg_control":
            bad = (r["verdict"] != "FAIL")
        elif arm == "neg_control_unwritten":
            if u is None or u == 0:
                bad = False  # nothing to catch at this shape; not evidence
                print(f"[{r['regime']}] neg_control_unwritten is VACUOUS here "
                      f"(measured unwritten sublanes = {u}); not counted.")
            else:
                bad = (r["verdict"] != "FAIL")
        else:
            bad = (r["verdict"] != "PASS")
        if bad:
            print(f"RED: {r['regime']}/{arm} verdict={r['verdict']}")
        red += bad

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    if red:
        print(f"\n{red} RED result(s).")
    else:
        print("\nAll verdicts as required.")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
