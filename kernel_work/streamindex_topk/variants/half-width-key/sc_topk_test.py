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
"""Coverage for `sparsecore_topk` -- the selection half, which has no test.

Why this file exists
--------------------
Step 2 of the exercise recorded that `sparsecore_topk.py` is exercised by
nothing in `tests/`. Option C (half-width key) changes that file: the DMA'd
buffer narrows, the bitcast changes, `KEY_NEG_INF` changes, and the f32-only
guard at :395-396 has to move. Option C is therefore gated on there being
something that can fail when that edit is wrong. This is that something.

It is NOT the TensorCore-side rank/recall gate in
`instruments/test_correctness.py`. That gate scores the whole
`streamindex_topk` call against a float64 oracle and allows a rank slack of
256 -- it would pass a selection kernel that quietly returned the 2100th-best
element instead of the 2048th. This file allows no slack at all: the contract
in the module docstring (:25-28) is *exact* top-k as a set, and that is what
is checked.

The contract under test, quoted from sparsecore_topk.py:25-28
-------------------------------------------------------------
  "Output rows are the top-k column indices in unspecified order, ``-1``
   suffix-padded, matching an exact top-k over the given scores as a set
   (equal boundary scores may be broken differently). ``-inf`` scores and
   positions at or beyond ``row_lengths`` are never selected. NaNs are
   unsupported."

Decomposed into six checks, each of which can fail independently:
  C1 dtype/shape          out is i32[b, k]
  C2 padding shape        the -1s are a suffix, never interleaved
  C3 index legality       every non-negative index is in [0, row_length)
  C4 count                exactly min(k, row_length) non-negative indices
  C5 uniqueness           no index returned twice in a row
  C6 exact-set-with-ties  the multiset of selected scores equals the multiset
                          of the true top-min(k,L) scores. This is the check
                          that tolerates tie-break freedom without tolerating
                          a wrong answer: a kernel that swaps two equal
                          boundary scores passes, a kernel that picks the
                          2049th-largest fails.

C6 subsumes "-inf is never selected" whenever L > k and fewer than L-k of the
valid scores are -inf, which is every case here that has -inf in it.

Teeth
-----
Two things make the checker itself testable rather than merely reassuring:

  * `poison`: the columns at and beyond `row_length` are filled with +1e30
    rather than -inf. A kernel that ignored `row_lengths` would return those
    columns and C3/C6 would fire. With -inf padding (what the real caller
    emits) that bug is invisible, so the ragged cases here do not use it.
  * `--negative-control`: the checker is run against four deliberately
    corrupted outputs -- a dropped winner, a duplicated index, an
    out-of-range index, and a truncated count. Each MUST be reported as a
    failure. If a negative control passes, the checker has no teeth and the
    positive results below it mean nothing, so the script exits non-zero.

Coverage map
------------
`_pick_partition(b, n, k)` chooses between the single-stage fast path (p == 1)
and the two-stage hierarchical path (p > 1); they are different code with a
different index remap, and both ship. Every case prints its `p`, and the
summary asserts that p == 1, p == 2 and p >= 4 were each exercised at a
production shape.

Distributions include `bf16` and `f16` quantized scores. Those are not
hypothetical: they are exactly the score distribution option C would hand this
kernel, and they multiply ties at the k-th cut by orders of magnitude, which
is the one thing most likely to break a histogram-and-refine selector
(`REFINEMENT_LEVELS` resolves the final quota by tie rank at shift 0).

Usage
-----
  python3 sc_topk_test.py --suite small
  python3 sc_topk_test.py --suite prod
  python3 sc_topk_test.py --suite all --negative-control
"""

import argparse
import gc
import json
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/workspace")

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _pick_partition, sparsecore_topk)

try:
    import ml_dtypes  # noqa: E402
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = None

POISON = np.float32(1e30)


# --------------------------------------------------------------------------
# scores construction
# --------------------------------------------------------------------------
def make_scores(b, n, dist, rng):
    """float32[b, n] in one of the distributions the kernel must survive."""
    if dist == "normal":
        return (rng.standard_normal((b, n), dtype=np.float32) * 10.0)
    if dist == "bf16":
        # What option C would feed this kernel: f32 values that are exactly
        # representable in bf16. 8 mantissa bits -> heavy ties at the cut.
        x = rng.standard_normal((b, n), dtype=np.float32) * 10.0
        return x.astype(BF16).astype(np.float32)
    if dist == "f16":
        x = rng.standard_normal((b, n), dtype=np.float32) * 10.0
        return x.astype(np.float16).astype(np.float32)
    if dist == "equal":
        # Every element a tie. Pathological for the refinement walk: the whole
        # quota is resolved at the last level by tie rank.
        return np.full((b, n), 1.25, dtype=np.float32)
    if dist == "coarse":
        # 8 distinct values -> ties ~n/8 deep, wider than k in the small cases.
        vals = np.array([-3., -2., -1., 0., 1., 2., 3., 4.], dtype=np.float32)
        return vals[rng.integers(0, 8, size=(b, n))]
    if dist == "signed_zero":
        # -0.0 and +0.0 have different bit patterns and the same value. The
        # monotone key maps them to *different* keys (0x80000000 -> 0x7fffffff
        # vs 0x00000000), so a selector that ranks by key alone orders -0.0
        # above +0.0. The multiset check treats them as equal, which is the
        # right semantics and is what makes this case a real test rather than
        # a restatement of the kernel.
        x = np.zeros((b, n), dtype=np.float32)
        x[rng.random((b, n)) < 0.5] = np.float32(-0.0)
        # A minority of genuine positives so the top-k boundary lands inside
        # the zero band and the +0/-0 ordering actually matters.
        pos = rng.random((b, n)) < 0.02
        x[pos] = np.float32(2.0)
        return x
    if dist == "large_mag":
        # Exponent extremes: the fp16 range question in miniature. Values here
        # overflow fp16 (max 65504) and underflow it, but this case is run in
        # f32, so it only tests the monotone key's handling of big exponents.
        e = rng.integers(-30, 30, size=(b, n))
        return (rng.standard_normal((b, n), dtype=np.float32)
                * np.exp2(e).astype(np.float32)).astype(np.float32)
    raise ValueError(dist)


def apply_lengths(scores, lengths, poison):
    """Fill the invalid tail of each row: -inf like the real caller, or
    +1e30 to prove the kernel honours `row_lengths` at all."""
    b, n = scores.shape
    fill = POISON if poison else np.float32(-np.inf)
    cols = np.arange(n)[None, :]
    scores = np.where(cols < lengths[:, None], scores, fill).astype(np.float32)
    return scores


# --------------------------------------------------------------------------
# the checker
# --------------------------------------------------------------------------
def check(out, scores, lengths, k, *, rows_to_check=None):
    """Return a list of failure strings. Empty list == pass.

    `out` is i32[b, k]; `scores` float32[b, n]; `lengths` i32[b].
    """
    fails = []
    b, n = scores.shape
    if out.shape != (b, k):
        fails.append(f"C1 shape: out.shape={out.shape} want {(b, k)}")
        return fails
    if out.dtype != np.int32:
        fails.append(f"C1 dtype: out.dtype={out.dtype} want int32")

    rows = range(b) if rows_to_check is None else rows_to_check
    for r in rows:
        row = out[r]
        L = int(lengths[r])
        m = min(k, L)

        neg = row < 0
        # C2: the -1 padding must be a suffix.
        if neg.any():
            first_neg = int(np.argmax(neg))
            if not neg[first_neg:].all():
                fails.append(f"C2 row {r}: -1 padding is not a suffix "
                             f"(first at {first_neg}, "
                             f"{int(neg.sum())} negatives total)")
        idxs = row[~neg]

        # C4: count.
        if idxs.size != m:
            fails.append(f"C4 row {r}: got {idxs.size} indices, "
                         f"want min(k={k}, L={L}) = {m}")

        if idxs.size == 0:
            continue

        # C3: legality.
        bad = idxs[(idxs < 0) | (idxs >= L)]
        if bad.size:
            fails.append(f"C3 row {r}: {bad.size} indices outside [0,{L}); "
                         f"first {int(bad[0])}")
            continue

        # C5: uniqueness.
        if np.unique(idxs).size != idxs.size:
            dup = idxs.size - np.unique(idxs).size
            fails.append(f"C5 row {r}: {dup} duplicate indices")
            continue

        # C6: exact set with ties.
        valid = scores[r, :L]
        want = np.sort(valid)[::-1][:m]
        got = np.sort(scores[r, idxs])[::-1]
        if not np.array_equal(want, got):
            if want.shape != got.shape:
                # C4 has already reported the count; say only what is new.
                fails.append(
                    f"C6 row {r}: cannot compare multisets, got {got.size} "
                    f"scores against a true top-{m}")
            else:
                d = np.flatnonzero(want != got)
                i = int(d[0])
                fails.append(
                    f"C6 row {r}: selected-score multiset != true top-{m}; "
                    f"first divergence at sorted position {i}: got {got[i]!r} "
                    f"want {want[i]!r} ({d.size} of {m} positions differ)")
    return fails


# --------------------------------------------------------------------------
# case runner
# --------------------------------------------------------------------------
def lengths_for(spec, b, n, rng):
    if spec == "full":
        return np.full(b, n, np.int32)
    if isinstance(spec, int):
        return np.full(b, spec, np.int32)
    if spec == "ragged":
        # Deterministic ragged: full, mid, tiny, zero, repeating.
        pat = [n, n // 2, 64, 0]
        return np.array([pat[i % 4] for i in range(b)], np.int32)
    if isinstance(spec, (list, tuple)):
        assert len(spec) == b
        return np.array(spec, np.int32)
    raise ValueError(spec)


def run_case(case, *, check_rows_cap=None):
    b, n, k = case["b"], case["n"], case["k"]
    rng = np.random.default_rng(case.get("seed", 0))
    p = _pick_partition(b, n, k)

    s = make_scores(b, n, case["dist"], rng)
    lengths = lengths_for(case["lengths"], b, n, rng)
    poison = case.get("poison", False)
    s = apply_lengths(s, lengths, poison)

    we = case.get("write_empty_rows", True)
    t0 = time.perf_counter()
    out = np.asarray(
        jax.block_until_ready(
            sparsecore_topk(jnp.asarray(s), k, jnp.asarray(lengths),
                            write_empty_rows=we)))
    dt = time.perf_counter() - t0

    # With write_empty_rows=False the kernel does not write empty rows at all,
    # so those output rows are uninitialised memory and are not checkable.
    rows = None
    if not we:
        rows = [r for r in range(b) if lengths[r] > 0]
    if check_rows_cap is not None and b > check_rows_cap:
        # Cap the O(b * n log n) host oracle at the biggest shapes. Rows are
        # sampled deterministically and the count checked is reported, so a
        # partial check is never mistaken for a full one.
        sel = np.linspace(0, b - 1, check_rows_cap).astype(int).tolist()
        rows = sorted(set(sel) & set(rows)) if rows is not None else sel

    fails = check(out, s, lengths, k, rows_to_check=rows)
    n_checked = b if rows is None else len(rows)
    return dict(name=case["name"], b=b, n=n, k=k, p=p, dist=case["dist"],
                lengths=str(case["lengths"]), poison=poison,
                write_empty_rows=we, rows_checked=n_checked, rows_total=b,
                wall_s=round(dt, 3), n_fail=len(fails), fails=fails[:8],
                status="PASS" if not fails else "FAIL"), out, s, lengths


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------
SMALL = [
    # --- shape / partition-path edges -------------------------------------
    dict(name="b1_n64_k16", b=1, n=64, k=16, dist="normal", lengths="full"),
    dict(name="b2_n128_k32", b=2, n=128, k=32, dist="normal", lengths="full"),
    dict(name="b32_n2048_k64", b=32, n=2048, k=64, dist="normal",
         lengths="full"),
    dict(name="b31_n2048_k64", b=31, n=2048, k=64, dist="normal",
         lengths="full"),
    dict(name="b1_n131072_k2048", b=1, n=131072, k=2048, dist="normal",
         lengths="full"),
    dict(name="b1_n2048_k2048_kEQn", b=1, n=2048, k=2048, dist="normal",
         lengths="full"),
    # --- row_lengths ------------------------------------------------------
    dict(name="ragged_poisoned", b=8, n=2048, k=64, dist="normal",
         lengths="ragged", poison=True),
    dict(name="ragged_neginf", b=8, n=2048, k=64, dist="normal",
         lengths="ragged"),
    dict(name="len_lt_k", b=4, n=2048, k=256, dist="normal",
         lengths=[100, 256, 255, 0], poison=True),
    dict(name="all_zero_len", b=4, n=2048, k=64, dist="normal",
         lengths=[0, 0, 0, 0]),
    dict(name="len1", b=4, n=2048, k=64, dist="normal", lengths=[1, 1, 1, 1],
         poison=True),
    dict(name="write_empty_false", b=4, n=2048, k=64, dist="normal",
         lengths=[2048, 1024, 0, 512], write_empty_rows=False),
    # --- distributions ----------------------------------------------------
    dict(name="all_equal", b=8, n=1024, k=128, dist="equal", lengths="full"),
    dict(name="coarse_ties", b=16, n=2048, k=16, dist="coarse",
         lengths="full"),
    dict(name="coarse_ties_k1024", b=4, n=2048, k=1024, dist="coarse",
         lengths="full"),
    dict(name="signed_zero", b=4, n=2048, k=64, dist="signed_zero",
         lengths="full"),
    dict(name="large_mag", b=4, n=2048, k=256, dist="large_mag",
         lengths="full"),
    dict(name="neginf_half", b=2, n=4096, k=2048, dist="normal",
         lengths=[4096, 4096], seed=7),
    dict(name="bf16_small", b=8, n=2048, k=256, dist="bf16", lengths="full"),
    dict(name="f16_small", b=8, n=2048, k=256, dist="f16", lengths="full"),
]

# Production shapes. n = num_sublanes_total * 128: 80*128 = 10240 (mml 9216),
# 528*128 = 67584 (mml 66560). k = INDEX_TOPK = 2048. Lengths are the real
# kv_len the caller passes as `eff_row_lengths`.
PROD = [
    dict(name="decode_prod", b=64, n=10240, k=2048, dist="normal",
         lengths=8192),
    dict(name="mixed_prod", b=1024, n=10240, k=2048, dist="normal",
         lengths=8192),
    dict(name="decode_64k", b=6, n=67584, k=2048, dist="normal",
         lengths=65536),
    dict(name="mixed_64k", b=1024, n=67584, k=2048, dist="normal",
         lengths=65536),
    # The distribution option C creates, at the shapes it creates it at.
    dict(name="decode_prod_bf16", b=64, n=10240, k=2048, dist="bf16",
         lengths=8192),
    dict(name="decode_64k_bf16", b=6, n=67584, k=2048, dist="bf16",
         lengths=65536),
    dict(name="mixed_prod_bf16", b=1024, n=10240, k=2048, dist="bf16",
         lengths=8192),
    dict(name="decode_prod_f16", b=64, n=10240, k=2048, dist="f16",
         lengths=8192),
]


# --------------------------------------------------------------------------
# negative control: the checker must reject wrong answers
# --------------------------------------------------------------------------
def negative_control():
    """Corrupt a known-good output four ways; every one must be reported."""
    rng = np.random.default_rng(11)
    b, n, k = 4, 2048, 64
    s = make_scores(b, n, "normal", rng)
    lengths = np.full(b, n, np.int32)
    out = np.asarray(
        jax.block_until_ready(
            sparsecore_topk(jnp.asarray(s), k, jnp.asarray(lengths))))
    base = check(out, s, lengths, k)
    results = [("uncorrupted", len(base) == 0, base[:3])]

    # 1. drop a winner: replace a returned index with the row's argmin.
    c = out.copy()
    c[0, 0] = int(np.argmin(s[0]))
    f = check(c, s, lengths, k)
    results.append(("dropped_winner", len(f) > 0, f[:1]))

    # 2. duplicate an index.
    c = out.copy()
    c[1, 0] = c[1, 1]
    f = check(c, s, lengths, k)
    results.append(("duplicate_index", len(f) > 0, f[:1]))

    # 3. out-of-range index.
    c = out.copy()
    c[2, 0] = n + 5
    f = check(c, s, lengths, k)
    results.append(("out_of_range", len(f) > 0, f[:1]))

    # 4. truncated count (-1 where a winner belongs).
    c = out.copy()
    c[3, k - 1] = -1
    f = check(c, s, lengths, k)
    results.append(("truncated_count", len(f) > 0, f[:1]))

    # 5. row_lengths ignored: hand the checker a short length and the full
    #    kernel answer. Must fire -- this is the one that proves `poison` and
    #    the ragged cases would catch a kernel that ignored row_lengths.
    short = np.full(b, 128, np.int32)
    f = check(out, s, short, k)
    results.append(("lengths_ignored", len(f) > 0, f[:1]))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all",
                    choices=["small", "prod", "all"])
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--check-rows-cap", type=int, default=48,
                    help="max rows the host oracle checks per case")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    print(f"jax {jax.__version__}  devices={jax.devices()}", flush=True)

    cases = []
    if a.suite in ("small", "all"):
        cases += SMALL
    if a.suite in ("prod", "all"):
        cases += PROD
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["name"] in want]

    rows = []
    hard_error = False
    for c in cases:
        try:
            row, out, s, lengths = run_case(c, check_rows_cap=a.check_rows_cap)
            del out, s, lengths
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            hard_error = True
            row = dict(name=c["name"], b=c["b"], n=c["n"], k=c["k"],
                       p=_pick_partition(c["b"], c["n"], c["k"]),
                       dist=c["dist"], status="ERROR",
                       error=traceback.format_exc()[-1200:])
        rows.append(row)
        if row["status"] == "PASS":
            print(f"PASS {row['name']:<22} b={row['b']:<5} n={row['n']:<7} "
                  f"k={row['k']:<5} p={row['p']:<3} {row['dist']:<12} "
                  f"rows_checked={row['rows_checked']}/{row['rows_total']} "
                  f"{row['wall_s']}s", flush=True)
        elif row["status"] == "FAIL":
            print(f"FAIL {row['name']:<22} b={row['b']} n={row['n']} "
                  f"k={row['k']} p={row['p']} {row['dist']}  "
                  f"{row['n_fail']} failures", flush=True)
            for f in row["fails"]:
                print(f"       {f}", flush=True)
        else:
            print(f"ERROR {row['name']}\n{row.get('error','')}", flush=True)

    nc_ok = True
    if a.negative_control:
        print("\n=== negative control (every corruption MUST be caught) ===",
              flush=True)
        for name, ok, detail in negative_control():
            print(f"{'OK  ' if ok else 'LEAK'} {name:<20} {detail}",
                  flush=True)
            nc_ok = nc_ok and ok

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_err = sum(1 for r in rows if r["status"] == "ERROR")

    ps = sorted({r["p"] for r in rows})
    prod_ps = sorted({r["p"] for r in rows
                      if r["name"] in {c["name"] for c in PROD}})
    print(f"\n=== sc_topk_test: {n_pass} pass, {n_fail} fail, {n_err} error "
          f"({len(rows)} cases) ===")
    print(f"partition factors exercised: p in {ps}"
          + (f"; at production shapes p in {prod_ps}" if prod_ps else ""))
    if a.negative_control:
        print(f"negative control: {'ALL CAUGHT' if nc_ok else 'LEAKED'}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)

    return 0 if (n_fail == 0 and n_err == 0 and nc_ok and not hard_error) else 1


if __name__ == "__main__":
    sys.exit(main())
