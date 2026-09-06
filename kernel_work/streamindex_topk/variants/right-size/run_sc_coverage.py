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
"""Coverage for the part of `sparsecore_topk` Option B can break.

`measurement_and_design.md` section 11 item 4: `grep -rn "sparsecore_topk"
tests/` returns nothing. There is no unit test under this kernel at all, and
the `streamindex_topk` rank/recall gate does not stand in for one -- see the
list at the bottom of this docstring for what that gate cannot see.

Option B does not edit `sparsecore_topk.py`. It changes the `n` that kernel is
DRIVEN at, and `n` is not an inert parameter there:

  * `_pick_partition` (:360) derives the per-row slice count P from `n`, so a
    narrower buffer can move the call between the one-stage direct path
    (`_sc_topk_direct` alone) and the two-stage path (stage-1 slices, a gather,
    stage-2 merge). At the confound shape it does exactly that: 67584 -> P=2,
    8192 -> P=1. That is a different code path, not just fewer bytes.
  * `slice_len`, the stage-1 reshape `b*P x n/P`, and the resident-VMEM DMA
    extent all move with it.

So the thing to test is not the source, it is the shapes: run the kernel at
exactly the widths and partition factors B produces, against a numpy oracle,
with the padded region filled with LARGE POSITIVE garbage rather than `-inf`.
That last detail is the point. B's whole safety argument is that
`valid = (j < i_hi)` with `i_hi = min(slice_len, row_lengths[row])` (:151-155,
:171) makes positions at or past `row_lengths` unselectable no matter what is
in them. `-inf` padding would pass whether or not that argument holds. Garbage
that would be selected on sight passes only if it holds.

Ragged and empty rows are included because `_effective_row_lengths`
(`streamindex_topk.py:592`) emits per-row lengths and emits 0 for every token
row past `cu_q_lens[num_seqs]`, which is the `write_empty_rows` path.

What this still does not cover, stated rather than papered over:
  * the SparseCore half of Option B as DESIGNED (section 4.3(b)'s padded read
    sized to `row_lengths`) -- that needs a source change to
    `sparsecore_topk.py`, which section 4.6 forbids before this coverage
    exists. This file is that coverage; the source change is not made here.
  * numerical behaviour under real (score-shaped, near-tied) data at these
    widths beyond what the random draw happens to produce.
  * anything about the kernel's own performance; that is `run_topk_width.py`.
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from kernel_work.streamindex_topk.instruments import shapes as S
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as SC

GARBAGE = 1e30

# (label, b, n, live, ragged) -- every one of these is a width some arm of the
# Option B measurement actually drives the kernel at.
CASES = [
    ("confound/incumbent", 1024, 67584, 8192, False),
    ("confound/right_size", 1024, 8192, 8192, False),
    ("confound/right_size ragged", 1024, 8192, 8192, True),
    ("mixed_prod/incumbent", 1024, 10240, 8192, False),
    ("mixed_prod/right_size", 1024, 8192, 8192, True),
    ("mask8000/right_size", 1024, 8064, 8000, True),
    ("mixed_64k/incumbent", 1024, 67584, 65536, False),
    ("mixed_64k/right_size", 1024, 65536, 65536, False),
    ("decode_prod/incumbent", 64, 10240, 8192, False),
    ("decode_prod/right_size", 64, 8192, 8192, True),
    ("decode_64k/incumbent", 6, 67584, 65536, False),
    ("decode_64k/right_size", 6, 65536, 65536, False),
]


def oracle_sets(arr, lengths, k):
    """Per-row set of top-k column indices over the row's live prefix."""
    out = []
    for t in range(arr.shape[0]):
        L = int(lengths[t])
        if L == 0:
            out.append(set())
            continue
        row = arr[t, :L]
        kk = min(k, L)
        idx = np.argpartition(-row, kk - 1)[:kk]
        out.append(set(int(i) for i in idx))
    return out


def one(label, b, n, live, ragged, k, seed=11):
    rng = np.random.default_rng(seed)
    arr = np.full((b, n), GARBAGE, dtype=np.float32)
    arr[:, :live] = (rng.standard_normal((b, live)) * 10.0).astype(np.float32)
    if ragged:
        # Vary the live length per row and zero out a tail of rows, which is
        # exactly the shape `_effective_row_lengths` emits for a real batch.
        lengths = rng.integers(low=max(1, live // 4), high=live + 1,
                               size=b).astype(np.int32)
        lengths[: max(1, b // 16)] = 0
        # Below its own length a row must still be real data; above it, garbage
        # that must never be selected.
        for t in range(b):
            L = int(lengths[t])
            if L < live:
                arr[t, L:] = GARBAGE
    else:
        lengths = np.full((b, ), live, np.int32)

    got = np.asarray(jax.block_until_ready(SC.sparsecore_topk(
        jnp.asarray(arr), k, row_lengths=jnp.asarray(lengths),
        write_empty_rows=True)))

    want = oracle_sets(arr, lengths, k)
    bad_rows = 0
    bad_count = 0
    garbage_selected = 0
    first = None
    for t in range(b):
        L = int(lengths[t])
        g = got[t]
        gv = set(int(x) for x in g if x >= 0)
        if any(x >= L for x in gv):
            garbage_selected += 1
        if gv != want[t]:
            bad_rows += 1
            bad_count += len(gv ^ want[t])
            if first is None:
                over = sorted(x for x in gv if x >= L)[:4]
                first = (f"row {t}: L={L} returned {len(gv)} want "
                         f"{len(want[t])} symdiff {len(gv ^ want[t])} "
                         f"indices>=L {over}")
    return dict(
        case=label, b=b, n=n, live=live, ragged=ragged, k=k,
        partition_p=SC._pick_partition(b, n, k),
        two_stage=SC._pick_partition(b, n, k) > 1,
        rows_wrong=bad_rows, total_symmetric_difference=bad_count,
        rows_selecting_past_row_length=garbage_selected,
        verdict=("PASS" if bad_rows == 0 and garbage_selected == 0 else "FAIL"),
        first_failure=first,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on the label")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    k = S.INDEX_TOPK
    rows = []
    for label, b, n, live, ragged in CASES:
        if a.only and a.only not in label:
            continue
        try:
            row = one(label, b, n, live, ragged, k)
        except Exception as exc:  # noqa: BLE001
            row = dict(case=label, b=b, n=n, error=repr(exc)[:3000])
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)

    print("\n=== sparsecore_topk direct oracle coverage at Option B's widths ===")
    print("Padding is +1e30, NOT -inf: a row that reads past its row_length "
          "selects garbage and fails.")
    hdr = (f"{'case':<30}{'b':>6}{'n':>8}{'live':>8}{'rag':>5}{'P':>3}"
           f"{'verdict':>9}{'rows_wrong':>12}{'past_len':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<30}  ERROR")
            continue
        print(f"{r['case']:<30}{r['b']:>6}{r['n']:>8}{r['live']:>8}"
              f"{str(r['ragged'])[:1]:>5}{r['partition_p']:>3}"
              f"{r['verdict']:>9}{r['rows_wrong']:>12}"
              f"{r['rows_selecting_past_row_length']:>10}")
    for r in rows:
        if "error" in r:
            print(f"\nERROR [{r['case']}]: {r['error']}")
        elif r.get("first_failure"):
            print(f"\nFIRST FAILURE [{r['case']}]: {r['first_failure']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
