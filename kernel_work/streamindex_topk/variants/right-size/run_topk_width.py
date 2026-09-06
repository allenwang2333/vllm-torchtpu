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
"""What narrowing the scores buffer does to `sparsecore_topk` alone.

Design section 4.3(b) puts the SparseCore side of Option B's confound-shape
prize at a RANGE, 0.7-0.9 ms `modelled`, and says so because area 8's streaming
floor runs 3.2-7.2x optimistic against the `topk ctrl` probe and the two
derivation routes only agree to within their own precision. This script is the
measurement that replaces that range with a number.

It also records something the roofline had no term for: `_pick_partition`
(`sparsecore_topk.py:360`) chooses the number of row slices from `n` itself, so
narrowing the buffer can move the kernel between its one-stage and two-stage
paths. That is a code-path change in a kernel with no unit test, not just fewer
bytes, and it is why `run_sc_coverage.py` exists next to this file.

Standalone `sparsecore_topk` timing is a COMPONENT, not a result -- see the
`microbench` docstring. It is reported only as an attribution of the full-call
delta that `run_perf.py` measures.
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from kernel_work.streamindex_topk.instruments import microbench as MB
from kernel_work.streamindex_topk.instruments import shapes as S
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as SC

import _rs_support as RS


def one(b, n, live, k, reps, inner):
    """Time `sparsecore_topk` on a b x n buffer with `live` real columns."""
    rng = np.random.default_rng(1234)
    arr = np.full((b, n), -np.inf, dtype=np.float32)
    lv = min(live, n)
    arr[:, :lv] = (rng.standard_normal((b, lv)) * 10.0).astype(np.float32)
    scores = jnp.asarray(arr)
    row_lengths = jnp.full((b, ), lv, jnp.int32)
    fn = jax.jit(lambda s, r: SC.sparsecore_topk(
        s, k, row_lengths=r, write_empty_rows=True))
    ms, spread = MB._median_spread(
        MB._time(fn, scores, row_lengths, reps=reps, inner=inner))
    return dict(b=b, n=n, live=lv, k=k,
                partition_p=SC._pick_partition(b, n, k),
                two_stage=SC._pick_partition(b, n, k) > 1,
                bytes_MB=b * n * 4 / 1e6,
                ms=ms, spread_pct=spread, inner=inner, reps=reps)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--reps", type=int, default=MB.REPS)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    k = S.INDEX_TOPK
    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        r = RS.regime(name)
        n_inc, n_rs = RS.widths(r)
        b = r.num_tokens
        inner = max(MB.MIN_INNER,
                    min(MB.INNER, int(MB.INFLIGHT_BUDGET_BYTES //
                                      max(1, b * n_inc * 4))))
        pair = {}
        for arm, n in (("incumbent", n_inc), ("right_size", n_rs)):
            try:
                row = one(b, n, r.kv_len, k, a.reps, inner)
            except Exception as exc:  # noqa: BLE001
                row = dict(error=repr(exc)[:2000])
            row["regime"] = name
            row["arm"] = arm
            rows.append(row)
            pair[arm] = row
            print(json.dumps(row), flush=True)
        if "error" in pair["incumbent"] or "error" in pair["right_size"]:
            continue
        old, new = pair["incumbent"]["ms"], pair["right_size"]["ms"]
        d = dict(kind="delta", regime=name, incumbent_ms=old, right_size_ms=new,
                 incumbent_spread_pct=pair["incumbent"]["spread_pct"],
                 right_size_spread_pct=pair["right_size"]["spread_pct"],
                 incumbent_n=n_inc, right_size_n=n_rs,
                 incumbent_p=pair["incumbent"]["partition_p"],
                 right_size_p=pair["right_size"]["partition_p"],
                 d_time_pct=(new - old) / old * 100.0, speedup_x=old / new)
        rows.append(d)
        print(json.dumps(d), flush=True)

    print("\n=== sparsecore_topk alone, incumbent width vs right-sized width ===")
    print("COMPONENT, not a result. d time % negative is faster.")
    hdr = (f"{'regime':<16}{'n_inc':>8}{'p':>3}{'inc_ms':>10}{'sp%':>8}"
           f"{'n_rs':>8}{'p':>3}{'rs_ms':>10}{'sp%':>8}{'d time %':>10}"
           f"{'speedup x':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("kind") != "delta":
            continue
        print(f"{r['regime']:<16}{r['incumbent_n']:>8}{r['incumbent_p']:>3}"
              f"{r['incumbent_ms']:>10.5f}{r['incumbent_spread_pct']:>8.2f}"
              f"{r['right_size_n']:>8}{r['right_size_p']:>3}"
              f"{r['right_size_ms']:>10.5f}{r['right_size_spread_pct']:>8.2f}"
              f"{r['d_time_pct']:>10.2f}{r['speedup_x']:>11.3f}")
    for r in rows:
        if "error" in r:
            print(f"\nERROR [{r.get('regime')}/{r.get('arm')}]: {r['error']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
