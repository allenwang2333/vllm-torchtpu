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
"""Correctness gate for variant A ("Narrow Record").

`measurement_and_design.md` section 3.4 asks for three things, and this runs
all three:

  1. BIT-EXACT EQUALITY against the incumbent. The variant changes only which
     bytes cross HBM, not the arithmetic, so the returned index arrays must be
     identical element for element. This is the strong check: it needs no
     tolerance and no oracle, and it fails on any reordering, any dropped
     scale and any mis-indexed page.
  2. The oracle rank criterion from the READ-ONLY
     `instruments/test_correctness.py` -- `RANK_SLACK=256`,
     `MIN_RECALL=0.99` -- run on the VARIANT, so the variant is judged by the
     same absolute standard step 1 judged the incumbent by, not only relative
     to it.
  3. The negative control, on the variant, at `decode_prod` and `mixed_prod`.
     A check that has never failed is not evidence: if the mutated variant
     passes, checks 1 and 2 are vacuous for this variant and must not be
     quoted.

Every `worst_rank=` and `min_recall=` the instrument computes is printed, on
pass as well as on fail, so the margin can be read rather than assumed.
"""

import argparse
import sys

import _load  # noqa: F401  -- puts the repo root on sys.path


def _call(impl, d, S):
    import jax
    import numpy as np
    idxs = impl.streamindex_topk(
        d["q"], d["indexer_weights"], d["cache_kv"], d["seq_lens"],
        d["page_indices"], d["cu_q_lens"], d["distribution"],
        k=S.INDEX_TOPK,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=False,
    )
    return np.asarray(jax.block_until_ready(idxs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="decode_prod,decode_64k,mixed_prod")
    ap.add_argument("--control-regimes", default="decode_prod,mixed_prod")
    a = ap.parse_args(argv)

    import jax
    import numpy as np

    from kernel_work.streamindex_topk.instruments import inputs as I
    from kernel_work.streamindex_topk.instruments import shapes as S
    from kernel_work.streamindex_topk.instruments import test_correctness as TC
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as incumbent

    narrow_inputs = _load.load("narrow_inputs")
    variant = _load.load("streamindex_topk_narrow")

    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    print(f"RANK_SLACK={TC.RANK_SLACK} MIN_RECALL={TC.MIN_RECALL} "
          f"k={S.INDEX_TOPK}", flush=True)

    rc = 0

    # --- 1. bit-exact equality, same input, both impls ---------------------
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        d = I.build(regime, seed=0)
        base_idx = _call(incumbent, d, S)
        d_split = dict(d)
        d_split["cache_kv"] = narrow_inputs.split_cache(d["cache_kv"])
        var_idx = _call(variant, d_split, S)
        same = bool(np.array_equal(base_idx, var_idx))
        n_diff = int((base_idx != var_idx).sum())
        print(f"[{name}] EXACT-EQUALITY {'PASS' if same else 'FAIL'} -- "
              f"shape={base_idx.shape} dtype={base_idx.dtype}, "
              f"{n_diff} of {base_idx.size} elements differ", flush=True)
        if not same:
            rc = 1
            bad = np.argwhere(base_idx != var_idx)[:5]
            for t, j in bad:
                print(f"    [{t},{j}] incumbent={base_idx[t, j]} "
                      f"variant={var_idx[t, j]}", flush=True)
        del d, d_split, base_idx, var_idx

    # --- 2. oracle rank criterion on the variant ---------------------------
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        with narrow_inputs.patched_inputs():
            n, failures, st = TC.run(variant, regime)
        margin = (f"worst_rank={st['worst_rank_seen']} of allowed "
                  f"{S.INDEX_TOPK + TC.RANK_SLACK}, "
                  f"min_recall={st['min_recall']:.5f} of required "
                  f"{TC.MIN_RECALL}")
        if failures:
            print(f"[{name}] ORACLE FAIL -- {len(failures)} of {n} rows bad "
                  f"({margin})", flush=True)
            for f in failures[:5]:
                print("    ", f, flush=True)
            rc = 1
        else:
            print(f"[{name}] ORACLE PASS -- all {n} rows are a valid top-"
                  f"{S.INDEX_TOPK} against the float64 oracle. {margin}",
                  flush=True)

    # --- 3. negative control on the variant --------------------------------
    for name in [n for n in a.control_regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        with narrow_inputs.patched_inputs():
            n, failures, st = TC.run(variant, regime, negative_control=True)
        margin = (f"worst_rank={st['worst_rank_seen']} of allowed "
                  f"{S.INDEX_TOPK + TC.RANK_SLACK}, "
                  f"min_recall={st['min_recall']:.5f} of required "
                  f"{TC.MIN_RECALL}")
        if failures:
            print(f"[{name}] NEGATIVE CONTROL OK -- check fired on a mutated "
                  f"variant ({len(failures)} of {n} rows bad). {margin}. "
                  f"first: {failures[0]}", flush=True)
        else:
            print(f"[{name}] NEGATIVE CONTROL DID NOT FIRE -- the mutated "
                  f"variant PASSED over {n} rows ({margin}). The checks above "
                  f"are vacuous and must not be quoted as evidence.",
                  flush=True)
            rc = 1

    print(f"\ncorrectness rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
