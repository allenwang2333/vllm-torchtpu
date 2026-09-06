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
"""What the extra static widths cost at compile time. Step 3 left this
`unmodelled`; this gives it a number.

`kv_len_bucket` is a static argument, so every distinct value is a distinct
Mosaic kernel: `pallas.py:246-262` hashes static args by value into the
invocation key and `pallas.py:618-660` re-exports, re-lowers, re-fingerprints
and re-registers on an unseen key, keeping one resident kernel per value in an
unbounded per-op dict. The question this file answers is how much wall clock
and how many kernels that is.

It matters differently for the two bucket modes:

  static  ONE bucket for the whole deployment, chosen from `max_model_len` at
          op-build time. It REPLACES the incumbent's single compile rather
          than adding to it, so the marginal cost is (composed compile -
          incumbent compile), which is what the `delta_s` column shows.
  exact   one compile per rung. At 8192 granularity up to the 65536 cap that
          is 8 rungs, and this file prices all of them so the number exists if
          anyone builds the per-step plumbing.

Wall clock here is AOT `.lower().compile()`, measured on a contended 4-chip pod
shard. It is a COMPILE-TIME number and belongs nowhere near a throughput
table; treat it as an order of magnitude, not a benchmark.
"""

import argparse
import json
import sys
import time

import _support as SUP


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="mixed_64k")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import jax

    from kernel_work.streamindex_topk.instruments import inputs as I
    from kernel_work.streamindex_topk.instruments import shapes as S

    SUP.banner()
    incumbent = SUP.load_incumbent()
    composed = SUP.load_composed()
    narrow = SUP.load_narrow_inputs()

    r = SUP.regime(a.regime)
    static_bucket = SUP.bucket_for(r, composed, "static")
    rungs = composed.bucket_ladder(r.max_model_len, S.INDEX_TOPK)
    print(f"\nregime={r.name} max_model_len={r.max_model_len}", flush=True)
    print(f"static bucket (WHAT SHIPS, one value)      : {static_bucket}",
          flush=True)
    print(f"per-step ladder rungs (NOT reachable today): {rungs}", flush=True)

    d_packed = I.build(r)
    with narrow.patched_inputs():
        d_split = I.build(r)

    def kwargs():
        return dict(k=S.INDEX_TOPK, compression_ratio=S.COMPRESSION_RATIO,
                    num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
                    num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
                    decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
                    enable_early_exit=False)

    def args_of(d):
        return (d["q"], d["indexer_weights"], d["cache_kv"], d["seq_lens"],
                d["page_indices"], d["cu_q_lens"], d["distribution"])

    def compile_once(impl, d, bucket):
        extra = {} if bucket is None else dict(kv_len_bucket=bucket)
        # A fresh closure each call so nothing is served from a jit cache.
        fn = jax.jit(lambda *xs: impl.streamindex_topk(*xs, **kwargs(),
                                                       **extra))
        t0 = time.perf_counter()
        c = fn.lower(*args_of(d)).compile()
        t1 = time.perf_counter()
        del c
        return t1 - t0

    rows = []

    # Warm the compiler once so the first measured entry is not paying for
    # process-level one-time setup.
    compile_once(incumbent, d_packed, None)

    t_inc = compile_once(incumbent, d_packed, None)
    rows.append(dict(arm="incumbent", bucket=None, compile_s=t_inc))
    print(json.dumps(rows[-1]), flush=True)

    for bucket in rungs:
        try:
            t = compile_once(composed, d_split, bucket)
            row = dict(arm="composed", bucket=bucket, compile_s=t,
                       delta_s=t - t_inc,
                       is_static_shipping_bucket=(bucket == static_bucket))
        except Exception as exc:  # noqa: BLE001
            row = dict(arm="composed", bucket=bucket, error=repr(exc)[:2000])
        rows.append(row)
        print(json.dumps(row), flush=True)

    print("\n=== compile cost per distinct static bucket (AOT "
          "lower+compile, contended shard) ===")
    hdr = f"{'arm':<11}{'bucket':>9}{'compile_s':>11}{'delta_s':>10}  ships"
    print(hdr)
    print("-" * (len(hdr) + 2))
    for row in rows:
        if "error" in row:
            print(f"{row['arm']:<11}{row['bucket']:>9}  ERROR "
                  f"{row['error'][:70]}")
            continue
        b = "-" if row["bucket"] is None else str(row["bucket"])
        ds = row.get("delta_s")
        print(f"{row['arm']:<11}{b:>9}{row['compile_s']:>11.2f}"
              f"{('--' if ds is None else f'{ds:+.2f}'):>10}  "
              f"{'<== SHIPS' if row.get('is_static_shipping_bucket') else ''}")

    ok = [row for row in rows if row["arm"] == "composed" and "error" not in row]
    if ok:
        total = sum(row["compile_s"] for row in ok)
        print(f"\nAll {len(ok)} rungs together: {total:.1f} s. The SHIPPING "
              f"configuration compiles ONE of them and replaces the "
              f"incumbent's {t_inc:.2f} s, so its marginal warmup cost is the "
              f"`delta_s` on the `<== SHIPS` row, not this total.")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
    errs = [row for row in rows if "error" in row]
    print(f"\nRECOMPILE DONE errors={len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
