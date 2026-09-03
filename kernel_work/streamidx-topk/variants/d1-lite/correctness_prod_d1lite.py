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
"""D1-lite correctness at the *production* configuration.

`test_d1lite_equivalence.py` runs a wide small-shape grid at k=64.  This script
runs the exact configuration the perf tables are measured at -- batch 64,
`max_model_len` 9216, `page_size` 1024, `k=2048`, `compression_ratio=1`,
`num_kv_pages_per_block=(2,2,2)`, `num_queries_per_block=(1,128,128)` -- across
the three contexts the tables use, so the correctness evidence and the perf
evidence are taken at the same shape rather than at a proxy for it.

  ctx 1024, 2048 : `max(seq_lens) // compression_ratio <= k` is TRUE -> the
                   early-exit `_fast_path` fires.
  ctx 8704       : FALSE -> `_common_path` runs.

Every comparison is `np.array_equal` against a separately-constructed device
call to the *incumbent* `streamindex_topk` with `enable_early_exit=True`, which
is what production runs.  Zero tolerance: D1-lite moves where the metadata is
computed and nothing else, so any difference at all is a failure.

Exit code 0 iff every comparison passed.
"""

from __future__ import annotations

import json
import os
import sys

INSTRUMENTS = os.environ.get("DL_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)
VARIANT = os.environ.get("DL_VARIANT",
                         os.path.dirname(os.path.abspath(__file__)))
if VARIANT not in sys.path:
    sys.path.insert(0, VARIANT)

import numpy as np  # noqa: E402

import bench_streamidx_topk as B  # noqa: E402
import streamindex_topk_d1lite as DL  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402

CONTEXTS = (1024, 2048, 8704)
BATCH = 64
HOISTS = ("none", "plain", "cond")
STATIC = dict(
    k=B.K_IDX,
    compression_ratio=B.COMPRESSION_RATIO,
    num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
    num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
)


def log(msg: str) -> None:
    print(f"[d1lite:correctness-prod] {msg}", flush=True)


def fast_path_closed_form(seq_lens, cu, num_tokens, k, cr):
    """`streamindex_topk.py:840-856`, recomputed in numpy."""
    token_idx = np.arange(num_tokens)
    seq_idx = np.minimum(np.searchsorted(cu[1:], token_idx, side="right"),
                         seq_lens.shape[0] - 1)
    seq_len = seq_lens[seq_idx]
    q_len = cu[seq_idx + 1] - cu[seq_idx]
    q_start = cu[seq_idx]
    q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
    max_valid = np.minimum(seq_len // cr - 1, q_abs_pos // cr)
    max_valid = np.where(token_idx < cu[-1], max_valid, -1)
    s_idx = np.arange(k, dtype=np.int32)[None, :]
    return np.where(s_idx <= max_valid[:, None], s_idx, -1).astype(np.int32)


def main() -> int:
    import jax  # noqa: F401

    results = []
    failures = 0
    for ctx in CONTEXTS:
        shape = B.Shape(f"decode_b{BATCH}_ctx{ctx}", "decode", BATCH, 1, ctx,
                        max_model_len=B.PROD_MAX_MODEL_LEN)
        geo = B.geometry(shape)
        inp = B.build_inputs(shape, geo, seed=0)
        fires = (ctx // B.COMPRESSION_RATIO) <= B.K_IDX
        log(f"ctx={ctx} num_tokens={geo.num_tokens} "
            f"num_positions={geo.num_positions} k={B.K_IDX} "
            f"fast_path_predicate={fires}")

        base = np.asarray(
            streamindex_topk(**inp, enable_early_exit=True, **STATIC))

        # Independent evidence that the fast path is or is not being taken.
        cf = fast_path_closed_form(np.asarray(inp["seq_lens"]),
                                   np.asarray(inp["cu_q_lens"]),
                                   geo.num_tokens, B.K_IDX,
                                   B.COMPRESSION_RATIO)
        base_is_closed_form = bool(np.array_equal(base, cf))
        log(f"ctx={ctx} incumbent output == _fast_path closed form: "
            f"{base_is_closed_form} (expected {fires})")
        if base_is_closed_form != fires:
            log(f"ctx={ctx} FAIL: fast-path regime is not what the predicate "
                f"says; the perf arms are not measuring what they claim")
            failures += 1

        for hoist in HOISTS:
            got = np.asarray(
                DL.streamindex_topk_dl(**inp,
                                       enable_early_exit=True,
                                       hoist=hoist,
                                       **STATIC))
            ok = bool(got.shape == base.shape and np.array_equal(got, base))
            ndiff = int((got != base).sum()) if got.shape == base.shape else -1
            results.append({"ctx": ctx, "arm": f"dl_{hoist}", "equal": ok,
                            "n_differing": ndiff, "shape": list(got.shape)})
            log(f"ctx={ctx} hoist={hoist:<5} array_equal_to_incumbent={ok} "
                f"n_differing={ndiff}")
            if not ok:
                failures += 1

        # The D1 arm, for reference: it is NOT expected to match on a firing
        # context, and that mismatch is the whole reason D1-lite exists.
        d1 = np.asarray(
            streamindex_topk(**inp, enable_early_exit=False, **STATIC))
        d1_ok = bool(np.array_equal(d1, base))
        results.append({"ctx": ctx, "arm": "d1", "equal": d1_ok,
                        "n_differing": int((d1 != base).sum()),
                        "shape": list(d1.shape), "informational": True})
        log(f"ctx={ctx} arm=d1    array_equal_to_incumbent={d1_ok} "
            f"(informational: on a firing ctx the two take different paths "
            f"and may legitimately differ in ORDER only)")

        del base, d1, inp
        import gc
        gc.collect()

    out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "d1lite")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "correctness_prod.json"), "w") as fh:
        json.dump({"results": results, "failures": failures}, fh, indent=1)
    log(f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
