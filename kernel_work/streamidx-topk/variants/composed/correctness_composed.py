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
"""Index-identity of the composed candidate against the incumbent.

Every comparison is `np.array_equal` against a **separately constructed device
call** to the incumbent `streamindex_topk` from `src/`, with
`enable_early_exit=True` -- what production runs.  Zero tolerance: the composed
candidate moves where metadata is computed and changes the *order* the MXU
accumulates the head axis in on one of three launches, so a difference in the
returned indices is a failure, not a rounding artifact.

## Why the grid is shaped like this

Option C only touches the **MIXED** launch.  A decode-only grid would run the
head-major kernel zero times and pass while proving nothing, which is the most
plausible way this check could be hollow.  So the grid deliberately spans:

* **prefill / mixed shapes**, where `bq_sz = num_queries_per_block[2] = 128`
  and the head-major kernel actually executes -- including two token counts
  that are *not* multiples of 128, so the partial-`bq` block and its `sz`
  clamp are exercised, and one two-sequence batch;
* the **production decode point** (batch 64, ctx 8704, `max_model_len` 9216),
  where the head-major kernel must *not* execute and the DECODE launches must
  stay byte-for-byte the incumbent's;
* two **fast-path-firing** shapes (ctx 1024 and 2048 at k=2048), where the
  `lax.cond` takes `_fast_path` and no scoring kernel runs at all.  The
  incumbent's own output is additionally checked against a numpy recomputation
  of `_fast_path`'s closed form, so "the fast path fired" is evidence rather
  than an assumption about the predicate.

All four composed arms are checked, not just `c_dl`.  `ctl` is the port
control: if it differs from the incumbent, the module is wrong before any
optimization is considered.

Exit code 0 iff every comparison passed.
"""

from __future__ import annotations

import gc
import json
import os
import sys

INSTRUMENTS = os.environ.get("CX_INSTRUMENTS", "/root/kw/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)
VARIANT = os.environ.get("CX_VARIANT",
                         os.path.dirname(os.path.abspath(__file__)))
if VARIANT not in sys.path:
    sys.path.insert(0, VARIANT)

import numpy as np  # noqa: E402

import bench_streamidx_topk as B  # noqa: E402
import streamindex_topk_composed as CX  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402

ARMS = ("ctl", "c", "dl", "c_dl")

STATIC = dict(
    k=B.K_IDX,
    compression_ratio=B.COMPRESSION_RATIO,
    num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
    num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
)

# name, phase, num_seqs, q_len, ctx, max_model_len, why
GRID = (
    # --- MIXED launch: this is where Option C's kernel actually runs. -------
    ("pre_q1024_ctx16384", "prefill", 1, 1024, 16384, 0,
     "the 21-layer prefill config, full 128-row bq blocks"),
    ("pre_q1024_ctx4096", "prefill", 1, 1024, 4096, 0,
     "shorter context, same bq geometry"),
    ("pre_q4096_ctx4096", "prefill", 1, 4096, 4096, 0,
     "32 bq blocks"),
    ("pre_q200_ctx4096", "prefill", 1, 200, 4096, 0,
     "200 tokens: a partial bq block (200 = 128 + 72), exercising the `sz` "
     "clamp under the head-major transpose"),
    ("pre_q1000_ctx8192", "prefill", 1, 1000, 8192, 0,
     "1000 tokens: 7 full bq blocks + a 104-row remainder"),
    ("pre_b2_q512_ctx4096", "prefill", 2, 512, 4096, 0,
     "two sequences in one mixed batch, so the per-sequence grid walk is "
     "exercised rather than a single-sequence special case"),
    # --- the production decode point: head-major must NOT run here. --------
    ("dec_b64_ctx8704", "decode", 64, 1, 8704, B.PROD_MAX_MODEL_LEN,
     "PRODUCTION: batch 64, ctx 8704, mml 9216 -- `_common_path`, "
     "head_minor on all three launches"),
    ("dec_b32_ctx16384", "decode", 32, 1, 16384, 0,
     "a second decode point, longer context"),
    # --- the fast path. ----------------------------------------------------
    ("dec_b64_ctx1024", "decode", 64, 1, 1024, B.PROD_MAX_MODEL_LEN,
     "FAST PATH fires (ctx <= k)"),
    ("dec_b64_ctx2048", "decode", 64, 1, 2048, B.PROD_MAX_MODEL_LEN,
     "FAST PATH fires (ctx == k)"),
)


def log(msg: str) -> None:
    print(f"[cx:correctness] {msg}", flush=True)


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

    log(f"layouts head_major_mixed=True  -> {CX.describe_layouts(True)}")
    log(f"layouts head_major_mixed=False -> {CX.describe_layouts(False)}")

    results = []
    failures = 0
    rows_compared = 0
    rows_differing = 0

    for name, phase, nseq, qlen, ctx, mml, why in GRID:
        shape = B.Shape(name, phase, nseq, qlen, ctx, max_model_len=mml)
        geo = B.geometry(shape)
        inp = B.build_inputs(shape, geo, seed=0)
        fires = (ctx // B.COMPRESSION_RATIO) <= B.K_IDX
        log(f"--- {name}: {why}")
        log(f"{name}: num_tokens={geo.num_tokens} "
            f"num_positions={geo.num_positions} "
            f"block_table_len={shape.block_table_len} "
            f"fast_path_predicate={fires} "
            f"mixed_launch_runs={not fires and phase == 'prefill'}")

        base = np.asarray(
            streamindex_topk(**inp, enable_early_exit=True, **STATIC))

        # Independent evidence about which path the incumbent took.
        cf = fast_path_closed_form(np.asarray(inp["seq_lens"]),
                                   np.asarray(inp["cu_q_lens"]),
                                   geo.num_tokens, B.K_IDX,
                                   B.COMPRESSION_RATIO)
        base_is_cf = bool(np.array_equal(base, cf))
        log(f"{name}: incumbent output == _fast_path closed form: "
            f"{base_is_cf} (expected {fires})")
        if base_is_cf != fires:
            log(f"{name}: FAIL -- the regime is not what the predicate says")
            failures += 1

        for arm in ARMS:
            fn, kw = CX.for_arm(arm)
            got = np.asarray(
                fn(**inp, enable_early_exit=True, **STATIC, **kw))
            same_shape = got.shape == base.shape
            ok = bool(same_shape and np.array_equal(got, base))
            if same_shape:
                # A "row" is one query token's k indices.
                nrow_diff = int((got != base).any(axis=1).sum())
                nelem_diff = int((got != base).sum())
            else:
                nrow_diff = nelem_diff = -1
            rows_compared += int(base.shape[0])
            rows_differing += max(0, nrow_diff)
            results.append({
                "config": name, "arm": arm, "equal": ok,
                "rows": int(base.shape[0]), "rows_differing": nrow_diff,
                "elements_differing": nelem_diff,
                "shape": list(got.shape), "fast_path": fires,
            })
            log(f"{name}: arm={arm:<5} array_equal_to_incumbent={ok} "
                f"rows={base.shape[0]} rows_differing={nrow_diff} "
                f"elements_differing={nelem_diff}")
            if not ok:
                failures += 1
            del got

        del base, cf
        for v in inp.values():
            try:
                v.delete()
            except Exception:  # noqa: BLE001
                pass
        del inp
        gc.collect()

    out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "composed")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "correctness.json"), "w") as fh:
        json.dump({"results": results, "failures": failures,
                   "rows_compared": rows_compared,
                   "rows_differing": rows_differing}, fh, indent=1)
    log(f"TOTAL rows_compared={rows_compared} rows_differing={rows_differing} "
        f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
