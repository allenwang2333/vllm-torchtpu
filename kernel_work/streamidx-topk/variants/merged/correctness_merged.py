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
"""`merged` vs `inc`, on top-k **indices**, per config.

The gate is index-set equality against the incumbent, per row.  It is not a
recall-vs-f64 gate: `results.md` 11 established that the k=2048-of-8704
boundary is degenerate in f32 (k-th place margin median 0.017 against sigma
2.2408), so an absolute recall gate fails on the *incumbent* and measures the
wrong thing.  The incumbent is the reference here, and the question is whether
the three merged changes move its answer.

**A tie broken differently is not an error.**  When two indices are swapped
between the arms, this script recomputes both of their scores from a
first-principles f32 reference -- dequantize the fp8 K record, dot with each
q head, relu, weight, sum over heads, scale -- and prints both, so the reader
can see whether the two scores are equal rather than taking "mismatch" at face
value.  The reference is built from the same arrays the kernel was handed, and
is evaluated only at the handful of positions that disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import merged_common as MC  # noqa: E402


def log(msg):
    print(f"[merged-correct] {msg}", flush=True)


def _seq_of_token(cfg, t: int) -> int:
    return t // cfg.q_len


def _reference_scores(np, cfg, inputs_np, token: int, positions):
    """f32 reference indexer score for one token at a list of KV positions."""
    import ml_dtypes
    q = inputs_np["q"]                    # (T, H, D) float32 (already upcast)
    w = inputs_np["indexer_weights"]      # (T, H) float32
    cache = inputs_np["cache_kv"]         # (P, 256, 4, 256) uint8
    bt = inputs_np["block_table"]         # (num_seqs, pages_per_seq) int32
    page_size = 1024
    seq = _seq_of_token(cfg, token)
    out = []
    for j in positions:
        if j < 0:
            out.append(float("nan"))
            continue
        page = int(bt[seq, j // page_size])
        slot = int(j % page_size)
        rec = cache[page, slot // 4, slot % 4]           # uint8[256]
        kv = rec[:128].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        exp = int(rec[128])                               # e8m0
        scale = float(2.0**(exp - 127)) if exp != 255 else float("nan")
        per_head = np.maximum((q[token] * kv[None, :]).sum(axis=1), 0.0)
        out.append(float((per_head * w[token]).sum() * scale))
    return out


def compare(np, cfg, inputs_np, a_name, a_idx, b_name, b_idx,
            max_report: int = 8):
    """Row-wise index comparison, with score read-back on every mismatch."""
    res = {
        "a": a_name, "b": b_name,
        "shape": list(a_idx.shape),
        "array_identical": bool(np.array_equal(a_idx, b_idx)),
    }
    a_sorted = np.sort(a_idx, axis=1)
    b_sorted = np.sort(b_idx, axis=1)
    row_eq = (a_sorted == b_sorted).all(axis=1)
    bad_rows = np.nonzero(~row_eq)[0]
    res["rows_total"] = int(a_idx.shape[0])
    res["rows_set_identical"] = int(row_eq.sum())
    res["rows_mismatched"] = int(bad_rows.size)
    res["entries_differing"] = int((a_sorted != b_sorted).sum())
    res["index_identical"] = bool(bad_rows.size == 0)
    res["mismatches"] = []
    for r in bad_rows[:max_report]:
        r = int(r)
        sa, sb = set(a_idx[r].tolist()), set(b_idx[r].tolist())
        only_a = sorted(sa - sb)
        only_b = sorted(sb - sa)
        entry = {"row": r, f"only_in_{a_name}": only_a[:16],
                 f"only_in_{b_name}": only_b[:16],
                 "n_only_a": len(only_a), "n_only_b": len(only_b)}
        try:
            entry[f"scores_only_in_{a_name}"] = _reference_scores(
                np, cfg, inputs_np, r, only_a[:16])
            entry[f"scores_only_in_{b_name}"] = _reference_scores(
                np, cfg, inputs_np, r, only_b[:16])
        except Exception as err:  # noqa: BLE001
            entry["score_readback_error"] = repr(err)[:300]
        res["mismatches"].append(entry)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-on-mismatch", action="store_true", default=True,
                    help="when merged != inc, also run d1 and bkvp_d1 so the "
                         "difference is attributable to one of the three")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "merged",
        f"correctness.shard{args.shard}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    import jax
    import numpy as np

    B = MC.load_instrument()
    mods = {"merged": MC.load_merged(), "incumbent": MC.load_incumbent()}
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod

    payload = {"jax": jax.__version__,
               "device_kind": jax.devices()[0].device_kind,
               "shard": args.shard, "modules": {}, "configs": []}
    for m in (mods["incumbent"], mods["merged"], scmod, B):
        h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
        payload["modules"][os.path.basename(m.__file__)] = {
            "path": m.__file__, "md5": h}
        log(f"MODULE {m.__file__} md5={h}")

    cfgs = MC.CONFIGS
    if args.configs != "all":
        keep = {s for s in args.configs.split(",") if s}
        cfgs = [c for c in MC.CONFIGS if c.name in keep]
    cfgs = [c for i, c in enumerate(cfgs)
            if i % args.num_shards == args.shard]
    log(f"configs: {[c.name for c in cfgs]}")

    def flush():
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=1, default=str)

    flush()
    for cfg in cfgs:
        entry = {"config": cfg.name, "phase": cfg.phase,
                 "num_seqs": cfg.num_seqs, "q_len": cfg.q_len,
                 "ctx": cfg.ctx, "max_model_len": cfg.max_model_len}
        t0 = time.perf_counter()
        try:
            shape = MC.to_shape(B, cfg)
            geo = B.geometry(shape)
            inputs = B.build_inputs(shape, geo, seed=args.seed)
            entry["num_positions"] = geo.num_positions
            entry["num_tokens"] = geo.num_tokens

            bkv_p, ladder = MC.resolve_decode_bkv_p(B, mods, cfg, inputs,
                                                    log=log)
            entry["bkv_p_ladder"] = ladder
            if bkv_p is None:
                entry["error"] = "no bkv_p on the ladder lowered"
                payload["configs"].append(entry)
                flush()
                continue

            outs = {}
            arms = ["inc", "merged"]
            for arm in arms:
                call, spec = MC.make_call(B, mods, arm, inputs, bkv_p)
                outs[arm] = np.asarray(jax.device_get(call()))
                entry.setdefault("specs", {})[arm] = {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in spec.items()}

            inputs_np = {
                "q": np.asarray(jax.device_get(
                    inputs["q"].astype(jax.numpy.float32))),
                "indexer_weights": np.asarray(jax.device_get(
                    inputs["indexer_weights"].astype(jax.numpy.float32))),
                "cache_kv": np.asarray(jax.device_get(inputs["cache_kv"])),
                "block_table": np.asarray(jax.device_get(
                    inputs["page_indices"])).reshape(cfg.num_seqs, -1),
            }

            entry["inc_vs_merged"] = compare(np, cfg, inputs_np, "inc",
                                             outs["inc"], "merged",
                                             outs["merged"])
            log(f"{cfg.name}: index_identical="
                f"{entry['inc_vs_merged']['index_identical']} "
                f"rows_mismatched="
                f"{entry['inc_vs_merged']['rows_mismatched']}"
                f"/{entry['inc_vs_merged']['rows_total']} "
                f"array_identical={entry['inc_vs_merged']['array_identical']}")

            if (args.extra_on_mismatch
                    and not entry["inc_vs_merged"]["index_identical"]):
                for arm in ("d1", "bkvp_d1"):
                    call, spec = MC.make_call(B, mods, arm, inputs, bkv_p)
                    outs[arm] = np.asarray(jax.device_get(call()))
                    entry[f"inc_vs_{arm}"] = compare(
                        np, cfg, inputs_np, "inc", outs["inc"], arm,
                        outs[arm])
                    log(f"{cfg.name}: inc vs {arm} index_identical="
                        f"{entry[f'inc_vs_{arm}']['index_identical']}")

            for value in inputs.values():
                try:
                    value.delete()
                except AttributeError:
                    pass
        except Exception as err:  # noqa: BLE001
            entry["error"] = repr(err)[:600]
            log(f"{cfg.name}: FAILED {err!r}")
        entry["wall_s"] = time.perf_counter() - t0
        payload["configs"].append(entry)
        flush()

    ok = [c for c in payload["configs"]
          if c.get("inc_vs_merged", {}).get("index_identical")]
    log(f"index-identical on {len(ok)}/{len(payload['configs'])} configs")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
