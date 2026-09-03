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
"""Capture one `*.xplane.pb` per (config, arm) for the merged-op A/B.

Method: `hlo_stats` **device self time**, the primary method `results.md` 1
mandates.  It is per-instruction, carries no host dispatch component and is
immune to pod contention, which is what makes a fanned-out capture legitimate.

The wall clock this script prints per arm is **contended and is not a result.**
It is printed only so a hung arm is visible.  Everything reportable comes out
of the reader (`read_merged_hlo_stats.py`) run off-pod against the synced
xplanes.

One trace per arm, never one trace per config: `hlo_stats` keys its rows on
the HLO op name, and `inc` and `d1` differ only in a `lax.cond` that leaves
the Pallas op names identical.  Two arms in one trace would be summed.

Needs jax/jaxlib 0.11.0.  The image's pinned 0.10.2 SIGSEGVs against libtpu
0.0.44.1 (`PLUGIN_Profiler_Api size: expected 80, got 104`) the first time
`jax.profiler.trace` is called.  `merged_shard.sh` builds the side venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import merged_common as MC  # noqa: E402


def log(msg):
    print(f"[merged-trace] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--configs", default="all",
                    help="comma-separated config names, or 'all'")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    out_root = args.outdir or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "merged", "trace")
    os.makedirs(out_root, exist_ok=True)

    import jax

    B = MC.load_instrument()
    mods = {"merged": MC.load_merged(), "incumbent": MC.load_incumbent()}
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod

    meta = {
        "jax": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "num_devices": len(jax.devices()),
        "shard": args.shard,
        "num_shards": args.num_shards,
        "iters": args.iters,
        "modules": {},
        "runs": [],
    }
    for m in (mods["incumbent"], mods["merged"], scmod, B):
        h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
        meta["modules"][os.path.basename(m.__file__)] = {
            "path": m.__file__, "md5": h}
        log(f"MODULE {m.__file__} md5={h}")

    wanted_cfgs = MC.CONFIGS
    if args.configs != "all":
        keep = {s for s in args.configs.split(",") if s}
        wanted_cfgs = [c for c in MC.CONFIGS if c.name in keep]
    wanted_cfgs = [c for i, c in enumerate(wanted_cfgs)
                   if i % args.num_shards == args.shard]
    arms = (list(MC.ARMS) if args.arms == "all"
            else [a for a in args.arms.split(",") if a])
    log(f"configs: {[c.name for c in wanted_cfgs]}")
    log(f"arms: {arms}")

    meta_path = os.path.join(out_root, f"trace_meta.shard{args.shard}.json")

    def flush_meta():
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    flush_meta()

    for cfg in wanted_cfgs:
        shape = MC.to_shape(B, cfg)
        geo = B.geometry(shape)
        log(f"=== {cfg.name}: tokens={geo.num_tokens} "
            f"num_positions={geo.num_positions} "
            f"pages_per_seq={geo.pages_per_seq} "
            f"score_buf={geo.score_buffer_bytes / 2**20:.1f}MiB "
            f"kv_cache={geo.kv_cache_bytes / 2**20:.1f}MiB")
        try:
            inputs = B.build_inputs(shape, geo, seed=args.seed)
        except Exception as err:  # noqa: BLE001
            log(f"{cfg.name}: input build FAILED {err!r}")
            meta["runs"].append({"config": cfg.name, "ok": False,
                                 "error": repr(err)[:400],
                                 "stage": "build_inputs"})
            flush_meta()
            continue

        bkv_p, ladder = MC.resolve_decode_bkv_p(B, mods, cfg, inputs, log=log)
        log(f"{cfg.name}: recommended bkv_p={ladder['recommended_bkv_p']} "
            f"chosen={bkv_p} fell_back={ladder['fell_back']}")

        for arm in arms:
            rec = {
                "config": cfg.name,
                "arm": arm,
                "phase": cfg.phase,
                "num_seqs": cfg.num_seqs,
                "q_len": cfg.q_len,
                "ctx": cfg.ctx,
                "max_model_len": cfg.max_model_len,
                "block_table_len": cfg.block_table_len,
                "num_tokens": geo.num_tokens,
                "num_positions": geo.num_positions,
                "pages_per_seq": geo.pages_per_seq,
                "bkv_p_ladder": ladder,
            }
            if bkv_p is None and arm in MC.BKVP_ARMS:
                rec.update(ok=False, error="no bkv_p on the ladder lowered")
                meta["runs"].append(rec)
                flush_meta()
                continue
            decode_bkv_p = bkv_p if bkv_p is not None else 2
            call, spec = MC.make_call(B, mods, arm, inputs, decode_bkv_p)
            rec["spec"] = {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in spec.items()}
            rec["num_sublanes_total_for_tuple"] = MC.num_sublanes_total(
                geo.pages_per_seq, spec["num_kv_pages_per_block"])
            try:
                t0 = time.perf_counter()
                out = call()
                jax.block_until_ready(out)
                rec["compile_s"] = time.perf_counter() - t0
                rec["out_shape"] = list(out.shape)
                del out
                for _ in range(args.warmup):
                    jax.block_until_ready(call())

                logdir = os.path.join(out_root, f"{cfg.name}__{arm}")
                n = 0
                t0 = time.perf_counter()
                with jax.profiler.trace(logdir):
                    while n < args.iters:
                        batch = [call()
                                 for _ in range(min(args.chunk,
                                                    args.iters - n))]
                        jax.block_until_ready(batch[-1])
                        n += len(batch)
                        del batch
                wall = time.perf_counter() - t0
                rec["traced_iters"] = n
                rec["wall_us_per_call_CONTENDED_NOT_A_RESULT"] = (
                    wall / n * 1e6)
                pbs = []
                for root, _d, files in os.walk(logdir):
                    for f in files:
                        if f.endswith(".xplane.pb"):
                            pbs.append(os.path.join(root, f))
                rec["xplane_files"] = pbs
                rec["ok"] = bool(pbs)
                log(f"  {cfg.name}/{arm}: bkv_tuple="
                    f"{spec['num_kv_pages_per_block']} "
                    f"compile={rec['compile_s']:.1f}s "
                    f"wall={rec['wall_us_per_call_CONTENDED_NOT_A_RESULT']:.1f}"
                    f"us (CONTENDED) xplanes={len(pbs)}")
                if not pbs:
                    log(f"  {cfg.name}/{arm}: NO XPLANE -- trace did not take")
            except Exception as err:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = repr(err)[:600]
                log(f"  {cfg.name}/{arm}: FAILED {err!r}")
                traceback.print_exc()
            meta["runs"].append(rec)
            flush_meta()

        for value in inputs.values():
            try:
                value.delete()
            except AttributeError:
                pass
        del inputs

    n_ok = sum(1 for r in meta["runs"] if r.get("ok"))
    log(f"done: {n_ok}/{len(meta['runs'])} arms produced an xplane")
    return 0 if n_ok == len(meta["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
