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
"""Capture one `*.xplane.pb` per (plan, config, arm, rep) for the gap probes.

Same method as `bench_merged_trace.py`, which it is a near-copy of, and
deliberately so: `hlo_stats` **device self time**, one trace per arm, read
off-pod by a reader that gates on the partition residual and row completeness.
The two differences are the two things `bench_merged_trace.py` cannot express.

  * **Repeats.** `--plan e1` re-enters the measurement loop `reps` times per
    arm -- a fresh `jax.profiler.trace` block and a fresh `--iters` iterations
    each time, not simply more iterations in one block.  Reps are interleaved
    across arms (rep 0 of every arm, then rep 1 of every arm, ...) so a drift
    over the life of the job spreads across arms instead of loading onto the
    one that happened to run last.  Compilation happens once per arm and is
    outside every trace block.
  * **Arms.** `gaps_common.GAPS_ARM_SPECS` adds `c`, `cctl` and `inc_nee` on
    top of the five in `merged_common.ARMS`.

The wall clock this script prints is **contended and is not a result**; it is
printed only so a hung arm is visible.  No dispatch floor is subtracted from
anything, here or downstream: per-call wall clock is `max(device, host)` and
subtracting a floor from it is invalid.

`--check` (on by default) also compares every arm's returned top-k indices
against `inc` on the same inputs, before the timing loop.  It is one extra
already-compiled call per arm and it is what keeps a fast arm that skips work
from being reported as a fast arm.

Needs jax/jaxlib 0.11.0 for the profiler; the image's pinned 0.10.2 SIGSEGVs
against libtpu 0.0.44.1 the first time `jax.profiler.trace` is called.
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

import gaps_common as GC  # noqa: E402
import merged_common as MC  # noqa: E402

EXPECT_MD5 = {
    "streamindex_topk.py": "1e221f10cbf3573aa0d2ae78c3463532",
    "sparsecore_topk.py": "8bc9e25b8701a53a97704564b9fbc658",
    "bench_streamidx_topk.py": "e8e492d13e59c4a1675cc7a98c5a0813",
    "streamindex_topk_merged.py": "5bc0578427412d56209ef8b5d764fcae",
    "streamindex_topk_cond.py": "951981dbae9e1e8d58709f574eb4abea",
}


def log(msg):
    print(f"[gaps-trace] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=sorted(GC.PLANS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=0,
                    help="override the plan's rep count (0 = use the plan's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", dest="check", action="store_true", default=True)
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    out_root = args.outdir or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "gaps", "trace")
    os.makedirs(out_root, exist_ok=True)

    import jax
    import numpy as np

    import correctness_merged as CM  # index comparison, reused verbatim

    B = MC.load_instrument()
    mods = GC.load_all_modules()
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    import vllm_torchtpu

    meta = {
        "plan": args.plan,
        "experiment": GC.PLAN_EXPERIMENT[args.plan],
        "jax": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "num_devices": len(jax.devices()),
        "iters": args.iters,
        "vllm_torchtpu_file": vllm_torchtpu.__file__,
        "modules": {},
        "md5_ok": True,
        "runs": [],
    }
    log(f"vllm_torchtpu.__file__ = {vllm_torchtpu.__file__}")
    for m in (mods["incumbent"], mods["merged"], mods["cond"], scmod, B):
        base = os.path.basename(m.__file__)
        h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
        want = EXPECT_MD5.get(base)
        ok = want is None or want == h
        meta["modules"][base] = {"path": m.__file__, "md5": h,
                                 "expected": want, "ok": ok}
        meta["md5_ok"] = meta["md5_ok"] and ok
        log(f"MODULE {m.__file__} md5={h} expected={want} "
            f"{'OK' if ok else 'MISMATCH'}")
    if not meta["md5_ok"]:
        log("FATAL: an on-pod module md5 does not match. The --map-dir did "
            "not take, or the checkout moved. Refusing to measure stale code.")

    meta_path = os.path.join(out_root, f"trace_meta.{args.plan}.json")

    def flush_meta():
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    flush_meta()
    if not meta["md5_ok"]:
        return 2

    plan = GC.PLANS[args.plan]
    log(f"plan {args.plan}: " +
        ", ".join(f"{c.name}x{len(a)}x{args.reps or r}" for c, a, r in plan))

    n_ok = n_total = 0
    for cfg, arms, plan_reps in plan:
        reps = args.reps or plan_reps
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
        decode_bkv_p = bkv_p if bkv_p is not None else MC.PREFILL_BKV_P

        # -- correctness, once per config, before any timing ----------------
        checks: dict[str, dict] = {}
        if args.check:
            inputs_np = None
            ref = None
            for arm in arms:
                try:
                    call, _spec = GC.make_call(B, mods, arm, inputs,
                                               decode_bkv_p)
                    out = np.asarray(jax.device_get(call()))
                    if arm == "inc":
                        ref = out
                        checks[arm] = {"is_reference": True}
                        continue
                    if ref is None:
                        call_i, _ = GC.make_call(B, mods, "inc", inputs,
                                                 decode_bkv_p)
                        ref = np.asarray(jax.device_get(call_i()))
                    if inputs_np is None:
                        inputs_np = {
                            "q": np.asarray(jax.device_get(
                                inputs["q"].astype(jax.numpy.float32))),
                            "indexer_weights": np.asarray(jax.device_get(
                                inputs["indexer_weights"].astype(
                                    jax.numpy.float32))),
                            "cache_kv": np.asarray(
                                jax.device_get(inputs["cache_kv"])),
                            "block_table": np.asarray(jax.device_get(
                                inputs["page_indices"])).reshape(
                                    cfg.num_seqs, -1),
                        }
                    res = CM.compare(np, cfg, inputs_np, "inc", ref, arm, out)
                    checks[arm] = res
                    log(f"  CHECK {cfg.name}/{arm}: index_identical="
                        f"{res['index_identical']} "
                        f"array_identical={res['array_identical']} "
                        f"rows_mismatched={res['rows_mismatched']}"
                        f"/{res['rows_total']}")
                    del out
                except Exception as err:  # noqa: BLE001
                    checks[arm] = {"error": repr(err)[:400]}
                    log(f"  CHECK {cfg.name}/{arm}: FAILED {err!r}")
            del ref, inputs_np

        # -- timing ---------------------------------------------------------
        calls = {}
        for arm in arms:
            calls[arm] = GC.make_call(B, mods, arm, inputs, decode_bkv_p)

        base_rec = {
            "plan": args.plan,
            "experiment": GC.PLAN_EXPERIMENT[args.plan],
            "config": cfg.name,
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
            "reps": reps,
        }

        compiled = set()
        for rep in range(reps):
            for arm in arms:
                call, spec = calls[arm]
                tg = GC.tag(args.plan, cfg, arm, rep, reps)
                rec = dict(base_rec)
                tup = spec["num_kv_pages_per_block"]
                rec.update(
                    tag=tg, arm=arm, rep=rep,
                    spec={k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in spec.items()},
                    num_sublanes_total_for_tuple=MC.num_sublanes_total(
                        geo.pages_per_seq, tup),
                    scan=GC.scanned_positions(cfg, tup),
                    check=checks.get(arm),
                )
                n_total += 1
                try:
                    if arm not in compiled:
                        t0 = time.perf_counter()
                        out = call()
                        jax.block_until_ready(out)
                        rec["compile_s"] = time.perf_counter() - t0
                        rec["out_shape"] = list(out.shape)
                        del out
                        compiled.add(arm)
                    for _ in range(args.warmup):
                        jax.block_until_ready(call())

                    logdir = os.path.join(out_root, tg)
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
                    n_ok += bool(pbs)
                    log(f"  {tg}: bkv_tuple={tup} "
                        f"scanned={rec['scan']['scanned_positions']} "
                        f"wall={rec['wall_us_per_call_CONTENDED_NOT_A_RESULT']:.1f}"
                        f"us (CONTENDED) xplanes={len(pbs)}")
                    if not pbs:
                        log(f"  {tg}: NO XPLANE -- trace did not take")
                except Exception as err:  # noqa: BLE001
                    rec["ok"] = False
                    rec["error"] = repr(err)[:600]
                    log(f"  {tg}: FAILED {err!r}")
                    traceback.print_exc()
                meta["runs"].append(rec)
                flush_meta()

        del calls
        for value in inputs.values():
            try:
                value.delete()
            except AttributeError:
                pass
        del inputs

    log(f"done: {n_ok}/{n_total} arms produced an xplane")
    return 0 if n_ok == n_total and n_total else 1


if __name__ == "__main__":
    sys.exit(main())
