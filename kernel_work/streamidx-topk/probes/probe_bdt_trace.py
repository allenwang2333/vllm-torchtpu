#!/usr/bin/env python3
"""Method 2 of the corrected baseline: `hlo_stats` device self time.

`probe_baseline_device_time.py` measures device time by **chain slope** -- a
host wall clock with the dispatch term differenced out.  This file measures
the same quantity by a completely different route: `jax.profiler.trace` writes
an `*.xplane.pb`, and xprof's `hlo_stats` tool reports **per-instruction
device self time**, which never touches the host clock and therefore carries
no dispatch floor at all.  R2 (`probes/R2.md`) established the route and found
its two arms, run on separate chips of a contended pod, agreed to 0.02 %.

Two independent methods are the point.  If chain slope and `hlo_stats`
disagree by more than a few percent, that disagreement is the finding.

Arms
----

``dec_ctx8704_bkvp2``  is the **control**: it is R2's `nightly` arm, at the
same shape and the same kernel md5s, so its TensorCore total must come back at
R2's 170.474 us.  If it does not, the pipeline moved and nothing else in the
run is comparable to R2.

``pre_q1024_ctx9216``  is the number nobody has ever measured: the device time
of the **prefill** call.  Every prefill "achieved fraction of the modelled
floor" in `measurement_and_design.md` divides by a *wall clock* of 857.3 us,
and the prefill gate consequently sits anywhere in 43 %-63 %.

The remaining nine arms are candidate 0: `bkv_p` x ctx at decode, so the
`bkv_p` sweep has an `hlo_stats` reading as well as a chain slope.

The wall clock this script prints per arm is **contended and is not a result**
-- the profiler is on and other agents share the pod.  Only the `hlo_stats`
self times are.

The reader is `bdt_hlo_stats.py`, which runs off-pod against the synced
`*.xplane.pb` and applies R2's partition check: bucketed self time summed must
equal the TensorCore total to ~0.

Needs jax/jaxlib 0.11.0 -- the image's pinned 0.10.2 SIGSEGVs against libtpu
0.0.44.1 (`PLUGIN_Profiler_Api size: expected 80, got 104`).  `bdt_shard.sh`
builds the side venv.
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
INSTRUMENTS = os.environ.get("BDT_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)

AP = argparse.ArgumentParser()
AP.add_argument("--arms", default="all",
                help="comma-separated arm names, or 'all', or 'task1'")
AP.add_argument("--iters", type=int, default=100)
AP.add_argument("--iters-task2", type=int, default=50)
AP.add_argument("--chunk", type=int, default=10)
AP.add_argument("--outdir", default="")
ARGS = AP.parse_args()

OUT = ARGS.outdir or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                                  "bdt", "trace")
os.makedirs(OUT, exist_ok=True)

CTX_SWEEP = (8192, 8704, 9216)
BKVP_SWEEP = (2, 3, 4)


def log(msg):
    print(f"[bdt-trace] {msg}", flush=True)


def main() -> int:
    import jax
    import bench_streamidx_topk as B
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

    meta = {"jax": jax.__version__,
            "device_kind": jax.devices()[0].device_kind,
            "modules": {}}
    for m in (sxmod, scmod, B):
        h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
        meta["modules"][os.path.basename(m.__file__)] = {
            "path": m.__file__, "md5": h}
        log(f"MODULE {m.__file__} md5={h}")

    shapes = {s.name: s for s in B.SHAPES}
    static = dict(k=B.K_IDX, compression_ratio=B.COMPRESSION_RATIO,
                  num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
                  enable_early_exit=B.ENABLE_EARLY_EXIT)

    dec = shapes["prod_decode_b64_ctx8704"]
    pre = shapes["prod_prefill_q1024_ctx9216"]
    gdec, gpre = B.geometry(dec), B.geometry(pre)
    idec = B.build_inputs(dec, gdec, seed=0)
    ipre = B.build_inputs(pre, gpre, seed=0)
    log(f"decode geo num_positions={gdec.num_positions} "
        f"num_tokens={gdec.num_tokens}")
    log(f"prefill geo num_positions={gpre.num_positions} "
        f"num_tokens={gpre.num_tokens}")

    import jax.numpy as jnp

    def with_ctx(inputs, ctx):
        d = dict(inputs)
        d["seq_lens"] = jnp.full_like(inputs["seq_lens"], ctx)
        return d

    arms = []
    # Task 1. The decode arm is R2's `nightly` arm and is the control.
    arms.append(("dec_ctx8704_bkvp2", idec, (2, 2, 2), ARGS.iters, "task1",
                 dec.name, gdec))
    arms.append(("pre_q1024_ctx9216", ipre, B.NUM_KV_PAGES_PER_BLOCK,
                 ARGS.iters, "task1", pre.name, gpre))
    # Task 2. candidate 0.
    for ctx in CTX_SWEEP:
        for bp in BKVP_SWEEP:
            if ctx == 8704 and bp == 2:
                continue  # already the task1 control arm
            arms.append((f"dec_ctx{ctx}_bkvp{bp}", with_ctx(idec, ctx),
                         (bp, 2, 2), ARGS.iters_task2, "task2",
                         f"prod_decode_b64_ctx{ctx}", gdec))

    want = ARGS.arms
    if want == "task1":
        arms = [a for a in arms if a[4] == "task1"]
    elif want != "all":
        keep = {s for s in want.split(",") if s}
        arms = [a for a in arms if a[0] in keep]
    log(f"arms: {[a[0] for a in arms]}")

    records = []
    for name, inputs, tup, iters, group, shape_name, geo in arms:
        rec = {"arm": name, "group": group, "shape": shape_name,
               "num_kv_pages_per_block": list(tup),
               "num_positions": geo.num_positions,
               "num_tokens": geo.num_tokens, "iters": iters}
        try:
            def call(_i=inputs, _t=tup):
                return streamindex_topk(**_i, num_kv_pages_per_block=_t,
                                        **static)

            t0 = time.perf_counter()
            out = call()
            out.block_until_ready()
            rec["compile_s"] = time.perf_counter() - t0
            for _ in range(3):
                call().block_until_ready()

            logdir = os.path.join(OUT, name)
            n = 0
            t0 = time.perf_counter()
            with jax.profiler.trace(logdir):
                while n < iters:
                    batch = [call() for _ in range(min(ARGS.chunk,
                                                       iters - n))]
                    batch[-1].block_until_ready()
                    n += len(batch)
                    del batch
            wall = time.perf_counter() - t0
            rec["traced_iters"] = n
            rec["wall_us_per_call_CONTENDED_NOT_A_RESULT"] = wall / n * 1e6
            pbs = []
            for root, _d, files in os.walk(logdir):
                for f in files:
                    if f.endswith(".xplane.pb"):
                        fp = os.path.join(root, f)
                        pbs.append(fp)
                        log(f"  xplane {fp} "
                            f"({os.path.getsize(fp) / 1e6:.2f} MB)")
            rec["xplane_files"] = pbs
            rec["ok"] = bool(pbs)
            if not pbs:
                log(f"{name}: NO XPLANE WRITTEN -- the trace did not take")
            log(f"{name}: {n} iters, "
                f"{rec['wall_us_per_call_CONTENDED_NOT_A_RESULT']:.1f} us/call "
                f"(contended wall, NOT a result)")
        except Exception as e:
            rec["ok"] = False
            rec["error"] = repr(e)
            log(f"{name}: FAILED {e!r}")
            traceback.print_exc()
        records.append(rec)
        meta["arms"] = records
        with open(os.path.join(OUT, "trace_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    n_ok = sum(1 for r in records if r.get("ok"))
    log(f"done: {n_ok}/{len(records)} arms produced an xplane")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
