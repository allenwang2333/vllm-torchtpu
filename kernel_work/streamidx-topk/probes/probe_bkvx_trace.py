#!/usr/bin/env python3
"""Method 2 for the extended `bkv_p` sweep: `hlo_stats` device self time.

`probe_bkvp_extended.py` measures by **chain slope** -- a host wall clock with
the dispatch term differenced out.  This file measures the same arms by a
completely different route: `jax.profiler.trace` writes an `*.xplane.pb`, and
xprof's `hlo_stats` tool reports per-instruction device self time, which never
touches the host clock and so carries no dispatch floor at all.  The two are
reported side by side; a disagreement above a few percent is the finding, not
something to average away.

Arms are the **full op** at `prod_decode_b64_ctx8704`, one per `bkv_p`.  The
full op is traced rather than isolated stage A because `hlo_stats` splits it
for free -- `StreamIdxTC-*` rows are stage A, `sc_topk*` rows are stage B --
so one trace confirms both the stage-A sweep and the full-op sweep, and R2's
partition check (`sum(buckets) - TensorCore total ~ 0`) still applies.

`dec_p2` is the **control**: `bkv_p = 2` at ctx 8704 is R2's `nightly` arm and
must come back at its 170.474 us TensorCore total.  If it does not, the
pipeline moved and nothing else in the job is comparable to the baseline.

`bkv_p in {1,2,3,5,9,10}` all give `num_sublanes_total = 80` for the tuple
`(p,2,2)`, so those arms are like-for-like; `p = 16` is included with
`nst = 128` and is flagged, because it is the only way to see whether a `p`
past the `pages_per_seq = 9` boundary still runs at all in the full op.

The wall clock printed per arm is **contended and is not a result** -- the
profiler is on and other agents share the pod.  Only the `hlo_stats` self
times are.  The reader is `bdt_hlo_stats.py`, run off-pod.

Needs jax/jaxlib 0.11.0; `bkvx_shard.sh` builds the side venv.
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
AP.add_argument("--arms", default="all")
AP.add_argument("--iters", type=int, default=50)
AP.add_argument("--chunk", type=int, default=10)
AP.add_argument("--bkvp", default="1,2,3,5,9,10,16")
AP.add_argument("--ctx", type=int, default=8704)
AP.add_argument("--outdir", default="")
ARGS = AP.parse_args()

OUT = ARGS.outdir or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                                  "bkvx", "trace")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[bkvx-trace] {msg}", flush=True)


def main() -> int:
    import jax
    import jax.numpy as jnp
    import bench_streamidx_topk as B
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

    meta = {"jax": jax.__version__,
            "device_kind": jax.devices()[0].device_kind,
            "ctx": ARGS.ctx, "modules": {}}
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
    gdec = B.geometry(dec)
    idec = B.build_inputs(dec, gdec, seed=0)
    log(f"decode geo num_positions={gdec.num_positions} "
        f"num_tokens={gdec.num_tokens} pages_per_seq={gdec.pages_per_seq}")

    names = ("q", "indexer_weights", "cache_kv", "seq_lens", "page_indices",
             "cu_q_lens", "distribution")
    args = [idec[n] for n in names]
    args[3] = jnp.full_like(idec["seq_lens"], ARGS.ctx)
    args = tuple(args)

    def align_to(x, a):
        return -(-x // a) * a

    ps = [int(x) for x in ARGS.bkvp.split(",") if x]
    arms = []
    for bp in ps:
        tup = (bp, 2, 2)
        nst = max(align_to(gdec.pages_per_seq, x) * B.PAGE_SIZE // 128
                  for x in tup)
        arms.append((f"dec_p{bp}", tup, nst))

    if ARGS.arms != "all":
        keep = {s for s in ARGS.arms.split(",") if s}
        arms = [a for a in arms if a[0] in keep]
    log(f"arms: {[a[0] for a in arms]}")

    records = []
    for name, tup, nst in arms:
        rec = {"arm": name, "shape": dec.name, "ctx": ARGS.ctx,
               "num_kv_pages_per_block": list(tup),
               "num_sublanes_total": nst,
               "like_for_like_with_bkvp2": nst == gdec.num_sublanes_total,
               "iters": ARGS.iters}
        try:
            def call(_t=tup):
                return streamindex_topk(*args, num_kv_pages_per_block=_t,
                                        **static)

            t0 = time.perf_counter()
            call().block_until_ready()
            rec["compile_s"] = time.perf_counter() - t0
            for _ in range(3):
                call().block_until_ready()

            logdir = os.path.join(OUT, name)
            n = 0
            t0 = time.perf_counter()
            with jax.profiler.trace(logdir):
                while n < ARGS.iters:
                    batch = [call()
                             for _ in range(min(ARGS.chunk, ARGS.iters - n))]
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
            rec["error"] = repr(e)[:900]
            log(f"{name}: FAILED {e!r}")
            traceback.print_exc()
        records.append(rec)
        meta["arms"] = records
        meta_name = "trace_meta_p" + ARGS.bkvp.replace(",", "_") + ".json"
        with open(os.path.join(OUT, meta_name), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    n_ok = sum(1 for r in records if r.get("ok"))
    log(f"done: {n_ok}/{len(records)} arms produced an xplane")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
