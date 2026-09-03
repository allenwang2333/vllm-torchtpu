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
"""Composed-candidate measurement probe: `hlo_stats` device self time.

Same instrument as `d1-lite/probe_d1lite.py` -- xprof's `hlo_stats` table, per
HLO **device self time**, TensorCore plane only.  Device self time carries no
host dispatch floor, which is why it is the right instrument for a question
about device work, and why nothing here reports a wall clock as a result.

## What is different from `probe_d1lite.py`, and why

1. **Many specs per process.**  `probe_d1lite.py` measured one
   `(config, arm)` per process.  This exercise has ~90 of them and process
   startup (JAX import + TPU init) dominates a 6-second compile, so this probe
   loops over every spec it is given.  Each spec still gets its own trace
   directory, so the off-pod reader is unchanged.

2. **Compile and trace in one process.**  `probe_d1lite.py` ran `--mode hlo`
   under the image's jax 0.10.2 and `--mode trace` under a jax 0.11.0 side
   venv, because jaxlib 0.10.2's profiler client SIGSEGVs against libtpu
   0.0.44.1.  Running everything under 0.11.0 gets the static census for free
   from the same `compiled` object the trace then runs.

3. **Prefill configs.**  The 21-layer chain and the single-call grid both need
   prefill shapes, which `probe_d1lite.py` did not have.

4. **Replicates.**  A spec may carry a rep index; the tag becomes
   `<cfg>_r<n>`, so each replicate is an independent trace with an independent
   compile and an independent `hlo_stats` read.  The loop is re-entered rather
   than run longer, which is what makes the spread an estimate of run-to-run
   variation rather than of within-loop jitter.

## Arms

`inc` is the incumbent from `src/`, unmodified.  The other four come from
`streamindex_topk_composed.py`: `ctl` (both switches off, the port control),
`c` (Option C gated to MIXED), `dl` (D1-lite hoist) and `c_dl` (both).

## Spec syntax

    <config>:<arm>[:<rep>]

`<config>` is a key of `CONFIGS` below.  Every configuration is the production
kernel configuration: `k=2048`, `compression_ratio=1`, `page_size=1024`,
`num_kv_pages_per_block=(2,2,2)`, `num_queries_per_block=(1,128,128)`,
`enable_early_exit=True`.  The `bkv_p` ladder is withdrawn and is not an axis
here.

Wall clock printed by this probe is **not a result**: the four shards are
contended and calls are enqueued asynchronously, so it measures
`max(device, host)`.  Every number this probe is read for comes out of
`hlo_stats`.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import re
import sys
import time
import traceback

INSTRUMENTS = os.environ.get("CX_INSTRUMENTS", "/root/kw/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)
VARIANT = os.environ.get("CX_VARIANT",
                         os.path.dirname(os.path.abspath(__file__)))
if VARIANT not in sys.path:
    sys.path.insert(0, VARIANT)

ARMS = ("inc", "ctl", "c", "dl", "c_dl")

# ---------------------------------------------------------------------------
# The configurations.
#
# `(phase, layers, batch, q_len, ctx, max_model_len)`.  `max_model_len=0` means
# "the deployment is sized to this request", i.e. the block table is
# `cdiv(ctx, page_size)` wide -- `bench_streamidx_topk.Shape`'s default.
# ---------------------------------------------------------------------------

PROD_MML = 9216  # the GLM-5.2 nightly perf config: 8192 in + 1024 out

CONFIGS: dict[str, tuple] = {}

# --- item 1: the headline.  21-layer chain at the production decode point.
CONFIGS["L21_dec_b64_ctx8704"] = ("decode", 21, 64, 1, 8704, PROD_MML)

# --- item 2: 21-layer chain at one prefill config.
CONFIGS["L21_pre_q1024_ctx16384"] = ("prefill", 21, 1, 1024, 16384, 0)

# --- item 5: the fast-path regime, at 21 layers.
#     ctx 1024 and 2048 with k=2048, cr=1 -> max(seq_lens)//cr <= k is TRUE.
for _ctx in (1024, 2048):
    CONFIGS[f"L21_dec_b64_ctx{_ctx}"] = ("decode", 21, 64, 1, _ctx, PROD_MML)

# --- item 3: the single-call coverage grid, 17 configurations.
for _b in (8, 32, 64):
    for _ctx in (4096, 16384, 32768, 65536):
        CONFIGS[f"L1_dec_b{_b}_ctx{_ctx}"] = ("decode", 1, _b, 1, _ctx, 0)
CONFIGS["L1_dec_b64_ctx8704_mml9216"] = ("decode", 1, 64, 1, 8704, PROD_MML)
for _q in (1024, 4096):
    for _ctx in (4096, 16384):
        CONFIGS[f"L1_pre_q{_q}_ctx{_ctx}"] = ("prefill", 1, 1, _q, _ctx, 0)

# The single-call grid above already contains the single-call twin of each of
# the two 21-layer chains -- `L1_dec_b64_ctx8704_mml9216` is the headline's
# shape at one layer, and `L1_pre_q1024_ctx16384` is the prefill chain's.  So
# the depth effect (D1-lite's benefit is cross-layer CSE) is a difference
# measured inside this one job rather than across two.  No extra entries: an
# earlier draft added twins that were byte-identical duplicates of these two,
# which would have measured the same thing twice under two names.


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="*", help="<config>:<arm>[:<rep>]")
    p.add_argument("--root", default="",
                   help="output root; default $CDK_OUTPUT_DIR/composed")
    p.add_argument("--iters", type=int, default=0,
                   help="invocations inside the trace window; 0 = auto "
                        "(200 for 1-layer, 40 for 21-layer)")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--list", action="store_true",
                   help="print the config table and exit")
    p.add_argument("--census", action="store_true", default=True,
                   help="also write the optimized HLO and a static census")
    p.add_argument("--no-census", dest="census", action="store_false")
    return p


ARGS = _parser().parse_args()
ROOT = ARGS.root or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                                 "composed")
LABEL = os.environ.get("TPU_FANOUT_SHARD", "0")


def log(msg: str) -> None:
    print(f"[cx:{LABEL}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Static census (Option D's, verbatim, so the tables stay comparable).
# ---------------------------------------------------------------------------


def census(hlo_text: str) -> dict:
    per_comp: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    comp = "<module>"
    n_instr = 0
    instr_re = re.compile(r"^\s+%?([\w.\-]+)\s*=\s*\S+\s+([\w\-]+)\(")
    comp_re = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s*(?:\(|\{)")
    for line in hlo_text.splitlines():
        m = comp_re.match(line)
        if m and line.rstrip().endswith("{"):
            comp = m.group(1)
            continue
        m = instr_re.match(line)
        if m:
            n_instr += 1
            per_comp[comp][m.group(2)] += 1
    return {
        "total_instructions": n_instr,
        "computations": len(per_comp),
        "per_computation": {c: dict(k) for c, k in per_comp.items()},
    }


# ---------------------------------------------------------------------------
# One spec.
# ---------------------------------------------------------------------------


def build(cfg_name: str, arm: str):
    import jax
    import jax.numpy as jnp

    import bench_streamidx_topk as B
    import streamindex_topk_composed as CX

    phase, layers, batch, q_len, ctx, mml = CONFIGS[cfg_name]
    shape = B.Shape(cfg_name, phase, batch, q_len, ctx, max_model_len=mml)
    geo = B.geometry(shape)
    inputs = B.build_inputs(shape, geo, seed=0)
    fast = (ctx // B.COMPRESSION_RATIO) <= B.K_IDX

    fn, extra_kw = CX.for_arm(arm)
    kw = dict(
        k=B.K_IDX,
        compression_ratio=B.COMPRESSION_RATIO,
        num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
        enable_early_exit=True,
        **extra_kw,
    )
    if arm == "inc":
        layouts = {"decode_grouped": "head_minor",
                   "decode_mopup": "head_minor",
                   "mixed": "head_minor"}
    else:
        layouts = CX.describe_layouts(extra_kw["head_major_mixed"])

    log(f"{cfg_name}__{arm}: phase={phase} layers={layers} batch={batch} "
        f"q_len={q_len} ctx={ctx} mml={shape.block_table_len} "
        f"num_tokens={geo.num_tokens} pages_per_seq={geo.pages_per_seq} "
        f"num_positions={geo.num_positions} "
        f"score_buf={geo.score_buffer_bytes / 2**20:.1f}MiB "
        f"kv_cache={geo.kv_cache_bytes / 2**20:.1f}MiB "
        f"fast_path_fires={fast} layouts={layouts} kw={extra_kw}")

    keys = jax.random.split(jax.random.key(1234), 2 * layers)
    q_dt = inputs["q"].dtype
    w_dt = inputs["indexer_weights"].dtype
    # Per-layer `q` and `indexer_weights`, shared everything else -- what the
    # 21 indexer layers of one forward pass do and do not share.  Giving every
    # layer the same `q` would let XLA CSE whole `pallas_call`s and the
    # measurement would be of a 1-layer program wearing a 21-layer name.
    qs = tuple(
        jax.random.normal(keys[2 * i], inputs["q"].shape,
                          jnp.float32).astype(q_dt) for i in range(layers))
    ws = tuple(
        jax.random.uniform(keys[2 * i + 1], inputs["indexer_weights"].shape,
                           jnp.float32, 0.25, 1.75).astype(w_dt)
        for i in range(layers))

    def program(qs, ws, cache_kv, seq_lens, page_indices, cu_q_lens,
                distribution):
        return tuple(
            fn(qs[i], ws[i], cache_kv, seq_lens, page_indices, cu_q_lens,
               distribution, **kw) for i in range(layers))

    jitted = jax.jit(program)
    args = (qs, ws, inputs["cache_kv"], inputs["seq_lens"],
            inputs["page_indices"], inputs["cu_q_lens"],
            inputs["distribution"])
    meta = {"config": cfg_name, "arm": arm, "phase": phase, "layers": layers,
            "batch": batch, "q_len": q_len, "ctx": ctx,
            "block_table_len": shape.block_table_len,
            "num_tokens": geo.num_tokens,
            "num_positions": geo.num_positions,
            "fast_path_fires": fast, "layouts": layouts}
    return jitted, args, layers, meta, (inputs, qs, ws)


def run_spec(cfg_name: str, arm: str, rep: int) -> tuple[int, dict]:
    import jax

    tag = cfg_name if rep is None else f"{cfg_name}_r{rep}"
    key = f"{tag}__{arm}"
    # The trace dir IS the key: the reader derives <config>__<arm> from the
    # path component directly above `plugins/`.
    trace_dir = os.path.join(ROOT, "traces", key)
    hlo_dir = os.path.join(ROOT, "hlo", key)
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(hlo_dir, exist_ok=True)

    jitted, args, layers, meta, keepalive = build(cfg_name, arm)
    iters = ARGS.iters or (200 if layers == 1 else 40)

    if ARGS.census:
        t0 = time.perf_counter()
        compiled = jitted.lower(*args).compile()
        text = compiled.as_text()
        with open(os.path.join(hlo_dir, "optimized.hlo.txt"), "w") as fh:
            fh.write(text)
        cen = census(text)
        cen["conditional_count"] = text.count(" conditional(")
        cen["custom_call_count"] = text.count(" custom-call(")
        # The gate, read off the compiled program rather than asserted: how
        # many `pallas_call`s carry the head-major scope name.
        cen["stage_a_hm_calls"] = len(re.findall(r"StreamIdxTC-hm-", text))
        cen["stage_a_calls"] = len(re.findall(r"StreamIdxTC-(?!hm-)", text))
        cen["stage_a_hm_mixed_calls"] = len(
            re.findall(r"StreamIdxTC-hm-M-", text))
        cen["stage_a_hm_decode_calls"] = len(
            re.findall(r"StreamIdxTC-hm-D-", text))
        with open(os.path.join(hlo_dir, "census.json"), "w") as fh:
            json.dump({"key": key, **meta, "census": cen}, fh, indent=1,
                      default=str)
        log(f"{key}: compile {time.perf_counter() - t0:.2f}s "
            f"hlo={len(text) / 1e6:.2f}MB instr={cen['total_instructions']} "
            f"comps={cen['computations']} conds={cen['conditional_count']} "
            f"stageA_hm={cen['stage_a_hm_calls']} "
            f"(M={cen['stage_a_hm_mixed_calls']} "
            f"D={cen['stage_a_hm_decode_calls']}) "
            f"stageA_hmnr={cen['stage_a_calls']}")
        del text, compiled
    else:
        cen = {}

    def call():
        return jitted(*args)

    t0 = time.perf_counter()
    out = call()
    jax.block_until_ready(out)
    log(f"{key}: warmup+compile {time.perf_counter() - t0:.3f}s "
        f"outs={len(out)} shape={out[0].shape}")
    for _ in range(3):
        jax.block_until_ready(call())

    n = 0
    t0 = time.perf_counter()
    with jax.profiler.trace(trace_dir):
        while n < iters:
            batch = [call() for _ in range(min(ARGS.chunk, iters - n))]
            jax.block_until_ready(batch[-1])
            n += len(batch)
            del batch
    wall = time.perf_counter() - t0
    log(f"{key}: traced {n} invocations in {wall:.3f}s = "
        f"{wall / n * 1e6:.1f} us/prog (CONTENDED wall clock, enqueued async "
        f"-- NOT A RESULT)")

    pbs = []
    for root, _dirs, files in os.walk(trace_dir):
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    with open(os.path.join(hlo_dir, "trace_meta.json"), "w") as fh:
        json.dump({"key": key, **meta, "rep": rep, "iters": n,
                   "wall_s": wall, "us_per_prog_contended": wall / n * 1e6,
                   "xplane_files": pbs, "census": cen}, fh, indent=1,
                  default=str)
    if not pbs:
        log(f"{key}: NO XPLANE WRITTEN -- the trace did not take")
        return 1, meta
    log(f"{key}: xplane {pbs[0]} "
        f"({os.path.getsize(pbs[0]) / 1e6:.2f} MB)")

    # Free this spec's device buffers before the next one builds a KV cache of
    # its own; the biggest config here allocates ~1 GiB and the shard runs
    # dozens of specs in one process.  Explicit `.delete()` on the arrays this
    # function owns, then a gc pass for whatever the tracer kept alive --
    # `jax.live_arrays()` is deliberately not used, since it would also reap
    # buffers the runtime still owns.
    inputs, qs, ws = keepalive
    for arr in list(inputs.values()) + list(qs) + list(ws) + list(out):
        try:
            arr.delete()
        except Exception:  # noqa: BLE001
            pass
    del out, jitted, args, keepalive, inputs, qs, ws
    gc.collect()
    return 0, meta


def main() -> int:
    if ARGS.list:
        for name, spec in CONFIGS.items():
            print(f"{name:<32} {spec}")
        return 0
    if not ARGS.specs:
        print("no specs given", file=sys.stderr)
        return 2

    import jax
    log(f"jax {jax.__version__} devices={jax.devices()}")

    rcs: dict[str, int] = {}
    for spec in ARGS.specs:
        parts = spec.split(":")
        cfg, arm = parts[0], parts[1]
        rep = int(parts[2]) if len(parts) > 2 else None
        if cfg not in CONFIGS:
            log(f"SPEC {spec}: unknown config {cfg!r}")
            rcs[spec] = 90
            continue
        if arm not in ARMS:
            log(f"SPEC {spec}: unknown arm {arm!r}")
            rcs[spec] = 91
            continue
        t0 = time.perf_counter()
        try:
            rc, _ = run_spec(cfg, arm, rep)
        except Exception:  # noqa: BLE001 - one dead spec must not kill the rest
            log(f"SPEC {spec}: EXCEPTION")
            traceback.print_exc()
            rc = 92
        rcs[spec] = rc
        log(f"SPEC {spec} rc={rc} in {time.perf_counter() - t0:.1f}s")

    print(f"===== CX/{LABEL} SPEC TABLE =====", flush=True)
    fail = 0
    for spec, rc in rcs.items():
        print(f"{spec:<48} rc={rc}")
        fail += (rc != 0)
    print(f"===== CX/{LABEL} failed_specs={fail}/{len(rcs)} =====", flush=True)
    return fail


if __name__ == "__main__":
    sys.exit(main())
