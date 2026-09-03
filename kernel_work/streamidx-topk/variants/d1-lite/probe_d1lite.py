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
"""D1-lite -- `hlo_stats` device self time, per arm, per config.

Same instrument as Option D's `probe_d_hlo_stats.py` and R2's
`probes/probe_r2_hlo_stats.py`: xprof's `hlo_stats` table, per-HLO **device
self time**.  Device self time carries no host dispatch floor, which is why it
is the right instrument for a question about device work.

Two program shapes, selected with `--layers`:

  ``--layers 1``   one `streamindex_topk` call in one `jax.jit`.
  ``--layers 21``  21 calls in one `jax.jit`, sharing `cache_kv`, `seq_lens`,
                   `cu_q_lens`, `page_indices` and `distribution` (what the 21
                   indexer layers of one forward pass share) and each with its
                   own `q` and `indexer_weights` (what they do not).  Giving
                   every layer the same `q` would let XLA CSE whole
                   `pallas_call`s and the measurement would be of a 1-layer
                   program wearing a 21-layer name.

The 21-layer program is the only one that can show the cross-layer CSE this
variant exists to buy.

Five arms:

  ``inc``     incumbent `streamindex_topk`, `enable_early_exit=True`.
  ``d1``      incumbent, `enable_early_exit=False`.  The `lax.cond` and the
              fast path are gone; this is Option D's D1.
  ``dl``      D1-lite, `hoist="plain"`: cond kept, metadata computed
              unconditionally above it.
  ``dlc``     D1-lite, `hoist="cond"`: cond kept, metadata computed inside its
              own inner `lax.cond` on the same predicate.
  ``dlctl``   D1-lite, `hoist="none"`: cond kept, metadata left inside the
              branch.  The port-fidelity control -- it should reproduce `inc`.

`--ctx` sets `context_len`; `max_model_len` is pinned to 9216 in every config,
so `pages_per_seq`, `num_positions` (10240) and every buffer shape are
identical across contexts and only the *values* in `seq_lens` change.  At
`--ctx 1024` and `--ctx 2048` with `k=2048` and `compression_ratio=1` the
early-exit predicate `max(seq_lens) // compression_ratio <= k` is TRUE, so the
fast path fires; at `--ctx 8704` it is FALSE.

Modes: ``--mode hlo`` lowers and compiles and writes the optimized HLO plus a
static census (no profiler, cannot SIGSEGV); ``--mode trace`` runs
`jax.profiler.trace` and writes the `*.xplane.pb`.

Wall clock printed by ``--mode trace`` is **not a result**: the shards are
contended and an asynchronously-enqueued arm measures `max(device, host)`.
Every number this probe is read for comes out of `hlo_stats`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

INSTRUMENTS = os.environ.get("DL_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)
VARIANT = os.environ.get("DL_VARIANT",
                         os.path.dirname(os.path.abspath(__file__)))
if VARIANT not in sys.path:
    sys.path.insert(0, VARIANT)

# arm -> (module, enable_early_exit, hoist)
ARM_SPECS = {
    "inc": ("incumbent", True, None),
    "d1": ("incumbent", False, None),
    "dl": ("d1lite", True, "plain"),
    "dlc": ("d1lite", True, "cond"),
    "dlctl": ("d1lite", True, "none"),
}
ARMS = tuple(ARM_SPECS)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--mode", required=True, choices=("hlo", "trace"))
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--ctx", type=int, default=8704)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=0,
                   help="invocations inside the trace window; 0 = auto "
                        "(200 for 1-layer, 40 for 21-layer)")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--tag", default="",
                   help="config key; the trace dir is <tag>__<arm>, which is "
                        "what the off-pod reader keys the table on")
    p.add_argument("--root", default="",
                   help="output root; default $CDK_OUTPUT_DIR/d1lite")
    return p


ARGS = _parser().parse_args()
LAYERS = ARGS.layers
ITERS = ARGS.iters or (200 if LAYERS == 1 else 40)
TAG = ARGS.tag or f"L{LAYERS}_ctx{ARGS.ctx}_b{ARGS.batch}"
ROOT = ARGS.root or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                                 "d1lite")
KEY = f"{TAG}__{ARGS.arm}"
# The trace dir IS the key: the reader derives <config>__<arm> from the path
# component directly above `plugins/`.
TRACE_DIR = os.path.join(ROOT, "traces", KEY)
HLO_DIR = os.path.join(ROOT, "hlo", KEY)
os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(HLO_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"[d1lite:{KEY}:{ARGS.mode}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# The program under test.
# ---------------------------------------------------------------------------


def build():
    import jax
    import jax.numpy as jnp

    import bench_streamidx_topk as B

    shape = B.Shape(f"decode_b{ARGS.batch}_ctx{ARGS.ctx}", "decode",
                    ARGS.batch, 1, ARGS.ctx,
                    max_model_len=B.PROD_MAX_MODEL_LEN)
    geo = B.geometry(shape)
    inputs = B.build_inputs(shape, geo, seed=0)
    fast = (ARGS.ctx // B.COMPRESSION_RATIO) <= B.K_IDX
    log(f"page_size={B.PAGE_SIZE} shape={shape.name} mml={B.PROD_MAX_MODEL_LEN} "
        f"num_positions={geo.num_positions} num_tokens={geo.num_tokens} "
        f"pages_per_seq={geo.pages_per_seq} "
        f"score_buffer={geo.score_buffer_bytes / 2**20:.2f} MiB "
        f"kv_cache={geo.kv_cache_bytes / 2**20:.2f} MiB layers={LAYERS} "
        f"k={B.K_IDX} cr={B.COMPRESSION_RATIO} fast_path_fires={fast}")

    static = dict(
        k=B.K_IDX,
        compression_ratio=B.COMPRESSION_RATIO,
        num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
    )

    keys = jax.random.split(jax.random.key(1234), 2 * LAYERS)
    q_dt = inputs["q"].dtype
    w_dt = inputs["indexer_weights"].dtype
    qs = tuple(
        jax.random.normal(keys[2 * i], inputs["q"].shape,
                          jnp.float32).astype(q_dt) for i in range(LAYERS))
    ws = tuple(
        jax.random.uniform(keys[2 * i + 1], inputs["indexer_weights"].shape,
                           jnp.float32, 0.25, 1.75).astype(w_dt)
        for i in range(LAYERS))

    mod_name, eee, hoist = ARM_SPECS[ARGS.arm]
    if mod_name == "incumbent":
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
            streamindex_topk as fn
        kw = dict(static, enable_early_exit=eee)
    else:
        import streamindex_topk_d1lite as DL
        fn = DL.streamindex_topk_dl
        kw = dict(static, enable_early_exit=eee, hoist=hoist)
    log(f"arm={ARGS.arm} module={mod_name} enable_early_exit={eee} "
        f"hoist={hoist}")

    def program(qs, ws, cache_kv, seq_lens, page_indices, cu_q_lens,
                distribution):
        return tuple(
            fn(qs[i], ws[i], cache_kv, seq_lens, page_indices, cu_q_lens,
               distribution, **kw) for i in range(LAYERS))

    jitted = jax.jit(program)
    args = (qs, ws, inputs["cache_kv"], inputs["seq_lens"],
            inputs["page_indices"], inputs["cu_q_lens"],
            inputs["distribution"])
    return B, shape, geo, jitted, args


# ---------------------------------------------------------------------------
# Static census (Option D's, verbatim, so the tables are comparable).
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


def run_hlo(B, shape, geo, jitted, args) -> int:
    lowered = jitted.lower(*args)
    compiled = lowered.compile()
    text = compiled.as_text()
    with open(os.path.join(HLO_DIR, "optimized.hlo.txt"), "w") as fh:
        fh.write(text)
    log(f"optimized HLO: {len(text) / 1e6:.2f} MB")

    cen = census(text)
    cen["conditional_count"] = text.count(" conditional(")
    cen["custom_call_count"] = text.count(" custom-call(")
    cen["branch_computations"] = sorted(
        c for c in cen["per_computation"] if "branch" in c or "region" in c)
    with open(os.path.join(HLO_DIR, "census.json"), "w") as fh:
        json.dump({"key": KEY, "arm": ARGS.arm, "layers": LAYERS,
                   "ctx": ARGS.ctx, "shape": shape.name, "census": cen}, fh,
                  indent=1, default=str)
    log(f"instructions={cen['total_instructions']} "
        f"computations={cen['computations']} "
        f"conditionals={cen['conditional_count']} "
        f"custom_calls={cen['custom_call_count']} "
        f"branch_comps={len(cen['branch_computations'])}")
    return 0


def run_trace(B, shape, geo, jitted, args) -> int:
    import jax

    log(f"jax {jax.__version__}")

    def call():
        return jitted(*args)

    t0 = time.perf_counter()
    out = call()
    jax.block_until_ready(out)
    log(f"warmup+compile {time.perf_counter() - t0:.3f}s "
        f"outs={len(out)} shape={out[0].shape}")
    for _ in range(3):
        jax.block_until_ready(call())

    n = 0
    t0 = time.perf_counter()
    with jax.profiler.trace(TRACE_DIR):
        while n < ITERS:
            batch = [call() for _ in range(min(ARGS.chunk, ITERS - n))]
            jax.block_until_ready(batch[-1])
            n += len(batch)
            del batch
    wall = time.perf_counter() - t0
    log(f"traced {n} invocations in {wall:.3f}s = {wall / n * 1e6:.1f} us/prog "
        f"(CONTENDED wall clock, enqueued async -- NOT A RESULT)")

    pbs = []
    for root, _dirs, files in os.walk(TRACE_DIR):
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    for p in pbs:
        log(f"xplane: {p} ({os.path.getsize(p) / 1e6:.2f} MB)")
    if not pbs:
        log("NO XPLANE WRITTEN -- the trace did not take")
        return 1
    with open(os.path.join(HLO_DIR, "trace_meta.json"), "w") as fh:
        json.dump({"key": KEY, "arm": ARGS.arm, "layers": LAYERS,
                   "ctx": ARGS.ctx, "shape": shape.name, "iters": n,
                   "wall_s": wall, "us_per_prog_contended": wall / n * 1e6,
                   "xplane_files": pbs}, fh, indent=1)
    return 0


def main() -> int:
    B, shape, geo, jitted, args = build()
    if ARGS.mode == "hlo":
        return run_hlo(B, shape, geo, jitted, args)
    return run_trace(B, shape, geo, jitted, args)


if __name__ == "__main__":
    sys.exit(main())
