"""Option D -- `hlo_stats` device self time, before and after.

Same instrument as R2 (`probes/probe_r2_hlo_stats.py`): xprof's `hlo_stats`
table, per-HLO **device self time**, over a microbenchmark of
`streamindex_topk` at the nightly shape (`prod_decode_b64_ctx8704`, batch 64).
Device self time carries no host dispatch floor, which is why it is the right
instrument for a question about device work and why these arms are directly
comparable to R2's table.

R2's substitution has one cost that matters here and did not matter there: a
**single-call** program cannot show a 21->1 amortization. R2 took the 21x
multiplicity structurally and did not measure it. Option D's whole second
mechanism *is* that amortization, so this probe adds a **21-layer** program:
21 `streamindex_topk` calls in one `jax.jit`, sharing `cache_kv`, `seq_lens`,
`cu_q_lens`, `page_indices` and `distribution` (which is what the 21 indexer
layers of one forward pass share) and each with its own `q` and
`indexer_weights` (which is what they do not).

Six arms:

  ``pre1``     incumbent, 1 call.  Reproduces R2's `nightly` arm exactly, so
               the two tables can be laid side by side.
  ``pre21``    incumbent, 21 calls.  The before.
  ``dcond1``   Option D with the `lax.cond` removed and nothing else
               (`hoisted=None`), 1 call.  Isolates D1.
  ``dcond21``  same, 21 calls.  Answers "once the cond is gone, does XLA's own
               CSE collapse the metadata across layers without help?"
  ``d1``       Option D with the metadata hoisted explicitly, 1 call.
  ``d21``      same, 21 calls.  The after.

Modes are R2's, unchanged: ``--mode hlo`` lowers and compiles and writes the
optimized HLO plus a static census (no profiler, cannot SIGSEGV);
``--mode trace`` runs `jax.profiler.trace` and writes the `*.xplane.pb`.

Wall clock is printed by ``--mode trace`` and is **not a result**: the shards
are contended, and `probes/stage-b-ksweep.md` establishes that an
asynchronously-enqueued arm measures `max(device, host)`, a clamp rather than
an addend.  Every number this probe is read for comes out of `hlo_stats`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

INSTRUMENTS = os.environ.get("D_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)
VARIANT = os.environ.get("D_VARIANT", os.path.dirname(os.path.abspath(__file__)))
if VARIANT not in sys.path:
    sys.path.insert(0, VARIANT)

ARMS = ("pre1", "pre21", "dcond1", "dcond21", "d1", "d21")
NUM_LAYERS = 21


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--mode", required=True, choices=("hlo", "trace"))
    p.add_argument("--iters", type=int, default=0,
                   help="invocations inside the trace window; 0 = auto "
                        "(200 for 1-layer arms, 40 for 21-layer)")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--outdir", default="")
    return p


ARGS = _parser().parse_args()
LAYERS = NUM_LAYERS if ARGS.arm.endswith("21") else 1
KIND = ARGS.arm.rstrip("0123456789")  # pre | dcond | d
ITERS = ARGS.iters or (200 if LAYERS == 1 else 40)
OUT = ARGS.outdir or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                                  "optd", f"{ARGS.arm}_{ARGS.mode}")
os.makedirs(OUT, exist_ok=True)


def log(msg: str) -> None:
    print(f"[optd:{ARGS.arm}:{ARGS.mode}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# The program under test.
# ---------------------------------------------------------------------------


def build():
    import jax
    import jax.numpy as jnp

    import bench_streamidx_topk as B

    shape = B.Shape("prod_decode_b64_ctx8704", "decode", 64, 1, 8704,
                    max_model_len=B.PROD_MAX_MODEL_LEN,
                    note="NIGHTLY: the row report.md models; R2's `nightly`")
    geo = B.geometry(shape)
    inputs = B.build_inputs(shape, geo, seed=0)
    log(f"page_size={B.PAGE_SIZE} shape={shape.name} "
        f"num_positions={geo.num_positions} num_tokens={geo.num_tokens} "
        f"score_buffer={geo.score_buffer_bytes / 2**20:.2f} MiB "
        f"kv_cache={geo.kv_cache_bytes / 2**20:.2f} MiB "
        f"enable_early_exit={shape.enable_early_exit} layers={LAYERS}")

    static = dict(
        k=B.K_IDX,
        compression_ratio=B.COMPRESSION_RATIO,
        num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
    )

    # Per-layer q / indexer_weights. The 21 indexer layers share the KV cache
    # and all the metadata; they do not share their queries or their weights.
    # Giving every layer the same q would let XLA CSE whole `pallas_call`s and
    # the measurement would be of a 1-layer program wearing a 21-layer name.
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

    shared = dict(
        cache_kv=inputs["cache_kv"],
        seq_lens=inputs["seq_lens"],
        page_indices=inputs["page_indices"],
        cu_q_lens=inputs["cu_q_lens"],
        distribution=inputs["distribution"],
    )

    if KIND == "pre":
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
            streamindex_topk

        def program(qs, ws, cache_kv, seq_lens, page_indices, cu_q_lens,
                    distribution):
            return tuple(
                streamindex_topk(qs[i], ws[i], cache_kv, seq_lens,
                                 page_indices, cu_q_lens, distribution,
                                 enable_early_exit=True, **static)
                for i in range(LAYERS))
    else:
        import streamidx_hoist as D

        num_tokens = int(inputs["q"].shape[0])
        num_positions = D.num_positions_for(
            page_indices_len=int(inputs["page_indices"].shape[0]),
            max_num_seqs=int(inputs["seq_lens"].shape[0]),
            page_size=B.PAGE_SIZE,
            num_kv_pages_per_blocks=B.NUM_KV_PAGES_PER_BLOCK,
            k=B.K_IDX,
        )
        hoist = (KIND == "d")

        def program(qs, ws, cache_kv, seq_lens, page_indices, cu_q_lens,
                    distribution):
            meta = None
            if hoist:
                # THE HOIST: once, above the layer loop.
                meta = D.precompute(
                    seq_lens,
                    cu_q_lens,
                    distribution,
                    num_tokens=num_tokens,
                    num_positions=num_positions,
                    compression_ratio=B.COMPRESSION_RATIO,
                    decode_req_batch_size=4,
                )
            return tuple(
                D.streamindex_topk_d(qs[i], ws[i], cache_kv, seq_lens,
                                     page_indices, cu_q_lens, distribution,
                                     meta, **static) for i in range(LAYERS))

    jitted = jax.jit(program)
    args = (qs, ws, shared["cache_kv"], shared["seq_lens"],
            shared["page_indices"], shared["cu_q_lens"],
            shared["distribution"])
    return B, shape, geo, jitted, args


# ---------------------------------------------------------------------------
# Static census (R2's, verbatim, so the two are comparable).
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
    with open(os.path.join(OUT, "optimized.hlo.txt"), "w") as fh:
        fh.write(text)
    log(f"optimized HLO: {len(text) / 1e6:.2f} MB")
    with open(os.path.join(OUT, "stablehlo.txt"), "w") as fh:
        fh.write(lowered.as_text())

    cen = census(text)
    cen["conditional_count"] = text.count(" conditional(")
    cen["custom_call_count"] = text.count(" custom-call(")
    cen["branch_computations"] = sorted(
        c for c in cen["per_computation"] if "branch" in c or "region" in c)
    with open(os.path.join(OUT, "census.json"), "w") as fh:
        json.dump({"arm": ARGS.arm, "layers": LAYERS, "shape": shape.name,
                   "page_size": B.PAGE_SIZE, "census": cen}, fh, indent=1,
                  default=str)
    log(f"instructions={cen['total_instructions']} "
        f"computations={cen['computations']} "
        f"conditionals={cen['conditional_count']} "
        f"custom_calls={cen['custom_call_count']}")
    log(f"branch/region computations: {cen['branch_computations'][:8]}"
        f"{' ...' if len(cen['branch_computations']) > 8 else ''}")
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

    logdir = os.path.join(OUT, "trace")
    n = 0
    t0 = time.perf_counter()
    with jax.profiler.trace(logdir):
        while n < ITERS:
            batch = [call() for _ in range(min(ARGS.chunk, ITERS - n))]
            jax.block_until_ready(batch[-1])
            n += len(batch)
            del batch
    wall = time.perf_counter() - t0
    log(f"traced {n} invocations in {wall:.3f}s = {wall / n * 1e6:.1f} us/prog "
        f"(CONTENDED wall clock, enqueued async -- NOT A RESULT)")

    pbs = []
    for root, _dirs, files in os.walk(logdir):
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    for p in pbs:
        log(f"xplane: {p} ({os.path.getsize(p) / 1e6:.2f} MB)")
    if not pbs:
        log("NO XPLANE WRITTEN -- the trace did not take")
        return 1
    with open(os.path.join(OUT, "trace_meta.json"), "w") as fh:
        json.dump({"arm": ARGS.arm, "layers": LAYERS, "shape": shape.name,
                   "iters": n, "wall_s": wall,
                   "us_per_prog_contended": wall / n * 1e6,
                   "xplane_files": pbs}, fh, indent=1)
    return 0


def main() -> int:
    B, shape, geo, jitted, args = build()
    if ARGS.mode == "hlo":
        return run_hlo(B, shape, geo, jitted, args)
    return run_trace(B, shape, geo, jitted, args)


if __name__ == "__main__":
    sys.exit(main())
