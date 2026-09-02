"""R2 -- the nightly composition of the decode residual.

The step-3 prerequisite R2 asks: *what is the **nightly** (not AgentX)
composition of the decode residual, and how much of it is metadata that
hoisting above the `lax.cond` would remove?*

`measurement_and_design.md` §5 names the instrument and says no new probe
design is needed: *"`hlo_stats` at the nightly shape -- the same pull as
`j-a71a648c`, different config."*  `j-a71a648c` is an **in-model** AgentX
profile that happened to be lying around; there is no nightly-config profile
anywhere and producing one would need an end-to-end model benchmark, which R2
is forbidden from launching.  So this probe produces the same *table* --
xprof's `hlo_stats`, per-HLO device self time -- from a **single-call
microbenchmark** of `streamindex_topk` at the nightly shape.

Why that substitution is the right one, and what it costs:

  * `hlo_stats` self time is **device** time per HLO instruction.  It is not a
    wall-clock arm, so it does not carry the ~42 us host dispatch floor that
    makes every arm in `bench_streamidx_topk.py` (`total`, `stage_b`, `init`)
    an upper bound rather than a device cost.  That floor is exactly what
    corrupted the "44.8 us residual" the design doc built Option D on: 44.8 us
    is the `init` arm, and the same JSON row reports
    `dispatch_floor_ms = 0.042`.
  * It resolves the residual *by instruction*, which a subtraction of three
    independently-optimistic best-of-N arms cannot do.
  * What it loses: the 21-layer module.  In the model the 21 indexer layers are
    21 private `cond` computations and CSE cannot span them; here there is one
    call.  The 21x multiplicity is already established structurally
    (`j-a71a648c`, every metadata row an exact multiple of 21) and is *not*
    re-measured here.  What is measured is the **per-call size of each bucket**
    at the nightly shape, which is the term the design doc is missing.

Arms (one per fanned-out chip):

  ``nightly``    prod_decode_b64_ctx8704 -- the row `report.md` §6 models.
  ``agentx``     agentx_decode_b16_ctx8192, `--page-size 256`, mml 262144 --
                 the deployment `j-a71a648c` captured.  Run in the *same job*,
                 by the *same instrument*, so the AgentX->nightly comparison is
                 within-instrument rather than across two profiles taken by two
                 different methods.
  ``nightly16``  nightly config at batch 16 -- the control that separates
                 "batch 16 vs batch 64" from "AgentX config vs nightly config"
                 in the pair above.  Without it the two arms differ on two axes
                 at once, which is the mistake `report.md` §6.2 retracted.

Modes:

  ``--mode hlo``    lower+compile and write the optimized HLO text plus a
                    static instruction census.  Needs no profiler, so it runs
                    on the image's pinned jax and cannot SIGSEGV.
  ``--mode trace``  `jax.profiler.trace` around N invocations, writing an
                    `*.xplane.pb` for `/xprof-dump`'s `hlo_stats`.  This needs
                    jax/jaxlib 0.11.0 (the image's 0.10.2 SIGSEGVs against
                    libtpu 0.0.44.1 -- see `run-command-in-cdk` SKILL.md).

Everything is written under ``$CDK_OUTPUT_DIR`` so `cdk job sync-outputs`
retrieves it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

INSTRUMENTS = os.environ.get("R2_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    p.add_argument("--arm",
                   required=True,
                   choices=("nightly", "agentx", "nightly16"))
    p.add_argument("--mode", required=True, choices=("hlo", "trace"))
    p.add_argument("--iters", type=int, default=200,
                   help="invocations inside the trace window")
    p.add_argument("--chunk", type=int, default=10,
                   help="invocations enqueued between block_until_ready calls")
    p.add_argument("--outdir", default="")
    return p


ARGS = _parser().parse_args()
OUT = ARGS.outdir or os.path.join(
    os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "r2", f"{ARGS.arm}_{ARGS.mode}")
os.makedirs(OUT, exist_ok=True)


def log(msg: str) -> None:
    print(f"[r2:{ARGS.arm}:{ARGS.mode}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# The three arms, expressed against the instrument's own Shape/geometry so the
# geometry cannot drift from `bench_streamidx_topk.py`'s.
# ---------------------------------------------------------------------------

def build_arm():
    import bench_streamidx_topk as B

    if ARGS.arm == "agentx":
        # PAGE_SIZE is a module global that feeds pages_per_seq,
        # num_positions, the cache shape and every modelled byte count.
        # Rebind it before anything reads it -- the instrument's own main()
        # does exactly this for `--page-size`.
        B.PAGE_SIZE = B.AGENTX_PAGE_SIZE
        shape = B.Shape("agentx_decode_b16_ctx8192", "decode", 16, 1, 8192,
                        max_model_len=B.AGENTX_MAX_MODEL_LEN,
                        note="AGENTX: the deployment j-a71a648c captured")
    elif ARGS.arm == "nightly16":
        shape = B.Shape("nightly_decode_b16_ctx8704", "decode", 16, 1, 8704,
                        max_model_len=B.PROD_MAX_MODEL_LEN,
                        note="NIGHTLY config at the AgentX batch -- the "
                             "control that separates batch from config")
    else:
        shape = B.Shape("prod_decode_b64_ctx8704", "decode", 64, 1, 8704,
                        max_model_len=B.PROD_MAX_MODEL_LEN,
                        note="NIGHTLY: the row report.md models")

    geo = B.geometry(shape)
    log(f"page_size={B.PAGE_SIZE} shape={shape.name} "
        f"num_positions={geo.num_positions} num_tokens={geo.num_tokens} "
        f"score_buffer={geo.score_buffer_bytes / 2**20:.2f} MiB "
        f"kv_cache={geo.kv_cache_bytes / 2**20:.2f} MiB "
        f"enable_early_exit={shape.enable_early_exit}")
    inputs = B.build_inputs(shape, geo, seed=0)
    static = dict(
        k=B.K_IDX,
        compression_ratio=B.COMPRESSION_RATIO,
        num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
        enable_early_exit=shape.enable_early_exit,
    )
    return B, shape, geo, inputs, static


# ---------------------------------------------------------------------------
# Static census. Classifies every instruction of the optimized module into the
# buckets Option D cares about, from the framework scope XLA carries on it.
# ---------------------------------------------------------------------------

# Ordered: first match wins.
BUCKETS = (
    ("stage_a_kernel", re.compile(r"StreamIdxTC-")),
    ("stage_b_kernel", re.compile(r"sc_topk_direct|sparse_?core_topk|"
                                  r"sc_topk|SparseCore")),
    ("init_neg_inf", re.compile(r"scores_init|full_like|/full\b|-inf")),
    ("eff_row_lengths", re.compile(r"_effective_row_lengths")),
    ("cond_control", re.compile(r"\bconditional\b|branch_index|get_tuple")),
)

# Opcodes that are pure scalar / index plumbing rather than bulk data movement.
SCALAR_OPCODES = {
    "slice", "concatenate", "gather", "dynamic-slice", "dynamic-update-slice",
    "reshape", "bitcast", "bitcast-convert", "convert", "iota", "compare",
    "select", "and", "or", "not", "add", "subtract", "multiply", "divide",
    "maximum", "minimum", "reduce", "broadcast", "constant", "pad",
    "transpose", "copy", "tuple", "get-tuple-element", "parameter",
}


def classify(name: str, metadata: str) -> str:
    blob = f"{name} {metadata}"
    for bucket, pat in BUCKETS:
        if pat.search(blob):
            return bucket
    return "other"


def census(hlo_text: str) -> dict:
    """Count instructions per opcode and per computation from HLO text."""
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


def run_hlo(B, shape, geo, inputs, static) -> int:
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

    lowered = streamindex_topk.lower(**inputs, **static)
    compiled = lowered.compile()
    text = compiled.as_text()
    path = os.path.join(OUT, "optimized.hlo.txt")
    with open(path, "w") as fh:
        fh.write(text)
    log(f"optimized HLO: {path} ({len(text) / 1e6:.2f} MB)")

    pre = lowered.as_text()
    with open(os.path.join(OUT, "stablehlo.txt"), "w") as fh:
        fh.write(pre)

    cen = census(text)
    # Instruction-level cross-check on the `cond` structure Option D attacks.
    cen["conditional_count"] = text.count(" conditional(")
    cen["custom_call_count"] = text.count(" custom-call(")
    cen["branch_computations"] = sorted(
        c for c in cen["per_computation"] if "branch" in c or "region" in c)
    payload = {
        "arm": ARGS.arm,
        "shape": shape.name,
        "page_size": B.PAGE_SIZE,
        "geo": {k: v for k, v in vars(geo).items()}
        if hasattr(geo, "__dict__") else {},
        "census": cen,
    }
    with open(os.path.join(OUT, "census.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    log(f"instructions={cen['total_instructions']} "
        f"computations={cen['computations']} "
        f"conditionals={cen['conditional_count']} "
        f"custom_calls={cen['custom_call_count']}")
    log(f"branch/region computations: {cen['branch_computations']}")
    return 0


def run_trace(B, shape, geo, inputs, static) -> int:
    import jax

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

    log(f"jax {jax.__version__} jaxlib {jax.lib.__version__ if hasattr(jax.lib, '__version__') else '?'}")

    def call():
        return streamindex_topk(**inputs, **static)

    # Warm up outside the trace so compilation is not in the window.
    t0 = time.perf_counter()
    out = call()
    out.block_until_ready()
    log(f"warmup+compile {time.perf_counter() - t0:.3f}s out={out.shape}")
    for _ in range(3):
        call().block_until_ready()

    logdir = os.path.join(OUT, "trace")
    n = 0
    t0 = time.perf_counter()
    with jax.profiler.trace(logdir):
        while n < ARGS.iters:
            batch = [call() for _ in range(min(ARGS.chunk, ARGS.iters - n))]
            batch[-1].block_until_ready()
            n += len(batch)
            del batch
    wall = time.perf_counter() - t0
    log(f"traced {n} invocations in {wall:.3f}s "
        f"= {wall / n * 1e6:.1f} us/call (CONTENDED wall, not a result)")

    pbs = []
    for root, _dirs, files in os.walk(logdir):
        for f in files:
            if f.endswith(".xplane.pb"):
                p = os.path.join(root, f)
                pbs.append((p, os.path.getsize(p)))
    for p, sz in pbs:
        log(f"xplane: {p} ({sz / 1e6:.2f} MB)")
    if not pbs:
        log("NO XPLANE WRITTEN -- the trace did not take")
        return 1
    with open(os.path.join(OUT, "trace_meta.json"), "w") as fh:
        json.dump({
            "arm": ARGS.arm,
            "shape": shape.name,
            "page_size": B.PAGE_SIZE,
            "iters": n,
            "wall_s": wall,
            "us_per_call_contended": wall / n * 1e6,
            "xplane_files": [p for p, _ in pbs],
        }, fh, indent=1)
    return 0


def main() -> int:
    B, shape, geo, inputs, static = build_arm()
    if ARGS.mode == "hlo":
        return run_hlo(B, shape, geo, inputs, static)
    return run_trace(B, shape, geo, inputs, static)


if __name__ == "__main__":
    sys.exit(main())
