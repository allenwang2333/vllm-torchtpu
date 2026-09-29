"""Probe 20: live-slice SparseCore stage-1 DMA (TPU_STREAMIDX_L1_LIVE) at layer scope.

  P20_SEQ=<seq> | P20_DUMP=<npz>  python scripts/benchmark_streamidx_l1_live.py <tag>

Runs on either tree. On a tree whose `streamindex_topk` has no `l1_live`
(stack3b) only the `off` arm exists. Arms use the stack3b serving config:
chunk_tokens=128, sctopk_l1=True, score_bf16_group=4, enable_early_exit=False,
topk_tail="segcD"; `on` adds l1_live=True; `fault` is `on` with stage 1 reading
every slice 16 words early (positive control, must differ).

Layer = indexer + q-glue + sparse-MLA attention, 1024 query tokens, one
sequence of P20_SEQ tokens (C = P20_SEQ - 1024), as probes 16/18. Timing arms
are interleaved (off, on, off, on, ...). Writes the top-k of each arm to
$CDK_OUTPUT_DIR/p20/topk_<tag>_<case>.npz for the cross-tree comparison, and
results_<tag>_<case>.json.
"""
import os
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "tree"
OUT = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "p20")
os.makedirs(OUT, exist_ok=True)
import vllm_torchtpu.env_override  # noqa: E402,F401  (same libtpu flags as serving)

import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

import vllm_torchtpu  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import sctopk_l1 as l1  # noqa: E402
from vllm_torchtpu.kernels.mla.sparse import kernel as mla  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_streamidx_topk_tail import (  # noqa: E402
    IDX_KEYS,
    LAYER_KEYS,
    SM,
    TOPK,
    _qglue,
    args_of,
    problem,
    problem_from_dump,
)


def log(*a):
    print(*a, flush=True)


HAS_LIVE = "l1_live" in inspect.signature(streamindex_topk).parameters
BF16G = int(os.environ.get("P20_BF16G", "4"))
TAIL = os.environ.get("P20_TAIL", "segcD")
ITERS = int(os.environ.get("P20_ITERS", "40"))
REPS = int(os.environ.get("P20_REPS", "3"))
RES = {}


def _topk(iq, w, cache, seq_lens, pi, cu, dist, live):
    kw = dict(l1_live=True) if live else {}
    return streamindex_topk(
        iq, w, cache, seq_lens, pi, cu, dist, k=TOPK, compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2), num_queries_per_block=(1, 128, 128),
        enable_early_exit=False, chunk_tokens=128, sctopk_l1=True,
        score_bf16_group=BF16G, topk_tail=TAIL, **kw)


def idx_fn(live):
    @jax.jit
    def f(iq, w, cache, seq_lens, pi, cu, dist):
        return _topk(iq, w, cache, seq_lens, pi, cu, dist, live)
    return f


def layer_fn(live):
    @jax.jit
    def f(iq, w, cache, seq_lens, pi, cu, dist, qb, wuk, cos, sin, nope, rope):
        topk = _topk(iq, w, cache, seq_lens, pi, cu, dist, live)
        q = _qglue(qb, wuk, cos, sin)
        out = mla.sparse_ragged_paged_attention(
            q, nope, rope, topk, None, cu, dist, sm_scale=SM,
            gather_and_attention_chunk_size=64, attention_kernel_batch_size=16,
            cache_layout="sparsecore")
        return out, topk
    return f


def timeit(fn, args, iters):
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append((time.perf_counter() - t) * 1e3)
    return statistics.median(s), min(s)


def sha(a):
    return hashlib.sha256(np.asarray(jax.device_get(a)).tobytes()).hexdigest()[:16]


def run_arm(name, fn, args, timed=True):
    try:
        out = jax.block_until_ready(fn(*args))
        r = {}
        if timed:
            med, mn = timeit(fn, args, ITERS)
            r.update(ms_med=round(med, 4), ms_min=round(mn, 4))
        RES[name] = r
        log(f"ARM {name:24s} {json.dumps(r)}")
        return out
    except Exception as e:  # noqa: BLE001
        RES[name] = dict(error=repr(e)[:400])
        log(f"ARM {name:24s} ERROR {e!r}"[:2000])
        traceback.print_exc()
        return None


def main():
    dump = os.environ.get("P20_DUMP")
    seq = int(os.environ.get("P20_SEQ", "33792"))
    p = problem_from_dump(dump) if dump else problem(1024, seq)
    case = os.path.basename(dump).replace(".npz", "") if dump else f"s{seq}"
    seq = int(np.asarray(p["seq_lens"])[0])
    log("TREE", TAG, vllm_torchtpu.__file__, "jax", jax.__version__, jax.devices()[0],
        "case", case, "seq", seq, "C", seq - 1024, "HAS_LIVE", HAS_LIVE, "BF16G", BF16G,
        "TAIL", TAIL)
    LA, IA = args_of(p, LAYER_KEYS), args_of(p, IDX_KEYS)
    arms = [False, True] if HAS_LIVE else [False]
    names = {False: "off", True: "on"}
    L = {live: layer_fn(live) for live in arms}
    I = {live: idx_fn(live) for live in arms}
    topks = {}
    # Interleaved timing: layer, then indexer alone.
    for rep in range(1, REPS + 1):
        for live in arms:
            out = run_arm(f"L_{names[live]}_{rep}", L[live], LA)
            if out is not None and rep == 1:
                topks[names[live]] = np.asarray(jax.device_get(out[1]))
                RES[f"sha_attn_{names[live]}"] = sha(out[0])
    for rep in range(1, REPS + 1):
        for live in arms:
            out = run_arm(f"I_{names[live]}_{rep}", I[live], IA)
            if out is not None and rep == 1:
                topks[f"I_{names[live]}"] = np.asarray(jax.device_get(out))
    if HAS_LIVE:
        # Positive control: stage 1 reads each slice 16 words early.
        orig = l1._stage1_spec

        def faulty(*a, live=False, **kw):
            if live:
                kw["live_fault"] = "shift"
            return orig(*a, live=live, **kw)

        l1._stage1_spec = faulty
        jax.clear_caches()
        try:
            out = run_arm("I_fault", idx_fn(True), IA, timed=False)
            if out is not None:
                topks["I_fault"] = np.asarray(jax.device_get(out))
        finally:
            l1._stage1_spec = orig
            jax.clear_caches()
    for k, v in topks.items():
        RES[f"sha_topk_{k}"] = sha(v)
    ref = topks.get("off")
    if ref is not None:
        for k, v in topks.items():
            RES[f"diff_topk_{k}_vs_off"] = int((v != ref).sum())
            if k != "off":
                log(f"DIFF topk {k} vs off: {RES[f'diff_topk_{k}_vs_off']} of {ref.size}")
    np.savez_compressed(os.path.join(OUT, f"topk_{TAG}_{case}.npz"), **topks)
    json.dump(RES, open(os.path.join(OUT, f"results_{TAG}_{case}.json"), "w"), indent=1)
    log("RESULTS", TAG, case, json.dumps(RES))
    bad = [k for k, v in RES.items() if isinstance(v, dict) and "error" in v]
    log("errors:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
