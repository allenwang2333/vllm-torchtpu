#!/usr/bin/env python3
"""P1 -- capture driver for the LLO study of stage A's decode kernel.

The question (CONTRADICTIONS.md §2, ranked P1 in §13): **what fraction of
`StreamIdxTC-d`'s cycles are spent waiting on DMA semaphores** -- the
`wait_fetch_bkv` semaphore wait at `streamindex_topk.py:398`?

  >= 50%  -- stage A is DMA-bound and near its achievable floor.  Confirms
             area 3's A3/D2 and closes `report.md` §4's "21.5% of floor"
             headroom claim with "no".
  <= 15%  -- stage A is compute-bound, §4's headroom is real, and the same
             capture must then give the VPU issue rate, which is the largest
             `assumed` number in the roofline.
  between -- report the number; it discriminates neither way.

This file is the **capture** half only; the reading is done off the artifacts
with `/xprof-llo`'s `xprof-dump` and `xprof-cli`.  It sets `LIBTPU_INIT_ARGS`
*before* `import jax`, which is the only time libtpu reads it -- everything
below the `import jax` line is therefore unreachable from the flags.

Modes
-----
  --mode trace   trace-side flags: puts a `Tensor Core` bundle-marker line in
                 the `.xplane.pb`.  Cheap.
  --mode dump    dump-side flags: writes LLO IR text and lowered Mosaic MLIR to
                 disk (hundreds of MB).  Read by `get_llo_fit_summary`,
                 `get_llo_static_utilization`, `get_llo_bundles`,
                 `get_custom_call_mlir`.
  --mode server  trace-side flags plus `jax.profiler.start_server`, then spins
                 the kernel until `--seconds` elapse, so a separate process can
                 take a periodic-GFC-counter capture over the profiler gRPC
                 service.  That capture -- and only that capture -- yields
                 *measured* `count_sync_wait` rather than static analysis.

Workload: the nightly decode stage-A `pallas_call` **in isolation**
(`StreamIdxTC-d-bq_1-bkvp_2`, `seq_batch_size=4`, grid 16, 64 seqs x 1 token,
ctx 8704, num_positions 10240, out 2.62 MB f32), launched through
`_stage_a.py`, which transcribes the incumbent's own launch without editing it.

A reference arm (`--arm ref`) runs the **same kernel at `bkv_p=1`** -- same
bytes, same compute, half the DMA granule and twice as many transfers.  This is
the reference the reading needs: /xprof-llo is explicit that a single kernel's
absolute numbers do not support a conclusion, and this particular pair is also
area 3's own crude DMA discriminator, so the LLO fields and the wall clock are
answering the same question two ways.
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("trace", "dump", "server"))
    ap.add_argument("--arm", default="stage_a", choices=("stage_a", "ref"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--port", type=int, default=9012)
    return ap.parse_args()


ARGS = _parse()
OUT = ARGS.outdir or os.path.join(
    os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "p1", ARGS.mode + "_" + ARGS.arm)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# LIBTPU_INIT_ARGS -- MUST be set before `import jax`.  These are libtpu flags,
# NOT XLA_FLAGS; putting them in XLA_FLAGS is accepted and does nothing.  LLO
# dumps go to --xla_jf_dump_to; XLA's --xla_dump_to does not receive them.
# ---------------------------------------------------------------------------
TRACE_FLAGS = ("--xla_enable_custom_call_region_trace=true "
               "--xla_xprof_register_llo_debug_info=true")

if ARGS.mode == "dump":
    JF = os.path.join(OUT, "jf")
    MOSAIC = os.path.join(OUT, "mosaic")
    os.makedirs(JF, exist_ok=True)
    os.makedirs(MOSAIC, exist_ok=True)
    os.environ["LIBTPU_INIT_ARGS"] = (
        f"--xla_jf_dump_to={JF} --xla_jf_dump_llo_text=true "
        f"--xla_mosaic_dump_to={MOSAIC}")
else:
    os.environ["LIBTPU_INIT_ARGS"] = TRACE_FLAGS

print(f"[p1] mode={ARGS.mode} arm={ARGS.arm} OUT={OUT}", flush=True)
print(f"[p1] LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import _stage_a  # noqa: E402


def build_stage_a(bkv_p: int = 2):
    g, inp = _stage_a.build_nightly_inputs()
    # `num_sublanes_total` is held at the production value (80) for both arms,
    # so the output buffer -- and therefore the write-back traffic -- is
    # identical and only the DMA granule moves.  bkv_p=1 writes 9 blocks x 8
    # sublanes = 72 <= 80, so it fits.
    fn = _stage_a.make_stage_a_fn(
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=g["bq_per_block"],
        static_q_len=1,
        seq_batch_size=g["decode_req_batch_size"],
        case=_stage_a.MlaCase.DECODE,
        compression_ratio=g["compression_ratio"],
    )
    n_seqs = jnp.int32(g["num_seqs"])
    zero = jnp.int32(0)

    @jax.jit
    def run(q, w, cache_kv, seq_lens, page_indices, cu_q_lens):
        init = jnp.full((g["num_tokens"], g["num_sublanes_total"], 128),
                        -jnp.inf, jnp.float32)
        return fn(q, w, cache_kv, init, seq_lens, page_indices, cu_q_lens,
                  zero, n_seqs)

    args = (inp["q"], inp["weights"], inp["cache_kv"], inp["seq_lens"],
            inp["page_indices"], inp["cu_q_lens"])
    return g, run, args


def main():
    g, run, args = build_stage_a(2 if ARGS.arm == "stage_a" else 1)
    print(f"[p1] stage_a source {_stage_a.source_path()} "
          f"md5={_stage_a.source_md5()}", flush=True)
    print(f"[p1] geometry {g}", flush=True)

    out = jax.block_until_ready(run(*args))
    print(f"[p1] compiled+ran once; out {out.shape} {out.dtype}", flush=True)

    if ARGS.mode == "dump":
        # One run is enough: the dump is written by the compiler, not the run.
        tot = sum(
            os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT)
            for f in fs)
        n = sum(len(fs) for _, _, fs in os.walk(OUT))
        print(f"[p1] dump: {n} files, {tot/1e6:.1f} MB under {OUT}",
              flush=True)
        return

    if ARGS.mode == "trace":
        logdir = os.path.join(OUT, "trace")
        os.makedirs(logdir, exist_ok=True)
        with jax.profiler.trace(logdir):
            for _ in range(ARGS.iters):
                out = run(*args)
            jax.block_until_ready(out)
        n = sum(len(fs) for _, _, fs in os.walk(logdir))
        print(f"[p1] trace: {ARGS.iters} iters, {n} files under {logdir}",
              flush=True)
        return

    # --mode server: hold the device busy while a client captures over gRPC.
    import time
    for _ in range(5):  # warm, so the client never captures a compile
        out = run(*args)
    jax.block_until_ready(out)
    jax.profiler.start_server(ARGS.port)
    ready = os.path.join(OUT, "READY")
    with open(ready, "w") as f:
        f.write(str(ARGS.port))
    print(f"[p1] profiler server on :{ARGS.port}; READY at {ready}; "
          f"spinning for {ARGS.seconds}s", flush=True)
    t_end = time.time() + ARGS.seconds
    n = 0
    while time.time() < t_end:
        for _ in range(20):
            out = run(*args)
        jax.block_until_ready(out)
        n += 20
    print(f"[p1] server: {n} iterations executed", flush=True)


if __name__ == "__main__":
    main()
