"""Option C measurements 2 and 3: LLO dump of the REAL stage-A program.

R3's arms were a standalone proxy.  This compiles the actual
`StreamIdxTC[-hm]-m-bq_128-bkvp_2` prefill program -- the incumbent kernel body
or the variant's, at production geometry, with the real DMA/pipeline/double
buffer machinery -- under the `/xprof-llo` dump-side `LIBTPU_INIT_ARGS`.

EVERYTHING THIS PRODUCES IS STATIC COMPILE-TIME ANALYSIS: instruction counts,
spill counts and scheduled-bundle counts the compiler emitted.  Not a timing,
not a throughput, and it must never reach `/log-experiment`.

One arm per process: libtpu reads `LIBTPU_INIT_ARGS` once at load time, and two
dump flags pointed at one directory interleave into an unreadable mess.
"""

import argparse
import os
import sys
import traceback

ARMS = ("incumbent", "head_minor", "head_major", "head_major_hbm", "f_bf16",
        "f_bf16_hm", "f_f32")


def _p(*rest):
    print("[C-dump]", *rest, flush=True)


def _set_dump_env(outdir):
    """MUST run before `import jax`.  These are LIBTPU_INIT_ARGS, not
    XLA_FLAGS; setting them in XLA_FLAGS is accepted and does nothing."""
    jf = os.path.join(outdir, "jf")
    mosaic = os.path.join(outdir, "mosaic")
    os.makedirs(jf, exist_ok=True)
    os.makedirs(mosaic, exist_ok=True)
    flags = (f"--xla_jf_dump_to={jf} "
             "--xla_jf_dump_llo_text=true "
             f"--xla_mosaic_dump_to={mosaic}")
    prev = os.environ.get("LIBTPU_INIT_ARGS", "")
    os.environ["LIBTPU_INIT_ARGS"] = (prev + " " + flags).strip()
    return jf, mosaic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--outdir", required=True)
    # Added for the Option F / decode follow-up.  `prefill` is the default and
    # reproduces the earlier dumps byte-for-byte; `decode` compiles the DECODE
    # launcher instead, which is what RESULTS.md 7.3's +59.62% needs explaining.
    ap.add_argument("--phase", default="prefill", choices=("prefill", "decode"))
    args = ap.parse_args()

    jf, mosaic = _set_dump_env(args.outdir)
    _p("LIBTPU_INIT_ARGS", os.environ["LIBTPU_INIT_ARGS"])

    import jax
    import jax.numpy as jnp
    _p("device", jax.devices()[0].device_kind, "jax", jax.__version__)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stage_a_c as S

    rc = 0
    try:
        if args.phase == "decode":
            import bench_numbkv as N
            g = N.decode_geometry()
            launcher = N.decode_stage_a
        else:
            g = S.prefill_geometry()
            launcher = S.prefill_stage_a
        _p("phase", args.phase, "geometry",
           {k: g[k] for k in ("num_seqs", "q_len", "context_len", "bq_sz",
                              "bkv_sz", "num_bq", "num_bkv", "num_tokens",
                              "num_sublanes_total")})
        raw = S.build_prefill_inputs(g)
        q, w = S.prepared_operands(args.arm, g, raw)
        f = launcher(args.arm, g)
        fn = jax.jit(lambda *a: f(*a))
        scores_init = jnp.full(
            (g["num_tokens"], g["num_sublanes_total"], 128), -jnp.inf,
            jnp.float32)
        out = fn(q, w, raw["cache_kv"], scores_init, raw["seq_lens"],
                 raw["page_indices"], raw["cu_q_lens"],
                 jnp.array(0, jnp.int32), jnp.array(g["num_seqs"], jnp.int32))
        jax.block_until_ready(out)
        _p("arm", args.arm, "ok; out", out.shape, out.dtype, "finite_frac",
           float(jnp.mean(jnp.isfinite(out))))
    except Exception:
        traceback.print_exc()
        rc = 1

    for d, label in ((jf, "jf"), (mosaic, "mosaic")):
        n = sum(len(fs) for _, _, fs in os.walk(d))
        sz = sum(
            os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d)
            for f in fs)
        _p(f"dump {label}", f"{n} files, {sz/1e6:.1f} MB, at {d}")
        if n == 0:
            _p(f"dump {label}", "EMPTY -- the capture did not take.  Gate this "
               "before reading anything into a reader's silence.")
            rc = rc or 2

    # Direct evidence of the dtype the MXU is handed, from the Mosaic MLIR
    # rather than from reading Python.  RESULTS.md 10.1 rests on this.
    import re
    seen = set()
    for r, _, fs in os.walk(mosaic):
        for fn in sorted(fs):
            try:
                txt = open(os.path.join(r, fn), errors="replace").read()
            except OSError:
                continue
            for line in txt.splitlines():
                if re.search(r"tpu\.matmul|vector\.contract|tpu\.dot", line):
                    s = line.strip()[:400]
                    if s not in seen:
                        seen.add(s)
                        _p("MATMUL", fn, s)
    if not seen:
        _p("MATMUL", "no tpu.matmul / vector.contract line found in the "
           "mosaic dump -- do not infer the operand dtype from this job")
    return rc


if __name__ == "__main__":
    sys.exit(main())
