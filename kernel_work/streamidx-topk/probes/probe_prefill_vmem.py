"""Step-2 probe P7b: does prefill's large intermediate spill?

At `bq_sz=128` stage A's einsum intermediate is modelled at 32-33.55 MB in f32
(area 2 from the iteration space, area 4 from operand precision -- two routes,
same figure; `briefs/CONTRADICTIONS.md` D5). Whether it spills decides whether
prefill is VPU-bound (a compute problem: fold `relu(x)*w -> relu(x*w)`, fewer
elementwise ops per element) or VMEM-bound (a blocking problem: reduce over `h`
incrementally, shrink `bq_sz`, accumulate in bf16). Those are opposite fixes and
a wall-clock number cannot separate them, so this probe is instrumented for
spills and for capacity, never for time.

READ FIRST -- one premise of the brief is already false from source.
`streamindex_topk.py:28` sets `DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024`
and the public entry point takes `vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES`
(`streamindex_topk.py:535`). Neither the production call site
(`mla_attention_op.py:107-122`) nor the microbenchmark
(`instruments/bench_streamidx_topk.py:770-777`) overrides it. So the scoped
ceiling in force is **100 MiB, not the 32 MiB Mosaic default** -- it is already
above the 64 MiB physical VMEM. The brief's "this may be spilling because 32-33
MB sits against a 32 MiB scope" cannot be the mechanism as stated. What remains
genuinely open, and what this probe measures, is the working set itself and
whether the compiler spilled regardless.

Two modes, run as two shards.

  --mode bisect  Compile the prefill shape at a ladder of `vmem_limit_bytes`
                 and record which values compile and which are rejected. The
                 smallest limit that compiles is a direct, static upper bound
                 on the kernel's VMEM working set -- it needs no dump, no
                 profiler and no trace, and it cannot be confused with a
                 timing. Runs the decode shape through the same ladder as a
                 reference, because a single kernel's absolute number does not
                 support a conclusion.

  --mode dump    Set the `/xprof-llo` dump-side LIBTPU_INIT_ARGS
                 (`--xla_jf_dump_to`, `--xla_jf_dump_llo_text`,
                 `--xla_mosaic_dump_to`) BEFORE importing jax, then run the
                 decode shape and the prefill shape once each in the same
                 process so both programs land in one dump directory and can be
                 diffed against each other. `get_llo_fit_summary` off that dump
                 reports VMEM high-water and spill/fill counts.

EVERYTHING THIS PROBE PRODUCES IS STATIC COMPILE-TIME ANALYSIS. Not a timing,
not a measurement of throughput, and it must never reach `/log-experiment`.
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_INSTR = os.path.join(_HERE, os.pardir, "instruments",
                      "bench_streamidx_topk.py")

MIB = 1024 * 1024

# Straddles the 32 MiB Mosaic scoped default, the 64 MiB physical ceiling, and
# the kernel's own 100 MiB default.
VMEM_LADDER_MIB = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 100, 128)

DECODE_SHAPE = "prod_decode_b64_ctx8704"
PREFILL_SHAPE = "prod_prefill_q1024_ctx9216"


def _p(tag, *rest):
    print(f"[P7b] {tag}:", *rest, flush=True)


def _set_dump_env(outdir):
    """Must run before `import jax`. libtpu reads LIBTPU_INIT_ARGS once, at
    load time, and these are LIBTPU_INIT_ARGS flags -- not XLA_FLAGS. Setting
    them in XLA_FLAGS is accepted and does nothing."""
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


def _load_instrument():
    spec = importlib.util.spec_from_file_location("bench_streamidx_topk",
                                                  _INSTR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_streamidx_topk"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_once(B, streamindex_topk, shape, vmem_limit_bytes, seed=0):
    """Build inputs and execute the op once. Returns (ok, error_text)."""
    import jax

    geo = B.geometry(shape)
    inputs = B.build_inputs(shape, geo, seed=seed)
    try:
        out = streamindex_topk(
            **inputs,
            k=B.K_IDX,
            compression_ratio=B.COMPRESSION_RATIO,
            num_kv_pages_per_block=B.NUM_KV_PAGES_PER_BLOCK,
            num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
            vmem_limit_bytes=vmem_limit_bytes,
            enable_early_exit=shape.enable_early_exit,
        )
        jax.block_until_ready(out)
        return True, ""
    except Exception as e:  # noqa: BLE001 - a rejected limit is the datum
        return False, f"{type(e).__name__}: {e}"
    finally:
        for v in inputs.values():
            try:
                v.delete()
            except Exception:  # noqa: BLE001
                pass


def mode_bisect(args):
    B = _load_instrument()
    import jax
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        DEFAULT_VMEM_LIMIT_BYTES, streamindex_topk)

    _p("device", jax.devices()[0].device_kind, "jax", jax.__version__)
    _p("kernel default vmem_limit_bytes",
       f"{DEFAULT_VMEM_LIMIT_BYTES} B = {DEFAULT_VMEM_LIMIT_BYTES/MIB:.0f} MiB "
       "(streamindex_topk.py:28) -- this is what production and the "
       "microbenchmark both run with; neither overrides it")

    shapes = {s.name: s for s in B.SHAPES}
    out = {}
    for name in (PREFILL_SHAPE, DECODE_SHAPE):
        shape = shapes[name]
        geo = B.geometry(shape)
        _p("shape", name, f"num_seqs={shape.num_seqs} q_len={shape.q_len} "
           f"ctx={shape.context_len} bq_sz={geo.bq_sz} bkv_sz={geo.bkv_sz} "
           f"num_bq={geo.num_bq} num_positions={geo.num_positions}")
        # The modelled intermediate, shown so the ladder result can be read
        # against it: [bq_sz, H, bkv_sz] f32 is what the einsum produces
        # before the reduction over h.
        interm = geo.bq_sz * B.H_IDX * geo.bkv_sz * 4
        _p("  modelled einsum intermediate",
           f"[{geo.bq_sz}, {B.H_IDX}, {geo.bkv_sz}] f32 = "
           f"{interm/MIB:.2f} MiB  (modelled: bq_sz * H * bkv_sz * 4 B)")

        rows = []
        for mib in VMEM_LADDER_MIB:
            ok, err = _run_once(B, streamindex_topk, shape, mib * MIB,
                                seed=args.seed)
            rows.append({"mib": mib, "ok": ok, "error": err[:400]})
            _p(f"  vmem_limit={mib:>4} MiB", "OK" if ok else "REJECTED",
               "" if ok else err[:200])
        out[name] = rows

        passing = [r["mib"] for r in rows if r["ok"]]
        if passing:
            lo = min(passing)
            _p(f"  {name} SMALLEST_LIMIT_THAT_COMPILES", f"{lo} MiB",
               f"(static upper bound on the working set; "
               f"{100*lo/32:.0f}% of the 32 MiB Mosaic scoped default, "
               f"{100*lo/64:.0f}% of the 64 MiB physical VMEM)")
            failing_below = [r["mib"] for r in rows
                             if not r["ok"] and r["mib"] < lo]
            if failing_below:
                _p(f"  {name} rejected below that", failing_below)
            else:
                _p(f"  {name} NOTE",
                   "nothing on the ladder was rejected -- the working set is "
                   f"at or below the smallest limit tried ({min(VMEM_LADDER_MIB)} "
                   "MiB), so this run gives only a loose bound. Extend the "
                   "ladder downward to tighten it.")
        else:
            _p(f"  {name} SMALLEST_LIMIT_THAT_COMPILES",
               "NONE ON THE LADDER - even "
               f"{max(VMEM_LADDER_MIB)} MiB was rejected")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        _p("wrote", args.json)
    return 0


def mode_dump(args):
    # Env must be set before jax is imported anywhere.
    jf, mosaic = _set_dump_env(args.outdir)
    _p("LIBTPU_INIT_ARGS", os.environ["LIBTPU_INIT_ARGS"])

    B = _load_instrument()
    import jax
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

    _p("device", jax.devices()[0].device_kind, "jax", jax.__version__)

    shapes = {s.name: s for s in B.SHAPES}
    # Decode first, prefill second, both in one process so both programs land
    # in one jf dump and `list_llo_programs` shows the pair. The skill is
    # explicit that a single kernel's absolute fields do not support a
    # conclusion; the decode arm is the reference.
    for name in (DECODE_SHAPE, PREFILL_SHAPE):
        shape = shapes[name]
        geo = B.geometry(shape)
        _p("running", name, f"bq_sz={geo.bq_sz} bkv_sz={geo.bkv_sz} "
           f"num_bq={geo.num_bq} num_positions={geo.num_positions} "
           f"vmem_limit={args.vmem_limit_mib} MiB")
        ok, err = _run_once(B, streamindex_topk, shape,
                            args.vmem_limit_mib * MIB, seed=args.seed)
        _p("  ", "OK" if ok else f"FAILED {err[:400]}")

    for d, label in ((jf, "jf"), (mosaic, "mosaic")):
        n = sum(len(f) for _, _, f in os.walk(d))
        sz = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(d) for f in fs)
        _p(f"dump {label}", f"{n} files, {sz/1e6:.1f} MB, at {d}")
        if n == 0:
            _p(f"dump {label}", "EMPTY - the capture did not take. These are "
               "LIBTPU_INIT_ARGS flags and must be set before `import jax`; "
               "an empty dump reads identically to a kernel with nothing to "
               "report, so this is a capture failure, not a finding.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bisect", "dump"), required=True)
    ap.add_argument("--outdir",
                    default=os.environ.get("CDK_OUTPUT_DIR", "/tmp") +
                    "/p7b-llo")
    ap.add_argument("--vmem-limit-mib", type=int, default=100,
                    help="the kernel default; 100 MiB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    try:
        return mode_bisect(args) if args.mode == "bisect" else mode_dump(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
