"""LLO/Mosaic capture for the incumbent `sparsecore_topk` at the shipping
prefill shape (b=1024, n=10240, k=2048 -> P=1, slice_len=10240, 32 waves).

Follows /xprof-llo's capture section:
  * both recipes' flags in ONE `LIBTPU_INIT_ARGS`, so one run yields the
    .xplane.pb AND the jf/mosaic dump trees;
  * the variable is written BEFORE anything imports jax or libtpu, and read
    back AFTER `vllm_torchtpu.env_override` has had its turn at it;
  * dump dirs live under $CDK_OUTPUT_DIR (--map-dir is upload-only);
  * every variant is warmed before any capture, and one throwaway profiler
    session is burned first (the first session comes back host-only).

Nothing here is a timing result. Fanned-out shards share the host and HBM.
"""

from __future__ import annotations

import argparse
import functools
import glob
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys

# --------------------------------------------------------------------------
# Everything above `import jax` -- argv is scanned by hand because the dump
# paths are arguments and LIBTPU_INIT_ARGS cannot wait for argparse.
# --------------------------------------------------------------------------
TRACE_FLAGS = ("--xla_enable_custom_call_region_trace=true "
               "--xla_xprof_register_llo_debug_info=true")


def _peek(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


_MODE = _peek("--mode", "both")
_OUTDIR = _peek("--outdir")
if _OUTDIR is None:
    raise SystemExit("--outdir is required")
_JF = os.path.join(_OUTDIR, "jf")
_MOSAIC = os.path.join(_OUTDIR, "mosaic")

_flags = []
if _MODE in ("trace", "both"):
    _flags += TRACE_FLAGS.split()
if _MODE in ("dump", "both"):
    os.makedirs(_JF, exist_ok=True)
    os.makedirs(_MOSAIC, exist_ok=True)
    _flags += [
        f"--xla_jf_dump_to={_JF}",
        "--xla_jf_dump_llo_text=true",
        f"--xla_mosaic_dump_to={_MOSAIC}",
    ]
_SCB = os.path.join(_OUTDIR, "scbundles")
_SCM = os.path.join(_OUTDIR, "scmlir")
_SCL = os.path.join(_OUTDIR, "scllvm")
_SCHWM = os.path.join(_OUTDIR, "sc_hwm.pb")
_SCS_PROG = os.path.join(_OUTDIR, "scs_program.txt")

# SparseCore-specific dump flags. These are NOT the jf/LLO flags: the jf dump
# only ever contains TensorCore (TLP) programs, and on an offloaded SC kernel
# that is the launcher, not the kernel. These names were read out of
# libtpu 0.0.44.1's own DebugOptions field-name table -- see --scdump-bogus
# for the control that says whether an unknown name here is loud or silent.
if _peek("--scdump", "0") == "1":
    for d in (_SCB, _SCM, _SCL):
        os.makedirs(d, exist_ok=True)
    _flags += [
        f"--xla_sc_dump_bundles_to={_SCB}",
        f"--xla_sc_dump_mlir_to={_SCM}",
        "--xla_sc_dump_mlir_with_debug_info=true",
        f"--xla_sc_dump_llvm_ir_to={_SCL}",
        "--xla_sc_include_debug_info=true",
        "--xla_sc_enable_logging_memory_high_water_marks=true",
        f"--xla_sc_high_water_mark_protobuf_file={_SCHWM}",
        f"--xla_sc_scs_program_dump_file={_SCS_PROG}",
        "--xla_mosaic_enable_llo_source_annotations=true",
    ]

# The control. A string-valued dump flag accepts any value, so pointing it at
# a directory and finding the directory empty does not distinguish "the flag
# ran and there was nothing to write" from "libtpu never heard of it". This
# shard sets a name that certainly does not exist; whatever libtpu does with
# it is what it also did with any real flag it did not recognise.
_BOGUS = _peek("--scdump-bogus", "0") == "1"
if _BOGUS:
    _flags += [f"--xla_sc_dump_bundles_to_NOSUCHFLAG={_SCB}",
               "--xla_sc_NOSUCHFLAG_at_all=true"]

_prev = os.environ.get("LIBTPU_INIT_ARGS", "")
os.environ["LIBTPU_INIT_ARGS"] = (" ".join(_flags) + " " + _prev).strip()
_WANTED = list(_flags)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `vllm_torchtpu` first: its env_override patches the Pallas fori_loop
# lowering, which changes the Mosaic these kernels lower to. It also
# rewrites LIBTPU_INIT_ARGS -- appending, but that is its implementation and
# not a promise, hence the read-back gate below.
import vllm_torchtpu  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _pick_partition, _sc_topk_direct, sparsecore_topk,
)

SEED = 20260904


def _fail(msg, code):
    print(f"\n!! {msg}", flush=True)
    print("CAPTURE: FAIL")
    return code


def _du(path):
    if not os.path.isdir(path):
        return 0, 0
    tot = nfiles = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
            nfiles += 1
    return tot, nfiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["incumbent", "incumbent_keys", "probe_scatter",
                             "probe_skeleton"])
    ap.add_argument("--mode", default="both", choices=["trace", "dump", "both"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--b", type=int, default=1024)
    ap.add_argument("--n", type=int, default=10240)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--prune-gib", type=float, default=3.0)
    ap.add_argument("--scdump", default="0", choices=["0", "1"])
    ap.add_argument("--scdump-bogus", default="0", choices=["0", "1"])
    args = ap.parse_args()

    print(f"host={socket.gethostname()} arm={args.arm} mode={args.mode}",
          flush=True)
    print(f"jax={jax.__version__} devices={jax.device_count()} "
          f"kind={jax.devices()[0].device_kind}", flush=True)
    live = os.environ.get("LIBTPU_INIT_ARGS", "")
    print(f"LIBTPU_INIT_ARGS={live}", flush=True)

    # ---- gates, before anything expensive ---------------------------------
    missing = [f for f in _WANTED if f not in live]
    if missing:
        return _fail(f"capture flags dropped from LIBTPU_INIT_ARGS: {missing}. "
                     f"Something rewrote the variable after this file set it; "
                     f"the artifacts would be silently absent.", 11)
    if args.mode in ("trace", "both"):
        if jax.__version__.split(".")[:2] != ["0", "11"]:
            return _fail(f"jax is {jax.__version__}, not 0.11.x. "
                         f"jax.profiler.trace() SIGSEGVs on 0.10.2 against "
                         f"libtpu 0.0.44.1 and takes the process with it.", 8)

    # The mapped source is the thing being profiled -- prove which file it is.
    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as sk
    print(f"kernel module: {sk.__file__}", flush=True)
    print(f"kernel md5   : "
          f"{subprocess.run(['md5sum', sk.__file__], capture_output=True, text=True).stdout.strip()}",
          flush=True)

    rng = np.random.default_rng(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    manifest = {"arm": args.arm, "mode": args.mode, "jax": jax.__version__,
                "libtpu_init_args": live, "kernel_file": sk.__file__,
                "b": args.b, "n": args.n, "k": args.k, "calls": args.calls}

    thunks = {}
    lowerable = {}

    if args.arm in ("incumbent", "incumbent_keys"):
        b, n, k = args.b, args.n, args.k
        p = _pick_partition(b, n, k)
        manifest["P"] = p
        print(f"shape: b={b} n={n} k={k} -> P={p} "
              f"slice_len={((n + 15) // 16) * 16} "
              f"num_waves={(b + 31) // 32}", flush=True)
        if p != 1:
            return _fail(f"expected the single-stage path, got P={p}. "
                         f"The scope name and the whole reading would be for "
                         f"a different kernel.", 9)
        scores = jnp.asarray(rng.standard_normal((b, n), dtype=np.float32))
        rl = jnp.asarray(np.full(b, n, np.int32))
        if args.arm == "incumbent":
            f = jax.jit(lambda s, l: sparsecore_topk(s, k, l))
            thunks["incumbent"] = lambda: f(scores, rl)
            lowerable["incumbent"] = (f, (scores, rl))
            manifest["expect_scope"] = f"sc_topk_direct_b{b}_n{n}_k{k}"
        else:
            g = jax.jit(lambda s, l: _sc_topk_direct(
                s, k, l, write_empty=True, emit_keys=True))
            thunks["incumbent_keys"] = lambda: g(scores, rl)
            lowerable["incumbent_keys"] = (g, (scores, rl))
            manifest["expect_scope"] = f"sc_topk_direct_b{b}_n{n}_k{k}_keys"
    else:
        # The pass-probe ladder from the measured harness, reused verbatim so
        # the static reading and the timed one are of the same programs.
        import sctopk_2stage_bench as bench
        slice_len = args.n
        waves = (args.b + 31) // 32
        # bitcast, exactly as `passes()` builds it, so the static reading and
        # the timed ladder are of byte-identical programs.
        src = jax.lax.bitcast_convert_type(
            jnp.asarray(rng.standard_normal((args.b, slice_len),
                                            dtype=np.float32)), jnp.int32)
        manifest["slice_len"] = slice_len
        manifest["waves"] = waves
        if args.arm == "probe_scatter":
            spec = [("cvt_p1", ("cvt", 1)), ("scatter_p1", ("scatter", 1)),
                    ("scatter_p2", ("scatter", 2))]
        else:
            spec = [("emit1_p1", ("emit1", 1)), ("emit2_p1", ("emit2", 1))]
        for label, (kind, npass) in spec:
            thunks[label] = bench.make_pass_thunk(src, slice_len, waves,
                                                  npass, kind, k=args.k)
            print(f"built probe {label}: kind={kind} passes={npass} "
                  f"scope=sc_pass_{kind}_p{npass}_sl{slice_len}_w{waves}",
                  flush=True)
        if args.arm == "probe_skeleton":
            thunks["dma_only"] = bench.make_dma_thunk(src, slice_len, waves)
            print(f"built probe dma_only: "
                  f"scope=sc_dma_only_sl{slice_len}_w{waves}", flush=True)
        manifest["expect_scope"] = sorted(thunks)

    # ---- warm EVERY thunk before ANY capture ------------------------------
    for name, th in thunks.items():
        jax.block_until_ready(th())
        print(f"warmed {name}", flush=True)

    # ---- StableHLO straight off the lowering, independent of dump flags --
    # The lowered text carries the serialized Mosaic module in the
    # tpu_custom_call backend_config, so Q6 does not depend on
    # --xla_mosaic_dump_to working for a SparseCore program.
    for name, (fn, fargs) in lowerable.items():
        try:
            txt = fn.lower(*fargs).as_text()
            dst = os.path.join(args.outdir, f"stablehlo_{name}.txt.gz")
            with gzip.open(dst, "wt") as fh:
                fh.write(txt)
            print(f"stablehlo {name}: {len(txt):,} chars -> {dst}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"stablehlo for {name} unavailable: {type(exc).__name__}: "
                  f"{exc}", flush=True)

    # ---- trace ------------------------------------------------------------
    if args.mode in ("trace", "both"):
        burn = os.path.join(args.outdir, "_burn")
        with jax.profiler.trace(burn):
            jax.block_until_ready(next(iter(thunks.values()))())
        print(f"burned a throwaway session -> {burn}", flush=True)
        manifest["traces"] = {}
        for name, th in thunks.items():
            d = os.path.join(args.outdir, "trace_" + name)
            os.makedirs(d, exist_ok=True)
            with jax.profiler.trace(d):
                for _ in range(args.calls):
                    jax.block_until_ready(th())
            pbs = sorted(glob.glob(os.path.join(d, "**", "*.xplane.pb"),
                                   recursive=True))
            manifest["traces"][name] = [{"path": p, "bytes": os.path.getsize(p)}
                                        for p in pbs]
            print(f"\n  trace {name}: {len(pbs)} xplane.pb", flush=True)
            for p in pbs:
                print(f"      {os.path.getsize(p):>13,} B  {p}", flush=True)
            if not pbs:
                return _fail(f"{name}: NO .xplane.pb written. A green job with "
                             f"no profile in it is the documented failure mode.",
                             10)
    else:
        # A dump-only run still has to execute, or the compiler may never be
        # asked for the program at all.
        for name, th in thunks.items():
            for _ in range(args.calls):
                jax.block_until_ready(th())
        print(f"ran {args.calls} calls of each thunk (dump-only)", flush=True)

    # ---- dump accounting, and prune if it is too big to sync --------------
    if args.mode in ("dump", "both"):
        for label, path in (("jf", _JF), ("mosaic", _MOSAIC),
                            ("scbundles", _SCB), ("scmlir", _SCM),
                            ("scllvm", _SCL)):
            tot, nf = _du(path)
            print(f"{label} dump: {nf:,} files  {tot / 2**30:.2f} GiB  {path}",
                  flush=True)
            manifest[f"{label}_bytes"] = tot
            manifest[f"{label}_files"] = nf
        tot, _ = _du(_JF)
        if tot > args.prune_gib * 2**30:
            keep = ("sc_topk_direct", "sc_pass_", "sc_dma_only")
            removed = kept = 0
            for root, _, files in os.walk(_JF):
                for f in files:
                    if not any(t in f for t in keep):
                        os.remove(os.path.join(root, f))
                        removed += 1
                    else:
                        kept += 1
            tot2, nf2 = _du(_JF)
            print(f"!! jf dump pruned to the kernel programs: removed "
                  f"{removed:,}, kept {kept:,} -> {nf2:,} files "
                  f"{tot2 / 2**30:.2f} GiB", flush=True)
            manifest["jf_pruned"] = {"removed": removed, "kept": kept,
                                     "bytes_after": tot2}
        # An index of what landed, so the reader does not have to walk 6k files
        names = sorted({os.path.basename(p)
                        for p in glob.glob(os.path.join(_JF, "**", "*"),
                                           recursive=True)
                        if os.path.isfile(p)})
        with gzip.open(os.path.join(args.outdir, "jf_filelist.txt.gz"),
                       "wt") as fh:
            fh.write("\n".join(names))
        mnames = sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(_MOSAIC, "*"))
                        if os.path.isfile(p))
        print(f"mosaic files ({len(mnames)}):", flush=True)
        for m in mnames[:60]:
            print(f"    {m}", flush=True)
        for label, path in (("scbundles", _SCB), ("scmlir", _SCM),
                            ("scllvm", _SCL)):
            got = sorted(glob.glob(os.path.join(path, "**", "*"),
                                   recursive=True))
            got = [g for g in got if os.path.isfile(g)]
            print(f"{label}: {len(got)} files", flush=True)
            for g in got[:40]:
                print(f"    {os.path.getsize(g):>12,} B  "
                      f"{os.path.relpath(g, path)}", flush=True)
        for extra in (_SCHWM, _SCS_PROG):
            print(f"{os.path.basename(extra)}: "
                  f"{os.path.getsize(extra) if os.path.exists(extra) else -1} B",
                  flush=True)

    with open(os.path.join(args.outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("\nNOTE: wall-clock inside a fanned-out capture is contended and is "
          "NOT a measurement.", flush=True)
    print("CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
