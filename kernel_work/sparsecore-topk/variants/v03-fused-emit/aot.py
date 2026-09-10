"""v3 (fused emit) STATIC COMPILE-TIME analysis. No TPU, no timing.

Compiles the incumbent and the v3 variant AOT against tpu7x:2x2x1 at the
production shape, diffs the optimized HLO ENTRY and the Mosaic MLIR, and runs
the mandatory spmem-allocator negative control.

Prologue is probes/17's, verbatim in effect: pin every tpu_info accessor to a
TPU7x descriptor so Pallas/Mosaic lower for SparseCore with no device present.
"""
import contextlib
import io
import importlib.util
import os
import re
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.experimental.pallas import tpu_sc as plsc  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
INCUMBENT = os.path.join(ROOT, "src", "vllm_torchtpu", "kernels",
                         "deepseek_v4", "sparsecore_topk.py")
VARIANT = os.path.join(HERE, "sparsecore_topk.py")

# Production GLM-5.2 DSA shape: prefill token bucket b=1024, n=9216+1024
# padded to 10240, k=2048, P=1.
B, N, K = 1024, 10240, 2048
# Decode: b is the token bucket, floor 16; the profiled decode bucket is 64.
B_DEC = 64


def load(tag, path):
    spec = importlib.util.spec_from_file_location(f"sctopk_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def compile_text(mod, b, n, k, emit_keys=False):
    def f(scores, lengths):
        return mod._sc_topk_direct(scores, k, lengths, emit_keys=emit_keys)

    scores = jax.ShapeDtypeStruct((b, n), jnp.float32)
    lengths = jax.ShapeDtypeStruct((b, ), jnp.int32)
    return jax.jit(f, out_shardings=SH).lower(scores,
                                              lengths).compile().as_text()


def entry(hlo):
    m = re.search(r"^ENTRY .*?^}", hlo, re.M | re.S)
    return m.group(0) if m else ""


def mosaic(mod, b, n, k, emit_keys=False):
    """Capture the Mosaic MLIR the kernel lowers to (debug=True prints it)."""
    orig = pl.kernel

    def patched(*a, **kw):
        kw["debug"] = True
        return orig(*a, **kw)

    buf = io.StringIO()
    mod.pl.kernel = patched
    try:
        with contextlib.redirect_stdout(buf):
            jax.jit(lambda s, l: mod._sc_topk_direct(
                s, k, l, emit_keys=emit_keys)).trace(
                    jax.ShapeDtypeStruct((b, n), jnp.float32),
                    jax.ShapeDtypeStruct((b, ), jnp.int32)).lower(
                        lowering_platforms=("tpu", ))
    finally:
        mod.pl.kernel = orig
    return buf.getvalue()


def oversized():
    """Mandatory negative control: 8 MiB of VMEM scratch on a 512 KiB tile.

    Must fail in the SparseCore allocator. If it compiles, every green in this
    file is worthless because Mosaic never ran.
    """
    def kernel(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0],
                                   sems.at[0])
        cp.start()
        cp.wait()

    return pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((16, K), jnp.int32),
        mesh=plsc.VectorSubcoreMesh(
            num_cores=_TPU7X.sparse_core.num_cores,
            num_subcores=_TPU7X.sparse_core.num_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore"),
        scratch_types=(pltpu.VMEM((2 * 1024 * 1024, ), jnp.int32),
                       pltpu.SemaphoreType.DMA((1, ))),
        name="aot_negctl_oversized_vmem")()


OPS = ("tpu.sc_strided_load", "tpu.sc_strided_store", "tpu.sc_load_gather",
       "tpu.sc_store_scatter", "tpu.sc_addupdate_scatter",
       "tpu.sc_store_compressed", "tpu.sc_cumsum",
       "tpu.sc_all_reduce", "tpu.enqueue_dma", "tpu.wait_dma", "scf.for",
       "scf.if", "vector.load", "vector.store", "memref.alloca",
       "arith.addi", "arith.cmpi", "arith.shrsi", "arith.andi")


def opcount(txt):
    out = {}
    for o in OPS:
        c = len(re.findall(re.escape(o), txt))
        if c:
            out[o] = c
    return out


if __name__ == "__main__":
    print("=" * 78)
    print("0. NEGATIVE CONTROL (must FAIL in the spmem allocator)")
    try:
        n = len(jax.jit(oversized, out_shardings=SH).lower().compile()
                .as_text())
        print(f"  AOT-NEGCTL-8MiB-vmem: *** COMPILED ({n} chars) -- "
              "ALL RESULTS BELOW ARE VOID ***")
    except Exception as e:  # noqa: BLE001
        s = str(e)
        ok = ("spmem" in s and "SparseCoreAllocation" in s)
        print(f"  AOT-NEGCTL-8MiB-vmem: FAILED as required -> "
              f"{type(e).__name__}")
        print("  " + re.sub(r"\s+", " ", s)[:300])
        print(f"  matches expected allocator signature: {ok}")

    mods = {}
    for tag, path in (("base", INCUMBENT), ("v3", VARIANT)):
        mods[tag] = load(tag, path)
        print(f"  loaded {tag}: {path}")

    print("=" * 78)
    print("1. AOT COMPILE, production shape b=%d n=%d k=%d P=1" % (B, N, K))
    hlo = {}
    for tag in ("base", "v3"):
        for ek in (False, True):
            key = (tag, ek)
            try:
                hlo[key] = compile_text(mods[tag], B, N, K, emit_keys=ek)
                e = entry(hlo[key])
                print(f"  {tag:5s} emit_keys={int(ek)}: COMPILED  "
                      f"hlo={len(hlo[key])} chars  "
                      f"ENTRY lines={len(e.splitlines())}")
            except Exception as ex:  # noqa: BLE001
                print(f"  {tag:5s} emit_keys={int(ek)}: FAILED -> "
                      f"{type(ex).__name__}: {re.sub(chr(10),' ',str(ex))[:400]}")

    print("  decode bucket b=%d:" % B_DEC)
    for tag in ("base", "v3"):
        try:
            t = compile_text(mods[tag], B_DEC, N, K)
            print(f"  {tag:5s} b={B_DEC}: COMPILED hlo={len(t)} chars")
        except Exception as ex:  # noqa: BLE001
            print(f"  {tag:5s} b={B_DEC}: FAILED -> {type(ex).__name__}: "
                  f"{re.sub(chr(10),' ',str(ex))[:300]}")

    print("  two-stage shape n=133120 P=4 (largest slice the tests reach):")
    for tag in ("base", "v3"):
        try:
            t = compile_text(mods[tag], 16, 133120 // 4, 2048, emit_keys=True)
            print(f"  {tag:5s} slice=33280 keys: COMPILED hlo={len(t)} chars")
        except Exception as ex:  # noqa: BLE001
            print(f"  {tag:5s} slice=33280 keys: FAILED -> "
                  f"{type(ex).__name__}: {re.sub(chr(10),' ',str(ex))[:300]}")

    print("=" * 78)
    print("2. custom-call / scratch shapes in the optimized HLO ENTRY")
    for key in sorted(hlo):
        e = entry(hlo[key])
        cc = [l.strip() for l in e.splitlines() if "custom-call" in l]
        print(f"  {key}: {len(e.splitlines())} ENTRY lines, "
              f"{len(cc)} custom-calls")
        for l in cc:
            print("    " + re.sub(r"\s+", " ", l)[:280])

    print("=" * 78)
    print("3. MOSAIC MLIR op census")
    mm = {}
    for tag in ("base", "v3"):
        try:
            mm[tag] = mosaic(mods[tag], B, N, K)
            open(os.path.join(HERE, f"mosaic_{tag}.mlir"), "w").write(mm[tag])
            print(f"  {tag}: {len(mm[tag].splitlines())} lines dumped to "
                  f"mosaic_{tag}.mlir")
        except Exception as ex:  # noqa: BLE001
            print(f"  {tag}: mosaic dump FAILED -> {type(ex).__name__}: "
                  f"{str(ex)[:300]}")
    if len(mm) == 2:
        a, b_ = opcount(mm["base"]), opcount(mm["v3"])
        keys = sorted(set(a) | set(b_))
        print(f"  {'op':28s} {'base':>8s} {'v3':>8s} {'delta':>8s}")
        for kk in keys:
            x, y = a.get(kk, 0), b_.get(kk, 0)
            print(f"  {kk:28s} {x:8d} {y:8d} {y-x:+8d}")
