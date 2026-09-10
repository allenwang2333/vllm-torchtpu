"""STATIC COMPILE-TIME evidence for variant v2 (R=1, one shared histogram).

No TPU is attached. `jax.experimental.topologies.get_topology_desc` builds a
tpu7x:2x2x1 topology from libtpu on the host, so `.compile()` runs the real
XLA-TPU + Mosaic SparseCore pipeline. Prologue and topology handle are
probes/17's, verbatim. NO TIMING IS PRODUCED OR IMPLIED.

Arms, all at the production single-stage shape b=1024, n=10240, k=2048:

  INC     the incumbent, `git show b11b6eb6:src/.../sparsecore_topk.py`
  V2      this folder's `sparsecore_topk.py`
  V2F     V2 with `fold_priv` written the way probes/08's `arm-R1.py` wrote it
          (`for m in range(1)` over a zero-seeded accumulator) and with the
          incumbent's 16x-sized `priv_vmem` scratch. This is the fidelity
          check: if V2 and V2F emit the same Mosaic module, the readable form
          in V2 is the measured mechanism and not a new one.
  V2FOLD  V2 with ONLY the fold_priv spelling reverted (scratch stays 1 KiB),
          isolating the spelling from the allocation.
  NEG     8 MiB of VMEM scratch. MUST FAIL with E3000
          CompileTimeSparseCoreAllocationFailure in memory space spmem;
          without this the greens above prove nothing.
"""

import contextlib
import hashlib
import importlib.util
import io
import pathlib
import re
import subprocess
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

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[3]
B, N, K = 1024, 10240, 2048

FOLD_V2 = """        def body(i):
            off = i * LANES
            glob_vmem[pl.ds(off, LANES)] = priv_vmem[pl.ds(off, LANES)]
"""
FOLD_ARM_R1 = """        def body(i):
            off = i * LANES
            acc = jnp.zeros((LANES, ), jnp.int32)
            for m in range(1):
                acc = acc + priv_vmem[pl.ds(m * NUM_BUCKETS + off, LANES)]
            glob_vmem[pl.ds(off, LANES)] = acc
"""
SCRATCH_V2 = "        pltpu.VMEM((NUM_BUCKETS + LANES, ), jnp.int32),\n"
SCRATCH_INC = "        pltpu.VMEM((LANES * NUM_BUCKETS + LANES, ), jnp.int32),\n"


def incumbent_text():
    return subprocess.run(
        ["git", "show",
         "b11b6eb6:src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout


def load(name, text):
    path = HERE / f"_gen_{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"sctopk_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def entry(hlo):
    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    return m.group(0) if m else hlo


DEBUG_ANCHOR = "        compiler_params=pltpu.CompilerParams("


def mosaic_dump(mod_text, name):
    """The pre-serialization Mosaic module, via pl.kernel(debug=True)."""
    mod = load(f"{name}_dbg",
               mod_text.replace(DEBUG_ANCHOR, "        debug=True,\n" +
                                DEBUG_ANCHOR, 1))
    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)
    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=sh)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.jit(lambda s, ln: mod._sc_topk_direct(s, K, ln)).trace(
            scores, lens).lower(lowering_platforms=("tpu", ))
    txt = buf.getvalue()
    (HERE / f"mosaic-{name}.txt").write_text(txt)
    return txt


OPS = ("tpu.sc_addupdate_scatter", "addupdate", "scatter", "memref.alloca",
       "tpu.enqueue_dma", "vector.load", "vector.store", "arith.addi",
       "scf.for", "tpu.sc_strided_load", "tpu.sc_strided_store")


def compile_variant(name, text):
    mod = load(name, text)
    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)
    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=sh)
    fn = jax.jit(lambda s, ln: mod._sc_topk_direct(s, K, ln), out_shardings=sh)
    traced = fn.trace(scores, lens)
    hlo = traced.lower().compile().as_text()
    (HERE / f"hlo-{name}.txt").write_text(hlo)
    e = entry(hlo)
    mos = mosaic_dump(text, name)
    print(f"--- {name}")
    print(f"  ENTRY instructions        : {len(e.splitlines()) - 2}")
    print(f"  optimized HLO chars       : {len(hlo)}")
    print(f"  Mosaic MLIR lines         : {len(mos.splitlines())}")
    for pat in OPS:
        print(f"  {pat:26s}: {mos.count(pat)}")
    allocas = [ln.strip()[:120] for ln in mos.splitlines()
               if "memref.alloca" in ln]
    for a in allocas:
        print(f"      alloca: {a}")
    for pat in ("spill", "sparse_core"):
        hits = [ln.strip()[:130] for ln in hlo.splitlines() if pat in ln]
        print(f"  {pat:26s}: {len(hits)} HLO line(s)")
    return {"hlo": hlo, "entry": e, "mosaic": mos}


def neg_control():
    """8 MiB VMEM scratch on a 512 KiB SC tile. Must fail."""

    def kernel(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0],
                                   sems.at[0])
        cp.start()
        cp.wait()

    mesh = plsc.VectorSubcoreMesh(
        num_cores=_TPU7X.sparse_core.num_cores,
        num_subcores=_TPU7X.sparse_core.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore")
    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)
    fn = jax.jit(lambda: pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
        mesh=mesh,
        scratch_types=(pltpu.VMEM((2 * 1024 * 1024, ), jnp.int32),
                       pltpu.SemaphoreType.DMA((1, ))),
        name="aot_negctl_oversized_vmem")(), out_shardings=sh)
    try:
        fn.trace().lower().compile()
    except Exception as exc:  # noqa: BLE001
        txt = str(exc)
        ok = ("E3000" in txt and "spmem" in txt
              and "SparseCoreAllocationFailure" in txt)
        print(f"NEG control FAILED as required: {ok}")
        print("   ", txt.replace("\n", " ")[:260])
        return ok
    print("NEG control COMPILED -- every green above is worthless.")
    return False


if __name__ == "__main__":
    inc = incumbent_text()
    print("jax", jax.__version__)
    print("incumbent md5", hashlib.md5(inc.encode()).hexdigest())
    v2 = (HERE / "sparsecore_topk.py").read_text()
    print("v2        md5", hashlib.md5(v2.encode()).hexdigest())
    assert FOLD_V2 in v2 and SCRATCH_V2 in v2, "v2 anchors moved"
    v2f = v2.replace(FOLD_V2, FOLD_ARM_R1).replace(SCRATCH_V2, SCRATCH_INC, 1)
    v2fold = v2.replace(FOLD_V2, FOLD_ARM_R1)

    res = {}
    for name, text in [("inc", inc), ("v2", v2), ("v2f", v2f),
                       ("v2fold", v2fold)]:
        res[name] = compile_variant(name, text)

    print("=" * 72)
    for a, b in (("v2", "v2fold"), ("v2", "v2f"), ("inc", "v2")):
        same_m = res[a]["mosaic"] == res[b]["mosaic"]
        same_e = res[a]["entry"] == res[b]["entry"]
        print(f"  {a} vs {b}: mosaic-identical={same_m} entry-identical={same_e}")
    print("=" * 72)
    neg_control()
