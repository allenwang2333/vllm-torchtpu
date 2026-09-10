"""ONE arm, ONE process. Cross-check for `static_compile.py`.

Three traps this addresses, all of which return a confident wrong answer
rather than an error:

  T1  `jax.jit` keys its cache on (function object, avals). An in-model A/B
      done by rebinding a module attribute silently compares an arm with
      itself. `static_compile.py` loads each arm as its own module object, so
      it should be immune -- but "should be" is not evidence. This script
      compiles exactly one arm per process, so no cache can be shared, and it
      prints the identity of the bound symbol (module __file__, source md5,
      code object filename, id) before compiling. Run the VARIANT first, so a
      residual bug reads as base-looking-like-variant, not the reverse.

  T3  `XLA_FLAGS=--xla_dump_to=<dir>` yields buffer-assignment,
      memory-usage-report and sparse_core_specific_metadata files that
      `compiled.as_text()` does not contain. SparseCore sequencer bundle
      counts and VMEM-vs-HBM placement are legible only there. Set it in the
      environment BEFORE this file is imported (jax reads it at backend init).

  T2 (the negative control) stays in `static_compile.py`: an 8 MiB VMEM
      scratch DMA'd to HBM, which must be refused with
      `E3000: CompileTimeSparseCoreAllocationFailure ... memory space spmem`.

Usage:  python aot_single_arm.py <v2|inc|bigscratch>
No TPU. No timing is produced or implied.
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

from jax.experimental import pallas as pl  # noqa: E402,F401
from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[3]
B, N, K = 1024, 10240, 2048
DEBUG_ANCHOR = "        compiler_params=pltpu.CompilerParams("


def arm_text(arm):
    if arm == "inc":
        return subprocess.run(
            ["git", "show",
             "b11b6eb6:src/vllm_torchtpu/kernels/deepseek_v4/"
             "sparsecore_topk.py"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
    if arm == "v2":
        return (HERE / "sparsecore_topk.py").read_text()
    if arm == "bigscratch":
        return (HERE / "sparsecore_topk_bigscratch.py").read_text()
    raise SystemExit(f"unknown arm {arm}")


def load(name, text):
    path = HERE / f"_solo_{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"solo_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(arm):
    text = arm_text(arm)
    md5 = hashlib.md5(text.encode()).hexdigest()
    mod = load(arm, text)
    fn = mod._sc_topk_direct

    # T1: prove which source this process is actually about to compile.
    print(f"=== SOLO ARM {arm}   jax {jax.__version__}")
    print(f"  source md5           : {md5}")
    print(f"  module __file__      : {mod.__file__}")
    print(f"  fn.__code__.co_file  : {fn.__code__.co_filename}")
    print(f"  fn id                : {hex(id(fn))}")
    # Marker strings that differ between the arms, read off the SOURCE that
    # this module object was built from -- not off any name we chose.
    src = pathlib.Path(mod.__file__).read_text()
    print(f"  'lane_iota * NUM_BUCKETS + bucket' occurrences: "
          f"{src.count('lane_iota * NUM_BUCKETS + bucket')}  (inc=2, R1=0)")
    print(f"  'NUM_BUCKETS // LANES' occurrences            : "
          f"{src.count('NUM_BUCKETS // LANES')}  (inc=0, v2=2)")
    print(f"  scratch 'LANES * NUM_BUCKETS + LANES'          : "
          f"{src.count('pltpu.VMEM((LANES * NUM_BUCKETS + LANES, )')}"
          f"  (1 = 16 KiB alloc, 0 = 1 KiB alloc)")

    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)
    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=sh)

    jf = jax.jit(lambda s, ln: fn(s, K, ln), out_shardings=sh)
    hlo = jf.trace(scores, lens).lower().compile().as_text()
    (HERE / f"solo-hlo-{arm}.txt").write_text(hlo)

    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    ent = m.group(0) if m else hlo
    print(f"  ENTRY instructions   : {len(ent.splitlines()) - 2}")
    print(f"  optimized HLO chars  : {len(hlo)}")
    print(f"  ENTRY md5            : "
          f"{hashlib.md5(ent.encode()).hexdigest()}")
    print(f"  full HLO md5         : {hashlib.md5(hlo.encode()).hexdigest()}")

    # The Pallas jaxpr, via pl.kernel(debug=True). Fresh module, fresh jit.
    dbg = load(f"{arm}_dbg",
               text.replace(DEBUG_ANCHOR, "        debug=True,\n"
                            + DEBUG_ANCHOR, 1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.jit(lambda s, ln: dbg._sc_topk_direct(s, K, ln)).trace(
            scores, lens).lower(lowering_platforms=("tpu", ))
    jx = buf.getvalue()
    (HERE / f"solo-jaxpr-{arm}.txt").write_text(jx)
    eqns = len(re.findall(r"^\s+\S+:\S+\[?.*?\]? = ", jx, re.M))
    print(f"  jaxpr lines          : {len(jx.splitlines())}")
    print(f"  jaxpr equations      : {eqns}")
    print(f"  'add ' equations     : {len(re.findall(r'= add ', jx))}")
    vm = {}
    for w in re.findall(r"Ref<vmem>\{i32\[(\d+)\]\}", jx):
        vm[w] = vm.get(w, 0) + 1
    print(f"  vmem ref census      : {vm}")
    print(f"  jaxpr md5            : {hashlib.md5(jx.encode()).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1])
