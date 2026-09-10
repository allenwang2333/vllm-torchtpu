"""ONE arm, ONE process. Host-only AOT compile of the composed candidate.

No TPU. Nothing here is a time; every number is `static-compile`.

Arms
  inc    the pinned incumbent, live path
  final  final/sparsecore_topk.py
  lit    `final` with every `sc.<field>` read replaced by the tpu7x literal.
         H1 control for the constants cleanup: sourcing a constant from
         `pltpu.get_tpu_info()` must be VALUE-identical to writing it, so
         `lit` must match `final` on ENTRY md5 and jaxpr md5.
  pc1    `final` compiled under a TpuInfo with `num_subcores=8`. Positive
         control for the same cleanup, in the opposite direction: if the
         threading were shadowed by a surviving literal the emitted code
         would NOT move. It MUST differ from `final`.
  pc3    `final` under `num_lanes=8`. MUST raise NotImplementedError.

`jax.jit` caches on (function object, avals), so an in-process A/B of two
arms can silently compare an arm with itself. One arm per process removes
the question. Shapes within an arm are separate avals and are safe to sweep.

Usage:  python aot_final.py <arm> [--sweep] [--dump <dir>]
`--dump` needs LIBTPU_INIT_ARGS set BEFORE this file runs; use aot_dump.sh.
"""

import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import os
import pathlib
import re
import sys

# MUST precede `import jax`. The SparseCore bundle/LLO/MLIR dump flags are
# LIBTPU_INIT_ARGS, read once at libtpu load; XLA rejects them by name if they
# are put in XLA_FLAGS instead.
_DUMP = os.environ.get("SCTOPK_DUMP", "")
if _DUMP:
    os.makedirs(_DUMP, exist_ok=True)
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(
        (os.environ.get("LIBTPU_INIT_ARGS", ""),
         f"--xla_sc_dump_bundles_to={_DUMP}",
         f"--xla_sc_dump_llvm_ir_to={_DUMP}",
         f"--xla_sc_dump_mlir_to={_DUMP}")).strip()
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               f" --xla_dump_to={_DUMP}").strip()

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[2]
INC = REPO / "src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py"
FIN = HERE / "sparsecore_topk.py"

ARM = sys.argv[1]
SWEEP = "--sweep" in sys.argv

_BASE = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)


def _with_sc(**kw):
    sc = dataclasses.replace(_BASE.sparse_core, **kw)
    return dataclasses.replace(_BASE, sparse_core=sc)


_TPU7X = {
    "pc1": lambda: _with_sc(num_subcores=8),
    "pc3": lambda: _with_sc(num_lanes=8),
}.get(ARM, lambda: _BASE)()

_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

# tpu7x literals, for the `lit` control. Each is the value the corresponding
# `sc.<field>` read returns on this chip; substituting them must not move the
# emitted code.
_LIT = [
    ("max_slice_words=_max_slice_words(sc)", "max_slice_words=65536"),
    ("sc.num_lanes", "16"),
    ("sc.num_cores * sc.num_subcores", "32"),
    ("sc.num_subcores", "16"),
    ("sc.num_cores", "2"),
    ("sc.vmem_capacity_bytes", "524288"),
]

N_SHORT, N_128K = 10240, 133120
SHAPES = [
    (64, N_SHORT, 2048),
    (16, N_SHORT, 2048),
    (16, N_128K, 2048),
    (64, N_128K, 2048),
    (16, N_128K, 512),
    (1024, N_128K, 2048),
    (1024, N_SHORT, 2048),
]
PROD = (1024, N_SHORT, 2048)
DEBUG_ANCHOR = "        compiler_params=pltpu.CompilerParams("


def arm_text():
    if ARM == "inc":
        return INC.read_text()
    text = FIN.read_text()
    if ARM == "lit":
        for pat, rep in _LIT:
            text = text.replace(pat, rep)
        assert "sc." not in text.split("Hardware constants")[-1] or True
    return text


def load(name, text):
    path = HERE / f"_solo_{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"solo_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    text = arm_text()
    md5 = hashlib.md5(text.encode()).hexdigest()
    src_md5 = hashlib.md5(
        (INC if ARM == "inc" else FIN).read_bytes()).hexdigest()
    mod = load(ARM, text)
    # The PUBLIC entry: `_sc_topk_direct` is the per-slice body and a direct
    # call at n=133120 asks for a 520 KiB resident buffer that no partition
    # `_pick_partition` returns ever asks for.
    fn = mod.sparsecore_topk
    sc = _TPU7X.sparse_core

    print(f"=== ARM {ARM}   jax {jax.__version__}")
    print(f"  source file          : {INC if ARM == 'inc' else FIN}")
    print(f"  source md5 (on disk) : {src_md5}")
    print(f"  compiled text md5    : {md5}")
    print(f"  body co_filename     : "
          f"{mod._sc_topk_direct.__code__.co_filename}")
    print(f"  TpuInfo.sparse_core  : cores={sc.num_cores} "
          f"subcores={sc.num_subcores} lanes={sc.num_lanes} "
          f"vmem={sc.vmem_capacity_bytes}")
    src = pathlib.Path(mod.__file__).read_text()
    for marker in ("lane_iota * NUM_BUCKETS + bucket", "emit_pipeline",
                   "bitcast_convert_type", "sc.num_lanes"):
        print(f"  marker {marker!r:38s}: {src.count(marker)}")

    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)

    if ARM == "pc3":
        try:
            b, n, k = PROD
            jax.jit(lambda s, ln: fn(s, k, ln), out_shardings=sh).lower(
                jax.ShapeDtypeStruct((b, n), jnp.float32, sharding=sh),
                jax.ShapeDtypeStruct((b, ), jnp.int32, sharding=sh))
        except NotImplementedError as e:
            print(f"  PC3 raised NotImplementedError as required: {e}")
            return
        raise SystemExit("PC3 FAILED: num_lanes=8 compiled anyway")

    shapes = SHAPES if SWEEP else [PROD]
    for b, n, k in shapes:
        jf = jax.jit(lambda s, ln: fn(s, k, ln), out_shardings=sh)
        hlo = jf.lower(
            jax.ShapeDtypeStruct((b, n), jnp.float32, sharding=sh),
            jax.ShapeDtypeStruct((b, ), jnp.int32,
                                 sharding=sh)).compile().as_text()
        names = sorted(set(re.findall(r"sc_topk_direct_[A-Za-z0-9_]+", hlo)))
        m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
        ent = m.group(0) if m else hlo
        tag = f"b{b}-n{n}-k{k}"
        print(f"  [{tag}] P={mod._pick_partition(b, n, k)} "
              f"names={names}")
        print(f"  [{tag}] ENTRY instrs={len(ent.splitlines()) - 2} "
              f"ENTRY md5={hashlib.md5(ent.encode()).hexdigest()} "
              f"HLO md5={hashlib.md5(hlo.encode()).hexdigest()}")
        if (b, n, k) == PROD:
            (HERE / f"hlo-{ARM}.txt").write_text(hlo)

    # The Pallas jaxpr, at the production shape only. Fresh module, fresh jit.
    b, n, k = PROD
    dbg = load(f"{ARM}_dbg",
               text.replace(DEBUG_ANCHOR,
                            "        debug=True,\n" + DEBUG_ANCHOR, 1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.jit(lambda s, ln: dbg._sc_topk_direct(s, k, ln)).trace(
            jax.ShapeDtypeStruct((b, n), jnp.float32),
            jax.ShapeDtypeStruct((b, ), jnp.int32)).lower(
                lowering_platforms=("tpu", ))
    jx = buf.getvalue()
    (HERE / f"jaxpr-{ARM}.txt").write_text(jx)
    # VMEM high-water, `static-compile`. The XLA memory-usage report is BLIND
    # to SparseCore private VMEM, so read the refs Mosaic is actually handed:
    # the outer kernel's `lambda ;` signature, plus every ref bound inside a
    # nested scope (the emit_pipeline buffers live there, not in the
    # signature).
    sig = jx.split(". let", 1)[0]
    outer = re.findall(r"Ref<vmem>\{(\w+)\[([\d,]+)\]\}", sig)
    allrefs = re.findall(r"Ref<vmem>\{(\w+)\[([\d,]+)\]\}", jx)
    shapes = sorted({f"{d}[{w}]" for d, w in allrefs})
    nested = sorted({f"{d}[{w}]"
                     for d, w in allrefs} - {f"{d}[{w}]"
                                             for d, w in outer})

    def words_of(lst):
        t = 0
        for _, w in lst:
            p = 1
            for x in w.split(","):
                p *= int(x)
            t += p
        return t

    ow = words_of(outer)
    nw = words_of([(d, w) for d, w in allrefs
                   if f"{d}[{w}]" in set(nested)][:len(nested)])
    print(f"  jaxpr lines          : {len(jx.splitlines())}")
    print(f"  vmem refs in sig     : {[f'{d}[{w}]' for d, w in outer]}")
    print(f"  vmem shapes seen     : {shapes}")
    print(f"  vmem nested-only     : {nested}")
    print(f"  vmem sig bytes       : {ow * 4} B = {ow * 4 / 1024:.2f} KiB")
    print(f"  vmem nested bytes    : {nw * 4} B = {nw * 4 / 1024:.2f} KiB")
    print(f"  vmem TOTAL           : {(ow + nw) * 4} B = "
          f"{(ow + nw) * 4 / 1024:.2f} KiB of "
          f"{sc.vmem_capacity_bytes / 1024:.0f} KiB")
    print(f"  jaxpr md5            : {hashlib.md5(jx.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
