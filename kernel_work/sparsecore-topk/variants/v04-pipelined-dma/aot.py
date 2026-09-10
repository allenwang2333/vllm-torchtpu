"""v4 feasibility: AOT-compile the pipelined kernel with no TPU attached.

`jax.experimental.topologies.get_topology_desc` builds a tpu7x:2x2x1 topology
from libtpu on the host, so `.compile()` runs the full XLA TPU pipeline and the
SparseCore Mosaic backend. STATIC COMPILE-TIME ANALYSIS ONLY -- no timing is
produced and no number here is a performance claim.

Arms:
  base            the shipped incumbent (src/.../sparsecore_topk.py), control
  v4-c{512,1024,2048}   this directory's kernel at three chunk sizes
  v4-c1024-nopipe       the decomposition arm (synchronous copies)
and both at the decode shape as well as the prefill shape.

Negative controls (must FAIL, or the harness proves nothing):
  neg-vmem        8 MiB VMEM scratch -> SparseCore spmem allocator error
  neg-chunk       a chunk that does not divide n -> our own guard
"""

import importlib.util
import os
import pathlib
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

from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
V4 = HERE / "sparsecore_topk.py"
N, K = 10240, 2048
OUT = HERE / "aot"
OUT.mkdir(exist_ok=True)


def load(tag, path, env):
    for key, val in env.items():
        os.environ[key] = val
    spec = importlib.util.spec_from_file_location(f"sctopk_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def device():
    return topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                        platform="tpu").devices[0]


def entry(hlo):
    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    return m.group(0) if m else hlo


def compile_variant(tag, mod, b):
    sh = SingleDeviceSharding(device())
    scores = jax.ShapeDtypeStruct((b, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((b, ), jnp.int32, sharding=sh)
    fn = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l), out_shardings=sh)
    hlo = fn.lower(scores, lens).compile().as_text()
    e = entry(hlo)
    (OUT / f"entry-{tag}-b{b}.txt").write_text(e)
    names = sorted(set(re.findall(r"sc_topk_direct_b\d+_n\d+_k\d+\w*", hlo)))
    scoped = re.findall(r"used_scoped_memory_configs=\[[^]]*\]", e)
    print(f"  {tag:22s} b={b:<5d} OK  entry_instrs="
          f"{len(e.splitlines()) - 2:2d}  scoped_cfgs={len(scoped)}  "
          f"name={names[0] if names else '?'}")
    return e


if __name__ == "__main__":
    print("jax", jax.__version__)
    print("incumbent:", INCUMBENT)
    print("v4       :", V4)

    ents = {}
    print("\n--- positive arms (all must compile)")
    for tag, path, env, b in [
        ("base", INCUMBENT, {}, 1024),
        ("base", INCUMBENT, {}, 64),
        ("v4-c512", V4, {"SCTOPK_V4_CHUNK": "512", "SCTOPK_V4_NOPIPE": "0"},
         1024),
        ("v4-c1024", V4, {"SCTOPK_V4_CHUNK": "1024", "SCTOPK_V4_NOPIPE": "0"},
         1024),
        ("v4-c2048", V4, {"SCTOPK_V4_CHUNK": "2048", "SCTOPK_V4_NOPIPE": "0"},
         1024),
        ("v4-c1024-nopipe", V4,
         {"SCTOPK_V4_CHUNK": "1024", "SCTOPK_V4_NOPIPE": "1"}, 1024),
        ("v4-c1024", V4, {"SCTOPK_V4_CHUNK": "1024", "SCTOPK_V4_NOPIPE": "0"},
         64),
    ]:
        try:
            ents[f"{tag}-b{b}"] = compile_variant(tag, load(tag, path, env), b)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag:22s} b={b:<5d} FAILED {type(exc).__name__}: "
                  f"{str(exc)[:1200]}")

    print("\n--- negative controls (all must fail)")
    # 1. the spmem allocator really runs
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu_sc as plsc

        def big(o_hbm, scratch, sems):
            cp = pltpu.make_async_copy(scratch.at[pl.ds(0, K)], o_hbm.at[0],
                                       sems.at[0])
            cp.start()
            cp.wait()

        f = pl.kernel(
            big,
            out_type=jax.ShapeDtypeStruct((16, K), jnp.int32),
            mesh=plsc.VectorSubcoreMesh(num_cores=2,
                                        num_subcores=16,
                                        core_axis_name="core",
                                        subcore_axis_name="subcore"),
            scratch_types=(pltpu.VMEM((8 * 1024 * 1024 // 4, ), jnp.int32),
                           pltpu.SemaphoreType.DMA((1, ))),
            name="neg_vmem",
        )
        jax.jit(f, out_shardings=SingleDeviceSharding(
            device())).lower().compile()
        print("  neg-vmem               DID NOT FAIL -- harness is unsound")
    except Exception as exc:  # noqa: BLE001
        s = str(exc)
        ok = "spmem" in s or "SparseCoreAllocation" in s
        print(f"  neg-vmem               {'FAILED as required' if ok else s[:300]}"
              f"  [{'E3000/spmem' if ok else 'WRONG ERROR'}]")

    # 2. our own chunk guard: 3072 does not divide 10240
    m = load("v4chk", V4, {"SCTOPK_V4_CHUNK": "3072", "SCTOPK_V4_NOPIPE": "0"})
    picked = m._pick_chunk(N, 3072)
    print(f"  neg-chunk              _pick_chunk(10240, 3072) = {picked} "
          f"({'rounded down to a divisor' if N % picked == 0 else 'ILLEGAL'})")
    print(f"  legal divisors <= 4096 : "
          f"{[c for c in range(16, 4097, 16) if N % c == 0]}")

    # 3. is the divides-n rule enforced by the compiler, or only by us?
    #    Force an illegal chunk past the guard and see whether anything
    #    objects. If it compiles, the rule is OURS and a future edit that
    #    hardcodes a chunk gets silent cross-row reads, not an error.
    m._pick_chunk = lambda n, want: 3072
    try:
        compile_variant("neg-chunk-forced", m, 1024)
        print("  neg-chunk-forced       COMPILES -- the divides-n rule is not "
              "compiler-enforced; _pick_chunk is the only thing holding it")
    except Exception as exc:  # noqa: BLE001
        print(f"  neg-chunk-forced       rejected: "
              f"{type(exc).__name__}: {str(exc)[:300]}")

    if "base-b1024" in ents and "v4-c1024-b1024" in ents:
        import difflib
        d = "\n".join(
            difflib.unified_diff(ents["base-b1024"].splitlines(),
                                 ents["v4-c1024-b1024"].splitlines(),
                                 fromfile="ENTRY-base",
                                 tofile="ENTRY-v4-c1024",
                                 lineterm="",
                                 n=1))
        (OUT / "diff-base-vs-v4.txt").write_text(d)
        print(f"\nENTRY diff base -> v4-c1024: {len(d.splitlines())} lines "
              f"-> aot/diff-base-vs-v4.txt")
