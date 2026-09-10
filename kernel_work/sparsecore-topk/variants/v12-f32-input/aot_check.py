"""v12 static-compile evidence: incumbent vs f32-input, on a host with no TPU.

STATIC COMPILE-TIME ANALYSIS ONLY -- no timing is produced anywhere in this file.

Three things are produced, in this order:

  1. a negative control -- an 8 MiB VMEM scratch, which must FAIL with the
     SparseCore spmem allocator error. Without it, "AOT COMPILED ok" only means
     the MLIR verifier was happy, not that the SC backend ran;
  2. the post-optimization ENTRY of both arms, its instruction count, and the
     buffer-assignment / memory_analysis totals -- the check that the 40 MiB
     temp buffer is GONE rather than renamed, copied or re-staged;
  3. the Mosaic MLIR body of both arms, with every `tpu.bitcast` located
     against the loop nest it sits in.

Run: uv_venv/bin/python3 aot_check.py     (from this directory)
"""

import contextlib
import importlib.util
import io
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

# Pin a tpu7x description onto a host with no TPU attached. Verbatim from
# probes/17/hlo_diff.py:27-33, which is verbatim from probes/12.
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
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
VARIANT = HERE / "sparsecore_topk.py"

# report.md section 4: the shipping prefill shape. n is the padded page-aligned
# KV capacity, not 133120.
B, N, K = 1024, 10240, 2048

DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)


def load(tag, path):
    spec = importlib.util.spec_from_file_location(f"sctopk_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def entry(hlo):
    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    return m.group(0) if m else hlo


def compiled(mod):
    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=SH)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=SH)
    fn = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l), out_shardings=SH)
    return fn.lower(scores, lens).compile()


def negative_control():
    """8 MiB of VMEM scratch on a 512 KiB SC tile. Must not compile."""

    def kernel(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0], sems.at[0])
        cp.start()
        cp.wait()

    def build():
        return pl.kernel(
            kernel,
            out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
            mesh=plsc.VectorSubcoreMesh(
                num_cores=_TPU7X.sparse_core.num_cores,
                num_subcores=_TPU7X.sparse_core.num_subcores,
                core_axis_name="core",
                subcore_axis_name="subcore"),
            scratch_types=(pltpu.VMEM((2 * 1024 * 1024, ), jnp.int32),
                           pltpu.SemaphoreType.DMA((1, ))),
            name="aot_negctl_oversized_vmem")()

    try:
        jax.jit(build, out_shardings=SH).lower().compile()
        return "COMPILED -- CONTROL FAILED, every green below is worthless"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc)[:260]}"


def mosaic_body(mod):
    """The lowered Mosaic MLIR for one arm, via pl.kernel(debug=True)."""
    src = VARIANT if mod.__name__.endswith("variant") else INCUMBENT
    text = pathlib.Path(src).read_text().replace(
        "compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,",
        "debug=True,\n        "
        "compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,", 1)
    path = HERE / f"_dbg_{mod.__name__}.py"
    path.write_text(text)
    dbg = load(f"dbg_{mod.__name__}", path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scores = jax.ShapeDtypeStruct((B, N), jnp.float32)
        lens = jax.ShapeDtypeStruct((B, ), jnp.int32)
        jax.jit(lambda s, l: dbg._sc_topk_direct(s, K, l)).trace(
            scores, lens).lower(lowering_platforms=("tpu", ))
    return buf.getvalue()


if __name__ == "__main__":
    print("jax", jax.__version__)
    print(f"incumbent {INCUMBENT}")
    print(f"variant   {VARIANT}")
    print("\n===== NEGATIVE CONTROL (8 MiB VMEM scratch) =====")
    print(negative_control())

    arms = {}
    for tag, path in (("incumbent", INCUMBENT), ("variant", VARIANT)):
        mod = load(tag, path)
        mod.__name__ = tag
        c = compiled(mod)
        hlo = c.as_text()
        e = entry(hlo)
        (HERE / f"hlo-entry-{tag}.txt").write_text(e)
        (HERE / f"hlo-full-{tag}.txt").write_text(hlo)
        mem = c.memory_analysis()
        arms[tag] = (e, mem)
        n_instr = len(e.splitlines()) - 2
        print(f"\n===== {tag} =====")
        print(f"  ENTRY instructions        : {n_instr}")
        print(f"  bitcast-convert in ENTRY  : "
              f"{len(re.findall(r'= \S+ bitcast-convert\(', e))}")
        print(f"  fusion in ENTRY           : "
              f"{len(re.findall(r'= \S+ fusion\(', e))}")
        print(f"  copy/copy-start in ENTRY  : "
              f"{len(re.findall(r'= \S+ copy(-start)?\(', e))}")
        print(f"  temp_size_in_bytes        : {mem.temp_size_in_bytes}")
        print(f"  argument_size_in_bytes    : {mem.argument_size_in_bytes}")
        print(f"  output_size_in_bytes      : {mem.output_size_in_bytes}")
        print(f"  alias_size_in_bytes       : {mem.alias_size_in_bytes}")
        print(f"  generated_code_size_bytes : "
              f"{mem.generated_code_size_in_bytes}")
        tot = 0
        for m in re.finditer(r'"used_scoped_memory_configs":\[(.*?)\]', e):
            for v in re.findall(r'"amount_of_scoped_memory_in_bytes":"(\d+)"',
                                m.group(1)):
                tot += int(v)
        print(f"  sum(used_scoped) over ENTRY: {tot}")
        for ln in e.splitlines():
            s = ln.strip()
            if re.search(r"= \S+ (fusion|bitcast-convert|copy|copy-start|"
                         r"bitcast|reshape)\(", s):
                print(f"    [op] {s[:150]}")

    # Name-normalized ENTRY diff: v12 renames the kernel to ..._f32in, which
    # would otherwise show up on every line that mentions the custom call.
    import difflib
    norm = lambda t: re.sub(r"_f32in", "", t)  # noqa: E731
    d = "\n".join(
        difflib.unified_diff(norm(arms["incumbent"][0]).splitlines(),
                             norm(arms["variant"][0]).splitlines(),
                             fromfile="ENTRY-incumbent",
                             tofile="ENTRY-v12",
                             lineterm="",
                             n=1))
    (HERE / "hlo-entry-diff.txt").write_text(d)
    print(f"\n===== ENTRY diff, kernel name normalized "
          f"({len(d.splitlines())} lines) -> hlo-entry-diff.txt =====")
    print(d[:4000])

    print("\n===== MOSAIC BODY =====")
    for tag, path in (("incumbent", INCUMBENT), ("variant", VARIANT)):
        mod = load(tag, path)
        mod.__name__ = tag
        txt = mosaic_body(mod)
        (HERE / f"mosaic-{tag}.mlir").write_text(txt)
        lines = txt.splitlines()
        print(f"\n--- {tag}  ({len(lines)} lines) -> mosaic-{tag}.mlir")
        for pat in ("tpu.bitcast", "arith.bitcast", "tpu.enqueue_dma",
                    "memref<10240x"):
            hits = [i for i, ln in enumerate(lines) if pat in ln]
            print(f"  {pat:20s}: {len(hits)} occurrence(s)")
            for i in hits[:6]:
                print(f"      L{i+1}: {re.sub(r'loc[(].*', '', lines[i].strip())[:180]}")
