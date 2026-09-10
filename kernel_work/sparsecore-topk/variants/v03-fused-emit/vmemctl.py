"""Price the kernel's PRIVATE SparseCore VMEM by allocation failure. No TPU.

XLA's memory-usage report is blind to Pallas/Mosaic SparseCore VMEM -- Mosaic
allocates below XLA, so the report's `vmem` table shows XLA-level buffers only
and is byte-identical between two arms that differ in kernel scratch.  The one
host-only instrument that can see the kernel's own allocation is the allocator
itself: sweep `n` until the compile is refused with E3000, for each arm.  The
ratio of the two thresholds is the per-element scratch ratio, measured.

Also runs the two controls that make the refusal meaningful:
  * NEGATIVE: 8 MiB VMEM scratch DMA'd to HBM -- must FAIL in the spmem allocator.
  * POSITIVE: 4 KiB VMEM scratch, IDENTICAL DMA form -- must COMPILE.
    Without it, "8 MiB fails" is equally consistent with the DMA form being
    broken for an unrelated reason.
"""
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
PATHS = {
    "base": os.path.join(ROOT, "src", "vllm_torchtpu", "kernels",
                         "deepseek_v4", "sparsecore_topk.py"),
    "v3": os.path.join(HERE, "sparsecore_topk.py"),
}
MESH = plsc.VectorSubcoreMesh(num_cores=_TPU7X.sparse_core.num_cores,
                              num_subcores=_TPU7X.sparse_core.num_subcores,
                              core_axis_name="core",
                              subcore_axis_name="subcore")


def probe(words, name):
    """Identical DMA form at two sizes: the only variable is the allocation."""
    def kernel(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, 512)], o_hbm.at[0],
                                   sems.at[0])
        cp.start()
        cp.wait()

    return pl.kernel(kernel,
                     out_type=jax.ShapeDtypeStruct((16, 512), jnp.int32),
                     mesh=MESH,
                     scratch_types=(pltpu.VMEM((words, ), jnp.int32),
                                    pltpu.SemaphoreType.DMA((1, ))),
                     name=name)()


print("=" * 74)
print("CONTROLS -- same kernel, same DMA form, allocation size is the only "
      "variable")
for words, want, tag in ((1024, "COMPILE", "POSITIVE  4 KiB"),
                         (2 * 1024 * 1024, "FAIL", "NEGATIVE  8 MiB")):
    try:
        n = len(jax.jit(lambda w=words: probe(w, f"aot_ctl_{w}"),
                        out_shardings=SH).lower().compile().as_text())
        got, det = "COMPILE", f"{n} chars of HLO"
    except Exception as e:  # noqa: BLE001
        s = re.sub(r"\s+", " ", str(e))
        got = "FAIL"
        det = s[:200]
        if "spmem" in s and "SparseCoreAllocation" in s:
            det = "E3000 CompileTimeSparseCoreAllocationFailure ... spmem  " + (
                re.search(r"upper bound \(\d+ words\)", s).group(0)
                if re.search(r"upper bound \(\d+ words\)", s) else "")
    print(f"  {tag:16s} want={want:8s} got={got:8s} {'OK' if got == want else '*** BROKEN ***'}")
    print(f"     {det}")

print("=" * 74)
print("PRIVATE VMEM, priced by allocation failure (the memory-usage report "
      "cannot see this)")
sc = _TPU7X.sparse_core
print(f"  sc.num_lanes={sc.num_lanes}  sc.num_cores={sc.num_cores}  "
      f"sc.num_subcores={sc.num_subcores}  "
      f"sc.vmem_capacity_bytes={sc.vmem_capacity_bytes} "
      f"({sc.vmem_capacity_bytes // 4} words)")

mods = {}
for tag, path in PATHS.items():
    spec = importlib.util.spec_from_file_location(f"m_{tag}", path)
    mods[tag] = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mods[tag]
    spec.loader.exec_module(mods[tag])


def try_n(mod, n):
    """-> 'ok' | 'guard' (our NotImplementedError) | 'alloc' (E3000) | other"""
    try:
        jax.jit(lambda s, l: mod._sc_topk_direct(s, 2048, l),
                out_shardings=SH).lower(
                    jax.ShapeDtypeStruct((16, n), jnp.float32),
                    jax.ShapeDtypeStruct((16, ), jnp.int32)).compile()
        return "ok"
    except NotImplementedError:
        return "guard"
    except Exception as e:  # noqa: BLE001
        s = str(e)
        if "spmem" in s and "SparseCoreAllocation" in s:
            return "alloc"
        return "other:" + re.sub(r"\s+", " ", s)[:120]


for tag in ("base", "v3"):
    lo, hi = 1024, 262144           # lo compiles, hi assumed to fail
    seen = {}
    r = try_n(mods[tag], hi)
    seen[hi] = r
    while hi - lo > 1024:
        mid = ((lo + hi) // 2 // 1024) * 1024
        r = try_n(mods[tag], mid)
        seen[mid] = r
        if r == "ok":
            lo = mid
        else:
            hi = mid
    print(f"  {tag:5s}: largest n that compiles = {lo:7d}   "
          f"first refusal at n = {hi:7d} ({seen.get(hi)})")
