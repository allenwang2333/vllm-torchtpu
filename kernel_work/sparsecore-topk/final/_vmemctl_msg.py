"""Read the ALLOCATOR's own accounting at the first refusal, per arm, and
pin the composed candidate's Python VMEM guard against it."""
import importlib.util, pathlib, re, sys
import jax, jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info
T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import SingleDeviceSharding
pltpu.get_tpu_info = lambda: T
SH = SingleDeviceSharding(topologies.get_topology_desc(topology_name="tpu7x:2x2x1", platform="tpu").devices[0])
HERE = pathlib.Path(__file__).parent
INC = HERE.parents[2] / "src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py"
FIN = HERE / "sparsecore_topk.py"
GUARD = "    if vmem_bytes > sc.vmem_capacity_bytes:"
SRC = {"base": INC.read_text(), "final": FIN.read_text(),
       "final-noguard": FIN.read_text().replace(GUARD, "    if False:", 1)}
mods = {}
for t, txt in SRC.items():
    p = HERE / f"_vmsg_{t.replace('-','_')}.py"; p.write_text(txt)
    s = importlib.util.spec_from_file_location(f"mm_{t}", p)
    m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
    mods[t] = m
def go(mod, n):
    try:
        jax.jit(lambda s, l: mod._sc_topk_direct(s, 2048, l), out_shardings=SH).lower(
            jax.ShapeDtypeStruct((16, n), jnp.float32), jax.ShapeDtypeStruct((16,), jnp.int32)).compile()
        return "ok", ""
    except NotImplementedError as e:
        return "guard", re.sub(r"\s+", " ", str(e))
    except Exception as e:
        s = re.sub(r"\s+", " ", str(e))
        if "spmem" in s and "SparseCoreAllocation" in s:
            m = re.search(r"[Rr]equest\w*[^.;]*|upper bound \(\d+ words\)|\d+ words", s)
            return "alloc", s[:400]
        return "other", s[:200]
print("== allocator message at first refusal")
for t, n in (("base", 124288), ("final-noguard", 124800)):
    k, msg = go(mods[t], n)
    print(f"  {t:14s} n={n} -> {k}\n     {msg}")
print("== the composed candidate's OWN guard threshold, bisected at 16-word steps")
lo, hi = 122880, 126000
while hi - lo > 16:
    mid = ((lo + hi)//2//16)*16
    k, _ = go(mods["final"], mid)
    if k == "ok": lo = mid
    else: hi = mid
kf, mf = go(mods["final"], hi)
print(f"  guard: largest n accepted = {lo}, first refusal n = {hi} ({kf})")
print(f"     {mf}")
print(f"  allocator (final-noguard) accepted up to 124672, refused at 124800")
print(f"  => guard is {'CONSERVATIVE' if lo <= 124672 else 'PERMISSIVE'} by {124672 - lo} words")
