"""Re-derive v12's headline numbers in FRESH SINGLE-ARM PROCESSES.

STATIC COMPILE-TIME ANALYSIS ONLY. No TPU, no job.

Answers three challenges raised against the AOT method, each of which can turn
a real difference into a clean-looking null (or the reverse):

  1. `jax.jit` caches on (function object, avals). `aot_check.py` and
     `aot_two_stage.py` compile BOTH arms in ONE process. They load each arm
     under a distinct module name via `spec_from_file_location`, so the traced
     callables are distinct objects and the cache key should differ -- but
     "should" is the word that makes it worth one cheap re-run. Every headline
     number is therefore recomputed here with exactly ONE arm per process.

  2. Order. If a residual cache or a lockfile retry ever did leak state, the
     safe ordering is VARIANT FIRST, so the failure mode is "base looks like
     variant" (loud: the base loses its 40 MiB buffer, which is impossible)
     rather than "variant looks like base" (silent: reads as a null).

  3. The negative control's message was previously truncated to 260 chars. A
     control only controls if the SparseCore *allocator* refused it -- an MLIR
     verifier refusal proves nothing. Full untruncated text is printed here so
     the memory space it names can be read.

Identity of the bound symbol is asserted per process, and the source md5 is
printed from inside the child, so the file that was compiled is the file on
disk rather than the file we think we passed.

`/tmp/libtpu_lockfile` serialises host AOT compiles across every agent on this
box, so each child is retried with backoff. It is NEVER removed: deleting it
would break a sibling's in-flight compile.

Run: uv_venv/bin/python3 verify_fresh_procs.py
"""

import hashlib
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
VARIANT = HERE / "sparsecore_topk.py"
PY = HERE.parents[3] / "uv_venv" / "bin" / "python3"

PRELUDE = r'''
import hashlib, importlib.util, pathlib, re, sys
import jax, jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info
_T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1
from jax.experimental import topologies
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import SingleDeviceSharding
pltpu.get_tpu_info = lambda: _T
DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)
B, N, K = 1024, 10240, 2048
'''

ARM = PRELUDE + r'''
path = pathlib.Path(sys.argv[1])
print("source_path", path)
print("source_md5", hashlib.md5(path.read_bytes()).hexdigest())
spec = importlib.util.spec_from_file_location("arm_under_test", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

# Challenge 1: prove the symbol we compile is the one this file defines, and
# that exactly one arm module is resident in this process.
fn = m._sc_topk_direct
print("bound_symbol_module", fn.__module__, "id", hex(id(fn)))
resident = [k for k, v in sys.modules.items()
            if getattr(v, "__file__", None) and
            pathlib.Path(v.__file__).name == "sparsecore_topk.py"]
print("resident_sparsecore_topk_modules", resident)
assert len(resident) == 1, resident
print("jit_cache_size_before", jax._src.pjit._cpp_pjit_cache_shard_count()
      if hasattr(jax._src.pjit, "_cpp_pjit_cache_shard_count") else "n/a")

s = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=SH)
l = jax.ShapeDtypeStruct((B,), jnp.int32, sharding=SH)
c = jax.jit(lambda a, b: m._sc_topk_direct(a, K, b),
            out_shardings=SH).lower(s, l).compile()
txt = c.as_text()
e = re.search(r"^ENTRY .*?^}", txt, re.S | re.M).group(0)
body = [ln for ln in e.splitlines()[1:-1] if ln.strip()]
print("ENTRY_instructions", len(body))
for pat in ("fusion(", "bitcast-convert(", " copy(", "copy-start(",
            "bitcast(", "custom-call"):
    print(f"ENTRY_count {pat!r}", sum(pat in ln for ln in body))
ma = c.memory_analysis()
for f in ("temp_size_in_bytes", "argument_size_in_bytes",
          "output_size_in_bytes", "generated_code_size_in_bytes"):
    print("mem", f, getattr(ma, f, None))
print("ENTRY_BEGIN")
print(e)
print("ENTRY_END")
'''

NEGCTL = PRELUDE + r'''
# Challenge 2: the control must be refused by the SparseCore ALLOCATOR, not by
# the MLIR layer. 8 MiB of VMEM scratch against a 512 KiB SC tile, with a real
# DMA to HBM so the scratch cannot be dead-code-eliminated.
def kernel(o_hbm, big, sems):
    cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0], sems.at[0])
    cp.start()
    cp.wait()

def build():
    return pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
        mesh=plsc.VectorSubcoreMesh(
            num_cores=_T.sparse_core.num_cores,
            num_subcores=_T.sparse_core.num_subcores,
            core_axis_name="core", subcore_axis_name="subcore"),
        scratch_types=(pltpu.VMEM((2 * 1024 * 1024,), jnp.int32),
                       pltpu.SemaphoreType.DMA((1,))),
        name="aot_negctl_oversized_vmem")()

try:
    jax.jit(build, out_shardings=SH).lower().compile()
    print("CONTROL_RESULT COMPILED -- CONTROL FAILED")
except Exception as exc:
    print("CONTROL_RESULT REFUSED")
    print("CONTROL_EXC_TYPE", type(exc).__name__)
    print("CONTROL_FULL_TEXT_BEGIN")
    print(str(exc))
    print("CONTROL_FULL_TEXT_END")
'''


def run(tag, src, attempts=6):
    child = HERE / f"_fresh_{tag}.py"
    child.write_text(ARM if src else NEGCTL)
    argv = [str(PY), str(child)] + ([str(src)] if src else [])
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env.pop("LIBTPU_INIT_ARGS", None)
    for i in range(1, attempts + 1):
        r = subprocess.run(argv, env=env, capture_output=True, text=True,
                           timeout=3600)
        if "lockfile" not in (r.stdout + r.stderr):
            return r
        # NEVER `sudo rm /tmp/libtpu_lockfile` -- a sibling is compiling.
        print(f"    [{tag}] libtpu lockfile contention, retry {i}", flush=True)
        time.sleep(i * 25)
    return r


if __name__ == "__main__":
    print("=" * 72)
    print("NEGATIVE CONTROL (full, untruncated)")
    print("=" * 72)
    print(run("negctl", None).stdout)

    entries = {}
    # VARIANT FIRST, deliberately -- see the module docstring.
    for tag, src in (("variant", VARIANT), ("incumbent", INCUMBENT)):
        print("=" * 72)
        print(f"FRESH SINGLE-ARM PROCESS: {tag}   (expected md5 "
              f"{hashlib.md5(pathlib.Path(src).read_bytes()).hexdigest()})")
        print("=" * 72)
        r = run(tag, src)
        if r.returncode:
            print(f"  rc={r.returncode}")
            print(r.stderr[-3000:])
            continue
        out = r.stdout
        head, _, rest = out.partition("ENTRY_BEGIN")
        print(head.strip())
        entries[tag] = rest.partition("ENTRY_END")[0].strip()
        (HERE / f"fresh-entry-{tag}.txt").write_text(entries[tag] + "\n")

    if len(entries) == 2:
        import difflib
        norm = lambda t: t.replace("_f32in", "")  # noqa: E731
        d = list(difflib.unified_diff(
            norm(entries["incumbent"]).splitlines(),
            norm(entries["variant"]).splitlines(),
            "incumbent", "variant", lineterm="", n=1))
        print("=" * 72)
        print(f"NAME-NORMALISED ENTRY DIFF  ({len(d)} lines)")
        print("=" * 72)
        print("\n".join(d) if d else "  IDENTICAL -- arms did not differ!")
        (HERE / "fresh-entry-diff.txt").write_text("\n".join(d) + "\n")
