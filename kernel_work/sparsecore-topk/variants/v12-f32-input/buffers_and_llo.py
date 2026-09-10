"""Buffer assignment for both arms, plus an attempt at an offline LLO dump.

STATIC COMPILE-TIME ANALYSIS ONLY.

`aot_check.py` reads `memory_analysis()`, which is a rollup. This reads the
allocation table XLA writes to `-buffer-assignment.txt`, which is the artifact
`probes/14` Q1 used to prove the 40 MiB `bitcast_convert_type` buffer is real.
The question here is the mirror image: is it GONE in v12, or renamed / turned
into a copy / re-staged somewhere else?

It also asks `--xla_mosaic_dump_to` / `--xla_jf_dump_to` for a compiler dump on
a host with no TPU, so the two in-body `tpu.bitcast`s can be priced against
emitted code rather than assumed free.

Run: uv_venv/bin/python3 buffers_and_llo.py
"""

import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
VARIANT = HERE / "sparsecore_topk.py"
PY = HERE.parents[3] / "uv_venv" / "bin" / "python3"

CHILD = r'''
import os, pathlib, sys
# LIBTPU_INIT_ARGS is read by libtpu once, at load time -- so it must be set
# before `import jax`. These are NOT XLA_FLAGS; XLA rejects them by name.
# (xprof-llo SKILL.md:109-111)
_d = sys.argv[2]
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={_d} --xla_jf_dump_llo_text=true "
    f"--xla_mosaic_dump_to={_d}")
import jax, jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info
_T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import SingleDeviceSharding
pltpu.get_tpu_info = lambda: _T
import importlib.util
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("m", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
sh = SingleDeviceSharding(dev)
B, N, K = 1024, 10240, 2048
s = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
l = jax.ShapeDtypeStruct((B,), jnp.int32, sharding=sh)
jax.jit(lambda a, b: m._sc_topk_direct(a, K, b),
        out_shardings=sh).lower(s, l).compile()
print("compiled ok")
'''


def run(tag, src, flags):
    out = HERE / f"xdump-{tag}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_dump_to={out} --xla_dump_hlo_as_text"
    child = HERE / "_child.py"
    child.write_text(CHILD)
    r = subprocess.run([str(PY), str(child), str(src), str(out)],
                       env=env,
                       capture_output=True,
                       text=True,
                       timeout=3600)
    print(f"--- {tag}: rc={r.returncode} {r.stdout.strip().splitlines()[-1:]}")
    if r.returncode:
        print(r.stderr[-1500:])
    return out


def report(tag, out):
    ba = sorted(out.glob("*buffer-assignment.txt"))
    print(f"\n===== {tag} buffer assignment ({len(ba)} file(s)) =====")
    for f in ba:
        txt = f.read_text()
        allocs = re.findall(r"^allocation (\d+): size (\d+), (.*)$", txt, re.M)
        total = sum(int(s) for _, s, _ in allocs)
        temp = [(int(i), int(s)) for i, s, k in allocs if "temp" in k]
        big = [(int(i), int(s), k) for i, s, k in allocs if int(s) >= 4 << 20]
        print(f"  {f.name}")
        print(f"    allocations           : {len(allocs)}")
        print(f"    sum(allocation sizes) : {total}")
        print(f"    preallocated-temp     : {temp}  peak={max([s for _, s in temp] or [0])}")
        for i, s, k in big:
            print(f"    >=4MiB alloc {i}: {s}  {k[:60]}")
        for pat in ("bitcast_convert_type", "bitcast-convert", "copy",
                    "sparse-core-data-format"):
            hits = [ln.strip() for ln in txt.splitlines()
                    if pat in ln and "size=" in ln]
            print(f"    values matching {pat!r}: {len(hits)}")
            for h in hits[:4]:
                print(f"        {h[:150]}")
    others = {}
    for f in out.rglob("*"):
        if f.is_file():
            others[f.suffix or f.name.split(".")[-1]] = others.get(
                f.suffix or "?", 0) + 1
    print(f"  dump file kinds: {dict(sorted(others.items()))}")
    for kw in ("llo", "mosaic", "bundle", "sc_topk"):
        hits = [p.name for p in out.rglob(f"*{kw}*")]
        print(f"  files matching *{kw}*: {len(hits)} {hits[:6]}")


if __name__ == "__main__":
    for tag, src in (("incumbent", INCUMBENT), ("variant", VARIANT)):
        report(tag, run(tag, src, ""))
