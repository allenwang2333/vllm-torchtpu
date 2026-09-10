"""ONE arm per process, with the XLA dump on. STATIC COMPILE-TIME ONLY.

Two things `aot.py` cannot do from inside a single process:

  * `jax.jit` keys its cache on `(function object, avals)`, and pallas caches
    below that. A multi-arm in-process A/B can therefore compare an arm against
    itself and return a byte-identical answer with no error. `aot.py` happens to
    escape this -- every arm builds a fresh module object under its own
    `sys.modules` name and a fresh lambda, and the seven ENTRY dumps have seven
    distinct md5s and seven distinct kernel names -- but "happens to escape" is
    not evidence. This file re-derives each arm in its own interpreter, so there
    is no shared cache to escape from. Run the VARIANT first: then a residual
    caching bug shows up as base-looking-like-variant, which is visible, rather
    than variant-looking-like-base, which reads as a clean null.

  * `compiled.as_text()` is the optimized HLO and nothing else.
    `XLA_FLAGS=--xla_dump_to` additionally writes the buffer assignment, the
    memory-usage report and -- the reason this exists --
    `*sparse_core_specific_metadata.txt`, which is where SparseCore sequencer
    bundle counts and VMEM-vs-HBM placement are legible. That prices two of v4's
    open terms (the double buffer's real VMEM cost, and the per-step sequencer
    work behind `g`) for zero cluster time.

The three rungs are three separate FILES, not one file with an import-time env
switch, so this harness selects an arm by path and prints the subject's md5.
An env-selected arm can be mis-set silently by anything upstream in a job; a
file cannot, and the md5 is what a timing job gates on.

`/tmp/libtpu_lockfile` serialises host AOT compiles across every agent on this
box, so a compile can die with `ABORTED: Internal error when accessing libtpu
multi-process lockfile`. That is contention, not a result: retry with backoff.
Never delete the lockfile -- it is shared.

  uv_venv/bin/python aot_single.py <arm>     # base | v4-ragged | v4-c2048 ...
"""

import hashlib
import os
import pathlib
import random
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
BASE = HERE / "sparsecore_topk_base.py"
RAGGED = HERE / "sparsecore_topk_ragged.py"
C2048 = HERE / "sparsecore_topk_c2048.py"
N, K = 10240, 2048
OUT = HERE / "aot3"

# Variant first, deliberately: see the module docstring.
ARMS = {
    "v4-c2048": (C2048, 1024),
    "v4-ragged": (RAGGED, 1024),
    "base": (BASE, 1024),
    "v4-c2048-b64": (C2048, 64),
    "v4-ragged-b64": (RAGGED, 64),
    "base-b64": (BASE, 64),
}


def run_one(arm):
    import importlib.util

    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info as _tpu_info

    tpu7x = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
    _tpu_info.get_tpu_info = lambda: tpu7x
    _tpu_info.is_tpu_device = lambda: True
    _tpu_info.get_device_kind = lambda: "TPU7x"
    _tpu_info.get_num_device_cores = lambda: 1

    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import SingleDeviceSharding

    pltpu.get_tpu_info = lambda: tpu7x

    path, b = ARMS[arm]

    spec = importlib.util.spec_from_file_location("sctopk_only", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Identity assertion: prove which file and which constants are bound BEFORE
    # trusting anything downstream. A cached arm would still print these.
    print(f"  bound module file : {mod.__file__}")
    print(f"  subject md5       : "
          f"{hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()}")
    print(f"  CHUNK_WORDS       : {getattr(mod, 'CHUNK_WORDS', 'n/a')}")
    # These must be absent on every shipping file. A rung that still reads an
    # env var is a rung that can be mis-selected in a job.
    for dead in ("NO_PIPELINING", "RAGGED_ONLY"):
        assert not hasattr(mod, dead), f"{path} still exposes {dead}"
    print(f"  modules loaded    : "
          f"{[m for m in sys.modules if m.startswith('sctopk')]}")

    sh = SingleDeviceSharding(
        topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                     platform="tpu").devices[0])
    scores = jax.ShapeDtypeStruct((b, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((b, ), jnp.int32, sharding=sh)
    fn = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l), out_shardings=sh)
    hlo = fn.lower(scores, lens).compile().as_text()

    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    ent = m.group(0) if m else hlo
    (OUT / arm / "entry.txt").write_text(ent)
    names = sorted(set(re.findall(r"sc_topk_direct_b\d+_n\d+_k\d+\w*", hlo)))
    print(f"  kernel name in HLO: {names}")
    print(f"  ENTRY instrs      : {len(ent.splitlines()) - 2}")
    print(f"ARM_OK {arm}")


def spawn(arm):
    d = OUT / arm
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") +
                        f" --xla_dump_to={d}").strip()
    env["SCTOPK_AOT_CHILD"] = arm
    for attempt in range(6):
        p = subprocess.run([sys.executable, __file__, arm], env=env,
                           capture_output=True, text=True)
        blob = p.stdout + p.stderr
        if "lockfile" in blob and "ABORTED" in blob:
            wait = 2 ** attempt + random.random() * 2
            print(f"  [{arm}] libtpu lockfile contention, retry in "
                  f"{wait:.1f}s (attempt {attempt + 1}/6)", flush=True)
            time.sleep(wait)
            continue
        return p
    return p


if __name__ == "__main__":
    if os.environ.get("SCTOPK_AOT_CHILD"):
        run_one(sys.argv[1])
        sys.exit(0)
    OUT.mkdir(exist_ok=True)
    for arm in (sys.argv[1:] or list(ARMS)):
        print(f"\n=== {arm}  (fresh process)", flush=True)
        p = spawn(arm)
        sys.stdout.write(p.stdout)
        if f"ARM_OK {arm}" not in p.stdout:
            sys.stdout.write(p.stderr[-2000:])
            print(f"  {arm}: FAILED")
