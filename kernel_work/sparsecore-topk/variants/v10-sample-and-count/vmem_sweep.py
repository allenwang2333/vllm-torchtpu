"""Price the kernel's PRIVATE SparseCore VMEM by allocation failure.

    python vmem_sweep.py base | sc2

XLA's `*-memory-usage-report.txt` and `after_codegen.txt` are BLIND to Mosaic's
in-kernel SparseCore VMEM -- they report 0 for a kernel that provably allocates
tens of KiB, and they are byte-identical between two arms that differ in it.
So nothing here reads them.  Instead this bisects the largest `n` that still
compiles: the SparseCore allocator refuses anything past the 2097151-word spmem
bound with `E3000: CompileTimeSparseCoreAllocationFailure`, and that threshold
is a direct, behavioural readout of the kernel's private high-water.

v10's claim is an EQUALITY: it adds no scratch, so its threshold must equal the
incumbent's exactly.  Run both arms and compare.  A one-sided run proves
nothing -- the point is the difference.

One arm per process, for the reason in `aot.py`: `jax.jit` caches on the
function object, so two arms in one interpreter can compare an arm to itself.
"""

import os
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).parent
ARM = (sys.argv[1] if len(sys.argv) > 1 else "base")
os.environ["XLA_FLAGS"] = "--xla_dump_to=/dev/null"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax._src.pallas.mosaic import tpu_info as _tpu_info  # noqa: E402

_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
sys.path.insert(0, str(HERE))
import sparsecore_topk as mod  # noqa: E402

B, K = 1024, 2048
ARMS = dict(base=dict(),
            sc2=dict(sample_and_count=True, count_arity=2),
            sc16=dict(sample_and_count=True, count_arity=16))


def _topology():
    last = None
    for attempt in range(8):
        try:
            return topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                                platform="tpu").devices[0]
        except Exception as exc:  # noqa: BLE001
            if "lockfile" not in str(exc) and "ABORTED" not in str(exc):
                raise
            last = exc
            time.sleep(2 ** attempt * 0.5 + random.random())
    raise SystemExit(f"libtpu lockfile contention, gave up: {last}")


SH = SingleDeviceSharding(_topology())


def compiles(n):
    kw = ARMS[ARM]
    scores = jax.ShapeDtypeStruct((B, n), jnp.float32, sharding=SH)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=SH)
    try:
        jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l, **kw),
                out_shardings=SH).lower(scores, lens).compile()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        m = str(exc)
        tag = ("E3000/spmem" if "CompileTimeSparseCoreAllocationFailure" in m
               and "spmem" in m else "OTHER: " + m.splitlines()[0][:90])
        return False, tag


def main():
    # `n` must be a multiple of LANES; step in whole 128-word tiles so the
    # threshold is not an artefact of a partial tile.
    step = 1024
    lo, hi = step, 512 * step  # lo assumed to pass, hi assumed to fail
    ok, tag = compiles(lo)
    if not ok:
        raise SystemExit(f"{ARM}: even n={lo} fails ({tag}) -- bad bracket")
    fail_tag = ""
    while hi - lo > step:
        mid = ((lo + hi) // 2 // step) * step
        ok, tag = compiles(mid)
        print(f"  n={mid:<8} {'compiles' if ok else 'REFUSED ' + tag}",
              flush=True)
        if ok:
            lo = mid
        else:
            hi, fail_tag = mid, tag
    print(f"\n{ARM}: largest n that compiles = {lo}  "
          f"(n={hi} refused: {fail_tag})")


main()
