"""Non-compiling model of the composed candidate's VMEM guard vs `n`.

The allocation-failure sweep in `vmemctl.py` bisects on `n`, which ASSUMES the
footprint is monotone in `n`. `_pick_chunk` breaks that assumption: it picks
the largest divisor of `n` that is <= 2048, so the pipeline buffer shrinks
discontinuously as `n` moves. This prints the model so the non-monotonicity is
visible rather than aliased by a bisection.
"""
import importlib.util, pathlib, sys
from jax._src.pallas.mosaic import tpu_info as _tpu_info
T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1
from jax.experimental.pallas import tpu as pltpu
pltpu.get_tpu_info = lambda: T
HERE = pathlib.Path(__file__).parent
s = importlib.util.spec_from_file_location("gm", HERE / "sparsecore_topk.py")
m = importlib.util.module_from_spec(s); sys.modules["gm"] = m; s.loader.exec_module(m)
sc = T.sparse_core
CAP = sc.vmem_capacity_bytes
print(f"  cap={CAP} B   b=16 (b_pad*4 = 64 B)")
print(f"  {'n':>8} {'chunk':>6} {'pipe B':>8} {'total B':>9} {'guard':>7}")
prev = None
flips = 0
for n in range(124000, 125400, 16):
    c = m._pick_chunk(n, m.CHUNK_WORDS, 16)
    total = n * 4 + 2 * c * 4 + 2 * (256 + 16) * 4 + (2048 + 16) * 4 + 64
    g = "REFUSE" if total > CAP else "pass"
    if prev is not None and g != prev:
        flips += 1
        print(f"  {n:>8} {c:>6} {2*c*4:>8} {total:>9} {g:>7}   <-- flip {flips}")
    prev = g
print(f"  guard verdict flips {flips} times over n in [124000, 125400) -- "
      f"a bisection on `n` is INVALID here")
for n in (10240, 65536):
    c = m._pick_chunk(n, m.CHUNK_WORDS, 16)
    for b_pad, tag in ((1024, "b=1024"), (16, "b=16")):
        total = n*4 + 2*c*4 + 2*272*4 + 2064*4 + b_pad*4
        print(f"  PRODUCTION-REACHABLE n={n:6d} chunk={c:5d} {tag}: "
              f"{total} B = {total/1024:.2f} KiB of {CAP/1024:.0f} KiB "
              f"({100*total/CAP:.1f}%)")
