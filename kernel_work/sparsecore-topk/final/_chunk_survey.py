"""What chunk does `_pick_chunk` resolve to at every REACHABLE per-slice n?

`_pick_chunk` returns the largest divisor of n that is <= CHUNK_WORDS, so an
awkward slice length can collapse the pipeline block to a handful of words. The
question is not whether that is possible in the abstract, it is whether any
shape `_pick_partition` can produce reaches it.
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
sp = importlib.util.spec_from_file_location("cs", HERE / "sparsecore_topk.py")
m = importlib.util.module_from_spec(sp); sys.modules["cs"] = m; sp.loader.exec_module(m)
LANES = 16
rows = []
worst = []
for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
    for pages in range(1, 1041):
        n = pages * 128
        for k in (256, 512, 1024, 2048):
            if k > n or n % LANES or k % LANES:
                continue
            try:
                p = m._pick_partition(b, n, k)
            except Exception:
                continue
            sl = m._align_to(-(-n // p), LANES) if p > 1 else n
            c = m._pick_chunk(sl, m.CHUNK_WORDS, LANES)
            rows.append((c, sl, b, n, k, p))
rows.sort()
print(f"  reachable (b,n,k) combinations surveyed: {len(rows)}")
print(f"  chunk_words min = {rows[0][0]}, max = {rows[-1][0]}")
import collections
h = collections.Counter(r[0] for r in rows)
print(f"  chunk distribution (top 12): {h.most_common(12)}")
print("  10 smallest chunks:")
for c, sl, b, n, k, p in rows[:10]:
    print(f"    chunk={c:5d} slice_len={sl:7d} b={b:5d} n={n:7d} k={k:5d} P={p}")
small = [r for r in rows if r[0] < 128]
print(f"  shapes with chunk < 128: {len(small)}")
