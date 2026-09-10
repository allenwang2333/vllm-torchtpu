"""Price v11's one `unmodelled` term: does the output store become strided?

Dumps the Mosaic module of the REAL kernel (not a toy) for base and v11 and
prints the out-DMA enqueue plus every memref type it connects, then reads the
memref tiling `tpu-infer-memref-layout` will assign FROM THE PLATFORM
(`jax._src.pallas.mosaic.tpu_info.infer_tiling`) rather than asserting it, and
turns that into a run count / run length / stride for the k-word row store.

STATIC COMPILE-TIME ANALYSIS ONLY. No TPU, no timing.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
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

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[3]
BASE_SRC = REPO / "src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py"
V11_SRC = HERE / "sparsecore_topk.py"
B, N, K = 1024, 10240, 2048


class PlDebug:
    """`pl` with `debug=True` forced on `kernel`, so Mosaic prints the module."""

    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def kernel(self, *a, **kw):
        kw["debug"] = True
        return self._r.kernel(*a, **kw)


def load(name, path):
    spec = importlib.util.spec_from_file_location(f"mdma_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.pl = PlDebug(mod.pl)
    return mod


def dump(name, mod):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l)).trace(
            jax.ShapeDtypeStruct((B, N), jnp.float32),
            jax.ShapeDtypeStruct((B, ), jnp.int32),
        ).lower(lowering_platforms=("tpu", ))
    txt = buf.getvalue()
    (HERE / "hlo" / f"mosaic-{name}.mlir").write_text(txt)
    print("=" * 74)
    print(f"{name}   ({len(txt.splitlines())} lines of Mosaic MLIR)")
    keep = []
    for ln in txt.splitlines():
        s = re.sub(r"loc\(.*", "", ln.strip())
        if ("func.func public" in s or "enqueue_dma" in s or "wait_dma" in s
                or "memref_slice" in s or "memref_squeeze" in s):
            keep.append(s)
    # The out DMA is the one whose target is an `hbm` memref.
    for s in keep:
        if "hbm" in s or "func.func public" in s:
            print("  " + s[:260])
    print(f"  memref_slice ops : {sum('memref_slice' in s for s in keep)}")
    print(f"  memref_squeeze   : {sum('memref_squeeze' in s for s in keep)}")
    print(f"  enqueue_dma ops  : {sum('enqueue_dma' in s for s in keep)}")


def tiling_arithmetic():
    """What `tpu-infer-memref-layout` assigns, read from the platform."""
    print("=" * 74)
    print("memref tiling, from jax._src.pallas.mosaic.tpu_info.infer_tiling")
    print(f"  chip generation (platform): {_TPU7X.generation}")
    i32 = jnp.dtype(jnp.int32)
    flat = jax.core.ShapedArray((B * K, ), i32)
    two_d = jax.core.ShapedArray((B, K), i32)
    t_flat = _tpu_info.infer_tiling(flat)
    t_2d = _tpu_info.infer_tiling(two_d)
    print(f"  incumbent  i32[{B * K}]      -> tiling {t_flat}")
    print(f"  v11        i32[{B},{K}]   -> tiling {t_2d}")
    itemsize = i32.itemsize
    # Incumbent: one row is k consecutive elements of a 1-D buffer tiled at
    # t_flat[0]; k % t_flat[0] == 0, so the row is whole tiles, contiguous.
    print(f"\n  incumbent row store: k={K} elems, k % {t_flat[0]} = "
          f"{K % t_flat[0]} -> {K // t_flat[0]} whole tiles, CONTIGUOUS "
          f"{K * itemsize} B in 1 run")
    sub, lane = t_2d
    runs = K // lane
    run_b = lane * itemsize
    stride_b = sub * lane * itemsize
    print(f"  v11 row store      : tile ({sub},{lane}); one row touches "
          f"{runs} tiles, {run_b} B each, stride {stride_b} B "
          f"({stride_b // itemsize} words)")
    print(f"  -> runs/row {1} -> {runs};  run length {K * itemsize} B -> "
          f"{run_b} B;  total bytes unchanged {K * itemsize} B")
    gran = 32  # SparseCore DMA granule, bytes
    print(f"  SC DMA granule {gran} B: incumbent run = "
          f"{K * itemsize // gran} granules; v11 run = {run_b // gran} "
          f"granules -> {'ALIGNED, no partial granule' if run_b % gran == 0 else 'MISALIGNED'}")
    for label, rows in (("prefill b=1024", 1024), ("decode b=64", 64)):
        print(f"  {label}: descriptors/runs across the kernel "
              f"{rows} -> {rows * runs}")


if __name__ == "__main__":
    (HERE / "hlo").mkdir(exist_ok=True)
    tiling_arithmetic()
    for nm, p in (("base", BASE_SRC), ("v11", V11_SRC)):
        try:
            dump(nm, load(nm, p))
        except Exception:
            import traceback
            traceback.print_exc()
