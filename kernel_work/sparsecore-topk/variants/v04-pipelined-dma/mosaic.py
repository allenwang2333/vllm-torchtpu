"""What did the score read become? Dump the Mosaic module for the incumbent
and for v4 at three chunk sizes, and count the ops the change is supposed to
move: DMA enqueues, DMA waits, and the VMEM allocations that hold the row.

STATIC COMPILE-TIME ANALYSIS ONLY -- no timing is produced.
"""

import collections
import contextlib
import importlib.util
import io
import os
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

from jax.experimental.pallas import tpu as pltpu  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
V4 = HERE / "sparsecore_topk.py"
N, K = 10240, 2048
OUT = HERE / "aot"
OUT.mkdir(exist_ok=True)

DEBUG_PATCH = ("        mesh=mesh,\n", "        mesh=mesh,\n        debug=True,\n")


def load(tag, path, env):
    for key, val in env.items():
        os.environ[key] = val
    text = path.read_text()
    if DEBUG_PATCH[0] not in text:
        raise SystemExit(f"debug anchor missing in {path}")
    text = text.replace(*DEBUG_PATCH)
    gen = OUT / f"_dbg_{tag}.py"
    gen.write_text(text)
    spec = importlib.util.spec_from_file_location(f"dbg_{tag}", gen)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def dump(tag, path, env, b):
    mod = load(tag, path, env)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        jax.eval_shape(
            lambda s, l: mod._sc_topk_direct(s, K, l),
            jax.ShapeDtypeStruct((b, N), jnp.float32),
            jax.ShapeDtypeStruct((b, ), jnp.int32))
    txt = buf.getvalue()
    (OUT / f"mosaic-{tag}-b{b}.txt").write_text(txt)

    ops = collections.Counter(re.findall(r"\"?(tpu\.[a-z_0-9]+)\"?[ (]", txt))
    # every VMEM buffer the kernel body is handed or allocates
    vmem = collections.Counter(re.findall(r"memref<(\d+)xi32>", txt))
    words = sum(int(s) * c for s, c in vmem.items())
    print(f"\n=== {tag}  b={b}   ({len(txt.splitlines())} MLIR lines)")
    print(f"  enqueue_dma        {ops.get('tpu.enqueue_dma', 0)}")
    print(f"  wait_dma           {ops.get('tpu.wait_dma', 0) + ops.get('tpu.wait_dma2', 0)}")
    for op in sorted(ops):
        if any(t in op for t in ("dma", "sem", "stream", "pipeline")):
            print(f"    {op:34s} {ops[op]}")
    print(f"  distinct i32 VMEM memref extents: "
          f"{sorted((int(s), c) for s, c in vmem.items())}")
    print(f"  sum of i32 VMEM memref words    : {words} "
          f"({words * 4} B)")
    return txt


if __name__ == "__main__":
    for tag, path, env, b in [
        ("base", INCUMBENT, {}, 1024),
        ("v4-c512", V4, {"SCTOPK_V4_CHUNK": "512", "SCTOPK_V4_NOPIPE": "0"}, 1024),
        ("v4-c1024", V4, {"SCTOPK_V4_CHUNK": "1024", "SCTOPK_V4_NOPIPE": "0"}, 1024),
        ("v4-c2048", V4, {"SCTOPK_V4_CHUNK": "2048", "SCTOPK_V4_NOPIPE": "0"}, 1024),
        ("v4-c1024-nopipe", V4,
         {"SCTOPK_V4_CHUNK": "1024", "SCTOPK_V4_NOPIPE": "1"}, 1024),
        ("v4-c1024", V4, {"SCTOPK_V4_CHUNK": "1024", "SCTOPK_V4_NOPIPE": "0"}, 64),
    ]:
        try:
            dump(tag, path, env, b)
        except Exception as exc:  # noqa: BLE001
            print(f"=== {tag} b={b}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:600]}")
