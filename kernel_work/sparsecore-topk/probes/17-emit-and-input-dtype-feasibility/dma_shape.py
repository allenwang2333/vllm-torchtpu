"""What does the v11 out DMA actually become? Dump the Mosaic module for the
flat (incumbent) and 2-D (v11) output forms of one toy SC kernel and print the
`tpu.enqueue_dma` on the output ref plus the memref types it connects.

STATIC COMPILE-TIME ANALYSIS ONLY.
"""
import contextlib
import io
import re

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
from jax.experimental.pallas import tpu_sc as plsc  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

LANES = 16
B, K = 1024, 2048


def build(two_d):

    def kernel(o_hbm, rowbuf, sems):
        row = jax.lax.axis_index("subcore")
        dst = o_hbm.at[row] if two_d else o_hbm.at[pl.ds(row * K, K)]
        cp = pltpu.make_async_copy(rowbuf.at[pl.ds(0, K)], dst, sems.at[0])
        cp.start()
        cp.wait()

    out_type = (jax.ShapeDtypeStruct((B, K), jnp.int32) if two_d else
                jax.ShapeDtypeStruct((B * K, ), jnp.int32))
    return pl.kernel(
        kernel,
        out_type=out_type,
        mesh=plsc.VectorSubcoreMesh(
            num_cores=_TPU7X.sparse_core.num_cores,
            num_subcores=_TPU7X.sparse_core.num_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore",
        ),
        scratch_types=(pltpu.VMEM((K + LANES, ), jnp.int32),
                       pltpu.SemaphoreType.DMA((1, ))),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        debug=True,
        name=f"dma_shape_{'2d' if two_d else 'flat'}",
    )()


for two_d in (False, True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.jit(lambda: build(two_d)).trace().lower(
            lowering_platforms=("tpu", ))
    txt = buf.getvalue()
    print("=" * 72)
    print("2-D out" if two_d else "flat out (incumbent)")
    for ln in txt.splitlines():
        s = ln.strip()
        if ("enqueue_dma" in s or "wait_dma" in s or "func.func public" in s
                or re.match(r"%\w+ = memref\.", s)):
            print("  " + re.sub(r"loc\(.*", "", s)[:300])
