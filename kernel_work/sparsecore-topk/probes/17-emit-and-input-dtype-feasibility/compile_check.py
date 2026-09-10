"""Local (no-TPU) compile check for U2 (f32 kernel input) and v11 (2-D out).

Extends probes/12/compile_check.py verbatim in its device-pinning prologue: we
force lowering to "tpu" so the full Pallas -> Mosaic SparseCore lowering and the
Mosaic verifier run without a device.

Everything here is STATIC COMPILE-TIME analysis. A lowering either succeeds or
raises; no timing is produced or implied.

Variants (production dims b=1024, n=10240, k=2048):
  C0  incumbent: i32 HBM in -> i32 VMEM scratch, monotone_key in place.  CONTROL
  U2a f32 HBM in -> i32 VMEM scratch (does the sync DMA cross dtypes?)
  U2b f32 HBM in -> f32 VMEM scratch, bitcast to i32 in-register, key, bitcast
      back, store  (does the *scratch* dtype have to change?)
  U2c f32 HBM in, ref-level `.bitcast(jnp.int32)` on the HBM ref -> i32 scratch
  V11a 2-D out_type (b, k) i32, out DMA writes one row via out_hbm.at[row]
  V11b 2-D out_type (b, k) i32, out DMA writes out_hbm.at[pl.ds(row, 1)]
  NEG  jnp.sin on SC -- probe 12's negative control, must still FAIL.
"""

import traceback

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

# --- pin device info to tpu7x (probes/12/compile_check.py:30-39) -------------
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
MONO_MASK = 0x7FFFFFFF

# production shape: scores f32[1024, 10240], out i32[1024, 2048]
B, N, K = 1024, 10240, 2048
SLICE = N  # slice_len; already a multiple of LANES


def _mesh():
    return plsc.VectorSubcoreMesh(
        num_cores=_TPU7X.sparse_core.num_cores,
        num_subcores=_TPU7X.sparse_core.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )


def monotone_key(bits):
    # sparsecore_topk.py:116-118, verbatim.
    m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
    return jnp.bitwise_xor(bits, m)


def _dma(src, dst, sem):
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()


# --------------------------- input-dtype variants ---------------------------
def _mk_in_variant(name, in_dtype, scratch_dtype, ref_bitcast, out_2d=False):
    """One kernel per (input dtype, scratch dtype, ref-bitcast) combination."""

    def kernel(x_hbm, o_hbm, resident, rowbuf, sems):
        src = x_hbm.bitcast(jnp.int32) if ref_bitcast else x_hbm
        _dma(src.at[pl.ds(0, SLICE)], resident, sems.at[0])

        def cvt(i):
            v = resident[pl.ds(i * LANES, LANES)]
            if scratch_dtype == jnp.float32:
                bits = jax.lax.bitcast_convert_type(v, jnp.int32)
                resident[pl.ds(i * LANES, LANES)] = (
                    jax.lax.bitcast_convert_type(monotone_key(bits),
                                                 jnp.float32))
            else:
                resident[pl.ds(i * LANES, LANES)] = monotone_key(v)

        plsc.parallel_loop(0, SLICE // LANES, unroll=4)(cvt)
        dst = o_hbm.at[0] if out_2d else o_hbm.at[pl.ds(0, K)]
        _dma(rowbuf.at[pl.ds(0, K)], dst, sems.at[1])

    out_type = (jax.ShapeDtypeStruct((B, K), jnp.int32) if out_2d else
                jax.ShapeDtypeStruct((B * K, ), jnp.int32))

    def build(x):
        return pl.kernel(
            kernel,
            out_type=out_type,
            mesh=_mesh(),
            scratch_types=(
                pltpu.VMEM((SLICE, ), scratch_dtype),
                pltpu.VMEM((K + LANES, ), jnp.int32),
                pltpu.SemaphoreType.DMA((2, )),
            ),
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                                 needs_layout_passes=False),
            name=name,
        )(x)

    build.__name__ = name
    build.in_dtype = in_dtype
    return build


C0 = _mk_in_variant("c0_incumbent_i32in_i32scratch", jnp.int32, jnp.int32,
                    False)
U2A = _mk_in_variant("u2a_f32in_i32scratch", jnp.float32, jnp.int32, False)
U2B = _mk_in_variant("u2b_f32in_f32scratch_inreg_bitcast", jnp.float32,
                     jnp.float32, False)
U2C = _mk_in_variant("u2c_f32in_refbitcast_i32scratch", jnp.float32,
                     jnp.int32, True)


# ------------------------------ v11 variants --------------------------------
def _mk_out_variant(name, indexer):

    def kernel(x_hbm, o_hbm, resident, rowbuf, sems):
        _dma(x_hbm.at[pl.ds(0, SLICE)], resident, sems.at[0])
        row = jax.lax.axis_index("subcore")
        _dma(rowbuf.at[pl.ds(0, K)], indexer(o_hbm, row), sems.at[1])

    def build(x):
        return pl.kernel(
            kernel,
            out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
            mesh=_mesh(),
            scratch_types=(
                pltpu.VMEM((SLICE, ), jnp.int32),
                pltpu.VMEM((K + LANES, ), jnp.int32),
                pltpu.SemaphoreType.DMA((2, )),
            ),
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                                 needs_layout_passes=False),
            name=name,
        )(x)

    build.__name__ = name
    build.in_dtype = jnp.int32
    return build


V11A = _mk_out_variant("v11a_2d_out_row_index", lambda o, r: o.at[r])
V11B = _mk_out_variant("v11b_2d_out_ds_row", lambda o, r: o.at[pl.ds(r, 1)])
V11C = _mk_out_variant(
    "v11c_2d_out_row_and_col_ds",
    lambda o, r: o.at[r, pl.ds(0, K)],
)


# --------------------- negative control (probes/12:148-179) -----------------
def NEG(x):

    def kernel(x_hbm, o_hbm, resident, sems):
        _dma(x_hbm.at[pl.ds(0, SLICE)], resident, sems.at[0])

        def cvt(i):
            v = resident[pl.ds(i * LANES, LANES)].astype(jnp.float32)
            resident[pl.ds(i * LANES, LANES)] = jnp.sin(v).astype(jnp.int32)

        plsc.parallel_loop(0, SLICE // LANES, unroll=1)(cvt)
        _dma(resident.at[pl.ds(0, K)], o_hbm.at[pl.ds(0, K)], sems.at[1])

    return pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((B * K, ), jnp.int32),
        mesh=_mesh(),
        scratch_types=(
            pltpu.VMEM((SLICE, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        name="neg_control_sin_on_sc",
    )(x)


NEG.in_dtype = jnp.int32


def run(tag, fn):
    print("=" * 72)
    print(f"{tag}: {getattr(fn, '__name__', fn)}")
    x = jax.ShapeDtypeStruct((B * N, ), fn.in_dtype)
    try:
        traced = jax.jit(fn).trace(x)
    except Exception:
        print(f"  TRACE FAILED [{tag}]")
        traceback.print_exc()
        return False
    print("  trace: OK")
    try:
        lowered = traced.lower(lowering_platforms=("tpu", ))
        txt = lowered.as_text()
    except Exception:
        print(f"  LOWER FAILED [{tag}]")
        traceback.print_exc()
        return False
    print(f"  lower: OK -> {len(txt)} chars")
    print(f"  tpu.enqueue_dma: {txt.count('tpu.enqueue_dma')}  "
          f"tpu.bitcast: {txt.count('tpu.bitcast')}  "
          f"memref_bitcast: {txt.count('memref_bitcast')}")
    try:
        compiled = traced.lower().compile()
        hlo = compiled.as_text()
        print(f"  COMPILE: OK ({len(hlo)} chars of optimized HLO)")
    except Exception as e:
        print(f"  compile: skipped/failed ({type(e).__name__}: "
              f"{str(e)[:120]})")
    return True


if __name__ == "__main__":
    print("jax", jax.__version__, "| devices:", jax.devices())
    ok = {}
    for tag, fn in [
        ("C0-control-incumbent", C0),
        ("U2a-f32in-i32scratch", U2A),
        ("U2b-f32in-f32scratch", U2B),
        ("U2c-f32in-refbitcast", U2C),
        ("V11a-2Dout-row", V11A),
        ("V11b-2Dout-ds", V11B),
        ("V11c-2Dout-row-col", V11C),
        ("NEG-sin", NEG),
    ]:
        ok[tag] = run(tag, fn)
    print("=" * 72)
    for tag, v in ok.items():
        print(f"  {tag:28s} {'LOWERS' if v else 'FAILS'}")
    print("negative control must read FAILS:", not ok["NEG-sin"])
