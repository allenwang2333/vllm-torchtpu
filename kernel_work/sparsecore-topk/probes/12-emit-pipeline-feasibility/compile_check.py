"""Local (no-TPU) compile check: does `pltpu.emit_pipeline` lower inside a
`plsc.VectorSubcoreMesh` (SparseCore) Pallas kernel?

No device is required: we force the lowering platform to "tpu" via
`jax.jit(f).trace(...).lower(lowering_platforms=("tpu",))`, which runs the full
Pallas -> Mosaic lowering (jax/_src/pallas/mosaic/sc_lowering.py) and the
Mosaic MLIR verifier without ever touching a TPU.

The only device-dependent bit is `tpu_info.get_tpu_info()`, which the
SparseCore mesh constructor calls to learn num_cores/num_subcores/num_lanes.
We pin it to the tpu7x table entry that jax itself ships
(`tpu_info.get_tpu_info_for_chip(ChipVersion.TPU_7X, 1)`); that is a pure table
lookup with no device query.

Variants:
  A  emit_pipeline in -> out, both HBM (the plain streaming shape).
  B  emit_pipeline HBM -> a *persistent* VMEM scratch (the shape sparsecore_topk
     v4 actually needs: the slice must stay resident for the later passes),
     then one manual DMA of the scratch out.
  C  the incumbent shape for reference: one synchronous make_async_copy of the
     whole slice, no pipeline. Proves the harness itself lowers.
"""

import traceback

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

# --- pin device info to tpu7x so the SC mesh can be constructed off-device ---
_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
# jax/_src/pallas/mosaic/pallas_call_registration.py:618 refuses to lower any
# SparseCore kernel off-device. That gate is about the *device*, not the
# lowering rules, so we lift it -- variant C (the incumbent, known-good on
# hardware) is included below as the control that the lift is sound.
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.experimental.pallas import tpu_sc as plsc  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

LANES = 16
N = 4096
BLOCK = 512
NBLOCKS = N // BLOCK


def _mesh():
    return plsc.VectorSubcoreMesh(
        num_cores=_TPU7X.sparse_core.num_cores,
        num_subcores=_TPU7X.sparse_core.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )


# ------------------------------- variant A ---------------------------------
def _kernel_a(x_hbm, o_hbm):

    def body(x_ref, o_ref):
        o_ref[...] = x_ref[...] + jnp.int32(1)

    pltpu.emit_pipeline(
        body,
        grid=(NBLOCKS, ),
        in_specs=pl.BlockSpec((BLOCK, ), lambda i: (i, )),
        out_specs=pl.BlockSpec((BLOCK, ), lambda i: (i, )),
    )(x_hbm, o_hbm)


def variant_a(x):
    return pl.kernel(
        _kernel_a,
        out_type=jax.ShapeDtypeStruct((N, ), jnp.int32),
        mesh=_mesh(),
        name="probe_emit_pipeline_hbm_to_hbm",
    )(x)


# ------------------------------- variant B ---------------------------------
# The v4 shape: pipeline HBM -> persistent VMEM scratch, then one DMA out.
def _kernel_b(x_hbm, o_hbm, resident_vmem, sems):

    def body(x_ref, i):
        resident_vmem[pl.ds(i * BLOCK, BLOCK)] = x_ref[...] + jnp.int32(1)

    def body_wrapper(x_ref):
        body(x_ref, pl.program_id(0))

    pltpu.emit_pipeline(
        body_wrapper,
        grid=(NBLOCKS, ),
        in_specs=pl.BlockSpec((BLOCK, ), lambda i: (i, )),
    )(x_hbm)

    cp = pltpu.make_async_copy(resident_vmem, o_hbm, sems.at[0])
    cp.start()
    cp.wait()


def variant_b(x):
    return pl.kernel(
        _kernel_b,
        out_type=jax.ShapeDtypeStruct((N, ), jnp.int32),
        mesh=_mesh(),
        scratch_types=(
            pltpu.VMEM((N, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        name="probe_emit_pipeline_into_resident_vmem",
    )(x)


# ------------------------------- variant C ---------------------------------
def _kernel_c(x_hbm, o_hbm, resident_vmem, sems):
    cp = pltpu.make_async_copy(x_hbm, resident_vmem, sems.at[0])
    cp.start()
    cp.wait()

    def cvt(i):
        resident_vmem[pl.ds(i * LANES, LANES)] = (
            resident_vmem[pl.ds(i * LANES, LANES)] + jnp.int32(1))

    plsc.parallel_loop(0, N // LANES, unroll=4)(cvt)
    cp2 = pltpu.make_async_copy(resident_vmem, o_hbm, sems.at[1])
    cp2.start()
    cp2.wait()


def variant_c(x):
    return pl.kernel(
        _kernel_c,
        out_type=jax.ShapeDtypeStruct((N, ), jnp.int32),
        mesh=_mesh(),
        scratch_types=(
            pltpu.VMEM((N, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        name="probe_incumbent_sync_dma",
    )(x)


# ------------------------------- variant D ---------------------------------
# Negative control: `sin` has a TC lowering rule but no SparseCore one
# (jax/_src/pallas/mosaic/lowering.py registers lax.sin_p for CoreType.TC only).
# If this FAILS inside sc_lowering, the harness above really is reaching the
# SparseCore lowering rules and A/B's success is meaningful.
def _kernel_d(x_hbm, o_hbm, resident_vmem, sems):
    cp = pltpu.make_async_copy(x_hbm, resident_vmem, sems.at[0])
    cp.start()
    cp.wait()

    def cvt(i):
        v = resident_vmem[pl.ds(i * LANES, LANES)].astype(jnp.float32)
        resident_vmem[pl.ds(i * LANES,
                            LANES)] = jnp.sin(v).astype(jnp.int32)

    plsc.parallel_loop(0, N // LANES, unroll=1)(cvt)
    cp2 = pltpu.make_async_copy(resident_vmem, o_hbm, sems.at[1])
    cp2.start()
    cp2.wait()


def variant_d(x):
    return pl.kernel(
        _kernel_d,
        out_type=jax.ShapeDtypeStruct((N, ), jnp.int32),
        mesh=_mesh(),
        scratch_types=(
            pltpu.VMEM((N, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        name="probe_negative_control_sin_on_sc",
    )(x)


def run(name, fn):
    print("=" * 72)
    print(f"variant {name}: {fn.__name__}")
    x = jax.ShapeDtypeStruct((N, ), jnp.int32)
    try:
        traced = jax.jit(fn).trace(x)
    except Exception:
        print(f"  TRACE FAILED for {name}")
        traceback.print_exc()
        return
    print("  trace: OK (jaxpr built)")
    try:
        lowered = traced.lower(lowering_platforms=("tpu", ))
    except Exception:
        print(f"  LOWER FAILED for {name}")
        traceback.print_exc()
        return
    txt = lowered.as_text()
    print(f"  lower: OK -> {len(txt)} chars of StableHLO/Mosaic")
    n_dma = txt.count("tpu.enqueue_dma")
    n_wait = txt.count("tpu.wait_dma")
    print(f"  tpu.enqueue_dma occurrences in module: {n_dma}")
    print(f"  tpu.wait_dma* occurrences in module:   {n_wait}")
    print(f"  mentions 'sparse_core'/'sc_':           "
          f"{txt.count('sparse_core') + txt.count('SC_')}")


if __name__ == "__main__":
    print("jax", jax.__version__, "| devices:", jax.devices())
    print("pinned tpu_info:", _TPU7X.chip_version, _TPU7X.sparse_core)
    run("C-incumbent", variant_c)
    run("A-emit_pipeline-hbm2hbm", variant_a)
    run("B-emit_pipeline-into-resident-vmem", variant_b)
    run("D-negative-control-sin", variant_d)
