"""Two extra controls, both STATIC COMPILE-TIME.

1. AOT negative control. `hlo_diff.py` claims its greens mean "the SparseCore
   Mosaic backend accepted this". That is only true if `.compile()` against the
   tpu7x topology actually runs Mosaic. So compile a kernel that passes the
   Mosaic *verifier* but must fail its *allocator*: 8 MiB of VMEM scratch on a
   512 KiB SC tile. If this compiles, every green above is worthless.

2. Is `ref.bitcast` usable at all on SC? It is not on a 1-D ref
   (jax/_src/state/utils.py:98 indexes shape[-2]); check 2-D.
"""
import contextlib  # noqa: F401
import io  # noqa: F401
import re  # noqa: F401

exec(  # reuse dma_shape.py's pinned prologue verbatim
    open(__file__.replace("aot_controls.py", "dma_shape.py")).read().split(
        "for two_d in")[0].replace("debug=True", "debug=False"))

from jax.experimental import topologies  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)
SLICE, ROWS = 10240, 10240 // LANES


def _mesh():
    return plsc.VectorSubcoreMesh(
        num_cores=_TPU7X.sparse_core.num_cores,
        num_subcores=_TPU7X.sparse_core.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore")


def oversized():

    def kernel(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0],
                                   sems.at[0])
        cp.start()
        cp.wait()

    return pl.kernel(kernel,
                     out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
                     mesh=_mesh(),
                     scratch_types=(pltpu.VMEM((2 * 1024 * 1024, ), jnp.int32),
                                    pltpu.SemaphoreType.DMA((1, ))),
                     name="aot_negctl_oversized_vmem")()


def refbitcast_2d(x):

    def kernel(x_hbm, o_hbm, resident_f32, rowbuf, sems):
        r = resident_f32.bitcast(jnp.int32)
        cp = pltpu.make_async_copy(x_hbm.at[pl.ds(0, ROWS)], resident_f32,
                                   sems.at[0])
        cp.start()
        cp.wait()

        def cvt(i):
            v = r[i]
            m = jnp.bitwise_and(jnp.right_shift(v, 31), jnp.int32(0x7FFFFFFF))
            r[i] = jnp.bitwise_xor(v, m)

        plsc.parallel_loop(0, ROWS, unroll=4)(cvt)
        cp = pltpu.make_async_copy(rowbuf.at[pl.ds(0, K)], o_hbm.at[0],
                                   sems.at[1])
        cp.start()
        cp.wait()

    return pl.kernel(kernel,
                     out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
                     mesh=_mesh(),
                     scratch_types=(pltpu.VMEM((ROWS, LANES), jnp.float32),
                                    pltpu.VMEM((K + LANES, ), jnp.int32),
                                    pltpu.SemaphoreType.DMA((2, ))),
                     name="refbitcast2d")(x)


if __name__ == "__main__":
    for tag, fn, args in (
        ("V11-2Dout-AOT", lambda: build(True), ()),
        ("AOT-NEGCTL-8MiB-vmem", oversized, ()),
        ("refbitcast-2D-f32-scratch", refbitcast_2d,
         (jax.ShapeDtypeStruct((B * ROWS, LANES), jnp.float32), )),
    ):
        try:
            n = len(jax.jit(fn, out_shardings=SH).lower(*args).compile()
                    .as_text())
            print(f"{tag}: AOT COMPILED ok ({n} chars)")
        except Exception as e:
            print(f"{tag}: FAILED -> {type(e).__name__}: {str(e)[:400]}")
