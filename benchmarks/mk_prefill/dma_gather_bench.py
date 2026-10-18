"""How fast can a TensorCore Pallas kernel gather rows by DMA? (prefill megakernel MoE grouping)

Gathers N rows of `row_bytes` (fp8 hidden rows: 4096 B) from x [T, W] in HBM into dst [N, W] in
HBM (or VMEM tiles) in a random order, one DMA per row, with `inflight` DMAs outstanding.
Times the slope over two repetition counts, like the other mk_prefill benches.

    python3 dma_gather_bench.py --t 8192 --n 81920 --inflight 8 16 32 64
"""

import argparse
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.environ.get("MK_CHECKOUT", os.path.abspath(os.path.join(HERE, "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=8192)
    ap.add_argument("--n", type=int, default=81920)
    ap.add_argument("--w", type=int, default=1024, help="row width in u32 words (1024 = 4 KiB)")
    ap.add_argument("--inflight", type=int, nargs="*", default=[8, 16, 32, 64])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--unroll", type=int, nargs="*", default=[1])
    ap.add_argument("--vmem-only", action="store_true")
    args = ap.parse_args()
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.path.insert(0, os.path.join(CHECKOUT, "src"))
    runpy.run_path(os.path.join(CHECKOUT, "src", "vllm_torchtpu", "env_override.py"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rng = np.random.default_rng(0)
    T, N, W = args.t, args.n, args.w
    x = jnp.asarray(rng.integers(0, 2**31, (T, W), dtype=np.int64).astype(np.uint32))
    idx = jnp.asarray(rng.integers(0, T, (N,), dtype=np.int64).astype(np.int32))
    results = []

    for to_vmem, k, un in [(tv, k, u) for tv in ((True,) if args.vmem_only else (False, True))
                           for k in args.inflight for u in args.unroll]:
        if True:
            if to_vmem:
                # Gather into a VMEM tile of R rows, flushed to HBM per tile (what a grouped matmul
                # would consume): R rows per tile, k DMAs in flight.
                R = 256

                def kernel(idx_ref, x_hbm, o_hbm, buf, sem, osem):
                    def tile(ti, c):
                        def issue(r):
                            pltpu.make_async_copy(x_hbm.at[pl.ds(idx_ref[ti * R + r], 1)],
                                                  buf.at[pl.ds(r, 1)], sem.at[r % k]).start()

                        def wait(r):
                            pltpu.make_async_copy(x_hbm.at[pl.ds(0, 1)], buf.at[pl.ds(r, 1)],
                                                  sem.at[r % k]).wait()

                        def body(r, c):
                            @pl.when(r >= k)
                            def _():
                                wait(r - k)
                            issue(r)
                            return c
                        jax.lax.fori_loop(0, R, body, 0, unroll=un)
                        jax.lax.fori_loop(R - k, R, lambda r, c: (wait(r), c)[1], 0)
                        cp = pltpu.make_async_copy(buf, o_hbm.at[pl.ds(ti * R, R)], osem.at[0])
                        cp.start()
                        cp.wait()
                        return c
                    jax.lax.fori_loop(0, N // R, tile, 0)

                scratch = [pltpu.VMEM((R, W), jnp.uint32), pltpu.SemaphoreType.DMA((k,)),
                           pltpu.SemaphoreType.DMA((1,))]
            else:
                def kernel(idx_ref, x_hbm, o_hbm, sem):
                    def issue(r):
                        pltpu.make_async_copy(x_hbm.at[pl.ds(idx_ref[r], 1)], o_hbm.at[pl.ds(r, 1)],
                                              sem.at[r % k]).start()

                    def wait(r):
                        pltpu.make_async_copy(x_hbm.at[pl.ds(0, 1)], o_hbm.at[pl.ds(r, 1)],
                                              sem.at[r % k]).wait()

                    def body(r, c):
                        @pl.when(r >= k)
                        def _():
                            wait(r - k)
                        issue(r)
                        return c
                    jax.lax.fori_loop(0, N, body, 0)
                    jax.lax.fori_loop(N - k, N, lambda r, c: (wait(r), c)[1], 0)

                scratch = [pltpu.SemaphoreType.DMA((k,))]

            call = pl.pallas_call(
                kernel,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1, grid=(),
                    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                    scratch_shapes=scratch),
                out_shape=jax.ShapeDtypeStruct((N, W), jnp.uint32),
            )
            f = jax.jit(call)
            res = {"to_vmem": to_vmem, "inflight": k, "unroll": un, "rows": N, "row_bytes": W * 4}
            try:
                out = jax.block_until_ready(f(idx, x))
                ok = bool(jnp.array_equal(out[:64], x[idx[:64]])) and bool(jnp.array_equal(out[-64:], x[idx[-64:]]))
                best = 1e9
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    jax.block_until_ready(f(idx, x))
                    best = min(best, time.perf_counter() - t0)
                res.update(ok=ok, ms=round(best * 1e3, 3), ns_per_row=round(best * 1e9 / N, 1),
                           gbps=round(N * W * 4 / best / 1e9, 1))
            except Exception as e:  # noqa: BLE001
                res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            print("[dma_bench] " + json.dumps(res), flush=True)
            results.append(res)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    with open(os.path.join(out, "dma_gather_bench.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
