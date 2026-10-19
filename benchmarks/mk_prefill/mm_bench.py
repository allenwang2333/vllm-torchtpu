"""Small-M fp8 matmul orientation bench (prefill megakernel MoE: ~160 tokens per expert).

Inside one kernel, operands resident in VMEM, `reps` back-to-back matmuls per call:
  nt:  x [M, K] @ w [K, N]          (weights as the stationary MXU operand)
  tn:  w^T [N, K] @ x^T [K, M]      (activations stationary, weights streamed) -> [N, M]
Reports us per matmul and TFLOP/s.

    python3 mm_bench.py --m 128 160 256 --k 4096 --n 2048
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[128, 160, 256])
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=64)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    fp8, f32 = jnp.float8_e4m3fn, jnp.float32
    res = []
    for m in args.m:
        for mode in ("nt", "tn"):
            k, n, reps = args.k, args.n, args.reps

            # The LHS is a loop-variant 32-row-aligned window of a taller buffer, so the matmul
            # cannot be hoisted out of the timed loop.
            if mode == "nt":
                a_rows, b_shape, o_shape = m, (k, n), (m, n)
            else:
                a_rows, b_shape, o_shape = n, (k, m), (n, m)
            a_shape = (a_rows + 256, k)
            dims = (((1,), (0,)), ((), ()))

            def kernel(a_ref, b_ref, o_ref):
                def body(i, acc):
                    a = a_ref[pl.ds(pl.multiple_of((i % 8) * 32, 32), a_rows), :]
                    return acc + jax.lax.dot_general(a, b_ref[...], dims,
                                                     preferred_element_type=f32)
                o_ref[...] = jax.lax.fori_loop(0, reps, body, jnp.zeros(o_shape, f32))

            vm = pl.BlockSpec(memory_space=pltpu.VMEM)
            call = jax.jit(pl.pallas_call(
                kernel, out_shape=jax.ShapeDtypeStruct(o_shape, f32), in_specs=[vm, vm],
                out_specs=vm, compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 << 20)))
            a = jax.random.normal(jax.random.PRNGKey(0), a_shape, f32).astype(fp8)
            b = jax.random.normal(jax.random.PRNGKey(1), b_shape, f32).astype(fp8)
            r = {"m": m, "mode": mode, "k": k, "n": n}
            try:
                jax.block_until_ready(call(a, b))
                best = 1e9
                for _ in range(5):
                    t0 = time.perf_counter()
                    jax.block_until_ready(call(a, b))
                    best = min(best, time.perf_counter() - t0)
                us = best / reps * 1e6
                r.update(us_per_mm=round(us, 2), tflops=round(2 * m * k * n / (us * 1e-6) / 1e12, 1))
            except Exception as e:  # noqa: BLE001
                r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print("[mm_bench] " + json.dumps(r), flush=True)
            res.append(r)
    with open(os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "mm_bench.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
