"""Measure the achievable MXU rate on this TPU for bf16 and fp8, and dump the
hardware table JAX itself reports.

Answers, by measurement rather than by citation: does tpu7x have a 2x fp8 MXU
rate over bf16, and does a mixed bf16 x fp8 dot get it?

Run on one chip; all work is placed on jax.devices()[0], i.e. ONE TensorCore,
so every TFLOP/s below is per core.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _bench_util import banner, bench, env_report  # noqa: E402

M = K = N = 8192


def dump_tpu_info():
    banner("jax.experimental.pallas.tpu.get_tpu_info() -- JAX's hardware table")
    from jax.experimental.pallas import tpu as pltpu
    info = pltpu.get_tpu_info()
    print(repr(info), flush=True)
    ncore = info.num_cores
    print(f"\nnum_cores (this logical device) = {ncore}", flush=True)
    for field in ("bf16_ops_per_second", "fp8_ops_per_second",
                  "int8_ops_per_second", "int4_ops_per_second",
                  "mem_bw_bytes_per_second", "hbm_capacity_bytes",
                  "vmem_capacity_bytes", "mxu_column_size", "num_mxus",
                  "num_accumulators", "generation", "chip_version"):
        try:
            v = getattr(info, field)
        except Exception as e:  # noqa: BLE001
            print(f"  {field}: <{type(e).__name__}>", flush=True)
            continue
        if isinstance(v, int) and "ops_per_second" in field:
            print(f"  {field}: {v:.4g}  = {v/1e12:.2f} TFLOP/s per core",
                  flush=True)
        elif field == "mem_bw_bytes_per_second":
            print(f"  {field}: {v:.4g}  = {v/1e12:.2f} TB/s per core",
                  flush=True)
        else:
            print(f"  {field}: {v}", flush=True)
    for lhs, rhs in [(jnp.bfloat16, jnp.bfloat16),
                     (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
                     (jnp.bfloat16, jnp.float8_e4m3fn),
                     (jnp.float8_e4m3fn, jnp.bfloat16),
                     (jnp.float8_e5m2, jnp.float8_e5m2),
                     (jnp.int8, jnp.int8)]:
        try:
            ok = info.is_matmul_supported(lhs, rhs)
        except Exception as e:  # noqa: BLE001
            ok = f"<{type(e).__name__}>"
        print(f"  is_matmul_supported({lhs.__name__}, {rhs.__name__}) = {ok}",
              flush=True)
    return info


def make(dtype, shape, dev, key):
    x = jax.random.normal(key, shape, jnp.float32) * 0.1
    if dtype == jnp.int8:
        x = (x * 100).astype(jnp.int8)
    else:
        x = x.astype(dtype)
    return jax.device_put(x, dev)


def main():
    dev = env_report()
    info = dump_tpu_info()

    banner(f"DENSE MATMUL PEAK  m=k=n={M}  (one TensorCore)")
    flop = 2.0 * M * K * N
    print(f"per call: {flop/1e9:.1f} GFLOP", flush=True)

    cases = [
        ("bf16 x bf16 -> f32", jnp.bfloat16, jnp.bfloat16, jnp.float32),
        ("bf16 x bf16 -> bf16", jnp.bfloat16, jnp.bfloat16, jnp.bfloat16),
        ("f8e4m3 x f8e4m3 -> f32", jnp.float8_e4m3fn, jnp.float8_e4m3fn,
         jnp.float32),
        ("f8e4m3 x f8e4m3 -> bf16", jnp.float8_e4m3fn, jnp.float8_e4m3fn,
         jnp.bfloat16),
        ("bf16 x f8e4m3 -> f32", jnp.bfloat16, jnp.float8_e4m3fn, jnp.float32),
        ("f8e5m2 x f8e5m2 -> f32", jnp.float8_e5m2, jnp.float8_e5m2,
         jnp.float32),
        ("f32 x f32 -> f32", jnp.float32, jnp.float32, jnp.float32),
    ]
    k0, k1 = jax.random.split(jax.random.key(0))
    results = {}
    for name, ld, rd, od in cases:
        try:
            a = make(ld, (M, K), dev, k0)
            b = make(rd, (K, N), dev, k1)
        except Exception as e:  # noqa: BLE001
            print(f"RESULT {name:<38s} SETUP FAILED: {e}", flush=True)
            continue

        def f(a, b, od=od):
            return jax.lax.dot_general(a, b, (((1, ), (0, )), ((), ())),
                                       preferred_element_type=od)

        us = bench(name, f, (a, b), dev, iters=30, reps=3)
        if us:
            tf = flop / (us * 1e-6) / 1e12
            results[name] = tf
            print(f"       -> {tf:8.1f} TFLOP/s per core", flush=True)
        del a, b

    banner("VERDICT")
    bf = results.get("bf16 x bf16 -> f32")
    f8 = results.get("f8e4m3 x f8e4m3 -> f32")
    mixed = results.get("bf16 x f8e4m3 -> f32")
    if bf and f8:
        print(f"measured fp8 / bf16 ratio = {f8/bf:.3f}", flush=True)
    if bf and mixed:
        print(f"measured mixed(bf16xfp8) / bf16 ratio = {mixed/bf:.3f}",
              flush=True)
    print(f"jax table bf16 = {info.bf16_ops_per_second/1e12:.2f} TF/s/core, "
          f"fp8 = {info.fp8_ops_per_second/1e12:.2f} TF/s/core, "
          f"ratio = {info.fp8_ops_per_second/max(info.bf16_ops_per_second,1):.3f}",
          flush=True)

    banner("HBM BANDWIDTH (for the sort roofline)")
    n = 1 << 26  # 64 Mi float32 = 268 MB
    x = jax.device_put(jnp.ones((n, ), jnp.float32), dev)
    us = bench("copy 268MB f32 (read+write)", lambda a: a * 2.0, (x, ), dev,
               iters=30)
    if us:
        gbs = 2 * n * 4 / (us * 1e-6) / 1e9
        print(f"       -> {gbs:8.1f} GB/s per core (r+w)", flush=True)
    print(f"jax table mem_bw = "
          f"{info.mem_bw_bytes_per_second/1e9:.1f} GB/s per core", flush=True)
    _ = np


if __name__ == "__main__":
    main()
