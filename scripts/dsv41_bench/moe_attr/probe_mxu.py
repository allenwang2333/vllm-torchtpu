#!/usr/bin/env python3
"""Single-TensorCore probes for the fused-EP MoE's per-expert matmul body.

`--probe native` : does the matrix unit take float4_e2m1fn operands as they
                   are, in XLA (jnp.dot) and in Pallas (lax.dot_general on a
                   bitcast fp4 value, no widening)? Prints compile/run outcome
                   and the error against an f32 reference.
`--probe body`   : the gate/up contraction exactly as the kernel does it --
                   [M, 5120] fp8 rows x [5120, N] weight, in 512-row blocks,
                   each block's f32 partial scaled and accumulated -- with the
                   weight VMEM-resident, so HBM is out of the picture. Weight
                   as packed fp4 widened in-kernel ('fp4'), packed fp4 fed to
                   the dot unwidened ('fp4n'), or stored fp8 ('fp8'). Time per
                   pass = slope over the in-kernel pass count, so launch and
                   the initial weight load cancel.
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
DN = (((1,), (0,)), ((), ()))
QB = 512
PACK = 8  # fp4 values per u32 word


def pipelined_ms(fn, args, n=20, bursts=3):
    jax.block_until_ready(fn(*args))
    out = []
    for _ in range(bursts):
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn(*args)
        jax.block_until_ready(r)
        out.append((time.perf_counter() - t0) * 1e3 / n)
    return statistics.median(out)


def random_fp4(key, shape):
    return (jax.random.normal(key, shape) * 2.0).astype(FP4)


def packed_u32(w4):
    """fp4 [K, N] -> u32 [K/8, N], the packing the kernel streams."""
    k, n = w4.shape
    bits = lax.bitcast_convert_type(w4, jnp.uint4).astype(jnp.uint32)
    bits = bits.reshape(k // PACK, PACK, n)
    shifts = (jnp.arange(PACK, dtype=jnp.uint32) * 4)[None, :, None]
    return jnp.sum(bits << shifts, axis=1, dtype=jnp.uint32)


# --------------------------------------------------------------------------- #
def probe_native(m=128, k=512, n=512):
    key = jax.random.key(0)
    ka, kb = jax.random.split(key)
    a4 = random_fp4(ka, (m, k))
    b4 = random_fp4(kb, (k, n))
    a8 = a4.astype(FP8)
    want = np.asarray(a4.astype(jnp.float32) @ b4.astype(jnp.float32))
    res = {}

    def attempt(name, fn, compare=True):
        try:
            got = np.asarray(jax.block_until_ready(fn()))
            err = float(np.abs(got - want).max() / np.abs(want).max())
            res[name] = dict(ok=True, max_rel_err=err if compare else None)
        except Exception as e:  # a probe: the failure IS the answer
            res[name] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")

    attempt(
        "xla_f4xf4",
        lambda: jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32))(
            a4, b4
        ),
    )
    attempt(
        "xla_f8xf4",
        lambda: jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32))(
            a8, b4
        ),
    )
    # What XLA emits for the mixed dot: does it widen first?
    try:
        hlo = (
            jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32))
            .lower(a8, b4)
            .compile()
            .as_text()
        )
        res["xla_f8xf4_hlo_has_convert"] = "convert(" in hlo
        res["xla_f8xf4_hlo_dot_line"] = next(
            (
                ln.strip()[:300]
                for ln in hlo.splitlines()
                if " dot(" in ln or "convolution(" in ln
            ),
            None,
        )
    except Exception as e:
        res["xla_f8xf4_hlo"] = f"{type(e).__name__}: {str(e)[:200]}"

    b_u32 = packed_u32(b4)

    def pallas_dot(lhs_kind, widen):
        def kernel(a_ref, b_ref, o_ref):
            b = pltpu.bitcast(b_ref[...], FP4)
            if widen:
                b = b.astype(FP8)
            a = a_ref[...]
            if lhs_kind == "f4":
                a = pltpu.bitcast(a, FP4)
            o_ref[...] = lax.dot_general(a, b, DN, preferred_element_type=jnp.float32)

        lhs = packed_u32(a4) if lhs_kind == "f4" else a8
        call = pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32)
        )
        return jax.jit(call)(lhs, b_u32)

    attempt("pallas_f8xf8_widened_ref", lambda: pallas_dot("f8", True))
    attempt("pallas_f8xf4_native", lambda: pallas_dot("f8", False))
    # The packed LHS is packed along M (rows), so its values differ from a4;
    # this asks only whether an f4 x f4 contraction lowers at all.
    attempt("pallas_f4xf4_native", lambda: pallas_dot("f4", False), compare=False)
    return res


# --------------------------------------------------------------------------- #
def build_body(m, k, n, weight, passes, scale_blocks=True):
    nb = k // QB

    def kernel(a_ref, w_ref, s_ref, o_ref):
        def one_pass(_, acc):
            for b in range(nb):
                if weight in ("fp4", "fp4n"):
                    blk = pltpu.bitcast(
                        w_ref[pl.ds(b * (QB // PACK), QB // PACK), :], FP4
                    )
                    if weight == "fp4":
                        blk = blk.astype(FP8)
                else:
                    blk = w_ref[pl.ds(b * QB, QB), :]
                part = lax.dot_general(
                    a_ref[:, pl.ds(b * QB, QB)],
                    blk,
                    DN,
                    preferred_element_type=jnp.float32,
                )
                if scale_blocks:
                    part = part * s_ref[pl.ds(b, 1), :]
                acc = acc + part
            return acc

        o_ref[...] = lax.fori_loop(0, passes, one_pass, jnp.zeros((m, n), jnp.float32))

    packed = weight in ("fp4", "fp4n")
    w_shape = (k // PACK, n) if packed else (k, n)
    w_dtype = jnp.uint32 if packed else FP8
    vmem = pltpu.MemorySpace.VMEM
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
        out_specs=pl.BlockSpec(memory_space=vmem),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 2**20),
    )
    return jax.jit(call), w_shape, w_dtype


def probe_body(ms, n, weights, pass_depths, scale_blocks):
    k = 5120
    key = jax.random.key(1)
    out = []
    for weight in weights:
        for m in ms:
            a = jax.random.normal(key, (m, k)).astype(FP8)
            s = jnp.full((k // QB, n), 0.01, jnp.float32)
            times = {}
            for p in pass_depths:
                fn, w_shape, w_dtype = build_body(m, k, n, weight, p, scale_blocks)
                if weight in ("fp4", "fp4n"):
                    w = packed_u32(random_fp4(key, (k, n)))
                else:
                    w = random_fp4(key, (k, n)).astype(FP8)
                assert w.shape == w_shape and w.dtype == w_dtype
                times[p] = pipelined_ms(fn, (a, w, s))
            lo, hi = pass_depths
            us = (times[hi] - times[lo]) / (hi - lo) * 1e3
            flops = 2 * m * k * n
            rec = dict(
                probe="body",
                weight=weight,
                m=m,
                k=k,
                n=n,
                scale_blocks=scale_blocks,
                us_per_pass=us,
                tflops=flops / (us * 1e-6) / 1e12,
                raw_ms=times,
            )
            print("RESULT", json.dumps(rec), flush=True)
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--probe", choices=("native", "body"), required=True)
    ap.add_argument("--ms", default="32,64,128,256")
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--weights", default="fp8,fp4")
    ap.add_argument("--passes", default="4,16")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with jax.default_device(jax.devices()[0]):
        if args.probe == "native":
            recs = [dict(probe="native", jax=jax.__version__, **probe_native())]
            print("RESULT", json.dumps(recs[0]), flush=True)
        else:
            recs = probe_body(
                [int(v) for v in args.ms.split(",")],
                args.n,
                args.weights.split(","),
                tuple(int(v) for v in args.passes.split(",")),
                not args.no_scale,
            )
    if args.out:
        with open(args.out, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
