"""Step-3 probe R3: what are stage A's 8,960 `vrot.slane` actually doing?

Option C ("head-major score layout", design rank 2) rests entirely on
attributing P7b's **8,960 `vrot.slane`** in the prefill kernel body to the
per-head reduction `s.sum(axis=2)`.  `measurement_and_design.md` §4 Flip 2
names the way that fails: the rotates may instead be Mosaic's lane/sublane
shuffling for the `vpop.f32.mrb[*]` drain out of the MXU result buffer, in
which case reordering the score layout removes none of them.

This probe compiles a standalone Pallas kernel that reproduces
`streamindex_topk.py::compute_scores` at prefill geometry, in several arms
that differ *only* in how (or whether) the head axis is reduced, dumps LLO for
each, and diffs the opcode histograms.

    head_minor        the incumbent chain: einsum -> reshape (sb,bq,H,bkv)
                      -> relu -> xw -> sum(axis=2) -> xscale -> mask
    head_major        Option C: einsum LHS emitted as (sb, H*bq, D), reshape
                      (sb,H,bq,bkv), reduce over axis 1.  The q operand is
                      handed to the kernel already head-major -- i.e. this arm
                      prices Option C on the assumption that `load_bq` can
                      produce that ordering for free.
    head_major_xpose  the same, but the kernel is handed the incumbent's
                      (sb, bq*H, D) q block and transposes it in-kernel.  This
                      is the arm that prices the assumption above.
    no_reduce         head_minor up to and including the weight multiply, then
                      `s[:, :, 0, :]` -- no head reduction at all.  Same matmul
                      is *requested*; whether the compiler keeps it is checked
                      by the `vmatmul` control, and the arm is inconclusive if
                      it does not.
    no_reduce_x4      head_minor, but the reduction is only the cross-vreg
                      phase: reshape (sb,bq,4,8,bkv).sum(axis=2)[:, :, 0, :].
                      With H=32 on sublanes one vreg holds 8 heads, so this is
                      3 whole-vreg adds per group and *no* in-vreg rotate tree,
                      while every one of the 4096 matmul rows stays live.  It
                      is the drain test with a `vmatmul` control that should
                      hold.

Geometry: seq_batch=1, bq_sz=128, H=32, head_dim=128, bkv_sz=2048, fp8_e4m3fn
operands, f32 accumulation -- matching the incumbent prefill program
`StreamIdxTC-m-bq_128-bkvp_2.1`.  Note seq_batch is **1**, not 2: the leading
2 in that program's `f8e4m3fn[2,1,128,32,128]` q block and `f32[2,128,16,128]`
score block is the double buffer, and the `1` after it is seq_batch_size.

EVERYTHING THIS PROBE PRODUCES IS STATIC COMPILE-TIME ANALYSIS -- issue slots
emitted by the compiler.  Not a timing, not a throughput, and it must never
reach `/log-experiment`.
"""

import argparse
import os
import sys
import traceback

SB = 1
BQ = 128
H = 32
D = 128
BKV = 2048

ARMS = ("head_minor", "head_major", "head_major_xpose", "no_reduce",
        "no_reduce_x4")


def _p(*rest):
    print("[R3]", *rest, flush=True)


def _set_dump_env(outdir):
    """Must run before `import jax`.  libtpu reads LIBTPU_INIT_ARGS once, at
    load time, and these are LIBTPU_INIT_ARGS flags -- not XLA_FLAGS."""
    jf = os.path.join(outdir, "jf")
    mosaic = os.path.join(outdir, "mosaic")
    os.makedirs(jf, exist_ok=True)
    os.makedirs(mosaic, exist_ok=True)
    flags = (f"--xla_jf_dump_to={jf} "
             "--xla_jf_dump_llo_text=true "
             f"--xla_mosaic_dump_to={mosaic}")
    prev = os.environ.get("LIBTPU_INIT_ARGS", "")
    os.environ["LIBTPU_INIT_ARGS"] = (prev + " " + flags).strip()
    return jf, mosaic


def build(arm):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    head_major_in = (arm == "head_major")
    rows = H * BQ if head_major_in else BQ * H

    def body(q_ref, kv_ref, w_ref, scale_ref, pos_ref, klen_ref, out_ref):
        q = q_ref[...]                      # (SB, rows, D) f8e4m3fn
        kv = kv_ref[...]                    # (SB, BKV, D)  f8e4m3fn
        w = w_ref[...]                      # weights, bf16
        scale = scale_ref[...]              # (SB, 1, BKV) bf16
        pos = pos_ref[...]                  # (SB, BQ) int32
        klen = klen_ref[...]                # (SB, 1) int32

        if arm == "head_major_xpose":
            # Incumbent (bq*H, D) block reordered to (H*bq, D) inside the
            # kernel.  This is the cost Option C's file plan assumes away.
            q = q.reshape(SB, BQ, H, D).transpose(0, 2, 1, 3).reshape(
                SB, H * BQ, D)
            w = w.reshape(SB, BQ, H).transpose(0, 2, 1)

        s = jnp.einsum("bnd,bmd->bnm", q, kv,
                       preferred_element_type=jnp.float32)

        if arm in ("head_major", "head_major_xpose"):
            s = s.reshape(SB, H, BQ, BKV)
            s = jnp.maximum(s, 0.0)
            s = s * w.astype(jnp.float32)[:, :, :, None]   # w: (SB, H, BQ)
            s = s.sum(axis=1)
        else:
            s = s.reshape(SB, BQ, H, BKV)
            s = jnp.maximum(s, 0.0)
            s = s * w.astype(jnp.float32)[:, :, :, None]   # w: (SB, BQ, H)
            if arm == "head_minor":
                s = s.sum(axis=2)
            elif arm == "no_reduce":
                s = s[:, :, 0, :]
            elif arm == "no_reduce_x4":
                # cross-vreg phase only: 4 vregs of 8 heads each, whole-vreg
                # adds, no in-vreg rotate tree.  All 4096 matmul rows stay live.
                s = s.reshape(SB, BQ, 4, 8, BKV).sum(axis=2)[:, :, 0, :]
            else:
                raise ValueError(arm)

        s = s * scale
        k_span = jnp.arange(BKV, dtype=jnp.int32)
        valid = k_span[None, None, :] < klen[:, :, None]     # (SB,1,BKV)
        causal = k_span[None, None, :] <= pos[:, :, None]    # (SB,BQ,BKV)
        out_ref[...] = jnp.where(jnp.logical_and(valid, causal), s, -jnp.inf)

    wshape = (SB, H, BQ) if head_major_in else (SB, BQ, H)

    f8 = jnp.float8_e4m3fn
    in_specs = [
        jax.ShapeDtypeStruct((SB, rows, D), f8),
        jax.ShapeDtypeStruct((SB, BKV, D), f8),
        jax.ShapeDtypeStruct(wshape, jnp.bfloat16),
        jax.ShapeDtypeStruct((SB, 1, BKV), jnp.bfloat16),
        jax.ShapeDtypeStruct((SB, BQ), jnp.int32),
        jax.ShapeDtypeStruct((SB, 1), jnp.int32),
    ]

    from jax.experimental.pallas import tpu as pltpu
    scope_name = f"R3Score-{arm}"
    kernel = jax.named_scope(scope_name)(pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((SB, BQ, BKV), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=scope_name,
    ))
    return jax.jit(kernel), in_specs


def run_arm(arm, outdir):
    import jax
    import jax.numpy as jnp

    _p("arm", arm, "geometry",
       f"SB={SB} bq={BQ} H={H} D={D} bkv={BKV} fp8_e4m3fn -> f32")
    try:
        fn, specs = build(arm)
    except Exception:
        traceback.print_exc()
        return 1
    args = []
    key = jax.random.PRNGKey(0)
    for i, s in enumerate(specs):
        if s.dtype == jnp.int32:
            # pos: a full causal ramp; klen: the whole block valid
            args.append(jnp.full(s.shape, BKV, jnp.int32))
        elif s.dtype == jnp.float8_e4m3fn:
            key, k = jax.random.split(key)
            args.append(
                jax.random.normal(k, s.shape, jnp.float32).astype(s.dtype))
        else:
            key, k = jax.random.split(key)
            args.append(jax.random.normal(k, s.shape, jnp.float32).astype(
                s.dtype))
    out = fn(*args)
    jax.block_until_ready(out)
    _p("arm", arm, "ok; out", out.shape, out.dtype,
       "finite_frac", float(jnp.mean(jnp.isfinite(out))))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    jf, mosaic = _set_dump_env(args.outdir)
    _p("LIBTPU_INIT_ARGS", os.environ["LIBTPU_INIT_ARGS"])
    import jax
    _p("device", jax.devices()[0].device_kind, "jax", jax.__version__)
    rc = run_arm(args.arm, args.outdir)

    for d, label in ((jf, "jf"), (mosaic, "mosaic")):
        n = sum(len(f) for _, _, f in os.walk(d))
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(d) for f in fs)
        _p(f"dump {label}", f"{n} files, {sz/1e6:.1f} MB, at {d}")
        if n == 0:
            _p(f"dump {label}", "EMPTY -- the capture did not take.  These are "
               "LIBTPU_INIT_ARGS flags and must be set before `import jax`.")
            rc = rc or 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
