"""What actually sets stage A's score error, and what dtype the MXU sees.

RESULTS.md 8.4(2) left an open question: stage A's f32 scores differ from an
f64 oracle by ~1.06 absolute / 7.4e-04 relative median, which is bf16-scale and
~10^4x more than f32 accumulation over 32 heads predicts.  The proposed answer
was `hardware-facts.md:104-116` -- f32 operands at DEFAULT precision get one
bf16 pass, so f32 storage buys no f32 arithmetic.

**That answer cannot be right for this kernel, and this probe is what settles
it rather than arguing it.**  The score matmul's operands are not f32.  Both
are `float8_e4m3fn`: q is bitcast to fp8 at `mla_attention_op.py:106` behind an
assertion at `:153-154`, and k is bitcast in `load_bkv`
(`streamindex_topk.py:266`) out of a cache written by `quantize_tensor(...,
torch.float8_e4m3fn, ...)` at `mla_attention_op.py:156`.

And the oracle already excludes fp8 quantization: `_build_g2` reads the fp8
values back into f64 (`q_exact`, `kv_exact`), so the oracle scores exactly the
numbers the kernel scores.  Whatever sigma is, it is arithmetic, not
quantization.

THE LADDER
----------
Every arm below gets **bit-identical exact inputs**, because widening
float8_e4m3fn to bf16 or f32 is lossless -- 3 mantissa bits fit in 7 and in 23.
So the arms differ in exactly one thing: which MXU pass the compiler picks.

    fp8_native    float8_e4m3fn x float8_e4m3fn -> f32   (what ships)
    bf16          bf16 x bf16 -> f32                     (Option F's arithmetic)
    f32_default   f32 x f32 -> f32, DEFAULT
    f32_highest   f32 x f32 -> f32, HIGHEST

Two statistics per arm, against a numpy float64 reference on the same values:
the raw matmul, and the full stage-A composition (ReLU, weight, 32-head sum,
e8m0 scale) which is what sigma was measured on in 8.

Read it like this.  If `fp8_native` reproduces sigma and `f32_highest` does
not, the error is the MXU pass and the source is innocent.  If `bf16` equals
`fp8_native`, Option F's claim of "an identical result distribution" is
confirmed -- and is worth nothing, because F was a bandwidth argument and the
bytes go the wrong way.

Runs on the G2 configuration so the numbers are directly comparable with 8's.
Not a timing; safe to share a pod.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _probe_util import emit_json, p, tpu_banner  # noqa: E402

LADDER = ("fp8_native", "bf16", "f32_default", "f32_highest")

_CAST = {
    "fp8_native": None,
    "bf16": jnp.bfloat16,
    "f32_default": jnp.float32,
    "f32_highest": jnp.float32,
}
_PREC = {
    "fp8_native": None,
    "bf16": None,
    "f32_default": jax.lax.Precision.DEFAULT,
    "f32_highest": jax.lax.Precision.HIGHEST,
}


def _stats(got, ref):
    """max abs and relative error of `got` against an f64 `ref`."""
    got = np.asarray(got, np.float64)
    d = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = d / denom
    return dict(max_abs=float(d.max()),
                p99_abs=float(np.percentile(d, 99)),
                median_abs=float(np.median(d)),
                max_rel=float(rel.max()),
                median_rel=float(np.median(rel)))


def run_ladder(cfg, seed=0):
    import run_correctness_c as R

    b = R._build_g2(cfg, seed)
    H = cfg["H_I"]

    q8 = b["q"]                     # (T, H, D) float8_e4m3fn
    # The fp8 keys and their e8m0 scale, read straight back out of the cache
    # the kernel is handed -- no reconstruction.  `_build_g2` builds the record
    # as `(num_pages*ps, width)` and then plain-`reshape`s it, and
    # `page_indices` is `arange`, so undoing that reshape recovers the rows in
    # the order the kernel walks them.
    rec = np.asarray(b["cache_kv"]).reshape(-1, b["cache_kv"].shape[-1])
    S_valid = b["S_valid"]
    kq_bytes = rec[:S_valid, :cfg["D"]].astype(np.uint8)
    kv8 = jax.lax.bitcast_convert_type(jnp.asarray(kq_bytes),
                                       jnp.float8_e4m3fn)
    scale_byte = rec[:S_valid, cfg["D"]].astype(np.uint8)
    scale8 = jax.lax.bitcast_convert_type(jnp.asarray(scale_byte),
                                          jnp.float8_e8m0fnu)
    scale_f64 = np.asarray(jnp.asarray(scale8,
                                       jnp.float32)).astype(np.float64)

    q_exact = np.asarray(jnp.asarray(q8, jnp.float32)).astype(np.float64)
    k_exact = np.asarray(jnp.asarray(kv8, jnp.float32)).astype(np.float64)
    w_exact = b["w_exact"]

    p("dtypes the score matmul actually receives:",
      dict(q=str(q8.dtype), k=str(kv8.dtype), scale=str(scale8.dtype),
           weights=str(b["w"].dtype), accumulator="float32 "
           "(preferred_element_type)"))
    p("exactness of the widening (must be 0.0 for the ladder to be fair):",
      dict(q_fp8_to_bf16=float(np.abs(
          np.asarray(jnp.asarray(q8.astype(jnp.bfloat16),
                                 jnp.float32)).astype(np.float64) -
          q_exact).max()),
           k_fp8_to_bf16=float(np.abs(
               np.asarray(jnp.asarray(kv8.astype(jnp.bfloat16),
                                      jnp.float32)).astype(np.float64) -
               k_exact).max())))

    # ---- the f64 reference, both stages -----------------------------------
    inner_ref = np.einsum("thd,sd->ths", q_exact, k_exact)
    scored = np.maximum(inner_ref, 0.0) * w_exact[:, :, None]
    summed_ref = scored.sum(axis=1) * scale_f64[None, :]

    rows = []
    for arm in LADDER:
        cast, prec = _CAST[arm], _PREC[arm]
        qa = q8 if cast is None else q8.astype(cast)
        ka = kv8 if cast is None else kv8.astype(cast)
        kw = {} if prec is None else {"precision": prec}

        @jax.jit
        def f(qa, ka, w, sc):
            inner = jnp.einsum("thd,sd->ths", qa, ka,
                               preferred_element_type=jnp.float32, **kw)
            s = jnp.maximum(inner, 0.0) * w.astype(jnp.float32)[:, :, None]
            return inner, s.sum(axis=1) * sc[None, :]

        inner, summed = jax.block_until_ready(
            f(qa, ka, b["w"], jnp.asarray(scale8, jnp.float32)))
        row = dict(arm=arm,
                   operand_dtype=str(qa.dtype),
                   precision=str(prec),
                   matmul=_stats(inner, inner_ref),
                   stage_a=_stats(summed, summed_ref))
        rows.append(row)
        p(f"{arm:12} matmul max_rel={row['matmul']['max_rel']:.4e} "
          f"| stage_a max_abs={row['stage_a']['max_abs']:.4e} "
          f"median_rel={row['stage_a']['median_rel']:.4e}")

    return rows, dict(ref_abs_max=float(np.abs(summed_ref).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    tpu_banner()
    import run_correctness_c as R
    import stage_a_c as S
    p("incumbent module", S.sx.__file__, S.source_md5(S.sx))
    p("variant module", S.sxc.__file__, S.source_md5(S.sxc))

    out = []
    for seed in (int(s) for s in args.seeds.split(",") if s):
        p("=" * 70)
        p("seed", seed, "cfg", R.G2)
        rows, meta = run_ladder(R.G2, seed)
        out.append(dict(seed=seed, rows=rows, **meta))

    print("\n===== MXU PRECISION LADDER, identical exact inputs =====")
    print(f"{'seed':>4} {'arm':12} {'operands':16} "
          f"{'matmul max_rel':>15} {'stageA max_abs':>15} "
          f"{'stageA med_rel':>15}")
    for e in out:
        for r in e["rows"]:
            print(f"{e['seed']:4d} {r['arm']:12} {r['operand_dtype']:16} "
                  f"{r['matmul']['max_rel']:15.4e} "
                  f"{r['stage_a']['max_abs']:15.4e} "
                  f"{r['stage_a']['median_rel']:15.4e}")
    print("\nRESULTS.md 8.3 measured the shipping kernel's stage-A error at "
          "max_abs 1.0628 / median_rel 7.4e-04 on this same configuration.")
    print("The arm that matches it is the arm that explains sigma.")

    emit_json("stage_a_precision_ladder",
              dict(what="which MXU pass sets stage A's score error",
                   note="all arms receive bit-identical exact inputs; "
                        "fp8->bf16 and fp8->f32 are lossless widenings",
                   seeds=out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
