"""Step-1 probe: how fast is the `relu -> *weights -> sum over heads` chain really?

The prefill floor in report.md is 623us and is VPU-bound by 14x. That verdict
rests entirely on two values tagged `assumed`, and if both are wrong in the same
direction prefill is balanced rather than VPU-bound and the whole design target
changes:

  1. VPU rate 1.72e12 ops/s -- derived from a *ratio* in 05-compute-units.md
     (MXU 661.4e12 / 385), not measured for this op.
  2. 3 VPU ops per (token, head, position) element -- a guess at what Mosaic
     emits for relu, multiply, accumulate.

Rather than probe the two separately and multiply two uncertainties, this
measures the thing the floor is actually a claim about: the wall-clock of that
exact chain at the exact prefill intermediate shape. That collapses both
assumptions into one measured number, and the comparison against 586us is
direct.

Shape, from report.md section 3: prefill chunk T=1024 tokens, H=32 indexer
heads, S=10240 padded positions. The [T, H, S] intermediate is 32x larger than
the [T, S] score matrix it reduces to and never touches HBM inside the kernel --
so this probe deliberately measures the *reduction*, not a memory round trip,
and its inputs are sized to make that distinction visible.

Reports ops/s achieved so the number can be reused as a rate, and the implied
prefill VPU time so it can be read straight against the 586us model.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

T, H, S = 1024, 32, 10240
MODELLED_VPU_RATE = 1.72e12       # ops/s, assumed
MODELLED_OPS_PER_ELEM = 3         # relu, multiply, accumulate -- assumed
MODELLED_PREFILL_VPU_US = 586.0   # what the two assumptions above produce


def _p(tag, *rest):
    print(f"[probe] {tag}:", *rest, flush=True)


def time_fn(fn, *args, reps=10):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


# The chain as streamindex_topk writes it (compute_scores, lines 304-354):
# relu on the [t, h, s] product, scale by the per-(t,h) weight, sum over h.
def chain_asis(prod, w):
    s = jnp.maximum(prod, 0.0)
    s = s * w[:, :, None]
    return s.sum(axis=1)


# Algebraically identical for w >= 0, because relu(x)*w == relu(x*w).
# If this is materially faster, one VPU op per element is removable for free --
# and the indexer weights are a softmax-scaled projection, so w >= 0 is a
# property to verify, not to assume. Checked below.
def chain_folded(prod, w):
    return jnp.maximum(prod * w[:, :, None], 0.0).sum(axis=1)


# Floor for this shape: the reduction alone, no relu and no weight. Whatever
# this costs is irreducible, so it says how much of the 586us is even
# addressable by removing elementwise ops.
def chain_sumonly(prod, w):
    del w
    return prod.sum(axis=1)


def main():
    _p("device", jax.devices()[0].device_kind)
    _p("shape", f"T={T} H={H} S={S}",
       f"elements={T*H*S/1e9:.3f}G",
       f"intermediate_f32={T*H*S*4/1e9:.2f}GB (exceeds HBM if materialized)")

    # T*H*S f32 is 1.34 GB -- materializing it is itself the thing the real
    # kernel avoids, so measure on a bq tile and scale. bq_sz=128 is the
    # production value (num_queries_per_block=(1,128,128)).
    BQ = 128
    n_tiles = T // BQ
    prod = jnp.asarray(
        np.random.default_rng(0).standard_normal((BQ, H, S), dtype=np.float32))
    w = jnp.asarray(
        np.random.default_rng(1).random((BQ, H)).astype(np.float32))
    _p("tile", f"bq={BQ}", f"tiles={n_tiles}",
       f"tile_f32={BQ*H*S*4/1e6:.1f}MB")

    ref = np.asarray(jax.jit(chain_asis)(prod, w))
    for name, fn, ops_per_elem in [
        ("as-is  relu,*w,sum", chain_asis, 3),
        ("folded relu(x*w),sum", chain_folded, 3),
        ("sum only", chain_sumonly, 1),
    ]:
        f = jax.jit(fn)
        dt_tile = time_fn(f, prod, w)
        got = np.asarray(f(prod, w))
        # Only the folded form claims to match; sum-only is a floor, not a
        # variant, so do not assert on it.
        match = ("n/a" if fn is chain_sumonly
                 else f"{np.abs(got - ref).max():.3e}")
        elems = BQ * H * S
        rate = elems * ops_per_elem / dt_tile
        full_us = dt_tile * n_tiles * 1e6
        _p(name,
           f"tile={dt_tile*1e6:.1f}us",
           f"full_prefill={full_us:.1f}us",
           f"rate={rate/1e12:.3f}Tops/s",
           f"max_abs_diff_vs_asis={match}")

    _p("MODEL_says",
       f"vpu_rate={MODELLED_VPU_RATE/1e12:.3f}Tops/s (assumed)",
       f"ops_per_elem={MODELLED_OPS_PER_ELEM} (assumed)",
       f"-> prefill_vpu={MODELLED_PREFILL_VPU_US:.1f}us (modelled)")
    _p("READ_THIS_AS",
       "compare the as-is full_prefill against 586us. Within ~2x: the prefill "
       "floor stands and prefill is VPU-bound. Much faster: the floor is wrong, "
       "prefill is not VPU-bound by 14x, and the design target moves to HBM.")

    # Does the fold actually apply? relu(x)*w == relu(x*w) needs w >= 0.
    _p("fold_precondition",
       "weights are indexer_weights * softmax_scale * n_head**-0.5 * q_scales "
       "(deepseek_v4_indexer.py:81); sign is NOT guaranteed by construction. "
       "If any w < 0 at runtime the fold is invalid -- must be checked against "
       "real weights, not random ones, before any design uses it.")


if __name__ == "__main__":
    main()
