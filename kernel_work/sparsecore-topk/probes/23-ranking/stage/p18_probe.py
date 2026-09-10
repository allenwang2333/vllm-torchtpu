"""Probe 18 -- the rho denominator: what does the incumbent ACTUALLY cost at
the production `row_lengths` distribution, and does the 1.219 hold?

WHY. Every probe in this exercise ran `row_lengths = jnp.full((b,), n)`, i.e.
rho = E[eff_len]/n = 1. The whole step-2 balance sheet is a REWEIGHTING of that
rho=1 measurement:

    measured B(rho=1) = 326.782 us   (probes/09, j-4e0df30a, 2-D entry)
      -> x per-term rho weights      (hardware_map.md:521-536, stage C")
    modelled harness  = 189.98 us    (design.md:17-20)
      -> - 34.165 us of harness-only XLA relayout (probes/11)
    modelled production = 155.82 us  (design.md:37-45), single factor 1.219

Nothing has ever executed at the production rho. So `189.98`, `155.82` and
`1.219` are three predictions, and every `Delta time %` in `design.md` is a
ratio of two of them. One job converts the denominator to `measured`.

WHAT IS MEASURED. The UNMODIFIED incumbent (md5 44353b34..., commit b11b6eb6),
timed at many `row_lengths` vectors in ONE pool. No source patching at all:
every prefill arm is the SAME compiled executable and differs ONLY in the
`lengths` operand, so an arm-to-arm difference cannot be a compile artifact.

Three axes, crossed only where the question needs it:

  1. rho, via `row_lengths`.  `eff_len = min(row_lengths[row], n)`;
     `i_hi = min(slice_len, eff_len)`; `walk_chunks = ceil(i_hi/16)`. So `cvt`,
     `hist0`, the refinement walks and `emit` are walk-bounded and scale with
     rho, while the score DMA (`pl.ds(s, slice_len)`) is STATIC and does not.
     `keep_all = total <= k` additionally kills the whole refinement loop when
     eff_len <= 2048.
  2. the DISTRIBUTION at a fixed mean.  rho is a mean; a top-k kernel's cost is
     not linear in it. Three distributions at E[eff_len] = 4096 are timed:
     flat (degenerate), the production uniform ramp, and a bimodal pair.
  3. the jit ENTRY SHAPE.  `f32[b,n]` (the harness's, 189.98) vs
     `f32[b,n/128,128]` (production's, 155.82). The 1.219 IS this difference,
     so it is measured as a within-pool A/B at the production rho.

THE DISTRIBUTION, AND WHERE IT COMES FROM. `prefill_lengths`/`decode_lengths`
are copied VERBATIM from `probes/13-level-count-refit/level_sim.py:254-266`
(gated by --level-sim). They are not invented here:

  prefill  eff_len ~ Uniform{1..8192}      E = 4096.5, rho = 0.40005
  decode   seq_len ~ Uniform{8193..9215}   E = 8704.0, rho = 0.850

which is `_effective_row_lengths` (`streamindex_topk.py:499-506`) at
`compression_ratio = 1`, over ISL 8192 / max-model-len 9216 / chunk 1024 from
`configs/glm-5.2-fp8-dp16-ep.sh` with `RANDOM_RANGE_RATIO=0.0`.

THE GATED SUB-QUESTION. `streamindex_topk.py:860-863` elides the kernel whenever
`max(seq_lens) <= k`, so prefill chunks 0 and 1 NEVER REACH IT. Stage C" applies
the conditional rho of 0.50005 to the refinement terms only, and the
unconditional 0.40005 to `hist0`/`emit`/`cvt`. Both cannot be right. The `gated`
arm (Uniform{2049..8192}, rho = 0.50005) and the eight per-chunk causal ramps
settle which rho the production denominator should be quoted at.

SCORES. The production DSA indexer distribution -- `make_kv` + `make_scores`
from `probes/13/level_sim.py`, verbatim, as `probes/16` used. One draw is shared
by every prefill arm, so the score data cancels exactly in every rho contrast.
Three EXTRA draws are registered as data-nulls, which prices what a different
draw is worth. Two standard-normal arms are also registered as the cross-job
control against `probes/09`'s `B`, which used `rng.standard_normal`.

FLOOR. Three nulls per shape: the identical incumbent source, exec'd into three
SEPARATE module objects, jitted separately, on the SAME operands. Their spread
is pure instrument noise -- three things that must be equal. Reported as the
resolution floor; anything inside it is `not resolvable`, not `no effect`.

NOT MEASURED HERE: correctness of any variant (there are none), end-to-end
effect, anything multi-chip.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import random
import sys
import time
import types

import ml_dtypes
import numpy as np

SEED13 = 20260910        # probes/13/level_sim.py SEED -- scores AND lengths
SEED09 = 20260904        # probes/05/07/08/09 stream -- the standard-normal arm
SEED_LEN = 20260911      # this probe's length-draw stream
REPS_LO = 20
REPS_HI = 60

# The pinned subject: the exact source probes/07/08/09/16 measured.
SUBJECT_MD5 = "44353b347c96e0eedbd742c710d4540a"

# ---- the sheet under test (design.md:17-45, hardware_map.md:521-536) --------
SHEET_B_RHO1 = 326.782     # us, MEASURED at rho=1, probes/09 j-4e0df30a, 2-D entry
SHEET_HARNESS = 189.98     # us, MODELLED at rho=0.40005, 2-D entry
SHEET_PRODUCTION = 155.82  # us, MODELLED at rho=0.40005, 3-D entry
SHEET_ENTRY_FACTOR = 1.219  # = 189.98 / 155.82
SHEET_RHO_FACTOR = SHEET_B_RHO1 / SHEET_HARNESS   # 1.7201, implied
RHO_PRE = 0.40005
RHO_PRE_GATED = 0.50005
RHO_DEC = 0.850


# ==========================================================================
# level_schedule -- VERBATIM from probes/13/level_sim.py. Gated by --level-sim.
# ==========================================================================
def level_schedule(num_buckets):
    """(l0_shift, [(lvl_shift, lvl_mask), ...]) for a given radix.

    NUM_BUCKETS=256 reproduces the shipped constants exactly:
    L0_SHIFT=24, REFINEMENT_LEVELS=((16,0xFF),(8,0xFF),(0,0xFF)).
    Smaller/larger radices keep the same rule -- consume `bits` per level,
    clamp the last level at shift 0 -- which is `modelled`, not in source.
    """
    bits = int(math.log2(num_buckets))
    assert 2**bits == num_buckets
    shifts = []
    s = 32 - bits
    while s > 0:
        s = max(0, s - bits)
        shifts.append(s)
    l0 = 32 - bits
    prev = l0
    levels = []
    for s in shifts:
        levels.append((s, (1 << (prev - s)) - 1))
        prev = s
    return l0, tuple(levels)


# ==========================================================================
# Score generators -- VERBATIM from probes/13/level_sim.py (and probes/16).
# ==========================================================================
FP8 = ml_dtypes.float8_e4m3fn
FP8_MAX = float(ml_dtypes.finfo(FP8).max)  # 448.0
E8M0_MIN_EXP, E8M0_MAX_EXP = -127, 127


def _quant_fp8_rowwise(x, ue8m0):
    """`quantization.py:217-300` on the last axis, block = whole axis."""
    absmax = np.abs(x).max(axis=-1, keepdims=True).astype(np.float32)
    scale = absmax * np.float32(1.0 / FP8_MAX)
    if ue8m0:
        with np.errstate(divide="ignore"):
            e = np.ceil(np.log2(scale))
        e = np.where(np.exp2(e) < scale, e + 1, e)
        e = np.clip(e, E8M0_MIN_EXP, E8M0_MAX_EXP)
        scale = np.exp2(e).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / np.maximum(scale, 1e-38), 0.0).astype(np.float32)
    q = np.clip(x * inv, -FP8_MAX, FP8_MAX).astype(FP8)
    return q, scale


def make_kv(n, head_dim, alpha, rng):
    """Compressed-KV rows after `k_norm` (a LayerNorm), fp8+ue8m0 quantized.

    `alpha` is the cosine weight of a direction shared by every KV row -- the
    residual-stream common mode.  Not determinable from source; swept.
    RoPE is skipped: it is an orthogonal per-pair rotation and leaves an
    isotropic Gaussian exactly invariant.
    """
    c = rng.standard_normal(head_dim).astype(np.float32)
    c /= np.linalg.norm(c) / math.sqrt(head_dim)
    kv = rng.standard_normal((n, head_dim)).astype(np.float32)
    kv = alpha * c[None, :] + math.sqrt(max(0.0, 1 - alpha * alpha)) * kv
    kv -= kv.mean(axis=-1, keepdims=True)                       # LayerNorm
    kv /= kv.std(axis=-1, keepdims=True)
    kq, kscale = _quant_fp8_rowwise(kv, ue8m0=True)
    # `scale_val` is read back as bf16 (streamindex_topk.py:267-268); a power
    # of two is exact in bf16.
    return kq, kscale.astype(np.float32).reshape(-1)


def make_scores(kq, kscale, b, n_head, head_dim, rng, signed_w=True,
                head_scale_sigma=0.0, chunk=32):
    """One score row per query token, exactly `streamindex_topk.py:326-338`.

        s = einsum(q_fp8, k_fp8, f32)      :326-331
        s = relu(s)                        :333
        s = s * weights_f32                :334   weights bf16, SIGNED
        s = sum over heads                 :335
        s = s * kscale                     :337
    """
    n = kq.shape[0]
    out = np.empty((b, n), np.float32)
    kqf = kq.astype(np.float32)
    softmax_scale = head_dim**-0.5
    for lo in range(0, b, chunk):
        hi = min(b, lo + chunk)
        m = hi - lo
        q = rng.standard_normal((m, n_head, head_dim)).astype(np.float32)
        if head_scale_sigma:
            q *= np.exp(head_scale_sigma * rng.standard_normal(
                (m, n_head, 1)).astype(np.float32))
        qq, qscale = _quant_fp8_rowwise(q, ue8m0=False)       # axis=-1, fp32
        w_raw = rng.standard_normal((m, n_head)).astype(np.float32)
        if not signed_w:
            w_raw = np.abs(w_raw)
        w = (w_raw * softmax_scale * (n_head**-0.5) * qscale[..., 0])
        w = w.astype(ml_dtypes.bfloat16).astype(np.float32)   # bf16 round-trip
        s = qq.astype(np.float32).reshape(m * n_head, head_dim) @ kqf.T
        np.maximum(s, 0.0, out=s)
        s = s.reshape(m, n_head, n)
        out[lo:hi] = np.einsum("bh,bhn->bn", w, s, dtype=np.float32)
    out *= kscale[None, :]
    return out


# ---------------------------------------------------------- rho models ------
def prefill_lengths(b, rng):
    """ISL 8192, chunked prefill, compression_ratio 1: token_pos+1 is uniform
    on [1, 8192].  E/n = 4096.5/10240 = 0.40005 and P(eff<=2048) = 0.25 --
    both stated facts, exactly (hardware_map.md:379, design.md:1106)."""
    return rng.integers(1, 8193, size=b).astype(np.int64)


def decode_lengths(b, rng):
    """seq_len in [8193, 9215] -> E/n = 8704/10240 = 0.850."""
    return rng.integers(8193, 9216, size=b).astype(np.int64)


def strat_uniform(lo, hi, b, seed):
    """A stratified draw from Uniform{lo..hi} whose SAMPLE mean equals the
    POPULATION mean exactly.

    `prefill_lengths(1024, ...)` is an i.i.d. draw, so its realized rho is
    0.39604, not 0.40005 -- a 1.0% miss, because sigma/sqrt(1024) is 0.0072 in
    rho. That error lands directly on the number this probe exists to produce.
    This is the quantile function of the SAME distribution on a regular grid
    (rounded so the total is exact), then randomly permuted across rows so the
    within-wave spread matches an i.i.d. draw. Same support, same shape, same
    quantiles to +/-1, zero sampling error in the mean.
    """
    m = hi - lo + 1
    j = np.arange(b)
    t = lo + (j + 0.5) * m / b - 0.5          # exact quantile midpoints
    v = np.floor(t).astype(np.int64)
    need = int(round(float(t.sum()))) - int(v.sum())
    order = np.argsort(-(t - v), kind="stable")
    v[order[:need]] += 1
    v = np.clip(v, lo, hi)
    assert v.sum() == int(round(float(t.sum()))), (v.sum(), t.sum())
    return np.random.default_rng(seed).permutation(v)


HEAD_DIM, N_HEAD = 128, 64   # probes/13/level_sim.py:293


def production_scores(b, n, rng):
    """probes/13/level_sim.py STEP 4b, verbatim: alpha=0."""
    kq, ks = make_kv(n, HEAD_DIM, 0.0, rng)
    return make_scores(kq, ks, b, N_HEAD, HEAD_DIM, rng)


# ==========================================================================
# The length vectors. Every one of these is a `row_lengths` operand for the
# SAME executable -- no recompiles, no source edits.
# ==========================================================================
def build_lengths(b: int, n: int, k: int) -> dict:
    """`{key: (np.int32[b], provenance)}`. `key` names the arm suffix."""
    rng = np.random.default_rng(SEED_LEN)
    out: dict[str, tuple[np.ndarray, str]] = {}

    def put(key, arr, prov):
        a = np.asarray(arr, np.int32)
        assert a.shape == (b, ), (key, a.shape)
        out[key] = (a, prov)

    # --- the two ends -------------------------------------------------------
    put("rho1", np.full(b, n), "row_lengths = n: what EVERY prior probe ran")

    # --- the production distributions, verbatim from probes/13 --------------
    if b >= 512:
        put("prefill", prefill_lengths(b, np.random.default_rng(SEED_LEN)),
            "level_sim.prefill_lengths VERBATIM: i.i.d. Uniform{1..8192}; "
            "target rho 0.40005, realized rho is the sample mean")
        # THE HEADLINE ARM. Same distribution, zero sampling error in rho.
        sp = strat_uniform(1, 8192, b, SEED_LEN)
        put("prefillX", sp,
            "stratified Uniform{1..8192}, permuted: rho=0.40005 EXACTLY")
        # Identical MULTISET to prefillX, monotone row order. Rows map to
        # subcores with stride 32 and a wave costs its slowest subcore, so a
        # sorted batch has almost no within-wave spread and a shuffled one has
        # the full range. This arm prices ARRANGEMENT at a fixed multiset.
        put("prefillXsorted", np.sort(sp),
            "prefillX multiset, sorted ascending: same rho, no within-wave "
            "spread")
        # Conditional on the outer lax.cond firing: chunks 2..7 only, i.e.
        # token_pos >= 2048, i.e. eff_len >= 2049. streamindex_topk.py:860-863.
        put("gated",
            np.random.default_rng(SEED_LEN + 1).integers(2049, 8193, size=b),
            "i.i.d. Uniform{2049..8192} = prefill CONDITIONAL on the lax.cond "
            "gate; target rho 0.50005")
        put("gatedX", strat_uniform(2049, 8192, b, SEED_LEN + 1),
            "stratified Uniform{2049..8192}: rho=0.50005 EXACTLY")
        # Same mean as `prefill` (4096), three different distributions.
        put("flat0.400", np.full(b, 4096), "degenerate at E[prefill]=4096")
        half = np.empty(b, np.int64)
        half[0::2] = 2048
        half[1::2] = 6144
        put("bimodal0.400", half, "bimodal {2048, 6144}, mean 4096")
        # The eight per-chunk causal ramps: chunk c row i sees 1024c+i+1.
        for c in range(8):
            put(f"chunk{c}", np.arange(b, dtype=np.int64) + 1024 * c + 1,
                f"chunk {c} causal ramp, eff_len = 1024*{c}+i+1")
        # A shape sweep of degenerate distributions.
        for L in (512, 1536, 2560, 5120, 6656, 8704):
            put(f"flat{L / n:.3f}", np.full(b, L), f"degenerate at {L}")
    else:
        put("decode", decode_lengths(b, np.random.default_rng(SEED_LEN + 2)),
            "level_sim.decode_lengths VERBATIM: i.i.d. Uniform{8193..9215}; "
            "target rho 0.850, realized rho is the sample mean")
        put("decodeX", strat_uniform(8193, 9215, b, SEED_LEN + 2),
            "stratified Uniform{8193..9215}: rho=0.850 EXACTLY")
        put("flat0.850", np.full(b, 8704), "degenerate at E[decode]=8704")
        for L in (2048, 4096, 6144, 8192, 9216):
            put(f"flat{L / n:.3f}", np.full(b, L), f"degenerate at {L}")
    del rng
    return out


# ==========================================================================
# Timing harness -- the FROZEN instrument, or a byte-identical inline copy.
# ==========================================================================
_LAST_ORDER: list[list[str]] = []


def _time(fn, reps: int) -> float:
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    import jax
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def _measure_inline(variants, rounds=9, shuffle_seed=20260902):
    import jax
    rng = random.Random(shuffle_seed)
    lo_best, hi_best, slopes, order_log = {}, {}, {}, []
    for name in variants:
        jax.block_until_ready(variants[name]())
    for _ in range(rounds):
        names = list(variants)
        rng.shuffle(names)
        order_log.append(names)
        lo_r = {}
        for name in names:
            lo_r[name] = _time(variants[name], REPS_LO)
            lo_best[name] = min(lo_best.get(name, math.inf), lo_r[name])
        for name in names:
            hi = _time(variants[name], REPS_HI)
            hi_best[name] = min(hi_best.get(name, math.inf), hi)
            slopes.setdefault(name, []).append(hi - lo_r[name])
    out = {}
    span = REPS_HI - REPS_LO
    for name in variants:
        s = sorted(slopes[name])
        nn = len(s)
        med = s[nn // 2] if nn % 2 else 0.5 * (s[nn // 2 - 1] + s[nn // 2])
        per_device = s[0] / span
        naive = hi_best[name] / REPS_HI
        host = max(naive - per_device, 0.0)
        out[name] = {
            "us": per_device * 1e6,
            "us_median": med / span * 1e6,
            "us_mean": (sum(s) / nn) / span * 1e6,
            "us_unpaired": (hi_best[name] - lo_best[name]) / span * 1e6,
            "hostbound_pct": (host / naive * 100.0) if naive > 0 else 0.0,
            "suspect": bool(min(s) <= 0.0 or per_device <= 0.0),
        }
        if out[name]["suspect"]:
            print(f"  !! SUSPECT {name}: min slope <= 0 -- unwarmed recompile. "
                  f"`us` invalid; read us_median "
                  f"({out[name]['us_median']:.2f}).", file=sys.stderr)
    global _LAST_ORDER
    _LAST_ORDER = order_log
    return out


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _load_harness(extra_dirs):
    for d in extra_dirs:
        cand = os.path.join(d, "glm_dsa_block_bench.py")
        if not os.path.isfile(cand):
            continue
        try:
            if d not in sys.path:
                sys.path.insert(0, d)
            mod = importlib.import_module("glm_dsa_block_bench")
            if (mod.REPS_LO, mod.REPS_HI) != (REPS_LO, REPS_HI):
                print(f"  harness at {cand} has REPS "
                      f"{mod.REPS_LO}/{mod.REPS_HI} != {REPS_LO}/{REPS_HI} "
                      f"-- refusing it, using inline", flush=True)
                continue
            print(f"  timing harness: FROZEN INSTRUMENT {cand} "
                  f"(md5 {_md5(open(cand, 'rb').read())})", flush=True)
            return (mod.measure, (lambda: mod.LAST_ORDER), "frozen",
                    mod.dispatch_floor)
        except Exception as exc:  # pragma: no cover
            print(f"  harness import from {d} failed: {exc!r}", flush=True)
    print("  timing harness: INLINE COPY (frozen instrument not importable)",
          flush=True)
    return _measure_inline, (lambda: _LAST_ORDER), "inline", (lambda: float("nan"))


_MODCOUNT = itertools.count()


def build_mod(src: str, tag: str) -> types.ModuleType:
    mod = types.ModuleType(f"sctopk18_{tag}_{next(_MODCOUNT)}")
    mod.__file__ = f"<probe18:{tag}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


# ==========================================================================
# Host-side replay of the kernel's own control flow. MODELLED -- it explains
# the measured curve, it does not stand in for it.
# ==========================================================================
def replay_levels(scores: np.ndarray, lengths: np.ndarray, k: int,
                  num_buckets: int = 256, sample: int = 128) -> dict:
    """Mean refinement levels executed per row, and the keep_all fraction.

    Mirrors `sparsecore_topk.py:187-265` exactly: monotone key, L0 bucket
    `(key >> 24) + 128`, `find_bucket` = #buckets whose inclusive cumsum is
    <= thresh, and the `done` short-circuit.
    """
    l0_shift, levels = level_schedule(num_buckets)
    b, n = scores.shape
    idx = np.linspace(0, b - 1, min(sample, b)).astype(int)
    lv, ka = [], []
    for r in idx:
        L = int(min(lengths[r], n))
        if L <= 0:
            lv.append(0)
            ka.append(True)
            continue
        bits = scores[r, :L].view(np.int32)
        key = bits ^ (np.right_shift(bits, 31) & np.int32(0x7FFFFFFF))
        key = key.astype(np.int64)
        bucket = (key >> l0_shift) + num_buckets // 2
        hist = np.bincount(np.clip(bucket, 0, num_buckets - 1),
                           minlength=num_buckets)
        total = int(hist.sum())
        keep_all = total <= k
        if keep_all:
            lv.append(0)
            ka.append(True)
            continue
        cum = np.cumsum(hist)
        b_star = int((cum <= total - k).sum())
        cum_at = int(cum[b_star])
        cum_lo = int(cum[b_star - 1]) if b_star > 0 else 0
        cnt = cum_at - cum_lo
        quota = k - (total - cum_at)
        prefix = b_star - num_buckets // 2
        shift = l0_shift
        done = quota == cnt
        used = 0
        for lvl_shift, lvl_mask in levels:
            if done:
                break
            used += 1
            match = (key >> shift) == prefix
            bb = (key[match] >> lvl_shift) & lvl_mask
            h2 = np.bincount(bb, minlength=num_buckets)
            sub_total = int(h2.sum())
            c2 = np.cumsum(h2)
            b2 = int((c2 <= sub_total - quota).sum())
            b2 = min(b2, num_buckets - 1)
            cum_at2 = int(c2[b2])
            cum_lo2 = int(c2[b2 - 1]) if b2 > 0 else 0
            cnt2 = cum_at2 - cum_lo2
            quota2 = quota - (sub_total - cum_at2)
            prefix = (prefix << (shift - lvl_shift)) | b2
            shift = lvl_shift
            quota, cnt = quota2, cnt2
            done = (quota2 == cnt2) or (quota2 == 0)
        lv.append(used)
        ka.append(False)
    return {"sampled_rows": len(idx), "mean_levels": float(np.mean(lv)),
            "keep_all_frac": float(np.mean(ka))}


# ==========================================================================
# Fits
# ==========================================================================
def _linfit(xs, ys):
    nn = len(xs)
    mx = sum(xs) / nn
    my = sum(ys) / nn
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx if sxx else float("nan")
    c = my - m * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (c + m * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return c, m, r2


def _spread(vals):
    return max(abs(a - c) for a, c in itertools.combinations(vals, 2))


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=9216)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--b-prefill", type=int, default=1024)
    ap.add_argument("--b-decode", type=int, default=64)
    ap.add_argument("--harness-dir", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--level-sim", default=None)
    args = ap.parse_args()

    if args.dry_run:
        return _dry_run(args)

    os.makedirs(args.outdir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import vllm_torchtpu  # noqa: F401  (registers the TPU platform)
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info()
    print(f"host={os.uname().nodename} devices={jax.device_count()} "
          f"kind={jax.devices()[0].device_kind}", flush=True)
    print(f"sparse_core={info.sparse_core}  jax={jax.__version__}", flush=True)

    # ---------------- SUBJECT GATE (in-job; the tag is mutable) -------------
    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as installed
    inst_md5 = _md5(open(installed.__file__, "rb").read())
    base_path = args.subject or installed.__file__
    raw = open(base_path, "rb").read()
    base_src = raw.decode()
    print(f"subject: {base_path}  md5={_md5(raw)}", flush=True)
    same = "SAME" if inst_md5 == _md5(raw) else (
        "DIFFERENT -- using the pinned --subject, not the image copy")
    print(f"installed: {installed.__file__}  md5={inst_md5}  {same}", flush=True)
    if _md5(raw) != SUBJECT_MD5:
        print(f"SUBJECT_GATE_FAIL: {_md5(raw)} != {SUBJECT_MD5}", flush=True)
        return 2
    print(f"SUBJECT_GATE_OK  {SUBJECT_MD5}", flush=True)

    inc = build_mod(base_src, "incumbent")
    k = args.k
    n = ((args.ctx + 1024) + 15) // 16 * 16
    assert n % 128 == 0, n
    assert n == 10240, (
        f"n={n}: --ctx 9216 is the shipping shape and every step-1 number is "
        f"at it. The bench default --ctx 132096 gives n=133120, a shape no "
        f"checked-in config produces.")

    bp, bd = args.b_prefill, args.b_decode
    for b in (bp, bd):
        p = inc._pick_partition(b, n, k)
        assert p == 1, f"b={b}: P={p}, single-stage path only"
    print(f"shapes: prefill b={bp} waves={(bp + 31) // 32}  "
          f"decode b={bd} waves={(bd + 31) // 32}  n={n} k={k} P=1  "
          f"ctx={args.ctx}", flush=True)

    results = {
        "b_prefill": bp, "b_decode": bd, "n": n, "k": k, "P": 1,
        "ctx": args.ctx, "rounds": args.rounds, "subject_md5": _md5(raw),
        "installed_md5": inst_md5, "seed_scores": SEED13,
        "seed_lengths": SEED_LEN,
        "sheet": {"B_rho1_us": SHEET_B_RHO1, "harness_us": SHEET_HARNESS,
                  "production_us": SHEET_PRODUCTION,
                  "entry_factor": SHEET_ENTRY_FACTOR,
                  "implied_rho_factor": SHEET_RHO_FACTOR},
    }

    # ---------------- operands ---------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("STAGE 0  OPERANDS", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    sc_pre = production_scores(bp, n, np.random.default_rng(SEED13))
    sc_dec = production_scores(bd, n, np.random.default_rng(SEED13 + 1))
    sc_norm = np.random.default_rng(SEED09).standard_normal(
        (bp, n)).astype(np.float32)
    sc_nulls = [production_scores(bp, n, np.random.default_rng(SEED13 + 10 + j))
                for j in range(3)]
    print(f"  scores built in {time.time() - t0:.1f}s", flush=True)
    for nm, a in (("prefill", sc_pre), ("decode", sc_dec), ("normal", sc_norm)):
        print(f"  {nm:8s} shape={a.shape} zeros={float((a == 0).mean())*100:.2f}%"
              f" md5={_md5(a.tobytes())[:16]}", flush=True)

    dev = {}

    def put_dev(key, arr2d):
        dev[key + "/2d"] = jnp.asarray(arr2d)
        dev[key + "/3d"] = jnp.asarray(arr2d.reshape(arr2d.shape[0],
                                                     n // 128, 128))

    put_dev("pre", sc_pre)
    put_dev("dec", sc_dec)
    put_dev("norm", sc_norm)
    for j, a in enumerate(sc_nulls):
        put_dev(f"null{j}", a)

    lens_pre = build_lengths(bp, n, k)
    lens_dec = build_lengths(bd, n, k)
    dev_len = {}
    len_meta = {}
    for tag, tbl in (("P", lens_pre), ("D", lens_dec)):
        for key, (arr, prov) in tbl.items():
            dev_len[f"{tag}.{key}"] = jnp.asarray(arr)
            eff = np.minimum(arr.astype(np.int64), n)
            len_meta[f"{tag}.{key}"] = {
                "rho": float(eff.mean() / n), "min": int(eff.min()),
                "max": int(eff.max()), "mean": float(eff.mean()),
                "frac_le_k": float((eff <= k).mean()), "provenance": prov,
                "md5": _md5(arr.tobytes())[:16]}
    print("\n  length vectors:", flush=True)
    for key, m in len_meta.items():
        print(f"  {key:18s} rho={m['rho']:.5f}  eff in [{m['min']},{m['max']}]"
              f"  P(eff<=k)={m['frac_le_k']*100:5.1f}%  md5={m['md5']}"
              f"  {m['provenance']}", flush=True)
    results["lengths"] = len_meta

    # =====================================================================
    # STAGE 1 -- LENGTHS GATE. Does `row_lengths` actually reach the kernel,
    # and is the answer still exact at every one of them?
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 1  LENGTHS GATE -- correctness AND arm activation", flush=True)
    print("=" * 78, flush=True)

    f_pre3 = jax.jit(lambda s, l: inc.sparsecore_topk(
        s.reshape(bp, n), k=k, row_lengths=l))
    f_pre2 = jax.jit(lambda s, l: inc.sparsecore_topk(s, k=k, row_lengths=l))
    f_dec3 = jax.jit(lambda s, l: inc.sparsecore_topk(
        s.reshape(bd, n), k=k, row_lengths=l))
    f_dec2 = jax.jit(lambda s, l: inc.sparsecore_topk(s, k=k, row_lengths=l))

    def oracle_check(sc_np, lengths, got):
        """rows_wrong, out_of_range, live-count mismatch, against a per-row
        numpy top-k restricted to [0, eff_len)."""
        bb, nn = sc_np.shape
        wrong = oor = nlive_bad = 0
        for r in range(bb):
            L = int(min(lengths[r], nn))
            g = got[r]
            live = g[g >= 0]
            want = min(k, max(L, 0))
            if live.size != want:
                nlive_bad += 1
            if live.size and (live.min() < 0 or live.max() >= max(L, 1)):
                oor += 1
                continue
            if L <= 0:
                continue
            row = sc_np[r, :L]
            if L <= k:
                ok = np.array_equal(np.sort(live), np.arange(L))
            else:
                part = np.argpartition(-row, k - 1)[:k]
                ok = np.array_equal(np.sort(row[part]),
                                    np.sort(row[live]))
            if not ok:
                wrong += 1
        return wrong, oor, nlive_bad

    gate = {}
    gate_ok = True
    checks = ([("P", key, sc_pre, f_pre3, dev["pre/3d"]) for key in lens_pre]
              + [("D", key, sc_dec, f_dec3, dev["dec/3d"]) for key in lens_dec])
    for tag, key, sc_np, fn, sarr in checks:
        nm = f"{tag}.{key}"
        larr = np.asarray(dev_len[nm])
        got = np.asarray(jax.block_until_ready(fn(sarr, dev_len[nm])))
        w, o, nb = oracle_check(sc_np, larr, got)
        live = int((got >= 0).sum())
        gate[nm] = {"rows_wrong": w, "rows_out_of_range": o,
                    "rows_live_count_wrong": nb, "total_live_slots": live}
        bad = (w or o or nb)
        gate_ok &= not bad
        print(f"  {nm:18s} rows_wrong={w:5d}  out_of_range={o:5d}  "
              f"live_count_wrong={nb:5d}  live_slots={live:8d}  "
              f"{'EXACT' if not bad else '!! FAIL'}", flush=True)

    # Negative control on the checker: score the rho=1 output against the
    # `prefill` lengths. If `row_lengths` did NOT reach the kernel every arm
    # above would be byte-identical and this MUST still go red.
    got_r1 = np.asarray(jax.block_until_ready(f_pre3(dev["pre/3d"],
                                                     dev_len["P.rho1"])))
    got_pf = np.asarray(jax.block_until_ready(f_pre3(dev["pre/3d"],
                                                     dev_len["P.prefill"])))
    nw, no, nnb = oracle_check(sc_pre, np.asarray(dev_len["P.prefill"]), got_r1)
    ident = bool(np.array_equal(got_r1, got_pf))
    print(f"\n  NEGCTL  rho=1 output scored against the `prefill` lengths -> "
          f"rows_wrong={nw} out_of_range={no} live_count_wrong={nnb}", flush=True)
    print(f"  NEGCTL  rho=1 output identical to `prefill` output? {ident}",
          flush=True)
    if (nw + no + nnb) == 0 or ident:
        print("  !! LENGTHS_GATE_FAIL: the checker cannot distinguish the two "
              "length vectors, so `EXACT` above proves nothing about whether "
              "row_lengths reached the kernel.", file=sys.stderr, flush=True)
        gate_ok = False
    gate["_negctl"] = {"rows_wrong": nw, "out_of_range": no,
                       "live_count_wrong": nnb, "outputs_identical": ident}
    results["lengths_gate"] = gate
    if not gate_ok:
        results["gate"] = "LENGTHS_GATE_FAIL"
        with open(os.path.join(args.outdir, "rho_denominator.json"), "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True, default=str)
        print("ABORTING before the timing stage.", file=sys.stderr, flush=True)
        return 2
    print("  LENGTHS_GATE_OK  (every arm exact AND the checker can go red)",
          flush=True)
    results["gate"] = "LENGTHS_GATE_OK"

    # ---- MODELLED: what the kernel's own control flow does at each rho -----
    print("\n  --- MODELLED host replay of the kernel's control flow "
          "(explains the curve; is NOT the curve) ---", flush=True)
    rep = {}
    for nm in ("P.rho1", "P.prefillX", "P.gatedX", "P.flat0.400",
               "P.bimodal0.400", "P.chunk1", "D.rho1", "D.decodeX"):
        if nm not in dev_len:
            continue
        sc_np = sc_pre if nm.startswith("P.") else sc_dec
        rep[nm] = replay_levels(sc_np, np.asarray(dev_len[nm]), k)
        print(f"  {nm:18s} mean_refinement_levels={rep[nm]['mean_levels']:.3f}"
              f"  keep_all={rep[nm]['keep_all_frac']*100:5.1f}%"
              f"  (n={rep[nm]['sampled_rows']} rows sampled)", flush=True)
    results["replay_modelled"] = rep

    # =====================================================================
    # STAGE 2 -- ARMS
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 2  ARMS", flush=True)
    print("=" * 78, flush=True)

    measure, last_order, harness_kind, dfloor = _load_harness(
        list(args.harness_dir) + [
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/microbench", "/app/microbench", os.getcwd(),
        ])
    results["harness"] = harness_kind

    thunks: dict[str, object] = {}
    arm_meta: dict[str, dict] = {}

    def reg(name, fn, sarr, lname, entry, b, scores_tag):
        thunks[name] = (lambda _f=fn, _s=sarr, _l=dev_len[lname]: _f(_s, _l))
        arm_meta[name] = {"b": b, "entry": entry, "lengths": lname,
                          "scores": scores_tag,
                          "rho": len_meta[lname]["rho"]}

    # -- prefill, production (3-D) entry: every length vector ---------------
    for key in lens_pre:
        reg(f"P3.{key}", f_pre3, dev["pre/3d"], f"P.{key}", "3d", bp, "prod")
    # -- prefill, harness (2-D) entry: the three that answer the 1.219 ------
    for key in ("rho1", "prefillX", "gatedX"):
        reg(f"P2.{key}", f_pre2, dev["pre/2d"], f"P.{key}", "2d", bp, "prod")
    # -- the cross-job control against probes/09's B (standard normal, 2-D) --
    for key in ("rho1", "prefillX"):
        reg(f"N2.{key}", f_pre2, dev["norm/2d"], f"P.{key}", "2d", bp, "normal")
    # -- decode --------------------------------------------------------------
    for key in lens_dec:
        reg(f"D3.{key}", f_dec3, dev["dec/3d"], f"D.{key}", "3d", bd, "prod")
    for key in ("rho1", "decodeX"):
        reg(f"D2.{key}", f_dec2, dev["dec/2d"], f"D.{key}", "2d", bd, "prod")

    # -- nulls: IDENTICAL SOURCE, separately staged, identical operands ------
    null_mods = {}
    for j, tg in enumerate("abc"):
        m3 = build_mod(base_src, f"nullpre_{tg}")
        null_mods[f"#nsrc_{tg}"] = m3
        g = jax.jit(lambda s, l, _m=m3: _m.sparsecore_topk(
            s.reshape(bp, n), k=k, row_lengths=l))
        reg(f"#nsrc_{tg}", g, dev["pre/3d"], "P.rho1", "3d", bp, "prod")
        m4 = build_mod(base_src, f"nulldec_{tg}")
        null_mods[f"#dsrc_{tg}"] = m4
        g4 = jax.jit(lambda s, l, _m=m4: _m.sparsecore_topk(
            s.reshape(bd, n), k=k, row_lengths=l))
        reg(f"#dsrc_{tg}", g4, dev["dec/3d"], "D.rho1", "3d", bd, "prod")
    # -- data nulls: ONE executable, three separate score draws --------------
    for j, tg in enumerate("abc"):
        reg(f"#ndat_{tg}", f_pre3, dev[f"null{j}/3d"], "P.rho1", "3d", bp,
            f"prod_draw{j}")

    for nm, mt in arm_meta.items():
        print(f"  arm {nm:18s} b={mt['b']:5d} entry={mt['entry']} "
              f"lengths={mt['lengths']:14s} rho={mt['rho']:.5f} "
              f"scores={mt['scores']}", flush=True)
    print(f"  {len(thunks)} arms, {len(null_mods) + 1} distinct module objects",
          flush=True)
    results["arms"] = arm_meta

    # =====================================================================
    # STAGE 3 -- TIMING
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print(f"STAGE 3  TIMING -- {len(thunks)} arms, ONE pool, "
          f"{args.rounds} interleaved rounds, reshuffled per round, "
          f"paired REPS {REPS_LO}/{REPS_HI} slope, read us_median", flush=True)
    print("=" * 78, flush=True)
    fl = dfloor()
    print(f"  dispatch floor: {fl:.2f} us/call", flush=True)
    results["dispatch_floor_us"] = fl

    out = measure(thunks, rounds=args.rounds)
    for name in thunks:
        r = out[name]
        print(f"  {name:18s} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  us_mean={r['us_mean']:10.3f}  "
              f"us_unpaired={r['us_unpaired']:10.3f}  "
              f"hostbound={r['hostbound_pct']:5.1f}%  "
              f"suspect={r.get('suspect')}", flush=True)
    results["raw"] = out
    results["order"] = last_order()
    m = {kk: out[kk]["us_median"] for kk in out}

    # =====================================================================
    # STAGE 4 -- THE FLOOR
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 4  RESOLUTION FLOOR", flush=True)
    print("=" * 78, flush=True)
    nsrc = [m[f"#nsrc_{t}"] for t in "abc"]
    dsrc = [m[f"#dsrc_{t}"] for t in "abc"]
    ndat = [m[f"#ndat_{t}"] for t in "abc"]
    fp, fd, fdat = _spread(nsrc), _spread(dsrc), _spread(ndat)
    ref_pre = m["P3.rho1"]
    ref_dec = m["D3.rho1"]
    print(f"  #nsrc (prefill, identical source, identical operands) "
          f"{nsrc[0]:.3f} / {nsrc[1]:.3f} / {nsrc[2]:.3f}", flush=True)
    print(f"    -> PREFILL RESOLUTION FLOOR = {fp:.3f} us "
          f"= {100.0 * fp / ref_pre:.3f}% of P3.rho1", flush=True)
    print(f"  #dsrc (decode,  identical source, identical operands) "
          f"{dsrc[0]:.3f} / {dsrc[1]:.3f} / {dsrc[2]:.3f}", flush=True)
    print(f"    -> DECODE  RESOLUTION FLOOR = {fd:.3f} us "
          f"= {100.0 * fd / ref_dec:.3f}% of D3.rho1", flush=True)
    print(f"  #ndat (prefill, SAME executable, three different score draws) "
          f"{ndat[0]:.3f} / {ndat[1]:.3f} / {ndat[2]:.3f}", flush=True)
    print(f"    -> score-draw spread = {fdat:.3f} us "
          f"= {100.0 * fdat / ref_pre:.3f}% of P3.rho1. Every rho arm shares "
          f"ONE draw, so this term does not enter a rho contrast; it bounds "
          f"how much a different draw could move an absolute.", flush=True)
    results["floor"] = {"prefill_us": fp, "decode_us": fd,
                        "score_draw_spread_us": fdat,
                        "nsrc": nsrc, "dsrc": dsrc, "ndat": ndat,
                        "prefill_pct": 100.0 * fp / ref_pre,
                        "decode_pct": 100.0 * fd / ref_dec}

    def resolvable(delta, b):
        f = fp if b == "P" else fd
        return "resolvable" if abs(delta) > f else "NOT RESOLVABLE"

    # =====================================================================
    # STAGE 5 -- THE rho CURVE
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 5  THE rho CURVE (absolutes first; percentages are derived "
          "from them in one place)", flush=True)
    print("=" * 78, flush=True)
    curve = {}
    for pfx, ref, fam in (("P3", ref_pre, "P"), ("D3", ref_dec, "D")):
        rows = sorted(((arm_meta[nm]["rho"], nm) for nm in arm_meta
                       if nm.startswith(pfx + ".")))
        print(f"\n  --- {pfx} (b={arm_meta[rows[0][1]]['b']}, 3-D entry) ---",
              flush=True)
        print(f"  {'arm':18s} {'rho':>8s} {'us_median':>11s} "
              f"{'vs rho=1 us':>12s} {'Delta time %':>13s}  note", flush=True)
        for rho, nm in rows:
            d = m[nm] - ref
            curve[nm] = {"rho": rho, "us_median": m[nm], "delta_us": d,
                         "delta_pct_vs_rho1": 100.0 * d / ref}
            print(f"  {nm:18s} {rho:8.5f} {m[nm]:11.3f} {d:12.3f} "
                  f"{100.0 * d / ref:+12.2f}%  {resolvable(d, fam)}", flush=True)
    results["curve"] = curve

    # -- the fit over the degenerate (flat) arms -----------------------------
    print("\n  --- fit T(rho) = a + s*rho over the DEGENERATE (flat) prefill "
          "arms. `a` is what rho cannot remove. ---", flush=True)
    flat = sorted((arm_meta[nm]["rho"], nm) for nm in arm_meta
                  if nm.startswith("P3.flat") or nm == "P3.rho1")
    xs = [r for r, _ in flat]
    ys = [m[nm] for _, nm in flat]
    a, s, r2 = _linfit(xs, ys)
    print("  points: " + "  ".join(f"{r:.3f}:{y:.2f}" for r, y in zip(xs, ys)),
          flush=True)
    print(f"  T(rho) = {a:.3f} + {s:.3f}*rho   r2={r2:.5f}   "
          f"rho-independent floor = {a:.3f} us = "
          f"{100.0 * a / (a + s):.1f}% of T(1)", flush=True)
    mono = all(ys[i] <= ys[i + 1] + fp for i in range(len(ys) - 1))
    print(f"  monotone in rho (within the {fp:.3f} us floor)? {mono}", flush=True)
    # residuals say whether a LINE is even the right shape
    resid = [y - (a + s * x) for x, y in zip(xs, ys)]
    print("  residuals vs the line: "
          + "  ".join(f"{r:+.2f}" for r in resid)
          + f"   max|resid| = {max(abs(r) for r in resid):.3f} us "
          f"({'inside' if max(abs(r) for r in resid) <= fp else 'OUTSIDE'} "
          f"the floor)", flush=True)
    results["fit_flat_prefill"] = {"a_us": a, "s_us": s, "r2": r2,
                                   "points": dict(zip(map(str, xs), ys)),
                                   "residuals": resid, "monotone": mono}

    # -- flat vs the causal chunk ramps --------------------------------------
    print("\n  --- the eight PRODUCTION per-chunk causal ramps, and the "
          "degenerate arm nearest each ---", flush=True)
    chunks = {}
    for c in range(8):
        nm = f"P3.chunk{c}"
        rho = arm_meta[nm]["rho"]
        near = min((abs(arm_meta[o]["rho"] - rho), o) for o in arm_meta
                   if o.startswith("P3.flat") or o == "P3.rho1")[1]
        d = m[nm] - m[near]
        chunks[c] = {"arm": nm, "rho": rho, "us_median": m[nm],
                     "nearest_flat": near, "nearest_flat_rho":
                     arm_meta[near]["rho"], "nearest_flat_us": m[near],
                     "delta_us": d}
        print(f"  chunk{c}  rho={rho:.5f}  {m[nm]:9.3f} us   vs {near:14s} "
              f"(rho={arm_meta[near]['rho']:.3f}) {m[near]:9.3f}  "
              f"delta={d:+8.3f}  {resolvable(d, 'P')}", flush=True)
    # The correctly-gated production denominator: chunks 2..7 only, equal weight.
    gated_chunks = [m[f"P3.chunk{c}"] for c in range(2, 8)]
    all_chunks = [m[f"P3.chunk{c}"] for c in range(8)]
    t_gated_w = sum(gated_chunks) / len(gated_chunks)
    t_all_w = sum(all_chunks) / len(all_chunks)
    print(f"\n  chunk-weighted mean over ALL 8 chunks          = {t_all_w:.3f} us"
          f"   (rho 0.40005 unconditionally)", flush=True)
    print(f"  chunk-weighted mean over the 6 that RUN (c2..c7) = "
          f"{t_gated_w:.3f} us   (rho 0.50005, the lax.cond survivors)",
          flush=True)
    print(f"  the single-vector `P3.gated` arm (Uniform 2049..8192) = "
          f"{m['P3.gated']:.3f} us   delta vs the chunk-weighted "
          f"{m['P3.gated'] - t_gated_w:+.3f} us "
          f"{resolvable(m['P3.gated'] - t_gated_w, 'P')}", flush=True)
    results["chunks"] = {"per_chunk": chunks,
                         "weighted_all8_us": t_all_w,
                         "weighted_gated_c2to7_us": t_gated_w,
                         "gated_single_vector_us": m["P3.gated"]}

    # -- distribution sensitivity at a FIXED mean ----------------------------
    print("\n  --- DOES THE DISTRIBUTION MATTER, at E[eff_len] = 4096 "
          "(rho = 0.400)? Three distributions, one mean. ---", flush=True)
    dist = {}
    base_d = m["P3.flat0.400"]
    for nm, lbl in (("P3.flat0.400", "degenerate 4096"),
                    ("P3.prefillX", "Uniform{1..8192} shuffled (PRODUCTION)"),
                    ("P3.prefillXsorted", "same multiset, sorted"),
                    ("P3.bimodal0.400", "bimodal {2048,6144}"),
                    ("P3.prefill", "i.i.d. draw, rho off by sampling error")):
        d = m[nm] - base_d
        dist[nm] = {"label": lbl, "us_median": m[nm], "delta_vs_flat_us": d,
                    "delta_vs_flat_pct": 100.0 * d / base_d,
                    "rho": arm_meta[nm]["rho"]}
        print(f"  {nm:18s} rho={arm_meta[nm]['rho']:.5f} {m[nm]:9.3f} us  "
              f"{lbl:40s} vs degenerate {d:+8.3f} us = "
              f"{100.0 * d / base_d:+6.2f}%  {resolvable(d, 'P')}", flush=True)
    d_arr = m["P3.prefillX"] - m["P3.prefillXsorted"]
    print(f"  ARRANGEMENT at an IDENTICAL multiset (shuffled - sorted) = "
          f"{d_arr:+.3f} us = {100.0 * d_arr / m['P3.prefillXsorted']:+.2f}%  "
          f"{resolvable(d_arr, 'P')}. Rows map to subcores with stride 32 and "
          f"a wave costs its slowest subcore, so this is the within-wave "
          f"length imbalance, which rho cannot see.", flush=True)
    dist["_arrangement_us"] = d_arr
    spread_d = _spread([m["P3.flat0.400"], m["P3.prefillX"],
                        m["P3.bimodal0.400"]])
    print(f"  -> the SHAPE of the distribution at a fixed mean is worth "
          f"{spread_d:.3f} us ({100.0 * spread_d / base_d:.2f}%), against a "
          f"{fp:.3f} us floor. A mean-only rho model is "
          f"{'ADEQUATE' if spread_d <= fp else 'NOT adequate'}.", flush=True)
    results["distribution_at_fixed_mean"] = {
        "arms": dist, "spread_us": spread_d, "floor_us": fp,
        "mean_only_model_adequate": bool(spread_d <= fp)}

    # =====================================================================
    # STAGE 6 -- THE ADJUDICATION
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 6  ADJUDICATION -- 189.98 / 155.82 / 1.219", flush=True)
    print("=" * 78, flush=True)

    adj = {}
    # (a) the ENTRY factor, measured directly, within the pool, at both rho.
    print("\n  (a) THE 1.219: harness (2-D) / production (3-D) entry, same "
          "scores, same lengths, same pool", flush=True)
    for key in ("rho1", "prefillX", "gatedX"):
        t2, t3 = m[f"P2.{key}"], m[f"P3.{key}"]
        fac = t2 / t3
        adj[f"entry_factor_{key}"] = {"harness_2d_us": t2,
                                      "production_3d_us": t3,
                                      "factor": fac,
                                      "sheet": SHEET_ENTRY_FACTOR,
                                      "delta_pct": 100.0 * (fac - SHEET_ENTRY_FACTOR)
                                      / SHEET_ENTRY_FACTOR,
                                      "relayout_us": t2 - t3}
        print(f"      {key:8s} 2-D {t2:9.3f}  3-D {t3:9.3f}  "
              f"relayout {t2 - t3:+8.3f} us  factor {fac:.4f}  "
              f"vs sheet {SHEET_ENTRY_FACTOR:.3f} -> "
              f"{100.0 * (fac - SHEET_ENTRY_FACTOR) / SHEET_ENTRY_FACTOR:+.2f}%",
              flush=True)

    # (b) the rho factor, measured directly.
    print("\n  (b) THE rho REWEIGHTING: T(rho=1) / T(rho=0.40005), which the "
          f"sheet implies to be {SHEET_B_RHO1:.3f}/{SHEET_HARNESS:.2f} = "
          f"{SHEET_RHO_FACTOR:.4f}", flush=True)
    for pfx, key1, key2, lbl in (("P2", "rho1", "prefillX", "harness 2-D entry"),
                                 ("P3", "rho1", "prefillX", "production 3-D entry"),
                                 ("N2", "rho1", "prefillX",
                                  "2-D entry, standard-normal scores "
                                  "(probes/09's own operand)")):
        t1, t2 = m[f"{pfx}.{key1}"], m[f"{pfx}.{key2}"]
        fac = t1 / t2
        adj[f"rho_factor_{pfx}"] = {"T_rho1_us": t1, "T_prod_us": t2,
                                    "factor": fac, "sheet": SHEET_RHO_FACTOR,
                                    "delta_pct": 100.0 * (fac - SHEET_RHO_FACTOR)
                                    / SHEET_RHO_FACTOR, "label": lbl}
        print(f"      {pfx}  T(1)={t1:9.3f}  T(0.40005)={t2:9.3f}  "
              f"factor {fac:.4f}  vs {SHEET_RHO_FACTOR:.4f} -> "
              f"{100.0 * (fac - SHEET_RHO_FACTOR) / SHEET_RHO_FACTOR:+.2f}%   "
              f"({lbl})", flush=True)

    # (c) the absolutes, two ways: raw, and re-anchored on the sheet's own
    #     rho=1 measurement (which removes any cross-pool level difference).
    print("\n  (c) THE ABSOLUTES. Raw, and RE-ANCHORED -- the sheet's own "
          f"rho=1 reading was {SHEET_B_RHO1:.3f} us (probes/09, 2-D entry, "
          "standard-normal scores); re-anchoring applies THIS job's measured "
          "rho ratio to THAT number, which cancels any cross-pool level "
          "difference between the two jobs.", flush=True)
    for pfx, sheet_val, lbl in (("P2", SHEET_HARNESS, "harness 189.98"),
                                ("N2", SHEET_HARNESS, "harness 189.98"),
                                ("P3", SHEET_PRODUCTION, "production 155.82")):
        t1, t2 = m[f"{pfx}.rho1"], m[f"{pfx}.prefillX"]
        anchor = SHEET_B_RHO1 if pfx.endswith("2") else (
            SHEET_B_RHO1 / SHEET_ENTRY_FACTOR)
        rean = anchor * (t2 / t1)
        adj[f"absolute_{pfx}"] = {
            "measured_rho1_us": t1, "measured_prod_rho_us": t2,
            "sheet_us": sheet_val, "raw_delta_pct":
            100.0 * (t2 - sheet_val) / sheet_val,
            "reanchor_base_us": anchor, "reanchored_us": rean,
            "reanchored_delta_pct": 100.0 * (rean - sheet_val) / sheet_val,
            "label": lbl}
        print(f"      {pfx}  measured T(0.40005) = {t2:9.3f} us  vs sheet "
              f"{sheet_val:7.2f} -> raw "
              f"{100.0 * (t2 - sheet_val) / sheet_val:+7.2f}%   |   "
              f"re-anchored {rean:9.3f} -> "
              f"{100.0 * (rean - sheet_val) / sheet_val:+7.2f}%   ({lbl})",
              flush=True)

    # (d) the cross-job control: does N2.rho1 reproduce probes/09's B?
    dB = 100.0 * (m["N2.rho1"] - SHEET_B_RHO1) / SHEET_B_RHO1
    print(f"\n  (d) CROSS-JOB CONTROL. N2.rho1 (2-D entry, standard-normal "
          f"scores, rho=1) = {m['N2.rho1']:.3f} us against probes/09's "
          f"B = {SHEET_B_RHO1:.3f} -> {dB:+.2f}%. This is the SAME "
          f"construction in a different pool; it says how much of any "
          f"disagreement above is pool level rather than rho.", flush=True)
    adj["crossjob_B"] = {"measured_us": m["N2.rho1"], "probe09_B_us":
                         SHEET_B_RHO1, "delta_pct": dB}
    # and what the production score distribution is worth against normal
    dsc = m["P2.rho1"] - m["N2.rho1"]
    print(f"      score distribution: production {m['P2.rho1']:.3f} vs "
          f"standard-normal {m['N2.rho1']:.3f} = {dsc:+.3f} us "
          f"({100.0 * dsc / m['N2.rho1']:+.2f}%)  {resolvable(dsc, 'P')}",
          flush=True)
    adj["score_distribution_us"] = dsc

    # (e) decode
    t1d, t2d = m["D3.rho1"], m["D3.decodeX"]
    print(f"\n  (e) DECODE. T(rho=1) = {t1d:.3f}, T(rho=0.850) = {t2d:.3f}, "
          f"factor {t1d / t2d:.4f}, delta {t2d - t1d:+.3f} us = "
          f"{100.0 * (t2d - t1d) / t1d:+.2f}%  "
          f"{resolvable(t2d - t1d, 'D')}", flush=True)
    adj["decode"] = {"T_rho1_us": t1d, "T_prod_us": t2d,
                     "factor": t1d / t2d,
                     "delta_pct": 100.0 * (t2d - t1d) / t1d,
                     "floor_us": fd}
    results["adjudication"] = adj

    path = os.path.join(args.outdir, "rho_denominator.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {path}", flush=True)
    return 0


# ==========================================================================
# host-side gate -- runs with no TPU, and gates the real run
# ==========================================================================
def _dry_run(args) -> int:
    path = args.subject
    raw = open(path, "rb").read()
    print(f"subject: {path}  md5={_md5(raw)}")
    if _md5(raw) != SUBJECT_MD5:
        print(f"SUBJECT_GATE_FAIL: {_md5(raw)} != {SUBJECT_MD5}")
        return 1
    print(f"SUBJECT_GATE_OK  {SUBJECT_MD5}")

    src = raw.decode()
    # The subject must be UNPATCHED -- this probe edits no source at all. Assert
    # the three constants the arms depend on are the shipped ones.
    for anchor in ("NUM_BUCKETS = 256", "L0_SHIFT = 24",
                   "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))",
                   "i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, "
                   "eff_len), 0)",
                   "dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)"):
        c = src.count(anchor)
        print(f"  anchor count={c} {'OK' if c == 1 else 'WRONG'}  {anchor[:60]}")
        if c != 1:
            return 1
    print("ANCHOR_GATE_OK  (the DMA is static-length and i_hi is rho-bounded: "
          "the two facts the whole reweighting rests on)")

    if args.level_sim:
        ls = open(args.level_sim).read()
        miss = []
        import inspect
        for fn in (level_schedule, _quant_fp8_rowwise, make_kv, make_scores,
                   prefill_lengths, decode_lengths):
            if inspect.getsource(fn) not in ls:
                miss.append(fn.__name__)
        if miss:
            print(f"VERBATIM_GATE_FAIL: {miss} differ from {args.level_sim}")
            return 1
        print(f"VERBATIM_GATE_OK  level_schedule/_quant_fp8_rowwise/make_kv/"
              f"make_scores/prefill_lengths/decode_lengths are byte-identical "
              f"to {args.level_sim}")
    else:
        print("VERBATIM_GATE_SKIPPED (--level-sim not given)")

    n = ((args.ctx + 1024) + 15) // 16 * 16
    if n != 10240:
        print(f"CTX_GATE_FAIL: --ctx {args.ctx} gives n={n}, not 10240")
        return 1
    print(f"CTX_GATE_OK  --ctx {args.ctx} -> n={n}")

    for b, tag in ((args.b_prefill, "prefill"), (args.b_decode, "decode")):
        tbl = build_lengths(b, n, args.k)
        print(f"  --- {tag} b={b}: {len(tbl)} length vectors ---")
        for key, (arr, prov) in tbl.items():
            eff = np.minimum(arr.astype(np.int64), n)
            print(f"    {key:14s} rho={eff.mean() / n:.5f} "
                  f"eff in [{eff.min()},{eff.max()}] "
                  f"P(eff<=k)={float((eff <= args.k).mean()) * 100:5.1f}% "
                  f"md5={_md5(arr.tobytes())[:16]}  {prov}")
    # the two rho values the sheet quotes must be reproduced EXACTLY
    n_ = 10240
    pl_ = prefill_lengths(1 << 20, np.random.default_rng(1))
    dl_ = decode_lengths(1 << 20, np.random.default_rng(1))
    print(f"  rho check: prefill_lengths exact mean "
          f"{(1 + 8192) / 2 / n_:.5f} (target {RHO_PRE}), "
          f"1M-sample {pl_.mean() / n_:.5f}")
    print(f"  rho check: decode_lengths  exact mean "
          f"{(8193 + 9215) / 2 / n_:.5f} (target {RHO_DEC}), "
          f"1M-sample {dl_.mean() / n_:.5f}")
    assert abs((1 + 8192) / 2 / n_ - RHO_PRE) < 2e-6
    assert abs((8193 + 9215) / 2 / n_ - RHO_DEC) < 2e-6
    assert abs((2049 + 8192) / 2 / n_ - RHO_PRE_GATED) < 2e-6
    print(f"  rho check: gated (Uniform 2049..8192) exact mean "
          f"{(2049 + 8192) / 2 / n_:.5f} (target {RHO_PRE_GATED})")
    # The stratified arms must hit the population mean to the last bit.
    for lo, hi, bb, tgt, lbl in ((1, 8192, args.b_prefill, RHO_PRE, "prefillX"),
                                 (2049, 8192, args.b_prefill, RHO_PRE_GATED,
                                  "gatedX"),
                                 (8193, 9215, args.b_decode, RHO_DEC,
                                  "decodeX")):
        v = strat_uniform(lo, hi, bb, SEED_LEN)
        r = float(v.mean()) / n_
        print(f"  strat check: {lbl:9s} rho={r:.7f} target {tgt}  "
              f"err={abs(r - tgt):.2e}  support [{v.min()},{v.max()}] "
              f"subset of [{lo},{hi}]")
        assert abs(r - tgt) < 2e-6, (lbl, r, tgt)
        assert v.min() >= lo and v.max() <= hi
    print("RHO_GATE_OK")
    print("DRY_RUN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
