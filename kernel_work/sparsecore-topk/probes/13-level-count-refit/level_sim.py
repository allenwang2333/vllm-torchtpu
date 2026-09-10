"""Probe 13 -- re-fit the sparsecore_topk refinement level count at the
PRODUCTION score distribution.

HOST ONLY.  numpy + ml_dtypes.  No TPU, no CDK job.  `done` is a deterministic
function of the score bits (`sparsecore_topk.py:209-265`), so the trip count is
obtainable by replaying the kernel's own integer arithmetic.

Three pieces:

1. `levels_for_row` -- a faithful replay of `sparsecore_topk.py:183-265`,
   generalised over NUM_BUCKETS so `v5` can be swept.  Validated against
   `probes/05-refinement-count/probe.py` (identical numbers at NUM_BUCKETS=256).
2. `production_scores` -- the DSA indexer score row built the way
   `streamindex_topk.py:326-338` builds it: fp8 q . fp8 k, ReLU, per-head
   SIGNED weight, sum over heads, times the per-KV-column ue8m0 dequant scale.
3. the harness that walks (1) over (2) at production shapes and rho.

    uv_venv/bin/python level_sim.py
"""

import math
import sys
import time

import ml_dtypes
import numpy as np

# ---------------------------------------------------------------- kernel ----
# Mirrors sparsecore_topk.py:41-51.
LANES = 16
MONO_MASK = np.int32(0x7FFFFFFF)
BITS_NEG_INF = np.float32(-np.inf).view(np.int32)
KEY_NEG_INF = np.int32(BITS_NEG_INF ^ MONO_MASK)
NUM_SUBCORES_TOTAL = 32  # num_sc_cores(2) * num_subcores(16), :94-98


def monotone_key(bits):
    """sparsecore_topk.py:116-118.  Involution, order-preserving on fp32."""
    m = np.right_shift(bits, 31) & MONO_MASK
    return np.bitwise_xor(bits, m).astype(np.int32)


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


def levels_for_row(scores_row, k, eff_len, num_buckets=256):
    """Replay sparsecore_topk.py:183-265 for one row.

    Returns (levels_run, keep_all, unconverged, l0_cnt) where `unconverged`
    means the walk exhausted every refinement level with `done` still False --
    the row then relies on the arbitrary `tie_rank` tie-break at :283-291 --
    and `l0_cnt` is the L0 pivot-bucket occupancy, i.e. `v3`'s straddler count.
    """
    l0_shift, refinement = level_schedule(num_buckets)
    half = num_buckets // 2

    # `i_hi = min(slice_len, eff_len)` (:164) then `valid = (j < i_hi) &
    # (key > KEY_NEG_INF)` (:180).
    keys = monotone_key(scores_row[:eff_len].view(np.int32))
    keys = keys[keys > KEY_NEG_INF]

    hist = np.bincount(np.right_shift(keys, l0_shift) + half,
                       minlength=num_buckets)
    cum = np.cumsum(hist)
    total = int(cum[-1])

    keep_all = total <= k                                        # :209
    b_star = int(np.count_nonzero(cum <= total - k))             # :211
    cum_at = int(cum[b_star])
    cum_lo = int(cum[b_star - 1]) if b_star > 0 else 0
    cnt = cum_at - cum_lo                                        # :216
    quota = k - (total - cum_at)                                 # :217

    prefix = b_star - half                                       # :219
    shift = l0_shift
    done = keep_all or (quota == cnt)                            # :221
    l0_cnt = 0 if keep_all else cnt

    ran = 0
    for lvl_shift, lvl_mask in refinement:
        if done:
            continue                                             # :224-245 @0
        ran += 1
        match = np.right_shift(keys, shift) == prefix            # :235
        sub = np.bincount(np.right_shift(keys[match], lvl_shift) & lvl_mask,
                          minlength=num_buckets)
        cum2 = np.cumsum(sub)
        sub_total = int(cum2[-1])
        b2 = int(np.count_nonzero(cum2 <= sub_total - quota))    # :248
        cum_at2 = int(cum2[b2])
        cum_lo2 = int(cum2[b2 - 1]) if b2 > 0 else 0
        cnt2 = cum_at2 - cum_lo2
        quota2 = quota - (sub_total - cum_at2)
        prefix = (prefix << (shift - lvl_shift)) | b2            # :257
        shift = lvl_shift
        quota, cnt = quota2, cnt2
        done = (quota2 == cnt2) or (quota2 == 0)                 # :258
    return ran, keep_all, (not done), l0_cnt


def _crit(x, waves):
    """max over subcores of the sum of ITS OWN rows, expressed per row.

    `_sc_topk_direct` runs `lax.fori_loop(0, num_waves, wave_body)` per subcore
    (sparsecore_topk.py:443) with `row = w * 32 + flat` (:157), and there is no
    inter-subcore barrier anywhere (:19, :528).  A subcore therefore runs its
    whole wave loop ahead of a slower neighbour; the kernel ends when the
    slowest SUBCORE ends, i.e. max-over-subcores-of-sum, NOT sum-over-waves-of
    -max.  `_wavemax` below computes the latter as a strict upper bound.
    """
    return max(x[f::NUM_SUBCORES_TOTAL].sum()
               for f in range(NUM_SUBCORES_TOTAL)) / waves


def _wavemax(x, waves):
    """sum over waves of the max within the wave, per row.  Upper bound: it is
    the cost IF every wave boundary were a barrier.  It is not one."""
    return sum(x[w * NUM_SUBCORES_TOTAL:(w + 1) * NUM_SUBCORES_TOTAL].max()
               for w in range(waves)) / waves


def walk_stats(levels, keep_all, unconv, max_levels, l0cnt=None):
    b = len(levels)
    lv = np.asarray(levels, np.float64)
    ka = np.asarray(keep_all, bool)
    walking = ~ka
    per_sc = [lv[f::NUM_SUBCORES_TOTAL].sum() for f in range(NUM_SUBCORES_TOTAL)]
    waves = max(len(lv[f::NUM_SUBCORES_TOTAL]) for f in range(NUM_SUBCORES_TOTAL))
    hist = [int((lv == i).sum()) for i in range(max_levels + 1)]
    c = np.asarray(l0cnt if l0cnt is not None else [0] * b, np.float64)
    prod = lv * c   # v3's per-row refinement work: levels x straddler count
    return dict(
        cnt_mean=c.mean(),
        cnt_crit=_crit(c, waves),
        cnt_wavemax=_wavemax(c, waves),
        prod_mean=prod.mean(),
        prod_crit=_crit(prod, waves),
        prod_wavemax=_wavemax(prod, waves),
        lv_wavemax=_wavemax(lv, waves),
        cnt_p99=float(np.percentile(c, 99)),
        cnt_max=float(c.max()),
        rows=b,
        mean_all=lv.mean(),
        se_all=lv.std(ddof=1) / math.sqrt(b),
        mean_walking=lv[walking].mean() if walking.any() else 0.0,
        se_walking=(lv[walking].std(ddof=1) / math.sqrt(int(walking.sum()))
                    if walking.sum() > 1 else 0.0),
        n_walking=int(walking.sum()),
        keep_all_frac=float(ka.mean()),
        crit=max(per_sc) / waves,
        waves=waves,
        hist=hist,
        unconverged=int(np.asarray(unconv).sum()),
    )


# ----------------------------------------------------- score generators -----
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


# ---------------------------------------------------------------- probe -----
def run_case(scores, lengths, k, num_buckets=256):
    max_levels = len(level_schedule(num_buckets)[1])
    lv, ka, un, cn = [], [], [], []
    for r in range(scores.shape[0]):
        a, c, d, e = levels_for_row(scores[r], k, int(lengths[r]), num_buckets)
        lv.append(a)
        ka.append(c)
        un.append(d and not c)
        cn.append(e)
    return walk_stats(lv, ka, un, max_levels, cn)


def fmt(tag, st):
    return (f"{tag:<34} rows={st['rows']:>5} mean_all={st['mean_all']:.3f}"
            f"+-{st['se_all']:.3f} mean_walk={st['mean_walking']:.3f}"
            f"+-{st['se_walking']:.3f} keep_all={st['keep_all_frac']*100:5.1f}%"
            f" crit={st['crit']:.3f} hist={st['hist']}"
            f" unconv={st['unconverged']} cnt(mean/p99/max)="
            f"{st['cnt_mean']:.0f}/{st['cnt_p99']:.0f}/{st['cnt_max']:.0f}")


K, N, HEAD_DIM, N_HEAD = 2048, 10240, 128, 64
KV_ROWS = 9216  # max eff_len at decode; columns past it are never valid
SEED = 20260910


def synth(dist, b, n, rng):
    """probes/05-refinement-count/probe.py:102-111, verbatim."""
    if dist == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if dist == "uniform":
        return rng.random((b, n), dtype=np.float32)
    if dist == "heavy_ties":
        vals = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
        return vals[rng.integers(0, 17, size=(b, n))]
    raise ValueError(dist)


CNT_MEAN = 0.0
V5_PREFILL = []
V5_T = {}
V3_IN = {}


def main():
    global CNT_MEAN
    t0 = time.time()
    print("=" * 100)
    print("STEP 3a  MODEL VALIDATION -- reproduce probes/05 table (rho=1, "
          "seed 20260904, NUM_BUCKETS=256)")
    print("=" * 100)
    print(f"{'dist':<12}{'b':>6}{'mean/row':>10}{'crit/row':>10}"
          f"{'probe05 mean':>14}{'probe05 crit':>14}")
    p05 = {("random", 64): (1.95, 2.50), ("random", 1024): (1.92, 2.03),
           ("uniform", 64): (2.05, 2.50), ("uniform", 1024): (2.06, 2.25),
           ("heavy_ties", 64): (3.00, 3.00), ("heavy_ties", 1024): (3.00, 3.00)}
    ok = True
    for dist in ("random", "uniform", "heavy_ties"):
        for b in (64, 1024):
            rng = np.random.default_rng(20260904)
            sc = synth(dist, b, N, rng)
            st = run_case(sc, np.full(b, N), K)
            ref = p05[(dist, b)]
            ok &= abs(st["mean_all"] - ref[0]) < 0.005 and abs(st["crit"] - ref[1]) < 0.005
            print(f"{dist:<12}{b:>6}{st['mean_all']:>10.2f}{st['crit']:>10.2f}"
                  f"{ref[0]:>14.2f}{ref[1]:>14.2f}")
    print(f"\nreproduces probes/05 exactly: {ok}")
    print("NOTE: 2.231 is NOT this table. It is probes/07's `implied active "
          "levels`\n      = (L3-L0)/level_slope, a HARDWARE TIMING fit at "
          "b=1024, n=10240, k=2048,\n      rho=1, scores=rng.standard_normal "
          "(probes/07/probe.py:269-291). Its simulated\n      companion is "
          "probe 05's crit/row = 2.03 (mean/row 1.92).")

    print()
    print("=" * 100)
    print("STEP 4  PRODUCTION DISTRIBUTION")
    print("=" * 100)
    print("score = kscale[j] * sum_h w[h] * relu(q_fp8[h] . k_fp8[j])   "
          "(streamindex_topk.py:326-338)")
    print(f"n_head={N_HEAD} head_dim={HEAD_DIM} (index_n_heads/index_head_dim, "
          "vllm deepseek_v2.py:664-665)")
    print()

    reps = 4
    agg = {}
    for phase, b, lenfn in (("prefill_b1024", 1024, prefill_lengths),
                            ("decode_b64", 64, decode_lengths),
                            ("decode_b1024", 1024, decode_lengths)):
        for tag, kwargs in (("alpha0.0", dict(alpha=0.0)),):
            per_rep, allst = [], []
            for rep in range(reps):
                rng = np.random.default_rng(SEED + 1000 * rep)
                kq, ks = make_kv(KV_ROWS, HEAD_DIM, kwargs["alpha"], rng)
                sc = np.full((b, N), -np.inf, np.float32)
                sc[:, :KV_ROWS] = make_scores(kq, ks, b, N_HEAD, HEAD_DIM, rng)
                lengths = lenfn(b, rng)
                st = run_case(sc, lengths, K)
                per_rep.append(st)
                allst.append(st["mean_all"])
                print(fmt(f"{phase} {tag} rep{rep}", st))
            m = np.array(allst)
            print(f"  -> {phase} {tag}: between-replicate mean_all "
                  f"{m.mean():.3f} +- {m.std(ddof=1)/math.sqrt(reps):.3f} "
                  f"(n={reps} independent KV draws)")
            mw = np.array([s["mean_walking"] for s in per_rep])
            print(f"  -> {phase} {tag}: between-replicate mean_walking "
                  f"{mw.mean():.3f} +- {mw.std(ddof=1)/math.sqrt(reps):.3f}")
            cr = np.array([s["crit"] for s in per_rep])
            print(f"  -> {phase} {tag}: between-replicate crit/row "
                  f"{cr.mean():.3f} +- {cr.std(ddof=1)/math.sqrt(reps):.3f}")
            hh = np.sum([s["hist"] for s in per_rep], axis=0)
            print(f"  -> {phase} {tag}: pooled levels histogram "
                  f"{dict(enumerate(hh.tolist()))}  total={hh.sum()}  "
                  f"unconverged={sum(s['unconverged'] for s in per_rep)}")
            cm = np.array([s["cnt_mean"] for s in per_rep])
            cc = np.array([s["cnt_crit"] for s in per_rep])
            cw = np.array([s["cnt_wavemax"] for s in per_rep])
            cx = max(s["cnt_max"] for s in per_rep)
            print(f"  -> {phase} {tag}: L0 pivot-bucket occupancy `cnt` "
                  f"(v3's straddler set): mean {cm.mean():.0f}  "
                  f"crit {cc.mean():.0f} (+-{cc.std(ddof=1)/math.sqrt(reps):.0f}, "
                  f"{cc.mean()/cm.mean():.3f}x mean)  wavemax {cw.mean():.0f}  "
                  f"max over {reps*b} rows {cx:.0f}")
            pm = np.array([s["prod_mean"] for s in per_rep])
            pc = np.array([s["prod_crit"] for s in per_rep])
            pw = np.array([s["prod_wavemax"] for s in per_rep])
            lw = np.array([s["lv_wavemax"] for s in per_rep])
            print(f"  -> {phase} {tag}: v3 work levels*cnt: mean {pm.mean():.0f}  "
                  f"crit {pc.mean():.0f}  wavemax {pw.mean():.0f}   "
                  f"=> cnt_eff(crit) = prod_crit/lvl_crit = "
                  f"{pc.mean()/cr.mean():.0f}   levels wavemax {lw.mean():.3f}")
            if phase == "prefill_b1024":
                CNT_MEAN = float(cm.mean())
                V3_IN["prefill"] = dict(
                    lvl_mean=float(m.mean()), lvl_walk=float(mw.mean()),
                    lvl_crit=float(cr.mean()), lvl_wavemax=float(lw.mean()),
                    cnt_mean=float(cm.mean()), cnt_crit=float(cc.mean()),
                    cnt_wavemax=float(cw.mean()), cnt_max=float(cx),
                    prod_mean=float(pm.mean()), prod_crit=float(pc.mean()),
                    prod_wavemax=float(pw.mean()),
                    prod_crit_se=float(pc.std(ddof=1) / math.sqrt(reps)))
            agg[(phase, tag)] = per_rep
            print()

    # rho=1 apples-to-apples with 2.03 / 2.231
    print("-" * 100)
    print("STEP 4b  production distribution at rho=1 (apples-to-apples with "
          "probe 05's 2.03 and probe 07's 2.231)")
    for phase, b in (("prefill", 1024), ("decode", 64)):
        ms = []
        for rep in range(reps):
            rng = np.random.default_rng(SEED + 1000 * rep)
            kq, ks = make_kv(N, HEAD_DIM, 0.0, rng)
            sc = make_scores(kq, ks, b, N_HEAD, HEAD_DIM, rng)
            st = run_case(sc, np.full(b, N), K)
            ms.append(st)
            print(fmt(f"{phase} rho=1 rep{rep}", st))
        m = np.array([s["mean_all"] for s in ms])
        c = np.array([s["crit"] for s in ms])
        print(f"  -> {phase} rho=1: mean {m.mean():.3f}+-"
              f"{m.std(ddof=1)/math.sqrt(reps):.3f}  crit {c.mean():.3f}+-"
              f"{c.std(ddof=1)/math.sqrt(reps):.3f}")
    print()

    # ---------------------------------------------------------- ties --------
    print("-" * 100)
    print("STEP 5b  tie / zero-mass census on the production distribution "
          "(prefill row, eff_len=8192)")
    rng = np.random.default_rng(SEED)
    kq, ks = make_kv(KV_ROWS, HEAD_DIM, 0.0, rng)
    sc = make_scores(kq, ks, 64, N_HEAD, HEAD_DIM, rng)
    row = sc[0, :8192]
    u = np.unique(row)
    print(f"  distinct values in one row: {u.size} of 8192 "
          f"({100*u.size/8192:.4f}%)")
    print(f"  exact zeros: {(row == 0.0).sum()}   negatives: "
          f"{(row < 0).sum()} ({100*(row<0).mean():.1f}%)   "
          f"positives: {(row > 0).sum()}")
    srt = np.sort(row)[::-1]
    print(f"  k-th/(k+1)-th boundary tied: {srt[K-1] == srt[K]}")
    print(f"  distinct L0 buckets occupied (256-radix, shift 24): "
          f"{np.unique(np.right_shift(monotone_key(row.view(np.int32)), 24)).size}")
    print(f"  distinct L0 buckets, standard_normal row: "
          f"{np.unique(np.right_shift(monotone_key(np.random.default_rng(0).standard_normal(8192).astype(np.float32).view(np.int32)), 24)).size}")
    print(f"  score row: mean={row.mean():.4g} sd={row.std():.4g} "
          f"skew={float(((row-row.mean())**3).mean()/row.std()**3):.3f}")
    print()

    # -------------------------------------------------- sensitivities -------
    print("-" * 100)
    print("STEP 2/4c  SENSITIVITY of the prefill mean to the undetermined "
          "generator parameters (b=256, 2 reps)")
    print(f"{'variant':<34}{'mean_all':>10}{'+-':>8}{'mean_walk':>11}"
          f"{'crit':>8}{'unconv':>8}")
    variants = [
        ("base alpha=0 signed w", dict(alpha=0.0, signed_w=True)),
        ("alpha=0.3", dict(alpha=0.3, signed_w=True)),
        ("alpha=0.6", dict(alpha=0.6, signed_w=True)),
        ("alpha=0.9", dict(alpha=0.9, signed_w=True)),
        ("alpha=0.99", dict(alpha=0.99, signed_w=True)),
        ("|w| (non-negative scores)", dict(alpha=0.0, signed_w=False)),
        ("|w| + alpha=0.6", dict(alpha=0.6, signed_w=False)),
        ("per-head q scale spread 1.0", dict(alpha=0.0, signed_w=True,
                                             head_scale_sigma=1.0)),
    ]
    for name, kw in variants:
        ms, mws, crs, un = [], [], [], 0
        for rep in range(2):
            rng = np.random.default_rng(SEED + 7 * rep)
            kq, ks = make_kv(KV_ROWS, HEAD_DIM, kw["alpha"], rng)
            sc = np.full((256, N), -np.inf, np.float32)
            sc[:, :KV_ROWS] = make_scores(
                kq, ks, 256, N_HEAD, HEAD_DIM, rng,
                signed_w=kw.get("signed_w", True),
                head_scale_sigma=kw.get("head_scale_sigma", 0.0))
            st = run_case(sc, prefill_lengths(256, rng), K)
            ms.append(st["mean_all"])
            mws.append(st["mean_walking"])
            crs.append(st["crit"])
            un += st["unconverged"]
        print(f"{name:<34}{np.mean(ms):>10.3f}{abs(ms[0]-ms[1])/2:>8.3f}"
              f"{np.mean(mws):>11.3f}{np.mean(crs):>8.3f}{un:>8}")
    print()
    print("n_head sensitivity (prefill, alpha=0, b=256, 1 rep):")
    for nh in (16, 32, 64, 128):
        rng = np.random.default_rng(SEED)
        kq, ks = make_kv(KV_ROWS, HEAD_DIM, 0.0, rng)
        sc = np.full((256, N), -np.inf, np.float32)
        sc[:, :KV_ROWS] = make_scores(kq, ks, 256, nh, HEAD_DIM, rng)
        st = run_case(sc, prefill_lengths(256, rng), K)
        print(f"  n_head={nh:>4}: mean_all={st['mean_all']:.3f} "
              f"mean_walk={st['mean_walking']:.3f} unconv={st['unconverged']}")
    print()
    print("global row-scale sensitivity (L0 buckets are exponent bands, so a "
          "non-power-of-4 rescale moves the alignment):")
    rng = np.random.default_rng(SEED)
    kq, ks = make_kv(KV_ROWS, HEAD_DIM, 0.0, rng)
    base = np.full((256, N), -np.inf, np.float32)
    base[:, :KV_ROWS] = make_scores(kq, ks, 256, N_HEAD, HEAD_DIM, rng)
    lens = prefill_lengths(256, rng)
    for mult in (1.0, 2.0, 4.0, 1.5, 1e-6, 1e6):
        st = run_case((base * np.float32(mult)).astype(np.float32), lens, K)
        print(f"  x{mult:<8g}: mean_all={st['mean_all']:.3f} "
              f"mean_walk={st['mean_walking']:.3f}")
    print()

    # -------------------------------------------------------------- v5 ------
    print("-" * 100)
    print("STEP 6  v5 -- NUM_BUCKETS sweep at the production distribution "
          "(b=512 prefill / 64 decode, 2 reps)")
    print(f"{'phase':<10}{'NB':>6}{'max_lv':>8}{'mean_all':>10}{'mean_walk':>11}"
          f"{'crit':>8}{'unconv':>8}  hist")
    for phase, b, lenfn in (("prefill", 512, prefill_lengths),
                            ("decode", 512, decode_lengths)):
        cache = []
        for rep in range(2):
            rng = np.random.default_rng(SEED + 31 * rep)
            kq, ks = make_kv(KV_ROWS, HEAD_DIM, 0.0, rng)
            sc = np.full((b, N), -np.inf, np.float32)
            sc[:, :KV_ROWS] = make_scores(kq, ks, b, N_HEAD, HEAD_DIM, rng)
            cache.append((sc, lenfn(b, rng)))
        for nb in (64, 128, 256, 512):
            sts = [run_case(sc, ln, K, num_buckets=nb) for sc, ln in cache]
            hh = np.sum([s["hist"] for s in sts], axis=0).tolist()
            print(f"{phase:<10}{nb:>6}{len(level_schedule(nb)[1]):>8}"
                  f"{np.mean([s['mean_all'] for s in sts]):>10.3f}"
                  f"{np.mean([s['mean_walking'] for s in sts]):>11.3f}"
                  f"{np.mean([s['crit'] for s in sts]):>8.3f}"
                  f"{sum(s['unconverged'] for s in sts):>8}  {hh}")
            if phase == "prefill":
                V5_PREFILL.append((nb, float(np.mean([s["crit"]
                                                      for s in sts]))))
    # standard_normal control at the same radices, rho=1
    rng = np.random.default_rng(20260904)
    sn = synth("random", 512, N, rng)
    for nb in (64, 128, 256, 512):
        st = run_case(sn, np.full(512, N), K, num_buckets=nb)
        print(f"{'SN/rho=1':<10}{nb:>6}{len(level_schedule(nb)[1]):>8}"
              f"{st['mean_all']:>10.3f}{st['mean_walking']:>11.3f}"
              f"{st['crit']:>8.3f}{st['unconverged']:>8}  {st['hist']}")

    print()
    print("-" * 100)
    print("STEP 6b  CONSUMERS -- v3 and v5 re-priced on the new level count")
    print("-" * 100)
    # Measured inputs, all from probes/07 (j-098b260f) unless noted.
    LVL_WALK_R1 = 38.004   # us, one refinement L-walk, b=1024, rho=1
    LVL_FIXED_256 = 19.192  # us, one refinement fixed sweep at NUM_BUCKETS=256
    RHO_PRE = 0.40005
    BASE = 189.98          # us, production-rho prefill kernel (design.md:18)
    EMIT = 33.54           # us, emit traversal at production rho (hardware_map.md:470)
    REF_HIST_AT_2231 = 20.08  # us, refinement L walk at production rho, at 2.231
    EMIT1 = 27.456         # us, probes/03 conditional store pass at rho=1
    OLD = 2.231
    print(f"inputs: level walk {LVL_WALK_R1} us/level (rho=1), fixed sweep "
          f"{LVL_FIXED_256} us/level (NB=256), rho={RHO_PRE}, base {BASE} us")
    print()
    print("AGGREGATION.  2.231 is a HARDWARE timing inversion, so it is already")
    print("critical-subcore.  Both terms below must therefore use the critical")
    print("aggregation: max over the 32 subcores of the sum of ITS OWN rows,")
    print("per row (sparsecore_topk.py:157/:443, no barrier at :19/:528).")
    print("v3's added work is levels_row * cnt_row jointly, so the critical")
    print("statistic is crit(levels*cnt), not crit(levels) * crit(cnt).")
    print()
    v = V3_IN["prefill"]
    print("v3 (fused emit). Deletes the refinement L-walk and `emit`; pays one")
    print("extra L0 conditional-store pass + level_count walks of `cnt`.")
    print(f"  {'variant':<46}{'lvl':>7}{'cnt_eff':>9}{'deleted':>9}"
          f"{'added':>8}{'net us':>9}{'% base':>9}")
    rows = [
        ("A old: lvl 2.231 (crit), cnt 4096 assumed", OLD, 4096.0),
        ("B lvl crit, cnt mean  (WRONG: mismatched agg)",
         v["lvl_crit"], v["cnt_mean"]),
        ("C lvl walking-row, cnt mean (README v1)",
         v["lvl_walk"], v["cnt_mean"]),
        ("D CENTRAL: crit(levels*cnt), no-barrier truth",
         v["lvl_crit"], v["prod_crit"] / v["lvl_crit"]),
        ("E upper bd: wavemax(levels*cnt) (if barriered)",
         v["lvl_wavemax"], v["prod_wavemax"] / v["lvl_wavemax"]),
        ("F absurd bd: crit levels x GLOBAL max cnt",
         v["lvl_crit"], v["cnt_max"]),
    ]
    nets = {}
    for name, lvl, cnt in rows:
        deleted = REF_HIST_AT_2231 * (lvl / OLD) + EMIT
        added = EMIT1 * RHO_PRE + lvl * LVL_WALK_R1 * (cnt / 10240.0)
        net = added - deleted
        nets[name[0]] = net
        print(f"  {name:<46}{lvl:>7.3f}{cnt:>9.0f}{deleted:>9.2f}"
              f"{added:>8.2f}{net:>+9.2f}{100*net/BASE:>+9.2f}")
    lo, hi = min(nets["D"], nets["E"]), max(nets["D"], nets["E"])
    print(f"  -> v3 band over PLAUSIBLE aggregations (D..E): "
          f"{lo:+.2f} .. {hi:+.2f} us = {100*lo/BASE:+.2f}% .. {100*hi/BASE:+.2f}%")
    print(f"  -> v3 turns POSITIVE (a loss) only above cnt_eff = "
          f"{10240.0*(REF_HIST_AT_2231*(v['lvl_crit']/OLD)+EMIT-EMIT1*RHO_PRE)/(v['lvl_crit']*LVL_WALK_R1):.0f}"
          f"  (measured crit {v['prod_crit']/v['lvl_crit']:.0f}, "
          f"global max {v['cnt_max']:.0f})")
    print()
    print("v5 (NUM_BUCKETS sweep), prefill, production rho. Fixed sweep is "
          "assumed proportional to NUM_BUCKETS (modelled, NOT measured).")
    print(f"  T(NB) = (1 + levels(NB)) * fixed(NB) + levels(NB) * walk * rho ; "
          f"fixed(NB) = {LVL_FIXED_256} * NB/256")
    for nb, lvl in V5_PREFILL:
        fixed = LVL_FIXED_256 * nb / 256.0
        t = (1 + lvl) * fixed + lvl * LVL_WALK_R1 * RHO_PRE
        V5_T[nb] = t
    for nb, lvl in V5_PREFILL:
        d = V5_T[nb] - V5_T[256]
        print(f"  NB={nb:>4} levels(crit)={lvl:5.3f}  T={V5_T[nb]:7.2f} us"
              f"  delta vs 256 = {d:+7.2f} us = {100*d/BASE:+6.2f}% of {BASE}")
    print()
    print("v5 INTERCEPT SENSITIVITY.  probes/09 measured the sweep's four")
    print("primitives at NB=256 only (zero_priv 9.238 / fold_priv 4.900 /")
    print("scan_glob 3.498 / find_bucket 1.200 us, additive to 0.998).  All four")
    print("have trip counts proportional to NUM_BUCKETS in source (:223 NUM_BUCKETS,")
    print(":206/:208/:211 NUM_BUCKETS//LANES), but the NON-scaling per-loop")
    print("intercept c is unmeasured.  fixed(NB) = c + (19.192-c)*NB/256:")
    lv64 = dict(V5_PREFILL)[64]
    lv256 = dict(V5_PREFILL)[256]
    print(f"  {'c/19.192':>10}{'fixed(64)':>11}{'fixed(256)':>12}"
          f"{'T(64)':>9}{'T(256)':>9}{'delta us':>10}{'% base':>9}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = frac * LVL_FIXED_256
        f64 = c + (LVL_FIXED_256 - c) * 64 / 256.0
        t64 = (1 + lv64) * f64 + lv64 * LVL_WALK_R1 * RHO_PRE
        t256 = (1 + lv256) * LVL_FIXED_256 + lv256 * LVL_WALK_R1 * RHO_PRE
        d = t64 - t256
        print(f"  {frac:>10.2f}{f64:>11.2f}{LVL_FIXED_256:>12.2f}"
              f"{t64:>9.2f}{t256:>9.2f}{d:>+10.2f}{100*d/BASE:>+9.2f}")
    # crossover: solve for c where T(64) == T(256)
    t256 = (1 + lv256) * LVL_FIXED_256 + lv256 * LVL_WALK_R1 * RHO_PRE
    rhs = t256 - lv64 * LVL_WALK_R1 * RHO_PRE
    # (1+lv64)*(c + (F-c)*0.25) = rhs  ->  c*(1+lv64)*0.75 = rhs - (1+lv64)*F*0.25
    cx = (rhs - (1 + lv64) * LVL_FIXED_256 * 0.25) / ((1 + lv64) * 0.75)
    print(f"  -> NB=64 stops winning once the non-scaling intercept exceeds "
          f"c = {cx:.2f} us = {100*cx/LVL_FIXED_256:.0f}% of the NB=256 sweep.")
    print("  -> UNMEASURED at every NB but 256.  v5 is OPEN, not a result.")
    print()
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
