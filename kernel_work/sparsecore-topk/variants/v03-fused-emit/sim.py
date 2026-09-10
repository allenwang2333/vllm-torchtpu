"""Host NumPy oracle for the incumbent vs v3 selection logic. No TPU.

Reimplements both control flows exactly as the kernels express them (level-0
histogram on key>>24, up to three refinement levels, tie-rank emit) over the
same score kinds the frozen test uses, and checks:

  * v3's winner SET equals the incumbent's, bit for bit, on every row;
  * v3's winner set equals a true numpy top-k as a multiset of scores;
  * the straddler buffer never exceeds its allocation;
  * how big `cnt` actually gets (the plan's "cnt_max = 4096" claim).
"""
import numpy as np

LANES, NB, L0_SHIFT = 16, 256, 24
LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
NEG_INF_BITS = np.array(-np.inf, np.float32).view(np.int32).item()
KEY_NEG_INF = np.int32(NEG_INF_BITS) ^ np.int32(0x7FFFFFFF)


def mono(bits):
    bits = bits.astype(np.int64)
    m = (bits >> 31) & 0x7FFFFFFF
    return ((bits ^ m).astype(np.int64) << 32 >> 32).astype(np.int64)


def keys_of(row, eff):
    b = row.view(np.int32)[:eff]
    return mono(b)


def _pivot(key, k):
    """Level-0 split, shared by both variants."""
    live = key > int(KEY_NEG_INF)
    kl = key[live]
    idx = np.nonzero(live)[0]
    total = kl.size
    hist = np.zeros(NB, np.int64)
    np.add.at(hist, (kl >> L0_SHIFT) + NB // 2, 1)
    cum = np.cumsum(hist)
    keep_all = total <= k
    b_star = int(np.sum(cum <= total - k))
    cum_at = int(cum[b_star]) if b_star < NB else 0
    cum_lo = int(cum[b_star - 1]) if b_star > 0 else 0
    return dict(kl=kl, idx=idx, total=total, keep_all=keep_all,
                b_star=b_star, c_hi=total - cum_at, cnt=cum_at - cum_lo,
                quota=k - (total - cum_at),
                prefix=b_star - NB // 2, shift=L0_SHIFT)


def _refine(st, keys, k):
    """The refinement ladder. `keys` is whatever buffer the variant walks."""
    prefix, shift = st["prefix"], st["shift"]
    quota, cnt, c_hi = st["quota"], st["cnt"], st["c_hi"]
    done = st["keep_all"] or (quota == cnt)
    levels = 0
    for lvl_shift, lvl_mask in LEVELS:
        if done:
            break
        levels += 1
        match = (keys >> shift) == prefix
        sub = keys[match]
        hist = np.zeros(NB, np.int64)
        np.add.at(hist, (sub >> lvl_shift) & lvl_mask, 1)
        cum = np.cumsum(hist)
        sub_total = int(cum[-1])
        b2 = int(np.sum(cum <= sub_total - quota))
        cum_at2 = int(cum[b2]) if b2 < NB else 0
        cum_lo2 = int(cum[b2 - 1]) if b2 > 0 else 0
        c_above2, cnt2 = sub_total - cum_at2, cum_at2 - cum_lo2
        quota2 = quota - c_above2
        prefix = (prefix << (shift - lvl_shift)) | b2
        shift = lvl_shift
        c_hi, quota, cnt = c_hi + c_above2, quota2, cnt2
        done = quota2 == cnt2 or quota2 == 0
    return prefix, shift, quota, levels


def incumbent(row, eff, k):
    key = keys_of(row, eff)
    st = _pivot(key, k)
    kl, idx = st["kl"], st["idx"]
    prefix, shift, quota, levels = _refine(st, kl, k)
    if st["keep_all"]:
        return idx.copy(), st, levels
    hi = kl >> shift
    strict = hi > prefix
    tie = hi == prefix
    tie_rank = np.cumsum(tie)
    sel = strict | (tie & (tie_rank <= quota))
    return idx[sel], st, levels


def v3(row, eff, k):
    key = keys_of(row, eff)
    st = _pivot(key, k)
    kl, idx = st["kl"], st["idx"]
    p0 = st["prefix"]
    done0 = st["keep_all"] or (st["quota"] == st["cnt"])
    hi0 = kl >> L0_SHIFT
    strict0, tie0 = hi0 > p0, hi0 == p0
    win = np.ones_like(strict0) if st["keep_all"] else (
        (strict0 | tie0) if done0 else strict0)
    out = [idx[win]]
    strad_key, strad_idx = kl[tie0], idx[tie0]
    assert strad_key.size == st["cnt"], (strad_key.size, st["cnt"])
    if not done0:
        prefix, shift, quota, _ = _refine(st, strad_key, k)
        hi = strad_key >> shift
        strict, tie = hi > prefix, hi == prefix
        tie_rank = np.cumsum(tie)
        sel = strict | (tie & (tie_rank <= quota))
        out.append(strad_idx[sel])
    return np.concatenate(out), st


def scores(kind, rng, n):
    """VERBATIM from tests/kernels/test_dsv4_sparsecore_topk.py::_scores.

    Transcribed, not approximated: an earlier version of this file used its own
    `strict_desc` (positive, ascending-reversed) and `all_negative` (uniform),
    and both put `cnt` in the wrong place.  `few_distinct` is mine and is not a
    test kind -- it is kept only as an adversarial upper bound.
    """
    if kind == "strict_desc":
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], 1, axis=0)[0]
    if kind == "random":
        return rng.standard_normal(n, dtype=np.float32)
    if kind == "heavy_ties":
        return rng.integers(0, 17, size=n, dtype=np.int8).astype(np.float32)
    if kind == "all_negative":
        return (-np.abs(rng.standard_normal(n, dtype=np.float32))
                - np.float32(1.0))
    if kind == "few_distinct":          # NOT a test kind; adversarial bound
        return rng.choice(np.float32([8.0, 9.5, 12.0, 15.9]), n)
    raise ValueError(kind)


# Which score kinds actually reach which shapes, read off the test file:
#   test_exact_topk_full_rows      SHAPES (incl. P=1) x [strict_desc, random,
#                                  heavy_ties]           -- row_lengths = n, so rho = 1
#   test_emit_pairs_..._incumbent  TWO_STAGE_SHAPES (P>1 ONLY) x [random,
#                                  heavy_ties, all_negative]
# `all_negative` therefore never runs at the P=1 production shape.
P1_KINDS = ("strict_desc", "random", "heavy_ties")
P4_ONLY_KINDS = ("all_negative", )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, K = 10240, 2048
    worst_cnt, worst = 0, None
    bad = 0
    print(f"{'kind':14s} {'runs at':>9s} {'rho':>5s} {'rows':>5s} "
          f"{'set==':>6s} {'topk==':>7s} {'max cnt':>8s} {'mean cnt':>9s} "
          f"{'>4096':>6s}")
    for kind in ("strict_desc", "random", "heavy_ties", "all_negative",
                 "few_distinct"):
        where = ("P=1 too" if kind in P1_KINDS else
                 "P>1 ONLY" if kind in P4_ONLY_KINDS else "untested")
        for rho in (1.0, 0.40005):
            cnts, ok_set, ok_topk, over = [], 0, 0, 0
            rows = 96
            for _ in range(rows):
                row = scores(kind, rng, N)
                eff = max(1, int(round(rho * N)))
                a, st, _ = incumbent(row, eff, K)
                b, st2 = v3(row, eff, K)
                cnts.append(st["cnt"])
                over += st["cnt"] > 4096
                ok_set += set(a.tolist()) == set(b.tolist())
                got = np.sort(row[:eff][b])[::-1]
                want = np.sort(row[:eff])[::-1][:min(K, b.size)]
                ok_topk += np.array_equal(got, want)
                if st["cnt"] > worst_cnt:
                    worst_cnt, worst = st["cnt"], (kind, rho)
            bad += (rows - ok_set) + (rows - ok_topk)
            print(f"{kind:14s} {where:>9s} {rho:5.2f} {rows:5d} "
                  f"{ok_set:4d}/{rows} {ok_topk:5d}/{rows} {max(cnts):8d} "
                  f"{np.mean(cnts):9.1f} {over:6d}")
    print()
    print(f"largest cnt seen: {worst_cnt} in {worst}   "
          f"(plan's claimed safe bound: 4096; "
          f"v3 allocation: slice_len + LANES = {N + LANES})")
    print(f"MISMATCHES: {bad}")

    # ---- cost arithmetic, all inputs measured elsewhere -------------------
    print("\ncost model (production rho=0.40005, crit, us):")
    rho = 0.40005
    W = 20.681 * rho          # probes/15 `cnt`: read-only walk of L
    s = 6.675 * rho           # probes/15 Probe B: one store_compressed/pass
    Rlev = 38.004 * rho       # probes/07: one refinement level's L walk
    lv = 1.570                # probes/13: crit level count
    alpha = 1423 / 4096       # probes/13: crit cnt / mean eff_len
    T = lv * (Rlev - W)       # refinement transaction term (length-invariant)
    print(f"  W={W:.3f}  s={s:.3f}  Rlev={Rlev:.3f}  levels={lv}  "
          f"alpha={alpha:.3f}  T={T:.3f}")
    for name, R_v3 in (("B walk-bound  ", lv * alpha * W + T),
                       ("A' txn-bound  ", lv * Rlev)):
        base = lv * Rlev + W + s
        var = W + 3 * s + R_v3 + 2 * alpha * W + alpha * s
        d = var - base
        print(f"  {name} base={base:7.3f}  v3={var:7.3f}  "
              f"delta={d:+7.3f} us  = {100*d/191.76:+6.2f}% harness  "
              f"{100*d/155.82:+6.2f}% production")
    # break-even on the store price
    for tag, sv in (("s low 3.686", 3.686 * rho), ("s high 8.632",
                                                   8.632 * rho)):
        base = lv * Rlev + W + sv
        var = W + 3 * sv + (lv * alpha * W + T) + 2 * alpha * W + alpha * sv
        print(f"  ModelB {tag:12s} delta={var-base:+7.3f} us  "
              f"{100*(var-base)/191.76:+6.2f}% harness")

    # The disputed emit price. probes/09 prices the emit REGION at 83.830 us
    # (rho=1) where the probes/03 per-pass ladder prices the same pass at
    # 27.456. Read the 3.05x as "an emit-class walk is underpriced": W' is
    # whatever makes the pass cost 33.54 us at production once its one store
    # is charged.
    Wp = 83.830 * rho - s
    base = lv * Rlev + Wp + s
    var = Wp + 3 * s + (lv * alpha * W + T) + 2 * alpha * Wp + alpha * s
    print(f"  region-emit W'={Wp:.3f}  base={base:7.3f}  v3={var:7.3f}  "
          f"delta={var-base:+7.3f} us  {100*(var-base)/191.76:+6.2f}% harness")

    # Break-even in alpha = cnt / i_hi, Model B (the reading most favourable
    # to v3).  base = v3  =>  lv*W + s = 3s + alpha*(lv*W + 2W + s)
    a_be = (lv * W - 2 * s) / (lv * W + 2 * W + s)
    print(f"\n  break-even alpha (Model B) = {a_be:.4f}; "
          f"measured alpha = {alpha:.4f} "
          f"({'WIN' if alpha < a_be else 'LOSS'})")
    print(f"  decode: crit cnt 5539 / eff_len {0.850*10240:.0f} "
          f"-> alpha = {5539/(0.850*10240):.4f}")
