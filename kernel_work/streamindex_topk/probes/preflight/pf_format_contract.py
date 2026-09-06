# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PREFLIGHT: the KV-cache FORMAT CONTRACT, against an INDEPENDENT oracle.

Everything the step-3/step-4 exercise measured judged the composed kernel
against the *incumbent kernel* (index-set equality) and against a float64
re-implementation of the same scoring the kernel does, on inputs built by
`instruments/inputs.py`.  Nothing has ever judged Option A's split layout
against the oracle the REPO already owns:
`tests/kernels/test_dsv4_streamindex_topk.py::test_streamindex_topk_quantized`
builds a packed uint8 record with the repo's own `quantize_fp8_ue8m0`, then
compares the kernel's top-k to a plain `np.argsort` over exactly-dequantized
keys.  That oracle knows nothing about either kernel.

This probe reuses that oracle verbatim (the two helpers and the scoring loop
are copied out of the test file, not re-derived) and runs FOUR arms on the
SAME quantized bytes:

  incumbent    packed  u8[P, ps//4, 4, 256]      -- the format that ships today
  composed     split   u8[P, ps//4, 4, 128]  +  u8[P, ps//128, 128]
  wrong_map    split, but the SCALE PLANE IS WRITTEN WITH THE WRONG INDEX MAP
               ([g, p % (ps//128), p // (ps//128)] instead of
                [g, p // 128,      p % 128]) -- the exact writer bug Option A
               can ship, since the writer half does not exist in `src/` yet.
  scale_poison split, one scale byte perturbed by one exponent step.

`wrong_map` and `scale_poison` are the NEGATIVE CONTROLS.  They are REQUIRED
to disagree with the oracle.  If they pass, this check cannot fail and its
`composed` PASS is worth nothing -- the script says so and exits non-zero.

Why `page_size = 128` and not the production 1024: Option A's scale plane is
declared `u8[num_pages, page_size // 128, 128]`, so `page_size` must be a
multiple of 128 for the plane to be non-degenerate.  The existing repo test
runs at page_size 4/8/16, where `page_size // 128 == 0`; those shapes are
OUTSIDE Option A's supported domain and the repo test cannot be pointed at the
composed kernel unadapted.  128 is the smallest legal page size and keeps the
oracle's host cost bounded.  A second arm runs at the production 1024.
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir,
                                     os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

sys.path.insert(0, os.path.join(_REPO, "kernel_work", "streamindex_topk",
                                "final"))
import _support  # noqa: E402  (final/_support.py: load_incumbent/load_composed)

# ---------------------------------------------------------------------------
# VERBATIM from tests/kernels/test_dsv4_streamindex_topk.py -- the repo's own
# quantizer and byte-lane view.  Copied rather than imported so that this probe
# does not depend on pytest collection, but NOT re-derived.
# ---------------------------------------------------------------------------


def _to_byte_lane(x: jax.Array) -> jax.Array:
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


# ---------------------------------------------------------------------------


def build_case(seed, page_size, num_pages, T_seq, H_I, D, comp_ratio):
    """Quantize once; emit the packed record and the two split planes."""
    rng = np.random.default_rng(seed)
    kv_packing = 4
    assert page_size % kv_packing == 0
    assert page_size % 128 == 0, "Option A needs page_size % 128 == 0"

    q = rng.standard_normal((T_seq, H_I, D)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=(T_seq, H_I)).astype(np.float32)

    f32_kv = rng.standard_normal((num_pages, page_size, D)).astype(np.float32)

    # One vectorized call over every position: same maths as the repo test's
    # per-row loop, since quantize_fp8_ue8m0 blocks on the trailing dim.
    flat = jnp.asarray(f32_kv.reshape(-1, D))
    q_fp8, scale = quantize_fp8_ue8m0(flat, 128)         # [N, D], [N]
    q_bytes = np.asarray(_to_byte_lane(q_fp8))           # u8[N, 128]
    scale_bytes = np.asarray(_to_byte_lane(scale.reshape(-1, 1)))  # u8[N, 1]
    # `scale` is [N, D // 128] == [N, 1]; broadcasting against q_fp8's [N, D]
    # needs it left at rank 2, not promoted to rank 3.
    sc_f32 = np.asarray(scale.astype(jnp.float32)).reshape(-1, 1)
    assert sc_f32.shape == (N_TOTAL := q_bytes.shape[0], 1), sc_f32.shape
    deq = np.asarray(q_fp8).astype(np.float32) * sc_f32
    assert deq.shape == (N_TOTAL, D), deq.shape
    deq = deq.reshape(num_pages, page_size, D)

    N = num_pages * page_size
    ps_pk = page_size // kv_packing

    # -- incumbent writer: one 256-byte record, values || scale || padding ----
    width = 256
    packed = np.zeros((num_pages, ps_pk, kv_packing, width), dtype=np.uint8)
    rec = np.concatenate([q_bytes, scale_bytes], axis=-1)          # [N, 129]
    rec = np.pad(rec, ((0, 0), (0, width - rec.shape[-1])))        # [N, 256]
    for i in range(N):
        p, s = divmod(i, page_size)
        packed[p, s // kv_packing, s % kv_packing] = rec[i]

    # -- Option A writer: value plane + 3-D scale plane ----------------------
    val = np.zeros((num_pages, ps_pk, kv_packing, 128), dtype=np.uint8)
    for i in range(N):
        p, s = divmod(i, page_size)
        val[p, s // kv_packing, s % kv_packing] = q_bytes[i]

    sc_rows = page_size // 128
    good = np.zeros((num_pages, sc_rows, 128), dtype=np.uint8)
    wrong = np.zeros((num_pages, sc_rows, 128), dtype=np.uint8)
    for i in range(N):
        p, s = divmod(i, page_size)
        # THE CONTRACT: scale for position `s` of page `p` at [p, s//128, s%128]
        good[p, s // 128, s % 128] = scale_bytes[i, 0]
        # THE BUG: row-major transposed within the page.
        wrong[p, s % sc_rows, s // sc_rows] = scale_bytes[i, 0]

    # One exponent step (a factor of two, since e8m0 is a bare exponent) on the
    # first 64 scales.  A single byte would be a weak control: it has to move
    # one position of `S_valid` across the top-k boundary to be seen at all.
    poison = good.copy()
    n_pois = 0
    for i in range(min(64, N)):
        p, s = divmod(i, page_size)
        poison[p, s // 128, s % 128] = np.uint8(
            (int(good[p, s // 128, s % 128]) + 1) % 256)
        n_pois += 1
    print(f"  poisoned {n_pois} scale bytes (+1 exponent = x2)")

    # At sc_rows == 1 the index map [p, s//128, s%128] and its transpose
    # [p, s%1, s//1] are the SAME map, so `wrong` is not a perturbation at all.
    # Say so, rather than let it be scored as a negative control that "passed":
    # that would report HOLLOW for a reason about the probe's geometry rather
    # than about the kernel.
    wrong_degenerate = bool(np.array_equal(wrong, good))
    if wrong_degenerate:
        print(f"  NOTE: sc_rows={sc_rows}, so the transposed scale index map "
              f"is identical to the correct one at this page_size; the "
              f"wrong_map control is inexpressible here.")

    return dict(q=q, weights=weights, deq=deq, packed=packed, val=val,
                scale_good=good, scale_wrong=wrong, scale_poison=poison,
                wrong_degenerate=wrong_degenerate,
                page_size=page_size, num_pages=num_pages,
                comp_ratio=comp_ratio)


def oracle_topk(q, weights, deq, S_valid, k):
    """VERBATIM scoring from test_streamindex_topk_quantized (H_KV == 1)."""
    T_seq, H_I, _ = q.shape
    seq_kv = deq.reshape(-1, deq.shape[-1])[:S_valid]
    scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)
    for t in range(T_seq):
        for s in range(S_valid):
            acc = 0.0
            for h in range(H_I):
                acc += max(0.0, float(np.dot(q[t, h], seq_kv[s]))) * \
                    float(weights[t, h])
            scores[t, s] = acc
    top = np.argsort(-scores, axis=-1)[:, :k]
    for t in range(T_seq):
        for i in range(k):
            if scores[t, top[t, i]] == -np.inf:
                top[t, i] = -1
    return top, scores


def diff_report(name, got, exp, scores, k):
    """How far apart are two index sets, and is the gap a boundary tie?

    An exact-equality verdict cannot tell "the reader is broken" from "the
    512th and 513th score out of 4096 are equal to seven digits and the MXU
    broke the tie the other way".  The first is a bug; the second is a
    property of comparing a Pallas kernel to a NumPy reference and says
    nothing about the change.  Print enough to tell them apart.
    """
    T = got.shape[0]
    worst = 0.0
    total = 0
    for t in range(T):
        g, e = set(got[t].tolist()), set(exp[t].tolist())
        sym = (g - e) | (e - g)
        total += len(sym)
        if not sym:
            continue
        kth = np.sort(scores[t])[::-1][k - 1]
        for idx in sym:
            if idx < 0 or idx >= scores.shape[1]:
                worst = float("inf")
                continue
            s = float(scores[t, idx])
            rel = abs(s - kth) / max(abs(kth), 1e-9)
            worst = max(worst, rel)
        print(f"    [{name}] t={t}: |sym diff|={len(sym)} of k={k}; "
              f"kth score={kth:.7g}; worst relative gap of a swapped "
              f"index to it = {worst:.3g}")
    if total == 0:
        print(f"    [{name}] exact match")
    return total, worst


def run_arm(fn, q, weights, cache, num_pages, page_size, S_seq, T_seq, k,
            comp_ratio, bkv_p, dist, extra):
    return np.asarray(
        fn(q=jnp.asarray(q),
           indexer_weights=jnp.asarray(weights),
           cache_kv=cache,
           seq_lens=jnp.asarray([S_seq], dtype=jnp.int32),
           page_indices=jnp.arange(num_pages, dtype=jnp.int32),
           cu_q_lens=jnp.asarray([0, T_seq], dtype=jnp.int32),
           distribution=jnp.asarray(dist, dtype=jnp.int32),
           k=k,
           compression_ratio=comp_ratio,
           num_kv_pages_per_block=bkv_p,
           num_queries_per_block=1,
           **extra))


def agrees(a, b):
    return bool(np.array_equal(np.sort(a, axis=-1), np.sort(b, axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--num-pages", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--comp-ratio", type=int, default=4)
    ap.add_argument("--bkv-p", default="2",
                    help="int, or 'd,p,m' for the (decode,prefill,mixed) tuple")
    ap.add_argument("--decode", action="store_true",
                    help="drive the DECODE band (T=1, distribution=[1,1,1]) "
                         "instead of the mixed band, so that slot 0 of the "
                         "bkv_p tuple is the one that executes")
    ap.add_argument("--bucket", type=int, default=None,
                    help="kv_len_bucket for the composed arm; default = live")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    bkv_p = (tuple(int(x) for x in a.bkv_p.split(","))
             if "," in a.bkv_p else int(a.bkv_p))
    if a.decode:
        a.tokens = 1
        dist = [1, 1, 1]
    else:
        dist = [0, 0, 1]

    D = 128
    ps, npg = a.page_size, a.num_pages
    S_valid = npg * ps
    S_seq = S_valid * a.comp_ratio
    k = a.k

    print(f"=== format contract: page_size={ps} num_pages={npg} "
          f"S_valid={S_valid} S_seq={S_seq} T={a.tokens} H={a.heads} "
          f"k={k} comp_ratio={a.comp_ratio} bkv_p={bkv_p} "
          f"band={'DECODE' if a.decode else 'MIXED'} ===", flush=True)

    c = build_case(a.seed, ps, npg, a.tokens, a.heads, D, a.comp_ratio)
    print("  built; running host oracle ...", flush=True)
    exp, oscores = oracle_topk(c["q"], c["weights"], c["deq"], S_valid, k)
    print(f"  oracle done, {exp.shape}", flush=True)

    inc = _support.load_incumbent()
    comp = _support.load_composed()
    print(f"  incumbent = {inc.__file__}")
    print(f"  composed  = {comp.__file__}")

    bucket = a.bucket if a.bucket is not None else S_valid
    common = dict(num_pages=npg, page_size=ps, S_seq=S_seq, T_seq=a.tokens,
                  k=k, comp_ratio=a.comp_ratio, bkv_p=bkv_p, dist=dist)

    res = {}
    res["incumbent"] = run_arm(inc.streamindex_topk, c["q"], c["weights"],
                               jnp.asarray(c["packed"]), extra={}, **common)
    for name, plane in (("composed", "scale_good"),
                        ("wrong_map", "scale_wrong"),
                        ("scale_poison", "scale_poison")):
        res[name] = run_arm(
            comp.streamindex_topk, c["q"], c["weights"],
            (jnp.asarray(c["val"]), jnp.asarray(c[plane])),
            extra=dict(kv_len_bucket=bucket), **common)

    out = {"config": vars(a), "bucket": bucket, "arms": {}}
    print()
    print(f"{'arm':<14} {'== oracle':<10} {'== incumbent':<13}")
    for name, got in res.items():
        vo, vi = agrees(got, exp), agrees(got, res["incumbent"])
        out["arms"][name] = {"matches_oracle": vo, "matches_incumbent": vi}
        print(f"{name:<14} {str(vo):<10} {str(vi):<13}")
    print()
    print("  --- how far apart, and is it a boundary tie? ---")
    for name in ("incumbent", "composed"):
        n_sym, worst = diff_report(name, res[name], exp, oscores, k)
        out["arms"][name]["sym_diff"] = n_sym
        out["arms"][name]["worst_rel_gap"] = worst
    # composed vs incumbent is the comparison this preflight actually needs:
    # same arithmetic on both sides, so it is exact, and it isolates the
    # layout change from every property of the NumPy reference.
    n_ci = int(np.sum(np.sort(res["composed"], axis=-1) !=
                      np.sort(res["incumbent"], axis=-1)))
    print(f"    [composed vs incumbent] differing sorted positions: {n_ci}")
    out["composed_vs_incumbent_differing"] = n_ci

    ok_inc = out["arms"]["incumbent"]["matches_oracle"]
    ok_comp = out["arms"]["composed"]["matches_oracle"]
    wrong_deg = c["wrong_degenerate"]
    neg_wrong = not out["arms"]["wrong_map"]["matches_oracle"]
    neg_pois = not out["arms"]["scale_poison"]["matches_oracle"]
    out["wrong_map_degenerate"] = wrong_deg

    print()
    print(f"BASELINE  (incumbent vs oracle)   : {'PASS' if ok_inc else 'FAIL'}")
    print(f"CHANGED   (composed  vs oracle)   : "
          f"{'PASS' if ok_comp else 'FAIL'}")
    print(f"NEG CTRL  wrong scale index map   : "
          + ("N/A -- inexpressible at sc_rows==1" if wrong_deg else
             ('FAILS as required' if neg_wrong else 'PASSED -- VACUOUS')))
    print(f"NEG CTRL  poisoned scale bytes    : "
          f"{'FAILS as required' if neg_pois else 'PASSED -- VACUOUS'}")

    discriminates = neg_pois and (wrong_deg or neg_wrong)
    out["baseline_pass"] = ok_inc
    out["changed_pass"] = ok_comp
    out["discriminates"] = discriminates

    # PRIMARY: composed vs incumbent.  Both sides are Pallas kernels doing the
    # same arithmetic on the same quantized bytes, differing only in how those
    # bytes are laid out in HBM -- which is exactly and only what Option A
    # changes.  It is an exact comparison, and the negative controls make it
    # non-vacuous by differing from the incumbent too.
    diff_ok = (out["arms"]["composed"]["matches_incumbent"]
               and not out["arms"]["wrong_map"]["matches_incumbent"]
               and not out["arms"]["scale_poison"]["matches_incumbent"])
    out["differential_pass"] = diff_ok
    print()
    print(f"PRIMARY   composed == incumbent    : "
          f"{'PASS' if out['arms']['composed']['matches_incumbent'] else 'FAIL'}")
    print(f"PRIMARY   controls != incumbent    : "
          f"{'yes -- differential is non-vacuous' if diff_ok else 'NO'}")

    if not ok_inc:
        print("SECONDARY: the NumPy oracle disagrees with the INCUMBENT too, "
              "so the oracle arm establishes nothing either way. Read the "
              "boundary-tie diagnostics above before blaming the change.")
    elif not discriminates:
        print("SECONDARY: HOLLOW -- a negative control matched the oracle.")
    elif ok_comp:
        print("SECONDARY: oracle PASS, and that check could have failed.")
    else:
        print("SECONDARY: oracle FAIL for the composed arm ALONE -- this "
              "would implicate the change.")
    print(f"VERDICT: {'PASS' if diff_ok else 'FAIL'} "
          f"(differential, composed vs incumbent)")

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
    return 0 if diff_ok else 1


if __name__ == "__main__":
    sys.exit(main())
