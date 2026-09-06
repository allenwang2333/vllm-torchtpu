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
    deq = (np.asarray(q_fp8).astype(np.float32) *
           np.asarray(scale.astype(jnp.float32))[:, None])
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

    poison = good.copy()
    # One exponent step on a single scale byte, at a position the oracle scores.
    poison[0, 0, 3] = np.uint8((int(poison[0, 0, 3]) + 1) % 256)

    return dict(q=q, weights=weights, deq=deq, packed=packed, val=val,
                scale_good=good, scale_wrong=wrong, scale_poison=poison,
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
    return top


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
    exp = oracle_topk(c["q"], c["weights"], c["deq"], S_valid, k)
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

    ok_inc = out["arms"]["incumbent"]["matches_oracle"]
    ok_comp = out["arms"]["composed"]["matches_oracle"]
    neg_wrong = not out["arms"]["wrong_map"]["matches_oracle"]
    neg_pois = not out["arms"]["scale_poison"]["matches_oracle"]

    print()
    print(f"BASELINE  (incumbent vs oracle)   : {'PASS' if ok_inc else 'FAIL'}")
    print(f"CHANGED   (composed  vs oracle)   : "
          f"{'PASS' if ok_comp else 'FAIL'}")
    print(f"NEG CTRL  wrong scale index map   : "
          f"{'FAILS as required' if neg_wrong else 'PASSED -- VACUOUS'}")
    print(f"NEG CTRL  one poisoned scale byte : "
          f"{'FAILS as required' if neg_pois else 'PASSED -- VACUOUS'}")

    discriminates = neg_wrong and neg_pois
    out["baseline_pass"] = ok_inc
    out["changed_pass"] = ok_comp
    out["discriminates"] = discriminates
    if not ok_inc:
        print("VERDICT: BASELINE RED -- this establishes nothing either way.")
    elif not discriminates:
        print("VERDICT: HOLLOW -- a negative control passed; composed PASS is "
              "not evidence.")
    elif ok_comp:
        print("VERDICT: PASS, and the check could have failed.")
    else:
        print("VERDICT: FAIL -- the composed reader disagrees with the "
              "independent oracle.")

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
    return 0 if (ok_inc and ok_comp and discriminates) else 1


if __name__ == "__main__":
    sys.exit(main())
