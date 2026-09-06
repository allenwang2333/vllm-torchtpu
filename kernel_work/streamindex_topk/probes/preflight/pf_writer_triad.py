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
"""PREFLIGHT: the relayout triad at `mla_attention_op.py:104`, IN SITU.

The profiled baseline (`j-074e570b-13e3-4ab7-bb0c`) shows the KV write costs
`reshape-in + scatter_fusion + reshape-out` per indexer layer -- a 108.79 MB
round trip to write 256 KB -- and that it is paid on the early-exit path too,
in steps where the scoring kernel never runs.  Option A's writer half targets
exactly that line.

A standalone store probe does NOT measure this.  The triad exists because two
consumers of the same donated buffer want different layouts: the XLA scatter
and the Pallas `tpu_custom_call` that reads the cache.  Time the scatter on
its own and XLA is free to pick one layout, the triad disappears, and the
number answers a different question -- which is exactly why the 3.342x already
in hand is not the number being asked for here.

So this probe rebuilds `_streamidx_topk_jax` from `mla_attention_op.py:72-123`
as faithfully as a microbenchmark can: the same scatter, the same Pallas call
on the same donated buffer, the same tuning tuples, `donate_argnums=(2,)`, and
the output cache fed back in as the next iteration's input so donation behaves
as it does in production.  Four arms:

    inc_full   scatter + streamindex_topk, packed 256-byte record  (incumbent)
    a_full     scatter + streamindex_topk, split value/scale planes (Option A)
    inc_store  the scatter ALONE, packed        -- the standalone probe
    a_store    the scatter ALONE, split         -- the standalone probe

`inc_full` vs `a_full` is the number the coordinator asked for.  `inc_store`
vs `a_store` is printed beside it so the gap between the in-situ ratio and the
standalone ratio is visible rather than assumed.

It also dumps the compiled HLO and counts the reshapes and the scatter fusion
on the big buffer, because a probe that fails to reproduce the triad would
otherwise report a clean, meaningless speedup.  If the incumbent arm shows no
triad, the probe has not reached the thing it was written for and says so.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

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
# The write path, transcribed from mla_attention_op.py:84-104.
# ---------------------------------------------------------------------------
def _slots(k_packed_rows, block_size, kv_packing, seq_lens, cu_q_lens,
           page_indices):
    tok = jnp.arange(k_packed_rows, dtype=jnp.int32)
    seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(
        jnp.int32)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
    page = jnp.where(
        tok < cu_q_lens[-1],
        page_indices.reshape(seq_lens.shape[0], -1)[seq_id, pos // block_size],
        jnp.int32(2**30))
    slot = pos % block_size
    return page, slot


def build_incumbent(topk_fn, k, comp_ratio, bkv_p, bq, early_exit,
                    store_only):
    def f(q_bytes, weights, cache_kv, k_packed, seq_lens, page_indices,
          cu_q_lens, distribution):
        _, bs_pk, kv_packing, lkv_dim = cache_kv.shape
        block_size = bs_pk * kv_packing
        k_packed = jnp.pad(k_packed,
                           ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
        page, slot = _slots(k_packed.shape[0], block_size, kv_packing,
                            seq_lens, cu_q_lens, page_indices)
        cache_kv = cache_kv.at[page, slot // kv_packing,
                               slot % kv_packing].set(k_packed)
        if store_only:
            return cache_kv, jnp.zeros((q_bytes.shape[0], k), jnp.int32)
        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        idx = topk_fn(q, weights, cache_kv, seq_lens, page_indices, cu_q_lens,
                      distribution, k=k, compression_ratio=comp_ratio,
                      num_kv_pages_per_block=bkv_p,
                      num_queries_per_block=bq,
                      enable_early_exit=early_exit)
        return cache_kv, idx

    return jax.jit(f, donate_argnums=(2,))


def build_option_a(topk_fn, k, comp_ratio, bkv_p, bq, early_exit, bucket,
                   store_only):
    """Option A's writer: value bytes to one plane, the scale byte to another.

    Two scatters replace one.  The value scatter writes 128 bytes per row
    instead of 256 (the incumbent pads the 129-byte record out to 256), and
    the scale scatter writes one byte per row into a plane that is three
    orders of magnitude smaller than the value plane.
    """
    def f(q_bytes, weights, cache_kv, k_packed, seq_lens, page_indices,
          cu_q_lens, distribution):
        cache_val, cache_scale = cache_kv
        _, bs_pk, kv_packing, d = cache_val.shape
        block_size = bs_pk * kv_packing
        page, slot = _slots(k_packed.shape[0], block_size, kv_packing,
                            seq_lens, cu_q_lens, page_indices)
        vals = k_packed[:, :d]
        scales = k_packed[:, d]
        cache_val = cache_val.at[page, slot // kv_packing,
                                 slot % kv_packing].set(vals)
        cache_scale = cache_scale.at[page, slot // 128, slot % 128].set(scales)
        if store_only:
            return (cache_val, cache_scale), jnp.zeros(
                (q_bytes.shape[0], k), jnp.int32)
        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        idx = topk_fn(q, weights, (cache_val, cache_scale), seq_lens,
                      page_indices, cu_q_lens, distribution, k=k,
                      compression_ratio=comp_ratio,
                      num_kv_pages_per_block=bkv_p,
                      num_queries_per_block=bq,
                      enable_early_exit=early_exit,
                      kv_len_bucket=bucket)
        return (cache_val, cache_scale), idx

    return jax.jit(f, donate_argnums=(2,))


# ---------------------------------------------------------------------------


def hlo_triad(text, big_bytes):
    """Count the relayout ops the profile attributes to `:104`.

    Look for reshape/copy/transpose/bitcast-convert on operands whose element
    count is within 2x of the cache, plus the scatter fusion itself.  The
    absolute counts are less interesting than whether they are NON-ZERO on the
    incumbent arm: that is what says the probe reproduced the in-situ case and
    is not the standalone store probe in disguise.
    """
    n_reshape = n_copy = n_scatter = n_fusion = 0
    for line in text.splitlines():
        s = line.strip()
        if " = " not in s:
            continue
        m = re.search(r"u8\[([0-9,]+)\]", s)
        big = False
        if m:
            dims = [int(x) for x in m.group(1).split(",") if x]
            n = 1
            for d_ in dims:
                n *= d_
            big = n >= big_bytes // 2
        if re.search(r"=\s*u8\[[0-9,]*\]\s*reshape\(", s) and big:
            n_reshape += 1
        if re.search(r"=\s*u8\[[0-9,]*\]\s*(copy|transpose|bitcast)\(", s) \
                and big:
            n_copy += 1
        if "scatter(" in s:
            n_scatter += 1
        if "fusion(" in s and big:
            n_fusion += 1
    return dict(big_reshape=n_reshape, big_copy=n_copy, scatter=n_scatter,
                big_fusion=n_fusion)


def timeit(fn, args, cache_slot, reps, warmup):
    """Time `fn`, feeding the donated cache back in, as production does."""
    a = list(args)
    for _ in range(warmup):
        out, idx = fn(*a)
        jax.block_until_ready((out, idx))
        a[cache_slot] = out
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out, idx = fn(*a)
        jax.block_until_ready((out, idx))
        ts.append((time.perf_counter() - t0) * 1e6)   # microseconds
        a[cache_slot] = out
    return ts


def main():
    ap = argparse.ArgumentParser()
    # The A/B shape: cache_kv is u8[415, 256, 4, 256] == 108.79 MB, which is
    # the round trip the profile measured.
    ap.add_argument("--num-blocks", type=int, default=415)
    ap.add_argument("--bs-pk", type=int, default=256)
    ap.add_argument("--kv-packing", type=int, default=4)
    ap.add_argument("--lkv-dim", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--pages-per-seq", type=int, default=65)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="short enough that early exit fires, which is the "
                         "prefill_only case the profile attributes the triad "
                         "to: the triad is paid, the scoring kernel is not")
    ap.add_argument("--bucket", type=int, default=65536,
                    help="kv_len_bucket for the composed arm; the shipping "
                         "value at the A/B shape is min(66560, 65536)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--hlo-dir", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    D = 128
    nb, bs_pk, kvp, lkv = a.num_blocks, a.bs_pk, a.kv_packing, a.lkv_dim
    block_size = bs_pk * kvp
    big_bytes = nb * bs_pk * kvp * lkv
    T, S = a.tokens, a.num_seqs

    print(f"=== writer triad IN SITU ===")
    print(f"  cache_kv    = u8[{nb}, {bs_pk}, {kvp}, {lkv}] = "
          f"{big_bytes / 2**20:.2f} MiB   (page_size = {block_size})")
    print(f"  option A    = u8[{nb}, {bs_pk}, {kvp}, {D}] + "
          f"u8[{nb}, {block_size // 128}, 128] = "
          f"{(nb * bs_pk * kvp * D + nb * (block_size // 128) * 128) / 2**20:.2f}"
          f" MiB")
    print(f"  tokens T    = {T}; rows written = {T} x {lkv} B = "
          f"{T * lkv / 2**10:.0f} KiB (incumbent), {T * (D + 1) / 2**10:.0f} "
          f"KiB of payload")
    print(f"  seqs        = {S} x seq_len {a.seq_len}, pages/seq "
          f"{a.pages_per_seq}, topk {a.topk}, heads {a.heads}")
    print(f"  early exit  = True, seq_len {a.seq_len} <= topk {a.topk} "
          f"-> scoring kernel skipped, triad still paid", flush=True)

    rng = np.random.default_rng(7)
    q_bytes = jnp.asarray(rng.integers(0, 255, (T, a.heads, D), dtype=np.uint8))
    weights = jnp.asarray(rng.standard_normal((T, a.heads)).astype(np.float32))
    k_packed = jnp.asarray(
        rng.integers(0, 255, (T, D + 1), dtype=np.uint8))
    seq_lens = jnp.full((S,), a.seq_len, dtype=jnp.int32)
    pi = np.arange(S * a.pages_per_seq, dtype=np.int32) % nb
    page_indices = jnp.asarray(pi)
    q_per = T // S
    cu_q_lens = jnp.asarray(np.arange(S + 1, dtype=np.int32) * q_per)
    distribution = jnp.asarray([0, 0, S], dtype=jnp.int32)

    cache_inc = jnp.zeros((nb, bs_pk, kvp, lkv), jnp.uint8)
    cache_a = (jnp.zeros((nb, bs_pk, kvp, D), jnp.uint8),
               jnp.zeros((nb, block_size // 128, 128), jnp.uint8))

    inc_mod = _support.load_incumbent()
    comp_mod = _support.load_composed()
    print(f"  incumbent kernel = {inc_mod.__file__}")
    print(f"  composed  kernel = {comp_mod.__file__}", flush=True)

    bkv_p, bq = (2, 2, 2), (1, 128, 128)
    bucket = a.bucket

    arms = {}
    for name, cache, fn in (
        ("inc_full", cache_inc,
         build_incumbent(inc_mod.streamindex_topk, a.topk, 1, bkv_p, bq,
                         True, False)),
        ("a_full", cache_a,
         build_option_a(comp_mod.streamindex_topk, a.topk, 1, bkv_p, bq,
                        True, bucket, False)),
        ("inc_store", cache_inc,
         build_incumbent(inc_mod.streamindex_topk, a.topk, 1, bkv_p, bq,
                         True, True)),
        ("a_store", cache_a,
         build_option_a(comp_mod.streamindex_topk, a.topk, 1, bkv_p, bq,
                        True, bucket, True)),
    ):
        args = (q_bytes, weights, cache, k_packed, seq_lens, page_indices,
                cu_q_lens, distribution)
        print(f"\n--- {name}: compiling ---", flush=True)
        try:
            text = fn.lower(*args).compile().as_text()
        except Exception as e:                     # noqa: BLE001
            print(f"  {name}: COMPILE FAILED: {type(e).__name__}: {e}")
            arms[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        tri = hlo_triad(text, big_bytes)
        print(f"  HLO on the big buffer: {tri}")
        if a.hlo_dir:
            os.makedirs(a.hlo_dir, exist_ok=True)
            with open(os.path.join(a.hlo_dir, f"{name}.hlo"), "w") as f:
                f.write(text)
        try:
            ts = timeit(fn, args, 2, a.reps, a.warmup)
        except Exception as e:                     # noqa: BLE001
            print(f"  {name}: RUN FAILED: {type(e).__name__}: {e}")
            arms[name] = {"error": f"{type(e).__name__}: {e}", "hlo": tri}
            continue
        med = statistics.median(ts)
        arms[name] = {"median_us": med, "min_us": min(ts), "max_us": max(ts),
                      "stdev_us": statistics.pstdev(ts), "hlo": tri,
                      "reps": a.reps}
        print(f"  {name}: median {med:.2f} us  (min {min(ts):.2f}, "
              f"max {max(ts):.2f}, sd {statistics.pstdev(ts):.2f}, "
              f"n={a.reps})")

    print()
    print("=== per indexer layer, one call ===")
    print(f"{'arm':<12} {'median us':>12} {'Δ time %':>10}")
    for name in ("inc_full", "a_full", "inc_store", "a_store"):
        r = arms.get(name, {})
        if "median_us" not in r:
            print(f"{name:<12} {'FAILED':>12} {'-':>10}")
            continue
        base = arms.get("inc_full" if name.endswith("full") else "inc_store",
                        {}).get("median_us")
        d = ("" if base is None or name.startswith("inc")
             else f"{100.0 * (r['median_us'] - base) / base:+.2f}")
        print(f"{name:<12} {r['median_us']:>12.2f} {d:>10}")

    ok = all("median_us" in arms.get(n, {})
             for n in ("inc_full", "a_full"))
    print()
    if ok:
        i, aa = arms["inc_full"]["median_us"], arms["a_full"]["median_us"]
        print(f"IN SITU   inc_full -> a_full : {i:.2f} -> {aa:.2f} us  "
              f"({i / aa:.3f}x, Δ time {100.0 * (aa - i) / i:+.2f}%)")
    if all("median_us" in arms.get(n, {}) for n in ("inc_store", "a_store")):
        i, aa = arms["inc_store"]["median_us"], arms["a_store"]["median_us"]
        print(f"STANDALONE inc_store -> a_store: {i:.2f} -> {aa:.2f} us  "
              f"({i / aa:.3f}x, Δ time {100.0 * (aa - i) / i:+.2f}%)")
    tri_inc = arms.get("inc_full", {}).get("hlo", {})
    reproduced = (tri_inc.get("big_reshape", 0) + tri_inc.get("big_copy", 0)
                  + tri_inc.get("big_fusion", 0)) > 0
    if reproduced:
        print("TRIAD REPRODUCED IN THE INCUMBENT ARM: yes -- "
              f"{tri_inc}")
    else:
        print("TRIAD REPRODUCED IN THE INCUMBENT ARM: NO. This probe did not "
              "reach the thing it was written for; the numbers above are a "
              "standalone store probe and must NOT be reported as in situ.")

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"config": vars(a), "arms": arms,
                       "triad_reproduced": reproduced}, f, indent=2,
                      default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
