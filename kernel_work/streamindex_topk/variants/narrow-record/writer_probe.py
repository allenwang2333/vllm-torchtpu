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
"""PRICE THE WRITER. `measurement_and_design.md` section 9.4 item 2.

Variant A moves the fusion boundary UP into the producer: the scatter at
`mla_attention_op.py:103-104` stops writing one 256 B record and starts writing
a 128 B value row plus a 1 B scale. That producer-side cost is the term the
ranking turns on, and section 3.5 names it as what would kill A. It is NOT
netted against the reader-side win anywhere -- this probe produces its own
number and RESULTS.md reports the two separately.

  probe:  Does splitting the cache scatter in two make the producer slower,
          and by how much, at the production cache size?
  config: The scatter is pure XLA -- no Pallas -- so it is reproduced here
          exactly as `mla_attention_op.py:93-104` writes it, at the production
          cache shape `u8[415, 256, 4, 256]` (424960 positions x 256 B =
          108.79 MB, which is the `u8[424960,256]` the profile names) and at
          the production token counts.

Arms, all `jax.jit` with the same `donate_argnums` production uses:

  W0    today: pad k_packed[T,129] -> [T,256], one scatter into the 256 B cache
  W1    split, value plane only:  scatter k_val[T,128] into u8[415,256,4,128]
  W2    split, scale plane only:  scatter k_scale[T] into u8[415,1024]
  W12   split, both planes in one jit -- THIS is what replaces W0
  N0/N1/N2  donated single-element scatters on each buffer. Not results: they
          are the floor of "touch this buffer at all under donation", so that
          W12 - (N1+N2) can be read as work rather than as machinery.

ABSOLUTES DRIFT, RATIOS DO NOT. The number this probe exists to produce is
`W12 / W0`; the microseconds are printed to make it checkable, not to be
quoted against a profile taken on another day at another batch composition.
"""

import argparse
import json
import statistics
import sys
import time

import _load  # noqa: F401  -- puts the repo root on sys.path

# The production allocation the profile names: `u8[424960, 256]`.
PROD_NUM_PAGES = 415
WARMUP = 3
REPS = 10
INNER = 5


def _median_spread(times_s):
    med = statistics.median(times_s)
    spread = (max(times_s) - min(times_s)) / med * 100.0 if med else float("nan")
    return med * 1e3, spread


def _time_chain(fn, buf, *rest, reps=REPS, warmup=WARMUP, inner=INNER):
    """Time a donated in-place update by chaining the buffer through itself.

    Donation consumes the input, so a loop that reuses one buffer would raise.
    Feeding the result back in is also what production does: the cache is a
    single live buffer that every step updates.
    """
    import jax
    buf = jax.block_until_ready(fn(buf, *rest))
    for _ in range(warmup):
        buf = jax.block_until_ready(fn(buf, *rest))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            buf = fn(buf, *rest)
        jax.block_until_ready(buf)
        ts.append((time.perf_counter() - t0) / inner)
    return buf, ts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", default="1024,64",
                    help="prefill (max_num_batched_tokens) and decode")
    ap.add_argument("--num-pages", type=int, default=PROD_NUM_PAGES)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from kernel_work.streamindex_topk.instruments import shapes as S

    print("jax", jax.__version__, "devices", jax.devices(), flush=True)

    npages = a.num_pages
    pspp = S.PAGE_SIZE_PER_KV_PACKING          # 256
    kvp = S.KV_PACKING                          # 4
    page_size = pspp * kvp                      # 1024
    wide = S.LKV_DIM                            # 256
    narrow = S.INDEX_HEAD_DIM                   # 128
    used = S.CACHE_RECORD_USED_BYTES            # 129

    n_pos = npages * page_size
    print(f"cache: {npages} pages x {page_size} positions = {n_pos} positions",
          flush=True)
    print(f"  wide   u8[{npages},{pspp},{kvp},{wide}] = "
          f"{n_pos*wide/1e6:.2f} MB", flush=True)
    print(f"  val    u8[{npages},{pspp},{kvp},{narrow}] = "
          f"{n_pos*narrow/1e6:.2f} MB", flush=True)
    print(f"  scale  u8[{npages},{page_size}] = {n_pos/1e6:.2f} MB "
          f"(2-D form, W2/W12)", flush=True)
    print(f"  scale  u8[{npages},{page_size//128},128] = {n_pos/1e6:.2f} MB "
          f"(3-D form the reader needs, W2b/W12b)", flush=True)

    rng = np.random.default_rng(0)
    wide_cache = jnp.asarray(
        rng.integers(0, 256, (npages, pspp, kvp, wide), dtype=np.uint8))
    val_cache = jnp.asarray(
        rng.integers(0, 256, (npages, pspp, kvp, narrow), dtype=np.uint8))
    scale_cache = jnp.asarray(
        rng.integers(0, 256, (npages, page_size), dtype=np.uint8))
    # The 3-D scale plane the reader's tiling actually requires. Same bytes,
    # same order, same total size -- only the declared shape differs, so the
    # writer pays one extra integer op splitting `slot` into (slot//128,
    # slot%128). W2b/W12b price that; W2/W12 keep the 2-D form so the number
    # already reported for job j-ce74a9f2-a4cd-40cb-9e87 reproduces in the
    # same job as its replacement rather than across two jobs.
    scale_cache3 = jnp.asarray(
        rng.integers(0, 256, (npages, page_size // 128, 128), dtype=np.uint8))

    # --- the index arithmetic, verbatim from mla_attention_op.py:93-104 ----
    def indices(k_rows, cu_q_lens, seq_lens, page_indices):
        tok = jnp.arange(k_rows, dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(
            jnp.int32)
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
        page = jnp.where(
            tok < cu_q_lens[-1],
            page_indices.reshape(seq_lens.shape[0], -1)[seq_id,
                                                        pos // page_size],
            jnp.int32(2**30))
        slot = pos % page_size
        return page, slot

    rows = []
    for T in [int(x) for x in a.num_tokens.split(",") if x]:
        # One sequence per token for decode (T seqs, q_len 1 each); one long
        # sequence for prefill. Both are `searchsorted`-shaped exactly as
        # production builds them.
        if T <= 64:
            num_seqs, q_lens = T, np.ones(T, dtype=np.int32)
        else:
            num_seqs, q_lens = 8, np.full(8, T // 8, dtype=np.int32)
        cu = np.zeros(num_seqs + 1, dtype=np.int32)
        cu[1:] = np.cumsum(q_lens)
        seq_lens_np = (np.arange(num_seqs, dtype=np.int32) * 7 + 4096
                       + q_lens).astype(np.int32)
        pps = max(1, int(np.ceil(seq_lens_np.max() / page_size)))
        if num_seqs * pps > npages:
            pps = npages // num_seqs
        page_idx = rng.permutation(npages)[:num_seqs * pps].astype(np.int32)

        cu_j = jnp.asarray(cu)
        sl_j = jnp.asarray(seq_lens_np)
        pi_j = jnp.asarray(page_idx)

        k_packed = jnp.asarray(
            rng.integers(0, 256, (T, used), dtype=np.uint8))
        k_val = jnp.asarray(k_packed[:, :narrow])
        k_scale = jnp.asarray(k_packed[:, narrow])

        def w0_fn(cache, kp, cu_, sl_, pi_):
            kp = jnp.pad(kp, ((0, 0), (0, wide - kp.shape[-1])))
            page, slot = indices(kp.shape[0], cu_, sl_, pi_)
            return cache.at[page, slot // kvp, slot % kvp].set(kp)

        def w1_fn(cache, kv_, cu_, sl_, pi_):
            page, slot = indices(kv_.shape[0], cu_, sl_, pi_)
            return cache.at[page, slot // kvp, slot % kvp].set(kv_)

        def w2_fn(cache, ks_, cu_, sl_, pi_):
            page, slot = indices(ks_.shape[0], cu_, sl_, pi_)
            return cache.at[page, slot].set(ks_)

        def w12_fn(vc, sc, kv_, ks_, cu_, sl_, pi_):
            page, slot = indices(kv_.shape[0], cu_, sl_, pi_)
            vc = vc.at[page, slot // kvp, slot % kvp].set(kv_)
            sc = sc.at[page, slot].set(ks_)
            return vc, sc

        def w2b_fn(cache, ks_, cu_, sl_, pi_):
            page, slot = indices(ks_.shape[0], cu_, sl_, pi_)
            return cache.at[page, slot // 128, slot % 128].set(ks_)

        def w12b_fn(vc, sc, kv_, ks_, cu_, sl_, pi_):
            page, slot = indices(kv_.shape[0], cu_, sl_, pi_)
            vc = vc.at[page, slot // kvp, slot % kvp].set(kv_)
            sc = sc.at[page, slot // 128, slot % 128].set(ks_)
            return vc, sc

        _w0 = jax.jit(w0_fn, donate_argnums=(0,))
        _w1 = jax.jit(w1_fn, donate_argnums=(0,))
        _w2 = jax.jit(w2_fn, donate_argnums=(0,))
        _w12 = jax.jit(w12_fn, donate_argnums=(0, 1))
        _w2b = jax.jit(w2b_fn, donate_argnums=(0,))
        _w12b = jax.jit(w12b_fn, donate_argnums=(0, 1))

        n0 = jax.jit(lambda c: c.at[0, 0, 0, 0].set(1), donate_argnums=(0,))
        n1 = jax.jit(lambda c: c.at[0, 0, 0, 0].set(1), donate_argnums=(0,))
        n2 = jax.jit(lambda c: c.at[0, 0].set(1), donate_argnums=(0,))

        def rec(name, ms, spread, nbytes=None):
            r = dict(kind="writer", num_tokens=T, arm=name, ms=ms,
                     spread_pct=spread)
            if nbytes:
                r["implied_GBps_rw"] = 2 * nbytes / (ms * 1e-3) / 1e9
            rows.append(r)
            print(json.dumps(r), flush=True)
            return r

        wide_cache, ts = _time_chain(n0, wide_cache)
        rec("N0_noop_wide", *_median_spread(ts))
        val_cache, ts = _time_chain(n1, val_cache)
        rec("N1_noop_val", *_median_spread(ts))
        scale_cache, ts = _time_chain(n2, scale_cache)
        rec("N2_noop_scale", *_median_spread(ts))

        wide_cache, ts = _time_chain(_w0, wide_cache, k_packed, cu_j, sl_j,
                                     pi_j)
        w0 = rec("W0_today_wide", *_median_spread(ts), nbytes=n_pos * wide)
        val_cache, ts = _time_chain(_w1, val_cache, k_val, cu_j, sl_j, pi_j)
        w1 = rec("W1_val_only", *_median_spread(ts), nbytes=n_pos * narrow)
        scale_cache, ts = _time_chain(_w2, scale_cache, k_scale, cu_j, sl_j,
                                      pi_j)
        w2 = rec("W2_scale_only", *_median_spread(ts), nbytes=n_pos)

        import jax as _jax
        pair = (val_cache, scale_cache)
        pair = _jax.block_until_ready(_w12(*pair, k_val, k_scale, cu_j, sl_j,
                                           pi_j))
        for _ in range(WARMUP):
            pair = _jax.block_until_ready(
                _w12(*pair, k_val, k_scale, cu_j, sl_j, pi_j))
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for _ in range(INNER):
                pair = _w12(*pair, k_val, k_scale, cu_j, sl_j, pi_j)
            _jax.block_until_ready(pair)
            ts.append((time.perf_counter() - t0) / INNER)
        val_cache, scale_cache = pair
        w12 = rec("W12_split_both", *_median_spread(ts),
                  nbytes=n_pos * (narrow + 1))

        # --- the same two scatters against the 3-D scale plane -------------
        scale_cache3, ts = _time_chain(_w2b, scale_cache3, k_scale, cu_j,
                                       sl_j, pi_j)
        w2b = rec("W2b_scale_only_3d", *_median_spread(ts), nbytes=n_pos)

        pair = (val_cache, scale_cache3)
        pair = _jax.block_until_ready(_w12b(*pair, k_val, k_scale, cu_j, sl_j,
                                            pi_j))
        for _ in range(WARMUP):
            pair = _jax.block_until_ready(
                _w12b(*pair, k_val, k_scale, cu_j, sl_j, pi_j))
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for _ in range(INNER):
                pair = _w12b(*pair, k_val, k_scale, cu_j, sl_j, pi_j)
            _jax.block_until_ready(pair)
            ts.append((time.perf_counter() - t0) / INNER)
        val_cache, scale_cache3 = pair
        w12b = rec("W12b_split_both_3d", *_median_spread(ts),
                   nbytes=n_pos * (narrow + 1))

        d = dict(kind="writer_summary", num_tokens=T,
                 w0_ms=w0["ms"], w12_ms=w12["ms"],
                 w1_ms=w1["ms"], w2_ms=w2["ms"],
                 sum_w1_w2_ms=w1["ms"] + w2["ms"],
                 w2b_ms=w2b["ms"], w12b_ms=w12b["ms"],
                 delta_time_pct=(w12["ms"] - w0["ms"]) / w0["ms"] * 100.0,
                 speedup_x=w0["ms"] / w12["ms"],
                 delta_time_pct_3d=(w12b["ms"] - w0["ms"]) / w0["ms"] * 100.0,
                 speedup_x_3d=w0["ms"] / w12b["ms"])
        rows.append(d)
        print(json.dumps(d), flush=True)

    print("\n=== writer-side scatter, production cache size ===")
    print("delta_time% = (W12 - W0)/W0; NEGATIVE IS FASTER. This number is "
          "NOT netted against the reader-side win.")
    print("W12 uses the 2-D scale plane (reproduces j-ce74a9f2); W12b uses "
          "the 3-D plane the reader needs. Same bytes either way.")
    hdr = (f"{'T':>6}{'W0_ms':>10}{'W1_ms':>10}{'W2_ms':>10}{'W12_ms':>10}"
           f"{'dtime%':>9}{'spdup':>8}{'W12b_ms':>10}{'dtime%_3d':>11}"
           f"{'spdup_3d':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("kind") != "writer_summary":
            continue
        print(f"{r['num_tokens']:>6}{r['w0_ms']:>10.4f}{r['w1_ms']:>10.4f}"
              f"{r['w2_ms']:>10.4f}{r['w12_ms']:>10.4f}"
              f"{r['delta_time_pct']:>9.2f}{r['speedup_x']:>8.3f}"
              f"{r['w12b_ms']:>10.4f}{r['delta_time_pct_3d']:>11.2f}"
              f"{r['speedup_x_3d']:>10.3f}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
