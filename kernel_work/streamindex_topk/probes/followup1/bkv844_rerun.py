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
"""Re-run the one `bkv_p` tuple row that came back unusable.

Job `j-b54e9716-3db5-4f1f-b8d4` swept `num_kv_pages_per_block` tuples at the
production regimes. Its `mixed_prod` table was:

    tuple      full_ms                 spread %     d time %
    (2,2,2)    0.6724997947458178         2.83       (baseline)
    (4,4,4)    0.6789957056753336         1.16         +0.97
    (8,4,4)    0.7057664450258017       491.23  <--     +4.95  <-- unusable
    (8,8,8)    0.7232144766021520         2.26         +7.54

A 4.95% effect measured against a 491.23% spread is not a measurement of
anything. The three neighbouring rows sit between 1.16% and 2.83% spread, so
491% is one arm misbehaving, not the regime being noisy. This re-runs **only**
that comparison -- `(2,2,2)` against `(8,4,4)` at `mixed_prod` -- and tries to
get the spread under the effect.

What is done differently, and why
---------------------------------
The instrument's spread statistic is `(max - min) / median`. It is a **range**,
so it can only grow as reps are added: "more repeats" alone cannot drive it
under an effect, and a single host-side outlier fixes it high forever. Adding
reps is still right -- it makes the *median* trustworthy -- but the spread has
to be read alongside a statistic that is not a range, or the arm can never be
declared usable no matter how good it is.

So this reports three things per arm and does not let any one of them stand
alone:

  * the instrument's own `(max - min) / median` spread, on the same footing as
    the published table, so this run is comparable to it;
  * robust dispersion -- IQR as a percentage of the median, and the median
    absolute deviation -- which a single outlier cannot inflate;
  * the **paired** per-block delta. Both arms are timed inside the same block,
    alternating which goes first, so a host hiccup or a thermal drift that
    lands in one block hits both arms and cancels in the difference. The
    per-block delta distribution, its median, its IQR and a bootstrap interval
    are what actually decide whether 4.95% is real.

Also: both arms are compiled once, up front, and the inputs are built **once**
and shared, so no arm pays a build or a compile inside its timed region.

`inner` comes from the read-only `microbench._inner_for`, which derives it from
`S.scores_bytes(num_tokens, max_model_len)` -- the *default* `bkv_ps`. So it is
the same for both arms, which is what a comparison needs; it also means the
in-flight budget is computed against the narrower arm's buffer. That was
checked as a possible cause of the 491% and it is not one: at `mixed_prod` the
budget gives `inner = 20` and the outstanding bytes are 20 x 41.9 MB for
`(2,2,2)` and 20 x 67.1 MB for `(8,4,4)`, both far under the 1.5 GB cap.

Fidelity to the original measurement
------------------------------------
`--mode repro` runs each arm through the read-only `microbench.run_regime` with
`early_exit=False`, exactly as `probes/bkv_p_prod.py --mode sweep` did, so the
published numbers have something directly comparable to be checked against
before any of the new statistics are read. `--mode paired` is the new
alternating-block measurement. `--mode both` (the default) does the repro first
and then the paired run, in one process.

Sign convention: every percentage is a change in TIME, `(new - old) / old`,
with `(2,2,2)` as old. NEGATIVE IS FASTER. A `speedup x` figure is a different
number (>1 is faster).

This is a measurement of an existing sweep row. It is not a proposal to ship
`(8,4,4)`.
"""

import argparse
import importlib
import json
import random
import statistics
import sys
import time

import jax

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import microbench as MB  # noqa: E402

REGIME = "mixed_prod"
BASE_TUPLE = (2, 2, 2)
CAND_TUPLE = (8, 4, 4)

# The published row this re-run exists to replace. Quoted, not recomputed.
PUBLISHED = {
    (2, 2, 2): dict(full_ms=0.6724997947458178, spread=2.83, d=0.0),
    (4, 4, 4): dict(full_ms=0.6789957056753336, spread=1.16, d=+0.97),
    (8, 4, 4): dict(full_ms=0.7057664450258017, spread=491.23, d=+4.95),
    (8, 8, 8): dict(full_ms=0.7232144766021520, spread=2.26, d=+7.54),
}


class tuple_set:
    """Rebind `shapes.NUM_KV_PAGES_PER_BLOCK`, restore in `__exit__`.

    Same mechanism as `probes/bkv_p_prod.py`, and for the same reason: the
    instruments read the module attribute and are read-only, so the constant is
    changed where they read it rather than by forking them.
    """

    def __init__(self, tup):
        self.old = S.NUM_KV_PAGES_PER_BLOCK
        self.new = tuple(int(x) for x in tup)

    def __enter__(self):
        S.NUM_KV_PAGES_PER_BLOCK = self.new
        return self.new

    def __exit__(self, *exc):
        S.NUM_KV_PAGES_PER_BLOCK = self.old
        return False


def widths(regime, tup):
    """`num_sublanes_total` MUST be recomputed with an explicit `bkv_ps`.

    `shapes.num_sublanes_total` takes `bkv_ps` as a DEFAULT argument, bound at
    function-definition time, so rebinding the module attribute does not reach
    it. Reading the default would report the shipped tuple's width for every
    arm and hide the widened buffer entirely.
    """
    subl = S.num_sublanes_total(regime.max_model_len, bkv_ps=tup)
    return subl, regime.num_tokens * subl * 128 * 4 / 1e6


def make_call(impl, regime, tup, early_exit):
    return jax.jit(lambda q, w, c, s, p, cu, dist: impl.streamindex_topk(
        q, w, c, s, p, cu, dist,
        k=S.INDEX_TOPK,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=tuple(tup),
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=early_exit,
    ))


def stats(ts_s):
    """Everything the report needs about one arm's sample, in ms."""
    ms = sorted(t * 1e3 for t in ts_s)
    n = len(ms)
    med = statistics.median(ms)
    q1 = statistics.median(ms[:n // 2])
    q3 = statistics.median(ms[(n + 1) // 2:])
    mad = statistics.median([abs(x - med) for x in ms])
    return dict(
        n=n, median_ms=med, mean_ms=statistics.fmean(ms),
        min_ms=ms[0], max_ms=ms[-1],
        # the instrument's own statistic, so this run is comparable to the
        # published table
        spread_pct=(ms[-1] - ms[0]) / med * 100.0,
        iqr_ms=q3 - q1, iqr_pct=(q3 - q1) / med * 100.0,
        mad_ms=mad, mad_pct=mad / med * 100.0,
        stdev_pct=(statistics.pstdev(ms) / med * 100.0) if n > 1 else 0.0,
    )


def boot_ci(vals, reps=4000, lo=2.5, hi=97.5, seed=7):
    rng = random.Random(seed)
    n = len(vals)
    meds = sorted(statistics.median([vals[rng.randrange(n)] for _ in range(n)])
                  for _ in range(reps))
    return (meds[int(lo / 100.0 * reps)], meds[min(reps - 1,
                                                   int(hi / 100.0 * reps))])


def mode_repro(impl, regime, reps):
    print("\n=== mode=repro: each arm through the read-only "
          "microbench.run_regime, early_exit=False, exactly as "
          "probes/bkv_p_prod.py --mode sweep did ===", flush=True)
    out = {}
    for tup in (BASE_TUPLE, CAND_TUPLE):
        with tuple_set(tup):
            r = MB.run_regime(impl, regime, early_exit=False, reps=reps,
                              measure_topk=False)
        subl, mb = widths(regime, tup)
        r.update(bkv_tuple=list(tup), sublanes=subl, scores_MB=mb)
        out[tup] = r
        print(json.dumps(r), flush=True)

    old, new = out[BASE_TUPLE]["full_ms"], out[CAND_TUPLE]["full_ms"]
    hdr = (f"{'tuple':>10}{'subl':>6}{'scoresMB':>10}{'inner':>7}{'reps':>6}"
           f"{'full_ms':>11}{'spread %':>10}{'d time %':>10}"
           f"{'published ms':>14}{'published sp%':>15}")
    print(hdr)
    print("-" * len(hdr))
    for tup in (BASE_TUPLE, CAND_TUPLE):
        r = out[tup]
        d = (r["full_ms"] - old) / old * 100.0
        p = PUBLISHED[tup]
        print(f"{str(tup):>10}{r['sublanes']:>6}{r['scores_MB']:>10.1f}"
              f"{r['inner']:>7}{r['reps']:>6}{r['full_ms']:>11.5f}"
              f"{r['full_spread_pct']:>10.2f}{d:>10.2f}"
              f"{p['full_ms']:>14.5f}{p['spread']:>15.2f}")
    print("d time % is against (2,2,2). NEGATIVE IS FASTER.")
    print(f"The published (8,4,4) spread was {PUBLISHED[CAND_TUPLE]['spread']}%; "
          f"this run's is {out[CAND_TUPLE]['full_spread_pct']:.2f}%. If it "
          f"reproduces, the arm is genuinely unstable; if it does not, the "
          f"published 491.23% was a one-off host or allocator outlier.")
    return out


def mode_paired(impl, regime, blocks, inner, early_exit, seed):
    print(f"\n=== mode=paired: {blocks} alternating blocks, both arms in one "
          f"process, inputs built once, both arms compiled once ===",
          flush=True)

    d = I.build(regime)
    args = (d["q"], d["indexer_weights"], d["cache_kv"], d["seq_lens"],
            d["page_indices"], d["cu_q_lens"], d["distribution"])

    calls = {t: make_call(impl, regime, t, early_exit)
             for t in (BASE_TUPLE, CAND_TUPLE)}
    for t, fn in calls.items():
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        print(f"compiled {t} in {time.perf_counter()-t0:.1f}s "
              f"(sublanes {widths(regime, t)[0]}, "
              f"{widths(regime, t)[1]:.1f} MB scores)", flush=True)
    for t, fn in calls.items():
        for _ in range(MB.WARMUP):
            jax.block_until_ready(fn(*args))

    def one_block(fn):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = fn(*args)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / inner

    samples = {BASE_TUPLE: [], CAND_TUPLE: []}
    deltas = []
    for b in range(blocks):
        # Alternate which arm runs first so any within-block drift -- a clock
        # ramp, an allocator warm-up -- does not land systematically on one arm.
        order = ((BASE_TUPLE, CAND_TUPLE) if b % 2 == 0
                 else (CAND_TUPLE, BASE_TUPLE))
        got = {}
        for t in order:
            got[t] = one_block(calls[t])
            samples[t].append(got[t])
        o, n = got[BASE_TUPLE], got[CAND_TUPLE]
        deltas.append((n - o) / o * 100.0)
        if (b + 1) % 10 == 0 or b == 0:
            print(f"  block {b+1}/{blocks}: base {o*1e3:.5f} ms, "
                  f"cand {n*1e3:.5f} ms, d {deltas[-1]:+.2f}%", flush=True)

    st = {t: stats(samples[t]) for t in samples}

    print("\n--- per-arm, all blocks pooled ---")
    hdr = (f"{'tuple':>10}{'subl':>6}{'n':>5}{'median ms':>12}{'min ms':>11}"
           f"{'max ms':>11}{'spread %':>10}{'IQR %':>8}{'MAD %':>8}")
    print(hdr)
    print("-" * len(hdr))
    for t in (BASE_TUPLE, CAND_TUPLE):
        s = st[t]
        print(f"{str(t):>10}{widths(regime,t)[0]:>6}{s['n']:>5}"
              f"{s['median_ms']:>12.5f}{s['min_ms']:>11.5f}"
              f"{s['max_ms']:>11.5f}{s['spread_pct']:>10.2f}"
              f"{s['iqr_pct']:>8.2f}{s['mad_pct']:>8.2f}")
    print("`spread %` is the instrument's (max-min)/median, a RANGE: it can "
          "only grow with n, so it is quoted for comparability with the "
          "published table and is not the statistic the verdict rests on. "
          "IQR % and MAD % are robust.")

    old, new = st[BASE_TUPLE]["median_ms"], st[CAND_TUPLE]["median_ms"]
    unpaired_d = (new - old) / old * 100.0

    ds = sorted(deltas)
    nb = len(ds)
    dmed = statistics.median(ds)
    dq1 = statistics.median(ds[:nb // 2])
    dq3 = statistics.median(ds[(nb + 1) // 2:])
    lo, hi = boot_ci(deltas, seed=seed)
    n_slower = sum(1 for x in deltas if x > 0)

    print("\n--- the comparison ---")
    hdr2 = (f"{'old ms (2,2,2)':>16}{'new ms (8,4,4)':>17}"
            f"{'d time %':>11}{'speedup x':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    print(f"{old:>16.5f}{new:>17.5f}{unpaired_d:>11.2f}{old/new:>11.4f}")
    print("d time % = (new - old)/old, NEGATIVE IS FASTER. speedup x = "
          "old/new, >1 is faster. Both computed from the two medians in this "
          "same row.")

    print(f"\npaired per-block delta, n = {nb} blocks:")
    print(f"  median {dmed:+.2f}%   IQR [{dq1:+.2f}, {dq3:+.2f}] "
          f"(width {dq3-dq1:.2f} pp)")
    print(f"  min {ds[0]:+.2f}%   max {ds[-1]:+.2f}%")
    print(f"  bootstrap 95% CI on the median delta: "
          f"[{lo:+.2f}, {hi:+.2f}] %")
    print(f"  blocks where (8,4,4) was slower: {n_slower}/{nb} "
          f"({n_slower/nb*100:.0f}%)")

    # The verdict, stated by rule rather than by eye.
    eff = abs(dmed)
    worst_spread = max(st[BASE_TUPLE]["spread_pct"],
                       st[CAND_TUPLE]["spread_pct"])
    worst_iqr = max(st[BASE_TUPLE]["iqr_pct"], st[CAND_TUPLE]["iqr_pct"])
    ci_excludes_zero = (lo > 0) or (hi < 0)
    print("\n--- verdict ---")
    print(f"  effect |median paired delta| = {eff:.2f}%")
    print(f"  worst arm range spread       = {worst_spread:.2f}%")
    print(f"  worst arm IQR                = {worst_iqr:.2f}%")
    print(f"  bootstrap CI excludes zero   = {ci_excludes_zero}")
    if not ci_excludes_zero:
        print("  => UNUSABLE. The bootstrap interval on the paired delta "
              "spans zero, so this run cannot say which tuple is faster at "
              "mixed_prod. Report it as unusable; do not quote a number.")
    elif eff <= worst_iqr:
        print("  => WEAK. The interval excludes zero but the effect is at or "
              "below the robust dispersion of the arms it is a difference "
              "of. It cannot support a ranking on its own.")
    else:
        print("  => USABLE. The paired interval excludes zero and the effect "
              "exceeds the robust dispersion of both arms. The range spread "
              "above may still be large; a range is not a reason to discard "
              "a paired result, but quote both.")
    print("\nWhichever it is: this prices an existing sweep row. It is not a "
          "proposal to ship (8,4,4), whose 128-sublane scores buffer is 1.6x "
          "the shipped tuple's 80.")

    return dict(arms={str(k): v for k, v in st.items()}, deltas=deltas,
                paired_median_pct=dmed, paired_iqr=[dq1, dq3],
                boot_ci=[lo, hi], unpaired_d_time_pct=unpaired_d,
                speedup_x=old / new, blocks=nb, inner=inner,
                usable=bool(ci_excludes_zero and eff > worst_iqr))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl",
                    default="vllm_torchtpu.kernels.deepseek_v4.streamindex_topk")
    ap.add_argument("--mode", choices=["repro", "paired", "both"],
                    default="both")
    ap.add_argument("--blocks", type=int, default=60,
                    help="alternating paired blocks; each block times both "
                         "arms once")
    ap.add_argument("--reps", type=int, default=40,
                    help="reps for the run_regime repro pass (published run "
                         "used 10)")
    ap.add_argument("--inner", type=int, default=0,
                    help="0 = take microbench._inner_for(regime), which is "
                         "what the published run used")
    ap.add_argument("--early-exit", default="false",
                    choices=["false", "true"],
                    help="the published sweep used false; keep it false")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    impl = importlib.import_module(a.impl)
    regime = S.REGIMES_BY_NAME[REGIME]
    inner = a.inner or MB._inner_for(regime)
    ee = (a.early_exit == "true")

    print(f"regime={regime.name} phase={regime.phase} "
          f"max_num_seqs={regime.max_num_seqs} num_tokens={regime.num_tokens} "
          f"kv_len={regime.kv_len} max_model_len={regime.max_model_len}")
    print(f"shipped NUM_KV_PAGES_PER_BLOCK={S.NUM_KV_PAGES_PER_BLOCK}  "
          f"base={BASE_TUPLE} cand={CAND_TUPLE}  early_exit={ee}  "
          f"inner={inner}  blocks={a.blocks}  repro reps={a.reps}")
    for t in (BASE_TUPLE, CAND_TUPLE):
        subl, mb = widths(regime, t)
        print(f"  {t}: num_sublanes_total={subl}  scores={mb:.1f} MB  "
              f"in-flight at inner={inner}: {mb*inner/1e3:.2f} GB "
              f"(budget {MB.INFLIGHT_BUDGET_BYTES/1e9:.2f} GB)")

    out = {}
    if a.mode in ("repro", "both"):
        out["repro"] = {str(k): v for k, v in
                        mode_repro(impl, regime, a.reps).items()}
    if a.mode in ("paired", "both"):
        out["paired"] = mode_paired(impl, regime, a.blocks, inner, ee, a.seed)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
