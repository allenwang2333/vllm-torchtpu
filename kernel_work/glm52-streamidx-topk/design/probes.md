# Wave 3 — probe results

What the hardware said. Five probes were commissioned by `combined.md`'s final section; **all five
ran**. `P-GATE` was excluded by the caller (no approximation is on the table). Two arms of `P-ITER`
were deliberately not run and are named in that section.

This document chooses no design and ranks nothing. Where a measurement contradicts a derived claim
in `combined.md`, the contradiction is stated with the arithmetic; where it confirms one, that is
stated too.

## Conventions, binding on every number below

- **Provenance tags.** `measured (job id)` / `modelled` / `assumed` / `unmodelled`. A term valued at
  zero because nothing was measured for it is `unmodelled`, never `0`.
- **Sign.** Every percentage is a change in **time**; negative is faster. Signed percentages appear
  only in a `Δ time %` column. Prose says the direction in words.
- **Paired deltas only.** Every `Δ time %` was computed *inside* a repeat against the reference arm
  measured microseconds earlier, then medianed over 5 repeats. `Δ spread` is the range of those 5
  per-repeat deltas and is the resolution floor: an effect smaller than its own spread is not a
  result. `across-repeat` is the arm's own drift over the 5 repeats and is *not* the error bar on
  the delta — the drift is common-mode and cancels in the pair.
- **`full − sc` is never formed.** Where a statement needs both levels it is a difference of two
  paired deltas (difference-in-differences), which is labelled as such.
- **Cross-shard absolutes are not comparable.** `j-69ee326d` ran three timing shards concurrently on
  one pod. Only arms *within* a shard are paired. No conclusion below rests on comparing a µs figure
  in one shard to a µs figure in another; the two walk-rate slopes in §P-WALK/§P-SCBW are each
  within-shard.

## Jobs

| job | `JobStatus` | contents | `RUN_CMD`, read back from `cdk job recipe <id> --no-color` |
| --- | --- | --- | --- |
| `j-cefba169-8b6e-4fc6-80f6` | Succeeded | P-P16, whole pod | `export INST_DIR=/root/inst && cd /root/probes && python3 -u p_p16.py` |
| `j-dcf1e707-14ff-40d0-b27b` | Succeeded | P-DECOMP, whole pod | `export INST_DIR=/root/inst && cd /root/probes && python3 -u p_decomp.py` |
| `j-69ee326d-e045-41fe-824a` | Succeeded | P-WALK / P-SCBW / P-ITER, 3 chips | `export INST_DIR=/root/inst && cd /root/probes && bash /root/fanout/tpu-fanout.sh "python3 -u /root/probes/p_walk.py" "python3 -u /root/probes/p_scbw.py" "python3 -u /root/probes/p_iter.py"` |

Image on all three: `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.
`j-69ee326d` fan-out exit codes: `shard0 rc=0`, `shard1 rc=0`, `shard2 rc=0`, `3/3 shards succeeded`.

**No shipping file was modified.** `streamindex_topk.py` and `sparsecore_topk.py` are untouched.
P-P16 needed a modified selector, so `probes/sc_nogrow.py` and `probes/sit_nogrow.py` are `cp`
copies under `kernel_work/glm52-streamidx-topk/probes/`; `diff` against the shipping files shows
exactly one changed hunk each (the growth guard, and the import that binds to it). Whole-module
copies rather than a monkey-patch because both entry points are `jax.jit` objects and a patched
module would have handed arm 2 arm 1's cached executable — the probe would have measured nothing.

---

## P-P16 — does `_pick_partition`'s growth branch cost time at b=16?

```
probe:    _pick_partition raises p to 2 at b=16 without reducing the subcore-wave
          count cdiv(b*p,32). Does the resulting second pl.kernel dispatch and its
          two XLA gathers cost measurable time, and how much?
config:   A-D16-8k and A-D32-8k adjacent, plus a suppressed-growth arm at b=16.
          tpu7x, 1 chip, whole pod to itself. Full op and sparsecore_topk alone.
          5 interleaved repeats x 20 iters. Selector arm = probes/sc_nogrow.py,
          which differs from the shipping selector only by
          `if _cdiv(b*p*2,32) >= _cdiv(b*p,32): break`.
job:      j-cefba169-8b6e-4fc6-80f6  (Succeeded)
answer:   growth off is faster at b=16 in 5/5 repeats, at both levels.
decided:  the growth branch is a real, located, isolated cost; combined.md's
          -9.0% prediction is confirmed in sign and location and is too small.
```

Build lines confirm the branch under test fired: `A-D32-8k nogrow=False: p=1`,
`A-D16-8k nogrow=False: p=2`, `A-D16-8k nogrow=True: p=1`.

**Full op**, all `measured (j-cefba169)`:

| arm | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: |
| `fA16` A-D16-8k, incumbent (p=2) | 265.9 | 7.0% | ref | — |
| `fA16n` A-D16-8k, growth off (p=1) | 230.1 | 3.2% | −13.7% | 5.5 pt |
| `fA32` A-D32-8k (p=1) | 250.0 | 3.8% | −6.0% | 7.1 pt |

Per-repeat deltas for `fA16n`: −14.57, −12.94, −13.70, −18.43, −12.94 %. All five negative.

**`sparsecore_topk` alone**, all `measured (j-cefba169)`:

| arm | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: |
| `sA16` incumbent (p=2) | 203.3 | 8.2% | ref | — |
| `sA16n` growth off (p=1) | 168.6 | 7.0% | −16.8% | 2.0 pt |
| `sA32` b=32 (p=1) | 169.2 | 6.1% | −16.1% | 3.0 pt |

Per-repeat deltas for `sA16n`: −16.75, −17.07, −17.34, −15.37, −15.73 %. Spread 2.0 pt against a
16.8% effect — the cleanest signal in this document.

**Correctness held.** Full op: identical selected *sets* on all 16 rows, symmetric-difference total
0 of 32768 slots. SC alone: identical sets. So this is a pure removal of work, not a change of
answer.

### What it decided

1. **`combined.md`'s −9.0% claim is confirmed in direction and location and is conservative in
   magnitude.** The predicted saving was 9.0% of the A-D16-8k baseline 278.7 µs = 25.1 µs
   `modelled`. Measured 35.8 µs = 13.7% `measured (j-cefba169)`. Every one of the five repeats came
   in faster than 12.9%, so −9.0% lies outside the measured range entirely. Use the measured figure.
2. **The whole win is the SparseCore half.** The full-op saving is 35.8 µs and the SC-alone saving
   is 34.7 µs — two *independent* paired deltas, measured in different arm sets, agreeing to 1.1 µs.
   (This is a cross-check of two paired deltas, not the banned `full − sc`.) There is no TC-side
   component to look for.
3. **The cost is the dispatch and the gathers, not the walk.** At A-D16 with p=2, stage 1 is 1 wave
   of `slice_len` 4608 and stage 2 is 1 wave of 4096; with p=1 it is 1 wave of 9216. The walk model
   in §P-WALK values that difference at 0.46 µs `modelled` in favour of p=2. It measures 34.7 µs
   against p=2. The second `pl.kernel` dispatch plus two `jnp.take_along_axis` gathers therefore
   cost **34.7 µs at this shape, `measured (j-cefba169)`** — a fixed cost with no term in M1.
4. **`M1` is confirmed on the pair it can see.** `sA16n` 168.6 and `sA32` 169.2 differ by 0.4% and
   have identical `X_pad` = 9216. Wave-padding does describe *that* comparison. It does not describe
   the third arm: M1 predicts p=2 to be marginally faster (`X_pad` 8704 → 176.0 µs `modelled` vs
   p=1's 9216 → 176.7 µs `modelled`); it measures 20.6% slower.
5. **Correction owed to `roofline.py:86-95`.** That comment concludes the A-D16 > A-D32 inversion
   "is inside this band and is NOT established". It **is** established: A-D32 was faster than A-D16
   in all five adjacent repeats, by 3.05% to 10.20%. And with growth suppressed the ordering
   reverses in all five — so the growth branch is the entire cause of the inversion, which the
   comment treats as unexplained noise.

### Caveat on this probe, and it is real

The SC-alone arms ran at n=9216, not the production 10240. `instruments/shapes.py:177` sets
`meta['n'] = pages_per_seq * page_size`, which omits the `align_to(pages_per_seq, bkv_p)` the kernel
applies — the same 11.1% error `combined.md` a0.2 already charges against `roofline.py:31`. The
full-op arms are unaffected (the kernel computes its own extent, and the build line shows the op
running at its own n). `_pick_partition` returns 2 at both 9216 and 10240, so the branch under test
is the right one and conclusions 1–3 stand; the **absolute** SC µs in the second table are for a
slice 10% narrower than production. `shapes.py` needs the fix before any further SC-alone number is
taken from it.

---

## P-DECOMP — where does the 16 ms B-P1024-1M / B-P1024-256k gap live?

```
probe:    combined.md a0.5 measures the gap across two jobs and closes 82% of it
          by charging the SparseCore's stage-1 read (4 GiB vs 1 GiB) at an ASSUMED
          337 GB/s. Is the gap real when measured adjacently, and is it SC-side?
config:   B-P1024-1M and B-P1024-256k, both rows in one process, four arms.
          The two SC arms share one 4 GiB score array, one executable, b=1024,
          n=1048576, p=16, identical X_pad = 34,603,008; only the i32[1024]
          row-lengths vector differs (16/16 live slices vs 4/16).
          tpu7x, whole pod to itself, 5 repeats x 10 iters.
job:      j-dcf1e707-14ff-40d0-b27b  (Succeeded)
answer:   the gap is real (+31.8%) and the SparseCore contributes to it with the
          WRONG SIGN: the 1M row's SC is 8.4% FASTER than the 256k row's.
decided:  a0.5's closure is refuted. The gap is TC-side. M1 must not be used as a
          length model at these rows.
```

All figures `measured (j-dcf1e707)`:

| arm | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: |
| full B-P1024-1M | 67090.8 | 0.2% | +31.8% | 0.5 pt |
| full B-P1024-256k | 50881.3 | 0.3% | ref | — |
| `sc` B-P1024-1M | 40007.8 | 0.6% | −8.4% | 0.6 pt |
| `sc` B-P1024-256k | 43633.9 | 0.3% | ref | — |

(The two tables are anchored on their own 256k arm; the `+31.8%` and the `−8.4%` are not against a
common reference and must not be subtracted as percentages. The µs are.)

### What it decided

1. **The gap survives the adjacency test.** 67090.8 − 50881.3 = **16209.5 µs**
   `measured (j-dcf1e707)`, at 0.2–0.3% drift and 0.5 pt delta spread — as clean as anything in this
   exercise. `combined.md` a0.5's cross-job figure of 16041.9 µs agrees to 1.1%. The
   `roofline.py:93` objection to that cross-job comparison was well founded as a matter of method
   and turns out not to have mattered here: the number was right.
2. **a0.5's closure is refuted, on two independent grounds.**
   - *Sign.* The closure needs the SC to be **slower** at 1M by roughly 3 GiB / 337 GB/s = 9559 µs
     `assumed`. The SC is **faster** at 1M, by 3626.1 µs `measured (j-dcf1e707)`. The term does not
     shrink; it changes sign.
   - *Premise.* The 4 GiB-vs-1 GiB read never happens. `sparsecore_topk.py:160` DMAs the full
     `slice_len` unconditionally, and both rows run at n=1048576 (printed by the build line for
     both), so both read 1024 × 1048576 × 4 B = 4 GiB `source`. There were never 3 GiB to save.
3. **The gap is outside `sparsecore_topk`.** Difference-in-differences of two paired deltas:
   16209.5 − (−3626.1) = **19835.6 µs** `measured (j-dcf1e707)` sits in scoring + fill, not in
   selection. This is a DiD of two clean paired deltas (spreads 0.5 pt and 0.6 pt), not the
   forbidden `full − sc`; it is stated with that provenance because it is the quantity wave 4 needs
   to size a TC-side change.
4. **The leading explanation is scoring FLOPs, and it is close to the machine.** The extra 786432
   keys × 1024 query tokens × 32 heads × 128 dims × 2 = 6.60e12 FLOPs `modelled`. Over the 16209.5
   µs full-op gap that is 407 TFLOP/s; over the 19835.6 µs DiD, 333 TFLOP/s. Both are within a
   factor of ~2–3 of a plausible tpu7x bf16 peak, so the gap is roughly *accounted for* by scoring
   arithmetic. The MXU peak itself is `unmodelled` here — I did not measure it — so this is an
   explanation with the right magnitude, not a closure. Wave 4 should get the peak from area 5
   before leaning on it.
5. **Limitation.** The SC arms use `synth_scores`, not the score matrix the TC actually produces.
   Shape, `p`, `X_pad` and the DMA are exact; the *value distribution* is a model of the real one
   (right-skewed, ReLU floor), and radix-select pass counts depend on the distribution. The sign of
   the SC result is robust — the 256k arm is slower in 5/5 repeats at 0.6 pt spread — but the 3626.1
   µs magnitude is distribution-dependent.

---

## P-WALK — does SparseCore time depend on walked words at fixed padded extent?

```
probe:    M1 (T_sc = C + f*X_pad) has no live-data term and therefore predicts
          exactly zero for E3 (block-max pruning). Is that true?
config:   LEVEL A: _sc_topk_direct, one f32[256,65536] array, ONE executable.
          Arms differ ONLY in the i32[256] row-lengths handed in as data, so
          slice_len (65536), num_waves (8), X_pad (524288) and the 64 MiB input
          DMA are byte-identical in every arm and only walk_chunks moves.
          LEVEL B: sparsecore_topk(f32[16,1048576]), p=16, X_pad = 557056,
          row_lengths 1M / 256k / 64k. tpu7x, shard0 of a 3-shard fan-out,
          5 repeats x 20 iters.
job:      j-69ee326d-e045-41fe-824a (Succeeded), shard0 rc=0
answer:   LEVEL A: time falls 52.3% as walked words fall 8x at fixed padding --
          M1's padding-only form is FALSE. But the dead-half arm shows the driver
          is the per-wave MAXIMUM row length, not the sum of walked words.
          LEVEL B: shorter rows are SLOWER, by 20.6% and 27.9%.
decided:  M1 is refuted as a length model. E3's value is NOT zero and NOT the sum
          of pruned words: it is worth something only where it shortens the
          LONGEST row in a wave, and at the two-stage p>1 path it is negative.
```

**Level A** — `X_pad` = 524288 and input bytes identical in all five arms. All `measured (j-69ee326d)`:

| arm | walked words | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| walk-100% (65536) | 16777216 | 794.5 | 7.1% | ref | — |
| walk-50% (32768) | 8388608 | 557.6 | 9.0% | −30.3% | 3.1 pt |
| walk-25% (16384) | 4194304 | 446.3 | 12.8% | −44.0% | 5.0 pt |
| walk-12.5% (8192) | 2097152 | 383.9 | 6.6% | −52.3% | 4.2 pt |
| dead-half (65536/0) | 8388608 | 768.6 | 4.1% | −3.4% | 5.5 pt |

**Level B** — two-stage p=16, `X_pad` = 557056 in all three. All `measured (j-69ee326d)`:

| arm | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: |
| p16-full (1M live, 16/16 slices) | 831.9 | 0.5% | ref | — |
| p16-quarter (256k live, 4/16) | 1003.2 | 1.1% | +20.6% | 1.0 pt |
| p16-16th (64k live, 1/16) | 1065.3 | 0.8% | +27.9% | 1.1 pt |

### What it decided

1. **M1's padding-only form is refuted.** M1 predicts 165.4 + 1.2237e-3 × 524288 = **806.9 µs
   `modelled` for every level-A arm**. It is right for walk-100% (+1.6%) and wrong by +110% for
   walk-12.5% (806.9 vs 383.9). The same at level B: M1 predicts 847.0 µs `modelled` for all three
   arms, which is +1.8% on p16-full and −20.5% on p16-16th.
2. **M1's coefficient `f` is the walk rate wearing a padding label.** M1's fitted
   f = 1.2237 ns per *padded* word `modelled` sits right on the two independently measured *walk*
   rates below (0.895 and 1.158 ns per walked word). M1 fits its corpus because that corpus is
   nearly all full-length rows, where padded ≈ walked. It has no predictive power the moment those
   diverge — which is precisely the regime every length-based optimization proposes to create.
3. **The driver is the per-wave MAXIMUM, not the sum.** `dead-half` walks exactly as many words as
   `walk-50%` (8388608) and costs 37.8% more (768.6 vs 557.6). It walks the same *maximum* as
   `walk-100%` (65536, since every wave of 32 subcores still contains 16 full-length rows) and
   measures within 3.4% of it — inside that arm's own 7.1% drift, i.e. indistinguishable. A wave
   costs what its longest row costs; dead lanes are free but buy nothing.

   The model this supports, call it **M2**, is `T = C_cfg + f_w · X_crit` with
   `X_crit = Σ_waves max_row_walked_words`. Fitting the level-A endpoints gives
   **f_w = 0.895 ns/walked word** and `C_cfg` = 325.2 µs `measured (j-69ee326d)`. Its interpolation
   is excellent — 559.9 predicted vs 557.6 measured at walk-50% (−0.4%), 442.5 vs 446.3 at walk-25%
   (+0.9%) — and it predicts the arm it was not fitted on: dead-half 794.4 predicted vs 768.6
   measured (−3.3%, inside drift), where the sum-of-walked-words model predicts 557.6 and is 27%
   wrong. **`C_cfg` is not portable**: it was fitted at 8 waves / b=256 and does not carry to other
   wave counts (see §P-SCBW). Only `f_w` should be reused.
4. **At the two-stage p>1 path, shortening rows makes the SparseCore SLOWER.** Level B is decisive
   (1.0–1.1 pt spread on 20.6% and 27.9% effects), and P-DECOMP reproduces the direction
   independently at b=1024 (256k slower by 9.1%). M2 explains half of why: with p=16 and rows
   shortened uniformly, the live slices are still full-length, so every wave's maximum is unchanged
   and **zero walk is saved**. The extra time on top is stage-2 degeneracy — stage 2 receives
   b×k×p = 32768 candidates of which most are `-1`/`-inf` padding, and the radix select runs against
   a distribution with a huge atom. That mechanism is `modelled`; the cost of it is
   `measured (j-69ee326d)` at +20.6% (b=16) and `measured (j-dcf1e707)` at +9.1% (b=1024).
5. **Consequence for E3 and for anything length-aware.** E3's value is not zero (as M1 said) and not
   proportional to pruned words (as the naive reading of level A would say). It is worth `f_w` per
   word **only where it shortens the longest row in a wave**, which means E3 is inseparable from a
   *packing* decision: rows must be grouped into waves by length or the maximum does not move. On
   the p>1 two-stage path with slices of one row, pruning without repacking is measurably negative.

---

## P-SCBW — what is the SparseCore's achieved HBM read bandwidth?

```
probe:    combined.md a0.5 charges the SC's read at an ASSUMED 337 GB/s (carried
          from area 5). What is it really?
config:   _sc_topk_direct, b=32 (exactly one subcore wave), row_lengths=16 so the
          walk is one chunk and only the input DMA scales. n in {4096, 16384,
          32768, 65536} = 16/64/128/256 KiB per subcore, 0.5/2/4/8 MiB aggregate.
          Plus an n=65536 FULLWALK control at row_lengths=n. tpu7x, shard1 of a
          3-shard fan-out, 5 repeats x 20 iters.
job:      j-69ee326d-e045-41fe-824a (Succeeded), shard1 rc=0
answer:   the DMA term is barely resolvable over a 16x byte range. Point estimate
          1212.5 GB/s; >= 591 GB/s even on the worst repeat. 337 GB/s is excluded
          by all five repeats.
decided:  a0.5's bandwidth assumption is wrong by 2-3.6x, in the direction that
          shrinks the term it was invoked to explain. Read bytes are not a
          first-order cost at these slice sizes; walked words are.
```

All `measured (j-69ee326d)`:

| arm | aggregate bytes/call | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| n=4096 (16 KiB/subcore) | 0.50 MiB | 140.1 | 5.6% | ref | — |
| n=16384 (64 KiB/subcore) | 2.00 MiB | 139.6 | 6.7% | +0.0% | 4.9 pt |
| n=32768 (128 KiB/subcore) | 4.00 MiB | 141.0 | 6.0% | −0.4% | 10.7 pt |
| n=65536 (256 KiB/subcore) | 8.00 MiB | 146.1 | 5.3% | +3.6% | 10.4 pt |
| n=65536 FULLWALK (control) | 8.00 MiB | 222.0 | 3.9% | +58.4% | 12.8 pt |

Least squares over the four DMA-only arms: slope 8.2471e-13 s/byte → **1212.5 GB/s**, intercept
138.6 µs, residuals ≤ 0.8% on every point.

### What it decided

1. **The 337 GB/s assumption is excluded.** At 337 GB/s the 0.5 MiB → 8 MiB step would add 23.3 µs
   `modelled`. The five per-repeat differences for that pair are +13.3, −1.2, +5.1, 0.0, +9.0 µs
   `measured (j-69ee326d)`, median +5.1. No repeat reached even 60% of the predicted effect. The
   worst repeat implies ≥ 591 GB/s; the fit implies 1212.5 GB/s.
2. **But read this as a bound, not as a bandwidth number.** The four DMA arms are individually
   *unresolvable* — the 10.4 pt delta spread on the +3.6% arm exceeds the effect. What the probe
   established is an upper bound on the per-byte cost, which is enough to kill a 337 GB/s charge and
   is not enough to publish a bandwidth. Quote it as "≥ ~590 GB/s `measured (j-69ee326d)`, point
   estimate ~1200 GB/s, low leverage". The 480 KiB per-subcore VMEM ceiling means no arm here
   exceeded 256 KiB, so nothing is known above that.
3. **The walk, by contrast, is enormous and clean.** FULLWALK vs the same-bytes DMA-only arm:
   222.0 − 146.1 = 75.9 µs for 65536 words = **1.158 ns/walked word `measured (j-69ee326d)`**. That
   is a second, independent estimate of §P-WALK's `f_w` = 0.895 ns/word, from a different arm set
   and a different wave count; they bracket M1's fitted 1.2237. **On an 8 MiB slice the walk costs
   more than the transfer that delivered it** — 75.9 µs of walk against a DMA term the probe could
   not even resolve. Every bytes-side optimization is therefore competing for a term that does not
   exist at this scale.
4. **`C_cfg` is wave-count-dependent, which bounds M2's reuse.** This shard's 1-wave intercept is
   138.6 µs; §P-WALK's 8-wave intercept is 325.2 µs. Those are in different shards so their absolute
   levels are contended and not directly comparable — but the gap is far too large to be
   co-tenancy alone, and it says plainly that the "C" in both M1 and M2 grows with wave count and
   must not be treated as one constant across configurations. How it grows is `unmodelled`.

---

## P-ITER — is the in-body fixed cost per iteration, per copy, or per index computation?

```
probe:    combined.md a1 has three hypotheses predicting opposite levers. Which?
config:   B-D1-1M, num_kv_pages_per_block=(bp,bp,bp) for bp in {2,4,8,16}, one
          built input shared by all four arms. Held CONSTANT by construction:
          total make_async_copy count 2048, KV bytes 256 MiB, bytes/copy 128 KiB,
          and score n = align_to(2048,bp)*512 = 1048576 -- so the SparseCore half
          of the op does not move at all. Varied: iterations, 1024/512/256/128.
          tpu7x, shard2 of a 3-shard fan-out, 5 repeats x 20 iters.
job:      j-69ee326d-e045-41fe-824a (Succeeded), shard2 rc=0
answer:   time falls monotonically and linearly with iteration count at 0.500 us
          per iteration. Per-copy-only is refuted.
decided:  the in-body fixed cost is PER ITERATION. Raising bkv_p is a real lever
          worth up to 31.4% at (B) -- with a shape-dependent cost at (A).
```

All `measured (j-69ee326d)`:

| arm | iterations | copies/iter | med µs | across-repeat | Δ time % | Δ spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bkv_p=2 | 1024 | 2 | 1429.6 | 1.3% | ref | — |
| bkv_p=4 | 512 | 4 | 1167.6 | 0.4% | −18.5% | 0.9 pt |
| bkv_p=8 | 256 | 8 | 1037.0 | 0.8% | −27.5% | 0.6 pt |
| bkv_p=16 | 128 | 16 | 981.4 | 0.7% | −31.4% | 1.4 pt |

**Correctness held**: the selected sets are identical to the bkv_p=2 arm for all three other arms,
and `filled row0=2048/2048` everywhere.

### What it decided

1. **Per-iteration, decisively; per-copy alone is refuted.** The per-copy hypothesis predicts a flat
   table (copies were held at 2048 in every arm) and the table falls 31.4%.
2. **The rate is 0.500 µs per iteration `measured (j-69ee326d)`**, from
   (1429.6 − 981.4)/(1024 − 128) = 0.5002. The fit is close to exact on the two arms it was not
   fitted on: 1173.4 predicted vs 1167.6 measured at 512 iterations (−0.5%), 1045.4 vs 1037.0 at 256
   (−0.8%). Intercept 917.4 µs. **This confirms area 3's 512 ns grid-step constant to within 2.3%**
   and transfers it from the grid step to the `fori_loop` body.
3. **Sizing the headroom at (B).** At B-D1-1M with the shipping bkv_p=2, loop overhead is
   1024 × 0.500 = 512.2 µs of a 1429.6 µs op = **35.8% `measured (j-69ee326d)`**. Going to bkv_p=16
   recovers 448.2 µs of it.
4. **The lever is not free and the cost is at the other shape family.** `n` is
   `align_to(pages_per_seq, bkv_p) · page_size`. At (B) with pages_per_seq = 2048 the alignment is a
   no-op for every bkv_p tested, which is why this probe could hold n fixed. At (A), pages_per_seq
   is 9: bkv_p=2 gives 10 → n = 10240, but bkv_p=16 gives 16 → n = 16384, a 60% wider padded score
   extent `source` feeding a SparseCore whose cost §P-WALK just showed is driven by exactly that
   extent when rows are full. **bkv_p must be chosen per shape family, not globally.** The cost of
   bkv_p=16 at (A) is `unmodelled` — no arm measured it.
5. **It does not explain the (B) prefill gap.** At B-P1024, seqlen 1M vs 256k differs by 768
   iterations at bkv_p=2 → 384 µs `modelled` at this rate, which is 2.4% of P-DECOMP's 16209.5 µs
   gap. Iteration count is not where that gap lives.

### Arms deliberately not run, and why

- **a1 arm 4 (coalesced strided `make_async_copy`)** — legal only when the pages of a block are
  contiguous in the cache. The synthetic `page_indices` this instrument builds *are* contiguous and
  production's are not, so a measurement here would over-claim by construction. It needs a probe
  with realistic fragmentation, which was out of budget.
- **a1 arm 5 (host-precomputed byte-offset table)** — the arm that would separate area 7's scalar
  index-arithmetic hypothesis from the other two. It needs kernel surgery plus the `uint16`
  `page_indices` narrowing of `combined.md` b3. **Area 7's hypothesis is therefore neither confirmed
  nor refuted** — note that it predicts a *flat* table here (index arithmetic is per copy, and
  copies were constant), so this probe was never able to discriminate it, and the 0.500 µs/iteration
  finding does not exclude an additional per-copy term. Sizing that term is open.

---

## Corrections owed to earlier artifacts

Listed so they are not lost; making the edits is wave 4's call, not mine.

| artifact | claim | status |
| --- | --- | --- |
| `roofline.py:86-95` | the A-D16 > A-D32 inversion "is NOT established" | **established** `measured (j-cefba169)`, 5/5 repeats, and fully explained by the `_pick_partition` growth branch |
| `instruments/shapes.py:177` | `meta['n'] = pages_per_seq * page_size` | wrong by the missing `align_to(pages_per_seq, bkv_p)`; 9216 vs 10240 at (A), the same 11.1% error a0.2 charges against `roofline.py:31` |
| `combined.md` a0.3, model M1 | `T_sc = C + f·X_pad`, padding-only | **refuted as a length model** (§P-WALK). Survives only where rows are full-length, where `X_pad = X_crit`. Its `f` = 1.2237 ns/word is really a walk rate. |
| `combined.md` a0.5 | 337 GB/s SC read `assumed` | **excluded** by all five repeats; ≥ ~590 GB/s, point estimate ~1200 |
| `combined.md` a0.5 | 82% of the B-P1024 gap is the SC's larger read | **refuted twice over**: the extra read does not happen (`sparsecore_topk.py:160` copies the full slice regardless of length, and both rows run at n=1048576), and the SC measures *faster* at 1M by 3626.1 µs |
| `combined.md` P-P16 | `Δ time −9.0%` at A-D16-8k | **confirmed in sign and location, too small in magnitude**: −13.7% measured, and every repeat beat −12.9% |

## What is still open

- **The MXU peak** that §P-DECOMP's 407 TFLOP/s should be judged against — `unmodelled`.
- **A per-copy term** in the in-body loop, on top of the 0.500 µs/iteration — a1 arm 5 not run.
- **bkv_p=16 at shape family (A)**, where the `align_to` cost lands — `unmodelled`.
- **How the SparseCore's fixed cost scales with wave count** — bracketed at 138.6 µs (1 wave) and
  325.2 µs (8 waves) but in different shards, so the shape of the growth is `unmodelled`.
- **Stage-2 degeneracy**: the mechanism behind the p>1 path getting slower on short rows is
  `modelled` only. Its cost is measured at two points (+20.6% at b=16, +9.1% at b=1024).
