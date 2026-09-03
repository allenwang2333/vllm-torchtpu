# R1b — The gate at the `bkv_p` we now ship, and the changed consumer

R1 answered Flip 1 at `bkv_p = 2` and found Option A short of its gate. Two things have changed
since, and both are measured here.

1. **The shipping `bkv_p` moved.** Candidate 0 selects `bkv_p = cdiv(ctx, page_size)` — **9** at
   ctx 8704 (`bkvp-extended.md` §4). R1's own mechanism finding says the gate should behave
   differently there, and R1 said so explicitly rather than extrapolating. **Task 1 measures it.**
2. **R1 named the changed consumer as the largest unclosed hole in Option A's case.** The design
   asserts at `measurement_and_design.md:714-716` that "the unpack does not change". It does change.
   **Task 2 measures both halves and gives the net.**

**One job, `j-18e51a27-db20-4088-9fb4`, `Succeeded`.** Three phases chained with `&&`, strictly
serial (init timestamps 23 s and 15 s apart), **not fanned out** — one container holding all four
chips (`google.com/tpu: "4"`, `TPU_TOPOLOGY: 2x2x1`), nothing pinned, `num_devices: 8` in each
phase. As in R1, an uncontended pod means **the times below are absolutes, not only ratios.**
Image `…vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`, device `TPU7x`,
jax `0.10.2`, best-of-9, grid 128.

`RUN_CMD` as read back from `cdk job recipe` (YAML line folds rejoined):

```
timeout 1200 python /root/kw/probes/probe_kpack_consumer.py --num-grid 8 --bkv-p 9 --trials 2
  --no-linearity-check && echo ==CSMOKE_OK== && timeout 2400 python
  /root/kw/instruments/bench_kpack.py --suite deep --num-grid 128 --trials 9 && echo ==DEEP128_OK==
  && timeout 5400 python /root/kw/probes/probe_kpack_consumer.py --num-grid 128 --trials 9
  && echo ==CONS128_OK==
```

All three markers printed. No `BUILD FAIL`, no `RUN FAIL`, no `ELISION WARNING`, no `MISMATCH`, no
traceback; every arm `"ok": true`.

**Scripts.** Task 1 extends my own `instruments/bench_kpack.py` with a `deep` suite. Task 2 is a new
file, `probes/probe_kpack_consumer.py`. Nothing pre-existing in `instruments/` was touched and no
file under `src/` was modified.

**Raw data:** `R1_kpack_deep.json`, `R1b_consumer.json`, `R1b_consumer_grid8.json`. Every figure
below was recomputed from those files, not copied from the probe's printed summary lines.

---

## 0. Verdict

| | |
| --- | --- |
| **Task 1 — the gate at `bkv_p` 8 and 9** | **PASSES**, under all three readings of the design's self-contradictory gate text. 47.0 % and 47.9 % saved as measured; **49.2 % and 49.9 %** once the dispatch floor is removed, i.e. the indirect, clamped DMA is **fully byte-proportional** at 36 in flight. R1's "A drops to rank 4" verdict is reversed *on this criterion*. |
| **Task 2 — the changed consumer** | **The net is positive, and it is positive for a reason nobody predicted.** Option A's stage-A inner arithmetic is not more expensive; it is **973.9 ns/block cheaper** at `bkv_p = 9`. |
| **Task 2 — the sting** | At `bkv_p = 9` the halved bytes are worth **≈ nothing on the critical path**: 94.3 % of the incumbent's fetch is already hidden behind compute. The entire measured net is the layout change's effect on the *consumer*, not on the DMA. |
| **Correctness** | **Bit-identical.** Same output checksum, exactly, at both `bkv_p` and both A consumers, from byte-equivalent sources. |
| **Absolute stake** | **15.30 µs** on the decode call with A on top of `bkv_p = 9`. |
| **Against the requirement** | The bar at the corrected denominator is **18.64-31.55 µs**. A delivers 15.30 µs. **Still short of its own −13 % floor**, by ~18 %. |

Option A is a real, correct, positive-value change that is **smaller than advertised and smaller
than its floor**, and whose value comes from the half of the change the design never costed.

**This document was reviewed adversarially (`/critical-review`, clean context, recomputing from the
JSON) before it was committed.** Every figure reproduced; four substantive corrections were made and
are marked **[rev]** where they land. The review's disposition: `PLAUSIBLE-UNCONFIRMED` →
**`provisional`**, and **`⚠️ unverified`** on correctness, because §3.4's layout-equivalence check
lives inside the probe and is not a named test job against the real kernel.

---

## 1. Task 1 — the charge ratio at `bkv_p` 8 and 9

Same `indirect` form as R1, same grid 128, best-of-9, `num_bkv = 2` throughout so pipeline depth is
constant and only the DMAs in flight per wait change. Charge ratio ≡ `t(262144 B) / t(132096 B)` at
**the same DMA count** — verified equal per pair in the recomputation.

| `bkv_p` | in flight | DMAs | t(256 B rec) | t(132 B rec) | **charge ratio** | **saved** | % of perfect proportionality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 24 | 6144 | 477.37 µs | 275.01 µs | 1.736 | **42.4 %** | 87.5 % |
| **8** | **32** | 8192 | 633.41 µs | 335.50 µs | **1.888** | **47.0 %** | 95.1 % |
| **9** | **36** | 9216 | 712.20 µs | 370.73 µs | **1.921** | **47.9 %** | 96.8 % |

Perfect byte-proportionality would be 262144/132096 = **1.9845**. Medians track the bests to within
0.4 % (1.737 / 1.895 / 1.925); worst per-arm spread is 0.85 %.

**[rev] The table above is *not* floor-corrected, and it should be read alongside the version that
is.** `bench_kpack.py` printed its own warning into the job log — *"the fastest arm is 275.01 µs
against a 28.19 µs empty-launch floor (10.3 %) … a fixed per-call cost is compressing the ratio
toward 1.0 and the gate is biased AGAINST Option A"* — and it is right. Subtracting the 28.19 µs
floor from both arms of each pair:

| `bkv_p` | corrected t(256) | corrected t(132) | **charge ratio** | **saved** | % of perfect proportionality |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 449.18 µs | 246.82 µs | 1.820 | 45.1 % | 91.7 % |
| **8** | 605.22 µs | 307.31 µs | **1.969** | **49.2 %** | 99.2 % |
| **9** | 684.01 µs | 342.54 µs | **1.997** | **49.9 %** | **100.6 %** |

So the honest statement at `bkv_p = 9` is not "96.8 % byte-proportional" but **byte-proportional,
full stop** — the 3.2 % shortfall in the uncorrected table is the dispatch floor, not residual
per-descriptor cost. Both tables are given because the correction is a model (one additive
per-call floor, independent of grid and kernel), not a measurement; the gate passes on either, and
§1.2's per-DMA delta is floor-invariant by construction.

**[rev] The gate text is self-contradictory, and the pass survives every reading of it.**
`measurement_and_design.md:1312` says *"If the charge ratio is ≥1.5 (i.e. halving bytes saves ≤33 %
of time), A drops to rank 4"* — but ratio ≥1.5 ⟺ saved ≥33.3 %, so the two halves are opposites.
`R1.md` §2.1 sets out all three readings and asks for a doc fix. Discharging it here rather than
picking one silently:

| reading | `bkv_p = 8` | `bkv_p = 9` | |
| --- | ---: | ---: | --- |
| parenthetical: saved > 33.3 % | 47.0 % | 47.9 % | **pass** |
| corrected literal: charge ratio ≥ 1.5 | 1.888 | 1.921 | **pass** |
| P6-style `time_ratio / byte_ratio` ≥ 1.5 | 1.051 | 1.033 | **pass** (fails the ≥1.5 bar, i.e. does not drop to rank 4) |

**The `bkv_p = 6` row is a replication control, not a new result.** R1 measured 42.7 % there on a
different job with a different `pages_per_seq`; this run gives 42.4 %. Agreement to 0.3 pp is the
evidence that the `deep` suite is measuring the same thing R1 measured.

**8 and 9 both pass, comfortably, on every reading and with or without the floor correction.** R1's
rank-4 verdict was correct for the configuration R1 measured and is wrong for the configuration we
now ship. What §3.3 goes on to show is that passing this gate turns out not to decide anything.

### 1.1 Why, and why the extrapolation would have been wrong in the safe direction

R1's mechanism was that the cost has a per-descriptor component that concurrency hides. The
per-DMA table shows it being hidden:

| `bkv_p` | in flight | 256 KiB record | 132 KiB record | per-DMA delta |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 24 | 77.7 ns (3.374e12 B/s) | 44.8 ns (2.951e12 B/s) | 32.9 ns |
| 8 | 32 | 77.3 ns (3.390e12 B/s) | 41.0 ns (3.225e12 B/s) | 36.4 ns |
| 9 | 36 | 77.3 ns (3.392e12 B/s) | 40.2 ns (3.284e12 B/s) | 37.1 ns |

The large record saturates by 24 in flight and does not move. The small record is still climbing at
36: it is the one paying the fixed per-descriptor cost as a larger fraction of its time, and more
concurrency keeps buying it back. At 36 in flight the residual is ~1.3 ns/DMA against a 38.9 ns
bandwidth-bound floor, i.e. the overhead is nearly gone — which is the whole content of the 96.8 %
figure.

**The coordinator was right to refuse the extrapolation, and the reason is worth recording.** The
monotone trend predicted "at least 42.7 %" at 36 in flight. The measurement is 47.9 % — the
extrapolation was *conservative*, and it would have produced the right decision. That is luck, not
method: the same trend, read at `bkv_p = 2`, predicts nothing about whether the curve flattens at
1.95 or at 1.80, and the difference between those two straddles nothing important here but would
have straddled the gate itself two `bkv_p` earlier.

### 1.2 Absolute stake, DMA-only

Multiplying the per-DMA delta by the real decode call's page-DMA count at ctx 8704 (64 sequences;
`bkv_p = 2` fetches 10 pages/seq = 640 DMAs, `bkv_p = 9` fetches 9 pages/seq = **576**):

| `bkv_p` | per-DMA delta | × 640 (p=2 shape) | × 576 (p=9 shape) |
| ---: | ---: | ---: | ---: |
| 6 | 32.9 ns | 21.08 µs | — |
| 8 | 36.4 ns | 23.27 µs | — |
| 9 | 37.1 ns | 23.71 µs | **21.34 µs** |

The probe's own `STAKE` line hardcodes 640 and therefore **overstates the `bkv_p = 9` figure by
11 %**; 21.34 µs is the correct one and is what I carry forward. This is a DMA-only number with no
compute to overlap with, so it is an **upper bound** on the call-level saving — and §3 shows the
actual bound is far below it.

---

## 2. The requirement, recomputed against the right denominator

The design's Option A claim is **−13 % to −22 % on the decode call**. It was written against
**221.4 µs**, which `results.md:346-347` establishes is a *contended wall clock*, not device time.

| denominator | what it is | source | −13 % | −22 % |
| ---: | --- | --- | ---: | ---: |
| 221.4 µs | contended **wall** clock, the design's baseline | `measurement_and_design.md` §3.1 | 28.78 µs | 48.71 µs |
| **170.58 µs** | **device** self time, incumbent `bkv_p = 2`, ctx 8704 | `results.md:52`; `hlo_stats` 170.724 | **22.18 µs** | **37.53 µs** |
| **143.39 µs** | **device** self time at `bkv_p = 9` — the ship configuration | `bkvp-extended.md` §1.1 (`hlo_stats`), job `j-6a35b078` | **18.64 µs** | **31.55 µs** |
| 145.95 µs | chain slope at `bkv_p = 9` (agrees with the above to 1.8 %) | `bkvp-extended.md` §4.2 | 18.97 µs | 32.11 µs |

**The bar is 18.64-31.55 µs**, not 28.78-48.71 µs. Correcting the denominator makes Option A's
target *easier* by 35 %, because the original baseline was inflated by contention and because
Candidate 0 has since removed 27 µs that A can no longer claim credit for.

Two hard ceilings sit above whatever A can deliver, and both are below the top of the band:

- **Stage A at `bkv_p = 9` is 84.279 µs in total** (`bkvp-extended.md` §4.1) and A can only attack
  part of it. −22 % of the call (31.55 µs) is 37 % of all of stage A.
- **[rev]** Decode stage A is **35.8 %** in the modelled HBM floor (`results.md:585-600`) — but that
  share was taken at the incumbent's `bkv_p = 2` shape, and it does not transfer. At `bkv_p = 9` the
  K traffic is 576 × 262144 B = 151.0 MB, which at the 3.70e12 B/s spec is 40.81 µs = **48.4 %** of
  the 84.279 µs stage A. Halving it removes at most **~20.4 µs**, not ~15 µs. *This is a modelled
  bound, not a measurement*, and §4 explains why its numerical proximity to the measured stake is a
  coincidence rather than corroboration.

---

## 3. Task 2 — the changed consumer

`probes/probe_kpack_consumer.py`. `load_bkv` and `compute_scores` are transcribed verbatim from
`streamindex_topk.py` (the `_stage_a.py` precedent), with a `source_md5` drift detector. Three
modes — `fetch` (DMA, no unpack), `compute` (unpack + einsum, no DMA), `full` (both) — crossed with
three consumers: `inc` (the incumbent's `(bkv_sz,1) → (1,bkv_sz)` lane transpose of one scale
column) and Option A's `(bkv_p,2,4,128) → (1,bkv_sz)` sublane-into-lane collapse, in a naive
`reshape` form and an unrolled `concat` form.

**Both A forms lower and run.** `a_concat` was built because Mosaic reshape lowering was the known
risk; it turns out to be unnecessary — `a_reshape` is within 0.6 % of it everywhere. The naive form
is the one to ship.

All figures below are **floor-corrected**: the empty-`pallas_call` dispatch floor (18.25 µs best) is
present identically in every arm and is subtracted as 142.6 ns/block at `bkv_p = 9` (128 blocks) and
28.5 ns/block at `bkv_p = 2` (640 blocks). It cancels exactly in every delta; correcting it only
changes the *percentages*, by under 1 pp.

### 3.1 The three modes at `bkv_p = 9`, ns per grid block

| mode | `inc` | `a_reshape` | `a_concat` | delta (`a_reshape`) |
| --- | ---: | ---: | ---: | ---: |
| `compute` — unpack + einsum, no DMA | 4192.9 | 3219.0 | 3199.9 | **−973.9 (−23.2 %)** |
| `fetch` — DMA only, no unpack | 2656.5 | 1332.7 | — | **−1323.8 (−49.8 %)** |
| `full` — both | 4345.0 | 3389.0 | 3368.7 | **−956.1 (−22.0 %)** |

Worst spread 1.7 %. And at `bkv_p = 2`:

| mode | `inc` | `a_reshape` | `a_concat` | delta (`a_reshape`) |
| --- | ---: | ---: | ---: | ---: |
| `compute` | 921.2 | 715.2 | 711.2 | −206.0 (−22.4 %) |
| `fetch` | 837.0 | 653.7 | — | −183.3 (−21.9 %) |
| `full` | 1160.4 | 972.8 | 974.2 | −187.6 (−16.2 %) |

### 3.2 The design's premise is false, in Option A's favour

`measurement_and_design.md:714-716` says "the unpack does not change". It changes, and the
coordinator's stated fear — that A's unpack adds VPU work on the axis that is already the
bottleneck — is **measured to be backwards**. Option A is **973.9 ns/block cheaper** at `bkv_p = 9`
and 206.0 ns/block cheaper at `bkv_p = 2`, in a DMA-free arm where nothing else differs.

**[rev] Say "973.9 ns/block", not "23.2 % cheaper unpack".** The `compute` arm is unpack *plus* the
einsum, relu, head reduction, scale multiply, both masks and the accumulate — all identical between
arms. 23.2 % is of that whole; as a fraction of the unpack alone the saving is larger by an unknown
factor. The absolute nanoseconds are the load-bearing figure and the only one carried forward.

**[rev] Which half of the layout change produces it is *not determined*, and the evidence points
away from my first guess.** The obvious story is the transpose: the incumbent extracts its scales as
`flat[:, head_dim:head_dim+1].T`, a `(9216,1) → (1,9216)` lane shuffle, where A's scale rows are
already lane-major and the same extraction is a free `reshape`. Against it stands this document's
own control. `a_concat` builds that operand a structurally *very* different way — 72 separate
`(1,128)` rows materialised and concatenated along lanes — and lands within **0.6 %** of
`a_reshape` in every single arm. If the scale relayout dominated, two formulations that unlike
would not agree that closely.

That redirects suspicion to the payload path: the incumbent materialising a 256-lane buffer where A
materialises 128 (9.4 MB vs 4.76 MB per block for four batches) — which is the halved record
itself, the half that **cannot** be had without changing the cache format. Both mechanisms are
quantitatively plausible at this magnitude, and this probe cannot separate them; the counter-reading
that Mosaic simply lowers the two A forms to identical code is also live. **Settling it needs an
`/xprof-llo` dump of the two `compute` arms**, and §6 explains why that question is worth more than
the next benchmark.

### 3.3 The sting: at `bkv_p = 9` the byte saving is worth nothing

| `bkv_p` | consumer | compute | fetch | serial sum | **measured `full`** | fetch hidden |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 9 | `inc` | 4192.9 | 2656.5 | 6849.4 | **4345.0** | **94.3 %** |
| 9 | `a_reshape` | 3219.0 | 1332.7 | 4551.7 | **3389.0** | 87.2 % |
| 2 | `inc` | 921.2 | 837.0 | 1758.2 | **1160.4** | 71.4 % |
| 2 | `a_reshape` | 715.2 | 653.7 | 1368.9 | **972.8** | 60.6 % |

At `bkv_p = 9` the incumbent's entire 2656 ns/block of DMA costs it **152 ns/block** of exposed
time. The pipeline is deep enough that the fetch is 94.3 % hidden behind the compute it overlaps.

Decomposing the net at `bkv_p = 9`:

```
unpack saving in isolation    :  -973.9 ns/block
fetch  saving in isolation    : -1323.8 ns/block
sum, if the two were additive : -2297.7 ns/block
MEASURED composite (full)     :  -956.1 ns/block
```

**The composite equals the unpack term alone to 1.8 %.** The fetch term contributes **+17.8
ns/block** — the wrong sign. Halving the K record, at the `bkv_p` we ship, buys nothing on the
critical path, because the DMA it shrinks was not on the critical path.

**[rev] That +17.8 ns is not noise; it is a per-block constant, and it probably belongs to the
LICM guard.** The same residual is +18.4 ns/block at `bkv_p = 2`, across a 4.5× change in DMA bytes
per block (36 vs 8 page DMAs) and a 5× change in `num_bkv`. Nothing on the fetch side is that
invariant. A fixed per-block difference between the two `compute` arms is: the guard stores
`(4,256)` u8 for `inc` and `(4,128)` u8 for `a`, once per block, and `full` never pays it. If that
is the explanation, the true unpack delta is ≈ −956 ns/block and the byte reduction contributes
**exactly zero** rather than +17.8 — which strengthens this section's claim rather than weakening
it, and shaves the §3.1 unpack percentage from 23.2 % to 22.8 %.

**[rev] An independent route to the same conclusion, which needs none of the three-arm
decomposition.** Real decode stage A at `bkv_p = 9` moves 151.0 MB in 84.279 µs = **1.79e12 B/s**,
against the **3.37e12 B/s** the DMA-only arm sustains on the identical descriptor form. The K fetch
is running at 53 % of what the hardware will give it at that shape — it has ~47 % of headroom, so it
is not what stage A is waiting on. At `bkv_p = 2` the same comparison is 1.50e12 of 2.42e12 = 62 %,
and the ordering matches the 94.3 % / 71.4 % hidden figures above.

This is the finding that matters most, and it is the one that inverts the whole case for A.
**Option A is not a byte-reduction win. It is a relayout win wearing a byte-reduction's clothes.**
The 47.9 % charge ratio Task 1 was commissioned to measure is real, is the correct answer to the
question asked, and is irrelevant to the outcome. R1's gate turns out to have been testing the
wrong half of the change all along.

The `bkv_p = 2` row tells a weaker version of the same story: the fetch is only 71.4 % hidden, and
A's net (−187.6) is **91.1 %** of its unpack term (−206.0) rather than the 98.2 % seen at
`bkv_p = 9`, with the fetch again contributing +18 ns. Shallower pipeline, less of the DMA hidden,
same conclusion — but the two are not the identical result, and the difference is in the direction
you would expect.

**[rev] The 47.3 % / 1.897 the consumer probe's own `fetch` arms report is an independent
replication of Task 1**, at a different geometry (`num_bkv = 1`, `pages_per_seq = 10`) and from a
different script, against Task 1's 47.9 % / 1.921 (`num_bkv = 2`, `pages_per_seq = 18`). Two
instruments, two pipeline depths, 0.6 pp apart.

### 3.4 Correctness — bit-identical

Both layouts are built by `make_source` from **one** payload array and **one** scale array, so a
matching output checksum is a real test that A's unpack recovers the same operands rather than two
random arrays coincidentally summing alike.

| `bkv_p` | consumer | `inc` checksum | A checksum | finite | |
| ---: | --- | ---: | ---: | ---: | --- |
| 9 | `a_reshape` | 4238582136.0 | 4238582136.0 | 34816 / 34816 | **exact** |
| 9 | `a_concat` | 4238582136.0 | 4238582136.0 | 34816 / 34816 | **exact** |
| 2 | `a_reshape` | 1246460376.0 | 1246460376.0 | 2048 / 2048 | **exact** |
| 2 | `a_concat` | 1246460376.0 | 1246460376.0 | 2048 / 2048 | **exact** |

Relative difference 0.000e+00 in all four. This is the design's "bit-identical" claim, confirmed on
the **reader** side.

**[rev] A correction, because the first draft of this section stated the opposite of its own data.**
At grid 128 the `compute`-mode checksums *do* match across all three consumers (4234290640.0 at
`bkv_p = 9`, 1227707599.0 at `bkv_p = 2`). It is the grid-32 `compute_lin` arms and the grid-8 smoke
run whose compute-mode checksums differ.

**That grid-128 match is an accident and must not be counted as extra correctness evidence.** The
LICM guard (§3.5) stores `remainder(i,251)` into one buffer slot per grid step, and the poisoned
region genuinely differs between layouts — for `inc` it covers payload *and* the lane-128 scale byte
for slots 0-3, for `a` it covers payload only, because A's scale rows live at 256-257. But at grid
128 the loop reaches `i = 127`, and `0x7F` is **NaN in `float8_e4m3fn`**; the payload poison turns
the same four slots into NaN in *both* arms, and `finish()`'s `sum(where(isfinite, a, 0))` masks
them out. That is exactly the 34816 → **34812** drop visible in every grid-128 compute arm. At grid
32 and grid 8 the poison byte never reaches 127, the four slots stay finite and differ, and the
checksums diverge — which is what those files show.

Only the `full` arms are the correctness test, and those are the four rows above.

### 3.5 Guards, and why the compute-mode result is not an artifact

- **LICM.** A `compute`-mode arm whose buffer is never written inside the grid loop can have the
  whole unpack hoisted out of it and report as free. Guarded by a per-step store of
  `remainder(i,251)` into the buffer, plus a quarter-grid linearity re-run: `inc` 1.052,
  `a_reshape` 1.054, `a_concat` 1.053 — all within the 0.85-1.18 band and, more tellingly, **all
  three move together**, which a hoist in one arm and not the other could not produce.
  **[rev] The guard has a real gap and a decisive backstop.** The gap: it writes only
  `bkv_x2_ref[0,0,0]` — batch 0 — leaving batches 1-3 loop-invariant and hoistable in principle, and
  the 0.85-1.18 linearity band is too loose to exclude that (3/4 hoisting predicts a raw grid-32/128
  ratio of ≈1.17). The backstop is the ratio of the two modes: `full/compute` is **1.035** for `inc`
  and 1.053 for `a`. `full` writes the buffer by DMA every step, so hoisting there is impossible; had
  three quarters of `inc`'s unpack been hoisted out of `compute`, `full` would have to exceed it
  several-fold. It exceeds it by 3.5 %. Nothing was hoisted.
- **DCE.** Scratch written and never read is deletable. The accumulator is reduced into the kernel's
  real `(2,128) f32` VMEM output at the last grid step. The reduction stops at a lane vector because
  a full vector→SMEM-scalar reduction is not reliably lowerable in Mosaic; the host finishes it.
- **Dispatch contamination. [rev]** The 16 headline arms span **188.8-760.9 µs** against an 18.25 µs
  floor — **2.4 % to 9.7 %**, not the 2.4-4.3 % this section claimed in draft. The 9.7 % arm is
  `a_reshape/fetch/bkvp9` (188.83 µs), which carries the −1323.8 ns fetch delta; `inc/fetch/bkvp9` is
  5.1 %. The floor is subtracted from every arm and cancels in every delta, and the `hidden %`
  figures move by 0.1 pp under a ±10 % change in the assumed floor — but the honest range is 2.4-9.7,
  and on the fetch arms contamination is 2.3× larger than the draft said. The probe's own 16.2 %
  warning is separately triggered by the small grid-32 *linearity* arm, not by any arm carrying a
  reported number.

---

## 4. The stake, and whether A clears its bar

Per-block deltas × the real decode call's grid at ctx 8704 (16 blocks at `bkv_p = 9`; 80 at
`bkv_p = 2`):

| configuration | delta/block | blocks | **call-level saving** |
| --- | ---: | ---: | ---: |
| **A on top of `bkv_p = 9`** (`a_reshape`) | −956.1 ns | 16 | **−15.30 µs** |
| A on top of `bkv_p = 9` (`a_concat`) | −976.3 ns | 16 | −15.62 µs |
| A on top of `bkv_p = 2` (`a_reshape`) | −187.6 ns | 80 | −15.01 µs |
| A on top of `bkv_p = 2` (`a_concat`) | −186.1 ns | 80 | −14.89 µs |

**The stake is ~15 µs and is nearly insensitive to `bkv_p`.** **[rev]** The clean version of that
story — saving ∝ rows processed, rows ∝ context length — is only approximately true, and the
agreement is partly cancellation: the per-row saving is 0.0264 ns at `bkv_p = 9` against 0.0251 at
`bkv_p = 2` (5 % apart), while total rows go 589,824 against 655,360 (11 % apart the other way).

**Coverage check.** The probe's `inc/full` inner loop is 69.52 µs at `bkv_p = 9` against a real
stage A of 84.279 µs (**82.5 %**), and 92.83 µs at `bkv_p = 2` against 111.775 µs (**83.0 %**). The
probe reproduces 82-83 % of measured stage A at two operating points that differ by 5× in grid size.
**[rev] That is the best evidence available that the 15 µs transfers, and it is weaker than it
looks**: it validates the *total*, not the *delta* being extrapolated, and a probe omitting 17.5 %
of the real work while adding 17.5 % of unrelated work would pass it identically.

### The verdict against the requirement

| | µs |
| --- | ---: |
| Requirement, −13 % of the 143.39 µs `bkv_p = 9` device call | **18.64** |
| Requirement, −22 % | **31.55** |
| **Measured stake** | **15.30** |

**Option A misses its own floor by ~18 %.** It delivers **10.7 %** of the `bkv_p = 9` device call
(8.97 % of the 170.58 µs `bkv_p = 2` baseline), against a claimed 13-22 %.

**[rev]** An earlier draft noted that a modelled HBM-floor bound predicted ≈15 µs and read the
measured 15.30 µs as corroboration. It is not, for two independent reasons: the modelled bound is a
bound on the *byte* saving, and §3.3 measures the byte saving to be ~zero; and the 35.8 % share the
bound came from was taken at `bkv_p = 2`, so transferring it to `bkv_p = 9` was itself wrong — the
correct modelled bound there is ~20.4 µs (§2). Two unrelated quantities, one of them miscomputed,
landing near the same number.

### Composite with Candidate 0 — arithmetic, not measured

`bkv_p = 9` alone is 143.39 µs device (`hlo_stats`) / 145.95 µs (chain slope), from 170.58 µs.
Subtracting A's 15.30 µs:

| | result | vs 170.58 µs |
| --- | ---: | ---: |
| `hlo_stats` 143.39 − 15.30 | 128.09 µs | **−24.9 %** (vs `hlo_stats` 170.58) |
| chain slope 145.95 − 15.30 | 130.65 µs | −23.5 % (vs the *matched* chain-slope baseline 170.89) |

**This is arithmetic composition, not a measurement.** What was measured is A's inner loop *at*
`bkv_p = 9` — the combination the coordinator asked for, and the configuration anyone would ship —
but the two savings were measured on different instruments against different baselines, and §3.3 is
a direct demonstration that savings in this kernel do not add. Treat the composite as an upper
bound pending an end-to-end measurement of the two together.

---

## 5. What this could not reproduce

The field most likely to make the result hollow, first:

1. **The writer is not exercised, at all.** Everything here is the *reader*. Producing layout A in
   the KV cache requires a changed scatter on the caller side, in XLA, outside the kernel. It is not
   measured, not costed, and not proven correct. `results.md` §prefill records that prefill stage A
   is **75.4 % issue/VPU** with no roofline headroom — a writer-side relayout lands there, and A's
   15.30 µs of decode saving is small enough that a modest prefill cost erases it. **This is now the
   single largest unclosed item, and it has been the largest one since R1.**
2. **`start_send_scores` is absent** from the probe's inner loop — the per-block score write-back,
   ~147 KB/block against a 9.4 MB K fetch at `bkv_p = 9` (≈1.6 %), and equal across arms. Part of
   the 17-18 % of stage A the probe does not cover.
3. **No stage B, no top-k, no sparsecore.** The probe models stage A's inner loop only.
4. **`bkv_p = 9` is ctx 8704 specific.** Candidate 0's rule is `cdiv(ctx, page_size)`, so a
   production traffic mix runs many `bkv_p`. §4 argues the ~15 µs is `bkv_p`-insensitive and
   measures that at two points; it is not measured across the nine ctx buckets.
5. **The unpack saving is not apportioned** between the eliminated transpose and the halved payload
   width (§3.2), and §3.2's control argues against the reading that would have been convenient.
6. **[rev] Grid-invariance is checked at 32 vs 128; the extrapolation target is 16** — a factor of
   two below anything measured — and only in `compute` mode, never in `full`, which is the mode
   −956.1 comes from.
7. **[rev] The probe runs with nothing else on the memory system.** The real op has SparseCore
   stage B (~30.9 µs) live concurrently. Contention would *un-hide* some of the fetch and make A
   worth **more** than 15.30 µs — the one unexamined direction that cuts against this document's
   headline. The 53 %-of-achievable K-fetch utilisation in §3.3 leaves room before it bites.
9. **Best-of-N is optimistic by construction**, as in every probe here.
10. **The grid-8 smoke phase is not a result.** Its floor is 28.11 µs against a 30.01 µs fastest arm
   — **93.7 %** dispatch. It reports a 4.9 % fetch saving and a 1.052 charge ratio at `bkv_p = 9`,
   both pure dispatch artifacts, contradicted by the grid-128 run of the same code. It was a
   build/lower smoke test and did its job. `R1b_consumer_grid8.json` is retained only so the
   contradiction is on the record.
11. **Source drift, checked and immaterial.** The image's `streamindex_topk.py` is md5
   `b0099c0a…`; the checkout HEAD I transcribed from is `1e221f10…`, 235/162 lines apart across two
   upstream commits. I diffed the three functions that matter: **`_fetch_bkv` and `compute_scores`
   are byte-identical**, and `load_bkv` differs only by two deleted comments. The transcription is
   faithful to both.
12. **Single-tenant within the JobSet, not certified beyond it.** One container held all four chips
   for the job's life; whether the physical host was otherwise idle is not something `cdk` exposes.

---

## 6. What I would do with this

Not a recommendation to ship, and not a recommendation to drop:

- **A's case has changed shape, not just size.** It is a correct, positive, ~15 µs change that
  misses its stated floor, and whose value is in the consumer, which the design costed at zero — not
  in the byte reduction, which is the only part the gate ever tested.
- **[rev] The next thing to buy is an `/xprof-llo` dump of the two `compute` arms, not a benchmark.**
  §3.2 leaves two candidate mechanisms, and they imply opposite plans. If the 973.9 ns is the
  eliminated *transpose*, there is very likely a change that captures it **without touching the
  cache format** — no writer, no prefill exposure, none of §5 item 1 — and that change is strictly
  better than Option A. If it is the halved *payload width*, no such variant exists and A must be
  taken or left whole. The concrete question the dump answers is whether Mosaic fuses
  `flat[:, :128]` into the VMEM load, i.e. whether the incumbent reads 9.4 MB or 4.76 MB per block.
  It is minutes on a 4-chip pod against hours for the run it would redirect.
  *An earlier draft asserted the transpose and recommended the no-writer probe on that basis; the
  `a_concat` control argues the other way, and the honest position is that it is undetermined.*
- **The writer has to be measured before A ships in this form**, and prefill's 75.4 % issue/VPU is
  the reason.
- **Do not log 15.30 µs as a measured call-level saving.** The measured quantities are the per-block
  deltas (−956.1 ns at `bkv_p = 9`, −187.6 at `bkv_p = 2`, floor-corrected, best-of-9, grid 128);
  15.30 µs is a ×16 extrapolation from a probe covering 82.5 % of stage A. Two corrections worth
  carrying so nobody re-derives them: **576, not 640, page DMAs** at `bkv_p = 9`, and **143.39 µs,
  not 221.4 µs**, as the decode denominator.
