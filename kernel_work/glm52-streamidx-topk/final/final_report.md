# `streamidx_topk` for GLM-5.2 — composed candidate

One kernel carrying every adopted step-3 change, measured **as a whole, in one job, against the
incumbent**. Not a sum of the step-3 numbers; where it differs from that sum, §2 says where the
difference went.

**Sign convention.** Every percentage is a **change in time; negative is faster**. Signed numbers
appear only in a `Δ time %` column. `Δ spread` is max−min of the per-repeat paired deltas.
**An effect smaller than its own spread is UNRESOLVED (`!`) and is not a result.**

**Provenance.** `measured (job)` / `modelled` / `assumed` / `unmodelled`. A term valued at zero is
`unmodelled`, never `0`.

- **Measurement job:** `j-01b857bb-d11e-4d44-ab29`, `JobStatus: Succeeded`, 4/4 shards `rc=0`.
  Image reused, nothing built: `…/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.
  All arms timed adjacently inside one process per shape via `instruments/bench.interleaved`;
  the delta is formed **inside each repeat**. Nothing here computes `full − sc`.
- **Incumbent identity:** `sparsecore_topk.py` sha256 `3d058de2…b534806` verified `== expected` in
  all four shards. `inc_copy` (this candidate, every option OFF) is UNRESOLVED against `inc` at
  all ten shape×level cells (worst `+1.4% !`, spread 5.5pt) — the transcription is clean, so the
  other arms are comparisons against the shipping code and not against a copy of it.

---

## 1. The headline — composed candidate vs incumbent

`measured (j-01b857bb-d11e-4d44-ab29)`. `full` = the whole op; `sc` = the selector alone.

| row | geometry | level | inc µs | final µs | Δ time % | Δ spread |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| B-D1-1M | B=16 N=2²⁰, p=16, C=32768 | full | 1438.4 | 518.4 | **−64.0%** | 3.8pt |
| B-D1-1M | | sc | 767.6 | 405.0 | **−47.3%** | 3.1pt |
| B-D4-256k | B=16 N=2²⁰, p_run=4, C=8192 | full | 1085.2 | 477.3 | **−56.0%** | 1.7pt |
| B-D4-256k | | sc | 799.5 | 362.8 | **−54.5%** | 6.5pt |
| B-P1024-256k | B=1024 N=2²⁰, p_run=4 | full | 50961.6 | 15879.9 | **−68.8%** | 0.5pt |
| B-P1024-256k | | sc | 43369.8 | 12487.4 | **−71.2%** | 0.5pt |
| A-D16-8k | B=16 N=10240, p 2→1 | full | 284.6 | 239.7 | **−15.6%** | 6.6pt |
| A-D16-8k | | sc | 217.4 | 180.1 | **−17.6%** | 7.3pt |
| A-D64-8k | B=64 N=10240, **p_inc=1** | full | 336.9 | 328.4 | −3.3% `!` | 6.9pt |
| A-D64-8k | | sc | 168.6 | 160.8 | −4.2% `!` | 12.7pt |

**Eight of ten cells resolve.** Long context (B) wins 56–69% of full-op time. Short context (A) at
B=16 wins 15.6% — and it **did** resolve at 25 repeats, contrary to the pre-registered worry that
this shape is unresolvable at the harness floor; the caveat stands for the *mechanisms* (§2) but
not for the composed effect.

**`A-D64-8k` is structural, not a measurement.** `p_inc = 1`: the two-stage path is not entered, so
`no-o8` and `no-o3` are *the same program* as `final` there. Both cells are UNRESOLVED and the
−3.3 / −4.2% must not be read as a win.

**Fraction of the step-1 modelled floor** (floor `modelled`; times `measured (j-01b857bb)`), full
level:

| row | floor µs (`modelled`) | inc | final | floor/inc | floor/final |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-D1-1M | 36.6 | 1438.4 | 518.4 | 2.5% | **7.1%** |
| B-D4-256k | 36.6 | 1085.2 | 477.3 | 3.4% | **7.7%** |
| B-P1024-256k | 956.1 | 50961.6 | 15879.9 | 1.9% | **6.0%** |
| A-D64-8k | 18.5 | 336.9 | 328.4 | 5.5% | 5.6% `!` |

Roughly a 3× move toward the floor at (B), to 6–8% of it. Nowhere near step 3's ≥80% stopping bar,
so the block is **not** exhausted — but the residual is no longer the mechanisms modelled in step 1
(see carried 6, 7 and the harness finding in §2).

**The composition is the best arm.** No leave-one-out arm resolvably beats `final` at any
shape×level. Three cells show a `no-*` arm nominally faster (B-D1-1M `no-o3` −0.2pt; A-D16-8k
`no-o8` −0.6pt, `no-o3` −0.7pt) and every one is inside its own spread. `final` ships.

---

## 2. Was composition additive? Leave-one-out attribution

Each mechanism's contribution **on top of the other four**, in points of the incumbent, paired
inside each repeat: `median[100·(t_noX − t_final)/t_inc]`. Positive = removing it costs time.
`measured (j-01b857bb-d11e-4d44-ab29)`.

| row / level | O8 seam | O3 waves | O7 `scan_count` | O1 guard | O2 KV tile | Σ LOO | headline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-D1-1M full | **+26.2** | −0.2 `!` | +2.1 `!` | +2.6 `!` | **+33.2** | +63.8 | 64.0 |
| B-D1-1M sc | **+43.6** | −0.2 `!` | +3.9 `!` | −0.2 `!` | — | +47.1 | 47.3 |
| B-D4-256k full | **+12.4** | **+6.4** | +2.2 `!` | +3.5 `!` | **+8.1** | +32.6 | 56.0 |
| B-D4-256k sc | **+11.0** | +8.0 `!` | +2.2 `!` | +0.1 `!` | — | +21.3 | 54.5 |
| B-P1024 full | **+7.2** | **+35.4** | **+3.8** | **+2.6** | +0.4 `!` | +49.4 | 68.8 |
| B-P1024 sc | **+1.3** | **+41.5** | **+4.4** | +0.1 `!` | — | +47.3 | 71.2 |
| A-D16-8k full | −0.6 `!` | −0.7 `!` | +0.5 `!` | **+7.9** | −0.3 `!` | +6.8 | 15.6 |
| A-D16-8k sc | −1.4 `!` | −1.2 `!` | +0.5 `!` | **+8.2** | — | +6.2 | 17.6 |
| A-D64-8k full | −0.4 `!` | +0.1 `!` | +2.6 `!` | −0.1 `!` | +0.1 `!` | +2.4 | 3.3 `!` |
| A-D64-8k sc | +0.5 `!` | +0.1 `!` | +4.4 `!` | −0.1 `!` | — | +4.9 | 4.2 `!` |

**Verdict per shape.**

- **B-D1-1M — additive.** Σ LOO 63.8pt against a 64.0pt headline, and the two resolved terms match
  their step-3 standalone numbers almost exactly (O8 +26.2 vs +26.3; O2 +33.2 vs +33.0; O1 +2.6 vs
  +2.6). At this shape the mechanisms are disjoint: O8 owns the stage-2 seam, O2 owns the scoring
  pass, and they do not touch each other's work.
- **B-D4-256k — sub-additive, and the shortfall is exactly the predicted hazard.** Step-3 standalone
  sum 82.3pt; measured 56.0pt; shortfall 26.3pt. Σ LOO is only 32.6pt, i.e. **23.4pt of the saving
  is shared** — removable by more than one mechanism, so no single removal costs it. It sits where
  composition hazard #2 said it would: O8 and O3 both rewrite stage 1. Standalone O8 −34.4% and O3
  −33.4%; marginal +12.4pt and +6.4pt. Neither is wrong; they were largely buying the same work.
- **B-P1024-256k — sub-additive, same seam.** Standalone sum 80.7pt, measured 68.8pt, shortfall
  11.9pt; 19.4pt shared. Here the split reverses — O3 carries +35.4pt and O8 only +7.2pt, because
  at B=1024 the wave packing is what removes the launch-count term.
- **A-D16-8k — the composed win *is* O1's win.** O1 contributes +7.9pt of a 15.6pt headline and is
  the only resolved term; everything else is UNRESOLVED. O1's step-3 standalone was −16.0% at this
  shape and its marginal here is +7.9pt — halved, `unmodelled`. Note this shape is ~70% harness
  (`g_write` 128 µs against an `inc` of 185 µs), so both figures understate the kernel-level effect
  by roughly 3×; the *ratio* between them is the finding, not either absolute.
- **A-D64-8k — nothing resolved, and nothing could be.** `p_inc = 1`.

**O7 is settled: keep it.** The brief said to drop it if it does not resolve on top of the rest. It
does — at B-P1024-256k it resolves at both levels (**+3.8pt** full, spread 1.1pt; **+4.4pt** sc,
spread 0.8pt), the shape with the tightest spreads in the job. At the other four shapes it is
UNRESOLVED, but its sign is positive in **9 of 10** cells and it never costs anything resolvably.
It ships on the composed measurement, not on its standalone number.

---

## 3. The adopt set as shipped, and its gating

Every change is one field of the frozen `pallas_mosaic_tpu.Config` (stdlib dataclass, hashable,
therefore a `static_argname` — each arm gets its own jit cache entry). `Config()` is what ships.

| flag | change | gating as shipped |
| --- | --- | --- |
| `emit_keys` (O8) | stage 1 emits (index, key) pairs; the `[B,N]` seam gather is **deleted** | on; inert where `p == 1` (stage 2 not entered) |
| `pack_waves` (O3) | partition ladder instead of the static ceiling — `row_id = slot_c >> log2p` | on; ladder collapses to `[1]` at `p_max == 1`, so inert at (A) |
| `scan_count_hist` (O7) | `plsc.scan_count` + one `addupdate_scatter` per run, replacing the lane-private histogram | on, unconditional |
| `fix_growth`, `skip_empty_launches`, `elide_score_prefill` (O1) | growth guard, zero-extent launch skip, `-inf` prefill elision | growth guard fires only when doubling P does not cut the wave count — at (B) `p_max` is capacity-bound at 16 so it is inert; at A-D16-8k it moves p 2→1 |
| `wide_kv_tile` (O2) | regime-conditional KV tile | keyed on `page_size` read off `cache_kv`: **1024 → returns the incumbent tile** (inert at (A) by construction, not by a flag); **512 → `bkv_p=(32,2,4) bq_sz=(1,128,1024)`**, announced once per process |

**Not adopted**, and why (also in `DEVIATIONS.md` §9–10): **O3's flat-index gather** — its full-level
win had no identified mechanism (`unmodelled`), it was UNRESOLVED at selector level everywhere, its
gather measured 2.7–28.5% *slower* than `take_along_axis` in isolation, and under `emit_keys` the
gather it replaces **does not exist**. **O9** did not resolve at any shape; **O5** was never built.
Neither has a flag here, because a flag implies a measured arm.

---

## 4. Correctness

**Named job: `j-01b857bb-d11e-4d44-ab29`.** The gate ran **before** the benchmark in the same
shard, `&&`-chained, in all four shards — a failing gate would have taken the timings with it.
`instruments/correctness.py` is imported, never copied or edited. **No tolerance is defined
anywhere in `final/`** — `grep` over every `.py` here returns none; the two values appearing in the
separation block below are quoted verbatim from the shared gate's own printout, which is what makes
them reportable rather than something this candidate chose.

- **Gate, `final`, all four shards:** `5/5 cases passed, 0 assertion failures`. Worst across cases
  `worst_deficit=0.00486`, `worst_churn=0.439%` (both `pre-1x8k`).
- **Index-set symmetric difference vs the incumbent:** `total symdiff 0: IDENTICAL SETS` at every
  case, for `final` and for `inc_copy`. `dec-2x8k 0/2 · dec-ragged 0/4 · pre-1x8k 0/128 ·
  dec-B-2x8k 0/2 · pre-B-1x4k 0/128`. In the benchmark shards, `setdiff <arm> vs inc: 0` for all
  seven arms at all five shapes, both levels. The composition changes **how** the top-k is found,
  not **what** it is — so no threshold argument is load-bearing for this candidate.
- **Negative control (`--negative-control`), shard 3:** `negative control: PASS`.
  - `drop-block` **FAILED on all 5 cases**, as required — worst deficits `0.07755 / 0.07495 /
    0.09307 / 0.04605 / 0.15972`, worst churns `25.000% / 24.951% / 27.686% / 15.869% / 27.002%`,
    with hard undercounts on two (`dec-ragged tok1: 952 selected, want 2048`; `tok3: 1 selected`).
  - `tail-swap` **PASSED on all 5**, worst deficit ≤ `0.00249`, worst churn ≤ `0.391%`.
  - Separation, quoted verbatim:
    ```
      deficit  SEPARATES: any tol in (0.00486, 0.04605) works, ratio 9.5x
               tol=0.015 is INSIDE that window -- load-bearing
      churn    SEPARATES: any tol in (0.00439, 0.15869) works, ratio 36.1x
               tol=0.03 is INSIDE that window -- load-bearing
    ```
    Both shipped tolerances sit **inside** their separation window, so the gate discriminates rather
    than merely passing.

This is evidence of correctness against the gate and of index-set identity with the incumbent. It
is **not** an end-to-end accuracy check; the end-to-end benchmark is still gated by
`/preflight-check`.

---

## 5. `carried:` — measured here, not kernel edits

| # | item | route | next action |
| --- | --- | --- | --- |
| 1 | **`report.md`'s (A) table is overstated.** A-D16-8k is ~70% harness (`g_write` 128 µs vs `inc` 185 µs), and raising repeats 5→25 *widened* the spread. Unpriced item 12 fires. | `report.md` §3 | correct the (A) rows with the harness arm subtracted, or mark them as harness-dominated. |
| 2 | **B-P1024 gap closed:** 407 TFLOP/s = 35.2% of measured bf16 peak. | `probes.md` | append the closure. |
| 3 | **`roofline.py:86-95`** says the A-D16 > A-D32 inversion "is NOT established". It **is** — 5/5 adjacent repeats, fully caused. | this checkout | fix the comment. |
| 4 | **The 129-of-256 cache row is a capacity win**: 127 B/token × 21 layers ≈ **2.7 kB/token of HBM freed**; at (B) concurrency is KV-capacity-capped. Belongs to the cache writer. | `mla_attention_op.py` / cache spec | own exercise; the arithmetic is done. |
| 5 | **`args[2].copy_(new_cache)` at `mla_attention_op.py:140` right after `donate_argnums=(2,)` at `:128`.** If not elided by storage aliasing, a full indexer-KV-cache copy per layer per step — ~1 GB/layer at (B). `unmodelled`. | call site | **one xprof of a real AgentX step. Larger than anything in this exercise if real.** |
| 6 | **1.65 ms/decode-step of Pallas launch fixed cost** over 21 layers, 1.10 ms of it empty-grid launches. O1 recovers the per-layer share; **merging the three launches into one** recovers the rest and is uncosted. | this block, next exercise | measure 1-launch vs 3-launch at fixed work. |
| 7 | **`MAX_SLICE_WORDS = 65536` caps the selector at ~2 M columns at b=1**; AgentX at 1 M is at 50% of the ceiling. Capability limit, not perf. | roadmap | note before the next context-length increase. |
| 8 | **E6 (11-bit L0) is worse than it looks**: 2048 buckets make `priv_vmem` 128 KiB and `zero_priv` 2048 iterations, against 256 saved per skipped refinement. A candidate correctly *not* on the table. | `combined.md` | record it. |
| 9 | **`num_kv_pages_per_block[1]`, the dead PREFILL slot**, is load-bearing for `num_sublanes_total` and so for the whole score-buffer width. One line, still unmeasured. | this block | fold in as a fourth arm next time. |
| 10 | **`instruments/bench.interleaved` earned its keep and should be the default.** Run-order drift measured 3–9%; several effects here are 2–4pt and only exist in the paired difference. | `instruments/` | promote it in the harness docs; deprecate unpaired A/B. |

---

## 6. Contradictions with `references/*.md`

Checked against the references **as they stand after `3d5ba12` and `0b9bf49`**; the six claims those
commits corrected are excluded rather than re-reported. Classification: **(a)** reference wrong,
**(b)** right for a different configuration, **(c)** right but narrower than it reads.

| # | file:line | verbatim claim | measurement | class |
| --- | --- | --- | --- | --- |
| C1 | `03-pipelining.md:35` | "Use **2.97 TB/s** as the denominator for every roofline estimate." | materialisation inside a `lax.switch` branch runs at **1.30 TB/s** — 51 µs at 64 MiB, 3310 µs at 4 GiB, same rate `measured (j-fad1a0f8)`. Over-predicts by 2.3×. | (c) |
| C2 | `hardware-facts.md:179` | "add / drop / permute a **leading** axis \| **free**" | same operand copy as C1. The table's own scope line at `:171` is register-resident inside a Pallas body, priced in ps/elem — not an HBM-resident operand crossing a conditional. | (c) |
| C3 | `01-fusion-boundary.md:731-733`, restated `code-style.md:348-350` | "Every conditional is an optimization barrier… Splitting a body into several statically specialized branches — including a no-op branch — lets the compiler overlap work inside each." | the opposite move won here: **hoisting the bitcast+reshape *out* of the switch**. Each extra branch touching the operand pays another full copy. Both quotes are about `pl.when` inside a kernel body. | (c) |
| C4 | `01-fusion-boundary.md:886-887` | "Compute the round-trip saving: `2 * intermediate_bytes / 2.97e12` seconds, which is an upper bound. Small against the op's runtime? Stop." | the `-inf` fill is **28–59% of modelled HBM traffic** and deleting it won only −1.7 to −2.5%. The rule has no "large → still not worth it" branch. | (c) |
| C5 | `02-partitioning.md:262-268` | "A single unbatched dispatch under ~750us is measuring the host… a batched probe reports a 21.69us kernel perfectly well." | a **batched** harness at B=16/N=16384 still carries **~70%** overhead (`g_write` 128 µs vs `inc` 185 µs). Batching defeats the ~150 µs host round trip only; an in-jit control floor survives it. | (c) |
| C6 | `02-partitioning.md:253-255` with `03-pipelining.md:101` | "512 ns fixed per grid step" | **0.500 µs per in-body iteration** with copies, bytes and extent held constant. Magnitude reproduces, but none of the four enumerated mechanisms can be paying for it and the cost is not confined to a managed grid step. | (c) |
| **C7** | `seams.md:212-216` | "Probe the primitive, not the variant." | **the probe inverted the sign.** Isolated, the flat gather looked 2.7–28.5% *slower*; substituted in situ it won −5.1/−3.5/−4.4%. As an unqualified ranking instruction this is **wrong**, and internally inconsistent with `05-compute-units.md:413-414` and `08-sparsecore.md:1519-1521`. | **(a)** |
| C8 | `08-sparsecore.md:1793-1797` | "Partition columns until the pipeline has enough stages… split until the pipeline is deep enough, then stop" | the stopping criterion is **wave count**, not stage count; the measured move was P 2→1 for −16.0%. `08:1721` already flags the section "read, not measured". | (c) |
| C9 | `08-sparsecore.md:1526-1529` | "Per-primitive wins do not compose." | selector −13.8% → full op −16.9%, **better than 1:1** `measured (j-70f24ae2)`. But the pair cuts both ways: O7 gave −4.0/−4.3% at selector level and UNRESOLVED +0.4/−0.3% at full-op level. Correct rewrite: *the composition ratio is not predictable in either direction — measure the full op.* | (c) |
| C10 | `02-partitioning.md:226-231` | TensorCore budget `sum(block_bytes…)*2 + scratch_bytes <~ 31 MiB` | as literally written it passes both tiles and the measurement confirms it. What is missing: **Mosaic tiles einsum temporaries internally, so undeclared intermediates must not be added to the left-hand side.** Caution before quoting: `hardware-facts.md:290-294` records a prior bisect where an apparent capacity ladder "was not a capacity at all"; confirm the einsum output is genuinely consumed. | (c) |

**Confirmed by this exercise** (worth citing, not correcting): `05-compute-units.md:413-414` (measure
an offload against its neighbour); `08-sparsecore.md:1656-1661` (the `scan_count` + single
`addupdate_scatter` idiom — this is O7, and it shipped); `03-pipelining.md:116-118` ("a floor quoted
without a step count is not a floor"); `08-sparsecore.md:1698-1701` ("minimise passes, not
operations"); `seams.md:34-36` (the layout licence); `code-style.md:342-344` (fuse the zeroing into
the first accumulation).

**Two staleness notes, not contradictions** — no measurement disagrees, the text simply lags a
retraction already made: `05-compute-units.md:49-53` and `:501` still carry the per-subcore-480-KiB
reading that `3d5ba12` retracted in `08:128-136` and `02:162-168` (file `05` was touched by neither
commit; `hardware-facts.md:479-480` has the same wording but defers to `08` in the same sentence).
And `08-sparsecore.md:115-117` ("The 512 KiB is per vector subcore and is not poolable") still
stands **13 lines above its own retraction** at `:128-136`.
