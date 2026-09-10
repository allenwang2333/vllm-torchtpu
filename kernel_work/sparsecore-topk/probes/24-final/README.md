# probe 24 — the composed candidate

✅ **`j-4dd339df-0495-4d6c-9afc` — Succeeded. `P.prefillX | whole` −74.206 µs, RESOLVED FASTER.**
Jump to [The result](#-the-result--j-4dd339df-0495-4d6c-9afc-succeeded-348-s-of-ranking).

🚀 **Submitted as `j-4dd339df-0495-4d6c-9afc`** (created 2026-09-10T22:46:32Z, `cdk-p5`, image
`…:fa63d6df`, `--map-dir stage:tpu-job:/root/p24`, `-t agent-jobs,measurer`). RUN_CMD read back from
`cdk job recipe` and md5-compared byte-for-byte with what was sent:
**`a24612a7f51432714ea7306aa620d2bb` = `a24612a7f51432714ea7306aa620d2bb`** ✅ — which verifies
*sent*, not *working*; the in-job gate banners are what establish the run was real.

Everything from here to the "The build as delivered" section was **pre-registered**: written and
committed to disk before the composite existed and before any number from it existed. That is the
point of having written it then.

**Question.** Measured whole against the incumbent in one job, how fast is the composition of
`v2` + `v4-c2048` + `v12`?

**Arms.** The incumbent at **4 slots** (`base` + three nulls that are the same source loaded as
separate module objects) and **one** composed arm, scope suffix `_final`. Same protocol as probe 23:
one pool, 12 interleaved rounds reshuffled per round, paired REPS 20/60, `us_median`, **both
brackets**, `--ctx 9216`, `k=2048`, `b ∈ {1024, 64}`, both production ρ with production arrangement,
and the **ρ=1 control retained** — it earned its place in probe 23 by catching `v4-ragged`'s
by-construction null.

---

## What is being reused, and why it is a wrapper and not a rewrite

`stage/rank.py` is byte-identical to the harness that ran clean as `j-66c7e331-fa41-461e-8686`
(md5 `e0bde4c103d277aa933fa585c1c49603`). `stage/final_rank.py` imports it, swaps the arm table, and
appends two stages. The pooling, the two brackets, the eager `pl.kernel` spy, the HLO scope-name
check, the numpy-oracle smoke and the in-job md5 gate are all code that has already executed on this
hardware and produced an accepted result. Re-typing them to change an arm table would throw that
away for nothing.

`rank.py`'s stage 7 is fully guarded (`if a is None: continue`), so the probe 23 pre-registered
readings simply do not print with the new arm table — no edit required, and none was made.

## The two added stages

**STAGE 8 — the one-signed null correction.** `rank.py` measures every delta against `base`, one
particular incumbent slot, while its floor is the **range of all four**. When the three nulls come
out one-signed, `base` is sitting at an extreme of its own floor and a candidate at the incumbent
centroid reads as a win of up to a full floor width. That is exactly what happened in probe 23's
`D.decodeX|inside` cell (nulls at −0.205, −0.211, −0.103 µs) and it is why four printed verdicts
were noise. The correction: **measure the delta against the median of the four incumbent slots and
keep the range as the floor.** Both are reported; the corrected one governs.

**STAGE 9 — three-estimator agreement.** `us_median` is the pre-registered read but it is not
sufficient. A call is **RESOLVED** only if `us_median`, the pooled slope `us` and `us_unpaired` all
agree in sign **and** each clears its own corrected floor computed from its own four incumbent
values. It prints the **margin over the floor** per estimator, in µs and as a percentage of the
floor, and appends `[MARGINAL: weakest estimator clears the floor by only N% of it]` to any resolved
verdict whose weakest estimator clears by under 10%. The rule is not weakened by the margin; the
margin is printed beside the verdict so the reader can weight it. In the replay below the flag fires
on exactly one call — the `v4-c2048` promotion, at 0.61% — and on nothing else.

### Both stages were regression-tested against probe 23's real data, for zero cluster time

`stage/final_rank.py`'s STAGE 8/9 were replayed over `probes/23-ranking/ranking.json` with each of
five known arms relabelled as `final` in turn. Known-answer test, on the host, no TPU:

| impostor | corrected verdict | matches the probe 23 reading? |
| --- | --- | --- |
| `v2` | RESOLVED FASTER in all 6 cells | ✅ |
| `v4-ragged` | prefill both brackets FASTER; **decode both NOT RESOLVED**; ρ=1 both NOT RESOLVED | ✅ reproduces the flip **and** the by-construction null control |
| `v12` | prefill whole FASTER, **prefill inside NOT RESOLVED**; decode whole FASTER, **decode inside NOT RESOLVED** | ✅ reproduces the glue-only finding and the flip |
| `v11` | same shape as `v12` | ✅ reproduces the flip from a printed "SLOWER" |
| `v4-c2048` | prefill both FASTER; decode inside NOT RESOLVED (2/3 clear); **decode whole RESOLVED FASTER** | ✅ three flips, and one **promotion** |

The correction reproduces **all four** manual flips reported in probe 23, and it additionally
**promotes** `v4-c2048` at `D.decodeX|whole` from "not resolvable" (−0.267 µs against a 0.730 µs
floor, measured from a biased `base`) to a resolved win of −0.734 / −1.044 / −1.250 µs across the
three estimators. That strengthens a candidate the coordinator had already adopted on the prefill
evidence; it does not create one.

⚠ **That promotion is marginal.** Its weakest estimator clears the floor by **0.0045 µs — 0.61% of the
floor**. It is resolved by the pre-registered rule and the rule is not being changed to suit it, but
it is not the same kind of fact as `c2048`'s prefill win (−4.500 µs against a 0.040 µs floor). It
changes no decision.

**The decomposition says more than the flag.** One estimator on the line at +0.61% with the other two clearing by **98.2%** and **80.7%** is a different object from three estimators clustered at the boundary: it says the call is probably **real** and that `us_median` is the conservative estimator here, not that the effect is borderline. Same flag, opposite readings — so the flag is always printed with its spread.

### ⚠ Only one of those five checks is an external anchor

**Four of the five are partly circular.** They reproduce probe 23's conclusions — and probe 23's
conclusions already contained the hand cross-checks this rule was written from. So agreement there
is, in part, the rule agreeing with its own author. It is still worth running: it proves the code
implements the reasoning, which is a real failure mode. It is not independent evidence that the
reasoning is right.

**The one genuinely external anchor is `v4-ragged` at ρ=1.** That arm is a null *by construction* —
it optimizes ragged rows, and at ρ=1 there are no ragged rows, so it cannot do anything. The ground
truth there was fixed by the arm's own mechanism before any measurement, not by an earlier reading of
mine. The rule could have failed against it and did not. A reader who takes all five checks as
independent will over-trust the rule; the honest count is **one external anchor and four
implementation checks.**

## Pre-registered readings for the final job — written before the numbers exist

**(1) ⛔ The composed number comes from this job alone.** No sum of `v2` + `v4-c2048` + `v12` is
computed anywhere in `final_rank.py` — not as an expected value, not as a residual. The three prior
per-arm deltas are printed beside each cell for orientation, loudly labelled *probe 23, a different
job*, and nothing is subtracted from them. If the composite lands short of the parts, **that gap is
the finding**; overlapping arms are what step 4 exists to detect.

**(2) The composition succeeded iff `P.prefillX|whole` is RESOLVED FASTER** under STAGE 9. Prefill is
the prize; that one cell is the headline. Anything else is secondary.

**(3) The two brackets localize `v12`'s glue win without any cross-job arithmetic.** `v12`'s
contribution is entirely outside the kernel (−25.15 µs whole, −0.08 µs inside). So, *within this one
job*:

| observation | reading |
| --- | --- |
| whole-call delta **materially exceeds** the inside-kernel delta | `v12`'s entry-glue deletion **survived** composition |
| whole-call delta ≈ inside-kernel delta | `v12`'s glue deletion **did not survive** — the merged entry reintroduced the fusion, and the composite is effectively `v2` + `c2048` |

This is a within-job difference of two brackets, not a reconciliation against a sum, and it is the
one question the composite's own numbers can answer about which parts made it through.

**(4) ⚠ `v12` carries an unclosed risk into the composite.** Its win is the *shape* of the known
"blackout residual is XLA, not your kernel" trap — mitigated by the bracket being the real 3-D
production entry `f32[b, n/128, 128]`, not closed by it. If the composer cannot show the deleted
fusion is one the **production** caller pays, `v12`'s share of the composite is tagged
`unmodelled` regardless of how the number here lands. A faster number in this harness is not
evidence against that risk, because this harness is where the risk lives.

**(5) Quote µs, not percent-of-call.** ⛔ *Pre-registered text kept verbatim; **its mechanism was
wrong** and is corrected below in "⛔ Reading (5)'s mechanism was wrong". The conclusion stands, the
reasoning in this paragraph does not. Left unedited because rewriting a pre-registration after the
fact is how a pre-registration stops being one.* The 31.53 µs/call dispatch floor sits inside every absolute.
It cancels in an in-cell delta but it inflates every denominator: up to 31.5 µs of the ~49 µs decode
baseline is dispatch, so a decode "−N% of the call" is a lower bound on the fractional kernel change
by up to ~2.8×.

**(6) Decode cannot be the headline.** sctopk is ~1.2% of decode device time, so even a large
fractional decode win is worth **~0.08% of the decode step at most**. Prefill is the prize. This
goes in the report *before* the number, so nobody extrapolates a −13% kernel win into an e2e claim.

**(7) The ρ=1 cell is a control, not a result.** It is retained to keep an instrument check in the
pool, and because `v2` dominates the composite there.

## Gates

| gate | how |
| --- | --- |
| composed md5 | `SCTOPK_FINAL_MD5`, **required** — unset or non-hex is a hard refusal, printed as `FINAL_GATE_FAIL`. A default would let a stale subject through the gate it exists to close. Also asserted by `md5sum` in the job script before python starts, and echoed in `cdk job desc` as part of the command. |
| activation | the `_final` substring in the Pallas scope name, read from the `pl.kernel` `name=` kwarg in `rank.py` STAGE 3 — fatal. Plus the HLO scope-name check. |
| substring, never `endswith` | probe 23 found `v2`/`v2big` **prepend** their tag while `v4`/`v11`/`v12` **append** it, so an `endswith` checker is convention-dependent. This one is not. |
| foreign suffixes | **non-fatal here, deliberately.** The composite merges three variants and may legitimately carry their tags in its scope name; aborting on that would be a false negative on a correct build. STAGE 8 prints any legacy tag it sees as a NOTE. |
| `base` is not the composite | `ALL_SUFFIXES = ["_final"]`, so a bare incumbent name carrying `_final` still aborts. |
| the harness itself | `final_rank.py` is the one staged file still under edit, so `make_run_cmd.py` computes its md5 at generation time rather than pinning it. It is still gated **in the job** — the value is baked into RUN_CMD at submit time and is visible in the recipe read-back, so the harness that runs is provably the harness that was staged. |

## Submitted — what ran

1. `cp ../../final/sparsecore_topk.py stage/subjects/final/sparsecore_topk.py` — 36,599 B,
   md5 **`109a12a572116c00c795ee170256db33`**, confirmed at source and again after the copy. No
   sibling-module import: the file needs nothing staged beyond itself.
2. `python3 make_run_cmd.py 109a12a572116c00c795ee170256db33 > run_cmd.txt` — 1,263 B, one line,
   **7 `exit 1` guards**, zero `'`, zero `#`, `bash -n` clean.
3. `cdk … job create simple-run-command DOCKER_IMAGE=…:fa63d6df RUN_CMD="$(cat run_cmd.txt)"
   --map-dir "$(pwd)/stage:tpu-job:/root/p24" -t agent-jobs,measurer --active-deadline-seconds 14400`
   → **`j-4dd339df-0495-4d6c-9afc`**.
4. `cdk job recipe` read back; RUN_CMD md5 matches what was sent (see the header). ⚠ That check
   verifies **sent**, not working — the in-job gate banners are the evidence the run was real.
5. Expect `MAP_GATE_OK` → `FINAL_ARMS_OK` → `MD5_GATE_OK` → `DRY_RUN_OK` → `ARM_CHECK_OK` →
   `SMOKE_OK` → `CAPTURE_OK` → `RANKING_DONE` → `FINAL_DONE`.

Probe 23 took **61.9 s** of pod time with 11 modules; this pool is 5 slots per cell-bracket instead
of 11, so it should be well under that.

⚠ **The activation gate reads a substring, and it needs to.** The composite's real scope name is
`sc_topk_direct_b{b}_n{n}_k{k}[_keys]_c{chunk}_final` — production is
**`sc_topk_direct_b1024_n10240_k2048_c2048_final`**. `_final` is a *suffix of the whole name* here
only by luck of `v4`'s tag ordering; the substring test does not depend on that, an `endswith` test
would have, and a naive `== "_final"` tag comparison would have failed outright.

## The build as delivered — the composer's facts, recorded before the number lands

**Composition: `v2` + `v4-c2048` + `v12` + a constants cleanup.** `c2048` composed; the `ragged`
fallback was **not needed**. Diff `final/variant.diff` (`fca4c3ac608d611ce84ca6e2484fbadb`),
368+/88−.

**Correctness is green and it is evidence, not a green run** — job `j-49ee915f-6906-43dc-ba3f`:
68/0/0/0 on both shards, xgate 30/30 **bit-identical**, and a **negative control that went RED** at
max score difference 4.8411946. The md5 was gated twice. A bit-identical output also closes every
value-mediated mechanism, so the correctness job doubles as the cheapest causal filter available
here.

⚠ **`v2` and `v4` were not disjoint.** Four mechanical collisions, plus a **fifth that `diff`
reported as clean** because the added text matched `v4`'s relocated copy as *context*. A mechanical
diff-intersection test would have missed that one and declared the merge safe. Worth carrying: the
absence of a conflict in a three-way diff is not evidence of independence when one side moved code.

**VMEM: +1024 B, agreed by two independent methods, zero spills**, 14.0% of budget. `v2`'s −15,360 B
very nearly pays for `v4`'s +16,384 B double buffer. Tagged **`static-compile`**.

**`v12`'s call site is confirmed in situ** — `streamindex_topk.py:831`, casting to f32, with the
`kLoop` fusion becoming a free `bitcast`; bytes model ≈24.9 µs against the measured −25.15 µs. It
stays **`static-compile`, not `measured`**, because the AOT lowering feeds from a *parameter* where
production feeds from a *custom-call output*. This is exactly the pre-registered risk (4) above, and
this job's two brackets are what test whether the glue deletion survived composition.

⚠ **One qualification to carry into the report: `_pick_chunk` has a small-chunk cliff.** 6,208 of
49,608 reachable `(b, n, k)` combinations — **12.5%** — land under 128 words. It is inherited from
`v4-c2048`, it is **unreached by any production or frozen-suite shape**, and it is deliberately
**not patched**: the md5 is already gated inside a submitted job, so touching the file now would
invalidate the gate and the correctness evidence behind it. Routed as a follow-up.

**The raggedness share, corrected.** "Two thirds" was the *modelled* split (8.08 / 12.12 = 66.7%).
Measured, from probe 23's ranking table: the raggedness-only arm captures **61% of `v4`'s whole-call
win (7.09 / 11.59 µs)** and **55% inside (6.17 / 11.19 µs)**. Quote the pair — it is a metric family
whose members disagree by 6 points, and the bracket you pick decides whether the two halves of `v4`
look 3:2 or nearly even.

## ✅ The result — `j-4dd339df-0495-4d6c-9afc`, Succeeded, 34.8 s of ranking

All nine gates green in order, no lockfile retry, `inside_kernel_dropped = []`. Outputs synced here:
`final.json` (`5cfe4faad4684b6d02d271e8d99a5445`), `ranking.json`
(`283d5a74abbaf2668d160bfd89ff7c7e`), `final_1.log`, `job.log`.

**Activation, from the observed scope name, not from the env that was set:**
`sc_topk_direct_b1024_n10240_k2048_c2048_final` and `sc_topk_direct_b64_n10240_k2048_c2048_final`.
The incumbent's four slots all read the bare `sc_topk_direct_b{1024,64}_n10240_k2048`. The substring
test was the right choice — `_final` is last here only by luck of `v4`'s tag ordering.

**Headline — pre-registered reading (2), `P.prefillX | whole`: `275.196 → 200.989 µs`, `−74.206 µs`,
RESOLVED FASTER**, all three estimators, against a 0.200 µs floor. The composition succeeded.

| cell / bracket | incumbent median µs | composite µs | Δ µs corrected | floor µs | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **`P.prefillX` whole** | 275.196 | **200.989** | **−74.206** | 0.200 | ✅ **RESOLVED FASTER** |
| `P.prefillX` inside | 240.593 | 191.325 | −49.268 | 0.105 | ✅ RESOLVED FASTER |
| `D.decodeX` whole | 49.487 | 44.751 | −4.737 | 0.465 | ✅ RESOLVED FASTER |
| `D.decodeX` inside | 46.747 | 42.760 | −3.987 | 0.155 | ⚠ NOT RESOLVED — 2/3 |
| `P.rho1` whole *(control)* | 425.511 | 341.134 | −84.377 | 1.419 | ✅ RESOLVED FASTER |
| `P.rho1` inside *(control)* | 391.927 | 333.834 | −58.093 | 3.362 | ✅ RESOLVED FASTER |

**Reading (3) resolves in favour of "`v12`'s glue deletion survived."** Whole −74.206 µs against
inside −49.268 µs — the whole-call delta materially exceeds the inside-kernel delta, by 24.9 µs. This
is a within-job difference of two brackets, which is the whole reason both were measured. It also
lands on `v12`'s static bytes model (≈24.9 µs) and its standalone −25.15 µs; that agreement is a
cross-check, **not** the source of the claim. Reading (4)'s risk is thereby mitigated further but
still not *closed* — the bracket is the real 3-D production entry, which is the strongest thing this
harness can say, and `v12`'s share stays `static-compile`.

**Reading (1) honoured: no sum was computed, anywhere.** And the composer's finding makes that more
than a formality — `v2` and `v4` collided in five places, so their deltas are not additive quantities
and a sum is not a prediction the composite could fall short of.

### ⛔ Reading (5)'s mechanism was wrong — the conclusion survives, the reason does not

Reading (5) as pre-registered said the dispatch floor "sits inside every absolute" and therefore
inflates every denominator; my first write-up of this job then argued that because the floor moved
31.53 → 22.86 µs between probes 23 and 24, "the same physical win quotes as a different percentage in
the two jobs." **Both claims are false, and the data that refutes them is in these two jobs.**

**The baselines did not move.** The incumbent reproduced across the two jobs at
`P.prefillX|whole` 275.213 → 275.196 (**0.017 µs, 0.006%**), `P.prefillX|inside` 240.659 → 240.593,
`D.decodeX|whole` 49.302 → 49.557. An 8.67 µs/call cost that was genuinely inside the bracket and
genuinely changed between jobs would have moved the prefill incumbent by 8.67 µs and the decode
incumbent by ~18%. **A percentage would in fact have travelled between exactly these two jobs.**

Of the three candidate explanations, the answer is **(1) primarily, with (3) as a separate and
independently evidenced finding. (2) is excluded** — no arithmetic error; both floors are what the
frozen instrument returned.

**(1) The floor is not additive, so it is not in the denominator at all.** `_time` launches `reps`
calls and calls `block_until_ready` **once** at the end, so the marginal cost of one more call is
`max(host issue, device execution)` — **not their sum**. `dispatch_floor()` times `jax.jit(a + 1.0)`
on an `(8,128)` array, where the device work is ~nothing and the host is the bottleneck, so its slope
*is* the host issue rate. In a 275 µs device-bound arm that same issue cost is entirely hidden. The
harness's own comment, "it is IN every absolute below", is wrong in the same way and was the source of
the error.

`hostbound_pct` measures the exposed intercept directly and settles it:

| cell (base slot) | measured `hostbound_pct` | implied exposed intercept | what it would read if 22.86 µs were inside |
| --- | ---: | ---: | ---: |
| `P.prefillX` whole, probe 24 | **1.16%** | 3.22 µs | 8.3% |
| `D.decodeX` whole, probe 24 | **5.06%** | 2.58 µs | **46.1%** |
| `P.prefillX` whole, probe 23 | **1.14%** | 3.18 µs | 11.5% (at 31.53) |
| `D.decodeX` whole, probe 23 | **5.45%** | 2.80 µs | **64.0%** (at 31.53) |

The retracted probe-23 sentence — "up to 31.5 µs of the ~49 µs decode baseline is dispatch, so a
decode percentage understates the fractional kernel change by up to ~2.8×" — is refuted by a column of
the very table it was printed under. **It should be struck from `final_report.md`, not softened.** The
real exposed per-call intercept is **~2.6–3.2 µs**, an order of magnitude smaller, and it is roughly
constant across both jobs and both cells.

The floor is still worth printing, in its correct sense: **a resolution bound.** You cannot measure a
per-call time below ~23–35 µs with this harness, because below that the host becomes the bottleneck
and you are timing the issue rate. `probes/19` is the consistency check — its *fitted* intercept
(32.013 / 31.685 µs) lands on its floor (31.19 µs), exactly as it must, because at zero descriptors
the call has no device work left and falls back to the host issue rate.

**(3) And the floor estimate is separately the least stable number in the pool.** Five job-level
estimates of the same quantity, from the same frozen function:

| job / probe | `dispatch_floor_us` |
| --- | ---: |
| `probes/24-final` | **22.86** |
| `probes/19-descriptor-cost` | 31.19 |
| `probes/23-ranking` | 31.53 |
| `probes/18-rho-denominator` | 31.96 |
| `probes/15-mandatory-v3-v10` | 35.18 |

**A ±40% spread on the instrument used to interpret a measurement that reproduces to 0.016%.** The
cause is visible in `dispatch_floor()`: one warm pass, one `lo`, one `hi`, a slope from a single
round — **no pooling, no interleaving, no replication, no null arm.** It is the one measurement in
these jobs that satisfies *none* of the nine mandatory probe properties, in a file that spends forty
lines explaining why each of them is mandatory. The general form is worth carrying: **a normalizing
constant is a measurement too, and it can easily be the noisiest number on the page — check its
stability before you divide by it.**

**The conclusion is unchanged and now rests on plainer ground.** Quote µs because the denominator is
**bracket-specific** — the same win is 26.96% of `whole` and 20.48% of `inside` — and **cell-specific**,
and because the quantity that converts into an e2e claim is absolute device time freed per call.
`−74.206 µs` travels; `−26.96%` needs its bracket, its cell and its baseline attached to mean anything.

**On `−26.98%` vs `−26.96%`:** the coordinator is right that my pair did not divide. `−26.98%` is
`74.268 / 275.257`, the **base-referenced** delta over the **base** slot; the corrected pair is
`74.206 / 275.196 = **26.96%**`, delta-vs-median over the incumbent median. I quoted the corrected µs
against the uncorrected percentage. **`−74.206 µs = −26.96%` is the self-consistent pair** and is what
the report should print; the 0.02-point difference is the one-signed correction, and it is immaterial.

**Reading (6) stands: decode is not the headline.** `D.decodeX | whole` resolved at −4.737 µs, which
at ~1.2% of decode device time is worth **~0.08% of the decode step at most**.

⚠ **`D.decodeX | inside` — NOT RESOLVED, and the decomposition says more than the flag.** All three
point estimates agree (−3.987 / −4.257 / −4.044 µs) and two clear by 2469% and 528%. The single
failure is the pooled slope, whose floor blows out to **5.062 µs** because one incumbent slot,
`#null_c`, read `us = 40.900` against its own `us_median = 46.663` and `us_unpaired = 46.298`, at
**hostbound 15.0% where every other slot in the cell is ~5%**. A host-bound excursion corrupted that
slot's pooled slope; the range floor absorbed the excursion and swallowed a real effect. The verdict
stands as pre-registered — the rule is not reached back and changed — but this is one contaminated
slot, not three estimators disagreeing, and it is exactly why `us_median` was the pre-registered read.
Same discipline as the marginal flag: print the spread beside the verdict, never the verdict alone.

**STAGE 8 fired on 3 of 6 cells and changed nothing.** `P.prefillX` (both brackets) and
`P.rho1 | inside` came out one-signed, so the corrected delta governed; the largest correction was
1.563 µs against a 58 µs effect. **The `[MARGINAL]` flag did not fire in any cell.** A correction
built for marginal calls should do nothing to a −74 µs one, and it did nothing.

**STAGE 2 smoke re-confirms correctness independently of the composer's job**: all 15 arm × cell
combinations `rows_wrong=0 oor=0 live_bad=0 bit==base=True EXACT`. Bit-identical output also closes
every value-mediated mechanism for the speedup — it cannot be coming from doing less work on the
values.

---

## For `final_report.md` — method note, lift verbatim

⏳ Placed here and not in `final/` because `final/` is the composer's working directory and is empty
as of writing; dropping a file into it mid-build is how two agents overwrite each other.

> **A resolution floor measured from one slot is measured from a biased slot.** The harness gives the
> incumbent four slots — `base` plus three nulls that are the same source loaded as separate module
> objects — and takes the floor as the range of the four. But it took every candidate's delta against
> `base`, one particular slot. When the three nulls come out **one-signed**, `base` is sitting at an
> extreme of its own floor, and a candidate at the incumbent centroid shows a delta of up to a full
> floor width and reads as a win. In the GLM-5.2 sctopk decode inside-kernel cell the nulls were
> −0.205, −0.211 and −0.103 µs — all one-signed — and four printed verdicts were noise. The fix is
> one line of statistics: **measure the delta against the median of the incumbent slots and keep the
> range as the floor.** Pair it with a three-estimator agreement test (`us_median`, the pooled slope,
> and the unpaired estimate must agree in sign and each clear its own corrected floor); `us_median`
> alone was the pre-registered read and it was not sufficient.
>
> **Apply the correction to every row, not only the rows you already doubt.** That is what separates
> an instrument from an adjustment. Applied uniformly here it removed four wins and *added* one — a
> promotion for a candidate that had already been adopted on other evidence. **A correction that only
> ever removes wins is one that was applied only to the rows you already doubted.**
>
> **When you validate a corrected estimator against known arms, count the anchors honestly.** Five
> replays were run; four reproduce conclusions that already contained the hand cross-checks the rule
> was written from, so they prove the code implements the reasoning, not that the reasoning is right.
> The one external anchor was an arm that is a null **by construction** — a ragged-row optimization
> at ρ=1, where there are no ragged rows — because its ground truth was fixed by its own mechanism
> before any measurement. One external anchor, four implementation checks.
>
> **Report a marginal resolved call with its margin attached — and with its spread.** One promotion
> above cleared its floor by 0.61% on its weakest estimator. It is resolved by the pre-registered
> rule and the rule was not reached back and changed, but "resolved by 0.61% of the floor" and
> "resolved" read very differently to someone deciding what to build on. The *decomposition* then
> says more than the flag does: the other two estimators cleared by 98.2% and 80.7%, and one
> estimator on the line while the others clear widely is a different object from three clustered at
> the boundary. The first says the call is probably real and that one estimator is the conservative
> one; the second says the effect itself is borderline. Same flag, opposite readings — so print the
> spread beside the flag, never the flag alone.
>
> **A floor taken as the range of N slots is set by the worst slot, so one contaminated slot can
> swallow a real effect.** In the composed-candidate job the decode inside-kernel cell came out NOT
> RESOLVED on the pooled-slope estimator alone, whose floor blew out to 5.06 µs against 0.16 µs for
> `us_median`. The cause was a single incumbent slot reading `us = 40.9` against its own
> `us_median = 46.7`, at **15% host-bound where every other slot in the cell was ~5%** — a host-bound
> excursion corrupting one slope, and the range floor faithfully reporting it. Keep the verdict: the
> rule was pre-registered and reaching back to change it is how a floor stops being a floor. But
> print the four slot values and their host-bound fractions next to it, because "three estimators
> disagree about an effect" and "one slot went host-bound" are different conclusions with the same
> printed verdict, and only the second one is a defect in the instrument.
>
> **A normalizing constant is a measurement too — check its stability before you divide by it.** Two
> of these probes carried the claim that a ~31.5 µs/call dispatch floor "sits inside every absolute"
> and therefore inflates every percentage-of-call, by up to ~2.8× at decode. It was wrong twice. The
> harness launches `reps` calls and blocks **once**, so per-call host issue and device time combine as
> **`max`, not sum**: a device-bound arm hides its issue cost, and the harness's own `hostbound_pct`
> put the exposed intercept at **2.6–3.2 µs**, not 31.5 — had 31.5 µs really been inside the 49.3 µs
> decode call, that column would have read 64%, and it read 5.45%. And the constant itself was the
> least stable number in the whole exercise: **22.86 / 31.19 / 31.53 / 31.96 / 35.18 µs across five
> jobs, a ±40% spread**, because `dispatch_floor()` is a single unpooled, un-interleaved,
> unreplicated shot with no null — the one measurement that satisfies none of the nine probe
> properties, sitting in a file that argues for all nine. The kernel it was being used to interpret
> reproduced across jobs to **0.016%**. The tell is cheap and general: **if your correction factor is
> less reproducible than the thing you are correcting, it is a bound, not a correction.** The
> conclusion it was supporting — quote µs, not percent-of-call — was right on other grounds
> (the denominator is bracket- and cell-specific, and only absolute freed time converts to e2e), which
> is exactly why nobody checked it.

## Files

| | |
| --- | --- |
| `stage/final_rank.py` | the wrapper: arm table, STAGE 8, STAGE 9 |
| `stage/rank.py` | **unmodified**, `e0bde4c103d277aa933fa585c1c49603` |
| `stage/p18_probe.py` | `ceafc734d10632672d3db48f2cea23f6` |
| `stage/glm_dsa_block_bench.py` | frozen instrument, `08375d7068365e1a776c58fef09cd6d0` |
| `stage/subjects/base/` | incumbent, `44353b347c96e0eedbd742c710d4540a` |
| `stage/subjects/final/` | the composite, `109a12a572116c00c795ee170256db33`, 36,599 B |
| `stage/final_rank.py` md5 | `ac5650822168fe6284bb28d3549c2722`, gated in-job |
| `make_run_cmd.py` | RUN_CMD generator with the quote/hash/`bash -n` assertions |
| `run_cmd.txt` | what was sent, `a24612a7f51432714ea7306aa620d2bb` |
| `recipe.txt` | the `cdk job recipe` read-back for `j-4dd339df-0495-4d6c-9afc` |
| `final.json` | STAGE 8/9 output, `5cfe4faad4684b6d02d271e8d99a5445` |
| `ranking.json` | `rank.py` STAGE 6 output, `283d5a74abbaf2668d160bfd89ff7c7e` |
| `job.log` / `final_1.log` | the full container log |
