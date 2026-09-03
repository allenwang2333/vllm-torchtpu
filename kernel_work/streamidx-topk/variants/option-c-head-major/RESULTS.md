# Option C — head-major score layout, built and measured

**Verdict in one line: Option C is real. Prefill stage A goes 246.58 µs → 195.69 µs, −20.64%,
measured by chain slope at r² ≥ 0.99999; all 7,936 reduction `vrot.slane` are gone, exactly as R3
predicted; the +4,523/+4,649 spill increase does not eat the win. Two things in the brief were
wrong and both are corrected below: the 335.1 µs stage-A baseline is 36% too high, and R3's
optimistic `head_major` arm — q head-major in HBM — does not compile.**

## Provenance

| | |
| --- | --- |
| jobs | **`j-a960ad54-a0a1-4e1d-8ade`** (`Succeeded`) — timing, unpinned, whole pod<br>**`j-43359a39-0495-4602-a79b`** (`Succeeded`) — LLO dumps + first correctness attempt, 4-way fan-out. Its gate 1 is **void**, see §5<br>**`j-cf9e7189-8331-4613-9c2a`** (`Succeeded`, `FANOUT_RC=0`, 4/4 shards `rc=0`) — correctness re-run after the harness fix, plus the LLO re-read at the committed md5. **This is the correctness job.** |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` (prebuilt; no image was built for this work) |
| `RUN_CMD` (read back from `cdk job recipe j-a960ad54-a0a1-4e1d-8ade`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/timing_c.sh` |
| `RUN_CMD` (read back from `cdk job recipe j-43359a39-0495-4602-a79b`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/fanout/tpu-fanout.sh "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh incumbent head_minor" "bash …/shard_llo_c.sh head_major" "bash …/shard_llo_c.sh head_major_hbm" "bash …/shard_correct_c.sh all"; echo FANOUT_RC=$?; echo ===== OPTC READ =====; python …/read_llo_c.py --dump …` |
| `RUN_CMD` (read back from `cdk job recipe j-cf9e7189-8331-4613-9c2a`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/fanout/tpu-fanout.sh "bash /root/kw/variants/option-c-head-major/shard_correct_c.sh all" "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh incumbent" "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh head_minor" "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh head_major"; echo FANOUT_RC=$?; echo ===== OPTC READ =====; python /root/kw/variants/option-c-head-major/read_llo_c.py --dump incumbent=$CDK_OUTPUT_DIR/optc/pass/incumbent --dump head_minor=$CDK_OUTPUT_DIR/optc/pass/head_minor --dump head_major=$CDK_OUTPUT_DIR/optc/pass/head_major --json $CDK_OUTPUT_DIR/optc/llo_c.json` |
| `--map-dir` | `…/vllm-torchtpu/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`, `…/vllm-torchtpu/tests:tpu-job:/root/cloud-devkit/vllm-torchtpu/tests`, `…/kernel_work/streamidx-topk:tpu-job:/root/kw`, and (fan-out jobs only) the skill's `fanout` dir at `/root/fanout`. Every one printed a `Packaging dir` line |
| the mapping took | `MODULE /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532` on every shard of every job — the branch's kernel at `c3a18c95`, not the image's older one. Test file `md5=becf9f23ace9424a7d67cd3b34de9afd` |
| incumbent commit | `c3a18c95`; `src/.../streamindex_topk.py`, `sparsecore_topk.py` and `mla_attention_op.py` are **unmodified** — the variant is a copy |
| pinning | timing job: **none**, one unpinned process on the whole 4-chip pod, deliberately (absolute µs must be comparable). LLO/correctness jobs: 4-way fan-out, which is fine because nothing in them is a timing |
| hardware | `TPU7x`, jax/jaxlib `0.10.2` |

**Nothing here is an end-to-end model benchmark, and nothing here went to `/log-experiment`.**
Everything in §2 and §3 is *static compile-time analysis* — instructions the compiler emitted, not
instructions the machine executed. Only §1 is a measurement.

---

## 0. What was built, and what "Option C" turned out to mean

`variants/option-c-head-major/streamindex_topk_c.py` is the incumbent copied verbatim at
`c3a18c95` and given one new static parameter, `q_layout`, threaded from `streamindex_topk` into
`kernel`. Three values, and the difference between them is the whole story:

| arm | what it is | R3's counterpart |
| --- | --- | --- |
| `head_minor` | the incumbent's ordering, through the copy. **The identity control.** | `head_minor` |
| `head_major` | **Option C as actually deployable.** The q block is transposed `(bq_sz,H,D)→(H,bq_sz,D)` *inside* the kernel, once per bq block. Nothing outside the Pallas call changes — not q's HBM layout, not `_fetch_bq`'s DMA, not `mla_attention_op.py`. | `head_major_xpose` |
| `head_major_hbm` | q pre-transposed to `(H,T,D)` in HBM by `prepare_q_inputs`, so the kernel gets head-major rows with no in-kernel shuffle. R3's optimistic bound. | `head_major` |

A fourth arm, `incumbent`, launches the **unmodified** `src/.../streamindex_topk.py::kernel`
through the same wrapper. It exists because `stage_a_c.py` is a hand transcription of
`run_scores_kernel`, and a transcription is somewhere a divergence can hide. It is measured, not
asserted: `head_minor` came back **bit-identical** to `incumbent` and within +0.09% on time.

**The in-kernel transpose is affordable because `load_bq` runs 8 times per call at this shape and
`compute_scores` runs 40.** The shuffle is paid 8 times to save 40 — which is why `head_major`,
and not `head_major_hbm`, is the shape of the real option.

---

## 1. Prefill stage-A time — the measurement

Method: **chain slope**, per `probes/stage-b-ksweep.md` §0. `time_arm` enqueues asynchronously and
blocks once, so its per-call number is `max(device, host)`; a dispatch floor is a clamp, not an
addend, and subtracting one is invalid. So R ∈ {1,2,3,4} stage-A calls are chained inside a single
`jit`, linked by the kernel's own `input_output_aliases` `scores_init → output` dependency — a real
data dependency that cannot be CSEd, reordered or folded, and one that needs no artificial
`+ min(out[0,0], 0)` term. The **slope** is marginal device time per call; the **intercept** is the
host term. Best-of-9. Geometry `prod_prefill_q1024_ctx9216`, verbatim from the job log:

```
[probe] geometry: {'num_seqs': 1, 'q_len': 1024, 'context_len': 9216, 'bq_sz': 128, 'bkv_sz': 2048,
 'num_bq': 8, 'num_bkv': 5, 'num_tokens': 1024, 'num_sublanes_total': 80, 'total_pages': 9}
```

```
===== STAGE A PREFILL, bq_sz=128 =====
arm               chain slope us        r2  batched wall us  serial wall us
incumbent                 246.58  1.000000           267.49          421.87
head_minor                246.36  0.999997           267.51          437.78
head_major                195.69  0.999999           217.56          367.72
head_major_hbm    FAILED/skipped
  incumbent            vs head_minor: +0.09%
  head_major           vs head_minor: -20.57%
```

Full-precision r²: incumbent `0.999999524929704`, `head_minor` `0.9999973728767706`, `head_major`
`0.9999990020629973`. The two wall-clock columns are reported because the ksweep probe requires
all three statistics to be distinguishable; **they are not the result** and must not be compared
against a chain slope.

| | chain slope, device µs/call |
| --- | ---: |
| incumbent (`src/…/streamindex_topk.py`, untouched) | **246.58** |
| `head_minor` (the copy, incumbent ordering) | 246.36 |
| **`head_major` (Option C)** | **195.69** |
| Δ vs incumbent | **−50.89 µs, −20.64%** |
| Δ vs `head_minor` | −50.67 µs, −20.57% |
| identity control (`incumbent` vs `head_minor`) | +0.09%, and bit-identical output |

### Against the pass criterion — and why the criterion's premise was wrong

The brief set **stage A ≤ 280 µs, down from 335.1 µs**. Option C measures **195.69 µs**, so the
bar is cleared with 84 µs to spare.

**But the bar was not discriminating, because the incumbent already clears it.** Measured by the
same method on the same job, stage A at this shape is **246.58 µs, not 335.1 µs** — the target
figure was 36% high. This is not a surprise: `probes/stage-b-ksweep.md` already flagged that
`bench_prod`'s derived stage A of 335.07 µs is "a subtraction across inconsistent statistics" and
"the largest remaining uncertainty in the prefill record". It now has a direct measurement instead
of a subtraction, and the subtraction was wrong by 88.5 µs.

**So the number to carry forward is the ratio, −20.64%, and the absolute delta, −50.9 µs — not
"stage A becomes 255.9 µs".** Anything downstream that was sized off 335.1 µs needs re-deriving.
In particular: §3.2 predicted a **−7.9%** call-level win by taking 67.7 µs off an 857.3 µs call.
The real stage-A saving is 50.9 µs. If the 857.3 µs call figure is itself sound — I did not measure
it, and given what just happened to 335.1 it should be re-measured before it is trusted — the
call-level effect is **−5.9%, not −7.9%**. That is *modelled*, from one measured term and one
inherited one.

---

## 2. Mechanism — `vrot.slane`, in situ

Static, from `…-53-post-ra.txt` of the real `StreamIdxTC[-hm]-m-bq_128-bkvp_2.1` program (not
R3's standalone proxy). Rotate-amount histogram, verbatim:

| rotate amount | incumbent | `head_minor` | **`head_major`** |
| --- | ---: | ---: | ---: |
| by 1 | 2,304 | 2,304 | 0 |
| by 2 | 2,304 | 2,304 | 0 |
| by 3 | 256 | 256 | 0 |
| by 4 | 2,304 | 2,304 | 0 |
| by 5 | 256 | 256 | 0 |
| by 6 | 256 | 256 | 0 |
| by 7 | 256 | 256 | 0 |
| by register | 1,024 | 1,024 | **1,280** |
| **total** | **8,960** | **8,960** | **1,280** |

**The criterion was `vrot.slane` → 0 and the measured answer is 1,280, so read literally it fails —
but the literal reading is the wrong one, and the mechanism is confirmed exactly.** Decomposed:

- **Every one of the 7,936 immediate-amount rotates is gone.** 6,912 (the 1/2/4 binary tree) +
  1,024 (the 3/5/6/7 pack) = 7,936. That is **precisely** the 7,936 R3 attributed to the head
  reduction, to the instruction. R3's attribution was right, and it is now confirmed on the real
  program rather than a proxy.
- **The 1,024 register-amount rotates are the incumbent's, not Option C's.** R3 already identified
  them as unrelated dynamic relayout and explicitly called them not removable by this change. They
  are present in the incumbent and unchanged.
- **The +256 are the in-kernel transpose paying for itself.** R3's `head_major_xpose` proxy showed
  exactly 256 as well; its zero-rotate `head_major` arm is the one that does not compile (§4).

So: **the time improved and the rotates it was attributed to did go away.** The attribution is not
coincidental. The rest of the census moves the way the mechanism requires — `vadd` 12,320 → 7,968,
`vmul` 10,240 → 8,448, all instructions 56,690 → 42,143 (−25.7%) — while the two controls hold:
`vmatmul` stays at **1,024** on every arm (no arm won by losing its matmul) and `vmax` stays at
**8,192** (the ReLU is untouched). `vpack` 0 → 384, `vshrl` 1 → 385 and `vcombine` 1,408 → 1,792
are the transpose's cost, and they are the reason 256 rotates come back.

---

## 3. Spills, in situ — the question R3 could not settle

R3's counterweight was measured on a standalone proxy with **645** spill slots; the incumbent runs
at **1,375**, so the concern was that head-major's live-range extension — 32 addends per output
vreg, 16 vregs apart — would be *worse* under real register pressure. Measured on the real program:

| | incumbent | `head_minor` | **`head_major`** | Δ vs incumbent |
| --- | ---: | ---: | ---: | ---: |
| spill stores | 8,458 | 8,458 | **12,981** | **+4,523 (+53.5%)** |
| fill loads | 10,130 | 10,130 | **14,779** | **+4,649 (+45.9%)** |
| spill slots | 1,375 | 1,375 | **1,162** | **−213 (−15.5%)** |
| **scheduled bundles** | **11,600** | **11,600** | **9,404** | **−2,196 (−18.93%)** |

**No — the spills do not eat the win, and the answer is not close.** Three things settle it:

1. **The spill delta did not get worse in situ.** R3's proxy showed +4,203/+4,355; the real program
   shows +4,523/+4,649. Essentially the same magnitude at more than twice the register pressure.
   The pessimistic reading of R3 §7.4 is disconfirmed.
2. **Spill *slots* went down, not up** — 1,375 → 1,162. Head-major extends live ranges but removes
   far more short-lived reduction temporaries than it adds long-lived accumulators.
3. **The schedule is 18.93% shorter, and that is the statistic that can net the two**, because a
   removed VALU op and an added spill issue on different slots so op counts cannot be subtracted
   from each other. −15,987 VALU against +9,172 spill/fill nets to −2,196 bundles.

And the measurement agrees with the schedule: −20.64% of device time against −18.93% of bundles.
The bundle count *slightly under-predicts* the win (a bundle-ratio model would have said 199.9 µs;
the measurement says 195.69 µs), which is the opposite of the failure mode to worry about.

R3's in-situ bundle ratio prediction was `head_major_xpose` at −20.16%; the real program gives
−18.93%. **R3's ratio transferred to within 1.3 points.** What did not transfer was the µs
baseline it was multiplied by (§1).

---

## 4. `head_major_hbm` does not compile — R3's optimistic arm is not implementable here

Both jobs, identically:

```
MosaicError: INTERNAL: Mosaic failed to compile TPU kernel ... HLO name
`StreamIdxTC-hmhbm-m-bq_128-bkvp_2.1`: Slice sizes along tiled dimensions must be aligned to
tiles. Failed to verify that the size at dimension 1 is divisible by the tile dimension 8.

  %301 = "tpu.memref_slice"(%299, ...) : (memref<32x128x128xf8E4M3FN,
         #tpu.tiled<(32,128)(4,1),[4,1,1]>, #tpu.memory_space<vmem>>, i32, i32, i32, i32)
      -> memref<32x?x128xf8E4M3FN, ...>
```
at `streamindex_topk_c.py:212` (`_async_copy` → `_fetch_bq` → `start_fetch_bq` → `prologue`).

**Why, and why it is structural rather than a bug in my code.** `_fetch_bq` copies a *dynamic*
number of tokens, `sz = min(bq_sz, curr_q_end - q_len_start)` — the last bq block of a sequence is
short. In the incumbent's `(T,H,D)` layout the token axis is dim 0, which is untiled, so a dynamic
slice on it is free. Putting q head-major in HBM makes the token axis the **sublane** axis, and
Mosaic requires slices on a tiled dimension to be tile-aligned. The dynamic per-sequence token
count and a head-major HBM layout are in direct conflict. Getting around it needs either
`tpu.assume_multiple` (which asserts something that is not true for a partial block) or always
copying a static `bq_sz` rows (which over-reads past the end of `q` at the last block).

This is worth stating plainly because **R3's headline `head_major` arm — 0 rotates, 8,078 bundles,
modelled −9.24% — is that configuration**, and R3 §7.2 flagged that it "assumes q arrives
head-major for free" and that "the true cost of getting q head-major is outside the four corners
of this dump". It is now inside them: the cost is that it does not build. R3's −9.24% is not an
available option; the −7.88% `head_major_xpose` end of its range is the whole range.

I did not pursue a fix. `head_major` is the deployable form, it is what the −20.64% was measured
on, and the extra arm was only ever a bound.

---

## 5. Correctness

**Established. Gate 1 passes on the unmodified suite, job `j-cf9e7189-8331-4613-9c2a`.**

### Gate 1 — `tests/kernels/test_dsv4_streamindex_topk.py`, unmodified

`shard_correct_c.sh` monkeypatches `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk`
to the variant with `q_layout` bound, then runs the suite as-is. The test file is not edited; its
md5 was read back on the pod as `becf9f23ace9424a7d67cd3b34de9afd`, matching the branch.

| arm | patch line | **verbatim summary line** | exit |
| --- | --- | --- | --- |
| `incumbent` | `[C-correct] control: the unmodified src kernel, no patch applied` | **`20 passed, 14 warnings in 36.06s`** | 0 |
| `head_minor` | `… -> jit(streamindex_topk_c(q_layout='head_minor')) type=PjitFunction` | **`20 passed, 14 warnings in 36.42s`** | 0 |
| **`head_major`** | `… -> jit(streamindex_topk_c(q_layout='head_major')) type=PjitFunction` | **`20 passed, 14 warnings in 34.51s`** | 0 |

`===== OPTC GATE1 END arm=head_major rc=0 =====`. No traceback anywhere in the 352-line log.

**The collection count is 20, which is the current bar.** The brief quoted `13 passed in 21.41s`
from baseline job `j-8b1e7acc`; tests have been added since, and 20 is what `c3a18c95` collects.
The control arm — the untouched `src` kernel through the same harness — collects 20 as well, so
this is not a stale test file. Since gate 1 passed, **gate 2 was not required.** It was run anyway
and is reported below because what it found is worth knowing.

### The first attempt at gate 1 was void, and this is why it was re-run

Job `j-43359a39`'s gate 1 reported 7 of 20 failing with
`ValueError: Non-hashable static arguments are not supported … TypeError: unhashable type: 'DynamicJaxprTracer'`
— **on `head_minor` as well as `head_major`.** Identical failures on the identity control mean the
gate discriminated nothing; it was a defect in my harness, not in Option C. Cause: I patched in a
`functools.partial(pjit_fn, q_layout=…)`, and `tests/…:208` calls `jax.eval_shape(streamindex_topk, …, k=k, …)`.
`jax.eval_shape` honours a `PjitFunction`'s `static_argnames` but double-jits any other callable,
so `k` arrived as a tracer in a static slot. Reproduced locally on jax 0.10.2 and fixed by
`bind_layout()` in `run_correctness_c.py`, which rebuilds a genuine `PjitFunction` from
`streamindex_topk.__wrapped__` with the same static names minus `q_layout` — hence the
`type=PjitFunction` readback printed in the table above, which exists specifically so this failure
cannot recur silently. `j-43359a39`'s gate-1 result should be disregarded entirely; its *static*
dumps are unaffected and are what §2 and §3 report.

### Gate 2 — f64 oracle, run as a supplement

Verbatim, `T=8, S_valid=2048, k=512, H=32`:

```
[C-correct] oracle built: T=8 S_valid=2048 k=512 H=32 masked=4; boundary gap min=2.525e-04 max=2.127e-01
[C-correct] incumbent        vs f64 oracle: min_recall=0.996094 mean=0.997559 missed=10 worst_gap=4.738e-01
[C-correct] head_minor       vs f64 oracle: min_recall=0.996094 mean=0.997559 missed=10 worst_gap=4.738e-01
[C-correct] head_major       vs f64 oracle: min_recall=0.996094 mean=0.997559 missed=10 worst_gap=4.738e-01
[C-correct] -- index-set agreement vs incumbent (the discriminating statistic) --
[C-correct] incumbent        rows_differing=0/8 indices_differing=0 worst_swap_gap=0.000e+00 identical=YES
[C-correct] head_minor       rows_differing=0/8 indices_differing=0 worst_swap_gap=0.000e+00 identical=YES
[C-correct] head_major       rows_differing=0/8 indices_differing=0 worst_swap_gap=0.000e+00 identical=YES
```

Read it in this order:

1. **`head_major` selects the identical index set to the incumbent — 0 of 8 rows differ, 0 indices
   differ.** This is the statistic that speaks to Option C, and it is as strong as it can be. Not
   one selection changed.
2. **The absolute 0.999 criterion in the brief fails, and it fails on the incumbent too** —
   min_recall 0.996094 on all three arms, byte-identical across them. Per `CLAUDE.md`, a check that
   fails on the baseline as well "establishes nothing either way". This is the f64 oracle disagreeing
   with the *existing production kernel* about 10 borderline indices out of 2,560, not Option C
   introducing an error. It is a pre-existing property of the fp8 path and is out of scope here,
   but somebody should look at it.

Numeric agreement on the raw scores, from `j-43359a39` (at the pre-hoist md5 `2999947a…`):
`head_minor` vs incumbent `bit_identical: True, max_abs: 0.0`; `head_major` vs incumbent
`bit_identical: False, same_mask: True, max_abs: 1.220703125e-04, max_rel: 2.329e-07`. Bit identity
is not expected from `head_major` and its absence is not a defect — reassociating an f32 sum of 32
terms is exactly the change being made, and 2.3e-07 relative is the size that change should have.
`head_minor`'s bit identity is the stronger statement: the copy and the hand transcription in
`stage_a_c.py` introduced nothing.

---

## 6. Verdict — should C go into the final recommendation?

**Yes, as the `head_major` in-kernel-transpose form, scoped to prefill — with the caveats in §6.2
carried alongside it, not dropped.** This is the strongest-evidenced of the options I can speak to:
the win is measured rather than modelled, the mechanism it was predicted from is confirmed
instruction-for-instruction, the countervailing cost was found and quantified rather than assumed
away, and the output is index-identical to the incumbent with the full suite green.

### 6.1 What is actually established

| claim | evidence | strength |
| --- | --- | --- |
| Prefill stage A −20.64% at `q1024/ctx9216`, `bq_sz=128` | chain slope, best-of-9, r² ≥ 0.99999, unpinned pod, `j-a960ad54` | **measured** |
| The win comes from the head reduction, not from somewhere else | all 7,936 immediate `vrot.slane` removed, exactly R3's attributed set; `vmatmul` and `vmax` unchanged | **static, but decisive** — the controls hold |
| Spills do not eat it | +4,523/+4,649 against −2,196 bundles, and slots *down* 213; the measurement beats the bundle ratio | **static + measured, agreeing** |
| Output is unchanged in the way that matters | 0/8 rows and 0 indices differ from the incumbent; `20 passed` on the unmodified suite | **measured** |
| The harness is not measuring the wrong thing | `head_minor` bit-identical to the incumbent and +0.09% on time | **measured** |

R3's ratio was right and its baseline was not: predicted −20.16% on bundles, delivered −18.93%;
predicted a 335.1 µs stage A, delivered 246.58 µs.

### 6.2 What argues against it, in the order I would want to hear it

1. **The transpose amortizes over `num_bkv`, and `num_bkv` is 5 here.** `load_bq` runs `num_bq`=8
   times and `compute_scores` runs `num_bq × num_bkv`=40, so the shuffle is paid once per five
   uses. **At `context_len ≤ bkv_sz` — `num_bkv`=1 — it is paid once per use, and Option C may be
   flat or negative.** No short-context shape was measured. This is the single most likely way C
   disappoints, it is cheap to check, and it should be checked before C is committed to: one
   `bench_headmajor.py` run at `num_bkv` ∈ {1,2,5} settles it.
2. **`head_major_hbm` is not implementable, so the ceiling is lower than R3's range suggests.**
   R3's headline −9.24% call-level arm is unavailable (§4); only the −7.88% end is real, and after
   §1's baseline correction even that becomes ≈ −5.9% *modelled*.
3. **Nothing above is a call-level or end-to-end number.** The 50.9 µs is a stage-A delta. The
   857.3 µs call figure it would be divided into is inherited and, given that the other inherited
   figure in this brief turned out 36% wrong, should be re-measured before anyone quotes −5.9%.
4. **Decode is unexamined.** §3.2 predicts −0% to −2% because `bq_sz=1` leaves nothing to reorder.
   Plausible, unmeasured. If C is gated to prefill this is moot; if it is not, the decode path
   still pays the transpose's `vpack`/`vshrl` for no reduction to remove, so a decode measurement
   is needed before enabling it there.
5. **The +50% spill traffic is a static count and is real.** It did not cost anything at this
   shape and at this VMEM budget. A different `bq_sz`, `bkv_sz` or `vmem_limit_bytes` moves the
   register pressure, and this is the term that would turn a win into a loss if it moved. Any
   shape C is enabled at should have its bundle count checked, which is minutes.
6. **The oracle disagrees with the incumbent** (§5, 10 of 2,560 indices). Not caused by C — C
   reproduces the incumbent's selections exactly — but it means "matches the incumbent" is the
   strongest correctness statement available here, and the incumbent itself has not been shown
   correct against f64.

### 6.3 Recommendation

Take `head_major`. Keep `q_layout` as the switch so the incumbent ordering stays reachable and the
comparison stays reproducible; do not delete the `head_minor` path. Before it lands, run the
`num_bkv` sweep in 6.2(1) — it is the one cheap check that could still overturn this, and it is
much cheaper than finding out at the model level. Do not carry the −9.24%, the −7.9%, or the
335.1 µs figure forward anywhere; carry **−50.9 µs of stage A, −20.64%, at `q1024/ctx9216`**.

---

## 7. What this does not establish

1. **Prefill only, one shape.** Everything is `prod_prefill_q1024_ctx9216`, `bq_sz=128`,
   `seq_batch_size=1`, one sequence. §3.2 predicts **−0% to −2% at decode** because `bq_sz=1`
   leaves nothing to reorder, and **that is still unexamined** — nothing here measured decode. Note
   also the coordinator's correction that `min_token_size=16` is hardcoded in
   `tpu_platform.py:1236-1276`, so the instrument's `prod_decode_b8_ctx8704` row is synthetic and
   no b=8 figure should be quoted as a production effect.
2. **Stage A only.** The −50.9 µs is a stage-A delta measured in isolation. The call-level and
   end-to-end effects are not measured, and the 857.3 µs call figure they would be divided into is
   inherited, not measured here — see §1 on what happened to the other inherited figure.
3. **§2, §3 and §4 are static.** They are what the compiler emitted. A static instruction mix is
   not a dynamic one; the bundle count is code length, not cycles. It happens to agree with the
   measurement here, which is reassurance, not proof.
4. **The md5 the timing was taken at — checked, not argued.** §1's timing and job `j-43359a39`'s
   dumps ran `streamindex_topk_c.py` at `md5=2999947ae33d65e9d6fd3dff10ae742f`; the committed file
   is `md5=c37e159b975935ee41482475e6cdcd12`. The difference is the module docstring and hoisting
   the `static_argnames` tuple into a named constant `STREAMINDEX_TOPK_STATIC_ARGNAMES` with
   identical members. Rather than assert that was non-semantic, job `j-cf9e7189` re-ran all three
   LLO dumps at the committed md5 (readback `c37e159b…` on every shard) and **every column of the
   summary table is unchanged from `j-43359a39`**: 8,960/12,320/10,240/8,192/1,024/8,458/10,130/1,375/11,600
   for the incumbent and 1,280/7,968/8,448/8,192/1,024/12,981/14,779/1,162/9,404 for `head_major`.
   The compiler emitted the same program, so §1's timing stands for the committed file. §1 itself
   was **not** re-timed at the new md5.
5. **`head_major_hbm` was not measured at all**, only shown not to compile. If someone makes the
   dynamic-`sz` slice work, its numbers are still open.

## Appendix — reproducing

```bash
# timing: one unpinned process on the whole pod
cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/timing_c.sh

# LLO dumps (one arm per process; LIBTPU_INIT_ARGS is read once at load) + correctness
bash /root/fanout/tpu-fanout.sh \
  "bash /root/kw/variants/option-c-head-major/shard_correct_c.sh all" \
  "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh incumbent" \
  "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh head_minor" \
  "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh head_major"

# read the dumps
python /root/kw/variants/option-c-head-major/read_llo_c.py \
  --dump incumbent=$CDK_OUTPUT_DIR/optc/pass/incumbent \
  --dump head_major=$CDK_OUTPUT_DIR/optc/pass/head_major --json /tmp/llo_c.json
```

Files: `streamindex_topk_c.py` (the kernel copy + `q_layout`), `stage_a_c.py` (stage-A launcher,
four arms), `bench_headmajor.py` (chain-slope timing), `dump_llo_c.py` + `read_llo_c.py` (static
analysis), `run_correctness_c.py` (both gates), and the four `*.sh` shard drivers.
