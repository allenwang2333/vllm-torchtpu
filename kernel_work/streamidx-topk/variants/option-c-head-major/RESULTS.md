# Option C — head-major score layout, built and measured

**Verdict in one line: Option C is real, and it is a prefill-only change. Prefill stage A goes
246.58 µs → 195.69 µs, −20.64%, measured by chain slope at r² ≥ 0.99999, and holds −17.8% to −20.7%
across the whole prefill `num_bkv` range (§7); all 7,936 reduction `vrot.slane` are gone, exactly as
R3 predicted; the +4,523/+4,649 spill increase does not eat the win. But **decode regresses
+59.62%** (§7.3), which nobody predicted, so C must be gated off there. Two things in the brief were
wrong and both are corrected below: the 335.1 µs stage-A baseline is 36% too high, and R3's
optimistic `head_major` arm — q head-major in HBM — does not compile. Separately, §8 clears the
shipping kernel of the f64-oracle disagreement §5 turned up: those are benign near-ties, not a
defect.**

## Provenance

| | |
| --- | --- |
| jobs | **`j-a960ad54-a0a1-4e1d-8ade`** (`Succeeded`) — timing, unpinned, whole pod<br>**`j-43359a39-0495-4602-a79b`** (`Succeeded`) — LLO dumps + first correctness attempt, 4-way fan-out. Its gate 1 is **void**, see §5<br>**`j-cf9e7189-8331-4613-9c2a`** (`Succeeded`, `FANOUT_RC=0`, 4/4 shards `rc=0`) — correctness re-run after the harness fix, plus the LLO re-read at the committed md5. **This is the correctness job.**<br>**`j-fd7b8801-fce9-46ec-bb63`** (`Succeeded`, `rc=0`) — the `num_bkv` sweep of §7, unpinned, whole pod<br>**`j-5d02a60e-2266-477c-82d6`** (`Succeeded`, `rc=0`) — the oracle-gap probe of §8; not a timing, so fan-out safe |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` (prebuilt; no image was built for this work) |
| `RUN_CMD` (read back from `cdk job recipe j-a960ad54-a0a1-4e1d-8ade`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/timing_c.sh` |
| `RUN_CMD` (read back from `cdk job recipe j-43359a39-0495-4602-a79b`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/fanout/tpu-fanout.sh "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh incumbent head_minor" "bash …/shard_llo_c.sh head_major" "bash …/shard_llo_c.sh head_major_hbm" "bash …/shard_correct_c.sh all"; echo FANOUT_RC=$?; echo ===== OPTC READ =====; python …/read_llo_c.py --dump …` |
| `RUN_CMD` (read back from `cdk job recipe j-fd7b8801-fce9-46ec-bb63`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/numbkv_c.sh` |
| `RUN_CMD` (read back from `cdk job recipe j-5d02a60e-2266-477c-82d6`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/oraclegap_c.sh` |
| `RUN_CMD` (read back from `cdk job recipe j-cf9e7189-8331-4613-9c2a`) | `cd /root/cloud-devkit/vllm-torchtpu && bash /root/fanout/tpu-fanout.sh "bash /root/kw/variants/option-c-head-major/shard_correct_c.sh all" "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh incumbent" "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh head_minor" "bash /root/kw/variants/option-c-head-major/shard_llo_c.sh head_major"; echo FANOUT_RC=$?; echo ===== OPTC READ =====; python /root/kw/variants/option-c-head-major/read_llo_c.py --dump incumbent=$CDK_OUTPUT_DIR/optc/pass/incumbent --dump head_minor=$CDK_OUTPUT_DIR/optc/pass/head_minor --dump head_major=$CDK_OUTPUT_DIR/optc/pass/head_major --json $CDK_OUTPUT_DIR/optc/llo_c.json` |
| `--map-dir` | `…/vllm-torchtpu/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`, `…/vllm-torchtpu/tests:tpu-job:/root/cloud-devkit/vllm-torchtpu/tests`, `…/kernel_work/streamidx-topk:tpu-job:/root/kw`, and (fan-out jobs only) the skill's `fanout` dir at `/root/fanout`. Every one printed a `Packaging dir` line |
| the mapping took | `MODULE /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532` on every shard of every job — the branch's kernel at `c3a18c95`, not the image's older one. Test file `md5=becf9f23ace9424a7d67cd3b34de9afd` |
| incumbent commit | `c3a18c95`; `src/.../streamindex_topk.py`, `sparsecore_topk.py` and `mla_attention_op.py` are **unmodified** — the variant is a copy |
| pinning | timing job: **none**, one unpinned process on the whole 4-chip pod, deliberately (absolute µs must be comparable). LLO/correctness jobs: 4-way fan-out, which is fine because nothing in them is a timing |
| hardware | `TPU7x`, jax/jaxlib `0.10.2` |

**Nothing here is an end-to-end model benchmark, and nothing here went to `/log-experiment`.**
Everything in §2 and §3 is *static compile-time analysis* — instructions the compiler emitted, not
instructions the machine executed. §1 and §7 are the measurements; §8 is a numerical comparison,
not a timing.

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
   introducing an error. It is a pre-existing property of the score path. **It was subsequently
   investigated — see §8. The answer is benign near-ties; the shipping kernel is not wrong.**

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

1. ~~**The transpose amortizes over `num_bkv`, and `num_bkv` is 5 here.**~~ **Retired — measured in
   §7.1 and the concern does not survive.** The prefill curve is monotone and never crosses zero:
   −17.81% at `num_bkv=1`, −20.69% at `num_bkv=8`. The whole amortisation effect is 2.9 percentage
   points, and the worst case is still a large win.
2. **`head_major_hbm` is not implementable, so the ceiling is lower than R3's range suggests.**
   R3's headline −9.24% call-level arm is unavailable (§4); only the −7.88% end is real, and after
   §1's baseline correction even that becomes ≈ −5.9% *modelled*.
3. **Nothing above is a call-level or end-to-end number.** The 50.9 µs is a stage-A delta. The
   857.3 µs call figure it would be divided into is inherited and, given that the other inherited
   figure in this brief turned out 36% wrong, should be re-measured before anyone quotes −5.9%.
4. **Decode regresses +59.62%, and this is now the strongest caveat on the list.** ~~§3.2 predicts
   −0% to −2% because `bq_sz=1` leaves nothing to reorder.~~ Measured in §7.3: stage A goes
   109.83 → 175.31 µs at `prod_decode_b64_ctx8704`. The prediction — mine as well as §3.2's — was
   that the `bq_sz=1` permutation is the identity and therefore free. It is the identity and it is
   not free. **Gating C to the prefill/MIXED call is a hard requirement, not a preference.**
5. **The +50% spill traffic is a static count and is real.** It did not cost anything at this
   shape and at this VMEM budget. A different `bq_sz`, `bkv_sz` or `vmem_limit_bytes` moves the
   register pressure, and this is the term that would turn a win into a loss if it moved. Any
   shape C is enabled at should have its bundle count checked, which is minutes.
6. ~~**The oracle disagrees with the incumbent** (§5, 10 of 2,560 indices).~~ **Resolved in §8, and
   it is not a defect in anything.** All 31 disagreements across three seeds are near-ties: every
   one is a sign flip where the kernel's own f32 scores rank the displacer higher, every gap is
   below one σ of the kernel's scoring error, stage B's selection is `EXACT`, and the row lengths
   agree. "Matches the incumbent" remains the strongest correctness statement available for C, but
   the incumbent is no longer under suspicion. What *is* now known is that the absolute
   recall-vs-f64 criterion is the wrong gate for this op — see §8.4(1) — and that stage A's score
   error is larger than the source explains (§8.4(2)).

### 6.3 Recommendation

Take `head_major`, **gated to the prefill/MIXED call site and explicitly off for decode**
(§7.3: +59.62%). Keep `q_layout` as the switch so the incumbent ordering stays reachable and the
comparison stays reproducible; do not delete the `head_minor` path — and note it is now doing a
second job, since decode needs it. The `num_bkv` check that 6.2(1) said to run before landing has
been run (§7) and it passes across the whole prefill range, so nothing further gates this at the
kernel level; the remaining unknown is call-level and end-to-end, which is not something a
microbenchmark can settle. Do not carry the −9.24%, the −7.9%, or the 335.1 µs figure forward
anywhere; carry **−50.9 µs of stage A, −20.64% at `q1024/ctx9216`, −17.8% to −20.7% across prefill
`num_bkv` 1–8, and +59.6% at decode.**

---

## 7. Follow-up: does the win survive at small `num_bkv`? — yes in prefill, **no in decode**

§6.2(1) named the transpose's amortisation over `num_bkv` as the likeliest way Option C
disappoints. It was measured. Job `j-fd7b8801-fce9-46ec-bb63`, `bench_numbkv.py`, same chain-slope
method, best-of-7, one unpinned process on the whole pod, `rc=0`, no failed arm.

```
===== OPTION C vs num_bkv (chain slope, device us/call) =====
shape                        ph        nbkv  nbq   bq   incumbent  head_major   delta%     min r2
prefill_q1024_ctx2048        prefill      1    8  128       52.60       43.23   -17.81   0.999834
prefill_q1024_ctx4096        prefill      2    8  128      100.85       81.35   -19.34   0.999993
prefill_q1024_ctx6144        prefill      3    8  128      149.31      119.23   -20.15   0.999985
prefill_q1024_ctx8192        prefill      4    8  128      197.69      157.67   -20.25   0.999997
prefill_q1024_ctx9216        prefill      5    8  128      246.35      196.25   -20.34   0.999996
prefill_q1024_ctx16384       prefill      8    8  128      392.28      311.13   -20.69   0.999998
decode_b64_ctx8704           decode       5    1    1      109.83      175.31   +59.62   0.999995
```

`bkv_sz = 2048` throughout (`num_kv_pages_per_block=(2,2,2)` at `mla_attention_op.py:119`, page
size a literal 1024 at `attention.py:1323`) and `compression_ratio = 1`, so
`num_bkv = cdiv(context_len, 2048)`. The prefill points up to ctx 9216 hold `max_model_len = 9216`
— which is what all three `prod_prefill_q1024_ctx*` rows do — so `num_sublanes_total`, the score
buffer and the output shape are **constant** across those points and `num_bkv` is the only thing
moving. ctx 16384 is outside that control and is marked as such; ctx 32768 was deliberately not
run (deep in the amortised regime the curve already establishes, and it costs four more Mosaic
compiles to learn nothing).

### 7.1 Prefill: the concern was real, and it is small

The curve is monotone and it **never crosses zero**. Worst case is `num_bkv = 1` at **−17.81%**;
the asymptote is about **−20.7%**. The entire amortisation effect is 2.9 percentage points.
**§6.2(1) is retired**: there is no prefill shape at which Option C is flat, let alone negative.

Two supporting checks. `numeric vs incumbent` is `same_mask: True` with `max_abs` of 6.1e-05 or
1.22e-04 at every point, so the fast arm is not a wrong arm anywhere on the curve. And ctx 9216
reproduces §1 across two independent jobs on different pods — 246.35 vs 246.58 µs (0.09% apart) and
196.25 vs 195.69 µs (0.29%) — which is the best evidence in this document that the chain-slope
method is repeatable.

### 7.2 Prefill `num_bkv = 1` is mostly unreachable anyway — a second-order point, not the answer

Early exit is on in production (`STREAMIDX_ENABLE_EARLY_EXIT = True`, a module-level literal at
`mla_attention_op.py:57`, no env override) and its condition is
`jnp.max(seq_lens) // compression_ratio <= k` (`streamindex_topk.py:859-864`), which at
`compression_ratio = 1` is `max(seq_lens) <= k`. For GLM-5.2 `k = topk_tokens = 2048`, which is
**exactly `bkv_sz`** — so a call whose longest sequence would give `num_bkv = 1` takes `_fast_path`
and runs no stage A at all.

Two caveats keep this from being the whole answer, which is why the point was measured rather than
argued away: `num_bkv` is recomputed per grid step from the *group's* max
(`streamindex_topk.py:275-276`), so a short sequence sharing a batch with a long one does execute a
`num_bkv = 1` step; and `k` comes from the model spec (`self.indexer.topk_tokens`,
`mla_attention_op.py:510`), so a model with `k < 2048` reaches `num_bkv = 1` for real. §7.1 covers
both cases and says they are fine.

### 7.3 Decode: **+59.62%. This is the finding, and it scopes the recommendation**

Head-major must **not** be enabled on the decode call. Stage A goes 109.83 → 175.31 µs at
`prod_decode_b64_ctx8704` — a 65.5 µs regression — with r² 0.999995 on both arms.

What makes this worth saying loudly is that it contradicts an analytic prediction that was already
written down and believed. Decode uses `num_queries_per_block[0] = 1` and `static_q_len = 1`, so
`bq_sz = min(1,1) = 1`, and at `bq_sz = 1` the head-major permutation is **the identity**:
head-minor's row index is `t*H + h = h` and head-major's is `h*bq_sz + t = h`; the reshape
`(SB,1,H,D) → transpose(0,2,1,3) → (SB,H*1,D)` yields the same `(SB,H,D)` as `(SB,1*H,D)`. §3.2
predicted −0% to −2% on exactly this reasoning, and so did I. The data says +59.62%. **Reordering
nothing is not free.**

**Mechanism: not established.** No LLO dump was taken of the decode program, so what follows is
judgement, not a finding. The plausible cause is that the degenerate `(SB,H,1,D)` intermediate puts
a length-1 axis on the sublane dimension, where Mosaic's `(8,128)` tiling pads each head's `1×128`
row to a full tile — 8× the vector registers for the same data — turning a no-op permutation into a
real relayout. It has a cheap test (one `shard_llo_c.sh` run against a decode launcher) and nobody
needs to run it: the decision is the same either way.

Two limits on the number. It is **stage A only** — decode stage A's share of the whole
`streamindex_topk` call was not measured, so +59.62% of stage A is not +59.62% of the op. And it is
**one decode shape**. Neither limit is load-bearing: the sign is unambiguous and the remedy is to
not enable the layout there. Note the decode shape used is `b=64`, not the instrument's
`prod_decode_b8_ctx8704`, which is synthetic — `min_token_size = 16` is hardcoded at
`tpu_platform.py:1236-1276`, so bucket b=8 is never compiled.

---

## 8. Follow-up: the f64 oracle's disagreement with the **shipping** kernel

§5 flagged that gate 2's absolute recall criterion fails on the incumbent as well as on Option C —
`min_recall 0.996094`, 10 of 2,560 indices, byte-identical across arms — and said somebody should
look. Job `j-5d02a60e-2266-477c-82d6`, `probe_oracle_gap.py`, three seeds, `rc=0`, no traceback.

**Answer: benign near-ties. The shipping kernel is not wrong.** Every gap is smaller than the
kernel's own scoring error, and in the kernel's own arithmetic every "missed" item genuinely ranks
*below* the item that displaced it.

### 8.1 The probe captures the shipping kernel, not a reconstruction of it

`probe_oracle_gap.py` patches the module-global `streamindex_topk.sparsecore_topk` with a spy and
calls the shipping `streamindex_topk.__wrapped__` inside one trace, so what is compared is the
exact array production hands to stage B. No launcher is transcribed, so none can drift.
`enable_early_exit` stays at its default `False`, which keeps `_common_path` out of a `lax.cond`
(making the captured tracer legal to return) and changes nothing here — `max(seq_lens)//cr = 2048 >
k = 512`, so the fast path could not have fired. Confirmed on the pod:
`streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532`, the branch's kernel.

### 8.2 Three candidate causes; two are eliminated outright

| candidate | result | verdict |
| --- | --- | --- |
| stage B picked the wrong items | `top-k of the kernel's OWN f32 scores vs returned indices: rows_differing=0/8 indices_differing=0 -> EXACT`, all 6 seed×arm runs | **eliminated.** Independently confirms probe P2 against a different reference |
| `_effective_row_lengths` truncates the search | `_effective_row_lengths vs oracle visible count: ALL AGREE` (`[2047,2047,2047,2047,2048,2048,2048,2048]`), all 6 runs | **eliminated** |
| stage A's f32 score | `max=1.062794e+00 p99=5.293128e-01 median=1.246159e-01 max_rel=6.320e-03` over 16,380 positions, on scores up to \|344.87\| (seed 0) | **this is the whole of it** |

### 8.3 The gaps, and why they are near-ties

Verbatim, seed 0, `incumbent` (head_major is identical row for row; seeds 1 and 2 are in the job
log). `gap_f64` is the oracle's score margin between the missed item and the one that displaced it;
`gap_f32` is the same margin computed in the kernel's own arithmetic; σ is the kernel's max score
error, 1.0628:

```
tok    idx  orank   s_f64(missed)   s_f32(missed)   disp    s_f64(disp)      gap_f64      gap_f32  gap/sigma
  1   1451    506    168.25961736    167.84741211    531   167.92745876    3.322e-01   -3.287e-01       0.31
  2   1229    510    161.08455229    160.88854980   1750   160.97339869    1.112e-01   -9.030e-02       0.10
  3    164    507    170.16551080    169.74064636   1047   169.64879879    5.167e-01   -3.555e-02       0.49
  4    484    510    153.35880325    153.11758423     24   153.00872025    3.501e-01   -3.820e-01       0.33
  4    975    511    153.34041163    152.97906494   1984   153.06846052    2.720e-01   -2.175e-01       0.26
  5    676    508    166.34600592    165.75311279   1054   166.15408105    1.919e-01   -4.597e-01       0.18
  6   1938    507    174.29222816    173.96865845   1197   173.86395270    4.283e-01   -2.676e-01       0.40
  6   1508    511    174.01062727    173.90795898    987   173.94995540    6.067e-02   -5.011e-01       0.06
  7    622    509    182.13083488    181.88305664   1861   181.81577235    3.151e-01   -1.804e-01       0.30
  7   1714    511    181.99227560    181.89883423   1545   181.97142100    2.085e-02   -1.711e-01       0.02
```

Read the last two numeric columns. **`gap_f64 > 0` and `gap_f32 < 0` in all 31 disagreements across
all three seeds, without a single exception.** The oracle says the missed item is better; the
kernel's own scores say the displacer is better; and the kernel returned what its own scores told
it to. Given the scores it computed, the selection is *right* — which is the same thing §8.2's
`EXACT` row says, arrived at from the other direction.

That leaves the question of whether those scores were off by an amount the kernel should be
embarrassed by. Recomputed from the raw table (`err = |s_f64 − s_f32|` at each of the 62 positions
involved across all three seeds):

| | value |
| --- | --- |
| largest error at a missed item | **0.5929** |
| largest error at a displacer | **0.6925** |
| largest \|`gap_f64`\| anywhere | **0.6701** |
| the kernel's own error distribution (16,380 positions) | median **0.125**, p99 **0.53**, max **1.06** |
| positions whose error exceeds p99 of that distribution | **2 of 62**, at 1.12× and 1.33× p99, both below the max |

Every score that participated in a flip is an ordinary draw from the kernel's own error
distribution — not one outlier among them. The verdict rule was fixed in the probe's docstring
**before any number existed** (`max|gap_f64| / σ ≤ 4` → benign), and the measured ratios are
**0.49 / 0.58 / 0.66** across the three seeds: every gap is smaller than a *single* σ, let alone
four.

The underlying reason is structural. At rank 512 of 2,048 the true score separation is a few tenths
on scores of ~170, while the kernel's scoring noise is the same order — the oracle itself reports
`boundary gap min=2.525e-04`. **Items that close together are not distinguishable by this kernel at
all**, so which of them comes back is arbitrary and no downstream consumer can tell. That is a
property of an approximate indexer, not a defect.

### 8.4 Two things this turns up that are worth acting on

1. **An absolute "recall vs f64 ≥ 0.999" gate is the wrong gate for this op**, and will fail
   forever on correct code — §5 watched it fail on the incumbent. The discriminating statistic is
   arm-vs-arm index-set agreement, which is what `run_correctness_c.py` reports and which Option C
   passes perfectly (0/8 rows, 0 indices). Whoever writes correctness gates for `streamindex_topk`
   should not have to re-derive this.
2. **The size of stage A's score error is unexplained, and I could not account for it from the
   source.** Median relative error is 7.4e-04 and the max is 6.3e-03 — bf16-scale, roughly 10⁴×
   what f32 accumulation over 32 heads should produce. The score path reads as f32 throughout:
   `jnp.einsum(..., preferred_element_type=jnp.float32)` (`streamindex_topk.py:326-330`),
   `jnp.maximum(s, 0.0)`, weights explicitly cast to f32 (`:334`, and `prepare_index_weights` at
   `:458-472` does no quantization). The only bf16 in the path is the e8m0 scale at `:268`, and a
   power of two is lossless in bf16, so that is not it either. Candidates I cannot rule out from
   Python: the MXU's internal fp8 accumulation path, or Mosaic lowering the f32×bf16 multiply at
   `:341` in bf16. **This is judgement, not a finding** — nothing was dumped or instrumented to
   test it. It does not disturb the benign verdict (§8.3 is computed from the kernel's own numbers
   and does not care where the error came from), but a 7e-04-relative indexer score is worth
   someone knowing deliberately rather than by accident.

### 8.5 One caveat on the `head_major` half of this job

The `head_major` rows are byte-identical to the incumbent's in all three seeds — every σ, every
gap, every row of every table. That is consistent with the reassociation being exactly
score-preserving at this configuration (`bq_sz=8`, `T=8`, and the 32-way head sum running over the
same `h` order in both layouts), and it matches §5's finding that the two arms select identical
index sets. But **this job contains no line proving the two arms compiled to different programs**,
so on its own evidence it cannot distinguish "identical output" from "the same program run twice".
That the `head_major` path is real and distinct is established elsewhere — §7's sweep shows it
−17.8% to −20.7% faster with non-zero `max_abs` at every prefill point, and §2 shows a different
opcode histogram — not here. The incumbent row, which is the row this section is about, is
unaffected either way.

---

## 9. What this does not establish

1. **§1–§6 are prefill only, one shape** — `prod_prefill_q1024_ctx9216`, `bq_sz=128`,
   `seq_batch_size=1`, one sequence. §7 widens this to six prefill contexts and **one** decode
   shape (`b=64, ctx=8704`); decode remains a single point, and its +59.62% mechanism is not
   established (no LLO dump of the decode program was taken — §7.3 says why nobody needs to). Note
   `min_token_size=16` is hardcoded in `tpu_platform.py:1236-1276`, so the instrument's
   `prod_decode_b8_ctx8704` row is synthetic and no b=8 figure appears anywhere here.
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

```bash
# §7 num_bkv sweep: one unpinned process on the whole pod (it is a timing)
cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/numbkv_c.sh

# §8 oracle-gap probe: not a timing, safe to share a pod
cd /root/cloud-devkit/vllm-torchtpu && bash /root/kw/variants/option-c-head-major/oraclegap_c.sh
```

Files: `streamindex_topk_c.py` (the kernel copy + `q_layout`), `stage_a_c.py` (stage-A launcher,
four arms), `bench_headmajor.py` (chain-slope timing), `bench_numbkv.py` (§7's sweep — reuses
`bench_headmajor`'s `run_arm`/`_fit` verbatim and swaps in a decode launcher), `dump_llo_c.py` +
`read_llo_c.py` (static analysis), `run_correctness_c.py` (both gates), `probe_oracle_gap.py`
(§8's stage-A-vs-stage-B localisation), and the six `*.sh` shard drivers.
