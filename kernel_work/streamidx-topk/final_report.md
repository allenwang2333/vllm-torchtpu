# `streamindex_topk` — final report

**Status: REOPENED 2026-09-03 — do not review as final. §7's "land the merged op" is WITHDRAWN.**

Three things landed after this was written and two of them change the recommendation:

1. **D1 is not safe to ship as written.** §7 recommends it; `j-d17207d1-05fb-4e10-93b3` measured the
   regime the deleted branch actually serves. At runtime `ctx` 1024/2048 with `k=2048` the incumbent
   takes the fast path in **7.7 µs with no stage A at all**; D1 runs the full scoring path,
   **7.44×/7.60× slower**. It buys **4.67%** at production. Risk 5's "provably never fires" is wrong.
2. **`bkv_p` sized to `max_model_len` is not safe either.** The block is the quantum of overfetch and
   `bkv_p` is compile-time static. At `mml=9216` the recommender returns 9, so every request scans
   9216 positions: **+98% on the scoring path at ctx 1024**. The win lives only in
   `6144 < ctx ≤ 9216`. Composed, merged is **14.7× slower** than the incumbent at ctx 1024.
3. **Option B is back in the candidate set** — the runner change it needs is now authorized.
   **B×C = −26.18% on the prefill op** at G=4, `j-caf973f4-163d-42e0-89e6`. §5's "defer" is stale.

What survives unqualified: **Option C, prefill-gated**. Its composition is now properly tested (the
old additivity check was algebraically circular on the C axis) and the genuine three-way residual is
**+0.113% / +0.005% / −0.026% / −0.046%** across the four prefill configs.

Also: ~5.5 pp of the headline −20.24% is scan-extent reduction unique to `ctx=8704 / mml=9216`; the
grid's honest expectation at buckets whose context divides evenly is −13% to −16%.

A runner prototype and a `D1-lite` variant are being measured. No end-to-end benchmark has been run.

The shipping candidate is **three changes composed into one op**, measured together, index-identical
to the incumbent at every config tested. Detail lives in `results.md` (per-option) and
`variants/merged/RESULTS.md` (the composition sweep); this file is the summary the change is raised
with.

---

## 1. The number, and what it is a number *of*

**Op-level: −20.24% at the production decode point, −7.6% at prefill, no config worse than −7.65%.**

**Model-level: ~0.77% of GLM-5.2 decode device time.** That conversion is the most important line
in this report and it is new — see §4.

---

## 2. What ships

`variants/merged/streamindex_topk_merged.py`. Three changes, at three different layers:

| # | change | layer | measured alone |
| --- | --- | --- | ---: |
| 1 | `bkv_p` — decode block size from `recommended_num_kv_pages_per_block()` instead of the constant 2 | **call site** | −14.6% of op |
| 2 | **D1** — delete the early-exit `lax.cond` and its `_fast_path` branch | **caller / control flow** | −10.0% of op |
| 3 | **Option C** — head-major score layout, **gated to prefill/mixed** | **kernel** | −20.6% of prefill stage A |

Only (3) is a kernel change. (1) is a tuple, (2) is the removal of a branch. Nothing in
`src/vllm_torchtpu/` has been modified; the merged module is a copy so the incumbent stays
measurable.

### Two things about (1) that are load-bearing

`recommended_num_kv_pages_per_block` is **not** `cdiv(max_model_len, page_size)`, and it is **not**
the constant 9, though both are right at the production shape:

- flat `cdiv` gives a 64 MiB bkv double buffer against 63.94 MiB of TensorCore VMEM at ctx ≥ 32768 —
  a compile failure (`Used 85.00M of 63.94M vmem`, reproduced on the pod) across a third of the
  decode grid. The function descends a block-count ladder to the first rung inside a measured budget.
- the obvious ladder then returns `p=11` at `pages_per_seq=32`, where `align_to(32,11)=33` widens
  `num_sublanes_total` 256→264 **for every launch**. That one returns a number rather than an error,
  and the number would be a comparison against *more work* sold as a tuning win. Rungs are filtered
  to those preserving `num_sublanes_total`, verified identical at all 17 configs.

Both bugs were mine and were caught by the sweep harness, not by inspection. `bkv_p` is
compile-time static while `num_bkv` is runtime, so a server serving several context buckets needs
this **compiled per bucket**; a single fixed value for all buckets gives back most of the win.

### Why (3) is gated

Option C measured **+59.62% at decode** — a 1.6× regression. At `bq_sz=1` the head-major
permutation is literally the identity and the compiler still fails to make it free; LLO confirms it
*adds* work. The gate is not a refinement, it is the difference between a win and a large loss. The
two DECODE launches get `q_layout="head_minor"`; only the MIXED launch gets `"head_major"`.

---

## 3. Evidence

| what | job | result |
| --- | --- | --- |
| **composition sweep, 5 arms × 17 configs** | `j-2c00d88c-e7de-415f-968b` | Succeeded, 4/4 shards `rc=0`, 85 traces, `hlo_stats` device self time |
| **correctness** | `j-2c00d88c-e7de-415f-968b` shard 3 | **17/17 `array_identical`** — element-for-element in order, 10 720 rows, 0 mismatches |
| repeatability bound | `j-f06ee0fd-8592-46cf-a2ce` | two independent jobs disagree by up to **1.01%** on the same arm |
| Option C timing | `j-a960ad54-a0a1-4e1d-8ade` | −20.64% prefill stage A, chain slope r² ≥ 0.99999 |
| Option C `num_bkv` sweep | `j-fd7b8801-fce9-46ec-bb63` | −17.81% … −20.69%, monotone, never crosses zero; decode +59.62% |
| Option C correctness | `j-cf9e7189-8331-4613-9c2a` | index-identical |
| **model-level denominator** | `j-9b23bb32-0426-4080-be91` | op is **3.80%** of a GLM-5.2 decode step |

**Method.** `hlo_stats` device self time throughout, cross-checked by chain slope. Per-call wall
clock is `max(device, host)`, not `device + host` — the host dispatch floor is a **clamp**, and
subtracting it is invalid. It was subtracted in early work here and produced numbers wrong by up to
10×; `results.md` §1 tabulates the corrections. No number in this report has a dispatch floor
subtracted from it.

### Full-op change, `inc` → `merged`

| decode ctx | b8 | b32 | b64 |
| --- | ---: | ---: | ---: |
| 4096 | −9.94% | −16.79% | −16.24% |
| 8704 *(production, mml 9216)* | — | — | **−20.24%** |
| 16384 | −7.65% | −13.59% | −14.77% |
| 32768 | −9.00% | −14.63% | −16.30% |
| 65536 | −11.21% | −16.34% | −16.02% |

Prefill: `q1024/ctx4096` −7.51%, `q1024/ctx16384` −7.56%, `q4096/ctx4096` −7.92%,
`q4096/ctx16384` −7.63%.

**Composition is additive.** Largest departure from additivity is 0.72% of `inc`, ≤0.2% on 14 of 17
cells — below the 1.01% instrument resolution. The three changes land in disjoint buckets: (1) in
decode stage A, (2) as a flat 7.6–8.5 µs off *every* config regardless of batch/ctx/phase, (3) in
prefill stage A.

---

## 4. What this is worth on the model, and the honest reading

The op is **3.80% of a GLM-5.2 decode step** — `j-9b23bb32-0426-4080-be91`, recipe read back as
`GLM-5.2-FP8 --max-model-len=9216 --max-num-seqs=64 --max-num-batched-tokens=1024`, rank 8, mean of
decode steps 1–4, reproduced on rank 12 to within 0.2%. Stage A 2.80%, stage B 1.00%, against a
67.756 ms TC-busy denominator.

**So −20.24% of the op is ~0.77% of decode device time.** Prefill is worse: the op is order 1% of a
prefill step, so −7.6% there is ~0.08%.

**`report.md`'s 32.8%-of-decode figure must not be used.** It is AgentX —
`max_model_len=262144`, `n=65536`, TP=16 — a ~9× mismatch with the nightly shape this exercise
targeted. It went unnoticed for the whole exercise because `cdk job list` defaults to 32 jobs and
the GLM-5.2 profile predates that window, so an earlier search concluded no denominator existed.
This is the single largest error in the exercise and it is one of framing, not arithmetic: the work
was correct and aimed at something smaller than believed.

**Prefill is Amdahl-capped.** −7.6% rather than C's −20% because **stage B is 53–60% of the prefill
op** and nothing here touches it. Stage B is the largest untouched block in this op and has no
modelled floor at all — `hardware-facts.md` carries no measured SparseCore fact.

**The bigger neighbour.** `sc_dsa_gather` — the gather consuming the top-k indices this op produces
— is **4.205 ms/step, 6.21% of the decode step, 1.6× this entire op**. The whole DSA-indexer complex
is 10.0%. If the goal is a faster GLM-5.2 rather than a faster kernel, that is the larger object and
it was never in scope here.

---

## 5. What does not ship

| option | verdict | why |
| --- | --- | --- |
| **A** — 128-byte K record | reject | −15.30 µs against its own −13% floor of 18.64 µs, **17.9% short**. 94.3% of the fetch is already hidden; byte reduction contributes +17.8 ns/block. The mechanism was settled (scale placement, not record width) and the only reader-side route needs a bf16 K operand — which *is* Option F |
| **B** — two-unit pipeline | **STALE — reopened, see banner** | Written when the runner change was out of scope. Sign convention below was also imported unconverted from Option B's document (`+` = faster there, `−` = faster here): the correct figure in this document's convention is **−24.4%** prefill on the op alone, and **−26.18%** for B×C at G=4 (`j-caf973f4-163d-42e0-89e6`). The runner change is now authorized and a prototype is being measured |
| **E** — `_pick_partition` | reject | Correct (1 100 arms, 0 mismatches) but lands on a bucket `min_token_size=16` makes unreachable |
| **F** — bf16 operands | reject | Premise false: both matmul operands are already `float8_e4m3fn`. bf16 doubles bytes and halves MXU throughput. Measured **+4.63% prefill / +9.80% decode** |

A B×C composition measurement is still running; it can only affect B's disposition, which is already
"defer".

---

## 6. Risks and gaps

1. **No end-to-end benchmark has been run.** Every number is a microbenchmark or a profile read.
   The 0.77% is arithmetic on two measurements, not an observed model speedup.
2. **`/critical-review` has not run on this report.** Required before anything is called `accepted`.
3. **`bkv_p` needs per-bucket compilation** to deliver the table above. Deployed as one fixed value
   across buckets, most of the win is lost, and at `dec_b8_*` the merged win is almost entirely D1.
4. **The microbenchmark overstates stage A by 19%** (111.7 µs/layer vs the profile's 90.3), so
   stage-A-driven wins carried to the model are slightly optimistic. Stage B agrees to ~2%.
5. **D1 removes a branch that has tests.** `test_streamindex_topk_early_exit_*` test the path being
   deleted. The branch provably never fires at production context lengths (it needs the whole
   compressed sequence to fit in `k`), and `hlo_stats` shows `branch_1_fun` with zero rows — but the
   tests need deleting or rewriting with the change.
6. **Correctness is index-identity against the incumbent, not against ground truth.** That is the
   right gate here — an absolute recall-vs-f64 gate was shown to fail forever on correct code
   (`results.md` §11) — but it inherits any error the incumbent has.

---

## 7. Recommendation

**Land the merged op.** It is index-identical at 17/17 configs, additive, wins everywhere tested,
and two of its three parts are not kernel code. Gate C to prefill and compile `bkv_p` per context
bucket; neither is optional.

**Then stop optimizing this op.** At 3.80% of a decode step it has ~3% of device time left to give
in total, and the changes above take a quarter of it. `sc_dsa_gather` at 6.21%, and the 10.0%
DSA-indexer complex around it, is where the next hour is better spent.
