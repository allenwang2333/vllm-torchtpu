# Area 6 — Whether the algorithm itself should change

Constraint sheet for `streamindex_topk` (GLM-5.2 DSA lightning indexer). Step 2 briefer, area (6),
against `.claude/skills/write-tpu-kernels/references/06-algorithm.md`.

**This sheet decides nothing.** It is input to the single downstream designer.

Read for it: the reference file, `AREAS.md`, `report.md` in full including §6.3 and every inline
retraction, `streamindex_topk.py`, `sparsecore_topk.py` in full, `mla_attention_op.py`,
`deepseek_v4_indexer.py`, upstream `vllm/model_executor/models/deepseek_v2.py`,
`test_dsv4_streamindex_topk.py`, `scripts/vllm/benchmarking/check_regression.py`,
`scripts/vllm/benchmarking/baselines/eval/`, `glm-5.2-fp8-dp16-ep.sh`, and
`git show 6142d5e7` + PR #724.

Baseline of record throughout: the **nightly** config (`max_model_len` 9216, page 1024, score width
10240, `max_num_seqs` 64, `max_num_batched_tokens` 1024, k=2048, 32 indexer heads, 21 of 78 layers).
AgentX (`max_model_len` 262144, page 256, score width 262144) is a secondary operating point and is
explicitly **not** the target.

---

## A. Constraints

1. **The top-k output is order-free and tie-free.** `sparsecore_topk`'s contract is "top-k column
   indices in **unspecified order**, `-1` suffix-padded… equal boundary scores may be broken
   differently" (`sparsecore_topk.py:26-29`); the oracle compares sorted index **sets**
   (`test_dsv4_streamindex_topk.py:58-118`). A candidate may permute and may break ties arbitrarily.
   `from source`
2. **`-1` padding and "never select a position at or beyond `row_lengths`, never select `-inf`" are
   part of the contract, not an implementation detail.** `write_empty_rows=True` is passed at
   `streamindex_topk.py:839`. `from source`
3. **Exactness has never been given up on this op, and nothing in the history argues it must be
   kept.** See §C — the pre-`6142d5e7` path was `approx_max_k(recall_target=1.0)`, which is exact.
   So "the authors moved away from approximate" is **false as stated in `report.md` §8** and must
   not be used as evidence against approximation. `from git history`
4. **But approximation is still the wrong first move here, on a mechanism argument, not a policy
   one.** k=2048 of 10240 is a 1-in-5 selection. Every published approximate top-k
   (partition-and-merge, sampled thresholds, `ApproxTopK`'s `recall_target<1`) buys its speed from
   `k/n → 0`; at k/n = 0.2 the candidate set that must survive stage 1 is already a fifth of the
   row, and the incumbent's radix select is O(row) with 4 passes regardless. `modelled`
5. **A stage-B approximation cannot be gated by any existing artifact.** `vllm bench serve` runs
   `--ignore-eos`; the kernel unit test is a zero-tolerance set compare that any approximation fails
   by construction; and **there is no `glm-5.2-*.baseline.json` under
   `scripts/vllm/benchmarking/baselines/eval/`** — 32 eval baselines exist there (deepseek-v2,
   gemma-4, qwen3, qwen3.5), none for GLM-5.2. The gate would have to be *created* before it could
   be run. `from source`
6. **Any rewrite whose exactness depends on `indexer_weights >= 0` is an approximation, not an exact
   restructuring.** `weights = weights_proj(hidden_states) * q_scale * softmax_scale * n_head**-0.5`
   (`deepseek_v4_indexer.py:81`, upstream `deepseek_v2.py:817`). The three scalars are positive;
   `weights_proj` is a plain `MergedColumnParallelLinear` output with no relu/softplus/abs anywhere.
   Sign is **activation-dependent**, so it is not decidable from the checkpoint and no probe can
   prove it — only bound it empirically. `from source`
7. **f32 is forced on stage B's input.** `sparsecore_topk` raises on non-f32 (`:396`), `k` must be a
   multiple of 16 and ≤4096 (`:399`), `n` a multiple of 16 (`:392`). Narrowing scores to bf16 to
   halve the HBM round trip is an **approximation** (bf16 has 8 mantissa bits; ties across a
   10240-wide row become common) *and* would need a kernel change. `from source`

---

## B. Exact restructurings, then approximations

### Exact — free to take

- **E1. Bound stage B's HBM→VMEM DMA by `eff_row_lengths` (the largest one).** Every compute loop in
  `_topk_kernel` is already bounded by `walk_chunks = cdiv(i_hi,16)`,
  `i_hi = min(slice_len, eff_len)` (`:151-155`, and the `parallel_loop` bounds at `:225, :235, :246,
  :287`). The **DMA is not**: `:160` copies a static `slice_len = align_to(n,16)` words. So stage
  B's compute is *already* `seq_len`-bounded and only its transfer is `max_model_len`-bounded. Exact
  because everything past `eff_len` is `-inf` and masked by `valid = j < i_hi`. This resolves
  `report.md` §8's open question — "whether the radix select's cost tracks `row_lengths` or the
  buffer width" — to *both, in different terms*. See **§E and §F below** for the quantitative
  reconciliation, the convergence from three other areas, and the corrected prize. `from source`
- **E2. Per-sequence KV bound in stage A instead of the group max.** `kv_len_max =
  jnp.max(jnp.stack(kv_lens))` over `decode_req_batch_size=4` (`:275-276`); results beyond a slot's
  own `kv_len` are already masked to `-inf` at `:344`. Exact. Value = within-group length spread,
  priced by §6.3 at ≈0.050 ms per full-length slot at the profile geometry. `from source`
- **E3. Sort/group sequences by length before batching (the TODO at `:751-754`).** A permutation of
  output rows — exact by construction. It is the multiplier on E2, not an alternative to it. Needs
  the served length distribution, which no artifact here has. `from source`
- **E4. Fix the early-exit predicate to read only real sequences.** `jnp.max(seq_lens) //
  compression_ratio <= k` (`:861`) maxes over all `max_num_seqs=64` entries;
  `_effective_row_lengths` already shows the correct construction (`:492-494`, mask by
  `distribution[2]`). Exact, and the failure direction is *safe* — a stale entry can only inflate
  the max and suppress the exit, never wrongly take it. **But it is worth almost nothing at the
  baseline of record:** ISL is 8192 and k is 2048, so the predicate is false on every nightly step
  and on every AgentX step (`STREAMIDX_ENABLE_EARLY_EXIT = True`, `mla_attention_op.py:57`, so the
  branch is live but never taken). Flagging it as area (7)'s, not mine, and as a non-lever either
  way. `from source`
- **E5. Named so it is not proposed: `*w` + `sum_h` on the MXU does not pay at decode.** The
  weighted reduction over h is a per-token contraction, so as a matmul its operand is
  block-diagonal — 0.8% dense at bq_sz=128 and a 1-row matmul at bq_sz=1. `06-algorithm.md` is
  explicit that the MXU must be filled and that an 8-row pass costs 10.57× per unit of work. At
  prefill (bq_sz=128, 128 rows) the modelled arithmetic is 2.1 GFLOP at 661 Tops/s = 3.2 µs against
  25.2 Mops at 1.72 Tops/s = 14.6 µs — a modelled 4.5× — but that denominator **is the `assumed` VPU
  rate §7's probe failed to measure**, so it inherits exactly the defect that makes prefill
  `unresolved`. Do not design on it until that probe is redone in VMEM. `modelled`
- **E6. Already closed, do not reopen:** the `-inf` init (6.02 µs at 2.79 TB/s, ~2%, §6.2).
- **E7 (added after cross-area reconciliation). Cap `_pick_partition`'s `p` by a cost model, not by
  subcore utilization alone.** `_pick_partition` (`:360-378`) raises `p` as far as `max_p = 32//b`
  allows, purely to fill subcores. Each doubling of `p` doubles the stage-2 candidate width
  (`p * k_p`) and the `take_along_axis` gather that feeds it — the `gather_offload` the AgentX
  profile prices at 51.54 µs each, 2 per indexer layer. Exact (the two-stage select is exact for any
  `p`), and it is the mechanism behind two of the three §6.3 anomalies. See §F. `from source`

### Approximations available — each with error, metric, gate

- **A1. Shrink `k_p` in the 2-stage partition path.** `sparsecore_topk` currently takes
  `k_p = min(k, n_p)` per slice (`:420`) and merges `p*k_p` candidates. Setting `k_p < k` is the
  textbook approximate top-k. **Applies only when `p > 1`; at the nightly config
  `_pick_partition(64, 10240, 2048)` returns `p = 1`, so this lever does not exist at the baseline
  of record.** Error: recall of the exact top-k set, worst-case over rows. Not recommended.
- **A2. Threshold-and-drop instead of exact selection** (take everything above a score quantile,
  pad/truncate to k). Error: recall, and it is unbounded in the tail.

### The gate, concretely, for either

Three tiers, in this order, and the perf number is believed only after tier 3:

1. **Kernel tier — index-set recall.** Metric: `|selected ∩ exact| / k`, reported as the **minimum
   over rows** and the mean, against `streamindex_topk_ref` (`test_dsv4_streamindex_topk.py:58-118`,
   an exact NumPy triple loop imported by path). Threshold: **min-row recall ≥ 0.99**. Job: extend
   the existing pytest via `/run-command-in-cdk` — minutes.
2. **Kernel tier — score-mass recall**, because recall@k weights a dropped rank-2 position the same
   as a dropped rank-2047 one. Metric: `sum(scores[selected]) / sum(scores[exact top-k])`, min over
   rows. Threshold: **≥ 0.999**. Same job.
3. **Model tier — the one that actually decides.** `check_regression.py --mode eval`, whose GLM-5.2
   threshold is already named in the repo: **`EVAL_TOLERANCE="0.025"`**
   (`glm-5.2-fp8-dp16-ep.sh:80`, 2.5 percentage points; the script's own eval default is 0.015).
   The config also sets `BENCHMARK_TEMPERATURE="0"`, so greedy decoding makes a token-divergence
   check available too. **Blocker: no `baselines/eval/glm-5.2-fp8-dp16-ep.*.baseline.json` exists.**
   A GLM-5.2 eval baseline must be established on the unmodified kernel first — that cost belongs to
   the approximation, not to the repo.

Tiers 1-2 are cheap and are *not* a substitute for tier 3: this op selects which tokens the model
attends to, so a 99% recall on a synthetic-score microbenchmark says nothing about whether the 1% it
dropped was the token the answer depended on.

**Stated plainly, because it is the fact that turns "we can check accuracy later" from a plan into a
cost: nobody in this repo has ever evaluated GLM-5.2 with an approximate indexer, and there is no
`baselines/eval/glm-5.2-*` file with which to do it.** Establishing that baseline is a full
model-level eval run on the *unmodified* kernel, before the approximation is even written. Any
design that proposes an approximation and defers the gate is proposing an unbudgeted model-level run
plus a new checked-in baseline, not a follow-up test.

---

## C. What `6142d5e7` says — and it inverts the report's premise

The history says **nothing about accuracy, because no accuracy was traded.** The pre-commit call was
(`git show 6142d5e7^:src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py:817-823`):

```python
# jax.lax.approx_max_k(recall_target=1.0) is equivalent to
# jax.lax.top_k but faster.
top_vals, top_idxs = jax.lax.approx_max_k(scores, k, reduction_dimension=-1, recall_target=1.0)
```

`recall_target=1.0` is **exact**. The commit title "Replace `jax.lax.approx_max_topk` with sparse
core topk" names the API, not the semantics: it is an **exact→exact** swap, and the reason is purely
performance. PR #724's body: output token throughput 2320.69 → 2537.45 tok/s (**+9.3%**), median
TPOT 292.05 → 276.99 ms, and an xprof decode figure of **430 µs → 33.768 µs** for the top-k.
`glm-5.2-fp8-dp16-ep.baseline.json` was rewritten to those numbers in the same commit. The only
review comment (`ShobhitBehl`, APPROVED) is about updating that baseline file. There is no
discussion of recall, accuracy or eval anywhere in the PR.

Two consequences the designer must have:

- **`report.md` §8's "`sparsecore_topk` replaced `approx_max_topk` in `6142d5e7`; the reason is not
  recorded" implies a reason that does not exist, and the probe it schedules ("whether exact top-k
  is required") is asking the wrong question.** Exact top-k was never given up. Nobody in this repo
  has ever evaluated GLM-5.2 with an approximate indexer, and — per constraint 5 — nobody has an
  artifact that would let them.
- **`6142d5e7` deleted a TODO that is a live design pointer:** *"Re-evaluate replacing this with the
  sparsecore_topk kernel once SparseCore supports direct VMEM access (e.g., on TPU v8). Currently,
  jax.lax.approx_max_k wins due to the HBM read/write tax, but direct VMEM streaming will allow SC
  to beat TensorCore performance."* The author expected SparseCore to lose on the HBM tax on tpu7x
  and it won anyway by 12.7×. **A TensorCore-resident exact top-k is therefore a legitimate step-3
  candidate, not a settled loss** — it was the incumbent until 2026-08-31, and the ~43 µs/call the
  report first attributed to it was later shown to be host dispatch.

---

## D. Gaps this block turns on, and one contradiction

- **`06-algorithm.md` `## Gaps` item 3 — "No probe measures a genuine approximation being *rejected*
  by a gate. The gate machinery is untested against a real failure."** This block turns on it
  directly: if the designer proposes A1/A2, tier 3 of the gate above has never been run for *any*
  approximation in this repo and its baseline does not exist for this model. Decision that flips:
  with a working eval gate, an approximate stage B is a candidate to build and measure; without one,
  "we will check accuracy later" is the failure mode `06-algorithm.md` names, and the honest design
  records **N/A because no accuracy gate exists for GLM-5.2** rather than proposing an approximation
  it cannot evidence.
- **`## Gaps` item 5's spirit — reassociation floor vs. an *ordinal* output.** The whole reference
  file calibrates against 2.46e-06 relative on a **value**. This block's output is a **set of
  integers**, and its error metric has no relationship to that floor: a score perturbation of 1e-06
  that straddles a rank-2048 boundary flips one index, and the file offers no guidance on how many
  flipped indices are acceptable. Decision that flips: whether E1/E2/E3 (which reorder nothing) need
  any accuracy evidence at all — they do not, they are bit-identical — versus anything touching the
  *score* arithmetic, which becomes ordinal-unstable at a magnitude the file would call noise.
  **Name this as the probe worth returning to the reference file**: the sensitivity of the selected
  index set to a 1e-06 relative score perturbation at n=10240, k=2048.
- The other gaps (flash-vs-materialized, cumsum crossover, `DOES-NOT-LOWER` under jax 0.10.2, the
  unlocated triangular solve) are not reachable from any decision in this block. Leave them open.

**Contradiction between `report.md` and this area:** §8's step-2 probe list frames `6142d5e7` as
evidence bearing on whether exact selection is required; §C shows it is not, and the probe as
written ("whether exact top-k is required or approximate would do") should be replaced with the two
separate questions it conflates — *(a)* what stage B's own floor is with the DMA bounded by
`eff_row_lengths` (exact, E1, no gate needed), and *(b)* whether a GLM-5.2 eval baseline can be
established at all, which is the precondition for even discussing A1/A2.

---

## E. Convergence on E1

**Four independent readings, four routes, one conclusion.** Three other briefers reached E1 from
their own areas: the scan loops in `_topk_kernel` are bounded by `i_hi = min(slice_len, eff_len)`
while the DMA at `sparsecore_topk.py:160` fills `resident_vmem` with a static
`slice_len = align_to(n,16)`. **E1 is the best-evidenced item in the whole of step 2, and it is
exact, which puts it ahead of everything requiring a gate.** Nothing below changes that ranking by
evidence; §F does change its ranking by *prize*, and the designer needs both numbers.

### Is E1 worth taking if only the transfer can shrink and the allocation cannot?

**Yes, unambiguously, and this is the case to design for.** The two are independent:

- `resident_vmem` at a static `slice_len` costs VMEM *capacity*, which is reserved at compile time
  and costs no runtime. The tail beyond `i_hi` is simply never read: `cvt` converts only
  `walk_chunks` chunks (`:161-166`), and `load_keys` masks with `valid = (j < i_hi)` (`:170`). Stale
  bytes in the tail are harmless.
- The DMA at `:160` is the HBM traffic and the time. Shrinking it to `walk_chunks * LANES` words is
  the whole of E1.

So E1 is fully realizable with the allocation left exactly as it is. **Recommend the designer not
couple them.**

**On whether a dynamic-length DMA is even expressible:** the risk is that Pallas requires the
*size* argument of `pl.ds` to be static (only the start may be dynamic), which would block
`resident_vmem.at[pl.ds(0, i_hi)]`. The workaround is already in this codebase on the TensorCore
side — `_fetch_bkv` (`streamindex_topk.py:168-194`) emits a loop of constant-length `_async_copy`s.
The SparseCore equivalent is a `fori_loop`/`plsc.parallel_loop` bounded by
`cdiv(i_hi, CHUNK_WORDS)` issuing fixed-size `CHUNK_WORDS` copies, giving a dynamic total length at
static granularity. `CHUNK_WORDS` then trades DMA-issue overhead against quantization of the saving.
**This is a step-2 design decision and a step-3 measurement, not a blocker.** `assumed`

### Area 5's VMEM budget, and what it unlocks (which is not E1)

Area 5 reports the resident slice as `4·align(n/p,16)` bytes against
`sparse_core.vmem_capacity_bytes = 524288` **per subcore**, with ~26 KB of other scratch
(`priv_vmem` 16·256+16 words ≈ 16.4 KB, `rowbuf_vmem` k+16 words ≈ 8.3 KB, `glob_vmem` ≈ 1.1 KB,
`len_vmem` small). `MAX_SLICE_WORDS = 65536` words = 256 KiB against ≈486 KiB of headroom — area 5
calls it conservative by ~1.5×; the arithmetic above gives ~1.9×. Either way it is conservative.

**But raising `MAX_SLICE_WORDS` does not help the operating points that matter, and the designer
should know that before spending on it.** `_pick_partition` takes `p_min = cdiv(n, MAX_SLICE_WORDS)`
and then *raises* `p` further to fill subcores. Worked through:

| shape | b | n | `p_min` | `max_p = 32//b` | **p** | stage-2 width `p·k_p` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nightly decode | 64 | 10240 | 1 | 1 | **1** | — single stage |
| nightly prefill chunk | 1024 | 10240 | 1 | 1 | **1** | — single stage |
| §6.3 prefill, both widths | 1024 / 4096 | 4096 / 65536 | 1 | 1 | **1** | — single stage |
| §6.3 decode b8, n=4096 | 8 | 4096 | 1 | 4 | **2** | 4096 |
| §6.3 decode b8, n=65536 | 8 | 65536 | 1 | 4 | **4** | 8192 |
| AgentX decode | 16 | 262144 | 4 | 2 | **4** | 8192 |
| `profile_b1_ctx110592` | 1 | 262144 | 4 | 32 | **32** | 65536 |

At AgentX, `p=4` would need `MAX_SLICE_WORDS ≥ 131072` words = 512 KB > the whole per-subcore
capacity to reach `p=2`. So the VMEM headroom cannot buy a smaller `p` there. **What it can buy is
E7**: `p` for `b=1` is 32 purely because `max_p = 32//b` maximizes subcore occupancy, and that
inflates the stage-2 select 8× (65536-wide vs 8192-wide) plus its gather. Capping `p` by a cost
model is exact and is the lever the VMEM reading actually points at.

**Independent confirmation that this model of `_pick_partition` is right:** the AgentX production
HLO in §6.1 names `sc_topk_direct_b64_n65536` and `sc_topk_direct_b16_n8192`. For b=16, n=262144 the
table above gives stage 1 = `b·p = 64` rows of `n_p = 65536` and stage 2 = 16 rows of
`p·k_p = 8192`. Both names match exactly. `measured (j-a71a648c-3064-471a-a131, rank 0)`

---

## F. Reconciling E1 with §6.3 and with area 3 — E1 explains the largest ratios, and is small at the baseline

Area 3 says stage B's DMA is ~2-5% of stage B (≈0.05 TB/s demanded), which bounds E1's prize hard at
nightly. My original sheet said "~15% of stage B's transfer at nightly and ~97% at AgentX". **Both
are correct and we were pricing different things: mine is the fraction of the *transfer* reclaimed;
area 3's is the share of *stage B* the transfer occupies.** Multiplying them is what matters, and it
makes E1 small at the baseline of record. Area 3's framing is the more useful one and I defer to it.

### Fitting E1's mechanism to §6.3's prefill pairs

Each §6.3 pair holds context, page size and batch fixed and changes only `max_model_len`. Therefore
`eff_len` — and so *all* of stage B's compute, which is bounded by `i_hi` — is **identical between
the two members of a pair**. The only thing that changes is `slice_len`, i.e. the DMA. Per area 5,
prefill stays on `p=1` at both widths (confirmed in the table in §E), so the partition branch cannot
explain those rows. E1's mechanism is the only candidate left. Testing it:

| pair | b | n low → high | ΔDMA | Δt (stage B) | implied rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_q1024_ctx4096` | 1024 | 4096 → 65536 | 251.7 MB | 0.706 − 0.245 = 0.461 ms | **0.546 TB/s** |
| `prefill_q4096_ctx4096` | 4096 | 4096 → 65536 | 1006.6 MB | 2.373 − 0.695 = 1.678 ms | **0.600 TB/s** |

ΔDMA = `b · (n_hi − n_lo) · 4 B`. **Two shapes, a 4× difference in extra traffic, and one consistent
rate agreeing to 9.9%.** That is a quantitative fit, not a coincidence, and it means **E1's
unconditional-`slice_len` DMA accounts for essentially all of the 2.88× and 3.41×** — the two
largest ratios in the measured table. `measured (j-63b9f2e4-26c9-4e3a-99ce)`, rate `modelled` from
the fit.

### Why the decode rows do not falsify it

Applying the same 0.55-0.60 TB/s to the decode pairs over-predicts by 2-3×:

| pair | ΔDMA | predicted Δt @ 0.57 TB/s | measured Δt |
| --- | ---: | ---: | ---: |
| `decode_b64_ctx4096` | 15.7 MB | 27.6 µs | 9 µs |
| `decode_b32_ctx4096` | 7.9 MB | 13.8 µs | 7 µs |

**Those rows are host-floor-clamped and are not device measurements.** §6.3 records a per-shard host
dispatch floor of **31.9–39.8 µs** and flags five rows `[HOST BOUND]`. The low arms are 42 µs (b32)
and 55 µs (b64) against that floor, and their predicted *device* DMA time is 0.9 µs and 1.8 µs
respectively — so the low arm is essentially pure host dispatch. A pair of the form
`max(host, small)` vs `max(host, larger)` compresses the ratio exactly the way these do. The decode
rows neither confirm nor refute E1; they should not be used either way.

The `b8` rows (1.77×, 1.78×) are a **different mechanism**: per the §E table, `p` changes 2 → 4
across that pair, adding a wider stage-2 select and a wider `take_along_axis` gather. Stage-1 DMA is
`b·n·4` regardless of `p`, so E1 predicts only ~3.5 µs there against a measured 56 µs. **That row is
E7's, not E1's.**

### This closes §6.3's "cost model nobody here understands"

§6.3 lists three observations any explanation has to fit and concludes "the largest single component
of this op has a cost model nobody here understands… This is the first probe of step 2." **Two
mechanisms fit all three**, and both are read from source rather than hypothesized:

1. *"Effect 2–3× stronger at prefill than at decode."* — E1. Prefill has 16-64× more rows, so the
   per-row `slice_len` DMA dominates and is measurable; decode's row count leaves the whole
   measurement inside the host floor.
2. *"Small batches hit harder than large ones (b8 1.77× vs b64 1.16×)."* — E7. `max_p = 32//b`
   raises `p` only for `b < 32`, and b8 crosses `p = 2 → 4` across the pair while b32/b64 stay at
   `p = 1`.
3. *"One row of 262144 (0.340 ms) is more expensive than sixteen of the same width
   (0.254–0.269 ms)."* — E7 again, and quantitatively: `b=1` gets `p = 32` and a **65536-wide**
   stage-2 select, while `b=16` gets `p = 4` and an **8192-wide** one. 8× the stage-2 work and 8× the
   candidate gather for one sixteenth of the rows.

§6.3's own "cliff rather than a slope" hypothesis is right, and the cliff is `_pick_partition`. The
slope underneath it is E1. **Recommend the designer treat stage B's cost model as
`DMA(b · align(n,16) · 4) + compute(b · eff_len) + two_stage(p)` and probe to confirm rather than to
discover.** `modelled`, fitted to `measured (j-63b9f2e4-26c9-4e3a-99ce)` and
`measured (j-a71a648c-3064-471a-a131)`.

### E1's corrected prize — small at the baseline of record

The reclaimable fraction of stage-B DMA is `1 − mean(eff_len)/align(n,16)`. Pricing at the fitted
0.57 TB/s:

| operating point | b | n | stage-B DMA | as % of stage B | reclaimable | **E1 saves** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nightly decode, ctx 8704 | 64 | 10240 | 2.62 MB → 4.6 µs | 6.8% of 67.4 µs | 15% | **≈0.7 µs, 1% of stage B** |
| nightly prefill, last chunk ctx 9216 | 1024 | 10240 | 41.9 MB → 73.6 µs | 15% of 481 µs | 15% | **≈11 µs, 2% of stage B** |
| nightly prefill, averaged over 8 chunks | 1024 | 10240 | 41.9 MB/chunk | 15% | ≈60% | **≈44 µs/chunk, ≈9% of stage B** |
| AgentX decode, ctx 8192 | 16 | 262144 | 16.8 MB → 29 µs | large | 97% | dominant |

Area 3's "≈0.05 TB/s demanded" checks out exactly: 2.62 MB / 67.4 µs = 0.039 TB/s at nightly decode.

**The honest ranking.** E1 is exact, needs no gate, is confirmed from four independent readings, and
quantitatively explains the two largest ratios in the only deconfounded measurement this project
has. **But at the nightly baseline of record it is worth ~1% of stage B at decode and ~2-9% at
prefill, because `max_model_len` 9216 leaves almost no slack between `slice_len` and `eff_len` to
reclaim.** Its prize scales with `max_model_len / context`, which is exactly finding (1). The
designer should take it — it is cheap and free of risk — but should **not** rank it as the headline
lever at the nightly config, and should not carry the 97% figure forward, since that is AgentX and
AgentX is not the target. Its larger value is explanatory: it and E7 together retire §6.3's
"first probe of step 2".

---

## G. One thing to drop: the `relu(x) * w → relu(x * w)` fold

**Recommend the designer drop it, and recommend `report.md` §8 remove it from the step-2 probe
list.** The reasoning, on disk rather than in a message:

1. **It is not probeable.** Per constraint 6, the sign of `w` is activation-dependent —
   `weights_proj(hidden_states)` is a plain linear layer with no non-negativity anywhere in its
   construction, scaled only by positive constants. No probe can *prove* `w >= 0`; a probe can only
   sample activations and report that it did not observe a negative one, which is a bound, not a
   proof. Scheduling it as a probe misrepresents what the probe could return.
2. **That makes it an approximation, not an exact restructuring.** A rewrite whose exactness rests
   on a data property that cannot be established is, by `06-algorithm.md`'s own classification, a
   change to the numbers coming out. It therefore carries the **full three-tier gate of §B**,
   including the GLM-5.2 eval baseline that does not exist. The failure mode is silent and severe:
   one negative weight zeroes that head's entire contribution for that token, changing the selected
   token set with no error signal anywhere.
3. **The prize does not survive contact with either measurement.** `report.md` §7 measures ~5%
   (215.7 vs 227.1 µs) in a probe the report itself disowns — it materialized the `[T,H,S]`
   intermediate to HBM, which is the one thing `compute_scores` is built to avoid, so all three arms
   were bound by memory traffic and not by the op count the fold removes. Area 4 independently
   prices the fold at **~2% of decode traffic**. A ≤5%-in-the-wrong-regime, ~2%-in-the-right-one
   prize does not justify a model-level eval gate plus a new checked-in baseline.
4. **There is no cheaper exact form.** The sign-safe rewrite `relu(x·|w|)·sign(w)` is two
   elementwise ops where the incumbent has one — strictly worse. Folding `|w|` into `q` before the
   matmul (relu is positively homogeneous, so this *would* move the scale from `[T,H,S]` to
   `[T,H,D]`, 80× fewer multiplies) requires re-quantizing `q` per (token, head) in fp8, which is
   itself an approximation — and the existing code already made the opposite call deliberately, by
   folding `q_scales` into the weights rather than into `q` (`deepseek_v4_indexer.py:81`).

**Verdict: drop. Do not spend a probe on the weight sign.** If a future design wants it, the correct
framing is "an approximation with a data-dependent precondition", entering at §B's gate, not at the
probe list.
