# Step 2, wave 2 — combined

What is only visible from holding all nine constraint sheets at once. Nine briefers each read one
reference file and chose nothing; this document reconciles them, adds up what they jointly spend,
and works `references/seams.md` in order.

**This document chooses nothing either.** Wave 4 picks the design. A candidate here that I find
unpromising still goes on the list, with the reason attached.

---

## Conventions

**Sign convention, binding.** Every percentage is a change in **time**, `(new − old)/old`. Negative
is faster. Signed percentages appear only in tables, under a `Δ time %` column, with the convention
restated at the table. Prose says the direction in words ("9.6% faster"), never `−9.6%`. A
`speedup ×` column, where present, is `old/new` and >1 is faster — a different number from `Δ time %`.

**Tags.** Every number carries exactly one:

| tag | meaning |
| --- | --- |
| `measured (job id)` | observed on hardware in a named job |
| `modelled` | computed from a term the roofline has, using stated inputs |
| `assumed` | the roofline has a term for it and I guessed the value |
| `unmodelled` | the roofline has **no term** for the quantity. Never written beside a number. |

**Terms that collide.** Pinned here because three of them mean different things in different sheets:

| term | here | not |
| --- | --- | --- |
| **the block** | the code under study: `compute_scores` and the loop nest that feeds it, `streamindex_topk.py:180-460` | a Pallas grid block, and not an MXU row block |
| **bkv block** | `bkv_p` pages of compressed KV handled in one inner-loop trip. `bkv_sz = bkv_p × 1024` | a page. At `bkv_p = 2` one bkv block is two pages |
| **inner step / inner iteration** | one `(bq, bkv)` trip of the hand-rolled `lax.fori_loop` at `streamindex_topk.py:418-420`. There are `num_bq × num_bkv` of them per grid step | a `pallas_call` grid step, of which there are `(end_seq_idx − start_seq_idx) // seq_batch_size` |
| **M (row count)** | the MXU's non-contracted left row extent, `seq_batch_size × bq_sz × num_q_heads` | the number of tokens |
| **declared scratch** | VMEM named in `scratch_shapes`, `streamindex_topk.py:654-692` | all VMEM the kernel occupies. See §(b). |

**Floor formula, used throughout.** `floor = max(t_hbm, t_mxu)` where `t_hbm` is the larger of
`hbm_id` (each distinct byte crossed once) and — where the loop structure is taken as given —
`hbm_st` (the bytes *this* loop structure moves, including KV re-reads). It is **not** the largest
of three independent columns, and it is not a sum. `achieved = floor / measured`; a higher
percentage means *less* headroom.

**A number without its provenance is not a number.** Every figure below that I did not compute in
this document from a stated formula carries its job id or its source file and line.

---

## (a) Reconcile

### a.0 — The premise correction: 21 layers, not 78

**Area 7 §0 is right, and it is sourced.** GLM-5.2 runs the indexer on **21 of 78 layers**.
`_skip_topk = max(layer_id − offset + 1, 0) % freq != 0` with `index_topk_freq = 4` and
`index_skip_topk_offset = 3` yields the set {0, 1, 2, 6, 10, …, 74} — 21 layers. The other 57 take
the `else` branch at `mla_attention_op.py:606-607` and reuse `self.topk_indices_buffer`.

I checked the consequence the sheets could not, because it needs a second file: on a skipped layer
`self.indexer(...)` is **never called** (`mla_attention_op.py:602`), and `self.indexer` is only
constructed when `not skip_topk` (`:508`). So the *entire* `_streamidx_topk_jax` body — the scatter
at `:103`, the reshape, the `-inf` fill at `:196`, and the Pallas launches — runs 21 times per model
step, not 78. Every per-step aggregate in every sheet scales by **21/78 = 0.269**.

**Two independent corroborations**, neither of which any single briefer was positioned to make:

1. The checkpoint's `indexer_types` list is 21 `"full"` / 57 `"shared"` (area 7's own source read).
2. **The profile's own call counts divide by 21 and do not divide by 78.** `report.md`'s dominant
   population is **630 calls** and the tail is **105**; 630 = 21 × 30 and 105 = 21 × 5, while
   630/78 = 8.08 and 105/78 = 1.35. The profile is a third, independent witness
   [`measured`, j-6d675a58-7ecd-4b44-9171].

**What rescales.** Exactly one live figure in the nine sheets:

| sheet | as written | corrected | notes |
| --- | --- | --- | --- |
| area 2 §4 | "3 launches × 78 = 234 Pallas launches per model step, of which 78 do work" | **63 launches, of which 21 do work** | the launch-overhead argument survives at 27% of its stated size |
| area 2 §0 row (h) | layer axis extent 78 | **21** | |
| area 1, X3/X4 | "~405 MB per layer per step" | unchanged per layer; **× 21 = 8.5 GB/step**, not 31.6 GB/step | the item stays the largest crossing in the ledger either way |

**What survives untouched — which is nearly everything.** Every per-call quantity: all five floors,
all achieved fractions, every `hbm_id` / `hbm_st` / `mxu` figure in `report.md`'s roofline, every
per-inner-step DMA count, and the whole of §(b) and §(c). Those are properties of one call. The
correction moves only end-to-end Amdahl shares and per-model-step aggregates, and it moves them all
by the same 0.269.

**Nothing in `report.md` used 78.** `report.md`'s 2.69% decode-step share is read off a profile,
which already reflects reality; it does not need rescaling and must not be rescaled.

### a.1 — The rates: spec vs measured, and what flips

`report.md`'s roofline uses **spec** figures from `V7X.md`. Several sheets carry **measured** rates
from a probe job. Both are legitimate; they answer different questions, and mixing them silently is
how a floor becomes unfalsifiable.

| quantity | `report.md` (spec, `V7X.md`, per TensorCore) | measured | source |
| --- | ---: | ---: | --- |
| HBM, whole-kernel ceiling | 3364.9 GB/s | **2.97 TB/s** | `measured`, wave-1 sheets |
| HBM, marginal rate | — | 3.48 TB/s | `measured` |
| MXU bf16, under XLA | 892.8 TFLOP/s | 764.8 Tops/s | `measured`, j-ed517191-5f2d-4d00-8c0c |
| MXU bf16, **under Pallas** | — | **569.9 Tops/s** | `measured`, j-ed517191-5f2d-4d00-8c0c |
| MXU fp8, under XLA | 1785.6 TFLOP/s | 1419.9 Tops/s | `measured`, j-ed517191-5f2d-4d00-8c0c |
| MXU fp8, under Pallas | — | *none* | — |
| fp8 : bf16 ratio | 2.00× | **1.86×** | derived, 1419.9/764.8 |
| M=32 row derate | 4 (= 128/32, assumed) | **3.57×** per unit of work | `measured`, j-ed517191, `mxu_block_rows_cost` |

**Which rate this document uses, and why.** For anything ranking a **Pallas** kernel: the measured
Pallas rates — 569.9 Tops/s bf16, 2.97 TB/s HBM, 3.57× derate. The 892.8 figure is a device spec
that no Pallas kernel has been shown to reach, and the 764.8 XLA figure is a different code
generator. Where I quote `report.md`'s table I keep its spec rates and say so. **fp8 under Pallas
has no measurement at all**; below I use the XLA fp8 rate and mark it, because there is nothing
better, and I note that if fp8 suffers the same Pallas/XLA gap bf16 does (569.9/764.8 = 0.745), the
Pallas fp8 rate would be ≈1058 Tops/s — `assumed`, and it is what P1 on the probe list is for.

**Recomputation.** `bytes = t × rate`, then `t' = bytes / rate'`. FLOPs are recovered as
`FLOP = t_mxu × rate × (1/derate)` and re-divided.

| regime | denominator (µs) | `hbm_id` spec → measured | `mxu` bf16 spec → Pallas | `mxu` fp8 spec → XLA | floor bf16 spec → meas. | achieved bf16 spec → meas. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 76.42 (profile scope) | 41.7 → **47.2** | 19.2 → 26.9 | 9.6 → 10.8 | 41.7 → **47.2** | 55% → **62%** |
| `decode_64k` | 115.0 (by diff) | 30.9 → 35.0 | 14.4 → 20.1 | 7.2 → 8.1 | 30.9 → 35.0 | 27% → 30% |
| `mixed_prod` | 269.5 (by diff, clean) | 29.3 → 33.2 | 77.0 → **120.6** | 38.5 → 48.4 | 77.0 → **120.6** | 29% → **45%** |
| `mixed_32k` ⚠ | 795.0 (by diff, ⚠) | 91.0 → 103.1 | 307.9 → 482.4 | 153.9 → 193.6 | 307.9 → 482.4 | 39% → 61% |
| `mixed_64k` ⚠ | 1753.2 (by diff, ⚠) | 173.3 → 196.3 | 615.8 → 964.7 | 307.9 → 387.1 | 615.8 → 964.7 | 35% → 55% |

⚠ marks the two rows whose denominator is a difference against a control arm with 514.8% / 133.1%
spread (`report.md`). They are shown for completeness and carry no weight.

**What flips, and what does not.**

- **Nothing about the gate.** `decode_prod` moves 55% → 62%, still inside the 50–80% band. Every
  substitution I can make — spec or measured, `hbm_id` or the Pallas-scope-restricted 40.7 µs, with
  or without the grid-step fixed term — lands between 53% and 62%. **The gate does not turn on the
  rate question.** It turned on the step-count question, which is settled in a.2.
- **The ordering of the regimes survives.** Mixed is still further from its floor than decode
  (45% vs 62% at bf16). `report.md`'s "mixed is where the headroom is" stands, but the margin
  shrinks a lot: at spec rates it was 29% vs 55%, a 1.9× gap; at measured rates it is 45% vs 62%,
  a 1.4× gap. Any option ranked on "mixed has twice the headroom" needs re-reading.
- **What the measured rates *do* flip is the importance of the precision question.** At spec rates
  mixed's floor spanned 38.5–77.0 µs, a 2.0× band. At measured rates it spans 48.4–120.6 µs, a
  **2.5×** band, and the two ends sit on opposite sides of a design decision: at fp8 mixed is at 18%
  of floor (5.6× headroom, chase it), at bf16 it is at 45% (2.2× headroom, comparable to decode).
  `report.md` says which precision the einsum gets "is not settleable from the source". It is
  settleable from the *compiler output*, and that is probe **P1**.
- **One derate correction worth one line.** `report.md` used derate 4 at M=32 as an assumption
  (`128/32`); the measured value is 3.57× [j-ed517191]. This makes decode's MXU term 11% *smaller*
  than the naive scaling, and since decode is HBM-bound it changes nothing. Recorded so nobody
  "corrects" it back.

### a.2 — The per-step fixed term: **resolved, measured, gate holds**

This is the disagreement between area 2 (disagreement 3) and area 3, over whether
`03-pipelining.md`'s 512 ns per-step fixed term attaches to a `pallas_call` grid step (16 of them)
or to this kernel's inner `lax.fori_loop` iteration (64 of them). It was in flight when this wave
was briefed. **It has landed**: `probes/step_cost_RESULT.md`, job **`j-e8eaacd3-6be4-4cdf-95a2`**,
`Succeeded`, 2/2 shards rc=0.

**Both sheets were right about part of it, and the disagreement was real.**

- The extrapolation **transfers**: a per-inner-iteration fixed cost of **≈500 ns** is real, fitted
  from successive differences at two batch sizes that agree rung for rung (500/503, 448/433 ns)
  [`measured`, j-e8eaacd3-6be4-4cdf-95a2]. Area 3's reading of the mechanism was correct; area 2's
  caution that the extrapolation was "not licensed in either direction" is resolved in favour of it
  transferring.
- But it is **not a floor term**, because the step count is a design parameter. A floor is taken
  over valid implementations, so it uses the smallest feasible step count, not the incumbent's.
  The measurement refutes the 93% reading directly: changing one static constant made the decode
  call **9.64% faster** (0.16624 → 0.15021 ms at `bkv_p = 8`, shard 0) [`measured`, j-e8eaacd3].
  You cannot go 9.6% below a floor by editing a tuple.

**Consequence for the gate: it stands open at 55–61%** (spec rates) or 62% (measured HBM). The
exercise continues. The reading that would have ended it rests on a floor definition the
measurement falsifies.

**And the cost decays.** 500 → 448 → 106 ns per step as the step count falls 128 → 64 → 32 → 16.
That decay is the kernel reaching something the step count no longer controls, which is what a floor
looks like from above.

**What this hands the rest of the document**, and it is the largest measured lever in the exercise:
`bkv_p` is on the alternative list of three sheets (area 2 B1, area 3 items 8 and 9) and now has a
number. Four things stand between it and a shipping candidate, per the probe's own write-up: the
shape was `max_model_len = 8192`, not production 9216; correctness at `bkv_p ≠ 2` was never checked;
only the decode band was varied; and VMEM at mixed was flagged as possibly insufficient. **§(b)
settles the fourth by arithmetic** and it settles it against the lever. The first is being measured
now by `probes/bkv_p_prod.py` — **in flight, do not queue**. The second and third are probe **P5**.

### a.3 — bf16: area 5's refusal and area 4's narrowing are compatible

Adjudicated as **apparent, not real** — two sheets using one word for two things.

- **Area 5** (corroborated independently by area 6, from a different job) is about the **MXU
  accumulator**: `preferred_element_type=jnp.bfloat16` raises a Mosaic `VerificationError`
  [`measured`, j-4fba8dc6-d72c-41f9-948b and j-050c7480-40a6-4a50-9e1b]. The accumulator must be
  f32. This is a hard refusal, not a cost.
- **Area 4**'s bf16 proposals touch three other places: (P5) the **storage dtype of the scores
  buffer** after the accumulation and reduction are complete; (P9) the **VPU dtype** of the
  relu / weight-multiply / head-sum chain; (P8) the **einsum operand** dtype, explicitly with an f32
  accumulator retained.

None of the three narrows `preferred_element_type`. Area 6's row A5 states the same split
explicitly ("bf16 relu/weight-multiply/head sum, f32 matmul accumulator"). **They are compatible and
both can be carried.** The constraint binds only if a future candidate touches the accumulator, and
none on the list does.

**But holding three sheets at once surfaces a constraint on P5 that none of them states alone:**
`sparsecore_topk` raises on non-f32 input (`sparsecore_topk.py:395-396`, area 6). So a bf16 scores
buffer is feasible only with a matching change on the SparseCore side — and area 8 supplies the
missing fact that makes that change plausible rather than speculative: bf16 on the SparseCore is
carried as packed `uint32` and widened, **measured exact**. P5 is therefore a two-file change, not a
one-line dtype swap, and it should be costed as one.

### a.4 — The `mixed_prod` sequence-count ambiguity: **settled, in `report.md`'s favour**

Areas 1, 2 and 3 each independently flagged that `mixed_prod` (nseq = 64, ntok = 1024) admits two
readings: 64 sequences of 16 tokens each (grid = 64, `num_bq` = 1), or one sequence of 1024 tokens
(grid = 1, `num_bq` = 8). Area 3 argued for the second by an infeasibility argument. **It is settled
directly from the instrument**, which no sheet read:

`instruments/inputs.py:138-147` sets `q_lens[0] = num_tokens`, `active_seqs = 1`, and
`distribution = [0, 0, 1]` for the mixed phase. **One prefill sequence carrying the whole chunk.**
The file's own comment says why, and reports that the first build did it the other way and "inflated
mixed KV traffic 64-fold". `report.md`'s `nbq = 8`, its `M = 4096` and area 3's 32 inner steps are
all correct; area 2's grid = 64 reading is not.

**A caveat that only matters from this position.** The *instrument's* mixed shape is a deliberately
chosen corner — one long sequence, maximising `num_bq` and therefore the KV re-read. Production
mixed batches may hold several sequences with smaller `q_len` each, giving a smaller `bq_sz`
(`streamindex_topk.py:633`: `bq_sz = min(num_queries_per_block, static_q_len)`), a smaller M, and —
per §(b) — a much smaller VMEM peak. **Every mixed number in this exercise is measured at
`bq_sz = 128`, and §(b)'s binding constraint is proportional to `bq_sz`.** Any candidate tuned to
that corner should say so. This is not a defect of the instrument; the corner is the conservative
one for time. It is a defect only if a VMEM conclusion drawn there is quoted as production.

### a.5 — Corroborations

Nine independent readings were paid for; agreement is part of the yield. One line each.

| finding | sheets that reached it independently |
| --- | --- |
| The `-inf` fill has no consumer that requires it — the mask at `compute_scores` already writes `-inf` over every invalid lane | **6**: areas 1, 2, 4, 5, 6, 7 |
| `num_sublanes_total` is derived from `max_model_len`, not from the batch's actual `kv_len`, and the gap is pure waste | **6**: areas 1, 2, 4, 5, 7, 8 |
| Declared scratch is 4.10 MiB at decode / 4.03 MiB at mixed | **4**: areas 1, 2, 3, 5 |
| 704 DMAs per call, and the per-DMA reading of the 512 ns term is refuted by the count | **2**: areas 2, 5 — and now by measurement, j-e8eaacd3's constant-page-DMA design |
| Fold `w` into `q` to delete a broadcast and a multiply; precondition `w ≥ 0` is unproven | **3**: areas 4, 5, 6 — area 6 adds DeepSeek's own source as prior art |
| `sparsecore_topk`'s HBM DMA is O(padded width) while its compute loops are O(`row_lengths`) | **2**: areas 7, 8, from independent source reads |

The two six-way agreements are the two strongest signals in the whole wave, and both point at the
same thing from different directions: **the kernel moves and writes a scores buffer sized for a
context length the batch does not have, and then fills the unused part with a constant nobody
reads.** That is one defect with two symptoms, and it is why C1/C2 lead the candidate list.

---

## (b) Reconcile the budget

Areas 2, 3, 4 and 7 are each individually satisfiable. They spend **one VMEM** and **one SMEM**.
This section adds them up. Every figure is recomputable from the formulas given.

### b.0 — The ceilings

| resource | ceiling | source |
| --- | ---: | --- |
| VMEM, physical per core | **64 MiB** | `hardware-facts.md` |
| VMEM, default *scoped* limit | ~32 MiB (largest single scratch observed 31 MiB) | `hardware-facts.md` |
| VMEM, this kernel's configured limit | **100 MiB** | `streamindex_topk.py:28`, `DEFAULT_VMEM_LIMIT_BYTES` |
| SMEM per core | 1,048,576 B | `hardware-facts.md` |
| CMEM | 0 on tpu7x | `hardware-facts.md` |

Note the 100 MiB limit exceeds physical. `vmem_limit_bytes` gates a compile-time check; it does not
create memory. Exceeding 64 MiB physical fails by a different route or spills.

### b.1 — Declared scratch, from source

Recomputed from `streamindex_topk.py:654-692`, not from any sheet. Constants:
`page_size = 1024`, `kv_packing = 4`, `lkv_dim = 256`, `num_q_heads = 32`, `head_dim = 128`,
`bkv_sz = bkv_p × 1024`, `num_sublanes_bkv = bkv_sz / 128` (verified at `:107-108`, `:638-639`).
Write `B` for `seq_batch_size`.

| buffer | shape | formula (bytes) | ≡ |
| --- | --- | --- | --- |
| `bkv_double_buf` | `(2, B, bkv_p·256, 4, 256)` u8 | `2 · B · bkv_p·256 · 4 · 256` | `B · bkv_p · 0.5 MiB` |
| `bq_double_bufq` | `(2, B, bq_sz, 32, 128)` fp8 | `2 · B · bq_sz · 32 · 128` | `B · bq_sz · 8 KiB` |
| `bq_weights_double_buf` | `(2, B, bq_sz, 32)` f32 | `2 · B · bq_sz · 32 · 4` | `B · bq_sz · 256 B` |
| `bo_scores_double_buf` | `(2, B·bq_sz, bkv_sz/128, 128)` f32 | `2 · B·bq_sz · bkv_p·8 · 128 · 4` | `B · bq_sz · bkv_p · 8 KiB` |
| DMA semaphores | `(4, 2, B)` | sync memory, not VMEM | negligible |

### b.2 — The buffer nobody declared

`compute_scores` (`streamindex_topk.py:304-354`) computes, with the comment `# Compute in Registers`
directly above it:

```python
s = jnp.einsum("bnd,bmd->bnm", bq_stacked, bkv_stacked,
               preferred_element_type=jnp.float32)   # (B, bq_sz·32, bkv_sz) f32
```

It does not fit in registers. Its size is

> **`s_bytes = B · bq_sz · 32 · bkv_p · 1024 · 4` = `B · bq_sz · bkv_p · 0.125 MiB`**  ⟵ `modelled`

That is **1.0 MiB at decode** and **32.0 MiB at mixed** — 50% of physical VMEM, undeclared. Area 5's
figure is right, and the formula above reproduces it: `1 · 128 · 2 · 0.125 = 32`.

Note it is **independent of `kv_len`**: `s` is per-bkv-block, so `mixed_32k` and `mixed_64k` carry
the same 32 MiB as `mixed_prod`.

### b.3 — Area 3 and area 5 are both right, about different things

Area 3 reports ~27 MiB unused; area 5 reports 32 MiB live and undeclared. The brief is correct that
these cannot both be the whole story. Neither is wrong:

- Area 3's 27 MiB = 31 MiB scoped limit − 4.03 MiB **declared** scratch. Arithmetically correct
  about declared scratch, and it excludes the temporary.
- Area 5's 32 MiB is the temporary, and it excludes nothing.

**Resolved: real free VMEM at `mixed_prod` is ~26.5 MiB of the 64 MiB physical**, not 27 MiB on top
of the 32, and not ~60 MiB. The near-coincidence of 27 and 26.5 is why this is worth stating: an
option that reads area 3's headroom as available *in addition to* area 5's temporary has
double-counted by 32 MiB.

**Three things follow.**

1. **`DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` is load-bearing, not gratuitous.** At mixed the kernel is
   resident at ≥36 MiB, above the ~31–32 MiB default scoped limit; under the default it would hit
   `CompileTimeScopedVmemOom`. This **contradicts** area 1's reading ("advertises 3× more than the
   reference's measured ceiling") and area 2's ("already set past the wall; raising it buys
   nothing"). Raising it bought the kernel the ability to compile at mixed at all.
2. **Mosaic must be fusing the elementwise chain in place.** If `jnp.maximum(s, 0)` and `s * w` each
   materialised a fresh 32 MiB buffer, peak would be `4.03 + 3 × 32 = 100.03 MiB`, above 64 MiB
   physical, and the kernel would not run. It runs, at 269.5 µs and no sign of catastrophic
   spilling. So **exactly one live copy of `s`** is the only reading consistent with the kernel
   working. `assumed`, and it is half of probe **P2**.
3. The other half of P2: whether Mosaic materialises `s` at full `bkv_sz` width at all, or tiles the
   contraction internally. If it tiles, the constraint in b.5 dissolves and most of this section is
   moot. Tagged `assumed` because there *is* a term — VMEM bytes — and I computed it from the
   source-level shape; the compiler's actual allocation is what P2 reads.

### b.4 — The joint table

`Δ over ceiling` is `peak − 64 MiB`; negative means it fits. All figures MiB.

| regime | B | bq_sz | bkv_p | kv buf | q buf | w buf | scores buf | **declared** | `s` | `s_summed`+masks | **peak** | 64 MiB ceiling | Δ over ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 4 | 1 | 2 | 4.000 | 0.031 | 0.001 | 0.062 | **4.095** | 1.000 | 0.04 | **5.14** | 64 | −58.9 |
| `decode_64k` | 4 | 1 | 2 | 4.000 | 0.031 | 0.001 | 0.062 | **4.095** | 1.000 | 0.04 | **5.14** | 64 | −58.9 |
| `mixed_prod` | 1 | 128 | 2 | 1.000 | 1.000 | 0.031 | 2.000 | **4.031** | 32.000 | 1.50 | **37.53** | 64 | −26.5 |
| `mixed_32k` | 1 | 128 | 2 | 1.000 | 1.000 | 0.031 | 2.000 | **4.031** | 32.000 | 1.50 | **37.53** | 64 | −26.5 |
| `mixed_64k` | 1 | 128 | 2 | 1.000 | 1.000 | 0.031 | 2.000 | **4.031** | 32.000 | 1.50 | **37.53** | 64 | −26.5 |

`s_summed` = `B · bq_sz · bkv_sz · 4` B; masks are two boolean `(B, bq_sz, bkv_sz)` arrays plus a
2048-element iota. Together ~1.5 MiB at mixed, ~40 KiB at decode. Small, included for completeness.

Declared totals match areas 1, 2, 3 and 5 exactly (4.095 / 4.031), which is the four-way
corroboration recorded in a.5.

### b.5 — The constraint that replaces four separate asks

Every mixed-regime peak above is dominated by one term. Setting `peak ≤ 64 MiB` and keeping only it:

> **`B · bq_sz · bkv_p ≤ ~500`**

Today: `decode_prod` = `4 · 1 · 2` = **8** (62× slack); `mixed_prod` = `1 · 128 · 2` = **256** (2×
slack). This single inequality prices four areas' asks at once. `modelled`, contingent on P2.

**Now the asks, against it.**

| area | ask | cost | verdict |
| --- | --- | --- | --- |
| 2, B1 | `bkv_p` 2→4 at **decode** | product 8 → 16; peak 5.14 → 10.3 MiB | **fits.** And it ran: `measured`, j-e8eaacd3 |
| 2, B1 | `bkv_p` 2→8 at **decode** | product 8 → 32; peak 5.14 → 20.3 MiB | **fits.** Ran; 16 MiB KV scratch accepted by Mosaic [j-e8eaacd3] |
| 2, B1 | `bkv_p` 2→4 at **mixed** | product 256 → 512; `s` = 64 MiB, peak **70.1 MiB** | **infeasible** |
| 2, B1 | `bkv_p` 2→8 at **mixed** | product 256 → 1024; `s` = 128 MiB, peak **134 MiB** | **infeasible** |
| 2, B2 | `bq_sz` 128→1024 at mixed | product 256 → 2048; `s` = **256 MiB**, peak ~271 MiB | **infeasible.** Area 2 costed this arm at 25.25 MiB of *declared* scratch and did not carry the temporary, which scales with `bq_sz` too. The largest B2 arms are dead as scoped. |
| 2, B2 | `bq_sz` 128→256 at mixed | product 512; peak ~70 MiB | **infeasible** by the same margin as `bkv_p`=4 |
| 3 | deepen `bo_scores` 2 → 3 or 4 buffers | +1 / +2 MiB at mixed; +32 / +64 KiB at decode | **fits everywhere.** Note the `buffer_count ≤ 2` cap in `hardware-facts.md` applies to `emit_pipeline`'s `BufferedRef`; this kernel hand-rolls its scratch and is not bound by it |
| 3 | deepen `bkv` 2 → 3 | +0.5 MiB mixed, +2 MiB decode | fits |
| 3 | pipeline operands via `BlockSpec` instead of `pltpu.HBM` | adds each block's bytes again in Mosaic-managed buffers, roughly doubling the declared 4 MiB | fits at both, but see the note below |
| 4, P8 | bf16 einsum **operands** | q buf ×2, kv buf ×2: +2 MiB mixed, +4 MiB decode. `s` unchanged (f32 accumulator) | fits; but costs the fp8 MXU rate, 1.86× `measured` |
| 4, P5/P7 | 16-bit scores buffer | `bo_scores` halves: −1 MiB mixed | **frees budget.** Requires the SparseCore-side change from a.3 |
| 4, L3 | 3-D contraction landing scores as (…,16,128) | no VMEM change | fits |
| 7 | hoist metadata to SMEM | see b.6 | fits with 3 orders of magnitude to spare |

**The finding this section exists to produce.** `step_cost_RESULT.md` left open, as its item 4,
whether the measured `bkv_p` lever survives at mixed. **It does not.** At mixed the `bkv_p` ladder is
capped at 2 by `s`, and the ≈500 ns/step lever that made decode 9.64% faster is **unavailable at
mixed** for exactly that reason.

**And the obvious workaround does not work either**, which is worth writing down before someone
spends a job on it. One might hold the product constant by trading `bq_sz` down for `bkv_p` up. But
the inner-step count is
`inner = num_bq × num_bkv = cdiv(ntok, bq_sz) × cdiv(kv_len, bkv_p · 1024)`, and that product is
**invariant** under the trade: at mixed, `(bq_sz=128, bkv_p=2)` gives 8 × 4 = 32 and
`(bq_sz=32, bkv_p=8)` gives 32 × 1 = 32. Identical step count, and the KV re-read factor rises from
8× to 32×. The trade is a wash on the lever and strictly worse on bytes. `modelled`.

**Which is what makes C4 the structural candidate.** Tiling the einsum over `bkv` *inside*
`compute_scores` — so `s` is bounded by a sub-block instead of the full `bkv_sz` — costs zero FLOPs
and zero HBM bytes and dissolves `B · bq_sz · bkv_p ≤ 500` entirely. It buys nothing by itself. It
buys the ability to spend `bkv_p` at mixed, which is a lever measured at 9.64% at decode. Its
ranking turns entirely on P2: if Mosaic already tiles internally, C4 is a no-op and `bkv_p` at mixed
was never blocked.

**One caution on area 3's `BlockSpec` ask.** Area 2 reports every `in_spec` is `pltpu.HBM`, so the
kernel pipelines no operand and hand-rolls all DMAs. Converting to Mosaic-managed pipelining fits the
budget, but it hands Mosaic the step structure — and a.2 established that the step count is the
single measured lever here. That is a trade of a measured lever for compiler convenience and should
be priced as such, not as a refactor.

### b.6 — SMEM: not a constraint, anywhere

The kernel's scalar operands are `seq_lens`, `page_indices`, `cu_q_lens`, `distribution`.
`page_indices` is `num_seqs × pages_per_seq` i32 and is the largest:

| regime | `page_indices` | + others | total | of 1,048,576 B |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` / `mixed_prod` | 64 × 9 × 4 = 2,304 B | ~528 B | 2,832 B | **0.27%** |
| `mixed_32k` | 12 × 33 × 4 = 1,584 B | ~112 B | 1,696 B | 0.16% |
| `mixed_64k` / `decode_64k` | 6 × 65 × 4 = 1,560 B | ~76 B | 1,636 B | 0.16% |

Even a 64-sequence batch at 65 pages/seq is 16.6 KB, 1.6% of SMEM. **Area 7's entire
`PrefetchScalarGridSpec` ladder is free on this axis** and no SMEM arbitration between areas is
needed. Stated explicitly so the SMEM half of this section is closed rather than merely unmentioned.

---

## (c) Combine the seams

`references/seams.md`, in order. Every seam gets a verdict line, including the ones that do not
apply. Each instance is a **candidate** and joins wave 4's option table on equal terms.

Per the seams file's ranking procedure: these candidates spend something the floor has a term for
(FLOPs, bytes) to buy something it does not (layout, chain width, MXU occupancy). They are **not**
ranked on FLOPs and bytes — that model has decided against them by construction. The deciding
quantity is tagged `unmodelled` and the strike test is applied: strike the term from both options
and re-read the ranking; if the ranking survives, the tag was bookkeeping; if it changes or ties,
probe.

### S1 — (4) layout × (6) algebra — **instance, three of them**

*Inversion, in my words:* don't ask what the transpose costs; ask what algebraic form of this
computation would have produced the value on the axis its consumer already wanted.

Area 4's value ledger has three rows with a layout mismatch, all tagged `unmodelled` there. Holding
them against area 6's algebra sheet turns two of the three into algebra questions the layout briefer
was not positioned to ask.

**S1-a — `scale_val` `.T` on (2048, 1), sublane → lane.** The e8m0 position scale is sliced out of
`flat_bkv[:, 128:129]`, landing on sublanes; its consumer is `s_summed`, which has `bkv` on lanes.
*Algebraic form that makes it zero:* fold the scale into `bkv` before the contraction. It is exact —
the scale is a positive power of two and `relu(q · (kv·σ)) = σ · relu(q · kv)` for `σ > 0`, so it
commutes past the relu. **But it widens `kv` off fp8**, costing the 1.86× fp8:bf16 MXU rate
[`measured`, j-ed517191] to save one 2048-element transpose. *The modelled term dominates the
unmodelled one by two orders of magnitude; this loses.* Recorded so it is not re-proposed.
*The other algebraic form is S1-c*, which deletes this move as a side effect.

**S1-b — `bq_weights_stacked[..., None]`, lane → sublane broadcast.** Two resolutions from different
sheets, and they are not the same move:
- Area 6 E1 / area 5 B: fold `w` into `q` before the einsum. Deletes the broadcast *and* the
  multiply. **Precondition `w ≥ 0` is unproven** — the fold is exact only if the weight is
  non-negative, since it must commute past the relu. `assumed`; probe **P6**.
- Area 4 L13: store `indexer_weights` transposed as `(32, ntok)`. Deletes the broadcast, keeps the
  multiply, **needs no precondition**. Strictly the safer arm.

Seams form 5 is the reason this is a seam at all: "broadcast and repeat last, and only on a leading
axis" — measured, an interleaved sublane repeat is 3.9 ps/elem, 5× a transpose, while a
leading-axis repeat is inside noise. This broadcast is on a trailing axis.

**S1-c — the big one: `s_summed (B, bq_sz, 2048)` → `(B·bq_sz, 16, 128)`.** 262,144 elements per
inner iteration at mixed; area 4's ledger row 12, its largest. Area 4 files it under Layout with no
algebra attached; area 6 does not have it at all. **It is seams form 4** — reassociate so the axis
you need is already the contracting axis. Area 4's own L3 is the mechanism: reshape `bkv` in VMEM to
`(16, 128, head_dim)` and contract in 3-D, so the einsum's output lands already tiled `(…, 16, 128)`
and the reshape never happens. Same FLOPs, same bytes, different parenthesisation. This is the
cleanest S1 instance in the block and it is only visible from both sheets at once.

*Strike test on S1-c:* strike the relayout term from both options and they cost identically — same
FLOPs, same HBM. The ranking is **entirely** decided by the struck term. **Probe (P3).**

*Note on the reference's measured relayout table:* `hardware-facts.md`'s 0.8 ps/element is for
**f32 values that never reach the MXU**, at 2048×256. S1-c's value is f32 at 128×2048 — same dtype,
different shape and a different consumer. S1-a and S1-b are 8-bit and f32-small. The table gives the
*order* of the prize, not these numbers. Do not quote 0.8 ps/elem as this kernel's cost.

**S1-d — a form the seams file warns about, and it applies.** Form 1 ("get the layout from the load")
is the natural fix for the 129-of-256-byte record: read the value bytes and the scale byte with a
strided descriptor. `hardware-facts.md` records **three distinct Mosaic/Pallas refusals** of an
in-kernel strided read of a value or a ref, at every shape tried. What remains is a strided
`BlockSpec`/`index_map`, a different mechanism — and this kernel uses no `BlockSpec` at all
(every `in_spec` is `pltpu.HBM`). **So the layout fix for the record is not available in-kernel and
must be a format change on the writer side, which is S4-b.** Naming this closes what would otherwise
be a wasted design cycle.

### S2 — (5) units × (6) algebra — **instance, and it is decidable without a probe**

*Inversion, in my words:* find two operations sharing an operand or a contraction, and price them as
one wider operation. Check the row axis specifically.

**The instance: decode's M = 32.** At `decode_prod`, `B = 4` sequences each contract at
`M = bq_sz · num_q_heads = 1 · 32 = 32` rows — the einsum's `b` axis batches them, so the MXU sees
four 32-row passes. 32 rows costs **3.57× per unit of work** [`measured`, j-ed517191,
`mxu_block_rows_cost`]. Concatenating along rows to M = 128 hits the 1.00× point exactly. Areas 5 (K)
and 6 (E5) both name this and both guess "break-even or loss".

**The arithmetic settles it, and it is a loss.** The four sequences have **disjoint** KV — different
pages — so the concatenated form must contract 128 rows against all four sequences' KV to recover
all four results, which is 4× the useful work:

| form | rows | work | cost per unit | total cost |
| --- | ---: | ---: | ---: | ---: |
| today, 4 × separate | 32 | W | 3.57× | **3.57 W** |
| concatenated | 128 | 4W | 1.00× | **4.00 W** |

`Δ time % = (4.00 − 3.57)/3.57 = +12.0%` — 12% **slower**. And it does not matter: decode's floor is
`max(47.2, 26.9) = 47.2 µs`, HBM-bound, so the MXU term is not binding and a 12% move in it changes
the floor by **zero**. **Verdict: no, twice over — it loses on the MXU and the MXU is not the
constraint.** Decided from `measured` data; **no probe.**

**The inverse form of the seam also fires.** Area 6 E6 and area 5 H independently propose the 32-head
sum as a block-diagonal ones-matmul. Fill is 1/32 = 3.1%. "Filling the MXU with zeros is not filling
the MXU." **Rejected, corroborated by two sheets, no probe.**

**One S2-adjacent case that is *not* decidable and belongs to S1-c.** Area 6's E2 (transpose to
`k · qᵀ`) raises M from 32 to `bkv_sz = 2048` — comfortably past the 1.00× point — but drops N from
128 to 32 lanes. The measured N sweep covers 128 → 512 (1.01×) and records a non-monotonicity at
N = 256 (1.23×); **N = 32 is below the measured range.** `unmodelled`. Probe **P4**, and one timing
loop answers both E2 and any future narrow-N variant — "probe the primitive, not the variant".

### S3 — (3) pipelining × (6) algebra — **no instance in the canonical form; one in the general form**

*Inversion, in my words:* for each sequential chain, ask what rides down it, and whether a narrower
object would generate the same result — classically, solve for the operator rather than the answer.

**The canonical form has no instance, and the reason is structural.** The block's only sequential
construct is the `(bq, bkv)` `fori_loop` nest, and it is **not a recurrence**: every iteration's
scores are independent and are written to disjoint sublane windows of the output. There is no
operator to solve for, because there is nothing being solved. The only reduction is the 32-way head
sum, already a fixed-depth tree (area 5: ~6 levels, 3 full adds + 3 rotate-and-add). Nothing rides a
dependent chain. **"No instance, because the block contains no recurrence."**

The seams file's own note — "area (3)'s emission is the predicted per-step cost and which side
dominates; nothing in that emission asks whether a different algebra would make the chain shorter" —
is exactly why this verdict has to be written rather than assumed: area 3's sheet cannot say "no
recurrence", only "latency-bound".

**The general form does have an instance, and it is the biggest thing in this document.** Widen
"sequential chain" from the inner loop to the **producer-consumer chain from the TensorCore to the
SparseCore**, and the question becomes well-posed: the TC writes the full padded f32 scores matrix to
HBM and the SC reads it back. At `mixed_64k` that is 276.8 MB written, 276.8 MB read, and a further
276.8 MB for the `-inf` init — 830 MB — to deliver 8.39 MB of indices. **The width carried is
`ntok × padded_width × 4` B; the width of the result is `ntok × k × 4` B.**

Three narrowings, ordered by how much of the chain's semantics they preserve:

| # | narrowing | exact? | width factor | status |
| --- | --- | --- | ---: | --- |
| i | 16-bit monotone-encoded scores (area 4 P7) | rank-exact only if 16 bits separate ranks 2047/2048 | 2× | `unmodelled`; and `sparsecore_topk` **already** bitcasts f32 to a monotone i32 internally (`:320, 51-52`, area 6) — the encoding exists, it is just applied *after* the crossing |
| ii | size the width from the batch, not `max_model_len` (six sheets, a.5) | **exact** | up to 6.6× at the confound | the strongest candidate in the document; cost is jit-static recompiles |
| iii | cross only a candidate set — threshold-first, two-pass | exact only with an exact tie-break | ~130× | mechanism is the problem, see below |

**On (iii), because it is seductive and I do not want wave 4 to chase it blind.** The idea is
sound — compute a per-row threshold, then cross only survivors — and it is what `sparsecore_topk`
does internally anyway. Priced at `mixed_prod` it looks excellent: 2× the einsum is +120.6 µs
(`modelled`, Pallas bf16) or +48.4 µs (`modelled`, XLA fp8), against ~42 µs of HBM and up to ~400 µs
of SparseCore control time. **But the mechanism is unestablished.** Building the threshold needs a
histogram or a radix pass, and the TensorCore VPU has **no scatter-add**; a 256-bucket histogram by
compare-and-mask is 256 passes over the data, which is not affordable. `unmodelled`, and what is
unmodelled is not the cost but whether a mechanism exists at all. Area 8 also forecloses the easy
route: one Pallas kernel cannot hold both a TC mesh and an SC mesh, so this cannot be a fusion.
**Carried as a candidate with the mechanism named as the open question, not as a costed option.**

Narrowings (i) and (ii) are both real and both cheap. (ii) needs no algebra at all.

### S4 — (7) precomputation × (4) layout — **instance; the classic case is absent, a bigger one is present**

*Inversion, in my words:* which layout is being rebuilt on every call that could have been built once
and further out?

**S4's canonical case — a weight transpose hoisted to level 2 — has no instance here, and the reason
is clean: this block has no weight operand.** Everything it touches is an activation or a cache.
Area 7's ladder correctly marks level 2 N/A. "No instance, because there are no weights."
`07-precomputation.md § Gaps` notes level 2 has no measurement at all, so nothing is lost.

**But the inversion generalises from a transpose to a format, and then it fires twice.**

**S4-a — `indexer_weights` upcast to f32 per call** (`prepare_index_weights:463`, area 1 X1). Level 3
is correct — it is an activation. The hoist available is not to level 2 but *sideways*: have the
producer emit f32 directly and delete an XLA pass. Small, real, and it composes with S1-b's L13
(emit it transposed *and* in f32 in one change). Worth one line in a design.

**S4-b — the 256-byte cache record itself. This is the S4 instance that matters.** The record is
`align_to(align_to(128,128) + 1, 128) = 256` bytes, of which **129 are read** — 127 bytes of padding
crossing HBM on every read of every layer of every step. `report.md`: counting only used bytes takes
decode's ideal floor from 41.7 µs to **21.9 µs** at spec rates, or 47.2 → **24.8 µs** at the measured
2.97 TB/s [`modelled`]. That is the largest single structural prize in the exercise.

It is an S4 instance precisely in the seams file's sense: **the layout is chosen by the writer, the
cost is paid by the reader, and the code that would change it lives in the serving framework, so it
is nobody's to fix.** `mla_attention_op.py:103-104` decides the format; the kernel lives with it. The
format is invariant for the life of the process — level 1/2 — and is currently being chosen to suit
a scatter. S1-d established that the reader cannot work around it in-kernel (strided reads are
refused). **This routes as a handoff**, and the seams file is explicit that such a candidate is worth
naming even when not taken.

**S4-c — the scatter and reshape pair, `u8[424960, 256]`, ~291 µs each.** `report.md` sources both to
`mla_attention_op.py:104` [`measured`, prefill profile]. Area 1 reports the reshape exists only
because the scatter's output layout does not match the kernel's read — a layout conversion executed
per call that a different storage layout would make unnecessary once. Textbook S4.

Scaled by a.0's correction: **21 × 291 µs ≈ 6.1 ms per model step for the reshape alone**, and the
same again for the scatter. That is larger than everything else in this document combined, and — as
area 1 notes — it is **invisible to the microbenchmark's `full ms`**, which is why nine sheets could
read the kernel carefully and not have to reckon with it. It is not a kernel edit and it does not
belong in a kernel candidate's number. **It belongs on the `carried:` list, and dropping it would
waste the largest measurement in step 1.**

**S4-d — the `-inf` fill.** Area 7 #3 correctly identifies it as a compile-time-shaped constant,
invariant for the life of the process, rebuilt at level 3 twenty-one times per step. Area 7's own A4
correctly rejects hoisting it: the buffer is `input_output_aliases`'d and destroyed by the write. So
the S4-shaped resolution is **elimination, not hoisting** — six sheets agree the fill has no consumer
(a.5), because `compute_scores` already writes `-inf` over every invalid lane via `mask`. Recorded
here so the two readings ("hoist it" / "delete it") do not both end up in the option table.

### S5 — (2) partitioning × (1) fusion boundary — **one real instance, one refuted**

*Inversion, in my words:* which axis is present in the data but absent from the loop? Availability
test in einsum notation: an axis is batchable iff it appears in the output and is never contracted.

**Area 2's axis (e), the 32-head axis: refuted, and the sheet is right to refute it.** In
`score[t, s] = Σ_h relu(q[t,h,:] · k[s,:]) · w[t,h]`, the head index `h` **is** contracted. It fails
the availability test. It is a reduction axis, not a batchable one. Area 2 calls it correctly.

**The real instance: the sequence axis at MIXED.** `streamindex_topk.py:788` passes
`seq_batch_size = 1` for the mixed band, and `:278` asserts it. The MIXED band walks
`d[1] .. d[2]` sequences **one at a time**, while the DECODE band batches four. The sequence index
appears in the output and is never contracted — it passes the availability test cleanly. Area 2 names
it as axis (e)'s neighbour; area 6's E4 proposes it.

Payoff, against the seams file's two:
- **Payoff 1 (a larger matmul): unavailable.** It applies "only when the per-item matmul is under
  128 rows", and mixed's M is already 4096.
- **Payoff 2 (fewer grid steps): available.** And the file is explicit that the second comes with
  none of the first, and that the natural check — *does packing fill the MXU?* — "can correctly
  answer **no** while the transformation is still one of the largest wins available."

**Two things gate it, and neither sheet has both.** First, a.4: at the *instrument's* mixed shape
there is only **one** sequence, so there is nothing to batch and the candidate is unmeasurable on the
current harness. It is a production-shape candidate that the instrument cannot see — which makes it
the one candidate here whose absence from a microbenchmark result must not be read as a negative.
Second, §(b): batching `B` sequences multiplies `s` by `B` (`s ∝ B · bq_sz · bkv_p`), so at
`bq_sz = 128` even `B = 2` needs 64 MiB and does not fit. **S5 at mixed is blocked by the same
undeclared buffer that blocks `bkv_p`, and is unblocked by the same C4.** That connection is not
available from either sheet alone.

---

## Candidate list

Each spends something to buy something; the deciding quantity and its tag are the third column.
Ranking is wave 4's.

| # | candidate | areas it decides differently | deciding quantity | tag |
| --- | --- | --- | --- | --- |
| C1 | Narrow the cache record from 256 B to 129/160 B; requires changing the writer (S4-b) | 1 (boundary), 4 (precision), 7 (level) | writer-side scatter cost at the new width | `unmodelled` |
| C2 | Size `num_sublanes_total` from the batch's `kv_len`, bucketed to a few jit-static shapes (S3-ii) | 1, 2, 4, 5, 7, 8 — six sheets agree | number of recompiles across buckets | `assumed` |
| C3 | `bkv_p` 2 → 4 or 8 at **decode**; 8.62% / 9.64% faster [j-e8eaacd3] | 2 (B1), 3 (items 8, 9) | interaction with `align_to(9, bkv_p)` widening sublanes 80→96→128 at production | `unmodelled` (in flight) |
| C4 | Tile the einsum over `bkv` inside `compute_scores` so `s` never materialises at full width | 2, 3, 4, 5 jointly — unblocks C3 and C12 at mixed | whether Mosaic already tiles the `(B, bq_sz·32, bkv_sz)` f32 result | `assumed` |
| C5 | L3 — 3-D contraction landing scores already as `(…, 16, 128)`, deleting the largest relayout (S1-c) | 4 (L3) × 6 (dimension numbers) | relayout cost of a 128×2048 f32 tile at this shape | `unmodelled` |
| C6 | E2/L2 — transpose the score tile to `k · qᵀ`, putting `bkv` on sublanes (raises M to 2048, drops N to 32) | 4, 5, 6 | MXU cost at N = 32 lanes, below the measured sweep | `unmodelled` |
| C7 | 16-bit monotone-encoded scores across HBM; `sparsecore_topk` already bitcasts to monotone i32 internally | 4 (P7) × 6 × 8 | rank error at ranks 2047/2048 from a 16-bit key | `unmodelled` |
| C8 | Delete the `w` broadcast — L13 (transposed weights, no precondition) or E1 (fold `w` into `q`, needs `w ≥ 0`) | 4 (L13), 5 (B), 6 (E1) | whether `w ≥ 0` holds for this checkpoint | `assumed` |
| C9 | Delete the `-inf` fill; the mask already writes `-inf` over every invalid lane | 1, 2, 4, 5, 6, 7 — six sheets agree | existence of any consumer reading a lane the mask does not cover | `assumed` |
| C10 | P8/P5 — bf16 operands or bf16 scores buffer (accumulator stays f32; needs the SC-side change) | 4 × 5 × 6 × 8 | recall at rank 2048 under bf16 scores | `unmodelled` |
| C11 | Threshold-first two-pass select: 2× the einsum, scores never cross HBM (S3-iii) | 1 (boundary), 3, 6, 8 | whether a TC-side histogram mechanism exists at all — the VPU has no scatter-add | `unmodelled` |
| C12 | S5 — batch the sequence axis into the MIXED band's `seq_batch_size` | 1 × 2 × 6 | grid-step fixed cost × sequences saved, at a production mixed shape the instrument cannot produce | `unmodelled` |
| C13 | S4-a — producer emits `indexer_weights` in f32 and transposed, deleting an XLA pass; composes with C8's L13 arm | 4 × 7 | cost of the standalone upcast pass | `unmodelled` |
| **rejected** | S2 — concatenate decode's 4 sequences to M = 128 | 5 (K), 6 (E5) | **decided, no probe:** 3.57 W → 4.00 W, 12% slower on a term that is not binding | `measured` |
| **rejected** | S1-a — fold the e8m0 scale into `kv` before the contraction | 4 × 6 | **decided, no probe:** costs the 1.86× fp8 MXU rate to save one 2048-element transpose | `measured` |
| **rejected** | E6/H — 32-head sum as a block-diagonal ones-matmul | 5 (H), 6 (E6) | **decided, no probe:** 3.1% MXU fill | `modelled` |
| **infeasible** | B2 — mixed `bq_sz` 128 → 256/1024 | 2 (B2) | **decided by §(b) arithmetic:** `s` = 64 / 256 MiB, above 64 MiB physical | `modelled` |
| **carried** | S4-c — the writer-side scatter + reshape, 21 × ~291 µs each per step | not a kernel edit | route: handoff to whoever owns `mla_attention_op.py:103-104` | `measured` |

## Probe list

Each line: the question, what it decides, and what I would do differently under each answer.

| # | question | decides | under answer A | under answer B |
| --- | --- | --- | --- | --- |
| **P1** | Does the einsum reach the MXU as fp8, or does Mosaic upcast to bf16 first? Read `/xprof-llo`'s Mosaic MLIR operand dtypes and MXU matmul counts on the shipped kernel. | Mixed's floor: 48.4 µs (18% achieved) vs 120.6 µs (45%) at measured Pallas/XLA rates — a 2.5× swing `report.md` calls unsettleable from source. Also whether C10's P8 arm is already the status quo. | **fp8:** mixed has 5.6× headroom and is the priority; C10's bf16-operand arm is a 1.86× regression and dies. | **bf16:** mixed has 2.2× headroom, comparable to decode's 1.6×; rank C1/C2 (decode-side, HBM) above the mixed work, and C10's P8 arm is free because it is already happening. |
| **P2** | Does Mosaic materialise the `(B, bq_sz·32, bkv_sz)` f32 einsum result as one VMEM buffer, or tile the contraction? Same `/xprof-llo` dump; read the VMEM allocation. | The whole of §(b): whether `B · bq_sz · bkv_p ≤ 500` is real, whether C3 is available at mixed, and whether C4 and C12 exist as candidates. | **Materialised:** C4 is a prerequisite for C3-at-mixed and C12, and must be built first. `bkv_p` stays 2 at mixed. | **Tiled:** C4 is a no-op and drops off the list; take `bkv_p` to 8 at mixed directly, and §(b)'s mixed peaks fall from 37.5 MiB to ~6 MiB. |
| **P3** | Cost of the two relayouts at *this kernel's* shapes: `s_summed (128, 2048) f32 → (128, 16, 128)`, and `w (128, 32) f32 → (128, 32, 1)`, against their absence. | C5 and C8. The strike test on C5 says the ranking is decided **entirely** by this term: struck, the two options are identical in FLOPs and bytes. | **>10% of an inner step:** build C5 (L3's 3-D contraction) and take C8's L13 arm; both are pure wins. | **Inside noise:** drop C5 and C8, and spend the design effort on C1/C2, where the prize is `modelled` at 2× decode's HBM floor. |
| **P4** | MXU cost per unit work at N = 32 lanes vs N = 128, at fixed M and K. The measured sweep covers N ∈ [128, 512]; N = 32 is below it. | C6 (the E2 transpose). Probes the primitive, not the variant — it answers any future narrow-N candidate too. | **≤1.3× (like the measured N=256 non-monotonicity):** C6 is close to free and deletes the S1-c relayout by construction, making it strictly better than C5. | **≥4× (linear in wasted lanes):** C6 dies, and S1-c's relayout must be paid or absorbed by C5 instead. |
| **P5** | Does `bkv_p ∈ {4, 8}` change the kernel's output? Run the existing `instruments/test_correctness.py` at both values. And does the mixed band move when `num_kv_pages_per_block[1]` is varied, not just `[0]`? | Whether C3's measured 9.64% is bankable at all, and whether the lever exists in the band with the most headroom. `step_cost_RESULT.md` lists both as unsettled; a blocking factor *should* be value-neutral, which is exactly the class of belief that is cheap to check and embarrassing to assume. | **Match, and mixed moves:** C3 generalises and jumps the ranking. | **Mismatch:** C3 is dead, the ≈500 ns lever is unavailable, and §(b)'s whole `bkv_p` analysis becomes moot — which also removes C4's main reason to exist. |
| **P6** | Is `w ≥ 0` guaranteed? Read the `weights_proj` output path for a non-negativity constraint, or assert on the real checkpoint. Note `instruments/inputs.py:180` generates weights on `uniform(-1.5, 1.5)`, so **the instrument's synthetic weights are not evidence either way** and would silently fail an exactness check that real weights pass. | C8's arm: E1 (fold `w` into `q`, deleting a broadcast *and* a multiply) vs L13 (transposed storage, deleting only the broadcast). | **Guaranteed non-negative:** take E1; it is strictly stronger and DeepSeek's own kernel does it. | **Not guaranteed:** take L13 only, and record E1 as rejected-on-correctness so it is not re-proposed. |

**Not on this list, deliberately.** `probes/bkv_p_prod.py` — `bkv_p` at production shapes, where
`align_to(9, bkv_p)` widens `num_sublanes_total` — is **already in flight**. And the per-inner-step
fixed-cost question is **already answered** (j-e8eaacd3-6be4-4cdf-95a2, a.2); do not re-queue either.
