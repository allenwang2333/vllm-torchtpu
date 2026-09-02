# Area (1) — The fusion boundary — constraint sheet for the step-2 designer

Block: `streamidx_topk` (stage A = three TensorCore `pallas_call`s in
`streamindex_topk.py`; stage B = `sparsecore_topk`), GLM-5.2, tpu7x.
Baseline of record: **nightly**, DP=16, 64 seqs, decode `bq_sz=1`, `bkv_sz=2048`,
`num_positions=10240`, page 1024, `lkv_dim=256`, H=32, D=128, k=2048, 21 of 78 layers.
AgentX (`max_model_len` 262144, b=16) is **not** the baseline of record; every AgentX figure below
is labelled as such.

Reference file: `references/01-fusion-boundary.md`. Peer sheets read: `area2-partitioning.md`,
`area8-layout-code-style.md`. Facts handed to me by the coordinator from areas 4, 5, 6 are cited as
such.

Tags on every claim: `read from source (path:line)` · `measured (job)` · `modelled` · `assumed` ·
`peer (areaN)`. Paths are relative to `/home/annyan_google_com/workspaces/copy1/vllm-torchtpu`,
`sx:` = `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`,
`sc:` = `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`,
`mla:` = `src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py`.

---

## A. Constraints

### The arithmetic that decides this area

**A1. The score-matrix round trip is worth 0.79% of the decode call and 3.3% of the prefill call.
This is the single most important number on this sheet, and it says *do not fuse for bytes*.**
The reference's own test is step 1 of its `## Deciding`: `saving_seconds ≈ 2·intermediate_bytes/3.0e12`,
cross-checked against a measured 2.973 TB/s read-write stream
(`from 01-fusion-boundary.md`, `verified: j-050c7480 / j-ed517191`). Instantiated here, `modelled`:

| phase | score matrix | `2·bytes/3.0e12` | measured call | fraction |
| --- | ---: | ---: | ---: | ---: |
| nightly decode, T=64, n=10240 | 2.62 MB | **1.75 µs** | 221.4 µs `measured (j-8b1e7acc)` | **0.79%** |
| nightly prefill, T=1024, n=10240 | 41.9 MB | **28.0 µs** | 857.3 µs `measured (j-8b1e7acc)` | **3.3%** |
| AgentX decode, T=16, n=262144 | 16.8 MB | 11.2 µs | 269 µs `measured (j-c13c8436)` | 4.2% |

The reference says of this test: *"If that is small against the op's own runtime, fusing buys
nothing… This is arithmetic, not judgement."* It is small at every configuration. A design whose
stated advantage is "the score matrix never reaches HBM" has, at most, 0.79% at the baseline of
record. **This independently confirms `report.md` §4's rejection of the caller's original prior, by
a different route** (§4 argued from the traffic *share*; this argues from the absolute saving
against the measured call).

**A2. The `-inf` init is a third pass over the same buffer and is also closed.** `jnp.full(-inf)` at
`read from source (sx:745)`, T×80×128 f32 = 2.62 MB at nightly decode → **0.94 µs** `modelled` at
the 2.79 TB/s the profile observed. In-model it is `%broadcast.33911 = f32[16,2048,128]`,
**6.02 µs at 2.79 TB/s** `measured (j-a71a648c)` — but that is the **AgentX** 16.8 MB buffer; do not
quote 6.02 µs at nightly. `report.md` §8 closes this and I agree.

**A3. Fusing stage A and stage B has no *capacity* obstacle. The obstacle is the unit boundary.**
The bytes a fused kernel would need simultaneously resident, `modelled` from
`read from source (sx:684, :418-420)`:

| granularity | resident score bytes | against |
| --- | ---: | --- |
| decode, 1 row | 40 KiB | 31 MiB default scoped, 64 MiB physical `peer (area2 c6/c8)` |
| decode, one grid step (group of 4) | 160 KiB | " |
| prefill, one bq block (128 rows) | 5.24 MB | " |
| whole decode matrix | 2.62 MB | " |
| whole prefill matrix | 41.9 MB | over the 31 MiB default, under 64 MiB physical |
| SparseCore, one row `n=10240` | 40 KiB | **524288 B per subcore** `measured (j-0b008512)` |

Everything except the whole prefill matrix fits everywhere. The reference's operand-count /
block-row-collapse table (*"each doubling of the operand count roughly halves the block rows"*) is
**N/A here** and the designer must not budget from it: it measures double-buffered `BlockSpec`
operands, and the incumbent has **zero** — all four in-specs and the out-spec are `pltpu.HBM`
`read from source (sx:645-652)`, with the resident set hand-sized in `scratch_shapes` at 4.09 MiB
decode / 4.03 MiB prefill `peer (area2 c6)`.

**A4. The incumbent's iteration order already completes whole score rows, so the A-side of an A→B
fusion needs no restructuring.** `bq` is the outer `fori_loop` and `bkv` the inner
`read from source (sx:418, :420)`. At decode `bq_sz=1`, `num_bq=1`, so one grid step emits 4
complete rows; at prefill each bq block emits 128 complete rows across its 8 bkv steps. A fused
select could therefore fire at a natural point in the existing loop nest. **This is a positive
finding: the reason A→B is not fused is not the loop structure.**

**A5. There is no backward pass and no residual pressure. Area (1)'s residual sub-question is N/A,
with reason.** The op returns `int32` indices — non-differentiable — plus the aliased cache;
neither module defines a `custom_vjp`, a `Residuals` type or a backward
`read from source (sx:509-865, sc:380-457)`. The reference's four residual mechanisms
(de-fuse the activation, the `fuse_reciprocal` knob, refuse-and-recompute, the drop policy) all
apply only to training kernels. Per the reference: *"An inference-only kernel has no backward and
so has none of this constraint."* Say so in the design; do not omit it.

**A6. Stage A and stage B do not share a contraction axis, but the reference's `gmm`/`tgmm`
accumulator-blowup mechanism does not apply.** Stage A contracts `d = head_dim = 128` in
`jnp.einsum("bnd,bmd->bnm", …)` `read from source (sx:326-331)`. Stage B contracts nothing — it is a
selection over the position axis `m`, which is stage A's *inner-loop* axis
`read from source (sx:418, sc:147-303)`. So step 6 of the reference's `## Deciding` is answered:
the axes differ, but because stage B has no accumulator over a matmul reduction there is no
"whole-output residency" to size. What replaces it is A3's row-residency table, which passes.

### What the boundary actually costs today

**A7. Two of the three stage-A `pallas_call`s launch with a zero-length grid at a pure-decode step,
costing ~10.6 µs/layer.** `measured (j-a71a648c, AgentX)`: `StreamIdxTC-d` is bimodal — 21
instructions at **114.63 µs** and 21 at **5.40 µs** — and `StreamIdxTC-m` is 21 at **5.22 µs**.
Per indexer layer the profile reads 125.25 µs of TensorCore compute, so the two empty launches are
**10.62 / 125.25 = 8.5% of stage A's TensorCore time**, and 21 × 10.62 ≈ **223 µs/step**.
`peer (area2 c11)` reaches the same number. The *count* is deployment-independent; the µs is
AgentX-only and unmeasured at nightly.

The exact mechanism, `read from source (mla:183, sx:755-809)`, is finer than "the grids are always
zero" and the designer needs the precise form:

```
distribution = torch.stack([rd[0], rd[0], rd[2]])          # mla:183
call 1 (DECODE,  sbs=4, bq=1):  grid = (rd0 // 4)
call 2 (DECODE,  sbs=1, bq=1):  grid = (distribution[1] - decode_batch_end) = rd0 % 4
call 3 (MIXED,   sbs=1, bq=128): grid = (rd2 - distribution[1])            = rd2 - rd0
```

Because `distribution[1]` is set to `rd[0]`, the "chunked-prefill-only" band the kernel's docstring
describes `read from source (sx:549-551)` is **empty by construction**, and call 2 degenerates into
the decode remainder. So: call 2 is empty iff `rd0 % 4 == 0` (true at the nightly 64 and the AgentX
16), call 3 is empty iff there is no prefill or mixed sequence. At a pure-prefill step the roles
invert — calls 1 and 2 are empty and call 3 runs `modelled` from the same three lines.

**A8. The three calls are chained through one aliased buffer and therefore cannot run
concurrently.** Call 1's output is call 2's `scores_init`, and call 2's is call 3's
`read from source (sx:761, :779, :797)`, with `input_output_aliases={9: 0}` on each
`read from source (sx:729)`. Any candidate that proposes to run them in parallel must first break
that chain (e.g. three disjoint output slices, merged afterwards) — and doing so reintroduces the
`-inf` init A2 currently gets for free through the alias.

**A9. The XLA plumbing around the kernels costs more than the round trip would save.**
`measured (j-a71a648c, AgentX)`: the bucket *indexer plumbing (`conditional`, `bitcast_convert`,
cond copies)* is **1.230 ms/core-step = 4.6%** of a 26.584 ms step — **58.6 µs per indexer layer**,
against a modelled 11.2 µs the AgentX round trip would save (A1). The bucket's internal composition
is **not attributed** in any artifact I read; treat 58.6 µs as an upper bound on what op-count
fusion can reach, not as a target. This is the strongest argument on the sheet for spending fusion
effort on the *surrounding XLA ops* rather than on the A→B intermediate.

**A10. The op's real boundary is two torch-level statements outside the JAX graph, which no fusion
can reach and the microbenchmark does not measure.** `read from source (mla:138-141, :196-199)`:

```python
def op(*args):
    new_cache, topk_indices = jax_op(*args)
    args[2].copy_(new_cache)          # mla:140
    return topk_indices
...
self.topk_indices_buffer[:hidden_states.shape[0]] = -1                   # mla:196
self.topk_indices_buffer[:num_out, :self.topk_tokens] = topk_indices[…]  # mla:198-199
```

The two buffer writes are 2 × T·k·4 = 1.05 MB at nightly decode → **0.35 µs** `modelled`, ×21
layers ≈ 7 µs/step. Small. `args[2].copy_(new_cache)` is the one to check: `donate_argnums=(2,)`
`read from source (mla:128)` should make it a self-copy, but **whether donation actually elides it
is not established by any artifact I read**. If it does not, it is a full KV-cache copy per indexer
layer — 198 MB at the AgentX shape `u8[3027,64,4,256]` `measured (j-a71a648c)`, ≈67 µs/layer
`modelled`, which would land inside A9's unattributed "cond copies". **Named probe, D3.**

**A11. The hand-rolled DMA schedule is justified for `cache_kv` and only for `cache_kv`.** Step 7 of
the reference's `## Deciding` allows taking the schedule over only for *"an operand the `index_map`
cannot address at all"*, and warns that absent that the managed pipeline wins by **1.62×**
(`verified: j-050c7480, manual_dma_vs_managed`) while hand-issuing a transfer early buys **1.00×**
(`verified: j-a4210c9f`). Per operand, `read from source`:

| operand | schedule | `index_map`-expressible? |
| --- | --- | --- |
| `cache_kv` | `_fetch_bkv` (sx:160-198): `bkv_p` DMAs/slot through a page table, index `page_indices_ref[…]` clamped at runtime; trip count `num_bkv` derived in-kernel from `kv_len_max = jnp.max(...)` (sx:275-276) | **No.** Gather, not a block; and the loop bound is an in-kernel scalar. Hand-roll justified. |
| `q`, `indexer_weights` | `_fetch_bq` (sx:200-219): variable-length copy, `sz = min(bq_sz, curr_q_end - q_len_start)` | **At prefill no** (ragged `cu_q_lens`). **At decode arguably yes** — `bq_sz=1`, rows contiguous per group. A decode-only candidate that hands q back to a managed `BlockSpec` is testing the reference's 1.62× directly. |
| `scores` out | `start_send_scores` / `wait_send_scores` (sx:128-151): starts the write, defers the wait to the *other* buffer's next turn | Expressible, but see D2 — this is the deferred-write pattern the reference's gap G3 says was never tested, and the incumbent is a natural experiment for it. |

**A12. A fallback threshold exists for hardware but not for size, and the reference says the size
gate is the part most often missing.** `_sc_topk_direct` raises `NotImplementedError` when
`get_tpu_info().sparse_core is None` `read from source (sc:315-318)` — exactly the `ragged_gather`
pattern the reference endorses. There is **no** size gate anywhere: nothing falls back to
`lax.top_k` below a shape, and `ragged_gather_reduce`'s 60%-of-VMEM gate has no analogue here.
Per the reference, *"a fallback to XLA is part of the design, and the threshold is a number you have
to pick."* Every option must state its gate and its number, or state that it deliberately has none.

**A13. `enable_early_exit` is a fusion *knob* in the reference's sense and it can only unfuse.**
`read from source (sx:858-864)`: when `max(seq_lens)//compression_ratio <= k` the whole common path
— both kernels and all the plumbing — is replaced by pure index arithmetic. `measured (j-8b1e7acc)`:
`prod_prefill_q1024_ctx1024` costs **50.6 µs** against 857.3 µs for the same shape at ctx 9216.
At the nightly steady state (ISL 8192 > k=2048) it never fires, but it is compiled into all 21
layers either way and is the `conditional` in A9's bucket.

**A14. Two `jnp.pad`s in the block are statically dead at GLM-5.2's shapes and two more are
identities.** `read from source`: `sx:812-817` fires only if `num_positions < k` (10240 vs 2048 —
never); `sx:820-821` casts f32→f32 — never; `prepare_q_inputs` pads to `align_to(32,4)=32` and
`align_to(128,128)=128` (sx:444-445) — identity; `prepare_index_weights` likewise (sx:464);
`sc:322` pads `row_lengths` to `align_to(64,16)=64` — identity. Do not count these as fusion
opportunities and do not let a candidate claim credit for removing them. XLA should already elide
them; that they still appear in the compiled program is not established.

**A15. Do not fuse an epilogue onto the einsum without first matching it alone.** The reference's
sharpest measured result: a hand-written fused matmul epilogue ran **2.6× slower** than XLA's
(`verified: j-ed517191`), and the best hand-written Pallas matmul on this chip is 397.6 Tops/s
against XLA's 764.8. Stage A's `relu → *weights → sum_h → *scale → mask` chain
`read from source (sx:333-352)` *is* a fused epilogue on an einsum. It is already written and
already inside a kernel, so the incumbent has paid this; what the constraint forbids is a candidate
that **adds** work to that epilogue on the theory that it is free. Note the counterweight: the
reference's own "one case where fusion is unambiguous" is fusing onto an intermediate *already
resident for another reason*, and `s` at `sx:326` is exactly that. So the existing epilogue is the
good kind; a wider one is not automatically.

---

## B. The `Emit …` schema, instantiated

**Area (1) is the one area in `AREAS.md` with no `Emit …` line** (`read from source
(AREAS.md:21-31)`) — it closes on *Advantage / Disadvantage* prose instead, unlike areas (2)-(7).
`AREAS.md:15-17` nonetheless holds an option to the area's emission, so the schema has to come from
somewhere. I take it from the reference's **`## Deciding`**, which is that area's procedure and the
only enumerable thing area (1) has. Ten fields; the seven `## Deciding` steps plus three this block
forces (the unit assignment, the launch inventory, and the fallback gate). **An option that leaves
a field blank has not decided area (1).**

### B1. The schema

1. **Boundary inventory** — every op in the block, `inside` / `outside`, and *which kind* of reason
   (bytes, capacity, control flow, aliasing, addressing, unit).
2. **Round-trip saving** — for every intermediate the option proposes to eliminate:
   `2·bytes/3.0e12`, at decode and at prefill, as a fraction of the measured call.
3. **Already-resident test** — for anything fused: is the intermediate resident for a reason the
   fusion did not invent? (Reference: fuse onto one that is; do not restructure to make one.)
4. **Residency budget** — bytes simultaneously resident, against 31 MiB (TC default scoped),
   64 MiB (TC physical), and 524288 B/subcore (SC). Say which unit each budget is on.
5. **Operand count** — and whether the block-row-collapse table applies. It does not if operands are
   `pl.ANY`; say so rather than skipping.
6. **Residuals** — what a backward would need saved. `N/A because —` is a valid answer and is
   required to be written.
7. **Contraction axes** — do the fused pieces contract the same axis? If not, size the loser's
   accumulator.
8. **Schedule justification, per operand** — managed `BlockSpec` or hand-rolled, and for each
   hand-roll, which of the reference's three cases justifies it.
9. **Launch inventory** — kernels per call, per layer, per step; how many have a dynamically empty
   grid; which unit each runs on.
10. **Fallback gate** — the predicate, the threshold, and the number. "None" must be deliberate.

### B2. The incumbent, at nightly **decode** (T=64, 64 seqs, ctx 8704, n=10240)

| field | value |
| --- | --- |
| 1. inventory | ~9 XLA ops before stage A (`mla:88-106`: pad, arange, searchsorted, 3 gathers, where, scatter, bitcast); **3 TC `pallas_call`** (`sx:757/775/793`); ~8 XLA ops between (`sx:811-830`: reshape, 2 dead ops, `_effective_row_lengths`); **1 SC `pl.kernel`** (`sc:333`) preceded by `bitcast_convert` (`sc:320`); 1 slice (`sx:838`); all wrapped in `lax.cond` (`sx:860`) whose other branch (`sx:841-856`) is also compiled; 2 torch-level writes outside the graph (A10) |
| 2. round trip | A→B intermediate 2.62 MB → **1.75 µs, 0.79%** `modelled`. init 2.62 MB → 0.94 µs. A→A chain: 0 bytes (aliased, both grids empty) |
| 3. already resident | The score matrix is **not** resident for another reason — it is materialised precisely to hand to stage B. By the reference's rule this is the *bad* kind of fusion target |
| 4. residency | see A3. 40 KiB / 160 KiB / 2.62 MB on TC; 40 KiB `resident_vmem` per subcore on SC (`sc:347`, `slice_len = align_to(10240,16) = 10240` words) |
| 5. operands | stage A 4 in + 1 out, **all `pltpu.HBM`** (`sx:645-652`) → collapse table N/A; 5 scratch, 4.09 MiB `peer (area2 c6)`. stage B 2 in + 1 out + 6 scratch (`sc:346-353`) |
| 6. residuals | **N/A — inference-only, int32 index output, no `custom_vjp` in either module** (A5) |
| 7. contraction | A contracts `d=128`; B selects over `m`, A's inner-loop axis. No shared axis, no accumulator blowup (A6) |
| 8. schedule | `cache_kv` hand-rolled, justified (reference case 1 + in-kernel trip count); q/weights hand-rolled, **not** clearly justified at decode; scores-out hand-rolled with a deferred wait (A11) |
| 9. launches | **3 TC + 1 SC per call; 63 TC + 21 SC per step; 42 of the 63 empty** (A7) `peer (area2 B)`. Stage A on TensorCore, stage B on SparseCore |
| 10. fallback | hardware gate only (`sc:317-318`). No size gate. `enable_early_exit` fires at `max(seq_lens)/cr <= k` — never at nightly steady state (A12, A13) |

### B3. The incumbent, at nightly **prefill** (T=1024, 1 seq, ctx 9216, `bq_sz=128`, `num_bq=8`)

Differences from B2 only:

| field | value |
| --- | --- |
| 2. round trip | 41.9 MB → **28.0 µs, 3.3%** of 857.3 µs `modelled`. init 41.9 MB → 15.0 µs `modelled` |
| 4. residency | per bq block 5.24 MB; whole matrix 41.9 MB — **over the 31 MiB default**, under 64 MiB physical. Separately, `peer (area2 c7)` flags a **32 MiB f32 einsum intermediate** at `bq_sz=128` as an alternative hypothesis for prefill's 857 µs; a fused candidate adds to that, not to an empty budget |
| 9. launches | grids are **0 / 0 / 1** — calls 1 and 2 empty, call 3 runs (A7). Same 3 TC + 1 SC |
| — | `report.md` §8 records prefill as **`unresolved`** at 100.8% of its adjusted floor, its dominant term an `assumed` VPU rate. **Do not rank a prefill fusion candidate on the 72.7% figure** |

### B4. What each design option must return

Every option returns B1's ten fields, at **both** decode and prefill, plus:

- **The delta against B2/B3 field-by-field**, not a prose summary. Step 3 is held to this.
- **Its launch count** — the single field where the incumbent has a known, cheap, 8.5%-of-stage-A
  defect (A7). An option that does not change it must say so.
- **A named number for `2·bytes/3.0e12`** on any intermediate it claims to eliminate, and the
  fraction of the measured call that is. If the fraction is under the ±8.5% run-to-run spread
  `measured (j-c13c8436)`, the option must say it cannot be measured end to end and name the
  microbenchmark that can.
- **Which unit every fused piece lands on**, cross-referenced to area (5). A fusion that silently
  moves work from SparseCore to TensorCore is a unit decision wearing a boundary decision's clothes.
- **Its file plan**, per `peer (area8 §2)`: *"an option that fuses stage B into stage A collapses
  this to one operation and one file set"* — the differing file count is itself the signal that two
  options sit at different boundaries, so both plans get written out.

---

## C. The fusion boundary itself

### C1. Where the boundaries are, and which are forced

Six boundaries. Three are forced, three are incidental, and they are routinely confused.

| # | boundary | kind | forced? |
| --- | --- | --- | --- |
| 1 | XLA plumbing → stage A | kernel | **incidental** — index arithmetic on tiny arrays, fusable or scalar-prefetchable |
| 2 | stage A call 1 → call 2 → call 3 | kernel, same unit | **incidental** — exists because `seq_batch_size` and `bq_sz` are static Python ints sizing scratch (`sx:654-684`), not because of hardware |
| 3 | **stage A → stage B** | kernel **and unit** (TensorCore → SparseCore) | **forced** as long as the select stays on SC |
| 4 | stage B internal: stage 1 → gather → stage 2 (`p>1` only) | kernel, same unit | **incidental** — and **absent at the baseline of record** `peer (area5)`: `_pick_partition(64,10240,2048)` returns `p=1` (`sc:360-377`) |
| 5 | the `lax.cond` at `sx:860` | control flow | **forced** — a kernel cannot decline to launch |
| 6 | the torch-level `copy_` / buffer writes (`mla:140, :196-199`) | framework | **forced** — outside the JAX graph entirely |

**Boundary 3 is two edges, not one, and the distinction decides the design.** A *kernel* boundary is
a place where an intermediate is materialised and a launch is paid; it is crossed by writing to HBM
and reading back, and it can be removed by putting both sides in one `pallas_call`. A *unit*
boundary is a place where work moves between physically distinct compute — TensorCore's VPU/MXU and
SparseCore's 2 cores × 16 subcores `measured (j-0b008512)`. Removing a kernel boundary is a
refactor. Removing a unit boundary means either moving the top-k onto TensorCore (an area (5)/(6)
decision, and it would be undoing `6142d5e7`) or writing a single kernel that spans both units,
which is **unprobed** (C4).

The practical consequence: **the two edges have different prizes.** The kernel edge is worth
1.75 µs at nightly decode (A1). The unit edge is worth up to the TensorCore-stall figure —
**174.6 µs/layer** `measured (j-a71a648c, AgentX)` — and those are not the same order of magnitude.
A design that conflates them will optimise the 1.75 µs.

### C2. The three stage-A calls — what merging costs and buys

**Buys**, `measured (j-a71a648c, AgentX)`: removal of 10.62 µs/layer of empty launch, **8.5% of
stage A's TensorCore time**, ≈223 µs/step across 21 layers. Also removes 2 of the 3 aliased
read-modify-write hops (A8), which today serialise three kernels that do nothing.

**Costs**, `read from source (sx:628-692)`:

1. **`seq_batch_size` and `bq_sz` are static and size the VMEM scratch.** One merged kernel gets the
   union of the three scratch sets. The expensive term is `bo_scores_double_buf`,
   `(2, seq_batch_size × bq_sz, num_sublanes_bkv, 128)` f32 (`sx:683-684`): decode
   2×4×16×128×4 = 64 KiB, mixed 2×128×16×128×4 = **2 MiB**. Plus the einsum intermediate, which at
   `bq_sz=128` is **32 MiB** `peer (area2 c7)` against decode's 0.5 MB. **A merged kernel makes
   every decode step carry prefill's footprint** `modelled`. That is the real cost and it is why
   merging call 3 is a different proposition from merging call 2.
2. **The grid extents are dynamic** — `(end_seq_idx - start_seq_idx)//seq_batch_size` with both ends
   traced from `distribution` (`sx:643`). A merged call needs a single grid covering
   `rd0//4` batched groups + `rd0%4` singletons + `(rd2-rd0)` mixed sequences, with the *kind* of
   work varying per grid step. Inside a kernel that is a `lax.cond` or `pl.when` nest on
   `pl.program_id(0)` — which the reference warns is an **optimization barrier**, though it also
   documents the mitigation (*"splitting this into 6 possible cases … allows the compiler to overlap
   the work within the branches"*, `ragged_dot`) and notes `gmm_v2` does the whole thing with a
   `lax.cond` nest of statically specialised bodies.
3. **`assert static_q_len == 1` when `seq_batch_size > 1`** (`sx:628-629`) — batching across
   sequences is decode-only, so a merged kernel cannot batch the mixed case at all.

**Recommendation to the designer, in three tiers:**

- **Tier 1, cheapest and the one I would take: merge calls 1 and 2.** Both are `MlaCase.DECODE`,
  both `bq_sz=1`, both the same `bkv_p`; they differ only in `seq_batch_size` (4 vs 1). Handle the
  remainder by masking inside the batched kernel — the group of 4 already pays the group max
  unguarded `measured (j-63b9f2e4, 4.7% agreement)`, so a partial group costs what a full one costs
  anyway and masking is free in the epilogue. Scratch grows by nothing. Recovers ~5.4 µs/layer.
  `peer (area2 c12)` reaches the same conclusion.
- **Tier 2: `lax.cond` the mixed call away when `rd2 == rd0`.** Cheaper to write than a merge, but
  it adds 21 more `conditional`s per step to the bucket A9 already measures at 4.6% of the step, and
  the cost of an XLA `conditional` around a `pallas_call` is not measured anywhere. **Probe D1
  before choosing this over tier 1.**
- **Tier 3: merge all three.** Pays cost 1 in full. Only defensible if the design has already
  reduced the mixed case's scratch — i.e. if it is solving the prefill problem too, which
  `report.md` §8 says is `unresolved` and out of scope.

**One thing to check before spending anything here.** The 5.40/5.22 µs figures are AgentX, and they
are *inferred* to be the empty launches rather than isolated — `report.md` §8 lists "whether
`StreamIdxTC-d`'s bimodality is by design" as an open step-2 probe and calls A7's explanation a
*candidate*. It is cheap to confirm (D1) and confirming it turns an 8.5% estimate into a number.

### C3. What can be pulled inside, what cannot, and why the reasons differ

**Can be pulled in, ranked by prize:**

| op | prize | mechanism |
| --- | --- | --- |
| the two `jnp.take_along_axis` gathers in stage B (`sc:439, :452`) | **up to 206.6 µs/layer** — 2 × 51.54 µs SC + 2 × 51.77 µs TC stall `measured (j-a71a648c)` | stage 2 reads `scores_hbm` at the candidate indices instead of materialising `cand_scores`. **`p=1` at nightly `peer (area5)`, so this is worth zero at the baseline of record.** Largest measured prize in the block, at the wrong deployment |
| `_effective_row_lengths` (`sx:482-506`) | op count only; feeds A9's 58.6 µs/layer bucket | it is `cu_q_lens`/`seq_lens` arithmetic already available in SMEM to stage B; recompute in-kernel or scalar-prefetch. Area (7) owns *where* |
| the three token→sequence maps | op count only | computed **three times per call**: `jnp.searchsorted` at `mla:93`, again at `sx:843` (fast path), and an equivalent `T × max_num_seqs` mask-sum at `sx:494-497`. `read from source`. **De-duplicating them in XLA needs no kernel at all** and should be done before any fusion is designed |
| the `-inf` init (`sx:745`) | 0.94 µs/call `modelled` at nightly decode | stage A writes `num_bkv × bkv_sz` columns per real row; only the tail columns and the padding token rows need initialising. Fold into the epilogue. **Not worth doing alone**; worth doing if the epilogue is being rewritten |
| `bitcast_convert_type` ×2 (`mla:106`, `sc:320`) | free in principle | both are pure reinterpretations. They appear by name in A9's bucket, so whether XLA elides them is `[unverified]` |

**Cannot be pulled in — and the three reasons are genuinely different:**

1. **The scatter into `cache_kv` (`mla:103-104`) — an *aliasing* reason, plus a hazard.** The buffer
   is donated (`donate_argnums=(2,)`, `mla:128`) and copied back at the torch boundary (`mla:140`),
   so its identity has to survive the whole op. Independently, stage A **reads the pages this
   scatter just wrote** — a read-after-write on HBM that today is ordered by XLA's data dependency
   between two separate ops. Fusing them puts both sides inside one kernel whose grid steps touch
   different sequences and therefore different pages, so the hazard is *probably* benign at decode —
   but "probably" is not an argument, and the prize is 16 KB of traffic at nightly decode
   (`T × lkv_dim = 64 × 256`), i.e. **0.01 µs** `modelled`. **Do not fuse this.** Note that the
   reference's MLA example (`input_output_aliases` on `cache_kv`) is the *good* version of this
   pattern — but there the cache write is the kernel's own output, not a producer the same kernel
   then reads back.
2. **The `take_along_axis` / page-table gathers — an *addressing* reason, and it is not a
   prohibition.** Indices are computed at runtime but the shapes are static; the incumbent already
   does exactly this inside a kernel for `page_indices` (`sx:179-185`, with a runtime `jnp.minimum`
   clamp). So "outside" here is a choice. The thing that genuinely cannot be expressed is not a
   gather, it is a **dynamic grid extent** — and that is what forces the three separate calls (C2).
3. **The `lax.cond` (`sx:860`) — a *control-flow* reason.** It selects between two programs, one of
   which launches no kernel at all. A kernel cannot decline to launch; the in-kernel equivalent is
   `pl.when` around the body, which still pays the launch and, per the reference, erects an
   optimization barrier where there is currently none inside the kernel. It also cannot be hoisted
   out of the 21 layers: each layer's predicate is the same, but they are 21 separate HLO regions.

### C4. The score-matrix round trip — the largest fusion question, and the answer is "reframe it"

**What is established.** Stage A writes `f32[T, num_sublanes_total, 128]` to HBM via `pltpu.HBM`
out-spec `read from source (sx:652, :725-728)`. Stage B receives `words.reshape(-1)` as a kernel
operand and DMAs it from HBM into `resident_vmem` inside `_topk_kernel`
`read from source (sc:160, :347, :356)`. So the matrix crosses HBM, once each way: **2.62 MB at
nightly decode, 41.9 MB at nightly prefill**. SparseCore is reachable from Pallas via
`jax.experimental.pallas.tpu_sc as plsc` `measured (j-0b008512)` `peer (area5)`, with
`num_cores=2, num_subcores=16, vmem_capacity_bytes=524288` per subcore — so both
`references/hardware-facts.md` and `05-compute-units.md` are wrong to say SparseCore is unreachable,
and a design must not inherit that.

**What the arithmetic says.** Eliminating the round trip is worth **1.75 µs, 0.79% of the decode
call** (A1). Under the reference's `## Deciding` step 1 — *"Small against the op's runtime? Stop."* —
this fusion is dead on the bytes argument at the baseline of record. It is 3.3% at prefill, which
`report.md` §8 has ruled `unresolved` anyway.

**What is unknown, and it is a different question.** Whether a TensorCore `pallas_call` and a
SparseCore `plsc` kernel can share any memory space other than HBM is **not established by any
artifact I read**, and I did not find a mechanism in `sparsecore_topk.py` or in the `plsc` surface
it uses. Two sub-questions, both open:

- **(i) Shared memory.** Can a value produced on TensorCore reach a SparseCore subcore's tile memory
  without an HBM hop? `_topk_kernel` takes `scores_hbm` as an operand and DMAs it in; nothing in the
  file suggests an alternative. **Unknown.** Even if the answer is yes, the prize is 1.75 µs.
- **(ii) Concurrency.** Can a TensorCore grid and a `VectorSubcoreMesh` be in flight at the same
  time? `peer (area5)` records this as **unprobed** and prices it at the **174.6 µs/layer of
  measured TensorCore stall on SparseCore** `measured (j-a71a648c, AgentX)`. Decomposing that
  figure: 2 × 51.77 µs is TC stalling on the `gather_offload`s, which are `p=2` machinery **absent
  at nightly**, leaving ≈71 µs/layer attributable to stalling on `sc_topk_direct`. At nightly the
  analogue is unmeasured; stage B is 67.4 µs standalone `measured (j-8b1e7acc)` and 30% of the call.

**The reframe, and it is the main recommendation of this section.** Question (ii) is worth ~40× more
than question (i), and — crucially — **it does not require shared memory.** Overlapping the two
units means pipelining *through* HBM: stage A finishes the score rows for sequence group *g* while
stage B selects on group *g−1*. A4 establishes that the incumbent's loop order already produces
complete rows at a natural granularity (4 rows per decode grid step, 128 per prefill bq block), and
A11 establishes that the score write-back is already a deferred-wait async copy. The missing piece
is a dependency finer than "op B consumes op A's whole output", which XLA cannot express between two
`custom_call`s.

This is precisely the reference's **"second reason to fuse: a schedule you could not otherwise
write"** — and precisely the mechanism its `## Gaps` says was never validated (D2). The reference's
one attempt found the naive hand-roll **27-31% slower** than not fusing (`verified: j-a4210c9f`),
with the diagnosis that `write.start(); write.wait()` in the same step exposes the write in full,
and the explicit note that tokamax does *not* write it that way. **The incumbent already writes it
the right way** (`start_send_scores` / `wait_send_scores` on alternating buffers, `sx:128-151`), so
this block is an unusually good place to settle the reference's open question.

**What must not be claimed.** That fusing A and B removes the HBM traffic *and* delivers the
overlap. Those are separate mechanisms with separate prizes, and the reference's `saving_seconds`
formula captures only the first. Quote them separately or the option's predicted speedup is
double-counted.

**Marked clearly as unknown:** (a) whether TC↔SC shared memory exists on this stack; (b) whether a
TC grid and a `VectorSubcoreMesh` can overlap; (c) at nightly, how much of stage B's 67.4 µs is TC
stall versus SC work — no nightly profile exists, and `report.md` §6.1 records that the four
captured profiles contain no nightly-config step and no indexer-active prefill step at all.

---

## D. Gaps this block turns on, and contradictions

### Gaps in `01-fusion-boundary.md`

**D1 (a gap the file does not have, and should — highest priority).** The file prices a fused kernel,
an HBM round trip, an operand slot and a grid step, but has **no entry for (a) a cross-unit
TensorCore↔SparseCore boundary, and (b) a `pallas_call` whose dynamic grid extent is zero.** Both
are this block's central facts: (b) is 42 of 63 launches per step and 8.5% of stage A's TC time
(A7), and (a) is the boundary the whole design turns on (C4). `peer (area2)` independently flags the
same missing empty-launch price. The only figure anywhere is **5.40 µs**, AgentX, inferred rather
than isolated.
*Decision it flips:* whether tier 1/tier 2 of C2 is worth writing. Under "an empty launch is ~5 µs",
merging calls 1+2 recovers ~5.4 µs/layer ≈ 113 µs/step and is the cheapest win on this sheet. Under
"an empty launch is ~0.2 µs and the 5.40 µs was something else", C2 collapses entirely and the
`StreamIdxTC-d` bimodality is still unexplained.
*Probe:* a `/run-command-in-cdk` microbenchmark timing `run_scores_kernel` at grid extents 0, 1 and
16 in the same job, plus an XLA `lax.cond`-guarded launch as a third arm to price tier 2. Minutes on
a 4-chip pod. **I did not run it — this brief is source-reading and arithmetic only.**

**D2 — *"The pipelining argument in 'The second reason to fuse' is sourced, not measured… A version
that defers the write the way the cross-entropy kernel does would test the actual claim, and has not
been written."*** **Turns on, and this block is the natural experiment.** The reference's failed
probe did `write.start(); write.wait()` in the same grid step; the incumbent defers the wait to the
other buffer's next turn (`sx:128-151`), which is the version the gap says was never written.
*Decision it flips:* whether to design a cross-unit A/B pipeline at all (C4).
Under **"the mechanism is real when the write is deferred"** → the ≈71 µs/layer of TC stall
attributable to `sc_topk_direct` (AgentX; nightly analogue unmeasured) is attackable by overlap, and
that is the largest lever on this sheet at the baseline of record. Under **"the mechanism is
absent"** → the A→B boundary is fixed at 1.75 µs of prize, stage B's 30% can only be attacked from
inside stage B, and area (1) has almost nothing to contribute to this block.
*Probe:* the concurrency half of C4(ii) — can a TC grid and a `VectorSubcoreMesh` be in flight
together. `peer (area5)` already names this as unprobed; it is the same probe and should be run
once, for both areas.

**D3 — *"Whether Mosaic's hoisting of a hand-issued `.start()` holds at other shapes is unknown —
it was observed at one grid and one tile size."*** **Turns on, weakly but cheaply.** The incumbent
hand-rolls *every* operand (A11), and at decode `q` is contiguous and could be a managed
`BlockSpec`. The reference measures the managed pipeline winning by **1.62×** where there is nothing
to overlap, and hand-issuing early buying **1.00×**.
*Decision it flips:* whether a candidate hands `q`/`weights` back to `BlockSpec` at decode. Under
"Mosaic hoists everywhere", hand-rolling them is pure code cost and the managed path is better.
Under "hoisting is shape-dependent", the hand-roll may be load-bearing at `bq_sz=1`.
*Also folded into this probe:* A10's open question — does `donate_argnums=(2,)` actually elide
`args[2].copy_(new_cache)` at `mla:140`, or is there a full KV-cache copy per indexer layer? That is
worth ≈67 µs/layer `modelled` at the AgentX shape if the answer is bad, which would make it larger
than everything else in section C. It is an HLO read, not a hardware run.

**D4 — *"No probe here found a fused Pallas kernel that beats XLA at a realistic shape."*** **Turns
on, and this block appears to be the counterexample the file says it lacks.** `peer (area6)`:
`6142d5e7` replaced `approx_max_k(recall_target=1.0)` — which is **exact** — with `sparsecore_topk`,
for performance. That is a Pallas/SparseCore kernel beating a tuned XLA op at the production shape.
**The margin was never recorded, and no GLM-5.2 eval baseline exists in the repo** `peer (area6)`.
*Decision it flips:* step 5 of the reference's `## Deciding` — *"Is a tuned XLA op inside the
proposed boundary? Then the burden of proof is on the kernel."* Under "XLA's `approx_max_k` was
close", a candidate is allowed to propose returning stage B to XLA and deleting 457 lines. Under
"the kernel was much better", stage B's floor has to be found inside the kernel and the 30% is
attacked from within.
*Probe:* time `lax.approx_max_k(recall_target=1.0)` and `lax.top_k` against `sparsecore_topk` at
`(64, 10240, 2048)` in one job. Minutes. This also feeds the report's own first-ranked step-2 probe
("what stage B's cost is actually a function of"), and its answer belongs back in the reference file
— it is the file's most honestly-flagged weakness.

**Not turned on, with reason:** the 512³-blocks matmul-epilogue rerun (no 512³ matmul here; stage A's
decode einsum is M=32, far below the sweep's box `peer (area2 C)`); and the `rhs_bias` /
`gmm_v2`-v1 divergence (no `ragged_dot` anywhere in this block).

### Where the reference and `report.md` disagree

**Contradiction 1 — the value of eliminating the score round trip.** The reference's whole
"What the round trip actually costs" section frames the HBM saving as fusion's primary and only
*measured* prize (45% on a workload that is nothing but the round trip). `report.md` §4 says the
prize here is negligible at both phases. **I believe `report.md`, and they are not actually in
conflict** — the reference supplies the formula that proves the report's point, and states the
caveat itself: *"the 45% saving here is what you get when the op is nothing but the round trip;
anything with real compute in it dilutes the fraction."* Instantiated: 0.79% at decode. *Designer
consequence:* run the reference's step-1 arithmetic before proposing any A→B fusion, and quote the
result. It takes a minute and it will say stop.

**Contradiction 2 — "fuse onto an intermediate that is already resident".** The reference's one
unambiguous fusion win rests on the accumulator being *already in VMEM for a reason the fusion did
not invent*, and it explicitly warns: *"do not restructure a kernel to make one resident."*
`report.md` §2's framing of the score matrix as the thing to eliminate points the other way — the
score matrix exists **only** to be handed to stage B (A3). **I believe the reference.** By its rule,
the A→B fusion is the bad kind, and the good kind in this block is the epilogue already fused onto
`s` at `sx:326-352` (A15). *Designer consequence:* an option proposing A→B fusion has to argue from
the *overlap* mechanism (C4), not the residency one, and must say which.

**Contradiction 3 — the operand-count table's scope.** The reference presents the block-row-collapse
table as *the* constraint deciding how much you can fuse. It measures double-buffered `BlockSpec`
operands; the incumbent has none — all five specs are `pltpu.HBM` (`sx:645-652`). **I believe the
source**: this is not a contested measurement, it is a scope mismatch, and a designer budgeting from
that table would produce a fiction. `peer (area2 D4)` makes the same point about the file's idiom
taxonomy — the incumbent is a `PrefetchScalarGridSpec` + all-`pl.ANY` + hand-rolled-DMA hybrid that
neither reference file names. *Designer consequence:* B1 field 5 must say "N/A, all operands
`pl.ANY`" rather than silently skipping, and the VMEM budget comes from A3 and `peer (area2 c6-c8)`.

**Contradiction 4 — SparseCore's existence.** `references/hardware-facts.md` and
`05-compute-units.md` state there is no SparseCore path on this stack. **They are wrong**
`measured (j-0b008512)` `peer (area5)`. `01-fusion-boundary.md` itself does not repeat the error and
in fact supplies the right pattern (`ragged_gather`'s `sc_info is None` fallback), which
`_sc_topk_direct` already implements at `sc:315-318`. Recorded so the designer does not inherit the
error from the sibling files.

**Not a contradiction, but a figure that will be misquoted:** `report.md` §8 closes the `-inf` init
at **6.02 µs, ~2% of the call**. That is the **AgentX** 16.8 MB buffer `measured (j-a71a648c)`. At
the nightly 2.62 MB it is **0.94 µs** `modelled`, ~0.4%. Both closed; quote the right one.

**One figure on this sheet that is soft and should be re-derived, not trusted:** `report.md` §6's
`stage A (derived)` column is `total − stage B − init` over three independently-optimistic
best-of-N timings, and the same subtraction goes negative on three rows of `j-63b9f2e4`. The 8.5%
empty-launch fraction in A7 comes from the **profile**, not from that column, and is the sounder of
the two. Run-to-run spread on a derived stage A is **±8.5%/+27%** `measured (j-c13c8436)`, so per
`report.md` §6 the incumbent must be re-measured in the same job as every challenger.
