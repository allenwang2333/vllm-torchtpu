# Probe 14 — boundary provenance: is the input `bitcast-convert` real, and does the output retiling survive in-model?

Compile-only. No timing anywhere in this probe. Every claim is tagged **[read]** (seen in
code/HLO/buffer-assignment), **[measured]** (a number produced by a job), or **[inferred]**.

## The two answers

**Q1 — production's `%bitcast_convert_type.3` is MATERIALISED.** Not aliased, not free.
XLA's own buffer assignment gives it a *separate, disjoint, exactly-adjacent* 40 MiB temp buffer
from its operand, in both the standalone module and the real production entry. **[read, decisive]**
The 26.251 µs open leg is real, and the production denominator is **155.82 µs, not 129.56 µs**.
This corrects probe 11's refusal to transfer the harness number (see "Correcting probe 11" below).

**Q2 — the output retiling SURVIVES in-model.** It is not a standalone-compilation artifact. In the
real enclosing entry the kernel's flat `s32[2097152]{0:T(1024)}` is still retiled to
`s32[1024,2048]{1,0:T(8,128)}` — *and* it now costs slightly more, because the reshape lands in VMEM
`S(1)` and a separate 8 MiB `copy-start/copy-done` DMA carries it back to HBM. **[read+measured]**
The Stage E designer's 60/40 bet against in-model survival is **wrong**.

**⚠ Scope limit found late.** In the one GLM-5.2 prefill step captured in
`j-ecf4ee85-fa1b-4bbb-9649`, the entire `StreamIdxTC -> bitcast-convert -> sc_topk_direct` chain sits
in the **untaken** branch of an XLA `conditional` and executed **0 times**. The gate is a per-step
runtime predicate, `max(seq_lens) <= 2048` (`streamindex_topk.py:858-864`), **not** a build-time
switch — so this says the captured chunk was gated, **not** that the kernel is dead. On the config's
8-chunk prefill it should run on ~6 of 8 chunks. See "What actually runs today" below; an earlier
draft of this file said "dead code at the benchmark" and that was an over-reach, now corrected.

Job: `j-23884c12-c23e-4d8f-a979` (Succeeded). Verbatim log beside this file.
Script: `probe14.py` (md5 `eee2acfb9164718cda553e35a88f7f58`). Artifacts in `hlo/` and `xdump/`.

---

## Q1 evidence

### 1. Buffer assignment — decisive [read]

`xdump/module_0000.jit_streamindex_topk...-buffer-assignment.txt:445-446` (standalone
`streamindex_topk`), both inside `allocation 194: size 85262592, preallocated-temp`:

```
value: <912 StreamIdxTC-m-bq_128-bkvp_2.1 @0> (size=41943040,offset=1179648):  f32[1024,80,128]{2,1,0:T(8,128)}
value: <913 bitcast_convert_type.3        @0> (size=41943040,offset=43122688): s32[1024,80,128]{2,1,0:T(8,128)}
```

`1179648 + 41943040 = 43122688` — exactly adjacent, therefore disjoint. Two live 40 MiB buffers.
Same result in the entry that actually ships,
`xdump/module_0031.jit__streamidx_topk_jax...-buffer-assignment.txt:744-745`, inside
`allocation 222: size 151781632, preallocated-temp`:

```
value: <4113 StreamIdxTC-m-bq_128-bkvp_2.1 @0> (size=41943040,offset=1753088):  f32[1024,80,128]{2,1,0:T(8,128)}
value: <4114 bitcast_convert_type.4        @0> (size=41943040,offset=43696128): s32[1024,80,128]{2,1,0:T(8,128)}
```

`1753088 + 41943040 = 43696128`, again adjacent and disjoint. Operand and result
shape-**with-layout** match (`{2,1,0:T(8,128)}` both sides), so these are the same op, checked by
shape and not by name. In the outer module it is renamed `.4`; `.3` there is the fp8 `q` bitcast
(`f8e4m3fn[1024,64,128]{2,1,0:T(32,128)(4,1)}`) — a different op.

### 2. Falsification attempt — no counterexample found [read]

The requested counter-test: any *known-free* instruction in the same module carrying a non-zero
`used_scoped_memory_configs`? Census of the 281-instruction ENTRY of
`hlo/hlo-inner_streamindex_topk.txt`: **16** have non-zero `used_scoped`, and they are exactly the
work-doing opcodes (`fusion` 4, `broadcast` 3, `custom-call` 3, `slice` 2, `copy` 1,
`bitcast-convert` 1, `dynamic-slice` 1, `reshape` 1). All **74** `bitcast`, **130**
`get-tuple-element`, **9** `constant`, **7** `parameter` are **zero** — and 215 of those 220 *do*
carry a `backend_config`, so the field is present-and-zero, not absent. **Zero counterexamples.**
`%bitcast_convert_type.3` is the *single largest* `used_scoped` consumer in the entry (16,777,216 B),
larger than the `StreamIdxTC` kernel that produced its input (13,742,080 B).

### 3. Window arithmetic is an exact identity, and `used_scoped` is the number to read [read]

`iteration_bounds ["2","5","1"]` x `output_window_bounds ["512","2","1"]` = 1024 rows x 10
sublane-tiles x 1 lane-tile = the whole `s32[1024,80,128]` T(8,128) buffer. One window =
512x16x128 elem = 4 MiB; x2 for `buffering_level "2"`, x2 for in+out = **16 MiB = exactly
`used_scoped_memory_configs`**, with `estimated_cycles: 66712` — a planned double-buffered
10-iteration full traversal. On `scoped` vs `used_scoped`: the difference does mean something.
`scoped` values repeat across many unrelated instructions (67043328, 33554432, 22649243, ...) while
their `used_scoped` range from 1,024 to 38,912 — `scoped` is the VMEM budget at that program point,
`used_scoped` the actual reservation. Read the 16 MiB, not the 22.6 MiB.

### 4. Synthetic control — reproduced the op; its temp-delta arm is a NULL [read/null]

`hlo/hlo-syn_f32_bcvt.txt` emits `s32[1024,80,128]{2,1,0:T(8,128)} bitcast-convert(f32[...same...])`
with a byte-identical plan (`cycles=66712, iter=['2','5','1'], win=['512','2','1'], buf=2,
used_scoped=['16777216']`) — a deterministic emitter path, not a one-module quirk. **But the paired
temp-size A/B failed as a test**: `syn_f32_bcvt` temp=0, `syn_s32_none` temp=0, delta 0 — in both
arms the 40 MiB landed in `output_size_in_bytes`, not `temp_size_in_bytes`, so it could not
discriminate. **A null, not evidence of aliasing.**

### 5. Correcting probe 11

Probe 11 refused to transfer its 26.251 µs because the harness op "converts **and reshapes and
retiles**" while production's is a pure reinterpret. That reading is wrong.
`probes/11-relayout-provenance/hlo/hlo-A0_base.txt:35-36` is the harness fusion's *body*:

```
%bitcast_convert_type.0 = s32[1024,10240]{1,0:T(8,128)}    bitcast-convert(%param_0.1)
ROOT %bitcast.2         = s32[128,80,8,128]{3,2,1,0:T(8,128)} bitcast(%bitcast_convert_type.0)
```

XLA emits the reshape/retile as a **free `bitcast`** — both shapes are the same bytes. The fusion's
only real work is the same 40 MiB in / 40 MiB out dtype traversal production does, and its
`backend_config` is byte-identical (`estimated_cycles 66712`, `used_scoped 16777216`) except window
rank. The transfer is justified. **[read]**

### Carried handoff candidate — priced, one objection checked

The op is **our own**, not XLA's: `sparsecore_topk.py:438`
`words = jax.lax.bitcast_convert_type(scores, jnp.int32)`, outside the Pallas call (metadata
`op_name="jit(streamindex_topk)/jit(sparsecore_topk)/bitcast_convert_type"`). Declaring the kernel
input `f32` and reinterpreting inside the kernel body deletes the HLO op outright:
**~26.251 µs of a 155.82 µs kernel = ~16.8%, from a call-site dtype change.**

Objection checked — does anything downstream need an `int32`-typed *ref*? **No**, all [read]:
`:485-487` `)(words.reshape(-1), lengths)` is dtype-agnostic; `:169`
`dma(scores_hbm.at[...], resident_vmem, 1)` only needs src/dst dtypes to *match*, and `f32`/`f32`
matches; `:172-173` already does a full in-place load / `monotone_key` / store over every word of
`resident_vmem` in VMEM, so the reinterpretation folds into an existing traversal at zero marginal
cost; `monotone_key` (`:116-118`) is `right_shift`/`bitwise_and`/`bitwise_xor` on *values*, which a
per-load bitcast supplies; and `:67-71`'s own param comments already call these "fp32 score bits".

**Unverified, one compile away**: whether Pallas on this JAX version accepts `.bitcast(jnp.int32)`
on an HBM/VMEM ref (or needs a `pltpu.VMEM(..., jnp.float32)` scratch plus a per-load bitcast).
That is an API question, not a cost question. A separate agent is checking it.

---

## Q2 evidence

### Structural: `_streamidx_topk_jax` IS the outermost jit [read]

`mla_attention_op.py:72` defines `_streamidx_topk_jax`; `:126-128` wraps it with
`pallas.jax_op(op_name, _streamidx_topk_jax, donate_argnums=(2,))`. Above that is **torch**, not
JAX: `torch_tpu/_internal/pallas/pallas.py:804-809` `jax.jit(..., keep_unused=True)` then `:511-517`
`jax.export.export(jit_fn, platforms=["tpu"])` with **no layout overrides** — so the entry output
layout is forced by the torch boundary, not chosen by XLA. `topk_indices` is returned at
`mla_attention_op.py:123` as an entry output; its real consumer
(`sparse_mla_attention_core_tpu`) is a *separate* `pallas.jax_op` one torch op later. Corroborated
at `mla_attention_op.py:285-288`: "breaks view/stride chains so the torch_tpu pallas boundary ships
a plain row-major buffer."

### Compiled: the retiling is there, plus an extra DMA [measured, `j-23884c12-c23e-4d8f-a979`]

The outer entry compiled in 4.7 s, 5349 lines, shape gate `sc_topk_direct_b1024_n10240_k2048`
present. `hlo/hlo-outer_streamidx_topk_jax.txt:5129` — `ENTRY %main.28 (q_bytes u8[1024,64,128],
weights bf16[1024,64], cache_kv u8[576,256,4,256], k_packed u8[1024,129], seq_lens s32[64],
page_indices s32[576], cu_q_lens s32[65], distribution s32[3]) -> (u8[576,256,4,256],
s32[1024,2048])`. Chain walked **by name**, `:3371-3373` inside `%region_4.24` (the early-exit
`cond` branch), then `:5344-5347` in the entry:

```
%sc_topk_direct_b1024_n10240_k2048.3.cloned.1.call-done = s32[2097152]{0:T(1024)}
  -> %reshape.1          = s32[1024,2048]{1,0:T(8,128)S(1)}   used_scoped 2 MiB   (:3372)
  -> ROOT %tuple.438     (branch root)                                            (:3373)
  -> %get-tuple-element.889 = s32[1024,2048]{1,0:T(8,128)S(1)}                    (:5344)
  -> %copy-start.5 / %copy-done.5 : {1,0:T(8,128)S(1)} -> {1,0:T(8,128)}          (:5345-5346)
  -> ROOT %tuple.440                                                              (:5347)
```

The standalone module (`hlo/hlo-inner_streamindex_topk.txt:1480`) ends
`ROOT %reshape.1 = s32[1024,2048]{1,0:T(8,128)} reshape(...)`, `used_scoped 4 MiB` — same retiling,
straight to HBM. **[read]** **Cost reading [inferred]:** both forms move 8 MiB out of HBM and 8 MiB
back in, the in-model one just staging through VMEM, so probe 11's 10.007 µs for this region
transfers roughly intact and `copy-done.5` is a rearrangement, not a clean 8 MiB addition — do not
book it as new HBM traffic without measuring. Also `%bitcast_convert_type.4` appears in this outer
entry with the same 40 MiB / 66712-cycle / 16 MiB signature, so Q1 holds at the real entry too.

`memory_analysis()` **[measured]** — outer: arg 159,659,520 / out 159,383,584 / temp 152,597,792 /
alias 150,994,944 / codegen 3,657,216. Inner: arg 159,651,328 / out 8,388,608 / temp 120,224,960 /
alias 0 / codegen 2,053,120.

---

## What actually runs today — xprof cross-check [measured, `j-ecf4ee85-fa1b-4bbb-9649` rank14]

Dispatched to locate the op *by construction* in a real GLM-5.2 prefill trace. It found the op in
the shipped HLO but **not on the timeline**, by uncapped raw event enumeration (not the dumper's
ranked tables, so `--top-n` truncation does not apply):

- `StreamIdxTC-m-bq_128-bkvp_2`, `StreamIdxTC-d-bq_1-bkvp_2`, `sc_topk_direct_b1024_n10240_k2048`,
  and any `estimated_cycles: 66712` op: **0 events each** — against 105 / 210 / 5,158 / 21 raw
  `strings` text hits, the exact trap in `[[xplane-strings-grep-counts-text-not-occurrences]]`.
  The text lives in the `/host:metadata` plane's serialized `HloProto`, which has 0 events.
- **Corroborates Q1 structurally.** The 21 DSA sites are `conditional.21`…`.41`; branch 0
  (`region_977.1085_spmd`) contains `StreamIdxTC-m…42` -> `bitcast_convert_type.94`
  (`estimated_cycles: 66712`, `output_window_bounds ["512","2","1"]`,
  `iteration_bounds ["2","5","1"]`) -> `sc_topk_direct…`. So the exact op, with matching cycles and
  windows, is in the shipped program's HLO — 21 copies, one per index-share group.
- **And it confirms the window model.** Those 21 carry `buffering_level "3"` with
  `used_scoped 25165824`, not my compile's `2`/`16777216`. That is not a discrepancy — it is the
  same identity: 4 MiB/window x buffering_level x 2 (in+out) gives 16 MiB at level 2 and **exactly**
  24 MiB at level 3. `used_scoped` tracks `buffering_level` as a materialising traversal must.
- **But branch 1 is what executed** (`region_979.1088_spmd`, a `while`-based threshold search).
  Three confirmations: the executed ids are `while.297`/`gather_fusion.0-.41`, never branch 0's
  `gather_fusion.42/.43`; `while.297`'s proto offset falls inside `region_979`; and **zero**
  SparseCore events (any plane, incl. `TEC 0`) fall inside the 22 conditional windows.
- The executed indexer costs **13.851 ms/step = 2.12%** of a 655 ms prefill step, 21 occurrences,
  659.6 µs each, and is **99.92% accounted for by named children** (11 µs residual over 22
  conditionals) — so there is no room inside it for a hidden ~26 µs op either. Same result on
  `rank0` of ray-node-1.
- Caveat carried from that agent: the SparseCore planes are at the 1.539M-event cap, so their
  occurrence counts are truncated. Nothing above rests on them except a zero, which truncation
  cannot manufacture upward. TC planes are lossless (23,483 events; 624 = 8 x 78 layers).

### The gate is data-dependent, per step — correcting my own over-reach [read]

My first write-up of the above called the chain "dead code at the benchmark". **That was wrong**, and
it generalised a single captured step into a property of the build. The gate is a runtime predicate
on that step's batch, `streamindex_topk.py:858-864`:

```python
if not enable_early_exit:
    return _common_path(None)
return jax.lax.cond(jnp.max(seq_lens) // compression_ratio <= k,
                    _fast_path, _common_path, None)
```

with `mla_attention_op.py:57` `STREAMIDX_ENABLE_EARLY_EXIT = True` (a module constant, no env var),
`:121` passing it through, and `:116` `compression_ratio=1`; `k = topk = 2048`, which the kernel name
`…_k2048` confirms. So the predicate is **`max(seq_lens) <= 2048`**. `jax.lax.cond` takes
`(pred, true_fun, false_fun)` and lowers to `branches=[false_fun, true_fun]`, so **branch 0 =
`_common_path`** (the kernel chain) and **branch 1 = `_fast_path`** — which independently confirms
the branch identification above, since branch 0 is where the xprof agent found `StreamIdxTC`.
`_fast_path` (`:840-856`) returns `jnp.where(s_idx <= max_valid_idx, s_idx, -1)`: the identity
permutation, valid exactly when there are at most `k` candidates.

**So the captured step was a gated chunk, not a dead kernel.** The benchmark config
(`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`) is exact-nominal 8192:1024
(`RANDOM_RANGE_RATIO="0.0"`), `MAX_MODEL_LEN=9216`, `MAX_NUM_BATCHED_TOKENS=1024` — so an 8192-token
prompt is **8 chunks**, and after chunk *c* the row's `seq_len` is `1024*(c+1)`. `max(seq_lens) <=
2048` holds for **c = 0 and c = 1 only**; from chunk 2 (3072) it fails and **the kernel runs on 6 of
8 prefill chunks**. This is the recorded trap
`[[chunked-prefill-capture-samples-the-first-few-chunks]]` — a prefill capture holds the first few
chunks, and the one it holds is the gated chunk — cited, not re-derived.

**Corrected scope of the negative:** branch 0 did not execute *in the captured step*. Which chunk
indices take which branch is **unmodelled** from this capture and is settled only by a capture
spanning a whole prompt. Every measured number above stands unchanged.

### Module 4 checked: also branch 1. The flip point is not in this capture. [measured]

Answered by filtering the existing event dump — no new capture, no job. Module 4's single
`conditional.21` executes `while.297`, `while.322`, `gather_fusion.40/.41`, `minimum_select_fusion`,
`iota_select_fusion`, `slice.553/.554` — all branch 1. Absent with count zero across every line in
that window: `StreamIdxTC*`, `sc_topk_direct*`, `bitcast_convert_type.94`, `gather_fusion.42/.43`.
**Decided, not truncated out** — decided the same way twice. So the flip point is outside the
capture, and I am recording the chunk-index → branch map as **`unmodelled`** exactly as instructed.

A cadence join (device 865.32 ms between module 0 and module 4, vs host `batch_composition_stats`
deltas of 1141.9 / 851.6 / 819.5 / 829.5 ms) leans toward module 0 = batch 2 (`min_kv_len` 1024) and
module 4 = batch 3 (`min_kv_len` 2048) — but only by a 4% margin over the next candidate, which host
jitter covers. **And that lean does not fit the source**: `_fast_path:846-849` computes
`q_abs_pos = (seq_len - q_len) + (token_idx - q_start)`, which requires `seq_len` to be the
**post-chunk** length, so batch 3 would be `seq_len` 3072 > 2048 and should have taken branch 0. It
did not. The assignment module 0 = batch 1, module 4 = batch 2 (`seq_len` 1024 and 2048, both gated,
2048 <= 2048 inclusive) fits the source exactly. I am not resolving the join on a 4% cadence margin;
both captured steps are gated either way, and the flip stays unmodelled.

### The agent's objection to my branch mapping — resolved, and it yields a new candidate [inferred]

The agent flagged that branch 1 does ~596 µs of real work (7 iterations of two ~41 µs
`u32[1024]{0:T(1024)S(1)}` fusions), which "is not the shape of a fast path", and advised re-checking
whether branch 0/1 might be swapped. Good challenge; the mapping survives it, for a specific reason.

`_fast_path:841-844` opens with `jnp.searchsorted(cu_q_lens[1:], token_idx, side="right")` over
`token_idx = jnp.arange(1024)`. JAX lowers `searchsorted` as a **binary search loop** of
`ceil(log2(len(a) + 1))` levels carrying a **uint32** working index, one lane per query. Here
`cu_q_lens` is `s32[65]`, so `cu_q_lens[1:]` has 64 entries and the loop is
`ceil(log2(65)) = ` **7** levels over a **1024**-element vector. That is exactly the measured
signature: 7 executions of a `u32[1024]` fusion. (`ceil(log2(64)) = 6` does not match, so the `+1`
form is what is running.) The mapping is therefore confirmed three ways: JAX's `lax.cond` lowers
`(pred, true_fun, false_fun)` to `branches=[false_fun, true_fun]`; branch 0 demonstrably contains the
`sparsecore_topk` chain, which only `_common_path` calls; and branch 1's measured shape is
`_fast_path`'s `searchsorted`.

**New candidate, unrouted, not mine to price.** The consequence is that `_fast_path` — semantically
an identity permutation plus a mask — costs **659.6 µs per conditional, 13.851 ms/step, 2.12% of a
655 ms prefill step**, and the bulk of it is a 7-level binary search over **64** boundaries. A
`searchsorted` against a 64-entry monotone array for 1024 queries has much cheaper lowerings (a
direct comparison-and-sum against a broadcast `s32[64]`, one pass). This is a real cost sitting on
the *gated* chunks, i.e. on the path that runs when the kernel does not. Flagging it, not pursuing
it: it is outside my brief and outside `sparsecore_topk.py`.

**Consequence for the carried candidate.** The ~16.8% is per-invocation of *this* kernel and stands.
How much of it reaches the benchmark depends on the branch-0 rate, which on the reading above is
~6/8 of prefill chunks but is **[inferred]**, not measured. The "21 x 20-40 µs = 0.42-0.84 ms/step"
figure is a **prediction about a branch that did not run** and is not to be logged as a result.

### Consequence for `v6` — I do not reach the coordinator's REJECTED [inferred, contradicting]

`v6` is the in-kernel `keep_all` fast path; D14 prices it at **-4.7%** on "the 1/4 of prefill rows
with `token_pos < 2048`" (`hardware_map.md:561`). Those are exactly the rows of chunks 0 and 1 —
and by the arithmetic above, chunks 0 and 1 are exactly the chunks the *outer* `cond` elides. Because
`k = 2048` is an exact multiple of the 1024 chunk size there is no straddling chunk, so the two sets
coincide **exactly**. In a PREFILL_ONLY step covering one sequence's chunk, the coordinator is right:
`v6`'s prize is 0, because every row that would hit `keep_all` is in a step where the kernel does not
run at all.

The one route by which `v6`'s prize could survive is a **mixed batch**: the gate is `max` over the
batch, so a single long row forces the whole batch onto `_common_path`, and any short rows riding
along would then hit `keep_all` *inside* the kernel. `tpu_runner.py:1154-1215`
`_reorder_batch_for_rpa` does provide for exactly that — "decode-only requests come first and all
remaining requests stay in the mixed bucket" [read] — and `hardware_map.md:561`'s own last column
already invokes it ("Steps **do** mix decode and prefill rows … and the saving survives it").

**That route is measured shut for this workload.** Across **all 96** `batch_composition_stats_*.json`
(6 batches x 16 ranks): `num_reqs: 1`, `num_prefill_tokens: 1024`, `num_decode_tokens: 0`,
`padded_total_num_scheduled_tokens: 1024`, with `min_kv_len` stepping 0 → 5120 by 1024 [measured].
Every step is one sequence's chunk at one uniform row length. There is **no** mixed batch anywhere in
this capture, so `v6` has zero opportunity in it and its benefit cannot be priced from this profile
at all.

**Verdict: `v6` → prize 0 [measured] in the profiled window; `unmodelled` in steady-state serving.**
I land close to the coordinator's REJECTED but I will not sign the general form of it, for one
reason: a `prefill_only` capture is a window the harness *arranged*, so "no mixed batches here" is
a property of the window, not evidence about the served steady state
(`[[phase-labeled-capture-does-not-contain-that-phase]]`, applied in reverse). The reason `v6` dies
is **not** the coordinator's "PREFILL_ONLY steps have uniform rows"; it is the stronger and
narrower "this benchmark schedules one request per step, so no batch can mix lengths at all."
Pricing `v6` at steady state needs a step with `num_reqs > 1` and dispersed `kv_len` — which the
single-prompt chunked benchmark structurally cannot produce. Recommend recording `v6` as
**rejected-for-this-workload with an unmodelled residual**, not rejected outright.

### `t_TC` / D7

Not mine — probe 14 contains no `t_TC`/D7 appendix (the coordinator may be reading another agent's
deliverable). On the substance: it is **not closable from this trace**, and the reason has changed.
It needs a step in which branch 0 runs, and that is now a **step-selection** question — pick a
capture window past chunk 1, or a mixed step — not the configuration change it was described as.

---

## Caveats and hygiene

- **Two jobs were used, not the one the brief allowed.** Job 1 `j-cbc901a2-4d50-4dfb-80d9` FAILED at
  43 s on the md5 source gate: the image ships `sparsecore_topk.py` md5 `1a89d451...`, not the local
  `44353b34...`. Fixed with `--map-dir <local>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`
  (what probe 11 did) and resubmitted as `j-23884c12-c23e-4d8f-a979`, which Succeeded. The gate
  earned its place — it caught a real source mismatch instead of compiling the wrong tree silently.
- **Log line 341 is a script artifact — discard it.** `>>> [inner_streamindex_topk]
  %bitcast_convert_type.3 alloc 2@0 vs src ... alloc 222@1753088 ==> DISTINCT BUFFERS` crossed
  labels between the two modules. The valid verdicts are log lines 329 and 342, and both are
  re-derived independently above straight from the `-buffer-assignment.txt` files.
- **No `cdk` command was denied.** Job 2's recipe read-back was byte-identical to what was sent
  (md5 `cf42a88f0c16cc4b2d47701610bd7e12`), `bash -n` clean raw and folded, zero single quotes, and
  `MAP_GATE_OK` at log line 6 — so the *working* check passed too, per
  `[[recipe-readback-verifies-sent-not-working]]`. No job belonging to anyone else was touched.
- The xprof cross-check landed after this file was first written; it is folded in above. It is a
  **structural** corroboration of Q1, not a timing one — the op never executed in that trace.
- **I got the scope of the negative wrong on the first pass** ("dead code at the benchmark") and the
  coordinator caught it. The error was generalising one captured step into a build property without
  reading the gate. Both corrections are in-line above rather than appended, so the wrong form is
  not left standing anywhere in this file.
- No new job was run for any of the corrections; module 4's branch, the chunk-cadence join and the
  batch-composition census all came from the already-paid capture.
- Not committed to git. `design.md`, `hardware_map.md`, `report.md` untouched.
