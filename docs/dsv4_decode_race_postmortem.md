# DSv4 decode-corruption root cause: the untracked-mutation race

2026-08-12. Companion to `dsv4_NEXT_STEPS.md` (chronological log) and
`dsv4_status.md` (background). This doc is the durable statement of the root
cause, the fix options, and the prevention tooling. Verification status of
the intervention is tracked in `dsv4_NEXT_STEPS.md`.

## The bug

At decode, `attn_gemm`'s

```python
kv_score = hidden_states @ compressor.fused_wkv_wgate.weight.T
```

materializes against a **mutated** `hidden_states` buffer, while
`fused_wqa_wkv(hidden_states)` two lines earlier materializes against the
correct one. The two consumers differ because:

- the attention path's output is needed by the step's logits, so the lazy
  engine materializes it early;
- `kv_score`'s only consumer is the compressor's **cache write** — a pure
  side effect nothing in the step's output depends on — so its
  materialization is legally deferred;
- in between, an untracked in-place write (fused-MHC stream buffer reuse;
  exact site not yet pinned) overwrites the buffer `hidden_states` aliases.
  The lazy engine tracks no dependency for it, so the deferred branch reads
  the new bytes.

Every decode step therefore writes compressor state (and hence compressed-KV
rows) computed from wrong data. Long-range context rots row by row;
generation falls back to generic completions (the "domain-specific
*language*" flip, instruction bleed, loops). Prefill's larger graphs happen
to schedule safely, which is why prefill was bit-deterministic all along and
why the pure-JAX reference (SSA, no aliasing) cannot express this bug class.

## Evidence chain (each link measured)

1. Teacher forcing: our prefill state scores the reference's tokens
   correctly at every step; our decode state is 12-18 logits off at
   specific steps. Reference shows no prefill/decode disparity.
2. Page dumps: all metadata correct at 149/149 decode steps; SWA rows
   byte-identical decode-vs-prefill; decode-written state rows 30-70% off,
   compressed slots off proportional to decode-state fraction per window.
3. Synced dumps: kv_score already polluted at the op boundary.
4. Batched vs single-row matmul: bit-identical (kernel honest).
5. Offline arbiter: `dumped_hidden @ checkpoint_W` == polluted runtime
   kv_score at cos 0.999999, while the sibling projection of the same
   variable is byte-perfect => two snapshots of one buffer.

## Fixes, in tiers

1. **Diagnostic (committed, `16de88ae`)**: `DSV4_KVSCORE_SYNC=1` forces
   `kv_score` to materialize at program order via
   `torch_tpu._internal.sync.synchronize`. Proves the mechanism by
   intervention; costs one host sync per CSA/HCA layer per step. Not for
   production.
2. **Ordering anchor (the intended fix)**: give the side-effect branch a
   tracked dependency instead of a host sync. Options, in preference order:
   a. Compute `kv_score` inside an op boundary that the step's output
      consumes, or fold the matmul into the compressor's own `pallas.jax_op`
      with `hidden_states` as an input — the op's tracing snapshots the
      value.
   b. Thread a token of `kv_score` (e.g., a 0-weighted scalar) into the
      attention output so the scheduler sees a consumer. Ugly but minimal.
3. **Structural**: eliminate the untracked in-place mutation itself — find
   the fused-MHC stream write that recycles the buffer and make it write to
   a fresh tensor (or a properly functionalized buffer). Required regardless,
   because tier 2 protects only the consumers we know about.

**Audit list (same shape, must be checked/fixed alongside):**

- `attn_gemm`'s indexer path:
  `hidden_states @ indexer.compressor.fused_wkv_wgate.weight.T` — identical
  bare-matmul-into-cache-side-effect structure.
- Any other tensor whose only consumer is a cache write. Current known set:
  compressor kv_score, indexer kv_score, indexer_weights (check whether the
  indexer's topk output is consumed by the step; if yes it is anchored).

## Prevention tooling

| tool | status | what it catches |
|---|---|---|
| `tools/dsv4_debug/chain_prefill_decode.sh` + `diff_teacher_forcing.py` | promoted from scratchpad | ANY decode-vs-prefill behavioral divergence, zero instrumentation, 2 model loads |
| `tools/dsv4_debug/chain_cache_diff.sh` + `diff_cache_pages.py` | promoted from scratchpad | byte-level decode-vs-prefill divergence in written cache state, single model load; prefill is bit-deterministic so byte-equality is a legitimate assertion |
| `DSV4_TF_FILES` (examples/run_dsv4_inference.py) | committed | growing-prefix scoring of any token sequence through our stack (our runner lacks prompt_logprobs) |
| `DSV4_DUMP_PAGES` (+`DSV4_TF_AFTER_PROMPTS`) | committed | block-table-indexed page dumps + synced activation reads, one-load decode/prefill differentials |
| side-effect-consumer audit | manual, list above | future untracked-ordering hazards |
| torch-tpu debug mode: version-stamp aliased buffers, assert on stale reads | to file upstream | this entire bug class, deterministically |

## Process lessons (cheapest first next time)

1. When a symptom offers a behavioral A/B (prefill/decode, eager/compiled),
   run it BEFORE any internal instrumentation.
2. Never characterize a divergence from one side's logprobs; diff both
   stacks' distributions (`diff_logprobs.py`).
3. "Host reads of lazy tensors must be synchronized" generalizes to: any
   read without a tracked dependency can cross an untracked mutation —
   including device-side consumers. All unsynced reads lie; no exceptions
   for "cache tensors seemed fine."
4. Side-effect-only computations are the unique places with no natural
   ordering anchor; audit them whenever code is ported from a functional
   (JAX) reference into the mutable-lazy runtime.

## Intervention log

**#1 — DSV4_KVSCORE_SYNC=1 (16de88ae): FAILED, byte-inert.** Output
byte-identical to baseline (md5 ab92f8f5). In retrospect the v4 evidence
predicted this: the synced dump of hidden inside attn_gemm was ALREADY
polluted, so the mutation lands before those lines run — a sync there pins
the already-dirty value. The failure narrows the timing: the buffer is
mutated between fused_wqa_wkv's value capture and the very next python
line. Needle x2 unchanged (pass), as expected.

**Refined mechanism:** something concurrent to the host thread mutates the
buffer — and this stack runs vLLM **async scheduling** (the log announces
it), which preps step s+1's inputs while step s's deferred graph branches
are unmaterialized. Composed with a memory pool that recycles step-s
intermediates, the deferred side-effect branch reads recycled bytes.

**#2 — DSV4_ASYNC_SCHEDULING=0 (3d7ca7a0): queued.** Same four
instruments. If clean: mechanism = async prep racing deferred reads; the
tiered fixes in this doc still apply (the branch needs an ordering anchor
regardless — async scheduling is a legitimate feature we want back).
If still broken: the concurrency source is inside torch-tpu's own
execution (allocator reuse without host-visible async), and the next probe
is dumping kv_score computed at the TOP of attn_gemm (before
fused_wqa_wkv's dispatch can flush anything).

**#2 — DSV4_ASYNC_SCHEDULING=0: FAILED, byte-identical to baseline (md5
ab92f8f5).** Confirmed "Asynchronous scheduling is disabled" in the log.
Combined with #1, the corruption is **deterministic** — two interventions
that radically change timing produced bit-identical outputs. Not a race:
the mutation is ORDERED between the two consumers' value capture, and the
only code between those adjacent lines is fused_wqa_wkv's own dispatch.
Refined suspect: **input-buffer donation/reuse by the quantized-linear
custom op** — the op consumes the true value (byte-perfect SWA rows), then
its dispatch clobbers the buffer for every later reader (the bare matmuls
AND all synced dumps, which were always taken after it and always read
polluted bytes — consistent with every observation including both failed
interventions).

**#3 — DSV4_KVSCORE_FIRST=1 (4cd1123e): queued.** Computes kv_score and
the indexer projections BEFORE fused_wqa_wkv. If generation flips
prompt-faithful, this is simultaneously the mechanism proof and the
minimal fix; the principled follow-up is fixing the donation in the
quantized-linear op path (and the same audit for every other custom-op
call site whose input is later re-read).

**#3 — DSV4_KVSCORE_FIRST=1 (4cd1123e): FAILED, output unchanged** (all
four instruments; caveat: like #1 and unlike #2, no positive in-log
confirmation the flag was active — future interventions must log an
active-marker). Three byte-identical nulls under radically different
orderings eliminate the entire "later mutation" family including
input-buffer donation by fused_wqa_wkv.

**Surviving fork, split by measurement not intervention
(DSV4_DUMP_HIDDEN_FULL, run in flight):** compare the FULL synced hidden
at attn_gemm entry, decode row vs prefill row for the same token.
- Rows equal => the pollution is manufactured between attn_gemm entry and
  the kv_score consumer in a way reorders cannot touch — which would
  indict the synced-dump/off-op read path itself and demand a torch-tpu
  runtime-level investigation.
- Rows differ => the retracted "decode forward path differs" finding was
  right; the SWA byte-identity is the anomaly to explain (per-row
  normalization + fp8 absorption?), and the hunt moves up to layers 0-1
  outputs with the same full-dump instrument.

## RESOLUTION OF THE FORK (04:50): the retraction was wrong — decode hidden genuinely differs

Full synced hidden dumps at attn_gemm entry (`hiddenfull` run):

```
decode step 0 row vs prefill row, same token/history/caches:
    cos 0.928, mean |diff| 0.0147   (pure prefill-built caches!)
by step 9: cos 0.580
prefill self-check: bit-identical (max diff 0.0)
```

The SWA byte-identity that forced the earlier retraction is explained:
**fp8 quantization absorbs percent-level input differences** — byte-equal
cache rows never proved equal inputs. The kv_score/state path preserved
the difference only because state stores raw f32. All three intervention
nulls are consistent: there was never a mutation race; the two "snapshots"
were always the same (diverged) value, and the "clean" sibling was clean
only after quantization.

**Standing model:** the decode forward diverges from prefill starting in
layers 0-1's attention reads (layer 0 pre-attention is bit-exact; the
designed fp8-cache-vs-bf16-in-batch K/V asymmetry is the seed), and the
divergence is AMPLIFIED across steps (0.93 -> 0.58 cos by step 9) as each
decode-written row freezes it into the caches. The reference shares the
same kernels and the same asymmetry but stays within 0.3 logprob — so the
defect is OUR AMPLIFICATION, somewhere in layers 0-1 / MHC mixing /
epsilon-vs-reference differences that a healthy loop damps.

**Next experiment (single, well-defined):** the same per-layer hidden-row
decode-vs-prefill measurement ON THE REFERENCE stack, then compare
divergence-vs-layer and divergence-vs-step curves between the stacks. Where
ours departs from theirs is the defect's address. Secondary: the same
measurement at layers 0 and 1 outputs in our stack narrows it to one
sublayer before touching the reference at all.

Corrections to earlier sections of this doc: the "untracked-mutation race"
framing is retired; interventions 1-3 and their nulls are the evidence.
The tier-2/tier-3 fixes aimed at ordering are moot; the audit list and the
prevention tooling remain valid and used.

## Injection point found (05:30): LAYER 0's block

Per-layer hidden dumps (`layercurve` run, layers 0-4, decode step s row vs
TF-control row for the same token):

```
L0 input: BIT-EQUAL at every step (max diff exactly 0)
L1 input (= L0's output): cos 0.94 at decode step 0
L2: 0.93   L3: 0.86   L4: 0.69   (step 0 -- depth-compounding,
no step accumulation needed)
```

Layer 0 (SWA-only attention + hash MoE + MHC) turns a bit-identical input
into a ~3%-divergent output at decode, and every subsequent layer adds
more. Candidate per-layer injection mechanism -- decode's single-query
kernel path + fp8 cache reads vs prefill's ragged path + in-batch bf16 K/V
-- is SHARED with the reference (same kernels), so the discriminating
measurement is the reference's own curve: instrumented (mirror probe in
tpu-inference's deepseek_v4_attention.py forward + DSV4_REF_SELF_TF in
run_dsv4_needle_ref.py) and running.

- Ref curve similar => this channel is normal-and-tolerated; the true
  defect lies elsewhere and our TF-logprob agreement becomes the thing to
  re-examine.
- Ref curve flat (~0.999) => our layer block injects 10-30x more per-layer
  noise than the identical-architecture reference block -- diff our layer
  compute (MHC mixing epsilons, norm placement, hash-MoE dispatch) against
  theirs with this exact number as the target.

## QUANTIFIED (05:45): our layer-0 block injects ~15-220x the reference's divergence

Reference layer curve (identical probe via jax.debug.callback, same
checkpoint, decode step 0): L0 bit-equal; L1-input cos 0.99973 (mean diff
0.0007); L2 0.99914; L3 0.99048; L4 0.99157. The decode-vs-prefill
asymmetry is inherent (shared kernels) but SMALL and non-compounding in
the reference.

Ours at the same points: L1 0.94016 (mean 0.011 -- 15x mean error, ~220x
in 1-cos), then compounding to 0.685 by L4. Same bit-equal input, byte-
equal kernels: the excess is manufactured in OUR layer-0 block between the
kernel boundary and the block output (attention invocation glue, hash-MoE,
or MHC mixing).

Next split (in flight): layer-0 raw attention output decode-vs-control
via DSV4_DUMP_PAGES attn_output at DSV4_DEBUG_LAYER=0.
- attn out ~0.001 divergent, block out 0.011 => amplifier is our MoE/MHC
  glue in the block.
- attn out already 0.011 => our swa_only merge/sinks/scale invocation
  deviates from the reference's despite equal inputs.

## ROOT CAUSE #1 PROVEN AND FIXED BY INTERVENTION (06:45): stale padding rows

DSV4_ZERO_PAD=1 (92f379ee, active-marker confirmed) — first intervention of
four to change output at all, and it changed the diagnostic token:

- xla_llvm: step 13 now ' compiler', coherent sentence; tracks the
  reference through step 16; first flip at 17 on a token the reference
  itself rates ambiguous (gap 2.25). diff_logprobs verdict: "no confident
  divergence."
- needle 81 + needle 256: retrieval intact.
- Mechanism chain, all measured: runner reuses decode input buffers
  without clearing (stale positions/ids visible) -> padding rows carry
  full-magnitude stale data (junk q rows |x| 0.70 vs real 0.59) -> the
  shared attention kernel is padding-SENSITIVE (zeroing pads changes real
  output; suites never catch it because tests pad with zeros) -> ~50%
  attention-output error at layer 0 -> compounds across 43 layers ->
  context blindness by ~step 13. The reference is immune because its pads
  are zeros (its own decode-vs-prefill curve: cos 0.9997 at L1 vs our
  0.94).

Fix tiers: (1) call-site mask = DSV4_ZERO_PAD, proven; (2) runner buffer
hygiene at batch assembly (the real fix; covers all consumers); (3) kernel
padding-insensitivity upstream (CPU-interpreter repro with garbage pads
recommended for the tpu-inference bug report).

## ROOT CAUSE #2 (separate, still open): water_cycle step-8 class

05_water_cycle still flips at step 8 with the reference confident (gap
6.25); ours@ref moved only -1.25 -> -1.12 with pads zeroed — the padding
fix is irrelevant to it. Residual repetition/instruction-bleed in long
tails likely belongs to this. ~5-logit error at a fixed 22-token context,
reproducible, instrumented: hunt with the exact playbook that found #1
(teacher-force this prompt, layer curve, etc.). Indexer/CSA-selection
inert at this length, so it is none of the previously exonerated paths.
