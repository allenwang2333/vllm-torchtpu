# DSv4 — in-flight state, 2026-08-11 evening

Read `dsv4_status.md` first for background. This file is only the things that
were still moving at the end of the day.

## -1. REFRAMING (2026-08-11, late night): step 13 is NOT a near-tie

The "small numerical difference meets an early coin-flip" framing in
`dsv4_status.md` §8 is wrong. It was based only on OUR distribution at step
13 (' language' -0.25 vs ' compiler' -1.50, gap 1.25). The reference's
distribution at the same step was sitting unread in
`scratchpad/runAM_ref_logprobs.log`:

    ref step 13:  ' compiler' -0.0000516 (rank 1)     <- probability ~1.0
                  ','         -11.0      (rank 2)
    (' language' is not in the ref's top-5, i.e. below -12.9)

So on an IDENTICAL 30-token context, our distribution deviates from the
reference's by >12 logits for our picked token. That is a large, real
distribution shift — far beyond any rounding-noise explanation. The XLA
prompt is a fine instrument after all; nothing about this is luck.

Shape of the divergence over the shared prefix (both trees generated
identical tokens for steps 0-12; full table from
`scratchpad/{fix_test,runAM_ref_logprobs}.log`):

    steps 0-8:   |ours - ref| <= 0.02   (prefill-built state: healthy)
    steps 9-12:  0.12 .. 0.35
    step 13:     ' compiler' ours -1.5 vs ref -0.00005; ours puts a
                 token the ref rates < -12.9 at -0.25

Two more observations from the same logs:

- Our logprobs are quantized to 0.125 steps (bf16-looking); the reference's
  are smooth fp32-looking values. Our logits reach the sampler at lower
  precision. Not necessarily the bug, but a real pipeline difference.
- ' language' after "a domain-specific" is exactly what a model picks when
  it cannot see the compiler/LLVM context — same "generic prior wins over
  actual context" signature as the instruction-bleed failure mode across the
  pool. At seq 31 the 128-token SWA window covers the entire sequence, so
  the prompt rows are physically present in the SWA cache; suspicion falls
  on decode-written cache rows (SWA and/or compressed) polluting attention,
  with error growing as decode-written rows accumulate: 0.02 at step 0,
  0.35 at step 9, blown at step 13.

### In flight: teacher-forcing chain (`scratchpad/chain_teacher_forcing.sh`)

Running as of this writing; ~1.5 h total. Legs:

1. **ours, growing-prefix scoring** of the ref's 59 XLA tokens and of our
   own degenerate 150 tokens (`tf_ours.log`). Our runner does not support
   `prompt_logprobs` (tpu_runner.py:3430 hardcodes None), hence one prefill
   per step. THE readout: if a fresh prefill of the step-13 context scores
   ' compiler' ~0, our prefill path is healthy and the bug is decode-built
   cache state — proven within-tree, no DP-vs-TP confound.
2. **ref, prompt_logprobs scoring** of the same two sequences
   (`tf_ref.log`). Instrument control (how much prefill-vs-decode disparity
   is normal) + does the ref find our degenerate loop plausible.
3. **ref 10-prompt pool rerun** (`pool_ref.log`) — the owed systematicity
   answer.

Analysis: `python3 scratchpad/analyze_tf.py` once legs finish. Harness
support added: `DSV4_TF_FILES` in `examples/run_dsv4_inference.py` (ours,
growing prefix) and `run_dsv4_needle_ref.py` (ref, prompt_logprobs); token
files `scratchpad/tf_xla_{ref,ours}.json` extracted from the logs above.

### Static diff of the attention module (done, mostly exonerating)

`deepseek_v4_attention.py` vs the reference's, every deviation checked:

- `kv_rope`/`qnorm_rope` fp32 casts: **non-deviation.**
  `DeepseekV4ScalingRotaryEmbedding` keeps `cos_sin_cache` fp32 and
  bf16*fp32 promotes to fp32 anyway; output cast back to bf16 either way.
- `indexer_rotary_emb` fallback to `rotary_emb`: **non-deviation.** Upstream
  vLLM sets `self.indexer_rotary_emb = self.rotary_emb` (attention.py:249)
  — same object.
- Metadata fallback chains (`attn_ctx.get(...) or ...` vs the reference's
  hard `attn_metadata[name]`): probe shows `chosen_is_main=True` (warmup
  only, but the dict keys are static per trace). Low suspicion.
- `_o_proj` einsum layout ours `(g,r,d)` vs ref `(d,g,r)`: layouts differ
  because the storage differs (our requant); byte-identical short
  generations prove neither is scrambled. Only rounding differs (ref:
  fp8-weight einsum with fp32 accumulate then per-channel scale; ours:
  dequant to bf16 then bf16 einsum).
- Remaining true deviations, all "tiny rounding" class — too small to
  explain a >12-logit shift on their own.

## 0. RESULT (2026-08-11, late): compressor fix is a NO-OP, and safe

Regression check passed: needle still returns `7391-QUARTZ-4408`, byte-identical
to the reference. So the change is safe to keep, and it is committed
(`6f0cc201`). It is **not** a fix -- see below.

Tested on the XLA prompt. Output is **byte-identical** to the pre-fix run
(same md5 over the generated text). So `state_metadata.input_positions` and
`k_cache_metadata.input_positions` already carry the same values as the
model-level `positions` in our stack, and threading them through changes
nothing.

The divergence from the reference is real *structurally* -- the reference
addresses each cache with its own metadata, we used one vector for both -- but
it is not the cause of the generation bug. Keep the change (it matches the
reference and costs nothing) but do not count it as a fix.

### Attention sinks: checked, probably fine

Ours shards `attention_sinks` on the head axis (`get_valid_spec(mesh,
"model")`, and `attn_head_axis == "model"` so both spec sites agree); the
reference replicates it with an explicit comment, "`attention_sinks` is
replicated. The kernels run per-shard."

That difference is explained by parallelism, not by a bug: the reference runs
attention **DP=8** (every rank holds all heads, so replicated is correct),
while we run attention **TP** (heads sharded, so a head-axis shard is
correct). Worth a positive confirmation that our shard actually lines up with
the head split each rank owns, but it is not obviously wrong.

**Next target: the attention module's metadata construction**
(`deepseek_v4_attention.py`, 1705 lines, no reference test). The kernels are
proven equivalent, so if the kernels are right and the output is wrong, what
they are being *fed* is where the bug is. Worth checking first: the attention
sinks, and how the SWA/main split and page indices are built per decode step.

## 1. The fix that was in the working tree (now known inert)

`src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/deepseek_v4_compressor.py`
is modified and deliberately **not committed**. A copy is saved at
`scratchpad/PENDING_compressor_position_fix.patch`.

    revert:   git checkout -- src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/deepseek_v4_compressor.py
    reapply:  git apply scratchpad/PENDING_compressor_position_fix.patch

### What it changes

The reference addresses each cache with **its own metadata's positions**:

    state_metadata.input_positions,     # state cache
    k_cache_metadata.input_positions,   # compressed-KV cache

We used a single model-level `positions` for both — `input_positions` appeared
**zero times** in our compressor — and compensated with a
`% max_num_state_blocks_per_req` that the reference does not have.

The state cache and the compressed-KV cache are separate kv-cache groups with
separate block tables and, under a sliding-window spec, separate position
numbering. One position vector for both writes state to whatever slot that
arithmetic lands on, and the error grows with position — which matches
degradation tracking generated-token count (see §3).

Our own code already made this argument for `k_block_tables` ("must come from
k_cache's own metadata"). It was never applied to positions.

The fix threads both through: metadata extraction, custom-op signature,
`shard_map` in_specs, inner function, and the fake impl. All four arities
verified aligned at 13. Falls back to `positions` if a metadata object lacks
`input_positions`, so it cannot hard-fail. The modulo is kept as a guard — a
no-op when positions are already request-local.

## 2. Run results (resolved)

| run | result |
|---|---|
| XLA prompt, 150 tok | unchanged, byte-identical md5 -- still loops |
| needle, 16 tok | `7391-QUARTZ-4408` -- intact, no regression |
| 10-prompt pool, 120 tok | byte-identical to pre-fix (md5 53a4376c35bd both) -- inert across all 10 |

Original decision table, kept for reference:

    grep -E "^\[SWEEP\]" scratchpad/fix_test.log     # XLA prompt, 150 tok
    grep -E "^\[SWEEP\]" scratchpad/fix_needle.log   # needle, 16 tok
    grep -E "^\[SWEEP\]" scratchpad/fix_pool.log     # 10-prompt pool, 120 tok

| needle | XLA | verdict |
|---|---|---|
| still `7391-QUARTZ-4408` | clean sentence | fix works — commit it, then run MMLU-Pro for a systematic number |
| **broken** | anything | revert. Position semantics are not what was assumed |
| still `7391-QUARTZ-4408` | still loops | a real divergence, correctly fixed, but not the cause. Keep it; go to attention metadata next |

The needle result is the one that matters most: it is currently the only
byte-identical-to-reference result, and this change touches exactly the cache
addressing it depends on.

## 3. What was established today (verified, safe to build on)

- **All four kernels are equivalent to the reference.** `mla`, `mla_swa`,
  `compressor`, `streamindex_topk` all pass tpu-inference's own correctness
  suites when their test modules are pointed at our implementations, with the
  reference kernel as a passing control. `mla_swa` matched to 5.17e-26 —
  identical to the reference's own figure. Even `streamindex_topk`, which has
  665 differing lines. Run it yourself in seconds:

      PYTHONPATH=src:$TPU_INFERENCE_ROOT:$TPU_INFERENCE_ROOT/tests \
        python tools/dsv4_debug/diff_kernels.py

  This is the highest-leverage tool built today — it replaces 20-minute model
  runs with seconds for any kernel question.

- **The compressed main branch contributes correctly.** `DSV4_FORCE_SWA_ONLY=1`
  makes output *worse* ("XLA (Accelerated Learning) … 'The Lottery' by Shirley
  Jackson"), so the long-range branch is doing real work and is not the
  corruption source.

- **Degradation tracks generated-token count, not any prompt property.** Every
  clean result is a short generation (capital-of-France ~7 tokens at four
  different prompt alignments; needle at 2 and 16 tokens). Every degraded one
  is 100+ tokens. This points at decode-time accumulation.

- **Prompt-length alignment is NOT the trigger.** The same question at
  `prompt_tokens % 4` = 0, 1, 2, 3 is clean in all four cases
  (`tools/dsv4_debug/prompts/align/`). The earlier correlation was difficulty
  confounded with alignment.

- **The failure mode is instruction bleed**, not generic repetition: the model
  emits prompt/instruction-shaped text mid-generation ("Use simple language.",
  "in your own words.", "? Provide a short example.") then restarts after
  `</think>`. Nine of ten pool prompts.

## 4. If the fix does not pan out

Next target is the attention module's own metadata construction
(`deepseek_v4_attention.py`, 1705 lines, no reference test). Specifically how
`main_kv_lens`, page indices and the SWA/main split are built per step — the
kernels consume that metadata correctly, so if the kernels are right and the
output is wrong, the metadata is where to look.

Note `main_kv_lens = m_seq_lens // compress_ratio` matches the reference
exactly, so that line is not it.

## 5. Owed regardless

- MMLU-Pro has never been run. It aggregates over thousands of prompts, so it
  catches a systematic gap that single prompts can't — but it is multiple
  choice and will not reproduce the long-generation failure.
- The dumper materialization fix (`_dump_to_numpy`) is still unverified. Ride
  it along on the next run that uses `DSV4_DUMP_HIDDEN_STATES=1` rather than
  spending a dedicated run.
- The reference leg of the 10-prompt pool never completed (killed by a path bug
  since fixed). Rerunning it gives the systematicity answer: whether the other
  nine prompts diverge at steps where the reference was confident, like the XLA
  prompt did at step 13.
