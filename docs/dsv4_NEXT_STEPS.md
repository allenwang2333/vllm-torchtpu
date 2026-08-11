# DSv4 — in-flight state, 2026-08-11 evening

Read `dsv4_status.md` first for background. This file is only the things that
were still moving at the end of the day.

## 0. RESULT (2026-08-11, late): the compressor fix is a NO-OP

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

```python
state_metadata.input_positions,     # state cache
k_cache_metadata.input_positions,   # compressed-KV cache
```

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

## 2. Three runs were in flight. How to read them

```bash
grep -E "^\[SWEEP\]" scratchpad/fix_test.log     # XLA prompt, 150 tok
grep -E "^\[SWEEP\]" scratchpad/fix_needle.log   # needle, 16 tok
grep -E "^\[SWEEP\]" scratchpad/fix_pool.log     # 10-prompt pool, 120 tok
```

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
