# DeepSeek-V4 design-parity plan

> **Written 2026-08-16 20:00 UTC.** Companion to `dsv4_tracker_20260816.md`
> (evidence, run log, refuted claims) and `dsv4_next_session_20260816.md`
> (pickup prompt). Where any older doc disagrees, this and the tracker win.

## Objective

Accuracy parity with tpu-inference's DSv4, expressed in the structure `kimi_k3`
already uses in this repo. Accuracy parity is **met** — MMLU-Pro 0.7821 ± 0.0243
against the reference's 0.7786 ± 0.0154. What remains is structural.

Not in scope: byte-level parity with tpu-inference, its memory characteristics,
DP attention, or the perf gap against it.

## The five items

Ordered by value, not by size. G5 is first because it is the only one that costs
serving capacity; the rest cost idiomaticity and review friction.

### G5 — fold DSv4 KV allocation into the shared path

*Why it matters:* DSv4 cannot exceed ~0.83 `gpu-memory-utilization` while every
other model on this hardware runs 0.92–0.95. That is lost KV capacity on every
deployment, and it has been mistaken for a property of the model.

*What is wrong:* `_initialize_ds_v4_kv_cache` (`tpu_runner.py:5169-5258`) takes
`num_blocks` from the profiler and allocates
`get_kv_cache_shape(num_blocks, spec.storage_block_size, …)` without checking
that the result fits the `kv_cache_tensor.size` the profiler budgeted. The
shared path at `:5375-5405` does exactly that reconciliation, using
`PallasAttentionBackend.get_kv_cache_page_size_bytes` and noting in a comment
that `spec.real_page_size_bytes` is 2× off on fp8. `_is_ds_v4` is the only
model-specific allocation branch left in the runner.

*Steps:*

1. **Measure before changing anything.** Log, per `kv_cache_tensor`,
   `num_blocks × actual page bytes` against `kv_cache_tensor.size`. If they
   disagree, that difference is the bug and its size predicts the utilization
   ceiling. This step alone may be the whole diagnosis.
2. Reconcile DSv4's page-size accounting the way the shared path does, so
   `num_blocks` and the allocation agree.
3. Collapse the `_is_ds_v4` branch into the shared tail, per the existing TODO
   at `tpu_runner.py:5364`. The overlay planning (`_plan_ds_v4_swa_overlay`,
   `_plan_ds_v4_auxiliary_overlay`) stays DSv4-specific; only the allocation
   tail is shared.

*Verify:* compiled gate for correctness, then raise
`gpu-memory-utilization` stepwise and record the highest value that starts. The
target is parity with the house standard, 0.92.

*Risk:* highest of the five. It touches KV geometry, which has a long history of
subtle breakage on this model. Do not attempt it in the same change as anything
else.

### G1 — replace the MHC monkey-patch with `register_oot`

*Why it matters:* `model.py:147` does
`HCHeadOp.forward_native = mhc_hc_head_native`. Patching a vLLM class is the
pattern gxd3 asked us to leave on PR #462, and it is still present in the model
we would submit as the fix for that review.

*Not a mistake to have reimplemented:* vLLM's `HCHeadOp.forward_native` raises
`NotImplementedError`; only `forward_cuda` and `forward_xpu` exist. The
arithmetic has to come from us. The reference simply registers it properly.

*Steps:*

1. Add `src/vllm_torchtpu/layers/vllm/custom_ops/mhc.py`, mirroring
   `tpu_inference/layers/vllm/custom_ops/mhc.py` (131 lines): four subclasses,
   `@MHCPreOp.register_oot`, `@MHCPostOp.register_oot`, `@HCHeadOp.register_oot`,
   `@MHCFusedPostPreOp.register_oot`, each defining `forward_tpu`.
2. Move the bodies of `mhc_pre_torch`, `mhc_post_torch`, `mhc_hc_head_native`
   (`model.py:57-146`) into those methods verbatim — no numerics change.
3. Delete the patch line at `model.py:147` and the three free functions.
4. Hold `self.mhc_pre = MHCPreOp()` / `self.mhc_post = MHCPostOp()` on the
   decoder layer and call them, as the reference does at `deepseek_v4.py:368-370`.
5. Import the new module where the platform registers custom ops —
   `platforms/tpu_platform.py:366` notes that `register_oot` only takes effect
   once the module is imported.

*Verify:* compiled gate. Generations should be byte-identical to
`gate_09f6c694c.sub`; a diff means the bodies did not transfer cleanly.

*Precedent in this repo:* `gdn_attention_op.py`, `mla_attention_op.py` and
`vision_attention.py` already use `register_oot`. DSv4's MHC is the outlier.

### G2 + G3 — inline the head, split the module

Do these together: both restructure `model.py`, and pairing them costs one
re-gate instead of two.

*G2:* the reference has no `Head` class — `DeepseekV4Model.forward` calls
`hc_head_op(...)` then `norm(...)` inline at `deepseek_v4.py:656-664`. Ours
wraps that in `DeepseekV4Head` (`model.py:418`). Inline it and delete the class.

*G3:* split the remaining 795-line `model.py` to match `kimi_k3`:

| target | contents |
|---|---|
| `layers.py` | `DeepseekV4MLP`, and the `mhc_*` helpers if G1 has not already moved them |
| `moe.py` | `DeepseekV4MoE` |
| `model.py` | `DeepseekV4DecoderLayer`, `DeepseekV4Model`, `DeepseekV4ForCausalLM`, `_make_deepseek_v4_weights_mapper` |

*Verify:* compiled gate, generations byte-identical. These are pure moves; any
drift means something was not a pure move.

*Sequencing:* do G1 first. It removes the `mhc_*` functions from `model.py`
entirely, which makes the split smaller.

### G4 — move attention into the model package

*Why it matters, weakly:* `kimi_k3` keeps `attention.py` inside its model
package; DSv4's lives in `layers/vllm/custom_ops/deepseek_v4/`. Cosmetic, and
the largest diff of the five.

*Recommendation:* defer. It buys layout consistency and nothing else, it touches
the most files, and the custom-ops location is defensible — the compressor and
indexer live there too, and they are genuinely custom ops rather than model
code. Raise it with gxd3 before doing it rather than deciding unilaterally.

## D1 — submitting the stack

Nine commits, `ff1033a2..HEAD`, absent from `origin/main`. This is the refactor
gxd3 asked for on PR #462 and it has never been sent. No longer blocked: A8 was
refuted, and PR11 was dropped as a consideration on 2026-08-16.

**The open decision is ordering**, and it is Pate's:

- *Submit now.* Gets the largest piece of promised work in front of a reviewer
  sooner, at the cost of reviewing code we already intend to restructure — G1 in
  particular is a pattern that reviewer specifically objected to.
- *Submit after G1–G3.* Sends something already idiomatic, at the cost of
  further delay on work that has been outstanding since 2026-08-14.

A reasonable middle: send the stack once G1 lands, since G1 is the one gxd3
would notice, and treat G2/G3 as follow-ups.

## Verification standard

Every item above needs the compiled gate:

```bash
setsid nohup scratchpad/gate_dsv4_compile.sh . <tag> > scratchpad/<tag>_wrapper.log 2>&1 &
```

Pass is `tracebacks=0 sweeps=13 raises=0`, needle exact (`7391-QUARTZ-4408`),
and generations diffed against `scratchpad/gate_09f6c694c.sub` — **not**
`gate_cache_compile.sub`, which predates the requantizer removal.

Commit before gating, and tag the run after the commit. Today's runs shared a
SHA but differed in working tree, which made the SHA meaningless.

The gate runs at `DSV4_MAX_MODEL_LEN=1024` and therefore proves nothing about
long context. Two separate failures on 2026-08-16 lived entirely above that
size. For anything touching compilation or memory, also run the ceiling probe:

```bash
setsid nohup scratchpad/probe_compile_ceiling.sh 9216 > scratchpad/<tag>_wrapper.log 2>&1 &
```

Accuracy only needs re-checking if a change moves weights. MMLU-Pro at
`--limit 20` is ~30 min compiled; the bar is 0.78 ± 0.03.
"""
