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

## Status, 2026-08-16 evening

G5 fixed and gated. G1 dropped. G2/G3/G4 untouched.

| item | state |
|---|---|
| G5 — KV allocation | **done** (`36abf661`), util 0.55 -> 0.75 |
| G1 — MHC `register_oot` | **dropped**, misattributed to the reviewer |
| G2 + G3 — inline head, split module | open, one gate, low risk |
| G4 — move attention into the model package | deferred, raise with gxd3 first |
| D1 — submit the nine-commit stack | **open and ungated** |

### G5 — fixed 2026-08-16 (`36abf661`)

Root cause was not the allocation but the overlay plan: host selection used one
global `taken` set, so with 21 compressed-KV layers only 21 of 43 sliding-window
caches could overlay and the other 22 took standalone `num_blocks` arrays —
1.3223x the budget vLLM sized. Cache groups have disjoint block-id namespaces,
so one array hosts one layer per group, which is what the reference does.
Result: overlay 42 of 43, ratio 0.9839, gate byte-identical, ceiling 9216
passes, and utilization 0.55 -> 0.75 (KV 309,223 -> 485,612 tokens).

"Fold `_is_ds_v4` into the shared tail" turned out to be inapplicable: that path
computes `num_blocks = tensor_size // total_group_page_size` per tensor, but
under vLLM's packed layout `tensor_size` is the whole slab. The TODO at
`tpu_runner.py:5364` stands as tidiness only.

Full measurement in the tracker.

### Superseded: the original G5 plan

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

### G1 — dropped 2026-08-16

G1 claimed `model.py:147`'s `HCHeadOp.forward_native = ...` was "the pattern
gxd3 asked us to leave on PR #462". Verified against the threads: it is not.
gxd3's #462 comment asks for a native `model.py` plus a registry entry, which is
D1 and is already implemented. No comment on #459, #462 or #464 mentions MHC,
`forward_native` or `register_oot`.

The difference from the reference is real but cosmetic and gates nothing. See
the tracker's "G1 was dropped" section for what the reviewer did ask for, and
for the one finding worth keeping: our HC-head body is higher precision than the
reference's, so a future port must move ours rather than copy theirs.

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

**Nothing gates it.** The earlier advice to "send the stack once G1 lands"
rested on the misattribution corrected above: D1 *is* what gxd3 asked for on
PR #462, and G1 is not something he raised. G2/G3 are layout cleanups that can
follow.

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
