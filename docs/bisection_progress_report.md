# DeepSeek-V4 Bisection Debugging Progress Report

**Date:** 2026-07-30
**Goal:** Implement the hidden-state comparison harness, identify the first layer where `vllm-torchtpu` and `tpu-inference` diverge, fix the bug, and verify correct E2E output generation.

**Validation update (2026-07-30, later same day):** Iteration 1 below ("Input
Embeddings" root cause) has been **refuted** — it was a methodology bug in
`compare_hidden_states.py`, not a real PyTorch defect. See §2 "Iteration 2"
for the corrected finding: the divergence is real, but it happens *inside*
Layer 0's own attention/hc-mixing/FFN computation, not at the embedding
lookup. Re-verify anything below this line against fresh output before
trusting it further — don't reuse the old per-iteration conclusions as-is.

## 1. Plan and Setup

- [x] Local workspace configuration (`~/dsv4-comparison`).
- [x] Instrument `patched_layer_forward` in `vllm-torchtpu` to dump output tensors.
- [x] Instrument `DeepseekV4DecoderLayer` in `tpu-inference` to dump output tensors via JAX host callback.
- [x] Deploy instrumented codebases to `alyssa`.
- [x] Deploy comparison script `compare_hidden_states.py`.
- [x] Run `vllm-torchtpu` baseline inference step on `alyssa` to produce reference hidden state dumps.
- [x] Run `tpu-inference` baseline inference step on `alyssa` to produce JAX hidden state dumps.
- [x] Execute `compare_hidden_states.py` to identify the first diverging layer $N$ — original result (input embeddings) refuted 2026-07-30; corrected result is Layer 0's *output*, not its input (see §2 Iteration 2).
- [x] Instrument sub-layer checkpoints inside `_forward_unfused_post_pre` (after hc_pre/attn/hc_post/hc_pre/ffn/hc_post) to localize the Layer 0 divergence to a specific sub-block.
- [x] Debug the isolated sub-block and resolve the all-zero-attention discrepancy (see §2 Iteration 4) — fixed and verified end-to-end (no more crashes).
- [x] A second, narrower numerical bug remains: `sub_02_post_attn` (attention output) is now non-zero but still numerically wrong vs. the reference (see §2 Iteration 4). Localized further to inside the kernel call itself, not projections/RoPE/`_o_proj` (see §2 Iteration 5). **Current step: find the actual kernel/cache bug.**
- [x] Found and fixed the Iteration 5/6 kernel bug (`mla_swa.py` softmax-accumulator `exp_cond`) — verified: first generated token now exactly matches the reference. **Full generation is still garbled** — a *third*, distinct bug remains in CSA/HCA layers' decode-step attention (Iteration 7, below). **Not yet found — this is where the next session should start.** See "HANDOFF FOR NEXT SESSION" at the bottom of this doc.
- [ ] Verify fix and ensure correct English text generation (XLA vs. LLVM comparison in 2 bullet points).

---

## 2. Bisection Iterations

| Iteration | First Divergent Layer | Diverging Component | Root Cause / Fix | Parity Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 (**REFUTED**, see below) | Layer 0 (Input) | Input Embeddings | ~~PyTorch prompt token embeddings at indices 1, 2, and 3 are incorrect and match token 0's embedding (repetition).~~ **This was a comparison-script bug, not a real defect.** | N/A |
| 2 | Layer 0 (Output) | Layer 0's internal computation (attention / hc_pre-hc_post mixing / FFN — not yet narrowed further) | Input to Layer 0 matches perfectly (cos_sim=1.000000 for all 15 real prompt tokens); Layer 0's *output* diverges hard (mean cos_sim 0.767, min 0.52 across tokens, growing monotonically worse through all 43 layers, max abs diff reaching 10^4 by layer 40). Root cause not yet isolated to attention vs. FFN vs. HC-mixing — needs sub-layer instrumentation. | **Diverged, confirmed** |

### Iteration 3: Layer 0 sub-block bisection — exact root cause localized (2026-07-30, later same day)

After fixing a real KV-cache-sizing bug (see below) that was blocking every full-model run
from completing at all, got a clean end-to-end run producing sub-layer dumps at all 6
checkpoints inside `_forward_unfused_post_pre` for Layer 0, prefill step, all 15 real
prompt tokens, both frameworks:

```
sub_01_hc_pre_attn   mean_cos=1.000000 min_cos=1.000000 max_abs_diff=0.000000  OK
sub_02_post_attn     mean_cos=0.000000 min_cos=0.000000 max_abs_diff=13.21     DIVERGE
sub_03_hc_post_attn  mean_cos=0.788512 min_cos=0.636208                       DIVERGE
sub_04_hc_pre_ffn    mean_cos=0.499015 min_cos=0.029891                       DIVERGE
sub_05_post_ffn      mean_cos=0.779293 min_cos=0.457769                       DIVERGE
```

**`self.attn(positions, x2, None)` — the MLA attention call itself — outputs exactly all
zeros** for every one of the 15 real prompt tokens in vllm-torchtpu (mean=std=min=max=0.0,
0/4096 nonzero elements), while tpu-inference's reference output at the identical point is
normal (std≈1.2-2.0, full value range, all nonzero). This is Layer 0, a **plain dense
attention layer** (`compress_ratio=1`) — not HCA/CSA compression, ruling out the
compressed-KV-cache code paths entirely. The bug is specifically in
`VllmDeepseekV4MLAAttention.forward`'s attention computation on the **prefill** step.

Next: determine why the attention kernel returns exactly zero on prefill — likely
candidates: KV cache not yet populated at the point attention reads it, an
all-invalid/empty attention mask on the first forward pass, or a similar prefill-specific
wiring gap (since this is a dense, uncompressed layer, kernel numerics itself are a much
less likely culprit than cache/mask wiring).

### Iteration 4: root cause found and fixed (all-zero attention); a new, narrower numerical bug found downstream (2026-07-30, later same day)

**Root cause of the all-zero attention output, confirmed:**
`_get_active_sw_cache`/`_get_active_main_cache` in
`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py` gated every
candidate cache on `cache.dtype == torch.uint8`. The real, correctly-bound
cache tensor's actual dtype is whatever `kv_cache_dtype` resolves to for the
checkpoint (`torch.float8_e4m3fn` for this run's `fp8_e4m3` config) — never
`torch.uint8` — so the dtype check never matched and both functions always
returned `None`, unconditionally, for every layer, on every step. `forward_mqa`
then fell back to its own zero-page dummy cache
(`torch.zeros((0, 16, 1, head_dim), dtype=torch.uint8)`), and
`_deepseek_v4_attention_jax`'s early-return branch
(`if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0)`)
skipped the real kernel call entirely, returning zeros. This is the direct,
now-verified cause of the "attention outputs exactly zero" finding above, and
of the incoherent repetition-loop generation in §3. Confirmed introduced same-day
in commit `2ac94b211` ("refactor(dsv4): split custom_ops/deepseek_v4.py to
mirror tpu-inference's file structure") — likely stale code carried over from
an earlier project phase when the SWA cache really was uint8 (see
`docs/old_attempts.md` Step 3). **Fix:** dropped the dtype gate; both functions
now only check presence (`is not None`).

**Making that fix land end-to-end required four more real fixes**, because the
SWA-only (`compress_ratio<=1`, i.e. dense/SWA layers) real Pallas kernel call
path had never been exercised with real data before — every bug below was
dormant and silently masked by the all-zero shortcut above, not introduced by
the dtype fix itself:

1. **Degenerate placeholder cache during `profile_run`.** Before
   `initialize_kv_cache()` actually runs, vLLM binds `.kv_cache` to a
   `numel()==0` placeholder rather than leaving it unset. Once the dtype gate
   no longer filtered it out, this placeholder got treated as a real cache,
   and — because reading zero elements is fully constant-foldable — XLA
   silently dropped it from the compiled kernel's operand list, desyncing the
   Pallas kernel from its statically-declared 14-operand signature (Mosaic
   `number of refinements must match number of op operands 10 vs 9`). Fix:
   both `_get_active_*_cache` functions now also require `cache.numel() > 0`.
2. **`None`-valued kernel arguments silently dropped.** For `swa_only` layers,
   `forward_mqa` hardcoded `main_kv_lens`/`main_page_indices`/`main_cu_q_lens`/
   `main_distribution` to Python `None`. A `None` is an empty JAX pytree leaf —
   it never becomes a real compiled operand at all, which (once the cache was
   real, exercising this branch for the first time) further desynced the
   operand count. Fix: reuse the SWA-side metadata tensors instead (same
   underlying batch; these fields are never read for real math in the
   `swa_only` path, only needed as real operands).
3. **Identical-tensor operand deduplication.** `main_cache_kv = sw_cache` (a
   plain alias, same buffer) and the reused SWA metadata tensors above (also
   plain aliases) each made two formal kernel parameters resolve to one
   physical operand; XLA's input deduplication then merged them, again
   desyncing the operand count (`14 vs 13`). Fix: `.clone()` each of these five
   values so every formal parameter is backed by a distinct buffer.
4. **Cache storage dtype vs. kernel contract.** Even with a real, correctly-
   shaped, non-deduplicated cache reaching the kernel, `mla_swa.py`'s
   `mla_sliding_window_ragged_paged_attention` hard-asserts
   `cache_kv.dtype == jnp.uint8` (the cache stores a custom packed byte layout —
   quantized fp8 nope-values + raw rope bits + per-block scales, see
   `quantize_kv_inputs` — not a uniform semantic dtype). This assertion is
   identical in `tpu-inference`'s reference kernel, confirming the kernel's
   uint8 contract is correct and shared, not a bug. The actual bug: this
   class's own `get_kv_cache_spec()` declares `dtype=torch.uint8` in source
   (verified via a construction-time print), but the tensor vLLM actually
   allocates for it is `torch.float8_e4m3fn` (verified via the existing
   `[KV_SIZE_DEBUG]` log) — vLLM's internal KV-cache-spec/allocation pipeline
   does not preserve this per-layer dtype override through to allocation, and
   the exact point where it's lost was not tracked down (multiple vLLM-core
   candidate sites — `unify_hybrid_kv_cache_specs`, `_get_kv_cache_config_packed`,
   `copy_with_new_block_size` — were checked and ruled out by reading their
   source; the actual mutation site is still unidentified). Since
   `torch.float8_e4m3fn` and `torch.uint8` are both exactly 1 byte/element,
   fixed by bitcasting instead of fighting the allocator:
   `cache.view(torch.uint8)` right before returning a real cache from
   `_get_active_*_cache` (same trick already used for MXFP4 weights elsewhere
   in this codebase, `mxfp4.py`'s `loaded_weight.view(torch.uint8)`) — free,
   lossless, no repacking, since the underlying bytes are identical either way.

**Result: the pipeline now runs end-to-end without crashing** — full prefill +
decode to `max_tokens`, no exceptions, for the first time in this entire
investigation with a real (non-dummy, non-zero) cache flowing through the
SWA-only attention path.

**But the output is still wrong** — a new, narrower, purely numerical bug,
confirmed via the same sub-layer bisection methodology as Iteration 3, rerun
with matching prompts (`"Compare XLA and LLVM in two bullet points."`, verified
identical tokenization: `[0, 128803, 43651, 2137, 11318, 305, 33792, 24695, 295,
1234, 37561, 4365, 16, 128804, 128822]` on both sides) and fresh dumps:

```
sub_01_hc_pre_attn   mean_cos=1.000000 min_cos=1.000000 max_abs_diff=0.000000  OK
sub_02_post_attn     mean_cos=0.133476 min_cos=0.000000  max_abs_diff=13.214   DIVERGE
sub_03_hc_post_attn  mean_cos=0.803918 min_cos=0.636208                       DIVERGE
sub_04_hc_pre_ffn    mean_cos=0.537558 min_cos=0.101142                       DIVERGE
sub_05_post_ffn      mean_cos=0.779215 min_cos=0.349473                       DIVERGE
```

Everything up to and including `hc_pre_attn` matches exactly (as in Iteration
3). The attention call itself no longer outputs all-zero, but its output
(`sub_02_post_attn`) is now numerically wrong (near-zero correlation with the
reference, not just "different") for Layer 0 (`compress_ratio=1`, the
`swa_only` path specifically — same layer/path as all four fixes above).
Generated text is still incoherent (garbled "package"-repetition text,
different from the pre-fix repetition loop but still wrong) — see updated §3.

**Not yet done:** the actual kernel-wiring bug causing `sub_02_post_attn` to
diverge. Compared vllm-torchtpu's `forward_mqa` against `tpu-inference`'s
equivalent (`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py:294-402`)
side by side — the `swa_only` field substitutions, `sm_scale`,
`logical_page_size`, and `unnormalized_output` wiring all match; the reference
sources its cache via a JAX-native `wrapper_ctx.kv_caches[index]` lookup table
with no equivalent to compare `_get_active_sw_cache`'s PyTorch-side attribute
search against. The divergence is somewhere in the actual read/write cycle
through the SWA cache (candidates not yet checked: block-table/page-index
correctness, whether the `.view(torch.uint8)` bitcast's byte layout actually
lines up with what `quantize_kv_inputs` produces on write vs. what the kernel
expects on read, or a genuine numerics bug elsewhere in the kernel call).

### Iteration 5: sub_02_post_attn divergence localized to inside the kernel call — projections/RoPE ruled out, kernel/cache read-write cycle implicated (2026-07-30, later same day)

Added two finer-grained dump points inside `attention_impl`/`forward_mqa` in
both `deepseek_v4_attention.py` files (vllm-torchtpu and tpu-inference,
identically named so `scratch/compare_sublayer.py` can diff them directly):
`sub_02a_pre_kernel_q`/`sub_02a_pre_kernel_kv` right after
`qnorm_rope`/`kv_rope` (before `self.attn_op(...)`), and
`sub_02b_post_kernel_output` right after `self.attn_op(...)` returns (before
`_o_proj`). Reran the exact repro command (§5.1/§5.2 below), fresh dumps on
both sides, `Generated text` unchanged (torch still garbled, JAX still
coherent — confirms nothing regressed).

**Shape note (new, worth keeping for future sub-layer comparisons):**
`sub_02a`/`sub_02b` are 3D (`[tokens, n_local_heads, head_dim]`), and the two
frameworks shard the head axis differently — vllm-torchtpu shards heads
across TP ranks (this dump is always rank 0, so its middle axis is only 8
*local* heads out of 64), while tpu-inference uses attention-data-parallelism
instead of head-sharding, so its dump carries the full 64 heads. Verified
empirically which reference heads correspond to rank 0: sliced the
reference's heads `[0:8]` against torch's `[0:8]` → cos_sim≈0.9995 (matches);
every other contiguous 8-head slice (`[8:16]`, `[16:24]`, ...) → cos_sim
0.17-0.43 (no match). Rank 0 owns the reference's first 8 heads, contiguous,
as expected for a standard TP head-shard. `compare_sublayer.py` now slices
the reference down to `t.shape[1]` heads automatically when a 3D shape
mismatch is detected.

**Result — q/kv into the kernel match; the kernel's own output does not:**

```
sub_01_hc_pre_attn   torch=(16,4096)   jax=(128,4096)  mean_cos=1.000000 min_cos=1.000000               OK
sub_02a_pre_kernel_q torch=(16,8,512)  jax=(128,8,512) mean_cos=0.999533 min_cos=0.999388 (tok 0)        OK
sub_02a_pre_kernel_kv torch=(16,512)   jax=(128,512)   mean_cos=0.999898 min_cos=0.999836 (tok 8)        OK
sub_02b_post_kernel_output torch=(16,8,512) jax=(128,8,512) mean_cos=0.171405 min_cos=0.000000 (tok 0)   DIVERGE
sub_02_post_attn     torch=(16,4096)   jax=(128,4096)  mean_cos=0.133476 min_cos=0.000000 (tok 0)        DIVERGE
```

(min_cos for `sub_02a_pre_kernel_*` is 0.9994+, i.e. real PyTorch/JAX numeric
noise — the same tolerance level as `sub_01`'s exact match — not a real
divergence.) **This rules out `wq_b`, `qnorm_rope`, `kv_rope`, and the
indexer/compressor gemms as the bug's location** — everything through RoPE is
correct. The bug is specifically inside `self.attn_op(...)` — Layer 0's
`swa_only` (`compress_ratio=1`) call into
`mla_sliding_window_ragged_paged_attention` (`kernels/deepseek_v4/mla_swa.py`)
— either the kernel's cache read/write cycle or its numerics, not anything
upstream (projections) or downstream (`_o_proj`, confirmed already-wrong by
the time `sub_02_post_attn` is captured, consistent with `sub_02b` already
being wrong one step earlier).

**New, more specific clue found while investigating `sub_02b`'s per-token
values, not yet chased down:** the vllm-torchtpu kernel output isn't
uniformly wrong — it's **exactly zero for prompt tokens 0-11 (12 of 15) and
only non-zero for tokens 12, 13, 14** (the reference is non-zero and
plausible-looking for all 15 tokens throughout). This run's `[DEBUG_DECODE_STEP]`
log for Layer 0's prefill call shows `swa_seq_lens=[15,0,0]`,
`main_cu_q_lens=[0,15,15]` (single request, 15 real tokens, matches
tokenization), and the SWA cache tensor shape logged at
`[DEBUG_FORWARD_MQA_CACHES]` is `sw_cache.shape=[1922, 4, 1, 4, 640]` — page
size (dim 1) is **4** rows/page, so 15 tokens spans exactly **4 pages** (rows
0-3, 4-7, 8-11, 12-14) — meaning **only the last page's tokens (12-14)
produce real output; every earlier page's tokens are exactly zero.** This
looks like a real, specific bug (not generic numeric drift): something in the
kernel's block/page loop is only computing/writing valid attention output for
the final KV page instead of accumulating across all pages a query should
attend to — a classic "forgot to accumulate/mask across kv-block loop
iterations" shape of bug, though not yet confirmed against the kernel source.
**Not yet investigated**: `kernels/deepseek_v4/mla_swa.py`'s block-loop
structure (`num_kv_pages_per_block=1` — one grid step per page — vs. how the
output accumulator carries across those grid steps), diffed directly against
`tpu-inference`'s identical-looking kernel invocation (same file, same
kwargs) to see whether the *kernel* differs or only its *inputs* (block
tables / page indices / `cache_kv`'s `.view(torch.uint8)` byte layout) do.

### Iteration 6: root cause found, fixed, and verified for prefill; a new, distinct decode-only bug uncovered (2026-07-30, later same day)

Read `mla_swa.py`'s `flash_attention` (its FlashAttention-2-style running-softmax update) end to end and traced the exact per-block arithmetic for query token 0 by hand. Root cause, confirmed:

```python
# Before (buggy):
exp_cond = (m_curr == m_prev) | (m_curr <= mask_value)
```

`m_curr == m_prev` is true whenever a later KV block doesn't raise the running max — an **extremely common, completely legitimate** case (e.g. any block that's entirely causally-masked for this query, which is most blocks for an early-position query). In that case `exp_m_diff` should be `exp(m_prev - m_curr) = exp(0) = 1`, correctly preserving the previously-accumulated `l`/`o`. Instead the code forced `exp_m_diff = 0`, **discarding every earlier block's accumulated output and normalizer** each time a later block didn't move the max — for a query whose entire causal range fits inside blocks before the final one, this zeroes the output outright (exactly matching the observed "tokens 0-11 exactly zero, only the last KV page's tokens 12-14 non-zero"). Confirmed the true NaN hazard the comment was guarding against (`m_prev`/`m_curr` both literally `-inf`, no valid entry seen by anyone yet) is fully captured by `m_curr` alone being `-inf` (since `m_curr = max(m_prev, s_rowmax) >= m_prev`, `m_curr == -inf` already implies `m_prev == -inf`). Cross-checked `kernels/deepseek_v4/mla.py` (the non-SWA/compressed-attention kernel) — it does **not** have this bug; its equivalent line only zeroes on `m_curr <= mask_value`, the correct condition, confirming `mla_swa.py`'s extra `m_curr == m_prev` clause was the sole defect (not a pattern that needed fixing in both files this time).

**Fix**: `exp_cond = jnp.isneginf(m_curr)`.

**Verification:**
1. `pytest -sv --use-tpu` on the standard 30-test kernel suite (§2 above) — all 30 still pass, no regression.
2. Full 43-layer real-checkpoint run, same prompt: **the very first generated token is now `'Here'` at `logprob=0.0` (i.e. ~100% model confidence)** — exactly the first word of `tpu-inference`'s coherent reference output (`'Here are two key comparisons between XLA and LLVM...'`), up from complete garbage (`'I understand the nature of thepackage...'`) pre-fix. This is strong, decisive confirmation the root-cause fix is correct for the prefill path.

**New problem, not yet root-caused: every token after the first is degenerate.** Generated token IDs: `[9544, 0, 0, 0, ..., 0]` (48 more steps, all token 0). Every decode step's top-5 logprobs are **exactly identical** (`-10.375` for token IDs 0,1,2,3,4, every single step) — the signature of a perfectly uniform/flat logit distribution (`exp(-10.375) ≈ 3.1e-5 ≈ 1/32000`, matching the tokenizer's vocab size), not a NaN crash (no `[NAN_DETECTION]` fired) and not an EOS stop (`ignore_eos=False` in `examples/run_dsv4_inference.py`, and token 0 isn't the real EOS ID — generation kept running the full 49 steps). Layer 0's per-step debug (`[DEBUG_LAYER_0]`, checked across 3 separate decode steps) shows finite, sane, non-degenerate values (`mean≈0.002-0.012`, `min/max` in the `[-26,25]` range) and correct position/kv-length bookkeeping (`positions=[15,...]`, `swa_seq_lens` growing `16→17→18` step over step) — so the corruption is **not** at layer 0 and **not** a bookkeeping bug; it happens somewhere between layer 0 and the final logits, specifically for decode-shaped (single new query token, growing KV) calls, and was very likely already present but completely undetectable before this session's fix (previously *every* step, including the first, was already garbage, so a decode-specific defect had no clean baseline to show up against). **Not yet investigated**: bisect layer-by-layer specifically on a decode step (not just prefill, which is what all this session's `sub_0X` dumps cover) to find where the degenerate collapse happens; MoE routing/dequant behavior specifically for a real batch size of 1 is an untested edge case worth checking first given `test_dsv4_moe_routing.py`/`test_mxfp4_dsv4_correctness.py` likely exercise larger synthetic batches.

Files modified this session, uncommitted: `kernels/deepseek_v4/mla_swa.py` (the `exp_cond` fix — this is the substantive bug fix, keep as its own commit), plus the `deepseek_v4_attention.py` dump points, `compare_sublayer.py`, and this doc from Iteration 5.

### Also fixed today: a real KV-cache-sizing bug blocking all full-model runs

Unrelated to the Layer 0 investigation, but required to get any working run at all:
`tpu_runner.py`'s `initialize_kv_cache`, for DeepSeek-V4's grouped/shared KV-cache tensors
(`len(shared_by) > 1`, non-Mamba), computed `num_blocks = tensor_size // page_size_bytes`
where `tensor_size` is `KVCacheTensor.size` — but vLLM's own packed allocator
(`_get_kv_cache_config_packed`, aliased as `_get_kv_cache_config_deepseek_v4` in
`vllm/v1/core/kv_cache_utils.py`) sets `size` to the **global total** across every emitted
KVCacheTensor (all groups share one block-strided backing buffer, addressed via
`byte_offset`/`block_stride`), not a per-group size. Dividing that global total by one
group's own page size inflated `num_blocks` ~193x (98790 instead of the correct,
override-respecting `kv_cache_config.num_blocks`), causing a ~15GiB-per-layer allocation
and a `RuntimeBufferAllocationFailure` OOM instead of respecting
`num_gpu_blocks_override`. Fixed by using `kv_cache_config.num_blocks` directly (same value
the single-layer branch already correctly uses) instead of re-deriving it.

### Why Iteration 1 was wrong

The checkpoint's `config.json` has `"hc_mult": 4` — DeepSeek-V4 uses
[Hyper-Connections](https://github.com/deepseek-ai), where every token's
residual stream is expanded into 4 parallel copies right after the embedding
layer (`vllm/models/deepseek_v4/amd/model.py:564`:
`hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)`,
also documented at `model.py:538-541`: *"hidden_states of shape (num_tokens,
hc_mult, hidden_size) — V4 expands the token embedding to hc_mult streams
before the first decoder layer."*). So every dumped hidden-state tensor is
shaped `(num_tokens, hc_mult=4, hidden_size)`, not `(num_blocks,
tokens_per_block, hidden_size)`.

The original `compare_hidden_states.py` (and `check_stride_matching.py`/
`check_stride_decode.py`) treated the middle axis of size 4 as if it were a
KV-cache-style token-block axis, and picked `[:prompt_len, 0, :]` expecting
15 sequential real tokens. Because the HC streams are identical copies of
each other right after embedding (by design — they haven't been mixed yet),
this slicing accidentally sampled the *same* real token repeated across
what looked like "distinct token indices," producing the illusion that
"tokens 1, 2, 3 match token 0." Direct verification against the raw `.npy`
dumps (dim0 = token, dim1 = hc stream, dim2 = hidden, no reshaping) shows
**all 15 real prompt tokens' input embeddings match between `vllm-torchtpu`
and `tpu-inference` with cos_sim = 1.000000**, using plain sequential
indexing — no device-stride correction needed either (§5.1's device-stride
theory was built on the same axis confusion and is not needed: the JAX dump
path already forces `with_sharding_constraint(..., sharding_replicated)`
before saving, i.e. it's already a real, sequentially-ordered, all-gathered
global tensor).

`scratch/compare_hidden_states.py` (at `/home/patemotter_google_com/workspace/scratch/`)
has been rewritten to use the correct axis semantics and to walk all 43
layers instead of stopping at Layer 0. Re-running it against the existing
dumps (prefill step, JAX step 0 vs. Torch step 1) gives:

```
=== Input embedding check (dim0=token, dim1=hc stream, dim2=hidden) ===
  token 0..14: cos_sim=1.000000  (all 15 real prompt tokens match exactly)

Layer  | MeanCosSim  | MinCosSim  | MinTok | MaxAbsDiff  | Status
0      | 0.767006    | 0.519972   | 12     | 7.622932    | DIVERGE
1      | 0.704101    | 0.461410   | 7      | 7.020983    | DIVERGE
...     (monotonically worse through all 43 layers)
42     | 0.221521    | 0.071171   | 13     | 35564.927120 | DIVERGE
```

**This is a real, reproducible divergence** — inputs match bit-for-bit in
cosine-similarity terms, and the very first layer's output does not. This
supersedes Iteration 1 as the actual next investigation target: instrument
*inside* `DeepseekV4DecoderLayer.forward`/`_forward_unfused_post_pre` (both
frameworks already have this exact function, see
`vllm/models/deepseek_v4/amd/model.py:389-408` and `tpu-inference`'s
`deepseek_v4.py:389-408`) to dump hidden states after `hc_pre`, after
`self.attn(...)`, after the first `hc_post`, after the FFN's `hc_pre`, after
`self.ffn(...)`, and after the final `hc_post` — six checkpoints instead of
one — to localize divergence to attention vs. FFN vs. one of the two HC
mixing steps, before touching kernel-level code.

---

## 3. End-to-End Validation

**Update (2026-07-30, after Iteration 4's fixes):** re-run below with matching
prompt/tokenization against a fresh `tpu-inference` baseline. The crash is
gone (full generation to `max_tokens` with no exceptions), but output is still
incoherent — consistent with the still-open `sub_02_post_attn` numerical bug
in Iteration 4, not yet a passing end-to-end result.

- **Prompt:** `"Compare XLA and LLVM in two bullet points."`
- **Reference Output (`tpu-inference`):**
  
```text
  'Here are two key comparisons between XLA and LLVM:\n\n- **Scope and Purpose:** LLVM is a general-purpose, modular compiler infrastructure and toolchain designed to optimize and compile code for a wide variety of programming languages (e.g., C'
  ```
  
*(coherent and correct)*
- **Test Output (`vllm-torchtpu`, post Iteration-4 fixes):**
  
  ```text
'I understand the nature of thepackage com.ibpackage, "spackage",\n  "package.json\n<package> <package 1package",\npackage main\n\nimport\n- * "Ir)package":package com.azurep\n>'
  ```
  
  *(Incoherent, but no longer a crash and no longer the pre-fix repetition
loop — a different symptom of the still-open Iteration 4 numerical bug.)*
- **Coherence Audit:** Still failing on PyTorch, passing on JAX.

---

### Superseded (pre-Iteration-4) result, kept for history

- **Prompt:** `"Compare XLA and LLVM in exactly two bullet points."`
- **Test Output (`vllm-torchtpu`):**
  
```text
  ',Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,Irästyle,'
  ```
  
*(Incoherent repetition loop — this was the all-zero-attention symptom,
  since fixed; see §2 Iteration 4.)*

---

## 4. Timeline of Overnight Work

- **22:48 (Local):** Spawned Custom Owl Orchestrator (`self_owl`, ID: `7868b42a-5e6f-4b4d-abb2-7c92a8334c9c`).
- **22:52 (Local):** Verified JAX callback feasibility script. Verified local instrumented files and copied them to `alyssa`.
- **22:53 (Local):** Cleared remote hidden states dump folders, stopped conflicting container, verified 8 TPU devices available.
- **22:53 (Local):** Launched `vllm-torchtpu` baseline inference run on `alyssa` (background task `task-243`).
- **22:57 (Local):** `vllm-torchtpu` baseline run completed successfully. Hidden state files dumped to `scratch/hidden_states/vllm_torchtpu/` on `alyssa`.
- **23:27 (Local):** Subagent resolved JAX run import issue (ran from `/tmp`) and block size constraint (added `prefix_match_unit=1`).
- **23:35 (Local):** Subagent resolved JAX cache config check by setting `enable_prefix_caching=True`, launched JAX reference run (background task `task-371`), waiting for completion.
- **01:39 (Local):** Subagent completed JAX reference run (task `task-890`) and successfully generated coherent English text comparing XLA and LLVM.
- **03:35 (Local):** Subagent executed the bisection comparison harness and isolated a mathematical divergence starting at Layer 0 step 0.
- **03:36 (Local):** Discovered JAX's block sharding mapping stride (16 blocks in prefill, 8 blocks in decode). Applying this stride confirms high parity (>0.996 CosSim) for all gathered token hidden states.
- **04:40 (Local):** Isolated the root divergence: in PyTorch, the input embeddings for prompt tokens 1, 2, and 3 are incorrect (they replicate token 0's embedding), whereas JAX correctly embeds them. This points to an input-IDs or parallel-embedding initialization bug in the PyTorch pipeline.
- **06:24 (Local):** Subagent patched the AMD model wrapper to print instantiated attention class types, verified it swaps to `VllmDeepseekV4MLAAttention`, and launched PyTorch run 19 (task `task-2380`) with active input ID logging.

---

## 5. Handoff & Replication Commands

All commands should be executed from `/home/patemotter_google_com/workspace` on the remote host `alyssa`.

### 5.1 Run PyTorch Baseline (`vllm-torchtpu`)
Ensure the PyTorch virtualenv is activated and `PYTHONPATH` includes the source checkouts.

```bash
export PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src
cd /tmp
source /home/patemotter_google_com/uv_venv/bin/activate
VLLM_TP_SIZE=8 VLLM_EP_SIZE=8 DSV4_DUMP_HIDDEN_STATES=1 \
  python3 /home/patemotter_google_com/workspace/scratch/run_dsv4_reference.py \
  > /home/patemotter_google_com/workspace/scratch/run_pytorch_latest.log 2>&1
```

### 5.2 Run JAX Native Baseline (`tpu-inference`)
Run using the native JAX virtual environment.

```bash
cd /tmp
source /home/patemotter_google_com/workspace/tpu_inference_venv_setup/vllm_env/bin/activate
DSV4_DUMP_HIDDEN_STATES=1 \
  python3 /home/patemotter_google_com/workspace/scratch/run_dsv4_reference.py \
  > /home/patemotter_google_com/workspace/scratch/run_tpu_inference_latest.log 2>&1
```

### 5.3 Run Bisection Verification Scripts
Run the following scripts from the reference venv to calculate Cosine Similarity between dumps:

```bash
source /home/patemotter_google_com/uv_venv/bin/activate
# Check token permutation/stride mapping in prefill (Step 0)
python3 /home/patemotter_google_com/workspace/scratch/check_stride_matching.py
# Check token permutation/stride mapping in decode (Step 1)
python3 /home/patemotter_google_com/workspace/scratch/check_stride_decode.py
```

These scripts will load the dumped `.npy` arrays from `/home/patemotter_google_com/workspace/scratch/hidden_states/` and compare them layer-by-layer, step-by-step.

---

## 6. Iteration 7 (2026-07-31): two more real bugs found and fixed; a third, distinct bug remains open in CSA/HCA decode

Continuation of Iteration 6. Goal was the same: get real English output for the `"Compare XLA and LLVM in two bullet points."` prompt. Iteration 6's `mla_swa.py` fix was necessary but not sufficient — full generation is **still garbled**. This section documents everything found, fixed, ruled out, and still open, plus exact repro commands, so the next session can resume without re-deriving anything below.

### 6.1 Bug #2 found and fixed: `mla_swa.py` `exp_cond` (this is Iteration 6's fix, restated here for continuity)

`kernels/deepseek_v4/mla_swa.py`'s flash-attention softmax accumulator had:

```python
exp_cond = (m_curr == m_prev) | (m_curr <= mask_value)
```

`m_curr == m_prev` is true whenever a later KV block doesn't raise the running max (the ordinary case for any causally-masked block) — in that case `exp_m_diff` should be `exp(0) = 1` to preserve the accumulator, but the code zeroed it, discarding every earlier block's contribution. **Fixed**: `exp_cond = jnp.isneginf(m_curr)` (the only case that actually risks NaN: no valid entry seen by anyone yet).

**Verified**: all 30 kernel tests pass; the first generated token is now `'Here'` at `logprob=0.0` (100% confidence) — exactly the start of the reference's coherent answer. This was a real, confirmed, high-impact fix. It did not fully resolve generation, though (see below).

### 6.2 Investigation methodology note: several of this session's own diagnostic tools were themselves buggy — read this before trusting any raw output below

This session hit three separate cases where a diagnostic *I* wrote gave a wrong answer, was corrected, and reversed the conclusion. Listed here so the next session doesn't have to rediscover the same traps:

1. **JAX tracing closure bug** in `tpu_inference`'s `deepseek_v4.py` `_dump_tensor`: `step = self.step_idx` was read *outside* the `jax.debug.callback` closure. Since that line runs during Python-level tracing (which only happens once per distinct compiled shape), `step` got frozen as a compile-time constant the first time each shape (prefill-shape, decode-shape) was traced — every subsequent same-shape call silently overwrote the *same* file instead of incrementing. **Fixed** by moving `self.step_idx` access inside the callback body (already fixed in the working tree; see §6.6 below for the file).
2. **`jax.debug.print` / `jax.debug.callback` break this codebase's AOT/export-based persistent compile cache** (`/dev/shm/torch_tpu_cache`) with `NotImplementedError: serialization of host_callbacks is not yet implemented`, for any custom op registered via `pallas_patched.py`'s `pallas.jax_op` (this includes the compressor op and the attention op, i.e. most of the interesting kernels). Plain Python `print()` at the **torch level** (outside any `jax.jit`/export boundary) works fine and was the only reliable way found this session to get value-level diagnostics out. `pl.debug_print` (Pallas's own in-kernel print primitive, see §6.5) does *not* hit the serialization error, but also produced **zero visible output** across 4 attempts including with a cleared compile cache — unresolved, do not assume it works.
3. **torch/XLA buffer donation makes naive before/after `.clone()`-based diffing unreliable.** A `cache_before = orig_cache.clone()` snapshot taken immediately before a donated-buffer JAX call, even with a forced `.cpu().to(device)` "sync", gave two different, mutually-inconsistent answers across two otherwise-identical runs (`n_changed_rows=3` then `n_changed_rows=8` after adding the sync). The fix that actually worked: don't diff at all — directly recompute the expected write slot in torch (mirroring the write-side address formula) and dequantize *that* slot from the function's genuine return value, with no "before" snapshot involved. See `deepseek_v4_compressor.py`'s current `debug` block for the working pattern.
4. **The Tier-2 compile cache at `/dev/shm/torch_tpu_cache/default` is keyed by a config-derived string (`op_name`), not a source hash**, for ops registered via `pallas.jax_op`. Editing the *body* of a function reachable from such an op (e.g. `quantize_fp8_ue8m0`, or adding a `pl.debug_print` inside `mla.py`'s kernel) requires `rm -rf /dev/shm/torch_tpu_cache/default/*` before the next run, or the stale compiled version runs instead with zero indication anything is wrong. Editing plain host-side Python that runs *before* tracing (e.g. `deepseek_v4_attention.py`'s `main_attn_metadata` lookup, which just selects which already-valid tensor to pass in) does **not** need this — confirmed by testing both ways this session.

### 6.3 Bug #3 found and fixed: wrong KV-cache-group metadata used for CSA/HCA attention reads

`deepseek_v4_attention.py`'s `forward_mqa` resolves which vLLM `AttentionMetadata` to use for reading the shared main/compressed-KV cache:

```python
main_attn_metadata = attn_ctx.get(compressor_state_prefix) or attn_ctx.get(main_layer_name) or swa_attn_metadata
```

This tries `compressor_state_prefix` (the compressor's own **dedicated, disjoint** state-cache KV-cache-group) *before* `main_layer_name` (this layer's own attention prefix, which addresses the actual shared main/compressed-KV buffer `main_page_indices` reads from). Traced via `git log -p -L` to the exact introducing commit, `3cd0d62f` (2026-07-23), which changed the original simple `attn_ctx.get(main_layer_name, swa_attn_metadata)` to this ordering. This was harmless *at the time* (state cache and main cache aliased the same physical buffer then), but a later fix (documented in `dsv4_running_and_testing.md`, "session 3") gave the compressor's state cache its own genuinely separate physical buffer — since then, this ordering has silently pointed CSA/HCA layers' attention reads at the *wrong* cache's block table.

**Fixed**: swapped the priority so `main_layer_name` is tried first (matching `tpu-inference`'s reference, which has no `compressor_state_prefix` indirection at all — confirmed by reading it directly).

**Verified as a real, active bug** (not just theoretically wrong) via a debug print (`DSV4_DEBUG_MAIN_META=1`, see §6.6) added right at this line: for the real decode call on layer 2 (`positions=[15,...]`), `main_layer_name`'s block table and `compressor_state_prefix`'s block table are **genuinely different** (`main_bt=[1,0,0,...]` vs `state_bt=[6,7,8,9,...]`) — confirming the old code really was selecting the wrong physical page. Kernel tests still pass after this fix.

**However: this fix did NOT resolve the observed corruption.** Verified via the reliable dump-comparison methodology (§2/§5) that layer 2's decode-step attention output *still* blows up to the same astronomical magnitude (`cos_sim≈-0.03`, `max_abs_diff≈7.4e27`) even after this fix — including when reading from the now-correctly-selected page 1, which was independently confirmed clean (see §6.4). This is a real, legitimate bug fix that should stay, but it is not the (sole, or even primary) cause of the garbled-text symptom.

### 6.4 Ruled out: NaN/Inf in the compressor's cache write

Built a reliable (non-diffing) check: recompute the real decode token's own `kv_slot_mapping` in torch (mirroring the JAX write-side formula exactly), then dequantize *that exact slot* from the compressor op's real return value using the same `unpack_sparse_kv_cache()` helper the passing unit test (`tests/kernels/test_dsv4_compressor.py`) already trusts. Result for the real decode call: `nope_nan_count=0/448`, all values finite. **The compressor's write is clean.** (An earlier "poison bytes" finding from a cruder byte-pattern/diffing check was a false positive caused by trap #3 above — retracted.)

Also directly tested and ruled out, via a tiny standalone script (no vLLM/torch, plain JAX, `source ~/uv_venv/bin/activate && python3 -c "..."`) that casting exactly at `fp8_max` (448.0) correctly produces the valid max-finite byte (`0x7E`), not the NaN pattern (`0x7F`) — so fp8 boundary-rounding-into-NaN is not the mechanism either (this matches session 8's own earlier, independent verification of the same fact).

### 6.5 Still open: the actual corruption mechanism inside CSA/HCA decode attention

**What's established:**
- Layer 2 (CSA, `compress_ratio=4`) is representative; it's the layer used throughout this investigation, but other CSA/HCA layers have not been separately checked and may or may not share the exact same bug.
- Layer 2's decode-step attention *input* (`sub_01_hc_pre_attn_L2`, i.e. the residual stream right before this layer's attention, after the fix in §6.2 item 1 made this measurement reliable) is only mildly diverged from the reference (`cos_sim≈0.97`), consistent with ordinary cross-framework FP noise compounding from earlier layers.
- Layer 2's decode-step attention *output* (`sub_02_post_attn_L2`) is catastrophically diverged (`cos_sim≈-0.03`, `max_abs_diff` in the `1e27`-`1e29` range) — the corruption is introduced *inside* this layer's attention computation, specifically for the CSA/HCA (non-swa-only) kernel path (`kernels/deepseek_v4/mla.py`, not `mla_swa.py`).
- The magnitude (`~1e27`-`1e29`) is suspicious for `exp()` overflow specifically (`exp(62)≈8.4e26`, `exp(67)≈1.3e29`) rather than "wrong but bounded" data corruption — this pointed toward the attention-sinks epilogue in `mla.py` (`exp_attention_sinks = jnp.where((m<=-1e30)|(l_val<=1e-10), 0.0, jnp.exp(attention_sinks - m))`), the only place in this kernel with an unguarded-looking `exp()` outside the already-verified-correct main flash-attention loop.
- A line-by-line diff of `mla.py` against `tpu-inference`'s reference kernel (same technique that found bug #2 in `mla_swa.py`) shows this exact guard exists in vllm-torchtpu but **not** in the reference (reference: plain `jnp.exp(attention_sinks - m_ref[...])`, no guard at all) — structurally analogous to the bug #2 pattern, but **not yet proven to be the actual cause here** (reasoning-only: the `-1e30` threshold is 8 orders of magnitude less extreme than `mask_value` (`-2.4e38`), so on paper it should still correctly separate "never updated" from "any real computed score" — unlike bug #2's threshold, which provably didn't).
- Attempted to get value-level confirmation via `pl.debug_print` (Pallas's native in-kernel print, gated behind `DSV4_DEBUG_MLA_EPILOGUE=1`, added right after this epilogue in `mla.py`) — **produced zero visible output** across 4 attempts (including a `TypeError` fixed by bitcasting floats to int32 for the placeholder format, a `ValueError: Not implemented: bitcast 1D` fixed by reshaping to `(1,1)`, and a full compile-cache clear) with no crash and no output either. Not understood why. Do not assume this infrastructure works without further investigation — it may need a different print mechanism, a different flag, or the code path may genuinely not be reached in the way assumed (e.g. grid-size-zero for the `MlaCase.DECODE` invocation if this request is scheduled into the "mixed" bucket instead — not checked).

**Recommended next steps, roughly in order of expected cost-effectiveness:**
1. First, determine *why* `pl.debug_print` produced no output — this is the fastest path to a definitive answer if it can be gotten working. Sanity-check with a trivial `pl.debug_print("hello")` unconditionally at the very top of `_mla_ragged_paged_attention_kernel` (no gating, no formatting) to confirm the mechanism works *at all* in this environment before trying anything more elaborate. Check whether stdout needs an explicit flush/sync, whether output goes to a different fd, or whether `VLLM_ENFORCE_EAGER=1` + multiprocess workers buffer/swallow it.
2. If that's a dead end, adapt the *working* pattern from §6.4 (recompute-and-dequantize-the-real-value-in-torch, no host callback, no diffing) to inspect `m`/`l`/`acc` — but these live only as Pallas *scratch* refs inside the kernel, not as function inputs/outputs, so this requires either (a) temporarily adding them as extra kernel *outputs* (a real code change to the kernel's output spec, more invasive but guaranteed to work since it's just normal JAX array plumbing, no host callback involved), or (b) finding another way to surface them.
3. Actually test the attention-sinks-guard hypothesis directly and cheaply: temporarily change `mla.py`'s guard to exactly match the reference's un-guarded version (`exp_m_diff = jnp.exp(attention_sinks - m)`, no `jnp.where`) and see if the blowup changes. This needs a compile-cache clear (`rm -rf /dev/shm/torch_tpu_cache/default/*`) since it edits kernel body code (see §6.2 item 4). Cheap to try even without full understanding.
4. If none of the above pans out, reconsider whether the bug is upstream of the epilogue after all — e.g. in the SWA→CSA "unnormalized accumulator handoff" (`swa_accumulation`/`swa_l`/`swa_m`, fed from `mla_swa.py`'s own `unnormalized_output=True` output into `mla.py`'s `_fetch_swa`'s `wait=True` branch) — not yet checked in detail this session, and is architecturally where the SWA and CSA components of a hybrid layer's attention get combined.

### 6.6 Debug instrumentation left in the tree (all inert by default, gated behind env vars)

- **`DSV4_DUMP_HIDDEN_STATES=1`**: the main hidden-state dump mechanism (pre-existing + this session's extensions). Dumps all 43 layers' outputs (`layer{NN}_step{NNN}.npy`), plus sub-layer checkpoints for **layer 0 and layer 2** (extended from layer-0-only this session; look for `sub_01..sub_06[_L2]_step{NNN}.npy`), plus finer q/kv/kernel-output checkpoints for layer 0's attention specifically (`sub_02a_pre_kernel_{q,kv}`, `sub_02b_post_kernel_output`). Works reliably (once the JAX-side closure bug in §6.2 item 1 was fixed). Compare with `python3 scratch/compare_sublayer.py` (also extended this session) or the `compare_hidden_states.py`-style layer-by-layer script in `/home/patemotter_google_com/workspace/scratch/compare_hidden_states.py`.
- **`DSV4_DEBUG_COMPRESSOR=1`**: in `deepseek_v4_compressor.py`, prints `[COMPRESSOR_META_DEBUG]` (metadata/validity-gate values) and `[COMPRESSOR_DEQUANT_DEBUG]` (real dequantized NaN check on the actual write slot) for layer 2's main (non-indexer) compressor calls only. This is the **working, trustworthy** diagnostic pattern from §6.4 — reuse this approach for other value-level checks rather than diffing or host callbacks.
- **`DSV4_DEBUG_MAIN_META=1`**: in `deepseek_v4_attention.py`, prints `[MAIN_META_DEBUG]` — confirms whether `main_layer_name` and `compressor_state_prefix` resolve to the same or different metadata/block-tables, for layer 2 only. Confirmed the bug #3 fix is real and active.
- **`DSV4_DEBUG_MLA_EPILOGUE=1`**: in `mla.py`, a `pl.debug_print` at the attention-sinks epilogue. **Confirmed non-functional this session** (see §6.5) — needs further debugging of the print mechanism itself before it's useful, or should be replaced with the kernel-output-plumbing approach.

---

## 7. Iteration 8 (2026-07-31): bug #4 localized to a single bad compressed-KV row; the §6.5 "attention-sinks epilogue" hypothesis is refuted

Continuation of Iteration 7. **No TPU run was needed to kill the leading hypothesis, and two of §6.5's four next-steps turned out to be unnecessary.** The bug is *not* in `mla.py` at all — that kernel faithfully computes attention over a KV row that is already garbage before it is read.

### 7.1 §6.5's leading hypothesis (attention-sinks `exp()`) is refuted — by construction, not by measurement

`mla.py`'s epilogue (current line ~631) reads:

```python
L = l_val + exp_attention_sinks
L = jnp.maximum(L, 1.0)          # <-- vllm-torchtpu-only, not in the reference
L = broadcast_minor(L, acc.shape)
out = lax.div(acc.astype(jnp.float32), L.astype(jnp.float32)).astype(q_dtype)
```

Because `L` is clamped to `>= 1` *before* the divide, `|out| <= |acc|` **elementwise, always**. The attention-sinks term enters only through `L`, and a larger `L` can only ever *shrink* `out`. Therefore a `~1e27` output cannot originate in the sinks epilogue; the magnitude must already be present in `acc`. (The clamp itself is a benign guard, not a bug: `l >= 1` whenever any valid KV entry was attended, since `m` is the max over valid scores so at least one `exp(s - m)` term equals 1. It only bites in the all-masked case, where it prevents a divide-by-zero.)

This also retires §6.5 next-step #1: **`pl.debug_print` never needed to be unblocked.** It remains non-functional and un-investigated.

`acc` has exactly two sources, both bounded by construction:
- the SWA unnormalized-accumulator handoff (`_fetch_swa`), and
- `acc <- exp_m_diff * acc_prev + pv`, where `exp_m_diff = exp(m_prev - m_curr) <= 1` (since `m_curr = max(m_prev, s_rowmax) >= m_prev`) and `pv = p @ kv` with `p = exp(s - m_curr) <= 1`.

So `|acc| <= |swa_acc| + sum_blocks(bkv_sz * max|kv|)` — the only way to get `1e27` is for `swa_acc` or `kv` itself to be that large. That reduced the search to a single binary question, answered below.

### 7.2 Root cause localized: compressed-KV slot 3 is garbage, and decode is the first step that reads it

New host-side probe `DSV4_DEBUG_MAIN_KV=1` (in `deepseek_v4_attention.py`, immediately before `self.attn_op(...)`) dequantizes the compressed-KV rows layer 2's attention is *about to read*, using the same `unpack_sparse_kv_cache()` helper the passing compressor unit test trusts. Fresh run, layer 2, real decode step (`positions=[15,...]`, `main_kv_lens[0]=4`, `page0=1`, `k_block_size=256`):

| compressed slot | flat row | `nope_absmax` | `rope_absmax` | `scale_max` |
|---|---|---|---|---|
| 0 | 256 | 416 | 2.672 | 0.0039 |
| 1 | 257 | 448 | 3.062 | 0.0078 |
| 2 | 258 | 448 | 2.531 | 0.0078 |
| **3** | **259** | **0** | **1.585e+29** | **3.815e-06** |

**`rope_absmax = 1.585e+29` on slot 3 is the `1e27`-`1e29` blowup.** A huge `kv` row makes `s = q . kv * sm_scale` enormous, which drags `m` up and lets `pv` (and hence `acc`, and hence `out`) reach that magnitude — exactly the observed `max_abs_diff ~ 7.4e27`.

**The probe is self-validating**: slots 0-2 decode to entirely sane values (`nope <= 448`, `rope ~ 2.5-3.1`, `scale ~ 0.004-0.008`) via the *same decoder in the same call*. A broken decoder would have corrupted all four rows, not one. This avoids the §6.2 class of trap where the diagnostic itself was the bug.

**Why prefill/first-token is correct and only decode breaks**: `main_kv_lens = seq_lens // compress_ratio`. During prefill (`seq_len=15`) that is **3**, so slots 0-2 only — slot 3 is never read. At the first decode step (`seq_len=16`) it becomes **4** and slot 3 enters the read range for the first time. This is precisely why the first generated token matches the reference exactly while everything after it is garbled.

`main_kv_lens` itself is **correct** and matches the reference (`main_attn_metadata.seq_lens // self.compress_ratio`, identical formula) — the length bound is not the bug; the row's *contents* are.

### 7.3 Attribution: the compressor wrote that row, with already-bad data

Compressed slot 3 covers positions 12-15 and is finalized by the boundary token at position 15 — which arrives in the decode step. `_boundary_dest` (`compress_norm_rope.py`) gates on `is_boundary = ((positions + 1) % compress_ratio) == 0`; at position 15, `(15+1) % 4 == 0`, so the write **does** fire, and the layer-2 compressor is confirmed to run at `positions[0]=15`.

The write is atomic — `record = concat([nope_fp8, rope_bf16, scale])` then `flat.at[dest].set(record, mode="drop")` — so a *partial* write (nope written, rope not) is structurally impossible. The row therefore was written, with `nope` already all-zero and `rope` already `~1e29` when it was packed.

**Confirmed fresh, both probes armed in the same run** (`scratch/run_slot3_attribution.log`, `DSV4_DEBUG_MAIN_KV=1 DSV4_DEBUG_COMPRESSOR=1`) — the write side and the read side agree on the same row, same step, same signature:

```
[COMPRESSOR_DEQUANT_DEBUG] positions[0]=15 kv_slot0=259 nope_absmax=0.0 scale_min=scale_max=3.814697265625e-06   # from the compressor op's own return value
[MAIN_KV_DEQUANT_DEBUG]    positions=[15,...] slot3(row259): nope_absmax=0 rope_absmax=1.585e+29 scale_max=3.815e-06   # what attention then reads
```

The read-side numbers are also bit-identical to the earlier independent run (`scratch/run_mainkv_probe.log`), so this is reproducible, not a one-off.

So the corruption is produced *upstream of the pack*, inside `compress_norm_rope` / `gather_state_windows`, for this one boundary window at the prefill->decode transition.

### 7.4 Still open: why `compress_norm_rope` produces garbage for the position-15 boundary window

Prime suspect: **`gather_state_windows` reads uninitialized state-cache slots** for the positions 12-15 window at the decode step. Supporting reasoning (not yet proven):
- An all-zero `kv_window` cannot explain it: RMSNorm of zeros is zero, and RoPE of zeros is zero, so `rope` would be `0`, not `1e29`. Garbage (uninitialized) input *can* produce it.
- The compressor's state cache is a genuinely separate buffer from the main/compressed cache (confirmed fresh, see 7.5), addressed by its own block table (`state_bt=[6,7,8,9,...]` vs `main_bt=[1,...]`). A prefill-time state write landing on different slots than the decode-time gather reads would produce exactly this.

Suggested next steps, in order:
1. Mirror `gather_state_windows`' gather addresses in torch for the position-15 call (the proven §6.4 recompute-in-torch pattern) and dequantize the 4 state-cache slots it reads. If any are uninitialized, that is the bug.
2. Check whether position 15's *own* state-cache entry is written **before** the boundary gather in the same op — if the state write is sequenced after the gather, the window is built from 3 valid entries plus 1 uninitialized one.
3. Diff `gather_state_windows` and the state-write path against `tpu-inference`'s reference (the technique that found bugs #2 and #3).

### 7.5 Ruled out this iteration

- **Attention-sinks epilogue `exp()`** — refuted by construction (7.1). §6.5's primary suspect.
- **`pl.debug_print`** — never needed; still non-functional, still un-investigated.
- **`tp_size` double-division**: `mla_swa.py:1149` computes `num_q_heads = num_q_heads_global // tp_size` on a `q` that is *already* the local shard (`_attention_local` runs under `jax.shard_map` with `data_spec = P(None, "model")`). This is a genuine latent bug — the variable name `num_q_heads_global` reflects an incorrect mental model, and the reference kernel has no `tp_size` parameter at all — but it is **inert in this configuration**: `_create_mesh_for_parallelism` (`tpu_runner.py:715`) deliberately builds a single-chip per-worker mesh (`axis_names=("model",)`, size 1), so `tp_size == 1`. Note the kernel tests never pass `tp_size` either, so nothing would catch it if the mesh ever became multi-chip. Worth fixing on its own merits; not this bug.
- **`two_caches_same_buffer`**: hardcoded to `not swa_only` here, whereas the reference derives it (`main_cache_index == swa_cache_index`). Verified benign for layer 2: `model.layers.2.attn.swa_cache` has **no KV-cache group at all** (only layers 0 and 1 do), so `_get_active_sw_cache()` falls through to the layer's own `self.kv_cache` — the same buffer as main. The overlay is intentional and matches the reference's own "main cache and swa cache overlay on the same buffer" path. Hardcoding rather than deriving it is a latent fragility, not an active bug.
- **The compressor state cache still has its own separate buffer** (the "session 3" fix still holds): fresh `[KV_SIZE_DEBUG]` group dump shows `model.layers.2.attn.compressor.state_cache` in group 0 (shared with `model.layers.0.attn`) while `model.layers.2.attn` is in group 3 — distinct allocations.
- **`extra`/topk clamping**: vllm-torchtpu clamps topk indices to `< (positions+1)//compress_ratio` where the reference passes them raw. At position 15 this permits indices 0-3, consistent with `main_kv_lens=4`. It is protective (more conservative than the reference), so it cannot be the source of a blowup — though note it does *not* protect against slot 3 here, because slot 3 is legitimately in range; it is the row's contents that are bad.

### 7.6 Debug instrumentation added this iteration (inert by default)

- **`DSV4_DEBUG_MAIN_KV=1`** — in `deepseek_v4_attention.py`, immediately before `self.attn_op(...)`. Prints `[MAIN_KV_DEQUANT_DEBUG]` (per-slot dequantized `nope`/`rope`/`scale` magnitudes for the compressed rows layer 2 is about to read) and `[CACHE_IDENTITY_DEBUG]` (`data_ptr` comparison of `sw_cache` vs `main_cache_kv`, plus both block tables). This is the probe that found bug #4. Host-side only — **does not require a compile-cache clear**. Reuse this pattern.
- **`DSV4_DEBUG_RETURN_SWA=1`** — in `_attention_local`, returns `swa_output` in place of the MLA output so the existing `sub_02b_post_kernel_output` dump captures the SWA accumulator instead (identical shape/dtype, so no extra kernel outputs or host callbacks are needed). Written as the fallback bisection for "is the blowup in the SWA handoff or in mla.py's own KV read"; **never exercised**, because 7.2 answered that question directly. It is inside the traced function, so it *would* require `rm -rf /dev/shm/torch_tpu_cache/default/*` to take effect.

---

## HANDOFF FOR NEXT SESSION

> **See §8 below — §7.4's framing is superseded.** Slot 3 is *written but empty*
> (`0/448` nope bytes), not "garbage". Several §7 next-steps were chased and
> refuted; read §8 before acting on §7.4.

**One-line status**: bug #4 is localized (not fixed). Layer 2's decode-step attention blowup is caused by **compressed-KV slot 3 (flat row 259) containing `rope ~ 1.585e+29`**, written by the compressor at the position-15 boundary and read by attention for the first time at the first decode step (`main_kv_lens` goes 3 -> 4). `mla.py` is exonerated — the §6.5 attention-sinks hypothesis is refuted by construction (see §7.1), and `pl.debug_print` was never needed. The remaining question is why `compress_norm_rope`/`gather_state_windows` produces garbage for that one boundary window; start at §7.4.

### Exact repro command (copy-paste)

```bash
cd /home/patemotter_google_com/workspace/vllm-torchtpu
source /home/patemotter_google_com/uv_venv/bin/activate
PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
DSV4_MODEL_PATH=/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 \
DSV4_PROMPT="Compare XLA and LLVM in two bullet points." \
VLLM_ENFORCE_EAGER=1 \
GPU_MEMORY_UTILIZATION=0.40 \
DSV4_MAX_NUM_SEQS=8 \
DSV4_NUM_GPU_BLOCKS=512 \
DSV4_MAX_TOKENS=3 \
nohup python examples/run_dsv4_inference.py > /home/patemotter_google_com/workspace/scratch/run_<name>.log 2>&1 &
disown
```

Add `DSV4_DUMP_HIDDEN_STATES=1` for hidden-state dumps, or `DSV4_DEBUG_COMPRESSOR=1` / `DSV4_DEBUG_MAIN_META=1` / `DSV4_DEBUG_MAIN_KV=1` (§7.6 — the probe that found bug #4) / `DSV4_DEBUG_MLA_EPILOGUE=1` for the targeted diagnostics in §6.6 and §7.6.

**Launching the run**: do **not** use `nohup ... & disown` from a tool-driven shell — the worker processes get killed when the wrapper exits (this cost one full run this session, dying partway through weight loading). Run it in the foreground of a backgrounded task instead. When a run dies this way it leaves the stray `/dev/vfio` holders described below. `DSV4_MAX_TOKENS=3` is enough to reach the first real decode step (position 15) where the bug manifests; raise it for a longer/more representative run once actually testing for coherent output.

**Before any run that changes kernel-body code** (not just host-side Python argument selection — see §6.2 item 4 for the distinction): `rm -rf /dev/shm/torch_tpu_cache/default/*` or you will silently test stale code.

**Gotcha**: runs that crash (traceback, `EngineDeadError`, RPC timeout) can leave orphaned `VLLM::Worker_TPx_EPx` processes holding `/dev/vfio/*`, invisible to a plain `pgrep -f run_dsv4_inference` (their process title changes). Before every run: `for i in 0 1 2 3 4 5 6 7; do sudo fuser /dev/vfio/$i; done` — if any PIDs show up, `kill -9` them and wait (`until ! ps -p <pid> >/dev/null 2>&1; do sleep 5; done`) before retrying.

**Reference (JAX) side**, only needed if regenerating `tpu-inference`'s hidden-state dumps (the existing ones at `/home/patemotter_google_com/workspace/scratch/hidden_states/tpu_inference/` are still valid — same reference implementation, unchanged):

```bash
cd /tmp
source /home/patemotter_google_com/workspace/tpu_inference_venv_setup/vllm_env/bin/activate
DSV4_DUMP_HIDDEN_STATES=1 \
  python3 /home/patemotter_google_com/workspace/scratch/run_dsv4_reference.py \
  > /home/patemotter_google_com/workspace/scratch/run_tpu_inference_latest.log 2>&1
```

### Kernel test suite (fast correctness gate, run after any kernel change)

```bash
cd /home/patemotter_google_com/workspace/vllm-torchtpu
docker run --rm --net=host --ipc=host --privileged \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  -w /home/patemotter_google_com/workspace/vllm-torchtpu \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e PJRT_DEVICE=TPU -e JAX_PLATFORMS=tpu,cpu \
  vllm-torchtpu-local:latest \
  pytest -sv --use-tpu \
    tests/kernels/test_dsv4_mla.py tests/kernels/test_dsv4_mla_swa.py \
    tests/kernels/test_dsv4_compressor.py tests/kernels/test_dsv4_moe_routing.py \
    tests/kernels/test_dsv4_streamindex_topk.py \
    tests/layers/vllm/quantization/test_mxfp4_dsv4_correctness.py \
    tests/layers/vllm/quantization/test_mxfp4_tpu.py
```

Currently: **30/30 pass** with all of this session's fixes applied.

### Key files (this session's substantive changes, all uncommitted)
- `src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py` — bug #2 fix (`exp_cond`), real and verified.
- `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py` — bug #3 fix (`main_attn_metadata` ordering) + `DSV4_DEBUG_MAIN_META` diagnostic + this session's sub-layer dump extensions (`_maybe_dump_sublayer`, layer 0's finer q/kv/kernel-output checkpoints).
- `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/deepseek_v4_compressor.py` — `DSV4_DEBUG_COMPRESSOR` diagnostic (the working, non-diffing dequant-check pattern — reuse this approach).
- `src/vllm_torchtpu/kernels/deepseek_v4/mla.py` — `DSV4_DEBUG_MLA_EPILOGUE` diagnostic (not yet working, see §6.5/6.6).
- `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py` — layer-2 sub-checkpoint dump extension (torch side).
- `/home/patemotter_google_com/workspace/tpu-inference/tpu_inference/models/vllm/experimental/deepseek_v4.py` — **separate repo/venv** (`tpu_inference_venv_setup`), the JAX-tracing-closure fix (§6.2 item 1) + layer-2 sub-checkpoint dump extension (JAX side). Also uncommitted.
- `/home/patemotter_google_com/workspace/scratch/compare_sublayer.py` — extended this session to compare the new checkpoints, including a head-sharding shape-mismatch fix (vllm-torchtpu shards attention heads across TP ranks; tpu-inference doesn't — slice the reference down to match).
- `/home/patemotter_google_com/workspace/scratch/debug_compressor_validity.py` — a standalone (no vLLM/TPU-run-needed) script for testing `_deepseek_v4_compressor_jax`'s validity-gating logic in isolation; superseded by the live `DSV4_DEBUG_COMPRESSOR`/`DSV4_DEBUG_MAIN_META` findings but kept as a template for writing similarly fast, isolated repros of other hypotheses (much cheaper than a full TPU run — the user explicitly asked for this pattern and it should be used more).

### Suggested prompt to start the next session

> Continue the DeepSeek-V4 garbled-text investigation in vllm-torchtpu. Read `docs/bisection_progress_report.md` §6 "Iteration 7" in full, especially §6.5 "Still open" and §6.6 "Debug instrumentation left in the tree" — this is a direct continuation, don't re-derive anything already established there.
>
> Status: 3 real bugs already found and fixed this session (verified: 30/30 kernel tests pass, first generated token now exactly matches the reference with 100% confidence). A 4th, distinct bug remains in CSA/HCA layers' decode-step attention (`kernels/deepseek_v4/mla.py`) — layer 2's attention output blows up to ~1e27-1e29 magnitude during decode even though its input and the KV cache it reads from are both confirmed clean. Leading (unproven) suspect: the attention-sinks epilogue's `exp()` call. Start with §6.5's numbered next-steps list, in order — item 1 (get `pl.debug_print` working, starting with a trivial unconditional `pl.debug_print("hello")` sanity check) is the fastest path to a real answer if it can be unblocked.
>
> Use the exact repro command in the "HANDOFF FOR NEXT SESSION" section (§ above) — it bakes in the compile-cache-clearing rule (§6.2 item 4) and the stray-worker-process gotcha, both of which cost real time this session. `git status`/`git diff` in vllm-torchtpu (and separately in `../tpu-inference`) to see the full uncommitted state before touching anything — nothing from this investigation has been committed yet.

---

## 8. Iteration 8b (2026-07-31, later): slot 3 is *empty*, not corrupted; four hypotheses refuted

Supersedes §7.4's framing. §7.2/§7.3's **measurements** all still hold; the *interpretation* ("garbage rope") was wrong.

### 8.1 The decisive observation: raw bytes with a never-written control

The decoded views were repeatedly ambiguous, so the probe was extended to dump raw byte-region populations plus a control row (slot 40, past `kv_len`, never written). Layer 2, decode step:

| slot | nonzero bytes `[nope/rope/scale/pad]` | verdict |
|---|---|---|
| 0, 1, 2 | `448/127/7/0` | healthy |
| **3** | **`0/5/7/0`** | **written, but empty payload** |
| 40 (control) | `0/0/0/0` | never written |

Slot 3's scale bytes are all present (`109`), unlike the control's zeros — so the write lands. Its *payload* is empty. **The compressor had nothing to compress.** This is not data corruption.

`rope_absmax = 1.5845632502852868e+29` is **exactly `2**97`** (bf16 `0x7000`, bytes `00 70`); `0x70` read as fp8_e4m3 is an ordinary `128.0`. An exact power of two is a structured artifact of an essentially-empty record, not random garbage — chasing it as "corrupted RoPE" was a dead end.

**Why only slot 3**: slots 0-2's entire gather window lies inside the single prefill call. Slot 3's window is the only one straddling prefill->decode.

### 8.2 Refuted this iteration (all had looked strong)

1. **`interleaved_rope`'s allocate-then-strided-scatter** (`jnp.empty_like` + two `.at[..., 0::2/1::2].set()`) vs the reference's `stack + reshape`. Reverted to the reference form; output was **bit-identical**, with both compile caches genuinely cleared. A true no-op. The change is kept only because it now matches the reference.
2. **A second writer overwriting the row** (SWA/main buffer overlay). Refuted by adding `rope_absmax` to the compressor's own probe: the value is already `1.5845632502852868e+29` in the compressor op's **return value**, identical to what attention later reads. One writer.
3. **`unpack_state_cache`'s `ndim < 4` silent-zeros fallback.** Does not fire: the buffer is 5-D `[1922, 4, 1, 4, 8192]` and the op squeezes to 4-D.
4. **State not persisting across calls.** Refuted by the corrected probe (§8.3): positions 12-14, written during prefill, read back at `nz~6120` at the decode step.

### 8.3 CRITICAL tooling lesson: the state-cache probe was itself wrong (twice-burned)

The first `[COMPRESSOR_STATE_WINDOW]` probe reported `nz=0` for **every** window position, which looked like a clean confirmation that state never persists. **It was reading unrelated memory.** `slot_mapping` indexes the *state view* (`[num_pages, state_block_size, state_dim]`, 7688 rows); the probe flattened the *raw byte buffer* (`[num_pages, rows, packing, width]`, 30752 rows). Slot 36 should be page 9; it landed on page 2.

Fixed by importing `_state_chunk_dims` from the kernel and mirroring `unpack_state_cache`'s own indexing rather than re-deriving the geometry. **Rule: never hand-derive a paged-cache address in a probe — import and reuse the kernel's own helper.** (The KV probe in §7.2 was trustworthy precisely because healthy slots 0-2 acted as a built-in control; a probe with no control row is not evidence.)

Corrected result:

```
decode (pos0=15): pos12->pg9,off0 nz=6128 | pos13->pg9,off1 nz=6125
                  pos14->pg9,off2 nz=6119 | pos15->pg9,off3 nz=0
next step (16):   pos15->pg9,off3 nz=6125
```

`pos15: nz=0` is **expected**, not a bug: the probe runs after the op but reads the *persistent* buffer, which `state_cache_buf.copy_(...)` only updates afterwards. Position 15 lives in the op's returned `updated_state_cache_buf` — confirmed present, since the next step reads it back.

### 8.4 REFUTED: "one of the four head-blocks is zero" (kept as a worked example of a bad inference)

> **This section's hypothesis is wrong — see §8.7 for the measured result.** The
> `~3/4` ratio is a dtype artifact, not a dead block. Reasoning preserved
> because the failure mode (reading structure into a ratio) recurred repeatedly
> this session.

`nz ~ 6120 / 8192` is almost exactly **3/4**. Each state row is 2048 f32 = **four 512-float head-blocks**, packed by `save_partial_states` as `[kv_a | kv_b | score_a | score_b]`, and split by `gather_state_windows` into `kv_window` (blocks 0/1) and `score_window` (blocks 2/3). Three-quarters nonzero => **exactly one of those four blocks is entirely zero**. `8192 - 2048 = 6144`, against ~6120 observed.

Also note a **window-width error in the earlier probe**: with `overlap=True`, `coff=2`, so the gather window is `coff * compress_ratio = 8`, not 4. Position 15's real window is positions **8-15**; `w_idx < compress_ratio` selects head-block 0 (positions 8-11) and the remainder selects block 1 (positions 12-15). The two halves live in *different state blocks/pages* (8-11 -> `block_table[2]`, page 8). The earlier probe inspected only positions 12-15 — half the window.

If the block feeding half the window is zero, the softmax pool is heavily suppressed, which matches the near-empty record.

**Result: refuted — see §8.7.** No block is dead; the ratio is a bf16-in-f32 storage artifact.

### 8.5 Compile cache: BOTH directories must be cleared

`/dev/shm/torch_tpu_cache/` contains **two** caches: `default` (~1.1 GB) and `tpu_tier2_cache` (~4.7 GB). Earlier docs mention only `default`. Clearing just `default` leaves stale compiled kernels active — a bit-identical result after a real code edit is the tell. Always:

```bash
rm -rf /dev/shm/torch_tpu_cache/default/* /dev/shm/torch_tpu_cache/tpu_tier2_cache/*
```

### 8.6 Reference-alignment changes applied (user: "where needed we can mostly default to reference")

- `interleaved_rope`: output construction **and** even/odd extraction now match the reference exactly (no strided slicing left).
- `mla_swa.py`: **`tp_size` parameter removed entirely** — the reference has no such parameter. It divided an already-`shard_map`-local `q` a second time; inert only because the per-worker mesh is single-chip (`tpu_runner.py:715`). The dead `mesh.shape["model"]` lookup at the call site went with it.
- `quantize_fp8_ue8m0` and `mla_swa.quantize_kv_inputs`: the E8M0 scale byte now **saturates** (`clip(exponent + 127, 0, 254)`) instead of wrapping mod 256. The reference's `.astype(jnp.float8_e8m0fnu)` is unavailable on the pinned libtpu (documented), so this matches its *semantics* rather than blind-reverting. Real latent bug, independent of #4.

Still-open reference deviations, deliberately not yet touched: `mla.py`'s `L = jnp.maximum(L, 1.0)`; `unpack_state_cache`'s `ndim < 4` silent-zeros fallback; `two_caches_same_buffer` hardcoded rather than derived.

### 8.7 Measured: the gather window is COMPLETE and HEALTHY — the state cache is exonerated

Per-head-block nonzero counts over the full 8-wide window, layer 2, decode step (`positions[0]=15`):

```
pos8 ->slot32(pg8,off0): nz=6122  blk[kv_a/kv_b/sc_a/sc_b]=1023/1022/2039/2038
pos9 ->slot33(pg8,off1): nz=6132  blk=1021/1024/2044/2043
pos10->slot34(pg8,off2): nz=6122  blk=1023/1021/2038/2040
pos11->slot35(pg8,off3): nz=6112  blk=1022/1023/2033/2034
pos12->slot36(pg9,off0): nz=6128  blk=1021/1023/2043/2041
pos13->slot37(pg9,off1): nz=6125  blk=1021/1020/2043/2041
pos14->slot38(pg9,off2): nz=6119  blk=1022/1022/2035/2040
pos15->slot39(pg9,off3): nz=0     blk=0/0/0/0      <- expected, see 8.3
```

**No block is zero.** Each block is 512 f32 = 2048 bytes. Score blocks are ~2039/2048 (essentially full); kv blocks are ~1023/2048 (almost exactly half).

The `~1/2` on kv blocks is a **dtype artifact, not missing data**: bf16-precision values stored in f32 containers have their low two mantissa bytes zero, so half the bytes read as zero. kv comes from a bf16 model tensor; `score + ape[ape_rows]` is a genuine f32 add and fills all four bytes. Hence `(2*1024 + 2*2048)/8192 = 74.7%` — matching §8.4's observed ratio exactly, with no dead block involved.

**Conclusion: the state cache, its persistence, its addressing, and the gathered window are all correct.** All seven persisted window positions (8-14) are populated, and position 15 is supplied in-op by `save_partial_states` (confirmed: the next step reads it back at `nz=6125`). `compress_norm_rope` therefore receives a complete, healthy 8-wide window and still emits an essentially-empty record.

### 8.8 Next step: offline repro of `compress_norm_rope` (no TPU required)

The bug is now confined to `compress_norm_rope` itself (softmax pool -> RMSNorm -> RoPE) or to which record ends up scattered by `_boundary_dest`/`.at[dest].set()`. Everything upstream is measured-good.

`compress_norm_rope` is a **pure JAX function**, and every one of its inputs is reachable host-side:
- `kv_window` / `score_window` / `valid_mask` — reconstructable from the state-cache rows dumped above (use `_state_chunk_dims` + mirror `gather_state_windows`; do **not** hand-derive addresses, see §8.3).
- `rms_weight` = `self.norm.weight`, `cos_sin_cache` = `rotary_emb.cos_sin_cache`, `compressed_pos = (15//4)*4 = 12`.

Run it on CPU and compare against the TPU-produced record:
- **Empty on CPU too** -> logic bug, debuggable offline for free, no TPU runs.
- **Healthy on CPU** -> TPU/Mosaic lowering bug in this function; bisect by lowering pieces individually.

This is the cheap-isolated-repro pattern (cf. `scratch/debug_compressor_validity.py`) and is strongly preferred over more ~8-minute full-model runs — each of which has, this session, mostly served to refute a hypothesis that a 30-second CPU script could have refuted.

Secondary check if the offline repro comes back healthy: confirm the scattered `record[0]` really is token 0's, i.e. that `num_valid_tokens`/`slot_mapping` masking is not letting a padding token (`positions=[15,1,2,3,4,5,6,7]`; note position 3 also satisfies `((p+1) % 4) == 0`) win the `.at[dest].set()` for row 259.

---

## 9. ROOT CAUSE FOUND AND FIXED (2026-07-31): compressor caches passed as fp8 instead of uint8

**Supersedes §7 and §8's framing entirely.** Slot 3 was never "empty" or "garbage" in any interesting way — it was collateral damage from a dtype bug that corrupted *every* record the compressor ever wrote.

### 9.1 The bug

vLLM allocates the KV / compressor-state caches as `kv_cache_dtype`, i.e.
**`torch.float8_e4m3fn`**. But every compressor kernel treats them as raw
**byte buffers**:
- `compress_norm_rope_store` packs `[fp8 nope | bf16 rope | e8m0 scale]` into a `uint8` record and scatters it with `flat.at[dest].set(record)`.
- `pack_state_cache` writes f32 state via `_to_byte_lane(...)`, also `uint8`.

A JAX `.at[...].set(uint8_values)` into an **fp8-typed** array performs a
**numeric conversion, not a bitcast**. Verified directly:

```
set uint8 [108,119,112,109] into float8_e4m3fn array -> reads back [110,111,110,110]
set uint8 [108,119,112,109] into uint8            array -> reads back [108,119,112,109]
```

Byte 108 becomes the fp8 *number* 108.0 — a different bit pattern. The packed byte layout was destroyed on every write.

`deepseek_v4_compressor.py`'s `forward` passed `self.k_cache.kv_cache` and `self.state_cache.kv_cache` **raw** into the op. The attention path never hit this because it routes through `_as_kernel_cache_view()`, which applies exactly the missing `.view(torch.uint8)`.

### 9.2 Causal chain (every link measured)

1. Compressor writes packed byte records -> numerically cast into fp8 storage -> bit patterns destroyed.
2. State cache decodes back as **~7.87e31**.
3. RMSNorm's `variance = mean(x^2)` ~ 6e63 -> **overflows float32 to inf**.
4. `rsqrt(inf) = 0` -> `normed = 0` -> **`compressed` all zeros**.
5. `quantize_fp8_ue8m0(zeros)` -> empty record (scale byte 105), scattered over the healthy row 259.
6. Layer 2 decode attention reads it -> ~1e29 blowup -> garbled generation.

### 9.3 Why it hid for so long

- Layer 2's **SWA cache is the same tensor** as its compressed-KV cache, and `forward_mqa` does `orig_sw_cache.copy_(new_sw_cache)` over the whole buffer after attention. The SWA kernel (which *does* get a correct uint8 view) kept overwriting the compressor's damage. Slots 0-2 therefore always looked healthy by the time anything inspected them.
- Row 259 was simply the one row read in the window between corruption and masking — making a global bug look like a single anomalous row.
- **Every isolated test passed** because each allocated its own `uint8` array: `kv_score` healthy; `gather_state_windows` correct; `compress_norm_rope_store` byte-identical and healthy on **CPU and TPU**; `pack`/`unpack` round-trip lossless. The defect lived only in the dtype of the buffer they were handed in production.

### 9.4 The fix

`deepseek_v4_compressor.py` `forward`: view both caches as `uint8` before the op (mirroring `_as_kernel_cache_view`). `.view()` shares storage, so copy-back still writes the bound cache memory.

### 9.5 Verification

| | before | after |
|---|---|---|
| gather window absmax (pos 15) | `7.87e+31` | `3.736` |
| `compressed` nope / rope | `0 / 0` | `2.396 / 3.351` |
| quantised nonzero | `0/448` | `448/448` |
| scale bytes | `[105,...]` | `[120,120,120,120,120,120,120]` |
| slot 3 (row 259) | `nope 0, rope 1.585e+29, nz 0/5/7` | `nope 320, rope 3.344, nz 448/128/7` |
| generation | `[9544, 0, 0, 0, ...]` | fluent English, 32 tokens |

Generated text after the fix:
> `Here are two bullet points comparing the two given entities:\n\n**Entity 1: "The quick brown fox jumps over the fox""\nEntity 21`

**Kernel suite: 27/27 pass** (`test_dsv4_mla`, `test_dsv4_mla_swa`, `test_dsv4_compressor`, `test_dsv4_moe_routing`, `test_dsv4_streamindex_topk`). The 3 `test_mxfp4_*` failures are **pre-existing** — `TypeError: FusedMoEConfig.__init__() missing 'intermediate_size'`, a vLLM-API/test-harness mismatch — confirmed identical with the fix stashed (A/B via `git stash`).

### 9.6 Still open

Output is now fluent but **not yet correct**: it answers about "entities" and "the quick brown fox" rather than XLA vs LLVM. The catastrophic failure is fixed; a semantic-accuracy gap remains. Since this bug corrupted every CSA layer's compressed KV for the whole run, the remaining divergence should be re-bisected from scratch against `tpu-inference` — **prior per-layer divergence measurements in this document were all taken with corrupt compressed KV and should not be trusted.**

Also worth revisiting now that records are real: `mla.py`'s `L = jnp.maximum(L, 1.0)`, `unpack_state_cache`'s `ndim < 4` silent-zeros fallback, and `two_caches_same_buffer` being hardcoded rather than derived (§8.6).
