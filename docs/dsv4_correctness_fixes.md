# DeepSeek-V4 TPU Integration: Garbage-Token Root Causes & Fixes

**Branch:** `patemotter/dsv4-correctness-fixes` (commit `83d244e4`, based on `patemotter/dsv4-integration` @ `a587bb39`)
**Date:** 2026-07-24
**Symptom:** Model compiles and generates on v7x-8 (EP=8, TP=8), but output tokens are garbage. Unit tests for individual kernels (MLA, SWA, compressor, MoE GMM, StreamIndex TopK) all pass.

The diagnosis compared the TPU integration against the reference implementation in
vLLM v0.25.1 (`vllm/models/deepseek_v4/`, especially the ROCm/AMD variant the TPU
port is modeled on). Because the kernels test clean in isolation, every root cause
found is in the **wiring between components**, not inside a kernel.

---

## Fix 1 (critical): RoPE was a complete no-op in the attention layer

**File:** `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`
(`qnorm_rope`, `kv_rope`, `_o_proj`)

### What was wrong

DeepSeek-V4 uses `DeepseekV4ScalingRotaryEmbedding`, which — unlike vLLM's standard
rope classes — applies rotation to the **last** `rotary_dim` (64) channels of
whatever tensor you pass it:

```python
# vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
query_rot = query[..., -self.rotary_dim:]   # trailing channels
query_pass = query[..., :-self.rotary_dim]
...
query = torch.cat((query_pass, query_rot), dim=-1)
```

Commit `a587bb39` ("rotate trailing channels on TPU fallback") replaced previously
correct direct calls with a zero-padding trick that assumed **leading**-channel
semantics:

```python
q_padded = torch.cat([q_rot, torch.zeros_like(q_nope)], dim=-1)  # [rope | zeros]
q_padded_rotated, _ = rotary_emb(positions, q_padded)
q_rot_rotated = q_padded_rotated[..., :rot_dim]
```

Trace it through the V4 rope class: the module rotates the *last* 64 channels of
`q_padded` — which are the **zeros**. The actual rope values sit in the leading
channels and pass through untouched as `query_pass`. Taking `[..., :rot_dim]` then
returns the **unrotated** input. The same trick was used in all three call sites, so:

- q was never rotated (`qnorm_rope`)
- the new KV token was never rotated before cache insert (`kv_rope`)
- the inverse-RoPE on the attention output was a no-op (`_o_proj`)

Meanwhile the compressor kernel (`kernels/deepseek_v4/compress_norm_rope.py`)
*does* write properly rotated K into the compressed (C4A/C128A) cache. Result:
unrotated q attending against rotated K on every compressed-attention layer — a
guaranteed scramble, with no NaNs and all shapes correct.

Two independent confirmations:

1. `tests/layers/vllm/test_dsv4_rope.py` itself validates the **direct-call**
   semantics (`forward_native(positions, query)` rotating the trailing 64).
2. `_o_proj` passes `inverse=True`, which only the V4 rope class accepts — if the
   model had a standard rope, that call would have raised a `TypeError`. Since
   generation ran, the V4 class (trailing-channel semantics) is definitively active.

### The fix

Revert to direct calls (what the code did before `a587bb39`):

```python
# qnorm_rope: per-head unweighted RMSNorm, then
q_rotated, _ = self.mla_attn.rotary_emb(positions, qf)

# kv_rope: unsqueeze a head dim so cos/sin broadcast
kv_rotated, _ = self.mla_attn.rotary_emb(positions, kv.unsqueeze(1).to(torch.float32))

# _o_proj: inverse GPT-J rope on trailing channels of each head
o_ref, _ = self.mla_attn.rotary_emb(positions, o_f, inverse=True)
```

This mirrors the reference: the ROCm `_o_proj` uses `rocm_inv_rope_einsum` →
`_fused_inverse_rope_gptj`, which is documented as "inverse GPT-J RoPE on the
trailing rope_dim of each (token, head)".

---

## Fix 2 (critical if `num_hash_layers > 0`): hash-MoE routing was not implemented

**Files:** `layers/vllm/moe_routing.py`, `quantization/mxfp4.py`,
`quantization/fp8.py`, `models/vllm/deepseek_v4_patch.py`

### What was wrong

DeepSeek-V4 has two kinds of MoE layers:

- **Hash layers** (`layer_idx < config.num_hash_layers`): expert selection is a
  pure table lookup `gate.tid2eid[token_id]` — router logits are *not* used for
  selection at all. Reference: `_topk_softplus_sqrt_torch` in
  `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`.
- **Learned layers**: `scores = sqrt(softplus(logits))`; the
  `e_score_correction_bias` is added **for selection only**, and the routing
  weights are gathered from the **unbiased** scores; then renormalize and multiply
  by `routed_scaling_factor`.

The TPU path (`select_experts`, called from `apply_monolithic` in the mxfp4/fp8
MoE quant methods) discarded `input_ids` entirely (no hash path) and ignored the
correction bias during selection. Any hash layer routed every token to the wrong
experts.

### The fix

- `select_experts` now accepts `input_ids` and routes via
  `hash_indices_table[input_ids]` when the table is present, gathering weights from
  the `sqrtsoftplus` scores (identical semantics to the reference torch fallback).
- Non-hash layers now add `e_score_correction_bias` for selection and gather
  weights from unbiased scores.
- `input_ids` is plumbed through `apply_monolithic` in both `mxfp4.py` and `fp8.py`.
- Since the table lives on the model's `GateLinear` (`gate.tid2eid`) but the TPU
  routing only sees the `RoutedExperts` module, the DSV4 patch now exposes it as a
  plain `hash_indices_table` attribute on the runner and `routed_experts` (same
  `Parameter` object, so it observes the checkpoint data after weight loading).

**Note:** if DSV4-Flash's config has `num_hash_layers = 0`, the hash part is inert
and Fix 1 is the main event. Check the config.

---

## Fix 3: CSA early-position causality leak

**File:** `custom_ops/deepseek_v4.py` (`forward_mqa`)

`max_valid_kv = clamp((pos+1) // 4, min=1)` allowed positions 0–2 to attend to
compressed slot 0 — which at that point is unwritten (garbage bytes) and, once
written, summarizes positions 0–3 (i.e. the future). The reference computes
`topk_len = (pos+1) // COMPRESS_RATIO` with **no clamp**, so early tokens attend
via the sliding window only. The clamp was removed.

---

## Fix 4: indexer weight scale constant

**File:** `custom_ops/deepseek_v4.py` (`_deepseek_v4_indexer_jax`)

The reference (`fused_indexer_q.py`) computes
`weights_out = weights * q_scale * softmax_scale * n_head**-0.5`.
The TPU code used `head_dim**-0.5` (128⁻⁰·⁵) instead of `n_head**-0.5` (64⁻⁰·⁵).
This is a uniform positive constant, so top-k *ordering* was unaffected (i.e. this
was not a garbage cause) — fixed for parity. `num_heads` is now passed into the op.

---

## Fix 5: `fused_q_kv_rmsnorm` patch used `(1 + w)` instead of `w`

**File:** `models/vllm/deepseek_v4_patch.py`

The reference Triton kernel (`_fused_q_kv_rmsnorm_kernel`) computes
`y = x * rrms * w` — plain weight, no Gemma-style `(1 + w)` offset. The PyTorch
replacement in the patch used `(1.0 + weight)`. The TPU custom attention path
doesn't call this function (it uses `self.q_norm` / `self.kv_norm`, which are
correct), but any path that does — e.g. MTP — would have been wrong. Fixed to `w`.

---

## Fix 6: removed `torch.nan_to_num` masking

**File:** `custom_ops/deepseek_v4.py` (`forward`, `_o_proj`)

`nan_to_num` calls on `attn_output`, `z_flat`, `s_flat`, and the final output were
converting numerical blow-ups into silent zeros — i.e. converting crashes into
garbage tokens and defeating the `check_nan` instrumentation. Removed so failures
surface at the layer where they originate.

---

## Verified-correct areas (no change needed)

For completeness, these were checked closely against the reference and match:

- **`_o_proj` grouped einsum**: `view(t, n_local_groups, -1)` +
  `einsum("tgd,grd->tgr", o, wo_a)` matches `rocm_inv_rope_einsum`; the
  ColumnParallel dim-0 sharding of `wo_a` is group-major and lines up with the
  contiguous head sharding of `wq_b`.
- **Compressor**: window softmax-pool → RMSNorm(weight) → interleaved GPT-J rope on
  trailing 64 channels at `(pos // 4) * 4`, fp8-UE8M0 nope + bf16 rope packing —
  matches the reference `compress_norm_rope_store_triton` contract.
- **Shared state/KV buffer**: state cache and compressed KV overlay one uint8
  buffer with independent block tables from separate KV-cache groups (same design
  as documented in `compress_norm_rope.py`); the metadata-selection order in
  `forward_mqa` (compressor-state prefix for the main cache, layer prefix for SWA)
  is consistent with where the compressor kernel writes.
- **Weight requantization flow**: the requantizer sees raw checkpoint names
  (`layers.N.attn.wq_a.{weight,scale}`), which do match its patterns; `.scale` →
  `weight_scale_inv` → `weight_scale` mapping chain is coherent with the
  per-channel (`is_channel_quant`) TPU fp8 config.
- **sm_scale** (`head_dim**-0.5` = 512⁻⁰·⁵), sink handling
  (`L = l + exp(sink - m)`), and `main_kv_lens = seq_lens // 4` all match.

## Known remaining gaps (not fixed on this branch)

1. **fp8 MoE activation clamp**: `_get_activation_str` in `quantization/fp8.py`
   does not map silu → `silu_and_mul_with_clamp` (the mxfp4 version does). Only
   matters if experts run as fp8 instead of fp4. Also `silu_and_mul_with_clamp`
   hardcodes `limit=10.0` in `kernels/megablox/gmm_v2.py` — verify it equals
   `config.swiglu_limit`.
2. **Dead code**: `patched_decoder_load_weights` in `deepseek_v4_patch.py` is never
   wired up, and its `"attn.wo_a.weight" in name` substring checks would also match
   `weight_scale` names if ever activated. Delete it.
3. **Debug scaffolding**: per-step `.item()` / `.cpu()` prints in `forward_mqa` and
   the patched `MoERunner._apply_quant_method` force a device sync per layer per
   step; the compressor's `attn_ctx.get(prefix, next(iter(...)))` fallback silently
   grabs a random layer's metadata if the prefix is missing.
4. **`kv_score` precision**: reference uses `torch.mm(..., out_dtype=torch.float32)`
   for the compressor score GEMM; the TPU path round-trips through bf16.
5. **Sequence-start state overlap**: in the shared buffer, compressed rows written
   to the entry-0 page can clobber stale state chunks for positions < ~12; harmless
   within a batch (compress reads the pre-write fp32 state view) but worth an
   audit for tiny-prompt decode.

## Validation plan on v7x-8

1. `pytest tests/kernels/test_dsv4_mla.py tests/kernels/test_dsv4_mla_swa.py
   tests/kernels/test_dsv4_compressor.py tests/kernels/test_dsv4_moe_routing.py`
   (these passed before; they should still pass — the fixes are above the kernels).
2. Short generation on the 4-layer model, then the full model. The RoPE fix should
   change output immediately.
3. If output is still wrong: check `num_hash_layers` in the model config, and run
   with `DSV4_TRACE_ACTIVATIONS=1` — with the `nan_to_num` masking gone, the first
   divergent/NaN layer in the trace is now trustworthy.
4. Caveat: routing findings were verified against **public vLLM v0.25.1**. The
   deployed Docker image appears to carry a fork (its
   `MoERunner._apply_quant_method` takes an extra `layer` arg), so double-check
   `fused_topk_bias_router.py` in that fork if routing behavior still looks off.
