# DSV4 mainline-merge fixes (overnight session, 2026-07-29)

Context: `patemotter/dsv4-correctness-fixes-main-merged` had just merged mainline into
the DSV4 correctness branch, and the real-model repro (`examples/run_dsv4_inference.py`
against the full DeepSeek-V4-Flash checkpoint, TP=8/EP=8) went from "was running,
producing structured-but-incoherent output" pre-merge to "won't even start" post-merge.
This doc is the handoff for what was found/fixed and what's still open.

## Fixed tonight (in order hit)

1. **jax/jaxlib version mismatch.** `pyproject.toml` pins `jax==0.10.2`, but
   `vllm-torchtpu-local:latest` was still built against jax 0.10.0 (predates the
   mainline merge's jax bump, commit `799c9fe3`). jax 0.10.0's `sc_lowering.py`
   unconditionally rejects unsupported SparseCore `Get` shapes; 0.10.2 gates that
   check behind `needs_layout_passes`, which `ragged_gather_v2.py` relies on.
   **Fix:** `docker/Dockerfile.local` now force-reinstalls
   `jax==0.10.2 jaxlib==0.10.2 "numpy<2.4,>=2.0"` (the numpy pin avoids a
   numba/flax/mistral-common conflict from jax dragging in numpy 2.5.x).

2. **`TpuPlatform.support_hybrid_kv_cache()` regression** (`platforms/tpu_platform.py`).
   Confirmed via `git blame`: Pate's own commit `60da50f89` (2026-07-24) had already
   fixed this to `return True` (matching the upstream `tpu-inference` reference,
   which has always been unconditional `return True`). The mainline merge commit
   `57d27ccd` silently reverted it back to `return getattr(cls, "_is_hybrid", True)`
   during conflict resolution. `_is_hybrid` is the *real* Mamba/SSM flag
   (`model_config.is_hybrid`), unrelated to whether the platform supports vLLM's
   grouped/hybrid KV-cache-manager feature — conflating the two disabled HMA for
   DSV4 (not a real Mamba model), which routed `get_kv_cache_groups` into a buggy
   generic-hybrid conversion path instead of DSV4's own
   `group_and_unify_kv_cache_specs`/`_get_kv_cache_groups_uniform_groups` path.
   **Fix:** restored `return True`.

3. **`UniformTypeKVCacheSpecs` not unwrapped in `tpu_runner.py:initialize_kv_cache`**
   (non-unified-layout path). Two separate asserts here assumed every KV-cache
   group's spec is `AttentionSpec`/`MambaSpec` directly:
   - `"Only AttentionSpec and MambaSpec are supported in KV cache groups > 1."` —
     fired because DSV4's groups wrap multiple layers in a `UniformTypeKVCacheSpecs`
     container. Fixed by unwrapping to a representative per-layer spec (same
     pattern already used ~100 lines further down in the same file) before the
     `isinstance` checks, and dropping the `attn_block_size` uniformity assert
     (unused after that point, and incompatible with DSV4's legitimately-different
     per-group block sizes by design — see `unify_hybrid_kv_cache_specs`'s own
     comment in vLLM mainline: "For DeepseekV4, block sizes can be different for
     different KV cache groups").
   - `"KV cache duplication is only supported for hybrid models with Mamba and
     Full Attention layers."` — fired because DSV4's own page-size-bucketed
     `shared_by` grouping (`_get_kv_cache_config_deepseek_v4` in vLLM mainline)
     *also* produces multi-layer `shared_by` tensors, for an unrelated reason
     (same page size, not Mamba+Attention combining). Fixed by only taking the
     "sum all shared layers' page sizes" branch when Mamba is actually present in
     *that specific* `shared_by` group (matching `tpu_inference`'s reference
     `kv_cache_manager.py`'s `duplicate_shared_layers` logic exactly); otherwise
     use one representative layer's `page_size_bytes` (not a sum).

4. **`fp8_ds_mla` KV-cache dtype not recognized** in three places:
   - `platforms/tpu_block_size_utils.py`'s `_TPU_CACHE_DTYPE_TO_TORCH_DTYPE`
   - `layers/vllm/attention.py`'s `TPU_STR_DTYPE_TO_TORCH_DTYPE`
   - Both now map `"fp8_ds_mla": torch.uint8`, matching vLLM mainline's own
     canonical mapping (`vllm/utils/torch_utils.py`).

5. **`PallasMLAttentionBackend.get_kv_cache_shape`/`get_kv_cache_page_size_bytes`
   never implemented `fp8_ds_mla`'s packed-byte shape** (`layers/vllm/attention.py`).
   This is a **missing TPU port of an existing mainline GPU feature**, not a
   regression — confirmed via `git log -S "fp8_ds_mla" -- attention.py` (zero
   hits, ever) and `git blame` (the blocking `is_floating_dtype` check predates
   `fp8_ds_mla` entirely, from the original 2026-07-17 DSV3 MLA work). Cross-checked
   against vLLM mainline's own DeepSeek-V4 GPU backend
   (`vllm/v1/attention/backends/mla/sparse_swa.py`'s `DeepseekSparseSWABackend`),
   which returns `(num_blocks, block_size, 584)` for this dtype (448 NoPE fp8 +
   128 RoPE bf16 + 8B scale, no padding needed on GPU). TPU needs the packed
   layout padded to a 128-aligned minor dim per
   `kernels/deepseek_v4/compress_norm_rope.py`'s own docstring
   (`width = align_to(583, 128) = 640`).
   **Fix:** added `PallasMLAttentionBackend._is_ds_mla_packed_cache()` (handles
   both the raw string and the pre-resolved `torch.uint8` dtype — the latter is
   what `get_kv_cache_page_size_bytes` passes internally) and computes the
   packed width via the already-existing
   `compress_norm_rope.sparse_packed_width(nope_dim, rope_head_dim=64,
   quant_block=64)` helper, then delegates to the existing (already-correct)
   `mla_v2_kernel.get_kv_cache_shape`'s generic 128-alignment logic. Verified via
   pure-Python shape math (640 bytes/token, matches the docstring exactly), a new
   unit test (`tests/layers/vllm/test_mla_attention.py::
   test_pallas_mla_attention_backend_fp8_ds_mla`), a differential test run
   (`git stash` before/after — identical 31 pre-existing failures either way, zero
   regressions), and confirmed passing on **real TPU hardware**
   (`--use-tpu`, `tests/layers/vllm/test_mla_attention.py` +
   `tests/kernels/test_dsv4_mla.py` + `test_dsv4_mla_swa.py`; the two
   `test_dsv4_mla.py::CorrectnessTest` cases that need real kernel execution
   only pass with `--use-tpu`, not the CPU-only sandboxed check).

## Also found and confirmed NOT a regression

- `envs.py`'s `TPU_KV_CACHE_HEADROOM_MIB` removal — deliberate mainline PR
  (`9e00af34 Delete TPU_KV_CACHE_HEADROOM_MIB (#151)`), not clobbered.
- `kv_cache_spec_normalizer.py`'s `enable_unified_block_pool` →
  `enable_unified_kv_layout` rename, `fused_moe_gmm.py`'s v1/v2 ragged-gather
  env-var switch removal, `FusedMoE`→`RoutedExperts` renames throughout
  quantization/*.py — all legitimate mainline API changes, not clobbers.
- `moe_routing.py`'s `maybe_force_random_routing` addition — new mainline
  profiling feature, not a removed one.

## Infrastructure notes (compilation caching)

Three *separate* caching layers exist; only the third actually matters for the
custom Pallas kernels this repo uses:

1. `JAX_COMPILATION_CACHE_DIR` — only applies to plain `jax.jit`-dispatched
   compiles. Doesn't apply here: `torch_tpu`'s `pallas.jax_op`/`JaxCallable`
   wrapper uses `jax.export.export(...)` + hands the serialized MLIR to a native
   `tpu_torch_pallas` extension, entirely bypassing JAX's own dispatch/cache path.
2. **Tier-2** (`torch_tpu`'s in-process/cross-rank cache) — defaults on
   automatically, backed by `/dev/shm/torch_tpu_cache` (shared-memory tmpfs, so it
   persists across separate `docker run` invocations as long as `/dev/shm` is
   mounted, which it always is here). No env var needed. Confirmed to have been
   silently accumulating for days (82GB, July 26–29) independent of anything set
   tonight.
3. **Tier-3** (`torch_tpu`'s persistent on-disk cache) — disabled by default
   (empty root dir). Needs `TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=<dir>`
   set explicitly. The "official" single knob is `VLLM_XLA_CACHE_PATH`
   (`env_override.py` auto-wires both tiers from it via `setdefault`) — didn't
   switch to this mid-session to avoid losing continuity with the already-warm
   Tier-2 "default"-named bucket, but it's the cleaner way to do this going
   forward.
   - `TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0` forces a
     cache-only read (no parallel local-recompile race) — useful for verifying
     cache correctness unambiguously, not needed for normal runs.
   - Confirmed correct via a controlled kill/restart test: killed a run after
     ~108 cached kernels, restarted with the above env var, got bit-identical
     debug output at the same checkpoint (`layers.1`) — proves reads are
     functionally correct, not just that writes happen.
   - `ABSL_VLOG`-level "Tier-3 cache HIT/MISS for key X" log lines exist in
     `torch_tpu/common/compilation_cache.cc` but need verbosity bumped (no env
     var found for this yet — the K8s CI config sets
     `TPU_VMODULE=tpu_pjrt_client=1,pjrt_stream_executor_client=1,
     tpu_pjrt_compiler_utils=1` which may be the mechanism, untested locally).
   - The debug-MLIR-dump feature in `patches/pallas_patched.py` (writes to
     `/mnt/workspace/tmp/exported_mlir_*.mlir`) silently fails in this Docker
     setup since `/mnt/workspace` isn't mounted (workspace is at
     `/home/patemotter_google_com/workspace` instead) — harmless (bare
     try/except), but if the dump is ever wanted, fix the hardcoded path.

## New crash found live, right after the above 4 fixes landed

With all four fixes committed, the same repro run (TP=8/EP=8, full
DeepSeek-V4-Flash, prompt "Compare XLA and LLVM in two bullet points.") got
further than ever -- past the MoE anomaly's layer -- and then hit a **new**
crash across all 8 workers simultaneously:

```
ValueError: state row needs 512B but cache width is 128B
  File ".../kernels/deepseek_v4/compressor.py", line 156, in compressor_forward_indexer
    state_view = unpack_state_cache(state_cache_buf, state_block_size, ...)
  File ".../kernels/deepseek_v4/compress_norm_rope.py", line 196, in unpack_state_cache
    kv_slots, rpt, _, bpr = _state_chunk_dims(cache.shape, state_block_size, ...)
  File ".../kernels/deepseek_v4/compress_norm_rope.py", line 183, in _state_chunk_dims
    raise ValueError(f"state row needs {bytes_per_row}B but cache width is {width}B")
```

This is **not new code** -- it's the DSV4 HCA/CSA compressor's state-cache
(`VllmCompressorStateCache.get_kv_cache_spec`,
`layers/vllm/custom_ops/deepseek_v4.py:537-560`), which by design overlays
the same physical pages as the shared MLA cache (its own comment at line
528-533 spells this out, from an earlier fix that stopped state writes from
clobbering neighboring KV data on the same page). This is very likely the
same thing flagged as unresolved in prior project memory ("HCA
compress_ratio clamp confirmed real ... root cause still open") -- tonight
is the first time any run has gotten far enough in real forward-pass
execution to actually trigger it live, rather than being blocked earlier by
one of the four now-fixed crashes.

Mechanically: `get_kv_cache_spec` (allocation time) computes
`bytes_per_row = (state_dim // rows_per_token) * 4` where
`rows_per_token = page_size // self.block_size` (`self.block_size` is 4 for
CSA / 8 for HCA per that class's own comment), then declares
`head_size = align_to(bytes_per_row, 128)`. `_state_chunk_dims` (use time)
recomputes the same quantity from the *actual* allocated cache shape:
`rows_per_token = kv_slots // state_block_size` where
`kv_slots = rows * packing` read back from `cache.shape`. The mismatch
(needs 512B, cache is only 128B) means these two computations disagree --
either the actual allocated buffer's page geometry doesn't match what
`get_kv_cache_spec` assumed at declare time (plausible fallout from this
session's own KV-cache-groups fixes changing which buffer/group this
spec's tensor actually gets bucketed into), or `state_block_size` passed
at use time differs from `self.block_size` used at declare time. Someone
picking this up should start by printing, at both call sites in the same
run, the actual `cache.shape` / `state_block_size` / `self.block_size` /
`page_size` values to see exactly where the two computations diverge --
don't reason about it from the formulas alone, get live numbers.

## Open lead: MoE `fused_expert_output` anomaly (NOT YET INVESTIGATED TO GROUND)

With all five fixes above in place, the real full-model run (TP=8/EP=8,
DeepSeek-V4-Flash, prompt "Compare XLA and LLVM in two bullet points.") got
further than any run all session — past all five crash points, into real forward
passes with actual weight/activation data. `models/vllm/deepseek_v4_patch.py`'s
`[DEBUG_PATCH]` instrumentation on `MoERunner._apply_quant_method` showed, at
layer 0, across the 8 TP/EP ranks:

```
shared_expert_output: all 8 ranks consistent, std ~0.02-0.05 (sane)
fused_expert_output (routed/local experts):
  TP0, TP2, TP3, TP4, TP5: EXACT zero (mean=std=min=max=0.0)
  TP1: mean=0.0013, std=0.045   (comparable scale to shared_expert)
  TP6: mean=0.025,  std=5.2     (~100x larger than shared_expert -- suspicious)
  TP7: mean=0.006,  std=0.448   (~10x larger -- also suspicious)
```

This is squarely in DSV4's MXFP4 MoE / expert-parallel-routing territory
(`layers/vllm/quantization/mxfp4.py:apply_monolithic` →
`layers/common/fused_moe_gmm.py:fused_moe_func`/`moe_gmm`/
`prepare_routed_gmm_inputs` → `kernels/sparse_core/ragged_gather_reduce_v2.py`).
This is plausibly connected to the *original* motivating bug from before tonight's
session even started (pre-merge: DSV4 ran to completion but produced
"structured but incoherent" text) — a MoE contribution that's either missing
(exact zero on 5/8 ranks) or ~10-100x too large (2/8 ranks) would corrupt the
residual stream exactly this way.

**Traced and ruled out:** `ragged_gather_reduce_v2.py`'s `_fallback_implementation`
(the likely-active path for this tiny 64-token batch, since it's small enough to
fit comfortably in SparseCore VMEM) is *not* the bug — it's double-protected
against uninitialized/garbage memory: invalid (non-locally-routed) entries get
`topk_weight=0` upstream in `fused_moe_func` (`jnp.where(valid, topk_weights, 0)`)
*and* get explicitly re-masked via `jnp.where(valid_rows_mask, out, 0)` inside the
reduce itself. No NaN/Inf detected either (the `[NAN_DETECTION]` guard in the
same debug patch never fired), so it's not a garbage-read-through-zero-times-Inf
scenario.

**Not yet checked, next steps for whoever picks this up:**
- `moe_routing.select_experts`/`get_experts_start` — is `topk_ids`/`experts_start`
  actually correct per-rank? (`get_experts_start` computes
  `ep_rank * base + min(ep_rank, remainder)`; worth a live print of
  `topk_ids`/`experts_start`/`w1.shape[0]` per rank at layer 0 to see whether the
  *routing itself* is skewed/wrong, vs. correct-but-then-mis-scaled downstream.)
- The FP4 GMM dequantization path itself (`gmm_wrapper`/`moe_gmm`'s w1_scale/
  w2_scale handling) for a scale-factor bug that could produce a ~10-100x
  inflation on ranks whose local experts happen to get real tokens.
- Whether `zero_initialize=False` in the `gmm_wrapper` calls matters on some
  *other* code path than the fallback one traced above (e.g. the full SparseCore
  path, not the fallback, if a larger batch or different generation ever takes
  that branch instead).
- Reproduce with a smaller, faster setup than the full 43-layer/8-rank run if
  possible, to shorten the iteration loop for this specific investigation.

## Files changed (uncommitted as of this session's end)

- `docker/Dockerfile.local` — jax/jaxlib/numpy pin
- `src/vllm_torchtpu/platforms/tpu_platform.py` — `support_hybrid_kv_cache` fix
- `src/vllm_torchtpu/platforms/tpu_block_size_utils.py` — `fp8_ds_mla` dtype map
- `src/vllm_torchtpu/runner/tpu_runner.py` — `UniformTypeKVCacheSpecs` unwrap ×2
- `src/vllm_torchtpu/layers/vllm/attention.py` — `fp8_ds_mla` dtype map + shape fix
- `tests/layers/vllm/test_mla_attention.py` — new `fp8_ds_mla` shape test
