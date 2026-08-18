# Plan: Pooled Prefix Caching for KDA Models (Kimi-Linear)

Goal: enable `--enable-prefix-caching` for `KimiLinearForCausalLM` (and later
Kimi-K3) on the TPU unified KV pool, instead of reviving the deleted per-layer
`MambaApcStateCopier` path.

Background: hybrid prefix caching requires the unified pool
(`tpu_platform.py:888-905`). The pool packs KDA state into attention-shaped FA
token rows; the layout is derived by `derive_pooled_gdn_state_layout`
(`src/vllm_torchtpu/gdn_pool_layout.py`). Kimi-Linear TP8 geometry:
SSM state 262,144 B fp32 (2^18), conv state 9,216 B bf16, MLA token row
2,560 B packed row (2^9 * 5; 2 tokens per row) -> 103+4 packed rows, block
floor 214 tokens.

## Phase 0 - Verify the geometry end to end

- [x] Relax the layout derivation: SSM region uses ceil(token rows) with the
      last row padded; conv tile stays power-of-two but no longer needs to
      divide the SSM region; `v3_state_source` exactness assert relaxed to a
      capacity check; `pool_manifest` conv offset uses the padded SSM extent.
- [x] 0.1 Dry-run the full derivation for Kimi-Linear TP8: confirm the final
      block size `_align_block_to_backend` picks for the MLA backend, and that
      `_build_v3_pool_state_plan` constraints hold (SSM region fits one kernel
      block or spans whole ones; conv region does not straddle kernel blocks).
      Unit test with fake Kimi spec in `tests/platforms/test_tpu_block_size_utils.py`.

      Result: block_size lands at 214 tokens. The MLA pool packs the token
      axis (bf16 packing 2, so a pool row is a 2,560 B packed pair, not a
      1,280 B token): the state needs 103+4=107 packed rows = 214 tokens, and
      the derivation's row-vs-token confusion originally set 107 tokens and
      tripped the token-linearity check at serve time. `_pool_row_tokens`
      now detects the packing by probing page linearity, scales the fit
      floor, and rounds the manager block to the packing; the Raiden manifest
      derives the row quantum from the pool's own trailing dims. The MLA
      backend inherits MultipleOf(1) kernel block sizes, so no further
      rounding; the fit floor 214 beats the backend min page 64 and the
      default block 16. The padded 103-row SSM region builds a valid
      `v3_state_source` plan inside a single kernel block at split=1 (typed
      f32 rows tile the 2,560 B packed row exactly; 1,536 B padding in the
      last row). Finding: KDA's per-tap conv dim (1536 elems) does not divide
      the MLA pool's 640 lanes, so the GDN conv view rejects Kimi geometry —
      Phase 1.2 must build a KDA-specific conv view (the gather/scatter path
      of 1.1 sidesteps it). Tests: `test_kimi_linear_padded_ssm_fit`
      (`tests/platforms/test_tpu_block_size_utils.py`),
      `TestKimiLinearMlaPoolGeometry` (`tests/kernels/test_pool_adapters.py`).

## Phase 1 - Pooled KDA op (core kernel work)

- [x] 1.1 Correctness-first: gather/scatter the conv+ssm state out of the pool
      via `pool_adapters.gather_region`/`scatter_region` (currently uncalled)
      around the existing KDA ops (`build_kimi_kda_op`, `chunk_kda`,
      `dispatched_kda`). Isolates layout bugs from kernel bugs.

      Result: `build_kimi_pooled_kda_op`
      (`src/vllm_torchtpu/layers/vllm/custom_ops/kda_attention_op.py`) gathers
      both regions into dense batch-local buffers, prepends a scratch null
      slot (the kernels' slot-0 writes are all idempotent), runs the
      unmodified dispatched KDA core with identity indices, and scatters back.
      `gather_region`/`scatter_region` learned unaligned regions (`tok0 %
      ntok != 0`, needed for the conv region at token 103) as gcd-sized chunks
      on a flattened (request, chunk) grid inside ONE pallas_call — chaining
      aliased per-chunk calls corrupts earlier chunks (observed: first chunk's
      values re-read with bf16-word rounding). Parity vs the dense path:
      outputs and conv state bitwise-equal; SSM state equal to bf16 tolerance
      (fusion reassociation). Tests: `tests/kernels/test_kimi_pooled_kda.py`,
      `test_unaligned_region_gcd_decomposition_roundtrip`.
- [ ] 1.2 Define the KDA pool view: KDA conv layout
      `(kernel-1, 3, heads, head_dim)` differs from GDN; write the
      `v3_state_source`-equivalent plan for KDA shapes (possibly `rows_perm`
      reordering). Unit tests asserting byte-exact region placement.
- [ ] 1.3 (perf, deferrable) Fused streaming: extend KDA kernels to take
      `state_source`/`state_plan` directly, mirroring `fused_conv1d_gdn`.

## Phase 2 - Model and runner integration

- [x] 2.1 `KimiDeltaAttention.forward` learns the `len(kv_cache) == 1` pooled
      branch (mirror `VllmGatedDeltaNetAttention.forward`); reject PCP for KDA
      up front.

      Result: `_core_attention_pooled` dispatches to `pooled_kda_op` when the
      runner binds `[pool]`; naive-KDA and PCP streaming both raise
      NotImplementedError on the pooled path. The pooled op is built eagerly
      in `__init__` alongside the dense one.
- [x] 2.2 Verify `kv_cache_materializer` binds the pool to KDA layers as
      `[pool]` (should be arch-agnostic; confirm KDA spec flows through).

      Result: confirmed by inspection — `materialize_kv_cache_tensors` binds
      any MambaSpec layer as `[raw]`; the metadata builder derives
      `mamba_state_indices` from the block table for any mamba group under
      the unified layout (`attention_metadata.py:212`); the spec normalizer
      pads all groups to the pool page size generically.
- [x] 2.3 Validate align-mode seed copies
      (`_collect_mamba_state_seed_copies` + `copy_blocks`) fire for the KDA
      group on block-boundary crossings and prefix-resume.

      Result: mechanism is group-generic, but the serve smoke caught a real
      pool-geometry bug: `_pool_block_split` was derived as
      `block_size // raw.shape[1]`, which for the token-axis-packed MLA pool
      (107 packed rows per 214-token manager block) yielded split=2, so every
      seed copy read source blocks 2x off — garbage state on every crossing
      and resume (contamination smoke: 6/6 diverged). Now derived from bytes
      (manager page bytes / per-pool-block bytes). After the fix:
      contamination smoke 0/6 divergences, e2e divergence smoke OK,
      correctness smoke 57/58 (the one failure is an uncached short-QA
      strict-string case with cached_tokens=0 — a model-behavior artifact,
      not the cache path), and the exact-boundary token-id probe
      (`scripts/debug/probe_kda_apc_restore.py`) restores coherently on
      repeated hits. Regression tests for the split derivation in
      `tests/runner/test_mamba_state_seed_copies.py`.

## Phase 3 - Validation

- [x] 3.1 Kernel parity test mirroring `test_gdn_pooled_prefill_parity.py`:
      pooled vs dense two-tensor path, multi-step prefill+decode, byte-exact
      state comparison.

      Result: `test_pooled_kda_matches_dense_bitwise` (mixed / decode-only /
      prefill-fresh batches, complement preservation) and
      `test_pooled_kda_multistep_state_evolution` (two prefill chunks + three
      decodes threading the state through the pool each step). Outputs and
      conv state are bitwise-exact; the f32 SSM state matches within bf16
      tolerance (fusion reassociation across the two graphs). The timing arm
      of the GDN parity test is deferred to Phase 1.3: the gather/scatter
      path is correctness-first by design.
- [ ] 3.2 Serve smoke: `scripts/vllm/integration/run_kimi_linear_apc_correctness.sh`
      added (mirrors the Qwen3.5 gate). Manually validated on TP8: divergence
      smoke OK, contamination 0/6; the correctness smoke's one failure is the
      uncached translate_cn strict-string case (cached_tokens=0, model-content
      artifact), so the script runs it last and non-fatally. CI wiring pending.
- [ ] 3.3 GSM8K spot check, prefix caching on vs off
      (`scripts/eval_kimi_k3_gsm8k.py`).

## Phase 4 - Enablement

- [ ] 4.1 Add `KimiLinearForCausalLM` (and `KimiK3ForConditionalGeneration` if
      shared) to `POOLED_GDN_ARCHITECTURES`; keep
      `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0` as escape hatch.
- [ ] 4.2 Raiden/disagg manifest test with Kimi geometry (offset fix already
      generalizes).

## Later, separate workstream - partial hits (RFC #45702)

Wire `kv_cache_block_copies` into the TPU runner via `copy_blocks`, plumb
`prefix_match_unit`, pick a tail-checkpointing strategy (scheduler split vs
backend checkpoint materialization; the V3 GDN kernel's spec-decode checkpoint
mechanism is a head start for the latter).

## Risks (ordered)

1. Conv-region view for KDA's unusual state layout (1.2) - most likely source
   of silent corruption; needs byte-exact unit tests.
2. Kernel-block containment of the 103-token SSM region if the MLA backend
   picks a kernel block < 103 (0.1 catches this).
3. Metadata plumbing assumptions that all mamba layers are GDN-shaped
   (Phase 2).
