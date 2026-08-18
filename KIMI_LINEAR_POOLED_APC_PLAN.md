# Plan: Pooled Prefix Caching for KDA Models (Kimi-Linear)

Goal: enable `--enable-prefix-caching` for `KimiLinearForCausalLM` (and later
Kimi-K3) on the TPU unified KV pool, instead of reviving the deleted per-layer
`MambaApcStateCopier` path.

Background: hybrid prefix caching requires the unified pool
(`tpu_platform.py:888-905`). The pool packs KDA state into attention-shaped FA
token rows; the layout is derived by `derive_pooled_gdn_state_layout`
(`src/vllm_torchtpu/gdn_pool_layout.py`). Kimi-Linear TP8 geometry:
SSM state 262,144 B fp32 (2^18), conv state 9,216 B bf16, MLA token row
2,560 B (2^9 * 5) -> ssm_tokens=103, conv_tokens=4, block floor 107.

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

      Result: block_size lands at exactly 107 (MLA backend inherits
      MultipleOf(1) kernel block sizes, so no upward rounding; fit floor 107
      beats the backend min page 64 and the default block 16). The padded
      103-row SSM region builds a valid `v3_state_source` plan inside a single
      kernel block at split=1 (typed f32 rows tile the 2560 B token row
      exactly; 1536 B padding in the last row). Finding: KDA's per-tap conv
      dim (1536 elems) does not divide the MLA pool's 640 lanes, so the GDN
      conv view rejects Kimi geometry — Phase 1.2 must build a KDA-specific
      conv view. Tests: `test_kimi_linear_padded_ssm_fit`
      (`tests/platforms/test_tpu_block_size_utils.py`),
      `TestKimiLinearMlaPoolGeometry` (`tests/kernels/test_pool_adapters.py`).

## Phase 1 - Pooled KDA op (core kernel work)

- [ ] 1.1 Correctness-first: gather/scatter the conv+ssm state out of the pool
      via `pool_adapters.gather_region`/`scatter_region` (currently uncalled)
      around the existing KDA ops (`build_kimi_kda_op`, `chunk_kda`,
      `dispatched_kda`). Isolates layout bugs from kernel bugs.
- [ ] 1.2 Define the KDA pool view: KDA conv layout
      `(kernel-1, 3, heads, head_dim)` differs from GDN; write the
      `v3_state_source`-equivalent plan for KDA shapes (possibly `rows_perm`
      reordering). Unit tests asserting byte-exact region placement.
- [ ] 1.3 (perf, deferrable) Fused streaming: extend KDA kernels to take
      `state_source`/`state_plan` directly, mirroring `fused_conv1d_gdn`.

## Phase 2 - Model and runner integration

- [ ] 2.1 `KimiKDAAttention.forward` learns the `len(kv_cache) == 1` pooled
      branch (mirror `VllmGatedDeltaNetAttention.forward`); reject PCP for KDA
      up front.
- [ ] 2.2 Verify `kv_cache_materializer` binds the pool to KDA layers as
      `[pool]` (should be arch-agnostic; confirm KDA spec flows through).
- [ ] 2.3 Validate align-mode seed copies
      (`_collect_mamba_state_seed_copies` + `copy_blocks`) fire for the KDA
      group on block-boundary crossings and prefix-resume.

## Phase 3 - Validation

- [ ] 3.1 Kernel parity test mirroring `test_gdn_pooled_prefill_parity.py`:
      pooled vs dense two-tensor path, multi-step prefill+decode, byte-exact
      state comparison.
- [ ] 3.2 Serve smoke: copy
      `scripts/vllm/integration/run_qwen35_apc_correctness.sh` for Kimi-Linear;
      include the state-contamination smoke (interleaved prefixes).
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
