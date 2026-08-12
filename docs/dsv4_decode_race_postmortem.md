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
