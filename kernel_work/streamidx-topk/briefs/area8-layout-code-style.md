# Layout and code style — file plan for the step-2 designer

# File plan for `streamindex_topk` — instantiated from `references/code-style.md`

Everything below is sourced from `/home/annyan_google_com/workspaces/copy1/kernel5_copy/.claude/skills/write-tpu-kernels/references/code-style.md` §10 (the two decomposition tables), §4 (the ref-count rule), §11 and §12. Quoted column text is that file's own wording.

## 1. The four-file decomposition, instantiated

`code-style.md` §10 names **four** files for one operation, so the plan has four rows. Target package: `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk/`, with `__init__.py` re-exporting `streamindex_topk` so the existing import path in `mla_attention_op.py` and `tests/kernels/test_dsv4_streamindex_topk.py` survives.

| File | What it holds | What it must not hold | ~lines |
| --- | --- | --- | --- |
| `api.py` | "the free function users call; jaxtyping annotations; the backend fallback chain" — i.e. the public `streamindex_topk(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens, distribution, *, k, compression_ratio)`, its jaxtyping signature in the glossary's letters, the `enable_early_exit` / `_fast_path` fallback chain, and the `NotImplementedError` raises §9 requires for shapes this backend cannot serve | "tuning knobs — **no `config` at all**" — no `num_kv_pages_per_block`, `num_queries_per_block`, `vmem_limit_bytes`, `decode_req_batch_size`; no `pl`, no `pltpu` | ~120 |
| `base.py` | "the `Op` subclass, `bind()` validation, the plain-XLA reference `_fwd`, the module-level `vjp`" — `bind()` checking `cu_q_lens`/`seq_lens`/`page_indices`/`distribution` agree, and the plain-JAX `_fwd` computing `score[t,s] = sum_h relu(q[t,h]·k[s]) * w[t,h]`, causally masked, then `lax.top_k`. This is the oracle §12 names; today it exists only as `streamindex_topk_ref` inside the test file. No `vjp` — index output is non-differentiable; state that in the file rather than omitting it silently | "any backend-specific code" — no Pallas, no Mosaic, no SparseCore, no `lkv_dim` padding arithmetic, no page-table walk | ~180 |
| `pallas_mosaic_tpu.py` | "the backend `Op` subclass, its `Config`, `_get_heuristics_config`, `_get_autotuning_configs`, `supported_on`" — the pydantic `Config` carrying `bkv_p`, `bq_sz`, `seq_batch_size`, `vmem_limit_bytes` per `MlaCase` (decode / prefill / mixed); `_get_heuristics_config` deriving them from `page_size`, `pages_per_seq`, `max_num_seqs` and `pltpu.get_tpu_info().vmem_capacity_bytes`; `supported_on` gating on `get_tpu_info().sparse_core is not None` and `bkv_sz % 128 == 0`. The `(2,2,2)` / `(1,128,128)` tuples currently living at the *call site* (`mla_attention_op.py:119-120`) move here | "**`pl.pallas_call`**" — and no kernel body, no `Ref`, no `BlockSpec` | ~220 |
| `pallas_mosaic_tpu_kernel.py` | "the `pallas_call`, the kernel body, the index maps" — the `pltpu.PrefetchScalarGridSpec`, `in_specs` / `out_specs` / `scratch_shapes` / `input_output_aliases`, the grid `((end_seq_idx - start_seq_idx) // seq_batch_size,)`, the three `run_scores_kernel` invocations, and the kernel body itself | "policy decisions about *which* config to use" — it consumes `bkv_p` / `bq_sz` / `seq_batch_size`, it does not pick them, and no VMEM-budget arithmetic appears in it | ~250 as the top-level file, **but see §2** — the block's kernel-side concerns are ~700 lines, so this row does not close without the further split |

`_get_heuristics_config` is the spelling the reference gives; not `_default_config`, not `_heuristic`.

## 2. Yes — this block needs the further decomposition, and it is the load-bearing half of the plan

§10's threshold is explicit: **"Split at around 250 lines."** The incumbent's kernel body alone is lines 65-436 (~370) and its launcher lines 509-865 (~355). Row 4 above cannot hold both and stay under the threshold, so §10's second axis applies: **"the right axis is what each module is allowed to touch — refs, DMA, arrays, or plain ints. Make 'does this function take a `Ref`?' a file boundary rather than a per-function surprise."**

Which modules fall on which side, using the reference's own module names:

| Module | May touch | Instantiated for this block | ~lines |
| --- | --- | --- | --- |
| `tiling.py` | "ints only — block extents, grid extents" | `cdiv`, `align_to`, `get_dtype_packing`; `bkv_sz = bkv_p * page_size`, `bkv_sz_per_kv_packing`, `num_sublanes_bkv = bkv_sz // 128`, `num_sublanes_total`, `pages_per_seq`, the `lkv_dim = align(129, 128)` reasoning, the grid extent. "testable with no device at all" | ~90 |
| `compute.py` | "arrays only — **no refs**" | `compute_scores`' body — the `bnd,bmd->bnm` einsum, relu, `* weights`, `sum_h`, the ue8m0 scale multiply, the valid + causal mask; plus the host-side pure-array `_effective_row_lengths`, `prepare_q_inputs`, `prepare_index_weights`. "unit-testable in plain JAX against the `base.py` reference" | ~200 |
| `load_store.py` | "refs in, arrays out" | `load_bq`, `load_bq_weights`, `load_bkv` including the `pltpu.bitcast` to `float8_e4m3fn` and the ue8m0 scale-byte unpack. "the only file where `ref[...]` appears" | ~110 |
| `dma.py` | "`make_async_copy`, semaphores, memory spaces" | `_async_copy`, `_fetch_bkv` (dynamic `safe_page_offset`), `_fetch_bq`, `start_send_scores` / `wait_send_scores`, the `sem_ids_ref` rotation across the four semaphore banks. "the only file that can deadlock" | ~200 |
| `memory_ref.py` | the named ref bundle | **§4 forces this and §10's table does not name it.** "If a kernel's ref list grows past roughly eight, stop adding positional slots and pass a named bundle dataclass instead." The incumbent's `kernel` has **16** positional ref slots (6 prefetch + 4 in + 1 out + 5 scratch) and `_topk_kernel` has **9**. The repo already has the pattern: `gdn/v3/memory_ref.py`'s `MetadataRef` / `WeightRefs`, and `compress_and_store/buffered_ref.py` | ~80 |
| `config.py` | "the config dataclass" | folds into `pallas_mosaic_tpu.py` above unless the option carries per-`MlaCase` autotuning tables, in which case split it out | 0–120 |
| `wrapper.py` | "assembly: builds the `pallas_call` out of the above" | `run_scores_kernel` and the three decode / decode-remainder / mixed invocations, `scores_init`, the reshape and `-inf` pad, the `sparsecore_topk` hand-off | ~180 |

**Where stage B falls is a design decision the plan has to state, not inherit.** `sparsecore_topk` is a *second operation* — its own public entry, its own `pl.kernel`, its own `VectorSubcoreMesh`, its own callers — so under §10 it gets its own four-file set and stage A's `api.py` calls it as a dependency. Its internals split on the same axis: `_pick_partition` is ints-only → `tiling.py`; the stage-1→stage-2 candidate index remapping (`jnp.take_along_axis`, offset add-back, `-1` masking, lines 418-457) is arrays-only → `compute.py`, testable in plain JAX today and untested today; `_topk_kernel`'s `plsc.*` body is refs+DMA. **An option that fuses stage B into stage A collapses this to one operation and one file set** — per `SKILL.md`, that differing file count between options is itself the signal that the two options sit at different fusion boundaries, so both plans should be written out rather than one inherited from the other.

**§10's escape hatch is live here and must be exercised explicitly if taken.** "A kernel doing dynamic-offset DMA, in-kernel metadata construction and selective triple buffering may have nothing the four-file shape can hold. Opting out is a legitimate outcome, not a mess to be cleaned up later — **say so in the module docstring and leave.** What is not legitimate is half-using it." This block does all three: dynamic `safe_page_offset` DMA, `cu_q_lens` walked in-kernel to build per-sequence metadata, and double buffering across four semaphore banks. Opting out gives `streamindex_topk/{tiling,compute,load_store,dma,memory_ref,kernel,wrapper}.py` with a declaring docstring — **which is also what every multi-file kernel in this repo already looks like** (see the repo note at the end). Either choice is defensible; the undeclared middle is not.

## 3. `code-style.md`'s `## Checklist`, verbatim

```
- [ ] Dimension glossary at the top of the file; uppercase logical, `t`-prefixed tile, `n`-prefixed
      grid extent. Padded extents get their own letter.
- [ ] Every ref parameter has a trailing `# [shape] dtype` comment in the glossary's letters, saying
      whether it is the block or the whole array.
- [ ] Extents derived from the refs, not re-read from the config. No bare `x.shape[2]` in
      arithmetic.
- [ ] jaxtyping on the public wrapper using the same letters; bare refs inside — never a
      `jax.Array` annotation on a ref.
- [ ] Kernel signature carries `# Inputs. / # Outputs. / # Scratch.`
- [ ] Kernel body in order: indices → shapes → semaphores → init guard → loads → compute → masked
      store → epilogue.
- [ ] Single characters only for loop indices, annotated math tensors, uppercase extents and `d<x>`
      gradients. Grid indices spelled out.
- [ ] Memory space in the ref name wherever two spaces share a signature. Never `_ptr`.
- [ ] Comments name the compromise, not the operation. One-line module docstring.
- [ ] Unsupported features raise `NotImplementedError`, not `ValueError`.
- [ ] Four files: `api.py` / `base.py` / `pallas_mosaic_tpu.py` / `..._kernel.py`. No `pallas_call`
      outside the last, no `config` in the first.
- [ ] Where it splits further, the boundary is what the module may touch: arrays / refs / DMA /
      ints.
- [ ] Config is a frozen, kw_only, slots pydantic dataclass; cross-field checks in `__post_init__`.
- [ ] Tested against the XLA `_fwd` in `base.py`, tolerance visible at the call site, runtime skips.
- [ ] Hardware constants from `pltpu.get_tpu_info()`, never hardcoded.
```

---

## What the incumbent already violates

The designer should know it is departing from the repo, not only from the guide. Line references are `streamindex_topk.py` unless marked `sc:` (= `sparsecore_topk.py`).

**Hard, unambiguous:**

- **§10 four files, and the 250-line threshold.** One 865-line module holding six concerns (helpers, `MlaCase`, the kernel body, host input prep, `_effective_row_lengths`, and the launcher + orchestration + fast path), plus `sparsecore_topk.py` at 457. **3.5× and 1.8× the split threshold.** No `api.py`, no `base.py`, no `pallas_mosaic_tpu.py` anywhere in the block. The block-size decision does not even live in either prescribed home — it is at the *call site*, `mla_attention_op.py:119-120`.
- **§4 ref bundle.** 16 positional ref slots in `kernel` (:65-92); 9 in `sc:_topk_kernel`. The rule triggers at eight.
- **§11 Config.** There is none — not pydantic, not `dataclasses`, not anything. `pydantic` appears in **zero** files under `src/vllm_torchtpu/kernels/`. Consequently `_get_heuristics_config` and `_get_autotuning_configs` do not exist either.
- **§13 hardcoded hardware constants.** `DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024` (:28) is the guide's own ❌ example verbatim in form. `streamindex_topk.py` **never calls `get_tpu_info()`**. `128` appears in arithmetic at :108, :445, :598, :604, :639 (the guide permits the literal in a *shape comment*, not in a computation). `sc:LANES = 16`, `sc:MAX_SLICE_WORDS = 64*1024`, and `sc:num_subcores = 16` / `sc:num_sc_cores = 2` hardcoded inside `_topk_kernel` (:85-86) **while `_sc_topk_direct` reads the same two values from `info.sparse_core` at :315-331** — the file reads and hardcodes the same constants and they can disagree. Step 1 §7 already caught the downstream symptom: the module docstring's "all 32 subcores on the chip" contradicts the reported `num_cores=2, num_subcores=16`.
- **§1 dimension glossary.** Absent from both files. Names do not follow the `UPPERCASE` / `t`-tile / `n`-grid convention (`bkv_sz`, `bq_sz`, `lkv_dim`, `num_sublanes_bkv`, `bkv_p`), and `num_positions` — the padded extent that §5 of the step-1 report says is "not the sequence length; it is padded up" — has no distinct letter, exactly the case §1 singles out.
- **§8 one-line module docstring.** `sparsecore_topk.py`'s is 16 lines (:14-29).
- **§3 compute the extent once.** `bkv_sz_per_kv_packing`, `bkv_sz` and `num_sublanes_bkv` are each computed **twice**, at :106-108 in the kernel body and again at :636-639 in the launcher — "three chances to round differently."
- **§4 jaxtyping on the public wrapper.** Neither entry point has it; `jaxtyping` appears in exactly one file in the whole `src/` tree (`kimi_k3/chunk_kda.py`).

**Partial or judgement-call:**

- **§2 shape comments** — the kernel signature does carry them (:67-85), in a `# Shape: […], Memory: …` idiom rather than `# [shape] dtype`. But **none of the four scratch refs carries a dtype**, and §2 says "always on scratch." `sc:_topk_kernel` has bracket comments but no `# Inputs./# Outputs./# Scratch.` sections at all (§6).
- **§6 sectioning** — `kernel` uses `# Prefetch / # Input / # Output / # Scratch`, close to but not the prescribed spelling; `sc:_topk_kernel` has none.
- **§5 memory-space suffixes** — well applied to the HBM inputs (`q_hbm_ref`, `cache_kv_hbm_ref`), but the four VMEM scratch refs are `bkv_x2_ref` / `bq_x2_ref` / `bq_weights_x2_ref` / `scores_block_x2_ref` in a signature where HBM, VMEM and SMEM all appear, and `sems` lacks `_sem`. `sc:` drops `_ref` entirely (`scores_hbm`, `resident_vmem`).
- **§5 single-character budget** — `s` for a computed score tensor (:326), `sc:s = row_c * n` (:152), `sc:p`, `sc:w`, `sc:b/n/k`.
- **§7 body order** — `pl.program_id(0)` is at :112, after the shape constants rather than first; semaphore aliasing happens inside closures rather than being named before any DMA code. The init-guard / masked-store pair is genuinely inapplicable (no accumulator across a grid axis).
- **§9 `NotImplementedError`** — `_sc_topk_direct` gets it right for absent SparseCore (`sc:318`). But `bkv_sz % 128 != 0` (:598) and `sc:`'s `dtype != f32` / `n % LANES` / `k % LANES` (`sc:394-402`) raise `ValueError` for what read as *this backend cannot serve these arguments*, which §9 assigns to `NotImplementedError`. Worth a deliberate call rather than a copy.
- **§12 tests** — a real test exists (`tests/kernels/test_dsv4_streamindex_topk.py`, 742 lines) with an in-repo exact NumPy oracle, which is the right instinct. But the oracle lives **in the test file**, not as a `base.py` `_fwd`; there is no `atol`/`rtol` anywhere in the file; and there is no runtime `self.skipTest(...)` reading `pltpu.get_tpu_info()`.
- **§2's "no `Args:` block"** — `streamindex_topk`'s docstring (:541-565) is one. It carries semantics rather than shapes, so it is not the pure restatement §2 targets, but it does enumerate every parameter.

**One thing the designer should weigh before treating row 1–4 as settled.** `code-style.md` is the authority and I am reporting it as written — but **no file anywhere in `src/vllm_torchtpu/` is named `api.py`, `base.py`, or `pallas_mosaic_tpu*.py`.** The repo's actual convention for a kernel too big for one file is a package directory whose members are `config.py`, `kernel.py`, `wrapper.py`, `memory_ref.py`, `compute.py`, `metadata.py`, `vmem_ldst.py` — see `gdn/v3/` (11 files), `compress_and_store/` (9), `sparse_core/ragged_gather_reduce_v2/` (`config` / `kernel` / `memory_ref` / `wrapper`), `mla/v2/`, `fused_moe/v2/`. That is §10's **second** table almost name-for-name, and none of the first. So whichever way an option goes it is departing from something, and `SKILL.md` requires the departure be declared in the design rather than half-adopted. My recommendation to the designer: state which of the two shapes the option takes and why, in the module docstring, per §10's opt-out rule.
