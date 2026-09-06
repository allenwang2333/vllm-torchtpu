## Layout brief — file plan for `streamindex_topk`

Authority: `/home/annyan_google_com/workspaces/copy1/vllm-optimization-agent/.claude/skills/write-tpu-kernels/references/code-style.md`, §10 (Files) and the Checklist. Everything below is sourced from that file; the incumbent and the repo are reported only as departures.

---

## 1. The file decomposition, instantiated

`code-style.md` §10 names **four** files ("One operation, four files, each with a job you can state in a sentence"). Four rows, no more, no fewer. The "What it holds" / "What it must not hold" columns are the reference's own words, with the instantiation for this block below each.

| File | What it holds | What it must not hold | ~lines |
| --- | --- | --- | --- |
| `api.py` | "the free function users call; jaxtyping annotations; the backend fallback chain"<br>→ `streamindex_topk(...)`; jaxtyping on `q`, `indexer_weights`, `cache_kv`, `seq_lens`, `page_indices`, `cu_q_lens`, `distribution`; the call out to `sparsecore_topk` | "tuning knobs — **no `config` at all**"<br>→ so `num_kv_pages_per_block`, `num_queries_per_block`, `vmem_limit_bytes`, `decode_req_batch_size` and `DEFAULT_VMEM_LIMIT_BYTES` (all in the incumbent's public signature, L28/L531-537) cannot appear here | ~80–110 |
| `base.py` | "the `Op` subclass, `bind()` validation, the plain-XLA reference `_fwd`, the module-level `vjp`"<br>→ argument validation (incumbent L569-601); `_effective_row_lengths` (L482-506); `_fast_path` (L840-856) and the early-exit gate; **a plain-JAX `_fwd` that does not exist today** — the only reference in the tree is inside `tests/kernels/test_dsv4_streamindex_topk.py:100-106`. No `vjp`: inference-only | "any backend-specific code"<br>→ no `pltpu`, no `bkv_p`/`bq_sz`, no VMEM arithmetic | ~160–220 (of which the `_fwd` is new code, not moved code) |
| `pallas_mosaic_tpu.py` | "the backend `Op` subclass, its `Config`, `_get_heuristics_config`, `_get_autotuning_configs`, `supported_on`"<br>→ the `(2,2,2)` / `(1,128,128)` per-case tuples, `decode_req_batch_size=4`, the VMEM budget for the four double buffers (L654-692), `num_sublanes_total` (L603-605), and **the choice of which band gets which config** — `num_kv_pages_per_blocks[0]` vs `[2]` in `_common_path` is a policy decision about which config to use, so it selects here and is only *applied* next door | "**`pl.pallas_call`**" | ~150–200 |
| `pallas_mosaic_tpu_kernel.py` | "the `pallas_call`, the kernel body, the index maps"<br>→ `kernel()` (L65-436), `run_scores_kernel`'s `pallas_call` and specs (L607-738), the three band invocations, `prepare_q_inputs` / `prepare_index_weights` / `prepare_outputs` (L439-479) | "policy decisions about *which* config to use" | ~500–650 |

Two consequences the reference forces and the designer should not have to rediscover:

- **`sparsecore_topk.py` is a different operation, so it is its own four-file group, not a fifth file of this one.** `api.py` calls it. At 457 lines it is itself past the split point below.
- **§10 sanctions an opt-out for a kernel of exactly this description**, and this is not an escape hatch invented here: *"A kernel doing dynamic-offset DMA, in-kernel metadata construction and selective triple buffering may have nothing the four-file shape can hold. Opting out is a legitimate outcome, not a mess to be cleaned up later — **say so in the module docstring and leave.** What is not legitimate is half-using it: an `Op` subclass whose `_fwd` is a thin shim over a thousand lines that ignore the abstraction is worse than either choice made cleanly."* This block does dynamic-offset DMA (`safe_page_offset`, L179-185), in-kernel metadata construction (`q_lens`/`kv_lens`/`seq_lens` assembled from SMEM in the body, L115-126) and double buffering. The reference gives two clean choices and forbids the third; the incumbent has taken the third, silently.

---

## 2. Does the further decomposition apply? — **Yes**

§10, "Splitting further: draw the boundary at what a function may touch". It applies because the kernel file above projects to ~500–650 lines, roughly 2× the split point, and because "An operation big enough to outgrow four files needs a second axis, and the right axis is **what each module is allowed to touch** — refs, DMA, arrays, or plain ints. Make 'does this function take a `Ref`?' a file boundary rather than a per-function surprise."

The payoff the reference names is the reason it matters here: "**A function that takes only arrays can be called from XLA, from a unit test, and from inside a kernel unchanged** — no device, no `pallas_call`, no grid." The `base.py` `_fwd` this block has to write and the microbenchmark's correctness oracle both want exactly that.

The reference's table is "A shape to copy — these are module names for a large fused op, not a citation." Which side each module of this block falls on:

| Module | May touch | This block's content |
| --- | --- | --- |
| `compute.py` | "arrays only — **no refs**" | `compute_scores` (L304-354): the `bnd,bmd->bnm` einsum, ReLU, per-head weight multiply, head-axis sum, `e8m0` scale multiply, valid + causal mask. Plus `prepare_q_inputs` / `prepare_index_weights` / `prepare_outputs` (L439-479). Unit-testable in plain JAX against the `base.py` reference |
| `load_store.py` | "refs in, arrays out" | `load_bq` (L242-248), `load_bq_weights` (L250-255), `load_bkv` (L257-272) — the fp8 / `e8m0` unpack out of the 256-byte record. "the only file where `ref[...]` appears" |
| `dma.py` | "`make_async_copy`, semaphores, memory spaces" | `_async_copy` (L153-158), `_fetch_bkv` (L160-198), `_fetch_bq` (L200-228), `start_/wait_fetch_*` (L230-240), `wait_send_scores` (L128-135), `start_send_scores` (L137-151), the `get_next_bq_ids` / `get_next_bkv_ids` semaphore rotation (L283-302). "the only file that can deadlock" — non-theoretical here: the epilogue at L431-434 drains two in-flight output DMAs |
| `tiling.py` | "ints only — block extents, grid extents" | `cdiv`, `align_to`, `get_dtype_bitwidth`, `get_dtype_packing` (L31-46); `bkv_sz_per_kv_packing`, `bkv_sz`, `num_sublanes_bkv`, `pages_per_seq` (L104-108, L636-639); `num_sublanes_total` (L603-605); the `bkv_sz % 128` check (L596-601). "testable with no device at all" |
| `config.py` | "the config dataclass" | the per-case `(bkv_p, bq_sz)` tuples, `decode_req_batch_size`, `vmem_limit_bytes`, `compression_ratio`, `enable_early_exit` |
| `wrapper.py` | "assembly: builds the `pallas_call` out of the above" | `run_scores_kernel` (L607-738) and the three band invocations from `_common_path` (L757/775/793) |

What stays in the kernel body proper, because it takes refs and drives the grid: `process`, `compute_with_bq`, `compute_with_bkv`, the prologue/epilogue `@pl.when` pair, and the signature itself (~180 lines).

**One thing the reference does not settle, and I am not settling it either:** the second table's `config.py` and `wrapper.py` overlap the first table's `pallas_mosaic_tpu.py` and `pallas_mosaic_tpu_kernel.py`. §10 gives both tables and does not say how the two compose for an op that needs both. Flagged for the designer.

The reference also gives the granularity choice, and both instances are relevant: the axis can be drawn "**At file granularity**" (splash attention: pure host logic → host-built arrays → device refs) or "**At function granularity**" (ragged dot: "the array functions are annotated and above; the ref functions are bare and below", in one file). The second is the cheaper move if the designer wants one option that does not multiply files.

---

## 3. The checklist, verbatim and complete

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

---

## Where the incumbent departs

A redesign inherits whatever it does not deliberately change. Against the checklist, item by item:

| Checklist item | `streamindex_topk.py` |
| --- | --- |
| Dimension glossary | **Absent.** One-line module docstring is present (L14) and correct, but no `# Dimensions.` block. No `t`/`n`/UPPERCASE scheme: tile extents are `bkv_sz`, `bq_sz`, `bkv_p`; grid extents are `num_bkv`, `num_bq`. Padded extents are inverted — `num_q_heads`/`head_dim` are the *padded* names and the unpadded ones carry the marker (`actual_num_q_heads`, `actual_head_dim`, L442-445), so "the padded one" is the implicit default, which §1 forbids |
| Trailing shape comments | **The one rule it largely keeps** — every ref parameter L67-85 has a shape and states its memory space. Two gaps: format is `# Shape: [...], Memory: X` not `# [shape] dtype`, and **no dtype on any of them, including all four scratch buffers**, which §2 requires "always on scratch" |
| Extents from refs | Mostly kept (L95-108). Bare-index departures: `cache_kv_hbm_ref.shape[-1]` (L96), `cache_kv.shape[-1]` (L591), `bkv.shape[-1]` (L264), `q.shape[0]` (L746, L811, L827, L838), `scores.shape[1]` (L812). `bq_sz`/`bkv_p`/`seq_batch_size` arrive as `functools.partial` statics rather than off the refs — §3's "second source of truth" case, structural rather than incidental |
| jaxtyping | **Absent.** `jax.Array` + trailing comments. Refs inside are bare (that half is kept). The docstring L541-565 is an `Args:` block restating parameter names — §2 shows this verbatim as the ❌ |
| `# Inputs. / # Outputs. / # Scratch.` | **Near-miss:** `# Prefetch` / `# Input` / `# Output` / `# Scratch`, singular and unpunctuated, with a fifth section the reference does not name |
| Body order | **Not followed.** ~290 lines of nested closure definitions (L128-420) sit between the shape constants and any executed statement; execution is three lines at L424-436. No init-guard / masked-store pair in §7's sense — the prologue/epilogue `@pl.when` at L424/L431 is the analogue and sits at the bottom |
| Single-character budget | `s` (L326-338) is the score array — not a loop index, not annotated, not an uppercase extent. `cp` (L154). `i`/`_` are legitimate |
| Memory space in ref names | **Half-applied.** `q_hbm_ref` / `cache_kv_hbm_ref` are right, but in the same signature the four VMEM scratch buffers are `bkv_x2_ref`, `bq_x2_ref`, `bq_weights_x2_ref`, `scores_block_x2_ref` and the six SMEM prefetch refs carry no suffix at all — two spaces (three) sharing one signature is exactly when §5 requires it. The `_x2_` spelling occupies the slot §5 gives to `_scratch` / `_slot_ref`. No `_ptr` anywhere — clean |
| Ref-list length | **17 positional refs** (6 prefetch + 4 in + 1 out + 4 scratch + 1 sem) against §4's "if a kernel's ref list grows past roughly eight, stop adding positional slots and pass a named bundle dataclass instead." The largest single quantitative departure |
| Comments | Body comments are mostly shape restatements (`# Shape: (...)`) rather than compromises. The ones that do name a compromise — L379-380, L566-567, L641-642, L743-744 — are the exception. TODOs are consistently ownerless (L626, L751), which satisfies §8's "pick one" |
| `NotImplementedError` | Two `ValueError`s. L570 (block sizes unspecified) is genuinely a caller error and correct. **L599** — `bkv_sz` must be a multiple of 128 — reads as §9's "the arguments are fine and this backend cannot serve them" and is the one to decide deliberately rather than inherit |
| Four files | **One file holding all four jobs.** `pl.pallas_call` (L704) shares a file with `DEFAULT_VMEM_LIMIT_BYTES` (L28), the `num_sublanes_total` heuristic (L603) and the VMEM sizing (L654-692); the config knobs live in the public signature (L531-537), which is the "no `config` at all" rule inverted. `_get_heuristics_config` does not exist — the heuristic is inline |
| Second axis | Not drawn. Array functions, ref functions and DMA functions are interleaved as closures inside one body |
| Config class | **There is none.** Loose kwargs with defaults; no dataclass, no `__post_init__`, no pydantic |
| Tests | `tests/kernels/test_dsv4_streamindex_topk.py` exists, but there is no `base.py` and so no XLA `_fwd` to test against — the reference lives in the test file (:100-106). Tolerance placement and skip style not inspected; flagged as unchecked rather than asserted |
| Hardware constants | `DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024` (L28) is §13's exact ❌ shape. `pltpu.get_tpu_info()` is never called in this file. Literal `128` appears in *computation* at L108, L445, L598, L604, L639, L684 — §13 permits the literal only "in a shape comment, because there it is documentation rather than a computation" |

---

## Where `code-style.md` and the repo idiom disagree

Reported, not resolved.

1. **The four-file names have zero precedent in this repo.** No `api.py`, no `base.py`, no `pallas_mosaic_tpu*.py` anywhere in `vllm-torchtpu`, and no `Op` base class to subclass. What the repo *does* have, in `deepseek_v4/compress_and_store/` (`kernel.py`, `compute.py`, `config.py`, `buffered_ref.py`, `compress_store_ref.py`, `project_and_save_state_ref.py`) and `experimental/batched_rpa_longctx/` (`wrapper.py`, `kernel.py`, `configs.py`, `utils.py`, `stitch_utils.py`, `schedule.py`), is a package directory following **§10's second table** — the what-may-it-touch axis, `*_ref.py` for the ref-touching modules. So the repo has already adopted §10's second decomposition and never its first. This is the disagreement that most changes the plan's shape and the designer needs it in front of them.
2. **jaxtyping.** §4 requires it on the public wrapper. Exactly one file in the entire repo imports it (`kernels/kimi_k3/chunk_kda.py`); the universal idiom is `jax.Array` plus a trailing shape comment.
3. **pydantic config.** §11 requires "pydantic rather than plain `dataclasses`", frozen + kw_only + slots, justified by autotuning-cache round-tripping. `deepseek_v4/compress_and_store/config.py` uses plain `@dataclasses.dataclass(frozen=True)` (L186, L193, L231) — frozen only, no `kw_only`, no `slots` — and there is no pydantic import under `src/` at all. There is also no autotuning cache in this repo, which is the rule's stated reason.
4. **`_get_heuristics_config` / `_get_autotuning_configs` / `supported_on`.** None of these spellings exist in the repo. Heuristics are inline in the wrapper.
5. **`NotImplementedError`.** §9's rationale is "a backend's fallback chain catches `NotImplementedError` and nothing else." `deepseek_v4/` raises `ValueError` 13 times and `NotImplementedError` once, and there is no XLA fallback path for `streamindex_topk` for the exception type to select. The rule's mechanism is absent here even where its letter can be followed.
6. **Dimension naming.** `bkv_p`, `bq_sz`, `num_bkv`, `page_size_per_kv_packing`, `tile_n` are not §1's scheme — but they are **shared vocabulary across the stack**, not local style: the same names appear in `mla_attention_op.py:119-120`, in the profiled scope names `StreamIdxTC-d-bq_1-bkvp_2` / `StreamIdxTC-m-bq_128-bkvp_2`, and in step 1's own glossary. Renaming them to `tkv`/`tq` breaks the correspondence to the profile. Partial reconciliation already exists: the step-1 report uses `nbq`/`nbkv` for the grid extents, which *is* §1's `n`-prefix form.
7. **Where they agree and the repo is already ahead of this file:** trailing shape comments on every ref parameter are universal across `deepseek_v4/`, and `pltpu.get_tpu_info()` has real precedent (`experimental/batched_rpa_longctx/wrapper.py`, `configs.py`, `stitch_utils.py`, `partial_kv_update.py:132`) — so §13 is a repo practice that `streamindex_topk.py` happens to be outside of, not a novelty being imported.

---

## The split length

**250 lines.** §10, verbatim: *"Split at around 250 lines."*

It appears at the close of the four-file paragraph, immediately after "**The file that decides how big the blocks are is not the file that uses them**", so it reads as the threshold for the `pallas_mosaic_tpu.py` / `pallas_mosaic_tpu_kernel.py` split specifically. It is the only number the file gives.

Calibration for the designer:

- The incumbent is **865 lines in one file — 3.5× the split point.**
- `sparsecore_topk.py` is 457 — 1.8×.
- The four-file plan's `pallas_mosaic_tpu_kernel.py` projects to ~500–650, still ~2×, which is why item 2's second axis is a requirement of this plan and not a refinement of it. **A design option whose kernel file lands under ~250 without invoking the second axis has almost certainly left something out.**
- Repo context for what "past it" already looks like here: `core_attention/mla_swa.py` at 1743, `core_attention/mla.py` at 909, `core_attention/sparse_mla.py` at 892. The decomposed packages sit closer to the line — `compress_and_store/` is nine files, largest 523.
