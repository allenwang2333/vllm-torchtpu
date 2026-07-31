# DSV4 Bug-Hunt Prompt: Subtle Small-Number Bugs and Missing Components

**Created:** 2026-07-28. Companion to `docs/dsv4_architecture_and_rationale.md` (the
correctness investigation) and `docs/dsv4_pr_splitting_plan.md` (the PR-splitting
workstream) — this doc is neither; it's a reusable **prompt** to hand to a fresh
agent/tool session when you want a targeted audit for a specific class of bug this
codebase has repeatedly produced, rather than a general review.

Built from concrete bug patterns already found in this investigation, not generic
static-analysis advice. Paste the section below into a fresh session (Claude Code,
another reviewer, whatever tool) when you want this specific audit run.

---

## The prompt

You're auditing `vllm-torchtpu`'s DeepSeek-V4 TPU integration
(`src/vllm_torchtpu/kernels/deepseek_v4/`,
`src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`,
`src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`) for subtle bugs — specifically the
class of bug that doesn't crash, doesn't NaN, and doesn't fail existing tests, but
silently produces wrong output. This codebase has a documented track record of exactly
this bug shape (see below), so treat every instance of these patterns as a real lead, not
a style nit.

**Do not trust comments, docstrings, or existing test assertions at face value — compute
what the code actually evaluates to given real runtime config values, and verify by
tracing, not by reading intent.** Where `~/workspace/tpu-inference` has an equivalent
file, diff against it directly, but don't assume tpu-inference is bug-free either —
verify its formulas against its own real runtime config too, the same way, since it
shares provenance with this code and has already been found to carry at least one of the
same latent bugs (see pattern 1).

**Pattern 1 — chained integer division degenerating to 0 or 1.** Precedent:
`VllmCompressorStateCache.block_size` was computed as `cache_config.block_size //
compress_ratio // 4 // 2 // coff`. For real runtime `cache_config.block_size` values,
this silently floors to `1` for HCA layers (`compress_ratio=128`) instead of the intended
4-16 — and downstream, `state_block_size=1` makes one token's cache write span an
*entire physical page*, clobbering neighboring tokens' data. The bug existed unnoticed
for 7+ sessions because the exact config regime that triggers it (long context, crossing
a compression boundary) was never exercised.

Search for: every `//` chain of 2+ divisions anywhere config-derived values
(`block_size`, `page_size`, `compress_ratio`, `num_heads`, `head_dim`, `max_model_len`,
`sliding_window`) feed into cache addressing, slot mapping, or buffer geometry. For each
one, don't just read it — compute what it evaluates to for the actual config values this
checkpoint/setup uses (grep launch scripts/docs for real `block_size`, `max_model_len`,
etc.), and flag any result that's `0`, `1`, or otherwise much smaller than whatever
comment/adjacent code implies it should be.

**Pattern 2 — constants/formulas borrowed from a different physical layout without
re-derivation.** Precedent: vLLM's upstream CUDA base class hardcodes `block_size = 4`
(CSA) / `8` (HCA), derived from a fixed CUDA `[64, 584]` page-row assumption. TPU's
actual page geometry is completely different (and itself varies — 16 to 1024+ rows
depending on backend/config) and no one re-derived whether those constants, or the
formula meant to replace them, are actually valid for TPU's shape.

Search for: any hardcoded numeric constant (block sizes, head counts, alignment/padding
values, shape dims) sitting near a comment referencing GPU/CUDA/AMD/"reference"/upstream,
or copied verbatim from `tpu-inference` or vLLM's native path. For each, check whether
the assumption it encodes (a specific page/row/block shape) actually holds for this
repo's TPU cache layout, and whether it's been validated at more than one config (small
vs. large page size, short vs. long context).

**Pattern 3 — missing modulo/wraparound on circular or sliding-window index
arithmetic.** Precedent: `state_cache_block_num`/`k_cache_block_num` were computed
without `% max_num_blocks_per_req`, so any position past a request's allocated block
count silently aliased into the *next* request's block-table slice — a real,
previously-shipped bug (fixed by adding the modulo on both the write side and
`gather_state_windows`'s read side; fixing only one side would've created a new
mismatch).

Search for: every place a token position or block index is computed via `position //
block_size` or similar for a spec described as `SlidingWindowMLASpec`/circular/
ring-buffer, and confirm there's a matching `% max_blocks_per_req` (or equivalent bound)
on **both** the write path and any corresponding read path. A wraparound fixed on only
one side is itself a bug.

**Pattern 4 — wrong rank/parallelism-axis selection.** Precedent: attention-sink slicing
used `get_tensor_model_parallel_rank()` (always `0` under this model's TP=1/EP=8 config)
instead of `dist.get_rank()`, so every one of 8 workers loaded the identical sink slice
instead of its own — silent numerical corruption, full run completed with no crash.

Search for: every use of a rank/parallelism accessor (`get_tensor_model_parallel_rank`,
`get_ep_rank`, `dist.get_rank`, `mesh.local_rank`, etc.) near a slicing/indexing
operation. For each, confirm the rank source actually corresponds to the axis the tensor
is partitioned along (check the tensor's actual sharding, not just what the variable name
near it implies) — especially anywhere TP size and EP/DP size differ (TP=1 makes
TP-rank-based slicing a silent no-op, which is exactly what happened).

**Pattern 5 — silent fallback/default masking a real error.** Precedent:
`swa_cache_layer.block_size` had grown a silent `getattr(..., 16)` fallback for the case
where the real value was unavailable — meaning a wrong guessed page size would
misaddress every SWA cache slot with no error, ever. (Already fixed once by making it a
loud `RuntimeError`; check nothing has regressed this or added a similar fallback
elsewhere.)

Search for: every `getattr(obj, name, <numeric-default>)` or `x if x is not None else
<numeric-default>` pattern anywhere near cache addressing, block-size, or shape
computation. Any numeric fallback in this territory should be treated as a bug unless
there's a clear, checked argument for why that specific default is always safe.

**Pattern 6 — untested config regime / code path never actually exercised.**
Precedent: every full-model run for 7 sessions used `max_model_len=64` (a hardcoded cap
in `examples/run_dsv4_inference.py`, not a deliberate choice), so the HCA
`compress_ratio=128` compression boundary — and the block_size=1 bug living behind it —
was literally unreachable until session 8 deliberately extended the context. Similarly,
`tpu-inference`'s own kernel test (`compressor_test.py`) hardcodes
`state_block_size=4/8` directly rather than going through the real wiring code, so its
own test suite never exercises the value the wiring path actually computes at runtime.

For every `compress_ratio`/config-conditional branch in the compressor, indexer, MLA, and
SWA code, identify: (a) what real config values or context lengths are required to
activate it, and (b) whether any existing test or documented full-model run actually
reaches that condition. Flag every branch where the answer to (b) is "unclear" or "no" —
these are exactly where bugs like pattern 1-4 hide indefinitely.

**Pattern 7 — "missing component" check.** Separately from arithmetic bugs: identify
anywhere vllm-torchtpu's DSV4 wiring layer (`custom_ops/deepseek_v4.py`,
`deepseek_v4_patch.py`) calls a kernel or config path with fewer/different arguments than
`tpu-inference`'s equivalent call, or skips a step `tpu-inference` performs (e.g., a
normalization, a clamp, a cache write) — these show up as silent omissions rather than
wrong values, and are easy to miss with value-focused review alone. Cross-reference
function signatures and call sites directly, not just behavior.

**Output**: for each finding, give the exact file/line, the concrete formula or constant
involved, what it evaluates to under the real config this repo actually runs (not a
hypothetical), and — critically — whether it's currently reachable by any existing test
or documented run (per pattern 6). Deprioritize anything that's syntactically suspicious
but demonstrably unreachable or already covered by a passing numerical-correctness test;
prioritize anything reachable-but-untested.
