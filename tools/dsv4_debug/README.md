# DSv4 accuracy debugging tools

Scripts for comparing our DeepSeek-V4 implementation against the
`tpu-inference` JAX reference on the same prompt and config.

## The bug these exist for

On long-form generation we degrade and the reference does not. Same prompt,
same config (`temperature=0`, no repetition penalty, real checkpoint, DP
attention on), 150 tokens:

**Ours** — starts right, then loops until the token budget runs out:

> XLA (Accelerated Linear Algebra) is a domain-specific **language** for
> machine learning, covering key concepts, architectures, and the number of
> the first type of target sub-areas, the first type of target sub-area is a
> sub-area of the first type of target sub-area, and … *(repeats "the number
> of the first type of target sub-area" six more times)*

**Reference** — one clean sentence, correct content, natural stop:

> XLA (Accelerated Linear Algebra) is a domain-specific **compiler** for
> machine learning that uses LLVM (Low Level Virtual Machine) as one of its
> backends to generate optimized machine code for CPUs and GPUs, while LLVM
> is a general-purpose compiler infrastructure that provides code generation
> and optimization for many programming languages.

Note the divergence is visible early — "language" vs "compiler" — well before
the loop starts.

At short lengths the two are **byte-identical** (2- and 16-token runs of the
"needle" retrieval test — a code hidden in filler text that the model must find
and repeat — both pass), so any test with a small `max_tokens` will look green. Use
`DSV4_MAX_TOKENS=150` or more.

## Setup

Paths come from the environment, defaulting to `$HOME/workspace`:

```bash
export WORKSPACE=$HOME/workspace                 # holds both repos + scratch
export VLLM_TORCHTPU_ROOT=$WORKSPACE/vllm-torchtpu
export TPU_INFERENCE_ROOT=$WORKSPACE/tpu-inference
export DSV4_SCRATCH=$WORKSPACE/scratch           # dumps land here
```

`run_ours.sh` and `run_ref.sh` each pin their own venv and
checkpoint path near the top — check those match your machine before the
first run.

## Two ways to compare

### 1. Generated text and logprobs (start here)

Runs both trees on one prompt with no instrumentation, then diffs the output
distributions per generated token.

```bash
./chain_real_generation.sh                       # ~40 min, both trees
python diff_logprobs.py <ours.log> <ref.log>
```

`diff_logprobs.py` prints, per step, each side's top-1 token and its top1−top2
confidence gap, and reports the first step where the argmax flips.

How to read a flip:

- **both gaps small** — a near-tie broken by accumulated numeric error; look
  at precision in the quantized path
- **reference confident, we disagree** — our distribution is structurally
  wrong at that step; a real computation bug

This path is the most trustworthy one available: logprobs come from the real
generating path, so no dumper or hook can distort them.

### 2. Per-layer activation diff

Dumps every layer/sublayer from both trees and compares them.

```bash
./chain_full_dump_v2.sh                          # ~40 min, both trees + diff
python diff_full.py                              # re-run the diff alone
```

`diff_full.py` reports cos / relL2 / norm-ratio per sublayer and the first
divergence. It hard-exits rather than printing a table it can't trust.

## Traps worth knowing about

These each cost real debugging time. They're guarded now, but the reasoning
matters if you extend the tooling.

**Our tree emits a warmup pass; the reference doesn't.** Step indices are not
comparable across trees — our `step000` is a warmup of constant dummy rows,
our `step001` is the real prefill, and the reference's `step000` is *its*
prefill. `diff_full.py` detects prefill per-tree by looking for varying row
norms and prints the step inventory so you can see the warmup. The older
`diff_all.py` had no such check: it built paths containing the literal string
`stepNone`, matched nothing, printed dashes, and reported "FIRST DIVERGENCE:
None" — indistinguishable from a clean result.

**Cosine alone hides magnitude errors.** A pure 1.32x scaling gives
`cos = 1.000000`. Every cell in `diff_full.py` carries cos, relL2 *and*
norm-ratio for this reason.

**Reading a lazy TPU tensor to host is not ordered against the op graph.**
`t.detach().cpu()` copies whatever bytes are at the address, which need not be
the computed result. This manufactured a spectacular false finding — layer 42
appearing to reach 2.884e38 with `inf` downstream — that existed *only* when
dumping was enabled:

| | L42 post_ffn absmax |
|---|---|
| reference | 1.997e4 |
| ours, dumping **off** | 1.856e4 |
| ours, dumping **on** | 2.884e38 |

The tell was in the bits: 95.1% of the "corrupt" values were exact powers of
two with zero mantissas, against 1.2% in a magnitude-matched control —
reinterpreted memory, not arithmetic. Dumps now go through `_dump_to_numpy()`
in `models/vllm/deepseek_v4_patch.py`, which calls
`sync.synchronize(t, wait=True)` first. **This fix has not yet been verified
on a real run** — if you enable dumping and see values near 1e38, that's the
first thing to check.

**Enabling dumps changes the code path.** `DSV4_DUMP_HIDDEN_STATES=1` replaces
`DeepseekV4DecoderLayer.forward`. Don't compare a dumping run against a
non-dumping one and attribute the difference to the model.

## What's already been ruled out

All measured on real-call tensors, so these should not need redoing:

- **fp8 linear load path** — clean at every layer including 42; no scale
  anywhere retains the `create_weights` sentinel (`DSV4_DEBUG_LINEAR_SCALES=1`)
- **shared-expert MLP** — healthy on all 8 ranks, max output 330
  (`DSV4_DEBUG_MLP_STAGES=1`)
- **MoE combine and all-reduce** — finite and identical across all 8 ranks
  (`DSV4_DEBUG_MOE_COMBINE=1`)
- **our GMM kernel does no internal reduction** — so upstream's single late
  all-reduce is the correct contract; there is no double-reduce
- **routing** — `topk_weights` row-sums are 1.499–1.501, matching
  `renormalize` x `routed_scaling_factor` 1.5
- **`routed_scaling_factor`** — applied exactly once in each tree, just in
  different places
- **swiglu clamp and fp4 requant math** — identical to the reference
  (limit 10.0; same scale/clip/round, block size 512)

Note the `routed_scaling_factor` experiments from 2026-08-08 05:23–06:49 all
predate the zero-weight MoE fix (`665006c8`, 13:48 the same day) and are void.

## Debug switches

All no-ops unless set. See the commit adding them for detail.

| var | what it prints |
|---|---|
| `DSV4_DEBUG_LINEAR_SCALES=1` | fp8 linear scale shape/range/sentinel count per layer |
| `DSV4_DEBUG_MLP_STAGES=1` | shared-expert MLP at `x -> gate_up -> act -> out` |
| `DSV4_DEBUG_MOE_COMBINE=1` | each step of the MoE runner combine + the final `.view()` |
| `DSV4_DEBUG_MOE_ROWS=1` | per-row routed output for specific token rows |
| `DSV4_DEBUG_CACHE_SHAPES=1` | per-layer KV cache shape and `data_ptr` |
| `DSV4_DUMP_HIDDEN_STATES=1` | per-sublayer activation dumps (see traps above) |
| `DSV4_DUMP_LAYERS=40,41,42` | narrow dumping to specific layers |

## Prefill-vs-decode differentials (added 2026-08-12)

Born from the decode-corruption root cause
(`docs/dsv4_decode_race_postmortem.md`). Run these FIRST when decode output
looks worse than prefill-scored output — they need no instrumentation
beliefs, only generated logprobs and cache bytes.

| tool | cost | what it answers |
|---|---|---|
| `chain_prefill_decode.sh [prompt] [max_tokens]` | 2 model loads | does decode-built state diverge from prefill-built state, behaviorally? Prints per-step drift + verdict via `diff_teacher_forcing.py` |
| `chain_cache_diff.sh [prompt] [max_tokens] [layer]` | 2 loads (1 harvest + 1 combined) | WHICH cache rows decode writes differently, byte-for-byte, via `diff_cache_pages.py` (prefill is bit-deterministic, so byte-equality is a valid assertion) |

Underlying harness features (examples/run_dsv4_inference.py):
`DSV4_TF_FILES` (growing-prefix teacher forcing), `DSV4_TF_AFTER_PROMPTS`
(decode + TF scoring on one model load), `DSV4_DUMP_PAGES` +
`DSV4_DEBUG_LAYER` + `DSV4_DUMP_LABEL` (block-table-indexed page dumps with
synced activation reads).

Rules learned the hard way (full list in the postmortem): all unsynced host
reads of lazy tensors lie — use `_dump_to_numpy`; never characterize a
divergence from one side's logprobs; audit any tensor whose only consumer
is a cache write.
