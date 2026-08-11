# DeepSeek-V4 on vllm-torchtpu — status

Last updated 2026-08-11. This is the current state of the DeepSeek-V4 bring-up
on TPU, and the only status doc — earlier handoffs and investigation logs were
folded in here and removed, so there's one place to look rather than five that
disagree. `docs/tpu_kv_cache_sharing.md` goes deeper on one specific topic;
`tools/dsv4_debug/README.md` covers the debugging tools.

## Start here

**What this is.** We're running DeepSeek-V4-Flash — a 43-layer
mixture-of-experts model — on TPU through vLLM, using our own attention and
MoE kernels. "We" is `vllm-torchtpu`, the PyTorch/TPU backend.

**Why there's a reference implementation.** `tpu-inference` is a separate,
JAX-based Google stack that already runs this model correctly. We treat it as
the spec: when our output and its output differ, we're the ones who are wrong.
Most of the debugging described here is a comparison against it.

**Where things stand.** The model loads, runs, serves, and is reasonably fast.
It answers short questions correctly — byte-identically to the reference. But
on longer generation it goes off the rails and starts repeating itself, where
the reference produces a clean answer. We have not found the cause. That's the
one blocking problem; §8 has the details.

**If you're picking this up cold,** read §2 (how the model works), §5 (how to
run it), then §8 (the bug) and §13 (what to try next). Sections 9-11 exist so
you don't repeat work that's already been done, or trust tooling that has
produced confident wrong answers more than once.

---

## 1. The model

DeepSeek-V4-Flash, 43 layers, from
`hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash`.

| property | value |
|---|---|
| hidden_size | 4096 |
| num_hidden_layers | 43 |
| moe_intermediate_size | 2048 |
| n_routed_experts / experts_per_token | 256 / 6 (each token goes to 6 of 256 experts) |
| n_shared_experts | 1 (a dense MLP every token passes through, on top of the 6) |
| scoring_func | `sqrtsoftplus` |
| norm_topk_prob | True |
| routed_scaling_factor | 1.5 |
| swiglu_limit | 10.0 |
| num_hash_layers | 3 (layers 0-2 route by token id, not by gate) |
| topk_method | `noaux_tc` (layers 3+) |
| quantization | fp8 e4m3 weights in 128x128 blocks, power-of-two (ue8m0) scales |
| num_nextn_predict_layers | 1 (multi-token prediction draft head; our loader skips it) |

A few structural things that trip people up when reading dumps:

- Hyper-connections (MHC): the residual stream is 4 parallel streams
  (`hc_mult=4`), mixed by `hc_pre`/`hc_post` around each of attention and FFN,
  and collapsed by `hc_head` at the end. Tensors are `[tokens, 4, 4096]` at
  stream points and `[tokens, 4096]` between them. Several axes share small
  plausible sizes — confusing the `hc_mult` axis with a length axis has
  produced at least two refuted findings historically.
- Indexer/compressor on even layers only (2, 4, …, 42). So L41 is *not*
  a like-for-like control for L42; L40 is.
- Attention is MLA with a compressed KV path (CSA ratio 4, HCA ratio 128)
  plus a sliding-window branch (window 128).

## 2. How the model works

Four things make DSv4 different from a normal MoE transformer, and all four
are places we had to write custom code.

**MLA with a compressed KV path.** Attention is multi-head latent attention:
KV is projected down to a latent (`wkv`), cached compressed, and projected
back up per head. On top of that DSv4 splits attention into two branches per
layer:

- a *sliding-window* branch over the last 128 tokens, at full resolution
- a *main* branch over the whole sequence, at reduced resolution — tokens are
  compressed in groups (CSA compresses 4:1, HCA 128:1), so a long context
  costs a fraction of the rows

The two branches are merged with an online-softmax combine, plus learned
attention sinks. The compressed rows live in a byte-packed cache record:
`[fp8 nope | bf16 rope | ue8m0 block scale]`, 640 bytes per token. Anything
reading that cache must unpack it — the raw bytes cast to float look like
plausible numbers and are not.

**A learned indexer selects which compressed rows to attend.** Each even layer
runs a small scoring network (`indexer`) over 64 index heads, then takes the
top-`index_topk` (512) rows. At short context this selects everything, so
indexer bugs are dormant below ~2048 tokens — that dormancy has masked real
divergences before.

**Hash routing for the first 3 layers.** Layers 0-2 don't use the gate at all;
they route each token to 6 experts by a fixed `token_id -> experts` table
(`tid2eid`). Layers 3+ use `sqrtsoftplus` scoring with a `noaux_tc`
correction bias. Both paths renormalize the top-6 weights and then scale by
`routed_scaling_factor` 1.5.

**Hyper-connections instead of a single residual stream.** See the note above:
4 streams, mixed before and after each sublayer, collapsed at the head.

## 3. What's implemented here, and where

| file | lines | what it is |
|---|---|---|
| `kernels/deepseek_v4/mla.py` | 1091 | main-branch MLA attention kernel (compressed rows) |
| `kernels/deepseek_v4/mla_swa.py` | 1367 | sliding-window branch + the merge with the main branch |
| `kernels/deepseek_v4/compress_norm_rope.py` | 481 | packs/unpacks the compressed KV record; norm + rope on the way in |
| `kernels/deepseek_v4/streamindex_topk.py` | 816 | indexer top-k selection |
| `kernels/deepseek_v4/compressor.py` | 173 | KV compression |
| `layers/.../deepseek_v4_attention.py` | 1705 | the attention module: metadata, cache binding, kernel dispatch |
| `layers/.../deepseek_v4_compressor.py` | 742 | compressor module + cache aliasing |
| `layers/.../deepseek_v4_indexer.py` | 347 | indexer module |
| `layers/vllm/quantization/deepseek_v4_fp8.py` | 102 | quant-method dispatch (fp8 linears, fp4 or fp8 experts) |
| `models/vllm/deepseek_v4_patch.py` | 1744 | monkey-patches vLLM's DSv4 model onto TPU: weight loading, layer/head compile wrapping, instrumentation |
| `models/vllm/deepseek_v4_requant.py` | 93 | attention weight requantization |

Note the shape of that: we do **not** have our own model definition. We take
vLLM's `vllm/models/deepseek_v4/amd/model.py` and patch it — swapping the
attention class, rewriting the weight loader, extracting a head module. The
reference took the other approach and forked its own copy
(`tpu_inference/models/vllm/experimental/deepseek_v4.py`). That difference is
why our diffs against the reference are structural rather than line-by-line.

## 4. What's upstreamed, what isn't

Merged to `vllm-torchtpu` main:

| PR | content |
|---|---|
| #138 | initial DSv4 scaffolding |
| #184 | streamindex_topk kernel |
| #202 | MLA-SWA kernel |
| #203 | MLA kernel |
| #204 | fp8 DS-MLA |
| #214 | streamindex_topk update |
| #267 | kernel fixes to compressor and MLA-SWA |
| #274 | FP8/FP4 quantization plumbing |
| #305 | DSv4 custom MoE routing behind gates |

So the **kernels are largely upstreamed**. What is *not* upstream is most of
the glue: the model patch, the KV-cache aliasing, the compile scaffolding, the
debug tooling. `main` itself carries 62 commits that have never reached
`origin/main` — it is published as `origin/patemotter/dsv4-lean-main-e2e`, not
merged. Anything built on `main` inherits that.

Outstanding, roughly in order of how much it blocks others:

- the accuracy bug in §8 — blocks any acceptance number
- the model patch (`deepseek_v4_patch.py`) — 1744 lines of monkey-patching
  that should become either an upstream model or a registered OOT model
- KV-cache aliasing (`_alias_ds_v4_auxiliary_caches`) — works, but it is a
  runner-level tensor alias, which reviewers on PR #322 already flagged
- compile scaffolding — on this branch, unreviewed
- MMLU-Pro number — never produced

## 5. Running it

`examples/run_dsv4_inference.py` is the program that actually loads the model
and generates. You rarely call it directly: it needs a dozen environment
variables set correctly, several of which are load-bearing and easy to get
subtly wrong. `tools/dsv4_debug/run_ours.sh` wraps it with a known-good
environment, and `run_ref.sh` does the same for the tpu-inference reference in
its own virtualenv. Both take a log path, then any `VAR=value` overrides.

```bash
# ours, one prompt, 150 generated tokens
tools/dsv4_debug/run_ours.sh /tmp/out.log \
    DSV4_MAX_TOKENS=150 \
    DSV4_PROMPT_FILES=tools/dsv4_debug/prompts/xla_llvm_prompt.txt

# the reference, same prompt and config
tools/dsv4_debug/run_ref.sh /tmp/ref.log \
    DSV4_MAX_TOKENS=150 \
    DSV4_PROMPT_FILES=tools/dsv4_debug/prompts/xla_llvm_prompt.txt
```

A run takes roughly 15-20 minutes, most of it loading and requantizing the
148 GiB checkpoint. Both trees need the whole 8-chip TPU, so they can't run at
once — the chain scripts in `tools/dsv4_debug/` serialize them for you.

Our configuration: TP=8 (tensor parallel across 8 chips), EP=8 (each chip owns
32 of the 256 experts), MoE expert weights requantized to fp4 in blocks of
512, activations quantized to fp8 inside the grouped matmul — "W4A8" below.

The reference runs the same checkpoint and the same `fp4/512` quantization,
but its parallelism differs — it runs **attention DP=8** while we
run **TP attention**. tpu-inference hard-rejects MLA without DP attention, so
a DP-off reference control is impossible and *every* ours-vs-ref comparison
carries this confound.

### Paths and environment

Nothing here hardcodes a home directory. Scripts and dump locations resolve
from environment variables, with sensible defaults:

| variable | default | what it's for |
|---|---|---|
| `WORKSPACE` | `$HOME/workspace` | holds both repos, `hf_cache`, `scratch` |
| `VLLM_TORCHTPU_ROOT` | `$WORKSPACE/vllm-torchtpu` | this repo |
| `TPU_INFERENCE_ROOT` | `$WORKSPACE/tpu-inference` | the reference stack |
| `DSV4_SCRATCH` | `$WORKSPACE/scratch` | where dumps and logs land |
| `DSV4_DUMP_ROOT` | `$WORKSPACE/scratch` | overrides just the debug dump location |
| `VLLM_VENV` | `$HOME/uv_venv` | venv used by the benchmarking scripts |

If your layout matches `$HOME/workspace/{vllm-torchtpu,tpu-inference,hf_cache}`
you shouldn't need to set anything. Otherwise set `WORKSPACE` and the rest
follow.

### Larger runs: serving, accuracy, throughput

The single-prompt wrappers above are for debugging. For anything at scale, use
`scripts/dsv4/`, which shares one env file (`dsv4_env.sh`) so the settings stay
consistent between them.

**Serving.** Starts a vLLM server on this node and waits for health, then
leaves it running:

```bash
scripts/dsv4/serve_dsv4.sh server.log
# tunables, all overridable per run:
MAX_MODEL_LEN=9216 GPU_MEM_UTIL=0.7 scripts/dsv4/serve_dsv4.sh
```

Known-good shape is 9216 context, `max_num_batched_tokens=1024`,
`max_num_seqs=128`, TP=8/EP=8, fp8 KV, prefix caching off, non-eager. That is
the reference's shape, which matters if you intend to compare numbers.

What "known-good" covers: it serves, the DSv4 runner unit tests pass, and the
weight-cache tests pass. What it does **not** cover is that output is correct
at this configuration. The needle test that validated the MoE fix ran at
`max_model_len=1024`, eager, offline, single-sequence — this shape changes all
four. Any accuracy claim from a served run needs the needle control passing at
*that same* shape first, or it's measuring an unknown.

**Accuracy (MMLU-Pro).** Brings the server up, runs `lm_eval`, tears it down:

```bash
scripts/dsv4/run_mmlu_pro.sh my_run
# smoke test first -- the full benchmark is long:
LIMIT=10 scripts/dsv4/run_mmlu_pro.sh smoke
# knobs: NUM_CONCURRENT=128, NUM_FEWSHOT=5, GPU_MEM_UTIL, MAX_MODEL_LEN
```

Note what this does and doesn't measure. MMLU-Pro is multiple choice, so each
question produces a short answer — it aggregates over thousands of prompts and
will catch a *systematic* accuracy gap, but it does not exercise long
generation and so will not reproduce the repetition failure in §8. The two are
complementary: MMLU-Pro tells you whether we're worse on average, the prompt
pool tells you how we fail.

**Throughput.** One model load, several batch sizes:

```bash
PYTHONPATH=src python scripts/dsv4/perf_offline.py
# DSV4_PERF_BATCHES="1,8,32,128", DSV4_PERF_PROMPT_TOKENS=512,
# DSV4_PERF_GEN_TOKENS=64, DSV4_MAX_MODEL_LEN=4096
```

This exists because batch-1 serving showed ~1 tok/s and it wasn't obvious
whether that was real per-step cost or an artifact of running one sequence
through a graph compiled for `max_num_seqs=128`. It scales with batch size, so
it was the artifact — don't read batch-1 numbers as throughput.

**Finding a serving config.** `scripts/dsv4/search_serving_config.sh` sweeps
context/utilisation combinations to find ones that fit in HBM, which is worth
running before assuming a shape is unreachable.

Two things about memory that will otherwise cost you an afternoon:

*vLLM's DSv4 KV budget check over-estimates by ~62x.* `--max-model-len=9216`
is refused with "42.86 GiB KV cache is needed, larger than the available 37.64
GiB". Summing what vLLM bills each spec for one max-length request
(`spec.max_memory_usage_bytes`) gives 0.69 GiB — a constant 62.1x multiplier,
at 1024 context as well as 9216. Only the context length moves it across the
available-memory line, which is why 1024 starts and 9216 doesn't. This is a
faulty check, not real memory pressure; `GPU_MEM_UTIL` is raised to get past
it.

*`GPU_MEM_UTIL=0.67` is a floor, not a preference.* Below it there isn't
enough HBM left for compile and activation buffers and you get
`RuntimeBufferAllocationFailure`. 0.67 clears it with ~31 GiB spare. If the
weights or context change, recompute the floor rather than nudging the
number.

## 6. What works

- End-to-end generation on real weights, eager and compiled.
- Short prompts retrieve correctly. The "needle" test — a code buried in a
  block of filler text, which the model has to find and repeat back — passes
  byte-identically against the reference at 2 and 16 generated tokens, on both
  an 81-token and a 256-token prompt. The 256-token version matters because the
  code sits outside the 128-token sliding window, so retrieving it requires the
  compressed KV path to work.
- torch.compile, after the fixes on this branch: 43 layers, zero
  e0102/e0101 across the full compile sweep. Engine init 1392 s of which
  1042 s is compilation; artifact cache ~89 GiB.
- Weight loading, ~35-40 min → under a minute, via
  `safetensors_load_strategy=prefetch` plus a warm page cache. vLLM only
  auto-enables prefetch for network filesystems and our checkpoint is on local
  ext4, so it never engaged by default.
- Throughput (2026-08-09 baseline, after the host-sync fix): 1.27x slower
  than tpu-inference at batch 1, 3.34x at batch 128; we flatten after ~32.

### Serving

Serves at the reference's shape with real weights: 9216 context,
`max_num_batched_tokens=1024`, `max_num_seqs=128`, TP=8, EP=8 (32/256 experts
per rank), fp8 KV, prefix caching off, non-eager. Before 2026-08-09 it died in
`initialize_kv_cache` at any context past ~1024.

## 7. Bugs already found and fixed

Several of these were misdiagnosed more than once before landing. What's
listed is the final answer, not the first one.

| date | symptom | root cause |
|---|---|---|
| 08-01 | needle retrieval failed once the code sat outside the sliding window; HCA/CSA suspected for weeks | `is_unused=True` suppressed SWA's own `kv_cache_spec` for CSA/HCA layers — a vllm-torchtpu-only deviation — aliasing its `attn_metadata` onto main's and colliding compressed-KV rows |
| 08-08 | every DSv4 correctness result to that point was invalid | `_dequantize_and_pad` dequantized from `w1_temp`/`w3_temp`/`w2_temp`, which nothing ever writes. Routed MoE output was exactly 0.0 on all layers and ranks — the model ran shared-expert-only, missing 6 of 7 expert contributions per token. Fix: dequantize from the loaded params, w13 split as **contiguous** halves (contiguous cos 0.986 vs interleaved 0.623) |
| 08-08 | server died at 9216 context | we allocated one KV array per layer where vLLM budgets one per `KVCacheTensor` (~2.08x over) |
| 08-09 | KV overlay produced wrong shapes | `_alias_ds_v4_auxiliary_caches` ran *after* allocation, so overlay caches were allocated first and aliased afterwards |
| 08-06 | "coherent but repeating" output | harness bug — `scratch/run_e2e.py`'s `LIBTPU_INIT_ARGS`/`MOE_REQUANTIZE` env vars, not model code. Use `examples/run_dsv4_inference.py` |
| 08-10 | e0102 `RuntimeProgramInputMismatch` under torch.compile | two stacked bugs: `support_torch_compile` passes `compile_prefix=""` for non-encoder classes so all 43 layer instances shared one artifact cache, and `VLLM_USE_AOT_COMPILE` compiled against a different input signature than runtime |

Four of those six were instrumentation or harness bugs wearing a model bug's
clothes. Worth keeping in mind as a base rate before committing to a theory.

## 8. The open bug

Long-form generation degrades. 150 tokens, `temperature=0`, no repetition
penalty, identical prompt and checkpoint. Reproduced 2026-08-11 with **zero
instrumentation active**.

Prompt: *Explain the difference between XLA and LLVM in one sentence.*

**Ours** — starts correctly, then loops until the budget is exhausted:

> XLA (Accelerated Linear Algebra) is a domain-specific **language** for
> machine learning, covering key concepts, architectures, and the number of
> the first type of target sub-areas, the first type of target sub-area is a
> sub-area of the first type of target sub-area, and the second type of
> target sub-area is the sub-area where the target object is located, and the
> number of the first type of target sub-area, *(repeats ~6 more times)*

**Reference** — one clean sentence, correct content, natural stop:

> XLA (Accelerated Linear Algebra) is a domain-specific **compiler** for
> machine learning that uses LLVM (Low Level Virtual Machine) as one of its
> backends to generate optimized machine code for CPUs and GPUs, while LLVM
> is a general-purpose compiler infrastructure that provides code generation
> and optimization for many programming languages.

Two things make this easy to miss. The divergence shows up early — "language"
vs "compiler" — well before the repetition starts. And at 2-16 tokens the two
trees are byte-identical, so any test with a small `max_tokens` comes back
green. Use 150 or more.

### Where it actually goes wrong

Reading our own per-step logprobs, the split happens at **step 13**:

```
step 13  gap=1.25   ' language':-0.25, ' compiler':-1.50, ' programming':-3.88
```

The reference's word is our rank 2, 1.25 logprob behind — roughly 29%
relative odds. Steps 0-12 are ordinary, several of them very confident
(gaps of 10-19).

So this is not slow accumulation over 150 tokens. One near-tie at token 13
goes the other way, and everything downstream — the drift into "target
sub-areas," the repetition — follows from that single flip. The
"long-generation degrades" framing is wrong; the right framing is that a
small numerical difference meets an early coin-flip and gets amplified.

That has a practical consequence: **this prompt is a bad instrument.** A
1.25-gap step will flip under almost any perturbation, so it measures luck
as much as correctness. Localizing needs prompts where the model is
confident and we still diverge.

### W4A16 ablation result (2026-08-11)

Forcing bf16 activations in the MoE GMM (`DSV4_MOE_W4A16=1`) does **not**
fix it. Output is different garbage, still looping:

> XLA (Accelerated Linear Algebra) is a domain-specific **language** for the
> WebGPU API. Specifically, what is the relationship between the two words
> "sight" and "site" … *(loops)*

Note it still picks "language" at step 13. Two different activation
precisions land on the same side of that near-tie, which argues the
preference is systematic rather than a coin flip we happen to lose. Read
this as weak evidence — the ablation was set up before the near-tie was
understood, and a small-gap flip can't cleanly confirm or refute much.

## 9. Already ruled out

All of these were measured on real-call tensors rather than through the
dumper, so they shouldn't need redoing.

| area | evidence |
|---|---|
| fp8 linear load path | `sentinel_elems=0` at every probed layer and rank, including L42; scales healthy (6e-5..1e-3), shapes self-consistent at TP=8 |
| shared-expert MLP | max output 330 across every rank, layer and call; activation saturates at exactly 100 = limit², as designed |
| MoE combine / all-reduce | finite and identical on all 8 ranks (L42: 18560) |
| double all-reduce | `fused_moe_func` has no `psum`/`all_reduce`; it returns per-rank partials, so upstream's single late all-reduce is the correct contract |
| routing | `topk_weights` row sums 1.499-1.501 = renormalize x 1.5 |
| `routed_scaling_factor` | applied exactly once in each tree, in different places (ours folds it into topk weights; ref applies it in the runner) |
| swiglu clamp | limit 10.0 in both; on TPU `SiluAndMulWithClamp` correctly resolves to `forward_native` (our platform is `PlatformEnum.OOT`) |
| fp4 requant math | same formula, axis and block size (512) as the reference |
| MoE zero-weight bug | fixed 2026-08-08 (`665006c8`); was dequantizing from never-written `w*_temp` buffers |

**Void, do not cite:** the `routed_scaling_factor` experiments of 2026-08-08
05:23-06:49 all predate the zero-weight MoE fix (13:48 the same day) and were
run on a model with no routed experts.

## 10. Findings that turned out to be wrong

A lot of the activation-level analysis turned out to be measuring the
instrument rather than the model. Writing it down so nobody builds on it
again.

**The "layer 42 blowup" was not real.** Dumps showed L42's FFN output at
2.884e38 with `inf` downstream. Controlled comparison, same prompt and commit,
one variable:

| | L42 post_ffn absmax |
|---|---|
| reference | 1.997e4 |
| ours, dumping **off** | 1.856e4 (ratio 0.93) |
| ours, dumping **on** | 2.884e38 |

Cause: `t.detach().cpu()` on a lazy torch_tpu tensor is not ordered against
the deferred op graph, so it reads unmaterialized bytes. The tell was in the
bits — 95.1% of the "corrupt" values were exact powers of two with zero
mantissas, against 1.2% in a magnitude-matched control. Fixed by routing all
dumps through `_dump_to_numpy()`, which calls `sync.synchronize(t, wait=True)`
first. **The fix is not yet verified on a real run.**

**Consequently void:** every ours-vs-ref activation number from that tooling,
including "first cos<0.99 at L7", "`02_post_attn` noisiest from L2", and the
L40/L41 cosine drift.

**Also void:** the "cos ~0.986 / norm ratio 1.32" residual that
`dsv4_kv_cache_unification_worklog.md` lists as still open, along with the
2026-08-11 attempts to explain it. It was measured with `diff_all.py`, which had two
silent failure modes: it could not detect our warmup pass (built paths
containing the literal string `stepNone`, matched nothing, and printed
"FIRST DIVERGENCE: None"), and it compared cosine only, which reads
`1.000000` for a pure magnitude error.

## 11. Ways the tooling can mislead you

Each of these cost a day or more. They're guarded now, but the reasoning
matters if you extend the tooling.

1. **Our tree emits a warmup pass; the reference does not.** Step indices are
   not comparable across trees. Ours: `step000` = warmup (constant dummy
   rows), `step001` = prefill. Reference: `step000` = prefill.
   `diff_full.py` detects prefill per-tree and prints the inventory.
2. **Cosine alone hides magnitude errors.** A pure 1.32x scaling gives
   `cos = 1.000000`. Always carry norm-ratio too.
3. **Host reads of lazy TPU tensors must be synchronized** (section 6).
4. **`DSV4_DUMP_HIDDEN_STATES=1` replaces `DeepseekV4DecoderLayer.forward`.**
   Never compare a dumping run against a non-dumping one and attribute the
   difference to the model.
5. **The default test harness caps context at 64 tokens**, which makes the
   compressed KV path unreachable. State `max_model_len` with any correctness
   claim.
6. **`main_cache_kv` is a byte-packed record**, not float features — decode
   via `unpack_sparse_kv_cache` before any numeric comparison.

## 12. Tooling

`tools/dsv4_debug/` (committed):

| file | purpose |
|---|---|
| `chain_real_generation.sh` | both trees, one prompt, no instrumentation |
| `diff_logprobs.py` | per-token top-k logprob diff; read the `ours@ref` column, not the flip step |
| `chain_full_dump_v2.sh` | full per-layer activation dump, both trees |
| `diff_full.py` | per-sublayer cos / relL2 / norm-ratio; hard-exits rather than printing a table it cannot trust |
| `run_ours.sh`, `run_ref.sh` | wrappers that run our tree / the reference with a matched environment |
| `prompts/` | the prompts everything was characterized on |

Debug switches, all no-ops unless set: `DSV4_DEBUG_LINEAR_SCALES`,
`DSV4_DEBUG_MLP_STAGES`, `DSV4_DEBUG_MOE_COMBINE`, `DSV4_DEBUG_MOE_ROWS`,
`DSV4_DEBUG_CACHE_SHAPES`, `DSV4_DUMP_HIDDEN_STATES`, `DSV4_DUMP_LAYERS`,
`DSV4_MOE_W4A16`.

## 13. Where to look next

Roughly in order of what's cheapest and what narrows things down most.

**1. W4A16 ablation** — `DSV4_MOE_W4A16=1`. Running as of this writing.

The reference runs the *same nominal format* (fp4 experts, block 512, fp8
activations) and is clean, so this is **not** a precision-sufficiency test —
fp4/512 is already known to be enough. It is an implementation bisection:

- rambling stops → our quantized-MoE implementation loses more than the
  reference's at equal format. Hunt the activation-quant scale/axis and the
  ours-only pack path (`quantize_native_fp4_kmajor`, and the `should_unpack`
  bitcast in `gmm_v2.py:894` which the reference's `gmm_v2` does not have).
- rambling persists → quantized MoE is behaviorally exonerated end to end,
  and **attention** becomes the prime suspect. It is the only major subsystem
  with no trustworthy elimination, and it is where this repo's entire bug
  history lives.

**2. Logprob curve.** Read `ours@ref` — our logprob for the token the
reference picked — across all steps, not the flip location. The flip is a
threshold crossing; the divergence exists before it. Smooth decay means
accumulating error. A sharp cliff means check the step index against
structural boundaries (window 128, CSA ratio 4, page/packing 64); a
boundary-correlated cliff would be the most localizing observation available.

**3. If MoE is exonerated, go at attention** — but prefer *within-tree*
experiments, because the DP-vs-TP attention confound makes ours-vs-ref least
clean exactly there. Vary prompt length at fixed generation length and vice
versa. If degradation tracks generated-token count rather than total context
length, the decode-time KV write path is implicated, which fits the
repetition phenomenology.

Verifying the dumper fix isn't on the critical path — let it ride along on the
next dump run rather than burning a dedicated one.

Don't add `repetition_penalty`. It hides a real divergence.

### The other open ask

We still have no MMLU-Pro accuracy number. The serving path works at the
reference's shape so nothing blocks it mechanically, but running it before the
generation bug is understood would just measure the bug. Do it straight after
as the acceptance number.

## 14. Things worth knowing before you start

- Nine hypotheses died on 2026-08-11 alone, all of them to measurement rather
  than argument. Several were plausible enough that they'd have been committed
  on the strength of the story. Don't patch a location you can't name.
- Generated text is the one instrument here that nothing has managed to
  distort. When an activation result and a behavioural result disagree,
  believe the text.
- Older debug write-ups in this repo recorded claimed fixes that were never
  verified. Check any claim against the code and a fresh run before building
  on it.
