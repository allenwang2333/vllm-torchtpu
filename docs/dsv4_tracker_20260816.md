# DeepSeek-V4: everything in flight

> **Source of truth for DSv4, last updated 2026-08-17.** Supersedes the earlier
> handoff docs, which were deleted on 2026-08-16 because they contradicted each
> other: `dsv4_remaining_work.md`, `dsv4_next_session_prompt.md`,
> `handoff_refactor.md`, `possible_prompt.md`, `pr_review_tracking.md`,
> `docs/dsv4/`, and `../scratchpad/HANDOFF_20260814_evening.md`.
>
> `dsv4_handoff_20260806.md` survives only because `quant_weight_cache.py:28`
> cites its section 7c; it is historical and where it disagrees with this file,
> it is wrong.
>
> **Items are named, not lettered.** An earlier revision used codes (A1, G5,
> D1). They were opaque and are gone. Where an older doc or commit message cites
> one, the mapping is in "Retired item codes" at the end.
>
> **Companion:** `dsv4_design_parity_plan_20260816.md` for the remaining
> structural work; `dsv4_next_session_20260816.md` for the pickup prompt.

Nothing counts as finished until it is **written**, **verified**, **pushed** and
**merged**, in that order. Most of this branch is written and verified but not
pushed.

## Where things stand

The model is accurate, serves compiled at production context, and no longer
wastes KV cache memory. The largest piece of work — the native model refactor —
is written, validated, and has never been sent for review. That is the next
thing a human needs to do.

| Area | State |
|---|---|
| Accuracy | **Settled.** MMLU-Pro 0.7821 ± 0.0243, indistinguishable from the 0.7786 ± 0.0154 baseline |
| Compiled serving at long context | **Works.** 9216 tokens, request served |
| KV cache over-allocation | **Fixed** 2026-08-16 (`36abf661`), gated |
| Memory utilization | **0.75 verified** (was 0.55). 0.92 still fails, in compilation, cause unmeasured |
| Native model refactor | **Written, never submitted.** Nothing gates it |
| Model package structure | **Done** (`421698f0`). Split like `kimi_k3`, gate byte-identical |

## Picking this up cold

Everything is on host `alynie-v7x-8`, a single Google Cloud TPU v7x-8: **eight
chips, one tenant at a time**. There is no second machine.

| What | Where |
|---|---|
| Repo, working branch | `~/workspace/vllm-torchtpu`, branch `patemotter/dsv4-native-model` |
| Reference implementation | `~/workspace/tpu-inference` (pinned `f329f3480`) |
| Python env for everything | `~/workspace/uv_venv_dsv4/bin/python` |
| Model weights | `~/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1` |
| Quantized-weight cache | `~/workspace/dsv4_quant_cache` (132 GB, on disk — **never** `/dev/shm`) |
| Run logs, gate scripts | `~/workspace/vllm-torchtpu/scratchpad/` (untracked; not in git) |
| Historical eval artifacts | `~/workspace/eval_results/` (untracked) |
| `ruff` | not installed; use `uvx ruff@latest` |

Read `scratchpad/chain_progress.log` first — it is the append-only record of
every run, in order, with verdicts.

## File map

Model, in dependency order:

- `src/vllm_torchtpu/models/vllm/deepseek_v4/` — split like `kimi_k3` since
  `421698f0`: `layers.py` (164; `DeepseekV4MLP`, the `mhc_*` helpers),
  `moe.py` (142; `DeepseekV4MoE`), `attention.py` (628;
  `VllmDeepseekV4MLAAttention`), `model.py` (491; decoder, model, weights
  mapper, causal-LM). The package exports only `DeepseekV4ForCausalLM`.
- `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/` — the weightless
  half: `deepseek_v4_attention_op.py` (207; `_attention_jax`, the backend and
  SWA-cache subclasses), plus `deepseek_v4_compressor.py` and
  `deepseek_v4_indexer.py`.
- `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py` — the
  `deepseek_v4_fp8` quant config; dispatches linear vs MoE methods.
- `src/vllm_torchtpu/layers/vllm/quantization/fp8.py` — `VllmFp8LinearMethodTPU`
  and `VllmFp8MoEMethodTPU`. Shared with every other FP8 model, so DSv4-specific
  behavior here needs justifying.
- `src/vllm_torchtpu/kernels/deepseek_v4/` — Pallas kernels: `mla`, `mla_swa`,
  `compressor`, `compress_norm_rope`, `streamindex_topk`. **Trusted by default**
  per the working agreement; suspect our code first.
- `src/vllm_torchtpu/runner/tpu_runner.py` — KV cache allocation, including the
  DeepSeek-V4-specific overlay planning.

Tests: `tests/kernels/test_dsv4_*`, `tests/layers/vllm/test_dsv4_*`,
`tests/models/vllm/test_dsv4_model.py`.

Gate prompts: `tools/dsv4_debug/prompts/` — a 10-prompt `pool/`, plus
`needle_250.txt` (the in-window recall control, answer `7391-QUARTZ-4408`),
`short.txt`, `xla_llvm_prompt.txt`.

## Running things

```bash
# Compiled gate — the standard correctness check. ~30 min. 1024 context.
setsid nohup scratchpad/gate_dsv4_compile.sh . <tag> > scratchpad/<tag>_wrapper.log 2>&1 &
# Pass: tracebacks=0 sweeps=13 raises=0, needle exact, generations match baseline.

# Long-context check. Required for anything touching compilation or memory.
setsid nohup scratchpad/probe_compile_ceiling.sh 9216 > scratchpad/<tag>_wrapper.log 2>&1 &

# Startup probe at a chosen utilization; prints the KV allocation plan.
# Usage: probe_kv_budget.sh <util> [tag] [batched] [seqs] [ctx]
setsid nohup scratchpad/probe_kv_budget.sh 0.75 <tag> 1024 8 9216 > ... &

# XLA memory diagnostics (flag values + buffer assignment). Diagnostic only.
setsid nohup scratchpad/probe_xla_dump.sh 0.92 <tag> 1024 8 9216 > ... &

# MMLU-Pro accuracy, ~100 min at limit 20. Server-based.
setsid nohup scratchpad/run_dsv4_mmlupro.sh > scratchpad/mmlupro_wrapper.log 2>&1 &

# Scale-equivalence proof, CPU only, safe while a gate runs.
JAX_PLATFORMS=cpu ~/workspace/uv_venv_dsv4/bin/python scratchpad/compare_scale_equivalence.py
```

Baseline to diff generations against: `scratchpad/gate_09f6c694c.sub` — **not**
`gate_cache_compile.sub`, which predates the requantizer removal. A `.sub` file
is the extracted substantive lines from a `.log`; diff the `Generated text:` and
`Generated token IDs:` lines.

Name the tag after the commit being gated, and **commit before gating**. Every
2026-08-16 morning run shared commit `e8ecf947` with a working tree that changed
between runs, which made the SHA meaningless.

## Operating the TPU — hazards that cost time

**Is the TPU free?** The only check that works:

```bash
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q vfio && \
  echo "$p: $(tr '\0' ' ' < $p/cmdline | cut -c1-80)"; done
```

`lsof /dev/vfio/*` reports zero holders while a server is running — it is
useless here. `pgrep -f entrypoints.openai.api_server` also misses, because vLLM
renames its workers to `VLLM::Worker_TP0_EP0` and orphans keep no trace of the
parent command line. Both have misled repeatedly.

**Cleaning up after a kill.** `pkill -9 -f "VLLM::"`, then remove
`/dev/shm/psm_*` and `/tmp/libtpu_lockfile*`. `run_dsv4_mmlupro.sh` does this in
a trap and `gate_dsv4_compile.sh` does it at startup. Without it a hard-killed
run poisons the next gate with `PjRtClient is not initialized` / `Device or
resource busy; Couldn't open iommu group` — an error that looks like a code bug
and is not one. `scripts/vllm/benchmarking/cleanup_server.sh` handles all of it.

**Never run `pytest` while anything is serving** — it takes the chips.

**`LOAD_FORMAT=dummy`** is valid for shape, sizing and compile questions only.
Never for anything depending on weight values.

**Utilization.** Highest value verified to serve is **0.75** (`36abf661`, ctx
9216: KV 485,612 tokens, 52.69x concurrency). Before the KV fix it was 0.55.
0.92 fails in compilation, not in cache allocation.

**Disk, measured 2026-08-16.** `~/workspace` is a separate 2.0 TB NVMe with
681 GB free; `/` has 80 GB free; `/dev/shm` 281 GB free. `~/.cache/vllm` is a
symlink into `~/workspace`, so `torch_compile_cache` (312 GB), `hf_cache`
(149 GB) and `dsv4_quant_cache` (132 GB) all live on the NVMe, not on `/`.
`/dev/shm/torch_tpu_cache` holds 192 GB of RAM — the same category of problem as
the weight cache that was moved to disk.

## Settled — do not re-open without new evidence

### Accuracy

MMLU-Pro on this branch scores **0.7821 ± 0.0243** over 280 samples
(`09f6c694`), against 0.7893 ± 0.0241 measured the same morning and
0.7786 ± 0.0154 on 2026-08-12 over 700 samples. All within a fraction of a
standard error, with the same per-subject ordering (physics/biology ~0.95,
law/history/philosophy ~0.65). **The model has not regressed.**

The 2026-08-12 figure is the reference point:
`eval_results/phase3_limit50/.../results_2026-08-12T15-49-05.json`, `limit: 50`,
filter `custom-extract`.

**The `mmlu_llama` score of 0.238 was a harness artifact — solved 2026-08-17.**
The task ends its prompt with a partial *assistant* turn (`"The best answer
is"`) plus `continue_final_message: true`, expecting the model to continue it
with one letter. Routed through `/v1/chat/completions` that turn is closed and a
new one started, so the model answers a prompt that has already ended.

**Proven by a one-server A/B** (`scratchpad/run_dsv4_mmlu_endpoint_ab.sh`,
2026-08-17): same server, same weights, same three subjects, only the endpoint
differs.

| endpoint | A | B | C | D | junk |
|---|---|---|---|---|---|
| `/v1/chat/completions` | 0 | 0 | **50** | 0 | 10 |
| `/v1/completions` | 10 | 13 | 22 | 15 | 0 |

| subject | chat | completions |
|---|---|---|
| abstract_algebra | 0.40 | **0.80** |
| anatomy | **0.05** | **0.90** |
| astronomy | 0.35 | **0.95** |

The junk in the chat arm is the same signature seen before: `# Explanation`
blocks and stray Chinese text. This supersedes the earlier "inferred, unproven"
status of the `continue_final_message` explanation.

**Use `--model local-completions` for MMLU.** `scratchpad/run_dsv4_mmlu_full.sh`
does. tpu-inference reaches the same place from a different direction: its MMLU
job does not use `lm_eval` at all, but its own forked `benchmark_serving.py`
with a 1-shot HELM-format prompt and no chat template
(`tests/e2e/benchmarking/mmlu.sh:337-344`). A number from our harness is
therefore still not directly comparable to their CI figure.

**Do not switch MMLU-Pro to the completions endpoint.** It scores 0.7821 through
chat and is the accuracy baseline; changing its harness would break
comparability with the 2026-08-12 result.

**Concurrency scales.** MMLU-Pro at `--limit 20` (280 samples) takes ~104 min at
`num_concurrent=8` and **49 min at 32** — 10.7 s/sample, roughly 2x, with the
server confirmed running 32 concurrent requests. The KV headroom recovered by
the overlay fix is what makes that possible. Full MMLU-Pro (12,032 questions) is
therefore still a ~35 h job; size any run accordingly.

**Stragglers cost more than the tail they represent.** That same run reached
278/280 in 49 min, then spent ~50 min in `TimeoutError` retry loops on two
requests. Use `timeout=900,max_retries=2`, not `1800,5`.

Use `scratchpad/run_dsv4_mmlupro.sh` for future accuracy runs. The four
`scratch/run_mmlu*.sh` scripts export `LIBTPU_INIT_ARGS` and `MOE_REQUANTIZE_*`
— the environment behind the repeating-output harness bug — and are **not safe
to run verbatim**.

### Compiled serving at long context

**Compiled serving works at 9216 tokens.** Verified twice: on `d4f8c153` (KV
308,292 tokens, 33.45x concurrency, request served) and again after the KV fix
on `36abf661` (KV 309,223, 33.55x). There is no compile ceiling at production
context.

The earlier belief that it failed was self-inflicted: a session removed
`sys.setrecursionlimit(50000)`, and at 8192 tokens the compiled path then died
in Dynamo AOT capture — repeating `forward_model → aot_compile →
package.add_code → inspect.getsourcelines`, ending in `IndexError` / `OSError:
lineno is out of bounds` (`ceiling_8192_server.log` 15:25:40). The limit is
restored with a comment naming that failure. Its true required depth is still
unmeasured; 50000 is inherited, not derived.

The one-graph-backbone commit `de8a5a1c` was suspected of causing a compile OOM.
That is **unproven and weakened** — the reference also compiles its whole
backbone as one graph (`compilation_manager.py:_precompile_backbone_helper`), so
structure alone does not explain it.

### Eager mode costs about 3x

Median throughput from the servers' own `Avg ... throughput` lines:

| | prompt | generation | generation max |
|---|---|---|---|
| eager, 9216 ctx, 8 concurrent | 145.8 tok/s | **6.9 tok/s** | 9.6 |
| compiled, 08-12 baseline, 9216 ctx | 306.1 tok/s | **20.4 tok/s** | 53.5 |

~3x on generation, ~2x on prefill. The gap is larger on decode because per-layer
Python dispatch is paid 61 times per token with no large matmul to hide behind.
Matches the 3.6x end-to-end ratio (6.6 s/sample compiled vs 23.7 s/sample
eager). **Serve compiled.** Use eager only to isolate a compile-specific
suspicion.

### Weight requantization — removed, and it was safe

The custom requantizer was deleted in `83abd731`. It was **numerically
equivalent** to the reference, so this was a simplification rather than a fix:
routing through the standard block path gives scales **bit-identical** to
tpu-inference (0.000e+00 difference); the old requantizer differed only by
float32 rounding (6.9e-08). Our `axis=-1` on `[out,in]` is equivalent to the
reference's `axis=0` on `[in,out]`
(`scratchpad/compare_scale_equivalence.py`, CPU).

The 2.71% Frobenius error observed against a float64 block-dequantized ground
truth is **not a defect we introduced** — all three routes land on 2.4885%. It
is inherent to per-channel scales and the reference pays it too. Both codebases
end up per-channel at runtime (`layers/common/quantization/fp8.py:146-176`,
`requant_block_size=None`); the divergence was structural only.

Removing it did change generations, which had been predicted byte-identical. The
prediction was wrong for a reason the equivalence script could not see: the
**old path quantized twice** — the requantizer converted block scales to
per-channel over the weight iterator, then `_process_fp8_linear_weights` ran
again on that result, dequantizing to float32 and requantizing to fp8 a second
time. The new path does one conversion. Accuracy after the change is 0.7821, so
the drift cost nothing.

### Startup cost is not weight loading

Startup is 8.8 minutes and **diffuse** — the largest gap between logged
milestones is 30 s, spread across eight worker spawns, TPU init, weight read,
profiling and KV sizing. `--load-format dummy` would save little. With a warm
cache, requantization is not in the path at all.

## Solved 2026-08-16: KV cache over-allocation

**DeepSeek-V4 was allocating 1.3223x the KV cache memory vLLM had budgeted.**
That is why it could not exceed ~0.55 utilization while every other model on
this hardware runs 0.92–0.95. Fixed in `36abf661`.

### How the memory is supposed to be shared

The KV cache is divided into fixed-size pages; a per-request block table maps
positions to page indices. This model has two kinds of attention layer drawing
on that pool: 21 compressed-latent layers and 43 sliding-window layers. vLLM
places them in separate **cache groups**, and layers in different groups have
independent block-id namespaces, so the same physical page can back one layer
from each group without collision — vLLM says so itself in
`kv_cache_utils.py:_bucket_layers_by_page_size`.

TPU arrays cannot be byte views into a shared slab, so
`_initialize_ds_v4_kv_cache` rebuilds that sharing explicitly: each
sliding-window cache is pointed at a compressed-latent layer's array.

### The measurement

Run `kvprobe092_7744ac9b` logged the whole allocation plan before the first
`torch.zeros`. Per worker: `num_blocks` 14,324; 63 `KVCacheTensor`s, each
reporting `size = 72,810,840,064` — under vLLM's packed layout that is the whole
slab, not a per-tensor share (`vllm/v1/core/kv_cache_utils.py:1305-1341`,
`_get_kv_cache_config_packed`: `size = total_size`, `block_stride =
total_num_bytes_per_block`).

- vLLM's budget: 21x65,536 + 22x163,840 + 20x5,120 = **5,083,136 B/block**
- What we planned: 21x65,536 + 21x163,840 + 22x81,920 + 20x5,120 =
  **6,721,536 B/block**
- **1.3223x over, +22.4 GiB per worker**

The four distinct shapes, all `torch.uint8`:

| layer | block/storage | head | spec page | real page | array bytes |
|---|---|---|---|---|---|
| `*.attn.indexer.k_cache` (21) | 1024/256 | 256 | 65,536 | 65,536 | 938,737,664 |
| `*.attn` compressed-latent (21) | 1024/256 | 640 | 163,840 | 163,840 | 2,346,844,160 |
| `*.attn` dense (20) | 1024/8 | 640 | 5,120 | 5,120 | 73,338,880 |
| `*.attn.swa_cache` (43) | 128/128 | 640 | 163,840 | **81,920** | 1,173,422,080 |

### The cause

`_plan_ds_v4_swa_overlay` kept **one global set of already-used host layers**.
With 21 compatible compressed-latent layers and 43 sliding-window caches, only
21 could overlay; the other 22 each took a full `num_blocks` array that vLLM had
never budgeted. The log line said it plainly all along: `DeepSeek-V4 SWA cache
overlay: 21 of 43`.

Distinctness is required *within* a cache group, not across groups. The
reference rebuilds its host list inside the per-group loop and indexes it by
position (`tpu-inference/runner/kv_cache_manager.py:1159-1178`), so one array
hosts one layer from each group. The comment defending the global set had the
invariant backwards.

### The result

Every prediction was exact: overlay **42 of 43**, planned **5,001,216 B/block**,
ratio **0.9839** against vLLM's own `block_stride` of 5,083,136.

| | before | after |
|---|---|---|
| Sliding-window caches overlaid | 21 of 43 | **42 of 43** |
| Allocation vs budget | 1.3223x | **0.9839x** |
| Utilization verified to serve | 0.55 | **0.75** |
| KV cache at ctx 9216 | 309,223 tokens | **485,612 tokens** |
| Concurrency at ctx 9216 | 33.55x | **52.69x** |

Correctness: compiled gate passed with generations and token IDs
**byte-identical** to `gate_09f6c694c.sub`, needle exact. The 9216 long-context
probe passed. The change moves no output — it only changes which physical array
a cache lives in.

The instrumentation that produced this (`_probe_ds_v4_kv_budget`, printing
`[KV_PROBE]` lines) is still in `tpu_runner.py`. **Strip it before submitting
the refactor.**

## Serving-mode throughput, 2026-08-17 — KV-capped

**Not the first DSv4 throughput measurement.** Offline batch throughput was
measured 2026-08-09 (`eval_results/perf_offline.log`,
`perf_offline_nosync.log`, and `perf_ref.log` for tpu-inference), and serving
medians exist from every gate. What is new here is the first *completed run of
the benchmark harness*, i.e. serving-mode numbers with TTFT/TPOT/ITL/E2EL
percentiles at controlled concurrency. Two combos before the 2.5 h timeout
(`scratchpad/bench_20260817/`), 1024:1024, 128 prompts, util 0.75, ctx 9216,
`max-num-batched-tokens` 8192, `max-num-seqs` 512:

| metric | concurrency 64 | concurrency 256 |
|---|---|---|
| output throughput | **53.8 tok/s** | **54.7 tok/s** |
| median TPOT | 1079 ms | 1086 ms |
| median ITL | 999 ms | 1015 ms |
| median TTFT | 11.3 s | 131.2 s |
| median end-to-end | 1011 s | 1231 s |

**One token per second per sequence**, corroborated three ways: the server's own
generation throughput (median 64 tok/s, max 73, over 644 intervals), the batch
actually running (356 intervals at exactly 64 concurrent), and 64 tok/s / 64
sequences. Zero preemptions.

**Concurrency past 64 buys nothing** — 53.8 to 54.7 tok/s (+1.7%) while TTFT
goes 11 s to 131 s. Everything above 64 queues.

**The run was KV-capped, not compute-capped.** GPU KV cache usage peaked at
**99.9%**. 115,565 tokens divided by 2048 per request caps the batch at ~56,
which is exactly what was observed. The cache is small *because of the
benchmark's own config*: at `max-num-batched-tokens=1024` the same utilization
yields 485,612 tokens, 4.2x more.

**Context before calling this a defect.** tpu-inference measured 92.20 tok/s at
batch 128 (`eval_results/perf_ref.log`, 4096 ctx, util 0.67, offline), which is
0.72 tok/s per sequence — **1.39 s per token**. Per sequence we are slightly
faster; they get more aggregate by running twice the batch. Different configs,
so not a clean comparison, but ~1 s per token appears to be what this model does
on this hardware in both implementations rather than something this port broke.

Under test: `deepseek-v4-flash-tp8-ep-b1024`, identical except
`max-num-batched-tokens=1024`. If decode is weight-bound, throughput should
scale with the extra batch room.

### Full MMLU is infeasible at this decode rate

`mmlu_llama` through `/v1/completions` reached 879 of 14,042 in 1 h 52 m —
**7.32 s per request, projecting 26.8 hours**. Killed. The endpoint fix makes
MMLU *correct*; it does not make it fast. Size any MMLU run as a subset, and
expect MMLU-Pro (long outputs) to be worse per sample.

## Solved 2026-08-17: what caps utilization at 0.75

**The compile OOM is the KV cache itself, not recomputable activations.** The
buffer-assignment dump (`scratchpad/xladump_full_xla`, 5,827 files, 2.6 GB,
captured with no `--xla_dump_max_hlo_modules` cap) names the failing module:

```
Total bytes: 114309400896 (106.46GiB)
```

which matches the error's *HLO temporaries (106.45G)* exactly. Its largest
allocations are KV cache arrays -- `num_blocks` is 14,362 and appears in every
shape:

| share | size | contents |
|---|---|---|
| 20% | 7.06 GiB | `42xu8[14362,256,512]`, `21xu32[14362,256,128]`, `21xu8[14362,64,4,256]` |
| +7% | 2.19 GiB | `21xu8[3676672,640]`, `42xu8[14362,256,640]` |

`u8[14362,256,640]` is 2.19 GiB each; `u8[14362,256,512]` is 1.75 GiB. At the
measured 5,001,216 B/block the KV cache is **66.9 GiB, 63% of the module**.

**Consequences:**

- The compiled graph holds the whole KV cache live, so compile memory scales
  directly with `gpu-memory-utilization`. That is why the same configuration
  passes at 0.75 and fails at 0.92 with context, seqs and batched tokens all
  held fixed.
- **Rematerialization would not help**, which retires the lever this file
  previously called the untried option. Remat trades compute for memory by
  recomputing values; KV cache is state read back across steps, not a
  recomputable intermediate. `xla_tpu_rematerialization_min_size_in_bytes` and
  `_use_host_memory_offload` are irrelevant to a 67 GiB block of cache.
- The only levers that move this number are the ones that make the KV cache
  smaller: utilization, `max-num-batched-tokens`, or context.

This supersedes the earlier inferred explanation and closes the question. 0.75
is not a workaround for an unknown; it is the utilization at which the KV cache
plus activations fit the compiler's 94.74 GiB.

## Open: work written but never submitted

**The native model refactor.** Nine commits, `ff1033a2..HEAD`, absent from
`origin/main`. This is what gxd3 asked for on PR #462 — "define the our TPU DSv4
model.py ... and update the registry to point to it", citing `tpu-inference`
PR #3065 — and it has never been sent. **Nothing gates it:** accuracy is
settled, compiled serving works at 9216, and the belief that it should wait on
an MHC cleanup rested on a misattribution (below). Splitting it into reviewable
PRs and sending it is the highest-value remaining action, and it needs Pate —
the agent never opens PRs.

**The generic weight cache — not an open decision.** PR #468
(`feat/generic-weight-cache`) was opened by an agent **without Pate's consent**
and he closed it immediately, unmerged, with zero comments. It is not a pending
"resubmit vs drop" question and must not be recorded as one; whether it is ever
sent again is entirely his call and should not be raised unprompted.

The code itself is live on this branch and is a real dependency, not dormant:
`layers/common/quant_weight_cache.py` is imported by `mxfp4.py`, covered by
`tests/layers/test_quant_weight_cache.py`, and is what every DSv4 run uses via
`VLLM_QUANT_WEIGHT_CACHE_DIR` (the 132 GB `dsv4_quant_cache`). It is absent from
`origin/main`, so the refactor stack carries it either way. Its claimed ~9
minutes of startup saving is unverifiable today because the cache logs neither
hits nor misses.

**Nothing is owed on `tpu_compiler.py`.** `git diff origin/main HEAD` on that
file is empty. The earlier "54+/26− still on the branch" came from diffing
against a stale local `main` (`4b87cff5`) instead of `origin/main`
(`46a6a94b`). Whether the hardening landed elsewhere or was dropped is unknown.

## Done 2026-08-17: the model package is split like `kimi_k3`

`model.py` was one 795-line file with six classes. Split in `421698f0`:

| file | lines | contents |
|---|---|---|
| `layers.py` | 164 | `DeepseekV4MLP`, `mhc_pre_torch`, `mhc_post_torch`, `mhc_hc_head_native` |
| `moe.py` | 142 | `DeepseekV4MoE` |
| `attention.py` | 628 | `VllmDeepseekV4MLAAttention` |
| `model.py` | 491 | `DecoderLayer`, `Model`, weights mapper, `ForCausalLM` |
| `custom_ops/.../deepseek_v4_attention_op.py` | 207 | `_attention_jax`, `VllmDeepseekSparseSWABackend`, `VllmDeepseekV4SWACache` |

Against `kimi_k3`'s 112 / 161 / 667 / 637 the proportions line up.

**The dividing line for attention is whether the code holds weights**, which is
what `kimi_k3` does: its `attention.py` holds `MultiHeadLatentAttention` and
imports kernel dispatch from `custom_ops/kda_attention_op.py`. So
`VllmDeepseekV4MLAAttention` moved into the package; `_attention_jax` and the
backend/cache subclasses stayed behind. Compressor and indexer stayed put
untouched — attention pulls them in by rebinding upstream's module globals
during `__init__`, which is by name at call time and so is location-independent.

`DeepseekV4Head` is gone, inlined into `Model.forward` as the reference does
(`deepseek_v4.py:656-664`). It held no parameters of its own — it aliased the
model's — so `named_parameters` is unchanged.

**Verified:** compiled gate at `421698f0` passed `tracebacks=0 sweeps=13
raises=0`, needle exact, generations and token IDs **byte-identical** to
`gate_09f6c694c.sub`. Imports re-checked on the real TPU backend (8 devices),
including that `_attn_op_cache` remains one shared object across the new module
boundary.

**One wart left deliberately.** `_attn_op_cache` is a module-level mutable dict
defined in the op module and read/written by the attention class in the package.
It works because the dict is only mutated by item assignment, never rebound. The
`global _attn_op_cache` statement inside the method is now misleading — it names
the imported binding, not the defining module's. Left alone to keep the commit a
pure move; worth cleaning up separately.

## The MHC cleanup was dropped — what the reviewer actually asked

The MHC registration item was **not** a reviewer request. It was produced by
this project's own reference-vs-ours gap analysis on 2026-08-16 (`a59f6c0b`) and
then written up as though gxd3 had asked for it, and that misattribution was
used to argue the native-model refactor should wait for it.

What gxd3 actually asked, verified against the threads on 2026-08-16:

- **PR #462**, on `deepseek_v4_patch.py:19`: "Instead of patching, how about
  just define the our TPU DSv4 model.py ... and update the registry to point to
  it", citing `tpu-inference` PR #3065. That is the native model refactor, and
  it is **already implemented** on this branch — `deepseek_v4_patch.py` is gone.
  He accepted a TODO and deferring it.
- **PR #459**, on `deepseek_v4_attention.py`: drop the
  `DeepseekV4ROCMAiterMLAAttention` dependency. A TODO was accepted.
- **PR #464**: three asks about generic files carrying DSv4-specific changes —
  `tpu_compiler.py` (split out), the recursion-limit bump, and `__init__.py`.

None mentions MHC, `HCHeadOp`, `forward_native` or `register_oot`.

One measurement worth keeping from the attempt: our `mhc_pre_torch` and
`mhc_post_torch` (`layers.py:26-95`) are line-for-line identical to vLLM's
`model_executor/kernels/mhc/torch.py`, so they could delegate. Our
`mhc_hc_head_native` (`layers.py:97-113`) is **not** identical to the
reference's `forward_tpu`: ours keeps float32 through the linear and the
weighted sum, the reference downcasts to bfloat16 first. Ours is the
higher-precision path and is what the 0.78 MMLU-Pro results were measured with,
so any future port must move our body, not copy the reference's.

## Smaller open items

**Is the weight cache being read or rewritten?** 132 GB / 344 files in
`dsv4_quant_cache`, nothing in `/dev/shm`. Startup 32 min vs the baseline's 33 —
consistent with a warm read, not proof. `quant_weight_cache.py` logs no
hit/miss. Add one before anyone quotes a speedup.

**The ROCm class-name string — removed 2026-08-17.** The spec-override loop
tested four class names; searching the installed vLLM showed
`DeepseekV4MultiHeadLatentAttentionWrapper` and `DeepseekV4MLAAttention` do not
exist at all, and `DeepseekV4ROCMAiterMLAAttention` exists only in
`vllm/models/deepseek_v4/amd/rocm.py`, never imported on TPU. Only
`DeepseekV4Attention` can match, so the tuple became a single comparison. This
discharges gxd3's PR #459 ask.

**Performance against tpu-inference.** Recorded as 1.27x slower at batch 1 and
3.34x at batch 128, but that is **from memory of a 2026-08-09 run and
unverified**. `eval_results/perf_ref.log`, `perf_offline.log` and
`perf_offline_nosync.log` exist and would settle it. Re-derive before quoting.

**Data-parallel attention.** Deferred; needs its own planning pass. It requires
a sharding change through the runner and KV cache manager. Consequence:
tpu-inference hard-validates MLA together with DP attention, so **no
reference-side control with DP off can exist**, and therefore **no genuine
ours-vs-reference numerical comparison has ever run** — every "identical" result
so far compares us to ourselves.

## Flag audit, 2026-08-16

Prompted by "are we missing a flag that fixes the compile OOM". Answer: no flag
we were missing explains it, but the audit corrected several beliefs.

**Reading real flag values.** `XLA_FLAGS=--xla_dump_to=<dir>` emits a
`*.tpu_comp_env.txt` per module listing **all 1,530 TPU flags with their live
values**. That is how to read defaults; no help text is compiled into
`libtpu.so`, and `--xla_tpu_log_flags` is unnecessary. `libtpu.so` exposes 1,708
`xla_*` names in total.

Values that matter for the compile OOM:

| flag | value | consequence |
|---|---|---|
| `xla_tpu_user_reserved_hbm_bytes` | -1 | nothing reserved — 94.74 GB is real HBM |
| `xla_tpu_max_hbm_size_mib` | 0 | not capped; no knob raises the budget |
| `xla_tpu_rematerialization_use_host_memory_offload` | false | remat cannot spill to host |
| `xla_tpu_rematerialization_min_size_in_bytes` | 10485760 | buffers under 10 MB never remat |
| `xla_jf_rematerialization_percent_shared_memory_limit` | 95 | remat engages only at 95% |
| `xla_tpu_always_spill_to_default_memory` | false | |
| `xla_enable_megacore_hbm_spill` | true | already on |

**`XLA_FLAGS` and `LIBTPU_INIT_ARGS` are different namespaces.** `XLA_FLAGS`
accepts only `DebugOptions` proto fields; TPU-side flags go through
`LIBTPU_INIT_ARGS`. An unknown name in either is **fatal within ~36 seconds** of
startup, well before weight loading, so validating a flag name is cheap.

**Two settings in our own config did nothing; both removed 2026-08-17.**

- `NEW_MODEL_DESIGN=1` and `MODEL_IMPL_TYPE=vllm` were **tpu-inference**
  environment variables (`tpu_inference/envs.py:20-22`). Nothing under `src/`
  reads them. They read as a model-implementation switch and were not one.
- `ATTENTION_BACKEND="CUSTOM"` is **inert for any MLA model**:
  `tpu_platform.py:711-712` overrides the selected backend to
  `FLASH_ATTN_MLA` whenever MLA is in use, before the supported-backend check.
  Removed from the DSv4 config; still present in other models' configs, where
  it is only inert for the MLA ones (`deepseek-v2-tp8-ep.sh` looks like the
  same case, unverified).

**Applied already, no action needed:** `--async-scheduling` (auto-enabled when
unset, `config/vllm.py:1107`; every run logs "Asynchronous scheduling is
enabled"), `--quantization` (auto-detected to `deepseek_v4_fp8`),
`USE_MOE_SPARSE_CORE` (defaults True, `envs.py:38`), and the root README's
`TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false` /
`TORCHINDUCTOR_AUTOGRAD_CACHE=0`, both set by `env_override.py:30` and
`src/sitecustomize.py:8-10` — **the README's "set these workarounds" step is
stale**. Despite its name that first flag is not a memory knob: it disables a
torch_tpu graph-break guardrail so collectives compile under SPMD
(`env_override.py:25-29`).

**Do not adopt `ONEHOT_MOE_PERMUTE_THRESHOLD=32768`** (used by qwen3.5-397b): it
forces MoE permute onto the one-hot TensorCore path, materializing a
`[rows, num_experts]` matrix (`fused_moe_gmm.py:50-77`), which grows
temporaries.

**Noise, not ours:** JAX's `cloud_tpu_init.py:104-109` injects
`--xla_tpu_use_enhanced_launch_barrier=true`; libtpu warns it "can cause
distributed hangs" and rewrites it to false, eight times per run, in every log.

## Measuring memory properly

Running whole servers for a pass/fail bit is the wrong instrument.

- **`XLA_FLAGS=--xla_dump_to=<dir>`** emits buffer-assignment output on
  *successful* compiles too, not only failures, turning each probe into a
  measurement instead of one bit. Do not pair it with
  `--xla_dump_max_hlo_modules`.
- **Fit, don't bisect.** With peak memory at three context sizes, fit
  `peak(ctx) = c + m·ctx` and solve for the crossing. The slope-vs-intercept
  split is itself diagnostic: large `m` means activations scaling with context,
  large `c` means something fixed grew.
- **Harvest what is already printed.** Every startup logs `GPU KV cache size`
  and `Maximum concurrency` — a free second measurement of the same budget.

## Refuted — do not re-raise these

| Claim | Verdict | What disproved it |
|---|---|---|
| "The model has regressed / may be producing noise" | **false** | MMLU-Pro 0.7893 ± 0.0241 on this branch vs 0.7786 ± 0.0154 on 08-12 |
| "tpu-inference keeps DSv4 linear layers block-quantized; we deviate by going per-channel" | **false** | the reference dequantizes the 128x128 scales and requantizes per-channel by default (`layers/common/quantization/fp8.py:146-176`, `requant_block_size=None`). Both end per-channel |
| "The 2.71% Frobenius error is a defect our requantizer introduced" | **false** | all three routes measure 2.4885% on the same input; it is inherent to per-channel |
| "The recursion limit is not load-bearing" | **false** | the run that "verified" its removal only exercised 1024 tokens. At 8192 the compiled path dies in Dynamo AOT capture (`ceiling_8192_server.log` 15:25:40). Restored |
| "`de8a5a1c` (one graph for the backbone) causes the 9216 compile OOM" | **unproven, weakened** | the reference also compiles the whole backbone as one graph. Structure alone does not explain it |
| "The 9216 run that worked was tpu-inference" | **false** | `eval_results/phase3_limit50/vllm_server.log` shows `Loading plugin torchtpu` and `vllm_torchtpu`, zero `tpu_inference`. It was our own 08-12 branch |
| "tpu-inference is not comparable because it doesn't use torch.compile" | **wrong framing** | it is the design being emulated and serves this model on this hardware. Its graph boundaries are legitimate to copy, even though it is JAX-jitted |
| "Compiled serving fails at 9216" | **false** | passed on `d4f8c153` and again on `36abf661` |
| "The KV fix is to fold the DSv4 branch into the shared allocation tail" | **inapplicable as written** | the shared tail computes `num_blocks = tensor_size // total_group_page_size` per tensor, but under the packed layout `tensor_size` is the whole slab, so that formula yields nonsense per-tensor block counts — and DSv4's layers must share one block-id space. The allocation was never the problem: the overlay plan was. The TODO at `tpu_runner.py:5364` stands as tidiness only |
| "DSv4's bespoke KV allocation is a deviation from tpu-inference" | **false** | the reference has the same bespoke overlay and the same 43-vs-21 shortfall (`kv_cache_manager.py:1015-1053`). It is a deviation from the *shared vLLM path*, not from the reference. Fixing host selection makes our memory behavior better than the reference's, which is fine — memory parity with it is out of scope |
| "`max-num-batched-tokens` drives the compile-temporaries OOM" | **false** | 8192 → 1024 moved temporaries 106.41G → 106.38G. The batched-tokens finding is about KV cache size, a different budget; do not carry it over to compile scratch |
| "`max-num-seqs` drives the compile-temporaries OOM" | **false** | 512 → 32 moved temporaries 106.38G → 106.43G |
| "Context length drives the compile-temporaries OOM" | **false** | ctx 9216 passes at 0.75 and fails at 0.92, all else identical |
| "gxd3 asked us to drop the MHC monkey-patch" | **false** | no comment on PR #459, #462 or #464 mentions `HCHeadOp`, `forward_native`, `register_oot` or MHC. The item originated in this project's own gap analysis in `a59f6c0b` |
| "The native model refactor should wait for the MHC cleanup" | **false** | consequence of the above. The refactor *is* what gxd3 asked for, it has been unsubmitted since 2026-08-14, and nothing gates it |
| "No DSv4 throughput number has ever been obtained" | **false** | offline batch throughput was measured 2026-08-09 (`perf_offline.log`, `perf_offline_nosync.log`, `perf_ref.log`) and serving medians exist from every gate. Only the *benchmark harness* had never completed a run. This phrase was in the tracker and was repeated on 2026-08-17 even after those same logs had been re-derived that morning |
| "`lsof /dev/vfio/*` reliably shows whether the TPU is held" | **false** | it reported zero holders while a server was demonstrably running |
| "The `tpu_compiler.py` hardening is still owed as a standalone PR" | **false** | `git diff origin/main HEAD` on that file is empty; the earlier delta came from diffing against a stale local `main` |
| "PR11 gates the refactor" | **dropped** | no kv-admission PR exists in `gh pr list --author @me --state all`. Pate's call 2026-08-16: not relevant and not a gate |

## Run log

**Provenance warning.** Every 2026-08-16 morning run was branch
`patemotter/dsv4-native-model` at commit `e8ecf947` with an *uncommitted working
tree that changed between runs*, so the SHA does not identify what ran. "RL" is
whether `sys.setrecursionlimit(50000)` was present. Runs from `7744ac9b` onward
were committed before gating. Reference tree: `tpu-inference` at `f329f3480`.

| # | ctx | util | seqs | batched | mode | RL | outcome |
|---|---|---|---|---|---|---|---|
| 08-12 baseline | 9216 | ? | ? | 1024 | compiled | yes | MMLU-Pro 0.7786; KV 412,832 |
| mmlu_llama samples | 4096 | 0.48 | 16 | 4096 | eager | no | completed; exposed the constant-`C` artifact |
| mmlupro A | 9216 | 0.48 | 16 | 9216 | eager | no | startup `ValueError`: KV needs 47.8 GB, 26.29 available |
| mmlupro B | 9216 | 0.80 | 16 | 9216 | eager | no | KV 126,516; **execution OOM** |
| mmlupro C | 9216 | 0.75 | 8 | 1024 | compiled | **no** | **compile OOM, 103.09 GB temporaries vs 94.74 GB** |
| mmlupro D | 9216 | 0.75 | 8 | 1024 | eager | no | KV 486,321; execution OOM |
| mmlupro E | 9216 | 0.55 | 8 | 1024 | eager | no | reached 265/280, died on 300 s client timeouts |
| mmlupro F | 9216 | 0.55 | 8 | 1024 | eager | no | **PASS** — 0.7893 ± 0.0241, `timeout=1800` |
| cleanup-batch gate | 1024 | 0.48 | 16 | 1024 | compiled | no | **PASS**, generations byte-identical to baseline |
| ceiling 8192 | 8192 | 0.55 | 8 | 1024 | compiled | **no** | START_FAIL — Dynamo recursion, not memory |
| ceiling 8192b | 8192 | 0.55 | 8 | 1024 | compiled | yes | stopped at 15:37 before a verdict. No data |
| gate_09f6c694 / b | 1024 | 0.48 | 16 | 1024 | compiled | yes | **environmental, not code**: orphaned `VLLM::Worker_*` held the chips |
| gate_09f6c694c | 1024 | 0.48 | 16 | 1024 | compiled | yes | **PASS**. Generations differ from `gate_cache_compile.sub` — the requantizer round-trip removal. This `.sub` is the new baseline |
| ceiling 9216 | 9216 | 0.55 | 8 | 1024 | compiled | yes | **PASS** — KV 308,292, 33.45x, request served |
| mmlupro @ 09f6c694 | 9216 | 0.55 | 8 | 1024 | eager | yes | **0.7821 ± 0.0243** over 280. 114 min |
| bench 0.92 | 10240 | 0.92 | 512 | 8192 | compiled | yes | START_FAIL — KV init wanted 2.19 G, 1.60 G free |
| bench 0.85 | 10240 | 0.85 | 512 | 8192 | compiled | yes | START_FAIL — wanted 1009.6 M, 986.5 M free. KV 136,847 |
| bench 0.83 | 10240 | 0.83 | 512 | 8192 | compiled | yes | START_FAIL — wanted 978.4 M, **420.1 M** free. KV 132,611 |
| kvprobe @ `7744ac9b` | 10240 | 0.92 | 512 | 8192 | compiled | yes | **START_FAIL in KV init**, and the measurement: 6,721,536 vs 5,083,136 B/block, **1.3223x** |
| kvprobe @ `36abf661` | 10240 | 0.92 | 512 | 8192 | compiled | yes | **KV init passes** — overlay 42/43, ratio **0.9839**, KV 151,990. Then START_FAIL in compile, 106.41G |
| gate @ `36abf661` | 1024 | 0.48 | 16 | 1024 | compiled | yes | **PASS** — `tracebacks=0 sweeps=13 raises=0`, needle exact, **byte-identical** to `gate_09f6c694c.sub` |
| ceiling 9216 @ `36abf661` | 9216 | 0.55 | 8 | 1024 | compiled | yes | **PASS** — KV 309,223 (was 308,292), 33.55x |
| util 0.92, 1024 batched | 10240 | 0.92 | 512 | **1024** | compiled | yes | START_FAIL in compile, **106.38G**. KV fine: 702,983, 68.65x. Refutes batched tokens |
| util 0.92, 32 seqs | 10240 | 0.92 | **32** | 1024 | compiled | yes | START_FAIL in compile, **106.43G**. KV 703,473. Refutes seqs |
| util 0.92, ctx 9216 | 9216 | 0.92 | 8 | 1024 | compiled | yes | START_FAIL in compile, **106.45G**. The passing config with only util changed — util is the driver |
| **util 0.75, ctx 9216** | 9216 | **0.75** | 8 | 1024 | compiled | yes | **READY** — KV **485,612**, **52.69x**. Same config failed pre-fix |
| xla dump attempt 1 | 9216 | 0.92 | 8 | 1024 | compiled | yes | died in 36 s: `--xla_dump_hlo_memory_schedule_info` is not an `XLA_FLAGS` name |
| xla dump attempt 2 | 9216 | 0.92 | 8 | 1024 | compiled | yes | reached the failure, 106.45G. Flag table captured; **buffer breakdown lost to `--xla_dump_max_hlo_modules=20`** |

**The benchmark config had never completed a run** — though throughput itself
was measured offline on 2026-08-09 and from serving-log medians. The config
turned out to carry three separate defects, each of which alone prevented it
running. All fixed 2026-08-17:

1. `GPU_MEMORY_UTILIZATION=0.55` — a workaround for the KV over-allocation,
   obsolete once that was fixed. Now 0.75, the highest verified to serve.
2. `MAX_MODEL_LEN=10240` — at util 0.75 the profiler leaves 51.71 GiB for KV
   and 10240 needs 53.11, so startup failed outright with a `ValueError`
   naming 9968 as the achievable maximum. Now 9216, which is exactly
   8192+1024 and 1024+8192 so every combo fits.
3. `RANGE_RATIO_STYLE` unset — the default *symmetric* reading of
   `RANDOM_RANGE_RATIO=0.8` samples input lengths in `[(1-r)L, (1+r)L]`, i.e.
   up to **14,746 tokens** at ISL 8192, past any context this model serves.
   Now `"min"` (`[rL, L]`), which is what the other large configs use.

The sweep also lacked `1024:8192`, the decode-heavy case every other large
config runs — and the case that matters most here, since the gap against
tpu-inference widens on decode (1.45x at batch 8 to 3.34x at batch 128) and
decode is what the recovered KV capacity buys. Added, and listed second so a
timeout truncates the prefill-heavy case instead.

**Reading the results.** At the benchmark's dimensions the probe measured KV
115,565 tokens, 12.54x concurrency at 9216 — far below the 485,612 seen at
seqs 8 / batched 1024, confirming `max-num-batched-tokens` as the dominant
consumer. So on the `1024:8192` combo only ~12 requests are resident at once:
at `CONCURRENCY_OPTIONS=256` the rest queue, and median TTFT becomes
queue-position noise rather than latency. Throughput stays valid; TTFT at 256
does not.

## Standing caveats

Merged PRs were reviewed on code structure. No review thread mentions accuracy,
and a 26/26 PASS measures tracebacks, NaNs and decode steps completed — not
whether answers are correct. This model shipped all-zero MoE experts that
produced fluent output for weeks, so fluent output is not evidence.

Figures marked unverified come from memory or prior-session docs and were not
re-derived. Per the working agreement they are claims, not evidence — including
ones written by an agent.

## Retired item codes

Older commits and docs cite these. They are not used anymore.

| Code | Was |
|---|---|
| A1–A7, A15 | Accuracy and the `mmlu_llama` artifact → "Accuracy" |
| A8, A11, A14, A9 | Compiled serving at 9216 → "Compiled serving at long context" |
| A10 | Batched tokens as the KV-size driver → "Open: memory utilization" |
| A12 | PR11 as a gate → Refuted table |
| A13 | Eager costs 3x → "Eager mode costs about 3x" |
| B1–B7 | The 2026-08-16 cleanup batch → folded into the run log |
| C1–C3 | Requantizer equivalence → "Weight requantization" |
| D1–D3 | Unsubmitted work → "Open: work written but never submitted" |
| E1–E3 | Perf gap, DP attention, no reference comparison → "Smaller open items" |
| F1 | ROCm class-name string → "Smaller open items" |
| G1 | MHC registration → dropped, see its own section |
| G2, G3, G4 | Head class, file split, attention location → "The model package is one file" |
| G5 | KV over-allocation → "Solved: KV cache over-allocation" |
