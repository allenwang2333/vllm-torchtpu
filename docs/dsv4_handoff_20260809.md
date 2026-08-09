# DSv4 Handoff — 2026-08-09

Branch: `patemotter/dsv4-lean-main-e2e`. Node: this host *is* the TPU node.
Venv: `/home/patemotter_google_com/workspace/uv_venv_dsv4` (vllm 0.26.0,
libtpu 0.0.44.1, torch_tpu 0.1.1.dev20260728, lm_eval + `[api]` extra).

Supersedes `dsv4_handoff_20260806.md` for status. Detail on this session's
investigation: `dsv4_mmlu_pro_bringup.md`.

## TL;DR

DeepSeek-V4-Flash now **serves at the reference's shape** — 9216 context,
`max_num_batched_tokens=1024`, `max_num_seqs=128`, TP=8, EP=8 (32/256 experts
per rank), fp8 KV, prefix caching off, non-eager — with real weights. Before
this session it died in `initialize_kv_cache` at any context past ~1024.

**No MMLU-Pro accuracy number was produced.** That remains the open ask.

## 1. What was broken and what fixed it

**Root cause: KV caches that overlay another array were allocated first and
aliased afterwards.** `_alias_ds_v4_auxiliary_caches` ran *after* the
allocation loop, so every compressor/indexer state cache got a full array; all
169 arrays were live simultaneously at peak. The aliasing freed them a moment
later, which is why no post-allocation accounting ever showed the problem —
it is a peak-usage bug.

Measured: ~9.5 MiB/block at peak against the 6.09 MiB/block vLLM used to size
`num_blocks`. Planned KV dropped **80.65 GiB → 54.48 GiB** once the overlay was
planned before allocation, and startup went from failing every attempt to
serving on the first try.

Fixes, both in `runner/tpu_runner.py`:
- `_plan_ds_v4_auxiliary_overlay()` computes the state-cache → host mapping from
  layer names before allocation; the loop skips those layers (the SWA overlay
  already worked this way).
- One array per `KVCacheTensor` instead of one per layer (`_alloc_or_share`),
  matching `tpu_inference/runner/kv_cache_manager.py:914`. Sharing is
  pervasive here: of 65 tensors, 41 are shared by 2 layers and 21 by 4.

## 2. Numbers

Offline throughput, `scripts/dsv4/perf_offline.py`, 512-token prompts,
64 generated tokens, `ignore_eos`, TP=8/EP=8, `max_model_len=4096`:

| batch | wall | decode tok/s | total tok/s | vs b1 |
|---|---|---|---|---|
| 1 | 90.2 s | 0.71 | 6.4 | 1.00x |
| 8 | 97.5 s | 5.25 | 47.3 | 7.40x |
| 32 | 125.7 s | 16.29 | 146.6 | 22.96x |
| 128 | 369.3 s | 22.18 | 199.6 | 31.28x |

Reading: batch 1→32 is nearly free (23x throughput for 39% more wall), so the
"~1 tok/s" seen while serving was an **empty batch against a graph compiled for
`max_num_seqs`**, not a per-step kernel cost. It saturates after ~32 (32→128
buys 1.36x for 3x the wall), so past that the model is genuinely compute-bound.

**Whether 22 tok/s is good is unknown** — the reference comparison had not
finished at handoff time. See §4.

## 3. Verified vs not

Verified: serves at 9216/128-seq/TP8/EP8 on real weights; DSv4 runner unit
tests pass; the generic weight-cache tests pass (11).

**Not verified: that output is correct at this configuration.** The needle test
that validated the MoE fix ran at `max_model_len=1024`, eager, offline,
single-sequence. This config changes all four. Any accuracy claim needs the
needle control passing at *this* config first — see
[[project_dsv4_runA_invalid_baseline]] for why a claim without its control is
worthless here.

## 4. Open items, in priority order

1. **MMLU-Pro accuracy.** `scripts/dsv4/run_mmlu_pro.sh` works end-to-end
   (requests answered, harness preflighted). The reference ran a **10% subset**
   (`ds_v4_flash_mmlu_pro_10pct_fixed`), so use `LIMIT=0.1` for comparability:
   ~1,200 questions, ~3 h at current throughput. The full 12,032 would be ~30 h.
2. **Reference perf comparison.** `tpu-inference/perf_offline_ref.py` mirrors
   ours exactly. NOTE: the reference **refuses to run MLA without DP
   attention** (hard config validation), so there is no TP-only reference row —
   every ours-vs-theirs comparison is confounded by DP attention. Say so
   whenever quoting one.
3. **Residual over-allocation.** We still plan 54.48 GiB against vLLM's
   44.28 GiB budget (1.23x). It fits in physical HBM so it serves, but the
   accounting still disagrees and headroom is thinner than the config implies.
4. **vLLM's DSv4 KV budget check over-estimates ~62x** (`_max_memory_usage_bytes_from_groups`,
   the `UniformTypeKVCacheSpecs` branch: every group billed the *global* max
   layer-tuple count). Measured at two context lengths. This forces
   `gpu_memory_utilization=0.67` where the reference uses 0.4. Worth an upstream
   report; details in `dsv4_mmlu_pro_bringup.md` Finding 1.
5. **DP attention.** Not implemented here (`dp_attention` appears nowhere in
   `src/`). Estimate: ~1 week for a correctness prototype, 2–4 weeks
   production, with the risk concentrated in DSv4's custom caches (SWA overlay,
   compressor aliasing) which all assume replicated KV. **Profile a decode step
   before committing to this** — it only helps if the batch-32 saturation is
   KV-memory bound rather than MoE/GMM bound.

## 5. How to run

```bash
scripts/dsv4/serve_dsv4.sh [log]          # start a server, wait for health
LIMIT=0.1 scripts/dsv4/run_mmlu_pro.sh    # server + lm_eval + teardown
scripts/dsv4/perf_offline.py              # throughput sweep, one model load
scripts/dsv4/chain_mmlu_bringup.sh        # config arms, then smoke + full eval
```

Config and the reasoning behind every non-reference setting:
`scripts/dsv4/dsv4_env.sh` and `scripts/dsv4/README.md`.

## 6. Traps this session paid for

- **A traceback into our own code names the answer.** The KV over-allocation
  cause sat in a comment at the failing line, with the exact ratio, for an hour
  of wrong theories. Read the line first.
- **Dump everything at the first failure.** Instrumentation was added in three
  passes, each costing a ~14-minute model load. Worse, two of those dumps sat
  *after* the allocation loop and so never ran on the failures they targeted.
- **`pkill -f "vllm serve"` kills its own caller** when the caller's command
  line contains that string. Fixed in `dsv4_kill_server`; the same trap bites
  interactively (use `[V]LLM::`).
- **Preflight the eval harness.** A missing `lm_eval[api]` extra (tenacity)
  wasted a full server startup; `chain_mmlu_bringup.sh` now checks in seconds.
- **`--load-format dummy` saves less than expected** (~1.5 min, not ~11): it
  skips the safetensors read but not `process_weights_after_loading`, which is
  the expensive MoE FP4 stage.

## 7. Uncommitted / not-in-repo

- `tpu-inference/perf_offline_ref.py` lives in the *reference* repo, uncommitted.
- `examples/run_dsv4_inference.py` is still untracked here despite being the
  canonical correctness harness (memory: it is the one historically-coherent
  harness). Decide whether it should be tracked.
- Several older untracked docs (`MODEL_BRINGUP.md`, `handoff_refactor.md`,
  `pr_review_tracking.md`, ...) predate this session and were left alone.
