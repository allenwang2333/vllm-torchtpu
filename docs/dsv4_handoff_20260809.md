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

### Reference comparison (2026-08-09)

`tpu-inference/perf_offline_ref.py`, identical parameters, same node:

| batch | ours decode tok/s | reference decode tok/s | gap |
|---|---|---|---|
| 1 | 0.71 | 1.24 | 1.75x |
| 8 | 5.25 | 9.95 | 1.90x |
| 32 | 16.29 | 34.48 | 2.12x |
| 128 | 22.18 | 92.20 | **4.16x** |
| scaling b1 -> b128 | 31.3x | **74.2x** | — |

Total throughput at batch 128: **200 tok/s ours vs 830 tok/s reference**.
Reference wall time barely moves (51.5 s at batch 1 -> 88.9 s at 128); ours
grows 4x (90 s -> 369 s).

**The gap is a scaling failure, not a constant factor.** ~1.8x at batch 1 looks
like fixed per-step overhead, but the reference keeps scaling to 128 while we
flatten after ~32. Something in our path grows with batch where theirs does
not. That is the thing to profile next, and it is worth more than any of the
remaining correctness items for a production story.

**Confound, and it cannot be removed:** the reference refuses to run MLA
without DP attention (hard config validation -- see §4.2), so this is
ours-TP-only against theirs-DP-on. DP attention plausibly explains part of the
scaling difference (with `num_kv_heads=1`, TP cannot shard KV, so every rank of
ours holds the whole KV while each of theirs holds ~1/8), but nothing here
proves that, and a profile could just as easily land on the MoE/GMM path.
Do not quote "4x" as a clean like-for-like result.

Both numbers include a low-batch artifact: at batch 1 *both* implementations
are ~1 tok/s, because a graph compiled for `max_num_seqs` runs one real
sequence. That is not a defect unique to this repo.

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

1. **MMLU-Pro accuracy.** The original ask, still unanswered.
   `scripts/dsv4/run_mmlu_pro.sh` works end to end (requests answered, harness
   preflighted). The reference ran a **10% subset**
   (`ds_v4_flash_mmlu_pro_10pct_fixed`), so use `LIMIT=0.1` for comparability:
   ~1,200 questions, ~3 h at current throughput. The full 12,032 would be ~30 h.
   Run the needle control at *this* config alongside it, or the number cannot
   be interpreted (§3).

2. **Profile the batch-128 scaling gap.** We flatten after batch ~32 while the
   reference scales to 128 — 4.16x apart at 128 (§2). For a production story
   this outranks everything below, and it decides item 4: profile one decode
   step at batch 128 and find whether the time sits in attention (KV-bound, so
   DP attention would help) or in the MoE/GMM path (it would not).

3. **Residual over-allocation.** We still plan 54.48 GiB against vLLM's
   44.28 GiB budget (1.23x). It fits in physical HBM so it serves, but the
   accounting still disagrees and headroom is thinner than the config implies.
   The `DSV4_DUMP_KV_PLAN=1` PLAN lines report this ratio directly.

4. **DP attention.** Not implemented here (`dp_attention` appears nowhere in
   `src/`). Estimate: ~1 week for a correctness prototype, 2–4 weeks
   production, with the risk concentrated in DSv4's custom caches (SWA overlay,
   compressor aliasing) which all assume replicated KV. **Do item 2 first** —
   this only pays off if the saturation is KV-bound.

5. **vLLM's DSv4 KV budget check over-estimates ~62x**
   (`_max_memory_usage_bytes_from_groups`, the `UniformTypeKVCacheSpecs`
   branch: every group billed the *global* max layer-tuple count). Measured at
   two context lengths. Forces `gpu_memory_utilization=0.67` where the
   reference uses 0.4. Worth an upstream report; details in
   `dsv4_mmlu_pro_bringup.md` Finding 1.

6. **Consider a CPU-only parity test against the reference's allocation plan.**
   Three bugs now (this one, the `is_unused` cache-spec bug, the SWA overlay)
   were silent divergences from `tpu_inference/runner/kv_cache_manager.py`.
   Their spec/shape/alias decisions and ours can be computed and diffed without
   a TPU, turning that whole class from "a 14-minute run dies" into a failing
   unit test.

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
