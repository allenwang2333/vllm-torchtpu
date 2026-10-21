# ix-curve-oct21-merged: one tree with the code behind the 2026-10-21 IX P/D curve

This branch is upstream `infx-qwen35-agentx-disagg-hillclimbing` at `e1c7027a7` plus every functional change behind the 2026-10-21 Qwen3.5-397B P/D curve (bands A to H, p90 interactivity 55 to 937 tok/s) that upstream did not already have. The changes come from two code lineages, which are described below.

## Read this first: what has and has not been tested

- **This merged tree has not been curve-measured.** Each curve point in the table below was measured on its own source tree, which the table names. Two of those source trees (bands D to H) sit on a different base than upstream. Treat this branch as "the same code changes, applied to upstream", not as "the tree the numbers came from".
- **Unit tests:** see "Unit tests" at the end of this file.
- **GSM8K gate:** pending; the result will be added here. Gate job: not launched yet.

## What is in the branch

| Commit | Change | Origin |
|---|---|---|
| `c297c20e0` | `TPU_RUNNER_STAGE_TIMING_MODE=auto\|on\|off` override for the KV-transfer stage timings | combined lineage, d5c40130f (Shrihari M) |
| `4d61bcd7d` | batched_rpa: `TPU_RPA_PREFILL_BLOCKS` overrides the prefill/mixed tiles (default unset = upstream tiles) | prefill-kernel agent (71d6e2c75; same change as final-lineage 10a2d52) |
| `f596f8587` | qwen35_megakernel: single-call verify buckets up to 112 rows, rsag + deeper dense ring for 72..112 rows, static-bank EP refills | port of the final tree's 6fc248f / e25fc0d / 236115c onto the upstream megakernel (46e5e0103) |
| `2f16fe452` | R4 defaults-on: hierarchical rsag (HR) + hx collectives on 8-core meshes | D-mid / Integrator-1 (fc4021e53, smallb-r4-defaults-on @ 56b2bfe) |
| `f3c9b3a34` | PR #2243: quadratic attention cost in the PP8 chunk scheduler and the toy-proxy prefix router | Nan Wei (PR #2243, 463256968) |
| `ed366ff3a` | crash fix v3: a prefill-only MTP worker drops draft-placeholder rows padded onto a final prompt token (band-A crash: computed 126208 / scheduled 2 / prompt 126209) | prefill PP agent (0ad1965c2; same change as final-lineage df41429 / bb3f76f) |
| `d01053f51` | R5: runner stage timing off by default under P/D, merged with the stage-timing mode switch | D-mid (0447ff5 / fc49ff3) |
| `7c5440408` | `examples/disagg/toy_proxy_server_llmd.py`: the final lineage's llm-d proxy, verbatim from final-p2 `10a2d52` | Shrihari M (final lineage: ae5e50b, 12d7fb6, b8692e7) |
| `0b1601c6f` | PR #2243 quadratic prefill cost in the llm-d proxy's `route_prefill` (+ test) | PR-2243 agent (1215a7a) |
| `6c4f6dd73` | quad cost on by default in the llm-d proxy (`PROXY_PREFILL_QUAD_ATTN_COEFF=1.23e-5`) | PR-2243 agent (dd457e6, proxy part only) |
| `1b533d2f7` | `toy_proxy_server.py` runs `toy_proxy_server_llmd.py` when `PROXY_PREFILL_ROUTING` or `PROXY_DECODE_ROUTING` is `llmd` | merge agent |
| `79c96bd8e` | llm-d proxy reads 6 engine sizing variables as `PROXY_<name>` when the standard recipe does not pass them | merge agent |

Upstream already had these, so they were **not** re-applied:

- Benliang's padded-chunk fix (`e1c7027a7`).
- The covering draft-loop bucket (D-small `ea4c3b9`): upstream `9ba9e6bb7` has the same `_covering_buckets` logic, and also applies it to the verify buckets.
- The ~24 final-lineage commits by Shrihari M that Benliang ported upstream under other SHAs (decode_v2, KV prefetch, PAW sort, timeouts, tcmalloc release, GDN demotion and others).

Several of those ports changed a default into an opt-in knob. That is why the final-lineage configs need the "upstream knob set" listed below.

Not taken from the final lineage:

- The scheduler port of PR #2243 (`9fcfbdd`); this branch carries the PR's own upstream form instead.
- The `TPU_PP_QUAD_COST` switch, which exists only in that port.
- The bench-harness scripts (`370e133`, the script part of `dd457e6`).

### Two proxies

Upstream's `toy_proxy_server.py` has no llm-d router. Before this merge it treated `PROXY_*_ROUTING=llmd` as round robin. Every band D to H point ran `llmd` on both legs, using the final lineage's proxy. The two proxies read the same environment differently, so both are kept:

- **`PROXY_PREFILL_ROUTING` / `PROXY_DECODE_ROUTING` = `llmd` on either leg:** `toy_proxy_server.py` runs `toy_proxy_server_llmd.py` and prints `toy_proxy_server: routing=llmd, running the llm-d proxy ...` in the proxy log.
- **Any other value (`prefix_hash`, `round_robin`):** the upstream proxy runs, unchanged. Bands A and C used `prefix_hash`.
- **Band B caveat:** the band B point (o6, 1P1D) set `llmd` on the upstream proxy. That proxy fell back to round robin, which at one pod per leg behaves the same as `prefix_hash`. To reproduce band B on this tree, set `PROXY_PREFILL_ROUTING=prefix_hash PROXY_DECODE_ROUTING=prefix_hash`. With `llmd`, it would run the llm-d proxy instead.

## The curve (main-dash showcase `ix-tpu-pd-showcase-oct21`, 15 points)

Mtok/$ = tok/s/chip ÷ 336.1. Cap = `MAX_NUM_SEQS_DECODE` = `DECODE_TPU_SCHED_MK_MAX_RUNNING` = `PROXY_DECODE_MK_CEILING`; the P/D scheduler asserts that max-num-seqs equals the MK cap, so always set all three together. `c` = `CLIENT_CONCURRENCIES`. Prefill is always PP8 × TP1 (`PREFILL_VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4`). Decode is always TP8 + DCP4 with the megakernel. The full environment of any point is the `cmd:` line of `cdk job desc <job>`.

| Band | Mtok/$ @ p90 | Job | P:D | Cap | c | MK_MODE | Routing | Measured on (tree @ commit) | Distinctive env |
|---|---|---|---|---|---|---|---|---|---|
| A | 191.8 @ 58.7 | `j-895e2679` | 4P3D | 112 | 520 | ep | prefix_hash | o6q @ `69c56a021` (no R4) | E6: `DECODE_TPU_SP_MIN_ROWS=352`, `DECODE_TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=44`, placement `expert_placement_dp8_ep8.json`; `COMPILE_SIZES_DECODE` without 448 |
| B | 182.1 @ 88.8 | `j-c7c560d8` | 1P1D | 64 | 128 | ep | llmd (ran as round robin, see band B caveat) | combined4 @ `46e5e0103` (no R4, no PR #2243, no fix v3) | P5: `LIBTPU_EXTRA_PREFILL=--xla_tpu_rematerialization_min_size_in_bytes=1099511627776` |
| C | 164.1 @ 153.4 | `j-ab2fbb78` | 2P3D | 28 | 224 | ep | prefix_hash | c8q @ `0ad1965c2` (= this branch's first 6 commits) | placement `..._c64_mk2p.json`, o6 prefill env |
| C | 160.8 @ 191.4 | `j-024447b4` | 2P3D | 20 | 216 | ep | prefix_hash | c8q @ `0ad1965c2` | as above |
| C | 151.2 @ 196.4 | `j-64f6547b` | 2P3D | 20 | 200 | ep | prefix_hash | combined8 @ `fc4021e53` (R4, no PR #2243) | as above |
| D | 148.9 @ 209.0 | `j-99f2c652` | 3P4D | 16 | 320 | ep | llmd | final-p2 @ `10a2d52` | final-lineage env (below) |
| D | 141.7 @ 250.3 | `j-7f6a9686` | 3P5D | 12 | 320 | ep | llmd | final-p2 @ `10a2d52` | final-lineage env |
| D | 140.1 @ 293.4 | `j-6fc35bd1` | 3P5D | 10 | 300 | ep | llmd | dsfix3r5 @ `fc49ff3` (covering bucket + fix v3 + R5) | final-lineage env |
| E | 137.4 @ 336.6 | `j-b7adf1b1` | 2P4D | 8 | 224 | ep | llmd | fp2q @ `df41429` (PR #2243 port, quad on, fix v3) | final-lineage env |
| E | 134.9 @ 340.5 | `j-54a35f8c` | 2P4D | 8 | 208 | tp | llmd | fp2qr5 @ `81d4a7f` (fp2q + R5) | final-lineage env |
| F | 126.9 @ 400.2 | `j-a7b40ff9` | 2P4D | 6 | 208 | tp | llmd | fp2q-r5 @ `0447ff5` | final-lineage env |
| G | 109.6 @ 646.8 | `j-376fda2d` | 2P5D | 3 | 208 | tp | llmd | fp2q-r5 @ `0447ff5` | final-lineage env |
| G/H | 88.8 @ 819.6 | `j-4c743558` | 2P6D | 2 | 184 | tp | llmd | fp2q-r5 @ `0447ff5` | final-lineage env |
| H | 45.5 @ 920.9 | `j-1476661f` | 1P3D | 1 | 52 | tp | llmd | fp2q-r5 @ `0447ff5` | final-lineage env; TTFT p95 17.8 s |
| H | 20.8 @ 937 | `j-6d9762a1` | 1P2D | 1 | 12 | tp | llmd | fp2q-r5 @ `0447ff5` | final-lineage env |

**MK_MODE rule:** TP for caps 1 to 4, EP for caps 8 and above; at cap 8, TP and EP tie (137.4 EP vs 136.5 TP on fp2q).

**Env common to every point** (the commands below carry all of it):

- `DECODE_TPU_RPA_DECODE_V2=1`, `DECODE_TPU_SCHED_KV_PREFETCH_WAITING=8`, `PREFILL_TPU_SCHED_PAW_SORT_WAITING=1`, `PREFILL_TPU_PP_CHUNK_SLACK=0.5`.
- `KV_OFFLOAD_BYTES_PREFILL=550000000000`, `PREFILL_TPU_RAIDEN_PP_MAX_CPU_BYTES=720000000000`.
- `PROXY_PREFILL_HBM_MAX_BLOCKS=1616`, `PROXY_PREFILL_DRAM_MAX_BLOCKS=15427`, `PROXY_DECODE_HBM_MAX_BLOCKS=27481`, `PROXY_PEAK_PREFILL_TOK_S=62000`, `PROXY_DECODE_MAX_INFLIGHT=256`.

**Differences between the two lineages' configs:**

| Knob | Bands A to C (combined) | Bands D to H (final) |
|---|---|---|
| `PREFILL_TPU_RPA_PREFILL_BLOCKS` | `256,16,1024,1` | `256,32,1024,1` |
| `DECODE_VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B` | 128 | 160 |
| `DECODE_VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS` | up to 64 | up to 80 |
| `PROXY_DECODE_ADMIT_POINT` | `after_prefill` | unset |
| upstream knob set (below) | set | unset (the final tree had these as defaults) |

**Upstream knob set:** `TPU_TCMALLOC_RELEASE_FREE_MEMORY=1 TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 TPU_DIST_CPU_TIMEOUT_S=10800 TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S=10800 VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM=0`.

- Upstream made each of these an opt-in knob when it ported the final-lineage commits. The final tree had them on by default, and it forced per-layer Raiden pool tags on both peers.
- **On this branch, the band D to H configs must add the whole set.** `TPU_RAIDEN_POOL_TAGS_PER_LAYER=1` matters most: the final configs set only the `PREFILL_` form, which the standard recipe drops, and the prefill and decode peers must agree on pool tags.

## Reproducing a point on this branch

**Bands A and C.** Use the recorded config with `MAPDIR` = this tree. This branch is c8q plus the final-lineage commits.

- The band A and band B source trees had no R4. Here, R4 is on. Its off switches are `VLLM_TPU_QWEN35_MK_HRSAG=0` and `VLLM_TPU_QWEN35_MK_HX=0`, but the standard recipe does not pass them.
- For band B, set `prefix_hash` on both legs (see the band B caveat above).

**Bands D to H.** Use the recorded config, plus the upstream knob set. Under the standard recipe, also add the `PROXY_<name>` sizing variables:

`PROXY_PIPELINE_PARALLEL_SIZE_PREFILL=8 PROXY_BLOCK_SIZE_PREFILL=2176 PROXY_BLOCK_SIZE_DECODE=640 PROXY_KV_OFFLOAD_BYTES_PREFILL=550000000000 PROXY_DECODE_CONTEXT_PARALLEL_SIZE_DECODE=4 PROXY_MAX_NUM_SEQS_DECODE=<cap>`

Known differences from the trees these points were measured on:

- **Engine base.** The final lineage sits on a squashed Benliang base (final-p2), not on upstream. Its megakernel, runner and offload code reach this branch only as Benliang's upstream re-implementations. One example: every final config sets `DECODE_VLLM_TPU_QWEN35_MK_WIDE_160=1`, which only the final tree reads; this tree ignores it.
- **Quad cost.** `j-99f2c652`, `j-7f6a9686` and `j-6fc35bd1` ran without the quad cost in the llm-d proxy. Here it is on; set `PROXY_PREFILL_QUAD_ATTN_COEFF=0` to match.
- **final-p2 points.** `j-99f2c652` and `j-7f6a9686` also ran without fix v3 and without R5.

## Defaults that change behaviour

| Change | Default on this branch | How to turn it off |
|---|---|---|
| R5: runner stage timing under P/D | **off** (a default-on change; timing only, no numerics) | `TPU_RUNNER_STAGE_TIMING_MODE=on`, or `TPU_RUNNER_STAGE_TIMING=1` |
| PR #2243 scheduler (quadratic chunk cost, PAW sort, SRPF, PrefixAwareScheduler upgrade on PP > 1) | **on** whenever the PP chunk scheduler runs | no switch in the PR's own form |
| PR #2243 quad cost in both proxies | **on**, `PROXY_PREFILL_QUAD_ATTN_COEFF=1.23e-5` | `PROXY_PREFILL_QUAD_ATTN_COEFF=0` |
| R4 HR + hx collectives (8-core meshes) | **on** | `VLLM_TPU_QWEN35_MK_HRSAG=0`, `VLLM_TPU_QWEN35_MK_HX=0` |
| MK 112-row single-call buckets, rsag wide, EP fast refill | **on** | `VLLM_TPU_QWEN35_MK_MAX_SINGLE_B=80`, `VLLM_TPU_QWEN35_MK_RSAG_WIDE=0`, `VLLM_TPU_QWEN35_MK_EP_FAST_REFILL=0` |
| `TPU_RPA_PREFILL_BLOCKS` | off (unset = upstream tiles) | — |
| crash fix v3 | always on; it acts only on prefill-only MTP workers | none |
| llm-d proxy dispatch | only when routing is `llmd` | use `prefix_hash` |

## Using the standard recipe (cloud-devkit-recipes master, `agentx-397b-pd-baseline`)

In cloud-devkit-recipes master, `infx-qwen35-agentx` is the single-host aggregated sweep, not P/D. Use `agentx-397b-pd-baseline` for P/D. That recipe applies `--map-dir <tree>:*:/workspace/vllm-torchtpu` as an overlay.

The curve itself ran on fork recipes. I rendered the three configs below against master (`22ec9222`) and diffed them against the curve jobs' renders. The following variables do not reach the containers on master:

| Variable | Effect when dropped |
|---|---|
| `PREFILL_TPU_RPA_PREFILL_BLOCKS` | Prefill uses the default tiles. The commit measured the override ~20% faster on the prefill kernel at 87K context (j-10897a30). **Needs a recipe passthrough.** |
| `PREFILL_VLLM_TPU_LIGHTWEIGHT_STEP_STATS` | Lightweight step stats are off. Not measured. |
| `LIBTPU_EXTRA_PREFILL` | The band A to C prefill libtpu flag (P5) is not applied. |
| `PREFILL_TPU_RAIDEN_POOL_TAGS_PER_LAYER` | Use the global `TPU_RAIDEN_POOL_TAGS_PER_LAYER=1`, which master does pass. |
| `DECODE_TPU_RUNNER_STAGE_TIMING_MODE`, `TPU_RUNNER_STAGE_TIMING` | No effect by default, because R5 already makes timing off; the override cannot be set. |
| The R4 / MK off switches above | Cannot be set. |
| Proxy: `PIPELINE_PARALLEL_SIZE_PREFILL`, `BLOCK_SIZE_PREFILL`, `BLOCK_SIZE_DECODE`, `KV_OFFLOAD_BYTES_PREFILL`, `DECODE_CONTEXT_PARALLEL_SIZE_DECODE`, `MAX_NUM_SEQS_DECODE` | Handled: give them as `PROXY_<name>` (llm-d proxy only; the upstream proxy does not read them). |

Master also differs from the fork recipes in ways this tree cannot control:

- It loads weights from `gs://cloud-devkit/ckpt/hf/...` instead of the runai_streamer PVC.
- It installs pinned libtpu and tpu-raiden wheels at pod start.
- It sets `TPU_RAIDEN_TIER_DEMOTE_STATE=1` on decode.

None of these has been measured on this branch.

**Expert-placement files.** `DECODE_TPU_EXPERT_PLACEMENT=/workspace/vllm-torchtpu/expert_placement/<file>` resolves on both master and the fork recipes. With `--map-dir`, the recipe copies `scripts/vllm/benchmark/inferencex/qwen3.5/expert_placement*.json` into that directory. All three files the curve used (`expert_placement_tp8_ep8_agentx_corr_c32.json`, `..._c64_mk2p.json`, `expert_placement_dp8_ep8.json`) are tracked there, and they are byte-identical (md5) to the untracked copies in our trees. The untracked `qwen35-397b-config/` directory in our trees is referenced by no config, recipe or code.

## Commands for three representative points (standard recipe)

Set `TREE` to your clone of this branch and `TAGS` to your tags (for example `TAGS=agent-jobs,<you>`).

- Each command is the curve job's own environment, with only these changes: `BENCH_NAME`, `COMPILE_CACHE_TAG`, and, for band H, the upstream knob set and the `PROXY_<name>` variables.
- Variables that master drops are left in, so the commands stay correct once master gains those passthroughs.
- Each command renders on master (`cdk recipe render`, exit 0).
- **None of these three has been run on this tree.**

### Band A: 4P3D, cap 112, c520 (j-895e2679, 191.8 @ 58.7 on o6q)

```bash
cdk job create agentx-397b-pd-baseline \
  --map-dir "$TREE":*:/workspace/vllm-torchtpu \
  --tags "$TAGS" --pending-timeout-seconds=43200 \
  SERVER_IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-dev@sha256:ba3be4a9d0ed3ca63bc2110503e7e71d5ca40b5fe197f292781e68d6a34cb055 \
  AVOID_SPOT=false \
  MODEL_SOURCE=pvc \
  VLLM_KV_CACHE_LAYOUT=HND \
  ENABLE_PREFIX_CACHING=true \
  ENABLE_PAW=true \
  PREFILL_ENABLE_PAW=true \
  DECODE_ENABLE_PAW=true \
  MTP=true \
  MTP_NUM_SPECULATIVE_TOKENS_PREFILL=1 \
  MTP_NUM_SPECULATIVE_TOKENS_DECODE=3 \
  CHIP_HOUR_USD=3.025 \
  GPU_MEMORY_UTILIZATION_PREFILL=0.92 \
  GPU_MEMORY_UTILIZATION_DECODE=0.92 \
  MAX_MODEL_LEN=262144 \
  RAIDEN_MAX_TRANSFER_TOKENS=262144 \
  MAX_NUM_SEQS_PREFILL=32 \
  MAX_NUM_BATCHED_TOKENS_PREFILL=8192 \
  MAX_NUM_BATCHED_TOKENS_DECODE=2048 \
  ASYNC_SCHEDULING=true \
  ASYNC_SCHEDULING_DECODE=true \
  SC_ALL_GATHER_OFFLOAD_DECODE=true \
  PREFILL_CONTEXT_PARALLEL_SIZE_PREFILL=1 \
  DATA_PARALLEL_SIZE_PREFILL=1 \
  CP_KV_CACHE_INTERLEAVE_SIZE_PREFILL=1 \
  TPU_ENABLE_GDN_DYNAMIC_TILING=1 \
  FAILED_REQUEST_THRESHOLD=0.25 \
  SERVER_MEMORY=850Gi \
  SERVER_MEMORY_LIMIT=925Gi \
  PREFILL_RAIDEN_NUMA_POLICY=preferred \
  DECODE_RUNAI_STREAMER_MEMORY_LIMIT=5368709120 \
  PREFILL_TPU_EVICT_WEIGHTS_PAGE_CACHE=1 \
  DECODE_TPU_EVICT_WEIGHTS_PAGE_CACHE=1 \
  PREFILL_TPU_RAIDEN_TIER_DEMOTE_STATE=1 \
  COMPILE_CACHE_GCS=cloud-devkit/compile-cache \
  COMPILE_CACHE_TAG=ix-curve-merged-bandA-4p3d-c520 \
  PREFILL_TPU_MOE_LOWTOK=0 \
  PREFILL_TPU_GDN_FUSED_NORM_QUANT=1 \
  PREFILL_TPU_MOE_SKIP_PADDED_TOKENS=1 \
  PREFILL_VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 \
  PREFILL_TPU_KERNEL_PAGE_TOKENS=128 \
  PREFILL_TPU_GDN_CONV_QK_PAIR_LAYOUT=1 \
  PREFILL_TPU_RAIDEN_PREFIX_AWARE_LOAD=1 \
  PREFILL_TPU_RAIDEN_ASYNC_COMPLETE=1 \
  PREFILL_TPU_PREFILL_SKIP_VERIFY_WARMUP=1 \
  PREFILL_TPU_SCHED_PAW_AGING_TOK_PER_S=3000 \
  PREFILL_TPU_SCHED_PAW_MAX_WAIT_S=10.0 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE=1 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE_ENTRIES=2048 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE_MAX_TOKENS=64000000 \
  PREFILL_VLLM_TPU_QWEN35_MEGAKERNEL=0 \
  PREFILL_VLLM_TPU_QWEN35_MTP_MK=0 \
  BLOCK_SIZE_PREFILL=2176 \
  BLOCK_SIZE_DECODE=640 \
  ASYNC_SCHEDULING_PREFILL=false \
  PIPELINE_PARALLEL_SIZE_PREFILL=8 \
  TENSOR_PARALLEL_SIZE_PREFILL=1 \
  'COMPILE_SIZES_PREFILL=2176, 4352, 6528, 8192' \
  PREFILL_VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4 \
  PREFILL_TPU_RPA_NATIVE_FP8_PREFILL=0 \
  PREFILL_TPU_PP_CHUNK_SLACK=0.5 \
  KV_OFFLOAD_BYTES_PREFILL=550000000000 \
  PREFILL_TPU_RAIDEN_MAX_CPU_BYTES=550000000000 \
  PREFILL_TPU_RAIDEN_PP_MAX_CPU_BYTES=720000000000 \
  PREFILL_RUNAI_STREAMER_MEMORY_LIMIT=5368709120 \
  PREFILL_TPU_OFFLOAD_MAX_ASYNC_LOAD_REQS=16 \
  PREFILL_QX_OFFLOAD_FENCE_AFTER_DRAFT=1 \
  PREFILL_TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 \
  ENABLE_EXPERT_PARALLEL_PREFILL=false \
  PREFILL_USE_MOE_FUSED_EP_KERNEL=0 \
  PREFILL_TPU_MOE_SEQUENCE_PARALLEL=0 \
  PROXY_PREFILL_HBM_MAX_BLOCKS=1616 \
  PROXY_PREFILL_DRAM_MAX_BLOCKS=15427 \
  TENSOR_PARALLEL_SIZE_DECODE=8 \
  DATA_PARALLEL_SIZE_DECODE=1 \
  COMPILE_SIZES_DECODE=4,8,16,24,32,48,64,96,128,160,192,256,320,384,512,2048 \
  ENABLE_EXPERT_PARALLEL_DECODE=true \
  DECODE_USE_MOE_FUSED_EP_KERNEL=1 \
  DECODE_MOE_FUSED_EP_KERNEL_IMPL=v3 \
  DECODE_RAGGED_GATHER_REDUCE_VERSION=v3 \
  DECODE_TPU_MOE_SEQUENCE_PARALLEL=1 \
  DECODE_TPU_SP_FP8_GATHER=1 \
  DECODE_TPU_SP_MIN_ROWS=352 \
  DECODE_TPU_SMALL_ALL_REDUCE=1 \
  DECODE_TPU_GDN_FUSED_NORM_QUANT=1 \
  DECODE_TPU_MOE_SKIP_PADDED_TOKENS=1 \
  DECODE_VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 \
  DECODE_VLLM_TPU_QWEN35_MEGAKERNEL=1 \
  DECODE_VLLM_TPU_QWEN35_MK_MODE=ep \
  DECODE_VLLM_TPU_QWEN35_MTP_MK=1 \
  DECODE_VLLM_TPU_QWEN35_MK_SPLIT_WIDE=1 \
  DECODE_VLLM_TPU_QWEN35_MK_BUCKETS=0 \
  DECODE_TPU_KERNEL_PAGE_TOKENS=128 \
  DECODE_TPU_FUSE_POST_ATTN_NORM=1 \
  DECODE_TPU_FUSE_PRE_ATTN_NORM=1 \
  DECODE_TPU_SCHED_FINISH_DECODE_STEPS=0 \
  DECODE_TPU_SCHED_WARM_STREAK=0 \
  DECODE_TPU_RAIDEN_PREFIX_AWARE_LOAD=1 \
  DECODE_TPU_RAIDEN_ASYNC_COMPLETE=1 \
  DECODE_TPU_EXPERT_PLACEMENT=/workspace/vllm-torchtpu/expert_placement/expert_placement_dp8_ep8.json \
  MAX_NUM_SEQS_DECODE=112 \
  DECODE_TPU_SCHED_MK_MAX_RUNNING=112 \
  PROXY_PREFILL_ROUTING=prefix_hash \
  PROXY_DECODE_ROUTING=prefix_hash \
  PROXY_PEAK_PREFILL_TOK_S=62000 \
  PROXY_DECODE_MK_CEILING=112 \
  PROXY_DECODE_MAX_INFLIGHT=256 \
  DECODE_CONTEXT_PARALLEL_SIZE_DECODE=4 \
  CP_KV_CACHE_INTERLEAVE_SIZE_DECODE=128 \
  DECODE_TPU_MOE_LOWTOK=1 \
  DECODE_TPU_MOE_LOWTOK_MAX_TOKENS=448 \
  DECODE_VLLM_TPU_QWEN35_MK_WIDE_160=1 \
  DECODE_VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B=128 \
  DECODE_VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS=4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,96,104,112 \
  DECODE_VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS=4,8,12,16,20,24,32,40,48,56,64 \
  PROXY_DECODE_HBM_MAX_BLOCKS=27481 \
  NUM_PREFILL_SERVERS=4 \
  NUM_DECODE_SERVERS=3 \
  SERVER_WAIT_S=14400 \
  CLIENT_CONCURRENCIES=520 \
  FAST=false \
  ENABLE_XPROF=false \
  ENABLE_PERFETTO=false \
  MAX_RESTARTS=6 \
  SYNTHETIC_ACCEPTANCE_LENGTH=3.39 \
  PREFILL_TPU_SCHED_PAW_SORT_WAITING=1 \
  PREFILL_TPU_SCHED_PAW_SORT_WINDOW=128 \
  PREFILL_VLLM_TPU_LIGHTWEIGHT_STEP_STATS=1 \
  DECODE_TPU_SCHED_KV_PREFETCH_WAITING=8 \
  DECODE_TPU_RPA_DECODE_V2=1 \
  TPU_TCMALLOC_RELEASE_FREE_MEMORY=1 \
  TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 \
  TPU_DIST_CPU_TIMEOUT_S=10800 \
  TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S=10800 \
  VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM=0 \
  DECODE_TPU_RUNNER_STAGE_TIMING_MODE=off \
  PROXY_DECODE_ADMIT_POINT=after_prefill \
  PREFILL_TPU_RPA_PREFILL_BLOCKS=256,16,1024,1 \
  LIBTPU_EXTRA_PREFILL=--xla_tpu_rematerialization_min_size_in_bytes=1099511627776 \
  DECODE_TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=44 \
  BENCH_NAME=qwen3.5-397b-agentx-pd-ix-curve-merged-bandA-4p3d-c520
```

### Band C: 2P3D, cap 28, c224 (j-ab2fbb78, 164.1 @ 153.4 on c8q)

```bash
cdk job create agentx-397b-pd-baseline \
  --map-dir "$TREE":*:/workspace/vllm-torchtpu \
  --tags "$TAGS" --pending-timeout-seconds=43200 \
  SERVER_IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-dev@sha256:ba3be4a9d0ed3ca63bc2110503e7e71d5ca40b5fe197f292781e68d6a34cb055 \
  AVOID_SPOT=false \
  MODEL_SOURCE=pvc \
  VLLM_KV_CACHE_LAYOUT=HND \
  ENABLE_PREFIX_CACHING=true \
  ENABLE_PAW=true \
  PREFILL_ENABLE_PAW=true \
  DECODE_ENABLE_PAW=true \
  MTP=true \
  MTP_NUM_SPECULATIVE_TOKENS_PREFILL=1 \
  MTP_NUM_SPECULATIVE_TOKENS_DECODE=3 \
  CHIP_HOUR_USD=3.025 \
  GPU_MEMORY_UTILIZATION_PREFILL=0.92 \
  GPU_MEMORY_UTILIZATION_DECODE=0.92 \
  MAX_MODEL_LEN=262144 \
  RAIDEN_MAX_TRANSFER_TOKENS=262144 \
  MAX_NUM_SEQS_PREFILL=32 \
  MAX_NUM_BATCHED_TOKENS_PREFILL=8192 \
  MAX_NUM_BATCHED_TOKENS_DECODE=2048 \
  ASYNC_SCHEDULING=true \
  ASYNC_SCHEDULING_DECODE=true \
  SC_ALL_GATHER_OFFLOAD_DECODE=true \
  PREFILL_CONTEXT_PARALLEL_SIZE_PREFILL=1 \
  DATA_PARALLEL_SIZE_PREFILL=1 \
  CP_KV_CACHE_INTERLEAVE_SIZE_PREFILL=1 \
  TPU_ENABLE_GDN_DYNAMIC_TILING=1 \
  FAILED_REQUEST_THRESHOLD=0.25 \
  SERVER_MEMORY=850Gi \
  SERVER_MEMORY_LIMIT=925Gi \
  PREFILL_RAIDEN_NUMA_POLICY=preferred \
  DECODE_RUNAI_STREAMER_MEMORY_LIMIT=5368709120 \
  PREFILL_TPU_EVICT_WEIGHTS_PAGE_CACHE=1 \
  DECODE_TPU_EVICT_WEIGHTS_PAGE_CACHE=1 \
  PREFILL_TPU_RAIDEN_TIER_DEMOTE_STATE=1 \
  COMPILE_CACHE_GCS=cloud-devkit/compile-cache \
  COMPILE_CACHE_TAG=ix-curve-merged-bandC-2p3d-c224 \
  PREFILL_TPU_MOE_LOWTOK=0 \
  PREFILL_TPU_GDN_FUSED_NORM_QUANT=1 \
  PREFILL_TPU_MOE_SKIP_PADDED_TOKENS=1 \
  PREFILL_VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 \
  PREFILL_TPU_KERNEL_PAGE_TOKENS=128 \
  PREFILL_TPU_GDN_CONV_QK_PAIR_LAYOUT=1 \
  PREFILL_TPU_RAIDEN_PREFIX_AWARE_LOAD=1 \
  PREFILL_TPU_RAIDEN_ASYNC_COMPLETE=1 \
  PREFILL_TPU_PREFILL_SKIP_VERIFY_WARMUP=1 \
  PREFILL_TPU_SCHED_PAW_AGING_TOK_PER_S=3000 \
  PREFILL_TPU_SCHED_PAW_MAX_WAIT_S=10.0 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE=1 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE_ENTRIES=2048 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE_MAX_TOKENS=64000000 \
  PREFILL_VLLM_TPU_QWEN35_MEGAKERNEL=0 \
  PREFILL_VLLM_TPU_QWEN35_MTP_MK=0 \
  BLOCK_SIZE_PREFILL=2176 \
  BLOCK_SIZE_DECODE=640 \
  ASYNC_SCHEDULING_PREFILL=false \
  PIPELINE_PARALLEL_SIZE_PREFILL=8 \
  TENSOR_PARALLEL_SIZE_PREFILL=1 \
  'COMPILE_SIZES_PREFILL=2176, 4352, 6528, 8192' \
  PREFILL_VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4 \
  PREFILL_TPU_RPA_NATIVE_FP8_PREFILL=0 \
  PREFILL_TPU_PP_CHUNK_SLACK=0.5 \
  KV_OFFLOAD_BYTES_PREFILL=550000000000 \
  PREFILL_TPU_RAIDEN_MAX_CPU_BYTES=550000000000 \
  PREFILL_TPU_RAIDEN_PP_MAX_CPU_BYTES=720000000000 \
  PREFILL_RUNAI_STREAMER_MEMORY_LIMIT=5368709120 \
  PREFILL_TPU_OFFLOAD_MAX_ASYNC_LOAD_REQS=16 \
  PREFILL_QX_OFFLOAD_FENCE_AFTER_DRAFT=1 \
  PREFILL_TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 \
  ENABLE_EXPERT_PARALLEL_PREFILL=false \
  PREFILL_USE_MOE_FUSED_EP_KERNEL=0 \
  PREFILL_TPU_MOE_SEQUENCE_PARALLEL=0 \
  PROXY_PREFILL_HBM_MAX_BLOCKS=1616 \
  PROXY_PREFILL_DRAM_MAX_BLOCKS=15427 \
  TENSOR_PARALLEL_SIZE_DECODE=8 \
  DATA_PARALLEL_SIZE_DECODE=1 \
  'COMPILE_SIZES_DECODE=4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 2048' \
  ENABLE_EXPERT_PARALLEL_DECODE=true \
  DECODE_USE_MOE_FUSED_EP_KERNEL=1 \
  DECODE_MOE_FUSED_EP_KERNEL_IMPL=v3 \
  DECODE_RAGGED_GATHER_REDUCE_VERSION=v3 \
  DECODE_TPU_MOE_SEQUENCE_PARALLEL=1 \
  DECODE_TPU_SP_FP8_GATHER=1 \
  DECODE_TPU_SP_MIN_ROWS=512 \
  DECODE_TPU_SMALL_ALL_REDUCE=1 \
  DECODE_TPU_GDN_FUSED_NORM_QUANT=1 \
  DECODE_TPU_MOE_SKIP_PADDED_TOKENS=1 \
  DECODE_VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 \
  DECODE_VLLM_TPU_QWEN35_MEGAKERNEL=1 \
  DECODE_VLLM_TPU_QWEN35_MK_MODE=ep \
  DECODE_VLLM_TPU_QWEN35_MTP_MK=1 \
  DECODE_VLLM_TPU_QWEN35_MK_SPLIT_WIDE=1 \
  DECODE_VLLM_TPU_QWEN35_MK_BUCKETS=0 \
  DECODE_TPU_KERNEL_PAGE_TOKENS=128 \
  DECODE_TPU_FUSE_POST_ATTN_NORM=1 \
  DECODE_TPU_FUSE_PRE_ATTN_NORM=1 \
  DECODE_TPU_SCHED_FINISH_DECODE_STEPS=0 \
  DECODE_TPU_SCHED_WARM_STREAK=0 \
  DECODE_TPU_RAIDEN_PREFIX_AWARE_LOAD=1 \
  DECODE_TPU_RAIDEN_ASYNC_COMPLETE=1 \
  DECODE_TPU_EXPERT_PLACEMENT=/workspace/vllm-torchtpu/expert_placement/expert_placement_tp8_ep8_agentx_corr_c64_mk2p.json \
  MAX_NUM_SEQS_DECODE=28 \
  DECODE_TPU_SCHED_MK_MAX_RUNNING=28 \
  PROXY_PREFILL_ROUTING=prefix_hash \
  PROXY_DECODE_ROUTING=prefix_hash \
  PROXY_PEAK_PREFILL_TOK_S=62000 \
  PROXY_DECODE_MK_CEILING=28 \
  PROXY_DECODE_MAX_INFLIGHT=256 \
  DECODE_CONTEXT_PARALLEL_SIZE_DECODE=4 \
  CP_KV_CACHE_INTERLEAVE_SIZE_DECODE=128 \
  DECODE_TPU_MOE_LOWTOK=1 \
  DECODE_TPU_MOE_LOWTOK_MAX_TOKENS=448 \
  DECODE_VLLM_TPU_QWEN35_MK_WIDE_160=1 \
  DECODE_VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B=128 \
  DECODE_VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS=4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,96,104,112 \
  DECODE_VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS=4,8,12,16,20,24,32,40,48,56,64 \
  PROXY_DECODE_HBM_MAX_BLOCKS=27481 \
  NUM_PREFILL_SERVERS=2 \
  NUM_DECODE_SERVERS=3 \
  SERVER_WAIT_S=14400 \
  CLIENT_CONCURRENCIES=224 \
  FAST=false \
  ENABLE_XPROF=false \
  ENABLE_PERFETTO=false \
  MAX_RESTARTS=6 \
  SYNTHETIC_ACCEPTANCE_LENGTH=3.39 \
  PREFILL_TPU_SCHED_PAW_SORT_WAITING=1 \
  PREFILL_TPU_SCHED_PAW_SORT_WINDOW=128 \
  PREFILL_VLLM_TPU_LIGHTWEIGHT_STEP_STATS=1 \
  DECODE_TPU_SCHED_KV_PREFETCH_WAITING=8 \
  DECODE_TPU_RPA_DECODE_V2=1 \
  TPU_TCMALLOC_RELEASE_FREE_MEMORY=1 \
  TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 \
  TPU_DIST_CPU_TIMEOUT_S=10800 \
  TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S=10800 \
  VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM=0 \
  DECODE_TPU_RUNNER_STAGE_TIMING_MODE=off \
  PROXY_DECODE_ADMIT_POINT=after_prefill \
  PREFILL_TPU_RPA_PREFILL_BLOCKS=256,16,1024,1 \
  LIBTPU_EXTRA_PREFILL=--xla_tpu_rematerialization_min_size_in_bytes=1099511627776 \
  BENCH_NAME=qwen3.5-397b-agentx-pd-ix-curve-merged-bandC-2p3d-c224
```

### Band H: 1P3D, cap 1, c52 (j-1476661f, 45.5 @ 920.9 on fp2q-r5)

```bash
cdk job create agentx-397b-pd-baseline \
  --map-dir "$TREE":*:/workspace/vllm-torchtpu \
  --tags "$TAGS" --pending-timeout-seconds=43200 \
  SERVER_IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-dev@sha256:ba3be4a9d0ed3ca63bc2110503e7e71d5ca40b5fe197f292781e68d6a34cb055 \
  AVOID_SPOT=false \
  MODEL_SOURCE=pvc \
  VLLM_KV_CACHE_LAYOUT=HND \
  ENABLE_PREFIX_CACHING=true \
  ENABLE_PAW=true \
  PREFILL_ENABLE_PAW=true \
  DECODE_ENABLE_PAW=true \
  MTP=true \
  MTP_NUM_SPECULATIVE_TOKENS_PREFILL=1 \
  MTP_NUM_SPECULATIVE_TOKENS_DECODE=3 \
  CHIP_HOUR_USD=3.025 \
  GPU_MEMORY_UTILIZATION_PREFILL=0.92 \
  GPU_MEMORY_UTILIZATION_DECODE=0.92 \
  MAX_MODEL_LEN=262144 \
  RAIDEN_MAX_TRANSFER_TOKENS=262144 \
  MAX_NUM_SEQS_PREFILL=32 \
  MAX_NUM_BATCHED_TOKENS_PREFILL=8192 \
  MAX_NUM_BATCHED_TOKENS_DECODE=2048 \
  ASYNC_SCHEDULING=true \
  ASYNC_SCHEDULING_DECODE=true \
  SC_ALL_GATHER_OFFLOAD_DECODE=true \
  PREFILL_CONTEXT_PARALLEL_SIZE_PREFILL=1 \
  DATA_PARALLEL_SIZE_PREFILL=1 \
  CP_KV_CACHE_INTERLEAVE_SIZE_PREFILL=1 \
  TPU_ENABLE_GDN_DYNAMIC_TILING=1 \
  FAILED_REQUEST_THRESHOLD=0.25 \
  SERVER_MEMORY=850Gi \
  SERVER_MEMORY_LIMIT=925Gi \
  PREFILL_RAIDEN_NUMA_POLICY=preferred \
  DECODE_RUNAI_STREAMER_MEMORY_LIMIT=5368709120 \
  PREFILL_TPU_EVICT_WEIGHTS_PAGE_CACHE=1 \
  DECODE_TPU_EVICT_WEIGHTS_PAGE_CACHE=1 \
  PREFILL_TPU_RAIDEN_TIER_DEMOTE_STATE=1 \
  COMPILE_CACHE_GCS=cloud-devkit/compile-cache \
  COMPILE_CACHE_TAG=ix-curve-merged-bandH-1p3d-cap1-c52 \
  PREFILL_TPU_MOE_LOWTOK=0 \
  PREFILL_TPU_GDN_FUSED_NORM_QUANT=1 \
  PREFILL_TPU_MOE_SKIP_PADDED_TOKENS=1 \
  PREFILL_VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 \
  PREFILL_TPU_KERNEL_PAGE_TOKENS=128 \
  PREFILL_TPU_GDN_CONV_QK_PAIR_LAYOUT=1 \
  PREFILL_TPU_RAIDEN_PREFIX_AWARE_LOAD=1 \
  PREFILL_TPU_RAIDEN_ASYNC_COMPLETE=1 \
  PREFILL_TPU_PREFILL_SKIP_VERIFY_WARMUP=1 \
  PREFILL_TPU_SCHED_PAW_AGING_TOK_PER_S=3000 \
  PREFILL_TPU_SCHED_PAW_MAX_WAIT_S=10.0 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE=1 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE_ENTRIES=2048 \
  PREFILL_QX_TOKENIZE_PREFIX_CACHE_MAX_TOKENS=64000000 \
  PREFILL_VLLM_TPU_QWEN35_MEGAKERNEL=0 \
  PREFILL_VLLM_TPU_QWEN35_MTP_MK=0 \
  BLOCK_SIZE_PREFILL=2176 \
  BLOCK_SIZE_DECODE=640 \
  ASYNC_SCHEDULING_PREFILL=false \
  PIPELINE_PARALLEL_SIZE_PREFILL=8 \
  TENSOR_PARALLEL_SIZE_PREFILL=1 \
  'COMPILE_SIZES_PREFILL=2176, 4352, 6528, 8192' \
  PREFILL_VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4 \
  PREFILL_TPU_RPA_NATIVE_FP8_PREFILL=0 \
  PREFILL_TPU_PP_CHUNK_SLACK=0.5 \
  KV_OFFLOAD_BYTES_PREFILL=550000000000 \
  PREFILL_TPU_RAIDEN_MAX_CPU_BYTES=550000000000 \
  PREFILL_TPU_RAIDEN_PP_MAX_CPU_BYTES=720000000000 \
  PREFILL_RUNAI_STREAMER_MEMORY_LIMIT=5368709120 \
  PREFILL_TPU_OFFLOAD_MAX_ASYNC_LOAD_REQS=16 \
  PREFILL_QX_OFFLOAD_FENCE_AFTER_DRAFT=1 \
  PREFILL_TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 \
  ENABLE_EXPERT_PARALLEL_PREFILL=false \
  PREFILL_USE_MOE_FUSED_EP_KERNEL=0 \
  PREFILL_TPU_MOE_SEQUENCE_PARALLEL=0 \
  PROXY_PREFILL_HBM_MAX_BLOCKS=1616 \
  PROXY_PREFILL_DRAM_MAX_BLOCKS=15427 \
  TENSOR_PARALLEL_SIZE_DECODE=8 \
  DATA_PARALLEL_SIZE_DECODE=1 \
  'COMPILE_SIZES_DECODE=4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 2048' \
  ENABLE_EXPERT_PARALLEL_DECODE=true \
  DECODE_USE_MOE_FUSED_EP_KERNEL=1 \
  DECODE_MOE_FUSED_EP_KERNEL_IMPL=v3 \
  DECODE_RAGGED_GATHER_REDUCE_VERSION=v3 \
  DECODE_TPU_MOE_SEQUENCE_PARALLEL=1 \
  DECODE_TPU_SP_FP8_GATHER=1 \
  DECODE_TPU_SP_MIN_ROWS=512 \
  DECODE_TPU_SMALL_ALL_REDUCE=1 \
  DECODE_TPU_GDN_FUSED_NORM_QUANT=1 \
  DECODE_TPU_MOE_SKIP_PADDED_TOKENS=1 \
  DECODE_VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 \
  DECODE_VLLM_TPU_QWEN35_MEGAKERNEL=1 \
  DECODE_VLLM_TPU_QWEN35_MK_MODE=tp \
  DECODE_VLLM_TPU_QWEN35_MTP_MK=1 \
  DECODE_VLLM_TPU_QWEN35_MK_SPLIT_WIDE=1 \
  DECODE_VLLM_TPU_QWEN35_MK_BUCKETS=0 \
  DECODE_TPU_KERNEL_PAGE_TOKENS=128 \
  DECODE_TPU_FUSE_POST_ATTN_NORM=1 \
  DECODE_TPU_FUSE_PRE_ATTN_NORM=1 \
  DECODE_TPU_SCHED_FINISH_DECODE_STEPS=0 \
  DECODE_TPU_SCHED_WARM_STREAK=0 \
  DECODE_TPU_RAIDEN_PREFIX_AWARE_LOAD=1 \
  DECODE_TPU_RAIDEN_ASYNC_COMPLETE=1 \
  DECODE_TPU_EXPERT_PLACEMENT=/workspace/vllm-torchtpu/expert_placement/expert_placement_tp8_ep8_agentx_corr_c32.json \
  MAX_NUM_SEQS_DECODE=1 \
  DECODE_TPU_SCHED_MK_MAX_RUNNING=1 \
  PROXY_PREFILL_ROUTING=llmd \
  PROXY_DECODE_ROUTING=llmd \
  PROXY_PEAK_PREFILL_TOK_S=62000 \
  PROXY_DECODE_MK_CEILING=1 \
  PROXY_DECODE_MAX_INFLIGHT=256 \
  DECODE_CONTEXT_PARALLEL_SIZE_DECODE=4 \
  CP_KV_CACHE_INTERLEAVE_SIZE_DECODE=128 \
  DECODE_TPU_MOE_LOWTOK=1 \
  DECODE_TPU_MOE_LOWTOK_MAX_TOKENS=448 \
  DECODE_VLLM_TPU_QWEN35_MK_WIDE_160=1 \
  DECODE_VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B=160 \
  DECODE_VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS=4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,96,104,112 \
  DECODE_VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS=4,8,12,16,20,24,32,40,48,56,64,72,80 \
  PROXY_DECODE_HBM_MAX_BLOCKS=27481 \
  NUM_PREFILL_SERVERS=1 \
  NUM_DECODE_SERVERS=3 \
  SERVER_WAIT_S=14400 \
  CLIENT_CONCURRENCIES=52 \
  FAST=false \
  ENABLE_XPROF=false \
  ENABLE_PERFETTO=false \
  MAX_RESTARTS=6 \
  SYNTHETIC_ACCEPTANCE_LENGTH=3.39 \
  PREFILL_TPU_SCHED_PAW_SORT_WAITING=1 \
  PREFILL_TPU_SCHED_PAW_SORT_WINDOW=128 \
  PREFILL_VLLM_TPU_LIGHTWEIGHT_STEP_STATS=1 \
  DECODE_TPU_SCHED_KV_PREFETCH_WAITING=8 \
  DECODE_TPU_RPA_DECODE_V2=1 \
  PREFILL_TPU_RPA_PREFILL_BLOCKS=256,32,1024,1 \
  BENCH_NAME=qwen3.5-397b-agentx-pd-ix-curve-merged-bandH-1p3d-cap1-c52 \
  TPU_TCMALLOC_RELEASE_FREE_MEMORY=1 \
  TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 \
  TPU_DIST_CPU_TIMEOUT_S=10800 \
  TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S=10800 \
  VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM=0 \
  PROXY_PIPELINE_PARALLEL_SIZE_PREFILL=8 \
  PROXY_BLOCK_SIZE_PREFILL=2176 \
  PROXY_BLOCK_SIZE_DECODE=640 \
  PROXY_KV_OFFLOAD_BYTES_PREFILL=550000000000 \
  PROXY_DECODE_CONTEXT_PARALLEL_SIZE_DECODE=4 \
  PROXY_MAX_NUM_SEQS_DECODE=1
```

## Unit tests

The unit tests are running now; the results will be added here.
