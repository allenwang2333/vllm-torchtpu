# DSv4-Pro agentx at concurrency 16: serving settings

Settings for `infx-deepseek-v4-agentx-evolve.jsonnet` at `CLIENT_CONCURRENCIES=16` on v7x-16. The recipe defaults
(TP1xDP16, MNBT128, KV offload) target high concurrency and are unchanged; everything below is passed as job
variables. Measured over the 1 h steady-state window, KV offload off, greedy DSpark drafts.

```
DATA_PARALLEL_SIZE=8 TENSOR_PARALLEL_SIZE=2 MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=256
GPU_MEMORY_UTILIZATION=0.90 TPU_DSV4_ATTN_INTRA_CHIP_TP=0 SPEC_DRAFT_SAMPLE=greedy
KV_OFFLOAD=false ENABLE_KV_OFFLOAD=false OFFLOAD_MODE=off
TPU_STREAMIDX_SPLIT_MIXED=0 TPU_BLOCK_TABLE_STAGE_RING=1 TPU_BLOCK_TABLE_STAGE_SPARSE_ZERO=0
SERVER_EXTRA_ENV=VLLM_TPU_DSV4_MHC_FUSED_GATES_MAX_TOKENS=256,TPU_MOE_ONESHOT_ROUTING_GATHER=1,TPU_MOE_PALLAS_PLAN_MAX_TOKENS=256
```

| config | runs | mean E2E s | TTFT s | ITL p50 / p90 ms | total / output tok/s |
|---|---|---|---|---|---|
| TP4xDP4 MNBT512 | 4 | 14.58 | 1.58 | 12.0 / 26.0 | 67.1k / 499 |
| TP2xDP8 MNBT512 | 3 | 14.05 | 1.48 | 12.1 / 26.5 | 68.9k / 511 |
| above + kernel envs, MNBT512 | 3 | 12.87 | 1.47 | 10.9 / 26.6 | 71.6k / 525 |
| above + kernel envs, MNBT256 (settings above) | 3 | 12.29 | 1.75 | 10.9 / 17.9 | 72.6k / 533 |
| above + kernel envs, MNBT128 | 2 | 12.26 | 2.38 | 11.3 / 13.5 | 72.9k / 538 |

Why these:
- Every DP rank steps in lockstep with the expert-parallel MoE, so a step in which any rank carries a prefill
  chunk costs every rank a prefill-sized step (T256 ~75 ms vs ~30 ms for a decode-only step at TP2xDP8). Smaller
  chunks shorten those stalls; TP2 keeps the tensor-parallel all-reduce inside one chip.
- MNBT 256 or 128 needs the three `TPU_STREAMIDX_SPLIT_MIXED` / `TPU_BLOCK_TABLE_STAGE_*` settings: without them,
  runs died with a SparseCore fault on a step in which the rank only prefills and samples nothing (about one run
  in seven at MNBT256; a work-capped reproducer crashed 2/2 without them and 0/2 with them). No such fault was
  seen in ~44 runs at MNBT512 or above, which needs none of them.
- The three kernel envs (all default off) fold the mHC Sinkhorn gates into the mHC kernels, gather the MoE routing
  payload with one-shot remote DMAs, and build the MoE routing plan in one Pallas call. Each lowered the
  decode-only step: T16 p50 31.5 -> ~28 ms.
- Not helpful at c=16: TP16 or TP8xDP2 (prefill is not sharded by TP), DP16, BLOCK_SIZE 256/512 (needs the `has`
  indexer layout; no prefix-hit gain, slower steps), extra non-power-of-two token buckets, prefill-schedule
  interval with strict cadence, the attention-work cap, cache-aware vllm-router routing, the SparseCore
  all-reduce offload.

mmlu_pro (5-shot, real rejection sampling) for TP2xDP8 with these settings: 0.838 at T=0 (MNBT256) and 0.844
(MNBT512), within the spread of ten earlier runs (0.839 +/- 0.005).
