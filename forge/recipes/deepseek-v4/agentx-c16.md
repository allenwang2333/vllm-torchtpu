# DSv4-Pro agentx at concurrency 16: serving settings

Settings for `infx-deepseek-v4-agentx-evolve.jsonnet` at `CLIENT_CONCURRENCIES=16` on v7x-16. The recipe defaults
(TP1xDP16, MNBT128, KV offload) target high concurrency and are unchanged; everything below is passed as job
variables. Measured over the 1 h steady-state window, KV offload off, greedy DSpark drafts.

```
DATA_PARALLEL_SIZE=8 TENSOR_PARALLEL_SIZE=2 MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=512
GPU_MEMORY_UTILIZATION=0.90 TPU_DSV4_ATTN_INTRA_CHIP_TP=0 SPEC_DRAFT_SAMPLE=greedy
KV_OFFLOAD=false ENABLE_KV_OFFLOAD=false OFFLOAD_MODE=off
VLLM_TPU_DSV4_INDEXER_KV_LAYOUT=has   # or keep sal with the SAL writer fix (52f93a927)
SERVER_EXTRA_ENV=VLLM_TPU_DSV4_MHC_FUSED_GATES_MAX_TOKENS=256,TPU_MOE_ONESHOT_ROUTING_GATHER=1,TPU_MOE_PALLAS_PLAN_MAX_TOKENS=256,TPU_DSV4_CSA_TOPK_TIE_Q_MAX_TOKENS=32,VLLM_TPU_SEGMENT_TOKENIZE=1
```

All on the has indexer layout unless noted (sal without the writer fix leaves unwritten indexer-key rows that score
NaN and win the top-k; a long-context retrieval check scored 1/20 on unfixed sal vs 20/20 on has).

| config | runs | mean E2E s | TTFT s | ITL p50 / p90 ms | total / output tok/s |
|---|---|---|---|---|---|
| recipe default TP1xDP16 MNBT128 | 1 | 16.34 | 2.92 | 14.6 / 16.6 | 64.0k / 481 |
| TP4xDP4 MNBT512 | 3 | 14.58 | 1.58 | 12.0 / 28.0 | 66.9k / 499 |
| settings above (MNBT512) | 3 | 12.49 | 1.27 | 10.9 / 24.6 | 71.9k / 529 |
| same, sal + writer fix | 2 | 12.36 | 1.26 | 10.6 / 26.2 | 72.3k / 531 |
| same at MNBT256 (faster, see the fault below) | 3 | 11.83 | 1.49 | 10.8 / 17.6 | 73.7k / 540 |

Why these:
- Every DP rank steps in lockstep with the expert-parallel MoE, so a step in which any rank carries a prefill
  chunk costs every rank a prefill-sized step (T256 ~75 ms vs ~30 ms for a decode-only step at TP2xDP8). Smaller
  chunks shorten those stalls; TP2 keeps the tensor-parallel all-reduce inside one chip.
- MNBT 256/128 are faster but not deployable yet: about one run in eight dies with a SparseCore fault on a step
  in which a rank only prefills and samples nothing. No flag tried (split-mixed off, block-table ring 1, blocking
  host-to-device copies, SparseCore all-gather offload off, has layout) prevents it on a work-capped reproducer.
  No such fault was seen in ~44 runs at MNBT512 or above.
- The kernel envs (all default off) fold the mHC Sinkhorn gates into the mHC kernels, gather the MoE routing
  payload with one-shot remote DMAs, and build the MoE routing plan in one Pallas call. Each lowered the
  decode-only step: T16 p50 31.5 -> ~27.6 ms. TIE_Q lets the CSA SparseCore gather overlap the sliding-window
  pass; segment tokenization re-tokenizes only the new tail of a continued conversation (token ids verified
  identical), cutting TTFT by ~0.2 s.
- Not helpful at c=16: TP16 or TP8xDP2 (prefill is not sharded by TP), DP16, BLOCK_SIZE 256/512 (needs the `has`
  indexer layout; no prefix-hit gain, slower steps), extra non-power-of-two token buckets, prefill-schedule
  interval with strict cadence, the attention-work cap, cache-aware vllm-router routing, the SparseCore
  all-reduce offload.

mmlu_pro (5-shot, real rejection sampling) with these settings: 0.834 at T=0 on has, 0.834/0.838 on sal, within
the spread of ten earlier runs (0.839 +/- 0.005). Prompts there are under 10k tokens, so mmlu_pro cannot see the
sal indexer-key corruption; the long-context retrieval check above is the evidence for has / the writer fix.
