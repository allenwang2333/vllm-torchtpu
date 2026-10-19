# DSv4-Pro agentx at low concurrency (c=4, 8, 16): server settings for mean E2E latency

The evolve recipe's defaults (DP16 attention, MNBT 128, MNS 64) are tuned for c=160-192 throughput. At
c<=16 they leave prefill chunked at 128 tokens under DP lockstep and every decode step reading the full
attention weights on each core. The settings below use tensor-parallel attention instead; everything is a
recipe variable, the client side is untouched, and nothing changes the vllm-torchtpu source.

Common to all three (offload off, greedy DSpark drafts, read-only checkpoint PVC):

    KV_OFFLOAD=false ENABLE_KV_OFFLOAD=false OFFLOAD_MODE=off
    SPEC_DRAFT_SAMPLE=greedy
    TPU_DSV4_ATTN_INTRA_CHIP_TP=0 GPU_MEMORY_UTILIZATION=0.90
    WEIGHTS_PVC=gxd-dsv4-pro-weights-rox-pvc

c=4 and c=8 (TP8) also set, in LIBTPU_INIT_ARGS (the recipe default with the SparseCore all-gather
threshold lowered from 262144 to 131072):

    --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=131072
    --xla_all_reduce_latency_bound_threshold_in_bytes=524288
    --xla_tpu_small_all_reduce_binomial_threshold_bytes=524288

These cut mean latency 1.3% (c=4) and 2.6% (c=8) against the same TP8 config without them; at TP4 (c=16)
they measured +1.2% [-0.6, +3.3] and are not used.

| conc | layout | per-conc settings |
|---|---|---|
| 4 | TP8 x DP2 | `DATA_PARALLEL_SIZE=2 TENSOR_PARALLEL_SIZE=8 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 SERVER_EXTRA_ENV=TPU_TOKEN_BUCKET_EXTRA=8:24` |
| 8 | TP8 x DP2 | `DATA_PARALLEL_SIZE=2 TENSOR_PARALLEL_SIZE=8 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048` |
| 16 | TP4 x DP4 | `DATA_PARALLEL_SIZE=4 TENSOR_PARALLEL_SIZE=4 MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=512` |

Mean request latency, 3 server boots per arm (2026-10-19, tree 8c3e37bf5 src):

| conc | recipe defaults | settings above | change, 95% CI |
|---|---|---|---|
| 4 | 14.42 s | 11.14 s | -22.7% [-24.1, -21.4] |
| 8 | 14.28 s | 11.66 s | -18.3% [-20.0, -16.6] |
| 16 | 16.33 s | 14.58 s | -10.7% [-15.2, -6.1] |

Things that matter when changing these:

- `SPEC_DRAFT_SAMPLE=greedy` is required with TP > 1. With probabilistic drafts each TP rank draws its own
  draft noise, the ranks verify different drafts, and sampled (T > 0) output is corrupted: mmlu_pro at
  T=0.6 fell from 0.79 to 0.61 (TP16) and 0.66 (TP8 x DP2). Nothing in throughput or acceptance shows it.
- Host KV offload is off to match the B200 reference; at the c=16 setting turning it back on measured the
  same (14.79 s, n=2). With fewer KV pools (TP8 x DP2 at c=16) offload off cost ~5%.
- Pure TP16 (DP1) is good at c=4 but loses at c=16 (23 s): the MLA KV cache is replicated on every core, so
  there is one ~3.9M-token pool for the whole server and idle sessions lose their whole prefix. More DP
  engines give more pools.
- `GPU_MEMORY_UTILIZATION=0.95` runs out of HBM loading programs mid-run with TP16 and TP8 x DP2.
- `VLLM_USE_RUST_FRONTEND=1` (with `VLLM_RS_BIN`) cuts TTFT by ~0.3 s at DP1, but at DP > 1 it picks DP
  engines itself and bypasses the session-affinity routing hook, so prefix hits collapse.
- `EXPERT_PARALLEL=false` (MoE TP) does not fit: the 192-wide TP16 expert slices are padded to 256 and
  every core holds slices of all 384 experts, leaving too little HBM for one 1M-token request.
