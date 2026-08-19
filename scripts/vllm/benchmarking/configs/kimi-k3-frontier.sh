#!/bin/bash
# shellcheck disable=SC2034
#
# THE FRONTIER RECIPE. Sourced on top of a base config; carries every lever that
# has been SUPPORTED so far.
#
# The point is to stack rather than to collect isolated results. An experiment
# tests one new lever ON TOP of this file, not against a bare baseline, so gains
# compound and interactions surface early. A lever that helps alone but fights
# something already here is a fact worth having before it reaches a recipe.
#
# Promotion rule: a lever is added only after it is measured beyond the rig's
# noise band (+/-1%, TPOT +/-0.4%, measured across pods) AND passes Tier 0
# identity, or -- if it legitimately changes numerics -- Tier 1 accuracy.
# Every entry cites the experiment that earned it.
#
# Demotion is expected. If a later stack shows a lever no longer pays, it comes
# out and the reason is recorded here.

FRONTIER_ARGS=""

# Env-var levers, applied the same way. Most of what has moved this model is
# selected by environment variable rather than by a serve flag -- the MoE
# token-movement kernels, the GMM tiling override, the token-bucket floor -- so
# a frontier that could only carry serve args could not carry the frontier.
# Format: space-separated KEY=VALUE, exported by bench_k3.sh before the server
# starts. Same promotion rule and same citation requirement as FRONTIER_ARGS.
FRONTIER_ENV=""

# ---------------------------------------------------------------- pending ----
# Under test, NOT yet promoted:
#
#   --async-scheduling          v019. NULL at concurrency 1 on the full model
#                               (30.96 vs 30.94 tok/s, 27.80 vs 27.82 ms TPOT,
#                               inside a +/-1% band). Expected: the mechanism
#                               books step-N tokens as placeholders so the host
#                               can run ahead, which buys nothing with an empty
#                               queue. c4 decides. Proxy measured -13.1% TPOT.
#
#   --kv-cache-dtype fp8        v018c arm C. Halves MLA KV bytes. Changes
#                               numerics, so Tier 1 is required, not Tier 0.
#
#   --gpu-memory-utilization    v018c arm D. K3 currently leaves ~25% of HBM
#     0.95                      unclaimed; the sibling K2.6 config in
#                               tpu-inference uses 0.95.
#
#   --max-num-batched-tokens    v020b. 512->2048 collapsed P99 ITL 6.35x on the
#     {1024,2048}               proxy but did not move throughput and cost 7.7%
#                               TTFT. The 1024 point is the one worth having.
#                               compile_sizes MUST be extended in step.
#
#   fused o_norm epilogue       v021. Parity verified at max|diff|=0.0 on v6e.
#                               Timing pending on v7x.
#
#   TPU_MOE_HIERARCHICAL_EP=1   v026 arm B. -2.4% TPOT at c1 and -4.1% at c4 on
#                               the 12L rig, throughput +3.1%/+3.3%. Its own
#                               docstring names this operating point: low
#                               concurrency, weight-bandwidth-bound MoE,
#                               imbalanced per-device expert load. Also reduces
#                               cross-chip token movement, which is the right
#                               shape of fix for a latency-bound collective
#                               (F15). Awaiting full-model confirmation before
#                               promotion -- the 12L rig has mispredicted a
#                               full-model result once already (F2a).
#
#   compile_sizes floor < 16    v030. Every decode step pads 1-4 real tokens to
#                               16 because _get_token_paddings takes
#                               min_token_size=16. No floor is enforced when
#                               compile_sizes is passed explicitly.
#
# ---------------------------------------------------------------- refuted ----
# Do not re-propose without new evidence:
#
#   wider TP                    arithmetically impossible above 32 (96 heads)
#   MoE group padding           GEMM is 2.9% of prefill at 99-100% HBM BW
#   MXFP4 scales -> bf16/uint8  Mosaic strided_load is 32-bit only
#   MXFP4 scale re-blocking     same bandwidth finding; nothing to recover
#   fp4 MoE W4A8                MMLU 0.49 vs 0.86 AND slower, on a sibling model
