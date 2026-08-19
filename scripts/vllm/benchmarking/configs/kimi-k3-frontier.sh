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
# ---------------------------------------------------------------- refuted ----
# Do not re-propose without new evidence:
#
#   wider TP                    arithmetically impossible above 32 (96 heads)
#   MoE group padding           GEMM is 2.9% of prefill at 99-100% HBM BW
#   MXFP4 scales -> bf16/uint8  Mosaic strided_load is 32-bit only
#   MXFP4 scale re-blocking     same bandwidth finding; nothing to recover
#   fp4 MoE W4A8                MMLU 0.49 vs 0.86 AND slower, on a sibling model
