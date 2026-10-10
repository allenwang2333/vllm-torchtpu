# Decode MoE communication experiments

## Branch and runtime

`handsomehow/qwen38_decode_optimize` branches from `77eb6a850`.
The comparison runtime uses the previously deployed
`benliangw/pp-pcp-head` snapshot `a010b44ded004dc445aeb94b923557804bfd2b97`.
Only the MoE runner and the new host reduction module are overlaid for the
candidate; the runner before this change is identical in both source trees.
This is not a validation of a full deployment of every file on this branch.

Dependencies: torch_tpu 0.1.1.dev20260928182719, tpu_sync_torch
0.0.1.dev20261007210147, JAX/JAXLIB 0.10.2, libtpu 0.0.47.

## Candidate

Replace the final MoE sum with host-local ReduceScatter, cross-host
AllReduce, then host-local AllGather. Initialization gathers host membership
over the TP CPU group and constructs the op before compilation. The existing
upstream final-output method still controls truncation and already-reduced
outputs. Shared-expert partials already included in that sum remain included.

The tested envelope is a whole-world TP32/EP32 group on four hosts with eight
cores each, PCP1, no sequence parallelism, BF16 tensors with hidden8192 and
token counts 64, 80, 96, 112, 128, 256, 320, 384, 448 or 512. Other shapes
retain the original collective. The tested 1024-token case is slower and
also retains the original collective. The optimization
does not replace attention collectives or change GMM, routing, or weights.

## UT evidence

Fixed global token count, 512 experts, top-10, hidden8192, intermediate2048,
FP4 weights and block256 scales. Each measurement uses 100 dependent forwards,
11 repetitions, with compilation and profiling outside timing.

| Routing | Batch | Original us | Candidate us | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 64 | 309.08 | 259.87 | 15.9% |
| Uniform | 256 | 374.51 | 349.69 | 6.6% |
| Hot | 64 | 329.13 | 281.88 | 14.4% |
| Hot | 256 | 384.85 | 360.01 | 6.5% |

Independent-seed all-routes-to-one-rank stress also improved: 337.37 to
288.48 us at batch64 and 544.77 to 518.83 us at batch256. Numeric gates pass;
normal routing changes reduction order with relative L2 around 0.42%.
The standalone 32-process TorchTPU bridge checks exact rank coverage, shape,
and BF16 errors against CPU FP32 before timing compiled dependent calls.
There was a wall-time outlier; tail latency improvements are not established.

The batch256 trace attributes about 25 us of improvement to communication;
both GMM durations remain unchanged. Alternative EP16/TP2, EP8/TP4 and GMM
tile choices did not show robust gains. SparseCore permute failed continuous
compilation and is not enabled.

## End-to-end comparison contract

92-layer dummy load, TP32/DCP8/EP32, MTP K3, compile buckets64/256,
max-model-len65536, 64 requests with 8192 input and 1024 output tokens,
ignore EOS. Compare two unprofiled runs per variant after warmup, separately
capture a trace, verify completed token counts and actual batch/acceptance.
Raw logs and traces remain on the remote hosts.

Prior unmodified-collective runs measured 966.62 and 963.46 output tokens/s,
with near-zero dummy MTP acceptance. The matched comparison on October 10
measured 965.89/964.66 output tokens/s for the original collective and
972.16/971.51 for the candidate: mean improvement **0.68%**, not the UT-derived
3.6% projection. Both variants completed all 64 x 1024 outputs. Cold rounds
without a full 64-client overlap window were excluded. Prefix caching was
configured and identical prompts prewarmed; the client does not expose exact
cache hit counts.

The candidate trace confirms 64 requests / 256 verification tokens and 92
host AllGather calls. Main-forward median is 58.13 ms. In one complete
forward, AllReduce time falls from 17.58 to 11.48 ms, but host ReduceScatter
waits and AllGather consume about 3.35 and 2.12 ms. The AllGather label uses
an underscore (`all_gather`), so counting only `all-gather` understates
communication. After including it, synchronous communication savings are
about 0.6 ms, consistent with the modest whole-service gain. Async operation
lifetimes overlap and are not additive device execution times.

At 70% accepted/proposed MTP tokens and K=3, the illustrative multiplier is
3.1 tokens/iteration: candidate throughput about 3013 tokens/s and amortized
TPOT about 21.2 ms. These are estimates, not real-weight measurements.

A follow-up concurrency sweep covers 64, 80, 96, 112, 128 and 256 requests.
New B and 4B collective shapes must pass the standalone 32-process bridge
checks before enabling their serving dispatch. Larger-batch results are
pending.


### Extended collective shape sweep

32 TorchTPU processes, exact rank-stripe correctness plus random BF16 versus
CPU FP32, 30 dependent calls per timed sample, 11 repetitions. All shapes
pass; maximum relative L2 is below 0.44% and maximum absolute error below
0.0052. Resetting Dynamo between independent cases avoids its default
per-code-object recompile limit; reset/compilation are outside timing.

| Tokens | Native median us | Host reduction median us |
| ---: | ---: | ---: |
| 80 | 137.51 | 87.19 |
| 96 | 137.63 | 89.13 |
| 112 | 139.60 | 95.16 |
| 128 | 139.81 | 95.13 |
| 320 | 147.80 | 128.03 |
| 384 | 156.81 | 142.90 |
| 448 | 164.96 | 156.96 |
| 512 | 175.32 | 170.37 |
| 1024 | 266.95 | 292.89 |

The dispatcher therefore keeps native AllReduce at 1024 tokens. A batch256
MTP run uses host reduction for its 256-token draft path and native reduction
for its 1024-token target verification path. UT wall-time outliers remain;
these medians do not establish better tail latency.
