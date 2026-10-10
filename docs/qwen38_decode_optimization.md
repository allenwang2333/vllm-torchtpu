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
cores each, PCP1, no sequence parallelism, BF16 tensors shaped `[64,8192]`
or `[256,8192]`. Other shapes retain the original collective. The optimization
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
with near-zero dummy MTP acceptance. UT gains are not end-to-end gains;
the matched service comparison is pending.
