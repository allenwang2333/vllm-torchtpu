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

The follow-up concurrency sweep uses the original native AllReduce runner,
as requested, at 64, 80, 96, 112, 128 and 256 requests. It is separate from
the optimized B64 comparison above; extended collective UT coverage does
not mean the optimized dispatcher was used for these serving measurements.


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

The dispatcher therefore keeps native AllReduce at 1024 tokens. When this
candidate dispatcher is enabled, a batch256 MTP run would use host reduction for its 256-token draft path and native
reduction for its 1024-token target verification path. The native serving
batch sweep does not enable this dispatcher. UT wall-time outliers remain;
these medians do not establish better tail latency.


## Original-code concurrency sweep (October 10)

All measurements below use the original native AllReduce runner from the
`a010b44ded004dc445aeb94b923557804bfd2b97` branch snapshot, not the candidate
collective. Each batch has its own service restart, `max_num_seqs=B`,
`max_num_batched_tokens=4B`, and compile buckets `[B, 4B]`. Other conditions
match the end-to-end comparison contract. An earlier attempt to compile all
buckets together stalled on host file-page I/O before request measurements;
it is not a serving performance sample.

Two unprofiled full-client-overlap windows per batch provide these means.
Each request completes exactly 1024 tokens; dummy acceptance is nearly zero.
The 70% columns assume accepted/proposed=0.7 and unchanged iteration cost,
so use a 3.1 multiplier. They are illustrative real-weight projections,
not measured real-weight throughput or a P95 latency guarantee.

| Batch | Dummy tokens/s | Projected tokens/s at 70% | Projected mean TPOT ms |
| ---: | ---: | ---: | ---: |
| 64 | 965.49 | 2993 | 21.38 |
| 80 | 1061.56 | 3291 | 24.31 |
| 96 | 1153.49 | 3576 | 26.85 |
| 112 | 1202.27 | 3727 | 30.05 |
| 128 | 1298.74 | 4026 | 31.79 |
| 256 | No valid full-batch window | — | — |

B96 is the highest tested valid batch with headroom below the projected
30 ms mean-TPOT goal, gaining 19.47% throughput over B64. B112 is borderline
without headroom; the 0.05 ms difference is not a precise real-weight result.
B128 exceeds the projected goal. These results do not establish a global
optimal batch or a real-weight latency SLA.

Separate profiles confirm actual B requests and 4B verification tokens.
The full iteration includes three draft steps, sampling, and scheduling;
its period is consistent with B / unprofiled throughput, with roughly 2–4%
profiling overhead. Communication is the union of synchronous intervals in
a representative target forward, including underscore-named AllGather.

| Batch | Target median ms | Iteration median ms | Sync communication ms | GMM ms | GDN batched convolution ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 57.53 | 69.01 | 18.17 | 16.83 | 4.69 |
| 80 | 65.33 | 76.40 | 20.87 | 16.73 | 5.82 |
| 96 | 72.37 | 85.51 | 23.87 | 16.82 | 6.91 |
| 112 | 80.96 | 96.33 | 25.96 | 17.06 | 8.02 |
| 128 | 85.78 | 101.41 | 28.98 | 16.70 | 9.28 |

GMM remains near 16.8 ms while communication, GDN, attention, scatter, and
fusion costs grow with batch. Communication remains about one third of the
target forward, supporting further collective work but not a claim that
all communication time is removable.

B256 compiles and serves successfully with buckets 256/1024. After one-token
prewarm, all four rounds complete 256 requests with 1024 outputs each, but
none has a common full-B256 streaming window. The last three rounds reach
only 250 overlapping client streams. In the final round, the latest first
token arrives at 180.316 s while the earliest final token is at 166.278 s.
The validity assertion therefore stops the sweep before B256 profiling.
No B256 steady-state rate or TPOT projection is reported. Client overlap
is not proof of device batch size. Exact cache-hit and admission causes
remain unresolved; configured prefix caching and sufficient reported total
KV capacity do not prove cached admission. Changing output length or sharing
prompts would need a separately labeled experiment.

Raw requests, sweep states, and XPlane traces remain on the remote test
hosts. The deployed runner SHA256 is
`6ff1d55de58073242572c5976aa8597c9cfe940d2cb65d51c0f29a341cdedc6b`.
Versions: torch-tpu `0.1.1.dev20260928182719`, tpu-sync-torch
`0.0.1.dev20261007210147`, JAX/JAXLIB `0.10.2`.
