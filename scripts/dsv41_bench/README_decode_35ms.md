<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSv4.1 decode measurements, 2026-10-21 UTC

The 35 ms main-model forward and 14000 output tokens/chip/s targets remain
unmet. The serving measurements below cover an FP4 MoE height ladder for the
observed EP8 geometry and a joint request/scorer configuration for the
48-request full-context indexer, whole-batch router selection, WO_A row chunks
and the normalized mHC output placement.

## Configuration and provenance

The branch starts at `dsv41` revision `bdb38700dd25959c029c7d77f00cee91ccd3a0bc`.
The user-supplied reference is Falcon `exp-v35cbnk6j2`, source
`f12b6d7ac6430194ccf366033ee43c6ab208c641`. The base already contains other
optimizations; differences between that reference and this base are not
attributed to these changes.

All new serving runs use DSv4.1 Flash, 384 concurrent requests, a 98304-token
prefix, 512 output tokens, DSpark K5, TP1/DP8/EP8 and eight logical TPU7x
devices on four physical chips. Each run uses the same frozen input,
three unprofiled rounds and a separate six-step profile. Persistent compilation
caches are disabled. The runtime is torch 2.13.0+cpu, torch-tpu 0.2.0rc1,
JAX/JAXLIB 0.11.2 and libtpu 0.0.50, with vLLM revision
`e37e51dd246cf421c554e7d6f53e178e3fd29085`.

| Version | Source | Falcon experiment / result directory |
| --- | --- | --- |
| Base | `bdb38700` | `exp-qcbagjr3z8` / `results/baseline` |
| FP4 ladder | `c6ffd0fa` | `exp-qcbagjr3z8` / `results/rungs2-tuned` |
| FP4 ladder + indexer | `cd47b6c9` | `exp-1mv4ll6amh` / `results/indexer-tuned` |
| FP4 ladder + indexer + router | `7b971e97` | `exp-qg29pd6unq` / `results/router-tuned` |
| Previous + WO_A | `2881d570` | `exp-btjh485722` / `results/woa-tuned` |
| Previous + mHC VMEM output | `7bf96de9` | `exp-qfs10zulj7` / `results/mhc-vmem-tuned` |

The FP4 run's top-level runtime source and actual checkout are `c6ffd0fa`;
its copied nested source metadata still names the base. The first combined
trial in `exp-qcbagjr3z8` was interrupted by the experiment's two-hour limit
before formal results. Only the completed rerun above is reported.

## Serving results

Each cell with three values lists rounds 1, 2 and 3. Fixed-acceptance
throughput uses `384 * 3.47 / step_seconds / 4`, matching the reference
harness. Observed throughput uses accepted output tokens including bonus
tokens divided by the measured steady window; it has a different numerator.

| Metric | Base | FP4 ladder | FP4 ladder + indexer | FP4 ladder + indexer + router |
| --- | ---: | ---: | ---: | ---: |
| Host step median, ms | 55.011 / 55.019 / 54.959 | 54.090 / 54.278 / 54.170 | 53.086 / 52.586 / 52.530 | 53.495 / 53.188 / 53.376 |
| Fixed-acceptance output tokens/chip/s | 6056 / 6055 / 6061 | 6159 / 6137 / 6149 | 6275 / 6335 / 6341 | 6227 / 6263 / 6241 |
| Observed accepted+bonus tokens/chip/s | 6773 / 6840 / 6944 | 6945 / 6752 / 6891 | 7120 / 7048 / 7207 | 7021 / 7032 / 6819 |
| Main-model device forward, ms | 47.038 | 45.184 | 42.918 | 42.599 |
| Forward minus identified entry barriers, ms | 42.722 | 42.438 | 42.264 | 41.847 |
| Main-model FP4 MoE event time/step, ms | 15.442 | 15.163 | 15.119 | 15.127 |
| Full-context index score event time/step, ms | 2.836 | 2.835 | 2.670 | 2.669 |
| Startup HBM used per logical device, GiB range | 48.98–49.04 | 48.99–49.04 | 48.99–49.05 | 49.00–49.06 |
| Server launch to health ready, seconds | 1049.86 | 1059.46 | 1006.77 | 1015.20 |

Forward is the median of six per-step sums of the four sequential rank-0
main-model device modules, with 40 MoE layers verified in each step.
It excludes draft, logits/sampling and host gaps. It is not a sum of module
medians. Subtracting the identified `barrier-cores` intervals gives a separate
diagnostic metric, not a replacement for the actual forward measurement.
Most of the 47.04-to-42.60 ms change is entry-barrier variation; about
0.88 ms remains after this subtraction. Router selection's 43 inclusive events
(40 main and three draft) fall from 0.483 to 0.096 ms per step. The router
run's host step medians and throughput are worse than the preceding run,
despite the device gain; these runs do not establish a serving-throughput
improvement from the router change. One of its six profile steps contains
13.71 ms of entry barriers, retained in the reported samples and medians.
Operator durations are inclusive and
may overlap other work, so they must not be summed as a critical-path estimate.

HBM values are rounded startup server-log statistics, not sampled runtime
peaks. The joint indexer increases its isolated executable's temporary HBM by
12 MiB (42390016 to 54658304 bytes). These changes do not repack model weights.
Startup includes loading, compilation and warmup; it is not a weight-load
benchmark. Prefill, mixed batches and model-quality benchmarks were not run
for these changes.

At the reference fixed acceptance of 3.47, a 35 ms complete step would yield
9518 output tokens/chip/s; 14000 requires at most 23.794 ms. Counting all six
verification tokens, including rejected tokens, would be a different metric.
The target cannot be claimed by exchanging these definitions.

## Kernel validation and reproduction

The FP4 key includes TPU generation, hidden/intermediate dimensions, quantization
block, capacity, local expert count, EP size, allocated rows, activation, active
intermediate width, biases and scale padding. Only the measured key selects two
heights; misses keep four. Randomized paired EP8 runs at 288 local rows,
hidden 5120, active intermediate 2304 padded to 2560, 384 experts and top-k six
were bit-exact, with device kernel time approximately 0.308 to 0.292 ms.

```bash
python scripts/dsv41_bench/bench_fp4_moe_heights.py \
  --out /tmp/fp4-heights --seed 1020 --profile
```

Repeat with seed 1021. The harness checks outputs, retains individual timings,
dumps compiled HLO and optionally profiles the warmed executables.

The indexer was checked with seeds 1021–1024, lengths 49408 and 65536,
contiguous and shuffled pages, and a 17-request tail/ties case. Selected IDs
were exact. Complete score/top-k module time fell approximately
0.567 to 0.520 ms on contiguous pages and 0.741 to 0.680 ms on shuffled pages.
The larger request tile regressed the 17-request case, so the new configuration
is scoped to exactly 48 requests. The served full-context scorer count changes
from 15 to 12 per step; nine calls use the new 64K-capacity configuration and
three retain the 128K-capacity configuration.

Save these benchmark variants to `/tmp/indexer-variants.json`:

```json
{
  "reference": {
    "request_tile": 12,
    "score_config": {
      "key_tile": 99328,
      "compute_tile": 1024,
      "head_group": 32
    }
  },
  "automatic": {}
}
```

```bash
python scripts/dsv41_bench/bench_decode_overlap.py \
  --variants /tmp/indexer-variants.json --out /tmp/indexer-tuning \
  --seed 1024 --shuffled --profile --dump-hlo
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /path/to/frozen-inputs.json --out /path/to/results \
  --rounds 3 --startup-timeout 7200 --profile
```

Use the runtime and checkpoint setup in `server_steps/README.md`. A smaller
benchmark does not establish whole-model throughput. Relevant CPU and TPU
regressions and pre-commit checks passed. Full-environment type checking
reported the same 57 pre-existing errors as the base, with no new errors;
it was not a clean full-environment pass.

## Whole-batch expert selection

At 288 local rows, the previous power-of-two divisor selected 32-row blocks,
executing nine grid iterations. The measured TPU7x entry processes 288 rows
in one block for 384 experts and top-k six. Other shapes and devices retain
the previous divisor. Both the one-table and separate-weight-table paths have
entries; hash-based routing still bypasses selection.

In Falcon `exp-83rwu9buu9`, EP8 whole-layer paired runs were bit-exact.
With softmax gating, selector time fell from 11.095 to 1.329 microseconds.
With `sqrtsoftplus`, correction bias, routed scale 1.5 and biased expert
logits, it fell from 11.696 to 1.743 microseconds. The latter whole-layer
device time excluding entry barriers fell from 0.52108 to 0.51082 ms.
These are synthetic whole-layer measurements, not an additional serving gain.
The regression test checks the whole-batch tile against the previous tile and
NumPy, including ties, non-finite rows and independent output weights.
The final automatic-dispatch run at seed 1022 was bit-exact and used the same
profiled executable as the explicit 288-row variant. Eight selector regression
cases passed on TPU; seven layer integration cases passed on CPU. The full
type-check error set remained identical to the base.

```bash
python scripts/dsv41_bench/bench_fp4_moe_heights.py \
  --out /tmp/router-tiles --seed 1022 --tune-router \
  --scoring-fn sqrtsoftplus --score-bias --gate-bias 0.9 \
  --routed-scaling-factor 1.5 --profile
```

This mode compares 32, 96, 144 and 288 rows with automatic table dispatch,
retaining compiled HLO and checking complete EP8 outputs against the 32-row
reference. Height tuning remains automatic throughout the router sweep.

## WO_A row chunks and serving validation

The TPU7x decode shape `[288,64,512]` with FP8 weight `[4096,8192]`,
FP32 scales, tile T=288/R=1024, quantized activations and no fused RoPE now
uses one 288-row contraction instead of three 96-row contractions. The
lookup includes device, shapes, dtypes and tile settings; other workloads
keep the previous divisor rule, and explicit `sub_t` values take precedence.

| Seed / experiment | 96-row reference (us) | 288-row candidate (us) |
| --- | ---: | ---: |
| 1023 / `exp-qg29pd6unq` | 25.439 | 23.842 |
| 1033 / `exp-1yxqyt6ap6` | 25.452 | 23.857 |
| 1035 / `exp-1yxqyt6ap6`, automatic dispatch | 25.458 | 23.813 |

These are device-module medians excluding entry barriers. Seeds 1033 and
1035 passed comparisons of all output bit patterns, including signed zero.
Seed 1023 used array equality. In the automatic-dispatch run, the explicit
288-row variant measured 23.825 us. Compiled generated code shrank from
307200 to 255488 bytes, with unchanged reported argument/output storage and
zero temporary HBM. The roughly 1.6 us gain per layer projects to only
0.064 ms over 40 layers and is not an end-to-end result.

Serving experiment `exp-btjh485722` completed three formal rounds and the
six-step profile at source `2881d570`. Its result is **not a serving speedup**:

| Metric | Router run `7b971e97` | WO_A run `2881d570` |
| --- | ---: | ---: |
| Host step median, ms | 53.495 / 53.188 / 53.376 | 55.604 / 55.747 / 55.667 |
| Fixed-acceptance output tokens/chip/s | 6227 / 6263 / 6241 | 5991 / 5976 / 5984 |
| Observed accepted+bonus tokens/chip/s | 7021 / 7032 / 6819 | 6876 / 6726 / 6641 |
| Main-model device forward, ms | 42.599 | 46.039 |
| Forward minus identified entry barriers, ms | 41.847 | 41.912 |
| Main-model FP4 MoE event time/step, ms | 15.127 | 15.160 |
| Full-context index score event time/step, ms | 2.669 | 2.672 |
| Startup HBM used per logical device, GiB range | 49.00–49.06 | 49.02–49.04 |
| Server launch to health ready, seconds | 1015.20 | 1018.22 |

The WO_A run's entry-barrier median is 4.076 ms; its six individual forward
samples are 52.396, 45.878, 45.796, 66.516, 45.982 and 46.097 ms. All six
contain the expected 40 main-model MoE layers. Removing entry barriers gives
41.912 ms versus 41.847 ms previously, so the raw forward regression mostly
coincides with increased barriers. These are separate jobs, not a same-node
A/B comparison; this does not isolate the cause of the host regression.
Inclusive WO_A-named event time decreased by about 0.060 ms per step, consistent
with its paired microbenchmarks, while main MoE, indexer and mHC event times
were similar. The exact-shape tuning remains enabled for this isolated gain
and smaller generated code, with **no demonstrated whole-model throughput gain**.
The 35 ms and 14000 output tokens/chip/s targets remain unmet.

Reproduce on the pinned TPU runtime from the repository root:

```bash
python scripts/dsv41_bench/bench_wo_a_tiles.py --out /tmp/woa-1033 --seed 1033 --profile
python scripts/dsv41_bench/bench_wo_a_tiles.py --out /tmp/woa-1035 --seed 1035 --profile
```

## mHC output placement and consumer fusion

Two additional prototypes in `exp-1yxqyt6ap6` preserve the five mHC outputs
and the FP8 activation/scale values used by a following 5120-by-1792 Q/KV
projection. All eight output checks, including the projection, were bit-exact
at seeds 1034 and 1036. Timing returns the five mHC outputs and projection;
quantized intermediates remain internal.

| Prototype | Seed 1034 reference / candidate (us) | Seed 1036 reference / candidate (us) |
| --- | ---: | ---: |
| Quantize normalized values inside the mHC kernel | 27.796 / 23.396 | 27.824 / 23.362 |
| Keep the normalized BF16 value in VMEM for its consumer | 27.747 / 26.634 | 27.646 / 26.607 |

These are complete microbenchmark device-module medians excluding entry
barriers. Explicit prequantization remains a prototype because it requires
passing additional values to the consumers. The VMEM-output implementation
retains the existing interface and has the serving validation below.

The mHC VMEM-output implementation retains the normalized BF16 activation in VMEM
for following consumers, without changing their interface or arithmetic.
At seeds 1034 and 1036, all mHC outputs, FP8 activation/scale and projected
values were bit-exact. Device medians for mHC plus the following 5120-by-1792
projection fell from 27.747 to 26.634 us and from 27.646 to 26.607 us.
The dispatch is limited to the kernel's TPU7x envelope with 288 tokens,
four streams, width 5120, BF16 norm weights and HBM gate inputs.
Other shapes retain the previous output placement. The isolated timings are
reported separately from the serving measurements below.

Source `7bf96de9` was rechecked in `exp-qfs10zulj7` with seeds 1045 and 1046;
the integrated candidate and baseline remained bit-exact across all eight
checked outputs. A 128-token fallback case at seed 1047 was also bit-exact.
For 288 tokens, compiled code decreased from 1264128 to 1251328 bytes;
arguments (29219840 bytes), outputs (15814688 bytes) and temporary storage
(zero) were unchanged. The fallback's compiled memory sizes were unchanged.
These pre-serving checks did not explicitly pin a DVFS state, so their timings
are not compared with the earlier fixed-state measurements. The formal serving
launcher retains its fixed P-state 7 configuration.

Pre-commit passed. After installing the missing `uv` tool in the evaluation
container, full-environment pyrefly reported the same 57 errors as a source copy
with the baseline mHC files restored: zero added or removed diagnostics.
The numerical/HLO/device-event archive is `results/mhc-vmem-tuned/micro-evidence.tgz`
in that experiment (SHA-256
`a59f7e3de9b4686927e58407cae04f34bdab913c030f9fae057c099024d0c78b`).
Its saved typecheck log is the initial missing-tool failure; the subsequent
paired check is stored separately as `typecheck-base.json` and
`typecheck-candidate.json` in the same result directory.

```bash
git show a04817ac:src/vllm_torchtpu/kernels/deepseek_v41/mhc/shifted_post_pre_rms.py \
  > /tmp/mhc-output-reference.py
python scripts/dsv41_bench/bench_mhc_output.py \
  --reference-source /tmp/mhc-output-reference.py \
  --out /tmp/mhc-output --seed 1045 --profile
```

Serving and profile checks in `exp-qfs10zulj7` completed successfully. All six
profile steps contain the expected four main-model modules with 1, 13, 6 and
20 MoE layers, respectively.

| Metric | WO_A run `2881d570` | mHC VMEM output `7bf96de9` |
| --- | ---: | ---: |
| Host step median, ms | 55.604 / 55.747 / 55.667 | 52.724 / 52.465 / 52.254 |
| Fixed-acceptance output tokens/chip/s | 5991 / 5976 / 5984 | 6318 / 6349 / 6375 |
| Observed accepted+bonus tokens/chip/s | 6876 / 6726 / 6641 | 7118 / 7200 / 7146 |
| Main-model device forward, ms | 46.039 | 42.608 |
| Forward minus identified entry barriers, ms | 41.912 | 41.728 |
| Main-model mHC event time/step, ms | 1.370 | 1.292 |
| Main-model FP4 MoE event time/step, ms | 15.160 | 15.152 |
| Main-model CSA event time/step, ms | 5.195 | 5.117 |
| Startup HBM used per logical device, GiB range | 49.02–49.04 | 48.98–49.04 |
| Server launch to health ready, seconds | 1018.22 | 1042.22 |

The new forward samples are 46.199, 42.728, 42.489, 60.111, 42.304 and
42.320 ms; the entry-barrier median is 1.114 ms. Most of the raw forward
reduction coincides with barrier variation. The per-step diagnostic excluding
those intervals improves by 0.184 ms. Inclusive mHC event time decreases by
0.078 ms, consistent with the paired isolated measurements; inclusive event
times may overlap and must not be added as critical-path savings. These
separate jobs do not isolate the cause of the larger host-step difference.
The change is retained for its exact outputs, smaller generated code and
measured mHC gain. Both requested performance targets remain unmet.

The verified serving analysis archive is
`results/mhc-vmem-tuned/serving-evidence.tgz` in that experiment (SHA-256
`42438353c39424c06db0fbf3ff87121ecc9328ad8b4adaccec79e4f3ca761798`).

## Experiments not adopted

The following prototypes remain outside the production path. Unless labeled
otherwise, times are complete device-module medians after subtracting identified
entry barriers; operator-only measurements are explicitly identified.

| Prototype | Measurement | Decision |
| --- | --- | --- |
| Pack activation and routing metadata into one all-gather | Whole EP8 layer 0.469 to 0.918 ms, bit-exact | Reject |
| Compact FP4 weights with three slots and capacity 64 | Biased-gate MoE kernel 0.340 to 0.407 ms, or 0.367 to 0.448 ms, bit-exact | Reject |
| Smaller CSA interleave groups | Synchronized host latency 0.332 to 0.339–0.380 ms, bit-exact | Retain six-query interleave |
| Lossless FP8 candidate-key storage | Cache 576 to 297 MiB; packing plus four reindexes 1.482 to 3.584 ms after tile tuning | Reject despite smaller cache |
| Direct FP8 candidate QK with scales moved into the head reduction | Complete FP32 score bit-equality check failed | Do not promote or claim a speedup |
| Skip invalid candidate-tail DMA and compute | Dense synthetic unions of 3000–5400 blocks: 0.433 to 0.436 ms; full unions: 0.435 to 0.467 ms, bit-exact | Reject |
| One 49152-key DMA window with a larger VMEM budget | Full-union scorer/selection 0.436 to 0.450 ms, bit-exact | Retain 24576-key DMA windows |
| MoE routing rank windows of 32, 16 or 8 pairs | Whole EP8 layer 0.5125 to 0.5129–0.5139 ms, bit-exact | Retain existing rule |
| SparseCore gather writeback overlap and 1024 rows/subcore | Isolated gather 0.705 to 0.684 ms, bit-exact | Small synthetic gain; no serving validation or production change |
| mHC 32-token tiles or compiler-managed gate windows | With RMSNorm: 18.845 to 23.241 us (32-token/HBM), or 21.184 us (96-token/compiler), bit-exact | Retain 96-token/HBM configuration |
| mHC 288-token tile with the original 96-row projection chunks | With RMSNorm: 18.695 to 20.777 us, all five outputs bit-exact | Reject |
| Compact MoE gate/up staging with three gate/up slots and two down slots | Biased whole EP8 layer: 0.50020 to 0.50150 ms; uniform: 0.45972 to 0.45887 ms, bit-exact | No meaningful gain |
| Packed-word FP4-to-FP8 expansion | Biased whole EP8 layer: 0.50085 to 4.18015 ms; uniform: 0.45934 to 4.07370 ms, bit-exact | Reject |
| Compact gate/up weights in HBM, two staging slots | Biased whole EP8 layer: 0.53772 to 0.53714 ms; uniform: 0.45938 to 0.45773 ms, bit-exact | No meaningful performance gain; retain existing weight layout |

The compact-HBM experiment ran after serving in `exp-btjh485722`, with
biased `sqrtsoftplus` routing at seed 1037 and uniform softmax at seed 1038.
Compaction occurs outside the timed forward and removes the two unused
256-column gate/up tails. All output bit patterns, including signed zero,
match the original layout. Compiled argument storage decreases by exactly
62914560 bytes (60 MiB) per logical device and layer; temporary storage and
generated code sizes stay unchanged. Operator-only MoE medians are
0.369390 to 0.369348 ms (biased) and 0.292022 to 0.289723 ms (uniform).
The small time change does not justify changing the production weight loader.

A further compact-HBM/three-slot prototype initially halted before correctness
checks or profiling completed. Static inspection found a missing modulo-three
mapping in its DMA slot index. After that fix, `exp-bazj9i3rq6` completed both
seeds with bit-exact outputs. Whole-layer device time excluding entry barriers
was 0.53768 to 0.54127 ms for biased routing and 0.45957 to 0.45732 ms for uniform
routing. The third prefetch slot did not establish a useful gain. This was a
prototype indexing error; the failed execution is not evidence of a hardware
fault and supplies no performance result.

The same experiment tested consuming each 512-column gate/up pair through the
activation before computing the next pair, retaining the original 256-column
dots, K-block addition order and BF16 row-quantization reductions. All output
bits matched at biased seed 1039 and uniform seed 1040, but whole-layer device
time excluding entry barriers was 0.48145 to 0.48312 ms and 0.45931 to 0.45903 ms,
respectively. Compiled argument, output and temporary storage were unchanged;
generated code grew by 5120 bytes. Neither this fusion nor either compact-HBM
prototype is in production dispatch.

An exploratory G1024 requantization in `exp-bazj9i3rq6` reduced the number of
per-block scale operations but required padding the intermediate width from
2560 to 3072. The original VMEM accounting then selected one weight slot;
the profile's kernel name confirmed `nb1`. Compacting gate/up weights and
accounting for their actual staged width restored two slots. The conservative
VMEM estimate was 65748992 bytes against a 65766686-byte budget.

| Routing / seed | G512 reference (ms) | Padded G1024 (ms) | Compact G1024 (ms) |
| --- | ---: | ---: | ---: |
| Biased / 1041 | 0.55982 | 0.66395 | 0.54881 |
| Uniform / 1042 | 0.45970 | 0.62287 | 0.45388 |

These are complete EP8 device-module medians excluding entry barriers, with
the same two-height ladder. Quantization and compaction occur outside timing.
Compact and padded G1024 outputs match bit-for-bit, but both differ from G512:
relative L2 errors are 0.11498 and 0.11503; maximum absolute errors are 0.10229
and 0.10547, respectively. These inputs requantize synthetic G512 weights and
are **not a model-quality evaluation**. Compact G1024's compiled arguments use
964411392 bytes versus 970702848 for G512; temporary/output storage is unchanged.
The 1–2% isolated gain does not justify promoting the approximate quantization
and changed weight layout. G1024 remains outside production dispatch; no claim
of quality neutrality or serving improvement is made.

The corrected three-slot, gate/up activation and G1024 experiments have a
combined source/HLO/result archive at
`exp-bazj9i3rq6` / `results/final-micro-evidence.tgz`.

In `exp-c5bdngz3ad` (source `a04817ac`), explicit one- and two-block
FP4-to-FP8 lookahead retained the original K-block dot shapes, scales and
addition order. All complete EP8 output bits matched. At biased seed 1043,
whole-layer device medians excluding entry barriers were 0.571214 ms for the
reference, 0.571439 ms for one-block lookahead and 0.571498 ms for two blocks.
At uniform seed 1044 they were 0.459195, 0.459625 and 0.459485 ms. Generated
code sizes and compiled argument/output/temporary storage were unchanged.
Moving those reads earlier did not establish a speedup and is not adopted.

After serving completed in `exp-qfs10zulj7`, a two-rung MoE sweep retained
capacity 128 and changed only the lower height. Every EP8 output bit matched
at biased seed 1048 and uniform seed 1049, including signed zero.

| Routing / seed | 64/128 reference (ms) | 48/128 (ms) | 80/128 (ms) | 96/128 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Biased / 1048 | 0.561563 | 0.564828 | 0.573694 | 0.580213 |
| Uniform / 1049 | 0.459417 | 0.465987 | 0.472470 | 0.483886 |

These are complete device-module medians excluding entry barriers, with
P-state 7 and five ordered profile repetitions. Argument/output/temporary
storage was unchanged. None of the alternative lower heights improved
performance, so the 64/128 dispatch remains. Source, HLO and results are at
`results/nonuniform-heights/evidence.tgz` (SHA-256
`ac1e646c094a42334aa7d1c13177a1e4018497d3b2612889f038334598d8701c`).

MoE combine experiments in `exp-rtlysk81mx` used source `ea0558b8`,
the same pinned runtime, EP8 with 288 local rows, P-state 7, 31 rotating
host repetitions and five ordered device repetitions. All complete EP8
output bits matched, including signed zero. Each biased case used
`sqrtsoftplus`, correction bias, gate bias 0.9 and routed scale 1.5;
uniform cases used softmax. Each row below has its own paired reference.

| Change | Biased seed: reference → candidate (ms) | Uniform seed: reference → candidate (ms) |
| --- | --- | --- |
| Fused scale lookup, HBM output | 1050: 0.504663 → 0.505881 | 1051: 0.459752 → 0.459507 |
| Fused scale lookup, VMEM output | 1050: 0.504663 → 0.506629 | 1051: 0.459752 → 0.461006 |
| External coefficients, VMEM output | 1050: 0.504663 → 0.512965 | 1051: 0.459752 → 0.467408 |
| Resident arrivals, fused scale lookup | 1052: 0.591569 → 0.592461 | 1053: 0.459744 → 0.459872 |
| Resident arrivals, external coefficients | 1054: 0.558883 → 0.557544 | 1055: 0.460206 → 0.458994 |

These are whole-device-module medians excluding entry barriers. The first
three variants used the existing three-buffer row-DMA combine. Resident
variants instead staged the entire 2112-by-40-by-128 FP8 arrival buffer in
VMEM, then indexed rows there. The scale-fused variants read the scale mirror
through scalar memory; external-coefficient variants retained the XLA scale
gather. All retained the original accumulation order and masked-row handling.

The post-expert tail, measured from the expert kernel's end to the enclosing
module's end, did not improve. Fused-scale row DMA increased it from about
26.3 us to 28.0–29.3 us; external coefficients with row DMA took 34.4–34.7 us.
Resident arrivals with fused scales took about 28.0 us. With external
coefficients, their paired tails were 26.836 → 26.939 us at seed 1054 and
26.262 → 26.443 us at seed 1055. Thus the small whole-module reductions in
the final table row do not establish a faster combine.

A further resident-arrival/external-coefficient sweep used reduction groups
16, 32, 48 and 96 at biased seed 1056 and uniform seed 1057. All were bit-exact.
Their whole-module medians were 0.505005–0.505284 ms against 0.505263 ms and
0.460385–0.460719 ms against 0.460986 ms, respectively. Their tails remained
26.738–26.870 us against 26.541 us and 26.328–26.558 us against 26.266 us.
No combine replacement is enabled.

Compiled argument/output storage stayed at 970702848/2949120 bytes.
For biased seed 1052, resident arrivals with fused scales reduced generated
code from 4645888 to 4333568 bytes and temporary storage from 99334336 to
98886112 bytes. External coefficients at seed 1054 used 4403200 bytes of code
and 99293408 bytes of temporary storage. These smaller compiled artifacts
did not produce a measured tail improvement; they are not server HBM peaks.

Source, HLO, numerical checks and device events are preserved under
`exp-rtlysk81mx` / `art-l6lqmlmybw`:

| Artifact path | Verified SHA-256 |
| --- | --- |
| `results/combine-scale-evidence.tgz` | `b73b33bc53c21e19cb4aee4a9f5902354b04b74860e55de802c426c4490e8a35` |
| `results/combine-resident-evidence.tgz` | `d223531c6d9d522c09e6f12884f3ccff550318db9f99eedf55b217c9573ecbf7` |
| `results/combine-resident-coef-evidence.tgz` | `1d52001cad5fedd1800afd61ae95fcb97f1c956e820374e83602c598d8ea422f` |
| `results/combine-resident-groups-evidence.tgz` | `3f3e3a827e769287ef3a1159d23303403481f0fdb97285436cd50c83ec20963e` |

A separate small MXU operand probe in `exp-c5bdngz3ad` tested FP8 LHS with a native
FP4 RHS at M64/K256/N256 and M128/K512/N256. The pinned compiler rejected
both with `Unsupported element type for matmul push rhs: 'f4E2M1FN'`.
Explicit FP4-to-FP8 conversion followed by asynchronous MXU operations was
bit-exact against the ordinary dot reference. This is a compiler/API result
for JAX/JAXLIB 0.11.2 and libtpu 0.0.50, not a claim about hardware capability;
there is no native-FP4-RHS timing result.

The same job tested putting CSA cache words and scale bits in a single
SparseCore gather record. For two 524288-record banks and 288-by-512 indices,
the 136- and 160-word formats were bit-exact but increased device module time
excluding entry barriers from 0.174463 ms to 1.752655 and 1.724891 ms.
Temporary storage grew from 1376064 bytes to 1360035360 and 1410367008 bytes.
Compiled HLO contains full-bank layout copies for these candidates. Packing
occurred outside timing; the copies are inside the measured executable.
The 129-word format failed numerical checks and was excluded from timing.
A flat-input variant failed compilation with `Reshape 1D ref is not supported.`
These results reject these layouts, not every possible packed representation.
The job completed successfully; its result/HLO/device-event files were retrieved
through a read-only CPU helper. The verified local archive SHA-256 is
`85ae39d7abf1685f2fe382cca6014a62e1d7463e37028adb7e6121417d210991`.

Candidate storage, dense-tail, DMA-window, routing-window and gather experiments
above ran in `exp-qg29pd6unq` after its serving process exited. The initial
1024–1800-block tail test was faster, but that density was too low to represent
the existing serving captures, whose maximum unions exceeded 5000 blocks.
The dense and full-union results determine the rejection.

Increasing SparseCore gather read concurrency from eight to sixteen did not
complete within 300 seconds after the baseline variant. Only that benchmark
process was terminated; this does not establish the cause of the stall.
The subsequent writeback-overlap comparison retained eight reads.

The mHC and asymmetric MoE experiments ran in `exp-1yxqyt6ap6`. A single
288-row mHC projection initially changed only the FP32 gates by at most
2.384e-7; retaining three 96-row projection chunks restored bit equality but
did not improve performance. The asymmetric MoE kept capacity 128 and the
automatic height ladder. Compact gate/up staging and accounting for the actual
256-column widened blocks required 62,144,512 bytes against a 65,766,686-byte
VMEM budget; compilation and both whole-layer output checks passed. The biased
case used `sqrtsoftplus`, correction bias and routed scale 1.5 at seed 1030;
the uniform softmax case used seed 1031. The packed-word FP4 expansion used the
same two input distributions and preserved signed zero, but regressed both
whole-layer device timings by more than eight times.

A further FP8 candidate-score diagnostic in that experiment applied recovered
query/key scales before the head reduction. It still changed the selected ID
arrays at compute tiles 2048, 4096 and 8192. Those candidates were excluded from
timing and remain outside the production path.

A separate 192-by-8192 dot diagnostic confirmed bit-exact query/key
round-trips through the FP8 cache and power-of-two scales. Both XLA and Pallas
FP8 dots nevertheless differed from their BF16-input references, while XLA
and Pallas agreed for each input type. `highest` precision did not remove the
difference, and this pinned TPU runtime rejected `ANY_F8_ANY_F8_F32` as
unsupported. This rules out cache loss as the cause in that diagnostic; it
does not establish the backend's internal arithmetic.

Dense FP8 GEMM fusion was measured in `exp-sh81ry5nm6` at commit `0e742279`,
with the same pinned TPU runtime and seed 1028. All 57 baseline/candidate cases
were bit-exact, including signed zero. The table compares the reference with the
fastest Pallas candidate for each observed shape at 288 rows. Times are device
module medians excluding entry barriers, from five ordered profile repetitions.

| K | N | Reference (us) | Best Pallas (us) | Candidate |
| --- | --- | --- | --- | --- |
| 5120 | 1792 | 9.113 | 12.473 | Fused quantization, M=288/N=256 |
| 1280 | 32768 | 26.748 | 30.552 | Fused quantization, M=288/N=2048 |
| 8192 | 5120 | 25.608 | 24.849 | Fused quantization, M=288/N=1024 |
| 5120 | 4608 | 17.719 | 19.126 | Fused quantization, M=288/N=512 |
| 2304 | 5120 | 11.107 | 12.235 | Fused quantization, M=288/N=1024 |

Four shapes regressed. The remaining 0.759 us isolated gain has no second-seed
or serving validation and does not justify a dispatch change. Some synchronized
host medians improved by 1–4 us despite slower device execution; those host
differences are not evidence of a faster kernel. No GEMM replacement is enabled.

```bash
python scripts/dsv41_bench/bench_dense_fp8.py \
  --out /tmp/dense-fp8 --seed 1028 --profile
```

Use a fresh output directory. The benchmark records every numerical failure and
compilation error, and times only bit-exact candidates. `--interpret-check` runs
the small CPU check instead of TPU timing.
