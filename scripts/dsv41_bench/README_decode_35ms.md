<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSv4.1 decode measurements, 2026-10-21 UTC

The 35 ms main-model forward and 14000 output tokens/chip/s targets remain
unmet. The serving measurements below cover an FP4 MoE height ladder for the
observed EP8 geometry and a joint request/scorer configuration for the
48-request full-context indexer, whole-batch router selection and WO_A row chunks.

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

## mHC consumer-fusion prototypes

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
barriers. Neither prototype is in serving dispatch. The first requires
passing the prequantized values to the consumers; the second retains the
existing interface but changes output memory placement. Other consumers,
whole-model memory pressure and serving performance remain unvalidated.

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

A separate small MXU operand probe on that job tested FP8 LHS with a native
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

The mHC VMEM-output candidate retains the normalized BF16 activation in VMEM
for following consumers, without changing their interface or arithmetic.
At seeds 1034 and 1036, all mHC outputs, FP8 activation/scale and projected
values were bit-exact. Device medians for mHC plus the following 5120-by-1792
projection fell from 27.747 to 26.634 us and from 27.646 to 26.607 us.
The dispatch is limited to the kernel's TPU7x envelope with 288 tokens,
four streams, width 5120, BF16 norm weights and HBM gate inputs.
Other shapes retain the previous output placement. Serving validation is
pending; these isolated timings do not establish a whole-model improvement.

```bash
git show a04817ac:src/vllm_torchtpu/kernels/deepseek_v41/mhc/shifted_post_pre_rms.py \
  > /tmp/mhc-output-reference.py
python scripts/dsv41_bench/bench_mhc_output.py \
  --reference-source /tmp/mhc-output-reference.py \
  --out /tmp/mhc-output --seed 1045 --profile
```

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
