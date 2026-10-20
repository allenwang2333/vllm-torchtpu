<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSv4.1 decode measurements, 2026-10-20 UTC

The 35 ms main-model forward and 14000 output tokens/chip/s targets remain
unmet. The serving measurements below cover an FP4 MoE height ladder for the
observed EP8 geometry and a joint request/scorer configuration for the
48-request full-context indexer. A subsequent router tile change has separate
kernel evidence below; these serving results do not include it.

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

The FP4 run's top-level runtime source and actual checkout are `c6ffd0fa`;
its copied nested source metadata still names the base. The first combined
trial in `exp-qcbagjr3z8` was interrupted by the experiment's two-hour limit
before formal results. Only the completed rerun above is reported.

## Serving results

Each cell with three values lists rounds 1, 2 and 3. Fixed-acceptance
throughput uses `384 * 3.47 / step_seconds / 4`, matching the reference
harness. Observed throughput uses accepted output tokens including bonus
tokens divided by the measured steady window; it has a different numerator.

| Metric | Base | FP4 ladder | FP4 ladder + indexer |
| --- | ---: | ---: | ---: |
| Host step median, ms | 55.011 / 55.019 / 54.959 | 54.090 / 54.278 / 54.170 | 53.086 / 52.586 / 52.530 |
| Fixed-acceptance output tokens/chip/s | 6056 / 6055 / 6061 | 6159 / 6137 / 6149 | 6275 / 6335 / 6341 |
| Observed accepted+bonus tokens/chip/s | 6773 / 6840 / 6944 | 6945 / 6752 / 6891 | 7120 / 7048 / 7207 |
| Main-model device forward, ms | 47.038 | 45.184 | 42.918 |
| Forward minus identified entry barriers, ms | 42.722 | 42.438 | 42.264 |
| Main-model FP4 MoE event time/step, ms | 15.442 | 15.163 | 15.119 |
| Full-context index score event time/step, ms | 2.836 | 2.835 | 2.670 |
| Startup HBM used per logical device, GiB range | 48.98–49.04 | 48.99–49.04 | 48.99–49.05 |
| Server launch to health ready, seconds | 1049.86 | 1059.46 | 1006.77 |

Forward is the median of six per-step sums of the five sequential rank-0
main-model device modules, with 40 MoE layers verified in each step.
It excludes draft, logits/sampling and host gaps. It is not a sum of module
medians. Subtracting the identified `barrier-cores` intervals gives a separate
diagnostic metric, not a replacement for the actual forward measurement.
Most of the 47.04-to-42.92 ms change is entry-barrier variation; only about
0.46 ms remains after this subtraction. Operator durations are inclusive and
may overlap other work, so they must not be summed as a critical-path estimate.

HBM values are rounded startup server-log statistics, not sampled runtime
peaks. The joint indexer increases its isolated executable's temporary HBM by
12 MiB (42390016 to 54658304 bytes). Neither change repacks model weights.
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
