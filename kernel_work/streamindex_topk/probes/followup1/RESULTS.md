# Step-3 follow-up: four probes, one job

**Job `j-59bbfd2f-e408-4a75-a155` — `JobStatus: Succeeded`, `state: Complete`.**
`tpu-fanout: 4/4 shards succeeded`; per-shard `rc`:

```
===== tpu-fanout EXIT CODES =====
shard0   chips=0        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh 0
shard1   chips=1        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh 1
shard2   chips=2        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh 2
shard3   chips=3        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh 3
```

`RUN_CMD`, **as read back from `cdk job recipe j-59bbfd2f-e408-4a75-a155 --no-color`**
(line-wrapped by the YAML emitter; it is one line):

```
cd /root/cloud-devkit/vllm-torchtpu && export PYTHONPATH=/workspace:$PYTHONPATH
  && bash /root/fanout/tpu-fanout.sh "bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh
  0" "bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh
  1" "bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh
  2" "bash /workspace/kernel_work/streamindex_topk/probes/followup1/shards.sh
  3"
```

Image, from the same read-back:
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`
(the prebuilt image; nothing was rebuilt). Probe code carried in with
`--map-dir …/kernel_work:tpu-job:/workspace/kernel_work`, fan-out helper with
`--map-dir …/run-command-in-cdk/fanout:tpu-job:/root/fanout`. Branch
`annyan/streamidx-scores`; the probes are commit `3dc5932d`. Container reports
`numpy 2.3.5, jax 0.10.2, python 3.12.14`.

Nothing under `kernel_work/streamindex_topk/instruments/` was edited. Every
constant that moved was moved by rebinding a module attribute from the probe
script and restored in a `finally`; `git status` shows no modification under
`instruments/`.

**Sign convention, binding on every table here: every percentage is a change in
TIME, `(new - old) / old`. NEGATIVE IS FASTER.** Signed percentages appear only
in a column named `Δ time %`. A `speedup ×` column is a different number
(>1 is faster). Every delta sits in the same row as the two absolute times it
came from.

**Tags.** Every number below is `measured (j-59bbfd2f-e408-4a75-a155)` unless it
carries a different tag inline. Prior numbers quoted for comparison carry their
own job id. `modelled` marks anything computed rather than timed.

---

## Shard 0 — option A2 (`w <- max(w, 0)`, keep the ReLU): **FAIL at all five regimes**

Gate: the surrogate is scored by handing `instruments/test_correctness.oracle_scores`
a shallow dict copy whose `_weights` is `max(w, 0)` — so the A2 surrogate and the
float64 ReLU oracle differ by exactly one array and share every line of masking
and arithmetic. Same structure as the A1 gate (`variants/drop-relu/a1_gate.py`,
`measured (j-5acf2993-b56b-4a38-9abe)`), including its negative control and its
oracle self-reference arm.

`measured (j-59bbfd2f-e408-4a75-a155)`, verbatim from `fanout/shard0.log`:

| regime | rows | bad | `worst_rank` | allowed | `min_recall` | required | control | self-ref | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `decode_prod` | 64 | 64 | 8190 | 2304 | 0.47656 | 0.99 | FIRED | OK | **FAIL** |
| `mixed_prod` | 1024 | 1024 | 8175 | 2304 | 0.44141 | 0.99 | FIRED | OK | **FAIL** |
| `mixed_32k` | 1024 | 1024 | 32677 | 2304 | 0.21973 | 0.99 | FIRED | OK | **FAIL** |
| `mixed_64k` | 1024 | 1024 | 65286 | 2304 | 0.20215 | 0.99 | FIRED | OK | **FAIL** |
| `decode_64k` | 6 | 6 | 64614 | 2304 | 0.43750 | 0.99 | FIRED | OK | **FAIL** |

`0 PASS / 5 FAIL of 5 regimes.` Every row of every regime is bad — 3138 of 3138,
the same total A1 failed. This is not marginal: `worst_rank` overruns the allowed
2304 by up to 28×, and `min_recall` misses the required 0.99 by a factor of two
to five.

The two control arms both behaved, so the gate was not vacuous: the negative
control (zero the second half of the query heads) FIRED on every regime, and the
oracle self-reference arm returned `worst_rank = 2048, min_recall = 1.00000`
everywhere. `check_fast` was cross-validated against
`instruments/test_correctness.check` on the A2 arm at every regime.

**Why it fails, from the same log.** The clamp is not a small perturbation on
this input distribution. `instruments/inputs.py` draws
`weights = rng.uniform(-1.5, 1.5, (num_tokens, INDEX_N_HEADS))`, so about half of
every token's 32 head weights are negative and `max(w, 0)` deletes them:

| regime | negative weights | mean heads zeroed per token (of 32) | rows with every head zeroed |
| --- | ---: | ---: | ---: |
| `decode_prod` | 1021 / 2048 (49.85%) | 15.95 | 0 |
| `mixed_prod` | 16364 / 32768 (49.94%) | 15.98 | 0 |
| `mixed_32k` | 16566 / 32768 (50.56%) | 16.18 | 0 |
| `mixed_64k` | 16560 / 32768 (50.54%) | 16.17 | 0 |
| `decode_64k` | 103 / 192 (53.65%) | 17.17 | 0 |

Zeroing ~16 of 32 heads per token is very nearly the negative control's own
mutation, and the two arms land in the same place — e.g. `mixed_64k`: A2
`min_recall = 0.20215`, negative control `min_recall = 0.05176`. The brief
described A2 as "a different and much smaller change" than A1; measured, it is
not smaller in effect.

### Consequence for E1

**E1 stays disqualified.** A2 does not establish the non-negativity precondition
by construction, because A2 is not accuracy-preserving. E1's measured 13.70% win
remains disqualified on the same grounds it already was.

The one thing this does *not* settle is whether real GLM-5.2 indexer weights are
as symmetric about zero as the synthetic ones. They are not obviously
non-negative — `models/vllm/deepseek_v4/attention.py:397` produces
`indexer_weights` from a bare `self.indexer.weights_proj(hidden_states)` linear
with no non-negative activation, and
`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_indexer.py:81` only scales it
(`weights = indexer_weights * softmax_scale * n_head**-0.5 * q_scales`, all
positive factors). So negative `w` is reachable in production and the clamp is a
real functional change there too. But *how often* and *how large* is a property
of trained weights that this gate cannot see, and the gate's own input
distribution is the worst case for A2. If anyone wants to revive A2, the next
step is a weight histogram from a real checkpoint, not another run of this gate.

---

## Shard 1 — the `NUM_HEADS` ladder: the head-sum bundle is **dominated by the constant term**

> **This is a decomposition, not an optimisation. A smaller `NUM_HEADS` computes a
> different function and is not shippable. No row below is a win and none may be
> quoted as one.** The deliverable is the pair of coefficients.

Measured in `probes/vpu_mxu_strike.py`, the same probe the published 3.858 µs
head-sum figure came from, with `VM.NUM_HEADS` rebound per rung inside a
`try/finally` so the kernel body is byte-identical across the ladder.
`REPS/BATCH/WARMUP` raised to 30/8/5 from 10/4/3.

### Raw arms — `measured (j-59bbfd2f-e408-4a75-a155)`

`grid=32, bq_sz=128, bkv_sz=2048, head_dim=128`; µs per inner step.

| H | gen | epi | K | µs/step | spread % | s MiB | GFLOP/step |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | mxu | sum | 128 | 3.4991 | 27.03 | 4.0 | 0.268 |
| 4 | mxu | sum | 512 | 3.7668 | 14.14 | 4.0 | 1.074 |
| 4 | mxu | full | 128 | 4.4040 | 2.50 | 4.0 | 0.268 |
| 4 | bcast | sum | 128 | 1.5559 | 54.76 | 4.0 | 0.000 |
| 8 | mxu | sum | 128 | 2.8797 | 2.67 | 8.0 | 0.537 |
| 8 | mxu | sum | 512 | 3.2936 | 2.61 | 8.0 | 2.147 |
| 8 | mxu | full | 128 | 4.2198 | 16.04 | 8.0 | 0.537 |
| 8 | bcast | sum | 128 | 1.4999 | **300.57** | 8.0 | 0.000 |
| 16 | mxu | sum | 128 | 3.2612 | 5.32 | 16.0 | 1.074 |
| 16 | mxu | sum | 512 | 4.6818 | 1.92 | 16.0 | 4.295 |
| 16 | mxu | full | 128 | 4.8741 | 2.64 | 16.0 | 1.074 |
| 16 | bcast | sum | 128 | 1.5000 | 26.44 | 16.0 | 0.000 |
| 32 | mxu | sum | 128 | 4.1285 | 3.35 | 32.0 | 2.147 |
| 32 | mxu | sum | 512 | 6.9976 | 2.46 | 32.0 | 8.590 |
| 32 | mxu | full | 128 | 6.5704 | 1.85 | 32.0 | 2.147 |
| 32 | bcast | sum | 128 | 1.6058 | 12.87 | 32.0 | 0.000 |
| 64 | mxu | sum | 128 | 5.3992 | 1.28 | 64.0 | 4.295 |
| 64 | mxu | sum | 512 | 12.3218 | 2.31 | 64.0 | 17.180 |
| 64 | mxu | full | 128 | 9.9464 | 0.90 | 64.0 | 4.295 |
| 64 | bcast | sum | 128 | 1.4977 | 10.37 | 64.0 | 0.000 |

`H = 64` built and lowered; the ladder is five rungs, not four.

### The fits — `t(H) = a·H + b`, µs per inner step

`a` is the term **linear in `NUM_HEADS`** (the reduction arithmetic); `b` is the
term **independent of `NUM_HEADS`** (the output store plus fixed per-grid-step
cost).

| series | `a` (µs/head) | `b` (µs) | R² | rms resid | at H=32: linear / constant / total | which dominates |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `bcast_sum` | −0.000205 | 1.5370 | 0.011 | 0.0428 µs | −0.0066 / 1.5370 / 1.5304 µs → linear is −0.4% | **constant** |
| `mxu_sum_mxu_removed` | +0.001056 | 3.0144 | 0.009 | 0.2478 µs | 0.0338 / 3.0144 / 3.0482 µs → linear is 1.1% | **constant** |
| `mxu_sum_raw` | +0.038438 | 2.8803 | 0.905 | 0.2720 µs | 1.2300 / 2.8803 / 4.1103 µs → linear is 29.9% | **constant** |
| `mxu_full` | +0.097370 | 3.5882 | 0.986 | 0.2495 µs | 3.1158 / 3.5882 / 6.7040 µs → linear is 46.5% | **constant** |

**Answer to the question asked: the constant term dominates in every arm, and in
the two arms where the matmul's contribution is absent or removed the linear term
is statistically indistinguishable from zero (R² = 0.011 and 0.009).**

The ordering of the four rows is the whole story. `mxu_full` and `mxu_sum_raw`
contain a matmul whose FLOPs are themselves ∝ H (see the `GFLOP/step` column:
0.268 → 4.295 across the ladder), so their `a` prices the *matmul*, not the
head-sum reduction. Take the matmul out — either by never doing one (`bcast_sum`)
or by subtracting the per-rung K-slope estimate of it (`mxu_sum_mxu_removed`) —
and `a` collapses by a factor of 36 to 92 and R² collapses to noise.

### Is it linear? — fit on the low rungs, predict the top rung

| series | fit on | predicted t(64) | measured t(64) | error, % of measurement |
| --- | --- | ---: | ---: | ---: |
| `bcast_sum` | H = 4,8,16,32 | 1.6665 µs | 1.4977 µs | −11.27 |
| `mxu_sum_raw` | H = 4,8,16,32 | 5.0183 µs | 5.3992 µs | +7.05 |
| `mxu_sum_mxu_removed` | H = 4,8,16,32 | 3.0291 µs | 3.0916 µs | +2.02 |
| `mxu_full` | H = 4,8,16,32 | 9.1277 µs | 9.9464 µs | +8.23 |

The two matmul-carrying arms under-predict the top rung by 7-8%, i.e. they are
mildly *super*-linear in H — consistent with the matmul, not with a reduction.

### What this does and does not support

- **The `bcast_sum` fit cannot be quoted as a coefficient.** Its arm spreads run
  to 300.57%, 54.76% and 26.44%, hundreds of times the −0.4% "effect" the fitted
  `a` corresponds to. An effect at or below the spread of the arms it is a
  difference of cannot support a conclusion, and this one is far below.
- **The qualitative conclusion survives anyway**, because a linear term would
  have to be enormous to hide here. The five `bcast_sum` medians are 1.5559,
  1.4999, 1.5000, 1.6058, 1.4977 µs — a 7.2% total range while H grows 16×. If
  the reduction arithmetic were linear in H, the H=64 rung would carry 16× the
  H=4 rung's reduction cost. It carries the same time, and the H=64 rung is one
  of the *quieter* ones (10.37% spread).
- **`mxu_sum_mxu_removed` is derived from a two-point K slope that the probe
  itself flags as unsafe.** The implied matmul rates are 3008, 3891, 2267, 2245
  and 1861 Tops/s at H = 4, 8, 16, 32, 64 — at or above fp8 peak at every rung,
  which means the K slope is not cleanly measuring the matmul. If the matmul is
  under-attributed, `rest_sum` is over-attributed, so 3.0144 µs is an upper
  bound on the constant term of that arm and the real linear term is, if
  anything, smaller still.
- **`mxu_sum_raw`'s low rungs are not monotone**: H=4 (3.4991 µs) is *slower*
  than H=8 (2.8797 µs), and its H=4 spread is 27.03%. That single point drives
  most of its residual. Dropping it would raise `a` and lower `b`, but not enough
  to change which dominates.

### Reconciliation with the published bundle

| quantity | value | tag |
| --- | ---: | --- |
| published "head-sum bundle" | 3.858 µs/step | `measured (j-b54e9716-3db5-4f1f-b8d4)` |
| published whole mixed inner step | 8.42 µs/step | `measured (j-5f8f4a08-20b4-4fa0-bfc3)` |
| `bcast_sum` at H=32 (fitted) | 1.5304 µs | `measured (j-59bbfd2f-e408-4a75-a155)` |
| `mxu_sum_raw` at H=32 (fitted) | 4.1103 µs | `measured (j-59bbfd2f-e408-4a75-a155)` |
| `mxu_sum_mxu_removed` at H=32 (fitted) | 3.0482 µs | `measured (j-59bbfd2f-e408-4a75-a155)` |
| direct, `mxu/sum` at H=32, K=128 | 4.1285 µs (3.35% spread) | `measured (j-59bbfd2f-e408-4a75-a155)` |

The published 3.858 µs sits *between* the arms, and the arms disagree with each
other by 2.7× (1.53 vs 4.13 µs). **That disagreement is itself a finding: the
3.858 µs "head-sum bundle" is not a robust quantity — it depends heavily on which
arm is taken to represent it.** The direct `mxu/sum @ H=32, K=128` measurement
(4.1285 µs, 3.35% spread) reproduces the same probe's own configuration and is
the number to trust of the six.

### What this means for pricing E1, E2 and F

The deliverable was the coefficients, and they are `a ≈ 0` and `b ≈ 1.5-3.0 µs`
depending on the arm. Read literally, that says **the head-sum bundle is not
reduction-arithmetic-bound**, so an option that makes the per-head reduction
cheaper — fewer heads' worth of arithmetic, a cheaper reduction, moving the
reduction to the MXU — buys close to nothing, because the arithmetic is not what
the 3.858 µs is. Whatever `b` is (output store bandwidth, per-grid-step fixed
cost, or both) is where the term lives, and none of the three options is aimed at
it. That is a prediction from a decomposition, not a measurement of E1, E2 or F;
each still needs its own arm. But it moves all three down the queue rather than
up, which is the opposite of what the bundle being 52.5% of the step suggested.

---

## Shard 2 — option B: the linearity is **half real**, and the P claim is **confirmed directly**

Six live `kv_len` rungs at `max_model_len = 66560`, `max_num_seqs = 6`,
`num_tokens = 1024`, `early_exit = True` (matching `run_perf.py`'s default),
`reps = 10`, both arms through the read-only `microbench.run_regime` with
`measure_topk=False`, variant bound with `_rs_support.bind(var, kv_len_bucket=…)`.

The two ends of the ladder are not merely similar to §5c's anchors, they are them
— the script asserts it rather than claiming it, and both assertions printed
`True`:

```
anchor check: kv=8192 rung is field-for-field mixed_confound: True
anchor check: kv=65536 rung is field-for-field mixed_64k: True
```

### Cross-check against §5c before anything is refitted

| arm | w | published ms `measured (j-d9502595-54a2-45bc-947b)` | this run ms `measured (j-59bbfd2f-e408-4a75-a155)` | Δ time % |
| --- | ---: | ---: | ---: | ---: |
| B | 64 | 0.61196 | 0.61472 | +0.45 |
| B | 512 | 3.74624 | 3.78477 | +1.03 |
| incumbent | 64 | 2.27933 | 2.29785 | +0.81 |
| incumbent | 512 | 4.78598 | 4.78328 | −0.06 |

Negative is faster. All four reproduce within ~1%; this is a cross-job
reproduction check, not an A/B. §5c's four anchor numbers are sound.

### (a) Is the estimate's linearity real?

**B: yes, strikingly. The incumbent: no.**

| arm | refit, six points | R² | rms resid | max abs resid | §5c two-point fit |
| --- | --- | ---: | ---: | ---: | --- |
| incumbent | `t = 2.32604 + 0.005338·w` | **0.88555** | 292.3 µs | 507.4 µs | `1.92125 + 0.005595·w` |
| B (exact bucket) | `t = 0.14745 + 0.007103·w` | **0.99990** | 10.9 µs | 17.1 µs | `0.16422 + 0.006996·w` |

Per-point residuals, `measured (j-59bbfd2f-e408-4a75-a155)`:

| w | incumbent ms | incumbent fitted | resid | resid % | B ms | B fitted | resid | resid % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.29785 | 2.66766 | −0.36980 | −16.09 | 0.61472 | 0.60206 | +0.01266 | +2.06 |
| 128 | 2.93598 | 3.00928 | −0.07330 | −2.50 | 1.05915 | 1.05667 | +0.00247 | +0.23 |
| 192 | 3.54293 | 3.35089 | +0.19204 | +5.42 | 1.49414 | 1.51129 | −0.01714 | −1.15 |
| 256 | 4.19986 | 3.69251 | +0.50735 | +12.08 | 1.95545 | 1.96590 | −0.01045 | −0.53 |
| 384 | 4.39514 | 4.37574 | +0.01940 | +0.44 | 2.88717 | 2.87512 | +0.01205 | +0.42 |
| 512 | 4.78328 | 5.05897 | −0.27569 | −5.76 | 3.78477 | 3.78435 | +0.00042 | +0.01 |

B's residuals are ≤ 2.1% and unsigned-random; its arm spreads are 0.46-7.56%.
B is linear in live width to within the measurement.

The incumbent's residuals are up to 12.1% against arm spreads of 1.65-3.26%, and
they are not random — they trace an arch (−16, −2.5, +5.4, +12.1, +0.4, −5.8).
The incumbent is **not** linear. Its local slope between adjacent measured rungs:

| w interval | Δ t (ms) | local slope (ms/sublane) |
| --- | ---: | ---: |
| 64 → 128 | 0.63813 | 0.009971 |
| 128 → 192 | 0.60695 | 0.009484 |
| 192 → 256 | 0.65693 | 0.010265 |
| 256 → 384 | 0.19528 | **0.001526** |
| 384 → 512 | 0.38814 | 0.003032 |

There is a **kink between w = 256 and w = 384** (`kv_len` 32768 → 49152) where the
incumbent's marginal cost drops by a factor of 6.7 and never fully recovers.
Below the kink the incumbent costs ~0.0100 ms/sublane; above it, 0.0015-0.0030.
The same interval in B is perfectly ordinary (0.007279 then 0.007013 ms/sublane).
**This is unexplained.** It is not the partition count — the read-back below shows
the incumbent at P=2 throughout — and it is not a shape artefact, since the
pages-per-sequence, the block count and the buffer width all move smoothly. It is
a real, reproducible feature of the incumbent at 1.65-1.85% arm spreads, and it is
the reason a straight line does not fit.

### The flagged suspicion: is B's slope really steeper?

| | slope (ms/sublane) | source |
| --- | ---: | --- |
| incumbent, refitted (6 points) | 0.005338 | `measured (j-59bbfd2f-e408-4a75-a155)` |
| B, refitted (6 points) | 0.007103 | `measured (j-59bbfd2f-e408-4a75-a155)` |
| ratio B / incumbent | 1.331 | derived |
| §5c two-point ratio | 1.250 (0.006996 / 0.005595) | `measured (j-d9502595-54a2-45bc-947b)` |

**Yes — and it is a fitting artefact, now with an explanation.** B's slope really
is ~0.0070 ms/sublane, and it is that everywhere (0.00680-0.00728 across all five
intervals). The incumbent has no single slope: it is *steeper* than B below
w = 256 (≈0.0100 vs ≈0.0070) and much flatter above it (0.0015-0.0030). Fitting one
straight line to that curve averages the two regimes into 0.005338 and produces
the "B is steeper" appearance. So the answer to the §5c flag is not that B has an
anomalous per-byte cost; it is that **the incumbent's line was never a line**, and
§5c happened to place its two points (w = 64 and w = 512) one on each side of the
kink, which is the worst possible pair for detecting it.

**Consequence: the §5c extrapolation does need redoing, but not for the reason
suspected.** It is the incumbent leg that is wrong, not B's.

### (b) What bucketing actually costs — measured

`Δ time % = (B − incumbent) / incumbent`. **Negative is faster.**

| kv_len | w | ladder | bucket | B cols | inc ms | inc sp% | B ms | B sp% | Δ time % | speedup × |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 64 | exact | 8192 | 8192 | 2.29785 | 3.26 | 0.61472 | 0.75 | −73.25 | 3.738 |
| 8192 | 64 | pow2 | 8192 | 8192 | 2.29785 | 3.26 | 0.61472 | 0.75 | −73.25 | 3.738 |
| 8192 | 64 | gran8k | 8192 | 8192 | 2.29785 | 3.26 | 0.61472 | 0.75 | −73.25 | 3.738 |
| 16384 | 128 | exact | 16384 | 16384 | 2.93598 | 2.91 | 1.05915 | 0.66 | −63.93 | 2.772 |
| 16384 | 128 | pow2 | 16384 | 16384 | 2.93598 | 2.91 | 1.05915 | 0.66 | −63.93 | 2.772 |
| 16384 | 128 | gran8k | 16384 | 16384 | 2.93598 | 2.91 | 1.05915 | 0.66 | −63.93 | 2.772 |
| 24576 | 192 | exact | 24576 | 24576 | 3.54293 | 1.86 | 1.49414 | 0.46 | −57.83 | 2.371 |
| 24576 | 192 | **pow2** | **32768** | 32768 | 3.54293 | 1.86 | 1.52770 | 0.65 | −56.88 | 2.319 |
| 24576 | 192 | gran8k | 24576 | 24576 | 3.54293 | 1.86 | 1.49414 | 0.46 | −57.83 | 2.371 |
| 32768 | 256 | exact | 32768 | 32768 | 4.19986 | 1.65 | 1.95545 | 7.56 | −53.44 | 2.148 |
| 32768 | 256 | pow2 | 32768 | 32768 | 4.19986 | 1.65 | 1.95545 | 7.56 | −53.44 | 2.148 |
| 32768 | 256 | gran8k | 32768 | 32768 | 4.19986 | 1.65 | 1.95545 | 7.56 | −53.44 | 2.148 |
| 49152 | 384 | exact | 49152 | 49152 | 4.39514 | 1.85 | 2.88717 | 2.67 | −34.31 | 1.522 |
| 49152 | 384 | **pow2** | **65536** | 65536 | 4.39514 | 1.85 | 2.92069 | 1.29 | −33.55 | 1.505 |
| 49152 | 384 | gran8k | 49152 | 49152 | 4.39514 | 1.85 | 2.88717 | 2.67 | −34.31 | 1.522 |
| 65536 | 512 | exact | 65536 | 65536 | 4.78328 | 2.95 | 3.78477 | 1.73 | −20.88 | 1.264 |
| 65536 | 512 | pow2 | 65536 | 65536 | 4.78328 | 2.95 | 3.78477 | 1.73 | −20.88 | 1.264 |
| 65536 | 512 | pow2/gran8k | 65536 | 65536 | 4.78328 | 2.95 | 3.78477 | 1.73 | −20.88 | 1.264 |

Only two rungs distinguish the ladders — the other four have all three buckets
coincide. At those two, rounding up costs remarkably little:

| rung | exact bucket B ms | coarser bucket B ms | Δ time % of the coarsening | overshoot |
| --- | ---: | ---: | ---: | --- |
| kv 24576, pow2 | 1.49414 (bucket 24576) | 1.52770 (bucket 32768) | **+2.25** | 1.33× width |
| kv 49152, pow2 | 2.88717 (bucket 49152) | 2.92069 (bucket 65536) | **+1.16** | 1.33× width |

`Δ time %` here is `(coarse − exact) / exact`; positive means the coarser bucket
is slower. Both are +1 to +2% for a 33% width overshoot, against arm spreads of
0.46-2.67% — so at the kv 49152 rung the coarsening cost (+1.16%) is *at or below*
the spread of the arms it is a difference of and cannot on its own support a
conclusion; at kv 24576 (+2.25% against 0.46% and 0.65% spreads) it can.

The 8192-granularity ladder never differs from exact on this ladder, because every
rung is already a multiple of 8192. Its cost shows up only at `kv_len` values
between the rungs, which this ladder does not sample.

**The §5c claim — that coarsening costs only the byte term and not the large half
— holds.** A 33% width overshoot buys back 1-2% of B's time while B's prize over
the incumbent at those rungs is 34-58%.

### The P read-back — direct, not inferred

`impl.sparsecore_topk` was replaced in each implementation's own module namespace
by a shim that records the `(b, n, k)` the kernel really hands over and asks the
real `sparsecore_topk._pick_partition` what it returns for that triple, then
delegates. Trace-time only; it cannot affect a measured step.

```
  kv_len arm                  bucket   cols n      b     k  p_min   P
    8192 incumbent              None    67584   1024  2048      2   2
    8192 B[exact+pow2+gran8k]    8192     8192   1024  2048      1   1
   16384 B[exact+pow2+gran8k]   16384    16384   1024  2048      1   1
   24576 B[exact+gran8k]        24576    24576   1024  2048      1   1
   24576 B[pow2]                32768    32768   1024  2048      1   1
   49152 B[exact+gran8k]        49152    49152   1024  2048      1   1
   49152 B[pow2]                65536    65536   1024  2048      1   1

incumbent P values observed: [2]   B P values observed: [1]
```

`MAX_SLICE_WORDS = 65536`, read from the module at run time. **Confirmed:** the
incumbent is at P = 2 at every rung (its `n` is always 67584, so `p_min = 2`), and
every B bucket is at P = 1. All six distinct B widths (8192, 16384, 24576, 32768,
49152, 65536) were recorded; the rows printed as "no trace recorded" are the ones
whose `n` was already traced at an earlier rung, so the inner `jax.jit` cache was
hit and no new record was made — the P for that `n` is the one already in the
table. `b = 1024 ≥ 32` so `max_p = 1` and `P = p_min` exactly, which is why no
bucket ever climbs above 1.

### §5c's bucketing table, redone

Two versions, because the incumbent is not linear and the choice of model changes
the answer. Both are **`modelled`** — an aggregation over a hypothetical 65536-token
prefill in 64 chunks of 1024 tokens — built on measured points, but not themselves
measured.

**Version 1, straight-line refit evaluated at §5c's mean w** (what §5c did, with
this run's coefficients):

| bucketing | mean w | B ms | inc ms | Δ time % |
| --- | ---: | ---: | ---: | ---: |
| exact `kv_len` (a ceiling) | 260 | 1.994 | 3.714 | −46.3 |
| 8192-token granularity (8 buckets) | 292 | 2.222 | 3.885 | −42.8 |
| powers of two (4 buckets) | 346 | 2.605 | 4.173 | −37.6 |
| *§5c published, same three rows* | | | | *−41.3, −34.6, −23.4* |

**Version 2, chunk-by-chunk average with time interpolated linearly between the
measured rungs** — which does not assume the incumbent is a line and is therefore
the one to prefer:

| bucketing | mean bucket w | B ms | inc ms | Δ time % | speedup × |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact `kv_len` (a ceiling) | 260.0 | 1.9964 | 3.7094 | −46.2 | 1.858 |
| 8192-token granularity (8 buckets) | 288.0 | 2.1941 | 3.7094 | −40.9 | 1.691 |
| powers of two (4 buckets) | 344.0 | 2.5905 | 3.7094 | −30.2 | 1.432 |

The two versions agree closely, so the incumbent's non-linearity, while real,
does not swing the aggregate much. **Both say §5c under-stated B's prize**, by
5 pp on the exact ladder, 6 pp on 8192-granularity and 7-14 pp on powers of two.
The published −23.4% for the pow2 ladder should be read as roughly **−30%**.

A caution on all six of these figures: they are an aggregation over an assumed
prefill schedule (one 65536-token sequence, 1024-token chunks, uniform weighting),
not a workload measurement. The measured per-rung table above is the result; this
is a way of summarising it.

---

## Shard 3 — `bkv_p = (8,4,4)` at `mixed_prod`: **usable now, and it is a slowdown**

Two passes in one process. `mixed_prod`: `phase=mixed, max_num_seqs=64,
num_tokens=1024, kv_len=8192, max_model_len=9216` — read back from the log, not
assumed. `early_exit=False` and `inner=20`, matching
`probes/bkv_p_prod.py --mode sweep`.

Buffer widths, recomputed with an explicit `bkv_ps` (the module default argument
does not see the rebinding):

| tuple | `num_sublanes_total` | scores | in flight at `inner=20` | budget |
| --- | ---: | ---: | ---: | ---: |
| (2,2,2) | 80 | 41.9 MB | 0.84 GB | 1.50 GB |
| (8,4,4) | 128 | 67.1 MB | 1.34 GB | 1.50 GB |

Neither arm exceeds `INFLIGHT_BUDGET_BYTES`, so the in-flight budget is ruled out
as the cause of the published 491.23%.

### Pass 1 — reproduction through the read-only `microbench.run_regime`, `reps = 40`

| tuple | subl | scores MB | inner | reps | full ms | spread % | Δ time % | published ms | published sp% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (2,2,2) | 80 | 41.9 | 20 | 40 | 0.67241 | 3.05 | 0.00 | 0.67250 | 2.83 |
| (8,4,4) | 128 | 67.1 | 20 | 40 | 0.70091 | 8.00 | **+4.24** | 0.70577 | **491.23** |

`Δ time %` is against (2,2,2); negative is faster. Published figures are
`measured (j-b54e9716-3db5-4f1f-b8d4)`.

**The 491.23% did not reproduce.** The same code path at 4× the reps returns
8.00%. Both medians reproduce to within 0.7% of the published ones, so the
published *median* was fine all along — it was the range that was poisoned, by
what was evidently a one-off host or allocator outlier.

### Pass 2 — 60 alternating paired blocks, one input build, both arms compiled once

| tuple | subl | n | median ms | min ms | max ms | spread % | IQR % | MAD % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (2,2,2) | 80 | 60 | 0.67238 | 0.66433 | 0.82235 | 23.50 | 1.39 | 0.57 |
| (8,4,4) | 128 | 60 | 0.70413 | 0.69559 | 0.71900 | 3.33 | 1.29 | 0.70 |

| old ms (2,2,2) | new ms (8,4,4) | Δ time % | speedup × |
| ---: | ---: | ---: | ---: |
| 0.67238 | 0.70413 | **+4.72** | 0.9549 |

Negative is faster; `speedup × = old/new`, >1 is faster. Both derived from the two
medians in that same row.

Paired per-block delta, n = 60 blocks:

- median **+4.52%**, IQR [+3.91, +5.61] (width 1.71 pp)
- min −13.98%, max +6.86%
- bootstrap 95% CI on the median delta: **[+4.23, +5.09] %**
- blocks where (8,4,4) was slower: **59 / 60 (98%)**

### Verdict: **USABLE**

- effect |median paired delta| = 4.52%
- worst arm range spread = 23.50% — but that is one outlier block in the (2,2,2)
  arm (max 0.82235 ms against a 0.67238 ms median), and it is the same block that
  produced the −13.98% minimum delta. A range statistic cannot survive one
  outlier; that is why it is not the statistic the verdict rests on.
- worst arm IQR = 1.39%, worst arm MAD = 0.70% — the effect is 3.3× the larger
  IQR and 6.5× the larger MAD.
- bootstrap CI on the paired median excludes zero, and 59 of 60 independent
  blocks agree on the sign.

**`(8,4,4)` is 4.2-4.7% slower than `(2,2,2)` at `mixed_prod`.** The three
consistent estimates are +4.24% (run_regime, reps=40), +4.52% (paired median) and
+4.72% (pooled medians), against the published +4.95%. The published *direction
and magnitude were right*; only its uncertainty was unusable. That row can now be
quoted.

This prices an existing sweep row. **It is not a proposal to ship `(8,4,4)`**,
whose 128-sublane scores buffer is 1.6× the shipped tuple's 80 and which is
slower at the regime measured.

---

## Things found that contradict, or complicate, the brief

1. **A2 is not "a much smaller change" than A1.** The brief framed it that way.
   Measured, it fails 3138 of 3138 rows — the same total A1 failed — and lands
   near the negative control at every regime, because the instrument's synthetic
   weights are `uniform(-1.5, 1.5)` and the clamp deletes about half of every
   token's heads. The consequence for E1 is the opposite of the hoped-for one.

2. **The premise behind the `NUM_HEADS` ladder does not survive it.** The brief
   describes the 3.858 µs head-sum bundle as "the single largest term". It is,
   but it is **not linear in `NUM_HEADS`** — in the arms where the matmul is
   absent or removed, the H-dependence is indistinguishable from zero. Options
   priced on the assumption that reducing per-head reduction work shrinks that
   term will be over-priced.

3. **The 3.858 µs figure itself is arm-dependent to within a factor of 2.7.**
   `bcast/sum` at H=32 gives 1.53 µs and `mxu/sum` gives 4.13 µs for what is
   nominally the same bundle. Whichever downstream estimate used 3.858 µs should
   carry that as an error bar.

4. **§5c's suspicion was correct but mis-attributed.** B's slope really is steeper
   than the incumbent's *fitted* slope, but B is the well-behaved arm
   (R² = 0.99990). It is the **incumbent** that is not linear (R² = 0.88555), with
   an unexplained 6.7× drop in marginal cost between w = 256 and w = 384. §5c's
   two anchor points sit one on each side of that kink.

5. **§5c under-stated B's prize, in the safe direction.** Redone from six points
   per arm, the pow2-bucket estimate moves from −23.4% to about −30%. Nothing
   downstream that was justified by the old number is invalidated; it was
   conservative.

6. **The 491.23% spread was not a property of the `(8,4,4)` arm.** It did not
   reproduce at 4× the reps on the same code path (8.00%). The published median
   was accurate to 0.7%. Any other row in that sweep carrying an anomalous spread
   deserves the same treatment before it is discarded.

7. **The branch moved under this work.** At dispatch `HEAD` was `eb994645`; when
   these probes were committed it was `54e80686`, with other agents' variant work
   landing concurrently. The probes are commit `3dc5932d` on
   `annyan/streamidx-scores`. The checkout also carried uncommitted modifications
   to several `variants/*` files and untracked `variants/mxu-head-sum/`,
   `variants/right-size/shard.sh` and `optimization_work/` at the time the
   `--map-dir` tarball was taken — so the pod ran the *working tree*, not the
   commit. For shards 0, 1 and 3 that is immaterial (they touch only
   `instruments/`, `probes/` and `src/`, none of which were dirty). For shard 2 it
   matters in principle: `variants/right-size/run_perf.py` and friends were dirty.
   Shard 2 does not import `run_perf.py` — it uses `_rs_support.py` and
   `streamindex_topk_rs.py`, both clean — and its four anchor numbers reproduce
   §5c to within 1%, which is the practical check.

8. **A whole-pod caveat on the timings.** Four shards shared one 16-CPU host, and
   shard 0 is CPU-bound float64 numpy. Each timed shard had its own TPU chip
   (`TPU_VISIBLE_CHIPS` 0/1/2/3, all four `rc=0` concurrently, which is itself
   proof the pins took), so device time is not contended, but host dispatch is.
   The instrument amortises dispatch over `inner` calls precisely to make this
   not matter, and shard 2's anchors reproducing an uncontended prior run to
   within 1% is evidence it did not. It is still the reason to prefer the paired
   statistics in shard 3 over the range spreads.
