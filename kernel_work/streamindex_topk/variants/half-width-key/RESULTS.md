# Option C — Half-Width Key

**Verdict: do not build.** The accuracy objection to C dissolves (fp16 is free);
the perf case does not survive measurement. C's whole prize is **at most 2.3–2.5%
of the kernel in the mixed band and under 1% in the decode band**, and that is a
ceiling computed from a *measured* HBM copy rate, not a modelled one. The build is
~300 lines across two files, on a SparseCore component that had no test at all
until this session. Rank 4 of 5 is confirmed by measurement.

All numbers below are single-chip microbenchmark numbers on one TensorCore. **None
of them is an end-to-end number**, and the scoring kernel's share of a real GLM-5.2
step is not established here.

Sign convention: every percentage is a change in **time**, `(new − old) / old`.
Negative is faster. Signed percentages appear only in the `Δ time %` column, beside
both absolute times.

---

## 1. Jobs

| job | `JobStatus` | what it ran |
| --- | --- | --- |
| `j-9d6276c2-8eaa-4dc6-9bef` | Succeeded (4/4 shards) | `sc_topk_test.py --suite all --negative-control` — SparseCore coverage. Its keyfmt arms are **invalid**, see §5. |
| `j-cb875cb9-4177-4756-84af` | Succeeded (4/4 shards) | keyfmt arms **invalid** (excess-precision elision, §5). Its sentinel and host `dist` arms are sound. |
| `j-7f77a816-5968-4ebd-80c3` | Succeeded (4/4 shards) | `keyfmt_probe.py` — the valid accuracy gate: `margins`, `churn`, `dist`. |
| `j-ec4de538-453c-4956-b523` | Succeeded | `bench_ab.py` — the perf A/B, incumbent and variant in the same job. |

`RUN_CMD` for the perf job, **as read back from `cdk job recipe`**:

```
export PYTHONPATH=/workspace:$PYTHONPATH; python3 /workspace/kernel_work/streamindex_topk/variants/half-width-key/bench_ab.py
  --impls incumbent,bf16out --regimes decode_prod,decode_64k,mixed_prod,mixed_64k
  --json $CDK_OUTPUT_DIR/bench_ab.json
```

`TPU_TOPOLOGY=2x2x1`, `cloud.google.com/gke-tpu-accelerator: tpu7x`, `ENABLE_XPROF=0`,
image `…/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.

---

## 2. Format decision: **fp16, not bf16**

C ships 16 bits either way, so the two formats cost exactly the same bytes and the
choice is decided entirely on rank movement. It is not close.

Device margins, real kernel + real `sparsecore_topk` + float64 host oracle,
`measured (j-7f77a816)`. `worst_rank` allowed ≤ 2304 (`RANK_SLACK=256`);
`min_recall` floor 0.99. Every figure verbatim from the log:

| regime | key | verdict | `worst_rank=` | `min_recall=` | recall margin above floor |
| --- | --- | --- | --- | --- | --- |
| `decode_prod` | f32 (incumbent) | PASS | 2109 | 0.99561 | 0.00561 |
| `decode_prod` | bf16 | PASS | 2102 | 0.99561 | 0.00561 |
| `decode_prod` | f16p | PASS | 2080 | 0.99561 | 0.00561 |
| `decode_prod` | f16ieee | PASS | 2080 | 0.99561 | 0.00561 |
| `decode_prod` | **m4 (positive control)** | **FAIL** | **2276** | **0.96338** (25 bad rows) | — |
| `mixed_prod` | f32 (incumbent) | PASS | 2098 | 0.99512 | 0.00512 |
| `mixed_prod` | bf16 | PASS | 2098 | 0.99414 | 0.00414 |
| `mixed_prod` | f16p | PASS | 2098 | 0.99512 | 0.00512 |
| `mixed_prod` | **m4 (positive control)** | **FAIL** | **2313** | **0.96436** (240 bad rows) | — |
| `decode_64k` | f32 (incumbent) | PASS | 2105 | 0.99512 | 0.00512 |
| `decode_64k` | bf16 | PASS | **2170** | **0.99170** | **0.00170** |
| `decode_64k` | f16p | PASS | 2105 | 0.99512 | 0.00512 |
| `decode_64k` | f16ieee | PASS | 2105 | 0.99512 | 0.00512 |
| `decode_64k` | **m4 (positive control)** | **FAIL** | **2577** | **0.94482** (5 bad rows) | — |

Negative control (second half of the query heads zeroed) **FIRED on every arm**:
`decode_prod` `worst_rank=8177, min_recall=0.49512` (m4 `0.49561`); `decode_64k`
`worst_rank=62644, min_recall≈0.270`.

**The argument for fp16 over bf16.** Against the incumbent's own margins — not
against the pass threshold — f16p is *indistinguishable from f32* at all three
regimes: identical `min_recall` everywhere, identical `worst_rank` at `mixed_prod`
and `decode_64k`, and 2080 vs 2109 at `decode_prod`. bf16 is not. At `decode_64k`
bf16 moves `worst_rank` from 2105 to 2170 and eats **two thirds of the recall
margin** (0.00512 → 0.00170 above a 0.99 floor). It still passes, but a change that
keeps passing while consuming most of the margin is exactly the result the dispatch
asked to be separated from one that does not move it. bf16 has 7 explicit mantissa
bits against fp16's 10, and at 2048-of-67584 the cut lands in a dense part of the
score distribution where 3 bits is the difference between 2.7 and 54.5 tied
candidates at the boundary (§3).

**Range is not an obstacle, and this was checked rather than assumed.** `dist` mode,
host-only over the real f64 oracle scores, `measured (j-7f77a816)`:

- `decode_prod`: `max|s|=209.91`, `min|s|(nonzero)=0.00023057`, neg frac 0.4890.
  `fp16 reach: overflow=0 subnormal=0 flush_to_zero=0 of 524288 finite scores`.
- `decode_64k`: `max|s|=197.632`, `min|s|(nonzero)=0.000114252`, neg frac 0.5245.
  `fp16 reach: overflow=0 subnormal=0 flush_to_zero=0 of 393216 finite scores`.

`max|s| ≈ 210` against fp16's 65504 is 312× of headroom, and the smallest nonzero
score is ~2× the smallest fp16 *normal*. Independently, the `f16ieee` arm (true IEEE
fp16 semantics, `astype` inside an `optimization_barrier`) and the `f16p` arm
(10-bit mantissa mask, f32 exponent range) came out **bit-identical at every regime
and in churn**, which is what you would expect only if the exponent range is nowhere
near binding.

**Recommended format if C is ever built: the precision-only 10-bit key** (`f16p`:
round-to-nearest-even to 10 explicit mantissa bits, f32 exponent retained, shipped
as a shifted/truncated 16-bit field). It is measurably as accurate as IEEE fp16 on
these inputs and does not stake correctness on a range property that was only
observed on the instrument's synthetic `standard_normal` inputs. Production weights
could put `max|s|` somewhere else; the exponent-preserving key does not care and
IEEE fp16 does. That caveat is the reason for the choice, and it is a caveat about
the *instrument*, not about fp16.

---

## 3. Rank movement, C against the incumbent

Index-set difference against the f32 arm, same seed, same inputs — `churn` mode,
`measured (j-7f77a816)`. This is the direct answer to "how much rank movement did
narrowing cause", counted in indices out of `k=2048`:

| regime | key | mean indices changed / 2048 | % | max on any row | rows identical to f32 |
| --- | --- | --- | --- | --- | --- |
| `mixed_64k` | m4 (control) | 60.3 | 2.946 | 213 | 8 / 1024 |
| `mixed_64k` | bf16 | 7.5 | 0.368 | 30 | 48 / 1024 |
| `mixed_64k` | **f16p** | **1.1** | **0.055** | **5** | 285 / 1024 |
| `mixed_64k` | f16ieee | identical to f16p | | | |
| `decode_64k` | m4 (control) | 53.7 | 2.620 | 112 | 0 / 6 |
| `decode_64k` | bf16 | 9.2 | 0.448 | 12 | 0 / 6 |
| `decode_64k` | **f16p** | **1.0** | **0.049** | **3** | 3 / 6 |
| `decode_64k` | f16ieee | identical to f16p | | | |

Both regimes report `narrowing positive control (m4, 4 mantissa bits): OK`.

Tie multiplicity at the k-th cut, host-only, explains the ordering — the number of
candidates that collide at the boundary after rounding (`dist`, `j-7f77a816`):

| regime | key | ties at cut, mean | max | host `worst_rank` / `min_recall` |
| --- | --- | --- | --- | --- |
| `decode_prod` | m4 | 99.3 | 305 | 2316 / 0.96338 |
| `decode_prod` | bf16 | 13.4 | 47 | 2082 / 0.99658 |
| `decode_prod` | f16p | 2.7 | 8 | 2054 / 0.99902 |
| `decode_64k` | m4 | 393.5 | 568 | 2567 / 0.94971 |
| `decode_64k` | bf16 | 54.5 | 75 | 2106 / 0.99365 |
| `decode_64k` | f16p | 8.8 | 17 | 2058 / 0.99854 |

So with fp16 the answer to "how much rank movement" is **about one index in 2048,
never more than five**, and the identities it swaps are by construction at the
boundary, where the two candidates differ by less than one part in 1024.

---

## 4. Perf: the four-regime table

`measured (j-ec4de538)`. Median of `reps=10`, `full ms/call`, single TensorCore,
incumbent and variant back to back in the same job through
`instruments/microbench.run_regime` — the same yardstick and the same
`inputs.build` the step-1 baselines came out of. **Negative `Δ time %` is faster.**

| regime | arm | scores MB | ms | spread % | base ms | `Δ time %` | speedup × |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_prod` | incumbent | 2.6 | 0.16760 | 0.43 | 0.16760 | — | — |
| `decode_prod` | bf16out | 2.6 | 0.17134 | 0.61 | 0.16760 | +2.23 | 0.978 |
| `decode_prod` | cast_bf16_to_f32 | 2.6 | 0.03784 | **18.22** | — | — | — |
| `decode_64k` | incumbent | 1.6 | 0.20059 | 0.76 | 0.20059 | — | — |
| `decode_64k` | bf16out | 1.6 | 0.20242 | 0.70 | 0.20059 | +0.91 | 0.991 |
| `decode_64k` | cast_bf16_to_f32 | 1.6 | 0.03577 | **11.25** | — | — | — |
| `mixed_prod` | incumbent | 41.9 | 0.67109 | 1.93 | 0.67109 | — | — |
| `mixed_prod` | bf16out | 41.9 | 0.70419 | 2.44 | 0.67109 | +4.93 | 0.953 |
| `mixed_prod` | cast_bf16_to_f32 | 41.9 | 0.03684 | 9.74 | — | — | — |
| `mixed_64k` | incumbent | 276.8 | 4.78892 | 1.07 | 4.78892 | — | — |
| `mixed_64k` | bf16out | 276.8 | 4.84624 | 1.18 | 4.78892 | +1.20 | 0.988 |
| `mixed_64k` | cast_bf16_to_f32 | 276.8 | 0.16817 | 1.53 | — | — | — |

The incumbent reproduces the step-1 baselines within their spreads at every regime
(0.16760 vs 0.16701, 0.20059 vs 0.20116, 0.67109 vs 0.66911, 4.78892 vs 4.76284),
which is the check that the harness and the machine are the ones step 1 used.

**`bf16out` is not C and being slower is not C being slower.** Per `mixed_64k` call
relative to the incumbent, C moves **−285 MB** and this arm moves **+285 MB** — the
TC writes 142 instead of 285, but XLA then reads 142 and writes 285, and the
SparseCore still reads 285. They are mirror images, which is what makes the arm
useful: it prices scores traffic from the wrong side, so C's prize can be read off
it without building the packing.

### 4.1 The decomposition, and where it is valid

```
t(bf16out) − t(incumbent) − t(cast)  =  t(TC half write) − t(TC full write)
```

The left side negated is C's TensorCore-side prize. It is only meaningful where the
`cast` arm is actually measuring bytes. It mostly is not:

| regime | cast arm bytes (r+w) | achieved rate | bandwidth-bound? |
| --- | --- | --- | --- |
| `decode_prod` | 3.9 MB | 0.10 TB/s | **no** — dispatch floor |
| `decode_64k` | 2.4 MB | 0.07 TB/s | **no** — dispatch floor |
| `mixed_prod` | 62.9 MB | 1.71 TB/s | partly |
| `mixed_64k` | 415.2 MB | **2.47 TB/s** | **yes** |

At the two decode regimes the cast arm costs ~36 µs whether it moves 2.4 MB or
3.9 MB — that is a fixed per-launch floor in the harness, not traffic, and it is why
its spread is 11–18%. Feeding it into the decomposition there produces a nonsense
"prize" of 17–20% for a 2 MB buffer. **The decomposition is reported only at
`mixed_64k`**, and this is a caveat about the instrument that the data itself
exposed.

At `mixed_64k`: gap `= 4.84624 − 4.78892 = +0.05732 ms`; cast `= 0.16817 ms`;
so the TC-side prize `= 0.16817 − 0.05732 = 0.11085 ms` of a 4.78892 ms kernel
= **2.31% faster, measured (j-ec4de538)**. Combining the two arms' spreads gives
±0.077 ms on that difference, i.e. **2.3% ± 1.6%** — the effect is at the same scale
as the run-to-run spread and cannot be resolved finer by this instrument.

### 4.2 The ceiling — the number that actually decides C

The `mixed_64k` cast arm gives a *measured* price for a scores byte at this shape:
415.2 MB in 0.16817 ms = **2.469 TB/s**. C removes `B` bytes per call, where `B` is
the f32 scores buffer (`B/2` off the TC write, `B/2` off the SparseCore read).
Dividing by the measured rate:

| regime | scores `B` | bytes C removes | time removed | as % of the kernel |
| --- | --- | --- | --- | --- |
| `decode_prod` | 2.6 MB | 2.6 MB | 1.06 µs | **0.63%** |
| `decode_64k` | 1.6 MB | 1.6 MB | 0.66 µs | **0.33%** |
| `mixed_prod` | 41.9 MB | 41.9 MB | 17.0 µs | **2.53%** |
| `mixed_64k` | 276.8 MB | 276.8 MB | 112.1 µs | **2.34%** |

`modelled from a measured rate (j-ec4de538)`. This is a **ceiling**, and it is
generous in three ways: it assumes the SparseCore reads at the TensorCore's copy
bandwidth, it assumes the saved bytes are on the critical path rather than
overlapped with compute, and it charges nothing for the pack/unpack C requires. The
independent `mixed_64k` decomposition (2.3% ± 1.6%) lands on the ceiling, which is
the two routes agreeing rather than either one being precise.

**Even at the regime where C is largest, the scores buffer is only ~4.8% of the
kernel's time by the most generous bandwidth accounting** (570 MB written + read at
2.47 TB/s = 0.23 ms of 4.79 ms), so halving it cannot buy more than ~2.4%. The
mixed band spends ~95% of its time on something that is not scores traffic. Per the
dispatch I note only that this is **consistent** with the VPU-saturation hypothesis
in `measurement_and_design.md` §7 and leave the inference to the caller; I did not
run the strike test and this measurement does not distinguish VPU saturation from
any other non-HBM bound.

---

## 5. Correctness

### 5.1 SparseCore coverage — the precondition

Step 2 recorded that `sparsecore_topk` has **no test anywhere in `tests/`**. That is
now addressed: `sc_topk_test.py` in this folder, `measured (j-9d6276c2)`:

```
28 pass, 0 fail, 0 error
partition factors exercised: p in [1, 2, 4, 8, 32]; at production shapes p in [1, 2, 4]
negative control: ALL CAUGHT
```

It checks six properties per case against a host oracle — C1 shape/dtype, C2 the
`-1` padding is a suffix, C3 every index is legal for its `row_length`, C4 the count
is `min(k, L)`, C5 uniqueness, C6 the multiset of selected scores equals the multiset
of the true top-`min(k, L)` (exact set equality up to ties). 20 small cases plus 8 at
production shapes `(64,10240,2048)`, `(1024,10240,2048)`, `(6,67584,2048)`,
`(1024,67584,2048)`, over distributions `normal / bf16 / f16 / equal / coarse /
signed_zero / large_mag` — the bf16- and f16-quantized distributions are there
specifically because C makes tie multiplicity go up.

The `poison` variant fills the invalid tail past `row_length` with `+1e30` rather
than `−inf`, so a kernel that ignores `row_lengths` is caught rather than accidentally
passing. The negative control corrupts the output five ways — dropped winner,
duplicated index, out-of-range index, truncated count, `row_lengths` ignored — and
all five were reported, so the test is not vacuous.

### 5.2 What this does and does not verify

- **Verified:** `sparsecore_topk` at HEAD, on f32 keys, at production shapes,
  including the denser tie structure a 16-bit key produces. That is real coverage
  for the component C would modify, and it is durable — it stays on the branch
  whether or not C is ever built.
- **Verified:** that narrowing the ranking key to 10 mantissa bits does not move the
  end-to-end index set materially (§2, §3), with a positive control (m4) that fails
  the same gate and a negative control that fires on every arm.
- **NOT verified — and this is the honest gap:** *option C's SparseCore code does not
  exist*, so nothing here tests a 16-bit-key `sparsecore_topk`. `sc_topk_test.py`
  tests today's f32 kernel. The rank/recall gate runs the real SparseCore kernel but
  hands it f32 inputs whose values have been rounded — it proves the *ranking* is
  fine, it proves nothing about a packed-key load path, the `monotone_key` transform
  on a 16-bit field, or `KEY_NEG_INF` under packing. The TensorCore-side gate does
  not stand in for SparseCore coverage of code that has not been written.
- **NOT verified:** the `bf16out` arm's correctness on device was not separately
  gated; it is covered by the bf16 row of §2 only in the sense that it rounds to the
  same 7 mantissa bits. It is a price probe and is not proposed for merge.

---

## 6. What was not built, and why

Full option C — TensorCore packs two 16-bit keys per i32 word, SparseCore unpacks in
`_topk_kernel`, and the returned indices are remapped — is ~300 lines across
`streamindex_topk.py` and `sparsecore_topk.py`, with genuine Mosaic and plsc risk.
It was designed on paper and then not built, because the measurement above caps its
prize at 2.3–2.5% in the mixed band and under 1% in decode, for a rank-4-of-5
candidate. Recording the design so the next session does not re-derive it:

- **Block-half packing.** Pair sublane `s0 + r` with `s0 + S_b/2 + r` within one
  TensorCore block, unpacking into resident sublanes `2p` / `2p+1`. Keeping the
  permutation *inside* a block is what preserves the `i_hi` prefix bound in
  `_topk_kernel` (valid if rounded up to a block boundary); a global interleave
  would break it.
- **`bf16 → f32` is exactly `<< 16`**, so `KEY_NEG_INF` and all three
  `REFINEMENT_LEVELS` are untouched by the widen; only the load changes. A 10-bit
  precision-only key (§2) has the same property — it is an f32 with the low mantissa
  bits zeroed, so `monotone_key` is unchanged.
- **Stage 2 of `_pick_partition`** can keep the existing f32 kernel unmodified; only
  the first-pass DMA-and-bucket path needs the unpack.
- The pack itself is **VPU work added to the TensorCore inner loop**, which is
  unmeasured here and is not in the §4.2 ceiling. If the mixed band is VPU-bound,
  this cost lands exactly where there is no headroom — so the realizable number is
  below the ceiling by an unknown amount, and plausibly below zero at
  `mixed_prod`, where the measured TC-side saving was already 0.56% ± 3%.

---

## 7. Caveats that survive

1. **Every number here is a single-chip microbenchmark on one TensorCore, not an
   end-to-end number.** The scoring kernel's share of a GLM-5.2 step is not
   established in this document; a 2.3% kernel win is 2.3% of whatever that share is.
2. The `Δ time %` at `decode_64k` (+0.91%) and `mixed_64k` (+1.20%) sit inside or
   beside the 0.70–1.18% spreads on those rows. Those rows cannot support a
   conclusion about the arm's sign on their own; they are used only through the
   decomposition, and only at `mixed_64k`, where the cast term is the large and
   well-resolved one (1.53% spread).
3. The `cast_bf16_to_f32` arm is invalid as a bytes measurement at both decode
   regimes (§4.1) and only partly valid at `mixed_prod` (1.71 TB/s).
4. The score-range finding (zero fp16 overflow, zero subnormals) is a property of
   `inputs.build`'s synthetic `standard_normal` inputs. It is why the recommended
   format keeps the f32 exponent rather than adopting IEEE fp16 outright.
5. The rank/recall gate uses the same `RANK_SLACK=256` / `MIN_RECALL=0.99` criterion
   as the incumbent's own gate. Whether that criterion is the right bar for
   production DSA quality is a question this folder does not answer.
6. No end-to-end accuracy evaluation was run on any arm. `⚠️ unverified` in the
   end-to-end sense.

---

## 8. Files

| file | what it is |
| --- | --- |
| `keyfmt_probe.py` | the accuracy gate: `margins` (real kernel + f64 oracle, per key format), `churn` (index-set difference between formats), `dist` (range, fp16 reach, tie multiplicity). Includes the sentinel patch-mechanism check and the m4 positive control. |
| `sc_topk_test.py` | SparseCore coverage for `sparsecore_topk` — the carried item. 28 cases, six properties, five-way negative control. |
| `_load.py` | loads a patched copy of the installed `streamindex_topk.py`, asserting each anchor matches exactly once so a drift in the incumbent is loud. |
| `impl_bf16out.py` | option C's TensorCore half, as a one-line source patch. A price probe, not a candidate. |
| `bench_ab.py` | the A/B driver. Uses `instruments/microbench.run_regime` as a library, never forked. |
| `half_width_key.diff` | the diff of the arm that was measured, with a header saying what it is not. |

Two false-negative traps had to be closed before the gate produced a real number,
and both are now guarded by controls that fail loudly rather than by care:

- **The jaxpr memo.** `f.clear_cache()` on a jitted function clears pjit's fastpath
  but not the trace memo in `jax._src.linear_util`, which is keyed on the wrapped
  function object and has no idea a module global was rebound. Job `j-9d6276c2`'s
  keyfmt arms are byte-identical across every format for this reason and are
  **invalid**. Fix: `jax.clear_caches()`, plus a `-1`-returning sentinel patch that
  must make the kernel return all `-1` or the probe exits.
- **Excess precision.** `--xla_allow_excess_precision` defaults to **true**, so XLA
  is explicitly permitted to delete a `convert(f32→bf16) → convert(bf16→f32)` round
  trip, and it does. Job `j-cb875cb9`'s keyfmt arms are **invalid** for this reason
  even though its sentinel passed. Fix: round-half-to-even integer bit-masking,
  which the algebraic simplifier cannot see through and which is closer to what a
  shipped kernel emits anyway; the IEEE arm keeps `astype` inside an
  `optimization_barrier`. The m4 arm exists as a positive control and the probe exits
  if it fails to move any indices.
