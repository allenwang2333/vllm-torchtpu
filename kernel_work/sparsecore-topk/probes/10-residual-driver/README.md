# Probe 10 — what drives the 75 µs residual: input bytes, output bytes, or rows?

**Job** `j-770c6de9-2044-4216-bcdd` · Succeeded · image `…vllm-torchtpu-hning_google_com:fa63d6df` ·
one job, no fan-out · harness `glm_dsa_block_bench.py` md5 `08375d7068365e1a776c58fef09cd6d0`
(frozen, verified in-job) · probe md5 `a7724e68ee050d2cec2432c93f54fc67` · subject md5
`44353b347c96e0eedbd742c710d4540a` (in-job `MAP_GATE_OK`) · recipe read-back md5
`4aaab2b5ac1c8294ae563ee55b9caf5c`, byte-identical to what was sent, `bash -n` clean both sides.

Verbatim log: `j-770c6de9-2044-4216-bcdd.txt` (132 lines). Arms: `arms/` (8 files). Results:
`residual_driver.json`.

## The question

`probes/09` de-contaminated the unattributed residual to **75.683 µs — 39.8% of the production
prefill kernel — and showed it is flat in `num_waves` (0.009 µs/wave)**, refuting the per-wave
barrier framing that had stood since probe 08. But it also grows **7.6× from b=64 to b=1024**, a 16×
change in `b`. Something scales, and it is not the wave loop.

`b` moves three things at once — input bytes `b·n·4`, output bytes `b·k·4`, and the row count — so
probe 10 varies `n`, `k` and `b` **independently** on the `A0` arm (every kernel region dropped:
`dma,cvt,hist,sweep,emit,fill,outdma`) at a fixed patch, and fits all three axes.

The three answers route differently, which is why it was worth a job:

| if the residual tracks… | what it is | fix route |
| --- | --- | --- |
| input bytes `b·n·4` | something touching the score buffer outside the kernel | not reachable from inside the Pallas kernel |
| output bytes `b·k·4` | allocation or a copy on `out_hbm` | `input_output_aliases` + `donate_argnames` — one line |
| rows `b` | per-row setup — D11's `b_pad`-wide length DMA is the candidate | inside the kernel |

## Arms

Seven `A0` points plus a `B` control, all `P=1`, single-stage, `k` legal — **no point was excluded
and no point changes regime**, so all three fits sit inside one regime.

| arm | b | n | k | in MiB | out MiB | waves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_base` | 1024 | 10240 | 2048 | 40.0 | 8.0 | 32 |
| `A0_base` | 1024 | 10240 | 2048 | 40.0 | 8.0 | 32 |
| `A0_n/4` | 1024 | 2560 | 2048 | 10.0 | 8.0 | 32 |
| `A0_n/2` | 1024 | 5120 | 2048 | 20.0 | 8.0 | 32 |
| `A0_k/4` | 1024 | 10240 | 512 | 40.0 | 2.0 | 32 |
| `A0_k/2` | 1024 | 10240 | 1024 | 40.0 | 4.0 | 32 |
| `A0_b/4` | 256 | 10240 | 2048 | 10.0 | 2.0 | 8 |
| `A0_b/16` | 64 | 10240 | 2048 | 2.5 | 0.5 | 2 |

**Gates.** Anchor gate 13/13 `count==1`; compile gate 8/8 with `changed_lines_vs_B` printed per arm;
shape gate printing the derived `P`, `n_p`, `slice_len`, `two_stage` and `waves` per point before any
timing; in-job map gate. The prober added a gate the spec did not ask for — **`P09_SRC_GATE`,
asserting the patched sources reproduce probe 09's `fd46ef14…` (`B`) and `547031a2…` (`A0`) bit for
bit**, locally and in-job. "Reused verbatim" is therefore a checked fact here, not a claim. The whole
reporting/fit/JSON path was also exercised on CPU with stubbed kernels and an injected known driver,
which the fits recovered exactly.

## The answer: the residual is **buffer bytes**, and there is **no per-row term**

**The contrast that needs no fitting.** `A0_n/4` and `A0_b/4` carry **identical input bytes**
(10 MiB) and differ only in output bytes (8 → 2 MiB) and in `b` (1024 → 256). Measured delta
**6.028 µs**; the output-byte slope alone predicts 6.708 µs. Leftover for a **4× change in b**:
**−0.680 µs = −0.886 ns/row**, inside the 0.770 µs null spread. **A per-row mechanism is excluded**,
independently of every fit, and consistent with probe 09's 0.009 µs/wave.

⚠ **The b-axis fit's 71.4 ns/row at R² 0.992 is an artifact, and it is the trap this probe was built
to avoid.** Scaling `b` scales *both* byte counts by the same factor, so that "slope" is the byte
slopes wearing a different label. The bytes-only model leaves **+2.290** and **+10.410 µs** at the
two `b` points — leftovers of the **wrong sign** for a per-row cost.

| axis | slope | intercept | R² | reading |
| --- | ---: | ---: | ---: | --- |
| input MiB | 1.8246 µs/MiB (574.7 GB/s) | 0.754 | 0.97718 | ⚠ **a chord, not a rate — the axis is convex** |
| output MiB | 1.1180 µs/MiB (937.9 GB/s) | 66.424 | 0.99978 | genuinely linear |
| b (rows) | 0.0714 µs/row | 1.505 | 0.99182 | ⛔ **CONFOUNDED** — see above |

⚠ **The input axis is convex and its single slope hides a 2.1× change in rate.** Segment slopes:
**1.020 µs/MiB** over 10→20 MiB, **2.147 µs/MiB** over 20→40 MiB. That is why extrapolating the byte
model down to `A0_b/16`'s 2.5 MiB yields a nonsense −1.46 µs. The output axis, by contrast, is
linear across its whole range (1.166 then 1.099 µs/MiB). **Quote the segment, not the fit.**

## The mechanism, from the optimized HLO — and it is *outside* the Pallas kernel

The prober dumped the optimized HLO of `A0_base` as a best-effort diagnostic. Inside the timed `jit`,
before the custom call:

```
ENTRY %main.2 (scores.1: f32[1024,10240], row_lengths.1: s32[1024]) -> s32[1024,2048]
  %bitcast-convert_bitcast_fusion = s32[128,80,8,128]{3,2,1,0:T(8,128)} fusion(%scores.1), kind=kLoop
  %copy.3                         = s32[128,80,8,128]{3,1,2,0:T(8,128)} copy(%param_0.6)
  … sparse-core-data-format-call …
  %bitcast.3                      = s32[10485760]{0:T(1024)} bitcast(…call-done)
  ROOT %sc_topk_direct_b1024_n10240_k2048.6 = s32[2097152]{0:T(1024)} custom-call(…)
```

`%copy.3` changes layout `{3,2,1,0:T(8,128)}` → `{3,1,2,0:T(8,128)}` on the **whole 40 MiB score
buffer** — a full read plus a full write. 40 MiB × 1.82 µs/MiB ≈ **73 of the 75.3 µs**, at ~1.1 TB/s
across the 80 MiB of traffic.

⚠ **`A0` deletes the score DMA — the kernel never reads `scores_hbm` — and XLA reformats the buffer
anyway.** It is an operand of the custom call, so it must be materialised whether or not the kernel
body touches it. That is exactly why the cost survives a total kernel blackout, and it is the
mechanism the four refuted candidates in `probes/09` were all reaching for.

**This is an HLO reading, not a measurement.** It is `modelled`, it is best-effort, and it does not
yet establish that the same copy exists in the production call path rather than only under this
harness's `jit` signature. `probes/11-relayout-provenance` is dispatched to settle exactly that.

## Consistency — every cross-job control holds

| quantity | probe 10 | prior | Δ |
| --- | ---: | ---: | ---: |
| `B_base` | 326.732 | 326.782 (`probes/09`) | **−0.02%** |
| `B_base` | 326.732 | 327.166 (`probes/08`) | −0.13% |
| `A0_base` | 92.118 | 92.231 (`probes/09`) | −0.12% |
| `A0_base` residual | 75.349 | 75.683 (`probes/09`) | −0.44% |
| `A0_b/16` residual vs probe 09's own b=64 floor | 9.973 | 9.949 | **+0.24%** |

Three arms exceed the harness's own 5% host-bound limit — `A0_n/4` 5.9%, `A0_b/4` 9.1%, `A0_b/16`
**18.8%** — and `measure()` does not warn on it (the limit is only wired into the harness's `sweep()`
printer). `us_median` is a paired slope and still removes fixed dispatch, but at `A0_b/16` the naive
and slope estimates differ by 13%, so **8.950 is the softest number in the table**. It is also the
one that reproduced probe 09 to +0.24%.

## Caveats

1. ⚠ **The HLO block is the weakest evidence here and carries the strongest conclusion.** Everything
   above it is a timing fit; the mechanism is a static read of one arm's optimized module. Treat
   "the residual is buffer bytes" as measured and "the buffer bytes are an XLA relayout" as
   `modelled` until `probes/11` returns.
2. ⚠ **The input axis is convex.** Any model that uses 1.8246 µs/MiB as a rate is wrong at both ends.
3. **One pool, not one per `b`** — a spec deviation, and the right one: the question *is* the
   cross-shape comparison, and probe 09's per-`b` loop would have put b=1024 and b=64 under different
   drifts.
4. **Three nulls, at b=1024 only.** The harness null is `jax.jit(lambda a: a+1.0)` on `(8,128)` and
   does not depend on `b`, so a per-`b` null is a repeat sample, not a distinct quantity. A constant
   floor moves every intercept and no slope, so it cannot manufacture a driver. Probe 09's own
   b=1024-vs-b=64 floors differ by 0.801 against this job's 0.770 intra-pool spread — no evidence of
   `b`-dependence.
5. **The output-byte term, 1.118 µs/MiB ≈ 8.9 µs at the base shape, is unexplained.** The output is
   `s32[2097152]{0:T(1024)}` bitcast to `s32[1024,2048]`, and a bitcast is free. Carried to
   `probes/11` Q3.
6. The only `Error` string in the log is line 5, the `TPU_WORKER_HOSTNAMES` libtpu boilerplate that
   appears identically in probes 08 and 09. Artifact, not a failure. No `!!`, no `suspect=True`, no
   gate failure.

## Decided

1. **The residual is buffer bytes. There is no per-row term** — excluded by an unconfounded contrast
   at 0.886 ns/row against a 0.770 µs floor, not only by a fit.
2. **D11 (the `b_pad`-wide length DMA) stays `assumed` negligible and is *not* promoted.** It was the
   per-row candidate and the per-row axis is empty.
3. ⛔ **`probes/09`'s floor bracket collapses toward its optimistic leg.** The 75.683 µs is **not** an
   irreducible cost of invoking this op — it scales with buffers, so a version that changes what is
   handed to the kernel can reach it. The 95.73 µs "effective floor" leg of `design.md` §1 is
   withdrawn pending `probes/11`.
4. ⚠ **The mechanism, if `probes/11` confirms it, is not a Pallas change.** It is a layout or dtype
   decision at the call site. It would be a `carried:` handoff worth ~73 µs — **4× the caller's own
   double-buffering request** — and it would also mean the microbenchmark's 189.98 µs denominator
   overstates the production kernel, roughly doubling every share in the balance sheet.
5. **`A0_base`'s residual is 75.349 here vs 75.683 in probe 09.** Use whichever job a comparison is
   made within; they agree to 0.44% and neither is more correct.
6. **Never quote 71.4 ns/row.** It is in the log, it has R² 0.992, and it is an artifact of `b`
   scaling both byte axes together.
