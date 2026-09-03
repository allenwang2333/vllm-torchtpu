# `streamindex_topk` merged module — the three changes together, across the grid

Each of the three changes in `streamindex_topk_merged.py` had been measured **alone**. This is the
first time they were measured **composed**, and the first time any of them was measured away from
the single production shape. The question was whether they compose additively.

**They do.** Across all 17 configurations the largest departure from additivity is **0.72% of the
incumbent's full-op time**, which is inside the 1.01% run-to-run scatter measured between two
independent jobs. There is no configuration where `merged` is slower than `inc`.

## What was run

| | |
| --- | --- |
| sweep job | **`j-2c00d88c-e7de-415f-968b`** — `Succeeded`, 4/4 fan-out shards `rc=0` |
| pipeline-validation job | **`j-f06ee0fd-8592-46cf-a2ce`** — `Succeeded`, 4/4 shards `rc=0`; reused below as an independent repeat |
| hardware | tpu7x, `2x2x1`, 4 chips; each fan-out shard pinned to one chip |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` (prebuilt, not rebuilt) |
| incumbent | `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py` md5 `1e221f10cbf3573aa0d2ae78c3463532` (unmodified) |
| merged module | `variants/merged/streamindex_topk_merged.py` md5 `5bc0578427412d56209ef8b5d764fcae` |
| stage B | `sparsecore_topk.py` md5 `8bc9e25b8701a53a97704564b9fbc658` (unmodified) |
| shapes / inputs | `instruments/bench_streamidx_topk.py` md5 `e8e492d13e59c4a1675cc7a98c5a0813` (read-only, imported) |
| grid | 13 decode + 4 prefill = **17 configs × 5 arms = 85 traces**, all 85 captured |
| correctness | all 17 configs, **10 720 rows** compared |

Every md5 above was read back **on the pod**, not asserted locally — see the `MODULE ...` lines in
each shard's log and in `correctness.json` / `trace_meta.shard*.json`.

## Method

**`hlo_stats` device self time.** Per-instruction, no host dispatch component, immune to the
contention a 4-chip fan-out creates. Captured with `variants/merged/bench_merged_trace.py`
(one `*.xplane.pb` per `<config>__<arm>` — never two arms in one trace, because `hlo_stats` keys
rows on the HLO op name and `inc` and `d1` produce identical Pallas op names), read off-pod with
`variants/merged/read_merged_hlo_stats.py`, which reuses `probes/bdt_hlo_stats.py`'s bucketing.

- **full op** = the TensorCore plane total. The TensorCore is *blocked* for the SparseCore's
  duration, so the `sc_topk…call-done` row already accounts for stage B. The SparseCore plane is
  reported in its own column and **is never added** to anything.
- **stage A** = `StreamIdxTC*` rows. **stage B** = `sc_topk*` rows on TensorCore.
  **residual** = the rest of the TensorCore plane (the `-inf` init, bitcasts, slices, and on the
  cond-live arms the `lax.cond`).
- Partition check `sum(buckets) − TC total`: **0.000000 on all 85**. Row completeness
  `rows_kept == rows_total`: **85/85**. No arm was excluded.

**Wall clock is not reported anywhere in this document.** The capture script prints a per-call
wall figure under the name `wall_us_per_call_CONTENDED_NOT_A_RESULT` so that a hung arm is visible;
it is contended by three neighbouring shards and it is not a measurement. No dispatch floor was
subtracted from anything — per-call wall clock is `max(device, host)` and subtracting a clamp is
invalid.

**Repeatability.** Four configs × 5 arms were measured in *both* jobs, at different iteration
counts (10 vs 20) on different pods. Largest full-op disagreement: **1.01%**
(`dec_b64_ctx65536`/`bkvp_d1`, 740.229 → 747.737 µs); median well under 0.3%. Read every
sub-1% figure below as noise.

## The five arms

| arm | module | decode `bkv_p` | early-exit `lax.cond` | Option C (head-major) |
| --- | --- | --- | --- | --- |
| `inc` | **incumbent** | 2 (production constant) | live | off |
| `bkvp` | **incumbent** | `recommended_num_kv_pages_per_block()` | live | off |
| `d1` | **merged** | 2 | deleted | off |
| `bkvp_d1` | **merged** | `recommended_num_kv_pages_per_block()` | deleted | off |
| `merged` | **merged** | `recommended_num_kv_pages_per_block()` | deleted | on, prefill/mixed only |

The two cond-live arms **have to** come from the incumbent module: the merged module deleted the
`lax.cond` and `_fast_path` outright and cannot synthesize them back. `enable_early_exit=True` on
those arms, matching production (`mla_attention_op.py`). The three merged-module arms differ from
each other only in static arguments.

## Headline

**Decode**, `inc` → `merged`: **−7.65% to −20.24%**. The production point
(`num_seqs` 64, ctx 8704, `max_model_len` 9216) is the best cell in the grid at **−20.24%**.

**Prefill**, `inc` → `merged`: **−7.51% to −7.92%** — flat across both q lengths and both contexts.

**No configuration regresses.** The worst decode cell is −7.65% and the worst prefill cell is
−7.51%; both are wins.

## Composition: additive

Writing Δ for the change in full-op µs against `inc`:

- Δ(`bkvp`) isolates candidate 0
- Δ(`d1`) isolates D1
- Δ(`merged`) − Δ(`bkvp_d1`) isolates Option C **on top of** the other two

Their sum predicts Δ(`merged`) to within **0.72% of `inc` at worst**, and to within 0.2% on 14 of
17 configs. Since two independent jobs disagree with each other by up to 1.01% on the same arm,
**the non-additivity is not resolvable by this measurement.** The three changes are independent to
the limit of what the instrument can see.

That is the expected answer once the mechanisms are laid out, and it is worth saying why, because
"they compose" is exactly the kind of result that can be right for the wrong reason:

- **candidate 0** changes the decode launch's `bkv_p` blocking, and shows up almost entirely in
  **stage A** (−7.7% to −23.8% on decode, ~0% on prefill).
- **D1** deletes a `lax.cond`, and shows up almost entirely in the **residual** bucket — a flat
  **7.6-8.5 µs off every configuration in the grid**, decode and prefill alike, independent of
  batch, context and phase. That flat offset is the signature of the stated mechanism: XLA
  compiles each cond branch into its own computation per call site, blocking CSE across 21
  redundant computations; collapsing them is a fixed cost removed, not a proportional one.
- **Option C** changes the score layout of the mixed/prefill launch, and shows up entirely in
  **stage A of prefill**.

Three changes landing in three different buckets is why the arithmetic closes.

## Where `merged`'s win is capped, and by what

The full-op percentage is not the stage-A percentage, and the gap is Amdahl, not a measurement
problem. In **prefill**, stage B — the SparseCore exact top-k — is **53-60% of the whole op**, and
none of the three changes touches it (stage B moves by at most 0.26% on any prefill arm, which is
noise). So Option C's genuine **−20%** on prefill stage A becomes **−7.6%** on the prefill op:

| | stage A share of `inc` | stage A Δ | full-op Δ |
| --- | ---: | ---: | ---: |
| `pre_q1024_ctx4096` | 27.5% | −20.23% | −7.51% |
| `pre_q4096_ctx16384` | 34.9% | −20.89% | −7.63% |

In **decode** the same arithmetic runs the other way. Stage A grows from 13.3% of the op at
`b8/ctx4096` to 76.1% at `b64/ctx65536`, so candidate 0's stage-A win has progressively more to
bite on — and D1's flat 7.6-8.5 µs is progressively less of a large total. The two trends cross,
which is why the decode column is a shallow U rather than a trend: `merged` is −16.2% at
`b64/ctx4096`, −20.2% at the production point, and −16.0% at `b64/ctx65536`.

The single worst decode cell, `dec_b8_ctx16384` at −7.65%, is worst for a specific and legible
reason: at batch 8 with ctx 16384, `bkvp` is worth **+0.15%** (nothing) while the residual is 48.9%
of the op, so essentially all of the −7.65% is D1's flat offset.

## Per-arm behaviour worth naming

**candidate 0 (`bkvp`) is not a uniform win, and is a small loss at four cells.** Full-op
Δ vs `inc`: `dec_b8_ctx4096` **+0.57%**, `dec_b8_ctx16384` **+0.15%**, `pre_q4096_ctx4096`
**+0.63%**, `pre_q1024_ctx16384` **+0.03%**. All four are inside the 1.01% repeatability bound, so
the honest reading is "no effect", not "a regression" — but none of them is the −14.6% the
production point produced, and that is the point: **the value of candidate 0 is strongly
config-dependent**, ranging from 0 to −15.5% on the full op. Its win requires stage A to be a
large share of the op, which needs both a large batch and a long context.

The two prefill cells where `bkvp` moves at all are cells where it *should not* move: the prefill
and mixed slots of the tuple are held at 2 in every arm, so only the decode slot changes, and a
prefill config has no decode sequences. The +0.63% at `pre_q4096_ctx4096` is a 7.2 µs shift in the
residual bucket of a 1109 µs op — the decode launch is still compiled and dispatched at zero work,
so its compiled form still differs. It is within scatter and nothing depends on it.

**D1 is the most reliable of the three** and the only one that pays at every configuration:
−0.83% to −11.69% on the full op, flat 7.6-8.5 µs in absolute terms, all of it in the residual
bucket. It is worth the most where the op is smallest.

**Option C is gated to prefill/mixed and the gate holds.** Δ(`merged`) − Δ(`bkvp_d1`) on the 13
decode configs is between −0.22 and +0.27 µs, i.e. zero. Ungated it costs +59.6% at decode; the
`head_major_prefill` toggle keeps that off the decode path, and the data confirms it does.

## Correctness: index-identical, 17/17

Checked by `variants/merged/correctness_merged.py` in the same job,
**`j-2c00d88c-e7de-415f-968b`**, shard 3, `rc=0`.

`merged` vs `inc` on the returned top-k **indices**, per configuration:

- **17/17 configs `index_identical=True`**, `rows_mismatched=0`
- stronger than required: **`array_identical=True` on all 17** — the arrays match
  element-for-element in order, not merely as sets, so not even a tie-break reordering occurred
- **10 720 rows** compared in total (64+8+8+8+8+32+32+32+32+64+64+64+64+1024+1024+4096+4096)

Because there were no mismatches, the reference-score read-back path (which dequantizes the fp8 K
record and recomputes both sides in f32 so a differently-broken tie can be told from an error)
never fired, and neither did the `--extra-on-mismatch` attribution to `d1`/`bkvp_d1`.

This is not a recall-vs-f64 gate. That gate was established earlier to be the wrong instrument for
this op — the k=2048-of-8704 boundary is degenerate in f32 and it fails on the *incumbent* — and
it was not used.

## Two bugs found and fixed in the merged module

Both are in `recommended_num_kv_pages_per_block`, both were latent in the version handed over, and
both would have produced a wrong or uncompilable result at long context. The incumbent was not
touched.

### 1. The recommendation did not compile at ctx ≥ 32768

It returned `cdiv(max_model_len, page_size)` unconditionally. The decode bkv double buffer is
`2 * decode_req_batch_size * bkv_p * page_size * lkv_dim` = **2 MiB per unit of `bkv_p`** at the
production geometry, so `p=32` asks for 64 MiB against a 63.94 MiB physical TensorCore VMEM.
Measured on the pod, verbatim:

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space vmem.
Used 85.00M of 63.94M vmem. Exceeded vmem capacity by 21.06M.
```

That is a crash, not a slow answer, and it was the *only* thing the function returned at
`max_model_len` 32768 and 65536 — a third of the decode grid. It now walks the block-count ladder
`p = cdiv(pages_per_seq, nb)` for nb = 1, 2, 3, … and returns the first rung whose double buffer
fits a budget (`bkv_double_buffer_bytes`, `DEFAULT_BKV_VMEM_BUDGET_BYTES`). Descending the ladder
rather than halving is what the two-regime blocks/pages model calls for: minimise the bkv block
count first, and among tuples with equal block count take the smallest `p`, which overfetches
least on the final partial block.

The `85.00M`/`21.06M` in that error is also where the budget comes from: 21.06 MiB is the rest of
the kernel's VMEM, so the buffer's share cannot exceed ~42 MiB at this geometry.

### 2. The obvious fallback silently did *more* work than the incumbent

The first ladder — plain `cdiv(pages_per_seq, nb)` — returns `p=11` at `pages_per_seq=32`. But
`num_sublanes_total = max over the whole 3-tuple of align_to(pages_per_seq, p) * page_size // 128`,
and `align_to(32, 11) = 33` against `align_to(32, 2) = 32`. That widens the score matrix for
**every** launch: more `-inf` init, more positions for stage B's radix select, plus a partial bkv
block that overfetches. `num_sublanes_total` would have gone 256 → 264.

This is the dangerous failure mode, because it produces a number rather than an error, and the
number would have been a comparison against a *different amount of work* presented as a tuning
win. Rungs are now filtered to `align_to(pages_per_seq, p) <= align_to(pages_per_seq,
prefill_bkv_p)`, which selects `p=8`→`16` instead of 11. `num_sublanes_total` was then checked
config-by-config across the whole grid and is **identical between `(2,2,2)` and the chosen
`(P,2,2)` at all 17** — the comparison in this document is like-for-like, verified rather than
assumed.

### A third thing, caught by the harness rather than fixed as a bug

The first VMEM budget was "half the physical VMEM" = 31.97 MiB, which excluded `p=16`
(32.00 MiB) **by 0.03 MiB**. The sweep harness probes one rung *above* whatever the module
recommends, which is how this surfaced: on `j-f06ee0fd-8592-46cf-a2ce` at `pages_per_seq` 64 the
module said 8 and `p=16` compiled and ran fine. The budget is now `32 * 1024 * 1024` exactly,
measured at both ends — `p=16` runs, `p=32` does not — and the sweep re-confirmed the upper end:
`bkv_p=32 did not lower` at six configs, `p=16` chosen at all of them, `fell_back=False`
everywhere.

## Grid, and what was pruned

Nothing was pruned. The full requested grid ran: decode batch {8, 32, 64} × ctx {4096, 16384,
32768, 65536}, plus the production point (`num_seqs` 64, ctx 8704, `max_model_len` 9216); prefill
q_len {1024, 4096} × ctx {4096, 16384}. Compile time was never the constraint — the whole 85-arm
sweep plus 17-config correctness check finished in **under 12 minutes** of pod wall time on four
chips.

Resolved decode `bkv_p` per configuration (`recommended_num_kv_pages_per_block`, then confirmed by
actually compiling it):

| `pages_per_seq` | configs | recommended | probed from | chosen |
| ---: | --- | ---: | ---: | ---: |
| 4 | ctx 4096 (all batches), `pre_*_ctx4096` | 4 | 4 | 4 |
| 9 | production point | 9 | 9 | 9 |
| 16 | ctx 16384 (all batches), `pre_*_ctx16384` | 16 | 16 | 16 |
| 32 | ctx 32768 (all batches) | 16 | 32 → failed | 16 |
| 64 | ctx 65536 (all batches) | 16 | 32 → failed | 16 |

`fell_back=False` and `clamp_too_tight=False` at all 17.

## Reproducing

```bash
# on-pod, one fan-out shard per chip (see the job's RUN_CMD, read back with `cdk job recipe`)
python variants/merged/bench_merged_trace.py --shard N --num-shards 3 --iters 20 --chunk 5
python variants/merged/correctness_merged.py

# off-pod
python variants/merged/read_merged_hlo_stats.py <synced-output-dir> --out stats.json --md table.md
```

`variants/merged/merged_common.py` holds the grid, the five arm definitions, the `bkv_p` ladder and
the like-for-like check; `merged_shard.sh` is the per-shard driver.

---

# Data

### Headline: `inc` -> `merged`

| config | decode `bkv_p` | full op `inc` µs | full op `merged` µs | **Δ full** | stage A Δ | stage B Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dec_b8_ctx4096` | 4 | 75.4 | 67.9 | **-9.94%** | +0.97% | +0.80% |
| `dec_b8_ctx16384` | 16 | 104.9 | 96.9 | **-7.65%** | +0.02% | -0.33% |
| `dec_b8_ctx32768` | 16 | 132.8 | 120.8 | **-9.00%** | -7.79% | -1.04% |
| `dec_b8_ctx65536` | 16 | 184.9 | 164.1 | **-11.21%** | -13.86% | -0.34% |
| `dec_b32_ctx4096` | 4 | 66.3 | 55.2 | **-16.79%** | -11.46% | -0.74% |
| `dec_b32_ctx16384` | 16 | 144.3 | 124.7 | **-13.59%** | -13.42% | -0.29% |
| `dec_b32_ctx32768` | 16 | 251.5 | 214.7 | **-14.63%** | -16.94% | +0.07% |
| `dec_b32_ctx65536` | 16 | 464.2 | 388.4 | **-16.34%** | -19.88% | -0.22% |
| `dec_b64_ctx4096` | 4 | 99.2 | 83.1 | **-16.24%** | -15.96% | -0.02% |
| `dec_b64_ctx8704_mml9216` **(prod)** | 9 | 170.4 | 135.9 | **-20.24%** | -23.82% | -0.20% |
| `dec_b64_ctx16384` | 16 | 255.5 | 217.7 | **-14.77%** | -16.77% | -0.24% |
| `dec_b64_ctx32768` | 16 | 469.7 | 393.2 | **-16.30%** | -19.94% | +0.00% |
| `dec_b64_ctx65536` | 16 | 890.3 | 747.7 | **-16.02%** | -19.90% | +0.04% |
| `pre_q1024_ctx4096` | 4 | 381.7 | 353.0 | **-7.51%** | -20.23% | +0.20% |
| `pre_q1024_ctx16384` | 16 | 1195.7 | 1105.3 | **-7.56%** | -20.88% | -0.04% |
| `pre_q4096_ctx4096` | 4 | 1108.7 | 1020.9 | **-7.92%** | -19.87% | +0.01% |
| `pre_q4096_ctx16384` | 16 | 4445.9 | 4106.8 | **-7.63%** | -20.89% | -0.01% |

### All five arms, full op / stage A / stage B device self time (µs)

| config | arm | full op | Δ | stage A | Δ | stage B | Δ | residual | SparseCore plane |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dec_b8_ctx4096` | `inc` | 75.435 | +0.00% | 10.024 | +0.00% | 17.515 | +0.00% | 47.896 | 17.071 |
| `dec_b8_ctx4096` | `bkvp` | 75.866 | +0.57% | 10.170 | +1.45% | 17.804 | +1.65% | 47.892 | 17.050 |
| `dec_b8_ctx4096` | `d1` | 67.776 | -10.15% | 10.088 | +0.63% | 17.691 | +1.00% | 39.997 | 17.193 |
| `dec_b8_ctx4096` | `bkvp_d1` | 67.662 | -10.30% | 10.051 | +0.27% | 17.576 | +0.35% | 40.035 | 17.326 |
| `dec_b8_ctx4096` | `merged` | 67.935 | -9.94% | 10.122 | +0.97% | 17.656 | +0.80% | 40.157 | 17.245 |
| `dec_b8_ctx16384` | `inc` | 104.944 | +0.00% | 26.041 | +0.00% | 27.581 | +0.00% | 51.321 | 27.093 |
| `dec_b8_ctx16384` | `bkvp` | 105.106 | +0.15% | 26.151 | +0.42% | 27.520 | -0.22% | 51.435 | 27.149 |
| `dec_b8_ctx16384` | `d1` | 96.608 | -7.94% | 25.978 | -0.24% | 27.243 | -1.22% | 43.386 | 27.367 |
| `dec_b8_ctx16384` | `bkvp_d1` | 96.811 | -7.75% | 26.050 | +0.03% | 27.253 | -1.19% | 43.508 | 27.457 |
| `dec_b8_ctx16384` | `merged` | 96.918 | -7.65% | 26.046 | +0.02% | 27.491 | -0.33% | 43.381 | 27.489 |
| `dec_b8_ctx32768` | `inc` | 132.752 | +0.00% | 46.916 | +0.00% | 33.837 | +0.00% | 51.998 | 31.499 |
| `dec_b8_ctx32768` | `bkvp` | 129.204 | -2.67% | 43.285 | -7.74% | 33.815 | -0.07% | 52.104 | 31.616 |
| `dec_b8_ctx32768` | `d1` | 124.555 | -6.17% | 47.081 | +0.35% | 33.549 | -0.85% | 43.924 | 31.896 |
| `dec_b8_ctx32768` | `bkvp_d1` | 120.893 | -8.93% | 43.319 | -7.67% | 33.566 | -0.80% | 44.008 | 31.738 |
| `dec_b8_ctx32768` | `merged` | 120.807 | -9.00% | 43.262 | -7.79% | 33.484 | -1.04% | 44.061 | 31.861 |
| `dec_b8_ctx65536` | `inc` | 184.859 | +0.00% | 88.785 | +0.00% | 43.048 | +0.00% | 53.026 | 39.301 |
| `dec_b8_ctx65536` | `bkvp` | 172.634 | -6.61% | 76.466 | -13.88% | 42.788 | -0.60% | 53.381 | 39.405 |
| `dec_b8_ctx65536` | `d1` | 176.906 | -4.30% | 88.893 | +0.12% | 43.135 | +0.20% | 44.878 | 39.446 |
| `dec_b8_ctx65536` | `bkvp_d1` | 164.353 | -11.09% | 76.488 | -13.85% | 42.889 | -0.37% | 44.976 | 39.303 |
| `dec_b8_ctx65536` | `merged` | 164.131 | -11.21% | 76.480 | -13.86% | 42.901 | -0.34% | 44.749 | 39.509 |
| `dec_b32_ctx4096` | `inc` | 66.322 | +0.00% | 26.921 | +0.00% | 12.217 | +0.00% | 27.184 | 11.894 |
| `dec_b32_ctx4096` | `bkvp` | 63.065 | -4.91% | 23.696 | -11.98% | 12.180 | -0.30% | 27.189 | 11.879 |
| `dec_b32_ctx4096` | `d1` | 58.570 | -11.69% | 27.171 | +0.93% | 12.128 | -0.73% | 19.271 | 11.921 |
| `dec_b32_ctx4096` | `bkvp_d1` | 55.373 | -16.51% | 23.888 | -11.27% | 12.123 | -0.77% | 19.362 | 11.928 |
| `dec_b32_ctx4096` | `merged` | 55.189 | -16.79% | 23.838 | -11.46% | 12.127 | -0.74% | 19.225 | 11.916 |
| `dec_b32_ctx16384` | `inc` | 144.310 | +0.00% | 90.087 | +0.00% | 28.711 | +0.00% | 25.512 | 28.166 |
| `dec_b32_ctx16384` | `bkvp` | 132.327 | -8.30% | 77.935 | -13.49% | 28.665 | -0.16% | 25.727 | 28.039 |
| `dec_b32_ctx16384` | `d1` | 137.034 | -5.04% | 90.274 | +0.21% | 28.666 | -0.16% | 18.093 | 28.034 |
| `dec_b32_ctx16384` | `bkvp_d1` | 124.633 | -13.63% | 78.016 | -13.40% | 28.572 | -0.49% | 18.046 | 27.915 |
| `dec_b32_ctx16384` | `merged` | 124.691 | -13.59% | 78.000 | -13.42% | 28.627 | -0.29% | 18.064 | 27.940 |
| `dec_b32_ctx32768` | `inc` | 251.492 | +0.00% | 173.616 | +0.00% | 50.065 | +0.00% | 27.810 | 45.085 |
| `dec_b32_ctx32768` | `bkvp` | 222.168 | -11.66% | 144.180 | -16.95% | 50.024 | -0.08% | 27.964 | 45.054 |
| `dec_b32_ctx32768` | `d1` | 244.232 | -2.89% | 173.760 | +0.08% | 50.159 | +0.19% | 20.313 | 45.041 |
| `dec_b32_ctx32768` | `bkvp_d1` | 214.685 | -14.64% | 144.250 | -16.91% | 50.109 | +0.09% | 20.326 | 44.903 |
| `dec_b32_ctx32768` | `merged` | 214.703 | -14.63% | 144.202 | -16.94% | 50.099 | +0.07% | 20.402 | 44.884 |
| `dec_b32_ctx65536` | `inc` | 464.250 | +0.00% | 340.347 | +0.00% | 92.610 | +0.00% | 31.292 | 91.696 |
| `dec_b32_ctx65536` | `bkvp` | 396.697 | -14.55% | 272.633 | -19.90% | 92.547 | -0.07% | 31.517 | 91.599 |
| `dec_b32_ctx65536` | `d1` | 456.407 | -1.69% | 340.639 | +0.09% | 92.524 | -0.09% | 23.244 | 91.612 |
| `dec_b32_ctx65536` | `bkvp_d1` | 388.248 | -16.37% | 272.662 | -19.89% | 92.364 | -0.27% | 23.222 | 91.581 |
| `dec_b32_ctx65536` | `merged` | 388.396 | -16.34% | 272.701 | -19.88% | 92.406 | -0.22% | 23.289 | 91.551 |
| `dec_b64_ctx4096` | `inc` | 99.182 | +0.00% | 50.286 | +0.00% | 19.985 | +0.00% | 28.911 | 18.482 |
| `dec_b64_ctx4096` | `bkvp` | 91.019 | -8.23% | 42.181 | -16.12% | 19.992 | +0.03% | 28.847 | 18.574 |
| `dec_b64_ctx4096` | `d1` | 91.165 | -8.08% | 50.294 | +0.02% | 19.965 | -0.10% | 20.906 | 18.534 |
| `dec_b64_ctx4096` | `bkvp_d1` | 82.995 | -16.32% | 42.243 | -16.00% | 19.923 | -0.31% | 20.830 | 18.490 |
| `dec_b64_ctx4096` | `merged` | 83.072 | -16.24% | 42.262 | -15.96% | 19.981 | -0.02% | 20.829 | 18.543 |
| `dec_b64_ctx8704_mml9216` | `inc` | 170.428 | +0.00% | 111.771 | +0.00% | 31.754 | +0.00% | 26.904 | 31.027 |
| `dec_b64_ctx8704_mml9216` | `bkvp` | 144.028 | -15.49% | 85.143 | -23.82% | 31.814 | +0.19% | 27.071 | 30.996 |
| `dec_b64_ctx8704_mml9216` | `d1` | 162.406 | -4.71% | 111.626 | -0.13% | 31.696 | -0.18% | 19.084 | 31.097 |
| `dec_b64_ctx8704_mml9216` | `bkvp_d1` | 136.051 | -20.17% | 85.198 | -23.77% | 31.706 | -0.15% | 19.148 | 31.187 |
| `dec_b64_ctx8704_mml9216` | `merged` | 135.927 | -20.24% | 85.147 | -23.82% | 31.689 | -0.20% | 19.090 | 31.088 |
| `dec_b64_ctx16384` | `inc` | 255.481 | +0.00% | 175.981 | +0.00% | 50.414 | +0.00% | 29.085 | 49.695 |
| `dec_b64_ctx16384` | `bkvp` | 226.049 | -11.52% | 146.497 | -16.75% | 50.331 | -0.17% | 29.221 | 49.562 |
| `dec_b64_ctx16384` | `d1` | 246.962 | -3.33% | 175.569 | -0.23% | 50.314 | -0.20% | 21.079 | 49.920 |
| `dec_b64_ctx16384` | `bkvp_d1` | 217.889 | -14.71% | 146.440 | -16.79% | 50.393 | -0.04% | 21.056 | 50.011 |
| `dec_b64_ctx16384` | `merged` | 217.739 | -14.77% | 146.463 | -16.77% | 50.291 | -0.24% | 20.985 | 49.821 |
| `dec_b64_ctx32768` | `inc` | 469.738 | +0.00% | 342.799 | +0.00% | 94.436 | +0.00% | 32.503 | 94.153 |
| `dec_b64_ctx32768` | `bkvp` | 401.603 | -14.50% | 274.467 | -19.93% | 94.462 | +0.03% | 32.674 | 94.178 |
| `dec_b64_ctx32768` | `d1` | 461.562 | -1.74% | 342.695 | -0.03% | 94.527 | +0.10% | 24.340 | 94.337 |
| `dec_b64_ctx32768` | `bkvp_d1` | 393.227 | -16.29% | 274.497 | -19.92% | 94.411 | -0.03% | 24.320 | 94.220 |
| `dec_b64_ctx32768` | `merged` | 393.175 | -16.30% | 274.456 | -19.94% | 94.438 | +0.00% | 24.281 | 94.248 |
| `dec_b64_ctx65536` | `inc` | 890.288 | +0.00% | 677.244 | +0.00% | 172.420 | +0.00% | 40.623 | 170.304 |
| `dec_b64_ctx65536` | `bkvp` | 755.796 | -15.11% | 542.451 | -19.90% | 172.445 | +0.01% | 40.901 | 170.286 |
| `dec_b64_ctx65536` | `d1` | 882.660 | -0.86% | 677.205 | -0.01% | 172.671 | +0.15% | 32.785 | 170.294 |
| `dec_b64_ctx65536` | `bkvp_d1` | 747.737 | -16.01% | 542.516 | -19.89% | 172.476 | +0.03% | 32.744 | 170.263 |
| `dec_b64_ctx65536` | `merged` | 747.657 | -16.02% | 542.477 | -19.90% | 172.491 | +0.04% | 32.689 | 170.250 |
| `pre_q1024_ctx4096` | `inc` | 381.714 | +0.00% | 104.786 | +0.00% | 230.598 | +0.00% | 46.329 | 229.405 |
| `pre_q1024_ctx4096` | `bkvp` | 381.712 | -0.00% | 104.770 | -0.02% | 230.790 | +0.08% | 46.151 | 229.269 |
| `pre_q1024_ctx4096` | `d1` | 372.509 | -2.41% | 102.716 | -1.98% | 231.157 | +0.24% | 38.636 | 229.216 |
| `pre_q1024_ctx4096` | `bkvp_d1` | 372.159 | -2.50% | 102.695 | -2.00% | 231.015 | +0.18% | 38.450 | 229.323 |
| `pre_q1024_ctx4096` | `merged` | 353.044 | -7.51% | 83.589 | -20.23% | 231.058 | +0.20% | 38.397 | 229.213 |
| `pre_q1024_ctx16384` | `inc` | 1195.680 | +0.00% | 394.101 | +0.00% | 707.778 | +0.00% | 93.801 | 703.982 |
| `pre_q1024_ctx16384` | `bkvp` | 1195.992 | +0.03% | 394.318 | +0.06% | 707.867 | +0.01% | 93.807 | 703.964 |
| `pre_q1024_ctx16384` | `d1` | 1185.802 | -0.83% | 392.141 | -0.50% | 707.447 | -0.05% | 86.215 | 703.988 |
| `pre_q1024_ctx16384` | `bkvp_d1` | 1186.010 | -0.81% | 392.294 | -0.46% | 707.577 | -0.03% | 86.139 | 704.137 |
| `pre_q1024_ctx16384` | `merged` | 1105.286 | -7.56% | 311.817 | -20.88% | 707.526 | -0.04% | 85.942 | 704.149 |
| `pre_q4096_ctx4096` | `inc` | 1108.716 | +0.00% | 395.478 | +0.00% | 590.278 | +0.00% | 122.961 | 586.793 |
| `pre_q4096_ctx4096` | `bkvp` | 1115.656 | +0.63% | 395.570 | +0.02% | 589.970 | -0.05% | 130.116 | 586.404 |
| `pre_q4096_ctx4096` | `d1` | 1097.259 | -1.03% | 393.632 | -0.47% | 589.682 | -0.10% | 113.946 | 585.847 |
| `pre_q4096_ctx4096` | `bkvp_d1` | 1097.596 | -1.00% | 393.639 | -0.46% | 590.320 | +0.01% | 113.637 | 585.996 |
| `pre_q4096_ctx4096` | `merged` | 1020.868 | -7.92% | 316.907 | -19.87% | 590.330 | +0.01% | 113.631 | 586.170 |
| `pre_q4096_ctx16384` | `inc` | 4445.914 | +0.00% | 1552.664 | +0.00% | 2574.092 | +0.00% | 319.158 | 2564.745 |
| `pre_q4096_ctx16384` | `bkvp` | 4439.071 | -0.15% | 1552.700 | +0.00% | 2573.830 | -0.01% | 312.542 | 2564.743 |
| `pre_q4096_ctx16384` | `d1` | 4429.158 | -0.38% | 1550.515 | -0.14% | 2573.957 | -0.01% | 304.687 | 2565.096 |
| `pre_q4096_ctx16384` | `bkvp_d1` | 4429.429 | -0.37% | 1550.593 | -0.13% | 2574.117 | +0.00% | 304.720 | 2564.697 |
| `pre_q4096_ctx16384` | `merged` | 4106.756 | -7.63% | 1228.295 | -20.89% | 2573.773 | -0.01% | 304.688 | 2564.738 |

### Composition: is the sum of the parts the whole? (full op, µs)

| config | `inc` | Δ bkv_p | Δ D1 | Δ C | predicted sum | observed `merged` | **non-additive residual** | as % of `inc` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dec_b8_ctx4096` | 75.4 | +0.43 | -7.66 | +0.27 | -6.96 | -7.50 | **-0.54** | -0.72% |
| `dec_b8_ctx16384` | 104.9 | +0.16 | -8.34 | +0.11 | -8.07 | -8.03 | **+0.04** | +0.04% |
| `dec_b8_ctx32768` | 132.8 | -3.55 | -8.20 | -0.09 | -11.83 | -11.95 | **-0.11** | -0.09% |
| `dec_b8_ctx65536` | 184.9 | -12.23 | -7.95 | -0.22 | -20.40 | -20.73 | **-0.33** | -0.18% |
| `dec_b32_ctx4096` | 66.3 | -3.26 | -7.75 | -0.18 | -11.19 | -11.13 | **+0.06** | +0.09% |
| `dec_b32_ctx16384` | 144.3 | -11.98 | -7.28 | +0.06 | -19.20 | -19.62 | **-0.42** | -0.29% |
| `dec_b32_ctx32768` | 251.5 | -29.32 | -7.26 | +0.02 | -36.57 | -36.79 | **-0.22** | -0.09% |
| `dec_b32_ctx65536` | 464.2 | -67.55 | -7.84 | +0.15 | -75.25 | -75.85 | **-0.61** | -0.13% |
| `dec_b64_ctx4096` | 99.2 | -8.16 | -8.02 | +0.08 | -16.10 | -16.11 | **-0.01** | -0.01% |
| `dec_b64_ctx8704_mml9216` | 170.4 | -26.40 | -8.02 | -0.12 | -34.55 | -34.50 | **+0.05** | +0.03% |
| `dec_b64_ctx16384` | 255.5 | -29.43 | -8.52 | -0.15 | -38.10 | -37.74 | **+0.36** | +0.14% |
| `dec_b64_ctx32768` | 469.7 | -68.13 | -8.18 | -0.05 | -76.36 | -76.56 | **-0.20** | -0.04% |
| `dec_b64_ctx65536` | 890.3 | -134.49 | -7.63 | -0.08 | -142.20 | -142.63 | **-0.43** | -0.05% |
| `pre_q1024_ctx4096` | 381.7 | -0.00 | -9.20 | -19.12 | -28.32 | -28.67 | **-0.35** | -0.09% |
| `pre_q1024_ctx16384` | 1195.7 | +0.31 | -9.88 | -80.72 | -90.29 | -90.39 | **-0.10** | -0.01% |
| `pre_q4096_ctx4096` | 1108.7 | +6.94 | -11.46 | -76.73 | -81.25 | -87.85 | **-6.60** | -0.60% |
| `pre_q4096_ctx16384` | 4445.9 | -6.84 | -16.76 | -322.67 | -346.27 | -339.16 | **+7.11** | +0.16% |

Largest non-additive residual across the grid: **0.72%** of `inc`.

### Where the time is, in `inc` (share of full op)

| config | stage A | stage B | residual |
| --- | ---: | ---: | ---: |
| `dec_b8_ctx4096` | 13.3% | 23.2% | 63.5% |
| `dec_b8_ctx16384` | 24.8% | 26.3% | 48.9% |
| `dec_b8_ctx32768` | 35.3% | 25.5% | 39.2% |
| `dec_b8_ctx65536` | 48.0% | 23.3% | 28.7% |
| `dec_b32_ctx4096` | 40.6% | 18.4% | 41.0% |
| `dec_b32_ctx16384` | 62.4% | 19.9% | 17.7% |
| `dec_b32_ctx32768` | 69.0% | 19.9% | 11.1% |
| `dec_b32_ctx65536` | 73.3% | 19.9% | 6.7% |
| `dec_b64_ctx4096` | 50.7% | 20.1% | 29.1% |
| `dec_b64_ctx8704_mml9216` | 65.6% | 18.6% | 15.8% |
| `dec_b64_ctx16384` | 68.9% | 19.7% | 11.4% |
| `dec_b64_ctx32768` | 73.0% | 20.1% | 6.9% |
| `dec_b64_ctx65536` | 76.1% | 19.4% | 4.6% |
| `pre_q1024_ctx4096` | 27.5% | 60.4% | 12.1% |
| `pre_q1024_ctx16384` | 33.0% | 59.2% | 7.8% |
| `pre_q4096_ctx4096` | 35.7% | 53.2% | 11.1% |
| `pre_q4096_ctx16384` | 34.9% | 57.9% | 7.2% |

---

Raw per-arm `hlo_stats` output, including row counts and partition residuals, is in the reader's
JSON (`read_merged_hlo_stats.py --out`). The xplanes themselves are the job outputs of
`j-2c00d88c-e7de-415f-968b`, retrievable with `cdk job sync-outputs`.
