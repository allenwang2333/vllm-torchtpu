# Variant A — "Narrow Record": results

Step-3 variant of `measurement_and_design.md` § 3, option A.

Two builds. The first (`j-ce74a9f2-a4cd-40cb-9e87`) was rejected by Mosaic and
produced no reader-side number; § 6 keeps that error, because the layout rule it
established is the reason the second build is shaped the way it is. **All
reader-side results below are from the second build,
`j-c2e7edde-deee-4592-b81d`, `JobStatus: Succeeded`, `4/4 shards succeeded`.**

**Headline, one line each:**

- **Reader side, decode: a real win.** **10.45% faster** at `decode_prod` and
  **7.66% faster** at `decode_64k`, `measured (j-c2e7edde-deee-4592-b81d)`,
  against `modelled` 13.42% and 8.36%. The spreads are far below the effect.
- **Reader side, mixed: inconclusive, in both rows.** The measured deltas (1.43%
  and 0.36% faster) are smaller than the spreads that produced them. **This is
  the half of the workload that dominates the A/B shape**, so it matters more
  than the decode win — see § 4.2.
- **Writer side: a large, separate win.** The split scatter is **70.08% faster**
  than today's wide scatter at `T=1024`, `measured
  (j-c2e7edde-deee-4592-b81d)`, reproducing the first build's 70.64%. Never
  netted against the reader side.
- **Correctness: clean.** Bit-exact against the incumbent at all three regimes,
  0 of 2,240,512 index elements differing; oracle margins bit-identical to the
  incumbent's; negative controls fired.
- **MXU operand types: confirmed native fp8**, read from the lowering. 1064
  `vmatmul`, `bf16 = 0`.

**Sign convention, everywhere in this file:** every percentage is a change in
**time**, `(new − old) / old`. **Negative is faster.** Signed percentages appear
only in a column named `Δ time %`, in the same row as the two absolute times
they came from. Prose states the direction in words. A `speedup ×` column is a
different number: greater than 1 is faster.

**Tags:** `measured (job id)` / `modelled` / `assumed` / `unmodelled`.
`unmodelled` means the floor has no term for the quantity; it never has a number
beside it.

---

## 1. What was run

| | build 2 — the results | build 1 — the rejection |
| --- | --- | --- |
| job | `j-c2e7edde-deee-4592-b81d` | `j-ce74a9f2-a4cd-40cb-9e87` |
| `JobStatus` | **`Succeeded`** (`state: Complete`) | `Failed` (`state: Complete`) |
| fanout | **`tpu-fanout: 4/4 shards succeeded`**, all rc=0 | `2/4` — shard2 rc=1, shard3 rc=1 |
| variant md5 | **`bacd4a4b1e51a9fc8b5709373767b019`** | `a7128f58e0c597253be16a778ef78a17` |

Both on image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`,
jax `0.10.2`, each shard seeing 2 `TpuDevice`s, i.e. correctly pinned to one
chip of the four-chip pod.

The incumbent module in the image is
`/usr/local/lib/python3.12/site-packages/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`
md5 **`1e221f10cbf3573aa0d2ae78c3463532`**, verified byte-identical to the
checkout's `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`.
`narrow_record.diff` is therefore a diff against exactly what ran. Checked
rather than assumed:
`git log --oneline 950974c2..HEAD -- src/ kernel_work/streamindex_topk/instruments/`
is **empty** — no commit since the dispatch's stated HEAD has touched either
tree, despite the branch having moved several times under concurrent agents.

`RUN_CMD` for build 2, **as read back from
`cdk job recipe j-c2e7edde-deee-4592-b81d --no-color`** (not as typed):

```
export PYTHONPATH=/workspace && bash /root/fanout/tpu-fanout.sh
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/bench_narrow.py --regimes decode_prod,decode_64k --json $CDK_OUTPUT_DIR/bench_decode.json"
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/bench_narrow.py --regimes mixed_prod,mixed_64k --json $CDK_OUTPUT_DIR/bench_mixed.json"
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/correctness_narrow.py"
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/writer_probe.py --json $CDK_OUTPUT_DIR/writer.json ; python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/llo_narrow.py --regime mixed_prod --arm variant --out /tmp/llo_mv ; python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/llo_narrow.py --regime decode_prod --arm variant --out /tmp/llo_dv"
```

`--map-dir /home/annyan_google_com/workspaces/vllm-torchtpu/kernel_work:tpu-job:/workspace/kernel_work`
plus the fanout dir; tag `agent-jobs,narrow-record-variant`.

The shard-3 steps are separated by `;`, not `&&`, deliberately: in build 1 they
were `&&`-chained, and a failure in the LLO capture would have taken the writer
probe's numbers with it.

---

## 2. The format change

Precise enough to implement on the writer side.

**Incumbent — one array.**

```
cache_kv  u8[num_pages, page_size//kv_packing, kv_packing, 256]
          byte   0..127   the 128 fp8 (e4m3) key values
          byte     128    the single e8m0 block scale
          byte 129..255   padding — never read, always transferred
```

`256 = align_to(align_to(128,128) + 1, 128)`; 129 of 256 bytes are live, so the
indexer's read of this array is amplified 1.98×.

**Variant — two arrays.**

```
cache_val    u8[num_pages, page_size//kv_packing, kv_packing, 128]
             the 128 fp8 key values, and nothing else.
             Same index arithmetic as today, minor axis 128 instead of 256.

cache_scale  u8[num_pages, page_size // 128, 128]
             the e8m0 scale of position p of page g at [g, p // 128, p % 128].
```

At the production shape (`page_size=1024`, `kv_packing=4`, 415 pages):
`cache_val` is 54.4 MB and `cache_scale` is 0.42 MB, against 108.79 MB today.
129 live bytes per position instead of 256 — the indexer's KV read falls by
**49.6%**.

The pair is passed to `streamindex_topk` as a two-tuple
`cache_kv = (cache_val, cache_scale)`. `jax.jit` treats it as a pytree, so the
public signature — and therefore the read-only instruments' call site — is
unchanged. Confirmed in both jobs: no call-site change was needed.

Shapes as reported by the job, which is why the byte accounting above can be
checked against the log rather than against source:

| regime | `cache_val` | `cache_scale` |
| --- | --- | --- |
| `decode_prod`, `mixed_prod` | `(576, 256, 4, 128)` u8, 72.0 MiB | `(576, 8, 128)` u8, 0.6 MiB |
| `decode_64k`, `mixed_64k` | `(390, 256, 4, 128)` u8, 48.8 MiB | `(390, 8, 128)` u8, 0.4 MiB |

### 2.1 The scale plane must be declared 3-D

`[num_pages, page_size//128, 128]` holds exactly the same bytes in exactly the
same order as a flat `[num_pages, page_size]` plane would. The declaration is
not cosmetic:

**Mosaic tiles the last two dimensions of an HBM memref.** In the 2-D form the
tiled dimensions are `(page, position)`, so the *page* axis is the tiled sublane
axis — `(32,128)` at 576 pages, `(8,128)` at 390 — and the reader's one-page DMA
is a single-row slice of a tiled dimension at a data-dependent offset. That is
rejected outright (§ 6). In the 3-D form the page axis is untiled and `(8,128)`
is **exactly one page of 1024 scales**, so the per-page DMA is a whole number of
tiles, moves 1024 B, and wastes nothing.

Confirmed in the lowering — the HBM operand appears as `memref<576x8x128xi8>`
(§ 8) — and in VMEM as `s8[2,1,2,8,128]{T(8,128)(4,1)}`, 4096 B (§ 4.5).

### 2.2 Writer-side change

Today, `mla_attention_op.py:103-104` pads `k_packed[T,129]` up to `[T,256]` and
does one scatter. Under the split it does two, into two donated buffers, in one
`jit`:

```python
vc = vc.at[page, slot // kv_packing, slot % kv_packing].set(k_val)   # [T,128]
sc = sc.at[page, slot // 128, slot % 128].set(k_scale)               # [T]
```

with `page` and `slot` computed exactly as they are today, and
`donate_argnums=(0, 1)`. The `jnp.pad` to 256 disappears. The cache allocator
makes two buffers instead of one. This is arm `W12b` in § 5, and it is the form
that was measured — the caveat build 1 had to leave open (that only the 2-D
writer form had been timed) is closed.

**Only the reader side is implemented here.** The writer is priced in § 5, not
built. Nothing in the shared tree was modified: the variant is a copy of the
kernel inside this folder, and `narrow_record.diff` carries the change.

---

## 3. Why the variant is bit-exact by construction

The variant changes which bytes cross HBM, not the arithmetic — with one
subtlety introduced by the 3-D relayout, which is why this has its own section.

The scale plane lands in VMEM as `[bkv_p, 8, 128]` while `compute_scores` wants
`[1, bkv_sz]`, so `load_bkv` must regroup. It widens **e8m0 → f32 before the
regroup**, so the relayout runs on an f32 `(8,128)` tile rather than a packed
8-bit one. That widen is exact, and so is the incumbent's:

- e8m0 (`float8_e8m0fnu`) is a **bare 8-bit exponent** — every value it can hold
  is a power of two, with no mantissa to round.
- bf16 and f32 both have **8 exponent bits**, covering the same range e8m0
  spans.
- So both `e8m0 → bf16` (the incumbent) and `e8m0 → f32` (the variant) are
  exact, and neither rounds.

The incumbent widens to bf16 and then promotes to f32 in the multiply against
the f32 `s_summed`; the variant widens straight to f32. Same value, same
product. **Variant A therefore remains bit-exact against the incumbent by
construction even after the relayout** — the regroup moves bytes between lanes
and sublanes but does not touch a numeric value.

That is a claim about the design, and it is the kind of claim that should not be
trusted without a check. § 7 is that check, and it passes: 0 of 2,240,512 index
elements differ.

---

## 4. Reader side — four regimes, same job, same instruments, same shapes

Timed by the **read-only** `instruments/microbench.py` (`WARMUP=3`, `REPS=10`,
`early_exit=False`), incumbent and variant back to back in the same process on
the same built input. All `measured (j-c2e7edde-deee-4592-b81d)`.

`Δ time % = (variant − incumbent) / incumbent`. **Negative is faster.**

| regime | incumbent ms | sprd % | variant ms | sprd % | Δ time % | speedup × | inner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 0.166699 | 0.589 | 0.149275 | 0.789 | **−10.45** | 1.117 | 20 |
| `decode_64k` | 0.202102 | 2.438 | 0.186629 | 0.467 | **−7.66** | 1.083 | 20 |
| `mixed_prod` | 0.675440 | 2.512 | 0.665794 | 1.608 | −1.43 | 1.014 | 20 |
| `mixed_64k` | 4.788465 | 1.109 | 4.771312 | 1.518 | −0.36 | 1.004 | 5 |

**Drift control** — the same incumbent re-timed after the variant arm, third in
the order. `Δ time %` here is `(repeat − first) / first`:

| regime | incumbent ms | repeat ms | Δ time % | repeat sprd % |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 0.166699 | 0.166704 | +0.00 | 0.638 |
| `decode_64k` | 0.202102 | 0.199937 | −1.07 | 0.663 |
| `mixed_prod` | 0.675440 | 0.674938 | −0.07 | 1.858 |
| `mixed_64k` | 4.788465 | 4.812547 | +0.50 | 7.123 |

Every drift is inside its own spread, so the machine did not move under the
experiment and the deltas above may be read as deltas.

### 4.1 Which rows support a conclusion

Applying the dispatch's rule — *a row where the spread is comparable to the
effect cannot support a conclusion* — the four rows split cleanly in two:

| regime | Δ time % | larger of the two spreads | verdict |
| --- | ---: | ---: | --- |
| `decode_prod` | −10.45 | 0.789 | **conclusive** — effect is 13× the spread |
| `decode_64k` | −7.66 | 2.438 | **conclusive** — effect is 3.1× the spread |
| `mixed_prod` | −1.43 | 2.512 | **inconclusive** — spread exceeds the effect |
| `mixed_64k` | −0.36 | 1.518 | **inconclusive** — spread is 4× the effect |

So: **A is measurably faster on decode, and this instrument cannot tell whether
it changes mixed at all.** Both mixed rows point the right way, and both are
consistent with the `modelled` near-zero, but pointing the right way from inside
the noise is not a measurement and is not reported as one here.

### 4.2 The inconclusive rows are the ones that matter most

This is the finding that most affects A's standing, and it cuts against the
decode headline rather than with it.

Prefill/mixed carries **85.7–87.9% of indexer time** at the A/B shape
[`modelled (j-b54e9716-3db5-4f1f-b8d4)` — supplied to this exercise, not
measured in it]. The two rows that dominate that number are exactly the two this
instrument could not resolve. Composing the two — arithmetic on one `modelled`
input and one `measured` one, so **the result is a composition, not a
measurement** — gives a range for A's reader-side effect on total indexer time:

- taking the mixed rows at face value:
  `0.879 × (−1.43) + 0.121 × (−10.45)` = **−2.5%**, to
  `0.857 × (−1.43) + 0.143 × (−10.45)` = **−2.7%**
- taking the mixed rows as zero, which is what their spreads license:
  `0.121 × (−10.45)` = **−1.3%**, to `0.143 × (−10.45)` = **−1.5%**

**The defensible figure is the lower one: roughly 1.3–1.5% of indexer time
faster.** The reader-side prize is real but small, and it is small for a reason
the model already gave — in mixed, top-k dominates the call (0.400 of 0.675 ms
at `mixed_prod`, 3.067 of 4.788 ms at `mixed_64k`), so halving the KV read moves
little. A's reader side was always a decode optimization.

Resolving the mixed rows needs an instrument that can separate a ~1% effect from
a ~2% spread — more reps, or the scoring half timed directly rather than by
difference. Not run; it is the obvious next cheap job if the mixed rows matter
to the decision.

### 4.3 Modelled versus measured

`measurement_and_design.md` § 3 predicted, at rate 2.97 TB/s `measured`:

| regime | old µs | new µs `modelled` | Δ time % `modelled` | Δ time % measured | agreement |
| --- | ---: | ---: | ---: | ---: | --- |
| `decode_prod` | 167.01 | 144.60 | −13.42 | **−10.45** | 3.0 points short |
| `decode_64k` | 201.16 | 184.35 | −8.36 | **−7.66** | within 0.7 points |
| `mixed_prod` | 669.11 | 666.31 | −0.42 | −1.43 (inconclusive) | consistent |
| `mixed_64k` | 4762.84 | 4740.43 | −0.47 | −0.36 (inconclusive) | consistent |

`decode_64k` lands within 0.7 points of its model. `decode_prod` delivers 78% of
its modelled prize. The model assumed the whole 49.6% byte saving converts at
2.97 TB/s with no other term changing; the likely shortfall is the regroup that
replaced the incumbent's transpose, for which the model has no term. It is
`unmodelled`, and pinning it down would need the scoring half timed directly
rather than by difference.

### 4.4 Where the variant's time went

`scoring_ms_by_difference` = full call − the instrument's `topk`-only control:

| regime | incumbent scoring ms | variant scoring ms | Δ time % | incumbent topk ms | variant topk ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 0.111496 | 0.093009 | −16.58 | 0.055203 | 0.056266 |
| `decode_64k` | 0.117689 | 0.101609 | −13.66 | 0.084413 | 0.085020 |
| `mixed_prod` | 0.275782 | 0.266018 | −3.54 | 0.399658 | 0.399776 |
| `mixed_64k` | 1.721653 | 1.726948 | +0.31 | 3.066812 | 3.044364 |

The decode rows show the change landing where it should: the **scoring** half —
the half that reads KV — improves 16.6% and 13.7%, while the top-k control is
unchanged. That is the mechanism the design predicted, observed directly.

**Do not read the mixed rows of this table.** They are differences of two
medians and inherit both spreads, and at `mixed_prod` the variant's `topk`
control had a spread of **67.85%**, which makes the derived 0.266018 worthless.
The full-call rows in § 4 are unaffected and are the reportable ones.

### 4.5 VMEM

From `*-06-post-rematerialize-allocations.txt`,
`measured (j-c2e7edde-deee-4592-b81d)`. **Static compile-time analysis, not a
runtime number.** The incumbent column is P1's, from
`probes/llo_capture_RESULT.md`.

| program | incumbent VMEM | variant VMEM |
| --- | ---: | ---: |
| `StreamIdxTC-m-bq_128-bkvp_2.1` | 4.195 MiB | **3.699 MiB** |
| `StreamIdxTC-d-bq_1-bkvp_2.2` | 4.168 MiB | **2.184 MiB** |
| `StreamIdxTC-d-bq_1-bkvp_2.3` | 1.095 MiB | **0.599 MiB** |

The variant uses **less** VMEM, as expected: the KV double-buffer narrows from
256-wide to 128-wide (`s8[2,4,512,4,128]` = 2.000 MiB at `-d-...2.2`), and the
scale buffer it adds is 0.016 MiB. There was no VMEM rejection, and headroom
went up rather than down. The regrouped scale appears as
`f32[2,N,16,128]{T(8,128)}`.

---

## 5. Writer side — priced, not built. **Never netted against § 4.**

Pure XLA, no Pallas — the same code path `mla_attention_op.py:103-104` uses — at
the production cache size `u8[415,256,4,256]` (424,960 positions, 108.79 MB).
Index arithmetic (`seq_id`, `pos`, `page`, `slot`) copied verbatim from that
file. `WARMUP=3`, `REPS=10`, `INNER=5`, donated buffers chained through
themselves so no allocation is timed. All
`measured (j-c2e7edde-deee-4592-b81d)`.

| arm | what it does |
| --- | --- |
| `W0_today_wide` | today: `jnp.pad([T,129] → [T,256])`, one scatter into `u8[415,256,4,256]` |
| `W1_val_only` | scatter `k_val[T,128]` into `u8[415,256,4,128]` |
| `W2_scale_only` | scatter `k_scale[T]` into the **2-D** `u8[415,1024]` |
| `W2b_scale_only_3d` | scatter `k_scale[T]` into the **3-D** `u8[415,8,128]` |
| `W12_split_both` | `W1`+`W2` in one `jit` — 2-D form, reproduces build 1 |
| **`W12b_split_both_3d`** | `W1`+`W2b` in one `jit` — **the form the reader needs; this is the result** |
| `N0/N1/N2` | donated single-element scatters. **Not results** — the dispatch-and-donation floor |

### T = 1024 — prefill, `PROD_MAX_NUM_BATCHED_TOKENS`

| arm | ms | spread % | implied GB/s (r+w) |
| --- | ---: | ---: | ---: |
| `N0_noop_wide` | 0.054757 | 28.63 | — |
| `N1_noop_val` | 0.059934 | 25.28 | — |
| `N2_noop_scale` | 0.065855 | 16.87 | — |
| `W0_today_wide` | 0.844617 | 0.83 | 257.6 |
| `W1_val_only` | 0.208491 | 2.39 | 521.8 |
| `W2_scale_only` | 0.100466 | 4.96 | 8.5 |
| `W2b_scale_only_3d` | 0.099841 | 2.70 | 8.5 |
| `W12_split_both` | 0.251830 | 3.44 | 435.4 |
| `W12b_split_both_3d` | 0.252724 | 3.81 | 433.8 |

| baseline | ms | candidate | ms | Δ time % | speedup × |
| --- | ---: | --- | ---: | ---: | ---: |
| `W0_today_wide` | 0.844617 | **`W12b_split_both_3d`** | 0.252724 | **−70.08** | **3.342** |
| `W0_today_wide` | 0.844617 | `W12_split_both` (2-D) | 0.251830 | −70.18 | 3.354 |

**The split writer is 70.1% faster than today's writer at `T=1024`.** Spreads
0.83% and 3.81%, an order of magnitude below the effect, and both arms sit far
above the noop floor, so this row supports a conclusion.

Three things to read off it:

- **The 3-D declaration is free on the writer side.** `W12b` 0.252724 ms against
  `W12` 0.251830 ms — 0.35% apart, inside both spreads; and `W2b` 0.099841
  against `W2` 0.100466. Splitting `slot` into `slot // 128, slot % 128` costs
  nothing measurable, so the reader's tiling requirement imposes no writer-side
  penalty.
- **It reproduces.** Build 1 measured `W0` 0.842139 → `W12` 0.247234, −70.64%,
  3.406×; build 2 measures 0.844617 → 0.251830, −70.18%, 3.354×. Two jobs, half
  a point apart.
- **The direction is the opposite of what A's rank-3 placement assumed.** Today's
  wide scatter moves bytes at 258 GB/s against `W1`'s 522 — it pays the same
  1.98× amplification on the **write** side that motivated the read-side change,
  plus a `jnp.pad`. Fusing the two scatters is itself worth ~18%: `W1 + W2b`
  separately is 0.308332 ms against `W12b`'s 0.252724 ms.

**`T=1024` is the prefill/mixed path**, which is where 85.7–87.9% of indexer
time sits [`modelled (j-b54e9716-3db5-4f1f-b8d4)`]. Unlike the reader-side win,
the writer-side win lands squarely on the dominant phase.

### T = 64 — decode

| arm | ms | spread % |
| --- | ---: | ---: |
| `N0_noop_wide` | 0.059347 | 21.01 |
| `N1_noop_val` | 0.054990 | 24.43 |
| `N2_noop_scale` | 0.054483 | 23.84 |
| `W0_today_wide` | 0.068478 | 42.34 |
| `W1_val_only` | 0.061259 | 15.14 |
| `W2_scale_only` | 0.060834 | 28.71 |
| `W2b_scale_only_3d` | 0.071969 | 50.31 |
| `W12_split_both` | 0.088250 | 11.83 |
| `W12b_split_both_3d` | 0.079833 | 16.57 |

| baseline | ms | candidate | ms | Δ time % | speedup × |
| --- | ---: | --- | ---: | ---: | ---: |
| `W0_today_wide` | 0.068478 | `W12b_split_both_3d` | 0.079833 | +16.58 | 0.858 |

**This row cannot support a conclusion, and the `+16.58` must not be quoted as a
result.** Three reasons, any one sufficient:

1. The spreads (16–50%) are comparable to or larger than the effect.
2. Every arm sits within 0.01–0.02 ms of the donated-noop floor
   (0.054–0.061 ms). At `T=64` these kernels are dispatch-bound, not
   scatter-bound: `W12b` moves strictly fewer bytes than `W0`, and its extra
   time is most consistent with launching two scatters instead of one.
3. The two builds disagree by more than either effect — build 1 read +36.75%
   here, build 2 reads +16.58%. That disagreement is itself the evidence that
   neither is measuring anything.

**Not resolvable on this instrument.** A meaningful decode-side writer number
needs a shape where the scatter is not floor-bound — sweeping `T` from 64 to
1024 to find where the arms leave the floor would do it. That was judged not
worth the remaining budget; "not resolvable here" is a better answer than either
number.

### The two numbers are not netted

**§ 4 and § 5 are not netted against each other, and must not be.** They are
results at two different levels — one is the indexer kernel's read, the other is
the producer's write — measured by different instruments on different code. A
single combined figure would hide exactly the term A's ranking turns on.

### Caveats specific to § 5

- `W0` at 0.845 ms is not comparable to the ~291 µs `measured` for
  `scatter_fusion` in the step-1 profile: different `T`, different surrounding
  fusion, different pass. The **ratio** `W12b/W0` within one job is the result;
  the absolute is not a substitute for the profile's.
- One `jit`ted scatter in isolation is not the producer in situ. Whether XLA
  fuses `W12b` into its neighbours the way it fuses `W0` today is `unmodelled`.

---

## 6. The Mosaic rejection from build 1, and why it is kept

Build 1, `j-ce74a9f2-a4cd-40cb-9e87`, declared the scale plane 2-D as
`u8[num_pages, page_size]` and was rejected at all four regimes. At 576 pages:

```
MosaicError('INTERNAL: Mosaic failed to compile TPU kernel with MLIR module name `<no name>`
HLO name `StreamIdxTC-d-bq_1-bkvp_2.2`: Offsets along tiled dimensions must be aligned to
tiles. Failed to verify that the index at dimension 0 is divisible by the tile dimension 32.
If it is, use tpu.assume_multiple to suppress this error.

The MLIR operation involved:
  %766 = "tpu.memref_slice"(%arg10, %755, %21) <{operandSegmentSizes = array<i32: 1, 2, 0>}> :
  (memref<576x1024xi8, #tpu.tiled<(32,128)(4,1),[8,1]>, #tpu.memory_space<hbm>>, i32, i32) ->
  memref<1x1024xi8, #tpu.tiled<(32,128)(4,1),[8,1]>, #tpu.memory_space<hbm>>

Please report a bug at: https://github.com/jax-ml/jax/issues/new/choose
')
```

At 390 pages, the same error with tile dimension **8** and
`#tpu.tiled<(8,128)(4,1),[8,1]>`. The tile size tracking the array's *leading*
dimension is the tell: it is the page axis being tiled.

Location, from the `dma_start` source trace in the job log:
`_async_copy` ← `_fetch_bkv` (the scale-plane DMA) ← `start_fetch_bkv` ←
`prologue`.

`tpu.assume_multiple`, which the error suggests, is **not** applicable: the page
index genuinely is not a multiple of 32, so suppressing the check would produce
wrong data rather than working code.

This is kept because it is a reusable rule, not a mishap: **Mosaic tiles the
last two dimensions of an HBM memref, so any array you intend to DMA one row at
a time must not have that row on a tiled axis.** Declaring the same bytes in the
same order as `[num_pages, page_size//128, 128]` moves the page axis off the
tiling and the identical DMA becomes legal (§ 2.1). Per the dispatch, build 1
stopped at the rejection rather than retrying around it; the 3-D relayout was
run only after the caller authorised it, on the standing condition that a second
rejection would end the exercise. There was no second rejection.

Build 1 also supplied the incumbent baselines and the first writer measurement,
both of which build 2 reproduces (§ 4.3, § 5).

---

## 7. Correctness — clean. `measured (j-c2e7edde-deee-4592-b81d)`, shard 2 rc=0

`RANK_SLACK=256 MIN_RECALL=0.99 k=2048`.

**Check 1 — bit-exact equality against the incumbent, same input, both impls.**

```
[decode_prod] EXACT-EQUALITY PASS -- shape=(64, 2048) dtype=int32, 0 of 131072 elements differ
[decode_64k]  EXACT-EQUALITY PASS -- shape=(6, 2048) dtype=int32, 0 of 12288 elements differ
[mixed_prod]  EXACT-EQUALITY PASS -- shape=(1024, 2048) dtype=int32, 0 of 2097152 elements differ
```

2,240,512 index elements, zero differing. This is the strong check — no
tolerance, no oracle — and it confirms § 3's construction argument empirically:
the `e8m0 → f32` widen and the `[bkv_p,8,128] → [1,bkv_sz]` regroup changed no
numeric value.

**Check 2 — the read-only instrument's oracle rank criterion, on the variant.**

```
[decode_prod] ORACLE PASS -- all 64 rows are a valid top-2048 against the float64 oracle.
              worst_rank=2109 of allowed 2304, min_recall=0.99561 of required 0.99
[decode_64k]  ORACLE PASS -- all 6 rows are a valid top-2048 against the float64 oracle.
              worst_rank=2105 of allowed 2304, min_recall=0.99512 of required 0.99
[mixed_prod]  ORACLE PASS -- all 1024 rows are a valid top-2048 against the float64 oracle.
              worst_rank=2098 of allowed 2304, min_recall=0.99512 of required 0.99
```

These are **bit-identical to the incumbent's step-1 margins** (`report.md`, jobs
`j-6d675a58-7ecd-4b44-9171` and `j-5f8f4a08-20b4-4fa0-bfc3`): 2109 / 0.99561,
2105 / 0.99512, 2098 / 0.99512. Because A is bit-exact by construction, anything
other than an exact match here would have been a bug rather than a tolerance.
They match.

**Check 3 — negative control, on the variant.** A check that has never failed is
not evidence.

```
[decode_prod] NEGATIVE CONTROL OK -- check fired on a mutated variant (64 of 64 rows bad).
              worst_rank=8177 of allowed 2304, min_recall=0.49512 of required 0.99.
              first: row 0: worst_rank=7891 (allowed 2304), recall=0.49512 (min 0.99),
              1034 of 2048 indices outside the true top-2048
[mixed_prod]  NEGATIVE CONTROL OK -- check fired on a mutated variant (1024 of 1024 rows bad).
              worst_rank=8159 of allowed 2304, min_recall=0.42578 of required 0.99.
              first: row 0: worst_rank=6993 (allowed 2304), recall=0.61133 (min 0.99),
              796 of 2048 indices outside the true top-2048
```

Both fired on every row, at the same values the incumbent's controls fired at
(8177 / 0.49512 and 8159 / 0.42578). Checks 1 and 2 are therefore not vacuous.

`correctness rc=0`.

---

## 8. MXU operand types — **confirmed native fp8**, from the lowering

`measured (j-c2e7edde-deee-4592-b81d)`. This was the load-bearing risk: the
relayout widens the scales to f32, which is exactly the kind of change that can
drag the matmul operands with it, and step 2's P1 established that a
bf16-operand formulation is a **1.86× regression**. It did not happen. Read from
the lowering, not inferred from source, at both levels P1 used, on the variant,
at both `mixed_prod` and `decode_prod`.

**Mosaic, `*post-infer-vector-layout.txt`:**

```
func=['StreamIdxTC-m-bq_128-bkvp_2']
    hbm operand: memref<576x256x4x128xi8...>     <- the value plane
    hbm operand: memref<576x8x128xi8...>         <- the 3-D scale plane
    x1  tpu.matmul : vector<4096x128xf8E4M3FN>, vector<2048x128xf8E4M3FN>,
                     vector<4096x2048xf32> -> vector<4096x2048xf32>

func=['StreamIdxTC-d-bq_1-bkvp_2']
    x1/x4  tpu.matmul : vector<32x128xf8E4M3FN>, vector<2048x128xf8E4M3FN>,
                        vector<32x2048xf32> -> vector<32x2048xf32>
```

The `-m-` signature is **character-for-character P1's incumbent signature**. The
two HBM operands also confirm the split reached the compiler as intended.

**jf, `*-71-final_bundles.txt`:**

| program | vmatmul total | bf16 | f8 |
| --- | ---: | ---: | ---: |
| `StreamIdxTC-m-bq_128-bkvp_2.1` | 1024 | **0** | 1024 |
| `StreamIdxTC-d-bq_1-bkvp_2.2` | 32 | **0** | 32 |
| `StreamIdxTC-d-bq_1-bkvp_2.3` | 8 | **0** | 8 |
| **total** | **1064** | **0** | **1064** |

Opcodes in the `-m-` program: 1016 × `vmatmul.mubr.f8e4m3fn.gmra.mrb` and 8 ×
`vmatmul.mubr.f8e4m3fn.vlgmr.msra.gmra.mrb`. **1064 total with zero bf16 —
identical to P1's incumbent count and opcode mix.** The einsum still reaches the
MXU as native fp8; the f32 scale widen sits after the matmul, on the score,
exactly where it was intended to.

Everything in this section is **static compile-time analysis**. The
`compile+first` wall-clocks the capture printed (12042.8 ms, 11478.3 ms) are
LLO-instrumented and host-contended and are **not** performance numbers.

---

## 9. Surviving caveats

- **Single-chip microbenchmark, one TensorCore.** § 4 and § 5 are measurements
  on one chip of a four-chip pod with three other shards running concurrently.
  A win here is a **kernel-level** win and must be labelled one. It is not an
  end-to-end number and does not predict one.
- **The mixed rows are inconclusive, and they dominate.** See § 4.2. The
  defensible reader-side figure is ~1.3–1.5% of indexer time, not the 10.45%
  headline, and even that leans on a `modelled` phase split supplied from
  outside this exercise.
- **The composition in § 4.2 is arithmetic over a `modelled` input**, not a
  measurement, and is tagged as such wherever it appears.
- **The input was synthetically pre-split.** `narrow_inputs.split_cache` derives
  `cache_val` and `cache_scale` from the read-only instrument's packed cache
  (asserting bytes 129..255 are zero). No producer wrote this format; the two
  sides have never run together. The reader-side experiment therefore measures
  the reader in isolation, which is the point, but it is not the system.
- **The reader-side and writer-side numbers are not netted**, per the dispatch,
  and no figure in this file combines them.
- **The `T=64` writer row is not resolvable on this instrument** — see § 5.
- **The § 4.3 shortfall at `decode_prod`** — 10.45% delivered against 13.42%
  modelled — is not explained. The regroup that replaced the incumbent's
  transpose is the obvious suspect, and it is `unmodelled`.
- **HEAD drift.** The dispatch named HEAD `950974c2`; the branch has moved
  several times under concurrent agents. Verified rather than assumed:
  `git log 950974c2..HEAD -- src/ kernel_work/streamindex_topk/instruments/` is
  empty, and the shipped kernel's md5 still matches the image's, so
  `narrow_record.diff` applies against what ran.
- **Neither `/load-experiment` nor `/log-experiment` was run**, per the
  dispatch. Nothing here has gone into the shared experiment log.
- **No `/critical-review` has been run on these numbers.** Per the repo's
  standing rule that is required before an A/B result is reported as accepted.

---

## 10. Files in this folder

| file | what it is |
| --- | --- |
| `streamindex_topk_narrow.py` | the variant kernel, a self-contained copy. md5 `bacd4a4b1e51a9fc8b5709373767b019` |
| `narrow_record.diff` | that file as a unified diff against the shipped `src/.../streamindex_topk.py` |
| `narrow_inputs.py` | derives the split arrays from the read-only instrument's output; `patched_inputs()` rebinds `instruments.inputs.build` |
| `_load.py` | file-path loader — this folder's name has a hyphen and cannot be a package |
| `bench_narrow.py` | the A/B driver; three arms per regime through the read-only `microbench.run_regime` |
| `correctness_narrow.py` | the three checks in § 7 |
| `writer_probe.py` | § 5 |
| `llo_narrow.py` | § 8 |

`kernel_work/streamindex_topk/instruments/` was **not modified**. The variant
feeds it different input by rebinding a module attribute from
`narrow_inputs.py`, never by editing an instrument. Nothing outside
`variants/narrow-record/` was touched.
