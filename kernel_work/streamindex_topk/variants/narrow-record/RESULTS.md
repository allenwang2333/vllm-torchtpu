# Variant A — "Narrow Record": results

Step-3 variant of `measurement_and_design.md` § 3, option A.

**Headline, in one line each:**

- **Reader side: no result. The variant does not compile.** Mosaic rejects the
  scale-plane DMA at all four regimes. Full error in § 4. There is therefore no
  variant time, no correctness verdict, and **no confirmation of the `vmatmul`
  operand types** — the capture that would have answered the fp8 question could
  not run.
- **Writer side: a large win at prefill, not a cost.** The split scatter is
  **70.6% faster** than today's wide scatter at `T=1024`
  (`measured (j-ce74a9f2-a4cd-40cb-9e87)`). At `T=64` the measurement cannot
  support a conclusion — see § 5.
- The reader-side and writer-side numbers below are **never netted against each
  other**, per the dispatch.

**Sign convention, everywhere in this file:** every percentage is a change in
**time**, `(new − old) / old`. **Negative is faster.** Signed percentages appear
only in a column named `Δ time %`, in the same row as the two absolute times
they were computed from. Prose states the direction in words. A `speedup ×`
column, where present, is a different number: greater than 1 is faster.

**Tags:** `measured (job id)` / `modelled` / `assumed` / `unmodelled`.
`unmodelled` means the floor has no term for the quantity; it never has a number
beside it.

---

## 1. What was run

| | |
| --- | --- |
| job | `j-ce74a9f2-a4cd-40cb-9e87` |
| `JobStatus` | **`Failed`** (`state: Complete`) |
| fanout | `tpu-fanout: 2/4 shards succeeded` — shard0 rc=0, shard1 rc=0, shard2 rc=1, shard3 rc=1 |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694` |
| jax | `0.10.2`; each shard saw 2 `TpuDevice`s, i.e. correctly pinned to one chip |
| incumbent module | `/usr/local/lib/python3.12/site-packages/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py` md5 `1e221f10cbf3573aa0d2ae78c3463532` |
| variant module | `.../variants/narrow-record/streamindex_topk_narrow.py` md5 `a7128f58e0c597253be16a778ef78a17` |

The incumbent md5 in the image is **identical** to the checkout's shipped
`src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`, so
`narrow_record.diff` in this folder is a diff against exactly what ran.

`RUN_CMD`, **as read back from `cdk job recipe j-ce74a9f2-a4cd-40cb-9e87
--no-color`** (not as typed):

```
export PYTHONPATH=/workspace && bash /root/fanout/tpu-fanout.sh
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/bench_narrow.py --regimes decode_prod,decode_64k --json $CDK_OUTPUT_DIR/bench_decode.json"
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/bench_narrow.py --regimes mixed_prod,mixed_64k --json $CDK_OUTPUT_DIR/bench_mixed.json"
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/correctness_narrow.py"
  "python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/writer_probe.py --json $CDK_OUTPUT_DIR/writer.json && python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/llo_narrow.py --regime mixed_prod --arm variant --out /tmp/llo_mv && python3 /workspace/kernel_work/streamindex_topk/variants/narrow-record/llo_narrow.py --regime decode_prod --arm variant --out /tmp/llo_dv"
```

`--map-dir /home/annyan_google_com/workspaces/vllm-torchtpu/kernel_work:tpu-job:/workspace/kernel_work`
plus the fanout dir; tag `agent-jobs,narrow-record-variant`.

Shards 2 and 3 are red for the **same single cause** as shards 0 and 1: the
Mosaic rejection. Shard 3 ran the writer probe to completion first — its
numbers in § 5 are good — and then died in `llo_narrow.py`, which compiles the
variant.

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

`256 = align_to(align_to(128,128) + 1, 128)`. 129 of 256 bytes are live, so the
indexer's read of this array is amplified 1.98×.

**Variant — two arrays.**

```
cache_val    u8[num_pages, page_size//kv_packing, kv_packing, 128]
             the 128 fp8 key values, and nothing else.
             Same index arithmetic as today, minor axis 128 instead of 256.

cache_scale  u8[num_pages, page_size]
             the e8m0 scale for position p of page g at [g, p].
             Lane-major: one whole page of 1024 scales along the minor axis.
```

At the production shape (`page_size=1024`, `kv_packing=4`, 415 pages):
`cache_val` is 415 × 256 × 4 × 128 = 54.4 MB and `cache_scale` is
415 × 1024 = 0.42 MB, against 108.79 MB today. 129 live bytes per position
instead of 256 — the indexer's KV read falls by **49.6%**.

The two arrays are passed to `streamindex_topk` as a **two-tuple**
`cache_kv = (cache_val, cache_scale)`. `jax.jit` treats it as a pytree, so the
public signature is unchanged and the read-only instruments call it unmodified.
This part was confirmed working in the job: the variant arm built the split
cache and reached compilation with no call-site change.

**Writer-side change.** Today, `mla_attention_op.py:103-104` pads
`k_packed[T,129]` up to `[T,256]` and does one scatter. Under the split it does
two, into two donated buffers, in one `jit`:

```python
vc = vc.at[page, slot // kv_packing, slot % kv_packing].set(k_val)    # [T,128]
sc = sc.at[page, slot].set(k_scale)                                    # [T]
```

with `page` and `slot` computed exactly as they are today, and
`donate_argnums=(0, 1)`. The `jnp.pad` to 256 disappears. The cache allocator
must allocate two buffers instead of one.

**Lane-major is deliberate, and it is what Mosaic rejected.** The scale plane is
laid out so the DMA lands scales already along the lane axis, deleting the
`[bkv_sz, 1] → [1, bkv_sz]` transpose the incumbent pays inside `load_bkv`. That
choice is the direct cause of § 4. See § 7 for the one-line re-declaration that
keeps the same bytes in the same order and is expected to unblock it.

**Only the reader side is implemented here.** The writer side is priced in § 5,
not built. Nothing in the shared tree was modified: the variant is a copy of the
kernel inside this folder, and `narrow_record.diff` carries the change.

---

## 3. Reader side — four regimes, same job, same instruments, same shapes

All rows from `j-ce74a9f2-a4cd-40cb-9e87`, timed by the **read-only**
`instruments/microbench.py` (`WARMUP=3`, `REPS=10`, `early_exit=False`),
incumbent and variant back to back in the same process on the same built input.

`Δ time %` is `(variant − incumbent) / incumbent`; **negative is faster**. There
is no such column below, because there is no variant time to put in it.

| regime | incumbent ms | spread % | variant ms | Δ time % | speedup × | inner |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `decode_prod` | 0.166819 | 0.644 | **Mosaic rejection** | — | — | 20 |
| `decode_64k` | 0.201013 | 0.602 | **Mosaic rejection** | — | — | 20 |
| `mixed_prod` | 0.666106 | 2.064 | **Mosaic rejection** | — | — | 20 |
| `mixed_64k` | 4.797339 | 1.111 | **Mosaic rejection** | — | — | 5 |

All incumbent figures `measured (j-ce74a9f2-a4cd-40cb-9e87)`.

**Drift control** — the same incumbent, re-timed after the variant arm, third in
the order. `Δ time %` here is `(repeat − first) / first`:

| regime | incumbent ms | repeat ms | Δ time % | repeat spread % |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 0.166819 | 0.167097 | +0.17 | 0.748 |
| `decode_64k` | 0.201013 | 0.200790 | −0.11 | 0.631 |
| `mixed_prod` | 0.666106 | 0.671239 | +0.77 | 1.683 |
| `mixed_64k` | 4.797339 | 4.800038 | +0.06 | 1.788 |

Every drift is inside its own spread, so the machine did not move under the
experiment. Had there been a variant number, these rows would have licensed
reading the delta.

**Split of the incumbent call**, from the instrument's `topk`-only control:

| regime | full ms | top-k control ms | scoring by difference ms |
| --- | ---: | ---: | ---: |
| `decode_prod` | 0.166819 | 0.056313 | 0.110506 |
| `decode_64k` | 0.201013 | 0.085994 | 0.115019 |
| `mixed_prod` | 0.666106 | 0.398392 | 0.267714 |
| `mixed_64k` | 4.797339 | 3.052224 | 1.745115 |

These track step 1's full-call medians closely (167.01, 201.16, 669.11,
4762.84 µs) — the same instruments on the same shapes reproduce, which is the
only claim this table supports.

**Modelled versus measured.** § 3 of `measurement_and_design.md` predicts, at
rate 2.97 TB/s `measured`:

| regime | old µs `measured` | new µs `modelled` | Δ time % `modelled` | Δ time % measured |
| --- | ---: | ---: | ---: | --- |
| `decode_prod` | 167.01 | 144.60 | −13.42 | **not obtained** |
| `decode_64k` | 201.16 | 184.35 | −8.36 | **not obtained** |
| `mixed_prod` | 669.11 | 666.31 | −0.42 | **not obtained** |
| `mixed_64k` | 4762.84 | 4740.43 | −0.47 | **not obtained** |

The modelled 13.42% and 8.36% remain `modelled`. They are neither confirmed nor
refuted by this job, and must not be quoted as if they were.

**The split input path itself works.** The variant arm built and reported these
before compilation failed, which is why the byte accounting above can be checked
against the job log rather than against source:

| regime | `cache_val` | `cache_scale` |
| --- | --- | --- |
| `decode_prod`, `mixed_prod` | `(576, 256, 4, 128)` u8, 72.0 MiB | `(576, 1024)` u8, 0.6 MiB |
| `decode_64k`, `mixed_64k` | `(390, 256, 4, 128)` u8, 48.8 MiB | `(390, 1024)` u8, 0.4 MiB |

---

## 4. The Mosaic rejection — quoted in full

At `decode_prod` and `mixed_prod` (576 pages):

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

At `decode_64k` and `mixed_64k` (390 pages) the same error, with the tile
dimension 8 rather than 32:

```
... Failed to verify that the index at dimension 0 is divisible by the tile dimension 8. ...
  (memref<390x1024xi8, #tpu.tiled<(8,128)(4,1),[8,1]>, #tpu.memory_space<hbm>>, i32, i32) ->
  memref<1x1024xi8, #tpu.tiled<(8,128)(4,1),[8,1]>, #tpu.memory_space<hbm>>
```

Location, from the `dma_start` source trace in the job log:

```
streamindex_topk_narrow.py:197  kernel.<locals>._async_copy
streamindex_topk_narrow.py:245  kernel.<locals>._fetch_bkv      <- the scale-plane DMA
streamindex_topk_narrow.py:291  kernel.<locals>.start_fetch_bkv
streamindex_topk_narrow.py:490  kernel.<locals>.prologue
```

It fires on the `StreamIdxTC-d-bq_1-bkvp_2.2` (decode-slot) program in every
regime, because the kernel compiles all three slots.

**Cause.** Mosaic tiles the **last two** dimensions of an HBM memref. The scale
plane was declared `u8[num_pages, 1024]`, so its *leading* — page — dimension is
the tiled sublane dimension, `(32,128)` for 576 pages and `(8,128)` for 390. The
reader DMAs one page at a time, which is a **one-row slice on a tiled
dimension** at a data-dependent offset, and that is rejected outright.

This is a property of the **declared shape** of the scale plane, not of the
split itself and not of the values or their order. Per the dispatch — *"A Mosaic
or VMEM rejection is a real result: quote the full error and stop, do not retry
around it"* — no second kernel job was submitted. § 7 records the fix for the
caller to authorise.

`tpu.assume_multiple`, which the error suggests, is **not** applicable: the page
index genuinely is not a multiple of 32, so suppressing the check would produce
wrong data rather than working code.

---

## 5. Writer side — priced, not built. Never netted against § 3.

The dispatch asked for the scatter's cost to be measured if instruments could
reach it. They could, so this is **`measured (j-ce74a9f2-a4cd-40cb-9e87)`**, not
`unmodelled`.

Pure XLA, no Pallas — the same code path `mla_attention_op.py:103-104` uses, at
the production cache size `u8[415, 256, 4, 256]` (424,960 positions,
108.79 MB). Index arithmetic (`seq_id`, `pos`, `page`, `slot`) copied verbatim
from that file. `WARMUP=3`, `REPS=10`, `INNER=5`, donated buffers chained
through themselves so no allocation is timed.

Arms:

| arm | what it does |
| --- | --- |
| `W0_today_wide` | today: `jnp.pad(k_packed, [T,129] → [T,256])`, one scatter into `u8[415,256,4,256]` |
| `W1_val_only` | scatter `k_val[T,128]` into `u8[415,256,4,128]` |
| `W2_scale_only` | scatter `k_scale[T]` into `u8[415,1024]` |
| `W12_split_both` | `W1` and `W2` in one `jit`, `donate_argnums=(0,1)` — **this is what variant A's producer would run** |
| `N0/N1/N2` | donated single-element scatters on each buffer. **Not results** — they are the dispatch-and-donation floor |

### T = 1024 — prefill, `PROD_MAX_NUM_BATCHED_TOKENS`

| arm | ms | spread % | implied GB/s (r+w) |
| --- | ---: | ---: | ---: |
| `N0_noop_wide` | 0.051643 | 100.94 | — |
| `N1_noop_val` | 0.046667 | 39.56 | — |
| `N2_noop_scale` | 0.045764 | 33.39 | — |
| `W0_today_wide` | 0.842139 | 0.59 | 258.4 |
| `W1_val_only` | 0.206482 | 2.22 | 526.9 |
| `W2_scale_only` | 0.102424 | 12.30 | 8.3 |
| `W12_split_both` | 0.247234 | 1.91 | 443.5 |

| baseline | ms | candidate | ms | Δ time % | speedup × |
| --- | ---: | --- | ---: | ---: | ---: |
| `W0_today_wide` | 0.842139 | `W12_split_both` | 0.247234 | **−70.64** | **3.406** |

**The split writer is 70.6% faster than today's writer at `T=1024`.** The
`W0` and `W12` spreads are 0.59% and 1.91%, two orders of magnitude below the
effect, and both sit far above the noop floor, so this row does support a
conclusion.

The direction is the opposite of what option A's ranking assumed. The wide
scatter reads and writes 256 B per position where 129 are live, and it moves
bytes at 258 GB/s against `W1`'s 527 GB/s — it is paying the same 1.98×
amplification on the write side that motivated the read-side change, plus a
`jnp.pad`. Fusing the two scatters in one `jit` is itself worth about 20%:
`W1 + W2` separately is 0.308906 ms against `W12`'s 0.247234 ms.

### T = 64 — decode

| arm | ms | spread % |
| --- | ---: | ---: |
| `N0_noop_wide` | 0.041261 | 29.16 |
| `N1_noop_val` | 0.048221 | 34.70 |
| `N2_noop_scale` | 0.041388 | 25.31 |
| `W0_today_wide` | 0.059036 | 22.92 |
| `W1_val_only` | 0.059675 | 17.63 |
| `W2_scale_only` | 0.060921 | 17.06 |
| `W12_split_both` | 0.080732 | 35.70 |

| baseline | ms | candidate | ms | Δ time % | speedup × |
| --- | ---: | --- | ---: | ---: | ---: |
| `W0_today_wide` | 0.059036 | `W12_split_both` | 0.080732 | +36.75 | 0.731 |

**This row cannot support a conclusion, and the `+36.75` must not be quoted as a
result.** Two reasons, either sufficient:

1. The spreads (17–36%) are comparable to the effect (36.75%).
2. Every arm sits within 0.01–0.04 ms of the donated-noop floor (0.041–0.048 ms).
   At `T=64` these kernels are dispatch-bound, not scatter-bound, and `W12`'s
   extra time is most consistent with launching two scatters instead of one
   rather than with moving more bytes — it moves strictly fewer.

Discriminating this needs a probe that separates launch cost from scatter cost
at small `T` — for instance sweeping `T` from 64 to 1024 and finding where the
arms leave the floor. Not run.

**These writer figures are not netted against anything in § 3, and must not
be.** A single combined figure would hide the term A's ranking turns on.

### Caveats specific to § 5

- `W0` at 0.842 ms is not comparable to the ~291 µs `measured` for
  `scatter_fusion` in the step-1 profile: different `T`, different surrounding
  fusion, different pass. The **ratio** `W12/W0` in one job is the result here;
  the absolute is not a substitute for the profile's.
- `W2` scatters into `u8[415,1024]` — the 2-D scale plane that § 4 rejected on
  the *reader* side. Under the § 7 re-declaration the writer would instead do
  `sc.at[page, slot // 128, slot % 128].set(k_scale)`: identical bytes, identical
  count, one extra integer op on the index. The cost is expected to carry over
  unchanged, but that form was **not measured**.
- One `jit`ted scatter in isolation is not the producer in situ. Whether XLA
  fuses `W12` into its neighbours the way it fuses `W0` today is `unmodelled`.

---

## 6. What was not obtained, and why

Recording these as gaps rather than leaving them to be inferred.

**MXU operand types: NOT CONFIRMED.** The dispatch is explicit — *"Check the
lowering — confirm the `vmatmul` operand types on your variant, do not infer
them from the source."* `llo_narrow.py` was written to do exactly that, reading
`tpu.matmul` operand types out of `*post-infer-vector-layout.txt` and `vmatmul`
opcode mnemonics out of `*-71-final_bundles.txt`, the same two levels step 2's
P1 used. It could not run: it compiles the variant, and the variant does not
compile. **Variant A's operand types are unknown.** Step 2's P1 finding — that
the incumbent reaches the MXU as native `f8E4M3FN × f8E4M3FN → f32` with 1064
`vmatmul` and zero bf16 ops, and that any bf16-operand formulation is a 1.86×
regression — stands for the **incumbent** and says nothing about this variant.
Nothing here licenses claiming A stays in fp8.

**Correctness: NOT OBTAINED.** `correctness_narrow.py` would have run three
checks — bit-exact `np.array_equal` of the returned index arrays against the
incumbent on the same input; the read-only `instruments/test_correctness.py`
oracle rank criterion (`RANK_SLACK=256`, `MIN_RECALL=0.99`, `k=2048`) on the
variant; and the negative control at `decode_prod` and `mixed_prod`. It printed
its header (`RANK_SLACK=256 MIN_RECALL=0.99 k=2048`) and died on the first
variant call with the § 4 error. **There is no `worst_rank=` and no
`min_recall=` to quote for this variant** — not a passing one and not a failing
one. Any correctness statement about variant A would be fabricated.

For reference, the **incumbent's** step-1 margins (`report.md`, jobs
`j-6d675a58-7ecd-4b44-9171` and `j-5f8f4a08-20b4-4fa0-bfc3`) were
`decode_prod` `worst_rank=2109` / `min_recall=0.99561`, `decode_64k` `2105` /
`0.99512`, `mixed_prod` `2098` / `0.99512`, with negative controls firing at
`8177` / `0.49512` and `8159` / `0.42578`. These are the incumbent's and are
quoted only to show the bar the variant has not been held to.

---

## 7. Recommended next step — for the caller to authorise

Not acted on. The dispatch says to stop at a Mosaic rejection, and this was not
retried.

The rejection in § 4 is about the **declared tiling** of the scale plane, not
about splitting the record. Declaring the same bytes in the same order as a
3-D array makes the page axis untiled and the per-page slice legal:

```
cache_scale  u8[num_pages, page_size // 128, 128]      # e.g. u8[576, 8, 128]
             scale of position p of page g at [g, p // 128, p % 128]
```

Mosaic tiles the last two dimensions, so `(8, 128)` is now the tile and it is
**exactly one page of scales** — the per-page DMA is a whole number of tiles,
moves 1024 B, and wastes nothing. Dimension 0 is untiled, so
`cache_scale_hbm_ref.at[pl.ds(page, 1)]` is a legal slice. The bytes on the wire
and their order are unchanged, so **the writer-side format spec in § 2 is
unaffected** beyond splitting `slot` into `slot // 128, slot % 128`.

The reader then needs, inside `load_bkv`, a `[bkv_p, 8, 128] → [1, bkv_sz]`
regroup where the incumbent does a `[bkv_sz, 1] → [1, bkv_sz]` transpose. Do the
`e8m0 → f32` widen **before** the regroup, so the relayout is on an f32 `(8,128)`
tile. e8m0 is a power of two and both bf16 and f32 represent it exactly, so this
stays bit-identical to the incumbent's arithmetic. Whether Mosaic accepts that
regroup, and what it costs, is `unmodelled` — it is the thing the next job would
find out.

This is a one-job change: the same four-shard fanout, unmodified. It is the
cheapest way to convert this write-up's "no result" into the reader-side number
the ranking needs, and § 5 has already made that number more valuable than the
ranking assumed.

Also worth carrying, independent of whether A ever ships: **§ 5 shows today's
writer is leaving roughly 70% of its prefill time on the table**, and
`measurement_and_design.md` § 11 item 1 already lists that scatter (~291 µs
`measured`, × 21 layers ≈ 6.1 ms/step) as a carried item. The split writer is
faster at `T=1024` *even without* any reader-side change, since `cache_val` and
`cache_scale` can be gathered back into the wide format — or, more simply, the
`jnp.pad`-to-256 can be examined on its own.

---

## 8. Surviving caveats

- **The reader side has no measurement.** Every § 3 variant cell reads "Mosaic
  rejection", and nothing in this file should be read as evidence for or against
  option A's reader-side prize.
- **Single-chip microbenchmark, one TensorCore.** § 3 and § 5 are both
  `run_regime`-class and XLA-class measurements on one chip of a four-chip pod,
  with three other shards running concurrently. A win here would be a
  kernel-level win and would have to be labelled one; it is not an end-to-end
  number and does not predict one.
- **The input was synthetically pre-split.** `narrow_inputs.split_cache` derives
  `cache_val` and `cache_scale` from the read-only instrument's packed cache
  (asserting bytes 129..255 are zero). No producer wrote this format. The
  reader-side experiment therefore measures the reader in isolation, which is
  the point, but it also means the two sides have never run together.
- **HEAD drift.** The dispatch names HEAD `950974c2`. The checkout was
  `11f26d75` when this work began, `9f221354` when the job was submitted, and
  `879b5d23` at the time of writing — other agents are committing to the shared
  `annyan/streamidx-scores` branch concurrently. The variant is a self-contained
  copy under this folder and the incumbent md5 matched the image, so the
  comparison in § 3 is internally consistent regardless; but the diff in
  `narrow_record.diff` should be re-checked against `src/` before it is applied.
- **`T=64` writer row is not a result**, for the two reasons in § 5.
- **Neither `/load-experiment` nor `/log-experiment` was run**, per the
  dispatch. Nothing here has gone into the shared experiment log.

---

## 9. Files in this folder

| file | what it is |
| --- | --- |
| `streamindex_topk_narrow.py` | the variant kernel, a self-contained copy. md5 `a7128f58e0c597253be16a778ef78a17` |
| `narrow_record.diff` | that file as a unified diff against the shipped `src/.../streamindex_topk.py` |
| `narrow_inputs.py` | derives the split arrays from the read-only instrument's output; `patched_inputs()` rebinds `instruments.inputs.build` |
| `_load.py` | file-path loader — this folder's name has a hyphen and cannot be a package |
| `bench_narrow.py` | the A/B driver; three arms per regime through the read-only `microbench.run_regime` |
| `correctness_narrow.py` | the three correctness checks. Did not reach a verdict |
| `writer_probe.py` | § 5. Ran to completion |
| `llo_narrow.py` | the operand-type capture. Did not run |

`kernel_work/streamindex_topk/instruments/` was **not modified**. The variant
feeds it different input by rebinding a module attribute from
`narrow_inputs.py`, never by editing an instrument. Nothing outside
`variants/narrow-record/` was touched.
