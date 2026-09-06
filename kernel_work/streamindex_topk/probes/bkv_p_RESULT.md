# Probe result: `bkv_p` at production shapes — a measured candidate, and a coupling

**Job `j-98399433-7763-4385-81ca`**, `JobStatus: Succeeded`, 4/4 shards rc=0.
Image `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.
Single chip per shard, one TensorCore.

`RUN_CMD`, as read back from `cdk job recipe j-98399433-7763-4385-81ca` (YAML line-folded; reflowed
to one line, no other change):

```
export PYTHONPATH=/workspace:$PYTHONPATH && bash /root/fanout/tpu-fanout.sh "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode sweep --regimes decode_prod,decode_64k --json /tmp/sweep_decode.json" "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode sweep --regimes mixed_prod --all-bands --json /tmp/sweep_mixed.json" "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode correct --regimes decode_prod --ladder 4,8" "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode control --regimes decode_prod --ladder 4"
```

**Provenance, stated exactly.** The `git log` read back at submit time was **`543c5159`**, with
`probes/bkv_p_prod.py` still **untracked** — the commit carrying it (`fef2e9d9`) landed a few
minutes later. The uploaded file's mtime (`2026-09-06 01:09:52`) predates the 01:11:14 submit and
its md5 is `8b217089a73a6d8237f4d6c8ff11f818`, unchanged since, so the copy that ran is
byte-identical to the committed one. Recorded rather than smoothed over: the sha in the recipe does
not contain the script it ran.

Sign convention: every percentage is a change in **time**, `(new − old)/old`. Negative is faster.

---

## Terms

- **`bkv_p`** — `num_kv_pages_per_block`, the number of 1024-position KV pages a single inner
  `(bq, bkv)` iteration loads and scores. It is a **3-tuple**, `(decode, mixed, prefill)`, selected
  per band by which of the three `pallas_call`s runs (`streamindex_topk.py:757/775/793`). Shipped
  value `(2, 2, 2)` (`mla_attention_op.py:119`). It is a static argument, so changing it is a
  one-line change and a recompile, not a kernel edit.
- **`subl`** — `num_sublanes_total`, the width of the f32 scores buffer in 128-column sublanes.
  **It is a `max` over all three tuple entries** (`streamindex_topk.py:603-605`), because one
  buffer serves all three launches. This is the coupling the whole document turns on.
- **`num_bkv`** — inner iterations over KV for the regime's own band, `cdiv(kv_len, 1024 × bkv_p)`.
- **`full ms`** — median of 10 reps of the whole `streamindex_topk` call, selection included,
  through the read-only `instruments/microbench.py`. Directly comparable to step 1's baseline table.
- **`Δ time %`** — against the shipped `(2,2,2)` arm in the same regime.

How the arms were set without editing a read-only instrument: the probe rebinds the module
attribute `shapes.NUM_KV_PAGES_PER_BLOCK` around each call. `microbench.py` and
`test_correctness.py` are imported and executed exactly as written.

## Decode: the win survives production, and grows with context

| regime | tuple | subl | scores MB | num_bkv | full ms | spread % | Δ time % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | (2,2,2) | 80 | 2.62 | 4 | 0.1682009256910533 | 0.90 | 0.00 |
| `decode_prod` | (4,2,2) | 96 | 3.15 | 2 | 0.1526356732938439 | 1.32 | **−9.25** |
| `decode_prod` | (8,2,2) | 128 | 4.19 | 1 | 0.15269711730070412 | 1.43 | −9.22 |
| `decode_64k` | (2,2,2) | 528 | 1.62 | 32 | 0.19925692467950284 | 0.49 | 0.00 |
| `decode_64k` | (4,2,2) | 544 | 1.67 | 16 | 0.17668985528871417 | 0.99 | **−11.33** |
| `decode_64k` | (8,2,2) | 576 | 1.77 | 8 | 0.17178442794829607 | 0.59 | **−13.79** |

Spreads are 0.49–1.43%, well inside every effect, so these rows separate.

Three readings:

1. **The 8192 result was not an artefact of its own confound-removal.** `step_cost.py` measured
   8.62% faster at `bkv_p=4` with the scores width pinned; production, where the width grows 20%,
   gives **9.25% faster**. The step saving beats the widening at decode by a clear margin.
2. **`bkv_p=4` and `bkv_p=8` are tied at `decode_prod`** (0.15264 vs 0.15270, a 0.04% difference
   against 1.3–1.4% spreads). The step term is exhausted by 4 — consistent with `step_cost.py`'s
   finding that the ≈500 ns cost decays to zero once steps are few. `bkv_p=8` buys nothing and
   costs a 60% wider buffer, so it is dominated.
3. **The win is larger at long context**, 11.33% at `bkv_p=4` and 13.79% at `bkv_p=8`. `decode_64k`
   starts at 32 inner iterations, so there is more per-step cost to remove, and the buffer widens
   only 3%/9% there rather than 20%/60% because 65 pages is already near a multiple of 4 and 8.
   **This is the regime the requested end-to-end A/B targets**, which makes it the relevant row.
   Note `bkv_p=8` does still win here, unlike at `decode_prod` — the ranking between 4 and 8 is
   regime-dependent.

## Mixed: no win, and the tuple couples it to decode

Mixed's balance is the opposite of decode's. It carries 1024 tokens against decode's 64, so the
scores buffer is 16× larger in absolute terms (41.9 MB vs 2.6 MB) and a 20% widening costs 8.4 MB
of extra traffic — written by the kernel and read again by the SparseCore, which area 7 and area 8
independently found reads **O(padded width)**, not O(`row_lengths`). Against that, mixed's step
saving is small: 32 inner iterations (`num_bq=8 × num_bkv=4`) falling to 16, worth
`16 × 500 ns ≈ 8 µs` on a 667 µs call, about 1.2%.

**The optimum is therefore band-dependent — and `num_kv_pages_per_block` is already a per-band
tuple, which looks like it makes that free. It does not.** `num_sublanes_total` is a `max` over the
whole tuple, so `(4,2,2)` widens the buffer to 96 sublanes for *mixed as well*, and mixed then pays
the width with **none** of the step saving.

The first job (`j-98399433`) missed exactly that arm — it ran `(4,4,4)`, which gets both effects —
so `(4,2,2)` at mixed, the arm a deployment taking `bkv_p_decode=4` would actually ship, went
unmeasured. **Job `j-3aa755dd-900b-4f31-9fa3`** (`Succeeded`, 4/4 shards rc=0, submitted at HEAD
`d258052b` with `kernel_work/` clean) measured it. `RUN_CMD` as read back from `cdk job recipe`,
reflowed from the recipe printer's YAML folding:

```
export PYTHONPATH=/workspace:$PYTHONPATH && bash /root/fanout/tpu-fanout.sh "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode sweep --regimes mixed_prod --tuples \"2,2,2;4,2,2;8,2,2;4,4,4\"" "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode sweep --regimes mixed_64k --tuples \"2,2,2;4,2,2;4,4,4\"" "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode correct --regimes decode_64k,mixed_prod --tuples \"4,2,2\"" "python3 /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode control --regimes mixed_prod --tuples \"4,2,2\""
```

| regime | tuple | subl | scores MB | num_bkv | full ms | spread % | Δ time % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixed_prod` | (2,2,2) | 80 | 41.94 | 4 | 0.6666850706096739 | 1.68 | 0.00 |
| `mixed_prod` | (4,2,2) | 96 | 50.33 | 4 | 0.6867082207463682 | 2.61 | +3.00 |
| `mixed_prod` | (8,2,2) | 128 | 67.11 | 4 | 0.7007382460869849 | 2.27 | +5.11 |
| `mixed_prod` | (4,4,4) | 96 | 50.33 | 2 | 0.6774073000997305 | 1.77 | **+1.61** |
| `mixed_64k` | (2,2,2) | 528 | 276.82 | 32 | 4.789508902467787 | 1.30 | 0.00 |
| `mixed_64k` | (4,2,2) | 544 | 285.21 | 32 | 4.875155910849572 | 0.86 | +1.79 |
| `mixed_64k` | (4,4,4) | 544 | 285.21 | 16 | 4.86955470405519 | 1.71 | **+1.67** |

The `subl`/`num_bkv` columns confirm the band indexing does what `streamindex_topk.py:603-605` says:
`(4,2,2)` at `mixed_prod` widens the buffer to 96 while leaving mixed's step count at 4, and only
`(4,4,4)` halves it to 2. No arm errored. `mixed_64k` at 544 sublanes — a 285 MB scores buffer —
was accepted by Mosaic, as was `mixed_prod` at 128 sublanes.

**This reverses the ranking the first job implied.** `(4,2,2)` is the *worse* of the two composed
tuples, not the better one: 3.00% slower at `mixed_prod` against `(4,4,4)`'s 1.61%. Once the width
is paid — and `max(tuple)` means it is paid the moment the decode band moves — mixed may as well
take its own step saving, which recovers about 1.4 points of it. `(8,2,2)` is worse again at 5.11%,
the width cost with no saving at all, and it is the only mixed row whose delta clearly exceeds
combined spread.

**Decode is unaffected by the choice between them.** `subl` is `max(tuple)`, which is 4 for both
`(4,2,2)` and `(4,4,4)`, and decode runs only the decode band, which is 4 in both. So decode's
measured −9.25%/−11.33% transfers to `(4,4,4)` as an identity, not an extrapolation: the decode
`pallas_call` sees a bit-identical configuration.

**Two limits on how hard the mixed rows may be pushed.** `+1.61%` sits against a 1.77% spread on
its own row and 1.68% on the reference, and `+1.67%` against 1.71% and 1.30%; the `Δ time %` column
does not fold in the reference's spread, so the effective uncertainty is larger than the number
shown. Neither of the `(4,4,4)` rows clears it. **"Roughly free at mixed, and not measurably
slower" is the strongest reading these rows support** — not "1.6% slower", and not "free".

Separately, `mixed_prod`'s baseline reproduced across jobs: 0.66799 ms at 5.00% spread in
`j-98399433`, 0.66669 ms at 1.68% here, agreeing to 0.20%. The earlier 5.00% was the outlier, and
the conclusions the first job could not support are now measurable.

## Correctness

Float64 oracle, rank criterion (`RANK_SLACK=256`, `MIN_RECALL=0.99`). From `j-98399433` shard 2:

```
[decode_prod tuple=(4, 2, 2)] PASS -- all 64 rows a valid top-2048 vs the float64 oracle. worst_rank=2109 of allowed 2304, min_recall=0.99561 of required 0.99
[decode_prod tuple=(8, 2, 2)] PASS -- all 64 rows a valid top-2048 vs the float64 oracle. worst_rank=2109 of allowed 2304, min_recall=0.99561 of required 0.99
```

From `j-3aa755dd` shard 2, extending it to the two remaining regimes:

```
[decode_64k tuple=(4, 2, 2)] PASS -- all 6 rows a valid top-2048 vs the float64 oracle. worst_rank=2105 of allowed 2304, min_recall=0.99512 of required 0.99
[mixed_prod tuple=(4, 2, 2)] PASS -- all 1024 rows a valid top-2048 vs the float64 oracle. worst_rank=2098 of allowed 2304, min_recall=0.99512 of required 0.99
```

Negative controls, one per job, on different regimes:

```
[decode_prod tuple=(4, 2, 2)] NEGATIVE CONTROL OK -- fired on 64 of 64 rows. worst_rank=8177 of allowed 2304, min_recall=0.49512 of required 0.99
[mixed_prod tuple=(4, 2, 2)] NEGATIVE CONTROL OK -- fired on 1024 of 1024 rows. worst_rank=8159 of allowed 2304, min_recall=0.42578 of required 0.99
```

Both fired on every row and missed by 3.5× on `worst_rank`, so neither set of PASSes is vacuous.

**Every one of those six margins is bit-for-bit what the shipped kernel returned** at step 1 —
`decode_prod` 2109/0.99561 and `decode_64k` 2105/0.99512 from `j-6d675a58-7ecd-4b44-9171`,
`mixed_prod` 2098/0.99512 and its control 8159/0.42578 from `j-5f8f4a08-20b4-4fa0-bfc3`. That is
the strongest form this evidence could take: `bkv_p` is a blocking factor and should be
value-neutral, and the check says the selected set is not merely within tolerance but *identical*
to the incumbent's at all three regimes.

**One gap, and it is the tuple this document ends up recommending.** Every correctness run above
used `(4,2,2)`, where the mixed band is unchanged at 2. **`bkv_p_mixed = 4` has never been
value-checked** — `(4,4,4)` halves mixed's `num_bkv` from 4 to 2, which is a different blocking of
the same reduction and exactly the thing the decode checks establish is safe *for the decode band*.
The argument that it must also hold at mixed is a good one and it is still an argument. This is a
required check before `(4,4,4)` is proposed anywhere, and it is one shard of one job.

## Where this stands

**A measured, correctness-checked candidate from a one-line change to a static tuple.** The best
arm is **`(4,4,4)`**, not the `(4,2,2)` the decode result alone suggests:

| tuple | `decode_prod` | `decode_64k` | `mixed_prod` | `mixed_64k` |
| --- | ---: | ---: | ---: | ---: |
| (2,2,2) shipped | 0.00 | 0.00 | 0.00 | 0.00 |
| (4,2,2) | **−9.25** | **−11.33** | +3.00 | +1.79 |
| **(4,4,4)** | **−9.25** † | **−11.33** † | **+1.61** ‡ | **+1.67** ‡ |
| (8,2,2) | −9.22 | −13.79 | +5.11 | — |

`Δ time %` against shipped; negative is faster. † decode is bit-identical between `(4,2,2)` and
`(4,4,4)` — same decode band, same `subl = max(tuple) = 4` — so the measured decode rows carry
over unchanged. ‡ inside combined spread; read as "not measurably slower", not as a 1.6% cost.

So: **9.25% faster at `decode_prod`, 11.33% at `decode_64k`, and no measurable change at either
mixed regime**, with correctness identical to the incumbent's at all three checked regimes.

Five things it is not, and none should be elided when this is quoted:

1. **Single chip, one TensorCore, microbenchmark.** A kernel-level win. GLM-5.2 runs TP=1/DP=16/EP,
   so the indexer is replicated rather than sharded and single-chip is the right denominator for
   the kernel — but this still cannot see HBM pressure from the other 77 layers, SparseCore
   contention, or rank skew behind the step barrier.
2. **Not an end-to-end number, and it must not be scaled into one.** Area 7 established the
   indexer runs on **21 of 78 layers**, not all 78 (`index_topk_freq=4`,
   `index_skip_topk_offset=3`). Step 1 put the whole block at roughly 1–3% of device time at the
   production shape.
3. **`bkv_p_mixed = 4` is not correctness-checked**, per the gap noted above. `(4,4,4)` may not be
   proposed until it is.
4. **`bkv_p=8` is dominated at `decode_prod` but wins at `decode_64k`.** There is no single best
   scalar; the choice interacts with advertised context length. `(8,8,8)` and `(8,4,4)` were not
   measured, so the long-context optimum is bracketed rather than found.
5. **The mixed rows are inside noise, and "inside noise" is not "zero".** Four separate mixed
   arms all landed on the slower side of shipped (+1.61 to +5.11), which is weak evidence that the
   true effect is a small cost rather than nothing. It does not change the recommendation — a
   sub-2% mixed cost against a 9–11% decode win is a trade worth taking at any plausible phase mix
   — but it should not be rounded to zero when the composed candidate is priced in step 4.

This is a *step-2 probe result*, not a step-3 variant. It enters the option table on the same terms
as everything else and can still lose to a design that removes the per-step cost outright rather
than amortizing it — the ≈500 ns is a property of the hand-rolled inner loop, and a kernel that
pipelines its operands through `pallas_call` rather than issuing its own DMAs may not pay it at all.
Raising `bkv_p` amortizes a cost; it does not remove it.

## A design question this hands to wave 4

The coupling is not a fact of the hardware — it is a consequence of allocating **one** scores buffer
sized by `max(tuple)`. Two moves fall out, and both belong in the option table:

- **Decouple the width from the band.** Allocate per band, or size the buffer from the band actually
  running. Then decode takes `bkv_p=4` without taxing mixed.
- **Size `num_sublanes_total` from the batch's real KV length rather than `max_model_len`.** This is
  step 1's open item and `e2e_plan.md`'s confound, and it now has a second reason to exist: it
  removes the padding that makes the widening expensive in the first place. At
  `max_model_len=66560` with a short batch it is the difference between 528 sublanes and a handful.

Both are real kernel changes rather than constant edits, which is what step 2 is for.
