# Option B — Right-Size — step-3 results

**Verdict: build it. B is the largest measured win in step 3 so far, and the design
under-predicted it by roughly an order of magnitude at the shape it was designed for.**

Everything here is one job.

| | |
| --- | --- |
| job | `j-d9502595-54a2-45bc-947b`, `JobStatus=Succeeded`, 4/4 shards `rc=0` |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694` |
| checkout | `/home/annyan_google_com/workspaces/vllm-torchtpu`, branch `annyan/streamidx-scores` |
| variant commit | `9bc5f5cd` (the variant and its drivers) |
| instrument commit | `36aedcdc` (the sixth regime — **separate on purpose**, see §2) |
| `~/cloud-devkit` | `4c6383f`, 1 dirty file, captured at submit |
| hardware | one TPU v7x chip per shard (4-chip pod, fanned out); **single-chip microbenchmark, not an end-to-end number** |

`RUN_CMD` as read back from `cdk job recipe j-d9502595-54a2-45bc-947b --no-color`:

```
bash /root/fanout/tpu-fanout.sh "bash /workspace/kernel_work/streamindex_topk/variants/right-size/shard.sh 0" "bash /workspace/kernel_work/streamindex_topk/variants/right-size/shard.sh 1" "bash /workspace/kernel_work/streamindex_topk/variants/right-size/shard.sh 2" "bash /workspace/kernel_work/streamindex_topk/variants/right-size/shard.sh 3"
```

`--map-dir` ×3: `…/vllm-torchtpu/kernel_work:tpu-job:/workspace/kernel_work`,
`…/vllm-torchtpu/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`,
`…/vllm-optimization-agent/.claude/skills/run-command-in-cdk/fanout:tpu-job:/root/fanout`.
Tag `agent-jobs,variant-right-size`. Every shard printed
`vllm_torchtpu.__file__ = /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/__init__.py`,
so the mapped `src/` is the one that ran.

**Sign convention, used everywhere below.** `Δ time % = (variant − incumbent) / incumbent × 100`.
**Negative is faster.** Signed percentages appear only in tables, in a column named `Δ time %`.
Prose states the direction in words. A `speedup ×` column sits alongside; `>1` is faster and it is
a different number. Every delta is in the same row as the two absolute times it came from.

**Tagging.** Every number is `measured (job id)`, `modelled`, `assumed` or `unmodelled`.
`unmodelled` means the roofline has no term for it — the word appears with no number beside it.

---

## 1. What changed, and why

`streamindex_topk` allocates its scores buffer at `num_sublanes_total`, derived from
`max_model_len`:

```python
num_sublanes_total = max(align_to(pages_per_seq, bkv_p) * page_size // 128
                         for bkv_p in num_kv_pages_per_blocks)
```

and prefills the whole thing with `-inf` through an `input_output_aliases` operand, so that lanes
the kernel never writes cannot win the top-k. Both the width and the fill are paid at
`max_model_len`, whatever the batch's real `kv_len` is.

B sizes the buffer from the live `kv_len` and deletes the fill. Full diff in `variant.diff`;
variant source in `streamindex_topk_rs.py` (`md5 6c71d7b5c5738a2eccfebc97bdf98d40` on the pod).
The incumbent (`md5 1e221f10cbf3573aa0d2ae78c3463532`) is **not modified**;
`sparsecore_topk.py` is **not modified**. Five substantive pieces:

1. **Width from `kv_len`.** New static argname `kv_len_bucket: int | None`. When set,
   `num_sublanes_total = max(cdiv(min(kv_len_bucket, pages_per_seq*page_size), 128), cdiv(k, 128))`.
   When `None` the incumbent expression runs unchanged, so the file is a strict superset.
2. **No fill.** `scores_init = None` when a bucket is given; the aliased input operand,
   its `in_specs` entry and its `input_output_aliases` entry are all dropped, and `_body`
   splices `None` into the `scores_in_hbm_ref` slot.
3. **Masked partial-block store.** The width no longer has to be a multiple of `bkv_sz`, so the
   last `bkv` block can be partial. `start_send_scores` computes `is_tail = bkv_idx == n_full_slabs`
   and takes one of two `pl.when` branches; the tail branch stores `n_tail_sublanes` sublanes at the
   **static** destination offset `pl.ds(n_full_slabs * num_sublanes_bkv, n_tail_sublanes)`.
   Static, not the traced `sublane_start`, because per `references/hardware-facts.md` a traced
   offset the compiler cannot prove aligned is what fails to lower.
4. **A partial-flag prefetch.** `bo_partial_ref` (7th SMEM scalar prefetch) records whether the
   block whose DMA is in flight was the tail, so `wait_send_scores` waits on the same extent that
   was sent. Verified by reading: `wait_send_scores(bo_sem_idx)` at :521 runs *before*
   `start_send_scores(...)` at :525, so the flag is read before it is overwritten. It follows
   `bo_sz_ref`'s lifetime (initialised to `-1`, never reset), which is what makes the cross-grid-step
   wait work.
5. **A test hook.** `poison_fill: float | None` reinstates the aliased prefill with an arbitrary
   constant instead of `-inf`. It exists only for the poison arm in §4 and is `None` in the
   measured variant.

**A byte-identity guard inside the variant.** When `n_tail_sublanes == 0` — which is every one of
the five step-1 regimes and the confound, since their `kv_len` is a multiple of `bkv_sz = 2048` —
both `start_send_scores` and `wait_send_scores` early-return on code that is byte-for-byte the
incumbent's, and `bo_partial_ref` is neither written nor read. The masked store is not merely
inactive there; it is not emitted. This is why the masked store needed a shape of its own to be
tested at all (§4).

**The masked store lowered as expected.** No Mosaic rejection, no VMEM rejection, no retry: the
`mask_mixed_8000` / `mask_decode_8000` arms compiled and ran first time at width 63 sublanes
(3 full slabs + a 15-sublane tail). There is no error text to quote because none was produced.

### Where the implementation is more aggressive than the specification

`measurement_and_design.md` §4.3(a) models the rounding at **page** granularity — 80 → 72
sublanes at the prod shapes, 528 → 520 at 64k. This implementation rounds to **128 columns**
(1 sublane): 8192/128 = **64** sublanes, 65536/128 = **512**. That is strictly narrower than the
spec's arithmetic, and it is one of the two reasons the measured wins exceed §4.3(a)'s
predictions (the other is in §5). It is a deliberate divergence and it is stated here rather than
followed quietly. It is safe because `_pick_partition` (`sparsecore_topk.py:360`) needs only
`n ≥ k` and, for the doubling branch, divisibility — any multiple of 128 at or above `k = 2048`
is a legal `n`. Rounding to 32 sublanes instead would have made `n_tail_sublanes` always zero and
the masked store dead code, so 128 is also the granularity at which the store earns its keep.

---

## 2. The sixth regime is an instrument addition, not part of B

`extra_regimes.py`, committed **separately** as `36aedcdc`, adds three `Regime` records:

| name | phase | `max_num_seqs` | `num_tokens` | `kv_len` | `max_model_len` |
| --- | --- | --- | --- | --- | --- |
| `mixed_confound` | mixed | 6 | `PROD_MAX_NUM_BATCHED_TOKENS` | 8192 | 66560 |
| `mask_mixed_8000` | mixed | 64 | 1024 | 8000 | 9216 |
| `mask_decode_8000` | decode | 64 | 64 | 8000 | 9216 |

`mixed_confound` is §12 item 4's confound shape: it separates `kv_len` from `max_model_len`, which
none of step 1's five regimes do, and **B cannot be measured at all without it**. The two
`*_8000` shapes exist so the masked store fires (§1). Every later option benefits from a regime
that exercises the `kv_len / max_model_len` ratio — option C's prize is on the same bytes — which
is why this is committed on its own and belongs to the instrument set, not to B.

**`instruments/` was not touched.** `shapes.py` is unmodified. This was possible because
`microbench.run_regime` and `test_correctness.run` take a `Regime` *instance* and a module
*object*, so a regime defined in my own module and a variant bound with extra static kwargs both
pass straight in. The md5s printed in every shard log:

```
ea50f2829149e22eedbd9faed86ba85a  instruments/microbench.py
d7bd611b874f47c8050dc73a79da125b  instruments/test_correctness.py
8a9ac175495e73cf7b812cf6c6bda614  instruments/shapes.py
```

Since the five regimes were not edited, the "prove they are unmoved" re-run was not required —
but it came for free anyway, because shard 0 measures the incumbent arm at four of them in this
job. Against `report.md`'s step-1 numbers:

| regime | step-1 `report.md` | this job, incumbent arm (`ee=False` / `ee=True`) | agreement |
| --- | --- | --- | --- |
| `decode_prod` | 0.16701 ms, spread 0.57% | 0.16665 / 0.16805 ms | within spread |
| `decode_64k` | 0.20116 ms, spread 0.58% | 0.20079 / 0.20305 ms | within spread |
| `mixed_prod` | 0.66911 ms, spread 2.84% | 0.67398 / 0.67118 ms | within spread |
| `mixed_64k` | 4.76284 ms, spread 1.13% | 4.75884 / 4.78598 ms | within spread |

All `measured (j-d9502595-54a2-45bc-947b)` against `measured (j-6d675a58-7ecd-4b44-9171)` /
`measured (j-5f8f4a08-20b4-4fa0-bfc3)`.

---

## 3. Performance

Full-call time, single TensorCore, both arms in the **same process**, through the read-only
`instruments/microbench.run_regime`, on the same inputs, with the same `inner`, `reps` and warmup.
All `measured (j-d9502595-54a2-45bc-947b)`. **Negative `Δ time %` is faster.**

| regime | `ee` | cols inc | cols B | incumbent ms | sp % | B ms | sp % | Δ time % | speedup × | effect vs spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_prod` | False | 10240 | 8192 | 0.16665 | 0.56 | 0.16480 | 1.07 | −1.11 | 1.011 | **not conclusive** |
| `decode_prod` | True | 10240 | 8192 | 0.16805 | 0.75 | 0.16747 | 0.69 | −0.35 | 1.003 | **not conclusive** |
| `decode_64k` | False | 67584 | 65536 | 0.20079 | 1.01 | 0.19643 | 0.31 | −2.17 | 1.022 | weak |
| `decode_64k` | True | 67584 | 65536 | 0.20305 | 0.53 | 0.20014 | 0.50 | −1.43 | 1.015 | weak |
| `mixed_prod` | False | 10240 | 8192 | 0.67398 | 2.52 | 0.65610 | 2.26 | −2.65 | 1.027 | **not conclusive** |
| `mixed_prod` | True | 10240 | 8192 | 0.67118 | 2.53 | 0.65050 | 1.79 | −3.08 | 1.032 | **not conclusive** |
| `mixed_64k` | False | 67584 | 65536 | 4.75884 | 1.16 | 3.75686 | 1.13 | −21.06 | 1.267 | decisive |
| `mixed_64k` | True | 67584 | 65536 | 4.78598 | 1.15 | 3.74624 | 1.12 | −21.72 | 1.278 | decisive |
| **`mixed_confound`** | False | 67584 | 8192 | 2.28219 | 2.20 | 0.61891 | 0.77 | **−72.88** | **3.687** | decisive |
| **`mixed_confound`** | True | 67584 | 8192 | 2.27933 | 1.85 | 0.61196 | 0.59 | **−73.15** | **3.725** | decisive |

`ee` is `enable_early_exit`. The `topk` control arm of `run_regime` is disabled here
(`measure_topk=False`) because it sizes its buffer from `S.num_sublanes_total(regime.max_model_len)`
for *both* arms, so it cannot see B; the SparseCore side is measured explicitly instead (§3.1).

**Rows that cannot support a conclusion, said rather than ranked.** At `decode_prod` and
`mixed_prod` the effect is at or below the spread of the two arms it is a difference of. Those four
rows are consistent with B doing nothing at those shapes and equally consistent with it doing what
§4.3(a) modelled; this job cannot tell them apart. `decode_64k` sits at 2–4× the spread — weakly
supported, not established. Only `mixed_64k` and `mixed_confound` are decisive here.

**Headline: at the confound shape B is 73.15% faster, 3.725×, 2.27933 → 0.61196 ms.** That is the
row the design could not have, and it is the row that decides B.

### 3.1 `sparsecore_topk` alone — a COMPONENT, not a result

Standalone timing of `sparsecore_topk` at the two widths, driven directly. This is reported only
as an *attribution* of the full-call delta above; it is not a result on its own, and it is on
separately-built inputs, so subtracting it from the full call is an attribution and not a
decomposition of a single execution. All `measured (j-d9502595-54a2-45bc-947b)`.

| regime | `n` inc | P | inc ms | sp % | `n` B | P | B ms | sp % | Δ time % | speedup × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mixed_confound` | 67584 | 2 | 1.96386 | 0.90 | 8192 | 1 | 0.40939 | 2.13 | −79.15 | 4.797 |
| `mixed_64k` | 67584 | 2 | 3.03536 | 3.09 | 65536 | 1 | 2.26422 | 2.93 | −25.41 | 1.341 |
| `mixed_prod` | 10240 | 1 | 0.41179 | **885.74** | 8192 | 1 | 0.38538 | 1.22 | −6.41 | 1.069 |
| `decode_64k` | 67584 | 4 | 0.08633 | 0.88 | 65536 | 4 | 0.08585 | 1.95 | −0.55 | 1.006 |

**The `mixed_prod` row is unusable.** Its incumbent spread of 885.74% is two orders of magnitude
larger than the −6.41% effect. It is reported so it is not silently dropped, and it is **not**
ranked, quoted or carried into any conclusion.

**The `P` columns are the finding.** `_pick_partition(b, n, k)` derives the row-slice count from
`n`. At both `mixed_confound` and `mixed_64k`, narrowing the buffer moves the call from the
two-stage path (stage-1 slices → gather → stage-2 merge) to the one-stage `_sc_topk_direct` path.
`mixed_64k` isolates it: the width falls only 3% (67584 → 65536) and the kernel gets 25.41%
faster. That is almost entirely the path flip, not the bytes — the incumbent's rounding pushes
`n` just past `MAX_SLICE_WORDS = 65536` and forces two stages, and B's rounding lands it just
under.

This term is **`unmodelled`** — §4.3 has no term for a partition-count change, only for bytes.

---

## 4. Correctness

Every arm is judged by the read-only float64 oracle and rank/recall criterion in
`instruments/test_correctness.py` (`worst_rank ≤ k + RANK_SLACK = 2304`, `min_recall ≥ 0.99`).
The oracle is built **once per regime and shared**, so all four arms are scored against literally
the same reference. Margins verbatim from the job log. All `measured (j-d9502595-54a2-45bc-947b)`.

`set_diff` is the number of rows whose returned index **set** differs from the incumbent's,
ignoring order and `-1` padding — a strictly stronger check than the rank criterion, and the only
one that can show §4.4 claim (1)'s *bit-exact*.

| regime | arm | cols | verdict | worst_rank | allowed | min_recall | req | set_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_prod` | incumbent | 10240 | PASS | 2109 | 2304 | 0.99561 | 0.99 | – |
| `decode_prod` | right_size | 8192 | PASS | 2109 | 2304 | 0.99561 | 0.99 | **0** |
| `decode_prod` | poison | 8192 | PASS | 2109 | 2304 | 0.99561 | 0.99 | **0** |
| `decode_prod` | neg_control | 8192 | FAIL *(required)* | 8177 | 2304 | 0.49512 | 0.99 | 64 |
| `decode_64k` | incumbent | 67584 | PASS | 2105 | 2304 | 0.99512 | 0.99 | – |
| `decode_64k` | right_size | 65536 | PASS | 2105 | 2304 | 0.99512 | 0.99 | **0** |
| `decode_64k` | poison | 65536 | PASS | 2105 | 2304 | 0.99512 | 0.99 | **0** |
| `decode_64k` | neg_control | 65536 | FAIL *(required)* | 62644 | 2304 | 0.26953 | 0.99 | 6 |
| `mixed_prod` | incumbent | 10240 | PASS | 2098 | 2304 | 0.99512 | 0.99 | – |
| `mixed_prod` | right_size | 8192 | PASS | 2098 | 2304 | 0.99512 | 0.99 | **0** |
| `mixed_prod` | poison | 8192 | PASS | 2098 | 2304 | 0.99512 | 0.99 | **0** |
| `mixed_prod` | neg_control | 8192 | FAIL *(required)* | 8159 | 2304 | 0.42578 | 0.99 | 1024 |
| `mask_decode_8000` | incumbent | 10240 | PASS | 2089 | 2304 | 0.99512 | 0.99 | – |
| `mask_decode_8000` | right_size | 8064 | PASS | 2089 | 2304 | 0.99512 | 0.99 | **0** |
| `mask_decode_8000` | poison | 8064 | PASS | 2089 | 2304 | 0.99512 | 0.99 | **0** |
| `mask_decode_8000` | neg_control | 8064 | FAIL *(required)* | 7986 | 2304 | 0.49512 | 0.99 | 64 |
| `mixed_confound` | incumbent | 67584 | PASS | 2105 | 2304 | 0.99463 | 0.99 | – |
| `mixed_confound` | right_size | 8192 | PASS | 2105 | 2304 | 0.99463 | 0.99 | **0** |
| `mixed_confound` | poison | 8192 | PASS | 2105 | 2304 | 0.99463 | 0.99 | **0** |
| `mixed_confound` | neg_control | 8192 | FAIL *(required)* | 8165 | 2304 | 0.44141 | 0.99 | 1024 |
| `mixed_confound` | stale_after_poison | – | PASS | – | – | – | – | **0 of 1024** |
| `mask_mixed_8000` | incumbent | 10240 | PASS | 2103 | 2304 | 0.99512 | 0.99 | – |
| `mask_mixed_8000` | right_size | 8064 | PASS | 2103 | 2304 | 0.99512 | 0.99 | **0** |
| `mask_mixed_8000` | poison | 8064 | PASS | 2103 | 2304 | 0.99512 | 0.99 | **0** |
| `mask_mixed_8000` | neg_control | 8064 | FAIL *(required)* | 7968 | 2304 | 0.43555 | 0.99 | 1024 |
| `mask_mixed_8000` | stale_after_poison | – | PASS | – | – | – | – | **0 of 1024** |

For `neg_control` the **required** verdict is FAIL; the driver exits non-zero if any negative
control passes, because a criterion that has never failed is not evidence. All five fired, at
ranks 7968–62644 against an allowance of 2304.

At every regime `right_size` reproduces the incumbent's `worst_rank` and `min_recall` **to the
digit** and its index sets are identical on every row. §4.4 claim (1) — exact sizing is bit-exact,
not merely within slack — is established at six shapes, including two the masked store fires at.

### 4.1 The stale-index test — designed before the store was written

This is the failure mode that makes B *wrong* rather than *slow*: deleting the `-inf` prefill
removes the thing that made an unwritten lane lose the top-k, so if the masked store does not
cover exactly the region the prefill used to, the kernel selects **stale indices from a previous
step**. Those are plausible, not random, so neither a throughput number nor a negative control
built around a ranking change is guaranteed to see them — and even a rank check may not, since
stale scores are drawn from the same distribution and a handful can land inside the 256-rank slack.

**`poison` is the test, and it is a strict one rather than a probabilistic one.** It reinstates the
aliased prefill with `+1e30` instead of `-inf`. Every lane the kernel does not write is then the
largest value in its row, so *any* read of such a lane does not perturb the ranking — it takes over
the top-k and drives `worst_rank` to the end of the row. The poison arm's margins are bit-identical
to the clean arm's at all six regimes (rows above), and its index sets match the incumbent on every
row. **That is proof that no unwritten lane is reachable by selection**, decisive in one run.
This is §4.4 claim (2)'s check, and the design's trap is noted: `instruments/test_correctness.py`'s
plain pass is *not* the gate here — the poison arm is.

**`--stale` is the weaker, more literal corroboration**, run at the two mixed shapes: execute the
poisoned arm, drop its output, then run the clean variant on **different** inputs (seed 7 rather
than 0) and require its index sets to match the incumbent's on those inputs. If the allocator
hands the second call the first call's `1e30` buffer and the kernel reads any of it, the sets
diverge. Both runs: **0 of 1024 rows differ.** Best-effort by nature — nothing forces the
allocator's hand — so it corroborates `poison`, it does not substitute for it.

### 4.2 Coverage for `sparsecore_topk`, which had none

§11 item 4: `grep -rn "sparsecore_topk" tests/` returns nothing. There is no unit test under that
kernel at all, and **the `streamindex_topk` rank/recall gate does not stand in for one.** What
that gate does and does not see:

- It sees end-to-end index quality on `streamindex_topk`'s own generated inputs, at whatever
  `(b, n, P)` those inputs happen to produce.
- It does **not** see `sparsecore_topk` at a width `streamindex_topk` does not drive it at; it does
  not distinguish a one-stage from a two-stage call; it does not exercise ragged or empty
  `row_lengths` beyond what `_effective_row_lengths` emits for these six regimes; and — decisively
  — it runs with `-inf` in the padded region, so it would pass whether or not the
  `valid = (j < i_hi)` bound at `:151-155`/`:171` actually holds.

B does not edit `sparsecore_topk.py`. It changes the `n` that kernel is *driven* at, and `n` is not
inert there: `_pick_partition` derives P from it, and `slice_len`, the stage-1 `b*P × n/P` reshape
and the resident-VMEM DMA extent all move with it. So the thing to test is the shapes.
`run_sc_coverage.py` drives `sparsecore_topk` directly against a numpy oracle at exactly the widths
and partition factors both arms produce, **with the padded region filled with `+1e30` rather than
`-inf`** — garbage that would be selected on sight, so the test passes only if the
`row_lengths` bound holds. **All 12 cases PASS, `rows_wrong = 0` and
`rows_selecting_past_row_length = 0` in every one:**

| case | b | n | live | ragged | P |
| --- | --- | --- | --- | --- | --- |
| `confound/incumbent` | 1024 | 67584 | 8192 | no | 2 |
| `confound/right_size` | 1024 | 8192 | 8192 | no | 1 |
| `confound/right_size ragged` | 1024 | 8192 | 8192 | yes | 1 |
| `mixed_prod/incumbent` | 1024 | 10240 | 8192 | no | 1 |
| `mixed_prod/right_size` | 1024 | 8192 | 8192 | yes | 1 |
| `mask8000/right_size` | 1024 | 8064 | 8000 | yes | 1 |
| `mixed_64k/incumbent` | 1024 | 67584 | 65536 | no | 2 |
| `mixed_64k/right_size` | 1024 | 65536 | 65536 | no | 1 |
| `decode_prod/incumbent` | 64 | 10240 | 8192 | no | 1 |
| `decode_prod/right_size` | 64 | 8192 | 8192 | yes | 1 |
| `decode_64k/incumbent` | 6 | 67584 | 65536 | no | 4 |
| `decode_64k/right_size` | 6 | 65536 | 65536 | no | 4 |

The ragged cases vary the live length per row and zero the first `b/16` rows, which is the shape
`_effective_row_lengths` (`streamindex_topk.py:592`) emits for a real batch and the
`write_empty_rows` path.

### 4.3 What correctness here does **not** cover

- **The SparseCore half of B as designed.** §4.3(b)'s "padded read sized to `row_lengths`" needs a
  source change to `sparsecore_topk.py`, which §4.6 says should not be attempted before coverage
  exists. `run_sc_coverage.py` is now that coverage; the source change is deliberately **not made
  here**. Nothing in this document measures or verifies it.
- **Numerical behaviour under real, near-tied score data** at these widths, beyond what the
  random draw and the generated inputs happen to produce.
- **End-to-end model output.** This is a kernel microbenchmark. No accuracy check, no eval, no
  serving run.
- **Recompilation cost.** The driver passes the exact `kv_len` as the bucket. A shipping version
  needs the bucketed jit-static width ladder of §4.1 area 2 (bucket count `assumed`), and the
  correctness evidence above does not cover a bucket boundary other than the six widths measured.

---

## 5. Modelled versus measured

`modelled` figures are `measurement_and_design.md` §4.3; `measured` are
`measured (j-d9502595-54a2-45bc-947b)`, `ee=True`. **Negative is faster.**

| regime | incumbent ms | B ms | Δ time % modelled | Δ time % measured | measured / modelled |
| --- | --- | --- | --- | --- | --- |
| `decode_prod` | 0.16805 | 0.16747 | −0.63 | −0.35 *(≤ spread)* | not resolvable |
| `mixed_prod` | 0.67118 | 0.65050 | −2.53 | −3.08 *(≈ spread)* | not resolvable |
| `decode_64k` | 0.20305 | 0.20014 | *(not modelled)* | −1.43 | — |
| `mixed_64k` | 4.78598 | 3.74624 | −2.02 | **−21.72** | **10.8×** |
| `mixed_confound` | 2.27933 | 0.61196 | ≤ −5.40 *(see below)* | **−73.15** | **13.5×** |

### 5.1 The bracket is closed

§4.3(b) wrote the confound prize as `4762.84 → 4505.84, ≤ −5.40`. The `≤` is there because the
denominator is deliberately wrong: `mixed_64k`'s full call, 4762.84 µs, used as an upper-bound
stand-in because no job had ever run the confound shape. **The real denominator is 2279.33 µs
`measured`.** Two corrections follow, in opposite directions and both large:

- Against the real denominator, the design's own TC-only arithmetic (257.0 µs `modelled`) would
  read **−11.3%**, not −5.40%. Half the conservatism was the denominator.
- The measured saving is **1667.4 µs**, against a modelled 257.0 µs TC subtotal plus a
  0.7–0.9 ms `modelled` SparseCore row — i.e. 0.957–1.157 ms if both had been shipped, and only
  257.0 µs for the TC half that actually was. The measurement is **6.5× the TC-half model** and
  **1.4–1.7× the model with the deferred SparseCore half included**.

### 5.2 Which assumption broke — attribution

Subtracting the standalone `sparsecore_topk` timings (§3.1) from the full-call deltas. This is an
**attribution across two separate measurements on separately-built inputs**, not a decomposition of
one execution; treat the residual as indicative.

| regime | full-call saved | of which `sparsecore_topk` | residual (TensorCore side) | TC-side `modelled` |
| --- | --- | --- | --- | --- |
| `mixed_confound` | 1667.4 µs | 1554.5 µs | 112.9 µs | 93.2 µs (deleted `-inf` fill) |
| `mixed_64k` | 1039.7 µs | 771.1 µs | 268.6 µs | 93.2 µs (deleted `-inf` fill) |

Three findings, in order of how much they matter:

1. **The `_pick_partition` P=2 → P=1 flip is the dominant term, and it is `unmodelled`.**
   1554.5 µs of the confound's 1667.4 µs lands in `sparsecore_topk`, against a `modelled` range of
   **0.7–0.9 ms** — 1.7–2.2× more than modelled. §4.3 priced the SparseCore side in *bytes*; the
   width change also collapses a two-stage top-k into a one-stage one, and the roofline has no term
   for that. `mixed_64k` isolates it cleanly: 3% fewer bytes, 25.41% faster. This is the single
   biggest reason the design under-predicted B.

2. **The design's "TC subtotal 257.0 µs" double-counts the SparseCore read.** Its 486.53 MB /
   163.8 µs line is "TC write **+ SC read** at live width", and the SC-read half of that lands in
   the `sparsecore_topk` column, not the TensorCore one. The genuinely TensorCore-side term is the
   deleted fill alone: 276.82 MB / **93.2 µs `modelled`** versus **112.9 µs** residual measured at
   the confound. Those agree within about 20%, which is good for a roofline. The TC model was
   fine; its bookkeeping put an SC term on the TC row.

3. **The fill's measured cost is not consistent between the two shapes** — 112.9 µs at the
   confound against 268.6 µs at `mixed_64k`, for the same 276.82 MB. Both are residuals of an
   attribution rather than direct measurements, and at `mixed_64k` the variant still allocates a
   268 MB buffer while at the confound it allocates 33.55 MB, so allocation and first-touch
   behaviour differ. Named rather than reconciled; nothing in this document depends on it.

**A consequence worth carrying.** §4.6 said B's SparseCore half should wait for §11 item 4. It
still should — but the measurement shows the 0.7–0.9 ms `modelled` prize attributed to that
deferred work is **already collected, and exceeded, by the TensorCore half alone**, because
narrowing `n` moves the partition without touching the kernel. The SC-half source change would be
an *additional* prize on a much smaller base, and its expected value should be re-derived before
anyone spends on it.

---

## 6. Caveats that survive

1. **Single-chip microbenchmark. Not an end-to-end number.** One TensorCore, kernel in isolation,
   generated inputs. 21 of 78 layers run the indexer; this kernel's share of a real step is not
   measured here, and nothing in this document predicts a serving throughput change.
2. **`decode_prod` and `mixed_prod` cannot support a conclusion** — the effect is at or below the
   spread of the arms it is a difference of (§3). B is *not* shown to help at the production
   prod-length shapes by this job.
3. **`mixed_prod` in the `sparsecore_topk` table is unusable** — 885.74% incumbent spread against a
   6.41% effect.
4. **The measured prize is a ceiling.** The driver passes the exact `kv_len` as the bucket. A
   shipping version must bucket the width to a small set of jit-static values so the recompile
   count stays bounded (§4.1 area 2, bucket count `assumed`), and every bucket rounds *up*, which
   gives back part of the win. How much depends on the ladder, which is not designed here.
5. **`mixed_64k`'s 21.72% is partly a threshold effect.** The incumbent's 67584 sits just above
   `MAX_SLICE_WORDS = 65536` and B's 65536 sits at it. That is a real, shipping win at that
   `max_model_len`, but it will not appear at `max_model_len` values that do not straddle 65536.
   Do not generalise the 21.72% to other long-context configurations.
6. **The SparseCore design row is a range, 0.7–0.9 ms `modelled`, and stays one everywhere.**
   Area 8's streaming floor runs 3.2–7.2× optimistic against the `topk ctrl` probe and the two
   derivation routes agree only to within their own precision. It is never a point in this
   document.
7. **`sparsecore_topk.py` is unmodified and the SparseCore half of B is not attempted** (§4.3
   above, and §4.6).
8. **The confound shape is synthetic.** `kv_len 8192 / max_model_len 66560` is the ratio B is
   designed for, and it is a plausible production configuration, but no job in this repo has
   established that a real GLM-5.2 deployment runs at it. B's headline is the size of the prize
   *when that ratio occurs*, not evidence of how often it does.
9. **Recompilation and the flag plumbing are not measured.** `kv_len_bucket` is a static argname;
   the cost of the extra jit cache entries, and the plumbing that would compute the bucket at the
   call site, are not in any number here.

---

## 7. Files

| file | what it is | commit |
| --- | --- | --- |
| `variant.diff` | incumbent → B, unified diff, 372 lines | `9bc5f5cd` |
| `streamindex_topk_rs.py` | the variant, as run | `9bc5f5cd` |
| `_rs_support.py` | loads both impls, binds the extra static kwargs, computes both widths | `9bc5f5cd` |
| `run_perf.py` | full-call A/B through the read-only `microbench` | `9bc5f5cd` |
| `run_topk_width.py` | `sparsecore_topk` alone at both widths (component) | `9bc5f5cd` |
| `run_correctness.py` | 4 arms + `--stale`, through the read-only oracle | `9bc5f5cd` |
| `run_sc_coverage.py` | `sparsecore_topk` direct oracle coverage, `+1e30` padding | `9bc5f5cd` |
| `shard.sh` | the 4-way fan-out | `9bc5f5cd` |
| `extra_regimes.py` | **instrument addition, not part of B** — the confound + two mask shapes | `36aedcdc` |
| `RESULTS.md` | this file | |

Raw JSON from the job: `$CDK_OUTPUT_DIR/right-size/{perf_step1,perf_confound,topk_width,correctness_a,correctness_b,sc_coverage}.json`,
retrieved with `cdk job sync-outputs j-d9502595-54a2-45bc-947b`.
