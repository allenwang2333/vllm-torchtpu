# D1-lite — keep the `lax.cond`, hoist the branch-invariant metadata and prefetch

**One job, TPU7x `2x2x1`, four chips, one shard per chip.
Job `j-e985297a-67f8-4998-99f8`, `JobStatus: Succeeded`, `FANOUT_RC=0`, all four
shards `rc=0`.**

Headline:

> **D1-lite recovers 41% of D1's 21-layer win and keeps the fast path.**
> At 21 layers, production shape, the residual goes from the incumbent's
> **26.069** to **19.087 µs/layer** — **−6.982 µs/layer**, against D1's
> **−16.980**. In the fast-path regime at 21 layers it costs **+0.772
> µs/layer (+10.7%)**, against D1's **+30.467 µs/layer (+421%)**.
>
> The hoist worked on **metadata** (−4.389 of D1's −8.466, **52%**) and **not
> at all on prefetch** (−0.010 of D1's −2.218, **0.5%**): XLA sinks the hoisted
> scalar-prefetch constants straight back into each `conditional` branch, and
> the row count proves it — 63 `prefetch_c3` rows in both the incumbent and
> D1-lite at 21 layers, against 3 in D1. An unexpected third of the win comes
> from a bucket the design did not target: the branch body itself got cheaper
> (`cond` −2.543, 38% of D1's −6.770).
>
> The inner-`lax.cond` variant (`dlc`), designed to skip the hoisted work on
> the fast path, **is worse on both axes** and should be dropped.

Everything below is **`hlo_stats` device self time**. No wall clock is reported
anywhere. The probes print one and label it `NOT A RESULT`, because four shards
shared the pod and `time_arm` enqueues asynchronously.

---

## 1. Definitions — every term and every column

### Arms

| arm | module | `enable_early_exit` | `hoist` | what it is |
|---|---|---|---|---|
| `inc` | incumbent `streamindex_topk` | `True` | — | **the incumbent**, i.e. production. Baseline for every Δ. |
| `d1` | incumbent `streamindex_topk` | `False` | — | **D1**: the `lax.cond` and `_fast_path` deleted. No new code — `enable_early_exit=False` is already a supported argument. |
| `dl` | `streamindex_topk_d1lite` | `True` | `"plain"` | **D1-lite**. The cond and `_fast_path` are kept; the branch-invariant metadata and prefetch constants are computed unconditionally at the top level of the jitted function and closed over, so they become operands of the outer `conditional` and live in the entry computation where CSE spans layers. |
| `dlc` | `streamindex_topk_d1lite` | `True` | `"cond"` | **D1-lite, inner-cond**. Same hoist, but the hoisted computation is itself wrapped in `lax.cond(fast_pred, zeros, compute)` so the fast path can skip it. Intended to get the CSE *and* the 7.7 µs. |
| `dlctl` | `streamindex_topk_d1lite` | `True` | `"none"` | **Port control.** Metadata built inside `_common_path`, exactly where the incumbent builds it. Structurally the incumbent, run through the variant module. Its job is to show the port is faithful — it must reproduce `inc`. |

### Configurations

`L<layers>_ctx<ctx>`. Every configuration is the production kernel
configuration: batch 64, `max_model_len` 9216, `page_size` 1024, `k` 2048,
`compression_ratio` 1, `num_kv_pages_per_block=(2,2,2)`,
`num_queries_per_block=(1,128,128)`, `num_positions` 10240, `num_tokens` 64.
Read back from the pod: `BENCH_CONSTS k=2048 cr=1 page=1024 bkvp=(2, 2, 2)
nqpb=(1, 128, 128) mml=9216`.

| config | layers | ctx | `max(seq_lens)//cr <= k`? | what it is for |
|---|---:|---:|---|---|
| `L1_ctx8704` | 1 | 8704 | **False** — `_common_path` | single call at production. Comparable to Option D's `pre1`/`d1`. |
| `L21_ctx8704` | 21 | 8704 | **False** — `_common_path` | **the 21-layer chain at production. This is where the CSE effect lives and this is the headline.** |
| `L1_ctx1024` | 1 | 1024 | **True** — `_fast_path` | fast-path regime, single call. |
| `L1_ctx2048` | 1 | 2048 | **True** — `_fast_path` | fast-path regime, single call. |
| `L21_ctx2048` | 21 | 2048 | **True** — `_fast_path` | fast-path regime at the depth production actually runs. |

The 21-layer chain is 21 independent `streamindex_topk` calls with per-layer `q`
and `indexer_weights` (from `jax.random.split(key(1234), 2*21)`) over a shared
KV cache, `seq_lens`, `page_indices`, `cu_q_lens` and `distribution` — the same
harness Option D used. That the chain is real and not one layer CSEd 21 ways is
checked by the row counts: **63 `stage_a` rows** in every 21-layer `_common_path`
arm (21 layers × 3 grid arms) against **3** in every one-layer arm.

### Columns

* **full µs/layer** — sum of `Avg. self time (us)` over **TensorCore** rows of
  the whole program, divided by the layer count. The device time of one
  `streamindex_topk` call.
* **stage A µs/layer** — TensorCore rows matching `StreamIdxTC` — the
  Pallas/Mosaic scoring kernel.
* **stage B µs/layer** — TensorCore rows matching `sc_topk|sparsecore_topk|
  SparseCoreTopK` — the SparseCore exact top-k, as seen *from the TensorCore*
  (the `call-done` row, i.e. the time the TC is blocked).
* **residual µs/layer** — `full − stage A − stage B`. **This is the figure the
  change is judged on** (see §6): everything in the caller's XLA graph that is
  not one of the two kernels.
* **Δresidual µs/layer** — this arm's residual per layer minus `inc`'s.
* **SC µs/prog** — the **SparseCore** plane total for the whole program.
  Reported and **never added** to anything: the two planes run concurrently and
  the TC `call-done` row already carries stage B.
* **rows** — `rows_kept/rows_total` from the dump. Must be equal.
* **partition** — `sum(coarse buckets) − TensorCore total`. Must be ~0.
* **checks** — `ok` iff `rows_kept == rows_total` **and**
  `|partition| < 1e-6` **and** the fine bucket partition
  `|sum(fine buckets) − TC total| < 1e-6`.

### Fine buckets

Imported verbatim from `../option-d-hoist/analyze_hlo_stats.py`, so these
numbers are directly comparable to Option D's table.

| bucket | what it holds |
|---|---|
| `stage_a` | the Mosaic scoring kernel |
| `stage_b_kernel` | the SparseCore top-k as seen from the TC |
| `cond` | the `conditional` instruction itself |
| `predicate` | the `reduce_max` computing `max(seq_lens)` |
| `metadata` | `_effective_row_lengths` and its spillover — the row-length / batch-boundary arithmetic |
| `prefetch_c3` | the three scalar-prefetch constants handed to `pallas_call` |
| `stage_b_shaping` | the bitcast/reshape of this layer's score matrix into stage B's input |
| `init_neg_inf` | the `-inf` fill of `scores_init` |
| `weights_layout` | the indexer-weights layout copy and bf16→f32 convert |
| `boundary_copy` | `copy`/`copy-done` at computation boundaries |

---

## 2. Provenance — the mapped-in kernels are the ones that ran

**The pinned image's kernels are stale**: its `streamindex_topk.py` differs from
the checkout and it has **no `sparsecore_topk.py` at all**. Both were supplied
by `--map-dir`. A green `JobStatus` is not evidence that the mapped-in code ran;
the md5 read-back is. Printed by every one of the four shards
(`results/shard{0,1,2,3}.j-e985297a.txt`), identical in all four:

```
PKG vllm_torchtpu /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/__init__.py
MODULE .../deepseek_v4/streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532
MODULE .../deepseek_v4/sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658
MODULE /root/kw/variants/d1-lite/streamindex_topk_d1lite.py md5=9d287fb7f6512b4393b144c252d8f9ae
MODULE /root/kw/instruments/bench_streamidx_topk.py md5=e8e492d13e59c4a1675cc7a98c5a0813
EARLY_EXIT_IN_SIG True
KERNEL_IDENTITY d1lite.kernel is streamindex_topk.kernel: True
HELPER_IDENTITY d1lite._effective_row_lengths is incumbent's: True
```

Both kernel md5s match the checkout exactly. `KERNEL_IDENTITY` and
`HELPER_IDENTITY` are the assertion that **not one line of Mosaic changed**:
D1-lite imports the kernel body and every pure helper from the incumbent by
reference. The change is entirely in the caller's XLA graph.

Image: `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`.
Checkout `annyan/streamidx-topk-opt`. `~/cloud-devkit` at `c4e1d4e` (one
untracked file, `recipes/experimental/gdn-comparison.yaml`).

---

## 3. What was hoisted

`streamindex_topk_d1lite.py` moves two groups of computation out of the
`lax.cond` branches and into the entry computation, where CSE can span the 21
call sites. Everything moved is a pure function of `(seq_lens, cu_q_lens,
distribution)` plus static shape constants — none of it depends on which branch
is taken, and none of it is a function of `q`, `indexer_weights` or `cache_kv`,
which genuinely are per-layer.

**Metadata.** `_effective_row_lengths(...)`
(`src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py:482-506`), called
from `_common_path` at `streamindex_topk.py:823-830`; `sparsecore_topk`'s
`jnp.pad(row_lengths, (0, b_pad - b))`
(`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py:322`); and
`decode_batch_end = distribution[0] // decode_req_batch_size *
decode_req_batch_size` (`streamindex_topk.py:755-756`).

**Prefetch.** The three scalar-prefetch operands built inside
`run_scores_kernel` at `streamindex_topk.py:694-701` —
`jnp.array([start_seq_idx, end_seq_idx], jnp.int32)`, `jnp.zeros((3,),
jnp.int32)` (bq/bkv/bo semaphore ids) and `jnp.full((2,), -1, jnp.int32)`
(in-flight out-DMA row counts).

**Not hoisted, because it is genuinely per-layer:** the `-inf` `scores_init`
fill (donated into the kernel output via `input_output_aliases`), the stage-B
`bitcast_convert_type`/reshape of this layer's score matrix, and the
indexer-weights layout copy and bf16→f32 convert.

`sem_ids` and `bo_sz` are written by the Mosaic body
(`streamindex_topk.py:140,365,393,416`), so sharing one hoisted buffer across
call sites needed an argument rather than an assumption: scalar prefetch
operands are copied into each launch's SMEM, and Option D's `dcond21` arm
already ran 63 `pallas_call`s off 3 shared buffers with its 28 equivalence tests
passing. §7 records that this turned out to be moot — XLA re-materialises them
per call site anyway.

### Static effect on the graph (compile-time, not a measurement)

From the probe's HLO census (`results/shard*.j-e985297a.txt`), 21-layer programs:

| arm | instructions | computations | conditionals | custom-calls | branch computations |
|---|---:|---:|---:|---:|---:|
| `inc` | 8476 | 632 | 21 | 315 | 106 |
| `dl` | **7912** | **532** | 21 | 315 | 86 |
| `dlc` | 9207 | 674 | **42** | 315 | 148 |
| `d1` | **616** | **55** | 0 | 129 | 1 |

`dl` removes 564 instructions and 100 computations from the incumbent — real,
but a long way from D1's 616/55. `dlc`'s inner conditionals did **not** CSE:
`conditionals=42` is 2 per layer, 21 outer plus 21 inner.

At one layer `dlctl` is instruction-for-instruction the incumbent (415/32/1/15/6
for both), which is the static half of the port-fidelity check.

---

## 4. Table 1 — the 21-layer chain at production (`L21_ctx8704`)

**This is the headline configuration.** Per layer, TensorCore plane.

| arm | full µs/layer | Δfull vs `inc` | stage A | stage B | residual | Δresidual | SC µs/prog | rows | partition | checks |
|---|---:|---:|---:|---:|---:|---:|---:|:--|---:|:--|
| `inc` | 170.241 | +0.00% | 111.566 | 32.605 | **26.069** | +0.000 | 661.266 | 822/822 | −1.4e−12 | ok |
| `dl` | 163.216 | −4.13% | 111.519 | 32.610 | **19.087** | **−6.982** | 661.029 | 786/786 | −5.0e−12 | ok |
| `dlc` | 168.597 | −0.97% | 111.722 | 32.416 | 24.459 | −1.610 | 656.991 | 961/961 | −9.1e−13 | ok |
| `d1` | 121.361 | −28.71% | 112.134 | 0.138 | **9.089** | **−16.980** | 697.684 | 356/356 | +4.1e−12 | ok |

**D1-lite recovers 6.982 / 16.980 = 41.1% of D1's 21-layer residual win.**

`inc`'s 26.069 reproduces Option D's `pre21` residual of 26.142 (−0.28%), and
`d1`'s 9.089 reproduces Option D's `d21` 9.089 exactly, on a different job and a
different chip. The baseline and the D1 arm are therefore not new claims.

### The split into Option D's components

Per layer, `arm − inc`, `L21_ctx8704`:

| bucket | `dl − inc` | `dlc − inc` | `d1 − inc` | D1-lite recovery of D1 |
|---|---:|---:|---:|---:|
| `cond` | **−2.543** | −0.005 | **−6.770** | **37.6%** |
| `predicate` | −0.001 | −0.001 | −0.064 | 1.6% |
| **`metadata`** | **−4.389** | +0.297 | **−8.466** | **51.8%** |
| **`prefetch_c3`** | **−0.010** | −2.331 | **−2.218** | **0.5%** |
| `stage_b_shaping` | −0.025 | −0.057 | +0.037 | — |
| `init_neg_inf` | −0.020 | −0.018 | −0.005 | — |
| `weights_layout` | −0.003 | −0.002 | −0.006 | — |
| `boundary_copy` | +0.009 | +0.508 | +0.512 | — |
| `stage_a` | −0.048 | +0.156 | +0.568 | untouched |
| `stage_b_kernel` | +0.005 | −0.190 | −32.467 | untouched — see §6 |
| **residual** | **−6.982** | **−1.610** | **−16.980** | **41.1%** |
| **TC total** | −7.024 | −1.643 | −48.879 | |

Three things fall out.

**(a) The metadata hoist works, at about half strength.** −4.389 of D1's
−8.466. The row count says why it is half and not all: `metadata` goes 294 rows
→ 216 (14 → 10.3 per layer), where D1 collapses it to 34. So roughly a quarter
of the metadata instructions were pulled into the entry computation and CSEd;
the rest XLA re-materialised inside the branches.

**(b) The prefetch hoist does not work at all.** −0.010 of D1's −2.218.
`prefetch_c3` is **63 rows in both `inc` and `dl`** — exactly 3 per layer,
unchanged — against 3 rows total in `d1`. The three scalar constants were
hoisted in the source and XLA sank them straight back into each `conditional`'s
branch computation. This is the component the design most expected to win for
free, and it is the one that returned nothing.

**(c) A third of the win comes from a bucket the design did not target.**
`cond` drops 6.770 → 4.227 (−2.543). Moving work out of the branch bodies makes
the `conditional` instruction itself cheaper — fewer live values crossing the
branch boundary. Unplanned, but it is 36% of what D1-lite delivers.

**(d) `dlc` is a failure.** Its inner `lax.cond` did zero-out `prefetch_c3`
(−2.331) but paid it all back in `boundary_copy` (+0.508), `metadata` (+0.297)
and lost the `cond` saving entirely, for a net −1.610 — less than a quarter of
`dl`. And §5 shows it is also the *worst* arm on the fast path at depth. It has
no regime where it wins; drop it.

---

## 5. Table 2 — single call at production (`L1_ctx8704`)

Same three arms plus the port control, one layer. This is the harness Option D
called `pre1`/`d1`, so the two harnesses can be compared.

| arm | full µs | Δfull vs `inc` | stage A | stage B | residual | Δresidual | rows | partition | checks |
|---|---:|---:|---:|---:|---:|---:|:--|---:|:--|
| `inc` | 171.275 | +0.00% | 111.727 | 32.490 | 27.058 | +0.000 | 42/42 | +5.7e−14 | ok |
| `dlctl` | 171.333 | **+0.03%** | 111.731 | 32.473 | 27.129 | +0.072 | 42/42 | −2.8e−14 | ok |
| `dl` | 168.875 | −1.40% | 111.566 | 32.647 | 24.661 | −2.397 | 44/44 | +2.8e−14 | ok |
| `dlc` | 171.908 | +0.37% | 111.729 | 32.416 | 27.764 | +0.706 | 41/41 | +2.8e−14 | ok |
| `d1` | 163.294 | −4.66% | 111.735 | 32.382 | 19.177 | −7.881 | 38/38 | −8.5e−14 | ok |

**The port control is clean.** `dlctl` — the D1-lite module with the hoist
turned off — is +0.03% on the full call and +0.072 µs on the residual. The
variant module's plumbing costs nothing, so every `dl`/`dlc` delta is the hoist
and not the port. (`dlctl` was run only at this configuration; that is a gap,
though a small one — the static census shows it is instruction-identical to
`inc`.)

**The two harnesses agree with Option D.** `inc` 171.275 against Option D's
`pre1` 171.132 (+0.08%); `d1` 163.294 against Option D's `d1` 163.134 (+0.10%);
residuals 27.058 vs 27.015 and 19.177 vs 19.191.

**At one layer D1-lite is worth −2.397 µs**, almost all of it the cheaper
`cond` (−2.652); `metadata` moves only −0.351 because at one layer there is
nothing to CSE against. `weights_layout` regresses +0.620 (1 row → 2) at one
layer and vanishes at 21 (−0.003), i.e. it CSEs away. **The one-layer number is
not the number to quote** — 41% of D1's win requires 21 layers.

---

## 6. Table 3 — the fast-path regime

`max(seq_lens) // compression_ratio <= k` is **True**, so the incumbent takes
`_fast_path`: a closed-form index range, no stage A and no stage B at all. This
is the regime D1 destroys. **Full call, per layer** — the residual is the whole
call here for every arm but `d1`, so `full` is the right column to read.

| arm | `L1_ctx1024` | vs `inc` | `L1_ctx2048` | vs `inc` | **`L21_ctx2048`** | **vs `inc`** |
|---|---:|---:|---:|---:|---:|---:|
| `inc` | **7.805** | 1.00× | **7.779** | 1.00× | **7.238** | 1.00× |
| `dl` | 12.910 | 1.65× | 12.901 | 1.66× | **8.010** | **1.11×** |
| `dlc` | 10.813 | 1.39× | 10.732 | 1.38× | 11.338 | 1.57× |
| `d1` | 57.514 | **7.37×** | 59.072 | **7.59×** | 37.705 | **5.21×** |

All arms `ok`: rows 27/27, 33/33, 40/40, 38/38 at one layer; 507/507, 534/534,
940/940, 356/356 at 21.

**The literal pass condition is failed at one layer and met at 21.** The brief
asked whether `d1_lite` stays at the incumbent's ~7.7 µs. At a single call it
does not: `dl` is 12.9 µs, +5.1 µs, because the plain hoist makes the fast path
compute metadata it used to skip and there is no second layer to amortize it
over. **At 21 layers — the depth production runs — that cost amortizes almost
entirely: +0.772 µs/layer, +10.7%.** The 7.4–7.6× catastrophe that made D1
unlandable becomes 1.11×.

**`dlc` does not do what it was built to do.** The inner `lax.cond` was supposed
to be the mode that skips the work *and* CSEs. It is better than `dl` at one
layer (1.38–1.39× vs 1.65–1.66×) and **worse at 21** (1.57× vs 1.11×), because
its 21 inner conditionals did not CSE (`conditionals=42`) and it pays
`cond` +3.538 and `boundary_copy` +0.545 per layer. It is beaten by `dl` in the
only configuration that matters.

### Why the residual and not the TC total

Option D §5 established this and it applies unchanged. At 21 layers `d1`'s
`stage_b_kernel` collapses by −32.487 µs/layer, which is cross-layer
SparseCore/TensorCore overlap in a chain of 21 back-to-back calls with no
decoder work between them. It does not transfer to a real decoder. The same
collapse appears here (`d1 − inc` = −32.467). It is excluded from every claim
above, which is why the headline is the **residual** −6.982 and not the TC total
−7.024. For `dl` and `dlc` the two happen to be nearly equal, because neither
touches stage B.

---

## 7. Correctness

Shard 3, `rc=0`, three steps all green (`results/shard3.j-e985297a.txt`).

| step | what it ran | result |
|---|---|---|
| `grid` | `test_d1lite_equivalence.py` — small-shape grid, `PAGE_SIZE` 128, `k` 64, `compression_ratio` 1; 8 shapes × 3 hoist modes plus the metadata-equality and fast-path-fires tests | **42 passed** in 65.19s |
| `incumbent_kernel_test` | the checkout's own `tests/kernels/test_dsv4_streamindex_topk.py`, `-m "not nightly and not multichip"` — the guard that the mapped-in kernels are importable and correct | **16 passed** in 28.15s |
| `prod` | `correctness_prod_d1lite.py` — the **production** configuration, `np.array_equal` against a separately-constructed incumbent device call with `enable_early_exit=True` | **failures=0** |

The production check, verbatim from `results/shard3.j-e985297a.txt`:

```
ctx=1024 num_tokens=64 num_positions=10240 k=2048 fast_path_predicate=True
ctx=1024 incumbent output == _fast_path closed form: True (expected True)
ctx=1024 hoist=none  array_equal_to_incumbent=True n_differing=0
ctx=1024 hoist=plain array_equal_to_incumbent=True n_differing=0
ctx=1024 hoist=cond  array_equal_to_incumbent=True n_differing=0
ctx=2048 num_tokens=64 num_positions=10240 k=2048 fast_path_predicate=True
ctx=2048 incumbent output == _fast_path closed form: True (expected True)
ctx=2048 hoist=none  array_equal_to_incumbent=True n_differing=0
ctx=2048 hoist=plain array_equal_to_incumbent=True n_differing=0
ctx=2048 hoist=cond  array_equal_to_incumbent=True n_differing=0
ctx=8704 num_tokens=64 num_positions=10240 k=2048 fast_path_predicate=False
ctx=8704 incumbent output == _fast_path closed form: False (expected False)
ctx=8704 hoist=none  array_equal_to_incumbent=True n_differing=0
ctx=8704 hoist=plain array_equal_to_incumbent=True n_differing=0
ctx=8704 hoist=cond  array_equal_to_incumbent=True n_differing=0
failures=0
```

Two things this establishes beyond "the outputs match". First, the grid covers
**both regimes** — `decode_b8_fastpath_fires` and
`decode_b8_fastpath_fires_ragged` have `max(seq_lens) <= k`, and the rest do
not — so the fast path is exercised element-for-element, not merely assumed to
be untouched. Second, the `incumbent output == _fast_path closed form` line is
an independent check, against a numpy re-implementation of
`streamindex_topk.py:840-856`, that the arm labelled *fast-path regime* really
is taking the fast path. Without it, a perf table headed "fast-path regime"
could be measuring the scoring path at a small context and nobody would know.

Zero tolerance was the right bar: D1-lite moves *where* metadata is computed and
nothing else, so any difference at all would be a bug.

---

## 8. A deliberate deviation from the brief

The brief suggested building D1-lite on `../merged/streamindex_topk_cond.py`.
**It is built on the incumbent module instead**, so that `inc`, `d1`, `dl`,
`dlc` and `dlctl` differ *only* by the hoist and the `enable_early_exit` flag,
with no `head_major_prefill` or bkv-recommender confound riding along. The cost
of the deviation is that these numbers are not directly comparable to the
`merged` arm's; `merged/GAPS.md` already showed `inc_nee` reproduces the merged
`d1` to within 0.04–0.22%, and §4/§5 above show this job's `inc` and `d1`
reproduce Option D's `pre*`/`d*` to within 0.1–0.3%, so both baselines are
anchored.

## 9. What failed

Nothing on the pod. All four shards `rc=0`, `FANOUT_RC=0`, every step in every
shard `rc=0`.

One thing failed **off-pod**, on the first read, and is worth recording because
it would have produced a wrong table silently. `bdt_hlo_stats.dump()` defaults
to `--top-n 500`. A 21-layer program carrying 21 `lax.cond`s emits more rows
than that — 822 for `inc`, 786 for `dl`, 961 for `dlc` — so the first pass
truncated every 21-layer arm that has a cond, and **the coarse partition check
still passed at −4.5e−12**, because it sums only the rows it kept. The
`rows_kept == rows_total` gate is the only thing that caught it, reporting
`500/822` and marking the arms `FAILED`. The tables above are the re-read at
`--top-n 4000`, with every arm at `rows_kept == rows_total`. `d1`, having no
conds and only 356 rows, was never truncated — so the arm one would most want to
trust was fine while its baseline was not. `read_d1lite_hlo_stats.py` now takes
`--top-n` and defaults to 4000.

---

## 10. Verdict

**D1-lite works, at 41% strength, and it makes the fast-path regression
survivable rather than eliminating it.**

| | D1 | D1-lite (`dl`) |
|---|---:|---:|
| 21-layer residual win at production | −16.980 µs/layer | **−6.982 µs/layer** |
| as % of the incumbent's 170.241 µs call | −10.0% | **−4.1%** |
| 21-layer cost in the fast-path regime | +30.467 µs/layer (5.21×) | **+0.772 µs/layer (1.11×)** |
| single-call cost in the fast-path regime | +49.7 / +51.3 µs (7.4× / 7.6×) | +5.1 µs (1.65×) |
| correctness vs incumbent | differs by construction on a firing ctx | **`np.array_equal`, every shape, every mode** |

Whether that trade is worth landing is a decision this document does not make.
What it establishes is that the trade *exists*: the choice is no longer between
D1's −10.0% with a 7.4× cliff and doing nothing.

Two concrete follow-ups the data points at, neither of them measured here:

1. **The prefetch hoist returned 0.5% and should be attacked directly.** The
   63 unchanged `prefetch_c3` rows say XLA is sinking constants into the branch
   computations. If that sinking can be prevented — passing the three arrays in
   as explicit `lax.cond` operands rather than closing over them, or blocking
   the sink with `jax.lax.optimization_barrier` — another −2.2 µs/layer is on
   the table, which would take D1-lite from 41% to about 54% of D1.
2. **The `metadata` hoist is half-done for the same reason** (294 → 216 rows,
   not 294 → 34). The same fix plausibly moves both.

`dlc` should be dropped. It is beaten by `dl` in the configuration that matters
and in the configuration it was designed for.

---

## 11. Reproducing

```
cdk job submit ...                       # see results/shard*.j-e985297a.txt for the RUN_CMD read-back
cdk job sync-outputs j-e985297a-67f8-4998-99f8
python3 read_d1lite_hlo_stats.py <synced>/.../command-container/d1lite/traces \
    --dumps /tmp/d1lite_dumps --top-n 4000 \
    --json results/hlo_stats.j-e985297a.json \
    --md results/hlo_stats_tables.j-e985297a.md
```

Artifacts in `results/`: `hlo_stats_tables.j-e985297a.md` (the reader's own output, from
which §4–§6 are transcribed), `hlo_stats.j-e985297a.json` (every bucket, every row count,
every gate value), `correctness_prod.j-e985297a.json`, and the four raw shard logs.
