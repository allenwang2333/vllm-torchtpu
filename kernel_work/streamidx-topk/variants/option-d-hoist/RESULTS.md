# Option D — hoist the layer-invariant metadata above the `lax.cond`

**Measured on TPU7x, 2x2x1, one chip per shard. Nightly shape
`prod_decode_b64_ctx8704`: batch 64, `max_model_len` 9216, page 1024,
`num_positions` 10240, `k` 2048, `compression_ratio` 1.**

Headline, and read the caveat in §7 before quoting it:

> Removing the `lax.cond` takes the residual from **26.142 to 9.089 µs per
> layer** of TensorCore device self time — **−17.053 µs/layer, 93% of the
> 18.39 µs R2 called reachable**, and −10.0% of the incumbent's 170.372 µs
> per-layer call. ×21 indexer layers = **−358 µs/step/core**.
>
> The *explicit* hoist — the thing "Option D" names — is worth **−0.013
> µs/layer (0.01%)**. XLA's own CSE does the whole amortization for free the
> moment the `cond` is gone. D1 is the entire win; D2 is a no-op.

Everything below is **device self time from `hlo_stats`**. No wall clock is
reported anywhere in this document; the probe prints some and it is labelled
`NOT A RESULT` there, because the shards were contended four-ways and
`time_arm` enqueues asynchronously.

---

## 1. What was built, and what layer of the stack it edits

**The change is in the caller, not inside a Pallas kernel.** Not one line of
Mosaic changed. `streamidx_hoist.py` imports the incumbent's kernel bodies by
reference and the provenance read-back asserts the identity on the pod:

```
KERNEL_IDENTITY streamidx_hoist.kernel is streamindex_topk.kernel: True
```

What changes is the XLA graph the caller hands to the compiler. Two edits,
independent, and the measurement separates them:

| | edit | code needed |
|---|---|---|
| **D1** | the `lax.cond` goes away; every call takes `_common_path` | **none** — `enable_early_exit=False` is already a supported argument of the incumbent. Production turns the cond *on* via `STREAMIDX_ENABLE_EARLY_EXIT = True` in `mla_attention_op.py:57`. |
| **D2** | the layer-invariant metadata is computed once per step by the caller and passed in | `precompute()` → `HoistedMeta`, plus `_sc_topk_direct_pre` taking pre-padded lengths |

Files, all new, all in this directory:

- `streamidx_hoist.py` — `HoistedMeta`, `precompute()`, `_sc_topk_direct_pre`,
  `_sparsecore_topk_pre`, and `streamindex_topk_d(...)` which has no `lax.cond`
  and takes `hoisted=None` (D1 only) or `hoisted=meta` (D1+D2).
- `test_option_d_equivalence.py` — the §3.3 metadata-equality assertion, plus
  D-vs-incumbent output equality, 7 shapes.
- `probe_d_hlo_stats.py` — the six arms.
- `analyze_hlo_stats.py` — R2's partition, with the partition check enforced.
- `d_readback.sh`, `d_correctness.sh`, `d_perf_shard.sh`, `d_shard0.sh`.
- `results/` — the dumps this file is computed from.

**Is the hoist structurally possible at all? Yes, and it is also unnecessary.**
The metadata is a pure function of `seq_lens`, `cu_q_lens` and `distribution`,
all of which are step-level and identical across the 21 indexer layers, so
lifting it to the caller is mechanically easy. The finding is that it buys
nothing: see §4.

## 2. Arms

Six, three of them one-layer and three of them a single `jax.jit` containing 21
`streamindex_topk` calls. The 21-layer arms exist because the 21→1
amortization is the larger of D's two mechanisms and **a one-call microbenchmark
cannot show it at all** — which is why R2, a one-call probe, had to model it.

| arm | layers | program |
|---|---|---|
| `pre1` | 1 | incumbent, `enable_early_exit=True` — R2's measurement, re-taken |
| `dcond1` | 1 | D1: no `cond`, metadata still built per call |
| `d1` | 1 | D1+D2: no `cond`, metadata from `precompute()` |
| `pre21` | 21 | incumbent ×21 |
| `dcond21` | 21 | D1 ×21 |
| `d21` | 21 | D1+D2 ×21, one `precompute()` for all 21 |

Each layer gets its **own** `q` and `indexer_weights`; only `cache_kv`,
`seq_lens`, `cu_q_lens`, `page_indices` and `distribution` are shared. Giving
every layer the same `q` would let XLA CSE whole `pallas_call`s and the result
would be a one-layer program wearing a 21-layer name. The dumps confirm it did
not: 63 `stage_a` rows in every 21-layer arm — 21 layers × the 3 grid arms —
against 3 in every one-layer arm.

## 3. The table

Per **layer**, TensorCore plane, `hlo_stats` `Avg. self time (us)`. Bucketing is
R2's, and the partition check is enforced, not assumed.

| bucket | R2 | `pre1` | `dcond1` | `d1` | `pre21` | `dcond21` | `d21` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stage_a` | 111.688 | 111.637 | 111.576 | 111.572 | 111.605 | 112.176 | 112.160 |
| `stage_b_kernel` | 31.807 | 32.481 | 32.429 | 32.372 | 32.625 | 0.138 | 0.138 |
| `cond` | 6.591 | 6.597 | 0.000 | 0.000 | 6.766 | 0.000 | 0.000 |
| `predicate` | 1.330 | 1.336 | 0.000 | 0.000 | 0.062 | 0.000 | 0.000 |
| `metadata` | 8.172 | 9.029 | 9.128 | 9.194 | 9.225 | 0.681 | 0.692 |
| `prefetch_c3` | 2.298 | 2.311 | 2.283 | 2.287 | 2.335 | 0.112 | 0.114 |
| `stage_b_shaping` | 4.637 | 4.630 | 5.405 | 5.407 | 5.390 | 5.427 | 5.422 |
| `init_neg_inf` | 1.655 | 1.655 | 1.643 | 1.643 | 1.666 | 1.660 | 1.660 |
| `weights_layout` | 1.452 | 0.617 | 0.619 | 0.619 | 0.618 | 0.612 | 0.612 |
| `boundary_copy` | 0.844 | 0.840 | 0.041 | 0.041 | 0.081 | 0.593 | 0.589 |
| **TC total** | **170.474** | **171.132** | **163.124** | **163.134** | **170.372** | **121.400** | **121.387** |
| _residual (TC total − A − B)_ | 26.979 | 27.015 | 19.119 | 19.191 | 26.142 | 9.085 | 9.089 |

### Partition check — `sum(buckets) − TC total`, must be 0

```
  r2         +0.000000 us   rows=42  (TC  40, SC  2 excluded,  30.984 us)  layers= 1  prog  170.474 us
  pre1       +0.000000 us   rows=42  (TC  40, SC  2 excluded,  32.204 us)  layers= 1  prog  171.132 us
  dcond1     +0.000000 us   rows=38  (TC  36, SC  2 excluded,  32.246 us)  layers= 1  prog  163.124 us
  d1         +0.000000 us   rows=38  (TC  36, SC  2 excluded,  32.189 us)  layers= 1  prog  163.134 us
  pre21      +0.000000 us   rows=822 (TC 780, SC 42 excluded, 661.645 us)  layers=21  prog 3577.807 us
  dcond21    +0.000000 us   rows=356 (TC 314, SC 42 excluded, 697.898 us)  layers=21  prog 2549.390 us
  d21        -0.000000 us   rows=356 (TC 314, SC 42 excluded, 697.956 us)  layers=21  prog 2549.125 us
```

`rows_kept == rows_total` is asserted for every dump — a truncated `hlo_stats`
table looks exactly like a complete one, and R2's first dump was truncated.

### The baseline reproduces R2

`pre1` against R2's own numbers: TC total 171.132 vs 170.474 (**+0.39%**),
residual 27.015 vs 26.979 (**+0.13%**). Every bucket agrees to within 0.9 µs.
The one visible itemization difference is `weights_layout` 0.617 vs 1.452 with
`metadata` 0.857 higher to compensate: R2's probe fed bf16 indexer weights and
paid a layout copy this probe's inputs do not, and the analyzer files that copy
under `metadata` here. The totals and the residual — the quantities this
comparison turns on — agree.

## 4. Where the win comes from, and where it does not

Per-layer deltas, `d` minus `pre`:

| bucket | 1 layer (`d1−pre1`) | 21 layers (`d21−pre21`) | R2 said |
|---|---:|---:|---|
| `cond` | **−6.597** | **−6.766** | removed |
| `predicate` | −1.336 | −0.062 | removed |
| `metadata` | +0.165 | **−8.533** | amortize 20/21 |
| `prefetch_c3` | −0.024 | **−2.221** | amortize 20/21 |
| `stage_b_shaping` | +0.777 | +0.032 | keep |
| `init_neg_inf` | −0.012 | −0.005 | keep |
| `weights_layout` | +0.002 | −0.006 | keep |
| `boundary_copy` | −0.799 | **+0.508** | keep |
| `stage_a` | −0.065 | +0.555 | untouched |
| `stage_b_kernel` | −0.109 | **−32.487** | untouched — see §5 |
| **residual** | **−7.824** | **−17.053** | −17.83 to −18.39 |
| **TC total** | −7.998 | −48.985 | |

Four things fall out of this.

**(a) R2's itemization was right, item by item.** Everything it marked
*removed* is at 0.000 and everything it marked *keep* moved by less than
0.8 µs. The residual delta of −17.053 against a predicted −17.83 (the pinned
prediction) or −18.39 (the reachable figure) is **93–96% delivered**, and
−17.053/170.372 = **−10.0%** of the call, inside R2's −5.9% to −10.8% band and
just under its −10.5% point estimate.

**(b) The one item R2 over-counted is the predicate.** R2 measured
`reduce_max.7` at 1.330 µs in a one-call program and charged it per layer. In a
21-layer program XLA already CSEs it: `pre21` shows **one** `reduce_max` row for
the whole program, 1.301 µs, i.e. 0.062 µs/layer. So the predicate was worth
1.27 µs/layer less than the model assumed. The metadata amortization came in
correspondingly better (−8.533 against a predicted −7.78), and the two errors
happen to cancel.

**(c) D2 — the actual hoist — is worth nothing.** `d21` vs `dcond21`:
**−0.0126 µs/layer**, and the sign on `metadata` is the wrong way round (0.692
hoisted vs 0.681 not hoisted). The metadata bucket drops from 294 rows in
`pre21` to 34 in *both* D arms, and `prefetch_c3` from 63 rows to 3 in both.
XLA's CSE performs the entire 21→1 collapse by itself once the `cond` boundary
is removed — which is precisely what R2 said the `cond` was preventing, only
the consequence is stronger than the design doc supposed. **Passing metadata in
from the caller is unnecessary; deleting the `cond` is sufficient.**

**(d) D1 needs no new code.** The whole delivered win is reachable by setting
`STREAMIDX_ENABLE_EARLY_EXIT = False` in `mla_attention_op.py:57`. That is a
one-line change to a file this exercise was told not to modify, and it is the
recommendation.

## 5. The −32.487 µs/layer that is **not** a result

`stage_b_kernel` collapses from 32.625 to 0.138 µs/layer in the 21-layer D
arms, and that is 32.5 of the −48.985 headline. **It must be discarded.**

The row that vanishes is `sc_topk_direct_b64_n10240_k2048.*.call-done` — the
TensorCore's charge for *waiting* on the SparseCore. The SparseCore plane time
did not fall; it rose slightly, 661.645 → 697.956 µs/prog. The work is still
being done. With the `cond` gone, XLA is free to launch layer *i*'s SparseCore
top-k asynchronously and run layer *i+1*'s TensorCore stage A underneath it.

Two reasons that overlap does not transfer to the model:

1. **The one-layer arms prove it is cross-layer.** `dcond1` still pays the full
   32.429 µs — with a single layer there is nothing to overlap against. The
   effect appears only at 21 layers.
2. **The 21 layers in this probe are mutually independent, and in the model they
   are not.** Nothing here consumes layer *i*'s top-k before layer *i+1* runs.
   In the real decoder, layer *i*'s indices feed layer *i*'s sparse attention,
   which feeds its MLP, which feeds layer *i+1*. There is no later layer's
   stage A available to hide the SparseCore behind.

So the defensible figure is the **residual** delta, −17.053 µs/layer, not the TC
total delta of −48.985. The one-layer arms are the better analogue for stage B;
the 21-layer arms are the better analogue for the metadata. This document quotes
the residual everywhere.

This is also a live demonstration, inside this experiment's own data, of the
caveat in §7: a bucket of device self time can disappear because the work moved
off the TensorCore's charge rather than because it stopped happening.

## 6. Correctness

Verified, `✅`. Three named jobs. **Both kernel modules were md5-read-back on
the pod in every one of them** and matched R2's and `c3a18c95`'s:

```
MODULE .../deepseek_v4/streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532
MODULE .../deepseek_v4/sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658
MODULE /root/optd/streamidx_hoist.py       md5=7e6e4f7eee11684f81d8c07c5afee62a
KERNEL_IDENTITY streamidx_hoist.kernel is streamindex_topk.kernel: True
```

The kernels were mapped in from a `git archive c3a18c95` staging copy, not from
the working tree — the tree's `streamindex_topk.py` is dirty with another
agent's work (md5 `9bf2a25e…`) and using it would have measured a different
baseline than R2's.

### `j-bfecd181-4b1e-452f-8447` — incumbent, unmodified test file

```
INCUMBENT_TEST_FILE /root/optd_tests/test_dsv4_streamindex_topk.py
becf9f23ace9424a7d67cd3b34de9afd  /root/optd_tests/test_dsv4_streamindex_topk.py
1f536f26fff3224d362757b234e0d392  tests/kernels/test_dsv4_streamindex_topk.py

20 tests collected in 0.95s
20 passed in 39.93s
```

**The image's own copy of this file is stale and collects 16, not 20.** The two
md5s above are the point of printing them: `1f536f26…` is what ships in the
image `…:ec4bcecf78`, `becf9f23…` is `c3a18c95`'s, and the four tests that
differ are `test_streamindex_topk_early_exit_*` — i.e. exactly the path Option D
deletes. A first run of this shard against the image's copy returned a green
`16 passed, 14 warnings in 23.96s` while silently skipping them. The c3a18c95
copy is now mapped in at `/root/optd_tests` and preferred by
`d_correctness.sh`, and the collection count is printed so a stale file cannot
pass as a full run again.

Those four tests matter to D1's safety argument beyond just being new: they
establish that the fast path and the common path agree bit-for-bit
(`test_streamindex_topk_early_exit_no_short_is_inert`,
`…_is_batch_wide`) and as sets when the fast path fires
(`…_all_short`). Deleting the fast path is therefore not a semantic change —
which is the claim D1 rests on.

### `j-bfecd181-4b1e-452f-8447` and `j-30092894-2267-4953-9def` — Option D equivalence

```
28 passed in 52.14s      (j-bfecd181, shard1)
28 passed in 49.80s      (j-30092894, shard0)
```

`test_option_d_equivalence.py`, 28 tests over 7 shapes:

- **`test_hoisted_metadata_equals_per_layer`** — §3.3's assertion, asserted
  **directly and at zero tolerance**, 7/7 passing. Every field of `HoistedMeta`
  is compared with `assert_array_equal` against the per-layer construction:
  `eff_lengths_padded` against the incumbent's own `_effective_row_lengths`
  followed by the same `jnp.pad`, and `start_end` / `sem_ids` / `bo_sz` / the
  three grid extents against the expressions the incumbent builds inside the
  `cond`. There was no need to weaken this to a tolerance.
- **`test_option_d_matches_incumbent[…-d1_cond_only | …-d2_hoist]`** — D's
  output against the incumbent with `enable_early_exit=True`, compared as sorted
  index sets per row (top-k order is unspecified), 14/14.
- **`test_option_d_matches_incumbent_exact`** — raw rows against the incumbent
  with `enable_early_exit=False`, `assert_array_equal`, 7/7.

One of the 7 shapes is `decode_b8_all_short_fastpath_would_fire`, constructed so
that `max(seq_lens)//compression_ratio <= k` holds and the incumbent's
`_fast_path` actually executes. It is the case that tests D1's semantics rather
than its plumbing, and it passes in both the `d1` and `d2` parametrisations.

At the nightly shape the fast path is dead in any case: `compression_ratio=1`,
`k=2048`, context 8704–9216, so the predicate is always false and the `cond`
does nothing but cost 6.766 µs/layer and block CSE.

### `j-d3e3648a-481a-4611-9d99` — the run that failed, and what it found

`15 failed, 13 passed in 47.02s`, every failure the same
`NotImplementedError: hoisted lengths are only wired for the p==1 path`. The
first `_sparsecore_topk_pre` only handled `_pick_partition() == 1`, which is what
the nightly shape uses; the small-`b` test shapes pick `p=4` or `p=8`. Fixed by
delegating the `p>1` two-stage path to the incumbent's `sparsecore_topk`, fed
from the hoisted lengths — that path re-pads per stage at a different `b`, so
there was nothing to hoist there past the row lengths themselves. All 7
metadata-equality tests passed even in that run.

## 7. Verdict, and the caveat that bounds it

**`provisional`**, and adoption should be of **D1 only**.

- Correctness is `✅ verified` — named jobs above, 20 unmodified incumbent tests
  and 28 equivalence tests, including the §3.3 metadata assertion at zero
  tolerance.
- The perf figure is **device self time, at the kernel level, on a
  microbenchmark**. It is not an end-to-end result and no end-to-end run was
  made (out of scope by instruction).

### The caveat, stated as prominently as it deserves

**Device self time says the work exists. It does not say the TensorCore was
blocked on it.** Every number in §3 is an `hlo_stats` self-time charge. If the
17.053 µs/layer of metadata, `cond` and prefetch work already sat off the
critical path — overlapped with a DMA, or hidden behind the SparseCore — then
removing it wins nothing end to end, and nothing in this measurement can
distinguish that case from the case where it wins the full 358 µs/step/core.
Closing it needs a nightly in-model profile, i.e. an end-to-end run.

§5 is not a hypothetical version of this caveat; it is an instance of it that
this experiment tripped over and had to discard 32.5 µs/layer for.

### The strongest argument against adopting D

**Its whole delivered value is a `False` where the codebase deliberately wrote
`True`, and this measurement cannot see the workload that `True` was written
for.**

`STREAMIDX_ENABLE_EARLY_EXIT` exists to skip the scoring kernel when every
sequence in the batch is shorter than `k`. At the nightly decode shape that
never fires, so the `cond` is pure overhead and D1 is free. But `vllm bench
serve` at 8704 context is not the only thing the server runs. Early decode
steps, short-prompt traffic, a warmup, or any deployment with a smaller `k`
puts the batch under the threshold — and there the incumbent skips
111.6 µs/layer of stage A while D1 pays all of it. D1 trades a guaranteed
17 µs/layer against a conditional 111 µs/layer, and this exercise measured only
the regime where the trade is favourable. **R2's shape-invariance result does
not cover this**: it showed the *overhead* is issue-bound and flat across 16×
`max_model_len` and 4× batch, which says nothing about how often the predicate
is true in production.

A safer form of the same win, not measured here, is to keep the fast path but
hoist the predicate and the metadata out of the `cond` so CSE can span the
layers — but §4(c) is the warning that this is not obviously achievable: the
amortization observed here happened *because* the `cond` was gone, and a
`cond` that remains may keep blocking it.

Second-order arguments, both smaller:

- D1 adds **+0.508 µs/layer** of `boundary_copy` at 21 layers, from 46
  `copy-done` rows that `pre21` does not have. Already netted into the −17.053.
- D2 is dead weight: it adds `HoistedMeta`, a threading argument through every
  call site and a second SparseCore entry point, in exchange for
  −0.013 µs/layer. It should not be adopted. It earns its place in this
  directory as the control that proves XLA does the job.

## 8. Provenance

| | |
|---|---|
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` (prebuilt, not rebuilt) |
| topology | `2x2x1`, read back from `cdk job recipe` |
| kernels under test | `c3a18c95`, staged via `git archive`; md5 `1e221f10…` / `8bc9e25b…` read back on the pod |
| incumbent test file | `c3a18c95`, md5 `becf9f23…`, 20 collected (the image's `1f536f26…` collects 16) |
| `~/cloud-devkit` | `c4e1d4e88e1d6b8c81d7e6051f43d9bfe19f1cb2`, 1 dirty file |
| perf job | `j-30092894-2267-4953-9def` — 6 arms, `Succeeded`, `failed_steps=0` on all 4 shards |
| correctness job | `j-bfecd181-4b1e-452f-8447` — `Succeeded`, 20 passed / 28 passed |
| superseded job | `j-d3e3648a-481a-4611-9d99` — `Failed`, the `p>1` gap |
| jax | 0.10.2 for `--mode hlo`, 0.11.0 in a side venv for `--mode trace` (jaxlib 0.10.2's profiler SIGSEGVs against libtpu 0.0.44.1) |
| artifacts | `results/hlo_stats_summary.j-30092894.json`, `results/hlo_stats_table.j-30092894.txt` (every row, under its bucket) |

Not done, by instruction: no end-to-end model benchmark, and nothing here went
to the shared experiment log.
