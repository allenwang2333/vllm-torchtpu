# P2 — is the SparseCore top-k actually better than XLA?

**Verdict: yes, by ~11×, and the outputs are bit-identical.** `sparsecore_topk` is **10.2–11.8×
faster** than either XLA route at the nightly stage-B shape, and all three return exactly the same
selection. Commit `6142d5e7`'s replacement of `lax.approx_max_k(recall_target=1.0)` was justified;
the margin it bought — never recorded anywhere until now — is **~480–520 µs per indexer layer**.

**This probe was the only one that could have removed a design option, and it does the opposite: it
closes that option off.** No candidate should propose deleting stage B's 457 lines, and the
TensorCore↔SparseCore boundary the design is built around stays.

- job: **`j-e107f728-6fdc-47d8-98f9`**, shard 1. `JobStatus` at time of reading: `Running` — shard 1
  itself finished cleanly and printed its full JSON; the job stays `Running` only because shard 0's
  `pack` step is still tarring the P1 dumps. Do not read this as a green job.
- shape: nightly stage B — `b=64`, `n=10240`, `k=2048`, f32 scores, `ctx=8704`
- `sparsecore_topk` module read back from the pod:
  `/root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`
  (mapped in from the branch — the image at `ec4bcecf` predates the file)
- `_pick_partition(64, 10240, 2048) = 1`
- hardware: TPU7x, `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16,
  vmem_capacity_bytes=524288, dma_granule_size_bytes=32)`

## Timings — measured (`j-e107f728`, shard 1)

Best-of-7 trials, each trial an iteration-batched chain blocked once; iteration counts shown.

### Arm `full` — every row finite, `row_len = 10240`

| implementation | best | median | iters | speedup of SC |
| --- | ---: | ---: | ---: | ---: |
| `jax.lax.approx_max_k(k=2048, recall_target=1.0)` | 533.72 µs | 536.09 µs | 63 | 10.24× |
| `jax.lax.top_k(k=2048)` | 572.48 µs | 577.33 µs | 60 | 10.98× |
| **`sparsecore_topk`** | **52.12 µs** | 53.25 µs | 232 | — |

### Arm `prodrows` — `-inf` at and past 8704, `row_lengths=8704` passed to `sparsecore_topk`

| implementation | best | median | iters | speedup of SC |
| --- | ---: | ---: | ---: | ---: |
| `jax.lax.approx_max_k(k=2048, recall_target=1.0)` | 532.84 µs | 534.90 µs | 65 | 11.00× |
| `jax.lax.top_k(k=2048)` | 569.74 µs | 573.61 µs | 64 | 11.76× |
| **`sparsecore_topk`** | **48.45 µs** | 49.60 µs | 229 | — |

Note that only `sparsecore_topk` gets faster when the tail is masked (52.12 → 48.45 µs, −7.0 %);
both XLA routes are flat, because neither is told the rows are short. That is a second, smaller
advantage of the incumbent that the XLA routes cannot reach without a shape change.

## Correctness — measured, all three agree exactly

Compared against `jax.lax.top_k` as the reference, per row, on the selected *scores* sorted
descending (so ties that order differently do not register as a difference):

| arm | implementation | max \|Δscore\| | positions differing | index sets identical | selected |
| --- | --- | ---: | ---: | :-: | ---: |
| full | `approx_max_k_rt1.0` | 0.0 | 0 | yes | 131,072 / 131,072 |
| full | `lax_top_k` | 0.0 | 0 | yes | 131,072 / 131,072 |
| full | **`sparsecore_topk`** | **0.0** | **0** | **yes** | 131,072 / 131,072 |
| prodrows | `approx_max_k_rt1.0` | 0.0 | 0 | yes | 131,072 / 131,072 |
| prodrows | `lax_top_k` | 0.0 | 0 | yes | 131,072 / 131,072 |
| prodrows | **`sparsecore_topk`** | **0.0** | **0** | **yes** | 131,072 / 131,072 |

Not merely "scores agree" — the **index sets are identical**, on all 64 rows in both arms. The
inputs are i.i.d. `N(0,1)` f32, where exact ties are measure-zero, so this is a strong check on the
selection and a weak one on tie-handling. Tie behaviour was not tested.

`approx_max_k(recall_target=1.0)` being exact is confirmed here rather than assumed — the brief's
premise holds.

## What this costs if stage B were deleted

*Modelled, arithmetic shown, from the measured numbers above.* Replacing `sparsecore_topk` with
`lax.top_k` at the nightly shape costs `569.74 − 48.45 = 521.3 µs` per indexer layer. Against the
221.4 µs total measured for the whole nightly decode `streamindex_topk` call (`j-8b1e7acc`), that is
**a 3.4× regression of the call**, and across 21 indexer layers **≈ 10.9 ms per decode step**.

## Caveats

- **Contended.** All four shards shared one 4-chip pod, one chip each, with a common host and HBM.
  Absolute microseconds are not clean timings. All three arms were timed in the same process under
  the same contention, and an 11× margin is orders of magnitude larger than any plausible
  contention skew, so the comparison stands even though the absolutes do not.
- The 48.45–52.12 µs measured here for `sparsecore_topk` in isolation sits below the 67.4 µs that
  `report.md` §6 attributes to stage B inside the incumbent at the same shape. The difference is
  whatever the incumbent does around the call; it is not this probe's question.
- Compile time is a side observation, not a result: `sparsecore_topk` compiles in 0.51–0.64 s
  against 4.0–5.0 s for the XLA routes.

## What this settles

- **Settled: stage B stays.** No candidate may propose deleting `sparsecore_topk` in favour of XLA.
- **Settled: the correctness risk of `6142d5e7` is nil at this shape** — exact agreement with
  `lax.top_k` on indices, not just scores. This is the eval baseline that did not previously exist.
- **Settled: the TensorCore↔SparseCore boundary is load-bearing** and the design must be built
  around it rather than around dissolving it. P3 is therefore the probe that matters for the
  boundary, not this one.
- **Not settled: tie-handling and adversarial score distributions.** Random f32 normals do not
  produce exact ties. If the indexer can emit saturated or quantised scores with real ties, that
  needs its own check.
