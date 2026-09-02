# P3 — can a TensorCore grid and a SparseCore `VectorSubcoreMesh` be in flight at once?

**Verdict: YES — but only when both custom calls sit inside one XLA program.** With arm A (the real
stage-A decode `pallas_call`) and arm B (`sparsecore_topk`) returned from a single `jax.jit`,
**41.6–57.5 % of the SparseCore arm is hidden underneath the TensorCore arm**. Dispatched as two
separate executables, or from two Python threads, the overlap is **zero** (measured −1.8 % and
−11.5 %, i.e. slightly worse than serial).

This gates the largest lever identified in step 2, and it gates it *open* — with a constraint that
changes the design: **the stage-A(g) / stage-B(g−1) pipeline has to be expressed as one program.**
Two calls the runtime happens to enqueue back to back will not overlap.

- job: **`j-e107f728-6fdc-47d8-98f9`**, shard 2. `JobStatus` at time of reading: `Running` — shard 2
  itself finished cleanly and printed its full JSON; the job stays `Running` only because shard 0's
  `pack` step is still tarring the P1 dumps. Do not read this as a green job.
- arm A: nightly decode stage A, 64 seqs × 1 token, grid 16, `seq_batch_size=4`, `bq_sz=1`,
  `bkv_sz=2048`, out `(64,80,128)` f32 = 2.62 MB. Launched via `_stage_a.py`
  (`streamindex_topk.py` md5 `1e221f10cbf3573aa0d2ae78c3463532`)
- arm B: `sparsecore_topk` at `(64, 10240)` f32 → `(64, 2048)` i32, `k=2048`, `row_lengths=8704`
- hardware: TPU7x, `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16,
  vmem_capacity_bytes=524288)` — SparseCore *is* reachable from Pallas on this stack, contradicting
  `references/hardware-facts.md` and `05-compute-units.md`

## Independence — how it was ensured

Arm B is fed a **pre-made score matrix that arm A never writes**. There is no value flowing from A
to B, so XLA cannot serialise them on a data dependency. They share no buffer: each stage-A rep
allocates and aliases its own `-inf` score buffer. Both outputs are returned from the jit, so
neither is dead-code eliminated. Per-rep inputs are drawn from distinct PRNG keys
(`jax.random.key(100+i)` for q, `key(200+i)` for scores) so nothing can be CSE'd into a single
computation.

Dispatch overhead is amortised by iteration batching: each trial enqueues 40–444 calls and blocks
once, best of 5–7 trials.

## Measured — mechanism 1, one `jax.jit`

| arm | reps=1 best | reps=2 best |
| --- | ---: | ---: |
| A alone (TensorCore) | 118.45 µs | 230.63 µs |
| B alone (SparseCore) | 48.22 µs | 79.98 µs |
| **A and B together** | **140.35 µs** | **258.33 µs** |
| A + B, if serial | 166.67 µs | 310.61 µs |
| naive overlap fraction | 0.546 | 0.654 |

## Measured — mechanisms 2 and 3, and the same-unit controls

| | t_a | t_b | t_both | t_a + t_b | overlap fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| two executables, enqueued back to back | 120.73 | 51.21 | **172.86** | 171.94 | **−0.018** |
| two Python threads | (same 120.73 / 51.21) | | **177.82** | 171.94 | **−0.115** |

| control | value | reading |
| --- | ---: | --- |
| `t_a(2) / t_a(1)` | **1.947** | two TensorCore copies are essentially serial, as they must be — one unit. The harness's additive model holds. |
| `t_b(2) / t_b(1)` | **1.659** | two SparseCore copies are *not* serial. SparseCore has 2 cores / 16 subcores and the second copy partly hides in the first. |

## Correcting for fixed per-program overhead — this is the number to quote

`t_a + t_b` charges the fixed cost of a jitted program twice while `t_both` charges it once, so the
naive fraction above is inflated. That overhead is measurable from the a-arm control, where no
overlap is possible:

```
F = 2 × t_a(1) − t_a(2) = 2 × 118.45 − 230.63 = 6.27 µs      (measured, one program's fixed cost)
```

| | reps=1 | reps=2 |
| --- | ---: | ---: |
| no-overlap prediction = t_a + t_b − F | 160.40 µs | 304.34 µs |
| observed t_both | 140.35 µs | 258.33 µs |
| **genuinely hidden** | **20.05 µs** | **46.01 µs** |
| **as a fraction of the SparseCore arm** | **41.6 %** | **57.5 %** |

The two-executables row is the independent confirmation that 6.27 µs is the right correction and
not an underestimate: there, `t_both` still pays two program dispatches and comes out at
`171.94 → 172.86`, i.e. **exactly serial with no saving at all**. Whatever the one-jit saving is, it
is not a dispatch artefact — it is only available when the two custom calls are compiled together.

## What it is worth on the nightly shape

*Modelled, arithmetic shown, from measured inputs.* `report.md` §6 puts stage B at **67.4 µs** of the
221.4 µs nightly decode call (`j-8b1e7acc`). Hiding 41.6 % of it saves `0.416 × 67.4 = 28.0 µs` per
indexer layer, **12.7 % of the call**; at the 57.5 % figure, `38.8 µs` and **17.5 %**. Across 21
indexer layers that is 0.59–0.81 ms per decode step. This is a projection from an isolated
microbenchmark onto a call measured in a different job, and it is not an end-to-end prediction.

For context on why this lever is the one worth chasing: `CONTRADICTIONS.md` §9 separates the A→B
*kernel* edge (the score matrix's HBM round trip, modelled at 1.75 µs = 0.79 % of the call) from
the A→B *unit* edge (TensorCore stalled while SparseCore works, ~174.6 µs/layer at AgentX, of which
~71 µs/layer is `sc_topk_direct`). The unit edge is ~40× the kernel edge, and this probe shows it
is attackable — through HBM, with no TC↔SC shared memory needed.

## Caveats

- **Contended.** Four shards on one 4-chip pod sharing host and HBM. The absolute microseconds are
  not clean timings. Every quantity quoted as a conclusion here is a *ratio measured within one
  process against its own controls*, which is what the question needs; the µs are context.
- `t_b(2)/t_b(1) = 1.659` means the SparseCore arm is not perfectly serial with itself, so the
  reps=2 row's 57.5 % is the softer of the two figures. **Prefer the reps=1 figure, 41.6 %.**
- Overlap was demonstrated for *these two kernels at these shapes*. A real pipeline would have arm B
  consuming arm A's previous-group output through HBM, which adds a dependency this probe
  deliberately excludes; the dependency is between iterations, not within one, so it should not
  reintroduce serialisation — but that has not been measured.
- No framework refusal occurred. The `errors` map in the probe's JSON is empty; all three mechanisms
  ran.

## What this settles

- **Settled: TensorCore and SparseCore do run concurrently on this stack.** The largest lever in
  step 2 is not blocked by the hardware or the framework.
- **Settled: the concurrency is a compiler property, not a runtime one.** Two executables → 0 %.
  Two threads → 0 %. One program → 41.6 %. Any design that pipelines stage A against stage B must
  put both inside a single jitted program / XLA module. This is a hard constraint on the candidate
  designs and it was not previously known.
- **Settled: ~42 % of stage B is hideable**, worth a modelled 12.7 % of the nightly decode call.
- **Not settled: whether the overlap survives a real cross-iteration dependency**, and whether the
  incumbent's `_common_path` structure can be reshaped into one program without growing scratch.
