# P7 — `bkv_p` sweep (P7a) and prefill VMEM/spill (P7b)

Both halves ran as shards of CDK job **`j-9e113359-47c6-41b2-ad85`** (green, `tpu-fanout: 4/4 shards
succeeded`, all four rc=0), image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`,
with local `src/` mapped over the image's (md5s verified in-pod: `streamindex_topk.py`
`1e221f10…`, `sparsecore_topk.py` `8bc9e25b…`, `instruments/bench_streamidx_topk.py` `c72b23e8…`,
all matching the working tree on `annyan/streamidx-topk-opt`).

Scripts, committed on the branch:
- `probes/probe_bkvp_sweep.py` (P7a)
- `probes/probe_prefill_vmem.py` (P7b, two modes)

---

# P7a — `num_kv_pages_per_block[decode]` sweep

## VERDICT

**`bkv_p = 3` is a real, cheap, like-for-like win at the nightly decode shape: −11.91 % on the
kernel call and −19.99 % on derived stage A** versus the incumbent `bkv_p = 2`, in the same job.
And the sweep is **not flat** — `P7a_DMA_VERDICT: MOVES` — so **stage A's fetch is the constraint**.

But the *shape* of the movement refines area 3's question in a way the flat/moves dichotomy does
not capture, and the refinement matters more than the win. See [§A3](#a3-the-second-reading-not-bandwidth-bound-block-bound).

## A1. The measurement

`measured (j-9e113359, shard 1)`. Shape `prod_decode_b64_ctx8704`; inputs `q f8_e4m3fn[64,32,128]`,
`weights bf16[64,32]`, `cache_kv u8[576,256,4,256]`; H=32, D=128, k=2048, page size 1024,
`lkv_dim=256`. Tuple swept is `num_kv_pages_per_block=(bkv_p, 2, 2)` — the prefill and mixed entries
are pinned at 2 deliberately, see §A2. Inputs built **once** and shared across all four arms.
`time_arm` is best-of-N with `block_until_ready` per trial, per the read-only instrument.

| bkv_p | bkv_sz | num_positions | like-for-like | total µs | stage B µs | init µs | stage A (derived) µs | total vs bkv_p=2 | stage A vs bkv_p=2 |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 10240 | ✅ | 213.34 | 48.63 | 28.47 | 136.24 | **+11.88 %** | **+19.95 %** |
| **2** *(incumbent)* | 2048 | 10240 | ✅ | 190.68 | 48.63 | 28.47 | 113.58 | 0 | 0 |
| **3** | 3072 | 10240 | ✅ | **167.97** | 48.63 | 28.47 | **90.87** | **−11.91 %** | **−19.99 %** |
| 4 | 4096 | **12288** | ❌ **confounded** | 189.17 | 49.33 | 29.78 | 110.05 | −0.79 % | −3.11 % |

```
P7a_DMA_VERDICT: MOVES
like_for_like spread (bkv_p in {1,2,3}): total 23.79%  stage A (derived) 39.95%
P7a_BEST_LIKE_FOR_LIKE: bkv_p=3 at 167.97us, -11.91% vs the incumbent bkv_p=2
```

Thresholds the verdict used: <3 % FLAT, 3–10 % WEAK (inside `report.md`'s documented 8.5 %
run-to-run spread on the total), >10 % MOVES. The like-for-like spread is 23.79 % on the total and
39.95 % on derived stage A — comfortably clear of the noise floor.

## A2. The confound, identified before the run and held out of the conclusion

`bkv_p` does not only set the DMA block size. It also feeds

```
num_sublanes_total = max(align_to(pages_per_seq, p) * page_size // 128
                         for p in num_kv_pages_per_block)
```

which is the **width of the score matrix**, i.e. `num_positions`. At `pages_per_seq = 9`
(`modelled`: `ceil(8704 / 1024) = 9`):

| bkv_p | `align_to(9, bkv_p)` | max over the tuple `(bkv_p, 2, 2)` | num_positions |
| ---: | ---: | ---: | ---: |
| 1 | 9 | max(9, 10, 10) = 10 | 10240 |
| 2 | 10 | max(10, 10, 10) = 10 | 10240 |
| 3 | 9 | max(9, 10, 10) = 10 | 10240 |
| 4 | 12 | max(**12**, 10, 10) = 12 | **12288** |

So **bkv_p ∈ {1, 2, 3} are exactly like-for-like** — same 10240 positions, same score matrix, same
stage B and init (48.63 / 28.47 µs identically, which is itself a consistency check that the probe
held everything else fixed). **bkv_p = 4 is not**: it inflates the score matrix by 20 %, and its
stage B (49.33) and init (29.78) both move accordingly. Its −0.79 % is a mixture of a block-size
effect and a 20 % larger score matrix and means nothing on its own. It is reported and excluded.

Pinning the prefill and mixed tuple entries at 2 is what makes the {1,2,3} comparison clean: it
keeps the `max(...)` from being driven by a swept entry. A sweep that moved all three entries
together would have confounded every arm.

## A3. The second reading: not bandwidth-bound, *block*-bound

The brief asked for the flat/moves reading for area 3. It moves. But look at the bytes.

`modelled`, at `pages_per_seq = 9`:

| bkv_p | DMA blocks per seq = `ceil(9 / bkv_p)` | pages actually fetched per seq | stage A µs |
| ---: | ---: | ---: | ---: |
| 1 | 9 | 9 | 136.24 |
| 2 | 5 | **10** (one page of overfetch) | 113.58 |
| 3 | 3 | 9 | 90.87 |

**The three like-for-like arms move essentially the same number of bytes** — 9, 10, 9 pages — and
the *fastest* arm (bkv_p=3) moves the same bytes as the *slowest* (bkv_p=1), while the arm that
moves the **most** bytes (bkv_p=2, with a page of overfetch) sits in the middle. Total HBM traffic
is flat to within 11 % across a range over which stage A varies by 40 %.

A bandwidth-saturated fetch cannot behave this way. Bigger DMAs at constant bytes only help when
what is limiting you is per-block cost — descriptor issue, semaphore wait, pipeline depth, grid
overhead — not the HBM pipe. **Stage A is DMA-*block*-bound (latency / occupancy), not
DMA-*bandwidth*-bound.**

The refinement is not cosmetic; it changes two other decisions:

- **It weakens the roofline's decode story.** `report.md` §4 puts the K-cache read at 95.4 % of
  decode HBM traffic and calls the floor 47.5 µs HBM-bound. If the fetch is not saturating HBM,
  that floor is not the binding constraint at the current blocking, and the gap between the
  modelled 47.5 µs and the measured stage A is largely per-block overhead rather than bytes.
- **It caps what P6 can be worth.** P6 asks whether narrowing the read from 256 to 129 bytes is
  charged less. Even if the answer is "yes, fully charged by bytes", halving the bytes of a term
  that is not bandwidth-limited does **not** halve its time. P5 established the padding is the
  indexer's to reclaim; P7a says reclaiming it buys less than the byte count suggests. **The
  ordering implication is that `bkv_p` tuning is the cheaper and better-evidenced lever, and it
  should be taken first.**

I do not have a clean model of the per-block cost. A straight linear fit in block count does not
close: `stage A ≈ a·nblocks + c` fitted on (9, 136.24) and (5, 113.58) gives a = 5.67 µs/block,
c = 85.25 µs, which predicts 102.2 µs at nblocks = 3 against a measured 90.87 µs. So the mechanism
is *not* a fixed cost per block either. Recorded as unexplained rather than curve-fitted.

## A4. Caveats — read these before shipping `bkv_p = 3`

1. **The absolute times are from a contended pod and do not transfer.** This shard shared the
   4-chip pod with three other shards (one crashed immediately, two ran heavy compiles and IO).
   `/xprof-llo` and `/run-command-in-cdk` both warn that fanned-out wall-clock is contended. The
   **within-job relative comparison stands** — all four arms ran in the same process, on the same
   chip, back to back, so the contention is common-mode — and `report.md` §6 already requires that
   a challenger beat the incumbent *in the same job*, which this does. The absolute numbers do not.
2. **The absolute numbers also disagree with the baseline of record, by more than the documented
   spread.** `report.md` §6 has `prod_decode_b64_ctx8704` at 221.4 µs total / 109.2 µs derived
   stage A (`j-8b1e7acc`). This job's incumbent arm reads 190.68 / 113.58 — the total is 13.9 %
   *faster*, which is outside the +8.5 % run-to-run spread `report.md` measured on the total. That
   discrepancy is **unexplained** and I did not chase it. It is one more reason to treat only the
   in-job ratios as usable.
3. **`bkv_p = 3` may be exploiting an alignment coincidence.** At `pages_per_seq = 9`, 3 divides
   evenly (9 = 3 × 3) and 2 does not (`align_to(9,2) = 10`, the page of overfetch above). Context
   length in production varies per request, and 3 is not a power of two. Before this becomes the
   default, sweep it at two or three more context lengths — in particular one where
   `pages_per_seq mod 3 != 0` — or the win will be real at ctx 8704 and absent elsewhere.
4. **Derived stage A carries both other terms' error.** `stage A = total − stage B − init`. Stage B
   and init were each measured once per distinct `num_positions` and reused, so within {1,2,3} the
   subtraction is a constant and the *differences* are clean, but the absolute stage A values are
   only as good as those two measurements.
5. **This is a kernel-level number, not an end-to-end one.** The indexer runs in 21 of 78 layers.
   Nothing here says what −11.91 % on this call is worth to nightly throughput; that needs the
   end-to-end benchmark.

## A5. What P7a settles

- **Area 2's top candidate is confirmed and is larger than estimated.** The prior was a one-tuple
  change worth ≈2.5–3 %. Measured at the kernel call it is **−11.91 %** like-for-like, from
  `num_kv_pages_per_block=(2,2,2)` to `(3,2,2)` at `mla_attention_op.py:119`. Subject to caveat A4.3.
- **Area 3's discriminator returns MOVES**, and §A3 sharpens it: block-bound, not bandwidth-bound.
- **Not settled:** whether `bkv_p = 3` generalizes across context lengths, and whether the win
  survives end to end.

---

# P7b — Does prefill's large intermediate spill?

## VERDICT

**No — and the premise it was built on is false from source.** The prefill kernel's VMEM working
set has a **static upper bound of 12 MiB**, identical to the decode kernel's, against a modelled
32.00 MiB einsum intermediate. The 32 MiB intermediate is **not materialized in VMEM**. There is no
capacity pressure to relieve at 32 MiB, so **"prefill is VMEM-capacity-bound" is not supported** and
the blocking-side fixes (reduce over `h` incrementally, shrink `bq_sz`, accumulate in bf16) are
aimed at a constraint that is not binding.

> **Everything in P7b is static compile-time analysis, not measurement.** No number below is a
> timing, a throughput, or a benchmark result, and none of it may be reported as one or reach
> `/log-experiment`.

## B0. The premise, refuted from source before anything ran

The brief states that "the scoped VMEM ceiling defaults to 32 MiB unless `vmem_limit_bytes` is
raised". On this branch it is not 32 MiB:

- `streamindex_topk.py:28` — `DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024`
- `streamindex_topk.py:535` — `vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES`
- the production call site (`mla_attention_op.py:107-122`) does **not** override it
- the microbenchmark (`instruments/bench_streamidx_topk.py:770-777`) does **not** override it

**100 MiB is the scoped ceiling in force, and it is already above the 64 MiB physical VMEM.** So
"32-33 MB sitting against a 32 MiB scope" cannot be the mechanism as stated — that scope does not
exist here. Raising `vmem_limit_bytes` as a "second arm" is a no-op: it is already raised, by
default, everywhere.

That refutation is why the probe was built as a **capacity bisection** rather than as a wall-clock
A/B. A wall-clock number cannot separate VPU-bound from VMEM-bound, and the brief was explicit
about that; a bisection produces a bound that cannot be misread as a timing.

## B1. The `vmem_limit_bytes` bisection

`static compile-time analysis (j-9e113359, shard 2)`. Both shapes walked over
`VMEM_LADDER_MIB = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 100, 128)`. A rejected limit
means the compiler could not fit the kernel's scoped VMEM inside it; the smallest limit that
compiles is a direct upper bound on the working set.

| shape | 4 MiB | 8 MiB | 12 MiB | ≥ 16 MiB | `SMALLEST_LIMIT_THAT_COMPILES` |
| --- | :---: | :---: | :---: | :---: | ---: |
| `prod_prefill_q1024_ctx9216` (bq_sz=128, num_bq=8, bkv_sz=2048, ctx 9216) | REJECTED | REJECTED | **OK** | OK | **12 MiB** |
| `prod_decode_b64_ctx8704` (bq_sz=1, bkv_sz=2048, 64 seqs) — the reference | REJECTED | REJECTED | **OK** | OK | **12 MiB** |

12 MiB is **37.5 % of the 32 MiB Mosaic scoped default** and **18.75 % of the 64 MiB physical VMEM**
on tpu7x.

**The two shapes bottom out identically, and that is the load-bearing observation.** `streamindex_topk`
issues all three pallas calls (decode `-d-`, mixed `-m-`, prefill) regardless of which shape is being
executed, so the floor is set by whichever call has the largest scoped requirement — not by the shape
under test. That is why the reference arm was run: without it, 12 MiB for prefill alone would have
looked like a prefill-specific fact.

## B2. What the rejection messages name — and what they do not

The `CompileTimeScopedVmemOom` diagnostics name the **score buffers**, not the einsum intermediate:

- 4 MiB rejection: `%StreamIdxTC-d-bq_1-bkvp_2.2 = f32[1024,80,128]{2,1,0:T(8,128...`
- 8 MiB rejection: `%StreamIdxTC-m-bq_128-bkvp_2.1 = f32[1024,80,128]...`

`modelled`: `f32[1024, 80, 128]` = 1024 · 80 · 128 · 4 B = 41,943,040 B = **40.0 MiB** — the stage A
score matrix (1024 query tokens × 80 sublanes × 128 lanes = 10240 positions).

`modelled`, the intermediate the brief is about: `bq_sz · H · bkv_sz · 4 = 128 · 32 · 2048 · 4 =
33,554,432 B = 32.00 MiB`.

**The einsum intermediate is named nowhere in the diagnostics, and 12 MiB is less than a third of
its modelled 32.00 MiB.** It therefore cannot be resident in VMEM as a whole. Whatever the kernel is
doing with the `[bq_sz, H, bkv_sz]` product, it is consuming it incrementally — blocked over `h`,
or fused into the reduction — and never materializes it. Area 2 and area 4 both computed the 32 MiB
figure correctly *as an arithmetic product*; the mistake was in assuming the product is a buffer.

## B3. The LLO dump — captured, read separately

`/xprof-llo`'s dump-side capture ran as shard 3 and succeeded:

```
[P7b] LIBTPU_INIT_ARGS: --xla_jf_dump_to=<out>/jf --xla_jf_dump_llo_text=true --xla_mosaic_dump_to=<out>/mosaic
[P7b] dump jf: 22911 files, 2329.7 MB
[P7b] dump mosaic: 202 files, 179.4 MB
```

tarred to `p7b-llo.tar.gz`, 391,330,930 bytes, in the job's `command-container` output directory.
Decode and prefill were run in one process so both programs land in one dump and can be diffed. The
flags were set **before `import jax`**, as `LIBTPU_INIT_ARGS` (not `XLA_FLAGS`) — the two failure
modes that leave a job green with no data.

Non-empty dumps mean the capture gated clean. Reading it (`get_llo_fit_summary`,
`get_llo_static_utilization` on `StreamIdxTC-m-bq_128-bkvp_2` with `StreamIdxTC-d-bq_1-bkvp_2` as
reference) for VMEM high-water and spill/fill counts is **in flight at time of writing and is not
recorded here**. B1 and B2 do not depend on it: the bisection is an independent, self-contained
static bound, and it already answers the capacity question. The dump would add the spill counts and
the per-unit occupancy that would positively identify the VPU-bound alternative, rather than merely
failing to find the VMEM-bound one.

## B4. What P7b settles

- **Settled:** the prefill kernel's VMEM working set is ≤ 12 MiB, the same as decode's, and the
  32.00 MiB intermediate is not materialized. `briefs/area4-layout-precision.md` §D5's hypothesis —
  "prefill's binding term may be VMEM capacity/spill" — is **not supported**.
- **Settled:** raising `vmem_limit_bytes` is not an available lever. It is already 100 MiB by
  default at every call site, above physical VMEM.
- **Not settled by P7b alone:** that prefill is therefore VPU-bound. Ruling out one hypothesis does
  not establish the other. `report.md` §4's prefill floor of 623.3 µs "VPU-bound by 14×" rests on
  two `assumed` inputs (the 1.72e12 ops/s VPU rate and "3 VPU ops per element"), so it is not
  independent confirmation. The per-unit occupancy in the LLO fit summary is what would confirm it.
- **Also not settled:** whether the 40.0 MiB `f32[1024,80,128]` score matrix — which *is* named, and
  is larger than the intermediate that was being hunted — is worth attacking. It is the stage A
  output and lives in HBM; P7b says only that its scoped requirement is what sets the 12 MiB floor.

---

## Provenance

| number | tag |
| --- | --- |
| every µs in the P7a table | `measured (j-9e113359-47c6-41b2-ad85, shard 1)`, shape `prod_decode_b64_ctx8704`, `q f8_e4m3fn[64,32,128]` / `weights bf16[64,32]` / `cache_kv u8[576,256,4,256]`, best-of-N, **contended pod — ratios only** |
| `num_sublanes_total`, blocks/seq, pages fetched | `modelled` — arithmetic shown inline |
| every MiB in P7b | **static compile-time analysis** `(j-9e113359, shard 2)` — not a timing |
| `f32[1024,80,128]` = 40.0 MiB, `128·32·2048·4` = 32.00 MiB | `modelled` — arithmetic shown inline |
| `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` | source, `streamindex_topk.py:28` |
| 221.4 / 109.2 µs baseline of record | `report.md` §6, `measured (j-8b1e7acc)` — quoted, not re-measured |
| 47.5 µs decode floor, 623.3 µs prefill floor | `report.md` §4, `modelled`, and resting on `assumed` VPU inputs |
