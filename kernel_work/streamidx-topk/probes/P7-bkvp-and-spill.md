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

**Independent corroboration from P7b's LLO dump** (`static compile-time analysis (j-9e113359,
shard 3)`, §B3.4): the decode program `StreamIdxTC-d-bq_1-bkvp_2.2` shows **MXU occupancy 4.61 %,
MXU-only 0.2 %, overlap 6.3 %, mem-stall-MXU-idle 54.9 %, SALU bubble 15.5 %, fully idle 15.5 %.**
Thirty-one percent of the decode body is scalar bubble or outright idle. A kernel saturating HBM
does not idle for 15.5 % of its bundles; a kernel paying per-block address-computation and
fetch-issue cost at bq=1 does exactly this. Two instruments, one reading.

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

## VERDICT — two parts, and both matter

**1. Not VMEM-capacity-bound. Established twice, by independent methods that agree.** The prefill
kernel's total VMEM working set is **9.52 MiB** — against a configured `vmem_limit_bytes` of
**100 MiB** and a physical VMEM of at least 64 MiB. There is no capacity pressure to relieve, and
the blocking-side fixes the hypothesis pointed at (reduce over `h` incrementally, shrink `bq_sz`,
accumulate in bf16) are aimed at a constraint that is not binding.

**2. It nonetheless spills, heavily — but it is spilling *registers*, not VMEM.** 8,458 spills +
10,130 fills over 11,658 bundles (**1.59 per bundle**), with **1,506 live vector registers against
64 physical — 23.5× oversubscribed.** 97.1 % of all vector stores and 87.8 % of all vector loads
in the compiled body are spill traffic. This is a real and large finding, and it is *not* what the
hypothesis predicted: the pressure is inside a single grid step's register allocation, not against
a VMEM ceiling.

**3. The 32.00 MiB intermediate is never a buffer, but it is produced in full.** No VMEM allocation
in the program exceeds 2.0 MiB. The score staging buffer is `f32[2,128,16,128]` — the H=32 head
axis is *absent*, exactly 1/32 of the modelled figure, because heads are reduced inside the MXU
accumulation. Yet 8,744 MXU result pops × 1024 f32 = **34.16 MiB** of intermediate is generated,
6.6 % above the 32.00 MiB model. It is streamed and consumed — and ~4.87 MiB of it is resident at
any moment **in the spill arena**, put there by the register allocator rather than by the kernel
author.

**Design decision settled:** stop looking for a VMEM capacity fix. **Design decision opened:** the
lever is *vector register pressure*, which is a different resource with a different fix.

> **Everything in P7b is static compile-time analysis** — issue slots and allocation sizes emitted
> by the compiler. No number below is a timing, a throughput, or a benchmark result. None of it may
> be reported as one or reach `/log-experiment`.

## B0. The premise, refuted from source before anything ran

The brief states that "the scoped VMEM ceiling defaults to 32 MiB unless `vmem_limit_bytes` is
raised". On this branch it is not 32 MiB:

- `streamindex_topk.py:28` — `DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024`
- `streamindex_topk.py:535` — `vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES`
- the production call site (`mla_attention_op.py:107-122`) does **not** override it
- the microbenchmark (`instruments/bench_streamidx_topk.py:770-777`) does **not** override it

**100 MiB is the scoped ceiling in force, above even physical VMEM.** So "32-33 MB sitting against a
32 MiB scope" cannot be the mechanism as stated — that scope does not exist here. Raising
`vmem_limit_bytes` as a "second arm" is a no-op: it is already raised, by default, everywhere.

That refutation is why the probe was built as a **capacity bisection plus an LLO dump** rather than
as a wall-clock A/B. A wall-clock number cannot separate VPU-bound from VMEM-bound; a bisection
produces a bound that cannot be misread as a timing, and the dump produces the spill counts.

## B1. The `vmem_limit_bytes` bisection

`static compile-time analysis (j-9e113359, shard 2)`. Both shapes walked over
`VMEM_LADDER_MIB = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 100, 128)`.

| shape | 4 MiB | 8 MiB | 12 MiB | ≥ 16 MiB | `SMALLEST_LIMIT_THAT_COMPILES` |
| --- | :---: | :---: | :---: | :---: | ---: |
| `prod_prefill_q1024_ctx9216` (bq_sz=128, num_bq=8, bkv_sz=2048, ctx 9216) | REJECTED | REJECTED | **OK** | OK | **12 MiB** |
| `prod_decode_b64_ctx8704` (bq_sz=1, bkv_sz=2048, 64 seqs) — the reference | REJECTED | REJECTED | **OK** | OK | **12 MiB** |

**The two shapes bottom out identically**, because `streamindex_topk` issues all three pallas calls
regardless of which shape is executed. The floor is set by whichever call has the largest scoped
requirement — §B3 identifies it as the prefill/mixed `-m-` program in both runs. Running the decode
shape as a reference is what made that visible; without it, 12 MiB would have looked like a
prefill-specific fact.

## B2. What the rejection messages name — and the correction the dump forced

The `CompileTimeScopedVmemOom` diagnostics name:

- 4 MiB rejection: `%StreamIdxTC-d-bq_1-bkvp_2.2 = f32[1024,80,128]{2,1,0:T(8,128...`
- 8 MiB rejection: `%StreamIdxTC-m-bq_128-bkvp_2.1 = f32[1024,80,128]...`

`modelled`: `f32[1024, 80, 128]` = 1024 · 80 · 128 · 4 B = 41,943,040 B = **40.0 MiB** — the stage A
score matrix (1024 query tokens × 80 sublanes × 128 lanes = 10240 positions).

> **Correction.** An earlier reading of this file inferred from these messages that the *score
> buffers* set the VMEM floor. The Mosaic IR says otherwise: `f32[1024,80,128]` appears as
> `%arg10`/`%arg11: memref<1024x80x128xf32, …, #tpu.memory_space<hbm>>` — **HBM, not VMEM** — read
> through `tpu.memref_slice … -> memref<?x16x128xf32, hbm>`. The only VMEM-space arguments are
> `%arg12..%arg15`, the four scratch operands in §B3.1. The diagnostic names the *instruction* whose
> scoped allocation overflowed, not the buffer that overflowed it. What actually sets the floor is
> §B3.1's 9.52 MiB.

`modelled`, the intermediate the brief is about: `bq_sz · H · bkv_sz · 4 = 128 · 32 · 2048 · 4 =
33,554,432 B = 32.00 MiB`. It is named nowhere in the diagnostics, and it is not a buffer — §B3.3.

## B3. The LLO dump

`static compile-time analysis (j-9e113359, shard 3)`. Capture: `LIBTPU_INIT_ARGS` (not `XLA_FLAGS`)
set **before `import jax`**, with `--xla_jf_dump_to`, `--xla_jf_dump_llo_text=true`,
`--xla_mosaic_dump_to`; decode then prefill in one process. Produced `jf: 22911 files / 2329.7 MB`
and `mosaic: 202 files / 179.4 MB`, tarred to `p7b-llo.tar.gz` (391,330,930 B). Read with this
repo's vendored `xprof-cli` (`get_llo_fit_summary`, `get_llo_static_utilization`,
`get_llo_schedule_analysis`, `get_llo_bundles`, `--bypass_cache=True`).

> **A trap worth recording.** Each program has **two compile instances**, one per shape run, and
> `get_llo_fit_summary` picks a "primary" that is not the obvious one — for `StreamIdxTC-m-…` it
> picked the instance from the **decode** run. The dump had to be split into per-run views by
> timestamp (decode = the run whose custom-call output is `f32[64,80,128]`; prefill =
> `f32[1024,80,128]`) and re-read. Here the two compiles turned out near-identical (prefill `-m-`:
> 11601 vs 11600 bundles, VMEM byte-for-byte the same) so no answer changed — but it would have if
> the shapes had diverged. Anyone repeating this must split by run.

Three kernel programs exist: `StreamIdxTC-m-bq_128-bkvp_2.1` (subject) and
`StreamIdxTC-d-bq_1-bkvp_2.2` / `.3` (reference; `.2` stages 4 kv blocks per step, `.3` stages 1).

### B3.1 VMEM: 9.52 MiB, and the headline number is wrong by 2.3×

**`get_llo_fit_summary`'s headline VMEM figure counts only the named scratch allocations and omits
the register-allocator spill arena.** Recomputed from the `…-53-post-ra.txt` allocation table over
every `space=vmem` allocation:

| program | named scratch | **spill arena** | **total VMEM** | spill slots |
| --- | ---: | ---: | ---: | ---: |
| **prefill `-m-bq_128`** | 4,399,104 B = 4.20 MiB | 5,586,944 B = **5.33 MiB** | **9,986,048 B = 9.52 MiB** | 1364 × 0x1000 |
| decode `-d-.2` | 4,370,432 B = 4.17 MiB | 1,429,504 B = 1.36 MiB | 5,799,936 B = 5.53 MiB | 349 × 0x1000 |
| decode `-d-.3` | 1,147,904 B = 1.09 MiB | 630,784 B = 0.60 MiB | 1,778,688 B = 1.70 MiB | 154 × 0x1000 |

Every spill slot is exactly 0x1000 = 4096 B = one 8×128 f32 vreg. The five named prefill
allocations, verbatim from post-ra:

```
#allocation1 u32[144,128]                        0x12000  =    73,728 B  internal scratch
#allocation2 s8[2,1,512,4,256]                  0x100000  = 1,048,576 B  kv cache block, double buffered
#allocation3 f8e4m3fn[2,1,128,32,128]           0x100000  = 1,048,576 B  q block (bq=128, H=32, D=128), double buffered
#allocation4 f32[2,1,128,32]                     0x20000  =   131,072 B
#allocation5 f32[2,128,16,128]                  0x200000  = 2,097,152 B  score block (bq=128 x 2048 kv), double buffered
```

**The two methods agree tightly.** 9.52 MiB is **> 8 MiB and ≤ 12 MiB** — exactly the bracket B1's
bisection found. It also explains why the *decode* shape was rejected at 8 MiB despite its own
program needing 5.53 MiB: the decode run compiles all three calls, and the 9.52 MiB `-m-` program is
the binding constraint in both runs. Two independent instruments, one answer.

Against the limits: 9.52 MiB is **29.7 % of a 32 MiB scoped default**, **14.9 % of 64 MiB physical**,
**7.4 % if physical is 128 MiB**. The operative limit in this capture was 100 MiB. Nowhere near any
ceiling on any reading.

### B3.2 Spills: 1.59 per bundle, and 23.5× register oversubscription

| | **prefill `-m-`** | decode `-d-.2` | decode `-d-.3` |
| --- | ---: | ---: | ---: |
| spills (stores) | **8,458** | 996 | 196 |
| fills (loads) | **10,130** | 999 | 197 |
| bundles | 11,658 | 2,779 | 757 |
| (spills+fills)/bundle | **1.59** | 0.72 | 0.52 |
| `VSTORE:SPILL` as share of `VSTORE` | 36.27/37.37 = **97.1 %** | 99.2 % | 99.1 % |
| `VLOAD:FILL` as share of `VLOAD` | 28.96/32.98 = **87.8 %** | 31.4 % | 26.4 % |
| register pressure (live vregs) | **1,506** | 376 | 195 |

64 architectural vregs exist (highest index observed `_v63`), so the prefill body is **23.5×
oversubscribed**. Zero `vld` from a named non-spill allocation appears in the steady state of any of
the three programs — every vector load there is a fill. Instruction-level evidence from inside the
largest stall run (`0x28aa-0x2d89`, 1248 bundles, `mem_stall_mxu_idle`):

```
0x28b0  { vcmask ;; vadd.f32 ;; vmul.f32 ;;
          %162437 = vst [vmem:[#allocation203_spill]] /*vst_source=*/%v131854_v4 ;;
          vmax.f32 ;; vld [vmem:[#allocation302_spill]] ;; vld [vmem:[#allocation1136_spill]] }
```

### B3.3 The 32.00 MiB intermediate: not a buffer, but produced in full

*Not a buffer.* No VMEM allocation exceeds 2.0 MiB. The score staging buffer `f32[2,128,16,128]` is
bq 128 × 2048 kv × f32, double-buffered — **the H=32 axis is absent**, exactly 1/32 of the modelled
32 MiB per buffer. Heads are reduced inside the MXU accumulation. Materialized as modelled, that one
allocation would read 32 MiB (64 MiB double-buffered) and the kernel would not compile at 12 MiB.

*But produced in full.* The compiled body contains **8,744 `vpop.f32.mrb[..].mxu{0,1}`** MXU
result-buffer pops, each yielding one 8×128 f32 vreg:

- `modelled` intermediate: 128 · 32 · 2048 = 8,388,608 f32 = **32.00 MiB**
- popped from the MXU, statically counted: 8,744 · 1024 = 8,953,856 f32 = **34.16 MiB** — **6.6 %
  above the model**, the excess being padding and tail

Corroborated by the matmul count: 1,024 `vmatmul` for 128·32·2048·128 = 1.074e9 MACs is 1.049e6
MACs per instruction = 256·256·16, i.e. a full 256-lane MXU tile with 16 pushed rows. One compiled
body = one (bq=128 × bkv=2048) grid step, generating the whole intermediate.

Where it goes: **7,292 of the 8,458 spill stores (86.2 %) take an MXU `vpop` result as their
`vst_source`**, landing in 1,190 distinct spill slots = **4.87 MiB**. So ~4.9 MiB of the 5.33 MiB
spill arena is holding matmul output. The intermediate *is* streamed through VMEM, in 4 KiB pieces,
by the register allocator — and it is invisible in the tool's headline VMEM number.

### B3.4 What each kernel is bound by

| static field | **prefill `-m-bq_128`** | decode `-d-.2` | decode `-d-.3` |
| --- | ---: | ---: | ---: |
| **VALU occupancy** | **94.19 %** (saturated 89.45 %) | 62.08 % | 57.00 % |
| MXU occupancy | 35.13 % (sat 29.11 %) | 4.61 % | 4.23 % |
| VLOAD / of which FILL | 32.98 / 28.96 | 38.18 / 11.99 | 32.98 / 8.69 |
| VSTORE / of which SPILL | 37.37 / 36.27 | 18.08 / 17.93 | 13.08 / 12.96 |
| SALU | 1.43 % | 13.35 % | 18.63 % |
| XLU | 10.16 % | 20.09 % | 18.25 % |
| EUP | 0.0 % | 0.0 % | 0.0 % |
| **fully idle bundles** | **0.5 %** | 15.5 % | 9.5 % |
| overlap (MXU+mem) | 35.2 % | 6.3 % | 5.9 % |
| mem-stall, MXU idle | 45.7 % | 54.9 % | 43.7 % |
| SALU bubble | 1.5 % | 15.5 % | 19.9 % |
| MXU matmuls | 1,024 | 32 | 8 |
| compiler-inserted `tpu.relayout` | 16 | 37 | 13 |

All three verdict lines read `VPU-serialized (vector ALU dominates, MXU starved)`, which is a label,
not evidence. The raw fields separate them, consistently:

- **Prefill is the better-scheduled of the two.** 0.5 % fully idle against 15.5 %, 35.2 % overlap
  against 6.3 %, SALU bubble 1.5 % against 15.5 %. Near-zero idle with high overlap is a fed
  pipeline; the decode's 15.5 % idle + 15.5 % scalar bubble is the small-batch address-computation
  tax at bq=1.
- **Prefill is VALU-saturated.** 94.19 % occupancy with 89.45 % of bundles saturating all four VALU
  slots. Opcode histogram of the body: 12,320 `vadd.f32`, 10,240 `vmul.f32`, 8,960 `vrot.slane` (a
  lane-reduction tree, rotate amounts 1/2/4/7), 8,192 `vmax.f32`, 2,096 `vsel` (top-k `-inf`
  masking) — against only 1,024 `vmatmul`. **The vector work is the post-matmul reduce/score/select
  chain, not the einsum.**
- **But the saturation is partly self-inflicted.** With `VSTORE:SPILL` at 97.1 % and `VLOAD:FILL` at
  87.8 %, spill and fill instructions occupy the same VLIW bundles as the arithmetic. The tool's own
  flag says so: *"VPU-active% is inflated by spill traffic."*
- Relayouts are **not** the problem: `post-infer-vector-layout` reads 0 on every function, so all
  132 in the dump are compiler-inserted, and prefill's 16 is the lowest of the three.

## B4. What P7b settles, and what it opens

**Settled:**
- The prefill working set is **9.52 MiB**, confirmed by two independent methods (bisection bracket
  8 < x ≤ 12 MiB; post-ra allocation sum 9.52 MiB). It is nowhere near any VMEM ceiling.
- `briefs/area4-layout-precision.md` §D5's hypothesis — "prefill's binding term may be VMEM
  capacity/spill" — is **half right in a way that matters**. There is no capacity problem. There is
  a severe spill problem, and it is register spilling, which §D5's proposed fixes do not address.
- Raising `vmem_limit_bytes` is not an available lever; it is already 100 MiB everywhere by default.
- The 32.00 MiB intermediate is not a buffer. Area 2 and area 4 both computed the product correctly;
  the error was assuming a product is an allocation.
- Prefill is **VPU-bound**, on the post-matmul reduce/score/select chain rather than on the einsum —
  the opcode histogram is 1,024 `vmatmul` against ~40,000 vector-arithmetic ops.

**Opened, and this is the actionable one:** the prefill body holds **1,506 live vregs against 64
physical**, and 86.2 % of its spill stores are MXU results. Shortening the live ranges of popped
MXU results — consuming each `vpop` into the reduction chain before pushing the next tile, instead
of accumulating ~1,190 slots' worth — is the lever the evidence points at. **Treat it as a
hypothesis, not a plan:** it is close to `xprof-cli`'s own third recommendation, which ships
self-labelled `TRACK RECORD 0/5 — directly refuted twice`. The evidence here is counted facts
(7,292-of-8,458 vpop-sourced spills; 1,506 vs 64 registers) rather than the tool's inference, but
the base rate on this lever is bad. Refutable prediction: spills/fills fall toward the decode's
~1/bundle, `VSTORE:SPILL` drops well below `VSTORE`, and the spill arena shrinks from 5.33 MiB
toward ~1 MiB (total VMEM ~5 MiB). That is a cheap `/xprof-llo` re-capture, not a benchmark.

**Not settled:**
- **How much of the 45.7 % `mem_stall_mxu_idle` is spill traffic versus genuine memory latency.**
  These are entangled and the dump cannot separate them. What the dump *does* rule out is the
  capacity story.
- **Whether physical VMEM on this target is 64 or 128 MiB.** `xprof-cli` hardcodes 128 MiB; the
  brief says 64 MiB. Not resolvable from the dump, and it changes nothing — 9.52 MiB is far under
  either.
- **Whether the 1,364 spill slots overlap in address.** post-ra prints no offsets for spill
  allocations, so 5.33 MiB is an upper bound on the arena. The 8/12 MiB bisection bracket is
  consistent with it being exact.
- **Any timing.** There is no trace-side capture in this tarball — `jf/` and `mosaic/` only, no
  `.xplane.pb`. No `get_llo_utilization` (needs a sampled capture, which this is not) and no step-0
  `xprof-dump` gate was possible. Nothing here maps to a fraction of wall-clock.
- **Which decode variant (`.2` or `.3`) carries production's decode work, and in what proportion.**
  Both are emitted; the dump gives no invocation counts. Both are reported.
- `report.md` §4's prefill floor of 623.3 µs "VPU-bound by 14×" is **not** independently confirmed
  by this. It rests on two `assumed` inputs (1.72e12 ops/s VPU rate; "3 VPU ops per element"), and
  the dump's opcode histogram — ~40,000 vector ops per grid step — is a chance to replace that
  assumption with a count. That recomputation was not done here.

Artifacts, if anyone needs to re-read the dump: `/home/annyan_google_com/llo_p7b/p7b-llo/{jf,mosaic}`
(2.5 GB), with per-run views at `/tmp/jf_runA` (decode) and `/tmp/jf_runB` (prefill). These are
scratch and will not survive; the tarball in `j-9e113359`'s outputs is the durable copy.

---

## Provenance

| number | tag |
| --- | --- |
| every µs in the P7a table | `measured (j-9e113359-47c6-41b2-ad85, shard 1)`, shape `prod_decode_b64_ctx8704`, `q f8_e4m3fn[64,32,128]` / `weights bf16[64,32]` / `cache_kv u8[576,256,4,256]`, best-of-N, **contended pod — ratios only** |
| `num_sublanes_total`, blocks/seq, pages fetched | `modelled` — arithmetic shown inline |
| B1 bisection ladder | **static compile-time analysis** `(j-9e113359, shard 2)` — not a timing |
| every VMEM byte count, spill/fill count, occupancy %, opcode count, bundle count in B3 | **static compile-time analysis** from the LLO/Mosaic dump `(j-9e113359, shard 3)` — issue slots and allocation sizes emitted by the compiler, **not measured time** |
| `f32[1024,80,128]` = 40.0 MiB; `128·32·2048·4` = 32.00 MiB; 8,744·1024 = 34.16 MiB; 23.5× | `modelled` — arithmetic shown inline |
| `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` | source, `streamindex_topk.py:28` |
| 221.4 / 109.2 µs baseline of record | `report.md` §6, `measured (j-8b1e7acc)` — quoted, not re-measured |
| 47.5 µs decode floor, 623.3 µs prefill floor | `report.md` §4, `modelled`, resting on `assumed` VPU inputs |
| physical VMEM = 64 MiB | `assumed` (brief); `xprof-cli` asserts 128 MiB — **unresolved** |
