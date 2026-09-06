# Wave-2 combined sheet — GLM-5.2 DSA Lightning Indexer top-k

Input: the eight wave-1 sheets in this directory, `references/seams.md`, and the sources
`streamindex_topk.py`, `sparsecore_topk.py`, `mla_attention_op.py`, `roofline.py`,
`instruments/shapes.py`.

**This sheet chooses nothing and ranks nothing.** Every candidate below is stated with its cost and
its predicted effect and left for the wave-4 designer.

**Sign convention.** Every percentage is a change in **time**; negative is faster. Signed
percentages appear only in a `Δ time %` column. Prose says the direction in words.

**Provenance** on every number, one of exactly four: `measured (job)`, `modelled`, `assumed`,
`unmodelled`. A term valued at zero is `unmodelled`, never `0`.

Measurements six of the eight sheets did not have, from `j-31ca6c31-a430-4463-8dbf`
(JobStatus **Succeeded**, 4/4), medians in µs from `instruments/bench.py`, two 20-iteration blocks:

| row | full | `sparsecore_topk` alone | SC share | block-to-block spread |
|---|---|---|---|---|
| A-D64-9k | 365.1 | 183.6 | 50% | 1.0% |
| A-D32-8k | 264.8 | 175.6 | 66% | 1.9% |
| A-D16-8k | 278.7 | 200.5 | 72% | 1.9% |
| A-P1024-8k | 841.6 | 556.8 | 66% | 0.8% |
| A-P512-8k | 579.4 | 352.1 | 61% | 0.1% |
| B-P1024-1M | 66958.7 | — | — | 0.0% |

---

# (a) Reconcile

## a0. The four assigned tasks, resolved

### a0.1 `_pick_partition`: what `p` actually is

`sparsecore_topk.py:360-377`, quoted:

```python
def _pick_partition(b: int, n: int, k: int) -> int:
    p_min = _cdiv(n, MAX_SLICE_WORDS)          # MAX_SLICE_WORDS = 65536
    p = 1
    while p < p_min:
        p *= 2
    max_p = 32 // b if b < 32 else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= LANES and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p
```

The two loops do different jobs and **the second can only raise `p`, never lower it**. The first
raises `p` to the smallest power of two that makes the slice fit `MAX_SLICE_WORDS`; the second
raises it further *only when `b < 32`*, to fill idle subcores. There is no contradiction — area 6's
"reads contradictory" is a misreading, and the settling fact is that `while p*2 <= max_p` is
immediately false whenever `p_min >= max_p`.

**`p` at every shape in `shapes.py`** (`n` per a0.2 below; `k`=2048; `LANES`=16):

| shape | `b` | `n` | `p_min` | `max_p` | growth fires? | **`p`** | stages | `W` per stage | `Σ W·slice_len` = `X_pad` |
|---|---|---|---|---|---|---|---|---|---|
| A-D16-8k | 16 | 10240 | 1 | 2 | **yes** → 2 | **2** | 2 | 1, 1 | 5120 + 4096 = **9216** |
| A-D32-8k | 32 | 10240 | 1 | 1 | no | **1** | 1 | 1 | **10240** |
| A-D64-8k / -9k | 64 | 10240 | 1 | 1 | no | **1** | 1 | 2 | **20480** |
| A-P512-8k | 512 | 10240 | 1 | 1 | no | **1** | 1 | 16 | **163840** |
| A-P1024-8k | 1024 | 10240 | 1 | 1 | no | **1** | 1 | 32 | **327680** |
| B-D*-* | 16 | 1048576 | 16 | 2 | no (2 < 16) | **16** | 2 | 8, 1 | 8·65536 + 32768 = **557056** |
| B-P1024-* | 1024 | 1048576 | 16 | 1 | no | **16** | 2 | 512, 32 | 512·65536+32·32768 = **34603008** |

`W = cdiv(b·p, 32)` for stage 1 and `cdiv(b, 32)` for stage 2 (`:148`, `:418-457`); stage-2
slice length is `align_to(p·min(k, n/p), 16)`.

**Verdicts on the four sheets.**
- **Area 2 is right at (B)**: `p_min = cdiv(1048576, 65536) = 16` → `p = 16`. ✅
- **Area 6 is right at (B)** on the value (16) and right that `max_p = 32//16 = 2`; its "reads
  contradictory" is withdrawn — the growth loop simply does not execute. ✅ with the note above.
- **Area 4's "p=16 at (B)" is right.** ✅
- **Area 5's `p = 2^ceil(log2(n/65536))` is right at (B) and wrong at (A) `b=16`**, where it gives
  1 and the code gives 2. Correction owed; the settling line is `:370` `max_p = 32 // b if b < 32
  else 1`.
- **`roofline.py:41` has the same defect** — `p = 1 if n <= 65536 else 2**ceil(...)` omits the
  growth branch entirely. Correction owed. It changes `sc` bytes at A-D16-8k only.

### a0.2 `n` at (A): **10240, not 9216**

Settled by `streamindex_topk.py:603-605`:

```python
num_sublanes_total = max(
    align_to(pages_per_seq, bkv_p) * page_size // 128
    for bkv_p in num_kv_pages_per_blocks)
```

`num_kv_pages_per_blocks = BKV_P = (2, 2, 2)` (`shapes.py`), `pages_per_seq = 9`, `page_size = 1024`
→ `align_to(9,2)·1024/128 = 10·1024/128 = 80` sublanes → `80·128 = 10240` columns. That figure is
what allocates the score buffer (`:725-726` `out_shape=(T, num_sublanes_total, 128)`), what the
`-inf` fill is sized by (`:745-746`), and what `sparsecore_topk` receives after `:811`
`scores = scores.reshape(q.shape[0], -1)`. **Areas 4 and 7 are correct; `roofline.py`,
`shapes.py` and `report.md` are 11.1% low.**

Artifacts to correct, with the line:
- `roofline.py:31` — `n, bkv_sz, P = pages_per_seq * psz, ...` → must be
  `align_to(pages_per_seq, bkv_p) * psz`. Affects `inf_fill` and `sc` at all five (A) rows.
- `instruments/shapes.py:177` — `meta = dict(..., n=pps*psz, score_bytes=T*pps*psz*4, ...)`, same
  defect, same 11.1%.
- `report.md` — every place it says `n=9216` at (A).
- At (B), `align_to(2048, 2) = 2048`, so `n = 1048576` is unaffected. ✅

**Area 7's `bkv_p ∈ {1,3,9}` observation is correct and is a real candidate**, because
`align_to(9, bkv_p) = 9` exactly for those three. But it is a `max` over *all three* entries of
`num_kv_pages_per_blocks`, so **all three slots must move together** or the padding stays. And the
choice is not free — see a0.2b.

#### a0.2b The `bkv_p` trade at (A) is not one-sided (new, from combining areas 2, 4, 6, 7)

`nbkv = cdiv(kv_len_max, bkv_p·page_size)`; the kernel fetches `bkv_p` pages per iteration
regardless of how many the sequence has.

| `bkv_p` at (A) | `align_to(9,·)` | score `n` | in-body iters at 8k | KV bytes fetched | per-seq block | TC VMEM (decode `sbs=4`) |
|---|---|---|---|---|---|---|
| 1 | 9 | **9216** | 8 | 8192 tok (exact) | 256 KiB | 2.10 MiB |
| **2 (incumbent)** | 10 | 10240 | 4 | 8192 tok (exact) | 512 KiB | 4.09 MiB |
| 3 | 9 | **9216** | 3 | 9216 tok (+12.5%) | 768 KiB | 6.13 MiB |
| 4 | 12 | 12288 (+20%) | 2 | 8192 tok | 1 MiB | 8.16 MiB |
| 8 | 16 | 16384 (+60%) | 1 | 8192 tok | 2 MiB | 16.28 MiB |
| 9 | 9 | **9216** | 1 | 9216 tok (+12.5%) | 2.25 MiB | 18.31 MiB |

All `modelled` from the shape algebra; VMEM figures `modelled` from the scratch shapes at
`streamindex_topk.py:640-700`. Three readings the designer must weigh and this sheet does not:
- `bkv_p=1` is the only padding-free option with no KV overshoot, and it doubles the iteration
  count and drops the per-sequence block to 256 KiB — below area 4's 512 KiB–2 MiB bandwidth knee.
- `bkv_p=9` removes the padding and collapses the loop to one iteration, at 12.5% more KV bytes.
  KV is 94.8% of the (A) ledger (`roofline.py`), so that is **+11.8% of modelled traffic** against
  removing an 11% score-padding term that is 1.3% of the ledger. Net traffic: worse.
- `bkv_p=4` and `8` make the padding *worse*, which is why only `{1,3,9}` are on the table.

### a0.3 Area 5's cost model, repaired — the SC time is set by a **padding** quantity

Area 5 fitted `T ≈ C + W·F + A·M` with `C ≈ 150 µs`, `F ≈ 73 µs/wave at n_p = 65536`, and flagged
(its item 7) that A-P1024-8k at `W = 32` would predict ~3.0 ms inside an 832.9 µs measurement. The
measured SC time is **556.8 µs** `measured (j-31ca6c31)`. The wave-constant form is dead.

Define **`X_pad = Σ_stages W · slice_len`** — the total padded words the SC walks, summed over
subcore-waves. Four models fitted by least squares over **all 11 rows** in `MEASURED` that carry an
`sc` number (the brief says 9; there are 11 — `roofline.py:97-111`; correction recorded):

| model | form | fit | mean abs residual | extrapolation to B-P1024-256k |
|---|---|---|---|---|
| **M1** | `T = C + f·X_pad` | **C = 165.4 µs, f = 1.2237 ns/word** | **5.4%** | 42.5 ms (83% of the 50.9 ms full op) — sane |
| M2 | `T = C + W·F` (area 5's stated form) | C = 322 µs, F = 62.6 µs | **78.6%** | 32.4 ms — but in-sample it is useless |
| M3 | `T = C + f·X_pad + g·X_live` | C=159, f=1.0139, g=0.4481 ns/word | 4.5% | 51.9 ms = **102% of the whole op** — impossible |
| M4 | `T = C + f·X_pad`, A-D64-8k dropped | C = 166.1, f = 1.2216 | 5.5% | 42.4 ms |

**M2 is falsified; M3 is over-fit** (it buys 0.9 points in-sample and predicts more than the entire
measured operation out-of-sample). M1 is the repaired form. All four `modelled` from
`measured (j-31ca6c31, j-59d0b337, j-284a8842)`.

**Per-row M1 residuals — reported per row, as instructed, not only as a fit:**

| row | `X_pad` | M1 predicted µs | measured SC µs | Δ time % (resid) |
|---|---|---|---|---|
| A-D64-8k | 20480 | 190.5 | 189.5 | −0.5% |
| A-D64-9k | 20480 | 190.5 | 183.6 | −3.7% |
| A-D32-8k | 10240 | 177.9 | 175.6 | −1.3% |
| **A-D16-8k** | 9216 | 176.7 | 200.5 | **+11.9%** |
| A-P1024-8k | 327680 | 566.4 | 556.8 | −1.7% |
| A-P512-8k | 163840 | 365.9 | 352.1 | −3.9% |
| B-D4-256k | 557056 | 847.1 | 795.9 | −6.4% |
| B-D4-512k | 557056 | 847.1 | 832.1 | −1.8% |
| B-D4-1M | 557056 | 847.1 | 823.5 | −2.9% |
| B-D1-1M | 557056 | 847.1 | 780.6 | −8.5% |
| **B-D16-256k** | 557056 | 847.1 | 1012.8 | **+16.4%** |

**Rows M1 explains** (9 of 11, all within ±6.5%, most within ±4%): both A-D64 rows, A-D32,
both A-prefill rows, and B-D4 at all three lengths. Note that B-D4-256k, -512k and -1M have
**identical `X_pad`** and measure 795.9 / 832.1 / 823.5 — a 4.5% band across a **4× change in live
data**. That is M1's strongest evidence and it is the same fact area 6 named at its item 8.

**Rows M1 does not explain, and why:**
- **A-D16-8k, +11.9%.** M1 says it should be the *fastest* SC row at (A) (`X_pad` 9216, less than
  A-D32's 10240) and it is the slowest. The residual is the two-stage path — see a0.4.
- **B-D16-256k, +16.4%.** This is the live-data row: among the five (B) rows with identical
  `X_pad`, the residual tracks live data monotonically — B-D1-1M (16 live slices) −8.5%, B-D4-256k
  (64) −6.4%, B-D4-1M (256) −2.9%, B-D16-256k (256 live over 16 waves) +16.4%. **A residual live
  term worth roughly 15–20% of SC time exists**, `modelled`, and it is second-order against a
  padding term that spans 5.8× across the same rows. It is not zero and must not be recorded as
  `0`; it is a term M1 leaves `unmodelled`.

**Mechanism repair, which is the part that matters downstream.** Per-wave cost is not constant: it
is ~12.2 µs/wave at `n_p = 10240` and ~70 µs/wave at `n_p = 65536`, i.e. 0.89–1.52 ns/word. Area 6
counts **6 passes** over the resident slice (`cvt`, `hist0`, three refinement walks, `emit`;
`sparsecore_topk.py:158-291`). So `f = 1.2237 ns/word ÷ 6 = 0.204 ns/word/pass` `modelled`, which
is ~3.3 ns per 16-lane chunk per pass. **Area 5's three candidate per-wave-constant mechanisms
(barrier, dispatch, histogram-zero) are all ruled out as the dominant term** — they would be flat
in `n_p` and the measurement is linear in it.

That single number prices area 6's E5, E6 and E7, none of which had a number before:

| row | remove 1 of 6 passes (E5, E6 or E7 alone) | remove 3 of 6 (all three) |
|---|---|---|
| A-D64-9k | −1.1% | −3.4% |
| A-D32-8k | −0.8% | −2.4% |
| A-P1024-8k | −7.9% | −23.8% |
| A-P512-8k | −5.8% | −17.3% |
| B-D4-256k | −10.2% | −30.5% |
| B-D1-1M | −7.7% | −23.0% |
| B-D16-256k | −5.5% | −16.6% |
| B-P1024-256k | −13.9% | −41.6% |
| B-P1024-1M | −10.5% | −31.6% |

`Δ time %` of the **full op**, `modelled` under M1. Area 6 marked all three `unranked` for want of
a roofline term; M1 supplies one.

### a0.4 A-D16-8k slower than A-D32-8k: **the `_pick_partition` growth branch is a measured loss**

A-D16-8k is 278.7 µs against A-D32-8k's 264.8 at half the sequences and half the live work, and the
SC component is also higher (200.5 vs 175.6) `measured (j-31ca6c31)`. Decomposing:

| | A-D32-8k | A-D16-8k |
|---|---|---|
| `p` | 1 | **2** |
| SC kernel launches | 1 | **2**, plus 2 `jnp.take_along_axis` (`:439`, `:452`) |
| `W` stage 1 | 1 | 1 — **unchanged** |
| `X_pad` | 10240 | 9216 (10% *less*) |
| M1 predicted SC | 177.9 | 176.7 |
| measured SC | 175.6 | **200.5** |
| scoring (full − SC) | 89.2 | 78.2 |

The scoring half behaves exactly as expected (11.0 µs less at half the tokens). The whole inversion
is in the selector, and the mechanism is exact: at `b = 16`, `cdiv(16·1, 32) = cdiv(16·2, 32) = 1`,
so **the growth branch does not reduce the wave count at all**. It buys only fewer walked words per
subcore (`n/2` instead of `n`, worth −1.2 µs `modelled`) and pays a second `pl.kernel` dispatch plus
two XLA gathers over a padded `[16, 10240]` row (+26.1 µs `measured`, as the M1 residual).

**Candidate (not a recommendation):** suppress the growth loop when it would not reduce
`cdiv(b·p, 32)` — i.e. gate `:373` on `cdiv(b*p*2, 32) < cdiv(b*p, 32)`, or simply require
`b·p < 32`. Predicted effect at A-D16-8k: SC 200.5 → ~175.4, full 278.7 → ~253.6,
**`Δ time −9.0%`** `modelled`. At every other shape in `shapes.py` the branch never fires
(at (A), `b ≥ 32` → `max_p = 1`; at (B), `p_min = 16 ≥ max_p`), so the effect is `0` by
construction, not `unmodelled`.

**One caution on the evidence, from `roofline.py:86-95`.** `j-f90148cb` ran A-D16-8k twice in one
job and got 287.5 then 278.5 µs — 3.2% apart, against that row's own 1.9% block-to-block spread,
with `built in` falling 7.3 s → 4.3 s between them. The comment in `roofline.py` concludes the
inversion "is inside this band and is NOT established". **That conclusion is now too conservative
and should be corrected**: the SC-alone decomposition puts the anomaly at +14.2% on the SC
component with an exact source-level mechanism, which is well outside a 3.2% run-order band. But
the *prose* claim in `roofline.py:94-95` stands as written on the full-op number alone. The probe
below settles it for a few minutes of pod time.

### a0.5 B-P1024-1M vs B-P1024-256k: the 16 ms gap, and the term that closes it

Same `p` (16), same `W` (512 / 32), same `n_p`, same `X_pad` (34,603,008). M1 predicts **identical**
SC time. Measured: 66958.7 vs 50916.8 µs, **Δ = +16041.9 µs (+31.5%)** `measured (j-31ca6c31,
j-59d0b337 — a cross-job comparison; see the caution at the end of this item)`.

**First: a correction to the brief's own parenthetical.** It says "the `-inf` fill and score write
are 4 GiB vs 1 GiB". The `-inf` fill is **4 GiB in *both* rows**, because `:745-746` sizes it by
`num_sublanes_total·128 = max_model_len`, not by `seqlen`. Only the Pallas score write differs.
The settling lines are `:603-605` and `:745-746`.

Modelled ledger deltas (`roofline.py`), 256k → 1M:

| term | Δ MiB | provenance |
|---|---|---|
| KV read | +1536 | `modelled` |
| `-inf` fill | **0** | `modelled` — 4096 MiB in both |
| Pallas score write | +3072 | `modelled` |
| SC stage-1 read | +3072 | `modelled` |
| stage-2 glue / q,w / out | 0 | `modelled` |
| **total** | **+7680 MiB** | |

**Attribution attempts, all `modelled`:**

| reading | Δ µs | share of the 16041.9 measured |
|---|---|---|
| all 7680 MiB at TC's 3.70 TB/s | 2176.5 | 13.6% |
| area 3's stepped in-body model: +6144 iterations × (512 ns + 262144 B/3.48e12) | 3608.4 | 22.5% |
| MXU floor Δ (`t_mxu`) | 2868.3 | 17.9% |
| TC term alone (max of the three) | 3608.4 | 22.5% |
| **+ SC stage-1 aggregate read at the SC's own measured 337 GB/s** | **+9558** | **+59.6%** |
| **composite total** | **13166** | **82.1%** |

**This does not close on area 1's or area 2's traffic ledgers alone** — they account for at most
22.5% of it. It closes to 82% only if the SC's stage-1 read is charged at the SparseCore's
aggregate bandwidth (337 GB/s, area 5) rather than the TensorCore's.

**And that is in apparent tension with M1, which has no bytes term — but only apparently.** M1 was
fitted on rows whose SC aggregate stage-1 read is at most 64 MiB, where a 337 GB/s term is ≤199 µs,
i.e. inside its own residuals. At the two B-P1024 rows the aggregate read is 1 GiB and 4 GiB:

| row | live stage-1 slices | aggregate bytes | at 337 GB/s | share of measured full op |
|---|---|---|---|---|
| B-D4-256k | 64 | 16 MiB | 49.8 µs | 4.5% |
| B-D4-1M | 256 | 64 MiB | 199.1 µs | 10.9% |
| B-P1024-256k | 4096 | 1024 MiB | 3186 µs | 6.3% |
| B-P1024-1M | 16384 | 4096 MiB | **12745 µs** | **19.0%** |

**Reconciled statement, and this is the single most load-bearing modelling claim in this sheet:**
`T_sc ≈ 165.4 µs + 1.2237 ns/word · X_pad` is valid **only in the regime it was fitted**, SC
aggregate read ≤ 64 MiB. **It must not be extrapolated to either B-P1024 row without an added
bytes term.** With the term added: B-P1024-256k 42.5 + 3.2 = 45.7 ms of a 50.9 ms measurement
(90%); B-P1024-1M 42.5 + 12.7 = 55.2 ms of 67.0 ms (82%); their difference 9.5 ms, plus TC's
3.6 ms = 13.1 ms of the 16.0 ms gap (82%). Consistent on all three counts. `modelled`.

The remaining ~18% (≈2.9 ms) is `unmodelled`.

**Caution the designer must carry:** the two rows come from different jobs (`j-31ca6c31` and
`j-59d0b337`) and `roofline.py:93` says "no cross-JOB comparison here is worth anything" at sizes
under ~3%. This one is 31.5%, far outside that band, so the *gap* is real; but the 82% closure is
a fit to two numbers taken on different days and should be re-established in one job.

## a1. Where the per-iteration fixed cost lives — settled on data in hand

Area 3 fits `512 ns + bytes/3.48e12` per in-body KV iteration and reports the per-step model at
1.00–1.09×. Area 2 objects that the 512 ns term was fitted on `pallas_call` **grid steps** and it is
unknown whether it transfers to a hand-rolled `fori_loop` body, and offers the alternative that the
cost is per `make_async_copy`, of which `_fetch_bkv` (`streamindex_topk.py:160-197`) starts
`sbs × bkv_p` per iteration:

```python
for batch_idx in range(seq_batch_size):
    ...
    for i in range(bkv_p):
        _async_copy(...)
```

**The two predict opposite levers.** Per-iteration says: raise `bkv_p` (fewer iterations, same
bytes) → faster. Per-copy says: raise `bkv_p` → *more* copies per iteration and the same total
copies → no change; only **coalescing** the `bkv_p` copies into one strided descriptor helps.
Area 7's third reading — ~6% of the step is scalar-core page-index arithmetic — says neither, and
predicts a flat host-precomputed byte-offset table wins.

Backing out the implied fixed cost three ways, from the residual after the bytes term:

| shape | `sbs`×`bkv_p` copies/iter | implied **per-iteration** fixed ns | implied **per-copy** fixed ns |
|---|---|---|---|
| A-D64-8k | 4×2 = 8 | 524 | 65.5 |
| B-D4-256k | 4×2 = 8 | 582 | 72.8 |
| B-D1-1M | 1×2 = 2 | 511 | 255.5 |
| **spread** | | **14%** | **3.9×** |

`modelled` from `measured (j-284a8842, j-59d0b337)`. **The per-iteration reading is ~5× tighter on
the data in hand**, so area 3's transfer of the 512 ns grid-step constant to the `fori_loop` body is
supported and area 2's objection is not sustained on these three points — but all three points sit
in the same job family and none was designed to separate the hypotheses, so this is evidence, not
proof.

**Two corrections owed.**
- **Area 2's launch-count table is wrong by 4×.** It counts 128 / 512 / 1024 `make_async_copy`
  starts at A-D64-8k / B-D4-256k / B-D1-1M. The correct figures are iterations × `sbs` × `bkv_p` =
  **512 / 2048 / 4096**. The settling lines are `:160-197` (the double loop) and `:276`
  (`num_bkv = cdiv(max_i kv_lens[i], bkv_sz)`). Area 2's *conclusion* — that per-copy cost is a live
  hypothesis — is unaffected in kind, but every number attached to it moves.
- **Area 7's "~6% of the step is scalar page-index arithmetic" is an order of magnitude low if its
  own 6.9 ns/scalar-op slope holds.** At `sbs·bkv_p = 8` index computations per iteration with the
  clamp and bounds arithmetic each carries, ~440 ns of an 883 ns step is ≈50%, not 6%. Either the
  slope does not apply inside the loop body or the estimate does. This is unresolved and is the
  strongest reason to run the probe below rather than adjudicate on paper.

**The single sweep that separates all three** — `P-ITER`, five arms at **B-D1-1M**, bytes held
exactly fixed, one `run-command-in-cdk` job, four chips:

| arm | `bkv_p` | copies/iter | iterations | what each hypothesis predicts |
|---|---|---|---|---|
| 1 | 2 (incumbent) | 2 | 2048 | baseline |
| 2 | 4 | 4 | 1024 | per-iteration: **−about 0.5 ms**; per-copy: **no change**; scalar: −half the index ops |
| 3 | 8 | 8 | 512 | per-iteration: −0.75 ms; per-copy: no change |
| 4 | 4, the 4 copies coalesced into one strided `make_async_copy` | 1 | 1024 | per-copy: **−about 0.4 ms** on top of arm 2; per-iteration: no change vs arm 2 |
| 5 | 2, page offsets read from a host-precomputed flat `int32` table | 2 | 2048 | scalar: **−about 0.4 ms**; the other two: no change |

Arms 2-vs-3 separate per-iteration from nothing; arm 4 minus arm 2 isolates per-copy; arm 5 isolates
area 7. Any arm that moves by less than 3% is not resolvable — `roofline.py:86-95`.

## a2. The `-inf` fill: traffic share vs time share

Area 1 puts it at 56% of B-P1024's ledger with no consumer; area 4 calls it the largest violation of
its own mask-don't-pad rule; area 6 (E4) and area 7 propose removing or hoisting it; area 6 rates it
"≤10% of measured time". **All five are consistent, and the reconciliation is that the whole HBM
ledger is a small fraction of measured time at these rows.**

| row | fill MiB | share of modelled HBM ledger | at 3.70 TB/s | share of **measured** time | meas/model |
|---|---|---|---|---|---|
| A-D64-9k | 2.2 | 1.3% | 0.62 µs | 0.17% | 25.9× |
| B-D4-256k | 64 | 19.4% | 18.1 µs | 1.62% | 12.6× |
| B-P1024-256k | 4096 | 59.2% | 1160.8 µs | **2.28%** | 16.2× |
| B-P1024-1M | 4096 | 28.0% | 1160.8 µs | **1.73%** | 12.4× |

`modelled` from `roofline.py`; measured shares from `measured (j-31ca6c31, j-59d0b337)`. Area 1's
"56%" is 59.2% on the corrected `n` — close enough to stand. Area 6's "≤10% of measured time" is
correct and, at 1.7–2.3%, conservative in the right direction. **A term that is 59% of the traffic
and 2.3% of the time is exactly what a 16× meas/model ratio produces**; there is no contradiction to
resolve, only an instruction not to lead with it, which area 6 already gives.

**Two source findings that change the shape of the proposal, from tracing the consumer:**

1. **The fill genuinely has no consumer.** The Pallas kernel writes columns
   `[0, num_bkv·bkv_sz)` for every live row; `num_bkv·bkv_sz ≥ kv_len_max ≥ eff_len ≥ i_hi`; and
   `sparsecore_topk.py:171` gates every read on `valid = (j < i_hi) & ...`. Columns at or beyond
   `i_hi` are converted by `cvt` (up to the 16-word chunk boundary) and then excluded. `cvt` is
   integer bit arithmetic, so garbage there cannot trap. Area 1's premise is confirmed by source
   reading, not assumed.
2. **Area 7's "fatal across steps" staleness hazard for the hoist is therefore *not* fatal.** Its
   concern is that a persistent buffer carries last step's real scores where `-inf` is now required.
   Since nothing reads at or beyond `i_hi`, stale values there are as harmless as `-inf`.
   **Consequence: the hoist and the deletion are the same lever, and the persistent-donated-buffer
   route is legal.** That matters mechanically, because `input_output_aliases` (`:748`) needs *an*
   input buffer and JAX has no `jnp.empty`; a persistent donated scratch is how the fill is removed
   at all.
   Both statements are conditional on `_effective_row_lengths` (`streamindex_topk.py:504`,
   `visible = min(kv_len, token_pos//cr + 1)`) being correct — which is the same condition the
   incumbent already rests on.

**Candidate, priced:** delete the fill via a persistent donated score buffer.
`Δ time` −0.17% at A-D64-9k, −1.6% at B-D4-256k, −2.3% at B-P1024-256k, −1.7% at B-P1024-1M, all
`modelled`. Area 7's cross-layer variant saves 20/21 of the fills within a step, i.e. −2.2% of the
21-layer step at (B) prefill, `modelled` — the same order, and subsumed by deletion.

**One thing this does *not* buy, flagged because it is easy to over-claim:** 4 GiB per layer of HBM
*capacity* at (B) prefill is a real constraint on concurrency and on how many layers can be in
flight. That is a capacity argument, not a time argument, and area 6's gap-2 note ("the memory
argument for a fused kernel is a memory argument, not a speed argument") governs.

## a3. Empty launches: the two counts **agree**

- **Area 7:** 3 TC `pallas_call` launches per indexer layer × 21 layers = 63 per model step; 2 of 3
  are empty at any pure phase → **42 empty**; 11–18% of A-D64-8k.
- **Area 2:** counts *all* launches per layer (5 at (A), 8 at (B), of which 3 are the TC scoring
  launches) and says the grid extent is traced, so they cannot be compacted at trace time.

These are the same fact counted with different denominators, and they agree on the number that
matters. Verified from the `distribution` arithmetic (`streamindex_topk.py:700-810`):
`decode_batch_end = (distribution[0] // decode_req_batch_size) * decode_req_batch_size`, so at a
pure-decode step the mixed launch has `grid = (0,)` and at a pure-prefill step both decode launches
do. Cost: 42 × 19.7–31.9 µs = 827–1340 µs per step, i.e. 39.4–63.8 µs per layer of 365.1 µs =
**10.8–17.5%**, matching area 7's 11–18% `measured (j-284a8842)`.

**The remedies do not conflict either.** Area 2 is right that trace-time compaction is closed —
the extent is a traced value. Area 7's remedy is a runtime `lax.cond` (or `pl.when` at the call
site) skipping the launch when the extent is zero, which is open. **Recorded as one candidate, not
two.**

Minor correction: area 2's "5 launches at (A)" is 6 when `b = 16`, because A-D16-8k takes the
two-stage SC path (a0.1). Does not change the empty count.

## a4. `plsc.store_compressed`: **the reference is wrong, and that is owed back to `references/`**

`references/08-sparsecore.md:536-585` lists `store_compressed(ref, x, *, mask)` as "❌ refuses" and
states explicitly: "Turning `needs_layout_passes` off does not help **for these two** — this is a
Mosaic lowering gap, not the layout pass" `[verified: j-996520b8-2265-4c2c-b002,
sc_primitive_lowering]`.

The shipping kernel calls it:

```python
plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)], pos, mask=sel)   # sparsecore_topk.py:279
```

under `needs_layout_passes=False` (`:345`), in a kernel that measures 175.6–1012.8 µs on
**Succeeded** jobs `j-31ca6c31` and `j-59d0b337`. The call passes exactly the two features the
reference says are refused — a `TransformedRef` (`.at[pl.ds(...)]`) and an explicit `mask` — so this
is not a narrower form slipping through.

**Resolution: the reference is falsified, including its "turning `needs_layout_passes` off does not
help" sentence.** Area 5 reported the reference faithfully; area 5 is not at fault, and its
conclusion drawn from it (that a `cumsum` + `store_scatter` rewrite is required) should be
withdrawn.

**Owed back to `references/08-sparsecore.md`:** correct the table row and the accompanying sentence,
citing `sparsecore_topk.py:279` and `:345` and the two Succeeded jobs. The likely reason the
original probe failed is a difference in the probe, not in the primitive; the correction should say
so without guessing at which difference.

**Downstream consequence:** area 6's "already exploited, do not re-propose" note on
`store_compressed` (its item under §Already exploited) stands, and any variant is free to use the
primitive.

---

# (b) Reconcile the budget

## b1. TensorCore VMEM

**Judged against 64 MiB physical and the 31 MiB default scoped budget — never against the
`vmem_limit_bytes = 100 MiB` the incumbent asks for** (`shapes.py:VMEM_LIMIT`). Area 2's warning is
correct and is adopted here as the rule.

Scratch, from `streamindex_topk.py:640-700`, `modelled`:
`bkv = 2·sbs·bkv_p·psz·256`; `bq = 2·sbs·bq_sz·32·128`; `w = 2·sbs·bq_sz·32·4`;
`bo = 2·sbs·bq_sz·bkv_p·psz·4`. Note `bo/bkv = bq_sz/64`, so above `bq_sz = 64` the **score tile**,
not the KV tile, is the dominant consumer — a fact no single sheet states.

**(B) MIXED (`sbs=1`, `psz=512`), and the in-body iteration count at B-P1024-256k:**

| `bq_sz` | `bkv_p` | VMEM | vs 31 MiB | iterations | KV passes | source |
|---|---|---|---|---|---|---|
| 128 | 2 | **2.53 MiB** | fits | 2048 | 8 | incumbent |
| 1024 | 2 | **16.75 MiB** | fits | 256 | 1 | area 2 ✅ its figure is right |
| 512 | 4 | **13.13 MiB** | fits | 256 | 1 | **area 2 says 21 MiB — correction** |
| 512 | 8 | 22.13 MiB | fits | 128 | 1 | new here |
| **1024** | **4** | **25.25 MiB** | **fits** | **128** | **1** | **new here** |
| 1024 | 8 | 41.75 MiB | **over 31**, under 64 | 64 | 1 | new here |

**Correction to area 2:** its `bq_sz=512, bkv_p=4` figure of 21 MiB is 13.13 MiB. The arithmetic is
`2·1·512·32·128 + 2·1·512·32·4 + 2·1·4·512·256 + 2·1·512·4·512·4 = 13,762,560 B`. The correction
matters because it opens two configurations area 2 ruled out on budget, one of which
(`bq_sz=1024, bkv_p=4`) **strictly dominates area 2's own proposal on both iteration count (128 vs
256) and KV passes (1 vs 1), at 8.5 MiB more VMEM and still inside 31 MiB.**

**(A) DECODE (`sbs=4`, `psz=1024`, `bq_sz=1`)** — see the a0.2b table. Incumbent 4.09 MiB;
`bkv_p=9` is 18.31 MiB and fits.

**(B) DECODE (`sbs=4`, `psz=512`, `bq_sz=1`):** incumbent `bkv_p=2` = 2.06 MiB; area 2's
`bkv_p=16` = **16.28 MiB** (fits); `bkv_p=32` = 32.53 MiB (**over 31**, under 64).

**Pairs that cannot both be taken — TensorCore:**

| pair | combined | verdict |
|---|---|---|
| area 2 decode `bkv_p=16` (16.28 MiB) **+** area 1's C2 score-resident-in-VMEM at B-D4 (16 MiB) | **32.28 MiB** | **over the 31 MiB scoped budget.** One or the other, or raise `vmem_limit_bytes` deliberately and own the risk area 2 names. |
| area 2 `bq_sz=1024, bkv_p=8` (41.75 MiB) **+** anything | 41.75 MiB | over 31 on its own |
| area 3's deeper (triple) buffering **+** area 2's `bq_sz=1024, bkv_p=2` | 16.75 × 1.5 = **25.1 MiB** | fits, but leaves 5.9 MiB and excludes C2 |
| area 3's triple buffering **+** `bq_sz=1024, bkv_p=4` | 25.25 × 1.5 = **37.9 MiB** | **over 31** |
| area 6's E3 block-max plane on the TC side (`f32[sbs·bq_sz, n/128]`; at B MIXED `bq_sz=1024`: 1024×8192×4 = 32 MiB) **+** any `bq_sz=1024` retile | ≥48.75 MiB | **over 31, and E3's plane alone nearly fills 64.** E3's TC-side buffer must be tiled per KV block (`bkv_sz/128` columns, 4 KiB at `bkv_p=2`), not materialized whole — this is a real design constraint E3 does not state. |

## b2. SparseCore VMEM

**Judged against the 480 KiB per-subcore ceiling the compiler actually accepts** (area 5; area 5
also reports `p=8` refused at 512 KiB, so 512 is not available).

Scratch, from `sparsecore_topk.py:330-344`, per subcore:
`resident_vmem = slice_len·4`; `hist_vmem = (16·256+16)·4 = 16.06 KiB`;
`bucket_vmem = (256+16)·4 = 1.06 KiB`; `out_vmem = (k+16)·4 = 8.06 KiB`; `len_vmem = b_pad·4`.

| configuration | `slice_len` | resident | `len_vmem` | **total** | vs 480 KiB |
|---|---|---|---|---|---|
| A-D64, `p=1` | 10240 | 40.0 KiB | 0.25 KiB | **65.4 KiB** | 414.6 free |
| B-D*, stage 1 | 65536 | 256.0 KiB | 1.0 KiB | **282.2 KiB** | 197.8 free |
| **B-P1024, stage 1** | 65536 | 256.0 KiB | **64.0 KiB** | **345.2 KiB** | **134.8 free** |
| B-P1024, stage 2 | 32768 | 128.0 KiB | 64.0 KiB | **217.2 KiB** | 262.8 free |

**Correction to area 5:** its 62 KiB figure for A-D64 used `n = 9216`; the correct `n` is 10240
(a0.2) and the figure is **65.4 KiB**. Small, but it propagates into any headroom claim at (A).

**The binding row is B-P1024 stage 1 at 134.8 KiB free**, and `len_vmem` — sized `b_pad·4` with
`b_pad = align_to(1024, 16)` — is 64 KiB of it, replicated on every subcore though each subcore
reads at most `W` of those entries. Area 7's sizing fix (index `len_vmem` by the subcore's own rows)
would free about 62 KiB there. That fix is a prerequisite for the pair below.

| proposal | SC VMEM cost | fits at B-P1024 stage 1? |
|---|---|---|
| area 6 **E6** (L0 widened to 11 bits, 2048 buckets: `16·2048·4` = 128 KiB, replacing 16.06 KiB) | **+119 KiB** | **yes — 464.2 of 480 KiB.** Area 6 says E6 forces `MAX_SLICE_WORDS` down and `p` up; **at the current `MAX_SLICE_WORDS` it does not. Correction to area 6.** |
| area 6 **E3** block-max plane resident on SC (`n/128` f32 = 8192 words = 32 KiB) | +32 KiB | yes alone (377.2 KiB) |
| **E6 + E3 together** | **+151 KiB** | **no — 496.2 KiB, over 480.** |
| E6 + E3 + area 7's `len_vmem` sizing fix (−62 KiB) | +89 KiB net | **yes — 434.2 KiB** |

**The named conflicting pair on SparseCore: E6 and E3 cannot both be taken at B-P1024 stage 1
unless area 7's `len_vmem` sizing fix is taken first.** All three figures `modelled`; the 480 KiB
ceiling is `measured` (area 5's compiler-acceptance probe).

Two further notes the designer needs:
- Any proposal that raises `MAX_SLICE_WORDS` above 65536 to lower `p` is bounded by
  `resident = slice_len·4` against the same 480 KiB. With the incumbent's other scratch at
  89.2 KiB (B-P1024) the ceiling is `slice_len ≈ 100,000` — **less than 2× the current value**, so
  "just make the slice bigger" cannot get `p` from 16 to 8 at (B) and be combined with E6.
- Area 5's warning that `subcore_barrier` hangs above 4 subcores is a correctness constraint on any
  proposal that wants cross-subcore cooperation (E2's dynamic slice bounds are the one at risk).
  It is not a byte cost and does not appear above, but it gates the same proposals.

## b3. SMEM

Measured ceiling at (B): **1021.5 KiB** (area 7). `page_indices` is
`i32[max_num_seqs · pages_per_seq] = 64 · 2048 · 4` = **512.5 KiB**, half of it.

| configuration | SMEM | verdict |
|---|---|---|
| incumbent at (B) | 512.5 KiB | 509 KiB free |
| **+ area 7's `int32` byte-offset table** (same extent, 512 KiB) | **1024.5 KiB** | **refused — confirms area 7's own finding** |
| `page_indices` narrowed to `uint16` (256.5 KiB) + the byte-offset table | 768.5 KiB | fits, **gated on `total_num_pages ≤ 65535`** |
| `page_indices` `uint16` alone | 256.5 KiB | fits, 765 KiB free |
| incumbent at (A) | 2.25 KiB | not a constraint at (A) |

**The named conflicting pair on SMEM: area 7's byte-offset table and the incumbent `int32`
`page_indices` want the same bytes at (B), and only one can have them.** The `uint16` narrowing is
the only route that admits both, and it introduces a legality condition
(`total_num_pages ≤ 65535`) that must be checked in `bind()` and must raise `NotImplementedError`
per `code-style.md`'s checklist, not silently truncate.

Note that arm 5 of probe `P-ITER` (a1) is exactly the byte-offset table, so `P-ITER` cannot be run
at (B) without the `uint16` narrowing — run that arm at (A), where SMEM is 2.25 KiB, or narrow
first.

## b4. What fits, in one place

- **Everything on the SparseCore fits at (A)** with 414.6 KiB spare. The SC budget only binds at
  (B), and only at prefill.
- **At (B) prefill stage 1, the free 134.8 KiB admits E6 *or* E3, not both.**
- **On the TensorCore the 31 MiB scoped budget admits any one of the retiles**, up to and including
  `bq_sz=1024, bkv_p=4` at 25.25 MiB, but **not a retile plus a score-resident scheme, and not a
  retile plus triple buffering above `bq_sz=1024, bkv_p=2`.**
- **At (B), SMEM admits the byte-offset table only if `page_indices` narrows to `uint16`.**
- Nothing in any sheet exceeds the 64 MiB physical VMEM. Every conflict above is against the 31 MiB
  scoped budget or the 480 KiB SC ceiling.

---

# (c) Combine — every seam in `seams.md`

Every seam the reference names gets a concrete instance in this block or the sentence "No instance,
because —" with the reason. Candidates generated here enter the designer's option table on equal
terms with the single-area ones.

## S1 — layout × algebra

**Inversion: is there an algebraic form that makes the awkward layout natural?**

**Instance: the 256-byte packed cache row of which 129 bytes are used.** `load_bkv` reads
`LROW_ACTUAL = 256` B per KV token to consume `LROW_USEFUL = 129` (128 fp8 values + one e8m0 scale
byte). KV is 94.8% of the (A) modelled ledger and 15.7% of B-D4-256k's, so the padding is the single
largest byte term in the block. Area 4 proposes a strided 128-of-256 read; area 1 proposes the same
under C1.

The algebraic form that makes it natural: **the scale is per-row and multiplies out of the dot
product**. `scale_val = bitcast(flat_bkv[:, head_dim:head_dim+1].T, float8_e8m0fnu).astype(bf16)`
— an e8m0 scale is a pure exponent, so `sum_h w_h · relu(q_h · (s·k_h))` equals
`s · sum_h w_h · relu(q_h · k_h)` only when `s > 0`, which e8m0 guarantees. **The scale can
therefore be hoisted entirely out of the inner loop and applied to the finished score**, which
means the scale byte need not travel with the value bytes at all. That converts "read 129 of 256
bytes with an awkward stride" into "read 128 contiguous bytes, and read the scales as a separate
dense `[n]` vector" — a natural layout for both.

`Δ time`: at (A) decode, KV → 128/256 gives `roofline.py`'s Lever 1 = **−41% of modelled time** and,
against the 25.9× meas/model gap, an `unmodelled` share of measured time. Cost: the cache writer's
layout must change, which is outside this kernel — see S4.

**Second S1 instance: the `[T, n/128]` block-max plane E3 needs.** Its natural producer layout
(one f32 per 128 KV columns, written by the scoring kernel while the tile is in registers) and its
natural consumer layout (SC reading `n/128` contiguous words per row) agree, which is why E3 is
cheap in bytes (+0.8% of the ledger). Worked in full at c-iii below.

## S2 — compute units × algebra

**Inversion: which unit is idle, and what algebra could it be running?**

**Instance, and it is the largest one in the block: the SparseCore is 61–72% of measured time at
every (A) shape** `measured (j-31ca6c31)`, and the TensorCore is idle for that whole span unless the
next layer's scoring is already in flight. Area 5's §3 overlap table is the sheet that owns this.

Three algebra-changing candidates the seam generates, none of which appears on a single-area sheet
in this form:

1. **Split rows between the two units.** The selection is per-row and embarrassingly parallel over
   `T`. At (A), where `p=1` and the SC does one wave of 32 or 64 rows, giving a fraction of the
   rows to a TC `lax.top_k` over the resident score tile balances the two. At A-D64-9k the SC is
   183.6 µs against 181.5 µs of scoring; a perfectly balanced split predicts
   `max(183.6·(1−x), 181.5 + c·x)` — with `c` `unmodelled` (a TC top-k over 10240 f32 for `x·64`
   rows has never been measured on this stack). **The deciding quantity is `c`, tagged `unmodelled`,
   never a number**, per `seams.md`'s ranking rule.
2. **Run the SC selection for layer `L` concurrently with the TC scoring for layer `L+1`.** This is
   pipelining across the seam rather than within it, and it needs no algebra change at all — only
   that the 21 indexer layers not be serialized through a barrier. Whether they currently are is
   `unmodelled`; it is a profile question, not an arithmetic one.
3. **Move selection wholly to the TensorCore at (A) and keep the SparseCore for (B).** Area 6's E9
   (rate-dependent dispatch) is the wrapper this needs; area 6 explicitly declines to claim the
   placement half and flags it as a seam. This is that seam, and it is the one place where area 6's
   "(A) and (B) do not want the same algorithm" and area 5's placement analysis compose into a
   single candidate.

## S3 — pipelining × algebra

**Inversion: is there a form of the algorithm with a shorter or more overlappable dependence
chain?**

**Instance: the six sequential passes over the resident slice, and the four sequential refinement
levels inside them.** `sparsecore_topk.py:158-291` is a strict chain — `cvt` → `hist0` → L1 → L2 →
L3 → `emit` — where each level's shift and mask depend on the previous level's chosen bucket. The
chain length, not the byte count, is what `f = 1.2237 ns/word` is measuring (a0.3), and it is the
dominant term in nine of eleven rows.

The seam's question is whether a form exists with a shorter chain. Two, both already on area 6's
sheet as pass-count items but neither previously connected to pipelining or priced:

- **E5 + E7 shorten the chain from 6 to 4** by fusing `cvt` into `hist0` and `emit` into the last
  refinement. `Δ time −1.6%` at A-D32-8k, **−15.8%** at B-P1024-256k, `modelled` under M1 at
  0.204 ns/word/pass.
- **E6 shortens it from 4 refinement levels to 3** by widening L0 to 11 bits.
  `Δ time −0.8%` at A-D32-8k, **−13.9%** at B-P1024-256k, `modelled`. Its VMEM cost is +119 KiB
  and it **fits** — b2, correcting area 6.
- All three together: `Δ time −2.4%` at A-D32-8k, **−41.6%** at B-P1024-256k, `modelled`.

**A second, independent S3 instance:** `_fetch_bkv`'s double buffer is depth 2 and the in-body step
is fixed-cost-dominated (a1: ~524 ns fixed against ~75 ns of bytes at (B) prefill). A deeper buffer
does not help a chain whose stall is issue overhead rather than latency — **so area 3's deeper
buffering proposal and the a1 finding are in tension, and the a1 finding is the better-evidenced of
the two.** Recorded as a tension, not resolved: `P-ITER` arm 2-vs-3 settles it, because if raising
`bkv_p` recovers the fixed cost then depth was never the constraint.

## S4 — precomputation × layout

**Inversion: could a quantity be computed at a cheaper level if it were laid out differently?**

**Instance, and it corrects a misattribution in area 7:** area 7's handoff describes a "129→256
head-dim zero-pad in `prepare_q_inputs`". Reading `prepare_q_inputs` and `prepare_index_weights`,
**both pads are no-ops at this model's shape** (`H=32`, `DH=128`, fp8 packing 4 — the head dim is
already 128 and the pad computes to zero). The 129-of-256 waste is real but it lives in
**`cache_kv`'s layout**, produced by the cache writer, not in `prepare_q_inputs`. Correction owed;
the settling observation is that the pad expressions evaluate to zero at `H=32, DH=128`.

That relocation is what makes it an S4 instance rather than an area-7 one: the quantity (the
packed KV row) is *already* precomputed at the cheapest possible level — once, by the cache
writer, at write time — and the only thing wrong with it is its layout. So the lever is not "move
this computation earlier" but "**change the layout at the level where it is already computed**":
have `cache_kv` write values and scales into two planes rather than one padded row. That is a
change to a different component with a different owner, and it is the reason the S1 instance above
cannot be taken inside this kernel alone.

**Second S4 instance:** the page byte-offset table (area 7). The offsets are a pure function of
`page_indices` and `page_size`, recomputed on the scalar core every iteration — `sbs·bkv_p` = 8
times per step at (B) decode. Precomputing them on the host is free in time and costs 512 KiB of
SMEM, which does not fit alongside `int32` `page_indices` (b3). **The layout change that makes the
precomputation affordable is narrowing `page_indices` to `uint16`** — precomputation and layout are
the same decision here, which is exactly what S4 names.

## S5 — partitioning × fusion boundary

**Inversion: does the fusion boundary have to be where the partitioning boundary is?**

**Instance: `_pick_partition` chooses `p` from static `n`, never from `eff_len`** (`:360-377`,
confirmed at a0.1) — the partitioning is fixed at trace time by a *padding* quantity, while the
fusion boundary (the HBM score matrix between the two kernels) is where the length information
would have to cross. The two are coupled: `p` cannot depend on `eff_len` unless `eff_len` is
available where `p` is chosen, and today it is not, because the boundary is a materialized matrix
with no metadata alongside it.

This is area 6's E2 and area 2's idle-fraction finding meeting area 1's fusion-boundary analysis,
and it is worked in full at c-i below because it is also the most valuable seam in the sheet.

**Second S5 instance:** the boundary itself is what forces the `-inf` fill to exist (a2) — a fused
score+select would have no dead columns to fill because there would be no matrix. Area 6's gap 2
warns that the fused form lost 44% at S=8192 on the one hand-written flash kernel measured, so this
is a candidate with a known adverse precedent, not a free win.

## Seams with no instance

- **`seams.md`'s bf16 relayout gap.** No instance, because every array crossing a layout boundary
  in this block is f32 (the score matrix, the block maxima, the radix keys) or fp8 (the KV values),
  and the reference records the relayout cost as closed for f32 `[j-054e7ac4]`. The one bf16
  quantity is `scale_val`, which is a `[1, n]` vector and never relayouts. If S1's scale-hoist
  candidate is taken, the hoisted scale becomes a bf16 multiply against an f32 score — that would
  create the first bf16 boundary in the block, and the gap would then be live. **Recorded as a
  conditional instance: it does not exist today and S1 creates it.**

## c-i. Area 5's cost model × area 6's E2/E3 × the `_pick_partition` finding

The brief's question: if SC time is `C + f·X_pad`, it is set entirely by `cdiv(b·p, 32) · n_p`,
which is a **padding** quantity with no live-data term. Which proposals change `W`, which change
`n_p`, which change neither?

| proposal | changes `W`? | changes `n_p` / `slice_len`? | changes walked words? | changes SC bytes read? |
|---|---|---|---|---|
| E1 (DMA bounded by `eff_len`) | no | no | no | yes — but only within partially-live slices |
| **E2 (length-aware `p`)** | **yes** | no | no | no |
| E3 (block-max prune, resident slice unchanged) | no | no | **yes** | no |
| E3 + compacted/gathered resident slice | **yes** | **yes** | yes | **yes** |
| E5 / E6 / E7 (pass fusion) | no | no | no — but `f` falls | no |
| suppress the growth branch at `b=16` (a0.4) | no — `W`=1 either way | **yes** | yes | yes |
| `bkv_p ∈ {1,3,9}` at (A) (a0.2b) | no | **yes** (10240→9216) | yes | yes |
| stream the slice instead of residing it (area 5) | **yes** | **yes** | no | yes — re-read per pass |
| area 1's C2 (scores never reach HBM) | removes the SC input entirely | | | |
| move select to the TC (S2) | n/a | | | |
| E10 (sort by length) | no | no | no | no — TC-side only |
| delete the `-inf` fill | no | no | no | no |

**E2, priced per shape.** Choosing `p` and the slice bounds from `eff_len` rather than static `n`,
so that `p` is the smallest power of two with `cdiv(eff_len, p) ≤ 65536`:

| row | `p` now | `X_pad` now | `p` under E2 | `X_pad` under E2 | Δ SC µs | **Δ time %** of full op |
|---|---|---|---|---|---|---|
| A-D64-9k | 1 | 20480 | 1 | 18432 | −2.5 | −0.7% |
| A-D32-8k | 1 | 10240 | 1 | 8192 | −2.5 | −0.9% |
| A-D16-8k | 2 | 9216 | 2 | 8192 | −1.3 | −0.4% |
| A-P1024-8k | 1 | 327680 | 1 | 262144 | −80.2 | −9.5% |
| A-P512-8k | 1 | 163840 | 1 | 131072 | −40.1 | −6.9% |
| **B-D4-256k** | 16 | 557056 | **4** | 139264 | −511.3 | **−45.7%** |
| B-D4-512k | 16 | 557056 | 8 | 278528 | −340.8 | −24.4% |
| B-D4-1M | 16 | 557056 | 16 | 557056 | 0 | 0% |
| B-D1-1M | 16 | 557056 | 16 | 557056 | 0 | 0% |
| B-D16-256k | 16 | 557056 | 4 | 139264 | −511.3 | −25.0% |
| **B-P1024-256k** | 16 | 34603008 | **4** | 8650752 | −31758 | **−62.4%** |
| B-P1024-1M | 16 | 34603008 | 16 | 34603008 | 0 | 0% |

All `modelled` under M1. **E2 is worth nothing at full context length and a factor of two at
quarter context** — which is the opposite profile from E3, and the two are independent.

**E3 under M1 is `unmodelled`, and that is the finding.** E3 removes ~99% of the *walked* columns
at (B) but changes neither `W` nor `n_p`, so M1 predicts **zero**. Either M1's `f` is not purely
per-resident-word (in which case a walked-word term exists that the eleven fitted rows could not
see, because none of them varies walked words at fixed `X_pad`), or E3 is worth nothing on the SC
walk. **The deciding quantity is tagged `unmodelled`, never a number**, and it is the single
highest-value thing the wave-3 prober can settle.

**But E3 does have a modelled value at B-P1024, through the bytes term of a0.5.** Composed with a
gathered DMA that fetches only the surviving 128-column blocks, E3 collapses the 4 GiB stage-1
aggregate read at B-P1024-1M to ~0.13 GiB → `Δ time` **−18.8%** `modelled`. **Neither E3 nor E1
alone gets this** — E1 alone does nothing at B-P1024-1M because every slice is fully live, and E3
alone leaves the DMA at full width. **`E1 × E3` composed is a candidate that neither sheet
proposes**, and its risk is that the gathered DMA issues 128-column (512 B) descriptors instead of
one contiguous 256 KiB copy, which may cost more in descriptor issue than it saves in bytes —
`unmodelled`.

**Composition at B-P1024-256k**, the row where the most levers land at once:
E2 (−62.4%) and E5/E6/E7 (−41.6% of the SC term) act on the same `f·X_pad` product and **do not
add** — E2 cuts `X_pad` 4× and the pass fusion cuts `f` by half, giving jointly
`165.4 + 0.612·8650752` ≈ 5.5 ms against 45.7 ms, `Δ time` **−79%** of the SC term, but the
3.2 ms bytes term survives both and the TC's ~8 ms does too. The full-op figure is `Δ time −67%`
`modelled` — and it is a product, not a sum, which is exactly the arithmetic a per-variant
measurement would get wrong.

## c-ii. Area 2's `bq_sz=1024` MIXED retile × area 1's "KV is re-read 8×"

**These are the same defect seen from two areas, and they agree on the fix.** Area 1 observes that
at (B) prefill the KV block is re-read once per query tile; area 2 observes that with `bq_sz=128`
and 1024 query tokens there are `nbq = 8` query tiles. Same 8. The fix both imply is to tile the
full 1024 queries at once so `nbq = 1`.

**Where they do not agree is the cost, and area 2's figure is wrong** (b1): `bq_sz=512, bkv_p=4` is
13.13 MiB, not 21 MiB. With that corrected, the option table is:

| configuration | VMEM | iterations at B-P1024-256k | KV passes | modelled KV bytes at 256k |
|---|---|---|---|---|
| incumbent (128, 2) | 2.53 MiB | 2048 | 8 | 512 MiB |
| area 2's (1024, 2) | 16.75 MiB | 256 | 1 | 64 MiB |
| **(1024, 4)** | **25.25 MiB** | **128** | **1** | **64 MiB** |
| (512, 8) | 22.13 MiB | 128 | 1 | 64 MiB |

`Δ time` from KV bytes alone at B-P1024-256k: 448 MiB saved at 3.70 TB/s = 127 µs = **−0.25%**,
`modelled` — negligible, because at (B) prefill KV is only 7.4% of the ledger. **The retile's real
value is the iteration count**, at ~524–582 ns of fixed cost each (a1): 2048 → 128 iterations saves
1920 × 0.587 µs = **1127 µs = `Δ time` −2.2%** at B-P1024-256k, `modelled`.

**So area 1's byte argument and area 2's iteration argument point the same way but are worth very
different amounts, and the iteration argument is 9× the larger.** A designer ranking this on area
1's ledger alone would price it at a quarter of a percent and drop it.

**Budget interaction, from b1:** `(1024, 4)` at 25.25 MiB fits the 31 MiB scoped budget but leaves
5.75 MiB, which excludes area 3's triple buffering (37.9 MiB combined) and excludes any score-
resident scheme. **Named pair that cannot both be taken.**

## c-iii. Area 6's E3 × area 4's layout constraints × area 5's SC placement

E3 needs an `f32[T, n/128]` block-max plane written by the scoring kernel and read by the selector.
Composing the three areas' constraints on that one array:

**Producer side (area 4's layout rules).** The maxima are a VPU reduce over registers the scoring
kernel already holds, so the FLOPs are `unmodelled`-negligible. But the natural output tile is
`[sbs·bq_sz, bkv_sz/128]` — at B MIXED with `bq_sz=1024, bkv_p=2` that is `[1024, 8]`, an
**8-lane-wide** f32 tile against a 128-lane register file. Area 4's own rule against sub-native
tiles applies: writing 8 of 128 lanes wastes 94% of each store. **The fix that makes it natural is
to accumulate the maxima across KV blocks in a `[sbs·bq_sz, 128]` VMEM buffer and write once at
`bkv` block 16** — 0.5 MiB at `bq_sz=1024`, which fits alongside the 25.25 MiB retile (25.75 MiB
total, inside 31 MiB). Without that accumulation the plane costs 128 tiny writes per query tile.

**Traffic (area 1's ledger).** `n/128` f32 per row = 8192 words = 32 KiB per row at (B). At
B-P1024-1M, 1024 rows = 32 MiB written and 32 MiB read, against a 14604 MiB ledger = **+0.4%**,
`modelled`. Area 6's "+0.8%" was on the pre-correction ledger; 0.4% is the corrected figure.

**Consumer side (area 5's SC placement).** The plane must be resident on the SC subcore that owns
the row: +32 KiB per subcore, which fits at B-P1024 stage 1 (377.2 of 480 KiB) **but not alongside
E6** (b2). And the selection over the plane is itself a top-m — a second, smaller instance of the
same primitive, over 8192 words instead of 65536. Under M1 that inner select costs
`f · W · 8192` = 3.2 ms at B-P1024 against the 42.5 ms it is trying to avoid: **E3's own overhead
is 7.5% of what it saves**, `modelled`, which is the first number anyone has put on it.

**Correctness (area 6's own claim, checked).** E3 is exact: `τ_lb` = the k-th largest within the
union of the top-m blocks is a lower bound on the true `τ*` because a subset's k-th largest cannot
exceed the whole set's, so every block with `max > τ_lb` is retained and no true top-k member is
in a discarded block. The argument holds. **The condition area 6 does not state is that `m` must
satisfy `m·128 ≥ k`** — at (B), `k = 2048` needs `m ≥ 16` blocks of 8192, i.e. 0.2%, which is
exactly the (B) selection rate area 6 cites. **At (A) it needs `m ≥ 16` of 80 blocks = 20%
minimum**, against a 22–25% measured selection rate — so at (A) E3 prunes essentially nothing even
before the distributional argument, which is a cleaner version of area 6's own "(A)/(B) split"
finding and reaches it by arithmetic rather than by a `0.75^128` estimate area 6 itself flags as
unverified.

**Composed candidate: E3 + E1 + the accumulation buffer + area 7's `len_vmem` fix**, which is what
it takes to have E3 and E6 coexist. VMEM: TC 25.75 MiB (fits 31), SC 434.2 KiB (fits 480).
`Δ time`: −18.8% at B-P1024-1M from the bytes term, plus whatever the walked-word term is worth —
`unmodelled`.

---

# Handoff

**Corrections made to sheets on disk** (each stated above with its settling line, so that silence
is not read as agreement): area 5's `p` formula and its 62 KiB SC figure; area 6's "reads
contradictory" on `_pick_partition`, its "E6 forces `MAX_SLICE_WORDS` down", and its +0.8% for E3;
area 2's `make_async_copy` counts (4× low), its `bq_sz=512, bkv_p=4` VMEM figure, and its "5
launches at (A)"; area 7's `prepare_q_inputs` attribution and its ~6% scalar estimate; area 4's and
area 7's `n = 10240` upheld against `roofline.py`, `shapes.py` and `report.md`; the brief's own
"4 GiB vs 1 GiB" on the `-inf` fill and its "9 rows with an `sc` number" (there are 11);
`references/08-sparsecore.md` on `store_compressed`; and `roofline.py:86-95`'s conclusion that the
A-D16 inversion "is NOT established".

**Probes for wave 3.** None was run here; no hardware job was submitted.

| probe | question | what each outcome decides |
|---|---|---|
| **P-WALK** | Does SC time depend on **walked** words at fixed `X_pad`? One job, B-D4-256k with `eff_len` forced to 1/4 and to full, `p` held at 16. | The single highest-value unknown. If walked words matter, E3 and E8 have a modelled value at (B) and M1 needs a second term. If they do not, **E3 is worth nothing on the SC walk** and its only value is the bytes term at B-P1024. |
| **P-ITER** | Is the in-body fixed cost per iteration, per `make_async_copy`, or scalar index arithmetic? Five arms at B-D1-1M (a1). | Arms 2-vs-3 confirm or refute area 3's 512 ns transfer; arm 4 isolates area 2's per-copy hypothesis; arm 5 isolates area 7's. Also settles whether area 3's deeper buffering has a target at all. |
| **P-DECOMP** | Where does the 16 ms B-P1024-1M/256k gap live? `--decompose` on **both** B-P1024 rows **in one job**. | If SC is equal in both, M1's padding-only form survives and the gap is TC-side — contradicting a0.5's 82% closure. If SC differs by ~9.5 ms, the SC bytes term at 337 GB/s is real and **M1 must never be extrapolated to B-P1024 without it**. Also removes the cross-job caveat on the gap. |
| **P-P16** | Is the `_pick_partition` growth branch a net loss at `b=16`? A-D16-8k and A-D32-8k **in one job, adjacent**, plus an A-D16-8k arm with the growth branch suppressed. | Confirms or refutes the −9.0% at A-D16-8k and settles `roofline.py:86-95`. Cheapest probe on the list and the only one that closes a measurement anomaly. |
| **P-SCBW** | What is the SparseCore's achieved read bandwidth from HBM at a 256 KiB slice? | Supplies the constant a0.5 assumes at 337 GB/s. If it is materially higher, the 82% closure fails and the gap is `unmodelled`. |
| **P-GATE** | Does the accuracy gate discriminate? Area 6's gap-1 negative control: deliberately drop one KV block and confirm `churn` red-lights. | Under *untested*, A1 cannot be judged and must not be measured for speed; under *tested*, it can. Only relevant if any approximation reaches the table. |
