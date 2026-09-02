# P6 — Is a narrow strided HBM read charged for the bytes it asks for, or for the tile it lands in?

**Question.** The dominant nightly decode HBM term is `_fetch_bkv`'s DMA of the indexer K cache
(`streamindex_topk.py:160-198`), which copies whole
`[page_size_per_kv_packing, kv_packing, lkv_dim]` pages out of a `u8[..., 4, 256]` array. Only
**129 of every 256 minor bytes** are consumed (128 fp8 key bytes + 1 e8m0 scale byte); the other
127 are padding from `align_to(129, 128) == 256`. If a strided read that takes only the useful
prefix is charged for the prefix, narrowing the DMA is a movement-only change worth ~half the
dominant decode term. If it is charged for the whole 256-byte stride, the change is worth exactly
nothing and the padding can only be reclaimed by repacking the cache.

Nothing in the source settles it — it is a property of the DMA engine's granule — so it was
measured.

**Script:** `probes/probe_hbm_stride_charge.py`, committed.
**Job:** **`j-170ed342-8cc7-4719-82b5`**, `JobStatus: Succeeded`, `state: Complete`, image
`…vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`, device `TPU7x`, jax
`0.10.2`. **Run alone on the whole 4-chip pod, not fanned out** — a bandwidth measurement must be
uncontended, so nothing else shared the pod.

`RUN_CMD` as read back from `cdk job recipe`:
```
python /root/kw/probes/probe_hbm_stride_charge.py
```

---

## VERDICT

**Charged for the bytes it asks for.** A read of **128 of every 256 bytes costs 0.510× the
full-width read** against a byte ratio of 0.500 — a charge ratio of **1.02**, i.e. the 128-byte
stride is essentially free. Not ≈129/256 in the "charged for the whole stride" sense, and not ≈1.0
in the "pay for the tile" sense: **you pay for bytes.**

**But the 129-byte read the kernel would actually want does not exist.** Mosaic refuses to lower
*any* DMA slice whose width is not a multiple of 128:

```
MosaicError('INTERNAL: Mosaic failed to compile TPU kernel ... Slice sizes along tiled
dimensions must be aligned to tiles. Failed to verify that the size at dimension 1 is
divisible by the tile dimension 128. ...')
```

w = 32, 64, 96, **129**, 160, 192 and 224 were all rejected at compile time. Only 128 and 256 ran.
So `P6_VERDICT: INCONCLUSIVE at w=129` as a *number*, and the reason is more useful than the number
would have been.

**Design decision settled:** the "narrow the DMA to 129 bytes" optimization is **not expressible in
Pallas and cannot be written**, at any charge ratio. The only expressible narrowing is to 128,
which drops the e8m0 scale byte and therefore changes semantics. The saving is real and is
essentially fully realizable — **but only via a repack**: a 128-wide fp8 key array plus a separate
scale array, which is exactly the nope/rope split MLA already uses and which **P5 established the
indexer is free to adopt unilaterally**.

**Design decision NOT settled, and it caps the value:** P7a found stage A is DMA-*block*-bound
rather than bandwidth-bound at the current blocking, so halving the bytes does not halve the time.
See [§5](#5-what-this-is-worth-read-with-p7a).

---

## 1. What ran, and how it was made trustworthy

Two arms, both holding the **DMA count and the row count fixed** and varying only how many bytes of
each row are requested — same descriptors, same rows, same grid, so any time difference is
bytes/granules moved and nothing else. `NBUF = 4` software pipeline. No consumer of the fetched
bytes (Pallas DMAs are effectful and are not DCE'd).

- **arm "minor"** — source `u8[2097152, 256]` (536.87 MB), copy `x[rows, 0:w]` for
  w ∈ {32, 64, 96, 128, 129, 160, 192, 224, 256}. `rows_per_copy = 16384` (4 MiB per DMA at full
  width), `ngrid = 128`. This is the direct granule-locating experiment.
- **arm "major"** — source `u8[R, S, 128]`, copy `x[rows, 0:t, :]`, every slice 128-lane aligned.
  Intended as the robust fallback. It did not survive; see §4.

Three gates, all of which had to pass before any ratio is quotable:

| gate | why | result |
| --- | --- | --- |
| **physical-layout check** — is the source array charged more HBM than its logical size? | if the minor dim were padded by layout assignment, the stride is not what the arm assumes | ✅ `logical=536.87MB actual=537.01MB ratio=1.000` (minor arm); `536.87 / 536.96, ratio 1.000` for both major sources |
| **elision check** — is any arm reporting > 3.70e12 B/s useful? | a DMA that got optimized away would look infinitely fast | ✅ max observed 3.347e12 B/s, under spec |
| **calibration** — does the full-width point approach the achieved-stream reference? | if not, the measurement is latency-bound and the ratios are meaningless | ✅ see §2 |

## 2. Calibration — the measurement is bandwidth-bound

`measured (j-170ed342)`:

```
[P6] calibration: full-width achieved 3.347e+12 B/s = 112.7% of the 2.97e12
     achieved-stream reference, 90.5% of the 3.70e12 spec
```

**3.347e12 B/s** for a contiguous full-width read. Above the 2.97e12 B/s reference and below the
3.70e12 B/s spec — which is the right place for it to land: 2.97e12 is a **read-write** stream
figure and this is a pure read, so exceeding it is expected, while nothing may exceed spec. The
probe is measuring HBM bandwidth, not DMA issue latency, so the ratios below mean what they say.

## 3. The result

`measured (j-170ed342)`, source `u8[2097152, 256]`, `rows_per_copy=16384`, `ngrid=128`, `NBUF=4`:

| width w | lowered? | time | useful bytes | **B/s (useful)** | B/s if charged full 256 B stride |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 32 | ❌ Mosaic | — | — | — | — |
| 64 | ❌ Mosaic | — | — | — | — |
| 96 | ❌ Mosaic | — | — | — | — |
| **128** | ✅ | **81.84 µs** | 268.44 MB | **3.280e12** | 6.560e12 |
| **129** ← *the width of interest* | ❌ **Mosaic** | — | — | — | — |
| 160 | ❌ Mosaic | — | — | — | — |
| 192 | ❌ Mosaic | — | — | — | — |
| 224 | ❌ Mosaic | — | — | — | — |
| **256** *(control)* | ✅ | **160.39 µs** | 536.87 MB | **3.347e12** | 3.347e12 |

```
[P6] cost ratios: time(w) / time(256); a value of 1.0 means the narrow read costs the same
     as the full-width one
[P6]   w=128: ratio=0.510  (bytes ratio=0.500)
[P6]   w=256: ratio=1.000  (bytes ratio=1.000)  step_from_128=+0.490
```

### 3.1 The charge ratio

`modelled` (arithmetic on the two measured times):

- time ratio `81.84 / 160.39 = 0.5103`
- byte ratio `128 / 256 = 0.5000`
- **charge ratio = 0.5103 / 0.5000 = 1.021**

A charge ratio of **1.02** means the narrow read pays a 2 % overhead over a perfectly
byte-proportional read. For comparison, "charged for the tile it lands in" would give a time ratio
of 1.000 and a charge ratio of 2.00. The measured value is unambiguously the byte-proportional one.

Equivalently: the strided read sustains **3.280e12 B/s of useful bytes, 98.0 % of the 3.347e12 B/s
a contiguous read achieves.** Skipping every other 128-byte half-row costs 2 % of bandwidth.

### 3.2 The implied granule, and its honest bound

**The charge granule is ≤ 128 bytes**, and this experiment **cannot bound it below that**. Every
probe that would have located it more precisely — w = 32, 64, 96 — was rejected by Mosaic before
reaching hardware, and the rejection is a *tile-alignment* rule, not a hardware limit:

> `Slice sizes along tiled dimensions must be aligned to tiles. Failed to verify that the size at
> dimension 1 is divisible by the tile dimension 128.`

So what actually binds first is **Mosaic's 128-lane tile alignment, not the DMA engine's granule.**
The hardware granule could be far smaller and would be invisible through Pallas. In particular, the
`dma_granule_size_bytes = 32` figure reported for SparseCore is **not** confirmed or refuted here —
the user's caution that the TensorCore granule may differ and must be measured is correct, and the
measurement is blocked at 128 B by the compiler, not by the hardware. `assumed`-grade at best; do
not carry 32 B into a TensorCore model on this evidence.

For the design question this does not matter: 128 is already fine-grained enough that the fp8 key
payload can be read without its padding.

### 3.3 Why "129" is the wrong thing to have asked for

129 is not a multiple of 128 and never will be. There is no Pallas DMA that reads 129 bytes of a
256-byte row. The reachable designs are:

| design | expressible? | bytes moved per position |
| --- | :---: | ---: |
| status quo — read all 256 | ✅ | 256 |
| narrow the existing slice to 129 | ❌ **Mosaic rejects** | — |
| narrow the existing slice to 128, drop the scale | ✅ but **semantically wrong** — loses e8m0 | 128 |
| **repack: 128-wide fp8 array + separate scale array** | ✅ | 128 + 1 (from a second, small array) |

Only the last is both expressible and correct, and §3.1 says it would realize ~98 % of the ideal
byte saving on the key read. **P5 already established that this repack touches nothing outside the
indexer path** — two stage-A readers and one `scatter_fusion` writer — so it does not require any
MLA-side or `sc_dsa_gather` coordination. The precedent is in the tree: MLA splits nope/rope for
exactly this reason rather than padding one array to 256.

## 4. The major arm failed to contribute, and one point of it is anomalous

Reported because it happened, not because it supports anything.

| arm | lowered? | time | B/s (useful) |
| --- | :---: | ---: | ---: |
| S=2, t=1 (the "128 of every 256" bracket) | ❌ Mosaic — *"size at dimension 1 is divisible by the tile dimension **4**"* | — | — |
| S=2, t=2 (full copy, 536.87 MB) | ✅ | **1548.69 µs** | **3.467e11** |
| S=4, t=1, t=2, t=3 | ❌ Mosaic (tile dimension 4) | — | — |
| S=4, t=4 (full copy, 536.87 MB) | ✅ | 160.24 µs | 3.350e12 |

Two things went wrong:

1. **Every partial-`t` slice was rejected**, for the same tile-alignment reason as the minor arm but
   now on a *tile dimension of 4* rather than 128 — the 3-D `u8[R, S, 128]` source picks up a
   `(4, …)` second-minor tile. The arm was designed on the assumption that 128-lane alignment was
   the only constraint; it is not. **The arm's entire purpose — bracketing the 129-byte question
   from below with a guaranteed-lowerable slice — was not achieved.** The design error is mine.
2. **S=2/t=2 is 9.7× slower than S=4/t=4 at identical total bytes, identical DMA count (128),
   identical bytes per DMA (4.19 MB) and identical buffer size (16 MiB).** The only difference is
   the DMA block's second-minor extent, 2 vs 4. 3.467e11 vs 3.350e12 B/s. **I did not chase this**
   and have no explanation for it. It is flagged as a real, reproducible DMA-shape sensitivity worth
   knowing about — a `_fetch_bkv`-style copy whose second-minor extent is small may be paying an
   order of magnitude — but it is an observation, not a finding, and nothing above depends on it.

The one thing the major arm did contribute is a consistency check: S=4/t=4's 3.350e12 B/s agrees
with the minor arm's full-width 3.347e12 B/s to 0.1 %, on a differently-shaped source. The
full-width calibration is not an artifact of one array layout.

## 5. What this is worth — read with P7a

P6 says the bytes are chargeable-by-byte. It does **not** say the time saving follows, and P7a says
it largely will not, at the current blocking.

`modelled`, using P6's own measured 3.347e12 B/s rather than the `assumed` rates in `report.md` §4:

| | HBM bytes per decode call (`report.md` §4) | implied floor at 3.347e12 B/s |
| --- | ---: | ---: |
| current, `lkv_dim = 256` | 167.8 MB | **50.1 µs** |
| after a repack to 128 + scales | 84.5 MB | **25.2 µs** |

Measured stage A at the best like-for-like blocking (`bkv_p = 3`) is **90.87 µs**
`measured (j-9e113359, shard 1)` — **1.81× above even the un-repacked 50.1 µs floor.** Stage A is
not sitting on the HBM roofline, so removing bytes from a term that is not the binding constraint
buys a fraction of the byte saving, not the whole of it. This is the same conclusion P7a reached
from the other direction: across bkv_p ∈ {1, 2, 3} the bytes moved are essentially constant
(9 / 10 / 9 pages) while stage A varies by 40 %, which cannot happen on a bandwidth-limited term. And
P7b's LLO dump agrees from a third direction: the decode program is 15.5 % fully idle with 15.5 %
scalar bubble and 4.61 % MXU occupancy (`static compile-time analysis (j-9e113359, shard 3)`) —
not the profile of a kernel pinned to the HBM roofline.

**Ordering, therefore:**

1. **`bkv_p = 2 → 3` first.** One tuple entry, −11.91 % on the call `measured (j-9e113359)`, no
   layout change, no correctness surface.
2. **Re-measure.** If stage A then approaches ~50 µs, it has become bandwidth-bound and the repack's
   headroom becomes real.
3. **Repack only then.** It is a change to the cache layout, its writer and both readers — far more
   expensive than step 1, and its value is contingent on step 2's outcome.

## 6. Scope limits

- **Everything here is a synthetic HBM read on `u8[2097152, 256]`, not the indexer kernel.** It
  establishes the charging *rule* and the achievable bandwidth. It does not measure `_fetch_bkv`.
- **The DMA granule was not located**, only bounded at ≤ 128 B, and the thing that blocked locating
  it is Mosaic's tile alignment rather than the hardware.
- **No f8/quantization semantics were exercised.** The repack in §3.3 is a *layout* conclusion; that
  a 128-wide fp8 array plus a separate scale array is numerically equivalent to the current packed
  form is asserted from the field definitions, not tested.
- **The 3.347e12 B/s figure is a read-only stream on one array shape.** It is not a general HBM
  number for this machine and should not be substituted for the 2.97e12 read-write reference where
  writes are involved.

---

## Provenance

| number | tag |
| --- | --- |
| 81.84 µs, 160.39 µs, 1548.69 µs, 160.24 µs; 3.280e12 / 3.347e12 / 3.467e11 / 3.350e12 B/s; all layout ratios | `measured (j-170ed342-8cc7-4719-82b5)`, `u8[2097152,256]` and `u8[R,S,128]` uint8, TPU7x, jax 0.10.2, uncontended whole-pod run |
| every Mosaic rejection quoted | `measured (j-170ed342)` — verbatim from the job log |
| charge ratio 1.021; 98.0 %; 50.1 µs and 25.2 µs floors; 1.81× | `modelled` — arithmetic shown inline on the measured values |
| 167.8 MB / 84.5 MB decode HBM traffic | `report.md` §4, `modelled` at the nightly decode shape |
| 2.97e12 B/s achieved-stream, 3.70e12 B/s spec | `references/hardware-facts.md` via `report.md` §4 |
| stage A 90.87 µs at `bkv_p=3` | `measured (j-9e113359-47c6-41b2-ad85, shard 1)` — **contended pod**, derived as total − stage B − init |
| `dma_granule_size_bytes = 32` (SparseCore) | `assumed` for TensorCore — **neither confirmed nor refuted here** |
