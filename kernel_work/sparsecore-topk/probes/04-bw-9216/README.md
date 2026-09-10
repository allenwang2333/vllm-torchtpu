# Probe 04 — `bw-9216`: how much of `dma_only` is transfer, and how much is launch?

**Question.** Probe 01's `dma_only` arm reads 22.1 µs at `b=16` — one wave, 40 KiB. At the stored
2138 GB/s that transfer should take about 20 **nano**seconds per subcore, so almost all of 22.1 µs
must be something else. Separate the two: what is fixed per *kernel launch*, and what is
proportional to *waves*? Only the second is hideable by a pipeline.

**Why it needed hardware.** It is the correction that overturns probe 01's printed gate — the single
most decision-changing number in step 1.

## Arms

`microbench/sctopk_2stage_bench.py --bw`, sweeping the wave count at a fixed 10240-word slice so the
intercept and the slope separate by regression rather than by subtraction. `--ctx 9216`, `k=2048`.
9 interleaved rounds, `us_median`.

## Config

Same job, image `fa63d6df` + mapped `b11b6eb6` `src/`, in-job md5 `44353b34…`. One tpu7x chip,
`SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=524288,
dma_granule_size_bytes=32)`.

## Job

`j-fa60ff2a-c793-4f32-9576` → `j-fa60ff2a-c793-4f32-9576.txt`, `bw_9216.json`.

## Answer

```
time(waves) = 21.349 µs  +  1.0447 µs/wave        slice 10240   max residual 0.76%
```

Out-of-sample: it predicts probe 01's three independent `dma_only` rows to within
**+1.6% / +2.9% / −1.0%** — rows that were not in the fit.

- **21.349 µs is fixed per launch** — program load plus SparseCore mesh entry. Paid once per kernel
  invocation, unaffected by wave count, and **not removable by any pipelining scheme**. At the
  shipping decode shape (`b=64`, 51.7 µs total) this is **41.6% of the kernel**, which is how it
  became candidate 2 in report §9.
- **1.0447 µs/wave is the transfer** — this, and only this, is what a double buffer can hide, and
  only `(waves−1)/waves` of it, since the first load has nothing to overlap with.

**Reconciling the two bandwidths.** 40 KiB / 1.0447 µs = **1255 GB/s** apparent, against the stored
**2138 GB/s** at the 130 KiB slice — and probe 02 confirms the stored number still reproduces. Both
are correct; neither is the bandwidth. Fitting the two points as `fixed + bytes/BW`:

```
≈ 0.62 µs fixed per wave   +   ≈ 3113 GB/s asymptotic
```

So the shipping 40 KiB slice spends **59% of its per-wave DMA time on per-wave fixed cost**, and a
quoted GB/s at this size is mostly measuring that overhead. (An earlier print of mine read "3 GB/s"
here — a units slip in the print statement; 3113 GB/s is the value.)

## Decided

- Probe 01's honest hideable DMA fractions (3.1% / 4.0% / 7.4%) are computed with **this slope**,
  not with `dma_only`.
- Report §6's ceiling table, and therefore the **−7.18% / −2.02%** cap on the caller's mechanism, is
  this number times `(waves−1)/waves`.
- The 0.62 µs per-wave fixed component is a **second** thing a cross-wave double buffer might hide
  and a streaming pipeline might not; step 2 should not assume the two are the same term.
