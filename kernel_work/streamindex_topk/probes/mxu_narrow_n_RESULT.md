# P4 — a narrow-N matmul costs ~3× per useful FLOP, and C6 dies on it

**probe:** At fixed M and K, how much more does the MXU cost **per useful FLOP** when the output has
N = 32 lanes instead of N = 128?

Probed as a **primitive**, not as the C6 variant. The 128×128 systolic array is fed 128 lanes wide;
whether an N = 32 output leaves three quarters of it idle is a property of the MXU, reusable by any
later narrow-N design, and it cannot be contaminated by whatever else C6 does.
`combined.md` already carries a `measured` **3.57× derate on the M axis** at M = 32. Nobody had
measured the N axis.

**config:** `vllm-torchtpu` @ `4dff8488e0b137d636eea9117892ce9fef76f583`, branch
`annyan/streamidx-scores`; working tree at submit `?? kernel_work/streamindex_topk/step2/combined.md`,
`?? optimization_work/` only. Image `…/vllm-torchtpu-annyan_google_com:13617694`. One tpu7x pod,
one shard. `probes/mxu_narrow_n.py`: `M = 128`, `N·R = 512` held constant, `GRID = 64`,
`WARMUP = 3`, `REPS = 20`, `BATCH = 8`, `vmem_limit_bytes = 128 MiB`.

**job:** `j-8ed8555c-1d46-44dd-8f6c`, `JobStatus: Succeeded`.

`RUN_CMD`, as read back from `cdk job recipe j-8ed8555c-1d46-44dd-8f6c --no-color`:

```
export PYTHONPATH=/workspace && python3
  /workspace/kernel_work/streamindex_topk/probes/mxu_narrow_n.py --dtypes
  bf16,fp8_e4m3 --ns 32,64,128,256 --ks 128,512
```

## The design, and what makes it a strike test

`R = 512 / N` repetitions per grid step, so down each dtype block at fixed K:

- **useful FLOPs** `= 2·M·N·K·R = 2·128·512·K` — **constant**
- **output elements** `= M·N·R = 128·512` — **constant**

Holding both pinned is the whole method. If narrow N were free every rung would take the same time,
so **flatness is the null hypothesis and any rise at N = 32 is the derate** — and because the output
element count is pinned it cannot be an artifact of storing more or fewer results.

Two K values per rung, because at K = 128 the vector store is a large fraction of the arm and
`time / FLOP` at a single K would measure MXU and store together and *understate* the derate. The
MXU is recovered as the slope, `mxu_ns = (t(K=512) − t(K=128)) / 3`; the store, the Python-unrolled
loop and the fixed per-grid-step cost are identical at both K and cancel out of the difference.

Confounds deliberately dodged: `b` is indexed on a **leading** axis (`b_ref[0, r]`) — slicing the
lane axis would inject exactly the relayout P3 measures; `b` is cycled through two distinct blocks
via `index_map=lambda i: (i % 2, 0, 0, 0)` so the body cannot be hoisted out of the grid; `a` is
reused across all R matmuls, keeping HBM traffic far under the roofline so nothing here is bandwidth
bound.

## Measured (job `j-8ed8555c-1d46-44dd-8f6c`)

`ns/step` is one grid step, at the script's printed precision; the per-arm JSON lines in the job log
carry full precision (e.g. bf16 N=32: K=128 `1149.5837952679722`, K=512 `1664.4542029098375`;
fp8 N=32: K=128 `1011.0350103786914`, K=512 `1315.7118701201398`).

| dtype | N | R | K | ns/step | spread % | GFLOP/step | Tops/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 32 | 16 | 128 | 1149.6 | 11.54 | 0.017 | 14.6 |
| bf16 | 32 | 16 | 512 | 1664.5 | 1.88 | 0.067 | 40.3 |
| bf16 | 64 | 8 | 128 | 1043.0 | 10.12 | 0.017 | 16.1 |
| bf16 | 64 | 8 | 512 | 1344.0 | 3.99 | 0.067 | 49.9 |
| bf16 | 128 | 4 | 128 | 915.3 | 15.71 | 0.017 | 18.3 |
| bf16 | 128 | 4 | 512 | 1062.9 | 8.09 | 0.067 | 63.1 |
| bf16 | 256 | 2 | 128 | 896.9 | 9.83 | 0.017 | 18.7 |
| bf16 | 256 | 2 | 512 | 1033.8 | 14.06 | 0.067 | 64.9 |
| fp8_e4m3 | 32 | 16 | 128 | 1011.0 | 8.32 | 0.017 | 16.6 |
| fp8_e4m3 | 32 | 16 | 512 | 1315.7 | 7.57 | 0.067 | 51.0 |
| fp8_e4m3 | 64 | 8 | 128 | 945.0 | 6.27 | 0.017 | 17.8 |
| fp8_e4m3 | 64 | 8 | 512 | 1122.5 | 9.33 | 0.067 | 59.8 |
| fp8_e4m3 | 128 | 4 | 128 | 845.0 | 11.17 | 0.017 | 19.9 |
| fp8_e4m3 | 128 | 4 | 512 | 947.9 | 14.35 | 0.067 | 70.8 |
| fp8_e4m3 | 256 | 2 | 128 | 857.8 | 16.56 | 0.017 | 19.6 |
| fp8_e4m3 | 256 | 2 | 512 | 927.1 | 10.90 | 0.067 | 72.4 |

### MXU cost isolated by the K slope

`Δ time %` is the K=128 arm against the same dtype's N=128 arm, `(new − old) / old`.
**Negative is faster.** `derate ×` > 1 means the narrow arm costs more per useful FLOP.

| dtype | N | t(K=128) ns | t(K=512) ns | mxu ns | resid ns | mxu Tops/s | derate × | Δ time % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 32 | 1149.6 | 1664.5 | 171.6 | 978.0 | 97.8 | **3.49** | **+25.60** |
| bf16 | 64 | 1043.0 | 1344.0 | 100.3 | 942.7 | 167.2 | 2.04 | +13.96 |
| bf16 | 128 | 915.3 | 1062.9 | 49.2 | 866.0 | 340.8 | 1.00 | 0.00 |
| bf16 | 256 | 896.9 | 1033.8 | 45.6 | 851.3 | 367.7 | 0.93 | −2.00 |
| fp8_e4m3 | 32 | 1011.0 | 1315.7 | 101.6 | 909.5 | 165.2 | **2.96** | **+19.65** |
| fp8_e4m3 | 64 | 945.0 | 1122.5 | 59.1 | 885.9 | 283.7 | 1.72 | +11.84 |
| fp8_e4m3 | 128 | 845.0 | 947.9 | 34.3 | 810.6 | 488.9 | 1.00 | 0.00 |
| fp8_e4m3 | 256 | 857.8 | 927.1 | 23.1 | 834.7 | 726.5 | 0.67 | +1.52 |

### The headline that needs no subtraction

The derate column depends on a slope, and the N=128 denominator (49.2 ns, from a 147.6 ns difference
between two arms with 8–16% spreads) is the worst-determined number in the table. So here is the same
conclusion drawn straight from two measured times, at K=512 where the MXU dominates and the spreads
are tightest — identical FLOPs, identical output elements, one number divided by another:

| dtype | N=32 ns | N=128 ns | Δ time % |
| --- | ---: | ---: | ---: |
| bf16 | 1664.4542029098375 | 1062.9346 | **+56.59** |
| fp8_e4m3 | 1315.7118701201398 | 947.9 | **+38.80** |

At K=512, N=32 is **56.6% slower than N=128 in bf16 and 38.8% slower in fp8**, for exactly the same
useful work. That is the finding; the 3.49× is the same finding with the store and loop overhead
stripped out, and it is the number a designer should use when the MXU is the only term that moves.

**N=128 is already the plateau.** N=256 is flat against it — −2.74% (bf16) and −2.19% (fp8) at
K=512, both inside their spreads. There is nothing to gain above 128 lanes and everything to lose
below.

**Per matmul, the narrow one is nearly as expensive as the wide one.** bf16: N=32 is
171.6/16 = 10.7 ns per matmul against N=128's 49.2/4 = 12.3 ns — **87% of the cost for a quarter of
the work**. fp8: 6.3 ns against 8.6 ns, **74%**. This is the lane-underfill picture almost exactly:
a matmul's cost is set far more by issuing it than by how many lanes it fills.

## How large is the derate, honestly

`combined.md`'s fork offers ≤1.3× or ≥4×. The measurement lands at **2.96–3.49×**, in the gap, and
that needs stating plainly rather than rounded to whichever side is convenient:

- It is **decisively not ≤1.3×**. The premise C6 rests on — that narrow N is close to free — is
  refuted, and refuted twice independently (slope-derived derate, and the subtraction-free K=512
  comparison).
- It is **short of a clean 4×**. Call it *"≈3.5× in bf16, ≈3× in fp8, bounded roughly 2–4×"* and
  never quote 3.49 to three significant figures as though the denominator supported it.
- It cross-checks well against `combined.md`'s independently `measured` **M-axis derate of 3.57× at
  M = 32**. Two different axes, two different probes, the same ~3.5×. That agreement is the strongest
  reason to trust the magnitude even though this probe's own denominator is soft.

## Branch selected

**`combined.md` P4 fork (line 690): the `≥4×` branch.**

> *"**≤1.3×:** C6 is close to free and becomes a strong candidate."*
> *"**≥4× (linear in wasted lanes):** C6 dies, and S1-c's relayout must be paid or absorbed by C5
> instead."*

The measurement selects the second branch — not by reaching 4× exactly, but by refuting the first
branch's premise so thoroughly that C6's case does not survive. **C6 dies.**

The rest of that branch's sentence needs an amendment, and wave 4 should have it: *"S1-c's relayout
must be paid or absorbed by C5 instead"* anticipated a cost that **P3 measured and found not to
exist** — S1-c's relayout is ≤0.12% of an inner iteration at `mixed_prod`, below P3's own negative
control. So it is simply paid, for nothing, and C5 (which is also dead) is not needed to absorb it.
P3 and P4 together close C5, C6 and C8 in one move.

## Byproduct: the fp8-vs-bf16 Pallas MXU ratio, which `combined.md` records as unmeasured

`combined.md` has a Pallas bf16 rate (569.9 Tops/s) and **no Pallas measurement for fp8 at all**.
From the two best-determined rungs here — N=32 and N=64, where the MXU slope is largest relative to
its own error:

| N | bf16 mxu ns | fp8 mxu ns | bf16 / fp8 |
| ---: | ---: | ---: | ---: |
| 32 | 171.6 | 101.6 | **1.689** |
| 64 | 100.3 | 59.1 | **1.697** |

**fp8 e4m3 delivers ≈1.69× bf16's MXU throughput under Pallas**, and the two independent rungs agree
to within 0.5%. This is a byproduct, not the probe's question — P1 answers the *fp8 or bf16* question
properly and directly, off the Mosaic MLIR — but it is a real measured ratio where there was none,
and it corroborates P1's conclusion that the mixed band's fp8 floor (48.4 µs) is the right one.

Two caveats on it: the absolute rates here (340.8 Tops/s bf16 at N=128) sit well below
`combined.md`'s 569.9 Tops/s, because this ladder is built to expose overhead rather than to hit
peak — R small matmuls per grid step, M fixed at 128. **The ratio travels; the absolute rate does
not.** And it is measured at M=128 only.

## Verdict

**PASS** — the question is answered, with two independent derivations agreeing on magnitude and a
third external number (the M-axis 3.57×) agreeing too. No arm failed; no arm was skipped.

**decided:** C6 drops off the candidate list. Any future design that would split the contraction
into N < 128 pieces carries a ~3× penalty per useful FLOP and must show a prize larger than that
before it is worth building. N = 128 is the plateau; N = 256 buys nothing.
