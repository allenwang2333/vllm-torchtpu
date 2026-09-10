# Probe 03 — `passes-9216`: what does one VMEM-resident SparseCore pass cost?

**Question.** The kernel reads `resident_vmem` about five times per wave — `cvt`, `hist0`, ~2.06
`ref_body`, `emit`. If the DMA is only 7.4% (probe 01), those passes are the rest of the kernel, and
nothing prices them. What does each kind of pass cost per wave, and specifically how does
`addupdate_scatter` compare to a plain read?

**Why it needed hardware.** `references/08-sparsecore.md` prices no SparseCore pass at any shape.
Being wrong reorders the entire step-2 candidate list.

## Arms

`microbench/sctopk_2stage_bench.py --passes`, which sweeps a synthetic kernel over **0, 1, 2 and 4**
passes and takes the slope, so the launch intercept and the score DMA both cancel:

| arm | one pass is |
| --- | --- |
| `read_only` | read `LANES` words, accumulate |
| `cvt` | read, `monotone_key`, write back |
| `scatter_sparse` | `addupdate_scatter` with a mask that admits few lanes |
| `scatter_dense` | `addupdate_scatter` with a mask that admits all lanes |
| `emit1` | the `store_compressed` emit path |
| `null` | noise floor |

`--ctx 9216`, `b=1024` (32 waves — the most slope per job), `k=2048`. 9 interleaved rounds,
`us_median`.

## Config

Same job, image `fa63d6df` + mapped `b11b6eb6` `src/`, in-job md5 `44353b34…`. One tpu7x chip.

## Job

`j-fa60ff2a-c793-4f32-9576` → `j-fa60ff2a-c793-4f32-9576.txt`, `passes_9216.json`.

## Answer

**µs per pass over the full `L`, at 32 waves:**

| pass | µs/pass | × `cvt` |
| --- | --- | --- |
| `cvt` | 20.589 | 1.00 |
| `emit1` | 27.456 | 1.33 |
| `scatter_sparse` | 35.243 | 1.71 |
| **`scatter_dense`** | **116.350** | **5.65** |

Sweeps were linear in pass count — the slope is a slope, not two points.

**The headline is the 3.3× between sparse and dense scatter.** A masked-off lane is not free, but it
is 0.30× a live one, so `addupdate_scatter` cost is dominated by **mask density**, not by the walk
alone. That is what makes `hist0` (dense by construction — every valid word increments a bucket) the
single most expensive pass in the kernel at 116.4 µs, 25.8% of the whole thing.

The refinement pass is the interesting one, and this probe does **not** measure it directly: it is a
sparse scatter over a full walk. Report §5d fits its one free parameter against the measured total
and gets **114.3 µs — 0.98× a dense scatter, 3.24× `scatter_sparse`**, validated out of sample at
`b=64` to −0.14%. So a refinement pass costs nearly what a *dense* one costs despite touching almost
nothing: **the cost is the traversal of all `L` words, not the updates.**

## Decided

This is the price sheet every step-2 candidate is costed against. Two things follow immediately:

- `addupdate_scatter` is **77.2%** of the shipping prefill kernel, against the DMA's 7.4%. The
  ranking in report §9 is this number.
- Candidate 1 (compact L0's straddling bucket once, so refinement walks `cnt` instead of `L`)
  targets 51.4% of the kernel and is ranked above the caller's mechanism.
