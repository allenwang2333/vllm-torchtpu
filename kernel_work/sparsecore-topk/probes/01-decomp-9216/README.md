# Probe 01 — `decomp-9216`: what fraction of the kernel is the score DMA, at the shipping shape?

**Question.** The caller's premise is that the HBM→VMEM score load stalls the kernel. Before
building a pipelined version, price the thing it would hide — at `n=10240`, the context length
`glm-5.2-fp8-dp16-ep.sh:18` actually sets, not the harness's `n=133120` default.

**Why it needed hardware.** Nothing in `references/` prices a SparseCore DMA against the compute it
would overlap, and being wrong changes whether the requested change is built at all.

## Arms

`microbench/sctopk_2stage_bench.py --decomp`, one job, arms interleaved and reshuffled per round,
9 rounds, paired `REPS_LO=20`/`REPS_HI=60` slope, `us_median` read:

| arm | what it runs |
| --- | --- |
| `full` | the whole block as the caller invokes it |
| `stage1` | the first-stage `pl.kernel` alone |
| `gatherA`, `stage2`, `gatherB` | the `P>1` tail (**does not execute at `P=1`**) |
| `dma_only` | a kernel that does stage 1's score copies and nothing else |
| `null` | the noise floor |

Shapes `b ∈ {16, 64, 1024}` at `--ctx 9216`, `k=2048`.

## Config

Image `fa63d6df`; a pristine `git archive b11b6eb6 -- src` mapped over `/workspace/src`, md5 of
`sparsecore_topk.py` verified **in-job** as `44353b347c96e0eedbd742c710d4540a`. One tpu7x chip.
The image tag is *not* the commit under test — no image exists at `b11b6eb6`, and
`git diff fa63d6df b11b6eb6 -- sparsecore_topk.py` is 130 insertions / 37 deletions, so running
unmapped would have measured the flag-gated pre-ship kernel instead.

## Job

`j-fa60ff2a-c793-4f32-9576` → `j-fa60ff2a-c793-4f32-9576.txt`, `decomp_9216.json`.

## Answer

| `b` | `P` | waves | `full` | `stage1` | `dma_only` | harness prints | **honest hideable DMA** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 2 | 1 | 56.511 | 33.390 | 22.055 | "66.0% DMA" | **1.0 µs = 3.1%** |
| 64 | 1 | 2 | 51.658 | 51.645 | 22.775 | "44.1% DMA" | **2.1 µs = 4.0%** |
| 1024 | 1 | 32 | 450.260 | 451.226 | 54.598 | "12.1% DMA" | **33.4 µs = 7.4%** |

**The harness's own `CANDIDATE B` gate is wrong for this question.** It computes
`dma_only / stage1` and fires `-> BUILD B` above 25%. But `dma_only` carries the **21.3 µs launch
intercept** — program load plus mesh entry — inside its numerator, and the real kernel pays that
intercept too. Pipelining cannot remove it. Removing it (probe 04's two-point fit) turns 66% into
3.1%. Following the printed gate would have sent the build after a fabricated prize.

## Decided

The caller's mechanism is capped at **−7.18% at prefill, −2.02% at decode**, per report §6 (⛔ SUPERSEDED by `probes/08` (`j-300d4690`): **−4.12% / −1.97%**, measured by deleting the transfer rather than fitting its slope) — and
built anyway, since it is non-zero and explicitly asked for. It is carried as candidate 3 of 3,
below the refinement walk, and that ranking is stated to the caller rather than buried.
