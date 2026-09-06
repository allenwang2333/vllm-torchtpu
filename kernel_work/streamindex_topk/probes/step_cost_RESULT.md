# Probe result: the per-inner-iteration fixed cost, and what it does to the gate

**Job `j-e8eaacd3-6be4-4cdf-95a2`**, `JobStatus: Succeeded`, 2/2 shards rc=0.
Image `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.
Script `probes/step_cost.py` at commit `543c5159`. Single chip per shard, one TensorCore.

`RUN_CMD`, as read back from `cdk job recipe`:

```
export PYTHONPATH=/workspace:$PYTHONPATH && bash /root/fanout/tpu-fanout.sh "python3 /workspace/kernel_work/streamindex_topk/probes/step_cost.py --max-num-seqs 64 --json /tmp/step_cost_ms64.json" "python3 /workspace/kernel_work/streamindex_topk/probes/step_cost.py --max-num-seqs 32 --json /tmp/step_cost_ms32.json"
```

Sign convention: every percentage is a change in **time**, `(new − old)/old`. Negative is faster.

---

## The question

Step 1's `report.md` put `decode_prod` at **55%** of its modelled floor, inside the 50–80% band
where this skill says a kernel is worth writing. `references/03-pipelining.md` prescribes a
different floor — `Σ_steps [512 ns + bytes_step / 3.48e12]` — and says a floor quoted without a
step count understates a kernel whose steps are small. The 512 ns was fitted on `pallas_call`
**grid** steps [verified: j-5855c2a5-aef4-4a61-b161]; this kernel's iteration is a `lax.fori_loop`
inside one kernel body (`streamindex_topk.py:418-420`) that hand-rolls its own DMAs.

Four readings of "step", from the same measured 76.42 µs:

| unit the 512 ns attaches to | steps | stepped floor | achieved |
| --- | ---: | ---: | ---: |
| none (`report.md`, bytes ÷ 3364.9 GB/s) | — | 41.7 µs | 55% |
| `pallas_call` grid step | 16 | 46.8 µs | 61% |
| inner `(bq, bkv)` iteration | 64 | **71.3 µs** | **93%** |
| individual page DMA | 512 | 300.6 µs | refuted by the measurement itself |

At 93% the skill's gate says **do not write a kernel**. Two sheets (area 2's disagreement 3, area
3) flagged the extrapolation independently and neither would settle it, correctly: it is a
measurement.

## What the probe did

Swept `bkv_p` ∈ {1, 2, 4, 8} at `max_model_len = 8192`, where the sequence is exactly 8 pages so
`align_to(8, bkv_p) = 8` at every rung. Held constant **by construction**: page-DMA count, KV
bytes, grid length, scores width (`sublanes = 64`, confirmed on all 8 arms), einsum FLOPs. Only
the inner-iteration count moved. A per-DMA fixed cost therefore contributes a constant and cannot
produce a slope; only a per-inner-iteration cost can.

Two shards at different batch sizes, so the fitted quantity has to reproduce rather than just exist.

**Shard 0 — `max_num_seqs=64`** (the production decode batch):

| bkv_p | num_bkv | inner | pageDMA | subl | bkv MiB | full ms | spread % | Δ time % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 128 | 512 | 64 | 2.0 | 0.1982242742087692 | 0.88 | +19.24 |
| **2** | 4 | 64 | 512 | 64 | 4.0 | **0.16623659757897258** | 0.28 | 0.00 |
| 4 | 2 | 32 | 512 | 64 | 8.0 | 0.1519077573902905 | 0.77 | **−8.62** |
| 8 | 1 | 16 | 512 | 64 | 16.0 | 0.1502107479609549 | 0.66 | **−9.64** |

`Δ time %` is against `bkv_p=2`, the shipped value (`mla_attention_op.py:119`). Negative is faster.
`full ms` is the whole `streamindex_topk` call including selection. `subl` is `num_sublanes_total`.
Fit: `full_time_ns = 445.9 × inner_total + 139893`, R² = 0.9854.

**Shard 1 — `max_num_seqs=32`** (cross-check; halves `inner_total` at every rung):

| bkv_p | inner | full ms | spread % | Δ time % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.1280865806620568 | 1.47 | +14.37 |
| **2** | 32 | 0.1119938213378191 | 1.08 | 0.00 |
| 4 | 16 | 0.1050586753990501 | 0.44 | −6.19 |
| 8 | 8 | 0.10530232102610171 | 1.62 | −5.97 |

Fit: `full_time_ns = 430.1 × inner_total + 99706`, R² = 0.9718.

All eight arms ran; none errored. `bkv_p=8`'s 16 MiB KV scratch was accepted by Mosaic. Spread was
0.28–1.62% everywhere, so the arms are separable.

## The fixed cost is real, and it is not constant

Taking successive differences rather than the whole-ladder fit — the fit averages over a slope
that visibly bends, and the bend is the finding:

| shard | Δ inner | Δ time | ns per inner step |
| --- | ---: | ---: | ---: |
| ms64 | 128 → 64 | 31.99 µs | **500** |
| ms64 | 64 → 32 | 14.33 µs | **448** |
| ms64 | 32 → 16 | 1.70 µs | **106** |
| ms32 | 64 → 32 | 16.09 µs | **503** |
| ms32 | 32 → 16 | 6.94 µs | **433** |
| ms32 | 16 → 8 | −0.24 µs | **≈0** |

The two shards agree rung for rung — 500/503, 448/433, 106/≈0 — at batch sizes that differ 2×. So
a per-inner-iteration fixed cost of **≈500 ns** exists and is close to the reference's 512 ns. On
that much, `03-pipelining.md` is right and the extrapolation from grid step to `fori_loop`
iteration holds.

**But it decays to zero at low step counts, and that is what decides the gate.**

## Why this does *not* close the gate

A per-step cost whose step count is a **design parameter** is not a floor term.

The 71.3 µs figure computes the floor of *the incumbent's loop structure* — it takes `num_bkv=4` as
given, when `num_bkv` is `cdiv(kv_len, page_size × bkv_p)` and `bkv_p` is a static argument of the
kernel. Folding an implementation's own choices into its floor makes the floor unfalsifiable: every
kernel is at 100% of the floor of its own structure, and the metric stops being able to say
anything.

The measurement refutes it directly rather than by argument. Changing one static constant made the
decode call **9.64% faster** (0.16624 → 0.15021 ms, shard 0). That is impossible if 71.3 µs were a
floor — you cannot go below a floor by editing a tuple.

**A roofline floor is taken over valid implementations, so it uses the smallest feasible step
count, not the incumbent's.** At `bkv_p=8` the inner loop collapses to one iteration per grid step
and the inner-step count *equals* the grid-step count, which is the 46.8 µs row of the table above.
So the two surviving readings converge: the correct floor is the **grid-step** one, and the gate
stands open at roughly **55–61%**.

The decay is the same fact seen from the other side. Once the per-step term is removed, something
else becomes binding — the flat 32→16 rung is the kernel hitting a cost the step count no longer
controls. That is what a floor looks like when you reach it.

## Verdict

**The exercise continues.** Step 1's roofline gate is upheld; `report.md`'s 55% and the grid-step
reading's 61% are both inside the band where a kernel is worth writing, and the inner-iteration
reading that would have ended the exercise rests on a floor definition the measurement falsifies.

`report.md`'s roofline section should be read with the correction above: the achieved fraction is
55–61% depending on whether the grid-step fixed term is included, not 93%.

## What this also produced, and what is not yet established

`bkv_p` is on the alternative list of three separate sheets (area 2's B1, area 3's items 8 and 9),
and it now has a measured number attached: **8.62% faster at `bkv_p=4`, 9.64% at `bkv_p=8`**, from
a one-line change to a static tuple.

Four things stand between that and a candidate, and none of them is settled by this job:

1. **The shape is not production.** `max_model_len=8192` was chosen precisely to *remove* the
   confound that production has. At 9216, `pages_per_seq=9` and `align_to(9, bkv_p)` moves
   `num_sublanes_total` 80 → 96 → 128, widening the scores buffer 20% and 60%. That buffer is
   written by the kernel and read again by the SparseCore, so the two effects oppose each other and
   the sign of the net is not derivable from this job. **No number here may be quoted as a
   `decode_prod` measurement.**
2. **Correctness was never checked at `bkv_p ≠ 2`.** A blocking factor *should* be value-neutral,
   which is exactly the class of belief that is cheap to check and embarrassing to assume.
3. **Only the decode band moved.** `num_kv_pages_per_block` is a 3-tuple and this job varied entry
   [0]. The mixed and prefill bands are untested, and mixed is where the kernel is furthest from
   its floor.
4. **VMEM at `bkv_p=8` is 16 MiB of KV scratch at decode.** It was accepted here; at mixed, with
   area 5's reported 32 MiB of live f32 `s`, it may not be.

`probes/bkv_p_prod.py` measures 1–4 at the real regimes through the read-only instruments.

## Note for whoever promotes this to `references/`

`03-pipelining.md`'s per-step term transfers from a `pallas_call` grid step to a hand-rolled
`lax.fori_loop` iteration at ≈500 ns — measured twice here, at two batch sizes
[verified: j-e8eaacd3-6be4-4cdf-95a2]. That is worth promoting, and area 2's caution that the
extrapolation was "not licensed in either direction" is now resolved in favour of it transferring.

What should be promoted **with** it is the modelling caveat, because the fact alone is dangerous:
the term must be applied at the step count of the *design under consideration*, never at the
incumbent's, or it manufactures a floor that forbids improvements which are then trivially
measured. This probe came within one judgement call of ending a multi-day exercise on that error.
